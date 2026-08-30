"""Capability discovery. Runs first; every other module consumes its result.

The robot's firmware build is not knowable from this repo -- the vendored OpenCat tree is
the wrong board entirely, and even the correct OpenCatEsp32 source has compile-time flags
(PRINT_ACCELERATION, the IMU chip, whether these servos support feedback at all) that only
the running board can answer. So nothing downstream hardcodes a reply format: this module
asks, records what came back, and writes capabilities.json.
"""
from __future__ import annotations  # PEP 604 unions on Python < 3.10 (deploy targets
                                    # whatever Raspberry Pi OS ships; Bullseye is 3.9)


import json
import re
import time
from dataclasses import asdict, dataclass, field

from . import protocol as P


@dataclass
class Capabilities:
    port: str = ""
    firmware: str | None = None
    board_is_esp32: bool | None = None
    board_code: str | None = None      # "B" = BiBoard/ESP32, "N" = NyBoard/AVR
    board_variant: str | None = None   # the full token: "B10" = BiBoard V1.0, "B01"/"B02" = V0.x
    build_date: str | None = None

    imu_chip: str | None = None            # "ICM" (ICM42670) or "MCU" (MPU6050)
    imu_has_accel: bool | None = None      # PRINT_ACCELERATION compiled in?
    imu_stream_rate_hz: float | None = None
    realtime_mode_supported: bool | None = None
    imu_sample: str | None = None

    feedback_all_rate_hz: float | None = None
    feedback_one_rate_hz: float | None = None
    feedback_speedup: float | None = None
    frame_width: int | None = None

    joint_reply_lines: list = field(default_factory=list)
    joint_reply_parsed: bool | None = None

    voltage: float | None = None
    has_signal_generator: bool | None = None

    # Raw gyro rates: VectorInt16 gy exists in imu.h but print6Axis never emits it. Left
    # False unless a build surprises us, so that state.py refuses to fabricate ang_vel.
    has_raw_gyro: bool = False

    # Does the `f` stream survive a motion command? Measured, because the whole
    # streaming-vs-polling design of sweep/jointmap/latency turns on it.
    feedback_survives_command: bool | None = None
    feedback_polled_rate_hz: float | None = None

    notes: list = field(default_factory=list)

    def to_json(self):
        return json.dumps(asdict(self), indent=2)

    @property
    def ang_vel_available(self):
        return self.has_raw_gyro


def _rate(link, seconds, predicate):
    n, t0 = 0, link.clock()
    for _, text in link.drain(seconds):
        if predicate(text):
            n += 1
    dt = link.clock() - t0
    return (n / dt) if dt > 0 and n else 0.0


def run(link, recorder, seconds=4.0):
    """Probe the board. Read-mostly: wakes the servos, toggles streams, commands no motion."""
    caps = Capabilities(port=link.port)
    transcript = []

    def log(tag, lines):
        transcript.append(f"--- {tag} ---")
        transcript.extend(lines)

    # -- identity ----------------------------------------------------------
    ver_lines = link.query(P.T_QUERY, timeout=1.5)
    log("? (version)", ver_lines)
    caps.firmware = P.parse_version(" ".join(ver_lines))
    if caps.firmware:
        # The version token is BOARD + "_" + YYMMDD, e.g. "B10_251121" (BiBoard/ESP32) vs
        # "N_260806" (NyBoard/AVR). Match that shape rather than substring-searching the
        # whole banner -- "Bittle X ..." contains "bi" regardless of the board.
        caps.board_is_esp32 = None
        for tok in caps.firmware.split():
            m = re.fullmatch(r"([A-Za-z]+)\d*_(\d{6})", tok)
            if m:
                caps.board_code, caps.build_date = m.group(1).upper(), m.group(2)
                # The digits are the BiBoard revision (OpenCat.h: B01/B02/B10), and they
                # decide which PWM_pin[] table a build must use -- so an image for the
                # wrong one drives the wrong servos. board_code keeps its old letters-only
                # meaning; the full token gets its own field so existing capabilities.json
                # captures do not change meaning under a reader.
                caps.board_variant = m.group(0).split("_")[0].upper()
                caps.board_is_esp32 = caps.board_code.startswith("B")
                break
        if caps.board_is_esp32 is False:
            caps.notes.append(
                f"firmware {caps.firmware!r} does not look like a BiBoard build; this "
                "package speaks OpenCatEsp32 tokens and may not match."
            )
    else:
        caps.notes.append("no reply to `?` -- board identity unconfirmed")

    caps.voltage = link.voltage()
    print(f"  firmware : {caps.firmware or '(no reply)'}"
          f"{f'  (board {caps.board_variant})' if caps.board_variant else ''}")
    print(f"  voltage  : {caps.voltage if caps.voltage is not None else '(no reply)'}")

    # -- IMU ---------------------------------------------------------------
    # `gb` first: balancing actively moves the servos to correct attitude, which would
    # both perturb the static IMU reading and confuse the feedback timing below.
    link.gyro_balance_off()

    sample = link.imu_once()
    if sample is None:
        caps.notes.append("`gp` returned nothing parseable -- IMU unavailable or a "
                          "different print format")
        print("  imu      : NO REPLY to `gp`")
    else:
        caps.imu_chip = sample.chip
        caps.imu_has_accel = sample.accel is not None
        caps.imu_sample = sample.raw
        log("gp (imu once)", [sample.raw])
        print(f"  imu      : {sample.chip}, accel={'yes' if caps.imu_has_accel else 'NO'} "
              f"-> {sample}")

        # REALTIME MODE, and the IMU rate is measured with it ON deliberately.
        #
        # The patched firmware boots in PREPROGRAMMED mode, so gP is capped at 5 Hz until `XR`
        # asks for otherwise -- which means the stream rate on its own no longer tells a
        # patched robot from an unpatched one. Switching first makes this an unambiguous
        # gate again: acknowledged + ~250 Hz is a patched robot in the mode the control loop
        # needs, and anything else is a robot that loop cannot run on.
        #
        # On stock firmware `XR` reads as "activate the hardware module R" and deactivates
        # the others. That is recoverable (`X` with no code re-opens them) and probe is a
        # bring-up tool run on a suspended robot, but it is the reason this is here and not
        # in something that runs during normal operation.
        caps.realtime_mode_supported = link.realtime(True) is True
        print(f"  realtime : {'supported' if caps.realtime_mode_supported else 'NOT SUPPORTED'}"
              f"  (sent {P.REALTIME_ON!r})")

        imu_lines = []
        link.send(P.G_PRINT_STREAM)
        t0 = link.clock()
        for _, text in link.drain(seconds):
            imu_lines.append(text)
        dt = link.clock() - t0
        link.send(P.G_PRINT_ONCE)
        link.sleep(0.15)
        link._buf = ""
        link._lines.clear()
        n_imu = sum(1 for t in imu_lines if P.looks_like_imu(t))
        caps.imu_stream_rate_hz = round(n_imu / dt, 2) if dt > 0 else 0.0
        log("gP (imu stream)", imu_lines[:40])
        print(f"  imu rate : {caps.imu_stream_rate_hz} Hz  ({n_imu} samples in {dt:.1f}s)"
              f"  [realtime {'on' if caps.realtime_mode_supported else 'unavailable'}]")
        if caps.realtime_mode_supported:
            link.realtime(False)          # leave the robot as it was found

    # -- joint angles from memory -----------------------------------------
    j_lines = link.query(P.T_JOINTS, timeout=1.5, min_lines=2)
    log("j (joint table)", j_lines)
    caps.joint_reply_lines = j_lines
    try:
        P.parse_joint_table("\n".join(j_lines))
        caps.joint_reply_parsed = True
    except P.ProtocolError as e:
        caps.joint_reply_parsed = False
        caps.notes.append(f"`j` reply did not parse: {e}")
    print(f"  j reply  : {'parsed' if caps.joint_reply_parsed else 'DID NOT PARSE'} "
          f"({len(j_lines)} lines)")

    # -- servo feedback: the ~7 Hz vs ~50 Hz question ---------------------
    # A full frame pulses every servo in turn, so it costs roughly N x the per-servo
    # query. `f8` should therefore be several times faster -- and fast enough per-joint
    # sampling is what makes the chirp response in `sweep` actually resolvable.
    link.wake()

    link.start_feedback(joint=None, wake=False)
    caps.frame_width = link.frame_width
    caps.feedback_all_rate_hz = round(
        _rate(link, seconds, lambda t: bool(_is_frame(t, link.frame_width))), 2)
    link.stop_feedback()
    print(f"  f (all)  : {caps.feedback_all_rate_hz} Hz, width {caps.frame_width}")

    try:
        link.start_feedback(joint=P.LEG_JOINTS[0], wake=False)
        caps.feedback_one_rate_hz = round(
            _rate(link, seconds, lambda t: bool(_numeric(t))), 2)
        link.stop_feedback()
        print(f"  f8 (one) : {caps.feedback_one_rate_hz} Hz")
    except Exception as e:
        caps.notes.append(f"single-joint feedback (`f8`) failed: {e}")
        print(f"  f8 (one) : FAILED -- {e}")

    if caps.feedback_all_rate_hz and caps.feedback_one_rate_hz:
        caps.feedback_speedup = round(
            caps.feedback_one_rate_hz / caps.feedback_all_rate_hz, 2)
        print(f"  speedup  : {caps.feedback_speedup}x for single-joint feedback")

    # -- does the stream survive a motion command? -------------------------
    # bittle_bringup2.py assumed `f` is a free-running stream. It is -- but only while
    # nothing else is transmitted. Any experiment that interleaves commands with readings
    # must poll instead, so this is measured rather than assumed.
    try:
        link.start_feedback(joint=P.LEG_JOINTS[0], wake=False)
        idle_n = sum(1 for _, t in link.drain(1.0) if len(_numeric(t)) == 2)
        link.send(P.move_cmd([(P.LEG_JOINTS[0], 10)]))
        after_n = sum(1 for _, t in link.drain(1.0) if len(_numeric(t)) == 2)
        link.stop_feedback()
        link.send(P.move_cmd([(P.LEG_JOINTS[0], 0)]))
        caps.feedback_survives_command = after_n > idle_n * 0.5
        print(f"  stream after a move: {after_n} frames/s (idle {idle_n}) -> "
              f"{'survives' if caps.feedback_survives_command else 'STOPS'}")
        if not caps.feedback_survives_command:
            caps.notes.append(
                "a motion command stops the `f` stream; sweep/jointmap/latency poll "
                "f<joint> per sample instead of streaming")
        # Achievable rate when alternating command and request -- what sweep really gets.
        t0, n = link.clock(), 0
        while link.clock() - t0 < 2.0:
            link.send(P.move_cmd([(P.LEG_JOINTS[0], 0)]))
            if link.poll_joint(P.LEG_JOINTS[0], timeout=0.2)[0] is not None:
                n += 1
        caps.feedback_polled_rate_hz = round(n / (link.clock() - t0), 2)
        print(f"  polled (command+request): {caps.feedback_polled_rate_hz} Hz")
    except Exception as e:
        caps.notes.append(f"stream-survival probe failed: {e}")

    # -- optional onboard signal generator ---------------------------------
    sig = link.query(P.T_SIGNAL_GEN, timeout=1.0)
    log("o (signal generator)", sig)
    caps.has_signal_generator = bool(sig)

    recorder.write_transcript("probe_transcript.txt", transcript)
    recorder.write_json("capabilities.json", asdict(caps))
    recorder.note("probe", firmware=caps.firmware,
                  realtime_mode=caps.realtime_mode_supported,
                  imu_rate_hz=caps.imu_stream_rate_hz,
                  feedback_all_hz=caps.feedback_all_rate_hz,
                  feedback_one_hz=caps.feedback_one_rate_hz)

    _summarize(caps)
    return caps


def _numeric(text):
    from .link import _values
    return _values(text)


def _is_frame(text, width):
    v = _numeric(text)
    return v and len(v) == width


def _summarize(caps):
    print("\n--- what this means ---")
    if caps.imu_has_accel:
        print("* accel IS available -> projected_gravity has two independent estimates")
        print("  (roll/pitch and accel). `imu --phase tilt` cross-checks them.")
    elif caps.imu_has_accel is False:
        print("* accel NOT compiled in -> projected_gravity comes from roll/pitch only.")
    if not caps.has_raw_gyro:
        print("* raw gyro rates are NOT emitted. base_ang_vel must be finite-differenced")
        print("  from the IMU stream; `imu --phase angvel-fallback` measures whether that")
        print("  is good enough at 50 Hz, or whether a custom firmware build is needed.")
    if caps.realtime_mode_supported is False:
        print("* Realtime mode is NOT supported: this is stock firmware, or a build without")
        print("  the patched firmware. The control loop refuses to run on it --")
        print("  `i` would be interpolated and the IMU stream capped at 5 Hz.")
    if caps.imu_stream_rate_hz is not None and caps.imu_stream_rate_hz < 50:
        print(f"* IMU streams at {caps.imu_stream_rate_hz} Hz, BELOW the 50 Hz control rate")
        print("  -> observations will be held between samples. state.py flags them stale.")
    if caps.feedback_speedup and caps.feedback_speedup >= 2:
        print(f"* single-joint feedback is {caps.feedback_speedup}x faster -> `sweep` should")
        print("  use it; per-joint system ID is viable.")
    elif caps.feedback_one_rate_hz:
        print("* single-joint feedback is NOT meaningfully faster -- sweep stays limited by")
        print("  the feedback rate. Consider driving system ID open-loop instead.")
    if caps.feedback_survives_command is False:
        print(f"* the `f` stream STOPS on a motion command. Interleaved measurement runs")
        print(f"  request/response at {caps.feedback_polled_rate_hz} Hz -- that, not the idle")
        print("  stream rate, is the Nyquist limit for `sweep`.")
    for n in caps.notes:
        print(f"! {n}")
