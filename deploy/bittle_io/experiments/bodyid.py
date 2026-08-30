"""Actuator + body frequency response measured through the IMU. ROBOT ON THE GROUND.

The polled servo-feedback path cannot see above ~2 Hz. The IMU can: it streams at
~250 Hz (probe: imu_stream_rate_hz), and -- checked in firmware source -- `printGyroQ`
is NOT cleared by motion tokens (OpenCatEsp32 reaction.h clears only the servo-feedback
flags), so the stream keeps running while we command sinusoids. That turns the body into
the measurement instrument:

  drive one joint with amp*sin(2*pi*f*t)  ->  the leg pushes the ground  ->  the body
  pitches/rolls at f  ->  lock-in at f on the 250 Hz IMU stream gives amplitude + phase.

The rolloff of that response vs frequency is actuator lag convolved with body dynamics.
For small amplitudes on a stiff-legged robot the body term varies slowly with f in the
1-8 Hz band, so the knee frequency is actuator-dominated -- and unlike the polled sweep,
the servo is DRIVEN CONTINUOUSLY here (no detach windows, working gain), so this is the
response the walking policy actually experiences.

Caveats, stated up front:
  * This measures joint-command -> body-motion, not joint-command -> joint-angle. Use it
    for the SHAPE of the response (knee frequency, phase trend), not absolute gains.
  * Phase estimates ride on host arrival timestamps; USB delivery adds ~1-2 ms of jitter,
    i.e. a few degrees of phase noise at 8 Hz. Amplitudes are robust.
  * The robot may shuffle or creep. Clear space, small default amplitude, be ready to
    interrupt (Ctrl-C rests the servos on the way out).
"""

import math

import numpy as np

NAME = "bodyid"
HELP = "IMU-based frequency response, 0.5-8 Hz; needs the robot ON THE GROUND"

DEFAULT_FREQS = [0.5, 1.0, 2.0, 3.0, 4.0, 6.0, 8.0]


def add_args(p):
    p.add_argument("--joints", type=int, nargs="*", default=None,
                   help="joints to excite (default: all leg joints)")
    p.add_argument("--freqs", type=float, nargs="*", default=DEFAULT_FREQS)
    p.add_argument("--amp", type=float, default=6.0, help="deg; keep small on the ground")
    p.add_argument("--dwell", type=float, default=4.0, help="seconds per frequency")
    p.add_argument("--skip-transient", type=float, default=1.0,
                   help="seconds discarded at the start of each dwell")
    p.add_argument("--cmd-hz", type=float, default=50.0)


def run(link, caps, args, recorder):
    from .. import protocol as P

    if caps.imu_chip is None and not link.dry_run:
        raise RuntimeError("probe found no IMU -- run `probe` first")
    joints = args.joints if args.joints else list(P.LEG_JOINTS)

    print("\nROBOT ON THE GROUND, clear space around it. It may shuffle.")
    link.prompt("Press Enter to start (Ctrl-C aborts and rests the servos) -> ")

    link.gyro_balance_off()
    rows, per_joint = [], {}
    with recorder.bracket_voltage(NAME, link):
        link.wake()
        for j in joints:
            print(f"\n--- joint {j}: {args.amp:g} deg at "
                  f"{', '.join(f'{f:g}' for f in args.freqs)} Hz ---")
            mags = []
            if link._realtime is not True and link.realtime(True) is not True:
                print("  ! realtime mode unavailable -- the IMU stream may be capped at "
                      "5 Hz, which is far below what a frequency response needs.")
            link.send(P.G_PRINT_STREAM)
            try:
                for f in args.freqs:
                    r = _dwell(link, P, j, f, args)
                    rows.append(r)
                    mags.append(r["mag"])
                    print(f"  {f:4.1f} Hz: pitch {r['pitch_amp']:.3f}  "
                          f"roll {r['roll_amp']:.3f}  mag {r['mag']:.3f} deg  "
                          f"phase {r['phase_deg']:+6.1f}  snr {r['snr']:.1f}  "
                          f"({r['samples']} samples)")
            finally:
                link.send(P.G_PRINT_ONCE)   # stop the stream
                link.send(P.move_cmd([(j, 0)]))
                link.sleep(0.3)
            per_joint[j] = knee_frequency(args.freqs, mags)
            if per_joint[j]:
                print(f"  -3 dB knee ~{per_joint[j]:.1f} Hz")
            elif mags and max(mags) > 0:
                print(f"  no -3 dB crossing below {max(args.freqs):g} Hz "
                      "(bandwidth exceeds the tested band)")

    recorder.write_csv("bodyid.csv", rows)
    knees = [k for k in per_joint.values() if k]
    res = {"joints": len(per_joint),
           "knee_hz_min": round(min(knees), 2) if knees else None}
    recorder.note(NAME, **res)
    if knees:
        print(f"\n--- verdict ---")
        print(f"slowest joint's knee ~{min(knees):.1f} Hz. Set the sim actuator's")
        print("bandwidth/lag so its command->body response rolls off there too, and")
        print("randomize around it. Cross-check tau from `step`: "
              "f_3db = 1/(2*pi*tau).")
    return res


def _dwell(link, P, joint, freq, args):
    """One excitation frequency: command sinusoid at cmd_hz, lock-in on the IMU stream."""
    period = 1.0 / args.cmd_hz
    t0 = link.clock()
    next_cmd = t0
    times, pitch, roll = [], [], []
    for ts, text in link.drain(args.dwell):
        now = link.clock()
        if now >= next_cmd:
            cmd = args.amp * math.sin(2 * math.pi * freq * (now - t0))
            link.send(P.move_cmd([(joint, cmd)]))
            next_cmd += period
        if P.looks_like_imu(text):
            try:
                s = P.parse_imu(text)
            except P.ProtocolError:
                continue
            if ts - t0 >= args.skip_transient:
                times.append(ts - t0)
                pitch.append(s.pitch)
                roll.append(s.roll)

    if len(times) < 20:
        return {"joint": joint, "freq_hz": freq, "pitch_amp": 0.0, "roll_amp": 0.0,
                "mag": 0.0, "phase_deg": 0.0, "snr": 0.0, "samples": len(times)}

    t = np.array(times)
    ap, php = lockin(t, np.array(pitch), freq)
    ar, phr = lockin(t, np.array(roll), freq)
    mag = math.hypot(ap, ar)
    # Phase of the dominant axis, relative to the commanded sin(2*pi*f*(t-t0)) -- the
    # lock-in already uses t-t0 as its time base, so this is directly command-relative.
    phase = php if ap >= ar else phr
    # Noise floor: lock-in at a nearby non-harmonic frequency.
    noise = math.hypot(lockin(t, np.array(pitch), freq * 1.31)[0],
                       lockin(t, np.array(roll), freq * 1.31)[0])
    return {"joint": joint, "freq_hz": freq,
            "pitch_amp": round(ap, 4), "roll_amp": round(ar, 4),
            "mag": round(mag, 4), "phase_deg": round(phase, 1),
            "snr": round(mag / noise, 1) if noise > 1e-9 else float("inf"),
            "samples": len(times)}


# -- analysis (pure; tested offline) ----------------------------------------

def lockin(t, y, freq):
    """Amplitude and phase of the `freq` component in irregularly sampled data.

    Hann-weighted least squares on sin/cos at f. The weighting matters: a rectangular
    window leaks ~10% of a strong neighbouring tone into the estimate (sinc sidelobes),
    which would inflate the noise-floor lock-in at 1.31f and understate the SNR. Solving
    the weighted normal equations (rather than projecting) keeps the amplitude unbiased
    under both the window and the irregular sampling.
    """
    span = float(t.max() - t.min())
    if span <= 0:
        return 0.0, 0.0
    win = 0.5 - 0.5 * np.cos(2 * np.pi * (t - t.min()) / span)   # Hann
    ymean = float(np.sum(win * y) / np.sum(win))
    y = y - ymean                                                # weighted demean (DC/bias)
    w = 2 * np.pi * freq * t
    s, c = np.sin(w), np.cos(w)
    a11, a12, a22 = np.sum(win * s * s), np.sum(win * s * c), np.sum(win * c * c)
    b1, b2 = np.sum(win * y * s), np.sum(win * y * c)
    det = a11 * a22 - a12 * a12
    if abs(det) < 1e-12:
        return 0.0, 0.0
    a = (a22 * b1 - a12 * b2) / det
    b = (a11 * b2 - a12 * b1) / det
    return math.hypot(a, b), math.degrees(math.atan2(b, a))


def knee_frequency(freqs, mags):
    """First -3 dB crossing relative to the lowest-frequency magnitude, interpolated.

    Returns None when the response never drops 3 dB in-band (or there is no signal).
    """
    if not mags or mags[0] <= 0:
        return None
    ref = mags[0]
    target = ref / math.sqrt(2.0)
    for i in range(1, len(mags)):
        if mags[i] < target:
            lo_f, hi_f = freqs[i - 1], freqs[i]
            lo_m, hi_m = mags[i - 1], mags[i]
            if lo_m == hi_m:
                return hi_f
            frac = (lo_m - target) / (lo_m - hi_m)
            return lo_f + frac * (hi_f - lo_f)
    return None
