"""Serial transport for the BiBoard. Framing and stream handling only -- parsing is in
protocol.py.

Ported from bittle_bringup2.py's `Bittle` class, which was debugged against this exact
board. The behaviours preserved below are not stylistic; each one was a bug once.
"""

import glob
import time
from collections import Counter

import serial

from . import protocol as P

BAUD = 115200

# MEASURED on this board: a feedback request sent immediately after a motion command is
# DROPPED -- 0/4 replies at 0 ms gap, 4/4 at 20 ms and above. Consistent with the
# firmware's per-servo timing (readFeedback() holds a 15 ms attach delay, and the move
# handler re-attaches servos before parsing continues). Every command->request transition
# must leave this gap, which also sets the ceiling on polled sampling.
COMMAND_GAP = 0.02
_PORT_GLOBS = ("/dev/cu.usbmodem*", "/dev/cu.wchusbserial*", "/dev/cu.usbserial*",
               "/dev/ttyUSB*", "/dev/ttyACM*")
_PORT_EXCLUDE = ("Bluetooth", "debug-console", "wlan-debug")


class LinkError(RuntimeError):
    pass


class StreamTimeout(LinkError):
    pass


def find_port(explicit=None):
    if explicit:
        return explicit
    cands = [p for g in _PORT_GLOBS for p in glob.glob(g)
             if not any(s in p for s in _PORT_EXCLUDE)]
    if not cands:
        raise LinkError("No serial port found. Is the robot plugged in and powered on?")
    return sorted(cands)[0]


class _DryPort:
    """Stand-in for a serial port that records writes and never reads. Lets the whole CLI
    be exercised (and its token sequence asserted) with no robot attached."""

    def __init__(self):
        self.written = []

    def write(self, data):
        self.written.append(data.decode(errors="replace").strip())

    def flush(self):
        pass

    def read(self, _n):
        return b""

    def close(self):
        pass


class SerialLink:
    """Line-framed link to the board.

    Use as a context manager: __exit__ always disables gyro balancing and rests the
    servos, including on exception. Servos left fighting a target overheat.
    """

    def __init__(self, port=None, baud=BAUD, dry_run=False, clock=None,
                 confirm_calibration=None, verbose=True, voice_language="a"):
        self.dry_run = dry_run
        # Dry-run runs on a VIRTUAL clock. Every duration in this package is expressed in
        # seconds against link.clock(), so a clock that only advances when we ask it to
        # makes minute-long experiments terminate instantly with their real control flow
        # (and therefore their real token sequence) intact.
        self._virtual_t = 0.0
        self.clock = clock or (self._tick if dry_run else time.perf_counter)
        self.verbose = verbose
        self._confirm_calibration = confirm_calibration
        self.port = "(dry-run)" if dry_run else find_port(port)
        self.ser = _DryPort() if dry_run else serial.Serial(self.port, baud, timeout=0)
        self.sent = []
        self._buf = ""
        self._lines = []          # (text, arrival_ts)
        self._streaming = False
        self._feedback_cmd = None
        self.feedback_joint = None
        self.feedback_value_column = 0
        # Polled-read idiom; see poll_joint. MEASURED A/B on this board,
        # so do not "fix" this back to the one-shot form without re-measuring:
        #
        #                            interleaved (move+read)   consecutive (no motion)
        #   select-only  (f<n>)          4.4-6.6 Hz                28.4 Hz, 0% miss
        #   select-print (f<n> + fp)     2.6-4.4 Hz                16.4 Hz, 0% miss
        #
        # `fp` looks like the "proper" getter and is 1.7x SLOWER: with a stream already
        # running, `f<n>` re-selects and the next stream frame is already in flight,
        # whereas `fp` halts the stream and charges a full round trip per sample.
        # Streaming wins wherever it is usable. An inter-token gap did not help either.
        self.poll_mode = 'select-only'
        self._selected = None      # firmware's measureServoPin, as we believe it
        self.frame_width = None
        self.malformed = 0
        # Firmware realtime mode as far as THIS link knows: True/False once set, None while
        # unknown or when the firmware does not support the switch. One process owns the
        # link, so tracking it here is sound and saves a round trip per stream.
        self._realtime = None
        # 'a' English / 'b' Chinese. Only needed to put voice reactions BACK, and only
        # because the re-arm needs a language token and the board offers no way to read
        # which one it is set to. Wrong value = the robot's default language is changed.
        self.voice_language = voice_language
        # Whether THIS link turned voice reactions off, so close() can put them back.
        # Unlike gyro balancing, `XAd` never re-arms itself -- it sticks in the voice
        # module's own memory -- so forgetting the restore leaves the robot deaf to its
        # wake words after the process has exited.
        self._voice_off = False
        # Lines the IMU stream had to discard because they were not IMU frames. The firmware
        # only prints during `gP` if something ELSE ran -- a voice-triggered skill or a
        # randomMind demo -- and both move the servos, which corrupts any measurement in
        # progress. Non-zero here means the recording had a passenger; keep the text.
        self.chatter = 0
        self.chatter_lines = []
        if not dry_run:
            time.sleep(0.3)
            self.ser.read(65536)   # discard anything stale from a previous run
        self.version = None

    def _tick(self):
        """Virtual clock: advances 10 ms per read so timed loops always terminate."""
        self._virtual_t += 0.01
        return self._virtual_t

    def sleep(self, seconds):
        """Wall-clock sleep, or a virtual-clock advance under --dry-run."""
        if self.dry_run:
            self._virtual_t += seconds
        else:
            time.sleep(seconds)

    def prompt(self, message, default=""):
        """Operator prompt. Returns `default` under --dry-run or with no stdin."""
        if self.dry_run:
            print(f"{message}{default}   [dry-run]")
            return default
        try:
            return input(message)
        except EOFError:
            return default

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
        return False

    # -- raw io ------------------------------------------------------------
    def send(self, text):
        """Write one token line. Returns the send timestamp."""
        text = text.strip()
        if text[:1] == P.T_CALIBRATE:
            # `c` enters calibration and can overwrite the servo offsets stored on the
            # board. Guarded here rather than in the CLI so that NO code path -- including
            # a future experiment -- can transmit it unconfirmed.
            if self._confirm_calibration is None or not self._confirm_calibration(text):
                raise LinkError(
                    f"refusing to send {text!r}: the `c` token overwrites servo calibration "
                    "offsets. Pass confirm_calibration= to SerialLink if this is intended."
                )
        # The firmware resets measureServoPin = -1 on any token that is not f/F
        # (reaction.h), so our cached selection is invalidated by exactly the same rule.
        # Tracking it here rather than in poll_joint means no caller can desync it.
        if text[:1] not in (P.T_SERVO_FEEDBACK, "F"):
            self._selected = None
        ts = self.clock()
        self.sent.append(text)
        self.ser.write((text + "\n").encode())
        self.ser.flush()
        return ts

    def _is_own_echo(self, text):
        """True if this line is the firmware acknowledging a token WE sent.

        The board echoes the token it accepted, and truncates: sending `gP` comes back as
        `g`. Those must not be reported as unsolicited activity or the contamination check
        cries wolf on every stream -- measured on the first tilt run, which flagged a bare
        `g` that was its own `gP`.
        """
        t = text.strip()
        if not t or len(t) > 4:
            return False
        return any(cmd.startswith(t) for cmd in self.sent[-8:])

    def _pump(self):
        """Drain the port into complete LINES.

        Framing is on line boundaries, never on a running value count. Counting values
        meant that a single dropped number shifted column alignment permanently, which
        swapped the first and last columns in ~4% of frames.
        """
        chunk = self.ser.read(8192).decode(errors="replace")
        if not chunk:
            return None
        ts = self.clock()
        self._buf += chunk
        while "\n" in self._buf:
            line, self._buf = self._buf.split("\n", 1)
            line = line.strip()
            if line:
                self._lines.append((line, ts))
        return ts

    def drain(self, seconds, poll=0.002):
        """Yield (ts, line) for `seconds`.

        Note the swap: _lines holds (text, ts) because _pump appends in arrival order,
        but every consumer wants the timestamp first. Yielding the raw tuple silently
        handed callers a float where they expected a string.
        """
        t_end = self.clock() + seconds
        while self.clock() < t_end:
            self._pump()
            while self._lines:
                text, ts = self._lines.pop(0)
                yield ts, text
            self.sleep(poll)

    def collect(self, seconds):
        return list(self.drain(seconds))

    def flush_input(self):
        """Drop every buffered line and byte. Call before opening a timed baseline window.

        This is not tidiness, it is correctness. `_lines` carries ARRIVAL timestamps, so a
        backlog left over from a previous trial is not merely stale data -- its timestamps
        make a 0.6 s baseline window appear to fill instantly, out of samples recorded
        before the step was ever commanded. The mean and sigma then describe the PREVIOUS
        trial's attitude, and the comparison against them is meaningless: the first fresh
        sample either trips the threshold at once (reporting ~0 or negative latency) or
        moves toward the stale mean and never trips it at all (reported as "no onset").

        Both signatures, plus a clean alternating pattern, showed up in the post-reflash
        pre-reflash latency run and were initially misread as the robot swinging. The
        robot was still: `imu --phase static` measured 0.12 deg. It was this buffer.
        """
        self._buf = ""
        self._lines.clear()
        reset = getattr(self.ser, "reset_input_buffer", None)
        if reset is not None:
            try:
                reset()
            except Exception:      # a closed or fake port must not break an experiment
                pass

    def query(self, text, timeout=1.5, min_lines=1):
        """Send and gather reply lines until timeout or min_lines arrive."""
        self.send(text)
        out, t0 = [], self.clock()
        while self.clock() - t0 < timeout:
            self._pump()
            while self._lines:
                out.append(self._lines.pop(0)[0])
            if len(out) >= min_lines and self.clock() - t0 > 0.15:
                break
            self.sleep(0.01)
        # A trailing fragment with no newline is still a real reply (`?` often lacks one).
        if self._buf.strip():
            out.append(self._buf.strip())
            self._buf = ""
        return out

    # -- board state -------------------------------------------------------
    def query_version(self, timeout=1.5):
        """`?` -> board + build date.

        Native-USB boards do NOT reset when the port opens, so there is no boot banner to
        capture -- an empty boot log is normal and means nothing is wrong. `?` works any
        time instead.
        """
        self.version = P.parse_version(" ".join(self.query(P.T_QUERY, timeout)))
        return self.version

    def gyro_balance_off(self):
        """`gb`. Deterministic, unlike a bare `g` toggle whose parity depends on prior state.

        Mandatory before any actuation measurement: with balancing on, the firmware adds
        its own roll/pitch corrections on top of the angles we command, so a sweep or
        latency trial would be measuring the firmware's controller, not the servo.
        """
        self.send(P.G_BALANCE_OFF)
        self.sleep(0.1)

    def voice_reactions(self, on):
        """`XAc` / `XAd`. Off for the duration of an actuation run, and ALWAYS restored.

        The voice module recognises speech on its own core, and on a match the firmware
        queues a 2.5 s skill that drives the servos against whatever we are commanding --
        the same fight gyro balancing causes, from a second source that a word spoken near
        the robot is enough to start.

        Use this rather than sending the token: `XAd` is not self-clearing the way `gb` is,
        so close() has to undo it, and it can only undo what the link recorded.

        Turning it back ON is three tokens, not one -- `XAc` alone leaves the module silent,
        see protocol.voice_rearm_sequence() for the measurement and why a language change is
        part of it. The waits are the firmware's, not ours: sendVoiceModuleCmd() forwards
        each token to the module and then drains that UART for 200 ms, reading no host
        serial while it does, so a token sent straight after one of these is dropped. 0.8 s
        is the spacing this was confirmed working at on the robot.
        """
        if not on:
            self.send(P.VOICE_OFF)
            self.sleep(0.25)
            self._voice_off = True
            return
        for tok in P.voice_rearm_sequence(self.voice_language):
            self.send(tok)
            self.sleep(0.8)
        self._voice_off = False

    def wake(self, settle=1.0):
        """Drive all leg joints so the servo driver is active.

        Position feedback works by pulsing the PWM line and timing the reply. In rest mode
        the driver stops outputting PWM, so `f` returns NOTHING AT ALL -- no error, just
        silence. Native-USB boards do not reset on port open, so a rest state persists
        across runs and this is easy to mistake for a dead link.
        """
        self.send(P.move_cmd((j, 0) for j in P.LEG_JOINTS))
        self.sleep(settle)
        if not self.dry_run:
            self.ser.read(65536)
        self._buf = ""
        self._lines.clear()

    def voltage(self, timeout=1.0):
        """Battery voltage via `P`. Guards against stream leftovers.

        Right after a `gP` session the buffer still holds IMU lines whose first number
        (an accel component, e.g. -0.05) would parse as a "voltage". Clear first, skip
        anything that looks like an IMU line, and reject values outside a plausible
        battery range (2S LiPo: ~5-10 V).
        """
        if not self.dry_run:
            self.ser.read(65536)
        self._buf = ""
        self._lines.clear()
        for line in self.query(P.T_POWER, timeout):
            if P.looks_like_imu(line):
                continue
            v = P.parse_voltage(line)
            if v is not None and 2.0 < v < 13.0:
                return v
        return None

    # -- servo feedback stream ---------------------------------------------
    def start_feedback(self, joint=None, timeout=6.0, wake=True, min_frames=25):
        """Start the `f` position stream and auto-detect its frame width.

        `f` is a TOGGLE, so its parity can be inverted by a previous run that exited
        without stopping it -- hence the single retry before giving up.
        """
        if self._streaming:
            return self.frame_width
        if wake:
            self.wake()
        cmd = P.feedback_cmd(joint)
        self._feedback_cmd = cmd
        self.feedback_joint = joint
        for attempt in (1, 2):
            self.send(cmd)
            self._streaming = True
            t0 = self.clock()
            while len(self._lines) < min_frames and self.clock() - t0 < timeout:
                self._pump()
                self.sleep(0.005)
            if len(self._lines) >= 5 or self.dry_run:
                break
            if attempt == 1 and self.verbose:
                print("no data; retrying (f is a toggle -- parity may be off)")
        if self.dry_run:
            self.frame_width = len(P.LEG_JOINTS) if joint is None else 1
            return self.frame_width
        if len(self._lines) < 5:
            raise StreamTimeout(
                f"No feedback stream from {cmd!r}.\n"
                "  - servos powered down? a joint command must wake them first\n"
                "  - do these servos support position feedback at all?"
            )
        widths = Counter(len(_values(t)) for t, _ in self._lines if _values(t))
        if not widths:
            raise StreamTimeout(f"{cmd!r} produced lines but none were numeric")
        self.frame_width, n = widths.most_common(1)[0]
        # Single-joint feedback replies "<index>\t<angle>" -- e.g. `f8` -> "8\t0.7". The
        # value is in column 1; column 0 is the joint index, which is CONSTANT. Reading
        # column 0 as the angle yields a flat line that looks like a dead servo rather
        # than an error.
        self.feedback_value_column = (
            1 if (joint is not None and self.frame_width >= 2) else 0)
        if self.verbose:
            pct = 100.0 * n / max(sum(widths.values()), 1)
            print(f"feedback up: frame width = {self.frame_width} columns "
                  f"({pct:.0f}% of lines; saw {dict(widths)})")
            if joint is not None:
                print(f"  single-joint reply: value in column "
                      f"{self.feedback_value_column}")
        return self.frame_width

    def stop_feedback(self):
        if self._streaming:
            # Re-send the SAME token that started it. `f` is a toggle, so stopping an
            # `f8` stream with a bare `f` would flip parity *and* reset measureServoPin
            # to 16 (all joints) -- leaving the stream running over every servo.
            self.send(self._feedback_cmd or P.T_SERVO_FEEDBACK)
            self._streaming = False
            self.sleep(0.2)
            if not self.dry_run:
                self.ser.read(65536)
            self._buf = ""
            self._lines.clear()

    # -- polled feedback (request/response) --------------------------------
    # MEASURED ON THIS BOARD, then confirmed in firmware source: `f` free-runs only while
    # nothing else is transmitted (~57 frames/s idle -> 0 after one motion command).
    # OpenCatEsp32 src/reaction.h makes this explicit -- receiving ANY token other than
    # f/F while feedback is active resets measureServoPin=-1, clears readFeedbackQ, and
    # re-attaches all servos. Reading feedback also physically conflicts with driving:
    # readFeedback() (src/espServo.h) detaches the servo and turns its PWM pin into an
    # INPUT to time the reply pulse, at ~15 ms attach delay + 3 pulses per read. So any
    # measurement that interleaves commands with readings -- sweep, jointmap, latency --
    # must REQUEST each sample. (bittle_bringup2.py's "f starts a FREE-RUNNING STREAM"
    # holds only for a passive observer, which is why `stream` may keep using frames().)

    def poll_joint(self, joint, timeout=0.25, gap=COMMAND_GAP, attempts=2):
        """Request one position sample. Returns (angle, arrival_ts, request_ts).

        Two idioms, selected by `self.poll_mode` (see _poll_tokens):

          "select-print"  `f<joint>` then `fp`  -- the intended request/response form.
                          `f<joint>` only SELECTS the servo; `fp` prints one frame and
                          leaves streaming off, so it is idempotent and needs no retry.
          "select-only"   `f<joint>` alone     -- starts a stream and reads its first
                          frame. One token instead of two, but leaves the stream running
                          until the next motion command kills it.

        Which is actually faster is an empirical question about token round-trips versus
        stream startup, so both are kept and benchmarked rather than assumed.

        The reply is "<index>\\t<angle>"; the index is verified so a stale line left over
        from a previous joint cannot be mistaken for this one's position.
        """
        self.sleep(gap)
        self._lines.clear()
        self._buf = ""
        t_cmd = None
        for _ in range(max(attempts, 1)):
            for tok in self._poll_tokens(joint):
                ts_req = self.send(tok)
                if t_cmd is None:
                    t_cmd = ts_req
            for ts, text in self.drain(timeout / max(attempts, 1)):
                vals = _values(text)
                if len(vals) == 2 and int(vals[0]) == int(joint):
                    return vals[1], ts, t_cmd
        return None, None, t_cmd

    def _poll_tokens(self, joint):
        """Tokens for one polled read, re-selecting only when the selection is stale.

        The firmware resets `measureServoPin` to -1 on any non-`f` token (reaction.h), so
        a preceding motion command always invalidates the selection -- which is exactly
        the interleaved case. `_selected` tracks it so a run of consecutive reads on one
        joint costs a single `fp` each.
        """
        if self.poll_mode == "select-only":
            self._selected = joint
            return (P.feedback_cmd(joint),)
        if self._selected != joint:
            self._selected = joint
            return (P.feedback_cmd(joint), P.F_PRINT_ONCE)
        return (P.F_PRINT_ONCE,)

    def poll_all(self, timeout=0.5, width=None, gap=COMMAND_GAP):
        """Request one all-joint frame. Returns (values, ts) or (None, None)."""
        self.sleep(gap)
        self._lines.clear()
        self._buf = ""
        for tok in self._poll_tokens(P.FEEDBACK_ALL):
            self.send(tok)
        want = width or self.frame_width
        for ts, text in self.drain(timeout):
            vals = _values(text)
            if vals and (want is None or len(vals) == want):
                if want is None:
                    self.frame_width = len(vals)
                return vals, ts
        return None, None

    def settle_polled(self, joint, samples=5, timeout=0.25):
        """Median of several polled readings -- outlier tolerant, command-safe."""
        got = [v for v, _, _ in (self.poll_joint(joint, timeout) for _ in range(samples))
               if v is not None]
        return _median(got) if got else None

    def frames(self, seconds):
        """Yield (ts, values) for feedback lines matching the detected frame width.

        The width filter is CORRECTNESS, not tidiness: servoFeedback() prints nothing for
        a joint whose read failed, so a short frame has SHIFTED columns and its values
        would be attributed to the wrong joints. Only full-width frames are trustworthy.
        """
        if not self._streaming:
            self.start_feedback()
        for ts, text in self.drain(seconds):
            vals = _values(text)
            if vals and len(vals) == self.frame_width:
                yield ts, vals
            else:
                self.malformed += 1

    def settle(self, seconds):
        """Per-column median over a short window -- outlier tolerant."""
        rows = [v for _, v in self.frames(seconds)]
        if not rows:
            raise StreamTimeout("feedback stream stalled while settling")
        return [_median([r[c] for r in rows]) for c in range(self.frame_width)]

    # -- imu ---------------------------------------------------------------
    def imu_once(self, timeout=1.0):
        """`gp` -> one ImuSample, or None if nothing parseable arrived."""
        for line in self.query(P.G_PRINT_ONCE, timeout):
            if P.looks_like_imu(line):
                return P.parse_imu(line)
        return None

    def realtime(self, on):
        """Switch the firmware's realtime mode. True/False on success, None if unsupported.

        The patched firmware boots in PREPROGRAMMED mode, where gP is capped at 5 Hz -- so any
        experiment that reads the IMU stream has to ask for the other mode first or it will
        quietly measure a fifty-times-slower signal. Unsupported means stock firmware, which
        has no such token; the caller decides whether that is fatal.

        Waits for ITS OWN acknowledgement rather than accepting the first line that arrives,
        and RETRIES. Both halves are needed, and each was measured:

          * query() returns as soon as it has a line, so a reply still queued from the
            previous command is what it hands back. Same stale-backlog trap flush_input()
            documents.
          * flushing is not enough on its own, because the previous reply may still be IN
            FLIGHT when the flush runs -- it then lands afterwards and looks like the
            answer. Worse, a token sent while the firmware is mid-print is dropped outright,
            so the acknowledgement never comes. Measured: `b0` (mute) immediately before
            this call loses the switch every time.

        So: let the line go quiet, flush, send, wait for the ack, and try again if it does
        not arrive. None means the firmware has no such token.
        """
        cmd = P.realtime_cmd(on)
        for _ in range(3):
            self.sleep(0.15)          # let any in-flight reply land BEFORE the flush
            self.flush_input()
            self.send(cmd)
            text, t0 = [], self.clock()
            while self.clock() - t0 < 0.8:
                self._pump()
                while self._lines:
                    text.append(self._lines.pop(0)[0])
                ack = P.realtime_ack("\n".join(text) + "\n" + self._buf)
                if ack is not None:
                    self._realtime = ack
                    return ack
                self.sleep(0.01)
        self._realtime = None
        return None

    def imu_stream(self, seconds):
        """`gP` continuous IMU. Yields (ts, ImuSample); non-IMU chatter is skipped.

        Enters realtime mode for the duration if it is not already on, and restores the
        previous mode afterwards -- at the stock 200 ms interval this stream is 5 Hz, which
        is not a slower version of the measurement but a different one.
        """
        entered = False
        if self._realtime is not True:
            entered = self.realtime(True) is True
            if not entered:
                print("  ! realtime mode unavailable -- gP may be capped at 5 Hz. "
                      "Reflash from the firmware submodule for a usable IMU stream.")
        self.send(P.G_PRINT_STREAM)
        try:
            for ts, text in self.drain(seconds):
                if P.looks_like_imu(text):
                    try:
                        yield ts, P.parse_imu(text)
                    except P.ProtocolError:
                        self.malformed += 1
                elif text.strip() and not self._is_own_echo(text):
                    self.chatter += 1
                    if len(self.chatter_lines) < 20:
                        self.chatter_lines.append(text.strip())
        finally:
            self.send(P.G_PRINT_ONCE)   # `p` stops continuous printing
            if entered:
                self.realtime(False)    # leave the robot in the mode we found it in
            self.sleep(0.1)

    # -- teardown ----------------------------------------------------------
    def close(self, rest=True):
        try:
            if self._streaming:
                self.stop_feedback()
            if self._voice_off:
                # The one piece of firmware state here that outlives the process: `XAd`
                # sits in the voice module, so a run that does not undo it leaves a robot
                # that ignores its wake words in every mode. Restored first, because the
                # firmware spends 200 ms draining the voice UART after each of these and
                # the settle inside keeps that window from swallowing `Xr`.
                self.voice_reactions(True)
            if self._realtime is True:
                # Leave the robot able to run its own skills and gaits. Realtime mode is not
                # a state to strand it in: `i` stops being interpolated, so anything that
                # drives it in single poses moves in steps instead of smoothly -- and
                # read_voice() drops every recognised word, so the robot chimes at its wake
                # word and then ignores it.
                #
                # realtime(), not a bare send(). close() runs directly after other traffic,
                # and a token sent while the firmware is mid-print is dropped outright --
                # the same measurement realtime()'s docstring records for the entry path.
                # A dropped `Xr` strands the robot silently, and nothing clears the flag
                # short of an actual board reset: a battery cycle does not restart a
                # USB-powered board, so the mode survives what looks like a reboot.
                self.realtime(False)
            self.send(P.G_BALANCE_OFF)
            if rest:
                # Powers servos down -- thermally safe, but it means the NEXT `f` returns
                # nothing until something wakes them again.
                self.send(P.T_REST)
                self.sleep(0.4)
        except Exception:
            pass
        finally:
            self.ser.close()


def _values(text):
    """Numeric tokens of a feedback line, or [] if it isn't one."""
    if P.looks_like_imu(text):
        return []
    out = []
    for tok in text.replace(",", " ").split():
        try:
            out.append(float(tok))
        except ValueError:
            return []
    return out


def _median(xs):
    s = sorted(xs)
    n = len(s)
    return s[n // 2] if n % 2 else 0.5 * (s[n // 2 - 1] + s[n // 2])
