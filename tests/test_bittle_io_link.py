"""Link and state-layer tests. Run entirely against the dry-run port -- no hardware."""

import numpy as np
import pytest

from bittle_io import protocol as P
from bittle_io.link import LinkError, SerialLink
from bittle_io.probe import Capabilities
from bittle_io.state import StateReader


def link():
    return SerialLink(dry_run=True, verbose=False)


# -- safety -----------------------------------------------------------------

def test_calibration_token_is_refused_by_default():
    """`c` overwrites the servo offsets stored on the board. The guard lives in send()
    so no code path -- present or future -- can transmit it unconfirmed."""
    with link() as ln:
        with pytest.raises(LinkError, match="calibration"):
            ln.send("c0 7 1 -4")
    assert not any(t.startswith("c0") for t in ln.sent)


def test_calibration_token_passes_with_confirmation():
    ln = SerialLink(dry_run=True, verbose=False, confirm_calibration=lambda t: True)
    with ln:
        ln.send("c0 7")
    assert "c0 7" in ln.sent


def test_gyro_calibrate_is_not_caught_by_the_c_guard():
    """`gc` is an IMU calibration under the `g` token -- it starts with 'g', not 'c'."""
    with link() as ln:
        ln.send(P.G_CALIBRATE)
    assert P.G_CALIBRATE in ln.sent


def test_exit_always_disables_balancing_and_rests():
    with link() as ln:
        ln.send("j")
    assert ln.sent[-2:] == [P.G_BALANCE_OFF, P.T_REST]


def test_exit_restores_voice_reactions():
    """`XAd` is the one bit of firmware state a run leaves behind: set_voice() forwards it
    to the voice module over its own UART and the module keeps it, so a robot whose run
    ended without `XAc` stops answering its wake words -- in preprogrammed mode as much as
    realtime, with nothing in either to say why."""
    with link() as ln:
        ln.voice_reactions(False)
    assert P.VOICE_ON in ln.sent
    assert ln.sent.index(P.VOICE_OFF) < ln.sent.index(P.VOICE_ON)


def test_restoring_voice_sends_the_language_change_that_re_arms_the_module():
    """`XAc` alone leaves the module silent -- measured, 180 s with someone speaking at it.
    It re-arms on a language CHANGE, so the restore is three tokens for an English robot and
    the Chinese hop is load-bearing, not a typo."""
    with link() as ln:
        ln.voice_reactions(False)
    tail = ln.sent[ln.sent.index(P.VOICE_ON):]
    assert tail[:3] == [P.VOICE_ON, P.VOICE_LANG_ZH, P.VOICE_LANG_EN]


def test_a_chinese_robot_is_not_left_speaking_english():
    """The language tokens PERSIST to the board's config, so the restore must end on the
    robot's own language -- for `b` that is the hop itself, with no `XAa` after it."""
    ln = SerialLink(dry_run=True, verbose=False, voice_language="b")
    with ln:
        ln.voice_reactions(False)
    assert P.VOICE_LANG_EN not in ln.sent
    tail = ln.sent[ln.sent.index(P.VOICE_ON):]
    assert tail[:2] == [P.VOICE_ON, P.VOICE_LANG_ZH]


def test_exit_restores_voice_before_leaving_realtime():
    """Ordering, not taste: the firmware drains the voice UART for 200 ms after forwarding
    `XAc` and reads no host serial while it does, so `Xr` sent into that window is lost and
    the robot is stranded in realtime mode."""
    with _ModeLink() as ln:
        ln.realtime(True)
        ln.voice_reactions(False)
    assert ln.sent.index(P.VOICE_ON) < ln.sent.index(P.REALTIME_OFF)


def test_exit_leaves_voice_alone_when_the_run_never_touched_it():
    with link() as ln:
        ln.send("j")
    assert P.VOICE_ON not in ln.sent


def test_exit_rests_even_on_exception():
    ln = link()
    with pytest.raises(ValueError):
        with ln:
            raise ValueError("boom")
    assert ln.sent[-1] == P.T_REST


# -- feedback stream parity -------------------------------------------------

def test_single_joint_stream_is_stopped_with_the_same_token():
    """Stopping an `f8` stream with a bare `f` would flip toggle parity AND reset
    measureServoPin to 16, leaving the stream running over every servo."""
    with link() as ln:
        ln.start_feedback(joint=8)
        ln.stop_feedback()
    assert "f8" in ln.sent
    assert ln.sent[ln.sent.index("f8") + 1] == "f8"


def test_all_joint_stream_uses_bare_f():
    with link() as ln:
        ln.start_feedback()
        ln.stop_feedback()
    assert ln.sent.count("f") == 2


# -- dry-run virtual clock --------------------------------------------------

def test_virtual_clock_terminates_timed_loops():
    """A 600 s drain must not block; the virtual clock advances on read."""
    with link() as ln:
        assert list(ln.drain(600.0)) == []


def test_sleep_advances_the_virtual_clock_without_blocking():
    with link() as ln:
        t0 = ln.clock()
        ln.sleep(30.0)
        assert ln.clock() - t0 >= 30.0


# -- command construction round-trip ----------------------------------------

def test_wake_drives_every_leg_joint():
    with link() as ln:
        ln.wake(settle=0.0)
    wake_cmd = next(t for t in ln.sent if t.startswith("i"))
    for j in P.LEG_JOINTS:
        assert f"{j} 0" in wake_cmd


# -- state layer ------------------------------------------------------------

class FakeLink:
    """Serves a scripted list of IMU samples."""

    def __init__(self, samples, dt=0.02):
        self.samples = list(samples)
        self.dt = dt
        self.t = 0.0

    def clock(self):
        return self.t

    def imu_once(self, timeout=1.0):
        if not self.samples:
            return None
        self.t += self.dt
        return self.samples.pop(0)

    def query(self, *a, **k):
        return []


def _sample(yaw, pitch, roll, accel=(0.0, 0.0, -9.81)):
    return P.ImuSample("ICM", np.array(accel, dtype=float), yaw, pitch, roll, "")


def _caps(rate=50.0, accel=True):
    return Capabilities(imu_chip="ICM", imu_has_accel=accel, imu_stream_rate_hz=rate)


def test_first_read_reports_ang_vel_invalid():
    """There is no previous sample to difference against, so the rate is not knowable.
    It must be flagged, never returned as a confident zero."""
    r = StateReader(FakeLink([_sample(0, 0, 0)]), _caps())
    st = r.read(joints=False)
    assert not st.ang_vel_valid
    np.testing.assert_array_equal(st.ang_vel_rad_s, np.zeros(3))


def test_ang_vel_is_finite_differenced():
    fake = FakeLink([_sample(0, 0, 0), _sample(0, 2.0, 0)], dt=0.02)
    r = StateReader(fake, _caps())
    r.read(joints=False)
    st = r.read(joints=False)
    assert st.ang_vel_valid
    # 2 deg over 20 ms about pitch
    np.testing.assert_allclose(st.ang_vel_rad_s[1], np.radians(2.0) / 0.02, rtol=1e-6)
    np.testing.assert_allclose(st.ang_vel_rad_s[[0, 2]], [0, 0], atol=1e-12)


def test_yaw_wrap_does_not_produce_a_spurious_rate():
    """Yaw wraps at +-180 deg. Unwrapped, a 2 deg step across the seam reads as -358 deg."""
    fake = FakeLink([_sample(179.0, 0, 0), _sample(-179.0, 0, 0)], dt=0.02)
    r = StateReader(fake, _caps())
    r.read(joints=False)
    st = r.read(joints=False)
    np.testing.assert_allclose(st.ang_vel_rad_s[2], np.radians(2.0) / 0.02, rtol=1e-6)


def test_missing_imu_marks_state_stale():
    r = StateReader(FakeLink([]), _caps())
    st = r.read(joints=False)
    assert st.imu_stale and not st.ang_vel_valid
    assert st.projected_gravity is None


def test_gravity_is_down_when_level():
    r = StateReader(FakeLink([_sample(0, 0, 0)]), _caps())
    st = r.read(joints=False)
    np.testing.assert_allclose(st.projected_gravity, [0, 0, -1], atol=1e-12)


def test_accel_source_requires_the_accelerometer():
    with pytest.raises(ValueError, match="PRINT_ACCELERATION"):
        StateReader(FakeLink([]), _caps(accel=False), gravity_source="accel")


def test_raw_gyro_capability_refuses_to_silently_differentiate():
    """If a build ever does emit gy, falling back to differentiation would hide it."""
    caps = _caps()
    caps.has_raw_gyro = True
    r = StateReader(FakeLink([_sample(0, 0, 0)]), caps)
    with pytest.raises(NotImplementedError):
        r.read(joints=False)


def test_drain_yields_timestamp_first():
    """Regression: _lines stores (text, ts) but every consumer unpacks (ts, text).
    Yielding the raw tuple handed callers a float where a string was expected."""
    ln = link()
    ln._lines = [("ICM:  0.12 -0.34  9.81   12.3   -4.5    0.7", 1.0)]
    ts, text = next(ln.drain(1.0))
    assert isinstance(ts, float) and isinstance(text, str)
    assert P.looks_like_imu(text)


def test_single_joint_feedback_reads_column_1():
    """`f8` replies "8\\t0.7" -- (joint index, angle). Column 0 is the index and is
    CONSTANT, so reading it as the angle yields a flat trace that looks like a dead
    servo instead of an error. Confirmed against the real board."""
    ln = link()
    ln._lines = [(f"8\t{v}", 0.0) for v in (0.7, 0.8, 0.9, 1.0, 1.1)]
    ln._streaming = False
    ln.dry_run = False          # exercise the real width-detection path
    ln.ser = _ReplayPort([])
    ln.start_feedback(joint=8, wake=False, min_frames=5, timeout=0.0)
    assert ln.frame_width == 2
    assert ln.feedback_value_column == 1


def test_all_joint_feedback_reads_column_0_onward():
    ln = link()
    ln._lines = [("\t".join(["0.1"] * 9), 0.0)] * 5
    ln.dry_run = False
    ln.ser = _ReplayPort([])
    ln.start_feedback(joint=None, wake=False, min_frames=5, timeout=0.0)
    assert ln.frame_width == 9
    assert ln.feedback_value_column == 0


class _ReplayPort:
    def __init__(self, chunks):
        self.chunks = list(chunks)

    def write(self, data):
        pass

    def flush(self):
        pass

    def read(self, _n):
        return self.chunks.pop(0) if self.chunks else b""

    def close(self):
        pass


# -- polled-read token idioms -----------------------------------------------

def test_select_print_reselects_only_when_selection_changes():
    """`f<n>` SELECTS; `fp` prints once. A run of reads on one joint should cost a single
    `fp` each, not a redundant re-select.

    Not the default -- measured 1.7x slower than select-only (see SerialLink.poll_mode).
    Kept because it is the only form that guarantees no stream is left running.
    """
    ln = link()
    ln.poll_mode = "select-print"
    assert ln._poll_tokens(8) == ("f8", P.F_PRINT_ONCE)   # first: select + print
    assert ln._poll_tokens(8) == (P.F_PRINT_ONCE,)        # same joint: print only
    assert ln._poll_tokens(9) == ("f9", P.F_PRINT_ONCE)   # changed: re-select


def test_motion_command_invalidates_the_cached_selection():
    """The firmware resets measureServoPin on any non-f token, so the next read must
    re-select. Getting this wrong would `fp` against a stale/-1 selection."""
    ln = link()
    ln.poll_mode = "select-print"
    ln._poll_tokens(8)
    assert ln._selected == 8
    ln.send(P.move_cmd([(8, 10)]))
    assert ln._selected is None
    assert ln._poll_tokens(8) == ("f8", P.F_PRINT_ONCE)


def test_feedback_tokens_do_not_invalidate_the_selection():
    ln = link()
    ln.poll_mode = "select-print"
    ln._poll_tokens(8)
    ln.send(P.F_PRINT_ONCE)
    assert ln._selected == 8


def test_select_only_mode_uses_one_token():
    ln = link()
    assert ln._poll_tokens(8) == ("f8",)
    assert ln._poll_tokens(8) == ("f8",)


def test_poll_all_selects_sixteen():
    ln = link()
    ln.poll_mode = "select-print"
    assert ln._poll_tokens(P.FEEDBACK_ALL) == ("f16", P.F_PRINT_ONCE)


def test_default_poll_mode_is_the_measured_faster_one():
    """Guards the A/B result: select-only measured 28.4 Hz vs select-print's 16.4 Hz on
    consecutive reads, and was no worse interleaved."""
    assert link().poll_mode == "select-only"


# -- realtime mode ----------------------------------------------------------
# The patched firmware boots in preprogrammed mode, where the gP stream is capped at 5 Hz.
# Anything that reads the IMU stream has to ask for the other mode first -- 5 Hz is not a
# slower version of the measurement, it is a different one, and nothing about it looks
# broken from the outside.

class _ModeLink(SerialLink):
    """Dry-run link that answers the mode switch the way patched firmware would.

    The reply is injected into the line buffer on send, because realtime() reads the buffer
    directly rather than going through query() -- see its docstring for why.
    """

    #: lines queued AHEAD of the acknowledgement, i.e. a backlog from an earlier command
    noise = ()

    def __init__(self, supported=True, **kw):
        super().__init__(dry_run=True, verbose=False, **kw)
        self.supported = supported

    def send(self, text):
        ts = SerialLink.send(self, text)
        if text in (P.REALTIME_ON, P.REALTIME_OFF):
            for line in self.noise:
                self._lines.append((line, ts))
            if self.supported:
                self._lines.append(
                    (f"realtime {'on' if text == P.REALTIME_ON else 'off'}", ts))
            else:
                self._lines.append(("unknown command", ts))
        return ts


def test_realtime_reports_the_mode_it_set():
    with _ModeLink() as ln:
        assert ln.realtime(True) is True
        assert ln._realtime is True
        assert ln.realtime(False) is False
        assert ln._realtime is False


def test_realtime_reports_none_on_firmware_without_the_switch():
    """None, not False. False means "the mode is off"; None means "this robot cannot switch
    modes at all", and only the second one is a reason to stop."""
    with _ModeLink(supported=False) as ln:
        assert ln.realtime(True) is None
        assert ln._realtime is None


def test_imu_stream_enters_realtime_and_restores_it():
    with _ModeLink() as ln:
        list(ln.imu_stream(0.0))
        assert ln.sent.index(P.REALTIME_ON) < ln.sent.index(P.G_PRINT_STREAM)
        assert ln.sent[-1] == P.REALTIME_OFF, ln.sent


def test_leaving_realtime_is_acknowledged_not_fire_and_forget():
    """THE bug this exists for. `Xr` used to go out as a bare send() straight after `gp`
    stopped a 250 Hz stream -- exactly the case realtime() documents as losing the token.
    A dropped `Xr` strands the robot in realtime mode, where `i` is not interpolated and
    read_voice() discards every word, so it chimes at its wake word and then ignores it.
    Nothing clears the flag short of a real board reset, and a battery cycle is not one:
    the USB-powered board never restarts. So the exit must confirm, exactly like entry."""
    with _ModeLink() as ln:
        ln.realtime(True)
    assert ln._realtime is False, "close() must confirm the mode, not assume it"
    assert P.REALTIME_OFF in ln.sent


def test_a_dropped_xr_is_retried_until_acknowledged():
    """One silent drop must not strand the robot: realtime() retries."""
    class _DropsFirst(_ModeLink):
        drops = 1

        def send(self, text):
            if text == P.REALTIME_OFF and self.drops:
                self.drops -= 1
                return SerialLink.send(self, text)   # swallowed by the firmware, no reply
            return _ModeLink.send(self, text)

    ln = _DropsFirst()
    ln.realtime(True)
    assert ln.realtime(False) is False
    assert ln.sent.count(P.REALTIME_OFF) == 2, ln.sent


def test_imu_stream_leaves_an_already_realtime_link_alone():
    """control_loop puts the robot in realtime for the whole run; a stream inside that must
    not switch it back off underneath the loop."""
    with _ModeLink() as ln:
        ln.realtime(True)
        n = len(ln.sent)
        list(ln.imu_stream(0.0))
        assert P.REALTIME_OFF not in ln.sent[n:]


def test_imu_stream_still_runs_on_firmware_without_the_switch():
    """Degraded, warned about, but not a crash -- an unpatched robot still streams, just
    slowly, and a measurement session should say so rather than die."""
    with _ModeLink(supported=False) as ln:
        list(ln.imu_stream(0.0))
        assert P.G_PRINT_STREAM in ln.sent


def test_close_restores_preprogrammed_mode():
    """Realtime mode is not a state to strand the robot in -- with it on, `i` is no longer
    interpolated, so anything driving the robot in single poses moves in steps."""
    ln = _ModeLink()
    with ln:
        ln.realtime(True)
    assert P.REALTIME_OFF in ln.sent
    assert ln.sent.index(P.REALTIME_OFF) < ln.sent.index(P.T_REST)


def test_close_does_not_send_the_switch_when_it_was_never_on():
    ln = _ModeLink()
    with ln:
        pass
    assert P.REALTIME_OFF not in ln.sent


class _StaleLink(_ModeLink):
    """Answers correctly, but with the PREVIOUS command's reply queued in front of it."""

    noise = ("Changing volume to 0/10", "b")      # what `b0` leaves behind


def test_realtime_ignores_a_stale_reply_from_the_previous_command():
    """control_loop mutes the buzzer immediately before switching modes, and the mute reply
    ('Changing volume to 0/10') was being read as the answer to XR -- so a correctly patched
    robot was reported as having no realtime mode at all, and the loop refused to run."""
    with _StaleLink() as ln:
        assert ln.realtime(True) is True
        assert ln._realtime is True
