"""The firmware realtime-mode switch, from the host side.

The robot has two behaviours that cannot both be true. Stock: `i` pose commands are shaped
by the firmware's raised-cosine ramp and the gP IMU stream is capped at 5 Hz -- correct for
preprogrammed skills and gaits. Realtime (`XR`): `i` writes once and returns, gP runs at the
link's ceiling -- the only mode a 50 Hz host control loop can work in.

Neither failure announces itself. An interpolated `i` reads as sluggish walking; a 5 Hz IMU
stream reads as a policy that ignores its sensors, because obs_builder drops samples older
than 50 ms and ang_vel silently becomes zeros. So the control loop confirms the mode rather
than assuming it, and these tests pin that refusal.
"""

import pytest

import control_loop as CL
from bittle_io import protocol as P


class _StubLink:
    """Records what was sent; `realtime()` answers whatever the test dictates.

    Deliberately has NO query(): the loop must go through SerialLink.realtime(), which
    flushes and waits for its own acknowledgement. A query() would return the previous
    command's reply -- the buzzer mute sent moments earlier -- and read it as a refusal.
    """

    def __init__(self, ack=True):
        self.ack, self.sent = ack, []

    def realtime(self, on):
        self.sent.append(P.realtime_cmd(on))
        return self.ack

    def send(self, text):
        self.sent.append(text)


def test_entering_realtime_sends_the_switch_and_reports_success():
    link = _StubLink()
    assert CL._enter_realtime(link, dry_run=False) is True
    assert link.sent == [P.REALTIME_ON]


def test_stock_firmware_is_refused_rather_than_run_against():
    """THE case this exists for. Stock firmware has no `XR`; it reads the token as a module
    code, activates a module called R, deactivates the others, and says nothing about
    realtime. Running the loop anyway gives a robot with 16-24 ms of blocking per tick and
    an IMU stream that is stale on arrival -- both invisible in the loop's own output."""
    with pytest.raises(SystemExit, match="[Rr]ealtime"):
        CL._enter_realtime(_StubLink(ack=None), dry_run=False)


def test_a_robot_that_reports_mode_off_is_also_refused():
    """False proves the firmware understands the command, but the mode is still not on, so
    the loop must not proceed -- and the message must not blame unpatched firmware."""
    with pytest.raises(SystemExit, match="reported the mode OFF"):
        CL._enter_realtime(_StubLink(ack=False), dry_run=False)


def test_the_switch_goes_through_the_link_helper_not_a_bare_query():
    """Regression. control_loop had its own query()-based copy, which returned the reply to
    the buzzer mute sent immediately before it -- so a correctly patched robot was refused
    with 'got Changing volume to 0/10'."""
    assert not hasattr(_StubLink, "query")
    link = _StubLink()
    CL._enter_realtime(link, dry_run=False)
    assert link.sent == [P.REALTIME_ON]


def test_dry_run_does_not_touch_the_mode():
    """--dry-run writes nothing to the wire, so it must not claim a mode either -- and it
    must still work with no robot attached."""
    link = _StubLink()
    assert CL._enter_realtime(link, dry_run=True) is False
    assert link.sent == []
