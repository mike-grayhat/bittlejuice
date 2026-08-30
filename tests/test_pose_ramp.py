"""The host-side pose ramp is the only smoothing left between policy and servo.

Before the firmware patch, every `i` command was shaped by the firmware's
raised-cosine `transform()`, so even a single absolute pose command arrived as a ramp. That
interpolation was removed on purpose -- it blocked 16-24 ms per tick and was most of the
198 ms command->motion latency -- which is right for the 50 Hz stream, where the host IS the
trajectory, but left the loop's two ONE-SHOT moves (into the home pose at startup, back out
at teardown) as bare absolute commands. The servos then traverse them at full slew and the
robot jumps.

These tests pin the properties that make the replacement ramp worth having. They matter
because the ramp otherwise only ever executes against real hardware.
"""

import math

import numpy as np
import pytest

import control_loop as CL


class FakeLink:
    """Records what was sent; `sleep` is a no-op so the ramp runs at test speed."""

    def __init__(self, joint_reply=None):
        self.sent = []
        self._joint_reply = joint_reply

    def send(self, text):
        self.sent.append(text)

    def sleep(self, _seconds):
        pass

    def query(self, *_a, **_k):
        if self._joint_reply is None:
            raise RuntimeError("no reply")
        return self._joint_reply


JOINTS = [
    "left-back-shoulder-joint", "left-back-knee-joint",
    "left-front-shoulder-joint", "left-front-knee-joint",
    "right-back-shoulder-joint", "right-back-knee-joint",
    "right-front-shoulder-joint", "right-front-knee-joint",
]
DT = 0.02


def _sent_angles(link):
    """Recover the commanded degrees per tick, in the order _send_pose emitted them."""
    out = []
    for text in link.sent:
        nums = [float(x) for x in text.replace("i", " ").split()]
        out.append(np.array(nums[1::2], dtype=float))
    return np.array(out)


def test_ramp_ends_exactly_on_goal():
    """A ramp that lands near the goal would leave a step for the next command to take."""
    link = FakeLink()
    start = np.zeros(8)
    goal = np.radians(np.full(8, 40.0))
    CL._ramp_to_pose(link, JOINTS, start, goal, DT)

    final = _sent_angles(link)[-1]
    expected = np.array([CL.deg_to_petoi(n, a)[1] for n, a in zip(JOINTS, goal)])
    # The wire quantizes to integer degrees, so exactness is to the wire's resolution.
    assert np.allclose(final, np.round(expected), atol=0.5)


def test_ramp_bounds_peak_velocity_regardless_of_travel():
    """The point of the ramp: peak rate is bounded no matter how far the robot has to move.

    A fixed-duration ramp would not do this -- doubling the travel would double the peak
    rate, and the long moves are exactly the ones that jump.

    Measured over a 5-tick window, NOT tick to tick. The wire sends integer degrees
    (protocol.move_cmd rounds), so one quantum at 50 Hz is 50 deg/s: a 60 deg/s ramp comes
    out as alternating 1 and 2 degree steps, and a per-tick bound would be measuring the
    protocol's resolution rather than the trajectory. The first version of this test asserted
    per-tick and failed at 100 deg/s on a 10 degree move for exactly that reason. 100 ms is
    also the honest window -- it is several actuator time constants, so it is the rate the
    servo actually sees.
    """
    win = 5
    for travel_deg in (10.0, 40.0, 90.0):
        link = FakeLink()
        start = np.zeros(8)
        goal = np.radians(np.full(8, travel_deg))
        CL._ramp_to_pose(link, JOINTS, start, goal, DT)

        sent = _sent_angles(link)
        if len(sent) <= win:
            continue
        moved = np.abs(sent[win:] - sent[:-win]).max()
        peak_deg_s = moved / (win * DT)
        assert peak_deg_s <= CL.RAMP_DEG_PER_S * 1.1, (travel_deg, peak_deg_s)


def test_ramp_starts_and_ends_slowly():
    """Raised cosine, not linear: zero velocity at both ends is what makes it read as a
    settle rather than a lunge, and it is the shape the firmware used to supply."""
    link = FakeLink()
    CL._ramp_to_pose(link, JOINTS, np.zeros(8), np.radians(np.full(8, 60.0)), DT)

    step = np.abs(np.diff(_sent_angles(link)[:, 0]))
    mid = step[len(step) // 2]
    assert step[0] < 0.5 * mid and step[-1] < 0.5 * mid, step


def test_tiny_move_sends_the_goal_once_rather_than_ramping():
    link = FakeLink()
    goal = np.radians(np.full(8, 0.1))
    CL._ramp_to_pose(link, JOINTS, np.zeros(8), goal, DT)
    assert len(link.sent) == 1


def test_unparsable_joint_reply_returns_none_rather_than_guessing():
    """A ramp from a WRONG start pose is worse than no ramp: it opens with exactly the jump
    it exists to prevent. The caller must be able to tell that it has no start pose."""
    assert CL._read_current_pose(FakeLink(), JOINTS) is None
    assert CL._read_current_pose(FakeLink(joint_reply=["=", "garbage"]), JOINTS) is None


def test_current_pose_round_trips_through_the_wire_convention():
    """`j` degrees -> policy radians must invert deg_to_petoi, or the ramp starts in the
    wrong place on whichever joints have sign/offset set."""
    pose = np.radians(np.array([12.0, -30.0, 5.0, 22.0, -8.0, 17.0, 3.0, -25.0]))
    table = np.zeros(16)
    for name, ang in zip(JOINTS, pose):
        idx, deg = CL.deg_to_petoi(name, ang)
        table[idx] = deg
    reply = ["=", " ".join(str(i) for i in range(16)),
             " ".join(f"{v:.1f}" for v in table), "j"]

    got = CL._read_current_pose(FakeLink(joint_reply=reply), JOINTS)
    assert got is not None
    assert np.allclose(got, pose, atol=1e-6)
