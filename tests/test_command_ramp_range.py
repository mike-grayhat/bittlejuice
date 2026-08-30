"""The startup command ramp must stay inside the policy's trained command range.

Ramping the forward command from zero walks a command-responsive policy through a region it
never saw. Measured on a policy trained over 0.05-0.15 m/s, flat, deterministic, 32 envs:

    command 0.00   excess reversals 1.43x     <- off the bottom of the trained range
    command 0.02   excess reversals 0.47x
    command 0.05   excess reversals 0.60x

1.43x is the worst reading of any policy measured. It is easy to miss, because a policy
trained at one fixed speed ignores the command channel outright -- it walks +0.095 m/s at
every command from 0.00 to 0.20 -- so the ramp sweeps an input nothing reads.

It is currently masked by an accident: STARTUP_HOLD_TICKS is 15 and the ramp is 30 ticks, so
the hold happens to cover exactly the half of the ramp below 0.5 * target. That alignment
breaks for a target near the range floor, or with --startup-hold 0.
"""

import sys, os
import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "deploy"))
import control_loop as CL


class _Policy:
    control_dt = 0.02
    def __init__(self, vx=(0.05, 0.15), vy=(0.0, 0.0), wz=(-0.6, 0.6)):
        self.command_vx_range, self.command_vy_range, self.command_wz_range = vx, vy, wz


def _ramp(policy, command, ramp_ticks=30):
    floor, base = CL._ramp_floor_and_command(policy, np.asarray(command, dtype=float))
    import math
    return np.array([
        floor + (base - floor) * 0.5 * (1.0 - math.cos(math.pi * n / ramp_ticks))
        for n in range(ramp_ticks)
    ]), base


def test_no_ramp_tick_falls_below_the_trained_floor():
    """THE test. Every commanded value the policy ever acts on must be one it trained for."""
    p = _Policy()
    for target in (0.06, 0.08, 0.10, 0.15):
        vals, _ = _ramp(p, [target, 0.0, 0.0])
        assert vals[:, 0].min() >= p.command_vx_range[0] - 1e-12, (
            f"target {target} ramps down to {vals[:, 0].min():.4f}, below the "
            f"trained floor {p.command_vx_range[0]}")


def test_the_ramp_still_ends_at_the_requested_command():
    p = _Policy()
    vals, base = _ramp(p, [0.12, 0.0, 0.0])
    assert vals[-1, 0] == pytest.approx(0.12, abs=2e-3)
    assert base[0] == pytest.approx(0.12)


def test_the_ramp_is_still_monotonic_and_starts_at_zero_velocity():
    """Raised cosine: the command's own derivative must start at zero, or the ramp is a step."""
    p = _Policy()
    vals, _ = _ramp(p, [0.10, 0.0, 0.0])
    d = np.diff(vals[:, 0])
    assert np.all(d >= -1e-12), "ramp must not go backwards"
    assert d[0] < d[len(d) // 2], "must ease in, not start at full slope"


def test_a_command_at_the_floor_does_not_ramp_downward_through_nothing():
    p = _Policy()
    vals, _ = _ramp(p, [0.05, 0.0, 0.0])
    assert np.all(vals[:, 0] == pytest.approx(0.05))


def test_out_of_range_commands_are_clamped():
    p = _Policy()
    floor, base = CL._ramp_floor_and_command(p, np.array([0.30, 0.0, 0.0]))
    assert base[0] == pytest.approx(0.15)
    floor, base = CL._ramp_floor_and_command(p, np.array([0.01, 0.0, 2.0]))
    assert base[0] == pytest.approx(0.05)
    assert base[2] == pytest.approx(0.6)


def test_lateral_and_yaw_still_ramp_from_zero():
    """Zero is INSIDE their trained range, so the old behaviour is already correct there."""
    p = _Policy()
    floor, _ = CL._ramp_floor_and_command(p, np.array([0.10, 0.0, 0.4]))
    assert floor[1] == 0.0 and floor[2] == 0.0


def test_a_pre_range_export_keeps_the_old_behaviour_exactly():
    """Old .npz files must still fly, unchanged -- there is nothing to be consistent with."""
    p = _Policy(vx=None, vy=None, wz=None)
    floor, base = CL._ramp_floor_and_command(p, np.array([0.20, 0.0, 0.0]))
    assert np.all(floor == 0.0)
    assert base[0] == pytest.approx(0.20), "must NOT clamp when the range is unknown"
