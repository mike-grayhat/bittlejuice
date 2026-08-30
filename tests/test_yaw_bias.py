"""Yaw-rate bias calibration.

Measured on this robot: sitting perfectly still for 60 s, the fused yaw drifts -1855 deg
(-0.540 rad/s, 5.2 full turns) while roll and pitch stay flat to 0.0003 deg/s. There is no
magnetometer, so nothing bounds the gyro's z-axis offset. The policy reads that as a constant
rotation and steers to cancel it.

The bias measured -0.540 +-0.008 rad/s across six 10 s blocks, which is what makes a 2 s
stationary calibration worth doing at all.
"""


import numpy as np
import pytest


import obs_builder as ob


def _still(bias_rad_s, seconds=2.0, hz=250, noise=0.0015, seed=0):
    """A stationary robot whose yaw integrates a constant gyro offset, wrapping at +-pi."""
    rng = np.random.default_rng(seed)
    t = np.arange(0, seconds, 1.0 / hz)
    yaw = bias_rad_s * t + rng.normal(0.0, noise, len(t))
    yaw = (yaw + np.pi) % (2 * np.pi) - np.pi        # wrapped, as the firmware reports it
    return list(zip(t, yaw))


@pytest.mark.parametrize("truth", [-0.540, -0.31, 0.0, +0.42])
def test_recovers_bias_through_wrapping(truth):
    got = ob.calibrate_yaw_rate_bias(_still(truth))
    assert abs(got - truth) < 0.01, f"recovered {got:+.3f} for a {truth:+.3f} rad/s bias"


def test_measured_bias_wraps_several_times_in_the_window():
    """Guard the guard: at -0.54 rad/s a 2 s window crosses the seam, so a calibration that
    forgot to unwrap would pass the other tests by luck on short windows only."""
    pts = _still(-0.540, seconds=20.0)
    yaw = np.array([p[1] for p in pts])
    assert np.abs(np.diff(yaw)).max() > 6.0, "test fixture never wraps; it proves nothing"
    assert abs(ob.calibrate_yaw_rate_bias(pts) + 0.540) < 0.01


def test_subtracting_the_bias_zeroes_the_reported_rate():
    """End to end: a stationary robot must report ~0 yaw rate after calibration."""
    pts = _still(-0.540, seconds=3.0)
    bias = ob.calibrate_yaw_rate_bias(pts)
    s = ob.SensorState(yaw_rate_bias=bias)
    rates = []
    for t, yaw in pts[::5]:                     # 50 Hz, as the control loop samples
        _g, w, ok = s.update(np.array([yaw, 0.0, 0.0]), t)
        if ok:
            rates.append(w[2])
    assert abs(float(np.mean(rates))) < 0.02, f"residual yaw rate {np.mean(rates):+.3f} rad/s"


def test_refuses_short_or_degenerate_windows():
    assert ob.calibrate_yaw_rate_bias([]) == 0.0
    assert ob.calibrate_yaw_rate_bias(_still(-0.5, seconds=0.2)) == 0.0
    # A wildly out-of-family rate means the robot was moving, not that the gyro is that bad.
    assert ob.calibrate_yaw_rate_bias(_still(-9.0, seconds=2.0)) == 0.0


def test_default_is_no_correction():
    """An uncalibrated SensorState must behave exactly as before this change."""
    a = ob.SensorState()
    b = ob.SensorState(yaw_rate_bias=0.0)
    for s in (a, b):
        s.update(np.array([0.0, 0.0, 0.0]), 0.0)
    _g1, w1, _ = a.update(np.array([0.1, 0.0, 0.0]), 0.02)
    _g2, w2, _ = b.update(np.array([0.1, 0.0, 0.0]), 0.02)
    np.testing.assert_allclose(w1, w2)
    assert abs(w1[2] - 0.1 / 0.02) < 1e-9
