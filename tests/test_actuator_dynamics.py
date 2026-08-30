"""Tests for the measured-hardware actuator model in mj_vec_env.

These check the sim now reproduces the numbers bittle_io measured on the real robot
(measurements/reference/hardware_params.json), and that the per-episode randomization behaves
as per-episode rather than per-step.
"""

import numpy as np
import pytest

import mj_vec_env as E


class _Stub:
    """Just enough of MjVecEnv to exercise _apply_actuator_dynamics in isolation."""

    def __init__(self, n=4, n_act=8, dt=0.02, tau=0.033, slew=1.9):
        self.num_envs, self.num_actions, self.dt = n, n_act, dt
        self.domain_rand_enabled = True
        self._actuator_tau = np.full(n, tau)
        self._actuator_slew = np.full(n, slew)
        self._servo_target = np.zeros((n, n_act))

    _apply_actuator_dynamics = E.MjVecEnv._apply_actuator_dynamics


def test_lag_reproduces_measured_step_response():
    """A step command must approach it with the commanded time constant.

    After one tau the first-order response should be ~63% of the way there.
    """
    tau, dt = 0.033, 0.02
    s = _Stub(tau=tau, slew=1e6)          # slew effectively disabled to isolate the lag
    cmd = np.ones((s.num_envs, s.num_actions))
    n_steps = int(round(tau / dt))
    for _ in range(n_steps):
        out = s._apply_actuator_dynamics(cmd)
    assert 0.5 < out.mean() < 0.8         # ~1 - 1/e = 0.63


def test_bandwidth_matches_the_measured_knee():
    """Sinusoid in -> the -3dB point should land near 1/(2*pi*tau).

    bodyid measured a 4.8 Hz knee; with tau = 33ms the model gives 1/(2*pi*0.033) = 4.8 Hz.
    Drive at the knee and check the amplitude ratio is ~1/sqrt(2).
    """
    tau, dt = 0.033, 0.02
    f_knee = 1.0 / (2 * np.pi * tau)
    s = _Stub(n=1, n_act=1, tau=tau, slew=1e6)
    t = np.arange(0, 4.0, dt)
    out = np.array([s._apply_actuator_dynamics(np.array([[np.sin(2 * np.pi * f_knee * ti)]]))[0, 0]
                    for ti in t])
    settled = out[len(out) // 2:]         # discard the startup transient
    ratio = np.ptp(settled) / 2.0         # input peak-to-peak is 2
    assert 0.55 < ratio < 0.85            # 1/sqrt(2) = 0.707


def test_slew_clamp_enforces_measured_velocity_limit():
    """The slowest real joint slews 109.7 deg/s = 1.91 rad/s; a huge step must not exceed it."""
    slew, dt = 1.9, 0.02
    s = _Stub(tau=1e-9, slew=slew)        # lag effectively disabled to isolate the clamp
    prev = s._servo_target.copy()
    out = s._apply_actuator_dynamics(np.full((s.num_envs, s.num_actions), 100.0))
    assert np.all(np.abs(out - prev) <= slew * dt + 1e-9)


def test_saturation_reaches_the_full_measured_slew():
    """Regression: the velocity limit must apply to the INCREMENT, not to the target.

    Clamping the target first and lagging afterwards compounds both effects, capping the
    achievable velocity at alpha*slew -- only ~45% of the measured rate over this tau
    range, making the sim servo slower than the hardware it imitates. Under a large step
    the model must move at EXACTLY the slew budget.
    """
    dt, slew = 0.02, 1.9
    s = _Stub(tau=0.033, slew=slew)
    prev = s._servo_target.copy()
    out = s._apply_actuator_dynamics(np.full((s.num_envs, s.num_actions), 100.0))
    np.testing.assert_allclose(np.abs(out - prev), slew * dt, rtol=1e-9)


def test_small_error_is_pure_lag_not_rate_limited():
    """Below saturation the response must be the tau lag, untouched by the clamp."""
    dt, tau = 0.02, 0.033
    s = _Stub(tau=tau, slew=1.9)
    small = 0.01                          # well inside the slew budget (1.9*0.02 = 0.038)
    out = s._apply_actuator_dynamics(np.full((s.num_envs, s.num_actions), small))
    alpha = 1.0 - np.exp(-dt / tau)
    np.testing.assert_allclose(out, alpha * small, rtol=1e-9)


def test_actuator_model_applies_even_without_randomization():
    """The actuator is the PLANT, not a disturbance.

    This used to be a passthrough when domain_rand_enabled was False, which meant mj_eval
    -- randomization off by default -- rendered a robot with instant servos while the real
    one runs a 33 ms lag and a rate limit that binds nearly every tick. Any sim-vs-hardware
    comparison made through that eval was comparing two different plants.
    """
    s = _Stub()
    s.domain_rand_enabled = False
    cmd = np.full((s.num_envs, s.num_actions), 1.0)
    out = s._apply_actuator_dynamics(cmd)
    assert np.all(out < cmd), "actuator model did not apply with randomization off"
    # Falls back to the measured nominal, not to whatever the last episode sampled.
    start = 0.0                                     # _Stub starts at zero
    alpha = 1.0 - np.exp(-s.dt / E.ACTUATOR_TAU_S_NOMINAL)
    step = min(alpha * (1.0 - start), E.ACTUATOR_SLEW_RAD_S_NOMINAL * s.dt)
    np.testing.assert_allclose(out, start + step, rtol=1e-9)


def test_nominal_values_match_the_measurements():
    assert E.ACTUATOR_TAU_S_NOMINAL == 0.033
    assert abs(np.degrees(E.ACTUATOR_SLEW_RAD_S_NOMINAL) - 137.0) < 0.5
    # and the nominal must sit inside the randomized range it is the centre estimate of
    assert E.ACTUATOR_TAU_S_RANGE[0] <= E.ACTUATOR_TAU_S_NOMINAL <= E.ACTUATOR_TAU_S_RANGE[1]
    assert (E.ACTUATOR_SLEW_RAD_S_RANGE[0] <= E.ACTUATOR_SLEW_RAD_S_NOMINAL
            <= E.ACTUATOR_SLEW_RAD_S_RANGE[1])


# -- ranges are anchored to the measurements --------------------------------

def test_ranges_bracket_the_measured_values():
    """Guards against someone retuning these away from what the hardware actually did."""
    # bodyid knee 4.84 Hz -> tau 32.9 ms; step fit 84.1 ms. Both must be inside the range.
    assert E.ACTUATOR_TAU_S_RANGE[0] <= 0.0329 <= E.ACTUATOR_TAU_S_RANGE[1]
    assert E.ACTUATOR_TAU_S_RANGE[0] <= 0.0841 <= E.ACTUATOR_TAU_S_RANGE[1]
    # sweep: slowest joint 109.7 deg/s, fastest 165.9 deg/s
    assert E.ACTUATOR_SLEW_RAD_S_RANGE[0] <= np.radians(109.7)
    assert np.radians(165.9) <= E.ACTUATOR_SLEW_RAD_S_RANGE[1] * 1.05
    # IMU observation noise, derived from the 20-25 Hz band of a 20 s walking
    # capture (gait power is at 1-5 Hz, so the top of the band is noise).
    #
    # This assertion used to read `>= 0.41`, "the measured noise floor". It was defending a
    # number that was never a noise floor: `imu --phase angvel-fallback` reports
    # np.abs(rates).std() pooled across axes, and in that data the yaw channel carried a
    # -0.4164 rad/s BIAS -- one the deploy loop measures and subtracts at startup. A green
    # test held the mistake in place, which is the same way the t63 lock below once did.
    #
    # Bracketed on BOTH sides now. Too low and the policy over-trusts a channel the robot
    # delivers noisier than that; too high and the channel carries no learnable signal, the
    # policy stops reading it, and it cannot then use the clean signal the robot does provide.
    # The second failure is the one that actually happened, twice over.
    assert 0.28 <= E.OBS_NOISE_ANG_VEL <= 0.35, "measured 0.282 rad/s walking"
    assert 0.0029 <= E.OBS_NOISE_GRAVITY <= 0.010, "measured 0.0029 walking"
    # measured mounting tilt 4.0 deg pitch / 2.3 deg roll must sit inside the randomized range
    assert E.IMU_TILT_RAD_RANGE[1] >= np.radians(4.0)


# -- the COMPOSED plant, which is what the robot actually is -----------------
#
# Every test above exercises the filter alone, and nothing asserted the PAIR -- which is
# how the MJCF actuator behind the filter went unmeasured while being assumed transparent.
#
# Two separate lessons are baked into the bounds below, and the second cost more than the
# first. (1) Measure the composition, not the pieces. (2) Lock it to the RIGHT measurement:
# these bounds were first aimed at the bodyid frequency-response tau (33 ms -> t63 40 ms)
# rather than the step measurement (84 ms -> t63 ~106 ms), and a lock on the wrong number
# is worse than no lock at all -- it actively pulled the plant to armature 0.0002, where
# the joints ring on every footfall. See the docstring below.

def _composed_step_response(amp_deg, xml_path=None, record_s=0.6):
    """Joint angle under a step, through filter -> MJCF actuator, base pinned.

    The suspended-legs-free condition bittle_io's `step` experiment measured under.
    Returns (t, normalized_progress) with the joint starting at the home pose.
    """
    import math

    import mujoco

    model = mujoco.MjModel.from_xml_path(xml_path or E.XML_PATH)
    data = mujoco.MjData(model)
    dt = model.opt.timestep
    aid = model.actuator("lf_knee").id
    qadr = model.jnt_qposadr[model.actuator("lf_knee").trnid[0]]
    y0 = 0.56
    mujoco.mj_resetDataKeyframe(model, data, model.key("home").id)
    base_qpos = data.qpos[:7].copy()
    data.ctrl[:] = y0

    def settle_step():
        mujoco.mj_step(model, data)
        data.qpos[:7] = base_qpos          # pin the base: one knee, robot suspended
        data.qvel[:6] = 0.0

    for _ in range(int(0.3 / dt)):
        settle_step()

    y1 = y0 + math.radians(amp_deg)
    alpha = 1.0 - math.exp(-E.CONTROL_DT / E.ACTUATOR_TAU_S_NOMINAL)
    max_delta = E.ACTUATOR_SLEW_RAD_S_NOMINAL * E.CONTROL_DT
    filt, t, y = y0, [], []
    n_sub = int(round(E.CONTROL_DT / dt))
    for k in range(int(record_s / E.CONTROL_DT)):
        filt += float(np.clip(alpha * (y1 - filt), -max_delta, max_delta))
        data.ctrl[aid] = filt
        for s in range(n_sub):
            settle_step()
            t.append(k * E.CONTROL_DT + (s + 1) * dt)
            y.append(float(data.qpos[qadr]))
    return np.array(t), (np.array(y) - y0) / (y1 - y0)


def _cross(t, frac, level):
    i = int(np.argmax(frac >= level))
    return t[i] if frac[i] >= level else float("nan")


def test_composed_plant_matches_the_measured_servo_step_response():
    """filter -> MJCF actuator must reproduce the servo STEP measured on the robot.

    THE TARGET HERE IS THE STEP MEASUREMENT, and getting that wrong is what this test
    was originally locked to. measurements/reference/step_summary.csv, joint 8, suspended:
    dead time 22.0 ms + tau 84.1 ms, so t63 ~= 106 ms and the 10-90% rise ~= 100 ms.

    The first version of this test asserted t63 in [20, 70] ms, aimed at 40 ms -- a number
    derived from the bodyid tau of 33 ms. But bodyid is a FREQUENCY RESPONSE measured with
    the servo continuously driven, and its own docstring says to use it for the shape of
    the response rather than for absolute gains. The two methods disagree (33 vs 84 ms) and
    The randomization range covers both instead of adjudicating between them; fitting
    a STEP response to the frequency-response number picked the wrong one of the two.

    The cost was real: matching t63 = 40 ms required armature 0.0002, which made the joints
    ~6x lighter than the physical links and left them ringing on every footfall. Same fixed
    policy, high-frequency (>5 Hz) share of true dof_vel while walking:

        armature 0.01   (original)  ->   2%      fwd +0.100 m/s
        armature 0.0002 (first fix) ->  78%      fwd +0.056 m/s
        armature 0.007  (this)      ->   4%      fwd +0.097 m/s

    Bounds are generous: this catches a 2x regression, not a 10% retune.
    """
    t, frac = _composed_step_response(8.0)
    t63 = _cross(t, frac, 0.632)
    rise = _cross(t, frac, 0.9) - _cross(t, frac, 0.1)
    assert 0.070 <= t63 <= 0.150, f"t63 {t63*1000:.0f} ms, measured step is ~106 ms"
    assert 0.060 <= rise <= 0.160, f"10-90% rise {rise*1000:.0f} ms, measured step is ~100 ms"


def test_composed_plant_reaches_the_measured_slew_rate():
    """Two-sided, because a one-sided '<= the limit' passes on a plant that is too slow.

    That is the §5.2 trap repeated one level up: the filter alone was already asserted to
    REACH its slew budget, but nothing asserted the joint does.
    """
    import math

    t, frac = _composed_step_response(30.0)
    deg = frac * 30.0
    best = 0.0
    for i in range(0, len(t), 4):
        j = int(np.argmax(t >= t[i] + 0.025))
        if j > i:
            best = max(best, abs(deg[j] - deg[i]) / (t[j] - t[i]))
    limit = math.degrees(E.ACTUATOR_SLEW_RAD_S_NOMINAL)
    assert 0.7 * limit <= best <= 1.35 * limit, \
        f"sustained {best:.0f} deg/s against a measured {limit:.0f} deg/s limit"


def test_composed_plant_does_not_overshoot():
    """The real servo shows no overshoot; a transparent tracker must not invent one."""
    _, frac = _composed_step_response(8.0)
    assert frac.max() < 1.10, f"overshoot {100*(frac.max()-1):.0f}%"


# -- the wire is integer degrees --------------------------------------------

def test_wire_quantization_rounds_the_plant_target():
    """protocol.move_cmd does int(round(deg)); at 137 deg/s that is 36% of a tick's motion."""
    import math

    q = math.radians(1.0)
    for raw_deg in (32.4, 32.6, -5.5, 0.2):
        raw = math.radians(raw_deg)
        got = math.degrees(round(raw / q) * q)
        assert abs(got - round(raw_deg)) < 1e-9


def test_wire_quantization_defaults_off_for_old_configs():
    """A cfgs.pkl without this key MEANS something, and it is not today's default.

    Same contract as observe_servo_estimate: defaulting it on would evaluate an old
    checkpoint against a plant it never trained against.
    """
    import math

    assert math.radians(float({}.get("wire_quantize_deg", 0.0))) == 0.0
    from config import get_cfgs
    env_cfg = get_cfgs()[0]
    assert env_cfg["wire_quantize_deg"] == 1.0, "new runs must model the real wire"


def test_imu_tilt_biases_gravity_by_the_expected_angle():
    """A pitch tilt of p must rotate the gravity vector by ~p radians."""
    p = 0.1
    g = np.array([[0.0, 0.0, -1.0]])
    r = np.array([[0.0]])
    gx, gy, gz = g[:, 0], g[:, 1], g[:, 2]
    tilted = np.stack([gx + p * gz, gy - r[:, 0] * gz, gz - p * gx + r[:, 0] * gy], axis=-1)
    tilted = tilted / np.linalg.norm(tilted, axis=-1, keepdims=True)
    angle = np.arccos(np.clip(np.dot(tilted[0], g[0]), -1, 1))
    assert angle == pytest.approx(p, abs=0.01)
