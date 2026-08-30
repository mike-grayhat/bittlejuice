"""Builds the 33-dim policy observation from the real robot's sensors.

Mirrors sim/mj_vec_env.py's obs layout exactly:
    [base_ang_vel*ang_vel_scale (3), projected_gravity (3), commands*commands_scale (3),
     (dof_pos-default_dof_pos)*dof_pos_scale (8), dof_vel*dof_vel_scale (8), last_action (8)]

Parsing and orientation maths live in bittle_io (numpy + pyserial only, no mujoco/jax), so
the deploy path and the characterization suite share one implementation. Duplicating them
is how the two-line `j` reply bug survived as long as it did.

Three substitutions vs simulation, all measured rather than assumed
-------------------------------------------------------------------
1. projected_gravity comes from the IMU's FUSED roll/pitch, not from normalizing the raw
   accelerometer. The DMP has already removed linear acceleration, so it stays valid
   mid-gait where a raw accel reading does not. (The accel path exists as a cross-check;
   note its chip frame is rotated 180 deg about x -- see protocol.projected_gravity_*.)

2. base_ang_vel is finite-differenced from consecutive attitude reads. The firmware never
   emits raw gyro rates. Measured noise floor 0.41 rad/s at a 250 Hz sample rate, which
   the simulator's OBS_NOISE_ANG_VEL was raised to match.

3. dof_pos is ESTIMATED, not read. Reading real positions costs ~20 ms per joint, caps out
   near 9 Hz against a 50 Hz control loop, physically detaches the servo mid-read, and was
   measured returning intermittently corrupt values. Instead the host runs the same
   rate-limited first-order lag the simulator applies (mj_vec_env._apply_actuator_dynamics)
   over its own commanded targets. That reproduces what the sim calls dof_pos far better
   than either the raw command (no lag at all) or the real sensor (slow and unreliable).

   ServoModel is an OBSERVER, not a pre-filter on the command: control_loop sends the raw
   target and lets the physical servo perform what this models. Its state is the estimate
   and nothing else. It also integrates the UNQUANTIZED target while the wire rounds to
   integer degrees, so the estimate carries that divergence too -- which sim reproduces
   structurally rather than as noise (mj_vec_env's wire_quantize_deg).
"""


import math
import numpy as np

from bittle_io import protocol as P  # noqa: E402
from joint_calibration import JOINT_CALIBRATION  # noqa: E402
from policy_numpy import NumpyMLPPolicy  # noqa: E402


def parse_joint_angles(raw_reply: str, joint_names: list[str]) -> np.ndarray:
    """`j` reply -> joint angles in POLICY order, radians, MJCF convention.

    Note what this actually returns: the firmware's `currentAng`, i.e. the angle it
    believes it last commanded, read back from memory. It is not a measurement. Useful for
    checking the board agrees with us about the commanded pose; not a substitute for the
    servo feedback path, and not what feeds the observation (see module docstring).
    """
    table = P.parse_joint_table(raw_reply)          # 16 values, Petoi indices, degrees
    out = np.zeros(len(joint_names), dtype=np.float64)
    for i, name in enumerate(joint_names):
        cal = JOINT_CALIBRATION[name]
        out[i] = np.radians((table[cal.petoi_index] - cal.offset_deg) / cal.sign)
    return out


def parse_gyro_accel(raw_reply: str) -> tuple[np.ndarray, np.ndarray]:
    """`gp` reply -> (ypr_radians, accel_xyz).

    Kept for the bring-up scripts that call it directly. The control loop should use
    SensorState, which additionally handles rate estimation and staleness.
    """
    s = P.parse_imu(raw_reply)
    ypr = np.radians(s.ypr_deg)
    accel = s.accel if s.accel is not None else np.zeros(3)
    return ypr, accel


class ServoModel:
    """Host-side estimate of where the servos actually are.

    Rate-limited first-order lag, identical in form to the simulator's actuator model:
        y' = clip((x - y)/tau, -slew, +slew)
    with tau and slew carried in the policy .npz from the measured hardware values
    (tau ~33 ms from IMU-based frequency response, slew ~1.9 rad/s from the slowest joint).

    Uses the EXACT discretization, alpha = 1 - exp(-dt/tau), not the Euler dt/tau: at
    dt=20 ms and tau=33 ms those differ substantially, and the Euler form would make the
    estimate converge faster than the servo it is modelling.
    """

    def __init__(self, initial_pos: np.ndarray, dt: float, tau_s: float, slew_rad_s: float):
        self.pos = np.asarray(initial_pos, dtype=np.float64).copy()
        self.dt, self.tau, self.slew = dt, tau_s, slew_rad_s
        self._alpha = 1.0 - np.exp(-dt / tau_s)

    def update(self, target: np.ndarray) -> np.ndarray:
        delta = self._alpha * (np.asarray(target, dtype=np.float64) - self.pos)
        np.clip(delta, -self.slew * self.dt, self.slew * self.dt, out=delta)
        self.pos += delta
        return self.pos.copy()


class SensorState:
    """IMU-derived observation terms, with honest validity flags.

    YAW-RATE BIAS. Roll and pitch are gravity-referenced: the accelerometer continuously
    corrects them, so their rates come out unbiased (measured 0.000 +-0.08 rad/s with the
    robot stationary). Yaw has no such reference -- there is no magnetometer on this board --
    so it is pure gyro integration and inherits the gyro's z-axis offset directly.

    Measured on this robot, sitting perfectly still for 60 s: yaw_rate = -0.540 rad/s, i.e.
    -31 deg/s, accumulating -1855 deg (5.2 full turns) of phantom rotation. The simulator has
    no such bias, so a policy trained there reads a permanent -0.54 rad/s as "I am rotating"
    and steers to cancel it -- which is a constant turn. That is a mechanism for the heading
    drift seen on hardware, and it is invisible to any zero-mean noise model.

    The bias is extremely stable (-0.540 +-0.008 rad/s across six 10 s blocks), so a short
    stationary calibration removes ~98% of it. See calibrate_yaw_rate_bias below.
    """

    def __init__(self, stale_after_s: float = 0.05, yaw_rate_bias: float = 0.0):
        self._prev = None                 # (timestamp, ypr_rad)
        self.stale_after_s = stale_after_s
        self.yaw_rate_bias = float(yaw_rate_bias)

    def update(self, ypr_rad: np.ndarray, stamp: float):
        """Returns (projected_gravity, ang_vel_rad_s, ang_vel_valid)."""
        yaw, pitch, roll = ypr_rad
        gravity = P.projected_gravity_from_rpy(roll, pitch)

        prev, self._prev = self._prev, (stamp, np.asarray(ypr_rad, dtype=np.float64))
        if prev is None:
            return gravity, np.zeros(3), False
        dt = stamp - prev[0]
        if dt <= 0 or dt > self.stale_after_s:
            return gravity, np.zeros(3), False
        d = np.asarray(ypr_rad, dtype=np.float64) - prev[1]
        # Yaw wraps at +-pi; an unwrapped step across the seam reads as a huge false rate.
        # Measured: 5 wraps in 60 s of a stationary robot, each a 359.7 deg jump. Without
        # this line they alone put 12 rad/s of "noise" on a channel whose real noise is 0.19.
        d[0] = (d[0] + np.pi) % (2 * np.pi) - np.pi
        dyaw, dpitch, droll = d
        rates = np.array([droll, dpitch, dyaw]) / dt
        rates[2] -= self.yaw_rate_bias
        return gravity, rates, True


# One observation frame: ang_vel(3) + projected_gravity(3) + commands(3) + dof_pos(8)
# + dof_vel(8) + last_action(8). Mirrors sim/policy_io.NUM_OBS; the two are checked
# against each other at export time, where the .npz's first layer must accept
# NUM_BASE_OBS * (1 + obs_history_len).
NUM_BASE_OBS = 33
# Three appended channels when the policy carries a gait clock; see policy_io.GAIT_OBS.
NUM_GAIT_OBS = 3

# Channel names in build_observation's order, so an observation dump is readable and a
# sim-vs-hardware diff names the channel that disagrees rather than an index.
OBS_CHANNEL_NAMES = (
    [f"ang_vel_{a}" for a in "xyz"]
    + [f"gravity_{a}" for a in "xyz"]
    + ["cmd_vx", "cmd_vy", "cmd_wz"]
    + [f"dof_pos_{i}" for i in range(8)]
    + [f"dof_vel_{i}" for i in range(8)]
    + [f"last_action_{i}" for i in range(8)]
)
assert len(OBS_CHANNEL_NAMES) == NUM_BASE_OBS


class ObsHistory:
    """Ring of past observation frames, mirroring mj_vec_env's "history" group.

    WHY the policy may want this: the joint channels are open loop (see the module docstring),
    so the only real feedback is the IMU, and a footfall shows up there as a transient. A
    single-frame MLP cannot represent a transient -- it has no way to compute a difference
    between now and a moment ago. Stacking frames gives it that for the price of a ring buffer.

    ORDER MATTERS and must match the simulator exactly: the vector handed to the policy is
    [current, t-1, t-2, ..., t-N]. mj_vec_env snapshots its history BEFORE rolling the current
    frame in, so the current observation appears once, at the front, and never inside the
    history block. Getting this backwards would feed the policy a duplicated frame and drop
    the oldest one -- it would still run, and still be wrong.

    Zero-filled at startup, exactly as the simulator zero-fills on episode reset, so the first
    N ticks after standing up are consistent between the two.

    There used to be a prime() that seeded slot 0 with a synthetic "settled home pose" frame,
    on the stated grounds that mj_vec_env "clears the ring and then immediately rolls the
    post-reset observation into slot 0". It does not: measured on a real checkpoint, the
    actor's first observation after reset has 0 of 33 channels set in EVERY history slot.
    Priming therefore fed the policy a frame it never saw in training -- upright gravity and
    a live command where sim had zeros. Deleted rather than corrected, because the correct
    behaviour is this constructor's zeros and a method that only reintroduces the divergence
    has no caller worth having.
    """

    def __init__(self, history_len: int, frame_dim: int):
        self.history_len = int(history_len)
        self.frame_dim = int(frame_dim)
        self._past = np.zeros((self.history_len, self.frame_dim), dtype=np.float64)

    def extend(self, current: np.ndarray) -> np.ndarray:
        """Return [current, past...] and roll `current` into the ring for next tick."""
        current = np.asarray(current, dtype=np.float64)
        if self.history_len == 0:
            return current
        out = np.concatenate([current, self._past.reshape(-1)])
        self._past[1:] = self._past[:-1]
        self._past[0] = current
        return out


def calibrate_yaw_rate_bias(samples, min_samples: int = 100,
                            max_still_rate: float = 0.35) -> float:
    """Estimate the gyro's z-axis offset from a stationary robot.

    `samples` is an iterable of (stamp, yaw_rad). Returns the bias in rad/s, to be handed to
    SensorState. Uses the total unwrapped yaw over the whole window divided by its duration,
    not a mean of per-step rates: the two agree for a stationary robot, but the endpoint form
    is immune to the per-step differentiation noise entirely.

    Refuses to calibrate if the robot was evidently moving -- roll/pitch excursion is checked
    by the caller, and a |rate| above max_still_rate here means either real rotation or a
    bias far outside anything this hardware has shown (measured -0.54; the guard sits well
    above that but well below a hand-carried robot).
    """
    pts = [(float(t), float(y)) for t, y in samples]
    if len(pts) < min_samples:
        return 0.0
    ts = np.array([p[0] for p in pts])
    yaw = np.unwrap(np.array([p[1] for p in pts]))
    span = ts[-1] - ts[0]
    if span <= 0.5:
        return 0.0
    bias = (yaw[-1] - yaw[0]) / span
    if not np.isfinite(bias) or abs(bias) > max_still_rate * 4:
        return 0.0
    return float(bias)


def build_observation(
    dof_pos_prev: np.ndarray,
    dof_pos: np.ndarray,
    last_action: np.ndarray,
    command: np.ndarray,
    ang_vel: np.ndarray,
    projected_gravity: np.ndarray,
    policy: NumpyMLPPolicy,
    gait: "GaitClock | None" = None,
) -> np.ndarray:
    """Assemble the observation. Order must match mj_vec_env._update_observation.

    33 channels, or 36 when the policy was trained with a gait clock -- the three extra are
    APPENDED, so every index below 33 is unchanged and an obs dump stays comparable across
    both kinds of policy.
    """
    dof_vel = (dof_pos - dof_pos_prev) / policy.control_dt
    parts = [
        ang_vel * policy.obs_scale_ang_vel,
        projected_gravity,
        command * policy.commands_scale,
        (dof_pos - policy.default_dof_pos) * policy.obs_scale_dof_pos,
        dof_vel * policy.obs_scale_dof_vel,
        last_action,
    ]
    want = NUM_BASE_OBS
    if gait is not None:
        parts.append(np.asarray(gait.obs(), dtype=np.float64))
        want += 3
    obs = np.concatenate(parts).astype(np.float32)
    if obs.shape[0] != want:
        raise ValueError(f"observation is {obs.shape[0]}-dim, expected {want}")
    return obs


class GaitClock:
    """The host side of the gait phase, mirroring MjVecEnv exactly.

    Both sides call policy_io.gait_obs() for the channel values, so the sin/cos/normalisation
    cannot drift. What has to match HERE is the integration: phase += dt * hz each control
    tick, wrapped to [0,1). The env advances it once per step alongside the episode counter.

    Starts at phase 0 because the robot is standing still when the loop begins. The env
    randomises the start phase per episode precisely so that this is one of the cases the
    policy was trained on -- and so a timing slip mid-run is not out of distribution.
    """

    # DUPLICATED from sim/policy_io.py, deliberately: deploy/ runs on the robot with only
    # numpy and pyserial and must not import from sim/. Same arrangement as NUM_BASE_OBS.
    # tests/test_gait_phase.py asserts the two agree to float64, so the duplication cannot
    # drift silently -- which is the only reason it is acceptable.
    HZ_RANGE = (0.60, 1.15)
    HZ_MID = 0.5 * (HZ_RANGE[0] + HZ_RANGE[1])
    HZ_HALF = 0.5 * (HZ_RANGE[1] - HZ_RANGE[0])

    def __init__(self, hz: float, dt: float):
        lo, hi = self.HZ_RANGE
        if not (lo - 1e-9 <= hz <= hi + 1e-9):
            raise ValueError(
                f"gait cadence {hz} Hz is outside the trained range {lo}-{hi} Hz. The upper "
                f"bound is the servo: at ~34.5 deg peak-to-peak the gait fundamental alone "
                f"reaches the 137 deg/s slew limit at 1.26 Hz.")
        self.hz, self.dt, self.phase = float(hz), float(dt), 0.0

    def obs(self):
        ph = 2.0 * math.pi * self.phase
        return [math.sin(ph), math.cos(ph), (self.hz - self.HZ_MID) / self.HZ_HALF]

    def tick(self):
        self.phase = (self.phase + self.dt * self.hz) % 1.0
