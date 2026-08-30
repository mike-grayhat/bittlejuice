"""Shared runtime sensor layer: the robot's state in the units the policy expects.

This is what deploy/obs_builder.py's two NotImplementedError stubs need, and it uses
the same parsers as the offline experiments so the measurements and the 50 Hz control loop
cannot drift apart.

Honesty about angular velocity
------------------------------
The policy's observation ([policy_io.py] obs layout) opens with base_ang_vel (3). The
firmware does not emit angular rates: VectorInt16 gy exists in OpenCatEsp32's imu.h but
print6Axis never prints it. What we can do is finite-difference the fused yaw/pitch/roll
between consecutive IMU reads -- which is a genuinely different signal, band-limited by
the IMU stream rate and noisy where the attitude estimate is.

So `ang_vel` is always returned WITH an `ang_vel_valid` flag, and never fabricated. If the
IMU is slower than the control loop, samples are marked stale rather than silently reused
as if they were fresh. Whether the differentiated signal is good enough is what
`imu --phase angvel-fallback` measures; if it is not, the answer is a firmware build that
prints `gy`.
"""

import time
from dataclasses import dataclass

import numpy as np

from . import protocol as P


@dataclass
class RobotState:
    stamp: float
    dof_pos_deg: np.ndarray | None          # (16,) commanded angles from `j`, degrees
    ypr_deg: np.ndarray | None              # (3,) yaw, pitch, roll
    accel: np.ndarray | None                # (3,) raw, or None if not compiled in
    projected_gravity: np.ndarray | None    # (3,) body-frame down, unit
    ang_vel_rad_s: np.ndarray               # (3,) roll, pitch, yaw rates
    ang_vel_valid: bool
    imu_stale: bool

    @property
    def roll_pitch_deg(self):
        return None if self.ypr_deg is None else self.ypr_deg[[2, 1]]


class StateReader:
    """Reads joint angles and IMU each tick and derives policy-frame quantities.

    `gravity_source` picks which estimate feeds projected_gravity:
      "rpy"   -- from fused roll/pitch. Preferred: the DMP has already removed linear
                 acceleration, so it stays valid mid-gait.
      "accel" -- normalized accelerometer. Only valid when not strongly accelerating.
      "both"  -- use rpy, but also expose the accel estimate for cross-checking.
    """

    def __init__(self, link, caps, gravity_source="rpy", stale_after_s=None, clock=None):
        self.link = link
        self.caps = caps
        # Single-source the clock from the link. Using time.perf_counter here instead
        # would double-source it: the reader would measure wall time while the link ran
        # on its own (possibly virtual) clock, and every finite-differenced rate would be
        # scaled by the ratio between them.
        self.clock = clock or link.clock
        self.gravity_source = gravity_source
        if gravity_source in ("accel", "both") and not caps.imu_has_accel:
            raise ValueError(
                f"gravity_source={gravity_source!r} needs the accelerometer, but this "
                "firmware build has PRINT_ACCELERATION off (see capabilities.json)."
            )
        # A sample older than ~1.5 IMU periods is held-over, not fresh.
        if stale_after_s is None:
            rate = caps.imu_stream_rate_hz or 0.0
            stale_after_s = (1.5 / rate) if rate > 0 else 0.1
        self.stale_after_s = stale_after_s
        self._prev = None   # (ts, ypr_rad)
        self.last_gravity_disagreement = None

    def read(self, joints=True):
        now = self.clock()
        dof = None
        if joints:
            lines = self.link.query(P.T_JOINTS, timeout=0.2, min_lines=2)
            try:
                dof = P.parse_joint_table("\n".join(lines))
            except P.ProtocolError:
                dof = None

        sample = self.link.imu_once(timeout=0.2)
        if sample is None:
            return RobotState(now, dof, None, None, None,
                              np.zeros(3), False, True)

        ypr_rad = np.radians(sample.ypr_deg)
        yaw, pitch, roll = ypr_rad

        grav = P.projected_gravity_from_rpy(roll, pitch)
        if sample.accel is not None:
            grav_accel = P.projected_gravity_from_accel(sample.accel)
            self.last_gravity_disagreement = float(np.linalg.norm(grav - grav_accel))
            if self.gravity_source == "accel":
                grav = grav_accel

        ang_vel, valid = self._rates(now, ypr_rad)
        return RobotState(now, dof, sample.ypr_deg, sample.accel, grav,
                          ang_vel, valid, imu_stale=False)

    def _rates(self, now, ypr_rad):
        """Finite-difference the fused attitude. Returns (roll, pitch, yaw) rates."""
        if self.caps.has_raw_gyro:
            raise NotImplementedError(
                "capabilities.json reports raw gyro output, but no parser exists for it. "
                "Add one rather than falling back to differentiation."
            )
        prev, self._prev = self._prev, (now, ypr_rad)
        if prev is None:
            return np.zeros(3), False
        dt = now - prev[0]
        if dt <= 0 or dt > self.stale_after_s * 4:
            return np.zeros(3), False
        d = ypr_rad - prev[1]
        # Yaw wraps at +-180 deg; an unwrapped step would read as a huge spurious rate.
        d[0] = (d[0] + np.pi) % (2 * np.pi) - np.pi
        dyaw, dpitch, droll = d
        return np.array([droll, dpitch, dyaw]) / dt, True

    def reset(self):
        self._prev = None
