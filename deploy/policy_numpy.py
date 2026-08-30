"""Dependency-free (numpy only) forward pass for the trained Bittle policy.

Loads the .npz produced by sim/export_policy_mj.py, which also embeds the
obs/action scaling constants so this module never needs its own copy of
sim/policy_io.py (deploy is a standalone package -- no jax/mujoco/
bittlejuice imports).
"""

import numpy as np


def _elu(x: np.ndarray) -> np.ndarray:
    # np.minimum(x, 0) inside expm1: np.where evaluates BOTH branches, so a large
    # positive activation overflows expm1 and warns even though where() discards the
    # result. Harmless numerically -- and precisely the wrong thing to have firing on
    # every forward pass, because a genuine activation blow-up is a real failure mode
    # here (see NumpyMLPPolicy's note on the run that reached 2.8e20 and then NaN).
    # Clamping the dead branch keeps the output identical and leaves the warning
    # meaningful.
    return np.where(x > 0, x, np.expm1(np.minimum(x, 0.0)))


class NumpyMLPPolicy:
    def __init__(self, npz_path: str):
        data = np.load(npz_path)
        num_layers = int(data["num_layers"])
        self._layers = [(data[f"W{i}"], data[f"b{i}"]) for i in range(num_layers)]

        # Optional state estimator. A policy trained with --estimator reads
        # [obs, history, v_hat], where v_hat is this network's prediction of the robot's own
        # linear velocity from [obs, history]. It exists because the actor has NO velocity
        # sense otherwise: ang_vel and projected_gravity are unchanged by a steady drift, and
        # the joint channels are the host's own commands played back. Absent for every policy
        # for policies trained without an estimator, in which case act() is unchanged.
        self.estimator_dim = int(data["estimator_dim"]) if "estimator_dim" in data else 0
        self._est_layers = []
        if self.estimator_dim:
            n_est = int(data["estimator_num_layers"])
            self._est_layers = [(data[f"E_W{i}"], data[f"E_b{i}"]) for i in range(n_est)]

        self.default_dof_pos = data["default_dof_pos"].astype(np.float64)
        self.action_scale = float(data["action_scale"])
        self.control_dt = float(data["control_dt"])
        self.obs_scale_ang_vel = float(data["obs_scale_ang_vel"])
        self.obs_scale_dof_pos = float(data["obs_scale_dof_pos"])
        self.obs_scale_dof_vel = float(data["obs_scale_dof_vel"])
        self.commands_scale = data["commands_scale"].astype(np.float64)
        self.joint_names = [str(n) for n in data["joint_names"]]
        # Actuator model, measured on the real robot and embedded at export time so the
        # deploy loop cannot drift from the value the policy was exported with.
        self.actuator_tau_s = float(data["actuator_tau_s"]) if "actuator_tau_s" in data else 0.033
        self.actuator_slew_rad_s = (float(data["actuator_slew_rad_s"])
                                    if "actuator_slew_rad_s" in data else np.radians(137.0))
        # How many PAST observation frames the actor expects appended after the current one.
        # 0 when the policy trained without history. A policy trained with --obs-history N reads
        # [current(33), t-1(33), ..., t-N(33)] -- see ObsHistory below and mj_vec_env's
        # get_observations, which this must mirror exactly.
        self.obs_history_len = (int(data["obs_history_len"])
                                if "obs_history_len" in data else 0)
        # Whether this policy carries a gait clock, and the cadence range it trained over.
        # Absent in an older .npz, which correctly reads as False.
        self.gait_phase = bool(data["gait_phase"]) if "gait_phase" in data else False
        self.gait_hz_range = (tuple(float(v) for v in data["gait_hz_range"])
                              if "gait_hz_range" in data else (0.60, 1.15))
        # Command ranges the policy was trained over, or None for exports predating them.
        # None MEANS SOMETHING to control_loop: it cannot keep the startup ramp inside a
        # range it does not know, so it falls back to ramping from zero and says so.
        def _range(key):
            return tuple(float(v) for v in data[key]) if key in data else None
        self.command_vx_range = _range("command_lin_vel_x_range")
        self.command_vy_range = _range("command_lin_vel_y_range")
        self.command_wz_range = _range("command_ang_vel_range")
        # ACTION BOUND, and it is not cosmetic -- omitting it diverges the robot.
        #
        # mj_vec_env.step clips the incoming action to +-clip_actions and stores the CLIPPED
        # value, which is what then appears as `last_action` in the next observation. This
        # module used to return the raw network output, so on hardware the loop was:
        #
        #     action -> last_action -> observation -> action ...
        #
        # with no bound anywhere. One out-of-distribution observation (an early hardware
        # run started with the robot tilted 25.6 deg, which the yaw calibration flagged) pushed
        # the action past what training ever produced, the larger last_action pushed it further,
        # and 127 ticks later the policy was emitting 2.8e20 deg/s of joint rate and then NaN.
        # The servos got a NaN pose command and the run crashed.
        #
        # Clipping here rather than in control_loop means the caller uses the clipped value for
        # BOTH the servo target and the next observation, exactly as the simulator does.
        self.clip_actions = (float(data["clip_actions"])
                             if "clip_actions" in data else 100.0)
        self.num_obs = int(self._layers[0][0].shape[0])
        # What the CALLER must supply: the actor's input minus the part act() computes itself.
        self.num_input_obs = self.num_obs - self.estimator_dim

    def _forward(self, layers, x: np.ndarray) -> np.ndarray:
        for i, (kernel, bias) in enumerate(layers):
            x = x @ kernel + bias
            if i < len(layers) - 1:
                x = _elu(x)
        return x

    def estimate(self, obs: np.ndarray) -> np.ndarray:
        """Predicted base linear velocity from the deployable observation, or an empty array."""
        if not self.estimator_dim:
            return np.zeros(0)
        return self._forward(self._est_layers, obs)

    def act(self, obs: np.ndarray) -> np.ndarray:
        """obs: (num_input_obs,) -> action: (8,) raw MLP mean output.

        With an estimator present the caller still supplies only what the ROBOT can measure;
        this method runs the estimator and appends its output itself, so the deploy loop never
        has to know the composition and cannot get the concatenation order wrong.
        """
        if obs.shape[-1] != self.num_input_obs:
            raise ValueError(
                f"policy expects {self.num_input_obs} inputs, got {obs.shape[-1]}. With "
                f"obs_history_len={self.obs_history_len} the observation must be the current "
                f"33 followed by {self.obs_history_len} past frames"
                + (f" (the {self.estimator_dim}-wide velocity estimate is appended internally)."
                   if self.estimator_dim else ".")
            )
        if not np.all(np.isfinite(obs)):
            # Fail loudly rather than propagate: a non-finite observation means the loop has
            # already diverged, and the next thing that happens is a NaN sent to the servos.
            raise FloatingPointError(
                "non-finite observation reached the policy -- the control loop has diverged; "
                "check that clip_actions is being applied and that the IMU is reporting"
            )
        if self.estimator_dim:
            obs = np.concatenate([obs, self.estimate(obs)], axis=-1)
        action = self._forward(self._layers, obs)
        return np.clip(action, -self.clip_actions, self.clip_actions)
