"""Shared PPO and environment configuration for the MuJoCo training pipeline.

`get_cfgs()` returns the four dicts describing the task: the environment plant, the
observation layout, the reward terms and their scales, and the command distribution.
`get_train_cfg()` returns rsl_rl's PPO/runner configuration.

These live in one module rather than inside mj_train.py because the test suite reads them
too -- the invariants in tests/ assert against the same numbers the training run uses, so a
hyperparameter cannot drift away from the test that guards it.
"""


def get_train_cfg(exp_name, entropy_coef=0.01, desired_kl=0.01):
    train_cfg_dict = {
        "algorithm": {
            "class_name": "PPO",
            "clip_param": 0.2,
            "desired_kl": desired_kl,
            # Lower this for domain-randomised runs: randomised dynamics make each update's
            # advantage estimate noisier, which weakens the reward gradient's ability to
            # out-pull a fixed entropy bonus.
            "entropy_coef": entropy_coef,
            "gamma": 0.99,
            "lam": 0.95,
            "learning_rate": 0.001,
            "max_grad_norm": 1.0,
            "num_learning_epochs": 5,
            "num_mini_batches": 4,
            # Adaptive, not fixed. The schedule's downward half (kl_mean > 2*desired_kl =>
            # lr /= 1.5) is the safety net that keeps a KL blow-up recoverable.
            "schedule": "adaptive",
            "use_clipped_value_loss": True,
            "value_loss_coef": 1.0,
        },
        "actor": {
            "class_name": "MLPModel",
            "hidden_dims": [512, 256, 128],
            "activation": "elu",
            "distribution_cfg": {
                "class_name": "GaussianDistribution",
                "init_std": 1.0,
                "std_type": "scalar",
            },
        },
        "critic": {
            "class_name": "MLPModel",
            "hidden_dims": [512, 256, 128],
            "activation": "elu",
        },
        "obs_groups": {
            "actor": ["policy"],
            "critic": ["policy"],
        },
        "num_steps_per_env": 24,
        "save_interval": 100,
        "run_name": exp_name,
        "logger": "tensorboard",
    }

    return train_cfg_dict


def get_cfgs():
    env_cfg = {
        "num_actions": 8,
        # [rad], matching bittle.xml's "home" keyframe (symmetric stand pose)
        "default_joint_angles": {
            "left-back-shoulder-joint": 0.56,
            "left-back-knee-joint": 0.56,
            "left-front-shoulder-joint": 0.56,
            "left-front-knee-joint": 0.56,
            "right-back-shoulder-joint": 0.56,
            "right-back-knee-joint": 0.56,
            "right-front-shoulder-joint": 0.56,
            "right-front-knee-joint": 0.56,
        },
        # bittle.xml's own declared DFS order
        "joint_names": [
            "left-back-shoulder-joint",
            "left-back-knee-joint",
            "left-front-shoulder-joint",
            "left-front-knee-joint",
            "right-back-shoulder-joint",
            "right-back-knee-joint",
            "right-front-shoulder-joint",
            "right-front-knee-joint",
        ],
        # No separate foot bodies; the foot contact spheres are geoms on these links.
        "feet_link_names": [
            "left-back-knee-link",
            "left-front-knee-link",
            "right-back-knee-link",
            "right-front-knee-link",
        ],
        "shoulder_link_names": [
            "left-back-shoulder-link",
            "left-front-shoulder-link",
            "right-back-shoulder-link",
            "right-front-shoulder-link",
        ],
        "feet_contact_force_threshold": 0.05,  # [N], standing per-foot load is ~0.6 N
        # [s] swing duration rewarded before landing. Shorter than legged_gym's 0.5 s: this
        # robot is far smaller and faster-legged than the quadrupeds that constant is tuned for.
        "feet_air_time_threshold": 0.15,
        # [m] foot-sphere centre height at which the clearance reward saturates (radius 5 mm,
        # so 15 mm centre is 10 mm of ground clearance).
        "feet_clearance_target": 0.015,
        # [s] continuous ground contact beyond which a foot counts as stuck. A 1.8 Hz gait
        # gives each foot ~0.36 s of stance, so this is clear of a healthy stride.
        "feet_stuck_after_s": 0.5,
        # PD gains, matching bittle.xml's <position kp="20" kv="0.5"> actuator defaults. Also
        # the base values _randomize_physics() scales per-env when domain_rand_enabled.
        "kp": 20.0,
        "kd": 0.5,
        "force_limit": 0.18,  # [N*m] servo torque limit, matching the MJCF forcerange
        # Matching bittle.xml's <joint damping/armature/frictionloss> defaults.
        "joint_damping": 0.01,
        "joint_armature": 0.01,
        "joint_frictionloss": 0.03,
        # Friction/damping/motor-strength/mass/COM randomisation, pushes, observation noise
        # and action-latency jitter. See mj_vec_env's module-level ranges.
        "domain_rand_enabled": True,
        # Rough terrain. Amplitude is scaled to THIS robot: 8 mm is ~9% of the 94 mm stand
        # height, where a legged_gym-typical 5 cm would be taller than the whole machine.
        "terrain_enabled": False,
        "terrain_amplitude": 0.008,             # [m] peak height deviation
        "terrain_horizontal_scale": 0.04,       # [m] per heightfield cell
        "terrain_vertical_scale": 0.001,        # [m] height quantisation
        "terrain_correlation": 0.10,            # [m] feature size
        # Per-env terrain difficulty ladder. When off, every env sits at terrain_amplitude;
        # when on, terrain_amplitude becomes the hardest rung rather than the only one.
        "terrain_curriculum": False,
        "terrain_curriculum_levels": 10,
        # Rate limit on the penalty curriculum's RISE, per update call. 0.0 disables it.
        "curriculum_max_rise": 0.0,
        # Fractions of the COMMANDED distance that promote and demote a rung.
        "terrain_promote_frac": 0.60,
        "terrain_demote_frac": 0.35,
        # [m] square field. Only needs one robot's travel (0.10 m/s x 20 s = 2 m) plus spawn
        # jitter: hfield_data is deep-copied into every per-env model, so oversizing is costly.
        "terrain_size": 6.0,
        "terrain_seed": 0,
        # [deg] per-episode incline, applied as a gravity tilt. Coupled to foot friction: the
        # robot slides whenever tan(slope) > mu, and mu bottoms out at 0.05 (critical grade
        # 2.9 deg). Guarded by test_slope_and_friction_leave_the_ground_walkable.
        "slope_deg_range": [0.0, 4.0],
        # Cap each episode's grade at this fraction of its own critical sliding grade,
        # atan(mu), so every episode is walkable rather than merely standable.
        "slope_friction_margin": 0.5,
        # Outer P-loop on heading error, written into the yaw-rate command the policy already
        # observes and is rewarded for tracking. The actor has no absolute heading reference:
        # projected_gravity is exactly yaw-invariant, so without this a rotation is both
        # invisible and unpenalised. kp is rad/s of command per rad of error.
        "heading_command": False,
        "heading_kp": 0.5,
        # -- termination ------------------------------------------------------------------
        # [deg] Roughly the recoverable limit for this geometry: past it a foot cannot be
        # placed under the CoM (94 mm up, 104 mm long) without falling further. Terminating
        # earlier makes "never deviate" dominate "correct when you do", which forces an
        # open-loop optimum; terminating later spends samples on episodes already lost.
        "termination_if_roll_greater_than": 45,
        "termination_if_pitch_greater_than": 45,
        # -- disturbance ------------------------------------------------------------------
        # [m/s] peak per-axis velocity kick. Two constraints bind: large enough that the gait
        # cannot absorb it passively, small enough to stay inside tracking_lin_vel's support.
        "push_dv_xy": 0.10,
        "push_dv_z": 0.05,
        # [rad/s] roll/pitch kick. A linear push on an upright, non-rotating body moves none
        # of the actor's observable channels, so this is the disturbance its sensors can see.
        # Sized against episode survival: 10 is ~3x the baseline failure rate, where avoiding
        # falls matters but walking still dominates the return.
        "push_dw_rp": 10.0,
        # [m] base pose, matching bittle.xml's "home" keyframe stand height.
        "base_init_pos": [0.0, 0.0, 0.0941],
        "base_init_quat": [1.0, 0.0, 0.0, 0.0],
        # [deg] per-axis random roll/pitch at reset. This is the lever that makes the IMU
        # necessary rather than merely useful.
        #
        # *** THESE DEFAULTS ARE A FINETUNING ENVIRONMENT, NOT A BOOTSTRAP ONE. ***
        # base_init_tilt_deg, the pushes and the slope build recovery in a policy that ALREADY
        # walks. From scratch they prevent walking entirely: the robot starts up to 20 deg from
        # upright, gets kicked, on an incline, and falls before it can act. The smoothness
        # penalties (action_jerk, action_slew, feet_slip, feet_stuck) do the same thing for the
        # same reason -- before a fresh policy earns tracking reward they dominate the return
        # and the gradient points at "move less", which is a local optimum it never leaves.
        # Use --penalty-curriculum, which stages them behind achieved episode length.
        #
        # Training from scratch: --base-init-tilt-deg 0 --push-dv-xy 0 --push-dw-rp 0
        # --slope-deg 0, terrain off, then finetune the demands back in.
        "base_init_tilt_deg": 20.0,
        "episode_length_s": 20.0,
        "resampling_time_s": 4.0,
        "action_scale": 0.25,
        "simulate_action_latency": True,
        # Bound on the commanded joint target, in action units. 3.0 is +-43 deg from the
        # default pose, the whole usable range of this geometry.
        #
        # This must stay bounded. Commanding past a joint limit is free in sim -- the limit
        # clamps it at no cost -- and past the servo's slew rate it is nearly free too, because
        # the actuator model rate-limits the target and the motion still looks smooth. An
        # unbounded policy learns to bang-bang: it commands far beyond what the servo can
        # reach, guaranteeing maximum rate. Sim shows a clean gait; the real servo buzzes.
        "clip_actions": 3.0,
        # [deg] The serial protocol sends INTEGER degrees, so the plant receives a quantised
        # target while the host's ServoModel observer integrates the unquantised one.
        # mj_vec_env reproduces both effects.
        "wire_quantize_deg": 1.0,
    }
    obs_cfg = {
        "obs_scales": {
            "lin_vel": 2.0,
            "ang_vel": 0.25,
            "dof_pos": 1.0,
            "dof_vel": 0.05,
        },
    }
    reward_cfg = {
        # Width of the velocity-tracking exponential. Must come DOWN relative to command^2 on
        # this robot: the servo slew limit caps it near 0.10 m/s, so with a legged_gym-typical
        # sigma the whole achievable range sits inside one sigma of the target, where exp() is
        # nearly flat and doubling the speed pays less than slowing down does. Set per run with
        # --command-vx / --tracking-sigma; 0.0036 at a 0.10 m/s command is the measured pairing.
        "tracking_sigma": 0.01,
        # [m] must track bittle.xml's "home" keyframe base z; at scale -50 a stale target
        # dominates the whole return.
        "base_height_target": 0.0941,
        #
        # SIZING A REWARD TERM. Use the identity
        #
        #     logged rew_X  =  scale * mean magnitude of X
        #
        # read straight off a run's own logged rew_* values, all of which share one
        # normalisation, and aim for 15-25% of that run's logged tracking reward. Do not
        # reconstruct per-episode sums to get there; the units do not line up with the logs.
        # A term above ~100% of achieved tracking has become the objective.
        #
        "reward_scales": {
            "tracking_lin_vel": 1.0,
            "tracking_ang_vel": 0.2,
            "lin_vel_z": -1.0,
            # Squared horizontal projected gravity, i.e. body tilt. The only penalty in the
            # set that ONLY attitude feedback can reduce: no feedforward gait can lower its
            # own tilt on ground it cannot sense.
            "orientation": 0.0,
            # Count of joints reversing direction while moving. Leave at 0.0 -- see
            # _joint_reversals in mj_vec_env for why this works as a metric and not as a
            # reward. Kept because tools/mj_deployability.py reports the same quantity.
            "joint_reversal": 0.0,
            "base_height": -50.0,
            "action_rate": -0.01,
            # Excess commanded joint rate over the servo's slew limit. Charges only
            # max(0, rate - limit), so the gait fundamental contributes zero and only chatter
            # is charged. Enable with --reward-scale after sizing it against the run's logs.
            "action_slew": 0.0,
            # Second difference of the action, i.e. chatter. action_rate cannot do this job:
            # for a sinusoid the first difference scales as f and the second as f^2, so
            # action_rate is dominated by the gait fundamental and tightening it slows the gait
            # before it touches anything else.
            "action_jerk": 0.0,
            # Action MAGNITUDE. Without it, clip_actions is the only bound on how far the
            # policy may command.
            "action_magnitude": -0.005,
            # Mechanical power, sum |tau * qdot| in watts. Rides the penalty curriculum: an
            # energy penalty on a policy that cannot yet walk rewards standing still.
            "power": 0.0,
            "similar_to_default": -0.02,
            "feet_air_time": 1.0,
            # Horizontal foot speed while in contact, i.e. dragging. Nearly free on a flat
            # plane with a 5 mm sphere; on a real grippy surface it is not. Roughly 14x larger
            # on terrain than on flat ground, so it only bites where the behaviour appears.
            "feet_slip": 0.0,
            # Clearance bonus, capped at feet_clearance_target: a slip penalty alone is
            # satisfied by standing still. Note this REWARDS height summed over feet, so two
            # high-lifting legs can outscore four mediocre ones.
            "feet_clearance": 0.0,
            # Count of feet continuously grounded beyond feet_stuck_after_s. The one term that
            # catches a towed leg: unlike the two above it cannot be masked by a healthy leg
            # elsewhere, because summing a PENALTY charges the dead leg rather than crediting
            # the live ones.
            "feet_stuck": 0.0,
        },
    }
    command_cfg = {
        "num_commands": 3,
        # 0 is included deliberately, so standing still is a trained behaviour rather than an
        # edge case the policy never sees. feet_air_time already zeroes its own reward near a
        # zero command. Forward-only; y and yaw stay 0 unless a run widens them.
        "lin_vel_x_range": [0.0, 0.22],
        "lin_vel_y_range": [0, 0],
        "ang_vel_range": [0, 0],
    }

    return env_cfg, obs_cfg, reward_cfg, command_cfg
