"""MjVecEnv: batched Bittle locomotion env stepped by MuJoCo's C engine across CPU threads.

Why this exists
---------------
The Genesis/Metal pipeline (bittle_env.py) sustained ~7,700 env-steps/s at 512
envs on an M3 Pro, with 91% of training wall time in environment stepping.
Profiling that step showed only ~1/3 of it was physics -- the rest was per-step
PyTorch/MPS work on (512, <=14) tensors, where kernel-launch latency dominates
the actual arithmetic. Meanwhile plain `mujoco.rollout` across 12 CPU threads ran
the identical 512-env x 24-control-step workload in 0.082 s against Genesis's
1.45 s, while doing *twice* the physics work (4 substeps at dt=0.005 vs Genesis's
2 at dt=0.01). MJX was not an option: `jax.devices()` on this machine is
`[CpuDevice(id=0)]`.

So: MuJoCo C for physics, numpy for the per-step bookkeeping, and torch only at
the observation/reward boundary where rsl_rl needs tensors.

Contract
--------
Exposes the same rsl_rl VecEnv surface as BittleEnv, so OnPolicyRunner is
unchanged: `num_envs`, `num_actions`, `num_obs`, `max_episode_length`, `device`,
`get_observations()`, `reset()`, `step(actions)`.

Observation/action/reward semantics are ported from bittle_env.py verbatim --
including its axis convention (the MJCF chassis frame has local **y** forward,
not x) and its domain-randomization ranges. A policy trained here is meant to be
interchangeable with a Genesis-trained one, so any divergence is a bug.

Differences from bittle_env.py that are deliberate:
  - PD control is bittle.xml's own <position kp="20" kv="0.5" forcerange="+-0.18">
    actuators driven through `data.ctrl`, rather than re-implemented gains. Genesis
    needed explicit set_dofs_kp/kv because control_dofs_position bypasses the MJCF
    <actuator> block; MuJoCo does not.
  - Feet contact is read from the foot-site framepos sensors (returned in the same
    `rollout` call as the state) instead of per-link net contact force.
"""

import copy
import math
import os

import mujoco
import numpy as np
import torch
from mujoco import rollout
from tensordict import TensorDict

import policy_io as pio

XML_PATH = pio.XML_PATH

PHYSICS_DT = 0.005  # bittle.xml's <option timestep>
CONTROL_DT = pio.CONTROL_DT  # 0.02, the real robot's 50Hz control loop
N_SUBSTEPS = round(CONTROL_DT / PHYSICS_DT)  # 4

# A foot site sits at its sphere's centre (radius 0.005). At rest the settled contact leaves
# ~0.25mm of penetration, putting the centre at ~0.0048; a foot lifted by a 0.44 rad shoulder
# swing is already at ~0.0196. 0.008 separates those cleanly with room for terrain.
FEET_CONTACT_HEIGHT_THRESHOLD = 0.008

# A freejoint's 6 dofs are always 0:6 for this model (single floating base, nothing ahead of it
# in the tree), so the 8 hinge dofs are exactly 6:14 -- same assumption as domain_rand.py.
HINGE_DOF_SLICE = slice(6, 14)

# Domain-randomisation ranges.
#
# Foot-ground friction is set ABSOLUTELY rather than as a scale on the MJCF nominal, because
# the nominal is itself a measurement. Measured by inclined-plane test
# (bittle_io/experiments/friction.py), bare plastic feet:
#
#     smooth plastic   mu_s 0.156   (5 trials, 0.142-0.182)
#     fabric mat       mu_s 0.440   (2 clean trials, 0.435 / 0.445)
#
# The range runs below the lower measurement because these are STATIC coefficients and a
# walking foot is kinetic, which is lower again; and above the upper one because no two
# floors are alike. Slip is one of the two disturbances that cost this robot speed rather
# than episodes, and unlike a linear push it is IMU-observable: a slipping foot accelerates
# and rotates the body.
FOOT_FRICTION_RANGE = (0.05, 0.55)
# Everything that is not a foot keeps a modest multiplicative jitter. With priority=1 on the
# feet, MuJoCo takes the foot's coefficient at contact, so this has no effect on locomotion.
FRICTION_SCALE_RANGE = (0.8, 1.2)
DAMPING_SCALE_RANGE = (0.5, 2.0)
ARMATURE_SCALE_RANGE = (0.8, 1.2)
FRICTIONLOSS_SCALE_RANGE = (0.5, 2.0)
MOTOR_SCALE_RANGE = (0.8, 1.2)
TORQUE_SCALE_RANGE = (0.85, 1.0)
# The V1 unit these masses come from has plastic servos; V2's alloy P1S servos are heavier
# by an unmeasured amount across 8 joints.
BASE_MASS_ADD_RANGE = (0.0, 0.04)
# Battery placement alone moves the CoM by ~9.6 mm. Randomising wider is much cheaper than
# being wrong: an uncovered CoM offset shows up as a persistent lean.
COM_OFFSET_RANGE = (-0.012, 0.012)
LEG_MASS_SCALE_RANGE = (0.9, 1.1)
# The torso inertia is a uniform-box estimate, not a measurement (see bittle.xml), and a box
# ignores that the battery sits low and the shoulder servos sit at the corners. Randomise
# rather than assert precision we do not have.
BASE_INERTIA_SCALE_RANGE = (0.7, 1.8)

# At 2-4 s a 20 s episode takes ~6.7 pushes, and with ~1 s of recovery each that is a third
# of every episode spent in a state the speed reward cannot score.
PUSH_INTERVAL_S_RANGE = (4.0, 8.0)
# Defaults only: env_cfg["push_dv_xy"]/["push_dv_z"] override, so a run can sweep
# disturbance magnitude without editing this file.
#
# SIZE THE PUSH AGAINST THE REWARD, NOT AGAINST WALKING SPEED. tracking_lin_vel is
# exp(-err^2/tracking_sigma), so with sigma 0.0036 its velocity scale is sqrt(sigma) =
# 0.060 m/s. A push that lands the robot outside that support does not degrade the reward,
# it ZEROES it, and the only gradients left are survival and the penalty terms -- so the
# optimal policy becomes crawling. Median post-push tracking reward, as a fraction of the
# standing-still baseline: 174% at 0.05 m/s, 46% at 0.10, 0.2% at 0.20.
#
# Push size and the termination threshold are one knob, not two: raising pushes without
# raising the threshold converts disturbance into episode death, which teaches "never
# deviate" rather than "recover". tests/test_reward_support.py pins this.
PUSH_DV_XY_RANGE = (-0.10, 0.10)
PUSH_DV_Z_RANGE = (-0.05, 0.05)
# ANGULAR kick, roll and pitch, rad/s.
#
# A linear kick with the body upright and not rotating is UNOBSERVABLE to this policy by
# construction: the actor sees ang_vel, projected_gravity and its own servo echo, and a
# steady drift moves none of them. An angular kick is visible in both IMU channels
# immediately (ang_vel directly, projected_gravity as it integrates) and demands postural
# correction. It also leaves the velocity reward largely intact, which a large linear push
# does not -- see PUSH_DV_XY_RANGE.
PUSH_DW_RP_RANGE = (-3.0, 3.0)

# Per-axis initial roll/pitch at episode reset, degrees.
#
# This is what makes attitude feedback NECESSARY rather than merely useful. A disturbance
# punishes the absence of feedback but never requires it -- this robot is passively stable
# enough that a better open-loop gait absorbs anything thrown at it, and open-loop
# robustness is cheaper for PPO to find than sensing. A randomised initial attitude is a
# different category: the correct righting action depends on the tilt DIRECTION, which is
# knowable only from the IMU, so an open-loop rhythm generator fails by construction.
#
# Failure rate ramps steeply with this value: 0/128 episodes at 10 deg, 1 at 20, 37 at 30.
BASE_INIT_TILT_DEG = 20.0
EXTRA_LATENCY_PROB_RANGE = (0.0, 0.15)

# Below this the joint is not meaningfully moving and its velocity sign is dither. 5 deg/s,
# matching tools/mj_deployability.REVERSAL_MIN_DEG_S so the reward and the metric agree.
REVERSAL_MIN_RAD_S = math.radians(5.0)

# -- Penalty curriculum ------------------------------------------------------------------
# Which reward terms the curriculum factor scales: only those measured to block bootstrapping
# (see config.get_cfgs). With them on from iteration 0 a fresh policy holds ~18/1000 episode
# length indefinitely; with them off it reaches full length within ~10 iterations. The other
# penalties -- action_rate, action_magnitude, similar_to_default, base_height, lin_vel_z --
# stay at full strength throughout; there is no evidence they need staging.
# ---------------------------------------------------------------- terrain curriculum
# Per-environment terrain difficulty, following Rudin et al. (arXiv:2109.11978). Their rule,
# verbatim: promote when the robot walks past the border of its terrain patch; demote when it
# finished an episode having "moved by less than half of the distance required by its target
# velocity"; and loop environments that reach the top level back to a RANDOM level, so the
# population keeps seeing easy ground and does not forget how to walk on it.
#
# We have one heightfield rather than patches, so difficulty is its vertical scale: level L
# gets amplitude max * (L+1)/N. Each env already owns a private copy of hfield_size (verified:
# copy.copy(MjModel) does not share that array), so this costs one float write per promotion.
#
# The promote test is the part that is NOT Rudin's, because "walked past the border" has no
# analogue here -- we ask for a fraction of the commanded distance instead, which is the same
# quantity the demote test uses and keeps the two rules on one scale.
TERRAIN_CURRICULUM_LEVELS = 10
# Fractions of the commanded distance that promote and demote a rung. NOT Rudin's 0.80/0.50:
# those assume a robot that roughly tracks its commanded velocity, and this one does not.
# 0.10 m/s is near the Bittle's ceiling, so under training noise and terrain it delivers a
# fraction of what it is asked for -- 94% of command deterministic on flat, 76% during
# training, 43% deterministic on 20 mm. An 80% promote gate therefore sits above everything
# the robot ever achieves while a 50% demote gate fires constantly, which pins the ladder at
# the bottom and makes the curriculum inert.
#
# These are env_cfg-overridable (terrain_promote_frac / terrain_demote_frac) because they are
# properties of the ROBOT's achievable speed, not of the method. Retune them against measured
# fwd_vel_mps if the command range or the plant changes.
TERRAIN_PROMOTE_FRAC = 0.60
TERRAIN_DEMOTE_FRAC = 0.35

CURRICULUM_TERMS = ("action_jerk", "action_slew", "feet_slip", "feet_stuck", "power",
                    "joint_reversal")
# `power` and `joint_reversal` join the ramp for a different reason than the other four:
# both are minimised by a policy that does not move, so applying them before locomotion
# exists makes standing still optimal. The other four block bootstrap by pricing the
# flailing that precedes a gait; these two block it by pricing motion itself.
#
# Staging costs strength: the converged factor is ~0.8 rather than 1.0, so pick a scale for
# the strength you want AFTER that multiplier.

# The factor is driven by achieved episode length, NOT by an iteration schedule, which makes
# it a feedback controller rather than a timetable:
#
#     factor = clip((ema_episode_length - LO) / (HI - LO), 0, 1)
#
# Below LO the policy cannot yet walk and pays no smoothness cost at all; above HI it is
# surviving full episodes and pays in full. The feedback direction is the point: if the
# penalties ever start costing episodes, length falls, the factor backs off, and the policy
# recovers. A timetable cannot do that, because it does not know it went wrong.
#
# EMA rather than the instantaneous mean, so the factor cannot chase a single bad batch.

# Asymmetric rate limit on the factor's RISE, in factor-units per update call. 0.0 disables
# it.
#
# WHY: the controller above is proportional gain with lag -- the EMA delays the measurement,
# the penalties change episode length with further delay, and 300 steps of episode length
# spans the factor's whole range. That is the standard recipe for a limit cycle, and it duly
# produces one: the more penalty pressure a run carries, the wider the factor swings (std
# 0.13 with few penalties, 0.25 with many), so the policy optimises an objective that moves
# between "no smoothness penalties" and "full penalties" every few tens of iterations.
#
# It also contaminates measurement: within one converged run the factor correlates about
# -0.4 with slew saturation across checkpoints, so a checkpoint's smoothness is partly a
# record of where the curriculum happened to be when it was saved.
#
# ASYMMETRIC ON PURPOSE. Only the rise is limited; the fall stays instant, because the whole
# reason this is feedback rather than a schedule is that it must back off the moment the
# penalties start costing episodes. Slow engage, fast release.
CURRICULUM_MAX_RISE_PER_UPDATE = 0.0

CURRICULUM_EP_LEN_LO = 600.0
CURRICULUM_EP_LEN_HI = 900.0
CURRICULUM_EMA_ALPHA = 0.02

# Per-episode ground slope, applied by TILTING GRAVITY rather than the floor. On a plane the two
# are exactly equivalent -- rotating the world about a horizontal axis is rotating gravity in the
# opposite sense -- and gravity is a 3-vector in the model, so this costs nothing per step while
# a tilted floor geom would need re-collision. It is deliberately separate from the heightfield:
#
#   heightfield -> TRANSIENT, per-footfall disturbance, needs a reflex
#   slope       -> SUSTAINED, per-episode bias, needs an ESTIMATE
#
# The second is what makes projected_gravity worth reading. A slope is invisible to every other
# channel the policy has: joints are open-loop echoes of its own commands, and the commanded
# velocity does not change. The only way to know the ground is tilted is to look at the gravity
# vector, which is precisely the behaviour the blindfold test showed was missing.
SLOPE_DEG_RANGE = (0.0, 10.0)

# Name shared by the heightfield asset and the geom that draws it.
TERRAIN_NAME = "terrain"
# Spawn jitter on terrain, in metres from the field centre. Purpose is variety: the field
# itself is IDENTICAL across envs (see _add_terrain on why it cannot cheaply differ), so all
# the terrain diversity a run sees comes from starting each episode somewhere else on it.
# Kept well inside the field's radius so a full episode's travel stays on the hfield.
TERRAIN_SPAWN_XY_RANGE = (-0.6, 0.6)

# Deep action latency, in whole control steps of 20 ms. Off unless env_cfg names a range.
#
# Measured command -> visible body motion, same robot and battery, one session:
#
#     before the firmware patch    198 ms  (90-234, 15/15 trials)
#     after                         54 ms  (44-61,  13/15)
#     link round trip (control)   15.5 -> 20.3 ms
#     command -> JOINT motion, after                22 ms  (step, small amplitude)
#
# Removing the firmware's blocking interpolation ramp cut dead time 3.7x and collapsed the
# spread from 144 ms wide to 17. The patch did not require sim to change; it moved the
# HARDWARE onto what sim already assumed. The range below is set from the 22 ms measurement
# plus one step of pessimism on the upper side.
#
# WHAT LATENCY COSTS. Measured by sweeping dead time on the fitted plant, two seeds:
#
#     dead time   0 ms   20 ms   20-60   60-120   120-200   200-320
#     fwd m/s     .091    .090    .081     .064      .048      .035
#
# At the pre-patch 198 ms the robot sat in the 120-200 bucket; at 22 ms it sits at the top of
# the curve. Note this only shows up on a plant whose actuator model is correct -- a
# too-sluggish actuator masks dead time and makes latency look free.
#
# When implementing a delay, delay only what reaches the PLANT (a_{t-k}) and keep the
# observation honest (a_t). Feeding the delayed action back into `last_action` shows the
# policy an action it never emitted, and the resulting collapse looks like a latency cliff.
ACTION_LATENCY_STEPS_RANGE = (1, 2)   # 20-40 ms; measured 22 ms command -> joint motion

# IMU observation delay, in control steps. DISTINCT from action latency, because the delay
# lives in different channels:
#
#   dof_pos / dof_vel : NOT delayed. deploy computes them from its own commanded targets on
#                       the host, so they are available instantly. See obs_builder.
#   ang_vel / gravity : delayed by IMU sensing + on-chip fusion + serial transport. Measured
#                       under 10 ms: fusion contributes ~0 and only transport remains.
#   the action        : delayed by command transport -- and the servo's own 33 ms response is
#                       ALREADY modelled by _apply_actuator_dynamics, so it must not be
#                       double-counted here.
#
# Why the distinction matters: a uniform ACTION delay merely shifts the phase of a periodic
# gait. An OBSERVATION delay does not shift the gait -- it makes the feedback stale while the
# feedforward rhythm runs on time, so it degrades disturbance rejection specifically.
#
# ATTITUDE FUSION CONTRIBUTES NO MEASURABLE DELAY. Not "a small delay" -- none that 60 s of
# clean data can distinguish from zero. The method is differential, so almost nothing can
# contaminate it: print6Axis emits raw accelerometer and fused attitude on the SAME line from
# the same sample instant, so transport, dispatch and host timestamping are common-mode and
# cancel. Tilting the robot by hand at ~0.6 Hz keeps the accelerometer quasi-static, so both
# channels really are reporting the same angle (gain 1.00 +- 0.03, correlation 0.995).
# Cross-spectrum phase stays within -1.5..+2.4 deg across every bin carrying power, with no
# trend against frequency; a 20 ms delay would show 4.7 deg at 0.65 Hz and grow linearly.
#
# Two cautions, both pinned by tests/test_fusion_lag_estimators.py. Drive the robot by hand,
# not with its own legs: a swinging leg drives the accelerometer with tangential
# acceleration, so the two channels stop measuring the same quantity. And do not use
# cross-correlation -- on a band-limited signal it has a flat correlation ridge, and on
# synthetic data with a KNOWN 20 ms lag it reads 12.5 ms.
#
# So the range below models serial transport, not fusion: the host reads an IMU line roughly
# one-way-link-time old, ~10 ms against a round trip of 15-20 ms. Uniform over {0,1} steps
# averages 10 ms, which brackets both.
#
# The consequence is the useful result: fusion takes ~0 and transport ~20 ms both ways, so
# essentially all of the command->motion figure sits in the ACTION path, governed by
# ACTION_LATENCY_STEPS_RANGE. The observation path was never the bottleneck.
IMU_DELAY_STEPS_RANGE = (0, 1)

# ---------------------------------------------------------------------------------------
# WHY THE DELAY KNOBS ABOVE CAN LOOK INERT. A policy will use the IMU only if the
# environment makes it necessary. Blindfolding -- replacing ang_vel and projected_gravity
# with a constant "upright and still", in the current frame and every history frame -- costs
# a policy trained on flat ground with a low termination threshold essentially nothing, even
# under pushes 8x the trained magnitude. Such a policy is an open-loop rhythm generator, and
# a delay in a loop that does not exist changes nothing.
#
# That is not a bug in the observation pipeline; it is what the environment asked for. A low
# termination threshold means any disturbance big enough to need recovery ENDS the episode
# instead of teaching recovery, so "never deviate" strictly dominates "correct when you do".
#
# Making the policy use its one real sensor needs the environment to demand it: a high enough
# termination threshold that recovery is survivable, angular pushes, uneven ground, and a
# randomised initial attitude. tools/blindfold_test.py measures whether it worked.
# ---------------------------------------------------------------------------------------

# -- Measured actuator dynamics (bittle_io, see measurements/). The MJCF position actuator
# settles against the leg's tiny inertia in ~1-2 ms, i.e. effectively instantaneously.
# response rolls off at ~4.8 Hz (bodyid, IMU-measured with the servo continuously driven at
# 50Hz), which is tau ~= 33ms, or roughly 1.5 control steps.
#
# Without this, the policy learns gaits built on instantaneous joint jumps that the hardware
# physically cannot follow -- the single largest sim-to-real gap the characterization found.
#
# The range deliberately spans two disagreeing measurements rather than trusting either:
# bodyid's 33ms (continuously-driven, the regime a walking policy lives in) and step's 84ms
# (single interpolated move, feedback-polling-perturbed). Covering both is cheaper than being
# confidently wrong about which regime transfers.
ACTUATOR_TAU_S_RANGE = (0.020, 0.090)
# Nominal (best point estimate) used when randomization is OFF. The actuator model is the
# PLANT, not a disturbance, so it must apply even in a deterministic eval -- see
# _apply_actuator_dynamics. tau from `bodyid`; slew is the median across the eight joints.
ACTUATOR_TAU_S_NOMINAL = 0.033
ACTUATOR_SLEW_RAD_S_NOMINAL = math.radians(137.0)

# Slowest joint's measured slew, 109.7 deg/s (sweep, all 8 leg joints: 110-166 deg/s). The
# low end is what the safety clamp must respect, so it anchors the randomized range.
ACTUATOR_SLEW_RAD_S_RANGE = (1.6, 2.9)

# IMU mounting tilt, held CONSTANT within an episode (see _reset_idx). Measured 4.0 deg pitch /
# 2.3 deg roll on a level-ish stand, but treated as assembly variance rather than a calibration
# constant: it comes from leg attachment and servo-calibration imprecision, so it changes every
# time a leg is re-seated. That makes it un-subtractable and un-averageable -- a persistent
# within-episode bias, which is exactly what a per-step noise term fails to represent. Range is
# ~1.5x the measured magnitude to leave headroom after a rebuild.
IMU_TILT_RAD_RANGE = (-0.105, 0.105)  # +-6 deg

# Per-joint zero-offset error, constant within an episode. THIS WAS MISSING, and it is the
# randomization that most directly targets the drift seen on hardware: every servo's true
# zero differs from its commanded zero by a degree or two (leg seated a spline tooth off,
# imprecise servo calibration, assembly tolerance), which biases the stance asymmetrically.
#
# The policy cannot perceive it. The observation carries no linear velocity and no heading,
# so lateral drift is structurally unobservable -- it can correct yaw RATE but not sideways
# translation, and a small persistent asymmetry integrates unopposed. The only defence is to
# train across the distribution so the learned gait is robust to it rather than reliant on a
# perfectly symmetric robot.
#
# Applied to the ctrl target while being SUBTRACTED from the observed dof_pos, which is what
# makes it an unobservable disturbance rather than free information -- and it matches the
# deploy path, where obs_builder reports its own commanded estimate and is equally blind.
# +-2.5 deg, roughly 4x the <=0.6 deg offsets jointmap measured, since that sensor is itself
# unreliable and re-seating a leg can be much worse.
MOTOR_OFFSET_RAD_RANGE = (-0.0436, 0.0436)

# base_ang_vel is NOT a gyro reading -- the firmware never prints raw rates, so it is
# finite-differenced from the fused attitude stream, and differentiation sets its noise floor.
#
# Both values are derived from a 20 s WALKING capture on the patched firmware. Noise is
# separated from gait by taking the 20-25 Hz band of the PSD: gait power lives at 1-5 Hz, so
# the top of the band is noise.
#
#     channel            measured (walking)
#     roll/pitch rate     0.282 rad/s
#     yaw rate            0.192 rad/s
#     projected gravity   0.0029
#
# Measure this while WALKING, not while still: walking vibration is roughly double the static
# floor, so a static-only reading over-corrects. And beware two traps in estimating an
# angular-rate noise floor -- a constant yaw BIAS reads as noise unless it is subtracted
# first (deploy/control_loop.py measures and subtracts it at startup), and pooling axes with
# abs() before std() lets one noisy axis contaminate the clean ones.
#
# WHY OVERSTATING NOISE IS NOT MERELY CONSERVATIVE. Domain randomisation normally makes a
# policy robust by overstating noise. Past a point it does something else: the channel
# carries no learnable signal, the policy correctly stops reading it, and then it cannot read
# the clean signal the robot actually provides. projected_gravity is the channel that tells
# the policy which way is down, and burying it under an order of magnitude more noise than
# the robot has is one of the two independent reasons a policy ends up open-loop. This is the
# same failure as observing true dof_pos -- a sim-only fact the hardware cannot honour --
# with the sign reversed. tools/obs_diff.py checks it from the other side.
#
# Values below sit slightly above measurement, which is the right kind of pessimism.
OBS_NOISE_ANG_VEL = 0.30      # measured 0.282 walking (0.131 static)
OBS_NOISE_GRAVITY = 0.005     # measured 0.0029 walking; ~0.29 deg of tilt
# dof_pos / dof_vel carry NO noise when observe_servo_estimate is on, which is the default.
# They stop being sensor readings there: the robot cannot afford to read its joints (~20 ms
# each, capping near 9 Hz against a 50 Hz loop, and the read physically detaches the servo), so
# deploy/obs_builder.py computes them from its own commanded targets instead. A computed
# quantity has no measurement noise -- it has MODEL error, which this env represents properly by
# running the observer at nominal tau/slew while the plant runs at randomized ones.
#
# These two are kept for the observe_servo_estimate=False path only, where dof_pos really is a
# sensor reading and noise is the right model.
OBS_NOISE_DOF_POS = 0.01
# Deliberately larger than position noise: finite-differencing two noisy position reads at 50Hz
# amplifies position noise by roughly 1/dt*sqrt(2). Matches bittle_env.py's _add_obs_noise.
OBS_NOISE_DOF_VEL = 0.75


def make_terrain_field(nrow, ncol, horizontal_scale, correlation_m, seed):
    """Correlated random heightfield, normalised to [0,1] as MuJoCo's hfield_data requires.

    Smoothing is done in the frequency domain -- multiplying white noise by a Gaussian whose
    transform is exp(-2*pi^2*sigma^2*f^2) -- for two reasons. It needs no scipy, and the FFT
    makes the field exactly PERIODIC, so the tile has no seam and no edge artifact where a
    spatial-domain blur would have to invent boundary values.

    `correlation_m` is the feature size: below it the ground is smooth, above it uncorrelated.
    Set it near the body length (0.104 m) and the robot feels rolling ground; set it near the
    foot and it feels gravel, which at this robot's scale is closer to an obstacle course.
    """
    rng = np.random.default_rng(seed)
    noise = rng.standard_normal((nrow, ncol))
    fy = np.fft.fftfreq(nrow, d=horizontal_scale)[:, None]
    fx = np.fft.rfftfreq(ncol, d=horizontal_scale)[None, :]
    sigma = max(correlation_m, 1e-9) / 2.0
    kernel = np.exp(-2.0 * (np.pi ** 2) * (sigma ** 2) * (fy ** 2 + fx ** 2))
    field = np.fft.irfft2(np.fft.rfft2(noise) * kernel, s=(nrow, ncol))
    field -= field.min()
    peak = float(field.max())
    return (field / peak) if peak > 0 else field


def _build_model(strip_visuals: bool = True, terrain_cfg: dict | None = None) -> mujoco.MjModel:
    """Compile bittle.xml, optionally without any rendering-only assets.

    Training keeps one MjModel *per env* so domain randomization can vary physical properties
    per env slot (mujoco.rollout accepts a sequence of models). That only stays cheap if the
    visual payload is dropped first: with the 512x3072 skybox texture retained, 512 copies cost
    ~2.6 GB of RSS. Stripping meshes, textures and materials -- none of which affect physics,
    since the foot spheres are the only colliding geoms -- makes a copy negligible.
    """
    spec = mujoco.MjSpec.from_file(XML_PATH)
    if strip_visuals:
        # Strip by COLLISION PROPERTIES, not by geom type. Deleting only meshes missed the
        # primitive-shaped cosmetic geoms (the head), which then rode along in all 512 model
        # copies. contype==0 and conaffinity==0 is the precise definition of "cannot collide
        # with anything", i.e. visual-only, so this is both broader and safer: the floor and
        # the four foot spheres do collide and are untouched.
        for geom in list(spec.geoms):
            if geom.contype == 0 and geom.conaffinity == 0:
                spec.delete(geom)
        # Surviving geoms (the floor and the foot spheres) must drop their material references
        # before the materials themselves go, or compile fails on a dangling name.
        for geom in spec.geoms:
            geom.material = ""
        for mesh in list(spec.meshes):
            spec.delete(mesh)
        for material in list(spec.materials):
            spec.delete(material)
        for texture in list(spec.textures):
            spec.delete(texture)

    if terrain_cfg is not None:
        _add_terrain(spec, terrain_cfg)
    return spec.compile()


def _add_terrain(spec, cfg):
    """Add a heightfield ON TOP OF the existing floor plane, not instead of it.

    Keeping the plane matters: a hfield is finite, and a robot that walks off its edge would
    otherwise fall out of the world. With the field's base at z=0 and its elevations >= 0, the
    hfield is always the higher surface inside its footprint, so the plane only ever takes
    contacts outside it -- the robot steps down at most `amplitude` and keeps walking.

    Sizing is deliberately NOT the config's Genesis defaults. There, terrain_size 10 m had to
    span a shared grid holding every env; here each env is an independent world with its own
    origin, so the field only needs one robot's episode travel (0.10 m/s x 20 s = 2 m) plus
    spawn jitter. That difference is worth ~4x in memory: hfield_data is float32 and is
    DEEP-COPIED into all 512 per-env models, so a 10 m field at 0.03 m cells would cost
    ~230 MB of duplicated identical data.
    """
    size = float(cfg["size"])
    hscale = float(cfg["horizontal_scale"])
    n = max(int(round(size / hscale)), 4)
    data = make_terrain_field(n, n, hscale, float(cfg["correlation"]), int(cfg["seed"]))

    hf = spec.add_hfield()
    hf.name = TERRAIN_NAME
    # (radius_x, radius_y, elevation_z, base_z). base_z is the solid slab hanging below the
    # surface; it only needs to be thick enough that a foot cannot tunnel through.
    hf.size = [size / 2.0, size / 2.0, float(cfg["amplitude"]), 0.05]
    hf.nrow, hf.ncol = n, n
    hf.userdata = data.flatten().tolist()

    g = spec.worldbody.add_geom()
    g.name = TERRAIN_NAME
    g.type = mujoco.mjtGeom.mjGEOM_HFIELD
    g.hfieldname = TERRAIN_NAME
    g.pos = [0.0, 0.0, 0.0]
    # Same contact class as the floor it supplements -- contype=2/conaffinity=1 is what the foot
    # spheres are matched against (see bittle.xml's collision comment); anything else and the
    # feet would pass straight through the terrain.
    g.condim = 3
    g.contype, g.conaffinity = 2, 1
    g.friction = [0.9, 0.02, 0.01]   # per-foot friction is set on the FEET, not here


class MjVecEnv:
    def __init__(
        self,
        num_envs,
        env_cfg,
        obs_cfg,
        reward_cfg,
        command_cfg,
        device="cpu",
        num_threads=None,
        seed=0,
    ):
        self.num_envs = int(num_envs)
        self.num_actions = env_cfg["num_actions"]
        self.num_commands = command_cfg["num_commands"]
        # Gait phase: three appended channels and a per-episode cadence. Off unless asked,
        # so every existing config and every exported .npz keeps its 33-wide observation.
        self.gait_phase = bool(env_cfg.get("gait_phase", False))
        self.num_obs = pio.NUM_OBS_WITH_GAIT if self.gait_phase else pio.NUM_OBS
        self.device = torch.device(device)
        self.dt = CONTROL_DT
        self.max_episode_length = math.ceil(env_cfg["episode_length_s"] / self.dt)

        self.env_cfg = env_cfg
        self.cfg = env_cfg  # rsl_rl's VecEnv contract; the runner logs this into its config dump
        self.obs_cfg = obs_cfg
        self.reward_cfg = reward_cfg
        self.command_cfg = command_cfg
        self.obs_scales = obs_cfg["obs_scales"]
        self.reward_scales = dict(reward_cfg["reward_scales"])

        self.simulate_action_latency = env_cfg.get("simulate_action_latency", True)
        self.domain_rand_enabled = env_cfg.get("domain_rand_enabled", True)
        # Observe the servo-model estimate of joint state, as deploy does, rather than the
        # simulator's true dof_pos/dof_vel, which the hardware has no way to obtain. Default on;
        # see _update_observation for the measured consequence of getting this wrong.
        self.observe_servo_estimate = env_cfg.get("observe_servo_estimate", True)
        # The wire carries INTEGER DEGREES: protocol.move_cmd does int(round(deg)) because the
        # firmware parses with atoi. At the 137 deg/s slew ceiling a control tick moves at most
        # 2.74 deg, so 1-deg quantisation is ~36% of the largest possible per-tick step -- and
        # near stance, where the commanded deltas are small, it is most of the signal.
        #
        # Two structural consequences, both reproduced here rather than modelled as noise: the
        # PLANT receives the quantised target, while the host's ServoModel observer integrates
        # the unquantised one, so the dof_pos/dof_vel estimate (16 of the actor's 33 channels)
        # diverges from what the servo was actually told. Quantisation is applied to the plant
        # path only, before the actuator filter, matching where rounding happens on the wire.
        self._wire_quantize_rad = math.radians(float(env_cfg.get("wire_quantize_deg", 0.0)))
        self.rng = np.random.default_rng(seed)

        # -- terrain and slope, the two "not a flat level floor" mechanisms ------------------
        self.terrain_enabled = bool(env_cfg.get("terrain_enabled", False))
        self.slope_deg_range = tuple(env_cfg.get("slope_deg_range", (0.0, 0.0)))
        self.slope_enabled = float(max(self.slope_deg_range)) > 0.0
        # Fraction of the critical (sliding) grade a slope may reach. 0 disables the coupling
        # and samples slope and friction independently.
        self.slope_friction_margin = float(env_cfg.get("slope_friction_margin", 0.0))
        # Heading mode: the yaw-rate command becomes the output of a P-controller on heading
        # error rather than an independent sample. Off by default so every prior config is
        # unchanged. See _update_heading_command for why this is the lever on "where does it
        # end up" -- a heading error steers, and steering compounds quadratically.
        self.heading_command = bool(env_cfg.get("heading_command", False))
        self.heading_kp = float(env_cfg.get("heading_kp", 0.5))
        # ACTUATION LIMIT for the heading controller, kept SEPARATE from ang_vel_range.
        #
        # These were the same thing at first, and it broke mj_eval: ang_vel_range is the range
        # yaw-rate commands are SAMPLED from, which heading mode does not use at all -- it
        # computes the command instead -- while eval legitimately collapses that range to hold
        # one fixed command. Clipping the controller to a collapsed sampling range pinned its
        # output at zero, so the guard below (correctly) refused to run, and a trained heading
        # policy could not be evaluated. One concept, two jobs.
        #
        # Defaults to the width of ang_vel_range when that is non-degenerate, so training
        # configs need say nothing.
        default_max = float(max(abs(v) for v in command_cfg["ang_vel_range"]))
        self.heading_max_yaw_rate = float(
            env_cfg.get("heading_max_yaw_rate", default_max))
        if self.heading_command and self.heading_max_yaw_rate <= 0.0:
            # Still worth refusing: with a zero limit the controller commands exactly zero
            # forever and heading mode looks like a feature that simply did not help.
            raise ValueError(
                "heading_command needs a non-zero heading_max_yaw_rate (it defaults to the "
                "width of ang_vel_range, which here is degenerate). Pass --ang-vel-range at "
                "training time, or env_cfg['heading_max_yaw_rate'] when evaluating a fixed "
                "command."
            )
        terrain_cfg = None
        if self.terrain_enabled:
            terrain_cfg = {k: env_cfg[f"terrain_{k}"] for k in
                           ("amplitude", "horizontal_scale", "correlation", "size", "seed")}

        self._base_model = _build_model(terrain_cfg=terrain_cfg)
        self._nstate = mujoco.mj_stateSize(self._base_model, mujoco.mjtState.mjSTATE_FULLPHYSICS)
        self._init_terrain_sampler()

        self._resolve_indices()
        self._build_env_models()

        self.num_threads = int(num_threads or os.cpu_count() or 1)
        self._rollout = rollout.Rollout(nthread=self.num_threads)
        self._datas = [mujoco.MjData(self._base_model) for _ in range(self.num_threads)]

        self._init_state = self._make_init_state()
        self._alloc_buffers()

        self.reset()

    @property
    def episode_length_buf(self):
        """Torch view of the episode counter, required by rsl_rl's VecEnv contract.

        The counter lives in numpy because it is touched several times per step in the hot path;
        rsl_rl only reads it once, and `learn(init_at_random_ep_len=True)` reassigns it wholesale
        via `torch.randint_like`. The setter below funnels that back into the numpy buffer.
        """
        return torch.from_numpy(self._episode_length)

    @episode_length_buf.setter
    def episode_length_buf(self, value):
        if torch.is_tensor(value):
            value = value.detach().cpu().numpy()
        self._episode_length[:] = np.asarray(value, dtype=np.int64)

    # ------------------------------------------------------------------ setup
    def _init_terrain_sampler(self):
        """Cache what terrain_height() needs, so the hot path touches no MuJoCo structs."""
        if not self.terrain_enabled:
            self._terrain_grid = None
            return
        m = self._base_model
        hid = m.geom(TERRAIN_NAME).hfieldid[0] if hasattr(m.geom(TERRAIN_NAME), "hfieldid") \
            else m.hfield(TERRAIN_NAME).id
        self._terrain_hfield_id = hid
        self._terrain_nrow = int(m.hfield_nrow[hid])
        self._terrain_ncol = int(m.hfield_ncol[hid])
        adr = int(m.hfield_adr[hid])
        n = self._terrain_nrow * self._terrain_ncol
        self._terrain_grid = np.asarray(
            m.hfield_data[adr:adr + n], dtype=np.float64
        ).reshape(self._terrain_nrow, self._terrain_ncol)
        rx, ry, elev, _base = (float(v) for v in m.hfield_size[hid])
        self._terrain_radius = np.array([rx, ry])
        self._terrain_elev = elev
        # Per-env vertical scale. Identical to _terrain_elev for every env unless the terrain
        # curriculum is on, so every config written before it reproduces bit-for-bit.
        self._terrain_elev_per_env = np.full(self.num_envs, elev, dtype=np.float64)

    def terrain_height(self, xy):
        """Ground height [m] under world xy. Shape (..., 2) in, (...) out.

        Bilinear, matching how MuJoCo triangulates the field closely enough for the two uses
        here (contact test and clearance reward), both of which are thresholded and so tolerate
        sub-millimetre disagreement with the exact collision surface.

        Outside the field the answer is 0 -- the plane the hfield sits on. Clamping the indices
        gives that for free everywhere except a thin border, and the border is far outside the
        spawn jitter.

        This answers for the NOMINAL field. Per-env difficulty lives in terrain_height_env.
        """
        xy = np.asarray(xy, dtype=np.float64)
        if self._terrain_grid is None:
            return np.zeros(xy.shape[:-1])
        nrow, ncol = self._terrain_nrow, self._terrain_ncol
        # MuJoCo lays hfield rows along +y and columns along +x, spanning [-radius, +radius].
        u = (xy[..., 0] + self._terrain_radius[0]) / (2 * self._terrain_radius[0]) * (ncol - 1)
        v = (xy[..., 1] + self._terrain_radius[1]) / (2 * self._terrain_radius[1]) * (nrow - 1)
        u = np.clip(u, 0.0, ncol - 1.0)
        v = np.clip(v, 0.0, nrow - 1.0)
        c0 = np.floor(u).astype(np.int64); c1 = np.minimum(c0 + 1, ncol - 1)
        r0 = np.floor(v).astype(np.int64); r1 = np.minimum(r0 + 1, nrow - 1)
        du, dv = u - c0, v - r0
        g = self._terrain_grid
        top = g[r0, c0] * (1 - du) + g[r0, c1] * du
        bot = g[r1, c0] * (1 - du) + g[r1, c1] * du
        return (top * (1 - dv) + bot * dv) * self._terrain_elev

    def terrain_height_env(self, xy, env_idx=None):
        """Ground height under xy for SPECIFIC environments, honouring per-env difficulty.

        Split from terrain_height rather than folded into it because the two have different
        shape contracts and conflating them broke immediately: terrain_height answers for
        arbitrary query points (a 500-point sweep in the tests has no env axis at all), while
        this one requires xy's leading axis to be the env axis, so each row can be scaled by
        its own environment's amplitude.

        With the terrain curriculum off, every env shares one scale and this returns exactly
        what terrain_height does.
        """
        unit_scaled = self.terrain_height(xy)
        if self._terrain_grid is None:
            return unit_scaled
        elev = self._terrain_elev_per_env if env_idx is None \
            else self._terrain_elev_per_env[env_idx]
        # terrain_height already multiplied by the nominal scale; re-scale to this env's.
        ratio = elev / self._terrain_elev
        return unit_scaled * ratio.reshape((-1,) + (1,) * (np.ndim(unit_scaled) - 1))

    def _resolve_indices(self):
        m = self._base_model
        self._root_body_id = m.body("root").id
        self._joint_qpos_adr = np.array([m.joint(n).qposadr[0] for n in pio.JOINT_NAMES])
        self._joint_dof_adr = np.array([m.joint(n).dofadr[0] for n in pio.JOINT_NAMES])

        # bittle.xml declares its <actuator> block in a different per-leg order (lf,lf,lb,lb,...)
        # than our canonical joint order (lb,lb,lf,lf,...) -- same permutation mjx_env.py builds.
        joint_id_by_name = {n: m.joint(n).id for n in pio.JOINT_NAMES}
        actuator_joint_id = m.actuator_trnid[:, 0]
        self._canonical_to_actuator = np.array(
            [int(np.nonzero(actuator_joint_id == joint_id_by_name[n])[0][0]) for n in pio.JOINT_NAMES]
        )

        # Foot framepos sensors, added to bittle.xml for exactly this purpose. Each is 3 wide;
        # we only ever want z, so precompute those flat offsets into sensordata.
        self._foot_sensor_z_adr = np.array(
            [m.sensor(f"{name}_pos").adr[0] + 2 for name in pio.FOOT_GEOM_NAMES]
        )
        # x,y of the same framepos sensors, for the foot-slip (dragging) reward term.
        self._foot_sensor_xy_adr = np.array(
            [m.sensor(f"{name}_pos").adr[0] + k
             for name in pio.FOOT_GEOM_NAMES for k in (0, 1)]
        )
        # Actuator-force sensors, in CANONICAL joint order. The sensors are declared in
        # actuator order, which is not pio.JOINT_NAMES order, so reuse the same permutation
        # the ctrl path uses -- getting this wrong pairs each torque with another joint's
        # velocity and the power term still looks plausible.
        self._actuator_frc_adr = np.array(
            [m.sensor(f"{m.actuator(int(a)).name}_frc").adr[0]
             for a in self._canonical_to_actuator]
        )
        # Absolute offsets of the eight joint velocities within the FULLPHYSICS state row,
        # so the power term can read qdot at every substep without decoding the whole state.
        # Layout is [time, qpos, qvel, ...] -- see _decode_state.
        self._joint_qvel_state_adr = 1 + m.nq + self._joint_dof_adr
        self._joint_power = np.zeros(self.num_envs, dtype=np.float64)
        self.num_feet = len(pio.FOOT_GEOM_NAMES)
        # The geoms themselves, for the absolute foot-friction sampling in _build_env_models.
        self._foot_geom_ids = np.array([m.geom(n).id for n in pio.FOOT_GEOM_NAMES])

        self._leg_body_ids = np.array(
            [m.body(n).id for n in pio.SHOULDER_LINK_NAMES + pio.FEET_LINK_NAMES]
        )
        self._default_dof_pos = np.array(pio.DEFAULT_DOF_POS, dtype=np.float64)

    def _build_env_models(self):
        """One MjModel per env, each with its own sampled physical properties.

        brax/MJX randomize static model fields once per env slot and hold them fixed for the run;
        this mirrors that, and mirrors bittle_env.py's _randomize_physics, which samples once at
        build time rather than per episode.
        """
        # Recorded per env so the privileged critic can see which surface this robot is on --
        # the actor cannot, and friction changes what an action does to the body.
        self._foot_friction = np.full(
            self.num_envs, float(self._base_model.geom_friction[self._foot_geom_ids[0], 0]))

        if not self.domain_rand_enabled:
            if self.slope_enabled:
                # Slope is written into each model's opt.gravity per episode, so the envs cannot
                # share one model object -- mutating a shared model would tilt all 512 at once.
                # Physics is otherwise identical; only gravity will ever differ.
                self.models = [copy.copy(self._base_model) for _ in range(self.num_envs)]
            else:
                self.models = [self._base_model] * self.num_envs
            return

        # Coulomb friction must stay below the weakest randomized motor torque, or the joint is
        # immobile no matter what the policy commands. This silently held for several Genesis
        # runs at frictionloss=0.1 (max randomized 0.2 vs a 0.18 N*m limit), locking >=1 joint in
        # ~68% of robots and producing a dragging gait that looked like an RL tuning problem.
        frictionloss_max = FRICTIONLOSS_SCALE_RANGE[1] * self.env_cfg["joint_frictionloss"]
        torque_min = TORQUE_SCALE_RANGE[0] * self.env_cfg["force_limit"]
        assert frictionloss_max < torque_min, (
            f"max randomized joint frictionloss ({frictionloss_max:.3f} N*m) must stay below the "
            f"weakest randomized motor torque ({torque_min:.3f} N*m), else joints lock outright"
        )

        rng = self.rng
        n_legs = len(self._leg_body_ids)
        base_leg_masses = self._base_model.body_mass[self._leg_body_ids].copy()

        self.models = []
        for env_i in range(self.num_envs):
            m = copy.copy(self._base_model)
            m.geom_friction[:, 0] *= rng.uniform(*FRICTION_SCALE_RANGE)
            # ...then overwrite the feet with an absolute sample, so this env's ground contact is
            # one specific measured-plausible surface rather than a scaled guess.
            mu = rng.uniform(*FOOT_FRICTION_RANGE)
            m.geom_friction[self._foot_geom_ids, 0] = mu
            self._foot_friction[env_i] = mu
            m.dof_damping[HINGE_DOF_SLICE] *= rng.uniform(*DAMPING_SCALE_RANGE, size=8)
            m.dof_armature[HINGE_DOF_SLICE] *= rng.uniform(*ARMATURE_SCALE_RANGE, size=8)
            m.dof_frictionloss[HINGE_DOF_SLICE] *= rng.uniform(*FRICTIONLOSS_SCALE_RANGE, size=8)

            # kp (gainprm[:,0]) and kv (biasprm[:,2], stored negated) move together so the PD
            # relationship stays consistent -- overall servo strength / battery sag.
            motor_scale = rng.uniform(*MOTOR_SCALE_RANGE)
            m.actuator_gainprm[:, 0] *= motor_scale
            m.actuator_biasprm[:, 1] *= motor_scale
            m.actuator_biasprm[:, 2] *= motor_scale
            # Only ever reduced: real servos underperform spec far more often than exceed it.
            m.actuator_forcerange *= rng.uniform(*TORQUE_SCALE_RANGE)

            # Additive, not multiplicative: the literal planned Pi + wiring + mount payload, which
            # is always extra mass. Narrowed to 0-0.02kg in the Genesis runs because the body is
            # only 0.257kg, so the wider range put some envs in a "heavier AND weaker" corner that
            # was near-infeasible to stand tall in.
            m.body_mass[self._root_body_id] += rng.uniform(*BASE_MASS_ADD_RANGE)
            m.body_ipos[self._root_body_id, :2] += rng.uniform(*COM_OFFSET_RANGE, size=2)
            m.body_inertia[self._root_body_id] *= rng.uniform(*BASE_INERTIA_SCALE_RANGE)
            m.body_mass[self._leg_body_ids] = base_leg_masses * rng.uniform(
                *LEG_MASS_SCALE_RANGE, size=n_legs
            )
            self.models.append(m)

    def _make_init_state(self):
        m = self._base_model
        d = mujoco.MjData(m)
        mujoco.mj_resetDataKeyframe(m, d, 0)
        d.qpos[0:3] = self.env_cfg["base_init_pos"]
        d.qpos[3:7] = self.env_cfg["base_init_quat"]
        d.qpos[self._joint_qpos_adr] = self._default_dof_pos
        d.qvel[:] = 0.0
        mujoco.mj_forward(m, d)
        state = np.zeros(self._nstate)
        mujoco.mj_getState(m, d, state, mujoco.mjtState.mjSTATE_FULLPHYSICS)
        return state

    def _alloc_buffers(self):
        n, na = self.num_envs, self.num_actions
        f64, f32 = np.float64, np.float32

        self.state = np.tile(self._init_state, (n, 1))
        self._ctrl = np.zeros((n, N_SUBSTEPS, self._base_model.nu), dtype=f64)
        # Preallocated rollout outputs, so the hot loop allocates nothing per step.
        self._state_out = np.zeros((n, N_SUBSTEPS, self._nstate), dtype=f64)
        self._sensor_out = np.zeros((n, N_SUBSTEPS, self._base_model.nsensordata), dtype=f64)

        self.actions = np.zeros((n, na), dtype=f64)
        self.last_actions = np.zeros((n, na), dtype=f64)
        self.last_last_actions = np.zeros((n, na), dtype=f64)

        self.commands = np.zeros((n, self.num_commands), dtype=f64)
        self._commands_scale = np.array(pio.COMMANDS_SCALE, dtype=f64)
        self._commands_low = np.array(
            [self.command_cfg["lin_vel_x_range"][0], self.command_cfg["lin_vel_y_range"][0],
             self.command_cfg["ang_vel_range"][0]], dtype=f64)
        self._commands_high = np.array(
            [self.command_cfg["lin_vel_x_range"][1], self.command_cfg["lin_vel_y_range"][1],
             self.command_cfg["ang_vel_range"][1]], dtype=f64)
        self._resample_every = int(self.env_cfg["resampling_time_s"] / self.dt)

        # Desired world heading [rad] this episode; only meaningful in heading mode.
        self._heading_target = np.zeros(n, dtype=f64)
        self._episode_length = np.zeros(n, dtype=np.int64)
        self._prev_dof_vel = np.zeros((n, self.num_actions), dtype=f64)
        # Terrain curriculum: per-env difficulty level, off unless env_cfg asks.
        self.terrain_curriculum = bool(self.env_cfg.get("terrain_curriculum", False))
        self._terrain_levels = int(self.env_cfg.get(
            "terrain_curriculum_levels", TERRAIN_CURRICULUM_LEVELS))
        # Everyone starts on the easiest rung, as in Rudin: a population spread across
        # difficulty from step 0 learns the easy rungs from envs that are failing on hard ones.
        self._curriculum_max_rise = float(self.env_cfg.get(
            "curriculum_max_rise", CURRICULUM_MAX_RISE_PER_UPDATE))
        self._terrain_promote_frac = float(self.env_cfg.get(
            "terrain_promote_frac", TERRAIN_PROMOTE_FRAC))
        self._terrain_demote_frac = float(self.env_cfg.get(
            "terrain_demote_frac", TERRAIN_DEMOTE_FRAC))
        self._terrain_level = np.zeros(n, dtype=np.int64)
        self._episode_cmd_vx_sum = np.zeros(n, dtype=f64)

        # Penalty curriculum: off unless env_cfg asks, so every existing config reproduces.
        self.penalty_curriculum = bool(self.env_cfg.get("penalty_curriculum", False))
        self._curriculum_ep_len_ema = 0.0
        self._curriculum_factor = 0.0 if self.penalty_curriculum else 1.0
        # rsl_rl's OnPolicyRunner.learn(init_at_random_ep_len=True) overwrites
        # episode_length_buf with random values in [0, max_episode_length) to decorrelate
        # envs at startup. The first episode each env finishes is therefore an ARTIFACT: it
        # "ends" at up to 1001 steps having actually been stepped a handful of times. Feeding
        # those to the EMA seeds it at ~1000 and pins the factor at 1.0 from iteration 0 --
        # i.e. silently disables the curriculum, which is exactly what happened on the first
        # run using it. Count an env's endings only after WE have reset it once.
        self._curriculum_env_ready = np.zeros(n, dtype=bool)
        self.reset_buf = np.ones(n, dtype=bool)
        self.rew_buf = np.zeros(n, dtype=f64)

        self.feet_air_time = np.zeros((n, self.num_feet), dtype=f64)
        self.last_feet_contact = np.zeros((n, self.num_feet), dtype=bool)
        self.feet_air_time_rew = np.zeros(n, dtype=f64)
        self._feet_slip = np.zeros(n, dtype=f64)
        self._feet_stuck = np.zeros(n, dtype=f64)
        self._feet_ground_time = np.zeros((n, self.num_feet), dtype=f64)
        self._feet_clearance = np.zeros(n, dtype=f64)
        self._prev_foot_xy = None

        self.next_push_step = np.zeros(n, dtype=np.int64)
        # This episode's ground slope, in degrees. Privileged-critic material: the actor has to
        # infer it from gravity, which is the whole point, but the value function has no reason
        # to guess at something the simulator knows exactly.
        self._slope_deg = np.zeros(n, dtype=f64)
        self.extra_latency_prob = np.zeros(n, dtype=f64)
        imu_range = self.env_cfg.get("imu_delay_steps_range")
        if imu_range:
            d = int(max(imu_range)) + 1
            self._imu_buf = np.zeros((n, d, 6), dtype=f64)   # ang_vel(3) + gravity(3)
            self._imu_head = 0
            self._imu_delay = np.full(n, int(min(imu_range)))
        else:
            self._imu_buf = None
            self._imu_delay = None

        lat_range = self.env_cfg.get("action_latency_steps_range")
        if lat_range:
            depth = int(max(lat_range)) + 1
            self._action_buf = np.zeros((n, depth, self.num_actions), dtype=f64)
            self._action_head = 0
            self._action_delay = np.full(n, int(min(lat_range)))
        else:
            self._action_buf = None
            self._action_delay = None
        # Per-episode hardware character: actuator lag/slew and IMU mounting tilt. Sampled in
        # _reset_idx alongside extra_latency_prob, so each episode sees one consistent "robot"
        # rather than a new one every step.
        self._actuator_tau = np.full(n, ACTUATOR_TAU_S_RANGE[0], dtype=f64)
        self._actuator_slew = np.full(n, ACTUATOR_SLEW_RAD_S_RANGE[1], dtype=f64)
        self._imu_tilt = np.zeros((n, 2), dtype=f64)   # (roll, pitch) offset, radians
        self._motor_offset = np.zeros((n, self.num_actions), dtype=f64)
        # Filter state: the servo's believed position, which lags the commanded target.
        self._servo_target = np.zeros((n, self.num_actions), dtype=f64)
        # The OBSERVER's copy of that filter -- deploy/obs_builder.ServoModel, mirrored. It is
        # a second, independent filter on purpose: the plant (above) runs at this episode's
        # randomized tau/slew, while this one runs at the NOMINAL values the deploy .npz carries,
        # because that is the only pair of numbers the real host has. The gap between them is the
        # hardware's actual observer error, reproduced rather than approximated with noise.
        self._servo_estimate = np.zeros((n, self.num_actions), dtype=f64)
        # ...and its previous value, since the deploy loop reports dof_vel as a finite difference
        # of consecutive estimates rather than from any velocity sensor (there isn't one).
        self._servo_estimate_prev = np.zeros((n, self.num_actions), dtype=f64)

        self.obs_buf = np.zeros((n, self.num_obs), dtype=f32)
        # phase in [0,1); hz is this episode's commanded cadence.
        self._gait_phase = np.zeros(n, dtype=f64)
        self._gait_hz = np.full(n, pio.GAIT_HZ_MID, dtype=f64)
        self._obs_tensor = torch.zeros((n, self.num_obs), dtype=torch.float32, device=self.device)

        # --- privileged critic ------------------------------------------------------------
        # Everything the actor structurally cannot know, handed to the CRITIC only. This is
        # asymmetric actor-critic: the critic's job is to estimate the value of a state, and it
        # does that badly when the state is partially observed. The actor is untouched, so
        # nothing here reaches the robot -- the critic is discarded after training.
        #
        # Motivated by the observation audit: of the actor's 33 inputs, only 6 (ang_vel and
        # projected_gravity) are real measurements. base_lin_vel in particular is the quantity
        # the whole tracking_lin_vel reward is computed from, and the actor has NO velocity
        # sense whatsoever -- so without this the critic must infer, from a gravity vector and a
        # noisy rate, the very number the reward is built on.
        self.num_privileged_obs = (
            3      # base_lin_vel -- unobservable, and what tracking_lin_vel scores
            + 8    # true dof_pos error (the actor sees only its own servo estimate)
            + 8    # true dof_vel
            + 8    # servo tracking error: true dof_pos minus the estimate == contact/load
            + 4    # foot contacts
            + 1    # foot friction this episode
            + 1    # actuator tau
            + 1    # actuator slew
            + 8    # motor zero offsets
            + 2    # IMU mounting tilt
            + 1    # ground slope this episode [deg]
            + 1    # terrain height under the base
        )
        self.privileged_buf = np.zeros((n, self.num_privileged_obs), dtype=f32)
        self._privileged_tensor = torch.zeros((n, self.num_privileged_obs), dtype=torch.float32,
                                              device=self.device)

        # --- state-estimate slot ----------------------------------------------------------
        # A placeholder group the ALGORITHM fills, not the environment. PPOWithEstimator writes
        # its velocity estimate here each step before the actor reads it; the env only has to
        # declare the width so MLPModel can size the actor's input and RolloutStorage has
        # somewhere to keep the value that was actually used. Zero when the feature is off, in
        # which case the group is not emitted at all and every prior config is unchanged.
        self.estimator_dim = int(self.env_cfg.get("estimator_dim", 0))
        self._estimate_tensor = (
            torch.zeros((n, self.estimator_dim), dtype=torch.float32, device=self.device)
            if self.estimator_dim > 0 else None
        )

        # --- observation history ----------------------------------------------------------
        # A ring of past ACTOR observations. With the joint channel open loop, a footfall exists
        # only as a transient in the IMU, and a single-frame MLP cannot represent a transient at
        # all -- it has no way to compute a difference. This is the cheapest fix for that, and
        # unlike the recurrent actor it costs the deploy side only a ring buffer.
        self.obs_history_len = int(self.env_cfg.get("obs_history_len", 0))
        if self.obs_history_len > 0:
            self._obs_history = np.zeros((n, self.obs_history_len, self.num_obs), dtype=f32)
            self.num_history_obs = self.obs_history_len * self.num_obs
        else:
            self._obs_history = None
            self.num_history_obs = 0

        # Reward scales are pre-multiplied by dt, exactly as bittle_env.py does, so the reward
        # magnitudes stay comparable between the two pipelines.
        for name in self.reward_scales:
            self.reward_scales[name] *= self.dt
        self.reward_names = list(self.reward_scales)
        self.episode_sums = {k: np.zeros(n, dtype=f64) for k in self.reward_names}
        self.episode_fwd_vel_sum = np.zeros(n, dtype=f64)
        self.episode_slew_sat_sum = np.zeros(n, dtype=f64)
        self.episode_power_sum = np.zeros(n, dtype=f64)
        self.episode_cmd_rate_sum = np.zeros(n, dtype=f64)
        self.episode_lat_vel_sum = np.zeros(n, dtype=f64)
        self.episode_step_count = np.zeros(n, dtype=f64)

        self.extras = {}
        self._features = {}

    # ------------------------------------------------------------- kinematics
    @staticmethod
    def _inv_rotate(vec, quat):
        """Rotate world-frame `vec` (n,3) into the body frame given by `quat` (n,4, wxyz)."""
        w, xyz = quat[:, :1], quat[:, 1:]
        t = 2.0 * np.cross(xyz, vec)
        return vec - w * t + np.cross(xyz, t)

    @staticmethod
    def _quat_to_euler_deg(quat):
        """Intrinsic xyz (roll, pitch, yaw) in degrees, matching Genesis's quat_to_xyz(rpy=True)."""
        w, x, y, z = quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3]
        roll = np.arctan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y))
        pitch = np.arcsin(np.clip(2.0 * (w * y - z * x), -1.0, 1.0))
        yaw = np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
        return np.degrees(np.stack([roll, pitch, yaw], axis=-1))

    def _decode_state(self):
        """Split the FULLPHYSICS state array into the quantities obs/rewards need.

        FULLPHYSICS layout is [time, qpos, qvel, act, plugin_state]; this model has no actuator
        activation state and no plugins, so qpos starts at 1 and qvel right after it.
        """
        m = self._base_model
        qpos = self.state[:, 1 : 1 + m.nq]
        qvel = self.state[:, 1 + m.nq : 1 + m.nq + m.nv]

        base_quat = qpos[:, 3:7]
        # A MuJoCo freejoint's angular dofs (qvel[3:6]) are already expressed in the body-local
        # frame -- that is what makes quaternion integration work, and it was confirmed against
        # the imu_gyro sensor in mjx_env.py. Only the linear part needs rotating.
        self._features = {
            "base_pos": qpos[:, 0:3],
            "base_quat": base_quat,
            "base_ang_vel": qvel[:, 3:6],
            "base_lin_vel": self._inv_rotate(qvel[:, 0:3], base_quat),
            "projected_gravity": self._inv_rotate(
                np.broadcast_to(np.array([0.0, 0.0, -1.0]), (self.num_envs, 3)), base_quat
            ),
            "base_euler": self._quat_to_euler_deg(base_quat),
            "dof_pos": qpos[:, self._joint_qpos_adr],
            "dof_vel": qvel[:, self._joint_dof_adr],
        }
        return self._features

    # -------------------------------------------------------------- stepping
    def step(self, actions):
        actions = np.asarray(
            actions.detach().cpu().numpy() if torch.is_tensor(actions) else actions, dtype=np.float64
        )
        np.clip(actions, -self.env_cfg["clip_actions"], self.env_cfg["clip_actions"], out=self.actions)

        if self._action_buf is not None:
            # Deep latency: each env executes the action it emitted `_action_delay` steps ago.
            # Ring buffer rather than a shift, so cost does not grow with depth.
            self._action_head = (self._action_head + 1) % self._action_buf.shape[1]
            self._action_buf[:, self._action_head] = self.actions
            take = (self._action_head - self._action_delay) % self._action_buf.shape[1]
            exec_actions = self._action_buf[np.arange(self.num_envs), take]
        elif self.simulate_action_latency:
            # Legacy 1-2 step model, kept so older checkpoints reproduce.
            use_extra = self.rng.random(self.num_envs) < self.extra_latency_prob
            exec_actions = np.where(use_extra[:, None], self.last_last_actions, self.last_actions)
        else:
            exec_actions = self.actions

        # Which action drives the OBSERVER. The deploy loop computes
        # `target = last_action * scale + default` and feeds THAT to servo.update(), so its
        # observer is driven by the DELAYED command -- the one the servo is actually
        # executing, which is what a model of servo position should track. Driving it with
        # the undelayed action runs sim's observer one control step ahead of the robot's,
        # across dof_pos and dof_vel, i.e. 16 of the actor's 33 channels.
        #
        # Set observer_leads_plant=True to reproduce checkpoints trained the old way.
        observer_action = (self.actions if self.env_cfg.get("observer_leads_plant", False)
                           else exec_actions)
        # BEFORE the motor offset is folded in below, because the host has no idea its servo
        # zeros are off.
        self._update_servo_estimate(observer_action * self.env_cfg["action_scale"]
                                    + self._default_dof_pos)
        target_dof_pos = exec_actions * self.env_cfg["action_scale"] + self._default_dof_pos
        if self._wire_quantize_rad > 0.0:
            # Integer-degree wire format; see the __init__ note. Before the actuator filter,
            # because rounding happens on the wire and the filter models the servo's response
            # to what arrived. The observer above deliberately got the unquantized target.
            #
            # Rounding here in radians on a radians(1 deg) grid is equivalent to hardware's
            # round-after-conversion only because every joint's calibration is sign=+1,
            # offset=0.0 (deploy/joint_calibration.py) -- a nonzero offset would shift
            # the grid the robot rounds on and this would model the wrong lattice.
            target_dof_pos = np.round(target_dof_pos / self._wire_quantize_rad) \
                * self._wire_quantize_rad
        target_dof_pos = self._apply_actuator_dynamics(target_dof_pos)
        # Servo zeros are not where we think they are; see MOTOR_OFFSET_RAD_RANGE.
        target_dof_pos = target_dof_pos + self._motor_offset
        # Same target held across all substeps: the policy runs at 50Hz, the physics at 200Hz.
        self._ctrl[:, :, self._canonical_to_actuator] = target_dof_pos[:, None, :]

        self._rollout.rollout(
            self.models, self._datas, self.state, self._ctrl,
            state=self._state_out, sensordata=self._sensor_out,
        )
        self.state[:] = self._state_out[:, -1, :]
        foot_z = self._sensor_out[:, -1, self._foot_sensor_z_adr]

        # Mechanical power, sum_j |tau_j * qdot_j| over the eight joints, in watts.
        # Averaged across SUBSTEPS rather than sampled at the last one: torque and joint
        # velocity both swing within a 20 ms control step, and the final substep is not
        # representative of the work done during it. Both arrays are already materialised by
        # the rollout, so this costs one elementwise product.
        _frc = self._sensor_out[:, :, self._actuator_frc_adr]
        _qd = self._state_out[:, :, self._joint_qvel_state_adr]
        self._joint_power = np.abs(_frc * _qd).sum(axis=2).mean(axis=1)

        self._episode_length += 1
        if self.gait_phase:
            self._gait_phase = (self._gait_phase + self.dt * self._gait_hz) % 1.0
        if self.domain_rand_enabled:
            self._maybe_push_robots()
        features = self._decode_state()

        self._update_heading_command(np.radians(features["base_euler"][:, 2]))
        self._update_feet_air_time(foot_z)
        self._compute_reward(features)

        # local y is forward / local x is lateral, per _reward_tracking_lin_vel's axis note
        self.episode_fwd_vel_sum += features["base_lin_vel"][:, 1]
        self._episode_cmd_vx_sum += np.abs(self.commands[:, 0])
        self.episode_lat_vel_sum += features["base_lin_vel"][:, 0]
        self.episode_step_count += 1.0
        self.episode_power_sum += self._joint_power

        # How often the policy commands faster than the servo can physically move -- the
        # sim-to-real gap that is INVISIBLE in every other training metric, because the actuator
        # model rate-limits the target: a policy demanding 900 deg/s and one demanding 100 deg/s
        # produce the same smooth motion in sim. The real servo does not filter, it buzzes.
        #
        # READ IT AS A TREND, NOT AN ABSOLUTE. `self.actions` is the SAMPLED action, so this is
        # dominated by exploration noise -- at action_std 0.53, noise alone gives 80% over the
        # limit. The deployed policy runs the distribution MEAN, which the env never sees. The
        # absolute number comes from a deterministic rollout: tools/mj_deployability.py.
        cmd_rate = np.abs(self.actions - self.last_actions) * self.env_cfg["action_scale"] / self.dt
        self.episode_slew_sat_sum += (cmd_rate > ACTUATOR_SLEW_RAD_S_NOMINAL).mean(axis=1)
        self.episode_cmd_rate_sum += np.median(cmd_rate, axis=1)

        resample = (self._episode_length % self._resample_every) == 0
        if resample.any():
            self._resample_commands(resample)

        euler = features["base_euler"]
        self.reset_buf = (
            (self._episode_length > self.max_episode_length)
            | (np.abs(euler[:, 1]) > self.env_cfg["termination_if_pitch_greater_than"])
            | (np.abs(euler[:, 0]) > self.env_cfg["termination_if_roll_greater_than"])
            | ~np.isfinite(self.state).all(axis=1)
        )
        time_outs = (self._episode_length > self.max_episode_length).astype(np.float32)

        if self.penalty_curriculum:
            # Before _reset_idx zeroes them. Both terminations and timeouts count: a policy
            # that survives to the time limit is exactly the one that can afford penalties.
            ending = (self.reset_buf | (time_outs > 0)) & self._curriculum_env_ready
            if ending.any():
                self._update_curriculum_factor(self._episode_length[ending])

        self._reset_idx(self.reset_buf)
        # _reset_idx rewrites self.state for the envs that terminated, so the cached features must
        # be re-derived before building the observation -- otherwise a freshly-reset env reports
        # its TERMINAL state (tilted gravity, the velocity it fell with) as the first observation
        # of its new episode. bittle_env.py avoids this by having _reset_idx overwrite its own
        # base_quat/dof_pos/... buffers directly; here every feature is derived from self.state,
        # so one re-decode covers all of them.
        #
        # This was not a cosmetic bug: with ~1000 resets per rollout, every one of them fed PPO a
        # transition whose next-observation belonged to a different state. KL per update stayed far
        # above desired_kl, the adaptive schedule drove the learning rate to its 1e-5 floor by
        # iteration 0 and pinned it there, and the run plateaued at reward ~8 with action std
        # drifting UP (1.00 -> 1.04 over 2700 iterations) instead of converging.
        f = self._decode_state()
        self._update_heading_command(np.radians(f["base_euler"][:, 2]))
        self._update_observation()

        self.last_last_actions[:] = self.last_actions
        self.last_actions[:] = self.actions

        self.extras["time_outs"] = torch.from_numpy(time_outs).to(self.device)
        return (
            self.get_observations(),
            torch.from_numpy(self.rew_buf.astype(np.float32)).to(self.device),
            torch.from_numpy(self.reset_buf).to(self.device),
            self.extras,
        )

    def _update_feet_air_time(self, foot_z):
        """Rewards a foot for staying airborne past a threshold before landing again, which gives
        a dense signal toward stepping instead of freezing in the stand pose."""
        # Foot horizontal speed, finite-differenced from the framepos sensors. Only the
        # in-contact part is charged: a foot moving fast through the AIR is a good gait.
        foot_xy = self._sensor_out[:, -1, self._foot_sensor_xy_adr].reshape(self.num_envs, -1, 2)

        # Contact is height above the LOCAL ground, not absolute z. On a flat floor the two are
        # the same and this reduces exactly to the old test; on terrain they are not, and using
        # absolute z would read a foot resting on an 8 mm rise as airborne -- silently breaking
        # feet_air_time, feet_slip, feet_stuck and feet_clearance all at once, in the direction
        # of rewarding a robot for standing on high ground. config.get_cfgs' terrain
        # comment flagged this coupling before terrain existed here; this is the fix.
        foot_h = foot_z - self.terrain_height_env(foot_xy) if self.terrain_enabled else foot_z
        contact = foot_h < FEET_CONTACT_HEIGHT_THRESHOLD

        # -- dragging and clearance, both keyed off the same contact test ------------------
        if self._prev_foot_xy is None:
            self._prev_foot_xy = foot_xy.copy()
        slip = np.linalg.norm(foot_xy - self._prev_foot_xy, axis=-1) / self.dt
        self._prev_foot_xy = foot_xy.copy()
        self._feet_slip = np.sum(np.square(slip) * contact, axis=1)
        # Reward height only up to the target: this asks the foot to clear the ground, not to
        # kick as high as possible, which would just buy back the thrashing gait.
        target = self.env_cfg.get("feet_clearance_target", 0.015)
        self._feet_clearance = np.sum(np.minimum(foot_h, target) * ~contact, axis=1)

        # PER-FOOT stuck timer. The clearance sum above does NOT catch a towed leg: it sums
        # over feet, so two front feet lifting 15 mm at 36% duty outscore four feet lifting
        # 10 mm at 20%, and a degenerate two-legged gait measures BETTER than a healthy one.
        # Summing a REWARD lets good legs pay for dead ones.
        #
        # This asks the question directly: how long has each foot been continuously on the
        # ground? A foot in a normal 1.8 Hz gait resets every stride (~0.55 s cycle, ~0.36 s
        # stance). A towed foot accumulates without bound, so only the dead leg is charged and
        # its healthy neighbours cannot mask it.
        self._feet_ground_time = np.where(contact, self._feet_ground_time + self.dt, 0.0)
        stuck_after = self.env_cfg.get("feet_stuck_after_s", 0.5)
        # COUNT of currently-stuck feet, 0..4 -- not accumulated seconds. Accumulated time
        # grows without bound inside an episode, so the same gait would be charged ~3x more at
        # step 900 than at step 300 and the penalty would be entangled with survival. A count is
        # bounded, reads as "how many legs is it towing", and compares across episode lengths.
        self._feet_stuck = np.sum(self._feet_ground_time > stuck_after, axis=1).astype(np.float64)

        contact_filt = contact | self.last_feet_contact
        first_contact = (self.feet_air_time > 0.0) & contact_filt
        self.feet_air_time += self.dt
        self.feet_air_time_rew = np.sum(
            (self.feet_air_time - self.env_cfg["feet_air_time_threshold"]) * first_contact, axis=1
        )
        # No air-time reward while commanded to stand still, so it doesn't fight a zero command.
        self.feet_air_time_rew *= np.linalg.norm(self.commands[:, :2], axis=1) > 0.05
        self.feet_air_time *= ~contact_filt
        self.last_feet_contact = contact

    def _joint_reversals(self, f):
        """Per-env count of joints that reversed direction this step, while moving.

        The threshold matters: without it, a joint sitting still dithers around zero
        velocity and flips sign constantly, so a policy that FROZE would be charged hardest.
        """
        v = f["dof_vel"]
        moving = (np.abs(v) > REVERSAL_MIN_RAD_S) & (np.abs(self._prev_dof_vel) > REVERSAL_MIN_RAD_S)
        flipped = (np.sign(v) != np.sign(self._prev_dof_vel)) & moving
        self._prev_dof_vel[:] = v
        return flipped.sum(axis=1).astype(np.float64)

    def _compute_reward(self, f):
        cfg, scales = self.reward_cfg, self.reward_scales
        # Bittle's MJCF chassis frame has front/back legs separated along local Y (not X as in the
        # go2 convention this reward was ported from), so local y is the true forward/back axis.
        lin_vel_err = np.sum(
            np.square(self.commands[:, :2] - f["base_lin_vel"][:, [1, 0]]), axis=1
        )
        ang_vel_err = np.square(self.commands[:, 2] - f["base_ang_vel"][:, 2])

        # Separate sigmas. bittle_env.py (and legged_gym) share one tracking_sigma between both
        # terms, which couples them in a way that bites as soon as the linear command changes:
        # tracking_sigma is meant to scale as command^2, so raising the commanded speed forces it
        # up -- and that silently loosens the YAW-RATE penalty by the same factor. Measured on the
        # 0.2 m/s runs, sigma 0.01 -> 0.04 took residual yaw from 0.19 deg/s to 1.41 deg/s, which
        # over a 15s episode curves the path 17% of the distance travelled. The angular command
        # here is always 0, so its sigma should not follow the linear one anywhere.
        # Defaults to tracking_sigma when unset, so existing configs behave exactly as before.
        ang_sigma = cfg.get("tracking_sigma_ang", cfg["tracking_sigma"])
        terms = {
            "tracking_lin_vel": np.exp(-lin_vel_err / cfg["tracking_sigma"]),
            "tracking_ang_vel": np.exp(-ang_vel_err / ang_sigma),
            "lin_vel_z": np.square(f["base_lin_vel"][:, 2]),
            # Body tilt, as the squared horizontal component of projected gravity. This is
            # the one penalty in the set that ONLY attitude feedback can reduce: no
            # feedforward gait can lower its own tilt on ground it cannot sense. Without it
            # the sole consequence of being tilted is termination, which is far away and
            # sparse, so reading gravity pays a policy almost nothing and it learns not to.
            #
            # Squared, not absolute, so small wobble is nearly free and a developing tip is
            # charged sharply -- the regime where recovery is still possible.
            "orientation": np.sum(np.square(f["projected_gravity"][:, :2]), axis=1),
            # DIRECTION REVERSALS of the achieved joint motion. A COUNT, not a magnitude, and
            # that is the point: every other smoothness term charges how FAR or how FAST a
            # joint moves, and none of them reproduce what an operator sees. Jerk is actually
            # ANTI-correlated with perceived smoothness -- one big sweep has enormous jerk at
            # its turnaround and still looks smooth, while many small back-and-forth motions
            # have low jerk and look terrible.
            #
            # Charged only where the joint is moving on both sides of the flip, so dither
            # around a standstill is not counted; otherwise a frozen leg would score worst of
            # all. Uses ACHIEVED dof_vel rather than the action, because the actuator filter
            # sits between the two and the eye watches the joint.
            #
            # KEEP THIS AT SCALE 0.0 AND USE IT AS A METRIC, not a reward. Two properties
            # that make it a good metric make it a bad objective: it is a threshold crossing,
            # so most of what it charges on a sampled rollout is exploration noise rather than
            # policy behaviour; and the threshold is an escape hatch, since a joint that never
            # moves slowly is never charged. Optimising it directly produces bigger, faster
            # sweeps -- the opposite of smooth. tools/mj_deployability.py reports the same
            # quantity deterministically and normalised by cadence, where neither applies.
            "joint_reversal": self._joint_reversals(f),
            "action_rate": np.sum(np.square(self.last_actions - self.actions), axis=1),
            # SECOND difference -- a chatter term, and the reason ACTION_RATE cannot do this
            # job. For a sinusoid the first difference scales as f and the second as f^2, so
            # action_rate is dominated by the GAIT FUNDAMENTAL (0.7-0.8 Hz, ~79% of the action
            # power) and tightening it slows the gait before it touches anything else. This
            # weights 2.5 Hz chatter ~11x harder than the fundamental.
            #
            # action_slew below is not subject to that objection -- it charges only
            # max(0, rate - servo limit), and the gait fundamental peaks at 2*pi*f*A =
            # 59 deg/s against a 137 deg/s limit, so the gait contributes exactly zero. The
            # two are complements, not rivals.
            #
            # What this term is for, measured on the action spectrum of two policies with the
            # SAME action_rate, near-identical cadence and amplitude, one smooth on hardware
            # and one not:
            #
            #     share of action power    <1.5 Hz   1.5-5 Hz   >5 Hz
            #     smooth                     96.0%       3.5%     0.4%
            #     shaky                      79.0%      12.2%     8.8%
            #
            # The smooth one's joint rate is exactly the 2*pi*f*A of its own gait (50
            # predicted, 49 measured); the shaky one measures 109 against a predicted 59. The
            # surplus is entirely above the gait frequency.
            "action_jerk": np.sum(np.square(
                self.actions - 2.0 * self.last_actions + self.last_last_actions), axis=1),
            # Charge the policy for asking the servo to move FASTER THAN IT CAN, in
            # proportion to the excess. action_rate penalises change in general; this
            # penalises exactly the part of that change the hardware cannot deliver, which is
            # what separates a gait that transfers from one that buzzes. A quadratic on the
            # raw delta does not know where the servo's limit is, so tightening it slows the
            # whole gait rather than just its unusable tail. This term is zero for any motion
            # the servo can follow and grows only past that knee.
            # Nominal, not the episode's randomized slew: the deployed robot has one physical
            # servo, and the value baked into the .npz is the nominal.
            "action_slew": np.sum(np.maximum(
                np.abs(self.actions - self.last_actions) * self.env_cfg["action_scale"] / self.dt
                - ACTUATOR_SLEW_RAD_S_NOMINAL, 0.0), axis=1),
            "similar_to_default": np.sum(np.abs(f["dof_pos"] - self._default_dof_pos), axis=1),
            # Height above the LOCAL ground, for the same reason the foot contact test is:
            # measured against absolute z, terrain relief leaks in as apparent tracking error
            # and the term quietly pays the robot to walk in the hollows.
            "base_height": np.square(
                f["base_pos"][:, 2]
                - (self.terrain_height_env(f["base_pos"][:, :2]) if self.terrain_enabled else 0.0)
                - cfg["base_height_target"]
            ),
            "feet_air_time": self.feet_air_time_rew,
            "action_magnitude": np.sum(np.square(self.actions), axis=1),
            # Mechanical power, sum_j |tau_j * qdot_j| in watts, computed in step() from the
            # actuatorfrc sensors. This is the term the energy-reward literature uses to get
            # gait CHANGES rather than smaller versions of one gait: penalising
            # work per unit time makes a slow gait genuinely cheaper than a fast one, which
            # tracking reward alone never does: asked to walk slower, a policy shaped only by
            # tracking lengthens its stride rather than changing gait. 0.0 by default; size it
            # against achieved tracking reward before enabling, per config.py.
            "power": self._joint_power,
            # Horizontal foot speed WHILE IN CONTACT -- i.e. dragging.
            #
            # feet_air_time only PAYS a foot that lifts, so a foot that never lifts scores zero:
            # no bonus, but no cost either. And dragging is nearly free in sim, because a 5 mm
            # sphere on a perfectly flat plane has no texture to catch on. On a real mat
            # (mu 0.44, measured) the towed foot grips, the planted front leg levers against a
            # body that cannot advance, and the reaction drives the robot BACKWARDS. A policy
            # that walks on two legs and tows the other two is the failure this prices.
            "feet_slip": self._feet_slip,
            # ...and a positive pressure to actually pick the foot up, since penalising slip
            # alone can be satisfied by standing still.
            "feet_clearance": self._feet_clearance,
            # Charges only the leg that is actually being towed; see _update_feet_air_time.
            "feet_stuck": self._feet_stuck,
        }
        self.rew_buf[:] = 0.0
        for name in self.reward_names:
            scale = scales[name]
            if self.penalty_curriculum and name in CURRICULUM_TERMS:
                scale *= self._curriculum_factor
            r = terms[name] * scale
            self.rew_buf += r
            self.episode_sums[name] += r

    def _update_curriculum_factor(self, ended_lengths):
        """Track achieved episode length and set the penalty factor from it.

        Feedback, not a schedule: the factor rises only as the policy earns it and FALLS if
        the penalties start costing episodes. That self-correction is the whole point -- a
        fixed ramp cannot notice it has broken the policy, which is how a fresh run ends up
        holding 18/1000 forever.
        """
        batch = float(np.mean(ended_lengths))
        if self._curriculum_ep_len_ema == 0.0:
            self._curriculum_ep_len_ema = batch          # seed, or the EMA crawls up from 0
        else:
            a = CURRICULUM_EMA_ALPHA
            self._curriculum_ep_len_ema += a * (batch - self._curriculum_ep_len_ema)
        span = max(CURRICULUM_EP_LEN_HI - CURRICULUM_EP_LEN_LO, 1e-9)
        target = float(
            np.clip((self._curriculum_ep_len_ema - CURRICULUM_EP_LEN_LO) / span, 0.0, 1.0))
        # Damp only the rise; see CURRICULUM_MAX_RISE_PER_UPDATE.
        rise_cap = self._curriculum_max_rise
        if rise_cap > 0.0 and target > self._curriculum_factor:
            target = min(target, self._curriculum_factor + rise_cap)
        self._curriculum_factor = target

    def _apply_actuator_dynamics(self, target_dof_pos):
        """Rate-limited first-order lag: the model of the real servo's closed-loop response.

        This filter is the ONLY place the servo's measured tau/slew live, so the MJCF position
        actuator downstream must be a near-transparent tracker. MEASURE THAT RATHER THAN
        ASSUMING IT: with kv=0.5/armature=0.01 the MJCF stage alone has a 123 ms 10-90% rise on
        an 8 deg step (kv/kp is a 25 ms viscous lag, and armature under the 0.18 N*m force cap
        limits acceleration to ~15 rad/s^2), so the composed plant settles in ~141 ms against
        the real servo's ~40 ms -- the filter's dynamics counted twice, once here and once in
        the sluggish tracker behind it. A plant 2-3.5x slower than the hardware is as damaging
        as one that is too fast, in the opposite direction: a policy tuned to drag a sluggish
        plant answers with commands that make a snappy real servo tremble. bittle.xml carries
        kv=0.1/armature=0.0002, fitted so filter+actuator reproduce the measured servo curves
        (t63 45 ms vs 40 target, 10-90 75 ms vs 77, no overshoot); see the <position> default
        there. Lower kv is also the safe direction
        numerically: actuator kv integrates EXPLICITLY under the plain Euler integrator this
        model uses (MuJoCo docs recommend implicitfast when kv is significant).

        Continuous form:  y' = clip((x - y) / tau, -slew, +slew)

        A real position servo has two distinct regimes and this reproduces both:
          * small error  -> y' = (x - y)/tau, i.e. pure first-order lag with the measured tau
          * large error  -> y' saturates at exactly the measured slew rate

        The saturation must apply to the INCREMENT, not to the target. Clamping the target
        first and lagging afterwards compounds the two effects, capping the achievable
        velocity at alpha*slew (~45% of the measured value over this tau range) -- the model
        would then be slower than the hardware it is meant to imitate.

        This models the servo's closed-loop response as seen from outside, rather than retuning
        the MJCF actuator's kp: kp also sets tracking stiffness under load, which is a separate
        physical property we did not measure and do not want to distort.
        """
        # Applies ALWAYS, randomized or not. This was previously gated on
        # domain_rand_enabled and returned the target untouched when off -- which meant
        # mj_eval (randomization off by DEFAULT) rendered a robot with INSTANT servos,
        # while the real one runs a 33 ms lag and a rate limit that binds on ~100% of
        # ticks. Comparing that eval against hardware compares two different plants: the
        # sim takes full-amplitude strides where the robot takes small truncated ones.
        # Randomization belongs on the PARAMETERS (tau, slew), not on whether the actuator
        # exists -- pushes and observation noise are disturbances, an actuator is not.
        tau = self._actuator_tau if self.domain_rand_enabled else ACTUATOR_TAU_S_NOMINAL
        slew = self._actuator_slew if self.domain_rand_enabled else ACTUATOR_SLEW_RAD_S_NOMINAL
        # EXACT discrete first-order lag: alpha = 1 - exp(-dt/tau), not the Euler dt/tau.
        # Euler is only valid for dt << tau, but dt/tau spans 0.22..1.0 over the randomized
        # range, where it runs the filter markedly too fast -- tau would no longer mean the
        # time constant the hardware measured. The exact form also stays in (0, 1) by
        # construction, so a tau shorter than one control step degrades to "instant" instead
        # of overshooting into instability.
        alpha = np.reshape(1.0 - np.exp(-self.dt / tau), (-1, 1))
        max_delta = np.reshape(np.asarray(slew) * self.dt, (-1, 1))
        delta = alpha * (target_dof_pos - self._servo_target)
        np.clip(delta, -max_delta, max_delta, out=delta)
        self._servo_target += delta
        return self._servo_target.copy()

    def _update_servo_estimate(self, target_dof_pos):
        """deploy/obs_builder.ServoModel, step for step.

        Same filter as _apply_actuator_dynamics, deliberately at the NOMINAL tau/slew rather
        than this episode's randomized ones: those nominals are what export_policy_mj.py bakes
        into the .npz, so they are the only values the real host can use. Where the two filters
        disagree IS the deployed observer's error, which is why it is not modelled as noise.
        """
        alpha = 1.0 - math.exp(-self.dt / ACTUATOR_TAU_S_NOMINAL)
        max_delta = ACTUATOR_SLEW_RAD_S_NOMINAL * self.dt
        self._servo_estimate_prev[:] = self._servo_estimate
        delta = alpha * (target_dof_pos - self._servo_estimate)
        np.clip(delta, -max_delta, max_delta, out=delta)
        self._servo_estimate += delta

    def _update_observation(self):
        f = self._features
        ang_vel = f["base_ang_vel"]
        gravity = f["projected_gravity"]
        if self._imu_buf is not None:
            # Delay the TRUE values, then let noise and mounting tilt apply below: the sensor
            # measures the state it saw k steps ago, and its errors are added at read time.
            self._imu_head = (self._imu_head + 1) % self._imu_buf.shape[1]
            self._imu_buf[:, self._imu_head, :3] = ang_vel
            self._imu_buf[:, self._imu_head, 3:] = gravity
            take = (self._imu_head - self._imu_delay) % self._imu_buf.shape[1]
            got = self._imu_buf[np.arange(self.num_envs), take]
            ang_vel, gravity = got[:, :3], got[:, 3:]
        if self.observe_servo_estimate:
            # What the hardware actually has. The joint channels are OPEN LOOP on the real
            # robot: obs_builder integrates its own commanded targets and never reads a joint,
            # so these carry no information about contact, load or disturbance -- a leg blocked
            # by the floor reports the same angle as a leg swinging in air.
            #
            # Reading the simulator's true dof_pos instead gave the policy a proprioceptive
            # sense the robot does not possess, and it learned to lean on it. Measured on the
            # trained policy, feeding the estimate it would actually receive: gait cadence
            # 1.17 -> 4.65 Hz and forward speed +0.079 -> -0.020 m/s. That is both hardware
            # reports at once -- a tempo far faster than sim, and walking on the spot.
            #
            # No motor offset term appears here, and that is not an omission: the estimate is
            # built in commanded space, so the offset is already absent from it. That is
            # precisely the blindness MOTOR_OFFSET_RAD_RANGE is meant to impose.
            dof_pos_err = self._servo_estimate - self._default_dof_pos
            dof_vel = (self._servo_estimate - self._servo_estimate_prev) / self.dt
        else:
            # Legacy path: dof_pos as a real sensor reading, kept for A/B against
            # observe_servo_estimate.
            #
            # Subtract the calibration offset: the robot does not know its own servo zeros are
            # wrong, so neither may the policy. Observing the true position would hand it
            # exactly the information the real robot lacks.
            dof_pos_err = f["dof_pos"] - self._motor_offset - self._default_dof_pos
            dof_vel = f["dof_vel"]

        if self.domain_rand_enabled:
            rng, n = self.rng, self.num_envs
            ang_vel = ang_vel + rng.normal(0.0, OBS_NOISE_ANG_VEL, (n, 3))
            # Constant-per-episode IMU mounting tilt, applied BEFORE the zero-mean noise: a
            # small-angle rotation of the gravity vector about the body x (roll) and y (pitch)
            # axes. Unlike the noise term the policy cannot average this away, which is the
            # point -- on hardware it is a fixed offset for the whole run.
            r, p = self._imu_tilt[:, 0], self._imu_tilt[:, 1]
            gx, gy, gz = gravity[:, 0], gravity[:, 1], gravity[:, 2]
            gravity = np.stack([gx + p * gz, gy - r * gz, gz - p * gx + r * gy], axis=-1)
            gravity = gravity + rng.normal(0.0, OBS_NOISE_GRAVITY, (n, 3))
            gravity = gravity / (np.linalg.norm(gravity, axis=-1, keepdims=True) + 1e-6)
            if not self.observe_servo_estimate:
                # Only meaningful when these are sensor readings; see OBS_NOISE_DOF_POS.
                dof_pos_err = dof_pos_err + rng.normal(0.0, OBS_NOISE_DOF_POS,
                                                       (n, self.num_actions))
                dof_vel = dof_vel + rng.normal(0.0, OBS_NOISE_DOF_VEL, (n, self.num_actions))

        parts = [
            ang_vel * self.obs_scales["ang_vel"],
            gravity,
            self.commands * self._commands_scale,
            dof_pos_err * self.obs_scales["dof_pos"],
            dof_vel * self.obs_scales["dof_vel"],
            self.actions,
        ]
        if self.gait_phase:
            # sin/cos of a circular quantity, so there is no discontinuity at the wrap for
            # the policy to learn around, plus the normalised cadence it was told to walk at.
            ph = 2.0 * np.pi * self._gait_phase
            parts += [np.sin(ph)[:, None], np.cos(ph)[:, None],
                      ((self._gait_hz - pio.GAIT_HZ_MID) / pio.GAIT_HZ_HALF)[:, None]]
        np.concatenate(
            tuple(parts),
            axis=-1,
            out=self.obs_buf,
            casting="same_kind",
        )
        # A FRESH tensor every step -- never reuse one buffer, and never hand out a view of
        # self.obs_buf (torch.from_numpy aliases it).
        #
        # rsl_rl's PPO.act does `self.transition.observations = obs`, keeping a *reference* while
        # the runner then calls env.step(); only afterwards does process_env_step ->
        # RolloutStorage.add_transition copy it. So if step() writes the new observation into the
        # same memory, the stored observation for step t is silently overwritten with the one from
        # t+1, while its action, value and log-prob still belong to t. PPO then evaluates
        # pi_new(a_t | s_t+1) / pi_old(a_t | s_t): the ratio is meaningless, KL per update is
        # enormous, and the ADAPTIVE_KL schedule drives the learning rate to its 1e-5 floor on the
        # very first iteration and pins it there. Symptom: reward creeps to a standing-still local
        # optimum and stalls, value loss never falls, action std never converges.
        #
        # bittle_env.py never hit this because torch.concatenate allocates a new tensor each step.
        # An extra (num_envs, 33) float32 allocation per step is ~67KB at 512 envs -- irrelevant
        # next to the 2.8ms physics step.
        self._obs_tensor = torch.from_numpy(self.obs_buf).to(self.device, copy=True)

        if self._obs_history is not None:
            # Snapshot BEFORE rolling the current frame in, so the group handed out now holds
            # only PAST observations -- slot 0 is one step ago. Rolling first would put the
            # current frame at slot 0, duplicating it in both "policy" and "history" and
            # wasting num_obs inputs on a copy of something the actor already has.
            self._history_tensor = torch.from_numpy(
                self._obs_history.reshape(self.num_envs, -1)).to(self.device, copy=True)
            self._obs_history[:, 1:] = self._obs_history[:, :-1]
            self._obs_history[:, 0] = self.obs_buf

        self._update_privileged()

    # Where base_lin_vel sits inside the privileged vector. Exported so the state estimator's
    # supervised target cannot drift from this layout: _update_privileged builds the vector by
    # concatenating parts in order, and inserting anything ahead of base_lin_vel would silently
    # retarget the estimator onto dof_pos with nothing to complain. tests/test_estimator.py
    # asserts this slice really is base_lin_vel by comparing against the decoded feature.
    PRIVILEGED_LIN_VEL_SLICE = slice(0, 3)

    def _update_privileged(self):
        """Critic-only observation. Never reaches the actor, so never reaches the robot."""
        f = self._features
        # True minus estimated joint position. This is the single most informative privileged
        # channel: it is exactly the signal the actor lost when the observation became honest,
        # and it is large precisely when a foot is loaded or blocked.
        tracking_err = f["dof_pos"] - self._motor_offset - self._servo_estimate
        parts = (
            f["base_lin_vel"] * self.obs_scales["lin_vel"],
            (f["dof_pos"] - self._motor_offset - self._default_dof_pos) * self.obs_scales["dof_pos"],
            f["dof_vel"] * self.obs_scales["dof_vel"],
            tracking_err * self.obs_scales["dof_pos"],
            self.last_feet_contact.astype(np.float64),
            self._foot_friction[:, None],
            self._actuator_tau[:, None],
            self._actuator_slew[:, None],
            self._motor_offset,
            self._imu_tilt,
            # This episode's incline and the ground height under the base. Both are latent
            # variables the actor can only estimate -- from gravity and from footfalls -- so
            # handing them to the critic removes that uncertainty from the value function
            # without removing the reason the actor has to look.
            self._slope_deg[:, None],
            (self.terrain_height_env(f["base_pos"][:, :2])[:, None]
             if self.terrain_enabled else np.zeros((self.num_envs, 1))),
        )
        np.concatenate(parts, axis=-1, out=self.privileged_buf, casting="same_kind")
        self._privileged_tensor = torch.from_numpy(self.privileged_buf).to(self.device, copy=True)

    # ------------------------------------------------------ resets & sampling
    def _resample_commands(self, mask):
        n = int(mask.sum())
        self.commands[mask] = self.rng.uniform(self._commands_low, self._commands_high, (n, 3))
        if self.heading_command:
            # In heading mode the yaw-rate command is not sampled -- it is COMPUTED every step
            # from heading error by _update_heading_command. Only the target is drawn here.
            self._heading_target[mask] = self.rng.uniform(-np.pi, np.pi, n)

    def _update_heading_command(self, yaw_rad):
        """Close an outer loop on HEADING by writing the yaw-rate command each step.

        WHY: projected_gravity is exactly yaw-invariant -- spin the robot 180 deg and the
        observation is bit-identical -- so the actor has no absolute heading reference at all.
        It sees a yaw RATE, is commanded a yaw RATE, and tracking_ang_vel scores a yaw RATE.
        Nothing ever asks whether it is pointed the right way, so a disturbance that rotates the
        robot produces an error that is both invisible and unpenalised, and it walks off at the
        new angle forever. That is the dominant term in where the robot ends up: a heading error
        STEERS, so displacement grows quadratically with time, while a bounded lateral velocity
        error only displaces linearly.

        The fix needs no new observation. The policy already sees commands[2] and is already
        rewarded for tracking it; folding heading error into that command closes the loop
        outside the network. deploy runs the identical P-controller against the IMU's fused
        yaw, which the firmware already prints and whose gyro bias we already calibrate.

        Runs EVERY step, unlike command resampling: it is feedback, not a sample. Only the
        target heading is resampled on the usual schedule.
        """
        if not self.heading_command:
            return
        # Wrap to (-pi, pi] so the robot turns the short way round; without this a target just
        # across the +-pi seam commands a full rotation the wrong direction.
        err = np.arctan2(np.sin(self._heading_target - yaw_rad),
                         np.cos(self._heading_target - yaw_rad))
        self.commands[:, 2] = np.clip(self.heading_kp * err,
                                      -self.heading_max_yaw_rate, self.heading_max_yaw_rate)

    def _maybe_push_robots(self):
        """Instantaneous base velocity kick, on a per-env timer resampled every time it fires.

        Magnitudes are scaled to this robot's own task speed (0.1-0.22 m/s), not a generic
        quadruped constant: the original +-0.15/+-0.10 m/s were up to 1.5x the target walking
        speed itself and hit several times per 20s episode before a gait could settle.
        """
        do_push = self._episode_length >= self.next_push_step
        if not do_push.any():
            return
        idx = np.nonzero(do_push)[0]
        qvel0 = 1 + self._base_model.nq  # first linear dof of the freejoint within the state array
        dv_xy = float(self.env_cfg.get("push_dv_xy", PUSH_DV_XY_RANGE[1]))
        dv_z = float(self.env_cfg.get("push_dv_z", PUSH_DV_Z_RANGE[1]))
        self.state[idx, qvel0 : qvel0 + 2] += self.rng.uniform(-dv_xy, dv_xy, (idx.size, 2))
        self.state[idx, qvel0 + 2] += self.rng.uniform(-dv_z, dv_z, idx.size)
        # Angular kick on roll and pitch (qvel0+3, +4). Yaw (+5) is deliberately left
        # alone: heading error is already driven by the heading controller, and a yaw kick
        # would fight it rather than demand a postural correction. Default 0.0 so every
        # cfgs.pkl written before this existed reproduces exactly.
        dw_rp = float(self.env_cfg.get("push_dw_rp", 0.0))
        if dw_rp > 0.0:
            self.state[idx, qvel0 + 3 : qvel0 + 5] += self.rng.uniform(
                -dw_rp, dw_rp, (idx.size, 2))
        interval_s = self.rng.uniform(*PUSH_INTERVAL_S_RANGE, idx.size)
        self.next_push_step[idx] = self._episode_length[idx] + np.round(interval_s / self.dt)

    def _randomize_init_tilt(self, idx):
        """Start the episode at a random roll/pitch instead of perfectly upright.

        See BASE_INIT_TILT_DEG for why this exists. Quaternion is built from roll and pitch
        only -- yaw is left at the init value because the heading controller already owns
        it, and randomizing yaw would just hand the heading term a random error to chew on
        rather than a postural one.

        Default 0.0, so a cfgs.pkl written before this key existed reproduces exactly.
        """
        tilt = float(self.env_cfg.get("base_init_tilt_deg", 0.0))
        if tilt <= 0.0:
            return
        n = idx.size
        rp = np.radians(self.rng.uniform(-tilt, tilt, (n, 2)))
        cr, sr = np.cos(rp[:, 0] / 2), np.sin(rp[:, 0] / 2)
        cp, sp = np.cos(rp[:, 1] / 2), np.sin(rp[:, 1] / 2)
        # qpos[3:7] lives at state[4:8]: state = [time, qpos(nq), qvel(nv)].
        self.state[idx, 4] = cr * cp
        self.state[idx, 5] = sr * cp
        self.state[idx, 6] = cr * sp
        self.state[idx, 7] = -sr * sp

    def _respawn_on_terrain(self, idx):
        """Start the episode somewhere else on the field, at the right height for that spot.

        All the terrain variety a run sees comes from here: the heightfield itself is identical
        in every env, so without this the whole population would walk the same 2 m of ground
        every episode and the policy could memorise it instead of reacting to it.

        The height correction is not optional. _init_state puts the base at a fixed z tuned for
        a flat floor; drop that onto a 8 mm rise and the robot starts interpenetrating the
        ground, which MuJoCo resolves as a launch.
        """
        if not self.terrain_enabled or idx.size == 0:
            return
        # FULLPHYSICS is [time, qpos, qvel, ...], so the base xyz is state[1:4], NOT state[0:3]
        # -- index 0 is simulation time. Writing there would have silently rewritten each
        # episode's clock with a spawn coordinate.
        xy = self.rng.uniform(*TERRAIN_SPAWN_XY_RANGE, (idx.size, 2))
        self.state[idx, 1:3] = xy
        self.state[idx, 3] = self._init_state[3] + self.terrain_height_env(xy, idx)

    def _update_terrain_curriculum(self, idx):
        """Move each finishing environment up or down one difficulty rung.

        Rudin's rule, on our one axis of difficulty (heightfield amplitude). The distance
        comparison is the whole thing:

            covered  = integral of achieved forward velocity over the episode
            required = mean commanded forward speed * a FULL episode

        `required` deliberately uses the full episode length rather than the steps actually
        taken. That is what makes falling a demotion: a robot that terminates at step 200 of
        1000 has covered a fifth of what was asked, so the ratio fails on its own without a
        separate termination branch. Scoring it against only the steps it survived would let a
        robot that fell early look like a competent walker.

        Integrated forward velocity rather than straight-line displacement, because in heading
        mode the commanded direction rotates during the episode -- displacement would score a
        robot that correctly walked a curve as though it had barely moved.
        """
        if not self.terrain_curriculum or idx.size == 0:
            return
        steps = np.maximum(self.episode_step_count[idx], 1.0)
        covered = self.episode_fwd_vel_sum[idx] * self.dt
        required = (self._episode_cmd_vx_sum[idx] / steps) * self.max_episode_length * self.dt
        # A zero-speed command has no distance to fail; leave those envs where they are.
        ok = required > 1e-6
        ratio = np.where(ok, covered / np.maximum(required, 1e-9), np.nan)

        level = self._terrain_level[idx]
        level = np.where(ok & (ratio > self._terrain_promote_frac), level + 1, level)
        level = np.where(ok & (ratio < self._terrain_demote_frac), level - 1, level)

        # Rudin's anti-forgetting loop: an env that graduates off the top goes back to a random
        # rung instead of piling up at the ceiling. Without it the whole population eventually
        # sits on the hardest terrain and stops seeing the ground it will actually be deployed
        # on, and the easy-terrain gait quietly rots.
        graduated = level >= self._terrain_levels
        if graduated.any():
            level = np.where(graduated,
                             self.rng.integers(0, self._terrain_levels, size=level.shape),
                             level)
        level = np.clip(level, 0, self._terrain_levels - 1)
        self._terrain_level[idx] = level

        # Write the new vertical scale into each env's own model. hfield_size[hid] is
        # [radius_x, radius_y, elevation, base]; only elevation moves.
        elev = self._terrain_elev * (level + 1).astype(np.float64) / self._terrain_levels
        self._terrain_elev_per_env[idx] = elev
        for k, env_i in enumerate(idx):
            self.models[env_i].hfield_size[self._terrain_hfield_id, 2] = elev[k]

    def _resample_slope(self, idx):
        """Tilt gravity to put this episode's robot on an incline of random grade and aspect.

        Sampled per EPISODE rather than per env slot: with a fixed-per-slot slope the population
        would only ever see num_envs distinct grades, and any correlation between slot and slope
        is a shortcut worth denying. Costs one 3-vector write per reset.
        """
        if not self.slope_enabled or idx.size == 0:
            return
        g = float(np.linalg.norm(self._base_model.opt.gravity))
        # Uniform in ANGLE, not in the tilt vector: the latter concentrates samples near flat.
        lo, hi = self.slope_deg_range
        hi = np.full(idx.size, float(hi))
        if self.slope_friction_margin > 0.0:
            # CAP THE GRADE BY THIS ROBOT'S OWN FOOT FRICTION. Sampling slope and friction
            # independently puts some episodes on ground the robot physically cannot walk: it
            # slides whenever tan(slope) > mu. Those episodes contribute a termination and no
            # signal -- they do not teach recovery, they teach that nothing works.
            #
            # The margin is well below 1.0 because merely NOT SLIDING is the wrong bar. Measured
            # downhill drift reaches 0.086 m/s at 6-8 deg against a 0.063 m/s walk: the slope
            # wins even where friction nominally holds. Halving the critical grade leaves the
            # robot able to make progress, not just to stand.
            crit = np.degrees(np.arctan(
                self.slope_friction_margin * self._foot_friction[idx]))
            hi = np.minimum(hi, crit)
        tilt = np.radians(self.rng.uniform(float(lo), np.maximum(hi, float(lo)), idx.size))
        aspect = self.rng.uniform(0.0, 2 * np.pi, idx.size)
        gx = g * np.sin(tilt) * np.cos(aspect)
        gy = g * np.sin(tilt) * np.sin(aspect)
        gz = -g * np.cos(tilt)
        for k, env_i in enumerate(idx):
            self.models[env_i].opt.gravity[:] = (gx[k], gy[k], gz[k])
        self._slope_deg[idx] = np.degrees(tilt)

    def _reset_idx(self, mask=None):
        if mask is None:
            mask = np.ones(self.num_envs, dtype=bool)
        n_reset = int(mask.sum())
        if n_reset == 0:
            # The Genesis env paid ~3.2ms/step running this unconditionally, including a full
            # set_qpos and ~10 reductions building extras["episode"] even when nothing reset.
            return
        idx = np.nonzero(mask)[0]

        # Difficulty first: _respawn_on_terrain places the base at the ground height of the
        # terrain this env is ABOUT to walk, so the level has to be settled before the spawn.
        # It reads the episode accumulators, which are zeroed further down in this same call.
        self._update_terrain_curriculum(idx)

        self.state[idx] = self._init_state
        self._respawn_on_terrain(idx)
        self._resample_slope(idx)
        self._randomize_init_tilt(idx)
        self.actions[idx] = 0.0
        self.last_actions[idx] = 0.0
        self.last_last_actions[idx] = 0.0
        self.feet_air_time[idx] = 0.0
        self.last_feet_contact[idx] = False
        # Stale foot positions across a reset would read as one enormous slip spike.
        if self._prev_foot_xy is not None:
            self._prev_foot_xy[idx] = 0.0
        self._feet_ground_time[idx] = 0.0
        if self._action_buf is not None:
            # A fresh episode must not execute the previous one's actions out of the ring.
            self._action_buf[idx] = 0.0
        if self._imu_buf is not None:
            # ...nor observe the attitude the previous episode ended in. Seed upright and still.
            self._imu_buf[idx] = 0.0
            self._imu_buf[idx, :, 5] = -1.0        # gravity = (0,0,-1)
        if self.gait_phase:
            self._gait_hz[idx] = self.rng.uniform(*pio.GAIT_HZ_RANGE, idx.size)
            # Random start phase: the policy must work from ANY point in the cycle, because
            # deploy/ starts its clock at 0 on a robot that is standing still, and any timing
            # slip during a run shifts it. Training from phase 0 every episode would make
            # that shift out-of-distribution.
            self._gait_phase[idx] = self.rng.uniform(0.0, 1.0, idx.size)
        self._episode_length[idx] = 0
        # From here this env's episode length is genuine; see _curriculum_env_ready.
        self._curriculum_env_ready[idx] = True
        self._resample_commands(mask)

        # The lagged servo starts the episode already at the reset pose, matching the physical
        # reset: leaving it at its terminal value would make the first steps of a new episode
        # chase a target inherited from the previous one.
        self._servo_target[idx] = self._default_dof_pos
        # Both observer states too, matching control_loop.py, which seeds ServoModel at the home
        # pose and dof_pos_prev at the same value -- so the first tick's dof_vel is zero rather
        # than a spike out of whatever the previous episode ended on.
        self._servo_estimate[idx] = self._default_dof_pos
        self._servo_estimate_prev[idx] = self._default_dof_pos
        # Clear the history, or a fresh episode's first frames carry the previous episode's
        # motion -- including, for a robot that just fell over, the attitude it fell with.
        if self._obs_history is not None:
            self._obs_history[idx] = 0.0

        if self.domain_rand_enabled:
            self.extra_latency_prob[idx] = self.rng.uniform(*EXTRA_LATENCY_PROB_RANGE, n_reset)
            if self._action_delay is not None:
                lo, hi = self.env_cfg["action_latency_steps_range"]
                self._action_delay[idx] = self.rng.integers(int(lo), int(hi) + 1, n_reset)
            if self._imu_delay is not None:
                lo, hi = self.env_cfg["imu_delay_steps_range"]
                self._imu_delay[idx] = self.rng.integers(int(lo), int(hi) + 1, n_reset)
            interval_s = self.rng.uniform(*PUSH_INTERVAL_S_RANGE, n_reset)
            self.next_push_step[idx] = np.round(interval_s / self.dt)
            self._actuator_tau[idx] = self.rng.uniform(*ACTUATOR_TAU_S_RANGE, n_reset)
            self._actuator_slew[idx] = self.rng.uniform(*ACTUATOR_SLEW_RAD_S_RANGE, n_reset)
            self._imu_tilt[idx] = self.rng.uniform(*IMU_TILT_RAD_RANGE, (n_reset, 2))
            self._motor_offset[idx] = self.rng.uniform(
                *MOTOR_OFFSET_RAD_RANGE, (n_reset, self.num_actions))

        episode = {}
        for name in self.reward_names:
            episode["rew_" + name] = float(
                self.episode_sums[name][idx].mean() / self.env_cfg["episode_length_s"]
            )
            self.episode_sums[name][idx] = 0.0
        # Time-averaged body velocities in m/s, divided by actual step count rather than
        # episode_length_s so an early termination does not deflate them. tracking_lin_vel is a
        # lossy proxy: exp(-err^2/sigma) is averaged per-step, so intra-stride ripple drags it
        # down, and its scale changes entirely if tracking_sigma is retuned.
        steps = np.maximum(self.episode_step_count[idx], 1.0)
        if self.penalty_curriculum:
            # Log the controller, or it is invisible: a curriculum that silently stayed at 0
            # (penalties never applied) and one that jumped to 1 immediately look identical in
            # every other metric until the gait is already wrong.
            episode["curriculum_factor"] = float(self._curriculum_factor)
            episode["curriculum_ep_len_ema"] = float(self._curriculum_ep_len_ema)
        episode["fwd_vel_mps"] = float((self.episode_fwd_vel_sum[idx] / steps).mean())
        if self.terrain_curriculum:
            # The population's mean rung, and the amplitude it corresponds to. Log both: a
            # curriculum pinned at 0 and one pinned at the top are the same number of lines of
            # code away from correct and look identical in every other metric.
            episode["terrain_level"] = float(self._terrain_level[idx].mean())
            episode["terrain_amplitude_m"] = float(self._terrain_elev_per_env[idx].mean())
        # Fraction of commanded joint rates exceeding the servo's slew limit, and the median
        # commanded rate in deg/s. Above ~15% saturation the gait will not transfer.
        episode["slew_saturated_frac"] = float((self.episode_slew_sat_sum[idx] / steps).mean())
        episode["power_w"] = float((self.episode_power_sum[idx] / steps).mean())
        episode["cmd_rate_deg_s"] = float(
            np.degrees(self.episode_cmd_rate_sum[idx] / steps).mean())
        episode["lat_vel_mps"] = float((self.episode_lat_vel_sum[idx] / steps).mean())
        self.episode_fwd_vel_sum[idx] = 0.0
        self._episode_cmd_vx_sum[idx] = 0.0
        self.episode_slew_sat_sum[idx] = 0.0
        self.episode_power_sum[idx] = 0.0
        self.episode_cmd_rate_sum[idx] = 0.0
        self.episode_lat_vel_sum[idx] = 0.0
        self.episode_step_count[idx] = 0.0
        self.extras["episode"] = episode

    def reset(self):
        self._reset_idx(None)
        # That reset is construction, not an episode ending, and it happens BEFORE rsl_rl
        # randomises episode_length_buf. Clearing the flag here is what makes the guard work:
        # an env counts only from the first reset it takes while actually being stepped.
        self._curriculum_env_ready[:] = False
        self._decode_state()
        self._update_observation()
        return self.get_observations()

    def get_observations(self):
        """Observation GROUPS, wired to models by train_cfg's obs_groups.

        "policy"     -- the 33 the robot can actually produce. The actor's whole input.
        "history"    -- the previous obs_history_len frames of that same vector, present only
                        when obs_history_len > 0. Deployable: deploy keeps a ring buffer.
        "privileged" -- critic only, never deployed. See _update_privileged.

        Both extra groups are always emitted; which models consume them is a training-config
        decision, so a run can A/B them without touching this file.
        """
        groups = {"policy": self._obs_tensor, "privileged": self._privileged_tensor}
        if self._obs_history is not None:
            groups["history"] = self._history_tensor
        if self._estimate_tensor is not None:
            # Written by PPOWithEstimator.act before the actor forward; see _alloc_buffers.
            groups["estimate"] = self._estimate_tensor
        return TensorDict(groups, batch_size=[self.num_envs])

    def close(self):
        self._rollout.close()
