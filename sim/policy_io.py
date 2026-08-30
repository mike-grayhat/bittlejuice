"""Single source of truth for the Bittle policy's I/O contract.

Every stage of the pipeline -- mj_vec_env.py for training, mj_eval.py for the viewer,
export_policy_mj.py for the export, and deploy/'s obs_builder.py on the robot -- must
agree on joint order, default pose, observation scaling, and action scaling. Keeping
these constants in one place is what makes it possible to prove the numpy inference on
the Raspberry Pi produces the same actions as the torch policy that was trained.

Joint order matches bittle.xml's depth-first body order (and the Genesis pipeline's
joint_names, so checkpoints from that era carry over one-to-one; that pipeline was
deleted once mj_vec_env superseded it).
"""

import os

# The MJCF lives in model/ alongside its meshes (bittle.xml's meshdir="assets" is relative
# to the XML itself). Resolved from this file so it does not depend on the caller's cwd --
# four modules used to build this path independently and one of them went stale.
XML_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "model", "bittle.xml")

JOINT_NAMES = [
    "left-back-shoulder-joint",
    "left-back-knee-joint",
    "left-front-shoulder-joint",
    "left-front-knee-joint",
    "right-back-shoulder-joint",
    "right-back-knee-joint",
    "right-front-shoulder-joint",
    "right-front-knee-joint",
]

FEET_LINK_NAMES = [
    "left-back-knee-link",
    "left-front-knee-link",
    "right-back-knee-link",
    "right-front-knee-link",
]

SHOULDER_LINK_NAMES = [
    "left-back-shoulder-link",
    "left-front-shoulder-link",
    "right-back-shoulder-link",
    "right-front-shoulder-link",
]

# bittle.xml's <geom name="..." class="footL"/"footR"> foot-contact spheres, in the same
# per-leg order as FEET_LINK_NAMES.
FOOT_GEOM_NAMES = [
    "left_back_foot",
    "left_front_foot",
    "right_back_foot",
    "right_front_foot",
]

NUM_ACTIONS = len(JOINT_NAMES)

# [rad], matching bittle.xml's "home" keyframe (symmetric stand pose).
DEFAULT_JOINT_ANGLES = {name: 0.56 for name in JOINT_NAMES}
DEFAULT_DOF_POS = [DEFAULT_JOINT_ANGLES[name] for name in JOINT_NAMES]

# Sim-only reset pose; matches bittle.xml's "home" keyframe. Not used by deploy/ (the real
# robot has no notion of base height). Derived from the geometry with the foot spheres at the
# true toe -- see bittle.xml's keyframe comment.
BASE_INIT_POS = [0.0, 0.0, 0.0941]
BASE_INIT_QUAT = [1.0, 0.0, 0.0, 0.0]  # (w, x, y, z)

# Control frequency: real robot's control loop runs at 50Hz.
CONTROL_DT = 0.02
ACTION_SCALE = 0.25

OBS_SCALES = {
    "lin_vel": 2.0,
    "ang_vel": 0.25,
    "dof_pos": 1.0,
    "dof_vel": 0.05,
}

NUM_COMMANDS = 3  # (lin_vel_x, lin_vel_y, ang_vel_yaw)
COMMANDS_SCALE = [OBS_SCALES["lin_vel"], OBS_SCALES["lin_vel"], OBS_SCALES["ang_vel"]]

# obs = [base_ang_vel*ang_vel_scale (3), projected_gravity (3), commands*commands_scale (3),
#        (dof_pos-default_dof_pos)*dof_pos_scale (8), dof_vel*dof_vel_scale (8), last_action (8)]
NUM_OBS = 3 + 3 + NUM_COMMANDS + NUM_ACTIONS + NUM_ACTIONS + NUM_ACTIONS  # 33

# -- Gait phase (optional; env_cfg["gait_phase"]) -----------------------------------------
# Three extra channels APPENDED to the 33 above: sin(2*pi*phase), cos(2*pi*phase), and the
# commanded cadence normalised to [-1, 1]. Appended rather than inserted so the existing 33
# keep their indices and every dump, diff tool and channel-name list stays valid.
#
# WHY sin/cos rather than the raw phase: phase is circular, and a raw ramp has a
# discontinuity at the wrap that the policy would have to learn around. sin/cos is smooth
# across it and gives the same information.
#
# WHY the frequency is an OBSERVATION and not a constant: measured gait fundamentals across
# trained policies span 0.81-1.25 Hz and track terrain difficulty -- on rough ground a policy
# chooses a cadence up to 56% faster than on flat. A hardcoded clock would fight the policy
# exactly where it matters most.
GAIT_OBS = 3
NUM_OBS_WITH_GAIT = NUM_OBS + GAIT_OBS  # 36

# -- Left/right mirror map ----------------------------------------------------------------
# For the symmetry loss (Yu, Turk, Liu 2018; Abdolhosseini et al. 2019). A symmetric policy
# satisfies pi(mirror(s)) == mirror(pi(s)): show it the mirrored world and it should produce
# the mirrored action.
#
# THE SIGN CONVENTION IS ALREADY DONE FOR US. bittle.xml gives every right-side joint
# axis="0 0 -1" against the left's "0 0 1", and all eight entries of DEFAULT_DOF_POS are
# +0.56. So the model is pre-mirrored: a symmetric pose has NUMERICALLY EQUAL left and right
# angles, and mirroring the joint vector is a pure permutation with no sign flips. Had the
# axes matched instead, every one of these would need a negation and the loss would train
# happily against a wrong target.
#
# JOINT_NAMES order is [lb_sh, lb_kn, lf_sh, lf_kn, rb_sh, rb_kn, rf_sh, rf_kn], so mirroring
# swaps the two halves.
MIRROR_JOINT_PERM = [4, 5, 6, 7, 0, 1, 2, 3]

# The chassis frame has front/back along local Y and lateral along local X (see
# _compute_reward's axis note), so the plane of symmetry is x = 0 and mirroring negates x.
#
# Angular velocity is a PSEUDOVECTOR, and this is the part that is easy to get backwards: a
# true vector reflected in x = 0 flips only its x component, but a pseudovector flips the
# OTHER TWO. Roll rate (about x) survives; pitch and yaw rate inverT.
MIRROR_ANG_VEL_SIGNS = [1.0, -1.0, -1.0]     # pseudovector about (x, y, z)
MIRROR_GRAVITY_SIGNS = [-1.0, 1.0, 1.0]      # true vector
# commands are [forward, lateral, yaw_rate]: forward survives, lateral is a true-vector x
# component, yaw rate is a pseudovector z component.
MIRROR_COMMAND_SIGNS = [1.0, -1.0, -1.0]
# The estimator predicts base linear velocity in the body frame -- a true vector, but stored
# as (x=lateral, y=forward, z=up), so it is the x component that flips.
MIRROR_LIN_VEL_SIGNS = [-1.0, 1.0, 1.0]


def mirror_obs(obs):
    """Reflect one observation frame (..., NUM_OBS) through the robot's sagittal plane.

    Accepts numpy arrays or torch tensors; the operations are shared by both. Gait channels,
    if present, are passed through unchanged -- a symmetric gait is half a cycle out of phase
    on the mirrored side, so the honest handling is a phase shift, and --gait-phase is off in
    every run that matters. Assert rather than silently mirror it wrong.
    """
    n = obs.shape[-1]
    assert n == NUM_OBS, (
        f"mirror_obs expects {NUM_OBS} channels, got {n}. Gait-phase observations need a "
        f"half-cycle phase shift on the mirrored side, which is not implemented.")
    out = obs.copy() if hasattr(obs, "copy") else obs.clone()
    for i, sgn in enumerate(MIRROR_ANG_VEL_SIGNS):
        out[..., i] = obs[..., i] * sgn
    for i, sgn in enumerate(MIRROR_GRAVITY_SIGNS):
        out[..., 3 + i] = obs[..., 3 + i] * sgn
    for i, sgn in enumerate(MIRROR_COMMAND_SIGNS):
        out[..., 6 + i] = obs[..., 6 + i] * sgn
    perm = MIRROR_JOINT_PERM
    for base in (9, 9 + NUM_ACTIONS, 9 + 2 * NUM_ACTIONS):    # dof_pos, dof_vel, last_action
        for j, src in enumerate(perm):
            out[..., base + j] = obs[..., base + src]
    return out


def mirror_obs_groups(groups):
    """Mirror a whole rsl_rl observation dict, group by group.

    Only the groups the ACTOR reads are mirrored -- "policy", "history" and "estimate".
    "privileged" is passed through untouched: the critic never sees a mirrored observation
    in the symmetry loss, and silently mirroring 46 privileged floats (several of which,
    like foot contacts, have their own permutation) would be work that nothing checks.

    The history is a flat concatenation of whole frames, so it reshapes, mirrors and flattens
    back. Getting that reshape wrong scrambles channels across frame boundaries and still
    produces a finite loss.
    """
    out = {}
    for key, value in groups.items():
        if key == "policy":
            out[key] = mirror_obs(value)
        elif key == "history":
            n, width = value.shape
            frames = width // NUM_OBS
            out[key] = mirror_obs(value.reshape(n, frames, NUM_OBS)).reshape(n, width)
        elif key == "estimate":
            mirrored = value.clone() if hasattr(value, "clone") else value.copy()
            for i, sgn in enumerate(MIRROR_LIN_VEL_SIGNS):
                mirrored[..., i] = value[..., i] * sgn
            out[key] = mirrored
        else:
            out[key] = value
    return out


def mirror_action(action):
    """Reflect an action vector (..., NUM_ACTIONS). Pure permutation; see MIRROR_JOINT_PERM."""
    out = action.copy() if hasattr(action, "copy") else action.clone()
    for j, src in enumerate(MIRROR_JOINT_PERM):
        out[..., j] = action[..., src]
    return out

# Cadence range, in Hz. The ceiling is set by the SERVO, not the 50 Hz control loop: for a
# sinusoidal joint trajectory peak rate is A*2*pi*f, so at the ~34.5 deg peak-to-peak both
# current policies use, the fundamental ALONE reaches the measured 137 deg/s slew limit at
# 1.26 Hz. A policy running at 1.25 Hz saturates ~15% of joint-ticks with a perfectly clean
# gait -- its own fundamental eats the whole budget. 1.15 leaves headroom for correction.
GAIT_HZ_RANGE = (0.60, 1.15)
GAIT_HZ_MID = 0.5 * (GAIT_HZ_RANGE[0] + GAIT_HZ_RANGE[1])
GAIT_HZ_HALF = 0.5 * (GAIT_HZ_RANGE[1] - GAIT_HZ_RANGE[0])


def gait_obs(phase, hz):
    """The three gait channels, given phase in [0,1) and cadence in Hz.

    One function, imported by BOTH mj_vec_env and deploy/obs_builder, so the two cannot
    drift. The previous cross-module contract that lived as prose in a comment (the servo
    observer's phase) went stale and cost 16 channels of silent disagreement.
    """
    import math
    return [math.sin(2.0 * math.pi * phase),
            math.cos(2.0 * math.pi * phase),
            (hz - GAIT_HZ_MID) / GAIT_HZ_HALF]

BASE_HEIGHT_TARGET = 0.0941  # tracks BASE_INIT_POS[2]; reward scale is -50, so keep them equal
TERMINATION_ROLL_DEG = 10.0
TERMINATION_PITCH_DEG = 10.0

ACTOR_HIDDEN_DIMS = [512, 256, 128]
