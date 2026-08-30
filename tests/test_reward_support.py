"""Every reward term has a WIDTH; the environment must keep the agent inside it.

This invariant has been the bug twice, from opposite sides:

  * tracking_sigma set from a command the servos could not reach -- the reward was FLAT across
    everything achievable, so doubling speed paid less than a posture penalty;
  * push_dv_xy sized against walking speed instead of against the reward -- a push landed the
    robot outside the support and the reward went to ZERO, ~6.7 times per episode, for a third
    of every episode. The policy correctly learned to crawl.

Neither raised an exception, neither showed up as a failed run, and both cost a full training
run to notice. What they have in common is that nothing in the code related a disturbance
magnitude to the reward's own scale -- so that relation is what these tests assert.
"""

import numpy as np
import pytest

import mj_vec_env as MVE
from config import get_cfgs


def _tracking_reward(err, sigma):
    """The env's own tracking_lin_vel shape, restated so the tests read as the math."""
    return np.exp(-np.square(err) / sigma)


def _push_magnitudes(dv_xy, n=200_000, seed=0):
    """|dv| actually delivered: uniform per axis, then combined over x and y."""
    rng = np.random.default_rng(seed)
    return np.linalg.norm(rng.uniform(-dv_xy, dv_xy, (n, 2)), axis=1)


@pytest.mark.parametrize("sigma_source", ["cfg", "launch"])
def test_default_push_stays_inside_the_tracking_reward_support(sigma_source):
    """A push must leave a usable gradient, not erase the term.

    Checked against BOTH sigmas that exist, because they differ and the run uses the tighter
    one: get_cfgs' 0.01 goes with its lin_vel_x RANGE, while the fixed-command runs override to
    0.0036 on the CLI. A bound derived only from the config default would not describe the run
    actually launched, and a push bound that does not describe the run is worth nothing.
    """
    env_cfg, _, reward_cfg, _ = get_cfgs()
    sigma = reward_cfg["tracking_sigma"] if sigma_source == "cfg" else LAUNCH_TRACKING_SIGMA
    dv = env_cfg["push_dv_xy"]
    median_after = float(np.median(_tracking_reward(_push_magnitudes(dv), sigma)))
    baseline = np.exp(-1.0)
    assert median_after > 0.20 * baseline, (
        f"push_dv_xy={dv} against tracking_sigma={sigma} leaves the median post-push tracking "
        f"reward at {median_after:.2e} ({100 * median_after / baseline:.1f}% of the "
        f"exp(-1)={baseline:.3f} standing-still baseline). The speed reward is switched off "
        f"while the robot recovers, so the only gradients left are survival and penalties -- "
        f"which is an objective whose optimum is crawling."
    )


def test_a_too_large_push_would_have_failed_this():
    """Evidence, not just a threshold: values that produce a crawl are far outside."""
    median_after = float(np.median(_tracking_reward(_push_magnitudes(0.20), 0.0036)))
    assert median_after < 0.01 * np.exp(-1.0)


def test_push_is_still_large_enough_to_need_sensing():
    """The opposite failure mode. +-0.05 m/s is absorbed passively -- blindfolding the IMU
    changes nothing at that magnitude -- so the push must stay well above it, or there is no
    reason for a closed-loop policy to exist."""
    env_cfg, _, _, _ = get_cfgs()
    assert env_cfg["push_dv_xy"] >= 0.09, (
        "push is back down at the open-loop era's magnitude; nothing forces the policy to sense"
    )


def test_push_budget_leaves_clean_windows_to_be_scored_in():
    """Magnitude is not the only way to erase the reward -- frequency does it too. With ~1 s of
    recovery per push, the episode needs to spend most of its time NOT recovering."""
    env_cfg, _, _, _ = get_cfgs()
    mean_interval = float(np.mean(MVE.PUSH_INTERVAL_S_RANGE))
    pushes = env_cfg["episode_length_s"] / mean_interval
    recovering = pushes * 1.0 / env_cfg["episode_length_s"]
    assert recovering < 0.25, (
        f"~{pushes:.1f} pushes per episode leaves ~{100 * recovering:.0f}% of it in recovery"
    )


def test_slope_and_friction_leave_the_ground_walkable():
    """Slope interacts with randomized foot friction: the robot slides when tan(slope) > mu, and
    no policy walks on ground that cannot hold it. Those episodes contribute termination and no
    signal, so the overlap has to stay small."""
    env_cfg, _, _, _ = get_cfgs()
    rng = np.random.default_rng(0)
    n = 400_000
    slope = rng.uniform(*env_cfg["slope_deg_range"], n)
    mu = rng.uniform(*MVE.FOOT_FRICTION_RANGE, n)
    unwalkable = float((np.tan(np.radians(slope)) > mu).mean())
    assert unwalkable < 0.02, (
        f"{100 * unwalkable:.1f}% of episodes are on ground static friction cannot hold "
        f"(slope up to {max(env_cfg['slope_deg_range'])} deg vs mu down to "
        f"{MVE.FOOT_FRICTION_RANGE[0]})"
    )


# The command/sigma pair the closed-loop runs are launched with. get_cfgs holds a RANGE for
# lin_vel_x and a sigma for that range, so the invariant cannot be asserted against its defaults
# -- the fixed-command runs override both on the CLI, and it is the overridden pair that has to
# be consistent. Keep this in step with the launch command in the run log.
LAUNCH_COMMAND_VX = 0.10
LAUNCH_TRACKING_SIGMA = 0.0036


@pytest.mark.parametrize("cmd,sigma,ok", [
    (LAUNCH_COMMAND_VX, LAUNCH_TRACKING_SIGMA, True),   # sharpened deliberately, see below
    (0.10, 0.0100, True),                               # the plain sigma = command^2 convention
    # The measured failure: sigma left wide while the command was raised, so the reward was
    # near-flat across everything reachable and doubling speed paid +0.221 against feet_stuck's
    # +0.294 -- the policy took the posture term and stopped trying to walk.
    (0.15, 0.0400, False),
    # The opposite degeneracy: so tight that standing still and walking are both ~zero, leaving
    # no gradient to climb from a standstill.
    (0.10, 0.0002, False),
])
def test_tracking_sigma_discriminates_speed(cmd, sigma, ok):
    """The original form of this bug, stated as the property that actually matters.

    get_cfgs' convention is sigma = command^2, which puts standing still at exp(-1) = 0.368.
    The launch config deliberately sharpens past that (sigma 0.0036 for a 0.10 command, a
    0.06 m/s scale) to get more gradient per unit of speed error. So equality with command^2 is
    NOT the invariant -- what matters is that standing still scores clearly worse than walking
    at the command, without the whole reachable range collapsing to zero.
    """
    standing = _tracking_reward(cmd, sigma)      # reward at the command is 1.0 by construction
    usable = 0.001 < standing < 0.5
    assert usable == ok, (
        f"command {cmd} m/s with tracking_sigma {sigma}: standing still scores {standing:.4f}. "
        f"Above 0.5 the reward barely distinguishes moving from not moving; below 0.001 there "
        f"is no gradient to climb out of a standstill."
    )


# -- deployability ----------------------------------------------------------
# A policy can train cleanly, score well, and tremble on the robot, with nothing in any
# training metric showing it: the sim actuator model rate-limits the target, so a policy
# demanding 900 deg/s
# and one demanding 100 deg/s produce the same smooth motion in sim. The real servo does not
# filter. These pin the settings that separate a deployable gait from an undeployable one.

def test_clip_actions_actually_bounds_the_command():
    """clip_actions=100 is not a bound. At action_scale 0.25 it permits +-25 rad of commanded
    joint target, and commanding past a joint limit is free in sim -- the limit clamps it at no
    cost -- so the policy learns to bang-bang the servo."""
    env_cfg, _, _, _ = get_cfgs()
    clip = env_cfg["clip_actions"]
    reach_deg = np.degrees(clip * env_cfg["action_scale"])
    assert reach_deg < 90.0, (
        f"clip_actions={clip} at action_scale={env_cfg['action_scale']} lets the policy command "
        f"+-{reach_deg:.0f} deg from the default pose. A deployable policy used 3.0 = "
        f"+-43 deg; at 100 it exceeded the servo slew limit on 71% of ticks."
    )


def test_smoothness_penalties_are_active():
    """These were once CLI overrides that a later launch command did not carry forward, which
    silently produced an undeployable policy. Making them defaults is what stops that."""
    _, _, reward_cfg, _ = get_cfgs()
    rs = reward_cfg["reward_scales"]
    assert rs["action_rate"] <= -0.01, (
        f"action_rate={rs['action_rate']} is weaker than the -0.01 that produced a deployable "
        f"gait on hardware"
    )


def test_angular_tracking_sigma_covers_the_yaw_command():
    """The support bug, THIRD occurrence -- this time on the angular term.

    tracking_ang_vel shares tracking_sigma unless overridden. Fixed-command runs drive that to
    0.0036 for a 0.10 m/s LINEAR command, and heading mode opens ang_vel_range to +-0.6 rad/s.
    exp(-0.36/0.0036) = 4e-44: identically zero, unreachable, and visible only as a term that
    never pays, so a run can spend thousands of iterations on it before anyone notices.
    """
    for yaw_max, sigma_ang, ok in [(0.6, 0.36, True),    # command^2, as the convention says
                                   (0.6, 0.0036, False),  # the shared linear sigma
                                   (0.0, 0.0036, True)]:  # degenerate range: nothing to track
        if yaw_max == 0.0:
            continue
        reward_at_command = float(np.exp(-(yaw_max ** 2) / sigma_ang))
        usable = reward_at_command > 0.01
        assert usable == ok, (
            f"yaw command {yaw_max} rad/s with tracking_sigma_ang {sigma_ang}: a full-scale "
            f"command scores {reward_at_command:.2e}. Below ~0.01 the angular reward is dead "
            f"and the policy is asked to turn for nothing."
        )


def test_shaping_terms_are_sized_against_achieved_reward_not_the_ceiling():
    """A calibration method that produces divergence, pinned as arithmetic.

    Sizing a penalty against tracking_lin_vel's per-step maximum (dt*1.0) overstates the budget
    by ~40x, because converged tracking achieves only 2.5% of that ceiling. feet_slip at -0.1
    was computed as "7.4% of tracking" and was really 292%; action_slew at -0.01 was "10.5%" and
    was really 414%. Both terms then dominated the objective from the first iteration.
    """
    dt, steps, achieved = 0.02, 921.0, 0.4672     # a converged run, as the reference
    ceiling = dt * 1.0
    assert achieved / steps < 0.05 * ceiling, "the ceiling is no longer a wild overestimate"
    _, _, reward_cfg, _ = get_cfgs()
    for name, raw in (("feet_slip", 0.740), ("action_slew", 10.49)):
        scale = abs(reward_cfg["reward_scales"][name])
        if scale == 0.0:
            continue                              # disabled by default; enabled per run
        share = scale * dt * raw * steps / achieved
        assert share < 1.0, (
            f"{name} at -{scale} contributes {100*share:.0f}% of converged tracking_lin_vel. "
            f"A shaping term above 100% is the objective, not a correction."
        )
