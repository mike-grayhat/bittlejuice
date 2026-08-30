"""The joint_reversal term: a COUNT of direction changes, not a magnitude.

Measured across five policies, excess reversals is the only metric that ranks an
operator's smoothness verdicts correctly. These tests pin what it charges -- including
the one thing it deliberately does NOT charge, because that omission is why it has to
ride the penalty curriculum, and why it works as a metric but not as a reward.
"""

import math

import numpy as np
import pytest

import mj_vec_env as MVE
from config import get_cfgs
from mj_vec_env import MjVecEnv

FAST = 4.0 * MVE.REVERSAL_MIN_RAD_S          # unambiguously moving
DITHER = 0.4 * MVE.REVERSAL_MIN_RAD_S        # unambiguously not


def _env():
    env_cfg, obs_cfg, reward_cfg, command_cfg = get_cfgs()
    env_cfg.update({"terrain_enabled": False, "domain_rand_enabled": False,
                    "base_init_tilt_deg": 0.0})
    return MjVecEnv(num_envs=4, env_cfg=env_cfg, obs_cfg=obs_cfg, reward_cfg=reward_cfg,
                    command_cfg=command_cfg, device="cpu", num_threads=1, seed=0)


def _counts(env, *velocity_frames):
    """Feed successive dof_vel frames; return the count from each after the first."""
    out = []
    for i, v in enumerate(velocity_frames):
        vel = np.tile(np.asarray(v, dtype=np.float64), (env.num_envs, 1))
        c = env._joint_reversals({"dof_vel": vel})
        if i:                                 # the first frame only seeds _prev_dof_vel
            out.append(c)
    return out


def test_a_flip_while_moving_is_charged_once_per_joint():
    env = _env()
    n = env.num_actions
    (c,) = _counts(env, [FAST] * n, [-FAST] * n)
    assert np.all(c == n), f"all {n} joints reversed, got {c}"
    env.close()


def test_holding_a_direction_costs_nothing():
    env = _env()
    n = env.num_actions
    c1, c2 = _counts(env, [FAST] * n, [FAST] * n, [FAST] * n)
    assert np.all(c1 == 0) and np.all(c2 == 0)
    env.close()


def test_dither_around_standstill_is_not_charged():
    """Without the threshold a joint sitting still flips sign on numerical noise, so the
    policy charged hardest would be the one that FROZE. That is the opposite of the
    behaviour this term exists to price."""
    env = _env()
    n = env.num_actions
    counts = _counts(env, [DITHER] * n, [-DITHER] * n, [DITHER] * n, [-DITHER] * n)
    assert all(np.all(c == 0) for c in counts), counts
    env.close()


def test_one_side_below_threshold_is_not_charged():
    """Both sides of the flip must be moving -- a joint decelerating through zero and
    stopping has not reversed, it has arrived."""
    env = _env()
    n = env.num_actions
    (c,) = _counts(env, [FAST] * n, [-DITHER] * n)
    assert np.all(c == 0)
    env.close()


def test_only_the_reversing_joints_are_counted():
    env = _env()
    n = env.num_actions
    a = [FAST] * n
    b = [-FAST if i < 3 else FAST for i in range(n)]
    (c,) = _counts(env, a, b)
    assert np.all(c == 3), c
    env.close()


def test_a_frozen_robot_pays_zero_which_is_why_it_rides_the_curriculum():
    """THE loophole, pinned deliberately. A policy that does not move satisfies this
    penalty perfectly, exactly as an energy penalty does. That is survivable only because
    the term is staged behind achieved episode length: it arrives after locomotion, not
    before it. If it is ever removed from CURRICULUM_TERMS, this is what breaks."""
    env = _env()
    n = env.num_actions
    counts = _counts(env, [0.0] * n, [0.0] * n, [0.0] * n)
    assert all(np.all(c == 0) for c in counts)
    assert "joint_reversal" in MVE.CURRICULUM_TERMS
    env.close()


def test_scale_times_magnitude_is_the_logged_value():
    """The sizing identity the run configs are built on: logged rew_X = scale * mean
    magnitude of X. Pinned here because reconstructing it from per-episode sums instead
    gets the units wrong; see config.py."""
    env = _env()
    n, scale = env.num_actions, -0.30
    env.reward_scales = dict(env.reward_scales, joint_reversal=scale)
    env.penalty_curriculum = False
    env.episode_sums["joint_reversal"][:] = 0.0

    # Drive a known number of reversals through the real reward path.
    env._prev_dof_vel[:] = FAST
    steps, flips = 6, 0
    for i in range(steps):
        vel = np.full((env.num_envs, n), FAST if i % 2 else -FAST)
        env._compute_reward({**env._features, "dof_vel": vel})
        flips += 1
    mean_magnitude = flips * n / steps
    expected = scale * mean_magnitude * steps          # episode sum over `steps` steps
    assert env.episode_sums["joint_reversal"] == pytest.approx(expected)
    env.close()
