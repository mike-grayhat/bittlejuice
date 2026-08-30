"""The penalty curriculum is a feedback controller, not a schedule.

It exists because the four smoothness/foot penalties are measured to make from-scratch
training impossible: with them on from iteration 0 a fresh policy holds ~18/1000 episode
length indefinitely; with them off it reaches 1001 by iteration 10 (config.get_cfgs has
the paired curves). The curriculum lets one run do both jobs.

The property that matters, and that a fixed ramp cannot have: the factor must FALL when
episode length falls. A timetable cannot notice it has broken the policy.
"""

import numpy as np
import pytest

import mj_vec_env as MVE
from config import get_cfgs
from mj_vec_env import MjVecEnv


def _env(**over):
    env_cfg, obs_cfg, reward_cfg, command_cfg = get_cfgs()
    env_cfg.update({"terrain_enabled": False, "domain_rand_enabled": False,
                    "base_init_tilt_deg": 0.0})
    env_cfg.update(over)
    return MjVecEnv(num_envs=4, env_cfg=env_cfg, obs_cfg=obs_cfg, reward_cfg=reward_cfg,
                    command_cfg=command_cfg, device="cpu", num_threads=1, seed=0)


def test_off_by_default_and_factor_is_unity():
    """Every config written before this feature must reproduce exactly."""
    env = _env()
    assert env.penalty_curriculum is False
    assert env._curriculum_factor == 1.0
    env.close()


def test_starts_at_zero_so_a_fresh_policy_pays_nothing():
    env = _env(penalty_curriculum=True)
    assert env._curriculum_factor == 0.0
    env.close()


def test_factor_tracks_episode_length_across_the_band():
    env = _env(penalty_curriculum=True)
    for ep_len, expected in [(MVE.CURRICULUM_EP_LEN_LO - 100, 0.0),   # cannot walk yet
                             (MVE.CURRICULUM_EP_LEN_HI + 100, 1.0)]:  # surviving full episodes
        env._curriculum_ep_len_ema = 0.0
        for _ in range(600):                    # let the EMA converge
            env._update_curriculum_factor(np.full(4, ep_len))
        assert env._curriculum_factor == pytest.approx(expected, abs=1e-3), ep_len

    env._curriculum_ep_len_ema = 0.0
    mid = 0.5 * (MVE.CURRICULUM_EP_LEN_LO + MVE.CURRICULUM_EP_LEN_HI)
    for _ in range(600):
        env._update_curriculum_factor(np.full(4, mid))
    assert env._curriculum_factor == pytest.approx(0.5, abs=0.05)
    env.close()


def test_factor_backs_off_when_episodes_get_shorter():
    """THE reason this is feedback and not a ramp. Penalties that start costing episodes
    must relax on their own -- that is the collapse a fixed schedule walks into."""
    env = _env(penalty_curriculum=True)
    for _ in range(600):
        env._update_curriculum_factor(np.full(4, 1000.0))
    assert env._curriculum_factor == pytest.approx(1.0)

    for _ in range(600):                        # the policy starts falling
        env._update_curriculum_factor(np.full(4, 100.0))
    assert env._curriculum_factor < 0.05, "factor must relax when the policy is failing"
    env.close()


def test_only_the_blocking_terms_are_scaled():
    """action_rate, action_magnitude, similar_to_default, base_height and lin_vel_z are NOT
    staged: a fresh policy bootstraps fine with them at full strength.

    `power` and `joint_reversal` ARE staged, for a different reason than the other four:
    both are minimised by a policy that does not move, so applying them before locomotion
    exists makes standing still optimal. The other four block bootstrap by pricing the
    flailing that precedes a gait; these two block it by pricing motion itself."""
    assert set(MVE.CURRICULUM_TERMS) == {
        "action_jerk", "action_slew", "feet_slip", "feet_stuck", "power", "joint_reversal"}
    env_cfg, _, reward_cfg, _ = get_cfgs()
    for name in MVE.CURRICULUM_TERMS:
        assert name in reward_cfg["reward_scales"], name


def test_reward_actually_changes_with_the_factor():
    """Guards the wiring, not the arithmetic: an unused factor would pass every test above."""
    env = _env(penalty_curriculum=True, terrain_enabled=False)
    env.reward_scales["action_jerk"] = -1.0
    env._curriculum_factor = 0.0
    env.step(np.full((4, env.num_actions), 0.5, dtype=np.float64))
    quiet = env.rew_buf.copy()

    env2 = _env(penalty_curriculum=True, terrain_enabled=False)
    env2.reward_scales["action_jerk"] = -1.0
    env2._curriculum_factor = 1.0
    env2.step(np.full((4, env2.num_actions), 0.5, dtype=np.float64))
    charged = env2.rew_buf.copy()

    assert np.all(charged <= quiet + 1e-12)
    assert not np.allclose(charged, quiet), "factor 0 vs 1 must change the reward"
    env.close(); env2.close()


def test_artificial_first_episodes_are_ignored():
    """rsl_rl's OnPolicyRunner.learn(init_at_random_ep_len=True) overwrites
    episode_length_buf with values up to max_episode_length to decorrelate envs. The first
    episode each env "finishes" is therefore an artifact -- it reports up to 1001 steps
    having actually been stepped a handful of times.

    Feeding those to the EMA seeds it at ~1000 and pins the factor at 1.0 from iteration 0,
    silently disabling the curriculum. That is exactly what happened on the first run using
    it: the log showed curriculum_factor 0.9167 at iteration 0 while mean episode length was
    12.86.
    """
    env = _env(penalty_curriculum=True)
    assert not env._curriculum_env_ready.any(), "no env has been reset by us yet"

    # rsl_rl does this before the first step.
    env.episode_length_buf = np.full(4, env.max_episode_length, dtype=np.int64)
    env.step(np.zeros((4, env.num_actions)))

    assert env._curriculum_factor == 0.0, "artificial timeouts must not raise the factor"
    assert env._curriculum_ep_len_ema == 0.0, "EMA must still be unseeded"

    # After we have reset them once, their lengths are real and do count.
    assert env._curriculum_env_ready.all()
    env._update_curriculum_factor(np.full(4, 950.0))
    assert env._curriculum_ep_len_ema == pytest.approx(950.0)
    env.close()


def test_the_rise_can_be_rate_limited_and_the_fall_cannot():
    """The controller is proportional gain with lag, so undamped it limit-cycles: measured
    factor std 0.133 with no power/symmetry term and 0.249 with both, swinging across most
    of the range every few tens of iterations.

    Only the RISE is capped. The fall must stay instant, because being able to back off the
    moment penalties start costing episodes is the entire reason this is feedback rather
    than a schedule."""
    env = _env(penalty_curriculum=True, curriculum_max_rise=0.01)
    for _ in range(600):                       # drive the EMA to the top
        env._update_curriculum_factor(np.full(4, 1000.0))
    assert env._curriculum_factor == pytest.approx(1.0, abs=1e-6), "should still reach 1.0"

    env2 = _env(penalty_curriculum=True, curriculum_max_rise=0.01)
    for _ in range(20):
        env2._update_curriculum_factor(np.full(4, 1000.0))
    assert env2._curriculum_factor <= 0.20 + 1e-9, "rise must be capped at 0.01 per call"

    # ...and the fall is not capped: one collapsed batch is not enough to move a long EMA,
    # but the factor must track it down without a rate limit slowing it.
    before = env._curriculum_factor
    for _ in range(600):
        env._update_curriculum_factor(np.full(4, 100.0))
    assert env._curriculum_factor < 0.05, f"fall must not be rate-limited (was {before})"
    env.close()
    env2.close()


def test_damping_is_off_by_default():
    """The undamped controller is the default; those configs must reproduce."""
    env_cfg, _, _, _ = get_cfgs()
    assert env_cfg["curriculum_max_rise"] == 0.0
    assert MVE.CURRICULUM_MAX_RISE_PER_UPDATE == 0.0
