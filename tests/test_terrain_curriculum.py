"""Per-environment terrain difficulty, following Rudin et al. (arXiv:2109.11978).

The rule: promote on covering a fraction of the commanded distance, demote below a lower
one, and send environments that graduate off the top back to a random rung so the population
keeps practising easy ground.

The thresholds are NOT Rudin's 0.80/0.50. Those assume a robot that tracks its command; this
one delivers 43-76% of a 0.10 m/s command under training conditions, so the 0.80 gate sat
above everything it ever achieves and the ladder stalls near the bottom -- easier ground
than its own fixed-amplitude baseline. Recalibrated to 0.60/0.35, and made overridable,
because they are properties of the robot's achievable speed rather than of the method.

The property most worth pinning is the one that is invisible in training curves: `required`
is measured against a FULL episode, not against the steps actually survived. That is what
makes falling a demotion without a separate termination branch -- and if it ever regresses to
per-step normalisation, a robot that fell at step 200 scores like a competent walker and the
curriculum ratchets difficulty upward on failing policies.
"""

import numpy as np
import pytest

import mj_vec_env as MVE
from config import get_cfgs
from mj_vec_env import MjVecEnv


def _env(n=8, **over):
    env_cfg, obs_cfg, reward_cfg, command_cfg = get_cfgs()
    env_cfg.update({"terrain_enabled": True, "terrain_amplitude": 0.020,
                    "domain_rand_enabled": False, "base_init_tilt_deg": 0.0,
                    "terrain_curriculum": True, "terrain_curriculum_levels": 10})
    env_cfg.update(over)
    command_cfg = dict(command_cfg)
    command_cfg["lin_vel_x_range"] = [0.10, 0.10]
    return MjVecEnv(num_envs=n, env_cfg=env_cfg, obs_cfg=obs_cfg, reward_cfg=reward_cfg,
                    command_cfg=command_cfg, device="cpu", num_threads=1, seed=0)


def _fake_episode(env, idx, covered_fraction, steps=None):
    """Pretend these envs finished having covered `covered_fraction` of their command."""
    steps = steps if steps is not None else env.max_episode_length
    cmd = 0.10
    env.episode_step_count[idx] = steps
    env._episode_cmd_vx_sum[idx] = cmd * steps
    required_full = cmd * env.max_episode_length * env.dt
    env.episode_fwd_vel_sum[idx] = covered_fraction * required_full / env.dt


def test_off_by_default_and_all_envs_share_one_amplitude():
    env_cfg, obs_cfg, reward_cfg, command_cfg = get_cfgs()
    assert env_cfg["terrain_curriculum"] is False
    env = _env(terrain_curriculum=False)
    assert np.allclose(env._terrain_elev_per_env, env._terrain_elev)
    env.close()


def test_everyone_starts_on_the_easiest_rung():
    env = _env()
    assert np.all(env._terrain_level == 0)
    env.close()


def test_covering_the_distance_promotes():
    env = _env()
    idx = np.arange(env.num_envs)
    _fake_episode(env, idx, covered_fraction=0.95)
    env._update_terrain_curriculum(idx)
    assert np.all(env._terrain_level == 1)
    env.close()


def test_falling_short_demotes():
    env = _env()
    idx = np.arange(env.num_envs)
    env._terrain_level[:] = 5
    _fake_episode(env, idx, covered_fraction=0.30)
    env._update_terrain_curriculum(idx)
    assert np.all(env._terrain_level == 4)
    env.close()


def test_the_middle_band_holds_position():
    """Between the two gates nothing moves -- the rules must not overlap or leave a gap."""
    env = _env()
    idx = np.arange(env.num_envs)
    env._terrain_level[:] = 3
    _fake_episode(env, idx, covered_fraction=0.45)
    env._update_terrain_curriculum(idx)
    assert np.all(env._terrain_level == 3)
    env.close()


def test_an_early_termination_demotes_even_though_it_walked_well_while_upright():
    """THE test. A robot that fell at step 200 was tracking its command perfectly until it
    did. Normalising by surviving steps would score it 1.0 and PROMOTE it."""
    env = _env()
    idx = np.arange(env.num_envs)
    env._terrain_level[:] = 4
    steps = env.max_episode_length // 5
    cmd = 0.10
    env.episode_step_count[idx] = steps
    env._episode_cmd_vx_sum[idx] = cmd * steps
    env.episode_fwd_vel_sum[idx] = cmd * steps        # perfect tracking, for 1/5 of an episode
    env._update_terrain_curriculum(idx)
    assert np.all(env._terrain_level == 3), "a fall must not read as success"
    env.close()


def test_graduating_off_the_top_returns_to_a_random_rung():
    """Rudin's anti-forgetting loop. Without it the population piles up at the ceiling."""
    env = _env(n=64)
    idx = np.arange(env.num_envs)
    env._terrain_level[:] = env._terrain_levels - 1
    _fake_episode(env, idx, covered_fraction=0.95)
    env._update_terrain_curriculum(idx)
    assert np.all(env._terrain_level < env._terrain_levels)
    assert np.all(env._terrain_level >= 0)
    assert len(np.unique(env._terrain_level)) > 1, "should scatter, not collapse to one rung"
    env.close()


def test_level_never_leaves_its_range():
    env = _env(n=32)
    idx = np.arange(env.num_envs)
    rng = np.random.default_rng(0)
    for _ in range(200):
        _fake_episode(env, idx, covered_fraction=float(rng.uniform(0.0, 1.2)))
        env._update_terrain_curriculum(idx)
        assert np.all(env._terrain_level >= 0)
        assert np.all(env._terrain_level < env._terrain_levels)
    env.close()


def test_the_level_actually_reaches_the_physics():
    """A curriculum that updates an integer and never touches MuJoCo is a no-op that logs
    beautifully. Each env owns its hfield_size, so the write must land per env."""
    env = _env(n=4)
    idx = np.arange(env.num_envs)
    env._terrain_level[:] = np.array([0, 3, 6, 9])
    _fake_episode(env, idx, covered_fraction=0.45)          # hold position
    env._update_terrain_curriculum(idx)
    hid = env._terrain_hfield_id
    got = np.array([float(env.models[i].hfield_size[hid, 2]) for i in range(4)])
    want = env._terrain_elev * (np.array([0, 3, 6, 9]) + 1) / env._terrain_levels
    np.testing.assert_allclose(got, want, rtol=1e-12)
    np.testing.assert_allclose(env._terrain_elev_per_env, want, rtol=1e-12)
    # ...and the height lookup must follow the physics, or spawn heights drift from the ground.
    xy = np.zeros((4, 2))
    nominal = env.terrain_height(xy)
    np.testing.assert_allclose(env.terrain_height_env(xy), nominal * want / env._terrain_elev,
                               rtol=1e-12)
    env.close()


def test_zero_speed_commands_do_not_move_anyone():
    """No distance was asked for, so there is nothing to pass or fail."""
    env = _env()
    idx = np.arange(env.num_envs)
    env._terrain_level[:] = 5
    env.episode_step_count[idx] = env.max_episode_length
    env._episode_cmd_vx_sum[idx] = 0.0
    env.episode_fwd_vel_sum[idx] = 0.0
    env._update_terrain_curriculum(idx)
    assert np.all(env._terrain_level == 5)
    env.close()


def test_metrics_are_logged():
    env = _env(episode_length_s=0.1)
    env.reset()
    for _ in range(12):
        env.step(np.zeros((env.num_envs, env.num_actions)))
    assert "terrain_level" in env.extras["episode"]
    assert "terrain_amplitude_m" in env.extras["episode"]
    env.close()


def test_the_thresholds_are_reachable_and_overridable():
    """The failure in one test: a promote gate above what the robot can achieve makes the
    whole curriculum inert, and it looks like a feature that did not help."""
    assert MVE.TERRAIN_PROMOTE_FRAC < 0.76, (
        "promote gate must sit below the ~76% of command this robot achieves in training, "
        "or no environment is ever promoted")
    assert MVE.TERRAIN_DEMOTE_FRAC < MVE.TERRAIN_PROMOTE_FRAC

    env = _env(terrain_promote_frac=0.9, terrain_demote_frac=0.8)
    idx = np.arange(env.num_envs)
    env._terrain_level[:] = 4
    _fake_episode(env, idx, covered_fraction=0.85)     # between the overridden gates
    env._update_terrain_curriculum(idx)
    assert np.all(env._terrain_level == 4)
    _fake_episode(env, idx, covered_fraction=0.95)
    env._update_terrain_curriculum(idx)
    assert np.all(env._terrain_level == 5)
    env.close()
