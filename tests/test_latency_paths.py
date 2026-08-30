"""The deep-latency ring buffer must reduce to the legacy 1-step path exactly.

Motivation: the deep path was added to model a hardware latency that later turned out not to
matter. It stays for robustness randomization, which means it will be enabled in runs whose
results get compared against older ones -- so it has to be provably the same code at depth 1,
not merely similar. A silent off-by-one here shifts every action by 20 ms relative to the
baseline it is being compared with.
"""

import numpy as np
import pytest
import torch

import mj_vec_env as MVE
from config import get_cfgs
from mj_vec_env import MjVecEnv


def _env(monkeypatch_zero_extra, **over):
    env_cfg, obs_cfg, reward_cfg, command_cfg = get_cfgs()
    env_cfg["terrain_enabled"] = False
    env_cfg["domain_rand_enabled"] = False
    env_cfg.update(over)
    return MjVecEnv(num_envs=4, env_cfg=env_cfg, obs_cfg=obs_cfg, reward_cfg=reward_cfg,
                    command_cfg=command_cfg, device="cpu", num_threads=1, seed=0)


@pytest.fixture
def no_extra_latency(monkeypatch):
    # The legacy path adds a 2-step delay with per-episode probability extra_latency_prob;
    # pin it off so the two paths are comparable.
    monkeypatch.setattr(MVE, "EXTRA_LATENCY_PROB_RANGE", (0.0, 0.0))


def _rollout(env, steps=60, seed=0):
    rng = np.random.default_rng(seed)
    env.reset()
    out = []
    for _ in range(steps):
        a = torch.as_tensor(rng.normal(0.0, 0.6, (4, env.num_actions)))
        env.step(a)
        out.append(env._features["base_pos"].copy())
    return np.array(out)


def test_depth_one_reproduces_the_legacy_path(no_extra_latency):
    legacy = _env(no_extra_latency)
    deep = _env(no_extra_latency, action_latency_steps_range=[1, 1])
    a, b = _rollout(legacy), _rollout(deep)
    np.testing.assert_allclose(
        a, b, rtol=0, atol=0,
        err_msg="deep latency at depth 1 diverges from the legacy 1-step path")
    legacy.close(); deep.close()


def test_deeper_delay_actually_changes_the_trajectory(no_extra_latency):
    """Guard the guard: if depth had no effect, the test above would pass vacuously."""
    one = _env(no_extra_latency, action_latency_steps_range=[1, 1])
    five = _env(no_extra_latency, action_latency_steps_range=[5, 5])
    a, b = _rollout(one), _rollout(five)
    assert np.abs(a - b).max() > 1e-4, "a 5-step delay produced the same trajectory as 1 step"
    one.close(); five.close()


def test_imu_delay_zero_is_a_noop(no_extra_latency):
    """Same reduction property for the IMU observation delay."""
    off = _env(no_extra_latency)
    zero = _env(no_extra_latency, imu_delay_steps_range=[0, 0])
    np.testing.assert_allclose(_rollout(off), _rollout(zero), rtol=0, atol=0)
    off.close(); zero.close()


def test_imu_delay_shifts_the_observed_imu_not_the_joints(no_extra_latency):
    """The delay must hit ang_vel/gravity only -- dof_pos is host-computed and instant."""
    env = _env(no_extra_latency, imu_delay_steps_range=[3, 3])
    env.reset()
    seen = []
    for _ in range(12):
        obs = env.step(torch.full((4, env.num_actions), 1.5))[0]["policy"].numpy()
        seen.append(obs.copy())
    seen = np.array(seen)
    # first 6 columns are ang_vel(3) + gravity(3); they lag, so early frames still read the
    # seeded upright-and-still prior while the joint block has already moved.
    assert np.abs(seen[2, :, :3]).max() < 1e-9, "ang_vel should still be the seeded prior"
    assert np.abs(seen[2, :, 9:17]).max() > 1e-3, "dof_pos should have moved immediately"
    env.close()
