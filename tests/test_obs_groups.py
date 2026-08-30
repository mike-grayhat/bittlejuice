"""Observation groups: privileged critic, history stacking, recurrent actor.

All three attack the same weakness -- the actor's 33 inputs contain only 6 real measurements
(ang_vel, projected_gravity), since the joint channels are its own commands played back
through a servo model. The critical invariant is that privileged information reaches the
CRITIC only: if it ever leaked into the actor, training would look excellent and the robot
would fail, and that failure is silent -- the policy simply learns against the wrong inputs.
"""

import numpy as np
import pytest
import torch

from config import get_cfgs
from mj_vec_env import MjVecEnv


def _env(num_envs=8, **over):
    env_cfg, obs_cfg, reward_cfg, command_cfg = get_cfgs()
    env_cfg["terrain_enabled"] = False
    env_cfg.update(over)
    return MjVecEnv(num_envs=num_envs, env_cfg=env_cfg, obs_cfg=obs_cfg,
                    reward_cfg=reward_cfg, command_cfg=command_cfg,
                    device="cpu", num_threads=1, seed=0)


def test_groups_have_declared_shapes():
    env = _env(obs_history_len=5)
    obs = env.reset()
    assert set(obs.keys()) == {"policy", "privileged", "history"}
    assert obs["policy"].shape == (8, env.num_obs)
    assert obs["privileged"].shape == (8, env.num_privileged_obs)
    assert obs["history"].shape == (8, 5 * env.num_obs)
    env.close()


def test_history_absent_when_disabled():
    env = _env(obs_history_len=0)
    assert "history" not in env.reset().keys()
    env.close()


def test_history_holds_the_previous_frames_in_order():
    """history[:, 0:33] must be the observation from one step ago, and so on back."""
    env = _env(num_envs=4, obs_history_len=3, domain_rand_enabled=False)
    env.reset()
    seen = []
    rng = np.random.default_rng(0)
    for _ in range(8):
        out = env.step(torch.as_tensor(rng.normal(0, 1, (4, env.num_actions))))
        cur = out[0]["policy"].numpy()
        hist = out[0]["history"].numpy().reshape(4, 3, env.num_obs)
        # slot `lag` must be the observation from (lag + 1) steps ago
        for lag in range(min(len(seen), 3)):
            np.testing.assert_allclose(
                hist[:, lag], seen[-1 - lag], rtol=0, atol=0,
                err_msg=f"history slot {lag} is not the observation from {lag + 1} steps ago")
        # and the current frame must NOT appear in history -- it is already in "policy"
        if len(seen) >= 1:
            assert not np.allclose(hist[:, 0], cur), "history slot 0 duplicates the current obs"
        seen.append(cur)
    env.close()


def test_history_cleared_on_reset():
    """A fresh episode must not inherit the attitude the previous one fell with."""
    env = _env(num_envs=4, obs_history_len=4, domain_rand_enabled=False)
    env.reset()
    for _ in range(6):
        env.step(torch.full((4, env.num_actions), 2.0))
    assert np.abs(env._obs_history).sum() > 0
    env._reset_idx(np.array([True, False, True, False]))
    assert np.abs(env._obs_history[0]).sum() == 0.0
    assert np.abs(env._obs_history[2]).sum() == 0.0
    assert np.abs(env._obs_history[1]).sum() > 0.0
    env.close()


def test_privileged_carries_what_the_actor_cannot_know():
    """Spot-check the two most important channels against ground truth."""
    env = _env(num_envs=6, domain_rand_enabled=True)
    env.reset()
    for _ in range(10):
        out = env.step(torch.full((6, env.num_actions), 0.5))
    priv = out[0]["privileged"].numpy()
    f = env._features
    # base_lin_vel: the quantity tracking_lin_vel is scored on, and the actor has no velocity
    # sense at all.
    np.testing.assert_allclose(priv[:, :3], f["base_lin_vel"] * env.obs_scales["lin_vel"],
                               rtol=1e-5, atol=1e-5)
    # servo tracking error: true position minus the estimate == the contact/load signal that
    # the honest observation removed from the actor.
    err = (f["dof_pos"] - env._motor_offset - env._servo_estimate) * env.obs_scales["dof_pos"]
    np.testing.assert_allclose(priv[:, 19:27], err, rtol=1e-5, atol=1e-5)
    assert np.abs(err).max() > 1e-4, "tracking error is identically zero; the test proves nothing"
    env.close()


def test_privileged_friction_matches_the_sampled_model():
    env = _env(num_envs=16, domain_rand_enabled=True)
    out = env.reset()
    priv = out["privileged"].numpy()
    actual = np.array([m.geom_friction[env._foot_geom_ids[0], 0] for m in env.models])
    np.testing.assert_allclose(priv[:, 31], actual, rtol=1e-5)
    assert actual.std() > 0.01, "friction was not randomized; the test proves nothing"
    env.close()


def test_actor_observation_is_unchanged_by_the_extra_groups():
    """The whole point: adding a privileged critic must not alter what the actor receives."""
    a = _env(num_envs=4, domain_rand_enabled=False)
    b = _env(num_envs=4, obs_history_len=6, domain_rand_enabled=False)
    a.reset(); b.reset()
    act = torch.full((4, a.num_actions), 1.0)
    for _ in range(12):
        oa = a.step(act)[0]["policy"].numpy()
        ob = b.step(act)[0]["policy"].numpy()
    np.testing.assert_allclose(oa, ob, rtol=0, atol=0)
    a.close(); b.close()


# -- model wiring -------------------------------------------------------------
# The env emitting a group is only half of it; train_cfg's obs_groups decides who reads it.
# These build the real rsl_rl models and check the input dimensions they resolved to.

def _runner(tmp_path, **train_over):
    import mj_train  # noqa: F401  (ensures the same cfg helpers)
    from rsl_rl.runners import OnPolicyRunner
    from config import get_train_cfg
    env_cfg, obs_cfg, reward_cfg, command_cfg = get_cfgs()
    env_cfg["terrain_enabled"] = False
    env_cfg["obs_history_len"] = train_over.pop("obs_history_len", 0)
    train_cfg = get_train_cfg("_t", entropy_coef=0.005)
    train_cfg.update(train_over)
    env = MjVecEnv(num_envs=8, env_cfg=env_cfg, obs_cfg=obs_cfg, reward_cfg=reward_cfg,
                   command_cfg=command_cfg, device="cpu", num_threads=1, seed=0)
    return env, OnPolicyRunner(env, train_cfg, str(tmp_path), device="cpu")


def test_privileged_reaches_critic_but_not_actor(tmp_path):
    env, runner = _runner(
        tmp_path, obs_groups={"actor": ["policy"], "critic": ["policy", "privileged"]})
    actor, critic = runner.alg.actor, runner.alg.critic
    assert actor.obs_dim == env.num_obs, (
        f"actor sees {actor.obs_dim} inputs, expected the {env.num_obs} the robot can produce "
        f"-- privileged information has leaked into the deployed policy")
    assert critic.obs_dim == env.num_obs + env.num_privileged_obs
    env.close()


def test_history_widens_both_models(tmp_path):
    env, runner = _runner(
        tmp_path, obs_history_len=5,
        obs_groups={"actor": ["policy", "history"], "critic": ["policy", "history", "privileged"]})
    assert runner.alg.actor.obs_dim == env.num_obs * 6          # current + 5 past
    assert runner.alg.critic.obs_dim == env.num_obs * 6 + env.num_privileged_obs
    env.close()


def test_recurrent_actor_builds_and_is_recurrent(tmp_path):
    from config import get_train_cfg
    cfg = get_train_cfg("_t", entropy_coef=0.005)
    for role in ("actor", "critic"):
        cfg[role] = dict(cfg[role], class_name="RNNModel", rnn_type="gru",
                         rnn_hidden_dim=64, rnn_num_layers=1)
    env, runner = _runner(tmp_path, obs_groups={"actor": ["policy"],
                                                "critic": ["policy", "privileged"]},
                          actor=cfg["actor"], critic=cfg["critic"])
    assert runner.alg.actor.is_recurrent
    assert runner.alg.actor.rnn.rnn.__class__.__name__ == "GRU"
    assert runner.alg.actor.obs_dim == env.num_obs
    env.close()


# -- sim <-> deploy history ordering ------------------------------------------

def test_deploy_history_matches_the_simulator_exactly():
    """deploy.ObsHistory must produce the same vector the actor saw in training.

    The two live in different packages with no shared code. Reversed order, or including
    the current frame twice, would still RUN -- the network accepts any vector of the right
    width -- and would silently execute a different policy. Same class of bug as the
    dof_pos mismatch, which cost 4x gait tempo before it was found.
    """
    import obs_builder as ob

    N = 4
    env = _env(num_envs=1, obs_history_len=N, domain_rand_enabled=False)
    env.reset()
    hist = ob.ObsHistory(N, env.num_obs)
    # BOTH SIDES START ZERO-FILLED, and this test used to assert the opposite. It called a
    # deploy-side prime() on the claim that "the simulator rolls the post-reset observation
    # into slot 0 before the first step" -- measured, sim's first post-reset observation has
    # 0 of 33 channels set in EVERY history slot. The test was written to match the deploy
    # code rather than the simulator, so it certified the divergence it existed to catch:
    # on hardware the policy's opening frames carried upright gravity and a live command
    # where training had zeros. prime() is gone; the constructor's zeros are the contract.
    #
    # The RESET observation is part of the sequence and is where the bug lived, so the
    # comparison starts there. control_loop calls extend() on every tick including its
    # first, so the deploy side must be fed the post-reset frame too -- an earlier version
    # of this test began at step 1 and therefore never exercised initial state at all.
    first = env.get_observations()
    cur0 = first["policy"].numpy()[0]
    np.testing.assert_allclose(
        first["history"].numpy()[0], np.zeros(env.num_obs * N), rtol=0, atol=0,
        err_msg="sim no longer zero-fills history at reset; ObsHistory must match it")
    np.testing.assert_allclose(
        hist.extend(cur0), np.concatenate([cur0, first["history"].numpy()[0]]),
        rtol=0, atol=0, err_msg="reset frame: deploy history differs from the simulator's")

    rng = np.random.default_rng(7)
    for step in range(10):
        out = env.step(torch.as_tensor(rng.normal(0, 1, (1, env.num_actions))))
        cur = out[0]["policy"].numpy()[0]
        # what rsl_rl hands the actor: groups concatenated in obs_groups order
        sim_vec = np.concatenate([cur, out[0]["history"].numpy()[0]])
        deploy_vec = hist.extend(cur)
        np.testing.assert_allclose(
            deploy_vec, sim_vec, rtol=0, atol=0,
            err_msg=f"step {step}: deploy history layout differs from the simulator's")
    assert sim_vec.shape == (env.num_obs * (N + 1),)
    env.close()


def test_deploy_history_is_a_noop_when_disabled():
    import obs_builder as ob
    h = ob.ObsHistory(0, 33)
    v = np.arange(33, dtype=np.float64)
    np.testing.assert_allclose(h.extend(v), v)
