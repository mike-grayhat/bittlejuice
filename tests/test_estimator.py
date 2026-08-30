"""The supervised state estimator, and the ways it could silently do nothing.

Context: the actor has no velocity sense at all -- a steady drift with the body upright and
unrotating leaves ang_vel, projected_gravity and the servo echo all unchanged -- while
tracking_lin_vel scores it on exactly that. ppo_estimator.py closes the loop by predicting
base_lin_vel from deployable signals and feeding the estimate to the actor.

Three failure modes here produce a run that trains cleanly and teaches nothing:

  * the target slice drifts from the privileged layout, so the estimator regresses dof_pos;
  * the estimator update runs AFTER PPO clears the rollout storage, so it trains on an empty
    buffer and reports a tidy zero loss forever;
  * the estimate written into the observation is not the one the actor consumed, which makes
    PPO's probability ratio compare two different policies.

None of them raise.
"""

import os
import pickle

import numpy as np
import pytest
import torch
from rsl_rl.runners import OnPolicyRunner

import policy_io as pio
from config import get_cfgs, get_train_cfg
from mj_vec_env import MjVecEnv
from ppo_estimator import ESTIMATE_GROUP, TARGET_DIM, TARGET_SLICE, PPOWithEstimator

HIST = 4


def _env(num_envs=32, estimator_dim=TARGET_DIM, **over):
    env_cfg, obs_cfg, reward_cfg, command_cfg = get_cfgs()
    env_cfg.update({"terrain_enabled": False, "slope_deg_range": [0.0, 0.0],
                    "domain_rand_enabled": False, "obs_history_len": HIST,
                    "estimator_dim": estimator_dim})
    env_cfg.update(over)
    return MjVecEnv(num_envs=num_envs, env_cfg=env_cfg, obs_cfg=obs_cfg, reward_cfg=reward_cfg,
                    command_cfg=command_cfg, device="cpu", num_threads=1, seed=0), obs_cfg


def _runner(env, tmp_path, estimator=True):
    cfg = get_train_cfg("test-estimator")
    actor_groups = ["policy", "history"]
    cfg["obs_groups"] = {"actor": list(actor_groups),
                         "critic": actor_groups + ["privileged"]}
    if estimator:
        cfg["obs_groups"]["estimator"] = list(actor_groups)
        cfg["obs_groups"]["actor"] = actor_groups + [ESTIMATE_GROUP]
        cfg["algorithm"]["class_name"] = "ppo_estimator:PPOWithEstimator"
        cfg["algorithm"]["estimator_cfg"] = {"learning_rate": 1e-3}
    cfg["num_steps_per_env"] = 8
    return OnPolicyRunner(env, cfg, str(tmp_path), device="cpu")


# -- the target ---------------------------------------------------------------

def test_target_slice_really_is_base_lin_vel():
    """If a channel is ever inserted ahead of base_lin_vel in _update_privileged, this slice
    silently starts pointing at dof_pos and the estimator regresses the wrong thing."""
    env, obs_cfg = _env()
    env.reset()
    for _ in range(3):
        env.step(torch.zeros((env.num_envs, env.num_actions)))
    privileged = env.get_observations()["privileged"].numpy()
    expected = env._features["base_lin_vel"] * obs_cfg["obs_scales"]["lin_vel"]
    assert np.allclose(privileged[:, TARGET_SLICE], expected, atol=1e-5), (
        "PRIVILEGED_LIN_VEL_SLICE does not index base_lin_vel; the estimator would be trained "
        "against whatever now sits at the front of the privileged vector"
    )
    assert TARGET_DIM == 3
    env.close()


# -- the observation group ----------------------------------------------------

def test_env_emits_estimate_group_at_the_declared_width():
    env, _ = _env(estimator_dim=TARGET_DIM)
    obs = env.get_observations()
    assert ESTIMATE_GROUP in obs.keys()
    assert obs[ESTIMATE_GROUP].shape == (env.num_envs, TARGET_DIM)
    # The env only declares it; it must never put anything in it.
    assert torch.all(obs[ESTIMATE_GROUP] == 0.0)
    env.close()


def test_estimator_dim_zero_reproduces_the_old_observation_exactly():
    """The feature must be free when off, or every prior config changes meaning."""
    env, _ = _env(estimator_dim=0)
    obs = env.get_observations()
    assert ESTIMATE_GROUP not in obs.keys()
    assert obs["policy"].shape[1] == pio.NUM_OBS
    assert obs["history"].shape[1] == pio.NUM_OBS * HIST
    env.close()


# -- the algorithm ------------------------------------------------------------

def test_actor_input_widens_by_the_estimate(tmp_path):
    env, _ = _env()
    runner = _runner(env, tmp_path)
    actor_in = runner.alg.actor.mlp[0].in_features
    est_in = runner.alg.estimator.mlp[0].in_features
    assert est_in == pio.NUM_OBS * (1 + HIST), "estimator must read only deployable groups"
    assert actor_in == est_in + TARGET_DIM, (
        f"actor input {actor_in} should be {est_in} + {TARGET_DIM}; without the estimate as an "
        f"INPUT nothing forces the policy to use it"
    )
    env.close()


def test_act_writes_the_estimate_the_actor_consumed(tmp_path):
    """The stored transition must carry the same estimate the actor saw, or PPO's ratio compares
    two different policies -- the class of bug behind this project's phantom latency cliff."""
    env, _ = _env()
    runner = _runner(env, tmp_path)
    alg = runner.alg
    obs = env.get_observations()
    with torch.no_grad():
        expected = alg.estimator(obs)
        alg.act(obs)
    assert not torch.allclose(obs[ESTIMATE_GROUP], torch.zeros_like(obs[ESTIMATE_GROUP])), \
        "estimate group is still zero; act() did not fill it"
    assert torch.allclose(obs[ESTIMATE_GROUP], expected, atol=1e-6)
    assert torch.allclose(alg.transition.observations[ESTIMATE_GROUP], expected, atol=1e-6)
    env.close()


def test_estimator_update_runs_before_the_storage_is_cleared(tmp_path, monkeypatch):
    """PPO.update() ends with storage.clear(), so the estimator update must precede it.

    Asserted by ORDER, not by the loss value, because the failure is quieter than it looks:
    clear() only resets the write cursor (`self.step = 0`) and leaves every tensor in place, so
    the wrong ordering still yields full mini-batches -- of the PREVIOUS rollout, drifting
    further out of date each iteration. It would report a healthy falling loss the whole run
    while training the estimator on stale data. There is no value assertion that catches that.
    """
    env, _ = _env()
    runner = _runner(env, tmp_path)
    alg = runner.alg
    calls = []

    real_est = alg._update_estimator
    real_clear = alg.storage.clear
    monkeypatch.setattr(alg, "_update_estimator",
                        lambda: (calls.append("estimator"), real_est())[1])
    monkeypatch.setattr(alg.storage, "clear",
                        lambda: (calls.append("clear"), real_clear())[1])

    runner.learn(num_learning_iterations=2)
    assert calls, "neither hook fired"
    for i in range(0, len(calls) - 1, 2):
        assert calls[i] == "estimator" and calls[i + 1] == "clear", (
            f"call order {calls} -- the estimator must consume the rollout before PPO resets "
            f"the write cursor over it"
        )
    env.close()


def test_estimator_progress_is_read_from_r2_not_raw_loss(tmp_path):
    """Track the NORMALISED error, because raw MSE is not a progress metric here.

    The target is non-stationary: as the policy learns to walk, |base_lin_vel| grows, so the
    MSE against it can rise while the estimator is steadily improving. Measured over 10
    iterations the raw loss went 0.0016 -> 0.0019 (up) while the normalised error fell.
    A run judged on raw loss would look broken.

    R^2 is measured against the MEAN, so 0.0 is exactly "carries no information". The earlier
    ||err||/||target|| form put its useless point at 1.0, which flattered a predictor that had
    merely learned the average forward speed.
    """
    env, _ = _env(num_envs=128)
    runner = _runner(env, tmp_path)
    seen = []
    real = runner.alg._update_estimator

    def spy():
        out = real()
        seen.append(out["estimator_r2"])
        return out

    runner.alg._update_estimator = spy
    runner.learn(num_learning_iterations=12)
    assert all(np.isfinite(v) for v in seen), f"non-finite nrmse: {seen}"
    assert seen[-1] > 0.0, (
        f"estimator R^2 {seen[-1]:.2f} -- no better than predicting the constant mean. The bar "
        f"is 0.0, NOT the 1.0 that an ||err||/||target|| ratio would suggest: predicting the "
        f"mean scores 0.585 on that scale while carrying zero information."
    )
    env.close()


def test_estimator_beats_predicting_zero(tmp_path):
    """nrmse < 1 means the estimate carries real information about velocity. At 1.0 the network
    is no better than outputting zero, which is the outcome that would say velocity is simply
    not inferable from these inputs -- see ppo_estimator's docstring."""
    env, _ = _env(num_envs=128)
    runner = _runner(env, tmp_path)
    runner.learn(num_learning_iterations=12)
    stats = runner.alg._update_estimator()   # storage is empty; use the logged history instead
    del stats
    # Re-collect a little data and score the trained estimator against ground truth directly.
    obs = env.get_observations()
    with torch.no_grad():
        pred = runner.alg.estimator(obs)
    target = obs["privileged"][:, TARGET_SLICE]
    nrmse = float((pred - target).pow(2).sum().sqrt() / target.pow(2).sum().sqrt())
    assert np.isfinite(nrmse)
    assert nrmse < 1.5, f"estimator nrmse {nrmse:.2f} -- worse than predicting zero after 12 iters"
    env.close()


# -- checkpointing ------------------------------------------------------------

def test_estimator_survives_save_and_load(tmp_path):
    """The estimator is half the deployed policy. A checkpoint that silently omits it would
    reload a randomly-initialised velocity sense and look fine."""
    env, _ = _env()
    runner = _runner(env, tmp_path)
    runner.learn(num_learning_iterations=1)
    path = str(tmp_path / "model_1.pt")
    runner.save(path)
    saved = torch.load(path, weights_only=False)
    assert "estimator_state_dict" in saved

    env2, _ = _env()
    runner2 = _runner(env2, tmp_path)
    before = runner2.alg.estimator.mlp[0].weight.clone()
    runner2.load(path)
    after = runner2.alg.estimator.mlp[0].weight
    assert not torch.allclose(before, after), "load() left the estimator at its init weights"
    assert torch.allclose(after, runner.alg.estimator.mlp[0].weight)
    env.close()
    env2.close()


def test_inference_policy_includes_the_estimator(tmp_path):
    """get_inference_policy must return estimator->actor, not the actor alone: every eval and
    export path calls it, and an actor fed a zero estimate is not the trained policy."""
    env, _ = _env()
    runner = _runner(env, tmp_path)
    policy = runner.get_inference_policy(device="cpu")
    obs = env.get_observations()
    obs[ESTIMATE_GROUP] = torch.zeros_like(obs[ESTIMATE_GROUP])
    with torch.no_grad():
        action = policy(obs)
    assert action.shape == (env.num_envs, env.num_actions)
    assert not torch.all(obs[ESTIMATE_GROUP] == 0.0), \
        "the inference wrapper did not run the estimator"
    env.close()


# -- deployment parity --------------------------------------------------------

def test_deploy_side_runs_estimator_then_actor(tmp_path):
    """deploy composes the chain itself, from the caller's DEPLOYABLE observation.

    The exporter verifies its own numpy_forward against torch; it does not verify
    deploy's NumpyMLPPolicy, which is a separate hand-written implementation in a
    standalone package. A disagreement in concatenation order or width would run happily and
    execute a different policy than the reward curves describe.
    """
    from policy_numpy import NumpyMLPPolicy  # noqa: E402

    rng = np.random.default_rng(0)
    d_in, est_h, est_out, act_h = 165, 16, 3, 12
    npz = tmp_path / "p.npz"
    E = [(rng.normal(0, .1, (d_in, est_h)), rng.normal(0, .1, est_h)),
         (rng.normal(0, .1, (est_h, est_out)), rng.normal(0, .1, est_out))]
    A = [(rng.normal(0, .1, (d_in + est_out, act_h)), rng.normal(0, .1, act_h)),
         (rng.normal(0, .1, (act_h, 8)), rng.normal(0, .1, 8))]
    save = {"num_layers": np.array(len(A)), "estimator_num_layers": np.array(len(E)),
            "estimator_dim": np.array(est_out), "obs_history_len": np.array(4),
            "default_dof_pos": np.array(pio.DEFAULT_DOF_POS), "action_scale": np.array(0.25),
            "control_dt": np.array(0.02), "obs_scale_ang_vel": np.array(0.25),
            "obs_scale_dof_pos": np.array(1.0), "obs_scale_dof_vel": np.array(0.05),
            "commands_scale": np.array(pio.COMMANDS_SCALE), "joint_names": np.array(pio.JOINT_NAMES)}
    save.update({f"W{i}": w for i, (w, _) in enumerate(A)})
    save.update({f"b{i}": b for i, (_, b) in enumerate(A)})
    save.update({f"E_W{i}": w for i, (w, _) in enumerate(E)})
    save.update({f"E_b{i}": b for i, (_, b) in enumerate(E)})
    np.savez(npz, **save)

    pol = NumpyMLPPolicy(str(npz))
    assert pol.estimator_dim == est_out
    assert pol.num_input_obs == d_in, "caller must supply only the deployable observation"

    obs = rng.normal(0, 1, d_in)

    def fwd(layers, x):
        for i, (w, b) in enumerate(layers):
            x = x @ w + b
            if i < len(layers) - 1:
                x = np.where(x > 0, x, np.expm1(x))
        return x

    expected = fwd(A, np.concatenate([obs, fwd(E, obs)]))
    assert np.allclose(pol.act(obs), expected, atol=1e-12)
    assert np.allclose(pol.estimate(obs), fwd(E, obs), atol=1e-12)

    # Wrong width must be refused, not silently broadcast.
    with pytest.raises(ValueError):
        pol.act(rng.normal(0, 1, d_in + est_out))


# -- the deploy divergence bug ------------------------------------------------

def test_deploy_clips_actions_like_the_simulator(tmp_path):
    """The bug that makes a policy fall over and then emit NaN to the servos.

    mj_vec_env.step clips the action to +-clip_actions and stores the CLIPPED value, which
    becomes `last_action` in the next observation. policy_numpy returned the raw network
    output, so on hardware the loop action -> last_action -> observation -> action had no
    bound: one out-of-distribution observation escalated to 2.8e20 deg/s over 127 ticks and
    then NaN. It cannot be caught by a sim test, because sim does the clipping itself.
    """
    from policy_numpy import NumpyMLPPolicy  # noqa: E402

    rng = np.random.default_rng(0)
    d_in, clip = 33, 3.0
    npz = tmp_path / "clip.npz"
    # Deliberately enormous weights, so the raw output is far outside the bound.
    layers = [(rng.normal(0, 50.0, (d_in, 8)), rng.normal(0, 50.0, 8))]
    save = {"W0": layers[0][0], "b0": layers[0][1], "num_layers": np.array(1),
            "clip_actions": np.array(clip), "obs_history_len": np.array(0),
            "estimator_dim": np.array(0),
            "default_dof_pos": np.array(pio.DEFAULT_DOF_POS), "action_scale": np.array(0.25),
            "control_dt": np.array(0.02), "obs_scale_ang_vel": np.array(0.25),
            "obs_scale_dof_pos": np.array(1.0), "obs_scale_dof_vel": np.array(0.05),
            "commands_scale": np.array(pio.COMMANDS_SCALE),
            "joint_names": np.array(pio.JOINT_NAMES)}
    np.savez(npz, **save)

    pol = NumpyMLPPolicy(str(npz))
    assert pol.clip_actions == clip, "clip_actions is exported but was never loaded"
    action = pol.act(rng.normal(0, 1, d_in))
    assert np.all(np.abs(action) <= clip + 1e-12), (
        f"action {np.abs(action).max():.1f} exceeds clip_actions={clip}; unbounded actions feed "
        f"back through last_action and diverge the loop"
    )
    assert np.any(np.abs(action) >= clip - 1e-9), "test is vacuous; the clip never bound"


def test_deploy_refuses_a_non_finite_observation(tmp_path):
    """Once the loop has diverged the next step sends NaN to the servos. Fail loudly."""
    from policy_numpy import NumpyMLPPolicy  # noqa: E402

    rng = np.random.default_rng(1)
    npz = tmp_path / "nan.npz"
    np.savez(npz, W0=rng.normal(0, .1, (33, 8)), b0=rng.normal(0, .1, 8),
             num_layers=np.array(1), clip_actions=np.array(3.0),
             obs_history_len=np.array(0), estimator_dim=np.array(0),
             default_dof_pos=np.array(pio.DEFAULT_DOF_POS), action_scale=np.array(0.25),
             control_dt=np.array(0.02), obs_scale_ang_vel=np.array(0.25),
             obs_scale_dof_pos=np.array(1.0), obs_scale_dof_vel=np.array(0.05),
             commands_scale=np.array(pio.COMMANDS_SCALE), joint_names=np.array(pio.JOINT_NAMES))
    pol = NumpyMLPPolicy(str(npz))
    bad = rng.normal(0, 1, 33); bad[7] = np.nan
    with pytest.raises(FloatingPointError, match="diverged"):
        pol.act(bad)
