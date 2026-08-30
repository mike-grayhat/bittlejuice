"""Does the policy actually USE its observation history, or is it a 132-float decoration?

WHY THIS EXISTS. Two queued features -- retargeting the estimator, and replacing the flat
history concatenation with a temporal conv encoder (RMA's actual architecture) -- both
assume the history carries information the policy depends on. Neither is worth building if
it does not. Both cost a training run each to find out the expensive way; this costs one
rollout, because it asks the question of an already-trained checkpoint.

THE ABLATION. Replace the history block with the CURRENT frame repeated. That destroys
every temporal difference the policy could compute -- velocity, footfall transients, the
direction anything is moving -- while leaving every value inside the trained distribution.
Zero-filling would confound "history removed" with "input off-distribution", which is the
same mistake blindfold_test's docstring warns about for the IMU channels: a policy reacting
to an impossible input tells you nothing about what it reads from a possible one.

READING IT. Forward speed is the score, as in blindfold_test.

    cost < 2%    the history is decoration. A better history ENCODER cannot help, so the
                 estimator retarget and the TCN are both answering a question the policy is
                 not asking.
    cost > 5%    the history carries real signal and a better encoder has something to work
                 with.

The 2-5% band is deliberately a no-man's land: at that size the honest answer is that one
rollout cannot separate the effect from seed noise, and it needs more seeds.
"""

import argparse
import os
import pickle
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "sim"))
from mj_vec_env import MjVecEnv                                    # noqa: E402
from rsl_rl.runners import OnPolicyRunner                          # noqa: E402

from ppo_estimator import EVAL_LOAD_CFG                          # noqa: E402


# Observation layout, from policy_io.NUM_OBS = 3 + 3 + 3 + 8 + 8 + 8:
#   [0:3]   base angular velocity      | the two real sensors
#   [3:6]   projected gravity          |
#   [6:9]   commands                     constant within an episode
#   [9:17]  dof_pos                    | the servo-estimate echo: what the policy itself
#   [17:25] dof_vel                    | commanded, played back. Not a measurement.
#   [25:33] last_action                |
CHANNEL_GROUPS = {
    "all": slice(0, None),
    "imu": slice(0, 6),
    "proprio": slice(9, 33),          # dof_pos + dof_vel + last_action
    "dof": slice(9, 25),
    "action": slice(25, 33),
}


def flatten_history(obs, num_obs, group="all"):
    """Overwrite the chosen channels of every history frame with the current frame's values.

    Flattening a SUBSET is what separates the two things the history could be carrying.
    The whole-history result is not self-interpreting: this policy is an open-loop rhythm
    generator (blindfold_test: removing both real sensors costs 1-6%), and with no recurrent
    state the only place it can keep "where am I in the gait cycle" is the history. So a
    large whole-history cost may mean the ablation destroyed its CLOCK rather than removing
    information about the world.

    Ablating `imu` alone answers the world question; ablating `proprio` alone answers the
    clock question. If proprio carries nearly all of it, a better history encoder buys phase
    the policy already has, and the retarget is not what it looks like.
    """
    out = {k: v.clone() for k, v in obs.items()}
    if "history" not in out:
        return out
    n, width = out["history"].shape
    frames = width // num_obs
    h = out["history"].view(n, frames, num_obs).clone()
    sl = CHANNEL_GROUPS[group]
    cur = out["policy"][:, :num_obs][:, sl].unsqueeze(1).expand(n, frames, -1)
    h[:, :, sl] = cur
    out["history"] = h.reshape(n, width)
    return out


def load(exp_name, ckpt, seed, num_envs, flat):
    log_dir = f"logs/{exp_name}"
    with open(f"{log_dir}/cfgs.pkl", "rb") as f:
        env_cfg, obs_cfg, reward_cfg, command_cfg, train_cfg = pickle.load(f)
    reward_cfg["reward_scales"].setdefault("power", 0.0)
    env_cfg.setdefault("terrain_curriculum", False)
    env_cfg["domain_rand_enabled"] = False
    if flat:
        env_cfg["terrain_enabled"] = False
        env_cfg["slope_deg_range"] = [0.0, 0.0]
    trained_yaw = max((abs(v) for v in command_cfg["ang_vel_range"]), default=0.0) or 0.6
    command_cfg = dict(command_cfg)
    command_cfg["lin_vel_x_range"] = [0.10, 0.10]
    command_cfg["lin_vel_y_range"] = [0.0, 0.0]
    command_cfg["ang_vel_range"] = [0.0, 0.0]
    if env_cfg.get("heading_command"):
        env_cfg.setdefault("heading_max_yaw_rate", trained_yaw)
    env = MjVecEnv(num_envs=num_envs, env_cfg=env_cfg, obs_cfg=obs_cfg, reward_cfg=reward_cfg,
                   command_cfg=command_cfg, device="cpu", num_threads=min(num_envs, 8), seed=seed)
    runner = OnPolicyRunner(env, train_cfg, log_dir, device="cpu")
    if ckpt is None:
        ckpt = max(int(f[6:-3]) for f in os.listdir(log_dir) if f.startswith("model_"))
    runner.load(f"{log_dir}/model_{ckpt}.pt", load_cfg=EVAL_LOAD_CFG)
    return env, runner.get_inference_policy(device="cpu"), ckpt


def run(env, policy, steps, ablate, group="all"):
    env.reset()
    fwd, alive = [], []
    with torch.no_grad():
        obs = env.get_observations()
        for _ in range(steps):
            use = flatten_history(obs, env.num_obs, group) if ablate else obs
            obs, _, done, _ = env.step(policy(use))
            fwd.append(env._features["base_lin_vel"][:, 1].copy())
            alive.append(1.0 - done.numpy().astype(np.float64))
    return np.array(fwd), np.array(alive)


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("-e", "--exp-name", required=True)
    p.add_argument("--ckpt", type=int, default=None)
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3])
    p.add_argument("--num-envs", type=int, default=128)
    p.add_argument("--steps", type=int, default=600)
    p.add_argument("--group", default="all", choices=sorted(CHANNEL_GROUPS),
                   help="Which channels to flatten inside the history. 'all' asks whether the "
                        "history matters; 'imu' vs 'proprio' asks WHY.")
    p.add_argument("--flat", action="store_true",
                   help="Evaluate on flat ground. Terrain is where history SHOULD matter, so "
                        "the default keeps the run's own terrain.")
    a = p.parse_args()

    base, ablated, ckpt = [], [], a.ckpt
    for seed in a.seeds:
        env, policy, ckpt = load(a.exp_name, a.ckpt, seed, a.num_envs, a.flat)
        if env.obs_history_len == 0:
            print(f"{a.exp_name} trained without --obs-history; nothing to ablate.")
            env.close()
            return
        b, _ = run(env, policy, a.steps, ablate=False)
        h, _ = run(env, policy, a.steps, ablate=True, group=a.group)
        base.append(float(b.mean()))
        ablated.append(float(h.mean()))
        env.close()

    base, ablated = np.array(base), np.array(ablated)
    cost = 100.0 * (1.0 - ablated / base)
    print(f"\n{a.exp_name} @ {ckpt}   group={a.group}, history_len {env.obs_history_len}, "
          f"{a.num_envs} envs x {a.steps} steps x {len(a.seeds)} seeds"
          f"{', flat' if a.flat else ', own terrain'}")
    print("  seed    baseline    history flattened    cost")
    for s, bb, hh, cc in zip(a.seeds, base, ablated, cost):
        print(f"  {s:4d}    {bb:+8.4f}    {hh:+16.4f}    {cc:5.1f}%")
    print(f"  mean    {base.mean():+8.4f}    {ablated.mean():+16.4f}    {cost.mean():5.1f}%"
          f"   (spread {cost.std():.1f})")

    print("\nVERDICT")
    if cost.mean() < 2.0:
        print("  DECORATION. Flattening the history costs almost nothing, so the policy is not")
        print("  reading temporal structure. A better history ENCODER -- the estimator retarget,")
        print("  or a TCN -- has nothing to encode. Skip both.")
    elif cost.mean() > 5.0:
        print("  LOAD-BEARING. The history carries signal the policy depends on, so a better")
        print("  encoder has something to work with. The retarget is worth building.")
    else:
        print("  INCONCLUSIVE (2-5%). Too close to seed noise to act on; add seeds before")
        print("  spending a training run on it.")


if __name__ == "__main__":
    main()
