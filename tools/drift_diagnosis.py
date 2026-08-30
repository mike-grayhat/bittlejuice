"""Why does the robot drift sideways and not recover? Separate the candidate causes.

The symptom, in the viewer: after a push the robot keeps sliding in one direction, sometimes
backwards, and never recovers forward velocity. Two explanations fit that description and they
need different fixes, so this measures which one dominates instead of arguing about it.

  SLOPE      -- a slope sets a PERSISTENT downhill direction for the whole episode, which is
                what "keeps drifting the same way" sounds like. And it can be physically
                unrecoverable: the robot slides whenever tan(slope) > mu, and FOOT_FRICTION_RANGE
                bottoms out at 0.10 (critical slope 5.7 deg).
  PUSH       -- an instantaneous qvel kick. Note it is UNOBSERVABLE by construction: the actor
                sees ang_vel, projected_gravity and its own servo echo, so a steady drift with
                the body upright and not rotating produces literally zero signal. The policy
                would be penalised for a state it cannot perceive.

The discriminator is direction. A push direction is resampled every few seconds and averages to
zero over an episode; a slope's downhill direction is fixed. So if drift correlates with the
downhill azimuth, it is the slope.

Usage:
    uv run python tools/drift_diagnosis.py -e my-run --ckpt 14999
"""

import pathlib as _pathlib
import sys as _sys
_R = _pathlib.Path(__file__).resolve().parents[1]
_sys.path[:0] = [str(_R / "sim"), str(_R / "deploy")]

import argparse
import os
import pickle

import numpy as np
import torch
from rsl_rl.runners import OnPolicyRunner

from ppo_estimator import EVAL_LOAD_CFG

import mj_vec_env as MVE
from mj_vec_env import MjVecEnv

CONDITIONS = {
    #  name            slope_deg   push_dv_xy
    "flat, no push":     (0.0,       0.0),
    "flat, push":        (0.0,       0.10),
    "slope, no push":    (8.0,       0.0),
    "slope + push":      (8.0,       0.10),   # as trained
}


def load(exp_name, ckpt, overrides):
    log_dir = f"logs/{exp_name}"
    with open(f"{log_dir}/cfgs.pkl", "rb") as f:
        env_cfg, obs_cfg, reward_cfg, command_cfg, train_cfg = pickle.load(f)
    env_cfg.update(overrides)
    # Hold a single forward command so lateral velocity is unambiguously error, not commanded.
    command_cfg["lin_vel_x_range"] = [0.10, 0.10]
    command_cfg["lin_vel_y_range"] = [0.0, 0.0]
    command_cfg["ang_vel_range"] = [0.0, 0.0]
    env = MjVecEnv(num_envs=256, env_cfg=env_cfg, obs_cfg=obs_cfg, reward_cfg=reward_cfg,
                   command_cfg=command_cfg, device="cpu", num_threads=os.cpu_count(), seed=7)
    runner = OnPolicyRunner(env, train_cfg, log_dir, device="cpu")
    if ckpt is None:
        ckpts = [int(f[6:-3]) for f in os.listdir(log_dir) if f.startswith("model_")]
        ckpt = max(ckpts)
    runner.load(f"{log_dir}/model_{ckpt}.pt", load_cfg=EVAL_LOAD_CFG)
    return env, runner.get_inference_policy(device="cpu"), ckpt


def run(env, policy, steps=1200):
    """Body-frame velocities plus, per env, this episode's downhill azimuth."""
    env.reset()
    fwd, lat, slope, downhill_dot = [], [], [], []
    with torch.no_grad():
        obs = env.get_observations()
        for _ in range(steps):
            obs, _, _, _ = env.step(policy(obs))
            f = env._features
            v_world = f["base_lin_vel"]          # body frame (x lateral, y forward)
            fwd.append(v_world[:, 1].copy())
            lat.append(v_world[:, 0].copy())
            slope.append(env._slope_deg.copy())
            # World-frame downhill direction is the horizontal part of gravity.
            g = np.array([m.opt.gravity[:] for m in env.models])
            gh = g[:, :2]
            n = np.linalg.norm(gh, axis=1, keepdims=True)
            gh = np.divide(gh, n, out=np.zeros_like(gh), where=n > 1e-9)
            qv0 = 1 + env._base_model.nq
            vw = env.state[:, qv0:qv0 + 2]       # world-frame linear velocity
            downhill_dot.append(np.sum(vw * gh, axis=1))
    return (np.array(fwd), np.array(lat), np.array(slope), np.array(downhill_dot))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("-e", "--exp-name", required=True)
    p.add_argument("--ckpt", type=int, default=None)
    p.add_argument("--steps", type=int, default=1200)
    args = p.parse_args()

    print(f"{'condition':<16} {'fwd m/s':>9} {'|lat| m/s':>10} {'P(fwd<0)':>9} {'downhill m/s':>13}")
    print("-" * 62)
    rows = {}
    for name, (slope_deg, push) in CONDITIONS.items():
        env, policy, ckpt = load(args.exp_name, args.ckpt, {
            "slope_deg_range": [0.0, slope_deg],
            "push_dv_xy": push,
            "push_dv_z": push / 2.0,
            "domain_rand_enabled": True,
        })
        fwd, lat, slope, dh = run(env, policy, args.steps)
        rows[name] = (fwd, lat, slope, dh)
        print(f"{name:<16} {fwd.mean():9.4f} {np.abs(lat).mean():10.4f} "
              f"{(fwd < 0).mean():9.3f} {dh.mean():13.4f}")
        env.close()

    print(f"\n(checkpoint {ckpt}, command 0.10 fwd / 0 lat / 0 yaw, {args.steps} steps x 256 envs)")
    print("\nREADING IT:")
    print("  'downhill m/s' is velocity projected on the downhill direction. Near zero means the")
    print("  robot holds its ground; clearly positive means it is being carried down the slope.")
    print("  If drift is the SLOPE, that column is large whenever slope is on and ~0 when flat.")
    print("  If drift is the PUSH, |lat| rises with push and downhill stays ~0.")

    fwd, lat, slope, dh = rows["slope + push"]
    if slope.max() > 0:
        print("\n-- slope + push, split by this episode's grade --")
        print(f"  {'slope band':<14} {'fwd m/s':>9} {'downhill m/s':>13} {'share':>7}")
        edges = [0, 2, 4, 6, 8]
        for lo, hi in zip(edges[:-1], edges[1:]):
            m = (slope >= lo) & (slope < hi)
            if m.sum() < 100:
                continue
            print(f"  {f'{lo}-{hi} deg':<14} {fwd[m].mean():9.4f} {dh[m].mean():13.4f} "
                  f"{m.mean():7.3f}")
        print("\n  A grade-dependent downhill term is the slope. A flat one is not.")


if __name__ == "__main__":
    main()
