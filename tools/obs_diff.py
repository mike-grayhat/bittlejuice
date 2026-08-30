"""Diff the observation the ROBOT hands the policy against the one SIM does, channel by channel.

Why: under identical noise settings the same policy commands ~93 deg/s of joint rate in sim and
~310-350 on hardware, and the robot's gait is visibly shaky. Timing is ruled out (IMU intervals
measured clean at 19.9 ms median, 0% late) and the estimator is ruled out (a run without one
is equally shaky). That leaves observation CONTENT, and 33 channels is few enough to just look
rather than keep hypothesising.

TWO CAVEATS ON THAT PREMISE, both found after this file was written; the tool is still the
right one, the motivating number is not.

  1. Most of the "3.5x gap" was a STATISTIC, not physics: sim's figure was a pooled
     per-joint median and hardware's a max-over-joints. Compared in the same form the same
     data reads 135 against 120 (control_loop.py's note).
  2. The rest was mostly not in the observation at all. The sim PLANT was wrong: the MJCF
     actuator was assumed near-instant behind the servo-response filter and actually added
     123 ms of its own, so sim settled in ~141 ms against the robot's ~40 (see
     §6.1). And the deploy loop was sending the filter's output over the wire, making the
     real servo apply those dynamics twice.

So do not read a residual here as necessarily observational. What this file is good for is
unchanged and still worth running after a hardware session: a per-channel tick-to-tick delta
comparison localises a mismatch instead of guessing at one. Just size the expectation
against the plant fixes above, and compare like with like.

The statistic that matters is NOT each channel's mean or std -- a policy can be fed a slightly
mis-scaled channel and still walk smoothly. It is the TICK-TO-TICK DELTA: a channel that jitters
between consecutive frames drives the actor to jitter with it, and that is exactly what a
finite-differenced attitude signal or a quantised sensor produces while still looking correct in
aggregate. So the table below sorts on delta ratio.

Usage:
    # on hardware
    cd deploy && uv run python control_loop.py --policy policy/bittle_policy_v23.npz \\
        --command 0.10 0 0 --duration 10 --log-obs hw_obs.csv
    # then here
    uv run python tools/obs_diff.py -e my-run --ckpt 9100 --hw hw_obs.csv
"""

import pathlib as _pathlib
import sys as _sys
_R = _pathlib.Path(__file__).resolve().parents[1]
_sys.path[:0] = [str(_R / "sim"), str(_R / "deploy")]

import argparse
import pickle

import numpy as np
import torch
from rsl_rl.runners import OnPolicyRunner

from mj_vec_env import MjVecEnv
from obs_builder import OBS_CHANNEL_NAMES


def sim_obs(exp, ckpt, steps, command):
    d = f"logs/{exp}"
    with open(f"{d}/cfgs.pkl", "rb") as f:
        e, o, r, c, t = pickle.load(f)
    # Flat and quiet, matching a hardware run on a level floor. Domain randomization stays ON:
    # it carries the observation noise the policy trained against, which is the fair comparison.
    e.update({"terrain_enabled": False, "slope_deg_range": [0.0, 0.0],
              "push_dv_xy": 0.0, "push_dv_z": 0.0, "domain_rand_enabled": True})
    # Capture the TRAINED yaw limit before pinning the command, then re-assert it. Pinning
    # collapses ang_vel_range to zero width, and heading_max_yaw_rate defaults to that width
    # -- so a heading-trained policy becomes unevaluatable here, which is exactly what the
    # env's guard says. blindfold_test.py and mj_deployability.py already do this; obs_diff
    # was the one that did not, and it failed the moment a heading run was pointed at it.
    trained_yaw_limit = max((abs(v) for v in c["ang_vel_range"]), default=0.0) or 0.6
    c["lin_vel_x_range"] = [command[0], command[0]]
    c["lin_vel_y_range"] = [command[1], command[1]]
    c["ang_vel_range"] = [command[2], command[2]]
    if e.get("heading_command"):
        e.setdefault("heading_max_yaw_rate", trained_yaw_limit)
    env = MjVecEnv(num_envs=1, env_cfg=e, obs_cfg=o, reward_cfg=r, command_cfg=c,
                   device="cpu", num_threads=1, seed=0)
    run = OnPolicyRunner(env, t, d, device="cpu")
    run.load(f"{d}/model_{ckpt}.pt")
    pol = run.get_inference_policy(device="cpu")
    out = []
    with torch.no_grad():
        obs = env.get_observations()
        for _ in range(steps):
            out.append(obs["policy"].numpy()[0].copy())
            obs, _, _, _ = env.step(pol(obs))
    env.close()
    return np.array(out)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("-e", "--exp-name", required=True)
    p.add_argument("--ckpt", type=int, required=True)
    p.add_argument("--hw", required=True, help="CSV from control_loop --log-obs")
    p.add_argument("--command", type=float, nargs=3, default=[0.10, 0.0, 0.0])
    args = p.parse_args()

    hw = np.loadtxt(args.hw, delimiter=",", skiprows=1)
    sim = sim_obs(args.exp_name, args.ckpt, len(hw), args.command)
    print(f"\nhardware {hw.shape}   sim {sim.shape}\n")

    # Tick-to-tick change, the quantity that drives action jitter.
    dh = np.abs(np.diff(hw, axis=0)).mean(axis=0)
    ds = np.abs(np.diff(sim, axis=0)).mean(axis=0)
    rows = []
    for i, name in enumerate(OBS_CHANNEL_NAMES):
        ratio = dh[i] / ds[i] if ds[i] > 1e-12 else np.inf
        rows.append((ratio, name, hw[:, i].mean(), sim[:, i].mean(),
                     hw[:, i].std(), sim[:, i].std(), dh[i], ds[i]))
    rows.sort(reverse=True, key=lambda r: (np.isinf(r[0]), r[0]))

    print(f"{'channel':<16}{'hw mean':>9}{'sim mean':>10}{'hw std':>9}{'sim std':>9}"
          f"{'hw |d|':>9}{'sim |d|':>9}{'RATIO':>8}")
    print("-" * 79)
    for ratio, name, hm, sm, hs, ss, d1, d2 in rows:
        flag = "  <<<" if (ratio > 3.0 or ratio < 0.33) else ""
        rs = "inf" if np.isinf(ratio) else f"{ratio:.1f}"
        print(f"{name:<16}{hm:9.3f}{sm:10.3f}{hs:9.3f}{ss:9.3f}{d1:9.3f}{d2:9.3f}{rs:>8}{flag}")

    print("\nRATIO is hardware tick-to-tick change divided by sim's. Channels marked <<< move")
    print("more than 3x differently between consecutive frames than the policy ever saw in")
    print("training -- those are what make the actor jitter, regardless of whether their mean")
    print("and std look right.")


if __name__ == "__main__":
    main()
