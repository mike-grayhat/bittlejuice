"""Deterministic deployability check: is this checkpoint's commanded joint rate physical?

The training metric `slew_saturated_frac` cannot answer this. It is computed on the
SAMPLED action, so it is dominated by exploration noise -- at action_std 0.53 the noise
alone gives ~80% saturation -- while the deployed policy runs the distribution MEAN. The
number that predicts hardware behaviour comes from a deterministic rollout, and measured
that way a policy that walks on hardware reads 6.6% and one that trembled reads 71.3%.
Rule of thumb from those two and the runs between them: above ~15% the gait does not
transfer.

Reports the same statistic in the same POOLED per-joint form control_loop.py prints, so
sim and hardware are directly comparable -- reporting one as a max-over-joints and the
other as a pooled median once produced an apparent 3.5x sim-to-real gap that was entirely
the statistic (see control_loop.py's note).

Also reports the action spectrum, because action_rate cannot discriminate smooth from
jerky: two policies can carry the same action_rate and differ 25x in high-frequency action
power. High-band fraction is what separates them.

Usage:
    uv run python tools/mj_deployability.py -e my-run
    uv run python tools/mj_deployability.py -e my-run --ckpt 5999 --domain-rand
    uv run python tools/mj_deployability.py -e A -e B          # compare checkpoints
"""

import pathlib as _pathlib
import sys as _sys
_R = _pathlib.Path(__file__).resolve().parents[1]
_sys.path[:0] = [str(_R / "sim"), str(_R / "deploy")]

import argparse
import math
import os
import pickle

import numpy as np
import torch
from rsl_rl.runners import OnPolicyRunner

from ppo_estimator import EVAL_LOAD_CFG

from mj_vec_env import ACTUATOR_SLEW_RAD_S_NOMINAL, MjVecEnv

TRANSFER_THRESHOLD = 0.15          # measured: 0.066 walks on hardware, 0.713 trembles
# Ignore sign flips inside near-stationary dither -- only count a reversal when the joint is
# actually moving on both sides of it.
REVERSAL_MIN_DEG_S = 5.0

# Survival on the policy's OWN training terrain, per episode-equivalent. This gate exists
# because a policy can pass every other check here -- 9.6% saturation, +0.102 m/s, healthy
# sensor reliance -- and still fall over on hardware within seconds. Sim predicts it: terrain
# survival drops well before anything else does, and without this the tool reports three green
# numbers and omits the one that matters.
#
# Deliberately MILD. The warn band is advisory because a terrain-trained policy is SUPPOSED
# to fall sometimes -- that is what makes the disturbance real enough for feedback to pay,
# and tightening this would quietly push runs back toward the open-loop optimum that
# terrain amplitude was raised to escape. Only the floor refuses.
SURVIVAL_WARN = 0.75
SURVIVAL_FLOOR = 0.50


def evaluate(exp_name, ckpt, seconds, domain_rand, command, num_envs, wire_quantize_deg=None,
             terrain=None, terrain_amplitude=None, terrain_correlation=None, seed=0):
    log_dir = f"logs/{exp_name}"
    with open(f"{log_dir}/cfgs.pkl", "rb") as f:
        env_cfg, obs_cfg, reward_cfg, command_cfg, train_cfg = pickle.load(f)
    if ckpt is None:
        ckpt = max(int(f[6:-3]) for f in os.listdir(log_dir) if f.startswith("model_"))

    env_cfg = dict(env_cfg)
    env_cfg["domain_rand_enabled"] = domain_rand
    # Absent keys MEAN something: a cfgs.pkl predating a feature says the run trained
    # without it, and defaulting to today's value would evaluate the checkpoint against an
    # observation or a plant it never saw. Same contract as mj_eval.py.
    if "observe_servo_estimate" not in env_cfg:
        env_cfg["observe_servo_estimate"] = False
    env_cfg.setdefault("wire_quantize_deg", 0.0)
    if wire_quantize_deg is not None:
        env_cfg["wire_quantize_deg"] = wire_quantize_deg

    # Terrain overrides, so two runs can be compared on the SAME ground. Without them this
    # tool evaluates every policy on ITS OWN training terrain, which silently makes forward
    # speed incomparable across runs -- +0.097 on 0.10 m correlation against +0.044 on
    # 0.05 m is mostly the ground, not the policy.
    if terrain is not None:
        env_cfg["terrain_enabled"] = terrain
    if terrain_amplitude is not None:
        env_cfg["terrain_amplitude"] = terrain_amplitude
    if terrain_correlation is not None:
        env_cfg["terrain_correlation"] = terrain_correlation

    # PIN THE DIFFICULTY. A run trained with --terrain-curriculum carries
    # terrain_curriculum=True in its cfgs.pkl, and a freshly built env starts every
    # environment on rung 0 -- so --terrain-amplitude would set the HARDEST rung of a ladder
    # nobody climbs, and the tool would report rung-0 terrain under whatever amplitude was
    # asked for. The tell that this is happening: two different --terrain-amplitude values
    # return byte-identical numbers, which two different grounds cannot do.
    #
    # Evaluation wants one known ground for every env, so the curriculum is switched off here
    # and terrain_amplitude means what it means everywhere else in this tool: the actual
    # amplitude. Judging a checkpoint on the terrain it happens to have been promoted to is a
    # different question, and not one you can compare across runs.
    env_cfg["terrain_curriculum"] = False

    trained_yaw_limit = max(abs(v) for v in command_cfg["ang_vel_range"]) or 0.6
    command_cfg = dict(command_cfg)
    if command is None:
        lo, hi = command_cfg["lin_vel_x_range"]
        command = [0.5 * (lo + hi), 0.0, 0.0]
    command_cfg["lin_vel_x_range"] = [command[0], command[0]]
    command_cfg["lin_vel_y_range"] = [command[1], command[1]]
    command_cfg["ang_vel_range"] = [command[2], command[2]]
    if env_cfg.get("heading_command"):
        env_cfg.setdefault("heading_max_yaw_rate", trained_yaw_limit)

    env = MjVecEnv(num_envs=num_envs, env_cfg=env_cfg, obs_cfg=obs_cfg,
                   reward_cfg=reward_cfg, command_cfg=command_cfg, device="cpu",
                   num_threads=min(num_envs, 4), seed=seed)
    runner = OnPolicyRunner(env, train_cfg, log_dir, device="cpu")
    runner.load(os.path.join(log_dir, f"model_{ckpt}.pt"), load_cfg=EVAL_LOAD_CFG)
    policy = runner.get_inference_policy(device="cpu")

    obs = env.reset()
    n_steps = int(seconds / env.dt)
    actions, fwd, dof, resets = [], [], [], 0
    with torch.no_grad():
        for _ in range(n_steps):
            act = policy(obs)                       # deterministic: the distribution MEAN
            obs, _, dones, _ = env.step(act)
            actions.append(np.asarray(act.detach().cpu().numpy(), dtype=np.float64))
            fwd.append(env._features["base_lin_vel"][:, 1].copy())
            # ACHIEVED joint angles, not the action. Reversals are a property of the motion
            # the operator watches, and the actuator filter sits between the two.
            dof.append(env._features["dof_pos"].copy())
            resets += int(dones.sum())
    env.close()

    # Terminations are counted across the whole rollout, so normalise by how many full
    # episodes it actually contained rather than by env count -- a --duration longer than
    # episode_length_s gives each env more than one chance to fall.
    ep_steps = max(int(env_cfg.get("episode_length_s", 20.0) / env.dt), 1)
    episode_equivalents = num_envs * max(n_steps / ep_steps, 1e-9)
    survival = max(0.0, 1.0 - resets / episode_equivalents)

    # -- EXCESS DIRECTION REVERSALS ------------------------------------------------------
    # The only metric so far that reproduces the operator's smoothness ranking. Every other
    # one is a MAGNITUDE (rate, jerk, acceleration, saturation); this is a COUNT, and the two
    # are orthogonal. A single large sweep has enormous jerk at each turnaround and looks
    # perfectly smooth; many small back-and-forth motions have low jerk and look terrible.
    #
    # Measured on achieved joint angles, normalised by the 2*f reversals a clean periodic
    # gait needs (one at each end of the swing), so it does not reward slowing down:
    #
    #     0.73x  "smoothest"      0.95x
    #     0.88x                     1.04x  "less smooth"
    #                               1.07x  "quite jerky"
    #
    # The policy reading smoothest of those has the HIGHEST jerk (131k against 82k deg/s^3),
    # so jerk is anti-correlated with perceived smoothness here: reward_scales' action_jerk
    # term penalises something the eye does not track.
    q = np.degrees(np.stack(dof))                   # (T, num_envs, 8), degrees
    vq = np.diff(q, axis=0) / env.dt
    moving = np.abs(vq) > REVERSAL_MIN_DEG_S        # or sensor-scale dither inflates the count
    flips = (np.diff(np.sign(vq), axis=0) != 0) & moving[1:] & moving[:-1]
    rev_per_s = flips.sum(axis=0).mean() / (len(vq) * env.dt)
    qc = q - q.mean(axis=0)
    win = np.hanning(qc.shape[0])[:, None, None]
    Pq = (np.abs(np.fft.rfft(qc * win, axis=0)) ** 2).sum(axis=(1, 2))
    fq = np.fft.rfftfreq(qc.shape[0], env.dt)
    gait_band = (fq > 0.3) & (fq < 6.0)
    cadence = float(fq[gait_band][np.argmax(Pq[gait_band])])
    excess_reversals = rev_per_s / max(2.0 * cadence, 1e-9)

    a = np.stack(actions)                           # (T, num_envs, 8)
    scale = env_cfg["action_scale"]
    rate = np.abs(np.diff(a, axis=0)) * scale / env.dt          # rad/s, per joint
    pooled = np.degrees(rate).ravel()
    limit_deg = math.degrees(ACTUATOR_SLEW_RAD_S_NOMINAL)

    # Action spectrum, pooled over joints and envs. The bands are the ones that separated
    # a smooth policy from a buzzing one; a gait's own cadence lives below 1.5 Hz, so power
    # above 5 Hz is
    # buzz the servo has to physically produce.
    sig = a - a.mean(axis=0, keepdims=True)
    freqs = np.fft.rfftfreq(sig.shape[0], d=env.dt)
    power = (np.abs(np.fft.rfft(sig, axis=0)) ** 2).sum(axis=(1, 2))
    total = power.sum() or 1.0
    bands = {
        "<1.5Hz": power[freqs < 1.5].sum() / total,
        "1.5-5Hz": power[(freqs >= 1.5) & (freqs < 5.0)].sum() / total,
        ">5Hz": power[freqs >= 5.0].sum() / total,
    }
    return {
        "ckpt": ckpt,
        "sat_frac": float((pooled > limit_deg).mean()),
        "rate_median": float(np.median(pooled)),
        "rate_p95": float(np.percentile(pooled, 95)),
        "limit_deg": limit_deg,
        "fwd": float(np.mean(fwd)),
        "command": command[0],
        "resets": resets,
        "survival": survival,
        "cadence_hz": cadence,
        "excess_reversals": excess_reversals,
        "amplitude_deg": float(np.degrees((a.max(axis=0) - a.min(axis=0)).mean() * scale)),
        "bands": bands,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("-e", "--exp-name", action="append", required=True,
                   help="Experiment name under logs/; repeat to compare checkpoints")
    p.add_argument("--ckpt", type=int, default=None, help="default: highest available")
    p.add_argument("--duration", type=float, default=20.0, metavar="S")
    # Was 4, which is right for saturation -- the policy is deterministic and 4 envs already
    # give 32k joint-ticks -- but useless for SURVIVAL, which 4 episodes can only resolve to
    # the nearest 0.25. The first version of the survival gate duly reported 0.250 off three
    # terminations and would have failed a policy on noise. Stepping is cheap here (the wall
    # time is model loading), so pay for episodes.
    p.add_argument("--num-envs", type=int, default=64,
                   help="Averaged over; more envs is more terrain/spawn variety, not more noise "
                        "(the policy is deterministic). Survival needs >=32 to mean anything.")
    p.add_argument("--seed", type=int, default=0,
                   help="Evaluation seed: spawn positions and per-env physics draws. The "
                        "POLICY is deterministic, so varying this separates how much of a "
                        "reading belongs to the checkpoint from how much belongs to the "
                        "evaluation -- the difference between a policy that visits 0.3% "
                        "saturation and a lucky rollout of one that does not.")
    p.add_argument("--command", type=float, nargs=3, default=None, metavar=("VX", "VY", "WZ"))
    p.add_argument("--flat", action="store_true",
                   help="Evaluate on flat ground regardless of what the run trained on. THE "
                        "way to compare speed across runs -- and the deployment condition.")
    p.add_argument("--terrain-amplitude", type=float, default=None, metavar="M")
    p.add_argument("--terrain-correlation", type=float, default=None, metavar="M")
    p.add_argument("--domain-rand", action="store_true",
                   help="Keep pushes/noise/physics randomization on. Off by default: this "
                        "measures what the policy COMMANDS, and disturbance rejection is a "
                        "different question.")
    p.add_argument("--wire-quantize-deg", type=float, default=None, metavar="DEG",
                   help="Override the plant-path wire quantization. Default: whatever the run "
                        "trained with (0 for any cfgs.pkl predating it), so the checkpoint is "
                        "judged on the plant it learned. Set it to compare plants deliberately.")
    args = p.parse_args()

    print(f"{'run':<26} {'ckpt':>5} {'sat':>7} {'med':>6} {'p95':>6} {'amp':>6} "
          f"{'fwd':>7} {'revs':>6} {'<1.5':>6} {'1.5-5':>6} {'>5Hz':>6}")
    for name in args.exp_name:
        r = evaluate(name, args.ckpt, args.duration, args.domain_rand, args.command,
                     args.num_envs, args.wire_quantize_deg,
                     terrain=(False if args.flat else None),
                     terrain_amplitude=args.terrain_amplitude,
                     terrain_correlation=args.terrain_correlation, seed=args.seed)
        b = r["bands"]
        print(f"{name:<26} {r['ckpt']:>5} {100*r['sat_frac']:>6.1f}% "
              f"{r['rate_median']:>6.0f} {r['rate_p95']:>6.0f} {r['amplitude_deg']:>5.1f}d "
              f"{r['fwd']:>+7.3f} {r['excess_reversals']:>5.2f}x "
              f"{100*b['<1.5Hz']:>5.0f}% {100*b['1.5-5Hz']:>5.0f}% "
              f"{100*b['>5Hz']:>5.0f}%")
        if args.num_envs < 32:
            print(f"{'':<26}   ! survival over {args.num_envs} episodes is not a measurement; "
                  f"use --num-envs 64 or more.")
        smooth_ok = r["sat_frac"] <= TRANSFER_THRESHOLD
        stands_up = r["survival"] >= SURVIVAL_FLOOR
        verdict = "DEPLOYABLE" if (smooth_ok and stands_up) else "DO NOT DEPLOY"
        print(f"{'':<26} {verdict}: {100*r['sat_frac']:.1f}% of joint-ticks command faster "
              f"than {r['limit_deg']:.0f} deg/s (threshold {100*TRANSFER_THRESHOLD:.0f}%); "
              f"survival {r['survival']:.3f} on its own terrain "
              f"({r['resets']} termination(s)), commanded {r['command']:.3f} m/s")
        if not stands_up:
            print(f"{'':<26}   ^ survival below {SURVIVAL_FLOOR:.2f}: it falls more often than "
                  f"it walks. Saturation says nothing about this.")
        elif r["survival"] < SURVIVAL_WARN:
            print(f"{'':<26}   note: survival {r['survival']:.3f} < {SURVIVAL_WARN:.2f}. Expected "
                  f"on rough terrain, but it is the number that predicts a fall on hardware.")


if __name__ == "__main__":
    main()
