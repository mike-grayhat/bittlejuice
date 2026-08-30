"""PPO training entrypoint for MjVecEnv (MuJoCo C physics on CPU threads, rsl_rl's PPO).

The task itself is not defined here: get_cfgs()/get_train_cfg() come from config.py, which the
test suite reads too, so a hyperparameter cannot drift away from the test that guards it. What
this file owns is the CLI, the runner wiring, and the checkpoint/resume plumbing.

Why not Genesis: at 512 envs the Genesis/Metal pipeline sustained ~7,700 env-steps/s with 91% of
wall time in collection; this env measures ~157,000 env-steps/s on the same machine, taking a
PPO iteration's collection from 1.45 s to 0.078 s. See mj_vec_env.py's module docstring. That
pipeline was deleted once this one had superseded it; `git log -- bittle_env.py` has it.

Usage:
    uv run python sim/mj_train.py -e my-run --num-envs 512 --max-iterations 5000
"""

import argparse
import json
import pathlib
import shlex
import subprocess
import sys
import os
import pickle
import shutil

import torch
from rsl_rl.runners import OnPolicyRunner

import train_config
from config import get_cfgs, get_train_cfg
from mj_vec_env import MjVecEnv


def _write_run_manifest(args, log_dir, env_cfg, obs_cfg, reward_cfg, command_cfg, train_cfg):
    """Record everything needed to reproduce this run, in a VERSIONED directory.

    logs/ is gitignored -- checkpoints are large and regenerable -- so until now a run's
    provenance lived on exactly one disk. That is fine right up until a result is built on
    a checkpoint (`--resume-from`), at which point the result is unreproducible in
    principle, not merely inconvenient. The manifest is a few KB of text and closes that.

    JSON rather than the pickle next to the checkpoint, because the point is to DIFF runs:
    `git diff runs/a/config.json runs/b/config.json` answers "what actually changed between
    these two" in one command. Reconstructing that from two pickles is not practical, and a
    multi-stage finetuning lineage becomes unattributable without it.

    The command line is recorded too. cfgs.pkl captures the resulting config, which is
    nearly equivalent -- but not for `--resume-from`, whose value appears nowhere in the
    config it produces.
    """
    root = pathlib.Path(__file__).resolve().parents[1]
    out = root / "runs" / args.exp_name
    out.mkdir(parents=True, exist_ok=True)

    (out / "command.txt").write_text(
        "# rerun with:\n" + " ".join(shlex.quote(a) for a in sys.argv) + "\n")
    shutil.copy(out / "command.txt", pathlib.Path(log_dir) / "command.txt")

    # Copy the input config verbatim. command.txt names the path, but configs/ is tracked
    # and therefore edited: six months on, the file at that path is not necessarily what
    # this run read. The copy is the record; the original is the recipe.
    if getattr(args, "config", None):
        src = pathlib.Path(args.config)
        if src.is_file():
            shutil.copy(src, out / "config.input.yaml")

    def _plain(o):
        if isinstance(o, np.generic):
            return o.item()
        if isinstance(o, np.ndarray):
            return o.tolist()
        return repr(o)
    # The resolved flags, separately from the cfg dicts. get_cfgs() folds SOME args in
    # (terrain_amplitude, tracking_sigma, the command ranges) and not others: obs_history,
    # max_iterations, estimator, privileged_critic and penalty_curriculum reach the run
    # without ever appearing in a cfg. Diffing two config.json files therefore could not
    # show that one run had 12 frames of observation history and the other none, which is
    # exactly the comparison this directory exists to make.
    (out / "args.json").write_text(json.dumps(
        vars(args), indent=2, sort_keys=True, default=_plain) + "\n")

    (out / "config.json").write_text(json.dumps(
        {"env": env_cfg, "obs": obs_cfg, "reward": reward_cfg,
         "command": command_cfg, "train": train_cfg},
        indent=2, sort_keys=True, default=_plain) + "\n")

    try:
        head = subprocess.run(["git", "-C", str(root), "rev-parse", "HEAD"],
                              capture_output=True, text=True, timeout=10).stdout.strip()
        # Exclude meshes and binaries: an uncommitted .obj file is 200k lines of vertex data
        # and zero provenance, and will dwarf the diff that matters.
        diff = subprocess.run(
            ["git", "-C", str(root), "diff", "HEAD", "--",
             ".", ":(exclude)*.obj", ":(exclude)*.stl", ":(exclude)*.npz",
             ":(exclude)*.pt", ":(exclude)*.png", ":(exclude)*.jpg"],
            capture_output=True, text=True, timeout=30).stdout
    except Exception as e:                       # a manifest is not worth failing a run over
        head, diff = f"(unavailable: {e})", ""
    (out / "git.txt").write_text(
        f"commit {head}\ndirty   {'yes' if diff.strip() else 'no'}\n")
    if diff.strip():
        (out / "uncommitted.patch").write_text(diff)
    print(f"run manifest: {out}")


def main():
    # BooleanOptionalAction, not store_true, for every boolean below except --no-domain-rand
    # (already negative). A config file sets these as defaults, and store_true is one-way:
    # with `terrain: true` in configs/base.yaml there would be no way to turn it off from the
    # command line, so an experiment that needs flat ground could not be expressed as an
    # override at all -- only by writing a second config. --no-terrain costs nothing and
    # keeps every existing invocation working unchanged.
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-e", "--exp-name", type=str, default="bittle-mj",
        help="Experiment name; also the log directory under logs/",
    )
    parser.add_argument("-b", "--num-envs", type=int, default=512, help="Number of parallel environments")
    parser.add_argument("--max-iterations", type=int, default=101, help="Number of learning iterations")
    parser.add_argument("--seed", type=int, default=1, help="Random seed")
    parser.add_argument(
        "--num-threads", type=int, default=None,
        help="MuJoCo rollout worker threads (default: os.cpu_count())",
    )
    parser.add_argument(
        "--device", type=str, default=None, choices=["cpu", "mps"],
        help="Device for the PPO networks (default: mps when available, else cpu). Physics is "
             "always CPU. With collection down to ~0.2s/iteration the PPO update is now the "
             "larger term, and mps roughly halves it (0.274s -> 0.136s at 512 envs) -- worth more "
             "than the per-step host<->device transfer of the (num_envs, 33) observation costs.",
    )
    parser.add_argument(
        "--no-domain-rand", action="store_true", help="Disable sim-to-real domain randomization (debugging)"
    )
    # --- partial-observability knobs -------------------------------------------------
    # The actor's 33 inputs contain only 6 real measurements (ang_vel, projected_gravity);
    # the joint channels are its own commands played back through a servo model. These three
    # flags attack that from different sides and are deliberately independent so each can be
    # measured on its own.
    parser.add_argument(
        "--privileged-critic", action=argparse.BooleanOptionalAction, default=False,
        help="Give the CRITIC true base velocity, joint state, servo tracking error, foot "
             "contacts and this episode's latent physics. Actor untouched, so nothing changes "
             "at deploy -- the critic is discarded after training. Free accuracy on the value "
             "function, which is what the advantage estimate is built from.",
    )
    parser.add_argument(
        "--obs-history", type=int, default=0, metavar="N",
        help="Feed the actor the previous N observation frames alongside the current one. With "
             "the joint channel open-loop, a footfall exists only as an IMU transient, and a "
             "single-frame MLP cannot compute a difference. Deployable: deploy needs only a "
             "ring buffer. Try 4-8.",
    )
    parser.add_argument(
        "--recurrent", action=argparse.BooleanOptionalAction, default=False,
        help="Use a GRU actor+critic instead of a plain MLP. More expressive than --obs-history "
             "for the same problem, but deployment needs a numpy GRU in deploy. Combining "
             "it with --obs-history is redundant; pick one.",
    )
    parser.add_argument(
        "--rnn-hidden-dim", type=int, default=128,
        help="GRU hidden size when --recurrent is set.",
    )
    parser.add_argument(
        "--action-latency-steps", type=int, nargs=2, default=None, metavar=("LO", "HI"),
        help="Per-episode action delay in 20 ms control steps, sampled uniformly in [LO,HI]. "
             "Unset keeps the default 1-2 step model, which is right for the patched "
             "firmware: it measures 22 ms of command->joint dead time against the model's "
             "~21.5 ms mean. Pass this flag only to sweep deliberately.\n"
             "Cost of dead time, measured with domain rand on, two seeds:\n"
             "    dead time      0 ms   20 ms   20-60(now)  60-120  120-200  200-320\n"
             "    fwd m/s       .091    .090    .081         .064     .048     .035\n"
             "Unpatched firmware puts the robot at ~198 ms, in the 120-200 bucket; patched, "
             "it sits near the top of the curve. Training against MORE dead time than the "
             "robot has costs forward speed for nothing, so do not widen this without a "
             "measurement. Note the cost only appears on a correctly fitted actuator model -- "
             "a too-sluggish plant masks dead time and makes latency look free.",
    )
    parser.add_argument(
        "--observe-true-dof", action=argparse.BooleanOptionalAction, default=False,
        help="Observe the simulator's true dof_pos/dof_vel instead of the servo-model estimate "
             "deploy actually feeds the policy. It hands the policy proprioception the robot "
             "does not have: measured on a trained "
             "policy, swapping to the real signal took gait cadence 1.17 -> 4.65 Hz and forward "
             "speed +0.079 -> -0.020 m/s. For A/B only.",
    )
    parser.add_argument(
        "--fixed-command", action=argparse.BooleanOptionalAction, default=False,
        help="Use the original degenerate fixed lin_vel_x=[0.1,0.1] command instead of the widened "
             "[0.0,0.22] range (debugging)",
    )
    # --- closed-loop knobs ------------------------------------------------------------
    # These four exist to answer one measured finding: the policies are ~99% open-loop and
    # ignore the IMU entirely (see mj_vec_env's blindfold measurements). Each removes one
    # reason the environment let them get away with it. They are meant to be used TOGETHER --
    # in particular pushes without --termination-deg just convert disturbance into episode
    # death, which is the trap the 10 deg default was already in.
    parser.add_argument(
        "--terrain", action=argparse.BooleanOptionalAction, default=False,
        help="Walk on a correlated heightfield instead of a flat plane. Adds TRANSIENT "
             "per-footfall disturbance. The field sits on top of the existing floor, so "
             "walking off its edge steps down rather than falling out of the world.",
    )
    parser.add_argument(
        "--terrain-curriculum", action=argparse.BooleanOptionalAction, default=False,
        help="Per-environment terrain difficulty, promoted and demoted on achieved distance "
             "(Rudin et al. 2021). --terrain-amplitude becomes the HARDEST rung rather than "
             "the only one, and every env starts on the easiest. Watch terrain_level: pinned "
             "at 0 means the distance test is never passing, pinned at the top means it is "
             "too easy to pass.",
    )
    parser.add_argument(
        "--terrain-curriculum-levels", type=int, default=None, metavar="N",
        help="Rungs in the terrain ladder (default 10).",
    )
    parser.add_argument(
        "--curriculum-max-rise", type=float, default=None, metavar="R",
        help="Damp the penalty curriculum by capping how fast its factor may RISE, per "
             "update call (try 0.0005). Default 0 is undamped, which limit-cycles: the "
             "factor swings across most of its range every few tens of iterations, and more "
             "penalty pressure makes it worse (std 0.133 with no power/symmetry term, 0.249 "
             "with both). The fall is never capped -- slow engage, fast release.",
    )
    parser.add_argument(
        "--terrain-promote-frac", type=float, default=None, metavar="F",
        help="Fraction of commanded distance that promotes (default 0.60). Must be REACHABLE: "
             "at Rudin's 0.80 this robot never promotes and the ladder stalls near the "
             "bottom. Check achieved fwd_vel_mps against the command before changing it.",
    )
    parser.add_argument(
        "--terrain-demote-frac", type=float, default=None, metavar="F",
        help="Fraction of commanded distance below which an env is demoted (default 0.35).",
    )
    parser.add_argument(
        "--terrain-amplitude", type=float, default=None, metavar="M",
        help="Peak terrain height [m] (default 0.008). Scaled to Bittle, not to a Go2: 8 mm is "
             "~9%% of the 94 mm stand height, where a legged_gym-typical 5 cm would be taller "
             "than the whole robot.",
    )
    parser.add_argument(
        "--slope-deg", type=float, default=None, metavar="DEG",
        help="Max per-episode ground incline (default 8). Adds SUSTAINED disturbance, and it "
             "is the term that most directly forces the policy to read projected_gravity: a "
             "slope is invisible to every other channel it has.",
    )
    parser.add_argument(
        "--push-dv-xy", type=float, default=None, metavar="MS",
        help="Peak horizontal velocity kick [m/s] (default 0.10, ~1x walking speed). Size it "
             "against the REWARD's width, not against walking speed -- see mj_vec_env's note.",
    )
    parser.add_argument(
        "--lipschitz-coef", type=float, default=0.0, metavar="LAMBDA",
        help="LCP gradient penalty (arXiv:2410.11825): lambda * E[||grad_s log pi(a|s)||^2], "
             "added as a separate optimiser step. Paper uses 0.002. Requires --estimator "
             "(the penalty lives in PPOWithEstimator).\n"
             "Why: every smoothness term in reward_scales is TEMPORAL -- action now vs action "
             "a step ago. None ask whether a small change in OBSERVATION produces a small "
             "change in action, which is how sensor noise becomes action jitter. Measured "
             "here: raising joint_reversal drove excess reversals 1.01x -> 0.74x and "
             "saturation 9.1%% -> 24.9%%. Four interacting hand-tuned penalties, and pushing "
             "one distorts the others.")
    parser.add_argument(
        "--gait-phase", action=argparse.BooleanOptionalAction, default=False,
        help="Append sin/cos of a gait clock plus the commanded cadence to the observation, "
             "and randomise cadence per episode over 0.60-1.15 Hz.\n"
             "Why: measured across four trained policies, the fraction of action power in the "
             "gait FUNDAMENTAL tracks perceived smoothness better than any rate statistic: "
             "69%% of band power reads visibly smoother on hardware than 47%%, even when the "
             "latter wins every commanded-rate metric. Power outside the "
             "fundamental is off-beat motion, and a policy that must re-derive phase from its "
             "own history every tick has to produce it. The ceiling is the SERVO: at 34.5 deg "
             "peak-to-peak the fundamental alone hits the 137 deg/s slew limit at 1.26 Hz.")
    parser.add_argument(
        "--terrain-correlation", type=float, default=None, metavar="M",
        help="Terrain feature size [m] (default 0.10). THE difficulty lever, more than "
             "amplitude. Measured against this robot: at the default 0.10 m -- "
             "about one body length -- 20 mm amplitude gives 2.5 mm RMS, a 4.8 mm rise per "
             "83 mm stride against 8.1 mm of measured foot clearance, and 2.6 deg of body "
             "tilt. That is gentle rolling ground, and raising amplitude on it just tilts "
             "the whole robot more slowly, which the policies demonstrably ignore (blindfold "
             "reads -1.2 to -2.4%% on the slope condition).\n"
             "Halving it to 0.05 m brings feature size down toward stride length and roughly "
             "doubles local grade at the same amplitude, so each footfall becomes its own "
             "disturbance -- the structure IMU feedback can actually exploit.")
    parser.add_argument(
        "--penalty-curriculum", action=argparse.BooleanOptionalAction, default=False,
        help="Ramp the smoothness penalties in as the policy earns them, instead of applying "
             "them from iteration 0. Scales action_jerk/action_slew/feet_slip/feet_stuck by "
             "clip((ema_episode_length - 600) / 300, 0, 1).\n"
             "Why: those four are measured to make from-scratch training impossible -- with "
             "them on a fresh policy holds 18/1000 episode length indefinitely, with them off "
             "it reaches 1001 by iteration 10. This lets one run do both jobs. It is feedback "
             "rather than a schedule, so if the penalties ever start costing episodes the "
             "factor backs off on its own.")
    parser.add_argument(
        "--base-init-tilt-deg", type=float, default=None, metavar="DEG",
        help="Peak random roll/pitch the episode STARTS at [deg] (default 20). This is a "
             "recovery demand -- it is what makes the IMU necessary rather than merely "
             "useful -- and it belongs on a policy that already walks.\n"
             "PASS 0 WHEN TRAINING FROM SCRATCH. A random policy starting up to 20 deg from "
             "upright is already halfway to a fall before it acts, so it never experiences "
             "walking and gets no gradient toward it. Measured: a fresh run with the "
             "defaults sits at 25/1000 mean episode length for 1750 iterations, flat. A "
             "from-scratch run wants no initial tilt, no pushes and no slope. See get_cfgs().")
    parser.add_argument(
        "--push-dw-rp", type=float, default=None, metavar="RAD_S",
        help="Peak angular (roll/pitch) velocity kick [rad/s] (default 10). THE disturbance "
             "the IMU can see -- a linear push on an upright, non-rotating body is invisible "
             "to every channel the actor has. Size it against episode survival, not against "
             "walking speed; see get_cfgs().",
    )
    parser.add_argument(
        "--symmetry-coef", type=float, default=0.0, metavar="LAMBDA",
        help="Mirror-symmetry loss: penalise ||pi(s) - M_a(pi(M_s(s)))||^2, so the mirrored "
             "world produces the mirrored action (Yu et al. arXiv:1801.08093). 0 disables. "
             "Trained policies measure 45-48%% residual against their own action magnitude, "
             "so there is real headroom. Pairs naturally with --reward-scale power=...: "
             "energy minimisation on its own buys asymmetric gaits.",
    )
    parser.add_argument(
        "--termination-deg", type=float, default=None, metavar="DEG",
        help="Roll/pitch termination threshold (default 45, raised from 10). THE key line: at "
             "10 deg any disturbance needing recovery ended the episode, so not deviating beat "
             "recovering and the optimal policy was open-loop.",
    )
    parser.add_argument(
        "--heading", action=argparse.BooleanOptionalAction, default=False,
        help="Command a HEADING rather than a yaw rate. The yaw-rate command becomes the output "
             "of a P-controller on heading error, which the policy already observes and is "
             "already rewarded for tracking -- so this adds no observation channel. Fixes the "
             "structural blindness that projected_gravity is yaw-invariant: today a push that "
             "rotates the robot is neither visible nor penalised, and heading error is the "
             "dominant term in where it ends up because steering compounds.",
    )
    parser.add_argument(
        "--ang-vel-range", type=float, nargs=2, default=[-0.6, 0.6], metavar=("LO", "HI"),
        help="Yaw-rate command range [rad/s], used with --heading. The heading controller's "
             "output is clipped to this, so it also bounds how sharply the robot may be asked "
             "to turn. 0.6 rad/s is ~34 deg/s.",
    )
    parser.add_argument(
        "--estimator", action=argparse.BooleanOptionalAction, default=False,
        help="Train a supervised state estimator alongside PPO and feed its base_lin_vel "
             "estimate into the actor. This is the answer to the measured finding that the "
             "policies are open-loop: the actor has NO velocity sense -- a steady drift with "
             "the body upright and unrotating produces zero signal in every channel it has -- "
             "yet tracking_lin_vel scores it on exactly that. Requires --obs-history; the "
             "estimator needs a transient to work from. See ppo_estimator.py.",
    )
    parser.add_argument(
        "--estimator-lr", type=float, default=1e-3,
        help="Learning rate for the estimator's own optimizer (separate from PPO's).",
    )
    parser.add_argument(
        "--resume-from", type=str, default=None, metavar="EXP:CKPT",
        help="Warm-start from another run's checkpoint, e.g. 'my-run:7900'. "
             "Worth it when the change is a LOCAL edit to the objective -- adding a shaping "
             "penalty, widening domain randomization -- because most of what the old policy "
             "learned still applies. Requires an identical observation width; runs with "
             "different obs layouts cannot be warm-started from each other.",
    )
    parser.add_argument(
        "--reset-action-std", type=float, default=None, metavar="STD",
        help="Re-inflate the policy's exploration noise when resuming. THE thing that makes or "
             "breaks a fine-tune: std anneals to ~0.4 by convergence, so a warm start explores "
             "only the immediate neighbourhood and finds the nearest variant of the old gait "
             "rather than a genuinely different one. Try 0.7-1.0 if the change needs the policy "
             "to restructure rather than merely adjust.",
    )
    parser.add_argument(
        "--entropy-coef", type=float, default=0.01,
        help="PPO entropy bonus coefficient; lower for domain-rand runs (see get_train_cfg's comment)",
    )
    parser.add_argument(
        "--desired-kl", type=float, default=0.01,
        help="Target KL per update for the adaptive schedule. THE SETTLING LEVER, and the "
             "reason it is worth exposing: schedule='adaptive' is a controller, not a decay -- "
             "it moves the learning rate to hold each update at this KL, so the policy takes "
             "an equal-sized step forever and never converges. Measured: the learning rate "
             "sits flat at ~5e-5 for thousands of iterations while reward plateaus, and "
             "deployment quality still swings widely across late checkpoints. A smaller "
             "value makes the run settle instead of wandering the plateau.",
    )
    # --- speed-shaping knobs, for sweeping without touching the shared get_cfgs() ---
    parser.add_argument(
        "--command-vx", type=float, nargs="+", default=None, metavar="MPS",
        help="Forward speed command [m/s]. ONE value trains at that fixed speed; TWO values "
             "give a range sampled per episode.\n"
             "A single value means the policy can walk at exactly one speed -- "
             "tracking_lin_vel, the dominant reward term, pays "
             "only for hitting the one commanded value. A policy cannot slow down for rough "
             "ground when the objective forbids it. A range is the prerequisite for adapting "
             "gait to terrain, and for commanding a slower walk at deploy time.",
    )
    parser.add_argument(
        "--tracking-sigma", type=float, default=None,
        help="tracking_lin_vel sigma. get_cfgs' comment sets this to command^2 so the "
             "standing-still reward ratio stays exp(-1); raising the command without raising this "
             "makes the tracking reward nearly zero until the robot is already moving.",
    )
    parser.add_argument(
        "--tracking-sigma-ang", type=float, default=None,
        help="Separate sigma for tracking_ang_vel. Without this it shares tracking_sigma, so "
             "raising the latter for a faster linear command silently loosens the yaw-rate penalty "
             "by the same factor (measured: 0.01 -> 0.04 took residual yaw 0.19 -> 1.41 deg/s).",
    )
    parser.add_argument(
        "--action-scale", type=float, default=None,
        help="rad of joint target per unit action. This is what caps stride length: measured on "
             "this geometry, +-0.25 rad of shoulder swing moves the toe 43mm fore/aft, +-0.50 rad "
             "moves it 84mm. Raising it also amplifies exploration noise early in training.",
    )
    parser.add_argument(
        "--feet-air-time-threshold", type=float, default=None,
        help="Swing duration [s] beyond which feet_air_time pays. Any swing SHORTER than this "
             "scores negative, so at ~50%% duty it caps stride frequency at 1/(2*thr) Hz: the "
             "default 0.15 caps ~3.3Hz, i.e. roughly 0.17 m/s at a 50mm stride.",
    )
    parser.add_argument(
        "--clip-actions", type=float, default=None,
        help="bound on |action| before scaling. Default 3.0, which at action_scale 0.25 bounds "
             "commands to +-43 deg from the default pose -- the whole usable range of this "
             "geometry. Raising it towards the old 100 is effectively removing the bound, which "
             "lets the policy command far outside a joint's range for free (the joint limit "
             "clamps it in sim at no cost) and produced a hardware TREMBLE; see get_cfgs().",
    )
    parser.add_argument(
        "--wire-quantize-deg", type=float, default=None, metavar="DEG",
        help="Quantization of the joint target on the plant path, modelling the integer-degree "
             "serial wire (default 1.0; 0 disables). The host's servo-model observer stays "
             "unquantized, as it is on the robot, so this also reproduces the resulting "
             "observer/plant divergence across 16 of the 33 obs channels.",
    )
    parser.add_argument(
        "--config", type=str, default=None, metavar="PATH",
        help="YAML config supplying defaults for every flag below (configs/walk.yaml). Any "
             "flag given on the command line still wins, which is what keeps a four-seed "
             "pool as one config plus --seed $S. See sim/train_config.py.",
    )
    parser.add_argument(
        "--reward-scale", action="append", default=[], metavar="NAME=VALUE",
        help="Override one reward scale, repeatable (e.g. --reward-scale similar_to_default=-0.05). "
             "Overriding here rather than editing get_cfgs() keeps a one-off sweep from "
             "moving a default that config.py's tests assert against.",
    )
    # Load the config BEFORE parsing, as defaults, so the command line still overrides it.
    # A throwaway parser reads --config on its own; the real one has not run yet and its
    # defaults are exactly what we are about to replace.
    _pre = argparse.ArgumentParser(add_help=False)
    _pre.add_argument("--config", type=str, default=None)
    _cfg_path = _pre.parse_known_args()[0].config
    if _cfg_path:
        try:
            _cfg = train_config.load(
                _cfg_path, {a.dest for a in parser._actions} - {"help"})
        except train_config.ConfigError as e:
            raise SystemExit(f"config error:\n{e}")
        parser.set_defaults(**_cfg)

    args = parser.parse_args()

    device = args.device or ("mps" if torch.backends.mps.is_available() else "cpu")

    log_dir = f"logs/{args.exp_name}"
    env_cfg, obs_cfg, reward_cfg, command_cfg = get_cfgs()
    train_cfg = get_train_cfg(args.exp_name, entropy_coef=args.entropy_coef,
                              desired_kl=args.desired_kl)
    if args.no_domain_rand:
        env_cfg["domain_rand_enabled"] = False
    # Recorded in cfgs.pkl either way, so mj_eval and export_policy_mj reconstruct the exact
    # observation the checkpoint was trained against rather than today's default.
    env_cfg["observe_servo_estimate"] = not args.observe_true_dof
    env_cfg["obs_history_len"] = args.obs_history
    if args.action_latency_steps:
        env_cfg["action_latency_steps_range"] = list(args.action_latency_steps)

    # Wire the observation groups the env emits to the models that consume them. The env always
    # emits "privileged" and (when asked) "history"; these lines decide who sees what, so the
    # decision lives in cfgs.pkl next to the checkpoint rather than in env code.
    actor_groups = ["policy"] + (["history"] if args.obs_history > 0 else [])
    critic_groups = list(actor_groups) + (["privileged"] if args.privileged_critic else [])
    train_cfg["obs_groups"] = {"actor": actor_groups, "critic": critic_groups}
    if args.estimator:
        if args.obs_history <= 0:
            raise SystemExit(
                "--estimator needs --obs-history: velocity is not recoverable from a single "
                "frame. The only real measurements are ang_vel and projected_gravity, and a "
                "steady drift leaves both unchanged -- what carries velocity information is how "
                "they CHANGE across footfalls, which requires more than one frame to see."
            )
        # The estimator reads only groups the robot can actually produce. Handing it
        # "privileged" would train something undeployable that scores beautifully in sim.
        train_cfg["obs_groups"]["estimator"] = list(actor_groups)
        # ...and the actor gains the estimate as an input, which is what forces it to be used.
        # The CRITIC does not: it already has the true base_lin_vel in "privileged", and feeding
        # it an estimate of a quantity it can see exactly would only add noise.
        train_cfg["obs_groups"]["actor"] = actor_groups + ["estimate"]
        train_cfg["algorithm"]["class_name"] = "ppo_estimator:PPOWithEstimator"
        train_cfg["algorithm"]["estimator_cfg"] = {"learning_rate": args.estimator_lr}
        env_cfg["estimator_dim"] = 3   # base_lin_vel; see ppo_estimator.TARGET_SLICE
    # Both regularisers live in PPOWithEstimator, whose class_name is only selected above
    # under --estimator. Without that, these flags set a key that plain PPO never reads and
    # the run trains as if they were absent -- a silent no-op that looks exactly like "the
    # regulariser did not help". Refuse loudly instead. (LCP has carried this since it was
    # added; no run is affected, because every run that passed --lipschitz-coef also passed
    # --estimator, but the next one would not have been so lucky.)
    for flag, value in (("--lipschitz-coef", args.lipschitz_coef),
                        ("--symmetry-coef", args.symmetry_coef)):
        if value and not args.estimator:
            raise SystemExit(
                f"{flag} is implemented in ppo_estimator.PPOWithEstimator, which is only "
                f"selected by --estimator. Without it the coefficient is set on a config "
                f"plain PPO never reads, and the run trains as though you had not passed it. "
                f"Add --estimator, or drop {flag}.")
    if args.lipschitz_coef:
        train_cfg["algorithm"]["lipschitz_coef"] = args.lipschitz_coef
    if args.symmetry_coef:
        train_cfg["algorithm"]["symmetry_coef"] = args.symmetry_coef
    if args.recurrent:
        if args.obs_history > 0:
            raise SystemExit(
                "--recurrent and --obs-history solve the same problem twice: the GRU already "
                "carries state across steps, so stacking frames only widens its input. Pick one."
            )
        for role in ("actor", "critic"):
            train_cfg[role]["class_name"] = "RNNModel"
            train_cfg[role]["rnn_type"] = "gru"
            train_cfg[role]["rnn_hidden_dim"] = args.rnn_hidden_dim
            train_cfg[role]["rnn_num_layers"] = 1
    if args.fixed_command:
        command_cfg["lin_vel_x_range"] = [0.1, 0.1]
    if args.command_vx is not None:
        v = args.command_vx
        if len(v) == 1:
            command_cfg["lin_vel_x_range"] = [v[0], v[0]]
        elif len(v) == 2:
            command_cfg["lin_vel_x_range"] = [min(v), max(v)]
        else:
            parser.error("--command-vx takes one value (fixed) or two (a range)")
    if args.tracking_sigma is not None:
        reward_cfg["tracking_sigma"] = args.tracking_sigma
    if args.tracking_sigma_ang is not None:
        reward_cfg["tracking_sigma_ang"] = args.tracking_sigma_ang
    if args.feet_air_time_threshold is not None:
        env_cfg["feet_air_time_threshold"] = args.feet_air_time_threshold
    if args.action_scale is not None:
        env_cfg["action_scale"] = args.action_scale
    if args.clip_actions is not None:
        env_cfg["clip_actions"] = args.clip_actions
    if args.wire_quantize_deg is not None:
        env_cfg["wire_quantize_deg"] = args.wire_quantize_deg
    if args.push_dw_rp is not None:
        env_cfg["push_dw_rp"] = args.push_dw_rp
    if args.base_init_tilt_deg is not None:
        env_cfg["base_init_tilt_deg"] = args.base_init_tilt_deg
    if args.penalty_curriculum:
        env_cfg["penalty_curriculum"] = True
    if args.terrain_correlation is not None:
        env_cfg["terrain_correlation"] = args.terrain_correlation
    if args.gait_phase:
        env_cfg["gait_phase"] = True
    for item in args.reward_scale:
        name, _, value = item.partition("=")
        if name not in reward_cfg["reward_scales"]:
            raise SystemExit(
                f"unknown reward scale {name!r}; choices: "
                f"{sorted(reward_cfg['reward_scales'])}"
            )
        reward_cfg["reward_scales"][name] = float(value)

    # MjVecEnv builds its own MJCF heightfield.
    if args.terrain:
        env_cfg["terrain_enabled"] = True
    if args.terrain_amplitude is not None:
        env_cfg["terrain_amplitude"] = args.terrain_amplitude
    if args.terrain_curriculum:
        env_cfg["terrain_curriculum"] = True
    if args.terrain_curriculum_levels is not None:
        env_cfg["terrain_curriculum_levels"] = args.terrain_curriculum_levels
    if args.curriculum_max_rise is not None:
        env_cfg["curriculum_max_rise"] = args.curriculum_max_rise
    if args.terrain_promote_frac is not None:
        env_cfg["terrain_promote_frac"] = args.terrain_promote_frac
    if args.terrain_demote_frac is not None:
        env_cfg["terrain_demote_frac"] = args.terrain_demote_frac
    if args.slope_deg is not None:
        env_cfg["slope_deg_range"] = [0.0, args.slope_deg]
    if args.push_dv_xy is not None:
        env_cfg["push_dv_xy"] = args.push_dv_xy
    if args.heading:
        env_cfg["heading_command"] = True
        # tracking_ang_vel shares tracking_sigma unless told otherwise, and the fixed-command
        # runs drive that down to 0.0036 for a 0.10 m/s LINEAR command. Opening ang_vel_range to
        # +-0.6 rad/s against that sigma makes the angular reward exp(-0.36/0.0036) = 4e-44 --
        # identically zero, unreachable, and invisible except as a term that never pays. Set
        # it from the yaw command, the same command^2 convention tracking_sigma follows,
        # unless the caller overrides.
        if args.tracking_sigma_ang is None:
            reward_cfg["tracking_sigma_ang"] = float(max(abs(v) for v in args.ang_vel_range)) ** 2
        # get_cfgs ships ang_vel_range [0,0] ("walk straight"), and the heading controller is
        # clipped to it -- so heading mode without this is a silent no-op. Opened here rather
        # than left to the caller because forgetting it produces a run that trains fine and
        # measures no benefit, which is indistinguishable from the feature not working.
        command_cfg["ang_vel_range"] = list(args.ang_vel_range)
    if args.termination_deg is not None:
        env_cfg["termination_if_roll_greater_than"] = args.termination_deg
        env_cfg["termination_if_pitch_greater_than"] = args.termination_deg

    # logs/shipped/ is the one run directory that is TRACKED: it is the checkpoint the README
    # tells a fresh clone to watch, and it is the only reason sim eval works without training
    # first. The rmtree below is silent, so training into that name would delete it and the
    # first symptom would be the README's opening command failing.
    if os.path.basename(log_dir) == "shipped":
        raise SystemExit(
            "-e shipped is reserved: logs/shipped/ is the tracked checkpoint the README "
            "ships with, and starting a run there would delete it. Pick another name."
        )
    if os.path.exists(log_dir):
        shutil.rmtree(log_dir)
    os.makedirs(log_dir, exist_ok=True)

    with open(f"{log_dir}/cfgs.pkl", "wb") as f:
        pickle.dump([env_cfg, obs_cfg, reward_cfg, command_cfg, train_cfg], f)
    _write_run_manifest(args, log_dir, env_cfg, obs_cfg, reward_cfg, command_cfg, train_cfg)

    torch.manual_seed(args.seed)

    env = MjVecEnv(
        num_envs=args.num_envs,
        env_cfg=env_cfg,
        obs_cfg=obs_cfg,
        reward_cfg=reward_cfg,
        command_cfg=command_cfg,
        device=device,
        num_threads=args.num_threads,
        seed=args.seed,
    )
    print(f"MjVecEnv: {env.num_envs} envs on {env.num_threads} threads, PPO on {env.device}")

    runner = OnPolicyRunner(env, train_cfg, log_dir, device=device)
    if args.symmetry_coef:
        # Hand the algorithm the env so the symmetry loss can read the penalty curriculum's
        # factor. Set explicitly rather than plumbed through construct_algorithm: it is one
        # reference, and an unset one falls back to full strength rather than failing quietly.
        runner.alg.symmetry_env = env
    if args.resume_from:
        src_exp, _, src_ckpt = args.resume_from.partition(":")
        src = f"logs/{src_exp}/model_{src_ckpt}.pt"
        # Skip the optimizer: its Adam moments were accumulated against the OLD reward, and
        # carrying them in biases the first updates toward the objective we just changed.
        runner.load(src, load_cfg={"actor": True, "critic": True, "optimizer": False,
                                   "iteration": False, "rnd": False}, strict=False)
        print(f"warm-started from {src} (weights only, optimizer reset)")
        if args.reset_action_std is not None:
            with torch.no_grad():
                for m in runner.alg.actor.modules():
                    if hasattr(m, "std_param"):
                        m.std_param.fill_(args.reset_action_std)
            print(f"re-inflated action std to {args.reset_action_std}")
    try:
        runner.learn(num_learning_iterations=args.max_iterations, init_at_random_ep_len=True)
    finally:
        env.close()


if __name__ == "__main__":
    main()
