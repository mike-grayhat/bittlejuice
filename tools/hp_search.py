"""Random search over the training hyperparameters, ranked by policy_score.

WHAT IS IN THE SPACE. Every entry is something measured to matter on this robot; none of it
is here because it is a knob.

  lipschitz_coef     The strongest smoothness lever, and cleanly monotonic once checkpoint
                     noise is averaged out. Composite score against lambda: 0.0005 -> 0.880
                     (1.2% saturation), 0.0010 -> 0.696 (5.2%), 0.0020 -> 0.374 (8.5%), and
                     disabling it entirely is far worse than any of them. 0.0005 is the EDGE
                     of what has been tried; the optimum may be below it.
  symmetry_coef      Mirror-symmetry loss. Trained policies measure 45-48% residual against
                     their own action magnitude -- nothing has ever asked them to be
                     symmetric. Pairs with power for the reason Yu et al. pair them: energy
                     minimisation on its own buys asymmetric gaits.
  power              sum |tau*qdot|. The only term that has produced a GAIT change rather
                     than a scaled version of one gait.
  action_jerk        Anti-correlated with an operator's smoothness verdicts, which is an
  action_slew        argument for searching them rather than trusting the current values.
  feet_stuck         The largest penalty in the recipe, ~39% of achieved tracking.
  tracking_sigma     Sets how sharply speed error is punished, and interacts with every
                     penalty above through the ratio each is sized against.
  obs_history        Measured to act as a CLOCK rather than a window, so the question is not
                     "how much history" but "how many frames does a metronome need" -- and a
                     shorter one is a smaller network on an ESP32.
  terrain_amplitude  Training terrain. 12 mm generalises to 40 mm at 0.891.
  entropy_coef       THE SETTLING LEVERS, and the most important two. Nothing in this setup
  desired_kl         converges on its own: schedule="adaptive" is a controller that holds
                     each update at desired_kl, so the learning rate sits flat for thousands
                     of iterations, while entropy_coef pays the policy to stay noisy. Reward
                     plateaus while the composite still swings widely across late checkpoints.
                     Weight averaging papers over that; these two remove it at the source.

WHAT IS DELIBERATELY NOT IN IT.

  terrain curriculum   Tested and settled: it loses to a fixed amplitude on this robot.
  command range        Buys a speed knob at a consistent cost in gait quality. Fixed here so
                       the search optimises one thing.
  gait phase           Inert -- the policy already derives a clock from its own history.

RUN LENGTH is deliberately short: the ranking signal is present at roughly half the
iterations a final run needs. Saturation keeps improving after that, so this UNDERSTATES
final quality, which is fine for ranking. Re-run the top few at full length before believing
them.

BEFORE TRUSTING ANY RANKING, run --calibrate. It trains the same config N times under
different seeds and reports the spread of the composite score. A search cannot resolve
differences smaller than that spread, and on this task the spread is large.
"""

import argparse
import json
import os
import subprocess
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)

SEARCH_SPACE = {
    "lipschitz_coef":    ("log",    1e-4,  3e-3),
    "symmetry_coef":     ("log",    0.05,  5.0),
    "power":             ("lin",    0.0,   0.08),
    "action_jerk":       ("log",    0.01,  0.06),
    "action_slew":       ("log",    0.002, 0.02),
    "feet_stuck":        ("log",    0.08,  0.25),
    "tracking_sigma":    ("log",    0.002, 0.008),
    "obs_history":       ("choice", [4, 8, 12]),
    "terrain_amplitude": ("lin",    0.008, 0.025),
    "entropy_coef":      ("log",    0.001, 0.010),
    "desired_kl":        ("log",    0.003, 0.020),
}

FIXED = [
    "--num-envs", "512", "--privileged-critic", "--estimator", "--penalty-curriculum",
    "--terrain", "--terrain-correlation", "0.05",
    "--heading", "--ang-vel-range", "-0.6", "0.6",
    "--command-vx", "0.1", "--tracking-sigma-ang", "0.36",
    "--reward-scale", "feet_slip=-0.005",
]

# The baseline recipe, used as the calibration config and the point of comparison.
BASELINE = dict(lipschitz_coef=0.0005, symmetry_coef=0.0, power=0.0,
                entropy_coef=0.01, desired_kl=0.01, action_jerk=0.03, action_slew=0.008,
                feet_stuck=0.15, tracking_sigma=0.0036, obs_history=12,
                terrain_amplitude=0.012)


def sample(rng):
    out = {}
    for name, spec in SEARCH_SPACE.items():
        kind = spec[0]
        if kind == "choice":
            out[name] = int(rng.choice(spec[1]))
        elif kind == "log":
            out[name] = float(np.exp(rng.uniform(np.log(spec[1]), np.log(spec[2]))))
        else:
            out[name] = float(rng.uniform(spec[1], spec[2]))
    return out


def to_argv(cfg, exp_name, iterations, seed):
    argv = [sys.executable, os.path.join(ROOT, "sim", "mj_train.py"),
            "-e", exp_name, "--max-iterations", str(iterations)] + FIXED
    argv += ["--lipschitz-coef", f"{cfg['lipschitz_coef']:.6g}"]
    if cfg.get("symmetry_coef"):
        argv += ["--symmetry-coef", f"{cfg['symmetry_coef']:.6g}"]
    argv += ["--terrain-amplitude", f"{cfg['terrain_amplitude']:.6g}"]
    argv += ["--tracking-sigma", f"{cfg['tracking_sigma']:.6g}"]
    argv += ["--obs-history", str(cfg["obs_history"])]
    argv += ["--entropy-coef", f"{cfg['entropy_coef']:.6g}"]
    argv += ["--desired-kl", f"{cfg['desired_kl']:.6g}"]
    for term in ("action_jerk", "action_slew", "feet_stuck", "power"):
        argv += ["--reward-scale", f"{term}=-{cfg[term]:.6g}"]
    if seed is not None:
        argv += ["--seed", str(seed)]
    return argv


AVG_CKPT = 99998


def train_and_score(cfg, exp_name, iterations, seed, k, dry_run):
    """Train one config, then score the WEIGHT-AVERAGED policy it produces.

    Scoring the average rather than a single checkpoint, or the mean of several checkpoints'
    metrics, for two reasons.

    It is what gets deployed. Averaging a run's last three checkpoints beats every
    individual one on slew saturation while holding perfect survival on the training terrain,
    and beats the best single checkpoint on held-out rougher ground. Ranking configs by
    something you would not ship optimises the wrong thing.

    And it removes the checkpoint lottery, which would otherwise dominate. Within one
    converged run the composite spans ~0.22 across late checkpoints, while the gap between
    two genuinely different configs can be an order of magnitude smaller. Comparing single
    checkpoints ranks configs almost entirely on which iteration each happened to stop at.

    Cheaper too: two evaluations of one averaged checkpoint instead of ten across five.
    """
    argv = to_argv(cfg, exp_name, iterations, seed)
    if dry_run:
        print("  " + " ".join(argv[2:]))
        return None
    subprocess.run(argv, cwd=ROOT, check=True)
    sys.path.insert(0, HERE)
    import average_checkpoints
    import policy_score

    log_dir = os.path.join(ROOT, "logs", exp_name)
    saved = sorted(int(f[6:-3]) for f in os.listdir(log_dir) if f.startswith("model_"))
    window = [c for c in saved if c != AVG_CKPT][-k:]
    average_checkpoints.average(log_dir, window,
                                os.path.join(log_dir, f"model_{AVG_CKPT}.pt"))
    orig = policy_score.late_checkpoints
    policy_score.late_checkpoints = lambda _e, _k: [AVG_CKPT]
    try:
        total, raw, _parts, gates = policy_score.score(
            exp_name, 1, 64, 0.10, 0.020, 0.05, verbose=False)
    finally:
        policy_score.late_checkpoints = orig
    return dict(score=total, gates=gates, averaged_from=window, **raw)


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("-n", "--num-configs", type=int, default=16)
    p.add_argument("--iterations", type=int, default=3000)
    p.add_argument("--checkpoints", type=int, default=3,
                   help="Late checkpoints to WEIGHT-AVERAGE into the policy that gets "
                        "scored. Three is the best window measured; wider drags in the bad "
                        "draws it is meant to average away.")
    p.add_argument("--seed", type=int, default=0, help="Sampler seed; reproduces the sweep.")
    p.add_argument("--prefix", default="hps")
    p.add_argument("--seed-offset", type=int, default=0,
                   help="First training seed for --calibrate, so a pool can be extended "
                        "without recomputing the seeds already run.")
    p.add_argument("--calibrate", type=int, default=0, metavar="N",
                   help="Instead of searching, train the baseline recipe N times under "
                        "different "
                        "TRAINING seeds and report the spread of the composite score. Run "
                        "this first: it is the smallest difference the search can resolve.")
    p.add_argument("--dry-run", action="store_true", help="Print the commands and stop.")
    a = p.parse_args()

    results, rng = [], np.random.default_rng(a.seed)
    if a.calibrate:
        print(f"CALIBRATION: baseline recipe x {a.calibrate} training seeds, "
              f"{a.iterations} iterations each")
        for k in range(a.calibrate):
            i = a.seed_offset + k
            name = f"{a.prefix}-cal-s{i}"
            r = train_and_score(BASELINE, name, a.iterations, i, a.checkpoints, a.dry_run)
            if r:
                results.append((r["score"], name, BASELINE, r))
                print(f"  seed {i}: score {r['score']:.3f}  sat {r['transfer']:.2f}%  "
                      f"revs {r['smoothness']:.2f}x  surv {r['robustness']:.3f}")
        if results:
            s = np.array([r[0] for r in results])
            print(f"\n  composite: mean {s.mean():.3f}  std {s.std():.3f}  "
                  f"range {s.min():.3f}-{s.max():.3f}")
            print(f"  MINIMUM RESOLVABLE DIFFERENCE ~ {2*s.std():.3f} (2 sigma). A search "
                  f"result closer than this to the baseline is noise.")
    else:
        for i in range(a.num_configs):
            cfg = sample(rng)
            name = f"{a.prefix}-{a.seed:02d}-{i:02d}"
            print(f"\n[{i+1}/{a.num_configs}] {name}")
            for k2, v in cfg.items():
                print(f"    {k2:20s} {v}")
            r = train_and_score(cfg, name, a.iterations, None, a.checkpoints, a.dry_run)
            if r:
                results.append((r["score"], name, cfg, r))
                print(f"    -> score {r['score']:.3f}"
                      + ("  GATED" if r["gates"] else ""))

    if results and not a.dry_run:
        results.sort(reverse=True, key=lambda r: r[0])
        out = os.path.join(ROOT, "logs", f"{a.prefix}-{a.seed:02d}-results.json")
        with open(out, "w") as f:
            json.dump([dict(score=s, name=n, config=c, metrics=m) for s, n, c, m in results],
                      f, indent=2)
        print(f"\n{'RANKING':30s} {'score':>7s} {'sat':>7s} {'revs':>7s} {'surv':>7s}")
        for s, n, _c, m in results:
            print(f"  {n:28s} {s:7.3f} {m['transfer']:6.2f}% {m['smoothness']:6.2f}x "
                  f"{m['robustness']:7.3f}" + ("  GATED" if m["gates"] else ""))
        print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
