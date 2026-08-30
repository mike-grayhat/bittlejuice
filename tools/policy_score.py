"""One number for "is this policy better", built to survive the noise it is measured in.

WHY A COMPOSITE. Every comparison in this project has been a table of four or five numbers
that move in different directions -- the best slow-speed smoothness ever measured arriving
together with the worst survival -- so "better" kept requiring a judgement call, and judgement
calls made after seeing the numbers are how a failed idea survives three runs.

WHY AVERAGED OVER CHECKPOINTS. This is the part that is measured rather than assumed. Take a
converged run and read its own neighbouring checkpoints, same run, same eval:

    5100  1.5%   5200  0.5%   5300  1.6%   5400  2.2%   5500  1.5%
    5600  2.7%   5700  2.1%   5800  0.3%   5900  0.6%   5999  0.5%

Mean 1.35%, std 0.83pp, spanning 0.3-2.7% -- a 9x range inside one converged run. The 0.5%
was a lucky draw, not a property of the policy, and every comparison made against it
overstated its margin. Reversals move too (mean 0.447, std 0.076). Only forward speed is
stable (0.0931, std 0.0020, 2% relative).

So a single-checkpoint score measures the checkpoint. Averaging K late checkpoints divides
those standard deviations by sqrt(K); at K=5 saturation lands at +-0.37pp and reversals at
+-0.034, which is finally small against the differences worth acting on.

WHY GEOMETRIC. A policy that will not transfer is worth nothing regardless of how fast it
is, and an arithmetic mean happily sells that trade. The geometric mean cannot: one term
near zero takes the whole score with it.

WHY GATES ON TOP. Two failures are disqualifying rather than costly -- above ~15% slew
saturation the gait does not transfer at all (71% trembles on hardware, 6.6% walks), and
below 0.75 survival the tool already warns it predicts a fall. Those are reported as gates,
not folded into the score, because a number that quietly encodes "unusable" is a number
someone will average against something else.
"""

import argparse
import os
import subprocess
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))

# floor (scores 0), target (scores 1). Every one is anchored to a measurement in this repo.
TERMS = {
    #                    floor   target   weight  where the anchors come from
    "transfer":   dict(floor=15.0, target=1.0,  weight=0.30),
    "smoothness": dict(floor=1.00, target=0.35, weight=0.35),
    "speed":      dict(floor=0.05, target=0.10, weight=0.15),
    "robustness": dict(floor=0.75, target=1.00, weight=0.20),
}
GATE_SATURATION = 15.0
GATE_SURVIVAL = 0.75


def _normalise(value, floor, target):
    """Map to [0,1]; handles targets both below (lower-is-better) and above the floor."""
    return float(np.clip((value - floor) / (target - floor), 0.0, 1.0))


def _deployability(exp_name, ckpt, amplitude, correlation, num_envs, command):
    cmd = [sys.executable, os.path.join(HERE, "mj_deployability.py"),
           "-e", exp_name, "--ckpt", str(ckpt), "--num-envs", str(num_envs),
           "--command", str(command), "0", "0"]
    cmd += ["--flat"] if amplitude is None else \
           ["--terrain-amplitude", str(amplitude), "--terrain-correlation", str(correlation)]
    out = subprocess.run(cmd, capture_output=True, text=True, cwd=os.path.dirname(HERE)).stdout
    row = surv = None
    for line in out.splitlines():
        if line.startswith(exp_name):
            row = line.split()
        if "survival" in line and "termination" in line:
            surv = float(line.split("survival")[1].split()[0])
    if row is None:
        raise RuntimeError(f"no result row for {exp_name} @ {ckpt}\n{out[-2000:]}")
    return dict(sat=float(row[2].rstrip("%")), fwd=float(row[6]),
                revs=float(row[7].rstrip("x")), survival=surv)


def late_checkpoints(exp_name, k):
    """The k highest real training checkpoints -- WEIGHT-AVERAGED ONES EXCLUDED.

    average_checkpoints.py writes its output into the same logs/ directory under a
    sentinel iteration (99998/99999), so a naive top-k picks it up as if it were the end
    of training. That is not a fair comparison against a run without one: the averaged
    policy is systematically better than any single checkpoint (that is why the tool
    exists), so a run that happens to have been averaged earlier gets 1/k of a free boost.

    Symptom: a run that was averaged earlier scores slightly differently from an identical
    run that was not, for no reason visible in the config.
    """
    log_dir = os.path.join(os.path.dirname(HERE), "logs", exp_name)
    ck = sorted(int(f[6:-3]) for f in os.listdir(log_dir) if f.startswith("model_"))
    real = []
    for c in reversed(ck):
        blob = torch.load(os.path.join(log_dir, f"model_{c}.pt"), map_location="cpu",
                          weights_only=False)
        if isinstance(blob, dict) and "averaged_from" in blob:
            continue
        real.append(c)
        if len(real) == k:
            break
    return sorted(real)


def score(exp_name, k, num_envs, command, amplitude, correlation, verbose=True):
    cks = late_checkpoints(exp_name, k)
    flat = [_deployability(exp_name, c, None, correlation, num_envs, command) for c in cks]
    rough = [_deployability(exp_name, c, amplitude, correlation, num_envs, command) for c in cks]

    raw = dict(
        transfer=float(np.mean([r["sat"] for r in flat])),
        smoothness=float(np.mean([r["revs"] for r in flat])),
        speed=float(np.mean([r["fwd"] for r in flat])),
        robustness=float(np.mean([r["survival"] for r in rough])),
    )
    spread = dict(
        transfer=float(np.std([r["sat"] for r in flat])),
        smoothness=float(np.std([r["revs"] for r in flat])),
        speed=float(np.std([r["fwd"] for r in flat])),
        robustness=float(np.std([r["survival"] for r in rough])),
    )
    parts = {k2: _normalise(raw[k2], v["floor"], v["target"]) for k2, v in TERMS.items()}
    # Geometric mean. The epsilon keeps a single zero from erasing the ranking information in
    # the other three -- a policy that fails one term badly should score near zero, not
    # exactly zero, or every failure looks identical.
    total = float(np.exp(sum(TERMS[k2]["weight"] * np.log(max(parts[k2], 1e-3))
                             for k2 in TERMS)))
    gates = []
    if raw["transfer"] > GATE_SATURATION:
        gates.append(f"saturation {raw['transfer']:.1f}% > {GATE_SATURATION}% (will not transfer)")
    if raw["robustness"] < GATE_SURVIVAL:
        gates.append(f"survival {raw['robustness']:.3f} < {GATE_SURVIVAL} (predicts a fall)")

    if verbose:
        print(f"\n{exp_name}   {k} checkpoints {cks[0]}-{cks[-1]}, {num_envs} envs, "
              f"cmd {command}, rough {amplitude*1000:.0f} mm")
        print(f"  {'term':12s} {'raw':>9s} {'+-':>7s} {'floor':>7s} {'target':>7s} "
              f"{'score':>7s} {'weight':>7s}")
        for k2, v in TERMS.items():
            print(f"  {k2:12s} {raw[k2]:9.3f} {spread[k2]:7.3f} {v['floor']:7.2f} "
                  f"{v['target']:7.2f} {parts[k2]:7.3f} {v['weight']:7.2f}")
        print(f"  {'SCORE':12s} {total:9.3f}" + ("   GATED: " + "; ".join(gates) if gates else ""))
    return total, raw, parts, gates


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("-e", "--exp-name", nargs="+", required=True)
    p.add_argument("-k", "--checkpoints", type=int, default=5,
                   help="Late checkpoints to average. 1 measures the checkpoint, not the "
                        "policy: saturation can span 0.3-2.7%% across one run's last ten.")
    p.add_argument("--num-envs", type=int, default=64)
    p.add_argument("--command", type=float, default=0.10)
    p.add_argument("--terrain-amplitude", type=float, default=0.020)
    p.add_argument("--terrain-correlation", type=float, default=0.05)
    a = p.parse_args()

    rows = []
    for name in a.exp_name:
        total, raw, parts, gates = score(name, a.checkpoints, a.num_envs, a.command,
                                         a.terrain_amplitude, a.terrain_correlation)
        rows.append((total, name, raw, gates))

    rows.sort(reverse=True, key=lambda r: r[0])
    print(f"\n{'RANKING':38s} {'score':>7s} {'sat':>7s} {'revs':>7s} {'fwd':>7s} {'surv':>7s}")
    for total, name, raw, gates in rows:
        flag = "  GATED" if gates else ""
        print(f"  {name:36s} {total:7.3f} {raw['transfer']:6.2f}% {raw['smoothness']:6.2f}x "
              f"{raw['speed']:+7.4f} {raw['robustness']:7.3f}{flag}")


if __name__ == "__main__":
    main()
