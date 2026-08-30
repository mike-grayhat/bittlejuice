"""Average the weights of several late checkpoints into one, and write it as a new checkpoint.

WHY THIS SHOULD WORK HERE, specifically. Measured on a converged run: reward plateaus (3.31 -> 3.35
over the last 2000 iterations) while the composite policy score swings 0.713 to 0.930 across
that same window. Three things keep the policy moving after the reward stops improving:

  * schedule="adaptive" is a CONTROLLER, not a decay. It holds each update's KL divergence at
    desired_kl=0.01, so the learning rate sits flat at ~5e-5 for the whole run and every
    iteration takes an equal-sized step. Settling is not something it can do.
  * entropy_coef=0.01 pays the policy to stay noisy. Action std falls 0.537 -> 0.211 and then
    STOPS, flat to within 0.004 for the last 3000 iterations.
  * Slew saturation and joint reversals are only partly priced by the reward, so a whole
    manifold of near-equal-reward policies differ in exactly the properties that decide
    whether a gait transfers.

Put together, the policy random-walks across a reward plateau in the dimensions we care
about. If that is what is happening, the MEAN of several late draws should sit nearer the
centre of the plateau than any single draw -- which is the one option that does not force the
trade picking a single checkpoint does (one late checkpoint buys 40% fewer reversals and pays 0.891 ->
0.828 in 40 mm survival).

This is weight averaging in the SWA sense (Izmailov et al., arXiv:1803.05407). It is only
valid because the checkpoints are near each other in weight space -- consecutive points on
one training trajectory, no permutation between them. Averaging weights from DIFFERENT runs
would produce noise, however similar their scores.

The optimizer state is carried verbatim from the last input checkpoint and is NOT averaged,
so the file stays loadable by every tool here. Do not resume training from the result: its
weights are off-trajectory and its Adam moments belong to a different point.
"""

import argparse
import os

import torch

MODEL_KEYS = ("actor_state_dict", "critic_state_dict", "estimator_state_dict")


def average(log_dir, checkpoints, out_path):
    dicts = []
    for c in checkpoints:
        p = os.path.join(log_dir, f"model_{c}.pt")
        dicts.append(torch.load(p, map_location="cpu", weights_only=False))

    merged = {k: v for k, v in dicts[0].items()}
    for key in MODEL_KEYS:
        if key not in dicts[0]:
            continue
        out = {}
        for name, ref in dicts[0][key].items():
            if not torch.is_floating_point(ref):
                # Counters and integer buffers: averaging them is meaningless. Keep the last
                # checkpoint's value rather than silently producing a fractional index.
                out[name] = ref.clone()
                continue
            acc = torch.zeros_like(ref, dtype=torch.float64)
            for d in dicts:
                acc += d[key][name].to(torch.float64)
            out[name] = (acc / len(dicts)).to(ref.dtype)
        merged[key] = out

    # The optimizer states are the LAST input checkpoint's, carried verbatim and NOT
    # averaged: averaged Adam moments correspond to no trajectory, and resuming from them
    # would be training on fiction. They are kept only because rsl_rl's runner.load() reads
    # every key by default, so dropping them makes the file unloadable by every tool here.
    #
    # DO NOT --resume-from an averaged checkpoint. Its weights are off-trajectory and its
    # moments belong to a different point; the two disagree.
    merged["optimizer_state_dict"] = dicts[-1].get("optimizer_state_dict")
    merged["estimator_optimizer_state_dict"] = dicts[-1].get("estimator_optimizer_state_dict")
    merged["iter"] = int(max(checkpoints))
    merged["averaged_from"] = list(checkpoints)
    torch.save(merged, out_path)
    return out_path


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("-e", "--exp-name", required=True)
    p.add_argument("-c", "--checkpoints", type=int, nargs="+", required=True)
    p.add_argument("--out-ckpt", type=int, default=99999,
                   help="Iteration number to save as, so it sits alongside the originals and "
                        "every existing tool can load it by --ckpt.")
    a = p.parse_args()

    log_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                           "logs", a.exp_name)
    out = os.path.join(log_dir, f"model_{a.out_ckpt}.pt")
    average(log_dir, a.checkpoints, out)
    print(f"averaged {len(a.checkpoints)} checkpoints {a.checkpoints} -> {out}")
    print(f"evaluate with:  --ckpt {a.out_ckpt}")


if __name__ == "__main__":
    main()
