# Training your own policy

Everything here runs from the repository root. If you only want to watch the shipped policy,
you do not need this file — see the [README](../README.md).

## Your first run

```bash
uv run python sim/mj_train.py --config configs/walk.yaml -e my-run
```

That is the whole command. `configs/walk.yaml` is the recipe that produced the shipped
policy; it extends `configs/base.yaml`, which already sets the 512 environments and the 5000
iterations, so neither needs a flag. Budget ~1.1 h on a recent laptop CPU.

Then ask two questions of what came out:

```bash
uv run python tools/mj_deployability.py -e my-run    # will it survive real servos?
uv run mjpython sim/mj_eval.py -e my-run             # does it look like a gait?
```

The first is the one that matters, and [it has its own section below](#the-gate-that-decides-whether-it-transfers).
Expect its answer to be disappointing more often than not — which is the second of the two
things immediately below.

## Two things about this task that are unusual

Both are part of the recipe rather than refinements to it.

**Stop at 5000 iterations.** Deployment quality peaks there and then collapses while the
reward keeps improving. Measured out to 20000 on this exact config, the composite quality
score goes 0.944 → 0.854 → 0.644 → 0.065, with reward reaching its *maximum* at 10000 and
episode length, action std and curriculum factor all healthy the whole way. Nothing in the
training curves tells you it is happening. `base.yaml` already stops there, so the command
above obeys this — the number matters when you reach for `--max-iterations`.

**Run four seeds, not one.** Eight seeds of one identical config score anywhere from 0.07 to
0.94 — same command, same data, only `--seed` differs. One run is a lottery ticket, which is
why the section below exists.

## Once one run works, run a pool

Any flag overrides the config file, which is what keeps a four-seed pool as one config plus
`--seed`:

```bash
# ~1.1 h per seed at 512 envs on a recent laptop CPU
for S in 0 1 2 3; do
  uv run python sim/mj_train.py --config configs/walk.yaml -e pool-s$S --seed $S
done

# what each run actually ran, versioned and diffable:
#   runs/pool-s0/args.json          resolved flags -- obs_history, max_iterations,
#                                   estimator and friends never reach the cfg dicts
#   runs/pool-s0/config.json        resolved config, sorted for clean diffs
#   runs/pool-s0/config.input.yaml  the config file as it was at the time
#   runs/pool-s0/command.txt        the invocation
#   runs/pool-s0/git.txt            commit, and whether the tree was dirty
#   runs/pool-s0/uncommitted.patch  the diff, if it was

# rank them on one number
uv run python tools/policy_score.py -e pool-s0 pool-s1 pool-s2 pool-s3

# weight-average the winner's last three checkpoints, then verify that average
uv run python tools/average_checkpoints.py -e pool-s0 -c 4800 4900 5000
uv run python tools/mj_deployability.py -e pool-s0 --ckpt 99999

# export for the robot
uv run python sim/export_policy_mj.py -e pool-s0 --ckpt 99999 \
    --out deploy/policy/my_policy.npz
```

Averaging is worth real points: a single checkpoint's slew saturation can vary 9× inside one
converged run, so picking the best one measures the checkpoint rather than the policy. Never
resume *training* from an averaged checkpoint — its optimizer state is carried, not averaged.

This is the recipe that produced the shipped policy in `logs/shipped/`, which is the
weight-average of iterations 4800/4900/5000 of a run of exactly the command above.

## The gate that decides whether it transfers

```bash
uv run python tools/mj_deployability.py -e my-run
```

**Above ~15% slew saturation the gait does not reach hardware.** A policy that walks on the
robot reads 6.6%; one that trembled read 71.3%. This number cannot be read off the training
logs — the training metric is computed on the *sampled* action and is dominated by
exploration noise, and the simulated actuator rate-limits the target, so a policy demanding
900 °/s and one demanding 100 °/s produce the same smooth motion in sim. The real servo does
not filter; it buzzes.

Other questions worth asking of a checkpoint:

```bash
uv run mjpython sim/mj_eval.py -e my-run              # does it look like a gait?
uv run python tools/blindfold_test.py -e my-run       # does it use its IMU at all?
uv run python tools/obs_diff.py -e my-run --hw hw_obs.csv   # does sim match the robot?
```

`blindfold_test` is worth running early. Replace the IMU channels with a constant "upright
and still" and a policy trained the obvious way loses almost nothing — it is an open-loop
rhythm generator. Making feedback *necessary* takes a high termination threshold, angular
pushes, uneven ground and a randomised initial attitude, all of which are in the recipe above.

## Command ranges

The shipped policy was trained at a **fixed 0.10 m/s**, and the export carries that range, so
any other `--command` is clamped to it and the deploy loop says so on stdout. That is
deliberate: a policy has no idea what to do with a command it never saw, and the failure is
ugly rather than graceful. Train with a `--command-vx LO HI` range if you want a speed knob.

## Reward shaping, if you go there

Size every term with the identity

```
logged rew_X  =  scale × mean magnitude of X
```

read straight off a run's own logged `rew_*` values, which all share one normalisation. Aim
for 15–25% of that run's logged tracking reward. A term above ~100% has quietly become the
objective. Do not reconstruct per-episode sums to get there; the units do not line up with
what gets logged.

## Where the task is defined

- **`sim/policy_io.py` is the I/O contract.** Joint order, default pose, observation and
  action scaling. Sim and robot must agree on every constant in it or the gait dies.
- **`sim/config.py` holds the task.** The tests read the same `get_cfgs()` the training run
  does, so a hyperparameter cannot drift away from the test that guards it. Sweep with
  `--reward-scale` and the CLI flags rather than editing it.
