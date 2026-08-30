"""Equivalent-time sampling of step responses -> servo lag model beyond the poll rate.

The polled feedback path samples at ~9 Hz, which caps a *chirp* at ~2 Hz of usable band.
But a step is repeatable and its timing is under host control, so we borrow the sampling
oscilloscope trick: repeat the same step many times, jitter the poll phase relative to
the step within each trial, and FOLD every (t - t_step, angle) pair from all trials onto
one time axis. Effective resolution becomes a few ms despite the slow polling.

Two amplitudes, because a position servo has two regimes:
  * SMALL step (default 8 deg)  -- stays below the velocity limit, so the folded curve is
    the linear response: dead time + first-order lag. Yields tau and f_3db.
  * LARGE step (default 30 deg) -- saturates the velocity limit, so the folded curve is a
    ramp. Yields the slew rate, cross-checking `sweep`.

Both numbers feed the sim actuator model (bittle.xml gains / mj_vec_env lag) and the
domain-randomization range for actuator dynamics.

Floor on observability: the first poll after a step cannot land earlier than
COMMAND_GAP (20 ms) + one feedback read (~20 ms), so the folded curve starts ~40 ms in.
The dead-time fit extrapolates below that from the exponential portion -- treat dead
times under ~40 ms as "consistent with", not "measured as". The reply also reports the
position a few ms before its arrival timestamp (3 timing pulses); that bias is common to
every sample and folds into the fitted dead time.

Robot SUSPENDED, legs free.
"""

import random

import numpy as np

NAME = "step"
HELP = "equivalent-time step response -> dead time, tau, f_3db, slew"


def add_args(p):
    p.add_argument("--joint", type=int, default=8)
    p.add_argument("--all-joints", action="store_true",
                   help="run every leg joint (slow; ~2 min per joint)")
    p.add_argument("--amp-small", type=float, default=8.0)
    p.add_argument("--amp-large", type=float, default=30.0)
    p.add_argument("--trials", type=int, default=24,
                   help="steps per amplitude (directions alternate)")
    p.add_argument("--record", type=float, default=0.7, help="seconds recorded per step")
    p.add_argument("--settle", type=float, default=0.8)
    p.add_argument("--poll-timeout", type=float, default=0.25)
    p.add_argument("--seed", type=int, default=0, help="jitter RNG seed, for repeatability")


def run(link, caps, args, recorder):
    from .. import protocol as P

    joints = list(P.LEG_JOINTS) if args.all_joints else [args.joint]
    rng = random.Random(args.seed)
    link.gyro_balance_off()
    summary = []
    with recorder.bracket_voltage(NAME, link):
        link.wake()
        for j in joints:
            res = {"joint": j}
            for label, amp in (("small", args.amp_small), ("large", args.amp_large)):
                print(f"\n--- joint {j}: {args.trials} x {amp:g} deg steps ({label}) ---")
                points = _collect(link, P, j, amp, args, rng)
                recorder.write_csv(f"step_j{j}_{label}.csv",
                                   [{"dt": round(dt, 5), "deg": v, "trial": tr}
                                    for dt, v, tr in points])
                if len(points) < 20:
                    print(f"  only {len(points)} samples -- skipping fit")
                    continue
                fit = fold_and_fit([(dt, v) for dt, v, _ in points], amp)
                res.update({f"{label}_{k}": v for k, v in fit.items()})
                _print_fit(label, amp, fit)
            summary.append(res)
    recorder.write_csv("step_summary.csv", summary)

    taus = [r["small_tau_ms"] for r in summary if r.get("small_tau_ms")]
    slews = [r["large_slew_deg_s"] for r in summary if r.get("large_slew_deg_s")]
    note = {"joints": len(summary)}
    if taus:
        note.update(tau_ms=max(taus),                       # slowest joint dominates
                    f3db_hz=round(1000.0 / (2 * np.pi * max(taus)), 2))
    if slews:
        note["slew_deg_s"] = min(slews)
    recorder.note(NAME, **note)

    if taus:
        print(f"\n--- actuator lag model ---")
        print(f"slowest tau {max(taus):.0f} ms -> f_3db ~{note['f3db_hz']} Hz.")
        print("Model the sim servo as dead time + first-order lag with this tau, and")
        print("randomize tau over ~[0.5x, 2x] of it in mj_vec_env.py.")
    return note


def _collect(link, P, joint, amp, args, rng):
    """Alternating up/down steps between -amp/2 and +amp/2, poll phase jittered."""
    points, misses = [], 0
    half = amp / 2.0
    for trial in range(args.trials):
        a, b = (-half, half) if trial % 2 == 0 else (half, -half)
        link.send(P.move_cmd([(joint, a)]))
        link.sleep(args.settle)
        # Jitter the phase of the whole poll train relative to the step. Successive polls
        # land ~110 ms apart; spreading the start over one period makes the folded axis
        # dense instead of clumped at multiples of the poll period.
        link.sleep(rng.uniform(0.0, 0.11))
        t0 = link.send(P.move_cmd([(joint, b)]))
        while link.clock() - t0 < args.record:
            val, ts, _ = link.poll_joint(joint, timeout=args.poll_timeout)
            if val is None:
                misses += 1
                continue
            points.append((ts - t0, val, trial))
    if misses:
        print(f"  ({misses} polls timed out)")
    # Direction-normalize so both step directions fold onto one rising curve:
    # map angle -> progress from the trial's start toward its target.
    out = []
    for dt, v, tr in points:
        a, b = (-half, half) if tr % 2 == 0 else (half, -half)
        out.append((dt, (v - a) / (b - a) * amp, tr))   # 0 -> amp in commanded direction
    return out


# -- analysis (pure; tested offline) ----------------------------------------

def fold_and_fit(points, amp, bin_ms=10.0):
    """Fold (dt, progress_deg) points, fit dead time + first-order lag, extract slew.

    Model: y(t) = amp * (1 - exp(-(t - td)/tau)) for t > td, else 0.
    Grid-searched least squares on time-binned medians -- robust to the heavy-tailed
    outliers a serial link produces, and needs no scipy.
    """
    pts = sorted((dt, v) for dt, v in points if dt >= 0)
    if len(pts) < 10:
        return {}
    t = np.array([p[0] for p in pts])
    y = np.array([p[1] for p in pts])

    # Bin to medians so trials with many samples don't dominate the fit.
    nbins = max(int((t.max()) / (bin_ms / 1000.0)), 3)
    edges = np.linspace(0, t.max(), nbins + 1)
    tb, yb = [], []
    for i in range(nbins):
        m = (t >= edges[i]) & (t < edges[i + 1])
        if m.sum():
            tb.append(0.5 * (edges[i] + edges[i + 1]))
            yb.append(float(np.median(y[m])))
    tb, yb = np.array(tb), np.array(yb)

    best = None
    for td in np.arange(0.0, 0.101, 0.002):
        for tau in np.geomspace(0.005, 0.4, 60):
            model = np.where(tb > td, amp * (1.0 - np.exp(-(tb - td) / tau)), 0.0)
            sse = float(np.sum((yb - model) ** 2))
            if best is None or sse < best[0]:
                best = (sse, td, tau)
    sse, td, tau = best
    ss_tot = float(np.sum((yb - yb.mean()) ** 2))
    r2 = 1.0 - sse / ss_tot if ss_tot > 0 else 0.0

    # Slew from the folded curve directly: steepest secant over >=25 ms spans, so one
    # noisy bin cannot fake an extreme rate.
    slew = 0.0
    for i in range(len(tb)):
        for k in range(i + 1, len(tb)):
            span = tb[k] - tb[i]
            if span >= 0.025:
                slew = max(slew, abs(yb[k] - yb[i]) / span)
                break

    return {"samples": len(pts), "dead_time_ms": round(td * 1000, 1),
            "tau_ms": round(tau * 1000, 1),
            "f3db_hz": round(1.0 / (2 * np.pi * tau), 2),
            "slew_deg_s": round(slew, 1), "r2": round(r2, 3)}


def _print_fit(label, amp, fit):
    print(f"  {fit['samples']} folded samples | dead time {fit['dead_time_ms']} ms | "
          f"tau {fit['tau_ms']} ms (f_3db {fit['f3db_hz']} Hz) | "
          f"slew {fit['slew_deg_s']} deg/s | R^2 {fit['r2']}")
    if label == "small" and fit["r2"] < 0.85:
        print("  ! poor fit -- response may be slew-limited even at this amplitude;")
        print("    lower --amp-small.")
    if label == "large":
        print("  (large-step slew is the number to compare against `sweep`)")
