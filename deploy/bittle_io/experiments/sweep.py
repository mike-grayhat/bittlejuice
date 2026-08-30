"""Frequency sweep (chirp) per joint -> servo model.

Ported from bittle_bringup2.py's cmd_sweep, extended to loop over every leg joint and to
use single-joint feedback when the probe found it faster.

Why the feedback rate matters here: a chirp to 3 Hz sampled at ~7 Hz is at Nyquist, so the
amplitude ratio and phase lag at the top of the sweep are meaningless. If `f8` samples at
~50 Hz the whole band becomes resolvable, which is what makes this experiment worth
running at all.

Outputs feed two places: actuator gains in bittle.xml and the servo-response ranges in
mj_vec_env.py. They used to feed a third, control_loop.MAX_ANGLE_DELTA_PER_STEP_RAD, which
NO LONGER EXISTS -- the deploy loop now sends the raw target and keeps ServoModel as an
observer only, so there is no host-side clamp left to set.

VALIDITY AFTER THE FIRMWARE PATCH: this experiment polls joint feedback at ~15 Hz and
sweeps to 3 Hz, i.e. 5 samples per cycle. That was tolerable while the firmware
interpolation ramp stretched every response to ~100 ms. With the ramp removed the plant is
fast enough that the max-slew figure ALIASES: it read 39-46 deg/s post-patch against 109.7
on the same instrument pre-patch, which is the sampling, not the servo. Treat slew numbers
from this experiment as a lower bound until the sampling is fixed.

MEASUREMENT CAVEATS (from OpenCatEsp32 src/espServo.h + src/reaction.h):
  * Each poll DETACHES the servo for ~15-20 ms to time the feedback pulse, and the next
    move re-attaches it. The joint is therefore only driven part of each cycle, so treat
    the measured bandwidth/slew as a lower bound on the always-driven servo.
  * Each `f` request softens the servo gain (setServoP(P_SOFT)) and each `i` move
    restores working gain (reaction.h restores P_WORKING on move tokens). The cycle
    therefore alternates working-gain driving with soft/detached reading windows.
  * Polled sampling runs at ~9 Hz (see capabilities.json feedback_polled_rate_hz), so
    keep --f1 at or below ~2 Hz for un-aliased results.
"""

import math

import numpy as np

NAME = "sweep"
HELP = "chirp a joint and log commanded vs measured -> bandwidth, lag, deadband, slew"


def add_args(p):
    p.add_argument("--joint", type=int, default=None, help="single joint; omit for all")
    p.add_argument("--column", type=int, default=None,
                   help="feedback column (ignored with --joint-feedback)")
    p.add_argument("--joint-feedback", action="store_true", default=True,
                   help="stream f<joint> per joint (default; much faster)")
    p.add_argument("--full-frame", dest="joint_feedback", action="store_false",
                   help="use the all-joint frame instead")
    p.add_argument("--amp", type=float, default=20.0)
    p.add_argument("--f0", type=float, default=0.2)
    p.add_argument("--f1", type=float, default=3.0)
    p.add_argument("--cmd-hz", type=float, default=30.0)
    p.add_argument("--seconds", type=float, default=20.0)
    p.add_argument("--poll-timeout", type=float, default=0.2,
                   help="per-sample feedback request timeout")
    p.add_argument("--cooldown", type=float, default=3.0,
                   help="pause between joints so the servos can shed heat")


def run(link, caps, args, recorder):
    from .. import protocol as P

    joints = [args.joint] if args.joint is not None else list(P.LEG_JOINTS)
    if not args.joint_feedback and args.column is None and args.joint is not None:
        raise ValueError("pass --column (from jointmap) or keep --joint-feedback")

    # The relevant rate is the POLLED one, not the free-running stream: a motion
    # command stops the stream, so each sample costs a request round-trip. Assume
    # roughly half the idle stream rate and warn on that.
    idle = caps.feedback_one_rate_hz or caps.feedback_all_rate_hz
    if idle:
        polled = idle / 2.0
        print(f"  feedback ~{idle:.0f} Hz idle -> ~{polled:.0f} Hz polled "
              f"(Nyquist ~{polled / 2:.1f} Hz)")
        if args.f1 > polled / 2:
            print(f"! chirp reaches {args.f1} Hz, above Nyquist. Results up there are")
            print("  aliased -- lower --f1.")

    link.gyro_balance_off()
    summary = []
    with recorder.bracket_voltage(NAME, link):
        for j in joints:
            print(f"\n--- joint {j}: {args.f0}->{args.f1} Hz over {args.seconds}s ---")
            rows = _chirp(link, P, j, args)
            recorder.write_csv(f"sweep_j{j}.csv", rows)
            s = _analyze(rows, args)
            s["joint"] = j
            summary.append(s)
            print(f"  samples {s['samples']}  |  deadband {s['deadband_deg']} deg  |  "
                  f"max slew {s['max_slew_deg_per_s']} deg/s")
            if len(joints) > 1 and j != joints[-1]:
                link.sleep(args.cooldown)

    recorder.write_csv("sweep_summary.csv", summary)
    _report(summary)
    # Filter before reducing: a joint that produced no usable frames contributes None,
    # and max() over several Nones raises rather than returning one.
    slews = [s["max_slew_deg_per_s"] for s in summary if s["max_slew_deg_per_s"] is not None]
    recorder.note(NAME, joints=len(summary),
                  max_slew_deg_per_s=max(slews) if slews else None,
                  slowest_joint_slew_deg_per_s=min(slews) if slews else None)
    return {"joints": summary}


def _chirp(link, P, joint, args):
    """Command, then REQUEST a reading, one tick at a time.

    Streaming does not work here: a motion command stops the `f` stream on this board
    (measured -- ~57 frames/s idle drops to 0 after one `i` command). Interleaving
    therefore has to be request/response, which caps the sample rate at the poll
    round-trip rather than the stream rate.
    """
    rows, t0, misses = [], link.clock(), 0
    while True:
        t = link.clock() - t0
        if t > args.seconds:
            break
        # Linear chirp. Note the instantaneous frequency of sin(2*pi*f(t)*t) is not f(t)
        # but f0 + 2*(f1-f0)*t/T; the logged freq_hz column is the sweep parameter, and
        # any offline fit should derive true instantaneous frequency from the phase.
        f = args.f0 + (args.f1 - args.f0) * (t / args.seconds)
        cmd = args.amp * math.sin(2 * math.pi * f * t)
        link.send(P.move_cmd([(joint, cmd)]))
        val, ts, _ = link.poll_joint(joint, timeout=args.poll_timeout)
        if val is None:
            misses += 1
            continue
        rows.append({"t": round(ts - t0, 5), "freq_hz": round(f, 4),
                     "commanded_deg": round(cmd, 3), "measured_deg": val})
    link.send(P.move_cmd([(joint, 0)]))
    if misses:
        print(f"  ({misses} polls timed out)")
    return rows


def _analyze(rows, args):
    """Headline numbers. Bandwidth and phase need an offline fit -- this is the summary."""
    if len(rows) < 10:
        return {"samples": len(rows), "deadband_deg": None, "max_slew_deg_per_s": None,
                "low_freq_gain": None}
    t = np.array([r["t"] for r in rows])
    cmd = np.array([r["commanded_deg"] for r in rows])
    meas = np.array([r["measured_deg"] for r in rows])

    dt = np.diff(t)
    ok = dt > 1e-6
    slew = np.abs(np.diff(meas)[ok] / dt[ok])
    # 99th percentile, not max: a single dropped frame produces a spurious huge slew.
    max_slew = float(np.percentile(slew, 99)) if slew.size else None

    # Low-frequency gain: over the first third of the sweep the servo should track well,
    # so the amplitude ratio there is the DC gain.
    n3 = max(len(rows) // 3, 2)
    lo_gain = (float(np.ptp(meas[:n3]) / np.ptp(cmd[:n3]))
               if np.ptp(cmd[:n3]) > 1e-6 else None)

    # Deadband: how far the command departs from zero before the measurement follows.
    small = np.abs(cmd) < args.amp * 0.25
    deadband = None
    if small.sum() > 10:
        resid = meas[small] - np.median(meas[small])
        moved = np.abs(resid) > 0.5
        if moved.any():
            deadband = float(np.min(np.abs(cmd[small][moved])))

    return {"samples": len(rows),
            "deadband_deg": None if deadband is None else round(deadband, 3),
            "max_slew_deg_per_s": None if max_slew is None else round(max_slew, 1),
            "low_freq_gain": None if lo_gain is None else round(lo_gain, 3)}


def _report(summary):
    slews = [s["max_slew_deg_per_s"] for s in summary if s["max_slew_deg_per_s"]]
    print("\n--- what to do with this ---")
    print("Plot amplitude ratio and phase lag vs freq per joint -> servo bandwidth model.")
    if slews:
        worst = min(slews)     # the SLOWEST joint is what any safety limit must respect
        per_step = worst * 0.02
        print(f"Slowest joint's max slew: {worst:.0f} deg/s -> {per_step:.1f} deg per 20ms tick.")
        print("  LOWER BOUND ONLY on the patched firmware: 3 Hz against ~15 Hz polled feedback")
        print("  is 5 samples/cycle, so this aliases. Do not size anything from it, and note")
        print("  control_loop.MAX_ANGLE_DELTA_PER_STEP_RAD no longer exists -- the loop sends")
        print("  the raw target and ServoModel is an observer.")
