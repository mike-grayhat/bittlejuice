"""Command throughput with the feedback stream OFF.

Ported from bittle_bringup2.py's cmd_cmdrate, whose docstring makes the key argument:

    Reading servo positions costs a query pulse per servo, which caps the feedback stream
    around 7 Hz. But a policy does not need feedback in the loop -- it can use its own
    commanded targets as dof_pos, exactly and with no latency, as long as the sim uses the
    same convention. So the number that actually caps the control loop is this one, not
    the stream rate.

This is what validates policy_io.CONTROL_DT = 0.02. If 50 Hz does not hold here, that
constant is wrong and every policy trained against it inherits the error.
"""

import statistics

NAME = "cmdrate"
HELP = "control-rate ceiling and missed deadlines at a target rate"


def add_args(p):
    p.add_argument("--seconds", type=float, default=10.0)
    p.add_argument("--hz", type=float, default=50.0,
                   help="pace to this rate; pass 0 to measure the raw ceiling")


def run(link, caps, args, recorder):
    from .. import protocol as P

    link.stop_feedback()
    link.gyro_balance_off()
    link.wake()

    if not args.hz:
        print("UNPACED: writing flat out. This dithers the servos audibly and heats them.")
        print("Use --hz 50 for a realistic control-loop test.")

    period = 1.0 / args.hz if args.hz else 0.0
    payload = P.move_cmd((j, 0) for j in P.LEG_JOINTS)   # the payload a policy would send

    with recorder.bracket_voltage(NAME, link):
        n, gaps, overruns = 0, [], 0
        t0 = link.clock()
        last = next_due = t0
        while link.clock() - t0 < args.seconds:
            link.send(payload)
            now = link.clock()
            gaps.append((now - last) * 1000)
            last = now
            n += 1
            if period:
                next_due += period
                slack = next_due - link.clock()
                if slack > 0:
                    link.sleep(slack)
                else:
                    overruns += 1
                    next_due = link.clock()

    g = sorted(gaps)
    rate = n / args.seconds
    res = {"commands": n, "rate_hz": round(rate, 2),
           "median_gap_ms": round(statistics.median(gaps), 2),
           "p95_gap_ms": round(g[int(len(g) * 0.95)], 2),
           "max_gap_ms": round(max(gaps), 2),
           "target_hz": args.hz, "overruns": overruns,
           "overrun_pct": round(100.0 * overruns / max(n, 1), 2)}

    if link.dry_run:
        print("\n[dry-run] timings below come from the virtual clock and mean nothing.")
    print(f"\n{n} full 8-joint commands in {args.seconds}s -> {rate:.1f} Hz")
    print(f"  median gap {res['median_gap_ms']:6.2f} ms")
    print(f"  p95 gap    {res['p95_gap_ms']:6.2f} ms")
    print(f"  max gap    {res['max_gap_ms']:6.2f} ms")
    if args.hz:
        print(f"  overruns   {overruns} / {n} ({res['overrun_pct']:.1f}%)  <- missed deadlines")
        print(f"\nTarget was {args.hz:g} Hz. Overruns near zero means the loop holds, and")
        print("confirms policy_io.CONTROL_DT. A quiet robot here is the result you want.")
    print("NOTE: write-only. The firmware may still be executing when the call returns,")
    print("so treat this as an upper bound.")

    recorder.write_csv("cmdrate_gaps.csv",
                       [{"i": i, "gap_ms": round(x, 4)} for i, x in enumerate(gaps)])
    recorder.note(NAME, **res)
    return res
