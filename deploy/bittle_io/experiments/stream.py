"""Feedback stream rate, jitter, and per-column statistics. Nothing is commanded to move.

Ported from bittle_bringup2.py's cmd_stream. Two consumers: the frame rate caps any
closed-loop control scheme, and the per-column stdev is the real observation noise that
mj_vec_env.py should be randomizing over.
"""

import statistics

NAME = "stream"
HELP = "feedback rate, jitter and per-column noise (no motion)"


def add_args(p):
    p.add_argument("--seconds", type=float, default=15.0)
    p.add_argument("--joint", type=int, default=None,
                   help="single joint (f<idx>); omit for the all-joint frame")


def run(link, caps, args, recorder):
    link.gyro_balance_off()
    with recorder.bracket_voltage(NAME, link):
        link.start_feedback(joint=args.joint)
        rows, times = [], []
        for ts, vals in link.frames(args.seconds):
            rows.append(vals)
            times.append(ts)
        link.stop_feedback()

    if len(rows) < 10:
        raise RuntimeError(f"only {len(rows)} frames in {args.seconds}s -- stream stalled")

    gaps = [(b - a) * 1000 for a, b in zip(times, times[1:])]
    gaps_sorted = sorted(gaps)
    rate = len(rows) / args.seconds
    p95 = gaps_sorted[int(len(gaps) * 0.95)]

    print(f"\n{len(rows)} frames in {args.seconds}s -> {rate:.1f} Hz")
    print(f"  median gap {statistics.median(gaps):6.2f} ms")
    print(f"  p95 gap    {p95:6.2f} ms")
    print(f"  max gap    {max(gaps):6.2f} ms   <- jitter, watch this")
    print(f"  malformed  {link.malformed} lines dropped")

    print(f"\n{'col':>4} {'mean':>8} {'stdev':>7} {'min':>7} {'max':>7} {'uniq':>5}")
    cols = []
    for c in range(link.frame_width):
        col = [r[c] for r in rows]
        st = statistics.pstdev(col)
        cols.append({"column": c, "mean": round(statistics.mean(col), 3),
                     "stdev_deg": round(st, 4), "min": min(col), "max": max(col),
                     "unique": len(set(col))})
        print(f"{c:>4} {statistics.mean(col):8.2f} {st:7.3f} "
              f"{min(col):7.1f} {max(col):7.1f} {len(set(col)):5d}")

    suffix = "" if args.joint is None else f"_j{args.joint}"
    recorder.write_csv(f"stream_columns{suffix}.csv", cols)
    recorder.write_csv(f"stream_frames{suffix}.csv",
                       [{"t": round(t - times[0], 5),
                         **{f"c{i}": v for i, v in enumerate(r)}}
                        for t, r in zip(times, rows)])

    print("\nFrame rate caps any closed-loop scheme; stdev is your real observation noise.")
    result = {"rate_hz": round(rate, 2), "frames": len(rows),
              "median_gap_ms": round(statistics.median(gaps), 2),
              "p95_gap_ms": round(p95, 2), "max_gap_ms": round(max(gaps), 2),
              "frame_width": link.frame_width,
              "max_column_stdev_deg": round(max(c["stdev_deg"] for c in cols), 4)}
    recorder.note(NAME, **result)
    return result
