"""IMU characterization. The new experiment -- everything else is a port.

Four phases, each independently useful:

  static           noise, bias, drift and sample rate with the robot level and still
  tilt             operator-guided; VALIDATES the gravity sign/axis convention
  dynamics         IMU latency and lag against a commanded body-tipping motion
  angvel-fallback  can differentiated attitude stand in for the missing gyro rates?

The tilt phase is the one that catches a mirrored gravity vector -- a bug that trains
perfectly well in simulation and puts the robot on its face on contact with the floor.
"""

import math
import statistics

import numpy as np

from .. import protocol as P

NAME = "imu"
HELP = "IMU noise, gravity-convention validation, latency, ang_vel feasibility"

PHASES = ["static", "tilt", "dynamics", "angvel-fallback"]


def add_args(p):
    p.add_argument("--phase", choices=PHASES + ["all"], default="static")
    p.add_argument("--seconds", type=float, default=60.0, help="static phase duration")
    p.add_argument("--amp", type=float, default=25.0, help="dynamics phase tilt amplitude")
    p.add_argument("--trials", type=int, default=6)
    p.add_argument("--dyn-joints", type=int, nargs="*", default=[12, 13],
                   help="joints stepped in the dynamics phase; pick ones bodyid showed "
                        "actually pitch the body")


def run(link, caps, args, recorder):
    if caps.imu_chip is None:
        raise RuntimeError("probe found no IMU -- run `probe` first and check capabilities.json")
    link.gyro_balance_off()
    phases = PHASES if args.phase == "all" else [args.phase]
    out = {}
    with recorder.bracket_voltage(NAME, link):
        for ph in phases:
            out[ph] = _PHASES[ph](link, caps, args, recorder)
    recorder.note(NAME, **{f"{k}_{kk}": vv for k, v in out.items()
                           for kk, vv in (v or {}).items()})
    return out


# -- static -----------------------------------------------------------------

def _static(link, caps, args, recorder):
    print(f"\n[static] Robot LEVEL and STILL for {args.seconds:.0f}s. Do not touch it.")
    rows, times = [], []
    for ts, s in link.imu_stream(args.seconds):
        times.append(ts)
        rows.append(s)
    if len(rows) < 10:
        raise RuntimeError(f"only {len(rows)} IMU samples -- stream stalled")

    dt = times[-1] - times[0]
    rate = (len(rows) - 1) / dt if dt > 0 else 0.0
    gaps = [(b - a) * 1000 for a, b in zip(times, times[1:])]

    ypr = np.array([s.ypr_deg for s in rows])
    res = {"samples": len(rows), "rate_hz": round(rate, 2),
           "median_gap_ms": round(statistics.median(gaps), 2),
           "max_gap_ms": round(max(gaps), 2)}
    for i, name in enumerate(("yaw", "pitch", "roll")):
        res[f"{name}_mean_deg"] = round(float(ypr[:, i].mean()), 4)
        res[f"{name}_stdev_deg"] = round(float(ypr[:, i].std()), 4)
        # Drift: least-squares slope over the window.
        slope = float(np.polyfit(np.array(times) - times[0], ypr[:, i], 1)[0])
        res[f"{name}_drift_deg_per_s"] = round(slope, 5)

    if rows[0].accel is not None:
        acc = np.array([s.accel for s in rows])
        for i, ax in enumerate("xyz"):
            res[f"accel_{ax}_mean"] = round(float(acc[:, i].mean()), 4)
            res[f"accel_{ax}_stdev"] = round(float(acc[:, i].std()), 4)

    print(f"  {len(rows)} samples in {dt:.1f}s -> {rate:.1f} Hz "
          f"(median gap {res['median_gap_ms']} ms)")
    for name in ("yaw", "pitch", "roll"):
        print(f"  {name:>5}: mean {res[f'{name}_mean_deg']:8.3f}  "
              f"stdev {res[f'{name}_stdev_deg']:7.4f}  "
              f"drift {res[f'{name}_drift_deg_per_s']:+.5f} deg/s")
    print("\n  stdev -> the obs noise mj_vec_env.py should inject on projected_gravity.")
    print("  A level robot reporting non-zero pitch/roll mean is a MOUNTING OFFSET;")
    print("  subtract it, or the policy sees a permanently tilted world.")

    recorder.write_csv("imu_static.csv", [
        {"t": round(t - times[0], 5), "yaw": s.yaw, "pitch": s.pitch, "roll": s.roll,
         **({} if s.accel is None else
            {"ax": s.accel[0], "ay": s.accel[1], "az": s.accel[2]})}
        for t, s in zip(times, rows)])
    return res


# -- tilt -------------------------------------------------------------------
# Poses the operator can produce accurately by hand, with the body-frame gravity direction
# each one implies. This is the ground truth the IMU is checked against.
_POSES = [
    ("level, feet down", [0.0, 0.0, -1.0]),
    ("nose down 90 deg (standing on its face)", [1.0, 0.0, 0.0]),
    ("nose up 90 deg (standing on its tail)", [-1.0, 0.0, 0.0]),
    ("rolled 90 deg onto its LEFT side", [0.0, -1.0, 0.0]),
    ("rolled 90 deg onto its RIGHT side", [0.0, 1.0, 0.0]),
    ("upside down", [0.0, 0.0, 1.0]),
]


def _tilt(link, caps, args, recorder):
    print("\n[tilt] Validates the gravity sign and axis convention against the simulator.")
    print("Hold each pose steady; press Enter to sample, or 's' to skip.\n")
    rows = []
    for label, expected in _POSES:
        try:
            ans = link.prompt(f"  Place the robot: {label!r} -> ", "s").strip().lower()
        except EOFError:
            break
        if ans.startswith("s"):
            continue
        samples = [s for _, s in link.imu_stream(1.5)]
        if not samples:
            print("    no IMU data; skipped")
            continue
        ypr = np.median(np.array([s.ypr_deg for s in samples]), axis=0)
        roll, pitch = math.radians(ypr[2]), math.radians(ypr[1])
        g_rpy = P.projected_gravity_from_rpy(roll, pitch)
        exp = np.array(expected, dtype=float)
        err_rpy = float(np.linalg.norm(g_rpy - exp))

        row = {"pose": label,
               "yaw": round(float(ypr[0]), 2), "pitch": round(float(ypr[1]), 2),
               "roll": round(float(ypr[2]), 2),
               "expected_gx": exp[0], "expected_gy": exp[1], "expected_gz": exp[2],
               "rpy_gx": round(g_rpy[0], 4), "rpy_gy": round(g_rpy[1], 4),
               "rpy_gz": round(g_rpy[2], 4), "rpy_err": round(err_rpy, 4)}

        if samples[0].accel is not None:
            g_acc = P.projected_gravity_from_accel(
                np.median(np.array([s.accel for s in samples]), axis=0))
            row.update({"accel_gx": round(g_acc[0], 4), "accel_gy": round(g_acc[1], 4),
                        "accel_gz": round(g_acc[2], 4),
                        "accel_err": round(float(np.linalg.norm(g_acc - exp)), 4),
                        "rpy_vs_accel": round(float(np.linalg.norm(g_rpy - g_acc)), 4)})
        rows.append(row)
        print(f"    ypr=({ypr[0]:7.1f},{ypr[1]:7.1f},{ypr[2]:7.1f})  "
              f"g_rpy={np.array2string(g_rpy, precision=2)}  err={err_rpy:.3f}"
              + (f"  |rpy-accel|={row['rpy_vs_accel']:.3f}" if "rpy_vs_accel" in row else ""))

    if not rows:
        print("  no poses sampled")
        return {}

    recorder.write_csv("imu_tilt.csv", rows)
    # Split verdict: the rpy path is only DEFINED for
    # |roll| < 90 -- beyond that the firmware's Euler branch flips and a mirrored vector
    # is expected, not a bug. The accel path (chip-frame mapping in
    # projected_gravity_from_accel) must hold in every pose.
    in_domain = [r for r in rows if abs(r["roll"]) < 90.0]
    worst_rpy = max((r["rpy_err"] for r in in_domain), default=None)
    worst_accel = max((r["accel_err"] for r in rows if "accel_err" in r), default=None)
    res = {"poses": len(rows),
           "worst_rpy_err_in_domain": None if worst_rpy is None else round(worst_rpy, 4),
           "worst_accel_err": None if worst_accel is None else round(worst_accel, 4)}

    rpy_ok = worst_rpy is not None and worst_rpy < 0.35
    accel_ok = worst_accel is None or worst_accel < 0.35   # vacuous without accel
    print(f"\n  rpy path   (|roll|<90, {len(in_domain)} poses): worst err "
          f"{worst_rpy if worst_rpy is not None else 'n/a'}")
    print(f"  accel path (all {len(rows)} poses):        worst err "
          f"{worst_accel if worst_accel is not None else 'n/a'}")
    if rpy_ok and accel_ok:
        print("  PASS: both gravity paths match the sim convention in their domains.")
    else:
        print("  FAIL: convention mismatch. Compare imu_tilt.csv columns against")
        print("  expected_g* and fix the protocol.py functions -- do NOT compensate")
        print("  downstream.")
    recorder.note(NAME, tilt_pass=bool(rpy_ok and accel_ok))
    return res


# -- dynamics ---------------------------------------------------------------

def _dynamics(link, caps, args, recorder):
    """Tip the body with a known leg motion and time the IMU's response.

    The robot must be ON THE GROUND -- suspended, the legs move but the body never tilts.

    Sequencing matters: the move is sent MID-STREAM, with `gP` already running. Sending
    `gP` right after the move loses it to the same <20 ms post-move token-drop window
    that eats `f8` requests (measured; see link.COMMAND_GAP) -- the stream never starts
    and every trial reads "no response".
    """
    print(f"\n[dynamics] Robot ON THE GROUND, clear space around it. {args.trials} trials.")
    print(f"  Joints {args.dyn_joints} step, the body pitches; we time the IMU response.")
    link.wake()
    rows, lats = [], []
    step_joints = args.dyn_joints

    for trial in range(args.trials):
        link.send(P.move_cmd((j, 0) for j in P.LEG_JOINTS))
        _sleep_stream(link, 1.0)
        base, p0, t_cmd, t_first, hit = [], None, None, None, None
        for ts, s in link.imu_stream(2.2):
            if t_first is None:
                t_first = ts
            if t_cmd is None:
                if ts - t_first < 0.5:
                    base.append(s.pitch)
                    continue
                if not base:
                    break                      # stream produced nothing usable
                p0 = _median_f(base)
                t_cmd = link.send(P.move_cmd((j, args.amp) for j in step_joints))
                continue
            rows.append({"trial": trial, "t": round(ts - t_cmd, 5),
                         "yaw": s.yaw, "pitch": s.pitch, "roll": s.roll})
            if hit is None and abs(s.pitch - p0) > 2.0:
                hit = (ts - t_cmd) * 1000
        if hit:
            lats.append(hit)
        print(f"  trial {trial + 1}: " + (f"{hit:.1f} ms" if hit else "no pitch response"))
        link.send(P.move_cmd((j, 0) for j in P.LEG_JOINTS))
        _sleep_stream(link, 0.8)

    recorder.write_csv("imu_dynamics.csv", rows)
    if not lats:
        print("  no response detected -- was the robot suspended? it must be on the ground.")
        return {"trials": args.trials, "responses": 0}
    res = {"trials": args.trials, "responses": len(lats),
           "median_latency_ms": round(statistics.median(lats), 1),
           "min_latency_ms": round(min(lats), 1), "max_latency_ms": round(max(lats), 1)}
    print(f"\n  median {res['median_latency_ms']} ms "
          f"(min {res['min_latency_ms']}, max {res['max_latency_ms']})")
    print("  Command -> body-motion-seen-by-IMU, with the stream already running: the")
    print("  cleanest end-to-end latency number this robot can produce over serial.")
    return res


def _median_f(xs):
    ys = sorted(xs)
    n = len(ys)
    return ys[n // 2] if n % 2 else 0.5 * (ys[n // 2 - 1] + ys[n // 2])


def _sleep_stream(link, seconds):
    list(link.drain(seconds))


# -- angvel-fallback --------------------------------------------------------

def _angvel(link, caps, args, recorder):
    """Can differentiated attitude replace the gyro rates the firmware never prints?

    Differentiating a fused attitude estimate amplifies its noise by the sample rate, so
    the question is whether the resulting signal has usable bandwidth at 50 Hz or is
    mostly differentiated quantization noise.
    """
    print("\n[angvel-fallback] Deciding whether finite-differenced attitude can stand in")
    print("  for base_ang_vel. Hold the robot STILL first, then rock it by hand when asked.")

    still = _rates_over(link, 8.0)
    link.prompt("  Now ROCK the robot gently by hand for ~8s, then Enter -> ")
    moving = _rates_over(link, 8.0)

    if still is None or moving is None:
        print("  insufficient IMU data")
        return {}

    noise = float(np.abs(still["rates"]).std())
    signal = float(np.abs(moving["rates"]).std())
    snr = signal / noise if noise > 0 else float("inf")
    nyquist = moving["rate_hz"] / 2.0

    recorder.write_csv("imu_angvel_still.csv", still["rows"])
    recorder.write_csv("imu_angvel_moving.csv", moving["rows"])

    res = {"imu_rate_hz": round(moving["rate_hz"], 2),
           "nyquist_hz": round(nyquist, 2),
           "noise_floor_rad_s": round(noise, 5),
           "signal_rad_s": round(signal, 5),
           "snr": round(snr, 2)}

    print(f"\n  IMU rate        {res['imu_rate_hz']} Hz  (usable to ~{res['nyquist_hz']} Hz)")
    print(f"  noise floor     {res['noise_floor_rad_s']} rad/s  (robot still)")
    print(f"  signal          {res['signal_rad_s']} rad/s  (robot rocking)")
    print(f"  SNR             {res['snr']}")
    print("\n  --- verdict ---")
    if moving["rate_hz"] < 40:
        print(f"  IMU streams at {moving['rate_hz']:.0f} Hz, below the 50 Hz control rate.")
        print("  Differentiated rates cannot represent gait-frequency motion. Either drop")
        print("  base_ang_vel from the observation and retrain, or flash a build that")
        print("  prints gy (VectorInt16 in OpenCatEsp32 src/imu.h, already populated).")
    elif snr < 5:
        print(f"  SNR {snr:.1f} is too low -- mostly differentiated quantization noise.")
        print("  Flash a build that prints gy rather than trusting this.")
    else:
        print(f"  Usable: {moving['rate_hz']:.0f} Hz at SNR {snr:.1f}. Feed it through")
        print("  state.py with ang_vel_valid, and match the noise in mj_vec_env.py.")
    return res


def _rates_over(link, seconds):
    times, ypr = [], []
    for ts, s in link.imu_stream(seconds):
        times.append(ts)
        ypr.append(np.radians(s.ypr_deg))
    if len(times) < 5:
        return None
    t = np.array(times)
    a = np.array(ypr)
    a[:, 0] = np.unwrap(a[:, 0])        # yaw wraps at +-180 deg
    dt = np.diff(t)
    # Several IMU lines can arrive in one serial chunk and share an arrival timestamp,
    # making dt = 0 and the naive difference inf/NaN. Only difference across pairs with
    # a real time gap; the shared-stamp pairs carry no rate information anyway.
    ok = dt > 1e-4
    if not ok.any():
        return None
    rates = np.diff(a, axis=0)[ok] / dt[ok, None]
    tt = t[1:][ok]
    rows = [{"t": round(tt[i] - t[0], 5),
             "yaw_rate": round(float(rates[i, 0]), 5),
             "pitch_rate": round(float(rates[i, 1]), 5),
             "roll_rate": round(float(rates[i, 2]), 5)} for i in range(len(rates))]
    return {"rates": rates, "rows": rows, "rate_hz": (len(t) - 1) / (t[-1] - t[0])}


_PHASES = {"static": _static, "tilt": _tilt, "dynamics": _dynamics,
           "angvel-fallback": _angvel}
