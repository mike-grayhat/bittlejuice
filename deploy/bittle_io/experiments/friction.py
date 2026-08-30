"""Foot-ground friction by ramp test. The robot measures its own incline via the IMU.

WHY: bittle.xml's foot friction was 0.8, randomized x0.6-1.4 by mj_vec_env, and that 0.8 had
no measurement behind it -- the XML comment beside it admits a neighbouring constant was
"borrowed from a much larger quadruped". It was the last physical parameter in the model
with nothing behind it, and it governs contact timing, which governs gait cadence. Two
observations pointed straight at it: hardware paces faster than sim, and how well the robot
walks depends on the surface.

OUTCOME: the MJCF's nominal 0.8 was 3-5x too high. Real is 0.156 on smooth plastic and 0.440 on a
fabric mat, so the old randomized span of 0.48-1.12 sat entirely ABOVE the grippiest surface
this robot has ever stood on -- no policy had ever felt a foot break loose. bittle.xml now
carries 0.25 nominal and mj_vec_env randomizes the feet over 0.10-0.55.

But it did NOT explain the two observations that motivated it. Re-running trained policies
across mu 0.80/0.44/0.25/0.156 moved forward speed ~6% and left gait cadence
unchanged to three digits. The surface-dependence the operator sees on hardware is real, and
the friction number was genuinely wrong, but the gait-tempo gap lives elsewhere -- in the
dof_pos observation, where sim reads true joint angles and hardware reads a servo estimate.
Worth stating plainly because the causal story here was seductive and false: "the stance foot
slides out backwards instead of driving the body" is not what the physics does at 0.15 m/s.

METHOD: the classic inclined-plane test. Stand the robot on a board, tilt slowly until it
slides, and the tangent of the slip angle is the static coefficient:

    mu_s = tan(theta_slip)

The robot's own accelerometer gives theta directly -- no protractor -- via
protocol.projected_gravity_from_accel, which `imu --phase tilt` validated in all six
orientations. theta = acos(-g_z) is the angle between the body's down axis and gravity.

SLIP DETECTION -- and why the obvious method does not work. The first version watched for
|accel| to drop below g, reasoning that a sliding robot is in partial free fall. It never
fired once, on either surface, across ten trials. The reason is arithmetic: during a steady
slide the accelerometer measures specific force

    |accel| = g * cos(theta) * sqrt(1 + mu_k^2)

and at the slip angle mu_k ~= mu_s = tan(theta), which collapses that to g*cos(theta)*
sec(theta) = g EXACTLY. The signature is absent precisely at the angle we want to find; it
only emerges well past slip, once the board is far steeper than the robot needed.

So the operator's eye is the detector -- they see it go and stop tilting -- and this module's
job is to read the angle off the resulting plateau rather than to find the event itself.
The tilt channel carries 0.45 deg of noise at 250 Hz, so max(tilt) is a NOISE statistic that
overestimates by 1-3 deg; the plateau is extracted from a median-filtered signal instead.
Measured on this robot, five plastic trials plateaued within +-1 deg of each other, which is
a repeatability no hand-held board achieves by accident.

CAVEATS, so the number is not over-trusted:
  * MuJoCo's friction with condim=6 includes torsional and rolling terms; a ramp test
    measures only the tangential coefficient and cannot separate them.
  * This is STATIC friction on a stationary robot. A walking foot is loaded differently
    and involves kinetic friction, generally lower.
  * The feet are BARE PLASTIC (Bittle X ships no rubber pads), so low coefficients are
    expected: 0.156 on smooth plastic is a real result, not a broken measurement.
  * Measure the surface you actually run on. That is the point.
"""

import math

import numpy as np

from .. import protocol as P

NAME = "friction"
HELP = "ramp test for foot-ground friction (interactive); mu = tan(slip angle)"

# Nominal gravity, used only as a fallback. The real reference is measured per trial: this
# unit's accelerometer reads 10.01 +-0.01 at rest, a 2% scale error, and hardcoding 9.81 made
# every "how far from g" test 0.2 m/s^2 pre-biased before it saw any data.
G_NOMINAL = 9.81
# Tilt noise is ~0.45 deg std at 250 Hz. A 0.2 s median kills it (~7x) while passing a ramp
# of a few deg/s essentially untouched.
SMOOTH_SAMPLES = 51
# The plateau is the longest stretch the filtered tilt stays within this band of its own
# maximum -- i.e. where the operator stopped raising the board because it let go.
PLATEAU_BAND_DEG = 1.5
PLATEAU_MIN_SAMPLES = 250     # ~1 s at 250 Hz
# Bare plastic feet on a smooth floor genuinely let go near 9 deg (mu 0.156, measured). The only
# thing this floor rejects is a trial where the board barely moved at all.
MIN_PLAUSIBLE_TILT = 4.0
# Past this the robot is toppling over rather than sliding -- two mat trials reached 64 and 77
# deg, which is the board being turned on its side, not a friction measurement.
MAX_PLAUSIBLE_TILT = 45.0
# The board must actually be still and flat before a trial counts as started.
READY_TILT_DEG = 4.0
READY_STILL_DEV = 0.25


def add_args(p):
    p.add_argument("--trials", type=int, default=5)
    p.add_argument("--surface", type=str, default="",
                   help="free-text label recorded with the result, e.g. 'wood floor'")
    p.add_argument("--max-seconds", type=float, default=25.0,
                   help="give up on a trial after this long")
    p.add_argument("--stand", action="store_true", default=True,
                   help="hold the home pose so the feet are loaded as they are when walking")


def run(link, caps, args, recorder):
    if caps.imu_chip is None and not link.dry_run:
        raise RuntimeError("probe found no IMU -- run `probe` first")
    if caps.imu_has_accel is False:
        raise RuntimeError("this firmware build does not emit accelerometer data; the ramp "
                           "test needs it (the fused roll/pitch path cannot detect sliding)")

    link.gyro_balance_off()
    print("\n[friction] Inclined-plane test.")
    print("  Put the robot on a board, feet down, facing UP the slope.")
    print("  Then tilt the board SLOWLY -- a few degrees per second -- until it slides.")
    print(f"  Surface: {args.surface or '(unlabelled -- pass --surface)'}\n")

    rows, mus = [], []
    with recorder.bracket_voltage(NAME, link):
        if args.stand:
            # Feet loaded as they are in a stance, not splayed under a dead servo.
            link.send(P.move_cmd((j, 32) for j in P.LEG_JOINTS))
            link.sleep(1.0)

        for trial in range(args.trials):
            # TWO stages, because one Enter cannot both advance the trial and start the
            # recording: the operator uses it to move on from the previous trial and is
            # still handling the robot when capture begins. Measured consequence -- slips
            # "detected" at 3.8-7.5 deg that were the operator's hands, and trials whose
            # whole window elapsed during setup.
            link.prompt(f"\n  trial {trial + 1}/{args.trials}: place the robot, board FLAT, "
                        f"hands off. Press Enter -> ")
            g_ref = _wait_until_ready(link)
            if g_ref is None:
                print("    board not flat/still after 10 s -- skipping this trial")
                continue
            print(f"    ready (g_ref {g_ref:.2f}). TILT NOW, slowly -- STOP RAISING the moment "
                  f"you see it slide, and HOLD there for ~2 s.")
            trace = _capture(link, args.max_seconds)
            if not trace:
                print("    no IMU data; skipped")
                continue
            res = _analyse(trace)
            # Keep the raw ramp regardless of verdict, so a suspect number can be re-read
            # rather than re-run: a clean trial shows tilt rising, then holding flat. This is
            # what let the first run's ten trials be re-analysed offline after the detector
            # turned out to be looking for a signal that is not there.
            recorder.write_csv(
                f"friction_trace_{args.surface or 'x'}_{trial}.csv".replace(" ", "_"),
                [{"t": round(t - trace[0][0], 4), "tilt_deg": round(a, 2),
                  "accel_mag": round(m, 3)} for t, a, m in trace])
            if res is None:
                print("    board barely moved -- tilt further")
                continue
            theta, mu, why = res
            rows.append({"trial": trial, "surface": args.surface,
                         "slip_angle_deg": round(theta, 2), "mu_static": round(mu, 3),
                         "detected_by": why})
            if why.startswith("REJECTED"):
                print(f"    {why} -- excluded")
                continue
            mus.append(mu)
            print(f"    slip at {theta:.1f} deg  ->  mu = {mu:.2f}   ({why})")

    recorder.write_csv("friction.csv", rows)
    if not mus:
        print("\nNo usable trials.")
        return {"trials": 0}

    med = float(np.median(mus))
    res = {"trials": len(mus), "mu_median": round(med, 3),
           "mu_min": round(min(mus), 3), "mu_max": round(max(mus), 3),
           "surface": args.surface}
    print(f"\nmu_static median {med:.2f}  (range {min(mus):.2f}-{max(mus):.2f}, "
          f"{len(mus)} trials)")
    # Kept in sync with mj_vec_env.FOOT_FRICTION_RANGE by hand; a stale bound here would only
    # print a wrong comparison, never affect training.
    lo, hi = 0.10, 0.55
    print(f"\nmj_vec_env randomizes foot friction over {lo:.2f}-{hi:.2f} "
          f"(bittle.xml nominal 0.30).")
    if med < lo or med > hi:
        print(f"  ! measured {med:.2f} sits OUTSIDE that range -- the policy has never trained "
              f"on this surface. Widen FOOT_FRICTION_RANGE before the next run.")
    else:
        print(f"  measured {med:.2f} is inside the trained range.")
    print("  Kinetic friction while walking is lower again, so treat this as an upper bound.")
    print("  Bare plastic feet: ~0.17 on smooth plastic and ~0.45 on fabric are the values"
          "\n  measured on this robot, so a low number here is a result, not a fault.")
    recorder.note(NAME, **res)
    return res


def _tilt_of(sample):
    g = P.projected_gravity_from_accel(sample.accel)
    return math.degrees(math.acos(float(np.clip(-g[2], -1.0, 1.0))))


def _wait_until_ready(link, timeout=10.0):
    """Block until the board is flat AND still, so capture starts from a known state.

    Also returns this trial's resting |accel|, which is the gravity reference everything
    downstream compares against -- see G_NOMINAL on why the constant is not trusted.
    """
    recent = []
    for _ts, s in link.imu_stream(timeout):
        if s.accel is None:
            continue
        recent.append((_tilt_of(s), float(np.linalg.norm(s.accel))))
        recent = recent[-60:]
        if len(recent) >= 60:
            tilts = [r[0] for r in recent]
            mags = [r[1] for r in recent]
            g_ref = float(np.median(mags))
            if (max(tilts) < READY_TILT_DEG
                    and max(abs(m - g_ref) for m in mags) < READY_STILL_DEV):
                return g_ref
    return None


def _capture(link, max_seconds):
    """Stream (t, tilt_deg, |accel|) for the whole window.

    No early exit: there is no reliable in-band slip event to break on (see the module
    docstring), so the operator simply stops tilting when they see it go and the plateau
    that leaves in the trace is the measurement.
    """
    trace = []
    for ts, s in link.imu_stream(max_seconds):
        if s.accel is None:
            continue
        trace.append((ts, _tilt_of(s), float(np.linalg.norm(s.accel))))
    return trace


def _smooth(x, w=SMOOTH_SAMPLES):
    """Moving median. Rejects the accelerometer's spiky outliers, which a mean would smear."""
    if len(x) < w:
        return np.asarray(x, dtype=float)
    k = w // 2
    p = np.pad(np.asarray(x, dtype=float), k, mode="edge")
    return np.array([np.median(p[i:i + w]) for i in range(len(x))])


def _analyse(trace):
    """Return (slip_angle_deg, mu, reason) or None.

    The slip angle is the PLATEAU of the filtered tilt: the operator raises the board until
    the robot lets go, then stops, so the trace rises and then holds. Taking the raw maximum
    instead -- as the first version did -- reads the tallest noise spike and overestimates by
    1-3 deg.
    """
    if len(trace) < 4 * SMOOTH_SAMPLES:
        return None
    tilt = _smooth([t[1] for t in trace])
    peak = float(tilt.max())
    if peak < MIN_PLAUSIBLE_TILT:
        return None
    if peak > MAX_PLAUSIBLE_TILT:
        return (peak, math.tan(math.radians(peak)),
                f"REJECTED: reached {peak:.0f} deg -- robot toppled, not slid")

    # Longest run within PLATEAU_BAND_DEG of the peak. Anything shorter than ~1 s is the ramp
    # passing through, not the operator having stopped.
    inband = tilt >= peak - PLATEAU_BAND_DEG
    best_len = best_end = 0
    run = 0
    for i, v in enumerate(inband):
        run = run + 1 if v else 0
        if run > best_len:
            best_len, best_end = run, i
    if best_len < PLATEAU_MIN_SAMPLES:
        return (peak, math.tan(math.radians(peak)),
                f"no plateau -- peak {peak:.0f} deg is a LOWER BOUND, tilt further and hold")
    theta = float(np.median(tilt[best_end - best_len + 1:best_end + 1]))
    return theta, math.tan(math.radians(theta)), f"plateau, {best_len / 250.0:.1f}s"
