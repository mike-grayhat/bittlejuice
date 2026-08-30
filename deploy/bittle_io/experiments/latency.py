"""Command -> motion latency.

WHY THIS MATTERS, with two traps in measuring it. Delay ONLY what reaches the plant: the
observation must still get a_t while the plant gets a_{t-k}. Feeding the delayed action back
into `last_action` shows the policy an action it never emitted, and the resulting collapse
(0.087 -> 0.016 m/s) looks exactly like a hard latency cliff.

Implemented correctly, latency APPEARS to cost only ~3% of forward speed out to 500 ms, on the
argument that a mostly open-loop policy can only be phase-shifted by a uniform delay.

THAT IS FALSE, and the reason is worth knowing before trusting any latency sweep: the 3% was
measured on a plant whose actuator was so sluggish it dominated everything. Once kv/armature
are fitted to the hardware step response, dead time stops being masked. Re-measured on a
trained policy, two seeds, domain randomisation on:

    dead time      0 ms   20 ms   20-60   60-120(now)  120-200  200-320
    fwd m/s        .091    .090    .081      .064         .048     .035

Cutting today's 60-120 ms to 20-60 ms is worth ~27% of forward speed. That is the number that
makes the firmware interpolation patch worth doing, and it makes this measurement load-bearing
rather than merely interesting. See mj_vec_env.ACTION_LATENCY_STEPS_RANGE.

TWO METHODS, because the first one cannot answer it.

  poll -- the original. Sends a step, then polls `f<joint>` until the reading moves.
      Three problems, all measured: each poll costs ~89 ms, so 10 trials all detected motion
      on exactly poll 3 and reported 265 +-4 ms -- that number is 3x the poll cost, not the
      robot. The poll also physically DETACHES the servo to read it, perturbing the motion
      being timed. And its resolution is one poll period. Kept only for comparison.

  imu  -- the default. The `gP` stream runs at ~250 Hz (4 ms resolution) without polling and
      without touching the servo, so it neither costs a round trip nor disturbs the thing it
      measures. Fire a step at a known host timestamp, then find the first sustained
      departure of body attitude from its baseline.

WHAT THE IMU METHOD MEASURES, precisely: host-writes-byte -> body-attitude-visibly-moves.
That is serial transport + firmware parse/dispatch + servo start + mechanical response +
IMU sensing and fusion. It is the whole loop except host compute, which is what the policy
actually experiences, and it is an UPPER bound on the pure transport delay.

To split it, `--phase echo` separately times a request/response round trip (`P`, battery
voltage) that involves no mechanics at all. The difference between the two isolates how much
of the latency is mechanical + sensing rather than link + firmware.

Run SUSPENDED. On the ground the body cannot move until leg forces overcome stiction and
the ground reaction, which adds a mechanical delay that has nothing to do with transport;
hanging free, the reaction torque of a swinging leg acts on the body immediately.

SPLITTING OFF THE OBSERVATION SHARE: `--phase fusion` and `--phase tilt` both compare the raw
accelerometer against the fused attitude on the same printed line. `tilt` is the one to
trust -- see its docstring for why the actuated version is only an estimate.
"""

import statistics

import numpy as np

NAME = "latency"
HELP = "command -> motion delay (IMU-timed, 250 Hz); --phase echo for link round-trip"

# Attitude noise is 0.085 deg std (imu_static.csv, 60 s stationary), so 6 sigma is ~0.5 deg --
# far above noise, far below the several degrees a commanded swing produces.
ONSET_SIGMA = 6.0
# ...and it must persist, or a single noisy sample sets the timestamp. Three samples is 12 ms
# at 250 Hz, small next to the 20-80 ms we are trying to resolve. The reported onset is the
# FIRST sample of the run, not the last -- the run only confirms the event.
ONSET_RUN = 3
BASELINE_S = 0.6            # quiet window before the step, for mean and sigma
# Gap between trials. Raised from 1.2 s as a precaution, NOT as the fix -- see below.
SETTLE_S = 3.0
# A settled, suspended robot measures 0.12 deg pitch stdev (`imu --phase static`; 0.085 deg
# on the bench). Above that the trial's premise is broken and its number is meaningless, so
# reject rather than average.
#
# WHAT THIS GATE CAUGHT, and the wrong turn on the way. A run returned a mixture: some trials
# with no detectable onset, some near-zero or negative, and a cluster at 164-212 ms against a
# settled reference of 103 ms, alternating trial by trial. The obvious reading was residual
# body ring from the previous large swing, since a suspended robot has almost nothing damping
# it. That reading was WRONG, and this gate is what proved it: raising SETTLE_S changed
# nothing and the sigma gate never once fired, i.e. every baseline was quiet.
#
# The real cause was Link's input buffer. A trial `break`s the moment it detects onset and
# leaves the rest of the stream unconsumed, and `_lines` carries ARRIVAL timestamps, so the
# next trial's 0.6 s baseline window fills instantly from the PREVIOUS trial's samples. The
# mean and sigma then describe the previous trial's attitude: quiet (hence no sigma trip) but
# in the wrong place. Fixed by Link.flush_input() before each baseline window.
#
# The lesson worth keeping: a gate that never fires is evidence. It refuted the hypothesis
# that motivated it, which is more than it was built to do.
MAX_BASELINE_SIGMA_DEG = 0.35


def add_args(p):
    p.add_argument("--phase", choices=("imu", "echo", "poll", "fusion", "tilt"), default="imu")
    p.add_argument("--joint", type=int, default=8)
    p.add_argument("--amp", type=float, default=30.0,
                   help="step size in degrees; larger gives a cleaner onset")
    p.add_argument("--trials", type=int, default=15)
    p.add_argument("--settle", type=float, default=SETTLE_S,
                   help="seconds between trials; must outlast the body ring or the baseline "
                        "is contaminated and the trial is silently wrong")
    p.add_argument("--max-baseline-sigma", type=float, default=MAX_BASELINE_SIGMA_DEG,
                   help="reject a trial whose pre-step pitch stdev (deg) exceeds this -- it "
                        "means the previous swing has not died out")
    p.add_argument("--timeout", type=float, default=1.0, help="give up on a trial after this")
    p.add_argument("--poll-timeout", type=float, default=0.15)
    p.add_argument("--thresh", type=float, default=2.0, help="poll method only: degrees moved")
    # -- tilt phase: operator-driven, quasi-static ---------------------------------------
    # The gates below are the point of this phase. The actuated `fusion` phase returned a
    # number with no way to tell whether its premise held; every one of these rejects a
    # window in which the accelerometer has stopped being a tilt sensor.
    p.add_argument("--duration", type=float, default=45.0,
                   help="tilt phase: seconds of hand-tilt recording")
    p.add_argument("--window", type=float, default=8.0, help="tilt phase: analysis window [s]")
    p.add_argument("--hop", type=float, default=4.0, help="tilt phase: window hop [s]")
    p.add_argument("--min-corr", type=float, default=0.90,
                   help="tilt phase: reject a window whose peak correlation is below this. "
                        "Under quasi-static tilt both channels see the SAME angle, so a real "
                        "window correlates ~0.99; the actuated phase managed only 0.23-0.51.")
    p.add_argument("--min-amp", type=float, default=3.0,
                   help="tilt phase: reject a window whose accel-pitch std is below this [deg] "
                        "-- too little motion and the lag is fitted to noise")
    p.add_argument("--max-freq", type=float, default=1.5,
                   help="tilt phase: reject a window whose dominant frequency exceeds this [Hz]. "
                        "This is the gate that enforces 'quasi-static': above it, tangential "
                        "acceleration contaminates the accel channel and the premise fails.")
    p.add_argument("--min-cycles", type=float, default=3.0,
                   help="tilt phase: reject a window holding fewer than this many tilt cycles. "
                        "The phase estimator fits a slope across frequency bins, and a window "
                        "with ~1 cycle puts all its energy in the FIRST bin, where there is no "
                        "slope to fit -- measured on the pilot run, where 0.17 Hz in a 6 s "
                        "window scattered phase 1.9-3.8 ms against a tight 5.1-5.2 ms from the "
                        "derivative fit. Raise --window or nod faster.")
    p.add_argument("--gain-band", type=float, nargs=2, default=(0.7, 1.4), metavar=("LO", "HI"),
                   help="tilt phase: accepted fused/accel amplitude ratio. ~1 confirms the two "
                        "channels are measuring the same physical angle; the actuated phase "
                        "showed 1-3 deg of fused against 8-13 deg of accel, i.e. ~0.2.")


def run(link, caps, args, recorder):
    if args.phase == "tilt":
        return _tilt(link, caps, args, recorder)
    if args.phase == "fusion":
        return _fusion(link, caps, args, recorder)
    if args.phase == "echo":
        return _echo(link, args, recorder)
    if args.phase == "poll":
        return _poll(link, args, recorder)
    return _imu(link, caps, args, recorder)


# ---------------------------------------------------------------- IMU method

def _imu(link, caps, args, recorder):
    from .. import protocol as P

    if caps.imu_chip is None and not link.dry_run:
        raise RuntimeError("probe found no IMU -- run `probe` first")

    link.gyro_balance_off()
    print("\n[latency] IMU-timed. SUSPEND the robot with the legs hanging free.")
    print("  On the ground, stiction delays the body response and inflates the number.\n")
    link.prompt("  Press Enter when suspended -> ")

    rows, lats, unsettled = [], [], 0
    with recorder.bracket_voltage(NAME, link):
        link.wake()
        # All four shoulders together: the reaction torque of four legs swinging the same way
        # is a clean body pitch, several degrees, well clear of the 0.085 deg noise floor.
        # 8/9/10/11 are LF/RF/RB/LB shoulders -- the mapping jointmap validated four ways.
        joints = [8, 9, 10, 11]

        for trial in range(args.trials):
            link.send(P.move_cmd([(j, 32) for j in joints]))
            link.sleep(args.settle)
            # Discard the settle move and anything the previous trial left unconsumed --
            # see Link.flush_input. Without this the baseline is computed from the previous
            # trial's samples and every number below is fiction.
            link.flush_input()

            base, t_cmd, onset, run_len = [], None, None, 0
            mu = sigma = None
            for ts, s in link.imu_stream(BASELINE_S + args.timeout):
                pitch = float(s.ypr_deg[1])
                if t_cmd is None:
                    base.append((ts, pitch))
                    if base and ts - base[0][0] >= BASELINE_S:
                        vals = [b[1] for b in base]
                        mu = statistics.fmean(vals)
                        sigma = max(statistics.pstdev(vals), 0.02)   # floor: never divide by ~0
                        if sigma > args.max_baseline_sigma:
                            break       # premise broken -- do not even fire the step
                        # Timestamp taken as close to the write as possible.
                        t_cmd = link.clock()
                        link.send(P.move_cmd([(j, 32 - args.amp) for j in joints]))
                    continue
                if abs(pitch - mu) > ONSET_SIGMA * sigma:
                    run_len += 1
                    if run_len == 1:
                        first = ts
                    if run_len >= ONSET_RUN:
                        onset = first
                        break
                else:
                    run_len = 0

            if sigma is not None and sigma > args.max_baseline_sigma:
                unsettled += 1
                print(f"  trial {trial + 1}/{args.trials}: baseline unsettled "
                      f"(sigma {sigma:.3f} deg > {args.max_baseline_sigma:.2f}) -- "
                      f"the previous swing is still ringing; raise --settle")
                continue
            if onset is None or t_cmd is None:
                print(f"  trial {trial + 1}/{args.trials}: no onset -- raise --amp")
                continue
            lat = (onset - t_cmd) * 1000.0
            if lat < 0:
                # Onset timestamped before the command was written: the sample that tripped
                # the threshold predates t_cmd, so it is residual motion, not a response.
                # Averaging it in drags the median toward zero.
                unsettled += 1
                print(f"  trial {trial + 1}/{args.trials}: {lat:6.1f} ms -- onset PRECEDES "
                      f"the command; residual motion, rejected")
                continue
            lats.append(lat)
            rows.append({"trial": trial, "latency_ms": round(lat, 1),
                         "baseline_sigma_deg": round(sigma, 3)})
            print(f"  trial {trial + 1}/{args.trials}: {lat:6.1f} ms")

    recorder.write_csv("latency_imu.csv", rows)
    if not lats:
        print("\nNo usable trials.")
        return {"trials": 0}

    med = statistics.median(lats)
    res = {"trials": len(lats), "latency_ms_median": round(med, 1),
           "latency_ms_min": round(min(lats), 1), "latency_ms_max": round(max(lats), 1),
           "method": "imu"}
    print(f"\ncommand -> body motion: median {med:.0f} ms "
          f"(range {min(lats):.0f}-{max(lats):.0f}, {len(lats)} trials)")
    print("  includes serial + firmware + servo start + mechanics + IMU fusion.")
    print("  This number is worth real forward speed: cutting 60-120 ms to 20-60 ms is")
    print("  worth ~27% of it. Beware sweeps run against a too-sluggish actuator model --")
    print("  they mask dead time and make latency look free. Use `--phase tilt` to split")
    print("  out the OBSERVATION share.")
    if unsettled:
        print(f"  {unsettled} trial(s) rejected for an unsettled baseline. If that is most")
        print("  of them, raise --settle: the body ring is outlasting the gap between trials.")
    recorder.note(NAME, **res)
    return res


# --------------------------------------------------------------- echo method

def _echo(link, args, recorder):
    """Link + firmware round trip with no mechanics involved.

    `P` (battery voltage) is a pure request/response: the firmware parses a token and prints
    a number. Subtracting this from the IMU-timed figure leaves the mechanical + sensing
    share, which is the part sim already models via the actuator lag.
    """
    from .. import protocol as P

    rows, rtts = [], []
    with recorder.bracket_voltage(NAME, link):
        for trial in range(args.trials):
            link.sleep(0.05)
            link._pump()
            while link._lines:                 # start from a clean buffer
                link._lines.pop(0)
            t0 = link.clock()
            link.send("P")
            got = None
            for ts, text in link.drain(args.timeout):
                if any(ch.isdigit() for ch in text):
                    got = ts
                    break
            if got is None:
                continue
            rtt = (got - t0) * 1000.0
            rtts.append(rtt)
            rows.append({"trial": trial, "rtt_ms": round(rtt, 1)})

    recorder.write_csv("latency_echo.csv", rows)
    if not rtts:
        print("\nNo responses.")
        return {"trials": 0}
    med = statistics.median(rtts)
    res = {"trials": len(rtts), "rtt_ms_median": round(med, 1),
           "rtt_ms_min": round(min(rtts), 1), "method": "echo"}
    print(f"\nlink+firmware round trip: median {med:.1f} ms "
          f"(min {min(rtts):.1f}, {len(rtts)} trials)")
    print("  This is the floor for any command reaching the servo. One-way is roughly half,")
    print("  and the rest of the IMU-timed figure is servo start + mechanics + IMU fusion.")
    recorder.note(NAME + "_echo", **res)
    return res


# --------------------------------------------------------------- poll method

def _poll(link, args, recorder):
    """The original. Retained for comparison only -- see the module docstring for why its
    265 ms answer is three poll periods rather than a property of the robot."""
    from .. import protocol as P

    link.gyro_balance_off()
    with recorder.bracket_voltage(NAME, link):
        link.wake()
        lats, periods, rows = [], [], []
        for trial in range(args.trials):
            link.send(P.move_cmd([(args.joint, 0)]))
            link.sleep(0.8)
            base = link.poll_joint(args.joint, timeout=args.poll_timeout)
            if base is None:
                continue
            t0 = link.clock()
            link.send(P.move_cmd([(args.joint, int(args.amp))]))
            polls, seen = 0, None
            while link.clock() - t0 < args.timeout:
                v = link.poll_joint(args.joint, timeout=args.poll_timeout)
                polls += 1
                if v is not None and abs(v - base) > args.thresh:
                    seen = link.clock()
                    break
            if seen is None:
                continue
            lat = (seen - t0) * 1000.0
            lats.append(lat)
            periods.append(lat / max(polls, 1))
            rows.append({"trial": trial, "latency_ms": round(lat, 2), "polls": polls,
                         "baseline_deg": round(base, 1)})
    recorder.write_csv("latency_poll.csv", rows)
    if not lats:
        return {"trials": 0}
    med = statistics.median(lats)
    print(f"\npolled latency {med:.0f} ms over {len(lats)} trials, "
          f"poll period ~{statistics.median(periods):.0f} ms")
    print("  UPPER BOUND, and dominated by the poll period -- use --phase imu.")
    res = {"trials": len(lats), "latency_ms_median": round(med, 1),
           "poll_period_ms": round(statistics.median(periods), 1), "method": "poll"}
    recorder.note(NAME + "_poll", **res)
    return res


# ------------------------------------------------------------- fusion method

def _fusion(link, caps, args, recorder):
    """How much of the latency belongs to the IMU's attitude FUSION, specifically.

    This is the one measurement in this file that does not depend on host timestamps, serial
    transport, or when the command was sent -- and therefore cannot be contaminated by any of
    them. print6Axis emits the raw accelerometer AND the fused yaw/pitch/roll on the SAME LINE,
    from the same sample instant. So a single disturbance appears in two channels that differ
    only by the filtering between them, and the lag between those two waveforms IS the fusion
    delay. Everything common to both cancels.

    Why it matters now rather than earlier: while the policy ignores attitude, this delay is
    unobservable. The moment terrain or larger pushes make attitude worth using, it lands
    inside the control loop, and IMU_DELAY_STEPS_RANGE needs to be a measurement rather than
    the guess it currently is. See that constant's comment.

    Measured by CROSS-CORRELATION rather than by comparing onsets: an onset test needs a
    threshold on each channel, and the two channels have different noise and different scaling,
    so the thresholds would bias the answer. Cross-correlation uses the whole waveform.
    """
    from .. import protocol as P

    if caps.imu_has_accel is False:
        raise RuntimeError("this firmware build does not print acceleration; the fusion "
                           "measurement needs both channels on the same line")

    link.gyro_balance_off()
    print("\n[latency --phase fusion] SUPERSEDED by --phase tilt; see that docstring.")
    print("  SUSPEND the robot, legs hanging free.\n")
    link.prompt("  Press Enter when suspended -> ")

    lags, rows = [], []
    with recorder.bracket_voltage(NAME, link):
        link.wake()
        joints = [8, 9, 10, 11]
        for trial in range(args.trials):
            link.send(P.move_cmd([(j, 32) for j in joints]))
            link.sleep(args.settle)
            link.flush_input()      # same reason as _imu; see Link.flush_input

            ts, acc, fus, sent = [], [], [], False
            t0 = None
            for t, s in link.imu_stream(BASELINE_S + args.timeout):
                if s.accel is None:
                    continue
                ts.append(t)
                # Raw channel: the accelerometer's own view of PITCH -- it must be the same
                # axis as the fused channel below or there is nothing to correlate. From the
                # validated convention (protocol.projected_gravity_from_accel, cross-checked
                # against mj_vec_env._inv_rotate in six orientations), body gravity is
                # [a0, -a1, -a2]/|a| and gravity[0] = sin(pitch). So pitch comes from a0.
                # The first version used arctan2(a1, -a2), which is ROLL, and duly found no
                # correlation with fused pitch on any of 12 trials.
                v = np.asarray(s.accel, dtype=float)
                n_ = float(np.linalg.norm(v)) or 1.0
                acc.append(float(np.degrees(np.arcsin(np.clip(v[0] / n_, -1.0, 1.0)))))
                fus.append(float(s.pitch))          # the DMP/firmware-fused estimate
                if not sent and t - ts[0] >= BASELINE_S:
                    t0 = link.clock()
                    link.send(P.move_cmd([(j, 32 - args.amp) for j in joints]))
                    sent = True
            if len(ts) < 200 or not sent:
                print(f"  trial {trial + 1}/{args.trials}: too few samples")
                continue

            # Always keep the trace: a failed correlation is only debuggable offline, and
            # the first run of this experiment failed all 12 trials on an axis mistake.
            t_arr = np.array(ts)
            recorder.write_csv(f"fusion_trace_{trial}.csv",
                               [{"t": round(t - t_arr[0], 4), "accel_pitch": round(a, 3),
                                 "fused_pitch": round(f, 3)}
                                for t, a, f in zip(t_arr, acc, fus)])
            lag = _xcorr_lag(t_arr, np.array(acc), np.array(fus))
            if lag is None:
                print(f"  trial {trial + 1}/{args.trials}: no usable correlation")
                continue
            lags.append(lag)
            rows.append({"trial": trial, "fusion_lag_ms": round(lag, 1)})
            print(f"  trial {trial + 1}/{args.trials}: fused attitude lags raw accel "
                  f"by {lag:6.1f} ms")

    recorder.write_csv("latency_fusion.csv", rows)
    if not lags:
        print("\nNo usable trials.")
        return {"trials": 0}
    med = float(np.median(lags))
    steps = med / 20.0
    print(f"\nattitude fusion lag: median {med:.0f} ms "
          f"(range {min(lags):.0f}-{max(lags):.0f}, {len(lags)} trials)")
    print(f"  = {steps:.1f} control steps at 50 Hz.")
    print(f"  This is the share of the 103 ms command->motion figure that belongs to the")
    print(f"  OBSERVATION path. Set mj_vec_env.IMU_DELAY_STEPS_RANGE around {max(round(steps)-1,0)}"
          f"-{round(steps)+1}; the remainder of the 103 ms belongs to the ACTION path")
    print(f"  (firmware move interpolation) and to ACTION_LATENCY_STEPS_RANGE.")
    res = {"trials": len(lags), "fusion_lag_ms_median": round(med, 1),
           "fusion_lag_steps": round(steps, 2)}
    recorder.note(NAME + "_fusion", **res)
    return res


# --------------------------------------------------------------- tilt method

# Expected answer is ~20 ms; searching +-150 ms leaves room to be wrong by an order of
# magnitude while keeping the peak unambiguous. Negative lags are searched deliberately:
# a gyro-driven filter that predicts forward can genuinely LEAD the accelerometer, and a
# search that cannot express that would report 0 and hide it.
TILT_MAX_LAG_MS = 150.0
# Central-difference half-width for the derivative estimator. 8 samples at ~250 Hz is 32 ms
# -- long enough to average down accel noise, short enough that a 2 s tilt cycle is locally
# linear over the stencil.
DERIV_HALF = 8


def _tilt(link, caps, args, recorder):
    """Fusion lag from a SLOW hand tilt. This is the trustworthy version of `--phase fusion`.

    Both phases rest on the same identity: print6Axis emits raw accel and fused ypr on one
    line from one sample instant, so everything upstream (serial, firmware dispatch, host
    timestamping) is common-mode and cancels, and what remains between the two waveforms is
    the fusion filter. That part is sound and is why this measurement needs no host clock.

    What the ACTUATED phase got wrong is the other premise -- that the accelerometer reports
    tilt. It reports SPECIFIC FORCE, a_body = R'g - a_linear. Snap four shoulders and the body
    swings; the tangential term dominates, and the accel channel reports an "angle" that is
    mostly not an angle. The fused channel correctly refuses to believe it, because a
    complementary filter trusts the gyro at high frequency and uses accel only to correct
    low-frequency drift. Measured: accel swung 8-13 deg while fused moved 1-3 deg, peak
    correlation 0.23-0.51. The two channels had stopped measuring the same quantity, so the
    lag between them stopped meaning anything. The property that makes the fusion GOOD is
    exactly what breaks that experiment.

    Tilting by hand at ~0.5 Hz drives a_linear to near zero, so both channels report the same
    tilt and the premise holds. The cost is that the operator is in the loop, so this phase
    verifies rather than assumes: every window must clear a correlation, amplitude, frequency
    and GAIN gate, the last being the direct test that the two channels agree on the angle.

    TWO INDEPENDENT ESTIMATORS, because a single one cannot be checked:

      deriv -- for a lag small against the signal's timescale, fused(t) ~ accel(t - tau)
          ~ accel(t) - tau*accel'(t). Regressing fused on [accel, accel'] gives gain and
          -gain*tau, hence tau. Time domain, uses every sample, no lag grid at all.
      phase -- the cross-spectrum's phase is -2*pi*f*tau, so tau is the slope of phase
          against frequency. Frequency domain, and sharpest exactly where a smooth signal
          has its energy. Fails differently from the regression, which is the point.

    Agreement between the two is the evidence; disagreement means neither should be believed.

    NOT cross-correlation, which is what `--phase fusion` used and is wrong for this regime.
    Quasi-static tilt is band-limited below ~1 Hz, so shifting by one 4 ms sample barely
    changes the overlap: measured on synthetic data with a KNOWN 20 ms lag, the correlation
    peak was 0.9993 against 0.9992 at both neighbours, and the argmax -- picked out of that
    noise -- came back 12.5 ms. Across 8/20/44/80 ms of true lag it read consistently ~8 ms
    short, while these two estimators land within 0.3 ms (tests/test_fusion_lag_estimators.py
    pins all of it). The low frequency that makes the tilt quasi-static is precisely what
    flattens the correlation ridge; the two requirements are in direct conflict. Correlation
    is still computed here, but only as a GATE -- its peak VALUE is a fine measure of whether
    the channels agree, even though its LOCATION is not a usable lag.
    """
    from .. import protocol as P

    if caps.imu_has_accel is False:
        raise RuntimeError("this firmware build does not print acceleration; the fusion "
                           "measurement needs both channels on the same line")

    # Servos stay at rest: this phase needs no actuation, and a powered servo adds vibration
    # to the accelerometer for nothing. Balancing off is still required -- with it on, the
    # firmware would fight the tilt and change the very attitude being measured.
    link.gyro_balance_off()
    print("\n[latency --phase tilt] Fusion lag from a slow hand tilt.")
    print("  Hold the robot OFF the ground in both hands.")
    print("  NOD it nose-down / nose-up about the pitch axis, about +-25 deg.")
    print("  One full nod every 2-4 s. Slow and continuous -- no flicks, no pauses.")
    print("  VARY the speed between nods. A single frequency gives the phase fit one point")
    print("    to put a line through; several let it check that the lag really is constant")
    print("    across frequency, which is what distinguishes a delay from a filter.")
    print("  ROTATE the body, do not lift it up and down. Translation is precisely the")
    print("    contamination this phase exists to avoid -- keep the centre still.")
    print("  SILENCE THE VOICE MODULE first (say \"be quiet\", or unplug it). This phase")
    print("    sends nothing for the whole recording, so unlike a 50 Hz policy run it")
    print("    cannot hold the firmware's idle timer off -- see the chatter check below.")
    print(f"  Recording {args.duration:.0f} s.\n")
    link.prompt("  Press Enter to start -> ")

    chatter0 = link.chatter
    ts, acc, fus = [], [], []
    with recorder.bracket_voltage(NAME, link):
        next_mark = 5.0
        for t, s in link.imu_stream(args.duration):
            if s.accel is None:
                continue
            # Same validated convention as _fusion: body gravity is [a0,-a1,-a2]/|a| and
            # gravity[0] = sin(pitch), cross-checked against mj_vec_env._inv_rotate.
            v = np.asarray(s.accel, dtype=float)
            n_ = float(np.linalg.norm(v)) or 1.0
            ts.append(t)
            acc.append(float(np.degrees(np.arcsin(np.clip(v[0] / n_, -1.0, 1.0)))))
            fus.append(float(s.pitch))
            if ts[-1] - ts[0] >= next_mark:
                print(f"    {next_mark:4.0f}s  accel +-{np.std(acc[-500:]):5.1f} deg  "
                      f"fused +-{np.std(fus[-500:]):5.1f} deg")
                next_mark += 5.0

    if len(ts) < 500:
        print(f"\nOnly {len(ts)} samples -- the stream did not run. Nothing to analyse.")
        return {"trials": 0}

    # Nothing but IMU frames should arrive: this phase transmits no command for the whole
    # recording, so any other line means the firmware acted on its own -- a voice-triggered
    # skill or a randomMind demo. Both drive the servos, and servo motion is exactly the
    # linear acceleration the quasi-static premise rules out.
    noise = link.chatter - chatter0
    if noise:
        print(f"\n  WARNING: {noise} non-IMU lines during the recording -- the robot was "
              f"taking orders from something else.")
        for line in link.chatter_lines[-5:]:
            print(f"    {line!r}")
        print("  Silence the voice module and re-run; the windows it touched are not valid.")

    dt, grid, (a_u, f_u) = _uniform(ts, acc, fus)
    recorder.write_csv("fusion_tilt_trace.csv",
                       [{"t": round(t - grid[0], 4), "accel_pitch": round(a, 3),
                         "fused_pitch": round(f, 3)}
                        for t, a, f in zip(grid, a_u, f_u)])
    print(f"\n  {len(grid)} samples at {1.0 / dt:.0f} Hz over {grid[-1] - grid[0]:.1f} s")

    w = max(int(args.window / dt), 100)
    hop = max(int(args.hop / dt), 1)
    lo, hi = args.gain_band
    rows, good = [], []
    for i, start in enumerate(range(0, max(len(grid) - w, 0) + 1, hop)):
        est = _window_estimate(dt, a_u[start:start + w], f_u[start:start + w])
        est["window"] = i
        est["t_start"] = round(grid[start] - grid[0], 1)
        est["cycles"] = round(est["freq_hz"] * w * dt, 1)
        why = []
        if est["corr"] < args.min_corr:
            why.append(f"corr {est['corr']:.2f}")
        if est["amp_deg"] < args.min_amp:
            why.append(f"amp {est['amp_deg']:.1f}deg")
        if est["freq_hz"] > args.max_freq:
            why.append(f"freq {est['freq_hz']:.2f}Hz")
        if est["cycles"] < args.min_cycles:
            why.append(f"cycles {est['cycles']:.1f}")
        if not (lo <= est["gain"] <= hi):
            why.append(f"gain {est['gain']:.2f}")
        est["rejected"] = ";".join(why)
        rows.append(est)
        flag = "REJECT " + est["rejected"] if why else "ok"
        print(f"  win {i:2d} @{est['t_start']:5.1f}s  deriv {est['lag_deriv_ms']:6.1f} ms  "
              f"phase {est['lag_phase_ms']:6.1f} ms  r={est['corr']:.3f} "
              f"gain={est['gain']:.2f} {est['freq_hz']:.2f}Hz x{est['cycles']:.1f}  {flag}")
        if not why:
            good.append(est)

    recorder.write_csv("latency_fusion_tilt.csv", rows)
    if not good:
        print(f"\nNo window passed the gates ({len(rows)} tried). The recording is telling you")
        print("  the tilt was not quasi-static -- most often too fast, or the body was")
        print("  translated rather than rotated. Re-run and slow down.")
        print("  A negative correlation instead would mean the accel-pitch sign is wrong.")
        return {"trials": 0, "windows_rejected": len(rows)}

    dv = [g["lag_deriv_ms"] for g in good if np.isfinite(g["lag_deriv_ms"])]
    ph = [g["lag_phase_ms"] for g in good if np.isfinite(g["lag_phase_ms"])]
    if not dv or not ph:
        print("\nAccepted windows but no finite lag estimate -- inspect fusion_tilt_trace.csv.")
        return {"trials": 0, "windows": len(rows)}
    med_d, med_p = float(np.median(dv)), float(np.median(ph))
    spread = abs(med_d - med_p)
    med = 0.5 * (med_d + med_p)
    steps = med / 20.0
    print(f"\nfusion lag over {len(good)}/{len(rows)} accepted windows:")
    print(f"  derivative fit  median {med_d:6.1f} ms   (range {min(dv):.1f} to {max(dv):.1f})")
    print(f"  phase slope     median {med_p:6.1f} ms   (range {min(ph):.1f} to {max(ph):.1f})")
    if spread > 10.0:
        print(f"  ESTIMATORS DISAGREE by {spread:.0f} ms -- believe neither. They fail in")
        print("    different ways by design, so a split this wide means an assumption broke.")
        print("    Inspect fusion_tilt_trace.csv before using either number.")
    else:
        print(f"  the two agree to {spread:.1f} ms; take {med:.0f} ms = {steps:.2f} control "
              f"steps at 50 Hz.")
        print(f"  Set mj_vec_env.IMU_DELAY_STEPS_RANGE around "
              f"{max(round(steps) - 1, 0)}-{round(steps) + 1}. The remainder of the 103 ms "
              f"command->motion")
        print("  figure then belongs to the ACTION path (firmware move interpolation).")
    res = {"trials": len(good), "windows": len(rows),
           "fusion_lag_ms_deriv": round(med_d, 1), "fusion_lag_ms_phase": round(med_p, 1),
           "estimator_spread_ms": round(spread, 1),
           "fusion_lag_ms": round(med, 1), "fusion_lag_steps": round(steps, 2),
           "method": "tilt"}
    recorder.note(NAME + "_tilt", **res)
    return res


def _uniform(ts, *chans):
    """Resample channels onto an evenly spaced grid at the stream's median rate.

    Safe for this measurement specifically: both channels come off the SAME line, so serial
    jitter displaces them identically and cancels in their relative lag. The interpolation is
    a mild low-pass, applied to both alike, so it too is common-mode.
    """
    t = np.asarray(ts, dtype=float)
    dt = float(np.median(np.diff(t)))
    grid = np.arange(t[0], t[-1], dt)
    return dt, grid, [np.interp(grid, t, np.asarray(c, dtype=float)) for c in chans]


def _window_estimate(dt, a, b):
    """Both lag estimators plus the diagnostics the gates read, for one window."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    ad, bd = a - a.mean(), b - b.mean()
    sa, sb = float(ad.std()), float(bd.std())

    # -- correlation, for the GATE only --------------------------------------------
    # Peak value over the plausible lag range: "do these two channels agree at all", which
    # is a question correlation answers well. Its argmax is NOT read -- see the docstring.
    corr = 0.0
    if sa > 1e-6 and sb > 1e-6:
        max_k = min(int(TILT_MAX_LAG_MS / 1000.0 / dt), len(a) // 4)
        corr = max(_shifted_corr(ad, bd, k) for k in range(-max_k, max_k + 1))

    # -- phase slope of the cross-spectrum -----------------------------------------
    lag_p = _phase_slope_lag(dt, ad, bd)

    # -- derivative regression -----------------------------------------------------
    # fused ~ gain * (accel - tau*accel'), so the design matrix is [accel, accel'].
    lag_d, gain = float("nan"), float("nan")
    da = _deriv(ad, dt)
    n = len(ad) - 2 * DERIV_HALF
    if n > 50:
        s = slice(DERIV_HALF, len(ad) - DERIV_HALF)
        X = np.column_stack([ad[s], da[s]])
        try:
            coef, *_ = np.linalg.lstsq(X, bd[s], rcond=None)
            gain = float(coef[0])
            if abs(gain) > 1e-6:
                lag_d = -float(coef[1]) / gain * 1000.0
        except np.linalg.LinAlgError:
            pass
    if not np.isfinite(gain):
        gain = sb / sa if sa > 1e-6 else float("nan")

    return {"lag_phase_ms": round(lag_p, 1) if np.isfinite(lag_p) else float("nan"),
            "lag_deriv_ms": round(lag_d, 1) if np.isfinite(lag_d) else float("nan"),
            "corr": round(corr, 3),
            "gain": round(gain, 3) if np.isfinite(gain) else float("nan"),
            "amp_deg": round(sa, 2),
            "freq_hz": round(_dominant_freq(ad, dt), 3)}


# Hand tilt lives well under 1 Hz; above PHASE_FMAX there is only noise, and noise-only bins
# contribute meaningless phase. Below PHASE_FMIN a window holds barely one cycle, so the
# phase estimate there is dominated by the window edges.
PHASE_FMIN, PHASE_FMAX = 0.1, 3.0
# Bins carrying less than this share of peak power are excluded from the fit for the same
# reason: their phase is noise, and an unweighted fit would let them dominate by count.
PHASE_MIN_POWER = 0.02


def _phase_slope_lag(dt, a, b):
    """Lag in ms from the slope of the cross-spectrum phase; positive = b lags a.

    A pure delay tau maps to phase(f) = -2*pi*f*tau, a straight line through the origin, so
    the fit is one weighted slope with no intercept -- an intercept would absorb exactly the
    constant offset a real delay cannot produce. Weighted by power, so the frequencies the
    operator actually excited decide the answer.
    """
    n = len(a)
    if n < 64:
        return float("nan")
    win = np.hanning(n)
    A = np.fft.rfft(a * win)
    B = np.fft.rfft(b * win)
    f = np.fft.rfftfreq(n, dt)
    p = np.abs(A) ** 2
    if p.max() <= 0:
        return float("nan")
    band = (f >= PHASE_FMIN) & (f <= PHASE_FMAX)
    if band.sum() < 4:
        return float("nan")
    # Unwrap across the CONTIGUOUS band before masking on power: unwrapping a sparse subset
    # would step over gaps and invent 2*pi jumps.
    lo, hi = int(np.argmax(band)), int(len(band) - np.argmax(band[::-1]))
    ph = np.unwrap(np.angle(B[lo:hi] * np.conj(A[lo:hi])))
    fb, pb = f[lo:hi], p[lo:hi]
    keep = pb > PHASE_MIN_POWER * p.max()
    if keep.sum() < 2:
        return float("nan")
    fb, pb, ph = fb[keep], pb[keep], ph[keep]
    denom = float(np.sum(pb * fb * fb))
    if denom <= 0:
        return float("nan")
    slope = float(np.sum(pb * fb * ph)) / denom          # rad per Hz
    return -slope / (2.0 * np.pi) * 1000.0


def _shifted_corr(a, b, k):
    """Pearson correlation of a against b delayed by k samples; k>0 means b lags a.

    Normalised over the OVERLAP rather than the full window. Using full-window statistics
    biases the result toward small |k|, because the truncated segments do not carry the same
    variance as the whole -- one of the two things that made the correlation estimator
    unusable here.
    """
    if k >= 0:
        x, y = a[:len(a) - k], b[k:]
    else:
        x, y = a[-k:], b[:len(b) + k]
    n = len(x)
    if n < 50:
        return -2.0
    x = x - x.mean()
    y = y - y.mean()
    d = float(np.linalg.norm(x) * np.linalg.norm(y))
    return float(np.dot(x, y) / d) if d > 1e-12 else -2.0


def _deriv(x, dt):
    """Central difference over +-DERIV_HALF samples; edges are left at zero and sliced off."""
    d = np.zeros_like(x)
    h = DERIV_HALF
    if len(x) > 2 * h:
        d[h:-h] = (x[2 * h:] - x[:-2 * h]) / (2.0 * h * dt)
    return d


def _dominant_freq(x, dt):
    """Frequency of peak power, ignoring DC. The 'is this quasi-static' number."""
    if len(x) < 16:
        return float("inf")
    spec = np.abs(np.fft.rfft(x * np.hanning(len(x))))
    freqs = np.fft.rfftfreq(len(x), dt)
    spec[0] = 0.0
    return float(freqs[int(np.argmax(spec))])


def _xcorr_lag(ts, a, b, max_lag_ms=300.0):
    """Lag (ms, positive = b lags a) maximising correlation between two channels.

    Both are detrended and normalised first, so the answer depends on waveform SHAPE rather
    than on their very different amplitudes and offsets.
    """
    if len(ts) < 100:
        return None
    dt = float(np.median(np.diff(ts)))
    if dt <= 0:
        return None
    a = a - a.mean(); b = b - b.mean()
    if a.std() < 1e-6 or b.std() < 1e-6:
        return None
    a /= a.std(); b /= b.std()
    max_k = int(max_lag_ms / 1000.0 / dt)
    best, best_k = -2.0, 0
    for k in range(0, max_k + 1):
        n = len(a) - k
        if n < 50:
            break
        c = float(np.dot(a[:n], b[k:k + n]) / n)
        if c > best:
            best, best_k = c, k
    if best < 0.3:                 # no real common signal
        return None
    return best_k * dt * 1000.0
