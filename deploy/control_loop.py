"""50 Hz control loop: read IMU -> build obs -> policy -> safety -> command servos.

Mirrors sim/mj_vec_env.py's control convention exactly: the action sent each tick is
the PREVIOUS tick's policy output (one-step delay, matching training's always-on
simulate_action_latency), not the action just computed.

Usage:
    uv run python control_loop.py --port /dev/cu.usbmodem5AA90270111 \
        --policy policy/bittle_policy_v11.npz --command 0.15 0 0 --duration 10

    # always do this first, with the robot SUSPENDED:
    ... --dry-run          # no serial writes; prints what it would send
    ... --hold             # commands the home pose only; no policy output

Design notes, all from measured hardware (see measurements/)
-------------------------------------------------------------------
* The IMU streams continuously (`gP`, ~250 Hz) and we take the newest sample each tick.
  Verified in firmware source that motion commands do NOT stop the gyro stream, unlike
  the servo feedback stream. Polling `gp` per tick would cost a round trip we don't have.
* Joint positions are never read. See obs_builder's docstring: reads are ~9 Hz, physically
  detach the servo, and were measured returning corrupt values. The host runs the same
  actuator model the simulator does -- as an OBSERVER; the wire carries the raw target.
* The servo model's tau/slew are not constants here: they live in the policy .npz
  (measured on this robot: tau 33 ms, slew 137 deg/s).
"""

import argparse
import math
import os
import time

import numpy as np

import obs_builder as ob  # noqa: E402
from bittle_io import SerialLink  # noqa: E402
from bittle_io import protocol as P  # noqa: E402
from joint_calibration import JOINT_CALIBRATION, deg_to_petoi  # noqa: E402
from policy_numpy import NumpyMLPPolicy  # noqa: E402

# Looser than the sim's 10 deg termination bound: that is an RL curriculum signal, this is
# a last-resort "the robot is actually falling" check with margin for real sensor noise.
TILT_WATCHDOG_DEG = 45.0

# A fall must PERSIST to count. The accelerometer measures gravity plus linear
# acceleration, so any sharp motion swings the apparent "down" direction wildly -- the
# snap into the home pose produced a 147.7 deg reading on a robot that never fell over.
# A real fall holds; a transient does not.
TILT_WATCHDOG_TICKS = 8                  # ~160 ms at 50 Hz
# Reject samples that are not measuring gravity in the first place. At rest |accel| ~= 9.8;
# far from that and the vector is dominated by acceleration, so its DIRECTION says nothing
# about orientation. Cheaper and more decisive than filtering the direction itself.
ACCEL_G_BAND = (7.0, 13.0)

# The actuator rate limit is NOT a constant here: it lives in the policy .npz alongside
# tau, measured on this robot, and is applied by ServoModel exactly as mj_vec_env applies
# it in sim. A module-level copy existed and went stale -- the control path had moved to
# 137 deg/s while the telemetry still printed the old 109.7, which is precisely the kind of
# duplicate that makes a run look unchanged when it was not.


# Ticks to keep commanding the home pose after the policy starts running.
#
# The policy has a large OPENING TRANSIENT out of a settled stance. Measured, mean over 16
# envs, |action| peaks at 4.14 against a steady-state 1.13 -- 3.7x, and
# past the +-3.0 clip, i.e. a commanded 43 deg step away from the stand pose. The servo
# needs ~16 ticks at its 137 deg/s limit to cover that, so from standstill it is a lurch
# rather than a stride, and on the floor it has been toppling the robot at start.
#
# Sim does not fall from it (0% over 128 envs, flat, with and without randomization), which
# is why this is a deploy-side mitigation and not a reward term -- there is nothing in sim
# to optimise against. What sim DOES confirm is that suppressing it is free: holding the
# default pose for 0/5/10/15/25 ticks all give 0% falls and an identical +0.109 m/s steady
# speed, because the transient decays by tick ~10 and the gait that follows is unchanged.
#
# The policy keeps RUNNING during the hold -- only the servo command is held, and
# last_action is held at zero with it (see the hold block below for why that pairing is the
# whole fix). The history ring fills normally. Set to 0 to disable.
# Back to 15 after the fix below. Raising this to 40 made the startup shake WORSE, which is
# what identified the real cause: the hold was inconsistent, not too short.
STARTUP_HOLD_TICKS = 15
# Seconds to ease the commanded velocity from 0 up to the requested value. The robot is
# standing still when the loop begins, so a step command is the one input guaranteed to
# demand the policy's largest action on its very first tick.
COMMAND_RAMP_S = 0.6

# Host-side pose ramps, for the two moves this loop makes one-at-a-time rather than as a
# 50 Hz stream: the step INTO the home pose at startup and the step back OUT at teardown.
#
# The firmware used to shape these for us. Every `i` command went through transform()'s
# raised-cosine interpolation at transformSpeed=2, so even a single absolute command arrived
# as a ramp. The firmware patch removes that for `i` on purpose -- at 50 Hz the
# host IS the trajectory and the ramp cost 16-24 ms of blocking per tick, which was most of
# the 198 ms command->motion latency. But it also removed the smoothing from these two
# one-shot moves, which then became single absolute commands: the servos traverse them at
# full slew (~150 deg/s) and the robot visibly jumps into its stance.
#
# So the host generates them now. Same helper both ends, non-blocking, and it is the shape
# the firmware used -- a raised cosine has zero velocity at both ends, which is what makes
# the move read as a settle rather than a lunge.
RAMP_DEG_PER_S = 60.0    # PEAK, not mean: well under what the servo can do
RAMP_MAX_S = 3.0         # ceiling, so a bad start-pose read cannot stall the run


def _ramp_floor_and_command(policy, base_command):
    """Where the startup ramp should START, and the command it should ramp TO.

    Two jobs, both about staying inside the distribution the policy was trained on.

    The FLOOR. Ramping the forward command from zero walks a command-responsive policy
    through a region it has never seen. Measured on a policy trained over 0.05-0.15 m/s: at
    command 0.00 its excess joint reversals -- the one metric that tracks how the gait looks
    to an operator -- read 1.43x, against 0.47x at 0.02. Starting the ramp at the range floor
    costs nothing: the robot is standing still either way, and the first command it acts on is
    one it was trained for.

    The CEILING. The same argument applies to the requested command itself. The default here
    is easy to leave at a value no policy was trained for -- and a policy trained at a single
    fixed speed ignores the command channel entirely, so nothing complains. The moment a
    policy actually reads it, an out-of-range request is an extrapolation with nothing holding
    it down, so clamp and say so.

    An export carrying no command ranges returns a zero floor and an unclamped command, since
    there is nothing to be consistent with.
    """
    base_command = np.asarray(base_command, dtype=np.float64).copy()
    ranges = (policy.command_vx_range, policy.command_vy_range, policy.command_wz_range)
    if ranges[0] is None:
        print("     ! policy export carries no command ranges: "
              "ramping from zero and not range-checking the command")
        return np.zeros_like(base_command), base_command

    for k, rng in enumerate(ranges):
        lo, hi = rng
        want = float(base_command[k])
        got = float(np.clip(want, lo, hi))
        if got != want:
            print(f"     ! command[{k}] {want:+.3f} is outside the trained range "
                  f"[{lo:+.3f}, {hi:+.3f}] -- clamped to {got:+.3f}")
            base_command[k] = got

    # Ramp the forward command only. Lateral and yaw commands are centred on zero in every
    # run so far, so zero is inside their range and the existing ramp is already correct.
    floor = np.zeros_like(base_command)
    lo_vx, hi_vx = ranges[0]
    if lo_vx > 0.0:
        # Ramp toward the target from whichever end of the range it sits nearer, so a command
        # AT the floor does not ramp downward through nothing.
        floor[0] = min(lo_vx, float(base_command[0]))
    return floor, base_command


class StandDown(Exception):
    """Raised to stop the loop and put the robot in a safe state."""


def check_tilt(sample, state) -> None:
    """Upright means gravity points along body -z.

    Uses the ACCELEROMETER path, not the fused roll/pitch one, because the watchdog has to
    be correct precisely where the rpy path is not. protocol.projected_gravity_from_rpy is
    only valid for |roll| < 90 deg -- beyond that the firmware's Euler output switches
    branch and returns a MIRRORED vector, so a robot on its back reads as upright. That is
    exactly the case this check exists to catch, and it is how a fall got past it on the
    first floor run: 400 ticks, no stand-down, robot on its back.

    The accel mapping was validated against all six orientations in `imu --phase tilt`,
    including inverted. It is noisier under acceleration, which is why the policy still
    uses the fused path for its observation -- but for a binary "have we fallen over"
    decision, correct-everywhere beats smooth-but-wrong-when-inverted.
    """
    if sample.accel is None:
        return
    mag = float(np.linalg.norm(sample.accel))
    if not (ACCEL_G_BAND[0] <= mag <= ACCEL_G_BAND[1]):
        return                            # accelerating hard; direction is not "down"
    g = P.projected_gravity_from_accel(sample.accel)
    tilt = math.degrees(math.acos(float(np.clip(-g[2], -1.0, 1.0))))
    state["n"] = state["n"] + 1 if tilt > TILT_WATCHDOG_DEG else 0
    if state["n"] >= TILT_WATCHDOG_TICKS:
        raise StandDown(f"tilt watchdog: {tilt:.1f} deg from upright for "
                        f"{state['n']} consecutive ticks")


def run(port, command, policy_path, duration_s, dry_run, hold, obs_log_path=None,
        startup_hold_ticks=STARTUP_HOLD_TICKS, gait_hz=None,
        command_ramp_s=COMMAND_RAMP_S):
    policy = NumpyMLPPolicy(policy_path)
    default = policy.default_dof_pos
    servo = ob.ServoModel(default, policy.control_dt,
                          tau_s=policy.actuator_tau_s,
                          slew_rad_s=policy.actuator_slew_rad_s)
    sensors = ob.SensorState()
    # Zero-length unless the policy was trained with --obs-history, in which case extend()
    # appends the past frames in the simulator's order. See ObsHistory.
    # Width must follow the policy: a gait-clock policy observes 36 channels, not 33, and
    # the history ring has to match or every past frame is misaligned by three.
    obs_width = ob.NUM_BASE_OBS + (ob.NUM_GAIT_OBS if policy.gait_phase else 0)
    history = ob.ObsHistory(policy.obs_history_len, obs_width)
    gait = None
    if policy.gait_phase:
        if gait_hz is None:
            gait_hz = 0.5 * (policy.gait_hz_range[0] + policy.gait_hz_range[1])
        gait = ob.GaitClock(gait_hz, policy.control_dt)
        print(f"gait clock: {gait_hz:.2f} Hz "
              f"(trained over {policy.gait_hz_range[0]:.2f}-{policy.gait_hz_range[1]:.2f})")

    dof_pos_prev = default.copy()
    last_action = np.zeros(len(policy.joint_names))
    prev_raw_target = default.copy()
    n_stale = 0
    # IMU HEALTH. ang_vel is finite-differenced from attitude (there is no raw gyro in the
    # firmware's output), so its quality depends entirely on sample TIMING -- and the firmware's
    # transform() blocks its main loop for the whole duration of every move, reading no serial
    # and updating no IMU while it does. At 50 Hz we send a move every 20 ms, so if a move
    # blocks longer than that the stream is disrupted continuously and the differenced rate is
    # not noise but structured garbage. Sim models a clean 0.45 rad/s Gaussian on this channel
    # against a measured real floor of 0.19, so it is if anything OVER-noisy -- which means a
    # plain noise model cannot explain hardware commanding 321 deg/s where sim commands 93.
    # These counters are what tell us whether timing is the missing term.
    imu_dt, n_missing, n_invalid = [], 0, 0
    req_pooled = []
    # Every observation the policy was actually handed, for offline comparison against the same
    # channels in sim. The robot commands ~3x the joint rate sim does under identical noise, and
    # timing has been ruled out (clean 19.9 ms intervals), so the discrepancy is in observation
    # CONTENT -- and 33 channels is few enough to just look.
    obs_log = []
    # Seeded so a dropped IMU sample reuses the last known attitude instead of raising.
    # Upright with zero rate is the safe prior: it makes the tilt watchdog pass (we have
    # no evidence of a fall) while telling the policy the robot is level and still. The
    # stale counter below is what actually catches a dead sensor.
    gravity = np.array([0.0, 0.0, -1.0])
    ang_vel = np.zeros(3)
    tick_gaps, n_overrun = [], 0
    tilt_state = {"n": 0}
    clamp_hits, req_deg = 0, []
    prev_dof = default.copy()

    with SerialLink(port=port, dry_run=dry_run, verbose=True) as link:
        # The firmware beeps on EVERY command token it does not exclude (reaction.h:357),
        # and `i` is not excluded -- at 50 Hz that is 50 beeps/second, the constant
        # chirping. `b0` sets buzzer volume to 0. Worth doing for more than quiet: beep()
        # runs inline in the command handler, so muting removes work from every tick.
        link.send(P.MUTE)
        # REALTIME MODE. The firmware boots in PREPROGRAMMED mode, where every `i` is shaped by
        # a blocking raised-cosine ramp and the gP stream is capped at 5 Hz -- and this loop
        # cannot work under either. `XR` switches both. It is sent before anything else that
        # matters, and `Xr` is sent on the way out so the robot is left able to run skills
        # and gaits normally.
        #
        # An unpatched robot has no such token: stock firmware reads `XR` as "activate the
        # hardware module named R" and deactivates the others. _enter_realtime refuses to
        # continue when the acknowledgement does not come back, rather than running a loop
        # whose latency and IMU rate are both silently wrong.
        realtime = _enter_realtime(link, dry_run)
        # The OTHER source of firmware-initiated servo commands that would fight the policy
        # mid-episode. Gyro balancing re-arms itself, hence the periodic re-assert ~1 s below.
        #
        # Voice reactions used to be handled here too, by sending `XAd`. They are not any
        # more, and deliberately: the patched firmware drops recognised words for as long as
        # realtime mode is on (read_voice() in voice.h), which the XR above has already
        # confirmed. `XAd` reached past the firmware's own flag into the voice MODULE's
        # state, and re-arming that needs a language change the module announces OUT LOUD --
        # so every run ended in the robot talking to itself. This loop therefore requires the
        # voice half of the patch, not just XR; build_firmware.sh refuses to build an image
        # missing either.
        link.gyro_balance_off()          # else the firmware fights the policy
        # Straight to the home pose. Deliberately NOT link.wake(), which commands every
        # leg joint to 0 deg and holds it for a second: 0 deg is a fully-extended splay,
        # nothing like the 32 deg stand the policy expects, and holding it can topple the
        # robot before the first control tick. (wake() exists to power the servo driver so
        # `f` feedback works; this loop never reads feedback, so the 0 deg detour is pure
        # risk.) Any move command wakes the servos, so sending the home pose does both jobs.
        start_pose = _read_current_pose(link, policy.joint_names)
        if start_pose is None:
            print("  ! `j` did not parse -- stepping straight to the home pose (expect a jump)")
            _send_pose(link, policy.joint_names, default)
        else:
            _ramp_to_pose(link, policy.joint_names, start_pose, default, policy.control_dt)
        time.sleep(1.5)                  # let it settle into the stance before stepping
        link.send(P.G_PRINT_STREAM)      # ~250 Hz IMU, survives motion commands
        time.sleep(0.2)
        # Gyro z-axis offset, measured while the robot is already standing still in the home
        # pose. Costs 2 s once; without it the policy reads a permanent -0.54 rad/s of
        # phantom yaw and steers to cancel it for the whole run. See SensorState's docstring.
        sensors.yaw_rate_bias = _calibrate_yaw_bias(link, seconds=2.0)
        # The observation history is left ZERO-FILLED, which is what mj_vec_env's _reset_idx
        # does and therefore what the policy trained against. This used to call a prime()
        # that seeded slot 0 with a synthetic settled-pose frame; measured on a checkpoint,
        # sim's first post-reset observation has 0 of 33 channels set in every slot, so
        # priming fed the policy a frame it had never seen. See ObsHistory's docstring.
        # The firmware RE-ENABLES gyro balancing on its own: reaction.h:370 restores it as
        # "the default state" after a rest token, and a 'G' echo was observed after our
        # move commands even though `gb` had already been sent and acknowledged with 'g'.
        # With it on, the firmware runs its own recovery skills -- on the first floor run it
        # queued `k rc` and self-righted the robot mid-episode, fighting the policy for the
        # servos. Rather than chase which code path re-arms it, just re-assert `gb`
        # periodically: one token, negligible against 19.7 ms of per-tick headroom.
        gb_every = max(int(1.0 / policy.control_dt), 1)      # ~1 s

        base_command = np.asarray(command, dtype=np.float64).copy()
        ramp_ticks = max(int(command_ramp_s / policy.control_dt), 1)
        ramp_floor, base_command = _ramp_floor_and_command(policy, base_command)
        t_start = next_due = time.perf_counter()
        t_end = None
        last_sent = None
        try:
            n_tick = 0
            while time.perf_counter() - t_start < duration_s:
                tick = time.perf_counter()
                if n_tick % gb_every == 0:
                    link.send(P.G_BALANCE_OFF)
                n_tick += 1

                sample = _newest_imu(link)
                if sample is None:
                    n_stale += 1
                    n_missing += 1
                    if n_stale > 10:
                        raise StandDown("no IMU data for 10 consecutive ticks")
                else:
                    n_stale = 0
                    prev_stamp = sensors._prev[0] if sensors._prev is not None else None
                    gravity, ang_vel, _ok = sensors.update(
                        np.radians(sample.ypr_deg), tick)
                    if prev_stamp is not None:
                        imu_dt.append(tick - prev_stamp)
                    if not _ok:
                        n_invalid += 1
                    check_tilt(sample, tilt_state)

                if hold:
                    target = default.copy()
                else:
                    # Ease the command in rather than stepping it. Raised cosine, so the
                    # command's own derivative starts at zero as well.
                    if ramp_ticks > 1 and n_tick < ramp_ticks:
                        frac = 0.5 * (1.0 - math.cos(math.pi * n_tick / ramp_ticks))
                        # Ramp from the FLOOR of the trained range, not from zero. A
                        # command-responsive policy has never seen a command below its
                        # training range and does not degrade gracefully there: a policy
                        # trained over 0.05-0.15 reads 1.43x excess joint reversals at
                        # command 0.00 against 0.47x at 0.02. Only the startup hold keeps
                        # that off the robot today, and only by coincidence -- the hold is
                        # 15 ticks and the ramp 30, so it happens to cover exactly the
                        # half of the ramp below 0.5 * target. Command anything closer to
                        # the range floor, or pass --startup-hold 0, and the alignment
                        # stops holding.
                        command = ramp_floor + (base_command - ramp_floor) * frac
                    else:
                        command = base_command
                    obs = ob.build_observation(dof_pos_prev, prev_dof, last_action,
                                               command, ang_vel, gravity, policy, gait)
                    obs_log.append(obs.copy())
                    action = policy.act(history.extend(obs))
                    # One-step delay, matching training.
                    target = last_action * policy.action_scale + default
                    last_action = action

                # Send the RAW target; ServoModel runs as the OBSERVER only.
                #
                # WHAT THE WIRE IS. In sim, ctrl is the OUTPUT of _apply_actuator_dynamics --
                # a model of what the REAL SERVO does to its input. The wire carries that
                # INPUT. Sending the filter's output instead makes the physical servo apply
                # its own tau/slew on top of the host's copy of them, and the composed step
                # response settles in ~67 ms where the bare servo settles in ~40-45 ms.
                # Hard-clamping the slew host-side is worse still: truncating a fast
                # oscillation yields a reduced-amplitude triangle wave, i.e. a shorter stride
                # with the legs lagging, marching in place.
                #
                # servo.update() still runs every tick regardless, because its state IS the
                # observation (dof_pos/dof_vel), exactly like sim's _update_servo_estimate --
                # and, like sim's, it deliberately integrates the UNQUANTISED target while the
                # wire rounds to integer degrees (mj_vec_env's wire_quantize_deg models that
                # plant-side rounding).
                if n_tick <= startup_hold_ticks:
                    # Ride out the opening transient on the home pose.
                    #
                    # last_action IS ZEROED TOO, and that is the whole point. The previous
                    # version let the policy's own output flow into last_action while the
                    # joints were pinned to the home pose, so the observation said "I just
                    # commanded a large move" (last_action, 8 channels) and simultaneously
                    # "nothing moved" (dof_pos/dof_vel, 16 channels). No training state looks
                    # like that -- mj_vec_env._reset_idx zeroes actions AND last_actions
                    # alongside the pose.
                    #
                    # Measured on hardware: a ~0.5 s shake starting at t=0.36 s,
                    # one tick after a 15-tick hold released, peaking at 2047 deg/s against a
                    # 217 deg/s steady-state maximum. LENGTHENING the hold to 40 ticks made it
                    # worse, which is the tell -- more ticks of inconsistency, not more ticks
                    # of settling.
                    target = default.copy()
                    last_action = np.zeros_like(last_action)
                raw_target = target
                before = servo.pos.copy()
                servo.update(target)
                # The slew term bound iff the unclamped first-order step would have exceeded
                # it -- the ticks where the real servo is presumably rate-saturated too.
                clamped_this_tick = bool(
                    np.any(np.abs(servo._alpha * (raw_target - before))
                           > servo.slew * servo.dt + 1e-12))
                _send_pose(link, policy.joint_names, raw_target)
                last_sent = raw_target
                if gait is not None:
                    gait.tick()      # once per control tick, exactly as the env advances it

                # Two DIFFERENT quantities, previously conflated into one misleading number.
                #
                # req_deg is the policy's own rate of change: how fast it is moving its
                # commanded target. Compare it against prev_RAW_target, not against the servo
                # position -- the earlier version differenced the raw command against the
                # LAGGED one, which measures the servo's tracking error expressed as a rate.
                # That reported "85% of ticks clamped, median 224 deg/s" on a run where the
                # limit actually truncated 32% of ticks; sim reproduces 243 deg/s under the
                # same wrong definition, so it was never a sim-to-real gap.
                # BOTH forms, because reporting only one cost a long wrong investigation.
                # This line used to publish the MAX over the eight joints, which was then
                # compared against a sim number computed as the POOLED per-joint median --
                # 339 vs 93 deg/s, an apparent 3.5x sim-to-real gap that sent us through the
                # firmware timing, the observer offset and the estimator before the statistic
                # itself turned out to be the discrepancy. In the pooled form the same data
                # reads 135 against sim's 120.
                delta = np.abs(raw_target - prev_raw_target)
                raw_deg_s = float(np.degrees(delta.max()) / policy.control_dt)
                req_deg.append(raw_deg_s)
                req_pooled.extend(np.degrees(delta) / policy.control_dt)
                # ...and this is whether the rate limit actually BOUND this tick, which is the
                # thing worth knowing: it is the only case where the servo receives something
                # different from what the policy asked for.
                if clamped_this_tick:
                    clamp_hits += 1
                prev_raw_target = raw_target.copy()

                dof_pos_prev = prev_dof
                prev_dof = servo.pos.copy()

                # ABSOLUTE-deadline pacing. Relative pacing (sleep(dt - spent)) accumulates
                # sleep's overshoot: time.sleep() on macOS routinely returns late by a few ms,
                # and with only ~0.3 ms of compute per tick that overshoot IS the period.
                # Measured: relative pacing ran 413 ticks in 10 s = 41.3 Hz against a 50 Hz
                # target -- an 18% slow gait -- while reporting 0% overruns, because compute
                # time was never the problem. cmdrate always paced this way; the control loop
                # did not, so its own measurement of the achievable rate did not transfer.
                spent = time.perf_counter() - tick
                tick_gaps.append(spent * 1000.0)
                next_due += policy.control_dt
                slack = next_due - time.perf_counter()
                if slack > 0:
                    time.sleep(slack)
                else:
                    n_overrun += 1
                    next_due = time.perf_counter()   # do not chase a missed deadline

            # Stop the clock BEFORE the teardown ramp. `achieved` divides tick count by
            # wall time since t_start, and the ramp below plus its settle add ~0.8 s of
            # non-loop time -- enough to report 47 Hz for a loop that held 50 with zero
            # overruns, which trips the runbook's own off-rate gate on a healthy run.
            t_end = time.perf_counter()
            # Normal completion only: ramp back to the stance before resting. Without this the
            # run ends wherever the last policy action left the legs -- mid-gait, one leg up --
            # and `d` shuts the servos from there (reaction.h T_REST -> shutServos), so the
            # robot is abandoned in a pose it never chose. Deliberately NOT in `finally`: on a
            # stand-down the robot has fallen or gone out of distribution, and the right move
            # then is to rest immediately, not to drive it through a stance it cannot hold.
            if last_sent is not None:
                _ramp_to_pose(link, policy.joint_names, last_sent, default, policy.control_dt)
                time.sleep(0.3)
        except StandDown as e:
            print(f"\nSTAND-DOWN: {e}")
        except KeyboardInterrupt:
            print("\ninterrupted")
        finally:
            if tick_gaps:
                g = sorted(tick_gaps)
                achieved = len(g) / max((t_end or time.perf_counter()) - t_start, 1e-9)
                print(f"\nachieved {achieved:.1f} Hz "
                      f"(target {1.0/policy.control_dt:.0f} Hz)")
                if abs(achieved - 1.0/policy.control_dt) > 2.0:
                    print("  ! off-rate: dof_vel, the actuator-lag model and gait frequency "
                          "all assume the target rate")
                print(f"loop: {len(g)} ticks, compute per tick | "
                      f"median {g[len(g)//2]:.1f} | p95 {g[int(len(g)*0.95)]:.1f} | "
                      f"max {g[-1]:.1f} ms | overruns {n_overrun} "
                      f"({100.0*n_overrun/len(g):.1f}%)")
                if n_overrun > 0.05 * len(g):
                    print("  ! deadline missed often -- treat the gait as unrepresentative")
            if obs_log_path and obs_log:
                np.savetxt(obs_log_path, np.array(obs_log), delimiter=",",
                           header=",".join(ob.OBS_CHANNEL_NAMES), comments="")
                print(f"obs: wrote {len(obs_log)} frames to {obs_log_path}")
            if imu_dt:
                d = sorted(imu_dt)
                nom = 1.0 / 50.0
                late = sum(1 for x in d if x > 1.5 * nom)
                print(f"imu: {len(d)} samples | interval median {1000*d[len(d)//2]:.1f} ms, "
                      f"p95 {1000*d[int(len(d)*0.95)]:.1f}, max {1000*d[-1]:.1f} "
                      f"(control tick is {1000*nom:.0f} ms)")
                print(f"     {late} intervals ({100.0*late/len(d):.0f}%) exceeded 1.5 ticks | "
                      f"{n_missing} ticks with no sample | {n_invalid} rejected as stale")
                if late > 0.15 * len(d):
                    print("     ! ang_vel is finite-differenced from attitude, so irregular")
                    print("       sampling corrupts it directly -- this is the prime suspect")
                    print("       for hardware shaking more than sim. See firmware transform().")
            if req_deg:
                r = sorted(req_deg)
                lim = math.degrees(servo.slew)
                if req_pooled:
                    q = sorted(req_pooled)
                    print(f"     pooled per-joint: median {q[len(q)//2]:.0f}, "
                          f"p95 {q[int(len(q)*0.95)]:.0f} deg/s | "
                          f"{100.0*sum(1 for x in q if x > lim)/len(q):.0f}% over the limit "
                          f"<- compare sim in THIS form, not the max-over-joints one above")
                print(f"slew: rate limit BOUND on {clamp_hits}/{len(r)} ticks "
                      f"({100.0*clamp_hits/len(r):.0f}%) | policy's own rate of change: median "
                      f"{r[len(r)//2]:.0f}, p95 {r[int(len(r)*0.95)]:.0f}, max {r[-1]:.0f} "
                      f"deg/s | limit {lim:.0f} deg/s")
            link.send(P.G_PRINT_ONCE)    # stop the IMU stream
            if realtime:
                # Acknowledged, not fire-and-forget. This fires immediately after `gp` stops
                # a 250 Hz stream -- precisely the case link.realtime() documents as losing
                # the token ("a token sent while the firmware is mid-print is dropped
                # outright"). A dropped `Xr` leaves the robot in realtime mode, where `i` is
                # no longer interpolated and every voice command is discarded, and no
                # battery cycle clears it because the USB-powered board never restarts.
                if link.realtime(False) is not False:
                    print("  ! could NOT confirm the robot left realtime mode. Send `Xr` by "
                          "hand: until then it ignores voice commands and moves in steps.")
            # SerialLink.__exit__ restores voice reactions (`XAc`) and sends `gb` then
            # `d` (rest), even on exception.


def _enter_realtime(link, dry_run):
    """Switch the firmware into realtime mode and CONFIRM it, or refuse to run.

    Both halves of this loop's contract with the plant depend on the mode: `i` must not be
    interpolated (the host is the trajectory) and the gP stream must not be capped at 5 Hz
    (obs_builder rejects samples older than 50 ms, so a capped stream sends ang_vel to
    zeros without saying so). Neither failure is visible in the loop's own output -- one
    reads as sluggish walking, the other as a policy that ignores its IMU -- which is why
    the acknowledgement is checked rather than assumed.
    """
    if dry_run:
        print("  realtime mode: skipped (--dry-run)")
        return False
    # link.realtime() rather than a query() here: it flushes first and waits for ITS OWN
    # acknowledgement. A query() returns the first line available, which is whatever the
    # previous command left in the buffer -- and the previous command is the buzzer mute
    # just above, whose reply is "Changing volume to 0/10".
    ack = link.realtime(True)
    if ack is not True:
        raise SystemExit(
            "Realtime mode was not confirmed, so this loop will not run.\n"
            + ("The robot answered, but reported the mode OFF. Retry; if it persists the\n"
               "firmware is accepting XR without acting on it.\n" if ack is False else
               "No acknowledgement came back. Most likely this is unpatched firmware,\n"
               "which has no XR token and read it as a module code. Check with:\n"
               "  uv run --directory deploy python -m bittle_io --port PORT "
               "--run-id check probe\n"
               "and reflash from the pinned firmware submodule if it says NOT\n"
               "SUPPORTED -- see docs/firmware.md.\n")
        )
    print("  realtime mode: ON (firmware ramp off, gP at the link ceiling)")
    return True


def _send_pose(link, joint_names, angles_rad):
    pairs = [deg_to_petoi(n, a) for n, a in zip(joint_names, angles_rad)]
    link.send(P.move_cmd(pairs))


def _read_current_pose(link, joint_names):
    """Current joint angles in the policy frame (rad), or None if the reply does not parse.

    `j` returns the firmware's `currentAng`, which is its COMMANDED belief and not a
    measurement (see protocol.py). That is the right quantity here: it is the target the
    servos are being held at, so it is the true starting point of the next move. It also
    costs one round trip, once, before the control loop starts.

    Returns None rather than guessing. A ramp from a wrong start pose is worse than no ramp:
    it would begin with exactly the jump it exists to prevent.
    """
    try:
        table = P.parse_joint_table("\n".join(link.query(P.T_JOINTS, timeout=1.5, min_lines=2)))
    except Exception:
        return None
    out = []
    for name in joint_names:
        cal = JOINT_CALIBRATION[name]
        if cal.petoi_index is None or cal.sign is None or cal.offset_deg is None:
            return None
        out.append(math.radians((float(table[cal.petoi_index]) - cal.offset_deg) / cal.sign))
    return np.array(out, dtype=float)


def _ramp_to_pose(link, joint_names, start_rad, goal_rad, dt, deg_per_s=RAMP_DEG_PER_S):
    """Stream a raised-cosine interpolation from start to goal at the control rate.

    Duration is set from the LARGEST joint move so peak velocity is bounded whatever the
    start pose is: for s(t) = (1 - cos(pi t/T))/2 the peak rate is pi/(2T) times the travel,
    so T = pi * travel / (2 * deg_per_s).
    """
    start_rad = np.asarray(start_rad, dtype=float)
    delta = np.asarray(goal_rad, dtype=float) - start_rad
    travel_deg = math.degrees(float(np.abs(delta).max())) if delta.size else 0.0
    if travel_deg < 0.5:                       # already there; a ramp would be noise
        _send_pose(link, joint_names, goal_rad)
        return
    seconds = min(RAMP_MAX_S, math.pi * travel_deg / (2.0 * deg_per_s))
    n = max(int(seconds / dt), 2)
    for k in range(1, n + 1):
        frac = 0.5 * (1.0 - math.cos(math.pi * k / n))
        _send_pose(link, joint_names, start_rad + frac * delta)
        link.sleep(dt)                          # virtual under --dry-run


def _calibrate_yaw_bias(link, seconds=2.0):
    """Collect a stationary window and hand it to obs_builder.calibrate_yaw_rate_bias.

    Called after the home pose has settled, so the robot is standing still by construction.
    Aborts (returning 0.0, i.e. no correction) if roll or pitch moved meaningfully during
    the window -- that means someone was holding the robot, and a bias measured then would
    bake their hand movement into every observation for the rest of the run.
    """
    t0 = time.perf_counter()
    pts, tilts = [], []
    while time.perf_counter() - t0 < seconds:
        s = _newest_imu(link)
        if s is not None:
            yaw, pitch, roll = np.radians(s.ypr_deg)
            pts.append((time.perf_counter(), yaw))
            tilts.append((pitch, roll))
        time.sleep(0.004)
    if len(tilts) < 100:
        print("yaw-bias calibration: too few IMU samples, skipping (no correction applied)")
        return 0.0
    tilts = np.array(tilts)
    excursion = float(np.degrees(np.ptp(tilts, axis=0)).max())
    # 2.0 was far too tight: a robot standing on its own legs, servos holding, shows 5-6 deg
    # of tilt range from servo jitter and settling -- both real hardware runs skipped
    # calibration for that reason. The gate exists to catch a HAND-HELD robot, which moves far
    # more than this, so it can be loose without losing its purpose.
    if excursion > 12.0:
        print(f"yaw-bias calibration: robot moved ({excursion:.1f} deg of tilt) -- skipping. "
              f"Stand it on a flat surface, hands off, and rerun.")
        return 0.0
    bias = ob.calibrate_yaw_rate_bias(pts)
    print(f"yaw-bias calibration: {bias:+.3f} rad/s ({math.degrees(bias):+.1f} deg/s) "
          f"over {seconds:.0f} s, {len(pts)} samples -- subtracting from ang_vel[2]")
    return bias


def _newest_imu(link):
    """Drain the stream and return only the LATEST sample.

    Taking the newest rather than the oldest matters: the IMU produces ~5 samples per
    control tick, and acting on the stalest one would add up to 20 ms of avoidable delay
    on top of the actuator lag we already model.
    """
    latest = None
    link._pump()
    while link._lines:
        text, _ts = link._lines.pop(0)
        if P.looks_like_imu(text):
            try:
                latest = P.parse_imu(text)
            except P.ProtocolError:
                pass
    return latest


def _resolve_policy(path):
    """Make --policy mean the same thing from the repo root and from deploy/.

    Policies live in deploy/policy/, but this script is legitimately launched both as
    `python control_loop.py` (from deploy/, and on the Pi) and as
    `python deploy/control_loop.py` (from the repo root, alongside sim/ and tools/).
    A path relative to the caller's cwd wins if it exists; otherwise it is resolved
    against this file's own directory, so `--policy policy/x.npz` works either way.
    """
    given = os.path.expanduser(path)
    if os.path.exists(given):
        return given
    here = os.path.join(os.path.dirname(os.path.abspath(__file__)), path)
    if os.path.exists(here):
        return here
    raise SystemExit(
        f"policy not found: {path!r}\n"
        f"  looked in cwd:      {os.path.abspath(given)}\n"
        f"  and next to script: {here}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--port", default=None)
    p.add_argument("--policy", required=True, metavar="PATH",
                   help="Exported .npz from sim/export_policy_mj.py. Required, and "
                        "deliberately has no default: it carries the servo tau/slew, the "
                        "home pose and the observation width this loop runs with, so a "
                        "stale one silently drives the robot with the wrong plant. "
                        "Needed even with --hold, which reads the home pose from it. "
                        "Relative paths resolve against deploy/ if not found in cwd.")
    p.add_argument("--command", type=float, nargs=3, default=[0.15, 0.0, 0.0],
                   metavar=("VX", "VY", "WZ"))
    p.add_argument("--duration", type=float, default=10.0)
    p.add_argument("--dry-run", action="store_true", help="no serial writes")
    p.add_argument("--log-obs", default=None, metavar="PATH",
                   help="Write every observation handed to the policy as CSV, for diffing "
                        "against the same channels in sim. This is how an observation-content "
                        "mismatch gets localised instead of guessed at.")
    p.add_argument("--startup-hold", type=int, default=STARTUP_HOLD_TICKS, metavar="TICKS",
                   help=f"Command the home pose for the first N ticks while the policy runs "
                        f"(default {STARTUP_HOLD_TICKS}, ~{STARTUP_HOLD_TICKS*20} ms). Rides out "
                        f"the policy's opening transient, which reaches the action clip out of "
                        f"a settled stance. 0 disables.")
    p.add_argument("--hold", action="store_true",
                   help="hold the home pose; ignore policy output (safe first test)")
    p.add_argument("--command-ramp", type=float, default=COMMAND_RAMP_S, metavar="S",
                   help=f"Ease the commanded velocity in over this many seconds (default "
                        f"{COMMAND_RAMP_S}). 0 steps straight to the target.")
    p.add_argument("--gait-hz", type=float, default=None, metavar="HZ",
                   help="Cadence for a gait-clock policy [Hz]; ignored otherwise. Defaults to "
                        "the middle of the range the policy trained over. The upper bound is "
                        "the SERVO, not the control loop: at ~34.5 deg peak-to-peak the gait "
                        "fundamental alone reaches the 137 deg/s slew limit at 1.26 Hz.")
    a = p.parse_args()
    run(a.port, np.array(a.command), _resolve_policy(a.policy), a.duration, a.dry_run,
        a.hold, a.log_obs,
        startup_hold_ticks=a.startup_hold, gait_hz=a.gait_hz,
        command_ramp_s=a.command_ramp)


if __name__ == "__main__":
    main()
