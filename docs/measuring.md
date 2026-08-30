# Measuring your own robot

Every physical constant in `sim/` came from the robot rather than a datasheet, and yours will
differ. `bittle_io` is the instrument. Everything here runs from the repository root.

You do not need this file to run the shipped policy, or to train your own against the
measurements already in the repo. Read it when you want the simulator fitted to *your*
hardware.

**Safety, every time:** robot suspended with legs free (except `imu --phase dynamics`,
`bodyid` and `friction`, which need it on the ground), battery charged, short runs. Touch the
servo cases between runs — servos overheat when they are fighting a target.

```bash
export PORT=/dev/ttyUSB0     # or /dev/cu.usbmodem* on macOS
R="uv run --directory deploy python -m bittle_io --port $PORT --run-id my_session"

$R probe                        # firmware, IMU chip, stream rates -- run FIRST
$R jointmap                     # which index is which leg, and which sign
$R step --all-joints            # servo tau, f_3db, slew rate
$R sweep                        # per-joint slew, all leg joints
$R cmdrate --hz 50              # does the link hold 50 Hz?
$R latency --joint 8            # command -> motion dead time
$R imu --phase static           # noise floor
$R imu --phase tilt             # attitude-fusion lag (operator-guided)
$R bodyid                       # ON THE GROUND: frequency response
$R friction                     # inclined-plane test, foot mu

uv run --directory deploy python -m bittle_io report measurements/my_session
```

`--directory deploy` runs the command *as if* from `deploy/`, so it picks up that project's
numpy+pyserial environment — the same one the Pi has. Runs are written to
`measurements/<run-id>/` regardless of where you launched from. On the Pi you are already in
`deploy/` and drop the prefix.

`--dry-run` exercises any subcommand with no robot attached (virtual clock, token sequence
printed at exit), which is the cheapest way to see what an experiment does.

A reference capture from a patched robot is in [`measurements/reference/`](../measurements/reference/) —
compare against it before trusting a new number. Read
[`measurements/reference/CAVEATS.md`](../measurements/reference/CAVEATS.md) first: some recorded
values there are invalid and are kept anyway, because deleting a measurement hides that it
was attempted.

## Feeding the results back

| measured | lands in |
|---|---|
| servo τ and slew | `ACTUATOR_TAU_S_RANGE` / `ACTUATOR_SLEW_*` in `sim/mj_vec_env.py` |
| command→motion latency | `ACTION_LATENCY_STEPS_RANGE` in `sim/mj_vec_env.py` |
| IMU noise | `OBS_NOISE_*` in `sim/mj_vec_env.py` |
| foot friction | `FOOT_FRICTION_RANGE` in `sim/config.py` |
| joint index / sign / offset | `deploy/joint_calibration.py` |
| MJCF gains fitted to the step response | `model/bittle.xml` (`kv`, `armature`) |

The full instrument documentation — what each experiment does, how it gets past the ~2 Hz
serial bandwidth limit, and the firmware facts it was built against — is in
[`deploy/bittle_io/README.md`](../deploy/bittle_io/README.md). New to servos, serial links,
IMUs or system ID? [`deploy/bittle_io/READING.md`](../deploy/bittle_io/READING.md) is a short
curated list tied to the specific problems this package hits.

## Four ways a hardware measurement lies

Each of these produced a confident, plausible, wrong number here.

1. **A stale input buffer.** An experiment that stops reading the moment it detects an event
   leaves the rest of the stream queued, and the next trial computes its "quiet baseline"
   from the *previous* trial's samples. The bias is consistent, which is exactly what makes
   it look trustworthy. `Link.flush_input()` before every baseline window.
2. **A bias read as noise.** Pooling axes with `abs()` before `std()` lets one axis with a
   constant offset contaminate the others, and the offset comes back as a noise floor.
   Subtract bias first; the control loop does this at startup for yaw.
3. **Two channels that are not measuring the same thing.** Estimating fusion lag by swinging
   the robot's own legs fails, because a swinging leg drives the accelerometer with tangential
   acceleration rather than tilt. Drive it by hand. And do not use cross-correlation — on a
   band-limited signal its ridge is flat, and on synthetic data with a known 20 ms lag it
   reads 12.5 ms.
4. **Measuring while still when the robot will be walking.** Walking vibration is roughly
   double the static noise floor, so a static-only reading over-corrects.

The general form: **a gate that never fires is evidence.** If a rejection threshold never
trips while the data still looks wrong, the fault is upstream of what you are gating.
