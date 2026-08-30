# bittle_io — Bittle X hardware characterization + I/O layer

Talks to the Petoi Bittle X (BiBoard/ESP32, OpenCatEsp32 firmware) over USB serial, runs
the measurements that sim-to-real training needs, and provides the runtime sensor layer
the deployed control loop imports. The point: **every physical constant in the MuJoCo
training setup should be a measured number with a CSV behind it, not a guess.**

```
uv run python -m bittle_io --port /dev/cu.usbmodem5AA90270111 probe
```

## Why this exists

Sim-to-real for this robot is blocked on numbers nobody had measured:

| Placeholder | Where | Filled by |
|---|---|---|
| `parse_joint_angles` / `parse_gyro_accel` stubs | `deploy/obs_builder.py` | `bittle_io.state.StateReader` + `protocol` parsers |
| `JOINT_CALIBRATION` (all empty) | `deploy/joint_calibration.py` | `jointmap` → `joint_map.csv` |
| `MAX_ANGLE_DELTA_PER_STEP_RAD` placeholder | `deploy/control_loop.py` | `sweep` → slowest-joint slew |
| obs-noise magnitudes | `mj_vec_env.py` | `stream` + `imu --phase static` |
| actuator gain/latency model | `bittle.xml`, `mj_vec_env.py` | `step` + `bodyid` (+ `sweep`, `latency`) |
| `CONTROL_DT = 0.02` sanity | `policy_io.py` | `cmdrate --hz 50` |

## Workflow for a characterization session

Safety first, every time: **robot suspended with legs free** (except `imu --phase
dynamics`, which needs it on the ground), battery on, short runs, touch the servo cases
between runs — servos overheat when fighting.

```bash
# 1. Discover what this board actually does. Read capabilities.json before going on.
uv run python -m bittle_io --port $PORT probe

# 2. Sensors first (no motion), then actuation experiments:
uv run python -m bittle_io --port $PORT --run-id my_session stream
uv run python -m bittle_io --port $PORT --run-id my_session imu --phase static
uv run python -m bittle_io --port $PORT --run-id my_session imu --phase tilt     # operator-guided
uv run python -m bittle_io --port $PORT --run-id my_session jointmap
uv run python -m bittle_io --port $PORT --run-id my_session cmdrate --hz 50
uv run python -m bittle_io --port $PORT --run-id my_session latency --joint 8
uv run python -m bittle_io --port $PORT --run-id my_session step --all-joints    # tau, f_3db, slew
uv run python -m bittle_io --port $PORT --run-id my_session sweep                # all leg joints
uv run python -m bittle_io --port $PORT --run-id my_session imu --phase dynamics # ON THE GROUND
uv run python -m bittle_io --port $PORT --run-id my_session bodyid               # ON THE GROUND
uv run python -m bittle_io --port $PORT --run-id my_session imu --phase angvel-fallback

# 3. Aggregate into hardware_params.json + a checklist of what each number unblocks:
uv run python -m bittle_io report ../../measurements/my_session
```

Or `... all` to run everything with defaults after a probe. `--dry-run` exercises any
subcommand with no robot attached (virtual clock, token sequence printed at exit).

Every run writes to `measurements/<run-id>/` at the repo root: parsed CSVs, `capabilities.json`,
`manifest.json`, and **raw serial transcripts** — when the board surprises the parsers,
the bytes that caused it are on disk.

## Feeding the numbers back into MuJoCo training

From `hardware_params.json`:

- **`imu.pitch_stdev_deg` / `roll_stdev_deg`, `feedback.column_noise_deg`** → the
  observation-noise magnitudes in `mj_vec_env.py`'s
  `_add_obs_noise`. Train with the noise you actually measured.
- **`imu.pitch_bias_deg` / `roll_bias_deg`** → if a level robot reports non-zero
  pitch/roll, that is a mounting offset. Subtract it in `StateReader`, or the policy
  sees a permanently tilted world.
- **`control.action_latency_ms`** → compare with the one-step (20 ms) delay
  `mj_vec_env.py` already simulates (`simulate_action_latency`). If measured latency is
  ~2 steps, simulate 2.
- **`control.slowest_joint_slew_deg_per_s`** → `control_loop.py`'s
  `MAX_ANGLE_DELTA_PER_STEP_RAD` (slew × 0.02), and a velocity clamp worth mirroring in
  the sim actuator model in `bittle.xml`.
- **`control.actuator_tau_ms` / `actuator_f3db_hz`** (from `step`) → model the sim
  servo as dead time + first-order lag with this tau; randomize tau over ~[0.5×, 2×] in
  `mj_vec_env.py`. This is the primary actuator-lag number.
- **`control.bodyid_knee_hz`** (from `bodyid`) → independent cross-check of the same
  rolloff, measured through the IMU with the servo continuously driven — the response
  the walking policy actually experiences. `step` and `bodyid` should agree via
  f_3db = 1/(2π·tau); if they disagree badly, believe `bodyid` (the step measurement
  time-slices the servo, see caveats in `experiments/step.py`).
- **`sweep_j*.csv`** → low-frequency gain/deadband detail per joint; use with the above
  to set actuator `kp`/damping in `bittle.xml` so the sim servo lags like the real one.
- **`control.overrun_pct` at 50 Hz** → confirms `policy_io.CONTROL_DT = 0.02` is
  achievable. If it is not, that constant is wrong and every trained policy inherits it.
- **`joint_map.csv`** → fill `deploy/joint_calibration.py`'s `JOINT_CALIBRATION`
  (index, sign, offset). The `urdf_joint` column is completed **by hand** against
  `policy_io.JOINT_NAMES` — sign conventions are a human's job to confirm.
- **`imu --phase tilt` PASS** → validates `projected_gravity` sign/axis against
  `mj_vec_env._inv_rotate`. A mirrored gravity vector trains fine in sim and falls over
  on the real floor; do not skip this one.

## What we measured on this robot (firmware `Bittle X B10_251121`)

All confirmed against OpenCatEsp32 source in the `firmware/` submodule
(`src/reaction.h`, `src/espServo.h`, `src/imu.h`). Note that PetoiCamp's *other* repo,
OpenCat, is the **AVR/NyBoard firmware and does not match this board's protocol** — do not
read tokens out of it.

| Fact | Value | Consequence |
|---|---|---|
| IMU | ICM42670, accel enabled, **~249 Hz** via `gP` | 5× the 50 Hz control rate; finite-differenced `ang_vel` is plausible |
| IMU line format | `ICM:%6.2f%6.2f%6.2f%7.1f%7.1f%7.1f` (accel xyz, yaw, pitch, roll), **no separators** | fields can run together; parser matches decimal counts, never `.split()` |
| Raw gyro rates | **never printed** (`gy` exists in imu.h but is not emitted) | `ang_vel` is differentiated + flagged, never fabricated |
| `f` stream idle | **40.0 Hz** single joint, **9.25 Hz** all joints (5.5 Hz on stock firmware) | one joint is 4.3x a whole-body frame |
| `f` stream + any other token | **stops** (reaction.h resets `measureServoPin`) | sweep/jointmap/latency poll per sample instead of streaming |
| Feedback read mechanics | attach → 15 ms delay → query pulse → **detach, pin=INPUT** → time 3 pulses | reading and driving are mutually exclusive at the pin |
| Command→request gap | **≥20 ms** or the request is dropped (0/4 at 0 ms, 4/4 at 20 ms) | `link.COMMAND_GAP` |
| Polled rate (move+read) | **~9 Hz** → usable sweep band ≤ ~2 Hz | the real Nyquist limit for system ID |
| All-joint frame | width varies (8/9): a failed read prints **nothing**, columns **shift** | short frames are dropped; `jointmap` uses index-tagged single-joint polls only |
| Single-joint reply | `"<index>\t<angle>"` — value in column 1 | column 0 is the constant index; reading it looks like a dead servo |
| `f` mode side effects | sets servo gain soft (`P_SOFT`), clears `gyroBalanceQ`; move tokens restore `P_WORKING` | sweep alternates working-gain driving with detached read windows |
| `j` reply | 2 lines (`=` + index header, then values); returns `currentAng` **from memory** | commanded, not measured — only `f` is a sensor |
| `gb` | deterministic gyro-balance off | preamble of every actuation experiment (balance silently adds its own corrections) |
| `c` token | overwrites servo calibration offsets | refused by `link.send()` unless explicitly confirmed |

## Package layout

```
bittle_io/
  protocol.py   OpenCatEsp32 tokens + parsers (pure functions; fixed-format aware)
  link.py       SerialLink: framing, streams, polled feedback, COMMAND_GAP, safety guards
  probe.py      capability discovery -> capabilities.json (run FIRST)
  state.py      StateReader: joints + IMU -> policy-obs quantities (what deploy imports)
  record.py     run dirs, manifests, CSVs, voltage bracketing
  report.py     run dir -> hardware_params.json + unblock checklist
  cli.py        python -m bittle_io <probe|stream|imu|jointmap|cmdrate|latency|step|sweep|bodyid|all|report|ports>
  experiments/  one module per measurement, registered in experiments/__init__.py
```

Design rules that earned their place:

- **Nothing hardcodes a reply format the probe can discover.** The vendored firmware tree
  was the wrong board; even the right repo has compile-time flags only the running board
  can answer.
- **Teardown always runs**: `SerialLink.__exit__` sends `gb` + `d` (rest) even on
  exception — servos left fighting a target overheat.
- **`ang_vel` is honest**: returned with `ang_vel_valid`, stale-flagged when the IMU lags
  the control loop, never silently zero-filled.
- **Battery voltage is bracketed** around every experiment; a sagging battery otherwise
  shows up as fake drift in servo measurements.

## Tests

```bash
uv run python -m pytest tests/test_bittle_io_protocol.py tests/test_bittle_io_link.py -q
```

No hardware needed. Fixtures are built with the firmware's own printf formats (including
the run-together-fields case `.split()` gets wrong), and `projected_gravity_from_rpy` is
property-tested against `mj_vec_env._inv_rotate` — the sim definition the policy trains
on — so a mirrored gravity convention fails in `pytest`, not on the floor.

## Getting past the ~2 Hz serial bandwidth limit

The polled feedback path samples at ~9 Hz, capping a naive chirp (`sweep`) at ~2 Hz of
usable band. Two experiments work around it — both software-only:

- **`step`** — equivalent-time sampling: repeat identical steps with the poll phase
  jittered, fold all trials onto one time axis. Millisecond-scale effective resolution
  from 9 Hz polling, because the step timing is under host control. Yields dead time,
  tau (→ f_3db), and slew. Runs suspended.
- **`bodyid`** — uses the ~250 Hz IMU as the sensor instead of the serial feedback:
  excite one joint sinusoidally at 0.5–8 Hz on the ground, lock-in on body pitch/roll.
  Measures the command→body response with the servo continuously driven (no detach
  windows), which is exactly what the policy experiences. Checked against firmware
  source: motion tokens do NOT stop the `gP` gyro stream (unlike `f`).

Remaining true limits: `step` still observes nothing before ~40 ms after the step (poll
floor — the fit extrapolates dead time below that), and `bodyid` measures actuator ⊗ body
dynamics, so use its knee frequency and phase trend rather than absolute gains.

## Known limitations / open items
- **`angvel-fallback` decides the firmware question**: if differentiated attitude is too
  noisy at 50 Hz, the fix is an OpenCatEsp32 build that prints `gy` (raw gyro — already
  populated in `imu.h`, just never printed).
- `imu --phase tilt` and `dynamics` are operator-guided and still to be run on this
  robot, as are `step --all-joints`, `bodyid`, and a full `jointmap`/`sweep` session.
- `bodyid` is not part of `all` — it needs the robot on the ground while everything else
  runs suspended.
- Feedback-servo support requires servos manufactured after ~March 2024 (per Petoi docs);
  `jointmap` reports `NO_REPLY` per joint if yours predate that.

## Background reading

New to servos, serial links, IMUs, or system ID? See [READING.md](READING.md) — a short
curated list tied to the specific problems this package deals with.
