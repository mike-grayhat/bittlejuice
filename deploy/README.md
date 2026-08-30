# Bittle X Raspberry Pi deployment

Standalone runtime for the trained locomotion policy. Deliberately has **no**
dependency on jax/mujoco or on `sim/` -- just `numpy` + `pyserial`, so it installs
cleanly on the Pi's own ARM Python:

```
cd deploy
pip install -e .
```

Every command in this file assumes you are in `deploy/`, which is how the Pi runs it.
From the repo root the same commands are `python deploy/control_loop.py ...` and
`uv run --directory deploy python -m bittle_io ...`; `--policy` takes the same
`policy/x.npz` either way, because the CLI falls back to resolving it against this
directory.

The Pi is expected to be physically mounted on the robot and wired (serial)
to the existing NyBoard/BiBoard, which keeps driving the servos and owns the
IMU. This package runs the trained policy's *inference* only and talks to
the board over Petoi's OpenCat ASCII serial protocol.

## What the hardware turned out to be

All measured on this robot; the raw capture is in `measurements/reference/` and the
constants derived from it live in `sim/mj_vec_env.py`.

- **Serial: 115200 baud**, ASCII, newline-terminated (`bittle_io/link.py`).
  Commands must be spaced at least 20 ms apart or the MCU drops them.
- **Board: ESP32 / BiBoard**, firmware B10_251121. Do not consult PetoiCamp/OpenCat
  (the AVR/NyBoard tree) for protocol details: it is a *different board*, its `v`/`V`
  gyro tokens do not exist here (the IMU lives under `g` sub-commands) and it has no
  `f` token at all. `firmware/` is the right source.
- **Joint positions are not readable, at all.** A feedback read costs ~20 ms per joint
  and physically detaches the servo while it happens; a whole-body frame arrives at a
  measured 9.25 Hz (5.5 Hz on stock firmware) against a 50 Hz loop, and interleaving a
  move with a read -- what a control loop does -- measured 14.57 Hz. The values also came
  back intermittently corrupt. `j` returns `currentAng` --
  the firmware's memory of what it commanded -- not a measurement. So `dof_pos`/
  `dof_vel` are *estimated* by `obs_builder.ServoModel`, which replays the host's own
  commanded targets through the measured servo response (τ=33 ms, ≤137 °/s). The
  simulator observes the same estimate, and its domain randomization represents the
  resulting model error structurally rather than as sensor noise.
- **Calibration is measured and filled in** (`joint_calibration.py`): indices 8-11
  shoulders, 12-15 knees, sign +1 and offset 0.0 on all eight, each established three
  independent ways. `deg_to_petoi` still raises on an unfilled entry, so a *different*
  robot fails loudly.
- **The rate limit lives in the policy `.npz`**, not in a constant here -- a
  module-level copy existed and went stale.
- **Stand-down** sends `gb` then `d` (rest) via `SerialLink.close()`, even on exception.

## Bring-up sequence (staged, off-the-ground first)

This is the procedure for a new robot. Re-run (c) onward after any firmware reflash --
see `docs/reflashing.md`.

**a. Serial handshake only.** `uv run python -m bittle_io` (from `deploy/`) for the
   probe/experiment CLI: confirm the port, the 115200 link, and the `gP` IMU stream
   rate before any motion command.

**b. Joint calibration.** `bittle_io ... legmap` / `jointmap`, operator-guided, one
   joint wiggled at a time. Do not infer indices from the nominal Petoi layout.

**c. Propped up, legs free.** `bittle_io ... step` and `sweep` to measure the servo's
   τ and slew rate; these are baked into the exported policy `.npz`, and the sim's
   actuator model is fitted to them.

**d. Propped up, closed loop.** `control_loop.py --policy <npz> --hold` first (home pose
   only), then the same policy at a near-zero commanded velocity. `--policy` is required
   even for `--hold`, which takes the home pose from it -- there is no default, because a
   stale one drives the robot with the wrong servo model. Watch for sign errors or
   false-triggered watchdogs before trusting it further.

**e. Floor walking.** Only after (a)-(d) pass cleanly: closed-loop walking on the
   floor, spotted by hand for the first attempt. Use `--log-obs` and diff against sim
   with `tools/obs_diff.py`.

## Files

- `control_loop.py` -- the 50 Hz loop: IMU -> obs -> policy -> safety -> servos, with
  the tilt watchdog, IMU-staleness and non-finite-observation guards.
- `obs_builder.py` -- builds the 33-dim observation, including the `ServoModel`
  joint-state estimator and the yaw-rate bias calibration.
- `policy_numpy.py` -- dependency-free MLP forward pass, loads the `.npz` produced by
  `sim/export_policy_mj.py` (which verifies it against torch in float64 first).
- `joint_calibration.py` -- MJCF joint <-> Petoi index/sign/offset, measured.
- `bittle_io/` -- the measured, tested serial layer: link, protocol, state reader, and
  the offline characterization experiments. Imported directly by `control_loop.py`.

`serial_bridge.py` was deleted here after the removal was confirmed to break nothing.
It documented the AVR token table, which does not apply to this board, and its
single-line `j` read is a known trap that `bittle_io/protocol.py` still cites. Recover
it with `git log --diff-filter=D -- deploy/serial_bridge.py`.
