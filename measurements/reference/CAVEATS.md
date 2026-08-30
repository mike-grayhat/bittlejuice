# reference — firmware B10_260717 + the three RL patches

Taken immediately after flashing, robot suspended. The `pre` column below is the same robot
on the same battery minutes earlier, running stock firmware.

## Trustworthy — and this is the result

| quantity | pre | post | source |
|---|---|---|---|
| firmware | B10_251121 | `G Bittle X B10_260717 ?` | `probe` |
| **gP IMU stream** | 249.91 Hz | **248.63 Hz** | `probe` — **the gate on the imu.h patch** |
| **command → body motion** | **197.8 ms** | **54.3 ms** (43.9–60.8) | `latency` |
| link round trip | 15.5 ms | 20.3 ms | `latency --phase echo` |
| command rate @ 50 Hz | 50.0 Hz, 0 overruns | 50.0 Hz, 0 overruns | `cmdrate` |
| polled request/response | 11.45 Hz | 14.57 Hz | `probe` |
| `f` joint stream | 5.5 Hz | 9.25 Hz | `probe` |

3.7× less dead time, and the spread fell from 144 ms wide to 17.

## Do NOT consume — `step` and `sweep` are INVALID on this firmware

Both are in `manifest.json` and their CSVs, unmarked, and both are wrong. They are kept
because deleting a measurement hides that it was attempted.

- **`step`**: `tau_ms: 400.0`, `slew_deg_s: 395.1`. `step_summary.csv` gives the fit
  quality the manifest drops: **R² = −0.29 and −0.081**. Both worse than fitting a constant.
- **`sweep`**: `slowest_joint_slew_deg_per_s: 39.0`, against 109.7 for the same instrument
  on stock firmware.

**Neither is the servo changing.** Both experiments poll joint feedback at ~15 Hz, and
`sweep` runs to 3 Hz — five samples per cycle. That was tolerable while the firmware
interpolation ramp stretched every response to ~100 ms. With the ramp removed the plant is
fast enough to alias, so a first-order-lag fit has nothing to lock onto and the max-slew
figure is a lower bound on the sampling rate rather than a measurement of the joint.

In particular: **do not** size anything from the 39 °/s figure. `sweep` prints advice to set
`control_loop.MAX_ANGLE_DELTA_PER_STEP_RAD` from it — that constant no longer exists (the
loop sends the raw target and keeps ServoModel as an observer), and the number is an
artifact. Both the advice and the stale reference are corrected in `sweep.py`.

Re-fitting tau/slew needs a method that survives a fast plant — most likely the IMU-based
`bodyid` route rather than polled joint readback. Until then `model/bittle.xml`'s kv and
armature stay as they are, with the standing caveat that they were fitted to a t63 that
included the firmware ramp the patch deletes.
