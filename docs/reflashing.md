# Reflash runbook — patched ESP32 firmware

> **If you just want to flash a robot and run the policy, read [firmware.md](firmware.md)
> instead.** This file is the maintainer's runbook: the full post-flash measurement battery,
> what each check is gating, and the two different rollback targets. Work through it after
> any change to the patches, or whenever a measurement needs to be trusted again.

The `firmware/` submodule is pinned to a fork branch carrying patches
the robot does **not** yet have. Flashing is the only way to get them, and flashing is not a
neutral act: it also replaces the firmware build the measurement corpus was taken against.
Read this whole file before plugging anything in.

Make sure the submodule is actually checked out first -- an empty directory builds nothing:

```bash
git submodule update --init
```

## What changes, and why it matters

Since the patch set became a **runtime mode** (`XR` / `Xr`, off at boot), flashing it is a
much smaller commitment than it used to be: with realtime mode off the robot behaves exactly
like stock, so skills, gaits and the app are unaffected. What follows still applies, because
the build itself is a different image and the post-flash battery is what proves it.


| Patch | File | Effect |
|---|---|---|
| `i` dispatches `transform(..., speedRatio=0)` | `src/reaction.h` | No interpolation ramp on real-time pose commands. Removes 16–24 ms of blocking per tick. |
| `delay()` skipped on the zero-step path | `src/motion.h` | The single-write path no longer pays the 8 ms pacing delay. |
| `PRINT6AXIS_MIN_INTERVAL` 200 → 4 ms | `src/imu.h` | **Mandatory.** Without it the `gP` stream is capped at 5 Hz. 4 ms, not 3 — see the bandwidth note. |

**The version trap.** The robot runs **B10_251121**. The vendored tree is **B10_260717** —
about eight months newer. The 5 Hz `gP` limiter arrived in between
(`31ec912`, "Fixed the bug that caused the program to restart unexpectedly after enabling
the gesture sensor") — collateral from a crash fix, and it does not exist at all in the
build every hardware measurement was taken against. Flashing the
vendored tree *without* the `imu.h` patch caps the IMU stream at 5 Hz, `obs_builder`
rejects everything older than 50 ms, and `ang_vel` goes silently to zeros for the whole
run: the policy walks blind, the loop does not raise, and nothing in the telemetry says
why. Verify the patch is present before building.

**Everything else in eight months of upstream changes is unreviewed.** The three patches
are the intended changes; the version bump is a side effect that comes with them. That is
the main risk of this reflash, and it is why the measurement battery below is not
optional.

## Before flashing

1. **The way back exists, and it is in this checkout.** B10_251121 does not need to be
   found as a binary: `#define DATE` is versioned, and exactly two commits carry `"251121"`
   — `32a1fcb` (2025-11-21) and `b2e5818` (2025-11-23). They differ **only** in
   `PetoiWebCodingBlocks/*.js`, so their firmware is identical. `./tools/build_firmware.sh
   --baseline` rebuilds it, verified to compile at 1 806 577 bytes.

   Functionally, not bit-identically: core 2.0.17 and today's library versions are not
   necessarily what Petoi built with in November 2025. If the board supports firmware
   readback, dumping the running image is still worth the minute — it is the only artifact
   that settles a "did the rebuild really match?" question later.
2. Note the exact port and record `probe` output as the "before" reference:
   ```
   uv run --directory deploy python -m bittle_io --port $PORT --run-id before_flash probe
   uv run --directory deploy python -m bittle_io --port $PORT --run-id before_flash stream
   ```
3. **Build it, and build it from the patched tree.** The submodule is pinned to the
   `rl-realtime` branch, so its checked-out HEAD already carries the patches; the build
   script refuses to proceed if it does not. Building an unpatched tree would give the
   newer upstream
   build *with* the 5 Hz IMU limiter and *with* the blocking ramp: every downside of the
   version bump and none of the fixes. `tools/build_firmware.sh` refuses to build unless
   all three patches are present, so use it rather than calling `arduino-cli` by hand:

   ```
   ./tools/build_firmware.sh                 # patched  -> build/firmware/patched/bin/
   ./tools/build_firmware.sh --stock         # upstream -> build/firmware/stock/bin/
   ./tools/build_firmware.sh --upload $PORT  # build, then flash
   ```

   The `--stock` image is worth building too: it is the control that separates "the patches
   broke it" from "eight months of upstream changes broke it", and it is the only rollback
   target that exists in this repo (see **Rollback**).

4. **Toolchain**, one-time. The build is reproducible but the setup is not in this repo:

   ```
   arduino-cli core install esp32:esp32@2.0.17
   arduino-cli lib install ArduinoJson WebSockets WiFiManager
   git -C ~/Documents/Arduino/libraries clone --depth 1 \
       https://github.com/mu-opensource/MuVisionSensor3.git
   ```

   Core **2.x, not 3.x**: `src/PetoiESP32Servo` calls `ledcSetup()` / `ledcAttachPin()`,
   which core 3.0 removed. Everything else the sketch needs is vendored under `src/`
   (IRremote, Adafruit_NeoPixel, Seeed_Arduino_SSCMA, icm42670, ...).

   Two deviations from the upstream README's board settings, both deliberate:

   - **`PartitionScheme=huge_app`, not "Default 4MB with spiffs".** The current tree with
     every module enabled links to 1.81 MB and does not fit `default`'s 1.25 MB app slot —
     the README and `ESP32config/ESP32Configuration.jpg` predate that growth. Upstream's own
     `Log/ASYNC_UPGRADE_GUIDE.md` specifies `huge_app`. **The `nvs` partition is identical
     in both schemes** (0x9000, 0x5000), so joint and IMU calibration survive the change;
     that was checked, not assumed. `huge_app` drops the OTA slot, which nothing in the
     tree uses.
   - The sketch is staged into a directory named `OpenCatEsp32/` before compiling, because
     `arduino-cli` requires the folder name to match the `.ino` and the checkout is named
     `firmware`. The script copies; the checkout is never touched.

   **Compile both variants** (arduino-cli 1.5.1, core 2.0.17). The patched and stock
   images differ by a few dozen bytes, and that delta is the cheapest available
   confirmation that the build really did pick up the patched tree. Record the md5 the
   script prints and check it against the image you flash.

   **That md5 is of the app image** (`OpenCatEsp32.ino.bin`), which is what you flash when
   you build locally. A release image is the *merged* binary — bootloader + partitions +
   app — so it has a different md5, the one printed in the release notes. Compare like with
   like: the script's md5 against a local build, the release notes' md5 against a download.

5. Battery charged. Servos cool. Robot **suspended, legs free**.

## After flashing — the battery, in this order

Nothing is trusted until these pass. Each one has caught a plausible-looking wrong number
before.

```
export RUN=after_flash

# 1. Does it talk, what does it claim to be, can it switch modes, and how fast is gP?
#    `probe` carries both gates. It is the only experiment that sends gP and counts IMU
#    lines -- NOT `stream`, which starts the `f` JOINT feedback stream and never touches
#    the IMU. Reading the imu.h patch off `stream` would be the wrong instrument entirely.
#
#    NOTE the order matters and probe does it for you: realtime mode is OFF at boot, so a
#    correctly patched robot ALSO streams gP at 5 Hz until `XR` is sent. probe switches
#    the mode on, measures, and switches it back. Measuring the rate without switching
#    first cannot tell a patched robot from an unpatched one.
uv run --directory deploy python -m bittle_io --port $PORT --run-id $RUN probe
#    -> firmware  : version string CHANGED. It comes from `#define DATE` in the base
#                   commit the patch branch sits on, so read it out of the source rather
#                   than expecting a fixed value:
#                     git -C firmware show HEAD:src/OpenCat.h \
#                       | grep '#define DATE'
#                   Record what you get.
#    -> realtime  : THE gate on the whole patch set. "NOT SUPPORTED" means the build did
#                   not pick up the patched tree. STOP -- control_loop.py will refuse to
#                   run against it anyway.
#    -> imu rate  : measured WITH realtime on, so it gates imu.h. Fails both directions.
#       ~5 Hz     : mode switched but the interval did not follow it. STOP.
#       ~250 Hz   : the expected case -- 250 Hz is the link's ceiling and the 4 ms limiter
#                   caps at the same place. See the section below.
#       lower than 250 but not 5: NOT explained by this patch set. STOP and find out why.
#       above ~255: arithmetically impossible at 46 bytes/line over 115200. Check the raw
#                   transcript for truncated lines (protocol.py rejects them silently).

# 2. Joint feedback stream: rate, jitter, per-column noise. Not a gate on any patch --
#    nothing here touches the `f` path -- so a change would itself be the finding.
uv run --directory deploy python -m bittle_io --port $PORT --run-id $RUN stream

# 3. Host command rate still holds at 50 Hz.
uv run --directory deploy python -m bittle_io --port $PORT --run-id $RUN cmdrate --hz 50

# 4. Command -> motion latency. THE measurement the reaction.h patch targets.
uv run --directory deploy python -m bittle_io --port $PORT --run-id $RUN latency
#    -> was 103 ms median (94-113) with the ramp. Expect a large drop.

# 5. Servo tau / slew. The ramp was upstream of the servo, so removing it may change
#    what these fits see. These numbers are baked into every exported policy.
uv run --directory deploy python -m bittle_io --port $PORT --run-id $RUN step
uv run --directory deploy python -m bittle_io --port $PORT --run-id $RUN sweep

uv run --directory deploy python -m bittle_io --run-id $RUN report
```

## First boot after a flash blocks on a prompt

A freshly flashed board comes up as a **new board** and stops inside `imuSetup()`:

```
- Calibrate the Inertial Measurement Unit (IMU)? (Y/n):
```

It is a blocking `getUserInputChar()`, so until it is answered the board looks dead in a
specific and misleading way: version queries return the prompt text, the mode switch returns
nothing, and the gP stream reads 0 Hz. Nothing is broken.

**Answer `n` unless the robot is flat on a table.** The calibration wants a level, still
robot, and a suspended one is neither. Declining keeps the offsets already in NVS, which
survive the flash — a correct boot then prints `calibration already done`.

```bash
python3 -c "import serial,time; s=serial.Serial('$PORT',115200,timeout=.5); time.sleep(.5); s.write(b'n\n')"
```

Then reset and read the banner before trusting anything else. A healthy boot ends at
`Waiting for a BLE client connection to notify...` and passes through `Using ICM offsets:`,
`calibration already done` and `Calibrated Zero Position`.

## Streaming experiments need realtime mode, and will wedge the board without it

`bittle_io` switches the session into realtime mode for every command except `probe`, which
owns the switch because testing it is the gate. Do not bypass that.

The reason is the interpolation ramp, not the IMU: it BLOCKS the firmware's main loop for
16–24 ms per pose command, longer for a large move. Anything streaming motion faster than
that overruns the serial buffer. Measured: `cmdrate --hz 50` in preprogrammed mode left
the board emitting empty lines and answering no command at all, recovered only by a reset.
`sweep` and `bodyid` drive at 30–50 Hz and have the same exposure.

## The stream rate is the serial link, not the IMU (read before flashing)

This section previously said the post-patch direction of the stream rate was "genuinely
unknown until measured". Reading the code and doing the arithmetic resolves most of it.

`print6Axis()` is called from two places that matter: `readEnvironment()` in the main loop
(`moduleManager.h:573`, gated on `imuUpdated && printGyroQ`), and **inside `transform()`'s
interpolation loop** (`motion.h:186`). Today the ramp blocks the main loop for 16–24 ms
out of every 20 ms tick, so `readEnvironment()` barely runs, and the `transform()` call
site supplies most of the traffic.

But neither call site is what sets the observed rate. One IMU line is exactly
`"ICM:" + 3×%6.2f + 3×%7.1f + "\t"` = 44 characters, plus the `PTL()` CRLF = **46 bytes**.
At 115200 8N1 the link carries 11 520 B/s. The measured 249.44 Hz is 11 474 B/s —
**99.6% of capacity.** The stream rate that the whole system treats as a property of the
IMU is the saturation point of the serial link.

That changes the risk picture in three ways:

- **The rate cannot fall because of this patch.** After it, `transform()` contributes one
  call per command instead of 3–4, but the main loop is no longer blocked, so
  `readEnvironment()` runs far more often. Total *demand* for `print6Axis()` goes up, and
  the link caps the *delivered* rate at ~250 Hz either way. A large drop would mean
  something else broke — still a STOP, but now an unexpected one rather than a coin flip.
- **`PRINT6AXIS_MIN_INTERVAL` is 4 ms, not 3.** 3 ms permits 333 Hz = 15.3 kB/s = 133% of
  the link. The firmware would then spend its main loop blocked in `Serial.write()` waiting
  for the TX buffer — trading the interpolation-ramp stall this patch set removes for a
  UART-backpressure stall. 4 ms caps at exactly 250 Hz, the rate the link can actually
  sustain and the rate every hardware measurement was taken at, and it gives step 2 an
  unambiguous expected value instead of a range.
- **The "native USB CDC, baud is nominal" escape hatch is closed.** That was an inference
  from the port enumerating as `/dev/cu.usbmodem*`. It does not hold: `BOARD "B10"` is
  BiBoard V1.0, a classic ESP32-WROOM-32 with no native USB, and both `BT_SSP` and
  `BT_BLE` are defined, which only classic ESP32 supports. Whatever the host side
  enumerates as, the ESP32 side of the bridge is a hardware UART running at the
  `Serial.begin(115200)` the sketch sets. 11 520 B/s is a real ceiling.

**Why the residual UART blocking is acceptable.** At 250 Hz the link is busy essentially
100% of the time, so `printToAllPorts` will block on a full TX buffer — but in ~46-byte
units spread across ~5 writes per control tick, between which the main loop still reaches
`readSignal()`. The ramp was worse in kind, not just degree: it blocked for 16–24 ms in
one chunk *inside* `transform()`, where `readSignal()` is never called, so commands piled
up in the RX buffer. Replacing one big blind stall with five small ones that still poll is
the point of the patch.

If step 4 comes back disappointing and the `cmdrate` jitter looks UART-shaped, the ready
follow-up is `PRINT6AXIS_MIN_INTERVAL = 5` (200 Hz, 80% of the link, real headroom, still
4 samples per control tick). It is deliberately *not* in this flash: changing the ramp and
the stream rate together would make step 4 uninterpretable.

**Why the stream rate matters at all:** `deploy/obs_builder.py` rejects samples older than
`stale_after_s = 0.05`, and `base_ang_vel` is finite-differenced from consecutive samples.
A materially slower stream degrades that channel; a stream slower than 20 Hz starts failing
the staleness check outright, and the failure is silent (ang_vel goes to zeros, the loop
does not raise).

## Rollback

There are two different rollbacks and it is worth being precise about which one you have,
because "everything is in the repository, worst case just roll back" is only half true:

| target | what it gets you | where it comes from |
|---|---|---|
| **unpatched B10_260717** | isolates *the patches* from *eight months of upstream change* | `./tools/build_firmware.sh --stock`, in this repo |
| **B10_251121** | the build the entire measurement corpus was taken against | `./tools/build_firmware.sh --baseline` — rebuilt from `32a1fcb` in this checkout's history |

If the battery says the new build is worse in a way the `--stock` control also reproduces,
the cause is upstream drift rather than the patches, and `--baseline` puts the robot back
on the source the measured constants were taken against.

The one residual caveat: `--baseline` reconstructs 251121 from *source*, with a toolchain
that is not necessarily Petoi's. Same source, therefore same behaviour, but not the same
bytes — so if the rebuilt baseline does not reproduce the old measurements, "the rebuild
differs from what shipped" stays on the list of explanations.

The patches themselves are three small diffs and can be re-applied to whatever source tree
matches the robot.
