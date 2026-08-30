# Firmware: what it changes, and how to flash it

**Stock firmware cannot run this control loop**, and the reason is worth understanding rather
than working around. Petoi's firmware shapes every pose command through a blocking
raised-cosine ramp, and caps the IMU stream at 5 Hz. Both are right for the preprogrammed
skills and gaits the robot ships with. Both are fatal for a 50 Hz host loop: at 50 Hz the
host *is* the trajectory, so the ramp is redundant smoothing that blocks 16–24 ms per tick
against a 20 ms budget, and a 5 Hz attitude stream is stale on arrival.

So the patches make it a **runtime mode** rather than a permanent change:

| command | effect |
|---|---|
| `XR` | realtime: `i` writes its targets once and returns, IMU stream at the link's ceiling (~250 Hz) |
| `Xr` | preprogrammed: interpolation ramp on, IMU capped at 5 Hz |

**Realtime mode is off at boot**, so flashing this changes nothing about how the robot walks,
dances or answers the app until something asks for it.

| patch | file | effect |
|---|---|---|
| the mode flag and its serial code | `src/OpenCat.h` | `rlRealtimeQ`, `setRealtimeMode()`. Off unless built with `-DRL_REALTIME_DEFAULT=1`. |
| `i` consults the flag, plus the `XR`/`Xr` handler | `src/reaction.h` | In realtime mode only, `i` dispatches `transform(..., speedRatio=0)`. The big one: command→motion latency 198 ms → 54 ms, spread 144 ms → 17 ms. |
| skip `delay()` on the zero-step path | `src/motion.h` | The single-write path stops paying an 8 ms pacing delay. |
| the IMU interval becomes a variable | `src/imu.h` | 4 ms in realtime mode, upstream's 200 ms outside it. Without this the observation builder rejects every sample as stale and `ang_vel` silently goes to zeros. |
| voice reactions defer to the mode | `src/voice.h` | In realtime mode only, `read_voice()` drops the reaction but still acknowledges it. A recognised word otherwise queues a 2.5 s skill that drives the servos against the policy — and any bystander can trigger it by speaking. |
| the English cold-boot branch, restored | `src/voice.h` | Not part of realtime mode; a straight fix — without it voice is dead after every power cycle. |
| the completion echo goes to Grove, not the voice UART | `src/io.h` | Not part of realtime mode either. On BiBoard V1.0 the two are different ports, and the stray byte killed the next voice command. |
| re-arm voice when battery power returns | `src/reaction.h` | Not part of realtime mode either. The module is on the battery rail; the main board on USB is not. |

**On voice:** four of the rows above touch the voice module, and none of them needs anything
from you — on this image voice arms itself silently at boot, answers when you speak to it,
and is gated off while a policy is running so a bystander cannot drive the servos mid-run.
Two of the four are plain upstream bug fixes, worth having even if you never run a policy:
without them voice recognition is dead after every power cycle on an English-default robot.
The full account, needed only if you are working on the module itself, is in
[voice.md](voice.md).

## Two words that are easy to confuse

The difference matters when something is not working.

- **preprogrammed mode** is this firmware with realtime *off*. Same image, switch flipped.
  It is how the robot boots.
- **stock firmware** is an unpatched build (`tools/build_firmware.sh --stock`, kept as a
  control). It has no `XR` token at all.

A robot in preprogrammed mode and a robot on stock firmware behave identically; only the
first can be switched out of it, and `bittle_io probe` is what tells them apart.
`control_loop.py` sends `XR` at startup and `Xr` on the way out, and refuses to run if the
acknowledgement does not come back — an unpatched robot reads `XR` as a hardware module code,
and running against one would give you sluggish walking and a blind policy with nothing in the
output to say so.

---

## Option A: flash the prebuilt image

Release images are built by [`.github/workflows/firmware-release.yml`](../.github/workflows/firmware-release.yml)
and attached to this repo's GitHub releases: the workflow checks out the pinned `firmware/`
submodule, compiles it with `tools/build_firmware.sh` — patch-verification gates and all —
merges the three binaries into one, and publishes the image with its md5 and the robot,
board and revision it was built for, read out of the tree that was actually compiled.

**If the releases page is empty, there is no image yet — use [Option B](#option-b-build-it-yourself),
which is the same build on your own machine.** Tagging a commit (`v*`) is what produces one.

### Check that it is for your robot first

The image is ESP32 machine code for the robot's board, so your computer's OS does not matter.
What matters is the board. These are compile-time and cannot adapt:

| the image assumes | if yours differs |
|---|---|
| **Bittle X** (`#define BITTLE`) | Nybble, Cub, VT and MINI need their own build — use Option B |
| **BiBoard V1.0** (`#define BiBoard_V1_0`) | V0.1 / V0.2 / BiBoard2 have a different `PWM_pin[]` table; the wrong servos would move |
| **Rev D/E** (`#define RevDE`) | a RevB board reads battery voltage on the wrong pin |

Good news on the two things that vary most between individual Bittle X units: the **IMU chip
is auto-detected** (both MPU6050 and ICM42670 drivers are compiled in, selected by I2C address
scan), and so are the optional peripheral modules. You do not need to match those.

### Flash it

Per-robot servo calibration and IMU offsets live in the `nvs` partition (`0x9000`, size
`0x5000`), which this image does not move — so a normal flash preserves them.

> **Do not run `esptool erase_flash`.** It is the most common piece of flashing advice on the
> internet and it wipes `nvs`, destroying your robot's factory servo calibration. Nothing here
> needs it. The single merged image below is exactly why: there are no offsets to get wrong,
> so there is no reason to reach for a wipe.

```bash
pip install 'esptool<5'      # 5.x drops the esptool.py entry point
esptool.py --chip esp32 --port /dev/ttyUSB0 write_flash 0x0 bittle-rl-<version>.bin
```

Check the image's md5 against the one in the release notes before flashing.

### First boot blocks on a prompt

A freshly flashed board comes up as a **new board** and stops inside `imuSetup()`:

```
- Calibrate the Inertial Measurement Unit (IMU)? (Y/n):
```

It is a blocking read, so until it is answered the board looks dead in a specific and
misleading way: version queries return the prompt text, the mode switch returns nothing, and
the IMU stream reads 0 Hz. Nothing is broken.

**Answer `n` unless the robot is flat on a table.** The calibration wants a level, still robot,
and a suspended one is neither. Declining keeps the offsets already in NVS, which survive the
flash — a correct boot then prints `calibration already done`.

```bash
python3 -c "import serial,time; s=serial.Serial('$PORT',115200,timeout=.5); time.sleep(.5); s.write(b'n\n')"
```

Then reset and read the banner. A healthy boot ends at
`Waiting for a BLE client connection to notify...`.

### Confirm it took

```bash
uv run --directory deploy python -m bittle_io --port $PORT --run-id after_flash probe
```

Three lines matter, and each fails in both directions:

- **realtime** — `NOT SUPPORTED` means the image did not carry the patches. Stop;
  `control_loop.py` will refuse to run against it anyway.
- **imu rate** — measured *with* realtime on. ~250 Hz is the expected case (that is the
  serial link's ceiling, not the IMU's). ~5 Hz means the mode switched but the interval did
  not follow. Anything else is unexplained — stop and find out why.
- **firmware** — record the version string you get.

---

## Option B: build it yourself

Needed for any robot outside the table above, and for any change to the patches.

```bash
git submodule update --init                  # an empty firmware/ builds nothing

./tools/build_firmware.sh                    # patched  -> build/firmware/patched/bin/
./tools/build_firmware.sh --realtime-default # image that boots in realtime mode
./tools/build_firmware.sh --stock            # the unpatched base, as a control
./tools/build_firmware.sh --upload $PORT     # build, then flash
```

The script refuses to build unless all the patches are present, so use it rather than calling
`arduino-cli` by hand. For a different robot or board, edit the `#define` block at the top of
`firmware/OpenCatEsp32.ino` first.

One-time toolchain setup:

```bash
arduino-cli core install esp32:esp32@2.0.17
arduino-cli lib install ArduinoJson WebSockets WiFiManager
git -C ~/Documents/Arduino/libraries clone --depth 1 \
    https://github.com/mu-opensource/MuVisionSensor3.git
```

Core **2.x, not 3.x**: `src/PetoiESP32Servo` calls `ledcSetup()` / `ledcAttachPin()`, which
core 3.0 removed.

Then follow **[reflashing.md](reflashing.md)** — the full post-flash measurement battery, the
rollback targets, and why each check exists. Do not skip its post-flash checks: the IMU-rate
gate in particular can fail in both directions, and a silently blind policy looks like a bad
policy.

---

## Cutting a release image (maintainer)

`arduino-cli` emits three binaries that must be flashed together — and a partition scheme
(`huge_app`, 3 MB app slot) that is *not* the ESP32 default. Flashing only the app image onto
a board still carrying a 1.25 MB app partition would run off the end of the slot. Merging them
removes both hazards:

```bash
cd build/firmware/patched/bin
esptool.py --chip esp32 merge_bin -o bittle-rl-<version>.bin \
    --flash_mode qio --flash_freq 80m --flash_size 4MB \
    0x1000  OpenCatEsp32.ino.bootloader.bin \
    0x8000  OpenCatEsp32.ino.partitions.bin \
    0x10000 OpenCatEsp32.ino.bin
```

The flash parameters must match the `FQBN` in `tools/build_firmware.sh`.

**In practice you do not run this by hand.** Bump the `firmware/` submodule pointer, commit,
and push a `v*` tag; `.github/workflows/firmware-release.yml` runs exactly the build and the
merge above and publishes the result, so the released image is always the pinned submodule
rather than whatever was in someone's `build/` directory. The commands here are the
definition of what that workflow does, and what to fall back to if it is unavailable. Run the
workflow from the Actions tab to test a build without tagging: it uploads the image as a
workflow artifact and skips the release.
