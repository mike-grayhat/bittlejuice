#!/usr/bin/env bash
# Build (and optionally flash) the patched OpenCat ESP32 firmware.
#
#   tools/build_firmware.sh                  # build the patched tree
#   tools/build_firmware.sh --stock          # build the unpatched base as a control
#                                            (unpatched IMAGE -- not the same as the
#                                             patched image's preprogrammed MODE)
#   tools/build_firmware.sh --baseline       # rebuild B10_251121, the version on the robot
#   tools/build_firmware.sh --rev SHA        # build any commit, unpatched
#   tools/build_firmware.sh --upload PORT    # build, then flash
#   tools/build_firmware.sh --realtime-default   # image that boots in realtime mode
#
# Realtime mode (XR / Xr) is a runtime switch and defaults to OFF, so a freshly flashed
# robot behaves exactly like an unpatched one: skills and gaits are interpolated, the gP
# stream is
# limited to 5 Hz. --realtime-default builds an image that comes up in realtime mode
# instead, for a robot that only ever runs the policy.
#
# Artifacts land in build/firmware/{patched,stock,baseline,rev-SHA}/.
#
# --baseline is the real rollback. The firmware the robot shipped with, B10_251121, is not
# a binary anyone needs to find: `#define DATE` is in this checkout's history, and the two
# commits that carry "251121" (32a1fcb, b2e5818) differ only in PetoiWebCodingBlocks/*.js,
# so they produce identical firmware. Building 32a1fcb reconstructs it from source.
# Functionally, not bit-identically -- core and library versions differ from whatever Petoi
# used in Nov 2025 -- but the source is what determines behaviour.
#
# Two things about this build are not what the upstream README says, and both are
# deliberate -- see docs/reflashing.md "Building" for the reasoning:
#
#   1. PartitionScheme=huge_app, not "Default 4MB with spiffs". The current tree with all
#      modules enabled is 1.81 MB and does not fit the 1.25 MB app slot of `default`; the
#      README predates that growth. Upstream's own Log/ASYNC_UPGRADE_GUIDE.md specifies
#      huge_app. The `nvs` partition is identical (0x9000, 0x5000) in both, so joint and
#      IMU calibration survive the change.
#   2. The sketch is staged into a directory named OpenCatEsp32/ before compiling.
#      arduino-cli requires the sketch folder name to match the .ino, and the submodule is
#      named firmware/. Staging is a copy, so the submodule is never
#      touched and the build cannot pick up stray files.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
FW_TREE="$REPO_ROOT/firmware"
CORE="esp32:esp32@2.0.17"   # 2.x: PetoiESP32Servo uses ledcSetup/ledcAttachPin, removed in core 3.x
FQBN="esp32:esp32:esp32:UploadSpeed=921600,CPUFreq=240,FlashFreq=80,FlashMode=qio,FlashSize=4M,PartitionScheme=huge_app,DebugLevel=none,PSRAM=disabled"

BASELINE_REV=32a1fcb   # last firmware-affecting commit of B10_251121; see header
# The commit the patch branch is based on. --stock must build THIS, not HEAD: the submodule
# is pinned to the patched branch, so its HEAD is patched and a --stock build from HEAD
# would silently be a patched build.
#
# Derived rather than pinned, so it cannot go stale when the patches are rebased onto a
# newer upstream: it is where the patch branch diverged from the fork's tracking branch.
STOCK_REV="$(git -C "$FW_TREE" merge-base HEAD origin/main 2>/dev/null || true)"

REV=""          # empty = build the working tree (patched); set = build that commit, unpatched
VARIANT=patched
PORT=""
RT_DEFAULT=0
VOICE_ARM=""    # empty = voice.h's default (3, the only sequence measured to arm the module)
while [ $# -gt 0 ]; do
  case "$1" in
    --voice-arm) VOICE_ARM="${2:?--voice-arm needs 0|1|2|3}"; shift 2 ;;
    --realtime-default) RT_DEFAULT=1; shift ;;
    --stock)    REV="${STOCK_REV:?cannot resolve the unpatched base: no merge-base between \
the submodule HEAD and origin/main. Fetch the submodule, or pass --rev SHA.}"
                VARIANT=stock;    shift ;;
    --baseline) REV=$BASELINE_REV; VARIANT=baseline; shift ;;
    --rev)      REV="${2:?--rev needs a commit}"; VARIANT="rev-${2}"; shift 2 ;;
    --upload)   PORT="${2:?--upload needs a port}"; shift 2 ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done
[ "$RT_DEFAULT" = 1 ] && VARIANT="$VARIANT-rt"
[ -n "$VOICE_ARM" ] && VARIANT="$VARIANT-arm$VOICE_ARM"
OUT="$REPO_ROOT/build/firmware/$VARIANT"
STAGE="$OUT/sketch/OpenCatEsp32"

# macOS ships `md5 -q`; Linux (and every CI runner) ships `md5sum`. The build is expected to
# run in both, and reflashing.md tells you to record what this prints.
md5_of() {
  if command -v md5 >/dev/null 2>&1; then md5 -q "$1"
  else md5sum "$1" | cut -d' ' -f1
  fi
}

command -v arduino-cli >/dev/null || { echo "arduino-cli not found" >&2; exit 1; }
arduino-cli core list 2>/dev/null | grep -q '^esp32:esp32 *2\.0\.17' || {
  echo "ESP32 core 2.0.17 not installed. Run:" >&2
  echo "  arduino-cli core install $CORE" >&2
  echo "  arduino-cli lib install ArduinoJson WebSockets WiFiManager" >&2
  echo "  git -C ~/Documents/Arduino/libraries clone --depth 1 https://github.com/mu-opensource/MuVisionSensor3.git" >&2
  exit 1
}

rm -rf "$STAGE"; mkdir -p "$STAGE"
if [ -n "$REV" ]; then
  git -C "$FW_TREE" archive "$REV" | tar -x -C "$STAGE"
else
  rsync -a --exclude '.git' "$FW_TREE/" "$STAGE/"
  # Fail loudly rather than silently flashing an image that cannot do realtime mode. Each
  # grep is one half of the switch: the flag and the code that reads it. A build carrying
  # the flag but not the dispatch would accept XR and change nothing.
  grep -q 'bool rlRealtimeQ = RL_REALTIME_DEFAULT;' "$STAGE/src/OpenCat.h" \
    || { echo "PATCH MISSING: no realtime-mode flag in OpenCat.h -- refusing to build" >&2; exit 1; }
  grep -q 'print6AxisMinInterval' "$STAGE/src/imu.h" \
    || { echo "PATCH MISSING: gP interval is still a constant -- refusing to build" >&2; exit 1; }
  grep -q 'rlRealtimeQ && token == T_INDEXED_SIMULTANEOUS_ASC' "$STAGE/src/reaction.h" \
    || { echo "PATCH MISSING: 'i' does not consult the realtime flag" >&2; exit 1; }
  grep -q 'case EXTENSION_RL_REALTIME:' "$STAGE/src/reaction.h" \
    || { echo "PATCH MISSING: no XR/Xr handler -- the mode could not be switched" >&2; exit 1; }
  grep -q 'if (steps > 0)' "$STAGE/src/motion.h" \
    || { echo "PATCH MISSING: transform() still delays on the zero-step path" >&2; exit 1; }
  grep -q 'Re-arming the voice module after power restoration' "$STAGE/src/reaction.h" \
    || { echo "PATCH MISSING: the voice module is not re-armed when battery power returns \
-- it cold-starts deaf and setup() does not re-run" >&2; exit 1; }
  grep -q 'Serial2.write((uint8_t)textResponse.charAt(0))' "$STAGE/src/io.h" \
    || { echo "PATCH MISSING: the Xiaozhi completion echo still targets SERIAL_VOICE -- on \
BiBoard V1.0 that injects unframed bytes into the voice module's command channel" >&2; exit 1; }
  grep -q 'voice: refusing the calibration token' "$STAGE/src/voice.h" \
    || { echo "PATCH MISSING: voice can still reach the 'c' token -- a spoken word could \
enter servo calibration and overwrite the factory joint offsets" >&2; exit 1; }
  grep -q 'if (rlRealtimeQ)' "$STAGE/src/voice.h" \
    || { echo "PATCH MISSING: read_voice() does not defer to realtime mode -- control_loop.py \
no longer sends XAd and would run with voice skills fighting the policy" >&2; exit 1; }
fi

echo "=== building $VARIANT from $FW_TREE${REV:+ @ $REV}"
echo "=== firmware version: $(sed -n 's/^#define DATE "\([0-9]*\)".*/\1/p' "$STAGE/src/OpenCat.h")"
BUILD_PROPS=()
EXTRA_FLAGS=""
if [ -n "$VOICE_ARM" ]; then
  # 0 sends nothing at boot, which is what B10_251121 effectively did. Use it to check
  # whether the module's own persistent armed state has been restored -- if it has, the
  # arming sequence (and its two spoken announcements) is not needed at all.
  EXTRA_FLAGS="$EXTRA_FLAGS -DVOICE_ARM_SEQUENCE=$VOICE_ARM"
  echo "=== voice arming sequence: $VOICE_ARM"
fi
if [ "$RT_DEFAULT" = 1 ]; then
  # compiler.cpp.extra_flags, not build.extra_flags: the latter is defined per board in the
  # ESP32 core and overriding it drops flags the core needs.
  EXTRA_FLAGS="$EXTRA_FLAGS -DRL_REALTIME_DEFAULT=1"
  echo "=== realtime mode ON at boot"
fi
if [ -n "$EXTRA_FLAGS" ]; then
  BUILD_PROPS+=(--build-property "compiler.cpp.extra_flags=$EXTRA_FLAGS")
fi
arduino-cli compile -b "$FQBN" "${BUILD_PROPS[@]+"${BUILD_PROPS[@]}"}" \
  --build-path "$OUT/build" --output-dir "$OUT/bin" "$STAGE"

BIN="$OUT/bin/OpenCatEsp32.ino.bin"
echo
echo "image:  $BIN"
echo "md5:    $(md5_of "$BIN")"
echo "bytes:  $(wc -c < "$BIN")"

if [ -n "$PORT" ]; then
  [ -n "$REV" ] && echo "NOTE: flashing the UNPATCHED $VARIANT image." >&2
  echo "=== flashing to $PORT"
  arduino-cli upload -b "$FQBN" -p "$PORT" --input-dir "$OUT/bin"
  echo "Flashed. Run the post-reflash battery in docs/reflashing.md before trusting anything."
fi
