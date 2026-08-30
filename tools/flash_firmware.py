"""Flash a firmware image only if it is for THIS robot, and take a restorable backup first.

Why this exists rather than a bare `esptool write_flash`: the released image is compile-time
specific to one robot, board and revision, and the two failure modes are asymmetric.

  * Wrong robot or board -> the wrong servos move. Recoverable by reflashing, but it can
    slam joints against their stops first, so it is worth preventing.
  * `esptool erase_flash` -> wipes the `nvs` partition holding the FACTORY servo calibration
    and IMU offsets. Those are per-robot, set at the factory, and not reproducible at home.
    This is the one thing here that is genuinely unrecoverable, and the reason step 1 of a
    flash is a backup rather than a write.

What can and cannot be checked before writing:

  | model (Bittle X / Nybble Q / ...) | printed in the `?` reply     | GATED       |
  | board (B01 / B02 / B10)          | printed in the `?` reply     | GATED       |
  | revision (RevB / RevDE)          | never printed, not readable  | post-flash  |

The revision only selects which analog pin reads the battery, so a mismatch shows up as a
nonsense voltage. That is what the post-flash check looks for -- it is a real signal, but it
arrives after the write, which is why the docs say the gate does not cover revision.

    uv run python tools/flash_firmware.py --port $PORT --image bittle-rl-v0.1-...bin
    uv run python tools/flash_firmware.py --port $PORT --restore backups/<stamp>.bin
"""

import pathlib as _pathlib
import sys as _sys

_R = _pathlib.Path(__file__).resolve().parents[1]
_sys.path[:0] = [str(_R / "deploy")]        # reuse the tested serial layer, do not reimplement

import argparse
import datetime
import hashlib
import json
import pathlib
import re
import shutil
import subprocess

from bittle_io.link import LinkError, SerialLink, find_port

from firmware_target import TargetError, read_target

FLASH_SIZE = 0x400000                 # 4 MB: bootloader + partitions + nvs + app, in one file
# 2S LiPo. Matches link.voltage()'s own sanity band; outside it, the reading is not a battery.
VOLTAGE_MIN, VOLTAGE_MAX = 5.0, 9.0
# The `?` reply carries "<MODEL> <BOARD>_<DATE>", e.g. "G Bittle X B10_260820 ?".
_VERSION_TOKEN = re.compile(r"\b([A-Za-z]+\d*)_(\d{6})\b")
# A freshly flashed board blocks in imuSetup() and answers every query with this instead.
_BOOT_PROMPT = "Calibrate the Inertial Measurement Unit"


class FlashError(RuntimeError):
    pass


# -- identity ---------------------------------------------------------------
# Pure functions over the banner text, so the gate is testable without a robot.

def parse_banner(text):
    """`?` reply -> {'model', 'board_code', 'fw_date'}; values are None when absent.

    The board token is matched by SHAPE (letters, optional digits, underscore, six digits)
    rather than by searching for "B", because the model string itself contains letters that
    would match a laxer pattern -- "Bittle X" is not a board code.
    """
    text = " ".join((text or "").split())
    out = {"model": None, "board_code": None, "fw_date": None, "raw": text}
    if not text:
        return out
    m = _VERSION_TOKEN.search(text)
    if m:
        out["board_code"], out["fw_date"] = m.group(1).upper(), m.group(2)
        # The model is whatever precedes the version token, minus the leading status letter
        # the firmware prints ("G Bittle X B10_260820 ?" -> "Bittle X").
        head = text[:m.start()].strip().split()
        if head and len(head[0]) == 1:
            head = head[1:]
        out["model"] = " ".join(head) or None
    return out


def check_identity(banner, expected):
    """-> list of human-readable refusal reasons. Empty list means safe to flash.

    Anything unknown is a refusal, not a pass: a board that will not say what it is could be
    anything, and this function exists precisely for the case where we do not know.
    """
    reasons = []
    if not banner.get("raw"):
        return ["the board did not answer `?` at all, so its identity is unknown"]
    if _BOOT_PROMPT in banner["raw"]:
        return ["the board is waiting at the IMU calibration prompt, so `?` returns the "
                "prompt instead of a version"]
    if banner["model"] is None or banner["board_code"] is None:
        return [f"could not parse a model and board from the reply: {banner['raw']!r}"]

    want_model, want_board = expected.get("model"), expected.get("board_code")
    if want_model and banner["model"] != want_model:
        reasons.append(f"robot is {banner['model']!r}, image is for {want_model!r}")
    if want_board and banner["board_code"] != want_board:
        reasons.append(
            f"board reports {banner['board_code']!r}, image is for {want_board!r} -- "
            "the PWM pin table differs, so the wrong servos would move"
        )
    return reasons


def expected_from_image(image, meta=None, tree=None):
    """Where the image says it is for: sidecar JSON if present, else the local firmware tree."""
    side = pathlib.Path(meta) if meta else pathlib.Path(str(image) + ".json")
    if not side.exists() and not meta:
        side = pathlib.Path(image).with_suffix(".json")     # ...-fw260820.json
    if side.exists():
        d = json.loads(side.read_text())
        d["_source"] = str(side)
        return d
    d = read_target(tree)
    d["_source"] = "local firmware/ tree (no sidecar found beside the image)"
    return d


# -- serial + esptool -------------------------------------------------------

def read_board(port, answer_prompt=True):
    """Open the link, answer the first-boot prompt if it is blocking, return the banner."""
    with SerialLink(port=port, verbose=False) as ln:
        raw = ln.query_version() or ""
        if answer_prompt and _BOOT_PROMPT in raw:
            # Blocking read inside imuSetup(). `n` keeps the offsets already in NVS, which is
            # what you want on a robot that is not flat on a table. See docs/firmware.md.
            print("  board is at the IMU calibration prompt; answering 'n' (keeps NVS offsets)")
            ln.send("n")
            ln.sleep(1.0)
            raw = ln.query_version() or ""
        volts = None
        try:
            volts = ln.voltage()
        except LinkError:
            pass
        return parse_banner(raw), volts


def _esptool(*args, port, baud="921600"):
    exe = shutil.which("esptool.py") or shutil.which("esptool")
    if not exe:
        raise FlashError("esptool not found. Install it: pip install 'esptool<5'")
    cmd = [exe, "--chip", "esp32", "--port", port, "--baud", baud, *args]
    print(f"  $ {' '.join(cmd)}")
    if subprocess.run(cmd).returncode != 0:
        raise FlashError(f"esptool failed: {' '.join(args[:1])}")


def backup(port, out_dir):
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    path = out_dir / f"{re.sub(r'[^A-Za-z0-9]+', '_', port).strip('_')}-{stamp}.bin"
    print(f"\nBacking up the whole 4 MB flash -> {path}")
    print("  (bootloader + partition table + nvs + app; this is the exact-restore path)")
    _esptool("read_flash", "0", hex(FLASH_SIZE), str(path), port=port)
    return path


def md5_of(path):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# -- main -------------------------------------------------------------------

def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--port", default=None, help="default: first usbmodem-like device")
    p.add_argument("--image", help="merged .bin to flash")
    p.add_argument("--restore", metavar="BIN", help="write a previous backup back instead")
    p.add_argument("--meta", help="target sidecar JSON (default: alongside --image)")
    p.add_argument("--tree", default=None, help="firmware tree for the fallback target read")
    p.add_argument("--expect-model", help="override the expected MODEL, e.g. 'Bittle X'")
    p.add_argument("--expect-board", help="override the expected BOARD code, e.g. B10")
    p.add_argument("--backup-dir", default=str(_R / "backups"))
    p.add_argument("--no-backup", action="store_true",
                   help="skip the pre-flash dump. Only if you already have one.")
    p.add_argument("--force", action="store_true",
                   help="flash despite a refused identity check")
    p.add_argument("--dry-run", action="store_true",
                   help="run the gate and stop before touching the board")
    a = p.parse_args(argv)

    if not a.image and not a.restore:
        p.error("one of --image or --restore is required")
    image = pathlib.Path(a.restore or a.image)
    if not image.is_file():
        print(f"flash: {image} not found", file=_sys.stderr)
        return 1

    # -- what the image is for
    if a.restore:
        expected = {"model": a.expect_model, "board_code": a.expect_board,
                    "_source": "--expect flags" if a.expect_model else "(none: restore)"}
    else:
        try:
            expected = expected_from_image(image, a.meta, a.tree)
        except (TargetError, json.JSONDecodeError) as e:
            print(f"flash: cannot determine what this image is for: {e}", file=_sys.stderr)
            return 1
    if a.expect_model:
        expected["model"] = a.expect_model
    if a.expect_board:
        expected["board_code"] = a.expect_board.upper()

    print(f"image    : {image.name}")
    print(f"expects  : model={expected.get('model')!r} board={expected.get('board_code')!r}"
          f" rev={expected.get('rev')!r}")
    print(f"  (from {expected.get('_source')})")

    if expected.get("md5"):
        actual = md5_of(image)
        if actual != expected["md5"]:
            print(f"\nREFUSING: md5 mismatch.\n  image   {actual}\n  sidecar {expected['md5']}"
                  "\nThe download is corrupt or the sidecar is for a different build.",
                  file=_sys.stderr)
            return 1
        print(f"md5      : {actual} (matches sidecar)")

    # -- what the robot is
    try:
        port = find_port(a.port)
    except LinkError as e:
        print(f"flash: {e}", file=_sys.stderr)
        return 1
    print(f"port     : {port}\n")
    print("Reading the board's identity...")
    try:
        banner, volts = read_board(port)
    except LinkError as e:
        print(f"flash: could not talk to the board: {e}", file=_sys.stderr)
        return 1
    print(f"  reply  : {banner['raw'] or '(nothing)'}")
    print(f"  model  : {banner['model']!r}   board: {banner['board_code']!r}"
          f"   built: {banner['fw_date']}")
    print(f"  battery: {volts if volts is not None else '(no reading)'} V")

    reasons = check_identity(banner, expected)
    if reasons:
        head = "OVERRIDING" if a.force else "REFUSING TO FLASH"
        print(f"\n{head}:", file=_sys.stderr)
        for r in reasons:
            print(f"  - {r}", file=_sys.stderr)
        if not a.force:
            print("\nA wrong-board image drives the wrong servos. If you are certain, "
                  "re-run with --force.", file=_sys.stderr)
            return 2
    else:
        print("\nIdentity check PASSED.")

    if a.dry_run:
        print("\n--dry-run: stopping before any write.")
        return 0

    # -- write
    try:
        saved = None
        if not a.no_backup:
            saved = backup(port, pathlib.Path(a.backup_dir))
        else:
            print("\n--no-backup: skipping the flash dump. NVS holds factory servo "
                  "calibration that cannot be regenerated at home.")

        print(f"\nWriting {image.name} at 0x0 (never erase_flash -- that would wipe nvs)")
        _esptool("write_flash", "0x0", str(image), port=port)
    except FlashError as e:
        print(f"\nflash: {e}", file=_sys.stderr)
        return 1

    # -- verify
    print("\nRe-reading the board...")
    try:
        after, volts_after = read_board(port)
    except LinkError as e:
        print(f"  could not re-open the port ({e}). Reset the board and run "
              f"`bittle_io probe` by hand.", file=_sys.stderr)
        after, volts_after = {"raw": "", "model": None, "board_code": None, "fw_date": None}, None

    print(f"  reply  : {after['raw'] or '(nothing)'}")
    ok = True
    want_date = expected.get("fw_date")
    if want_date and after.get("fw_date") and after["fw_date"] != want_date:
        print(f"  WARNING: reports build {after['fw_date']}, expected {want_date}", file=_sys.stderr)
        ok = False
    if volts_after is None:
        print("  WARNING: no battery reading after flashing.", file=_sys.stderr)
        ok = False
    elif not (VOLTAGE_MIN <= volts_after <= VOLTAGE_MAX):
        # This is the revision check. RevB and RevDE read different analog pins, so an image
        # built for the wrong one reports a voltage that is not a battery.
        print(f"  WARNING: battery reads {volts_after} V, outside {VOLTAGE_MIN}-{VOLTAGE_MAX} V."
              f"\n  This is what a REVISION mismatch looks like: the image is built for "
              f"{expected.get('rev')} and reads the battery off that revision's pin.",
              file=_sys.stderr)
        ok = False
    else:
        print(f"  battery: {volts_after} V (plausible -- revision looks right)")

    if saved:
        print(f"\nBackup: {saved}")
        print(f"  restore with: uv run python tools/flash_firmware.py --port {port} "
              f"--restore {saved}")
    print("\nFlashed. Run the post-flash battery in docs/reflashing.md before trusting it.")
    return 0 if ok else 3


if __name__ == "__main__":
    raise SystemExit(main())
