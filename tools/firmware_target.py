"""What a firmware build is FOR: robot, board, revision, build date.

These are compile-time `#define`s and cannot adapt at runtime, so flashing an image onto a
robot it was not built for is a real hazard rather than a theoretical one -- a BiBoard V0.x
has a different `PWM_pin[]` table, so the wrong servos move, and a RevB board reads battery
voltage off a pin the RevDE build does not use.

Two consumers, one implementation, which is the point of this file:

  * `.github/workflows/firmware-release.yml` names the release asset and fills its notes
    from this, so both describe the tree that was actually compiled.
  * `tools/flash_firmware.py` compares it against what the robot reports over serial and
    refuses a mismatch.

The two fields the gate actually compares -- `model` and `board_code` -- are the ones the
firmware PRINTS (`OpenCat.h`: `printToAllPorts(MODEL)`, then `BOARD + "_" + DATE`). Every
mapping below is read out of the header rather than hardcoded here, so a `#define` that
moves upstream shows up as a parse failure instead of a wrong answer.

    uv run python tools/firmware_target.py [--tree firmware] [--github-output]
"""

import argparse
import json
import pathlib
import re
import subprocess
import sys

_R = pathlib.Path(__file__).resolve().parents[1]

# Robot selector -> the MODEL string that selector produces. The nesting inside `#elif
# defined BITTLE` (arm/VT variants) is resolved by _model_for(), not here.
_ROBOTS = ("BITTLE", "NYBBLE", "CUB", "MINI")
_BOARDS = ("BiBoard_V0_1", "BiBoard_V0_2", "BiBoard_V1_0", "BiBoard2")
_REVS = ("RevB", "RevDE")


class TargetError(RuntimeError):
    pass


def _active(name, text):
    """Is `#define <name>` live, i.e. present and not commented out?

    Anchored at line start so a commented `// #define RevB` -- which is exactly how
    OpenCat.h ships the inactive revision -- does not match.
    """
    return re.search(rf"^[ \t]*#define[ \t]+{re.escape(name)}\b", text, re.M) is not None


def _one_of(names, text, what, path):
    hits = [n for n in names if _active(n, text)]
    if len(hits) != 1:
        raise TargetError(
            f"expected exactly one {what} #define in {path}, found {hits or 'none'}. "
            "The firmware tree is configured for something this tool does not understand; "
            "fix the #define block rather than the tool."
        )
    return hits[0]


def _board_code(board, header):
    """BiBoard_V1_0 -> "B10", read from OpenCat.h's own mapping.

    This is the token the robot reports in its `?` reply, so it is the one worth getting
    from the source of truth rather than a table here that could drift.
    """
    # #ifdef BiBoard_V0_1 / #define BOARD "B01" / #elif defined BiBoard_V0_2 / ...
    block = re.search(r"#ifdef\s+BiBoard_V0_1(.*?)#endif", header, re.S)
    if not block:
        raise TargetError("could not find the BOARD mapping block in OpenCat.h")
    pairs = re.findall(r"(?:#ifdef|#elif\s+defined)\s+(\w+)[^\n]*\n\s*#define\s+BOARD\s+\"([^\"]+)\"",
                       block.group(0))
    table = dict(pairs)
    if board not in table:
        raise TargetError(f"OpenCat.h's BOARD mapping has no entry for {board}: {table}")
    return table[board]


def _model_for(robot, ino, header):
    """The MODEL string, resolving BITTLE's arm/VT sub-variants the way the header does."""
    if robot == "BITTLE":
        if _active("ROBOT_ARM", ino):
            want = "Bittle X+Arm"
        elif _active("VT", ino):
            want = "Bittle X+VT"
        else:
            want = "Bittle X"
        if f'"{want}"' not in header:
            raise TargetError(f"expected MODEL {want!r} in OpenCat.h and it is not there")
        return want
    # Non-Bittle robots define MODEL once, directly under their selector.
    m = re.search(rf"#(?:ifdef|elif\s+defined)\s+{robot}\s*\n\s*#define\s+MODEL\s+\"([^\"]+)\"",
                  header)
    if not m:
        raise TargetError(f"could not resolve MODEL for {robot} in OpenCat.h")
    return m.group(1)


def read_target(tree=None):
    """-> dict describing what the firmware tree at `tree` would compile to."""
    tree = pathlib.Path(tree or _R / "firmware")
    ino_path, hdr_path = tree / "OpenCatEsp32.ino", tree / "src" / "OpenCat.h"
    for p in (ino_path, hdr_path):
        if not p.is_file():
            raise TargetError(
                f"{p} not found. Is the submodule checked out? `git submodule update --init`"
            )
    ino, header = ino_path.read_text(errors="replace"), hdr_path.read_text(errors="replace")

    robot = _one_of(_ROBOTS, ino, "robot", ino_path)
    board = _one_of(_BOARDS, ino, "board", ino_path)
    rev = _one_of(_REVS, header, "board revision", hdr_path)

    date = re.search(r'^\s*#define\s+DATE\s+"(\d{6})"', header, re.M)
    if not date:
        raise TargetError(f"no `#define DATE \"YYMMDD\"` in {hdr_path}")

    out = {
        "robot": robot,
        "model": _model_for(robot, ino, header),
        "board": board,
        "board_code": _board_code(board, header),
        "rev": rev,
        "fw_date": date.group(1),
    }
    try:
        out["submodule"] = subprocess.run(
            ["git", "-C", str(tree), "rev-parse", "HEAD"],
            capture_output=True, text=True, check=True).stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        out["submodule"] = None      # a tarball checkout is not an error
    return out


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--tree", default=None, help="firmware tree (default: ./firmware)")
    p.add_argument("--github-output", action="store_true",
                   help="emit key=value lines for $GITHUB_OUTPUT instead of JSON")
    p.add_argument("--extra", action="append", default=[], metavar="KEY=VALUE",
                   help="additional field for the JSON output; repeatable. Used by the "
                        "release workflow to fold in md5/size/tag, which are properties of "
                        "the built artifact rather than of the source tree.")
    a = p.parse_args(argv)
    try:
        t = read_target(a.tree)
    except TargetError as e:
        print(f"firmware_target: {e}", file=sys.stderr)
        return 1
    for kv in a.extra:
        k, _, v = kv.partition("=")
        if not k or not _:
            print(f"firmware_target: --extra expects KEY=VALUE, got {kv!r}", file=sys.stderr)
            return 1
        t[k] = int(v) if v.isdigit() else v

    if a.github_output:
        for k, v in t.items():
            if v is not None:
                print(f"{k}={v}")
    else:
        print(json.dumps(t, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
