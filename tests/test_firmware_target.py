"""What the pinned firmware tree compiles to.

This is the test that fails when the submodule pointer moves to a differently-configured
tree -- which matters because the release workflow names its asset and fills its notes from
exactly these values, and `tools/flash_firmware.py` refuses a flash based on them. A silent
change here would ship an image labelled for the wrong hardware.
"""

import pathlib
import sys

import pytest

sys.path[:0] = [str(pathlib.Path(__file__).resolve().parents[1] / "tools")]

from firmware_target import TargetError, read_target  # noqa: E402

FW = pathlib.Path(__file__).resolve().parents[1] / "firmware"
needs_submodule = pytest.mark.skipif(
    not (FW / "src" / "OpenCat.h").is_file(),
    reason="firmware submodule not checked out",
)


@needs_submodule
def test_pinned_tree_is_the_documented_target():
    """docs/firmware.md and the release notes both promise this exact combination."""
    t = read_target()
    assert t["robot"] == "BITTLE"
    assert t["model"] == "Bittle X"
    assert t["board"] == "BiBoard_V1_0"
    assert t["rev"] == "RevDE"


@needs_submodule
def test_board_code_is_read_from_the_headers_own_mapping():
    """B10 is what the robot PRINTS, and what the flash gate compares against.

    Read out of OpenCat.h's `#define BOARD` block rather than tabled in our code, so an
    upstream renumbering surfaces as a parse error instead of a wrong comparison.
    """
    assert read_target()["board_code"] == "B10"


@needs_submodule
def test_fw_date_is_six_digits():
    d = read_target()["fw_date"]
    assert len(d) == 6 and d.isdigit()


@needs_submodule
def test_commented_out_defines_are_not_active():
    """OpenCat.h ships `// #define RevB` above the live `#define RevDE`.

    A substring search would find both and pick the wrong one, so the anchored regex is
    load-bearing rather than stylistic.
    """
    header = (FW / "src" / "OpenCat.h").read_text(errors="replace")
    assert "// #define RevB" in header, "fixture assumption changed upstream"
    assert read_target()["rev"] == "RevDE"


def test_missing_tree_is_a_clear_error(tmp_path):
    with pytest.raises(TargetError, match="submodule"):
        read_target(tmp_path)


def test_ambiguous_config_is_refused(tmp_path):
    """Two robots defined at once must fail loudly rather than pick one."""
    (tmp_path / "src").mkdir()
    (tmp_path / "OpenCatEsp32.ino").write_text("#define BITTLE\n#define NYBBLE\n")
    (tmp_path / "src" / "OpenCat.h").write_text('#define RevDE\n#define DATE "260820"\n')
    with pytest.raises(TargetError, match="exactly one robot"):
        read_target(tmp_path)
