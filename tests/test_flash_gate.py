"""The pre-flash identity gate, exercised without a robot.

The gate's whole job is to refuse. These tests are therefore mostly about the refusal
cases, and in particular about the ones where the board says something UNEXPECTED rather
than something wrong -- silence, a boot prompt, an unparseable banner. Treating any of
those as a pass would defeat the tool, because "I could not tell what this is" is exactly
the situation a wrong-board flash arrives in.
"""

import pathlib
import sys

import pytest

sys.path[:0] = [str(pathlib.Path(__file__).resolve().parents[1] / "tools")]

from flash_firmware import check_identity, parse_banner  # noqa: E402

# Real replies. The Bittle one is copied from measurements/reference/capabilities.json.
BITTLE_B10 = "G Bittle X B10_260717 ?"
NYBBLE_B01 = "G Nybble Q B01_251121 ?"
BOOT_PROMPT = "- Calibrate the Inertial Measurement Unit (IMU)? (Y/n):"

TARGET = {"model": "Bittle X", "board_code": "B10", "rev": "RevDE", "fw_date": "260820"}


# -- parsing ----------------------------------------------------------------

def test_parses_model_and_board_from_a_real_banner():
    b = parse_banner(BITTLE_B10)
    assert b["model"] == "Bittle X"
    assert b["board_code"] == "B10"
    assert b["fw_date"] == "260717"


def test_model_string_is_not_mistaken_for_a_board_code():
    """"Bittle X" contains letters; only the <letters><digits>_<6 digits> token is a board."""
    assert parse_banner(BITTLE_B10)["board_code"] == "B10"
    assert parse_banner(NYBBLE_B01)["model"] == "Nybble Q"


@pytest.mark.parametrize("text", ["", None, "   ", "random noise with no version"])
def test_unparseable_replies_yield_no_identity(text):
    b = parse_banner(text)
    assert b["model"] is None and b["board_code"] is None


# -- the gate ---------------------------------------------------------------

def test_matching_board_passes():
    assert check_identity(parse_banner(BITTLE_B10), TARGET) == []


def test_build_date_difference_alone_does_not_refuse():
    """Reflashing a robot that already runs an older build is the NORMAL case."""
    assert parse_banner(BITTLE_B10)["fw_date"] != TARGET["fw_date"]
    assert check_identity(parse_banner(BITTLE_B10), TARGET) == []


def test_wrong_robot_and_wrong_board_are_both_reported():
    """One reason per mismatch, so the operator sees everything wrong at once."""
    reasons = check_identity(parse_banner(NYBBLE_B01), TARGET)
    assert len(reasons) == 2
    assert any("Nybble Q" in r for r in reasons)
    assert any("B01" in r and "servos" in r for r in reasons)


def test_wrong_board_alone_is_refused():
    """A Bittle X on a V0.2 board: right robot, fatal board difference."""
    reasons = check_identity(parse_banner("G Bittle X B02_251121 ?"), TARGET)
    assert len(reasons) == 1 and "B02" in reasons[0]


@pytest.mark.parametrize("reply, expect", [
    ("", "did not answer"),
    (BOOT_PROMPT, "IMU calibration prompt"),
    ("G ??? ?", "could not parse"),
])
def test_unknown_identity_is_refused_not_ignored(reply, expect):
    reasons = check_identity(parse_banner(reply), TARGET)
    assert reasons and expect in reasons[0]


def test_absent_expectations_do_not_manufacture_a_pass():
    """With nothing to compare against, a parseable board is not refused --

    that is the --restore path, where the image carries no target metadata. The refusal for
    an unidentifiable board still applies, which is what keeps this from being a hole.
    """
    assert check_identity(parse_banner(BITTLE_B10), {}) == []
    assert check_identity(parse_banner(""), {}) != []
