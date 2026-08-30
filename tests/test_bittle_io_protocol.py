"""Protocol parser tests. No hardware required -- every fixture is built by formatting
strings exactly the way OpenCatEsp32's snprintf does.
"""

import numpy as np
import pytest

from bittle_io import protocol as P

# print6Axis()'s two format strings, verbatim from OpenCatEsp32 src/imu.h.
FMT_ACCEL = "%s:%6.2f%6.2f%6.2f%7.1f%7.1f%7.1f"
FMT_NO_ACCEL = "%s%7.1f%7.1f%7.1f"


def imu_line(chip="ICM", accel=(0.12, -0.34, 9.81), ypr=(12.3, -4.5, 0.7)):
    return FMT_ACCEL % (chip, *accel, *ypr)


# -- IMU parsing ------------------------------------------------------------

def test_parses_padded_line():
    s = P.parse_imu(imu_line())
    assert s.chip == "ICM"
    np.testing.assert_allclose(s.accel, [0.12, -0.34, 9.81])
    assert (s.yaw, s.pitch, s.roll) == (12.3, -4.5, 0.7)


def test_parses_concatenated_accel_fields():
    """The case a .split() parser gets wrong.

    %6.2f pads to width 6, so a value needing 6+ chars gets no padding and runs together
    with its neighbour. Here the three accel fields render as one unbroken token.
    """
    line = imu_line(accel=(100.0, -100.0, 100.0))
    assert "100.00-100.00100.00" in line          # the fixture really is degenerate
    assert len(line.split()) < 7                  # .split() cannot recover 6 fields

    s = P.parse_imu(line)
    np.testing.assert_allclose(s.accel, [100.0, -100.0, 100.0])
    assert (s.yaw, s.pitch, s.roll) == (12.3, -4.5, 0.7)


def test_repeated_accel_values_do_not_confuse_the_angle_slice():
    """Regression: slicing past the accel block by searching for the matched TEXT finds
    the first occurrence when values repeat, leaking accel digits into the angle region."""
    s = P.parse_imu(imu_line(accel=(100.0, 100.0, 100.0), ypr=(1.5, 2.5, 3.5)))
    np.testing.assert_allclose(s.accel, [100.0, 100.0, 100.0])
    assert (s.yaw, s.pitch, s.roll) == (1.5, 2.5, 3.5)


def test_overflowing_accel_field():
    """%6.2f does not truncate -- a wider value simply uses more characters."""
    s = P.parse_imu(imu_line(accel=(-1234.56, 0.0, 9.81)))
    np.testing.assert_allclose(s.accel, [-1234.56, 0.0, 9.81])


def test_mpu6050_prefix():
    assert P.parse_imu(imu_line(chip="MCU")).chip == "MCU"


def test_build_without_accel_has_no_colon():
    """PRINT_ACCELERATION off emits "ICM" with no colon and only three angles."""
    line = FMT_NO_ACCEL % ("ICM", 12.3, -4.5, 0.7)
    s = P.parse_imu(line)
    assert s.accel is None
    assert (s.yaw, s.pitch, s.roll) == (12.3, -4.5, 0.7)


def test_negative_angles_round_trip():
    s = P.parse_imu(imu_line(ypr=(-179.9, -89.9, -0.1)))
    assert (s.yaw, s.pitch, s.roll) == (-179.9, -89.9, -0.1)


@pytest.mark.parametrize("bad", [
    "",
    "garbage",
    "=0\t1\t2",                                  # a `j` reply, not an IMU line
    "ICM:  0.12 -0.34",                          # truncated mid-frame
    "ICM:  0.12 -0.34  9.81   12.3",             # too few angles
])
def test_rejects_malformed(bad):
    with pytest.raises(P.ProtocolError):
        P.parse_imu(bad)


def test_looks_like_imu_discriminates():
    assert P.looks_like_imu(imu_line())
    assert not P.looks_like_imu("=0\t1\t2")


# -- joint table ------------------------------------------------------------

def test_parses_two_line_joint_reply():
    vals = [float(i * 3 - 20) for i in range(P.DOF)]
    text = "=" + "\t".join(str(i) for i in range(P.DOF)) + "\n" + ",\t".join(str(v) for v in vals) + ",\t"
    np.testing.assert_allclose(P.parse_joint_table(text), vals)


def test_header_only_reply_raises():
    """The superseded serial_bridge.py did a single _read_line() and would capture only the
    index header. That must fail loudly, not return 0..15 as if they were angles."""
    header = "=" + "\t".join(str(i) for i in range(P.DOF))
    with pytest.raises(P.ProtocolError, match="only the index header"):
        P.parse_joint_table(header)


def test_joint_reply_short_row_raises():
    with pytest.raises(P.ProtocolError):
        P.parse_joint_table("=0\t1\t2\n10,\t20,\t30,")


# -- command construction ---------------------------------------------------

def test_feedback_cmd_forms():
    assert P.feedback_cmd() == "f"
    assert P.feedback_cmd(8) == "f8"


@pytest.mark.parametrize("bad", [-1, 17, 2500, 3000, 4000])
def test_feedback_cmd_rejects_out_of_range(bad):
    """2500..4000 lands in the branch that reprograms feedbackSignal."""
    with pytest.raises(ValueError):
        P.feedback_cmd(bad)


def test_feedback_all_is_a_legal_selection():
    """16 is the firmware's "all joints" value for measureServoPin -- selecting it
    explicitly beats a bare `f`, which only sets it as a side effect of streaming."""
    assert P.feedback_cmd(P.FEEDBACK_ALL) == "f16"


def test_move_cmd():
    assert P.move_cmd([(8, 0), (9, -20.4)]) == "i8 0 9 -20"


# -- orientation ------------------------------------------------------------

def _rpy_to_matrix(roll, pitch, yaw):
    """R = Rz(yaw) @ Ry(pitch) @ Rx(roll), intrinsic xyz -- built independently of the
    closed form under test."""
    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)
    rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    return rz @ ry @ rx


def _rpy_to_quat(roll, pitch, yaw):
    """Standard intrinsic-xyz rpy -> (w, x, y, z)."""
    cr, sr = np.cos(roll / 2), np.sin(roll / 2)
    cp, sp = np.cos(pitch / 2), np.sin(pitch / 2)
    cy, sy = np.cos(yaw / 2), np.sin(yaw / 2)
    return np.array([
        cr * cp * cy + sr * sp * sy,
        sr * cp * cy - cr * sp * sy,
        cr * sp * cy + sr * cp * sy,
        cr * cp * sy - sr * sp * cy,
    ])


def test_gravity_is_down_when_level():
    np.testing.assert_allclose(P.projected_gravity_from_rpy(0.0, 0.0), [0, 0, -1], atol=1e-12)


def test_gravity_matches_independent_matrix_path():
    rng = np.random.default_rng(0)
    for roll, pitch in rng.uniform(-1.2, 1.2, size=(200, 2)):
        expected = _rpy_to_matrix(roll, pitch, 0.0).T @ np.array([0.0, 0.0, -1.0])
        np.testing.assert_allclose(P.projected_gravity_from_rpy(roll, pitch), expected, atol=1e-12)


def test_gravity_is_unit_norm_and_yaw_independent():
    rng = np.random.default_rng(1)
    for roll, pitch, yaw in rng.uniform(-np.pi / 2, np.pi / 2, size=(200, 3)):
        g = P.projected_gravity_from_rpy(roll, pitch)
        assert np.isclose(np.linalg.norm(g), 1.0)
        # yaw must not appear anywhere in the result
        full = _rpy_to_matrix(roll, pitch, yaw).T @ np.array([0.0, 0.0, -1.0])
        np.testing.assert_allclose(g, full, atol=1e-12)


def test_gravity_agrees_with_the_simulator():
    """Ties this parser to mj_vec_env.py:362, which is the definition the policy trained
    against. A mirrored gravity vector trains fine in sim and falls over on the floor."""
    from mj_vec_env import MjVecEnv

    rng = np.random.default_rng(2)
    for roll, pitch, yaw in rng.uniform(-1.0, 1.0, size=(100, 3)):
        quat = _rpy_to_quat(roll, pitch, yaw)[None, :]
        sim = MjVecEnv._inv_rotate(np.array([[0.0, 0.0, -1.0]]), quat)[0]
        np.testing.assert_allclose(P.projected_gravity_from_rpy(roll, pitch), sim, atol=1e-9)


def test_accel_path_agrees_with_rpy_path_at_rest():
    """At rest the accelerometer reads gravity in the CHIP frame, which is rotated
    180 deg about x vs the attitude frame (measured; see projected_gravity_from_accel).
    A chip-frame reading [gx, -gy, -gz]*9.81 must map back to g."""
    rng = np.random.default_rng(3)
    for roll, pitch in rng.uniform(-1.0, 1.0, size=(100, 2)):
        g = P.projected_gravity_from_rpy(roll, pitch)
        chip = np.array([g[0], -g[1], -g[2]]) * 9.81
        np.testing.assert_allclose(P.projected_gravity_from_accel(chip), g, atol=1e-12)


# Measured on the real robot: a 6-pose tilt test. Each row is
# (printed accel reading, expected body-frame gravity). These six poses are what
# validated the chip-frame mapping -- if projected_gravity_from_accel changes, they fail.
MEASURED_TILT_POSES = [
    ((-0.01, 0.04, 1.00), (0.0, 0.0, -1.0)),      # level, feet down
    ((0.98, 0.03, -0.20), (1.0, 0.0, 0.0)),       # nose down
    ((-0.99, -0.01, 0.11), (-1.0, 0.0, 0.0)),     # nose up
    ((-0.01, 1.00, -0.03), (0.0, -1.0, 0.0)),     # on LEFT side
    ((-0.01, -1.00, -0.02), (0.0, 1.0, 0.0)),     # on RIGHT side
    ((-0.04, 0.03, -1.00), (0.0, 0.0, 1.0)),      # upside down
]


@pytest.mark.parametrize("accel,expected", MEASURED_TILT_POSES)
def test_accel_gravity_matches_measured_poses(accel, expected):
    g = P.projected_gravity_from_accel(np.array(accel) * 9.81)
    assert np.linalg.norm(g - np.array(expected)) < 0.35


def test_accel_zero_norm_raises():
    with pytest.raises(P.ProtocolError):
        P.projected_gravity_from_accel([0.0, 0.0, 0.0])


# -- misc -------------------------------------------------------------------

def test_parse_voltage():
    assert P.parse_voltage("Voltage: 7.42") == 7.42
    assert P.parse_voltage("no numbers here") is None


def test_parse_version():
    assert P.parse_version("  BiBoard_V1_0   250101 \n") == "BiBoard_V1_0 250101"


def test_joint_names_match_policy_io():
    """bittle_io duplicates JOINT_NAMES so it stays importable without the training deps.
    They must not drift from the source of truth."""
    import policy_io

    from bittle_io import policy_names
    assert policy_names.JOINT_NAMES == policy_io.JOINT_NAMES


# -- realtime mode switch ---------------------------------------------------
# `XR` / `Xr` exist because the robot cannot have both behaviours at once: the firmware's
# interpolation ramp and 5 Hz IMU cap are right for preprogrammed skills and fatal for a
# 50 Hz host loop. Boot default is stock, so nothing changes until something asks.

def test_realtime_commands_are_the_documented_tokens():
    assert P.realtime_cmd(True) == "XR"
    assert P.realtime_cmd(False) == "Xr"
    assert P.REALTIME_ON == "XR" and P.REALTIME_OFF == "Xr"


def test_realtime_commands_live_in_the_extension_namespace():
    """Not a new top-level token. `X` is the sanctioned extension namespace, so this cannot
    collide with a future upstream command letter."""
    assert P.REALTIME_ON.startswith(P.T_EXTENSION)
    assert P.REALTIME_OFF.startswith(P.T_EXTENSION)


def test_realtime_ack_reads_both_directions():
    assert P.realtime_ack("realtime on") is True
    assert P.realtime_ack("realtime off") is False
    assert P.realtime_ack("  Realtime ON  \r\n") is True


def test_realtime_ack_is_none_when_the_reply_says_nothing():
    """THE case that matters: stock firmware has no such token and answers with something
    else entirely. None must not be confused with False -- one means "mode is off", the
    other means "this robot cannot switch modes", and only the second is a refusal."""
    for reply in ["", "ICM: -0.42  0.13  9.98  -42.8   -2.3    0.6", "unknown command",
                  None, "=0"]:
        assert P.realtime_ack(reply) is None, reply


def test_realtime_ack_does_not_match_a_bare_mode_word():
    assert P.realtime_ack("on") is None
    assert P.realtime_ack("realtime") is None
