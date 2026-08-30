"""OpenCatEsp32 serial tokens and reply parsers. No I/O -- pure functions over text.

PROTOCOL SOURCE: PetoiCamp/OpenCatEsp32 (src/OpenCat.h, src/imu.h, src/reaction.h).

PetoiCamp/OpenCat is a DIFFERENT repo for a different board -- the AVR/NyBoard firmware
(its OpenCat.h declares SOFTWARE_VERSION "N_260806", N = NyBoard). The two token tables
differ in ways that matter: the AVR's `v`/`V` gyro-print tokens do not exist on ESP32 (the
IMU lives under `g` sub-commands), and the AVR has no `f` token at all. When changing this
file, read `firmware/src/`, never the AVR tree.
"""

import re

import numpy as np

DOF = 16
LEG_JOINTS = [8, 9, 10, 11, 12, 13, 14, 15]

# -- tokens (OpenCatEsp32 src/OpenCat.h) ------------------------------------
T_QUERY = "?"
T_REST = "d"               # rest posture, shuts servos down
T_SERVO_FEEDBACK = "f"     # servo position; takes an optional joint index
T_JOINTS = "j"             # currentAng from memory -- COMMANDED, not measured
T_INDEXED_SIMULTANEOUS = "i"
T_GYRO = "g"               # bare `g` is a TOGGLE; always prefer a sub-command
T_POWER = "P"              # print battery voltage
T_SIGNAL_GEN = "o"
T_CALIBRATE = "c"          # overwrites servo offsets -- guarded in link.send()
T_RESET = "!"
T_BEEP = "b"               # `b0` mutes all sound; a bare `b` toggles it
T_EXTENSION = "X"          # module/behaviour namespace; `XR`/`Xr` are the realtime switch
MUTE = "b0"

# Voice reactions. The Grove voice module recognises speech on its own core and, on a
# match, the firmware queues a SKILL -- tQueue->addTask(token, cmd, 2500) in voice.h's
# read_voice(). That skill then drives the servos for ~2.5 s, fighting whatever the policy
# is commanding. Exactly the failure mode gyro balancing already causes, from a second
# source, and a word spoken near the robot is enough to trigger it -- so anything that
# actuates the robot turns them off for the duration.
#
# `XAd` does NOT clear itself, which is why VOICE_ON exists and why link.close() sends it.
# set_voice() in voice.h clears the firmware's enableVoiceQ flag AND forwards the token to
# the voice module over its own UART, where it sticks. So a run that ends without the
# restore leaves a robot that no longer answers its wake words -- in preprogrammed mode as
# much as realtime, long after the run is over, with nothing to say why.
VOICE_OFF = "XAd"
VOICE_ON = "XAc"
# Language tokens. These PERSIST -- set_voice() writes defaultLan to the board's config --
# so sending the wrong one permanently changes which language the robot listens in. There
# is no read-only query for the current setting; only the firmware knows it.
VOICE_LANG_EN = "XAa"
VOICE_LANG_ZH = "XAb"


def voice_rearm_sequence(language="a"):
    """Tokens that bring the voice module back after `XAd`, in order. `language` is 'a'/'b'.

    `XAc` alone does NOT do it. Measured on the robot: after a run had sent `XAd`, an `XAc`
    left the module silent for a full 180 s capture with someone speaking at it; this
    sequence and it answered on the next line.

    The Chinese hop is not a typo. The module re-arms on a language CHANGE, and an
    English-default robot only takes `XAa` after one. This mirrors voiceSyncAtStartup() as
    written in the firmware commit that shipped alongside
    firmware/docs/voice-module-fix-20260527.md (its section 7) -- note that the firmware in
    the submodule no longer does this, see that file's history.
    """
    seq = [VOICE_ON, VOICE_LANG_ZH]
    if language == "a":
        seq.append(VOICE_LANG_EN)
    return seq

# `g` sub-commands. Uppercase enables, lowercase disables (reaction.h dispatches
# character-by-character over the argument).
G_PRINT_STREAM = "gP"      # print IMU continuously
G_PRINT_ONCE = "gp"        # print IMU once, then stop
G_BALANCE_ON = "gB"
G_BALANCE_OFF = "gb"       # DETERMINISTIC off -- see note in imu_preamble()
G_UPDATE_ON = "gU"
G_UPDATE_OFF = "gu"
G_CALIBRATE = "gc"

# `f<n>` SELECTS which servo to report (firmware's `measureServoPin`), it does not itself
# request one reading -- and the selection persists until the next non-`f` token, which
# resets it to -1 (reaction.h). 16 means "all joints".
#
# `fp` / `fP` are the one-shot / continuous forms, mirroring `gp` / `gP` on the gyro token
# (lowercase = once and stop, uppercase = stream). `fp` is idempotent: it prints exactly
# one frame for the current selection and leaves streaming OFF, which makes
# select-then-`fp` a genuine request/response getter rather than a stream to wrestle with.
FEEDBACK_ALL = 16
F_PRINT_ONCE = "fp"
F_PRINT_STREAM = "fP"

# In the same handler an argument in 2500..4000 is interpreted as a *feedback PWM signal*
# value and reprograms `feedbackSignal` instead of selecting a servo, so every index is
# bounds-checked before it can reach the wire.
FEEDBACK_SIGNAL_BAND = (2500, 4000)


class ProtocolError(ValueError):
    """A reply did not match the format the firmware is documented to emit."""


def feedback_cmd(joint=None):
    """`f` for all joints, or `f<idx>` to select one. Rejects the feedbackSignal band.

    `joint=FEEDBACK_ALL` (16) selects all joints explicitly, which is preferable to a bare
    `f`: the bare form only sets the selection as a side effect of starting a stream.
    """
    if joint is None:
        return T_SERVO_FEEDBACK
    if not isinstance(joint, (int, np.integer)) or not 0 <= int(joint) <= FEEDBACK_ALL:
        raise ValueError(
            f"feedback joint index must be 0..{FEEDBACK_ALL} ({FEEDBACK_ALL} = all), got "
            f"{joint!r}. Values in {FEEDBACK_SIGNAL_BAND[0]}..{FEEDBACK_SIGNAL_BAND[1]} "
            "would silently reprogram the servo feedback PWM signal instead of selecting "
            "a joint."
        )
    return f"{T_SERVO_FEEDBACK}{int(joint)}"


def move_cmd(pairs):
    """`i idx deg idx deg ...` -- simultaneous indexed move."""
    return T_INDEXED_SIMULTANEOUS + " ".join(f"{int(i)} {int(round(d))}" for i, d in pairs)


# -- realtime mode ----------------------------------------------------------
# `XR` / `Xr`, added by the patched firmware. The robot has two behaviours that
# cannot both be true at once, so the patch makes them a mode rather than a permanent edit:
#
#   OFF -- "preprogrammed mode", the boot default: `i` pose commands are shaped by the
#       firmware's raised-cosine ramp, and the gP IMU stream is limited to 5 Hz. Skills,
#       gaits and the Petoi app work exactly as upstream intends.
#   ON  : `i` writes its targets once and returns, and gP runs at the link's ceiling
#       (~250 Hz). This is what a 50 Hz host control loop needs -- the host IS the
#       trajectory, so the firmware ramp is redundant smoothing that blocks 16-24 ms per
#       tick against a 20 ms period, and a 5 Hz attitude stream is stale on arrival.
#
# The firmware replies "realtime on" / "realtime off"; realtime_ack() recognises both.
# Stock firmware answers `XR` by trying to activate a hardware module called R, which
# deactivates the others -- so never send this to an unpatched robot. probe reports
# whether the patch is present.
REALTIME_ON = "XR"
REALTIME_OFF = "Xr"
_REALTIME_ACK_RE = re.compile(r"realtime\s+(on|off)", re.I)


def realtime_cmd(on):
    return REALTIME_ON if on else REALTIME_OFF


def realtime_ack(text):
    """True/False for an acknowledged mode, None if the reply says nothing about it."""
    m = _REALTIME_ACK_RE.search(text or "")
    return None if m is None else m.group(1).lower() == "on"


# -- IMU --------------------------------------------------------------------
# print6Axis() emits ONE line, formatted by snprintf with NO separators:
#
#     with accel:     "ICM:%6.2f%6.2f%6.2f%7.1f%7.1f%7.1f"   accel x,y,z then yaw,pitch,roll
#     without accel:  "ICM%7.1f%7.1f%7.1f"                   yaw,pitch,roll  (note: no colon)
#
# The prefix is "ICM"/"MCU" depending on the IMU chip (ICM42670 vs MPU6050), and the colon
# is present only on the accel-bearing variant -- which is how we tell them apart.
#
# WHY NOT .split(): %6.2f pads to width 6, but a value needing 6+ characters ("-12.34",
# "100.00", "1234.56") gets no padding at all and runs together with its neighbour.
# "100.00-100.00100.00" is three accel values that .split() returns as ONE token. The
# angles are safe (%7.1f can't overflow for values under 100000) but the accels are not.
#
# Instead we match on the EXACT decimal count, which is what makes adjacent fields
# separable: accel is always 2 decimals, angles always 1.
_ACCEL_RE = re.compile(r"-?\d+\.\d{2}")
_ANGLE_RE = re.compile(r"-?\d+\.\d{1}(?!\d)")
_IMU_PREFIX_RE = re.compile(r"^\s*(ICM|MCU)(:?)")


class ImuSample:
    """One print6Axis() line. `accel` is None when the build omits PRINT_ACCELERATION."""

    __slots__ = ("chip", "accel", "yaw", "pitch", "roll", "raw")

    def __init__(self, chip, accel, yaw, pitch, roll, raw):
        self.chip, self.accel, self.raw = chip, accel, raw
        self.yaw, self.pitch, self.roll = yaw, pitch, roll

    @property
    def ypr_deg(self):
        return np.array([self.yaw, self.pitch, self.roll], dtype=float)

    def __repr__(self):
        a = "None" if self.accel is None else np.array2string(self.accel, precision=2)
        return (f"ImuSample({self.chip}, accel={a}, "
                f"yaw={self.yaw:.1f}, pitch={self.pitch:.1f}, roll={self.roll:.1f})")


def parse_imu(line):
    """Parse one print6Axis() line. Raises ProtocolError on anything unexpected."""
    m = _IMU_PREFIX_RE.match(line)
    if not m:
        raise ProtocolError(f"no ICM:/MCU: IMU prefix in {line!r}")
    chip, has_accel = m.group(1), bool(m.group(2))
    body = line[m.end():]

    if has_accel:
        hits = list(_ACCEL_RE.finditer(body))
        if len(hits) < 3:
            raise ProtocolError(f"expected 3 accel fields (2 decimals) in {line!r}, "
                                f"found {len(hits)}")
        accel = np.array([float(h.group()) for h in hits[:3]], dtype=float)
        # Slice past the accel block by MATCH POSITION, not by searching for the matched
        # text: repeated values ("100.00-100.00100.00") would otherwise resolve to the
        # first occurrence and leave accel digits in the angle region.
        rest = body[hits[2].end():]
    else:
        accel, rest = None, body

    ang = _ANGLE_RE.findall(rest)
    if len(ang) < 3:
        raise ProtocolError(f"expected 3 angle fields (1 decimal) in {line!r}, found {len(ang)}")
    yaw, pitch, roll = (float(x) for x in ang[:3])
    return ImuSample(chip, accel, yaw, pitch, roll, line)


def looks_like_imu(line):
    return _IMU_PREFIX_RE.match(line) is not None


# -- orientation ------------------------------------------------------------
# The sim's ground truth is mj_vec_env.py:362 --
#     projected_gravity = _inv_rotate([0, 0, -1], base_quat)
# i.e. the world-frame down vector expressed in the body frame, and its Euler convention
# (_quat_to_euler_deg) is intrinsic xyz (roll, pitch, yaw) matching Genesis's
# quat_to_xyz(rpy=True). For R = Rz(yaw) @ Ry(pitch) @ Rx(roll), R^T @ [0,0,-1] is the
# negated third row of R, which reduces to the closed form below -- note it is independent
# of yaw, exactly as it must be.


def projected_gravity_from_rpy(roll_rad, pitch_rad):
    """Body-frame gravity direction from roll and pitch. Yaw is irrelevant.

    Unit vector, (0, 0, -1) when level. This is the preferred path over the accelerometer:
    the DMP has already fused out linear acceleration, so it stays valid mid-gait where a
    raw accel reading does not.
    """
    sr, cr = np.sin(roll_rad), np.cos(roll_rad)
    sp, cp = np.sin(pitch_rad), np.cos(pitch_rad)
    # VALIDITY DOMAIN (measured, 6-pose tilt test): correct for |roll| < 90 deg. Beyond
    # that the firmware's Euler output switches branch (it reports e.g. pitch ~180 on a
    # rolled/inverted robot) and this formula returns a mirrored vector. Walking never
    # leaves the valid region (sim terminates at 10 deg tilt); for full-orientation
    # gravity use projected_gravity_from_accel.
    return np.array([sp, -cp * sr, -cp * cr], dtype=float)


def projected_gravity_from_accel(accel):
    """Body-frame gravity direction from the printed accel vector, normalized.

    MEASURED CONVENTION (6-pose tilt test):
    the chip's accel frame is rotated 180 deg about x relative to the attitude frame,
    so gravity = normalize([ax, -ay, -az]). This mapping matched all six calibration
    poses (level, nose up/down, both sides, upside down) within 0.21 -- including the
    |roll| >= 90 poses where the rpy path's Euler branch flips. That makes this the
    trustworthy FULL-ORIENTATION gravity source; the rpy path is preferred only inside
    |roll| < 90 (walking range), where the DMP has already fused out linear acceleration.

    Only valid when the robot is not strongly accelerating.
    """
    a = np.asarray(accel, dtype=float)
    n = np.linalg.norm(a)
    if n < 1e-9:
        raise ProtocolError(f"accel vector has ~zero norm: {a}")
    return np.array([a[0], -a[1], -a[2]]) / n


# -- joint angles -----------------------------------------------------------
# `j` replies with '=' then range2String(DOF) then list2String(currentAng) (reaction.h).
# Two logical rows: an index header and the values. The exact separators are a firmware
# detail we deliberately do not hardcode -- probe.py captures the raw transcript, and this
# parser stays tolerant of tab/comma/space.
#
# NOTE: currentAng is the angle the firmware BELIEVES it commanded, read back from memory.
# It is not a measurement. Only the `f` feedback stream reads the servos.
_NUM_RE = re.compile(r"-?\d+(?:\.\d+)?")


def parse_joint_table(text, dof=DOF):
    """Extract the DOF-length angle row from a `j` reply. Returns float array (dof,)."""
    rows = []
    for raw in text.replace("=", "\n").splitlines():
        nums = _NUM_RE.findall(raw)
        if len(nums) >= dof:
            rows.append([float(x) for x in nums[:dof]])
    if not rows:
        raise ProtocolError(f"no {dof}-value row in `j` reply: {text!r}")
    # The index header is exactly 0..dof-1; the values row is whatever isn't that. If the
    # only row we found IS the header, the caller read one line where there are two --
    # the bug in the superseded serial_bridge.py's single _read_line() (deleted; `git log
    # --diff-filter=D -- deploy/serial_bridge.py` has it).
    header = [float(i) for i in range(dof)]
    values = [r for r in rows if r != header]
    if not values:
        raise ProtocolError(
            f"`j` reply contained only the index header, no values: {text!r}. "
            "The reply spans two lines -- read both."
        )
    return np.array(values[-1], dtype=float)


def parse_voltage(text):
    """Battery voltage from a `P` reply, or None if no number is present."""
    nums = _NUM_RE.findall(text)
    return float(nums[0]) if nums else None


def parse_version(text):
    """`?` reply. ESP32 builds SoftwareVersion as BOARD + "_" + DATE."""
    return " ".join(text.split()) or None
