"""MJCF-joint <-> Petoi board index/sign/offset calibration table.

MEASURED on the real robot; the provenance of each field is
recorded above JOINT_CALIBRATION below. Petoi's index/sign convention is
per-board-revision and a leg re-seated one spline tooth off makes the nominal
layout a lie, so none of it is inferred -- it was read off this hardware, and must
be re-measured for a different robot. `deg_to_petoi` still raises
NotImplementedError on an unfilled entry, so a fresh robot fails loudly rather
than silently sending garbage angles to the servos.
"""

import math
from dataclasses import dataclass
from typing import Optional

# Must match sim/policy_io.py's JOINT_NAMES exactly (same order too, though this module
# doesn't depend on order -- lookups are by name).
JOINT_NAMES = [
    "left-back-shoulder-joint",
    "left-back-knee-joint",
    "left-front-shoulder-joint",
    "left-front-knee-joint",
    "right-back-shoulder-joint",
    "right-back-knee-joint",
    "right-front-shoulder-joint",
    "right-front-knee-joint",
]


@dataclass
class JointCal:
    petoi_index: Optional[int] = None
    sign: Optional[int] = None  # +1 or -1
    offset_deg: Optional[float] = None


# MEASURED on the real robot -- see measurements/.
#
# petoi_index: identified operator-guided (`bittle_io ... legmap`), one joint wiggled at a
#   time with the operator naming which leg moved. Not inferred from the nominal Petoi
#   layout: a leg re-seated one spline tooth off makes that table a lie.
#
# sign: +1 for every joint, established three independent ways --
#   (a) MJCF mirrors left/right (axis="0 0 1" vs "0 0 -1") and the firmware's
#       rotationDirection[] does the same, so one rule covers all eight;
#   (b) +0.3 rad in sim moves every foot BACKWARD (25 mm shoulders, 14 mm knees), and a
#       positive Petoi command was observed to move every joint backward too;
#   (c) forced-choice pose test: commanding all joints to +32 deg produced a normal stand
#       matching the sim home pose, -32 deg did not. The robot's own rest posture also
#       reads +30 deg on joints 8-15, against the sim home pose of +32.1 deg.
#
# offset_deg: measured offsets at commanded zero were all <=0.6 deg. Deliberately set to
#   0.0 rather than baked in: the position feedback that produced them is intermittently
#   corrupted (the read cycle detaches the servo and can physically perturb it), so a
#   sub-degree offset from that sensor is not distinguishable from its own noise. It is
#   also negligible beside the ~4 deg IMU mounting tilt the policy is trained to tolerate.
JOINT_CALIBRATION: dict[str, JointCal] = {
    "left-front-shoulder-joint":  JointCal(petoi_index=8,  sign=+1, offset_deg=0.0),
    "right-front-shoulder-joint": JointCal(petoi_index=9,  sign=+1, offset_deg=0.0),
    "right-back-shoulder-joint":  JointCal(petoi_index=10, sign=+1, offset_deg=0.0),
    "left-back-shoulder-joint":   JointCal(petoi_index=11, sign=+1, offset_deg=0.0),
    "left-front-knee-joint":      JointCal(petoi_index=12, sign=+1, offset_deg=0.0),
    "right-front-knee-joint":     JointCal(petoi_index=13, sign=+1, offset_deg=0.0),
    "right-back-knee-joint":      JointCal(petoi_index=14, sign=+1, offset_deg=0.0),
    "left-back-knee-joint":       JointCal(petoi_index=15, sign=+1, offset_deg=0.0),
}


def deg_to_petoi(joint_name: str, angle_rad: float) -> tuple[int, float]:
    """Converts a policy-frame joint angle (radians, MJCF convention) to a
    (petoi_index, angle_degrees) pair ready for SerialBridge.send_indexed_move.
    """
    cal = JOINT_CALIBRATION[joint_name]
    if cal.petoi_index is None or cal.sign is None or cal.offset_deg is None:
        raise NotImplementedError(
            f"JOINT_CALIBRATION['{joint_name}'] is not filled in yet -- see README.md bring-up "
            "step (b) before attempting to command the real robot."
        )
    angle_deg = math.degrees(angle_rad)
    return cal.petoi_index, cal.sign * angle_deg + cal.offset_deg
