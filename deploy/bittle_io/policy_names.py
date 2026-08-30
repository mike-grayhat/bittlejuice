"""Joint names, mirrored from sim/policy_io.py.

bittle_io must remain importable on the deploy target, which has neither MuJoCo nor the
training deps that policy_io's siblings pull in. Duplicating eight strings is the lesser
evil -- the test below asserts they stay identical to the source of truth.
"""

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
