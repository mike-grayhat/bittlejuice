"""Bittle X hardware I/O and characterization.

Speaks OpenCatEsp32 (BiBoard/ESP32) as carried by the `firmware/` submodule, NOT the
AVR/NyBoard OpenCat firmware from PetoiCamp's other repo -- see protocol.py for why that
distinction matters.

Two uses:

  Offline characterization
      python -m bittle_io --port /dev/cu.usbmodem5AA90270111 probe
      python -m bittle_io --port ... all
      python -m bittle_io report ../../measurements/<run-id>

  Runtime sensor layer (what deploy's control loop imports)
      from bittle_io import SerialLink, StateReader, Capabilities
"""

from .link import LinkError, SerialLink, StreamTimeout, find_port
from .probe import Capabilities
from .protocol import (ImuSample, ProtocolError, feedback_cmd, move_cmd, parse_imu,
                       parse_joint_table, projected_gravity_from_accel,
                       projected_gravity_from_rpy)
from .record import Recorder
from .state import RobotState, StateReader

__all__ = [
    "SerialLink", "LinkError", "StreamTimeout", "find_port",
    "Capabilities", "Recorder", "StateReader", "RobotState",
    "ImuSample", "ProtocolError", "parse_imu", "parse_joint_table",
    "projected_gravity_from_rpy", "projected_gravity_from_accel",
    "feedback_cmd", "move_cmd",
]
