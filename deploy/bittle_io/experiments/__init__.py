"""Experiment registry. cli.py and report.py enumerate this rather than hardcoding names.

Each module exposes:
    NAME, HELP, add_args(parser), run(link, caps, args, recorder) -> dict
`run` returns the headline numbers, which the recorder folds into the run manifest.
"""

from . import (bodyid, cmdrate, friction, imu, jointmap, latency, legmap, step,
               stream, sweep)

MODULES = [stream, imu, jointmap, legmap, cmdrate, latency, step, sweep, bodyid,
           friction]
EXPERIMENTS = {m.NAME: m for m in MODULES}

# The order `all` runs in: characterize the sensors before anything actuates, and leave
# the long joint-by-joint sweeps for last (servos heat up and the battery sags).
# `bodyid` (needs the ground), `legmap` and `friction` (need an operator) are
# deliberately NOT here. Run those standalone.
DEFAULT_ORDER = ["stream", "imu", "jointmap", "cmdrate", "latency", "step", "sweep"]

__all__ = ["EXPERIMENTS", "MODULES", "DEFAULT_ORDER"]
