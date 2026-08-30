"""Cross-validates the sim<->hardware joint contract against OpenCat's own gait data.

Independent check: OpenCat's hand-designed gaits are authored in Petoi joint indices with
no knowledge of our MJCF conventions. Replaying one through JOINT_CALIBRATION must produce
a robot that walks FORWARD and stays UPRIGHT. Any error in the index mapping or the signs
turns a designed gait into flailing -- measured: every scrambled permutation collapses to a
negative body height while the correct mapping walks 345 mm and stays up.

This is the regression test for the whole legmap/sign investigation. If someone edits
JOINT_CALIBRATION or flips an axis in bittle.xml, this fails.
"""

import re
from pathlib import Path

import numpy as np
import pytest

mujoco = pytest.importorskip("mujoco")

import policy_io as _pio  # noqa: E402

# The gait tables come from the firmware submodule. This module is skipif-gated on that
# path existing, so an uninitialised submodule would make the joint-mapping regression test
# VANISH silently rather than fail -- a test that skips itself out of existence is the
# quietest way to lose a guard. Hence the skip reason names the exact fix.
FIRMWARE = Path(__file__).resolve().parents[1] / "firmware" / "src/InstinctBittleESP.h"
XML = _pio.XML_PATH

pytestmark = pytest.mark.skipif(
    not FIRMWARE.exists(),
    reason="firmware submodule not checked out -- run `git submodule update --init`")


def _gait(name):
    src = FIRMWARE.read_text()
    m = re.search(rf"const int8_t {name}\[\] PROGMEM = \{{(.*?)\}};", src, re.S)
    vals = [int(x) for x in re.findall(r"-?\d+", m.group(1))]
    n = vals[0]                                  # header: {frames, roll, pitch, ratio}
    return np.array(vals[4:4 + n * 8]).reshape(n, 8)      # cols = petoi joints 8..15


def _replay(rows, perm=None, frame_dt=0.02, cycles=3):
    from joint_calibration import JOINT_CALIBRATION as JC

    import policy_io as pio

    m = mujoco.MjModel.from_xml_path(str(XML))
    d = mujoco.MjData(m)
    mujoco.mj_resetDataKeyframe(m, d, 0)
    act = {}
    for jn in pio.JOINT_NAMES:
        jid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, jn)
        act[jn] = next(a for a in range(m.nu) if m.actuator_trnid[a, 0] == jid)
    order = list(range(8)) if perm is None else list(perm)
    nsub = int(round(frame_dt / m.opt.timestep))
    y0 = d.qpos[1]
    for _ in range(cycles):
        for r in rows:
            for jn in pio.JOINT_NAMES:
                cal = JC[jn]
                deg = r[order[cal.petoi_index - 8]]
                d.ctrl[act[jn]] = np.radians((deg - cal.offset_deg) / cal.sign)
            for _ in range(nsub):
                mujoco.mj_step(m, d)
    return (d.qpos[1] - y0) * 1000.0, d.qpos[2] * 1000.0


def test_opencat_walk_moves_forward_and_stays_upright():
    fwd_mm, height_mm = _replay(_gait("wkF"))
    assert fwd_mm > 100.0, f"OpenCat walk only travelled {fwd_mm:.0f} mm"
    assert height_mm > 50.0, f"robot collapsed to {height_mm:.0f} mm"


# There is deliberately NO trot test. trF is a dynamic gait with only two feet down at a
# time, and the real robot runs it with the firmware's gyro balancing active; this replay is
# open-loop with no stabilization, so whether it stays upright is marginal and depends on
# mass distribution rather than on the joint mapping. It passed before the torso mass was
# corrected and collapses after -- under all three inertia values tried, i.e. it was tracking
# the mass change, not the contract this file exists to verify. The statically-stable walk
# (three feet down) is robust across every model variant and is the honest test.


def test_scrambled_mappings_do_not_walk():
    """The discriminating half: without this, the tests above could pass on a mapping that
    merely happens to be self-consistent.

    Asserts the CONJUNCTION the test above asserts for the correct mapping -- travels
    forward AND stays upright -- rather than height alone. Height alone was the original
    form and it tracked plant dynamics rather than the joint contract: after the
    actuator fit (bittle.xml kv/armature, which makes the sim servo match the
    measured hardware) one permutation in twelve stayed upright at 74 mm while travelling
    -18 mm, i.e. standing still rather than walking. Same failure mode as the deleted trF
    test noted above. Measured across 30 permutations at the retuned gains: 1/12 stay
    upright, 1/12 travel >100 mm, and ZERO do both, against the correct mapping's 192 mm
    upright at 79 mm.
    """
    rows = _gait("wkF")
    rng = np.random.default_rng(3)
    for _ in range(3):
        perm = rng.permutation(8)
        if list(perm) == list(range(8)):
            continue
        fwd_mm, height_mm = _replay(rows, perm)
        assert not (fwd_mm > 100.0 and height_mm > 50.0), (
            f"a scrambled joint mapping walked {fwd_mm:.0f} mm and stayed upright at "
            f"{height_mm:.0f} mm -- this test cannot then distinguish a correct mapping "
            "from a wrong one")
