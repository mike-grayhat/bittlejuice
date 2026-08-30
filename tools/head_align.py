"""Hand-tune the head's placement in bittle.xml, in degrees and millimetres.

The head geoms are cosmetic only -- mass="0" class="visualgeom", and _build_model deletes them
before training -- so nothing here can affect a policy or a run in progress. It is purely how
the robot looks in mj_eval's viewer.

WHY THIS EXISTS: the four head geoms carry a quaternion, which is not hand-tunable, and
bittle.xml compiles with angle="radian" so even `euler` would be in radians. This lets you
nudge one number at a time in degrees and see the result immediately.

WHAT THE KNOBS DO (body frame: +x is the robot's LEFT-RIGHT, +y is FORWARD, +z is UP):
    --yaw    turn the head left/right   (the "it's rotated" axis)
    --roll   tip it side to side        (the "it's tilted" axis)
    --pitch  nod it up/down
    --x/--y/--z  slide it, in mm

NOTE the two are coupled: MuJoCo rotates a geom about its OWN origin, not about the neck, so
changing --yaw also swings the head sideways and you will need to correct --x to bring it back.
That coupling is why fixing rotation and offset separately does not converge.

    # see where it stands, change nothing
    uv run python tools/head_align.py --show

    # try a tilt correction
    uv run python tools/head_align.py --roll -10
    uv run mjpython sim/mj_eval.py -e my-run --ckpt 4000

    # back to the file's original values
    uv run python tools/head_align.py --reset
"""

import pathlib as _pathlib
import sys as _sys
_R = _pathlib.Path(__file__).resolve().parents[1]
_sys.path[:0] = [str(_R / "sim"), str(_R / "deploy")]

import argparse
import re

import numpy as np

import policy_io as pio

XML = pio.XML_PATH
GEOMS = ("head_c", "jaw_c", "neck_c", "neck_servo_c")
# What bittle.xml shipped with, for --reset.
ORIGINAL = dict(yaw=90.0, roll=0.0, pitch=0.0, x=0.0, y=0.0002, z=-0.0241)


def to_quat(yaw, pitch, roll):
    """Body-frame yaw(z) then pitch(y) then roll(y-forward), as (w, x, y, z)."""
    def q(axis, deg):
        h = np.radians(deg) / 2.0
        v = np.asarray(axis, float)
        return np.array([np.cos(h), *(v * np.sin(h))])

    def mul(a, b):
        w1, x1, y1, z1 = a
        w2, x2, y2, z2 = b
        return np.array([w1*w2 - x1*x2 - y1*y2 - z1*z2, w1*x2 + x1*w2 + y1*z2 - z1*y2,
                         w1*y2 - x1*z2 + y1*w2 + z1*x2, w1*z2 + x1*y2 - y1*x2 + z1*w2])
    # roll is about the head's FORWARD axis, which after the yaw is body +y.
    return mul(q([0, 1, 0], roll), mul(q([1, 0, 0], pitch), q([0, 0, 1], yaw)))


def landmarks(path=XML):
    """Where the neck base and snout actually land, as MuJoCo draws them.

    These two are the whole test: both must sit at x = 0. Measured through geom_xmat/geom_xpos
    rather than raw mesh_vert, because mesh_vert is pre-transform and is NOT what gets placed.
    """
    import mujoco
    m = mujoco.MjModel.from_xml_path(path)
    d = mujoco.MjData(m)
    mujoco.mj_resetDataKeyframe(m, d, 0)
    mujoco.mj_forward(m, d)
    root = d.xpos[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "root")]

    def cloud(nm):
        g = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, nm)
        mid = m.geom_dataid[g]
        a, n = m.mesh_vertadr[mid], m.mesh_vertnum[mid]
        return (m.mesh_vert[a:a + n].astype(np.float64) @ d.geom_xmat[g].reshape(3, 3).T
                + d.geom_xpos[g] - root)

    head = cloud("head_c")
    return cloud("neck_c").mean(0), head[np.argmax(head[:, 1])]


def current():
    src = open(XML).read()
    qm = re.search(r'name="head_c".*?quat="([-\d.eE ]+)"', src, re.S)
    pm = re.search(r'name="head_c"[^>]*?pos="([-\d.eE ]+)"', src, re.S)
    return [float(v) for v in qm.group(1).split()], [float(v) for v in pm.group(1).split()]


def write(quat, pos):
    src = open(XML).read()
    q_old, p_old = current()
    qs_old = re.search(r'name="head_c".*?(quat="[-\d.eE ]+")', src, re.S).group(1)
    ps_old = re.search(r'name="head_c"[^>]*?(pos="[-\d.eE ]+")', src, re.S).group(1)
    src = src.replace(qs_old, f'quat="{quat[0]:.6f} {quat[1]:.6f} {quat[2]:.6f} {quat[3]:.6f}"')
    src = src.replace(ps_old, f'pos="{pos[0]:.5f} {pos[1]:.5f} {pos[2]:.5f}"')
    open(XML, "w").write(src)


def report(tag):
    neck, snout = landmarks()
    print(f"  {tag:<10} neck base x = {1000*neck[0]:+7.2f} mm    "
          f"snout x = {1000*snout[0]:+7.2f} mm    (both should be 0)")


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--yaw", type=float, help="turn left/right [deg]")
    p.add_argument("--roll", type=float, help="tilt side to side [deg]")
    p.add_argument("--pitch", type=float, help="nod up/down [deg]")
    p.add_argument("--x", type=float, help="slide left/right [mm]")
    p.add_argument("--y", type=float, help="slide FORWARD (+) / back (-) [mm]")
    p.add_argument("--z", type=float, help="slide up/down [mm]")
    p.add_argument("--show", action="store_true", help="report and change nothing")
    p.add_argument("--reset", action="store_true", help="restore bittle.xml's original values")
    a = p.parse_args()

    quat, pos = current()
    print(f"current quat {quat}  pos {pos}")
    report("now")
    if a.show:
        return
    if a.reset:
        write(to_quat(ORIGINAL["yaw"], ORIGINAL["pitch"], ORIGINAL["roll"]),
              [ORIGINAL["x"], ORIGINAL["y"], ORIGINAL["z"]])
        report("reset")
        return

    new_pos = list(pos)
    for i, v in enumerate((a.x, a.y, a.z)):
        if v is not None:
            new_pos[i] = v / 1000.0

    if a.yaw is None and a.roll is None and a.pitch is None:
        # Position-only edit: keep the existing orientation EXACTLY as it is. Recomputing it
        # from defaults would silently discard a hand-tuned quat, which is what this tool did
        # when it was first written -- passing --y alone reset the angles to 115.25/0/0.
        write(quat, new_pos)
        print(f"\nmoved to pos {[round(1000 * v, 2) for v in new_pos]} mm; orientation untouched")
    else:
        # Angles are ABSOLUTE, not deltas: pass every angle you want kept, together.
        yaw = a.yaw if a.yaw is not None else 115.25
        roll = a.roll or 0.0
        pitch = a.pitch or 0.0
        write(to_quat(yaw, pitch, roll), new_pos)
        print(f"\nwrote yaw {yaw} roll {roll} pitch {pitch}, "
              f"pos {[round(1000 * v, 2) for v in new_pos]} mm")
    report("after")


if __name__ == "__main__":
    main()
