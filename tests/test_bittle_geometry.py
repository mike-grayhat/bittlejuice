"""Geometry invariants for bittle.xml.

Every check here corresponds to a defect that was actually shipped and trained against, and that
passed the eyeball test at the time. Run after any edit to bittle.xml or assets/:

    uv run python tests/test_bittle_geometry.py
"""

import itertools
import sys

import mujoco
import numpy as np

import policy_io as pio

XML = pio.XML_PATH
MIRROR = np.array([-1.0, 1.0, 1.0])  # reflect across the x=0 plane

failures = []


def check(name, ok, detail=""):
    print(f"  {'PASS' if ok else 'FAIL'}  {name}{'  ' + detail if detail else ''}")
    if not ok:
        failures.append(name)


def settled(model):
    d = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, d, 0)
    d.ctrl[:] = model.key_ctrl[0]
    for _ in range(8000):
        mujoco.mj_step(model, d)
    return d


def foot_bottom(model, data, name):
    gid = model.geom(name).id
    return data.geom_xpos[gid][2] - model.geom_size[gid][0]


def main():
    m = mujoco.MjModel.from_xml_path(XML)
    print(f"bittle.xml: nq={m.nq} nv={m.nv} ngeom={m.ngeom}\n")

    # 1. Only the four toe spheres collide. The file once had 164 candidate pairs, 108 of them
    #    mesh-vs-mesh self-collision, while its comments claimed self-collision was disabled.
    pairs = [
        (a, b) for a, b in itertools.combinations(range(m.ngeom), 2)
        if m.body_weldid[m.geom_bodyid[a]] != m.body_weldid[m.geom_bodyid[b]]
        and m.body_parentid[m.body_weldid[m.geom_bodyid[a]]] != m.body_weldid[m.geom_bodyid[b]]
        and m.body_parentid[m.body_weldid[m.geom_bodyid[b]]] != m.body_weldid[m.geom_bodyid[a]]
        and ((m.geom_contype[a] & m.geom_conaffinity[b]) or (m.geom_contype[b] & m.geom_conaffinity[a]))
    ]
    check("only 4 collision pairs", len(pairs) == 4, f"got {len(pairs)}")

    d = settled(m)

    # 2. The feet, and nothing else, touch the ground.
    touching = {mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, d.contact[i].geom2) for i in range(d.ncon)}
    check("all contacts are feet", touching <= set(pio.FOOT_GEOM_NAMES), f"{sorted(touching)}")

    # 3. No visual mesh is underground. This is the check whose absence let a revision ship with
    #    the foot spheres 4cm up the shin: contact counts and stand height both looked right
    #    while the knee meshes hung 41mm below the floor and the robot walked on its knees.
    worst_z, worst_g = 0.0, None
    for i in range(m.ngeom):
        if m.geom_type[i] != mujoco.mjtGeom.mjGEOM_MESH:
            continue
        mid = int(m.geom_dataid[i])
        va, vn = int(m.mesh_vertadr[mid]), int(m.mesh_vertnum[mid])
        v = np.array(m.mesh_vert[va:va + vn]).reshape(-1, 3)
        z = (v @ np.array(d.geom_xmat[i]).reshape(3, 3).T + np.array(d.geom_xpos[i]))[:, 2].min()
        if z < worst_z:
            worst_z, worst_g = z, mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, i)
    check("no visual mesh below the floor", worst_z > -0.001,
          f"lowest {worst_z * 1000:+.2f} mm ({worst_g})")

    # 4. Stand height agrees with every constant derived from it. base_height's reward scale is
    #    -50, so a stale target silently dominates the return.
    check("settled height matches BASE_HEIGHT_TARGET", abs(d.qpos[2] - pio.BASE_HEIGHT_TARGET) < 0.001,
          f"settled {d.qpos[2]:.5f} vs {pio.BASE_HEIGHT_TARGET}")
    check("BASE_INIT_POS agrees with BASE_HEIGHT_TARGET", abs(pio.BASE_INIT_POS[2] - pio.BASE_HEIGHT_TARGET) < 1e-9)

    # 5. The stand pose is holdable at the real servo torque limit.
    check("servos hold the stand pose", np.abs(d.actuator_force).max() < 0.5 * 0.18,
          f"max {np.abs(d.actuator_force).max():.4f} of 0.18 N*m")

    # 6. Left/right symmetry. Was violated since bittle_scaled.xml: right toes stood 11mm wider
    #    and 2mm shorter, because neither the transforms nor the meshes were mirrored.
    d2 = mujoco.MjData(m)
    rng = np.random.default_rng(0)
    err = np.zeros(3)
    for _ in range(200):
        q = rng.uniform(-0.6, 1.2, 8)
        q[4:] = q[:4]  # mirrored joint command
        mujoco.mj_resetDataKeyframe(m, d2, 0)
        d2.qpos[7:] = q
        mujoco.mj_forward(m, d2)
        p = {n: d2.geom_xpos[m.geom(n).id].copy() for n in pio.FOOT_GEOM_NAMES}
        for r, l in [("right_back_foot", "left_back_foot"), ("right_front_foot", "left_front_foot")]:
            err = np.maximum(err, np.abs(p[r] - p[l] * MIRROR))
    check("toes mirror across x=0", err.max() < 1e-6, f"max {err.max() * 1e6:.2f} um over 200 poses")

    for l, r in [("left-back-shoulder-link", "right-back-shoulder-link"),
                 ("left-back-knee-link", "right-back-knee-link"),
                 ("left-front-shoulder-link", "right-front-shoulder-link"),
                 ("left-front-knee-link", "right-front-knee-link")]:
        a, b = m.body(l), m.body(r)
        ok = (abs(a.mass[0] - b.mass[0]) < 1e-9
              and np.abs(a.ipos * MIRROR - b.ipos).max() < 1e-9
              and np.abs(a.inertia - b.inertia).max() < 1e-12)
        check(f"inertials mirror: {l.replace('left-', '')}", ok)

    # 7. The feet_air_time contact test can actually fire. It was a silent no-op in mjx_env.py
    #    for every rollout that env ever produced.
    grounded = max(d.geom_xpos[m.geom(n).id][2] for n in pio.FOOT_GEOM_NAMES)
    d3 = mujoco.MjData(m)
    mujoco.mj_resetDataKeyframe(m, d3, 0)
    d3.qpos[7] = 1.5
    mujoco.mj_forward(m, d3)
    lifted = d3.geom_xpos[m.geom(pio.FOOT_GEOM_NAMES[0]).id][2]
    check("foot-height contact test separates stance from swing", grounded < 0.008 < lifted,
          f"grounded {grounded:.4f} < 0.008 < lifted {lifted:.4f}")

    # 8. Strict XML: Genesis parses this file with ElementTree, which rejects "--" inside comments
    #    where MuJoCo's own parser tolerates it.
    import xml.etree.ElementTree as ET
    try:
        ET.parse(XML)
        check("parses as strict XML (Genesis uses ElementTree)", True)
    except ET.ParseError as e:
        check("parses as strict XML (Genesis uses ElementTree)", False, str(e))

    print(f"\n{len(failures)} failure(s)" if failures else "\nall geometry invariants hold")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
