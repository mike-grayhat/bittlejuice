"""Operator-guided identification: which Petoi joint index is which MJCF joint.

This is the sim<->hardware contract, and it is deliberately NOT inferred. `jointmap`
establishes that each index responds and with what feedback sign; it cannot know which
physical leg moved. Getting this wrong produces a robot that walks mirrored or with
swapped legs -- trains perfectly in sim, useless on the floor.

Commands one joint at a time with a large, obvious motion and asks you to name it. Writes
the `urdf_joint` and `observed_direction` columns of joint_map.csv.

Robot SUSPENDED, legs free, where you can see all four legs.
"""

import csv
from pathlib import Path

NAME = "legmap"
HELP = "operator-guided: map Petoi joint index -> MJCF joint name (interactive)"


def add_args(p):
    p.add_argument("--amp", type=float, default=35.0,
                   help="deg; large on purpose so the moving joint is unmistakable")
    p.add_argument("--repeats", type=int, default=3, help="wiggles per joint")
    p.add_argument("--settle", type=float, default=0.45)
    p.add_argument("--joints", type=int, nargs="*", default=None,
                   help="re-run only these indices; results merge into the existing "
                        "joint_map.csv, leaving other rows untouched")


def run(link, caps, args, recorder):
    from .. import policy_names as names
    from .. import protocol as P

    joint_names = names.JOINT_NAMES
    link.gyro_balance_off()
    print("\n[legmap] Robot SUSPENDED with all four legs visible.")
    print("Each joint will wiggle several times; say which one it is.\n")
    for i, n in enumerate(joint_names, 1):
        print(f"   {i}. {n}")
    print("   s. skip / cannot tell\n")

    rows = []
    with recorder.bracket_voltage(NAME, link):
        link.wake()
        for idx in (args.joints or P.LEG_JOINTS):
            # Wiggle repeatedly so the operator can look away and back without missing it.
            for _ in range(args.repeats):
                link.send(P.move_cmd([(idx, args.amp)]))
                link.sleep(args.settle)
                link.send(P.move_cmd([(idx, -args.amp)]))
                link.sleep(args.settle)
            link.send(P.move_cmd([(idx, 0)]))
            link.sleep(args.settle)

            choice = link.prompt(f"  joint {idx} -> which one? (1-8, s, or r to repeat) ")
            while choice.strip().lower() == "r":
                for _ in range(args.repeats):
                    link.send(P.move_cmd([(idx, args.amp)]))
                    link.sleep(args.settle)
                    link.send(P.move_cmd([(idx, -args.amp)]))
                    link.sleep(args.settle)
                link.send(P.move_cmd([(idx, 0)]))
                link.sleep(args.settle)
                choice = link.prompt(f"  joint {idx} -> which one? (1-8, s, r) ")

            name = ""
            if choice.strip().isdigit() and 1 <= int(choice) <= len(joint_names):
                name = joint_names[int(choice) - 1]
                # Direction is the other half of the contract: which way does a POSITIVE
                # Petoi command move the joint? Recorded in plain language and converted
                # against the MJCF axis by hand -- a sign nobody has looked at is the
                # classic mirrored-robot bug.
                d = link.prompt(f"    with +{args.amp:g} deg, did it move FORWARD/UP (f) "
                                f"or BACKWARD/DOWN (b)? ")
                direction = {"f": "forward_up", "b": "backward_down"}.get(
                    d.strip().lower(), "unknown")
                print(f"    -> {name}  ({direction})")
            else:
                direction = ""
                print("    -> skipped")
            rows.append({"opencat_index": idx, "urdf_joint": name,
                         "observed_direction": direction})

    merged = _merge_into_joint_map(recorder, rows)

    # Validate the WHOLE map, not just this run's rows -- a partial re-run must still be
    # judged against the full contract. It is a bijection or it is unusable: eight indices,
    # eight names, no repeats, none missing.
    assigned = [r.get("urdf_joint", "") for r in merged if r.get("urdf_joint")]
    dupes = sorted({n for n in assigned if assigned.count(n) > 1})
    missing = [n for n in joint_names if n not in assigned]
    print()
    if dupes:
        for n in dupes:
            idxs = [r["opencat_index"] for r in merged if r.get("urdf_joint") == n]
            print(f"! DUPLICATE: indices {idxs} both mapped to {n}")
    if missing:
        print(f"! UNMAPPED joints: {missing}")
    if dupes or missing:
        print("\nMap is NOT usable yet. Re-run the affected indices, e.g.:")
        bad = sorted({int(r["opencat_index"]) for r in merged
                      if r.get("urdf_joint") in dupes or not r.get("urdf_joint")})
        print(f"  python -m bittle_io --port $PORT --run-id {recorder.run_id} "
              f"legmap --joints {' '.join(str(i) for i in bad)}")
    else:
        print(f"All {len(joint_names)} joints mapped, one-to-one. Map is usable.")
        dirs = {r.get("observed_direction") for r in merged if r.get("observed_direction")}
        if len(dirs) > 1:
            print(f"! directions are not uniform across joints: {sorted(dirs)}")
            print("  Expected uniform if the firmware's rotationDirection[] normalizes "
                  "left/right.\n  Re-check the odd ones before trusting the signs.")
    print("\nNext: fill deploy/joint_calibration.py from joint_map.csv.")

    res = {"identified": len(assigned), "duplicates": dupes, "missing": missing,
           "usable": not (dupes or missing)}
    recorder.note(NAME, **res)
    return res


def _merge_into_joint_map(recorder, rows):
    """Add urdf_joint/observed_direction to an existing joint_map.csv, or write a new one."""
    path = Path(recorder.dir) / "joint_map.csv"
    by_idx = {str(r["opencat_index"]): r for r in rows}
    if path.exists():
        existing = list(csv.DictReader(path.open()))
        for e in existing:
            r = by_idx.get(str(e.get("opencat_index")))
            if r:
                e["urdf_joint"] = r["urdf_joint"]
                e["observed_direction"] = r["observed_direction"]
        fields = list(existing[0].keys())
        for extra in ("urdf_joint", "observed_direction"):
            if extra not in fields:
                fields.append(extra)
        merged = existing
    else:
        merged, fields = rows, ["opencat_index", "urdf_joint", "observed_direction"]
    recorder.write_csv("joint_map.csv", merged, fieldnames=fields)
    return merged
