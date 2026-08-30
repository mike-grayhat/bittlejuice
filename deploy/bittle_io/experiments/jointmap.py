"""Per-joint feedback verification: does each OpenCat joint index respond, with which sign?

Ported from bittle_bringup2.py's cmd_map, then rebuilt on single-joint polling after the
firmware source (OpenCatEsp32 src/espServo.h, servoFeedback()) showed the all-joint frame
is NOT positionally stable: a failed read prints nothing for that joint, so columns SHIFT
within the frame and a narrower-than-expected frame is ambiguous about which joint
dropped. Single-joint replies are index-tagged ("<index>\\t<angle>"), which makes the
column-mapping problem dissolve entirely -- we ask each joint by index and verify it
moves when commanded.

The output fills deploy/joint_calibration.py's JOINT_CALIBRATION table: petoi index,
sign of feedback vs command, and the offset between commanded 0 and the feedback zero.
"""


NAME = "jointmap"
HELP = "verify each joint index responds to commands; sign + offset per joint"


def add_args(p):
    p.add_argument("--amp", type=float, default=25.0)
    p.add_argument("--settle", type=float, default=0.6)
    p.add_argument("--thresh", type=float, default=3.0)
    p.add_argument("--samples", type=int, default=5,
                   help="polled readings to median per settle")
    p.add_argument("--poll-timeout", type=float, default=0.3)
    p.add_argument("--cooldown", type=float, default=0.5,
                   help="pause between joints; servos overheat when fighting")


def run(link, caps, args, recorder):
    from .. import protocol as P

    link.gyro_balance_off()
    with recorder.bracket_voltage(NAME, link):
        link.wake()
        print(f"\nProbing joints {P.LEG_JOINTS} with +-{args.amp} deg, one at a time.")
        print("Robot SUSPENDED, legs free.\n")

        rows, missing = [], []
        for idx in P.LEG_JOINTS:
            link.send(P.move_cmd([(idx, 0)]))
            link.sleep(args.settle)
            base = link.settle_polled(idx, samples=args.samples,
                                      timeout=args.poll_timeout)
            link.send(P.move_cmd([(idx, args.amp)]))
            link.sleep(args.settle)
            moved = link.settle_polled(idx, samples=args.samples,
                                       timeout=args.poll_timeout)
            link.send(P.move_cmd([(idx, 0)]))
            link.sleep(args.settle)

            if base is None or moved is None:
                print(f"  joint {idx}: NO FEEDBACK reply")
                missing.append(idx)
                rows.append({"opencat_index": idx, "responds": "NO_REPLY",
                             "delta_deg": "", "sign_vs_command": "",
                             "offset_at_zero_deg": "", "urdf_joint": ""})
            else:
                delta = moved - base
                if abs(delta) < args.thresh:
                    print(f"  joint {idx}: feedback replies but DID NOT MOVE "
                          f"(delta {delta:+.2f} deg)")
                    missing.append(idx)
                    rows.append({"opencat_index": idx, "responds": "NO_MOTION",
                                 "delta_deg": round(delta, 2), "sign_vs_command": "",
                                 "offset_at_zero_deg": round(base, 2), "urdf_joint": ""})
                else:
                    # `base` is the feedback reading with 0 commanded: the residual
                    # offset between the firmware's zero and the servo's actual zero.
                    rows.append({"opencat_index": idx, "responds": "OK",
                                 "delta_deg": round(delta, 2),
                                 "sign_vs_command": "+1" if delta > 0 else "-1",
                                 "offset_at_zero_deg": round(base, 2),
                                 "urdf_joint": ""})
                    print(f"  joint {idx}: delta {delta:+7.2f} deg  "
                          f"sign {'+1' if delta > 0 else '-1'}  "
                          f"offset@0 {base:+.2f} deg")
            link.sleep(args.cooldown)

    recorder.write_csv("joint_map.csv", rows,
                       fieldnames=["opencat_index", "responds", "delta_deg",
                                   "sign_vs_command", "offset_at_zero_deg", "urdf_joint"])
    print("\nWrote joint_map.csv")
    if missing:
        print(f"PROBLEM joints: {missing}")
        print("NO_REPLY: servo predates the feedback batch, or is disconnected.")
        print("NO_MOTION: feedback works but the joint did not track the command.")
    print("Fill in urdf_joint by hand against policy_io.JOINT_NAMES -- that column is the")
    print("sim<->hardware contract, and it is a human's job to confirm the sign.")

    res = {"mapped": len(rows) - len(missing), "missing": missing}
    recorder.note(NAME, **res)
    return res
