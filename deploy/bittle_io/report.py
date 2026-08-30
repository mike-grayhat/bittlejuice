"""Aggregate a run directory into hardware_params.json and a what-this-unblocks checklist.

Deliberately does NOT auto-edit joint_calibration.py or bittle.xml. Those encode sign
conventions, and a sign a human has not read off and confirmed is exactly the kind of
error that trains fine in sim and falls over on the floor.
"""

import json
from pathlib import Path


def run(run_dir, recorder=None):
    run_dir = Path(run_dir)
    manifest = _load(run_dir / "manifest.json")
    caps = _load(run_dir / "capabilities.json")
    if not manifest and not caps:
        raise FileNotFoundError(f"no manifest.json or capabilities.json in {run_dir}")

    exp = manifest.get("experiments", {})
    params = {
        "run_id": manifest.get("run_id"),
        "firmware": caps.get("firmware"),
        "imu": {
            "chip": caps.get("imu_chip"),
            "has_accel": caps.get("imu_has_accel"),
            "has_raw_gyro": caps.get("has_raw_gyro"),
            "rate_hz": caps.get("imu_stream_rate_hz"),
            "pitch_stdev_deg": exp.get("imu", {}).get("static_pitch_stdev_deg"),
            "roll_stdev_deg": exp.get("imu", {}).get("static_roll_stdev_deg"),
            "pitch_bias_deg": exp.get("imu", {}).get("static_pitch_mean_deg"),
            "roll_bias_deg": exp.get("imu", {}).get("static_roll_mean_deg"),
            "gravity_convention_validated": exp.get("imu", {}).get("tilt_pass"),
            "latency_ms": exp.get("imu", {}).get("dynamics_median_latency_ms"),
            "angvel_snr": exp.get("imu", {}).get("angvel-fallback_snr"),
        },
        "feedback": {
            "all_joints_hz": caps.get("feedback_all_rate_hz"),
            "single_joint_hz": caps.get("feedback_one_rate_hz"),
            "frame_width": caps.get("frame_width"),
            "column_noise_deg": exp.get("stream", {}).get("max_column_stdev_deg"),
        },
        "control": {
            "command_rate_hz": exp.get("cmdrate", {}).get("rate_hz"),
            "overrun_pct": exp.get("cmdrate", {}).get("overrun_pct"),
            "target_hz": exp.get("cmdrate", {}).get("target_hz"),
            "action_latency_ms": exp.get("latency", {}).get("median_ms"),
            "max_slew_deg_per_s": exp.get("sweep", {}).get("max_slew_deg_per_s"),
            "actuator_tau_ms": exp.get("step", {}).get("tau_ms"),
            "actuator_f3db_hz": exp.get("step", {}).get("f3db_hz"),
            "step_slew_deg_s": exp.get("step", {}).get("slew_deg_s"),
            "bodyid_knee_hz": exp.get("bodyid", {}).get("knee_hz_min"),
            # The SLOWEST joint is what the per-tick safety limit must respect -- a limit
            # set from the fastest joint would ask the slowest one to do the impossible.
            "slowest_joint_slew_deg_per_s":
                exp.get("sweep", {}).get("slowest_joint_slew_deg_per_s"),
        },
        "battery": {
            "probe_voltage": caps.get("voltage"),
        },
    }

    out = run_dir / "hardware_params.json"
    out.write_text(json.dumps(params, indent=2) + "\n")
    print(f"Wrote {out}\n")
    _checklist(params, caps, run_dir)
    return params


def _checklist(p, caps, run_dir):
    imu, fb, ctl = p["imu"], p["feedback"], p["control"]
    print("--- what these numbers unblock ---\n")

    _item("deploy/obs_builder.py :: parse_gyro_accel",
          imu["chip"] is not None,
          f"replace with bittle_io.state.StateReader ({imu['chip']}, "
          f"accel={'yes' if imu['has_accel'] else 'no'}, {imu['rate_hz']} Hz)")

    _item("deploy/obs_builder.py :: parse_joint_angles",
          caps.get("joint_reply_parsed") is True,
          "use protocol.parse_joint_table -- and note the `j` reply spans TWO lines, "
          "which a single _read_line() does not handle -- the trap serial_bridge.py fell into")

    _item("projected_gravity sign convention",
          imu["gravity_convention_validated"] is True,
          "validated against mj_vec_env._inv_rotate by `imu --phase tilt`"
          if imu["gravity_convention_validated"] else "run `imu --phase tilt`")

    _item("base_ang_vel in the observation",
          bool(imu["angvel_snr"] and imu["rate_hz"] and imu["rate_hz"] >= 40
               and imu["angvel_snr"] >= 5),
          f"finite-differenced attitude is usable (SNR {imu['angvel_snr']}, "
          f"{imu['rate_hz']} Hz)" if imu["angvel_snr"] else
          "no raw gyro output; run `imu --phase angvel-fallback` to decide between "
          "differentiation and a firmware build that prints gy")

    _item("deploy/joint_calibration.py :: JOINT_CALIBRATION",
          (run_dir / "joint_map.csv").exists(),
          "fill from joint_map.csv -- the urdf_joint column is yours to complete by hand"
          if (run_dir / "joint_map.csv").exists() else "run `jointmap`")

    slow = ctl["slowest_joint_slew_deg_per_s"]
    # No longer names a constant: control_loop sends the raw target and keeps ServoModel as an
    # observer, so the host-side clamp this used to size is gone. Kept as a reported quantity
    # because the slowest joint still bounds what any gait can ask for.
    _item("slowest-joint slew (informational; no host-side clamp any more)",
          slow is not None,
          f"slowest joint slews {slow} deg/s -> {slow * 0.02:.1f} deg per 20 ms tick "
          f"(LOWER BOUND on patched firmware -- `sweep` aliases; see its docstring)"
          if slow else "run `sweep`")

    tau, knee = ctl["actuator_tau_ms"], ctl["bodyid_knee_hz"]
    _item("sim actuator lag model (bittle.xml / mj_vec_env)",
          tau is not None or knee is not None,
          " / ".join(x for x in (
              f"step: tau {tau} ms (f_3db {ctl['actuator_f3db_hz']} Hz)" if tau else "",
              f"bodyid: knee {knee} Hz" if knee else "") if x)
          or "run `step` (suspended) and `bodyid` (on the ground)")

    _item("policy_io.CONTROL_DT = 0.02 (50 Hz)",
          ctl["overrun_pct"] is not None and ctl["overrun_pct"] < 1.0,
          f"{ctl['command_rate_hz']} Hz with {ctl['overrun_pct']}% overruns"
          if ctl["overrun_pct"] is not None else "run `cmdrate --hz 50`")

    _item("mj_vec_env.py observation noise",
          fb["column_noise_deg"] is not None or imu["pitch_stdev_deg"] is not None,
          f"joint {fb['column_noise_deg']} deg, pitch {imu['pitch_stdev_deg']} deg")

    if imu["pitch_bias_deg"] and abs(imu["pitch_bias_deg"]) > 1.0:
        print(f"\n! A level robot reports pitch {imu['pitch_bias_deg']} deg -- that is a "
              "mounting offset.\n  Subtract it, or the policy sees a permanently tilted world.")


def _item(what, ok, detail):
    print(f"  [{'x' if ok else ' '}] {what}\n      {detail}")


def _load(path):
    try:
        return json.loads(Path(path).read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return {}
