"""Is the MJCF position actuator fast enough that the host-side ServoModel pre-filter
double-counts the servo's dynamics on hardware?

The question, concretely. The plant the policy trains against is
    F (rate-limited first-order lag, tau=33ms/137deg/s) -> M (MuJoCo kp=20 servo),
where F = _apply_actuator_dynamics stands in for the measured response of the REAL servo
(bittle_io/experiments/step.py: tau ~= 33 ms, slew <= 137 deg/s, measured from raw wire
commands). The deploy loop, however, runs F on the host and sends F's OUTPUT over the
wire -- so the physical plant becomes F -> S, where S is the real servo, i.e. the very
thing F was fitted to. If M is near-instant, sim's plant is one lag (F) while hardware's
is two (F then S ~= F) -- an extra ~33 ms stage that exists only on the robot.

This script settles the sim half of that question by measurement rather than assumption:
  1. M alone      -- ctrl step straight into the MJCF actuator, base pinned, legs free
                     (the suspended-robot condition step.py measured under).

  2. F -> M       -- the training plant: host filter at 50 Hz driving the same actuator.
  3. F -> S       -- hardware today, with S modelled as the measured tau/slew filter.
  4. S alone      -- hardware if the loop sends the raw target (the proposed fix).

  REFERENCE VALUE, and getting this wrong once already cost a bad plant fit: compare the
  composed F->M row against the STEP measurement in measurements/reference/step_summary.csv --
  joint 8, dead time 22.0 ms + tau 84.1 ms, so t63 ~= 106 ms, and 165.9 deg/s on the large
  step. NOT against the 40 ms implied by the bodyid tau of 33 ms: bodyid is a frequency
  response taken with the servo continuously driven and is documented as good for the SHAPE
  of the response rather than absolute gains. The two methods disagree (33 vs 84 ms) and the
  randomization range covers both; a point fit has to pick, and for a step response the step
  measurement is the one.

  This script also cannot see contact behaviour at all -- the base is pinned. A pair of
  (kv, armature) that matches the step response here can still ring on every footfall; check
  the walking dof_vel spectrum before believing a fit.

If (1) settles in a few ms, then (2) ~= F and (4) ~= F, so the fix makes hardware match
the training plant; (3) is the mismatch. If instead (1) is comparable to 33 ms, F->M was
already a double lag IN SIM and the whole correspondence needs rethinking -- which is why
this is measured, not argued.

Usage:  uv run python tools/step_response.py
"""

import pathlib as _pathlib
import sys as _sys
_R = _pathlib.Path(__file__).resolve().parents[1]
_sys.path[:0] = [str(_R / "sim"), str(_R / "deploy")]

import math

import mujoco
import numpy as np

from mj_vec_env import (ACTUATOR_SLEW_RAD_S_NOMINAL, ACTUATOR_TAU_S_NOMINAL, XML_PATH)

CONTROL_DT = 0.02
STEP_JOINT = "lf_knee"          # actuator name; a knee moving barely torques the base
AMPLITUDES_DEG = (8.0, 30.0)    # step.py's small (linear regime) and large (slew regime)
RECORD_S = 0.6


def filter_response(target_traj, dt, tau, slew, y0):
    """The exact discrete rate-limited first-order lag used by ServoModel and sim."""
    alpha = 1.0 - math.exp(-dt / tau)
    max_delta = slew * dt
    y, out = y0, []
    for x in target_traj:
        y += float(np.clip(alpha * (x - y), -max_delta, max_delta))
        out.append(y)
    return np.array(out)


def mujoco_response(ctrl_traj, ctrl_dt, act_name, physics_dt=0.0005):
    """Joint angle under a ctrl trajectory held piecewise-constant, base welded in place.

    physics_dt is far below the training 0.005 so a millisecond-scale settling is
    resolvable instead of aliased into one frame. The base free joint is re-pinned every
    physics step: crude, but reaction torque from one knee is negligible and this is the
    suspended-legs-free condition the hardware tau was measured under.
    """
    model = mujoco.MjModel.from_xml_path(XML_PATH)
    model.opt.timestep = physics_dt
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, model.key("home").id)
    base_qpos = data.qpos[:7].copy()
    act_id = model.actuator(act_name).id
    joint = model.actuator(act_name).trnid[0]
    qpos_adr = model.jnt_qposadr[joint]
    data.ctrl[:] = model.key("home").ctrl if model.nkey else 0.56
    # Let the stance settle before stepping so the transient is the actuator's, not the drop's.
    for _ in range(int(0.3 / physics_dt)):
        mujoco.mj_step(model, data)
        data.qpos[:7] = base_qpos
        data.qvel[:6] = 0.0
    n_sub = int(round(ctrl_dt / physics_dt))
    t, y = [], []
    for k, c in enumerate(ctrl_traj):
        data.ctrl[act_id] = c
        for s in range(n_sub):
            mujoco.mj_step(model, data)
            data.qpos[:7] = base_qpos
            data.qvel[:6] = 0.0
            t.append(k * ctrl_dt + (s + 1) * physics_dt)
            y.append(float(data.qpos[qpos_adr]))
    return np.array(t), np.array(y)


def rise_metrics(t, y, y0, y1):
    """(10-90% rise time, time to 63.2%) toward the commanded final value y1."""
    frac = (y - y0) / (y1 - y0)
    def first_cross(level):
        idx = np.argmax(frac >= level)
        return t[idx] if frac[idx] >= level else float("nan")
    return first_cross(0.9) - first_cross(0.1), first_cross(0.632)


def main():
    tau, slew = ACTUATOR_TAU_S_NOMINAL, ACTUATOR_SLEW_RAD_S_NOMINAL
    y0 = 0.56
    n_ticks = int(RECORD_S / CONTROL_DT)
    print(f"F: tau={tau*1000:.0f} ms, slew={math.degrees(slew):.0f} deg/s "
          f"(the measured real-servo response)")
    for amp_deg in AMPLITUDES_DEG:
        y1 = y0 + math.radians(amp_deg)
        step_traj = np.full(n_ticks, y1)

        # (1) M alone: ctrl step straight in, fine physics resolution.
        t_m, y_m = mujoco_response(step_traj, CONTROL_DT, STEP_JOINT)
        r_m, tau_m = rise_metrics(t_m, y_m, y0, y1)

        # (2) F -> M: the training plant.
        f_out = filter_response(step_traj, CONTROL_DT, tau, slew, y0)
        t_fm, y_fm = mujoco_response(f_out, CONTROL_DT, STEP_JOINT)
        r_fm, tau_fm = rise_metrics(t_fm, y_fm, y0, y1)

        # (3) F -> S: hardware today. S modelled as the measured filter at 1 kHz
        #     (the real servo runs continuously; only F is tied to the 50 Hz tick).
        fine = np.repeat(f_out, 20)
        s_today = filter_response(fine, CONTROL_DT / 20, tau, slew, y0)
        t_fine = (np.arange(len(fine)) + 1) * CONTROL_DT / 20
        r_fs, tau_fs = rise_metrics(t_fine, s_today, y0, y1)

        # (4) S alone: hardware sending the raw target (proposed fix).
        raw_fine = np.repeat(step_traj, 20)
        s_fix = filter_response(raw_fine, CONTROL_DT / 20, tau, slew, y0)
        r_s, tau_s = rise_metrics(t_fine, s_fix, y0, y1)

        print(f"\n=== {amp_deg:g} deg step on {STEP_JOINT} ===")
        print(f"{'plant':>34} | 10-90% rise | t(63.2%)")
        for name, r, tc in (
            ("M alone (MJCF kp=20 servo)", r_m, tau_m),
            ("F->M  (training plant)", r_fm, tau_fm),
            ("F->S  (hardware TODAY)", r_fs, tau_fs),
            ("S alone (hardware after fix)", r_s, tau_s),
        ):
            print(f"{name:>34} | {r*1000:9.1f} ms | {tc*1000:6.1f} ms")
        print(f"hardware-vs-sim settling gap: today {tau_fs*1000:.0f} ms vs {tau_fm*1000:.0f} ms; "
              f"after fix {tau_s*1000:.0f} ms vs {tau_fm*1000:.0f} ms")


if __name__ == "__main__":
    main()
