"""Exports an mj_train.py (rsl_rl/MuJoCo) checkpoint to a dependency-free numpy .npz, and
verifies the numpy forward pass reproduces the torch actor exactly.

The verification is the point. deploy runs a hand-written numpy MLP; if it silently
disagrees with the network that was trained, the robot executes a different policy than
the one the reward curves describe, and nothing downstream would reveal it.

It verifies against the torch state dict directly rather than instantiating a simulator,
so this runs anywhere. Paths are relative to the repo root, like every other entrypoint.

Usage:
    uv run python sim/export_policy_mj.py -e my-run --ckpt 14999 \
        --out deploy/policy/bittle_policy.npz
"""

import argparse
import os
import pickle

import numpy as np
import torch

import policy_io as pio


def extract_mlp_layers(actor_state_dict):
    """(kernel, bias) pairs in forward order from rsl_rl's actor.

    Keys look like "mlp.0.weight"/"mlp.0.bias", "mlp.2.weight", ... -- activations occupy
    the odd indices and carry no parameters. Sorting the numeric indices reconstructs the
    chain without hardcoding depth, so a re-tuned ACTOR_HIDDEN_DIMS still exports.
    Torch stores Linear weights as (out, in); numpy's `x @ W` wants (in, out), hence .T.
    """
    idxs = sorted({int(k.split(".")[1]) for k in actor_state_dict
                   if k.startswith("mlp.") and k.endswith(".weight")})
    layers = []
    for i in idxs:
        layers.append((actor_state_dict[f"mlp.{i}.weight"].detach().cpu().numpy().T,
                       actor_state_dict[f"mlp.{i}.bias"].detach().cpu().numpy()))
    return layers


def _as_torch_mlp(state_dict):
    """Rebuild an rsl_rl MLP chain in torch, for verifying the numpy port against it."""
    idxs = sorted({int(k.split(".")[1]) for k in state_dict
                   if k.startswith("mlp.") and k.endswith(".weight")})
    mods = []
    for n, i in enumerate(idxs):
        w, b = state_dict[f"mlp.{i}.weight"], state_dict[f"mlp.{i}.bias"]
        lin = torch.nn.Linear(w.shape[1], w.shape[0])
        lin.weight.data, lin.bias.data = w, b
        mods.append(lin)
        if n < len(idxs) - 1:
            mods.append(torch.nn.ELU())
    return torch.nn.Sequential(*mods).eval()


def numpy_forward(layers, obs):
    """Must match deploy/policy_numpy.py exactly: ELU between layers, none on output."""
    x = obs
    for i, (w, b) in enumerate(layers):
        x = x @ w + b
        if i < len(layers) - 1:
            # Clamp the dead branch; see policy_numpy.elu for why. Must stay identical.
            x = np.where(x > 0, x, np.expm1(np.minimum(x, 0.0)))
    return x


def main():
    p = argparse.ArgumentParser()
    p.add_argument("-e", "--exp-name", required=True)
    p.add_argument("--ckpt", type=int, required=True)
    p.add_argument("--out", default="deploy/policy/bittle_policy.npz")
    p.add_argument("--tol", type=float, default=1e-5,
                   help="max |numpy - torch| action difference accepted")
    args = p.parse_args()

    log_dir = f"logs/{args.exp_name}"
    with open(f"{log_dir}/cfgs.pkl", "rb") as f:
        env_cfg, obs_cfg, reward_cfg, command_cfg, train_cfg = pickle.load(f)

    ckpt = torch.load(f"{log_dir}/model_{args.ckpt}.pt", map_location="cpu",
                      weights_only=False)
    # rsl_rl stores the actor under its own key, with plain "mlp.N.weight" names.
    actor_sd = ckpt["actor_state_dict"]
    layers = extract_mlp_layers(actor_sd)

    # deploy's obs_builder ALWAYS reports the servo-model estimate for dof_pos/dof_vel -- the
    # robot cannot read its joints at all. A checkpoint trained observing the simulator's true
    # joint state is therefore deployed against a different observation than it learned on, which
    # measured out at 1.17 -> 4.65 Hz of gait cadence and +0.079 -> -0.020 m/s. Export it anyway,
    # because comparing the two on hardware is a legitimate experiment, but never silently.
    if not env_cfg.get("observe_servo_estimate", False):
        print("WARNING: this checkpoint observed the simulator's TRUE dof_pos/dof_vel, which the\n"
              "         robot cannot measure. deploy will feed it the servo-model estimate\n"
              "         instead -- expect a much faster gait and little forward progress.\n"
              "         Retrain without --observe-true-dof for a matched policy.\n")

    # A recurrent actor's weights are not a plain MLP chain; deploy has no GRU.
    if any(k.startswith("rnn.") for k in actor_sd):
        raise SystemExit(
            "this checkpoint has a recurrent actor (rnn.* weights), which policy_numpy.py "
            "cannot evaluate -- it implements a feedforward MLP only. Either train with "
            "--obs-history instead, or add a numpy GRU to deploy/policy_numpy.py and "
            "extend this exporter to carry W_ih/W_hh/b_ih/b_hh plus the hidden state."
        )

    shapes = [w.shape for w, _ in layers]
    print(f"actor MLP: {' -> '.join(str(s[0]) for s in shapes)} -> {shapes[-1][1]}")
    # A policy trained with --obs-history reads the current frame plus N past ones, so its
    # input is a MULTIPLE of the base observation. Infer N from the weights rather than
    # trusting env_cfg, then cross-check: the exported net is ground truth for what the
    # deploy loop must build.
    history_len = env_cfg.get("obs_history_len", 0)
    gait_phase = bool(env_cfg.get("gait_phase", False))
    base_obs = pio.NUM_OBS_WITH_GAIT if gait_phase else pio.NUM_OBS
    expected = base_obs * (1 + history_len)

    # -- state estimator, if this checkpoint has one ------------------------
    # A policy trained with --estimator reads [obs, history, v_hat], where v_hat is produced by
    # a second network from [obs, history]. Both halves must be exported: an actor fed a zero
    # estimate is NOT the trained policy, and it would still run.
    est_layers = []
    if "estimator_state_dict" in ckpt:
        est_layers = extract_mlp_layers(ckpt["estimator_state_dict"])
        est_shapes = [w.shape for w, _ in est_layers]
        estimator_dim = est_shapes[-1][1]
        print(f"estimator MLP: {' -> '.join(str(s[0]) for s in est_shapes)} -> {estimator_dim}")
        if est_shapes[0][0] != expected:
            raise SystemExit(
                f"estimator takes {est_shapes[0][0]} inputs but the deployable observation is "
                f"{expected}. It must read ONLY what the robot can produce."
            )
        expected += estimator_dim      # the actor's input is widened by the estimate
    else:
        estimator_dim = 0

    if shapes[0][0] != expected or shapes[-1][1] != pio.NUM_ACTIONS:
        raise SystemExit(
            f"shape mismatch: exported net takes {shapes[0][0]} obs and emits "
            f"{shapes[-1][1]} actions, but policy_io declares {base_obs} x "
            f"(1 + {history_len} history) + {estimator_dim} estimate = {expected} and "
            f"{pio.NUM_ACTIONS} actions. "
            f"The deploy-side obs builder would feed it garbage."
        )
    if history_len:
        deployable = base_obs * (1 + history_len)
        print(f"observation history: {history_len} past frames "
              f"({base_obs} current + {base_obs * history_len} history = {deployable}"
              + (f", + {estimator_dim} estimate = {expected})" if estimator_dim else ")"))

    # -- verify numpy == torch on random observations ----------------------
    torch_mlp = torch.nn.Sequential()
    idxs = sorted({int(k.split(".")[1]) for k in actor_sd
                   if k.startswith("mlp.") and k.endswith(".weight")})
    modules = []
    for n, i in enumerate(idxs):
        lin = torch.nn.Linear(actor_sd[f"mlp.{i}.weight"].shape[1],
                              actor_sd[f"mlp.{i}.weight"].shape[0])
        lin.weight.data = actor_sd[f"mlp.{i}.weight"]
        lin.bias.data = actor_sd[f"mlp.{i}.bias"]
        modules.append(lin)
        if n < len(idxs) - 1:
            modules.append(torch.nn.ELU())
    torch_mlp = torch.nn.Sequential(*modules).eval()

    rng = np.random.default_rng(0)
    if estimator_dim:
        # Drive the chain from the DEPLOYABLE observation and let the estimator produce the
        # rest, so the verification covers the composition rather than the actor in isolation.
        deployable = rng.normal(0.0, 1.0, (256, expected - estimator_dim))
        est_torch = _as_torch_mlp(ckpt["estimator_state_dict"])
        with torch.no_grad():
            est_out = est_torch.double()(torch.from_numpy(deployable)).numpy()
        obs = np.concatenate([deployable, est_out], axis=1)
    else:
        obs = rng.normal(0.0, 1.0, (256, expected))

    # Correctness check in float64. This is the one that must be tight: it asks whether
    # the numpy implementation computes the SAME FUNCTION as torch (layer order, ELU
    # placement, weight transposition). Comparing in float32 conflates that question with
    # accumulated rounding across four layers, which lands around 1e-5 and would force a
    # tolerance loose enough to hide a real discrepancy.
    with torch.no_grad():
        ref64 = torch_mlp.double()(torch.from_numpy(obs)).numpy()
    got64 = numpy_forward([(w.astype(np.float64), b.astype(np.float64))
                           for w, b in layers], obs)
    err64 = float(np.abs(ref64 - got64).max())
    print(f"numpy vs torch (float64): max|diff| = {err64:.3e}")
    if err64 > 1e-9:
        raise SystemExit(f"VERIFICATION FAILED: implementations differ ({err64:.3e})")

    # float32 difference is informational: the practical deployment error, reported in
    # physical units so it can be judged against something real.
    with torch.no_grad():
        ref32 = torch_mlp.float()(torch.from_numpy(obs.astype(np.float32))).numpy()
    err32 = float(np.abs(ref32 - got64).max())
    rad = err32 * float(env_cfg["action_scale"])
    print(f"float32 rounding:         max|diff| = {err32:.3e} "
          f"-> {np.degrees(rad):.2e} deg of joint target")
    print("verification PASSED")

    save = {f"W{i}": w for i, (w, _) in enumerate(layers)}
    save.update({f"b{i}": b for i, (_, b) in enumerate(layers)})
    save["num_layers"] = np.array(len(layers))
    save["default_dof_pos"] = np.array(pio.DEFAULT_DOF_POS, dtype=np.float64)
    save["action_scale"] = np.array(env_cfg["action_scale"])
    save["control_dt"] = np.array(pio.CONTROL_DT)
    save["obs_scale_ang_vel"] = np.array(obs_cfg["obs_scales"]["ang_vel"])
    save["obs_scale_dof_pos"] = np.array(obs_cfg["obs_scales"]["dof_pos"])
    save["obs_scale_dof_vel"] = np.array(obs_cfg["obs_scales"]["dof_vel"])
    save["commands_scale"] = np.array(pio.COMMANDS_SCALE, dtype=np.float64)
    save["joint_names"] = np.array(pio.JOINT_NAMES)
    # Carried so the deploy loop's OBSERVER reproduces the sim's -- these parameterize
    # obs_builder.ServoModel, whose state is the dof_pos/dof_vel the policy reads, because
    # the robot cannot measure its joints. They are NOT applied to the outgoing command:
    # the wire carries the raw target and the physical servo performs what this filter
    # models. Sending the filtered value instead made the real servo apply these dynamics
    # a second time, on top of the host's copy of them.
    # Measured on this robot (see measurements/). tau from `bodyid` -- the IMU-based
    # frequency response with the servo continuously driven, i.e. the regime a walking
    # policy lives in. slew is the MEDIAN across the eight joints (130-145 deg/s), not the
    # 109.7 minimum: that outlier came from the polling-perturbed sweep, and this value is
    # an estimate of what the servo DOES, not a safety limit -- the servo self-limits
    # physically, so a pessimistic model just under-drives all eight joints.
    save["actuator_tau_s"] = np.array(0.033)
    save["actuator_slew_rad_s"] = np.array(np.radians(137.0))
    save["clip_actions"] = np.array(env_cfg.get("clip_actions", 100.0))
    # Read by policy_numpy so the deploy loop builds an observation of the right width.
    save["obs_history_len"] = np.array(history_len)
    # deploy/ must know whether to append the three gait channels AND run a clock. Without
    # this the loop would build a 33-wide observation for a 36-wide first layer and fail --
    # loudly, which is fine, but it would fail on the robot rather than at export.
    save["gait_phase"] = np.array(gait_phase)
    save["gait_hz_range"] = np.array(pio.GAIT_HZ_RANGE, dtype=np.float64)
    # The COMMAND RANGES this policy was trained over. Deploy needs them to keep its
    # startup ramp inside the trained distribution: the ramp used to sweep the forward
    # command from 0, and for a command-responsive policy 0 is off the bottom of the range.
    # Measured on a policy trained over 0.05-0.15 m/s: at command 0.00 its excess joint
    # reversals read 1.43x against 0.47x at 0.02 -- three times worse. A policy that ignores
    # the command channel shows nothing here, which is exactly why the hazard is easy to miss.
    save["command_lin_vel_x_range"] = np.array(
        command_cfg["lin_vel_x_range"], dtype=np.float64)
    save["command_lin_vel_y_range"] = np.array(
        command_cfg["lin_vel_y_range"], dtype=np.float64)
    save["command_ang_vel_range"] = np.array(
        command_cfg["ang_vel_range"], dtype=np.float64)
    # Estimator weights under their own prefix, plus the width so policy_numpy knows whether
    # to run it at all. Absent for policies trained without an estimator.
    save["estimator_dim"] = np.array(estimator_dim)
    if estimator_dim:
        save.update({f"E_W{i}": w for i, (w, _) in enumerate(est_layers)})
        save.update({f"E_b{i}": b for i, (_, b) in enumerate(est_layers)})
        save["estimator_num_layers"] = np.array(len(est_layers))

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    np.savez(args.out, **save)
    print(f"\nwrote {args.out}")
    print(f"  action_scale {float(save['action_scale'])}  "
          f"control_dt {float(save['control_dt'])}  "
          f"default_dof_pos {save['default_dof_pos'][0]:.3f} rad")


if __name__ == "__main__":
    main()
