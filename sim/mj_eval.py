"""Visual evaluation for a policy trained by mj_train.py, in MuJoCo's native viewer.

It exists because neither earlier eval script fitted this pipeline: bittle_eval.py rendered
through Genesis (a different simulator from the one the policy trained in), and mjx_eval.py
loaded a brax/JAX `final_params.pkl` rather than an rsl_rl checkpoint. Both are deleted.

This one drives the *same* MjVecEnv used for training (num_envs=1) and mirrors its physics state
into a viewer model each control step. That means the gait you watch is produced by exactly the
observation pipeline, action latency, actuator permutation and contact model the policy was
trained against -- nothing is re-implemented here and allowed to drift, which is how the old
mjx_eval.py's hand-rolled observation builder could silently disagree with its env.

The viewer loads the full bittle.xml (meshes, textures) while the env runs its stripped
physics-only model. Stripping removes geoms and assets but no bodies or joints, so nq/nv and the
mjSTATE_FULLPHYSICS layout are identical and the state copies across directly.

On macOS the interactive viewer must run under `mjpython`, not `python`: launch_passive needs the
Cocoa event loop on the process's main thread, which the normal interpreter does not give it.
`mjpython` ships with the mujoco wheel and is already in .venv/bin. Use --video instead to render
offscreen, which works under plain python (and over SSH).

Usage:
    uv run mjpython sim/mj_eval.py -e my-run --ckpt 3999
    uv run mjpython sim/mj_eval.py -e my-run --command 0.15 0 0
    uv run mjpython sim/mj_eval.py -e my-run --domain-rand      # with pushes/noise
    uv run python   sim/mj_eval.py -e my-run --video gait.mp4   # no viewer needed
"""

import argparse
import os
import pickle
import time

import mujoco
import mujoco.viewer
import numpy as np
import torch
from rsl_rl.runners import OnPolicyRunner

from ppo_estimator import EVAL_LOAD_CFG

import policy_io as pio
from mj_vec_env import XML_PATH, MjVecEnv


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--exp-name", type=str, required=True,
                        help="Experiment name; the log directory under logs/")
    parser.add_argument("--ckpt", type=int, default=None,
                        help="Checkpoint iteration to load (default: the highest available)")
    parser.add_argument("--command", type=float, nargs=3, default=None,
                        metavar=("VX", "VY", "WZ"),
                        help="Velocity command to hold. Default: the midpoint of the run's own "
                             "lin_vel_x_range, so it matches what the policy was trained on.")
    parser.add_argument("--duration", type=float, default=60.0, help="Seconds to run")
    parser.add_argument("--video", type=str, default=None, metavar="PATH",
                        help="Render offscreen to an mp4 instead of opening the viewer. Runs under "
                             "plain python (no mjpython needed) and works headless.")
    parser.add_argument("--camera-distance", type=float, default=0.45,
                        help="Tracking camera distance for --video, in metres")
    parser.add_argument("--seed", type=int, default=0,
                        help="Env seed. Was hardcoded to 0, which makes every run the SAME "
                             "episode -- including the same terrain field and spawn point. That "
                             "is good for comparing checkpoints and misleading for judging a "
                             "gait: seed 0 happens to draw a heightfield that pushes this robot "
                             "696 mm sideways and 31 deg off heading, where seeds 1/2/3/7 give "
                             "-525/+294/-207/-28 mm. Vary it before concluding the policy drifts.")
    parser.add_argument("--flat", action="store_true",
                        help="Disable terrain and slope. The run's own cfgs.pkl is used by "
                             "default, so a checkpoint trained with --terrain is EVALUATED on "
                             "terrain -- correct for judging what it learned, but it makes "
                             "lateral drift look like a gait property when it is the ground. "
                             "On flat, drift across those same seeds is +24..+88 mm.")
    parser.add_argument("--domain-rand", action="store_true",
                        help="Keep push perturbations / observation noise / physics randomization "
                             "on. Off by default: the policy is trained to be robust to them, but "
                             "they make the gait noisier to watch, not more representative.")
    parser.add_argument("--init-tilt", type=float, default=None, metavar="DEG",
                        help="Random roll/pitch at episode start. Defaults to 0 for viewing "
                             "and to the run's own value under --domain-rand, because that "
                             "is what it is: at the default 20 deg every episode spawns the "
                             "robot leaning up to 20 deg in a random direction and it spends "
                             "~25 ticks recovering, which looks like a fault if you are not "
                             "expecting it. Pass a value to see it deliberately.")
    args = parser.parse_args()

    log_dir = f"logs/{args.exp_name}"
    with open(f"{log_dir}/cfgs.pkl", "rb") as f:
        env_cfg, obs_cfg, reward_cfg, command_cfg, train_cfg = pickle.load(f)

    ckpt = args.ckpt
    if ckpt is None:
        ckpt = max(int(f[6:-3]) for f in os.listdir(log_dir) if f.startswith("model_"))
    ckpt_path = os.path.join(log_dir, f"model_{ckpt}.pt")

    env_cfg["domain_rand_enabled"] = args.domain_rand
    # Initial tilt is randomization, and it is the most visually jarring kind -- but it was
    # gated on base_init_tilt_deg alone, not on domain_rand_enabled, so --domain-rand being
    # off silently left it on. That is what makes a fresh episode look wrong before the
    # policy has done anything: the robot spawns leaning, not walking badly.
    if args.init_tilt is not None:
        env_cfg["base_init_tilt_deg"] = args.init_tilt
    elif not args.domain_rand:
        env_cfg["base_init_tilt_deg"] = 0.0
    if float(env_cfg.get("base_init_tilt_deg", 0.0)) > 0.0:
        print(f"  init tilt: +-{float(env_cfg['base_init_tilt_deg']):.0f} deg random roll/pitch "
              f"at every episode start")
    if args.flat:
        env_cfg["terrain_enabled"] = False
        env_cfg["slope_deg_range"] = [0.0, 0.0]
    # A cfgs.pkl without this key MEANS something: the run trained observing the simulator's
    # true dof_pos. Defaulting it to today's value
    # would evaluate every old checkpoint against an observation it never saw -- which is
    # precisely the sim-to-real gap this flag exists to close, re-created inside the evaluator.
    if "observe_servo_estimate" not in env_cfg:
        env_cfg["observe_servo_estimate"] = False
        print("note: this checkpoint predates the servo-estimate observation; evaluating it "
              "against the true dof_pos it was trained on.")
    if args.command is None:
        lo, hi = command_cfg["lin_vel_x_range"]
        args.command = [0.5 * (lo + hi), 0.0, 0.0]
    # Pin the command for the whole run so the gait is watched at one steady speed.
    # Capture the TRAINED yaw-rate limit before collapsing the ranges below: a heading-mode
    # policy computes its own yaw command, so the controller's actuation limit must survive
    # even though the sampling range does not. Without this, evaluating a heading policy at a
    # fixed command clips its yaw command to zero and the gait shown is not the trained one.
    trained_yaw_limit = max(abs(v) for v in command_cfg["ang_vel_range"]) or 0.6
    command_cfg = dict(command_cfg)
    command_cfg["lin_vel_x_range"] = [args.command[0], args.command[0]]
    command_cfg["lin_vel_y_range"] = [args.command[1], args.command[1]]
    command_cfg["ang_vel_range"] = [args.command[2], args.command[2]]
    # A heading-mode policy computes its own yaw-rate command, so collapsing the SAMPLING range
    # above must not also collapse the controller's actuation limit -- carry the trained limit
    # across explicitly. Without this, evaluating a heading policy at a fixed command clips its
    # yaw command to zero and the gait shown is not the one that was trained.
    if env_cfg.get("heading_command"):
        env_cfg = dict(env_cfg)
        env_cfg.setdefault("heading_max_yaw_rate", trained_yaw_limit)

    env = MjVecEnv(num_envs=1, env_cfg=env_cfg, obs_cfg=obs_cfg, reward_cfg=reward_cfg,
                   command_cfg=command_cfg, device="cpu", num_threads=1, seed=args.seed)

    runner = OnPolicyRunner(env, train_cfg, log_dir, device="cpu")
    runner.load(ckpt_path, load_cfg=EVAL_LOAD_CFG)
    policy = runner.get_inference_policy(device="cpu")

    # Full-detail model purely for rendering; the env keeps stepping its own stripped copy.
    view_model = mujoco.MjModel.from_xml_path(XML_PATH)
    view_data = mujoco.MjData(view_model)

    print(f"checkpoint : {ckpt_path}")
    print(f"command    : vx={args.command[0]:.3f} vy={args.command[1]:.3f} wz={args.command[2]:.3f} "
          f"(local +y is forward, see _reward_tracking_lin_vel)")
    print(f"domain rand: {args.domain_rand}  seed: {args.seed}  "
          f"terrain: {env_cfg.get('terrain_enabled', False)}")

    obs = env.reset()
    fwd, lat, height, steps, resets = [], [], [], 0, 0

    def advance():
        """One control step; mirrors the env's physics state into the render model."""
        nonlocal obs, steps, resets
        actions = policy(obs)
        obs, _, dones, _ = env.step(actions)
        mujoco.mj_setState(view_model, view_data, env.state[0], mujoco.mjtState.mjSTATE_FULLPHYSICS)
        mujoco.mj_forward(view_model, view_data)
        f = env._features
        fwd.append(f["base_lin_vel"][0, 1])
        lat.append(f["base_lin_vel"][0, 0])
        height.append(f["base_pos"][0, 2])
        steps += 1
        resets += int(dones.sum())

    with torch.no_grad():
        if args.video:
            import imageio

            renderer = mujoco.Renderer(view_model, height=480, width=640)
            cam = mujoco.MjvCamera()
            mujoco.mjv_defaultCamera(cam)
            cam.distance, cam.elevation, cam.azimuth = args.camera_distance, -20.0, 135.0
            cam.trackbodyid = view_model.body("root").id
            cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING

            n_frames = int(args.duration / env.dt)
            with imageio.get_writer(args.video, fps=round(1.0 / env.dt)) as writer:
                for _ in range(n_frames):
                    advance()
                    renderer.update_scene(view_data, camera=cam)
                    writer.append_data(renderer.render())
            print(f"\nwrote {args.video}")
        else:
            try:
                viewer_ctx = mujoco.viewer.launch_passive(view_model, view_data)
            except RuntimeError as e:
                if "mjpython" not in str(e):
                    raise
                raise SystemExit(
                    "\nmacOS needs the interactive viewer on the process main thread, so it must "
                    "run under mjpython:\n"
                    f"    uv run mjpython sim/mj_eval.py -e {args.exp_name}"
                    + (f" --ckpt {args.ckpt}" if args.ckpt is not None else "")
                    + "\n\nOr render offscreen instead, which works under plain python:\n"
                    f"    uv run python sim/mj_eval.py -e {args.exp_name} --video gait.mp4\n"
                ) from e
            with viewer_ctx as viewer:
                t_start = time.time()
                while viewer.is_running() and time.time() - t_start < args.duration:
                    step_t0 = time.time()
                    advance()
                    viewer.sync()
                    elapsed = time.time() - step_t0
                    if elapsed < env.dt:
                        time.sleep(env.dt - elapsed)

    env.close()
    if steps:
        print(f"\n{steps} control steps ({steps * env.dt:.1f}s simulated), {resets} termination(s)")
        print(f"  forward vel : {np.mean(fwd):+.4f} m/s  (commanded {args.command[0]:.3f})")
        print(f"  lateral vel : {np.mean(lat):+.4f} m/s  (commanded {args.command[1]:.3f})")
        print(f"  base height : {np.mean(height):.4f} m  (target {reward_cfg['base_height_target']})")


if __name__ == "__main__":
    main()
