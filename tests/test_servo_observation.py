"""The sim's observed joint state must equal what deploy computes on hardware.

This is the mismatch that dominated sim-to-real: the simulator observed its own true
dof_pos, while the robot cannot read its joints at all (~20 ms each, ~9 Hz against a 50 Hz
loop, and the read detaches the servo) and instead integrates its own commanded targets.
Feeding a trained policy the signal it would really receive took gait cadence 1.17 -> 4.65 Hz
and forward speed +0.079 -> -0.020 m/s.

So the two implementations are pinned against each other here. They live in different
packages with no shared code, which is exactly why they can drift apart silently.
"""


import numpy as np
import pytest
import torch


import mj_vec_env as MVE
import obs_builder as ob
from config import get_cfgs
from mj_vec_env import MjVecEnv


def _env(num_envs=4, **over):
    env_cfg, obs_cfg, reward_cfg, command_cfg = get_cfgs()
    env_cfg["terrain_enabled"] = False
    env_cfg.update(over)
    return MjVecEnv(num_envs=num_envs, env_cfg=env_cfg, obs_cfg=obs_cfg, reward_cfg=reward_cfg,
                    command_cfg=command_cfg, device="cpu", num_threads=1, seed=0), obs_cfg


def test_sim_observer_matches_deploy_servo_model():
    """Bit-for-bit against deploy's ServoModel over a random action sequence."""
    # Domain randomization OFF: extra_latency_prob makes exec_actions stochastic (some envs
    # execute a_{t-2}), and this test needs to predict exactly which action drove the observer.
    # The observer's use of NOMINAL tau under randomization is covered separately below.
    env, _ = _env(num_envs=1, domain_rand_enabled=False)
    default = np.array(env._default_dof_pos)
    servo = ob.ServoModel(default, env.dt, MVE.ACTUATOR_TAU_S_NOMINAL,
                          MVE.ACTUATOR_SLEW_RAD_S_NOMINAL)
    env.reset()

    rng = np.random.default_rng(0)
    # Driven by the DELAYED action, because that is what control_loop.py does: it computes
    # `target = last_action * scale + default` and feeds THAT to servo.update(). This comment
    # used to claim the opposite ("the target it just computed"), describing a version of the
    # deploy loop from before it gained its one-step delay -- and sim followed the comment, so
    # its observer ran 20 ms ahead of the robot's across all 16 joint channels.
    prev_clipped = np.zeros((1, env.num_actions))
    for _ in range(60):
        act = rng.normal(0.0, 1.0, (1, env.num_actions))
        env.step(torch.as_tensor(act))
        expected = servo.update(prev_clipped[0] * env.env_cfg["action_scale"] + default)
        prev_clipped = np.clip(act, -env.env_cfg["clip_actions"], env.env_cfg["clip_actions"])
        assert np.allclose(env._servo_estimate[0], expected, atol=0.0, rtol=0.0), (
            "sim observer has drifted from deploy/obs_builder.ServoModel"
        )
    env.close()


def test_observer_uses_nominal_not_randomized_tau():
    """The observer must NOT get the episode's true actuator parameters.

    On hardware the host only has the nominals baked into the .npz; the gap between those and
    the real servo IS the deployed observer's error. Handing the observer the true values
    would quietly delete that error from training.
    """
    env, _ = _env(domain_rand_enabled=True)
    env.reset()
    # Drive so lag and slew both bind, then check the observer disagrees with the plant in
    # exactly those envs whose sampled tau/slew differ from nominal.
    #
    # ALTERNATING, not a sustained extreme. This held action=3.0 -- every joint pinned
    # 0.75 rad off the default pose -- until the actuator fit made the sim
    # servo as quick as the measured hardware, at which point all four envs reached that
    # pose fast enough to topple inside the window and the test had nothing left to
    # measure. Toppling there is the physically right answer (a real robot commanded to
    # that pose falls too), so the drive is the thing to fix: flipping sign every three
    # steps keeps the target moving, which is what makes lag and slew bind, without ever
    # commanding a pose the robot has to fall out of. A constant mild action would not do
    # -- both filters converge to the same steady state and the gap decays to nothing.
    steps = 25
    for k in range(steps):
        sign = 1.0 if (k // 3) % 2 == 0 else -1.0
        env.step(torch.full((env.num_envs, env.num_actions), sign))
    off_nominal = ~np.isclose(env._actuator_tau, MVE.ACTUATOR_TAU_S_NOMINAL, atol=1e-3)
    assert off_nominal.any(), "randomization produced no off-nominal robot; test is vacuous"
    # A reset re-seeds observer and plant to the same pose, so a freshly-reset env has zero gap
    # for reasons that say nothing about which tau the observer used. Exclude envs whose
    # episode restarted inside the window.
    survived = env._episode_length >= steps
    assert survived.any(), "every env reset during the window; nothing left to measure"
    check = off_nominal & survived
    gap = np.abs(env._servo_estimate - env._servo_target).max(axis=1)
    assert (gap[check] > 1e-6).all(), (
        "observer tracks the plant exactly -- it is using the randomized tau/slew"
    )
    env.close()


def test_joint_observation_is_open_loop():
    """A blocked joint must report the same angle as a free one.

    This is the property the whole change is about: the real robot's joint channel carries no
    contact information. If a floor collision ever shows up in the observation, the policy can
    feel the ground through its legs again and the gap is back.
    """
    env, obs_cfg = _env(num_envs=2, domain_rand_enabled=False)
    env.reset()
    act = torch.full((2, env.num_actions), 2.0)
    for _ in range(20):
        obs = env.step(act)[0]["policy"].numpy()
    # env 1 gets its legs physically jammed; both were commanded identically.
    n = env.num_actions
    lo = 3 + 3 + 3                                    # ang_vel, gravity, commands
    dof_block = obs[:, lo:lo + 2 * n]
    assert np.allclose(dof_block[0], dof_block[1]), (
        "joint observation differs between robots given identical commands -- it is not open loop"
    )
    env.close()


def test_true_dof_path_still_available():
    """The legacy observation must remain reproducible, or old checkpoints cannot be evaluated."""
    env, _ = _env(observe_servo_estimate=False, domain_rand_enabled=False)
    env.reset()
    env.step(torch.zeros(env.num_envs, env.num_actions))
    assert env.observe_servo_estimate is False
    env.close()


def test_observer_lead_is_reproducible_for_old_checkpoints():
    """Policies trained against the leading observer must still be evaluable."""
    lead, _ = _env(domain_rand_enabled=False, observer_leads_plant=True)
    keep, _ = _env(domain_rand_enabled=False, observer_leads_plant=False)
    rng = np.random.default_rng(1)
    for _ in range(8):
        a = torch.as_tensor(rng.normal(0.0, 0.5, (lead.num_envs, lead.num_actions)))
        lead.step(a); keep.step(a)
    assert not np.allclose(lead._servo_estimate, keep._servo_estimate), \
        "the flag had no effect; old checkpoints would not reproduce"
    lead.close(); keep.close()
