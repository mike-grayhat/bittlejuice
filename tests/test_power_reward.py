"""Mechanical power: sum_j |tau_j * qdot_j| over the eight joints, in watts.

For the energy-reward direction. The term exists
because a velocity command alone makes the policy shorten its STRIDE rather
than change its GAIT: across a 3x command range the action-power split across frequency
bands barely moved (85/15 -> 88/11) while amplitude went 38.6 -> 46.6 deg. Penalising work
per unit time is what the literature uses to make a slow gait genuinely cheaper.

The failure this file is mainly guarding: torque comes from sensors declared in ACTUATOR
order and joint velocity from state in CANONICAL order. Pair them wrongly and every number
downstream is still finite, still positive, still plausible -- and measures nothing.
"""

import numpy as np
import pytest

import policy_io as pio
from config import get_cfgs
from mj_vec_env import MjVecEnv


def _env(**over):
    env_cfg, obs_cfg, reward_cfg, command_cfg = get_cfgs()
    env_cfg.update({"terrain_enabled": False, "domain_rand_enabled": False,
                    "base_init_tilt_deg": 0.0})
    env_cfg.update(over)
    return MjVecEnv(num_envs=4, env_cfg=env_cfg, obs_cfg=obs_cfg, reward_cfg=reward_cfg,
                    command_cfg=command_cfg, device="cpu", num_threads=1, seed=0)


def test_torque_sensors_are_in_canonical_joint_order():
    """THE test. Walks the mapping back to joint names independently of how it was built."""
    env = _env()
    m = env._base_model
    for k, joint_name in enumerate(pio.JOINT_NAMES):
        adr = env._actuator_frc_adr[k]
        sensor_ids = [i for i in range(m.nsensor) if m.sensor(i).adr[0] == adr]
        assert len(sensor_ids) == 1, f"ambiguous sensor address for {joint_name}"
        sensor = m.sensor(sensor_ids[0])
        actuator_id = int(sensor.objid[0])
        driven_joint_id = int(m.actuator_trnid[actuator_id, 0])
        assert m.joint(driven_joint_id).name == joint_name, (
            f"canonical slot {k} ({joint_name}) reads the torque of "
            f"{m.joint(driven_joint_id).name}")
    env.close()


def test_velocity_offsets_match_the_decoded_dof_vel():
    """The other half of the pairing: qdot read from raw state must equal decoded dof_vel."""
    env = _env()
    env.reset()
    env.step(np.zeros((env.num_envs, env.num_actions)))
    from_state = env.state[:, env._joint_qvel_state_adr]
    np.testing.assert_allclose(from_state, env._features["dof_vel"], rtol=0, atol=0)
    env.close()


def test_power_is_nonnegative_and_bounded_by_the_force_limit():
    """|tau| is clamped to forcerange, so power cannot exceed sum_j forcerange * |qdot_j|."""
    env = _env()
    obs = env.reset()
    rng = np.random.default_rng(0)
    for _ in range(50):
        env.step(rng.uniform(-1.0, 1.0, (env.num_envs, env.num_actions)))
        assert np.all(env._joint_power >= 0.0)
        assert np.all(np.isfinite(env._joint_power))
        ceiling = (env.models[0].actuator_forcerange[:, 1].max()
                   * np.abs(env._features["dof_vel"]).sum(axis=1))
        # Substep-averaged power against a ceiling built from end-of-step velocity, so this
        # is a sanity bound rather than a tight one -- it catches unit and pairing errors.
        assert np.all(env._joint_power <= ceiling + 1.0)
    env.close()


def test_a_still_robot_costs_almost_nothing():
    """The term must price MOTION. A standing robot holding its pose does no work."""
    env = _env()
    env.reset()
    for _ in range(60):                       # settle onto its legs at the default pose
        env.step(np.zeros((env.num_envs, env.num_actions)))
    still = float(env._joint_power.mean())

    for _ in range(30):                       # now sweep the joints hard
        env.step(np.full((env.num_envs, env.num_actions), 1.0))
        env.step(np.full((env.num_envs, env.num_actions), -1.0))
    moving = float(env._joint_power.mean())
    assert moving > 5.0 * max(still, 1e-6), f"still={still:.4f} moving={moving:.4f}"
    env.close()


def test_scale_is_zero_by_default():
    """Every config written before this feature must train exactly as before."""
    _, _, reward_cfg, _ = get_cfgs()
    assert reward_cfg["reward_scales"]["power"] == 0.0


def test_power_appears_in_episode_metrics():
    env = _env(episode_length_s=0.1)
    env.reset()
    for _ in range(12):
        env.step(np.zeros((env.num_envs, env.num_actions)))
    assert "power_w" in env.extras["episode"]
    assert env.extras["episode"]["power_w"] >= 0.0
    env.close()
