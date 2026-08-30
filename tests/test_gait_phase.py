"""The gait clock is a cross-module contract, so it gets a cross-module test.

sim/mj_vec_env advances phase and appends three channels; deploy/obs_builder.GaitClock has
to produce byte-identical values from its own clock on the robot. deploy/ runs with only
numpy and pyserial and cannot import sim/, so the constants are DUPLICATED -- the same
arrangement as NUM_BASE_OBS. Duplication is only acceptable while something checks it.

The repo has been bitten by exactly this before: the servo-observer phase was a cross-module
invariant stated in a comment, the other module changed, nothing checked, and 16 observation
channels silently disagreed by one control step.
"""

import math

import numpy as np
import pytest

import obs_builder as ob
import policy_io as pio
from config import get_cfgs
from mj_vec_env import MjVecEnv


def test_deploy_clock_matches_the_shared_contract_exactly():
    for hz in (0.60, 0.875, 1.0, 1.15):
        g = ob.GaitClock(hz, 0.02)
        for phase in (0.0, 0.013, 0.25, 0.5, 0.751, 0.999):
            g.phase = phase
            assert np.array_equal(np.array(g.obs()), np.array(pio.gait_obs(phase, hz))), (hz, phase)


def test_deploy_constants_match_sim():
    assert ob.GaitClock.HZ_RANGE == pio.GAIT_HZ_RANGE
    assert ob.GaitClock.HZ_MID == pio.GAIT_HZ_MID
    assert ob.GaitClock.HZ_HALF == pio.GAIT_HZ_HALF
    assert ob.NUM_GAIT_OBS == pio.GAIT_OBS
    assert ob.NUM_BASE_OBS + ob.NUM_GAIT_OBS == pio.NUM_OBS_WITH_GAIT


def test_cadence_outside_the_trained_range_is_refused():
    """The ceiling is the servo, not the control loop: at ~34.5 deg peak-to-peak the gait
    fundamental alone reaches the 137 deg/s slew limit at 1.26 Hz. A policy asked to walk
    faster than it ever trained would saturate on its fundamental."""
    for bad in (0.4, 1.3, 2.0):
        with pytest.raises(ValueError, match="outside the trained range"):
            ob.GaitClock(bad, 0.02)


def test_phase_integration_matches_between_env_and_clock():
    """Both must advance by exactly dt*hz per control tick and wrap identically."""
    hz, dt, n = 1.03, 0.02, 137
    g = ob.GaitClock(hz, dt)
    for _ in range(n):
        g.tick()
    assert g.phase == pytest.approx((n * dt * hz) % 1.0, abs=1e-12)


def test_env_advances_phase_by_dt_times_its_own_cadence():
    env_cfg, obs_cfg, reward_cfg, command_cfg = get_cfgs()
    env_cfg.update({"terrain_enabled": False, "domain_rand_enabled": False,
                    "base_init_tilt_deg": 0.0, "gait_phase": True})
    env = MjVecEnv(num_envs=4, env_cfg=env_cfg, obs_cfg=obs_cfg, reward_cfg=reward_cfg,
                   command_cfg=command_cfg, device="cpu", num_threads=1, seed=0)
    assert env.num_obs == pio.NUM_OBS_WITH_GAIT
    hz = env._gait_hz.copy()
    before = env._gait_phase.copy()
    env.step(np.zeros((4, env.num_actions)))
    assert np.allclose((env._gait_phase - before) % 1.0, env.dt * hz, atol=1e-12)
    env.close()


def test_env_appends_rather_than_inserts():
    """Appended, not inserted: every existing channel index must keep its meaning, or every
    obs dump, diff tool and channel-name list silently means something else.

    Checked structurally against the env's own state rather than by diffing a gait env
    against a plain one -- enabling the clock draws two extra random numbers per reset
    (cadence and start phase), which shifts the RNG stream, so two separately seeded envs
    diverge in their COMMANDS while the observation layout is perfectly correct.
    """
    env_cfg, obs_cfg, reward_cfg, command_cfg = get_cfgs()
    env_cfg.update({"terrain_enabled": False, "domain_rand_enabled": False,
                    "base_init_tilt_deg": 0.0, "gait_phase": True})
    env = MjVecEnv(num_envs=4, env_cfg=env_cfg, obs_cfg=obs_cfg, reward_cfg=reward_cfg,
                   command_cfg=command_cfg, device="cpu", num_threads=1, seed=0)
    env.step(np.zeros((4, env.num_actions)))
    o = env.obs_buf
    assert o.shape[1] == 36

    # The 33 keep their documented slices.
    assert np.allclose(o[:, 3:6], env._features["projected_gravity"], atol=1e-5), "gravity at 3:6"
    assert np.allclose(o[:, 6:9], env.commands * env._commands_scale, atol=1e-5), "commands at 6:9"
    assert np.allclose(o[:, 25:33], env.actions, atol=1e-5), "last_action at 25:33"

    # ...and the three new ones are what they claim.
    for i in range(env.num_envs):
        want = pio.gait_obs(env._gait_phase[i], env._gait_hz[i])
        assert np.allclose(o[i, 33:36], want, atol=1e-6), i
    env.close()
