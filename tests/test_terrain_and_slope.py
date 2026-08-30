"""Terrain, slope and the raised termination threshold.

These three exist to answer one measured finding: a policy trained the obvious way ignores the IMU
entirely, because termination at 10 deg made "never deviate" strictly better than "recover".
The tests below are mostly about the ways that change can go WRONG quietly:

  * a heightfield whose height is not subtracted from foot z turns every foot resting on a rise
    into an "airborne" foot, corrupting four reward terms at once in the direction of paying
    the robot to stand on high ground;
  * a per-episode slope written into a SHARED model object tilts all 512 envs together;
  * a spawn that ignores terrain height drops the robot inside the ground.

None of those raise an exception. They just train a worse policy.
"""

import numpy as np
import pytest
import torch

import mj_vec_env as MVE
from config import get_cfgs
from mj_vec_env import MjVecEnv


def _env(num_envs=8, ang_vel_range=None, **over):
    env_cfg, obs_cfg, reward_cfg, command_cfg = get_cfgs()
    if ang_vel_range is not None:
        command_cfg["ang_vel_range"] = list(ang_vel_range)
    env_cfg["domain_rand_enabled"] = False
    env_cfg["terrain_enabled"] = False
    env_cfg["slope_deg_range"] = [0.0, 0.0]
    # Start upright. These tests are about terrain GEOMETRY -- where the ground is under a
    # foot -- and a randomized initial attitude (get_cfgs ships 20 deg) puts the robot in a
    # recovery transient with its feet in the air, which is a fact about the reset, not
    # about the contact test. Any test here that wants the tilt can pass it via `over`.
    env_cfg["base_init_tilt_deg"] = 0.0
    env_cfg.update(over)
    return MjVecEnv(num_envs=num_envs, env_cfg=env_cfg, obs_cfg=obs_cfg, reward_cfg=reward_cfg,
                    command_cfg=command_cfg, device="cpu", num_threads=1, seed=0)


# -- the field itself -------------------------------------------------------

def test_terrain_field_is_normalised_and_correlated():
    """MuJoCo requires hfield_data in [0,1]; and the field must actually be smooth, or
    'terrain' is per-cell white noise, which at this robot's scale is an obstacle course."""
    f = MVE.make_terrain_field(64, 64, 0.04, 0.10, seed=0)
    assert f.min() == pytest.approx(0.0)
    assert f.max() == pytest.approx(1.0)
    # Neighbouring cells (0.04 m apart, well inside the 0.10 m correlation length) must be far
    # more alike than distant ones.
    near = np.abs(np.diff(f, axis=1)).mean()
    far = np.abs(f[:, :-8] - f[:, 8:]).mean()
    assert near < 0.5 * far, f"field is not correlated: near={near:.4f} far={far:.4f}"


def test_terrain_field_is_periodic():
    """The FFT construction should tile seamlessly -- the wrap-around seam must be no worse
    than an ordinary interior step, or the field has an edge artifact."""
    f = MVE.make_terrain_field(64, 64, 0.04, 0.10, seed=1)
    seam = np.abs(f[:, -1] - f[:, 0]).mean()
    interior = np.abs(np.diff(f, axis=1)).mean()
    assert seam < 3.0 * interior


def test_correlation_length_controls_feature_size():
    rough = MVE.make_terrain_field(64, 64, 0.04, 0.05, seed=2)
    smooth = MVE.make_terrain_field(64, 64, 0.04, 0.40, seed=2)
    assert np.abs(np.diff(smooth, axis=1)).mean() < np.abs(np.diff(rough, axis=1)).mean()


# -- the sampler ------------------------------------------------------------

def test_terrain_height_matches_the_model_grid():
    """terrain_height() is a reimplementation of MuJoCo's hfield lookup in numpy; if it drifts
    from the model's own data, contact and clearance are computed against imaginary ground."""
    env = _env(terrain_enabled=True)
    g = env._terrain_grid
    nrow, ncol = env._terrain_nrow, env._terrain_ncol
    rx, ry = env._terrain_radius
    # Sample exactly on grid nodes, where bilinear interpolation must return the node itself.
    for r, c in [(0, 0), (nrow - 1, ncol - 1), (nrow // 3, ncol // 2)]:
        x = -rx + 2 * rx * c / (ncol - 1)
        y = -ry + 2 * ry * r / (nrow - 1)
        assert env.terrain_height(np.array([x, y])) == pytest.approx(
            g[r, c] * env._terrain_elev, abs=1e-9)
    env.close()


def test_terrain_height_is_zero_without_terrain():
    """Flat runs must not pay for the feature, and must reduce to the old absolute-z test."""
    env = _env(terrain_enabled=False)
    assert env._terrain_grid is None
    assert env.terrain_height(np.zeros((5, 4, 2))).shape == (5, 4)
    assert np.all(env.terrain_height(np.random.default_rng(0).normal(size=(9, 2))) == 0.0)
    env.close()


def test_terrain_height_is_bounded_by_amplitude():
    env = _env(terrain_enabled=True, terrain_amplitude=0.008)
    xy = np.random.default_rng(3).uniform(-1.0, 1.0, (500, 2))
    h = env.terrain_height(xy)
    assert h.min() >= 0.0 and h.max() <= 0.008 + 1e-9
    env.close()


# -- the coupling that would break silently ---------------------------------

def test_contact_uses_height_above_local_ground():
    """The failure this guards: a foot resting on a rise reads as airborne if contact is tested
    against absolute z. Verified by driving the same foot trace through both interpretations."""
    env = _env(terrain_enabled=True, terrain_amplitude=0.008)
    env.reset()
    for _ in range(5):
        env.step(torch.zeros((env.num_envs, env.num_actions)))
    foot_xy = env._sensor_out[:, -1, env._foot_sensor_xy_adr].reshape(env.num_envs, -1, 2)
    foot_z = env._sensor_out[:, -1, env._foot_sensor_z_adr]
    ground = env.terrain_height(foot_xy)
    assert ground.shape == foot_z.shape
    # Standing still on terrain: essentially every foot is loaded. Absolute z would disagree
    # for any foot whose patch of ground is raised.
    assert (foot_z - ground < MVE.FEET_CONTACT_HEIGHT_THRESHOLD).mean() > 0.9
    assert ground.max() > 1e-4, "terrain is flat here; the test would pass vacuously"
    env.close()


def test_spawn_sits_on_the_terrain_not_inside_it():
    """A spawn at the flat-floor height drops the robot into any rise, and MuJoCo resolves
    interpenetration by launching it."""
    env = _env(terrain_enabled=True, terrain_amplitude=0.008)
    env.reset()
    # FULLPHYSICS is [time, qpos, ...]: base xyz is state[1:4]. Getting this wrong is exactly
    # the bug this suite caught -- an earlier _respawn_on_terrain wrote x into the clock.
    base_z = env.state[:, 3]
    ground = env.terrain_height(env.state[:, 1:3])
    clearance = base_z - ground
    assert np.allclose(clearance, clearance[0], atol=1e-9), \
        "every env should start the same height ABOVE ITS OWN ground"
    assert clearance[0] == pytest.approx(env._init_state[3], abs=1e-9)
    env.close()


def test_spawn_positions_differ_across_envs():
    """All terrain variety comes from respawning elsewhere on a shared field; if the spawn is
    constant, 512 envs walk the same 2 m for the whole run."""
    env = _env(num_envs=32, terrain_enabled=True)
    env.reset()
    assert env.state[:, 1].std() > 0.1
    assert env.state[:, 2].std() > 0.1
    assert np.allclose(env.state[:, 0], 0.0), "index 0 is simulation time, not a coordinate"
    env.close()


# -- slope ------------------------------------------------------------------

def test_slope_tilts_gravity_per_env_not_globally():
    """The sharp edge: with domain randomization off the env models used to be ONE shared
    object, so writing gravity per episode would tilt the whole population together."""
    env = _env(num_envs=16, slope_deg_range=[3.0, 10.0])
    env.reset()
    grav = np.array([m.opt.gravity[:] for m in env.models])
    assert len({id(m) for m in env.models}) == env.num_envs, "models are shared; slope leaks"
    assert grav[:, 2].std() > 0 or grav[:, 0].std() > 0
    # magnitude preserved -- a tilt must not change how strong gravity is
    assert np.allclose(np.linalg.norm(grav, axis=1),
                       np.linalg.norm(env._base_model.opt.gravity), atol=1e-9)
    # ...and the tilt angle stays inside the configured band
    tilt = np.degrees(np.arccos(np.clip(-grav[:, 2] / np.linalg.norm(grav, axis=1), -1, 1)))
    assert tilt.min() >= 3.0 - 1e-6 and tilt.max() <= 10.0 + 1e-6
    assert tilt.std() > 0.5, "all envs got the same slope"
    env.close()


def test_slope_is_resampled_per_episode():
    env = _env(num_envs=8, slope_deg_range=[2.0, 12.0])
    env.reset()
    first = env._slope_deg.copy()
    env._reset_idx(np.ones(env.num_envs, dtype=bool))
    assert not np.allclose(first, env._slope_deg)
    env.close()


def test_no_slope_leaves_gravity_untouched():
    env = _env(slope_deg_range=[0.0, 0.0])
    env.reset()
    assert env.models[0] is env._base_model, "should not pay for per-env copies when flat"
    assert np.allclose(env.models[0].opt.gravity[:2], 0.0)
    env.close()


# -- termination ------------------------------------------------------------

def test_termination_threshold_is_configurable_and_raised():
    env_cfg, _, _, _ = get_cfgs()
    assert env_cfg["termination_if_roll_greater_than"] > 10, \
        "the 10 deg threshold is what made every policy open-loop; see get_cfgs"
    assert env_cfg["termination_if_pitch_greater_than"] == \
        env_cfg["termination_if_roll_greater_than"]


def test_tilt_below_threshold_survives():
    """A robot lying at 30 deg must keep its episode -- that is the entire point of raising the
    threshold, and it is what gives a recovery policy something to learn from."""
    env = _env(num_envs=4, termination_if_roll_greater_than=45,
               termination_if_pitch_greater_than=45)
    env.reset()
    half = np.radians(30.0) / 2.0
    env.state[:, 4:8] = np.array([np.cos(half), np.sin(half), 0.0, 0.0])  # 30 deg roll
    env.step(torch.zeros((env.num_envs, env.num_actions)))
    assert not env.reset_buf.any(), "30 deg terminated under a 45 deg threshold"
    env.close()


def test_tilt_above_threshold_terminates():
    env = _env(num_envs=4, termination_if_roll_greater_than=45,
               termination_if_pitch_greater_than=45)
    env.reset()
    half = np.radians(75.0) / 2.0
    env.state[:, 4:8] = np.array([np.cos(half), np.sin(half), 0.0, 0.0])
    env.step(torch.zeros((env.num_envs, env.num_actions)))
    assert env.reset_buf.all()
    env.close()


# -- pushes -----------------------------------------------------------------

def test_push_magnitude_follows_config():
    env = _env(num_envs=64, domain_rand_enabled=True, push_dv_xy=0.5, push_dv_z=0.0)
    env.reset()
    qvel0 = 1 + env._base_model.nq
    env.next_push_step[:] = 0
    before = env.state[:, qvel0:qvel0 + 3].copy()
    env._maybe_push_robots()
    dv = env.state[:, qvel0:qvel0 + 3] - before
    assert np.abs(dv[:, :2]).max() <= 0.5 + 1e-9
    assert np.abs(dv[:, :2]).max() > 0.2, "pushes far below the configured magnitude"
    assert np.allclose(dv[:, 2], 0.0), "push_dv_z=0 should produce no vertical kick"
    env.close()


# Push magnitude is asserted in tests/test_reward_support.py instead, which bounds it from BOTH
# sides. A one-sided ">= 0.15" lived here first and was exactly the reasoning that produced
# the crawl failure: the push was checked against walking speed, never against the reward's width.


# -- the whole thing runs ---------------------------------------------------

def test_full_closed_loop_config_steps_cleanly():
    """Everything on at once: terrain, slope, big pushes, raised threshold, domain rand."""
    env = _env(num_envs=16, domain_rand_enabled=True, terrain_enabled=True,
               slope_deg_range=[0.0, 10.0], push_dv_xy=0.2, push_dv_z=0.1,
               termination_if_roll_greater_than=45, termination_if_pitch_greater_than=45)
    env.reset()
    rng = np.random.default_rng(0)
    for _ in range(40):
        obs, rew, done, _ = env.step(
            torch.as_tensor(rng.normal(0.0, 0.4, (env.num_envs, env.num_actions))))
        assert torch.isfinite(rew).all()
        for name, tensor in obs.items():
            assert torch.isfinite(tensor).all(), f"non-finite {name}"
    env.close()


def test_privileged_width_matches_what_is_written():
    env = _env(num_envs=4, domain_rand_enabled=True, terrain_enabled=True,
               slope_deg_range=[0.0, 8.0])
    env.reset()
    env.step(torch.zeros((env.num_envs, env.num_actions)))
    assert env.get_observations()["privileged"].shape[1] == env.num_privileged_obs
    env.close()


def test_slope_is_capped_by_this_robots_own_friction():
    """Every episode must be walkable. Sampling slope and friction independently puts some
    robots on ground that physically cannot hold them (tan(slope) > mu), which teaches nothing
    -- and even where friction nominally holds, a policy drifts downhill at 0.086 m/s against a
    0.063 m/s walk, so the bar is progress, not merely standing."""
    env = _env(num_envs=256, domain_rand_enabled=True, slope_deg_range=[0.0, 30.0],
               slope_friction_margin=0.5)
    env.reset()
    # The constraint is on the FORCE RATIO -- the robot slides when tan(slope) > mu -- so the
    # margin multiplies mu, not the angle. atan(0.5*mu) != 0.5*atan(mu), and asserting the
    # latter is what this test got wrong first.
    assert (np.tan(np.radians(env._slope_deg)) <= 0.5 * env._foot_friction + 1e-9).all(), (
        "some episode exceeded half the friction it needs to hold station")
    assert (np.tan(np.radians(env._slope_deg)) < env._foot_friction).all(), (
        "an episode is on ground its own friction cannot hold")
    assert env._slope_deg.std() > 0.3, "the cap collapsed the slope distribution to a constant"
    env.close()


def test_slope_margin_zero_restores_independent_sampling():
    """The coupling must be opt-in, so configs written without it reproduce exactly."""
    env = _env(num_envs=128, domain_rand_enabled=True, slope_deg_range=[7.9, 8.0],
               slope_friction_margin=0.0)
    env.reset()
    assert env._slope_deg.min() > 7.8, "margin 0 should ignore friction entirely"
    env.close()


# -- heading ----------------------------------------------------------------

def test_projected_gravity_is_yaw_invariant():
    """The premise of heading mode, stated as a test. Gravity in body frame does not change
    under rotation about the vertical, so the actor has NO absolute heading reference -- spin
    the robot 180 deg and its observation is bit-identical."""
    env = _env(num_envs=4)
    env.reset()
    seen = []
    for yaw_deg in (0.0, 90.0, 180.0):
        half = np.radians(yaw_deg) / 2.0
        env.state[:, 4:8] = np.array([np.cos(half), 0.0, 0.0, np.sin(half)])
        seen.append(env._decode_state()["projected_gravity"].copy())
    for g in seen[1:]:
        assert np.allclose(g, seen[0], atol=1e-9), \
            "gravity changed under pure yaw; the heading-mode premise would be wrong"
    env.close()


def test_heading_command_drives_yaw_rate_toward_the_target():
    env = _env(num_envs=64, heading_command=True, heading_kp=0.5,
                ang_vel_range=(-0.6, 0.6))
    env.reset()
    # Face 0, want +90 deg -> a positive yaw-rate command; want -90 -> negative.
    env.state[:, 4:8] = np.array([1.0, 0.0, 0.0, 0.0])
    for target, sign in ((np.pi / 2, +1), (-np.pi / 2, -1)):
        env._heading_target[:] = target
        env._update_heading_command(np.zeros(env.num_envs))
        assert np.all(np.sign(env.commands[:, 2]) == sign), \
            f"heading target {target:.2f} produced yaw-rate command of the wrong sign"
    # On target -> no turn commanded.
    env._heading_target[:] = 0.0
    env._update_heading_command(np.zeros(env.num_envs))
    assert np.allclose(env.commands[:, 2], 0.0, atol=1e-9)
    env.close()


def test_heading_error_wraps_the_short_way():
    """Without wrapping, a target just across the +-pi seam commands a near-full rotation the
    wrong way round -- and it would look like a plausible large heading error."""
    env = _env(num_envs=8, heading_command=True, heading_kp=0.5,
                ang_vel_range=(-0.6, 0.6))
    env.reset()
    env._heading_target[:] = np.radians(179.0)
    env._update_heading_command(np.full(env.num_envs, np.radians(-179.0)))
    # Facing -179, want +179: those are 2 deg apart across the seam, and the short way is
    # NEGATIVE (-179 -> -181 == +179), not the +358 an unwrapped subtraction would give.
    assert np.all(env.commands[:, 2] < 0.0)
    assert np.all(np.abs(env.commands[:, 2]) < 0.5 * np.radians(10.0)), \
        "turned the long way round the seam"
    env.close()


def test_heading_command_respects_the_yaw_rate_limits():
    env = _env(num_envs=16, heading_command=True, heading_kp=50.0,   # absurd gain
                ang_vel_range=(-0.6, 0.6))
    env.reset()
    env._heading_target[:] = np.pi
    env._update_heading_command(np.zeros(env.num_envs))
    lo, hi = env._commands_low[2], env._commands_high[2]
    assert np.all(env.commands[:, 2] >= lo) and np.all(env.commands[:, 2] <= hi), \
        "heading controller commanded a yaw rate outside the trained command range"
    env.close()


def test_heading_mode_off_leaves_yaw_command_sampled():
    """Must be opt-in: without it the yaw rate is sampled independently."""
    env = _env(num_envs=64, heading_command=False)
    env.reset()
    before = env.commands[:, 2].copy()
    env._update_heading_command(np.zeros(env.num_envs))
    assert np.array_equal(before, env.commands[:, 2])
    env.close()


def test_heading_mode_refuses_a_degenerate_yaw_range():
    """The silent-inertness guard. get_cfgs ships ang_vel_range [0,0], and the heading
    controller clips to it -- so without this, --heading would train a full run commanding
    exactly zero yaw rate and report no benefit, which looks identical to the idea not working.
    """
    with pytest.raises(ValueError, match="ang_vel_range"):
        _env(heading_command=True, ang_vel_range=(0.0, 0.0)).close()
