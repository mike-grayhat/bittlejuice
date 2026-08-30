"""The left/right mirror map, which everything symmetry-related depends on being right.

A wrong mirror map does not crash. It trains a symmetry loss against a target that is not
the mirror of anything, and the run comes back looking like symmetry did not help.

The two traps this file exists for:
  1. Angular velocity is a PSEUDOVECTOR. Reflected in x=0 a true vector flips its x
     component; a pseudovector flips the other two. Treating ang_vel like gravity inverts
     the roll signal and leaves yaw untouched -- exactly backwards.
  2. The MJCF is already mirrored (right joints carry axis="0 0 -1", all defaults +0.56), so
     the joint swap is a pure permutation. Adding sign flips "for the right side" would
     double-count the mirroring the model already did.
"""

import numpy as np
import pytest
import torch

import policy_io as pio


def _frame(rng):
    return rng.normal(size=pio.NUM_OBS)


def test_mirroring_twice_is_the_identity():
    """The single cheapest check that catches most sign errors."""
    rng = np.random.default_rng(0)
    for _ in range(50):
        o = _frame(rng)
        np.testing.assert_allclose(pio.mirror_obs(pio.mirror_obs(o)), o, atol=1e-12)
        a = rng.normal(size=pio.NUM_ACTIONS)
        np.testing.assert_allclose(pio.mirror_action(pio.mirror_action(a)), a, atol=1e-12)


def test_the_joint_swap_is_a_pure_permutation():
    """Left-back must land on right-back, not right-front, and with no sign change."""
    names = pio.JOINT_NAMES
    for j, src in enumerate(pio.MIRROR_JOINT_PERM):
        a, b = names[j], names[src]
        assert a.replace("left", "X").replace("right", "X") == \
               b.replace("left", "X").replace("right", "X"), f"{a} mirrored to {b}"
        assert ("left" in a) != ("left" in b), f"{a} mirrored to {b}: side did not swap"


def test_a_symmetric_pose_is_its_own_mirror():
    """DEFAULT_DOF_POS is all +0.56, which IS the symmetric stance -- so an observation of a
    robot standing still, level, commanded straight ahead must be mirror-invariant."""
    o = np.zeros(pio.NUM_OBS)
    o[3:6] = [0.0, 0.0, -1.0]        # gravity, upright
    o[6] = 0.5                        # forward command only
    np.testing.assert_allclose(pio.mirror_obs(o), o, atol=1e-12)


def test_ang_vel_is_treated_as_a_pseudovector():
    """Roll rate survives the reflection; pitch and yaw rate invert. The opposite convention
    (copied from gravity) passes the involution test above and is still wrong."""
    assert pio.MIRROR_ANG_VEL_SIGNS == [1.0, -1.0, -1.0]
    assert pio.MIRROR_GRAVITY_SIGNS == [-1.0, 1.0, 1.0]
    o = np.zeros(pio.NUM_OBS)
    o[0:3] = [1.0, 1.0, 1.0]
    m = pio.mirror_obs(o)
    assert m[0] == pytest.approx(+1.0), "roll rate must survive a left/right reflection"
    assert m[1] == pytest.approx(-1.0)
    assert m[2] == pytest.approx(-1.0)


def test_a_left_turn_mirrors_to_a_right_turn():
    o = np.zeros(pio.NUM_OBS)
    o[6:9] = [0.4, 0.1, 0.7]          # forward, lateral, yaw
    m = pio.mirror_obs(o)
    assert m[6] == pytest.approx(0.4), "forward command must not flip"
    assert m[7] == pytest.approx(-0.1)
    assert m[8] == pytest.approx(-0.7), "a left turn must mirror to a right turn"


def test_swapping_the_legs_of_an_observation_swaps_them_in_all_three_blocks():
    o = np.zeros(pio.NUM_OBS)
    for base in (9, 17, 25):
        o[base:base + 8] = np.arange(8, dtype=float)
    m = pio.mirror_obs(o)
    for base in (9, 17, 25):
        np.testing.assert_allclose(m[base:base + 8], [4, 5, 6, 7, 0, 1, 2, 3])


def test_works_on_torch_tensors_and_batches():
    rng = np.random.default_rng(1)
    batch = torch.from_numpy(rng.normal(size=(7, pio.NUM_OBS)))
    out = pio.mirror_obs(batch)
    assert isinstance(out, torch.Tensor) and out.shape == batch.shape
    torch.testing.assert_close(pio.mirror_obs(out), batch)


def test_gait_observations_are_refused_rather_than_mirrored_wrong():
    """A symmetric gait is half a cycle out of phase on the mirrored side. Passing the phase
    through unchanged would be a silent error, so the map refuses the wider vector."""
    with pytest.raises(AssertionError, match="half-cycle phase shift"):
        pio.mirror_obs(np.zeros(pio.NUM_OBS_WITH_GAIT))


def test_the_mirror_map_matches_mujoco_physics():
    """THE load-bearing test. Everything above checks the map against its own conventions;
    this checks it against the simulator. Put the robot in a mirrored PHYSICAL state and
    every derived feature must come back mirrored, with no policy involved.

    A reflection is improper, so a rotation quaternion (w,x,y,z) becomes (w,x,-y,-z): the
    axis reflects AND the handedness flips. Getting only the first half right passes the
    involution test and fails here.
    """
    from config import get_cfgs
    from mj_vec_env import MjVecEnv

    env_cfg, obs_cfg, reward_cfg, command_cfg = get_cfgs()
    env_cfg.update(terrain_enabled=False, domain_rand_enabled=False, base_init_tilt_deg=0.0)
    env = MjVecEnv(num_envs=2, env_cfg=env_cfg, obs_cfg=obs_cfg, reward_cfg=reward_cfg,
                   command_cfg=command_cfg, device="cpu", num_threads=1, seed=0)
    env.reset()
    nq = env._base_model.nq
    rng = np.random.default_rng(0)
    q = rng.normal(size=4); q /= np.linalg.norm(q)
    jp = rng.normal(size=8) * 0.3 + 0.56
    jv = rng.normal(size=8) * 0.5
    lin, ang = rng.normal(size=3), rng.normal(size=3)
    perm = pio.MIRROR_JOINT_PERM

    def write(i, quat, jpos, jvel, linvel, angvel):
        env.state[i, 1:4] = [0.0, 0.0, 0.09]
        env.state[i, 4:8] = quat
        env.state[i, 1 + env._joint_qpos_adr] = jpos
        env.state[i, 1 + nq:1 + nq + 3] = linvel
        env.state[i, 1 + nq + 3:1 + nq + 6] = angvel
        env.state[i, 1 + nq + env._joint_dof_adr] = jvel

    write(0, q, jp, jv, lin, ang)
    write(1, [q[0], q[1], -q[2], -q[3]], jp[perm], jv[perm],
          [-lin[0], lin[1], lin[2]], [ang[0], -ang[1], -ang[2]])

    f = env._decode_state()
    for key, signs in (("projected_gravity", pio.MIRROR_GRAVITY_SIGNS),
                       ("base_ang_vel", pio.MIRROR_ANG_VEL_SIGNS),
                       ("base_lin_vel", pio.MIRROR_LIN_VEL_SIGNS)):
        np.testing.assert_allclose(f[key][1], f[key][0] * np.array(signs), atol=1e-9,
                                   err_msg=f"{key} does not mirror as declared")
    np.testing.assert_allclose(f["dof_pos"][1], f["dof_pos"][0][perm], atol=1e-9)
    env.close()


def test_group_mirror_leaves_privileged_alone_and_reshapes_history_correctly():
    rng = np.random.default_rng(2)
    frames = 4
    groups = {
        "policy": torch.from_numpy(rng.normal(size=(3, pio.NUM_OBS))),
        "history": torch.from_numpy(rng.normal(size=(3, frames * pio.NUM_OBS))),
        "estimate": torch.from_numpy(rng.normal(size=(3, 3))),
        "privileged": torch.from_numpy(rng.normal(size=(3, 46))),
    }
    out = pio.mirror_obs_groups(groups)
    torch.testing.assert_close(out["privileged"], groups["privileged"])
    # Each history FRAME must mirror independently; a bad reshape mixes them.
    h_in = groups["history"].reshape(3, frames, pio.NUM_OBS)
    h_out = out["history"].reshape(3, frames, pio.NUM_OBS)
    for k in range(frames):
        torch.testing.assert_close(h_out[:, k], pio.mirror_obs(h_in[:, k]))
    torch.testing.assert_close(pio.mirror_obs_groups(out)["history"], groups["history"])
