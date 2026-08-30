"""The YAML config loader for mj_train.

These exist because every failure mode here is SILENT on the training side: a typo'd key,
an overlay that quietly drops the base's reward scales, or a config that beats the command
line would all produce a run that looks like it used your settings and did not. You would
find out a GPU-hour later, from a policy that scores wrong for no visible reason.
"""

import pathlib
import re
import sys

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "sim"))
import train_config  # noqa: E402

CONFIGS = pathlib.Path(__file__).resolve().parents[1] / "configs"

# A representative slice of mj_train's real destinations.
DESTS = {"num_envs", "max_iterations", "seed", "obs_history", "terrain", "terrain_amplitude",
         "terrain_correlation", "penalty_curriculum", "lipschitz_coef", "privileged_critic",
         "estimator", "command_vx", "heading", "ang_vel_range", "tracking_sigma",
         "tracking_sigma_ang", "reward_scale", "exp_name", "config"}


def write(tmp_path, name, text):
    p = tmp_path / name
    p.write_text(text)
    return p


def test_shipped_configs_all_load():
    """Every config in configs/ must validate against mj_train's real flags. This is what
    catches a config left stranded by a renamed flag -- otherwise the rename lands, the
    config keeps its old key, and the setting silently reverts to a default.

    Read out of the source rather than by importing mj_train, which would pull in torch,
    mujoco and rsl_rl to look at a list of strings.
    """
    src = (pathlib.Path(__file__).resolve().parents[1] / "sim" / "mj_train.py").read_text()
    dests = {flag[2:].replace("-", "_") for flag in re.findall(r'"(--[a-z0-9-]+)"', src)}
    assert "obs_history" in dests, "flag extraction broke, not the configs"

    for cfg in sorted(CONFIGS.glob("*.yaml")):
        train_config.load(cfg, dests)          # raises ConfigError on anything unknown


def test_objective_inherits_the_base_recipe():
    cfg = train_config.load(CONFIGS / "walk.yaml", DESTS)
    assert cfg["max_iterations"] == 5000, "the 5000-iteration ceiling must survive the overlay"
    assert cfg["estimator"] is True
    assert cfg["command_vx"] == [0.1], "the objective's own setting must win"


def test_reward_scales_merge_key_by_key():
    """An overlay retuning ONE term must not silently drop the other nine. Whole-key
    replacement here would train against a completely different reward and look fine."""
    import tempfile
    with tempfile.TemporaryDirectory() as d:
        d = pathlib.Path(d)
        write(d, "base.yaml", "reward_scale:\n  a: -1.0\n  b: -2.0\n")
        over = write(d, "over.yaml", "extends: base.yaml\nreward_scale:\n  b: -9.0\n")
        cfg = train_config.load(over, {"reward_scale"})
    assert cfg["reward_scale"] == ["a=-1.0", "b=-9.0"]


def test_unknown_key_is_refused_with_a_suggestion():
    import tempfile
    with tempfile.TemporaryDirectory() as d:
        bad = write(pathlib.Path(d), "bad.yaml", "obs_histry: 12\n")
        with pytest.raises(train_config.ConfigError, match="obs_history"):
            train_config.load(bad, DESTS)


def test_circular_extends_is_caught():
    import tempfile
    with tempfile.TemporaryDirectory() as d:
        d = pathlib.Path(d)
        write(d, "a.yaml", "extends: b.yaml\n")
        write(d, "b.yaml", "extends: a.yaml\n")
        with pytest.raises(train_config.ConfigError, match="circular"):
            train_config.load(d / "a.yaml", DESTS)


def test_reward_scale_must_be_a_mapping():
    import tempfile
    with tempfile.TemporaryDirectory() as d:
        bad = write(pathlib.Path(d), "bad.yaml", "reward_scale: -0.05\n")
        with pytest.raises(train_config.ConfigError, match="mapping"):
            train_config.load(bad, DESTS)
