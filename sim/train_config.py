"""YAML training configs for mj_train.

The recipe in docs/training.md is twenty flags across eight lines, and every seed of every
sweep re-types it. That is not just tedious: a flag dropped by accident is invisible, and
two runs that differ by one character look identical in a shell history. A config file makes
the recipe a reviewable object, and `git diff configs/walk.yaml configs/run.yaml` answers
what actually separates two objectives.

Three rules, each of which exists because the obvious alternative goes wrong:

CONFIG IS INPUT, THE RUN DIRECTORY IS OUTPUT. Configs live in configs/ and are tracked;
runs/<exp>/ is written once and never edited. A config that lived in the run directory
could always have been changed after the checkpoints were written, and nothing would say
so -- provenance you cannot trust is worse than none, because you will trust it.

THE COMMAND LINE WINS. Values here are argparse DEFAULTS, so any flag still overrides them.
That is what keeps a four-seed pool as one config plus `--seed $S` instead of four files
that drift apart.

UNKNOWN KEYS ARE AN ERROR. A typo'd key that is silently ignored produces a run that looks
like it used your setting and did not. Every key is checked against the parser's own
destinations, and a near-miss is suggested.
"""

import difflib
import pathlib

import yaml


class ConfigError(Exception):
    """Raised for anything wrong in a config file, with the file named."""


#: Keys that are structure rather than training settings.
_META = {"extends", "description"}


def _read(path):
    try:
        raw = yaml.safe_load(path.read_text())
    except FileNotFoundError:
        raise ConfigError(f"{path}: no such config file") from None
    except yaml.YAMLError as e:
        raise ConfigError(f"{path}: not valid YAML -- {e}") from None
    if raw is None:
        return {}
    if not isinstance(raw, dict):
        raise ConfigError(f"{path}: top level must be a mapping, got {type(raw).__name__}")
    return raw


def _resolve(path, seen):
    """Load `path`, applying `extends:` beneath it. Returns a flat dict.

    Single inheritance, resolved relative to the extending file, so a config can say
    `extends: base.yaml` without knowing where the tree is rooted. Deliberately shallow:
    an objective overlay overrides whole keys, and `reward_scale` merges key by key so an
    overlay can retune one term without restating the other nine.
    """
    path = path.resolve()
    if path in seen:
        chain = " -> ".join(p.name for p in seen) + f" -> {path.name}"
        raise ConfigError(f"circular extends: {chain}")
    seen.append(path)

    raw = _read(path)
    base_name = raw.pop("extends", None)
    if base_name is None:
        return raw

    merged = _resolve(path.parent / base_name, seen)
    scales = {**merged.get("reward_scale", {}), **raw.get("reward_scale", {})}
    merged.update(raw)
    if scales:
        merged["reward_scale"] = scales
    return merged


def load(path, valid_dests):
    """Read a config into a dict of argparse destinations.

    `valid_dests` is the parser's own set of destinations, so this cannot drift out of sync
    with the flags -- a renamed flag makes every config naming the old key fail loudly on
    the next run rather than silently reverting to a default.
    """
    path = pathlib.Path(path)
    cfg = _resolve(path, [])

    unknown = sorted(set(cfg) - valid_dests - _META)
    if unknown:
        lines = []
        for key in unknown:
            near = difflib.get_close_matches(key, sorted(valid_dests), n=1)
            lines.append(f"  {key!r}" + (f"  -- did you mean {near[0]!r}?" if near else ""))
        raise ConfigError(
            f"{path}: unknown key(s):\n" + "\n".join(lines)
            + "\nKeys are argparse destinations: the flag without '--', hyphens as "
              "underscores (--obs-history -> obs_history)."
        )

    for key in _META:
        cfg.pop(key, None)

    # argparse takes --reward-scale as repeatable NAME=VALUE strings; a mapping is the
    # readable form, and the only one an overlay can merge into key by key.
    scales = cfg.get("reward_scale")
    if isinstance(scales, dict):
        cfg["reward_scale"] = [f"{k}={v}" for k, v in sorted(scales.items())]
    elif scales is not None and not isinstance(scales, list):
        raise ConfigError(f"{path}: reward_scale must be a mapping of NAME: VALUE")

    return cfg
