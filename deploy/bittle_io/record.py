"""Run directories, manifests, and CSV/JSON output.

Every experiment writes into one timestamped run directory so that a characterization
session is a single reviewable artifact. Raw serial transcripts are saved alongside the
parsed results: when the board does something the parsers did not expect, the bytes that
caused it are still on disk.
"""

import csv
import json
import platform
import time
from datetime import datetime
from pathlib import Path

# Repo-root measurements/, NOT deploy/logs/hw. Hardware sessions are hand-collected and
# unreproducible -- a robot, a battery charge and an afternoon each -- so they live in one
# named place at the top of the tree rather than under whichever tool happened to write
# them. Resolved from THIS FILE, not the cwd, so the path does not depend on where the
# command was launched from. See measurements/README.md.
DEFAULT_ROOT = Path(__file__).resolve().parents[2] / "measurements"


class Recorder:
    def __init__(self, root=None, run_id=None, dry_run=False):
        self.dry_run = dry_run
        root = Path(root) if root else DEFAULT_ROOT
        self.run_id = run_id or datetime.now().strftime("%Y%m%d-%H%M%S")
        self.dir = root / self.run_id
        if not dry_run:
            self.dir.mkdir(parents=True, exist_ok=True)
        self.manifest = {
            "run_id": self.run_id,
            "started": datetime.now().isoformat(timespec="seconds"),
            "host": platform.node(),
            "experiments": {},
        }
        # MERGE with an existing manifest: a characterization session is many CLI
        # invocations sharing one --run-id, and each used to overwrite manifest.json
        # wholesale -- the last command silently clobbered every earlier experiment's
        # results (the CSVs survived; the summary numbers did not).
        prior = self.dir / "manifest.json"
        if not dry_run and prior.exists():
            try:
                old = json.loads(prior.read_text())
                self.manifest["started"] = old.get("started", self.manifest["started"])
                self.manifest["experiments"] = old.get("experiments", {})
            except json.JSONDecodeError:
                pass

    # -- writers -----------------------------------------------------------
    def write_json(self, name, obj):
        path = self.dir / name
        if self.dry_run:
            print(f"[dry-run] would write {path}")
            return path
        path.write_text(json.dumps(obj, indent=2, default=_jsonable) + "\n")
        return path

    def write_csv(self, name, rows, fieldnames=None):
        path = self.dir / name
        rows = list(rows)
        if self.dry_run:
            print(f"[dry-run] would write {path} ({len(rows)} rows)")
            return path
        if not rows:
            path.write_text("")
            return path
        fieldnames = fieldnames or list(rows[0].keys())
        with path.open("w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=fieldnames)
            w.writeheader()
            w.writerows(rows)
        return path

    def write_transcript(self, name, lines):
        """Raw serial lines, exactly as received."""
        path = self.dir / name
        if self.dry_run:
            print(f"[dry-run] would write {path}")
            return path
        with path.open("w") as fh:
            for item in lines:
                fh.write((item[1] if isinstance(item, tuple) else item) + "\n")
        return path

    # -- manifest ----------------------------------------------------------
    def note(self, experiment, **fields):
        """Record an experiment's headline numbers into the run manifest."""
        self.manifest["experiments"].setdefault(experiment, {}).update(fields)

    def bracket_voltage(self, experiment, link):
        """Context manager logging battery voltage either side of an experiment.

        Servo torque sags as the battery drains, so an unlogged voltage drop across a long
        session shows up as fake drift in whatever was being measured.
        """
        return _VoltageBracket(self, experiment, link)

    def finish(self):
        self.manifest["finished"] = datetime.now().isoformat(timespec="seconds")
        return self.write_json("manifest.json", self.manifest)


class _VoltageBracket:
    def __init__(self, rec, experiment, link):
        self.rec, self.experiment, self.link = rec, experiment, link

    def __enter__(self):
        self.t0 = time.time()
        self.v0 = self.link.voltage()
        self.rec.note(self.experiment, voltage_start=self.v0)
        return self

    def __exit__(self, *exc):
        v1 = self.link.voltage()
        self.rec.note(self.experiment, voltage_end=v1,
                      duration_s=round(time.time() - self.t0, 2))
        if self.v0 is not None and v1 is not None and self.v0 - v1 > 0.3:
            print(f"  ! battery fell {self.v0:.2f} -> {v1:.2f} V during this run; "
                  "servo torque is not comparable across it")
        return False


def _jsonable(o):
    if hasattr(o, "tolist"):
        return o.tolist()
    if isinstance(o, Path):
        return str(o)
    return str(o)
