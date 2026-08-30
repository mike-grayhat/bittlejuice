"""Command-line entry point.

    uv run python -m bittle_io --port /dev/cu.usbmodem5AA90270111 probe

SAFETY
  * Robot SUSPENDED with legs free, unless an experiment says otherwise
    (`imu --phase dynamics` needs it on the ground).
  * Battery ON and USB connected.
  * Servos overheat when fighting. Short runs; touch the cases between them.
"""

import argparse
import sys

from . import probe as probe_mod
from . import report as report_mod
from .experiments import DEFAULT_ORDER, EXPERIMENTS, MODULES
from .link import LinkError, SerialLink, find_port
from .record import Recorder


def _confirm_calibration(text):
    print(f"\n!! {text!r} enters CALIBRATION and can overwrite servo offsets.")
    try:
        return input("!! type CALIBRATE to confirm: ").strip() == "CALIBRATE"
    except EOFError:
        return False


def build_parser():
    p = argparse.ArgumentParser(
        prog="bittle_io", description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--port", default=None, help="default: first usbmodem-like device")
    p.add_argument("--baud", type=int, default=115200)
    p.add_argument("--dry-run", action="store_true",
                   help="never open the port; print the token sequence instead")
    p.add_argument("--run-id", default=None, help="reuse an existing run directory")
    p.add_argument("--out", default=None,
                   help="run root (default: the repo's measurements/ directory)")

    sub = p.add_subparsers(dest="cmd", required=True)

    pr = sub.add_parser("probe", help="discover firmware capabilities (run this FIRST)")
    pr.add_argument("--seconds", type=float, default=4.0)

    for mod in MODULES:
        sp = sub.add_parser(mod.NAME, help=mod.HELP)
        mod.add_args(sp)

    # `all` deliberately does NOT accept the experiments' own flags: several of them
    # define --seconds/--joint with different meanings, so a merged namespace would be
    # both a name collision and ambiguous to read. Each experiment runs with its own
    # defaults; tune one by running it individually.
    al = sub.add_parser("all", help="probe, then every experiment in order (defaults only)")
    al.add_argument("--skip", nargs="*", default=[], choices=list(EXPERIMENTS))

    rp = sub.add_parser("report", help="aggregate a run directory")
    rp.add_argument("run_dir")

    sub.add_parser("ports", help="list candidate serial ports")
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)

    if args.cmd == "report":
        report_mod.run(args.run_dir)
        return 0

    if args.cmd == "ports":
        try:
            print(find_port(args.port))
        except LinkError as e:
            print(e, file=sys.stderr)
            return 1
        return 0

    recorder = Recorder(root=args.out, run_id=args.run_id, dry_run=args.dry_run)
    print(f"run: {recorder.dir}")

    try:
        link = SerialLink(port=args.port, baud=args.baud, dry_run=args.dry_run,
                          confirm_calibration=_confirm_calibration)
    except LinkError as e:
        print(f"error: {e}", file=sys.stderr)
        return 1

    print(f"connected: {link.port}\n")
    rc = 0
    try:
        with link:
            # REALTIME MODE FOR THE WHOLE SESSION, except probe, which owns the switch
            # because testing it is the gate probe exists to be.
            #
            # This is not only about the IMU stream. The firmware's interpolation ramp
            # BLOCKS its main loop for 16-24 ms per pose command -- longer for a big move --
            # so any experiment that streams motion faster than that overruns the serial
            # buffer and wedges the board. `cmdrate --hz 50` does exactly that by design,
            # and `sweep`/`bodyid` drive at 30-50 Hz. Measured: cmdrate against stock
            # behaviour left the board emitting empty lines and answering nothing.
            #
            # The measured constants in sim/ were all taken with the ramp off, so this is
            # also what makes a new session comparable to measurements/reference/.
            if args.cmd != "probe" and not args.dry_run:
                if link.realtime(True) is None:
                    print("  ! realtime mode unsupported -- unpatched firmware. Motion "
                          "commands will be interpolated and the IMU stream capped at "
                          "5 Hz; streaming experiments may overrun the board.\n")
                else:
                    print("  realtime mode ON for this session\n")
            if args.cmd == "probe":
                probe_mod.run(link, recorder, seconds=args.seconds)
            elif args.cmd == "all":
                caps = probe_mod.run(link, recorder, seconds=4.0)
                for name in DEFAULT_ORDER:
                    if name in args.skip:
                        print(f"\n=== {name}: skipped ===")
                        continue
                    print(f"\n=== {name} ===")
                    try:
                        EXPERIMENTS[name].run(link, caps, _defaults_for(name), recorder)
                    except Exception as e:
                        # One failed experiment must not discard the rest of the session.
                        print(f"  {name} FAILED: {e}")
                        recorder.note(name, error=str(e))
                        rc = 1
            else:
                caps = _load_caps(recorder)
                EXPERIMENTS[args.cmd].run(link, caps, args, recorder)
    except KeyboardInterrupt:
        print("\ninterrupted")
        rc = 130
    finally:
        recorder.finish()
        if args.dry_run:
            print("\n--- tokens that would have been sent ---")
            for t in link.sent:
                print(f"  {t}")

    if not args.dry_run:
        print(f"\nrun directory: {recorder.dir}")
        print(f"summarize with: python -m bittle_io report {recorder.dir}")
    return rc


def _defaults_for(name):
    """That experiment's own default namespace, used by `all`."""
    p = argparse.ArgumentParser(add_help=False)
    EXPERIMENTS[name].add_args(p)
    return p.parse_args([])


def _load_caps(recorder):
    """Reuse capabilities.json from this run dir if present, else probe defaults.

    Experiments consult caps for the feedback rate (to warn about Nyquist) and whether the
    accelerometer exists, so running one standalone without a probe is allowed but noted.
    """
    import json

    from .probe import Capabilities

    path = recorder.dir / "capabilities.json"
    if path.exists():
        data = json.loads(path.read_text())
        known = {f for f in Capabilities.__dataclass_fields__}
        return Capabilities(**{k: v for k, v in data.items() if k in known})
    print("note: no capabilities.json in this run dir -- run `probe` for rate-aware checks")
    return Capabilities()


if __name__ == "__main__":
    sys.exit(main())
