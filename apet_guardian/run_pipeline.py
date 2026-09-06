# Train driver
import argparse
import logging
import shutil
import sys
import time
from pathlib import Path

from . import config as C
from .pipeline import Pipeline

log = logging.getLogger("apet")

# Main
def main() -> int:
    ap = argparse.ArgumentParser(description="APET-GUARDIAN pipeline driver")
    ap.add_argument("--quick", action="store_true", help="small subset for a smoke run")
    ap.add_argument("--fresh", action="store_true", help="wipe the output dir before running")
    ap.add_argument("--out", type=str, default=str(C.OUT_ROOT))
    ap.add_argument("--normal", type=int, default=None)
    ap.add_argument("--faulty", type=int, default=None)
    ap.add_argument("--rul", type=int, default=None)
    ap.add_argument("--limit", type=int, default=None, help="apply to all three kinds")
    ap.add_argument("--log", type=str, default="INFO")
    args = ap.parse_args()

    logging.basicConfig(level=getattr(logging, args.log.upper(), logging.INFO),
                        format="%(asctime)s %(levelname)-7s %(name)s %(message)s",
                        datefmt="%H:%M:%S")

    out = Path(args.out)
    limits = dict(C.LIMITS)
    if args.limit:
        limits = {"normal": args.limit, "faulty": args.limit, "rul": args.limit}
    if args.normal:
        limits["normal"] = args.normal
    if args.faulty:
        limits["faulty"] = args.faulty
    if args.rul:
        limits["rul"] = args.rul
    if args.quick:
        limits = {"normal": min(limits["normal"], 24),
                  "faulty": min(limits["faulty"], 40),
                  "rul": min(limits["rul"], 6)}
    if args.fresh and out.exists():
        shutil.rmtree(out)

    t0 = time.time()
    pl = Pipeline(out_root=out, limits=limits)
    log.info("splitting by engine/life/mission -> %s", pl.assign_splits())
    physics = pl.build_physics_model()
    pl.build_windows(physics)
    report = pl.train_and_report()
    log.info("pipeline finished in %.1f s", time.time() - t0)

    te = report.get("test", {})
    print("\n================ APET-GUARDIAN TEST REPORT (stratified split) ================")
    for k, v in te.items():
        if isinstance(v, dict):
            print(f"[{k}]")
            for kk, vv in v.items():
                if isinstance(vv, dict):
                    print(f"    {kk:<34} " + ", ".join(f"{a}={float(b):.3f}" for a, b in vv.items()
                                                       if isinstance(b, (int, float))))
                else:
                    print(f"    {kk:<34} {float(vv):.3f}" if isinstance(vv, (int, float)) else f"    {kk}: {vv}")
    print("===============================================================================")
    print(f"artifacts -> {out / 'artifacts' / 'apet_bundle.pkl'}")
    print(f"report    -> {out / 'report.json'}")
    return 0

if __name__ == "__main__":
    sys.exit(main())
