# Predict CLI
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from . import config as C
from .inference import APETGuardian

# Stream
def stream(filename: str, guardian: APETGuardian, every: int = 1, max_rows: bool = True):
    df = pd.read_csv(filename, low_memory=False)
    print(f"predicting {filename} ({len(df)} rows, dt={C.SAMPLE_DT:.1f}s)")
    n = 0
    for end in range(C.SEQ_LEN, len(df) + 1, C.STRIDE):
        sub = df.iloc[:end]
        if "phase" not in sub.columns:
            sub = sub.assign(phase="cruise")
        try:
            out = guardian.predicts(sub)
        except Exception as e:
            print(f"  window@row{end}: ERROR {e}")
            continue
        if n % every == 0:
            fl = " | ".join(f"{r['class']}:{r['confidence']:.2f}" for r in out["probable_faults"][:3])
            ru = out["rul_hours"]
            ru_s = f"RUL {ru['mean']:.2f}" + (f" [{ru['lower']:.2f},{ru['upper']:.2f}]" if ru["upper"] else "") if ru["mean"] is not None else "RUL n/a"
            print(f"  t={out['snapshot']['time_s']:7.1f}s phase={out['snapshot']['phase']:<7s} "
                  f"anom={out['anomaly_score']:.3f} faultP={out['fault_probability']:.3f} "
                  f"health={out['health_index'] if out['health_index'] is not None else 'n/a'} "
                  f"{ru_s} | {fl}")
        n += 1
    print(f"  windows: {n}\n")

# Write Csv
def write_csv(filename: str, guardian: APETGuardian, out_path: Path) -> dict:
    df = pd.read_csv(filename, low_memory=False)
    if "time_in_mission_s" not in df.columns:
        if "global_time_s" in df.columns:
            df = df.assign(time_in_mission_s=df["global_time_s"] - df["global_time_s"].min())
        else:
            df = df.assign(time_in_mission_s=np.arange(len(df)) * C.SAMPLE_DT)
    if "phase" not in df.columns:
        df = df.assign(phase="cruise")

    rows = []
    for idx, end in enumerate(range(C.SEQ_LEN, len(df) + 1, C.STRIDE)):
        sub = df.iloc[:end]
        out = guardian.predicts(sub)
        confs = {r["class"]: r["confidence"] for r in out["probable_faults"]}
        rows.append({
            "window_index": idx,
            "window_end_row": int(end),
            "time_s": out["snapshot"]["time_s"],
            "phase": out["snapshot"]["phase"],
            "engine_id": out["snapshot"]["engine_id"],
            "mission_id": out["snapshot"]["mission_id"],
            "anomaly_score": out["anomaly_score"],
            "fault_probability": out["fault_probability"],
            "normal_probability": out["normal_probability"],
            "active_faults": ";".join(out["active_faults"]) or "none",
            **{f"prob_{n}": confs.get(n) for n in guardian.faults},
            "rul_mean_h": out["rul_hours"]["mean"],
            "rul_lower_h": out["rul_hours"]["lower"],
            "rul_upper_h": out["rul_hours"]["upper"],
            "health_index": out["health_index"],
        })
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out_path, index=False)
    return {"file": str(out_path), "windows": len(rows)}

# Main
def main() -> int:
    ap = argparse.ArgumentParser(description="stream APET-GUARDIAN predictions")
    ap.add_argument("files", nargs="+", help="one or more telemetry CSVs")
    ap.add_argument("--bundle", type=str, default=None)
    ap.add_argument("--every", type=int, default=1)
    ap.add_argument("--json", action="store_true", help="dump raw prediction dicts as JSON lines")
    ap.add_argument("--csv-out", type=str, default=None,
                    help="write one per-window prediction CSV per input into this directory")
    args = ap.parse_args()

    guardian = APETGuardian(Path(args.bundle) if args.bundle else None)
    if args.csv_out:
        csv_dir = Path(args.csv_out)
        csv_dir.mkdir(parents=True, exist_ok=True)
        for fn in args.files:
            out_path = csv_dir / (Path(fn).stem + "_predictions.csv")
            info = write_csv(fn, guardian, out_path)
            print(f"wrote {info['windows']} windows -> {info['file']}")
        return 0
    for fn in args.files:
        if args.json:
            df = pd.read_csv(fn, low_memory=False)
            for end in range(C.SEQ_LEN, len(df) + 1, C.STRIDE):
                sub = df.iloc[:end]
                if "phase" not in sub.columns:
                    sub = sub.assign(phase="cruise")
                out = guardian.predicts(sub)
                out["row_end"] = int(end)
                print(json.dumps(out, default=str))
        else:
            stream(fn, guardian, every=args.every)
    return 0

if __name__ == "__main__":
    sys.exit(main())
