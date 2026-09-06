# Validate pipeline
from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np

from . import config as C
from .pipeline import Pipeline
from . import training as T

ROOT = Path(C.OUT_ROOT).parent if (Path(C.OUT_ROOT) / "artifacts").exists() else Path(C.OUT_ROOT)
sys.path.insert(0, str(ROOT.resolve()))

import run_apet_guardian_pipeline as pipe

BASELINE_FILE = Path(C.OUT_ROOT) / "validate_baseline.json"

CONTRACT = {
    "anomaly_auc": 0.02,
    "anomaly_pr_auc": 0.02,
    "binary_f1": 0.02,
    "fault_f1_macro": 0.02,
    "healthy_fpr": 0.02,
    "healthy_fpr_alert": 0.03,
}

# Auc
def _auc(y, p):
    if len(np.unique(y)) < 2 or np.ptp(p) < 1e-12:
        return float("nan")
    from sklearn.metrics import roc_auc_score, average_precision_score
    return float(roc_auc_score(y, p)), float(average_precision_score(y, p))

# F1
def _f1(pred, truth):
    from sklearn.metrics import f1_score, precision_score, recall_score
    return (float(f1_score(truth, pred, zero_division=0)),
            float(precision_score(truth, pred, zero_division=0)),
            float(recall_score(truth, pred, zero_division=0)))

# Run
def run(use_gate: bool):
    with open(Path(C.OUT_ROOT) / "artifacts" / "apet_bundle.pkl", "rb") as f:
        art = pickle.load(f)
    for m in ["lstm", "fusion", "quantile", "health"]:
        if art.get(m) is not None:
            art[m] = art[m].to(T.DEVICE)

    pl = Pipeline(out_root=C.OUT_ROOT)
    pl.assign_splits()

    st_n_te = pl._load("normal", "test", stats=True)["stats"]
    st_f_te = pl._load("faulty", "test", stats=True)["stats"]
    seq_n_te = pl._load("normal", "test", seq=True)["seq"]
    seq_f_te = pl._load("faulty", "test", seq=True)["seq"]
    meta_f_te = pl._load("faulty", "test", meta=True)["meta"]

    st_scaler, seq_scaler = art["stat_scaler"], art["seq_scaler"]
    st_te = st_scaler.transform(np.vstack([st_n_te, st_f_te])).astype("float32")
    y_te = np.vstack([np.zeros((len(st_n_te), len(C.FAULTS)), dtype="float32"),
                      meta_f_te[:, -len(C.FAULTS):].astype("float32")])
    fault_any = y_te.sum(axis=1) > 0

    seq_te = np.concatenate([seq_n_te, seq_f_te])
    seq_te_s = seq_scaler.transform(seq_te.reshape(-1, seq_te.shape[-1])).reshape(*seq_te.shape)
    emb_te, prob_bin = T.lstm_predict(art["lstm"], seq_te_s)

    if_model, if_ref = art["if"]
    score_if = T.anomaly_score(if_model, st_te, if_ref)
    score = T.blended_anomaly(if_model, if_ref, art.get("anomaly_extra"), st_te, prob_bin)
    multi = T.fusion_predict(art["fusion"], st_te, score_if, emb_te)

    thr = np.array([pipe.class_threshold(f) for f in C.FAULTS])
    pred = multi > thr
    if use_gate:
        hot = multi[:, C.FAULTS.index("overheating")] > thr[C.FAULTS.index("overheating")]
        hotp = np.zeros(len(hot), dtype=bool)
        hotp[1:] = hot[:-1]
        pred[:, C.FAULTS.index("overheating")] = hot & hotp

    pred_any = pred.any(axis=1)
    alert = score >= pipe.ANOMALY_ALERT_THRESHOLD

    fault_auc, fault_pr = _auc(fault_any.astype(int), score)
    fault_any_gt = fault_any.astype(int)
    bf1, _, _ = _f1(pred_any, fault_any_gt)

    per = {}
    for i, f in enumerate(C.FAULTS):
        f1v, pr, rc = _f1(pred[:, i], y_te[:, i])
        per[f] = {"f1": f1v, "precision": pr, "recall": rc}

    n_norm = len(st_n_te)
    healthy_fpr = float(pred_any[:n_norm].mean())
    healthy_fpr_alert = float(alert[:n_norm].mean())
    fault_f1_macro = float(np.mean([per[f]["f1"] for f in C.FAULTS]))

    rul_gt = pl._load("normal", "test", meta=True)["meta"]
    rul_t = np.concatenate([rul_gt[:, 4], meta_f_te[:, 4]])
    heal_t = np.concatenate([rul_gt[:, 5], meta_f_te[:, 5]])
    rul_p = T.quantile_predict(art["quantile"], st_te)[:, 1]
    heal_p = T.health_predict(art["health"], st_te)
    ii = np.isfinite(rul_t)
    rul_mae = float(np.mean(np.abs(rul_t[ii] - rul_p[ii]))) if ii.sum() else float("nan")
    jj = np.isfinite(heal_t)
    heal_mae = float(np.mean(np.abs(heal_t[jj] - heal_p[jj]))) if jj.sum() else float("nan")

    return {
        "anomaly_auc": fault_auc,
        "anomaly_pr_auc": fault_pr,
        "binary_f1": bf1,
        "fault_f1_macro": fault_f1_macro,
        "healthy_fpr": healthy_fpr,
        "healthy_fpr_alert": healthy_fpr_alert,
        "overheating_f1": per["overheating"]["f1"],
        "overheating_precision": per["overheating"]["precision"],
        "overheating_recall": per["overheating"]["recall"],
        "rul_mae": rul_mae,
        "health_mae": heal_mae,
        "per_fault_f1": {f: per[f]["f1"] for f in C.FAULTS},
        "per_fault_precision": {f: per[f]["precision"] for f in C.FAULTS},
        "per_fault_recall": {f: per[f]["recall"] for f in C.FAULTS},
    }

# Fmt
def fmt(v: float) -> str:
    return "n/a" if v != v else f"{v:.3f}"

# Main
def main(argv: list | None = None):
    ap = argparse.ArgumentParser(description="APET-GUARDIAN regression harness")
    ap.add_argument("--update-baseline", action="store_true")
    ap.add_argument("--gate-off", action="store_true")
    args = ap.parse_args(argv)

    if args.gate_off:
        print("gate OFF metrics (raw fusion thresholds, no overheating gate):")
        off = run(use_gate=False)
        for k, v in off.items():
            if not isinstance(v, dict):
                print(f"  {k:<24s} {fmt(v)}")
        print()

    m = run(use_gate=True)
    print("\n================ APET-GUARDIAN REGRESSION HARNESS ================")
    print("leak-free grouped TEST windows - current deployed bundle")
    for k in ["anomaly_auc", "anomaly_pr_auc", "binary_f1", "fault_f1_macro",
              "healthy_fpr", "healthy_fpr_alert", "overheating_f1",
              "overheating_precision", "overheating_recall", "rul_mae", "health_mae"]:
        print(f"  {k:<24s} {fmt(m[k])}")
    print("\n  per-fault F1 (deployed decision):")
    for f, v in m["per_fault_f1"].items():
        print(f"    {f:<22s} {fmt(v)}")

    base_p = BASELINE_FILE
    if args.update_baseline or not base_p.exists():
        base_p.write_text(json.dumps(m, indent=2, sort_keys=True, default=float))
        print(f"\n[validate] baseline written: {base_p}")
        print("(no regression check on the first run)")
        return

    base = json.loads(base_p.read_text())
    fails = []
    print(f"\n  regression check vs {base_p.name}:")
    for k, tol in CONTRACT.items():
        cur, ref = m[k], base.get(k, float("nan"))
        drop = ref - cur if np.isfinite(ref) and np.isfinite(cur) else float("nan")
        state = "FAIL" if np.isfinite(drop) and drop > tol else ("n/a" if not np.isfinite(drop) or not np.isfinite(ref) else "ok")
        if state == "FAIL":
            fails.append(k)
        print(f"    {k:<24s} baseline={fmt(ref)} current={fmt(cur)} diff={fmt(-drop):>8s}  {state}")
    if fails:
        print(f"\n[validate] REGRESSION in: {', '.join(fails)}")
        raise SystemExit(1)
    print("\n[validate] all contracted metrics hold.")

if __name__ == "__main__":
    main()
