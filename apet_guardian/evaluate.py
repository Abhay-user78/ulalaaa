# Evaluation harness
import argparse
import json
import math
import pickle
import sys
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import warnings

import numpy as np
import pandas as pd
import torch

from . import config as C
from .windowing import compute_split, window_sequence, window_stats
from .inference import DEVICE
from . import training as T

MODEL_VERSION_BASE = "apet-guardian/1.0.0"

FROZEN_MULTILABEL_THRESHOLD = 0.5
FROZEN_BINARY_THRESHOLD = 0.5
ANOMALY_ALERT_THRESHOLD = 0.8
RUL_MIN_HOURS_RISK = 2.0
HEALTH_RISK_LEVEL = 50.0

RAW_SENSOR_COLS = [
    "throttle_actual", "true_rpm", "air_density_kg_m3", "altitude_m",
    "rpm", "boost_pressure_kPa", "manifold_pressure_kPa", "cht_cyl_avg_C",
    "egt_cyl_avg_C", "oil_pressure_kPa", "oil_temp_C", "coolant_temp_C",
    "fuel_flow_g_s", "fuel_temp_C", "rail_pressure_bar", "vib_rms_g",
    "battery_voltage_V", "airspeed_mps", "vertical_speed_mps",
    "engine_power_command_kW", "true_brake_power_kW",
]

IDENTIFIER_COLS = ["mission_id", "engine_id", "global_time_s", "time_in_mission_s"]

UNIT_BOUNDS = {
    "rpm": (0.0, 7000.0), "true_rpm": (0.0, 7000.0), "throttle_actual": (0.0, 1.0),
    "boost_pressure_kPa": (0.0, 600.0), "manifold_pressure_kPa": (0.0, 600.0),
    "cht_cyl_avg_C": (-50.0, 500.0), "egt_cyl_avg_C": (0.0, 1200.0),
    "oil_pressure_kPa": (0.0, 1500.0), "oil_temp_C": (-50.0, 300.0),
    "coolant_temp_C": (-50.0, 300.0), "fuel_flow_g_s": (0.0, 200.0),
    "fuel_temp_C": (-50.0, 300.0), "rail_pressure_bar": (0.0, 500.0),
    "vib_rms_g": (0.0, 100.0), "battery_voltage_V": (5.0, 60.0),
    "airspeed_mps": (0.0, 400.0), "altitude_m": (-500.0, 50000.0),
    "vertical_speed_mps": (-500.0, 500.0), "engine_power_command_kW": (0.0, 1500.0),
    "true_brake_power_kW": (0.0, 1500.0), "air_density_kg_m3": (0.1, 2.2),
}

GT_FAULT_COLUMNS = C.FAULT_COLUMNS

# Read Dataset
def _read_dataset(path: Path, cols: List[str]) -> pd.DataFrame:
    if path.suffix.lower() == ".parquet":
        avail = pd.read_parquet(path, columns=None).columns.tolist()
        return pd.read_parquet(path, columns=[c for c in cols if c in avail])
    return pd.read_csv(path, usecols=lambda c: c in cols, low_memory=False)

# Kind Of
def _kind_of(mission_id: str) -> str:
    m = str(mission_id)
    if m.startswith("life_") or m.startswith("engine_") or "life" in m.lower():
        return "rul"
    if "normal" in m.lower():
        return "normal"
    return "faulty"

# Load Bundle
def load_bundle(bundle_path: Path):
    with open(bundle_path, "rb") as f:
        art = pickle.load(f)
    mtime = datetime.fromtimestamp(bundle_path.stat().st_mtime)
    version = f"{MODEL_VERSION_BASE}/bundle-{mtime:%Y%m%d-%H%M%S}"
    return art, version

# Known Split Lookup
def known_split_lookup() -> Dict[str, str]:
    out, kinds = {}, {}
    for kind, kf in (("normal", 0), ("faulty", 1), ("rul", 2)):
        root = C.DATA_ROOT / kind
        if not root.exists():
            continue
        files = sorted(list(root.rglob("*.parquet")) + list(root.rglob("*.csv")))
        if not files:
            continue
        ids = []
        for f in files:
            stem = f.stem
            if kind == "rul":
                stem = "engine_" + stem.replace("life_", "")
            ids.append(stem)
        assign = compute_split(ids, seed=C.SPLIT_SEED + 10 * kf)
        for gid, sp in assign.items():
            out[gid] = sp
            kinds[gid] = kind
    return out, kinds

# Qualityreporter
class QualityReporter:
    # Init
    def __init__(self):
        self.per_file = {}
        self.mission_warnings = defaultdict(list)
        self.global_flags = defaultdict(int)

    # File
    def file(self, name: str, notes: List[str]):
        self.per_file[name] = {"warnings": list(dict.fromkeys(notes)), "count": len(set(notes))}

    # Mission
    def mission(self, mission_id: str, note: str):
        if note not in self.mission_warnings[mission_id]:
            self.mission_warnings[mission_id].append(note)

    # Flag
    def flag(self, key: str, n: int = 1):
        self.global_flags[key] += n

    # To Dict
    def to_dict(self) -> dict:
        return {
            "per_file": self.per_file,
            "per_mission_warnings": {k: v for k, v in self.mission_warnings.items()},
            "global_counts": dict(self.global_flags),
        }

# Validate And Clean
def validate_and_clean(df: pd.DataFrame, path: Path, qr: QualityReporter) -> pd.DataFrame:
    df = df.copy()
    for c in ["mission_id", "engine_id", "phase"]:
        if c in df.columns:
            df[c] = df[c].astype(str)

    notes = []
    if df.empty:
        qr.file(path.stem, ["empty file"])
        return df

    n_dup = int(df.duplicated().sum())
    if n_dup:
        df = df.drop_duplicates(keep="first")
        notes.append(f"removed {n_dup} duplicate rows")
        qr.flag("duplicate_rows_removed", n_dup)

    if "time_in_mission_s" not in df.columns:
        df["time_in_mission_s"] = df.groupby("mission_id")["global_time_s"].transform(
            lambda s: s - s.min())
        notes.append("derived time_in_mission_s from global_time_s")

    if "phase" not in df.columns:
        df["phase"] = "unknown"
        notes.append("missing phase column (defaulted to 'unknown')")

    num_cols = df.select_dtypes(include=[np.number]).columns
    gt_sparse = {"simulated_rul_hours", "true_overall_health_index"}
    impute_cols = [c for c in num_cols if c not in gt_sparse]
    miss = int(df[impute_cols].isna().sum().sum()) if impute_cols else 0
    if miss:
        notes.append(f"imputed {miss} missing numeric sensor/flag values (ffill/bfill)")
        qr.flag("missing_values_imputed", miss)
        df[num_cols] = df[num_cols].ffill().bfill()

    for col, (lo, hi) in UNIT_BOUNDS.items():
        if col not in df.columns:
            continue
        v = pd.to_numeric(df[col], errors="coerce")
        frac = float(int((v < lo).sum() + (v > hi).sum())) / max(len(v), 1)
        if frac > 1e-3:
            notes.append(f"unit/range anomaly: {col} {frac:.1%} outside [{lo},{hi}]")
            qr.flag("unit_range_warnings", 1)

    df = df.sort_values(
        ["engine_id", "mission_id", "global_time_s"], kind="mergesort").reset_index(drop=True)
    descents = 0
    for _, g in df.groupby(["engine_id", "mission_id"]):
        t = g["global_time_s"].to_numpy(dtype="float64")
        descents += int(np.sum(np.diff(t) < 0))
    if descents:
        notes.append(f"re-sorted rows; {descents} within-mission timestamp descents fixed")
        qr.flag("timestamp_descents", descents)

    qr.file(path.stem, notes)
    return df

# Sampling Check
def sampling_check(df: pd.DataFrame, qr: QualityReporter):
    for mid, g in df.groupby("mission_id", sort=False):
        t = g.sort_values("time_in_mission_s")["time_in_mission_s"].to_numpy(dtype="float64")
        if len(t) < 3:
            qr.mission(str(mid), "fewer than 3 rows; cadence not checkable")
            continue
        dt = np.diff(t)
        med = float(np.median(dt))
        if not (0.35 <= med <= 0.75):
            qr.mission(str(mid), f"median sample interval {med:.3f}s not near 2 Hz")
            qr.flag("missions_not_2hz", 1)
        else:
            qr.global_flags["2hz_like_missions"] += 1

# Build Windows For Mission
def build_windows_for_mission(mission_id, sub: pd.DataFrame, seq_features: List[str]):
    sub = sub.sort_values("time_in_mission_s")
    seq_mat = sub[seq_features].to_numpy(dtype="float32")
    time_s = sub["time_in_mission_s"].to_numpy(dtype="float64")
    windows, starts, ends = window_sequence(seq_mat, time_s=time_s)
    if len(windows) == 0:
        return None
    pos = sub.index.to_numpy()
    phase = sub["phase"].to_numpy(dtype=object)
    return windows, time_s[starts], time_s[ends], pos[ends], phase[ends]

# Binary Metrics
def binary_metrics(y, prob) -> Optional[Dict]:
    from sklearn.metrics import (accuracy_score, average_precision_score,
                                 balanced_accuracy_score, confusion_matrix, f1_score,
                                 precision_recall_curve, precision_score, roc_auc_score, roc_curve)
    y = np.asarray(y, dtype=int)
    p = np.asarray(prob, dtype="float64")
    if len(np.unique(y)) < 2:
        return None
    pred = (p > FROZEN_BINARY_THRESHOLD).astype(int)
    cm = confusion_matrix(y, pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    fprc, tpr, _ = roc_curve(y, p)
    pxx, rxx, _ = precision_recall_curve(y, p)
    return {
        "roc_auc": float(roc_auc_score(y, p)),
        "pr_auc": float(average_precision_score(y, p)),
        "accuracy": float(accuracy_score(y, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
        "precision": float(precision_score(y, pred, zero_division=0)),
        "recall": float(tp / max(tp + fn, 1e-9)),
        "f1": float(f1_score(y, pred, zero_division=0)),
        "false_positive_rate": float(fp / max(fp + tn, 1e-9)),
        "false_negative_rate": float(fn / max(fn + tp, 1e-9)),
        "confusion_matrix": cm.tolist(),
        "n_samples": int(len(y)),
        "n_positive": int(y.sum()),
        "roc_curve": [fprc.tolist(), tpr.tolist()],
        "pr_curve": [pxx.tolist(), rxx.tolist()],
    }

# Multilabel Metrics
def multilabel_metrics(y, prob, labels) -> Optional[Dict]:
    from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                                 hamming_loss, precision_recall_fscore_support,
                                 precision_score, recall_score, roc_auc_score)
    y = np.asarray(y, dtype=int)
    p = np.asarray(prob, dtype="float64")
    if len(y) == 0 or y.sum() == 0:
        return None
    pred = (p > FROZEN_MULTILABEL_THRESHOLD).astype(int)
    per, rec, f1v, _ = precision_recall_fscore_support(y, pred, average=None, zero_division=0)
    aucs, cm8 = {}, []
    for i, lab in enumerate(labels):
        yi, pi = y[:, i], p[:, i]
        aucs[lab] = float(roc_auc_score(yi, pi)) if len(np.unique(yi)) > 1 else None
        cm8.append(confusion_matrix(yi, (pi > FROZEN_MULTILABEL_THRESHOLD).astype(int),
                                    labels=[0, 1]).tolist())
    return {
        "labels": labels,
        "per_label": {lab: {"precision": float(pr), "recall": float(rc), "f1": float(f1t),
                            "auc_roc": aucs[lab]}
                      for lab, pr, rc, f1t in zip(labels, per, rec, f1v)},
        "exact_match": float(accuracy_score(y, pred)),
        "hamming_loss": float(hamming_loss(y, pred)),
        "macro_f1": float(f1_score(y, pred, average="macro", zero_division=0)),
        "micro_f1": float(f1_score(y, pred, average="micro", zero_division=0)),
        "samples_f1": float(f1_score(y, pred, average="samples", zero_division=0)),
        "macro_precision": float(precision_score(y, pred, average="macro", zero_division=0)),
        "macro_recall": float(recall_score(y, pred, average="macro", zero_division=0)),
        "per_fault_confusion": cm8,
        "n_samples": int(len(y)),
    }

# Regression Metrics
def regression_metrics(yt, pred_mean, pred_lo=None, pred_hi=None, quantile=False) -> Dict:
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    yt = np.asarray(yt, dtype="float64")
    ym = np.asarray(pred_mean, dtype="float64")
    out = {
        "r2": float(r2_score(yt, ym)),
        "mae": float(mean_absolute_error(yt, ym)),
        "rmse": float(np.sqrt(mean_squared_error(yt, ym))),
        "n_samples": int(len(yt)),
    }
    if quantile and pred_lo is not None and pred_hi is not None:
        lo = np.maximum(np.asarray(pred_lo, dtype="float64"), 0)
        hi = np.maximum(np.asarray(pred_hi, dtype="float64"), lo)
        out["interval_coverage"] = float(np.mean((yt >= lo) & (yt <= hi)))
        out["avg_interval_width_h"] = float(np.mean(hi - lo))
    try:
        from scipy.stats import pearsonr
        out["correlation"] = float(pearsonr(yt, ym)[0])
    except Exception:
        out["correlation"] = None
    return out

# Per Engine Mae
def per_engine_mae(engine_ids, gt, pred) -> Dict[str, float]:
    engine_ids = np.asarray(engine_ids)
    out = {}
    for eid in sorted(set(str(e) for e in engine_ids)):
        m = engine_ids == eid
        if m.sum() > 0:
            out[eid] = float(np.mean(np.abs(np.asarray(gt)[m] - np.asarray(pred)[m])))
    return out

# Mission Summary Rows
def mission_summary_rows(rec: pd.DataFrame, mlabels: List[str],
                         mission_warnings: Dict[str, List[str]]) -> pd.DataFrame:
    rows = []
    for mid, g in rec.groupby("mission_id", sort=False):
        anom = g["anomaly_score"].to_numpy(dtype="float64")
        alerts = int((anom >= ANOMALY_ALERT_THRESHOLD).sum())
        fprob = g["fault_probability"].to_numpy(dtype="float64")
        det = np.zeros(len(g), dtype=bool)
        det |= anom >= ANOMALY_ALERT_THRESHOLD
        det |= fprob > FROZEN_BINARY_THRESHOLD
        first_t = float(g["window_start_s"].to_numpy()[det].min()) if det.any() else None
        probable = []
        for lab in mlabels:
            mx = float(g["prob_" + lab].max())
            if mx > FROZEN_MULTILABEL_THRESHOLD:
                probable.append(f"{lab}({mx:.2f})")
        ru = g["rul_mean_h"].to_numpy(dtype="float64")
        ru = ru[np.isfinite(ru)]
        he = g["health_index"].to_numpy(dtype="float64")
        he = he[np.isfinite(he)]
        final_rul = float(ru[-1]) if len(ru) else None
        min_rul = float(ru.min()) if len(ru) else None
        final_health = float(he[-1]) if len(he) else None
        low_health = float(he.min()) if len(he) else None
        high_risk = bool(alerts > 0 or (final_rul is not None and final_rul <= RUL_MIN_HOURS_RISK)
                         or (final_health is not None and final_health <= HEALTH_RISK_LEVEL))
        rows.append({
            "mission_id": str(mid),
            "engine_id": str(g["engine_id"].iloc[0]),
            "n_windows": int(len(g)),
            "mean_anomaly_score": float(anom.mean()),
            "max_anomaly_score": float(anom.max()),
            "anomaly_alerts": alerts,
            "max_fault_probability": float(fprob.max()),
            "predicted_probable_faults": "; ".join(probable) or "none",
            "first_detection_time_s": first_t,
            "lowest_health_index": low_health,
            "final_health_index": final_health,
            "min_rul_h": min_rul,
            "final_rul_h": final_rul,
            "data_quality_warnings": " | ".join(mission_warnings.get(str(mid), [])) or "none",
            "high_risk": high_risk,
        })
    return pd.DataFrame(rows)

# Timeline Plot
def timeline_plot(mission_id, rec: pd.DataFrame, mlabels: List[str], outpath: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    g = rec[rec["mission_id"] == mission_id].sort_values("window_start_s")
    if len(g) == 0:
        return
    fig, axes = plt.subplots(5, 1, figsize=(11, 13), sharex=True)
    t = g["window_start_s"].to_numpy(dtype="float64")

    axes[0].plot(t, g["anomaly_score"], color="#c0392b")
    axes[0].axhline(ANOMALY_ALERT_THRESHOLD, ls="--", color="#7f8c8d", lw=1)
    axes[0].set_ylabel("anomaly")
    axes[0].set_title(f"mission {mission_id} — anomaly / fault / RUL / health timeline")

    axes[1].plot(t, g["fault_probability"], color="#16a085")
    if "gt_anomaly" in g.columns:
        gt = g["gt_anomaly"].to_numpy(dtype="float64")
        ok = np.isfinite(gt)
        axes[1].fill_between(t[ok], 0, gt[ok], color="#bdc3c7", alpha=0.5, label="label")
        axes[1].legend(loc="upper left", fontsize=7)
    axes[1].set_ylabel("fault P")

    for lab in mlabels:
        axes[2].plot(t, g["prob_" + lab], label=lab, lw=1)
    axes[2].set_ylabel("fault probs")
    axes[2].legend(loc="upper left", fontsize=6, ncol=4)

    rul = g["rul_mean_h"].to_numpy(dtype="float64")
    ok = np.isfinite(rul)
    if ok.any():
        tt = t[ok]
        axes[3].plot(tt, rul[ok], color="#2980b9")
        lo = np.maximum(g["rul_lower_h"].to_numpy(dtype="float64"), 0)[ok]
        hi = np.maximum(g["rul_upper_h"].to_numpy(dtype="float64"), lo)[ok]
        axes[3].fill_between(tt, lo, hi, color="#2980b9", alpha=0.25)
        if "gt_rul_h" in g.columns:
            gr = g["gt_rul_h"].to_numpy(dtype="float64")
            okg = np.isfinite(gr)
            axes[3].plot(tt[okg], gr[okg], color="#8e44ad", ls="--", label="true")
            axes[3].legend(loc="upper left", fontsize=7)
    axes[3].set_ylabel("RUL h")

    he = g["health_index"].to_numpy(dtype="float64")
    ok = np.isfinite(he)
    if ok.any():
        axes[4].plot(t[ok], he[ok], color="#27ae60")
        if "gt_health" in g.columns:
            gh = g["gt_health"].to_numpy(dtype="float64")
            okg = np.isfinite(gh)
            axes[4].plot(t[okg], gh[okg], color="#8e44ad", ls="--", label="true")
            axes[4].legend(loc="upper left", fontsize=7)
        axes[4].set_ylim(0, 105)
    axes[4].set_ylabel("health")
    axes[4].set_xlabel("window start time (s from mission start)")

    fig.tight_layout()
    fig.savefig(outpath, dpi=110)
    plt.close(fig)

# Global Labeled Plots
def global_labeled_plots(out_root: Path, rec: pd.DataFrame, metrics: Dict, mlabels: List[str]):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plots = out_root / "plots"

    m = metrics.get("anomaly")
    if m and isinstance(m, dict) and m.get("roc_curve"):
        fig, ax = plt.subplots(1, 2, figsize=(11, 4.5))
        fpr, tpr = m["roc_curve"]
        ax[0].plot(fpr, tpr, lw=2, color="#c0392b")
        ax[0].plot([0, 1], [0, 1], ls="--", color="#95a5a6")
        ax[0].set_title(f"anomaly ROC  AUC={m['roc_auc']:.3f}")
        ax[0].set_xlabel("FPR"); ax[0].set_ylabel("TPR")
        pxx, rxx = m["pr_curve"]
        ax[1].plot(rxx, pxx, lw=2, color="#c0392b")
        ax[1].set_title(f"anomaly PR  AP={m['pr_auc']:.3f}")
        ax[1].set_xlabel("recall"); ax[1].set_ylabel("precision")
        fig.tight_layout(); fig.savefig(plots / "anomaly_roc_pr.png", dpi=130); plt.close(fig)

    m = metrics.get("bilstm")
    if m and isinstance(m, dict) and m.get("confusion_matrix"):
        cm = np.asarray(m["confusion_matrix"])
        fig, ax = plt.subplots(figsize=(4.6, 4.2))
        ax.imshow(cm, cmap="Blues")
        for i in range(2):
            for j in range(2):
                ax.text(j, i, int(cm[i, j]), ha="center", va="center")
        ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
        ax.set_xticklabels(["normal", "fault"]); ax.set_yticklabels(["normal", "fault"])
        ax.set_title(f"Bi-LSTM confusion  (AUC {m['roc_auc']:.3f})")
        ax.set_xlabel("predicted"); ax.set_ylabel("true")
        fig.tight_layout(); fig.savefig(plots / "bilstm_confusion_matrix.png", dpi=130); plt.close(fig)

    m = metrics.get("multilabel")
    if m and isinstance(m, dict) and m.get("per_label"):
        names = list(m["per_label"])
        f1v = [m["per_label"][n]["f1"] for n in names]
        fig, ax = plt.subplots(figsize=(10, 4.2))
        bars = ax.bar(names, f1v, color="#16a085")
        ax.bar_label(bars, fmt="%.2f", fontsize=8)
        ax.set_ylim(0, 1.05); ax.set_ylabel("F1")
        ax.set_title(f"multi-label per-fault F1 (macro {m['macro_f1']:.3f}, "
                     f"micro {m['micro_f1']:.3f})")
        plt.setp(ax.get_xticklabels(), rotation=25, ha="right")
        fig.tight_layout(); fig.savefig(plots / "multilabel_per_fault_f1.png", dpi=130); plt.close(fig)

    m = metrics.get("rul")
    sub = rec[rec["gt_rul_h"].notna()]
    if m and isinstance(m, dict) and len(sub):
        lo, hi = float(sub["gt_rul_h"].to_numpy(dtype="float64").min()), \
            float(sub["gt_rul_h"].to_numpy(dtype="float64").max())
        fig, ax = plt.subplots(1, 2, figsize=(11, 4.6))
        ax[0].scatter(sub["gt_rul_h"], sub["rul_mean_h"], s=6, alpha=0.4, color="#2980b9")
        ax[0].plot([lo, hi], [lo, hi], "r--", lw=1)
        ax[0].set_xlabel("true RUL h"); ax[0].set_ylabel("predicted RUL h")
        ax[0].set_title(f"RUL true vs predicted (R2 {m['r2']:.3f}, MAE {m['mae']:.3f}h)")
        mn = sub["rul_mean_h"].to_numpy(dtype="float64")
        lo_ = np.maximum(sub["rul_lower_h"].to_numpy(dtype="float64"), 0)
        hi_ = np.maximum(sub["rul_upper_h"].to_numpy(dtype="float64"), lo_)
        order = np.argsort(mn)
        ax[1].plot(mn[order], mn[order], lw=1, color="#2980b9")
        ax[1].fill_between(mn[order], lo_[order], hi_[order], alpha=0.25, color="#2980b9")
        ax[1].set_xlabel("predicted RUL mean h"); ax[1].set_ylabel("hours")
        ax[1].set_title(f"RUL intervals (coverage {m['interval_coverage']:.3f})")
        fig.tight_layout(); fig.savefig(plots / "rul_true_vs_pred.png", dpi=130); plt.close(fig)

    m = metrics.get("health")
    sub = rec[rec["gt_health"].notna()]
    if m and isinstance(m, dict) and len(sub):
        lo, hi = float(sub["gt_health"].to_numpy(dtype="float64").min()), \
            float(sub["gt_health"].to_numpy(dtype="float64").max())
        fig, ax = plt.subplots(figsize=(6, 5))
        ax.scatter(sub["gt_health"], sub["health_index"], s=6, alpha=0.4, color="#27ae60")
        ax.plot([lo, hi], [lo, hi], "r--", lw=1)
        ax.set_xlabel("true health index"); ax.set_ylabel("predicted health index")
        ax.set_title(f"Health true vs predicted (R2 {m['r2']:.3f})")
        fig.tight_layout(); fig.savefig(plots / "health_true_vs_pred.png", dpi=130); plt.close(fig)

# Metrics Csv
def metrics_csv(metrics: Dict) -> pd.DataFrame:
    rows = []

    # Add
    def add(group, metric, value, unit=""):
        as_str = "" if value is None or (isinstance(value, float) and not math.isfinite(value)) else str(value)
        rows.append({"group": group, "metric": metric, "value": as_str, "unit": unit})

    for gk, prefix, unit in [("anomaly", "isolation_forest", ""),
                             ("bilstm", "bilstm", "")]:
        m = metrics.get(gk)
        if not isinstance(m, dict):
            continue
        for k, u in [("roc_auc", ""), ("pr_auc", ""), ("accuracy", ""),
                     ("balanced_accuracy", ""), ("precision", ""), ("recall", ""),
                     ("f1", ""), ("false_positive_rate", ""), ("false_negative_rate", ""),
                     ("n_samples", "")]:
            if k in m:
                add(prefix, k, m.get(k), u)
    m = metrics.get("multilabel")
    if isinstance(m, dict):
        for k in ["exact_match", "hamming_loss", "macro_f1", "micro_f1", "samples_f1",
                  "macro_precision", "macro_recall", "n_samples"]:
            if k in m:
                add("multilabel_faults", k, m.get(k))
        for lab, d in (m.get("per_label") or {}).items():
            for k in ["precision", "recall", "f1", "auc_roc"]:
                add("multilabel_faults", f"{k}_{lab}", d.get(k))
    m = metrics.get("rul")
    if isinstance(m, dict):
        for k, u in [("r2", ""), ("mae", "h"), ("rmse", "h"), ("interval_coverage", ""),
                     ("avg_interval_width_h", "h"), ("n_samples", "")]:
            if k in m:
                add("rul", k, m.get(k), u)
        for eid, v in (m.get("per_engine_mae") or {}).items():
            add("rul", f"per_engine_mae_{eid}", v, "h")
    m = metrics.get("health")
    if isinstance(m, dict):
        for k in ["r2", "mae", "rmse", "correlation", "n_samples"]:
            if k in m:
                add("health_index", k, m.get(k))
        for eid, v in (m.get("per_engine_mae") or {}).items():
            add("health_index", f"per_engine_mae_{eid}", v)
    if not rows:
        rows.append({"group": "overview", "metric": "status", "value": "ground truth unavailable", "unit": ""})
    return pd.DataFrame(rows)

# Main
def main() -> int:
    ap = argparse.ArgumentParser(description="APET-GUARDIAN offline evaluation harness")
    ap.add_argument("dataset", type=str, help="CSV file, Parquet file, or directory of telemetry")
    ap.add_argument("--bundle", type=str, default=None)
    ap.add_argument("--out-base", type=str, default="artifacts")
    ap.add_argument("--limit", type=int, default=None, help="cap number of input files (smoke run)")
    ap.add_argument("--kind", type=str, default="all", choices=["all", "normal", "faulty", "rul"])
    ap.add_argument("--plot-missions", type=int, default=None, help="cap per-mission timeline plots")
    ap.add_argument("--no-plots", action="store_true")
    args = ap.parse_args()

    t_start = time.time()
    warnings.filterwarnings(
        "ignore", message="RNN module weights are not part of single contiguous chunk.*")
    dataset = Path(args.dataset)
    if not dataset.exists():
        print(f"ERROR: dataset not found: {dataset}")
        return 2
    if dataset.is_dir():
        files = sorted(list(dataset.rglob("*.csv")) + list(dataset.rglob("*.parquet")))
        name = dataset.name
    else:
        files = [dataset]
        name = dataset.stem
    if args.kind != "all":
        files = [f for f in files if _kind_of(f.stem) == args.kind]
    if args.limit:
        files = files[:args.limit]
    if not files:
        print("ERROR: no telemetry files matched the dataset/kind filter.")
        return 2

    out_root = Path(args.out_base) / "dataset_test" / name
    pred_dir = out_root / "predictions"
    met_dir = out_root / "metrics"
    plot_dir = out_root / "plots"
    for d in [pred_dir, met_dir, plot_dir, plot_dir / "missions"]:
        d.mkdir(parents=True, exist_ok=True)

    bundle_path = Path(args.bundle) if args.bundle else \
        Path(__file__).resolve().parents[1] / "APET_GUARDIAN_OUT" / "artifacts" / "apet_bundle.pkl"
    art, model_version = load_bundle(bundle_path)

    mlabels = list(art["faults"])
    seq_features = list(art["seq_features"])
    stat_scaler, seq_scaler = art["stat_scaler"], art["seq_scaler"]
    physics = art["physics"]
    if_model, if_ref = art["if"]
    lstm, fusion, classical = art["lstm"], art.get("fusion"), art.get("classical")
    quantile, health = art.get("quantile"), art.get("health")
    lstm.eval()
    if fusion is not None:
        fusion.eval()
    n_feat = lstm.nfeat
    assert n_feat == len(seq_features), "bundle seq schema mismatch"

    qr = QualityReporter()
    known_splits, known_kinds = known_split_lookup()

    req_cols = sorted(set(IDENTIFIER_COLS) | set(RAW_SENSOR_COLS) |
                      set(GT_FAULT_COLUMNS) | {"phase", "fault_active",
                                               "simulated_rul_hours",
                                               "true_overall_health_index"})
    req_sensor = ["mission_id", "engine_id", "global_time_s"] + RAW_SENSOR_COLS

    engine_frames = defaultdict(list)
    file_errors = []
    n_rows = 0
    for f in files:
        try:
            df = _read_dataset(f, req_cols)
        except Exception as e:
            file_errors.append(f.name + ": " + str(e))
            qr.file(f.name, [f"unreadable: {e}"])
            continue
        missing = [c for c in req_sensor if c not in df.columns]
        if missing:
            err = f"missing required columns: {missing}"
            file_errors.append(f.name + ": " + err)
            qr.file(f.name, [err])
            continue
        df = validate_and_clean(df, f, qr)
        n_rows += len(df)
        if df.empty:
            continue
        sampling_check(df, qr)
        for eid, g in df.groupby("engine_id", sort=False):
            engine_frames[str(eid)].append(g)
    if file_errors:
        print(f"WARNING: {len(file_errors)} file(s) rejected:")
        for e in file_errors:
            print("   ", e)
    if not engine_frames:
        print("ERROR: no usable engine frames after validation.")
        return 2

    rec_cols = (["engine_id", "mission_id", "window_index",
                 "window_start_s", "window_end_s",
                 "window_start_global_s", "window_end_global_s", "phase",
                 "anomaly_score", "anomaly_alert",
                 "normal_probability", "fault_probability", "binary_prediction",
                 "predicted_faults"] +
                ["prob_" + l for l in mlabels] +
                ["rul_mean_h", "rul_lower_h", "rul_upper_h", "health_index",
                 "model_version", "data_quality_warnings"])
    rec_lists = {c: [] for c in rec_cols}
    gt_any, gt_mat, gt_rul, gt_health, g_eid, g_mid = [], [], [], [], [], []
    g_has_any = []
    n_windows = n_skipped = global_win_idx = 0

    engines = sorted(engine_frames)
    for idx, eid in enumerate(engines):
        e_df = pd.concat(engine_frames[eid], ignore_index=False)
        if "time_in_mission_s" not in e_df.columns:
            e_df["time_in_mission_s"] = e_df.groupby("mission_id")["global_time_s"].transform(
                lambda s: s - s.min())
        e_df = e_df.sort_values("global_time_s", kind="mergesort")
        e_df = physics.transform(e_df)
        e_df.index = np.arange(len(e_df))

        win_lists = {"seq": [], "t0": [], "t1": [], "gend": [], "phase": []}
        any_flag = None
        for mid, sub in e_df.groupby("mission_id", sort=False):
            built = build_windows_for_mission(str(mid), sub, seq_features)
            if built is None:
                qr.mission(str(mid), "fewer than 20 rows; windows skipped")
                qr.flag("windows_skipped", 1)
                n_skipped += 1
                continue
            windows, t0s, t1s, gend, phases = built
            expected = int(np.floor((len(sub) - C.SEQ_LEN) / C.STRIDE)) + 1
            sk = max(expected - len(windows), 0)
            if sk:
                qr.mission(str(mid), f"{sk} window(s) skipped (gaps/short blocks)")
                qr.flag("windows_skipped", sk)
            n_skipped += sk
            n_windows += len(windows)
            win_lists["seq"].append(windows)
            win_lists["t0"].append(t0s)
            win_lists["t1"].append(t1s)
            win_lists["gend"].append(gend)
            win_lists["phase"].append(phases)

        if not win_lists["seq"]:
            continue

        seq_win = np.concatenate(win_lists["seq"])
        t0 = np.concatenate(win_lists["t0"])
        t1 = np.concatenate(win_lists["t1"])
        gend_all = np.concatenate(win_lists["gend"])
        phase_all = np.concatenate(win_lists["phase"])

        stats_mat, _ = window_stats(seq_win, seq_features)
        cummat = e_df.loc[gend_all, C.CUMULATIVE_FEATURES].to_numpy(dtype="float32")
        stats_full = np.concatenate([stats_mat, cummat], axis=1).astype("float32")

        stats_s = stat_scaler.transform(stats_full)
        score = T.anomaly_score(if_model, stats_s, if_ref)

        seq_flat = seq_scaler.transform(seq_win.reshape(-1, len(seq_features))).reshape(*seq_win.shape)
        emb, prob_bin = T.lstm_predict(lstm, seq_flat)
        if fusion is not None:
            fu = T.fusion_predict(fusion, stats_s, score, emb)
        elif classical is not None:
            fu = T.classical_predict(classical, stats_s)
        else:
            raise RuntimeError("bundle has neither fusion nor classical fault model")

        if quantile is not None:
            q = T.quantile_predict(quantile, stats_s)
            q_lo, q_mid, q_hi = q[:, 0], q[:, 1], q[:, 2]
        else:
            q_lo = q_mid = q_hi = np.full(len(stats_s), np.nan)
        h_pred = T.health_predict(health, stats_s) if health is not None \
            else np.full(len(stats_s), np.nan)

        n = len(stats_s)
        any_gt = np.zeros(n)
        has_any = np.zeros(n, dtype=bool)
        mat_gt = np.zeros((n, len(mlabels)))
        rul_gt = np.full(n, np.nan)
        health_gt = np.full(n, np.nan)
        if "fault_misfire_active" in e_df.columns:
            mat_gt[:] = e_df[GT_FAULT_COLUMNS].to_numpy(dtype="float64")[gend_all]
            any_gt[:] = (mat_gt.sum(axis=1) > 0)
            has_any[:] = True
        elif "fault_active" in e_df.columns:
            any_gt[:] = e_df["fault_active"].to_numpy(dtype="float64")[gend_all]
            has_any[:] = True
        if "simulated_rul_hours" in e_df.columns:
            rul_gt[:] = e_df["simulated_rul_hours"].to_numpy(dtype="float64")[gend_all]
        if "true_overall_health_index" in e_df.columns:
            health_gt[:] = e_df["true_overall_health_index"].to_numpy(dtype="float64")[gend_all]

        gtimes = e_df["global_time_s"].to_numpy(dtype="float64")
        t0g = gtimes[gend_all - C.SEQ_LEN + 1]
        t1g = gtimes[gend_all]

        alerts = (score >= ANOMALY_ALERT_THRESHOLD)
        bin_pred = (prob_bin > FROZEN_BINARY_THRESHOLD).astype(int)
        pred_faults = []
        for i in range(n):
            act = [mlabels[j] for j in range(len(mlabels))
                   if fu[i, j] > FROZEN_MULTILABEL_THRESHOLD]
            pred_faults.append("|".join(act) if act else "none")

        warn_map = qr.mission_warnings
        mids = e_df["mission_id"].to_numpy(dtype=object)[gend_all]
        for i in range(n):
            rec_lists["engine_id"].append(eid)
            rec_lists["mission_id"].append(str(mids[i]))
            rec_lists["window_index"].append(global_win_idx)
            global_win_idx += 1
            rec_lists["window_start_s"].append(float(t0[i]))
            rec_lists["window_end_s"].append(float(t1[i]))
            rec_lists["window_start_global_s"].append(float(t0g[i]))
            rec_lists["window_end_global_s"].append(float(t1g[i]))
            rec_lists["phase"].append(str(phase_all[i]))
            rec_lists["anomaly_score"].append(float(score[i]))
            rec_lists["anomaly_alert"].append(bool(alerts[i]))
            rec_lists["normal_probability"].append(float(1.0 - prob_bin[i]))
            rec_lists["fault_probability"].append(float(prob_bin[i]))
            rec_lists["binary_prediction"].append(int(bin_pred[i]))
            rec_lists["predicted_faults"].append(pred_faults[i])
            for j, lab in enumerate(mlabels):
                rec_lists["prob_" + lab].append(float(fu[i, j]))
            rec_lists["rul_mean_h"].append(None if not np.isfinite(q_mid[i]) else float(q_mid[i]))
            rec_lists["rul_lower_h"].append(None if not np.isfinite(q_lo[i]) else float(q_lo[i]))
            rec_lists["rul_upper_h"].append(None if not np.isfinite(q_hi[i]) else float(q_hi[i]))
            rec_lists["health_index"].append(None if not np.isfinite(h_pred[i]) else float(h_pred[i]))
            rec_lists["model_version"].append(model_version)
            wlist = " | ".join(warn_map.get(str(mids[i]), [])) or "none"
            rec_lists["data_quality_warnings"].append(wlist)

        gt_any.append(any_gt)
        gt_mat.append(mat_gt)
        gt_rul.append(rul_gt)
        gt_health.append(health_gt)
        g_eid.append(np.full(n, eid, dtype=object))
        g_mid.append(mids)
        g_has_any.append(has_any)

        if (idx + 1) % 20 == 0 or idx == len(engines) - 1:
            print(f"  engine {idx+1}/{len(engines)}  windows so far {n_windows:,}  "
                  f"({time.time() - t_start:.1f}s)", flush=True)

    if not rec_lists["mission_id"]:
        print("ERROR: no windows could be generated.")
        return 2

    rec = pd.DataFrame(rec_lists)
    flat_any = np.concatenate(gt_any)
    flat_mat = np.vstack(gt_mat)
    flat_rul = np.concatenate(gt_rul)
    flat_health = np.concatenate(gt_health)
    flat_eid = np.concatenate(g_eid)
    flat_mid = np.concatenate(g_mid)
    flat_has = np.concatenate(g_has_any)

    rec["gt_anomaly"] = flat_any
    for j, lab in enumerate(mlabels):
        rec["gt_" + lab] = flat_mat[:, j]
    rec["gt_rul_h"] = flat_rul
    rec["gt_health"] = flat_health

    metrics = {}
    ybin = flat_any[flat_has]
    if flat_has.sum() and len(np.unique(ybin)) > 1:
        ys = rec["anomaly_score"].to_numpy(dtype="float64")[flat_has]
        metrics["anomaly"] = binary_metrics(ybin, ys)
        metrics["bilstm"] = binary_metrics(
            ybin, rec["fault_probability"].to_numpy(dtype="float64")[flat_has])
    else:
        reason = "no ground-truth fault labels (or single class)" if flat_has.sum() else \
            "no ground-truth fault label columns"
        metrics["anomaly"] = {"available": False, "reason": reason}
        metrics["bilstm"] = {"available": False, "reason": reason}

    mmat = flat_mat[flat_has]
    ml = multilabel_metrics(mmat, rec[[ "prob_" + l for l in mlabels ]].to_numpy(dtype="float64")[flat_has],
                            mlabels)
    metrics["multilabel"] = ml if ml else {"available": False,
                                           "reason": "fault labels all-negative or missing"}

    ok_rul = np.isfinite(rec["gt_rul_h"].to_numpy(dtype="float64")) & \
        np.isfinite(rec["rul_mean_h"].to_numpy(dtype="float64"))
    if ok_rul.sum() > 20:
        yt = rec["gt_rul_h"].to_numpy(dtype="float64")[ok_rul]
        pm = rec["rul_mean_h"].to_numpy(dtype="float64")[ok_rul]
        pl = rec["rul_lower_h"].to_numpy(dtype="float64")[ok_rul]
        ph = rec["rul_upper_h"].to_numpy(dtype="float64")[ok_rul]
        m = regression_metrics(yt, pm, pl, ph, quantile=True)
        m["per_engine_mae"] = per_engine_mae(flat_eid[ok_rul], yt, pm)
        metrics["rul"] = m
    else:
        metrics["rul"] = {"available": False, "reason": "not enough RUL ground truth",
                          "n_samples": int(ok_rul.sum())}

    ok_health = np.isfinite(rec["gt_health"].to_numpy(dtype="float64")) & \
        np.isfinite(rec["health_index"].to_numpy(dtype="float64"))
    if ok_health.sum() > 20:
        yt = rec["gt_health"].to_numpy(dtype="float64")[ok_health]
        ph = rec["health_index"].to_numpy(dtype="float64")[ok_health]
        m = regression_metrics(yt, ph)
        m["per_engine_mae"] = per_engine_mae(flat_eid[ok_health], yt, ph)
        metrics["health"] = m
    else:
        metrics["health"] = {"available": False, "reason": "not enough health ground truth",
                             "n_samples": int(ok_health.sum())}

    mission_warnings = {k: v for k, v in qr.mission_warnings.items()}
    summary = mission_summary_rows(rec, mlabels, mission_warnings)
    rec.to_csv(pred_dir / "window_predictions.csv", index=False)
    summary.to_csv(pred_dir / "mission_summary.csv", index=False)

    input_keys = sorted({
        ("engine_" + f.stem[len("life_"):] if f.stem.startswith("life_") else f.stem)
        for f in files})
    overlap_hits = set(input_keys) & set(known_splits)
    split_hist = defaultdict(int)
    for m in overlap_hits:
        split_hist[str(known_splits[m])] += 1
    n_total_files = max(len(files), 1)
    pct = 100.0 * len(overlap_hits) / n_total_files
    overlap_statement = (
        f"{len(overlap_hits)} of {len(files)} input file(s) ({pct:.1f}%) match the "
        f"APET-GUARDIAN training corpus (deterministic split replay: {dict(split_hist)}); "
        "results for those files are NOT unseen. Remaining input is treated as unseen external data."
        if overlap_hits else
        "No input files match the prior training/validation/test corpus: this is treated as "
        "unseen external data.")

    if not args.no_plots:
        maxpm = args.plot_missions or n_total_missions
        plotted = 0
        for mid in list(rec["mission_id"].unique())[:maxpm]:
            timeline_plot(mid, rec, mlabels, plot_dir / "missions" / f"{mid}.png")
            plotted += 1
        global_labeled_plots(out_root, rec, metrics, mlabels)
        print(f"  plots: {plotted} per-mission timelines + labeled figures written to {plot_dir}")

    metrics["_overview"] = {
        "dataset_name": name,
        "input_files": len(files),
        "files_rejected": len(file_errors),
        "rows": int(n_rows),
        "missions": int(rec["mission_id"].nunique()),
        "engines": int(rec["engine_id"].nunique()),
        "valid_windows": int(len(rec)),
        "skipped_windows": int(n_skipped),
        "model_version": model_version,
        "overlap_missions": int(len(overlap_hits)),
        "overlap_statement": overlap_statement,
        "frozen_thresholds": {"binary_fault": FROZEN_BINARY_THRESHOLD,
                              "multilabel": FROZEN_MULTILABEL_THRESHOLD,
                              "anomaly_alert": ANOMALY_ALERT_THRESHOLD},
        "advisory_only": True,
    }

    with open(met_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2, default=str)
    metrics_csv(metrics).to_csv(met_dir / "metrics.csv", index=False)
    with open(out_root / "data_quality_report.json", "w") as f:
        json.dump(qr.to_dict(), f, indent=2, default=str)

    body = [f"# APET-GUARDIAN evaluation report — dataset `{name}`",
            "",
            f"**Model version:** {model_version}",
            "",
            "**Inputs:** " + ", ".join(str(f) for f in files[:10]) +
            (f", … ({len(files)} total files)" if len(files) > 10 else ""),
            "",
            "## Overlap status",
            overlap_statement,
            "",
            "## Data quality",
            f"- Rows: {int(n_rows):,} · Files read: {len(files)} · Rejected: {len(file_errors)} "
            f"· Engines: {rec['engine_id'].nunique()} · Missions: {rec['mission_id'].nunique()} "
            f"· Valid windows: {len(rec):,} · Skipped windows: {n_skipped:,}",
            "- Global flags: " + json.dumps(dict(qr.global_flags)),
            "",
            "## Metrics"]
    gname = {"anomaly": "Isolation Forest anomaly", "bilstm": "Bi-LSTM binary",
             "multilabel": "Fusion multi-label faults", "rul": "RUL (h)",
             "health": "Health Index"}
    for grp in ["anomaly", "bilstm", "multilabel", "rul", "health"]:
        m = metrics.get(grp)
        body.append(f"### {gname[grp]}")
        if isinstance(m, dict) and m.get("available") is False:
            body.append(f"_unavailable — {m.get('reason')} (n={m.get('n_samples', 0)})_")
        elif isinstance(m, dict):
            skip = {"roc_curve", "pr_curve", "confusion_matrix", "per_fault_confusion",
                    "labels", "gt", "per_label", "per_engine_mae"}
            for k, v in m.items():
                if k not in skip:
                    body.append(f"- {k}: {v}")
            if m.get("per_engine_mae"):
                body.append("- per-engine MAE: " + ", ".join(
                    f"{e}: {v:.3f}" for e, v in m["per_engine_mae"].items()))
            if m.get("per_label"):
                body.append("- per-fault F1: " + ", ".join(
                    f"{k}: {v['f1']:.3f} ({v['precision']:.2f}/{v['recall']:.2f})"
                    for k, v in m["per_label"].items()))
        else:
            body.append("_unavailable_")
        body.append("")

    n_risk = int(summary["high_risk"].sum())
    body += ["## Mission risk summary",
             f"- High-risk missions: {int(n_risk)}",
             "- Top missions by risk:",
             "| mission | alerts | final RUL h | final health | probable faults |",
             "|---|---|---|---|---|"]
    for _, r in summary.sort_values("high_risk", ascending=False).head(15).iterrows():
        body.append(f"| {r['mission_id']} | {r['anomaly_alerts']} | "
                    f"{'%.2f' % r['final_rul_h'] if pd.notna(r['final_rul_h']) else 'n/a'} | "
                    f"{'%.1f' % r['final_health_index'] if pd.notna(r['final_health_index']) else 'n/a'} | "
                    f"{str(r['predicted_probable_faults'])[:80]} |")
    body += ["",
             "## Method",
             "- Windows: 20 rows (10 s at 2 Hz), causal, stride 10 rows (5 s), per mission.",
             "- Physics features: saved `PhysicsFeatureModel.transform` (no fit). Scalers: "
             "saved train scalers (`.transform` only, no fit/fit_transform).",
             f"- Frozen thresholds (never recalibrated): binary fault > {FROZEN_BINARY_THRESHOLD}, "
             f"multi-label > {FROZEN_MULTILABEL_THRESHOLD}, anomaly alert >= {ANOMALY_ALERT_THRESHOLD}.",
             "- **All outputs are advisory maintenance and pilot decision-support only.**"]
    (out_root / "report.md").write_text("\n".join(body), encoding="utf-8")

    print("\n================ APET-GUARDIAN DATASET EVALUATION SUMMARY ================")
    print(f"dataset            : {name}  ({len(files)} files, {int(n_rows):,} rows)")
    print(f"missions / engines : {rec['mission_id'].nunique()} / {rec['engine_id'].nunique()}")
    print(f"windows            : {len(rec):,} valid, {n_skipped:,} skipped")
    print(f"missing-data warns : {int(summary['data_quality_warnings'].astype(str).str.contains('imputed|missing|unit', regex=True).sum())} missions  (flags: {dict(qr.global_flags)})")
    print(f"anomaly alerts     : {int(rec['anomaly_alert'].sum()):,} windows")
    print(f"high-risk missions : {int(n_risk)}")
    mh = rec["health_index"].to_numpy(dtype="float64")
    mh = mh[np.isfinite(mh)]
    print(f"mean health index  : {'n/a' if len(mh) == 0 else round(float(mh.mean()), 1)}")
    rmu = rec["rul_mean_h"].to_numpy(dtype="float64")
    rmu = rmu[np.isfinite(rmu)]
    print(f"minimum RUL        : {'--' if len(rmu) == 0 else f'{float(rmu.min()):.2f} h'}")
    pred_cats = [c[5:] for c in rec.columns
                 if c.startswith("prob_") and rec[c].max() > FROZEN_MULTILABEL_THRESHOLD]
    print("pred fault cats    : " + (", ".join(
        f"{c}({rec['prob_' + c].max():.2f})" for c in pred_cats) if pred_cats else "none"))
    print(f"overlap            : {overlap_statement}")
    # Gv
    def gv(grp, key, fallback="n/a"):
        m = metrics.get(grp)
        return fallback if not (isinstance(m, dict) and key in m) else round(float(m[key]), 3)
    print(f"model metrics      : anomaly_auc={gv('anomaly','roc_auc')}, "
          f"bilstm_auc={gv('bilstm','roc_auc')}, ml_macro_f1={gv('multilabel','macro_f1')}, "
          f"rul_r2={gv('rul','r2')}, health_r2={gv('health','r2')}")
    print(f"outputs            : {out_root}")
    print(f"runtime            : {time.time() - t_start:.1f}s")
    print("NOTE: advisory maintenance and pilot decision support only.")
    return 0

if __name__ == "__main__":
    sys.exit(main())
