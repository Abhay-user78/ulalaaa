# Run pipeline
from __future__ import annotations

import pickle
import warnings
from collections import OrderedDict
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from apet_guardian import config as C
from apet_guardian import training as T
from apet_guardian.windowing import window_sequence, window_stats

INPUT_CSV = r"D:\testing\inp.csv"
OUTPUT_CSV = r"D:\testing\op.csv"
MODEL_DIR = r"D:\testing\APET_GUARDIAN_OUT\artifacts"
VERIFY_OUTPUT_CSV = None
LIVE_STATE_FILE = r"D:\testing\live_state.json"
LIVE_TILE_SECONDS = 10.0
LIVE_STATE_RESET_TILES = 12

ANOMALY_ALERT_THRESHOLD = 0.5
MULTILABEL_THRESHOLD = 0.5
PER_CLASS_THRESHOLD = {
    "misfire": 0.400,
    "injector_degradation": 0.360,
    "turbo_issue": 0.380,
    "lubrication_issue": 0.360,
    "sensor_drift": 0.360,
    "overheating": 0.280,
    "electrical_fault": 0.680,
    "bearing_fault": 0.240,
}
FAULT_PROB_HIGH = 0.5
CONSECUTIVE_RISK = 3

OVERHEAT_SUSTAIN_WINDOWS = 2

SAMPLE_DT = 0.5
WINDOW_ROWS = 20
WINDOW_STRIDE = 10
MAX_GAP = 1.5

REQUIRED_SENSORS = [
    "throttle_actual", "true_rpm", "air_density_kg_m3", "altitude_m",
    "rpm", "boost_pressure_kPa", "manifold_pressure_kPa", "cht_cyl_avg_C",
    "egt_cyl_avg_C", "oil_pressure_kPa", "oil_temp_C", "coolant_temp_C",
    "fuel_flow_g_s", "fuel_temp_C", "rail_pressure_bar", "vib_rms_g",
    "battery_voltage_V", "airspeed_mps", "vertical_speed_mps",
    "engine_power_command_kW", "true_brake_power_kW",
]
ALIASES = OrderedDict([
    ("engine_id", ["engine_id", "engine", "engineid"]),
    ("mission_id", ["mission_id", "mission", "missionid"]),
    ("global_time_s", ["global_time_s", "timestamp", "time_s", "time", "t_s"]),
    ("throttle_actual", ["throttle_actual", "throttle"]),
    ("true_rpm", ["true_rpm", "engine_rpm"]),
    ("air_density_kg_m3", ["air_density_kg_m3", "air_density"]),
    ("altitude_m", ["altitude_m", "altitude"]),
    ("rpm", ["rpm", "propeller_rpm", "prop_rpm"]),
    ("boost_pressure_kPa", ["boost_pressure_kPa", "boost_kPa", "boost_pressure"]),
    ("manifold_pressure_kPa", ["manifold_pressure_kPa", "manifold_kPa"]),
    ("cht_cyl_avg_C", ["cht_cyl_avg_C", "cht_c", "cht"]),
    ("egt_cyl_avg_C", ["egt_cyl_avg_C", "egt_c", "egt"]),
    ("oil_pressure_kPa", ["oil_pressure_kPa", "oil_pressure"]),
    ("oil_temp_C", ["oil_temp_C", "oil_temp"]),
    ("coolant_temp_C", ["coolant_temp_C", "coolant_temp"]),
    ("fuel_flow_g_s", ["fuel_flow_g_s", "fuel_flow"]),
    ("fuel_temp_C", ["fuel_temp_C", "fuel_temp"]),
    ("rail_pressure_bar", ["rail_pressure_bar", "rail_pressure"]),
    ("vib_rms_g", ["vib_rms_g", "vibration_rms_g", "vibration", "vib_rms"]),
    ("battery_voltage_V", ["battery_voltage_V", "battery_voltage"]),
    ("airspeed_mps", ["airspeed_mps", "airspeed"]),
    ("vertical_speed_mps", ["vertical_speed_mps", "vertical_speed"]),
    ("engine_power_command_kW", ["engine_power_command_kW", "power_command_kw"]),
    ("true_brake_power_kW", ["true_brake_power_kW", "brake_power_kw"]),
])
UNIT_BOUNDS = {
    "rpm": (0.0, 7000.0), "true_rpm": (0.0, 7000.0), "throttle_actual": (0.0, 1.0),
    "boost_pressure_kPa": (0.0, 600.0), "manifold_pressure_kPa": (0.0, 600.0),
    "cht_cyl_avg_C": (-50.0, 500.0), "egt_cyl_avg_C": (0.0, 1200.0),
    "oil_pressure_kPa": (0.0, 1500.0), "oil_temp_C": (-50.0, 300.0),
    "coolant_temp_C": (-50.0, 300.0), "fuel_flow_g_s": (0.0, 200.0),
    "fuel_temp_C": (-50.0, 300.0), "rail_pressure_bar": (0.0, 500.0),
    "vib_rms_g": (0.0, 100.0), "battery_voltage_V": (5.0, 60.0),
    "airspeed_mps": (0.0, 400.0), "altitude_m": (-500.0, 50000.0),
    "vertical_speed_mps": (-500.0, 500.0),
    "engine_power_command_kW": (0.0, 1500.0),
    "true_brake_power_kW": (0.0, 1500.0), "air_density_kg_m3": (0.1, 2.2),
}
FAULT_TO_GT_COL = {
    "misfire": "fault_misfire_active",
    "injector_degradation": "fault_injector_active",
    "turbo_issue": "fault_turbo_active",
    "lubrication_issue": "fault_lubrication_active",
    "sensor_drift": "fault_sensor_active",
    "overheating": "fault_overheating_active",
    "electrical_fault": "fault_electrical_active",
    "bearing_fault": "fault_bearing_active",
}

# Load Bundle
def load_bundle(model_dir):
    p = Path(model_dir)
    if p.is_dir():
        p = p / "apet_bundle.pkl"
    candidates = [p]
    cpu_copy = p.with_name("apet_bundle_cpu.pkl")
    if cpu_copy.exists() and cpu_copy != p:
        candidates.append(cpu_copy)
    last_err = None
    for cand in candidates:
        if not cand.exists():
            continue
        try:
            import io
            import torch

            # Cpuunpickler
            class _CpuUnpickler(pickle.Unpickler):
                # Find Class
                def find_class(self, module, name):
                    if module.startswith("torch.cuda"):
                        try:
                            return getattr(torch, name)
                        except AttributeError:
                            try:
                                return getattr(torch.storage, name)
                            except AttributeError:
                                pass
                    if module == "torch.storage" and name == "_load_from_bytes":
                        orig = getattr(torch.storage, "_load_from_bytes")

                        # Cpu Load
                        def _cpu_load(b):
                            try:
                                return orig(b)
                            except RuntimeError as re:
                                if "CUDA" in str(re):
                                    return torch.UntypedStorage.from_buffer(b, byte_order="little", dtype=torch.uint8)
                                raise

                        return _cpu_load
                    return super().find_class(module, name)

            try:
                art = torch.load(str(cand), map_location=torch.device("cpu"), weights_only=False)
                if not isinstance(art, dict) or "lstm" not in art:
                    raise ValueError("torch.load did not return bundle dict")
            except Exception as te:
                last_err = te
                try:
                    with open(cand, "rb") as fh:
                        art = _CpuUnpickler(fh).load()
                except Exception as pe:
                    last_err = pe
                    orig_avail = torch.cuda.is_available
                    orig_count = getattr(torch.cuda, "device_count", None)
                    try:
                        torch.cuda.is_available = lambda: True
                        if orig_count is not None:
                            torch.cuda.device_count = lambda: 1
                        with open(cand, "rb") as fh2:
                            art = pickle.load(fh2)
                        for k in list(art.keys()):
                            v = art[k]
                            if isinstance(v, torch.nn.Module):
                                try:
                                    art[k] = v.to(torch.device("cpu"))
                                except Exception:
                                    pass
                            elif isinstance(v, tuple) and len(v) == 2 and hasattr(v[0], "to"):
                                pass
                    finally:
                        torch.cuda.is_available = orig_avail
                        if orig_count is not None:
                            torch.cuda.device_count = orig_count
            break
        except Exception as e:
            last_err = e
            continue
    else:
        raise FileNotFoundError(
            f"no loadable bundle found under {p} (last error: {last_err})")
    for key in ["physics", "if", "lstm", "fusion", "quantile", "health",
                "seq_features", "seq_scaler", "stat_scaler", "faults"]:
        if key not in art:
            raise KeyError(f"bundle is missing required artifact: {key}")
    for m in ["lstm", "fusion", "quantile", "health"]:
        if art[m] is not None:
            art[m].eval().to(T.DEVICE)
    warnings.filterwarnings(
        "ignore", message="RNN module weights are not part of single contiguous chunk.*")
    return art

# Read And Validate Input
def read_and_validate_input(path):
    df = pd.read_csv(str(path), low_memory=False)
    rename = {}
    for canonical, aliases in ALIASES.items():
        hit = next((c for c in aliases if c in df.columns), None)
        if hit is not None:
            rename[hit] = canonical
    df = df.rename(columns=rename)
    required = ["engine_id", "mission_id", "global_time_s"] + REQUIRED_SENSORS
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(
            f"input CSV is missing required columns: {missing}\n"
            f"present columns: {sorted(df.columns.tolist())}")
    keep = required + [c for c in df.columns if c.startswith("fault_")
                       or c in ("fault_active", "simulated_rul_hours",
                                "true_overall_health_index")]
    keep = [c for c in dict.fromkeys(keep) if c in df.columns]
    df = df[keep].copy()
    df["engine_id"] = df["engine_id"].astype(str)
    df["mission_id"] = df["mission_id"].astype(str)
    return df

# Preprocess
def preprocess(df):
    df = df.sort_values(["engine_id", "mission_id", "global_time_s"],
                        kind="mergesort").reset_index(drop=True)

    mission_warnings = {}
    for (eid, mid), g in df.groupby(["engine_id", "mission_id"], sort=False):
        warns = set()
        t = g["global_time_s"].to_numpy(dtype="float64")
        tt = t[np.isfinite(t)]
        base = float(np.min(tt)) if len(tt) else 0.0
        df.loc[g.index, "time_in_mission_s"] = t - base

        dt = np.diff(tt)
        if len(dt):
            med = float(np.median(dt))
            if not (0.35 <= med <= 0.75):
                warns.add(f"sampling_median_dt_{med:.2f}s_not_2hz")
            big = int(np.sum(dt > MAX_GAP))
            if big:
                warns.add(f"{big}_sampling_gaps")

        for col, (lo, hi) in UNIT_BOUNDS.items():
            v = pd.to_numeric(g[col], errors="coerce")
            if float(((v < lo) | (v > hi)).sum()) / max(len(v), 1) > 1e-3:
                warns.add(f"{col}_out_of_range")
                break
        mission_warnings[(str(eid), str(mid))] = warns

    imp_by_mission = {
        key: int(g[REQUIRED_SENSORS].isna().sum().sum())
        for key, g in df.groupby(["engine_id", "mission_id"], sort=False)
    }
    if any(v > 0 for v in imp_by_mission.values()):
        df[REQUIRED_SENSORS] = df[REQUIRED_SENSORS].ffill().bfill()
        for key, imp in imp_by_mission.items():
            if imp:
                mission_warnings[key].add(f"{imp}_missing_values_imputed")
    return df, mission_warnings

# Class Threshold
def class_threshold(name):
    return PER_CLASS_THRESHOLD.get(name, MULTILABEL_THRESHOLD)

# Build Engine Windows
def build_engine_windows(engine_df, seq_features, physics):
    eng = physics.transform(engine_df).reset_index(drop=True)

    w_sig, w_t0, w_t1, w_end = [], [], [], []
    skipped = 0
    for _, sub in eng.groupby("mission_id", sort=False):
        n = len(sub)
        if n < WINDOW_ROWS:
            continue
        seq = sub[seq_features].to_numpy(dtype="float32")
        t = sub["time_in_mission_s"].to_numpy(dtype="float64")
        win, starts, ends = window_sequence(seq, WINDOW_ROWS, WINDOW_STRIDE,
                                            time_s=t, max_gap=MAX_GAP)
        skipped += max((n - WINDOW_ROWS) // WINDOW_STRIDE + 1 - len(win), 0)
        if len(win) == 0:
            continue
        w_sig.append(win)
        w_t0.append(t[starts])
        w_t1.append(t[ends])
        w_end.append(sub.index.to_numpy()[ends])

    if not w_sig:
        return None, None, None, None, None, skipped
    return (eng, np.concatenate(w_sig), np.concatenate(w_t0),
            np.concatenate(w_t1), np.concatenate(w_end), skipped)

# Gated Overheating
def gated_overheating(multilabel, mids, faults):
    out = np.zeros(len(mids), dtype=bool)
    if "overheating" not in faults or len(mids) == 0:
        return out
    oj = faults.index("overheating")
    hot = multilabel[:, oj] > class_threshold("overheating")
    run, prev_mid = 0, None
    for k in range(len(mids)):
        mid_k = str(mids[k])
        run = (run + 1) if (mid_k == prev_mid and hot[k]) else (1 if hot[k] else 0)
        out[k] = run >= OVERHEAT_SUSTAIN_WINDOWS
        prev_mid = mid_k
    return out

# Inference Rows
def inference_rows(art, df, mission_warnings):
    physics = art["physics"]
    if_model, if_ref = art["if"]
    lstm, fusion = art["lstm"], art.get("fusion")
    quantile, health = art.get("quantile"), art.get("health")
    seq_features = list(art["seq_features"])
    stat_scaler, seq_scaler = art["stat_scaler"], art["seq_scaler"]
    faults = list(art["faults"])

    rows = []
    total_skipped = 0
    for eid, ef in df.groupby("engine_id", sort=False):
        eng, w_sig, t0, t1, end_rows, skipped = \
            build_engine_windows(ef, seq_features, physics)
        total_skipped += skipped
        if w_sig is None:
            continue

        stats_mat, _ = window_stats(w_sig, seq_features)
        cum = eng[C.CUMULATIVE_FEATURES].to_numpy(dtype="float32").take(end_rows, axis=0)
        stats_full = np.concatenate([stats_mat, cum], axis=1).astype("float32")

        stats_s = stat_scaler.transform(stats_full)
        score_if = T.anomaly_score(if_model, stats_s, if_ref)

        seq_flat = seq_scaler.transform(
            w_sig.reshape(-1, len(seq_features))).reshape(*w_sig.shape)
        emb, prob_bin = T.lstm_predict(lstm, seq_flat)
        score = T.blended_anomaly(if_model, if_ref, art.get("anomaly_extra"), stats_s, prob_bin)
        multilabel = T.fusion_predict(fusion, stats_s, score_if, emb)

        rul_lo, rul_mid, rul_hi = T.quantile_predict(quantile, stats_s).T
        health_idx = T.health_predict(health, stats_s)

        mids = eng["mission_id"].to_numpy(dtype=object)[end_rows]
        over_gated = gated_overheating(multilabel, mids, faults)

        gt_any = gt_rul = gt_health = None
        per_fault_gt = None
        present_gt = [FAULT_TO_GT_COL[f] for f in faults if FAULT_TO_GT_COL[f] in eng.columns]
        if present_gt:
            per_fault_gt = eng[present_gt].to_numpy(dtype="float64").take(end_rows, axis=0)
            gt_any = (per_fault_gt.sum(axis=1) > 0)
        elif "fault_active" in eng.columns:
            gt_any = eng["fault_active"].to_numpy(dtype="float64").take(end_rows) > 0
        if "simulated_rul_hours" in eng.columns:
            gt_rul = eng["simulated_rul_hours"].to_numpy(dtype="float64").take(end_rows)
        if "true_overall_health_index" in eng.columns:
            gt_health = eng["true_overall_health_index"].to_numpy(dtype="float64").take(end_rows)

        for k in range(len(w_sig)):
            mid = str(mids[k])
            warns = mission_warnings.get((str(eid), mid), set())
            active = [faults[j] for j in range(len(faults))
                      if (over_gated[k] if faults[j] == "overheating"
                          else multilabel[k, j] > class_threshold(faults[j]))]
            rows.append(dict(
                engine_id=str(eid),
                mission_id=mid,
                window_start_time=float(t0[k]),
                window_end_time=float(t1[k]),
                anomaly_score=float(score[k]),
                anomaly_alert=bool(score[k] >= ANOMALY_ALERT_THRESHOLD),
                normal_probability=float(1.0 - prob_bin[k]),
                fault_probability=float(prob_bin[k]),
                predicted_faults="; ".join(active) if active else "none",
                **{f"fault_probability_{name}": float(multilabel[k, j])
                   for j, name in enumerate(faults)},
                rul_mean_hours=(None if not np.isfinite(rul_mid[k]) else float(rul_mid[k])),
                rul_lower_hours=(None if not np.isfinite(rul_lo[k]) else float(rul_lo[k])),
                rul_upper_hours=(None if not np.isfinite(rul_hi[k]) else float(rul_hi[k])),
                health_index=(None if not np.isfinite(health_idx[k]) else float(health_idx[k])),
                data_quality_warning="; ".join(sorted(warns)) if warns else "ok",
                gt_anomaly=(None if gt_any is None else bool(gt_any[k])),
                gt_faults="; ".join(faults[j] for j in range(len(faults))
                                    if per_fault_gt is not None and per_fault_gt[k, j] > 0.5)
                if per_fault_gt is not None and (per_fault_gt[k].sum() > 0) else "none",
                **({f"gt_{name}": bool(per_fault_gt[k, j]) for j, name in enumerate(faults)}
                   if per_fault_gt is not None else {}),
                gt_rul_hours=(None if gt_rul is None else float(gt_rul[k])),
                gt_health_index=(None if gt_health is None else float(gt_health[k])),
            ))
    if not rows:
        raise RuntimeError(
            f"no windows could be generated: the input CSV has no mission with at "
            f"least {WINDOW_ROWS} consecutive rows (10 s @ {1.0 / SAMPLE_DT:.1f} Hz). "
            f"Input has {len(df):,} row(s).")
    return pd.DataFrame(rows), total_skipped

# Has Ground Truth
def _has_ground_truth(rec: pd.DataFrame) -> bool:
    return "gt_anomaly" in rec.columns and rec["gt_anomaly"].notna().any()

# Fmt
def _fmt(v, nd: int = 3) -> str:
    return "n/a" if v is None or not np.isfinite(v) else f"{v:.{nd}f}"

# Fval
def _fval(x) -> Optional[float]:
    if x is None:
        return None
    try:
        f = float(x)
    except (TypeError, ValueError):
        return None
    return f if np.isfinite(f) else None

# Verify Against Ground Truth
def verify_against_ground_truth(rec: pd.DataFrame) -> None:
    if not _has_ground_truth(rec):
        print("\n[verify] no ground-truth columns in the input CSV; verification skipped.")
        return

    # Safe
    def safe(a: pd.Series, b: pd.Series):
        ii = a.notna() & b.notna()
        if int(ii.sum()) == 0:
            return None, None, None
        a, b = a[ii].to_numpy(dtype="float64"), b[ii].to_numpy(dtype="float64")
        if np.allclose(a, b, atol=1e-9) and not np.any(a):
            return 0.0, 0.0, 0.0
        mae = float(np.mean(np.abs(a - b)))
        rmse = float(np.sqrt(np.mean((a - b) ** 2)))
        denom = float(np.var(a)) * float(len(a))
        r2 = 1.0 - float(np.sum((a - b) ** 2)) / denom if denom > 1e-12 else None
        return mae, rmse, r2

    n = len(rec)
    g = rec["gt_anomaly"].to_numpy(dtype=float)
    p = rec["anomaly_score"].to_numpy(dtype="float64")

    from sklearn.metrics import (accuracy_score, average_precision_score,
                                 f1_score, precision_score, recall_score,
                                 roc_auc_score)
    print("\n================ APET-GUARDIAN GROUND-TRUTH VERIFICATION ================")
    print("comparison is on the REAL labels already present in the input CSV")

    pred_any = rec["predicted_faults"] != "none"
    labels = g[g >= 0]
    pbin = pred_any.loc[g >= 0]
    if len(labels) and labels.sum() > 0 and labels.sum() < len(labels):
        print(f"binary fault    : acc={_fmt(accuracy_score(labels, pbin))}  "
              f"precision={_fmt(precision_score(labels, pbin))}  "
              f"recall={_fmt(recall_score(labels, pbin))}  "
              f"f1={_fmt(f1_score(labels, pbin))}")
    else:
        print("binary fault    : (normal-only or fully-faulty sample; label balance too skewed)")

    if labels.sum() > 0 and labels.sum() < len(labels) and np.ptp(p[g >= 0]) > 1e-12:
        print(f"anomaly channel : auc={_fmt(roc_auc_score(labels, p[g >= 0]))}  "
              f"pr_auc={_fmt(average_precision_score(labels, p[g >= 0]))}")

    faults = [c[len("fault_probability_"):] for c in rec.columns
              if c.startswith("fault_probability_")]
    if _has_ground_truth(rec) and all(f"gt_{f}" in rec.columns for f in faults):
        truth = rec[[f"gt_{f}" for f in faults]].to_numpy(dtype="float")
        pred = np.array([[f in row for f in faults]
                         for row in rec["predicted_faults"]], dtype=float)
        print("multi-label     : per-fault report below (predicted_faults vs real)")
        for j, f in enumerate(faults):
            ok = np.isfinite(truth[:, j])
            if ok.sum() == 0 or truth[ok, j].sum() == 0:
                print(f"  {f:<22s}: (no positive labels)")
                continue
            acc = accuracy_score(truth[ok, j], pred[ok, j])
            pr = precision_score(truth[ok, j], pred[ok, j], zero_division=0)
            rc = recall_score(truth[ok, j], pred[ok, j], zero_division=0)
            f1 = f1_score(truth[ok, j], pred[ok, j], zero_division=0)
            print(f"  {f:<22s}: acc={_fmt(acc)} precision={_fmt(pr)} recall={_fmt(rc)} f1={_fmt(f1)}")

    if "gt_rul_hours" in rec.columns and rec["gt_rul_hours"].notna().any() \
            and rec["rul_mean_hours"].notna().any():
        mae, rmse, r2 = safe(rec["gt_rul_hours"], rec["rul_mean_hours"])
        print(f"RUL (h)         : mae={_fmt(mae)} rmse={_fmt(rmse)} r2={_fmt(r2)}"
              f"  ([lower,upper] coverage vs gt included in CSV)")
    if "gt_health_index" in rec.columns and rec["gt_health_index"].notna().any() \
            and rec["health_index"].notna().any():
        mae, rmse, r2 = safe(rec["gt_health_index"], rec["health_index"])
        print(f"health index    : mae={_fmt(mae)} rmse={_fmt(rmse)} r2={_fmt(r2)}")
    print("========================================================================")

    if VERIFY_OUTPUT_CSV:
        cols = [c for c in ["engine_id", "mission_id", "window_start_time",
                            "window_end_time", "predicted_faults", "gt_faults",
                            "anomaly_score", "gt_anomaly", "rul_mean_hours",
                            "rul_lower_hours", "rul_upper_hours", "gt_rul_hours",
                            "health_index", "gt_health_index", "fault_probability",
                            "data_quality_warning"] if c in rec.columns]
        rec[cols].to_csv(VERIFY_OUTPUT_CSV, index=False)
        print(f"verification CSV          : {VERIFY_OUTPUT_CSV}")

# Add Mission Risk
def add_mission_risk(rec):
    risk = {}
    for mid, g in rec.groupby("mission_id", sort=False):
        hot = g["anomaly_alert"].to_numpy(dtype=bool) | \
              (g["fault_probability"].to_numpy(dtype="float64") >= FAULT_PROB_HIGH)
        run = 0
        elevated = False
        for v in hot:
            run = run + 1 if v else 0
            if run >= CONSECUTIVE_RISK:
                elevated = True
                break
        risk[mid] = "elevated" if elevated else "normal"
    rec["mission_risk"] = rec["mission_id"].map(risk)
    return rec

# Main
def main(live: bool = False):
    art = load_bundle(MODEL_DIR)
    raw = read_and_validate_input(INPUT_CSV)
    df, mission_warnings = preprocess(raw)

    if df.empty:
        print(f"INPUT_CSV ({INPUT_CSV}) is empty - it only contains column headers right now.")
        print(f"Add your flight data below the headers (2 Hz rows, one row per 0.5 s) and run again.")
        return

    try:
        rec, skipped = inference_rows(art, df, mission_warnings)
    except RuntimeError as e:
        print(e)
        if live:
            print("live mode waits for more flight data - append ~20 rows per mission")
            print("(10 s @ 2 Hz, or use LIVE_TILE_SECONDS to change the tile) and re-run --live.")
        else:
            print(f"add at least {WINDOW_ROWS} consecutive rows (10 s) per mission to {INPUT_CSV} and re-run.")
        return
    rec = add_mission_risk(rec)

    if live:
        run_live(rec, skipped, len(df), df)
    else:
        rec.to_csv(OUTPUT_CSV, index=False)
        print(f"total input rows           : {len(df):,}")
        print(f"total missions             : {df['mission_id'].nunique():,}")
        print(f"valid windows processed    : {len(rec):,}")
        print(f"skipped windows            : {skipped:,}")
        print(f"anomaly alerts             : {int(rec['anomaly_alert'].sum()):,}")
        print(f"fault detections           : {int((rec['predicted_faults'] != 'none').sum()):,}")
        print(f"output CSV                 : {OUTPUT_CSV}")
        verify_against_ground_truth(rec)
    print("NOTE: advisory engine-health and maintenance decision support only.")

# Run Live
def run_live(rec: pd.DataFrame, skipped: int, n_rows: int, df: pd.DataFrame) -> None:
    import json
    sp = Path(LIVE_STATE_FILE)
    state = {}
    if sp.exists():
        try:
            state = json.loads(sp.read_text(encoding="utf-8"))
        except Exception:
            state = {}

    rec = rec.copy()
    rec.to_csv(OUTPUT_CSV, index=False)
    if rec.empty:
        print(f"no windows yet: a reading needs 20 rows (10 s @ 2 Hz); "
              f"input has {n_rows} rows - append telemetry and run --live again.")
        return

    rec["_k"] = rec["engine_id"].astype(str) + "/" + rec["mission_id"].astype(str)
    fault_cols = [c for c in rec.columns if c.startswith("fault_probability_")]

    anchors = {}
    if df is not None and not df.empty:
        dfk = df.copy()
        dfk["_k"] = dfk["engine_id"].astype(str) + "/" + dfk["mission_id"].astype(str)
        for key, gg in dfk.groupby("_k", sort=False):
            t = np.sort(gg["time_in_mission_s"].to_numpy(dtype="float64"))
            dt = float(np.median(np.diff(t))) if len(t) > 1 else 0.5
            anchors[key] = (float(t[0]), max(dt, 1e-4))

    lines = []
    state_changed = False
    for key, g in rec.groupby("_k", sort=False):
        g = g.sort_values("window_end_time")
        last = float(state.get(key, -1.0))
        m0, dt = anchors.get(key, (0.0, 0.5))
        reset_gap = LIVE_STATE_RESET_TILES * LIVE_TILE_SECONDS
        if last >= 0.0 and (float(g["window_end_time"].iloc[0]) - last > reset_gap
                            or float(g["window_end_time"].iloc[0]) < last - dt):
            last = -1.0
        rel = g["window_end_time"].to_numpy(dtype="float64") - m0 + dt
        is_tile = np.isclose(rel % LIVE_TILE_SECONDS, 0.0, atol=max(1e-6, dt / 4.0))
        neww = (g["window_end_time"].to_numpy(dtype="float64") > last)
        tiles = g[is_tile & neww]
        for _, row in tiles.iterrows():
            m = int(round((float(row["window_end_time"]) - m0 + dt) / LIVE_TILE_SECONDS))
            t_start = m0 + (m - 1) * LIVE_TILE_SECONDS
            t_end = m0 + m * LIVE_TILE_SECONDS
            probs = {c[len("fault_probability_"):]: float(row[c]) for c in fault_cols}
            top = sorted(probs.items(), key=lambda kv: kv[1], reverse=True)[:2]
            conf = ", ".join(f"{nm}={pv:.2f}" for nm, pv in top if pv > 0.01)
            ft = "none" if str(row["predicted_faults"]) == "none" else str(row["predicted_faults"])
            lines.append(
                f"   {t_start:7.1f}-{t_end:6.1f}s"
                f" | ANOMALY={float(row['anomaly_score']):6.3f}"
                f" | FAULT={ft:<10s} ({conf if conf else 'p<0.01'})"
                f" | RUL={_fmt(_fval(row['rul_mean_hours']), 1)}h"
                f" [{_fmt(_fval(row['rul_lower_hours']), 1)}..{_fmt(_fval(row['rul_upper_hours']), 1)}]"
                f" | HEALTH={_fmt(_fval(row['health_index']), 1)}")
            state[key] = max(last, float(row["window_end_time"]))
            state_changed = True

    if state_changed:
        sp.write_text(json.dumps(state, sort_keys=True, indent=2), encoding="utf-8")

    print(f"total input rows           : {n_rows:,}")
    print(f"skipped rows               : {skipped:,}")
    print(f"new 10 s tiles completed   : {len(lines):,}")
    if lines:
        print()
        print("        time   ANOMALY   FAULT          RUL(h)              HEALTH")
        for ln in lines:
            print(ln)
        print(f"\nnote: {len(lines):,} new 10 s reading(s). Now add more rows to {INPUT_CSV}")
        print(f"      (each 10 s of 2 Hz data = 20 rows) and run --live again.")
    else:
        print("no NEW completed 10 s tile since the last run - append ~20 rows (10 s @ 2 Hz)")
        print("to the input CSV, then run --live again.")
    if not lines and not rec.empty and all(float(state.get(k, -1.0)) < 0 for k in rec["_k"].unique()):
        print(f"tip: the first reading needs rows 0..19 (t=0..10 s) collected in {INPUT_CSV}.")

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="APET-GUARDIAN inference")
    ap.add_argument("--live", action="store_true",
                    help="incremental mode: report only 10 s tiles completed since the last run")
    args = ap.parse_args()
    main(live=args.live)
