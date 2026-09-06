# High altitude probe
from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from . import config as C
from .pipeline import Pipeline

ROOT = Path(C.OUT_ROOT).parent
sys.path.insert(0, str(ROOT.resolve()))

import generator_v30_physics_locked as G

PROBE_OUT = Path(C.OUT_ROOT) / "probe_highalt"
REPORT = PROBE_OUT / "envelope_probe_report.json"
FAULTS = ["misfire", "injector_degradation", "turbo_issue", "lubrication_issue",
          "sensor_drift", "overheating", "electrical_fault", "bearing_fault"]
_CRUISE_ALT = 5000.0
_PROBE_DURATION_S = 9000.0

# High Alt Profile
def _high_alt_profile(n_samples, mission_duration_s, t_start, mission_type="standard_recon", dt=G.DT):
    timestamps = np.arange(n_samples, dtype=float) * dt
    t_global_s = t_start + timestamps
    duration = max(mission_duration_s, 1.0)
    alt = _CRUISE_ALT

    phases = ["takeoff", "climb", "cruise", "loiter", "descent", "landing"]
    splits = [0.04, 0.10, 0.50, 0.72, 0.86, 0.96, 1.0]
    boundaries = np.array([0.0] + [s * duration for s in splits])
    idx = np.clip(np.digitize(timestamps, boundaries) - 1, 0, len(phases) - 1)
    phases_arr = np.array(phases, dtype=object)[idx]
    time_phase = timestamps - boundaries[idx]

    thr = np.zeros(n_samples)
    pitch = np.zeros(n_samples)
    vz = np.zeros(n_samples)
    target_speed = np.zeros(n_samples)
    target_altitude = np.zeros(n_samples)

    cruise_thr = {5000.0: 0.72, 8000.0: 0.84}.get(alt, 0.78)
    climb_dur = max(duration * (0.50 - 0.10), 60.0)
    vz_climb = min(float(alt / climb_dur), 6.0)

    for p in phases:
        m = phases_arr == p
        nn = int(m.sum())
        if not nn:
            continue
        if p == "takeoff":
            thr[m] = np.linspace(0.85, 1.00, nn); pitch[m] = 0.86
            vz[m] = 1.5; target_speed[m] = 42.0; target_altitude[m] = 50.0
        elif p == "climb":
            thr[m] = 0.95; pitch[m] = 0.78; vz[m] = vz_climb
            target_speed[m] = 48.0; target_altitude[m] = alt
        elif p == "cruise":
            thr[m] = cruise_thr; pitch[m] = 0.70; vz[m] = 0.0
            target_speed[m] = 54.0; target_altitude[m] = alt
        elif p == "loiter":
            thr[m] = cruise_thr - 0.05; pitch[m] = 0.62; vz[m] = 0.0
            target_speed[m] = 55.0; target_altitude[m] = alt
        elif p == "descent":
            thr[m] = 0.28; pitch[m] = 0.38; vz[m] = -2.8
            target_speed[m] = 48.0; target_altitude[m] = 1000.0
        elif p == "landing":
            thr[m] = np.linspace(0.25, 0.12, nn); pitch[m] = 0.24
            vz[m] = -1.6; target_speed[m] = 40.0; target_altitude[m] = 0.0

    rng = np.random.default_rng(int(abs(t_start) * 10 + mission_duration_s) % (2 ** 32 - 1))
    low_freq = G.first_order_lag(rng.normal(0.0, 1.0, n_samples), tau=12.0, dt=dt)
    low_freq = low_freq / max(np.std(low_freq), 1e-6)
    thr = np.clip(thr + 0.018 * low_freq, 0.0, 1.0)
    pitch = np.clip(pitch + 0.012 * low_freq, 0.05, 1.0)
    target_speed = np.clip(target_speed, G.CONST.min_operating_speed_mps, G.CONST.max_operating_speed_mps)
    target_speed += 0.5 * G.first_order_lag(rng.normal(0.0, 1.0, n_samples), tau=20.0, dt=dt)
    target_speed = np.clip(target_speed, G.CONST.min_operating_speed_mps, G.CONST.max_operating_speed_mps)
    vz = vz + 0.15 * G.first_order_lag(rng.normal(0.0, 1.0, n_samples), tau=10.0, dt=dt)
    return (timestamps, t_global_s, phases_arr, time_phase,
            G.first_order_lag(vz, tau=3.0, dt=dt), target_speed, target_altitude,
            G.first_order_lag(thr, tau=1.5, dt=dt), G.first_order_lag(pitch, tau=1.2, dt=dt))

# Generate
def generate(altitudes, missions_per_alt, force=False):
    missions_dir = PROBE_OUT / "missions"
    missions_dir.mkdir(parents=True, exist_ok=True)
    G.generate_mission_profile = _high_alt_profile
    created, reused = [], 0
    for ai, alt in enumerate(altitudes):
        globals()["_CRUISE_ALT"] = alt
        for seas in range(missions_per_alt):
            mid = f"highalt_{int(alt)}_normal_{seas:02d}"
            p = missions_dir / f"{mid}.csv"
            if p.exists() and not force:
                reused += 1
                continue
            tol = G.EngineManufacturingTolerance(engine_id=f"probe_eng_{ai:02d}_{seas:02d}")
            twin = G.ApexPhysicalEngineTwin(tolerance=tol, initial_wear=0.08)
            df = G.create_mission(mid, twin, fault_type="normal",
                                  mission_duration_s=_PROBE_DURATION_S)
            df.to_csv(p, index=False)
            created.append(mid)
        for seas, fname in enumerate(FAULTS):
            mid = f"highalt_{int(alt)}_{fname}_{seas:02d}"
            p = missions_dir / f"{mid}.csv"
            if p.exists() and not force:
                reused += 1
                continue
            tol = G.EngineManufacturingTolerance(engine_id=f"probe_eng_{ai:02d}_{seas:02d}")
            twin = G.ApexPhysicalEngineTwin(tolerance=tol, initial_wear=0.15)
            s_mode = "gradual_drift" if fname == "sensor_drift" else "none"
            df = G.create_mission(
                mid, twin, fault_type=fname, mission_duration_s=_PROBE_DURATION_S,
                fault_onset_s=3000.0, fault_ramp_s=60.0, fault_peak_severity=0.5,
                sensor_fault_mode=s_mode,
            )
            df.to_csv(p, index=False)
            created.append(mid)
    print(f"[probe] generated {len(created)} new high-alt mission(s), reused {reused}")

# Infer
def infer(altitudes, missions_per_alt):
    import run_apet_guardian_pipeline as pipe

    flights = sorted(PROBE_OUT.glob("missions/*.csv"))
    frames = [pd.read_csv(str(p), low_memory=False) for p in flights]
    probe_inp = pd.concat(frames, ignore_index=True)
    inp = PROBE_OUT / "inp_probe.csv"
    probe_inp.to_csv(inp, index=False)

    art = pipe.load_bundle(pipe.MODEL_DIR)
    raw = pipe.read_and_validate_input(inp)
    df, warns = pipe.preprocess(raw)
    rec, skipped = pipe.inference_rows(art, df, warns)
    rec = pipe.add_mission_risk(rec)
    op = PROBE_OUT / "op_probe.csv"
    rec.to_csv(op, index=False)

    train_win = _train_envelope()
    report = _report(rec, df, skipped, train_win)
    REPORT.write_text(json.dumps(report, indent=2, sort_keys=True, default=float))
    print(f"[probe] inference done: {len(rec)} windows -> {op}")
    return report

# Train Envelope
def _train_envelope():
    pl = Pipeline(out_root=C.OUT_ROOT)
    pl.assign_splits()
    alt, spd = [], []
    for kind in ("normal", "faulty"):
        for split in ("train",):
            st = pl._load(kind, split, stats=True)["stats"]
            names = pl.stats_names()
            alt.append(st[:, names.index("altitude_m__mean")])
            spd.append(st[:, names.index("airspeed_mps__mean")])
    alt = np.concatenate(alt); spd = np.concatenate(spd)
    return {
        "altitude_m": {"p50": float(np.percentile(alt, 50)), "p99": float(np.percentile(alt, 99)),
                       "p99.9": float(np.percentile(alt, 99.9)), "max": float(alt.max())},
        "airspeed": {"min": float(spd.min()), "p50": float(np.percentile(spd, 50)),
                     "max": float(spd.max())},
        "n_windows": int(len(alt)),
    }

# Report
def _report(rec, df, skipped, env):
    from sklearn.metrics import f1_score, precision_score, recall_score
    needs = ["altitude_m", "airspeed_mps", "air_density_kg_m3"]
    missions = {}
    for mid, g in df.groupby(["engine_id", "mission_id"], sort=False):
        m = mid[1]
        a = g[needs].mean()
        missions[m] = {
            "max_altitude_m": round(float(g["altitude_m"].max()), 1),
            "cruise_alt_median_m": round(float(g["altitude_m"].median()), 1),
            "air_density_at_max": round(float(g["air_density_kg_m3"].loc[g["altitude_m"].idxmax()]), 4),
        }
    out = rec.merge(df[["engine_id", "mission_id", "altitude_m", "airspeed_mps",
                        "air_density_kg_m3"]].groupby(["engine_id", "mission_id"],
                                                      as_index=False).max(),
                    on=["engine_id", "mission_id"], how="left")
    ood = out["altitude_m"] > env["altitude_m"]["p99"]
    faults = [c[len("fault_probability_"):] for c in rec.columns if c.startswith("fault_probability_")]
    per = {}
    for f in faults:
        gt = rec[f"gt_{f}"].to_numpy(dtype=float)
        pred = np.array([f in r for r in rec["predicted_faults"]], dtype=float)
        if gt.sum() < 3:
            per[f] = None
            continue
        per[f] = {
            "f1": float(f1_score(gt, pred, zero_division=0)),
            "precision": float(precision_score(gt, pred, zero_division=0)),
            "recall": float(recall_score(gt, pred, zero_division=0)),
            "n_positive_windows": int(gt.sum()),
        }
    healthy = rec["gt_anomaly"].to_numpy(dtype=bool)
    return {
        "training_envelope": env,
        "probe": {
            "n_missions": int(df["mission_id"].nunique()),
            "n_windows": len(rec),
            "skipped_windows": int(skipped),
            "max_altitude_m": round(float(df["altitude_m"].max()), 1),
            "n_windows_above_train_p99": int(ood.sum()),
            "mission_overview": missions,
            "per_fault_high_alt": per,
            "anomaly_alert_rate": round(float(rec["anomaly_alert"].mean()), 4),
            "mission_risk_elevated": [m for m, v in rec.groupby("mission_id")["mission_risk"].first().items() if v == "elevated"],
        },
    }

# Main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--alt", nargs="+", type=float, default=[5000.0, 8000.0])
    ap.add_argument("--missions-per-alt", type=int, default=3)
    ap.add_argument("--force", action="store_true", help="regenerate existing missions")
    args = ap.parse_args()
    generate(args.alt, args.missions_per_alt, force=args.force)
    report = infer(args.alt, args.missions_per_alt)
    env, pr = report["training_envelope"], report["probe"]
    print(f"training altitude p50/p99/p99.9/max = "
          f"{env['altitude_m']['p50']:.0f}/{env['altitude_m']['p99']:.0f}/"
          f"{env['altitude_m']['p99.9']:.0f}/{env['altitude_m']['max']:.0f} m")
    print(f"probe max altitude = {pr['max_altitude_m']:.0f} m "
          f"({pr['n_windows_above_train_p99']}/{pr['n_windows']} windows OOD)")
    print("per-fault high-alt F1:")
    for f, v in pr["per_fault_high_alt"].items():
        if v:
            print(f"  {f:<22s} F1={v['f1']:.3f} P={v['precision']:.3f} R={v['recall']:.3f} n={v['n_positive_windows']}")
        else:
            print(f"  {f:<22s} (too few positives)")
    print("elevated-risk missions:", pr["mission_risk_elevated"])
    print(f"report -> {REPORT}")

if __name__ == "__main__":
    main()
