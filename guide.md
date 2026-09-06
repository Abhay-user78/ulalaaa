# APET-GUARDIAN — Pipeline Guide

Physics-locked UAV engine digital twin (RUSTOM-2 / TAPAS, 4-stroke 2.0L 125 kW diesel, BEM propeller D=1.65m, Cp 0.0821 @ pitch 0.68 J0.5) → 2 Hz telemetry (119 cols) → 10 s causal windows (`SEQ_LEN 20`, `STRIDE 10`) → 4 heads: **anomaly, fault type (8), RUL, health**. One-command training, frozen single-file inference.

## 1. Folder Map (pipeline-essential vs cache)

```
D:\testing\
├── generator_v30_physics_locked.py       # CREATES DATA — deterministic aero-thermo + 3-DOF flight
├── apet_guardian/                       # CODE — 15 files (keep)
│   ├── __init__.py          # re-exports Pipeline, APETGuardian
│   ├── config.py            # single source: paths, windowing, faults, hyperparams
│   ├── features.py          # Stage A: physics baseline (healthy-only)
│   ├── windowing.py         # Stage B: causal 10s windows, 192-dim stats
│   ├── models.py            # MODEL DEFINITIONS (see §3)
│   ├── training.py          # ALL TRAINING (see §3)
│   ├── pipeline.py          # ORCHESTRATOR A-G → bundle.pkl
│   ├── inference.py         # FROZEN DEPLOY (one window)
│   ├── predict.py           # CLI stream
│   ├── evaluate.py          # frozen harness (metrics/plots/report.md)
│   ├── retrain_heads.py     # head-only retrain → v2 bundle
│   ├── tune.py              # per-class threshold tuning
│   ├── validate.py          # regression vs baseline
│   ├── run_pipeline.py      # train driver --quick
│   └── probe_highalt.py     # OOD probe (monkey-patches generator)
├── run_apet_guardian_pipeline.py # FROZEN SINGLE-FILE: inp.csv → op.csv (no training)
├── DATA/                    # INPUT — keep 6.95 GB mandatory
│   ├── normal/      (120 csv)  # healthy missions
│   ├── faulty/      (600, 9 subdirs: misfire, bearing_fault, compound_faults...) # fault missions
│   ├── rul/         (20 csv)   # lifecycle for RUL/health
│   ├── feature_target_manifest.json # schema docs (keep)
│   ├── ml/          (28 GB, 7 files) # precomputed windows — CACHE, delete to reclaim
│   └── vibration_raw/ (11 MB, 1201 npz) # raw 2048 Hz vibe — optional
├── APET_GUARDIAN_OUT/       # OUTPUT — generated
│   ├── artifacts/apet_bundle.pkl (516 MB) # MODEL STORE — keep ONE (see §4)
│   ├── windows/ (0.82 GB, 24 .bin) # sharded windows — cache, delete
│   ├── probe_highalt/ (0.85 GB) # OOD probe — cache, delete
│   └── *.json (reports)     # validate_baseline, threshold_tuned — logs
├── inp.csv (2.5 MB)         # example input (5.6k rows, 15 missions)
├── op.csv (0.27 MB)         # example output (547 windows)
└── .git/ (0.93 GB)          # VCS
```

Keep `apet_guardian/ + generator + run_apet_guardian_pipeline.py + DATA/normal+f faulty+rul + APET_GUARDIAN_OUT/artifacts/apet_bundle.pkl + inp.csv`. Delete `__pycache__/`, `windows/`, duplicates (`*_cpu.pkl`, `*_v2.pkl`, `*_v1_backup.pkl`), `probe_highalt/`, `ml/`, `vibration_raw/`, `Temp/opencode/` to reclaim ~31 GB.

## 2. Where Is Model Code

| Model | Definition | Training | Key lines |
|-------|------------|----------|-----------|
| **BiLSTMEncoder** (128 hidden, 2 layers, bidir, emb 256) | `apet_guardian/models.py:16` | `training.py:131 fit_bilstm()` Adam `lr1e-3`, BCE `pos_weight`, early-stop `PATIENCE 6`, `EPOCHS 40` | `encode() → out[:,-1,:]` |
| **FaultFusionMLP** (`stats 192 + if_score 1 + emb 256 → [256,128] → 8`) | `models.py:38` | `training.py:215 fit_fusion()` `BCE/focal(alpha0.25,g2.0) + class_weights` | `hidden [256,128] Dropout 0.25` |
| **QuantileMLP** (`[256,128] → 3 heads q10/q50/q90`) | `models.py:75` | `training.py:357 fit_quantile()` pinball loss `line 99` | `_fit_tabular:311` |
| **HealthMLP** (`[256,128] → 1`) | `models.py:102` | `training.py:371 fit_health()` MSE | `torso [256,128]` |
| **IsolationForest** (`300 est, contam 0.01`) | `sklearn` | `training.py:30 fit_isolated_forest()` `IsolationForest(...).fit(healthy_stats)` | `line 40` |
| **PCA** (`n=32`) | `sklearn` | `training.py:76 fit_pca_anomaly()` `PCA().fit(healthy_stats)` | `line 86` |
| **Physics baseline** | `features.py:38 PhysicsFeatureModel` | `features.py:60 fit()` Ridge+Poly2 per fault (boost/cht/egt/oilp) + `pipeline.py:134 build_physics_model()` | `phase_stats:81`, `transform:148` |
| **Orchestrator** | `pipeline.py:79 Pipeline` | `train_and_report:241` (scalers→IF→PCA→BiLSTM→Fusion→RUL/Health→evaluate_all:378) | Stages A-G |

No other ML code — `generator_v30_physics_locked.py:984 ApexPhysicalEngineTwin` is **deterministic physics**, not ML.

## 3. Where Are Models Stored

- **Trained bundle:** `APET_GUARDIAN_OUT/artifacts/apet_bundle.pkl` (516 MB, pickle, also `*_cpu.pkl` duplicate). Contains `physics` (Ridge models), `if` (IsolationForest + ref), `anomaly_extra` (PCA), `lstm`, `fusion`, `quantile`, `health`, `stat_scaler`, `seq_scaler`, `faults` (8), thresholds. **Keep one copy** — `run_apet_guardian_pipeline.py:43 MODEL_DIR` reads it.
- **Reports:** `validate_baseline.json` (AUC 0.926), `threshold_tuned.json` (per-class thresholds 0.24–0.68, hardcoded in `run_apet_guardian_pipeline.py:75`), `head_retrain_report.json`, `fusion_ab_report.json`.

## 4. How to Execute Pipeline

### 4.1 Train from scratch (5–15 min)
```bash
python -m apet_guardian.run_pipeline --quick          # 24/40/6 limits, fast
python -m apet_guardian.run_pipeline --fresh          # full: normal 120, faulty 600, rul 20
python -m apet_guardian.run_pipeline --normal 120 --faulty 600 --rul 20
# outputs: APET_GUARDIAN_OUT/artifacts/apet_bundle.pkl + windows/*.bin + report.json
# env override: APET_DATA_ROOT=D:\testing\DATA APET_OUT_ROOT=D:\testing\APET_GUARDIAN_OUT
```

### 4.2 Inference — single CSV (frozen, no training)
```bash
# 1. put 2 Hz rows into D:\testing\inp.csv (≥20 rows/mission, header required)
# required: engine_id, mission_id, global_time_s + 24 sensors:
#   throttle_actual, true_rpm, air_density_kg_m3, altitude_m, rpm, boost_pressure_kPa,
#   manifold_pressure_kPa, cht_cyl_avg_C, egt_cyl_avg_C, oil_pressure_kPa, oil_temp_C,
#   coolant_temp_C, fuel_flow_g_s, fuel_temp_C, rail_pressure_bar, vib_rms_g,
#   battery_voltage_V, airspeed_mps, vertical_speed_mps, engine_power_command_kW, true_brake_power_kW
# aliases allowed: timestamp→global_time_s, throttle→throttle_actual, boost_kPa etc. (run_apet_guardian_pipeline.py:103)
python run_apet_guardian_pipeline.py                 # writes D:\testing\op.csv (547 windows for 5.6k rows)
python run_apet_guardian_pipeline.py --live          # per 10s tile, state live_state.json, remembers last window_end
```
`op.csv` (35 cols): `window_end_time, anomaly_score@0.5, anomaly_alert, predicted_faults, fault_probability_*(8), rul_mean/lower/upper (q10/50/90), health_index, data_quality_warning, gt_* if present, mission_risk (elevated if ≥3 consecutive hot)`

### 4.3 Streaming / programmatic
```bash
python -m apet_guardian.predict DATA/normal/mission_*.csv --csv-out out_dir --bundle APET_GUARDIAN_OUT/artifacts/apet_bundle.pkl
```
```python
from apet_guardian.inference import APETGuardian
g = APETGuardian()  # loads APET_GUARDIAN_OUT/artifacts/apet_bundle.pkl
g.predicts(df_last20)  # → {anomaly_score, fault_prob, probable_faults, rul_hours, health_index, top_influencing_features}
```

### 4.4 Evaluate frozen bundle (no retrain)
```bash
python -m apet_guardian.evaluate --dataset D:\testing\DATA --bundle APET_GUARDIAN_OUT/artifacts/apet_bundle.pkl --out-base APET_GUARDIAN_OUT/eval
# writes predictions/window_predictions.csv, metrics/metrics.json, report.md, plots/
```

### 4.5 Retrain heads / Tune thresholds / Validate
```bash
python -m apet_guardian.retrain_heads                 # head-only (keeps LSTM) → apet_bundle_v2.pkl
python -m apet_guardian.retrain_heads --tune-thresholds  # smoothed F1 on VAL → threshold_tuned.json
python -m apet_guardian.retrain_heads --evaluate      # alias for validate
python -m apet_guardian.tune                          # same as above, standalone
python -m apet_guardian.validate                      # vs validate_baseline.json, CONTRACT 0.02
python -m apet_guardian.validate --update-baseline    # snapshot new baseline
python -m apet_guardian.validate --gate-off           # show without overheating gate
```

### 4.6 Probe high-altitude (OOD, no retrain)
```bash
python -m apet_guardian.probe_highalt --alt 5000 8000 --missions-per-alt 3
# writes APET_GUARDIAN_OUT/probe_highalt/missions/*.csv, inp_probe.csv, op_probe.csv, envelope_probe_report.json
```

## 5. Data Format

**inp.csv** 35 cols: `engine_id, mission_id, global_time_s, 21 sensors, fault_active, fault_*_active (8), simulated_rul_hours, true_overall_health_index` — example row: `eng_normal_cruise,normal_cruise,0.5,0.925...,2812.5,... ,0,0,... ,99.97`

**op.csv** 35 cols: `engine_id, mission_id, window_end_time, anomaly_score, anomaly_alert, normal_probability, fault_probability, predicted_faults, fault_probability_*(8), rul_*, health_index, data_quality_warning, gt_*, mission_risk`

**Windowing:** `SAMPLE_DT 0.5, SEQ_LEN 20 (10s), STRIDE 10 (5s), MAX_GAP 1.5s` (`config.py:14`), `SPLIT 70/15/15 SEED 2026`, grouped by mission (no leakage) `windowing.py:20`.

## 6. Verification After Clean
```bash
python run_apet_guardian_pipeline.py  # expect 547 windows, anomaly AUC 0.881, binary F1 0.925
python -m apet_guardian.validate       # expect all contracted metrics hold (AUC 0.926)
```
