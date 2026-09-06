# Data & Models — Inputs / Outputs

This repo is shareable: `sample_data/` has 11 files (64 MB, 1 per fault + 1 normal + 1 RUL truncated to 5000 rows) so you and your friend can clone and run without the full 6.95 GB `DATA/` (~120 normal + 600 faulty + 20 RUL) or 460 MB bundle. Full `DATA` regenerates via `generator_v30_physics_locked.py` and `APET_GUARDIAN_OUT/artifacts/apet_bundle.pkl` regenerates via `python -m apet_guardian.run_pipeline --quick`.

## Sample Data (committed, for sharing)

```
sample_data/
├── normal/mission_normal_001.csv              # healthy, 2.17 MB, ~1200 rows, 2 Hz
├── faulty/
│   ├── mission_misfire_001_sev1.csv           # 6.58 MB
│   ├── mission_injector_degradation_002_sev0.csv  # 6.6 MB
│   ├── mission_turbo_issue_003_sev0.csv       # 6.2 MB
│   ├── mission_lubrication_issue_012_sev1.csv # 6.38 MB
│   ├── mission_sensor_drift_005_sev0.csv      # 3.96 MB
│   ├── mission_overheating_014_sev0.csv       # 7.06 MB
│   ├── mission_electrical_fault_031_sev0.csv  # 7.93 MB
│   ├── mission_bearing_fault_032_sev0.csv     # 4.91 MB
│   └── mission_compound_bearing_fault_electrical_fault_004_sev0.csv # 4.44 MB (2 faults)
└── rul/life_001.csv                           # 8.38 MB, 5000 rows truncated from 2 min lifecycle (health 100→0)
```

All have 119 cols `SCHEMA_COLUMNS` (see `generator_v30_physics_locked.py:277`): 9 metadata (`engine_id, mission_id, global_time_s, time_in_mission_s, phase...`) + 72 features (flight, engine, vibration, electrical) + 38 targets (`true_*`, `fault_*_active`, `simulated_rul_hours`, `health_state`).

Copy to `inp.csv` for demo: `Copy-Item sample_data/faulty/*.csv -Destination inp.csv` or `cat sample_data/normal/*.csv > inp.csv` then `python run_apet_guardian_pipeline.py`.

## Full Data (ignored, local only)

* `DATA/normal` 120, `DATA/faulty` 600 (9 subdirs), `DATA/rul` 20 — required for training (`config.py:LIMITS`). Ignored via `.gitignore: DATA/` for sharing.

## Pipeline I/O

**Input `inp.csv` → 2 Hz telemetry** (`run_apet_guardian_pipeline.py:93`):
- Required `engine_id, mission_id, global_time_s` + 24 sensors `throttle_actual, true_rpm, air_density_kg_m3, altitude_m, rpm, boost_pressure_kPa, manifold_pressure_kPa, cht_cyl_avg_C, egt_cyl_avg_C, oil_pressure_kPa, oil_temp_C, coolant_temp_C, fuel_flow_g_s, fuel_temp_C, rail_pressure_bar, vib_rms_g, battery_voltage_V, airspeed_mps, vertical_speed_mps, engine_power_command_kW, true_brake_power_kW` (aliases allowed)
- Output `op.csv` per 10s window (`SEQ_LEN 20, STRIDE 10, MAX_GAP 1.5`): `window_end_time, anomaly_score@0.5, predicted_faults, fault_probability_*(8), rul_mean/lower/upper (q10/50/90), health_index, mission_risk`

## Model Inputs / Outputs

| Model | Code | Input (per 10s window) | Output |
|---|---|---|---|
| **Physics baseline** | `features.py:38` `PhysicsFeatureModel` (Ridge+Poly2 per fault, healthy-only) | 4 `CONTEXT_COLS` (`throttle_actual, true_rpm, air_density, altitude`) | 31 `SEQ_FEATURES` residuals: `expected_boost/cht/egt/oilp`, `*_residual_norm` (phase z-score), `thermal_stress, fuel_per_rpm, oil_efficiency, vibration_trend` + 8 `CUMULATIVE_FEATURES` (`cum_thermal_stress...`) |
| **IsolationForest** (anomaly) | `training.py:30` `fit_isolated_forest` `300 est, contam 0.01` | 192-dim `STATS_SIGNALS 23 × 8 stats` (`mean/std/min/max/p90/range/slope/final` `windowing.py:92`) + 8 cum → `stat_scaler` | `anomaly_score` 0-1 (`anomaly_score()` 0.30 weight in blended) |
| **PCA** (anomaly residual) | `training.py:76` `fit_pca_anomaly` `n=32` | 192-dim stats (healthy only) | residual RMSE 0-1 (0.15 weight) |
| **BiLSTMEncoder** | `models.py:16` `128 hidden ×2 bidir → emb 256` | `seq (20,31)` `SEQ_FEATURES` scaled via `seq_scaler` | `emb (256)` + `fault_prob` 0-1 (sigmoid logit, 0.55 weight in blended) |
| **FaultFusionMLP** | `models.py:38` `[256,128] Dropout 0.25 → 8` | `stats 192 + if_score 1 + emb 256 = 449` | 8 `fault_probability_*` 0-1 (`misfire, injector_degradation, turbo_issue, lubrication_issue, sensor_drift, overheating, electrical_fault, bearing_fault`), thresholded `PER_CLASS_THRESHOLD` (e.g., `overheating 0.28` + 2-window persistence `voice_alert_system.py`) |
| **QuantileMLP** (RUL) | `models.py:75` `[256,128] → 3 heads` | 192-dim stats | `rul_lower/median/upper` hours `0-50` (`RUL_MAX_TARGET_H`, pinball loss `QUANTILES 0.10/0.50/0.90`) |
| **HealthMLP** | `models.py:102` `[256,128] → 1` | 192-dim stats | `health_index` 0-100 (MSE,approx `true_overall_health_index`) |

**Blended anomaly:** `0.30*IF +0.15*PCA +0.55*LSTM` (`training.py:106`).

## For 3D UAV Flight Demo (you + friend)

* Use `sample_data` to drive a 3D flight: each `sample_data/faulty/*.csv` has `altitude_m, airspeed_mps, vertical_speed_mps, flight_path_angle_deg, rpm, vibration` at 2 Hz — map `altitude_m` (0-3200m) to Z, `time_in_mission_s` to timeline, `fault_type` + `health_index` to color/particle effects, `anomaly_score` to warning pulse, `predicted_faults` from `op.csv` to trigger voice (`nlp/voice_alert_system.py: female=True, rate 150`) and health report.
* Full `DATA` (6.95 GB) or `generator_v30_physics_locked.py` can generate longer missions (e.g., 9000s high-alt) for extended 3D flights.
* Keep `sample_data` in git, keep `DATA/` and `APET_GUARDIAN_OUT/` ignored — clone is ~70 MB (code + sample), not 41 GB.

Regenerate full bundle if needed: `python -m apet_guardian.run_pipeline --quick` → `APET_GUARDIAN_OUT/artifacts/apet_bundle.pkl` (then `run_apet_guardian_pipeline.py` will map CUDA→CPU automatically for sharing).
