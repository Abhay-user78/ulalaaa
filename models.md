# APET-GUARDIAN Models Documentation

## Overview

APET-GUARDIAN is a UAV engine digital twin that detects anomalies, classifies 8 fault types, predicts remaining useful life (RUL), and estimates overall health index. It runs in real-time from raw CSV telemetry.

The system has 6 neural network / ML models plus a physics-based feature engine. All models are trained offline on synthetic data and bundled into `APET_GUARDIAN_OUT/artifacts/apet_bundle.pkl`.

---

## Input: CSV Telemetry

`inp.csv` contains raw sensor readings from the UAV engine. **No ground-truth labels are required** — the pipeline infers everything from telemetry.

**Leakage-free:** All `true_*`, `fault_*`, `sensor_drift_flag_*`, `simulated_rul_hours`, `health_state`, `failure_*`, `severity_*` columns are removed. The model never sees the answers.

### Required Input Columns (76 total)

| # | Column | Unit | Description |
|---|--------|------|-------------|
| 1 | `engine_id` | string | Unique engine identifier |
| 2 | `mission_id` | string | Unique mission identifier |
| 3 | `global_time_s` | seconds | Timestamp in mission |
| 4 | `phase` | string | Flight phase (takeoff, climb, cruise, loiter, dash, descent, approach, landing, etc.) |
| 5 | `mission_type` | string | normal / faulty / rul |

#### Engine Core
| # | Column | Unit | Description |
|---|--------|------|-------------|
| 6 | `rpm` | RPM | Engine shaft speed |
| 7 | `throttle_actual` | 0-1 | Actual throttle position |
| 8 | `throttle_cmd` | 0-1 | Commanded throttle position |
| 9 | `fadec_mode` | int | FADEC control mode |

#### Intake / Boost
| # | Column | Unit | Description |
|---|--------|------|-------------|
| 10 | `boost_pressure_kPa` | kPa | Turbocharger boost pressure |
| 11 | `manifold_pressure_kPa` | kPa | Intake manifold pressure |

#### Temperatures
| # | Column | Unit | Description |
|---|--------|------|-------------|
| 12 | `egt_cyl_avg_C` | °C | Average exhaust gas temperature |
| 13 | `cht_cyl_avg_C` | °C | Average cylinder head temperature |
| 14 | `coolant_temp_C` | °C | Coolant temperature |
| 15 | `oil_temp_C` | °C | Oil temperature |
| 16 | `oil_pressure_kPa` | kPa | Oil pressure |

#### Fuel System
| # | Column | Unit | Description |
|---|--------|------|-------------|
| 17 | `fuel_flow_g_s` | g/s | Fuel flow rate |
| 18 | `fuel_temp_C` | °C | Fuel temperature |
| 19 | `rail_pressure_bar` | bar | Common rail fuel pressure |
| 20 | `inj_duration_ms` | ms | Injector pulse width |
| 21 | `fuel_mass_kg` | kg | Current fuel mass on board |
| 22 | `fuel_used_kg` | kg | Total fuel consumed |
| 23 | `sortie_start_fuel_kg` | kg | Fuel at mission start |
| 24 | `SOI_deg_bTDC` | degrees | Start of injection timing |

#### Vibration (15 channels)
| # | Column | Unit | Description |
|---|--------|------|-------------|
| 25 | `vib_rms_g` | g | Overall vibration RMS |
| 26 | `vib_1x_rms` | g | 1x RPM vibration amplitude (bearing wear indicator) |
| 27 | `vib_2x_rms` | g | 2x RPM vibration amplitude (misalignment indicator) |
| 28 | `vib_bpfo_hz` | Hz | Ball Pass Frequency Outer race |
| 29 | `vib_bpfi_hz` | Hz | Ball Pass Frequency Inner race |
| 30 | `vib_bsf_hz` | Hz | Ball Spin Frequency |
| 31 | `vib_psd_peak` | g²/Hz | Peak power spectral density |
| 32 | `vib_psd_peak_freq` | Hz | Frequency of PSD peak |
| 33 | `vib_1x_hz` | Hz | 1x RPM frequency |
| 34 | `vib_2x_hz` | Hz | 2x RPM frequency |
| 35 | `vib_firing_hz` | Hz | Firing frequency |
| 36 | `vib_fire_order_rms` | g | Fire-order vibration RMS |
| 37 | `vib_crest_factor` | ratio | Peak-to-RMS ratio |
| 38 | `vib_kurtosis` | - | Vibration kurtosis (spikiness) |
| 39 | `vib_spectral_centroid` | Hz | Spectral centroid |

#### Electrical
| # | Column | Unit | Description |
|---|--------|------|-------------|
| 40 | `battery_voltage_V` | V | Battery bus voltage |

#### Flight / Environment
| # | Column | Unit | Description |
|---|--------|------|-------------|
| 41 | `airspeed_mps` | m/s | True airspeed |
| 42 | `vertical_speed_mps` | m/s | Climb/descent rate |
| 43 | `ground_speed_mps` | m/s | Ground speed |
| 44 | `altitude_m` | m | Altitude above sea level |
| 45 | `ambient_P_kPa` | kPa | Ambient pressure |
| 46 | `ambient_T_C` | °C | Ambient temperature |
| 47 | `air_density_kg_m3` | kg/m³ | Air density |
| 48 | `aircraft_mass_kg` | kg | Current aircraft mass |
| 49 | `wind_u_mps` | m/s | Wind component |
| 50 | `wind_accel_mps2` | m/s² | Wind acceleration |

#### Power / Propulsion
| # | Column | Unit | Description |
|---|--------|------|-------------|
| 51 | `engine_power_command_kW` | kW | Commanded power |
| 52 | `prop_power_kW` | kW | Propeller power |
| 53 | `prop_pitch_actual` | - | Actual propeller pitch |
| 54 | `prop_pitch_cmd` | - | Commanded propeller pitch |
| 55 | `prop_Cp` | - | Propeller power coefficient |
| 56 | `prop_Ct` | - | Propeller thrust coefficient |
| 57 | `prop_advance_ratio` | - | Propeller advance ratio J |
| 58 | `prop_efficiency` | - | Propeller efficiency |
| 59 | `desired_thrust_N` | N | Commanded thrust |
| 60 | `excess_thrust_N` | N | Thrust minus drag |
| 61 | `drag_N` | N | Aerodynamic drag |
| 62 | `thrust_equilibrium_error_N` | N | Thrust-drag imbalance |
| 63 | `shaft_torque_error_Nm` | Nm | Torque error |
| 64 | `flight_path_angle_deg` | deg | Flight path angle |

#### BSFC / Operating
| # | Column | Unit | Description |
|---|--------|------|-------------|
| 65 | `actual_bsfc_g_kWh` | g/kWh | Actual brake-specific fuel consumption |
| 66 | `engine_bsfc_g_kWh` | g/kWh | Engine BSFC |
| 67 | `actual_operating_hours` | hours | Total engine operating hours |
| 68 | `telemetry_valid` | bool | Telemetry validity flag |

#### Misc (may be present)
| Column | Description |
|--------|-------------|
| `altitude_quantized_m` | Quantized altitude |
| `rpm_quantized` | Quantized RPM |
| `rail_pressure_quantized_bar` | Quantized rail pressure |
| `battery_voltage_raw_V` | Raw battery voltage |
| `refueled_at_sortie_start` | Refuel flag |
| `settled_state` | Settled state flag |
| `target_airspeed_mps` | Target airspeed |
| `target_altitude_m` | Target altitude |

---

## How the Pipeline Processes Input

```
inp.csv (raw telemetry)
    │
    ▼
┌─────────────────────────────┐
│  1. read_and_validate_input │  Read CSV, rename aliases, check required columns
│                             │  Fallback: create true_rpm from rpm, true_brake_power_kW from prop_power_kW
└──────────────┬──────────────┘
               │
               ▼
┌─────────────────────────────┐
│  2. preprocess              │  Compute time_in_mission_s, impute missing values
│                             │  Create missing vibe columns from estimates
└──────────────┬──────────────┘
               │
               ▼
┌─────────────────────────────┐
│  3. Physics Transform       │  Ridge + Poly2 regression on 4 signals
│     (features.py)           │  Produces: expected_boost, expected_cht, expected_egt, expected_oilp
│                             │  Residuals: boost_residual_kPa, cht_residual_C, egt_residual_C, oilp_residual_kPa
│                             │  Normalized residuals: boost_residual_norm, cht_residual_norm, etc.
│                             │  Derived: thermal_stress, cht_stress, fuel_per_rpm_g_rev, oil_efficiency_KPa_per_C
│                             │  Cumulative: cum_thermal_stress, cum_load_stress, cum_egt_excess, etc.
│                             │  vibration_trend (diff of vib_rms_g)
└──────────────┬──────────────┘
               │
               ▼
┌─────────────────────────────┐
│  4. Windowing               │  Sliding windows: 20 rows x 0.5s stride
│     (windowing.py)          │  Each window = 10 seconds of telemetry (20 samples @ 2Hz)
└──────────────┬──────────────┘
               │
               ├──────────────────────────────────────┐
               │                                      │
               ▼                                      ▼
┌──────────────────────────┐          ┌──────────────────────────┐
│ 5a. Sequence features    │          │ 5b. Stat features         │
│  35 channels x 20 timesteps│        │  29 signals x 8 stats     │
│  = 700 floats per window │          │  = 232 floats per window  │
│                          │          │  + 8 cumulative features   │
│  (for LSTM)              │          │  = 240 floats per window   │
└──────────┬───────────────┘          └──────────┬────────────────┘
           │                                     │
           ▼                                     ▼
    ┌─────────────┐                     ┌─────────────────┐
    │  BiLSTM     │                     │  Isolation Forest│
    │  Encoder    │                     │  + PCA Anomaly   │
    └──────┬──────┘                     └────────┬────────┘
           │                                     │
           │  emb: 256-dim vector               │  score: 0-1 anomaly score
           │  prob: binary fault logit          │
           │                                     │
           └──────────────┬─────────────────────┘
                          │
                          ▼
               ┌─────────────────────┐
               │  Fusion MLP         │  Input: stats(240) + anomaly_score(1) + embedding(256) = 497
               │  256 → 128 → 8     │  Output: 8 fault probabilities
               └──────────┬──────────┘
                          │
                          ├──────────────┬──────────────┬──────────────┐
                          ▼              ▼              ▼              ▼
                    ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐
                    │ Fault    │  │ Quantile │  │ Health   │  │ Binary   │
                    │ Labels   │  │ RUL MLP  │  │ Index MLP│  │ Anomaly  │
                    │ (8 type) │  │ (3 quant)│  │ (0-100)  │  │ Score    │
                    └──────────┘  └──────────┘  └──────────┘  └──────────┘
```

---

## Model 1: Physics Feature Engine (`features.py`)

**Type:** Ridge regression + polynomial features (degree 2)

**Purpose:** Learn expected sensor values from operating context (throttle, RPM, air density, altitude), then compute residuals as anomaly features.

**Input:** 4 context columns: `throttle_actual`, `true_rpm`, `air_density_kg_m3`, `altitude_m`

**4 separate Ridge models:**
| Target Signal | Expected Output | Residual | Normalized Residual |
|---------------|----------------|----------|-------------------|
| `boost_pressure_kPa` | `expected_boost` | `boost_residual_kPa` | `boost_residual_norm` |
| `cht_cyl_avg_C` | `expected_cht` | `cht_residual_C` | `cht_residual_norm` |
| `egt_cyl_avg_C` | `expected_egt` | `egt_residual_C` | `egt_residual_norm` |
| `oil_pressure_kPa` | `expected_oilp` | `oilp_residual_kPa` | `oilp_residual_norm` |

**Architecture:**
```
Input: 4 features → PolynomialFeatures(degree=2, include_bias=True) → 15 features
→ Ridge(alpha=10.0) → 1 output (expected sensor value)
```

**Additional derived features (hand-crafted):**
- `thermal_stress` = EGT × (0.4 + 0.6 × RPM_norm) × air_density
- `cht_stress` = CHT × (0.5 + 0.5 × throttle)
- `fuel_per_rpm_g_rev` = fuel_flow / rev_per_second
- `oil_efficiency_KPa_per_C` = oil_pressure / oil_temp
- `vibration_trend` = d(vib_rms_g)/dt

**Cumulative features (8):**
- `cum_thermal_stress`, `cum_load_stress`, `cum_egt_excess`, `cum_cht_excess`
- `cum_vib`, `boost_resid_integrated`, `cum_oil_loss`, `cum_fuel`

**Training data:** Healthy engine missions only (120 missions, ~70% train split)

---

## Model 2: BiLSTM Encoder (`models.py` → `BiLSTMEncoder`)

**Type:** Bidirectional LSTM with classification head

**Purpose:** Encode 20-timestep vibration/sensor sequences into a 256-dim embedding and produce a binary fault/no-fault logit.

**Input:** 35 channels × 20 timesteps = (batch, 20, 35) float32

**Architecture:**
```
Input: (batch, 20, 35)
  → nn.LSTM(input=35, hidden=128, layers=2, bidirectional=True, dropout=0.3)
  → take last hidden state: (batch, 256)        ← embedding
  → nn.Linear(256, 1)                            ← binary logit
```

**Parameters:**
| Parameter | Value |
|-----------|-------|
| Hidden size | 128 |
| Num layers | 2 |
| Bidirectional | Yes |
| Dropout | 0.30 |
| Embedding dim | 256 (128 × 2) |
| Output | 1 logit (binary fault) |

**Training:**
- Loss: BCEWithLogitsLoss with pos_weight for class imbalance
- Optimizer: Adam (lr=1e-3, weight_decay=1e-5)
- Epochs: max 40, early stopping patience=6
- Batch size: 512
- Sequence features (35): rpm, throttle, boost, manifold, CHT, EGT, oil_p, oil_t, coolant, fuel_flow, fuel_temp, rail, vib_rms, vib_1x_rms, vib_2x_rms, vib_bpfo, vib_bpfi, vib_bsf, vib_psd_peak, battery, airspeed, altitude, v_speed, engine_power_cmd, expected_boost, boost_residual, boost_residual_norm, cht_residual_norm, egt_residual_norm, oilp_residual_norm, thermal_stress, cht_stress, fuel_per_rpm, oil_efficiency, vibration_trend

**Output:**
- `embedding`: 256-dim feature vector per window
- `prob`: binary fault probability per window

---

## Model 3: Isolation Forest + PCA Anomaly (`training.py`)

**Type:** Unsupervised anomaly detection (3-component blend)

**Purpose:** Detect anomalous operating conditions from window statistics.

**Input:** 240-dim stat vector per window (29 signals × 8 stats + 8 cumulative)

**Architecture (3 components):**

### Component A: Isolation Forest
```
n_estimators = 300
contamination = 0.01
max_samples = min(4096, N)
Trained on: healthy train windows only
Output: anomaly score ∈ [0, 1]
```

### Component B: PCA Reconstruction Error
```
n_components = 32
Trained on: healthy train windows only
Score = ||x - PCA.inverse_transform(PCA(x))||₂
Normalized to [0, 1] using percentiles from healthy data
```

### Component C: LSTM Binary Probability
```
Already computed by BiLSTM (sigmoid output)
```

**Blend weights:**
| Component | Weight |
|-----------|--------|
| Isolation Forest | 0.30 |
| PCA | 0.15 |
| LSTM probability | 0.55 |

**Final anomaly score:**
```
anomaly = 0.30 × IF_score + 0.15 × PCA_score + 0.55 × LSTM_prob
```

---

## Model 4: Fusion MLP (`models.py` → `FaultFusionMLP`)

**Type:** Multi-layer perceptron for multi-label fault classification

**Purpose:** Classify 8 simultaneous fault types from fused features.

**Input:** Concatenation of:
- 240-dim stat features (from window stats)
- 1-dim anomaly score (from Isolation Forest)
- 256-dim embedding (from BiLSTM)
- **Total: 497 features**

**Architecture:**
```
Input: 497 features
  → Linear(497, 256) → ReLU → Dropout(0.25)
  → Linear(256, 128) → ReLU → Dropout(0.25)
  → Linear(128, 8)
```

**Parameters:**
| Parameter | Value |
|-----------|-------|
| Hidden layers | [256, 128] |
| Dropout | 0.25 |
| Output | 8 logits |
| Loss | BCEWithLogitsLoss with class weights |
| Optimizer | Adam (lr=1e-3, weight_decay=1e-5) |
| Epochs | max 40, patience=6 |
| Batch size | 2048 |

**Output:** 8 fault probabilities (sigmoid)

| Index | Fault Type |
|-------|-----------|
| 0 | misfire |
| 1 | injector_degradation |
| 2 | turbo_issue |
| 3 | lubrication_issue |
| 4 | sensor_drift |
| 5 | overheating |
| 6 | electrical_fault |
| 7 | bearing_fault |

---

## Model 5: Quantile RUL MLP (`models.py` → `QuantileMLP`)

**Type:** Multi-quantile regression MLP

**Purpose:** Predict remaining useful life with uncertainty bounds.

**Input:** 240-dim stat features (same as Fusion input, without anomaly score or embedding)

**Architecture:**
```
Input: 240 features
  → Linear(240, 256) → ReLU → Dropout(0.2)
  → Linear(256, 128) → ReLU → Dropout(0.2)
  → 3 parallel heads:
      Linear(128, 1)  ← q=0.10 (lower bound)
      Linear(128, 1)  ← q=0.50 (median)
      Linear(128, 1)  ← q=0.90 (upper bound)
```

**Parameters:**
| Parameter | Value |
|-----------|-------|
| Hidden layers | [256, 128] |
| Dropout | 0.2 |
| Quantiles | 0.10, 0.50, 0.90 |
| Loss | Pinball loss (quantile regression) |
| Max RUL target | 50 hours |
| Optimizer | Adam (lr=1e-3, weight_decay=1e-5) |

**Output:** 3 values per window: [lower_RUL, median_RUL, upper_RUL] in hours

---

## Model 6: Health Index MLP (`models.py` → `HealthMLP`)

**Type:** Single-output regression MLP

**Purpose:** Predict overall engine health index (0-100 scale).

**Input:** 240-dim stat features

**Architecture:**
```
Input: 240 features
  → Linear(240, 256) → ReLU → Dropout(0.2)
  → Linear(256, 128) → ReLU → Dropout(0.2)
  → Linear(128, 1)
```

**Parameters:**
| Parameter | Value |
|-----------|-------|
| Hidden layers | [256, 128] |
| Dropout | 0.2 |
| Loss | MSE |
| Output range | [0, 100] (clipped) |
| Optimizer | Adam (lr=1e-3, weight_decay=1e-5) |

**Output:** 1 value per window: health index (100=perfect, 0=failure)

---

## Bundle Structure (`apet_bundle.pkl`)

The trained bundle contains:

| Key | Type | Description |
|-----|------|-------------|
| `physics` | PhysicsFeatureModel | Ridge models, phase stats, context scaler |
| `if` | tuple | (IsolationForest model, reference decision scores) |
| `anomaly_extra` | dict | PCA model, reference percentiles, blend weights |
| `lstm` | BiLSTMEncoder | Trained bidirectional LSTM |
| `fusion` | FaultFusionMLP | Trained fault classification MLP |
| `quantile` | QuantileMLP | Trained RUL quantile regression MLP |
| `health` | HealthMLP | Trained health index regression MLP |
| `seq_features` | list[str] | 35 feature names for sequence input |
| `seq_scaler` | StandardScaler | Scaler for sequence features |
| `stat_scaler` | StandardScaler | Scaler for stat features |
| `stats_names` | list[str] | 240 stat feature names |
| `faults` | list[str] | 8 fault type names |

---

## Feature Dimensions Summary

| Feature Set | Dimension | Used By |
|-------------|-----------|---------|
| Raw input CSV | 81 columns | Pipeline input |
| Sequence features | 35 × 20 = 700 | BiLSTM |
| Stat features (per signal) | 29 signals × 8 stats = 232 | All ML models |
| Cumulative features | 8 | All ML models |
| Total stat vector | 240 | Fusion, RUL, Health |
| LSTM embedding | 256 | Fusion |
| Fusion input | 240 + 1 + 256 = 497 | Fusion MLP |
| Fault output | 8 probabilities | Fusion MLP |
| RUL output | 3 quantiles | QuantileMLP |
| Health output | 1 value (0-100) | HealthMLP |

---

## Per-Class Detection Thresholds

Thresholds are tuned on the validation set and stored in `threshold_tuned.json`:

| Fault Type | Threshold |
|-----------|-----------|
| misfire | 0.860 |
| injector_degradation | 0.810 |
| turbo_issue | 0.880 |
| lubrication_issue | 0.910 |
| sensor_drift | 0.880 |
| overheating | 0.920 |
| electrical_fault | 0.900 |
| bearing_fault | 0.810 |

---

## Output (`op.csv`)

Each row in `op.csv` corresponds to one 10-second window and contains:

| Column | Description |
|--------|-------------|
| `engine_id` | Engine identifier |
| `mission_id` | Mission identifier |
| `window_start_time` | Window start (seconds) |
| `window_end_time` | Window end (seconds) |
| `anomaly_score` | 0-1, higher = more anomalous |
| `anomaly_alert` | 1 if anomaly_score > 0.5 |
| `fault_probability` | Overall fault probability (max of 8) |
| `predicted_faults` | Semicolon-separated list of detected faults |
| `normal_probability` | Probability of no fault |
| `fault_probability_misfire` | Misfire probability |
| `fault_probability_injector_degradation` | Injector degradation probability |
| `fault_probability_turbo_issue` | Turbo issue probability |
| `fault_probability_lubrication_issue` | Lubrication issue probability |
| `fault_probability_sensor_drift` | Sensor drift probability |
| `fault_probability_overheating` | Overheating probability |
| `fault_probability_electrical_fault` | Electrical fault probability |
| `fault_probability_bearing_fault` | Bearing fault probability |
| `rul_lower_h` | RUL lower bound (hours) |
| `rul_median_h` | RUL median estimate (hours) |
| `rul_upper_h` | RUL upper bound (hours) |
| `health_index` | 0-100, higher = healthier |

---

## Training Data

All training data is synthetic, generated by `generator_v30_physics_locked.py`:

| Dataset | Missions | Purpose |
|---------|----------|---------|
| Normal (healthy) | 120 | Train physics model, Isolation Forest, PCA, LSTM |
| Faulty (8 types) | 600 | Train Fusion MLP, classical baseline |
| RUL life cycles | 20 | Train Quantile RUL, Health Index |
| Early-severity augmentation | 80 | Additional fault missions at severity 0.18-0.30 |
| **Total** | **860** | |

Data split: 70% train / 15% validation / 15% test (stratified by mission ID).
