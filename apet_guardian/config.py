# Config constants
import os
from pathlib import Path

DATA_ROOT = Path(os.getenv("APET_DATA_ROOT", r"D:\testing\DATA"))
OUT_ROOT = Path(os.getenv("APET_OUT_ROOT", r"D:\testing\APET_GUARDIAN_OUT"))
CACHE_DIR = OUT_ROOT / "_cache"

SAMPLE_DT = 0.5
SEQ_LEN = 20
STRIDE = 10
MIN_ROWS = SEQ_LEN
TIME_TOL = 0.05
MAX_GAP = 1.5

SPLIT_FRAC = (0.70, 0.15, 0.15)
SPLIT_SEED = 2026
FAULT_RNG_SEED = 7

FAULTS = [
    "misfire", "injector_degradation", "turbo_issue", "lubrication_issue",
    "sensor_drift", "overheating", "electrical_fault", "bearing_fault",
]
FAULT_COLUMNS = [
    "fault_misfire_active", "fault_injector_active", "fault_turbo_active",
    "fault_lubrication_active", "fault_sensor_active", "fault_overheating_active",
    "fault_electrical_active", "fault_bearing_active",
]
FAULT_ID = {f: i for i, f in enumerate(FAULTS)}

PHASE_ORDER = [
    "takeoff", "climb", "cruise", "loiter", "dash", "bench",
    "descent", "approach", "go_around", "landing",
]
PHASE_ID = {p: i for i, p in enumerate(PHASE_ORDER)}

CONTEXT_COLS = ["throttle_actual", "true_rpm", "air_density_kg_m3", "altitude_m"]

RESIDUAL_TARGETS = {
    "boost_pressure_kPa": "expected_boost",
    "cht_cyl_avg_C": "expected_cht",
    "egt_cyl_avg_C": "expected_egt",
    "oil_pressure_kPa": "expected_oilp",
}

SEQ_FEATURES = [
    "rpm", "throttle_actual", "boost_pressure_kPa", "manifold_pressure_kPa",
    "cht_cyl_avg_C", "egt_cyl_avg_C", "oil_pressure_kPa", "oil_temp_C",
    "coolant_temp_C", "fuel_flow_g_s", "fuel_temp_C", "rail_pressure_bar",
    "vib_rms_g", "vib_1x_rms", "vib_2x_rms", "vib_bpfo_hz", "vib_bpfi_hz", "vib_bsf_hz", "vib_psd_peak",
    "battery_voltage_V", "airspeed_mps", "altitude_m",
    "vertical_speed_mps", "engine_power_command_kW",
    "expected_boost", "boost_residual_kPa", "boost_residual_norm",
    "cht_residual_norm", "egt_residual_norm", "oilp_residual_norm",
    "thermal_stress", "cht_stress", "fuel_per_rpm_g_rev",
    "oil_efficiency_KPa_per_C", "vibration_trend",
]

STATS_SIGNALS = [
    "rpm", "cht_cyl_avg_C", "egt_cyl_avg_C", "oil_pressure_kPa", "oil_temp_C",
    "coolant_temp_C", "boost_pressure_kPa", "manifold_pressure_kPa",
    "fuel_flow_g_s", "throttle_actual", "vib_rms_g", "vib_1x_rms", "vib_2x_rms", "vib_bpfo_hz", "vib_bpfi_hz", "vib_bsf_hz", "vib_psd_peak", "battery_voltage_V",
    "airspeed_mps", "altitude_m", "vertical_speed_mps",
    "boost_residual_kPa", "boost_residual_norm", "thermal_stress",
    "fuel_per_rpm_g_rev", "oil_efficiency_KPa_per_C",
    "cht_residual_norm", "egt_residual_norm", "oilp_residual_norm",
]

STAT_NAMES = ["mean", "std", "min", "max", "p90", "range", "slope", "final"]

CUMULATIVE_FEATURES = [
    "cum_thermal_stress", "cum_load_stress", "cum_egt_excess",
    "cum_cht_excess", "cum_vib", "boost_resid_integrated",
    "cum_oil_loss", "cum_fuel",
]

PHYSICS_COLUMNS = [
    "expected_boost", "boost_residual_kPa", "boost_residual_norm",
    "expected_cht", "cht_residual_C", "cht_residual_norm",
    "expected_egt", "egt_residual_C", "egt_residual_norm",
    "expected_oilp", "oilp_residual_kPa", "oilp_residual_norm",
    "thermal_stress", "cht_stress", "fuel_per_rpm_g_rev",
    "oil_efficiency_KPa_per_C", "vibration_trend",
] + CUMULATIVE_FEATURES

META_NUMERIC_COLUMNS = [
    "win_start_s", "win_end_s", "phase_id", "rul_h", "health",
    "any_fault", "n_faults",
] + ["f_" + f for f in FAULTS]

HEALTHY_PHASE_MIN_ROWS = 200

BEM_RATED_RPM = 2800.0
BEM_RATED_KW = 125.0
RHO_SEA = 1.225

IF_N_ESTIMATORS = 300
IF_CONTAMINATION = 0.01
IF_MAX_SAMPLES = 200000

FUSION_HIDDEN = [256, 128]
FUSION_DROPOUT = 0.25
REG_HIDDEN = [256, 128]
REG_DROPOUT = 0.2
LSTM_HIDDEN = 128
LSTM_LAYERS = 2
LSTM_DROPOUT = 0.30
QUANTILES = (0.10, 0.50, 0.90)

BATCH_SIZE = 512
LSTM_BATCH = 512
EPOCHS = 40
PATIENCE = 6
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-5

CLASSICAL_ESTIMATORS = 200
RUL_MAX_TARGET_H = 50.0

LIMITS = {
    "normal": int(os.getenv("APET_NORMAL", "120")),
    "faulty": int(os.getenv("APET_FAULTY", "600")),
    "rul": int(os.getenv("APET_RUL", "20")),
}

# Quick Limits
def quick_limits():
    LIMITS.update({"normal": 24, "faulty": 40, "rul": 4})
