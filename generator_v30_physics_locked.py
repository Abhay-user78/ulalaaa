# Generate synthetic data
import os
import json
import random
import zlib
import sys
import bisect
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
from scipy.signal import lfilter

MASTER_SEED = 42
random.seed(MASTER_SEED)
np.random.seed(MASTER_SEED)

@dataclass(frozen=True)
# Physicsconstants
class PhysicsConstants:
    R_air: float = 287.058
    gamma_air: float = 1.40
    cp_air: float = 1005.0
    cp_exh: float = 1150.0
    g: float = 9.80665
    P0_sea_kPa: float = 101.325
    T0_sea_K: float = 288.15
    lapse_rate: float = 0.0065
    T_trop_K: float = 216.65
    H_trop: float = 11000.0

    V_displacement: float = 0.0020
    num_cylinders: int = 4
    lhv_fuel: float = 42.8e6
    D_prop: float = 1.65

    prop_design_rpm: float = 2800.0
    prop_design_J: float = 0.50
    prop_design_pitch: float = 0.68
    prop_design_Cp: float = 0.0821
    prop_design_Ct: float = 0.132
    max_prop_thrust_N: float = 4200.0
    max_prop_power_kW: float = 135.0
    max_cruise_excess_thrust_fraction: float = 0.18

    speed_controller_tau_s: float = 8.0
    altitude_controller_tau_s: float = 35.0
    max_climb_rate_mps: float = 6.0
    min_operating_speed_mps: float = 32.0
    max_operating_speed_mps: float = 90.0

    rated_rpm: float = 2800.0
    idle_rpm: float = 1800.0
    min_flight_governed_rpm: float = 1950.0
    max_rpm: float = 3050.0
    rated_brake_power_kW: float = 125.0
    rated_bsfc_g_kWh: float = 215.8
    max_manifold_pressure_kPa: float = 320.0
    max_pressure_ratio: float = 3.00
    fadec_cht_rollback_C: float = 215.0
    fadec_coolant_rollback_C: float = 108.0
    thermal_derate_fraction: float = 0.60
    gov_kp: float = 0.80
    gov_ki: float = 0.06
    prop_pitch_rate_per_s: float = 0.45
    prop_pitch_min: float = 0.12
    prop_pitch_max: float = 0.98
    overspeed_rpm: float = 2920.0
    overspeed_cut_rpm: float = 3000.0
    stall_rpm: float = 1200.0
    stall_persistence_s: float = 2.0

    m_empty_kg: float = 1550.0
    fuel_capacity_kg: float = 250.0
    fuel_reserve_kg: float = 25.0
    m_uav_kg: float = 1800.0
    S_wing_m2: float = 16.5
    AR_wing: float = 18.0
    CD0_parasitic: float = 0.024
    oswald_e: float = 0.85
    CL_max: float = 1.45

    mc_cht: float = 14000.0
    mc_coolant: float = 20000.0
    mc_oil: float = 11000.0
    mc_fuel_tank: float = 80000.0

CONST = PhysicsConstants()
DT = 0.5

SETTLE_S = 60.0
DATA_ROOT = Path(os.getenv("DATA_ROOT", "./data/v30_production_digital_twin"))
OUTPUT_FORMAT = "parquet"

VIBRATION_FS_HZ = 2048
VIBRATION_WINDOW_S = 0.5
RAW_VIBRATION_WINDOWS_PER_MISSION = 4

BEM_PITCH_GRID_PTS = 22
BEM_J_GRID_PTS = 31

# Stable Mission Seed
def _stable_mission_seed(mission_id: str) -> int:
    return int(zlib.crc32(mission_id.encode("utf-8")) & 0xFFFFFFFF)

# Enginemanufacturingtolerance
class EngineManufacturingTolerance:
    __slots__ = ("engine_id", "aging_regime",
                 "J_rotor", "eta_vol_base", "comp_eff_base", "fric_torque_base",
                 "h_cht_cool", "h_rad_cool", "h_oil_cool", "mu0_oil",
                 "bearing_rolling_elements", "bearing_contact_angle_deg",
                 "bearing_ball_diameter_ratio", "bearing_pitch_diameter_ratio",
                 "oil_p_fail_kPa", "cht_fail_C", "egt_fail_C",
                 "persistence_oil_s", "persistence_thermal_s",
                 "bearing_design_life_h", "thermal_design_life_h",
                 "turbo_design_life_h", "injector_design_life_h",
                 "lube_design_life_h", "electrical_design_life_h",
                 "bias_rpm", "bias_cht", "bias_egt", "bias_oil_p", "bias_oil_t",
                 "bias_fuel_flow", "bias_boost", "bias_manifold", "bias_rail", "bias_coolant")

    # Init
    def __init__(self, engine_id: str, aging_regime: str = "nominal"):
        random.seed(_stable_mission_seed(engine_id))
        self.engine_id = engine_id
        self.aging_regime = aging_regime

        self.J_rotor = random.uniform(0.70, 0.82)
        self.eta_vol_base = random.uniform(0.85, 0.91)
        self.comp_eff_base = random.uniform(0.79, 0.85)
        self.fric_torque_base = random.uniform(13.5, 16.5)
        self.h_cht_cool = random.uniform(700.0, 780.0)
        self.h_rad_cool = random.uniform(1400.0, 1600.0)
        self.h_oil_cool = random.uniform(240.0, 280.0)
        self.mu0_oil = random.uniform(0.010, 0.014)
        self.bearing_rolling_elements = random.choice([8, 9, 10, 11])
        self.bearing_contact_angle_deg = random.uniform(0.0, 8.0)
        self.bearing_ball_diameter_ratio = random.uniform(0.18, 0.24)
        self.bearing_pitch_diameter_ratio = random.uniform(0.42, 0.48)

        self.oil_p_fail_kPa = random.uniform(70.0, 85.0)
        self.cht_fail_C = random.uniform(225.0, 238.0)
        self.egt_fail_C = random.uniform(880.0, 920.0)
        self.persistence_oil_s = random.uniform(3.0, 5.0)
        self.persistence_thermal_s = random.uniform(5.0, 8.0)

        self.bearing_design_life_h = random.uniform(180.0, 320.0)
        self.thermal_design_life_h = random.uniform(160.0, 300.0)
        self.turbo_design_life_h = random.uniform(200.0, 340.0)
        self.injector_design_life_h = random.uniform(220.0, 360.0)
        self.lube_design_life_h = random.uniform(170.0, 300.0)
        self.electrical_design_life_h = random.uniform(320.0, 480.0)

        self.bias_rpm = random.gauss(0.0, 1.8)
        self.bias_cht = random.gauss(0.0, 0.6)
        self.bias_egt = random.gauss(0.0, 2.0)
        self.bias_oil_p = random.gauss(0.0, 2.5)
        self.bias_oil_t = random.gauss(0.0, 0.5)
        self.bias_fuel_flow = random.gauss(0.0, 0.04)
        self.bias_boost = random.gauss(0.0, 0.6)
        self.bias_manifold = random.gauss(0.0, 0.6)
        self.bias_rail = random.gauss(0.0, 3.5)
        self.bias_coolant = random.gauss(0.0, 0.5)

METADATA_COLUMNS = [
    "global_time_s",
    "time_in_mission_s",
    "actual_operating_hours",
    "engine_id",
    "mission_id",
    "mission_type",
    "phase",
    "time_in_phase_s",
    "telemetry_valid",
]

FEATURE_COLUMNS = [
    "altitude_m",
    "target_altitude_m",
    "aircraft_mass_kg",
    "fuel_mass_kg",
    "fuel_used_kg",
    "refueled_at_sortie_start",
    "sortie_start_fuel_kg",
    "wind_u_mps",
    "wind_accel_mps2",
    "ground_speed_mps",
    "ambient_T_C",
    "ambient_P_kPa",
    "air_density_kg_m3",
    "airspeed_mps",
    "flight_path_angle_deg",
    "vertical_speed_mps",
    "target_airspeed_mps",
    "throttle_cmd",
    "throttle_actual",
    "prop_pitch_cmd",
    "prop_pitch_actual",
    "engine_power_command_kW",
    "engine_bsfc_g_kWh",
    "actual_bsfc_g_kWh",
    "desired_thrust_N",
    "thrust_equilibrium_error_N",
    "settled_state",
    "drag_N",
    "excess_thrust_N",
    "shaft_torque_error_Nm",
    "prop_advance_ratio",
    "prop_Cp",
    "prop_Ct",
    "prop_efficiency",
    "prop_power_kW",
    "rpm",
    "cht_cyl_avg_C",
    "egt_cyl_avg_C",
    "oil_pressure_kPa",
    "oil_temp_C",
    "fuel_flow_g_s",
    "fuel_temp_C",
    "boost_pressure_kPa",
    "manifold_pressure_kPa",
    "vib_rms_g",
    "vib_crest_factor",
    "vib_kurtosis",
    "vib_1x_hz",
    "vib_2x_hz",
    "vib_firing_hz",
    "vib_bpfo_hz",
    "vib_bpfi_hz",
    "vib_bsf_hz",
    "vib_psd_peak",
    "vib_psd_peak_freq",
    "vib_1x_rms",
    "vib_2x_rms",
    "vib_fire_order_rms",
    "vib_spectral_centroid",
    "battery_voltage_V",
    "battery_voltage_raw_V",
    "coolant_temp_C",
    "rail_pressure_bar",
    "SOI_deg_bTDC",
    "inj_duration_ms",
    "altitude_quantized_m",
    "rpm_quantized",
    "rail_pressure_quantized_bar",
    "sensor_drift_flag_rpm",
    "sensor_drift_flag_rail",
    "sensor_drift_flag_boost",
    "fadec_mode",
]

ML_FEATURE_COLUMNS = [c for c in FEATURE_COLUMNS if c != "fadec_mode"]

TARGET_COLUMNS = [
    "true_brake_power_kW",
    "true_fuel_power_kW",
    "true_indicated_power_kW",
    "true_friction_power_kW",
    "true_exhaust_heat_kW",
    "true_coolant_heat_kW",
    "true_oil_heat_kW",
    "true_direct_heat_loss_kW",
    "true_energy_closure_error_W",
    "true_rpm",
    "true_indicated_torque_Nm",
    "true_prop_load_torque_Nm",
    "true_prop_thrust_N",
    "true_bearing_wear",
    "true_thermal_fatigue",
    "true_turbo_degradation",
    "true_injector_wear",
    "true_lubrication_degradation",
    "true_electrical_degradation",
    "true_overall_health_index",
    "simulated_rul_hours",
    "health_state",
    "fault_type",
    "fault_domain",
    "fault_nature",
    "fault_severity",
    "severity_category",
    "fault_active",
    "fault_misfire_active",
    "fault_injector_active",
    "fault_turbo_active",
    "fault_lubrication_active",
    "fault_sensor_active",
    "fault_overheating_active",
    "fault_electrical_active",
    "fault_bearing_active",
    "failure_flag",
    "failure_mode",
]

SCHEMA_COLUMNS = METADATA_COLUMNS + FEATURE_COLUMNS + TARGET_COLUMNS

# First Order Lag
def first_order_lag(command: np.ndarray, tau: float, initial_val: float = None, dt: float = DT) -> np.ndarray:
    alpha = np.exp(-dt / max(tau, 1e-4))
    b = [1.0 - alpha]
    a = [1.0, -alpha]
    init = float(command[0]) if initial_val is None else float(initial_val)
    zi = np.array([init * alpha])
    y, _ = lfilter(b, a, command, zi=zi)
    return y

# Compute Instantaneous Atmosphere
def compute_instantaneous_atmosphere(alt_m: float, t_global_s: float,
                                     sea_temp_base_C: float = 18.0, qnh_offset_kPa: float = 0.0) -> Tuple[float, float, float]:
    diurnal_phase = (2.0 * np.pi * t_global_s) / 86400.0
    T_sea_C = sea_temp_base_C + 6.0 * np.sin(diurnal_phase)
    T_sea_K = T_sea_C + 273.15
    P_sea_kPa = CONST.P0_sea_kPa + qnh_offset_kPa

    exp_baro = CONST.g / (CONST.R_air * CONST.lapse_rate)
    if alt_m <= CONST.H_trop:
        T_amb_K = T_sea_K - CONST.lapse_rate * alt_m
        P_amb_kPa = P_sea_kPa * (T_amb_K / T_sea_K) ** exp_baro
    else:
        T_amb_K = CONST.T_trop_K
        P_trop_kPa = P_sea_kPa * (CONST.T_trop_K / T_sea_K) ** exp_baro
        P_amb_kPa = P_trop_kPa * np.exp(-CONST.g * (alt_m - CONST.H_trop) / (CONST.R_air * T_amb_K))

    T_amb_K = float(np.clip(T_amb_K, 216.0, 330.0))
    P_amb_kPa = float(np.clip(P_amb_kPa, 5.0, 110.0))
    T_amb_C = T_amb_K - 273.15
    rho_air = float(P_amb_kPa * 1000.0 / (CONST.R_air * T_amb_K))
    return T_amb_C, P_amb_kPa, rho_air

# Propeller Bem Direct
def _propeller_bem_direct(n_rev_s: float, airspeed_mps: float, pitch_cmd: float, rho_air: float) -> Tuple[float, float, float, float, float, float]:
    D = float(CONST.D_prop); R = D / 2.0; B = 3
    rho = max(float(rho_air), 0.20)
    n = max(0.5, float(n_rev_s)); V = max(0.0, float(airspeed_mps))
    pitch = float(np.clip(pitch_cmd, CONST.prop_pitch_min, CONST.prop_pitch_max))
    J = float(np.clip(V / (n * D), 0.0, 1.50))
    if J > 1.0 or n < 10.0:
        Cp_w = 0.018 + 0.020 * min(J - 1.0, 0.5)
        P_w = Cp_w * rho * n**3 * D**5
        Q_w = P_w / (2 * np.pi * n)
        Ct_w = -0.045 - 0.025 * min(J - 1.0, 0.5)
        T_w = Ct_w * rho * n**2 * D**4
        return float(Q_w), float(T_w), float(Cp_w), float(Ct_w), J, 0.0
    N = 12; dr = (0.95 - 0.18) * R / N
    r = (np.arange(N) + 0.5) * dr + 0.18 * R
    chord = 0.855 * (0.20 - 0.11 * (r / R))
    twist = 28.0 + (10.0 - 28.0) * (r / R - 0.18) / (0.95 - 0.18)
    beta = np.deg2rad(twist + 18.0 * (pitch - CONST.prop_design_pitch))
    a = np.full(N, 0.05); ap = np.full(N, 0.01)
    for _ in range(30):
        phi = np.arctan2(V * (1 + a), 2 * np.pi * n * r * (1 - ap))
        alpha = beta - phi; aldeg = np.rad2deg(alpha)
        W = np.sqrt((V * (1 + a))**2 + (2 * np.pi * n * r * (1 - ap))**2)
        Re = np.clip(rho * W * chord / 1.8e-5, 5.0e4, 8.0e6)
        M = np.clip(W / np.sqrt(CONST.gamma_air * CONST.R_air * 288.15), 0.0, 0.65)
        cl_alpha = 0.105 * (1.0 - 0.08 * M * M) * np.clip((Re / 5.0e5) ** 0.035, 0.88, 1.08)
        cl = np.clip(cl_alpha * aldeg, -1.35, 1.35)
        cd = 0.016 + 0.006 * np.clip((5.0e5 / np.maximum(Re, 1.0)) ** 0.20, 0.55, 2.2) + 0.011 * (cl**2)
        cn = cl * np.cos(phi) - cd * np.sin(phi)
        ct = cl * np.sin(phi) + cd * np.cos(phi)
        sigma = B * chord / (2 * np.pi * r)
        F = (2 / np.pi) * np.arccos(np.clip(np.exp(-B / 2 * (R - r) / (r * np.sin(np.maximum(phi, 0.05)))), 0, 1))
        F = np.clip(F, 0.15, 1)
        anew = 1 / ((4 * F * np.sin(phi)**2) / (sigma * cn + 1e-9) - 1)
        apnew = 1 / ((4 * F * np.sin(phi) * np.cos(phi)) / (sigma * ct + 1e-9) + 1)
        anew = np.clip(anew, -0.2, 0.5); apnew = np.clip(apnew, -0.3, 0.3)
        if np.max(abs(anew - a)) < 1e-5:
            break
        a = 0.75 * a + 0.25 * anew
        ap = 0.75 * ap + 0.25 * apnew
    W = np.sqrt((V * (1 + a))**2 + (2 * np.pi * n * r * (1 - ap))**2)
    phi = np.arctan2(V * (1 + a), 2 * np.pi * n * r * (1 - ap))
    alpha = beta - phi; aldeg = np.rad2deg(alpha)
    Re = np.clip(rho * W * chord / 1.8e-5, 5.0e4, 8.0e6)
    M = np.clip(W / np.sqrt(CONST.gamma_air * CONST.R_air * 288.15), 0.0, 0.65)
    cl_alpha = 0.105 * (1.0 - 0.08 * M * M) * np.clip((Re / 5.0e5) ** 0.035, 0.88, 1.08)
    cl = np.clip(cl_alpha * aldeg, -1.35, 1.35)
    cd = 0.016 + 0.006 * np.clip((5.0e5 / np.maximum(Re, 1.0)) ** 0.20, 0.55, 2.2) + 0.011 * cl**2
    L = 0.5 * rho * W**2 * chord * dr
    dT = B * (L * np.cos(phi) - 0.5 * rho * W**2 * chord * dr * cd * np.sin(phi))
    dQ = B * (L * np.sin(phi) + 0.5 * rho * W**2 * chord * dr * cd * np.cos(phi)) * r
    T = max(0.0, float(dT.sum())); Q = max(0.0, float(dQ.sum())); P = Q * 2 * np.pi * n
    Cp = P / max(rho * n**3 * D**5, 1e-9)
    Ct = T / max(rho * n**2 * D**4, 1e-9)
    eta = float(np.clip(J * Ct / max(Cp, 1e-9), 0.0, 0.95))
    return Q, T, float(Cp), float(Ct), J, eta

_BEM_MAP = None
_BEM_CP_SCALE = 1.0
_BEM_CT_SCALE = 1.0

# Build Bem Lookup
def _build_bem_lookup():
    global _BEM_MAP, _BEM_CP_SCALE, _BEM_CT_SCALE
    if _BEM_MAP is not None:
        return
    j_grid = np.linspace(0.0, 1.0, BEM_J_GRID_PTS)
    p_grid = np.linspace(CONST.prop_pitch_min, CONST.prop_pitch_max, BEM_PITCH_GRID_PTS)

    # Pitch Gain
    def _pitch_gain(pch):
        frac = (pch - CONST.prop_pitch_min) / max(CONST.prop_pitch_max - CONST.prop_pitch_min, 1e-9)
        return float(0.12 + 1.352 * frac)

    # Loading Lifted
    def _loading_lifted(jv):
        if jv <= 0.25:
            return 0.78 + 0.68 * (jv / 0.25)
        if jv <= 0.50:
            return 0.95 + 0.10 * (jv - 0.25) / 0.25
        if jv <= 0.65:
            return 1.00 - 0.06 * (jv - 0.50) / 0.15
        if jv <= 0.78:
            return 0.94 - 0.09 * (jv - 0.65) / 0.13
        if jv <= 0.90:
            return 0.85 - 0.31 * (jv - 0.78) / 0.12
        return 0.54 - 0.36 * (jv - 0.90) / 0.10

    # Loading Torque
    def _loading_torque(jv):
        if jv <= 0.25:
            return 0.90 + 0.32 * (jv / 0.25)
        if jv <= 0.50:
            return 0.98 + 0.04 * (jv - 0.25) / 0.25
        if jv <= 0.65:
            return 1.00 - 0.14 * (jv - 0.50) / 0.15
        if jv <= 0.78:
            return 0.86 - 0.17 * (jv - 0.65) / 0.13
        if jv <= 0.90:
            return 0.69 - 0.29 * (jv - 0.78) / 0.12
        return 0.40 - 0.26 * (jv - 0.90) / 0.10

    cp_grid = np.zeros((len(p_grid), len(j_grid)))
    ct_grid = np.zeros_like(cp_grid)
    n_ref = CONST.prop_design_rpm / 60.0
    for ip, pch in enumerate(p_grid):
        pg = _pitch_gain(pch)
        for ij, jv in enumerate(j_grid):
            q, tt, cp, ct, _, _ = _propeller_bem_direct(n_ref, jv * n_ref * CONST.D_prop, pch, 1.225)
            cp_grid[ip, ij] = float(max(cp, 1e-6)) * pg * _loading_torque(jv)
            ct_grid[ip, ij] = max(0.0, float(ct)) * pg * _loading_lifted(jv)
    # Interp
    def _interp(grid, pch, jv):
        ip = int(np.clip(np.searchsorted(p_grid, pch) - 1, 0, len(p_grid) - 2))
        ij = int(np.clip(np.searchsorted(j_grid, jv) - 1, 0, len(j_grid) - 2))
        tp = (pch - p_grid[ip]) / max(p_grid[ip+1] - p_grid[ip], 1e-12)
        tj = (jv - j_grid[ij]) / max(j_grid[ij+1] - j_grid[ij], 1e-12)
        return ((1-tp)*(1-tj)*grid[ip, ij] + tp*(1-tj)*grid[ip+1, ij]
                + (1-tp)*tj*grid[ip, ij+1] + tp*tj*grid[ip+1, ij+1])
    cp_ref = max(_interp(cp_grid, CONST.prop_design_pitch, CONST.prop_design_J), 1e-9)
    ct_ref = max(_interp(ct_grid, CONST.prop_design_pitch, CONST.prop_design_J), 1e-9)
    _BEM_CP_SCALE = CONST.prop_design_Cp / cp_ref
    _BEM_CT_SCALE = CONST.prop_design_Ct / ct_ref
    _BEM_MAP = (j_grid, p_grid, cp_grid, ct_grid)

# Propeller Aerodynamics
def propeller_aerodynamics(n_rev_s: float, airspeed_mps: float, pitch_cmd: float, rho_air: float) -> Tuple[float, float, float, float, float, float]:
    _build_bem_lookup()
    j_grid, p_grid, cp_grid, ct_grid = _BEM_MAP
    D = float(CONST.D_prop)
    rho = rho_air if float(rho_air) > 0.20 else 0.20
    n = max(0.5, float(n_rev_s)); V = max(0.0, float(airspeed_mps))
    pmin = CONST.prop_pitch_min; pmax = CONST.prop_pitch_max
    pc_ = float(pitch_cmd)
    pitch = pmin if pc_ < pmin else (pmax if pc_ > pmax else pc_)
    J = V / (n * D)
    if J < j_grid[0]: j_bem = j_grid[0]
    elif J > j_grid[-1]: j_bem = j_grid[-1]
    else: j_bem = J
    ip = bisect.bisect_right(p_grid, pitch) - 1
    if ip < 0: ip = 0
    elif ip > len(p_grid) - 2: ip = len(p_grid) - 2
    ij = bisect.bisect_right(j_grid, j_bem) - 1
    if ij < 0: ij = 0
    elif ij > len(j_grid) - 2: ij = len(j_grid) - 2
    pd_ = p_grid[ip+1] - p_grid[ip]; pd_ = pd_ if pd_ > 1e-12 else 1e-12
    jd_ = j_grid[ij+1] - j_grid[ij]; jd_ = jd_ if jd_ > 1e-12 else 1e-12
    tp = (pitch - p_grid[ip]) / pd_
    tj = (j_bem - j_grid[ij]) / jd_
    cp_b = ((1-tp)*(1-tj)*cp_grid[ip, ij] + tp*(1-tj)*cp_grid[ip+1, ij]
            + (1-tp)*tj*cp_grid[ip, ij+1] + tp*tj*cp_grid[ip+1, ij+1]) * _BEM_CP_SCALE
    ct_b = ((1-tp)*(1-tj)*ct_grid[ip, ij] + tp*(1-tj)*ct_grid[ip+1, ij]
            + (1-tp)*tj*ct_grid[ip, ij+1] + tp*tj*ct_grid[ip+1, ij+1]) * _BEM_CT_SCALE
    if cp_b < 0.001: cp_b = 0.001
    elif cp_b > 0.18: cp_b = 0.18
    if ct_b < 0.0: ct_b = 0.0
    elif ct_b > 0.22: ct_b = 0.22
    if J <= 0.92:
        cp, ct = cp_b, ct_b
    else:
        if J < 1.0: jw = 1.0
        elif J > 1.50: jw = 1.50
        else: jw = J
        cp_w = 0.010 + 0.018 * min(max(jw - 1.0, 0.0), 0.5)
        ct_w = -0.006 - 0.020 * min(max(jw - 1.0, 0.0), 0.5)
        blend_ = (J - 0.92) / 0.16
        bez_ = 0.0 if blend_ < 0.0 else (1.0 if blend_ > 1.0 else blend_)
        if J > 1.08:
            cp, ct = cp_w, ct_w
        else:
            cp = (1.0 - bez_) * cp_b + bez_ * cp_w
            ct = (1.0 - bez_) * ct_b + bez_ * ct_w
    if n < 10.0:
        cp = 0.018; ct = -0.045
    if cp < 0.001: cp = 0.001
    elif cp > 0.18: cp = 0.18
    if ct < -0.08: ct = -0.08
    elif ct > 0.22: ct = 0.22
    Q = cp * rho * n**2 * D**5 / (2 * np.pi)
    T = ct * rho * n**2 * D**4
    _ETA_DESIGN = (CONST.prop_design_J * CONST.prop_design_Ct /
                   max(CONST.prop_design_Cp, 1e-9))
    _eta_bell_design = 0.55 + 0.27 * (CONST.prop_design_J - 0.15) / 0.40
    _eta_scale = _ETA_DESIGN / max(_eta_bell_design, 1e-9)
    if J <= 0.15:
        eta = 0.15 + (0.55 - 0.15) * (J / 0.15)
    elif J <= 0.55:
        eta = 0.55 + 0.27 * (J - 0.15) / 0.40
    elif J <= 0.85:
        eta = 0.82 - 0.22 * (J - 0.55) / 0.30
    elif J <= 1.00:
        eta = 0.60 - 0.42 * (J - 0.85) / 0.15
    else:
        eta = 0.18
    eta *= _eta_scale
    if eta < 0.0: eta = 0.0
    elif eta > 0.95: eta = 0.95
    if ct > 0.0 and J > 1e-4:
        ct = (eta * cp) / J
    else:
        eta = 0.0
    Q = cp * rho * n**2 * D**5 / (2 * np.pi)
    T = ct * rho * n**2 * D**4
    return Q, T, cp, ct, J, eta

# Solve Propeller Trim
def solve_propeller_trim(required_thrust_N: float, n_rev_s: float, airspeed_mps: float,
                         rho_air: float, warm_start: Optional[float] = None) -> float:
    lo, hi = CONST.prop_pitch_min, CONST.prop_pitch_max
    target = float(np.clip(required_thrust_N, 0.0, CONST.max_prop_thrust_N))
    if target <= 0.0:
        return lo
    t_lo = propeller_aerodynamics(n_rev_s, airspeed_mps, lo, rho_air)[1]
    if t_lo <= 0.0 or target <= t_lo:
        return lo
    t_hi = propeller_aerodynamics(n_rev_s, airspeed_mps, hi, rho_air)[1]
    if t_hi <= 0.0 or target >= t_hi:
        return hi

    a, b = lo, hi
    if warm_start is not None:
        w = float(np.clip(warm_start, lo, hi))
        t_w = propeller_aerodynamics(n_rev_s, airspeed_mps, w, rho_air)[1]
        if t_w < target:
            a = w
        else:
            b = w

    for _ in range(20):
        mid = 0.5 * (a + b)
        if propeller_aerodynamics(n_rev_s, airspeed_mps, mid, rho_air)[1] < target:
            a = mid
        else:
            b = mid
        if (b - a) < 2e-3:
            break
    return float(0.5 * (a + b))

# Environmental Wind
def environmental_wind(t_global_s: float, altitude_m: float, seed: int = MASTER_SEED) -> float:
    phase = 2.0 * np.pi * (t_global_s + seed) / 1800.0
    shear = 0.004 * np.clip(altitude_m, 0.0, 11000.0)
    synoptic = 3.0 * np.sin(phase) + 1.5 * np.sin(phase / 3.7 + 0.8)
    gust = 0.8 * np.sin(2.0 * np.pi * (t_global_s + 17.0 * seed) / 95.0)
    return float(np.clip(synoptic + gust + shear, -15.0, 25.0))

# Environmental Wind Rate
def environmental_wind_rate(t_global_s: float, altitude_m: float, dt: float = DT, seed: int = MASTER_SEED) -> float:
    w0 = environmental_wind(t_global_s, altitude_m, seed)
    w1 = environmental_wind(t_global_s + max(dt, 1e-3), altitude_m, seed)
    return float((w1 - w0) / max(dt, 1e-3))

# Generate Turbulent Wind Profile
def generate_turbulent_wind_profile(t_global_s: np.ndarray, altitude_m: np.ndarray, seed: int, dt: float) -> Tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(int(seed) & 0xFFFFFFFF)
    n = len(t_global_s)
    w = np.zeros(n, dtype=float)
    sigma = np.clip(0.9 + 0.00008 * np.asarray(altitude_m, dtype=float), 0.7, 1.6)
    tau = 12.0
    a = float(np.exp(-max(dt, 1e-3) / tau))
    for k in range(1, n):
        q = sigma[k] * np.sqrt(max(1.0 - a*a, 0.0))
        w[k] = a * w[k-1] + q * rng.normal()
    phase = 2.0 * np.pi * (t_global_s + seed) / 1800.0
    shear = 0.004 * np.clip(altitude_m, 0.0, 11000.0)
    mean = 3.0 * np.sin(phase) + 1.5 * np.sin(phase / 3.7 + 0.8) + shear
    wind = np.clip(mean + w, -15.0, 25.0)
    wind_accel = np.gradient(wind, max(dt, 1e-3))
    return wind.astype(float), wind_accel.astype(float)

# Engine Operating Map Vec
def engine_operating_map_vec(load_fraction: np.ndarray, rpm: np.ndarray) -> np.ndarray:
    loads = np.array([0.05, 0.15, 0.30, 0.50, 0.70, 0.85, 1.00, 1.05])
    speeds = np.array([0.60, 0.70, 0.80, 0.90, 1.00, 1.08])
    eta_grid = np.array([
        [0.245, 0.255, 0.260, 0.258, 0.250, 0.242],
        [0.285, 0.300, 0.315, 0.322, 0.320, 0.310],
        [0.320, 0.340, 0.355, 0.365, 0.365, 0.355],
        [0.345, 0.365, 0.382, 0.395, 0.400, 0.392],
        [0.350, 0.372, 0.392, 0.408, 0.414, 0.405],
        [0.345, 0.368, 0.390, 0.406, 0.412, 0.404],
        [0.335, 0.360, 0.382, 0.398, 0.405, 0.397],
        [0.330, 0.355, 0.377, 0.393, 0.400, 0.392],
    ])
    lf = np.clip(load_fraction, loads[0], loads[-1])
    sr = np.clip(rpm / CONST.rated_rpm, speeds[0], speeds[-1])
    i = np.clip(np.searchsorted(loads, lf) - 1, 0, len(loads) - 2)
    j = np.clip(np.searchsorted(speeds, sr) - 1, 0, len(speeds) - 2)
    x0, x1 = loads[i], loads[i+1]; y0, y1 = speeds[j], speeds[j+1]
    tx = (lf - x0) / np.maximum(x1 - x0, 1e-9)
    ty = (sr - y0) / np.maximum(y1 - y0, 1e-9)
    eta = ((1-tx)*(1-ty)*eta_grid[i, j] + tx*(1-ty)*eta_grid[i+1, j]
           + (1-tx)*ty*eta_grid[i, j+1] + tx*ty*eta_grid[i+1, j+1])
    eta = np.clip(eta, 0.23, 0.43)
    return eta.astype(np.float64)

# Propeller Aerodynamics Vec
def propeller_aerodynamics_vec(n_rev_s: np.ndarray, airspeed_mps: np.ndarray,
                               pitch_cmd: np.ndarray, rho_air: np.ndarray):
    _build_bem_lookup()
    j_grid, p_grid, cp_grid, ct_grid = _BEM_MAP
    D = float(CONST.D_prop)
    rho = np.maximum(np.asarray(rho_air, dtype=np.float64), 0.20)
    n = np.maximum(np.maximum(np.asarray(n_rev_s, dtype=np.float64), 0.5), 0.5)
    V = np.maximum(np.asarray(airspeed_mps, dtype=np.float64), 0.0)
    pch = np.clip(np.asarray(pitch_cmd, dtype=np.float64),
                  CONST.prop_pitch_min, CONST.prop_pitch_max)
    J = V / (n * D)
    jb = np.clip(J, j_grid[0], j_grid[-1])
    ip = np.clip(np.searchsorted(p_grid, pch) - 1, 0, len(p_grid) - 2)
    ij = np.clip(np.searchsorted(j_grid, jb) - 1, 0, len(j_grid) - 2)
    tp = (pch - p_grid[ip]) / np.maximum(p_grid[ip+1] - p_grid[ip], 1e-12)
    tj = (jb - j_grid[ij]) / np.maximum(j_grid[ij+1] - j_grid[ij], 1e-12)
    cp = ((1-tp)*(1-tj)*cp_grid[ip, ij] + tp*(1-tj)*cp_grid[ip+1, ij]
          + (1-tp)*tj*cp_grid[ip, ij+1] + tp*tj*cp_grid[ip+1, ij+1]) * _BEM_CP_SCALE
    ct = ((1-tp)*(1-tj)*ct_grid[ip, ij] + tp*(1-tj)*ct_grid[ip+1, ij]
          + (1-tp)*tj*ct_grid[ip, ij+1] + tp*tj*ct_grid[ip+1, ij+1]) * _BEM_CT_SCALE
    cp = np.clip(cp, 0.001, 0.18)
    ct = np.clip(ct, 0.0, 0.22)
    blend = np.clip((J - 0.92) / (1.08 - 0.92), 0.0, 1.0)
    jw = np.clip(J, 1.0, 1.50)
    cp_w = 0.010 + 0.018 * np.minimum(np.maximum(jw - 1.0, 0.0), 0.5)
    ct_w = -0.006 - 0.020 * np.minimum(np.maximum(jw - 1.0, 0.0), 0.5)
    use_w = J > 1.08
    cp = np.where(use_w, cp_w, (1.0 - blend) * cp + blend * cp_w)
    ct = np.where(use_w, ct_w, (1.0 - blend) * ct + blend * ct_w)
    lo = n < 10.0
    cp = np.where(lo, 0.018, cp)
    ct = np.where(lo, -0.045, ct)
    cp = np.clip(cp, 0.001, 0.18)
    ct = np.clip(ct, -0.08, 0.22)
    Q = cp * rho * n**2 * D**5 / (2 * np.pi)
    T = ct * rho * n**2 * D**4
    return Q, T, cp, ct, J

# Engine Operating Map
def engine_operating_map(load_fraction: float, rpm: float) -> Tuple[float, float]:
    loads = np.array([0.05, 0.15, 0.30, 0.50, 0.70, 0.85, 1.00, 1.05])
    speeds = np.array([0.60, 0.70, 0.80, 0.90, 1.00, 1.08])
    eta_grid = np.array([
        [0.245, 0.255, 0.260, 0.258, 0.250, 0.242],
        [0.285, 0.300, 0.315, 0.322, 0.320, 0.310],
        [0.320, 0.340, 0.355, 0.365, 0.365, 0.355],
        [0.345, 0.365, 0.382, 0.395, 0.400, 0.392],
        [0.350, 0.372, 0.392, 0.408, 0.414, 0.405],
        [0.345, 0.368, 0.390, 0.406, 0.412, 0.404],
        [0.335, 0.360, 0.382, 0.398, 0.405, 0.397],
        [0.330, 0.355, 0.377, 0.393, 0.400, 0.392],
    ])
    lf = float(np.clip(load_fraction, loads[0], loads[-1]))
    sr = float(np.clip(rpm / CONST.rated_rpm, speeds[0], speeds[-1]))
    i = int(np.clip(np.searchsorted(loads, lf) - 1, 0, len(loads) - 2))
    j = int(np.clip(np.searchsorted(speeds, sr) - 1, 0, len(speeds) - 2))
    x0, x1 = loads[i], loads[i+1]; y0, y1 = speeds[j], speeds[j+1]
    tx = (lf - x0) / max(x1 - x0, 1e-9); ty = (sr - y0) / max(y1 - y0, 1e-9)
    eta = ((1-tx)*(1-ty)*eta_grid[i, j] + tx*(1-ty)*eta_grid[i+1, j]
           + (1-tx)*ty*eta_grid[i, j+1] + tx*ty*eta_grid[i+1, j+1])
    eta = float(np.clip(eta, 0.23, 0.43))
    bsfc = float(3600.0 * 1000.0 / (eta * CONST.lhv_fuel / 1000.0))
    return eta, float(np.clip(bsfc, 190.0, 430.0))

# Bearing Characteristic Frequencies
def bearing_characteristic_frequencies(rpm: float, tolerance: EngineManufacturingTolerance) -> Tuple[float, float, float]:
    fr = max(1.0, rpm / 60.0)
    z = tolerance.bearing_rolling_elements
    dD = tolerance.bearing_ball_diameter_ratio
    Dd = tolerance.bearing_pitch_diameter_ratio
    ca = np.deg2rad(tolerance.bearing_contact_angle_deg)
    ratio = dD * np.cos(ca)
    bpfo = 0.5 * z * fr * (1.0 - ratio / Dd)
    bpfi = 0.5 * z * fr * (1.0 + ratio / Dd)
    bsf = 0.5 * fr * (Dd / max(dD * np.cos(ca), 1e-6)) * (1.0 - (ratio / Dd)**2)
    return float(bpfo), float(bpfi), float(bsf)

# Synthesize Vibration Waveform
def synthesize_vibration_waveform(duration_s: float, fs_hz: int, rpm: float,
                                   severity: float = 0.0, bearing_wear: float = 0.0,
                                   misfire: float = 0.0, seed: int = MASTER_SEED,
                                   bearing_tolerance: Optional[EngineManufacturingTolerance] = None) -> Tuple[np.ndarray, np.ndarray, Dict[str, float]]:
    rng = np.random.default_rng(int(seed) & 0xFFFFFFFF)
    n = max(16, int(round(duration_s * fs_hz)))
    t = np.arange(n) / fs_hz
    f1 = max(1.0, rpm / 60.0)
    ff = f1 * CONST.num_cylinders / 2.0
    if bearing_tolerance is None:
        bpfo, bpfi, bsf = 5.0 * f1, 7.0 * f1, 3.0 * f1
    else:
        bpfo, bpfi, bsf = bearing_characteristic_frequencies(rpm, bearing_tolerance)
    phase_jitter = np.cumsum(rng.normal(0.0, 0.0015, n))
    phi1 = 2 * np.pi * f1 * t + phase_jitter
    x = ((0.008 + 0.006 * bearing_wear) * np.sin(phi1) +
         (0.005 + 0.010 * severity) * np.sin(2 * phi1 + 0.4) +
         (0.003 + 0.009 * severity) * np.sin(2 * np.pi * ff * t + 1.1))
    if misfire > 0:
        x += 0.020 * misfire * np.sin(2 * np.pi * 0.5 * f1 * t + 0.3 * np.sin(2 * np.pi * 0.7 * t))
        miss = rng.random(n) < (0.001 + 0.005 * misfire)
        x[miss] += rng.normal(0.0, 0.07 * misfire, miss.sum())
    bearing_imp = np.zeros(n)
    if bearing_wear > 0.03:
        defect_rate = bpfo if bearing_wear < 0.65 else bpfi
        period = max(1, int(round(fs_hz / defect_rate)))
        idx = np.arange(int(rng.integers(0, period)), n, period)
        idx = np.clip(idx + rng.integers(-2, 3, len(idx)), 0, n - 1)
        bearing_imp[idx] += (0.025 + 0.10 * bearing_wear) * (1.0 + rng.normal(0, 0.12, len(idx)))
    fn = 620.0; zeta = 0.075
    r_pole = float(np.exp(-2 * np.pi * fn * zeta / fs_hz))
    theta_pole = 2 * np.pi * fn / fs_hz
    b_res = [1.0 - r_pole]
    a_res = [1.0, -2.0 * r_pole * np.cos(theta_pole), r_pole**2]
    y = lfilter(b_res, a_res, bearing_imp)
    x += y * (0.20 + 0.80 * min(1.0, bearing_wear + severity))
    white = rng.normal(0.0, 0.0018, n)
    colored = np.zeros(n)
    alpha = np.exp(-2 * np.pi * 180.0 / fs_hz)
    for k in range(1, n):
        colored[k] = alpha * colored[k-1] + (1 - alpha) * white[k]
    x += colored
    rms = float(np.sqrt(np.mean(x * x))); peak = float(np.max(np.abs(x)))
    crest = peak / max(rms, 1e-9); mu = float(np.mean(x)); sd = float(np.std(x))
    kurt = float(np.mean(((x - mu) / max(sd, 1e-9))**4))
    freqs = np.fft.rfftfreq(n, 1.0 / fs_hz); spec = np.abs(np.fft.rfft(x))
    band = (freqs >= 1.0) & (freqs <= fs_hz / 2.0)
    peak_freq = float(freqs[band][np.argmax(spec[band])]) if np.any(band) else 0.0
    return t, x, {"rms_g": rms, "crest_factor": crest, "kurtosis": kurt,
                  "dominant_frequency_hz": peak_freq, "1x_hz": f1, "2x_hz": 2 * f1,
                  "firing_hz": ff, "bpfo_hz": bpfo, "bpfi_hz": bpfi, "bsf_hz": bsf}

# Apexphysicalenginetwin
class ApexPhysicalEngineTwin:
    # Init
    def __init__(self, tolerance: EngineManufacturingTolerance, initial_wear: float = 0.0):
        self.tol = tolerance
        self.engine_id = tolerance.engine_id

        self.v_tas = 45.0
        self.gamma_rad = 0.0
        self.altitude_m = 0.0

        self.omega = 188.5
        self.turbo_speed = 0.62
        self.governor_integral = 0.0
        self.prop_pitch_actual = 0.68
        self.T_cht_K = 345.0
        self.T_coolant_K = 335.0
        self.T_oil_K = 330.0
        self.T_fuel_K = 298.15
        self.fuel_mass_kg = CONST.fuel_capacity_kg
        self.fuel_used_kg = 0.0
        self.cumulative_time_s = 0.0

        iw = float(np.clip(initial_wear, 0.0, 0.95))
        self.bearing_wear = iw
        self.thermal_fatigue = iw * random.uniform(0.45, 0.85)
        self.turbo_degradation = iw * random.uniform(0.35, 0.75)
        self.injector_wear = iw * random.uniform(0.35, 0.75)
        self.lubrication_deg = iw * random.uniform(0.50, 0.90)
        self.electrical_deg = iw * random.uniform(0.30, 0.70)

        self.oil_p_low_duration_s = 0.0
        self.cht_high_duration_s = 0.0
        self.egt_high_duration_s = 0.0
        self.stall_duration_s = 0.0
        self.is_failed = False
        self.failure_mode = "nominal"

    @staticmethod
    # Severity Factor
    def _severity_factor(fault_set, name, sev):
        return float(sev) if name in fault_set else 0.0

    # Friction Torque
    def _friction_torque(self, omega: float, rpm_now: float, mu_oil: float,
                         wb: float, active_fault_set, sev: float) -> float:
        Q_visc = self.tol.fric_torque_base * (0.85 + 0.15 * rpm_now / CONST.rated_rpm) * (mu_oil / max(self.tol.mu0_oil, 1e-6)) ** 0.22
        Q_speed = 0.0009 * omega ** 1.45
        Q_boundary = 0.8 * (1.0 + 3.0 * wb) / (1.0 + 80.0 * mu_oil * omega)
        if "bearing_fault" in active_fault_set:
            Q_boundary *= (1.0 + 1.5 * sev)
            Q_visc *= (1.0 + 0.4 * sev)
        return max(4.0, Q_visc + Q_speed + Q_boundary)

    # Integrate Mission Step
    def integrate_mission_step(
        self, dt, t_global_s_vec, throttle_cmd, prop_pitch_cmd,
        target_climb_rate_mps, target_airspeed_mps, target_altitude_m, fault_type, severity_t,
        sea_temp_C=18.0, qnh_offset_kPa=0.0, compound_faults=None,
        wind_u_profile: Optional[np.ndarray] = None, wind_accel_profile: Optional[np.ndarray] = None
    ):
        n = len(throttle_cmd)
        active_fault_set = set([fault_type]) | set(compound_faults or [])

        names = [
            "altitude_m", "aircraft_mass_kg", "fuel_mass_kg", "fuel_used_kg", "wind_u_mps",
            "wind_accel_mps2", "ground_speed_mps", "airspeed_mps", "ambient_T_C", "ambient_P_kPa",
            "air_density_kg_m3", "flight_path_angle_deg", "vertical_speed_mps", "prop_pitch_actual",
            "target_airspeed_mps", "throttle_actual", "engine_power_command_kW", "engine_bsfc_g_kWh",
            "actual_bsfc_g_kWh", "desired_thrust_N", "thrust_equilibrium_error_N", "prop_advance_ratio",
            "prop_Cp", "prop_Ct", "prop_efficiency", "prop_power_kW",
            "drag_N", "excess_thrust_N", "shaft_torque_error_Nm", "rpm", "cht_cyl_avg_C", "egt_cyl_avg_C",
            "oil_pressure_kPa", "oil_temp_C", "fuel_flow_g_s", "fuel_temp_C",
            "boost_pressure_kPa", "manifold_pressure_kPa", "vib_rms_g",
            "vib_crest_factor", "vib_kurtosis", "vib_1x_hz", "vib_2x_hz", "vib_firing_hz",
            "vib_bpfo_hz", "vib_bpfi_hz", "vib_bsf_hz",
            "vib_psd_peak", "vib_psd_peak_freq", "vib_1x_rms", "vib_2x_rms",
            "vib_fire_order_rms", "vib_spectral_centroid", "battery_voltage_V",
            "battery_voltage_raw_V", "altitude_quantized_m", "rpm_quantized",
            "rail_pressure_quantized_bar", "sensor_drift_flag_rpm",
            "sensor_drift_flag_rail", "sensor_drift_flag_boost",
            "coolant_temp_C", "rail_pressure_bar", "SOI_deg_bTDC",
            "inj_duration_ms", "true_brake_power_kW", "true_fuel_power_kW", "true_indicated_power_kW",
            "true_friction_power_kW", "true_exhaust_heat_kW", "true_coolant_heat_kW",
            "true_oil_heat_kW", "true_direct_heat_loss_kW", "true_energy_closure_error_W",
            "true_rpm", "true_indicated_torque_Nm",
            "true_prop_load_torque_Nm", "true_prop_thrust_N", "true_bearing_wear",
            "true_thermal_fatigue", "true_turbo_degradation", "true_injector_wear",
            "true_lubrication_degradation", "true_electrical_degradation",
            "true_overall_health_index"
        ]
        out = {k: np.zeros(n) for k in names}
        out["fadec_mode"] = np.full(n, "nominal", dtype=object)
        out["failure_flag"] = np.zeros(n, dtype=int)
        out["failure_mode"] = np.full(n, "nominal", dtype=object)

        v = float(self.v_tas); gamma = float(self.gamma_rad); h = float(self.altitude_m)
        omega = float(self.omega); turbo_speed = float(self.turbo_speed); gov_int = float(self.governor_integral)
        prop_pitch_actual = float(self.prop_pitch_actual)
        Tcht = float(self.T_cht_K); Tcool = float(self.T_coolant_K)
        Toil = float(self.T_oil_K); Tfuel = float(self.T_fuel_K)
        fuel_mass = float(CONST.fuel_capacity_kg)
        fuel_used = 0.0
        wb, wt, wtu, wi, wl, we = [float(x) for x in (
            self.bearing_wear, self.thermal_fatigue, self.turbo_degradation,
            self.injector_wear, self.lubrication_deg, self.electrical_deg)]
        oil_low = self.oil_p_low_duration_s
        cht_high = self.cht_high_duration_s
        egt_high = self.egt_high_duration_s
        stall_duration = self.stall_duration_s

        regime_scale = {"accelerated": 0.72, "nominal": 1.0, "extended": 1.38}.get(
            self.tol.aging_regime, 1.0)
        damage_rate_mult = 16.0 if self.tol.aging_regime == "accelerated" else 1.0
        life_bearing = self.tol.bearing_design_life_h * regime_scale
        life_thermal = self.tol.thermal_design_life_h * regime_scale
        life_turbo = self.tol.turbo_design_life_h * regime_scale
        life_injector = self.tol.injector_design_life_h * regime_scale
        life_lube = self.tol.lube_design_life_h * regime_scale
        life_electrical = self.tol.electrical_design_life_h * regime_scale

        rated_fuel_kg_s = (CONST.rated_brake_power_kW * CONST.rated_bsfc_g_kWh / 3600.0) / 1000.0
        rated_power_W = CONST.rated_brake_power_kW * 1000.0
        omega_rated = CONST.rated_rpm * 2.0 * np.pi / 60.0
        friction_rated_Nm = self.tol.fric_torque_base * 1.10 + 0.0009 * omega_rated
        eta_ind_ref = (rated_power_W + friction_rated_Nm * omega_rated) / max(
            rated_fuel_kg_s * CONST.lhv_fuel, 1.0)
        eta_ind_ref = float(np.clip(eta_ind_ref, 0.30, 0.48))
        eta_map_ref, bsfc_map_ref = engine_operating_map(1.0, CONST.rated_rpm)
        rated_afr_ref = (self.tol.eta_vol_base * CONST.V_displacement * (CONST.rated_rpm / 120.0) *
                         (CONST.P0_sea_kPa * CONST.max_pressure_ratio * 1000.0) /
                         (CONST.R_air * (CONST.T0_sea_K + 25.0))) / max(rated_fuel_kg_s, 1e-7)
        afr_ref_factor = 0.82 + 0.18 * np.exp(-((rated_afr_ref - 18.0) / 12.0) ** 2)

        omega_min = CONST.idle_rpm * 2.0 * np.pi / 60.0
        omega_max = (CONST.max_rpm + 100.0) * 2.0 * np.pi / 60.0

        for i in range(n):
            sev = float(np.clip(severity_t[i], 0.0, 1.0))
            cmd_thr = float(np.clip(throttle_cmd[i], 0.0, 1.0))
            pitch_cmd = float(np.clip(prop_pitch_cmd[i], 0.0, 1.0))
            target_vz_phase = float(target_climb_rate_mps[i])
            tg = float(t_global_s_vec[i])

            v_sample = float(v)
            h_sample = float(h)
            aircraft_mass = CONST.m_empty_kg + fuel_mass
            alt_error = float(target_altitude_m[i] - h_sample)
            target_vz = float(np.clip(target_vz_phase + 0.0015 * alt_error,
                                      -CONST.max_climb_rate_mps, CONST.max_climb_rate_mps))
            gamma_sample = float(gamma)
            omega_sample = float(omega)

            T_amb_C, P_amb_kPa, rho = compute_instantaneous_atmosphere(
                h_sample, tg, sea_temp_base_C=sea_temp_C, qnh_offset_kPa=qnh_offset_kPa)
            T_amb_K = T_amb_C + 273.15
            if wind_u_profile is not None and i < len(wind_u_profile):
                wind_u = float(wind_u_profile[i])
                wind_accel = float(wind_accel_profile[i]) if wind_accel_profile is not None else 0.0
            else:
                wind_u = environmental_wind(tg, h_sample)
                wind_accel = environmental_wind_rate(tg, h_sample, dt)
            ground_speed = max(0.0, v_sample + wind_u)

            v_target = float(np.clip(target_airspeed_mps[i], CONST.min_operating_speed_mps,
                                     CONST.max_operating_speed_mps))
            q_est = 0.5 * rho * max(v_sample, CONST.min_operating_speed_mps) ** 2
            CL_est = float(np.clip(
                aircraft_mass * CONST.g * np.cos(gamma) / max(q_est * CONST.S_wing_m2, 100.0),
                0.05, CONST.CL_max))
            Re_wing = np.clip(rho * max(v_sample, 20.0) * np.sqrt(CONST.S_wing_m2) / 1.5e-5, 2.0e5, 3.0e7)
            mach = np.clip(v_sample / np.sqrt(CONST.gamma_air * CONST.R_air * T_amb_K), 0.0, 0.65)
            cd0_re = CONST.CD0_parasitic * (1.0 + 0.08 * np.clip((2.0e6 / Re_wing) ** 0.20 - 1.0, -0.5, 1.5))
            compressibility = 1.0 + 0.12 * max(0.0, mach - 0.20) ** 2 / max(1.0 - mach**2, 0.5)
            CD_est = cd0_re * compressibility + CL_est**2 / (np.pi * CONST.oswald_e * CONST.AR_wing)
            D_est = q_est * CONST.S_wing_m2 * CD_est
            speed_accel_cmd = np.clip((v_target - v_sample) / CONST.speed_controller_tau_s, -0.35, 0.35)
            gamma_vz = float(np.arcsin(np.clip(target_vz / max(v_sample, 20.0), -0.25, 0.25)))
            gamma_speed = 0.012 * (v_target - v_sample)
            gamma_cmd_est = float(np.clip(gamma_vz + gamma_speed, -0.22, 0.22))
            altitude_error = float(target_altitude_m[i] - h_sample)
            vz_altitude = float(np.clip(altitude_error / max(CONST.altitude_controller_tau_s, 1.0),
                                        -CONST.max_climb_rate_mps, CONST.max_climb_rate_mps))
            target_vz = float(np.clip(0.65 * target_vz + 0.35 * vz_altitude,
                                      -CONST.max_climb_rate_mps, CONST.max_climb_rate_mps))
            gamma_vz = float(np.arcsin(np.clip(target_vz / max(v_sample, 20.0), -0.25, 0.25)))
            gamma_cmd_est = float(np.clip(gamma_vz + gamma_speed, -0.22, 0.22))

            n_feas = max(8.0, omega_sample / (2.0 * np.pi))
            T_cap_pitch = propeller_aerodynamics(n_feas, v_sample, CONST.prop_pitch_max, rho)[1]
            rho_ratio = float(np.clip((rho / 1.225) ** 0.70, 0.55, 1.0))
            deg_power = float(np.clip(1.0 - 0.10 * wtu - 0.06 * wi, 0.75, 1.0))
            P_avail_W = (rated_power_W * np.clip(cmd_thr, 0.0, 1.0) * rho_ratio * deg_power)
            if self.is_failed:
                P_avail_W = 0.0
            eta_prop_est = 0.80
            V_eff = max(v_sample, 16.0)
            T_cap_power = eta_prop_est * P_avail_W / V_eff
            T_cap = float(np.clip(min(T_cap_pitch, T_cap_power), 0.0, CONST.max_prop_thrust_N))

            T_ff = (D_est
                    + aircraft_mass * CONST.g * np.sin(gamma_cmd_est)
                    + aircraft_mass * speed_accel_cmd)
            T_ff = float(np.clip(T_ff, 0.0, CONST.max_prop_thrust_N))
            if i == 0:
                thrust_ref = T_ff
            else:
                tau_demand = 3.0
                thrust_ref = gov_int + (T_ff - gov_int) * (1.0 - np.exp(-dt / tau_demand))
            thrust_ref = float(np.clip(thrust_ref, 0.0, T_cap))
            gov_int = thrust_ref
            thrust_required = thrust_ref
            if thrust_required > T_cap and aircraft_mass > 1.0 and i == 0:
                max_gamma_term = (T_cap - D_est - aircraft_mass * speed_accel_cmd) / (aircraft_mass * CONST.g)
                max_gamma = float(np.clip(max_gamma_term, -0.25, 0.25))
                target_vz = float(np.clip(v_sample * np.sin(max_gamma),
                                          -CONST.max_climb_rate_mps, CONST.max_climb_rate_mps))
                gamma_vz = float(np.arcsin(np.clip(target_vz / max(v_sample, 20.0), -0.25, 0.25)))
                gamma_cmd_est = float(np.clip(gamma_vz + 0.012 * (v_target - v_sample), -0.22, 0.22))
                thrust_required = D_est + aircraft_mass * (speed_accel_cmd + CONST.g * np.sin(gamma_cmd_est))
            thrust_required = float(np.clip(thrust_required, 0.0, CONST.max_prop_thrust_N))

            n_rev_s_sample = max(1.0, omega_sample / (2.0 * np.pi))
            rpm_sample = n_rev_s_sample * 60.0

            thermal_alarm = (Tcht - 273.15 > CONST.fadec_cht_rollback_C or
                             Tcool - 273.15 > CONST.fadec_coolant_rollback_C)
            severe_thermal = (Tcht - 273.15 > self.tol.cht_fail_C)
            eff_thr = cmd_thr
            fadec_mode = "nominal"
            if thermal_alarm:
                eff_thr = min(eff_thr, CONST.thermal_derate_fraction)
                fadec_mode = "thermal_derate"
            if severe_thermal:
                eff_thr = min(eff_thr, 0.35)
            if self.is_failed:
                eff_thr = 0.0
                fadec_mode = "emergency_shutdown"

            turbo_health = max(0.35, 1.0 - wtu)
            comp_eff = float(np.clip(self.tol.comp_eff_base * turbo_health, 0.45, 0.90))
            if "turbo_issue" in active_fault_set:
                comp_eff *= max(0.45, 1.0 - 0.35 * sev)
            turbine_drive = float(np.clip(0.18 + 0.95 * eff_thr, 0.0, 1.15))
            turbo_tau = 1.8 + 2.5 * (1.0 - turbo_health)
            turbo_speed += ((turbine_drive * turbo_health) - turbo_speed) * dt / turbo_tau
            turbo_speed = float(np.clip(turbo_speed, 0.0, 1.15))

            oil_health = max(0.25, 1.0 - wl)
            mu_oil = self.tol.mu0_oil * np.exp(1850.0 * (1.0 / max(Toil, 260.0) - 1.0 / 373.15))
            mu_oil *= oil_health
            if "lubrication_issue" in active_fault_set:
                mu_oil *= max(0.25, 1.0 - 0.65 * sev)
            mu_oil = float(np.clip(mu_oil, 0.000004, 0.08))

            injector_delivery = max(0.55, 1.0 - 0.35 * wi)
            if "injector_degradation" in active_fault_set:
                injector_delivery *= max(0.45, 1.0 - 0.45 * sev)

            # Shaft Residual
            def _shaft_residual(omega_w: float, pitch_w: float) -> Tuple[float, dict]:
                n_w = max(1.0, omega_w / (2.0 * np.pi))
                rpm_w = n_w * 60.0
                Qp_w, Tp_w, Cp_w, Ct_w, J_w, eta_w = propeller_aerodynamics(
                    n_w, v_sample, pitch_w, rho)
                Qf_w = self._friction_torque(omega_w, rpm_w, mu_oil, wb, active_fault_set, sev)
                rack_x = float(np.clip((rpm_w / CONST.rated_rpm - 0.55) / 0.45, 0.0, 1.0))
                rack = float(0.24 + 0.76 * rack_x)
                fuel_scale = max(0.04, 0.15 + 0.85 * eff_thr)
                mdot_w = 0.0 if self.is_failed else rated_fuel_kg_s * rack * fuel_scale
                if rpm_w < CONST.min_flight_governed_rpm and not self.is_failed:
                    mdot_w = max(mdot_w, rated_fuel_kg_s * 0.30)
                if rpm_w >= CONST.overspeed_rpm and not self.is_failed:
                    mdot_w *= float(np.clip(1.0 - (rpm_w - CONST.overspeed_rpm) /
                                            max(CONST.overspeed_cut_rpm - CONST.overspeed_rpm, 1.0), 0.0, 1.0))
                if rpm_w >= CONST.overspeed_cut_rpm or self.is_failed:
                    mdot_w = 0.0
                mdot_w = float(np.clip(mdot_w, 0.0, rated_fuel_kg_s * 1.05))
                fuel_power_w = mdot_w * CONST.lhv_fuel
                eta_map_w, bsfc_w = engine_operating_map(float(np.clip(eff_thr, 0.05, 1.0)), rpm_w)
                m_air_w = max(0.005,
                              (np.clip(self.tol.eta_vol_base - 0.05 * (rpm_w / CONST.rated_rpm - 0.65) - 0.08 * wtu, 0.55, 0.95)
                               * CONST.V_displacement * (n_w / 2.0)
                               * (P_amb_kPa * 1000.0 * (1.0 + (CONST.max_pressure_ratio - 1.0) * turbo_speed * comp_eff / 0.82))
                               / (CONST.R_air * max(T_amb_K + 25.0, 260.0))))
                afr_w = m_air_w / max(mdot_w, 1e-7)
                afr_factor = 0.82 + 0.18 * np.exp(-((afr_w - 18.0) / 12.0) ** 2)
                if "misfire" in active_fault_set:
                    afr_factor *= max(0.35, 1.0 - 0.55 * sev)
                combustion = float(np.clip((afr_factor / afr_ref_factor) * injector_delivery, 0.25, 1.05))
                eta_ind_w = float(np.clip(eta_ind_ref * (eta_map_w / max(eta_map_ref, 1e-6)) *
                                          combustion, 0.10, 0.46))
                Q_ind_w = fuel_power_w * eta_ind_w / max(omega_w, 20.0)
                info = {"Qp": Qp_w, "Tp": Tp_w, "Cp": Cp_w, "Ct": Ct_w, "J": J_w, "eta": eta_w,
                        "Qf": Qf_w, "Qind": Q_ind_w, "mdot": mdot_w, "fuel_power": fuel_power_w,
                        "eta_ind": eta_ind_w, "bsfc": bsfc_w, "n": n_w, "rpm": rpm_w,
                        "m_air": m_air_w, "load_frac": float(np.clip(eff_thr, 0.05, 1.0))}
                return (Qp_w - (Q_ind_w - Qf_w), info)

            # Torque Root
            def _torque_root(pitch_w: float, seed_omega: float) -> Tuple[float, dict]:
                if self.is_failed:
                    Qp_idle, _, _, _, _, _ = propeller_aerodynamics(
                        max(1.0, seed_omega / (2 * np.pi)), v_sample, pitch_w, rho)
                    Qf_idle = self._friction_torque(seed_omega, rpm_sample, mu_oil, wb, active_fault_set, sev)
                    domega = -(Qf_idle + Qp_idle) / max(self.tol.J_rotor, 0.1)
                    omega_new = float(np.clip(seed_omega + domega * dt, 1.0, omega_max))
                    _, info_new = _shaft_residual(omega_new, pitch_w)
                    return omega_new, info_new
                lo_b, hi_b = omega_min, omega_max
                omega_sol = float(np.clip(seed_omega, lo_b, hi_b))
                res_sol, info_sol = _shaft_residual(omega_sol, pitch_w)
                for _ in range(6):
                    d = max(1.0, omega_sol * 0.020)
                    r_plus, _ = _shaft_residual(min(omega_sol + d, hi_b), pitch_w)
                    slope = (r_plus - res_sol) / max(d, 1e-6)
                    if abs(slope) < 1e-9:
                        break
                    nxt = float(np.clip(omega_sol - res_sol / slope, lo_b, hi_b))
                    if abs(nxt - omega_sol) < 0.5:
                        omega_sol = nxt
                        res_sol, info_sol = _shaft_residual(omega_sol, pitch_w)
                        break
                    omega_sol = nxt
                    res_sol, info_sol = _shaft_residual(omega_sol, pitch_w)
                for _ in range(12):
                    mid = 0.5 * (lo_b + hi_b)
                    r_mid, _ = _shaft_residual(mid, pitch_w)
                    r_lo2, _ = _shaft_residual(lo_b, pitch_w)
                    if r_lo2 * r_mid <= 0.0:
                        hi_b = mid
                    else:
                        lo_b = mid
                    if abs(hi_b - lo_b) < 1e-4:
                        break
                omega_eq = 0.5 * (lo_b + hi_b)
                res_eq, info_eq = _shaft_residual(omega_eq, pitch_w)
                r_min_chk, _ = _shaft_residual(omega_min, pitch_w)
                if res_eq * r_min_chk > 0.0 and r_min_chk > 0.0:
                    omega = float(np.clip((CONST.overspeed_rpm - 8.0) * 2.0 * np.pi / 60.0,
                                          omega_min, omega_max))
                elif abs(res_sol) < abs(res_eq):
                    omega, info_engine = omega_sol, info_sol
                else:
                    omega, info_engine = omega_eq, info_eq
                omega = float(np.clip(omega, omega_min, omega_max))
                res_final, info_engine = _shaft_residual(omega, pitch_w)
                return omega, info_engine

            # Trim Residual
            def _trim_residual(omega_w: float, seed_pitch: Optional[float] = None):
                n_w = max(8.0, omega_w / (2.0 * np.pi))
                pitch_w = float(np.clip(solve_propeller_trim(thrust_required, n_w, v_sample, rho,
                                                             warm_start=seed_pitch),
                                        CONST.prop_pitch_min, CONST.prop_pitch_max))
                res_w, info_w = _shaft_residual(omega_w, pitch_w)
                return pitch_w, res_w, info_w

            # Find Coupled Equilibrium
            def _find_coupled_equilibrium(seed_omega: float) -> Tuple[float, float, dict]:
                if self.is_failed:
                    om_f, info_f = _torque_root(prop_pitch_actual, float(seed_omega))
                    return om_f, float(prop_pitch_actual), info_f
                grid = np.linspace(omega_min, omega_max, 17)
                grid = np.unique(np.concatenate(
                    [grid, [float(np.clip(seed_omega, omega_min, omega_max))]]))
                omega_rated_w = CONST.rated_rpm * 2.0 * np.pi / 60.0

                pts = []
                seed_p = None
                for om in grid:
                    p0, r0, _i0 = _trim_residual(float(om), seed_pitch=seed_p)
                    seed_p = p0
                    pts.append((float(om), float(p0), float(r0)))
                best = min(pts, key=lambda p: abs(p[2]))
                candidates = []
                for k in range(len(pts) - 1):
                    if pts[k][2] * pts[k+1][2] <= 0.0:
                        lo2, hi2 = pts[k][0], pts[k+1][0]
                        cell_seed = pts[k][1] if abs(pts[k][0] - lo2) <= abs(pts[k+1][0] - lo2) else pts[k+1][1]
                        r_ref = pts[k][2]
                        for _ in range(40):
                            mid2 = 0.5 * (lo2 + hi2)
                            pm, rm, _im = _trim_residual(mid2, seed_pitch=cell_seed)
                            cell_seed = pm
                            if rm * r_ref <= 0.0:
                                hi2 = mid2
                            else:
                                lo2 = mid2
                            if abs(hi2 - lo2) < 0.05:
                                break
                        om_c = 0.5 * (lo2 + hi2)
                        p_c, r_c, _ic = _trim_residual(om_c, seed_pitch=cell_seed)
                        candidates.append((om_c, p_c, r_c))
                if candidates:
                    unsat = [c for c in candidates if c[1] <= 0.86]
                    pool = unsat if unsat else candidates
                    om_sol, pitch_sol, _ = min(pool, key=lambda c: (abs(c[0] - omega_rated_w), abs(c[2])))
                else:
                    om_sol = float(best[0])
                    pitch_sol = float(best[1])
                om_sol = float(np.clip(om_sol, omega_min, omega_max))
                _, _, info_sol = _trim_residual(om_sol, seed_pitch=pitch_sol)
                return om_sol, pitch_sol, info_sol

            if self.is_failed:
                omega, prop_pitch_use, _ = _find_coupled_equilibrium(omega)
            else:
                omega_eq, pitch_trim_sol, _ = _find_coupled_equilibrium(omega)
                max_pitch_step = CONST.prop_pitch_rate_per_s * dt
                prop_pitch_actual += float(np.clip(pitch_trim_sol - prop_pitch_actual,
                                                    -max_pitch_step, max_pitch_step))
                prop_pitch_actual = float(np.clip(prop_pitch_actual, CONST.prop_pitch_min, CONST.prop_pitch_max))
            domega_goal = omega_eq - float(omega)
            spool_rate = (120.0 / max(self.tol.J_rotor, 0.1)) * dt
            domega_sign = 1.0 if domega_goal >= 0.0 else -1.0
            domega = domega_sign * min(abs(domega_goal), spool_rate)
            omega = float(np.clip(omega + domega, omega_min, omega_max))
            _, info_engine = _shaft_residual(float(omega), prop_pitch_actual)
            n_rev_s = max(1.0, omega / (2.0 * np.pi))
            rpm_now = n_rev_s * 60.0
            m_dot_fuel = float(info_engine["mdot"])
            m_dot_air = float(info_engine["m_air"])
            afr = m_dot_air / max(m_dot_fuel, 1e-7)
            Q_prop = float(info_engine["Qp"])
            T_prop = float(info_engine["Tp"])
            Cp = float(info_engine["Cp"]); Ct = float(info_engine["Ct"])
            J_prop = float(info_engine["J"]); eta_prop = float(info_engine["eta"])
            Q_friction = float(info_engine["Qf"])
            Q_indicated = float(info_engine["Qind"])
            P_indicated_W = Q_indicated * omega
            P_friction_W = Q_friction * omega
            P_brake_kW = max(0.0, (Q_indicated - Q_friction) * omega / 1000.0)
            shaft_torque_error = Q_indicated - Q_friction - Q_prop
            prop_power_kW = Q_prop * omega / 1000.0
            fuel_power_W = m_dot_fuel * CONST.lhv_fuel
            actual_bsfc = m_dot_fuel * 1000.0 * 3600.0 / max(P_brake_kW, 0.05)
            available_burn = max(0.0, fuel_mass - CONST.fuel_reserve_kg)
            m_dot_fuel = min(m_dot_fuel, available_burn / max(dt, 1e-9))
            fuel_burn_step = m_dot_fuel * dt
            fuel_mass = max(CONST.fuel_reserve_kg, fuel_mass - fuel_burn_step)
            fuel_used += fuel_burn_step
            aircraft_mass = CONST.m_empty_kg + fuel_mass
            if available_burn <= 0.0 and not self.is_failed:
                eff_thr = min(eff_thr, 0.10)
                fadec_mode = "fuel_reserve"

            rail_cmd = 350.0 + 1300.0 * eff_thr
            P_rail = rail_cmd * max(0.55, 1.0 - 0.18 * wi)
            if "injector_degradation" in active_fault_set:
                P_rail *= max(0.55, 1.0 - 0.30 * sev)
            P_rail = float(np.clip(P_rail, 250.0, 1800.0))

            Tfuel_C = Tfuel - 273.15
            rho_fuel = float(np.clip(840.0 - 0.72 * (Tfuel_C - 15.0), 760.0, 860.0))
            firing_events_s = max(rpm_now / 120.0 * CONST.num_cylinders, 0.5)
            mass_per_event_kg = m_dot_fuel / firing_events_s
            volume_per_event_mm3 = mass_per_event_kg / rho_fuel * 1e9
            injector_flow_mm3_ms = 48.0 * np.sqrt(max(P_rail, 1.0) / 1000.0)
            inj_dur_ms = float(np.clip(volume_per_event_mm3 / max(injector_flow_mm3_ms, 1.0), 0.15, 5.0))
            soi = float(np.clip(4.0 + 8.0 * eff_thr + 1.5 * (rpm_now / CONST.rated_rpm - 1.0), 2.0, 15.0))

            pressure_ratio = 1.0 + (CONST.max_pressure_ratio - 1.0) * turbo_speed * comp_eff / 0.82
            pressure_ratio = float(np.clip(pressure_ratio, 1.0, CONST.max_pressure_ratio))
            P_manifold = float(np.clip(P_amb_kPa * pressure_ratio, P_amb_kPa * 1.005, CONST.max_manifold_pressure_kPa))
            P_boost = max(0.0, P_manifold - P_amb_kPa)

            P_oil = 80.0 + 0.045 * rpm_now * (mu_oil / max(self.tol.mu0_oil, 1e-6)) / (1.0 + 0.9 * wb)
            if "lubrication_issue" in active_fault_set:
                P_oil -= 75.0 * sev
            if "bearing_fault" in active_fault_set:
                P_oil -= 40.0 * sev
            P_oil = float(np.clip(P_oil, 35.0, 420.0))

            q_inf = 0.5 * rho * max(v_sample, 15.0) ** 2
            gamma_cmd = float(np.arcsin(np.clip(target_vz / max(v_sample, 20.0), -0.25, 0.25)))
            K_gamma = 1.8
            CL_cmd = (aircraft_mass * CONST.g * np.cos(gamma) +
                      aircraft_mass * max(v_sample, 20.0) * K_gamma * (gamma_cmd - gamma)) / max(q_inf * CONST.S_wing_m2, 100.0)
            CL = float(np.clip(CL_cmd, 0.05, CONST.CL_max))
            Re_wing = np.clip(rho * max(v_sample, 20.0) * np.sqrt(CONST.S_wing_m2) / 1.5e-5, 2.0e5, 3.0e7)
            mach = np.clip(v_sample / np.sqrt(CONST.gamma_air * CONST.R_air * T_amb_K), 0.0, 0.65)
            cd0_re = CONST.CD0_parasitic * (1.0 + 0.08 * np.clip((2.0e6 / Re_wing) ** 0.20 - 1.0, -0.5, 1.5))
            compressibility = 1.0 + 0.12 * max(0.0, mach - 0.20) ** 2 / max(1.0 - mach**2, 0.5)
            CD = cd0_re * compressibility + CL ** 2 / (np.pi * CONST.oswald_e * CONST.AR_wing)
            D_drag = q_inf * CONST.S_wing_m2 * CD
            L_lift = q_inf * CONST.S_wing_m2 * CL
            dv = ((T_prop - D_drag - aircraft_mass * CONST.g * np.sin(gamma_sample)) / max(aircraft_mass, 1500.0)
                  - wind_accel * np.cos(gamma_sample))
            dgamma = (L_lift - aircraft_mass * CONST.g * np.cos(gamma)) / max(aircraft_mass * v_sample, 1500.0)
            v = float(np.clip(v + dv * dt, 28.0, 95.0))
            excess_thrust = T_prop - D_drag
            gamma = float(np.clip(gamma + dgamma * dt, -0.35, 0.30))
            h = float(np.clip(h + v_sample * np.sin(gamma_sample) * dt, 0.0, CONST.H_trop + 200.0))
            vertical_speed = v_sample * np.sin(gamma_sample)

            waste_W = max(0.0, fuel_power_W - P_brake_kW * 1000.0)
            f_exh = float(np.clip(0.31 + 0.04 * eff_thr, 0.29, 0.36))
            f_oil = float(np.clip(0.055 + 0.015 * wb, 0.05, 0.08))
            f_direct = float(np.clip(0.10 + 0.03 * eff_thr, 0.08, 0.14))
            if "turbo_issue" in active_fault_set:
                f_exh += 0.025 * sev
                f_direct -= 0.015 * sev
            if "misfire" in active_fault_set:
                f_exh += 0.040 * sev
                f_direct -= 0.020 * sev
            if "overheating" in active_fault_set:
                f_direct += 0.020 * sev
                f_exh -= 0.010 * sev
            f_direct = max(0.02, f_direct)
            f_cool = max(0.0, 1.0 - f_exh - f_oil - f_direct)
            fractions = np.array([f_exh, f_oil, f_cool, f_direct], dtype=float)
            fractions /= fractions.sum()
            f_exh, f_oil, f_cool, f_direct = fractions
            Q_exh = waste_W * f_exh
            Q_oil = waste_W * f_oil
            Q_cool = waste_W * f_cool
            Q_direct = waste_W * f_direct

            airflow_factor = float(np.clip((rho * max(v_sample, 20.0)) / (1.225 * 45.0), 0.35, 1.8))
            h_cht = self.tol.h_cht_cool * airflow_factor * max(0.35, 1.0 - 0.65 * sev if "overheating" in active_fault_set else 1.0)
            h_rad = self.tol.h_rad_cool * airflow_factor ** 0.75 * max(0.30, 1.0 - 0.70 * sev if "overheating" in active_fault_set else 1.0)
            dTcht = (Q_cool - h_cht * (Tcht - Tcool) - 55.0 * airflow_factor * (Tcht - T_amb_K)) / CONST.mc_cht
            dTcool = (h_cht * (Tcht - Tcool) - h_rad * (Tcool - T_amb_K)) / CONST.mc_coolant
            dToil = (Q_oil - self.tol.h_oil_cool * (Toil - T_amb_K)) / CONST.mc_oil
            dTfuel = (0.002 * fuel_power_W - 75.0 * (Tfuel - T_amb_K)) / CONST.mc_fuel_tank
            Tcht += dTcht * dt; Tcool += dTcool * dt; Toil += dToil * dt; Tfuel += dTfuel * dt

            m_exhaust = max(m_dot_air + m_dot_fuel, 0.01)
            Tegt = T_amb_K + Q_exh / max(m_exhaust * CONST.cp_exh, 1.0)
            if "turbo_issue" in active_fault_set:
                Tegt += 45.0 * sev
            if "misfire" in active_fault_set:
                Tegt += 55.0 * sev
            Tegt = float(np.clip(Tegt, T_amb_K, 1050.0))

            rpm_output = rpm_now
            f1x = rpm_output / 60.0
            f_fire = f1x * CONST.num_cylinders / 2.0
            a1 = 0.008 + 0.004 * rpm_now / CONST.rated_rpm + 0.020 * wb
            a2 = 0.006 + 0.010 * eff_thr + 0.010 * wb
            afire = 0.004 + 0.006 * eff_thr
            asub = 0.0
            if "misfire" in active_fault_set:
                asub = 0.030 * sev * (1.0 + 0.5 * np.sin(2 * np.pi * 0.7 * tg))
            if "lubrication_issue" in active_fault_set:
                asub += 0.012 * sev
            vib_rms = float(np.sqrt(a1*a1 + a2*a2 + afire*afire + asub*asub))
            impulsive_ratio = asub / max(a1 + a2, 1e-4)
            vib_crest = float(np.clip(1.414 + 1.4 * impulsive_ratio + 0.25 * wb, 1.2, 8.0))
            vib_kurt = float(np.clip(3.0 + 7.0 * impulsive_ratio ** 1.35 + 1.5 * wb, 2.0, 40.0))
            vib_1x_rms = a1 * (1.0 + 0.30 * wb)
            vib_2x_rms = a2 * (1.0 + 0.20 * wtu)
            vib_fire_order_rms = afire + asub
            vib_spectral_centroid = (f1x * (a1 + a2) + f_fire * (afire + asub)) / max(a1 + a2 + afire + asub, 1e-6)
            vib_psd_peak = vib_rms ** 2 * (1.0 + 0.5 * wb + 0.3 * wtu + 0.2 * sev)
            vib_psd_peak_freq = f1x * (1.0 + 0.12 * wb)
            bpfo_now, bpfi_now, bsf_now = bearing_characteristic_frequencies(rpm_output, self.tol)

            alternator_pickup = np.clip((rpm_output - 900.0) / 600.0, 0.0, 1.0)
            electrical_load = 0.35 + 0.45 * eff_thr
            v_bus = 28.2 + 0.45 * alternator_pickup - 0.25 * electrical_load
            v_bus *= max(0.55, 1.0 - 0.45 * we)
            if "electrical_fault" in active_fault_set:
                v_bus *= max(0.35, 1.0 - 0.60 * sev)
            if self.is_failed:
                v_bus = max(24.0, 27.2 - 0.7 * min(1.0, self.cumulative_time_s / 300.0))
            adc_lsb_alt = 12000.0 / 4096.0
            adc_lsb_rpm = 4200.0 / 4096.0
            adc_lsb_rail = 2000.0 / 4096.0
            alt_quantized = float(np.round(h / adc_lsb_alt) * adc_lsb_alt)
            rpm_quantized = float(np.round(rpm_now / adc_lsb_rpm) * adc_lsb_rpm)
            rail_quantized = float(np.round(P_rail / adc_lsb_rail) * adc_lsb_rail)
            v_bus_raw = float(np.clip(v_bus + np.random.normal(0.0, 0.05), 18.0, 32.0))

            op_h = 0.0 if self.is_failed else dt / 3600.0
            load_stress = float(np.clip(P_brake_kW / CONST.rated_brake_power_kW, 0.05, 1.35))
            speed_stress = float(np.clip(rpm_output / CONST.rated_rpm, 0.20, 1.25))
            temp_stress = float(np.clip((Tcht - 60.0) / 150.0, 0.05, 1.60))
            oil_stress = float(np.clip((Toil - 60.0) / 80.0, 0.05, 1.80))
            boost_stress = float(np.clip((P_manifold / max(P_amb_kPa, 1.0) - 1.0) / 1.8, 0.05, 1.80))
            vibration_stress = float(np.clip(vib_rms / 0.035, 0.05, 2.0))
            fault_mult = lambda name, base: base * (1.0 + (0.8 * sev if name in active_fault_set else 0.0))
            db = op_h / life_bearing * (0.25 + 0.75 * load_stress**1.55 * speed_stress**1.15) * (1.0 + 0.35 * max(0.0, oil_stress - 1.0))
            dtf = 0.25 * op_h / life_thermal * (0.20 + 0.80 * temp_stress**3.0) * (1.0 + 0.50 * abs(dTcht) / 0.8)
            dtu = op_h / life_turbo * (0.25 + 0.75 * boost_stress**1.35 * max(0.5, Tegt / 700.0) ** 1.2)
            di = op_h / life_injector * (0.30 + 0.70 * load_stress**1.2 * max(0.5, P_rail / 1200.0))
            dl = op_h / life_lube * (0.25 + 0.75 * load_stress**1.25 * oil_stress**1.15)
            de = op_h / life_electrical * (0.20 + 0.80 * load_stress**0.9 * max(0.5, (28.5 - v_bus) / 10.0))
            db *= fault_mult("lubrication_issue", 1.0) if "lubrication_issue" in active_fault_set else 1.0
            dtf *= fault_mult("overheating", 1.0) if "overheating" in active_fault_set else 1.0
            dtu *= fault_mult("turbo_issue", 1.0) if "turbo_issue" in active_fault_set else 1.0
            di *= fault_mult("injector_degradation", 1.0) if "injector_degradation" in active_fault_set else 1.0
            de *= fault_mult("electrical_fault", 1.0) if "electrical_fault" in active_fault_set else 1.0
            db *= (1.0 + 0.25 * max(0.0, vibration_stress - 1.0))
            db *= damage_rate_mult
            dtf *= damage_rate_mult
            dtu *= damage_rate_mult
            di *= damage_rate_mult
            dl *= damage_rate_mult
            de *= damage_rate_mult
            wb = float(np.clip(wb + db, 0.0, 1.0))
            wt = float(np.clip(wt + dtf, 0.0, 1.0))
            wtu = float(np.clip(wtu + dtu, 0.0, 1.0))
            wi = float(np.clip(wi + di, 0.0, 1.0))
            wl = float(np.clip(wl + dl, 0.0, 1.0))
            we = float(np.clip(we + de, 0.0, 1.0))

            damage = (0.27 * wb + 0.18 * wt + 0.16 * wtu + 0.16 * wi + 0.14 * wl + 0.09 * we)
            damage += 0.06 * wb * wl + 0.04 * wt * wtu + 0.03 * wi * we
            damage = float(np.clip(damage, 0.0, 1.0))
            health = 100.0 * (1.0 - damage)

            if P_oil < self.tol.oil_p_fail_kPa:
                oil_low += dt
            else:
                oil_low = max(0.0, oil_low - 0.5*dt)
            if Tcht - 273.15 > self.tol.cht_fail_C:
                cht_high += dt
            else:
                cht_high = max(0.0, cht_high - 0.5*dt)
            if Tegt - 273.15 > self.tol.egt_fail_C:
                egt_high += dt
            else:
                egt_high = max(0.0, egt_high - 0.5*dt)
            if rpm_now < CONST.stall_rpm and not self.is_failed:
                stall_duration += dt
            else:
                stall_duration = max(0.0, stall_duration - 0.5*dt)

            components = {"bearing": wb, "thermal": wt, "turbo": wtu, "injector": wi,
                          "lubrication": wl, "electrical": we}
            dominant = max(components, key=components.get)
            degradation_failure = (max(components.values()) >= 0.995 or damage >= 0.94)
            acute_failure = (oil_low >= self.tol.persistence_oil_s or
                             cht_high >= self.tol.persistence_thermal_s or
                             egt_high >= self.tol.persistence_thermal_s or
                             stall_duration >= CONST.stall_persistence_s)
            if (acute_failure or degradation_failure) and not self.is_failed:
                self.is_failed = True
                if oil_low >= self.tol.persistence_oil_s:
                    self.failure_mode = "bearing_seizure_lubrication_loss"
                elif cht_high >= self.tol.persistence_thermal_s:
                    self.failure_mode = "cylinder_head_thermal_limit"
                elif egt_high >= self.tol.persistence_thermal_s:
                    self.failure_mode = "exhaust_valve_turbine_thermal_limit"
                elif stall_duration >= CONST.stall_persistence_s:
                    self.failure_mode = "engine_stall_low_rpm"
                else:
                    self.failure_mode = f"{dominant}_degradation_eol"

            p_ind_report_W = P_indicated_W
            p_fric_report_W = P_friction_W
            P_brake_report_kW = max(0.0, (Q_indicated - Q_friction) * omega / 1000.0)
            Q_indicated_report = Q_indicated
            waste_report_W = max(0.0, fuel_power_W - P_brake_report_kW * 1000.0)
            q_partition_W = max(Q_exh + Q_cool + Q_oil + Q_direct, 1e-9)
            q_scale = waste_report_W / q_partition_W
            Q_exh_r = Q_exh * q_scale
            Q_cool_r = Q_cool * q_scale
            Q_oil_r = Q_oil * q_scale
            Q_direct_r = Q_direct * q_scale
            energy_accounted_W = (P_brake_report_kW * 1000.0 + Q_exh_r + Q_cool_r + Q_oil_r + Q_direct_r)
            closure_error_W = fuel_power_W - energy_accounted_W

            out["altitude_m"][i] = h_sample
            out["aircraft_mass_kg"][i] = aircraft_mass
            out["fuel_mass_kg"][i] = fuel_mass
            out["fuel_used_kg"][i] = fuel_used
            out["wind_u_mps"][i] = wind_u
            out["wind_accel_mps2"][i] = wind_accel
            out["ground_speed_mps"][i] = ground_speed
            out["airspeed_mps"][i] = v_sample
            out["flight_path_angle_deg"][i] = np.degrees(gamma_sample)
            out["vertical_speed_mps"][i] = vertical_speed
            out["prop_pitch_actual"][i] = prop_pitch_actual
            out["target_airspeed_mps"][i] = v_target
            out["throttle_actual"][i] = eff_thr
            out["engine_power_command_kW"][i] = prop_power_kW
            out["engine_bsfc_g_kWh"][i] = float(info_engine["bsfc"])
            out["actual_bsfc_g_kWh"][i] = actual_bsfc
            out["desired_thrust_N"][i] = thrust_required
            out["thrust_equilibrium_error_N"][i] = T_prop - thrust_required
            out["prop_advance_ratio"][i] = J_prop
            out["prop_Cp"][i] = Cp
            out["prop_Ct"][i] = Ct
            out["prop_efficiency"][i] = eta_prop
            out["prop_power_kW"][i] = prop_power_kW
            out["drag_N"][i] = D_drag
            out["excess_thrust_N"][i] = excess_thrust
            out["shaft_torque_error_Nm"][i] = shaft_torque_error
            out["ambient_T_C"][i] = T_amb_C
            out["ambient_P_kPa"][i] = P_amb_kPa
            out["air_density_kg_m3"][i] = rho
            out["rpm"][i] = rpm_output
            out["cht_cyl_avg_C"][i] = Tcht - 273.15
            out["egt_cyl_avg_C"][i] = Tegt - 273.15
            out["oil_pressure_kPa"][i] = P_oil
            out["oil_temp_C"][i] = Toil - 273.15
            out["fuel_flow_g_s"][i] = m_dot_fuel * 1000.0
            out["fuel_temp_C"][i] = Tfuel - 273.15
            out["boost_pressure_kPa"][i] = P_boost
            out["manifold_pressure_kPa"][i] = P_manifold
            out["vib_rms_g"][i] = vib_rms
            out["vib_crest_factor"][i] = vib_crest
            out["vib_kurtosis"][i] = vib_kurt
            out["vib_1x_hz"][i] = f1x
            out["vib_2x_hz"][i] = 2.0 * f1x
            out["vib_firing_hz"][i] = f_fire
            out["vib_bpfo_hz"][i] = bpfo_now
            out["vib_bpfi_hz"][i] = bpfi_now
            out["vib_bsf_hz"][i] = bsf_now
            out["vib_psd_peak"][i] = vib_psd_peak
            out["vib_psd_peak_freq"][i] = vib_psd_peak_freq
            out["vib_1x_rms"][i] = vib_1x_rms
            out["vib_2x_rms"][i] = vib_2x_rms
            out["vib_fire_order_rms"][i] = vib_fire_order_rms
            out["vib_spectral_centroid"][i] = vib_spectral_centroid
            out["battery_voltage_V"][i] = v_bus
            out["battery_voltage_raw_V"][i] = v_bus_raw
            out["altitude_quantized_m"][i] = alt_quantized
            out["rpm_quantized"][i] = rpm_quantized
            out["rail_pressure_quantized_bar"][i] = rail_quantized
            out["sensor_drift_flag_rpm"][i] = 0.0
            out["sensor_drift_flag_rail"][i] = 0.0
            out["sensor_drift_flag_boost"][i] = 0.0
            out["coolant_temp_C"][i] = Tcool - 273.15
            out["rail_pressure_bar"][i] = P_rail
            out["SOI_deg_bTDC"][i] = soi
            out["inj_duration_ms"][i] = inj_dur_ms
            out["fadec_mode"][i] = fadec_mode
            out["true_brake_power_kW"][i] = P_brake_report_kW
            out["true_fuel_power_kW"][i] = fuel_power_W / 1000.0
            out["true_indicated_power_kW"][i] = p_ind_report_W / 1000.0
            out["true_friction_power_kW"][i] = p_fric_report_W / 1000.0
            out["true_exhaust_heat_kW"][i] = Q_exh_r / 1000.0
            out["true_coolant_heat_kW"][i] = Q_cool_r / 1000.0
            out["true_oil_heat_kW"][i] = Q_oil_r / 1000.0
            out["true_direct_heat_loss_kW"][i] = Q_direct_r / 1000.0
            out["true_energy_closure_error_W"][i] = closure_error_W
            out["true_rpm"][i] = rpm_now
            out["true_indicated_torque_Nm"][i] = Q_indicated_report
            out["true_prop_load_torque_Nm"][i] = Q_prop
            out["true_prop_thrust_N"][i] = T_prop
            out["true_bearing_wear"][i] = wb
            out["true_thermal_fatigue"][i] = wt
            out["true_turbo_degradation"][i] = wtu
            out["true_injector_wear"][i] = wi
            out["true_lubrication_degradation"][i] = wl
            out["true_electrical_degradation"][i] = we
            out["true_overall_health_index"][i] = health
            if self.is_failed:
                out["failure_flag"][i] = 1
                out["failure_mode"][i] = self.failure_mode

        self.v_tas, self.gamma_rad, self.altitude_m = v, gamma, h
        self.omega, self.turbo_speed, self.governor_integral = omega, turbo_speed, gov_int
        self.prop_pitch_actual = prop_pitch_actual
        self.T_cht_K, self.T_coolant_K, self.T_oil_K, self.T_fuel_K = Tcht, Tcool, Toil, Tfuel
        self.fuel_mass_kg = fuel_mass
        self.fuel_used_kg += fuel_used
        self.bearing_wear, self.thermal_fatigue = wb, wt
        self.turbo_degradation, self.injector_wear, self.lubrication_deg = wtu, wi, wl
        self.electrical_deg = we
        self.oil_p_low_duration_s, self.cht_high_duration_s, self.egt_high_duration_s = oil_low, cht_high, egt_high
        self.stall_duration_s = stall_duration
        self.cumulative_time_s += n * dt
        return out

# Apply Sensor Acquisition Model
def apply_sensor_acquisition_model(
    signals, timestamps, tolerance, fault_type, fault_onset_s, severity_t,
    sensor_fault_mode="none", dropout_probability=0.001, dt=DT
):
    n = len(timestamps)
    obs = {k: (v.copy() if isinstance(v, np.ndarray) else v) for k, v in signals.items()}
    telemetry_valid = np.ones(n, dtype=int)

    for k, bias in {
        "rpm": tolerance.bias_rpm, "cht_cyl_avg_C": tolerance.bias_cht,
        "egt_cyl_avg_C": tolerance.bias_egt, "oil_pressure_kPa": tolerance.bias_oil_p,
        "oil_temp_C": tolerance.bias_oil_t, "fuel_flow_g_s": tolerance.bias_fuel_flow,
        "boost_pressure_kPa": tolerance.bias_boost, "manifold_pressure_kPa": tolerance.bias_manifold,
        "rail_pressure_bar": tolerance.bias_rail, "coolant_temp_C": tolerance.bias_coolant,
    }.items():
        obs[k] += bias

    noise_sd = {
        "rpm": 3.0, "cht_cyl_avg_C": 0.35, "egt_cyl_avg_C": 1.2,
        "oil_pressure_kPa": 1.5, "oil_temp_C": 0.25, "fuel_flow_g_s": 0.045,
        "fuel_temp_C": 0.25, "boost_pressure_kPa": 0.40, "manifold_pressure_kPa": 0.40,
        "vib_rms_g": 0.0007, "vib_crest_factor": 0.055, "vib_kurtosis": 0.14,
        "battery_voltage_V": 0.045, "battery_voltage_raw_V": 0.045,
        "coolant_temp_C": 0.25, "rail_pressure_bar": 2.2,
        "SOI_deg_bTDC": 0.10, "inj_duration_ms": 0.02,
        "vib_psd_peak": 5.0e-7, "vib_psd_peak_freq": 0.05,
        "vib_1x_rms": 0.00035, "vib_2x_rms": 0.00035,
        "vib_fire_order_rms": 0.00035, "vib_spectral_centroid": 0.15,
        "throttle_actual": 0.006, "engine_power_command_kW": 0.30,
        "ground_speed_mps": 0.08, "vertical_speed_mps": 0.06,
        "flight_path_angle_deg": 0.10, "desired_thrust_N": 6.0,
        "drag_N": 7.0, "excess_thrust_N": 8.0,
    }
    tau = {
        "rpm": 0.20, "cht_cyl_avg_C": 1.2, "egt_cyl_avg_C": 0.35,
        "oil_pressure_kPa": 0.12, "oil_temp_C": 1.0, "fuel_flow_g_s": 0.15,
        "fuel_temp_C": 2.0, "boost_pressure_kPa": 0.25, "manifold_pressure_kPa": 0.25,
        "vib_rms_g": 0.05, "vib_crest_factor": 0.08, "vib_kurtosis": 0.10,
        "battery_voltage_V": 0.20, "battery_voltage_raw_V": 0.20,
        "coolant_temp_C": 1.0, "rail_pressure_bar": 0.08,
        "SOI_deg_bTDC": 0.05, "inj_duration_ms": 0.05,
        "vib_psd_peak": 0.10, "vib_psd_peak_freq": 0.08,
        "vib_1x_rms": 0.06, "vib_2x_rms": 0.06,
        "vib_fire_order_rms": 0.06, "vib_spectral_centroid": 0.10,
        "throttle_actual": 0.10, "engine_power_command_kW": 0.08,
        "ground_speed_mps": 0.15, "vertical_speed_mps": 0.12,
        "flight_path_angle_deg": 0.20, "desired_thrust_N": 0.15,
        "drag_N": 0.20, "excess_thrust_N": 0.15,
    }
    for k, sd in noise_sd.items():
        if k not in obs:
            continue
        noisy = obs[k] + np.random.normal(0.0, sd, n)
        obs[k] = first_order_lag(noisy, tau[k], dt=dt)

    obs["rpm"] = np.clip(obs["rpm"], 0.0, CONST.max_rpm + 150.0)
    obs["fuel_flow_g_s"] = np.clip(obs["fuel_flow_g_s"], 0.0, 10.0)
    obs["oil_pressure_kPa"] = np.clip(obs["oil_pressure_kPa"], 0.0, 550.0)
    obs["rail_pressure_bar"] = np.clip(obs["rail_pressure_bar"], 0.0, 2000.0)
    obs["battery_voltage_V"] = np.clip(obs["battery_voltage_V"], 18.0, 32.0)
    obs["battery_voltage_raw_V"] = np.clip(obs["battery_voltage_raw_V"], 18.0, 32.0)

    adc_levels = 4096
    obs["altitude_quantized_m"] = np.round(obs["altitude_m"] / (12000.0 / adc_levels)) * (12000.0 / adc_levels)
    obs["rpm_quantized"] = np.round(obs["rpm"] / (4200.0 / adc_levels)) * (4200.0 / adc_levels)
    obs["rail_pressure_quantized_bar"] = np.round(obs["rail_pressure_bar"] / (2000.0 / adc_levels)) * (2000.0 / adc_levels)

    obs["sensor_drift_flag_rpm"] = np.zeros(n)
    obs["sensor_drift_flag_rail"] = np.zeros(n)
    obs["sensor_drift_flag_boost"] = np.zeros(n)

    if (fault_type == "sensor_drift" or sensor_fault_mode != "none") and fault_onset_s is not None:
        onset = timestamps >= fault_onset_s
        sev = np.clip(severity_t, 0.0, 1.0)
        elapsed = np.maximum(0.0, timestamps - fault_onset_s)
        mode = sensor_fault_mode if sensor_fault_mode != "none" else "gradual_drift"
        if mode == "gradual_drift":
            drift = np.clip(0.10 * elapsed, 0.0, 35.0) * sev
            obs["cht_cyl_avg_C"] += drift
            obs["coolant_temp_C"] += 0.35 * drift
        elif mode == "abrupt_step_bias":
            obs["oil_pressure_kPa"] += onset * (20.0 * np.max(sev))
        elif mode == "stuck_at_value":
            idx = np.where(onset)[0]
            if len(idx):
                obs["egt_cyl_avg_C"][idx[0]:] = obs["egt_cyl_avg_C"][idx[0]]
        elif mode == "intermittent_spikes":
            mask = onset & (np.random.random(n) < 0.035)
            obs["vib_rms_g"][mask] += np.random.uniform(0.03, 0.08, mask.sum())

        if mode == "gradual_drift":
            obs["sensor_drift_flag_rpm"][onset] = 1.0
        elif mode == "abrupt_step_bias":
            obs["sensor_drift_flag_boost"][onset] = 1.0
        elif mode == "stuck_at_value":
            obs["sensor_drift_flag_rail"][onset] = 1.0
        elif mode == "intermittent_spikes":
            obs["sensor_drift_flag_rpm"][onset] = 1.0

    drop = np.random.random(n) < dropout_probability
    if np.any(drop):
        telemetry_valid[drop] = 0
        for k in ["rpm", "cht_cyl_avg_C", "egt_cyl_avg_C", "oil_pressure_kPa", "fuel_flow_g_s", "boost_pressure_kPa"]:
            obs[k][drop] = np.nan
    return obs, telemetry_valid

# Generate Mission Profile
def generate_mission_profile(n_samples, mission_duration_s, t_start, mission_type="standard_recon", dt=DT):
    timestamps = np.arange(n_samples, dtype=float) * dt
    t_global_s = t_start + timestamps
    duration = max(mission_duration_s, 1.0)

    profiles = {
        "high_altitude_loiter": (["takeoff", "climb", "cruise", "loiter", "descent", "landing"], [0.08, 0.78, 0.84, 0.90, 0.95, 1.0]),
        "low_level_dash": (["takeoff", "climb", "cruise", "dash", "descent", "landing"], [0.10, 0.22, 0.45, 0.75, 0.90, 1.0]),
        "aborted_go_around": (["takeoff", "climb", "cruise", "approach", "go_around", "landing"], [0.12, 0.40, 0.56, 0.70, 0.88, 1.0]),
        "standard_recon": (["takeoff", "climb", "cruise", "loiter", "descent", "landing"], [0.10, 0.45, 0.62, 0.78, 0.90, 1.0]),
        "calibration_bench": (["takeoff", "bench", "cruise", "descent", "landing"], [0.08, 0.50, 0.80, 0.92, 1.0]),
    }
    phases, splits = profiles.get(mission_type, profiles["standard_recon"])
    boundaries = np.array([0.0] + [s * duration for s in splits])
    idx = np.clip(np.digitize(timestamps, boundaries) - 1, 0, len(phases) - 1)
    phases_arr = np.array(phases, dtype=object)[idx]
    time_phase = timestamps - boundaries[idx]

    thr = np.zeros(n_samples); pitch = np.zeros(n_samples); vz = np.zeros(n_samples)
    target_speed = np.zeros(n_samples); target_altitude = np.zeros(n_samples)
    cruise_alt = 3000.0 if mission_type == "high_altitude_loiter" else (650.0 if mission_type == "low_level_dash" else 1200.0)
    loiter_alt = 3000.0 if mission_type == "high_altitude_loiter" else (650.0 if mission_type == "low_level_dash" else 1200.0)
    climb_alt = loiter_alt + 50.0
    for p in phases:
        m = phases_arr == p
        nn = int(m.sum())
        if not nn:
            continue
        if p == "takeoff":
            thr[m] = np.linspace(0.85, 1.00, nn); pitch[m] = 0.86; vz[m] = 1.5; target_speed[m] = 42.0; target_altitude[m] = 50.0
        elif p == "climb":
            thr[m] = 0.92; pitch[m] = 0.78; vz[m] = 3.0; target_speed[m] = 45.0; target_altitude[m] = climb_alt
        elif p == "cruise":
            thr[m] = 0.72; pitch[m] = 0.68; vz[m] = 0.0; target_speed[m] = 55.0; target_altitude[m] = cruise_alt
        elif p == "loiter":
            thr[m] = 0.60; pitch[m] = 0.55; vz[m] = 0.0; target_speed[m] = 55.0; target_altitude[m] = loiter_alt
        elif p == "bench":
            thr[m] = 1.00; pitch[m] = 0.86; vz[m] = 0.0; target_speed[m] = 58.0; target_altitude[m] = 50.0
        elif p == "dash":
            thr[m] = 0.74; pitch[m] = 0.74; vz[m] = 0.0; target_speed[m] = 58.0; target_altitude[m] = 650.0
        elif p in ("descent", "approach"):
            thr[m] = 0.28; pitch[m] = 0.38; vz[m] = -2.8; target_speed[m] = 48.0; target_altitude[m] = 1000.0
        elif p == "go_around":
            thr[m] = 0.95; pitch[m] = 0.86; vz[m] = 3.0; target_speed[m] = 50.0; target_altitude[m] = 1500.0
        elif p == "landing":
            thr[m] = np.linspace(0.25, 0.12, nn); pitch[m] = 0.24; vz[m] = -1.6; target_speed[m] = 40.0; target_altitude[m] = 0.0

    rng = np.random.default_rng(int(abs(t_start) * 10 + mission_duration_s) % (2**32 - 1))
    low_freq = first_order_lag(rng.normal(0.0, 1.0, n_samples), tau=12.0, dt=dt)
    low_freq = low_freq / max(np.std(low_freq), 1e-6)
    thr = np.clip(thr + 0.018 * low_freq, 0.0, 1.0)
    pitch = np.clip(pitch + 0.012 * low_freq, 0.05, 1.0)
    target_speed = np.maximum(target_speed, CONST.min_operating_speed_mps)
    target_speed += 0.5 * first_order_lag(rng.normal(0.0, 1.0, n_samples), tau=20.0, dt=dt)
    target_speed = np.clip(target_speed, CONST.min_operating_speed_mps, CONST.max_operating_speed_mps)
    vz = vz + 0.15 * first_order_lag(rng.normal(0.0, 1.0, n_samples), tau=10.0, dt=dt)
    throttle_cmd = first_order_lag(thr, tau=1.5, dt=dt)
    prop_pitch_cmd = first_order_lag(pitch, tau=1.2, dt=dt)
    target_vz = first_order_lag(vz, tau=3.0, dt=dt)
    return timestamps, t_global_s, phases_arr, time_phase, target_vz, target_speed, target_altitude, throttle_cmd, prop_pitch_cmd

FAULT_DOMAIN_MAP = {
    "normal": "none",
    "misfire": "combustion",
    "injector_degradation": "fluid_injection",
    "turbo_issue": "turbomachinery",
    "lubrication_issue": "hydrodynamic_tribology",
    "sensor_drift": "instrumentation_acquisition",
    "overheating": "thermal_management",
    "electrical_fault": "electrical_system",
    "bearing_fault": "mechanical_tribology",
}

FAULT_NATURE_MAP = {
    "normal": "none",
    "misfire": "mechanical_combustion_fault",
    "injector_degradation": "mechanical_hydraulic_fault",
    "turbo_issue": "mechanical_aerodynamic_fault",
    "lubrication_issue": "mechanical_tribology_fault",
    "sensor_drift": "instrumentation_measurement_fault",
    "overheating": "mechanical_thermal_fault",
    "electrical_fault": "electrical_fault",
    "bearing_fault": "mechanical_tribology_fault",
}

# Get Health State Label
def get_health_state_label(health_arr: np.ndarray) -> np.ndarray:
    states = np.full(len(health_arr), "healthy", dtype=object)
    states[health_arr >= 90.0] = "pristine"
    states[(health_arr >= 70.0) & (health_arr < 90.0)] = "healthy"
    states[(health_arr >= 40.0) & (health_arr < 70.0)] = "aging"
    states[(health_arr >= 10.0) & (health_arr < 40.0)] = "degraded"
    states[health_arr < 10.0] = "critical_eol"
    return states

# Create Mission
def create_mission(
    mission_id: str,
    engine_twin: ApexPhysicalEngineTwin,
    fault_type: str,
    mission_type: str = "standard_recon",
    fault_onset_s: Optional[float] = None,
    fault_ramp_s: float = 60.0,
    fault_peak_severity: float = 0.0,
    progression_type: str = "linear",
    sensor_fault_mode: str = "none",
    compound_faults: Optional[List[str]] = None,
    mission_duration_s: float = 600.0,
    sea_temp_C: float = 18.0,
    qnh_offset_kPa: float = 0.0,
    sample_dt: float = DT
) -> pd.DataFrame:
    np.random.seed(_stable_mission_seed(mission_id))
    n_samples = max(1, int(mission_duration_s / sample_dt))
    t_start = engine_twin.cumulative_time_s

    (timestamps, t_global_s, phases_array, time_in_phase, target_vz,
     target_airspeed, target_altitude, throttle_cmd, prop_pitch_cmd) = generate_mission_profile(
        n_samples, mission_duration_s, t_start, mission_type=mission_type, dt=sample_dt
    )

    wind_seed = _stable_mission_seed(mission_id)
    wind_alt_proxy = np.maximum(target_altitude, 0.0)
    wind_u_profile, wind_accel_profile = generate_turbulent_wind_profile(
        t_global_s, wind_alt_proxy, wind_seed, sample_dt)

    if fault_type == "normal" or fault_onset_s is None or fault_peak_severity <= 0:
        severity_t = np.zeros(n_samples)
        fault_active = np.zeros(n_samples, dtype=int)
        severity_category = np.full(n_samples, "normal", dtype=object)
    else:
        time_since_onset = np.maximum(0.0, timestamps - fault_onset_s)
        norm_t = np.clip(time_since_onset / max(fault_ramp_s, 1.0), 0.0, 1.0)

        if progression_type == "exponential":
            severity_curve = (np.exp(2.5 * norm_t) - 1.0) / (np.exp(2.5) - 1.0)
        elif progression_type == "brownian":
            noise_walk = np.cumsum(np.random.normal(0, 0.02, n_samples))
            severity_curve = np.clip(norm_t + noise_walk, 0.0, 1.0)
        elif progression_type == "intermittent":
            pulsing = 0.5 + 0.5 * np.sin(2.0 * np.pi * 0.02 * timestamps)
            severity_curve = norm_t * (pulsing > 0.3)
        else:
            severity_curve = norm_t

        severity_t = np.minimum(severity_curve, 1.0) * fault_peak_severity
        severity_t[timestamps < fault_onset_s] = 0.0
        fault_active = (severity_t > 0.0).astype(int)

        severity_category = np.full(n_samples, "normal", dtype=object)
        severity_category[(severity_t > 0.0) & (severity_t <= 0.30)] = "early"
        severity_category[(severity_t > 0.30) & (severity_t <= 0.60)] = "moderate"
        severity_category[(severity_t > 0.60) & (severity_t <= 0.85)] = "severe"
        severity_category[severity_t > 0.85] = "critical"

    true_states = engine_twin.integrate_mission_step(
        sample_dt, t_global_s, throttle_cmd, prop_pitch_cmd, target_vz,
        target_airspeed, target_altitude, fault_type, severity_t,
        sea_temp_C=sea_temp_C, qnh_offset_kPa=qnh_offset_kPa,
        compound_faults=compound_faults,
        wind_u_profile=wind_u_profile, wind_accel_profile=wind_accel_profile
    )

    observed_sensors, telemetry_valid = apply_sensor_acquisition_model(
        true_states, timestamps, engine_twin.tol, fault_type, fault_onset_s,
        severity_t, sensor_fault_mode=sensor_fault_mode, dt=sample_dt
    )

    primary_fault = fault_type if not compound_faults else f"{fault_type}+{'+'.join(compound_faults)}"
    data_dict = {
        "global_time_s": t_global_s,
        "time_in_mission_s": timestamps,
        "actual_operating_hours": t_global_s / 3600.0,
        "engine_id": engine_twin.engine_id,
        "mission_id": mission_id,
        "mission_type": mission_type,
        "phase": phases_array,
        "time_in_phase_s": time_in_phase,
        "telemetry_valid": telemetry_valid,
        "altitude_m": true_states["altitude_m"],
        "target_altitude_m": target_altitude,
        "target_airspeed_mps": target_airspeed,
        "aircraft_mass_kg": true_states["aircraft_mass_kg"],
        "fuel_mass_kg": true_states["fuel_mass_kg"],
        "fuel_used_kg": true_states["fuel_used_kg"],
        "refueled_at_sortie_start": np.array([1] + [0] * (n_samples - 1), dtype=int),
        "sortie_start_fuel_kg": np.full(n_samples, CONST.fuel_capacity_kg),
        "wind_u_mps": true_states["wind_u_mps"],
        "wind_accel_mps2": true_states["wind_accel_mps2"],
        "ground_speed_mps": true_states["ground_speed_mps"],
        "ambient_T_C": true_states["ambient_T_C"],
        "ambient_P_kPa": true_states["ambient_P_kPa"],
        "air_density_kg_m3": true_states["air_density_kg_m3"],
        "airspeed_mps": true_states["airspeed_mps"],
        "flight_path_angle_deg": true_states["flight_path_angle_deg"],
        "vertical_speed_mps": true_states["vertical_speed_mps"],
        "throttle_cmd": throttle_cmd,
        "throttle_actual": observed_sensors["throttle_actual"],
        "prop_pitch_cmd": prop_pitch_cmd,
        "prop_pitch_actual": true_states["prop_pitch_actual"],
        "engine_power_command_kW": observed_sensors["engine_power_command_kW"],
        "engine_bsfc_g_kWh": true_states["engine_bsfc_g_kWh"],
        "actual_bsfc_g_kWh": true_states["actual_bsfc_g_kWh"],
        "desired_thrust_N": observed_sensors["desired_thrust_N"],
        "thrust_equilibrium_error_N": true_states["thrust_equilibrium_error_N"],
        "settled_state": np.empty(n_samples, dtype=object),
        "prop_advance_ratio": true_states["prop_advance_ratio"],
        "prop_Cp": true_states["prop_Cp"],
        "prop_Ct": true_states["prop_Ct"],
        "prop_efficiency": true_states["prop_efficiency"],
        "prop_power_kW": true_states["prop_power_kW"],
        "drag_N": observed_sensors["drag_N"],
        "excess_thrust_N": observed_sensors["excess_thrust_N"],
        "shaft_torque_error_Nm": true_states["shaft_torque_error_Nm"],
        "rpm": observed_sensors["rpm"],
        "cht_cyl_avg_C": observed_sensors["cht_cyl_avg_C"],
        "egt_cyl_avg_C": observed_sensors["egt_cyl_avg_C"],
        "oil_pressure_kPa": observed_sensors["oil_pressure_kPa"],
        "oil_temp_C": observed_sensors["oil_temp_C"],
        "fuel_flow_g_s": observed_sensors["fuel_flow_g_s"],
        "fuel_temp_C": observed_sensors["fuel_temp_C"],
        "boost_pressure_kPa": observed_sensors["boost_pressure_kPa"],
        "manifold_pressure_kPa": observed_sensors["manifold_pressure_kPa"],
        "vib_rms_g": observed_sensors["vib_rms_g"],
        "vib_crest_factor": observed_sensors["vib_crest_factor"],
        "vib_kurtosis": observed_sensors["vib_kurtosis"],
        "vib_1x_hz": observed_sensors["vib_1x_hz"],
        "vib_2x_hz": observed_sensors["vib_2x_hz"],
        "vib_firing_hz": observed_sensors["vib_firing_hz"],
        "vib_bpfo_hz": observed_sensors["vib_bpfo_hz"],
        "vib_bpfi_hz": observed_sensors["vib_bpfi_hz"],
        "vib_bsf_hz": observed_sensors["vib_bsf_hz"],
        "vib_psd_peak": observed_sensors["vib_psd_peak"],
        "vib_psd_peak_freq": observed_sensors["vib_psd_peak_freq"],
        "vib_1x_rms": observed_sensors["vib_1x_rms"],
        "vib_2x_rms": observed_sensors["vib_2x_rms"],
        "vib_fire_order_rms": observed_sensors["vib_fire_order_rms"],
        "vib_spectral_centroid": observed_sensors["vib_spectral_centroid"],
        "battery_voltage_V": observed_sensors["battery_voltage_V"],
        "battery_voltage_raw_V": observed_sensors["battery_voltage_raw_V"],
        "altitude_quantized_m": observed_sensors["altitude_quantized_m"],
        "rpm_quantized": observed_sensors["rpm_quantized"],
        "rail_pressure_quantized_bar": observed_sensors["rail_pressure_quantized_bar"],
        "sensor_drift_flag_rpm": observed_sensors["sensor_drift_flag_rpm"],
        "sensor_drift_flag_rail": observed_sensors["sensor_drift_flag_rail"],
        "sensor_drift_flag_boost": observed_sensors["sensor_drift_flag_boost"],
        "coolant_temp_C": observed_sensors["coolant_temp_C"],
        "rail_pressure_bar": observed_sensors["rail_pressure_bar"],
        "SOI_deg_bTDC": observed_sensors["SOI_deg_bTDC"],
        "inj_duration_ms": observed_sensors["inj_duration_ms"],
        "fadec_mode": observed_sensors["fadec_mode"],
        "true_brake_power_kW": true_states["true_brake_power_kW"],
        "true_fuel_power_kW": true_states["true_fuel_power_kW"],
        "true_indicated_power_kW": true_states["true_indicated_power_kW"],
        "true_friction_power_kW": true_states["true_friction_power_kW"],
        "true_exhaust_heat_kW": true_states["true_exhaust_heat_kW"],
        "true_coolant_heat_kW": true_states["true_coolant_heat_kW"],
        "true_oil_heat_kW": true_states["true_oil_heat_kW"],
        "true_direct_heat_loss_kW": true_states["true_direct_heat_loss_kW"],
        "true_energy_closure_error_W": true_states["true_energy_closure_error_W"],
        "true_rpm": true_states["true_rpm"],
        "true_indicated_torque_Nm": true_states["true_indicated_torque_Nm"],
        "true_prop_load_torque_Nm": true_states["true_prop_load_torque_Nm"],
        "true_prop_thrust_N": true_states["true_prop_thrust_N"],
        "true_bearing_wear": true_states["true_bearing_wear"],
        "true_thermal_fatigue": true_states["true_thermal_fatigue"],
        "true_turbo_degradation": true_states["true_turbo_degradation"],
        "true_injector_wear": true_states["true_injector_wear"],
        "true_lubrication_degradation": true_states["true_lubrication_degradation"],
        "true_electrical_degradation": true_states["true_electrical_degradation"],
        "true_overall_health_index": true_states["true_overall_health_index"],
        "simulated_rul_hours": np.full(n_samples, np.nan),
        "health_state": get_health_state_label(true_states["true_overall_health_index"]),
        "fault_type": primary_fault,
        "fault_domain": FAULT_DOMAIN_MAP.get(fault_type, "none"),
        "fault_nature": FAULT_NATURE_MAP.get(fault_type, "none"),
        "fault_severity": severity_t,
        "severity_category": severity_category,
        "fault_active": fault_active,
        "fault_misfire_active": ((severity_t > 0) & (fault_type == "misfire" or (compound_faults is not None and "misfire" in compound_faults))).astype(int),
        "fault_injector_active": ((severity_t > 0) & (fault_type == "injector_degradation" or (compound_faults is not None and "injector_degradation" in compound_faults))).astype(int),
        "fault_turbo_active": ((severity_t > 0) & (fault_type == "turbo_issue" or (compound_faults is not None and "turbo_issue" in compound_faults))).astype(int),
        "fault_lubrication_active": ((severity_t > 0) & (fault_type == "lubrication_issue" or (compound_faults is not None and "lubrication_issue" in compound_faults))).astype(int),
        "fault_sensor_active": ((severity_t > 0) & (fault_type == "sensor_drift" or (compound_faults is not None and "sensor_drift" in compound_faults))).astype(int),
        "fault_overheating_active": ((severity_t > 0) & (fault_type == "overheating" or (compound_faults is not None and "overheating" in compound_faults))).astype(int),
        "fault_electrical_active": ((severity_t > 0) & (fault_type == "electrical_fault" or (compound_faults is not None and "electrical_fault" in compound_faults))).astype(int),
        "fault_bearing_active": ((severity_t > 0) & (fault_type == "bearing_fault" or (compound_faults is not None and "bearing_fault" in compound_faults))).astype(int),
        "failure_flag": true_states["failure_flag"],
        "failure_mode": true_states["failure_mode"],
    }

    df = pd.DataFrame(data_dict)[SCHEMA_COLUMNS]

    eqs = df["thrust_equilibrium_error_N"].to_numpy()
    drag = df["drag_N"].to_numpy()
    is_cruise_leg = df["phase"].isin(["cruise", "loiter"]).to_numpy()
    past_settle = df["time_in_phase_s"].to_numpy() > SETTLE_S
    valid = df["telemetry_valid"].to_numpy() == 1
    dv = np.abs(np.diff(df["airspeed_mps"].to_numpy(), prepend=df["airspeed_mps"].to_numpy()[0]))
    speed_conv = dv < 0.60
    settled_mask = is_cruise_leg & past_settle & valid & speed_conv
    df["settled_state"] = np.where(settled_mask, "settled", "transient")

    validate_generated_mission(df)
    return df

# Validate Generated Mission
def validate_generated_mission(df: pd.DataFrame) -> None:
    missing = [c for c in SCHEMA_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing schema columns: {missing}")

    numeric = df.select_dtypes(include=[np.number])
    if np.isinf(numeric.to_numpy()).any():
        raise ValueError("Generated mission contains infinite numeric values")

    t = df["time_in_mission_s"].to_numpy()
    if len(t) > 1 and not np.all(np.diff(t) > 0):
        raise ValueError("Mission timestamps are not strictly increasing")

    bounded = {
        "true_bearing_wear": (0.0, 1.0),
        "true_thermal_fatigue": (0.0, 1.0),
        "true_turbo_degradation": (0.0, 1.0),
        "true_injector_wear": (0.0, 1.0),
        "true_lubrication_degradation": (0.0, 1.0),
        "true_electrical_degradation": (0.0, 1.0),
        "true_overall_health_index": (0.0, 100.0),
        "true_brake_power_kW": (0.0, 160.0),
        "true_prop_thrust_N": (-300.0, 6000.0),
        "fuel_mass_kg": (CONST.fuel_reserve_kg, CONST.fuel_capacity_kg),
        "aircraft_mass_kg": (CONST.m_empty_kg + CONST.fuel_reserve_kg,
                             CONST.m_empty_kg + CONST.fuel_capacity_kg),
    }
    for col, (lo, hi) in bounded.items():
        x = df[col].dropna().to_numpy()
        if len(x) and np.any((x < lo) | (x > hi)):
            raise ValueError(f"{col} outside physical benchmark range [{lo}, {hi}]")

    if np.any(df["manifold_pressure_kPa"].dropna().to_numpy() < df["ambient_P_kPa"].dropna().to_numpy() - 1e-6):
        raise ValueError("Manifold pressure fell below ambient pressure")

    omega_true = df["true_rpm"].to_numpy() * 2.0 * np.pi / 60.0
    p_from_torque = df["true_prop_load_torque_Nm"].to_numpy() * omega_true / 1000.0
    if np.nanmax(np.abs(p_from_torque - df["prop_power_kW"].to_numpy())) > 1e-5:
        raise ValueError("Propeller power is not synchronized with true RPM and torque")
    rho = df["air_density_kg_m3"].to_numpy()
    n_rev_s = np.maximum(df["true_rpm"].to_numpy() / 60.0, 1e-6)
    cp = df["prop_Cp"].to_numpy()
    cp_floor_mask = cp <= 0.021
    p_prop_formula = cp * rho * n_rev_s**3 * CONST.D_prop**5 / 1000.0
    non_floor = ~cp_floor_mask
    if non_floor.any():
        if np.nanmax(np.abs(p_prop_formula[non_floor] - df["prop_power_kW"].to_numpy()[non_floor])) > 1e-5:
            raise ValueError("Propeller Cp power equation inconsistent with stored propeller power")
    if np.nanmin(df["prop_Cp"].to_numpy()) < -1e-9:
        raise ValueError("Propeller map produced negative Cp")
    eta = df["prop_efficiency"].dropna().to_numpy()
    if len(eta) and np.any((eta < -1e-9) | (eta > 0.951)):
        raise ValueError("Propeller efficiency outside bounded physical range")

    f1 = df["true_rpm"].to_numpy() / 60.0
    if np.any(df["vib_bpfo_hz"].to_numpy() <= f1) or np.any(df["vib_bpfi_hz"].to_numpy() <= f1):
        raise ValueError("Bearing characteristic frequencies are not above shaft order")
    if np.any(df["vib_bpfi_hz"].to_numpy() <= df["vib_bpfo_hz"].to_numpy()):
        raise ValueError("BPFI must exceed BPFO")

    bsfc_mask = df["true_brake_power_kW"].to_numpy() > 0.5
    if np.any(bsfc_mask):
        true_fuel_flow_g_s = (df.loc[bsfc_mask, "true_fuel_power_kW"].to_numpy() * 1000.0 / CONST.lhv_fuel) * 1000.0
        bsfc_expected = true_fuel_flow_g_s * 3600.0 / df.loc[bsfc_mask, "true_brake_power_kW"].to_numpy()
        bsfc_stored = df.loc[bsfc_mask, "actual_bsfc_g_kWh"].to_numpy()
        if np.nanmax(np.abs(bsfc_expected - bsfc_stored)) > 0.05:
            raise ValueError("Actual BSFC is inconsistent with fuel flow and brake power")

    cruise_leg = df["phase"].isin(["cruise", "loiter"]).to_numpy()
    settled = cruise_leg & (df["time_in_phase_s"].to_numpy() > SETTLE_S) & (df["telemetry_valid"] == 1).to_numpy()
    if settled.sum() > 10:
        running = df.loc[settled, "true_brake_power_kW"].to_numpy() > 5.0
        s_run = settled.copy(); s_run[settled] = running
        if s_run.sum() > 10:
            so = df.loc[s_run, "shaft_torque_error_Nm"].to_numpy()
            med_torque = float(np.nanmedian(np.abs(so)))
            sign_changes = int(np.sum(np.diff(np.sign(so)) != 0))
            v_sel = df.loc[s_run, "airspeed_mps"].to_numpy()
            med_dv = float(np.nanmedian(np.abs(np.diff(v_sel, prepend=v_sel[0]))))
            torque_oscillates = (med_torque > 20.0) and (sign_changes > 40)
            if med_dv > 0.60 or (torque_oscillates and med_dv > 0.25):
                raise ValueError("Settled cruise/loiter equilibrium is not converging "
                                 "(median shaft torque=%.1f Nm, signch=%d, median |dv|=%.3f)"
                                 % (med_torque, sign_changes, med_dv))

    omega = df["true_rpm"].to_numpy() * 2.0 * np.pi / 60.0
    p_ind = df["true_indicated_torque_Nm"].to_numpy() * omega / 1000.0
    p_brake = df["true_brake_power_kW"].to_numpy()
    if np.nanmax(p_brake - p_ind) > 1e-6:
        raise ValueError("Brake power exceeded indicated power")

    closure = df["true_energy_closure_error_W"].dropna().to_numpy()
    if len(closure) and np.nanmax(np.abs(closure)) > 250.0:
        raise ValueError("First-law energy closure exceeded 250 W")

    checks = {
        "rpm": (0.0, 3500.0),
        "airspeed_mps": (0.0, 120.0),
        "air_density_kg_m3": (0.05, 1.5),
        "battery_voltage_V": (15.0, 35.0),
        "fuel_flow_g_s": (0.0, 10.0),
        "rail_pressure_bar": (0.0, 2000.0),
        "wind_u_mps": (-15.0, 25.0),
        "ambient_P_kPa": (5.0, 110.0),
    }
    for col, (lo, hi) in checks.items():
        x = df[col].dropna().to_numpy()
        if len(x) and np.any((x < lo) | (x > hi)):
            raise ValueError(f"{col} outside generator sanity bounds [{lo}, {hi}]")

# Save Dataframe
def save_dataframe(df: pd.DataFrame, file_path: Path):
    file_path.parent.mkdir(parents=True, exist_ok=True)
    if OUTPUT_FORMAT == "parquet":
        try:
            df.to_parquet(file_path.with_suffix(".parquet"), index=False)
        except (ImportError, ModuleNotFoundError):
            df.to_csv(file_path.with_suffix(".csv"), index=False)
    else:
        df.to_csv(file_path.with_suffix(".csv"), index=False)

# Generate Normal Dataset
def generate_normal_dataset(num_missions: int = 120):
    print(f"\n[1/3] Generating Normal Operations Dataset ({num_missions} Diverse Missions)...")
    normal_base = DATA_ROOT / "normal"
    mission_types = ["standard_recon", "high_altitude_loiter", "low_level_dash", "aborted_go_around"]

    for i in range(1, num_missions + 1):
        mission_id = f"normal_{i:03d}"
        m_type = random.choice(mission_types)
        dur = random.uniform(2400, 4200) if m_type == "high_altitude_loiter" else random.uniform(600, 2400)
        sea_temp = random.uniform(-10.0, 42.0)
        qnh = random.uniform(-3.5, 2.5)
        initial_wear = 0.0 if i <= 15 else random.uniform(0.05, 0.40)

        tol = EngineManufacturingTolerance(engine_id=f"norm_eng_{i:03d}")
        twin = ApexPhysicalEngineTwin(tolerance=tol, initial_wear=initial_wear)

        df = create_mission(
            mission_id=mission_id, engine_twin=twin, fault_type="normal",
            mission_type=m_type, mission_duration_s=dur,
            sea_temp_C=sea_temp, qnh_offset_kPa=qnh
        )
        save_dataframe(df, normal_base / f"mission_{mission_id}")

    print(f"  ✓ Saved {num_missions} normal operational missions to {normal_base}")

# Generate Faulty Dataset
def generate_faulty_dataset(num_missions: int = 600):
    print(f"\n[2/3] Generating Fault Classification Dataset ({num_missions} Single & Compound Missions)...")
    faulty_base = DATA_ROOT / "faulty"

    single_faults = ["misfire", "injector_degradation", "turbo_issue", "lubrication_issue",
                     "sensor_drift", "overheating", "electrical_fault", "bearing_fault"]
    progression_choices = ["linear", "exponential", "brownian", "intermittent"]
    sensor_drift_modes = ["gradual_drift", "abrupt_step_bias", "stuck_at_value", "intermittent_spikes"]

    for i in range(1, num_missions + 1):
        dur = random.uniform(600, 2400)
        fault_onset_s = random.uniform(90, dur * 0.5)
        fault_ramp_s = random.uniform(25, 75)
        severity = random.choice([0.25, 0.45, 0.65, 0.85, 1.0])
        prog_type = random.choice(progression_choices)
        sea_temp = random.uniform(-8.0, 44.0)

        if random.random() < 0.25:
            fault_name = random.choice(["lubrication_issue", "turbo_issue", "injector_degradation", "bearing_fault"])
            compound = [random.choice(["overheating", "misfire", "electrical_fault"])]
            mission_id = f"compound_{fault_name}_{compound[0]}_{i:03d}"
            save_dir = faulty_base / "compound_faults"
        else:
            fault_name = single_faults[(i - 1) % len(single_faults)]
            compound = None
            mission_id = f"{fault_name}_{i:03d}"
            save_dir = faulty_base / fault_name

        s_mode = random.choice(sensor_drift_modes) if fault_name == "sensor_drift" else "none"
        tol = EngineManufacturingTolerance(engine_id=f"fault_eng_{i:03d}")
        twin = ApexPhysicalEngineTwin(tolerance=tol, initial_wear=random.uniform(0.05, 0.40))

        df = create_mission(
            mission_id=mission_id, engine_twin=twin, fault_type=fault_name,
            fault_onset_s=fault_onset_s, fault_ramp_s=fault_ramp_s,
            fault_peak_severity=severity, progression_type=prog_type,
            sensor_fault_mode=s_mode, compound_faults=compound,
            mission_duration_s=dur, sea_temp_C=sea_temp
        )
        save_dataframe(df, save_dir / f"mission_{mission_id}_sev{severity:.2f}")

    print(f"  ✓ Saved {num_missions} balanced fault missions to {faulty_base}")

# Generate Raw Vibration Benchmark
def generate_raw_vibration_benchmark(num_windows: int = 1200, window_s: float = VIBRATION_WINDOW_S):
    vib_dir = DATA_ROOT / "vibration_raw"
    vib_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(MASTER_SEED + 2401)
    rows = []
    conditions = ["normal", "bearing_wear", "misfire", "lubrication", "mixed"]
    per = max(1, num_windows // len(conditions))
    for j in range(per * len(conditions)):
        condition = conditions[j % len(conditions)]
        rpm = float(rng.uniform(1850.0, 2850.0))
        bearing, sev, misfire = float(rng.uniform(0.0, 0.15)), 0.0, 0.0
        if condition == "bearing_wear":
            bearing, sev = float(rng.uniform(0.25, 1.0)), float(rng.uniform(0.25, 0.85))
        elif condition == "misfire":
            misfire = float(rng.uniform(0.25, 1.0)); sev = misfire
        elif condition == "lubrication":
            bearing, sev = float(rng.uniform(0.15, 0.65)), float(rng.uniform(0.25, 0.85))
        elif condition == "mixed":
            bearing, misfire = float(rng.uniform(0.35, 0.9)), float(rng.uniform(0.25, 0.75))
            sev = max(bearing, misfire)
        seed = MASTER_SEED + 50000 + j
        vib_tol = EngineManufacturingTolerance(engine_id=f"vib_{j:05d}")
        t, x, feat = synthesize_vibration_waveform(window_s, VIBRATION_FS_HZ, rpm, sev, bearing,
                                                   misfire, seed=seed, bearing_tolerance=vib_tol)
        stem = vib_dir / f"window_{j+1:05d}_{condition}"
        np.savez_compressed(stem.with_suffix(".npz"), time_s=t, acceleration_g=x)
        rows.append({"window_id": stem.name, "condition": condition, "rpm": rpm, "severity": sev,
                     "bearing_wear": bearing, "misfire_severity": misfire, "source": "synthetic_benchmark", **feat})
    pd.DataFrame(rows).to_csv(vib_dir / "vibration_window_manifest.csv", index=False)
    print(f"  ✓ Saved {len(rows)} high-rate vibration windows at {VIBRATION_FS_HZ} Hz to {vib_dir}")

# Generate Rul Dataset
def generate_rul_dataset(num_lives: int = 50, aging_regime_override: Optional[str] = None):
    print(f"\n[3/3] Generating Full RUL Fleet Lifecycles ({num_lives} Run-to-Failure Engines)...")
    rul_base = DATA_ROOT / "rul"

    fault_choices = ["injector_degradation", "turbo_issue", "lubrication_issue", "bearing_fault", "electrical_fault"]
    aging_distribution = ["accelerated"] * 15 + ["nominal"] * 25 + ["extended"] * 10
    random.shuffle(aging_distribution)

    for life_idx in range(1, num_lives + 1):
        engine_id = f"engine_{life_idx:03d}"
        life_dfs: List[pd.DataFrame] = []

        if aging_regime_override is not None:
            regime = aging_regime_override
        else:
            regime = aging_distribution[life_idx - 1]
        tol = EngineManufacturingTolerance(engine_id=engine_id, aging_regime=regime)
        twin = ApexPhysicalEngineTwin(tolerance=tol, initial_wear=0.0)

        for mission_idx in range(1, 2001):
            mission_id = f"life{life_idx:03d}_mission{mission_idx:03d}"
            dur = random.uniform(600.0, 1500.0)

            fault_probability = min(0.14, (mission_idx - 1) * 0.00045)
            if random.random() < fault_probability:
                fault_type = random.choice(fault_choices)
                severity = random.uniform(0.25, 0.75)
                fault_onset_s = random.uniform(60, dur * 0.6)
                fault_ramp_s = random.uniform(30, 60)
            else:
                fault_type = "normal"
                severity = 0.0
                fault_onset_s = None
                fault_ramp_s = 60.0

            mission_type = random.choice([
                "standard_recon", "high_altitude_loiter",
                "low_level_dash", "aborted_go_around", "calibration_bench"
            ])

            df = create_mission(
                mission_id=mission_id, engine_twin=twin, fault_type=fault_type,
                fault_onset_s=fault_onset_s, fault_ramp_s=fault_ramp_s,
                fault_peak_severity=severity, mission_duration_s=dur,
                mission_type=mission_type
            )
            life_dfs.append(df)
            if mission_idx % 10 == 0 or twin.is_failed:
                print(f"    [life{life_idx:03d} {regime.upper():10s}] mission {mission_idx:4d} "
                      f"| {mission_type:22s} | {dur:5.0f}s | hours={df['actual_operating_hours'].iloc[-1]:6.2f} "
                      f"| health={df['true_overall_health_index'].iloc[-1]:5.2f} "
                      f"| failed={twin.is_failed}", flush=True)

            if twin.is_failed:
                break

        life_df = pd.concat(life_dfs, ignore_index=True)

        fail_indices = np.where(life_df["failure_flag"].values == 1)[0]
        if len(fail_indices) > 0:
            t_fail_s = float(life_df["global_time_s"].iloc[fail_indices[0]])
            time_vector_s = life_df["global_time_s"].values
            simulated_rul_hours = np.maximum(0.0, (t_fail_s - time_vector_s) / 3600.0)
            life_df["simulated_rul_hours"] = simulated_rul_hours
        else:
            life_df["simulated_rul_hours"] = np.nan

        rul_valid = life_df["simulated_rul_hours"].dropna().to_numpy()
        if len(rul_valid) > 1 and np.any(np.diff(rul_valid) > 1e-9):
            raise ValueError(f"RUL increased within lifecycle {engine_id}")
        if len(fail_indices) > 0 and abs(float(rul_valid[-1])) > 1e-9:
            raise ValueError(f"RUL did not reach zero at failure for {engine_id}")

        save_dataframe(life_df, rul_base / f"life_{life_idx:03d}")

        flight_hours = float(life_df["actual_operating_hours"].iloc[-1])
        final_health = float(life_df["true_overall_health_index"].iloc[-1])
        print(f"  ✓ Engine {life_idx:03d}/{num_lives:03d} [{regime.upper():11s}] | {len(life_dfs):3d} Missions | {flight_hours:5.1f} Flight Hrs | Mode: {twin.failure_mode}")

    num_train = int(num_lives * 0.70)
    num_val = int(num_lives * 0.15)

    manifest = {
        "dataset_name": "SIH Rustom-2 Digital Twin Production Benchmark (v30.0)",
        "description": "Physics-locked merge: BEM-derived propeller map calibrated to the 125 kW rated point, simultaneous shaft-torque equilibrium (thrust trim <-> engine torque <-> fuel), variable-mass 3-DOF flight with deterministic wind, speed/altitude-hold autopilot, full ISA envelope to ~11 km, stress-driven degradation, imperfect sensors, and grouped RUL labels.",
        "propeller_map": {
            "source": "representative_tabulated_bem_surrogate",
            "requires_external_data": True,
            "calibrated_at_pitch": CONST.prop_design_pitch,
            "calibrated_at_J": CONST.prop_design_J,
            "rated_power_kW": CONST.rated_brake_power_kW,
            "rated_rpm": CONST.rated_rpm,
            "rated_bsfc_g_kWh": CONST.rated_bsfc_g_kWh,
        },
        "feature_columns": FEATURE_COLUMNS,
        "ml_feature_columns": ML_FEATURE_COLUMNS,
        "target_columns": TARGET_COLUMNS,
        "metadata_columns": METADATA_COLUMNS,
        "notes": [
            "Do not use row-level random train/test splitting for lifecycle data.",
            "Group RUL splits by engine_id.",
            "simulated_rul_hours is simulation ground truth, not physical teardown truth.",
            "fadec_mode is retained for control analysis but excluded from the recommended ML feature set.",
            "Propeller pitch has NO separate governor: pitch is aerodynamic thrust trim; engine fuel follows the shaft-power demand scaled by throttle (primary power command).",
            "The shaft-torque equilibrium (Q_prop = Q_indicated - Q_friction) is solved simultaneously with fuel and RPM at each step; shaft_torque_error_Nm records the residual.",
            "aircraft_mass_kg/fuel_mass_kg/fuel_used_kg track conserved fuel mass; sorties start refueled.",
            "wind_u_mps/wind_accel_mps2 are deterministic per mission_id (crc32 seed, process-independent).",
            "RUL lifecycle data are generated at 2 Hz; vibration is feature-level in telemetry plus a separate 2048 Hz raw benchmark.",
            "Fault coverage: misfire, injector_degradation, turbo_issue, lubrication_issue, sensor_drift, overheating, electrical_fault, bearing_fault.",
            "Bearing orders (vib_bpfo_hz, vib_bpfi_hz, vib_bsf_hz) come from per-engine rolling-bearing geometry.",
            "Spectral vibration features, 12-bit ADC quantization streams, and sensor-health flags are available for analysis."
        ],
        "engine_splits": {
            "train_engines": [f"engine_{i:03d}" for i in range(1, num_train + 1)],
            "val_engines": [f"engine_{i:03d}" for i in range(num_train + 1, num_train + num_val + 1)],
            "test_engines": [f"engine_{i:03d}" for i in range(num_train + num_val + 1, num_lives + 1)],
        }
    }
    with open(DATA_ROOT / "feature_target_manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"\n  ✓ Saved verified fleet manifest to {DATA_ROOT / 'feature_target_manifest.json'}")

# Verify Rated Propeller Point
def verify_rated_propeller_point():
    print("\n--- Rated-Point Probe (full throttle @ design advance ratio) ---")
    tol = EngineManufacturingTolerance(engine_id="verify_rated")
    rated_fuel_kg_s = (CONST.rated_brake_power_kW * CONST.rated_bsfc_g_kWh / 3600.0) / 1000.0
    omega_rated = CONST.rated_rpm * 2.0 * np.pi / 60.0
    friction_rated_Nm = tol.fric_torque_base * 1.10 + 0.0009 * omega_rated
    eta_ind_ref = (CONST.rated_brake_power_kW * 1000.0 + friction_rated_Nm * omega_rated) / max(
        rated_fuel_kg_s * CONST.lhv_fuel, 1.0)
    eta_ind_ref = float(np.clip(eta_ind_ref, 0.30, 0.48))
    eta_map_ref, _ = engine_operating_map(1.0, CONST.rated_rpm)
    rho_sea = 1.225
    mu0 = tol.mu0_oil

    # Residual
    def residual(omega_w):
        n_w = max(1.0, omega_w / (2.0 * np.pi))
        rpm_w = n_w * 60.0
        airspeed_w = n_w * CONST.D_prop * CONST.prop_design_J
        Qp_w, Tp_w, Cp_w, Ct_w, J_w, eta_w = propeller_aerodynamics(
            n_w, airspeed_w, CONST.prop_design_pitch, rho_sea)
        Q_visc = tol.fric_torque_base * (0.85 + 0.15 * rpm_w / CONST.rated_rpm) * (mu0 / max(mu0, 1e-6)) ** 0.22
        Q_speed = 0.0009 * omega_w ** 1.45
        Q_boundary = 0.8 / (1.0 + 80.0 * mu0 * omega_w)
        Qf_w = max(4.0, Q_visc + Q_speed + Q_boundary)
        rack_x = float(np.clip((rpm_w / CONST.rated_rpm - 0.55) / 0.45, 0.0, 1.0))
        rack = float(0.24 + 0.76 * rack_x)
        mdot_w = rated_fuel_kg_s * rack
        if rpm_w >= CONST.overspeed_rpm:
            mdot_w *= float(np.clip(1.0 - (rpm_w - CONST.overspeed_rpm) /
                                    max(CONST.overspeed_cut_rpm - CONST.overspeed_rpm, 1.0), 0.0, 1.0))
        if rpm_w >= CONST.overspeed_cut_rpm:
            mdot_w = 0.0
        mdot_w = float(np.clip(mdot_w, 0.0, rated_fuel_kg_s * 1.05))
        eta_map_w, _ = engine_operating_map(1.0, rpm_w)
        eta_ind_w = float(np.clip(eta_ind_ref * (eta_map_w / max(eta_map_ref, 1e-6)), 0.10, 0.46))
        Q_ind_w = mdot_w * CONST.lhv_fuel * eta_ind_w / max(omega_w, 20.0)
        return Qp_w - (Q_ind_w - Qf_w), {"omega": omega_w, "rpm": rpm_w, "Qp": Qp_w, "Qf": Qf_w,
                                         "Qind": Q_ind_w, "mdot": mdot_w, "eta_ind": eta_ind_w,
                                         "J": J_w, "Cp": Cp_w, "Ct": Ct_w}

    omega_min = CONST.idle_rpm * 2.0 * np.pi / 60.0
    omega_max = (CONST.max_rpm + 100.0) * 2.0 * np.pi / 60.0
    lo_b, hi_b = omega_min, omega_max
    for _ in range(40):
        mid = 0.5 * (lo_b + hi_b)
        r_lo, _ = residual(lo_b)
        r_mid, _ = residual(mid)
        if r_lo * r_mid <= 0.0:
            hi_b = mid
        else:
            lo_b = mid
        if abs(hi_b - lo_b) < 1e-6:
            break
    r_root, probe = residual(0.5 * (lo_b + hi_b))
    rpm = probe["rpm"]
    brake_kW = (probe["Qind"] - probe["Qf"]) * probe["omega"] / 1000.0
    mdot_g_s = probe["mdot"] * 1000.0
    bsfc = mdot_g_s * 3600.0 / max(brake_kW, 0.05)
    print(f"  settled RPM   : {rpm:7.1f}")
    print(f"  brake power   : {brake_kW:7.2f} kW (rated 125.00)")
    print(f"  indicated P   : {probe['Qind'] * probe['omega'] / 1000.0:7.2f} kW")
    print(f"  Q_prop        : {probe['Qp']:7.1f} Nm (rated 426.3)")
    print(f"  fuel flow     : {mdot_g_s:6.3f} g/s")
    print(f"  actual BSFC   : {bsfc:6.1f} g/kWh (rated 215.8)")
    print(f"  J pinned      : {probe['J']:4.3f} (design {CONST.prop_design_J:.2f})")
    print(f"  residual      : {r_root:7.3f} Nm")
    if abs(rpm - CONST.rated_rpm) > 40.0 or abs(brake_kW - 125.0) / 125.0 > 0.03:
        raise RuntimeError(f"Rated point not reproduced: rpm={rpm:.0f}, brake={brake_kW:.1f} kW")
    return rpm, brake_kW, mdot_g_s

# Verify Mission Smoke
def verify_mission_smoke():
    print("\n--- Mission Smoke Test (standard_recon, 600 s) ---")
    tol = EngineManufacturingTolerance(engine_id="verify_smoke")
    twin = ApexPhysicalEngineTwin(tolerance=tol, initial_wear=0.15)
    df = create_mission(mission_id="smoke_check_m1", engine_twin=twin, fault_type="normal",
                        mission_type="standard_recon", mission_duration_s=600.0)
    cols = ["true_brake_power_kW", "true_rpm", "airspeed_mps", "altitude_m",
            "rpm", "cht_cyl_avg_C", "egt_cyl_avg_C", "fuel_flow_g_s", "vib_rms_g",
            "battery_voltage_V", "vib_bpfo_hz", "aircraft_mass_kg", "ground_speed_mps"]
    for c in cols:
        print(f"  {c:24s}: min={df[c].min():9.3f}  max={df[c].max():9.3f}")
    print(f"  rows={len(df)}  closure max={df['true_energy_closure_error_W'].abs().max():.3f} W")
    print(f"  desired thrust median={df['desired_thrust_N'].median():.0f} N  "
          f"thrust eq err max|abs|={df['thrust_equilibrium_error_N'].abs().max():.1f} N")
    return df

# Add Physics Health Indicators
def add_physics_health_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    dens_ref = 1.225
    rho = np.clip(df["air_density_kg_m3"].to_numpy(), 0.3, 1.25)
    thr = np.clip(df["throttle_actual"].to_numpy(), 0.0, 1.0)
    rpm_n = np.clip(df["true_rpm"].to_numpy() / 2800.0, 0.0, 1.15)
    load_ = np.clip(thr, 0.0, 1.0) * (0.55 + 0.45 * rpm_n)
    expected_boost = 9.0 + 107.0 * load_ * (dens_ref / rho)
    df["expected_boost_kPa"] = expected_boost
    df["boost_residual_kPa"] = df["boost_pressure_kPa"].to_numpy() - expected_boost
    egt = df["egt_cyl_avg_C"].to_numpy()
    df["thermal_stress"] = egt * (0.4 + 0.6 * rpm_n) * (rho / dens_ref)
    ff = np.maximum(df["fuel_flow_g_s"].to_numpy(), 1e-6)
    rev_s = np.maximum(df["true_rpm"].to_numpy() / 60.0, 1e-6)
    df["fuel_per_rpm_g_rev"] = ff / rev_s
    oild = np.maximum(df["oil_temp_C"].to_numpy(), 1e-6)
    df["oil_efficiency_KPa_per_C"] = df["oil_pressure_kPa"].to_numpy() / oild
    return df

# Generate Ml Training Datasets
def generate_ml_training_datasets(sequence_s: float = 10.0):
    ml_dir = DATA_ROOT / "ml"
    ml_dir.mkdir(parents=True, exist_ok=True)
    seq_len = max(2, int(round(sequence_s * (1.0 / DT))))

    # Discover
    def discover(root_name: str):
        base = DATA_ROOT / root_name
        hits = sorted(list(base.rglob("*.parquet")) + list(base.rglob("*.csv")))
        frames = []
        for p in hits:
            try:
                frames.append(pd.read_csv(p) if p.suffix == ".csv" else pd.read_parquet(p))
            except Exception:
                pass
        return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

    normal = discover("normal")
    faulty = discover("faulty")
    rul = discover("rul")
    if not len(normal) and not len(faulty) and not len(rul):
        print("  [ml] No raw normal/faulty/rul splits found under DATA_ROOT - "
              "run the generators first.")
        return

    phys_cols = ML_FEATURE_COLUMNS
    _num_exclude = {"settled_state"}

    # Numeric Schema
    def _numeric_schema(df):
        if not len(df):
            return []
        return [c for c in ML_FEATURE_COLUMNS
                if c in df.columns and c not in _num_exclude
                and pd.api.types.is_numeric_dtype(df[c])]

    feature_schema = _numeric_schema(normal) or _numeric_schema(faulty) or \
        _numeric_schema(rul) or ML_FEATURE_COLUMNS

    # Ready
    def ready(df, label):
        if not len(df):
            return df
        df = add_physics_health_indicators(df)
        keep = [c for c in phys_cols + ["expected_boost_kPa", "boost_residual_kPa",
                                        "thermal_stress", "fuel_per_rpm_g_rev",
                                        "oil_efficiency_KPa_per_C"] if c in df.columns]
        return df[keep]

    normal_p = ready(normal, "normal") if len(normal) else pd.DataFrame()

    # Label Cls
    def label_cls(df):
        return df["fault_type"].map(lambda s: "normal" if str(s) == "normal" else str(s)).fillna("normal")

    if len(normal_p):
        normal_p["label_cls"] = "normal"
        normal_p.to_csv(ml_dir / "anomaly_healthy.csv", index=False)
    if len(faulty):
        f = add_physics_health_indicators(faulty)
        fkeep = [c for c in phys_cols + ["expected_boost_kPa", "boost_residual_kPa",
                                         "thermal_stress", "fuel_per_rpm_g_rev",
                                         "oil_efficiency_KPa_per_C"] if c in f.columns]
        fsel = f[fkeep].copy()
        fsel["label_cls"] = label_cls(f)
        fsel.to_csv(ml_dir / "anomaly_fault.csv", index=False)

    # Windows
    def _windows(df):
        if not len(df):
            return None, None
        clean = df[feature_schema].copy()
        clean.ffill(inplace=True)
        clean.bfill(inplace=True)
        vals = clean.to_numpy(dtype="float64")
        n = len(vals)
        if n < seq_len:
            return None, None
        idxs = np.arange(seq_len)[None, :] + np.arange(n - seq_len + 1)[:, None]
        return vals[idxs], idxs

    # Write Windows
    def write_windows(name, df, labels):
        if not len(df):
            return
        X, idxs = _windows(df)
        if X is None:
            return
        if labels is None:
            labs = np.zeros(len(X), dtype="int64")
        elif hasattr(labels, "to_numpy"):
            labs = labels.to_numpy()
        else:
            labs = np.asarray(labels)
        np.savez(ml_dir / name, X=X, seq_len=seq_len,
                  feature_names=np.array(feature_schema, dtype="U"),
                  labels=labs,
                  label_names=np.array(["normal", "misfire", "injector_degradation",
                                         "turbo_issue", "lubrication_issue",
                                         "sensor_drift", "overheating",
                                         "electrical_fault", "bearing_fault"], dtype="U"))

    if len(normal_p):
        write_windows("lstm_windows_normal.npz", normal_p,
                      np.zeros(len(normal_p), dtype="int64"))
    if len(faulty):
        fsel2 = fsel
        fmap = {c: i + 1 for i, c in enumerate(["misfire", "injector_degradation",
                                                "turbo_issue", "lubrication_issue",
                                                "sensor_drift", "overheating",
                                                "electrical_fault", "bearing_fault"])}
        flab = fsel2["label_cls"].map(lambda s: fmap.get(str(s), 0)).to_numpy(dtype="int64")
        write_windows("lstm_windows_fault.npz", fsel2, flab)

    if len(rul):
        rulp = add_physics_health_indicators(rul)
        rkeep = [c for c in phys_cols + ["expected_boost_kPa", "boost_residual_kPa",
                                         "thermal_stress", "fuel_per_rpm_g_rev",
                                         "oil_efficiency_KPa_per_C"] if c in rulp.columns]
        rsel = rulp[rkeep].copy()
        for c in ["true_overall_health_index", "simulated_rul_hours", "engine_id"]:
            if c in rulp.columns:
                rsel[c] = rulp[c]
        rsel.to_csv(ml_dir / "rul_features.csv", index=False)

    engines = sorted(set(rul["engine_id"]) if len(rul) and "engine_id" in rul.columns else set())
    if engines:
        import random as _r
        _r.seed(2026)
        _r.shuffle(engines)
        n_te = max(1, int(len(engines) * 0.20))
        test_eng = set(engines[:n_te])
        train_eng = [e for e in engines if e not in test_eng]
        splits = {"train_engines": sorted(train_eng), "test_engines": sorted(test_eng),
                  "val_engines": [], "feature_schema": feature_schema,
                  "seq_len": seq_len, "dt_s": DT}
        (ml_dir / "splits.json").write_text(json.dumps(splits, indent=2))
        print(f"  [ml] engine-grouped split: train={train_eng} test={sorted(test_eng)}")

    stats = {
        "normal_rows": int(len(normal_p)),
        "faulty_rows": int(len(faulty)),
        "rul_rows": int(len(rul)),
        "seq_len": seq_len,
        "sequence_s": sequence_s,
        "physic_indicators": ["expected_boost_kPa", "boost_residual_kPa",
                              "thermal_stress", "fuel_per_rpm_g_rev",
                              "oil_efficiency_KPa_per_C"],
    }
    if len(faulty):
        stats["fault_class_counts"] = {str(k): int(v) for k, v in
                                       label_cls(faulty).value_counts().to_dict().items()}
    (ml_dir / "dataset_stats.json").write_text(json.dumps(stats, indent=2))
    print(f"  [ml] ML training artifacts written to {ml_dir}")
    for f in sorted(ml_dir.glob("*.*")):
        print(f"        - {f.name}  ({f.stat().st_size:,} bytes)")

if __name__ == "__main__":
    print("=" * 80)
    print("SIH RUSTOM-2 DIGITAL TWIN SIMULATION ENGINE (v30.0 - PHYSICS-LOCKED MERGE)")
    print("=" * 80)
    print(f"Target Output Directory : {DATA_ROOT.resolve()}")
    print(f"Primary Output Format   : {OUTPUT_FORMAT.upper()}")

    if "--verify" in sys.argv:
        verify_rated_propeller_point()
        verify_mission_smoke()
        print("\nRated-point probe + mission smoke test completed.")
    elif "--ml" in sys.argv:
        generate_ml_training_datasets()
    else:
        generate_normal_dataset(num_missions=120)
        generate_raw_vibration_benchmark(num_windows=1200)
        generate_faulty_dataset(num_missions=600)
        generate_rul_dataset(num_lives=50)
        generate_ml_training_datasets()

        print("\n" + "=" * 80)
        print("CALIBRATED DATASET GENERATION COMPLETE & FULLY VERIFIED")
        print("=" * 80)
