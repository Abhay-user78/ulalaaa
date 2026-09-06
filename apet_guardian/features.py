# Physics features
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import PolynomialFeatures

from . import config as C

_RESID_RESID = {
    "boost_pressure_kPa": "boost_residual_kPa",
    "cht_cyl_avg_C": "cht_residual_C",
    "egt_cyl_avg_C": "egt_residual_C",
    "oil_pressure_kPa": "oilp_residual_kPa",
}
_EXP_PREFIX = {
    "boost_pressure_kPa": "expected_boost",
    "cht_cyl_avg_C": "expected_cht",
    "egt_cyl_avg_C": "expected_egt",
    "oil_pressure_kPa": "expected_oilp",
}

# Physicsfeaturemodel
class PhysicsFeatureModel:
    # Init
    def __init__(self):
        self._input_mean = None
        self._input_std = None
        self._models = {}
        self._polys = {}
        self._phase_stats = {}

    @staticmethod
    # Context
    def _context(df: pd.DataFrame) -> np.ndarray:
        ctx = df[C.CONTEXT_COLS].reindex(columns=C.CONTEXT_COLS).to_numpy(dtype="float64")
        return np.nan_to_num(ctx, nan=0.0, posinf=0.0, neginf=0.0)

    # Fit Context Scaler
    def fit_context_scaler(self, healthy_df: pd.DataFrame):
        ctx = self._context(healthy_df)
        self._input_mean = ctx.mean(axis=0)
        self._input_std = ctx.std(axis=0) + 1e-9
        return self

    # Fit
    def fit(self, healthy_df: pd.DataFrame, max_rows: int = 250_000):
        df = healthy_df
        ctx = self._context(df)
        z = (ctx - self._input_mean) / self._input_std
        if len(z) > max_rows:
            rng = np.random.default_rng(C.SPLIT_SEED)
            idx = rng.choice(len(z), max_rows, replace=False)
            z = z[idx]
            df = df.iloc[idx]
        for target, prefix in _EXP_PREFIX.items():
            if target not in df.columns:
                continue
            y = df[target].to_numpy(dtype="float64")
            y = np.nan_to_num(y, nan=np.nanmedian(y))
            poly = PolynomialFeatures(degree=2, include_bias=True)
            model = Ridge(alpha=10.0)
            model.fit(poly.fit_transform(z), y)
            self._models[target] = model
            self._polys[target] = poly
        return self

    # Fit Phase Stats
    def fit_phase_stats(self, healthy_df: pd.DataFrame):
        stat_map = {"boost_residual": "boost_residual_kPa",
                    "cht_residual": "cht_residual_C",
                    "egt_residual": "egt_residual_C",
                    "oilp_residual": "oilp_residual_kPa"}
        tmp = self._residual_frame(healthy_df)
        stats = {}
        if "phase" in tmp.columns:
            for phase, sub in tmp.groupby("phase"):
                if len(sub) < C.HEALTHY_PHASE_MIN_ROWS:
                    continue
                row = {}
                for key, coln in stat_map.items():
                    v = sub[coln].to_numpy(dtype="float64")
                    v = v[np.isfinite(v)]
                    if len(v):
                        row[key] = (float(np.mean(v)), float(np.std(v) + 1e-9))
                stats[str(phase)] = row
        fallback = {}
        for key, coln in stat_map.items():
            v = tmp[coln].to_numpy(dtype="float64")
            v = v[np.isfinite(v)]
            if len(v):
                fallback[key] = (float(np.mean(v)), float(np.std(v) + 1e-9))
        stats["ALL"] = fallback
        self._phase_stats = stats
        return self

    # Residual Frame
    def _residual_frame(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        ctx = self._context(df)
        z = (ctx - self._input_mean) / self._input_std
        for target, exp_name in _EXP_PREFIX.items():
            if target in self._models and target in df.columns:
                Xp = self._polys[target].transform(z)
                df[exp_name] = self._models[target].predict(Xp)
        for target, resid in _RESID_RESID.items():
            exp_name = _EXP_PREFIX[target]
            if exp_name in df.columns and target in df.columns:
                df[resid] = df[target].to_numpy(dtype="float64") - df[exp_name].to_numpy(dtype="float64")
            else:
                df[resid] = np.zeros(len(df), dtype="float64")
        return df

    # Transform Residuals Only
    def transform_residuals_only(self, df: pd.DataFrame) -> pd.DataFrame:
        tmp = self._residual_frame(df)
        stat_map = {"boost_residual": "boost_residual_kPa",
                    "cht_residual": "cht_residual_C",
                    "egt_residual": "egt_residual_C",
                    "oilp_residual": "oilp_residual_kPa"}
        norm = {}
        for key, col in stat_map.items():
            vals = tmp[col].to_numpy(dtype="float64")
            out = np.zeros(len(tmp), dtype="float64")
            if "phase" in tmp.columns:
                phases = tmp["phase"].to_numpy(dtype=object)
                for pname in pd.unique(phases):
                    m = phases == pname
                    stat = (self._phase_stats.get(str(pname)) or {}).get(key) or \
                        (self._phase_stats.get("ALL") or {}).get(key) or (0.0, 1.0)
                    out[m] = (vals[m] - stat[0]) / stat[1]
            else:
                stat = (self._phase_stats.get("ALL") or {}).get(key) or (0.0, 1.0)
                out = (vals - stat[0]) / stat[1]
            tmp[key + "_norm"] = out
        return tmp

    # Transform
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        df = self.transform_residuals_only(df)
        n = len(df)

        # Col
        def col(name, default=0.0):
            if name in df.columns:
                return np.nan_to_num(df[name].to_numpy(dtype="float64"))
            return np.full(n, default, dtype="float64")

        rpm = np.clip(col("true_rpm", C.BEM_RATED_RPM) / C.BEM_RATED_RPM, 0.0, 1.25)
        rho = np.clip(col("air_density_kg_m3", C.RHO_SEA) / C.RHO_SEA, 0.2, 1.6)
        thr = np.clip(col("throttle_actual"), 0.0, 1.0)
        egt = col("egt_cyl_avg_C")
        cht = col("cht_cyl_avg_C")
        oil_p = col("oil_pressure_kPa")
        oil_t = col("oil_temp_C")
        fuel = np.maximum(col("fuel_flow_g_s"), 0.0)
        vib = col("vib_rms_g")
        p_br = col("true_brake_power_kW")
        br = col("boost_residual_kPa")
        exp_oilp = col("expected_oilp", oil_p)

        thermal = egt * (0.4 + 0.6 * rpm) * rho
        df["thermal_stress"] = thermal
        df["cht_stress"] = cht * (0.5 + 0.5 * thr)
        rev_s = np.maximum(rpm * C.BEM_RATED_RPM / 60.0, 1e-6)
        df["fuel_per_rpm_g_rev"] = fuel / rev_s
        df["oil_efficiency_KPa_per_C"] = oil_p / np.maximum(oil_t, 1e-6)
        df["vibration_trend"] = np.concatenate([[0.0], np.diff(vib)]) / C.SAMPLE_DT

        cums = {
            "cum_thermal_stress": np.cumsum(thermal),
            "cum_load_stress": np.cumsum(p_br / C.BEM_RATED_KW),
            "cum_egt_excess": np.cumsum(np.maximum(egt - 700.0, 0.0)),
            "cum_cht_excess": np.cumsum(np.maximum(cht - 200.0, 0.0)),
            "cum_vib": np.cumsum(vib),
            "boost_resid_integrated": np.cumsum(br),
            "cum_oil_loss": np.cumsum(np.maximum(exp_oilp - oil_p, 0.0)),
            "cum_fuel": np.cumsum(fuel),
        }
        for k, v in cums.items():
            df[k] = v
        return df

# Build Physics Model
def build_physics_model(train_healthy_df: pd.DataFrame) -> PhysicsFeatureModel:
    df = train_healthy_df.copy()
    df = df.sort_values(["engine_id", "global_time_s"]).reset_index(drop=True)
    pm = PhysicsFeatureModel()
    pm.fit_context_scaler(df)
    pm.fit(df)
    pm.fit_phase_stats(df)
    return pm
