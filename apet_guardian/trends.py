import numpy as np
import pandas as pd
from typing import List, Tuple, Optional

from . import config as C


def _linear_slope(y: np.ndarray) -> float:
    n = len(y)
    if n < 2:
        return 0.0
    t = np.arange(n, dtype="float64")
    tm = t.mean()
    denom = np.sum((t - tm) ** 2)
    if denom < 1e-12:
        return 0.0
    return float(np.sum((y - y.mean()) * (t - tm)) / denom)


def compute_trend_features_single(sig: np.ndarray, sample_dt: float = 0.5) -> np.ndarray:
    n = len(sig)
    if n < 4:
        return np.zeros(len(C.TREND_STATS), dtype="float32")

    slope = _linear_slope(sig)
    dt_total = (n - 1) * sample_dt
    rate = (sig[-1] - sig[0]) / max(dt_total, 1e-6)

    if n >= 4:
        diff1 = np.diff(sig)
        acceleration = float(np.mean(np.abs(np.diff(diff1)))) / max(sample_dt, 1e-6)
    else:
        acceleration = 0.0

    mean_recent = sig[-min(10, n):].mean()
    std_recent = max(sig[-min(10, n):].std(), 1e-6)
    deviation = (sig[-1] - mean_recent) / std_recent

    return np.array([slope, rate, acceleration, deviation], dtype="float32")


def compute_trend_features(seq_windows: np.ndarray,
                           feature_names: List[str],
                           signals: Optional[List[str]] = None,
                           sample_dt: float = 0.5) -> Tuple[np.ndarray, List[str]]:
    if signals is None:
        signals = C.TREND_SENSOR_SIGNALS
    sig_idx = {name: i for i, name in enumerate(feature_names)}
    avail = [s for s in signals if s in sig_idx]

    parts = []
    cols = []
    for s in avail:
        j = sig_idx[s]
        w = seq_windows[:, :, j].astype("float64")
        trend_per_window = np.zeros((len(w), len(C.TREND_STATS)), dtype="float32")
        for i in range(len(w)):
            trend_per_window[i] = compute_trend_features_single(w[i], sample_dt)
        parts.append(trend_per_window)
        cols.extend([f"{s}__{st}" for st in C.TREND_STATS])

    if not parts:
        return np.zeros((seq_windows.shape[0], 0), dtype="float32"), []

    return np.concatenate(parts, axis=1), cols


def label_pre_fault_windows(df: pd.DataFrame,
                            fault_columns: List[str],
                            lookahead_rows: int = 360,
                            sample_dt: float = 0.5) -> np.ndarray:
    n = len(df)
    labels = np.zeros(n, dtype="float32")

    for fc in fault_columns:
        if fc not in df.columns:
            continue
        fault_active = df[fc].to_numpy(dtype="float64") > 0.5
        fault_indices = np.where(fault_active)[0]
        if len(fault_indices) == 0:
            continue
        first_fault = fault_indices[0]
        start = max(0, first_fault - lookahead_rows)
        for i in range(start, first_fault):
            labels[i] = 1.0

    return labels


def label_time_to_fault(df: pd.DataFrame,
                        fault_columns: List[str],
                        sample_dt: float = 0.5) -> np.ndarray:
    n = len(df)
    ttf = np.full(n, 9999.0, dtype="float32")
    time_s = df["global_time_s"].to_numpy(dtype="float64") if "global_time_s" in df.columns else np.arange(n) * sample_dt

    for fc in fault_columns:
        if fc not in df.columns:
            continue
        fault_active = df[fc].to_numpy(dtype="float64") > 0.5
        fault_indices = np.where(fault_active)[0]
        if len(fault_indices) == 0:
            continue
        first_fault = fault_indices[0]
        fault_time = time_s[first_fault]
        for i in range(n):
            remaining = fault_time - time_s[i]
            if remaining > 0:
                ttf[i] = np.minimum(ttf[i], remaining)

    return ttf
