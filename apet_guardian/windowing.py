# Windowing logic
import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

from . import config as C

# Compute Split
def compute_split(groups: Iterable[str], frac: Tuple[float, float, float] = C.SPLIT_FRAC,
                  seed: int = C.SPLIT_SEED) -> Dict[str, str]:
    rng = np.random.default_rng(seed)
    ids = sorted(set(str(g) for g in groups))
    rng.shuffle(ids)
    n = len(ids)
    nt = int(round(n * frac[0]))
    nv = int(round(n * frac[1]))
    assign = {}
    for i, gid in enumerate(ids):
        if i < nt:
            assign[gid] = "train"
        elif i < nt + nv:
            assign[gid] = "val"
        else:
            assign[gid] = "test"
    return assign

# Contiguous Blocks
def _contiguous_blocks(time_s: np.ndarray) -> List[Tuple[int, int]]:
    blocks = []
    if len(time_s) == 0:
        return blocks
    start = 0
    for i in range(1, len(time_s)):
        dt = time_s[i] - time_s[i - 1]
        if dt < -C.TIME_TOL or dt > C.MAX_GAP or not np.isfinite(dt):
            blocks.append((start, i))
            start = i
    blocks.append((start, len(time_s)))
    return blocks

# Window Sequence
def window_sequence(sig: np.ndarray, seq_len: int = C.SEQ_LEN, stride: int = C.STRIDE,
                    time_s: Optional[np.ndarray] = None,
                    max_gap: float = C.MAX_GAP) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if sig.ndim == 1:
        sig = sig[:, None]
    n, d = sig.shape
    if n < seq_len:
        return np.empty((0, seq_len, d), dtype=sig.dtype), np.empty(0, dtype=int), np.empty(0, dtype=int)
    starts = np.arange(0, n - seq_len + 1, stride)
    if time_s is not None:
        valid = np.zeros(len(starts), dtype=bool)
        for k, s in enumerate(starts):
            block_t = time_s[s:s + seq_len]
            dt = np.diff(block_t)
            valid[k] = np.all(np.isfinite(dt)) and (np.all(dt >= -C.TIME_TOL)) and (np.all(dt <= max_gap))
        starts = starts[valid]
    if len(starts) == 0:
        return np.empty((0, seq_len, d), dtype=sig.dtype), np.empty(0, dtype=int), np.empty(0, dtype=int)
    idx = starts[:, None] + np.arange(seq_len)[None, :]
    windows = sig[idx]
    return windows, starts, starts + seq_len - 1

# Slope
def _slope(windows: np.ndarray) -> np.ndarray:
    w = windows.astype("float64")
    n = w.shape[1]
    t = np.arange(n, dtype="float64")
    tm = t.mean()
    denom = np.sum((t - tm) ** 2)
    return np.sum((w - w.mean(axis=1, keepdims=True)) * (t - tm)[None, :, None], axis=1) / max(denom, 1e-9)

# Window Stats
def window_stats(seq_windows: np.ndarray, feature_names: List[str],
                 signals: Optional[List[str]] = None) -> Tuple[np.ndarray, List[str]]:
    if signals is None:
        signals = C.STATS_SIGNALS
    sig_idx = {name: i for i, name in enumerate(feature_names)}
    avail = [s for s in signals if s in sig_idx]
    cols = []
    parts = []
    for s in avail:
        j = sig_idx[s]
        w = seq_windows[:, :, j].astype("float64")
        mean = w.mean(axis=1)
        std = w.std(axis=1)
        mn = w.min(axis=1)
        mx = w.max(axis=1)
        p90 = np.percentile(w, 90, axis=1)
        rng = mx - mn
        sl = _slope(w[:, :, None])[:, 0]
        fin = w[:, -1]
        parts.append(np.stack([mean, std, mn, mx, p90, rng, sl, fin], axis=1))
        cols.extend([f"{s}__{st}" for st in C.STAT_NAMES])
    return np.concatenate(parts, axis=1).astype("float32"), cols

# Floatstore
class FloatStore:
    # Init
    def __init__(self, path: Path, ncols: int):
        self.path = path
        meta = path.with_suffix(path.suffix + ".meta.json")
        self.ncols = ncols
        self.rows = 0
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if meta.exists() and path.exists():
            self.rows = int(json.loads(meta.read_text()).get("rows", 0))

    # Append
    def append(self, x: np.ndarray):
        x = np.ascontiguousarray(np.asarray(x, dtype="float32"))
        flat = x.reshape(-1)
        with open(self.path, "ab") as f:
            flat.tofile(f)
        self.rows += len(x)
        meta = self.path.with_suffix(self.path.suffix + ".meta.json")
        meta.write_text(json.dumps({"rows": self.rows, "ncols": self.ncols}))

    # Read
    def read(self) -> np.ndarray:
        if self.rows == 0 or not self.path.exists():
            return np.empty((0, self.ncols), dtype="float32")
        mm = np.memmap(self.path, dtype="float32", mode="r")
        return mm.reshape(self.rows, self.ncols)

    # Npy
    def npy(self) -> np.ndarray:
        return np.array(self.read())

# Seqstore
class SeqStore:
    # Init
    def __init__(self, path: Path, seq_len: int, nch: int):
        self.path = path
        self.seq_len = seq_len
        self.nch = nch
        self.rows = 0
        meta = path.with_suffix(path.suffix + ".meta.json")
        path.parent.mkdir(parents=True, exist_ok=True)
        if meta.exists() and path.exists():
            self.rows = int(json.loads(meta.read_text()).get("rows", 0))

    # Append
    def append(self, x: np.ndarray):
        x = np.ascontiguousarray(np.asarray(x, dtype="float32"))
        flat = x.reshape(-1)
        with open(self.path, "ab") as f:
            flat.tofile(f)
        self.rows += len(x)
        meta = self.path.with_suffix(self.path.suffix + ".meta.json")
        meta.write_text(json.dumps({"rows": self.rows, "seq_len": self.seq_len, "nch": self.nch}))

    # Read
    def read(self) -> np.ndarray:
        if self.rows == 0 or not self.path.exists():
            return np.empty((0, self.seq_len, self.nch), dtype="float32")
        mm = np.memmap(self.path, dtype="float32", mode="r")
        return mm.reshape(self.rows, self.seq_len, self.nch)

# Stringmetastore
class StringMetaStore:
    # Init
    def __init__(self, path: Path):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)

    # Append
    def append(self, rows: List[dict]):
        with open(self.path, "a", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, default=str) + "\n")

    # Read
    def read(self) -> pd.DataFrame:
        if not self.path.exists():
            return pd.DataFrame()
        return pd.read_json(self.path, lines=True)

# Window Mission Dataframe
def window_mission_dataframe(df: pd.DataFrame, seq_features: List[str],
                             mission_col: str = "mission_id") -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if not len(df):
        zero = np.empty((0, C.SEQ_LEN, len(seq_features)), dtype="float32")
        return zero, np.empty(0), np.empty(0), np.empty(0), np.empty(0, dtype=int)
    seq_all, start_all, end_all, phase_all, rowend_all = [], [], [], [], []
    for _, sub_orig in df.groupby(mission_col, sort=False):
        sub = sub_orig.sort_values("time_in_mission_s")
        pos = sub.index.to_numpy()
        seq_mat = sub[seq_features].to_numpy(dtype="float32")
        time_s = sub["time_in_mission_s"].to_numpy(dtype="float64")
        windows, starts, ends = window_sequence(seq_mat, time_s=time_s)
        if len(windows) == 0:
            continue
        seq_all.append(windows)
        start_all.append(time_s[starts])
        end_all.append(time_s[ends])
        phase_all.append(np.array([C.PHASE_ID.get(str(p), 0)
                                   for p in sub["phase"].to_numpy(dtype=object)[ends]], dtype="float32"))
        rowend_all.append(pos[ends])
    if not seq_all:
        zero = np.empty((0, C.SEQ_LEN, len(seq_features)), dtype="float32")
        return zero, np.empty(0), np.empty(0), np.empty(0), np.empty(0, dtype=int)
    return (np.concatenate(seq_all), np.concatenate(start_all), np.concatenate(end_all),
            np.concatenate(phase_all), np.concatenate(rowend_all))
