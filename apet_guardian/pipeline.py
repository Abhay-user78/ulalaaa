# Pipeline orchestration
import json
import logging
import pickle
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from . import config as C
from .features import build_physics_model
from .windowing import compute_split, window_mission_dataframe, window_stats
from . import training as T

log = logging.getLogger("apet")

# Load Columns
def _load_columns() -> List[str]:
    cols = set(C.CONTEXT_COLS) | set(C.RESIDUAL_TARGETS) | set(C.SEQ_FEATURES) | set(C.STATS_SIGNALS)
    cols |= {"global_time_s", "time_in_mission_s", "engine_id", "mission_id",
             "mission_type", "phase", "true_overall_health_index",
             "simulated_rul_hours", "true_brake_power_kW"}
    cols |= set(C.FAULT_COLUMNS)
    return sorted(cols)

# Discover
def discover(kind: str, limit: int, seed: int = C.SPLIT_SEED) -> List[Path]:
    root = C.DATA_ROOT / kind
    files = sorted(list(root.rglob("*.parquet")) + list(root.rglob("*.csv")))
    files = [f for f in files if f.name.lower().endswith((".csv", ".parquet"))]
    if not limit or limit >= len(files):
        return files[:limit] if limit else files
    if kind == "faulty":
        by_class: Dict[str, List[Path]] = {}
        for f in files:
            by_class.setdefault(f.parent.name, []).append(f)
        classes = sorted(by_class)
        rng = np.random.default_rng(seed)
        for c in classes:
            rng.shuffle(by_class[c])
        out = []
        per = max(limit // len(classes), 1)
        take = {c: min(len(v), per + (i < limit % len(classes))) for i, (c, v) in enumerate(sorted(by_class.items()))}
        for c in classes:
            out.extend(by_class[c][:take[c]])
        return sorted(out)[:limit]
    rng = np.random.default_rng(seed)
    rng.shuffle(files)
    return sorted(files[:limit])

# Read Csv
def read_csv(path: Path, cols: List[str]) -> pd.DataFrame:
    if path.suffix == ".parquet":
        available = pd.read_parquet(path, columns=None).columns.tolist()
        return pd.read_parquet(path, columns=[c for c in cols if c in available])
    return pd.read_csv(path, usecols=lambda c: c in cols, low_memory=False)

# Object Id
def _object_id(kind: str, path: Path) -> str:
    if kind == "rul":
        stem = path.stem.replace("life_", "")
        return stem if stem.startswith("engine_") else "engine_" + stem
    return path.stem

# Pipeline
class Pipeline:
    # Init
    def __init__(self, out_root: Path = C.OUT_ROOT, limits: Optional[dict] = None):
        self.out = Path(out_root)
        self.stores = self.out / "windows"
        self.artifacts = self.out / "artifacts"
        for d in [self.out, self.stores, self.artifacts]:
            d.mkdir(parents=True, exist_ok=True)
        self.limits = limits or dict(C.LIMITS)
        self.n_stats_cols = 8 * len(C.STATS_SIGNALS) + len(C.CUMULATIVE_FEATURES)
        self.n_meta_cols = 3 + 1 + 1 + 1 + 1 + len(C.FAULTS)
        self._assign: Dict[str, Dict[str, str]] = {}
        self._store_counter = {}

    # Fp
    def _fp(self, kind: str, split: str, tag: str) -> Path:
        return self.stores / f"{kind}__{split}.{tag}"

    # Append
    def _append(self, kind: str, split: str, tag: str, x: np.ndarray):
        path = self._fp(kind, split, tag)
        x = np.ascontiguousarray(np.asarray(x, dtype="float32"))
        with open(path, "ab") as f:
            x.reshape(-1).tofile(f)
        self._store_counter[(kind, split, tag)] = self._store_counter.get((kind, split, tag), 0) + len(x)

    # Reset Store
    def _reset_store(self, kind: str, split: str, tag: str):
        path = self._fp(kind, split, tag)
        if path.exists():
            path.unlink()

    # Read
    def _read(self, kind: str, split: str, tag: str, ncols: int) -> np.ndarray:
        path = self._fp(kind, split, tag)
        n = self._store_counter.get((kind, split, tag))
        if n is None:
            if not path.exists() or path.stat().st_size == 0:
                return np.empty((0, ncols), dtype="float32")
            n = path.stat().st_size // (4 * ncols)
        if not path.exists() or n == 0:
            return np.empty((0, ncols), dtype="float32")
        mm = np.memmap(path, dtype="float32", mode="r")
        return np.array(mm.reshape(n, ncols))

    # Assign Splits
    def assign_splits(self):
        assign = {}
        for kind in ["normal", "faulty", "rul"]:
            files = discover(kind, self.limits.get(kind, 0) or 10 ** 9)
            ids = [_object_id(kind, f) for f in files]
            assign[kind] = compute_split(ids, seed=C.SPLIT_SEED + 10 * {"normal": 0, "faulty": 1, "rul": 2}[kind])
        self._assign = assign
        return assign

    # Split Of
    def split_of(self, kind: str, obj_key: str) -> str:
        return self._assign[kind].get(obj_key, "test")

    # Build Physics Model
    def build_physics_model(self):
        files = discover("normal", self.limits.get("normal", 0) or 10 ** 9)
        train_files = [f for f in files if self.split_of("normal", _object_id("normal", f)) == "train"]
        frames = []
        for f in train_files[:40]:
            df = read_csv(f, _load_columns())
            if "mission_id" not in df.columns:
                df["mission_id"] = f.stem
            if "engine_id" not in df.columns:
                df["engine_id"] = f.stem
            frames.append(df)
        if not frames:
            raise RuntimeError("No healthy train rows for the physics baseline")
        healthy = pd.concat(frames, ignore_index=True).sort_values(["engine_id", "global_time_s"]).reset_index(drop=True)
        log.info("physics baseline from %d healthy rows (%d files)", len(healthy), len(frames))
        return build_physics_model(healthy)

    # Process File
    def process_file(self, kind: str, path: Path, physics):
        obj_key = _object_id(kind, path)
        split = self.split_of(kind, obj_key)
        t0 = time.time()
        df = read_csv(path, _load_columns())
        for c in ["mission_id", "engine_id", "phase"]:
            if c not in df.columns:
                df[c] = path.stem if c != "phase" else "cruise"
        if "mission_type" not in df.columns:
            df["mission_type"] = "unknown"
        num_cols = df.select_dtypes(include=[np.number]).columns
        df[num_cols] = df[num_cols].ffill().bfill()
        df = df.sort_values(["engine_id", "global_time_s"]).reset_index(drop=True)
        df = physics.transform(df)

        seq_w, start_t, end_t, phase_ids, row_end = window_mission_dataframe(df, C.SEQ_FEATURES)
        if len(seq_w) == 0:
            return 0
        stats_mat, _ = window_stats(seq_w, C.SEQ_FEATURES)
        cum = df[C.CUMULATIVE_FEATURES].to_numpy(dtype="float32")[row_end]
        stats_full = np.concatenate([stats_mat, cum], axis=1).astype("float32")

        fault_flags = np.zeros((len(seq_w), len(C.FAULTS)), dtype="float32")
        for i, fc in enumerate(C.FAULT_COLUMNS):
            if fc in df.columns:
                fault_flags[:, i] = df[fc].to_numpy(dtype="float32")[row_end]

        rul_h = df["simulated_rul_hours"].to_numpy(dtype="float32")[row_end] \
            if "simulated_rul_hours" in df.columns else np.full(len(seq_w), -1.0, dtype="float32")
        health = df["true_overall_health_index"].to_numpy(dtype="float32")[row_end] \
            if "true_overall_health_index" in df.columns else np.full(len(seq_w), -1.0, dtype="float32")
        any_fault = (fault_flags.sum(axis=1) > 0).astype("float32")

        num = np.concatenate([
            start_t[:, None], end_t[:, None], phase_ids[:, None],
            rul_h[:, None], health[:, None], any_fault[:, None],
            fault_flags.sum(axis=1, keepdims=True), fault_flags], axis=1).astype("float32")

        engines = df["engine_id"].to_numpy(dtype=object)
        missions = df["mission_id"].to_numpy(dtype=object)
        mtypes = df["mission_type"].to_numpy(dtype=object)
        phases = df["phase"].to_numpy(dtype=object)
        str_file = self._fp(kind, split, "meta.jsonl")
        with open(str_file, "a", encoding="utf-8") as f:
            for k in range(len(seq_w)):
                r = int(row_end[k])
                record = {"engine_id": str(engines[r]), "life_id": str(engines[r]),
                          "mission_id": str(missions[r]), "mission_type": str(mtypes[r]),
                          "phase": str(phases[r]), "kind": kind, "split": split}
                f.write(json.dumps(record, default=str) + "\n")

        self._append(kind, split, "stats.bin", stats_full)
        self._append(kind, split, "meta.bin", num)
        if kind in ("normal", "faulty"):
            self._append(kind, split, "seq.bin", seq_w)
        log.info("windowed %-7s %-28s -> %6d windows [%s] (%5.1fs)",
                 kind, path.name, len(seq_w), split, time.time() - t0)
        return len(seq_w)

    # Build Windows
    def build_windows(self, physics):
        total = 0
        for kind in ["normal", "faulty", "rul"]:
            for f in discover(kind, self.limits.get(kind, 0) or 10 ** 9):
                total += self.process_file(kind, f, physics)
        log.info("total windows built: %d", total)
        return total

    # Load
    def _load(self, kind, split, stats=False, seq=False, meta=False):
        col_count = {"stats": self.n_stats_cols,
                     "seq": C.SEQ_LEN * len(C.SEQ_FEATURES),
                     "meta": self.n_meta_cols}
        out = {}
        if stats:
            out["stats"] = self._read(kind, split, "stats.bin", col_count["stats"])
        if seq:
            raw = self._read(kind, split, "seq.bin", col_count["seq"])
            out["seq"] = raw.reshape(-1, C.SEQ_LEN, len(C.SEQ_FEATURES)) if len(raw) else raw.reshape(0, C.SEQ_LEN, len(C.SEQ_FEATURES))
        if meta:
            out["meta"] = self._read(kind, split, "meta.bin", col_count["meta"])
        return out

    # Stats Names
    def stats_names(self) -> List[str]:
        cols = []
        for s in C.STATS_SIGNALS:
            cols.extend([f"{s}__{st}" for st in C.STAT_NAMES])
        cols.extend([f"{c}__final" for c in C.CUMULATIVE_FEATURES])
        return cols

    # Train And Report
    def train_and_report(self):
        report = {}
        art = {}

        st_n_tr = self._load("normal", "train", stats=True)["stats"]
        st_f_tr = self._load("faulty", "train", stats=True)["stats"]
        stat_scaler = StandardScaler().fit(np.vstack([st_n_tr, st_f_tr]))
        seq_n_tr = self._load("normal", "train", seq=True)["seq"]
        seq_scaler = StandardScaler().fit(seq_n_tr.reshape(-1, seq_n_tr.shape[-1]))

        st_n_tr_s = stat_scaler.transform(st_n_tr)
        log.info("IsolationForest on %d healthy train windows", len(st_n_tr))
        if_model, if_ref = T.fit_isolated_forest(st_n_tr_s)
        art["if"] = (if_model, if_ref)
        anomaly_extra = T.fit_pca_anomaly(st_n_tr_s)
        art["anomaly_extra"] = anomaly_extra
        ref_d = if_ref if len(if_ref) else np.array([0.0])
        report["if_state"] = {
            "healthy_train_windows": int(len(st_n_tr)),
            "ref_p1": float(np.percentile(ref_d, 1)),
            "ref_median": float(np.percentile(ref_d, 50)),
        }

        meta_n_tr = self._load("normal", "train", meta=True)["meta"]
        meta_f_tr = self._load("faulty", "train", meta=True)["meta"]
        seq_f_tr = self._load("faulty", "train", seq=True)["seq"]
        y_f_mat = meta_f_tr[:, -len(C.FAULTS):]
        rng = np.random.default_rng(C.FAULT_RNG_SEED)
        cap_n = min(len(seq_n_tr), 40000)
        cap_f = min(len(seq_f_tr), 80000)
        ix_n = np.sort(rng.choice(len(seq_n_tr), cap_n, replace=False))
        ix_f = np.sort(rng.choice(len(y_f_mat), cap_f, replace=False))

        seq_bin_tr = np.concatenate([seq_n_tr[ix_n], seq_f_tr[ix_f]])
        y_bin_tr = np.concatenate([np.zeros(cap_n),
                                   np.clip(meta_f_tr[ix_f, 6], 0, 1)])

        st_bin_tr = np.vstack([st_n_tr[ix_n], st_f_tr[ix_f]])
        y_bin_mat = np.vstack([np.zeros((cap_n, len(C.FAULTS))), y_f_mat[ix_f]])

        seq_n_vl = self._load("normal", "val", seq=True)["seq"]
        seq_f_vl = self._load("faulty", "val", seq=True)["seq"]
        st_n_vl = self._load("normal", "val", stats=True)["stats"]
        st_f_vl = self._load("faulty", "val", stats=True)["stats"]
        meta_f_vl = self._load("faulty", "val", meta=True)["meta"]

        cap_v = 4000
        kv_n = min(cap_v, len(seq_n_vl), len(st_n_vl))
        kv_f = min(cap_v, len(seq_f_vl), len(st_f_vl))
        seq_bin_vl = np.concatenate([seq_n_vl[:kv_n], seq_f_vl[:kv_f]])
        y_bin_vl = np.concatenate([np.zeros(kv_n), np.clip(meta_f_vl[:kv_f, 6], 0, 1)])
        st_bin_vl = np.vstack([st_n_vl[:kv_n], st_f_vl[:kv_f]])
        y_bin_vl_mat = np.vstack([np.zeros((kv_n, len(C.FAULTS))),
                                  meta_f_vl[:kv_f, -len(C.FAULTS):]])

        seq_tr_n = seq_scaler.transform(seq_bin_tr.reshape(-1, seq_bin_tr.shape[-1])).reshape(*seq_bin_tr.shape)
        seq_vl_n = seq_scaler.transform(seq_bin_vl.reshape(-1, seq_bin_vl.shape[-1])).reshape(*seq_bin_vl.shape)

        log.info("Bi-LSTM binary (%d train / %d val windows)", len(seq_tr_n), len(seq_vl_n))
        lstm = T.fit_bilstm(seq_tr_n, y_bin_tr, seq_vl_n, y_bin_vl, seq_scaler)
        lstm_model = lstm["model"]
        emb_tr, prob_tr = T.lstm_predict(lstm_model, seq_tr_n)
        emb_vl, prob_vl = T.lstm_predict(lstm_model, seq_vl_n)
        report["bilstm_val"] = T.evaluate_binary(y_bin_vl, prob_vl)
        report["bilstm_train_val_loss"] = lstm["val_loss"]
        art["lstm"] = lstm_model
        art["emb_dim"] = lstm_model.embedding_dim

        log.info("classical multi-label on %d train windows", len(st_bin_tr))
        classical = T.fit_classical(stat_scaler.transform(st_bin_tr), y_bin_mat)
        art["classical"] = classical["model"]

        score_tr = T.anomaly_score(if_model, stat_scaler.transform(st_bin_tr), if_ref)
        score_vl = T.anomaly_score(if_model, stat_scaler.transform(st_bin_vl), if_ref)
        log.info("fusion multi-label on %d train windows", len(st_bin_tr))
        fusion_w = T.class_balances(y_bin_mat)
        fusion = T.fit_fusion(stat_scaler.transform(st_bin_tr), score_tr, emb_tr, y_bin_mat,
                              stat_scaler.transform(st_bin_vl), score_vl, emb_vl, y_bin_vl_mat,
                              class_weights=fusion_w)
        art["fusion"] = fusion["model"]
        report["fusion_train_val_loss"] = fusion["val_loss"]

        qr = hr = None
        st_r_tr = self._load("rul", "train", stats=True, meta=True)
        st_r_vl = self._load("rul", "val", stats=True, meta=True)
        if len(st_r_tr["stats"]) and len(st_r_vl["stats"]):
            rulY_tr = st_r_tr["meta"][:, 3]
            rulY_vl = st_r_vl["meta"][:, 3]
            hY_tr = st_r_tr["meta"][:, 4]
            hY_vl = st_r_vl["meta"][:, 4]
            ok_tr = np.isfinite(rulY_tr) & (rulY_tr >= 0) & np.isfinite(hY_tr)
            ok_vl = np.isfinite(rulY_vl) & (rulY_vl >= 0) & np.isfinite(hY_vl)
            if ok_tr.sum() > 200 and ok_vl.sum() > 20:
                Xr_tr = stat_scaler.transform(st_r_tr["stats"][ok_tr])
                Xr_vl = stat_scaler.transform(st_r_vl["stats"][ok_vl])
                yr_tr = np.clip(rulY_tr[ok_tr], 0, C.RUL_MAX_TARGET_H)
                yr_vl = np.clip(rulY_vl[ok_vl], 0, C.RUL_MAX_TARGET_H)
                log.info("quantile RUL regressor (%d train)", len(Xr_tr))
                qr = T.fit_quantile(Xr_tr, yr_tr, Xr_vl, yr_vl)
                log.info("health-index regressor (%d train)", len(Xr_tr))
                hr = T.fit_health(Xr_tr, hY_tr[ok_tr], Xr_vl, hY_vl[ok_vl])
                art["quantile"] = qr["model"]
                art["health"] = hr["model"]
                report["rul_val"] = T.evaluate_regression(
                    yr_vl, T.quantile_predict(qr["model"], Xr_vl), quantile=True)
                report["health_val"] = T.evaluate_regression(
                    hY_vl[ok_vl], T.health_predict(hr["model"], Xr_vl), quantile=False)
        art["stat_scaler"] = stat_scaler
        art["seq_scaler"] = seq_scaler
        art["physics"] = self.build_physics_model()
        art["seq_features"] = C.SEQ_FEATURES
        art["stats_names"] = self.stats_names()
        art["faults"] = C.FAULTS

        report["test"] = self.evaluate_all(art, seq_scaler)

        report["history"] = {
            "lstm_val_loss": lstm["val_loss"],
            "fusion_val_loss": fusion["val_loss"],
            "classical_algo": "RandomForest MultiOutputClassifier",
            "quantile_algo": "MLP pinball (0.10/0.50/0.90)",
            "health_algo": "MLP MSE",
        }
        with open(self.artifacts / "apet_bundle.pkl", "wb") as f:
            pickle.dump(art, f, protocol=5)
        (self.out / "report.json").write_text(json.dumps(report, indent=2, default=str))
        log.info("report written to %s", self.out / "report.json")
        return report

    # Evaluate All
    def evaluate_all(self, art, seq_scaler):
        res = {}
        st_n_te = self._load("normal", "test", stats=True)["stats"]
        st_f_te = self._load("faulty", "test", stats=True)["stats"]
        seq_n_te = self._load("normal", "test", seq=True)["seq"]
        seq_f_te = self._load("faulty", "test", seq=True)["seq"]
        meta_f_te = self._load("faulty", "test", meta=True)["meta"]

        label_any = np.concatenate([np.zeros(len(st_n_te)), np.clip(meta_f_te[:, 6], 0, 1)])
        st_te = np.vstack([st_n_te, st_f_te])
        st_te_s = art["stat_scaler"].transform(st_te)

        seq_te = np.concatenate([seq_n_te, seq_f_te])
        seq_te_n = seq_scaler.transform(seq_te.reshape(-1, seq_te.shape[-1])).reshape(*seq_te.shape)
        emb_te, prob_bin = T.lstm_predict(art["lstm"], seq_te_n)

        if_model, if_ref = art["if"]
        score_te = T.anomaly_score(if_model, st_te_s, if_ref)
        res["anomaly_vs_injected_faults"] = T.evaluate_anomaly(label_any, score_te)
        res["anomaly_versus_barely_active"] = res["anomaly_vs_injected_faults"]
        if art.get("anomaly_extra") is not None:
            res["anomaly_versus_barely_active_blended"] = T.evaluate_anomaly(
                label_any, T.blended_anomaly(if_model, if_ref, art["anomaly_extra"],
                                             st_te_s, prob_bin))

        res["bilstm_test"] = T.evaluate_binary(label_any, prob_bin)

        y_mat = np.vstack([np.zeros((len(st_n_te), len(C.FAULTS))),
                           meta_f_te[:, -len(C.FAULTS):]])
        res["classical_fault_test"] = T.evaluate_multilabel(
            y_mat, T.classical_predict(art["classical"], st_te_s), C.FAULTS)
        fu = T.fusion_predict(art["fusion"], st_te_s, score_te, emb_te)
        res["fusion_fault_test"] = T.evaluate_multilabel(y_mat, fu, C.FAULTS)

        st_r_te = self._load("rul", "test", stats=True, meta=True)
        if len(st_r_te["stats"]):
            m = st_r_te["meta"]
            Xr_te = art["stat_scaler"].transform(st_r_te["stats"])
            score_rul = T.anomaly_score(if_model, Xr_te, if_ref)
            if art.get("anomaly_extra") is not None:
                score_rul = T.blended_anomaly(if_model, if_ref, art["anomaly_extra"], Xr_te)
            h = m[:, 3]
            ok = np.isfinite(h) & (h >= 0)
            near_eol = ok & (h < 2.0)
            start_ok = ok
            res["unknown_degradation_anomaly"] = {
                "test_life_windows": int(start_ok.sum()),
                "near_eol_median_anomaly": float(np.median(score_rul[near_eol])) if near_eol.sum() else None,
                "all_life_median_anomaly": float(np.median(score_rul[start_ok])) if start_ok.sum() else None,
            }
            if "quantile" in art:
                ru = np.clip(h[ok], 0, C.RUL_MAX_TARGET_H)
                hh = m[:, 4][ok]
                res["rul_test"] = T.evaluate_regression(
                    ru, T.quantile_predict(art["quantile"], Xr_te[ok]), quantile=True)
                res["health_test"] = T.evaluate_regression(
                    hh, T.health_predict(art["health"], Xr_te[ok]), quantile=False)
        return res
