# Tune thresholds
from __future__ import annotations

import json
import pickle
from pathlib import Path

import numpy as np

from . import config as C
from .pipeline import Pipeline
from . import training as T

REPORT_FILE = Path(C.OUT_ROOT) / "threshold_tuned.json"
SMOOTH_WIN = 0.16
PREFER = (0.20, 0.25, 0.28, 0.30, 0.36, 0.40, 0.45, 0.50)

# F1 Curve
def _f1_curve(truth, prob, grid=np.arange(0.0, 1.001, 0.01)):
    from sklearn.metrics import f1_score
    return np.array([f1_score(truth, prob > t, zero_division=0) for t in grid])

# Smoothed
def _smoothed(curve):
    w = int(round(SMOOTH_WIN / 0.01))
    return np.convolve(curve, np.ones(w) / w, mode="same")

# Pick Threshold
def _pick_threshold(grid, smooth):
    i = int(np.argmax(smooth))
    best = float(grid[i])
    for p in PREFER:
        ip = int(np.argmin(np.abs(grid - p)))
        if smooth[ip] >= smooth[i] - 0.005:
            best = float(grid[ip])
    return best

# Matrices
def _matrices(bundle, pl, split):
    st_n = pl._load("normal", split, stats=True)["stats"]
    st_f = pl._load("faulty", split, stats=True)["stats"]
    mt_f = pl._load("faulty", split, meta=True)["meta"]
    st = bundle["stat_scaler"].transform(np.vstack([st_n, st_f])).astype("float32")
    y = np.vstack([np.zeros((len(st_n), len(C.FAULTS)), dtype="float32"),
                   mt_f[:, -len(C.FAULTS):].astype("float32")])
    seq_n = pl._load("normal", split, seq=True)["seq"]
    seq_f = pl._load("faulty", split, seq=True)["seq"]
    sq = np.concatenate([seq_n, seq_f])
    sq = bundle["seq_scaler"].transform(
        sq.reshape(-1, sq.shape[-1])).reshape(*sq.shape)
    emb, prob_bin = T.lstm_predict(bundle["lstm"], sq)
    im, ir = bundle["if"]
    score_if = T.anomaly_score(im, st, ir)
    score = T.blended_anomaly(im, ir, bundle.get("anomaly_extra"), st, prob_bin)
    return {
        "st": st, "y": y, "score": score,
        "multi": T.fusion_predict(bundle["fusion"], st, score_if, emb),
    }

# Persist Overheating
def _persist_overheating(pred, n_norm):
    oj = C.FAULTS.index("overheating")
    prev = np.zeros(len(pred), dtype=bool)
    prev[1:] = pred[:-1, oj]
    prev[n_norm] = False
    pred[:, oj] = pred[:, oj] & prev
    return pred

# Tune
def tune():
    with open(Path(C.OUT_ROOT) / "artifacts" / "apet_bundle.pkl", "rb") as f:
        bundle = pickle.load(f)
    for m in ["lstm", "fusion", "quantile", "health"]:
        if bundle.get(m) is not None:
            bundle[m] = bundle[m].to(T.DEVICE)

    pl = Pipeline(out_root=C.OUT_ROOT)
    pl.assign_splits()
    vl = _matrices(bundle, pl, "val")
    te = _matrices(bundle, pl, "test")
    n_norm_te = len(pl._load("normal", "test", stats=True)["stats"])

    grid = np.arange(0.0, 1.001, 0.01)
    tuned, verified = {}, {}
    for f in C.FAULTS:
        j = C.FAULTS.index(f)
        tuned[f] = _pick_threshold(grid, _smoothed(_f1_curve(vl["y"][:, j], vl["multi"][:, j])))

    for f in C.FAULTS:
        j = C.FAULTS.index(f)
        pred = (te["multi"] > np.array([tuned[c] for c in C.FAULTS])).copy()
        _persist_overheating(pred, n_norm_te)
        from sklearn.metrics import f1_score, precision_score, recall_score
        verified[f] = {
            "threshold": tuned[f],
            "f1": float(f1_score(te["y"][:, j], pred[:, j], zero_division=0)),
            "precision": float(precision_score(te["y"][:, j], pred[:, j], zero_division=0)),
            "recall": float(recall_score(te["y"][:, j], pred[:, j], zero_division=0)),
        }
        print(f"{f:<22s} thr={tuned[f]:.3f}  testF1={verified[f]['f1']:.3f}  "
              f"P={verified[f]['precision']:.3f}  R={verified[f]['recall']:.3f}")

    REPORT_FILE.write_text(json.dumps(
        {"tuned_thresholds": tuned, "verified_on_test": verified},
        indent=2, sort_keys=True, default=float))
    print(f"\ntuned thresholds written to {REPORT_FILE}")

if __name__ == "__main__":
    tune()
