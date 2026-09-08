# Training routines
import json
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.ensemble import IsolationForest, RandomForestClassifier, \
    HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score, average_precision_score, hamming_loss, \
    roc_auc_score, f1_score, mean_absolute_error, mean_squared_error, r2_score, precision_score
from sklearn.multioutput import MultiOutputClassifier

from . import config as C
from .models import BiLSTMEncoder, FaultFusionMLP, HealthMLP, QuantileMLP, \
    gradient_features

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Curate
def _curate(y: np.ndarray) -> np.ndarray:
    y = np.asarray(y, dtype="float32")
    return np.nan_to_num(y)

# Fit Isolated Forest
def fit_isolated_forest(healthy_stats: np.ndarray):
    X = healthy_stats.astype("float32")
    if len(X) > C.IF_MAX_SAMPLES:
        rng = np.random.default_rng(C.FAULT_RNG_SEED)
        X = X[rng.choice(len(X), C.IF_MAX_SAMPLES, replace=False)]
    model = IsolationForest(n_estimators=C.IF_N_ESTIMATORS,
                            contamination=C.IF_CONTAMINATION,
                            max_samples=min(4096, len(X)),
                            random_state=C.FAULT_RNG_SEED).fit(X)
    ref_decision = model.decision_function(X)
    return model, ref_decision

# Anomaly Score
def anomaly_score(model, X: np.ndarray, ref_decision: np.ndarray) -> np.ndarray:
    d = model.decision_function(X)
    lo = float(np.percentile(ref_decision, 1))
    mid = float(np.percentile(ref_decision, 50))
    scale = max(mid - lo, 1e-6)
    return np.clip((mid - d) / scale, 0.0, 1.0)

# Class Balances
def class_balances(y: np.ndarray, cap: float = 20.0) -> np.ndarray:
    y = np.asarray(y, dtype="float64")
    pos = y.mean(axis=0)
    return np.where(pos > 0.0, np.minimum((1.0 - pos) / np.maximum(pos, 1e-9), cap), 1.0)

# Pca Resid
def _pca_resid(pca, X: np.ndarray) -> np.ndarray:
    Xr = pca.inverse_transform(pca.transform(X))
    return np.sqrt(np.maximum(((X - Xr) ** 2).sum(axis=1), 0.0))

# Fit Pca Anomaly
def fit_pca_anomaly(healthy_stats: np.ndarray, n_components: int = 32) -> Dict:
    from sklearn.decomposition import PCA
    X = healthy_stats.astype("float32")
    pca = PCA(n_components=min(n_components, X.shape[1]),
              random_state=C.FAULT_RNG_SEED).fit(X)
    resid = _pca_resid(pca, X)
    return {
        "pca": pca,
        "ref_p1": float(np.percentile(resid, 1)),
        "ref_p50": float(np.percentile(resid, 50)),
        "w_if": 0.30,
        "w_pca": 0.15,
        "w_lstm": 0.55,
    }

# Pca Anomaly Score
def pca_anomaly_score(extra: Dict, X: np.ndarray) -> np.ndarray:
    r = _pca_resid(extra["pca"], X)
    lo, mid = float(extra["ref_p1"]), float(extra["ref_p50"])
    scale = max(mid - lo, 1e-6)
    return np.clip((r - mid) / scale, 0.0, 1.0)

# Blended Anomaly
def blended_anomaly(if_model, if_ref, extra: Optional[Dict], X: np.ndarray,
                    lstm_prob: Optional[np.ndarray] = None) -> np.ndarray:
    a = anomaly_score(if_model, X, if_ref)
    if extra is None:
        return a
    b = pca_anomaly_score(extra, X)
    if lstm_prob is None:
        w = extra["w_if"] + extra["w_pca"]
        return (extra["w_if"] * a + extra["w_pca"] * b) / max(w, 1e-6)
    return extra["w_if"] * a + extra["w_pca"] * b + extra["w_lstm"] * lstm_prob

# Weighted Cross Entropy
def _weighted_cross_entropy(y: np.ndarray) -> Optional[torch.Tensor]:
    pos = float(np.mean(y))
    if pos <= 0.0 or pos >= 1.0:
        return None
    return torch.tensor((1.0 - pos) / max(pos, 1e-6), dtype=torch.float32, device=DEVICE)

# Fit Bilstm
def fit_bilstm(seq_train: np.ndarray, y_train: np.ndarray,
               seq_val: np.ndarray, y_val: np.ndarray,
               seq_scaler) -> Dict:
    from sklearn.preprocessing import StandardScaler
    n_ch = seq_train.shape[-1]
    model = BiLSTMEncoder(n_ch, C.LSTM_HIDDEN, C.LSTM_LAYERS, C.LSTM_DROPOUT).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=C.LEARNING_RATE, weight_decay=C.WEIGHT_DECAY)
    w = _weighted_cross_entropy(y_train)
    crit = nn.BCEWithLogitsLoss(pos_weight=w)

    # Evaluate
    def evaluate(seq, y, batch=C.LSTM_BATCH):
        model.eval()
        tot, cnt = 0.0, 0
        with torch.no_grad():
            for i in range(0, len(seq), batch):
                xb = torch.from_numpy(seq[i:i + batch]).float().to(DEVICE)
                yb = torch.from_numpy(y[i:i + batch]).float().to(DEVICE)
                _, logit = model(xb)
                tot += crit(logit, yb).item() * len(yb)
                cnt += len(yb)
        return tot / max(cnt, 1)

    best_val = float("inf")
    best_state = None
    patience = 0
    history = []
    for ep in range(C.EPOCHS):
        model.train()
        perm = np.random.default_rng(C.FAULT_RNG_SEED + ep).permutation(len(seq_train))
        tl = 0.0
        for i in range(0, len(seq_train), C.LSTM_BATCH):
            idx = perm[i:i + C.LSTM_BATCH]
            perm_i = np.sort(idx)
            xb = torch.from_numpy(seq_train[perm_i]).float().to(DEVICE)
            yb = torch.from_numpy(y_train[perm_i]).float().to(DEVICE)
            opt.zero_grad()
            _, logit = model(xb)
            loss = crit(logit, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            tl += loss.item() * len(perm_i)
        vl = evaluate(seq_val, y_val)
        history.append({"epoch": ep + 1, "train_loss": tl / len(seq_train), "val_loss": vl})
        if vl < best_val - 1e-4:
            best_val = vl
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience = 0
        else:
            patience += 1
            if patience >= C.PATIENCE:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    return {"model": model, "history": history, "val_loss": best_val}

# Lstm Predict
def lstm_predict(model: nn.Module, seq: np.ndarray, batch: int = 4096) -> Tuple[np.ndarray, np.ndarray]:
    model.eval()
    embs, probs = [], []
    with torch.no_grad():
        for i in range(0, len(seq), batch):
            xb = torch.from_numpy(seq[i:i + batch]).float().to(DEVICE)
            emb, logit = model(xb)
            embs.append(emb.cpu().numpy())
            probs.append(torch.sigmoid(logit).cpu().numpy())
    return np.concatenate(embs), np.concatenate(probs)

# Multi Focal Loss
def _multi_focal_loss(logit, y, alpha: float = 0.25, gamma: float = 2.0):
    p = torch.sigmoid(logit)
    pt = p * y + (1.0 - p) * (1.0 - y)
    alpha_t = alpha * y + (1.0 - alpha) * (1.0 - y)
    bce = torch.nn.functional.binary_cross_entropy_with_logits(
        logit, y, reduction="none")
    return torch.mean(alpha_t * (1.0 - pt).pow(gamma) * bce)

# Fit Fusion
def fit_fusion(stats_train: np.ndarray, score_train: np.ndarray, emb_train: np.ndarray,
               y_train: np.ndarray, stats_val: np.ndarray, score_val: np.ndarray,
               emb_val: np.ndarray, y_val: np.ndarray, class_weights: Optional[np.ndarray] = None,
               loss: str = "bce", focal_gamma: float = 2.0) -> Dict:
    model = FaultFusionMLP(stats_train.shape[-1], emb_train.shape[-1],
                           len(C.FAULTS), C.FUSION_HIDDEN, C.FUSION_DROPOUT).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=C.LEARNING_RATE, weight_decay=C.WEIGHT_DECAY)
    if loss == "focal":
        # Crit
        def crit(logit, y):
            return _multi_focal_loss(logit, y, gamma=focal_gamma)
    else:
        kw = {}
        if class_weights is not None:
            kw["pos_weight"] = torch.tensor(
                np.asarray(class_weights, dtype="float32"), dtype=torch.float32, device=DEVICE)
        crit = nn.BCEWithLogitsLoss(**kw)

    # T
    def _t(x): return torch.from_numpy(x).float().to(DEVICE)

    # Evaluate
    def evaluate(stats, score, emb, y):
        model.eval()
        tot, cnt = 0.0, 0
        with torch.no_grad():
            for i in range(0, len(stats), 2048):
                x = (torch.from_numpy(stats[i:i + 2048]).float().to(DEVICE),
                     torch.from_numpy(score[i:i + 2048]).float().to(DEVICE),
                     torch.from_numpy(emb[i:i + 2048]).float().to(DEVICE))
                yb = _t(y[i:i + 2048])
                logit = model(*x)
                tot += crit(logit, yb).item() * len(yb)
                cnt += len(yb)
        return tot / max(cnt, 1)

    best_val, best_state, patience, history = float("inf"), None, 0, []
    Xtr = (torch.from_numpy(stats_train).float().to(DEVICE),
           torch.from_numpy(score_train).float().to(DEVICE),
           torch.from_numpy(emb_train).float().to(DEVICE))
    ytr = _t(y_train)
    for ep in range(C.EPOCHS):
        model.train()
        perm = np.random.default_rng(C.FAULT_RNG_SEED + ep).permutation(len(stats_train))
        tl = 0.0
        for i in range(0, len(stats_train), 2048):
            idx = np.sort(perm[i:i + 2048])
            opt.zero_grad()
            logit = model(Xtr[0][idx], Xtr[1][idx], Xtr[2][idx])
            loss = crit(logit, ytr[idx])
            loss.backward()
            opt.step()
            tl += loss.item() * len(idx)
        vl = evaluate(stats_val, score_val, emb_val, y_val)
        history.append({"epoch": ep + 1, "train_loss": tl / len(stats_train), "val_loss": vl})
        if vl < best_val - 1e-4:
            best_val, best_state, patience = vl, {k: v.cpu().clone() for k, v in model.state_dict().items()}, 0
        else:
            patience += 1
            if patience >= C.PATIENCE:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    return {"model": model, "history": history, "val_loss": best_val}

# Fusion Predict
def fusion_predict(model: nn.Module, stats: np.ndarray, score: np.ndarray,
                   emb: np.ndarray, batch: int = 4096) -> np.ndarray:
    model.eval()
    probs = []
    with torch.no_grad():
        for i in range(0, len(stats), batch):
            x = (torch.from_numpy(stats[i:i + batch]).float().to(DEVICE),
                 torch.from_numpy(score[i:i + batch]).float().to(DEVICE),
                 torch.from_numpy(emb[i:i + batch]).float().to(DEVICE))
            probs.append(torch.sigmoid(model(*x)).cpu().numpy())
    return np.concatenate(probs)

# Fit Classical
def fit_classical(stats_train: np.ndarray, y_train: np.ndarray) -> Dict:
    base = RandomForestClassifier(
        n_estimators=C.CLASSICAL_ESTIMATORS, class_weight="balanced_subsample", n_jobs=-1,
        random_state=C.FAULT_RNG_SEED)
    model = MultiOutputClassifier(base, n_jobs=2).fit(stats_train, y_train)
    return {"model": model}

# Classical Predict
def classical_predict(model, stats: np.ndarray) -> np.ndarray:
    return packed_proba(model, stats)

# Fit Pre-Fault Gbm
def fit_preault_gbm(stats_train: np.ndarray, y_train: np.ndarray) -> Dict:
    models = []
    for i in range(y_train.shape[1]):
        base = HistGradientBoostingClassifier(
            max_iter=C.HIST_GBM_MAX_ITER,
            learning_rate=C.HIST_GBM_LEARNING_RATE,
            max_depth=C.HIST_GBM_MAX_DEPTH,
            class_weight="balanced",
            random_state=C.FAULT_RNG_SEED)
        base.fit(stats_train, y_train[:, i])
        models.append(base)
    return {"model": models}

# Pre-Fault Gbm Predict
def preault_gbm_predict(model, stats: np.ndarray) -> np.ndarray:
    probs = np.column_stack([m.predict_proba(stats)[:, 1] if m.classes_.shape[0] > 1
                              else np.zeros(len(stats)) for m in model])
    return probs

# Fit Tabular
def _fit_tabular(model: nn.Module, Xt: np.ndarray, yt: np.ndarray,
                 Xv: np.ndarray, yv: np.ndarray, loss_fn, name: str) -> Dict:
    opt = torch.optim.Adam(model.parameters(), lr=C.LEARNING_RATE, weight_decay=C.WEIGHT_DECAY)
    model = model.to(DEVICE)
    Xt_t = torch.from_numpy(Xt).float().to(DEVICE)
    yt_t = torch.from_numpy(yt).float().to(DEVICE)
    Xv_t = torch.from_numpy(Xv).float().to(DEVICE)
    yv_t = torch.from_numpy(yv).float().to(DEVICE)

    # Evaluate
    def evaluate(X, y):
        model.eval()
        with torch.no_grad():
            yp = model(X)
        return float(loss_fn(yp, y).item())

    best_val, best_state, patience, history = float("inf"), None, 0, []
    n = len(Xt)
    for ep in range(C.EPOCHS):
        model.train()
        perm = np.random.default_rng(C.FAULT_RNG_SEED + ep).permutation(n)
        total, cnt = 0.0, 0
        for i in range(0, n, C.BATCH_SIZE):
            idx = np.sort(perm[i:i + C.BATCH_SIZE])
            opt.zero_grad()
            pred = model(Xt_t[idx])
            loss = loss_fn(pred, yt_t[idx])
            loss.backward()
            opt.step()
            total += loss.item() * len(idx)
            cnt += len(idx)
        vl = evaluate(Xv_t, yv_t)
        history.append({"epoch": ep + 1, "train_loss": total / max(cnt, 1), "val_loss": vl})
        if vl < best_val - 1e-5:
            best_val = vl
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience = 0
        else:
            patience += 1
            if patience >= C.PATIENCE:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    return {"model": model, "history": history, "val_loss": best_val}

# Fit Quantile
def fit_quantile(Xt, yt, Xv, yv) -> Dict:
    model = QuantileMLP(Xt.shape[-1], C.QUANTILES, C.REG_HIDDEN, C.REG_DROPOUT)

    # Loss Fn
    def loss_fn(preds, y):
        y = y.float()
        return sum(_pinball_torch_local(p, y, q) for p, q in zip(preds, C.QUANTILES))
    return _fit_tabular(model, Xt, yt, Xv, yv, loss_fn, "quantile")

# Pinball Torch Local
def _pinball_torch_local(pred, target, q):
    err = target - pred
    return torch.mean(torch.maximum(q * err, (q - 1.0) * err))

# Fit Health
def fit_health(Xt, yt, Xv, yv) -> Dict:
    model = HealthMLP(Xt.shape[-1], C.REG_HIDDEN, C.REG_DROPOUT)

    # Loss Fn
    def loss_fn(pred, y):
        return torch.mean((pred - y.float()) ** 2)
    return _fit_tabular(model, Xt, yt, Xv, yv, loss_fn, "health")

# Quantile Predict
def quantile_predict(model: nn.Module, X: np.ndarray) -> np.ndarray:
    model.eval()
    Xt = torch.from_numpy(X).float().to(DEVICE)
    with torch.no_grad():
        preds = model(Xt)
    y = torch.stack([p for p in preds], dim=1).cpu().numpy()
    mid = len(C.QUANTILES) // 2
    lower = np.maximum(y[:, 0], 0.0)
    upper = np.maximum(y[:, 2], lower)
    mean = np.maximum(y[:, 1], 0.0)
    return np.stack([lower, mean, upper], axis=1)

# Health Predict
def health_predict(model: nn.Module, X: np.ndarray) -> np.ndarray:
    model.eval()
    Xt = torch.from_numpy(X).float().to(DEVICE)
    with torch.no_grad():
        y = model(Xt).cpu().numpy()
    return np.clip(y, 0.0, 100.0)

# Packed Proba
def packed_proba(multiout, X: np.ndarray, batch: int = 8192) -> np.ndarray:
    probs = []
    for est in multiout.estimators_:
        p = est.predict_proba(X)
        probs.append(p[:, 1] if p.shape[1] > 1 else np.zeros(len(X)))
    return np.stack(probs, axis=1)

# Roc Can
def roc_can(y_true, y_prob):
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    if len(np.unique(y_true)) < 2:
        return float("nan")
    return float(roc_auc_score(y_true, y_prob))

# Evaluate Anomaly
def evaluate_anomaly(y_fault: np.ndarray, score: np.ndarray) -> Dict:
    y = np.asarray(y_fault, dtype=int)
    score = np.asarray(score, dtype="float64")
    d = {"anomaly_auc_roc": roc_can(y, score),
         "anomaly_auc_pr": float(average_precision_score(y, score)) if len(np.unique(y)) > 1 else float("nan")}
    prec = []
    rec = []
    for t in [0.0, 0.1, 0.2, 0.5, 0.8, 1.0]:
        pred = (score > t).astype(int)
        tp = float(np.sum((pred == 1) & (y == 1)))
        fp = float(np.sum((pred == 1) & (y == 0)))
        fn = float(np.sum((pred == 0) & (y == 1)))
        prec.append(tp / max(tp + fp, 1e-9))
        rec.append(tp / max(tp + fn, 1e-9))
    d["anomaly_precision_at_thresholds"] = prec
    d["anomaly_recall_at_thresholds"] = rec
    return d

# Evaluate Binary
def evaluate_binary(y, prob, threshold: float = 0.5) -> Dict:
    y = np.asarray(y, dtype=int)
    prob = np.asarray(prob, dtype="float64")
    pred = (prob > threshold).astype(int)
    return {
        "binary_auc_roc": roc_can(y, prob),
        "binary_accuracy": float(accuracy_score(y, pred)),
        "binary_f1": float(f1_score(y, pred, zero_division=0)),
        "binary_precision": float(precision_score(y, pred, zero_division=0)),
    }

# Evaluate Multilabel
def evaluate_multilabel(y_true: np.ndarray, prob: np.ndarray, labels: List[str]) -> Dict:
    y_true = np.asarray(y_true)
    prob = np.asarray(prob)
    pred = (prob > 0.5).astype(int)
    aucs = {}
    for i, lab in enumerate(labels):
        aucs[lab] = roc_can(y_true[:, i], prob[:, i])
    aucs["macro_avg"] = float(np.nanmean(list(filter(
        lambda v: not np.isnan(v), aucs.values()))) if any(
        not np.isnan(v) for v in aucs.values()) else float("nan"))
    return {
        "per_label_auc": aucs,
        "macro_auc": aucs.get("macro_avg"),
        "exact_match": float(np.mean(np.all(pred == y_true, axis=1))),
        "hamming_loss": float(hamming_loss(y_true, pred)),
        "macro_f1": float(f1_score(y_true, pred, average="macro", zero_division=0)),
        "micro_f1": float(f1_score(y_true, pred, average="micro", zero_division=0)),
    }

# Evaluate Regression
def evaluate_regression(y_true, y_pred, quantile: bool = False) -> Dict:
    yt = np.asarray(y_true, dtype="float64")
    if quantile:
        yp = np.asarray(y_pred, dtype="float64")
        mean_p = yp[:, 1]
        lo, hi = yp[:, 0], yp[:, 2]
        out = {
            "mae": float(mean_absolute_error(yt, mean_p)),
            "rmse": float(np.sqrt(mean_squared_error(yt, mean_p))),
            "r2": float(r2_score(yt, mean_p)),
            "picp": float(np.mean((yt >= lo) & (yt <= hi))),
            "mean_interval_width_h": float(np.mean(hi - lo)),
        }
    else:
        yp = np.asarray(y_pred, dtype="float64").ravel()
        out = {
            "mae": float(mean_absolute_error(yt, yp)),
            "rmse": float(np.sqrt(mean_squared_error(yt, yp))),
            "r2": float(r2_score(yt, yp)),
        }
    return out
