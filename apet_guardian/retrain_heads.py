# Head retrain
import json
import pickle
from pathlib import Path

import numpy as np
import torch

from . import config as C
from .pipeline import Pipeline
from . import training as T

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load Old
def _load_old():
    p = Path(C.OUT_ROOT) / "artifacts" / "apet_bundle.pkl"
    with open(p, "rb") as f:
        art = pickle.load(f)
    for m in ["lstm", "fusion", "quantile", "health"]:
        if art.get(m) is not None:
            art[m].to(DEVICE)
    return art, p

# Main
def main():
    import logging
    logging.basicConfig(level=logging.INFO, format="%(levelname)-7s %(name)s %(message)s")
    log = logging.getLogger("retrain_heads")

    old, old_p = _load_old()
    pl = Pipeline(out_root=C.OUT_ROOT)
    pl.assign_splits()

    from sklearn.metrics import roc_auc_score, f1_score

    # Auc
    def auc(y, p):
        if len(np.unique(y)) < 2 or np.ptp(p) < 1e-12:
            return float("nan")
        return float(roc_auc_score(y, p))

    st_n_tr = pl._load("normal", "train", stats=True)["stats"]
    st_f_tr = pl._load("faulty", "train", stats=True)["stats"]
    st_scaler = old["stat_scaler"]
    st_n_tr_s = st_scaler.transform(st_n_tr).astype("float32")
    st_f_tr_s = st_scaler.transform(st_f_tr).astype("float32")

    seq_n_tr = pl._load("normal", "train", seq=True)["seq"]
    seq_f_tr = pl._load("faulty", "train", seq=True)["seq"]
    seq_scaler = old["seq_scaler"]
    meta_f_tr = pl._load("faulty", "train", meta=True)["meta"]

    emb_n_tr, _ = T.lstm_predict(old["lstm"], seq_scaler.transform(
        seq_n_tr.reshape(-1, seq_n_tr.shape[-1])).reshape(*seq_n_tr.shape))
    Nn, Nf = len(emb_n_tr), len(seq_f_tr)
    emb_f_tr, _ = T.lstm_predict(old["lstm"], seq_scaler.transform(
        seq_f_tr.reshape(-1, seq_f_tr.shape[-1])).reshape(*seq_f_tr.shape))
    log.info("train embeddings: normal=%d faulty=%d", Nn, Nf)

    st_tr = np.vstack([st_n_tr_s, st_f_tr_s])
    emb_tr = np.vstack([emb_n_tr, emb_f_tr])
    y_tr = np.vstack([np.zeros((Nn, len(C.FAULTS)), dtype="float32"),
                      meta_f_tr[:, -len(C.FAULTS):].astype("float32")])

    if_model, if_ref = T.fit_isolated_forest(st_n_tr_s)
    anomaly_extra = T.fit_pca_anomaly(st_n_tr_s)
    log.info("IsolationForest + PCA residual anomaly stages fitted on %d healthy windows",
             len(st_n_tr_s))

    st_n_vl = pl._load("normal", "val", stats=True)["stats"]
    st_f_vl = pl._load("faulty", "val", stats=True)["stats"]
    seq_n_vl = pl._load("normal", "val", seq=True)["seq"]
    seq_f_vl = pl._load("faulty", "val", seq=True)["seq"]
    meta_f_vl = pl._load("faulty", "val", meta=True)["meta"]

    st_vl = np.vstack([st_scaler.transform(st_n_vl).astype("float32"),
                       st_scaler.transform(st_f_vl).astype("float32")])
    emb_n_vl, _ = T.lstm_predict(old["lstm"], seq_scaler.transform(
        seq_n_vl.reshape(-1, seq_n_vl.shape[-1])).reshape(*seq_n_vl.shape))
    emb_f_vl, _ = T.lstm_predict(old["lstm"], seq_scaler.transform(
        seq_f_vl.reshape(-1, seq_f_vl.shape[-1])).reshape(*seq_f_vl.shape))
    emb_vl = np.vstack([emb_n_vl, emb_f_vl])
    y_vl = np.vstack([np.zeros((len(emb_n_vl), len(C.FAULTS)), dtype="float32"),
                      meta_f_vl[:, -len(C.FAULTS):].astype("float32")])

    score_tr = T.anomaly_score(if_model, st_tr, if_ref)
    score_vl = T.anomaly_score(if_model, st_vl, if_ref)
    cw = T.class_balances(y_tr)
    log.info("per-class weights: %s",
             ", ".join("%s=%.2f" % (f, w) for f, w in zip(C.FAULTS, cw)))

    fusion = T.fit_fusion(st_tr, score_tr, emb_tr, y_tr, st_vl, score_vl, emb_vl, y_vl,
                          class_weights=None, loss="focal", focal_gamma=2.0)
    log.info("fusion head fitted (val loss %.4f)", fusion["val_loss"])

    classical = T.fit_classical(st_tr, y_tr)
    log.info("classical head refitted")

    art = dict(old)
    art["if"] = (if_model, if_ref)
    art["anomaly_extra"] = anomaly_extra
    art["fusion"] = fusion["model"].to(DEVICE)
    art["classical"] = classical["model"]
    out = Path(C.OUT_ROOT) / "artifacts" / "apet_bundle_v2.pkl"
    with open(out, "wb") as f:
        pickle.dump(art, f, protocol=5)
    log.info("v2 bundle written: %s", out)

    st_n_te = pl._load("normal", "test", stats=True)["stats"]
    st_f_te = pl._load("faulty", "test", stats=True)["stats"]
    seq_n_te = pl._load("normal", "test", seq=True)["seq"]
    seq_f_te = pl._load("faulty", "test", seq=True)["seq"]
    meta_f_te = pl._load("faulty", "test", meta=True)["meta"]

    st_te = np.vstack([st_scaler.transform(st_n_te).astype("float32"),
                       st_scaler.transform(st_f_te).astype("float32")])
    y_te = np.vstack([np.zeros((len(st_n_te), len(C.FAULTS)), dtype="float32"),
                      meta_f_te[:, -len(C.FAULTS):].astype("float32")])
    active_any = y_te.sum(axis=1) > 0
    label_any = np.concatenate([np.zeros(len(st_n_te)),
                                np.clip(meta_f_te[:, 6], 0, 1)])

    seq_te = np.concatenate([seq_n_te, seq_f_te])
    seq_te_s = seq_scaler.transform(seq_te.reshape(-1, seq_te.shape[-1])).reshape(*seq_te.shape)
    emb_te, prob_bin = T.lstm_predict(old["lstm"], seq_te_s)

    res = {"old": {}, "new": {}}
    for name, a in (("old", old), ("new", art)):
        f_model, f_ref = a["if"]
        score_if = T.anomaly_score(f_model, st_te, f_ref)
        if name == "new":
            score = T.blended_anomaly(f_model, f_ref, a.get("anomaly_extra"), st_te, prob_bin)
        else:
            score = score_if
        fu = T.fusion_predict(a["fusion"], st_te, score_if, emb_te)
        prv = res[name]
        prv["anomaly_auc_all"] = auc(label_any, score)
        prv["anomaly_auc_active_only"] = auc(active_any.astype(int), score)
        for i, f in enumerate(C.FAULTS):
            yv = y_te[:, i]
            prv.setdefault("auc", {})[f] = auc(yv, fu[:, i])
            f1s = [f1_score(yv, (fu[:, i] > t).astype(int), zero_division=0) for t in np.arange(0.0, 1.0, 0.05)]
            prv.setdefault("best_f1", {})[f] = float(np.max(f1s))
        hi = (y_te.sum(axis=1) > 0).astype(int)
        fu_any = fu.max(axis=1)
        prv["binary_f1_any0_5"] = float(f1_score(hi, (fu.max(axis=1) > 0.5).astype(int), zero_division=0))
        prv["binary_auc"] = auc(hi, fu_any)
        prv["fault_auc_macro"] = float(np.nanmean([prv["auc"][f] for f in C.FAULTS]))

    print("\n========== OLD vs NEW on grouped TEST windows (leak-free) ==========")
    hdr = ["key"] + [("old" if k == "old" else "new") for k in res]
    for key in ["anomaly_auc_all", "anomaly_auc_active_only", "binary_auc", "binary_f1_any0_5",
                "fault_auc_macro"]:
        print(f"{key:<26s} old={res['old'][key]:.3f}  new={res['new'][key]:.3f}")
    print("\nper-fault BEST-F1 (threshold scan 0..1, step .05)")
    print(f"{'fault':<20s} " + "".join(f"{k:<10s}" for k in res))
    for f in C.FAULTS:
        print(f"{f:<20s} " + "".join(f"{res[k]['best_f1'][f]:.3f}     " for k in res))
    print("per-fault AUC")
    print(f"{'fault':<20s} " + "".join(f"{k:<10s}" for k in res))
    for f in C.FAULTS:
        print(f"{f:<20s} " + "".join(f"{res[k]['auc'][f]:.3f}     " for k in res))
    print("=========================================================================")
    (C.OUT_ROOT / "head_retrain_report.json").write_text(
        json.dumps(res, indent=2, default=float))

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="APET-GUARDIAN model improvement CLI")
    ap.add_argument("--retrain", action="store_true",
                    help="head-only retrain (default): IF + PCA + focal fusion + classical, "
                         "then evaluates old vs new on grouped TEST windows")
    ap.add_argument("--tune-thresholds", action="store_true",
                    help="re-curate per-class + anomaly thresholds (smoothed-F1 on VAL, "
                         "verified on TEST with gate) and write the report")
    ap.add_argument("--evaluate", action="store_true",
                    help="run the regression harness against validate_baseline.json")
    args = ap.parse_args()

    if args.evaluate:
        from .validate import main as _validate_main
        _validate_main([])
    elif args.tune_thresholds:
        from .tune import tune as _tune
        _tune()
    else:
        main()
