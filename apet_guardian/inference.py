# Inference engine
import pickle
import tempfile
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import torch

from . import config as C
from .models import BiLSTMEncoder, FaultFusionMLP, HealthMLP, QuantileMLP
from .windowing import window_stats
from . import training as T

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load Pickle
def _load_pickle(path: Path):
    import io

    # Cpuunpickler
    class _CpuUnpickler(pickle.Unpickler):
        # Find Class
        def find_class(self, module, name):
            if module.startswith("torch.cuda"):
                try:
                    return getattr(torch, name)
                except AttributeError:
                    try:
                        return getattr(torch.storage, name)
                    except AttributeError:
                        pass
            if module == "torch.storage" and name == "_load_from_bytes":
                orig = getattr(torch.storage, "_load_from_bytes")

                # Cpu Load
                def _cpu_load(b):
                    try:
                        return orig(b)
                    except RuntimeError as re:
                        if "CUDA" in str(re) or "device_count" in str(re):
                            return torch.UntypedStorage.from_buffer(b, byte_order="little", dtype=torch.uint8)
                        raise

                return _cpu_load
            return super().find_class(module, name)

    try:
        return torch.load(str(path), map_location=torch.device("cpu"), weights_only=False)
    except Exception:
        try:
            with open(path, "rb") as f:
                return _CpuUnpickler(f).load()
        except Exception:
            orig_avail = torch.cuda.is_available
            orig_count = getattr(torch.cuda, "device_count", None)
            try:
                torch.cuda.is_available = lambda: True
                if orig_count is not None:
                    torch.cuda.device_count = lambda: 1
                with open(path, "rb") as f:
                    art = pickle.load(f)
                for k in list(art.keys()):
                    v = art[k]
                    if isinstance(v, torch.nn.Module):
                        try:
                            art[k] = v.to(torch.device("cpu"))
                        except Exception:
                            pass
                return art
            finally:
                torch.cuda.is_available = orig_avail
                if orig_count is not None:
                    torch.cuda.device_count = orig_count

# Apetguardian
class APETGuardian:
    # Init
    def __init__(self, bundle_path: Optional[Path] = None):
        if bundle_path is None:
            bundle_path = Path(__file__).resolve().parents[1] / "APET_GUARDIAN_OUT" / "artifacts" / "apet_bundle.pkl"
        self.art = _load_pickle(bundle_path)
        self.seq_scaler = self.art["seq_scaler"]
        self.stat_scaler = self.art["stat_scaler"]
        self.physics = self.art["physics"]
        self.seq_features = list(self.art["seq_features"])
        self.stats_names = list(self.art["stats_names"])
        self.faults = list(self.art["faults"])
        self.if_model, self.if_ref = self.art["if"]
        self._build_nets()

    # Build Nets
    def _build_nets(self):
        self.lstm = self.art["lstm"]
        self.lstm.eval()
        self.fusion = self.art.get("fusion")
        if self.fusion is not None and isinstance(self.fusion, torch.nn.Module):
            self.fusion.eval()
        self.classical = self.art.get("classical")
        self.quantile = self.art.get("quantile")
        self.health = self.art.get("health")

    @staticmethod
    # Saliency
    def _saliency(model, x: torch.Tensor, agg: str = "sum") -> np.ndarray:
        model.eval()
        x = x.detach().clone().requires_grad_(True)
        out = model(x)
        if isinstance(out, (list, tuple)):
            out = torch.stack([o for o in out], dim=1)
        if out.dim() > 1 and out.shape[1] > 1:
            if agg == "pos":
                out = out.clamp(min=0).sum(dim=1)
            else:
                out = out.sum(dim=1)
        else:
            out = out.flatten()
        grad = torch.autograd.grad(out.sum(), x, retain_graph=False)[0]
        sal = (grad.abs() * x.abs()).detach().cpu().numpy()
        return sal

    # Seq Tensor
    def _seq_tensor(self, df: pd.DataFrame) -> torch.Tensor:
        last = df.iloc[-C.SEQ_LEN:]
        mat = last[self.seq_features].to_numpy(dtype="float32")
        mat = np.nan_to_num(mat, nan=0.0)
        mat = self.seq_scaler.transform(mat)
        return torch.from_numpy(mat).float().to(DEVICE)[None, :, :]

    # Row Stats
    def _row_stats(self, df: pd.DataFrame) -> np.ndarray:
        last = df.iloc[-C.SEQ_LEN:]
        mat = last[self.seq_features].to_numpy(dtype="float32")
        mat = np.nan_to_num(mat, nan=0.0)
        stats_mat, _ = window_stats(mat[None, :, :], self.seq_features)
        cum = df[C.CUMULATIVE_FEATURES].to_numpy(dtype="float32")[-1][None, :]
        stats = np.concatenate([stats_mat, cum], axis=1).astype("float32")
        return self.stat_scaler.transform(stats)

    # Predicts
    def predicts(self, df: pd.DataFrame, top_k: int = 5) -> Dict:
        if len(df) < C.SEQ_LEN:
            raise ValueError(f"Need at least {C.SEQ_LEN} rows for a 10 s window")
        df = df.sort_values("time_in_mission_s")
        if "phase" not in df.columns:
            df["phase"] = "cruise"
        ph = self.physics.transform(df)

        stats_s = self._row_stats(ph)
        seq_t = self._seq_tensor(ph)
        with torch.no_grad():
            emb, logit = self.lstm(seq_t)
        fault_prob = float(torch.sigmoid(logit[0]).cpu())
        normal_prob = 1.0 - fault_prob

        anomaly = float(T.blended_anomaly(self.if_model, self.if_ref,
                                          self.art.get("anomaly_extra"), stats_s,
                                          np.array([fault_prob])))

        fusion_probs = None
        classical_probs = None
        if getattr(self, "fusion", None) is not None:
            stats_t = torch.from_numpy(stats_s).float().to(DEVICE)
            fu = self.fusion(stats_t, torch.tensor([anomaly], device=DEVICE), emb)
            fusion_probs = torch.sigmoid(fu[0]).detach().cpu().numpy()
        if getattr(self, "classical", None) is not None:
            classical_probs = T.classical_predict(self.classical, stats_s)[0]

        faults = []
        src = fusion_probs if fusion_probs is not None else classical_probs
        if src is not None:
            for i, name in enumerate(self.faults):
                faults.append({"class": name, "confidence": float(src[i])})
            faults.sort(key=lambda r: r["confidence"], reverse=True)
            for r in faults:
                r["active"] = bool(r["confidence"] >= 0.5)
            active = [r for r in faults if r["active"]]

        rul = {"mean": None, "lower": None, "upper": None}
        if self.quantile is not None:
            q = T.quantile_predict(self.quantile, stats_s)[0]
            rul = {"lower": float(q[0]), "mean": float(q[1]), "upper": float(q[2])}
        health = None
        if self.health is not None:
            health = float(T.health_predict(self.health, stats_s)[0])

        top = self._top_features(stats_s, top_k)

        snapshot = {
            "time_s": float(ph["time_in_mission_s"].iloc[-1]),
            "phase": str(ph["phase"].iloc[-1]),
            "engine_id": str(ph["engine_id"].iloc[-1]) if "engine_id" in ph.columns else None,
            "mission_id": str(ph["mission_id"].iloc[-1]) if "mission_id" in ph.columns else None,
            "boost_residual_kPa": float(ph["boost_residual_kPa"].iloc[-1]),
            "rpm": float(ph["rpm"].iloc[-1]) if "rpm" in ph.columns else None,
        }
        return {
            "anomaly_score": anomaly,
            "normal_probability": normal_prob,
            "fault_probability": fault_prob,
            "probable_faults": faults,
            "active_faults": [f["class"] for f in faults if f.get("active")],
            "rul_hours": rul,
            "health_index": health,
            "top_influencing_features": top,
            "snapshot": snapshot,
        }

    # Top Features
    def _top_features(self, stats_s: np.ndarray, top_k: int) -> List[Dict]:
        scores = {}
        if self.quantile is not None:
            x = torch.from_numpy(stats_s).float().to(DEVICE)
            s = self._saliency(self.quantile, x, agg="pos")
            scores["rul"] = s[0]
        if self.health is not None:
            x = torch.from_numpy(stats_s).float().to(DEVICE)
            s = self._saliency(self.health, x, agg="sum")
            scores["health"] = s[0]
        combined = np.zeros(stats_s.shape[-1], dtype="float64")
        n = 0
        for v in scores.values():
            combined += v
            n += 1
        if n == 0:
            return []
        combined /= n
        order = np.argsort(-combined)
        out = []
        for idx in order[:top_k]:
            out.append({"feature": self.stats_names[int(idx)],
                        "importance": float(combined[int(idx)])})
        return out
