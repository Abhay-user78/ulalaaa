# Chatbot Q&A
from __future__ import annotations
import re
from typing import Any, Dict, Optional
import pandas as pd

# Safe Float
def _safe_float(v: Any) -> Optional[float]:
    try:
        if v is None:
            return None
        f = float(v)
        if f != f:
            return None
        return f
    except Exception:
        return None

# Maintenancechatbot
class MaintenanceChatbot:
    """Maintenance chatbot Q&A."""
    # Init
    def __init__(self, telemetry_df: Optional[pd.DataFrame] = None, predictions: Optional[Dict[str, Any] | pd.DataFrame] = None):
        self.telemetry_df = telemetry_df
        if isinstance(predictions, pd.DataFrame):
            if len(predictions):
                self.predictions: Dict[str, Any] = dict(predictions.iloc[-1].to_dict())
            else:
                self.predictions = {}
        elif isinstance(predictions, dict):
            self.predictions = dict(predictions)
        else:
            self.predictions = {}
        if self.telemetry_df is not None and not isinstance(self.telemetry_df, pd.DataFrame):
            try:
                self.telemetry_df = pd.DataFrame(self.telemetry_df)
            except Exception:
                self.telemetry_df = None

    # Most Likely Fault
    def _most_likely_fault(self) -> tuple[Optional[str], Optional[float]]:
        best = None
        best_p = -1.0
        for k, v in self.predictions.items():
            if k.startswith("fault_probability_"):
                p = _safe_float(v)
                if p is not None and p > best_p:
                    best_p = p
                    best = k.replace("fault_probability_", "")
        if best is None:
            pf = self.predictions.get("predicted_faults")
            if isinstance(pf, str) and pf != "none" and pf:
                best = pf.split(";")[0].strip()
                best_p = _safe_float(self.predictions.get(f"fault_probability_{best}"))
            elif isinstance(pf, list) and pf:
                best = str(pf[0])
                best_p = _safe_float(self.predictions.get(f"fault_probability_{best}"))
        if best is None:
            return None, None
        return best, _safe_float(best_p)

    # Ask
    def ask(self, question: str) -> str:
        """Answer operator question."""
        if not question or not isinstance(question, str):
            return "Please ask: What's wrong? How many hours? Should I abort? What maintenance? Health trend?"
        q = question.lower().strip()
        health = _safe_float(self.predictions.get("health_index"))
        if health is None:
            health = _safe_float(self.predictions.get("health"))
        rul = _safe_float(self.predictions.get("rul_mean_hours"))
        if rul is None and isinstance(self.predictions.get("rul_hours"), dict):
            rul = _safe_float(self.predictions["rul_hours"].get("mean"))
        anomaly = _safe_float(self.predictions.get("anomaly_score"))
        fault, conf = self._most_likely_fault()
        if any(kw in q for kw in ["what's wrong", "what is wrong", "whats wrong", "fault", "wrong with", "problem", "issue"]):
            if fault and conf is not None and conf >= 0.3:
                label = fault.replace("_", " ")
                if health is not None and anomaly is not None:
                    return f"Engine shows {label} with {conf:.0%} confidence. Health {health:.1f}/100, anomaly score {anomaly:.3f}."
                return f"Most likely fault: {label} ({conf:.0%} confidence)."
            if fault:
                return f"Possible fault: {fault.replace('_',' ')} (confidence {conf:.0%})." if conf is not None else f"Possible fault: {fault.replace('_',' ')}."
            return "No active fault detected. Engine appears normal."
        if any(kw in q for kw in ["how many hours", "hours until", "hours left", "rul", "remaining", "until failure", "time left"]):
            if rul is not None:
                lo = _safe_float(self.predictions.get("rul_lower_hours"))
                hi = _safe_float(self.predictions.get("rul_upper_hours"))
                if lo is not None and hi is not None:
                    return f"Remaining useful life: {rul:.1f} hours (confidence interval {lo:.1f}-{hi:.1f} hours)."
                return f"Remaining useful life: {rul:.1f} hours."
            return "RUL is currently unavailable — insufficient history."
        if "abort" in q:
            if health is not None and health < 40:
                return "Yes — abort mission immediately. Health is critical (<40). Reduce throttle to idle and land."
            if (health is not None and health < 60) or (rul is not None and rul < 10):
                return "Caution — consider aborting. Health is degraded or RUL <10h. Reduce throttle to 50% and prepare to return."
            if fault and conf is not None and conf > 0.7:
                return f"Monitor closely — {fault.replace('_',' ')} detected ({conf:.0%}). No immediate abort needed if health {health:.0f} and RUL {rul:.0f}h." if health is not None and rul is not None else f"Monitor closely — {fault.replace('_',' ')} detected."
            return "No — continue mission. No critical fault, health and RUL are nominal."
        if any(kw in q for kw in ["maintenance", "needed", "fix", "repair", "service"]):
            if not fault or fault == "none":
                return "No maintenance needed at this time. Continue normal monitoring."
            actions = {
                "misfire": "Inspect ignition and combustion, check spark/fuel quality.",
                "injector_degradation": "Clean or replace injector, check rail pressure.",
                "turbo_issue": "Inspect turbocharger, check boost and manifold pressure.",
                "lubrication_issue": "Check oil level, pressure and temperature.",
                "sensor_drift": "Calibrate or replace sensor, verify wiring.",
                "overheating": "Reduce throttle, improve cooling, inspect thermal path.",
                "electrical_fault": "Inspect battery and wiring.",
                "bearing_fault": "Inspect bearings, check vibration spectrum.",
            }
            act = actions.get(fault, "Schedule inspection for the indicated fault.")
            return f"For {fault.replace('_',' ')}: {act} Schedule maintenance within 10 hours."
        if any(kw in q for kw in ["health trend", "trend", "health over time", "show.*health"]):
            df = self.telemetry_df
            if df is None or len(df) == 0 or "health_index" not in df.columns:
                if health is not None:
                    return f"Current health: {health:.1f}/100. No historical trend available."
                return "Health trend unavailable — no telemetry history provided."
            try:
                h = pd.to_numeric(df["health_index"], errors="coerce").dropna()
                if len(h) < 2:
                    return f"Health: {h.iloc[-1]:.1f}/100 (insufficient history)."
                first = float(h.iloc[0])
                last = float(h.iloc[-1])
                delta = last - first
                direction = "improving" if delta > 2 else "declining" if delta < -2 else "stable"
                return f"Health trend over last {len(h)} windows: {first:.1f} -> {last:.1f} ({direction}, delta {delta:+.1f}). Current {last:.1f}/100."
            except Exception:
                return "Health trend unavailable due to data error."
        return "I can answer: 'What's wrong with engine 1?', 'How many hours until failure?', 'Should I abort?', 'What maintenance is needed?', 'Show me the health trend'."

if __name__ == "__main__":
    print("=== MaintenanceChatbot Demo ===")
    import pandas as pd
    df = pd.DataFrame([{"health_index": 80 - i * 1.5, "fault_probability_injector_degradation": 0.82, "rul_mean_hours": 12.3, "predicted_faults": "injector_degradation"} for i in range(10)])
    pred = df.iloc[-1].to_dict()
    bot = MaintenanceChatbot(df, pred)
    for q in ["What's wrong with engine 1?", "How many hours until failure?", "Should I abort the mission?", "What maintenance is needed?", "Show me the health trend"]:
        print(f"\nQ: {q}\nA: {bot.ask(q)}")
