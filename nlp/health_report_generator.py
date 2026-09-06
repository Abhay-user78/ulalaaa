# Health report
from __future__ import annotations
from typing import Any, Dict, Optional

FAULT_LABELS = {
    "misfire": "Misfire",
    "injector_degradation": "Injector degradation",
    "turbo_issue": "Turbocharger issue",
    "lubrication_issue": "Lubrication issue",
    "sensor_drift": "Sensor drift",
    "overheating": "Overheating",
    "electrical_fault": "Electrical fault",
    "bearing_fault": "Bearing fault",
}

FAULT_EVIDENCE_HINT = {
    "misfire": "combustion instability / RPM variation",
    "injector_degradation": "fuel flow / rail pressure deviation",
    "turbo_issue": "boost pressure lower than expected for throttle",
    "lubrication_issue": "oil pressure / oil temperature deviation",
    "sensor_drift": "sensor reading drift vs expected baseline",
    "overheating": "CHT/EGT higher than expected for current throttle",
    "electrical_fault": "battery voltage / electrical system anomaly",
    "bearing_fault": "vibration (RMS) elevated",
}

FAULT_ACTION = {
    "misfire": "Inspect ignition/combustion, check spark/fuel quality",
    "injector_degradation": "Clean/replace injector, check rail pressure",
    "turbo_issue": "Inspect turbocharger, check boost and manifold pressure",
    "lubrication_issue": "Check oil level, pressure and temperature; inspect lubrication system",
    "sensor_drift": "Calibrate/replace affected sensor, verify wiring",
    "overheating": "Reduce throttle, improve cooling, inspect CHT/EGT thermal path",
    "electrical_fault": "Inspect battery, wiring and electrical connections",
    "bearing_fault": "Inspect bearings, check vibration spectrum, schedule borescope",
}

# Safe Float
def _safe_float(v: Any, default: Optional[float] = None) -> Optional[float]:
    if v is None:
        return default
    try:
        f = float(v)
        if f != f or f in (float("inf"), float("-inf")):
            return default
        return f
    except Exception:
        return default

# Health Band
def _health_band(health: Optional[float]) -> tuple[str, str]:
    if health is None:
        return "[?]", "unknown"
    if health >= 80:
        return "[OK]", "healthy"
    if health >= 60:
        return "[WARN]", "moderate degradation"
    if health >= 40:
        return "[WARN]", "degraded"
    return "[CRITICAL]", "critical"

# Format Rul
def _format_rul(mean: Optional[float], lower: Optional[float], upper: Optional[float]) -> str:
    if mean is None:
        return "Remaining useful life: unavailable — insufficient history."
    lo = _safe_float(lower)
    hi = _safe_float(upper)
    if lo is not None and hi is not None:
        return f"Remaining useful life: {mean:.1f} hours (confidence: {lo:.1f}-{hi:.1f} hours)."
    return f"Remaining useful life: {mean:.1f} hours."

# Top Fault
def _top_fault(ai: Dict[str, Any]) -> tuple[Optional[str], Optional[float]]:
    if "probable_faults" in ai:
        probs = ai["probable_faults"]
        if isinstance(probs, list) and probs:
            top = max(probs, key=lambda x: x.get("confidence", 0) or 0)
            return top.get("class"), _safe_float(top.get("confidence"))
        elif isinstance(probs, dict):
            top_k = max(probs.items(), key=lambda kv: kv[1] or 0)
            return top_k[0], _safe_float(top_k[1])
    best = None
    best_p = -1.0
    for k, v in ai.items():
        if k.startswith("fault_probability_"):
            f = k.replace("fault_probability_", "")
            p = _safe_float(v)
            if p is not None and p > best_p:
                best_p = p
                best = f
    if best is None:
        pf = ai.get("predicted_faults") or ai.get("active_faults")
        if isinstance(pf, str) and pf != "none" and pf:
            first = pf.split(";")[0].strip()
            p = _safe_float(ai.get(f"fault_probability_{first}"))
            return first, p
        if isinstance(pf, list) and pf:
            first = pf[0]
            p = _safe_float(ai.get(f"fault_probability_{first}"))
            return first, p
        return None, None
    return best, _safe_float(best_p)

# Evidence Text
def _evidence_text(telemetry: Dict[str, Any], ai: Dict[str, Any], fault: Optional[str]) -> str:
    tif = ai.get("top_influencing_features")
    if isinstance(tif, list) and tif:
        feats = ", ".join(f"{x.get('feature', '?')} ({x.get('importance', 0):.2f})" for x in tif[:2])
        return f"Top influencing features: {feats}."
    hints = []
    br = _safe_float(telemetry.get("boost_residual_kPa") or ai.get("boost_residual_kPa") or telemetry.get("boost_residual_norm"))
    if br is not None and abs(br) > 5:
        hints.append(f"boost residual {br:.1f} kPa")
    cht = _safe_float(telemetry.get("cht_residual_norm") or telemetry.get("cht_cyl_avg_C"))
    if fault == "overheating" and cht is not None:
        egt = _safe_float(telemetry.get("egt_cyl_avg_C"))
        if egt is not None:
            hints.append(f"EGT {egt:.0f}°C")
        hints.append(FAULT_EVIDENCE_HINT["overheating"])
    elif fault:
        hints.append(FAULT_EVIDENCE_HINT.get(fault, "telemetry deviation"))
    if not hints:
        rpm = _safe_float(telemetry.get("rpm") or telemetry.get("true_rpm"))
        vib = _safe_float(telemetry.get("vib_rms_g"))
        if vib is not None and vib > 0.5:
            hints.append(f"vibration {vib:.2f} g")
        if rpm is not None:
            hints.append(f"RPM {rpm:.0f}")
    if hints:
        return "Evidence: " + "; ".join(hints) + "."
    return "Evidence: telemetry deviation from expected baseline."

# Recommended Action
def _recommended_action(health: Optional[float], rul_mean: Optional[float], fault: Optional[str]) -> str:
    if health is not None and health < 40:
        return "Recommended action: Abort mission, reduce throttle to idle and land immediately."
    if (health is not None and health < 60) or (rul_mean is not None and rul_mean < 10):
        base = "Recommended action: Reduce throttle to 50% and schedule maintenance within 10 hours."
        if fault and fault in FAULT_ACTION:
            base += f" For {FAULT_LABELS.get(fault, fault)}: {FAULT_ACTION[fault]}."
        return base
    if fault and fault in FAULT_ACTION:
        return f"Recommended action: Continue monitoring. For {FAULT_LABELS.get(fault, fault)}: {FAULT_ACTION[fault]}."
    return "Recommended action: Continue normal operation and monitor health trend."

# Generate Health Report
def generate_health_report(telemetry_row: Optional[Dict[str, Any]], ai_predictions: Optional[Dict[str, Any]]) -> str:
    """Generate health report."""
    telemetry_row = dict(telemetry_row or {})
    ai_predictions = dict(ai_predictions or {})
    engine_id = str(telemetry_row.get("engine_id") or ai_predictions.get("engine_id") or ai_predictions.get("snapshot", {}).get("engine_id") or "Engine 1")
    if engine_id.startswith("eng_"):
        engine_id = engine_id.replace("eng_", "Engine ")
    health = _safe_float(ai_predictions.get("health_index") if ai_predictions.get("health_index") is not None else ai_predictions.get("health"))
    if health is None:
        health = _safe_float(telemetry_row.get("true_overall_health_index"))
    emoji, band = _health_band(health)
    if health is not None:
        health_line = f"{emoji} {engine_id} is experiencing {band} ({health:.1f}/100 health)."
    else:
        health_line = f"{emoji} {engine_id} health is unavailable — insufficient data."
    fault, conf = _top_fault(ai_predictions)
    if fault is None:
        fault_line = "Most likely fault: None detected (all fault probabilities below threshold)."
    else:
        label = FAULT_LABELS.get(fault, fault.replace("_", " "))
        if conf is not None:
            fault_line = f"Most likely fault: {label} ({conf:.0%} confidence)."
        else:
            fault_line = f"Most likely fault: {label}."
    evidence_line = _evidence_text(telemetry_row, ai_predictions, fault)
    rul_mean = _safe_float(ai_predictions.get("rul_mean_hours"))
    rul_lo = _safe_float(ai_predictions.get("rul_lower_hours"))
    rul_hi = _safe_float(ai_predictions.get("rul_upper_hours"))
    if rul_mean is None and isinstance(ai_predictions.get("rul_hours"), dict):
        rh = ai_predictions["rul_hours"]
        rul_mean = _safe_float(rh.get("mean"))
        rul_lo = _safe_float(rh.get("lower"))
        rul_hi = _safe_float(rh.get("upper"))
    if rul_mean is None:
        rul_mean = _safe_float(telemetry_row.get("simulated_rul_hours"))
    rul_line = _format_rul(rul_mean, rul_lo, rul_hi)
    anomaly = _safe_float(ai_predictions.get("anomaly_score"))
    anomaly_line = ""
    if anomaly is not None:
        flag = " (ALERT)" if anomaly >= 0.5 else ""
        anomaly_line = f"Anomaly score: {anomaly:.3f}{flag}."
    action_line = _recommended_action(health, rul_mean, fault)
    parts = [health_line, fault_line, evidence_line, rul_line]
    if anomaly_line:
        parts.append(anomaly_line)
    parts.append(action_line)
    warn = ai_predictions.get("data_quality_warning") or telemetry_row.get("data_quality_warning")
    if warn and warn != "ok" and isinstance(warn, str):
        parts.append(f"Note: data quality warning - {warn}.")
    return "\n".join(parts)

if __name__ == "__main__":
    _tel = {"engine_id": "Engine 1", "egt_cyl_avg_C": 420, "vib_rms_g": 0.8}
    _pred = {
        "health_index": 68.5,
        "fault_probability_injector_degradation": 0.82,
        "fault_probability_misfire": 0.1,
        "rul_mean_hours": 12.3,
        "rul_lower_hours": 8.5,
        "rul_upper_hours": 16.1,
        "anomaly_score": 0.82,
        "top_influencing_features": [{"feature": "egt_residual_norm", "importance": 0.82}],
    }
    print(generate_health_report(_tel, _pred))
    try:
        import pandas as pd
        from pathlib import Path
        p = Path("op.csv")
        if not p.exists():
            p = Path("D:/testing/op.csv")
        if p.exists():
            df = pd.read_csv(p, low_memory=False)
            row = df.iloc[-1].to_dict()
            print(generate_health_report(row, row))
    except Exception as e:
        print(f"op.csv demo skipped: {e}")
