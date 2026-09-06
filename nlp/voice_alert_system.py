# Voice alerts
from __future__ import annotations
import logging
import threading
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

CUSTOM_PHRASES = {
    "electrical_fault": {
        "critical": "MAYDAY. Engine {engine_id} electrical and bearing failure. Return to base {base_name} immediately. Fly low to {altitude} meters.",
        "params": {"base_name": "Delta", "altitude": 500},
    },
    "bearing_fault": {
        "critical": "Engine {engine_id} bearing failure imminent. Emergency landing required. Prepare crash zone.",
        "params": {},
    },
    "injector_degradation": {
        "moderate": "Engine {engine_id} injector degradation. Reduce throttle to {throttle} percent. Continue mission with monitoring. Schedule maintenance within 10 hours.",
        "critical": "Engine {engine_id} injector failure. Reduce throttle to {throttle} percent. Return to base immediately.",
        "params": {"throttle": 50},
    },
    "overheating": {
        "critical": "Engine {engine_id} overheating. Cylinder head temperature {cht} degrees. Reduce throttle to {throttle} percent. Prepare emergency landing.",
        "urgent": "Engine {engine_id} overheating. Cylinder head temperature {cht} degrees. Reduce throttle to {throttle} percent. Prepare emergency landing.",
        "params": {"throttle": 30, "cht": 225},
    },
    "misfire": {
        "moderate": "Engine {engine_id} misfire detected. Vibration {vib} G. Reduce airspeed to {speed} meters per second. Monitor for worsening.",
        "params": {"vib": 4.2, "speed": 35},
    },
    "turbo_issue": {
        "moderate": "Engine {engine_id} turbo issue. Boost low. Reduce throttle. Inspect turbo. Fly low if possible.",
        "params": {},
    },
    "lubrication_issue": {
        "moderate": "Engine {engine_id} lubrication issue. Oil pressure low. Reduce throttle. Fly low. Schedule inspection.",
        "params": {},
    },
    "sensor_drift": {
        "low": "Engine {engine_id} sensor drift detected. EGT sensor reading high. Cross-check with CHT. Schedule calibration.",
        "params": {},
    },
}

MISSION_PHASE_PHRASES = {
    "takeoff": "Engine {engine_id} fault during takeoff. Abort takeoff. Return to runway.",
    "climb": "Engine {engine_id} fault during climb. Level off at {altitude} meters. Return to base.",
    "cruise": "Engine {engine_id} fault during cruise. Reduce throttle. Continue to waypoint or return.",
    "loiter": "Engine {engine_id} fault during loiter. Abort loiter. Return to base immediately.",
    "descent": "Engine {engine_id} fault during descent. Maintain descent. Prepare emergency landing.",
    "landing": "Engine {engine_id} fault during landing. Continue landing. Emergency services on standby.",
}

BASE_INSTRUCTIONS = {
    "Delta": "Return to base Delta. Coordinates 28.6139 North, 77.2090 East. Runway heading 090 degrees.",
    "Alpha": "Return to base Alpha. Coordinates 19.0760 North, 72.8777 East. Runway heading 270 degrees.",
    "Bravo": "Return to base Bravo. Coordinates 12.9716 North, 77.5946 East. Runway heading 180 degrees.",
    "emergency": "Emergency landing required. Nearest airfield: {airfield}. Distance {distance} kilometers.",
}

# Safe Float
def _safe_float(v):
    try:
        if v is None:
            return None
        f = float(v)
        if f != f:
            return None
        return f
    except Exception:
        return None

# Voicealertsystem
class VoiceAlertSystem:
    """Voice alerts system."""
    # Init
    def __init__(self, rate: int = 180, volume: float = 0.9, enabled: bool = True, female: bool = False):
        if female and rate == 180:
            rate = 150
            volume = 1.0
        self.rate = rate
        self.volume = volume
        self.enabled = enabled
        self.female = female
        self._engine = None
        self._init_error: Optional[str] = None
        if enabled:
            try:
                import pyttsx3

                eng = pyttsx3.init()
                try:
                    eng.setProperty("rate", rate)
                except Exception:
                    pass
                try:
                    eng.setProperty("volume", volume)
                except Exception:
                    pass
                if female:
                    try:
                        voices = eng.getProperty("voices")
                        chosen = None
                        for v in voices:
                            if "zira" in v.name.lower():
                                chosen = v.id
                                break
                        if chosen is None:
                            for v in voices:
                                if "female" in v.name.lower():
                                    chosen = v.id
                                    break
                        if chosen is None and len(voices) > 1:
                            chosen = voices[1].id
                        if chosen:
                            eng.setProperty("voice", chosen)
                    except Exception:
                        pass
                self._engine = eng
            except Exception as e:
                self._init_error = str(e)
                logger.warning("pyttsx3 init failed: %s", e)
                self._engine = None

    # Should Alert
    def should_alert(self, health_index: Optional[float], rul_mean_hours: Optional[float]) -> bool:
        try:
            if health_index is not None and float(health_index) == float(health_index) and float(health_index) < 60:
                return True
        except Exception:
            pass
        try:
            if rul_mean_hours is not None and float(rul_mean_hours) == float(rul_mean_hours) and float(rul_mean_hours) < 10:
                return True
        except Exception:
            pass
        return False

    # Alert
    def alert(self, message: str, priority: str = "normal") -> None:
        """Speak alert message."""
        if not message or not isinstance(message, str):
            return
        if not self.enabled:
            logger.info("[voice disabled] %s", message)
            return
        t = threading.Thread(target=self._speak, args=(message, priority), daemon=True)
        t.start()

    # Speak
    def _speak(self, message: str, priority: str) -> None:
        if self._engine is not None:
            try:
                if priority == "critical" and not self.female:
                    try:
                        self._engine.setProperty("rate", max(120, self.rate + 20))
                    except Exception:
                        pass
                self._engine.say(message)
                self._engine.runAndWait()
                if priority == "critical" and not self.female:
                    try:
                        self._engine.setProperty("rate", self.rate)
                    except Exception:
                        pass
                return
            except Exception as e:
                logger.warning("pyttsx3 speak failed: %s", e)
        logger.info("VOICE ALERT (%s): %s", priority, message)
        print(f"[VOICE {priority.upper()}] {message}")

    # Get Action Advice
    def _get_action_advice(self, fault: str, health: Optional[float], rul: Optional[float]) -> str:
        f = fault.lower().replace(" ", "_") if fault else ""
        if health is not None and health < 40:
            return "Abort mission. Reduce throttle to idle. Land immediately."
        if health is not None and health < 60 or (rul is not None and rul < 10):
            base = "Reduce throttle to 50 percent. Fly low and return to base."
            if "overheating" in f:
                return base + " Improve cooling. Check CHT."
            if "bearing" in f or "turbo" in f or "lubrication" in f:
                return base + " Reduce load. Inspect vibration."
            if "electrical" in f:
                return base + " Check electrical system."
            if "misfire" in f or "injector" in f:
                return base + " Check fuel and ignition."
            if "sensor" in f:
                return base + " Calibrate sensor."
            return base + " Schedule maintenance within 10 hours."
        if fault:
            if "overheating" in f:
                return "Reduce throttle slightly. Monitor temperature. Improve cooling."
            if "bearing" in f:
                return "Reduce RPM. Fly low. Inspect bearings soon."
            if "turbo" in f:
                return "Reduce throttle. Inspect turbo. Fly low if possible."
            if "lubrication" in f:
                return "Check oil pressure. Fly low. Schedule inspection."
            if "electrical" in f:
                return "Check battery and wiring. Prepare to land if worsens."
            if "misfire" in f or "injector" in f:
                return "Reduce throttle. Check fuel system."
            if "sensor" in f:
                return "Sensor drift. Verify reading. Fly with caution."
        return "Continue monitoring. No immediate action."

    # Alert For Prediction
    def alert_for_prediction(self, ai_predictions: dict) -> bool:
        if not isinstance(ai_predictions, dict):
            return False
        health = ai_predictions.get("health_index")
        if health is None:
            health = ai_predictions.get("health")
        rul = ai_predictions.get("rul_mean_hours")
        if rul is None and isinstance(ai_predictions.get("rul_hours"), dict):
            try:
                rul = float(ai_predictions["rul_hours"].get("mean"))
            except Exception:
                rul = None
        if self.should_alert(health, rul):
            h_str = f"{float(health):.0f} percent" if health is not None else "unknown"
            fault = ""
            pf = ai_predictions.get("predicted_faults")
            if isinstance(pf, str) and pf != "none":
                fault = pf.split(";")[0].replace("_", " ")
            elif isinstance(pf, list) and pf:
                fault = str(pf[0]).replace("_", " ")
            else:
                best = None
                best_p = -1
                for k, v in ai_predictions.items():
                    if k.startswith("fault_probability_"):
                        try:
                            p = float(v)
                            if p > best_p:
                                best_p = p
                                best = k.replace("fault_probability_", "").replace("_", " ")
                        except Exception:
                            pass
                fault = best or ""
            rul_str = f"Remaining useful life {float(rul):.0f} hours. " if rul is not None else ""
            action = self._get_action_advice(fault, _safe_float(health) if health is not None else None, _safe_float(rul) if rul is not None else None)
            msg = f"WARNING. Engine health {h_str}. {fault + ' detected.' if fault else ''} {rul_str}{action}"
            prio = "critical" if (health is not None and float(health) < 40) else "normal"
            self.alert(msg.strip(), priority=prio)
            return True
        return False

    # Speak Custom Alert
    def speak_custom_alert(self, fault_type: str, health: Optional[float] = None, rul: Optional[float] = None, **kwargs: Any) -> None:
        if not fault_type:
            self.alert("Engine fault detected.", priority="normal")
            return
        ft = fault_type.strip().lower()
        data = CUSTOM_PHRASES.get(ft)
        if data is None:
            eid = kwargs.get("engine_id", "one")
            self.alert(f"Engine {eid} {ft.replace('_', ' ')} detected.", priority="critical" if (health is not None and health < 60) else "normal")
            return
        template = None
        if health is not None and health < 60:
            template = data.get("critical") or data.get("urgent") or data.get("moderate") or data.get("low")
        else:
            template = data.get("moderate") or data.get("low") or data.get("urgent") or data.get("critical")
        if template is None:
            for v in data.values():
                if isinstance(v, str):
                    template = v
                    break
        params: Dict[str, Any] = {}
        params.update(data.get("params", {}))
        params.update(kwargs)
        if "engine_id" not in params:
            params["engine_id"] = kwargs.get("engine_id", "one")
        if "cht" not in params and "cht_cyl_avg_C" in kwargs:
            params["cht"] = kwargs["cht_cyl_avg_C"]
        if "vib" not in params and "vib_rms_g" in kwargs:
            params["vib"] = kwargs["vib_rms_g"]
        if "cht" not in params:
            params["cht"] = params.get("cht", 225)
        if "vib" not in params:
            params["vib"] = params.get("vib", 4.2)
        try:
            msg = template.format(**params)
        except Exception:
            msg = template
        prio = "critical" if (health is not None and health < 60) else "normal"
        self.alert(msg, priority=prio)

    # Speak Mission Phase Alert
    def speak_mission_phase_alert(self, mission_phase: str, fault_type: str = "", **kwargs: Any) -> None:
        if not mission_phase:
            self.alert(f"Engine {kwargs.get('engine_id','one')} {fault_type} detected.", priority="normal")
            return
        mp = mission_phase.strip().lower()
        phrase = MISSION_PHASE_PHRASES.get(mp)
        if phrase is None:
            self.alert(f"Engine {kwargs.get('engine_id','one')} {fault_type} detected.", priority="normal")
            return
        params: Dict[str, Any] = {"engine_id": kwargs.get("engine_id", "one"), "altitude": kwargs.get("altitude", kwargs.get("altitude_m", 1000))}
        params.update(kwargs)
        try:
            msg = phrase.format(**params)
        except Exception:
            msg = phrase
        self.alert(msg, priority="critical")

    # Speak Return To Base
    def speak_return_to_base(self, base_name: str = "Delta", **kwargs: Any) -> None:
        airfield = kwargs.get("nearest_airfield", kwargs.get("airfield", "Palam"))
        distance = kwargs.get("distance_km", kwargs.get("distance", 50))
        tmpl = BASE_INSTRUCTIONS.get(base_name, BASE_INSTRUCTIONS["emergency"])
        try:
            instruction = tmpl.format(airfield=airfield, distance=distance, base_name=base_name)
        except Exception:
            try:
                instruction = tmpl.format(airfield=airfield, distance=distance)
            except Exception:
                instruction = tmpl
        self.alert(f"Engine failure. {instruction}", priority="critical")

# Speak First Anomaly Female
def _speak_first_anomaly_female():
    import pandas as pd
    from pathlib import Path

    p = Path("op.csv")
    if not p.exists():
        p = Path("D:/testing/op.csv")
    if not p.exists():
        print("op.csv not found")
        return
    df = pd.read_csv(p, low_memory=False)
    first = df[df["anomaly_alert"] == True].iloc[0] if (df["anomaly_alert"] == True).any() else df.iloc[0]
    all_engines = sorted(df["engine_id"].astype(str).unique().tolist())
    eng_map = {eid: f"Engine {i+1}" for i, eid in enumerate(all_engines)}
    engine_label = eng_map.get(str(first["engine_id"]), str(first["engine_id"]))
    t_real = float(first["window_end_time"])
    fault = str(first.get("predicted_faults", "unknown")).replace("_", " ")
    if fault == "none":
        fault = "anomaly"
    tmp_va = VoiceAlertSystem(enabled=False)
    action = tmp_va._get_action_advice(fault, _safe_float(first["health_index"]), _safe_float(first["rul_mean_hours"]))
    msg = (
        f"Fault detected. "
        f"{engine_label}. "
        f"At time, {t_real:.1f} seconds. "
        f"Anomaly score {float(first['anomaly_score']):.2f}. "
        f"Fault type, {fault}. "
        f"Health, {float(first['health_index']):.0f} percent. "
        f"Remaining useful life, {float(first['rul_mean_hours']):.1f} hours. "
        f"{action}"
    )
    print(f"First anomaly: {engine_label} at {t_real:.1f}s -> {first['predicted_faults']}")
    print(f"Action: {action}")
    print(f"Speaking (female Zira, slow): {msg}")
    v = VoiceAlertSystem(enabled=True, female=True)
    v._speak(msg, priority="critical")
    print("Done speaking.")

# Speak Anomaly Log Female
def speak_anomaly_log_female(limit: int = 5):
    import pandas as pd
    from pathlib import Path

    p = Path("op.csv")
    if not p.exists():
        p = Path("D:/testing/op.csv")
    df = pd.read_csv(p, low_memory=False)
    anom = df[df["anomaly_alert"] == True]
    if len(anom) == 0:
        print("No anomalies in op.csv")
        return
    all_engines = sorted(df["engine_id"].astype(str).unique().tolist())
    eng_map = {eid: f"Engine {i+1}" for i, eid in enumerate(all_engines)}
    print(f"Found {len(anom)} anomalies, speaking first {min(limit, len(anom))} with female voice...")
    tmp_va = VoiceAlertSystem(enabled=False)
    for _, row in anom.head(limit).iterrows():
        eng = eng_map.get(str(row["engine_id"]), str(row["engine_id"]))
        t = float(row["window_end_time"])
        fault = str(row["predicted_faults"]).replace("_", " ")
        action = tmp_va._get_action_advice(fault, _safe_float(row["health_index"]), _safe_float(row["rul_mean_hours"]))
        msg = f"Fault detected. {eng}. At {t:.0f} seconds. Fault type, {fault}. {action}"
        print(f"  {eng} at {t:.0f}s: {fault} -> {action}")
        v = VoiceAlertSystem(enabled=True, female=True)
        v._speak(msg, priority="critical")

if __name__ == "__main__":
    print("=== VoiceAlertSystem Demo ===")
    v = VoiceAlertSystem(enabled=False)
    print(f"should_alert(55, 5) -> {v.should_alert(55, 5)} (expect True)")
    print(f"should_alert(68, 12) -> {v.should_alert(68, 12)} (expect False)")
    print("Calling alert('Test alert') with enabled=False (logs only)...")
    v.alert("Test alert - engine health critical. This is a demo.")
    print("alert() returned without crash - OK")
    print("\nTo hear speech, run: python -c \"from nlp.voice_alert_system import VoiceAlertSystem; VoiceAlertSystem(enabled=True).alert('Engine health critical')\"")
    print("To hear FIRST ANOMALY in FEMALE voice: python nlp/voice_alert_system.py --female-first-anomaly")
    demo_pred = {"health_index": 55, "rul_mean_hours": 5, "predicted_faults": "injector_degradation"}
    print(f"\nalert_for_prediction({demo_pred}) -> {v.alert_for_prediction(demo_pred)} (expect True)")
    import sys

    if "--female-first-anomaly" in sys.argv or "--female" in sys.argv:
        _speak_first_anomaly_female()
    if "--test-custom" in sys.argv or "--test-all" in sys.argv:
        print("\n=== Custom Voice Phrases Test ===")
        v2 = VoiceAlertSystem(enabled=False, female=True)
        orig = v2.alert

        # Print
        def _print(msg, priority="normal"):
            print(f"[{priority.upper()}] {msg}")

        v2.alert = _print
        print("\n-- Scenario 1: Electrical --")
        v2.speak_custom_alert("electrical_fault", health=45, engine_id="one", base_name="Delta", altitude=500)
        print("\n-- Scenario 2: Injector --")
        v2.speak_custom_alert("injector_degradation", health=68, engine_id="one", throttle=50)
        print("\n-- Scenario 3: Overheating --")
        v2.speak_custom_alert("overheating", health=55, engine_id="one", cht_cyl_avg_C=225, throttle=30)
        print("\n-- Scenario 4: Misfire --")
        v2.speak_custom_alert("misfire", health=65, engine_id="one", vib_rms_g=4.2, speed=35)
        print("\n-- Scenario 5: Sensor --")
        v2.speak_custom_alert("sensor_drift", health=75, engine_id="one")
        print("\n-- Mission Phase --")
        for phase in ["takeoff", "climb", "cruise", "loiter", "descent", "landing"]:
            v2.speak_mission_phase_alert(phase, fault_type="electrical_fault", engine_id="one", altitude=3000)
        print("\n-- Base Return --")
        for base in ["Delta", "Alpha", "Bravo", "emergency"]:
            v2.speak_return_to_base(base_name=base, nearest_airfield="Palam", distance_km=45)
        print("\nCustom test done.")
    if "--speak-custom-demo" in sys.argv:
        print("\n=== Speaking custom phrases (female) ===")
        vv = VoiceAlertSystem(enabled=True, female=True)
        vv.speak_custom_alert("electrical_fault", health=45, engine_id="one", base_name="Delta", altitude=500)
        import time

        time.sleep(4)
        vv.speak_custom_alert("injector_degradation", health=68, engine_id="one", throttle=50)
        time.sleep(4)
        print("Custom demo spoken.")
