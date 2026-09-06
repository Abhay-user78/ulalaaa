# NLP tests
import time
from pathlib import Path

import pandas as pd
import pytest

@pytest.fixture
# Sample Telemetry
def sample_telemetry():
    return {
        "engine_id": "Engine 1",
        "rpm": 2500,
        "cht_cyl_avg_C": 180,
        "egt_cyl_avg_C": 420,
        "oil_pressure_kPa": 350,
        "vib_rms_g": 0.8,
        "boost_residual_kPa": -12.3,
        "throttle_actual": 0.82,
    }

@pytest.fixture
# Sample Predictions
def sample_predictions():
    return {
        "engine_id": "Engine 1",
        "anomaly_score": 0.82,
        "fault_probability_misfire": 0.1,
        "fault_probability_injector_degradation": 0.82,
        "fault_probability_turbo_issue": 0.15,
        "fault_probability_lubrication_issue": 0.2,
        "fault_probability_sensor_drift": 0.05,
        "fault_probability_overheating": 0.3,
        "fault_probability_electrical_fault": 0.08,
        "fault_probability_bearing_fault": 0.12,
        "predicted_faults": "injector_degradation",
        "rul_mean_hours": 12.3,
        "rul_lower_hours": 8.5,
        "rul_upper_hours": 16.1,
        "health_index": 68.5,
        "top_influencing_features": [
            {"feature": "egt_residual_norm", "importance": 0.82},
            {"feature": "fuel_flow_g_s", "importance": 0.45},
        ],
    }

@pytest.fixture
# Sample Df
def sample_df(sample_predictions):
    df = pd.DataFrame([sample_predictions] * 10)
    df["health_index"] = [80 - i * 1.5 for i in range(10)]
    return df

# Testhealthreport
class TestHealthReport:
    # Test Generate Normal
    def test_generate_normal(self, sample_telemetry, sample_predictions):
        from nlp.health_report_generator import generate_health_report

        report = generate_health_report(sample_telemetry, sample_predictions)
        assert isinstance(report, str)
        assert len(report) > 50
        assert "68.5" in report
        assert "Injector degradation" in report
        assert "12.3" in report

    # Test Handles None Predictions
    def test_handles_none_predictions(self, sample_telemetry):
        from nlp.health_report_generator import generate_health_report

        report = generate_health_report(sample_telemetry, None)
        assert isinstance(report, str)
        assert len(report) > 20

    # Test Handles Empty Dicts
    def test_handles_empty_dicts(self):
        from nlp.health_report_generator import generate_health_report

        report = generate_health_report({}, {})
        assert isinstance(report, str)
        assert len(report) > 20

    # Test Handles Missing Rul
    def test_handles_missing_rul_health(self, sample_telemetry):
        from nlp.health_report_generator import generate_health_report

        report = generate_health_report(sample_telemetry, {"anomaly_score": 0.2})
        assert isinstance(report, str)
        assert "unavailable" in report.lower() or "N/A" in report or "unknown" in report.lower()

    # Test Performance Under 100Ms
    def test_performance_under_100ms(self, sample_telemetry, sample_predictions):
        from nlp.health_report_generator import generate_health_report

        start = time.time()
        for _ in range(100):
            generate_health_report(sample_telemetry, sample_predictions)
        elapsed = (time.time() - start) / 100 * 1000
        assert elapsed < 100, f"Health report {elapsed:.1f}ms exceeds 100ms"

    # Test Has Docstring And
    def test_has_docstring_and_type_hints(self):
        from nlp.health_report_generator import generate_health_report

        assert generate_health_report.__doc__ is not None
        assert "telemetry_row" in generate_health_report.__doc__ or "telemetry" in str(generate_health_report.__annotations__)

# Testvoicealert
class TestVoiceAlert:
    # Test Should Alert True
    def test_should_alert_true(self):
        from nlp.voice_alert_system import VoiceAlertSystem

        v = VoiceAlertSystem(enabled=False)
        assert v.should_alert(59, 20) is True
        assert v.should_alert(68, 9) is True
        assert v.should_alert(30, 5) is True

    # Test Should Alert False
    def test_should_alert_false(self):
        from nlp.voice_alert_system import VoiceAlertSystem

        v = VoiceAlertSystem(enabled=False)
        assert v.should_alert(68, 12) is False
        assert v.should_alert(80, 20) is False
        assert v.should_alert(None, None) is False

    # Test Alert Does Not
    def test_alert_does_not_crash(self):
        from nlp.voice_alert_system import VoiceAlertSystem

        v = VoiceAlertSystem(enabled=False)
        v.alert("Test alert")
        v.alert("", priority="critical")
        v.alert(None)

    # Test Alert For Prediction
    def test_alert_for_prediction(self, sample_predictions):
        from nlp.voice_alert_system import VoiceAlertSystem

        v = VoiceAlertSystem(enabled=False)
        crit = dict(sample_predictions, health_index=55)
        assert v.should_alert(crit["health_index"], crit["rul_mean_hours"]) is True
        result = v.alert_for_prediction(crit)
        assert result is True

    # Test Has Docstring
    def test_has_docstring(self):
        from nlp.voice_alert_system import VoiceAlertSystem

        assert VoiceAlertSystem.__doc__ is not None
        assert VoiceAlertSystem.alert.__doc__ is not None

# Testchatbot
class TestChatbot:
    # Test Whats Wrong
    def test_whats_wrong(self, sample_df, sample_predictions):
        from nlp.maintenance_chatbot import MaintenanceChatbot

        c = MaintenanceChatbot(sample_df, sample_predictions)
        ans = c.ask("What's wrong with engine 1?")
        assert "injector" in ans.lower()

    # Test How Many Hours
    def test_how_many_hours(self, sample_df, sample_predictions):
        from nlp.maintenance_chatbot import MaintenanceChatbot

        c = MaintenanceChatbot(sample_df, sample_predictions)
        ans = c.ask("How many hours until failure?")
        assert "12.3" in ans or "remaining" in ans.lower()

    # Test Should Abort
    def test_should_abort(self, sample_df, sample_predictions):
        from nlp.maintenance_chatbot import MaintenanceChatbot

        c = MaintenanceChatbot(sample_df, sample_predictions)
        ans = c.ask("Should I abort the mission?")
        assert isinstance(ans, str) and len(ans) > 10
        crit = dict(sample_predictions, health_index=30)
        c2 = MaintenanceChatbot(sample_df, crit)
        assert "abort" in c2.ask("Should I abort?").lower()

    # Test Maintenance Needed
    def test_maintenance_needed(self, sample_df, sample_predictions):
        from nlp.maintenance_chatbot import MaintenanceChatbot

        c = MaintenanceChatbot(sample_df, sample_predictions)
        ans = c.ask("What maintenance is needed?")
        assert "injector" in ans.lower() or "maintenance" in ans.lower()

    # Test Health Trend
    def test_health_trend(self, sample_df, sample_predictions):
        from nlp.maintenance_chatbot import MaintenanceChatbot

        c = MaintenanceChatbot(sample_df, sample_predictions)
        ans = c.ask("Show me the health trend")
        assert "health" in ans.lower()
        assert "→" in ans or "trend" in ans.lower()

    # Test Fallback
    def test_fallback(self, sample_df, sample_predictions):
        from nlp.maintenance_chatbot import MaintenanceChatbot

        c = MaintenanceChatbot(sample_df, sample_predictions)
        ans = c.ask("Tell me a joke")
        assert "I can answer" in ans

    # Test Empty Question
    def test_empty_question(self, sample_df, sample_predictions):
        from nlp.maintenance_chatbot import MaintenanceChatbot

        c = MaintenanceChatbot(sample_df, sample_predictions)
        ans = c.ask("")
        assert isinstance(ans, str) and len(ans) > 10

    # Test Has Docstring
    def test_has_docstring(self):
        from nlp.maintenance_chatbot import MaintenanceChatbot

        assert MaintenanceChatbot.__doc__ is not None
        assert MaintenanceChatbot.ask.__doc__ is not None

# Testpdfreport
class TestPDFReport:
    # Test Generate Pdf
    def test_generate_pdf(self, sample_df, sample_predictions):
        from nlp.pdf_report_generator import generate_pdf_report

        out = Path("APET_GUARDIAN_OUT/test_report.pdf")
        try:
            import reportlab
        except ImportError:
            pytest.skip("reportlab not installed")
        generate_pdf_report(sample_df, sample_predictions, out)
        assert out.exists()
        assert out.stat().st_size > 1000
        assert out.read_bytes()[:4] == b"%PDF"

    # Test Generate Pdf With
    def test_generate_pdf_with_none(self):
        from nlp.pdf_report_generator import generate_pdf_report

        try:
            import reportlab
        except ImportError:
            pytest.skip("reportlab not installed")
        out = Path("APET_GUARDIAN_OUT/test_report_none.pdf")
        generate_pdf_report(None, {}, out)
        assert out.exists()
        assert out.read_bytes()[:4] == b"%PDF"

    # Test Has Docstring
    def test_has_docstring(self):
        from nlp.pdf_report_generator import generate_pdf_report

        assert generate_pdf_report.__doc__ is not None
