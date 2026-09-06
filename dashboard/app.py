# Dashboard app
from __future__ import annotations
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
import pandas as pd

try:
    from nlp.health_report_generator import generate_health_report
    from nlp.voice_alert_system import VoiceAlertSystem
    from nlp.maintenance_chatbot import MaintenanceChatbot
    from nlp.pdf_report_generator import generate_pdf_report
except ImportError:
    sys.path.insert(0, str(ROOT / "nlp"))
    from health_report_generator import generate_health_report
    from voice_alert_system import VoiceAlertSystem
    from maintenance_chatbot import MaintenanceChatbot
    from pdf_report_generator import generate_pdf_report

# Load Predictions
def load_predictions(csv_path: str | Path = "op.csv") -> pd.DataFrame:
    p = Path(csv_path)
    if not p.exists():
        p = ROOT / "op.csv"
    if p.exists():
        return pd.read_csv(p, low_memory=False)
    return pd.DataFrame()

# Main
def main():
    try:
        import streamlit as st

        st.set_page_config(page_title="APET-GUARDIAN Dashboard", layout="wide")
        st.title("APET-GUARDIAN — Engine Health Dashboard")
        df = load_predictions()
        if df.empty:
            st.warning("No predictions found (op.csv missing). Run: python run_apet_guardian_pipeline.py")
            return
        last = df.iloc[-1].to_dict()
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Health", f"{last.get('health_index', 0):.1f}/100")
        c2.metric("Anomaly", f"{last.get('anomaly_score', 0):.3f}")
        c3.metric("RUL (h)", f"{last.get('rul_mean_hours', 0):.1f}")
        c4.metric("Faults", str(last.get("predicted_faults", "none")))
        st.subheader("Health Report")
        tel = last
        report = generate_health_report(tel, last)
        st.text_area("Report", report, height=220)
        vas = VoiceAlertSystem(enabled=False)
        if vas.should_alert(last.get("health_index"), last.get("rul_mean_hours")):
            st.warning("Voice alert would trigger: health <60 or RUL <10")
            if st.button("Speak Alert"):
                vas2 = VoiceAlertSystem(enabled=True)
                vas2.alert_for_prediction(last)
        st.subheader("Maintenance Chatbot")
        q = st.text_input("Ask:", placeholder="What's wrong with engine 1?")
        if q:
            bot = MaintenanceChatbot(df, last)
            st.info(bot.ask(q))
        if st.button("Download PDF Report"):
            out = Path("APET_GUARDIAN_OUT/report.pdf")
            generate_pdf_report(df, last, out)
            st.success(f"PDF generated: {out}")
            with open(out, "rb") as f:
                st.download_button("Download", f, file_name="report.pdf", mime="application/pdf")
        st.subheader("Recent Windows")
        st.dataframe(df.tail(20))
        return
    except ImportError:
        pass
    print("=== APET-GUARDIAN Dashboard (console) ===")
    df = load_predictions()
    if df.empty:
        print("No op.csv found. Run: python run_apet_guardian_pipeline.py")
        return
    last = df.iloc[-1].to_dict()
    report = generate_health_report(last, last)
    print("\n--- Health Report ---")
    print(report)
    print("\n--- Chatbot demo ---")
    bot = MaintenanceChatbot(df, last)
    for qq in ["What's wrong with engine 1?", "How many hours until failure?", "Should I abort the mission?"]:
        print(f"Q: {qq}\nA: {bot.ask(qq)}\n")
    try:
        out = ROOT / "APET_GUARDIAN_OUT" / "report.pdf"
        generate_pdf_report(df, last, out)
        print(f"PDF generated: {out}")
    except Exception as e:
        print(f"PDF failed: {e}")
    vas = VoiceAlertSystem(enabled=False)
    if vas.should_alert(last.get("health_index"), last.get("rul_mean_hours")):
        print("Voice alert would trigger (health<60 or RUL<10)")

if __name__ == "__main__":
    main()
