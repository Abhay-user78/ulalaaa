# PDF report
from __future__ import annotations
from pathlib import Path
from typing import Any, Dict, Optional
import pandas as pd

# Safe
def _safe(v: Any, default: str = "N/A") -> str:
    if v is None or (isinstance(v, float) and v != v):
        return default
    try:
        if isinstance(v, float):
            return f"{v:.2f}"
        return str(v)
    except Exception:
        return default

# Generate Pdf Report
def generate_pdf_report(telemetry_df: Optional[pd.DataFrame], ai_predictions: Optional[Dict[str, Any] | pd.DataFrame], output_path: str | Path) -> None:
    """Generate PDF report."""
    try:
        from reportlab.lib import colors
        from reportlab.lib.pagesizes import A4
        from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
        from reportlab.lib.units import inch
        from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle
    except ImportError as e:
        raise ImportError("reportlab required: pip install reportlab==4.0.7") from e
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pred: Dict[str, Any] = {}
    if isinstance(ai_predictions, pd.DataFrame):
        if len(ai_predictions):
            pred = dict(ai_predictions.iloc[-1].to_dict())
    elif isinstance(ai_predictions, dict):
        pred = dict(ai_predictions)
    elif ai_predictions is not None:
        try:
            pred = dict(ai_predictions)
        except Exception:
            pred = {}
    health_summary = ""
    try:
        from .health_report_generator import generate_health_report

        tel_dict: Dict[str, Any] = {}
        if isinstance(telemetry_df, pd.DataFrame) and len(telemetry_df):
            tel_dict = dict(telemetry_df.iloc[-1].to_dict())
        health_summary = generate_health_report(tel_dict, pred)
    except Exception:
        health_summary = "Health summary unavailable."
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle("Title2", parent=styles["Title"], fontSize=18, spaceAfter=12)
    heading = ParagraphStyle("Heading2", parent=styles["Heading2"], fontSize=12, spaceAfter=6, spaceBefore=12)
    normal = styles["Normal"]
    normal.fontSize = 9
    normal.leading = 11
    story: list[Any] = []
    story.append(Paragraph("UAV Digital Twin — Mission Health Report", title_style))
    story.append(Paragraph("APET-GUARDIAN • Automated Health Assessment", styles["Normal"]))
    story.append(Spacer(1, 0.2 * inch))
    story.append(Paragraph("Executive Summary", heading))
    for line in health_summary.split("\n"):
        story.append(Paragraph(line.replace("[WARN]", "").replace("[OK]", "").replace("[CRITICAL]", "").strip() or "&nbsp;", normal))
    story.append(Spacer(1, 0.15 * inch))
    story.append(Paragraph("Telemetry Overview", heading))
    n_rows = len(telemetry_df) if isinstance(telemetry_df, pd.DataFrame) and telemetry_df is not None else 0
    tcols = len(telemetry_df.columns) if isinstance(telemetry_df, pd.DataFrame) and telemetry_df is not None else 0
    story.append(Paragraph(f"Telemetry rows: {n_rows} &nbsp;|&nbsp; Columns: {tcols}", normal))
    story.append(Spacer(1, 0.1 * inch))
    story.append(Paragraph("Fault Probabilities", heading))
    fault_rows = [["Fault", "Probability", "Status"]]
    for fault in ["misfire", "injector_degradation", "turbo_issue", "lubrication_issue", "sensor_drift", "overheating", "electrical_fault", "bearing_fault"]:
        prob = pred.get(f"fault_probability_{fault}")
        try:
            p = float(prob) if prob is not None else None
            p_str = f"{p:.2%}" if p is not None else "N/A"
            status = "ACTIVE" if p is not None and p >= 0.5 else "--"
        except Exception:
            p_str = _safe(prob)
            status = "--"
        fault_rows.append([fault.replace("_", " "), p_str, status])
    t = Table(fault_rows, colWidths=[2.5 * inch, 1.2 * inch, 1.0 * inch])
    t.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#2F5496")), ("TEXTCOLOR", (0, 0), (-1, 0), colors.whitesmoke), ("ALIGN", (0, 0), (-1, -1), "LEFT"), ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"), ("FONTSIZE", (0, 0), (-1, -1), 8), ("BOTTOMPADDING", (0, 0), (-1, 0), 8), ("GRID", (0, 0), (-1, -1), 0.5, colors.grey), ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#D9E1F2")])]))
    story.append(t)
    story.append(Spacer(1, 0.15 * inch))
    story.append(Paragraph("RUL & Health", heading))
    rul_m = pred.get("rul_mean_hours")
    rul_lo = pred.get("rul_lower_hours")
    rul_hi = pred.get("rul_upper_hours")
    if rul_m is None and isinstance(pred.get("rul_hours"), dict):
        rh = pred["rul_hours"]
        rul_m = rh.get("mean")
        rul_lo = rh.get("lower")
        rul_hi = rh.get("upper")
    health = pred.get("health_index")
    if health is None:
        health = pred.get("health")
    rul_str = f"{float(rul_m):.1f} h" if rul_m is not None else "N/A"
    try:
        if rul_lo is not None and rul_hi is not None:
            rul_str += f" [{float(rul_lo):.1f}-{float(rul_hi):.1f}]"
    except Exception:
        pass
    health_str = f"{float(health):.1f}/100" if health is not None else "N/A"
    anomaly = pred.get("anomaly_score")
    anomaly_str = f"{float(anomaly):.3f}" if anomaly is not None else "N/A"
    story.append(Paragraph(f"Health Index: <b>{health_str}</b> &nbsp;|&nbsp; RUL: <b>{rul_str}</b> &nbsp;|&nbsp; Anomaly Score: <b>{anomaly_str}</b>", normal))
    story.append(Spacer(1, 0.15 * inch))
    story.append(Paragraph("Recommendations", heading))
    try:
        h = float(health) if health is not None else None
        r = float(rul_m) if rul_m is not None else None
        if h is not None and h < 40:
            rec = "Abort mission, reduce throttle to idle and land immediately."
        elif (h is not None and h < 60) or (r is not None and r < 10):
            rec = "Reduce throttle to 50% and schedule maintenance within 10 hours."
        else:
            rec = "Continue normal operation and monitor health trend."
    except Exception:
        rec = "Continue monitoring."
    story.append(Paragraph(rec, normal))
    warn = pred.get("data_quality_warning")
    if warn and warn != "ok" and isinstance(warn, str):
        story.append(Spacer(1, 0.08 * inch))
        story.append(Paragraph(f"<i>Note: data quality - {warn}</i>", normal))
    story.append(Spacer(1, 0.3 * inch))
    story.append(Paragraph("Generated by APET-GUARDIAN NLP • Advisory only.", styles["Italic"]))
    doc = SimpleDocTemplate(str(output_path), pagesize=A4, rightMargin=48, leftMargin=48, topMargin=48, bottomMargin=48, title="UAV Mission Health Report")
    doc.build(story)

if __name__ == "__main__":
    print("=== PDF Report Demo ===")
    try:
        import pandas as pd
        from pathlib import Path

        df = pd.DataFrame([{"health_index": 68.5}])
        pred = df.iloc[-1].to_dict()
        p = Path("op.csv")
        if not p.exists():
            p = Path("D:/testing/op.csv")
        if p.exists():
            df = pd.read_csv(p, low_memory=False)
            pred = df.iloc[-1].to_dict()
            print(f"Using op.csv: {len(df)} windows")
        out = Path("APET_GUARDIAN_OUT/report.pdf")
        if not out.parent.exists():
            out.parent.mkdir(parents=True, exist_ok=True)
        generate_pdf_report(df, pred, out)
        print(f"PDF generated: {out} ({out.stat().st_size} bytes)")
    except ImportError as e:
        print(f"reportlab not installed: {e}")
    except Exception as e:
        print(f"Demo failed: {e}")
