# APET-GUARDIAN — Process Guide (Pipeline + NLP)

This guide shows **how to run** the pipeline and **how to use** the NLP layer (health reports, voice, chatbot, PDF, dashboard) that turns `op.csv` numbers into plain English for operators. All commands are copy-paste from `D:\testing`.

## 0. Prerequisites

```bash
# from D:\testing
pip install -r requirements.txt
# installs: numpy pandas scikit-learn torch scipy pyttsx3==2.90 reportlab==4.0.7 pytest
# heavy chatbot deps are deferred (langchain etc. commented in requirements.txt) — MVP is rule-based, no LLM needed
```

Branch: `main` already contains `feature/nlp-assistant` (merged, fast-forward `56f96b8..7d3c2a6`). Pipeline code in `apet_guardian/` is untouched.

## 1. Prepare Input — `inp.csv`

**File:** `D:\testing\inp.csv` (example: 5643 rows, 15 missions, 2 Hz).

**Required header** (`run_apet_guardian_pipeline.py:93` `REQUIRED_SENSORS` + aliases `ALIASES:103`):

```
engine_id, mission_id, global_time_s,
throttle_actual, true_rpm, air_density_kg_m3, altitude_m,
rpm, boost_pressure_kPa, manifold_pressure_kPa,
cht_cyl_avg_C, egt_cyl_avg_C, oil_pressure_kPa, oil_temp_C,
coolant_temp_C, fuel_flow_g_s, fuel_temp_C, rail_pressure_bar,
vib_rms_g, battery_voltage_V, airspeed_mps, vertical_speed_mps,
engine_power_command_kW, true_brake_power_kW
+ optional: fault_*_active (8), simulated_rul_hours, true_overall_health_index (for verification)
```

* Aliases allowed: `timestamp→global_time_s`, `throttle→throttle_actual`, `boost_kPa→boost_pressure_kPa`, `engine_rpm→true_rpm` etc.
* Sampling: `0.5s` (2 Hz), `≥20 rows` (10 s) per `mission_id`. Gaps `>1.5s` split windows. Short missions (`<20` rows, e.g. `short_12rows`) are skipped.
* To use your own data: replace `D:\testing\inp.csv` (keep header) or change `INPUT_CSV` at `run_apet_guardian_pipeline.py:37`, or set `APET_DATA_ROOT`.

## 2. Run Pipeline — `inp.csv → op.csv`

```bash
# batch (frozen, no training)
python run_apet_guardian_pipeline.py
# → D:\testing\op.csv (example: 547 windows, 5s stride, 10s causal)

# live / incremental (appends rows to inp.csv, reports per 10s tile)
python run_apet_guardian_pipeline.py --live
# → prints per-tile: ANOMALY | FAULT (prob) | RUL [lower..upper] | HEALTH
#   state in APET_GUARDIAN_OUT/../live_state.json

# verify (optional)
# op.csv columns (35): window_end_time, anomaly_score@0.5, anomaly_alert, predicted_faults,
# fault_probability_*(8), rul_mean/lower/upper (q10/50/90), health_index, data_quality_warning, gt_*, mission_risk
```

**Models used** (`apet_guardian/models.py:16` BiLSTM, `38` FusionMLP, `75` QuantileMLP, `102` HealthMLP, `training.py:30` IsolationForest, `76` PCA; orchestration `pipeline.py:79` Stages A-G): stored at `APET_GUARDIAN_OUT/artifacts/apet_bundle.pkl` (516 MB, CPU-mapped loader `run_apet_guardian_pipeline.py:194` handles CUDA→CPU). See `guide.md` §3 for locations.

**If bundle load fails** (`torch.cuda.device_count() is 0` on CPU-only `torch+cpu`): already patched to map to CPU and fallback to `apet_bundle_cpu.pkl`. No env needed. If you retrain on GPU, the same loader will move to GPU via `T.DEVICE`.

## 3. Use NLP — `op.csv` → Plain English

All NLP is **sidecar** (`nlp/`), reads `op.csv` dicts only, never modifies pipeline. Works with 35-col `op.csv` or `APETGuardian.predicts()` dict.

### 3.1 Health Report — `nlp/health_report_generator.py: generate_health_report(telemetry_row, ai_predictions) -> str`

Template f-strings, `<100ms`, never crashes on `None`.

```python
import pandas as pd
from nlp.health_report_generator import generate_health_report

df = pd.read_csv('op.csv', low_memory=False)
# single window (last of a mission)
row = df[df.mission_id == 'fault_misfire'].iloc[15].to_dict()
print(generate_health_report(row, row))
```

**Output example:**
```
[WARN] Engine fault_misfire is experiencing degraded (43.5/100 health).
Most likely fault: Misfire (98% confidence).
Evidence: combustion instability / RPM variation.
Remaining useful life: 0.3 hours (confidence: 0.0-0.8 hours).
Anomaly score: 1.000 (ALERT).
Recommended action: Reduce throttle to 50% and schedule maintenance within 10 hours. For Misfire: Inspect ignition/combustion, check spark/fuel quality.
Note: data quality warning - rail_pressure_bar_out_of_range.
```

**Loop all missions:**
```python
for mid, g in df.groupby('mission_id'):
    print(f"=== {mid} ===")
    print(generate_health_report(g.iloc[-1].to_dict(), g.iloc[-1].to_dict()))
```

Health bands: `>80 healthy [OK], 60-80 moderate [WARN], 40-60 degraded [WARN], <40 critical [CRITICAL]`. Evidence from `top_influencing_features` or residuals (`boost_residual_kPa`, `cht_residual_norm`).

### 3.2 Voice Alert — `nlp/voice_alert_system.py: VoiceAlertSystem`

Speaks when `health <60 or RUL <10` (`should_alert`), non-blocking thread, fallback to log/print.

```python
from nlp.voice_alert_system import VoiceAlertSystem
v = VoiceAlertSystem()  # rate 180, volume 0.9
# check
v.should_alert(health_index=55, rul_mean_hours=5)  # → True
# speak (auto-builds message)
v.alert("WARNING. Engine health 55 percent. Misfire detected. RUL 0.3 hours.", priority='critical')
# auto from prediction dict
v.alert_for_prediction(last_row_dict)  # returns True if spoken

# in tests/CI without speakers:
v = VoiceAlertSystem(enabled=False)  # logs only, never raises
```

Dashboard auto-triggers `v.alert_for_prediction(last)` when `health_index<60`.

### 3.3 Chatbot — `nlp/maintenance_chatbot.py: MaintenanceChatbot.ask(question) -> str`

Rule-based regex, no LLM, 5 spec questions.

```python
import pandas as pd
from nlp.maintenance_chatbot import MaintenanceChatbot

df = pd.read_csv('op.csv')
bot = MaintenanceChatbot(df, df.iloc[-1].to_dict())

bot.ask("What's wrong with engine 1?")      # → Engine shows misfire with 98% confidence...
bot.ask("How many hours until failure?")    # → Remaining useful life: 0.3 hours [0.0-0.8]
bot.ask("Should I abort the mission?")     # → health<40 → Abort; health<60 or RUL<10 → Caution + Reduce 50%; else Continue
bot.ask("What maintenance is needed?")     # → For Misfire: Inspect ignition...
bot.ask("Show me the health trend")        # → Health trend over last 547 windows: 53.4 -> 89.1 (improving...)

# any other question → fallback list of 5
bot.ask("Tell me a joke")  # → I can answer: What's wrong..., How many hours...
```

Constructor also accepts `telemetry_df` for trend: `MaintenanceChatbot(telemetry_df=df, predictions=row_dict)`.

### 3.4 PDF Report — `nlp/pdf_report_generator.py: generate_pdf_report(telemetry_df, ai_predictions, output_path) -> None`

`reportlab` Platypus, handles `None`.

```python
from nlp.pdf_report_generator import generate_pdf_report
import pandas as pd

df = pd.read_csv('op.csv')
generate_pdf_report(df, df.iloc[-1].to_dict(), 'APET_GUARDIAN_OUT/report.pdf')
# → 2-page report: Title, Executive Summary (health report), Telemetry Overview, Fault table (8 probs), RUL/Health, Recommendations
# check
import pathlib; print(pathlib.Path('APET_GUARDIAN_OUT/report.pdf').stat().st_size)  # 3350 bytes
# header is %PDF
```

### 3.5 Dashboard — `dashboard/app.py`

Minimal, tries Streamlit else console fallback.

```bash
python dashboard/app.py
# → console fallback (no Streamlit needed):
#   Health Report (last window)
#   Chatbot demo (3 Q/A)
#   PDF generated: D:\testing\APET_GUARDIAN_OUT/report.pdf
#   Voice alert would trigger...

# with web UI (optional):
pip install streamlit==1.28.0
streamlit run dashboard/app.py
# → 4 metrics (Health/Anomaly/RUL/Faults), text_area report, voice button, chat input, Download PDF button
```

Dashboard reads `op.csv` via `load_predictions()` and calls all 4 NLP modules; `VoiceAlertSystem(enabled=False)` by default to avoid spam, button enables speak.

## 4. Verify Everything

```bash
# pipeline unchanged after merge
python run_apet_guardian_pipeline.py
# → 547 windows, anomaly AUC 0.881, binary F1 0.925

# NLP unit tests (22, all <100ms, null-safe, docstrings)
pytest tests/test_nlp.py -v
# → 22 passed (health 6, voice 5, chatbot 8, PDF 3)

# task spec one-liners
python -c "from nlp.health_report_generator import generate_health_report; ..."
python -c "from nlp.voice_alert_system import VoiceAlertSystem; v=VoiceAlertSystem(enabled=False); v.alert('Test alert')"
python -c "import pandas as pd; from nlp.maintenance_chatbot import MaintenanceChatbot; df=pd.DataFrame([{'health_index':68.5}]); print(MaintenanceChatbot(df).ask('What is wrong?'))"
```

## 5. Troubleshooting

* **Bundle load `CUDA device 0 but device_count 0` / `PyTorch was compiled without CUDA support`:** Fixed by CPU-mapped loader (`run_apet_guardian_pipeline.py:194`, `inference.py:32`). Works on both CPU-only `torch+cpu` and CUDA `torch+cu128`. If you retrain on GPU, the same loader will move to `cuda` via `T.DEVICE`.
* **`inp.csv` missing columns:** Check `run_apet_guardian_pipeline.py:93` `REQUIRED_SENSORS` and aliases `timestamp→global_time_s`. `data_quality_warning` will list `out_of_range` / `missing_values_imputed` but pipeline still runs.
* **`op.csv` has 0 windows:** Need `≥20 rows` (10 s) per `mission_id`, `global_time_s` monotonic (`config.py:14` `SAMPLE_DT 0.5, MAX_GAP 1.5`).
* **PDF fails `ImportError: reportlab`:** `pip install reportlab==4.0.7`.
* **Voice fails (no speakers):** Use `VoiceAlertSystem(enabled=False)` — logs only, as in tests.

## 6. File Map (quick)

* **Code:** `apet_guardian/models.py:16` defs, `training.py:30` training, `pipeline.py:79` orchestration, `inference.py:37` deploy
* **Store:** `APET_GUARDIAN_OUT/artifacts/apet_bundle.pkl` (keep one, `*_cpu.pkl` duplicate)
* **Run:** `run_apet_guardian_pipeline.py:37 INPUT_CSV`, `guide.md` for full pipeline guide
* **NLP:** `nlp/__init__.py` exports 4 modules, `dashboard/app.py:137`, `tests/test_nlp.py:242`, `requirements.txt`
