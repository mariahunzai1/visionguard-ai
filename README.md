---
title: VisionGuard AI
emoji: 🔍
colorFrom: blue
colorTo: indigo
sdk: gradio
app_file: app.py
app_port: 7860
pinned: false
---

# VisionGuard AI — Autonomous Visual Defect Inspection & Event-Driven Triage Pipeline

Production-structured industrial inspection platform combining **classical Computer Vision (OpenCV)**, **ONNX Runtime Deep Learning (`YOLOv8-Lite` + `MobileNetV3-INT8`)**, and **deterministic backend Automation (`FastAPI` + `SQLite/PostgreSQL` + `HMAC-SHA256` Webhooks)**.

---

## Core Architecture & Modules

| Module | Path | Responsibilities |
| :--- | :--- | :--- |
| **OpenCV Metrology** | `app/cv_pipeline.py` | LAB CLAHE illumination normalization (`clipLimit=2.5`, `8x8`), Otsu contour + `minAreaRect` skew measurement, perspective homography rectification (`512x384`), golden-reference highpass subtraction, pixel-to-mm calibration (`~4.64 px/mm`), and HUD overlay rendering. |
| **ONNX DL Engine** | `app/dl_engine.py`, `app/onnx_builder.py` | Pure-Python ONNX Protobuf (IR v8 / Opset 17) exporter + `onnxruntime` CPU inference for `defect_detector_yolov8_lite.onnx` (FP32 `6x8` spatial grid detector) and `severity_mobilenetv3_int8.onnx` (`DequantizeLinear` INT8 depthwise-separable classifier). |
| **SLA Automation** | `app/automation_engine.py` | Deterministic priority rule engine (`SLA-01`..`SLA-04`): `< 0.80` confidence -> HITL QA Queue, critical defect/area/skew -> Quarantine + `HMAC-SHA256` signed JSON webhook dispatch, minor/clean -> auto-approve + JSON/Markdown audit reports. |
| **Persistence Layer** | `app/database.py` | Normalized 6-table relational schema (`inspection_batches`, `inspection_items`, `defect_detections`, `automation_events`, `webhook_deliveries`, `erp_alerts`) supporting SQLite and PostgreSQL (`DATABASE_URL`). |
| **FastAPI & Console** | `app/main.py`, `static/` | REST ingestion, QA override API, mock ERP webhook receiver (`POST /api/webhooks/erp-receiver` enforcing `X-VisionGuard-Signature`), and `#0B1120` dark-slate Operator Console. |
| **Dataset & Batch** | `scripts/generate_sample_batch.py`, `scripts/run_batch_verification.py` | Generates realistic metallic plate, PCB assembly, and packaging seal frames (`clean`, `scratch`, `crack`, `misaligned`, `borderline`) and runs end-to-end 50-item batch verification. |

---

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run 50-item synthetic batch verification (trains/exports ONNX models on first run)
python scripts/run_batch_verification.py

# 3. Start FastAPI backend & dark-slate Operator Console
uvicorn app.main:app --host 127.0.0.1 --port 8000
```

Open `http://127.0.0.1:8000` in your browser to inspect:
- **Live Inspection Feed & Annotated CV Overlay** (with pixel-to-mm bounding boxes & raw frame toggle)
- **Automation Rules & Signed Webhook Log** (`HMAC-SHA256` verification + mock ERP alert receiver)
- **Human-in-the-Loop QA Review Queue** (one-click `PASS` / `REJECT` override for `conf < 0.80`)
- **Batch Telemetry & Audit Reports** (stacked OpenCV vs ONNX latency chart & downloadable JSON/MD reports)

---

## Verified 50-Item Batch Execution Telemetry

Measured from `python scripts/run_batch_verification.py` (`seed=42`, mixed metallic/PCB/packaging):

- **Total Inspected**: `50` items (`17` metallic, `17` pcb, `16` packaging)
- **Auto-Approved (`APPROVED`)**: `26`
- **Quarantined + Signed Webhooks (`QUARANTINED`)**: `2` (`2/2` HTTP `200 OK`, HMAC-verified in `erp_alerts`)
- **HITL QA Review Queue (`PENDING_QA`)**: `22`
- **Mean OpenCV Metrology Latency**: `32.95 ms`
- **Mean ONNX Runtime Latency (`YOLOv8-Lite` + `MobileNetV3-INT8`)**: `2.48 ms`
- **Mean / P95 End-to-End Pipeline Latency**: `36.93 ms` / `47.56 ms`

---

## Running Tests & Portfolio Diagrams

```bash
# Run full pytest suite (CV metrology, ONNX inference, SLA rules/HMAC, FastAPI E2E)
python -m pytest -v

# Render 2400x1600 (2x Retina) IEEE dark-slate engineering diagrams via headless Chromium
python _portfolio_diagrams/generate_diagrams.py
```

Rendered portfolio diagrams in `_portfolio_diagrams/output/`:
1. `01_visionguard_architecture.png`
2. `02_cv_dl_triage_pipeline.png`
3. `03_automation_erd_state_machine.png`
