"""Hugging Face Spaces (Gradio SDK) entrypoint for VisionGuard AI.

Preserves the full dark-slate FastAPI Operator Console at `/` while mounting
a lightweight Gradio status card at `/gradio`.
"""

from __future__ import annotations

import gradio as gr
import uvicorn

from app.main import app


with gr.Blocks(title="VisionGuard AI — Status") as demo:
    gr.Markdown(
        """
        # VisionGuard AI — Autonomous Visual Defect Inspection & Event-Driven Triage Pipeline

        **Status:** Online · **ONNX Runtime** (`YOLOv8-Lite` + `MobileNetV3-INT8`) · **OpenCV Metrology**

        The full `#0B1120` dark-slate Operator Console (Live CV Overlay viewer, SLA Automation & HMAC-SHA256
        Webhook Dispatch log, HITL QA Review Queue, and Batch Telemetry charts) is served at the root path:

        - **[Open Operator Console](/)**
        - **[API Health (`/api/health`)](/api/health)**
        - **[Live Telemetry (`/api/telemetry`)](/api/telemetry)**
        """
    )

app = gr.mount_gradio_app(app, demo, path="/gradio")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=7860)
