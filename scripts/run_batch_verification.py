"""End-to-end 50-item batch verification script for VisionGuard AI.

Generates a 50-item synthetic batch, processes every frame through the FastAPI
service pipeline (including ASGI HMAC-SHA256 webhook dispatch to `/api/webhooks/erp-receiver`),
and captures real measured latency, triage counts, QA queue size, and webhook delivery logs.
"""

from __future__ import annotations

import json
from pathlib import Path
import sys

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.config import get_settings
from app.main import create_app
from scripts.generate_sample_batch import generate_batch_directory


def main() -> None:
    settings = get_settings()
    db_file = Path(settings.database_url.replace("sqlite:///", "", 1))
    if db_file.exists():
        db_file.unlink()

    sample_dir = settings.data_dir / "batches" / "sample_batch_50"
    manifest = generate_batch_directory(
        output_dir=sample_dir,
        count=50,
        product_line="mixed",
        seed=42,
    )

    fastapi_app = create_app(settings)
    service = fastapi_app.state.service

    batch_id = "bat_sample_50"
    service.db.create_batch(
        batch_id=batch_id,
        batch_name="Sample-Batch-50-Verification",
        product_line="mixed",
    )

    for entry in manifest:
        img_path = Path(entry["image_path"])
        raw_bytes = img_path.read_bytes()
        bgr = cv2.imdecode(np.frombuffer(raw_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)
        if bgr is None:
            raise RuntimeError(f"Failed to read generated frame: {img_path}")
        service.inspect_frame(
            bgr_image=bgr,
            batch_id=batch_id,
            serial_number=entry["serial_number"],
            surface_type=entry["surface_type"],
            raw_image_path=img_path,
        )

    report_path = service.automation.generate_batch_audit_report(batch_id)
    telemetry = service.db.get_telemetry_aggregates()
    telemetry["model_info"] = service.dl_engine.get_model_info()
    qa_queue = service.db.list_items(state="PENDING_QA", limit=100)
    webhooks = service.db.list_webhook_deliveries(limit=100)
    erp_alerts = service.db.list_erp_alerts(limit=100)
    batch_summary = service.db.get_batch(batch_id)

    summary = {
        "batch_summary": batch_summary,
        "telemetry": {k: v for k, v in telemetry.items() if k != "latency_series"},
        "qa_queue_count": len(qa_queue),
        "webhook_deliveries_count": len(webhooks),
        "erp_alerts_count": len(erp_alerts),
        "audit_report_path": str(report_path),
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
