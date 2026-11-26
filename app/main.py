"""FastAPI backend, ASGI webhook receiver, and VisionGuard AI pipeline orchestrator."""

from __future__ import annotations

from contextlib import asynccontextmanager
import json
from pathlib import Path
import shutil
import time
from typing import Any, Dict, List, Optional, Tuple
import uuid

import cv2
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
import httpx
import numpy as np

from app.automation_engine import AutomationEngine, verify_hmac_signature
from app.config import Settings, get_settings
from app.cv_pipeline import CVMetrologyPipeline
from app.database import DatabaseManager, utc_now_iso
from app.dl_engine import DLInferenceEngine
from app.models import (
    AutomationEventOut,
    BatchRunRequest,
    BatchSummaryOut,
    InspectionItemOut,
    QAResolutionRequest,
    SLARuleSpec,
    TelemetrySnapshotOut,
    WebhookDeliveryOut,
)
from scripts.generate_sample_batch import generate_batch_directory, generate_synthetic_frame

APP_DIR = Path(__file__).resolve().parent


class VisionGuardService:
    """Coordinates CV Metrology, ONNX DL Inference, Database Persistence, and SLA Automation."""

    def __init__(self, settings: Optional[Settings] = None) -> None:
        self.settings = settings or get_settings()
        self.settings.ensure_dirs()
        self.overlays_dir = (
            self.settings.data_dir / "annotated"
            if settings is not None
            else (APP_DIR.parent / "data" / "annotated")
        )
        self.overlays_dir.mkdir(parents=True, exist_ok=True)
        self.db = DatabaseManager(self.settings.database_url)
        self.cv_pipeline = CVMetrologyPipeline(self.settings)
        self.dl_engine = DLInferenceEngine(self.settings)
        self.automation = AutomationEngine(self.settings, self.db)

    @staticmethod
    def _clean_filename(path_str: str, fallback: str) -> str:
        normalized = str(path_str or "").replace("\\", "/").strip()
        name = Path(normalized).name if normalized else ""
        return name or fallback

    def _to_raw_media_url(self, item: Dict[str, Any]) -> str:
        raw_str = str(item.get("raw_image_path") or "").replace("\\", "/")
        try:
            rel = Path(raw_str).resolve().relative_to(self.settings.data_dir.resolve())
            return f"/media/{rel.as_posix()}"
        except Exception:
            pass
        if "/data/" in raw_str:
            tail = raw_str.split("/data/", 1)[1].lstrip("/")
            return f"/media/{tail}"
        fname = self._clean_filename(raw_str, f"{item['serial_number']}.png")
        return f"/media/batches/{item['batch_id']}/{fname}"

    def _enrich_item(self, item: Dict[str, Any]) -> Dict[str, Any]:
        overlay_name = self._clean_filename(
            item.get("annotated_image_path", ""), f"{item['item_id']}_overlay.png"
        )
        item["annotated_image_url"] = f"/overlays/{overlay_name}"
        item["raw_image_url"] = self._to_raw_media_url(item)
        return item

    def ensure_item_overlay(self, item_id: str) -> Optional[Path]:
        """Return existing overlay PNG path or regenerate it on the fly if missing from disk."""
        item = self.db.get_item(item_id)
        if not item:
            return None

        overlay_name = self._clean_filename(
            item.get("annotated_image_path", ""), f"{item_id}_overlay.png"
        )
        overlay_path = self.overlays_dir / overlay_name
        if overlay_path.exists() and overlay_path.is_file():
            return overlay_path

        # Attempt to load original raw frame; fall back to synthetic surface frame if raw was pruned
        raw_str = str(item.get("raw_image_path") or "").replace("\\", "/")
        raw_candidates = [
            Path(raw_str),
            self.settings.data_dir / "batches" / item["batch_id"] / Path(raw_str).name,
        ]
        bgr = None
        for cand in raw_candidates:
            if cand.exists() and cand.is_file():
                bgr = cv2.imdecode(np.frombuffer(cand.read_bytes(), dtype=np.uint8), cv2.IMREAD_COLOR)
                if bgr is not None:
                    break

        if bgr is None:
            surface = item.get("surface_type") or "metallic"
            if surface not in ("metallic", "pcb", "packaging"):
                surface = "metallic"
            bgr, _ = generate_synthetic_frame(surface_type=surface, defect_kind="clean", seed=0)

        norm, _ = self.cv_pipeline.apply_clahe(bgr)
        rect, _, _ = self.cv_pipeline.align_and_rectify_roi(norm)
        annotated_img = self.cv_pipeline.render_annotated_overlay(
            rectified_bgr=rect,
            detections=item.get("detections", []),
            item_id=item_id,
            state=item["state"],
            severity=item["predicted_severity"],
            confidence=float(item["confidence"]),
            px_per_mm=float(item["px_per_mm"]),
            alignment_angle_deg=float(item["alignment_angle_deg"]),
            total_latency_ms=float(item["total_latency_ms"]),
        )
        self.overlays_dir.mkdir(parents=True, exist_ok=True)
        ok, png_buf = cv2.imencode(".png", annotated_img)
        if not ok:
            return None
        overlay_path.write_bytes(png_buf.tobytes())
        return overlay_path

    def inspect_frame(
        self,
        bgr_image: np.ndarray,
        batch_id: str,
        serial_number: str,
        surface_type: str,
        raw_image_path: Path,
    ) -> Dict[str, Any]:
        t_start = time.perf_counter()
        item_id = f"itm_{uuid.uuid4().hex[:10]}"

        # 1. OpenCV Metrology
        cv_res = self.cv_pipeline.process(bgr_image)

        # 2. ONNX Runtime Deep Learning Inference
        dl_res = self.dl_engine.infer(cv_res)

        total_ms = round((time.perf_counter() - t_start) * 1000.0, 3)

        # Attach detection IDs
        det_records: List[Dict[str, Any]] = []
        for d in dl_res.detections:
            det_records.append(
                {
                    "detection_id": f"det_{uuid.uuid4().hex[:10]}",
                    **d,
                }
            )

        annotated_path = self.overlays_dir / f"{item_id}_overlay.png"
        annotated_path.parent.mkdir(parents=True, exist_ok=True)
        now = utc_now_iso()

        # Persist item in ANALYZED state first so downstream automation_events & webhook_deliveries satisfy FK constraints
        item_record = {
            "item_id": item_id,
            "batch_id": batch_id,
            "serial_number": serial_number,
            "surface_type": surface_type,
            "raw_image_path": str(raw_image_path),
            "annotated_image_path": str(annotated_path),
            "state": "ANALYZED",
            "predicted_severity": dl_res.predicted_severity,
            "confidence": dl_res.confidence,
            "defect_area_mm2": cv_res.total_defect_area_mm2,
            "max_defect_length_mm": cv_res.max_defect_length_mm,
            "px_per_mm": cv_res.px_per_mm,
            "alignment_angle_deg": cv_res.alignment_angle_deg,
            "cv_latency_ms": cv_res.cv_latency_ms,
            "dl_latency_ms": dl_res.dl_latency_ms,
            "total_latency_ms": total_ms,
            "qa_reviewer": None,
            "qa_notes": None,
            "created_at": now,
            "updated_at": now,
        }
        self.db.insert_item(item_record, det_records)

        # 3. Deterministic SLA Automation & Signed Webhook Dispatch
        state, _event, _webhook = self.automation.evaluate_item(
            item_id=item_id,
            batch_id=batch_id,
            serial_number=serial_number,
            surface_type=surface_type,
            predicted_severity=dl_res.predicted_severity,
            confidence=dl_res.confidence,
            defect_area_mm2=cv_res.total_defect_area_mm2,
            alignment_angle_deg=cv_res.alignment_angle_deg,
            detections=det_records,
        )

        # 4. Render High-Contrast Annotated Metrology Overlay & finalize item state
        annotated_img = self.cv_pipeline.render_annotated_overlay(
            rectified_bgr=cv_res.rectified_bgr,
            detections=det_records,
            item_id=item_id,
            state=state,
            severity=dl_res.predicted_severity,
            confidence=dl_res.confidence,
            px_per_mm=cv_res.px_per_mm,
            alignment_angle_deg=cv_res.alignment_angle_deg,
            total_latency_ms=total_ms,
        )
        ok, png_buf = cv2.imencode(".png", annotated_img)
        if ok:
            annotated_path.write_bytes(png_buf.tobytes())
        self.db.update_item_state_and_overlay(item_id, state, str(annotated_path))
        stored = self.db.get_item(item_id)
        return self._enrich_item(stored)  # type: ignore[arg-type]

    def run_synthetic_batch(
        self,
        batch_name: str = "Batch-Inspect-50",
        product_line: str = "mixed",
        count: int = 50,
        seed: int = 42,
    ) -> Dict[str, Any]:
        batch_id = f"bat_{uuid.uuid4().hex[:10]}"
        self.db.create_batch(batch_id=batch_id, batch_name=batch_name, product_line=product_line)

        batch_dir = self.settings.data_dir / "batches" / batch_id
        manifest = generate_batch_directory(
            output_dir=batch_dir,
            count=count,
            product_line=product_line,
            seed=seed,
        )

        items_out: List[Dict[str, Any]] = []
        for entry in manifest:
            img_path = Path(entry["image_path"])
            if not img_path.exists():
                continue
            bgr = cv2.imdecode(np.frombuffer(img_path.read_bytes(), dtype=np.uint8), cv2.IMREAD_COLOR)
            if bgr is None:
                continue
            item = self.inspect_frame(
                bgr_image=bgr,
                batch_id=batch_id,
                serial_number=entry["serial_number"],
                surface_type=entry["surface_type"],
                raw_image_path=img_path,
            )
            items_out.append(item)

        report_path = self.automation.generate_batch_audit_report(batch_id)
        self.prune_old_batches()
        batch_summary = self.db.get_batch(batch_id)
        return {
            "batch": batch_summary,
            "audit_report_path": str(report_path),
            "items_processed": len(items_out),
        }

    def prune_old_batches(self) -> List[str]:
        """Remove batches beyond `settings.max_retained_batches` from DB and disk."""
        result = self.db.prune_old_batches(keep_latest=self.settings.max_retained_batches)
        for fp in result["file_paths"]:
            p = Path(fp)
            if p.exists() and p.is_file():
                p.unlink(missing_ok=True)
            if p.suffix == ".json":
                p.with_suffix(".md").unlink(missing_ok=True)
        for bid in result["pruned_batch_ids"]:
            batch_dir = self.settings.data_dir / "batches" / bid
            if batch_dir.exists() and batch_dir.is_dir():
                shutil.rmtree(batch_dir, ignore_errors=True)
        return result["pruned_batch_ids"]


def create_app(settings: Optional[Settings] = None) -> FastAPI:
    cfg = settings or get_settings()
    service = VisionGuardService(cfg)

    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        if cfg.auto_seed_on_startup and service.db.count_batches() == 0:
            service.run_synthetic_batch(
                batch_name="Sample-Batch-50-Verification",
                product_line="mixed",
                count=50,
                seed=42,
            )
        yield

    app = FastAPI(title=cfg.app_title, version=cfg.version, lifespan=lifespan)
    app.state.service = service

    # Wire ASGI webhook dispatcher without creating event-loop / socketpair deadlocks
    def asgi_webhook_dispatcher(
        target_path: str, payload_bytes: bytes, headers: Dict[str, str]
    ) -> Tuple[int, str, float]:
        t0 = time.perf_counter()
        raw_headers = [
            (k.lower().encode("latin-1"), v.encode("latin-1"))
            for k, v in headers.items()
        ]
        scope = {
            "type": "http",
            "asgi": {"version": "3.0"},
            "http_version": "1.1",
            "method": "POST",
            "scheme": "http",
            "path": target_path,
            "raw_path": target_path.encode("ascii"),
            "query_string": b"",
            "headers": raw_headers,
            "client": ("127.0.0.1", 0),
            "server": ("visionguard.local", 80),
        }
        sent_body = False
        status_code = 500
        body_chunks: List[bytes] = []

        async def receive() -> Dict[str, Any]:
            nonlocal sent_body
            if not sent_body:
                sent_body = True
                return {"type": "http.request", "body": payload_bytes, "more_body": False}
            return {"type": "http.disconnect"}

        async def send(message: Dict[str, Any]) -> None:
            nonlocal status_code
            if message["type"] == "http.response.start":
                status_code = int(message["status"])
            elif message["type"] == "http.response.body":
                body_chunks.append(message.get("body", b""))

        coro = app(scope, receive, send)
        try:
            step = coro.send(None)
            while True:
                step = coro.send(None)
        except StopIteration:
            pass
        finally:
            coro.close()

        latency_ms = (time.perf_counter() - t0) * 1000.0
        return status_code, b"".join(body_chunks).decode("utf-8"), round(latency_ms, 3)

    service.automation.set_webhook_dispatcher(asgi_webhook_dispatcher)

    if cfg.static_dir.exists():
        app.mount("/static", StaticFiles(directory=str(cfg.static_dir)), name="static")
    app.mount("/overlays", StaticFiles(directory=str(service.overlays_dir)), name="overlays")
    app.mount("/media", StaticFiles(directory=str(cfg.data_dir)), name="media")

    @app.get("/", response_class=HTMLResponse)
    def serve_console() -> HTMLResponse:
        index_file = cfg.static_dir / "index.html"
        headers = {"Cache-Control": "no-store, no-cache, must-revalidate"}
        if not index_file.exists():
            return HTMLResponse("<h1>VisionGuard AI API Running</h1>", headers=headers)
        return HTMLResponse(index_file.read_text(encoding="utf-8"), headers=headers)

    @app.get("/api/health")
    def health() -> Dict[str, Any]:
        return {
            "status": "ok",
            "version": cfg.version,
            "database_backend": "postgresql" if service.db.is_postgres else "sqlite",
            "models": service.dl_engine.get_model_info(),
        }

    @app.get("/api/telemetry", response_model=TelemetrySnapshotOut)
    def get_telemetry() -> TelemetrySnapshotOut:
        agg = service.db.get_telemetry_aggregates()
        agg["model_info"] = service.dl_engine.get_model_info()
        return TelemetrySnapshotOut(**agg)

    @app.get("/api/sla-rules", response_model=List[SLARuleSpec])
    def get_sla_rules() -> List[SLARuleSpec]:
        return service.automation.get_sla_rules()

    @app.get("/api/batches", response_model=List[BatchSummaryOut])
    def list_batches(limit: int = 20) -> List[BatchSummaryOut]:
        return [BatchSummaryOut(**b) for b in service.db.list_batches(limit=limit)]

    @app.post("/api/batches/run")
    def run_batch(req: BatchRunRequest) -> Dict[str, Any]:
        return service.run_synthetic_batch(
            batch_name=req.batch_name,
            product_line=req.product_line,
            count=req.count,
            seed=req.seed,
        )

    @app.get("/api/batches/{batch_id}/report")
    def get_batch_report(batch_id: str) -> FileResponse:
        batch = service.db.get_batch(batch_id)
        if not batch or not batch.get("audit_report_path"):
            raise HTTPException(status_code=404, detail="Audit report not found")
        report_file = Path(batch["audit_report_path"])
        if not report_file.exists():
            raise HTTPException(status_code=404, detail="Audit report file missing")
        return FileResponse(path=str(report_file), media_type="application/json", filename=report_file.name)

    @app.post("/api/inspect", response_model=InspectionItemOut)
    async def inspect_upload(
        file: UploadFile = File(...),
        surface_type: str = Form("metallic"),
        serial_number: str = Form(""),
        batch_id: str = Form(""),
    ) -> InspectionItemOut:
        mime_type = (file.content_type or "").split(";")[0].strip().lower()
        if mime_type not in cfg.allowed_image_mimes:
            raise HTTPException(
                status_code=415,
                detail=f"Unsupported media type '{mime_type}'. Allowed: {', '.join(cfg.allowed_image_mimes)}",
            )

        raw_bytes = await file.read(cfg.max_upload_bytes + 1)
        if len(raw_bytes) > cfg.max_upload_bytes:
            raise HTTPException(
                status_code=413,
                detail=f"File exceeds {cfg.max_upload_bytes // (1024 * 1024)} MB upload limit",
            )
        if not raw_bytes:
            raise HTTPException(status_code=400, detail="Uploaded image file is empty")

        arr = np.frombuffer(raw_bytes, dtype=np.uint8)
        bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if bgr is None:
            raise HTTPException(status_code=400, detail="Invalid or unreadable image file")

        if not batch_id:
            batch_id = f"bat_live_{uuid.uuid4().hex[:8]}"
            service.db.create_batch(batch_id=batch_id, batch_name="Live-Upload-Stream", product_line=surface_type)
        elif not service.db.get_batch(batch_id):
            service.db.create_batch(batch_id=batch_id, batch_name=f"Batch-{batch_id}", product_line=surface_type)

        if not serial_number:
            serial_number = f"VG-LIVE-{uuid.uuid4().hex[:6].upper()}"

        raw_dir = cfg.data_dir / "batches" / batch_id
        raw_dir.mkdir(parents=True, exist_ok=True)
        raw_path = raw_dir / f"{serial_number}.png"
        ok, raw_buf = cv2.imencode(".png", bgr)
        if ok:
            raw_path.write_bytes(raw_buf.tobytes())

        item = service.inspect_frame(
            bgr_image=bgr,
            batch_id=batch_id,
            serial_number=serial_number,
            surface_type=surface_type,
            raw_image_path=raw_path,
        )
        service.db.finalize_batch(batch_id)
        service.prune_old_batches()
        return InspectionItemOut(**item)

    @app.get("/api/items", response_model=List[InspectionItemOut])
    def list_items(
        batch_id: Optional[str] = None, state: Optional[str] = None, limit: int = 100
    ) -> List[InspectionItemOut]:
        rows = [service._enrich_item(r) for r in service.db.list_items(batch_id=batch_id, state=state, limit=limit)]
        return [InspectionItemOut(**r) for r in rows]

    @app.get("/api/items/{item_id}", response_model=InspectionItemOut)
    def get_item(item_id: str) -> InspectionItemOut:
        item = service.db.get_item(item_id)
        if not item:
            raise HTTPException(status_code=404, detail="Inspection item not found")
        return InspectionItemOut(**service._enrich_item(item))

    @app.get("/api/items/{item_id}/overlay")
    def get_item_overlay(item_id: str) -> FileResponse:
        overlay_path = service.ensure_item_overlay(item_id)
        if not overlay_path or not overlay_path.exists():
            raise HTTPException(status_code=404, detail="Overlay image not found")
        return FileResponse(path=str(overlay_path), media_type="image/png", filename=overlay_path.name)

    @app.get("/api/qa-queue", response_model=List[InspectionItemOut])
    def get_qa_queue(limit: int = 100) -> List[InspectionItemOut]:
        rows = [
            service._enrich_item(r)
            for r in service.db.list_items(state="PENDING_QA", limit=limit)
        ]
        return [InspectionItemOut(**r) for r in rows]

    @app.post("/api/qa-queue/{item_id}/resolve", response_model=InspectionItemOut)
    @app.patch("/api/qa-queue/{item_id}/resolve", response_model=InspectionItemOut)
    def resolve_qa(item_id: str, body: QAResolutionRequest) -> InspectionItemOut:
        updated = service.automation.resolve_qa_item(
            item_id=item_id,
            decision=body.decision,
            reviewer=body.reviewer,
            notes=body.notes,
        )
        if not updated:
            raise HTTPException(status_code=404, detail="Inspection item not found")
        return InspectionItemOut(**service._enrich_item(updated))

    @app.get("/api/automation-events", response_model=List[AutomationEventOut])
    def list_events(limit: int = 100) -> List[AutomationEventOut]:
        return [AutomationEventOut(**e) for e in service.db.list_automation_events(limit=limit)]

    @app.get("/api/webhooks/deliveries", response_model=List[WebhookDeliveryOut])
    def list_webhooks(limit: int = 100) -> List[WebhookDeliveryOut]:
        return [WebhookDeliveryOut(**w) for w in service.db.list_webhook_deliveries(limit=limit)]

    @app.get("/api/webhooks/erp-alerts")
    def list_erp_alerts(limit: int = 50) -> List[Dict[str, Any]]:
        return service.db.list_erp_alerts(limit=limit)

    @app.post("/api/webhooks/erp-receiver")
    async def mock_erp_webhook_receiver(request: Request) -> Dict[str, Any]:
        raw_body = await request.body()
        sig_header = request.headers.get("X-VisionGuard-Signature", "")
        if not verify_hmac_signature(cfg.webhook_secret, raw_body, sig_header):
            raise HTTPException(status_code=401, detail="Invalid HMAC-SHA256 webhook signature")

        data = json.loads(raw_body.decode("utf-8"))
        alert_id = f"erp_{uuid.uuid4().hex[:10]}"
        service.db.insert_erp_alert(
            {
                "alert_id": alert_id,
                "item_id": data.get("item_id", "unknown"),
                "batch_id": data.get("batch_id", "unknown"),
                "severity": data.get("severity", "CRITICAL_REJECT"),
                "defect_area_mm2": float(data.get("defect_area_mm2", 0.0)),
                "signature_verified": True,
                "payload_json": raw_body.decode("utf-8"),
                "received_at": utc_now_iso(),
            }
        )
        return {
            "status": "acknowledged",
            "alert_id": alert_id,
            "signature_verified": True,
            "item_id": data.get("item_id"),
        }

    return app


app = create_app()
