"""Deterministic Event-Driven Automation, SLA Triage, Signed Webhook & Audit Engine."""

from __future__ import annotations

import hashlib
import hmac
import json
from pathlib import Path
import time
from typing import Any, Callable, Dict, List, Optional, Tuple
import uuid

from app.config import Settings
from app.database import DatabaseManager, utc_now_iso
from app.models import ItemState, SLARuleSpec, SeverityLevel


def compute_hmac_signature(secret: str, payload_bytes: bytes) -> str:
    digest = hmac.new(secret.encode("utf-8"), payload_bytes, hashlib.sha256).hexdigest()
    return f"sha256={digest}"


def verify_hmac_signature(secret: str, payload_bytes: bytes, signature_header: str) -> bool:
    expected = compute_hmac_signature(secret, payload_bytes)
    return hmac.compare_digest(expected, signature_header or "")


WebhookDispatcher = Callable[[str, bytes, Dict[str, str]], Tuple[int, str, float]]


class AutomationEngine:
    """Evaluates deterministic SLA rules on inspection telemetry, triggers HMAC-signed webhooks,
    routes uncertain edge cases to Human QA, and compiles structured batch audit reports."""

    def __init__(
        self,
        settings: Settings,
        db: DatabaseManager,
        webhook_dispatcher: Optional[WebhookDispatcher] = None,
    ) -> None:
        self.settings = settings
        self.db = db
        self.webhook_dispatcher = webhook_dispatcher

    def set_webhook_dispatcher(self, dispatcher: WebhookDispatcher) -> None:
        self.webhook_dispatcher = dispatcher

    def get_sla_rules(self) -> List[SLARuleSpec]:
        return [
            SLARuleSpec(
                rule_id="SLA-01",
                name="Confidence Gate -> HITL QA Review",
                condition=f"confidence < {self.settings.qa_confidence_threshold:.2f}",
                action="ROUTE_TO_HITL_QA",
                priority=1,
            ),
            SLARuleSpec(
                rule_id="SLA-02",
                name="Critical Defect -> Quarantine & Signed ERP Webhook",
                condition=(
                    f"severity == CRITICAL_REJECT OR area >= {self.settings.critical_area_mm2_threshold:.1f}mm2 "
                    f"OR |skew| > {self.settings.max_alignment_skew_deg:.1f}deg"
                ),
                action="QUARANTINE_AND_WEBHOOK",
                priority=2,
            ),
            SLARuleSpec(
                rule_id="SLA-03",
                name="Minor Cosmetic -> Log & Conditional Pass",
                condition=(
                    f"severity == MINOR_COSMETIC AND confidence >= {self.settings.qa_confidence_threshold:.2f}"
                ),
                action="MINOR_LOG_AND_PASS",
                priority=3,
            ),
            SLARuleSpec(
                rule_id="SLA-04",
                name="High-Confidence Clean -> Auto-Approve",
                condition=(
                    f"severity == PASS AND confidence >= {self.settings.qa_confidence_threshold:.2f}"
                ),
                action="AUTO_APPROVE",
                priority=4,
            ),
        ]

    def evaluate_item(
        self,
        item_id: str,
        batch_id: str,
        serial_number: str,
        surface_type: str,
        predicted_severity: str,
        confidence: float,
        defect_area_mm2: float,
        alignment_angle_deg: float,
        detections: List[Dict[str, Any]],
    ) -> Tuple[str, Dict[str, Any], Optional[Dict[str, Any]]]:
        """Determine target item state (`APPROVED`, `QUARANTINED`, `PENDING_QA`), emit automation event,
        and dispatch signed webhook if quarantined."""
        now = utc_now_iso()
        skew = abs(alignment_angle_deg)

        if confidence < self.settings.qa_confidence_threshold:
            target_state = ItemState.PENDING_QA.value
            rule_id = "SLA-01"
            rule_name = "Confidence Gate -> HITL QA Review"
            action = "ROUTE_TO_HITL_QA"
            reason = (
                f"Model confidence {confidence:.3f} below SLA threshold "
                f"{self.settings.qa_confidence_threshold:.2f}; escalated to Human QA."
            )
        elif (
            predicted_severity == SeverityLevel.CRITICAL_REJECT.value
            or defect_area_mm2 >= self.settings.critical_area_mm2_threshold
            or skew > self.settings.max_alignment_skew_deg
        ):
            target_state = ItemState.QUARANTINED.value
            rule_id = "SLA-02"
            rule_name = "Critical Defect -> Quarantine & Signed ERP Webhook"
            action = "QUARANTINE_AND_WEBHOOK"
            reason = (
                f"Critical defect triage triggered (severity={predicted_severity}, "
                f"area={defect_area_mm2:.2f}mm2, skew={alignment_angle_deg:+.2f}deg)."
            )
        elif predicted_severity == SeverityLevel.MINOR_COSMETIC.value:
            target_state = ItemState.APPROVED.value
            rule_id = "SLA-03"
            rule_name = "Minor Cosmetic -> Log & Conditional Pass"
            action = "MINOR_LOG_AND_PASS"
            reason = (
                f"Minor cosmetic defect within tolerance (area={defect_area_mm2:.2f}mm2, "
                f"conf={confidence:.3f})."
            )
        else:
            target_state = ItemState.APPROVED.value
            rule_id = "SLA-04"
            rule_name = "High-Confidence Clean -> Auto-Approve"
            action = "AUTO_APPROVE"
            reason = f"Clean inspection passed SLA gate (conf={confidence:.3f})."

        event = {
            "event_id": f"evt_{uuid.uuid4().hex[:12]}",
            "item_id": item_id,
            "batch_id": batch_id,
            "rule_id": rule_id,
            "rule_name": rule_name,
            "action": action,
            "reason": reason,
            "created_at": now,
        }
        self.db.insert_automation_event(event)

        webhook_record: Optional[Dict[str, Any]] = None
        if target_state == ItemState.QUARANTINED.value:
            webhook_record = self.dispatch_quarantine_webhook(
                event_id=event["event_id"],
                item_id=item_id,
                batch_id=batch_id,
                serial_number=serial_number,
                surface_type=surface_type,
                severity=predicted_severity,
                confidence=confidence,
                defect_area_mm2=defect_area_mm2,
                alignment_angle_deg=alignment_angle_deg,
                detections=detections,
            )

        return target_state, event, webhook_record

    def dispatch_quarantine_webhook(
        self,
        event_id: str,
        item_id: str,
        batch_id: str,
        serial_number: str,
        surface_type: str,
        severity: str,
        confidence: float,
        defect_area_mm2: float,
        alignment_angle_deg: float,
        detections: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        payload = {
            "event_type": "VISIONGUARD_CRITICAL_QUARANTINE",
            "event_id": event_id,
            "item_id": item_id,
            "batch_id": batch_id,
            "serial_number": serial_number,
            "surface_type": surface_type,
            "severity": severity,
            "confidence": confidence,
            "defect_area_mm2": defect_area_mm2,
            "alignment_angle_deg": alignment_angle_deg,
            "detections": [
                {
                    "defect_class": d["defect_class"],
                    "area_mm2": d["area_mm2"],
                    "confidence": d["confidence"],
                }
                for d in detections
            ],
            "timestamp": utc_now_iso(),
        }
        payload_bytes = json.dumps(payload, separators=(",", ":"), sort_keys=True).encode("utf-8")
        signature = compute_hmac_signature(self.settings.webhook_secret, payload_bytes)
        headers = {
            "Content-Type": "application/json",
            "X-VisionGuard-Signature": signature,
            "X-VisionGuard-Event": event_id,
        }

        target_url = self.settings.erp_webhook_path
        t0 = time.perf_counter()
        if self.webhook_dispatcher is not None:
            status_code, resp_body, latency_ms = self.webhook_dispatcher(target_url, payload_bytes, headers)
        else:
            # Direct in-process fallback verification when running headless without FastAPI HTTP loop
            verified = verify_hmac_signature(self.settings.webhook_secret, payload_bytes, signature)
            self.db.insert_erp_alert(
                {
                    "alert_id": f"erp_{uuid.uuid4().hex[:10]}",
                    "item_id": item_id,
                    "batch_id": batch_id,
                    "severity": severity,
                    "defect_area_mm2": defect_area_mm2,
                    "signature_verified": verified,
                    "payload_json": payload_bytes.decode("utf-8"),
                    "received_at": utc_now_iso(),
                }
            )
            status_code = 200
            resp_body = json.dumps({"status": "acknowledged", "signature_verified": verified})
            latency_ms = round((time.perf_counter() - t0) * 1000.0, 3)

        delivery = {
            "delivery_id": f"wh_{uuid.uuid4().hex[:12]}",
            "event_id": event_id,
            "item_id": item_id,
            "target_url": target_url,
            "payload_json": payload_bytes.decode("utf-8"),
            "hmac_signature": signature,
            "http_status": status_code,
            "response_body": resp_body,
            "latency_ms": round(latency_ms, 3),
            "delivered_at": utc_now_iso(),
        }
        self.db.insert_webhook_delivery(delivery)
        return delivery

    def resolve_qa_item(
        self, item_id: str, decision: str, reviewer: str, notes: str
    ) -> Optional[Dict[str, Any]]:
        item = self.db.get_item(item_id)
        if not item:
            return None

        new_state = (
            ItemState.QA_RESOLVED_PASS.value
            if decision.upper() == "PASS"
            else ItemState.QA_RESOLVED_REJECT.value
        )
        updated = self.db.update_item_qa(item_id, new_state, reviewer, notes)
        event = {
            "event_id": f"evt_{uuid.uuid4().hex[:12]}",
            "item_id": item_id,
            "batch_id": item["batch_id"],
            "rule_id": "SLA-QA-OVERRIDE",
            "rule_name": "Human-in-the-Loop QA Resolution",
            "action": "QA_MANUAL_OVERRIDE",
            "reason": f"QA operator '{reviewer}' resolved item as {decision.upper()}: {notes}",
            "created_at": utc_now_iso(),
        }
        self.db.insert_automation_event(event)

        if new_state == ItemState.QA_RESOLVED_REJECT.value:
            self.dispatch_quarantine_webhook(
                event_id=event["event_id"],
                item_id=item_id,
                batch_id=item["batch_id"],
                serial_number=item["serial_number"],
                surface_type=item["surface_type"],
                severity="CRITICAL_REJECT",
                confidence=float(item["confidence"]),
                defect_area_mm2=float(item["defect_area_mm2"]),
                alignment_angle_deg=float(item["alignment_angle_deg"]),
                detections=item.get("detections", []),
            )

        self.db.finalize_batch(item["batch_id"])
        return updated

    def generate_batch_audit_report(self, batch_id: str) -> Path:
        batch = self.db.finalize_batch(batch_id)
        items = self.db.list_items(batch_id=batch_id, limit=500)
        events = [e for e in self.db.list_automation_events(limit=500) if e["batch_id"] == batch_id]

        report_data = {
            "report_title": "VisionGuard AI — Batch Quality Audit Report",
            "generated_at": utc_now_iso(),
            "batch": batch,
            "sla_thresholds": {
                "qa_confidence_threshold": self.settings.qa_confidence_threshold,
                "critical_area_mm2_threshold": self.settings.critical_area_mm2_threshold,
                "max_alignment_skew_deg": self.settings.max_alignment_skew_deg,
            },
            "summary_metrics": {
                "total_items": batch["total_items"],
                "passed_count": batch["passed_count"],
                "quarantined_count": batch["quarantined_count"],
                "pending_qa_count": batch["pending_qa_count"],
                "pass_rate_pct": round(
                    100.0 * batch["passed_count"] / max(1, batch["total_items"]), 2
                ),
                "avg_latency_ms": batch["avg_latency_ms"],
                "automation_events_count": len(events),
            },
            "quarantined_items": [
                {
                    "item_id": i["item_id"],
                    "serial_number": i["serial_number"],
                    "surface_type": i["surface_type"],
                    "severity": i["predicted_severity"],
                    "confidence": i["confidence"],
                    "defect_area_mm2": i["defect_area_mm2"],
                    "alignment_angle_deg": i["alignment_angle_deg"],
                }
                for i in items
                if i["state"] in ("QUARANTINED", "QA_RESOLVED_REJECT")
            ],
            "pending_qa_items": [
                {
                    "item_id": i["item_id"],
                    "serial_number": i["serial_number"],
                    "surface_type": i["surface_type"],
                    "confidence": i["confidence"],
                    "defect_area_mm2": i["defect_area_mm2"],
                }
                for i in items
                if i["state"] == "PENDING_QA"
            ],
        }

        report_dir = self.settings.data_dir / "reports"
        report_dir.mkdir(parents=True, exist_ok=True)
        json_path = report_dir / f"{batch_id}_audit.json"
        json_path.write_text(json.dumps(report_data, indent=2), encoding="utf-8")

        md_lines = [
            f"# VisionGuard AI Audit Report — `{batch['batch_name']}` (`{batch_id}`)",
            "",
            f"- **Product Line**: `{batch['product_line']}`",
            f"- **Total Inspected**: {batch['total_items']}",
            f"- **Approved / Passed**: {batch['passed_count']} ({report_data['summary_metrics']['pass_rate_pct']}%)",
            f"- **Quarantined**: {batch['quarantined_count']}",
            f"- **Pending HITL QA**: {batch['pending_qa_count']}",
            f"- **Mean End-to-End Latency**: {batch['avg_latency_ms']:.2f} ms",
            "",
            "## Quarantined Items",
        ]
        for q in report_data["quarantined_items"]:
            md_lines.append(
                f"- `{q['serial_number']}` ({q['surface_type']}) — {q['severity']} | "
                f"conf={q['confidence']:.3f} | area={q['defect_area_mm2']:.2f} mm2 | skew={q['alignment_angle_deg']:+.2f} deg"
            )
        md_path = report_dir / f"{batch_id}_audit.md"
        md_path.write_text("\n".join(md_lines) + "\n", encoding="utf-8")

        self.db.finalize_batch(batch_id, audit_report_path=str(json_path))
        return json_path
