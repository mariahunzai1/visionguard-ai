"""Tests for SLA automation routing rules, HMAC-SHA256 webhooks, QA resolution, and audit reports."""

from __future__ import annotations

import json
from pathlib import Path

from app.automation_engine import AutomationEngine, compute_hmac_signature, verify_hmac_signature
from app.config import Settings
from app.database import DatabaseManager


def _make_engine(tmp_path: Path):
    settings = Settings(
        data_dir=tmp_path / "data",
        models_dir=tmp_path / "models",
        database_url=f"sqlite:///{(tmp_path / 'test.db').as_posix()}",
    )
    settings.data_dir.mkdir(parents=True, exist_ok=True)
    db = DatabaseManager(settings.database_url)
    db.create_batch("bat_1", "UnitTestBatch", "metallic")
    return settings, db, AutomationEngine(settings, db)


def test_hmac_signature_verification() -> None:
    secret = "super_secret_key"
    payload = b'{"item_id":"itm_1","severity":"CRITICAL_REJECT"}'
    sig = compute_hmac_signature(secret, payload)
    assert sig.startswith("sha256=")
    assert verify_hmac_signature(secret, payload, sig) is True
    assert verify_hmac_signature(secret, payload + b" ", sig) is False
    assert verify_hmac_signature("wrong_secret", payload, sig) is False


def _insert_stub_item(db: DatabaseManager, item_id: str, serial: str) -> None:
    db.insert_item(
        {
            "item_id": item_id,
            "batch_id": "bat_1",
            "serial_number": serial,
            "surface_type": "metallic",
            "raw_image_path": "raw.png",
            "annotated_image_path": "ann.png",
            "state": "ANALYZED",
            "predicted_severity": "PASS",
            "confidence": 0.90,
            "defect_area_mm2": 0.0,
            "max_defect_length_mm": 0.0,
            "px_per_mm": 4.6,
            "alignment_angle_deg": 0.2,
            "cv_latency_ms": 4.0,
            "dl_latency_ms": 7.0,
            "total_latency_ms": 11.0,
            "created_at": "2026-11-25T10:00:00.000+00:00",
            "updated_at": "2026-11-25T10:00:00.000+00:00",
        },
        [],
    )


def test_sla_routing_branches(tmp_path: Path) -> None:
    _, db, eng = _make_engine(tmp_path)
    _insert_stub_item(db, "itm_qa", "SN-1")
    _insert_stub_item(db, "itm_crit", "SN-2")
    _insert_stub_item(db, "itm_ok", "SN-3")

    # 1. Low confidence -> PENDING_QA (SLA-01)
    state_qa, evt_qa, wh_qa = eng.evaluate_item(
        item_id="itm_qa",
        batch_id="bat_1",
        serial_number="SN-1",
        surface_type="metallic",
        predicted_severity="MINOR_COSMETIC",
        confidence=0.62,
        defect_area_mm2=2.1,
        alignment_angle_deg=0.4,
        detections=[],
    )
    assert state_qa == "PENDING_QA"
    assert evt_qa["rule_id"] == "SLA-01"
    assert wh_qa is None

    # 2. Critical reject -> QUARANTINED + Signed Webhook (SLA-02)
    state_crit, evt_crit, wh_crit = eng.evaluate_item(
        item_id="itm_crit",
        batch_id="bat_1",
        serial_number="SN-2",
        surface_type="pcb",
        predicted_severity="CRITICAL_REJECT",
        confidence=0.96,
        defect_area_mm2=18.5,
        alignment_angle_deg=1.1,
        detections=[{"defect_class": "crack", "area_mm2": 18.5, "confidence": 0.96}],
    )
    assert state_crit == "QUARANTINED"
    assert evt_crit["rule_id"] == "SLA-02"
    assert wh_crit is not None
    assert wh_crit["http_status"] == 200
    assert wh_crit["hmac_signature"].startswith("sha256=")

    # 3. High-confidence PASS -> APPROVED (SLA-04)
    state_ok, evt_ok, wh_ok = eng.evaluate_item(
        item_id="itm_ok",
        batch_id="bat_1",
        serial_number="SN-3",
        surface_type="packaging",
        predicted_severity="PASS",
        confidence=0.95,
        defect_area_mm2=0.0,
        alignment_angle_deg=0.3,
        detections=[],
    )
    assert state_ok == "APPROVED"
    assert evt_ok["rule_id"] == "SLA-04"
    assert wh_ok is None


def test_qa_override_and_audit_report(tmp_path: Path) -> None:
    _, db, eng = _make_engine(tmp_path)
    db.insert_item(
        {
            "item_id": "itm_review",
            "batch_id": "bat_1",
            "serial_number": "SN-99",
            "surface_type": "metallic",
            "raw_image_path": "raw.png",
            "annotated_image_path": "ann.png",
            "state": "PENDING_QA",
            "predicted_severity": "MINOR_COSMETIC",
            "confidence": 0.68,
            "defect_area_mm2": 2.4,
            "max_defect_length_mm": 4.2,
            "px_per_mm": 4.6,
            "alignment_angle_deg": 0.5,
            "cv_latency_ms": 5.0,
            "dl_latency_ms": 8.0,
            "total_latency_ms": 13.0,
            "created_at": "2026-11-25T10:00:00.000+00:00",
            "updated_at": "2026-11-25T10:00:00.000+00:00",
        },
        [],
    )

    updated = eng.resolve_qa_item("itm_review", "REJECT", "qa_alice", "Confirmed hairline fracture")
    assert updated is not None
    assert updated["state"] == "QA_RESOLVED_REJECT"
    assert updated["qa_reviewer"] == "qa_alice"

    webhooks = db.list_webhook_deliveries()
    assert len(webhooks) == 1
    assert webhooks[0]["item_id"] == "itm_review"

    report_path = eng.generate_batch_audit_report("bat_1")
    assert report_path.exists()
    data = json.loads(report_path.read_text(encoding="utf-8"))
    assert data["summary_metrics"]["quarantined_count"] == 1
