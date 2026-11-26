"""End-to-end FastAPI integration tests covering upload, batch triage, QA resolution, and HMAC webhook receiver."""

from __future__ import annotations

import sys

import cv2
from fastapi.testclient import TestClient

from app.automation_engine import compute_hmac_signature
from app.config import Settings
from app.main import create_app
from scripts.generate_sample_batch import generate_synthetic_frame


def _test_client(tmp_path) -> TestClient:
    settings = Settings(
        data_dir=tmp_path / "data",
        models_dir=tmp_path / "models",
        database_url=f"sqlite:///{(tmp_path / 'api_test.db').as_posix()}",
    )
    app = create_app(settings)
    return TestClient(app)


def test_api_inspect_batch_qa_and_erp_receiver(tmp_path) -> None:
    assert "torch" not in sys.modules
    with _test_client(tmp_path) as client:
        # 1. Startup auto-seeding populates 50-item batch on empty DB + Console HTML & health
        assert client.get("/").status_code == 200
        health = client.get("/api/health").json()
        assert health["status"] == "ok"

        seeded_telemetry = client.get("/api/telemetry").json()
        assert seeded_telemetry["total_batches"] == 1
        assert seeded_telemetry["total_items"] == 50
        assert seeded_telemetry["webhook_total_count"] >= 1
        assert seeded_telemetry["pending_qa_count"] >= 1

        # 2. Upload guardrails: reject disallowed MIME (415) and >5MB upload (413)
        bad_mime = client.post(
            "/api/inspect",
            files={"file": ("notes.txt", b"hello world", "text/plain")},
            data={"surface_type": "metallic"},
        )
        assert bad_mime.status_code == 415

        oversized = client.post(
            "/api/inspect",
            files={"file": ("huge.png", b"\x89PNG" + b"0" * (5 * 1024 * 1024 + 16), "image/png")},
            data={"surface_type": "metallic"},
        )
        assert oversized.status_code == 413

        # 3. Valid multipart image upload inspection
        frame, _ = generate_synthetic_frame(surface_type="metallic", defect_kind="crack", seed=77)
        _, png_buf = cv2.imencode(".png", frame)
        resp = client.post(
            "/api/inspect",
            files={"file": ("crack_test.png", png_buf.tobytes(), "image/png")},
            data={"surface_type": "metallic", "serial_number": "VG-TEST-01"},
        )
        assert resp.status_code == 200
        item = resp.json()
        assert item["serial_number"] == "VG-TEST-01"
        assert item["annotated_image_url"].startswith("/overlays/")
        assert "\\" not in item["annotated_image_url"]
        assert client.get(item["annotated_image_url"]).status_code == 200

        # Verify fallback overlay endpoint regenerates missing overlay file on the fly
        overlay_file = client.app.state.service.overlays_dir / f"{item['item_id']}_overlay.png"
        overlay_file.unlink(missing_ok=True)
        fallback_resp = client.get(f"/api/items/{item['item_id']}/overlay")
        assert fallback_resp.status_code == 200
        assert fallback_resp.headers["content-type"] == "image/png"
        assert overlay_file.exists()

        # 4. Run mini batch via API & verify batch count bounds (<= 10)
        batch_resp = client.post(
            "/api/batches/run",
            json={"batch_name": "E2E-Batch", "product_line": "mixed", "count": 15, "seed": 42},
        )
        assert batch_resp.status_code == 200
        assert batch_resp.json()["items_processed"] == 15

        telemetry = client.get("/api/telemetry").json()
        assert telemetry["total_items"] == 66
        assert telemetry["avg_total_latency_ms"] > 0.0

        # Trigger 9 more small batches (total 12 created) -> verify pruning caps retained batches at 10
        for idx in range(9):
            r = client.post(
                "/api/batches/run",
                json={"batch_name": f"Prune-Check-{idx}", "product_line": "metallic", "count": 1, "seed": idx + 1},
            )
            assert r.status_code == 200
        assert len(client.get("/api/batches?limit=20").json()) == 10

        # 5. ERP receiver HMAC enforcement
        bad_wh = client.post(
            "/api/webhooks/erp-receiver",
            content=b'{"item_id":"x"}',
            headers={"X-VisionGuard-Signature": "sha256=invalid"},
        )
        assert bad_wh.status_code == 401

        payload = b'{"item_id":"itm_ok","batch_id":"bat_1","severity":"CRITICAL_REJECT","defect_area_mm2":14.0}'
        sig = compute_hmac_signature(client.app.state.service.settings.webhook_secret, payload)
        good_wh = client.post(
            "/api/webhooks/erp-receiver",
            content=payload,
            headers={"X-VisionGuard-Signature": sig},
        )
        assert good_wh.status_code == 200
        assert good_wh.json()["signature_verified"] is True

        # 6. QA resolution if any PENDING_QA item exists (test POST + PATCH)
        qa_items = client.get("/api/qa-queue").json()
        if qa_items:
            qid = qa_items[0]["item_id"]
            res = client.post(
                f"/api/qa-queue/{qid}/resolve",
                json={"decision": "PASS", "reviewer": "qa_e2e", "notes": "Cleared"},
            )
            assert res.status_code == 200
            assert res.json()["state"] == "QA_RESOLVED_PASS"
            assert all(x["item_id"] != qid for x in client.get("/api/qa-queue").json())
        if len(qa_items) > 1:
            qid2 = qa_items[1]["item_id"]
            res2 = client.patch(
                f"/api/qa-queue/{qid2}/resolve",
                json={"decision": "REJECT", "reviewer": "qa_e2e", "notes": "Quarantined"},
            )
            assert res2.status_code == 200
            assert res2.json()["state"] == "QA_RESOLVED_REJECT"
