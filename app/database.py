"""Normalized database persistence layer supporting SQLite and PostgreSQL."""

from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timezone
import json
from pathlib import Path
import sqlite3
import threading
from typing import Any, Dict, Iterator, List, Optional

try:
    import psycopg2
    import psycopg2.extras
except ImportError:  # pragma: no cover
    psycopg2 = None


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


SCHEMA_SQL_STATEMENTS = [
    """
    CREATE TABLE IF NOT EXISTS inspection_batches (
        batch_id TEXT PRIMARY KEY,
        batch_name TEXT NOT NULL,
        product_line TEXT NOT NULL,
        total_items INTEGER NOT NULL DEFAULT 0,
        passed_count INTEGER NOT NULL DEFAULT 0,
        quarantined_count INTEGER NOT NULL DEFAULT 0,
        pending_qa_count INTEGER NOT NULL DEFAULT 0,
        avg_latency_ms REAL NOT NULL DEFAULT 0.0,
        audit_report_path TEXT,
        created_at TEXT NOT NULL,
        completed_at TEXT
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS inspection_items (
        item_id TEXT PRIMARY KEY,
        batch_id TEXT NOT NULL REFERENCES inspection_batches(batch_id) ON DELETE CASCADE,
        serial_number TEXT NOT NULL,
        surface_type TEXT NOT NULL,
        raw_image_path TEXT NOT NULL,
        annotated_image_path TEXT NOT NULL,
        state TEXT NOT NULL,
        predicted_severity TEXT NOT NULL,
        confidence REAL NOT NULL,
        defect_area_mm2 REAL NOT NULL DEFAULT 0.0,
        max_defect_length_mm REAL NOT NULL DEFAULT 0.0,
        px_per_mm REAL NOT NULL,
        alignment_angle_deg REAL NOT NULL DEFAULT 0.0,
        cv_latency_ms REAL NOT NULL,
        dl_latency_ms REAL NOT NULL,
        total_latency_ms REAL NOT NULL,
        qa_reviewer TEXT,
        qa_notes TEXT,
        created_at TEXT NOT NULL,
        updated_at TEXT NOT NULL
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS defect_detections (
        detection_id TEXT PRIMARY KEY,
        item_id TEXT NOT NULL REFERENCES inspection_items(item_id) ON DELETE CASCADE,
        defect_class TEXT NOT NULL,
        severity TEXT NOT NULL,
        confidence REAL NOT NULL,
        bbox_x1 INTEGER NOT NULL,
        bbox_y1 INTEGER NOT NULL,
        bbox_x2 INTEGER NOT NULL,
        bbox_y2 INTEGER NOT NULL,
        area_px REAL NOT NULL,
        area_mm2 REAL NOT NULL,
        width_mm REAL NOT NULL,
        height_mm REAL NOT NULL,
        source TEXT NOT NULL
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS automation_events (
        event_id TEXT PRIMARY KEY,
        item_id TEXT NOT NULL REFERENCES inspection_items(item_id) ON DELETE CASCADE,
        batch_id TEXT NOT NULL REFERENCES inspection_batches(batch_id) ON DELETE CASCADE,
        rule_id TEXT NOT NULL,
        rule_name TEXT NOT NULL,
        action TEXT NOT NULL,
        reason TEXT NOT NULL,
        created_at TEXT NOT NULL
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS webhook_deliveries (
        delivery_id TEXT PRIMARY KEY,
        event_id TEXT NOT NULL REFERENCES automation_events(event_id) ON DELETE CASCADE,
        item_id TEXT NOT NULL REFERENCES inspection_items(item_id) ON DELETE CASCADE,
        target_url TEXT NOT NULL,
        payload_json TEXT NOT NULL,
        hmac_signature TEXT NOT NULL,
        http_status INTEGER NOT NULL,
        response_body TEXT NOT NULL,
        latency_ms REAL NOT NULL,
        delivered_at TEXT NOT NULL
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS erp_alerts (
        alert_id TEXT PRIMARY KEY,
        item_id TEXT NOT NULL,
        batch_id TEXT NOT NULL,
        severity TEXT NOT NULL,
        defect_area_mm2 REAL NOT NULL,
        signature_verified INTEGER NOT NULL,
        payload_json TEXT NOT NULL,
        received_at TEXT NOT NULL
    )
    """,
]


class DatabaseManager:
    """Thread-safe SQLite/PostgreSQL database manager with normalized inspection schema."""

    def __init__(self, database_url: str) -> None:
        self.database_url = database_url
        self.is_postgres = database_url.startswith("postgresql://") or database_url.startswith("postgres://")
        self._lock = threading.RLock()
        if not self.is_postgres:
            db_path = database_url.replace("sqlite:///", "", 1)
            Path(db_path).parent.mkdir(parents=True, exist_ok=True)
            self._sqlite_path = db_path
        else:
            self._sqlite_path = ""
        self._init_schema()

    def _adapt_sql(self, sql: str) -> str:
        if self.is_postgres:
            return sql.replace("?", "%s")
        return sql

    @contextmanager
    def connection(self) -> Iterator[Any]:
        with self._lock:
            if self.is_postgres:
                if psycopg2 is None:  # pragma: no cover
                    raise RuntimeError("psycopg2 is required for PostgreSQL connections")
                conn = psycopg2.connect(self.database_url)
                try:
                    yield conn
                    conn.commit()
                except Exception:
                    conn.rollback()
                    raise
                finally:
                    conn.close()
            else:
                conn = sqlite3.connect(self._sqlite_path, check_same_thread=False)
                conn.row_factory = sqlite3.Row
                conn.execute("PRAGMA foreign_keys = ON")
                try:
                    yield conn
                    conn.commit()
                except Exception:
                    conn.rollback()
                    raise
                finally:
                    conn.close()

    def _execute(self, conn: Any, sql: str, params: tuple = ()) -> Any:
        adapted = self._adapt_sql(sql)
        if self.is_postgres:
            cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
            cur.execute(adapted, params)
            return cur
        return conn.execute(adapted, params)

    def _init_schema(self) -> None:
        with self.connection() as conn:
            for stmt in SCHEMA_SQL_STATEMENTS:
                self._execute(conn, stmt)

    def create_batch(
        self, batch_id: str, batch_name: str, product_line: str
    ) -> Dict[str, Any]:
        now = utc_now_iso()
        with self.connection() as conn:
            self._execute(
                conn,
                """
                INSERT INTO inspection_batches (
                    batch_id, batch_name, product_line, created_at
                ) VALUES (?, ?, ?, ?)
                """,
                (batch_id, batch_name, product_line, now),
            )
        return self.get_batch(batch_id)  # type: ignore[return-value]

    def finalize_batch(self, batch_id: str, audit_report_path: Optional[str] = None) -> Dict[str, Any]:
        now = utc_now_iso()
        with self.connection() as conn:
            row = self._execute(
                conn,
                """
                SELECT
                    COUNT(*) AS total_items,
                    COALESCE(SUM(CASE WHEN state IN ('APPROVED', 'QA_RESOLVED_PASS') THEN 1 ELSE 0 END), 0) AS passed_count,
                    COALESCE(SUM(CASE WHEN state IN ('QUARANTINED', 'QA_RESOLVED_REJECT') THEN 1 ELSE 0 END), 0) AS quarantined_count,
                    COALESCE(SUM(CASE WHEN state = 'PENDING_QA' THEN 1 ELSE 0 END), 0) AS pending_qa_count,
                    COALESCE(AVG(total_latency_ms), 0.0) AS avg_latency_ms
                FROM inspection_items WHERE batch_id = ?
                """,
                (batch_id,),
            ).fetchone()
            stats = dict(row)
            self._execute(
                conn,
                """
                UPDATE inspection_batches
                SET total_items = ?,
                    passed_count = ?,
                    quarantined_count = ?,
                    pending_qa_count = ?,
                    avg_latency_ms = ?,
                    audit_report_path = COALESCE(?, audit_report_path),
                    completed_at = ?
                WHERE batch_id = ?
                """,
                (
                    int(stats["total_items"]),
                    int(stats["passed_count"]),
                    int(stats["quarantined_count"]),
                    int(stats["pending_qa_count"]),
                    round(float(stats["avg_latency_ms"]), 3),
                    audit_report_path,
                    now,
                    batch_id,
                ),
            )
        return self.get_batch(batch_id)  # type: ignore[return-value]

    def get_batch(self, batch_id: str) -> Optional[Dict[str, Any]]:
        with self.connection() as conn:
            row = self._execute(
                conn, "SELECT * FROM inspection_batches WHERE batch_id = ?", (batch_id,)
            ).fetchone()
            return dict(row) if row else None

    def list_batches(self, limit: int = 20) -> List[Dict[str, Any]]:
        with self.connection() as conn:
            rows = self._execute(
                conn,
                "SELECT * FROM inspection_batches ORDER BY created_at DESC LIMIT ?",
                (limit,),
            ).fetchall()
            return [dict(r) for r in rows]

    def count_batches(self) -> int:
        with self.connection() as conn:
            row = self._execute(
                conn, "SELECT COUNT(*) AS c FROM inspection_batches"
            ).fetchone()
            return int(dict(row)["c"]) if row else 0

    def prune_old_batches(self, keep_latest: int = 10) -> Dict[str, List[str]]:
        """Delete batches older than the latest `keep_latest` runs and return removed IDs + file paths."""
        with self.connection() as conn:
            all_rows = self._execute(
                conn,
                "SELECT batch_id, audit_report_path FROM inspection_batches ORDER BY created_at DESC",
            ).fetchall()
            old_rows = all_rows[keep_latest:]
            if not old_rows:
                return {"pruned_batch_ids": [], "file_paths": []}

            pruned_ids: List[str] = []
            file_paths: List[str] = []
            for r in old_rows:
                d = dict(r)
                bid = d["batch_id"]
                pruned_ids.append(bid)
                if d.get("audit_report_path"):
                    file_paths.append(str(d["audit_report_path"]))
                item_rows = self._execute(
                    conn,
                    "SELECT raw_image_path, annotated_image_path FROM inspection_items WHERE batch_id = ?",
                    (bid,),
                ).fetchall()
                for ir in item_rows:
                    item_d = dict(ir)
                    if item_d.get("raw_image_path"):
                        file_paths.append(str(item_d["raw_image_path"]))
                    if item_d.get("annotated_image_path"):
                        file_paths.append(str(item_d["annotated_image_path"]))
                self._execute(conn, "DELETE FROM erp_alerts WHERE batch_id = ?", (bid,))
                self._execute(conn, "DELETE FROM inspection_batches WHERE batch_id = ?", (bid,))

            return {"pruned_batch_ids": pruned_ids, "file_paths": file_paths}

    def insert_item(self, item: Dict[str, Any], detections: List[Dict[str, Any]]) -> None:
        with self.connection() as conn:
            self._execute(
                conn,
                """
                INSERT INTO inspection_items (
                    item_id, batch_id, serial_number, surface_type, raw_image_path,
                    annotated_image_path, state, predicted_severity, confidence,
                    defect_area_mm2, max_defect_length_mm, px_per_mm, alignment_angle_deg,
                    cv_latency_ms, dl_latency_ms, total_latency_ms, qa_reviewer, qa_notes,
                    created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    item["item_id"],
                    item["batch_id"],
                    item["serial_number"],
                    item["surface_type"],
                    item["raw_image_path"],
                    item["annotated_image_path"],
                    item["state"],
                    item["predicted_severity"],
                    float(item["confidence"]),
                    float(item["defect_area_mm2"]),
                    float(item["max_defect_length_mm"]),
                    float(item["px_per_mm"]),
                    float(item["alignment_angle_deg"]),
                    float(item["cv_latency_ms"]),
                    float(item["dl_latency_ms"]),
                    float(item["total_latency_ms"]),
                    item.get("qa_reviewer"),
                    item.get("qa_notes"),
                    item["created_at"],
                    item["updated_at"],
                ),
            )
            for det in detections:
                self._execute(
                    conn,
                    """
                    INSERT INTO defect_detections (
                        detection_id, item_id, defect_class, severity, confidence,
                        bbox_x1, bbox_y1, bbox_x2, bbox_y2, area_px, area_mm2,
                        width_mm, height_mm, source
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        det["detection_id"],
                        item["item_id"],
                        det["defect_class"],
                        det["severity"],
                        float(det["confidence"]),
                        int(det["bbox_x1"]),
                        int(det["bbox_y1"]),
                        int(det["bbox_x2"]),
                        int(det["bbox_y2"]),
                        float(det["area_px"]),
                        float(det["area_mm2"]),
                        float(det["width_mm"]),
                        float(det["height_mm"]),
                        det["source"],
                    ),
                )

    def update_item_qa(
        self, item_id: str, new_state: str, reviewer: str, notes: str
    ) -> Optional[Dict[str, Any]]:
        now = utc_now_iso()
        with self.connection() as conn:
            self._execute(
                conn,
                """
                UPDATE inspection_items
                SET state = ?, qa_reviewer = ?, qa_notes = ?, updated_at = ?
                WHERE item_id = ?
                """,
                (new_state, reviewer, notes, now, item_id),
            )
        return self.get_item(item_id)

    def update_item_state_and_overlay(
        self, item_id: str, new_state: str, annotated_image_path: str
    ) -> None:
        now = utc_now_iso()
        with self.connection() as conn:
            self._execute(
                conn,
                """
                UPDATE inspection_items
                SET state = ?, annotated_image_path = ?, updated_at = ?
                WHERE item_id = ?
                """,
                (new_state, annotated_image_path, now, item_id),
            )

    def get_item(self, item_id: str) -> Optional[Dict[str, Any]]:
        with self.connection() as conn:
            row = self._execute(
                conn, "SELECT * FROM inspection_items WHERE item_id = ?", (item_id,)
            ).fetchone()
            if not row:
                return None
            item = dict(row)
            dets = self._execute(
                conn, "SELECT * FROM defect_detections WHERE item_id = ?", (item_id,)
            ).fetchall()
            item["detections"] = [dict(d) for d in dets]
            return item

    def list_items(
        self,
        batch_id: Optional[str] = None,
        state: Optional[str] = None,
        limit: int = 100,
    ) -> List[Dict[str, Any]]:
        clauses = []
        params: List[Any] = []
        if batch_id:
            clauses.append("batch_id = ?")
            params.append(batch_id)
        if state:
            clauses.append("state = ?")
            params.append(state)
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        params.append(limit)
        with self.connection() as conn:
            rows = self._execute(
                conn,
                f"SELECT * FROM inspection_items {where} ORDER BY created_at DESC LIMIT ?",
                tuple(params),
            ).fetchall()
            items = []
            for r in rows:
                item = dict(r)
                dets = self._execute(
                    conn, "SELECT * FROM defect_detections WHERE item_id = ?", (item["item_id"],)
                ).fetchall()
                item["detections"] = [dict(d) for d in dets]
                items.append(item)
            return items

    def insert_automation_event(self, event: Dict[str, Any]) -> None:
        with self.connection() as conn:
            self._execute(
                conn,
                """
                INSERT INTO automation_events (
                    event_id, item_id, batch_id, rule_id, rule_name, action, reason, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    event["event_id"],
                    event["item_id"],
                    event["batch_id"],
                    event["rule_id"],
                    event["rule_name"],
                    event["action"],
                    event["reason"],
                    event["created_at"],
                ),
            )

    def list_automation_events(self, limit: int = 100) -> List[Dict[str, Any]]:
        with self.connection() as conn:
            rows = self._execute(
                conn,
                "SELECT * FROM automation_events ORDER BY created_at DESC LIMIT ?",
                (limit,),
            ).fetchall()
            return [dict(r) for r in rows]

    def insert_webhook_delivery(self, delivery: Dict[str, Any]) -> None:
        with self.connection() as conn:
            self._execute(
                conn,
                """
                INSERT INTO webhook_deliveries (
                    delivery_id, event_id, item_id, target_url, payload_json,
                    hmac_signature, http_status, response_body, latency_ms, delivered_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    delivery["delivery_id"],
                    delivery["event_id"],
                    delivery["item_id"],
                    delivery["target_url"],
                    delivery["payload_json"],
                    delivery["hmac_signature"],
                    int(delivery["http_status"]),
                    delivery["response_body"],
                    float(delivery["latency_ms"]),
                    delivery["delivered_at"],
                ),
            )

    def list_webhook_deliveries(self, limit: int = 100) -> List[Dict[str, Any]]:
        with self.connection() as conn:
            rows = self._execute(
                conn,
                "SELECT * FROM webhook_deliveries ORDER BY delivered_at DESC LIMIT ?",
                (limit,),
            ).fetchall()
            return [dict(r) for r in rows]

    def insert_erp_alert(self, alert: Dict[str, Any]) -> None:
        with self.connection() as conn:
            self._execute(
                conn,
                """
                INSERT INTO erp_alerts (
                    alert_id, item_id, batch_id, severity, defect_area_mm2,
                    signature_verified, payload_json, received_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    alert["alert_id"],
                    alert["item_id"],
                    alert["batch_id"],
                    alert["severity"],
                    float(alert["defect_area_mm2"]),
                    1 if alert["signature_verified"] else 0,
                    alert["payload_json"],
                    alert["received_at"],
                ),
            )

    def list_erp_alerts(self, limit: int = 50) -> List[Dict[str, Any]]:
        with self.connection() as conn:
            rows = self._execute(
                conn,
                "SELECT * FROM erp_alerts ORDER BY received_at DESC LIMIT ?",
                (limit,),
            ).fetchall()
            return [dict(r) for r in rows]

    def get_telemetry_aggregates(self) -> Dict[str, Any]:
        with self.connection() as conn:
            counts_row = dict(
                self._execute(
                    conn,
                    """
                    SELECT
                        COUNT(*) AS total_items,
                        COALESCE(SUM(CASE WHEN state = 'APPROVED' THEN 1 ELSE 0 END), 0) AS approved_count,
                        COALESCE(SUM(CASE WHEN state = 'QUARANTINED' THEN 1 ELSE 0 END), 0) AS quarantined_count,
                        COALESCE(SUM(CASE WHEN state = 'PENDING_QA' THEN 1 ELSE 0 END), 0) AS pending_qa_count,
                        COALESCE(SUM(CASE WHEN state = 'QA_RESOLVED_PASS' THEN 1 ELSE 0 END), 0) AS qa_resolved_pass_count,
                        COALESCE(SUM(CASE WHEN state = 'QA_RESOLVED_REJECT' THEN 1 ELSE 0 END), 0) AS qa_resolved_reject_count,
                        COALESCE(AVG(cv_latency_ms), 0.0) AS avg_cv_latency_ms,
                        COALESCE(AVG(dl_latency_ms), 0.0) AS avg_dl_latency_ms,
                        COALESCE(AVG(total_latency_ms), 0.0) AS avg_total_latency_ms
                    FROM inspection_items
                    """,
                ).fetchone()
            )
            batch_count = int(
                dict(self._execute(conn, "SELECT COUNT(*) AS c FROM inspection_batches").fetchone())["c"]
            )
            latencies = [
                float(dict(r)["total_latency_ms"])
                for r in self._execute(
                    conn, "SELECT total_latency_ms FROM inspection_items ORDER BY total_latency_ms ASC"
                ).fetchall()
            ]
            p95 = latencies[int(0.95 * (len(latencies) - 1))] if latencies else 0.0

            wh_row = dict(
                self._execute(
                    conn,
                    """
                    SELECT
                        COUNT(*) AS total_wh,
                        COALESCE(SUM(CASE WHEN http_status >= 200 AND http_status < 300 THEN 1 ELSE 0 END), 0) AS ok_wh
                    FROM webhook_deliveries
                    """,
                ).fetchone()
            )
            sev_rows = self._execute(
                conn,
                "SELECT predicted_severity, COUNT(*) AS cnt FROM inspection_items GROUP BY predicted_severity",
            ).fetchall()
            cls_rows = self._execute(
                conn,
                "SELECT defect_class, COUNT(*) AS cnt FROM defect_detections GROUP BY defect_class",
            ).fetchall()
            surf_rows = self._execute(
                conn,
                "SELECT surface_type, COUNT(*) AS cnt FROM inspection_items GROUP BY surface_type",
            ).fetchall()
            series_rows = self._execute(
                conn,
                """
                SELECT serial_number, cv_latency_ms, dl_latency_ms, total_latency_ms, state, confidence
                FROM inspection_items ORDER BY created_at ASC LIMIT 60
                """,
            ).fetchall()

            return {
                "total_batches": batch_count,
                "total_items": int(counts_row["total_items"]),
                "approved_count": int(counts_row["approved_count"]),
                "quarantined_count": int(counts_row["quarantined_count"]),
                "pending_qa_count": int(counts_row["pending_qa_count"]),
                "qa_resolved_pass_count": int(counts_row["qa_resolved_pass_count"]),
                "qa_resolved_reject_count": int(counts_row["qa_resolved_reject_count"]),
                "avg_cv_latency_ms": round(float(counts_row["avg_cv_latency_ms"]), 2),
                "avg_dl_latency_ms": round(float(counts_row["avg_dl_latency_ms"]), 2),
                "avg_total_latency_ms": round(float(counts_row["avg_total_latency_ms"]), 2),
                "p95_total_latency_ms": round(p95, 2),
                "webhook_success_count": int(wh_row["ok_wh"]),
                "webhook_total_count": int(wh_row["total_wh"]),
                "severity_breakdown": {dict(r)["predicted_severity"]: int(dict(r)["cnt"]) for r in sev_rows},
                "defect_class_breakdown": {dict(r)["defect_class"]: int(dict(r)["cnt"]) for r in cls_rows},
                "surface_breakdown": {dict(r)["surface_type"]: int(dict(r)["cnt"]) for r in surf_rows},
                "latency_series": [dict(r) for r in series_rows],
            }
