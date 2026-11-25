"""Pydantic domain schemas for inspection, metrology, automation, and telemetry."""

from __future__ import annotations

from enum import Enum
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class SeverityLevel(str, Enum):
    PASS = "PASS"
    MINOR_COSMETIC = "MINOR_COSMETIC"
    CRITICAL_REJECT = "CRITICAL_REJECT"


class ItemState(str, Enum):
    INGESTED = "INGESTED"
    ANALYZED = "ANALYZED"
    APPROVED = "APPROVED"
    QUARANTINED = "QUARANTINED"
    PENDING_QA = "PENDING_QA"
    QA_RESOLVED_PASS = "QA_RESOLVED_PASS"
    QA_RESOLVED_REJECT = "QA_RESOLVED_REJECT"


class DefectClass(str, Enum):
    CLEAN = "clean"
    SCRATCH = "scratch"
    CRACK = "crack"
    MISALIGNED = "misaligned"


class DefectDetectionOut(BaseModel):
    detection_id: str
    defect_class: str
    severity: str
    confidence: float
    bbox_x1: int
    bbox_y1: int
    bbox_x2: int
    bbox_y2: int
    area_px: float
    area_mm2: float
    width_mm: float
    height_mm: float
    source: str


class InspectionItemOut(BaseModel):
    item_id: str
    batch_id: str
    serial_number: str
    surface_type: str
    raw_image_path: str
    annotated_image_path: str
    annotated_image_url: Optional[str] = None
    raw_image_url: Optional[str] = None
    state: str
    predicted_severity: str
    confidence: float
    defect_area_mm2: float
    max_defect_length_mm: float
    px_per_mm: float
    alignment_angle_deg: float
    cv_latency_ms: float
    dl_latency_ms: float
    total_latency_ms: float
    qa_reviewer: Optional[str] = None
    qa_notes: Optional[str] = None
    created_at: str
    updated_at: str
    detections: List[DefectDetectionOut] = Field(default_factory=list)


class AutomationEventOut(BaseModel):
    event_id: str
    item_id: str
    batch_id: str
    rule_id: str
    rule_name: str
    action: str
    reason: str
    created_at: str


class WebhookDeliveryOut(BaseModel):
    delivery_id: str
    event_id: str
    item_id: str
    target_url: str
    payload_json: str
    hmac_signature: str
    http_status: int
    response_body: str
    latency_ms: float
    delivered_at: str


class BatchSummaryOut(BaseModel):
    batch_id: str
    batch_name: str
    product_line: str
    total_items: int
    passed_count: int
    quarantined_count: int
    pending_qa_count: int
    avg_latency_ms: float
    audit_report_path: Optional[str] = None
    created_at: str
    completed_at: Optional[str] = None


class QAResolutionRequest(BaseModel):
    decision: str = Field(..., pattern="^(PASS|REJECT)$")
    reviewer: str = Field(default="qa_lead_operator", min_length=2, max_length=64)
    notes: str = Field(default="Verified via console metrology overlay", max_length=500)


class BatchRunRequest(BaseModel):
    batch_name: str = Field(default="Batch-Inspect-50")
    product_line: str = Field(default="mixed")
    count: int = Field(default=50, ge=1, le=200)
    seed: int = Field(default=42)


class SLARuleSpec(BaseModel):
    rule_id: str
    name: str
    condition: str
    action: str
    priority: int
    enabled: bool = True


class TelemetrySnapshotOut(BaseModel):
    total_batches: int
    total_items: int
    approved_count: int
    quarantined_count: int
    pending_qa_count: int
    qa_resolved_pass_count: int
    qa_resolved_reject_count: int
    avg_cv_latency_ms: float
    avg_dl_latency_ms: float
    avg_total_latency_ms: float
    p95_total_latency_ms: float
    webhook_success_count: int
    webhook_total_count: int
    severity_breakdown: Dict[str, int]
    defect_class_breakdown: Dict[str, int]
    surface_breakdown: Dict[str, int]
    latency_series: List[Dict[str, Any]]
    model_info: Dict[str, Any]
