"""Application configuration and SLA threshold definitions."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent


@dataclass(frozen=True)
class Settings:
    app_title: str = "VisionGuard AI — Autonomous Visual Defect Inspection & Triage"
    version: str = "1.0.0"

    # Storage & Database (SQLite by default, PostgreSQL when DATABASE_URL is set)
    data_dir: Path = field(default_factory=lambda: BASE_DIR / "data")
    models_dir: Path = field(default_factory=lambda: BASE_DIR / "models")
    static_dir: Path = field(default_factory=lambda: BASE_DIR / "static")
    database_url: str = field(
        default_factory=lambda: os.getenv(
            "DATABASE_URL",
            f"sqlite:///{(BASE_DIR / 'data' / 'visionguard.db').as_posix()}",
        )
    )

    # Computer Vision Metrology Calibration
    canonical_width_px: int = 512
    canonical_height_px: int = 384
    nominal_part_width_mm: float = 100.0
    default_px_per_mm: float = 4.6
    clahe_clip_limit: float = 2.5
    clahe_tile_grid: tuple[int, int] = (8, 8)

    # Deep Learning ONNX Engine
    detector_input_size: tuple[int, int] = (256, 192)  # (W, H)
    classifier_input_size: tuple[int, int] = (128, 128)  # (W, H)
    use_quantized_classifier: bool = True

    # Deterministic SLA Triage Rules
    qa_confidence_threshold: float = 0.80
    critical_area_mm2_threshold: float = 12.0
    minor_area_mm2_threshold: float = 1.5
    max_alignment_skew_deg: float = 5.5

    # Webhook Security & Target
    webhook_secret: str = field(
        default_factory=lambda: os.getenv("VISIONGUARD_WEBHOOK_SECRET", "vg_prod_hmac_secret_2026_9f8e7d")
    )
    erp_webhook_path: str = "/api/webhooks/erp-receiver"

    # Public Demo / Container Guardrails
    auto_seed_on_startup: bool = True
    max_upload_bytes: int = 5 * 1024 * 1024  # 5 MB
    allowed_image_mimes: tuple[str, ...] = ("image/jpeg", "image/png", "image/webp")
    max_retained_batches: int = 10

    def ensure_dirs(self) -> None:
        self.data_dir.mkdir(parents=True, exist_ok=True)
        (self.data_dir / "annotated").mkdir(parents=True, exist_ok=True)
        (self.data_dir / "reports").mkdir(parents=True, exist_ok=True)
        (self.data_dir / "batches").mkdir(parents=True, exist_ok=True)
        self.models_dir.mkdir(parents=True, exist_ok=True)


def get_settings() -> Settings:
    settings = Settings()
    settings.ensure_dirs()
    return settings
