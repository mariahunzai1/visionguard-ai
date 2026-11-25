"""Tests for OpenCV CLAHE, contour ROI alignment, and pixel-to-mm metrology."""

from __future__ import annotations

import numpy as np

from app.config import Settings
from app.cv_pipeline import CVMetrologyPipeline
from scripts.generate_sample_batch import generate_synthetic_frame


def test_clahe_and_roi_alignment_calibration(tmp_path) -> None:
    settings = Settings(data_dir=tmp_path / "data", models_dir=tmp_path / "models")
    pipe = CVMetrologyPipeline(settings)

    frame, _ = generate_synthetic_frame(surface_type="metallic", defect_kind="clean", seed=101)
    res = pipe.process(frame)

    assert res.rectified_bgr.shape == (settings.canonical_height_px, settings.canonical_width_px, 3)
    assert 4.0 <= res.px_per_mm <= 5.5
    assert abs(res.alignment_angle_deg) < 4.5
    assert res.cv_latency_ms >= 0.0
    assert res.contrast_gain > 0.0


def test_crack_metrology_extraction_and_overlay(tmp_path) -> None:
    settings = Settings(data_dir=tmp_path / "data", models_dir=tmp_path / "models")
    pipe = CVMetrologyPipeline(settings)

    frame, _ = generate_synthetic_frame(surface_type="metallic", defect_kind="crack", seed=202)
    res = pipe.process(frame)

    assert len(res.candidates) >= 1
    assert res.total_defect_area_mm2 > 2.0
    assert res.max_defect_length_mm > 3.0

    overlay = pipe.render_annotated_overlay(
        rectified_bgr=res.rectified_bgr,
        detections=[
            {
                "defect_class": "crack",
                "severity": "CRITICAL_REJECT",
                "confidence": 0.95,
                "bbox_x1": res.candidates[0].bbox_x1,
                "bbox_y1": res.candidates[0].bbox_y1,
                "bbox_x2": res.candidates[0].bbox_x2,
                "bbox_y2": res.candidates[0].bbox_y2,
                "area_mm2": res.candidates[0].area_mm2,
                "width_mm": res.candidates[0].width_mm,
                "height_mm": res.candidates[0].height_mm,
            }
        ],
        item_id="itm_test",
        state="QUARANTINED",
        severity="CRITICAL_REJECT",
        confidence=0.95,
        px_per_mm=res.px_per_mm,
        alignment_angle_deg=res.alignment_angle_deg,
        total_latency_ms=14.2,
    )
    assert overlay.shape == res.rectified_bgr.shape
    assert not np.array_equal(overlay, res.rectified_bgr)


def test_misaligned_part_skew_detection(tmp_path) -> None:
    settings = Settings(data_dir=tmp_path / "data", models_dir=tmp_path / "models")
    pipe = CVMetrologyPipeline(settings)

    frame, _ = generate_synthetic_frame(surface_type="pcb", defect_kind="misaligned", seed=303)
    res = pipe.process(frame)

    assert abs(res.alignment_angle_deg) > settings.max_alignment_skew_deg
