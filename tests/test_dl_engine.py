"""Tests for ONNX Runtime YOLOv8-lite detector and MobileNetV3-INT8 severity classifier."""

from __future__ import annotations

from app.config import Settings
from app.cv_pipeline import CVMetrologyPipeline
from app.dl_engine import DLInferenceEngine
from scripts.generate_sample_batch import generate_synthetic_frame


def test_onnx_models_build_and_infer(tmp_path) -> None:
    settings = Settings(data_dir=tmp_path / "data", models_dir=tmp_path / "models")
    cv_pipe = CVMetrologyPipeline(settings)
    dl = DLInferenceEngine(settings)

    assert dl.detector_path.exists()
    assert dl.classifier_int8_path.exists()
    assert dl.classifier_fp32_path.exists()

    info = dl.get_model_info()
    assert "onnxruntime" in info["runtime"]
    assert "INT8" in info["classifier_quantization"]

    # Clean metallic plate -> PASS
    clean_frame, _ = generate_synthetic_frame(surface_type="metallic", defect_kind="clean", seed=11)
    clean_cv = cv_pipe.process(clean_frame)
    clean_dl = dl.infer(clean_cv)
    assert clean_dl.predicted_severity == "PASS"
    assert clean_dl.confidence >= 0.80
    assert clean_dl.dl_latency_ms > 0.0

    # Misaligned PCB -> CRITICAL_REJECT
    bad_frame, _ = generate_synthetic_frame(surface_type="pcb", defect_kind="misaligned", seed=22)
    bad_cv = cv_pipe.process(bad_frame)
    bad_dl = dl.infer(bad_cv)
    assert bad_dl.predicted_severity == "CRITICAL_REJECT"
    assert bad_dl.confidence >= 0.80
    assert len(bad_dl.detections) >= 1
