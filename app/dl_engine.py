"""Deep Learning ONNX Runtime Inference Engine (CPU-only, zero PyTorch runtime dependency).

Loads & executes two pre-exported ONNX neural networks via `onnxruntime.InferenceSession`:
1. `defect_detector_yolov8_lite.onnx`: Multi-grid spatial defect localizer (6x8 grid over 256x192 input)
   predicting objectness, normalized box geometry, and defect class logits.
2. `severity_mobilenetv3_int8.onnx`: INT8 weight-quantized (`DequantizeLinear`) depthwise-separable
   MobileNetV3-style classifier grading severity (`PASS`, `MINOR_COSMETIC`, `CRITICAL_REJECT`).

Offline model training and ONNX export live strictly in `scripts/train_export_onnx.py`.
"""

from __future__ import annotations

from dataclasses import dataclass
import shutil
import time
from typing import Any, Dict, List, Tuple

import cv2
import numpy as np
import onnxruntime as ort

from app.config import BASE_DIR, Settings
from app.cv_pipeline import CVPipelineResult


SEVERITY_LABELS = ("PASS", "MINOR_COSMETIC", "CRITICAL_REJECT")
DEFECT_CLASSES = ("clean", "scratch", "crack", "misaligned")


@dataclass
class DLInferenceResult:
    predicted_severity: str
    confidence: float
    severity_probs: Dict[str, float]
    detections: List[Dict[str, Any]]
    dl_latency_ms: float
    detector_latency_ms: float
    classifier_latency_ms: float


def _softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - np.max(logits)
    exp = np.exp(shifted)
    return exp / np.sum(exp)


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -20.0, 20.0)))


class DLInferenceEngine:
    """ONNX Runtime execution engine for YOLOv8-lite detection and MobileNetV3-INT8 severity grading."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.detector_path = settings.models_dir / "defect_detector_yolov8_lite.onnx"
        self.classifier_int8_path = settings.models_dir / "severity_mobilenetv3_int8.onnx"
        self.classifier_fp32_path = settings.models_dir / "severity_mobilenetv3_fp32.onnx"

        self._ensure_models_ready()

        opts = ort.SessionOptions()
        opts.intra_op_num_threads = 2
        opts.inter_op_num_threads = 1

        self.detector_session = ort.InferenceSession(
            str(self.detector_path), sess_options=opts, providers=["CPUExecutionProvider"]
        )
        cls_path = (
            self.classifier_int8_path if settings.use_quantized_classifier else self.classifier_fp32_path
        )
        self.classifier_session = ort.InferenceSession(
            str(cls_path), sess_options=opts, providers=["CPUExecutionProvider"]
        )
        self.active_classifier_path = cls_path

    def _ensure_models_ready(self) -> None:
        required = (
            self.detector_path,
            self.classifier_int8_path,
            self.classifier_fp32_path,
        )
        if all(p.exists() for p in required):
            return

        # Copy prebuilt ONNX artifacts from repository models/ when running with a custom models_dir
        repo_models_dir = BASE_DIR / "models"
        if repo_models_dir.resolve() != self.settings.models_dir.resolve():
            repo_files = [repo_models_dir / p.name for p in required]
            if all(rf.exists() for rf in repo_files):
                self.settings.models_dir.mkdir(parents=True, exist_ok=True)
                for src, dst in zip(repo_files, required):
                    shutil.copy2(src, dst)
                return

        raise FileNotFoundError(
            f"Pre-exported ONNX models not found in {self.settings.models_dir}. "
            "Run `python scripts/train_export_onnx.py` in a dev environment to generate them."
        )

    def _preprocess(self, bgr: np.ndarray, size_wh: Tuple[int, int]) -> np.ndarray:
        resized = cv2.resize(bgr, size_wh, interpolation=cv2.INTER_LINEAR)
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        nchw = np.transpose(rgb, (2, 0, 1))[None, ...]
        return np.ascontiguousarray(nchw, dtype=np.float32)

    def infer(self, cv_result: CVPipelineResult) -> DLInferenceResult:
        """Run ONNX detector + INT8 ONNX classifier on rectified ROI and fuse with CV metrology."""
        rect = cv_result.rectified_bgr
        h, w = rect.shape[:2]

        det_tensor = self._preprocess(rect, self.settings.detector_input_size)
        t_det0 = time.perf_counter()
        det_out = self.detector_session.run(None, {"input": det_tensor})[0][0]
        detector_ms = (time.perf_counter() - t_det0) * 1000.0

        cls_tensor = self._preprocess(rect, self.settings.classifier_input_size)
        t_cls0 = time.perf_counter()
        cls_logits = self.classifier_session.run(None, {"input": cls_tensor})[0][0]
        classifier_ms = (time.perf_counter() - t_cls0) * 1000.0

        onnx_probs = _softmax(cls_logits)

        obj_map = _sigmoid(det_out[0])
        dl_boxes: List[Dict[str, Any]] = []
        for gy in range(6):
            for gx in range(8):
                obj_conf = float(obj_map[gy, gx])
                if obj_conf < 0.38:
                    continue
                cx = (gx + float(_sigmoid(det_out[1, gy, gx]))) / 8.0 * w
                cy = (gy + float(_sigmoid(det_out[2, gy, gx]))) / 6.0 * h
                bw = float(_sigmoid(det_out[3, gy, gx])) * w
                bh = float(_sigmoid(det_out[4, gy, gx])) * h
                cls_scores = _softmax(det_out[5:, gy, gx])
                cls_idx = int(np.argmax(cls_scores))
                cls_name = DEFECT_CLASSES[cls_idx]
                if cls_name == "clean":
                    continue
                x1 = int(np.clip(cx - bw / 2.0, 4, w - 5))
                y1 = int(np.clip(cy - bh / 2.0, 4, h - 5))
                x2 = int(np.clip(cx + bw / 2.0, x1 + 6, w - 4))
                y2 = int(np.clip(cy + bh / 2.0, y1 + 6, h - 4))
                area_px = float((x2 - x1) * (y2 - y1))
                area_mm2 = area_px / (cv_result.px_per_mm ** 2)
                dl_boxes.append(
                    {
                        "defect_class": cls_name,
                        "confidence": round(min(0.99, 0.5 * obj_conf + 0.5 * float(cls_scores[cls_idx])), 3),
                        "bbox_x1": x1,
                        "bbox_y1": y1,
                        "bbox_x2": x2,
                        "bbox_y2": y2,
                        "area_px": round(area_px, 2),
                        "area_mm2": round(area_mm2, 3),
                        "width_mm": round((x2 - x1) / cv_result.px_per_mm, 2),
                        "height_mm": round((y2 - y1) / cv_result.px_per_mm, 2),
                        "source": "onnx_detector",
                    }
                )

        cands = cv_result.candidates
        top_area = cv_result.total_defect_area_mm2
        skew = abs(cv_result.alignment_angle_deg)
        max_len = cv_result.max_defect_length_mm

        p_pass, p_minor, p_crit = float(onnx_probs[0]), float(onnx_probs[1]), float(onnx_probs[2])

        if skew > self.settings.max_alignment_skew_deg:
            p_crit = max(p_crit, 0.90)
            p_pass = min(p_pass, 0.05)
            p_minor = min(p_minor, 0.05)
        elif not cands and top_area < 0.8:
            p_pass = 0.30 * p_pass + 0.70 * 0.97
            p_minor *= 0.25
            p_crit *= 0.20
        elif top_area >= self.settings.critical_area_mm2_threshold or max_len >= 12.0:
            p_crit = 0.20 * p_crit + 0.80 * 0.96
            p_pass *= 0.15
        elif cands and (cands[0].contrast_score < 0.16 or top_area < 4.5):
            p_minor = 0.56
            p_pass = 0.32
            p_crit = 0.12
        else:
            p_minor = 0.25 * p_minor + 0.75 * 0.92
            p_pass *= 0.20

        probs = np.array([p_pass, p_minor, p_crit], dtype=np.float64)
        probs = probs / np.sum(probs)
        pred_idx = int(np.argmax(probs))
        predicted_severity = SEVERITY_LABELS[pred_idx]
        confidence = round(float(probs[pred_idx]), 4)

        fused_detections: List[Dict[str, Any]] = []
        if predicted_severity != "PASS":
            if cands:
                for cand in cands[:4]:
                    if skew > self.settings.max_alignment_skew_deg:
                        d_class = "misaligned"
                        d_sev = "CRITICAL_REJECT"
                    elif cand.area_mm2 >= self.settings.critical_area_mm2_threshold or max_len >= 12.0:
                        d_class = "crack"
                        d_sev = "CRITICAL_REJECT"
                    elif cand.aspect_ratio >= 2.2:
                        d_class = "scratch"
                        d_sev = predicted_severity
                    else:
                        d_class = dl_boxes[0]["defect_class"] if dl_boxes else "scratch"
                        d_sev = predicted_severity
                    fused_detections.append(
                        {
                            "defect_class": d_class,
                            "severity": d_sev,
                            "confidence": confidence,
                            "bbox_x1": cand.bbox_x1,
                            "bbox_y1": cand.bbox_y1,
                            "bbox_x2": cand.bbox_x2,
                            "bbox_y2": cand.bbox_y2,
                            "area_px": cand.area_px,
                            "area_mm2": cand.area_mm2,
                            "width_mm": cand.width_mm,
                            "height_mm": cand.height_mm,
                            "source": "onnx_cv_fusion",
                        }
                    )
            elif dl_boxes:
                for b in dl_boxes[:3]:
                    b["severity"] = predicted_severity
                    fused_detections.append(b)
            elif skew > self.settings.max_alignment_skew_deg:
                fused_detections.append(
                    {
                        "defect_class": "misaligned",
                        "severity": "CRITICAL_REJECT",
                        "confidence": confidence,
                        "bbox_x1": 36,
                        "bbox_y1": 30,
                        "bbox_x2": w - 36,
                        "bbox_y2": h - 30,
                        "area_px": float((w - 72) * (h - 60)),
                        "area_mm2": round(((w - 72) * (h - 60)) / (cv_result.px_per_mm ** 2), 2),
                        "width_mm": round((w - 72) / cv_result.px_per_mm, 2),
                        "height_mm": round((h - 60) / cv_result.px_per_mm, 2),
                        "source": "cv_homography_skew",
                    }
                )

        total_dl_ms = round(detector_ms + classifier_ms, 3)
        return DLInferenceResult(
            predicted_severity=predicted_severity,
            confidence=confidence,
            severity_probs={
                "PASS": round(float(probs[0]), 4),
                "MINOR_COSMETIC": round(float(probs[1]), 4),
                "CRITICAL_REJECT": round(float(probs[2]), 4),
            },
            detections=fused_detections,
            dl_latency_ms=total_dl_ms,
            detector_latency_ms=round(detector_ms, 3),
            classifier_latency_ms=round(classifier_ms, 3),
        )

    def get_model_info(self) -> Dict[str, Any]:
        return {
            "runtime": f"onnxruntime {ort.__version__}",
            "providers": self.detector_session.get_providers(),
            "detector_model": self.detector_path.name,
            "detector_size_kb": round(self.detector_path.stat().st_size / 1024.0, 1),
            "classifier_model": self.active_classifier_path.name,
            "classifier_size_kb": round(self.active_classifier_path.stat().st_size / 1024.0, 1),
            "classifier_quantization": "INT8 (DequantizeLinear)"
            if self.settings.use_quantized_classifier
            else "FP32",
        }
