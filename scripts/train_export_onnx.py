"""Offline PyTorch model training and ONNX export script.

Trains the lightweight YOLOv8-lite defect localizer and MobileNetV3 severity classifier
on synthetic inspection frames, then serializes FP32 and INT8-quantized `.onnx` models
into `models/` for zero-torch runtime execution via `onnxruntime`.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import List, Optional, Sequence, Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.config import Settings, get_settings
from app.cv_pipeline import CVMetrologyPipeline
from app.dl_engine import DEFECT_CLASSES, SEVERITY_LABELS
from app.onnx_builder import OnnxGraphBuilder
from scripts.generate_sample_batch import SURFACE_TYPES, generate_synthetic_frame


class YOLOv8LiteDetector(nn.Module):
    """Lightweight convolutional spatial-grid detector (6x8 grid, 9 channels per cell)."""

    def __init__(self, num_classes: int = 4) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(16, 24, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(24),
            nn.ReLU(),
            nn.Conv2d(24, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 48, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(48),
            nn.ReLU(),
            nn.Conv2d(48, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 5 + num_classes, kernel_size=1, stride=1, padding=0),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class MobileNetV3SeverityNet(nn.Module):
    """Depthwise-separable MobileNetV3-style classifier for 3-way defect severity grading."""

    def __init__(self, num_classes: int = 3) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1, groups=16),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1, groups=32),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 48, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(48),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(48, 48, kernel_size=3, stride=1, padding=1, groups=48),
            nn.BatchNorm2d(48),
            nn.ReLU(),
            nn.Conv2d(48, 64, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(64, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def _fold_conv_bn(conv: nn.Conv2d, bn: Optional[nn.BatchNorm2d]) -> Tuple[np.ndarray, np.ndarray]:
    w = conv.weight.detach().cpu().numpy().astype(np.float32)
    if conv.bias is not None:
        b = conv.bias.detach().cpu().numpy().astype(np.float32)
    else:
        b = np.zeros(conv.out_channels, dtype=np.float32)

    if bn is None:
        return w, b

    gamma = bn.weight.detach().cpu().numpy().astype(np.float32)
    beta = bn.bias.detach().cpu().numpy().astype(np.float32)
    mean = bn.running_mean.detach().cpu().numpy().astype(np.float32)
    var = bn.running_var.detach().cpu().numpy().astype(np.float32)
    eps = float(bn.eps)

    inv_std = gamma / np.sqrt(var + eps)
    w_folded = w * inv_std.reshape(-1, 1, 1, 1)
    b_folded = (b - mean) * inv_std + beta
    return w_folded, b_folded


def export_sequential_cnn_to_onnx(
    module: nn.Module,
    input_shape: Tuple[int, int, int, int],
    output_shape: Sequence[int | str],
    output_path: Path,
    graph_name: str = "visionguard_net",
    quantize_int8: bool = False,
) -> Path:
    """Export a PyTorch module into a standard ONNX file with optional INT8 weight quantization."""
    module.eval()
    builder = OnnxGraphBuilder.create(graph_name)
    builder.add_input("input", list(input_shape), elem_type=1)

    children: List[nn.Module] = []

    def _collect(m: nn.Module) -> None:
        if len(list(m.children())) == 0:
            children.append(m)
        else:
            for c in m.children():
                _collect(c)

    _collect(module)
    cur = "input"
    idx = 0
    while idx < len(children):
        layer = children[idx]
        if isinstance(layer, nn.Conv2d):
            bn = None
            if idx + 1 < len(children) and isinstance(children[idx + 1], nn.BatchNorm2d):
                bn = children[idx + 1]
                idx += 1
            w, b = _fold_conv_bn(layer, bn)
            stride = layer.stride if isinstance(layer.stride, tuple) else (layer.stride, layer.stride)
            pad = layer.padding if isinstance(layer.padding, tuple) else (layer.padding, layer.padding)
            cur = builder.add_conv(
                cur,
                w,
                b,
                stride=stride,
                padding=pad,
                groups=int(layer.groups),
                quantize_int8=quantize_int8,
            )
        elif isinstance(layer, nn.ReLU):
            cur = builder.add_relu(cur)
        elif isinstance(layer, nn.Sigmoid):
            cur = builder.add_sigmoid(cur)
        elif isinstance(layer, nn.MaxPool2d):
            k = layer.kernel_size if isinstance(layer.kernel_size, int) else layer.kernel_size[0]
            s = layer.stride if isinstance(layer.stride, int) else layer.stride[0]
            p = layer.padding if isinstance(layer.padding, int) else layer.padding[0]
            cur = builder.add_maxpool(cur, kernel_size=k, stride=s, padding=p)
        elif isinstance(layer, nn.AdaptiveAvgPool2d):
            cur = builder.add_global_avg_pool(cur)
        elif isinstance(layer, nn.Flatten):
            cur = builder.add_flatten(cur, axis=int(layer.start_dim))
        elif isinstance(layer, nn.Linear):
            w = layer.weight.detach().cpu().numpy().astype(np.float32)
            b = (
                layer.bias.detach().cpu().numpy().astype(np.float32)
                if layer.bias is not None
                else np.zeros(layer.out_features, dtype=np.float32)
            )
            cur = builder.add_linear(cur, w, b, quantize_int8=quantize_int8)
        elif isinstance(layer, (nn.Dropout, nn.Identity)):
            pass
        else:
            raise ValueError(f"Unsupported layer type for ONNX export: {type(layer).__name__}")
        idx += 1

    builder.add_identity(cur, "output")
    builder.add_output("output", list(output_shape), elem_type=1)
    model_bytes = builder.build_model_bytes()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(model_bytes)
    return output_path


def train_and_export_models(settings: Optional[Settings] = None) -> List[Path]:
    """Train lightweight detector and severity classifier on synthetic frames and export `.onnx`."""
    cfg = settings or get_settings()
    cfg.ensure_dirs()

    torch.manual_seed(2026)
    np.random.seed(2026)

    cv_pipe = CVMetrologyPipeline(cfg)

    det_images: List[np.ndarray] = []
    det_targets: List[np.ndarray] = []
    cls_images: List[np.ndarray] = []
    cls_labels: List[int] = []

    train_scenarios = ["clean", "clean", "scratch", "crack", "misaligned"]
    for i in range(90):
        surf = SURFACE_TYPES[i % len(SURFACE_TYPES)]
        kind = train_scenarios[i % len(train_scenarios)]
        frame, meta = generate_synthetic_frame(surface_type=surf, defect_kind=kind, seed=9000 + i)
        norm, _ = cv_pipe.apply_clahe(frame)
        rect, _, _ = cv_pipe.align_and_rectify_roi(norm)

        det_in = cv2.resize(rect, cfg.detector_input_size, interpolation=cv2.INTER_LINEAR)
        det_rgb = cv2.cvtColor(det_in, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        det_images.append(np.transpose(det_rgb, (2, 0, 1)))

        grid = np.zeros((9, 6, 8), dtype=np.float32)
        grid[5, :, :] = 1.0
        cands = cv_pipe.extract_metrology_candidates(rect, cfg.default_px_per_mm)
        cls_idx = DEFECT_CLASSES.index(meta["defect_class"])
        if kind != "clean" and cands:
            top = cands[0]
            cx = ((top.bbox_x1 + top.bbox_x2) * 0.5) / cfg.canonical_width_px
            cy = ((top.bbox_y1 + top.bbox_y2) * 0.5) / cfg.canonical_height_px
            gx = int(np.clip(cx * 8, 0, 7))
            gy = int(np.clip(cy * 6, 0, 5))
            grid[0, gy, gx] = 1.0
            grid[1, gy, gx] = cx * 8 - gx
            grid[2, gy, gx] = cy * 6 - gy
            grid[3, gy, gx] = np.clip((top.bbox_x2 - top.bbox_x1) / cfg.canonical_width_px, 0.05, 0.9)
            grid[4, gy, gx] = np.clip((top.bbox_y2 - top.bbox_y1) / cfg.canonical_height_px, 0.05, 0.9)
            grid[5:, gy, gx] = 0.0
            grid[5 + cls_idx, gy, gx] = 1.0
        det_targets.append(grid)

        cls_in = cv2.resize(rect, cfg.classifier_input_size, interpolation=cv2.INTER_LINEAR)
        cls_rgb = cv2.cvtColor(cls_in, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        cls_images.append(np.transpose(cls_rgb, (2, 0, 1)))
        cls_labels.append(SEVERITY_LABELS.index(meta["expected_severity"]))

    x_det = torch.from_numpy(np.stack(det_images))
    y_det = torch.from_numpy(np.stack(det_targets))
    x_cls = torch.from_numpy(np.stack(cls_images))
    y_cls = torch.tensor(cls_labels, dtype=torch.long)

    detector = YOLOv8LiteDetector(num_classes=4)
    det_opt = torch.optim.Adam(detector.parameters(), lr=4e-3)
    detector.train()
    for _ in range(18):
        det_opt.zero_grad()
        pred = detector(x_det)
        obj_mask = y_det[:, 0:1, :, :]
        obj_loss = F.binary_cross_entropy_with_logits(pred[:, 0:1], obj_mask)
        box_loss = F.mse_loss(torch.sigmoid(pred[:, 1:5]) * obj_mask, y_det[:, 1:5] * obj_mask)
        cls_loss = F.mse_loss(torch.sigmoid(pred[:, 5:]) * obj_mask, y_det[:, 5:] * obj_mask)
        loss = obj_loss + 2.0 * box_loss + cls_loss
        loss.backward()
        det_opt.step()

    classifier = MobileNetV3SeverityNet(num_classes=3)
    cls_opt = torch.optim.Adam(classifier.parameters(), lr=7e-3)
    classifier.train()
    for _ in range(50):
        cls_opt.zero_grad()
        logits = classifier(x_cls)
        loss = F.cross_entropy(logits, y_cls)
        loss.backward()
        cls_opt.step()

    det_path = cfg.models_dir / "defect_detector_yolov8_lite.onnx"
    fp32_path = cfg.models_dir / "severity_mobilenetv3_fp32.onnx"
    int8_path = cfg.models_dir / "severity_mobilenetv3_int8.onnx"

    export_sequential_cnn_to_onnx(
        detector,
        input_shape=(1, 3, 192, 256),
        output_shape=(1, 9, 6, 8),
        output_path=det_path,
        graph_name="yolov8_lite_defect_detector",
        quantize_int8=False,
    )
    export_sequential_cnn_to_onnx(
        classifier,
        input_shape=(1, 3, 128, 128),
        output_shape=(1, 3),
        output_path=fp32_path,
        graph_name="mobilenetv3_severity_fp32",
        quantize_int8=False,
    )
    export_sequential_cnn_to_onnx(
        classifier,
        input_shape=(1, 3, 128, 128),
        output_shape=(1, 3),
        output_path=int8_path,
        graph_name="mobilenetv3_severity_int8",
        quantize_int8=True,
    )
    return [det_path, fp32_path, int8_path]


def main() -> None:
    parser = argparse.ArgumentParser(description="Train and export VisionGuard AI ONNX models")
    parser.add_argument("--models-dir", type=Path, default=None)
    args = parser.parse_args()
    settings = get_settings()
    if args.models_dir is not None:
        settings = Settings(models_dir=args.models_dir)
    paths = train_and_export_models(settings)
    for p in paths:
        print(f"Exported {p} ({p.stat().st_size / 1024.0:.1f} KB)")


if __name__ == "__main__":
    main()
