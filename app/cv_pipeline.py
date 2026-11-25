"""OpenCV Computer Vision & Metrology Pipeline.

Performs:
1. CLAHE illumination normalization in LAB color space
2. Contour-based ROI detection & homography/perspective rectification
3. Pixel-to-mm physical scale calibration & defect area/dimension metrology
4. High-contrast engineering overlay rendering for the dark-slate console
"""

from __future__ import annotations

from dataclasses import dataclass, field
import time
from typing import Dict, List, Tuple

import cv2
import numpy as np

from app.config import Settings


@dataclass
class CVDefectCandidate:
    bbox_x1: int
    bbox_y1: int
    bbox_x2: int
    bbox_y2: int
    area_px: float
    area_mm2: float
    width_mm: float
    height_mm: float
    aspect_ratio: float
    contrast_score: float


@dataclass
class CVPipelineResult:
    normalized_bgr: np.ndarray
    rectified_bgr: np.ndarray
    px_per_mm: float
    alignment_angle_deg: float
    contrast_gain: float
    candidates: List[CVDefectCandidate] = field(default_factory=list)
    total_defect_area_mm2: float = 0.0
    max_defect_length_mm: float = 0.0
    cv_latency_ms: float = 0.0


def _order_quad_points(pts: np.ndarray) -> np.ndarray:
    rect = np.zeros((4, 2), dtype=np.float32)
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]  # top-left
    rect[2] = pts[np.argmax(s)]  # bottom-right
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]  # top-right
    rect[3] = pts[np.argmax(diff)]  # bottom-left
    return rect


class CVMetrologyPipeline:
    """Deterministic OpenCV preprocessing, ROI rectification, and pixel-to-mm metrology."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.clahe = cv2.createCLAHE(
            clipLimit=settings.clahe_clip_limit,
            tileGridSize=settings.clahe_tile_grid,
        )
        self._golden_refs: Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
        self._ensure_golden_refs()

    @staticmethod
    def _highpass_gray(gray: np.ndarray) -> np.ndarray:
        sm = cv2.GaussianBlur(gray, (3, 3), 0).astype(np.float32)
        low = cv2.GaussianBlur(gray, (35, 35), 0).astype(np.float32)
        return sm - low

    def _ensure_golden_refs(self) -> Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]]:
        if self._golden_refs:
            return self._golden_refs
        from scripts.generate_sample_batch import SURFACE_TYPES, generate_synthetic_frame

        dil_kernel = np.ones((7, 7), np.uint8)
        for surf in SURFACE_TYPES:
            # Average multiple clean samples to obtain a grain-free canonical golden reference
            acc_gray = None
            for s_idx in range(6):
                frame, _ = generate_synthetic_frame(surface_type=surf, defect_kind="clean", seed=s_idx)
                norm, _ = self.apply_clahe(frame)
                rect, _, _ = self.align_and_rectify_roi(norm)
                g_f = cv2.cvtColor(rect, cv2.COLOR_BGR2GRAY).astype(np.float32)
                acc_gray = g_f if acc_gray is None else (acc_gray + g_f)
            gray = np.clip(acc_gray / 6.0, 0, 255).astype(np.uint8)
            sm_gray = cv2.GaussianBlur(gray, (5, 5), 0)
            edges = cv2.Canny(sm_gray, 55, 140)
            edge_mask = cv2.dilate(edges, dil_kernel)
            self._golden_refs[surf] = (gray, self._highpass_gray(gray), edge_mask)
        return self._golden_refs

    def apply_clahe(self, bgr_image: np.ndarray) -> Tuple[np.ndarray, float]:
        """Normalize uneven lighting using CLAHE on LAB L-channel."""
        lab = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2LAB)
        l_chan, a_chan, b_chan = cv2.split(lab)
        std_before = float(np.std(l_chan)) + 1e-6
        l_norm = self.clahe.apply(l_chan)
        std_after = float(np.std(l_norm))
        merged = cv2.merge((l_norm, a_chan, b_chan))
        normalized = cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)
        return normalized, round(std_after / std_before, 3)

    def align_and_rectify_roi(
        self, normalized_bgr: np.ndarray
    ) -> Tuple[np.ndarray, float, float]:
        """Locate workpiece contour, measure skew angle, rectify via perspective homography,
        and compute physical pixel-to-mm calibration."""
        h, w = normalized_bgr.shape[:2]
        gray = cv2.cvtColor(normalized_bgr, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)

        # Segment workpiece against dark conveyor background
        _, mask = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        target_w = self.settings.canonical_width_px
        target_h = self.settings.canonical_height_px

        if not contours:
            resized = cv2.resize(normalized_bgr, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
            return resized, 0.0, self.settings.default_px_per_mm

        largest = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(largest)
        if area < 0.18 * h * w:
            resized = cv2.resize(normalized_bgr, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
            return resized, 0.0, self.settings.default_px_per_mm

        rect = cv2.minAreaRect(largest)
        (_, _), (rw, rh), raw_angle = rect
        # Normalize angle relative to landscape part orientation
        if rw < rh:
            part_w_px = rh
            angle_deg = raw_angle + 90.0 if raw_angle < -45.0 else raw_angle - 90.0
        else:
            part_w_px = rw
            angle_deg = raw_angle

        if angle_deg > 45.0:
            angle_deg -= 90.0
        elif angle_deg < -45.0:
            angle_deg += 90.0

        box = cv2.boxPoints(rect)
        src_pts = _order_quad_points(box.astype(np.float32))

        margin_x = 24.0
        margin_y = 22.0
        dst_pts = np.array(
            [
                [margin_x, margin_y],
                [target_w - margin_x, margin_y],
                [target_w - margin_x, target_h - margin_y],
                [margin_x, target_h - margin_y],
            ],
            dtype=np.float32,
        )
        homography = cv2.getPerspectiveTransform(src_pts, dst_pts)
        rectified = cv2.warpPerspective(
            normalized_bgr,
            homography,
            (target_w, target_h),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(16, 22, 34),
        )

        # Calibrated px/mm based on rectified ROI span vs nominal part width (100 mm)
        rectified_part_span_px = float(target_w - 2 * margin_x)
        px_per_mm = rectified_part_span_px / self.settings.nominal_part_width_mm
        if part_w_px <= 1.0:
            px_per_mm = self.settings.default_px_per_mm

        return rectified, round(float(angle_deg), 2), round(px_per_mm, 3)

    def extract_metrology_candidates(
        self, rectified_bgr: np.ndarray, px_per_mm: float
    ) -> List[CVDefectCandidate]:
        """Extract surface anomaly candidates via golden-reference highpass subtraction
        inside the homography-aligned ROI and compute physical mm metrics."""
        h, w = rectified_bgr.shape[:2]
        gray = cv2.cvtColor(rectified_bgr, cv2.COLOR_BGR2GRAY)

        refs = self._ensure_golden_refs()
        best_surf = min(
            refs.keys(),
            key=lambda k: float(np.mean(np.abs(gray.astype(np.float32) - refs[k][0].astype(np.float32)))),
        )
        _, ref_hp, edge_mask = refs[best_surf]

        # Illumination-invariant highpass difference with static-feature edge suppression
        diff = np.abs(self._highpass_gray(gray) - ref_hp)
        diff[edge_mask > 0] *= 0.18
        diff_u8 = np.clip(diff, 0, 255).astype(np.uint8)

        # Mask out outer bezel/mounting holes and bottom serial stamp zone
        roi_mask = np.zeros((h, w), dtype=np.uint8)
        cv2.rectangle(roi_mask, (74, 68), (w - 74, h - 78), 255, -1)
        diff_u8 = cv2.bitwise_and(diff_u8, diff_u8, mask=roi_mask)

        # Suppress 1px brushed-metal horizontal streak noise while preserving true defects
        _, thresh = cv2.threshold(diff_u8, 25, 255, cv2.THRESH_BINARY)
        thresh = cv2.morphologyEx(
            thresh,
            cv2.MORPH_OPEN,
            cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2)),
        )
        thresh = cv2.morphologyEx(
            thresh,
            cv2.MORPH_CLOSE,
            cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9)),
        )

        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        candidates: List[CVDefectCandidate] = []
        for cnt in contours:
            area_px = float(cv2.contourArea(cnt))
            if area_px < 22.0:
                continue
            x, y, bw, bh = cv2.boundingRect(cnt)
            if bw > w * 0.72 or bh > h * 0.72:
                continue
            area_mm2 = area_px / (px_per_mm ** 2)
            width_mm = bw / px_per_mm
            height_mm = bh / px_per_mm
            aspect = max(bw, bh) / max(1.0, float(min(bw, bh)))
            patch_energy = float(np.mean(diff_u8[y : y + bh, x : x + bw]))
            candidates.append(
                CVDefectCandidate(
                    bbox_x1=int(x),
                    bbox_y1=int(y),
                    bbox_x2=int(x + bw),
                    bbox_y2=int(y + bh),
                    area_px=round(area_px, 2),
                    area_mm2=round(area_mm2, 3),
                    width_mm=round(width_mm, 2),
                    height_mm=round(height_mm, 2),
                    aspect_ratio=round(aspect, 2),
                    contrast_score=round(patch_energy / 255.0, 3),
                )
            )

        candidates.sort(key=lambda c: c.area_mm2, reverse=True)
        return candidates[:8]

    def process(self, bgr_image: np.ndarray) -> CVPipelineResult:
        t0 = time.perf_counter()
        normalized, contrast_gain = self.apply_clahe(bgr_image)
        rectified, angle_deg, px_per_mm = self.align_and_rectify_roi(normalized)
        candidates = self.extract_metrology_candidates(rectified, px_per_mm)
        total_area = round(sum(c.area_mm2 for c in candidates), 3)
        max_len = round(
            max((max(c.width_mm, c.height_mm) for c in candidates), default=0.0),
            2,
        )
        latency_ms = round((time.perf_counter() - t0) * 1000.0, 3)
        return CVPipelineResult(
            normalized_bgr=normalized,
            rectified_bgr=rectified,
            px_per_mm=px_per_mm,
            alignment_angle_deg=angle_deg,
            contrast_gain=contrast_gain,
            candidates=candidates,
            total_defect_area_mm2=total_area,
            max_defect_length_mm=max_len,
            cv_latency_ms=latency_ms,
        )

    def render_annotated_overlay(
        self,
        rectified_bgr: np.ndarray,
        detections: List[Dict],
        item_id: str,
        state: str,
        severity: str,
        confidence: float,
        px_per_mm: float,
        alignment_angle_deg: float,
        total_latency_ms: float,
    ) -> np.ndarray:
        """Render high-contrast engineering bounding boxes, pixel-to-mm scale bar, and telemetry HUD."""
        canvas = rectified_bgr.copy()
        h, w = canvas.shape[:2]

        color_map = {
            "PASS": (129, 185, 16),          # Emerald #10B981 in BGR
            "MINOR_COSMETIC": (11, 158, 245), # Amber #F59E0B in BGR
            "CRITICAL_REJECT": (68, 68, 239), # Crimson #EF4444 in BGR
        }
        accent = color_map.get(severity, (246, 130, 59))

        # Subtle ROI alignment reticle
        cv2.rectangle(canvas, (24, 22), (w - 24, h - 22), (75, 92, 118), 1)
        for rx, ry in ((24, 22), (w - 24, 22), (24, h - 22), (w - 24, h - 22)):
            cv2.circle(canvas, (rx, ry), 3, (232, 180, 56), -1)

        # Draw detections with corner brackets and physical mm labels
        for det in detections:
            x1, y1, x2, y2 = det["bbox_x1"], det["bbox_y1"], det["bbox_x2"], det["bbox_y2"]
            det_sev = det.get("severity", severity)
            box_col = color_map.get(det_sev, accent)
            cv2.rectangle(canvas, (x1, y1), (x2, y2), box_col, 2)

            # Corner ticks
            t = 8
            for (cx, cy), (sx, sy) in (
                ((x1, y1), (1, 1)),
                ((x2, y1), (-1, 1)),
                ((x1, y2), (1, -1)),
                ((x2, y2), (-1, -1)),
            ):
                cv2.line(canvas, (cx, cy), (cx + sx * t, cy), (255, 255, 255), 2)
                cv2.line(canvas, (cx, cy), (cx, cy + sy * t), (255, 255, 255), 2)

            label = (
                f"{det['defect_class'].upper()} {det['confidence']*100:.0f}% | "
                f"{det['area_mm2']:.1f}mm2 ({det['width_mm']:.1f}x{det['height_mm']:.1f}mm)"
            )
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.38, 1)
            ly = max(th + 8, y1 - 4)
            cv2.rectangle(canvas, (x1, ly - th - 6), (min(w - 4, x1 + tw + 8), ly + 2), (11, 17, 32), -1)
            cv2.rectangle(canvas, (x1, ly - th - 6), (min(w - 4, x1 + tw + 8), ly + 2), box_col, 1)
            cv2.putText(canvas, label, (x1 + 4, ly - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (241, 245, 249), 1)

        # Top HUD Telemetry Bar (#0B1120)
        hud = canvas.copy()
        cv2.rectangle(hud, (0, 0), (w, 26), (32, 17, 11), -1)
        cv2.addWeighted(hud, 0.88, canvas, 0.12, 0.0, canvas)
        cv2.line(canvas, (0, 26), (w, 26), accent, 2)
        hud_text = (
            f"{item_id} | {state} | {severity} ({confidence*100:.1f}%) | "
            f"SKEW {alignment_angle_deg:+.1f}deg | {total_latency_ms:.1f}ms"
        )
        cv2.putText(canvas, hud_text, (8, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.34, (241, 245, 249), 1)

        # Bottom-right 10 mm calibration scale bar (auto-sized so text never clips)
        bar_px = int(round(10.0 * px_per_mm))
        scale_label = f"10.0 mm ({px_per_mm:.2f} px/mm)"
        (stw, sth), _ = cv2.getTextSize(scale_label, cv2.FONT_HERSHEY_SIMPLEX, 0.34, 1)
        box_w = max(bar_px, stw) + 16
        box_x1 = w - box_w - 8
        box_x2 = w - 8
        by = h - 10
        cv2.rectangle(canvas, (box_x1, by - sth - 12), (box_x2, by + 5), (32, 17, 11), -1)
        cv2.rectangle(canvas, (box_x1, by - sth - 12), (box_x2, by + 5), (56, 189, 248), 1)
        bx2 = box_x2 - 8
        bx1 = bx2 - bar_px
        cv2.line(canvas, (bx1, by), (bx2, by), (56, 189, 248), 2)
        cv2.line(canvas, (bx1, by - 3), (bx1, by + 3), (56, 189, 248), 2)
        cv2.line(canvas, (bx2, by - 3), (bx2, by + 3), (56, 189, 248), 2)
        cv2.putText(
            canvas,
            scale_label,
            (box_x1 + 8, by - 6),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.34,
            (226, 232, 240),
            1,
        )
        return canvas
