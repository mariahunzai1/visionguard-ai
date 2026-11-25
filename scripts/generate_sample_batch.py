"""Realistic synthetic industrial inspection dataset generator.

Generates metallic plate, PCB assembly, and packaging seal camera frames with realistic
illumination gradients, fiducial alignment markers, part rotation jitter, and defects
(`clean`, `scratch`, `crack`, `misaligned`, plus low-confidence `borderline` edge cases).
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


SURFACE_TYPES = ("metallic", "pcb", "packaging")
DEFECT_SCENARIOS = ("clean", "scratch", "crack", "misaligned", "borderline")


def _draw_metallic_part(canvas: np.ndarray, rng: np.random.Generator) -> None:
    h, w = canvas.shape[:2]
    # Brushed steel / anodized aluminum base
    base = np.full((h, w, 3), (156, 162, 170), dtype=np.float32)
    # Horizontal brushed grain
    streak = rng.normal(0.0, 5.5, size=(h, 1, 1)).astype(np.float32)
    fine = rng.normal(0.0, 2.2, size=(h, w, 1)).astype(np.float32)
    base = np.clip(base + streak + fine, 30, 235).astype(np.uint8)
    canvas[:] = base

    # Machined inner pocket bevel
    cv2.rectangle(canvas, (34, 28), (w - 34, h - 28), (118, 124, 132), 2)
    cv2.rectangle(canvas, (40, 34), (w - 40, h - 34), (178, 184, 192), 1)
    # Mounting holes / fiducials
    for cx, cy in ((26, 26), (w - 26, 26), (26, h - 26), (w - 26, h - 26)):
        cv2.circle(canvas, (cx, cy), 9, (68, 74, 82), -1)
        cv2.circle(canvas, (cx, cy), 9, (210, 214, 220), 1)
    # Laser-etched serial block
    cv2.rectangle(canvas, (62, h - 66), (176, h - 46), (132, 138, 146), -1)
    cv2.putText(canvas, "VG-AL6061", (68, h - 51), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (220, 226, 232), 1)


def _draw_pcb_part(canvas: np.ndarray, rng: np.random.Generator) -> None:
    h, w = canvas.shape[:2]
    # Matte solder mask green PCB
    noise = rng.normal(0.0, 2.5, size=(h, w, 1)).astype(np.float32)
    base = np.full((h, w, 3), (38, 86, 32), dtype=np.float32)
    canvas[:] = np.clip(base + noise, 15, 220).astype(np.uint8)

    # Copper traces under solder mask
    for y in range(52, h - 52, 22):
        cv2.line(canvas, (45, y), (w - 45, y), (50, 114, 42), 2)
    for x in range(70, w - 70, 38):
        cv2.line(canvas, (x, 45), (x, h - 45), (48, 108, 40), 1)

    # Central QFP IC package + pins
    cx, cy = w // 2, h // 2
    cv2.rectangle(canvas, (cx - 46, cy - 46), (cx + 46, cy + 46), (34, 38, 42), -1)
    cv2.rectangle(canvas, (cx - 46, cy - 46), (cx + 46, cy + 46), (185, 195, 200), 1)
    cv2.circle(canvas, (cx - 34, cy - 34), 4, (200, 210, 215), -1)
    for offset in range(-36, 40, 10):
        cv2.rectangle(canvas, (cx + offset - 2, cy - 54), (cx + offset + 2, cy - 47), (195, 175, 95), -1)
        cv2.rectangle(canvas, (cx + offset - 2, cy + 47), (cx + offset + 2, cy + 54), (195, 175, 95), -1)
        cv2.rectangle(canvas, (cx - 54, cy + offset - 2), (cx - 47, cy + offset + 2), (195, 175, 95), -1)
        cv2.rectangle(canvas, (cx + 47, cy + offset - 2), (cx + 54, cy + offset + 2), (195, 175, 95), -1)

    # Gold fiducials & SMD pads
    for fx, fy in ((24, 24), (w - 24, 24), (24, h - 24), (w - 24, h - 24)):
        cv2.circle(canvas, (fx, fy), 6, (90, 185, 215), -1)
    for px, py in ((95, 85), (125, 85), (w - 125, 85), (w - 95, 85), (95, h - 85), (w - 95, h - 85)):
        cv2.rectangle(canvas, (px - 10, py - 6), (px + 10, py + 6), (195, 180, 110), -1)


def _draw_packaging_part(canvas: np.ndarray, rng: np.random.Generator) -> None:
    h, w = canvas.shape[:2]
    # Matte polymer pouch
    noise = rng.normal(0.0, 3.0, size=(h, w, 1)).astype(np.float32)
    base = np.full((h, w, 3), (186, 192, 200), dtype=np.float32)
    canvas[:] = np.clip(base + noise, 40, 240).astype(np.uint8)

    # Heat-seal crimped border
    cv2.rectangle(canvas, (18, 18), (w - 18, h - 18), (154, 160, 168), 3)
    for x in range(22, w - 22, 8):
        cv2.line(canvas, (x, 8), (x, 18), (146, 152, 160), 1)
        cv2.line(canvas, (x, h - 18), (x, h - 8), (146, 152, 160), 1)

    # Label block with barcode lines
    lx1, ly1, lx2, ly2 = w // 2 - 90, h // 2 - 48, w // 2 + 90, h // 2 + 48
    cv2.rectangle(canvas, (lx1, ly1), (lx2, ly2), (232, 236, 242), -1)
    cv2.rectangle(canvas, (lx1, ly1), (lx2, ly2), (90, 98, 112), 2)
    cv2.putText(canvas, "LOT-2026-VG", (lx1 + 14, ly1 + 26), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (40, 48, 62), 2)
    bx = lx1 + 16
    for i in range(22):
        bw = 2 if i % 3 == 0 else 4
        cv2.rectangle(canvas, (bx, ly1 + 38), (bx + bw, ly2 - 14), (30, 35, 44), -1)
        bx += bw + 3


def _inject_defect(
    part_img: np.ndarray,
    defect_kind: str,
    rng: np.random.Generator,
) -> Tuple[Optional[Tuple[int, int, int, int]], float, str]:
    """Injects realistic defect into part coordinates and returns (bbox, severity_score, actual_class)."""
    h, w = part_img.shape[:2]

    if defect_kind == "clean":
        return None, 0.0, "clean"

    if defect_kind == "scratch":
        # Noticeable surface abrasion scratch (minor cosmetic if moderate, critical if long/deep)
        x1 = int(rng.integers(75, w - 180))
        y1 = int(rng.integers(60, h - 90))
        length = int(rng.integers(45, 95))
        angle = float(rng.uniform(-0.45, 0.45))
        x2 = int(np.clip(x1 + length * math.cos(angle), 20, w - 20))
        y2 = int(np.clip(y1 + length * math.sin(angle), 20, h - 20))
        # High-contrast specular scratch groove
        cv2.line(part_img, (x1, y1), (x2, y2), (38, 42, 48), 3)
        cv2.line(part_img, (x1 + 1, y1 + 1), (x2 + 1, y2 + 1), (228, 232, 238), 2)
        bx1, by1 = max(0, min(x1, x2) - 8), max(0, min(y1, y2) - 8)
        bx2, by2 = min(w, max(x1, x2) + 8), min(h, max(y1, y2) + 8)
        return (bx1, by1, bx2, by2), 0.5, "scratch"

    if defect_kind == "crack":
        # Deep jagged structural crack (CRITICAL_REJECT)
        x = int(rng.integers(95, w - 170))
        y = int(rng.integers(75, h - 120))
        pts = [(x, y)]
        for _ in range(7):
            x = int(np.clip(x + rng.integers(14, 28), 35, w - 35))
            y = int(np.clip(y + rng.integers(-18, 22), 35, h - 35))
            pts.append((x, y))
        for i in range(len(pts) - 1):
            cv2.line(part_img, pts[i], pts[i + 1], (14, 16, 20), 5)
            cv2.line(part_img, pts[i], pts[i + 1], (42, 34, 34), 2)
        # Branching fracture
        mid = pts[len(pts) // 2]
        branch_end = (int(np.clip(mid[0] + 22, 35, w - 35)), int(np.clip(mid[1] + 34, 35, h - 35)))
        cv2.line(part_img, mid, branch_end, (16, 18, 22), 4)
        xs = [p[0] for p in pts] + [branch_end[0]]
        ys = [p[1] for p in pts] + [branch_end[1]]
        bbox = (max(0, min(xs) - 10), max(0, min(ys) - 10), min(w, max(xs) + 10), min(h, max(ys) + 10))
        return bbox, 1.0, "crack"

    if defect_kind == "misaligned":
        # Prominent shifted/skewed assembly cutout or misaligned fiducial block
        cx = int(rng.integers(120, w - 120))
        cy = int(rng.integers(90, h - 90))
        box = np.array(
            [[-46, -24], [46, -24], [46, 24], [-46, 24]], dtype=np.float32
        )
        theta = math.radians(float(rng.choice([-1, 1]) * rng.uniform(14.0, 26.0)))
        rot = np.array([[math.cos(theta), -math.sin(theta)], [math.sin(theta), math.cos(theta)]], dtype=np.float32)
        poly = (box @ rot.T + np.array([cx, cy], dtype=np.float32)).astype(np.int32)
        cv2.fillPoly(part_img, [poly], (28, 32, 46))
        cv2.polylines(part_img, [poly], True, (220, 92, 68), 3)
        cv2.line(part_img, tuple(poly[0]), tuple(poly[2]), (220, 92, 68), 2)
        x1, y1 = int(np.min(poly[:, 0])) - 6, int(np.min(poly[:, 1])) - 6
        x2, y2 = int(np.max(poly[:, 0])) + 6, int(np.max(poly[:, 1])) + 6
        return (max(0, x1), max(0, y1), min(w, x2), min(h, y2)), 0.95, "misaligned"

    if defect_kind == "borderline":
        # Faint low-contrast micro-scratch / smudge near boundary -> triggers low confidence (<0.80) HITL QA
        x1 = int(rng.integers(90, w - 130))
        y1 = int(rng.integers(70, h - 90))
        x2 = x1 + int(rng.integers(24, 38))
        y2 = y1 + int(rng.integers(-10, 12))
        overlay = part_img.copy()
        cv2.line(overlay, (x1, y1), (x2, y2), (78, 84, 92), 2)
        cv2.circle(overlay, ((x1 + x2) // 2, (y1 + y2) // 2), 9, (118, 124, 130), -1)
        cv2.addWeighted(overlay, 0.55, part_img, 0.45, 0.0, part_img)
        return (max(0, min(x1, x2) - 8), max(0, min(y1, y2) - 10), min(w, max(x1, x2) + 8), min(h, max(y1, y2) + 10)), 0.35, "scratch"

    raise ValueError(f"Unknown defect kind: {defect_kind}")


def generate_synthetic_frame(
    surface_type: str = "metallic",
    defect_kind: str = "clean",
    seed: int = 0,
    frame_size: Tuple[int, int] = (640, 480),
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Generate a synthetic 640x480 conveyor camera frame with part ROI, lighting gradient, and optional defect."""
    rng = np.random.default_rng(seed)
    fw, fh = frame_size
    pw, ph = 460, 340

    part = np.zeros((ph, pw, 3), dtype=np.uint8)
    if surface_type == "metallic":
        _draw_metallic_part(part, rng)
    elif surface_type == "pcb":
        _draw_pcb_part(part, rng)
    elif surface_type == "packaging":
        _draw_packaging_part(part, rng)
    else:
        raise ValueError(f"Unsupported surface_type: {surface_type}")

    bbox_part, severity_score, defect_class = _inject_defect(part, defect_kind, rng)

    # Dark industrial conveyor belt background
    scene = np.full((fh, fw, 3), (22, 26, 34), dtype=np.uint8)
    belt_noise = rng.normal(0.0, 2.0, size=(fh, fw, 1)).astype(np.float32)
    scene = np.clip(scene.astype(np.float32) + belt_noise, 8, 55).astype(np.uint8)

    # Place part into scene with small realistic conveyor skew
    if defect_kind == "misaligned":
        skew_deg = float(rng.choice([-1, 1]) * rng.uniform(6.5, 10.5))
    else:
        skew_deg = float(rng.uniform(-2.8, 2.8))

    dx = float(rng.uniform(-12, 12))
    dy = float(rng.uniform(-10, 10))

    # Build affine placement from part center -> scene center
    center_part = (pw / 2.0, ph / 2.0)
    M = cv2.getRotationMatrix2D(center_part, skew_deg, 1.0)
    M[0, 2] += (fw / 2.0 - pw / 2.0) + dx
    M[1, 2] += (fh / 2.0 - ph / 2.0) + dy

    mask = np.full((ph, pw), 255, dtype=np.uint8)
    warped_part = cv2.warpAffine(part, M, (fw, fh), flags=cv2.INTER_LINEAR, borderValue=(0, 0, 0))
    warped_mask = cv2.warpAffine(mask, M, (fw, fh), flags=cv2.INTER_NEAREST, borderValue=0)

    scene[warped_mask > 0] = warped_part[warped_mask > 0]

    # Apply realistic uneven overhead illumination falloff (so CLAHE demonstrates real normalization)
    yy, xx = np.mgrid[0:fh, 0:fw]
    lx = float(rng.uniform(0.25, 0.75)) * fw
    ly = float(rng.uniform(0.25, 0.75)) * fh
    dist = np.sqrt((xx - lx) ** 2 + (yy - ly) ** 2)
    illum = np.clip(1.12 - 0.38 * (dist / max(fw, fh)), 0.68, 1.15).astype(np.float32)
    scene = np.clip(scene.astype(np.float32) * illum[:, :, None], 0, 255).astype(np.uint8)

    # Transform ground-truth bbox into scene space if present
    scene_bbox = None
    if bbox_part is not None:
        bx1, by1, bx2, by2 = bbox_part
        corners = np.array([[bx1, by1, 1.0], [bx2, by1, 1.0], [bx2, by2, 1.0], [bx1, by2, 1.0]])
        mapped = corners @ M.T
        scene_bbox = (
            int(np.clip(np.min(mapped[:, 0]), 0, fw - 1)),
            int(np.clip(np.min(mapped[:, 1]), 0, fh - 1)),
            int(np.clip(np.max(mapped[:, 0]), 0, fw - 1)),
            int(np.clip(np.max(mapped[:, 1]), 0, fh - 1)),
        )

    expected_severity = (
        "PASS"
        if defect_kind == "clean"
        else ("CRITICAL_REJECT" if defect_kind in ("crack", "misaligned") else "MINOR_COSMETIC")
    )
    return scene, {
        "surface_type": surface_type,
        "defect_kind": defect_kind,
        "defect_class": defect_class,
        "expected_severity": expected_severity,
        "severity_score": severity_score,
        "skew_deg": round(skew_deg, 2),
        "scene_bbox": scene_bbox,
    }


def generate_batch_directory(
    output_dir: Path,
    count: int = 50,
    product_line: str = "mixed",
    seed: int = 42,
) -> List[Dict[str, Any]]:
    """Generate a balanced industrial inspection batch of `count` frames with manifest."""
    output_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)

    # Distribution: ~52% clean, ~18% scratch (minor), ~14% crack (critical), ~8% misaligned (critical), ~8% borderline (QA)
    weights = [0.52, 0.18, 0.14, 0.08, 0.08]
    manifest: List[Dict[str, Any]] = []

    for idx in range(count):
        if product_line in SURFACE_TYPES:
            surface = product_line
        else:
            surface = SURFACE_TYPES[idx % len(SURFACE_TYPES)]

        defect_kind = str(rng.choice(DEFECT_SCENARIOS, p=weights))
        item_seed = seed * 1000 + idx
        frame, meta = generate_synthetic_frame(surface_type=surface, defect_kind=defect_kind, seed=item_seed)

        serial = f"VG-2026-{idx + 1:04d}"
        filename = f"{serial}_{surface}_{defect_kind}.png"
        fpath = output_dir / filename
        ok, png_buf = cv2.imencode(".png", frame)
        if not ok:
            raise RuntimeError(f"Failed to encode frame {filename}")
        fpath.write_bytes(png_buf.tobytes())

        manifest.append(
            {
                "index": idx + 1,
                "serial_number": serial,
                "image_path": str(fpath),
                "filename": filename,
                **meta,
            }
        )

    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate synthetic VisionGuard AI inspection batch")
    parser.add_argument("--count", type=int, default=50)
    parser.add_argument("--product-line", type=str, default="mixed")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "data" / "batches" / "sample_batch_50",
    )
    args = parser.parse_args()
    manifest = generate_batch_directory(
        output_dir=args.output_dir,
        count=args.count,
        product_line=args.product_line,
        seed=args.seed,
    )
    print(f"Generated {len(manifest)} inspection frames in {args.output_dir}")


if __name__ == "__main__":
    main()
