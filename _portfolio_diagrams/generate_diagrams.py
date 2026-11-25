"""Render 3 publication-grade 2400x1600 (2x Retina) IEEE dark-slate engineering diagrams
using headless Chromium from standalone HTML/SVG templates.
"""

from __future__ import annotations

from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parent
OUT_DIR = ROOT / "output"
HTML_DIR = ROOT / "html"

CHROME_CANDIDATES = [
    Path(r"C:\Program Files\Google\Chrome\Application\chrome.exe"),
    Path(r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"),
]

COMMON_CSS = """
  * { box-sizing: border-box; margin: 0; padding: 0; }
  html, body {
    width: 1200px;
    height: 800px;
    overflow: hidden;
    background: #0B1120;
    color: #F1F5F9;
    font-family: "Inter", "Segoe UI", system-ui, -apple-system, sans-serif;
  }
  .canvas {
    width: 1200px;
    height: 800px;
    padding: 24px 28px;
    background:
      radial-gradient(circle at 12% 10%, rgba(56, 189, 248, 0.08), transparent 38%),
      radial-gradient(circle at 88% 90%, rgba(16, 185, 129, 0.06), transparent 36%),
      #0B1120;
    position: relative;
  }
  .grid-bg {
    position: absolute;
    inset: 0;
    background-image:
      linear-gradient(to right, rgba(30, 41, 59, 0.35) 1px, transparent 1px),
      linear-gradient(to bottom, rgba(30, 41, 59, 0.35) 1px, transparent 1px);
    background-size: 40px 40px;
    pointer-events: none;
  }
  .header {
    position: relative;
    z-index: 2;
    display: flex;
    justify-content: space-between;
    align-items: flex-end;
    border-bottom: 1.5px solid #1E293B;
    padding-bottom: 12px;
    margin-bottom: 16px;
  }
  .title-tag {
    font-size: 10px;
    font-weight: 700;
    letter-spacing: 1.4px;
    text-transform: uppercase;
    color: #38BDF8;
    margin-bottom: 4px;
  }
  h1 {
    font-size: 21px;
    font-weight: 800;
    color: #F8FAFC;
    letter-spacing: -0.2px;
  }
  .meta-badge {
    font-family: "JetBrains Mono", Consolas, monospace;
    font-size: 10px;
    color: #94A3B8;
    background: #111827;
    border: 1px solid #1E293B;
    padding: 5px 10px;
    border-radius: 5px;
    white-space: nowrap;
  }
  .footer {
    position: absolute;
    left: 28px;
    right: 28px;
    bottom: 14px;
    z-index: 2;
    display: flex;
    justify-content: space-between;
    font-size: 10px;
    color: #64748B;
    border-top: 1px solid #1E293B;
    padding-top: 7px;
  }
  .mono { font-family: "JetBrains Mono", Consolas, monospace; }
"""


def diagram_1_html() -> str:
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8" />
<style>
{COMMON_CSS}
  .arch-stage {{
    position: relative;
    z-index: 2;
    height: 660px;
  }}
  .zone {{
    position: absolute;
    border-radius: 10px;
    border: 1.5px solid #1E293B;
    background: rgba(15, 23, 42, 0.88);
    padding: 12px 14px;
  }}
  .zone-title {{
    font-size: 10.5px;
    font-weight: 800;
    text-transform: uppercase;
    letter-spacing: 1px;
    margin-bottom: 10px;
    display: flex;
    justify-content: space-between;
    align-items: center;
  }}
  .box {{
    background: #111827;
    border: 1px solid #263449;
    border-radius: 7px;
    padding: 9px 11px;
    margin-bottom: 9px;
  }}
  .box:last-child {{ margin-bottom: 0; }}
  .box-head {{
    font-size: 12px;
    font-weight: 700;
    color: #F1F5F9;
    margin-bottom: 4px;
    display: flex;
    justify-content: space-between;
    align-items: center;
  }}
  .box-sub {{
    font-size: 10.5px;
    color: #94A3B8;
    line-height: 1.38;
  }}
  .pill {{
    font-family: "JetBrains Mono", Consolas, monospace;
    font-size: 9.5px;
    padding: 2px 6px;
    border-radius: 4px;
    font-weight: 700;
  }}
  .c-cyan {{ color: #38BDF8; border-color: rgba(56,189,248,0.45); }}
  .bg-cyan {{ background: rgba(56,189,248,0.14); color: #38BDF8; }}
  .c-indigo {{ color: #818CF8; border-color: rgba(129,140,248,0.45); }}
  .bg-indigo {{ background: rgba(129,140,248,0.14); color: #A5B4FC; }}
  .c-emerald {{ color: #10B981; border-color: rgba(16,185,129,0.45); }}
  .bg-emerald {{ background: rgba(16,185,129,0.14); color: #6EE7B7; }}
  .c-amber {{ color: #F59E0B; border-color: rgba(245,158,11,0.45); }}
  .bg-amber {{ background: rgba(245,158,11,0.16); color: #FCD34D; }}
  .c-crimson {{ color: #EF4444; border-color: rgba(239,68,68,0.45); }}
  .bg-crimson {{ background: rgba(239,68,68,0.16); color: #FCA5A5; }}
</style>
</head>
<body>
<div class="canvas">
  <div class="grid-bg"></div>
  <header class="header">
    <div>
      <div class="title-tag">FIG 01 · SYSTEM ARCHITECTURE · IEEE SYSTEMS VIEW</div>
      <h1>VisionGuard AI — Autonomous Visual Defect Inspection &amp; Event-Driven Triage Architecture</h1>
    </div>
    <div class="meta-badge">FastAPI · OpenCV 4 · ONNX Runtime · SQLite/PostgreSQL · HMAC-SHA256</div>
  </header>

  <div class="arch-stage">
    <!-- SVG orthogonal connectors -->
    <svg width="1144" height="660" style="position:absolute;inset:0;z-index:1;pointer-events:none;">
      <defs>
        <marker id="arr-cyan" markerWidth="8" markerHeight="8" refX="7" refY="4" orient="auto">
          <path d="M0,0 L8,4 L0,8 Z" fill="#38BDF8"/>
        </marker>
        <marker id="arr-indigo" markerWidth="8" markerHeight="8" refX="7" refY="4" orient="auto">
          <path d="M0,0 L8,4 L0,8 Z" fill="#818CF8"/>
        </marker>
        <marker id="arr-emerald" markerWidth="8" markerHeight="8" refX="7" refY="4" orient="auto">
          <path d="M0,0 L8,4 L0,8 Z" fill="#10B981"/>
        </marker>
        <marker id="arr-crimson" markerWidth="8" markerHeight="8" refX="7" refY="4" orient="auto">
          <path d="M0,0 L8,4 L0,8 Z" fill="#EF4444"/>
        </marker>
        <marker id="arr-amber" markerWidth="8" markerHeight="8" refX="7" refY="4" orient="auto">
          <path d="M0,0 L8,4 L0,8 Z" fill="#F59E0B"/>
        </marker>
      </defs>
      <!-- Ingestion -> CV -->
      <path d="M275,180 L335,180" stroke="#38BDF8" stroke-width="2.2" fill="none" marker-end="url(#arr-cyan)"/>
      <!-- CV -> ONNX DL -->
      <path d="M530,295 L530,340" stroke="#818CF8" stroke-width="2.2" fill="none" marker-end="url(#arr-indigo)"/>
      <!-- CV -> SLA (metrology metrics) -->
      <path d="M720,175 L780,175" stroke="#38BDF8" stroke-width="2" stroke-dasharray="5,4" fill="none" marker-end="url(#arr-cyan)"/>
      <!-- ONNX DL -> SLA -->
      <path d="M720,455 L750,455 L750,235 L780,235" stroke="#818CF8" stroke-width="2.2" fill="none" marker-end="url(#arr-indigo)"/>
      <!-- SLA -> Persistence & ERP -->
      <path d="M962,330 L962,368" stroke="#EF4444" stroke-width="2.2" fill="none" marker-end="url(#arr-crimson)"/>
      <!-- Persistence -> Console (routed cleanly above footer in bottom gutter y=648) -->
      <path d="M782,636 L754,636 L754,648 L296,648 L296,620 L275,620" stroke="#10B981" stroke-width="2.2" fill="none" marker-end="url(#arr-emerald)"/>
      <!-- Console QA Override -> SLA (routed along left-middle corridor y=318 and vertical channel x=754) -->
      <path d="M275,545 L306,545 L306,318 L754,318 L754,285 L780,285" stroke="#F59E0B" stroke-width="1.9" stroke-dasharray="5,4" fill="none" marker-end="url(#arr-amber)"/>
    </svg>

    <!-- Zone 1: Edge Ingestion -->
    <div class="zone c-cyan" style="left:0;top:16px;width:275px;height:390px;">
      <div class="zone-title"><span>1. Edge Capture &amp; API</span><span class="pill bg-cyan">INGEST</span></div>
      <div class="box">
        <div class="box-head"><span>Synthetic Frame Generator</span><span class="pill bg-cyan">640×480</span></div>
        <div class="box-sub">Metallic plates, PCB assemblies, &amp; packaging seals with illumination gradients, rotation skew, and injected defects.</div>
      </div>
      <div class="box">
        <div class="box-head"><span>FastAPI Ingestion Gateway</span><span class="pill bg-cyan">REST</span></div>
        <div class="box-sub"><span class="mono">POST /api/inspect</span> (multipart)<br/><span class="mono">POST /api/batches/run</span> (50-item batch)<br/>Assigns <span class="mono">item_id</span> &amp; <span class="mono">batch_id</span>.</div>
      </div>
      <div class="box">
        <div class="box-head"><span>Canonical Calibration Spec</span><span class="pill bg-cyan">100 mm</span></div>
        <div class="box-sub">Nominal part width = 100.0 mm<br/>Target ROI = 512×384 (~4.64 px/mm)</div>
      </div>
    </div>

    <!-- Zone 1b: Operator Console -->
    <div class="zone c-emerald" style="left:0;top:430px;width:275px;height:215px;">
      <div class="zone-title"><span>5. Operator Console UI</span><span class="pill bg-emerald">#0B1120</span></div>
      <div class="box">
        <div class="box-head"><span>Dark-Slate Telemetry UI</span><span class="pill bg-emerald">HTML/JS</span></div>
        <div class="box-sub">• Live CV Overlay &amp; Raw Frame Toggle<br/>• Pixel-to-mm Metrology Readout<br/>• HITL QA Queue (<span class="mono">PASS/REJECT</span>)<br/>• Latency &amp; Defect SVG Charts</div>
      </div>
    </div>

    <!-- Zone 2: Classical CV & Metrology -->
    <div class="zone c-cyan" style="left:338px;top:16px;width:382px;height:278px;">
      <div class="zone-title"><span>2. OpenCV Metrology Pipeline</span><span class="pill bg-cyan">cv_pipeline.py</span></div>
      <div class="box">
        <div class="box-head"><span>LAB CLAHE Normalization</span><span class="pill bg-cyan">clip=2.5</span></div>
        <div class="box-sub">Adaptive histogram equalization on L-channel (8×8 tiles) removing overhead lighting falloff.</div>
      </div>
      <div class="box">
        <div class="box-head"><span>Contour ROI &amp; Homography Warp</span><span class="pill bg-cyan">512×384</span></div>
        <div class="box-sub">Otsu segmentation + <span class="mono">minAreaRect</span> skew angle (<span class="mono">°</span>) + <span class="mono">warpPerspective</span> rectification &amp; dynamic <span class="mono">px/mm</span>.</div>
      </div>
      <div class="box">
        <div class="box-head"><span>Morphological Residual Metrology</span><span class="pill bg-cyan">mm²</span></div>
        <div class="box-sub">BlackHat + TopHat residual extracts <span class="mono">area_mm2</span>, <span class="mono">width_mm</span>, <span class="mono">height_mm</span>, and renders HUD overlays.</div>
      </div>
    </div>

    <!-- Zone 3: ONNX Runtime DL Engine -->
    <div class="zone c-indigo" style="left:338px;top:344px;width:382px;height:301px;">
      <div class="zone-title"><span>3. ONNX Deep Learning Engine</span><span class="pill bg-indigo">dl_engine.py</span></div>
      <div class="box">
        <div class="box-head"><span>YOLOv8-Lite Grid Detector</span><span class="pill bg-indigo">FP32 ONNX</span></div>
        <div class="box-sub">Input <span class="mono">1×3×192×256</span> → 5-stage Conv/BN/ReLU → <span class="mono">6×8</span> spatial grid predicting objectness, bbox &amp; class (<span class="mono">clean/scratch/crack/misaligned</span>).</div>
      </div>
      <div class="box">
        <div class="box-head"><span>MobileNetV3 Severity Classifier</span><span class="pill bg-indigo">INT8 ONNX</span></div>
        <div class="box-sub">Depthwise-separable blocks quantized with <span class="mono">DequantizeLinear</span> (<span class="mono">1×3×128×128</span>) → <span class="mono">PASS / MINOR_COSMETIC / CRITICAL_REJECT</span>.</div>
      </div>
      <div class="box">
        <div class="box-head"><span>ONNX + Metrology Fusion</span><span class="pill bg-indigo">CPU EP</span></div>
        <div class="box-sub">Fuses neural probabilities with calibrated area/skew metrics; preserves genuine <span class="mono">&lt;0.80</span> uncertainty band.</div>
      </div>
    </div>

    <!-- Zone 4: Automation Triage -->
    <div class="zone c-amber" style="left:782px;top:16px;width:362px;height:312px;">
      <div class="zone-title"><span>4. Deterministic SLA Triage Engine</span><span class="pill bg-amber">automation_engine.py</span></div>
      <div class="box">
        <div class="box-head"><span>SLA-01 · Confidence Gate</span><span class="pill bg-amber">PENDING_QA</span></div>
        <div class="box-sub"><span class="mono">conf &lt; 0.80</span> → Pause &amp; route to Human QA Queue.</div>
      </div>
      <div class="box">
        <div class="box-head"><span>SLA-02 · Critical Quarantine</span><span class="pill bg-crimson">QUARANTINED</span></div>
        <div class="box-sub"><span class="mono">CRITICAL_REJECT</span> OR <span class="mono">area ≥ 12.0 mm²</span> OR <span class="mono">|skew| &gt; 5.5°</span> → Quarantine + Signed Webhook.</div>
      </div>
      <div class="box">
        <div class="box-head"><span>SLA-03 / SLA-04 · Pass Gates</span><span class="pill bg-emerald">APPROVED</span></div>
        <div class="box-sub">Minor cosmetic logged or clean part auto-approved.</div>
      </div>
    </div>

    <!-- Zone 4b: Persistence & Webhook Dispatch -->
    <div class="zone c-crimson" style="left:782px;top:370px;width:362px;height:275px;">
      <div class="zone-title"><span>6. Persistence, ERP &amp; Audit</span><span class="pill bg-crimson">HMAC-SHA256</span></div>
      <div class="box">
        <div class="box-head"><span>Signed ERP Webhook Dispatch</span><span class="pill bg-crimson">POST 200</span></div>
        <div class="box-sub"><span class="mono">X-VisionGuard-Signature: sha256=...</span> verified by <span class="mono">/api/webhooks/erp-receiver</span> (401 on tamper).</div>
      </div>
      <div class="box">
        <div class="box-head"><span>Normalized SQLite / PostgreSQL DB</span><span class="pill bg-emerald">6 TABLES</span></div>
        <div class="box-sub"><span class="mono">inspection_batches</span>, <span class="mono">inspection_items</span>, <span class="mono">defect_detections</span>, <span class="mono">automation_events</span>, <span class="mono">webhook_deliveries</span>, <span class="mono">erp_alerts</span>.</div>
      </div>
      <div class="box">
        <div class="box-head"><span>Batch Audit Report Exporter</span><span class="pill bg-cyan">JSON + MD</span></div>
        <div class="box-sub">Generates signed pass-rate, latency &amp; quarantine reports.</div>
      </div>
    </div>
  </div>

  <footer class="footer">
    <span>VisionGuard AI v1.0 · Portfolio Architecture Specification</span>
    <span>Theme: #0B1120 Dark-Slate IEEE Engineering · Scale: 2400×1600 (2× Retina)</span>
  </footer>
</div>
</body>
</html>"""


def diagram_2_html() -> str:
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8" />
<style>
{COMMON_CSS}
  .flow-stage {{
    position: relative;
    z-index: 2;
    height: 660px;
  }}
  .step-card {{
    position: absolute;
    background: #111827;
    border: 1.5px solid #263449;
    border-radius: 9px;
    padding: 12px 14px;
  }}
  .step-num {{
    font-family: "JetBrains Mono", Consolas, monospace;
    font-size: 10px;
    font-weight: 800;
    letter-spacing: 0.9px;
    text-transform: uppercase;
    margin-bottom: 5px;
    display: flex;
    justify-content: space-between;
  }}
  .step-title {{
    font-size: 13.5px;
    font-weight: 800;
    color: #F8FAFC;
    margin-bottom: 6px;
  }}
  .step-body {{
    font-size: 10.8px;
    color: #94A3B8;
    line-height: 1.42;
  }}
  .metric-row {{
    margin-top: 8px;
    padding-top: 7px;
    border-top: 1px dashed #1E293B;
    font-family: "JetBrains Mono", Consolas, monospace;
    font-size: 10px;
    color: #CBD5E1;
  }}
  .branch-card {{
    position: absolute;
    border-radius: 9px;
    padding: 11px 13px;
    background: #0F172A;
  }}
</style>
</head>
<body>
<div class="canvas">
  <div class="grid-bg"></div>
  <header class="header">
    <div>
      <div class="title-tag">FIG 02 · END-TO-END DATAFLOW &amp; DECISION PIPELINE</div>
      <h1>CV Preprocessing, ONNX Inference, Pixel-to-mm Metrology &amp; SLA Triage Flow</h1>
    </div>
    <div class="meta-badge">Frame Input → CLAHE + Homography → Dual ONNX → Metrology Fusion → SLA Branch</div>
  </header>

  <div class="flow-stage">
    <svg width="1144" height="660" style="position:absolute;inset:0;z-index:1;pointer-events:none;">
      <defs>
        <marker id="a-c" markerWidth="8" markerHeight="8" refX="7" refY="4" orient="auto"><path d="M0,0 L8,4 L0,8 Z" fill="#38BDF8"/></marker>
        <marker id="a-i" markerWidth="8" markerHeight="8" refX="7" refY="4" orient="auto"><path d="M0,0 L8,4 L0,8 Z" fill="#818CF8"/></marker>
        <marker id="a-e" markerWidth="8" markerHeight="8" refX="7" refY="4" orient="auto"><path d="M0,0 L8,4 L0,8 Z" fill="#10B981"/></marker>
        <marker id="a-r" markerWidth="8" markerHeight="8" refX="7" refY="4" orient="auto"><path d="M0,0 L8,4 L0,8 Z" fill="#EF4444"/></marker>
        <marker id="a-a" markerWidth="8" markerHeight="8" refX="7" refY="4" orient="auto"><path d="M0,0 L8,4 L0,8 Z" fill="#F59E0B"/></marker>
      </defs>
      <!-- Top row: 1 -> 2 -> 3 -->
      <path d="M350,120 L398,120" stroke="#38BDF8" stroke-width="2.4" fill="none" marker-end="url(#a-c)"/>
      <path d="M748,120 L796,120" stroke="#38BDF8" stroke-width="2.4" fill="none" marker-end="url(#a-c)"/>
      <!-- 3 -> 4 (vertical drop on right to middle row) -->
      <path d="M970,225 L970,268" stroke="#818CF8" stroke-width="2.4" fill="none" marker-end="url(#a-i)"/>
      <!-- Middle row: 4 -> 5 (right to left) -->
      <path d="M796,355 L748,355" stroke="#818CF8" stroke-width="2.4" fill="none" marker-end="url(#a-i)"/>
      <!-- 5 -> 6 (right to left) -->
      <path d="M398,355 L350,355" stroke="#38BDF8" stroke-width="2.4" fill="none" marker-end="url(#a-c)"/>
      <!-- 6 (SLA Gate) -> 3 bottom branches -->
      <path d="M175,445 L175,500" stroke="#10B981" stroke-width="2.4" fill="none" marker-end="url(#a-e)"/>
      <path d="M275,445 L275,472 L573,472 L573,500" stroke="#EF4444" stroke-width="2.4" fill="none" marker-end="url(#a-r)"/>
      <path d="M325,445 L325,462 L970,462 L970,500" stroke="#F59E0B" stroke-width="2.4" fill="none" marker-end="url(#a-a)"/>
    </svg>

    <!-- STEP 1 -->
    <div class="step-card" style="left:0;top:16px;width:350px;height:208px;border-color:#38BDF8;">
      <div class="step-num" style="color:#38BDF8;"><span>STAGE 01</span><span>RAW INPUT</span></div>
      <div class="step-title">Raw Camera Frame Capture</div>
      <div class="step-body">
        640×480 BGR conveyor image across <strong>metallic</strong>, <strong>pcb</strong>, or <strong>packaging</strong> lines. Includes realistic radial overhead lighting falloff and ±2.8°–10.5° placement skew.
      </div>
      <div class="metric-row">Tensor: uint8[480, 640, 3] · BGR Color</div>
    </div>

    <!-- STEP 2 -->
    <div class="step-card" style="left:398px;top:16px;width:350px;height:208px;border-color:#38BDF8;">
      <div class="step-num" style="color:#38BDF8;"><span>STAGE 02</span><span>OPENCV PREPROCESS</span></div>
      <div class="step-title">CLAHE &amp; Homography Rectification</div>
      <div class="step-body">
        1) LAB color conversion + <span class="mono">CLAHE(clip=2.5, 8×8)</span> on L-channel.<br/>
        2) Otsu contour mask + <span class="mono">minAreaRect</span> measures <span class="mono">alignment_angle_deg</span>.<br/>
        3) <span class="mono">warpPerspective</span> aligns ROI to 512×384.
      </div>
      <div class="metric-row">Calibration: px_per_mm ≈ 4.64 px/mm</div>
    </div>

    <!-- STEP 3 -->
    <div class="step-card" style="left:796px;top:16px;width:348px;height:208px;border-color:#818CF8;">
      <div class="step-num" style="color:#A5B4FC;"><span>STAGE 03</span><span>CLASSICAL METROLOGY</span></div>
      <div class="step-title">Morphological Residual Extraction</div>
      <div class="step-body">
        Applies <span class="mono">BlackHat</span> + <span class="mono">TopHat</span> (11×11 kernel) inside inner ROI mask to isolate cracks, scratches, and surface abrasions in physical millimeter coordinates.
      </div>
      <div class="metric-row">Outputs: area_mm2, width_mm, height_mm</div>
    </div>

    <!-- STEP 4 -->
    <div class="step-card" style="left:796px;top:268px;width:348px;height:176px;border-color:#818CF8;">
      <div class="step-num" style="color:#A5B4FC;"><span>STAGE 04</span><span>ONNX RUNTIME</span></div>
      <div class="step-title">YOLOv8-Lite + MobileNetV3-INT8</div>
      <div class="step-body">
        • Detector (<span class="mono">1×3×192×256</span>) outputs <span class="mono">6×8</span> objectness &amp; class grid.<br/>
        • INT8 Classifier (<span class="mono">1×3×128×128</span>) outputs softmax severity logits.
      </div>
      <div class="metric-row">Providers: CPUExecutionProvider · Opset 17</div>
    </div>

    <!-- STEP 5 -->
    <div class="step-card" style="left:398px;top:268px;width:350px;height:176px;border-color:#38BDF8;">
      <div class="step-num" style="color:#38BDF8;"><span>STAGE 05</span><span>MULTI-MODAL FUSION</span></div>
      <div class="step-title">Confidence &amp; Metrology Fusion</div>
      <div class="step-body">
        Fuses ONNX class probabilities with calibrated area (<span class="mono">mm²</span>) and homography skew (<span class="mono">°</span>). Renders high-contrast HUD overlay with 10.0 mm physical ruler bar.
      </div>
      <div class="metric-row">Output: (severity, confidence, detections[])</div>
    </div>

    <!-- STEP 6 -->
    <div class="step-card" style="left:0;top:268px;width:350px;height:176px;border-color:#F59E0B;">
      <div class="step-num" style="color:#FCD34D;"><span>STAGE 06</span><span>DETERMINISTIC GATE</span></div>
      <div class="step-title">Priority SLA Rule Evaluator</div>
      <div class="step-body">
        Evaluates ordered rules <span class="mono">SLA-01 → SLA-04</span> against confidence threshold (<span class="mono">0.80</span>), critical area (<span class="mono">12.0 mm²</span>), and skew (<span class="mono">5.5°</span>).
      </div>
      <div class="metric-row">Emits immutable automation_events row</div>
    </div>

    <!-- BRANCH A: AUTO-APPROVE -->
    <div class="branch-card" style="left:0;top:502px;width:350px;height:136px;border:1.5px solid #10B981;">
      <div class="step-num" style="color:#6EE7B7;"><span>BRANCH A · SLA-03 / SLA-04</span><span>PASS</span></div>
      <div class="step-title" style="color:#6EE7B7;">Auto-Approve &amp; Release</div>
      <div class="step-body">
        Triggered when <span class="mono">conf ≥ 0.80</span> and severity is <span class="mono">PASS</span> or <span class="mono">MINOR_COSMETIC</span> (&lt; 12 mm²). State set to <strong>APPROVED</strong>; logged to batch telemetry.
      </div>
    </div>

    <!-- BRANCH B: QUARANTINE + WEBHOOK -->
    <div class="branch-card" style="left:398px;top:502px;width:350px;height:136px;border:1.5px solid #EF4444;">
      <div class="step-num" style="color:#FCA5A5;"><span>BRANCH B · SLA-02</span><span>CRITICAL</span></div>
      <div class="step-title" style="color:#FCA5A5;">Quarantine + Signed ERP Webhook</div>
      <div class="step-body">
        Triggered on <span class="mono">CRITICAL_REJECT</span>, <span class="mono">area ≥ 12.0 mm²</span>, or <span class="mono">|skew| &gt; 5.5°</span>. State → <strong>QUARANTINED</strong>; signs JSON with <span class="mono">HMAC-SHA256</span> &amp; POSTs to ERP receiver.
      </div>
    </div>

    <!-- BRANCH C: HITL QA -->
    <div class="branch-card" style="left:796px;top:502px;width:348px;height:136px;border:1.5px solid #F59E0B;">
      <div class="step-num" style="color:#FCD34D;"><span>BRANCH C · SLA-01</span><span>UNCERTAIN</span></div>
      <div class="step-title" style="color:#FCD34D;">Human-in-the-Loop QA Review</div>
      <div class="step-body">
        Triggered when <span class="mono">conf &lt; 0.80</span>. State → <strong>PENDING_QA</strong>; operator inspects overlay in console and executes <strong>PASS</strong> or <strong>REJECT</strong> override.
      </div>
    </div>
  </div>

  <footer class="footer">
    <span>VisionGuard AI v1.0 · Deterministic Inspection &amp; Triage Pipeline</span>
    <span>Scale: 2400×1600 (2× Retina) · Zero-Overlap Vector Layout</span>
  </footer>
</div>
</body>
</html>"""


def diagram_3_html() -> str:
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8" />
<style>
{COMMON_CSS}
  .section-box {{
    position: relative;
    z-index: 2;
    border: 1.5px solid #1E293B;
    border-radius: 10px;
    background: rgba(15, 23, 42, 0.85);
    padding: 12px 16px;
  }}
  .sec-label {{
    font-size: 10.5px;
    font-weight: 800;
    letter-spacing: 1px;
    text-transform: uppercase;
    color: #38BDF8;
    margin-bottom: 10px;
  }}
  .state-node {{
    position: absolute;
    border-radius: 7px;
    padding: 7px 10px;
    text-align: center;
    font-family: "JetBrains Mono", Consolas, monospace;
    font-size: 11px;
    font-weight: 700;
    background: #111827;
    border: 1.5px solid #334155;
  }}
  .edge-tag {{
    position: absolute;
    font-family: "JetBrains Mono", Consolas, monospace;
    font-size: 9.2px;
    color: #CBD5E1;
    background: #0B1120;
    padding: 1px 5px;
    border-radius: 3px;
    border: 1px solid #1E293B;
  }}
  .erd-table {{
    position: absolute;
    background: #111827;
    border: 1.5px solid #263449;
    border-radius: 7px;
    overflow: hidden;
  }}
  .erd-head {{
    background: #162033;
    padding: 6px 9px;
    font-family: "JetBrains Mono", Consolas, monospace;
    font-size: 11px;
    font-weight: 800;
    color: #38BDF8;
    border-bottom: 1px solid #263449;
    display: flex;
    justify-content: space-between;
  }}
  .erd-cols {{
    padding: 6px 9px;
    font-family: "JetBrains Mono", Consolas, monospace;
    font-size: 9.6px;
    line-height: 1.5;
    color: #CBD5E1;
  }}
  .pk {{ color: #FCD34D; font-weight: 700; }}
  .fk {{ color: #A5B4FC; font-weight: 700; }}
</style>
</head>
<body>
<div class="canvas">
  <div class="grid-bg"></div>
  <header class="header">
    <div>
      <div class="title-tag">FIG 03 · AUTOMATION STATE MACHINE &amp; NORMALIZED DATABASE SCHEMA</div>
      <h1>Inspection Item Lifecycle State Machine &amp; 5-Table Relational ERD</h1>
    </div>
    <div class="meta-badge">SQLite / PostgreSQL · Foreign-Key Enforced · Immutable Audit Events</div>
  </header>

  <!-- TOP HALF: STATE MACHINE -->
  <div class="section-box" style="height:245px;margin-bottom:14px;">
    <div class="sec-label">A. Deterministic Inspection Item State Machine</div>
    <div style="position:relative;height:195px;">
      <svg width="1110" height="195" style="position:absolute;inset:0;z-index:1;pointer-events:none;">
        <defs>
          <marker id="sm-c" markerWidth="7" markerHeight="7" refX="6" refY="3.5" orient="auto"><path d="M0,0 L7,3.5 L0,7 Z" fill="#38BDF8"/></marker>
          <marker id="sm-e" markerWidth="7" markerHeight="7" refX="6" refY="3.5" orient="auto"><path d="M0,0 L7,3.5 L0,7 Z" fill="#10B981"/></marker>
          <marker id="sm-r" markerWidth="7" markerHeight="7" refX="6" refY="3.5" orient="auto"><path d="M0,0 L7,3.5 L0,7 Z" fill="#EF4444"/></marker>
          <marker id="sm-a" markerWidth="7" markerHeight="7" refX="6" refY="3.5" orient="auto"><path d="M0,0 L7,3.5 L0,7 Z" fill="#F59E0B"/></marker>
        </defs>
        <!-- INGESTED -> ANALYZED -->
        <path d="M135,92 L222,92" stroke="#38BDF8" stroke-width="2" fill="none" marker-end="url(#sm-c)"/>
        <!-- ANALYZED -> APPROVED (top) -->
        <path d="M352,78 L420,28 L525,28" stroke="#10B981" stroke-width="2" fill="none" marker-end="url(#sm-e)"/>
        <!-- ANALYZED -> QUARANTINED (mid) -->
        <path d="M352,92 L525,92" stroke="#EF4444" stroke-width="2" fill="none" marker-end="url(#sm-r)"/>
        <!-- ANALYZED -> PENDING_QA (bottom) -->
        <path d="M352,106 L420,156 L525,156" stroke="#F59E0B" stroke-width="2" fill="none" marker-end="url(#sm-a)"/>
        <!-- PENDING_QA -> QA_RESOLVED_PASS -->
        <path d="M685,146 L765,112 L865,112" stroke="#10B981" stroke-width="2" fill="none" marker-end="url(#sm-e)"/>
        <!-- PENDING_QA -> QA_RESOLVED_REJECT -->
        <path d="M685,164 L765,164 L865,164" stroke="#EF4444" stroke-width="2" fill="none" marker-end="url(#sm-r)"/>
      </svg>

      <div class="state-node" style="left:5px;top:72px;width:130px;border-color:#38BDF8;color:#38BDF8;">INGESTED</div>
      <div class="edge-tag" style="left:145px;top:70px;">CV + ONNX</div>

      <div class="state-node" style="left:226px;top:72px;width:126px;border-color:#818CF8;color:#A5B4FC;">ANALYZED</div>

      <div class="edge-tag" style="left:386px;top:16px;color:#6EE7B7;">SLA-03/04: conf ≥ 0.80</div>
      <div class="state-node" style="left:530px;top:8px;width:155px;border-color:#10B981;color:#6EE7B7;">APPROVED</div>

      <div class="edge-tag" style="left:372px;top:71px;color:#FCA5A5;">SLA-02: Crit / ≥12mm²</div>
      <div class="state-node" style="left:530px;top:72px;width:155px;border-color:#EF4444;color:#FCA5A5;">QUARANTINED</div>

      <div class="edge-tag" style="left:392px;top:155px;color:#FCD34D;">SLA-01: conf &lt; 0.80</div>
      <div class="state-node" style="left:530px;top:136px;width:155px;border-color:#F59E0B;color:#FCD34D;">PENDING_QA</div>

      <div class="edge-tag" style="left:725px;top:92px;color:#6EE7B7;">QA Approve</div>
      <div class="state-node" style="left:870px;top:92px;width:210px;border-color:#10B981;color:#6EE7B7;">QA_RESOLVED_PASS</div>

      <div class="edge-tag" style="left:715px;top:144px;color:#FCA5A5;">QA Reject + Webhook</div>
      <div class="state-node" style="left:870px;top:145px;width:210px;border-color:#EF4444;color:#FCA5A5;">QA_RESOLVED_REJECT</div>

      <div class="edge-tag" style="left:735px;top:18px;border-color:rgba(239,68,68,0.45);color:#FCA5A5;">
        QUARANTINED &amp; QA_RESOLVED_REJECT trigger HMAC-SHA256 ERP Webhook
      </div>
    </div>
  </div>

  <!-- BOTTOM HALF: NORMALIZED ERD -->
  <div class="section-box" style="height:400px;">
    <div class="sec-label">B. Normalized Relational Database ERD (5 Core Tables + ERP Receiver)</div>
    <div style="position:relative;height:350px;">
      <svg width="1110" height="350" style="position:absolute;inset:0;z-index:1;pointer-events:none;">
        <defs>
          <marker id="erd-arr" markerWidth="7" markerHeight="7" refX="6" refY="3.5" orient="auto"><path d="M0,0 L7,3.5 L0,7 Z" fill="#38BDF8"/></marker>
        </defs>
        <!-- batches -> items (1 to N) -->
        <path d="M210,105 L268,105" stroke="#38BDF8" stroke-width="2" fill="none" marker-end="url(#erd-arr)"/>
        <!-- items -> defect_detections -->
        <path d="M498,85 L570,85" stroke="#38BDF8" stroke-width="2" fill="none" marker-end="url(#erd-arr)"/>
        <!-- items -> automation_events -->
        <path d="M498,210 L570,210" stroke="#38BDF8" stroke-width="2" fill="none" marker-end="url(#erd-arr)"/>
        <!-- automation_events -> webhook_deliveries -->
        <path d="M815,210 L875,210" stroke="#EF4444" stroke-width="2" fill="none" marker-end="url(#erd-arr)"/>
      </svg>

      <!-- Table 1: inspection_batches -->
      <div class="erd-table" style="left:0;top:14px;width:210px;">
        <div class="erd-head"><span>inspection_batches</span><span>1</span></div>
        <div class="erd-cols">
          <span class="pk">PK</span> batch_id TEXT<br/>
          batch_name TEXT<br/>
          product_line TEXT<br/>
          total_items INT<br/>
          passed_count INT<br/>
          quarantined_count INT<br/>
          pending_qa_count INT<br/>
          avg_latency_ms REAL<br/>
          audit_report_path TEXT<br/>
          created_at TEXT
        </div>
      </div>

      <!-- Table 2: inspection_items -->
      <div class="erd-table" style="left:270px;top:6px;width:228px;border-color:#38BDF8;">
        <div class="erd-head"><span>inspection_items</span><span>N</span></div>
        <div class="erd-cols">
          <span class="pk">PK</span> item_id TEXT<br/>
          <span class="fk">FK</span> batch_id TEXT<br/>
          serial_number TEXT<br/>
          surface_type TEXT<br/>
          state TEXT<br/>
          predicted_severity TEXT<br/>
          confidence REAL<br/>
          defect_area_mm2 REAL<br/>
          max_defect_length_mm REAL<br/>
          px_per_mm REAL<br/>
          alignment_angle_deg REAL<br/>
          cv_latency_ms REAL<br/>
          dl_latency_ms REAL<br/>
          total_latency_ms REAL<br/>
          qa_reviewer TEXT
        </div>
      </div>

      <!-- Table 3: defect_detections -->
      <div class="erd-table" style="left:572px;top:6px;width:242px;">
        <div class="erd-head"><span>defect_detections</span><span>N</span></div>
        <div class="erd-cols">
          <span class="pk">PK</span> detection_id TEXT<br/>
          <span class="fk">FK</span> item_id TEXT<br/>
          defect_class TEXT<br/>
          severity TEXT<br/>
          confidence REAL<br/>
          bbox_x1..y2 INT<br/>
          area_px / area_mm2 REAL<br/>
          width_mm / height_mm REAL<br/>
          source TEXT
        </div>
      </div>

      <!-- Table 4: automation_events -->
      <div class="erd-table" style="left:572px;top:172px;width:242px;border-color:#F59E0B;">
        <div class="erd-head" style="color:#FCD34D;"><span>automation_events</span><span>N</span></div>
        <div class="erd-cols">
          <span class="pk">PK</span> event_id TEXT<br/>
          <span class="fk">FK</span> item_id TEXT<br/>
          <span class="fk">FK</span> batch_id TEXT<br/>
          rule_id TEXT<br/>
          rule_name TEXT<br/>
          action TEXT<br/>
          reason TEXT<br/>
          created_at TEXT
        </div>
      </div>

      <!-- Table 5: webhook_deliveries -->
      <div class="erd-table" style="left:877px;top:148px;width:232px;border-color:#EF4444;">
        <div class="erd-head" style="color:#FCA5A5;"><span>webhook_deliveries</span><span>0..1</span></div>
        <div class="erd-cols">
          <span class="pk">PK</span> delivery_id TEXT<br/>
          <span class="fk">FK</span> event_id TEXT<br/>
          <span class="fk">FK</span> item_id TEXT<br/>
          target_url TEXT<br/>
          payload_json TEXT<br/>
          hmac_signature TEXT<br/>
          http_status INT<br/>
          response_body TEXT<br/>
          latency_ms REAL<br/>
          delivered_at TEXT
        </div>
      </div>

      <!-- Cardinality & ERP Receiver callout -->
      <div class="erd-table" style="left:877px;top:6px;width:232px;border-color:#10B981;">
        <div class="erd-head" style="color:#6EE7B7;"><span>erp_alerts (Mock ERP)</span><span>LOG</span></div>
        <div class="erd-cols">
          <span class="pk">PK</span> alert_id TEXT<br/>
          item_id / batch_id TEXT<br/>
          severity / area_mm2<br/>
          signature_verified INT<br/>
          received_at TEXT
        </div>
      </div>
    </div>
  </div>

  <footer class="footer">
    <span>VisionGuard AI v1.0 · State Machine &amp; Normalized ERD</span>
    <span>Scale: 2400×1600 (2× Retina) · Dark-Slate #0B1120</span>
  </footer>
</div>
</body>
</html>"""


def find_browser() -> Path:
    for p in CHROME_CANDIDATES:
        if p.exists():
            return p
    raise FileNotFoundError("Chrome or Edge binary not found for headless rendering")


def render_diagram(browser: Path, html_path: Path, png_path: Path) -> None:
    png_path.parent.mkdir(parents=True, exist_ok=True)
    uri = html_path.resolve().as_uri()
    cmd = [
        str(browser),
        "--headless=new",
        "--no-sandbox",
        "--disable-gpu",
        f"--screenshot={png_path}",
        "--window-size=1200,800",
        "--force-device-scale-factor=2",
        uri,
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, check=True)
    if not png_path.exists():
        raise RuntimeError(f"Screenshot missing after rendering {png_path}: {proc.stderr}")


def main() -> None:
    browser = find_browser()
    HTML_DIR.mkdir(parents=True, exist_ok=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    specs = [
        ("01_visionguard_architecture", diagram_1_html()),
        ("02_cv_dl_triage_pipeline", diagram_2_html()),
        ("03_automation_erd_state_machine", diagram_3_html()),
    ]
    for name, html in specs:
        html_file = HTML_DIR / f"{name}.html"
        png_file = OUT_DIR / f"{name}.png"
        html_file.write_text(html, encoding="utf-8")
        render_diagram(browser, html_file, png_file)
        print(f"Rendered {png_file}")


if __name__ == "__main__":
    main()
