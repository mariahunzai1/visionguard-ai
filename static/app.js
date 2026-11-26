const state = {
  items: [],
  qaQueue: [],
  events: [],
  webhooks: [],
  erpAlerts: [],
  telemetry: null,
  selectedItemId: null,
  selectedQaId: null,
  viewMode: "overlay",
  auditFilter: "ALL",
  auditQuery: "",
};

const SVG_COPY = `<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.75" aria-hidden="true"><rect x="9" y="9" width="13" height="13" rx="2"/><path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"/></svg>`;

function showToast(msg) {
  const el = document.getElementById("toast");
  el.textContent = msg;
  el.classList.remove("hidden");
  clearTimeout(showToast._timer);
  showToast._timer = setTimeout(() => el.classList.add("hidden"), 3200);
}

async function apiJson(url, options = {}) {
  const res = await fetch(url, options);
  if (!res.ok) {
    const text = await res.text();
    throw new Error(`${res.status}: ${text}`);
  }
  return res.json();
}

function escapeHtml(str) {
  return String(str ?? "")
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;");
}

function formatUtc(iso) {
  if (!iso) return "--";
  return iso.replace("T", " ").slice(0, 19) + " UTC";
}

function statusPillMeta(itemState, severity) {
  if (itemState === "QUARANTINED" || itemState === "QA_RESOLVED_REJECT") {
    return { cls: "status-rose", label: itemState === "QA_RESOLVED_REJECT" ? "QA Quarantined" : "Quarantined" };
  }
  if (itemState === "PENDING_QA") {
    return { cls: "status-amber", label: "Pending QA" };
  }
  if (itemState === "APPROVED" && severity === "MINOR_COSMETIC") {
    return { cls: "status-slate", label: "Logged" };
  }
  if (itemState === "APPROVED" || itemState === "QA_RESOLVED_PASS") {
    return { cls: "status-emerald", label: itemState === "QA_RESOLVED_PASS" ? "QA Approved" : "Approved" };
  }
  return { cls: "status-slate", label: itemState || "Unknown" };
}

function primaryDefectClass(item) {
  if (item.detections && item.detections.length) {
    return item.detections[0].defect_class;
  }
  return item.predicted_severity === "PASS" ? "clean" : "surface_anomaly";
}

function formatDimensions(item) {
  if (item.detections && item.detections.length) {
    const d = item.detections[0];
    return `${d.width_mm.toFixed(1)} x ${d.height_mm.toFixed(1)} mm (${item.defect_area_mm2.toFixed(2)} mm2)`;
  }
  if (item.defect_area_mm2 > 0) {
    return `${item.max_defect_length_mm.toFixed(1)} mm (${item.defect_area_mm2.toFixed(2)} mm2)`;
  }
  return "0.0 x 0.0 mm";
}

function findItemEvent(itemId) {
  return state.events.find((e) => e.item_id === itemId) || null;
}

function setupTabs() {
  document.querySelectorAll(".nav-tab").forEach((btn) => {
    btn.addEventListener("click", () => {
      document.querySelectorAll(".nav-tab").forEach((b) => {
        b.classList.remove("active");
        b.setAttribute("aria-selected", "false");
      });
      document.querySelectorAll(".tab-view").forEach((v) => v.classList.remove("active"));
      btn.classList.add("active");
      btn.setAttribute("aria-selected", "true");
      document.getElementById(`tab-${btn.dataset.tab}`).classList.add("active");
    });
  });
}

function renderKPIs(t) {
  const total = t.total_items || 0;
  const passed = (t.approved_count || 0) + (t.qa_resolved_pass_count || 0);
  const passRate = total > 0 ? ((passed / total) * 100).toFixed(1) : "0.0";

  document.getElementById("kpi-total").textContent = total;
  document.getElementById("kpi-batches").textContent = `${t.total_batches} batch(es) recorded`;
  document.getElementById("kpi-approved").textContent = passed;
  document.getElementById("kpi-pass-rate").textContent = `${passRate}% automated pass rate`;
  document.getElementById("kpi-qa").textContent = t.pending_qa_count;
  document.getElementById("qa-tab-badge").textContent = t.pending_qa_count;
  document.getElementById("kpi-latency").textContent = `${t.avg_total_latency_ms.toFixed(1)} ms`;
  document.getElementById("kpi-latency-split").textContent =
    `CV ${t.avg_cv_latency_ms.toFixed(1)} ms · ONNX ${t.avg_dl_latency_ms.toFixed(1)} ms · P95 ${t.p95_total_latency_ms.toFixed(1)} ms`;

  if (t.model_info) {
    const q = t.model_info.classifier_quantization.includes("INT8") ? "FP32/INT8" : "FP32";
    document.getElementById("runtime-pill").textContent = `ONNX CPU · ${q}`;
  }
}

function renderPartSelector(items) {
  const sel = document.getElementById("part-selector");
  if (!items.length) {
    sel.innerHTML = `<option value="">No parts inspected</option>`;
    return;
  }
  if (!state.selectedItemId || !items.some((i) => i.item_id === state.selectedItemId)) {
    state.selectedItemId = items[0].item_id;
  }
  sel.innerHTML = items
    .map((item) => {
      const pill = statusPillMeta(item.state, item.predicted_severity);
      return `<option value="${escapeHtml(item.item_id)}">${escapeHtml(item.serial_number)} · ${escapeHtml(item.surface_type.toUpperCase())} · ${escapeHtml(pill.label)}</option>`;
    })
    .join("");
  sel.value = state.selectedItemId;
}

function drawMetrologyMask(item) {
  const canvas = document.getElementById("workbench-mask-canvas");
  const img = new Image();
  img.crossOrigin = "anonymous";
  img.onload = () => {
    const w = 512;
    const h = 384;
    canvas.width = w;
    canvas.height = h;
    const ctx = canvas.getContext("2d");

    ctx.fillStyle = "#050506";
    ctx.fillRect(0, 0, w, h);

    // Subtle monochrome luminance edge underlay from raw camera frame
    ctx.save();
    ctx.globalAlpha = 0.28;
    ctx.filter = "grayscale(100%) contrast(165%)";
    ctx.drawImage(img, 0, 0, w, h);
    ctx.restore();

    // Calibrated 32px metrology reticle
    ctx.strokeStyle = "rgba(59, 130, 246, 0.16)";
    ctx.lineWidth = 1;
    for (let x = 32; x < w; x += 32) {
      ctx.beginPath();
      ctx.moveTo(x, 0);
      ctx.lineTo(x, h);
      ctx.stroke();
    }
    for (let y = 32; y < h; y += 32) {
      ctx.beginPath();
      ctx.moveTo(0, y);
      ctx.lineTo(w, y);
      ctx.stroke();
    }

    // ROI active inspection boundary
    ctx.strokeStyle = "rgba(161, 161, 170, 0.45)";
    ctx.setLineDash([4, 4]);
    ctx.strokeRect(24, 22, w - 48, h - 44);
    ctx.setLineDash([]);

    // Highlight localized defect metrology masks
    const dets = item.detections || [];
    dets.forEach((d) => {
      const isCrit = d.severity === "CRITICAL_REJECT";
      ctx.fillStyle = isCrit ? "rgba(244, 63, 94, 0.24)" : "rgba(245, 158, 11, 0.24)";
      ctx.strokeStyle = isCrit ? "#F43F5E" : "#F59E0B";
      ctx.lineWidth = 1.5;
      const bw = Math.max(8, d.bbox_x2 - d.bbox_x1);
      const bh = Math.max(8, d.bbox_y2 - d.bbox_y1);
      ctx.fillRect(d.bbox_x1, d.bbox_y1, bw, bh);
      ctx.strokeRect(d.bbox_x1, d.bbox_y1, bw, bh);

      ctx.fillStyle = "#FAFAFA";
      ctx.font = "500 11px JetBrains Mono, monospace";
      ctx.fillText(
        `${d.defect_class.toUpperCase()} ${d.area_mm2.toFixed(1)}mm2`,
        d.bbox_x1,
        Math.max(14, d.bbox_y1 - 5)
      );
    });

    // Bottom calibration scale indicator
    const barPx = Math.round(10 * item.px_per_mm);
    ctx.strokeStyle = "#3B82F6";
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.moveTo(w - 16 - barPx, h - 16);
    ctx.lineTo(w - 16, h - 16);
    ctx.stroke();
    ctx.fillStyle = "#A1A1AA";
    ctx.font = "500 10.5px JetBrains Mono, monospace";
    ctx.fillText(`10.0 mm (${item.px_per_mm.toFixed(2)} px/mm)`, w - 16 - Math.max(barPx, 130), h - 22);
  };
  img.src = `${item.raw_image_url || item.annotated_image_url}?t=${Date.now()}`;
}

function renderWorkbench() {
  const item = state.items.find((i) => i.item_id === state.selectedItemId) || state.items[0];
  const imgEl = document.getElementById("workbench-image");
  const canvasEl = document.getElementById("workbench-mask-canvas");
  const emptyEl = document.getElementById("workbench-empty");

  if (!item) {
    imgEl.classList.add("hidden");
    canvasEl.classList.add("hidden");
    emptyEl.classList.remove("hidden");
    return;
  }

  state.selectedItemId = item.item_id;
  emptyEl.classList.add("hidden");

  if (state.viewMode === "mask") {
    imgEl.classList.add("hidden");
    canvasEl.classList.remove("hidden");
    drawMetrologyMask(item);
  } else {
    canvasEl.classList.add("hidden");
    const url = state.viewMode === "raw" ? item.raw_image_url : item.annotated_image_url;
    if (url) {
      imgEl.src = `${url}?t=${Date.now()}`;
      imgEl.classList.remove("hidden");
    }
  }

  // Bottom viewport telemetry bar
  const clsName = primaryDefectClass(item);
  document.getElementById("tele-cal").textContent = `${item.px_per_mm.toFixed(2)} px/mm`;
  document.getElementById("tele-class").textContent = clsName.toUpperCase();
  document.getElementById("tele-size").textContent = formatDimensions(item);
  document.getElementById("tele-time").textContent = `${item.total_latency_ms.toFixed(2)} ms`;

  // Right-hand SLA Triage & Metrology Dossier
  const pill = statusPillMeta(item.state, item.predicted_severity);
  const pillEl = document.getElementById("dossier-status-pill");
  pillEl.className = `status-pill ${pill.cls}`;
  pillEl.textContent = pill.label;

  const evt = findItemEvent(item.item_id);
  document.getElementById("dossier-subtitle").textContent =
    `${item.surface_type.toUpperCase()} line · ${evt ? evt.rule_name : "Deterministic SLA evaluation"}`;

  document.getElementById("kv-part-id").textContent = `${item.serial_number} (${item.item_id})`;
  document.getElementById("kv-timestamp").textContent = formatUtc(item.created_at);
  document.getElementById("kv-confidence").textContent = `${(item.confidence * 100).toFixed(1)}%`;
  document.getElementById("kv-severity").textContent = item.predicted_severity;
  document.getElementById("kv-dimensions").textContent = formatDimensions(item);
  document.getElementById("kv-skew").textContent = `${item.alignment_angle_deg >= 0 ? "+" : ""}${item.alignment_angle_deg.toFixed(2)} deg`;
  document.getElementById("kv-policy").textContent = evt ? `${evt.rule_id} · ${evt.action}` : item.state;

  const regionsEl = document.getElementById("dossier-regions");
  if (!item.detections || !item.detections.length) {
    regionsEl.innerHTML = `<div class="region-item"><span class="card-subtitle">No defect regions localized within tolerance.</span></div>`;
  } else {
    regionsEl.innerHTML = item.detections
      .map(
        (d) => `
      <div class="region-item">
        <span><strong>${escapeHtml(d.defect_class.toUpperCase())}</strong> · <span class="mono">${(d.confidence * 100).toFixed(1)}%</span></span>
        <span class="mono">${d.area_mm2.toFixed(2)} mm2 (${d.width_mm.toFixed(1)}x${d.height_mm.toFixed(1)} mm)</span>
      </div>
    `
      )
      .join("");
  }

  const actionsEl = document.getElementById("dossier-actions");
  if (item.state === "PENDING_QA") {
    actionsEl.classList.remove("hidden");
  } else {
    actionsEl.classList.add("hidden");
  }
}

function renderQAQueue() {
  const queue = state.qaQueue;
  const listEl = document.getElementById("qa-master-list");
  const totalPill = document.getElementById("qa-queue-total");
  const emptyEl = document.getElementById("qa-detail-empty");
  const bodyEl = document.getElementById("qa-detail-body");

  totalPill.textContent = `${queue.length} Pending`;

  if (!queue.length) {
    listEl.innerHTML = `<div class="empty-placeholder"><span>Queue is clear.</span></div>`;
    emptyEl.classList.remove("hidden");
    bodyEl.classList.add("hidden");
    return;
  }

  if (!state.selectedQaId || !queue.some((i) => i.item_id === state.selectedQaId)) {
    state.selectedQaId = queue[0].item_id;
  }

  listEl.innerHTML = queue
    .map(
      (item) => `
    <button class="qa-item ${item.item_id === state.selectedQaId ? "active" : ""}" data-qa-id="${escapeHtml(item.item_id)}" type="button">
      <img class="qa-thumb" src="${escapeHtml(item.annotated_image_url)}?t=${Date.now()}" alt="${escapeHtml(item.serial_number)}" />
      <div class="qa-item-meta">
        <div class="qa-item-top">
          <strong class="mono">${escapeHtml(item.serial_number)}</strong>
          <span class="status-pill status-amber">${(item.confidence * 100).toFixed(1)}%</span>
        </div>
        <div class="qa-item-sub">${escapeHtml(item.surface_type.toUpperCase())} · ${escapeHtml(primaryDefectClass(item))} · ${escapeHtml(item.predicted_severity)}</div>
      </div>
    </button>
  `
    )
    .join("");

  listEl.querySelectorAll(".qa-item").forEach((btn) => {
    btn.addEventListener("click", () => {
      state.selectedQaId = btn.dataset.qaId;
      renderQAQueue();
    });
  });

  const active = queue.find((i) => i.item_id === state.selectedQaId) || queue[0];
  emptyEl.classList.add("hidden");
  bodyEl.classList.remove("hidden");

  document.getElementById("qa-detail-title").textContent = `${active.serial_number} (${active.item_id})`;
  document.getElementById("qa-detail-sub").textContent = `${active.surface_type.toUpperCase()} · ${formatUtc(active.created_at)}`;
  document.getElementById("qa-detail-conf").textContent = `Conf ${(active.confidence * 100).toFixed(1)}%`;
  document.getElementById("qa-detail-image").src = `${active.annotated_image_url}?t=${Date.now()}`;
  document.getElementById("qa-m-grade").textContent = active.predicted_severity;
  document.getElementById("qa-m-area").textContent = `${active.defect_area_mm2.toFixed(2)} mm2`;
  document.getElementById("qa-m-skew").textContent = `${active.alignment_angle_deg >= 0 ? "+" : ""}${active.alignment_angle_deg.toFixed(2)} deg`;
  document.getElementById("qa-m-cal").textContent = `${active.px_per_mm.toFixed(2)} px/mm`;
}

async function resolveQAItem(itemId, decision, notesOverride) {
  const reviewer = document.getElementById("qa-reviewer-input")?.value?.trim() || "qa_lead_operator";
  const notes = notesOverride || document.getElementById("qa-notes-input")?.value?.trim() || "Verified in console";
  try {
    await apiJson(`/api/qa-queue/${encodeURIComponent(itemId)}/resolve`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ decision, reviewer, notes }),
    });
    showToast(`Item ${itemId} resolved as ${decision === "PASS" ? "Approved" : "Quarantined"}`);
    await loadAll();
  } catch (err) {
    showToast(`QA resolution error: ${err.message}`);
  }
}

window.resolveQA = resolveQAItem;

function matchesAuditFilter(item) {
  if (state.auditFilter === "APPROVED") {
    if (item.state !== "APPROVED" && item.state !== "QA_RESOLVED_PASS") return false;
  } else if (state.auditFilter === "PENDING_QA") {
    if (item.state !== "PENDING_QA") return false;
  } else if (state.auditFilter === "QUARANTINED") {
    if (item.state !== "QUARANTINED" && item.state !== "QA_RESOLVED_REJECT") return false;
  }
  const q = state.auditQuery.trim().toLowerCase();
  if (!q) return true;
  const hay = `${item.serial_number} ${item.item_id} ${item.surface_type} ${primaryDefectClass(item)} ${item.state}`.toLowerCase();
  return hay.includes(q);
}

function renderAuditLog() {
  const tbody = document.getElementById("audit-tbody");
  const filtered = state.items.filter(matchesAuditFilter);

  if (!filtered.length) {
    tbody.innerHTML = `<tr><td colspan="8" style="text-align:center;color:var(--text-muted);height:64px;">No matching inspection records.</td></tr>`;
    return;
  }

  tbody.innerHTML = filtered
    .map((item) => {
      const evt = findItemEvent(item.item_id);
      const pill = statusPillMeta(item.state, item.predicted_severity);
      return `
      <tr class="clickable-row" data-audit-id="${escapeHtml(item.item_id)}">
        <td class="mono">${escapeHtml(item.serial_number)}</td>
        <td class="mono">${escapeHtml(formatUtc(item.created_at))}</td>
        <td>${escapeHtml(primaryDefectClass(item))}</td>
        <td class="mono">${escapeHtml(formatDimensions(item))}</td>
        <td class="mono">${(item.confidence * 100).toFixed(1)}%</td>
        <td class="mono">${escapeHtml(evt ? evt.rule_id : "--")}</td>
        <td><span class="status-pill ${pill.cls}">${escapeHtml(pill.label)}</span></td>
        <td class="mono num-col">${item.total_latency_ms.toFixed(1)} ms</td>
      </tr>
    `;
    })
    .join("");

  tbody.querySelectorAll("tr.clickable-row").forEach((row) => {
    row.addEventListener("click", () => openItemDrawer(row.dataset.auditId));
  });
}

function renderWebhooks() {
  const tbody = document.getElementById("webhooks-tbody");
  const summaryPill = document.getElementById("webhooks-summary-pill");
  const okCount = state.webhooks.filter((w) => w.http_status >= 200 && w.http_status < 300).length;
  summaryPill.textContent = `${okCount} / ${state.webhooks.length} Verified (200 OK)`;

  if (!state.webhooks.length) {
    tbody.innerHTML = `<tr><td colspan="6" style="text-align:center;color:var(--text-muted);height:64px;">No ERP quarantine webhooks dispatched yet.</td></tr>`;
    return;
  }

  tbody.innerHTML = state.webhooks
    .map((w) => {
      const evt = state.events.find((e) => e.event_id === w.event_id);
      const ok = w.http_status >= 200 && w.http_status < 300;
      const trig = evt ? `${evt.rule_id} · ${evt.action}` : "CRITICAL_QUARANTINE";
      return `
      <tr>
        <td class="mono">${escapeHtml(w.event_id)}</td>
        <td class="mono">${escapeHtml(w.item_id)}</td>
        <td class="mono">${escapeHtml(trig)}</td>
        <td><span class="status-pill ${ok ? "status-emerald" : "status-rose"}">${w.http_status} ${ok ? "OK" : "ERR"}</span></td>
        <td>
          <span class="sig-cell">
            <span class="mono">${escapeHtml(w.hmac_signature.slice(0, 22))}...</span>
            <button class="icon-btn copy-sig-btn" data-sig="${escapeHtml(w.hmac_signature)}" type="button" title="Copy HMAC-SHA256 signature" aria-label="Copy signature">
              ${SVG_COPY}
            </button>
          </span>
        </td>
        <td class="num-col">
          <button class="btn btn-secondary btn-inline view-payload-btn" data-delivery-id="${escapeHtml(w.delivery_id)}" type="button">
            Payload
          </button>
        </td>
      </tr>
    `;
    })
    .join("");

  tbody.querySelectorAll(".copy-sig-btn").forEach((btn) => {
    btn.addEventListener("click", async () => {
      try {
        await navigator.clipboard.writeText(btn.dataset.sig);
        showToast("HMAC-SHA256 signature copied");
      } catch {
        showToast(btn.dataset.sig);
      }
    });
  });

  tbody.querySelectorAll(".view-payload-btn").forEach((btn) => {
    btn.addEventListener("click", () => openWebhookDrawer(btn.dataset.deliveryId));
  });
}

function openDrawer(title, subtitle, htmlBody) {
  document.getElementById("drawer-title").textContent = title;
  document.getElementById("drawer-subtitle").textContent = subtitle;
  document.getElementById("drawer-body").innerHTML = htmlBody;
  document.getElementById("drawer-backdrop").classList.remove("hidden");
  document.getElementById("detail-drawer").classList.remove("hidden");
}

function closeDrawer() {
  document.getElementById("drawer-backdrop").classList.add("hidden");
  document.getElementById("detail-drawer").classList.add("hidden");
}

function openItemDrawer(itemId) {
  const item = state.items.find((i) => i.item_id === itemId);
  if (!item) return;
  const evt = findItemEvent(item.item_id);
  const pill = statusPillMeta(item.state, item.predicted_severity);
  const body = `
    <div class="vision-stage" style="min-height:240px;">
      <img class="stage-media" src="${escapeHtml(item.annotated_image_url)}?t=${Date.now()}" alt="${escapeHtml(item.serial_number)}" />
    </div>
    <dl class="kv-list" style="padding:0;">
      <div class="kv-row"><dt>Disposition</dt><dd><span class="status-pill ${pill.cls}">${escapeHtml(pill.label)}</span></dd></div>
      <div class="kv-row"><dt>Surface Line</dt><dd class="mono">${escapeHtml(item.surface_type.toUpperCase())}</dd></div>
      <div class="kv-row"><dt>Confidence</dt><dd class="mono">${(item.confidence * 100).toFixed(1)}%</dd></div>
      <div class="kv-row"><dt>Severity Grade</dt><dd class="mono">${escapeHtml(item.predicted_severity)}</dd></div>
      <div class="kv-row"><dt>Calibrated Area</dt><dd class="mono">${item.defect_area_mm2.toFixed(2)} mm2</dd></div>
      <div class="kv-row"><dt>Calibration / Skew</dt><dd class="mono">${item.px_per_mm.toFixed(2)} px/mm · ${item.alignment_angle_deg.toFixed(2)} deg</dd></div>
      <div class="kv-row"><dt>CV / ONNX Latency</dt><dd class="mono">${item.cv_latency_ms.toFixed(2)} ms / ${item.dl_latency_ms.toFixed(2)} ms</dd></div>
      <div class="kv-row"><dt>Automation Rule</dt><dd class="mono">${escapeHtml(evt ? `${evt.rule_id} (${evt.action})` : "--")}</dd></div>
    </dl>
    <div>
      <div class="section-eyebrow">Audit Reason</div>
      <div class="code-block">${escapeHtml(evt ? evt.reason : "No automation event logged.")}</div>
    </div>
  `;
  openDrawer(`${item.serial_number}`, `${item.item_id} · ${formatUtc(item.created_at)}`, body);
}

function openWebhookDrawer(deliveryId) {
  const wh = state.webhooks.find((w) => w.delivery_id === deliveryId);
  if (!wh) return;
  let formatted = wh.payload_json;
  try {
    formatted = JSON.stringify(JSON.parse(wh.payload_json), null, 2);
  } catch {
    // keep raw string
  }
  const body = `
    <dl class="kv-list" style="padding:0;">
      <div class="kv-row"><dt>Delivery ID</dt><dd class="mono">${escapeHtml(wh.delivery_id)}</dd></div>
      <div class="kv-row"><dt>Target Endpoint</dt><dd class="mono">${escapeHtml(wh.target_url)}</dd></div>
      <div class="kv-row"><dt>HTTP Status</dt><dd><span class="status-pill ${wh.http_status === 200 ? "status-emerald" : "status-rose"}">${wh.http_status}</span></dd></div>
      <div class="kv-row"><dt>Round-Trip Latency</dt><dd class="mono">${wh.latency_ms.toFixed(2)} ms</dd></div>
      <div class="kv-row"><dt>Delivered At</dt><dd class="mono">${escapeHtml(formatUtc(wh.delivered_at))}</dd></div>
    </dl>
    <div>
      <div class="section-eyebrow">HMAC-SHA256 Signature Header</div>
      <div class="code-block">${escapeHtml(wh.hmac_signature)}</div>
    </div>
    <div>
      <div class="section-eyebrow">Signed JSON Payload</div>
      <pre class="code-block">${escapeHtml(formatted)}</pre>
    </div>
    <div>
      <div class="section-eyebrow">ERP Receiver Response</div>
      <div class="code-block">${escapeHtml(wh.response_body)}</div>
    </div>
  `;
  openDrawer(`Webhook Payload`, `${wh.event_id} · ${wh.item_id}`, body);
}

function openInspectModal() {
  document.getElementById("modal-backdrop").classList.remove("hidden");
  document.getElementById("inspect-modal").classList.remove("hidden");
}

function closeInspectModal() {
  document.getElementById("modal-backdrop").classList.add("hidden");
  document.getElementById("inspect-modal").classList.add("hidden");
}

async function inspectUploadedFile(file, surfaceType, serialNumber) {
  const allowed = ["image/jpeg", "image/png", "image/webp"];
  if (file.type && !allowed.includes(file.type)) {
    showToast("Unsupported file type. Use PNG, JPEG, or WebP.");
    return;
  }
  if (file.size > 5 * 1024 * 1024) {
    showToast("File exceeds 5 MB maximum upload limit.");
    return;
  }
  const form = new FormData();
  form.append("file", file);
  form.append("surface_type", surfaceType);
  if (serialNumber) {
    form.append("serial_number", serialNumber);
  }
  const item = await apiJson("/api/inspect", { method: "POST", body: form });
  state.selectedItemId = item.item_id;
  closeInspectModal();
  showToast(`Inspected ${item.serial_number}: ${statusPillMeta(item.state, item.predicted_severity).label}`);
  await loadAll();
}

async function runPresetInspection(targetGrade) {
  const match =
    state.items.find((i) =>
      targetGrade === "PENDING_QA" ? i.state === "PENDING_QA" : i.predicted_severity === targetGrade && i.raw_image_url
    ) || state.items[0];
  if (!match || !match.raw_image_url) {
    showToast("No seeded sample frame available; upload an image file.");
    return;
  }
  const res = await fetch(match.raw_image_url);
  const blob = await res.blob();
  const file = new File([blob], `${match.serial_number}_preset.png`, { type: "image/png" });
  const surface = document.getElementById("upload-surface").value || match.surface_type;
  const serial = document.getElementById("upload-serial").value.trim();
  await inspectUploadedFile(file, surface, serial);
}

async function loadAll() {
  const [telemetry, items, events, webhooks, erpAlerts, qaQueue] = await Promise.all([
    apiJson("/api/telemetry"),
    apiJson("/api/items?limit=100"),
    apiJson("/api/automation-events?limit=120"),
    apiJson("/api/webhooks/deliveries?limit=100"),
    apiJson("/api/webhooks/erp-alerts?limit=50"),
    apiJson("/api/qa-queue?limit=100"),
  ]);

  state.telemetry = telemetry;
  state.items = items;
  state.events = events;
  state.webhooks = webhooks;
  state.erpAlerts = erpAlerts;
  state.qaQueue = qaQueue;

  renderKPIs(telemetry);
  renderPartSelector(items);
  renderWorkbench();
  renderQAQueue();
  renderAuditLog();
  renderWebhooks();
}

document.addEventListener("DOMContentLoaded", () => {
  setupTabs();

  // Viewport toggles
  document.querySelectorAll(".seg-btn").forEach((btn) => {
    btn.addEventListener("click", () => {
      document.querySelectorAll(".seg-btn").forEach((b) => b.classList.remove("active"));
      btn.classList.add("active");
      state.viewMode = btn.dataset.view;
      renderWorkbench();
    });
  });

  // Part selector
  document.getElementById("part-selector").addEventListener("change", (e) => {
    state.selectedItemId = e.target.value;
    renderWorkbench();
  });

  // Workbench inline QA actions
  document.getElementById("dossier-approve-btn").addEventListener("click", () => {
    if (state.selectedItemId) resolveQAItem(state.selectedItemId, "PASS", "Approved from Workbench dossier");
  });
  document.getElementById("dossier-quarantine-btn").addEventListener("click", () => {
    if (state.selectedItemId) resolveQAItem(state.selectedItemId, "REJECT", "Quarantined from Workbench dossier");
  });

  // QA Review Queue actions
  document.getElementById("qa-confirm-pass-btn").addEventListener("click", () => {
    if (state.selectedQaId) resolveQAItem(state.selectedQaId, "PASS");
  });
  document.getElementById("qa-confirm-reject-btn").addEventListener("click", () => {
    if (state.selectedQaId) resolveQAItem(state.selectedQaId, "REJECT");
  });

  // Audit Log search & filter pills
  document.getElementById("audit-search").addEventListener("input", (e) => {
    state.auditQuery = e.target.value;
    renderAuditLog();
  });
  document.querySelectorAll(".filter-pill").forEach((btn) => {
    btn.addEventListener("click", () => {
      document.querySelectorAll(".filter-pill").forEach((b) => b.classList.remove("active"));
      btn.classList.add("active");
      state.auditFilter = btn.dataset.filter;
      renderAuditLog();
    });
  });

  // Drawer & Modal bindings
  document.getElementById("drawer-close-btn").addEventListener("click", closeDrawer);
  document.getElementById("drawer-backdrop").addEventListener("click", closeDrawer);
  document.getElementById("open-inspect-modal-btn").addEventListener("click", openInspectModal);
  document.getElementById("modal-close-btn").addEventListener("click", closeInspectModal);
  document.getElementById("modal-backdrop").addEventListener("click", closeInspectModal);

  document.addEventListener("keydown", (e) => {
    if (e.key === "Escape") {
      closeDrawer();
      closeInspectModal();
    }
  });

  // File upload & dropzone
  const uploadInput = document.getElementById("upload-input");
  const dropzone = document.getElementById("dropzone");

  uploadInput.addEventListener("change", async (e) => {
    const file = e.target.files?.[0];
    if (!file) return;
    try {
      const surface = document.getElementById("upload-surface").value;
      const serial = document.getElementById("upload-serial").value.trim();
      await inspectUploadedFile(file, surface, serial);
    } catch (err) {
      showToast(`Inspection failed: ${err.message}`);
    } finally {
      uploadInput.value = "";
    }
  });

  dropzone.addEventListener("dragover", (e) => {
    e.preventDefault();
    dropzone.classList.add("dragover");
  });
  dropzone.addEventListener("dragleave", () => dropzone.classList.remove("dragover"));
  dropzone.addEventListener("drop", async (e) => {
    e.preventDefault();
    dropzone.classList.remove("dragover");
    const file = e.dataTransfer?.files?.[0];
    if (!file) return;
    try {
      const surface = document.getElementById("upload-surface").value;
      const serial = document.getElementById("upload-serial").value.trim();
      await inspectUploadedFile(file, surface, serial);
    } catch (err) {
      showToast(`Inspection failed: ${err.message}`);
    }
  });

  // Sample presets inside modal
  document.querySelectorAll(".preset-btn").forEach((btn) => {
    btn.addEventListener("click", async () => {
      btn.disabled = true;
      try {
        await runPresetInspection(btn.dataset.preset);
      } catch (err) {
        showToast(`Preset error: ${err.message}`);
      } finally {
        btn.disabled = false;
      }
    });
  });

  // Simulate 50-Item Batch button
  document.getElementById("simulate-batch-btn").addEventListener("click", async () => {
    const btn = document.getElementById("simulate-batch-btn");
    const label = btn.querySelector("span");
    btn.disabled = true;
    label.textContent = "Simulating...";
    try {
      const res = await apiJson("/api/batches/run", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          batch_name: `Batch-${new Date().toISOString().slice(11, 19)}`,
          product_line: "mixed",
          count: 50,
          seed: Math.floor(Math.random() * 9000) + 100,
        }),
      });
      showToast(`Simulated batch complete: ${res.items_processed} parts inspected`);
      await loadAll();
    } catch (err) {
      showToast(`Batch simulation failed: ${err.message}`);
    } finally {
      btn.disabled = false;
      label.textContent = "Simulate Batch";
    }
  });

  loadAll().catch((err) => showToast(`Failed to load telemetry: ${err.message}`));
});
