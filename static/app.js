const state = {
  items: [],
  selectedItemId: null,
  viewMode: "overlay",
  telemetry: null,
};

function showToast(msg) {
  const el = document.getElementById("toast");
  el.textContent = msg;
  el.classList.remove("hidden");
  setTimeout(() => el.classList.add("hidden"), 3500);
}

async function apiJson(url, options = {}) {
  const res = await fetch(url, options);
  if (!res.ok) {
    const text = await res.text();
    throw new Error(`${res.status}: ${text}`);
  }
  return res.json();
}

function setupTabs() {
  document.querySelectorAll(".tab-btn").forEach((btn) => {
    btn.addEventListener("click", () => {
      document.querySelectorAll(".tab-btn").forEach((b) => b.classList.remove("active"));
      document.querySelectorAll(".tab-panel").forEach((p) => p.classList.remove("active"));
      btn.classList.add("active");
      document.getElementById(`tab-${btn.dataset.tab}`).classList.add("active");
    });
  });
}

function renderKPIs(t) {
  const total = t.total_items || 0;
  const passed = (t.approved_count || 0) + (t.qa_resolved_pass_count || 0);
  const quarantined = (t.quarantined_count || 0) + (t.qa_resolved_reject_count || 0);
  const passRate = total > 0 ? ((passed / total) * 100).toFixed(1) : "0.0";

  document.getElementById("kpi-total").textContent = total;
  document.getElementById("kpi-batches").textContent = `${t.total_batches} batch(es) logged`;
  document.getElementById("kpi-approved").textContent = passed;
  document.getElementById("kpi-pass-rate").textContent = `${passRate}% pass rate`;
  document.getElementById("kpi-quarantined").textContent = quarantined;
  document.getElementById("kpi-webhooks").textContent = `${t.webhook_success_count}/${t.webhook_total_count} signed webhooks 200 OK`;
  document.getElementById("kpi-qa").textContent = t.pending_qa_count;
  document.getElementById("qa-tab-badge").textContent = t.pending_qa_count;
  document.getElementById("kpi-latency").textContent = `${t.avg_total_latency_ms.toFixed(1)} ms`;
  document.getElementById("kpi-latency-split").textContent = `CV ${t.avg_cv_latency_ms.toFixed(1)} ms · ONNX ${t.avg_dl_latency_ms.toFixed(1)} ms · P95 ${t.p95_total_latency_ms.toFixed(1)} ms`;

  if (t.model_info) {
    document.getElementById("runtime-pill").textContent = `${t.model_info.runtime} · ${t.model_info.classifier_quantization}`;
  }
}

function renderItemsList(items) {
  const container = document.getElementById("items-list");
  if (!items.length) {
    container.innerHTML = `<div class="empty-state">No inspection items found.</div>`;
    return;
  }
  container.innerHTML = items
    .map(
      (item) => `
    <div class="item-row ${item.item_id === state.selectedItemId ? "selected" : ""}" data-id="${item.item_id}">
      <div class="item-row-top">
        <span class="item-serial">${item.serial_number}</span>
        <span class="badge badge-${item.state}">${item.state}</span>
      </div>
      <div class="item-meta">
        <span>${item.surface_type.toUpperCase()} · ${item.predicted_severity}</span>
        <span>${(item.confidence * 100).toFixed(1)}% · ${item.defect_area_mm2.toFixed(1)} mm² · ${item.total_latency_ms.toFixed(1)} ms</span>
      </div>
    </div>
  `
    )
    .join("");

  container.querySelectorAll(".item-row").forEach((el) => {
    el.addEventListener("click", () => {
      state.selectedItemId = el.dataset.id;
      renderItemsList(state.items);
      renderSelectedInspector();
    });
  });
}

function renderSelectedInspector() {
  const item = state.items.find((i) => i.item_id === state.selectedItemId) || state.items[0];
  const imgEl = document.getElementById("inspector-image");
  const emptyEl = document.getElementById("inspector-empty");
  const metGrid = document.getElementById("metrology-grid");
  const tbody = document.getElementById("detections-tbody");

  if (!item) {
    imgEl.classList.add("hidden");
    emptyEl.classList.remove("hidden");
    metGrid.innerHTML = "";
    tbody.innerHTML = "";
    return;
  }

  state.selectedItemId = item.item_id;
  document.getElementById("selected-title").textContent = `${item.serial_number} (${item.item_id})`;
  document.getElementById("selected-subtitle").textContent = `Surface: ${item.surface_type} | State: ${item.state} | Severity: ${item.predicted_severity}`;

  const url = state.viewMode === "overlay" ? item.annotated_image_url : item.raw_image_url;
  if (url) {
    imgEl.src = `${url}?t=${Date.now()}`;
    imgEl.classList.remove("hidden");
    emptyEl.classList.add("hidden");
  }

  metGrid.innerHTML = `
    <div class="met-cell"><span>Confidence</span><strong>${(item.confidence * 100).toFixed(1)}%</strong></div>
    <div class="met-cell"><span>Defect Area</span><strong>${item.defect_area_mm2.toFixed(2)} mm²</strong></div>
    <div class="met-cell"><span>Calibration</span><strong>${item.px_per_mm.toFixed(2)} px/mm</strong></div>
    <div class="met-cell"><span>ROI Skew</span><strong>${item.alignment_angle_deg >= 0 ? "+" : ""}${item.alignment_angle_deg.toFixed(2)}°</strong></div>
    <div class="met-cell"><span>CV Time</span><strong>${item.cv_latency_ms.toFixed(2)} ms</strong></div>
    <div class="met-cell"><span>ONNX Time</span><strong>${item.dl_latency_ms.toFixed(2)} ms</strong></div>
  `;

  if (!item.detections || !item.detections.length) {
    tbody.innerHTML = `<tr><td colspan="7" class="muted">No defect regions localized (Clean part within tolerance).</td></tr>`;
  } else {
    tbody.innerHTML = item.detections
      .map(
        (d) => `
      <tr>
        <td><strong>${d.defect_class}</strong></td>
        <td>${d.severity}</td>
        <td class="mono">${(d.confidence * 100).toFixed(1)}%</td>
        <td class="mono">${d.area_mm2.toFixed(2)}</td>
        <td class="mono">${d.width_mm.toFixed(1)} &times; ${d.height_mm.toFixed(1)}</td>
        <td class="mono">[${d.bbox_x1},${d.bbox_y1},${d.bbox_x2},${d.bbox_y2}]</td>
        <td class="mono">${d.source}</td>
      </tr>
    `
      )
      .join("");
  }
}

function renderAutomation(rules, events, webhooks, erpAlerts) {
  document.getElementById("sla-rules-list").innerHTML = rules
    .map(
      (r) => `
    <div class="rule-card">
      <div class="rule-card-head">
        <span>${r.rule_id} · ${r.name}</span>
        <span class="badge badge-APPROVED">${r.action}</span>
      </div>
      <div class="rule-cond">${r.condition}</div>
    </div>
  `
    )
    .join("");

  document.getElementById("erp-alerts-list").innerHTML = erpAlerts.length
    ? erpAlerts
        .map(
          (a) => `
      <div class="erp-card">
        <div class="rule-card-head">
          <span class="mono">${a.item_id} (${a.severity})</span>
          <span class="badge ${a.signature_verified ? "badge-APPROVED" : "badge-QUARANTINED"}">${
            a.signature_verified ? "HMAC VERIFIED" : "UNVERIFIED"
          }</span>
        </div>
        <div class="muted">Area: ${a.defect_area_mm2.toFixed(2)} mm² · Received: ${a.received_at.slice(11, 19)} UTC</div>
      </div>
    `
        )
        .join("")
    : `<div class="empty-state">No ERP quarantine alerts received yet.</div>`;

  document.getElementById("webhooks-tbody").innerHTML = webhooks
    .map(
      (w) => `
    <tr>
      <td class="mono">${w.delivered_at.slice(11, 19)}</td>
      <td class="mono">${w.item_id}</td>
      <td class="mono">${w.target_url}</td>
      <td><span class="badge ${w.http_status === 200 ? "badge-APPROVED" : "badge-QUARANTINED"}">${w.http_status}</span></td>
      <td class="mono">${w.latency_ms.toFixed(2)} ms</td>
      <td class="mono">${w.hmac_signature.slice(0, 26)}...</td>
      <td class="mono">${w.response_body.slice(0, 56)}</td>
    </tr>
  `
    )
    .join("");

  document.getElementById("events-tbody").innerHTML = events
    .slice(0, 60)
    .map(
      (e) => `
    <tr>
      <td class="mono">${e.created_at.slice(11, 19)}</td>
      <td class="mono">${e.rule_id}</td>
      <td><strong>${e.action}</strong></td>
      <td class="mono">${e.item_id}</td>
      <td>${e.reason}</td>
    </tr>
  `
    )
    .join("");
}

function renderQAQueue(queueItems) {
  const grid = document.getElementById("qa-queue-grid");
  if (!queueItems.length) {
    grid.innerHTML = `<div class="empty-state">QA Review Queue is clear — zero low-confidence items awaiting review.</div>`;
    return;
  }
  grid.innerHTML = queueItems
    .map(
      (item) => `
    <div class="qa-card">
      <div class="item-row-top">
        <span class="item-serial">${item.serial_number} (${item.surface_type})</span>
        <span class="badge badge-PENDING_QA">CONF ${(item.confidence * 100).toFixed(1)}%</span>
      </div>
      <img src="${item.annotated_image_url}?t=${Date.now()}" alt="${item.item_id}" />
      <div class="muted">Predicted: <strong>${item.predicted_severity}</strong> · Defect Area: <strong>${item.defect_area_mm2.toFixed(2)} mm²</strong> · Skew: ${item.alignment_angle_deg.toFixed(2)}°</div>
      <input type="text" id="notes-${item.item_id}" class="text-input" placeholder="Operator QA notes..." value="Inspected optical overlay" />
      <div class="qa-actions">
        <button class="btn btn-approve" onclick="resolveQA('${item.item_id}', 'PASS')">Approve (PASS)</button>
        <button class="btn btn-reject" onclick="resolveQA('${item.item_id}', 'REJECT')">Reject &amp; Quarantine</button>
      </div>
    </div>
  `
    )
    .join("");
}

window.resolveQA = async (itemId, decision) => {
  const reviewer = document.getElementById("qa-reviewer-input").value || "qa_lead_operator";
  const notes = document.getElementById(`notes-${itemId}`)?.value || "Reviewed in console";
  try {
    await apiJson(`/api/qa-queue/${itemId}/resolve`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ decision, reviewer, notes }),
    });
    showToast(`Item ${itemId} resolved as ${decision}`);
    await loadAll();
  } catch (err) {
    showToast(`QA Error: ${err.message}`);
  }
};

function renderTelemetryCharts(t, batches) {
  const latEl = document.getElementById("latency-chart");
  const series = t.latency_series || [];
  if (!series.length) {
    latEl.innerHTML = `<div class="empty-state">Run a batch to populate latency telemetry.</div>`;
  } else {
    const maxLat = Math.max(...series.map((s) => s.total_latency_ms), 10);
    const W = 580, H = 220, padL = 36, padB = 26;
    const barW = Math.max(4, Math.floor((W - padL - 12) / series.length) - 2);
    const bars = series
      .map((s, i) => {
        const x = padL + i * (barW + 2);
        const cvH = ((s.cv_latency_ms / maxLat) * (H - padB - 14));
        const dlH = ((s.dl_latency_ms / maxLat) * (H - padB - 14));
        const yDl = H - padB - dlH;
        const yCv = yDl - cvH;
        return `
          <rect x="${x}" y="${yDl.toFixed(1)}" width="${barW}" height="${dlH.toFixed(1)}" fill="#38BDF8" rx="1">
            <title>${s.serial_number}: ONNX ${s.dl_latency_ms.toFixed(1)}ms + CV ${s.cv_latency_ms.toFixed(1)}ms</title>
          </rect>
          <rect x="${x}" y="${yCv.toFixed(1)}" width="${barW}" height="${cvH.toFixed(1)}" fill="#818CF8" rx="1" />
        `;
      })
      .join("");
    latEl.innerHTML = `
      <svg viewBox="0 0 ${W} ${H}" width="100%" height="230">
        <line x1="${padL}" y1="10" x2="${padL}" y2="${H - padB}" stroke="#334155" />
        <line x1="${padL}" y1="${H - padB}" x2="${W - 8}" y2="${H - padB}" stroke="#334155" />
        <text x="4" y="20" fill="#94A3B8" font-size="10">${maxLat.toFixed(0)}ms</text>
        <text x="8" y="${H - padB}" fill="#94A3B8" font-size="10">0ms</text>
        ${bars}
        <rect x="${padL + 10}" y="${H - 14}" width="10" height="8" fill="#818CF8" />
        <text x="${padL + 24}" y="${H - 7}" fill="#CBD5E1" font-size="10">OpenCV Metrology</text>
        <rect x="${padL + 145}" y="${H - 14}" width="10" height="8" fill="#38BDF8" />
        <text x="${padL + 159}" y="${H - 7}" fill="#CBD5E1" font-size="10">ONNX Runtime (YOLOv8 + MobileNetV3-INT8)</text>
      </svg>
    `;
  }

  const distEl = document.getElementById("distribution-chart");
  const sev = t.severity_breakdown || {};
  const cls = t.defect_class_breakdown || {};
  const totalItems = Math.max(1, t.total_items || 1);
  const sevEntries = [
    ["PASS", sev.PASS || 0, "#10B981"],
    ["MINOR_COSMETIC", sev.MINOR_COSMETIC || 0, "#F59E0B"],
    ["CRITICAL_REJECT", sev.CRITICAL_REJECT || 0, "#EF4444"],
  ];
  const clsEntries = Object.entries(cls);
  const maxCls = Math.max(1, ...clsEntries.map(([, v]) => v));

  distEl.innerHTML = `
    <div style="display:flex;flex-direction:column;gap:10px;">
      <div class="muted" style="font-weight:600;">Severity Breakdown (${t.total_items} items)</div>
      ${sevEntries
        .map(
          ([k, v, col]) => `
        <div>
          <div style="display:flex;justify-content:space-between;font-size:11px;margin-bottom:3px;">
            <span>${k}</span><span class="mono">${v} (${((v / totalItems) * 100).toFixed(1)}%)</span>
          </div>
          <div style="background:#1E293B;height:9px;border-radius:4px;overflow:hidden;">
            <div style="width:${(v / totalItems) * 100}%;background:${col};height:100%;"></div>
          </div>
        </div>
      `
        )
        .join("")}
      <div class="muted" style="font-weight:600;margin-top:8px;">Localized Defect Classes</div>
      ${
        clsEntries.length
          ? clsEntries
              .map(
                ([k, v]) => `
        <div>
          <div style="display:flex;justify-content:space-between;font-size:11px;margin-bottom:3px;">
            <span>${k.toUpperCase()}</span><span class="mono">${v}</span>
          </div>
          <div style="background:#1E293B;height:8px;border-radius:4px;overflow:hidden;">
            <div style="width:${(v / maxCls) * 100}%;background:#38BDF8;height:100%;"></div>
          </div>
        </div>
      `
              )
              .join("")
          : `<div class="muted">No localized defects recorded yet.</div>`
      }
    </div>
  `;

  document.getElementById("batches-tbody").innerHTML = batches
    .map(
      (b) => `
    <tr>
      <td class="mono">${b.batch_id}</td>
      <td><strong>${b.batch_name}</strong></td>
      <td>${b.product_line}</td>
      <td class="mono">${b.total_items}</td>
      <td class="mono" style="color:#6EE7B7">${b.passed_count}</td>
      <td class="mono" style="color:#FCA5A5">${b.quarantined_count}</td>
      <td class="mono" style="color:#FCD34D">${b.pending_qa_count}</td>
      <td class="mono">${b.avg_latency_ms.toFixed(2)} ms</td>
      <td>${
        b.audit_report_path
          ? `<a href="/api/batches/${b.batch_id}/report" target="_blank" style="color:#38BDF8;">Download JSON</a>`
          : `<span class="muted">Pending</span>`
      }</td>
    </tr>
  `
    )
    .join("");
}

async function loadAll() {
  const filterState = document.getElementById("state-filter").value;
  const itemsUrl = filterState ? `/api/items?state=${encodeURIComponent(filterState)}&limit=100` : "/api/items?limit=100";

  const [telemetry, items, rules, events, webhooks, erpAlerts, qaQueue, batches] = await Promise.all([
    apiJson("/api/telemetry"),
    apiJson(itemsUrl),
    apiJson("/api/sla-rules"),
    apiJson("/api/automation-events?limit=80"),
    apiJson("/api/webhooks/deliveries?limit=80"),
    apiJson("/api/webhooks/erp-alerts?limit=50"),
    apiJson("/api/qa-queue?limit=80"),
    apiJson("/api/batches?limit=20"),
  ]);

  state.telemetry = telemetry;
  state.items = items;

  renderKPIs(telemetry);
  renderItemsList(items);
  renderSelectedInspector();
  renderAutomation(rules, events, webhooks, erpAlerts);
  renderQAQueue(qaQueue);
  renderTelemetryCharts(telemetry, batches);
}

document.addEventListener("DOMContentLoaded", () => {
  setupTabs();

  document.getElementById("state-filter").addEventListener("change", loadAll);
  document.getElementById("refresh-btn").addEventListener("click", loadAll);

  document.getElementById("view-overlay-btn").addEventListener("click", (e) => {
    state.viewMode = "overlay";
    e.target.classList.add("active");
    document.getElementById("view-raw-btn").classList.remove("active");
    renderSelectedInspector();
  });
  document.getElementById("view-raw-btn").addEventListener("click", (e) => {
    state.viewMode = "raw";
    e.target.classList.add("active");
    document.getElementById("view-overlay-btn").classList.remove("active");
    renderSelectedInspector();
  });

  document.getElementById("run-batch-btn").addEventListener("click", async () => {
    const btn = document.getElementById("run-batch-btn");
    btn.disabled = true;
    btn.textContent = "Running 50-Item Batch...";
    try {
      const res = await apiJson("/api/batches/run", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          batch_name: `Production-Batch-${new Date().toISOString().slice(11, 19)}`,
          product_line: "mixed",
          count: 50,
          seed: Math.floor(Math.random() * 9000) + 100,
        }),
      });
      showToast(`Batch complete: ${res.items_processed} items processed`);
      await loadAll();
    } catch (err) {
      showToast(`Batch failed: ${err.message}`);
    } finally {
      btn.disabled = false;
      btn.textContent = "Run 50-Item Batch";
    }
  });

  document.getElementById("upload-input").addEventListener("change", async (e) => {
    const file = e.target.files?.[0];
    if (!file) return;
    const form = new FormData();
    form.append("file", file);
    form.append("surface_type", "metallic");
    try {
      const item = await apiJson("/api/inspect", { method: "POST", body: form });
      state.selectedItemId = item.item_id;
      showToast(`Inspected ${item.serial_number}: ${item.state}`);
      await loadAll();
    } catch (err) {
      showToast(`Upload failed: ${err.message}`);
    } finally {
      e.target.value = "";
    }
  });

  loadAll().catch((err) => showToast(`Load error: ${err.message}`));
});
