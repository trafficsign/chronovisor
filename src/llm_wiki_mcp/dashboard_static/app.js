const els = {
  statePill: document.getElementById("state-pill"),
  stateText: document.getElementById("state-text"),
  pending: document.getElementById("pending-value"),
  pendingSub: document.getElementById("pending-sub"),
  stage: document.getElementById("stage-value"),
  raw: document.getElementById("raw-value"),
  batch: document.getElementById("batch-value"),
  batchSub: document.getElementById("batch-sub"),
  ollama: document.getElementById("ollama-value"),
  ollamaSub: document.getElementById("ollama-sub"),
  currentRaw: document.getElementById("current-raw"),
  currentOp: document.getElementById("current-op"),
  currentJob: document.getElementById("current-job"),
  lastSuccess: document.getElementById("last-success"),
  trendCaption: document.getElementById("trend-caption"),
  eventFeed: document.getElementById("event-feed"),
  pendingChart: document.getElementById("pending-chart"),
  batchChart: document.getElementById("batch-chart"),
};

function fmt(value, fallback = "--") {
  if (value === null || value === undefined || value === "") return fallback;
  return String(value);
}

function shortName(value) {
  const text = fmt(value);
  if (text.length <= 64) return text;
  return `${text.slice(0, 30)}...${text.slice(-28)}`;
}

function timeLabel(value) {
  if (!value) return "--";
  const date = new Date(value);
  if (!Number.isNaN(date.getTime())) {
    return date.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit", second: "2-digit" });
  }
  return String(value).slice(-8);
}

function setState(state) {
  const normalized = (state || "unknown").toLowerCase();
  els.statePill.classList.remove("running", "error");
  if (normalized === "running") els.statePill.classList.add("running");
  if (normalized === "error" || normalized === "blocked") els.statePill.classList.add("error");
  els.stateText.textContent = normalized;
}

function drawLineChart(canvas, rows) {
  const ctx = canvas.getContext("2d");
  const width = canvas.clientWidth || 640;
  const height = Number(canvas.dataset.baseHeight || canvas.getAttribute("height") || 260);
  canvas.dataset.baseHeight = String(height);
  canvas.style.height = `${height}px`;
  const dpr = window.devicePixelRatio || 1;
  canvas.width = Math.floor(width * dpr);
  canvas.height = Math.floor(height * dpr);
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, width, height);

  const pad = 28;
  const values = rows
    .map((row) => row.pending_after ?? row.pending)
    .filter((value) => typeof value === "number");

  ctx.fillStyle = "rgba(242,239,229,0.08)";
  ctx.fillRect(0, 0, width, height);
  ctx.strokeStyle = "rgba(242,239,229,0.12)";
  ctx.lineWidth = 1;
  for (let i = 0; i < 5; i += 1) {
    const y = pad + ((height - pad * 2) / 4) * i;
    ctx.beginPath();
    ctx.moveTo(pad, y);
    ctx.lineTo(width - pad, y);
    ctx.stroke();
  }

  if (values.length < 2) {
    ctx.fillStyle = "rgba(169,164,148,0.85)";
    ctx.font = "14px system-ui";
    ctx.fillText("Waiting for batch history", pad, height / 2);
    return;
  }

  const min = Math.min(...values);
  const max = Math.max(...values);
  const span = Math.max(1, max - min);
  const points = values.map((value, index) => {
    const x = pad + ((width - pad * 2) * index) / Math.max(1, values.length - 1);
    const y = height - pad - ((value - min) / span) * (height - pad * 2);
    return [x, y, value];
  });

  const gradient = ctx.createLinearGradient(0, pad, 0, height - pad);
  gradient.addColorStop(0, "rgba(102,217,232,0.34)");
  gradient.addColorStop(1, "rgba(102,217,232,0.02)");

  ctx.beginPath();
  points.forEach(([x, y], index) => {
    if (index === 0) ctx.moveTo(x, y);
    else ctx.lineTo(x, y);
  });
  ctx.lineTo(points[points.length - 1][0], height - pad);
  ctx.lineTo(points[0][0], height - pad);
  ctx.closePath();
  ctx.fillStyle = gradient;
  ctx.fill();

  ctx.beginPath();
  points.forEach(([x, y], index) => {
    if (index === 0) ctx.moveTo(x, y);
    else ctx.lineTo(x, y);
  });
  ctx.strokeStyle = "#66d9e8";
  ctx.lineWidth = 3;
  ctx.stroke();

  points.forEach(([x, y]) => {
    ctx.beginPath();
    ctx.arc(x, y, 4, 0, Math.PI * 2);
    ctx.fillStyle = "#f2efe5";
    ctx.fill();
  });

  ctx.fillStyle = "rgba(242,239,229,0.78)";
  ctx.font = "12px system-ui";
  ctx.fillText(`max ${max}`, pad, pad - 8);
  ctx.fillText(`min ${min}`, pad, height - 8);
}

function drawBatchChart(canvas, rows) {
  const ctx = canvas.getContext("2d");
  const width = canvas.clientWidth || 420;
  const height = Number(canvas.dataset.baseHeight || canvas.getAttribute("height") || 260);
  canvas.dataset.baseHeight = String(height);
  canvas.style.height = `${height}px`;
  const dpr = window.devicePixelRatio || 1;
  canvas.width = Math.floor(width * dpr);
  canvas.height = Math.floor(height * dpr);
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, width, height);
  ctx.fillStyle = "rgba(242,239,229,0.08)";
  ctx.fillRect(0, 0, width, height);

  const data = rows
    .filter((row) => typeof row.files_processed === "number" || typeof row.files_failed === "number")
    .slice(-18);
  if (!data.length) {
    ctx.fillStyle = "rgba(169,164,148,0.85)";
    ctx.font = "14px system-ui";
    ctx.fillText("No batch yield yet", 24, height / 2);
    return;
  }
  const pad = 24;
  const gap = 8;
  const barWidth = Math.max(8, (width - pad * 2 - gap * (data.length - 1)) / data.length);
  const maxTotal = Math.max(1, ...data.map((row) => (row.files_processed || 0) + (row.files_failed || 0)));
  data.forEach((row, index) => {
    const processed = row.files_processed || 0;
    const failed = row.files_failed ?? Math.max(0, (row.files_attempted || 0) - processed);
    const total = Math.max(1, processed + failed);
    const x = pad + index * (barWidth + gap);
    const totalHeight = ((height - pad * 2) * total) / maxTotal;
    const goodHeight = totalHeight * (processed / total);
    const badHeight = totalHeight - goodHeight;
    const y = height - pad - totalHeight;
    ctx.fillStyle = "#8fd694";
    ctx.fillRect(x, y + badHeight, barWidth, goodHeight);
    ctx.fillStyle = failed ? "#f0bc62" : "rgba(102,217,232,0.38)";
    ctx.fillRect(x, y, barWidth, badHeight || 2);
  });
}

function updateStageFlow(stage) {
  document.querySelectorAll(".stage-node").forEach((node) => {
    node.classList.toggle("active", node.dataset.stage === stage);
  });
}

function renderEvents(events) {
  const recent = [...events].slice(-18).reverse();
  els.eventFeed.innerHTML = "";
  if (!recent.length) {
    els.eventFeed.innerHTML = "<div class=\"event-message\">No events yet.</div>";
    return;
  }
  recent.forEach((event) => {
    const row = document.createElement("div");
    row.className = "event";
    const level = (event.level || "info").toLowerCase();
    const time = document.createElement("time");
    time.textContent = timeLabel(event.timestamp);
    const badge = document.createElement("span");
    badge.className = `event-level ${level}`;
    badge.textContent = level;
    const message = document.createElement("span");
    message.className = "event-message";
    message.textContent = fmt(event.message);
    row.append(time, badge, message);
    els.eventFeed.appendChild(row);
  });
}

function render(snapshot) {
  const status = snapshot.status || {};
  const metrics = snapshot.metrics || [];
  const batch = status.batch || {};
  const ollama = snapshot.ollama || {};
  const models = ollama.models || [];
  const model = models.find((item) => !String(item.name || item.model || "").includes("embed")) || models[0] || {};

  setState(status.state);
  els.pending.textContent = fmt(status.pending);
  els.pendingSub.textContent = `updated ${timeLabel(status.updated_at)}`;
  els.stage.textContent = fmt(status.stage);
  els.raw.textContent = shortName(status.current_raw || "no active raw");
  els.batch.textContent = batch.total ? `${batch.index || 0}/${batch.total}` : "--";
  els.batchSub.textContent = batch.total ? `${batch.succeeded || 0} ok / ${batch.failed || 0} miss` : "waiting";
  els.ollama.textContent = ollama.available ? "online" : "offline";
  els.ollamaSub.textContent = model.name || model.model || "no model";
  els.currentRaw.textContent = fmt(status.current_raw);
  els.currentOp.textContent = fmt(status.current_op);
  els.currentJob.textContent = fmt(status.current_job_id);
  els.lastSuccess.textContent = status.last_success ? `${fmt(status.last_success.raw)} -> ${[...(status.last_success.created || []), ...(status.last_success.updated || [])].join(", ") || "none"}` : "--";
  els.trendCaption.textContent = metrics.length ? `${metrics.length} points` : "waiting for data";
  updateStageFlow(status.stage);
  renderEvents(snapshot.events || []);
  drawLineChart(els.pendingChart, metrics);
  drawBatchChart(els.batchChart, metrics);
}

async function refresh() {
  try {
    const response = await fetch("/api/snapshot", { cache: "no-store" });
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    render(await response.json());
  } catch (error) {
    setState("error");
    els.stateText.textContent = "disconnected";
    els.eventFeed.textContent = "";
    const message = document.createElement("div");
    message.className = "event-message";
    message.textContent = `Dashboard fetch failed: ${error.message}`;
    els.eventFeed.appendChild(message);
  }
}

refresh();
setInterval(refresh, 2000);
window.addEventListener("resize", refresh);
