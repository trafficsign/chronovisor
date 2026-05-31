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
  llmSignal: document.getElementById("llm-signal"),
  llmState: document.getElementById("llm-state"),
  llmAge: document.getElementById("llm-age"),
  llmTarget: document.getElementById("llm-target"),
  llmStats: document.getElementById("llm-stats"),
  llmSparkline: document.getElementById("llm-sparkline"),
  currentJob: document.getElementById("current-job"),
  lastSuccess: document.getElementById("last-success"),
  trendCaption: document.getElementById("trend-caption"),
  pendingCurrent: document.getElementById("pending-current"),
  pendingDelta: document.getElementById("pending-delta"),
  pendingRate: document.getElementById("pending-rate"),
  pendingEta: document.getElementById("pending-eta"),
  batchCaption: document.getElementById("batch-caption"),
  batchOk: document.getElementById("batch-ok"),
  batchFailed: document.getElementById("batch-failed"),
  batchDuration: document.getElementById("batch-duration"),
  eventFeed: document.getElementById("event-feed"),
  pendingChart: document.getElementById("pending-chart"),
  batchChart: document.getElementById("batch-chart"),
};

const llmSignalHistory = {
  key: null,
  lastChars: null,
  lastSeenMs: null,
  rates: Array(32).fill(0),
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

function compactDuration(seconds) {
  if (!Number.isFinite(seconds) || seconds <= 0) return "--";
  if (seconds < 90) return `${Math.round(seconds)}s`;
  const minutes = seconds / 60;
  if (minutes < 90) return `${Math.round(minutes)}m`;
  const hours = minutes / 60;
  if (hours < 48) return `${hours.toFixed(hours < 10 ? 1 : 0)}h`;
  return `${Math.round(hours / 24)}d`;
}

function preciseDuration(seconds) {
  if (!Number.isFinite(seconds) || seconds < 0) return "--";
  if (seconds < 90) return `${Math.floor(seconds)}s`;
  return compactDuration(seconds);
}

function parseMs(value) {
  const ms = new Date(value).getTime();
  return Number.isNaN(ms) ? null : ms;
}

function numeric(value) {
  return typeof value === "number" && Number.isFinite(value);
}

function roundRect(ctx, x, y, width, height, radius) {
  const r = Math.min(radius, width / 2, height / 2);
  ctx.beginPath();
  ctx.moveTo(x + r, y);
  ctx.arcTo(x + width, y, x + width, y + height, r);
  ctx.arcTo(x + width, y + height, x, y + height, r);
  ctx.arcTo(x, y + height, x, y, r);
  ctx.arcTo(x, y, x + width, y, r);
  ctx.closePath();
}

function metricScore(row) {
  let score = 0;
  if (row.kind === "batch") score += 4;
  if (numeric(row.files_failed)) score += 2;
  if (numeric(row.elapsed_seconds)) score += 1;
  return score;
}

function completedRows(rows) {
  const byTransition = new Map();
  rows.forEach((row) => {
    if (!numeric(row.pending_after)) return;
    if (row.kind !== "batch" && row.kind !== "drain_batch") return;
    const before = numeric(row.pending_before) ? row.pending_before : "?";
    const key = `${before}->${row.pending_after}`;
    const previous = byTransition.get(key);
    if (!previous || metricScore(row) > metricScore(previous)) {
      byTransition.set(key, row);
    }
  });
  return [...byTransition.values()].sort((a, b) => (parseMs(a.timestamp) || 0) - (parseMs(b.timestamp) || 0));
}

function latestCurrentMetric(rows) {
  return [...rows].reverse().find((row) => row.kind === "current" && numeric(row.pending_after));
}

function buildTrendPoints(rows) {
  const completed = completedRows(rows);
  const points = [];
  completed.forEach((row, index) => {
    if (index === 0 && numeric(row.pending_before)) {
      points.push({
        timestamp: row.timestamp,
        label: "start",
        value: row.pending_before,
      });
    }
    points.push({
      timestamp: row.timestamp,
      label: row.kind === "batch" ? "batch" : "drain",
      value: row.pending_after,
    });
  });

  const current = latestCurrentMetric(rows);
  if (current && (!points.length || points[points.length - 1].value !== current.pending_after)) {
    points.push({
      timestamp: current.timestamp,
      label: "now",
      value: current.pending_after,
    });
  }
  return points.slice(-24);
}

function niceTicks(min, max, count = 4) {
  if (min === max) {
    return [min - 1, min, min + 1];
  }
  const rawStep = Math.max(1, (max - min) / count);
  const magnitude = 10 ** Math.floor(Math.log10(rawStep));
  const normalized = rawStep / magnitude;
  const step = (normalized <= 1 ? 1 : normalized <= 2 ? 2 : normalized <= 5 ? 5 : 10) * magnitude;
  const start = Math.floor(min / step) * step;
  const end = Math.ceil(max / step) * step;
  const ticks = [];
  for (let value = start; value <= end + step / 2; value += step) {
    ticks.push(Math.round(value * 100) / 100);
  }
  return ticks;
}

function measureTrend(points) {
  if (!points.length) {
    return { current: null, drained: null, rate: null, etaSeconds: null };
  }
  const first = points[0];
  const last = points[points.length - 1];
  const drained = first.value - last.value;
  const firstMs = parseMs(first.timestamp);
  const lastMs = parseMs(last.timestamp);
  const hours = firstMs !== null && lastMs !== null ? Math.max(0, (lastMs - firstMs) / 3_600_000) : 0;
  const rate = drained > 0 && hours > 0 ? drained / hours : null;
  return {
    current: last.value,
    drained,
    rate,
    etaSeconds: rate ? (last.value / rate) * 3600 : null,
  };
}

function setState(state) {
  const normalized = (state || "unknown").toLowerCase();
  els.statePill.classList.remove("running", "error");
  if (normalized === "running") els.statePill.classList.add("running");
  if (normalized === "error" || normalized === "blocked") els.statePill.classList.add("error");
  els.stateText.textContent = normalized;
}

function setLlmSignalClass(kind) {
  els.llmSignal.classList.remove("idle", "prefill", "live", "waiting", "stalled", "complete");
  els.llmSignal.classList.add(kind);
}

function llmSignalKey(llm) {
  return [llm.job_id || "", llm.phase || "", llm.target || ""].join("|");
}

function updateLlmRates(llm, nowMs) {
  const key = llmSignalKey(llm);
  const chars = Number(llm.generated_chars || 0);
  if (llmSignalHistory.key !== key) {
    llmSignalHistory.key = key;
    llmSignalHistory.lastChars = chars;
    llmSignalHistory.lastSeenMs = nowMs;
    llmSignalHistory.rates = Array(32).fill(0);
    return 0;
  }

  const elapsed = Math.max(0.001, (nowMs - (llmSignalHistory.lastSeenMs || nowMs)) / 1000);
  const delta = Math.max(0, chars - (llmSignalHistory.lastChars || 0));
  const rate = delta / elapsed;
  llmSignalHistory.lastChars = chars;
  llmSignalHistory.lastSeenMs = nowMs;
  llmSignalHistory.rates.push(rate);
  llmSignalHistory.rates = llmSignalHistory.rates.slice(-32);
  return rate;
}

function renderSparkline(kind) {
  const rates = llmSignalHistory.rates;
  const max = Math.max(1, ...rates);
  els.llmSparkline.innerHTML = "";
  rates.forEach((rate) => {
    const bar = document.createElement("span");
    const height = 3 + (rate / max) * 31;
    bar.style.height = `${Math.max(3, Math.min(34, height))}px`;
    bar.style.opacity = rate > 0 ? "1" : kind === "idle" ? "0.18" : "0.34";
    els.llmSparkline.appendChild(bar);
  });
}

function renderLlm(llm) {
  if (!llm) {
    setLlmSignalClass("idle");
    llmSignalHistory.key = null;
    llmSignalHistory.lastChars = null;
    llmSignalHistory.lastSeenMs = null;
    llmSignalHistory.rates = Array(32).fill(0);
    els.llmState.textContent = "idle";
    els.llmAge.textContent = "--";
    els.llmTarget.textContent = "--";
    els.llmStats.textContent = "--";
    renderSparkline("idle");
    return;
  }

  const nowMs = Date.now();
  const updatedMs = parseMs(llm.updated_at) ?? parseMs(llm.started_at) ?? nowMs;
  const startedMs = parseMs(llm.started_at);
  const active = Boolean(llm.active);
  const elapsed = active && startedMs !== null
    ? Math.max(0, (nowMs - startedMs) / 1000)
    : Number(llm.elapsed_seconds || 0);
  const chars = Number(llm.generated_chars || 0);
  const instantRate = updateLlmRates(llm, nowMs);
  const avgRate = elapsed > 0 ? chars / elapsed : Number(llm.chars_per_second || 0);
  const signalAge = Math.max(0, (nowMs - updatedMs) / 1000);
  const finalTokens = Number(llm.eval_count || 0);
  const phase = fmt(llm.phase || llm.event || "llm");

  let kind = "complete";
  let stateLabel = phase;
  if (active && chars === 0) {
    kind = "prefill";
    stateLabel = `${phase} prefill`;
  } else if (active && signalAge <= 5) {
    kind = "live";
    stateLabel = `${phase} live`;
  } else if (active && signalAge <= 15) {
    kind = "waiting";
    stateLabel = `${phase} waiting`;
  } else if (active) {
    kind = "stalled";
    stateLabel = `${phase} stalled`;
  }

  setLlmSignalClass(kind);
  els.llmState.textContent = stateLabel;
  els.llmAge.textContent = chars === 0
    ? `first token ${preciseDuration(elapsed)}`
    : `last chunk ${preciseDuration(signalAge)} ago`;
  els.llmTarget.textContent = fmt(llm.target || llm.raw || "operation");
  const parts = [
    `${chars.toLocaleString()} chars`,
    preciseDuration(elapsed),
  ];
  if (instantRate > 0) parts.push(`${instantRate.toFixed(instantRate < 10 ? 1 : 0)} c/s now`);
  if (avgRate > 0) parts.push(`${avgRate.toFixed(avgRate < 10 ? 1 : 0)} c/s avg`);
  if (finalTokens > 0) parts.push(`${finalTokens.toLocaleString()} tok`);
  els.llmStats.textContent = parts.join(" · ");
  renderSparkline(kind);
}

function drawLineChart(canvas, rows) {
  const ctx = canvas.getContext("2d");
  const width = canvas.clientWidth || 640;
  const height = Number(canvas.dataset.baseHeight || canvas.getAttribute("height") || 300);
  canvas.dataset.baseHeight = String(height);
  canvas.style.height = `${height}px`;
  const dpr = window.devicePixelRatio || 1;
  canvas.width = Math.floor(width * dpr);
  canvas.height = Math.floor(height * dpr);
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, width, height);

  const pointsRaw = buildTrendPoints(rows);
  const measure = measureTrend(pointsRaw);
  els.pendingCurrent.textContent = measure.current === null ? "--" : String(measure.current);
  els.pendingDelta.textContent = measure.drained === null ? "--" : measure.drained >= 0 ? String(measure.drained) : `+${Math.abs(measure.drained)}`;
  els.pendingRate.textContent = measure.rate ? `${measure.rate.toFixed(measure.rate < 10 ? 1 : 0)}/h` : "--";
  els.pendingEta.textContent = compactDuration(measure.etaSeconds);

  const pad = { top: 24, right: 28, bottom: 42, left: 48 };
  const plotWidth = width - pad.left - pad.right;
  const plotHeight = height - pad.top - pad.bottom;

  if (pointsRaw.length < 2) {
    ctx.fillStyle = "rgba(169,164,148,0.85)";
    ctx.font = "14px system-ui";
    ctx.fillText("Waiting for completed batches", pad.left, height / 2);
    return;
  }

  const values = pointsRaw.map((point) => point.value);
  const ticks = niceTicks(Math.min(...values), Math.max(...values), 4);
  const yMin = ticks[0];
  const yMax = ticks[ticks.length - 1];
  const ySpan = Math.max(1, yMax - yMin);
  const points = pointsRaw.map((point, index) => {
    const x = pad.left + (plotWidth * index) / Math.max(1, pointsRaw.length - 1);
    const y = pad.top + plotHeight - ((point.value - yMin) / ySpan) * plotHeight;
    return { ...point, x, y };
  });

  ctx.save();
  ctx.font = "11px system-ui";
  ctx.textBaseline = "middle";
  ticks.forEach((tick) => {
    const y = pad.top + plotHeight - ((tick - yMin) / ySpan) * plotHeight;
    ctx.strokeStyle = tick === yMin ? "rgba(242,239,229,0.2)" : "rgba(242,239,229,0.1)";
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(pad.left, y);
    ctx.lineTo(width - pad.right, y);
    ctx.stroke();
    ctx.fillStyle = "rgba(169,164,148,0.9)";
    ctx.textAlign = "right";
    ctx.fillText(String(tick), pad.left - 10, y);
  });

  ctx.textBaseline = "alphabetic";
  ctx.textAlign = "left";
  ctx.fillStyle = "rgba(169,164,148,0.85)";
  ctx.fillText(timeLabel(points[0].timestamp), pad.left, height - 16);
  ctx.textAlign = "right";
  ctx.fillText(timeLabel(points[points.length - 1].timestamp), width - pad.right, height - 16);

  const latest = points[points.length - 1];
  ctx.strokeStyle = "rgba(102,217,232,0.28)";
  ctx.setLineDash([5, 5]);
  ctx.beginPath();
  ctx.moveTo(pad.left, latest.y);
  ctx.lineTo(width - pad.right, latest.y);
  ctx.stroke();
  ctx.setLineDash([]);

  const gradient = ctx.createLinearGradient(0, pad.top, 0, height - pad.bottom);
  gradient.addColorStop(0, "rgba(102,217,232,0.34)");
  gradient.addColorStop(0.62, "rgba(102,217,232,0.1)");
  gradient.addColorStop(1, "rgba(102,217,232,0)");

  ctx.beginPath();
  ctx.moveTo(points[0].x, points[0].y);
  for (let i = 1; i < points.length; i += 1) {
    ctx.lineTo(points[i].x, points[i - 1].y);
    ctx.lineTo(points[i].x, points[i].y);
  }
  ctx.lineTo(latest.x, pad.top + plotHeight);
  ctx.lineTo(points[0].x, pad.top + plotHeight);
  ctx.closePath();
  ctx.fillStyle = gradient;
  ctx.fill();

  ctx.beginPath();
  ctx.moveTo(points[0].x, points[0].y);
  for (let i = 1; i < points.length; i += 1) {
    ctx.lineTo(points[i].x, points[i - 1].y);
    ctx.lineTo(points[i].x, points[i].y);
  }
  ctx.strokeStyle = "#66d9e8";
  ctx.lineWidth = 3;
  ctx.lineJoin = "round";
  ctx.stroke();

  points.forEach((point, index) => {
    ctx.beginPath();
    ctx.arc(point.x, point.y, index === points.length - 1 ? 5 : 3.5, 0, Math.PI * 2);
    ctx.fillStyle = index === points.length - 1 ? "#66d9e8" : "#f2efe5";
    ctx.fill();
  });

  const bubble = `${latest.value} now`;
  ctx.font = "12px system-ui";
  const bubbleWidth = ctx.measureText(bubble).width + 18;
  const bubbleX = Math.min(width - pad.right - bubbleWidth, latest.x + 10);
  const bubbleY = Math.max(pad.top + 8, latest.y - 18);
  roundRect(ctx, bubbleX, bubbleY - 14, bubbleWidth, 24, 12);
  ctx.fillStyle = "rgba(102,217,232,0.16)";
  ctx.fill();
  ctx.strokeStyle = "rgba(102,217,232,0.46)";
  ctx.stroke();
  ctx.fillStyle = "#f2efe5";
  ctx.textAlign = "left";
  ctx.textBaseline = "middle";
  ctx.fillText(bubble, bubbleX + 9, bubbleY - 2);
  ctx.restore();
}

function drawBatchChart(canvas, rows, status) {
  const ctx = canvas.getContext("2d");
  const width = canvas.clientWidth || 420;
  const height = Number(canvas.dataset.baseHeight || canvas.getAttribute("height") || 300);
  canvas.dataset.baseHeight = String(height);
  canvas.style.height = `${height}px`;
  const dpr = window.devicePixelRatio || 1;
  canvas.width = Math.floor(width * dpr);
  canvas.height = Math.floor(height * dpr);
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, width, height);

  const completed = completedRows(rows)
    .filter((row) => numeric(row.files_processed) || numeric(row.files_failed))
    .slice(-6)
    .map((row) => {
      const processed = row.files_processed || 0;
      const failed = numeric(row.files_failed)
        ? row.files_failed
        : Math.max(0, (row.files_attempted || processed) - processed);
      return {
        label: timeLabel(row.timestamp),
        sub: numeric(row.pending_before) ? `${row.pending_before}->${row.pending_after}` : "batch",
        processed,
        failed,
        attempted: row.files_attempted || processed + failed || processed,
        elapsed: row.elapsed_seconds,
        live: false,
      };
    });

  const batch = status.batch || {};
  if (batch.total) {
    completed.push({
      label: "live",
      sub: `${batch.index || 0}/${batch.total}`,
      processed: batch.succeeded || 0,
      failed: batch.failed || 0,
      attempted: batch.total,
      live: true,
    });
  }

  const data = completed.slice(-7);
  const totalOk = completed.filter((row) => !row.live).reduce((sum, row) => sum + row.processed, 0);
  const totalFailed = completed.filter((row) => !row.live).reduce((sum, row) => sum + row.failed, 0);
  const durations = completed.filter((row) => !row.live && numeric(row.elapsed)).map((row) => row.elapsed);
  const avgDuration = durations.length ? durations.reduce((sum, value) => sum + value, 0) / durations.length : null;
  els.batchOk.textContent = totalOk ? String(totalOk) : "--";
  els.batchFailed.textContent = totalFailed ? String(totalFailed) : "0";
  els.batchDuration.textContent = compactDuration(avgDuration);
  els.batchCaption.textContent = data.length ? `${Math.max(0, data.length - (batch.total ? 1 : 0))} batches` : "waiting";

  if (!data.length) {
    ctx.fillStyle = "rgba(169,164,148,0.85)";
    ctx.font = "14px system-ui";
    ctx.fillText("No batch yield yet", 24, height / 2);
    return;
  }

  const pad = { top: 30, right: 78, bottom: 20, left: 74 };
  const rowGap = 10;
  const rowHeight = Math.max(22, (height - pad.top - pad.bottom - rowGap * (data.length - 1)) / data.length);
  const barHeight = Math.min(18, rowHeight * 0.56);
  const barWidth = Math.max(80, width - pad.left - pad.right);
  const maxTotal = Math.max(1, ...data.map((row) => row.attempted || row.processed + row.failed));

  ctx.save();
  ctx.font = "11px system-ui";
  ctx.textBaseline = "middle";
  ctx.fillStyle = "rgba(169,164,148,0.88)";
  ctx.textAlign = "left";
  ctx.fillText("ok", pad.left, 14);
  ctx.fillStyle = "#8fd694";
  roundRect(ctx, pad.left + 18, 8, 16, 8, 4);
  ctx.fill();
  ctx.fillStyle = "rgba(169,164,148,0.88)";
  ctx.fillText("failed", pad.left + 44, 14);
  ctx.fillStyle = "#f0bc62";
  roundRect(ctx, pad.left + 88, 8, 16, 8, 4);
  ctx.fill();

  for (let i = 0; i <= 2; i += 1) {
    const x = pad.left + (barWidth * i) / 2;
    ctx.strokeStyle = "rgba(242,239,229,0.08)";
    ctx.beginPath();
    ctx.moveTo(x, pad.top - 2);
    ctx.lineTo(x, height - pad.bottom + 2);
    ctx.stroke();
  }

  data.forEach((row, index) => {
    const y = pad.top + index * (rowHeight + rowGap);
    const barY = y + (rowHeight - barHeight) / 2;
    const processedWidth = (barWidth * row.processed) / maxTotal;
    const failedWidth = (barWidth * row.failed) / maxTotal;
    const attemptedWidth = (barWidth * (row.attempted || row.processed + row.failed)) / maxTotal;

    ctx.fillStyle = row.live ? "rgba(102,217,232,0.13)" : "rgba(242,239,229,0.08)";
    roundRect(ctx, pad.left, barY, Math.max(2, attemptedWidth), barHeight, 7);
    ctx.fill();

    if (row.processed) {
      ctx.fillStyle = "#8fd694";
      roundRect(ctx, pad.left, barY, Math.max(2, processedWidth), barHeight, 7);
      ctx.fill();
    }
    if (row.failed) {
      ctx.fillStyle = "#f0bc62";
      roundRect(ctx, pad.left + processedWidth, barY, Math.max(2, failedWidth), barHeight, 7);
      ctx.fill();
    }
    if (row.live) {
      ctx.strokeStyle = "rgba(102,217,232,0.72)";
      ctx.lineWidth = 1.5;
      ctx.setLineDash([4, 4]);
      roundRect(ctx, pad.left, barY - 2, Math.max(4, attemptedWidth), barHeight + 4, 9);
      ctx.stroke();
      ctx.setLineDash([]);
    }

    ctx.fillStyle = row.live ? "#66d9e8" : "rgba(242,239,229,0.88)";
    ctx.font = row.live ? "700 12px system-ui" : "12px system-ui";
    ctx.textAlign = "right";
    ctx.fillText(row.label, pad.left - 12, y + rowHeight * 0.36);
    ctx.fillStyle = "rgba(112,109,99,0.95)";
    ctx.font = "10px system-ui";
    ctx.fillText(row.sub, pad.left - 12, y + rowHeight * 0.74);

    const count = row.failed ? `${row.processed} ok ${row.failed} fail` : `${row.processed} ok`;
    ctx.fillStyle = row.failed ? "#f0bc62" : "rgba(242,239,229,0.9)";
    ctx.font = "700 12px system-ui";
    ctx.textAlign = "left";
    ctx.fillText(count, pad.left + barWidth + 10, barY + barHeight / 2);
  });
  ctx.restore();
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
  renderLlm(status.llm);
  els.currentJob.textContent = fmt(status.current_job_id);
  els.lastSuccess.textContent = status.last_success ? `${fmt(status.last_success.raw)} -> ${[...(status.last_success.created || []), ...(status.last_success.updated || [])].join(", ") || "none"}` : "--";
  els.trendCaption.textContent = metrics.length ? `${completedRows(metrics).length} batches` : "waiting for data";
  updateStageFlow(status.stage);
  renderEvents(snapshot.events || []);
  drawLineChart(els.pendingChart, metrics);
  drawBatchChart(els.batchChart, metrics, status);
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
setInterval(refresh, 1000);
window.addEventListener("resize", refresh);
