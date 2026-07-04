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
  selfHealPanel: document.getElementById("self-heal-panel"),
  selfHealCaption: document.getElementById("self-heal-caption"),
  selfHealState: document.getElementById("self-heal-state"),
  selfHealLatest: document.getElementById("self-heal-latest"),
  selfHealDetail: document.getElementById("self-heal-detail"),
  selfHealLastCheck: document.getElementById("self-heal-last-check"),
  selfHealLastStatus: document.getElementById("self-heal-last-status"),
  selfHealPendingPackets: document.getElementById("self-heal-pending-packets"),
  selfHealPacketTotal: document.getElementById("self-heal-packet-total"),
  selfHealFrontierCard: document.getElementById("self-heal-frontier-card"),
  selfHealFrontierState: document.getElementById("self-heal-frontier-state"),
  selfHealFrontierDetail: document.getElementById("self-heal-frontier-detail"),
  selfHealCounts: document.getElementById("self-heal-counts"),
  selfHealFeed: document.getElementById("self-heal-feed"),
  trendCaption: document.getElementById("trend-caption"),
  pendingCurrent: document.getElementById("pending-current"),
  pendingDelta: document.getElementById("pending-delta"),
  pendingRate: document.getElementById("pending-rate"),
  pendingEta: document.getElementById("pending-eta"),
  batchCaption: document.getElementById("batch-caption"),
  batchOk: document.getElementById("batch-ok"),
  batchFailed: document.getElementById("batch-failed"),
  batchDuration: document.getElementById("batch-duration"),
  saveTotal: document.getElementById("save-total"),
  saveProcessed: document.getElementById("save-processed"),
  savePages: document.getElementById("save-pages"),
  saveFailed: document.getElementById("save-failed"),
  saveMonths: document.getElementById("save-months"),
  saveHeatmap: document.getElementById("save-heatmap"),
  saveDetail: document.getElementById("save-detail"),
  saveFeed: document.getElementById("save-feed"),
  saveModeButtons: document.querySelectorAll("[data-save-mode]"),
  recallPanel: document.getElementById("recall-panel"),
  recallCaption: document.getElementById("recall-caption"),
  recallR3: document.getElementById("recall-r3"),
  recallWaste: document.getElementById("recall-waste"),
  recallP50: document.getElementById("recall-p50"),
  recallP95: document.getElementById("recall-p95"),
  recallCounts: document.getElementById("recall-counts"),
  recallCalibration: document.getElementById("recall-calibration"),
  recallFeed: document.getElementById("recall-feed"),
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

let saveHistoryMode = "daily";
let latestSaveHistory = null;
let selectedSaveDate = null;

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

function axisTimeLabel(value) {
  if (!value) return "--";
  const date = new Date(value);
  if (!Number.isNaN(date.getTime())) {
    return date.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });
  }
  return String(value).slice(-5);
}

function dateLabel(value) {
  if (!value) return "--";
  const date = new Date(value);
  if (!Number.isNaN(date.getTime())) {
    return date.toLocaleDateString([], { month: "short", day: "numeric" });
  }
  return String(value).slice(0, 10);
}

function timestampFromMs(ms) {
  return Number.isFinite(ms) ? new Date(ms).toISOString() : null;
}

function ageLabel(value) {
  const ms = parseMs(value);
  if (ms === null) return "--";
  const ageSeconds = Math.max(0, (Date.now() - ms) / 1000);
  return `${compactDuration(ageSeconds)} ago`;
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

function intValue(value) {
  return numeric(value) ? value : 0;
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
  const candidates = rows
    .filter((row) => numeric(row.pending_after) && (row.kind === "batch" || row.kind === "drain_batch"))
    .sort((a, b) => (parseMs(a.timestamp) || 0) - (parseMs(b.timestamp) || 0));
  const deduped = [];
  candidates.forEach((row) => {
    const before = numeric(row.pending_before) ? row.pending_before : "?";
    const transition = `${before}->${row.pending_after}`;
    const timestamp = parseMs(row.timestamp);
    const previous = deduped[deduped.length - 1];
    const previousBefore = previous && numeric(previous.pending_before) ? previous.pending_before : "?";
    const previousTransition = previous ? `${previousBefore}->${previous.pending_after}` : null;
    const previousTimestamp = previous ? parseMs(previous.timestamp) : null;
    const isSameOperation = previous
      && transition === previousTransition
      && timestamp !== null
      && previousTimestamp !== null
      && Math.abs(timestamp - previousTimestamp) <= 2_000;

    if (isSameOperation) {
      if (metricScore(row) > metricScore(previous)) {
        deduped[deduped.length - 1] = row;
      }
    } else {
      deduped.push(row);
    }
  });
  return deduped;
}

function latestCurrentMetric(rows) {
  return [...rows].reverse().find((row) => row.kind === "current" && numeric(row.pending_after));
}

function buildTrendPoints(rows, completed = completedRows(rows)) {
  const points = [];
  completed.forEach((row) => {
    const endMs = parseMs(row.timestamp);
    const elapsedMs = numeric(row.elapsed_seconds) ? Math.max(0, row.elapsed_seconds * 1000) : 0;
    const startTimestamp = endMs !== null ? timestampFromMs(endMs - elapsedMs) : row.timestamp;
    if (numeric(row.pending_before)) {
      points.push({
        timestamp: startTimestamp || row.timestamp,
        label: "queued",
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
  const latestMs = points.length ? parseMs(points[points.length - 1].timestamp) : null;
  const currentMs = current ? parseMs(current.timestamp) : null;
  if (
    current
    && (
      !points.length
      || points[points.length - 1].value !== current.pending_after
      || (currentMs !== null && latestMs !== null && currentMs - latestMs > 60_000)
    )
  ) {
    points.push({
      timestamp: current.timestamp,
      label: "now",
      value: current.pending_after,
    });
  }
  return points.slice(-24);
}

function trendCaption(points, completedCount) {
  if (!points.length) return "waiting for data";
  const first = points[0];
  const last = points[points.length - 1];
  const firstDate = dateLabel(first.timestamp);
  const lastDate = dateLabel(last.timestamp);
  const range = firstDate === lastDate
    ? `${firstDate} ${axisTimeLabel(first.timestamp)}-${axisTimeLabel(last.timestamp)}`
    : `${firstDate} ${axisTimeLabel(first.timestamp)} - ${lastDate} ${axisTimeLabel(last.timestamp)}`;
  return `${range} · ${completedCount} ${completedCount === 1 ? "batch" : "batches"}`;
}

function completedRowsInPointRange(points, completed) {
  if (!points.length) return [];
  const firstMs = parseMs(points[0].timestamp);
  const lastMs = parseMs(points[points.length - 1].timestamp);
  if (firstMs === null || lastMs === null) return completed.slice(-Math.ceil(points.length / 2));
  return completed.filter((row) => {
    const rowMs = parseMs(row.timestamp);
    return rowMs !== null && rowMs >= firstMs && rowMs <= lastMs;
  });
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

function measureTrend(points, completed = []) {
  if (!points.length) {
    return { current: null, drained: null, rate: null, etaSeconds: null };
  }
  const first = points[0];
  const last = points[points.length - 1];
  const drained = completed.reduce((sum, row) => {
    if (numeric(row.pending_before) && numeric(row.pending_after)) {
      return sum + Math.max(0, row.pending_before - row.pending_after);
    }
    return sum + intValue(row.files_processed);
  }, 0);
  const firstMs = parseMs(first.timestamp);
  const lastMs = parseMs(last.timestamp);
  const hours = firstMs !== null && lastMs !== null ? Math.max(0, (lastMs - firstMs) / 3_600_000) : 0;
  const rate = drained > 0 && hours > 0 ? drained / hours : null;
  return {
    current: last.value,
    drained: completed.length ? drained : null,
    rate,
    etaSeconds: rate && last.value > 0 ? (last.value / rate) * 3600 : null,
  };
}

function axisTickIndexes(points, width) {
  const maxTicks = width >= 900 ? 4 : 3;
  const tickCount = Math.min(maxTicks, points.length);
  if (tickCount <= 1) return [0];
  const indexes = new Set();
  for (let i = 0; i < tickCount; i += 1) {
    indexes.add(Math.round(((points.length - 1) * i) / (tickCount - 1)));
  }
  return [...indexes].sort((a, b) => a - b);
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

  const completed = completedRows(rows);
  const pointsRaw = buildTrendPoints(rows, completed);
  const visibleCompleted = completedRowsInPointRange(pointsRaw, completed);
  const measure = measureTrend(pointsRaw, visibleCompleted);
  els.pendingCurrent.textContent = measure.current === null ? "--" : String(measure.current);
  els.pendingDelta.textContent = measure.drained === null ? "--" : measure.drained >= 0 ? String(measure.drained) : `+${Math.abs(measure.drained)}`;
  els.pendingRate.textContent = measure.rate ? `${measure.rate.toFixed(measure.rate < 10 ? 1 : 0)}/h` : "--";
  els.pendingEta.textContent = compactDuration(measure.etaSeconds);
  els.trendCaption.textContent = trendCaption(pointsRaw, visibleCompleted.length);

  const pad = { top: 24, right: 28, bottom: 56, left: 48 };
  const plotWidth = width - pad.left - pad.right;
  const plotHeight = height - pad.top - pad.bottom;

  if (pointsRaw.length < 2) {
    ctx.fillStyle = "rgba(169,164,148,0.85)";
    ctx.font = "14px system-ui";
    ctx.fillText(pointsRaw.length ? `Only one queue sample · ${dateLabel(pointsRaw[0].timestamp)} ${axisTimeLabel(pointsRaw[0].timestamp)}` : "Waiting for completed batches", pad.left, height / 2);
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
  axisTickIndexes(points, width).forEach((index) => {
    const point = points[index];
    const align = index === 0 ? "left" : index === points.length - 1 ? "right" : "center";
    ctx.strokeStyle = "rgba(242,239,229,0.08)";
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(point.x, pad.top);
    ctx.lineTo(point.x, pad.top + plotHeight);
    ctx.stroke();
    ctx.textAlign = align;
    ctx.font = "11px system-ui";
    ctx.fillStyle = "rgba(169,164,148,0.9)";
    ctx.fillText(dateLabel(point.timestamp), point.x, height - 28);
    ctx.fillStyle = "rgba(242,239,229,0.78)";
    ctx.fillText(axisTimeLabel(point.timestamp), point.x, height - 12);
  });

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

function renderSelfHeal(selfHeal) {
  const data = selfHeal || {};
  const latest = data.latest || null;
  const history = Array.isArray(data.history) ? data.history : [];
  const counts = data.counts || {};
  const watch = data.watch || {};
  const lastChecked = watch.last_checked || {};
  const packets = watch.packets || {};
  const frontier = watch.frontier_preflight || {};
  const status = (data.status || "quiet").toLowerCase();

  els.selfHealPanel.classList.remove("resolved", "pending", "failed", "quiet");
  els.selfHealPanel.classList.add(["resolved", "pending", "failed"].includes(status) ? status : "quiet");
  els.selfHealCaption.textContent = history.length ? `${history.length} records` : "quiet";
  els.selfHealState.textContent = status;
  els.selfHealLatest.textContent = latest ? fmt(latest.title) : "No repairs yet";
  els.selfHealDetail.textContent = latest
    ? [latest.raw_file, latest.detail].filter(Boolean).join(" · ")
    : "--";

  els.selfHealLastCheck.textContent = lastChecked.timestamp ? ageLabel(lastChecked.timestamp) : "--";
  els.selfHealLastStatus.textContent = lastChecked.timestamp
    ? `${lastChecked.status || "unknown"} · ${intValue(lastChecked.packets_seen)} packets`
    : "no drain checks";
  els.selfHealPendingPackets.textContent = String(intValue(packets.pending));
  els.selfHealPacketTotal.textContent = `${intValue(packets.total)} total · ${intValue(packets.failed)} failed`;

  els.selfHealFrontierCard.classList.remove("ready", "blocked", "unknown");
  if (frontier.ok === true) {
    els.selfHealFrontierCard.classList.add("ready");
    els.selfHealFrontierState.textContent = "ready";
  } else if (frontier.ok === false) {
    els.selfHealFrontierCard.classList.add("blocked");
    els.selfHealFrontierState.textContent = "blocked";
  } else {
    els.selfHealFrontierCard.classList.add("unknown");
    els.selfHealFrontierState.textContent = "--";
  }
  const failure = frontier.failure || {};
  const frontierDetails = [
    frontier.checked_at ? `checked ${ageLabel(frontier.checked_at)}` : null,
    frontier.cached ? "cached" : null,
    frontier.missing_exec_options && frontier.missing_exec_options.length
      ? `missing ${frontier.missing_exec_options.join(", ")}`
      : null,
    failure.failure_class || frontier.error || null,
  ].filter(Boolean);
  els.selfHealFrontierDetail.textContent = frontierDetails.join(" · ") || "preflight";

  const countItems = [
    ["resolved", counts.resolved || 0],
    ["pending", counts.pending || 0],
    ["failed", counts.failed || 0],
    ["frontier", counts.frontier || 0],
    ["human", counts.human_required || 0],
    ["review", counts.pending_frontier_review || 0],
  ];
  els.selfHealCounts.innerHTML = "";
  countItems.forEach(([label, value]) => {
    const item = document.createElement("div");
    item.className = "self-heal-count";
    const name = document.createElement("span");
    name.textContent = label;
    const count = document.createElement("strong");
    count.textContent = String(value);
    item.append(name, count);
    els.selfHealCounts.appendChild(item);
  });

  els.selfHealFeed.innerHTML = "";
  if (!history.length) {
    const empty = document.createElement("div");
    empty.className = "self-heal-empty";
    empty.textContent = "No self-heal records yet.";
    els.selfHealFeed.appendChild(empty);
    return;
  }

  [...history].slice(-5).reverse().forEach((item) => {
    const row = document.createElement("details");
    row.className = `self-heal-row ${(item.level || "info").toLowerCase()}`;
    const summary = document.createElement("summary");
    const time = document.createElement("time");
    time.textContent = timeLabel(item.timestamp);
    const body = document.createElement("div");
    body.className = "self-heal-row-body";
    const title = document.createElement("strong");
    title.textContent = fmt(item.title);
    const detail = document.createElement("span");
    detail.textContent = [item.raw_file, item.detail].filter(Boolean).join(" · ");
    body.append(title, detail);
    summary.append(time, body);
    row.append(summary);
    if (item.details) {
      const pre = document.createElement("pre");
      pre.className = "self-heal-json";
      pre.textContent = JSON.stringify(item.details, null, 2);
      row.appendChild(pre);
    }
    els.selfHealFeed.appendChild(row);
  });
}

function parseDateKey(value) {
  if (!value) return null;
  const parts = String(value).split("-").map((part) => Number(part));
  if (parts.length !== 3 || parts.some((part) => !Number.isFinite(part))) return null;
  return new Date(parts[0], parts[1] - 1, parts[2]);
}

function dateKeyFromDate(date) {
  const year = date.getFullYear();
  const month = `${date.getMonth() + 1}`.padStart(2, "0");
  const day = `${date.getDate()}`.padStart(2, "0");
  return `${year}-${month}-${day}`;
}

function startOfWeekKey(dateKey) {
  const date = parseDateKey(dateKey);
  if (!date) return dateKey;
  date.setDate(date.getDate() - date.getDay());
  return dateKeyFromDate(date);
}

function buildSaveModeValues(days) {
  const weekly = new Map();
  let cumulative = 0;
  const cumulativeByDate = new Map();
  days.forEach((day) => {
    const rawSaved = intValue(day.raw_saved);
    const week = startOfWeekKey(day.date);
    weekly.set(week, (weekly.get(week) || 0) + rawSaved);
    cumulative += rawSaved;
    cumulativeByDate.set(day.date, cumulative);
  });
  return days.map((day) => {
    const rawSaved = intValue(day.raw_saved);
    if (saveHistoryMode === "weekly") {
      return weekly.get(startOfWeekKey(day.date)) || 0;
    }
    if (saveHistoryMode === "cumulative") {
      return cumulativeByDate.get(day.date) || 0;
    }
    return rawSaved;
  });
}

function heatLevel(value, maxValue) {
  if (!value || !maxValue) return 0;
  const ratio = value / maxValue;
  if (ratio >= 0.75) return 4;
  if (ratio >= 0.45) return 3;
  if (ratio >= 0.2) return 2;
  return 1;
}

function formatSources(sources) {
  if (!Array.isArray(sources) || !sources.length) return "none";
  return sources.map((source) => `${source.name} ${source.count}`).join(", ");
}

function saveTooltip(day, value) {
  const metric = saveHistoryMode === "weekly"
    ? `${value} saved this week`
    : saveHistoryMode === "cumulative"
      ? `${value} saved cumulative`
      : `${intValue(day.raw_saved)} saved`;
  const pages = intValue(day.pages_created) + intValue(day.pages_updated);
  return [
    day.date,
    metric,
    `${intValue(day.processed)} processed`,
    `${intValue(day.failed)} failed`,
    `${pages} page changes`,
    `sources: ${formatSources(day.sources)}`,
  ].join(" · ");
}

function renderSaveMonths(days, startPad, columnCount) {
  els.saveMonths.innerHTML = "";
  els.saveMonths.style.gridTemplateColumns = `repeat(${columnCount}, 12px)`;
  const labelsByColumn = new Map();
  days.forEach((day, index) => {
    const date = parseDateKey(day.date);
    if (!date) return;
    const monthKey = `${date.getFullYear()}-${date.getMonth()}`;
    const previous = index > 0 ? parseDateKey(days[index - 1].date) : null;
    const previousMonthKey = previous ? `${previous.getFullYear()}-${previous.getMonth()}` : null;
    if (monthKey === previousMonthKey) return;
    const column = Math.floor((index + startPad) / 7) + 1;
    labelsByColumn.set(column, date.toLocaleDateString("ja-JP", { month: "short" }));
  });
  labelsByColumn.forEach((month, column) => {
    const label = document.createElement("span");
    label.textContent = month;
    label.style.gridColumnStart = String(column);
    els.saveMonths.appendChild(label);
  });
}

function selectSaveDay(days) {
  if (selectedSaveDate && days.some((day) => day.date === selectedSaveDate)) {
    return days.find((day) => day.date === selectedSaveDate);
  }
  const active = [...days].reverse().find((day) =>
    intValue(day.raw_saved) || intValue(day.processed) || intValue(day.pages_created) || intValue(day.pages_updated)
  );
  if (active) {
    selectedSaveDate = active.date;
    return active;
  }
  selectedSaveDate = days.length ? days[days.length - 1].date : null;
  return days.length ? days[days.length - 1] : null;
}

function renderSaveDetail(day) {
  els.saveDetail.innerHTML = "";
  if (!day) {
    els.saveDetail.textContent = "--";
    return;
  }
  const title = document.createElement("strong");
  title.textContent = day.date;
  const stats = document.createElement("div");
  stats.className = "save-detail-stats";
  [
    ["saved", intValue(day.raw_saved)],
    ["processed", intValue(day.processed)],
    ["created", intValue(day.pages_created)],
    ["updated", intValue(day.pages_updated)],
    ["failed", intValue(day.failed)],
  ].forEach(([label, value]) => {
    const item = document.createElement("span");
    item.textContent = `${label} ${value}`;
    stats.appendChild(item);
  });

  const sources = document.createElement("p");
  sources.textContent = `sources: ${formatSources(day.sources)}`;
  const samples = document.createElement("ul");
  [...(day.raw_samples || []).slice(0, 3), ...(day.page_samples || []).slice(0, 3)].forEach((sample) => {
    const item = document.createElement("li");
    item.textContent = sample;
    samples.appendChild(item);
  });
  if (!samples.childElementCount) {
    const empty = document.createElement("p");
    empty.textContent = "no saves";
    els.saveDetail.append(title, stats, sources, empty);
    return;
  }
  els.saveDetail.append(title, stats, sources, samples);
}

function renderSaveFeed(recent) {
  els.saveFeed.innerHTML = "";
  const rows = Array.isArray(recent) ? [...recent].slice(-8).reverse() : [];
  if (!rows.length) {
    els.saveFeed.innerHTML = "<div class=\"self-heal-empty\">No save records yet.</div>";
    return;
  }
  rows.forEach((day) => {
    const row = document.createElement("button");
    row.type = "button";
    row.className = "save-feed-row";
    if (day.date === selectedSaveDate) row.classList.add("active");
    const date = document.createElement("time");
    date.textContent = day.date.slice(5);
    const body = document.createElement("span");
    const pages = intValue(day.pages_created) + intValue(day.pages_updated);
    body.textContent = `${intValue(day.raw_saved)} saved · ${intValue(day.processed)} processed · ${pages} pages`;
    row.append(date, body);
    row.addEventListener("click", () => {
      selectedSaveDate = day.date;
      renderSaveHistory(latestSaveHistory);
    });
    els.saveFeed.appendChild(row);
  });
}

function renderSaveHistory(saveHistory) {
  const data = saveHistory || {};
  latestSaveHistory = data;
  const days = Array.isArray(data.days) ? data.days : [];
  const totals = data.totals || {};
  const pages = intValue(totals.pages_created) + intValue(totals.pages_updated);

  els.saveTotal.textContent = intValue(totals.raw_saved).toLocaleString();
  els.saveProcessed.textContent = intValue(totals.processed).toLocaleString();
  els.savePages.textContent = pages.toLocaleString();
  els.saveFailed.textContent = intValue(totals.failed).toLocaleString();
  els.saveModeButtons.forEach((button) => {
    button.classList.toggle("active", button.dataset.saveMode === saveHistoryMode);
  });

  els.saveHeatmap.innerHTML = "";
  if (!days.length) {
    els.saveMonths.innerHTML = "";
    renderSaveDetail(null);
    renderSaveFeed([]);
    return;
  }

  const firstDate = parseDateKey(days[0].date);
  const startPad = firstDate ? firstDate.getDay() : 0;
  const values = buildSaveModeValues(days);
  const maxValue = Math.max(1, ...values);
  const cellCount = startPad + days.length;
  const columnCount = Math.ceil(cellCount / 7);
  els.saveHeatmap.style.gridTemplateColumns = `repeat(${columnCount}, 12px)`;
  renderSaveMonths(days, startPad, columnCount);

  for (let index = 0; index < startPad; index += 1) {
    const blank = document.createElement("span");
    blank.className = "save-cell blank";
    els.saveHeatmap.appendChild(blank);
  }

  const selectedDay = selectSaveDay(days);
  days.forEach((day, index) => {
    const value = values[index];
    const level = heatLevel(value, maxValue);
    const cell = document.createElement("button");
    cell.type = "button";
    cell.className = `save-cell level-${level}`;
    if (day.date === selectedSaveDate) cell.classList.add("selected");
    if (intValue(day.failed)) cell.classList.add("has-failures");
    cell.title = saveTooltip(day, value);
    cell.setAttribute("aria-label", cell.title);
    cell.addEventListener("click", () => {
      selectedSaveDate = day.date;
      renderSaveHistory(data);
    });
    els.saveHeatmap.appendChild(cell);
  });

  renderSaveDetail(selectedDay);
  renderSaveFeed(data.recent || []);
}

function pctLabel(value) {
  if (!numeric(value)) return "--";
  return `${(value * 100).toFixed(1)}%`;
}

function msLabel(value) {
  if (!numeric(value)) return "--";
  return `${Math.round(value)}ms`;
}

function renderRecall(recall) {
  const data = recall || {};
  const decisions = data.decisions || {};
  const latency = data.latency_ms || {};
  const latestEval = data.latest_eval || null;
  const evalMetricsLatency = latestEval && latestEval.latency_ms ? latestEval.latency_ms : {};
  const calibration = data.calibration || {};
  const pulls = data.pulls || {};

  els.recallCaption.textContent = data.samples ? `${data.samples} decisions` : "quiet";
  els.recallR3.textContent = latestEval ? pctLabel(latestEval.recall_at_3) : "--";
  els.recallWaste.textContent = latestEval ? pctLabel(latestEval.waste_injection_rate) : "--";
  els.recallP50.textContent = msLabel(numeric(latency.p50) ? latency.p50 : evalMetricsLatency.p50);
  els.recallP95.textContent = msLabel(numeric(latency.p95) ? latency.p95 : evalMetricsLatency.p95);

  const countItems = [
    ["none", decisions.none || 0],
    ["search", decisions.search || 0],
    ["read", decisions.read || 0],
    ["judge", data.judge_used || 0],
    ["rewrite", data.rewrite_used || 0],
    ["pulls", pulls.total || 0],
  ];
  els.recallCounts.innerHTML = "";
  countItems.forEach(([label, value]) => {
    const item = document.createElement("div");
    item.className = "self-heal-count";
    const name = document.createElement("span");
    name.textContent = label;
    const count = document.createElement("strong");
    count.textContent = String(value);
    item.append(name, count);
    els.recallCounts.appendChild(item);
  });

  const lastApplied = calibration.last_applied || null;
  if (lastApplied) {
    const when = timeLabel(lastApplied.ts || lastApplied.timestamp);
    const note = lastApplied.reason || lastApplied.status || "applied";
    els.recallCalibration.textContent = `calibration: ${note} · ${when} · ${(calibration.history || []).length} records`;
  } else {
    els.recallCalibration.textContent = "calibration: not yet applied";
  }

  els.recallFeed.innerHTML = "";
  const recent = Array.isArray(data.recent) ? [...data.recent].slice(-6).reverse() : [];
  if (!recent.length) {
    const empty = document.createElement("div");
    empty.className = "self-heal-empty";
    empty.textContent = "No recall decisions yet.";
    els.recallFeed.appendChild(empty);
    return;
  }
  recent.forEach((item) => {
    const row = document.createElement("div");
    row.className = "event";
    const time = document.createElement("time");
    time.textContent = timeLabel(item.timestamp);
    const badge = document.createElement("span");
    const decision = String(item.decision || "none");
    badge.className = `event-level ${decision === "read" ? "success" : decision === "search" ? "info" : "warn"}`;
    badge.textContent = decision;
    const message = document.createElement("span");
    message.className = "event-message";
    const extra = [
      numeric(item.latency_ms) ? `${item.latency_ms}ms` : null,
      item.pages ? `${item.pages}p` : null,
      item.used_judge ? "judge" : null,
    ].filter(Boolean).join(" ");
    message.textContent = extra ? `${item.preview} · ${extra}` : fmt(item.preview);
    row.append(time, badge, message);
    els.recallFeed.appendChild(row);
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
  updateStageFlow(status.stage);
  renderSelfHeal(snapshot.self_heal || {});
  renderRecall(snapshot.recall || {});
  renderSaveHistory(snapshot.save_history || {});
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

els.saveModeButtons.forEach((button) => {
  button.addEventListener("click", () => {
    saveHistoryMode = button.dataset.saveMode || "daily";
    renderSaveHistory(latestSaveHistory);
  });
});

refresh();
setInterval(refresh, 1000);
window.addEventListener("resize", refresh);
