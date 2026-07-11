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
  workOverview: document.getElementById("work-overview"),
  workSummary: document.getElementById("work-summary"),
  workUpdated: document.getElementById("work-updated"),
  workDetail: document.getElementById("work-detail"),
  workSteps: document.querySelectorAll("[data-work-step]"),
  currentRaw: document.getElementById("current-raw"),
  currentOp: document.getElementById("current-op"),
  llmSignal: document.getElementById("llm-signal"),
  llmState: document.getElementById("llm-state"),
  llmAge: document.getElementById("llm-age"),
  llmTarget: document.getElementById("llm-target"),
  llmStats: document.getElementById("llm-stats"),
  llmSparkline: document.getElementById("llm-sparkline"),
  localConsensus: document.getElementById("local-consensus"),
  frontierRepair: document.getElementById("frontier-repair"),
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
  knowledgeCaption: document.getElementById("knowledge-caption"),
  knowledgeChart: document.getElementById("knowledge-chart"),
  knowledgeTotal: document.getElementById("knowledge-total"),
  knowledgeBars: document.getElementById("knowledge-bars"),
  knowledgePages: document.getElementById("knowledge-pages"),
  knowledgeSize: document.getElementById("knowledge-size"),
  knowledgeCategories: document.getElementById("knowledge-categories"),
  knowledgeModeButtons: document.querySelectorAll("[data-knowledge-mode]"),
  healthCaption: document.getElementById("health-caption"),
  healthSummaryCoverage: document.getElementById("health-summary-coverage"),
  healthCapture: document.getElementById("health-capture"),
  healthReadback: document.getElementById("health-readback"),
  healthSensitive: document.getElementById("health-sensitive"),
  healthDupes: document.getElementById("health-dupes"),
  healthGolden: document.getElementById("health-golden"),
  healthGoldenSet: document.getElementById("health-golden-set"),
  healthRetention: document.getElementById("health-retention"),
  healthDistill: document.getElementById("health-distill"),
  modelCaption: document.getElementById("model-caption"),
  modelInstalled: document.getElementById("model-installed"),
  modelLoaded: document.getElementById("model-loaded"),
  modelConfigured: document.getElementById("model-configured"),
  modelMissing: document.getElementById("model-missing"),
  modelGrid: document.getElementById("model-grid"),
  modelLabCaption: document.getElementById("model-lab-caption"),
  modelLabRoles: document.getElementById("model-lab-roles"),
  modelLabReplays: document.getElementById("model-lab-replays"),
  modelLabCandidates: document.getElementById("model-lab-candidates"),
  modelLabCanaries: document.getElementById("model-lab-canaries"),
  modelLabGrid: document.getElementById("model-lab-grid"),
  recallPanel: document.getElementById("recall-panel"),
  recallCaption: document.getElementById("recall-caption"),
  recallR3: document.getElementById("recall-r3"),
  recallWaste: document.getElementById("recall-waste"),
  recallP50: document.getElementById("recall-p50"),
  recallP95: document.getElementById("recall-p95"),
  recallCounts: document.getElementById("recall-counts"),
  recallCalibration: document.getElementById("recall-calibration"),
  recallFeed: document.getElementById("recall-feed"),
  recallLabPanel: document.getElementById("recall-lab-panel"),
  recallLabCaption: document.getElementById("recall-lab-caption"),
  recallLabState: document.getElementById("recall-lab-state"),
  recallLabLatest: document.getElementById("recall-lab-latest"),
  recallLabDetail: document.getElementById("recall-lab-detail"),
  recallLabActive: document.getElementById("recall-lab-active"),
  recallLabDev: document.getElementById("recall-lab-dev"),
  recallLabHoldout: document.getElementById("recall-lab-holdout"),
  recallLabModels: document.getElementById("recall-lab-models"),
  recallLabPolicy: document.getElementById("recall-lab-policy"),
  recallLabFeed: document.getElementById("recall-lab-feed"),
  eventFeed: document.getElementById("event-feed"),
  pendingChart: document.getElementById("pending-chart"),
  saveLoadTooltip: document.getElementById("save-load-tooltip"),
  batchChart: document.getElementById("batch-chart"),
};

const llmSignalHistory = {
  key: null,
  lastChars: null,
  lastSeenMs: null,
  rates: Array(32).fill(0),
};

const WORK_STAGE_ORDER = ["raw", "triage", "generate", "review", "apply", "index"];
const WORK_STAGE_ALIASES = {
  idle: "idle",
  queued: "raw",
  raw: "raw",
  triage: "triage",
  classify: "triage",
  generate: "generate",
  generating: "generate",
  llm: "generate",
  review: "review",
  apply: "apply",
  applying: "apply",
  write: "apply",
  index: "index",
  indexing: "index",
  complete: "index",
  done: "index",
};

let saveHistoryMode = "daily";
let latestSaveHistory = null;
let selectedSaveDate = null;
let knowledgeMixMode = "size";
let latestKnowledgeMix = null;
let saveLoadHitRegions = [];

const KNOWLEDGE_COLORS = [
  "#66d9e8",
  "#8fd694",
  "#f0bc62",
  "#c792ea",
  "#ff8a80",
  "#82aaff",
  "#ffd166",
  "#7bdcb5",
];

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

function formatBytes(value) {
  const bytes = intValue(value);
  if (bytes >= 1_000_000_000) {
    const gb = bytes / 1_000_000_000;
    return `${gb.toFixed(gb < 10 ? 1 : 0)} GB`;
  }
  if (bytes >= 1_000_000) {
    const mb = bytes / 1_000_000;
    return `${mb.toFixed(mb < 10 ? 1 : 0)} MB`;
  }
  if (bytes >= 1_000) {
    const kb = bytes / 1_000;
    return `${kb.toFixed(kb < 10 ? 1 : 0)} KB`;
  }
  return `${bytes} B`;
}

function shareLabel(value) {
  if (!Number.isFinite(value) || value <= 0) return "0%";
  return `${(value * 100).toFixed(value < 0.1 ? 1 : 0)}%`;
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

function lastSuccessTargets(lastSuccess) {
  if (!lastSuccess) return "";
  const targets = [...(lastSuccess.created || []), ...(lastSuccess.updated || [])].filter(Boolean);
  return targets.length ? targets.join(", ") : "no page changes";
}

function inferWorkStage(status, llm) {
  const rawStage = String(status.stage || status.current_op || "").toLowerCase();
  if (llm && llm.active) return "generate";
  return WORK_STAGE_ALIASES[rawStage] || (status.current_raw ? "raw" : "idle");
}

function setWorkStage(stage, stateKind) {
  const activeIndex = WORK_STAGE_ORDER.indexOf(stage);
  els.workSteps.forEach((step) => {
    const key = step.dataset.workStep;
    const index = WORK_STAGE_ORDER.indexOf(key);
    step.classList.toggle("active", index === activeIndex);
    step.classList.toggle("done", activeIndex >= 0 && index >= 0 && index < activeIndex);
  });
  els.workOverview.classList.remove("idle", "running", "complete", "warning", "stalled");
  els.workOverview.classList.add(stateKind);
}

function renderWorkStatus(status) {
  const llm = status.llm || null;
  const consensus = status.local_consensus || {};
  const repair = status.frontier_repair || {};
  const ingestActive = Boolean(status.current_raw || status.current_job_id || (llm && llm.active));
  const localReviewOnly = Boolean(consensus.active && !ingestActive);
  const localEvaluationOnly = Boolean(
    localReviewOnly && String((consensus.latest || {}).role || "").startsWith("model_eval:")
  );
  const repairOnly = Boolean(repair.active && !ingestActive && !localReviewOnly);
  const reviewOnly = localReviewOnly || repairOnly;
  const active = Boolean(ingestActive || consensus.active || repair.active);
  const lastSuccess = status.last_success || null;
  let stage = reviewOnly ? "review" : inferWorkStage(status, llm);
  if (!active && lastSuccess) stage = "index";
  const lastTargets = lastSuccessTargets(lastSuccess);
  const updated = status.updated_at ? `updated ${timeLabel(status.updated_at)}` : "--";
  let reviewLabel = "Frontier repair running";
  if (localReviewOnly) {
    reviewLabel = localEvaluationOnly ? "Local model evaluation" : "Local consensus reviewing";
  }
  const stageLabels = {
    raw: "Reading raw capture",
    triage: "Choosing page action",
    generate: "Generating wiki page",
    apply: "Writing page update",
    index: "Indexing completed work",
    review: reviewLabel,
  };

  let stateKind = "idle";
  let summary = "Waiting for new raw";
  let detail = "No active raw is being processed.";

  if (active) {
    stateKind = "running";
    summary = stageLabels[stage] || "Processing raw";
    if (reviewOnly) {
      if (localReviewOnly) {
        const latest = consensus.latest || {};
        const count = Number(consensus.count || 1);
        const subject = [latest.role, latest.model].filter(Boolean).join(" · ");
        const activity = localEvaluationOnly ? "local eval vote" : "local vote";
        detail = `${count} active ${activity}${count === 1 ? "" : "s"}${subject ? ` · ${subject}` : ""}`;
      } else {
        const process = repair.process_activity || {};
        const latest = process.latest || repair.active_incident || {};
        const subject = [latest.kind || latest.component, latest.model].filter(Boolean).join(" · ");
        detail = `exceptional code repair${subject ? ` · ${subject}` : ""}`;
      }
    } else {
      const current = shortName(status.current_raw || (llm && (llm.raw || llm.target)) || status.current_job_id);
      const op = fmt(status.current_op || stage, "work");
      detail = `${op} on ${current}`;
    }
  } else if (lastSuccess) {
    stateKind = "complete";
    summary = "Completed last raw";
    detail = `${shortName(lastSuccess.raw)} -> ${shortName(lastTargets || "none")}. Waiting for the next raw.`;
  } else if (status.last_problem) {
    stateKind = "warning";
    summary = "Idle after warning";
    detail = shortName(status.last_problem.message || status.last_error || "Check latest event.");
  }

  els.workSummary.textContent = summary;
  els.workUpdated.textContent = updated;
  els.workDetail.textContent = detail;
  setWorkStage(stage, stateKind);
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

function renderLlm(llm, status = {}) {
  if (!llm) {
    const lastSuccess = status.last_success || null;
    const waitingForLlm = Boolean(status.current_raw || status.current_job_id);
    setLlmSignalClass(waitingForLlm ? "waiting" : "idle");
    llmSignalHistory.key = null;
    llmSignalHistory.lastChars = null;
    llmSignalHistory.lastSeenMs = null;
    llmSignalHistory.rates = Array(32).fill(0);
    els.llmState.textContent = waitingForLlm ? "waiting" : "idle";
    els.llmAge.textContent = status.updated_at ? `status ${ageLabel(status.updated_at)}` : "--";
    els.llmTarget.textContent = waitingForLlm ? shortName(status.current_raw || status.current_job_id) : "No active LLM call";
    els.llmStats.textContent = lastSuccess
      ? `Last success ${ageLabel(status.updated_at)}`
      : "Waiting for ingest work";
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

function dateKeyLabel(dateKey) {
  const date = parseDateKey(dateKey);
  if (!date) return fmt(dateKey);
  return date.toLocaleDateString("ja-JP", { month: "numeric", day: "numeric" });
}

function rawDateKeyFromName(value) {
  const match = String(value || "").match(/(?:^|[^0-9])(20\d{6})(?:[^0-9]|$)/);
  if (!match) return null;
  const stamp = match[1];
  return `${stamp.slice(0, 4)}-${stamp.slice(4, 6)}-${stamp.slice(6, 8)}`;
}

function saveLoadRows(saveHistory, status) {
  const days = Array.isArray(saveHistory?.days) ? saveHistory.days.slice(-30) : [];
  const activeDate = rawDateKeyFromName(status?.current_raw);
  const hasActiveRaw = Boolean(status?.current_raw && status?.state !== "idle");
  return days.map((day) => {
    const rawSegments = Array.isArray(day.raw_segments) ? day.raw_segments : [];
    let segments = rawSegments
      .map((segment) => ({
        name: fmt(segment.name, "raw"),
        bytes: intValue(segment.bytes),
        status: ["processed", "pending", "failed"].includes(segment.status) ? segment.status : "pending",
        source: fmt(segment.source, "raw"),
      }))
      .filter((segment) => segment.bytes > 0);
    if (!segments.length && intValue(day.raw_bytes)) {
      [
        ["processed", intValue(day.processed_bytes)],
        ["failed", intValue(day.failed_bytes)],
        ["pending", intValue(day.pending_bytes)],
      ].forEach(([statusName, bytes]) => {
        if (bytes > 0) {
          segments.push({ name: statusName, bytes, status: statusName, source: "aggregate" });
        }
      });
    }
    const total = segments.reduce((sum, segment) => sum + segment.bytes, 0);
    const processed = segments.filter((segment) => segment.status === "processed").reduce((sum, segment) => sum + segment.bytes, 0);
    const failed = segments.filter((segment) => segment.status === "failed").reduce((sum, segment) => sum + segment.bytes, 0);
    const pending = Math.max(0, total - processed - failed);
    return {
      date: day.date,
      total,
      processed,
      failed,
      pending,
      segments,
      active: hasActiveRaw && day.date === activeDate,
    };
  });
}

function segmentColor(status, bytes, maxBytes) {
  const ratio = maxBytes > 0 ? Math.min(1, Math.sqrt(bytes / maxBytes)) : 0;
  if (status === "failed") {
    return `hsl(38, 84%, ${46 + ratio * 24}%)`;
  }
  if (status === "pending") {
    return `hsl(187, 72%, ${32 + ratio * 28}%)`;
  }
  return `hsl(126, 43%, ${40 + ratio * 30}%)`;
}

function drawStackSegment(ctx, x, baseY, width, height, color, options = {}) {
  if (height <= 0) return baseY;
  const y = baseY - height;
  ctx.fillStyle = color;
  ctx.fillRect(x, y, width, height);
  if (height >= 1) {
    ctx.save();
    ctx.strokeStyle = "rgba(0, 0, 0, 0.78)";
    ctx.lineWidth = 1;
    ctx.strokeRect(x + 0.5, y + 0.5, Math.max(1, width - 1), Math.max(1, height - 1));
    ctx.restore();
  }
  if (options.dashed) {
    ctx.save();
    ctx.strokeStyle = options.stroke || "rgba(102,217,232,0.72)";
    ctx.lineWidth = 1.4;
    ctx.setLineDash([4, 4]);
    ctx.strokeRect(x + 0.5, y + 0.5, Math.max(1, width - 1), Math.max(1, height - 1));
    ctx.restore();
  }
  return y;
}

function hideSaveLoadTooltip() {
  if (!els.saveLoadTooltip) return;
  els.saveLoadTooltip.hidden = true;
  els.pendingChart.style.cursor = "default";
}

function showSaveLoadTooltip(region, x, y) {
  if (!els.saveLoadTooltip) return;
  els.saveLoadTooltip.innerHTML = "";
  const title = document.createElement("strong");
  title.textContent = `${region.date} · ${formatBytes(region.bytes)}`;
  const raw = document.createElement("span");
  raw.textContent = region.name;
  const meta = document.createElement("small");
  meta.textContent = `${region.status} · ${region.source}`;
  els.saveLoadTooltip.append(title, raw, meta);
  const parent = els.pendingChart.parentElement;
  const maxLeft = Math.max(0, (parent?.clientWidth || 0) - 220);
  const maxTop = Math.max(0, (parent?.clientHeight || 0) - 86);
  els.saveLoadTooltip.style.left = `${Math.min(maxLeft, x + 12)}px`;
  els.saveLoadTooltip.style.top = `${Math.min(maxTop, y + 12)}px`;
  els.saveLoadTooltip.hidden = false;
  els.pendingChart.style.cursor = "crosshair";
}

function handleSaveLoadHover(event) {
  const rect = els.pendingChart.getBoundingClientRect();
  const x = event.clientX - rect.left;
  const y = event.clientY - rect.top;
  const region = saveLoadHitRegions.find((item) =>
    x >= item.x
    && x <= item.x + item.width
    && y >= item.y
    && y <= item.y + item.height
  );
  if (!region) {
    hideSaveLoadTooltip();
    return;
  }
  showSaveLoadTooltip(region, x, y);
}

function drawLineChart(canvas, saveHistory, status = {}) {
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

  const rows = saveLoadRows(saveHistory, status);
  saveLoadHitRegions = [];
  const totals = rows.reduce((acc, row) => {
    acc.total += row.total;
    acc.processed += row.processed;
    acc.pending += row.pending;
    acc.failed += row.failed;
    return acc;
  }, { total: 0, processed: 0, pending: 0, failed: 0 });

  els.pendingCurrent.textContent = formatBytes(totals.total);
  els.pendingDelta.textContent = formatBytes(totals.processed);
  els.pendingRate.textContent = formatBytes(totals.pending);
  els.pendingEta.textContent = formatBytes(totals.failed);
  els.trendCaption.textContent = rows.length
    ? `${dateKeyLabel(rows[0].date)}-${dateKeyLabel(rows[rows.length - 1].date)} · ${formatBytes(totals.total)} saved`
    : "waiting for data";

  const pad = { top: 34, right: 24, bottom: 50, left: 58 };
  const plotWidth = width - pad.left - pad.right;
  const plotHeight = height - pad.top - pad.bottom;

  if (!rows.length) {
    ctx.fillStyle = "rgba(169,164,148,0.85)";
    ctx.font = "14px system-ui";
    ctx.fillText("Waiting for save history", pad.left, height / 2);
    return;
  }

  const maxTotal = Math.max(1, ...rows.map((row) => row.total));
  const maxSegmentBytes = Math.max(1, ...rows.flatMap((row) => row.segments.map((segment) => segment.bytes)));
  const ticks = [0, maxTotal / 2, maxTotal];
  const slot = plotWidth / rows.length;
  const barWidth = Math.max(5, Math.min(18, slot * 0.62));
  const baseline = pad.top + plotHeight;

  ctx.save();
  ctx.font = "11px system-ui";
  ctx.textBaseline = "middle";
  ticks.forEach((tick) => {
    const y = baseline - (tick / maxTotal) * plotHeight;
    ctx.strokeStyle = tick === 0 ? "rgba(242,239,229,0.2)" : "rgba(242,239,229,0.1)";
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(pad.left, y);
    ctx.lineTo(width - pad.right, y);
    ctx.stroke();
    ctx.fillStyle = "rgba(169,164,148,0.9)";
    ctx.textAlign = "right";
    ctx.fillText(formatBytes(tick), pad.left - 10, y);
  });

  ctx.textAlign = "left";
  ctx.textBaseline = "middle";
  ctx.fillStyle = "#8fd694";
  roundRect(ctx, pad.left, 9, 16, 8, 4);
  ctx.fill();
  ctx.fillStyle = "rgba(169,164,148,0.9)";
  ctx.fillText("processed", pad.left + 22, 13);
  ctx.fillStyle = "rgba(102,217,232,0.34)";
  roundRect(ctx, pad.left + 92, 9, 16, 8, 4);
  ctx.fill();
  ctx.strokeStyle = "rgba(102,217,232,0.72)";
  ctx.setLineDash([4, 4]);
  roundRect(ctx, pad.left + 92, 9, 16, 8, 4);
  ctx.stroke();
  ctx.setLineDash([]);
  ctx.fillStyle = "rgba(169,164,148,0.9)";
  ctx.fillText("pending", pad.left + 114, 13);
  ctx.fillStyle = "#f0bc62";
  roundRect(ctx, pad.left + 176, 9, 16, 8, 4);
  ctx.fill();
  ctx.fillStyle = "rgba(169,164,148,0.9)";
  ctx.fillText("failed", pad.left + 198, 13);

  const pulse = 0.48 + 0.32 * Math.sin(Date.now() / 170);
  rows.forEach((row, index) => {
    const barX = pad.left + index * slot + (slot - barWidth) / 2;
    const fullHeight = row.total ? Math.max(2, (row.total / maxTotal) * plotHeight) : 0;
    let y = baseline;
    ctx.fillStyle = "rgba(242,239,229,0.055)";
    roundRect(ctx, barX, baseline - Math.max(2, fullHeight), barWidth, Math.max(2, fullHeight), Math.min(5, barWidth / 2));
    ctx.fill();
    row.segments.forEach((segment, segmentIndex) => {
      const segmentHeight = (segment.bytes / maxTotal) * plotHeight;
      const segmentTop = y - segmentHeight;
      y = drawStackSegment(
        ctx,
        barX,
        y,
        barWidth,
        segmentHeight,
        segmentColor(segment.status, segment.bytes, maxSegmentBytes),
        {
          dashed: segment.status === "pending",
        }
      );
      saveLoadHitRegions.push({
        x: barX - 2,
        y: Math.min(segmentTop, y) - 2,
        width: barWidth + 4,
        height: Math.max(4, Math.abs(segmentHeight) + 4),
        date: row.date,
        name: segment.name,
        bytes: segment.bytes,
        status: segment.status,
        source: segment.source,
      });
    });

    if (row.active && fullHeight > 0) {
      ctx.save();
      ctx.globalAlpha = pulse;
      ctx.strokeStyle = "#66d9e8";
      ctx.lineWidth = 2;
      ctx.setLineDash([4, 4]);
      roundRect(ctx, barX - 3, baseline - fullHeight - 4, barWidth + 6, fullHeight + 8, 7);
      ctx.stroke();
      ctx.restore();
    }

    const showLabel = index === 0 || index === rows.length - 1 || index % 5 === 4 || row.active;
    if (showLabel) {
      const x = barX + barWidth / 2;
      ctx.fillStyle = row.active ? "#66d9e8" : "rgba(169,164,148,0.86)";
      ctx.font = row.active ? "700 10px system-ui" : "10px system-ui";
      ctx.textAlign = "center";
      ctx.textBaseline = "alphabetic";
      ctx.fillText(dateKeyLabel(row.date), x, height - 14);
    }
  });
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
  if (batch.active === true) {
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
    const rawLevel = (event.level || "info").toLowerCase();
    const kind = (event.outcome_kind || "").toLowerCase();
    const eventText = String(event.message || "").toLowerCase();
    let level = rawLevel;
    let label = rawLevel;
    if (kind === "read_back_warning" || eventText.includes("read-back:")) {
      level = "search";
      label = "search";
    } else if (
      kind === "retry" ||
      eventText.includes("-> retry") ||
      eventText.includes("requested regeneration")
    ) {
      level = "retry";
      label = "retry";
    } else if (kind === "self_heal_queued") {
      level = "heal";
      label = "heal";
    } else if (rawLevel === "error") {
      level = "failure";
      label = "failure";
    }
    const time = document.createElement("time");
    time.textContent = timeLabel(event.timestamp);
    const badge = document.createElement("span");
    badge.className = `event-level ${level}`;
    badge.textContent = label;
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
  if (frontier.mode === "on_demand_only") {
    els.selfHealFrontierCard.classList.add(frontier.state === "active" ? "ready" : "unknown");
    els.selfHealFrontierState.textContent = frontier.state === "active" ? "active" : "standby";
  } else if (frontier.ok === true) {
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
    frontier.mode === "on_demand_only" ? "guard only" : null,
    frontier.incidents_started != null ? `${intValue(frontier.incidents_started)} starts` : null,
    frontier.checked_at ? `checked ${ageLabel(frontier.checked_at)}` : null,
    frontier.cached ? "cached" : null,
    frontier.missing_exec_options && frontier.missing_exec_options.length
      ? `missing ${frontier.missing_exec_options.join(", ")}`
      : null,
    failure.failure_class || frontier.error || null,
  ].filter(Boolean);
  els.selfHealFrontierDetail.textContent = frontierDetails.join(" · ") || "guard state";

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

function knowledgeColor(index) {
  return KNOWLEDGE_COLORS[index % KNOWLEDGE_COLORS.length];
}

function knowledgeMetric(row, mode = knowledgeMixMode) {
  return mode === "pages" ? intValue(row.pages) : intValue(row.bytes);
}

function knowledgeMetricLabel(value, mode = knowledgeMixMode) {
  return mode === "pages" ? `${intValue(value).toLocaleString()} pages` : formatBytes(value);
}

function sortedKnowledgeCategories(categories, mode = knowledgeMixMode) {
  return [...categories].sort((a, b) =>
    knowledgeMetric(b, mode) - knowledgeMetric(a, mode) || fmt(a.label || a.id).localeCompare(fmt(b.label || b.id))
  );
}

function donutSegments(categories, totalValue, mode = knowledgeMixMode) {
  if (!totalValue) return [];
  const top = categories.slice(0, 7);
  const otherValue = categories.slice(7).reduce((sum, row) => sum + knowledgeMetric(row, mode), 0);
  const segments = top.map((row, index) => ({
    label: fmt(row.label || row.id),
    value: knowledgeMetric(row, mode),
    share: knowledgeMetric(row, mode) / totalValue,
    color: knowledgeColor(index),
  }));
  if (otherValue > 0) {
    segments.push({
      label: "Other",
      value: otherValue,
      share: otherValue / totalValue,
      color: "rgba(169,164,148,0.72)",
    });
  }
  return segments;
}

function drawKnowledgeDonut(canvas, categories, totalValue, mode = knowledgeMixMode) {
  const ctx = canvas.getContext("2d");
  const width = canvas.clientWidth || 220;
  const height = Number(canvas.dataset.baseHeight || canvas.getAttribute("height") || 240);
  canvas.dataset.baseHeight = String(height);
  canvas.style.height = `${height}px`;
  const dpr = window.devicePixelRatio || 1;
  canvas.width = Math.floor(width * dpr);
  canvas.height = Math.floor(height * dpr);
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, width, height);

  const radius = Math.max(52, Math.min(width, height) * 0.36);
  const lineWidth = Math.max(18, radius * 0.26);
  const cx = width / 2;
  const cy = height / 2;

  ctx.save();
  ctx.lineWidth = lineWidth;
  ctx.lineCap = "butt";
  ctx.strokeStyle = "rgba(242,239,229,0.08)";
  ctx.beginPath();
  ctx.arc(cx, cy, radius, 0, Math.PI * 2);
  ctx.stroke();

  let start = -Math.PI / 2;
  donutSegments(categories, totalValue, mode).forEach((segment) => {
    const end = start + segment.share * Math.PI * 2;
    ctx.strokeStyle = segment.color;
    ctx.beginPath();
    ctx.arc(cx, cy, radius, start, end);
    ctx.stroke();
    start = end;
  });
  ctx.restore();
}

function renderKnowledgeMix(knowledgeMix) {
  const data = knowledgeMix || {};
  latestKnowledgeMix = data;
  const categories = Array.isArray(data.categories) ? data.categories : [];
  const totalPages = intValue(data.total_pages);
  const totalBytes = intValue(data.total_bytes);
  const sorted = sortedKnowledgeCategories(categories);
  const totalValue = knowledgeMixMode === "pages" ? totalPages : totalBytes;
  const top = sorted.slice(0, 6);

  els.knowledgeCaption.textContent = categories.length
    ? `${categories.length} areas · ${knowledgeMetricLabel(totalValue)} by ${knowledgeMixMode}`
    : "waiting";
  els.knowledgeTotal.textContent = totalPages ? totalPages.toLocaleString() : "--";
  els.knowledgePages.textContent = totalPages.toLocaleString();
  els.knowledgeSize.textContent = formatBytes(totalBytes);
  els.knowledgeCategories.textContent = String(categories.length);
  els.knowledgeModeButtons.forEach((button) => {
    button.classList.toggle("active", button.dataset.knowledgeMode === knowledgeMixMode);
  });
  drawKnowledgeDonut(els.knowledgeChart, sorted, totalValue);

  els.knowledgeBars.innerHTML = "";
  if (!top.length) {
    els.knowledgeBars.innerHTML = "<div class=\"self-heal-empty\">No pages indexed yet.</div>";
    return;
  }

  top.forEach((row, index) => {
    const value = knowledgeMetric(row);
    const share = totalValue ? value / totalValue : 0;
    const item = document.createElement("div");
    item.className = "knowledge-bar";
    item.style.setProperty("--bar-color", knowledgeColor(index));
    const topLine = document.createElement("div");
    topLine.className = "knowledge-bar-top";
    const label = document.createElement("strong");
    label.textContent = fmt(row.label || row.id);
    const pct = document.createElement("span");
    pct.textContent = shareLabel(share);
    topLine.append(label, pct);

    const track = document.createElement("div");
    track.className = "knowledge-bar-track";
    const fill = document.createElement("span");
    fill.style.width = `${Math.max(2, share * 100)}%`;
    track.appendChild(fill);

    const meta = document.createElement("small");
    meta.textContent = `${formatBytes(row.bytes)} · ${intValue(row.pages).toLocaleString()} pages`;
    item.title = Array.isArray(row.samples) && row.samples.length
      ? `${fmt(row.label || row.id)} · ${row.samples.join(", ")}`
      : fmt(row.label || row.id);
    item.append(topLine, track, meta);
    els.knowledgeBars.appendChild(item);
  });
}

function healthPercent(value) {
  return numeric(value) ? shareLabel(value) : "--";
}

function renderHealth(health) {
  const data = health || {};
  const coverage = data.coverage || {};
  const capture = data.capture || {};
  const integrity = data.memory_integrity || {};
  const cofire = data.cofire || {};
  const derived = data.derived || {};
  const readBack = data.read_back || {};
  const queues = data.queues || {};
  const convergence = data.convergence || {};
  const sensitivity = coverage.sensitivity || {};
  const rawFiles = intValue(capture.raw_files);
  const claimed = intValue(capture.claimed_raw_files);
  const checked = intValue(readBack.checked);
  const passed = intValue(readBack.passed);

  const convergenceBits = convergence.status === "ok"
    ? [
        `${intValue(convergence.actionable).toLocaleString()} active`,
        `${intValue(convergence.quarantined).toLocaleString()} quarantined`,
        `${intValue(convergence.human_required).toLocaleString()} human`,
      ]
    : [];
  els.healthCaption.textContent = rawFiles
    ? `${claimed.toLocaleString()} / ${rawFiles.toLocaleString()} raw claimed${convergenceBits.length ? ` · ${convergenceBits.join(" · ")}` : ""}`
    : (convergenceBits.length ? convergenceBits.join(" · ") : "waiting");
  els.healthSummaryCoverage.textContent = healthPercent(coverage.summary_coverage);
  els.healthCapture.textContent = healthPercent(numeric(integrity.capture_rate) ? integrity.capture_rate : capture.claim_coverage);
  els.healthReadback.textContent = checked
    ? `${passed.toLocaleString()}/${checked.toLocaleString()}`
    : "--";
  els.healthSensitive.textContent = intValue(sensitivity.high).toLocaleString();
  els.healthDupes.textContent = intValue(queues.duplicate_candidates).toLocaleString();
  els.healthGolden.textContent = intValue(derived.claims).toLocaleString();
  els.healthGoldenSet.textContent = intValue(derived.golden || queues.search_golden).toLocaleString();
  els.healthRetention.textContent = intValue(derived.retention_pages || cofire.edges).toLocaleString();
  els.healthDistill.textContent = intValue(derived.distill_rows).toLocaleString();
}

const MODEL_ROLE_LABELS = {
  ingest: "Ingest",
  audit: "Audit",
  improve: "Improve",
  gate: "Gate",
  rewrite: "Rewrite",
  embed: "Embed",
  rerank: "Rerank",
  installed: "Installed",
  configured: "Configured",
};

function modelStatusRank(row) {
  const order = { loaded: 0, missing: 1, ready: 2, external: 3, unknown: 4 };
  return order[row.status] ?? 9;
}

function modelRoleLabel(role) {
  return MODEL_ROLE_LABELS[role] || fmt(role);
}

function renderModelStatus(modelStatus) {
  const data = modelStatus || {};
  const summary = data.summary || {};
  const models = Array.isArray(data.models) ? [...data.models] : [];
  const installed = intValue(summary.installed);
  const loaded = intValue(summary.loaded);
  const configured = intValue(summary.configured);
  const missing = intValue(summary.missing);

  els.modelInstalled.textContent = String(installed);
  els.modelLoaded.textContent = String(loaded);
  els.modelConfigured.textContent = String(configured);
  els.modelMissing.textContent = String(missing);
  els.modelCaption.textContent = data.available
    ? `${formatBytes(summary.loaded_size_bytes)} loaded · ${formatBytes(summary.installed_size_bytes)} installed`
    : shortName(data.error || "Ollama offline");

  els.modelGrid.innerHTML = "";
  if (!models.length) {
    els.modelGrid.innerHTML = "<div class=\"self-heal-empty\">No model records yet.</div>";
    return;
  }

  models
    .sort((a, b) => {
      const configuredRank = Number(Boolean(b.configured)) - Number(Boolean(a.configured));
      if (configuredRank !== 0) return configuredRank;
      return modelStatusRank(a) - modelStatusRank(b) || fmt(a.name).localeCompare(fmt(b.name));
    })
    .forEach((row) => {
      const details = row.details || {};
      const roles = Array.isArray(row.roles) ? row.roles : [];
      const item = document.createElement("div");
      item.className = `model-row ${fmt(row.status, "unknown")}`;

      const main = document.createElement("div");
      main.className = "model-main";
      const state = document.createElement("span");
      state.className = "model-state";
      const stateDot = document.createElement("span");
      state.append(stateDot, document.createTextNode(fmt(row.status, "unknown")));
      const name = document.createElement("strong");
      name.className = "model-name";
      name.textContent = fmt(row.name);
      main.append(state, name);

      const roleWrap = document.createElement("div");
      roleWrap.className = "model-roles";
      const roleList = roles.length ? roles : row.installed ? ["installed"] : ["configured"];
      roleList.forEach((role) => {
        const chip = document.createElement("span");
        chip.className = `model-role role-${String(role).replace(/[^a-z0-9_-]/gi, "-").toLowerCase()}`;
        chip.textContent = modelRoleLabel(role);
        roleWrap.appendChild(chip);
      });

      const metaPieces = [];
      if (row.size_bytes) metaPieces.push(`disk ${formatBytes(row.size_bytes)}`);
      if (row.loaded_size_bytes) metaPieces.push(`loaded ${formatBytes(row.loaded_size_bytes)}`);
      if (row.context_length !== null && row.context_length !== undefined && Number(row.context_length) > 0) {
        metaPieces.push(`${Number(row.context_length).toLocaleString()} ctx`);
      }
      if (details.parameter_size) metaPieces.push(details.parameter_size);
      if (details.quantization_level) metaPieces.push(details.quantization_level);
      if (details.format) metaPieces.push(details.format);
      if (row.expires_at) metaPieces.push(`until ${timeLabel(row.expires_at)}`);
      const meta = document.createElement("div");
      meta.className = "model-meta";
      meta.textContent = metaPieces.join(" · ") || "configured outside local Ollama";

      item.append(main, roleWrap, meta);
      els.modelGrid.appendChild(item);
    });
}

function renderModelLab(lab) {
  const policy = lab.policy || {};
  const roles = policy.roles || {};
  const candidates = Array.isArray(lab.candidates) ? lab.candidates : [];
  const canaries = policy.canaries || {};
  if (els.modelLabCaption) els.modelLabCaption.textContent = lab.status === "ok" ? `updated ${fmt(policy.updated_at, "now")}` : fmt(lab.status, "unavailable");
  if (els.modelLabRoles) els.modelLabRoles.textContent = Object.keys(roles).length;
  if (els.modelLabReplays) els.modelLabReplays.textContent = fmt(lab.replay_cases, 0);
  if (els.modelLabCandidates) els.modelLabCandidates.textContent = candidates.length;
  if (els.modelLabCanaries) els.modelLabCanaries.textContent = Object.keys(canaries).length;
  if (!els.modelLabGrid) return;
  els.modelLabGrid.innerHTML = "";
  Object.entries(roles).forEach(([role, selected]) => {
    const item = document.createElement("div");
    item.className = "model-row loaded";
    const main = document.createElement("div");
    main.className = "model-main";
    const name = document.createElement("strong");
    name.className = "model-name";
    name.textContent = role.replaceAll("_", " ");
    const meta = document.createElement("span");
    meta.className = "model-meta";
    meta.textContent = `${fmt(selected.model, "--")} · ${fmt(selected.effort, "--")}`;
    main.append(name, meta);
    const state = document.createElement("span");
    state.className = "model-state";
    state.textContent = canaries[role] ? "CANARY" : "ACTIVE";
    item.append(main, state);
    els.modelLabGrid.append(item);
  });
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

function blockerSummaryText(summary) {
  const top = summary && Array.isArray(summary.top) ? summary.top : [];
  if (!top.length) return "";
  const labels = {
    dev_improved: "dev no-gain",
    holdout_score_ok: "holdout score",
    holdout_recall_ok: "holdout recall",
    holdout_waste_ok: "waste",
    latency_ok: "latency",
  };
  return top
    .slice(0, 3)
    .map((item) => `${labels[item.name] || item.name} ${item.count}`)
    .join(", ");
}

function renderRecallImprovement(lab) {
  const data = lab || {};
  const active = data.active || null;
  const latest = data.latest || null;
  const status = String(data.status || "quiet");
  const configuredModels = Array.isArray(data.models) ? data.models : [];
  const schedule = data.schedule || {};
  const scheduleDecision = schedule.last_decision || {};
  const scheduleStatus = scheduleDecision.dry_run && schedule.last_status
    ? `dry-run ${schedule.last_status}`
    : schedule.last_status;
  const models = latest && Array.isArray(latest.models) ? latest.models : configuredModels;
  const best = latest && latest.best ? latest.best : null;
  const bestDev = best && best.dev ? best.dev : {};
  const bestHoldout = best && best.holdout ? best.holdout : {};
  const baseline = latest && latest.baseline ? latest.baseline : {};
  const baselineDev = baseline.dev || {};
  const baselineHoldout = baseline.holdout || {};
  const activeOverrides = active && active.overrides ? active.overrides : {};
  const statusKind = status === "active" || status === "applied"
    ? "success"
    : status === "rejected" || status === "blocked" || status === "error" || status === "frontier_rejected"
      ? "warn"
      : "info";

  const scheduleLabel = scheduleStatus ? ` · schedule ${scheduleStatus}` : "";
  els.recallLabCaption.textContent = latest
    ? `${(data.history || []).length} runs${scheduleLabel}`
    : schedule.last_checked_at
      ? `schedule ${fmt(scheduleStatus)}`
      : "quiet";
  els.recallLabState.textContent = status;
  els.recallLabState.className = `self-heal-badge ${statusKind}`;
  els.recallLabLatest.textContent = latest
    ? `${fmt(latest.status)} · ${fmt(latest.run_id)}`
    : "No runs yet";
  const detailParts = latest
    ? [fmt(latest.reason)]
    : ["Run llm-wiki recall-improve run to start the local proposal tournament."];
  if (latest && latest.frontier_audit_recommended) {
    const audit = latest.frontier_audit || {};
    detailParts.push(`frontier ${fmt(audit.decision || "recommended")}`);
  }
  if (latest && latest.live_telemetry && numeric(latest.live_telemetry.episodes)) {
    detailParts.push(`${latest.live_telemetry.episodes} live episodes`);
  }
  const latestBlockers = blockerSummaryText(latest && latest.candidate_blockers);
  if (latestBlockers) {
    const label = latest.status === "applied" || latest.status === "shadow_pass"
      ? "other candidates blocked by"
      : "blocked by";
    detailParts.push(`${label} ${latestBlockers}`);
  }
  if (schedule.last_checked_at) detailParts.push(`checked ${ageLabel(schedule.last_checked_at)}`);
  els.recallLabDetail.textContent = detailParts.filter(Boolean).join(" · ");
  els.recallLabActive.textContent = active ? "on" : "off";
  const devScore = numeric(bestDev.score) ? bestDev.score : baselineDev.score;
  const holdoutScore = numeric(bestHoldout.score) ? bestHoldout.score : baselineHoldout.score;
  els.recallLabDev.textContent = numeric(devScore) ? devScore.toFixed(3) : "--";
  els.recallLabHoldout.textContent = numeric(holdoutScore) ? holdoutScore.toFixed(3) : "--";
  els.recallLabModels.textContent = models.length ? String(models.length) : "--";
  const overrideText = Object.entries(activeOverrides)
    .map(([key, value]) => `${key}=${value}`)
    .join(" · ");
  const policyParts = [
    overrideText ? `active policy: ${overrideText}` : "active policy: baseline",
  ];
  if (schedule.last_run_id) {
    policyParts.push(`scheduler: ${fmt(scheduleStatus)} ${shortName(schedule.last_run_id)}`);
  } else if (scheduleStatus) {
    policyParts.push(`scheduler: ${fmt(scheduleStatus)}`);
  }
  els.recallLabPolicy.textContent = policyParts.join(" · ");

  els.recallLabFeed.innerHTML = "";
  const history = Array.isArray(data.history) ? [...data.history].slice(-5).reverse() : [];
  if (!history.length) {
    const empty = document.createElement("div");
    empty.className = "self-heal-empty";
    empty.textContent = "No improvement runs yet.";
    els.recallLabFeed.appendChild(empty);
    return;
  }
  history.forEach((item) => {
    const row = document.createElement("div");
    row.className = "event";
    const time = document.createElement("time");
    time.textContent = timeLabel(item.ts);
    const badge = document.createElement("span");
    const itemStatus = String(item.status || "unknown");
    badge.className = `event-level ${
      itemStatus === "applied" ? "success"
        : itemStatus === "shadow_pass" || itemStatus === "pending_frontier_review" ? "info"
          : "warn"
    }`;
    badge.textContent = itemStatus;
    const message = document.createElement("span");
    message.className = "event-message";
    const rowBest = item.best || {};
    const proposal = rowBest.proposal || {};
    const score = rowBest.dev && numeric(rowBest.dev.score) ? ` · dev ${rowBest.dev.score.toFixed(3)}` : "";
    const audit = item.frontier_audit || {};
    const auditText = audit.decision ? ` · frontier ${audit.decision}` : "";
    const blockers = blockerSummaryText(item.candidate_blockers);
    const blockerLabel = itemStatus === "applied" || itemStatus === "shadow_pass"
      ? "other candidates blocked by"
      : "blocked by";
    const blockerText = blockers ? ` · ${blockerLabel} ${blockers}` : "";
    message.textContent = `${fmt(proposal.summary || item.reason)}${score}${auditText}${blockerText}`;
    row.append(time, badge, message);
    els.recallLabFeed.appendChild(row);
  });
}

function render(snapshot) {
  const status = snapshot.status || {};
  status.local_consensus = snapshot.local_consensus || status.local_consensus || {};
  status.frontier_repair = snapshot.frontier_repair || status.frontier_repair || {};
  const metrics = snapshot.metrics || [];
  const batch = status.batch || {};
  const ollama = snapshot.ollama || {};
  const modelStatus = snapshot.model_status || {};
  const modelSummary = modelStatus.summary || {};
  const models = ollama.models || [];
  const model = models.find((item) => !String(item.name || item.model || "").includes("embed")) || models[0] || {};

  setState(status.state);
  els.pending.textContent = fmt(status.pending);
  els.pendingSub.textContent = `updated ${timeLabel(status.updated_at)}`;
  els.stage.textContent = fmt(status.stage);
  els.raw.textContent = shortName(status.current_raw || "no active raw");
  els.batch.textContent = batch.total ? `${batch.index || 0}/${batch.total}` : "--";
  els.batchSub.textContent = batch.total ? `${batch.succeeded || 0} ok / ${batch.failed || 0} miss` : "waiting";
  els.ollama.textContent = modelStatus.available || ollama.available ? "online" : "offline";
  els.ollamaSub.textContent = modelSummary.installed !== undefined
    ? `${intValue(modelSummary.loaded)} loaded · ${intValue(modelSummary.installed)} installed`
    : model.name || model.model || "no model";
  els.currentRaw.textContent = status.current_raw ? shortName(status.current_raw) : "waiting";
  els.currentOp.textContent = status.current_op ? fmt(status.current_op) : fmt(status.stage || "idle");
  renderWorkStatus(status);
  renderLlm(status.llm, status);
  const consensus = status.local_consensus || {};
  const consensusSummary = consensus.summary || {};
  const decisionSummary = consensusSummary.decisions || {};
  const evaluationSummary = (consensusSummary.evaluation || {}).decisions || {};
  const policyCounts = ((status.decision_policies || {}).counts) || {};
  const activeModels = (consensus.activities || [])
    .map((item) => [item.role, item.model].filter(Boolean).join(" · "))
    .filter(Boolean);
  els.localConsensus.textContent = consensus.active
    ? `${intValue(consensus.count)} active · ${activeModels.join(" · ")}`
    : `${intValue(decisionSummary.total)} routine · ${intValue(decisionSummary.pair_agreement)} pair · ${intValue(decisionSummary.tie_break_used)} tie · ${intValue(decisionSummary.unresolved_quarantine)} quarantined · ${intValue(evaluationSummary.total)} eval · ${intValue(policyCounts.shadow)} shadow / ${intValue(policyCounts.enabled)} enabled`;
  const repair = status.frontier_repair || {};
  const repairSummary = repair.summary || {};
  const activeRepair = repair.active_incident || ((repair.process_activity || {}).latest) || {};
  els.frontierRepair.textContent = repair.active
    ? `active · ${[activeRepair.component || activeRepair.kind, activeRepair.status, activeRepair.model].filter(Boolean).join(" · ")}`
    : `${intValue(repairSummary.starts_24h)} starts / 24h · ${intValue(repairSummary.total)} total`;
  els.currentJob.textContent = status.current_job_id ? fmt(status.current_job_id) : "none";
  els.lastSuccess.textContent = status.last_success
    ? `${shortName(status.last_success.raw)} -> ${shortName(lastSuccessTargets(status.last_success) || "none")}`
    : "--";
  updateStageFlow(status.stage);
  renderSelfHeal(snapshot.self_heal || {});
  renderRecall(snapshot.recall || {});
  renderRecallImprovement(snapshot.recall_improvement || {});
  renderSaveHistory(snapshot.save_history || {});
  renderKnowledgeMix(snapshot.knowledge_mix || {});
  renderHealth(snapshot.health || {});
  renderModelStatus(modelStatus);
  renderModelLab(snapshot.model_lab || {});
  renderEvents(snapshot.events || []);
  drawLineChart(els.pendingChart, snapshot.save_history || {}, status);
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

els.knowledgeModeButtons.forEach((button) => {
  button.addEventListener("click", () => {
    knowledgeMixMode = button.dataset.knowledgeMode || "size";
    renderKnowledgeMix(latestKnowledgeMix);
  });
});

els.pendingChart.addEventListener("mousemove", handleSaveLoadHover);
els.pendingChart.addEventListener("mouseleave", hideSaveLoadTooltip);

refresh();
setInterval(refresh, 1000);
window.addEventListener("resize", refresh);
