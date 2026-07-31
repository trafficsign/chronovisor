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
  lanShareButton: document.getElementById("lan-share-button"),
  workOverview: document.getElementById("work-overview"),
  workSummary: document.getElementById("work-summary"),
  workUpdated: document.getElementById("work-updated"),
  workDetail: document.getElementById("work-detail"),
  decisionElapsed: document.getElementById("decision-elapsed"),
  decisionContext: document.getElementById("decision-context"),
  decisionBadge: document.getElementById("decision-badge"),
  decisionModelCalls: document.getElementById("decision-model-calls"),
  decisionQuorum: document.getElementById("decision-quorum"),
  decisionMutation: document.getElementById("decision-mutation"),
  decisionOutcomeReason: document.getElementById("decision-outcome-reason"),
  decisionOutcomeData: document.getElementById("decision-outcome-data"),
  decisionOutcomeNext: document.getElementById("decision-outcome-next"),
  decisionTraceCaption: document.getElementById("decision-trace-caption"),
  decisionOverallSteps: document.getElementById("decision-overall-steps"),
  decisionTransitionState: document.getElementById("decision-transition-state"),
  decisionTransitionFeed: document.getElementById("decision-transition-feed"),
  decisionLanes: document.querySelectorAll("[data-decision-lane]"),
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
  pendingDeferred: document.getElementById("pending-deferred"),
  pendingEta: document.getElementById("pending-eta"),
  batchCaption: document.getElementById("batch-caption"),
  batchOk: document.getElementById("batch-ok"),
  batchDeferred: document.getElementById("batch-deferred"),
  batchContinued: document.getElementById("batch-continued"),
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
  librarianPanel: document.getElementById("librarian-panel"),
  librarianState: document.getElementById("librarian-state"),
  librarianHeadline: document.getElementById("librarian-headline"),
  librarianDetail: document.getElementById("librarian-detail"),
  librarianGeneration: document.getElementById("librarian-generation"),
  librarianSweptGeneration: document.getElementById("librarian-swept-generation"),
  librarianUid: document.getElementById("librarian-uid"),
  librarianUidBar: document.getElementById("librarian-uid-bar"),
  librarianClassification: document.getElementById("librarian-classification"),
  librarianClassificationBar: document.getElementById("librarian-classification-bar"),
  librarianLinks: document.getElementById("librarian-links"),
  librarianLinksBar: document.getElementById("librarian-links-bar"),
  librarianMigration: document.getElementById("librarian-migration"),
  librarianMigrationBar: document.getElementById("librarian-migration-bar"),
  librarianSweep: document.getElementById("librarian-sweep"),
  librarianSweepBar: document.getElementById("librarian-sweep-bar"),
  librarianAuthority: document.getElementById("librarian-authority"),
  librarianQuality: document.getElementById("librarian-quality"),
  librarianCollectionCount: document.getElementById("librarian-collection-count"),
  librarianAssignment: document.getElementById("librarian-assignment"),
  librarianCrosswalk: document.getElementById("librarian-crosswalk"),
  librarianTopShare: document.getElementById("librarian-top-share"),
  librarianReviewQueue: document.getElementById("librarian-review-queue"),
  librarianSplitProposals: document.getElementById("librarian-split-proposals"),
  librarianRollout: document.getElementById("librarian-rollout"),
  librarianSoak: document.getElementById("librarian-soak"),
  librarianRecovery: document.getElementById("librarian-recovery"),
  librarianEvidenceStatus: document.getElementById("librarian-evidence-status"),
  librarianEvidenceFixture: document.getElementById("librarian-evidence-fixture"),
  librarianEvidenceRecall: document.getElementById("librarian-evidence-recall"),
  librarianEvidenceExternal: document.getElementById("librarian-evidence-external"),
  librarianEvidenceHold: document.getElementById("librarian-evidence-hold"),
  librarianEvidenceResource: document.getElementById("librarian-evidence-resource"),
  librarianEvidenceStorage: document.getElementById("librarian-evidence-storage"),
  librarianEvidenceAuthority: document.getElementById("librarian-evidence-authority"),
  librarianEvidenceUpdate: document.getElementById("librarian-evidence-update"),
  librarianQueue: document.getElementById("librarian-queue"),
  librarianFlow: document.getElementById("librarian-flow"),
  librarianReceipts: document.getElementById("librarian-receipts"),
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
  healthReplay: document.getElementById("health-replay"),
  healthDeadman: document.getElementById("health-deadman"),
  healthQuality: document.getElementById("health-quality"),
  healthHolds: document.getElementById("health-holds"),
  healthProvisional: document.getElementById("health-provisional"),
  healthLedger: document.getElementById("health-ledger"),
  healthResearchRuns: document.getElementById("health-research-runs"),
  healthResearchClaims: document.getElementById("health-research-claims"),
  healthResearchTrace: document.getElementById("health-research-trace"),
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
  recallFieldSummary: document.getElementById("recall-field-summary"),
  recallFieldState: document.getElementById("recall-field-state"),
  recallFieldCounts: document.getElementById("recall-field-counts"),
  recallFieldQuality: document.getElementById("recall-field-quality"),
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

const STAGE_METRIC_LABELS = {
  "local-consensus-review": "Local review",
  "local-regenerate": "Local retry",
  "frontier-review": "Frontier review",
  "frontier-regenerate": "Frontier retry",
  authorization: "Authorize",
};

let saveHistoryMode = "daily";
let latestSaveHistory = null;
let selectedSaveDate = null;
let knowledgeMixMode = "size";
let latestKnowledgeMix = null;
let saveLoadHitRegions = [];

const KNOWLEDGE_COLORS = [
  "#828fff",
  "#3dd68c",
  "#e8b04b",
  "#c983f7",
  "#ff7e79",
  "#56a8ff",
  "#e0cd6d",
  "#4fc8b4",
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

function stageMetricLabel(value) {
  const raw = fmt(value, "idle").trim();
  const normalized = raw.toLowerCase();
  if (STAGE_METRIC_LABELS[normalized]) return STAGE_METRIC_LABELS[normalized];
  const readable = raw.replace(/[-_]+/g, " ");
  if (readable.length <= 18) return readable;
  return `${readable.slice(0, 17).trimEnd()}…`;
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
  if (numeric(row.files_failed) || numeric(row.files_deferred)) score += 2;
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

function setWorkState(stateKind) {
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
  setWorkState(stateKind);
}

const DECISION_LANE_PHASES = [
  "trigger",
  "load",
  "context",
  "generate",
  "validate",
  "vote",
];
const decisionTracePlayback = {
  initialized: false,
  request: "",
  seen: new Set(),
  queue: [],
  playing: false,
  timer: null,
  target: null,
  current: null,
  focus: null,
};

const decisionReducedMotion = window.matchMedia("(prefers-reduced-motion: reduce)");

function cloneDecisionTrace(trace) {
  return JSON.parse(JSON.stringify(trace || {}));
}

function reconcileDecisionSteps(container, steps, className, focusKey = "") {
  const existing = new Map(
    [...container.querySelectorAll(`.${className}`)].map((node) => [
      node.dataset.traceKey,
      node,
    ])
  );
  (Array.isArray(steps) ? steps : []).forEach((step) => {
    const key = fmt(step.key, "step");
    const item = existing.get(key) || document.createElement("span");
    item.dataset.traceKey = key;
    if (className === "decision-step") item.dataset.decisionOverallStep = key;
    if (className === "decision-lane-step") item.dataset.decisionLaneStep = key;
    item.className = `${className} ${fmt(step.status, "pending")}`;
    item.classList.toggle("trace-focus", key === focusKey);
    let label = item.querySelector("span");
    if (!label) {
      label = document.createElement("span");
      item.appendChild(label);
    }
    label.textContent = fmt(step.label, "Step");
    container.appendChild(item);
    existing.delete(key);
  });
  existing.forEach((node) => node.remove());
}

function decisionEventText(event) {
  const lane = event?.lane
    ? event.lane === "tie_break"
      ? "Tie-break"
      : `${event.lane.charAt(0).toUpperCase()}${event.lane.slice(1)}`
    : "System";
  const attempt = Number(event?.attempt || 0);
  const suffix = event?.phase === "repair" ? ` · retry ${attempt}` : "";
  return `${lane} · ${fmt(event?.label, event?.phase || "transition")}${suffix}`;
}

function renderDecisionTransitionFeed(events, focusEvent = null) {
  const rows = Array.isArray(events) ? events : [];
  const visible = rows.slice(-5);
  if (
    focusEvent
    && !visible.some((event) => event.event_id === focusEvent.event_id)
  ) {
    visible.unshift(focusEvent);
  }
  els.decisionTransitionFeed.textContent = "";
  visible.slice(-6).forEach((event) => {
    const item = document.createElement("li");
    item.className = "decision-transition-event";
    item.classList.toggle(
      "current",
      Boolean(focusEvent && event.event_id === focusEvent.event_id)
    );
    item.textContent = decisionEventText(event);
    item.title = `${timeLabel(event.timestamp)} · ${decisionEventText(event)}`;
    els.decisionTransitionFeed.appendChild(item);
  });
}

function decisionTraceBlank(target) {
  const trace = cloneDecisionTrace(target);
  trace.state = "active";
  trace.active = true;
  trace.summary = "Observed transition replay";
  trace.overall = (trace.overall || []).map((step) => ({
    ...step,
    status: "pending",
  }));
  trace.lanes = (trace.lanes || []).map((lane) => ({
    ...lane,
    state: "pending",
    result: "Waiting",
    detail: "Not started",
    phase: null,
    steps: (lane.steps || []).map((step) => ({ ...step, status: "pending" })),
  }));
  return trace;
}

function applyDecisionTransition(current, target, event) {
  if (!event || event.phase === "decision") return cloneDecisionTrace(target);
  const frame = cloneDecisionTrace(current || decisionTraceBlank(target));
  frame.state = "active";
  frame.active = true;
  frame.request_sha256 = target.request_sha256;
  frame.task_role = target.task_role;
  frame.started_at = target.started_at || frame.started_at;
  frame.updated_at = event.timestamp || target.updated_at;
  frame.context_tokens = target.context_tokens;
  frame.quorum_flow = target.quorum_flow;
  frame.artifact_replay = target.artifact_replay;
  frame.outcome = {
    kind: "active",
    reason: "Replaying observed local transition",
    data: "No synthetic progress",
    next: "Mutation stays locked",
    code: "observed_transition_replay",
  };

  const targetLane = (target.lanes || []).find((lane) => lane.key === event.lane);
  const lane = (frame.lanes || []).find((item) => item.key === event.lane);
  if (!lane || !targetLane) return frame;
  Object.assign(lane, {
    label: targetLane.label,
    model: targetLane.model,
    phase: event.phase,
  });

  if (event.kind === "session") {
    const failed = event.status === "error";
    lane.state = failed ? "error" : "done";
    lane.result = failed ? "Invalid vote" : "Valid vote";
    lane.detail = failed ? "Validation failed" : "Observed session complete";
    lane.steps = (targetLane.steps || []).map((step) => ({
      ...step,
      status:
        failed && step.key === "validate"
          ? "error"
          : failed && step.key === "vote"
            ? "skipped"
            : "done",
    }));
  } else {
    const phase = event.phase === "repair" ? "generate" : event.phase;
    const index = Math.max(0, DECISION_LANE_PHASES.indexOf(phase));
    lane.state = "active";
    lane.result = fmt(event.label, phase);
    lane.detail = event.phase === "repair"
      ? `JSON repair · attempt ${Number(event.attempt || 0) + 1}`
      : "Observed live transition";
    lane.steps = (targetLane.steps || []).map((step, stepIndex) => ({
      ...step,
      status: stepIndex < index ? "done" : stepIndex === index ? "active" : "pending",
    }));
  }

  const overallKey = fmt(event.overall_key, "");
  const overallIndex = (target.overall || []).findIndex(
    (step) => step.key === overallKey
  );
  frame.overall = (target.overall || []).map((step, index) => ({
    ...step,
    status:
      event.kind === "session" && step.key === "quorum"
        ? "active"
        : index < overallIndex
          ? "done"
          : index === overallIndex
            ? "active"
            : "pending",
  }));
  frame.summary = decisionEventText(event);
  return frame;
}

function renderDecisionTraceFrame(trace, focusEvent = null) {
  trace = trace || {};
  const outcome = trace.outcome || {};
  const traceState = String(trace.state || "idle");
  const request = String(trace.request_sha256 || "");
  const active = trace.active === true;
  els.decisionTraceCaption.textContent = request ? `Job ${request.slice(0, 8)}` : "Job --";
  els.decisionElapsed.textContent = trace.started_at
    ? `Elapsed ${compactDuration(Math.max(0, (Date.now() - parseMs(trace.started_at)) / 1000))}`
    : "Elapsed --";
  const contextTokens = Number(trace.context_tokens || 0);
  els.decisionContext.textContent = contextTokens
    ? `Context ${Math.round(contextTokens / 1024)}K`
    : "Context --";

  const overall = Array.isArray(trace.overall) ? trace.overall : [];
  reconcileDecisionSteps(
    els.decisionOverallSteps,
    overall,
    "decision-step",
    fmt(focusEvent?.overall_key, "")
  );

  const activeOverallIndex = overall.findIndex((step) => step.status === "active");
  const doneOverallCount = overall.filter((step) => step.status === "done").length;
  const overallPosition = activeOverallIndex >= 0 ? activeOverallIndex + 1 : doneOverallCount;
  const overallStage =
    (activeOverallIndex >= 0 ? overall[activeOverallIndex]?.label : null) ||
    [...overall].reverse().find((step) => step.status === "done")?.label ||
    "Waiting";

  const lanes = new Map(
    (Array.isArray(trace.lanes) ? trace.lanes : []).map((lane) => [lane.key, lane])
  );
  els.decisionLanes.forEach((element) => {
    const lane = lanes.get(element.dataset.decisionLane) || {};
    const laneState = fmt(lane.state, "pending");
    element.classList.remove("active", "done", "error", "skipped", "pending");
    element.classList.add(laneState);
    const role = element.querySelector(".decision-role strong");
    const model = element.querySelector(".decision-model");
    const steps = element.querySelector(".decision-lane-steps");
    const result = element.querySelector(".decision-lane-result");
    element.classList.toggle(
      "event-focus",
      Boolean(focusEvent && focusEvent.lane === element.dataset.decisionLane)
    );
    role.textContent = fmt(lane.label, element.dataset.decisionLane);
    model.textContent = fmt(lane.model, "not configured");
    reconcileDecisionSteps(
      steps,
      lane.steps,
      "decision-lane-step",
      focusEvent?.lane === element.dataset.decisionLane
        ? focusEvent.phase === "repair"
          ? "generate"
          : fmt(focusEvent.phase, "")
        : ""
    );
    const resultLabel =
      laneState === "pending"
        ? "WAITING"
        : laneState === "skipped"
          ? element.dataset.decisionLane === "tie_break"
            ? "STANDBY"
            : "NOT NEEDED"
          : laneState === "error"
            ? "INVALID"
            : laneState === "done"
              ? "VALID"
              : fmt(lane.result, laneState).toUpperCase();
    result.querySelector("strong").textContent = resultLabel;
    result.querySelector("span").textContent = fmt(lane.detail, "Not started");
  });

  if (request) {
    const stateKind = active
      ? "running"
      : traceState === "agreed" || traceState === "ready"
        ? "complete"
        : traceState === "quarantined"
          ? "warning"
          : "idle";
    const modelCalls = (Array.isArray(trace.lanes) ? trace.lanes : []).filter((lane) =>
      ["active", "done", "error"].includes(lane.state)
    ).length;
    const validVotes = (Array.isArray(trace.lanes) ? trace.lanes : []).filter(
      (lane) => lane.state === "done"
    ).length;
    const tieBreakUsed = (Array.isArray(trace.lanes) ? trace.lanes : []).some(
      (lane) => lane.key === "tie_break" && ["active", "done", "error"].includes(lane.state)
    );
    const quorumTarget = tieBreakUsed ? 3 : 2;
    const badge =
      traceState === "agreed"
        ? "APPROVED"
        : traceState === "quarantined"
          ? "HELD"
          : trace.artifact_replay
            ? "REPLAYED"
            : active
              ? tieBreakUsed
                ? "RESOLVING"
                : "WAITING"
              : "READY";
    els.workSummary.textContent = `${overallStage} · ${Math.min(overallPosition, 7)} / 7`;
    els.workDetail.textContent = fmt(trace.summary, "Local decision");
    els.workUpdated.textContent = `${fmt(trace.task_role, "routine")} · request ${request.slice(0, 16)}`;
    els.workOverview.dataset.outcomeKind = fmt(outcome.kind, "idle");
    els.workOverview.title = outcome.code ? `Reason code: ${outcome.code}` : "";
    els.decisionOutcomeReason.textContent = fmt(outcome.reason, "Local decision");
    els.decisionOutcomeData.textContent = fmt(outcome.data, "Input retained");
    els.decisionOutcomeNext.textContent = fmt(outcome.next, "Wait for completion");
    els.decisionBadge.textContent = badge;
    els.decisionModelCalls.textContent = String(trace.artifact_replay ? 0 : modelCalls);
    els.decisionQuorum.textContent = `${Math.min(validVotes, quorumTarget)} / ${quorumTarget}`;
    els.decisionMutation.textContent =
      traceState === "agreed" || traceState === "ready" ? "Ready" : traceState === "quarantined" ? "Held" : "Locked";
    setWorkState(stateKind);
  } else {
    els.decisionBadge.textContent = "WAITING";
    els.decisionModelCalls.textContent = "0";
    els.decisionQuorum.textContent = "0 / 2";
    els.decisionMutation.textContent = "Locked";
    els.workOverview.dataset.outcomeKind = "idle";
    els.workOverview.title = "";
    els.decisionOutcomeReason.textContent = "Waiting for local work";
    els.decisionOutcomeData.textContent = "No active decision";
    els.decisionOutcomeNext.textContent = "Starts automatically";
  }
}

function setDecisionTransitionState(trace, mode = "steady") {
  const events = Array.isArray(trace?.events) ? trace.events : [];
  const latest = events[events.length - 1];
  els.decisionTransitionState.classList.toggle(
    "catching-up",
    mode === "catching-up"
  );
  if (mode === "catching-up") {
    els.decisionTransitionState.textContent = `Catching up · ${decisionTracePlayback.queue.length + 1}`;
  } else if (trace?.active) {
    els.decisionTransitionState.textContent = latest
      ? `Live · ${fmt(latest.label, latest.phase)}`
      : "Live · observing";
  } else if (trace?.request_sha256) {
    els.decisionTransitionState.textContent = `Sealed · ${events.length} events`;
  } else {
    els.decisionTransitionState.textContent = "Live · waiting";
  }
}

function finishDecisionTracePlayback() {
  decisionTracePlayback.playing = false;
  decisionTracePlayback.focus = null;
  decisionTracePlayback.current = cloneDecisionTrace(decisionTracePlayback.target);
  renderDecisionTraceFrame(decisionTracePlayback.current);
  renderDecisionTransitionFeed(decisionTracePlayback.target?.events);
  setDecisionTransitionState(decisionTracePlayback.target);
}

function playNextDecisionTransition() {
  if (
    decisionReducedMotion.matches
    || document.visibilityState === "hidden"
    || decisionTracePlayback.queue.length === 0
  ) {
    decisionTracePlayback.queue = [];
    finishDecisionTracePlayback();
    return;
  }

  decisionTracePlayback.playing = true;
  const event = decisionTracePlayback.queue.shift();
  decisionTracePlayback.focus = event;
  decisionTracePlayback.current = applyDecisionTransition(
    decisionTracePlayback.current,
    decisionTracePlayback.target,
    event
  );
  renderDecisionTraceFrame(decisionTracePlayback.current, event);
  renderDecisionTransitionFeed(decisionTracePlayback.target?.events, event);
  setDecisionTransitionState(decisionTracePlayback.target, "catching-up");
  const backlog = decisionTracePlayback.queue.length;
  const delay = backlog > 8 ? 90 : backlog > 4 ? 160 : 420;
  decisionTracePlayback.timer = window.setTimeout(playNextDecisionTransition, delay);
}

function renderDecisionTrace(consensus) {
  const target = cloneDecisionTrace(consensus?.decision_trace || {});
  const request = String(target.request_sha256 || "");
  const events = (Array.isArray(target.events) ? target.events : []).filter(
    (event) => event && event.event_id
  );
  target.events = events;

  if (!decisionTracePlayback.initialized) {
    decisionTracePlayback.initialized = true;
    decisionTracePlayback.request = request;
    decisionTracePlayback.target = target;
    decisionTracePlayback.current = cloneDecisionTrace(target);
    events.forEach((event) => decisionTracePlayback.seen.add(event.event_id));
    renderDecisionTraceFrame(target);
    renderDecisionTransitionFeed(events);
    setDecisionTransitionState(target);
    return;
  }

  if (request !== decisionTracePlayback.request) {
    if (decisionTracePlayback.timer !== null) {
      window.clearTimeout(decisionTracePlayback.timer);
      decisionTracePlayback.timer = null;
    }
    decisionTracePlayback.request = request;
    decisionTracePlayback.target = target;
    decisionTracePlayback.seen = new Set(events.map((event) => event.event_id));
    decisionTracePlayback.queue = [...events];
    decisionTracePlayback.playing = false;
    decisionTracePlayback.current = decisionTraceBlank(target);
    if (
      request
      && events.length
      && !decisionReducedMotion.matches
      && document.visibilityState !== "hidden"
    ) {
      renderDecisionTraceFrame(decisionTracePlayback.current);
      renderDecisionTransitionFeed(events);
      playNextDecisionTransition();
    } else {
      finishDecisionTracePlayback();
    }
    return;
  }

  decisionTracePlayback.target = target;
  const unseen = events.filter(
    (event) => !decisionTracePlayback.seen.has(event.event_id)
  );
  unseen.forEach((event) => decisionTracePlayback.seen.add(event.event_id));
  decisionTracePlayback.queue.push(...unseen);
  if (decisionTracePlayback.queue.length && !decisionTracePlayback.playing) {
    playNextDecisionTransition();
  } else if (!decisionTracePlayback.playing) {
    decisionTracePlayback.current = cloneDecisionTrace(target);
    renderDecisionTraceFrame(target);
    renderDecisionTransitionFeed(events);
    setDecisionTransitionState(target);
  }
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
        status: ["processed", "pending", "deferred", "failed"].includes(segment.status) ? segment.status : "pending",
        source: fmt(segment.source, "raw"),
      }))
      .filter((segment) => segment.bytes > 0);
    if (!segments.length && intValue(day.raw_bytes)) {
      [
        ["processed", intValue(day.processed_bytes)],
        ["deferred", intValue(day.deferred_bytes)],
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
    const deferred = segments.filter((segment) => segment.status === "deferred").reduce((sum, segment) => sum + segment.bytes, 0);
    const failed = segments.filter((segment) => segment.status === "failed").reduce((sum, segment) => sum + segment.bytes, 0);
    const pending = Math.max(0, total - processed - deferred - failed);
    return {
      date: day.date,
      total,
      processed,
      deferred,
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
  if (status === "deferred") {
    return `hsl(274, 58%, ${46 + ratio * 22}%)`;
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
    ctx.strokeStyle = options.stroke || "rgba(130,143,255,0.72)";
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
  const statusLabel = region.status === "deferred" ? "semantic deferred" : region.status;
  meta.textContent = `${statusLabel} · ${region.source}`;
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
    acc.deferred += row.deferred;
    acc.failed += row.failed;
    return acc;
  }, { total: 0, processed: 0, pending: 0, deferred: 0, failed: 0 });

  els.pendingCurrent.textContent = formatBytes(totals.total);
  els.pendingDelta.textContent = formatBytes(totals.processed);
  els.pendingRate.textContent = formatBytes(totals.pending);
  els.pendingDeferred.textContent = formatBytes(totals.deferred);
  els.pendingEta.textContent = formatBytes(totals.failed);
  els.trendCaption.textContent = rows.length
    ? `${dateKeyLabel(rows[0].date)}-${dateKeyLabel(rows[rows.length - 1].date)} · ${formatBytes(totals.total)} saved`
    : "waiting for data";

  const pad = { top: 34, right: 24, bottom: 50, left: 58 };
  const plotWidth = width - pad.left - pad.right;
  const plotHeight = height - pad.top - pad.bottom;

  if (!rows.length) {
    ctx.fillStyle = "rgba(138,143,152,0.85)";
    ctx.font = "14px 'Geist Mono', ui-monospace, monospace";
    ctx.fillText("Waiting for save history", pad.left, height / 2);
    return;
  }

  const maxTotal = Math.max(1, ...rows.map((row) => row.total));
  const maxSegmentBytes = Math.max(1, ...rows.flatMap((row) => row.segments.map((segment) => segment.bytes)));
  // sqrt scale keeps spike days from flattening every other bar.
  const yScale = (bytes) => Math.sqrt(Math.max(0, bytes) / maxTotal) * plotHeight;
  const ticks = [0, maxTotal / 4, maxTotal];
  const slot = plotWidth / rows.length;
  const barWidth = Math.max(5, Math.min(18, slot * 0.62));
  const baseline = pad.top + plotHeight;

  ctx.save();
  ctx.font = "11px 'Geist Mono', ui-monospace, monospace";
  ctx.textBaseline = "middle";
  ticks.forEach((tick) => {
    const y = baseline - yScale(tick);
    ctx.strokeStyle = tick === 0 ? "rgba(247,248,248,0.2)" : "rgba(247,248,248,0.1)";
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(pad.left, y);
    ctx.lineTo(width - pad.right, y);
    ctx.stroke();
    ctx.fillStyle = "rgba(138,143,152,0.9)";
    ctx.textAlign = "right";
    ctx.fillText(formatBytes(tick), pad.left - 10, y);
  });

  ctx.textAlign = "left";
  ctx.textBaseline = "middle";
  ctx.fillStyle = "#3dd68c";
  roundRect(ctx, pad.left, 9, 16, 8, 4);
  ctx.fill();
  ctx.fillStyle = "rgba(138,143,152,0.9)";
  ctx.fillText("processed", pad.left + 22, 13);
  ctx.fillStyle = "rgba(130,143,255,0.34)";
  roundRect(ctx, pad.left + 92, 9, 16, 8, 4);
  ctx.fill();
  ctx.strokeStyle = "rgba(130,143,255,0.72)";
  ctx.setLineDash([4, 4]);
  roundRect(ctx, pad.left + 92, 9, 16, 8, 4);
  ctx.stroke();
  ctx.setLineDash([]);
  ctx.fillStyle = "rgba(138,143,152,0.9)";
  ctx.fillText("pending", pad.left + 114, 13);
  ctx.fillStyle = "#b490f5";
  roundRect(ctx, pad.left + 176, 9, 16, 8, 4);
  ctx.fill();
  ctx.fillStyle = "rgba(138,143,152,0.9)";
  ctx.fillText("deferred", pad.left + 198, 13);
  ctx.fillStyle = "#e8b04b";
  roundRect(ctx, pad.left + 268, 9, 16, 8, 4);
  ctx.fill();
  ctx.fillStyle = "rgba(138,143,152,0.9)";
  ctx.fillText("failed", pad.left + 290, 13);

  const pulse = 0.48 + 0.32 * Math.sin(Date.now() / 170);
  rows.forEach((row, index) => {
    const barX = pad.left + index * slot + (slot - barWidth) / 2;
    const fullHeight = row.total ? Math.max(2, yScale(row.total)) : 0;
    let y = baseline;
    let cumulativeBytes = 0;
    ctx.fillStyle = "rgba(247,248,248,0.055)";
    roundRect(ctx, barX, baseline - Math.max(2, fullHeight), barWidth, Math.max(2, fullHeight), Math.min(5, barWidth / 2));
    ctx.fill();
    row.segments.forEach((segment, segmentIndex) => {
      const previousBytes = cumulativeBytes;
      cumulativeBytes += segment.bytes;
      const segmentHeight = yScale(cumulativeBytes) - yScale(previousBytes);
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
      ctx.strokeStyle = "#828fff";
      ctx.lineWidth = 2;
      ctx.setLineDash([4, 4]);
      roundRect(ctx, barX - 3, baseline - fullHeight - 4, barWidth + 6, fullHeight + 8, 7);
      ctx.stroke();
      ctx.restore();
    }

    const showLabel = index === 0 || index === rows.length - 1 || index % 5 === 4 || row.active;
    if (showLabel) {
      const x = barX + barWidth / 2;
      ctx.fillStyle = row.active ? "#828fff" : "rgba(138,143,152,0.86)";
      ctx.font = row.active ? "700 10px 'Geist Mono', ui-monospace, monospace" : "10px 'Geist Mono', ui-monospace, monospace";
      ctx.textAlign = "center";
      ctx.textBaseline = "alphabetic";
      ctx.fillText(dateKeyLabel(row.date), x, height - 14);
    }
  });
  ctx.restore();
}

function batchCountLabel(row) {
  const parts = [`${row.processed} ok`];
  if (row.deferred) parts.push(`${row.deferred} defer`);
  if (row.continued) parts.push(`${row.continued} continue`);
  if (row.failed) parts.push(`${row.failed} fail`);
  return parts.join(" ");
}

function fitCanvasText(ctx, text, maxWidth) {
  if (ctx.measureText(text).width <= maxWidth) return text;
  const suffix = "…";
  let fitted = text;
  while (fitted && ctx.measureText(`${fitted}${suffix}`).width > maxWidth) {
    fitted = fitted.slice(0, -1);
  }
  return fitted ? `${fitted}${suffix}` : suffix;
}

function drawBatchLegend(ctx, width, left, y) {
  const compact = width < 380;
  const items = [
    { label: "ok", compactLabel: "ok", color: "#3dd68c" },
    { label: "deferred", compactLabel: "def", color: "#b490f5" },
    { label: "continued", compactLabel: "cont", color: "#828fff" },
    { label: "failed", compactLabel: "fail", color: "#e8b04b" },
  ];
  const swatchWidth = compact ? 10 : 16;
  const swatchHeight = compact ? 6 : 8;
  const textGap = compact ? 4 : 7;
  const itemGap = compact ? 4 : 12;
  let cursor = compact ? 8 : left;

  ctx.font = `${compact ? 8 : 11}px 'Geist Mono', ui-monospace, monospace`;
  ctx.textBaseline = "middle";
  ctx.textAlign = "left";
  items.forEach((item, index) => {
    const label = compact ? item.compactLabel : item.label;
    ctx.fillStyle = "rgba(138,143,152,0.88)";
    ctx.fillText(label, cursor, y);
    cursor += Math.ceil(ctx.measureText(label).width) + textGap;
    ctx.fillStyle = item.color;
    roundRect(ctx, cursor, y - swatchHeight / 2, swatchWidth, swatchHeight, swatchHeight / 2);
    ctx.fill();
    cursor += swatchWidth + (index === items.length - 1 ? 0 : itemGap);
  });
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
    .filter((row) => numeric(row.files_processed) || numeric(row.files_deferred) || numeric(row.files_continued) || numeric(row.files_failed))
    .slice(-6)
    .map((row) => {
      const processed = row.files_processed || 0;
      const deferred = row.files_deferred || 0;
      const continued = row.files_continued || 0;
      const failed = numeric(row.files_failed)
        ? row.files_failed
        : Math.max(0, (row.files_attempted || processed + deferred + continued) - processed - deferred - continued);
      return {
        label: timeLabel(row.timestamp),
        sub: numeric(row.pending_before) ? `${row.pending_before}->${row.pending_after}` : "batch",
        processed,
        deferred,
        continued,
        failed,
        attempted: row.files_attempted || processed + deferred + continued + failed || processed,
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
      deferred: batch.deferred || 0,
      continued: batch.continued || 0,
      failed: batch.failed || 0,
      attempted: batch.total,
      live: true,
    });
  }

  const data = completed.slice(-7);
  const totalOk = completed.filter((row) => !row.live).reduce((sum, row) => sum + row.processed, 0);
  const totalDeferred = completed.filter((row) => !row.live).reduce((sum, row) => sum + row.deferred, 0);
  const totalContinued = completed.filter((row) => !row.live).reduce((sum, row) => sum + row.continued, 0);
  const totalFailed = completed.filter((row) => !row.live).reduce((sum, row) => sum + row.failed, 0);
  const durations = completed.filter((row) => !row.live && numeric(row.elapsed)).map((row) => row.elapsed);
  const avgDuration = durations.length ? durations.reduce((sum, value) => sum + value, 0) / durations.length : null;
  els.batchOk.textContent = totalOk ? String(totalOk) : "--";
  els.batchDeferred.textContent = String(totalDeferred);
  els.batchContinued.textContent = String(totalContinued);
  els.batchFailed.textContent = totalFailed ? String(totalFailed) : "0";
  els.batchDuration.textContent = compactDuration(avgDuration);
  els.batchCaption.textContent = data.length ? `${Math.max(0, data.length - (batch.active === true ? 1 : 0))} batches` : "waiting";

  if (!data.length) {
    ctx.fillStyle = "rgba(138,143,152,0.85)";
    ctx.font = "14px 'Geist Mono', ui-monospace, monospace";
    ctx.fillText("No batch yield yet", 24, height / 2);
    return;
  }

  ctx.font = "700 12px 'Geist Mono', ui-monospace, monospace";
  const maxCountWidth = Math.max(
    0,
    ...data.map((row) => ctx.measureText(batchCountLabel(row)).width),
  );
  const leftPad = 74;
  const minimumBarWidth = 80;
  const maximumRightPad = Math.max(78, width - leftPad - minimumBarWidth);
  const pad = {
    top: 30,
    right: Math.min(maximumRightPad, Math.max(78, Math.ceil(maxCountWidth) + 20)),
    bottom: 20,
    left: leftPad,
  };
  const rowGap = 10;
  const rowHeight = Math.max(22, (height - pad.top - pad.bottom - rowGap * (data.length - 1)) / data.length);
  const barHeight = Math.min(18, rowHeight * 0.56);
  const barWidth = Math.max(80, width - pad.left - pad.right);
  const maxTotal = Math.max(1, ...data.map((row) => row.attempted || row.processed + row.deferred + row.continued + row.failed));

  ctx.save();
  drawBatchLegend(ctx, width, pad.left, 14);

  for (let i = 0; i <= 2; i += 1) {
    const x = pad.left + (barWidth * i) / 2;
    ctx.strokeStyle = "rgba(247,248,248,0.08)";
    ctx.beginPath();
    ctx.moveTo(x, pad.top - 2);
    ctx.lineTo(x, height - pad.bottom + 2);
    ctx.stroke();
  }

  data.forEach((row, index) => {
    const y = pad.top + index * (rowHeight + rowGap);
    const barY = y + (rowHeight - barHeight) / 2;
    const processedWidth = (barWidth * row.processed) / maxTotal;
    const deferredWidth = (barWidth * row.deferred) / maxTotal;
    const continuedWidth = (barWidth * row.continued) / maxTotal;
    const failedWidth = (barWidth * row.failed) / maxTotal;
    const attemptedWidth = (barWidth * (row.attempted || row.processed + row.deferred + row.continued + row.failed)) / maxTotal;

    ctx.fillStyle = row.live ? "rgba(130,143,255,0.13)" : "rgba(247,248,248,0.08)";
    roundRect(ctx, pad.left, barY, Math.max(2, attemptedWidth), barHeight, 7);
    ctx.fill();

    if (row.processed) {
      ctx.fillStyle = "#3dd68c";
      roundRect(ctx, pad.left, barY, Math.max(2, processedWidth), barHeight, 7);
      ctx.fill();
    }
    if (row.failed) {
      ctx.fillStyle = "#e8b04b";
      roundRect(ctx, pad.left + processedWidth + deferredWidth + continuedWidth, barY, Math.max(2, failedWidth), barHeight, 7);
      ctx.fill();
    }
    if (row.deferred) {
      ctx.fillStyle = "#b490f5";
      roundRect(ctx, pad.left + processedWidth, barY, Math.max(2, deferredWidth), barHeight, 7);
      ctx.fill();
    }
    if (row.continued) {
      ctx.fillStyle = "#828fff";
      roundRect(ctx, pad.left + processedWidth + deferredWidth, barY, Math.max(2, continuedWidth), barHeight, 7);
      ctx.fill();
    }
    if (row.live) {
      ctx.strokeStyle = "rgba(130,143,255,0.72)";
      ctx.lineWidth = 1.5;
      ctx.setLineDash([4, 4]);
      roundRect(ctx, pad.left, barY - 2, Math.max(4, attemptedWidth), barHeight + 4, 9);
      ctx.stroke();
      ctx.setLineDash([]);
    }

    ctx.fillStyle = row.live ? "#828fff" : "rgba(247,248,248,0.88)";
    ctx.font = row.live ? "700 12px 'Geist Mono', ui-monospace, monospace" : "12px 'Geist Mono', ui-monospace, monospace";
    ctx.textAlign = "right";
    ctx.fillText(row.label, pad.left - 12, y + rowHeight * 0.36);
    ctx.fillStyle = "rgba(90,95,104,0.95)";
    ctx.font = "10px 'Geist Mono', ui-monospace, monospace";
    ctx.fillText(row.sub, pad.left - 12, y + rowHeight * 0.74);

    const count = batchCountLabel(row);
    ctx.fillStyle = row.failed ? "#e8b04b" : row.deferred ? "#b490f5" : row.continued ? "#828fff" : "rgba(247,248,248,0.9)";
    ctx.font = "700 12px 'Geist Mono', ui-monospace, monospace";
    ctx.textAlign = "left";
    ctx.fillText(
      fitCanvasText(ctx, count, Math.max(40, pad.right - 16)),
      pad.left + barWidth + 10,
      barY + barHeight / 2,
    );
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
  els.selfHealLatest.title = els.selfHealLatest.textContent;
  els.selfHealDetail.title = els.selfHealDetail.textContent;

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
  els.selfHealFrontierDetail.title = els.selfHealFrontierDetail.textContent;

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

function heatThresholds(values) {
  const active = values.filter((value) => value > 0).sort((a, b) => a - b);
  if (!active.length) return [1, 1, 1, 1];
  const quantile = (p) => active[Math.min(active.length - 1, Math.floor(p * active.length))];
  return [quantile(0.25), quantile(0.5), quantile(0.75), quantile(0.95)];
}

function heatLevel(value, thresholds) {
  // Quantile buckets keep one spike day from washing out the rest of the map.
  if (!value) return 0;
  if (value >= thresholds[3]) return 4;
  if (value >= thresholds[2]) return 3;
  if (value >= thresholds[1]) return 2;
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
  const pageChanges = intValue(day.pages_created) + intValue(day.pages_updated);
  return [
    day.date,
    metric,
    `${intValue(day.processed)} processed`,
    `${intValue(day.deferred)} deferred`,
    `${intValue(day.failed)} failed`,
    `${pageChanges} page changes`,
    `sources: ${formatSources(day.sources)}`,
  ].join(" · ");
}

function renderSaveMonths(days, startPad, columnCount) {
  els.saveMonths.innerHTML = "";
  els.saveMonths.style.gridTemplateColumns = `repeat(${columnCount}, 12px)`;
  const labelsByColumn = new Map();
  let lastLabelColumn = -Infinity;
  days.forEach((day, index) => {
    const date = parseDateKey(day.date);
    if (!date) return;
    const monthKey = `${date.getFullYear()}-${date.getMonth()}`;
    const previous = index > 0 ? parseDateKey(days[index - 1].date) : null;
    const previousMonthKey = previous ? `${previous.getFullYear()}-${previous.getMonth()}` : null;
    if (monthKey === previousMonthKey) return;
    const column = Math.floor((index + startPad) / 7) + 1;
    // Keep labels from crowding when a month owns fewer than three columns.
    if (column - lastLabelColumn < 3) return;
    labelsByColumn.set(column, date.toLocaleDateString("ja-JP", { month: "short" }));
    lastLabelColumn = column;
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
    ["deferred", intValue(day.deferred)],
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
    item.title = sample;
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
    const pageChanges = intValue(day.pages_created) + intValue(day.pages_updated);
    body.textContent = `${intValue(day.raw_saved)} saved · ${intValue(day.processed)} processed · ${pageChanges} changes`;
    body.title = body.textContent;
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
  const allDays = Array.isArray(data.days) ? data.days : [];
  // Trim the leading dead period so the heatmap starts where activity does.
  const firstActive = allDays.findIndex((day) =>
    intValue(day.raw_saved)
    || intValue(day.processed)
    || intValue(day.pages_created)
    || intValue(day.pages_updated)
    || intValue(day.failed)
  );
  const days = firstActive > 0 ? allDays.slice(firstActive) : allDays;
  const totals = data.totals || {};
  const pageChanges = intValue(totals.pages_created) + intValue(totals.pages_updated);

  els.saveTotal.textContent = intValue(totals.raw_saved).toLocaleString();
  els.saveProcessed.textContent = intValue(totals.processed).toLocaleString();
  els.savePages.textContent = pageChanges.toLocaleString();
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
  const thresholds = heatThresholds(values);
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
    const level = heatLevel(value, thresholds);
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
      color: "rgba(138,143,152,0.72)",
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
  ctx.strokeStyle = "rgba(247,248,248,0.08)";
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

function renderLibrarian(librarian) {
  const data = librarian || {};
  const state = fmt(data.state, "NOT_READY");
  const progress = data.progress || {};
  const queue = data.queue || {};
  const authority = data.authority || {};
  const quality = data.quality || {};
  const collectionPlane = data.collection_authority || {};
  const collectionMetrics = collectionPlane.metrics || quality.collection_metrics || {};
  const collectionQueue = collectionPlane.queue || {};
  const rollout = data.rollout || {};
  const soak = data.soak || {};
  const restorePoints = data.restore_points || {};
  const preimages = data.transaction_preimages || {};
  const evidence = data.library_evidence || {};
  const annif = evidence.annif || {};
  const profile = evidence.profile_retrieval || {};
  const query2doc = evidence.query2doc || {};
  const query2docUnseen = evidence.query2doc_unseen || {};
  const progressRows = [
    ["uid", els.librarianUid, els.librarianUidBar],
    ["classification_shadow", els.librarianClassification, els.librarianClassificationBar],
    ["links", els.librarianLinks, els.librarianLinksBar],
    ["migration_batch", els.librarianMigration, els.librarianMigrationBar],
    ["full_sweep", els.librarianSweep, els.librarianSweepBar],
  ];
  els.librarianState.textContent = state;
  els.librarianState.dataset.state = state;
  els.librarianPanel.dataset.state = state;
  const classification = progress.classification_shadow || {};
  const complete = intValue(classification.numerator);
  const total = intValue(classification.denominator);
  const collectionFirst = authority.mode === "collection-first";
  els.librarianHeadline.textContent = collectionFirst
    ? total
      ? `${complete.toLocaleString()} of ${total.toLocaleString()} pages assigned to stable collections`
      : "Collection registry has not been synchronized"
    : data.initial_organization_complete_at
      ? "Initial organization complete"
      : total
        ? `${complete.toLocaleString()} of ${total.toLocaleString()} pages shadow-classified`
        : "Shadow migration has not started";
  els.librarianDetail.textContent = fmt(data.detail, "Waiting for Librarian state.");
  const generation = String(data.scope_generation || "--");
  els.librarianGeneration.textContent = generation.startsWith("sha256:")
    ? generation.slice(7, 19)
    : generation;
  const sweptGeneration = String(data.last_swept_generation || "--");
  els.librarianSweptGeneration.textContent = sweptGeneration.startsWith("sha256:")
    ? sweptGeneration.slice(7, 19)
    : sweptGeneration;
  progressRows.forEach(([key, valueEl, barEl]) => {
    const row = progress[key] || {};
    const numerator = intValue(row.numerator);
    const denominator = intValue(row.denominator);
    valueEl.textContent = denominator
      ? `${numerator.toLocaleString()} / ${denominator.toLocaleString()}`
      : "-- / --";
    const ratio = denominator ? Math.max(0, Math.min(1, numerator / denominator)) : 0;
    barEl.style.width = `${Math.round(ratio * 100)}%`;
  });
  const authorityEpoch = intValue(
    (quality.holdout_metrics || {}).authority_epoch || authority.authority_epoch,
  );
  els.librarianAuthority.textContent = authority.active
    ? collectionFirst
      ? `Collection-first · active`
      : `Active${authorityEpoch ? ` · epoch ${authorityEpoch}` : ""}`
    : collectionFirst
      ? "Collection-first · shadow"
      : "Shadow only";
  const holdout = quality.holdout_metrics || {};
  const exact = numeric(holdout.exact_match_rate)
    ? `${(holdout.exact_match_rate * 100).toFixed(1)}% exact`
    : "";
  const forced = numeric(holdout.forced_misclassification_rate)
    ? `${(holdout.forced_misclassification_rate * 100).toFixed(1)}% forced`
    : "";
  els.librarianQuality.textContent = collectionFirst
    ? `${fmt(quality.collection_status, "not evaluated").replaceAll("_", " ")} · ` +
      `${intValue(collectionPlane.hard_failures?.length)} hard · ` +
      `${intValue(collectionPlane.warnings?.length)} warning`
    : [exact, forced].filter(Boolean).join(" · ") || fmt(quality.locked_holdout, "Not evaluated");
  els.librarianCollectionCount.textContent = numeric(collectionMetrics.active_collection_count)
    ? `${intValue(collectionMetrics.active_collection_count)} active`
    : "--";
  els.librarianAssignment.textContent = numeric(collectionMetrics.assignment_coverage)
    ? `${(collectionMetrics.assignment_coverage * 100).toFixed(1)}% · ` +
      `${intValue(collectionMetrics.assignment_count).toLocaleString()} pages`
    : "--";
  els.librarianCrosswalk.textContent = numeric(collectionMetrics.crosswalk_audit_coverage)
    ? `${(collectionMetrics.crosswalk_audit_coverage * 100).toFixed(1)}% audited`
    : "--";
  els.librarianTopShare.textContent = numeric(collectionMetrics.top_collection_share)
    ? `${fmt(collectionMetrics.top_collection_slug, "unknown")} · ` +
      `${(collectionMetrics.top_collection_share * 100).toFixed(1)}%`
    : "--";
  els.librarianReviewQueue.textContent = collectionFirst
    ? `${intValue(collectionQueue.open)} open · ` +
      `${intValue(collectionQueue.completed)} closed · ` +
      `${intValue(collectionQueue.primary_reviews)} primary / ` +
      `${intValue(collectionQueue.challenger_reviews)} challenge · ` +
      `${intValue(collectionQueue.consensus_recommended)} consensus`
    : "--";
  const splitProposals = Array.isArray(collectionPlane.split_proposals)
    ? collectionPlane.split_proposals
    : [];
  els.librarianSplitProposals.textContent = collectionFirst
    ? `${splitProposals.length} proposal · no auto-split`
    : "--";
  const rolloutStatus = fmt(rollout.status, "not_started").replaceAll("_", " ");
  const rolloutStage = fmt(rollout.stage, "").replaceAll("_", " ");
  els.librarianRollout.textContent = rolloutStage
    ? `${rolloutStatus} · ${rolloutStage}`
    : rolloutStatus;
  els.librarianRollout.title = rollout.updated_at
    ? `Updated ${timeLabel(rollout.updated_at)}`
    : "No rollout receipt yet";
  const remaining = intValue(soak.remaining_seconds);
  const elapsed = intValue(soak.elapsed_seconds);
  const observedThrough = fmt(soak.observed_through, "").replaceAll("_", " ");
  const observationStage = observedThrough ? ` · ${observedThrough}` : "";
  els.librarianSoak.textContent = soak.status === "running"
    ? soak.observation_mode === "concurrent_migration"
      ? `Running · ${(elapsed / 3600).toFixed(1)}h${observationStage}`
      : `${(remaining / 86400).toFixed(1)}d remaining`
    : soak.status === "complete"
      ? "Complete"
      : fmt(soak.status, "Not started").replaceAll("_", " ");
  els.librarianRecovery.textContent =
    `${intValue(restorePoints.verified)}/${intValue(restorePoints.count)} verified` +
    ` · ${intValue(preimages.count)} preimage`;
  const evidenceProgress = evidence.phase_progress || {};
  const evidenceMethod = fmt(evidence.method, "classification")
    .replaceAll("_", " ")
    .replaceAll("-", " ");
  els.librarianEvidenceStatus.textContent =
    `${evidenceMethod} · ${fmt(evidence.status, "not started").replaceAll("_", " ")} · ` +
    `${fmt(evidence.stage, "idle").replaceAll("_", " ")} · ` +
    `${intValue(evidenceProgress.numerator)}/${intValue(evidenceProgress.denominator)} phases`;
  const fixture = evidence.fixture || {};
  els.librarianEvidenceFixture.textContent = query2docUnseen.case_count
    ? `Unseen gate ${intValue(query2docUnseen.fused?.hit_count)}/${intValue(query2docUnseen.case_count)} · ` +
      `${fmt(query2docUnseen.decision, "evaluated").replaceAll("-", " ")}`
    : query2doc.case_count
    ? `Query2doc gate ${intValue(query2doc.fused?.hit_count)}/${intValue(query2doc.case_count)} · ` +
      `${fmt(query2doc.decision, "evaluated").replaceAll("-", " ")}`
    : profile.case_count
    ? `Profile gate ${intValue(profile.profile_hit_count)}/${intValue(profile.case_count)} · ` +
      `${fmt(profile.decision, "evaluated").replaceAll("-", " ")}`
    : annif.council_case_count
    ? `Council ${intValue(annif.council_hit_count)}/${intValue(annif.council_case_count)} · ` +
      `${fmt(annif.council_decision, "reviewed").replaceAll("-", " ")} · ` +
      `${intValue(annif.council_completed_rows)} rows stopped`
    : fixture.dev || fixture.holdout
      ? `${intValue(fixture.dev)} dev · ${intValue(fixture.holdout)} holdout · ${intValue(fixture.reserve)} reserve`
      : "Not locked";
  const candidateMetrics = evidence.candidate_metrics || {};
  const unionMetrics = candidateMetrics.union || {};
  const p0Metrics = candidateMetrics.official_baseline || {};
  els.librarianEvidenceRecall.textContent = query2docUnseen.case_count
    ? `Fused ${intValue(query2docUnseen.fused?.hit_count)}/${intValue(query2docUnseen.case_count)} · ` +
      `best raw ${intValue(query2docUnseen.best_raw_hit_count)}/${intValue(query2docUnseen.case_count)} · ` +
      `gate ${intValue(query2docUnseen.minimum_fused_hits)}/${intValue(query2docUnseen.case_count)}`
    : query2doc.case_count
    ? `Fused ${intValue(query2doc.fused?.hit_count)}/${intValue(query2doc.case_count)} · ` +
      `Q2D dense ${intValue(query2doc.query2doc_dense?.hit_count)}/${intValue(query2doc.case_count)} · ` +
      `gate ${intValue(query2doc.minimum_fused_hits)}/${intValue(query2doc.case_count)}`
    : profile.case_count
    ? `Profile ${intValue(profile.profile_hit_count)}/${intValue(profile.case_count)} · ` +
      `baseline ${intValue(profile.baseline_hit_count)}/${intValue(profile.case_count)} · ` +
      `gate ${intValue(profile.minimum_profile_hits)}/${intValue(profile.case_count)}`
    : annif.case_count
    ? `Annif top1 ${intValue(annif.best_top1_hit_count)}/${intValue(annif.case_count)} · ` +
      `top5 ${intValue(annif.best_top5_hit_count)}/${intValue(annif.case_count)}`
    : numeric(unionMetrics.recall_at_12)
      ? `P0 ${(intValue(p0Metrics.recall_at_12 * 1000) / 10).toFixed(1)}% · W ${(intValue(unionMetrics.recall_at_12 * 1000) / 10).toFixed(1)}%`
      : "Awaiting Annif 10-case gate";
  const externalTest = evidence.external_test || {};
  const externalUnion = (externalTest.metrics || {}).union || {};
  els.librarianEvidenceExternal.textContent = query2docUnseen.query_count
    ? `${intValue(query2docUnseen.query_count)} unseen queries · ` +
      `${shortName(query2docUnseen.model)} · ${intValue(query2docUnseen.model_calls)} model calls`
    : query2doc.query_count
    ? `${intValue(query2doc.query_count)} local queries · ` +
      `${shortName(query2doc.model)} · ${intValue(query2doc.model_calls)} model calls`
    : profile.profile_count
    ? `${intValue(profile.profile_count).toLocaleString()} official profiles · ` +
      `${fmt(profile.embedding_model, "embedding unknown")} · ` +
      `${intValue(profile.llm_calls)} LLM calls`
    : annif.train_documents
    ? `${intValue(annif.train_documents).toLocaleString()} train · ` +
      `${intValue(annif.test_documents).toLocaleString()} external test`
    : numeric(externalUnion.recall_at_20)
      ? `${intValue(externalTest.n).toLocaleString()} test · R@20 ${(externalUnion.recall_at_20 * 100).toFixed(1)}%`
      : "Corpus acquisition in progress";
  const holdoutEvidence = evidence.holdout_metrics || {};
  const unexpectedHold = numeric(holdoutEvidence.unexpected_hold_rate)
    ? `${(holdoutEvidence.unexpected_hold_rate * 100).toFixed(1)}% unexpected`
    : "";
  const severeCount = holdoutEvidence.severe_error_count;
  els.librarianEvidenceHold.textContent =
    query2docUnseen.case_count
      ? query2docUnseen.decision_trial_authorized
        ? "Decision trial authorized · corpus blocked"
        : "Decision trial blocked · corpus blocked"
      : query2doc.case_count
      ? query2doc.unseen_evaluation_authorized
        ? "Unseen evaluation authorized · corpus blocked"
        : "Unseen evaluation blocked"
      : [unexpectedHold, severeCount !== undefined ? `${intValue(severeCount)} severe` : ""]
        .filter(Boolean)
        .join(" · ") || "Not evaluated";
  const evidenceResource = evidence.resource || {};
  els.librarianEvidenceResource.textContent = numeric(evidenceResource.recall_p99_ms)
    ? `p99 ${intValue(evidenceResource.recall_p99_ms)}ms · max ${intValue(evidenceResource.recall_max_ms)}ms`
    : fmt(evidenceResource.status, "Not measured").replaceAll("_", " ");
  const evidenceStorage = evidence.storage || {};
  els.librarianEvidenceStorage.textContent = numeric(evidenceStorage.working_set_bytes)
    ? `${formatBytes(evidenceStorage.working_set_bytes)} working · ${formatBytes(evidenceStorage.audit_store_bytes)} audit`
    : "Not measured";
  const evidenceAuthority = evidence.authority || {};
  els.librarianEvidenceAuthority.textContent =
    `${fmt(evidenceAuthority.status, "inactive").replaceAll("_", " ")} · ` +
    `${evidenceAuthority.mutation_capability ? "mutation enabled" : "decision only"}`;
  const updateValidation = evidence.update_validation || {};
  const sourceUpdate = updateValidation.source_or_index || {};
  els.librarianEvidenceUpdate.textContent = sourceUpdate.fixture_requirement
    ? `${fmt(sourceUpdate.fixture_requirement).replaceAll("-", " ")} · epoch 3 sealed`
    : "Not adopted";
  els.librarianQueue.replaceChildren();
  [
    ["queued", "Queued"],
    ["actionable", "Actionable"],
    ["running", "Running"],
    ["held", "Held"],
    ["quarantined", "Quarantined"],
    ["completed", "Completed"],
  ].forEach(([key, label]) => {
    const chip = document.createElement("span");
    chip.className = `librarian-queue-chip ${key}`;
    chip.textContent = `${label} ${intValue(queue[key]).toLocaleString()}`;
    els.librarianQueue.appendChild(chip);
  });
  const flow24 = (data.flow || {})["24h"] || {};
  const flow7 = (data.flow || {})["7d"] || {};
  const eta = data.eta || {};
  const growth = data.growth || {};
  const etaText = eta.status === "estimated" && numeric(eta.days)
    ? ` · ETA ${eta.days.toFixed(1)}d`
    : eta.status === "falling_behind_or_unstable"
      ? " · ETA unavailable"
      : "";
  els.librarianFlow.textContent =
    `24h ${intValue(flow24.completed)} complete / ${intValue(flow24.arrivals)} arrival · ` +
    `7d ${intValue(flow7.completed)} complete / ${intValue(flow7.arrivals)} arrival · ` +
    `Active ${intValue(growth.active_page_delta) >= 0 ? "+" : ""}${intValue(growth.active_page_delta)} ` +
    `/ Raw ${intValue(growth.raw_unit_delta) >= 0 ? "+" : ""}${intValue(growth.raw_unit_delta)} · ` +
    `authority ${authority.active ? "active" : "shadow only"}${etaText}`;
  els.librarianReceipts.replaceChildren();
  const receipts = Array.isArray(data.recent_receipts) ? data.recent_receipts.slice(0, 6) : [];
  if (!receipts.length) {
    const empty = document.createElement("p");
    empty.className = "librarian-empty";
    empty.textContent = "No receipts yet";
    els.librarianReceipts.appendChild(empty);
    return;
  }
  receipts.forEach((receipt) => {
    const row = document.createElement("div");
    row.className = "librarian-receipt";
    const event = document.createElement("strong");
    event.textContent = fmt(receipt.event || receipt.operation || receipt.status, "receipt");
    const detail = document.createElement("span");
    const counts = [
      receipt.classified !== undefined ? `${intValue(receipt.classified)} classified` : "",
      receipt.held !== undefined ? `${intValue(receipt.held)} held` : "",
      receipt.transaction_id ? shortName(receipt.transaction_id) : "",
    ].filter(Boolean);
    detail.textContent = counts.join(" · ") || timeLabel(receipt.timestamp || receipt.recorded_at);
    row.append(event, detail);
    els.librarianReceipts.appendChild(row);
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
  const hardening = data.autonomy_hardening || {};
  const deadman = hardening.deadman || {};
  const quality = hardening.quality || {};
  const holds = hardening.managed_holds || {};
  const provisional = hardening.provisional_recall || {};
  const research = data.research || {};
  const researchTotals = research.totals || {};
  const artifacts = hardening.decision_artifacts || {};
  const ledger = readBack.derived_view_integrity || {};
  const sensitivity = coverage.sensitivity || {};
  const rawFiles = intValue(capture.raw_files);
  const claimed = intValue(capture.claimed_raw_files);
  const checked = intValue(readBack.checked);
  const passed = intValue(readBack.passed);

  const convergenceBits = convergence.status === "ok"
    ? [
        `${intValue(convergence.actionable).toLocaleString()} active`,
        `${intValue(convergence.semantic_deferred).toLocaleString()} correction uncertainty`,
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
  els.healthReplay.textContent = intValue(artifacts.count).toLocaleString();
  els.healthDeadman.textContent = deadman.main?.status === "ok" && deadman.observer?.status === "ok"
    ? "2/2"
    : `${[deadman.main, deadman.observer].filter((row) => row?.status === "ok").length}/2`;
  els.healthQuality.textContent = intValue(quality.frozen)
    ? `${intValue(quality.frozen)} frozen`
    : (quality.probe?.status === "ok" ? "ok" : fmt(quality.probe?.status || "--"));
  els.healthHolds.textContent = intValue(holds.total).toLocaleString();
  els.healthProvisional.textContent = intValue(provisional.entries).toLocaleString();
  els.healthLedger.textContent = fmt(ledger.status || "--");
  els.healthResearchRuns.textContent = intValue(researchTotals.runs).toLocaleString();
  els.healthResearchClaims.textContent = `${intValue(researchTotals.supported_claims).toLocaleString()}/${(
    intValue(researchTotals.supported_claims)
    + intValue(researchTotals.contradicted_claims)
    + intValue(researchTotals.unknown_claims)
  ).toLocaleString()}`;
  els.healthResearchTrace.textContent = healthPercent(research.decision_trace_coverage);
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
      name.title = fmt(row.name);
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
      meta.title = meta.textContent;

      item.append(main, roleWrap, meta);
      els.modelGrid.appendChild(item);
    });
}

function renderModelLab(lab) {
  const policy = lab.policy || {};
  const roles = policy.roles || {};
  const candidates = Array.isArray(lab.candidates) ? lab.candidates : [];
  const canaries = policy.canaries || {};
  if (els.modelLabCaption) {
    els.modelLabCaption.textContent =
      lab.status === "ok" ? `updated ${timeLabel(policy.updated_at)}` : fmt(lab.status, "unavailable");
    els.modelLabCaption.title = policy.updated_at ? `Updated ${policy.updated_at}` : "";
  }
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
    name.title = name.textContent;
    const meta = document.createElement("span");
    meta.className = "model-meta";
    meta.textContent = `${fmt(selected.model, "--")} · ${fmt(selected.effort, "--")}`;
    meta.title = meta.textContent;
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

function renderRecallField(fieldValue) {
  const field = fieldValue || {};
  const summary = field.summary || {};
  const latency = summary.latency_ms || {};
  const status = String(field.status || "offline");
  els.recallFieldSummary.className = `recall-field-summary ${status}`;
  els.recallFieldState.textContent =
    `FIELD ${status.toUpperCase()} · ${String(field.mode || "off").toUpperCase()}`;
  els.recallFieldCounts.textContent = [
    `active ${summary.active || 0}`,
    `candidate ${summary.candidate || 0}`,
    `commit ${summary.commit || 0}`,
    `reject ${summary.reject || 0}`,
  ].join(" · ");
  const agreement = numeric(summary.teacher_agreement)
    ? pctLabel(summary.teacher_agreement)
    : "collecting";
  const growth = summary.growth || {};
  const growthLabel = growth.authority_enabled
    ? `authority · canary ${growth.canary_percent || 0}%`
    : growth.field_learning_allowed
      ? "learning on · authority held"
      : `learning ${growth.strong_positive || 0}/${growth.strong_positive_target || 200}`;
  els.recallFieldQuality.textContent =
    `${growthLabel} · agreement ${agreement} · p95 ${msLabel(latency.p95)}`;
}

function renderRecall(recall) {
  const data = recall || {};
  const decisions = data.decisions || {};
  const latency = data.latency_ms || {};
  const latestEval = data.latest_eval || null;
  const evalMetricsLatency = latestEval && latestEval.latency_ms ? latestEval.latency_ms : {};
  const calibration = data.calibration || {};
  const pulls = data.pulls || {};
  const field = data.field || {};

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

  renderRecallField(field);

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
    : ["Run chronovisor recall-improve run to start the local proposal tournament."];
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
  const semanticDeferred = intValue(status.semantic_deferred?.count);
  els.pendingSub.textContent = semanticDeferred
    ? `${semanticDeferred} semantic held · updated ${timeLabel(status.updated_at)}`
    : `updated ${timeLabel(status.updated_at)}`;
  const stageValue = fmt(status.stage, "idle");
  els.stage.textContent = stageMetricLabel(stageValue);
  els.stage.title = stageValue;
  els.raw.textContent = shortName(status.current_raw || "no active raw");
  els.batch.textContent = batch.total ? `${batch.index || 0}/${batch.total}` : "--";
  els.batchSub.textContent = batch.total
    ? `${batch.succeeded || 0} ok / ${batch.deferred || 0} deferred / ${batch.continued || 0} continued / ${batch.failed || 0} fail`
    : "waiting";
  els.ollama.textContent = modelStatus.available || ollama.available ? "online" : "offline";
  els.ollamaSub.textContent = modelSummary.installed !== undefined
    ? `${intValue(modelSummary.loaded)} loaded · ${intValue(modelSummary.installed)} installed`
    : model.name || model.model || "no model";
  els.currentRaw.textContent = status.current_raw ? shortName(status.current_raw) : "waiting";
  els.currentOp.textContent = status.current_op ? fmt(status.current_op) : fmt(status.stage || "idle");
  renderWorkStatus(status);
  renderDecisionTrace(status.local_consensus || {});
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
  renderLibrarian(snapshot.librarian || {});
  renderHealth(snapshot.health || {});
  renderModelStatus(modelStatus);
  renderModelLab(snapshot.model_lab || {});
  renderEvents(snapshot.events || []);
  drawLineChart(els.pendingChart, snapshot.save_history || {}, status);
  drawBatchChart(els.batchChart, metrics, status);
}

let refreshInFlight = null;
let hasRenderedFullSnapshot = false;
const SNAPSHOT_TIMEOUT_MS = 180000;
const FAST_SNAPSHOT_TIMEOUT_MS = 3000;
const ACTIVE_REFRESH_DELAY_MS = 5000;
const IDLE_REFRESH_DELAY_MS = 10000;
const ERROR_REFRESH_DELAY_MS = 5000;
const DECISION_REFRESH_TIMEOUT_MS = 2500;
const ACTIVE_DECISION_REFRESH_DELAY_MS = 800;
const IDLE_DECISION_REFRESH_DELAY_MS = 2500;
let nextRefreshDelayMs = IDLE_REFRESH_DELAY_MS;
let decisionRefreshInFlight = false;
let nextDecisionRefreshDelayMs = IDLE_DECISION_REFRESH_DELAY_MS;

async function refreshFast() {
  const controller = new AbortController();
  const timeoutId = window.setTimeout(() => controller.abort(), FAST_SNAPSHOT_TIMEOUT_MS);
  try {
    const response = await fetch("/api/fast-snapshot", {
      cache: "no-store",
      signal: controller.signal,
    });
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    render(await response.json());
    document.body.dataset.snapshotState = "summary";
  } catch {
    // The full snapshot remains authoritative and reports its own failures.
  } finally {
    window.clearTimeout(timeoutId);
  }
}

function refresh() {
  if (refreshInFlight !== null) return refreshInFlight;
  refreshInFlight = (async () => {
    const controller = new AbortController();
    const timeoutId = window.setTimeout(() => controller.abort(), SNAPSHOT_TIMEOUT_MS);
    try {
      const response = await fetch("/api/snapshot", {
        cache: "no-store",
        signal: controller.signal,
      });
      if (!response.ok) throw new Error(`HTTP ${response.status}`);
      const snapshot = await response.json();
      render(snapshot);
      hasRenderedFullSnapshot = true;
      document.body.dataset.snapshotState = snapshot._dashboard?.stale ? "stale" : "full";
      const status = snapshot.status || {};
      const batch = status.batch || {};
      const llm = status.llm || {};
      const localConsensus = status.local_consensus || {};
      const frontierRepair = status.frontier_repair || {};
      nextRefreshDelayMs = (
        status.state === "running"
        || batch.active === true
        || llm.active === true
        || localConsensus.active === true
        || frontierRepair.active === true
      ) ? ACTIVE_REFRESH_DELAY_MS : IDLE_REFRESH_DELAY_MS;
    } catch (error) {
      nextRefreshDelayMs = ERROR_REFRESH_DELAY_MS;
      setState("error");
      els.stateText.textContent = "disconnected";
      els.eventFeed.textContent = "";
      const message = document.createElement("div");
      message.className = "event-message";
      message.textContent = `Dashboard fetch failed: ${error.message}`;
      els.eventFeed.appendChild(message);
    } finally {
      window.clearTimeout(timeoutId);
    }
  })().finally(() => {
    refreshInFlight = null;
  });
  return refreshInFlight;
}

async function refreshLoop() {
  try {
    if (!hasRenderedFullSnapshot) await refreshFast();
    await refresh();
  } catch {
    // refresh() reports normal fetch failures; keep polling after unexpected ones.
  } finally {
    window.setTimeout(refreshLoop, nextRefreshDelayMs);
  }
}

async function refreshDecisionTrace() {
  if (decisionRefreshInFlight) return;
  decisionRefreshInFlight = true;
  const controller = new AbortController();
  const timeoutId = window.setTimeout(
    () => controller.abort(),
    DECISION_REFRESH_TIMEOUT_MS
  );
  try {
    const response = await fetch("/api/local-consensus", {
      cache: "no-store",
      signal: controller.signal,
    });
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    const consensus = (await response.json()).local_consensus || {};
    renderDecisionTrace(consensus);
    nextDecisionRefreshDelayMs = consensus.active
      ? ACTIVE_DECISION_REFRESH_DELAY_MS
      : IDLE_DECISION_REFRESH_DELAY_MS;
  } catch {
    nextDecisionRefreshDelayMs = IDLE_DECISION_REFRESH_DELAY_MS;
  } finally {
    window.clearTimeout(timeoutId);
    decisionRefreshInFlight = false;
  }
}

async function decisionTraceRefreshLoop() {
  try {
    await refreshDecisionTrace();
  } finally {
    window.setTimeout(decisionTraceRefreshLoop, nextDecisionRefreshDelayMs);
  }
}

async function refreshRecallFieldLoop() {
  try {
    const response = await fetch("/api/cortex/field", { cache: "no-store" });
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    renderRecallField(await response.json());
  } catch {
    renderRecallField({ status: "offline", mode: "off", summary: {} });
  } finally {
    window.setTimeout(refreshRecallFieldLoop, IDLE_REFRESH_DELAY_MS);
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

async function copyLanLink() {
  const original = "Recovery link";
  try {
    const response = await fetch("/api/lan-access", { cache: "no-store" });
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    const access = await response.json();
    const url = Array.isArray(access.urls) ? access.urls[0] : null;
    if (!access.enabled || !url) throw new Error("LAN access disabled");
    if (navigator.clipboard?.writeText) {
      await navigator.clipboard.writeText(url);
    } else {
      const input = document.createElement("textarea");
      input.value = url;
      input.setAttribute("readonly", "");
      input.style.position = "fixed";
      input.style.opacity = "0";
      document.body.appendChild(input);
      input.select();
      document.execCommand("copy");
      input.remove();
    }
    els.lanShareButton.textContent = "Copied";
  } catch {
    els.lanShareButton.textContent = "Unavailable";
  } finally {
    window.setTimeout(() => {
      els.lanShareButton.textContent = original;
    }, 1800);
  }
}

els.lanShareButton.addEventListener("click", copyLanLink);

void refreshLoop();
void decisionTraceRefreshLoop();
void refreshRecallFieldLoop();
window.addEventListener("resize", refresh);
