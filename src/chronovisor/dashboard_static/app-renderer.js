function fmt(value, fallback = "--") {
  if (value === null || value === undefined || value === "") return fallback;
  return String(value);
}

function shortName(value) {
  const text = fmt(value);
  if (text.length <= 64) return text;
  return `${text.slice(0, 30)}...${text.slice(-28)}`;
}

function decisionTraceModelLabel(value) {
  const base = fmt(value, "not configured").split("/").at(-1).split(":")[0];
  const family = base.split(/[-_]/)[0].replace(/(?<![.])\d+$/u, "");
  return family ? `${family[0].toUpperCase()}${family.slice(1)}` : "Not configured";
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

const latestProcessingLanes = new Map();
let latestDecisionTrace = {};
let selectedProcessingLaneKey = "";
let pinnedProcessingLaneKey = "";
let latestProcessingGeneratedAtMs = null;
let latestProcessingRevision = "";

function setProcessingConnection(state, label) {
  els.processingConnection.dataset.state = state;
  els.processingConnection.querySelector("strong").textContent = label;
}

function reconcileProcessingLaneSteps(track, steps) {
  const existing = new Map(
    [...track.querySelectorAll(".processing-step")].map((node) => [
      node.dataset.processingStep,
      node,
    ])
  );
  const rows = Array.isArray(steps) ? steps : [];
  rows.forEach((step) => {
    const key = fmt(step.key, "step");
    const node = existing.get(key) || document.createElement("span");
    node.dataset.processingStep = key;
    node.className = `processing-step ${fmt(step.status, "pending")}`;
    let label = node.querySelector("span");
    if (!label) {
      label = document.createElement("span");
      node.appendChild(label);
    }
    label.textContent = fmt(step.label, key);
    track.appendChild(node);
    existing.delete(key);
  });
  existing.forEach((node) => node.remove());
}

function processingElapsedText(lane, now = Date.now()) {
  const started = parseMs(lane?.started_at);
  if (started === null) return "--";
  const finished = lane?.recent ? parseMs(lane?.updated_at) : null;
  return preciseDuration(Math.max(0, ((finished ?? now) - started) / 1000));
}

function processingLaneDetail(lane, now = Date.now()) {
  if (lane.state !== "active") return fmt(lane.detail, "waiting for work");
  const role = fmt(lane.role || lane.phase || lane.current_step, "work");
  const details = [lane.model, role, processingElapsedText(lane, now)].filter(Boolean);
  if (lane.recent) details.push("just completed");
  if (intValue(lane.active_jobs) > 1) details.push(`${intValue(lane.active_jobs)} jobs`);
  return details.join(" · ");
}

function processingLaneForTrace(trace) {
  const role = String(trace?.task_role || "").toLowerCase().replaceAll("-", "_");
  const named = role.includes("typed_graph")
      || role.startsWith("relation_")
      || role.startsWith("entity_merge")
      || role.startsWith("recall_rubric")
    ? "typed_graph"
    : role.startsWith("recall")
      ? "recall"
      : role.startsWith("ingest")
        ? "ingest"
        : role.startsWith("improve")
            || role.startsWith("model_eval")
            || role.startsWith("autonomy")
            || role.startsWith("orphan_link")
          ? "improve"
          : role.includes("repair")
            ? "repair"
            : trace?.request_sha256 ? "audit" : "";
  if (named && latestProcessingLanes.has(named)) return named;
  return [...latestProcessingLanes.entries()].find(([, lane]) =>
    lane.state === "active" && lane.current_step === "consensus"
  )?.[0] || [...latestProcessingLanes.entries()].find(([, lane]) => lane.state === "active")?.[0] || "";
}

function updateProcessingTraceSelection(trace = latestDecisionTrace) {
  latestDecisionTrace = trace || {};
  const rows = [...els.processingLanes.querySelectorAll(".processing-lane")];
  selectedProcessingLaneKey = (
    pinnedProcessingLaneKey && latestProcessingLanes.has(pinnedProcessingLaneKey)
      ? pinnedProcessingLaneKey
      : processingLaneForTrace(latestDecisionTrace) || rows[0]?.dataset.processingLane || ""
  );
  rows.forEach((row) => {
    const selected = row.dataset.processingLane === selectedProcessingLaneKey;
    row.setAttribute("aria-selected", String(selected));
    row.tabIndex = selected ? 0 : -1;
    if (selected) els.decisionTracePanel?.setAttribute("aria-labelledby", row.id);
  });
}

function selectProcessingLane(key, focus = false) {
  if (!latestProcessingLanes.has(key)) return false;
  pinnedProcessingLaneKey = key;
  updateProcessingTraceSelection();
  window.dispatchEvent(new window.CustomEvent(
    "chronovisor:processing-lane-select",
    { detail: { pipeline: key } },
  ));
  const row = els.processingLanes.querySelector(`[data-processing-lane="${key}"]`);
  if (focus) row?.focus();
  return true;
}

function renderProcessingActivity(activity) {
  const generatedAtMs = parseMs(activity?.generated_at);
  const revision = fmt(activity?.revision, "");
  if (
    generatedAtMs !== null
    && latestProcessingGeneratedAtMs !== null
    && generatedAtMs < latestProcessingGeneratedAtMs
  ) return false;
  if (generatedAtMs !== null) latestProcessingGeneratedAtMs = generatedAtMs;
  if (revision && revision === latestProcessingRevision) return false;
  latestProcessingRevision = revision;

  const lanes = Array.isArray(activity?.lanes) ? activity.lanes : [];
  const existing = new Map(
    [...els.processingLanes.querySelectorAll(".processing-lane")].map((node) => [
      node.dataset.processingLane,
      node,
    ])
  );
  lanes.forEach((lane) => {
    const key = fmt(lane.key, "lane");
    latestProcessingLanes.set(key, lane);
    const row = existing.get(key) || document.createElement("section");
    row.dataset.processingLane = key;
    row.id = `processing-lane-tab-${key}`;
    row.className = `processing-lane ${lane.state === "active" ? "active" : "idle"}`;
    row.setAttribute("role", "tab");
    row.setAttribute("aria-controls", "decision-trace-panel");

    let label = row.querySelector(".processing-lane-label");
    let track = row.querySelector(".processing-track");
    let meta = row.querySelector(".processing-lane-meta");
    if (!label) {
      label = document.createElement("strong");
      label.className = "processing-lane-label";
      row.appendChild(label);
    }
    if (!track) {
      track = document.createElement("div");
      track.className = "processing-track";
      row.appendChild(track);
    }
    if (!meta) {
      meta = document.createElement("div");
      meta.className = "processing-lane-meta";
      meta.append(document.createElement("strong"), document.createElement("span"));
      row.appendChild(meta);
    }
    if (!row.dataset.traceKeyboardBound) {
      row.addEventListener("click", () => selectProcessingLane(row.dataset.processingLane));
      row.addEventListener("keydown", (event) => {
        if (event.key === "Enter" || event.key === " ") {
          event.preventDefault();
          selectProcessingLane(row.dataset.processingLane);
          return;
        }
        const tabs = [...els.processingLanes.querySelectorAll('[role="tab"]')];
        const offset = ["ArrowRight", "ArrowDown"].includes(event.key)
          ? 1
          : ["ArrowLeft", "ArrowUp"].includes(event.key) ? -1 : 0;
        const target = event.key === "Home"
          ? tabs[0]
          : event.key === "End"
            ? tabs.at(-1)
            : offset
              ? tabs[(tabs.indexOf(row) + offset + tabs.length) % tabs.length]
              : null;
        if (!target) return;
        event.preventDefault();
        selectProcessingLane(target.dataset.processingLane, true);
      });
      row.dataset.traceKeyboardBound = "true";
    }

    label.textContent = fmt(lane.label, key);
    reconcileProcessingLaneSteps(track, lane.steps);
    meta.querySelector("strong").textContent = lane.state === "active"
      ? lane.recent ? "PULSE" : "ACTIVE"
      : "IDLE";
    meta.querySelector("span").textContent = processingLaneDetail(lane);
    meta.title = lane.work_item ? `work ${String(lane.work_item).slice(0, 20)}` : "";
    els.processingLanes.appendChild(row);
    existing.delete(key);
  });
  existing.forEach((node, key) => {
    latestProcessingLanes.delete(key);
    node.remove();
  });
  els.processingPanel.dataset.activeCount = String(intValue(activity?.active_count));
  document.body.dataset.processingRevision = revision;
  updateProcessingTraceSelection();
  return true;
}

function updateProcessingElapsed() {
  const now = Date.now();
  latestProcessingLanes.forEach((lane, key) => {
    if (lane.state !== "active") return;
    const row = [...els.processingLanes.querySelectorAll(".processing-lane")].find(
      (node) => node.dataset.processingLane === key
    );
    const detail = row?.querySelector(".processing-lane-meta span");
    if (detail) detail.textContent = processingLaneDetail(lane, now);
  });
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
let latestLiveConsensus = null;
const DECISION_TRACE_STATES = ["pending", "active", "done", "skipped", "error"];
const DECISION_TRACE_PROJECTION_SCHEMA = "chronovisor.decision-trace-projection.v1";

function setDecisionSvgState(node, state = "pending") {
  if (!node) return;
  node.classList.remove(...DECISION_TRACE_STATES);
  node.classList.add(DECISION_TRACE_STATES.includes(state) ? state : "pending");
  node.dataset.state = DECISION_TRACE_STATES.includes(state) ? state : "pending";
}

function setDecisionSvgText(selector, value) {
  const node = els.decisionTraceHarness.querySelector(selector);
  if (node) node.textContent = value;
}

function decisionTraceProjection(trace) {
  const projection = trace?.projection;
  return projection?.schema === DECISION_TRACE_PROJECTION_SCHEMA ? projection : null;
}
function updateDecisionSvgHarness(trace, focusEvent = null) {
  const harness = els.decisionTraceHarness;
  if (!harness) return;
  const projection = decisionTraceProjection(trace);
  harness.dataset.projectionStatus = projection ? "ok" : "missing";
  if (!projection) {
    harness.querySelectorAll(
      "[data-trace-key], [data-overall-key], [data-plan-key], [data-path-key], "
        + "[data-reasoning-output], [data-model-key], [data-decision-lane], "
        + "[data-decision-lane-step], [data-lane-path], [data-repair-lane]"
    ).forEach((node) => setDecisionSvgState(node, "pending"));
    harness.querySelectorAll("[data-context-option], [data-reasoning-key]")
      .forEach((node) => node.classList.remove("selected"));
    return;
  }

  const nodes = projection.nodes || {};
  harness.dataset.traceState = fmt(trace.state, "idle");
  harness.dataset.outcomeKind = fmt(trace.outcome?.kind, "idle");
  harness.dataset.taskRole = fmt(trace.task_role, "idle");
  harness.querySelectorAll("[data-trace-key]").forEach((node) => {
    setDecisionSvgState(node, nodes[node.dataset.traceKey]);
  });
  harness.querySelectorAll("[data-overall-key]").forEach((node) => {
    setDecisionSvgState(node, nodes[node.dataset.overallKey]);
  });
  harness.querySelectorAll("[data-plan-key]").forEach((node) => {
    setDecisionSvgState(node, nodes[node.dataset.planKey]);
  });
  harness.querySelectorAll("[data-path-key]").forEach((node) => {
    setDecisionSvgState(node, projection.paths?.[node.dataset.pathKey]);
  });
  for (const [selector, pathKey] of [
    ["[data-pair-yes-label]", "pair-artifact-join"],
    ["[data-pair-no-label]", "pair-tie_break"],
    ["[data-quorum-yes-label]", "quorum-artifact-join"],
    ["[data-quorum-no-label]", "quorum-hold"],
    ["[data-seal-yes-label]", "seal-decision"],
    ["[data-seal-no-label]", "seal-hold"],
  ]) {
    setDecisionSvgState(harness.querySelector(selector), projection.paths?.[pathKey]);
  }

  const context = projection.context || {};
  setDecisionSvgText("[data-plan-value=\"context-selection\"]", fmt(context.label, "required — → selected —"));
  const contextOptions = [...harness.querySelectorAll("[data-context-option]")];
  (context.options || []).forEach((option, index) => {
    const node = contextOptions[index];
    if (!node) return;
    node.dataset.contextTokens = String(option.tokens);
    const label = node.querySelector("[data-context-label]");
    if (label) label.textContent = fmt(option.label, "—");
    node.classList.toggle("selected", option.selected === true);
    setDecisionSvgState(node, option.state);
  });
  const selectedContextNode = contextOptions.find((node) => node.classList.contains("selected"));
  if (selectedContextNode) {
    const contextX = selectedContextNode.transform?.baseVal?.getItem(0)?.matrix?.e || 442;
    const contextDirection = Math.sign(contextX - 492) || -1;
    const contextRadius = Math.min(10, Math.abs(contextX - 492) / 2);
    harness.querySelector("[data-path-key=\"execution-plan-context\"]")?.setAttribute(
      "d",
      `M492 56 V${86 - contextRadius} Q492 86 ${492 + contextDirection * contextRadius} 86 `
        + `H${contextX - contextDirection * contextRadius} Q${contextX} 86 ${contextX} ${86 + contextRadius} V140`
    );
    harness.querySelector("[data-path-key=\"plan-context\"]")?.setAttribute(
      "d",
      `M${contextX} 160 V204 Q${contextX} 214 ${contextX + 10} 214 `
        + "H724 Q734 214 734 204 V174 Q734 164 744 164 H766"
    );
  }

  const reasoning = projection.reasoning || {};
  harness.querySelectorAll("[data-reasoning-key]").forEach((node) => {
    const mode = node.dataset.reasoningKey;
    node.classList.toggle("selected", mode === reasoning.selected);
    setDecisionSvgState(node, reasoning.options?.[mode]);
    setDecisionSvgState(
      harness.querySelector(`[data-reasoning-output="${mode}"]`),
      reasoning.options?.[mode]
    );
  });
  setDecisionSvgText("[data-plan-value=\"fit\"]", fmt(projection.labels?.fit, "WAITING"));
  harness.querySelector("[data-plan-fit-pass]")?.classList.toggle(
    "visible",
    projection.labels?.fit_pass === true
  );

  const traceLanes = new Map((trace.lanes || []).map((lane) => [lane.key, lane]));
  Object.entries(projection.lanes || {}).forEach(([key, laneState]) => {
    const lane = traceLanes.get(key) || {};
    const laneElement = harness.querySelector(`[data-decision-lane="${key}"]`);
    if (!laneElement) return;
    setDecisionSvgState(laneElement, laneState.state);
    laneElement.classList.toggle("event-focus", focusEvent?.lane === key);
    setDecisionSvgText(`[data-model-value="${key}"]`, decisionTraceModelLabel(lane.model));
    setDecisionSvgText(`[data-lane-label="${key}"]`, fmt(lane.label, key));
    const think = fmt(lane.think, "—");
    setDecisionSvgText(`[data-lane-think="${key}"]`, think === "—" ? "—" : `think:${think}`);
    laneElement.querySelectorAll("[data-decision-lane-step]").forEach((node) => {
      const phase = node.dataset.decisionLaneStep;
      setDecisionSvgState(node, laneState.steps?.[phase]);
      node.classList.toggle(
        "trace-focus",
        focusEvent?.lane === key
          && (focusEvent.phase === "repair" ? phase === "validate" : focusEvent.phase === phase)
      );
    });
    laneElement.querySelectorAll("[data-lane-path]").forEach((node) => {
      setDecisionSvgState(node, laneState.rails?.[node.dataset.lanePath]);
    });
    setDecisionSvgState(
      laneElement.querySelector(`[data-repair-lane="${key}"]`),
      laneState.repair
    );
    setDecisionSvgText(`[data-repair-count="${key}"]`, "REPAIR JSON");
    setDecisionSvgText(`[data-repair-number="${key}"]`, String(laneState.repair_attempt || 0));
    const resultLabel = laneState.state === "pending"
      ? "WAITING"
      : laneState.state === "skipped"
        ? key === "tie_break" ? "STANDBY" : "NOT NEEDED"
        : laneState.state === "error"
          ? "INVALID"
          : laneState.state === "done"
            ? "VALID"
            : fmt(lane.result, laneState.state).toUpperCase();
    setDecisionSvgText(`[data-lane-result="${key}"]`, resultLabel);
    setDecisionSvgState(
      harness.querySelector(`[data-model-key="${key}"]`),
      projection.model_routes?.[key]
    );
  });
  setDecisionSvgText("[data-hold-reason=\"true\"]", fmt(projection.labels?.hold, "No safe quorum"));
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

function decisionConsoleText(event, trace) {
  const lane = event?.lane
    ? event.lane === "tie_break"
      ? "tie-break"
      : event.lane
    : "system";
  const model = shortName(event?.model || "local model");
  const generation = event?.generation || {};
  const tokens = numeric(generation.output_tokens)
    ? `${generation.token_count_exact ? "" : "~"}${generation.output_tokens} tok`
    : "";
  const speed = numeric(generation.tokens_per_second)
    ? `${generation.tokens_per_second.toFixed(1)} tok/s`
    : "";
  if (event?.kind === "session") {
    const result = event.status === "error"
      ? `session failed at ${fmt(event.phase, "runtime")}`
      : "vote accepted";
    return [lane, tokens, speed, result].filter(Boolean).join(" · ");
  }
  if (event?.phase === "trigger") return `dispatch ${lane} → ${model}`;
  if (event?.phase === "load") return `load ${model}`;
  if (event?.phase === "context") {
    const required = Number(event.required_context_tokens || 0);
    const selected = Number(event.context_tokens || event.requested_context_tokens || 0);
    const requiredLabel = required ? `${Math.ceil(required / 1024)}K` : "auto";
    const selectedLabel = selected ? `${Math.round(selected / 1024)}K` : "auto";
    return `context required ${requiredLabel} → selected ${selectedLabel}`;
  }
  if (event?.phase === "generate") {
    const think = fmt(event.think, "—").toLowerCase();
    return think === "off"
      ? "direct generation · reasoning bypassed"
      : `reasoning ${think} · generation started`;
  }
  if (event?.phase === "repair") {
    return `repair JSON · retry ${Number(event.attempt || 0)}`;
  }
  if (event?.phase === "validate") return "validate structured output";
  if (event?.phase === "vote") return `${lane} · vote ready`;
  if (event?.phase === "decision") {
    const target = trace?.quorum_flow === false ? 1 : trace?.tie_break_used ? 3 : 2;
    const votes = Number.isInteger(trace?.valid_votes)
      ? `${Math.min(trace.valid_votes, target)}/${target}`
      : "sealed";
    return `${fmt(event.label, "decision sealed")} · quorum ${votes} · ${fmt(trace?.outcome?.reason, trace?.summary)}`;
  }
  return decisionEventText(event);
}

function renderDecisionGeneration(trace) {
  const lane = (trace?.lanes || []).find(
    (item) => item.state === "active" && ["generate", "repair"].includes(item.phase)
  );
  els.decisionGenerationMeter.hidden = !lane;
  if (!lane) return;
  const liveEvent = [...(trace?.events || [])].reverse().find(
    (event) => event.lane === lane.key && ["generate", "repair"].includes(event.phase)
  );
  const generation = {
    ...(liveEvent?.generation || {}),
    ...(lane.generation || {}),
  };
  const output = numeric(generation.output_tokens) ? generation.output_tokens : 0;
  const maximum = numeric(generation.max_output_tokens)
    ? generation.max_output_tokens
    : 0;
  const speed = numeric(generation.tokens_per_second)
    ? generation.tokens_per_second
    : 0;
  const seconds = numeric(generation.generation_seconds)
    ? generation.generation_seconds
    : 0;
  const think = fmt(liveEvent?.think ?? lane.think, "—").toUpperCase();
  const percent = maximum ? Math.min(100, (output / maximum) * 100) : 0;
  els.decisionGenerationModel.textContent = shortName(liveEvent?.model || lane.model);
  els.decisionGenerationReasoning.textContent = think === "OFF"
    ? "DIRECT · NO REASONING"
    : `REASONING · ${think}`;
  els.decisionGenerationTokens.textContent = output
    ? `${generation.token_count_exact ? "" : "~"}${output.toLocaleString()} tok${maximum ? ` / ${maximum.toLocaleString()}` : ""}`
    : "warming up";
  els.decisionGenerationSpeed.textContent = speed ? `${speed.toFixed(1)} tok/s` : "-- tok/s";
  els.decisionGenerationTime.textContent = seconds ? `${seconds.toFixed(1)}s` : "awaiting first token";
  els.decisionGenerationFill.style.width = maximum ? `${Math.max(2, percent)}%` : "24%";
  els.decisionGenerationTrack.classList.toggle("indeterminate", !maximum || !output);
  els.decisionGenerationTrack.setAttribute("aria-valuemin", "0");
  els.decisionGenerationTrack.setAttribute("aria-valuemax", String(maximum || 1));
  els.decisionGenerationTrack.setAttribute("aria-valuenow", String(output));
}

function decisionTimelineSteps(trace) {
  const events = Array.isArray(trace?.events) ? trace.events : [];
  if (!events.length && Array.isArray(trace?.overall) && trace.overall.length) {
    return trace.overall.map((step) => ({ ...step }));
  }
  const steps = [];
  if (trace?.request_sha256) {
    steps.push({ key: "packet", label: "Packet", status: "done" });
  }
  const counts = new Map();
  events.forEach((event) => {
    const phase = fmt(event?.phase, "transition");
    const lane = fmt(event?.lane, "system");
    if (event?.kind === "session") {
      const vote = [...steps].reverse().find(
        (step) => step.lane === lane && step.phase === "vote"
      );
      if (vote && event.status !== "error") {
        vote.status = "done";
        return;
      }
    }
    const countKey = `${lane}:${phase}`;
    const count = Number(counts.get(countKey) || 0) + 1;
    counts.set(countKey, count);
    const laneLabel = lane === "tie_break"
      ? "Tie-break"
      : lane === "system"
        ? "System"
        : `${lane.charAt(0).toUpperCase()}${lane.slice(1)}`;
    const terminalSession = event?.kind === "session";
    const phaseLabel = phase === "decision"
      ? fmt(event?.label, "Decision")
      : terminalSession
        ? fmt(event?.label, "Vote result")
        : `${phase.charAt(0).toUpperCase()}${phase.slice(1)}`;
    steps.push({
      key: fmt(event?.event_id, `${lane}:${phase}:${count}`),
      label: phase === "decision"
        ? phaseLabel
        : terminalSession
          ? `${laneLabel} ${phaseLabel}`
          : `${laneLabel} ${phaseLabel} #${count}`,
      status: event?.status === "error" ? "error" : "done",
      lane,
      phase,
    });
  });
  const latestEvent = events[events.length - 1];
  if (latestEvent?.kind === "phase" && trace?.active) {
    const latest = steps.find((step) => step.key === latestEvent.event_id);
    if (latest && latest.status !== "error") latest.status = "active";
  }
  return steps;
}

function decisionTimelineCurrent(steps) {
  const index = steps.reduce(
    (latest, step, candidate) => step.status === "pending" ? latest : candidate,
    -1
  );
  return { position: index + 1, label: steps[index]?.label || "Waiting" };
}

function renderDecisionTransitionFeed(trace, focusEvent = null) {
  const rows = Array.isArray(trace?.events) ? trace.events : [];
  const visible = [...rows];
  if (
    focusEvent
    && !visible.some((event) => event.event_id === focusEvent.event_id)
  ) {
    visible.unshift(focusEvent);
  }
  const pinned = (
    els.decisionTransitionFeed.scrollHeight
    - els.decisionTransitionFeed.scrollTop
    - els.decisionTransitionFeed.clientHeight
  ) < 24;
  els.decisionTransitionFeed.textContent = "";
  els.decisionEventCount.textContent = String(rows.length);
  const latest = focusEvent || rows.at(-1);
  els.decisionTransitionDetail.textContent = latest
    ? decisionConsoleText(latest, trace)
    : "Waiting for observed work";
  visible.forEach((event) => {
    const item = document.createElement("li");
    item.className = "decision-transition-event";
    item.dataset.phase = fmt(event.phase, "transition");
    item.dataset.status = fmt(event.status, "active");
    item.classList.toggle(
      "current",
      Boolean(
        focusEvent
          ? event.event_id === focusEvent.event_id
          : trace?.active && event.event_id === rows.at(-1)?.event_id
      )
    );
    const timestamp = document.createElement("time");
    timestamp.textContent = timeLabel(event.timestamp);
    const prompt = document.createElement("span");
    prompt.className = "decision-console-mark";
    prompt.textContent = event.status === "error"
      ? "×"
      : event.phase === "trigger"
        ? "$"
        : event.phase === "repair"
          ? "↻"
          : event.phase === "generate" && trace?.active
            ? "›"
            : "✓";
    const message = document.createElement("span");
    message.textContent = decisionConsoleText(event, trace);
    item.append(timestamp, prompt, message);
    item.title = `${timeLabel(event.timestamp)} · ${message.textContent}`;
    els.decisionTransitionFeed.append(item);
  });
  renderDecisionGeneration(trace);
  if (pinned) els.decisionTransitionFeed.scrollTop = els.decisionTransitionFeed.scrollHeight;
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
  const traceLanes = Array.isArray(trace.lanes) ? trace.lanes : [];
  const contextLane = traceLanes.find((lane) => lane.state === "active")
    || [...traceLanes].reverse().find((lane) => lane.state === "done");
  const contextTokens = Number(contextLane?.context_tokens ?? trace.context_tokens ?? 0);
  els.decisionContext.textContent = contextTokens
    ? `Context ${Math.round(contextTokens / 1024)}K`
    : "Context --";

  const overall = decisionTimelineSteps(trace);
  updateDecisionSvgHarness(trace, focusEvent);
  updateProcessingTraceSelection(trace);

  const timelineCurrent = decisionTimelineCurrent(overall);
  const overallPosition = timelineCurrent.position;
  const overallStage = timelineCurrent.label;

  if (request) {
    const stateKind = active
      ? "running"
      : traceState === "agreed" || traceState === "ready"
        ? "complete"
        : traceState === "quarantined"
          ? "warning"
          : "idle";
    const observedModelCalls = (Array.isArray(trace.lanes) ? trace.lanes : []).filter((lane) =>
      ["active", "done", "error"].includes(lane.state)
    ).length;
    const observedValidVotes = (Array.isArray(trace.lanes) ? trace.lanes : []).filter(
      (lane) => lane.state === "done"
    ).length;
    const modelCalls = Number.isInteger(trace.vote_count)
      ? trace.vote_count
      : observedModelCalls;
    const validVotes = Number.isInteger(trace.valid_votes)
      ? trace.valid_votes
      : observedValidVotes;
    const tieBreakUsed = trace.tie_break_used === true
      || (Array.isArray(trace.lanes) ? trace.lanes : []).some((lane) =>
        lane.key === "tie_break" && ["active", "done", "error"].includes(lane.state)
      );
    const quorumTarget = trace.quorum_flow === false ? 1 : tieBreakUsed ? 3 : 2;
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
    els.workSummary.textContent = active
      ? `${overallStage} · step ${overallPosition}`
      : `${overallStage} · ${overall.length} observed steps`;
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
    els.workSummary.textContent = "Waiting · 0 / 3";
    els.workDetail.textContent = fmt(trace.summary, "No local decision yet");
    els.workUpdated.textContent = `${fmt(trace.task_role, "idle")} · no decision`;
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

function setDecisionTransitionState(trace) {
  const events = Array.isArray(trace?.events) ? trace.events : [];
  const latest = events[events.length - 1];
  els.decisionTransitionState.classList.remove("catching-up");
  if (trace?.active) {
    els.decisionTransitionState.textContent = latest
      ? `Live · ${fmt(latest.label, latest.phase)}`
      : "Live · observing";
  } else if (trace?.request_sha256) {
    els.decisionTransitionState.textContent = `Sealed · ${events.length} events`;
  } else {
    els.decisionTransitionState.textContent = "Live · waiting";
  }
}

function renderDecisionTrace(consensus) {
  const trace = consensus?.decision_trace || {};
  renderDecisionTraceFrame(trace);
  renderDecisionTransitionFeed(trace);
  setDecisionTransitionState(trace);
}
window.__chronovisorDashboardTest = Object.assign(window.__chronovisorDashboardTest || {}, {
  decisionTimelineSteps,
  decisionTraceProjection,
  processingLaneForTrace,
  selectProcessingLane,
  renderDecisionTrace,
  renderDecisionTraceFrame,
  renderProcessingActivity,
});

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
  const materialization = data.materialization || data.materialized || data._dashboard || {};
  const runtimeStatus = data.runtime_status || {};
  const ingestLiveness = data.ingest_liveness || {};
  const librarian = data.librarian || {};
  const authorityValue = data.current_authority
    ?? data.authority
    ?? materialization.current_authority
    ?? materialization.authority
    ?? runtimeStatus.current_authority
    ?? runtimeStatus.mutation_authority
    ?? runtimeStatus.authority_preflight
    ?? ingestLiveness.current_authority
    ?? ingestLiveness.mutation_authority
    ?? ingestLiveness.authority_preflight
    ?? librarian.authority;
  const authorityBlocked = runtimeStatus.mutation_ready === false
    || (runtimeStatus.mutation_authority
      && typeof runtimeStatus.mutation_authority === "object"
      && runtimeStatus.mutation_authority.ok === false);
  const authorityLabel = authorityBlocked
    ? "BLOCKED"
    : authorityValue && typeof authorityValue === "object"
    ? fmt(
        authorityValue.status
          || authorityValue.id
          || authorityValue.authority_digest
          || authorityValue.digest
          || authorityValue.epoch,
        "--",
      )
    : fmt(authorityValue, "--");
  const staleValue = data.stale ?? materialization.stale;
  const livenessStale = ingestLiveness.stale ?? ingestLiveness.liveness?.stale;
  const staleDisplay = staleValue === undefined
    ? livenessStale
    : Boolean(staleValue) || livenessStale === true;
  const refreshingValue = data.refreshing ?? materialization.refreshing;
  const materializedAgeValue = data.age
    ?? data.materialized_age_seconds
    ?? data.materialized_age
    ?? materialization.materialized_age_seconds
    ?? materialization.age_seconds
    ?? materialization.materialized_age
    ?? data.materialized_at
    ?? materialization.materialized_at;
  const materializedAge = numeric(materializedAgeValue)
    ? `${Math.round(Math.max(0, materializedAgeValue))}s`
    : typeof materializedAgeValue === "string" && /(?:ago|[smhd])$/u.test(materializedAgeValue)
      ? materializedAgeValue
      : ageLabel(materializedAgeValue);
  const healthMeta = [
    `status=${fmt(data.status ?? runtimeStatus.status ?? ingestLiveness.status, "unknown")}`,
    `authority=${authorityLabel}`,
    `stale=${staleDisplay === undefined ? "--" : staleDisplay ? "yes" : "no"}${staleDisplay === true ? " · STALE" : ""}`,
    `refreshing=${refreshingValue === undefined ? "--" : refreshingValue ? "yes" : "no"}`,
    `materialized=${materializedAge}`,
  ].join(" · ");
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
  const distillation = data.recall_distillation || {};
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
  const healthSummary = rawFiles
    ? `${claimed.toLocaleString()} / ${rawFiles.toLocaleString()} raw claimed${convergenceBits.length ? ` · ${convergenceBits.join(" · ")}` : ""}`
    : (convergenceBits.length ? convergenceBits.join(" · ") : "waiting");
  if (typeof els.healthCaption?.setAttribute === "function") {
    els.healthCaption.setAttribute("aria-live", "polite");
    els.healthCaption.setAttribute("aria-atomic", "true");
  }
  els.healthCaption.textContent = `${healthMeta} · ${healthSummary}`;
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
  const distillationPolicy = (value) => {
    const id = fmt(value);
    return id === "--" ? id : id.slice(0, 6);
  };
  const distillationWorker = fmt(distillation.worker_status || distillation.status);
  const distillationRollout = fmt(distillation.rollout_status, "");
  const distillationStage = distillationRollout && distillationRollout !== distillationWorker
    ? `/${distillationRollout}`
    : "";
  els.healthRecallDistillation.textContent = `${distillationWorker}${distillationStage} · ${Math.round(numeric(distillation.rollout_percent) ? distillation.rollout_percent : 0)}% · ${distillationPolicy(distillation.active_policy_id)}/${distillationPolicy(distillation.candidate_policy_id)}/${distillationPolicy(distillation.lkg_policy_id)}`;
  const hold = fmt(distillation.hold_reason, "");
  els.healthRecallDistillationDetail.textContent = `teacher-only ${intValue(distillation.teacher_only)} · verified truth ${intValue(distillation.verified_truth)} · probe ${intValue(distillation.probe_not_truth)} (not truth) · paired ${intValue(distillation.paired_denominator)} · ${fmt(distillation.feature_revision, "feature unavailable")}${hold ? ` · hold ${hold}` : ""}`;
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
const MODEL_ACTIVITY_LABELS = {
  trigger: "Starting",
  load: "Loading",
  context: "Context",
  generate: "Generating",
  repair: "Repairing",
  validate: "Validating",
  vote: "Voting",
};
let latestLiveModelSnapshot = null;

function modelStatusRank(row) {
  const order = { loaded: 0, missing: 1, ready: 2, external: 3, unknown: 4 };
  return order[row.status] ?? 9;
}

function modelRoleLabel(role) {
  return MODEL_ROLE_LABELS[role] || fmt(role);
}

function renderRuntimeFailures(failures) {
  const rows = Array.isArray(failures) ? failures.slice(0, 8) : [];
  els.modelFailureFeed.textContent = "";
  els.modelFailureFeed.hidden = !rows.length;
  rows.forEach((failure) => {
    const row = document.createElement("div");
    row.className = "event";
    const time = document.createElement("time");
    time.textContent = timeLabel(failure.timestamp);
    const badge = document.createElement("span");
    badge.className = "event-level failure";
    badge.textContent = fmt(failure.category, "failure");
    const message = document.createElement("span");
    message.className = "event-message";
    message.textContent = [
      failure.role,
      failure.configured_model || failure.provider,
      failure.capability,
      failure.location,
      intValue(failure.retry_count) ? `${intValue(failure.retry_count)} retries` : null,
      failure.request_id ? `request ${failure.request_id}` : null,
    ].filter(Boolean).join(" · ");
    row.append(time, badge, message);
    els.modelFailureFeed.appendChild(row);
  });
}

function renderModelStatus(modelStatus, runtimeFailures, activities = []) {
  const data = modelStatus || {};
  const summary = data.summary || {};
  const models = Array.isArray(data.models) ? [...data.models] : [];
  const activityByModel = new Map();
  (Array.isArray(activities) ? activities : []).forEach((activity) => {
    const model = String(activity?.model || "").replace(/:latest$/, "");
    if (model) activityByModel.set(model, activity);
  });
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
    : shortName(data.error || "Local runtime offline");
  renderRuntimeFailures(runtimeFailures);

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
      const activity = activityByModel.get(String(row.name || "").replace(/:latest$/, ""));
      const item = document.createElement("div");
      item.className = `model-row ${fmt(row.status, "unknown")}${activity ? " processing" : ""}`;

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
      if (activity) {
        const chip = document.createElement("span");
        chip.className = "model-role model-activity";
        chip.textContent = MODEL_ACTIVITY_LABELS[activity.phase] || fmt(activity.phase, "Active");
        chip.title = [activity.role, activity.phase].filter(Boolean).join(" · ");
        roleWrap.appendChild(chip);
      }
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
      meta.textContent = metaPieces.join(" · ") || `${row.provider || "local"} configured model`;
      meta.title = meta.textContent;

      item.append(main, roleWrap, meta);
      els.modelGrid.appendChild(item);
    });
}

function renderLiveModelStatus(snapshot, activities) {
  latestLiveModelSnapshot = {
    model_status: snapshot?.model_status || {},
    local_runtime: snapshot?.local_runtime || snapshot?.ollama || {},
    ollama: snapshot?.ollama || {},
    runtime_failures: snapshot?.runtime_failures || [],
    activities: Array.isArray(activities) ? activities : [],
  };
  renderLocalRuntimeMetric(
    latestLiveModelSnapshot.model_status,
    latestLiveModelSnapshot.local_runtime,
  );
  renderModelStatus(
    latestLiveModelSnapshot.model_status,
    latestLiveModelSnapshot.runtime_failures,
    latestLiveModelSnapshot.activities,
  );
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
    : growth.positive_learning_allowed || growth.field_learning_allowed
      ? "positive co-fire on · authority held"
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
    detailParts.push(`consensus ${fmt(audit.decision || "recommended")}`);
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
    const auditText = audit.decision ? ` · consensus ${audit.decision}` : "";
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

function renderLocalConsensusSummary(status) {
  const consensus = status.local_consensus || {};
  const consensusSummary = consensus.summary || {};
  const decisionSummary = consensusSummary.decisions || {};
  const evaluationSummary = (consensusSummary.evaluation || {}).decisions || {};
  const policyCounts = ((status.decision_policies || {}).counts) || {};
  const dissentEffects = decisionSummary.dissent_effect_classes || {};
  const dissentSummary = Object.entries(dissentEffects)
    .filter(([, count]) => intValue(count) > 0)
    .map(([effect, count]) => `${effect} ${intValue(count)}`)
    .join(" / ") || "none";
  const modelConservativeRates = decisionSummary.model_conservative_vote_rates || {};
  const conservativeVoteSummary = Object.entries(modelConservativeRates)
    .map(([modelName, counts]) => {
      const rate = numeric(counts?.conservative_rate)
        ? pctLabel(counts.conservative_rate)
        : "0.0%";
      return `${shortName(modelName)} ${rate}`;
    })
    .join(" / ") || "none";
  const vetoSummary = [
    `veto ${intValue(decisionSummary.conservative_veto_fired)}`,
    `bypassed ${intValue(decisionSummary.conservative_veto_bypassed_by_lane_policy)}`,
    `dissent ${dissentSummary}`,
    `conservative votes ${conservativeVoteSummary}`,
  ].join(" · ");
  const activeModels = (consensus.activities || [])
    .map((item) => [item.role, item.model].filter(Boolean).join(" · "))
    .filter(Boolean);
  els.localConsensus.textContent = consensus.active
    ? `${intValue(consensus.count)} active · ${activeModels.join(" · ")} · ${vetoSummary}`
    : `${intValue(decisionSummary.total)} routine · ${intValue(decisionSummary.pair_agreement)} pair · ${intValue(decisionSummary.tie_break_used)} tie · ${intValue(decisionSummary.unresolved_quarantine)} quarantined · ${vetoSummary} · ${intValue(evaluationSummary.total)} eval · ${intValue(policyCounts.shadow)} shadow / ${intValue(policyCounts.enabled)} enabled`;
}

function renderLocalRuntimeMetric(modelStatus, runtime) {
  const summary = modelStatus?.summary || {};
  const models = Array.isArray(runtime?.models) ? runtime.models : [];
  const model = models.find((item) => !String(item.name || item.model || "").includes("embed"))
    || models[0]
    || {};
  const degraded = runtime?.partial === true
    || runtime?.status === "degraded"
    || modelStatus?.partial === true
    || modelStatus?.status === "degraded";
  els.ollama.textContent = degraded
    ? "DEGRADED"
    : modelStatus?.available || runtime?.available
      ? "online"
      : "offline";
  if (typeof els.ollama?.setAttribute === "function") {
    els.ollama.setAttribute("aria-live", "polite");
    els.ollama.setAttribute("aria-atomic", "true");
  }
  els.ollamaSub.textContent = summary.installed !== undefined
    ? `${intValue(summary.loaded)} loaded · ${intValue(summary.installed)} installed`
    : model.name || model.model || "no model";
}

function render(snapshot) {
  const snapshotStatus = snapshot.status || {};
  const snapshotConsensus = snapshot.local_consensus || snapshotStatus.local_consensus || {};
  const status = {
    ...snapshotStatus,
    local_consensus: latestLiveConsensus || snapshotConsensus,
    frontier_repair: snapshot.frontier_repair || snapshotStatus.frontier_repair || {},
  };
  latestRenderedStatus = status;
  const metrics = snapshot.metrics || [];
  const batch = status.batch || {};
  const localRuntime = latestLiveModelSnapshot?.local_runtime
    || snapshot.local_runtime
    || snapshot.ollama
    || {};
  const modelStatus = latestLiveModelSnapshot?.model_status || snapshot.model_status || {};

  setState(status.state);
  const ready = intValue(status.pending);
  const semanticDeferred = intValue(status.semantic_deferred?.count);
  const operationalDeferred = intValue(status.operational_deferred?.count);
  const held = semanticDeferred + operationalDeferred;
  els.pending.textContent = fmt(ready);
  els.pendingSub.textContent = numeric(status.source_raw_pending)
    ? `${intValue(status.source_raw_pending)} source raws · ${held} held`
    : `${ready} work units ready`;
  els.held.textContent = fmt(held);
  els.heldSub.textContent = `${semanticDeferred} semantic · ${operationalDeferred} operational`;
  const stageValue = fmt(status.stage, "idle");
  els.stage.textContent = stageMetricLabel(stageValue);
  els.stage.title = stageValue;
  els.raw.textContent = shortName(status.current_raw || "no active raw");
  els.batch.textContent = batch.total ? `${batch.index || 0}/${batch.total}` : "--";
  els.batchSub.textContent = batch.total
    ? `${batch.succeeded || 0} ok / ${batch.deferred || 0} deferred / ${batch.continued || 0} continued / ${batch.failed || 0} fail`
    : "waiting";
  renderLocalRuntimeMetric(modelStatus, localRuntime);
  els.currentRaw.textContent = status.current_raw ? shortName(status.current_raw) : "waiting";
  els.currentOp.textContent = status.current_op ? fmt(status.current_op) : fmt(status.stage || "idle");
  renderWorkStatus(status);
  renderDecisionTrace(status.local_consensus || {});
  renderLlm(status.llm, status);
  renderLocalConsensusSummary(status);
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
  renderSelfHeal(snapshot.self_heal || {});
  renderRecall(snapshot.recall || {});
  renderRecallImprovement(snapshot.recall_improvement || {});
  renderSaveHistory(snapshot.save_history || {});
  renderKnowledgeMix(snapshot.knowledge_mix || {});
  renderLibrarian(snapshot.librarian || {});
  renderHealth(snapshot.health || {});
  renderModelStatus(
    modelStatus,
    latestLiveModelSnapshot?.runtime_failures || snapshot.runtime_failures || [],
    latestLiveModelSnapshot?.activities || status.local_consensus?.activities || [],
  );
  renderModelLab(snapshot.model_lab || {});
  renderEvents(snapshot.events || []);
  drawLineChart(els.pendingChart, snapshot.save_history || {}, status);
  drawBatchChart(els.batchChart, metrics, status);
}
