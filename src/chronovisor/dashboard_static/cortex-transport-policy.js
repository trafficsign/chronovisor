"use strict";

((root, factory) => {
  const api = factory();
  if (typeof module === "object" && module.exports) module.exports = api;
  if (root) root.CortexTransportPolicy = api;
})(typeof globalThis === "object" ? globalThis : this, () => {
  const EVENT_SCHEMA = "chronovisor.cortex.event.v2";
  // Must match _CORTEX_PAGE_ID_MAX_LENGTH at the Python projection boundary.
  const PAGE_ID_MAX_LENGTH = 240;
  const MAX_ELECTRIC_PATHS = 12;
  const MAX_TRANSPORT_EFFECTS = 18;
  const PROCESSING_EFFECT_PULSE_MS = 1450;
  const PROCESSING_LANE_COMPLETE_HOLD_MS = 1800;
  const PROCESSING_LANE_ACTIVE_HOLD_MS = 4800;
  const PROCESSING_LANE_KEYS = Object.freeze([
    "raw_buffer",
    "ingest",
    "recall",
    "audit",
    "improve",
    "repair",
    "typed_graph",
  ]);
  const PROCESSING_LANE_KEY_SET = new Set(PROCESSING_LANE_KEYS);
  const PROCESSING_ACTIVITY_LANE_KEY_SET = new Set(
    PROCESSING_LANE_KEYS.filter((laneKey) => laneKey !== "raw_buffer"),
  );
  const RECALL_KINDS = new Set([
    "recall",
    "auto_recall",
    "read",
    "search",
    "used",
  ]);
  const TRANSPORT_KINDS = new Set([
    "save",
    "capture",
    "ingest",
    ...RECALL_KINDS,
    "processing",
  ]);
  const FIELD_KINDS = new Set([
    "stimulus",
    "spread",
    "inhibit",
    "reject",
    "commit_queued",
    "commit_applied",
    "topic_reset",
    "snapshot",
    "fault",
  ]);
  const V2_FAMILIES = new Set(["field", "transport", "telemetry"]);
  const VISUAL_PROFILES = Object.freeze({
    demo: Object.freeze({
      mode: "demo",
      captureWaveDurationMs: 3600,
      recallNodeDurationMs: 650,
      recallElectricDurationMs: null,
      recallTransportMinVisibleMs: 3200,
    }),
    live: Object.freeze({
      mode: "live",
      captureWaveDurationMs: 5000,
      recallNodeDurationMs: 3000,
      recallElectricDurationMs: 2800,
      recallTransportMinVisibleMs: 3200,
    }),
  });

  function finite(value, fallback = 0) {
    const number = Number(value);
    return Number.isFinite(number) ? number : fallback;
  }

  function text(value, fallback = "") {
    const result = String(value ?? fallback);
    return result || String(fallback);
  }

  function lower(value, fallback = "") {
    return text(value, fallback).toLowerCase();
  }

  function isRecallKind(kind) {
    return RECALL_KINDS.has(lower(kind));
  }

  function isTransportKind(kind) {
    return TRANSPORT_KINDS.has(lower(kind));
  }

  function modeFor(value) {
    const requested = lower(value?.mode || value?.presentation?.mode);
    if (requested === "demo") return "demo";
    return lower(value?.source) === "demo" ? "demo" : "live";
  }

  function profileFor(value) {
    const mode = typeof value === "string" ? lower(value) : modeFor(value);
    return mode === "demo" ? VISUAL_PROFILES.demo : VISUAL_PROFILES.live;
  }

  function processingEffectPhase(step) {
    const normalized = lower(step);
    if (["raw", "triage", "search", "select", "discover", "detect"].includes(normalized)) {
      return "triage";
    }
    if (
      [
        "generate",
        "rerank",
        "primary",
        "challenger",
        "inspect",
        "extract",
        "local_fix",
      ].includes(normalized)
    ) {
      return "generate";
    }
    if (["consensus", "tie_break", "verify", "evaluate", "escalate"].includes(normalized)) {
      return "consensus";
    }
    if (["apply", "commit", "report", "consolidate", "promote"].includes(normalized)) {
      return "apply";
    }
    return "generate";
  }

  function laneKeyFor(kind, requestedLaneKey = "") {
    if (kind === "save" || kind === "capture") return "raw_buffer";
    if (kind === "ingest") return "ingest";
    if (isRecallKind(kind)) return "recall";
    if (kind !== "processing") return "";
    const laneKey = lower(requestedLaneKey);
    return PROCESSING_ACTIVITY_LANE_KEY_SET.has(laneKey) ? laneKey : "audit";
  }

  function phaseFor(kind, requestedPhase = "", step = "") {
    if (kind === "save" || kind === "capture") return "capture";
    if (kind === "search") return "triage";
    if (kind === "used") return "apply";
    if (["recall", "auto_recall", "read"].includes(kind)) return "generate";
    if (kind === "processing" && !requestedPhase) return processingEffectPhase(step);
    return lower(requestedPhase, "generate");
  }

  function legacyFamilyFor(value, kind) {
    const source = lower(value?.source);
    if (source === "stateful-recall-field") return "field";
    if (isTransportKind(kind)) return "transport";
    return source === "telemetry-fallback" ? "telemetry" : "field";
  }

  function validV2Envelope(value, kind, family) {
    if (!V2_FAMILIES.has(family)) return false;
    const source = lower(value?.source);
    if (family === "transport") {
      return isTransportKind(kind) && source !== "stateful-recall-field";
    }
    if (family === "field") {
      return FIELD_KINDS.has(kind) && (!source || source === "stateful-recall-field");
    }
    return (
      source === "telemetry-fallback"
      && !isTransportKind(kind)
      && !FIELD_KINDS.has(kind)
    );
  }

  function normalizePageIds(value) {
    if (!Array.isArray(value)) return [];
    const result = [];
    const seen = new Set();
    for (const pageId of value) {
      if (typeof pageId !== "string" || !pageId) continue;
      const bounded = pageId.slice(0, PAGE_ID_MAX_LENGTH);
      if (!bounded || seen.has(bounded)) continue;
      seen.add(bounded);
      result.push(bounded);
      if (result.length >= 24) break;
    }
    return result;
  }

  function priorityClassFor(kind, requested = "") {
    const priorityClass = lower(requested);
    if (priorityClass) return priorityClass;
    if (kind === "processing") return "processing";
    if (kind === "save" || kind === "capture" || isRecallKind(kind)) {
      return "protected";
    }
    return "standard";
  }

  function normalizeEvent(value) {
    if (!value || typeof value !== "object" || Array.isArray(value)) return null;
    const suppliedSchema = value.schema;
    const hasSchema = suppliedSchema !== undefined
      && suppliedSchema !== null
      && String(suppliedSchema) !== "";
    if (hasSchema && suppliedSchema !== EVENT_SCHEMA) return null;
    const isV2 = suppliedSchema === EVENT_SCHEMA;
    const presentation = isV2
      && value.presentation
      && typeof value.presentation === "object"
      ? value.presentation
      : {};
    const kind = lower(value.kind);
    if (!kind) return null;
    const family = isV2 ? lower(value.family) : legacyFamilyFor(value, kind);
    if (isV2 && !validV2Envelope(value, kind, family)) return null;
    const mode = modeFor(value);
    const laneKey = laneKeyFor(
      kind,
      presentation.lane_key ?? value.lane_key,
    );
    const phase = phaseFor(
      kind,
      presentation.phase ?? value.phase,
      value.step,
    );
    const channelKey = text(
      presentation.channel_key ?? value.channel_key,
      kind === "processing"
        ? `processing:${laneKey || "audit"}`
        : kind || "memory",
    ).slice(0, 160);
    const priorityClass = priorityClassFor(
      kind,
      presentation.priority_class ?? value.priority_class,
    );
    const pageIds = normalizePageIds(value.page_ids);
    return {
      ...value,
      schema: EVENT_SCHEMA,
      family,
      origin: text(value.origin, value.source || "legacy"),
      mode,
      source: text(value.source, family === "field" ? "stateful-recall-field" : "telemetry-fallback"),
      kind,
      page_ids: pageIds,
      lane_key: laneKey,
      phase,
      channel_key: channelKey,
      priority_class: priorityClass,
      presentation: {
        ...presentation,
        lane_key: laneKey,
        phase,
        channel_key: channelKey,
        priority_class: priorityClass,
      },
    };
  }

  function normalizeProcessingActivity(lane, phase = "") {
    if (!lane || typeof lane !== "object") return null;
    const laneKey = text(lane.key, "process");
    const step = text(lane.current_step || lane.phase, "work");
    return normalizeEvent({
      schema: EVENT_SCHEMA,
      family: "transport",
      origin: "activity-stream",
      mode: "live",
      source: "processing-activity",
      kind: "processing",
      phase: phase || processingEffectPhase(step),
      lane_key: laneKey,
      step,
      label: `${lane.label || laneKey} · ${step}`,
      model: text(lane.model),
      role: text(lane.role),
      page_ids: [],
    });
  }

  function formatBytes(value) {
    const bytes = finite(value);
    if (!bytes) return "—";
    if (bytes < 1024) return `${bytes} B`;
    if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
    return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
  }

  function projectProcessingLane(value, sourcePhase = "") {
    const candidate = {
      ...value,
      phase: sourcePhase || value?.phase,
    };
    if (sourcePhase && candidate.presentation) {
      candidate.presentation = { ...candidate.presentation, phase: sourcePhase };
    }
    const event = normalizeEvent(candidate);
    if (!event || !PROCESSING_LANE_KEY_SET.has(event.lane_key)) return null;
    const phase = event.phase;
    let statusValue;
    if (phase === "complete") statusValue = "complete";
    else if (event.kind === "processing") statusValue = event.step || phase;
    else if (event.kind === "save" || event.kind === "capture") statusValue = "capture";
    else if (event.kind === "ingest") statusValue = value.phase || phase || "ingest";
    else statusValue = event.kind || phase;
    const status = text(statusValue, "active")
      .replaceAll("_", " ")
      .toUpperCase()
      .slice(0, 32);
    let detail;
    if (event.kind === "processing") {
      detail = ([event.model, event.role]
        .filter(Boolean)
        .map(String)
        .join(" · ") || text(event.label, "live processing")).slice(0, 160);
    } else if (event.lane_key === "raw_buffer") {
      const count = Math.max(1, finite(event.raw_count, 1));
      const captureId = text(event.capture_id, "pending").slice(0, 12);
      detail = `${formatBytes(event.byte_count)} · ${count} raw · ID ${captureId}`;
    } else {
      const pageId = event.page_ids[0] || "";
      if (pageId) {
        const operation = text(
          event.operation || (event.kind === "ingest" ? value.phase || "" : ""),
        ).toUpperCase();
        detail = `${operation ? `${operation} · ` : ""}${pageId}`.slice(0, 160);
      } else {
        detail = text(event.label, "local pipeline active").slice(0, 160);
      }
    }
    const state = phase === "complete" ? "complete" : "active";
    const holdMs = phase === "complete"
      ? PROCESSING_LANE_COMPLETE_HOLD_MS
      : phase === "capture"
        ? profileFor(event).captureWaveDurationMs + 400
        : PROCESSING_LANE_ACTIVE_HOLD_MS;
    return {
      event,
      laneKey: event.lane_key,
      phase,
      status,
      detail,
      state,
      holdMs,
      mode: event.mode,
    };
  }

  function processingLaneUpdateDecision(projection, lastLiveAt, updateAt) {
    const previousLiveAt = Math.max(-1, finite(lastLiveAt, -1));
    if (!projection) return { accept: false, lastLiveAt: previousLiveAt };
    if (projection.mode === "live") {
      return {
        accept: true,
        lastLiveAt: Math.max(previousLiveAt, finite(updateAt)),
      };
    }
    const demoStartedAt = Math.max(
      0,
      finite(projection.event?.demo_sequence_started_at),
    );
    return {
      accept: previousLiveAt < 0 || demoStartedAt > previousLiveAt,
      lastLiveAt: previousLiveAt,
    };
  }

  function transportTiming(value, requestedPhase, now) {
    const event = normalizeEvent(value);
    const phase = lower(requestedPhase || event?.phase, "generate");
    const startedAt = finite(now);
    const profile = profileFor(event || value);
    const baseDuration = phase === "capture"
      ? profile.captureWaveDurationMs
      : phase === "apply" || phase === "complete"
        ? 3600
        : phase === "consensus"
          ? 2600
          : phase === "generate"
            ? 2300
            : 2100;
    const recall = isRecallKind(event?.kind);
    const duration = recall
      ? Math.max(baseDuration, profile.recallTransportMinVisibleMs)
      : baseDuration;
    const retainedUntil = phase === "capture"
      ? startedAt + baseDuration
      : recall
        ? startedAt + profile.recallTransportMinVisibleMs
        : startedAt;
    return { duration, retainedUntil };
  }

  function recallVisualProfile(value) {
    const event = normalizeEvent(value);
    if (!event || !isRecallKind(event.kind)) return null;
    const profile = profileFor(event);
    return {
      mode: profile.mode,
      scale: event.kind === "auto_recall" ? 0.9 : 0.6,
      nodeDurationMs: profile.recallNodeDurationMs,
      electricDurationMs: profile.recallElectricDurationMs,
      electricRetainMs: profile.recallElectricDurationMs,
    };
  }

  function liveRecallElectricTiming(value, startedAt) {
    const recallProfile = recallVisualProfile(value);
    if (
      !recallProfile
      || recallProfile.mode !== "live"
      || !Number.isFinite(recallProfile.electricDurationMs)
    ) {
      return null;
    }
    const start = finite(startedAt);
    return {
      duration: recallProfile.electricDurationMs,
      retainedUntil: start + recallProfile.electricRetainMs,
    };
  }

  function isTransportEffectProtected(effect, now) {
    return finite(effect?.retainedUntil) > finite(now);
  }

  function supersedeCaptureVisuals(effects) {
    if (!Array.isArray(effects)) return effects;
    effects.forEach((effect) => {
      if (lower(effect?.phase) === "capture") {
        effect.captureVisualSuperseded = true;
      }
    });
    return effects;
  }

  function transportEvictionIndex(effects, now) {
    const supersededCaptureIndex = effects.findIndex(
      (effect) =>
        effect.captureVisualSuperseded === true
        && lower(effect.phase) === "capture",
    );
    if (supersededCaptureIndex >= 0) return supersededCaptureIndex;
    const processingIndex = effects.findIndex(
      (effect) =>
        (effect.kind === "processing" || effect.priorityClass === "processing")
        && !isTransportEffectProtected(effect, now),
    );
    if (processingIndex >= 0) return processingIndex;
    const unprotectedIndex = effects.findIndex(
      (effect) => !isTransportEffectProtected(effect, now),
    );
    if (unprotectedIndex >= 0) return unprotectedIndex;
    let latestCaptureIndex = -1;
    effects.forEach((effect, index) => {
      if (
        lower(effect.phase) !== "capture"
        || effect.captureVisualSuperseded === true
      ) return;
      if (
        latestCaptureIndex < 0
        || finite(effect.seq) > finite(effects[latestCaptureIndex].seq)
      ) latestCaptureIndex = index;
    });
    const protectLatestCapture = latestCaptureIndex >= 0 && effects.length > 1;
    return effects.reduce((candidateIndex, effect, index) => {
      if (protectLatestCapture && index === latestCaptureIndex) {
        return candidateIndex;
      }
      if (candidateIndex < 0) return index;
      const candidate = effects[candidateIndex];
      return finite(effect.retainedUntil) < finite(candidate.retainedUntil)
        || (
          finite(effect.retainedUntil) === finite(candidate.retainedUntil)
          && finite(effect.startedAt) < finite(candidate.startedAt)
        )
        ? index
        : candidateIndex;
    }, -1);
  }

  function pruneAndBoundTransportEffects(
    effects,
    now,
    maxEffects = MAX_TRANSPORT_EFFECTS,
  ) {
    for (let index = effects.length - 1; index >= 0; index -= 1) {
      const effect = effects[index];
      if (finite(effect.startedAt) + finite(effect.duration) <= finite(now)) {
        effects.splice(index, 1);
      }
    }
    const boundedLimit = Math.max(0, Math.floor(finite(maxEffects)));
    while (effects.length > boundedLimit) {
      const evictionIndex = transportEvictionIndex(effects, now);
      if (evictionIndex < 0) break;
      effects.splice(evictionIndex, 1);
    }
    return effects;
  }

  function isElectricPulseProtected(pulse, now) {
    return finite(pulse?.retainedUntil) > finite(now);
  }

  function pruneAndBoundElectricPulses(
    pulseQueue,
    now,
    maxPulses = MAX_ELECTRIC_PATHS,
  ) {
    for (let index = pulseQueue.length - 1; index >= 0; index -= 1) {
      const pulse = pulseQueue[index];
      if (finite(pulse.startedAt) + finite(pulse.duration) <= finite(now)) {
        pulseQueue.splice(index, 1);
      }
    }
    pulseQueue.sort(
      (left, right) =>
        finite(right.delta) - finite(left.delta)
        || finite(right.startedAt) - finite(left.startedAt)
        || finite(right.seq) - finite(left.seq),
    );
    const boundedLimit = Math.max(0, Math.floor(finite(maxPulses)));
    while (pulseQueue.length > boundedLimit) {
      let evictionIndex = -1;
      for (let index = pulseQueue.length - 1; index >= 0; index -= 1) {
        if (!isElectricPulseProtected(pulseQueue[index], now)) {
          evictionIndex = index;
          break;
        }
      }
      pulseQueue.splice(evictionIndex >= 0 ? evictionIndex : -1, 1);
    }
    return pulseQueue;
  }

  return Object.freeze({
    EVENT_SCHEMA,
    PAGE_ID_MAX_LENGTH,
    MAX_ELECTRIC_PATHS,
    MAX_TRANSPORT_EFFECTS,
    PROCESSING_EFFECT_PULSE_MS,
    PROCESSING_LANE_COMPLETE_HOLD_MS,
    PROCESSING_LANE_ACTIVE_HOLD_MS,
    PROCESSING_LANE_KEYS,
    VISUAL_PROFILES,
    isRecallKind,
    isTransportKind,
    normalizePageIds,
    normalizeEvent,
    normalizeProcessingActivity,
    processingEffectPhase,
    processingLaneUpdateDecision,
    projectProcessingLane,
    profileFor,
    recallVisualProfile,
    liveRecallElectricTiming,
    transportTiming,
    supersedeCaptureVisuals,
    isTransportEffectProtected,
    pruneAndBoundTransportEffects,
    isElectricPulseProtected,
    pruneAndBoundElectricPulses,
  });
});
