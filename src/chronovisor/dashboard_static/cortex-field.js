"use strict";

(() => {
  const EVENT_KINDS = new Set([
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
  const COMPONENTS = [
    "direct",
    "spread",
    "negative",
    "inhibition",
    "anti_index",
    "hub_penalty",
  ];
  const MAX_EVENTS = 256;

  function finite(value, fallback = 0) {
    return Number.isFinite(Number(value)) ? Number(value) : fallback;
  }

  function normalizedComponents(value) {
    const source = value && typeof value === "object" ? value : {};
    return Object.fromEntries(
      COMPONENTS.map((key) => [key, finite(source[key])]),
    );
  }

  function normalizedEvent(value) {
    if (!value || typeof value !== "object") return null;
    const seq = Number(value.seq);
    const sessionHash = String(value.session_hash || "");
    const kind = String(value.kind || "");
    if (!Number.isInteger(seq) || seq < 1 || !sessionHash || !EVENT_KINDS.has(kind)) {
      return null;
    }
    return {
      seq,
      timestamp_epoch: finite(value.timestamp_epoch),
      session_hash: sessionHash,
      topic_epoch: Math.max(0, Math.trunc(finite(value.topic_epoch))),
      kind,
      page_id: String(value.page_id || ""),
      source_page_id: String(value.source_page_id || ""),
      target_page_id: String(value.target_page_id || ""),
      edge_type: String(value.edge_type || ""),
      delta: finite(value.delta),
      activation: finite(value.activation),
      reason_code: String(value.reason_code || ""),
      certificate_id: String(value.certificate_id || ""),
      components: normalizedComponents(value.components),
      relation_id: String(value.components?.relation_id || ""),
      source: String(value.source || "stateful-recall-field"),
      received_at: performance.now(),
    };
  }

  function createState() {
    return {
      connection: "linking",
      status: "offline",
      source: "stateful-recall-field",
      mode: "off",
      sessionHash: "",
      sessions: [],
      topicEpoch: 0,
      turn: 0,
      seq: 0,
      updatedAt: 0,
      stale: true,
      fullSearchFallback: true,
      nodes: new Map(),
      events: [],
      lastByPage: new Map(),
      summary: {
        active: 0,
        candidate: 0,
        commit: 0,
        reject: 0,
        teacher_agreement: null,
        latency_ms: { p50: null, p95: null, max: null },
      },
      fault: "",
      seqGap: null,
      revision: 0,
    };
  }

  function applyProjection(state, projection) {
    const value = projection && typeof projection === "object" ? projection : {};
    const snapshot = value.snapshot && typeof value.snapshot === "object"
      ? value.snapshot
      : null;
    state.status = String(value.status || "offline");
    state.source = String(value.source || "stateful-recall-field");
    state.mode = String(value.mode || "off");
    state.sessionHash = String(value.session_hash || "");
    state.sessions = Array.isArray(value.sessions) ? value.sessions.slice(0, 12) : [];
    state.stale = Boolean(value.summary?.stale ?? true);
    state.summary = {
      ...state.summary,
      ...(value.summary || {}),
      latency_ms: {
        ...state.summary.latency_ms,
        ...(value.summary?.latency_ms || {}),
      },
    };
    state.nodes = new Map();
    state.lastByPage = new Map();
    state.events = [];
    state.fault = ["fault", "offline"].includes(state.status) ? state.status : "";
    state.seqGap = null;
    if (snapshot) {
      state.topicEpoch = Math.max(0, Math.trunc(finite(snapshot.topic_epoch)));
      state.turn = Math.max(0, Math.trunc(finite(snapshot.turn)));
      state.seq = Math.max(0, Math.trunc(finite(snapshot.seq)));
      state.updatedAt = finite(snapshot.updated_at_epoch);
      state.fullSearchFallback = Boolean(snapshot.full_search_fallback);
      (Array.isArray(snapshot.nodes) ? snapshot.nodes : []).forEach((row) => {
        const pageId = String(row?.page_id || "");
        if (!pageId) return;
        state.nodes.set(pageId, {
          activation: finite(row.activation),
          components: normalizedComponents(row.components),
          lastSeq: Math.max(0, Math.trunc(finite(row.last_seq))),
          certificateId: "",
          reasonCode: "",
          state: "active",
        });
      });
    } else {
      state.topicEpoch = 0;
      state.turn = 0;
      state.seq = 0;
      state.updatedAt = 0;
    }
    const history = (Array.isArray(value.events) ? value.events : [])
      .map(normalizedEvent)
      .filter((event) => event && event.session_hash === state.sessionHash)
      .slice(-MAX_EVENTS);
    state.events = history;
    history.forEach((event) => updatePageTrace(state, event, false));
    state.revision += 1;
    return state;
  }

  function eventPageId(event) {
    return event.page_id || event.target_page_id || event.source_page_id;
  }

  function updatePageTrace(state, event, updateActivation = true) {
    const pageId = eventPageId(event);
    if (!pageId) return;
    const previous = state.nodes.get(pageId) || {
      activation: 0,
      components: normalizedComponents({}),
      lastSeq: 0,
      certificateId: "",
      reasonCode: "",
      state: "inactive",
    };
    const components = { ...previous.components };
    COMPONENTS.forEach((key) => {
      if (event.components[key]) components[key] = event.components[key];
    });
    const next = {
      ...previous,
      components,
      lastSeq: event.seq,
      certificateId: event.certificate_id || previous.certificateId,
      reasonCode: event.reason_code || previous.reasonCode,
      state: event.kind,
    };
    if (updateActivation && Number.isFinite(event.activation)) {
      next.activation = event.activation;
    }
    state.nodes.set(pageId, next);
    state.lastByPage.set(pageId, event);
  }

  function applyEvents(state, values) {
    const accepted = [];
    for (const value of Array.isArray(values) ? values : []) {
      const event = normalizedEvent(value);
      if (!event || event.session_hash !== state.sessionHash) continue;
      if (event.seq <= state.seq) continue;
      if (event.seq !== state.seq + 1) {
        state.seqGap = { expected: state.seq + 1, received: event.seq };
        state.fault = `seq gap ${state.seq + 1}→${event.seq}`;
        state.status = "fault";
        break;
      }
      state.seq = event.seq;
      state.topicEpoch = event.topic_epoch;
      state.updatedAt = Math.max(state.updatedAt, event.timestamp_epoch);
      state.events.push(event);
      if (state.events.length > MAX_EVENTS) state.events.shift();
      updatePageTrace(state, event);
      if (event.kind === "commit_queued" || event.kind === "commit_applied") {
        state.summary.commit = Number(state.summary.commit || 0) + 1;
      }
      if (event.kind === "reject" || event.kind === "inhibit") {
        state.summary.reject = Number(state.summary.reject || 0) + 1;
      }
      accepted.push(event);
    }
    if (accepted.length) {
      state.stale = false;
      const active = [...state.nodes.values()].filter(
        (node) => node.activation >= 0.05,
      ).length;
      state.summary.active = active;
      state.summary.candidate = Math.min(30, active);
      state.revision += 1;
    }
    return accepted;
  }

  function setConnection(state, connection) {
    state.connection = connection;
    if (connection === "offline" && !state.fault) state.status = "offline";
    state.revision += 1;
  }

  function activeNodes(state, limit = 30) {
    return [...state.nodes.entries()]
      .map(([pageId, value]) => ({ pageId, ...value }))
      .filter((row) => row.activation > 0)
      .sort((left, right) => right.activation - left.activation || left.pageId.localeCompare(right.pageId))
      .slice(0, limit);
  }

  window.CortexField = Object.freeze({
    MAX_EVENTS,
    createState,
    applyProjection,
    applyEvents,
    setConnection,
    activeNodes,
    eventPageId,
  });
})();
