let refreshInFlight = null;
let hasRenderedFullSnapshot = false;
const SNAPSHOT_TIMEOUT_MS = 180000;
const FAST_SNAPSHOT_TIMEOUT_MS = 3000;
const ACTIVE_REFRESH_DELAY_MS = 5000;
const IDLE_REFRESH_DELAY_MS = 10000;
const ERROR_REFRESH_DELAY_MS = 5000;
const DECISION_REFRESH_TIMEOUT_MS = 2500;
const MODEL_STATUS_REFRESH_TIMEOUT_MS = 3500;
const ACTIVE_DECISION_REFRESH_DELAY_MS = 800;
const IDLE_DECISION_REFRESH_DELAY_MS = 2500;
const PROCESSING_FALLBACK_DELAY_MS = 1000;
let nextRefreshDelayMs = IDLE_REFRESH_DELAY_MS;
let decisionRefreshInFlight = false;
let modelStatusRefreshInFlight = false;
let nextDecisionRefreshDelayMs = IDLE_DECISION_REFRESH_DELAY_MS;
let decisionTracePinnedRequest = "";
let decisionTraceSelectedPipeline = "";
let processingRefreshInFlight = false;
let processingEventSource = null;

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

function renderLiveConsensus(consensus) {
  const currentConsensus = latestRenderedStatus?.local_consensus || {};
  const liveConsensus = consensus && typeof consensus === "object" ? consensus : {};
  const mergedConsensus = {
    ...currentConsensus,
    ...liveConsensus,
    summary: {
      ...(currentConsensus.summary || {}),
      ...(liveConsensus.summary || {}),
    },
  };
  latestLiveConsensus = mergedConsensus;
  if (!latestRenderedStatus) {
    renderDecisionTrace(mergedConsensus);
    return;
  }
  latestRenderedStatus = {
    ...latestRenderedStatus,
    local_consensus: mergedConsensus,
  };
  const underlyingState = String(latestRenderedStatus.state || "").toLowerCase();
  const displayState = ["error", "blocked"].includes(underlyingState)
    ? underlyingState
    : mergedConsensus.active ? "running" : latestRenderedStatus.state;
  setState(displayState);
  renderLocalConsensusSummary(latestRenderedStatus);
  renderWorkStatus(latestRenderedStatus);
  renderDecisionTrace(mergedConsensus);
}

async function refreshLiveModelStatus(activities) {
  if (modelStatusRefreshInFlight) return;
  modelStatusRefreshInFlight = true;
  const controller = new AbortController();
  const timeoutId = window.setTimeout(
    () => controller.abort(),
    MODEL_STATUS_REFRESH_TIMEOUT_MS,
  );
  try {
    const response = await fetch("/api/model-status", {
      cache: "no-store",
      signal: controller.signal,
    });
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    renderLiveModelStatus(await response.json(), activities);
  } catch {
    // Keep the last truthful model snapshot visible during a transient failure.
  } finally {
    window.clearTimeout(timeoutId);
    modelStatusRefreshInFlight = false;
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
    const selectedPipeline = decisionTraceSelectedPipeline;
    const query = selectedPipeline
      ? `?pipeline=${encodeURIComponent(selectedPipeline)}`
      : decisionTracePinnedRequest
      ? `?next=active&request_sha256=${encodeURIComponent(decisionTracePinnedRequest)}`
      : "?next=active";
    const response = await fetch("/api/local-consensus" + query, {
      cache: "no-store",
      signal: controller.signal,
    });
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    const consensus = (await response.json()).local_consensus || {};
    if (selectedPipeline !== decisionTraceSelectedPipeline) return;
    const trace = consensus.decision_trace || {};
    if (!selectedPipeline && trace.request_sha256) {
      decisionTracePinnedRequest = String(trace.request_sha256);
    }
    renderLiveConsensus(consensus);
    void refreshLiveModelStatus(consensus.activities || []);
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

window.addEventListener("chronovisor:processing-lane-select", (event) => {
  const pipeline = String(event.detail?.pipeline || "");
  if (!pipeline) return;
  decisionTraceSelectedPipeline = pipeline;
  decisionTracePinnedRequest = "";
  void refreshDecisionTrace();
});

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

async function refreshProcessingActivity() {
  if (processingRefreshInFlight) return;
  processingRefreshInFlight = true;
  try {
    const response = await fetch("/api/activity", { cache: "no-store" });
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    renderProcessingActivity(await response.json());
    const streamOpen = processingEventSource
      && processingEventSource.readyState === EventSource.OPEN;
    setProcessingConnection(
      streamOpen ? "live" : "polling",
      streamOpen ? "LIVE · ≤250MS" : "POLLING"
    );
  } catch {
    setProcessingConnection("connecting", "RECONNECTING");
  } finally {
    processingRefreshInFlight = false;
  }
}

function connectProcessingActivityStream() {
  if (!("EventSource" in window)) {
    setProcessingConnection("polling", "POLLING");
    return;
  }
  processingEventSource = new EventSource("/api/activity-stream");
  processingEventSource.addEventListener("activity", (event) => {
    try {
      renderProcessingActivity(JSON.parse(event.data));
      setProcessingConnection("live", "LIVE · ≤250MS");
    } catch {
      setProcessingConnection("polling", "POLLING");
    }
  });
  processingEventSource.onopen = () => {
    setProcessingConnection("live", "LIVE · ≤250MS");
  };
  processingEventSource.onerror = () => {
    setProcessingConnection("connecting", "RECONNECTING");
  };
}

async function processingFallbackLoop() {
  const streamOpen = processingEventSource
    && processingEventSource.readyState === EventSource.OPEN;
  if (!streamOpen) await refreshProcessingActivity();
  window.setTimeout(processingFallbackLoop, PROCESSING_FALLBACK_DELAY_MS);
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
void refreshProcessingActivity();
connectProcessingActivityStream();
void processingFallbackLoop();
window.setInterval(updateProcessingElapsed, 1000);
window.addEventListener("resize", refresh);
