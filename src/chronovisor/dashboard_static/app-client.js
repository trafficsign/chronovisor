let refreshInFlight = null;
const SNAPSHOT_TIMEOUT_MS = 180000;

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
      document.body.dataset.snapshotState = "full";
    } catch (error) {
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
    await refresh();
  } catch {
    // refresh() reports normal fetch failures; keep polling after unexpected ones.
  } finally {
    window.setTimeout(refreshLoop, 10000);
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
window.addEventListener("resize", refresh);
document.addEventListener("visibilitychange", () => {
  if (!document.hidden) void refresh();
});
