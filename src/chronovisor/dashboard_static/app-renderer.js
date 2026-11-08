function fmt(value, fallback = "--") {
  if (value === null || value === undefined || value === "") return fallback;
  return String(value);
}

function timeLabel(value) {
  if (!value) return "--";
  const date = new Date(value);
  if (!Number.isNaN(date.getTime())) {
    return date.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit", second: "2-digit" });
  }
  return String(value).slice(-8);
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

function setState(state) {
  const normalized = (state || "unknown").toLowerCase();
  els.statePill.classList.remove("running", "error");
  if (normalized === "running") els.statePill.classList.add("running");
  if (normalized === "error" || normalized === "blocked") els.statePill.classList.add("error");
  els.stateText.textContent = normalized;
}

function dateKeyLabel(dateKey) {
  const date = parseDateKey(dateKey);
  if (!date) return fmt(dateKey);
  return date.toLocaleDateString("ja-JP", { month: "numeric", day: "numeric" });
}

function saveLoadRows(saveHistory) {
  const days = Array.isArray(saveHistory?.days) ? saveHistory.days.slice(-30) : [];
  return days.map((day) => ({
    date: day.date,
    total: intValue(day.raw_bytes),
    segments: Array.isArray(day.raw_segments) ? day.raw_segments : [],
  }));
}

function segmentColor(bytes, maxBytes) {
  const ratio = maxBytes > 0 ? Math.min(1, Math.sqrt(bytes / maxBytes)) : 0;
  return `hsl(126, 43%, ${40 + ratio * 30}%)`;
}

function drawStackSegment(ctx, x, baseY, width, height, color) {
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
  meta.textContent = `captured · ${region.source}`;
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

function drawLineChart(canvas, saveHistory) {
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

  const rows = saveLoadRows(saveHistory);
  saveLoadHitRegions = [];
  const total = rows.reduce((sum, row) => sum + row.total, 0);
  els.pendingCurrent.textContent = formatBytes(total);
  els.trendCaption.textContent = rows.length
    ? `${dateKeyLabel(rows[0].date)}-${dateKeyLabel(rows[rows.length - 1].date)} · ${formatBytes(total)} saved`
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
  ctx.fillText("captured", pad.left + 22, 13);

  rows.forEach((row, index) => {
    const barX = pad.left + index * slot + (slot - barWidth) / 2;
    const fullHeight = row.total ? Math.max(2, yScale(row.total)) : 0;
    let y = baseline;
    let cumulativeBytes = 0;
    ctx.fillStyle = "rgba(247,248,248,0.055)";
    roundRect(ctx, barX, baseline - Math.max(2, fullHeight), barWidth, Math.max(2, fullHeight), Math.min(5, barWidth / 2));
    ctx.fill();
    row.segments.forEach((segment) => {
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
        segmentColor(segment.bytes, maxSegmentBytes)
      );
      saveLoadHitRegions.push({
        x: barX - 2,
        y: Math.min(segmentTop, y) - 2,
        width: barWidth + 4,
        height: Math.max(4, Math.abs(segmentHeight) + 4),
        date: row.date,
        name: segment.name,
        bytes: segment.bytes,
        source: segment.source,
      });
    });


    const showLabel = index === 0 || index === rows.length - 1 || index % 5 === 4;
    if (showLabel) {
      const x = barX + barWidth / 2;
      ctx.fillStyle = "rgba(138,143,152,0.86)";
      ctx.font = "10px 'Geist Mono', ui-monospace, monospace";
      ctx.textAlign = "center";
      ctx.textBaseline = "alphabetic";
      ctx.fillText(dateKeyLabel(row.date), x, height - 14);
    }
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
    const label = ["info", "warning", "error"].includes(event.level) ? event.level : "info";
    const level = label === "error" ? "failure" : label;
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
  return [
    day.date,
    metric,
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
    intValue(day.raw_saved)
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
  ].forEach(([label, value]) => {
    const item = document.createElement("span");
    item.textContent = `${label} ${value}`;
    stats.appendChild(item);
  });

  const sources = document.createElement("p");
  sources.textContent = `sources: ${formatSources(day.sources)}`;
  const samples = document.createElement("ul");
  (day.raw_samples || []).slice(0, 3).forEach((sample) => {
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
    els.saveFeed.innerHTML = "<div class=\"empty-state\">No save records yet.</div>";
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
      body.textContent = `${intValue(day.raw_saved)} saved · ${formatBytes(day.raw_bytes)}`;
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
  );
  const days = firstActive > 0 ? allDays.slice(firstActive) : allDays;
  const totals = data.totals || {};

  els.saveTotal.textContent = intValue(totals.raw_saved).toLocaleString();
  els.saveModeButtons.forEach((button) => {
    button.classList.toggle("active", button.dataset.saveMode === saveHistoryMode);
    button.setAttribute("aria-pressed", String(button.dataset.saveMode === saveHistoryMode));
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
    els.knowledgeBars.innerHTML = "<div class=\"empty-state\">No pages indexed yet.</div>";
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

function modelStatusRank(row) {
  const order = { loaded: 0, unknown: 1, disabled: 2 };
  return order[row.status] ?? 9;
}

function modelRoleLabel(role) {
  return ({ embed: "Semantic embedding", rerank: "Search rerank", configured: "Configured" })[role] || fmt(role);
}

function renderModelStatus(modelStatus) {
  const data = modelStatus || {};
  const summary = data.summary || {};
  const models = Array.isArray(data.models) ? [...data.models] : [];
  els.modelServices.textContent = String(models.length);
  els.modelLoaded.textContent = String(intValue(summary.loaded));
  els.modelConfigured.textContent = String(intValue(summary.configured));
  const enabled = models.filter((row) => row.enabled !== false).length;
  els.modelUnavailable.textContent = String(enabled - intValue(summary.loaded));
  els.modelCaption.textContent = `${intValue(summary.loaded)} / ${enabled} enabled services ready`;

  els.modelGrid.innerHTML = "";
  if (!models.length) {
    els.modelGrid.innerHTML = "<div class=\"empty-state\">No model records yet.</div>";
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
      name.textContent = fmt(row.display_name || row.name);
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
      if (row.display_name && row.display_name !== row.name) metaPieces.push(`API ${row.name}`);
      if (row.size_bytes) metaPieces.push(`disk ${formatBytes(row.size_bytes)}`);
      else if (row.installed && row.size_bytes === null) metaPieces.push("disk unknown");
      if (row.loaded_size_bytes) metaPieces.push(`loaded ${formatBytes(row.loaded_size_bytes)}`);
      else if (row.running && row.loaded_size_bytes === null) metaPieces.push("loaded size unknown");
      if (row.context_length !== null && row.context_length !== undefined && Number(row.context_length) > 0) {
        metaPieces.push(`${Number(row.context_length).toLocaleString()} ctx`);
      }
      if (details.parameter_size) metaPieces.push(details.parameter_size);
      if (details.quantization_level) metaPieces.push(details.quantization_level);
      if (details.format) metaPieces.push(details.format);
      else if (row.provider) metaPieces.push(row.provider);
      if (row.expires_at) metaPieces.push(`until ${timeLabel(row.expires_at)}`);
      const meta = document.createElement("div");
      meta.className = "model-meta";
      meta.textContent = row.enabled === false ? "Disabled in configuration" : metaPieces.join(" · ") || `${row.provider || "local"} configured model`;
      meta.title = meta.textContent;

      item.append(main, roleWrap, meta);
      els.modelGrid.appendChild(item);
    });
}

function renderHealth(health) {
  const capture = health.capture || {};
  const queue = health.queue || {};
  const last = capture.last_run;
  els.healthCapture.textContent = last ? timeLabel(last.timestamp) : "No runs";
  els.healthExit.textContent = last ? String(last.exit_code) : "--";
  els.healthQueued.textContent = intValue(queue.queued);
  els.healthRunning.textContent = intValue(queue.running);
  els.healthRetry.textContent = intValue(queue.retry_wait);
  els.healthFailed.textContent = intValue(queue.failed) + intValue(queue.quarantined);
  els.healthCaption.textContent = intValue(queue.failed) + intValue(queue.quarantined) ? "attention" : "capture queue";
}

function render(snapshot) {
  const status = snapshot.status || {};
  const capture = status.capture || {};
  const services = status.services || {};
  const ready = Object.values(services).filter((service) => service.ready).length;
  setState(status.state);
  els.saves.textContent = fmt(status.saves_today);
  els.queue.textContent = fmt(capture.queued);
  els.queueSub.textContent = `${intValue(capture.by_status?.running)} running`;
  els.capture.textContent = capture.last_run ? timeLabel(capture.last_run.timestamp) : "--";
  els.captureSub.textContent = capture.last_run ? `${capture.last_run.timestamp.slice(0, 10)} · exit ${capture.last_run.exit_code}` : "No capture runs";
  els.pages.textContent = fmt(status.page_count);
  els.pagesSub.textContent = "Frozen knowledge pages";
  els.services.textContent = `${ready} / ${Object.values(services).filter((service) => service.enabled !== false).length}`;
  els.servicesSub.textContent = Object.entries(services).map(([name, service]) => `${name}: ${service.enabled === false ? "disabled" : service.ready ? "ready" : "unavailable"}`).join(" · ");
  renderSaveHistory(snapshot.save_history);
  renderKnowledgeMix(snapshot.knowledge_mix);
  renderHealth(snapshot.health || {});
  renderModelStatus(snapshot.model_status);
  renderEvents(snapshot.events || []);
  drawLineChart(els.pendingChart, snapshot.save_history);
}
