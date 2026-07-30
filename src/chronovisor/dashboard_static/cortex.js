"use strict";

(() => {
  const STEEL = "#7d92b5";
  const FIRE = "#ffb454";
  const FIRE_HOT = "#fff3dd";
  const ELECTRIC = "#ffd84d";
  const VIOLET = "#9b7cff";
  const COMMIT = "#45d49b";
  const INHIBIT = "#54b9ff";
  const FAULT = "#ff5d68";
  const RGB_STEEL = hexRgb(STEEL);
  const RGB_FIRE = hexRgb(FIRE);
  const RGB_HOT = hexRgb(FIRE_HOT);
  const RGB_ELECTRIC = hexRgb(ELECTRIC);
  const RGB_VIOLET = hexRgb(VIOLET);
  const RGB_COMMIT = hexRgb(COMMIT);
  const RGB_INHIBIT = hexRgb(INHIBIT);
  const RGB_FAULT = hexRgb(FAULT);
  const TYPE_OFF = new Set([2]);
  const ACTIVE_LABEL_LIMIT = 5;
  const NODE_FLASH_ATTACK_MS = 90;
  const NODE_FLASH_HOLD_MS = 150;
  const NODE_FLASH_DECAY_MS = 1450;
  const NODE_FLASH_DURATION_MS =
    NODE_FLASH_ATTACK_MS + NODE_FLASH_HOLD_MS + NODE_FLASH_DECAY_MS;
  const EDGE_AFTERGLOW_MS = 650;

  let data;
  let nodes = [];
  let links = [];
  let nodeCount = 0;
  let neighbors = [];
  let outgoing = [];
  let byId = new Map();
  let packageList = [];
  let packageShade = {};
  let anchors = {};
  let nodeState;
  let edgeState;
  let drawOrder = [];
  let labelHubs = new Set();

  let selected = -1;
  let hovered = -1;
  let query = "";
  let matches = new Set();
  let stateDirty = true;
  const packageOff = new Set();
  let mode = "organic";
  let liveEventsEnabled = true;
  let motionEnabled = true;
  let autoRotate = true;
  let soundOn = false;
  let edgeVisibility = 1.6;
  let alpha = 1;
  let spikes = 0;
  let lastInteraction = 0;
  let lastVisualMetricsPublished = 0;

  const stage = document.getElementById("stage");
  const canvas = document.getElementById("gl");
  const context = canvas.getContext("2d");
  let width = 0;
  let height = 0;
  const pixelRatio = Math.min(2, window.devicePixelRatio || 1);
  const camera = { theta: 0.6, phi: 0.18, distance: 1650 };
  let cameraTarget = null;
  let focalLength = 900;
  let dragging = false;
  let downPoint = null;
  let moved = false;
  let audioContext = null;
  let lastSound = 0;
  let eventSocket = null;
  let eventReconnect = null;
  let eventSocketGeneration = 0;
  let sessionRequest = 0;
  const reducedMotion = window.matchMedia("(prefers-reduced-motion: reduce)");
  const fieldState = window.CortexField.createState();
  const cortexMetrics = {
    spread: [],
    frameDurations: [],
    maxPulseQueue: 0,
    violetNodes: 0,
    labelsPainted: 0,
    afterglowEdges: 0,
    flashPeak: 0,
  };
  window.chronovisorCortexMetrics = () => ({
    spread: cortexMetrics.spread.map((row) => ({ ...row })),
    frameDurations: cortexMetrics.frameDurations.slice(-240),
    maxPulseQueue: cortexMetrics.maxPulseQueue,
    pulseQueue: pulses.length,
    visual: {
      violetNodes: cortexMetrics.violetNodes,
      labelsPainted: cortexMetrics.labelsPainted,
      afterglowEdges: cortexMetrics.afterglowEdges,
      flashPeak: cortexMetrics.flashPeak,
      activeLabelLimit: ACTIVE_LABEL_LIMIT,
      attackMs: NODE_FLASH_ATTACK_MS,
      holdMs: NODE_FLASH_HOLD_MS,
      decayMs: NODE_FLASH_DECAY_MS,
      edgeAfterglowMs: EDGE_AFTERGLOW_MS,
    },
  });

  const pulses = [];
  const edgeAfterglows = [];
  const nodeEffects = [];
  const stars = Array.from({ length: 130 }, () => ({
    x: Math.random(),
    y: Math.random(),
    radius: Math.random() * 0.9 + 0.3,
    phase: Math.random() * Math.PI * 2,
    speed: 0.3 + Math.random() * 1.1,
  }));
  const glowFire = makeGlow(RGB_FIRE);
  const glowSteel = makeGlow(RGB_STEEL);
  const glowHot = makeGlow(RGB_HOT);
  const glowElectric = makeGlow(RGB_ELECTRIC);
  const glowViolet = makeGlow(RGB_VIOLET);
  const glowCommit = makeGlow(RGB_COMMIT);
  const glowInhibit = makeGlow(RGB_INHIBIT);
  const glowFault = makeGlow(RGB_FAULT);

  function hexRgb(hex) {
    return [
      Number.parseInt(hex.slice(1, 3), 16),
      Number.parseInt(hex.slice(3, 5), 16),
      Number.parseInt(hex.slice(5, 7), 16),
    ];
  }

  function rgba(color, opacity) {
    return `rgba(${color[0] | 0},${color[1] | 0},${color[2] | 0},${opacity})`;
  }

  function shade(color, factor) {
    return color.map((channel) => Math.min(255, channel * factor));
  }

  function mix(from, to, factor) {
    return from.map((channel, index) => channel + (to[index] - channel) * factor);
  }

  function clamp(value, minimum = 0, maximum = 1) {
    return Math.max(minimum, Math.min(maximum, value));
  }

  function smoothstep(value) {
    const unit = clamp(value);
    return unit * unit * (3 - 2 * unit);
  }

  function excitationLevel(node, time) {
    const age = time - node.flashStartedAt;
    if (!Number.isFinite(age) || age < 0 || age >= NODE_FLASH_DURATION_MS) {
      return 0;
    }
    if (age < NODE_FLASH_ATTACK_MS) {
      const attack = 1 - Math.pow(1 - age / NODE_FLASH_ATTACK_MS, 3);
      return node.flashBase + (node.flashPeak - node.flashBase) * attack;
    }
    if (age < NODE_FLASH_ATTACK_MS + NODE_FLASH_HOLD_MS) {
      return node.flashPeak;
    }
    const decay =
      (age - NODE_FLASH_ATTACK_MS - NODE_FLASH_HOLD_MS) / NODE_FLASH_DECAY_MS;
    return node.flashPeak * (1 - smoothstep(decay));
  }

  function exciteNode(node, delta, time) {
    const current = excitationLevel(node, time);
    const strength = clamp(Math.abs(delta));
    node.flashBase = current;
    node.flashPeak = clamp(
      current + strength * (1 - current * 0.35),
      current,
      1.25,
    );
    node.flashStartedAt = time;
    node.firedAt = time;
  }

  function escapeHtml(value) {
    return String(value ?? "")
      .replaceAll("&", "&amp;")
      .replaceAll("<", "&lt;")
      .replaceAll(">", "&gt;")
      .replaceAll('"', "&quot;")
      .replaceAll("'", "&#39;");
  }

  function deterministicUnit(value, salt) {
    let hash = 2166136261 ^ salt;
    const text = String(value);
    for (let index = 0; index < text.length; index += 1) {
      hash ^= text.charCodeAt(index);
      hash = Math.imul(hash, 16777619);
    }
    return (hash >>> 0) / 4294967295;
  }

  function makeGlow(color) {
    const sprite = document.createElement("canvas");
    sprite.width = 128;
    sprite.height = 128;
    const spriteContext = sprite.getContext("2d");
    const gradient = spriteContext.createRadialGradient(64, 64, 0, 64, 64, 64);
    gradient.addColorStop(0, rgba(color, 0.9));
    gradient.addColorStop(0.28, rgba(color, 0.28));
    gradient.addColorStop(1, rgba(color, 0));
    spriteContext.fillStyle = gradient;
    spriteContext.fillRect(0, 0, 128, 128);
    return sprite;
  }

  function resize() {
    width = stage.clientWidth;
    height = stage.clientHeight;
    canvas.width = Math.max(1, Math.round(width * pixelRatio));
    canvas.height = Math.max(1, Math.round(height * pixelRatio));
  }

  function initializeGraph(graphData) {
    data = graphData;
    packageList = (data.categories || []).map((category) => category.id);
    if (!packageList.length) {
      packageList = [...new Set(data.nodes.map((node) => node.pkg))];
    }
    packageShade = Object.fromEntries(
      packageList.map((packageName, index) => [
        packageName,
        0.72 + (index % 9) * 0.055,
      ]),
    );

    nodes = data.nodes.map((row, index) => ({
      index,
      id: row.id,
      packageName: row.pkg,
      lineCount: row.l || 1,
      byteCount: row.b || 0,
      fanIn: row.fi || 0,
      fanOut: row.fo || 0,
      entrypoint: row.ep || 0,
      title: row.title || row.id,
      updated: row.updated || "",
      tags: Array.isArray(row.tags) ? row.tags : [],
      name: row.id,
      radius: Math.max(
        2.3,
        Math.min(12, 2 + Math.sqrt(row.l || 1) * 0.12 + (row.fi || 0) * 0.025),
      ),
      base: shade(RGB_STEEL, packageShade[row.pkg] || 1),
      x: 0,
      y: 0,
      z: 0,
      vx: 0,
      vy: 0,
      vz: 0,
      firedAt: -1e9,
      flashStartedAt: -1e9,
      flashBase: 0,
      flashPeak: 0,
      fieldActivation: 0,
      fieldComponents: { direct: 0, spread: 0, negative: 0, inhibition: 0 },
      fieldState: "inactive",
      certificateId: "",
      reasonCode: "",
      arrivedAt: -1e9,
      screenX: 0,
      screenY: 0,
      screenScale: 0,
      viewDepth: 1e9,
    }));
    links = data.links.map((row) => ({
      source: row[0],
      target: row[1],
      kind: row[2] || 0,
      edgeType: row[3] || "wikilink",
      eventOnly: false,
    }));
    nodeCount = nodes.length;
    neighbors = Array.from({ length: nodeCount }, () => new Set());
    outgoing = Array.from({ length: nodeCount }, () => []);
    links.forEach((link, edgeIndex) => {
      if (!nodes[link.source] || !nodes[link.target]) return;
      neighbors[link.source].add(link.target);
      neighbors[link.target].add(link.source);
      if (link.kind < 2) outgoing[link.source].push(edgeIndex);
    });
    byId = new Map(nodes.map((node) => [node.id, node]));
    nodeState = new Uint8Array(nodeCount);
    edgeState = new Uint8Array(links.length);
    drawOrder = nodes.map((_node, index) => index);
    labelHubs = new Set(
      [...nodes]
        .sort((left, right) => right.fanIn - left.fanIn)
        .slice(0, 8)
        .map((node) => node.index),
    );
    buildAnchors();
    seedPositions();
  }

  function buildAnchors() {
    anchors = {};
    packageList.forEach((packageName, index) => {
      const offset = index + 0.5;
      const phi = Math.acos(1 - (2 * offset) / Math.max(1, packageList.length));
      const theta = Math.PI * (1 + Math.sqrt(5)) * offset;
      anchors[packageName] = {
        x: Math.sin(phi) * Math.cos(theta) * 430,
        y: Math.cos(phi) * 330,
        z: Math.sin(phi) * Math.sin(theta) * 430,
      };
    });
  }

  function seedPositions() {
    nodes.forEach((node) => {
      const anchor = anchors[node.packageName] || { x: 0, y: 0, z: 0 };
      node.x = anchor.x * 0.42 + (deterministicUnit(node.id, 11) - 0.5) * 330;
      node.y = anchor.y * 0.42 + (deterministicUnit(node.id, 29) - 0.5) * 280;
      node.z = anchor.z * 0.42 + (deterministicUnit(node.id, 47) - 0.5) * 330;
    });
  }

  function tick() {
    const centerForce = alpha * 0.00055;
    const anchorForce = alpha * (mode === "cluster" ? 0.015 : 0.0007);
    nodes.forEach((node) => {
      const anchor = anchors[node.packageName] || { x: 0, y: 0, z: 0 };
      node.vx -= node.x * centerForce * 0.8;
      node.vy -= node.y * centerForce * 1.25;
      node.vz -= node.z * centerForce;
      node.vx += (anchor.x - node.x) * anchorForce;
      node.vy += (anchor.y - node.y) * anchorForce;
      node.vz += (anchor.z - node.z) * anchorForce;
    });

    const spring = mode === "cluster" ? 0.0012 : 0.0007;
    links.forEach((link) => {
      if (link.kind === 2) return;
      const source = nodes[link.source];
      const target = nodes[link.target];
      if (!source || !target) return;
      const dx = target.x - source.x;
      const dy = target.y - source.y;
      const dz = target.z - source.z;
      const distance = Math.hypot(dx, dy, dz) || 1;
      const rest = 58 + source.radius + target.radius;
      const force = (distance - rest) * spring * alpha;
      const fx = (dx / distance) * force;
      const fy = (dy / distance) * force;
      const fz = (dz / distance) * force;
      source.vx += fx;
      source.vy += fy;
      source.vz += fz;
      target.vx -= fx;
      target.vy -= fy;
      target.vz -= fz;
    });

    nodes.forEach((node) => {
      node.vx *= 0.82;
      node.vy *= 0.82;
      node.vz *= 0.82;
      const velocity = Math.hypot(node.vx, node.vy, node.vz);
      if (velocity > 13) {
        const limit = 13 / velocity;
        node.vx *= limit;
        node.vy *= limit;
        node.vz *= limit;
      }
      node.x += node.vx;
      node.y += node.vy;
      node.z += node.vz;
    });
    alpha = Math.max(0.012, alpha * 0.986);
  }

  function reheat(value) {
    alpha = Math.max(alpha, value);
  }

  function projectAll() {
    focalLength = Math.min(width, height) * 1.12;
    const cosTheta = Math.cos(camera.theta);
    const sinTheta = Math.sin(camera.theta);
    const cosPhi = Math.cos(camera.phi);
    const sinPhi = Math.sin(camera.phi);
    nodes.forEach((node) => {
      const rotatedX = node.x * cosTheta + node.z * sinTheta;
      const rotatedZ = -node.x * sinTheta + node.z * cosTheta;
      const rotatedY = node.y * cosPhi - rotatedZ * sinPhi;
      const depthZ = node.y * sinPhi + rotatedZ * cosPhi;
      const viewDepth = camera.distance - depthZ;
      if (viewDepth < 60) {
        node.viewDepth = 1e9;
        return;
      }
      const scale = focalLength / viewDepth;
      node.screenX = rotatedX * scale + width / 2;
      node.screenY = rotatedY * scale + height / 2;
      node.screenScale = scale;
      node.viewDepth = viewDepth;
    });
  }

  function fog(viewDepth) {
    const amount = (viewDepth - camera.distance * 0.55) / (camera.distance * 1.1);
    return 1 - Math.max(0, Math.min(0.72, amount));
  }

  function graphRadius() {
    let maximum = 0;
    nodes.forEach((node) => {
      maximum = Math.max(maximum, node.x ** 2 + node.y ** 2 + node.z ** 2);
    });
    return Math.sqrt(maximum);
  }

  function fitView() {
    cameraTarget = {
      theta: camera.theta,
      phi: camera.phi,
      distance: Math.max(600, graphRadius() * 2.55),
    };
  }

  function angleLerp(from, to, factor) {
    const delta = ((to - from + Math.PI * 3) % (Math.PI * 2)) - Math.PI;
    return from + delta * factor;
  }

  function focusNode(index) {
    const node = nodes[index];
    if (!node) return;
    const theta = Math.atan2(node.x, node.z);
    const radial = Math.hypot(node.x, node.z);
    const phi = Math.max(-1.3, Math.min(1.3, Math.atan2(node.y, radial)));
    cameraTarget = {
      theta,
      phi,
      distance: Math.max(700, Math.min(camera.distance, 1400)),
    };
  }

  function recomputeState() {
    const focus = selected >= 0 ? selected : hovered;
    const ego = focus >= 0 ? neighbors[focus] : null;
    nodes.forEach((node, index) => {
      if (packageOff.has(node.packageName)) {
        nodeState[index] = 0;
      } else if (query) {
        nodeState[index] = matches.has(index) ? 3 : 1;
      } else if (focus >= 0) {
        nodeState[index] = index === focus ? 3 : ego.has(index) ? 2 : 1;
      } else {
        nodeState[index] = 2;
      }
    });
    links.forEach((link, edgeIndex) => {
      if (
        TYPE_OFF.has(link.kind)
        || nodeState[link.source] === 0
        || nodeState[link.target] === 0
      ) {
        edgeState[edgeIndex] = 0;
      } else if (focus >= 0 && !query) {
        edgeState[edgeIndex] =
          link.source === focus || link.target === focus ? 3 : 1;
      } else if (query) {
        edgeState[edgeIndex] =
          matches.has(link.source) || matches.has(link.target) ? 2 : 1;
      } else {
        edgeState[edgeIndex] = 2;
      }
    });
    stateDirty = false;
  }

  function crackle(volume) {
    if (!soundOn || !audioContext) return;
    const now = performance.now();
    if (now - lastSound < 22) return;
    lastSound = now;
    const duration = 0.03 + Math.random() * 0.04;
    const buffer = audioContext.createBuffer(
      1,
      Math.max(64, (audioContext.sampleRate * duration) | 0),
      audioContext.sampleRate,
    );
    const samples = buffer.getChannelData(0);
    for (let index = 0; index < samples.length; index += 1) {
      samples[index] =
        (Math.random() * 2 - 1) * Math.pow(1 - index / samples.length, 2.2);
    }
    const source = audioContext.createBufferSource();
    source.buffer = buffer;
    const filter = audioContext.createBiquadFilter();
    filter.type = "bandpass";
    filter.frequency.value = 1100 + Math.random() * 2800;
    filter.Q.value = 1.4;
    const gain = audioContext.createGain();
    gain.gain.value = volume;
    source.connect(filter);
    filter.connect(gain);
    gain.connect(audioContext.destination);
    source.start();
  }

  function visibleHub() {
    return nodes.reduce(
      (best, node) =>
        nodeState[node.index] > 0 && (!best || node.fanOut > best.fanOut)
          ? node
          : best,
      null,
    );
  }

  function scenarioNodes(name) {
    const preferred = {
      recall: ["current-state", "user-profile", "lessons-learned"],
      save: ["current-state"],
      ingest: ["current-state"],
    };
    const found = (preferred[name] || []).map((id) => byId.get(id)).filter(Boolean);
    if (found.length) return found;
    if (name === "save") {
      const rawNodes = nodes.filter((node) => node.packageName === "raw");
      if (rawNodes.length) return rawNodes.slice(0, 2);
    }
    return [visibleHub()].filter(Boolean);
  }

  function stimulate(name, label = "") {
    const roots = scenarioNodes(name);
    const now = performance.now();
    roots.forEach((node, rootIndex) => {
      const startedAt = now + rootIndex * 35;
      exciteNode(node, 0.9, startedAt);
      nodeEffects.push({
        nodeIndex: node.index,
        kind: "stimulus",
        startedAt,
        duration: 900,
        delta: 0.9,
        seq: -(rootIndex + 1),
        demo: true,
      });
      const candidates = outgoing[node.index]
        .filter((edgeIndex) => edgeState[edgeIndex] > 0)
        .sort((left, right) => left - right)
        .slice(0, 3);
      candidates.forEach((edgeIndex, branch) => {
        pulses.push({
          edgeIndex,
          startedAt: now + 120 + branch * 80,
          duration: 150 + branch * 25,
          delta: 0.78 - branch * 0.12,
          seq: -(rootIndex * 10 + branch + 1),
          edgeType: links[edgeIndex].edgeType,
          demo: true,
          paintedAt: 0,
        });
      });
    });
    trimVisualQueues();
    cortexMetrics.maxPulseQueue = Math.max(
      cortexMetrics.maxPulseQueue,
      pulses.length,
    );
    publishCortexMetrics();
    document.getElementById("stSp").textContent = spikes.toLocaleString();
    flashTicker(label || `DEMO/REPLAY · ${name.toUpperCase()} · backend unchanged`);
  }

  function trimVisualQueues() {
    if (pulses.length > window.CortexField.MAX_EVENTS) {
      pulses.splice(0, pulses.length - window.CortexField.MAX_EVENTS);
    }
    if (nodeEffects.length > window.CortexField.MAX_EVENTS) {
      nodeEffects.splice(0, nodeEffects.length - window.CortexField.MAX_EVENTS);
    }
    if (edgeAfterglows.length > window.CortexField.MAX_EVENTS) {
      edgeAfterglows.splice(
        0,
        edgeAfterglows.length - window.CortexField.MAX_EVENTS,
      );
    }
  }

  function publishCortexMetrics() {
    const target = document.getElementById("fieldAria");
    if (!target) return;
    const painted = cortexMetrics.spread.filter(
      (row) => Number.isFinite(row.paintedAt),
    );
    const latencies = painted
      .map((row) => row.paintedAt - row.receivedAt)
      .sort((left, right) => left - right);
    const p95 = latencies.length
      ? latencies[Math.min(latencies.length - 1, Math.round((latencies.length - 1) * 0.95))]
      : 0;
    const frames = cortexMetrics.frameDurations
      .filter((value) => value > 0)
      .slice(-120)
      .sort((left, right) => left - right);
    const frameP95 = frames.length
      ? frames[Math.min(frames.length - 1, Math.round((frames.length - 1) * 0.95))]
      : 0;
    const frameMean = frames.length
      ? frames.reduce((total, value) => total + value, 0) / frames.length
      : 0;
    target.dataset.spreadReceived = String(cortexMetrics.spread.length);
    target.dataset.spreadPainted = String(painted.length);
    target.dataset.paintP95Ms = p95.toFixed(1);
    target.dataset.maxPulseQueue = String(cortexMetrics.maxPulseQueue);
    target.dataset.frameP95Ms = frameP95.toFixed(1);
    target.dataset.fps = frameMean ? (1000 / frameMean).toFixed(1) : "0.0";
  }

  function publishVisualMetrics(time) {
    if (time - lastVisualMetricsPublished < 200) return;
    lastVisualMetricsPublished = time;
    const target = document.getElementById("fieldAria");
    if (!target) return;
    target.dataset.violetNodes = String(cortexMetrics.violetNodes);
    target.dataset.labelsPainted = String(cortexMetrics.labelsPainted);
    target.dataset.afterglowEdges = String(cortexMetrics.afterglowEdges);
    target.dataset.flashPeak = cortexMetrics.flashPeak.toFixed(3);
    target.dataset.activeLabelLimit = String(ACTIVE_LABEL_LIMIT);
    target.dataset.flashTiming =
      `${NODE_FLASH_ATTACK_MS}/${NODE_FLASH_HOLD_MS}/${NODE_FLASH_DECAY_MS}`;
    target.dataset.edgeAfterglowMs = String(EDGE_AFTERGLOW_MS);
  }

  function ensureActualEdge(event) {
    const source = byId.get(event.source_page_id);
    const target = byId.get(event.target_page_id);
    if (!source || !target) return -1;
    let edgeIndex = links.findIndex(
      (link) => link.source === source.index && link.target === target.index,
    );
    if (edgeIndex >= 0) return edgeIndex;
    edgeIndex = links.length;
    links.push({
      source: source.index,
      target: target.index,
      kind: 1,
      edgeType: event.edge_type || "field",
      eventOnly: true,
    });
    neighbors[source.index].add(target.index);
    neighbors[target.index].add(source.index);
    outgoing[source.index].push(edgeIndex);
    const expanded = new Uint8Array(links.length);
    expanded.set(edgeState);
    expanded[edgeIndex] = 2;
    edgeState = expanded;
    stateDirty = true;
    return edgeIndex;
  }

  function syncFieldNodes() {
    nodes.forEach((node) => {
      const field = fieldState.nodes.get(node.id);
      node.fieldActivation = field?.activation || 0;
      node.fieldComponents = field?.components || {
        direct: 0,
        spread: 0,
        negative: 0,
        inhibition: 0,
      };
      node.fieldState = field?.state || "inactive";
      node.certificateId = field?.certificateId || "";
      node.reasonCode = field?.reasonCode || "";
    });
  }

  function effectNode(event) {
    const pageId = window.CortexField.eventPageId(event);
    return byId.get(pageId);
  }

  function visualizeFieldEvent(event) {
    const now = performance.now();
    const node = effectNode(event);
    if (event.kind === "spread") {
      const edgeIndex = ensureActualEdge(event);
      if (edgeIndex < 0) {
        flashTicker(`◇ seq ${event.seq} · unmapped ${event.source_page_id}→${event.target_page_id}`);
        return;
      }
      const strength = Math.max(0, Math.min(1, Math.abs(event.delta)));
      pulses.push({
        edgeIndex,
        startedAt: now,
        duration: 250 - strength * 130,
        delta: strength,
        seq: event.seq,
        edgeType: event.edge_type || "field",
        demo: false,
        paintedAt: 0,
      });
      cortexMetrics.spread.push({
        seq: event.seq,
        source: event.source_page_id,
        target: event.target_page_id,
        edge: event.edge_type || "field",
        delta: strength,
        receivedAt: now,
        paintedAt: null,
        arrivalAt: null,
      });
      if (cortexMetrics.spread.length > window.CortexField.MAX_EVENTS) {
        cortexMetrics.spread.shift();
      }
      cortexMetrics.maxPulseQueue = Math.max(
        cortexMetrics.maxPulseQueue,
        pulses.length,
      );
      publishCortexMetrics();
      spikes += 1;
      crackle(0.035 + strength * 0.08);
    } else if (node) {
      if (event.kind === "stimulus") {
        exciteNode(node, event.delta, now);
      }
      nodeEffects.push({
        nodeIndex: node.index,
        kind: event.kind,
        startedAt: now,
        duration: event.kind === "stimulus" ? 900 : 820,
        delta: Math.abs(event.delta),
        seq: event.seq,
        reasonCode: event.reason_code,
        demo: false,
      });
      spikes += 1;
    } else if (event.kind === "fault") {
      fieldState.fault = event.reason_code || "field fault";
    }
    trimVisualQueues();
    document.getElementById("stSp").textContent = spikes.toLocaleString();
    const route = event.kind === "spread"
      ? `${event.source_page_id} → ${event.target_page_id}`
      : window.CortexField.eventPageId(event) || "field";
    flashTicker(`${event.kind.toUpperCase()} · seq ${event.seq} · ${route}`);
    document.getElementById("fieldAria").textContent =
      `${event.kind}, sequence ${event.seq}, ${route}, delta ${event.delta.toFixed(3)}`;
  }

  function drawEdges() {
    const paths = Array.from({ length: 9 }, () => new Path2D());
    const pathKinds = new Uint8Array(9);
    links.forEach((link, edgeIndex) => {
      const state = edgeState[edgeIndex];
      if (!state) return;
      const source = nodes[link.source];
      const target = nodes[link.target];
      if (
        !source
        || !target
        || source.viewDepth > 9e8
        || target.viewDepth > 9e8
      ) {
        return;
      }
      if (
        (source.screenX < -40 && target.screenX < -40)
        || (source.screenX > width + 40 && target.screenX > width + 40)
        || (source.screenY < -40 && target.screenY < -40)
        || (source.screenY > height + 40 && target.screenY > height + 40)
      ) {
        return;
      }
      const depth = (source.viewDepth + target.viewDepth) / 2;
      const band =
        depth < camera.distance * 0.85
          ? 0
          : depth < camera.distance * 1.15
            ? 1
            : 2;
      const batch = (state - 1) * 3 + band;
      paths[batch].moveTo(source.screenX, source.screenY);
      paths[batch].lineTo(target.screenX, target.screenY);
      pathKinds[batch] = 1;
    });

    const bandOpacity = [0.085, 0.05, 0.024];
    for (let state = 1; state <= 3; state += 1) {
      for (let band = 2; band >= 0; band -= 1) {
        const batch = (state - 1) * 3 + band;
        if (!pathKinds[batch]) continue;
        context.lineWidth =
          state === 3 ? 1.3 : 0.7 + Math.max(0, edgeVisibility - 1) * 0.28;
        let opacity =
          state === 3
            ? 0.42
            : state === 2
              ? bandOpacity[band] * edgeVisibility
              : bandOpacity[band] * 0.35 * edgeVisibility;
        opacity = Math.min(0.85, opacity);
        context.strokeStyle =
          state === 3 ? rgba(RGB_FIRE, opacity) : rgba(RGB_STEEL, opacity);
        context.stroke(paths[batch]);
      }
    }
  }

  function projectPoint(x, y, z) {
    const cosTheta = Math.cos(camera.theta);
    const sinTheta = Math.sin(camera.theta);
    const cosPhi = Math.cos(camera.phi);
    const sinPhi = Math.sin(camera.phi);
    const rotatedX = x * cosTheta + z * sinTheta;
    const rotatedZ = -x * sinTheta + z * cosTheta;
    const rotatedY = y * cosPhi - rotatedZ * sinPhi;
    const depthZ = y * sinPhi + rotatedZ * cosPhi;
    const viewDepth = camera.distance - depthZ;
    if (viewDepth < 60) return null;
    const scale = focalLength / viewDepth;
    return {
      x: rotatedX * scale + width / 2,
      y: rotatedY * scale + height / 2,
      scale,
      viewDepth,
    };
  }

  function completePulse(pulse, target, time) {
    target.arrivedAt = time;
    exciteNode(target, pulse.delta, time);
    nodeEffects.push({
      nodeIndex: target.index,
      kind: "arrival",
      startedAt: time,
      duration: 900,
      delta: pulse.delta,
      seq: pulse.seq,
      demo: pulse.demo,
    });
    edgeAfterglows.push({
      edgeIndex: pulse.edgeIndex,
      startedAt: time,
      duration: EDGE_AFTERGLOW_MS,
      delta: pulse.delta,
      seq: pulse.seq,
    });
    trimVisualQueues();
    const metric = cortexMetrics.spread.find((row) => row.seq === pulse.seq);
    if (metric) metric.arrivalAt = time;
  }

  function drawEdgeAfterglows(time) {
    context.globalCompositeOperation = "lighter";
    for (let index = edgeAfterglows.length - 1; index >= 0; index -= 1) {
      const afterglow = edgeAfterglows[index];
      const progress = (time - afterglow.startedAt) / afterglow.duration;
      const link = links[afterglow.edgeIndex];
      if (!link || progress >= 1) {
        edgeAfterglows.splice(index, 1);
        continue;
      }
      if (progress < 0) continue;
      const source = nodes[link.source];
      const target = nodes[link.target];
      if (
        !source
        || !target
        || source.viewDepth > 9e8
        || target.viewDepth > 9e8
      ) {
        continue;
      }
      const depthFade = fog((source.viewDepth + target.viewDepth) / 2);
      const fade = (1 - smoothstep(progress)) * depthFade;
      context.strokeStyle = rgba(
        RGB_ELECTRIC,
        fade * (0.08 + afterglow.delta * 0.24),
      );
      context.lineWidth = 1.1 + afterglow.delta * 1.8;
      context.beginPath();
      context.moveTo(source.screenX, source.screenY);
      context.lineTo(target.screenX, target.screenY);
      context.stroke();
      const glowSize = 10 + afterglow.delta * 12;
      context.globalAlpha = fade * (0.18 + afterglow.delta * 0.32);
      context.drawImage(
        glowElectric,
        target.screenX - glowSize / 2,
        target.screenY - glowSize / 2,
        glowSize,
        glowSize,
      );
      context.globalAlpha = 1;
      cortexMetrics.afterglowEdges += 1;
    }
    context.globalCompositeOperation = "source-over";
  }

  function drawPulses(time) {
    context.globalCompositeOperation = "lighter";
    for (let index = pulses.length - 1; index >= 0; index -= 1) {
      const pulse = pulses[index];
      const link = links[pulse.edgeIndex];
      if (!link) {
        pulses.splice(index, 1);
        continue;
      }
      const source = nodes[link.source];
      const target = nodes[link.target];
      const staticMotion = reducedMotion.matches || !motionEnabled;
      const rawProgress = (time - pulse.startedAt) / pulse.duration;
      const progress = staticMotion ? 0.72 : rawProgress;
      if (rawProgress < 0) continue;
      if (progress >= 1) {
        pulses.splice(index, 1);
        completePulse(pulse, target, time);
        continue;
      }
      if (source.viewDepth > 9e8 || target.viewDepth > 9e8) continue;
      const edgeId = `${source.id}>${target.id}:${pulse.edgeType}`;
      const tailLength = 0.08 + deterministicUnit(edgeId, pulse.seq) * 0.07;
      const tailProgress = Math.max(0, progress - tailLength);
      const screenSource = { x: source.screenX, y: source.screenY };
      const screenTarget = { x: target.screenX, y: target.screenY };
      const dx = screenTarget.x - screenSource.x;
      const dy = screenTarget.y - screenSource.y;
      const length = Math.hypot(dx, dy) || 1;
      const normalX = -dy / length;
      const normalY = dx / length;
      const points = [];
      const pointCount = 5;
      for (let pointIndex = 0; pointIndex < pointCount; pointIndex += 1) {
        const unit = pointIndex / (pointCount - 1);
        const edgeProgress = tailProgress + (progress - tailProgress) * unit;
        const endpoint = pointIndex === 0 || pointIndex === pointCount - 1;
        const jitter = endpoint
          ? 0
          : (deterministicUnit(edgeId, pulse.seq * 31 + pointIndex) - 0.5)
            * (2.5 + pulse.delta * 5.5);
        points.push({
          x: screenSource.x + dx * edgeProgress + normalX * jitter,
          y: screenSource.y + dy * edgeProgress + normalY * jitter,
        });
      }
      const head = points[points.length - 1];
      const depthFade = fog((source.viewDepth + target.viewDepth) / 2);
      context.lineCap = "round";
      context.lineJoin = "round";
      context.beginPath();
      points.forEach((point, pointIndex) => {
        if (pointIndex) context.lineTo(point.x, point.y);
        else context.moveTo(point.x, point.y);
      });
      context.strokeStyle = rgba(
        RGB_ELECTRIC,
        (0.38 + pulse.delta * 0.5) * depthFade,
      );
      context.lineWidth = 2.1 + pulse.delta * 3.4;
      context.stroke();
      context.strokeStyle = rgba(
        RGB_HOT,
        (0.7 + pulse.delta * 0.3) * depthFade,
      );
      context.lineWidth = 0.7 + pulse.delta * 1.25;
      context.stroke();
      const branchLength = 4 + pulse.delta * 8;
      const branchSign = deterministicUnit(edgeId, pulse.seq + 991) > 0.5 ? 1 : -1;
      context.strokeStyle = rgba(RGB_ELECTRIC, 0.45 + pulse.delta * 0.35);
      context.lineWidth = 0.7 + pulse.delta * 0.5;
      context.beginPath();
      context.moveTo(head.x, head.y);
      context.lineTo(
        head.x - (dx / length) * branchLength + normalX * branchLength * branchSign,
        head.y - (dy / length) * branchLength + normalY * branchLength * branchSign,
      );
      context.stroke();
      if (staticMotion) {
        const arrowLength = 7 + pulse.delta * 5;
        context.strokeStyle = rgba(RGB_ELECTRIC, 0.75 + pulse.delta * 0.25);
        context.lineWidth = 1.2 + pulse.delta;
        context.beginPath();
        context.moveTo(head.x, head.y);
        context.lineTo(
          head.x - (dx / length) * arrowLength + normalX * arrowLength * 0.55,
          head.y - (dy / length) * arrowLength + normalY * arrowLength * 0.55,
        );
        context.moveTo(head.x, head.y);
        context.lineTo(
          head.x - (dx / length) * arrowLength - normalX * arrowLength * 0.55,
          head.y - (dy / length) * arrowLength - normalY * arrowLength * 0.55,
        );
        context.stroke();
      }
      const glowSize = 10 + 12 * pulse.delta;
      context.globalAlpha = 0.7 + pulse.delta * 0.3;
      context.drawImage(
        glowElectric,
        head.x - glowSize / 2,
        head.y - glowSize / 2,
        glowSize,
        glowSize,
      );
      context.globalAlpha = 1;
      if (!pulse.paintedAt) pulse.paintedAt = time;
      const metric = cortexMetrics.spread.find((row) => row.seq === pulse.seq);
      if (metric && metric.paintedAt === null) metric.paintedAt = time;
      publishCortexMetrics();
      if (staticMotion && time - pulse.startedAt >= 900) {
        pulses.splice(index, 1);
        completePulse(pulse, target, time);
      }
    }
  }

  function drawNodes(time) {
    drawOrder.sort(
      (leftIndex, rightIndex) =>
        nodes[rightIndex].viewDepth - nodes[leftIndex].viewDepth,
    );
    drawOrder.forEach((index) => {
      const node = nodes[index];
      const state = nodeState[index];
      if (
        state === 0
        || node.viewDepth > 9e8
        || node.screenX < -60
        || node.screenX > width + 60
        || node.screenY < -60
        || node.screenY > height + 60
      ) {
        return;
      }
      const depthFade = fog(node.viewDepth);
      const dim = state === 1 ? 0.28 : 1;
      const fieldActivation = Math.max(0, Math.min(1, node.fieldActivation));
      const isFieldActive = fieldActivation >= 0.05;
      const excitation = excitationLevel(node, time);
      const radius =
        Math.max(0.75, node.radius * node.screenScale)
        * (1 + (isFieldActive ? fieldActivation * 0.18 : 0) + excitation * 0.1);
      if (isFieldActive) {
        const haloSize = radius * (4.8 + fieldActivation * 8);
        context.globalAlpha = (0.12 + fieldActivation * 0.4) * depthFade * dim;
        context.drawImage(
          glowViolet,
          node.screenX - haloSize / 2,
          node.screenY - haloSize / 2,
          haloSize,
          haloSize,
        );
        cortexMetrics.violetNodes += 1;
      }
      if (excitation > 0.01) {
        const glowSize = radius * (5 + 12 * excitation);
        context.globalAlpha =
          Math.min(1, 0.12 + excitation * 0.88) * depthFade * dim;
        context.drawImage(
          excitation > 0.68 ? glowHot : glowViolet,
          node.screenX - glowSize / 2,
          node.screenY - glowSize / 2,
          glowSize,
          glowSize,
        );
        cortexMetrics.flashPeak = Math.max(
          cortexMetrics.flashPeak,
          excitation,
        );
      } else if (state === 3 || node.fanIn >= 38) {
        const glowSize = radius * 3.4;
        context.globalAlpha = (state === 3 ? 0.5 : 0.14) * depthFade * dim;
        context.drawImage(
          glowSteel,
          node.screenX - glowSize / 2,
          node.screenY - glowSize / 2,
          glowSize,
          glowSize,
        );
      }
      context.globalAlpha = 1;

      let color =
        isFieldActive
          ? mix(
              node.base,
              RGB_VIOLET,
              0.5 + fieldActivation * 0.45,
            )
          : node.base;
      if (excitation > 0.55) {
        color = mix(color, RGB_HOT, (excitation - 0.55) / 0.7);
      } else if (excitation > 0.03) {
        color = mix(color, RGB_VIOLET, excitation * 0.72);
      }
      const coreOpacity =
        (state === 1
          ? 0.22
          : state === 3
            ? 1
            : 0.62
              + 0.18 * fieldActivation
              + 0.2 * Math.min(1, excitation * 2.5))
        * depthFade;
      context.globalCompositeOperation =
        excitation > 0.03 || isFieldActive
          ? "lighter"
          : "source-over";
      context.fillStyle = rgba(color, coreOpacity);
      context.beginPath();
      context.arc(node.screenX, node.screenY, radius, 0, Math.PI * 2);
      context.fill();
      if (excitation > 0.35) {
        context.fillStyle = rgba(RGB_HOT, excitation * depthFade);
        context.beginPath();
        context.arc(node.screenX, node.screenY, radius * 0.45, 0, Math.PI * 2);
        context.fill();
      }
      context.globalCompositeOperation = "lighter";
      const age = time - node.firedAt;
      if (age < 900 && excitation > 0.08) {
        const progress = age / 900;
        context.strokeStyle = rgba(
          RGB_VIOLET,
          (1 - smoothstep(progress)) * 0.5 * depthFade,
        );
        context.lineWidth = 1.2;
        context.beginPath();
        context.arc(
          node.screenX,
          node.screenY,
          radius + progress * 26 * Math.min(1.6, node.screenScale * 2),
          0,
          Math.PI * 2,
        );
        context.stroke();
      }
      context.globalCompositeOperation = "source-over";
      if (index === selected) {
        context.strokeStyle = rgba(RGB_HOT, 0.92);
        context.lineWidth = 1.5;
        context.beginPath();
        context.arc(
          node.screenX,
          node.screenY,
          radius + 5 + Math.sin(time * 0.005) * 1.5,
          0,
          Math.PI * 2,
        );
        context.stroke();
        context.strokeStyle = rgba(RGB_VIOLET, 0.72);
        context.lineWidth = 1;
        context.beginPath();
        context.arc(
          node.screenX,
          node.screenY,
          radius + 8 + Math.sin(time * 0.005) * 1.2,
          0,
          Math.PI * 2,
        );
        context.stroke();
      }
    });
  }

  function drawNodeEffects(time) {
    context.globalCompositeOperation = "lighter";
    for (let index = nodeEffects.length - 1; index >= 0; index -= 1) {
      const effect = nodeEffects[index];
      const node = nodes[effect.nodeIndex];
      const progress = (time - effect.startedAt) / effect.duration;
      if (!node || progress >= 1) {
        nodeEffects.splice(index, 1);
        continue;
      }
      if (progress < 0 || node.viewDepth > 9e8) continue;
      const radius = Math.max(3, node.radius * node.screenScale);
      const fade = 1 - progress;
      if (effect.kind === "stimulus") {
        [0, 0.24].forEach((delay) => {
          const phase = Math.max(0, Math.min(1, (progress - delay) / 0.62));
          if (!phase || phase >= 1) return;
          context.strokeStyle = rgba(RGB_FIRE, (1 - phase) * 0.9);
          context.lineWidth = 1.2 + effect.delta * 2;
          context.beginPath();
          context.arc(node.screenX, node.screenY, radius + phase * 24, 0, Math.PI * 2);
          context.stroke();
        });
        context.globalAlpha = fade;
        context.drawImage(glowFire, node.screenX - 18, node.screenY - 18, 36, 36);
      } else if (effect.kind === "inhibit" || effect.kind === "reject") {
        context.strokeStyle = rgba(RGB_INHIBIT, 0.35 + fade * 0.65);
        context.lineWidth = 1.4 + effect.delta * 2;
        context.beginPath();
        context.arc(node.screenX, node.screenY, radius + (1 - progress) * 28, 0, Math.PI * 2);
        context.stroke();
        context.fillStyle = rgba(RGB_INHIBIT, fade);
        context.font = `700 ${Math.max(9, radius * 1.2)}px ${getComputedStyle(document.body).getPropertyValue("--mono")}`;
        context.textAlign = "center";
        context.fillText("−", node.screenX, node.screenY + radius * 0.35);
      } else if (effect.kind === "commit_queued" || effect.kind === "commit_applied") {
        context.strokeStyle = rgba(RGB_COMMIT, 0.35 + fade * 0.65);
        context.lineWidth = 1.5 + effect.delta * 1.6;
        context.beginPath();
        context.arc(node.screenX, node.screenY, radius + progress * 18, 0, Math.PI * 2);
        context.stroke();
        context.fillStyle = rgba(RGB_COMMIT, fade);
        context.font = `700 ${Math.max(8, radius)}px ${getComputedStyle(document.body).getPropertyValue("--mono")}`;
        context.textAlign = "center";
        context.fillText("✓", node.screenX, node.screenY + radius * 0.35);
      } else if (effect.kind === "fault") {
        context.globalAlpha = fade;
        context.drawImage(glowFault, node.screenX - 24, node.screenY - 24, 48, 48);
        context.fillStyle = rgba(RGB_FAULT, fade);
        context.fillText("×", node.screenX, node.screenY);
      } else if (effect.kind === "arrival") {
        context.strokeStyle = rgba(RGB_ELECTRIC, fade * 0.95);
        context.lineWidth = 1 + effect.delta * 2;
        context.beginPath();
        context.arc(node.screenX, node.screenY, radius + progress * 20, 0, Math.PI * 2);
        context.stroke();
        context.globalAlpha = fade;
        context.drawImage(glowElectric, node.screenX - 20, node.screenY - 20, 40, 40);
      }
      context.globalAlpha = 1;
    }
    context.globalCompositeOperation = "source-over";
  }

  function drawLabels(time) {
    context.font = `10.5px ${getComputedStyle(document.body).getPropertyValue("--mono")}`;
    context.textAlign = "center";
    const occupied = [];
    const activeLabels = new Set(
      nodes
        .filter(
          (node) =>
            nodeState[node.index] >= 2
            && node.viewDepth <= 9e8
            && node.fieldActivation > 0.05,
        )
        .sort(
          (left, right) =>
            right.fieldActivation - left.fieldActivation
            || right.fanIn - left.fanIn
            || left.id.localeCompare(right.id),
        )
        .slice(0, ACTIVE_LABEL_LIMIT)
        .map((node) => node.index),
    );
    const candidates = nodes
      .filter((node) => {
        const state = nodeState[node.index];
        const focused = node.index === selected || node.index === hovered;
        return (
          state >= 2
          && node.viewDepth <= 9e8
          && (state === 3
            || focused
            || activeLabels.has(node.index)
            || (!activeLabels.size
              && labelHubs.has(node.index)
              && camera.distance < 1500
              && node.viewDepth < camera.distance))
        );
      })
      .sort((left, right) => {
        const leftPriority =
          (left.index === selected ? 100000 : 0)
          + (left.index === hovered ? 50000 : 0)
          + excitationLevel(left, time) * 1000
          + left.fieldActivation * 900
          + left.fanIn;
        const rightPriority =
          (right.index === selected ? 100000 : 0)
          + (right.index === hovered ? 50000 : 0)
          + excitationLevel(right, time) * 1000
          + right.fieldActivation * 900
          + right.fanIn;
        return rightPriority - leftPriority;
      });
    candidates.forEach((node) => {
      const state = nodeState[node.index];
      const hot = excitationLevel(node, time) > 0.5;
      const fieldActive = node.fieldActivation > 0.05;
      const y = node.screenY - node.radius * node.screenScale - 7;
      if (
        node.screenX < -40
        || node.screenX > width + 40
        || y < -20
        || y > height + 20
      ) {
        return;
      }
      const depthFade = fog(node.viewDepth);
      const labelWidth = context.measureText(node.name).width;
      const bounds = {
        left: node.screenX - labelWidth / 2 - 4,
        right: node.screenX + labelWidth / 2 + 4,
        top: y - 11,
        bottom: y + 3,
      };
      const overlaps = occupied.some(
        (other) =>
          bounds.left < other.right
          && bounds.right > other.left
          && bounds.top < other.bottom
          && bounds.bottom > other.top,
      );
      if (overlaps && state !== 3) return;
      occupied.push(bounds);
      cortexMetrics.labelsPainted += 1;
      context.fillStyle = `rgba(3,5,10,${0.7 * depthFade})`;
      context.fillRect(node.screenX - labelWidth / 2 - 3, y - 9, labelWidth + 6, 12);
      context.fillStyle =
        node.index === selected || node.index === hovered
          ? rgba(RGB_HOT, 0.92 * depthFade)
          : hot
            ? rgba(RGB_VIOLET, 0.95 * depthFade)
            : state === 3
              ? rgba(RGB_FIRE, 0.85 * depthFade)
              : fieldActive
                ? rgba(
                    RGB_VIOLET,
                    (0.55 + node.fieldActivation * 0.4) * depthFade,
                  )
                : rgba([160, 178, 210], 0.8 * depthFade);
      context.fillText(node.name, node.screenX, y);
    });
  }

  function draw(time) {
    cortexMetrics.violetNodes = 0;
    cortexMetrics.labelsPainted = 0;
    cortexMetrics.afterglowEdges = 0;
    cortexMetrics.flashPeak = 0;
    context.setTransform(pixelRatio, 0, 0, pixelRatio, 0, 0);
    context.clearRect(0, 0, width, height);
    context.fillStyle = STEEL;
    stars.forEach((star) => {
      context.globalAlpha =
        0.05 + 0.06 * Math.sin(time * 0.001 * star.speed + star.phase);
      context.beginPath();
      context.arc(star.x * width, star.y * height, star.radius, 0, Math.PI * 2);
      context.fill();
    });
    context.globalAlpha = 1;
    drawEdges();
    drawEdgeAfterglows(time);
    drawPulses(time);
    drawNodes(time);
    drawNodeEffects(time);
    drawLabels(time);
    publishVisualMetrics(time);
  }

  let previousTime = performance.now();
  let simulationAccumulator = 0;
  function frame(now) {
    const delta = Math.min(60, now - previousTime);
    cortexMetrics.frameDurations.push(delta);
    if (cortexMetrics.frameDurations.length > 240) {
      cortexMetrics.frameDurations.shift();
    }
    previousTime = now;
    simulationAccumulator += delta;
    while (simulationAccumulator > 15) {
      tick();
      simulationAccumulator -= 15;
    }
    if (autoRotate && !dragging && now - lastInteraction > 2600) {
      camera.theta += delta * 0.000045;
    }
    if (cameraTarget) {
      camera.theta = angleLerp(camera.theta, cameraTarget.theta, 0.1);
      camera.phi += (cameraTarget.phi - camera.phi) * 0.1;
      camera.distance += (cameraTarget.distance - camera.distance) * 0.1;
      if (
        Math.abs(cameraTarget.distance - camera.distance) < 2
        && Math.abs(cameraTarget.phi - camera.phi) < 0.005
        && Math.abs(
          ((cameraTarget.theta - camera.theta + Math.PI * 3) % (Math.PI * 2))
            - Math.PI,
        ) < 0.005
      ) {
        cameraTarget = null;
      }
    }
    if (stateDirty) recomputeState();
    projectAll();
    draw(now);
    requestAnimationFrame(frame);
  }

  function pick(x, y) {
    let best = -1;
    let bestDistance = 1e9;
    nodes.forEach((node) => {
      if (nodeState[node.index] === 0 || node.viewDepth > 9e8) return;
      const distance =
        Math.hypot(node.screenX - x, node.screenY - y)
        - Math.max(5, node.radius * node.screenScale);
      if (distance < 4 && distance < bestDistance) {
        bestDistance = distance;
        best = node.index;
      }
    });
    return best;
  }

  function select(index) {
    selected = index;
    stateDirty = true;
    renderPanel();
    document.getElementById("panelBody").scrollTop = 0;
    renderTreeSelection();
    if (index >= 0) focusNode(index);
  }

  function bindCanvasInteractions() {
    const tooltip = document.getElementById("tooltip");
    canvas.addEventListener("pointerdown", (event) => {
      canvas.setPointerCapture(event.pointerId);
      downPoint = { x: event.offsetX, y: event.offsetY };
      moved = false;
      dragging = true;
      lastInteraction = performance.now();
    });
    canvas.addEventListener("pointermove", (event) => {
      const x = event.offsetX;
      const y = event.offsetY;
      if (downPoint && Math.hypot(x - downPoint.x, y - downPoint.y) > 4) {
        moved = true;
      }
      if (downPoint && moved) {
        camera.theta -= event.movementX * 0.0045;
        camera.phi = Math.max(
          -1.35,
          Math.min(1.35, camera.phi + event.movementY * 0.0045),
        );
        cameraTarget = null;
        lastInteraction = performance.now();
        return;
      }
      const index = pick(x, y);
      if (index !== hovered) {
        hovered = index;
        stateDirty = true;
      }
      if (index >= 0) {
        const node = nodes[index];
        tooltip.style.display = "block";
        tooltip.style.left = `${Math.min(width - 240, x + 16)}px`;
        tooltip.style.top = `${Math.max(8, y - 14)}px`;
        tooltip.innerHTML = `<div class="t">${escapeHtml(node.id)}</div>
          <div class="s">${node.lineCount.toLocaleString()} lines · in ${node.fanIn} / out ${node.fanOut}${node.entrypoint ? " · entrypoint" : ""}</div>`;
        canvas.style.cursor = "pointer";
      } else {
        tooltip.style.display = "none";
        canvas.style.cursor = "grab";
      }
    });
    canvas.addEventListener("pointerup", (event) => {
      if (!moved) {
        const index = pick(event.offsetX, event.offsetY);
        select(index);
      }
      downPoint = null;
      dragging = false;
      lastInteraction = performance.now();
    });
    canvas.addEventListener("dblclick", (event) => {
      const index = pick(event.offsetX, event.offsetY);
      if (index >= 0) {
        select(index);
        flashTicker(`preview · ${nodes[index].id}`);
      } else {
        fitView();
      }
    });
    canvas.addEventListener(
      "wheel",
      (event) => {
        event.preventDefault();
        lastInteraction = performance.now();
        camera.distance = Math.max(
          240,
          Math.min(4200, camera.distance * Math.exp(event.deltaY * 0.0011)),
        );
        cameraTarget = null;
      },
      { passive: false },
    );
  }

  const packageStats = {};
  function buildTree() {
    const tree = document.getElementById("tree");
    packageList.forEach((packageName) => {
      const packageNodes = nodes.filter(
        (node) => node.packageName === packageName,
      );
      packageStats[packageName] = {
        count: packageNodes.length,
        lines: packageNodes.reduce((total, node) => total + node.lineCount, 0),
      };
    });
    document.getElementById("pkgCount").textContent = `${packageList.length} pkgs`;
    packageList.forEach((packageName) => {
      const row = document.createElement("div");
      row.className = "pkgRow";
      const dotColor = rgba(
        shade(RGB_STEEL, packageShade[packageName] || 1),
        1,
      );
      row.innerHTML = `<span class="dot" style="background:${dotColor}"></span>
        <span class="n">${escapeHtml(packageName)}</span>
        <span class="c">${packageStats[packageName].count}</span>
        <span class="ar">▶</span>`;
      const list = document.createElement("div");
      list.className = "modList";
      packageNodesByFanIn(packageName).forEach((node) => {
        const moduleRow = document.createElement("div");
        moduleRow.className = "modRow";
        moduleRow.dataset.index = String(node.index);
        moduleRow.innerHTML = `<span title="${escapeHtml(node.title)}">${escapeHtml(node.name)}</span>
          ${node.entrypoint ? '<span class="ep">▸ep</span>' : ""}
          <span class="l">${node.lineCount}</span>`;
        moduleRow.addEventListener("click", () => select(node.index));
        moduleRow.addEventListener("dblclick", () => {
          select(node.index);
          flashTicker(`preview · ${node.id}`);
        });
        list.appendChild(moduleRow);
      });
      row.addEventListener("click", (event) => {
        if (event.target.classList.contains("dot")) {
          if (packageOff.has(packageName)) packageOff.delete(packageName);
          else packageOff.add(packageName);
          row.classList.toggle("off", packageOff.has(packageName));
          stateDirty = true;
          return;
        }
        row.classList.toggle("open");
        list.classList.toggle("open");
      });
      tree.appendChild(row);
      tree.appendChild(list);
    });
  }

  function packageNodesByFanIn(packageName) {
    return nodes
      .filter((node) => node.packageName === packageName)
      .sort((left, right) => right.fanIn - left.fanIn);
  }

  function renderTreeSelection() {
    const tree = document.getElementById("tree");
    tree
      .querySelectorAll(".modRow.sel")
      .forEach((element) => element.classList.remove("sel"));
    if (selected < 0) return;
    const element = tree.querySelector(
      `.modRow[data-index="${selected}"]`,
    );
    if (!element) return;
    const list = element.parentElement;
    if (list && !list.classList.contains("open")) {
      list.classList.add("open");
      if (list.previousSibling) list.previousSibling.classList.add("open");
    }
    element.classList.add("sel");
    element.scrollIntoView({ block: "nearest" });
  }

  function fieldValue(value, digits = 3) {
    return Number.isFinite(Number(value)) ? Number(value).toFixed(digits) : "—";
  }

  function fieldStateHtml() {
    const summary = fieldState.summary || {};
    const latency = summary.latency_ms || {};
    const agreement = summary.teacher_agreement !== null
      && Number.isFinite(Number(summary.teacher_agreement))
      ? `${(Number(summary.teacher_agreement) * 100).toFixed(1)}%`
      : "collecting";
    const active = window.CortexField.activeNodes(fieldState, 14);
    const trace = [...fieldState.events].slice(-7).reverse();
    const statusLabel = fieldState.fault
      ? `FAULT · ${fieldState.fault}`
      : fieldState.stale
        ? "STALE"
        : fieldState.status.toUpperCase();
    return `
      <div class="sec fieldStateSec"><h3>FIELD STATE</h3>
        <div class="fieldStatus ${escapeHtml(fieldState.status)}">
          <span class="stateGlyph" aria-hidden="true">${fieldState.fault ? "×" : fieldState.stale ? "◷" : "●"}</span>
          <b>${escapeHtml(statusLabel)}</b>
          <span>${escapeHtml(fieldState.mode)} · ${escapeHtml(fieldState.source)}</span>
        </div>
        <div class="fieldSession">${escapeHtml(fieldState.sessionHash || "no session")}</div>
        <div class="fieldMetrics">
          <div><span>active</span><b>${summary.active || 0}</b></div>
          <div><span>candidate</span><b>${summary.candidate || 0}</b></div>
          <div><span>commit</span><b>${summary.commit || 0}</b></div>
          <div><span>reject</span><b>${summary.reject || 0}</b></div>
          <div><span>agreement</span><b>${agreement}</b></div>
          <div><span>field p95</span><b>${Number.isFinite(Number(latency.p95)) ? `${Math.round(latency.p95)}ms` : "—"}</b></div>
        </div>
      </div>
      <div class="sec"><h3>DECISION TRACE · SEQ ${fieldState.seq}</h3>
        <div class="decisionTrace">
          ${trace.length
            ? trace.map((event) => {
                const route = event.kind === "spread"
                  ? `${event.source_page_id} → ${event.target_page_id}`
                  : window.CortexField.eventPageId(event) || "field";
                return `<div class="traceRow ${escapeHtml(event.kind)}">
                  <span class="traceGlyph" aria-hidden="true">${event.kind === "spread" ? "↝" : event.kind.includes("commit") ? "✓" : ["inhibit", "reject"].includes(event.kind) ? "−" : event.kind === "fault" ? "×" : "●"}</span>
                  <b>${escapeHtml(event.kind)}</b>
                  <span title="${escapeHtml(route)}">${escapeHtml(route)}</span>
                  <small>#${event.seq} · Δ${fieldValue(event.delta)}</small>
                </div>`;
              }).join("")
            : '<div class="ghost">No Field events for this session.</div>'}
        </div>
      </div>
      <div class="sec"><h3>ACTIVE NODES · ${active.length}</h3>
        <table class="activeTable">
          <thead><tr><th>state</th><th>page</th><th>activation</th></tr></thead>
          <tbody>${active.length
            ? active.map((row) => {
                const node = byId.get(row.pageId);
                return `<tr>
                  <td><span class="stateText">${escapeHtml(row.state)}</span></td>
                  <td>${node ? `<button type="button" data-index="${node.index}">${escapeHtml(row.pageId)}</button>` : escapeHtml(row.pageId)}</td>
                  <td>${fieldValue(row.activation)}</td>
                </tr>`;
              }).join("")
            : '<tr><td colspan="3">No active nodes.</td></tr>'}</tbody>
        </table>
      </div>
      <div class="sec"><h3>LEGEND</h3>
        <div class="fieldLegend">
          <span class="orange">◎ stimulus</span>
          <span class="yellow">↝ spread</span>
          <span class="violet">◉ active</span>
          <span class="green">✓ commit</span>
          <span class="blue">− inhibit</span>
          <span class="red">× fault</span>
        </div>
      </div>`;
  }

  function selectedFieldHtml(node) {
    const state = fieldState.nodes.get(node.id);
    const components = state?.components || {
      direct: 0,
      spread: 0,
      negative: 0,
      inhibition: 0,
      anti_index: 0,
      hub_penalty: 0,
    };
    return `
      <div class="sec"><h3>ACTIVATION</h3>
        <div class="activationTotal">${fieldValue(state?.activation || 0)}</div>
        <div class="componentGrid">
          <div><span>direct</span><b>${fieldValue(components.direct)}</b></div>
          <div><span>spread</span><b>${fieldValue(components.spread)}</b></div>
          <div><span>negative</span><b>${fieldValue(components.negative)}</b></div>
          <div><span>inhibition</span><b>${fieldValue(components.inhibition)}</b></div>
          <div><span>anti-index</span><b>${fieldValue(components.anti_index)}</b></div>
          <div><span>hub penalty</span><b>${fieldValue(components.hub_penalty)}</b></div>
        </div>
        <div class="mrow"><span>state</span><b>${escapeHtml(state?.state || "inactive")}</b></div>
        <div class="mrow"><span>reason</span><b>${escapeHtml(state?.reasonCode || "—")}</b></div>
        <div class="mrow"><span>certificate</span><b title="${escapeHtml(state?.certificateId || "")}">${escapeHtml(state?.certificateId || "—")}</b></div>
      </div>`;
  }

  function overviewHtml() {
    const staticLinks = data.meta.static || 0;
    const deferredLinks = data.meta.deferred || 0;
    const denominator = Math.max(1, staticLinks + deferredLinks);
    const percent = Math.round((staticLinks / denominator) * 100);
    const topHubs = [...nodes]
      .sort((left, right) => right.fanIn - left.fanIn)
      .slice(0, 7);
    const topFanIn = Math.max(1, topHubs[0]?.fanIn || 1);
    const loadPackages = [...packageList]
      .sort(
        (left, right) =>
          packageStats[right].lines - packageStats[left].lines,
      )
      .slice(0, 18);
    const maxLines = Math.max(
      1,
      ...loadPackages.map((packageName) => packageStats[packageName].lines),
    );
    return `${fieldStateHtml()}
      <div class="sec"><h3>BINDING INTEGRITY</h3>
        <div id="gaugeWrap"><canvas id="gauge" width="236" height="236"></canvas>
          <div class="gLegend">
            <div><b>${percent}%</b> static bind</div>
            <div class="k"><span class="sw" style="background:var(--amber)"></span>static ${staticLinks.toLocaleString()}</div>
            <div class="k"><span class="sw" style="background:#7d92b5"></span>deferred ${deferredLinks.toLocaleString()}</div>
            <div class="k"><span class="sw" style="background:#4d5b76"></span>dynamic ${(data.meta.spawn || 0).toLocaleString()}</div>
          </div>
        </div>
      </div>
      <div class="sec"><h3>TOP HUBS · FAN-IN</h3>
        ${topHubs
          .map(
            (node) => `<div class="hubRow" data-index="${node.index}">
              <span class="nm">${escapeHtml(node.id)}</span>
              <span class="bar"><i style="width:${Math.round((node.fanIn / topFanIn) * 100)}%"></i></span>
              <span class="v">${node.fanIn}</span>
            </div>`,
          )
          .join("")}
      </div>
      <div class="sec"><h3>PACKAGE LOAD · LINES</h3>
        ${loadPackages
          .map(
            (packageName) => `<div class="pkgBar">
              <span class="nm">${escapeHtml(packageName)}</span>
              <span class="bar"><i style="width:${Math.round((packageStats[packageName].lines / maxLines) * 100)}%"></i></span>
              <span class="v">${(packageStats[packageName].lines / 1000).toFixed(1)}k</span>
            </div>`,
          )
          .join("")}
        ${packageList.length > loadPackages.length ? `<div class="ghost">+ ${packageList.length - loadPackages.length} more categories</div>` : ""}
      </div>
      <div class="sec"><h3>STIMULUS PATHWAYS</h3>
        <div class="mrow"><span>⚡ RECALL</span><b>read / search → pages</b></div>
        <div class="mrow"><span>⚡ SAVE</span><b>host records → raw</b></div>
        <div class="mrow"><span>⚡ INGEST</span><b>drain → pages</b></div>
        <div class="ghost">Liveは実Field eventのみ。DEMO/REPLAYは表示専用でbackendへ書きません。</div>
      </div>`;
  }

  function selectedHtml(node) {
    const dependsOn = links
      .filter((link) => link.source === node.index && link.kind < 2)
      .map((link) => nodes[link.target])
      .filter(Boolean);
    const dependedBy = links
      .filter((link) => link.target === node.index && link.kind < 2)
      .map((link) => nodes[link.source])
      .filter(Boolean);
    const chips = (items) =>
      items.length
        ? items
            .slice(0, 80)
            .map(
              (item) =>
                `<span data-index="${item.index}" title="${escapeHtml(item.title)}">${escapeHtml(item.id)}</span>`,
            )
            .join("")
        : '<div class="ghost">— none</div>';
    return `
      <div id="selCard">
        <button id="closeSel" type="button">✕</button>
        <div class="nm">${escapeHtml(node.id)}</div>
        <div class="title">${escapeHtml(node.title)}</div>
        <span class="pk"><span class="sw"></span>${escapeHtml(node.packageName)}</span>
        ${node.entrypoint ? '<span class="pk" style="color:var(--amber);border-color:rgba(255,180,84,.4)">entrypoint</span>' : ""}
        <div style="margin-top:10px">
          <div class="mrow"><span>lines</span><b>${node.lineCount.toLocaleString()}</b></div>
          <div class="mrow"><span>size</span><b>${formatBytes(node.byteCount)}</b></div>
          <div class="mrow"><span>fan-in</span><b>${node.fanIn}</b></div>
          <div class="mrow"><span>fan-out</span><b>${node.fanOut}</b></div>
          ${node.updated ? `<div class="mrow"><span>updated</span><b>${escapeHtml(node.updated)}</b></div>` : ""}
        </div>
      </div>
      ${selectedFieldHtml(node)}
      ${fieldStateHtml()}
      ${node.tags.length ? `<div class="sec"><h3>TAGS · ${node.tags.length}</h3><div class="tagChips">${node.tags.map((tag) => `<span>${escapeHtml(tag)}</span>`).join("")}</div></div>` : ""}
      <div class="sec"><h3>LINKS TO · ${dependsOn.length}</h3>
        <div class="depChips">${chips(dependsOn)}</div>
      </div>
      <div class="sec"><h3>LINKED BY · ${dependedBy.length}</h3>
        <div class="depChips">${chips(dependedBy)}</div>
      </div>`;
  }

  function renderPanel() {
    const panel = document.getElementById("panelBody");
    if (selected < 0) {
      panel.innerHTML = overviewHtml();
      drawGauge();
    } else {
      panel.innerHTML = selectedHtml(nodes[selected]);
    }
  }

  function drawGauge() {
    const gauge = document.getElementById("gauge");
    if (!gauge) return;
    const gaugeContext = gauge.getContext("2d");
    const staticLinks = data.meta.static || 0;
    const deferredLinks = data.meta.deferred || 0;
    const target = staticLinks / Math.max(1, staticLinks + deferredLinks);
    let animation = 0;
    function step() {
      if (!document.getElementById("gauge")) return;
      animation = Math.min(1, animation + 0.03);
      const value = target * (1 - Math.pow(1 - animation, 3));
      gaugeContext.setTransform(2, 0, 0, 2, 0, 0);
      gaugeContext.clearRect(0, 0, 118, 118);
      const centerX = 59;
      const centerY = 59;
      gaugeContext.strokeStyle = "rgba(110,140,190,.1)";
      gaugeContext.lineWidth = 1;
      for (let index = 0; index < 40; index += 1) {
        const angle = (index / 40) * Math.PI * 2;
        gaugeContext.beginPath();
        gaugeContext.moveTo(
          centerX + Math.cos(angle) * 52,
          centerY + Math.sin(angle) * 52,
        );
        gaugeContext.lineTo(
          centerX + Math.cos(angle) * 55,
          centerY + Math.sin(angle) * 55,
        );
        gaugeContext.stroke();
      }
      const start = -Math.PI / 2;
      gaugeContext.lineWidth = 7;
      gaugeContext.lineCap = "round";
      gaugeContext.strokeStyle = "rgba(125,146,181,.16)";
      gaugeContext.beginPath();
      gaugeContext.arc(centerX, centerY, 44, start, start + Math.PI * 2);
      gaugeContext.stroke();
      const gradient = gaugeContext.createLinearGradient(0, 0, 118, 118);
      gradient.addColorStop(0, "#ffb454");
      gradient.addColorStop(1, "#f5934b");
      gaugeContext.strokeStyle = gradient;
      gaugeContext.shadowColor = "rgba(255,180,84,.6)";
      gaugeContext.shadowBlur = 8;
      gaugeContext.beginPath();
      gaugeContext.arc(
        centerX,
        centerY,
        44,
        start,
        start + Math.PI * 2 * value,
      );
      gaugeContext.stroke();
      gaugeContext.shadowBlur = 0;
      const mono = getComputedStyle(document.body).getPropertyValue("--mono");
      gaugeContext.fillStyle = "#ffd9a0";
      gaugeContext.font = `700 19px ${mono}`;
      gaugeContext.textAlign = "center";
      gaugeContext.fillText(`${Math.round(value * 100)}%`, centerX, centerY + 2);
      gaugeContext.fillStyle = "rgba(127,144,176,.9)";
      gaugeContext.font = `7.5px ${mono}`;
      gaugeContext.fillText("STATIC BIND", centerX, centerY + 16);
      if (animation < 1) requestAnimationFrame(step);
    }
    step();
  }

  function formatBytes(bytes) {
    if (!bytes) return "—";
    if (bytes < 1024) return `${bytes} B`;
    if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
    return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
  }

  function bindInterface() {
    const panel = document.getElementById("panelBody");
    panel.addEventListener("click", (event) => {
      if (event.target.id === "closeSel") {
        select(-1);
        return;
      }
      const target = event.target.closest("[data-index]");
      if (target) select(Number(target.dataset.index));
    });
    panel.addEventListener("keydown", (event) => {
      if (!["ArrowDown", "ArrowUp"].includes(event.key)) return;
      const buttons = [...panel.querySelectorAll(".activeTable button")];
      const current = buttons.indexOf(document.activeElement);
      if (current < 0 || !buttons.length) return;
      event.preventDefault();
      const offset = event.key === "ArrowDown" ? 1 : -1;
      buttons[(current + offset + buttons.length) % buttons.length].focus();
    });

    const search = document.getElementById("search");
    const searchHits = document.getElementById("searchHits");
    search.addEventListener("input", () => {
      query = search.value.trim().toLowerCase();
      matches = new Set();
      if (query) {
        nodes.forEach((node) => {
          if (
            node.id.toLowerCase().includes(query)
            || node.title.toLowerCase().includes(query)
          ) {
            matches.add(node.index);
          }
        });
      }
      searchHits.textContent = query ? `${matches.size} hits` : "";
      stateDirty = true;
    });
    search.addEventListener("keydown", (event) => {
      if (event.key === "Enter" && matches.size) {
        const exact = nodes.find(
          (node) => node.id.toLowerCase() === query,
        );
        select(exact ? exact.index : [...matches][0]);
      }
      if (event.key === "Escape") {
        search.value = "";
        query = "";
        matches.clear();
        searchHits.textContent = "";
        stateDirty = true;
        search.blur();
      }
    });
    window.addEventListener("keydown", (event) => {
      if ((event.metaKey || event.ctrlKey) && event.key === "k") {
        event.preventDefault();
        search.focus();
        search.select();
      } else if (event.key === "Escape" && document.activeElement !== search) {
        select(-1);
      }
    });

    document.getElementById("mOrganic").addEventListener("click", () => {
      setMode("organic");
    });
    document.getElementById("mCluster").addEventListener("click", () => {
      setMode("cluster");
    });
    document.querySelectorAll(".stim").forEach((button) => {
      button.addEventListener("click", () => stimulate(button.dataset.s));
    });
    const liveToggle = document.getElementById("tLive");
    liveToggle.addEventListener("click", () => {
      liveEventsEnabled = !liveEventsEnabled;
      liveToggle.classList.toggle("on", liveEventsEnabled);
      flashTicker(
        liveEventsEnabled
          ? "LIVE PAINT resumed · Field state stayed current"
          : "LIVE PAINT paused · Field state still updating",
      );
    });
    const motionToggle = document.getElementById("tMotion");
    motionToggle.addEventListener("click", () => {
      motionEnabled = !motionEnabled;
      motionToggle.classList.toggle("on", motionEnabled);
      flashTicker(
        motionEnabled
          ? "motion enabled"
          : "motion paused · static edge arrows remain",
      );
    });
    const rotateToggle = document.getElementById("tRot");
    rotateToggle.addEventListener("click", () => {
      autoRotate = !autoRotate;
      rotateToggle.classList.toggle("on", autoRotate);
    });
    const soundToggle = document.getElementById("tSnd");
    soundToggle.addEventListener("click", () => {
      soundOn = !soundOn;
      soundToggle.classList.toggle("on", soundOn);
      if (soundOn) {
        if (!audioContext) {
          audioContext = new (window.AudioContext || window.webkitAudioContext)();
        }
        if (audioContext.state === "suspended") audioContext.resume();
      }
    });
    document.getElementById("visSlider").addEventListener("input", (event) => {
      edgeVisibility = Number(event.target.value) / 100;
    });
    document.getElementById("zIn").addEventListener("click", () => {
      camera.distance = Math.max(240, camera.distance / 1.4);
      cameraTarget = null;
    });
    document.getElementById("zOut").addEventListener("click", () => {
      camera.distance = Math.min(4200, camera.distance * 1.4);
      cameraTarget = null;
    });
    document.getElementById("zFit").addEventListener("click", () => {
      select(-1);
      fitView();
    });
    document.getElementById("sessionSelect").addEventListener("change", (event) => {
      loadField(event.target.value);
    });
    stage.addEventListener("keydown", (event) => {
      if (!["ArrowDown", "ArrowUp", "ArrowLeft", "ArrowRight"].includes(event.key)) {
        return;
      }
      const active = window.CortexField.activeNodes(fieldState, 30)
        .map((row) => byId.get(row.pageId))
        .filter(Boolean);
      if (!active.length) return;
      event.preventDefault();
      const current = active.findIndex((node) => node.index === selected);
      const backwards = event.key === "ArrowUp" || event.key === "ArrowLeft";
      const next = current < 0
        ? 0
        : (current + (backwards ? -1 : 1) + active.length) % active.length;
      select(active[next].index);
    });
    reducedMotion.addEventListener("change", () => {
      flashTicker(
        reducedMotion.matches
          ? "reduced motion · static electric arrows"
          : "electric arc motion restored",
      );
    });
  }

  function setMode(nextMode) {
    mode = nextMode;
    reheat(0.7);
    document
      .getElementById("mOrganic")
      .classList.toggle("on", mode === "organic");
    document
      .getElementById("mCluster")
      .classList.toggle("on", mode === "cluster");
    window.setTimeout(fitView, 900);
  }

  let factIndex = 0;
  let tickerHold = 0;
  let tickerTransition = 0;
  let facts = [];
  const tickerText = document.getElementById("tickerTx");
  function nextFact() {
    window.clearTimeout(tickerTransition);
    tickerText.style.opacity = 0;
    tickerTransition = window.setTimeout(() => {
      if (performance.now() <= tickerHold) return;
      tickerText.textContent = facts[factIndex % facts.length];
      factIndex += 1;
      tickerText.style.opacity = 1;
    }, 350);
  }

  function flashTicker(message) {
    window.clearTimeout(tickerTransition);
    tickerText.textContent = message;
    tickerText.style.opacity = 1;
    tickerHold = performance.now() + 3500;
  }

  function initializeHeader() {
    document.getElementById("stN").textContent = nodeCount.toLocaleString();
    document.getElementById("stE").textContent = (
      (data.meta.static || 0) + (data.meta.deferred || 0)
    ).toLocaleString();
    document.getElementById("stC").textContent = data.meta.commit || "local";
    document.getElementById("genInfo").textContent =
      `local wiki · ${data.meta.generated || "live"}`;
    const topHub = [...nodes].sort(
      (left, right) => right.fanIn - left.fanIn,
    )[0];
    facts = [
      "Stateful Recall Field — 実eventだけを可視化",
      `${topHub?.id || "—"} fan-in ${topHub?.fanIn || 0} — 最深部のハブニューロン`,
      `${nodeCount.toLocaleString()} neurons / ${((data.meta.static || 0) + (data.meta.deferred || 0)).toLocaleString()} synapses`,
      "DEMOボタンは表示専用 — backend state / metricsは不変",
      "orange=stimulus · yellow=spread · violet=activation",
      "ドラッグで回転 · ホイールでズーム · ⌘K で検索",
      `総行数 ${(data.meta.totalLines || 0).toLocaleString()} lines @ ${data.meta.commit || "local"}`,
    ];
    nextFact();
    window.setInterval(() => {
      if (performance.now() > tickerHold) nextFact();
    }, 4600);
  }

  function setEventStatus(online) {
    const badge = document.getElementById("cortexStatus");
    const live = document.getElementById("eventLive");
    const fault = Boolean(fieldState.fault);
    const stale = fieldState.stale && fieldState.sessionHash;
    badge.classList.toggle("offline", !online || fault || stale);
    badge.classList.toggle("fault", fault);
    live.classList.toggle("offline", !online || fault || stale);
    live.classList.toggle("fault", fault);
    if (fault) {
      badge.lastChild.textContent = "FIELD FAULT";
      live.textContent = "× FAULT";
    } else if (stale) {
      badge.lastChild.textContent = "FIELD STALE";
      live.textContent = "◷ STALE";
    } else {
      badge.lastChild.textContent = online ? "FIELD ONLINE" : "FIELD LINKING";
      live.textContent = online ? "● LIVE" : "○ LINKING";
    }
  }

  function updateSessionControl() {
    const selectElement = document.getElementById("sessionSelect");
    selectElement.innerHTML = "";
    if (!fieldState.sessions.length) {
      const option = document.createElement("option");
      option.value = "";
      option.textContent = "telemetry fallback";
      selectElement.appendChild(option);
      return;
    }
    fieldState.sessions.forEach((session, index) => {
      const option = document.createElement("option");
      option.value = session.session_hash;
      const age = Math.max(0, Date.now() / 1000 - Number(session.updated_at_epoch || 0));
      option.textContent = `${session.session_hash} · ${index ? `${Math.round(age / 60)}m` : "latest"}`;
      option.selected = session.session_hash === fieldState.sessionHash;
      selectElement.appendChild(option);
    });
  }

  async function loadField(sessionHash = "") {
    const request = ++sessionRequest;
    const queryString = sessionHash
      ? `?session=${encodeURIComponent(sessionHash)}`
      : "";
    try {
      const response = await fetch(`/api/cortex/field${queryString}`, {
        headers: { Accept: "application/json" },
        cache: "no-store",
      });
      if (!response.ok) throw new Error(`field API returned HTTP ${response.status}`);
      const projection = await response.json();
      if (request !== sessionRequest) return;
      window.CortexField.applyProjection(fieldState, projection);
      syncFieldNodes();
      updateSessionControl();
      setEventStatus(fieldState.status === "online");
      renderPanel();
      stateDirty = true;
      connectEvents(fieldState.sessionHash);
    } catch (error) {
      fieldState.status = "fault";
      fieldState.fault = error.message || String(error);
      fieldState.stale = true;
      setEventStatus(false);
      renderPanel();
    }
  }

  function connectEvents(sessionHash = fieldState.sessionHash) {
    const generation = ++eventSocketGeneration;
    if (eventSocket) eventSocket.close();
    const protocol = window.location.protocol === "https:" ? "wss:" : "ws:";
    const queryString = sessionHash
      ? `?session=${encodeURIComponent(sessionHash)}`
      : "";
    eventSocket = new WebSocket(
      `${protocol}//${window.location.host}/api/cortex/events${queryString}`,
    );
    eventSocket.addEventListener("open", () => {
      if (generation !== eventSocketGeneration) return;
      window.CortexField.setConnection(fieldState, "online");
      setEventStatus(fieldState.status === "online");
    });
    eventSocket.addEventListener("message", (message) => {
      if (generation !== eventSocketGeneration) return;
      let payload;
      try {
        payload = JSON.parse(message.data);
      } catch (_error) {
        return;
      }
      if (payload.type !== "events" || !Array.isArray(payload.events)) return;
      if (!fieldState.sessionHash) {
        const event = payload.events.at(-1);
        flashTicker(
          `TELEMETRY FALLBACK · ${event?.label || event?.kind || "activity"} · no synthetic firing`,
        );
        return;
      }
      const accepted = window.CortexField.applyEvents(fieldState, payload.events);
      if (fieldState.seqGap) {
        setEventStatus(false);
        renderPanel();
        window.setTimeout(() => loadField(fieldState.sessionHash), 250);
        return;
      }
      syncFieldNodes();
      if (liveEventsEnabled) accepted.forEach(visualizeFieldEvent);
      if (accepted.length) {
        fieldState.status = "online";
        fieldState.stale = false;
        setEventStatus(true);
        renderPanel();
        stateDirty = true;
      }
    });
    eventSocket.addEventListener("close", () => {
      if (generation !== eventSocketGeneration) return;
      window.CortexField.setConnection(fieldState, "offline");
      setEventStatus(false);
      window.clearTimeout(eventReconnect);
      eventReconnect = window.setTimeout(
        () => connectEvents(fieldState.sessionHash),
        2000,
      );
    });
    eventSocket.addEventListener("error", () => {
      if (generation !== eventSocketGeneration) return;
      setEventStatus(false);
    });
  }

  function showBootError(error) {
    const message = document.createElement("div");
    message.className = "bootError";
    message.innerHTML = `<b>CORTEX LINK FAILED</b>${escapeHtml(error.message || error)}`;
    stage.appendChild(message);
    setEventStatus(false);
  }

  async function boot() {
    try {
      const response = await fetch("/api/cortex/graph", {
        headers: { Accept: "application/json" },
      });
      if (!response.ok) {
        throw new Error(`graph API returned HTTP ${response.status}`);
      }
      const graphData = await response.json();
      if (!Array.isArray(graphData.nodes) || !Array.isArray(graphData.links)) {
        throw new Error("graph API returned an invalid payload");
      }
      initializeGraph(graphData);
      resize();
      new ResizeObserver(resize).observe(stage);
      buildTree();
      renderPanel();
      bindCanvasInteractions();
      bindInterface();
      initializeHeader();
      recomputeState();
      const settleTicks = nodeCount > 1500 ? 70 : 180;
      for (let index = 0; index < settleTicks; index += 1) tick();
      fitView();
      camera.distance = cameraTarget.distance;
      cameraTarget = null;
      previousTime = performance.now();
      requestAnimationFrame(frame);
      await loadField();
    } catch (error) {
      showBootError(error);
    }
  }

  boot();
})();
