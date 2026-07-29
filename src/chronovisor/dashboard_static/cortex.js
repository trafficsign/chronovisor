"use strict";

(() => {
  const STEEL = "#7d92b5";
  const FIRE = "#ffb454";
  const FIRE_HOT = "#fff3dd";
  const RGB_STEEL = hexRgb(STEEL);
  const RGB_FIRE = hexRgb(FIRE);
  const RGB_HOT = hexRgb(FIRE_HOT);
  const TYPE_OFF = new Set([2]);

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
  let autoStimulate = true;
  let autoRotate = true;
  let soundOn = false;
  let edgeVisibility = 1.6;
  let alpha = 1;
  let spikes = 0;
  let lastInteraction = 0;

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

  const pulses = [];
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
      potential: 0,
      refractoryUntil: 0,
      firedAt: -1e9,
      screenX: 0,
      screenY: 0,
      screenScale: 0,
      viewDepth: 1e9,
    }));
    links = data.links.map((row) => ({
      source: row[0],
      target: row[1],
      kind: row[2] || 0,
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

  function fire(index, depth, budget, micro = false) {
    const node = nodes[index];
    const now = performance.now();
    if (!node || nodeState[index] === 0 || now < node.refractoryUntil) return;
    node.refractoryUntil = now + 360 + Math.random() * 340;
    node.potential = 1;
    node.firedAt = now;
    spikes += 1;
    if (!(spikes & 3)) {
      document.getElementById("stSp").textContent = spikes.toLocaleString();
    }
    crackle(micro ? 0.03 : 0.1 * Math.min(1, node.radius / 7) + 0.04);
    if (micro || depth >= 8 || budget.remaining <= 0) return;

    const candidates = outgoing[index].filter(
      (edgeIndex) => edgeState[edgeIndex] > 0,
    );
    for (let index = candidates.length - 1; index > 0; index -= 1) {
      const swapIndex = (Math.random() * (index + 1)) | 0;
      [candidates[index], candidates[swapIndex]] = [
        candidates[swapIndex],
        candidates[index],
      ];
    }
    const probability = Math.pow(0.86, depth);
    let branches = 0;
    for (const edgeIndex of candidates) {
      if (branches >= 4 || budget.remaining <= 0) break;
      if (Math.random() > probability) continue;
      const link = links[edgeIndex];
      const target = nodes[link.target];
      if (!target || performance.now() < target.refractoryUntil) continue;
      budget.remaining -= 1;
      branches += 1;
      const distance = Math.hypot(
        target.x - node.x,
        target.y - node.y,
        target.z - node.z,
      );
      pulses.push({
        edgeIndex,
        startedAt: now,
        duration: Math.max(150, Math.min(560, distance * 2.1)),
        depth: depth + 1,
        budget,
      });
    }
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
    const budget = { remaining: 150 };
    scenarioNodes(name).forEach((node) => fire(node.index, 0, budget));
    flashTicker(label || `⚡ ${name.toUpperCase()} stimulus → cascade`);
  }

  function firePageIds(pageIds, kind, label) {
    const budget = { remaining: 180 };
    let fired = 0;
    pageIds.forEach((pageId) => {
      const node = byId.get(pageId);
      if (!node || nodeState[node.index] === 0) return;
      fire(node.index, 0, budget);
      fired += 1;
    });
    if (!fired) {
      stimulate(kind, label);
    } else {
      flashTicker(`⚡ ${label} · ${fired} neuron${fired === 1 ? "" : "s"}`);
    }
  }

  let ambientAccumulator = 0;
  function ambient(delta) {
    ambientAccumulator += delta;
    const interval = nodeCount > 1200 ? 230 : 150;
    while (ambientAccumulator > interval) {
      ambientAccumulator -= interval;
      const attempts = nodeCount > 1200 ? 1 : 2;
      for (let attempt = 0; attempt < attempts; attempt += 1) {
        if (Math.random() > 0.75) continue;
        const index = (Math.random() * nodeCount) | 0;
        if (nodeState[index] === 0) continue;
        const node = nodes[index];
        const now = performance.now();
        if (now < node.refractoryUntil) continue;
        if (Math.random() < 0.16) {
          fire(index, 6, { remaining: 6 });
        } else {
          node.refractoryUntil = now + 300;
          node.potential = Math.max(node.potential, 0.5 + Math.random() * 0.25);
          node.firedAt = now;
          spikes += 1;
          if (!(spikes & 7)) {
            document.getElementById("stSp").textContent = spikes.toLocaleString();
          }
          crackle(0.022);
        }
      }
    }
  }

  let autoAccumulator = 0;
  let autoIndex = 0;
  const autoSequence = ["recall", "save", "recall", "ingest"];
  function autoTick(delta) {
    if (!autoStimulate) return;
    autoAccumulator += delta;
    if (autoAccumulator > 6400) {
      autoAccumulator = 0;
      stimulate(autoSequence[autoIndex % autoSequence.length]);
      autoIndex += 1;
    }
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

  function drawPulses(time) {
    context.globalCompositeOperation = "lighter";
    for (let index = pulses.length - 1; index >= 0; index -= 1) {
      const pulse = pulses[index];
      const link = links[pulse.edgeIndex];
      const source = nodes[link.source];
      const target = nodes[link.target];
      const progress = (time - pulse.startedAt) / pulse.duration;
      if (progress >= 1) {
        pulses.splice(index, 1);
        fire(link.target, pulse.depth, pulse.budget);
        continue;
      }
      if (source.viewDepth > 9e8 || target.viewDepth > 9e8) continue;
      const tailProgress = Math.max(0, progress - 0.09);
      const head = projectPoint(
        source.x + (target.x - source.x) * progress,
        source.y + (target.y - source.y) * progress,
        source.z + (target.z - source.z) * progress,
      );
      const tail = projectPoint(
        source.x + (target.x - source.x) * tailProgress,
        source.y + (target.y - source.y) * tailProgress,
        source.z + (target.z - source.z) * tailProgress,
      );
      if (!head) continue;
      const amplitude = Math.sin(Math.PI * progress) * fog(head.viewDepth);
      if (tail) {
        context.strokeStyle = rgba(RGB_FIRE, 0.55 * amplitude);
        context.lineWidth = 1.4;
        context.beginPath();
        context.moveTo(tail.x, tail.y);
        context.lineTo(head.x, head.y);
        context.stroke();
      }
      const glowSize = (10 + 8 * amplitude) * head.scale * 1.6;
      context.globalAlpha = 0.85 * amplitude;
      context.drawImage(
        glowFire,
        head.x - glowSize / 2,
        head.y - glowSize / 2,
        glowSize,
        glowSize,
      );
      context.globalAlpha = 1;
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
      const radius = Math.max(0.75, node.radius * node.screenScale);
      const dim = state === 1 ? 0.28 : 1;
      const potential = node.potential;
      if (potential > 0.03) {
        const glowSize = radius * (4 + 10 * potential);
        context.globalAlpha = Math.min(1, potential * 1.15) * depthFade * dim;
        context.drawImage(
          potential > 0.6 ? glowHot : glowFire,
          node.screenX - glowSize / 2,
          node.screenY - glowSize / 2,
          glowSize,
          glowSize,
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

      let color = node.base;
      if (potential > 0.5) {
        color = mix(RGB_FIRE, RGB_HOT, (potential - 0.5) * 2);
      } else if (potential > 0.04) {
        color = mix(node.base, RGB_FIRE, potential * 2);
      }
      const coreOpacity =
        (state === 1
          ? 0.22
          : state === 3
            ? 1
            : 0.62 + 0.38 * Math.min(1, potential * 3))
        * depthFade;
      context.globalCompositeOperation =
        potential > 0.04 ? "lighter" : "source-over";
      context.fillStyle = rgba(color, coreOpacity);
      context.beginPath();
      context.arc(node.screenX, node.screenY, radius, 0, Math.PI * 2);
      context.fill();
      if (potential > 0.35) {
        context.fillStyle = rgba(RGB_HOT, potential * depthFade);
        context.beginPath();
        context.arc(node.screenX, node.screenY, radius * 0.45, 0, Math.PI * 2);
        context.fill();
      }
      context.globalCompositeOperation = "lighter";
      const age = time - node.firedAt;
      if (age < 450 && potential > 0.1) {
        const progress = age / 450;
        context.strokeStyle = rgba(RGB_FIRE, (1 - progress) * 0.55 * depthFade);
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
        context.strokeStyle = rgba(RGB_FIRE, 0.95);
        context.lineWidth = 1.6;
        context.beginPath();
        context.arc(
          node.screenX,
          node.screenY,
          radius + 5 + Math.sin(time * 0.005) * 1.5,
          0,
          Math.PI * 2,
        );
        context.stroke();
      }
    });
  }

  function drawLabels() {
    context.font = `10.5px ${getComputedStyle(document.body).getPropertyValue("--mono")}`;
    context.textAlign = "center";
    const occupied = [];
    const candidates = nodes
      .filter((node) => {
        const state = nodeState[node.index];
        const hot = node.potential > 0.5;
        return (
          state >= 2
          && node.viewDepth <= 9e8
          && (state === 3
            || hot
            || (labelHubs.has(node.index)
              && camera.distance < 1500
              && node.viewDepth < camera.distance))
        );
      })
      .sort((left, right) => {
        const leftPriority =
          (left.index === selected ? 100000 : 0)
          + left.potential * 1000
          + left.fanIn;
        const rightPriority =
          (right.index === selected ? 100000 : 0)
          + right.potential * 1000
          + right.fanIn;
        return rightPriority - leftPriority;
      });
    candidates.forEach((node) => {
      const state = nodeState[node.index];
      const hot = node.potential > 0.5;
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
      context.fillStyle = `rgba(3,5,10,${0.7 * depthFade})`;
      context.fillRect(node.screenX - labelWidth / 2 - 3, y - 9, labelWidth + 6, 12);
      context.fillStyle =
        hot || state === 3
          ? rgba(RGB_FIRE, Math.max(0.5, node.potential) * depthFade + 0.2)
          : rgba([160, 178, 210], 0.8 * depthFade);
      context.fillText(node.name, node.screenX, y);
    });
  }

  function draw(time) {
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
    drawPulses(time);
    drawNodes(time);
    drawLabels();
  }

  let previousTime = performance.now();
  let simulationAccumulator = 0;
  function frame(now) {
    const delta = Math.min(60, now - previousTime);
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
    ambient(delta);
    autoTick(delta);
    const decay = Math.exp(-delta / 430);
    nodes.forEach((node) => {
      if (node.potential > 0.003) node.potential *= decay;
    });
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
        if (index >= 0) fire(index, 0, { remaining: 90 });
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
    return `
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
        <div class="ghost">クリックしたニューロンも発火し、wikilink沿いにスパイクが伝播します。</div>
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
    const autoToggle = document.getElementById("tAuto");
    autoToggle.addEventListener("click", () => {
      autoStimulate = !autoStimulate;
      autoToggle.classList.toggle("on", autoStimulate);
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
  let facts = [];
  const tickerText = document.getElementById("tickerTx");
  function nextFact() {
    tickerText.style.opacity = 0;
    window.setTimeout(() => {
      tickerText.textContent = facts[factIndex % facts.length];
      factIndex += 1;
      tickerText.style.opacity = 1;
    }, 350);
  }

  function flashTicker(message) {
    tickerText.style.opacity = 0;
    window.setTimeout(() => {
      tickerText.textContent = message;
      tickerText.style.opacity = 1;
    }, 200);
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
      "記憶皮質オンライン — wikilink沿いにスパイク伝播中",
      `${topHub?.id || "—"} fan-in ${topHub?.fanIn || 0} — 最深部のハブニューロン`,
      `${nodeCount.toLocaleString()} neurons / ${((data.meta.static || 0) + (data.meta.deferred || 0)).toLocaleString()} synapses`,
      "⚡ボタンで recall / save / ingest 経路を刺激",
      "ニューロンをクリックすると発火して連鎖します",
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
    badge.classList.toggle("offline", !online);
    live.classList.toggle("offline", !online);
    live.textContent = online ? "● LIVE" : "○ LINKING";
  }

  function connectEvents() {
    if (eventSocket) eventSocket.close();
    const protocol = window.location.protocol === "https:" ? "wss:" : "ws:";
    eventSocket = new WebSocket(`${protocol}//${window.location.host}/api/cortex/events`);
    eventSocket.addEventListener("open", () => {
      setEventStatus(true);
    });
    eventSocket.addEventListener("message", (message) => {
      let payload;
      try {
        payload = JSON.parse(message.data);
      } catch (_error) {
        return;
      }
      if (payload.type !== "events" || !Array.isArray(payload.events)) return;
      payload.events.forEach((event) => {
        const kind = ["recall", "save", "ingest"].includes(event.kind)
          ? event.kind
          : "recall";
        firePageIds(
          Array.isArray(event.page_ids) ? event.page_ids : [],
          kind,
          event.label || kind.toUpperCase(),
        );
      });
    });
    eventSocket.addEventListener("close", () => {
      setEventStatus(false);
      window.clearTimeout(eventReconnect);
      eventReconnect = window.setTimeout(connectEvents, 2000);
    });
    eventSocket.addEventListener("error", () => {
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
      connectEvents();
      window.setTimeout(() => stimulate("recall"), 1400);
    } catch (error) {
      showBootError(error);
    }
  }

  boot();
})();
