"use strict";

(() => {

  const Runtime = window.CortexRuntime;
  const STEEL = "#7d92b5";
  const FIRE = "#ffb454";
  const FIRE_HOT = "#fff3dd";
  const ELECTRIC = "#ffd84d";
  const VIOLET = "#9b7cff";

  const CAPTURE = "#4fe4ff";

  const RGB_STEEL = hexRgb(STEEL);
  const RGB_FIRE = hexRgb(FIRE);
  const RGB_HOT = hexRgb(FIRE_HOT);
  const RGB_ELECTRIC = hexRgb(ELECTRIC);
  const RGB_VIOLET = hexRgb(VIOLET);

  const RGB_CAPTURE = hexRgb(CAPTURE);

  const TYPE_OFF = new Set([2]);

  const SIMULATION_STEP_MS = 15;
  const SIMULATION_ALPHA_FLOOR = 0.012;
  const SIMULATION_SLEEP_VELOCITY = 0.02;
  const SIMULATION_SLEEP_TICKS = 36;
  const SIMULATION_MAX_STEPS_PER_FRAME = 4;

  const FRAME_DURATION_CAPACITY = 240;
  const FRAME_CADENCE_IDLE_GAP_MS = 250;

  const CAPTURE_WAVE_NODE_LIMIT = 112;
  const CAPTURE_WAVE_MAX_VISIBLE = 4;
  const CAPTURE_WAVE_MIN_INTERVAL_MS = 420;
  const CAPTURE_WAVE_TAIL_RATIO = 0.42;
  const CAPTURE_WAVE_FRONT_GLOW_OPACITY = 1;
  const CAPTURE_WAVE_FRONT_GLOW_PADDING_PX = 5.5;
  const CAPTURE_WAVE_TAIL_GLOW_OPACITY = 0.66;
  const CAPTURE_WAVE_TAIL_GLOW_PADDING_PX = 3.6;
  const CAPTURE_HEARTBEAT_SETTLE_MS = 220;
  const CAPTURE_HEARTBEAT_DURATION_MS = 720;
  const EXPLORER_CHUNK_SIZE = 120;
  const AUTO_ROTATE_COOLDOWN_MS = 2600;
  const SPHERE_TARGET_FORCE = 0.018;
  const SPHERE_RADIAL_FORCE = 0.006;
  const SPHERE_LINK_SPRING = 0.00006;
  const SPHERE_NORMAL_EDGE_SCALE = 0.28;
  const SPHERE_BACK_HEMISPHERE_FADE = 0.72;
  const SPHERE_FIT_PADDING_PX = 28;

  const NODE_CORE_SCALE = 1;
  const NODE_GLOW_MAX_PADDING_PX = 4;
  const NODE_EFFECT_MAX_PADDING_PX = 3;

  const VIEW_PREFERENCES_KEY = "chronovisor.cortex.preferences.v1";
  const VIEW_PREFERENCES_DEFAULTS = Object.freeze({
    mode: "organic",
    motion: true,
    rotate: true,
    sound: false,
    synapseVisibility: 160,
  });
  const SYNAPSE_VISIBILITY_MIN = 30;
  const SYNAPSE_VISIBILITY_MAX = 420;

  let data;
  let nodes = [];
  let links = [];
  let simulationLinks = [];
  let nodeCount = 0;
  let neighbors = [];

  let packageList = [];
  let packageShade = {};
  let anchors = {};
  let sphereTargets = [];
  let sphereQuality = Runtime.measureSphereQuality([], []);
  let nodeState;
  let edgeState;
  let drawOrder = [];
  let labelHubs = new Set();
  let labelCandidateMarks = new Uint32Array(0);
  let labelCandidateGeneration = 0;
  const labelCandidates = [];
  const occupiedLabels = [];

  let captureWaveDistances = new Float32Array(0);

  let selected = -1;
  let hovered = -1;
  let query = "";
  let matches = new Set();
  let stateDirty = true;
  const packageOff = new Set();
  let mode = VIEW_PREFERENCES_DEFAULTS.mode;
  let liveEventsEnabled = true;

  let motionEnabled = VIEW_PREFERENCES_DEFAULTS.motion;
  let autoRotate = VIEW_PREFERENCES_DEFAULTS.rotate;
  let soundOn = VIEW_PREFERENCES_DEFAULTS.sound;

  let edgeVisibility = VIEW_PREFERENCES_DEFAULTS.synapseVisibility / 100;
  let alpha = 1;
  let simulationAwake = true;
  let simulationSettledTicks = 0;
  let simulationTicks = 0;
  let simulationSleepCount = 0;
  let simulationLastMaxVelocity = 0;
  let modeFitTimer = 0;

  let lastInteraction = 0;

  let frameDurationCursor = 0;
  let frameDurationCount = 0;
  let frameCadenceCursor = 0;
  let frameCadenceCount = 0;
  const stage = document.getElementById("stage");
  const baseCanvas = document.getElementById("gl");
  const canvas = document.getElementById("overlay");
  const modeBar = document.getElementById("modeBar");
  const context = canvas.getContext("2d");
  const spatialIndex = Runtime.createSpatialIndex(48);
  const stageMetrics = Runtime.createStageMetrics();
  let renderScheduler = null;
  let pendingPointer = null;
  let panelRenderPending = false;

  const longFrameMetrics = {
    longAnimationFrames: 0,
    longAnimationFrameMax: 0,
    longTasks: 0,
    longTaskMax: 0,
  };
  function invalidate(reason) {
    if (renderScheduler) renderScheduler.invalidate(reason);
  }
  const baseRenderer = window.CortexWebGL.createRenderer(baseCanvas, {
    onStateChange: () => invalidate("renderer-state"),
  });
  const autoRotateWake = Runtime.createCooldownWake({
    now: () => performance.now(),
    setTimer: (callback, delay) => window.setTimeout(callback, delay),
    clearTimer: (handle) => window.clearTimeout(handle),
    readyAt: () => lastInteraction + AUTO_ROTATE_COOLDOWN_MS,
    isEnabled: () => (
      autoRotate
      && motionEnabled
      && !reducedMotion.matches
      && !dragging
    ),
    isHidden: () => document.hidden,
    onReady: () => invalidate("auto-rotate-wake"),
  });
  let width = 0;
  let height = 0;
  const pixelRatio = Math.min(2, window.devicePixelRatio || 1);
  const camera = {
    theta: 0.6,
    phi: 0.18,
    distance: 1650,
    pivotX: 0,
    pivotY: 0,
    pivotZ: 0,
    pivotNodeIndex: -1,
  };
  let cameraTarget = null;
  let cameraPivotTarget = null;
  let focalLength = 900;
  let projectionCosTheta = 1;
  let projectionSinTheta = 0;
  let projectionCosPhi = 1;
  let projectionSinPhi = 0;
  let projectionCenterX = 0;
  let projectionCenterY = 0;
  let projectionTopInset = 0;
  let labelFont = "10.5px monospace";
  let dragging = false;
  let downPoint = null;
  let moved = false;
  let audioContext = null;
  let lastSound = 0;
  let eventSocket = null;
  let eventReconnect = null;
  let eventSocketGeneration = 0;

  const reducedMotion = window.matchMedia("(prefers-reduced-motion: reduce)");

  const cortexMetrics = {
    frameDurations: new Float32Array(FRAME_DURATION_CAPACITY),
    frameCadences: new Float32Array(FRAME_DURATION_CAPACITY),
    labelsPainted: 0,
    maxCoreScale: 0,
    maxGlowPadding: 0,
    transportReceived: 0,
    transportPainted: 0,
    captureEvents: 0,
    captureEffectCount: 0,
    captureEffectSeq: -1,
    captureEffectCaptureId: "",
    captureWaveEffects: 0,
    captureWaveNodes: 0,
    captureAfterglowNodes: 0,
    captureWavePeak: 0,
    captureHeartbeatPeak: 0,
    captureCenterX: 0,
    captureCenterY: 0,
    captureRadius: 0,
    captureBandWidth: 0,
    captureProgress: 0,
    capturePhase: "idle",
    captureHeartbeatCount: 0,
    captureHeartbeatSeq: -1,
    transportElectricPeak: 0,
  };

  window.chronovisorCortexMetrics = () => ({
    mode, nodeCount, linkCount: links.length,
    sphere: sphereQuality,
    renderer: baseRenderer.snapshot(),
    scheduler: renderScheduler ? renderScheduler.state() : {},
    simulation: { awake: simulationAwake, ticks: simulationTicks },
    visual: { ...cortexMetrics },
  });

  const performanceObservers = [];
  function observeLongFrames() {
    if (!("PerformanceObserver" in window)) return;
    const supported = window.PerformanceObserver.supportedEntryTypes || [];
    const observe = (type, counterKey, maxKey) => {
      if (!supported.includes(type)) return;
      try {
        const observer = new PerformanceObserver((list) => {
          list.getEntries().forEach((entry) => {
            const duration = Number(entry.duration) || 0;
            longFrameMetrics[counterKey] += 1;
            longFrameMetrics[maxKey] = Math.max(
              longFrameMetrics[maxKey],
              duration,
            );
          });
        });
        observer.observe({ type, buffered: true });
        performanceObservers.push(observer);
      } catch (_error) {
        // Performance observers are optional diagnostics.
      }
    };
    observe("long-animation-frame", "longAnimationFrames", "longAnimationFrameMax");
    observe("longtask", "longTasks", "longTaskMax");
  }

  const captureWaveTrain = [];
  const captureWaveSampleEffect = Object.freeze({
    captureId: "capture-wave-train",
    seq: 1,
  });
  const captureHeartbeatBurst = {
    finalSeq: -1,
    finalCaptureId: "",
    dueAt: 0,
    generation: 0,
    recorded: true,
    painted: false,
    wakeTimer: 0,
  };

  const stars = Array.from({ length: 130 }, () => ({
    x: Math.random(),
    y: Math.random(),
    radius: Math.random() * 0.9 + 0.3,
    phase: Math.random() * Math.PI * 2,
    speed: 0.3 + Math.random() * 1.1,
  }));

  const glowSteel = makeGlow(RGB_STEEL);

  const glowCapture = makeGlow(RGB_CAPTURE);

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

  function clamp(value, minimum = 0, maximum = 1) {
    return Math.max(minimum, Math.min(maximum, value));
  }

  function sanitizeViewPreferences(candidate) {
    const value = candidate && typeof candidate === "object" ? candidate : {};
    const rawVisibility = Number(value.synapseVisibility);
    return {
      mode: Runtime.normalizeLayoutMode(value.mode),
      motion:
        typeof value.motion === "boolean"
          ? value.motion
          : VIEW_PREFERENCES_DEFAULTS.motion,
      rotate:
        typeof value.rotate === "boolean"
          ? value.rotate
          : VIEW_PREFERENCES_DEFAULTS.rotate,
      sound:
        typeof value.sound === "boolean"
          ? value.sound
          : VIEW_PREFERENCES_DEFAULTS.sound,
      synapseVisibility: Number.isFinite(rawVisibility)
        ? Math.round(
          clamp(
            rawVisibility,
            SYNAPSE_VISIBILITY_MIN,
            SYNAPSE_VISIBILITY_MAX,
          ),
        )
        : VIEW_PREFERENCES_DEFAULTS.synapseVisibility,
    };
  }

  function loadViewPreferences() {
    try {
      const stored = window.localStorage.getItem(VIEW_PREFERENCES_KEY);
      return stored
        ? sanitizeViewPreferences(JSON.parse(stored))
        : { ...VIEW_PREFERENCES_DEFAULTS };
    } catch {
      return { ...VIEW_PREFERENCES_DEFAULTS };
    }
  }

  function currentViewPreferences() {
    return {
      mode,
      motion: motionEnabled,
      rotate: autoRotate,
      sound: soundOn,

      synapseVisibility: Math.round(edgeVisibility * 100),
    };
  }

  function saveViewPreferences() {
    try {
      window.localStorage.setItem(
        VIEW_PREFERENCES_KEY,
        JSON.stringify(currentViewPreferences()),
      );
    } catch {
      // Private browsing or a locked-down webview may reject local storage.
    }
  }

  function syncViewPreferenceControls() {
    const controls = {
      motion: document.getElementById("tMotion"),
      rotate: document.getElementById("tRot"),
      sound: document.getElementById("tSnd"),

    };
    for (const [name, control] of Object.entries(controls)) {
      const enabled = {
        motion: motionEnabled,
        rotate: autoRotate,
        sound: soundOn,

      }[
        name
      ];
      control.classList.toggle("on", enabled);
      control.setAttribute("aria-pressed", String(enabled));
    }
    [
      ["mOrganic", "organic"],
      ["mSphere", "sphere"],
      ["mCluster", "cluster"],
    ].forEach(([id, controlMode]) => {
      const control = document.getElementById(id);
      const active = mode === controlMode;
      control.classList.toggle("on", active);
      control.setAttribute("aria-pressed", String(active));
    });
    document.getElementById("visSlider").value = String(
      Math.round(edgeVisibility * 100),
    );

  }

  function applyViewPreferences(candidate) {
    const preferences = sanitizeViewPreferences(candidate);
    mode = preferences.mode;
    motionEnabled = preferences.motion;
    autoRotate = preferences.rotate;
    soundOn = preferences.sound;

    edgeVisibility = preferences.synapseVisibility / 100;
    syncViewPreferenceControls();
    stateDirty = true;
    invalidate("preferences");
  }

  function resetViewPreferences() {
    try {
      window.localStorage.removeItem(VIEW_PREFERENCES_KEY);
    } catch {
      // Keep the in-memory reset even when storage access is unavailable.
    }
    applyViewPreferences(VIEW_PREFERENCES_DEFAULTS);
    reheat(0.7);
    scheduleModeFit();
    flashTicker("view preferences reset");
  }

  function smoothstep(value) {
    const unit = clamp(value);
    return unit * unit * (3 - 2 * unit);
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

  function drawCompactGlow(sprite, x, y, radius, padding, opacity) {
    const boundedPadding = clamp(padding, 0, NODE_GLOW_MAX_PADDING_PX);
    const glowRadius = radius + boundedPadding;
    const glowSize = glowRadius * 2;
    context.globalAlpha = clamp(opacity);
    context.drawImage(
      sprite,
      x - glowRadius,
      y - glowRadius,
      glowSize,
      glowSize,
    );
    context.globalAlpha = 1;
    cortexMetrics.maxGlowPadding = Math.max(
      cortexMetrics.maxGlowPadding,
      boundedPadding,
    );
  }

  function updateProjectionCenter() {
    projectionCenterX = width / 2;
    projectionCenterY = height / 2;
    projectionTopInset = 0;
    if (!modeBar || width <= 0 || height <= 0) return;

    const stageBounds = stage.getBoundingClientRect();
    const modeBarBounds = modeBar.getBoundingClientRect();
    if (
      !Number.isFinite(stageBounds.top)
      || !Number.isFinite(stageBounds.height)
      || stageBounds.height <= 0
      || !Number.isFinite(modeBarBounds.bottom)
      || !Number.isFinite(modeBarBounds.height)
      || modeBarBounds.height <= 0
    ) {
      return;
    }

    const topInset =
      (modeBarBounds.bottom - stageBounds.top) * (height / stageBounds.height);
    if (!Number.isFinite(topInset) || topInset <= 0 || topInset >= height) return;
    projectionTopInset = topInset;
    projectionCenterY = (projectionTopInset + height) / 2;
  }

  function resize() {
    width = stage.clientWidth;
    height = stage.clientHeight;
    updateProjectionCenter();
    canvas.width = Math.max(1, Math.round(width * pixelRatio));
    canvas.height = Math.max(1, Math.round(height * pixelRatio));
    baseRenderer.resize(width, height, pixelRatio);
    labelFont = `10.5px ${getComputedStyle(document.body).getPropertyValue("--mono")}`;
    invalidate("resize");
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
      screenX: 0,
      screenY: 0,
      screenScale: 0,
      viewDepth: 1e9,
    }));
    links = data.links.map((row) => {
      const source = row[0];
      const target = row[1];
      return {
        source,
        target,
        kind: row[2] || 0,
        edgeType: row[3] || "wikilink",
        eventOnly: false,
        restLength:
          58 + (nodes[source]?.radius || 0) + (nodes[target]?.radius || 0),
      };
    });
    simulationLinks = links.filter((link) => link.kind !== 2);
    nodeCount = nodes.length;
    neighbors = Array.from({ length: nodeCount }, () => new Set());
    links.forEach((link, edgeIndex) => {
      if (!nodes[link.source] || !nodes[link.target]) return;
      neighbors[link.source].add(link.target);
      neighbors[link.target].add(link.source);
    });
    nodeState = new Uint8Array(nodeCount);
    edgeState = new Uint8Array(links.length);
    labelCandidateMarks = new Uint32Array(nodeCount);
    drawOrder = nodes.map((_node, index) => index);
    labelHubs = new Set(
      [...nodes]
        .sort((left, right) => right.fanIn - left.fanIn)
        .slice(0, 8)
        .map((node) => node.index),
    );
    buildAnchors();
    buildSphereTargets();
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

  function buildSphereTargets() {
    sphereTargets = Runtime.createSphereTargets(nodes);
    sphereTargets.forEach((target, index) => {
      nodes[index].sphereTarget = target;
    });
    sphereQuality = Runtime.measureSphereQuality(nodes, sphereTargets);
  }

  function seedPositions() {
    nodes.forEach((node) => {
      const anchor = anchors[node.packageName] || { x: 0, y: 0, z: 0 };
      node.anchor = anchor;
      node.x = anchor.x * 0.42 + (deterministicUnit(node.id, 11) - 0.5) * 330;
      node.y = anchor.y * 0.42 + (deterministicUnit(node.id, 29) - 0.5) * 280;
      node.z = anchor.z * 0.42 + (deterministicUnit(node.id, 47) - 0.5) * 330;
    });
  }

  function tick() {
    const sphereMode = mode === "sphere";
    const centerForce = sphereMode ? 0 : alpha * 0.00055;
    const anchorForce = sphereMode
      ? 0
      : alpha * (mode === "cluster" ? 0.015 : 0.0007);
    for (let index = 0; index < nodes.length; index += 1) {
      const node = nodes[index];
      const anchor = node.anchor;
      if (sphereMode) {
        const target = node.sphereTarget;
        if (!target) continue;
        const targetForce = alpha * SPHERE_TARGET_FORCE;
        node.vx += (target.x - node.x) * targetForce;
        node.vy += (target.y - node.y) * targetForce;
        node.vz += (target.z - node.z) * targetForce;
        const radius = Math.hypot(node.x, node.y, node.z);
        if (radius > 0.001) {
          const radialForce =
            (target.radius - radius) * alpha * SPHERE_RADIAL_FORCE;
          node.vx += (node.x / radius) * radialForce;
          node.vy += (node.y / radius) * radialForce;
          node.vz += (node.z / radius) * radialForce;
        }
        continue;
      }
      node.vx -= node.x * centerForce * 0.8;
      node.vy -= node.y * centerForce * 1.25;
      node.vz -= node.z * centerForce;
      node.vx += (anchor.x - node.x) * anchorForce;
      node.vy += (anchor.y - node.y) * anchorForce;
      node.vz += (anchor.z - node.z) * anchorForce;
    }

    const spring = sphereMode
      ? SPHERE_LINK_SPRING
      : mode === "cluster"
        ? 0.0012
        : 0.0007;
    for (let index = 0; index < simulationLinks.length; index += 1) {
      const link = simulationLinks[index];
      const source = nodes[link.source];
      const target = nodes[link.target];
      if (!source || !target) continue;
      const dx = target.x - source.x;
      const dy = target.y - source.y;
      const dz = target.z - source.z;
      const distance = Math.sqrt(dx * dx + dy * dy + dz * dz) || 1;
      const force = (distance - link.restLength) * spring * alpha;
      const fx = (dx / distance) * force;
      const fy = (dy / distance) * force;
      const fz = (dz / distance) * force;
      source.vx += fx;
      source.vy += fy;
      source.vz += fz;
      target.vx -= fx;
      target.vy -= fy;
      target.vz -= fz;
    }

    let maxVelocitySquared = 0;
    for (let index = 0; index < nodes.length; index += 1) {
      const node = nodes[index];
      const damping = sphereMode ? 0.78 : 0.82;
      node.vx *= damping;
      node.vy *= damping;
      node.vz *= damping;
      let velocitySquared =
        node.vx * node.vx + node.vy * node.vy + node.vz * node.vz;
      if (velocitySquared > 169) {
        const limit = 13 / Math.sqrt(velocitySquared);
        node.vx *= limit;
        node.vy *= limit;
        node.vz *= limit;
        velocitySquared = 169;
      }
      node.x += node.vx;
      node.y += node.vy;
      node.z += node.vz;
      maxVelocitySquared = Math.max(maxVelocitySquared, velocitySquared);
    }
    alpha = Math.max(SIMULATION_ALPHA_FLOOR, alpha * 0.986);
    simulationTicks += 1;
    simulationLastMaxVelocity = Math.sqrt(maxVelocitySquared);
    if (
      alpha <= SIMULATION_ALPHA_FLOOR
      && simulationLastMaxVelocity <= SIMULATION_SLEEP_VELOCITY
    ) {
      simulationSettledTicks += 1;
      if (simulationSettledTicks >= SIMULATION_SLEEP_TICKS) {
        sleepSimulation();
      }
    } else {
      simulationSettledTicks = 0;
    }
  }

  function sleepSimulation() {
    if (!simulationAwake) return;
    simulationAwake = false;
    simulationSleepCount += 1;
    simulationAccumulator = 0;
    alpha = 0;
    for (let index = 0; index < nodes.length; index += 1) {
      nodes[index].vx = 0;
      nodes[index].vy = 0;
      nodes[index].vz = 0;
    }
  }

  function reheat(value) {
    alpha = Math.max(alpha, value);
    simulationAwake = true;
    simulationSettledTicks = 0;
    simulationAccumulator = 0;
    invalidate("simulation");
  }

  function projectionFocalLength() {
    return Math.min(width, height) * 1.12;
  }

  function projectAll() {
    focalLength = projectionFocalLength();
    projectionCosTheta = Math.cos(camera.theta);
    projectionSinTheta = Math.sin(camera.theta);
    projectionCosPhi = Math.cos(camera.phi);
    projectionSinPhi = Math.sin(camera.phi);
    for (let index = 0; index < nodes.length; index += 1) {
      const node = nodes[index];
      const localX = node.x - camera.pivotX;
      const localY = node.y - camera.pivotY;
      const localZ = node.z - camera.pivotZ;
      const rotatedX = localX * projectionCosTheta + localZ * projectionSinTheta;
      const rotatedZ = -localX * projectionSinTheta + localZ * projectionCosTheta;
      const rotatedY = localY * projectionCosPhi - rotatedZ * projectionSinPhi;
      const depthZ = localY * projectionSinPhi + rotatedZ * projectionCosPhi;
      const viewDepth = camera.distance - depthZ;
      if (viewDepth < 60) {
        node.viewDepth = 1e9;
        continue;
      }
      const scale = focalLength / viewDepth;
      node.screenX = rotatedX * scale + projectionCenterX;
      node.screenY = rotatedY * scale + projectionCenterY;
      node.screenScale = scale;
      node.viewDepth = viewDepth;
    }
    spatialIndex.rebuild(
      nodes,
      (node, index) => nodeState[index] > 0 && node.viewDepth <= 9e8,
      (node) => Math.max(5, node.radius * node.screenScale),
    );
  }

  function fog(viewDepth) {
    const amount = (viewDepth - camera.distance * 0.55) / (camera.distance * 1.1);
    const baseFade = 1 - Math.max(0, Math.min(0.72, amount));
    if (mode !== "sphere") return baseFade;
    const shellRadius = Math.max(1, sphereQuality.targetRadius.max || 540);
    const backDepth = clamp(
      (viewDepth - camera.distance) / shellRadius,
    );
    return baseFade * (1 - backDepth * SPHERE_BACK_HEMISPHERE_FADE);
  }

  function graphRadius() {
    return graphRadiusFrom({ x: 0, y: 0, z: 0 });
  }

  function graphRadiusFrom(center) {
    let maximum = 0;
    nodes.forEach((node) => {
      maximum = Math.max(
        maximum,
        (node.x - center.x) ** 2
          + (node.y - center.y) ** 2
          + (node.z - center.z) ** 2,
      );
    });
    return Math.sqrt(maximum);
  }

  function graphCenter() {
    if (!nodes.length) return { x: 0, y: 0, z: 0 };
    const center = nodes.reduce(
      (total, node) => ({
        x: total.x + node.x,
        y: total.y + node.y,
        z: total.z + node.z,
      }),
      { x: 0, y: 0, z: 0 },
    );
    return {
      x: center.x / nodes.length,
      y: center.y / nodes.length,
      z: center.z / nodes.length,
    };
  }

  function syncCameraPivotControl() {
    const control = document.getElementById("resetCenter");
    if (!control) return;
    control.classList.toggle("on", camera.pivotNodeIndex >= 0);
    control.dataset.pivoted = String(camera.pivotNodeIndex >= 0);
  }

  function setCameraPivot(point, nodeIndex = -1) {
    cameraPivotTarget = {
      x: Number(point.x || 0),
      y: Number(point.y || 0),
      z: Number(point.z || 0),
    };
    camera.pivotNodeIndex = nodeIndex;
    syncCameraPivotControl();
    lastInteraction = performance.now();
    invalidate("camera-pivot");
  }

  function setNodeAsCameraPivot(index) {
    const node = nodes[index];
    if (!node) return;
    setCameraPivot(node, index);
    flashTicker(`rotation center · ${node.id}`);
  }

  function resetCameraPivot(announce = true) {
    setCameraPivot(graphCenter());
    if (announce) flashTicker("rotation center · graph center");
  }

  function fitView() {
    const center = graphCenter();
    resetCameraPivot(false);
    const distance = mode === "sphere"
      ? Runtime.fitSphereCamera({
          sphereRadius: graphRadiusFrom(center),
          focalLength: projectionFocalLength(),
          viewportWidth: width,
          viewportHeight: height,
          topInset: projectionTopInset,
          padding: SPHERE_FIT_PADDING_PX,
          minimumDistance: 600,
        }).distance
      : Math.max(600, graphRadius() * 2.55);
    cameraTarget = {
      theta: camera.theta,
      phi: camera.phi,
      distance,
    };
  }

  function scheduleModeFit(expectedMode = mode) {
    window.clearTimeout(modeFitTimer);
    modeFitTimer = window.setTimeout(() => {
      modeFitTimer = 0;
      if (mode === expectedMode) {
        fitView();
        invalidate("mode-fit");
      }
    }, 900);
  }

  function angleLerp(from, to, factor) {
    const delta = ((to - from + Math.PI * 3) % (Math.PI * 2)) - Math.PI;
    return from + delta * factor;
  }

  function focusNode(index) {
    const node = nodes[index];
    if (!node) return;
    if (index === camera.pivotNodeIndex) return;
    const localX = node.x - camera.pivotX;
    const localY = node.y - camera.pivotY;
    const localZ = node.z - camera.pivotZ;
    const theta = Math.atan2(localX, localZ);
    const radial = Math.hypot(localX, localZ);
    const phi = Math.max(-1.3, Math.min(1.3, Math.atan2(localY, radial)));
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

  function unlockSound() {
    if (!soundOn) return;
    const AudioContextType = window.AudioContext || window.webkitAudioContext;
    if (!AudioContextType) return;
    if (!audioContext) audioContext = new AudioContextType();
    if (audioContext.state === "suspended") {
      audioContext.resume().catch(() => {});
    }
  }

  function stimulate() {
    visualizeTransportEvent({ kind: "save", mode: "demo", label: "DEMO SAVE", capture_id: "demo" });
  }

  function scheduleCaptureHeartbeatWake(generation, wakeAt) {
    captureHeartbeatBurst.wakeTimer = window.setTimeout(() => {
      if (generation !== captureHeartbeatBurst.generation) return;
      captureHeartbeatBurst.wakeTimer = 0;
      invalidate("capture-heartbeat");
    }, Math.max(0, wakeAt - performance.now()));
  }

  function armCaptureHeartbeat(finalSeq, dueAt, finalCaptureId) {
    captureHeartbeatBurst.generation += 1;
    const generation = captureHeartbeatBurst.generation;
    if (captureHeartbeatBurst.wakeTimer) {
      window.clearTimeout(captureHeartbeatBurst.wakeTimer);
    }
    captureHeartbeatBurst.finalSeq = finalSeq;
    captureHeartbeatBurst.finalCaptureId = finalCaptureId;
    captureHeartbeatBurst.dueAt = dueAt;
    captureHeartbeatBurst.recorded = false;
    captureHeartbeatBurst.painted = false;
    scheduleCaptureHeartbeatWake(generation, dueAt);
  }

  function pruneCaptureWaveTrain(now) {
    for (let index = captureWaveTrain.length - 1; index >= 0; index -= 1) {
      if (now >= captureWaveTrain[index].arrivalAt) {
        captureWaveTrain.splice(index, 1);
      }
    }
  }

  function admitCaptureWave(effect, eventMultiplicity, now) {
    pruneCaptureWaveTrain(now);
    const newest = captureWaveTrain.at(-1) || null;
    if (
      newest
      && (
        now - newest.startedAt < CAPTURE_WAVE_MIN_INTERVAL_MS
        || captureWaveTrain.length >= CAPTURE_WAVE_MAX_VISIBLE
      )
    ) {
      newest.coalescedCount += eventMultiplicity;
      newest.latestCaptureId = effect.captureId;
      armCaptureHeartbeat(
        effect.seq,
        Math.max(
          captureHeartbeatBurst.dueAt,
          now + CAPTURE_HEARTBEAT_SETTLE_MS,
        ),
        effect.captureId,
      );
      return false;
    }
    captureWaveTrain.push(effect);
    let latestArrivalAt = effect.arrivalAt;
    for (let index = 0; index < captureWaveTrain.length; index += 1) {
      latestArrivalAt = Math.max(
        latestArrivalAt,
        captureWaveTrain[index].arrivalAt,
      );
    }
    armCaptureHeartbeat(
      effect.seq,
      latestArrivalAt + CAPTURE_HEARTBEAT_SETTLE_MS,
      effect.captureId,
    );
    return true;
  }

  function visualizeTransportEvent(event) {
    if (!event || event.kind !== "save" || !liveEventsEnabled && event.mode !== "demo") return;
    const now = performance.now();
    const duration = event.mode === "demo" ? 3600 : 5000;
    cortexMetrics.transportReceived += 1;
    cortexMetrics.captureEvents += 1;
    const captureId = String(event.capture_id || "").slice(0, 12);
    admitCaptureWave({
      captureId, startedAt: now, duration,
      seq: cortexMetrics.transportReceived, paintedAt: 0,
      coalescedCount: 1, latestCaptureId: captureId,
      arrivalAt: now + duration * Runtime.CAPTURE_WAVE_TRAVEL_END,
    }, 1, now);
    document.getElementById("stSp").textContent = String(cortexMetrics.captureEvents);
    crackle(0.055);
    flashTicker(String(event.label || "CAPTURED").slice(0, 160));
    document.getElementById("captureAria").textContent = String(event.label || "Captured").slice(0, 160);
    invalidate("capture");
  }

  function drawEdges() {
    const sphereMode = mode === "sphere";
    const depthBandCount = sphereMode ? 4 : 3;
    const paths = Array.from(
      { length: depthBandCount * 3 },
      () => new Path2D(),
    );
    const pathKinds = new Uint8Array(depthBandCount * 3);
    const now = performance.now();
    links.forEach((link, edgeIndex) => {
      if (link.eventOnly && link.expiresAt <= now) return;
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
      const depthFade = fog(depth);
      const band = sphereMode
        ? Runtime.sphereFogBand(depthFade)
        : depth < camera.distance * 0.85
          ? 0
          : depth < camera.distance * 1.15
            ? 1
            : 2;
      const batch = (state - 1) * depthBandCount + band;
      paths[batch].moveTo(source.screenX, source.screenY);
      paths[batch].lineTo(target.screenX, target.screenY);
      pathKinds[batch] = 1;
    });

    const bandOpacity = [0.085, 0.05, 0.024];
    for (let state = 1; state <= 3; state += 1) {
      for (let band = depthBandCount - 1; band >= 0; band -= 1) {
        const batch = (state - 1) * depthBandCount + band;
        if (!pathKinds[batch]) continue;
        context.lineWidth =
          state === 3 ? 1.3 : 0.7 + Math.max(0, edgeVisibility - 1) * 0.28;
        let opacity = sphereMode
          ? (state === 3 ? 0.42 : state === 2 ? 0.075 : 0.025)
            * Runtime.sphereFogOpacity(band)
            * edgeVisibility
          : state === 3
            ? 0.42
            : state === 2
              ? bandOpacity[band] * edgeVisibility
              : bandOpacity[band] * 0.35 * edgeVisibility;
        if (sphereMode && state !== 3) {
          opacity *= SPHERE_NORMAL_EDGE_SCALE;
        }
        opacity = Math.min(0.85, opacity);
        context.strokeStyle =
          state === 3 ? rgba(RGB_FIRE, opacity) : rgba(RGB_STEEL, opacity);
        context.stroke(paths[batch]);
      }
    }
  }

  function drawNodes(time, overlayOnly = false) {
    if (!overlayOnly) {
      drawOrder.sort(
        (leftIndex, rightIndex) =>
          nodes[rightIndex].viewDepth - nodes[leftIndex].viewDepth,
      );
    }
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
      if (
        overlayOnly
        && state !== 3
        && index !== selected
        && index !== hovered
      ) return;
      const baseRadius = Math.max(0.75, node.radius * node.screenScale);
      const radius = baseRadius * NODE_CORE_SCALE;
      cortexMetrics.maxCoreScale = Math.max(
        cortexMetrics.maxCoreScale,
        radius / baseRadius,
      );
      if (state === 3 || (!overlayOnly && node.fanIn >= 38)) {
        drawCompactGlow(
          glowSteel,
          node.screenX,
          node.screenY,
          radius,
          state === 3 ? 2.6 : 1.4,
          (state === 3 ? 0.38 : 0.1) * depthFade * dim,
        );
      }

      const color = node.base;
      const coreOpacity = (state === 1 ? 0.22 : state === 3 ? 1 : 0.62) * depthFade;
      context.globalCompositeOperation = "source-over";
      if (!overlayOnly) {
        context.fillStyle = rgba(color, coreOpacity);
        context.beginPath();
        context.arc(node.screenX, node.screenY, radius, 0, Math.PI * 2);
        context.fill();
      }
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

  function drawCameraPivot(time) {
    const node = nodes[camera.pivotNodeIndex];
    if (
      !node
      || nodeState[node.index] === 0
      || node.viewDepth > 9e8
      || node.screenX < -60
      || node.screenX > width + 60
      || node.screenY < -60
      || node.screenY > height + 60
    ) return;
    const radius = Math.max(0.75, node.radius * node.screenScale) + 12;
    const pulse = 0.62 + Math.sin(time * 0.0045) * 0.16;
    context.save();
    context.strokeStyle = rgba(RGB_ELECTRIC, pulse);
    context.lineWidth = 1;
    context.setLineDash([2, 5]);
    context.beginPath();
    context.arc(node.screenX, node.screenY, radius, 0, Math.PI * 2);
    context.stroke();
    context.setLineDash([]);
    context.restore();
  }

  function visibleMemoryNode(node) {
    return (
      nodeState[node.index] > 0
      && node.viewDepth <= 9e8
      && node.screenX >= -40
      && node.screenX <= width + 40
      && node.screenY >= 40
      && node.screenY <= height - 32
    );
  }

  function memoryNodeWeight(node) {
    return (
      clamp(node.screenScale, 0.3, 2.5)
      * (1 + Math.log1p(node.fanIn + node.fanOut) * 0.08)
    );
  }

  function captureWaveGeometry() {
    if (captureWaveDistances.length !== nodes.length) {
      captureWaveDistances = new Float32Array(nodes.length);
    }
    let weightedX = 0;
    let weightedY = 0;
    let totalWeight = 0;
    nodes.forEach((node) => {
      if (!visibleMemoryNode(node)) return;
      const weight = memoryNodeWeight(node);
      weightedX += node.screenX * weight;
      weightedY += node.screenY * weight;
      totalWeight += weight;
    });
    const centerX = totalWeight ? weightedX / totalWeight : width * 0.5;
    const centerY = totalWeight ? weightedY / totalWeight : height * 0.52;
    let maxDistance = 0;
    nodes.forEach((node) => {
      if (!visibleMemoryNode(node)) {
        captureWaveDistances[node.index] = -1;
        return;
      }
      const distance = Math.hypot(
        node.screenX - centerX,
        node.screenY - centerY,
      );
      captureWaveDistances[node.index] = distance;
      maxDistance = Math.max(maxDistance, distance);
    });
    const shortest = Math.max(240, Math.min(width, height));
    return {
      x: centerX,
      y: centerY,
      radius: Math.max(maxDistance + 8, shortest * 0.22),
    };
  }

  function captureWaveNodeIntensity(
    distance,
    waveRadius,
    falloffWidth,
    wavePeak,
  ) {
    const radialOffset = distance - waveRadius;
    const reach = radialOffset > 0
      ? falloffWidth * CAPTURE_WAVE_TAIL_RATIO
      : falloffWidth;
    const proximity = 1 - clamp(Math.abs(radialOffset) / Math.max(1, reach));
    return smoothstep(proximity) * wavePeak;
  }

  function drawCaptureWaveNodeGlow(node, intensity, fade, inTail) {
    const padding = inTail
      ? CAPTURE_WAVE_TAIL_GLOW_PADDING_PX
      : CAPTURE_WAVE_FRONT_GLOW_PADDING_PX;
    const opacity = inTail
      ? CAPTURE_WAVE_TAIL_GLOW_OPACITY
      : CAPTURE_WAVE_FRONT_GLOW_OPACITY;
    const radius = Math.max(0.8, node.radius * node.screenScale);
    const glowRadius = radius + padding;
    const glowSize = glowRadius * 2;
    context.globalAlpha = clamp(fade * intensity * opacity);
    context.drawImage(
      glowCapture,
      node.screenX - glowRadius,
      node.screenY - glowRadius,
      glowSize,
      glowSize,
    );
    context.globalAlpha = 1;
    cortexMetrics.maxGlowPadding = Math.max(
      cortexMetrics.maxGlowPadding,
      padding,
    );
  }

  function drawCaptureWaveTrain(time, star) {
    const staticMotion = reducedMotion.matches || !motionEnabled;
    const falloffWidth = clamp(star.radius * 0.18, 42, 112);
    let latestWave = null;
    for (let index = 0; index < captureWaveTrain.length; index += 1) {
      const wave = captureWaveTrain[index];
      const progress = clamp((time - wave.startedAt) / wave.duration);
      wave.frameState = Runtime.captureWaveState(
        progress,
        star.radius,
        staticMotion,
      );
      wave.framePainted = false;
      wave.frameProgress = progress;
      if (!latestWave || wave.seq > latestWave.seq) latestWave = wave;
    }
    cortexMetrics.captureEffectCount = captureWaveTrain.length;
    cortexMetrics.captureEffectSeq = captureHeartbeatBurst.finalSeq;
    cortexMetrics.captureEffectCaptureId = captureHeartbeatBurst.finalCaptureId;
    cortexMetrics.captureCenterX = star.x;
    cortexMetrics.captureCenterY = star.y;
    cortexMetrics.captureRadius = star.radius;
    cortexMetrics.captureBandWidth = falloffWidth;
    cortexMetrics.captureProgress = latestWave?.frameProgress || 0;
    cortexMetrics.capturePhase = captureWaveTrain.length > 1
      ? "wave-train"
      : latestWave?.frameState.phase || "idle";
    let waveNodes = 0;
    let afterglowNodes = 0;
    let drawnNodes = 0;
    const start = nodes.length
      ? Math.floor(
          deterministicUnit("capture-wave-train", 43)
          * nodes.length,
        )
      : 0;
    const stride = Runtime.captureWaveSampleStride(
      nodes.length,
      captureWaveSampleEffect,
    );
    for (let offset = 0; offset < nodes.length; offset += 1) {
      if (drawnNodes >= CAPTURE_WAVE_NODE_LIMIT) break;
      const node = nodes[(start + offset * stride) % nodes.length];
      const distance = captureWaveDistances[node.index];
      if (distance < 0) continue;
      let combinedIntensity = 0;
      let contributorMask = 0;
      let strongestIntensity = 0;
      let strongestWave = null;
      for (let index = 0; index < captureWaveTrain.length; index += 1) {
        const wave = captureWaveTrain[index];
        const intensity = captureWaveNodeIntensity(
          distance,
          wave.frameState.waveRadius,
          falloffWidth,
          wave.frameState.wavePeak,
        );
        combinedIntensity = 1 - (1 - combinedIntensity) * (1 - intensity);
        if (intensity > 0) contributorMask |= 1 << index;
        if (intensity > strongestIntensity) {
          strongestIntensity = intensity;
          strongestWave = wave;
        }
      }
      if (!strongestWave || combinedIntensity <= 0) continue;
      const inTail = distance > strongestWave.frameState.waveRadius;
      drawCaptureWaveNodeGlow(node, combinedIntensity, 1, inTail);
      for (let index = 0; index < captureWaveTrain.length; index += 1) {
        if (contributorMask & (1 << index)) {
          captureWaveTrain[index].framePainted = true;
        }
      }
      drawnNodes += 1;
      if (inTail) afterglowNodes += 1;
      else waveNodes += 1;
      cortexMetrics.captureWavePeak = Math.max(
        cortexMetrics.captureWavePeak,
        combinedIntensity,
      );
    }
    let paintedWaveCount = 0;
    for (let index = 0; index < captureWaveTrain.length; index += 1) {
      const wave = captureWaveTrain[index];
      if (!wave.framePainted) continue;
      paintedWaveCount += 1;
      if (!wave.paintedAt) {
        wave.paintedAt = time;
        cortexMetrics.transportPainted += 1;
      }
    }
    cortexMetrics.captureWaveEffects = paintedWaveCount;
    cortexMetrics.captureWaveNodes = waveNodes;
    cortexMetrics.captureAfterglowNodes = afterglowNodes;
    cortexMetrics.transportElectricPeak = Math.max(
      cortexMetrics.transportElectricPeak,
      cortexMetrics.captureWavePeak,
    );
    return drawnNodes > 0;
  }

  function drawCaptureHeartbeat(star, peak, fade) {
    const alpha = Runtime.captureHeartbeatAlpha(peak, fade);
    if (alpha <= 0) return false;
    const glowSize = star.radius * (2.05 + peak * 0.38);
    context.globalAlpha = alpha;
    context.drawImage(
      glowCapture,
      star.x - glowSize / 2,
      star.y - glowSize / 2,
      glowSize,
      glowSize,
    );
    context.globalAlpha = 1;
    return true;
  }

  function drawCaptureHeartbeatBurst(time, star) {
    if (captureHeartbeatBurst.recorded || time < captureHeartbeatBurst.dueAt) {
      return false;
    }
    const progress = (
      time - captureHeartbeatBurst.dueAt
    ) / CAPTURE_HEARTBEAT_DURATION_MS;
    if (progress >= 1) {
      if (captureHeartbeatBurst.painted) {
        cortexMetrics.captureHeartbeatCount += 1;
        cortexMetrics.captureHeartbeatSeq = captureHeartbeatBurst.finalSeq;
      }
      captureHeartbeatBurst.recorded = true;
      cortexMetrics.capturePhase = "settle";
      return false;
    }
    const staticMotion = reducedMotion.matches || !motionEnabled;
    const peak = staticMotion ? 0.3 : Runtime.captureHeartbeatPeak(progress);
    const painted = drawCaptureHeartbeat(star, peak, 1);
    captureHeartbeatBurst.painted ||= painted;
    if (staticMotion && !captureHeartbeatBurst.wakeTimer) {
      scheduleCaptureHeartbeatWake(
        captureHeartbeatBurst.generation,
        captureHeartbeatBurst.dueAt + CAPTURE_HEARTBEAT_DURATION_MS,
      );
    }
    cortexMetrics.captureHeartbeatPeak = peak;
    cortexMetrics.captureEffectSeq = captureHeartbeatBurst.finalSeq;
    cortexMetrics.captureEffectCaptureId = captureHeartbeatBurst.finalCaptureId;
    cortexMetrics.captureCenterX = star.x;
    cortexMetrics.captureCenterY = star.y;
    cortexMetrics.captureRadius = star.radius;
    cortexMetrics.capturePhase = "heartbeat";
    cortexMetrics.transportElectricPeak = Math.max(
      cortexMetrics.transportElectricPeak,
      peak,
    );
    return painted;
  }

  function captureHeartbeatHasRenderWork(time) {
    return (
      motionEnabled
      && !reducedMotion.matches
      && !captureHeartbeatBurst.recorded
      && time >= captureHeartbeatBurst.dueAt
    );
  }

  function drawCaptureEffects(time) {
    pruneCaptureWaveTrain(time);
    let star = null;
    let painted = false;
    if (captureWaveTrain.length) {
      star = captureWaveGeometry();
      painted = drawCaptureWaveTrain(time, star);
    }
    if (
      !captureHeartbeatBurst.recorded
      && time >= captureHeartbeatBurst.dueAt
    ) {
      if (!star) star = captureWaveGeometry();
      painted = drawCaptureHeartbeatBurst(time, star) || painted;
    }
    return painted;
  }

  function beginLabelCandidates() {
    labelCandidates.length = 0;
    labelCandidateGeneration = (labelCandidateGeneration + 1) >>> 0;
    if (!labelCandidateGeneration) {
      labelCandidateMarks.fill(0);
      labelCandidateGeneration = 1;
    }
  }

  function addLabelCandidate(node) {
    if (!node || labelCandidateMarks[node.index] === labelCandidateGeneration) {
      return;
    }
    labelCandidateMarks[node.index] = labelCandidateGeneration;
    labelCandidates.push(node);
  }

  function collectLabelCandidates(time) {
    beginLabelCandidates();
    for (let index = 0; index < nodes.length; index += 1) {
      const node = nodes[index];
      const state = nodeState[index];
      if (state < 2 || node.viewDepth > 9e8) continue;
      if (state === 3 || index === selected || index === hovered) {
        addLabelCandidate(node);
      }
    }
    if (camera.distance < 1500) {
      labelHubs.forEach((index) => {
        const node = nodes[index];
        if (
          node
          && nodeState[index] >= 2
          && node.viewDepth <= 9e8
          && node.viewDepth < camera.distance
        ) {
          addLabelCandidate(node);
        }
      });
    }
    for (let index = 0; index < labelCandidates.length; index += 1) {
      const node = labelCandidates[index];
      node.labelPriority =
        (node.index === selected ? 100000 : 0)
        + (node.index === hovered ? 50000 : 0)
        + (mode === "sphere"
          ? clamp(
              (camera.distance - node.viewDepth)
                / Math.max(1, sphereQuality.targetRadius.max || 540),
              -1,
              1,
            ) * 700
          : 0)
        + node.fanIn;
    }
    labelCandidates.sort(
      (left, right) =>
        right.labelPriority - left.labelPriority || left.index - right.index,
    );
    return labelCandidates;
  }

  function drawLabels(time) {
    context.font = labelFont;
    context.textAlign = "center";
    occupiedLabels.length = 0;
    const candidates = collectLabelCandidates(time);
    candidates.forEach((node) => {
      const state = nodeState[node.index];
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
      const overlaps = occupiedLabels.some(
        (other) =>
          bounds.left < other.right
          && bounds.right > other.left
          && bounds.top < other.bottom
          && bounds.bottom > other.top,
      );
      if (overlaps && state !== 3) return;
      occupiedLabels.push(bounds);
      cortexMetrics.labelsPainted += 1;
      context.fillStyle = `rgba(3,5,10,${0.7 * depthFade})`;
      context.fillRect(node.screenX - labelWidth / 2 - 3, y - 9, labelWidth + 6, 12);
      context.fillStyle =
        node.index === selected || node.index === hovered
          ? rgba(RGB_HOT, 0.92 * depthFade)
          : state === 3
              ? rgba(RGB_FIRE, 0.85 * depthFade)
              : rgba([160, 178, 210], 0.8 * depthFade);
      context.fillText(node.name, node.screenX, y);
    });
  }

  function draw(time, webglBaseRendered) {
    cortexMetrics.labelsPainted = 0;
    cortexMetrics.maxCoreScale = 0;
    cortexMetrics.maxGlowPadding = 0;
    cortexMetrics.captureEffectCount = 0;
    cortexMetrics.captureEffectSeq = -1;
    cortexMetrics.captureEffectCaptureId = "";
    cortexMetrics.captureWaveEffects = 0;
    cortexMetrics.captureWaveNodes = 0;
    cortexMetrics.captureAfterglowNodes = 0;
    cortexMetrics.captureWavePeak = 0;
    cortexMetrics.captureHeartbeatPeak = 0;
    cortexMetrics.captureCenterX = 0;
    cortexMetrics.captureCenterY = 0;
    cortexMetrics.captureRadius = 0;
    cortexMetrics.captureBandWidth = 0;
    cortexMetrics.captureProgress = 0;
    cortexMetrics.capturePhase = "idle";
    cortexMetrics.transportElectricPeak = 0;
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

    if (!webglBaseRendered) drawEdges();

    drawNodes(time, webglBaseRendered);
    drawCameraPivot(time);

    drawCaptureEffects(time);
    drawLabels(time);

  }

  let previousTime = performance.now();
  let simulationAccumulator = 0;
  function cameraIsMoving() {
    if (cameraTarget || cameraPivotTarget) return true;
    const pivotNode = nodes[camera.pivotNodeIndex];
    return Boolean(
      pivotNode
      && Math.hypot(
        pivotNode.x - camera.pivotX,
        pivotNode.y - camera.pivotY,
        pivotNode.z - camera.pivotZ,
      ) >= 0.12
    );
  }

  function hasRenderWork() {
    const autoRotationReady = autoRotateWake.sync();
    return Boolean(
      stateDirty
      || simulationAwake
      || cameraIsMoving()
      || pendingPointer
      || panelRenderPending
      || captureWaveTrain.length
      || captureHeartbeatHasRenderWork(performance.now())
      || autoRotationReady
    );
  }

  function flushFrameWork() {
    if (pendingPointer) {
      const pointer = pendingPointer;
      pendingPointer = null;
      processPointerMove(pointer);
    }

    if (panelRenderPending) {
      panelRenderPending = false;
      renderPanel();
    }
  }

  function frame(
    now,
    _reasons,
    frameState = { continuation: false, sinceLastRender: null },
  ) {
    const frameWorkStartedAt = performance.now();
    const rawDelta = Math.max(0, now - previousTime);
    const delta = Math.min(60, rawDelta);
    const cadence = Number(frameState.sinceLastRender);
    if (
      frameState.sinceLastRender !== null
      && Number.isFinite(cadence)
      && (
        frameState.continuation
        || cadence <= FRAME_CADENCE_IDLE_GAP_MS
      )
    ) {
      cortexMetrics.frameCadences[frameCadenceCursor] = cadence;
      frameCadenceCursor = (frameCadenceCursor + 1) % FRAME_DURATION_CAPACITY;
      frameCadenceCount = Math.min(
        FRAME_DURATION_CAPACITY,
        frameCadenceCount + 1,
      );
    }
    previousTime = now;
    let startedAt = performance.now();
    flushFrameWork();
    stageMetrics.record("domEventFlush", performance.now() - startedAt);
    startedAt = performance.now();
    if (simulationAwake) {
      simulationAccumulator = Math.min(
        SIMULATION_STEP_MS * SIMULATION_MAX_STEPS_PER_FRAME,
        simulationAccumulator + delta,
      );
      let simulationSteps = 0;
      while (
        simulationAwake
        && simulationAccumulator > SIMULATION_STEP_MS
        && simulationSteps < SIMULATION_MAX_STEPS_PER_FRAME
      ) {
        tick();
        simulationSteps += 1;
        if (!simulationAwake) break;
        simulationAccumulator -= SIMULATION_STEP_MS;
      }
    } else {
      simulationAccumulator = 0;
    }
    stageMetrics.record("simulation", performance.now() - startedAt);
    if (
      autoRotateWake.ready()
    ) {
      camera.theta += delta * 0.000045;
    }
    const pivotNode = nodes[camera.pivotNodeIndex];
    const pivotGoal = pivotNode || cameraPivotTarget;
    if (pivotGoal) {
      camera.pivotX += (pivotGoal.x - camera.pivotX) * 0.14;
      camera.pivotY += (pivotGoal.y - camera.pivotY) * 0.14;
      camera.pivotZ += (pivotGoal.z - camera.pivotZ) * 0.14;
      if (
        !pivotNode
        && Math.hypot(
          pivotGoal.x - camera.pivotX,
          pivotGoal.y - camera.pivotY,
          pivotGoal.z - camera.pivotZ,
        ) < 0.12
      ) {
        camera.pivotX = pivotGoal.x;
        camera.pivotY = pivotGoal.y;
        camera.pivotZ = pivotGoal.z;
        cameraPivotTarget = null;
      }
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
    startedAt = performance.now();
    if (stateDirty) recomputeState();
    projectAll();
    const webglBaseRendered = baseRenderer.render({
      now,
      width,
      height,
      pixelRatio,
      nodes,
      links,
      nodeState,
      edgeState,
      selected,
      hovered,
      mode,
      edgeVisibility,
      normalEdgeScale: mode === "sphere" ? SPHERE_NORMAL_EDGE_SCALE : 1,
      baseEdgeColor: RGB_STEEL,
      selectedEdgeColor: RGB_FIRE,
      fog,
    });
    stageMetrics.record("projectionBase", performance.now() - startedAt);
    startedAt = performance.now();
    draw(now, webglBaseRendered);
    stageMetrics.record("overlayEffects", performance.now() - startedAt);

    cortexMetrics.frameDurations[frameDurationCursor] =
      performance.now() - frameWorkStartedAt;
    frameDurationCursor = (frameDurationCursor + 1) % FRAME_DURATION_CAPACITY;
    frameDurationCount = Math.min(
      FRAME_DURATION_CAPACITY,
      frameDurationCount + 1,
    );
  }

  function pick(x, y) {
    return spatialIndex.pick(
      x,
      y,
      nodes,
      (node, pointerX, pointerY) =>
        Math.hypot(node.screenX - pointerX, node.screenY - pointerY)
        - Math.max(5, node.radius * node.screenScale),
    );
  }

  function select(index, setPivot = false) {
    selected = index;
    stateDirty = true;
    renderPanel();
    document.getElementById("panelBody").scrollTop = 0;
    renderTreeSelection();
    if (index >= 0) {
      if (setPivot) setNodeAsCameraPivot(index);
      else focusNode(index);
    }
    invalidate("selection");
  }

  function processPointerMove(pointer) {
    const tooltip = document.getElementById("tooltip");
    const { x, y } = pointer;
    if (downPoint && Math.hypot(x - downPoint.x, y - downPoint.y) > 4) {
      moved = true;
    }
    if (downPoint && moved) {
      camera.theta += pointer.movementX * 0.0045;
      camera.phi = Math.max(
        -1.35,
        Math.min(1.35, camera.phi - pointer.movementY * 0.0045),
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
  }

  function bindCanvasInteractions() {
    canvas.addEventListener("pointerdown", (event) => {
      canvas.setPointerCapture(event.pointerId);
      downPoint = { x: event.offsetX, y: event.offsetY };
      moved = false;
      dragging = true;
      lastInteraction = performance.now();
      invalidate("pointer-down");
    });
    canvas.addEventListener("pointermove", (event) => {
      if (pendingPointer && downPoint) {
        pendingPointer.x = event.offsetX;
        pendingPointer.y = event.offsetY;
        pendingPointer.movementX += event.movementX;
        pendingPointer.movementY += event.movementY;
      } else {
        pendingPointer = {
          x: event.offsetX,
          y: event.offsetY,
          movementX: event.movementX,
          movementY: event.movementY,
        };
      }
      invalidate("pointer-move");
    });
    canvas.addEventListener("pointerup", (event) => {
      if (pendingPointer) {
        processPointerMove(pendingPointer);
        pendingPointer = null;
      }
      if (!moved) {
        const index = pick(event.offsetX, event.offsetY);
        select(index, index >= 0);
      }
      downPoint = null;
      dragging = false;
      lastInteraction = performance.now();
      invalidate("pointer-up");
    });
    canvas.addEventListener("dblclick", (event) => {
      const index = pick(event.offsetX, event.offsetY);
      if (index >= 0) {
        select(index, true);
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
        invalidate("wheel");
      },
      { passive: false },
    );
  }

  const packageStats = {};
  const packageNodeCache = new Map();
  const packageTreeElements = new Map();

  function materializePackage(packageName, throughIndex = -1) {
    const list = packageTreeElements.get(packageName)?.list;
    const packageNodes = packageNodeCache.get(packageName) || [];
    if (!list) return;
    let rendered = Number(list.dataset.rendered || 0);
    const targetPosition = throughIndex >= 0
      ? packageNodes.findIndex((node) => node.index === throughIndex) + 1
      : 0;
    const target = Math.min(
      packageNodes.length,
      Math.max(rendered + EXPLORER_CHUNK_SIZE, targetPosition),
    );
    list.querySelector(".modMore")?.remove();
    const fragment = document.createDocumentFragment();
    for (; rendered < target; rendered += 1) {
      const node = packageNodes[rendered];
      const moduleRow = document.createElement("div");
      moduleRow.className = "modRow";
      moduleRow.dataset.index = String(node.index);
      moduleRow.innerHTML = `<span title="${escapeHtml(node.title)}">${escapeHtml(node.name)}</span>
        ${node.entrypoint ? '<span class="ep">▸ep</span>' : ""}
        <span class="l">${node.lineCount}</span>`;
      fragment.appendChild(moduleRow);
    }
    list.dataset.rendered = String(rendered);
    if (rendered < packageNodes.length) {
      const more = document.createElement("button");
      more.className = "modMore";
      more.type = "button";
      more.dataset.package = packageName;
      more.textContent = `load ${Math.min(EXPLORER_CHUNK_SIZE, packageNodes.length - rendered)} more…`;
      fragment.appendChild(more);
    }
    list.appendChild(fragment);
  }

  function buildTree() {
    const tree = document.getElementById("tree");
    packageNodeCache.clear();
    packageTreeElements.clear();
    nodes.forEach((node) => {
      if (!packageNodeCache.has(node.packageName)) {
        packageNodeCache.set(node.packageName, []);
      }
      packageNodeCache.get(node.packageName).push(node);
    });
    packageList.forEach((packageName) => {
      const packageNodes = packageNodeCache.get(packageName) || [];
      packageNodes.sort((left, right) => right.fanIn - left.fanIn);
      packageStats[packageName] = {
        count: packageNodes.length,
        lines: packageNodes.reduce((total, node) => total + node.lineCount, 0),
      };
    });
    document.getElementById("pkgCount").textContent = `${packageList.length} pkgs`;
    packageList.forEach((packageName) => {
      const row = document.createElement("div");
      row.className = "pkgRow";
      row.dataset.package = packageName;
      const dotColor = rgba(
        shade(RGB_STEEL, packageShade[packageName] || 1),
        1,
      );
      row.innerHTML = `<span class="dot"></span>
        <span class="n">${escapeHtml(packageName)}</span>
        <span class="c">${packageStats[packageName].count}</span>
        <span class="ar">▶</span>`;
      row.querySelector(".dot").style.background = dotColor;
      const list = document.createElement("div");
      list.className = "modList";
      list.dataset.package = packageName;
      list.dataset.rendered = "0";
      packageTreeElements.set(packageName, { row, list });
      tree.appendChild(row);
      tree.appendChild(list);
    });
    tree.addEventListener("click", (event) => {
      const moduleRow = event.target.closest(".modRow[data-index]");
      if (moduleRow) {
        select(Number(moduleRow.dataset.index));
        return;
      }
      const more = event.target.closest(".modMore[data-package]");
      if (more) {
        materializePackage(more.dataset.package);
        return;
      }
      const packageRow = event.target.closest(".pkgRow[data-package]");
      if (!packageRow) return;
      const packageName = packageRow.dataset.package;
      const list = packageTreeElements.get(packageName)?.list;
      if (!list) return;
      if (event.target.classList.contains("dot")) {
        if (packageOff.has(packageName)) packageOff.delete(packageName);
        else packageOff.add(packageName);
        packageRow.classList.toggle("off", packageOff.has(packageName));
        stateDirty = true;
        invalidate("package-visibility");
        return;
      }
      packageRow.classList.toggle("open");
      list.classList.toggle("open");
      if (list.classList.contains("open") && !Number(list.dataset.rendered)) {
        materializePackage(packageName);
      }
    });
    tree.addEventListener("dblclick", (event) => {
      const moduleRow = event.target.closest(".modRow[data-index]");
      if (!moduleRow) return;
      const index = Number(moduleRow.dataset.index);
      select(index);
      flashTicker(`preview · ${nodes[index].id}`);
    });
  }

  function renderTreeSelection() {
    const tree = document.getElementById("tree");
    tree
      .querySelectorAll(".modRow.sel")
      .forEach((element) => element.classList.remove("sel"));
    if (selected < 0) return;
    let element = tree.querySelector(
      `.modRow[data-index="${selected}"]`,
    );
    if (!element) {
      const node = nodes[selected];
      if (node) {
        materializePackage(node.packageName, selected);
        element = tree.querySelector(`.modRow[data-index="${selected}"]`);
      }
    }
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
            <div class="k"><span class="sw static"></span>static ${staticLinks.toLocaleString()}</div>
            <div class="k"><span class="sw deferred"></span>deferred ${deferredLinks.toLocaleString()}</div>
          </div>
        </div>
      </div>
      <div class="sec"><h3>TOP HUBS · FAN-IN</h3>
        ${topHubs
          .map(
            (node) => `<div class="hubRow" data-index="${node.index}">
              <span class="nm">${escapeHtml(node.id)}</span>
              <span class="bar"><i data-percent="${Math.round((node.fanIn / topFanIn) * 100)}"></i></span>
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
              <span class="bar"><i data-percent="${Math.round((packageStats[packageName].lines / maxLines) * 100)}"></i></span>
              <span class="v">${(packageStats[packageName].lines / 1000).toFixed(1)}k</span>
            </div>`,
          )
          .join("")}
        ${packageList.length > loadPackages.length ? `<div class="ghost">+ ${packageList.length - loadPackages.length} more categories</div>` : ""}
      </div>
      <div class="sec"><h3>CAPTURE</h3>
        <div class="mrow"><span class="capture-label">◉ SAVE</span><b>cyan · inward memory wave → heartbeat</b></div>
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
        ${node.entrypoint ? '<span class="pk entrypoint">entrypoint</span>' : ""}
        <div class="selected-metrics">
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
      const node = nodes[selected];
      panel.innerHTML = selectedHtml(node);
    }
    panel.querySelectorAll("[data-percent]").forEach((bar) => {
      bar.style.width = `${bar.dataset.percent}%`;
    });
  }

  function drawGauge() {
    const gauge = document.getElementById("gauge");
    if (!gauge) return;
    const gaugeContext = gauge.getContext("2d");
    const staticLinks = data.meta.static || 0;
    const deferredLinks = data.meta.deferred || 0;
    const target = staticLinks / Math.max(1, staticLinks + deferredLinks);
    let animation = 1;
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
      invalidate("search");
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
        invalidate("search-clear");
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
    document.getElementById("mSphere").addEventListener("click", () => {
      setMode("sphere");
    });
    document.getElementById("mCluster").addEventListener("click", () => {
      setMode("cluster");
    });
    document.querySelectorAll(".stim").forEach((button) => {
      button.addEventListener("click", () => stimulate(button.dataset.s));
    });
    document.getElementById("resetCenter").addEventListener("click", () => {
      resetCameraPivot();
    });
    syncCameraPivotControl();
    const liveToggle = document.getElementById("tLive");
    liveToggle.addEventListener("click", () => {
      liveEventsEnabled = !liveEventsEnabled;
      liveToggle.classList.toggle("on", liveEventsEnabled);
      flashTicker(
        liveEventsEnabled
          ? "Live capture effects resumed"
          : "Live capture effects paused",
      );
      invalidate("live-toggle");
    });
    const motionToggle = document.getElementById("tMotion");
    motionToggle.addEventListener("click", () => {
      motionEnabled = !motionEnabled;
      motionToggle.classList.toggle("on", motionEnabled);
      motionToggle.setAttribute("aria-pressed", String(motionEnabled));
      saveViewPreferences();
      flashTicker(
        motionEnabled
          ? "motion enabled"
          : "motion paused · static edge arrows remain",
      );
      invalidate("motion-toggle");
    });
    const rotateToggle = document.getElementById("tRot");
    rotateToggle.addEventListener("click", () => {
      autoRotate = !autoRotate;
      rotateToggle.classList.toggle("on", autoRotate);
      rotateToggle.setAttribute("aria-pressed", String(autoRotate));
      saveViewPreferences();
      invalidate("rotate-toggle");
    });
    const soundToggle = document.getElementById("tSnd");
    soundToggle.addEventListener("click", () => {
      soundOn = !soundOn;
      soundToggle.classList.toggle("on", soundOn);
      soundToggle.setAttribute("aria-pressed", String(soundOn));
      saveViewPreferences();
      invalidate("sound-toggle");
      unlockSound();
    });
    document.getElementById("visSlider").addEventListener("input", (event) => {
      edgeVisibility =
        clamp(
          Number(event.target.value),
          SYNAPSE_VISIBILITY_MIN,
          SYNAPSE_VISIBILITY_MAX,
        ) / 100;
      saveViewPreferences();
      invalidate("edge-visibility");
    });
    document
      .getElementById("tReset")
      .addEventListener("click", resetViewPreferences);
    window.addEventListener("pointerdown", unlockSound, {
      capture: true,
      once: true,
    });
    window.addEventListener("keydown", unlockSound, {
      capture: true,
      once: true,
    });
    document.getElementById("zIn").addEventListener("click", () => {
      camera.distance = Math.max(240, camera.distance / 1.4);
      cameraTarget = null;
      invalidate("zoom-in");
    });
    document.getElementById("zOut").addEventListener("click", () => {
      camera.distance = Math.min(4200, camera.distance * 1.4);
      cameraTarget = null;
      invalidate("zoom-out");
    });
    document.getElementById("zFit").addEventListener("click", () => {
      select(-1);
      fitView();
    });
    stage.addEventListener("keydown", (event) => {
      if (!["ArrowDown", "ArrowUp", "ArrowLeft", "ArrowRight"].includes(event.key)) {
        return;
      }
      const active = nodes.filter((node) => nodeState[node.index] > 0);
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
      invalidate("reduced-motion");
    });
  }

  function setMode(nextMode) {
    mode = Runtime.normalizeLayoutMode(nextMode);
    reheat(mode === "sphere" ? 0.9 : 0.7);
    syncViewPreferenceControls();
    saveViewPreferences();
    scheduleModeFit(mode);
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
      "Frozen memory pages and page links",
      `${topHub?.id || "—"} fan-in ${topHub?.fanIn || 0} — 最深部のハブニューロン`,
      `${nodeCount.toLocaleString()} neurons / ${((data.meta.static || 0) + (data.meta.deferred || 0)).toLocaleString()} synapses`,
      "DEMO SAVE is a visual preview",
      "cyan=capture",
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
    badge.lastChild.textContent = online ? "CORTEX ONLINE" : "CORTEX LINKING";
    live.textContent = online ? "● LIVE" : "○ LINKING";
  }

  function connectEvents() {
    const generation = ++eventSocketGeneration;
    if (eventSocket) eventSocket.close();
    const protocol = window.location.protocol === "https:" ? "wss:" : "ws:";
    eventSocket = new WebSocket(`${protocol}//${window.location.host}/api/cortex/events`);
    eventSocket.addEventListener("open", () => {
      if (generation === eventSocketGeneration) setEventStatus(true);
    });
    eventSocket.addEventListener("message", (message) => {
      if (generation !== eventSocketGeneration) return;
      let payload;
      try { payload = JSON.parse(message.data); } catch { return; }
      if (payload.type !== "events" || !Array.isArray(payload.events)) return;
      payload.events.slice(0, 32).forEach(visualizeTransportEvent);
    });
    eventSocket.addEventListener("close", () => {
      if (generation !== eventSocketGeneration) return;
      setEventStatus(false);
      window.clearTimeout(eventReconnect);
      eventReconnect = window.setTimeout(connectEvents, 2000);
    });
    eventSocket.addEventListener("error", () => {
      if (generation === eventSocketGeneration) setEventStatus(false);
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
      renderScheduler = Runtime.createRenderScheduler({
        requestFrame: (callback) => window.requestAnimationFrame(callback),
        cancelFrame: (handle) => window.cancelAnimationFrame(handle),
        isHidden: () => document.hidden,
        onFrame: frame,
        hasWork: hasRenderWork,
      });
      document.addEventListener("visibilitychange", () => {
        autoRotateWake.sync();
        renderScheduler.visibilityChanged();
      });
      window.addEventListener("beforeunload", () => {
        eventSocketGeneration += 1;
        window.clearTimeout(eventReconnect);
        if (eventSocket) eventSocket.close();
        window.clearTimeout(modeFitTimer);
        captureHeartbeatBurst.generation += 1;
        window.clearTimeout(captureHeartbeatBurst.wakeTimer);
        autoRotateWake.dispose();
        renderScheduler.dispose();
        baseRenderer.dispose();
        performanceObservers.forEach((observer) => observer.disconnect());
      }, { once: true });
      observeLongFrames();
      resize();
      const resizeObserver = new ResizeObserver(resize);
      resizeObserver.observe(stage);
      if (modeBar) resizeObserver.observe(modeBar);
      buildTree();
      renderPanel();
      bindCanvasInteractions();
      applyViewPreferences(loadViewPreferences());
      bindInterface();
      initializeHeader();
      recomputeState();
      const settleTicks = nodeCount > 1500 ? 8 : 32;
      for (let index = 0; index < settleTicks; index += 1) tick();
      fitView();
      camera.distance = cameraTarget.distance;
      cameraTarget = null;
      if (mode === "sphere") scheduleModeFit(mode);
      previousTime = performance.now();
      invalidate("boot");
      connectEvents();
    } catch (error) {
      showBootError(error);
    }
  }

  boot();
})();
