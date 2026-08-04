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
  const CAPTURE = "#4fe4ff";
  const CONSENSUS = "#ffe6ae";
  const RGB_STEEL = hexRgb(STEEL);
  const RGB_FIRE = hexRgb(FIRE);
  const RGB_HOT = hexRgb(FIRE_HOT);
  const RGB_ELECTRIC = hexRgb(ELECTRIC);
  const RGB_VIOLET = hexRgb(VIOLET);
  const RGB_COMMIT = hexRgb(COMMIT);
  const RGB_INHIBIT = hexRgb(INHIBIT);
  const RGB_FAULT = hexRgb(FAULT);
  const RGB_CAPTURE = hexRgb(CAPTURE);
  const RGB_CONSENSUS = hexRgb(CONSENSUS);
  const TYPE_OFF = new Set([2]);
  const ACTIVE_LABEL_LIMIT = 5;
  const NODE_FLASH_ATTACK_MS = 90;
  const NODE_FLASH_HOLD_MS = 150;
  const NODE_FLASH_DECAY_MS = 1450;
  const NODE_FLASH_DURATION_MS =
    NODE_FLASH_ATTACK_MS + NODE_FLASH_HOLD_MS + NODE_FLASH_DECAY_MS;
  const EDGE_AFTERGLOW_MS = 1550;
  const ELECTRIC_TRAVEL_MIN_MS = 420;
  const ELECTRIC_TRAVEL_MAX_MS = 760;
  const MAX_ELECTRIC_PATHS = 12;
  const MAX_TRANSPORT_EFFECTS = 18;
  const CAPTURE_COMET_DURATION_MS = 5200;
  const PROCESSING_EFFECT_PULSE_MS = 1450;
  const SIMULATION_STEP_MS = 15;
  const SIMULATION_ALPHA_FLOOR = 0.012;
  const SIMULATION_SLEEP_VELOCITY = 0.02;
  const SIMULATION_SLEEP_TICKS = 36;
  const SIMULATION_MAX_STEPS_PER_FRAME = 4;
  const METRICS_PUBLISH_INTERVAL_MS = 200;
  const FRAME_DURATION_CAPACITY = 240;
  const NODE_STIMULUS_SCALE = 0.38;
  const NODE_ARRIVAL_SCALE = 0.28;
  const NODE_CORE_SCALE = 1;
  const NODE_GLOW_MAX_PADDING_PX = 4;
  const NODE_EFFECT_MAX_PADDING_PX = 3;
  const LIVE_SESSION_VALUE = "__live__";
  const VIEW_PREFERENCES_KEY = "chronovisor.cortex.preferences.v1";
  const VIEW_PREFERENCES_DEFAULTS = Object.freeze({
    mode: "organic",
    motion: true,
    rotate: true,
    sound: false,
    relations: true,
    relationLifecycle: "all",
    synapseVisibility: 160,
  });
  const SYNAPSE_VISIBILITY_MIN = 30;
  const SYNAPSE_VISIBILITY_MAX = 420;
  const RELATION_STATUS_COLORS = Object.freeze({
    proposed: [108, 122, 148],
    held: RGB_INHIBIT,
    verified: RGB_VIOLET,
    repeatedly_used: RGB_COMMIT,
    authoritative: [103, 224, 184],
    stale: [107, 116, 132],
    retracted: RGB_FAULT,
  });

  let data;
  let nodes = [];
  let links = [];
  let simulationLinks = [];
  let nodeCount = 0;
  let neighbors = [];
  let neighborsByConnectivity = [];
  let neighborsByFanIn = [];
  let outgoing = [];
  let nodesByConnectivity = [];
  let byId = new Map();
  let edgeIndexByPair = new Map();
  let packageList = [];
  let packageShade = {};
  let anchors = {};
  let nodeState;
  let edgeState;
  let drawOrder = [];
  let labelHubs = new Set();
  let labelCandidateMarks = new Uint32Array(0);
  let labelCandidateGeneration = 0;
  const labelCandidates = [];
  const activeLabelNodes = [];
  const occupiedLabels = [];
  let formationCandidateMarks = new Uint32Array(0);
  let formationCandidateGeneration = 0;
  const formationCandidates = [];
  const consolidationCandidates = [];
  const transportFallbacks = [];
  let typedRelations = [];
  let typedCommunities = [];
  let communityHulls = [];
  let relationById = new Map();
  const activeRelationIds = new Set();

  let selected = -1;
  let hovered = -1;
  let query = "";
  let matches = new Set();
  let stateDirty = true;
  const packageOff = new Set();
  let mode = VIEW_PREFERENCES_DEFAULTS.mode;
  let liveEventsEnabled = true;
  let followLatestSession = true;
  let motionEnabled = VIEW_PREFERENCES_DEFAULTS.motion;
  let autoRotate = VIEW_PREFERENCES_DEFAULTS.rotate;
  let soundOn = VIEW_PREFERENCES_DEFAULTS.sound;
  let relationsVisible = VIEW_PREFERENCES_DEFAULTS.relations;
  let relationLifecycle = VIEW_PREFERENCES_DEFAULTS.relationLifecycle;
  let edgeVisibility = VIEW_PREFERENCES_DEFAULTS.synapseVisibility / 100;
  let alpha = 1;
  let simulationAwake = true;
  let simulationSettledTicks = 0;
  let simulationTicks = 0;
  let simulationSleepCount = 0;
  let simulationLastMaxVelocity = 0;
  let spikes = 0;
  let lastInteraction = 0;
  let lastVisualMetricsPublished = 0;
  let lastCortexMetricsPublished = -1e9;
  let frameDurationCursor = 0;
  let frameDurationCount = 0;
  let ingressReset = 0;
  let ingressRevision = 0;

  const stage = document.getElementById("stage");
  const canvas = document.getElementById("gl");
  const context = canvas.getContext("2d");
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
  let labelFont = "10.5px monospace";
  let dragging = false;
  let downPoint = null;
  let moved = false;
  let audioContext = null;
  let lastSound = 0;
  let eventSocket = null;
  let eventReconnect = null;
  let eventSocketGeneration = 0;
  let processingActivitySource = null;
  let processingPulseTimer = 0;
  let sessionRequest = 0;
  const reducedMotion = window.matchMedia("(prefers-reduced-motion: reduce)");
  const fieldState = window.CortexField.createState();
  const cortexMetrics = {
    spread: [],
    frameDurations: new Float32Array(FRAME_DURATION_CAPACITY),
    maxPulseQueue: 0,
    violetNodes: 0,
    labelsPainted: 0,
    afterglowEdges: 0,
    electricEdges: 0,
    electricPeak: 0,
    flashPeak: 0,
    maxCoreScale: 0,
    maxGlowPadding: 0,
    transportReceived: 0,
    transportPainted: 0,
    processingEvents: 0,
    activeProcessingLanes: 0,
    processingActivityConnected: false,
    captureEvents: 0,
    ingestEvents: 0,
    applyEvents: 0,
    cometTrailSegments: 0,
    cometHeads: 0,
    cometImpacts: 0,
    maxCometHeadRadius: 0,
    cometCenterX: 0,
    cometCenterY: 0,
    cometMinX: 0,
    cometMaxX: 0,
    cometMinY: 0,
    cometMaxY: 0,
    cometSafeLeft: 0,
    cometSafeRight: 0,
    cometSafeTop: 0,
    cometSafeBottom: 0,
    cometTurns: 0,
    explorationArcs: 0,
    triageCandidates: 0,
    generateParticles: 0,
    consensusOrbits: 0,
    processingNodeBlinks: 0,
    processingTargetNodeIndex: -1,
    formationOriginX: 0,
    formationOriginY: 0,
    formationTargetX: 0,
    formationTargetY: 0,
    consolidationEdges: 0,
    transportElectricPeak: 0,
  };
  function recentFrameDurations(limit = FRAME_DURATION_CAPACITY) {
    const count = Math.min(limit, frameDurationCount);
    const values = new Array(count);
    const start =
      (frameDurationCursor - count + FRAME_DURATION_CAPACITY)
      % FRAME_DURATION_CAPACITY;
    for (let index = 0; index < count; index += 1) {
      values[index] =
        cortexMetrics.frameDurations[
          (start + index) % FRAME_DURATION_CAPACITY
        ];
    }
    return values;
  }

  window.chronovisorCortexMetrics = () => ({
    spread: cortexMetrics.spread.map((row) => ({ ...row })),
    frameDurations: recentFrameDurations(),
    maxPulseQueue: cortexMetrics.maxPulseQueue,
    pulseQueue: pulses.length,
    simulation: {
      awake: simulationAwake,
      alpha,
      ticks: simulationTicks,
      sleepCount: simulationSleepCount,
      settledTicks: simulationSettledTicks,
      lastMaxVelocity: simulationLastMaxVelocity,
    },
    visual: {
      violetNodes: cortexMetrics.violetNodes,
      labelsPainted: cortexMetrics.labelsPainted,
      afterglowEdges: cortexMetrics.afterglowEdges,
      electricEdges: cortexMetrics.electricEdges,
      electricPeak: cortexMetrics.electricPeak,
      flashPeak: cortexMetrics.flashPeak,
      maxCoreScale: cortexMetrics.maxCoreScale,
      maxGlowPadding: cortexMetrics.maxGlowPadding,
      activeLabelLimit: ACTIVE_LABEL_LIMIT,
      electricPathLimit: MAX_ELECTRIC_PATHS,
      attackMs: NODE_FLASH_ATTACK_MS,
      holdMs: NODE_FLASH_HOLD_MS,
      decayMs: NODE_FLASH_DECAY_MS,
      edgeAfterglowMs: EDGE_AFTERGLOW_MS,
      electricTravelMinMs: ELECTRIC_TRAVEL_MIN_MS,
      electricTravelMaxMs: ELECTRIC_TRAVEL_MAX_MS,
      transportReceived: cortexMetrics.transportReceived,
      transportPainted: cortexMetrics.transportPainted,
      processingEvents: cortexMetrics.processingEvents,
      activeProcessingLanes: cortexMetrics.activeProcessingLanes,
      processingActivityConnected: cortexMetrics.processingActivityConnected,
      captureEvents: cortexMetrics.captureEvents,
      ingestEvents: cortexMetrics.ingestEvents,
      applyEvents: cortexMetrics.applyEvents,
      cometTrailSegments: cortexMetrics.cometTrailSegments,
      cometHeads: cortexMetrics.cometHeads,
      cometImpacts: cortexMetrics.cometImpacts,
      maxCometHeadRadius: cortexMetrics.maxCometHeadRadius,
      cometCenterX: cortexMetrics.cometCenterX,
      cometCenterY: cortexMetrics.cometCenterY,
      cometMinX: cortexMetrics.cometMinX,
      cometMaxX: cortexMetrics.cometMaxX,
      cometMinY: cortexMetrics.cometMinY,
      cometMaxY: cortexMetrics.cometMaxY,
      cometSafeLeft: cortexMetrics.cometSafeLeft,
      cometSafeRight: cortexMetrics.cometSafeRight,
      cometSafeTop: cortexMetrics.cometSafeTop,
      cometSafeBottom: cortexMetrics.cometSafeBottom,
      cometTurns: cortexMetrics.cometTurns,
      explorationArcs: cortexMetrics.explorationArcs,
      triageCandidates: cortexMetrics.triageCandidates,
      generateParticles: cortexMetrics.generateParticles,
      consensusOrbits: cortexMetrics.consensusOrbits,
      processingNodeBlinks: cortexMetrics.processingNodeBlinks,
      processingTargetNodeIndex: cortexMetrics.processingTargetNodeIndex,
      cameraPivotNodeIndex: camera.pivotNodeIndex,
      cameraPivotX: camera.pivotX,
      cameraPivotY: camera.pivotY,
      cameraPivotZ: camera.pivotZ,
      cameraDistance: camera.distance,
      formationOriginX: cortexMetrics.formationOriginX,
      formationOriginY: cortexMetrics.formationOriginY,
      formationTargetX: cortexMetrics.formationTargetX,
      formationTargetY: cortexMetrics.formationTargetY,
      consolidationEdges: cortexMetrics.consolidationEdges,
      transportElectricPeak: cortexMetrics.transportElectricPeak,
    },
  });

  const pulses = [];
  const edgeAfterglows = [];
  const nodeEffects = [];
  const transportEffects = [];
  const activeProcessingLanes = new Map();
  const demoTransportTimers = [];
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

  function mix(from, to, factor) {
    return from.map((channel, index) => channel + (to[index] - channel) * factor);
  }

  function clamp(value, minimum = 0, maximum = 1) {
    return Math.max(minimum, Math.min(maximum, value));
  }

  function sanitizeViewPreferences(candidate) {
    const value = candidate && typeof candidate === "object" ? candidate : {};
    const rawVisibility = Number(value.synapseVisibility);
    return {
      mode: value.mode === "cluster" ? "cluster" : "organic",
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
      relations:
        typeof value.relations === "boolean"
          ? value.relations
          : VIEW_PREFERENCES_DEFAULTS.relations,
      relationLifecycle: [
        "all", "proposed", "held", "verified", "repeatedly_used",
        "authoritative", "stale", "retracted",
      ].includes(value.relationLifecycle)
        ? value.relationLifecycle
        : VIEW_PREFERENCES_DEFAULTS.relationLifecycle,
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
      relations: relationsVisible,
      relationLifecycle,
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
      relations: document.getElementById("tRelations"),
    };
    for (const [name, control] of Object.entries(controls)) {
      const enabled = {
        motion: motionEnabled,
        rotate: autoRotate,
        sound: soundOn,
        relations: relationsVisible,
      }[
        name
      ];
      control.classList.toggle("on", enabled);
      control.setAttribute("aria-pressed", String(enabled));
    }
    document
      .getElementById("mOrganic")
      .classList.toggle("on", mode === "organic");
    document
      .getElementById("mCluster")
      .classList.toggle("on", mode === "cluster");
    document.getElementById("visSlider").value = String(
      Math.round(edgeVisibility * 100),
    );
    document.getElementById("relationLifecycle").value = relationLifecycle;
  }

  function applyViewPreferences(candidate) {
    const preferences = sanitizeViewPreferences(candidate);
    mode = preferences.mode;
    motionEnabled = preferences.motion;
    autoRotate = preferences.rotate;
    soundOn = preferences.sound;
    relationsVisible = preferences.relations;
    relationLifecycle = preferences.relationLifecycle;
    edgeVisibility = preferences.synapseVisibility / 100;
    syncViewPreferenceControls();
  }

  function resetViewPreferences() {
    try {
      window.localStorage.removeItem(VIEW_PREFERENCES_KEY);
    } catch {
      // Keep the in-memory reset even when storage access is unavailable.
    }
    applyViewPreferences(VIEW_PREFERENCES_DEFAULTS);
    reheat(0.7);
    window.setTimeout(fitView, 900);
    flashTicker("view preferences reset");
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

  function electricTravelDuration(strength) {
    return (
      ELECTRIC_TRAVEL_MAX_MS
      - clamp(strength) * (ELECTRIC_TRAVEL_MAX_MS - ELECTRIC_TRAVEL_MIN_MS)
    );
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

  function resize() {
    width = stage.clientWidth;
    height = stage.clientHeight;
    canvas.width = Math.max(1, Math.round(width * pixelRatio));
    canvas.height = Math.max(1, Math.round(height * pixelRatio));
    labelFont = `10.5px ${getComputedStyle(document.body).getPropertyValue("--mono")}`;
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
      communities: Array.isArray(row.communities) ? row.communities : [],
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
    links = data.links.map((row) => {
      const source = row[0];
      const target = row[1];
      return {
        source,
        target,
        kind: row[2] || 0,
        edgeType: row[3] || "wikilink",
        eventOnly: false,
        typed: false,
        restLength:
          58 + (nodes[source]?.radius || 0) + (nodes[target]?.radius || 0),
      };
    });
    typedRelations = Array.isArray(data.typedGraph?.relations)
      ? data.typedGraph.relations
      : [];
    typedCommunities = Array.isArray(data.typedGraph?.communities)
      ? data.typedGraph.communities
      : [];
    relationById = new Map(
      typedRelations.map((relation) => [relation.relation_id, relation]),
    );
    typedRelations.forEach((relation) => {
      if (!nodes[relation.source] || !nodes[relation.target]) return;
      links.push({
        source: relation.source,
        target: relation.target,
        kind: 1,
        edgeType: `relation:${relation.predicate}`,
        eventOnly: false,
        typed: true,
        relationId: relation.relation_id,
        predicate: relation.predicate,
        lifecycle: relation.status,
        direction: relation.direction,
        restLength:
          58
          + (nodes[relation.source]?.radius || 0)
          + (nodes[relation.target]?.radius || 0),
      });
    });
    simulationLinks = links.filter((link) => link.kind !== 2);
    nodeCount = nodes.length;
    neighbors = Array.from({ length: nodeCount }, () => new Set());
    outgoing = Array.from({ length: nodeCount }, () => []);
    links.forEach((link, edgeIndex) => {
      if (!nodes[link.source] || !nodes[link.target]) return;
      neighbors[link.source].add(link.target);
      neighbors[link.target].add(link.source);
      if (link.kind < 2 && !link.typed) outgoing[link.source].push(edgeIndex);
    });
    edgeIndexByPair = new Map();
    links.forEach((link, edgeIndex) => {
      const key = `${link.source}:${link.target}`;
      if (!edgeIndexByPair.has(key)) edgeIndexByPair.set(key, edgeIndex);
    });
    nodesByConnectivity = [...nodes].sort(
      (left, right) =>
        right.fanIn + right.fanOut - left.fanIn - left.fanOut,
    );
    neighborsByConnectivity = neighbors.map((indexes) =>
      [...indexes].sort(
        (left, right) =>
          nodes[right].fanIn + nodes[right].fanOut
          - nodes[left].fanIn - nodes[left].fanOut,
      ),
    );
    neighborsByFanIn = neighbors.map((indexes) =>
      [...indexes].sort(
        (left, right) => nodes[right].fanIn - nodes[left].fanIn,
      ),
    );
    byId = new Map(nodes.map((node) => [node.id, node]));
    communityHulls = typedCommunities
      .map((community) => ({
        id: community.community_id,
        members: (community.member_page_ids || [])
          .map((pageId) => byId.get(pageId)?.index)
          .filter((index) => Number.isInteger(index)),
        summarySha256: community.summary_sha256 || "",
      }))
      .filter((community) => community.members.length >= 3)
      .sort((left, right) => right.members.length - left.members.length)
      .slice(0, 18);
    communityHulls.forEach((community, communityIndex) => {
      const hue = deterministicUnit(community.id, communityIndex + 401);
      community.color = hue > 0.5 ? [126, 105, 210] : [79, 124, 180];
      community.points = [];
      community.lowerHull = [];
      community.upperHull = [];
      community.hull = [];
    });
    nodeState = new Uint8Array(nodeCount);
    edgeState = new Uint8Array(links.length + 256);
    labelCandidateMarks = new Uint32Array(nodeCount);
    formationCandidateMarks = new Uint32Array(nodeCount);
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
      node.anchor = anchor;
      node.x = anchor.x * 0.42 + (deterministicUnit(node.id, 11) - 0.5) * 330;
      node.y = anchor.y * 0.42 + (deterministicUnit(node.id, 29) - 0.5) * 280;
      node.z = anchor.z * 0.42 + (deterministicUnit(node.id, 47) - 0.5) * 330;
    });
  }

  function tick() {
    const centerForce = alpha * 0.00055;
    const anchorForce = alpha * (mode === "cluster" ? 0.015 : 0.0007);
    for (let index = 0; index < nodes.length; index += 1) {
      const node = nodes[index];
      const anchor = node.anchor;
      node.vx -= node.x * centerForce * 0.8;
      node.vy -= node.y * centerForce * 1.25;
      node.vz -= node.z * centerForce;
      node.vx += (anchor.x - node.x) * anchorForce;
      node.vy += (anchor.y - node.y) * anchorForce;
      node.vz += (anchor.z - node.z) * anchorForce;
    }

    const spring = mode === "cluster" ? 0.0012 : 0.0007;
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
      node.vx *= 0.82;
      node.vy *= 0.82;
      node.vz *= 0.82;
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
  }

  function projectAll() {
    focalLength = Math.min(width, height) * 1.12;
    projectionCosTheta = Math.cos(camera.theta);
    projectionSinTheta = Math.sin(camera.theta);
    projectionCosPhi = Math.cos(camera.phi);
    projectionSinPhi = Math.sin(camera.phi);
    projectionCenterX = width / 2;
    projectionCenterY = height / 2;
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
    resetCameraPivot(false);
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
        || (link.typed && (!relationsVisible
          || (relationLifecycle !== "all" && link.lifecycle !== relationLifecycle)))
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

  function clearDemoTransportTimers() {
    demoTransportTimers.splice(0).forEach((timer) => window.clearTimeout(timer));
  }

  function scheduleDemoTransport(delay, event) {
    demoTransportTimers.push(
      window.setTimeout(() => visualizeTransportEvent({ ...event, source: "demo" }), delay),
    );
  }

  function stimulate(name, label = "") {
    if (name === "save") {
      clearDemoTransportTimers();
      visualizeTransportEvent({
        kind: "save",
        phase: "capture",
        label: "DEMO CAPTURE · 18.4 KB · ID 9f3a7c21",
        byte_count: 18841,
        raw_count: 3,
        capture_id: "9f3a7c21",
        source: "demo",
      });
      return;
    }
    if (name === "ingest") {
      clearDemoTransportTimers();
      scheduleDemoTransport(0, {
        kind: "ingest",
        phase: "triage",
        label: "DEMO TRIAGE · scanning raw memory",
        page_ids: ["chronovisor-system"],
      });
      scheduleDemoTransport(1100, {
        kind: "ingest",
        phase: "generate",
        label: "DEMO GENERATE · semantic projection",
        page_ids: ["chronovisor-system"],
      });
      scheduleDemoTransport(2200, {
        kind: "ingest",
        phase: "consensus",
        label: "DEMO CONSENSUS · local verification",
        page_ids: ["chronovisor-system"],
      });
      scheduleDemoTransport(3300, {
        kind: "ingest",
        phase: "apply",
        operation: "updated",
        label: "DEMO MEMORY UPDATED · chronovisor-system",
        page_ids: ["chronovisor-system"],
      });
      return;
    }
    const roots = scenarioNodes(name);
    const now = performance.now();
    roots.forEach((node, rootIndex) => {
      const startedAt = now + rootIndex * 35;
      exciteNode(node, 0.9 * NODE_STIMULUS_SCALE, startedAt);
      nodeEffects.push({
        nodeIndex: node.index,
        kind: "stimulus",
        startedAt,
        duration: 650,
        delta: 0.9 * NODE_STIMULUS_SCALE,
        seq: -(rootIndex + 1),
        demo: true,
      });
      const candidates = outgoing[node.index]
        .filter((edgeIndex) => edgeState[edgeIndex] > 0)
        .sort((left, right) => left - right)
        .slice(0, 3);
      candidates.forEach((edgeIndex, branch) => {
        queueElectricPulse({
          edgeIndex,
          startedAt: now + 120 + branch * 80,
          duration: electricTravelDuration(0.78 - branch * 0.12),
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

  const STIMULATE_KINDS = new Set(["auto_recall", "recall", "read", "search", "used"]);

  function stimulateFromTransport(event) {
    if (event.source === "demo") return;
    if (!STIMULATE_KINDS.has(event.kind)) return;
    const pageIds = Array.isArray(event.page_ids) ? event.page_ids : [];
    const targets = pageIds.map((id) => byId.get(id)).filter(Boolean);
    if (!targets.length) {
      const fallback = visibleHub();
      if (fallback) targets.push(fallback);
    }
    if (!targets.length) return;
    const now = performance.now();
    const scale = event.kind === "auto_recall" ? 0.9 : 0.6;
    targets.slice(0, 5).forEach((node, index) => {
      const startedAt = now + index * 35;
      exciteNode(node, scale * NODE_STIMULUS_SCALE, startedAt);
      nodeEffects.push({
        nodeIndex: node.index,
        kind: "stimulus",
        startedAt,
        duration: 650,
        delta: scale * NODE_STIMULUS_SCALE,
        seq: -(index + 1),
        demo: false,
      });
      const candidates = outgoing[node.index]
        .filter((edgeIndex) => edgeState[edgeIndex] > 0)
        .sort((left, right) => left - right)
        .slice(0, 3);
      candidates.forEach((edgeIndex, branch) => {
        queueElectricPulse({
          edgeIndex,
          startedAt: now + 120 + branch * 80,
          duration: electricTravelDuration((scale - 0.12) - branch * 0.12),
          delta: (scale - 0.12) - branch * 0.12,
          seq: -(index * 10 + branch + 1),
          edgeType: links[edgeIndex].edgeType,
          demo: false,
          paintedAt: 0,
        });
      });
    });
    trimVisualQueues();
  }

  function queueElectricPulse(pulse) {
    pulses.push(pulse);
    pulses.sort(
      (left, right) =>
        right.delta - left.delta
        || right.startedAt - left.startedAt
        || right.seq - left.seq,
    );
    if (pulses.length > MAX_ELECTRIC_PATHS) {
      pulses.splice(MAX_ELECTRIC_PATHS);
    }
  }

  function trimVisualQueues() {
    if (pulses.length > MAX_ELECTRIC_PATHS) {
      pulses.splice(MAX_ELECTRIC_PATHS);
    }
    if (nodeEffects.length > window.CortexField.MAX_EVENTS) {
      nodeEffects.splice(0, nodeEffects.length - window.CortexField.MAX_EVENTS);
    }
    if (edgeAfterglows.length > MAX_ELECTRIC_PATHS) {
      edgeAfterglows.splice(0, edgeAfterglows.length - MAX_ELECTRIC_PATHS);
    }
  }

  function publishCortexMetrics(force = false) {
    const target = document.getElementById("fieldAria");
    if (!target) return;
    const now = performance.now();
    if (
      !force
      && now - lastCortexMetricsPublished < METRICS_PUBLISH_INTERVAL_MS
    ) {
      return;
    }
    lastCortexMetricsPublished = now;
    const painted = cortexMetrics.spread.filter(
      (row) => Number.isFinite(row.paintedAt),
    );
    const latencies = painted
      .map((row) => row.paintedAt - row.receivedAt)
      .sort((left, right) => left - right);
    const p95 = latencies.length
      ? latencies[Math.min(latencies.length - 1, Math.round((latencies.length - 1) * 0.95))]
      : 0;
    const frames = recentFrameDurations(120)
      .filter((value) => value > 0)
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
    target.dataset.transportReceived = String(cortexMetrics.transportReceived);
    target.dataset.transportPainted = String(cortexMetrics.transportPainted);
    target.dataset.processingEvents = String(cortexMetrics.processingEvents);
    target.dataset.activeProcessingLanes = String(
      cortexMetrics.activeProcessingLanes,
    );
    target.dataset.processingActivityConnected = String(
      cortexMetrics.processingActivityConnected,
    );
    target.dataset.captureEvents = String(cortexMetrics.captureEvents);
    target.dataset.ingestEvents = String(cortexMetrics.ingestEvents);
    target.dataset.applyEvents = String(cortexMetrics.applyEvents);
    target.dataset.simulationAwake = String(simulationAwake);
    target.dataset.simulationAlpha = alpha.toFixed(4);
    target.dataset.simulationTicks = String(simulationTicks);
    target.dataset.simulationSleepCount = String(simulationSleepCount);
    target.dataset.simulationSettledTicks = String(simulationSettledTicks);
    target.dataset.simulationMaxVelocity =
      simulationLastMaxVelocity.toFixed(4);
  }

  function publishVisualMetrics(time) {
    if (time - lastVisualMetricsPublished < 200) return;
    lastVisualMetricsPublished = time;
    publishCortexMetrics();
    const target = document.getElementById("fieldAria");
    if (!target) return;
    target.dataset.violetNodes = String(cortexMetrics.violetNodes);
    target.dataset.labelsPainted = String(cortexMetrics.labelsPainted);
    target.dataset.afterglowEdges = String(cortexMetrics.afterglowEdges);
    target.dataset.electricEdges = String(cortexMetrics.electricEdges);
    target.dataset.electricPeak = cortexMetrics.electricPeak.toFixed(3);
    target.dataset.flashPeak = cortexMetrics.flashPeak.toFixed(3);
    target.dataset.maxCoreScale = cortexMetrics.maxCoreScale.toFixed(3);
    target.dataset.maxGlowPadding =
      cortexMetrics.maxGlowPadding.toFixed(3);
    target.dataset.cameraTheta = camera.theta.toFixed(4);
    target.dataset.cameraPhi = camera.phi.toFixed(4);
    target.dataset.cameraDistance = camera.distance.toFixed(1);
    target.dataset.cameraPivotNodeIndex = String(camera.pivotNodeIndex);
    target.dataset.cameraPivotX = camera.pivotX.toFixed(2);
    target.dataset.cameraPivotY = camera.pivotY.toFixed(2);
    target.dataset.cameraPivotZ = camera.pivotZ.toFixed(2);
    target.dataset.cometTrailSegments = String(cortexMetrics.cometTrailSegments);
    target.dataset.cometHeads = String(cortexMetrics.cometHeads);
    target.dataset.cometImpacts = String(cortexMetrics.cometImpacts);
    target.dataset.maxCometHeadRadius =
      cortexMetrics.maxCometHeadRadius.toFixed(3);
    target.dataset.cometCenterX = cortexMetrics.cometCenterX.toFixed(1);
    target.dataset.cometCenterY = cortexMetrics.cometCenterY.toFixed(1);
    target.dataset.cometMinX = cortexMetrics.cometMinX.toFixed(1);
    target.dataset.cometMaxX = cortexMetrics.cometMaxX.toFixed(1);
    target.dataset.cometMinY = cortexMetrics.cometMinY.toFixed(1);
    target.dataset.cometMaxY = cortexMetrics.cometMaxY.toFixed(1);
    target.dataset.cometSafeLeft = cortexMetrics.cometSafeLeft.toFixed(1);
    target.dataset.cometSafeRight = cortexMetrics.cometSafeRight.toFixed(1);
    target.dataset.cometSafeTop = cortexMetrics.cometSafeTop.toFixed(1);
    target.dataset.cometSafeBottom = cortexMetrics.cometSafeBottom.toFixed(1);
    target.dataset.cometTurns = cortexMetrics.cometTurns.toFixed(2);
    target.dataset.explorationArcs = String(cortexMetrics.explorationArcs);
    target.dataset.consolidationEdges = String(cortexMetrics.consolidationEdges);
    target.dataset.triageCandidates = String(cortexMetrics.triageCandidates);
    target.dataset.generateParticles = String(cortexMetrics.generateParticles);
    target.dataset.consensusOrbits = String(cortexMetrics.consensusOrbits);
    target.dataset.processingNodeBlinks = String(
      cortexMetrics.processingNodeBlinks,
    );
    target.dataset.processingTargetNodeIndex = String(
      cortexMetrics.processingTargetNodeIndex,
    );
    target.dataset.formationOriginX = cortexMetrics.formationOriginX.toFixed(1);
    target.dataset.formationOriginY = cortexMetrics.formationOriginY.toFixed(1);
    target.dataset.formationTargetX = cortexMetrics.formationTargetX.toFixed(1);
    target.dataset.formationTargetY = cortexMetrics.formationTargetY.toFixed(1);
    target.dataset.transportElectricPeak =
      cortexMetrics.transportElectricPeak.toFixed(3);
    target.dataset.activeLabelLimit = String(ACTIVE_LABEL_LIMIT);
    target.dataset.electricPathLimit = String(MAX_ELECTRIC_PATHS);
    target.dataset.flashTiming =
      `${NODE_FLASH_ATTACK_MS}/${NODE_FLASH_HOLD_MS}/${NODE_FLASH_DECAY_MS}`;
    target.dataset.edgeAfterglowMs = String(EDGE_AFTERGLOW_MS);
    target.dataset.electricTravelMs =
      `${ELECTRIC_TRAVEL_MIN_MS}/${ELECTRIC_TRAVEL_MAX_MS}`;
  }

  function refreshNeighborPriorities(index) {
    const indexes = [...(neighbors[index] || [])];
    neighborsByConnectivity[index] = [...indexes].sort(
      (left, right) =>
        nodes[right].fanIn + nodes[right].fanOut
        - nodes[left].fanIn - nodes[left].fanOut,
    );
    neighborsByFanIn[index] = indexes.sort(
      (left, right) => nodes[right].fanIn - nodes[left].fanIn,
    );
  }

  function ensureActualEdge(event) {
    const source = byId.get(event.source_page_id);
    const target = byId.get(event.target_page_id);
    if (!source || !target) return -1;
    const pairKey = `${source.index}:${target.index}`;
    const existingEdgeIndex = edgeIndexByPair.get(pairKey);
    if (Number.isInteger(existingEdgeIndex)) return existingEdgeIndex;
    const edgeIndex = links.length;
    const link = {
      source: source.index,
      target: target.index,
      kind: 1,
      edgeType: event.edge_type || "field",
      eventOnly: true,
      typed: false,
      restLength: 58 + source.radius + target.radius,
    };
    links.push(link);
    simulationLinks.push(link);
    edgeIndexByPair.set(pairKey, edgeIndex);
    neighbors[source.index].add(target.index);
    neighbors[target.index].add(source.index);
    refreshNeighborPriorities(source.index);
    refreshNeighborPriorities(target.index);
    outgoing[source.index].push(edgeIndex);
    if (edgeIndex >= edgeState.length) {
      const expanded = new Uint8Array(Math.ceil(edgeState.length * 1.5));
      expanded.set(edgeState);
      edgeState = expanded;
    }
    edgeState[edgeIndex] = 2;
    stateDirty = true;
    reheat(0.18);
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

  function transportPageId(event) {
    return Array.isArray(event.page_ids) && event.page_ids.length
      ? String(event.page_ids[0])
      : "";
  }

  function updateMemoryIngress(event, phase) {
    const root = document.getElementById("memoryIngress");
    const state = document.getElementById("memoryIngressState");
    const detail = document.getElementById("memoryIngressDetail");
    if (!root || !state || !detail) return;
    ingressRevision += 1;
    const revision = ingressRevision;
    window.clearTimeout(ingressReset);
    root.dataset.phase = phase;
    const processing = event.kind === "processing";
    state.textContent = processing
      ? `${String(event.lane_key || "PROCESS").toUpperCase()} · ${phase.toUpperCase()}`
      : phase === "capture"
      ? "CAPTURING HOST MEMORY"
      : phase === "apply"
        ? "WRITING MEMORY GRAPH"
        : phase === "complete"
          ? "CONSOLIDATED"
          : `INGEST ${phase.toUpperCase()}`;
    if (processing) {
      detail.textContent = [event.model, event.role]
        .filter(Boolean)
        .map(String)
        .join(" · ") || String(event.label || "live processing");
    } else if (phase === "capture") {
      const bytes = Number(event.byte_count || 0);
      const count = Math.max(1, Number(event.raw_count || 1));
      const captureId = String(event.capture_id || "pending").slice(0, 12);
      detail.textContent = `${formatBytes(bytes)} · ${count} raw · ID ${captureId}`;
    } else {
      const pageId = transportPageId(event);
      detail.textContent = pageId
        ? `${String(event.operation || phase).toUpperCase()} · ${pageId}`
        : String(event.label || "local pipeline active");
    }
    ingressReset = window.setTimeout(() => {
      if (revision !== ingressRevision) return;
      root.dataset.phase = "idle";
      state.textContent = "IDLE";
      detail.textContent = "awaiting host capture";
    }, phase === "capture"
      ? CAPTURE_COMET_DURATION_MS + 400
      : phase === "apply" || phase === "complete"
        ? 6200
        : 4800);
  }

  function retireSupersededIngestEffects(pageId, now, channelKey) {
    transportEffects.forEach((effect) => {
      if (
        effect.phase === "capture"
        || effect.channelKey !== channelKey
        || (pageId && effect.pageId && effect.pageId !== pageId)
      ) return;
      const elapsed = Math.max(0, now - effect.startedAt);
      effect.duration = Math.min(effect.duration, elapsed + 280);
    });
  }

  function processingTargetNode(laneKey) {
    const candidates = nodes
      .filter((node) => !nodeState || nodeState[node.index] > 0)
      .sort(
        (left, right) =>
          right.fanIn + right.fanOut - left.fanIn - left.fanOut
          || left.id.localeCompare(right.id),
      )
      .slice(0, 12);
    if (!candidates.length) return null;
    const index = Math.floor(
      deterministicUnit(String(laneKey || "process"), 17) * candidates.length,
    );
    return candidates[index] || candidates[0];
  }

  function visualizeTransportEvent(event) {
    const phase = event.kind === "save"
      ? "capture"
      : String(event.phase || "generate");
    const now = performance.now();
    const pageId = transportPageId(event);
    const laneKey = String(event.lane_key || "");
    const node = pageId
      ? byId.get(pageId)
      : event.kind === "processing"
        ? processingTargetNode(laneKey)
        : null;
    const channelKey = String(
      event.channel_key
      || (event.kind === "processing"
        ? `processing:${event.lane_key || "unknown"}`
        : event.kind || "memory"),
    );
    if (phase !== "capture") {
      retireSupersededIngestEffects(pageId, now, channelKey);
    }
    const duration = phase === "capture"
      ? CAPTURE_COMET_DURATION_MS
      : phase === "apply" || phase === "complete"
        ? 3600
        : phase === "consensus"
          ? 2600
          : phase === "generate"
            ? 2300
            : 2100;
    transportEffects.push({
      kind: event.kind,
      phase,
      channelKey,
      laneKey,
      step: String(event.step || phase),
      pageId,
      nodeIndex: node?.index ?? -1,
      label: String(event.label || event.kind || "transport").slice(0, 160),
      captureId: String(event.capture_id || "").slice(0, 12),
      byteCount: Math.max(0, Number(event.byte_count || 0)),
      rawCount: Math.max(1, Number(event.raw_count || 1)),
      startedAt: now,
      duration,
      seq: ++cortexMetrics.transportReceived,
      demo: event.source === "demo",
      paintedAt: 0,
    });
    if (transportEffects.length > MAX_TRANSPORT_EFFECTS) {
      transportEffects.splice(0, transportEffects.length - MAX_TRANSPORT_EFFECTS);
    }
    if (phase === "capture") cortexMetrics.captureEvents += 1;
    else if (event.kind === "processing") cortexMetrics.processingEvents += 1;
    else cortexMetrics.ingestEvents += 1;
    if (phase === "apply" || phase === "complete") cortexMetrics.applyEvents += 1;
    crackle(
      phase === "apply" || phase === "complete"
        ? 0.12
        : phase === "capture"
          ? 0.055
          : 0.04 + Math.min(0.035, cortexMetrics.ingestEvents * 0.006),
    );
    updateMemoryIngress(event, phase);
    const prefix = event.source === "demo"
      ? "DEMO/REPLAY"
      : event.kind === "processing"
        ? "LIVE PROCESS"
        : "MEMORY I/O";
    flashTicker(`${prefix} · ${String(event.label || phase).toUpperCase()}`);
    const aria = document.getElementById("fieldAria");
    if (aria) {
      aria.textContent = `${phase}, ${event.label || "memory transport"}${pageId ? `, ${pageId}` : ""}`;
    }
    stimulateFromTransport(event);
    publishCortexMetrics();
  }

  function visualizeFieldEvent(event) {
    const now = performance.now();
    const node = effectNode(event);
    if (event.kind === "spread") {
      if (event.relation_id && relationById.has(event.relation_id)) {
        activeRelationIds.add(event.relation_id);
        window.setTimeout(() => {
          activeRelationIds.delete(event.relation_id);
          stateDirty = true;
        }, EDGE_AFTERGLOW_MS + ELECTRIC_TRAVEL_MAX_MS);
      }
      const edgeIndex = ensureActualEdge(event);
      if (edgeIndex < 0) {
        flashTicker(`◇ seq ${event.seq} · unmapped ${event.source_page_id}→${event.target_page_id}`);
        return;
      }
      const strength = Math.max(0, Math.min(1, Math.abs(event.delta)));
      queueElectricPulse({
        edgeIndex,
        startedAt: now,
        duration: electricTravelDuration(strength),
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
      const nodeDelta =
        event.kind === "stimulus"
          ? event.delta * NODE_STIMULUS_SCALE
          : event.delta;
      if (event.kind === "stimulus") {
        exciteNode(node, nodeDelta, now);
      }
      nodeEffects.push({
        nodeIndex: node.index,
        kind: event.kind,
        startedAt: now,
        duration: event.kind === "stimulus" ? 650 : 820,
        delta: Math.abs(nodeDelta),
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
      if (link.typed) return;
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
    drawTypedRelations();
  }

  function drawTypedRelations() {
    if (!relationsVisible) return;
    links.forEach((link, edgeIndex) => {
      if (!link.typed || edgeState[edgeIndex] === 0) return;
      const source = nodes[link.source];
      const target = nodes[link.target];
      if (!source || !target || source.viewDepth > 9e8 || target.viewDepth > 9e8) return;
      const focused = edgeState[edgeIndex] === 3;
      const active = activeRelationIds.has(link.relationId);
      const color = RELATION_STATUS_COLORS[link.lifecycle] || RGB_VIOLET;
      context.save();
      context.setLineDash(link.lifecycle === "authoritative" ? [] : [4, 5]);
      context.lineWidth = active ? 2 : focused ? 1.25 : 0.65;
      context.strokeStyle = rgba(
        color,
        (active ? 0.82 : focused ? 0.55 : 0.18) * fog((source.viewDepth + target.viewDepth) / 2),
      );
      context.beginPath();
      context.moveTo(source.screenX, source.screenY);
      context.lineTo(target.screenX, target.screenY);
      context.stroke();
      context.restore();
    });
  }

  function screenCross(origin, left, right) {
    return (
      (left.screenX - origin.screenX) * (right.screenY - origin.screenY)
      - (left.screenY - origin.screenY) * (right.screenX - origin.screenX)
    );
  }

  function convexHull(community) {
    const { points, lowerHull: lower, upperHull: upper, hull } = community;
    if (points.length < 3) return points;
    points.sort(
      (left, right) =>
        left.screenX - right.screenX || left.screenY - right.screenY,
    );
    lower.length = 0;
    for (let index = 0; index < points.length; index += 1) {
      const point = points[index];
      while (
        lower.length >= 2
        && screenCross(lower.at(-2), lower.at(-1), point) <= 0
      ) {
        lower.pop();
      }
      lower.push(point);
    }
    upper.length = 0;
    for (let index = points.length - 1; index >= 0; index -= 1) {
      const point = points[index];
      while (
        upper.length >= 2
        && screenCross(upper.at(-2), upper.at(-1), point) <= 0
      ) {
        upper.pop();
      }
      upper.push(point);
    }
    lower.pop();
    upper.pop();
    hull.length = 0;
    for (let index = 0; index < lower.length; index += 1) {
      hull.push(lower[index]);
    }
    for (let index = 0; index < upper.length; index += 1) {
      hull.push(upper[index]);
    }
    return hull;
  }

  function drawCommunityHulls() {
    if (!relationsVisible || camera.distance > 1900) return;
    communityHulls.forEach((community) => {
      const { points } = community;
      points.length = 0;
      for (let index = 0; index < community.members.length; index += 1) {
        const node = nodes[community.members[index]];
        if (node && nodeState[node.index] >= 2 && node.viewDepth <= 9e8) {
          points.push(node);
        }
      }
      if (points.length < 3) return;
      const hull = convexHull(community);
      if (hull.length < 3) return;
      context.save();
      context.beginPath();
      context.moveTo(hull[0].screenX, hull[0].screenY);
      for (let index = 1; index < hull.length; index += 1) {
        context.lineTo(hull[index].screenX, hull[index].screenY);
      }
      context.closePath();
      context.fillStyle = rgba(community.color, 0.018);
      context.strokeStyle = rgba(community.color, 0.10);
      context.lineWidth = 0.7;
      context.setLineDash([2, 7]);
      context.fill();
      context.stroke();
      context.restore();
    });
  }

  function projectPoint(x, y, z) {
    const localX = x - camera.pivotX;
    const localY = y - camera.pivotY;
    const localZ = z - camera.pivotZ;
    const rotatedX = localX * projectionCosTheta + localZ * projectionSinTheta;
    const rotatedZ = -localX * projectionSinTheta + localZ * projectionCosTheta;
    const rotatedY = localY * projectionCosPhi - rotatedZ * projectionSinPhi;
    const depthZ = localY * projectionSinPhi + rotatedZ * projectionCosPhi;
    const viewDepth = camera.distance - depthZ;
    if (viewDepth < 60) return null;
    const scale = focalLength / viewDepth;
    return {
      x: rotatedX * scale + projectionCenterX,
      y: rotatedY * scale + projectionCenterY,
      scale,
      viewDepth,
    };
  }

  function electricPathPoints(source, target, edgeId, seq, phase = 0) {
    const dx = target.screenX - source.screenX;
    const dy = target.screenY - source.screenY;
    const length = Math.hypot(dx, dy) || 1;
    const normalX = -dy / length;
    const normalY = dx / length;
    const pointCount = 10;
    const jitterScale = Math.min(11, 3.5 + length * 0.018);
    return Array.from({ length: pointCount }, (_value, pointIndex) => {
      const unit = pointIndex / (pointCount - 1);
      const envelope = Math.sin(Math.PI * unit);
      const jitter =
        (deterministicUnit(
          edgeId,
          seq * 131 + pointIndex * 31 + phase * 997,
        ) - 0.5)
        * jitterScale
        * envelope;
      return {
        x: source.screenX + dx * unit + normalX * jitter,
        y: source.screenY + dy * unit + normalY * jitter,
      };
    });
  }

  function electricPathPrefix(points, progress) {
    const capped = clamp(progress);
    if (!points.length || capped <= 0) return [];
    if (capped >= 1) return points;
    const scaled = capped * (points.length - 1);
    const segment = Math.floor(scaled);
    const partial = points.slice(0, segment + 1);
    const unit = scaled - segment;
    const start = points[segment];
    const end = points[Math.min(points.length - 1, segment + 1)];
    partial.push({
      x: start.x + (end.x - start.x) * unit,
      y: start.y + (end.y - start.y) * unit,
    });
    return partial;
  }

  function traceElectricPath(points) {
    if (points.length < 2) return false;
    context.beginPath();
    points.forEach((point, pointIndex) => {
      if (pointIndex) context.lineTo(point.x, point.y);
      else context.moveTo(point.x, point.y);
    });
    return true;
  }

  function completePulse(pulse, target, time) {
    target.arrivedAt = time;
    exciteNode(target, pulse.delta * NODE_ARRIVAL_SCALE, time);
    nodeEffects.push({
      nodeIndex: target.index,
      kind: "arrival",
      startedAt: time,
      duration: 620,
      delta: pulse.delta * NODE_ARRIVAL_SCALE,
      seq: pulse.seq,
      demo: pulse.demo,
    });
    edgeAfterglows.push({
      edgeIndex: pulse.edgeIndex,
      startedAt: time,
      duration: EDGE_AFTERGLOW_MS,
      delta: pulse.delta,
      seq: pulse.seq,
      demo: pulse.demo,
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
      const edgeId = `${source.id}>${target.id}:${link.edgeType || "field"}`;
      const phase = Math.floor((time - afterglow.startedAt) / 135);
      const points = electricPathPoints(
        source,
        target,
        edgeId,
        afterglow.seq,
        phase,
      );
      if (!traceElectricPath(points)) continue;
      const travelColor = afterglow.demo ? RGB_FIRE : RGB_ELECTRIC;
      const travelHot = afterglow.demo ? RGB_HOT : RGB_HOT;
      context.strokeStyle = rgba(
        travelColor,
        fade * (0.22 + afterglow.delta * 0.48),
      );
      context.lineWidth = 3.2 + afterglow.delta * 5.2;
      context.stroke();
      traceElectricPath(points);
      context.strokeStyle = rgba(
        travelHot,
        fade * (0.42 + afterglow.delta * 0.5),
      );
      context.lineWidth = 0.8 + afterglow.delta * 1.45;
      context.stroke();
      const glowSize = 8 + afterglow.delta * 9;
      context.globalAlpha = fade * (0.12 + afterglow.delta * 0.24);
      context.drawImage(
        afterglow.demo ? glowFire : glowElectric,
        target.screenX - glowSize / 2,
        target.screenY - glowSize / 2,
        glowSize,
        glowSize,
      );
      context.globalAlpha = 1;
      cortexMetrics.afterglowEdges += 1;
      cortexMetrics.electricEdges += 1;
      cortexMetrics.electricPeak = Math.max(
        cortexMetrics.electricPeak,
        fade * (0.55 + afterglow.delta * 0.45),
      );
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
      const dx = target.screenX - source.screenX;
      const dy = target.screenY - source.screenY;
      const length = Math.hypot(dx, dy) || 1;
      const normalX = -dy / length;
      const normalY = dx / length;
      const depthFade = fog((source.viewDepth + target.viewDepth) / 2);
      const phase = Math.floor((time - pulse.startedAt) / 55);
      const fullPath = electricPathPoints(
        source,
        target,
        edgeId,
        pulse.seq,
        phase,
      );
      const energizedPath = electricPathPrefix(fullPath, progress);
      const head = energizedPath.at(-1) || fullPath[0];
      const intensity =
        (0.48 + pulse.delta * 0.52)
        * depthFade;
      context.lineCap = "round";
      context.lineJoin = "round";
      const travelColor = pulse.demo ? RGB_FIRE : RGB_ELECTRIC;
      if (traceElectricPath(fullPath)) {
        context.strokeStyle = rgba(
          travelColor,
          (0.14 + pulse.delta * 0.28)
          * depthFade
          * (0.55 + progress * 0.45),
        );
        context.lineWidth = 3.8 + pulse.delta * 5.8;
        context.stroke();
      }
      if (traceElectricPath(energizedPath)) {
        context.strokeStyle = rgba(travelColor, intensity);
        context.lineWidth = 4.6 + pulse.delta * 5.4;
        context.stroke();
        traceElectricPath(energizedPath);
        context.strokeStyle = rgba(
          RGB_HOT,
          (0.72 + pulse.delta * 0.28) * depthFade,
        );
        context.lineWidth = 0.9 + pulse.delta * 1.55;
        context.stroke();
      }
      const branchLength = 4 + pulse.delta * 8;
      const branchSign = deterministicUnit(edgeId, pulse.seq + 991) > 0.5 ? 1 : -1;
      context.strokeStyle = rgba(travelColor, 0.45 + pulse.delta * 0.35);
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
        context.strokeStyle = rgba(travelColor, 0.75 + pulse.delta * 0.25);
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
        pulse.demo ? glowFire : glowElectric,
        head.x - glowSize / 2,
        head.y - glowSize / 2,
        glowSize,
        glowSize,
      );
      context.globalAlpha = 1;
      cortexMetrics.electricEdges += 1;
      cortexMetrics.electricPeak = Math.max(
        cortexMetrics.electricPeak,
        intensity,
      );
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
      const baseRadius = Math.max(0.75, node.radius * node.screenScale);
      const radius = baseRadius * NODE_CORE_SCALE;
      cortexMetrics.maxCoreScale = Math.max(
        cortexMetrics.maxCoreScale,
        radius / baseRadius,
      );
      if (isFieldActive) {
        drawCompactGlow(
          glowViolet,
          node.screenX,
          node.screenY,
          radius,
          1.4 + fieldActivation * 2.2,
          (0.14 + fieldActivation * 0.34) * depthFade * dim,
        );
        cortexMetrics.violetNodes += 1;
      }
      if (excitation > 0.01) {
        drawCompactGlow(
          excitation > 0.68 ? glowHot : glowViolet,
          node.screenX,
          node.screenY,
          radius,
          1.5 + excitation * 2.5,
          Math.min(0.82, 0.12 + excitation * 0.7) * depthFade * dim,
        );
        cortexMetrics.flashPeak = Math.max(
          cortexMetrics.flashPeak,
          excitation,
        );
      } else if (state === 3 || node.fanIn >= 38) {
        drawCompactGlow(
          glowSteel,
          node.screenX,
          node.screenY,
          radius,
          state === 3 ? 2.6 : 1.4,
          (state === 3 ? 0.38 : 0.1) * depthFade * dim,
        );
      }

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
          (1 - smoothstep(progress)) * 0.42 * depthFade,
        );
        context.lineWidth = 1;
        context.beginPath();
        context.arc(
          node.screenX,
          node.screenY,
          radius + NODE_EFFECT_MAX_PADDING_PX,
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
      const radius =
        Math.max(0.75, node.radius * node.screenScale) * NODE_CORE_SCALE;
      const fade = 1 - progress;
      const effectPadding = Math.min(
        NODE_EFFECT_MAX_PADDING_PX,
        1.2 + effect.delta * 1.8,
      );
      let effectColor = RGB_FIRE;
      let effectGlow = glowFire;
      if (effect.kind === "stimulus") {
        effectColor = RGB_FIRE;
        effectGlow = glowFire;
      } else if (effect.kind === "inhibit" || effect.kind === "reject") {
        effectColor = RGB_INHIBIT;
        effectGlow = glowInhibit;
      } else if (effect.kind === "commit_queued" || effect.kind === "commit_applied") {
        effectColor = RGB_COMMIT;
        effectGlow = glowCommit;
      } else if (effect.kind === "fault") {
        effectColor = RGB_FAULT;
        effectGlow = glowFault;
      } else if (effect.kind === "arrival") {
        effectColor = RGB_ELECTRIC;
        effectGlow = glowElectric;
      }
      drawCompactGlow(
        effectGlow,
        node.screenX,
        node.screenY,
        radius,
        effectPadding,
        fade * (0.12 + effect.delta * 0.48),
      );
      context.fillStyle = rgba(
        effectColor,
        fade * Math.min(0.9, 0.38 + effect.delta * 0.46),
      );
      context.beginPath();
      context.arc(node.screenX, node.screenY, radius, 0, Math.PI * 2);
      context.fill();
    }
    context.globalCompositeOperation = "source-over";
  }

  function transportTarget(effect) {
    const node = nodes[effect.nodeIndex];
    const processingTargetLocked = effect.kind === "processing" && Boolean(node);
    if (
      node
      && node.viewDepth <= 9e8
      && (
        processingTargetLocked
        || (
          node.screenX > 20
          && node.screenX < width - 20
          && node.screenY > 40
          && node.screenY < height - 40
        )
      )
    ) {
      return { x: node.screenX, y: node.screenY, node };
    }
    if (processingTargetLocked) return null;
    transportFallbacks.length = 0;
    for (let index = 0; index < nodesByConnectivity.length; index += 1) {
      const candidate = nodesByConnectivity[index];
      if (visibleMemoryNode(candidate)) transportFallbacks.push(candidate);
      if (transportFallbacks.length >= 12) break;
    }
    const fallbackIndex = effect.kind === "processing" && transportFallbacks.length
      ? Math.floor(
          deterministicUnit(effect.channelKey, effect.laneKey.length + 17)
          * transportFallbacks.length,
        )
      : 0;
    const fallback = transportFallbacks[fallbackIndex] || visibleHub();
    if (fallback) {
      return { x: fallback.screenX, y: fallback.screenY, node: fallback };
    }
    return {
      x: width * 0.5,
      y: height * 0.52,
      node: null,
    };
  }

  function transportElectricPoints(source, target, effect, lane, time) {
    const phase = Math.floor((time - effect.startedAt) / 68);
    return electricPathPoints(
      { screenX: source.x, screenY: source.y },
      { screenX: target.x, screenY: target.y },
      `transport:${effect.phase}:${lane}:${effect.pageId || effect.captureId}`,
      effect.seq * 41 + lane * 17,
      phase + lane,
    );
  }

  function strokeTransportBolt(
    points,
    progress,
    color,
    intensity,
    outerWidth,
    innerWidth,
  ) {
    const prefix = electricPathPrefix(points, progress);
    const head = prefix.at(-1) || points[0];
    if (traceElectricPath(points)) {
      context.strokeStyle = rgba(color, intensity * 0.16);
      context.lineWidth = outerWidth + 3.4;
      context.stroke();
    }
    if (traceElectricPath(prefix)) {
      context.strokeStyle = rgba(color, intensity * 0.94);
      context.lineWidth = outerWidth;
      context.stroke();
      traceElectricPath(prefix);
      context.strokeStyle = rgba(RGB_HOT, intensity * 0.88);
      context.lineWidth = innerWidth;
      context.stroke();
    }
    context.fillStyle = rgba(RGB_HOT, intensity);
    context.beginPath();
    context.arc(head.x, head.y, Math.max(1.2, innerWidth * 1.25), 0, Math.PI * 2);
    context.fill();
    cortexMetrics.transportElectricPeak = Math.max(
      cortexMetrics.transportElectricPeak,
      intensity,
    );
    return head;
  }

  function captureCometCount(effect) {
    const byteTier = effect.byteCount >= 1024 * 1024
      ? 4
      : effect.byteCount >= 128 * 1024
        ? 3
        : effect.byteCount >= 32 * 1024
          ? 2
          : effect.byteCount >= 8 * 1024
            ? 1
            : 0;
    const rawTier = Math.max(0, Math.ceil(Math.log2(effect.rawCount + 1)) - 1);
    return Math.round(clamp(3 + Math.max(byteTier, rawTier), 3, 7));
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

  function memoryStarGeometry() {
    const shortest = Math.max(240, Math.min(width, height));
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
    const centerX = totalWeight ? weightedX / totalWeight : width / 2;
    const centerY = totalWeight ? weightedY / totalWeight : height / 2;
    let weightedDistance = 0;
    nodes.forEach((node) => {
      if (!visibleMemoryNode(node)) return;
      const weight = memoryNodeWeight(node);
      weightedDistance += Math.hypot(
        node.screenX - centerX,
        node.screenY - centerY,
      ) * weight;
    });
    return {
      x: centerX,
      y: centerY,
      radius: clamp(
        totalWeight ? weightedDistance / totalWeight * 1.55 : shortest * 0.31,
        shortest * 0.17,
        shortest * 0.36,
      ),
    };
  }

  function captureSafeRect() {
    const top = clamp(height * 0.28, 156, 220);
    return {
      left: 34,
      right: Math.max(35, width - 34),
      top,
      bottom: Math.max(top + 24, height - 46),
    };
  }

  function recordCaptureCometBounds(points) {
    points.forEach((point) => {
      cortexMetrics.cometMinX = Math.min(cortexMetrics.cometMinX, point.x);
      cortexMetrics.cometMaxX = Math.max(cortexMetrics.cometMaxX, point.x);
      cortexMetrics.cometMinY = Math.min(cortexMetrics.cometMinY, point.y);
      cortexMetrics.cometMaxY = Math.max(cortexMetrics.cometMaxY, point.y);
    });
  }

  function traceCaptureCometPoints(points, start = 0, end = points.length - 1) {
    if (end <= start || !points[start]) return false;
    context.beginPath();
    context.moveTo(points[start].x, points[start].y);
    for (let index = start + 1; index <= end; index += 1) {
      context.lineTo(points[index].x, points[index].y);
    }
    return true;
  }

  function drawCaptureCometTrail(points, comet, fade) {
    const segmentCount = points.length - 1;
    const widthScale = comet === 0 ? 1 : 0.62;
    if (traceCaptureCometPoints(points)) {
      context.strokeStyle = rgba(RGB_CAPTURE, fade * 0.15);
      context.lineWidth = 5.2 * widthScale;
      context.stroke();
    }
    const bandCount = 5;
    for (let band = 0; band < bandCount; band += 1) {
      const start = Math.floor(segmentCount * band / bandCount);
      const end = Math.max(start + 1, Math.floor(segmentCount * (band + 1) / bandCount));
      const tailRatio = (start + end) / (segmentCount * 2);
      if (!traceCaptureCometPoints(points, start, end)) continue;
      context.strokeStyle = rgba(
        mix(RGB_CAPTURE, RGB_HOT, 1 - tailRatio),
        fade * Math.pow(1 - tailRatio, 1.65) * 0.92,
      );
      context.lineWidth = Math.max(0.35, (1.45 - tailRatio) * widthScale);
      context.stroke();
    }
    cortexMetrics.cometTrailSegments += segmentCount;
  }

  function drawCaptureCometHead(comet, head, fade) {
    const headRadius =
      (comet === 0 ? 2.2 : 1.35)
      + head.journey * (comet === 0 ? 2.4 : 1.2);
    const glowSize = headRadius * (comet === 0 ? 8.5 : 6.5);
    context.globalAlpha = fade * (comet === 0 ? 0.78 : 0.54);
    context.drawImage(
      glowCapture,
      head.x - glowSize / 2,
      head.y - glowSize / 2,
      glowSize,
      glowSize,
    );
    context.globalAlpha = 1;
    context.fillStyle = rgba(RGB_HOT, fade * 0.96);
    context.beginPath();
    context.arc(head.x, head.y, headRadius, 0, Math.PI * 2);
    context.fill();
    cortexMetrics.cometHeads += 1;
    cortexMetrics.maxCometHeadRadius = Math.max(
      cortexMetrics.maxCometHeadRadius,
      headRadius,
    );
    cortexMetrics.transportElectricPeak = Math.max(
      cortexMetrics.transportElectricPeak,
      fade * 0.96,
    );
  }

  function drawCaptureImpact(comet, localProgress, fade, star) {
    if (localProgress <= 0.91) return;
    const impact = clamp((localProgress - 0.91) / 0.09);
    const impactFade = fade * (1 - smoothstep(impact));
    const impactRadius = Math.min(
      comet === 0 ? 30 : 18,
      star.radius * (comet === 0 ? 0.14 : 0.085),
    );
    context.strokeStyle = rgba(RGB_CAPTURE, impactFade * 0.82);
    context.lineWidth = 1.15 - impact * 0.5;
    context.beginPath();
    context.arc(
      star.x,
      star.y,
      4 + Math.sqrt(impact) * impactRadius,
      0,
      Math.PI * 2,
    );
    context.stroke();
    context.fillStyle = rgba(RGB_HOT, impactFade * 0.9);
    context.beginPath();
    context.arc(star.x, star.y, Math.max(0.7, 2.8 - impact * 1.9), 0, Math.PI * 2);
    context.fill();
    cortexMetrics.cometImpacts += 1;
  }

  function captureCometOrbit(effect, comet, star, safe) {
    const seed = deterministicUnit(
      effect.captureId || effect.label,
      effect.seq * 101 + comet * 37,
    );
    const headMargin = comet === 0 ? 22 : 14;
    const availableX = Math.max(
      8,
      Math.min(star.x - safe.left, safe.right - star.x) - headMargin,
    );
    const availableY = Math.max(
      8,
      Math.min(star.y - safe.top, safe.bottom - star.y) - headMargin,
    );
    return {
      radiusX: availableX * (0.9 + seed * 0.09),
      radiusY: availableY * (0.84 + seed * 0.14),
      startAngle: -0.62 + comet * 2.23 + (seed - 0.5) * 0.7,
      turns: 2.85 + seed * 0.25 + (comet === 0 ? 0.1 : 0),
    };
  }

  function captureCometPoint(orbit, localProgress, star) {
    const journey = clamp(localProgress / 0.91);
    const accelerated = Math.pow(journey, 1.72);
    const angle = orbit.startAngle + accelerated * orbit.turns * Math.PI * 2;
    const contraction = 1 - smoothstep(accelerated);
    return {
      x: star.x + Math.cos(angle) * orbit.radiusX * contraction,
      y: star.y + Math.sin(angle) * orbit.radiusY * contraction,
      journey,
    };
  }

  function drawCaptureComet(effect, comet, progress, fade, star, safe) {
    const count = captureCometCount(effect);
    const delay = comet * Math.min(0.055, 0.22 / Math.max(1, count - 1));
    const localProgress = clamp((progress - delay) / (1 - delay));
    if (progress < delay) return;

    const orbit = captureCometOrbit(effect, comet, star, safe);
    if (comet === 0) cortexMetrics.cometTurns = orbit.turns;
    const head = captureCometPoint(orbit, localProgress, star);
    const tailSpan = 0.075 + head.journey * (comet === 0 ? 0.19 : 0.135);
    const segmentCount = comet === 0 ? 30 : 20;
    const points = [head];
    for (let segment = 1; segment <= segmentCount; segment += 1) {
      const tailRatio = segment / segmentCount;
      const tailProgress = Math.max(0, localProgress - tailSpan * tailRatio);
      points.push(captureCometPoint(orbit, tailProgress, star));
    }
    recordCaptureCometBounds(points);
    drawCaptureCometTrail(points, comet, fade);
    drawCaptureCometHead(comet, head, fade);
    drawCaptureImpact(comet, localProgress, fade, star);
  }

  function drawCaptureComets(effect, progress, fade) {
    const star = memoryStarGeometry();
    const safe = captureSafeRect();
    cortexMetrics.cometCenterX = star.x;
    cortexMetrics.cometCenterY = star.y;
    cortexMetrics.cometSafeLeft = safe.left;
    cortexMetrics.cometSafeRight = safe.right;
    cortexMetrics.cometSafeTop = safe.top;
    cortexMetrics.cometSafeBottom = safe.bottom;
    const count = captureCometCount(effect);
    for (let comet = count - 1; comet >= 0; comet -= 1) {
      drawCaptureComet(effect, comet, progress, fade, star, safe);
    }
  }

  function ingestFormationCandidates(target, count, includeTarget = false) {
    formationCandidates.length = 0;
    formationCandidateGeneration = (formationCandidateGeneration + 1) >>> 0;
    if (!formationCandidateGeneration) {
      formationCandidateMarks.fill(0);
      formationCandidateGeneration = 1;
    }
    const add = (node) => {
      if (
        !node
        || formationCandidates.length >= count
        || formationCandidateMarks[node.index] === formationCandidateGeneration
        || !visibleMemoryNode(node)
      ) {
        return;
      }
      formationCandidateMarks[node.index] = formationCandidateGeneration;
      formationCandidates.push(node);
    };
    if (includeTarget) add(target.node);
    if (target.node) {
      const neighborIndexes = neighborsByConnectivity[target.node.index] || [];
      for (let index = 0; index < neighborIndexes.length; index += 1) {
        add(nodes[neighborIndexes[index]]);
        if (formationCandidates.length >= count) break;
      }
    }
    if (formationCandidates.length < count) {
      for (let index = 0; index < nodesByConnectivity.length; index += 1) {
        add(nodesByConnectivity[index]);
        if (formationCandidates.length >= count) break;
      }
    }
    return formationCandidates;
  }

  function drawFormationNodeGlow(node, color, glow, intensity, padding = 2.4) {
    const radius = Math.max(0.75, node.radius * node.screenScale) * NODE_CORE_SCALE;
    drawCompactGlow(glow, node.screenX, node.screenY, radius, padding, intensity * 0.58);
    context.fillStyle = rgba(color, intensity * 0.9);
    context.beginPath();
    context.arc(node.screenX, node.screenY, radius, 0, Math.PI * 2);
    context.fill();
  }

  function formationLabel(effect, fallback) {
    if (effect.kind !== "processing") return fallback;
    return `${effect.laneKey || "PROCESS"} · ${effect.step || effect.phase}`
      .toUpperCase();
  }

  function drawTriageFormation(effect, progress, fade, star, target) {
    const wave = reducedMotion.matches || !motionEnabled ? 0.68 : smoothstep(progress);
    const radius = star.radius * (0.12 + wave * 1.18);
    cortexMetrics.formationOriginX = star.x;
    cortexMetrics.formationOriginY = star.y;
    context.strokeStyle = rgba(RGB_FIRE, fade * (0.82 - wave * 0.42));
    context.lineWidth = 1.35;
    context.beginPath();
    context.arc(star.x, star.y, radius, 0, Math.PI * 2);
    context.stroke();
    context.strokeStyle = rgba(RGB_HOT, fade * (0.42 - wave * 0.22));
    context.lineWidth = 0.7;
    context.beginPath();
    context.arc(star.x, star.y, radius * 0.82, 0, Math.PI * 2);
    context.stroke();
    ingestFormationCandidates(target, 8, true).forEach((node, index) => {
      const distance = Math.hypot(node.screenX - star.x, node.screenY - star.y);
      const arrival = clamp((radius - distance + 34) / 58);
      const shimmer = 0.62 + 0.38 * Math.sin(progress * 22 - index * 1.7);
      const intensity = fade * arrival * shimmer;
      if (intensity <= 0.04) return;
      drawFormationNodeGlow(node, RGB_FIRE, glowFire, intensity);
      cortexMetrics.triageCandidates += 1;
    });
    context.fillStyle = rgba(RGB_FIRE, fade * 0.88);
    context.textAlign = "center";
    context.fillText(
      formationLabel(effect, "EVALUATING CANDIDATES"),
      star.x,
      star.y - 12,
    );
  }

  function quadraticPoint(source, control, target, progress) {
    const inverse = 1 - progress;
    return {
      x: inverse * inverse * source.x + 2 * inverse * progress * control.x + progress * progress * target.x,
      y: inverse * inverse * source.y + 2 * inverse * progress * control.y + progress * progress * target.y,
    };
  }

  function drawGenerateFormation(effect, progress, fade, target) {
    const candidates = ingestFormationCandidates(target, 5, false);
    cortexMetrics.formationTargetX = target.x;
    cortexMetrics.formationTargetY = target.y;
    candidates.forEach((node, branch) => {
      const source = { x: node.screenX, y: node.screenY };
      const bend = (branch % 2 ? -1 : 1) * (24 + branch * 5);
      const control = {
        x: (source.x + target.x) / 2 + bend,
        y: (source.y + target.y) / 2 - 34 + branch * 7,
      };
      context.strokeStyle = rgba(RGB_VIOLET, fade * 0.16);
      context.lineWidth = 0.7;
      context.beginPath();
      context.moveTo(source.x, source.y);
      context.quadraticCurveTo(control.x, control.y, target.x, target.y);
      context.stroke();
      for (let particle = 0; particle < 3; particle += 1) {
        const delay = branch * 0.045 + particle * 0.17;
        const travel = reducedMotion.matches || !motionEnabled
          ? 0.72
          : clamp((progress - delay) / Math.max(0.1, 0.82 - delay));
        if (travel <= 0 || travel >= 1) continue;
        const point = quadraticPoint(source, control, target, smoothstep(travel));
        const tail = quadraticPoint(source, control, target, smoothstep(Math.max(0, travel - 0.055)));
        context.strokeStyle = rgba(RGB_VIOLET, fade * (0.34 + travel * 0.48));
        context.lineWidth = 1.15;
        context.beginPath();
        context.moveTo(tail.x, tail.y);
        context.lineTo(point.x, point.y);
        context.stroke();
        context.fillStyle = rgba(RGB_HOT, fade * 0.92);
        context.beginPath();
        context.arc(point.x, point.y, 1.15, 0, Math.PI * 2);
        context.fill();
        cortexMetrics.generateParticles += 1;
      }
    });
    const protoPulse = 0.72 + 0.28 * Math.sin(progress * Math.PI * 10);
    context.strokeStyle = rgba(RGB_VIOLET, fade * protoPulse * 0.88);
    context.lineWidth = 1.2;
    context.beginPath();
    context.arc(target.x, target.y, 7 + progress * 3, 0, Math.PI * 2);
    context.stroke();
    context.fillStyle = rgba(RGB_VIOLET, fade * 0.34);
    context.beginPath();
    context.arc(target.x, target.y, 4.2, 0, Math.PI * 2);
    context.fill();
    context.fillStyle = rgba(RGB_VIOLET, fade * 0.9);
    context.textAlign = "left";
    context.fillText(
      formationLabel(effect, "SYNTHESIZING MEMORY"),
      target.x + 12,
      target.y - 10,
    );
  }

  function drawConsensusFormation(effect, time, progress, fade, target) {
    const collapse = reducedMotion.matches || !motionEnabled ? 0.38 : smoothstep(progress);
    const orbitRadius = 35 - collapse * 22;
    cortexMetrics.formationTargetX = target.x;
    cortexMetrics.formationTargetY = target.y;
    for (let orbit = 0; orbit < 3; orbit += 1) {
      const rotation = orbit * Math.PI / 3 + 0.24;
      context.save();
      context.translate(target.x, target.y);
      context.rotate(rotation);
      context.scale(1, 0.42 + orbit * 0.08);
      context.strokeStyle = rgba(RGB_CONSENSUS, fade * (0.3 + orbit * 0.12));
      context.lineWidth = 1.05;
      context.beginPath();
      context.arc(0, 0, orbitRadius + orbit * 5, 0, Math.PI * 2);
      context.stroke();
      context.restore();
      const angle = (time - effect.startedAt) * 0.0045 * (orbit % 2 ? -1 : 1) + orbit * 2.1;
      const beadRadius = orbitRadius + orbit * 5;
      const cos = Math.cos(angle) * beadRadius;
      const sin = Math.sin(angle) * beadRadius * (0.42 + orbit * 0.08);
      const beadX = target.x + cos * Math.cos(rotation) - sin * Math.sin(rotation);
      const beadY = target.y + cos * Math.sin(rotation) + sin * Math.cos(rotation);
      context.fillStyle = rgba(RGB_HOT, fade * 0.95);
      context.beginPath();
      context.arc(beadX, beadY, 1.8, 0, Math.PI * 2);
      context.fill();
      cortexMetrics.consensusOrbits += 1;
    }
    drawCompactGlow(glowFire, target.x, target.y, 2.4, 3.2, fade * (0.28 + collapse * 0.55));
    context.fillStyle = rgba(RGB_CONSENSUS, fade * 0.92);
    context.textAlign = "left";
    context.fillText(
      formationLabel(effect, "LOCAL CONSENSUS"),
      target.x + 12,
      target.y - 10,
    );
  }

  function drawProcessingNodeBlink(effect, time, fade, target) {
    if (!target.node) return;
    const radius = Math.max(
      0.75,
      target.node.radius * target.node.screenScale,
    ) * NODE_CORE_SCALE;
    const blink = reducedMotion.matches || !motionEnabled
      ? 0.72
      : 0.16 + (0.84 * (0.5 + 0.5 * Math.sin(time * 0.009)));
    cortexMetrics.processingNodeBlinks += 1;
    cortexMetrics.processingTargetNodeIndex = target.node.index;
    cortexMetrics.formationTargetX = target.x;
    cortexMetrics.formationTargetY = target.y;
    drawCompactGlow(
      glowElectric,
      target.x,
      target.y,
      radius,
      2.4,
      fade * (0.22 + blink * 0.58),
    );
    context.fillStyle = rgba(
      RGB_ELECTRIC,
      fade * (0.34 + blink * 0.62),
    );
    context.beginPath();
    context.arc(target.x, target.y, radius, 0, Math.PI * 2);
    context.fill();
  }

  function visibleConsolidationNeighbors(node) {
    consolidationCandidates.length = 0;
    if (!node) return consolidationCandidates;
    const neighborIndexes = neighborsByFanIn[node.index] || [];
    for (let index = 0; index < neighborIndexes.length; index += 1) {
      const neighbor = nodes[neighborIndexes[index]];
      if (
        neighbor
        && nodeState[neighbor.index] > 0
        && neighbor.viewDepth <= 9e8
        && neighbor.screenX > 20
        && neighbor.screenX < width - 20
        && neighbor.screenY > 40
        && neighbor.screenY < height - 40
      ) {
        consolidationCandidates.push(neighbor);
        if (consolidationCandidates.length >= 6) break;
      }
    }
    return consolidationCandidates;
  }

  function drawApplyFormation(effect, time, progress, fade, target) {
    const arrival = reducedMotion.matches || !motionEnabled
      ? 0.72
      : clamp(progress * 1.35);
    cortexMetrics.formationTargetX = target.x;
    cortexMetrics.formationTargetY = target.y;
    if (arrival > 0) {
      visibleConsolidationNeighbors(target.node).forEach((neighbor, branch) => {
        const branchProgress = reducedMotion.matches || !motionEnabled
          ? 0.8
          : clamp(arrival * 1.8 - branch * 0.085);
        const points = transportElectricPoints(
          target,
          { x: neighbor.screenX, y: neighbor.screenY },
          effect,
          branch + 1,
          time,
        );
        strokeTransportBolt(
          points,
          branchProgress,
          RGB_COMMIT,
          fade * (0.54 - branch * 0.035),
          2.25,
          0.5,
        );
        cortexMetrics.consolidationEdges += 1;
      });
    }
    const radius = target.node
      ? Math.max(0.75, target.node.radius * target.node.screenScale) * NODE_CORE_SCALE
      : 2.2;
    drawCompactGlow(
      glowCommit,
      target.x,
      target.y,
      radius,
      3,
      fade * (0.38 + arrival * 0.45),
    );
    context.strokeStyle = rgba(RGB_COMMIT, fade * (0.55 + arrival * 0.4));
    context.lineWidth = 1.15;
    context.beginPath();
    for (let corner = 0; corner < 6; corner += 1) {
      const angle = corner * Math.PI / 3 - Math.PI / 6;
      const x = target.x + Math.cos(angle) * (radius + 3.5);
      const y = target.y + Math.sin(angle) * (radius + 3.5);
      if (corner === 0) context.moveTo(x, y);
      else context.lineTo(x, y);
    }
    context.closePath();
    context.stroke();
    if (!target.node) {
      context.fillStyle = rgba(RGB_COMMIT, fade * 0.88);
      context.beginPath();
      context.arc(target.x, target.y, radius, 0, Math.PI * 2);
      context.fill();
    }
    context.fillStyle = rgba(RGB_COMMIT, fade * 0.9);
    context.textAlign = "left";
    context.fillText(
      formationLabel(
        effect,
        `${effect.phase === "complete" ? "LOCKED" : "WRITE"} ${effect.pageId || "MEMORY"}`,
      ).slice(0, 42),
      target.x + 10,
      target.y - 9,
    );
  }

  function drawTransportEffects(time) {
    context.save();
    context.globalCompositeOperation = "lighter";
    context.lineCap = "round";
    context.lineJoin = "round";
    context.font = "8px ui-monospace, SFMono-Regular, Menlo, monospace";
    let memoryStar = null;
    for (let index = transportEffects.length - 1; index >= 0; index -= 1) {
      const effect = transportEffects[index];
      const rawProgress = (time - effect.startedAt) / effect.duration;
      if (rawProgress >= 1) {
        transportEffects.splice(index, 1);
        continue;
      }
      if (rawProgress < 0) continue;
      const progress = reducedMotion.matches || !motionEnabled
        ? 0.68
        : clamp(rawProgress);
      const fade = effect.phase === "capture"
        ? 1 - smoothstep(clamp((rawProgress - 0.91) / 0.09))
        : 1 - smoothstep(clamp(rawProgress));
      if (effect.phase === "capture") {
        drawCaptureComets(effect, progress, fade);
      } else {
        const target = transportTarget(effect);
        if (!target) continue;
        if (effect.phase === "apply" || effect.phase === "complete") {
          drawApplyFormation(effect, time, progress, fade, target);
        } else if (effect.phase === "consensus") {
          if (effect.kind === "processing" && effect.laneKey === "ingest") {
            drawProcessingNodeBlink(effect, time, fade, target);
          } else {
            drawConsensusFormation(effect, time, progress, fade, target);
          }
        } else if (effect.phase === "generate") {
          drawGenerateFormation(effect, progress, fade, target);
        } else {
          if (!memoryStar) memoryStar = memoryStarGeometry();
          drawTriageFormation(effect, progress, fade, memoryStar, target);
        }
      }
      if (!effect.paintedAt) {
        effect.paintedAt = time;
        cortexMetrics.transportPainted += 1;
        publishCortexMetrics();
      }
    }
    context.restore();
  }

  function compareActiveLabels(left, right) {
    return (
      right.fieldActivation - left.fieldActivation
      || right.fanIn - left.fanIn
      || left.id.localeCompare(right.id)
    );
  }

  function insertActiveLabel(node) {
    let insertAt = 0;
    while (
      insertAt < activeLabelNodes.length
      && compareActiveLabels(node, activeLabelNodes[insertAt]) >= 0
    ) {
      insertAt += 1;
    }
    if (insertAt >= ACTIVE_LABEL_LIMIT) return;
    const nextLength = Math.min(ACTIVE_LABEL_LIMIT, activeLabelNodes.length + 1);
    activeLabelNodes.length = nextLength;
    for (let index = nextLength - 1; index > insertAt; index -= 1) {
      activeLabelNodes[index] = activeLabelNodes[index - 1];
    }
    activeLabelNodes[insertAt] = node;
  }

  function beginLabelCandidates() {
    labelCandidates.length = 0;
    activeLabelNodes.length = 0;
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
      if (node.fieldActivation > 0.05) insertActiveLabel(node);
      if (state === 3 || index === selected || index === hovered) {
        addLabelCandidate(node);
      }
    }
    for (let index = 0; index < activeLabelNodes.length; index += 1) {
      const node = activeLabelNodes[index];
      addLabelCandidate(node);
    }
    if (!activeLabelNodes.length && camera.distance < 1500) {
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
        + excitationLevel(node, time) * 1000
        + node.fieldActivation * 900
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
    cortexMetrics.electricEdges = 0;
    cortexMetrics.electricPeak = 0;
    cortexMetrics.flashPeak = 0;
    cortexMetrics.maxCoreScale = 0;
    cortexMetrics.maxGlowPadding = 0;
    cortexMetrics.cometTrailSegments = 0;
    cortexMetrics.cometHeads = 0;
    cortexMetrics.cometImpacts = 0;
    cortexMetrics.maxCometHeadRadius = 0;
    cortexMetrics.cometCenterX = 0;
    cortexMetrics.cometCenterY = 0;
    cortexMetrics.cometMinX = width;
    cortexMetrics.cometMaxX = 0;
    cortexMetrics.cometMinY = height;
    cortexMetrics.cometMaxY = 0;
    cortexMetrics.cometSafeLeft = 0;
    cortexMetrics.cometSafeRight = width;
    cortexMetrics.cometSafeTop = 0;
    cortexMetrics.cometSafeBottom = height;
    cortexMetrics.cometTurns = 0;
    cortexMetrics.explorationArcs = 0;
    cortexMetrics.triageCandidates = 0;
    cortexMetrics.generateParticles = 0;
    cortexMetrics.consensusOrbits = 0;
    cortexMetrics.processingNodeBlinks = 0;
    cortexMetrics.processingTargetNodeIndex = -1;
    cortexMetrics.formationOriginX = 0;
    cortexMetrics.formationOriginY = 0;
    cortexMetrics.formationTargetX = 0;
    cortexMetrics.formationTargetY = 0;
    cortexMetrics.consolidationEdges = 0;
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
    drawCommunityHulls();
    drawEdges();
    drawEdgeAfterglows(time);
    drawPulses(time);
    drawNodes(time);
    drawCameraPivot(time);
    drawNodeEffects(time);
    drawTransportEffects(time);
    drawLabels(time);
    publishVisualMetrics(time);
  }

  let previousTime = performance.now();
  let simulationAccumulator = 0;
  function frame(now) {
    const delta = Math.min(60, now - previousTime);
    cortexMetrics.frameDurations[frameDurationCursor] = delta;
    frameDurationCursor = (frameDurationCursor + 1) % FRAME_DURATION_CAPACITY;
    frameDurationCount = Math.min(
      FRAME_DURATION_CAPACITY,
      frameDurationCount + 1,
    );
    previousTime = now;
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
    if (autoRotate && !dragging && now - lastInteraction > 2600) {
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
        camera.theta += event.movementX * 0.0045;
        camera.phi = Math.max(
          -1.35,
          Math.min(1.35, camera.phi - event.movementY * 0.0045),
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
        select(index, index >= 0);
      }
      downPoint = null;
      dragging = false;
      lastInteraction = performance.now();
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
    const growth = summary.growth || {};
    const growthProgress = growth.authority_enabled
      ? `AUTH · ${growth.canary_percent || 0}%`
      : growth.positive_learning_allowed || growth.field_learning_allowed
        ? "COFIRE ON · AUTH HELD"
        : `${growth.strong_positive || 0}/${growth.strong_positive_target || 200}`;
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
          <div><span>learning</span><b>${escapeHtml(growthProgress)}</b></div>
          <div><span>sessions</span><b>${growth.strong_sessions || 0}/${growth.strong_sessions_target || 20}</b></div>
          <div><span>used proof</span><b>${growth.processor_used_episodes || 0}/50</b></div>
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

  function graphStatusHtml() {
    const graph = data.typedGraph || {};
    const status = graph.status || {};
    const rollout = status.rollout || {};
    const rubric = status.rubric || {};
    const rubricGold = status.rubric_gold || {};
    const counts = status.relation_counts || {};
    const builder = status.builder || {};
    const entities = status.entities || {};
    const summaries = status.community_summary || {};
    const evaluation = status.evaluation || {};
    const authority = status.authority || {};
    const authorityCurrent = authority.current || {};
    const authorityTargets = authority.targets || {};
    const engineeringGates = status.engineering_gates || {};
    const failedEngineering = Object.entries(engineeringGates)
      .filter(([, passed]) => passed !== true)
      .map(([name]) => name);
    const countRows = [
      "proposed", "held", "verified", "repeatedly_used",
      "authoritative", "stale", "retracted",
    ];
    return `<div class="sec relationStatus"><h3>TYPED RELATION FIELD</h3>
      <div class="fieldStatus ${status.authority_mature ? "online" : "stale"}">
        <span class="stateGlyph" aria-hidden="true">${status.authority_mature ? "●" : "◷"}</span>
        <b>${status.engineering_complete ? "ENGINEERING COMPLETE" : "BUILDING"}</b>
        <span>${status.authority_mature ? "AUTHORITY MATURE" : "AUTHORITY COLLECTING"}</span>
      </div>
      <div class="fieldMetrics relationMetrics">
        ${countRows.map((name) => `<div><span>${escapeHtml(name)}</span><b>${Number(counts[name] || 0)}</b></div>`).join("")}
        <div><span>communities</span><b>${(graph.communities || []).length}</b></div>
        <div><span>external calls</span><b>${Number(status.external_model_calls || 0)}</b></div>
        <div><span>rollout</span><b>${escapeHtml(rollout.mode || "shadow")} ${Number(rollout.canary_percent || 0)}%</b></div>
        <div><span>canary evidence</span><b>${Number(rollout.sample_count || 0)} sessions</b></div>
        <div><span>rubric</span><b>${escapeHtml(rubric.status || "builtin")}</b></div>
        <div><span>rubric gold</span><b>${Number(rubricGold.cases || 0)}/30 · ${escapeHtml(rubricGold.step || rubricGold.status || "waiting")}</b></div>
        <div><span>builder queue</span><b>${Number(builder.remaining_pages || 0)} · ${escapeHtml(builder.model || "local")}</b></div>
        <div><span>merge holds</span><b>${Number(entities.held || 0)}</b></div>
        <div><span>summaries</span><b>${Number(summaries.generated || 0)} new · ${Number(summaries.reused || 0)} cached</b></div>
        <div><span>evaluation</span><b>${escapeHtml(evaluation.status || "waiting")} · ${escapeHtml(evaluation.winner || "current")}</b></div>
        <div><span>relation maturity</span><b>${Number(authorityCurrent.relation_strong || 0)}/${Number(authorityTargets.relation_strong || 0)} · ${Number(authorityCurrent.relation_sessions || 0)}/${Number(authorityTargets.relation_sessions || 0)} sessions</b></div>
        <div><span>entity maturity</span><b>${Number(authorityCurrent.entity_strong || 0)}/${Number(authorityTargets.entity_strong || 0)} · ${Number(authorityCurrent.entity_sessions || 0)}/${Number(authorityTargets.entity_sessions || 0)} sessions</b></div>
        <div><span>rubric maturity</span><b>${Number(authorityCurrent.rubric_gold || 0)}/${Number(authorityTargets.rubric_gold || 0)} · ${Number(authorityCurrent.rubric_sessions || 0)}/${Number(authorityTargets.rubric_sessions || 0)} sessions</b></div>
        <div><span>locked evaluation</span><b>${Number(authorityCurrent.evaluation_samples || 0)} arm rows</b></div>
      </div>
      ${failedEngineering.length ? `<div class="faultText">Engineering gates: ${failedEngineering.map(escapeHtml).join(" · ")}</div>` : ""}
      ${(authority.unmet_gates || []).length ? `<div class="ghost">Unmet gates: ${(authority.unmet_gates || []).map(escapeHtml).join(" · ")}. Re-evaluates on ${escapeHtml(authority.next_evaluation || "next sleep cycle")}.</div>` : ""}
      <div class="ghost">Static dashed lines are relation state. Yellow electricity appears only for a real Field spread event.</div>
    </div>`;
  }

  function relationDetailsHtml(node) {
    const relations = typedRelations.filter(
      (relation) => relation.source === node.index || relation.target === node.index,
    );
    if (!relations.length) {
      return '<div class="sec"><h3>TYPED RELATIONS · 0</h3><div class="ghost">— none</div></div>';
    }
    return `<div class="sec"><h3>TYPED RELATIONS · ${relations.length}</h3>
      <div class="relationCards">${relations.slice(0, 24).map((relation) => {
        const consensus = relation.consensus || {};
        const votes = Array.isArray(consensus.votes) ? consensus.votes : [];
        const evidence = Array.isArray(relation.evidence_refs) ? relation.evidence_refs : [];
        return `<details class="relationCard" ${activeRelationIds.has(relation.relation_id) ? "open" : ""}>
          <summary><span class="relationLife ${escapeHtml(relation.status)}">${escapeHtml(relation.status)}</span><b>${escapeHtml(relation.predicate)}</b><small>${escapeHtml(relation.direction)}</small></summary>
          <div class="relationId">${escapeHtml(relation.relation_id)}</div>
          <div class="mrow"><span>path</span><b>${escapeHtml(relation.source_page_id)} → ${escapeHtml(relation.target_page_id)}</b></div>
          <div class="mrow"><span>producer</span><b>${escapeHtml(relation.producer_role)}</b></div>
          <div class="mrow"><span>used</span><b>${Number(relation.used_count || 0)} · ${Number(relation.used_sessions || 0)} sessions</b></div>
          <div class="mrow"><span>reason</span><b>${escapeHtml(relation.reason_code || consensus.hold_reason || "—")}</b></div>
          <div class="mrow"><span>quorum</span><b>${Number(consensus.quorum || 0)} · ${escapeHtml(consensus.outcome || "pending")}</b></div>
          <div class="relationVotes">${votes.length ? votes.map((vote) => `<span class="${escapeHtml(vote.decision)}">${escapeHtml(vote.role)} · ${escapeHtml(vote.decision)} ${(Number(vote.confidence || 0) * 100).toFixed(0)}%</span>`).join("") : '<span>no votes</span>'}</div>
          <div class="relationEvidence">${evidence.map((ref) => `<span title="${escapeHtml(ref.content_sha256)} / ${escapeHtml(ref.span_sha256)}">${escapeHtml(ref.page_id)}:${Number(ref.source_line || 0)} · ${escapeHtml(String(ref.span_sha256 || "").slice(0, 10))}</span>`).join("")}</div>
        </details>`;
      }).join("")}</div>
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
    return `${fieldStateHtml()}${graphStatusHtml()}
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
        <div class="mrow"><span style="color:#ffd84d">⚡ RECALL</span><b>yellow · synaptic firing</b></div>
        <div class="mrow"><span style="color:#4fe4ff">☄ SAVE</span><b>cyan · comet capture → memory core</b></div>
        <div class="mrow"><span style="color:#ffb454">⌁ TRIAGE</span><b>orange · candidate evaluation</b></div>
        <div class="mrow"><span style="color:#9b7cff">✦ GENERATE</span><b>violet · memory synthesis</b></div>
        <div class="mrow"><span style="color:#ffe6ae">◎ CONSENSUS</span><b>platinum · local agreement</b></div>
        <div class="mrow"><span style="color:#45d49b">● APPLY</span><b>green · graph consolidation</b></div>
        <div class="ghost">Liveは実Field eventと実SAVE/INGEST telemetryのみ。Memory I/Oは発火と区別し、DEMO/REPLAYはbackendへ書きません。</div>
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
      ${graphStatusHtml()}
      ${relationDetailsHtml(node)}
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
          ? "LIVE PAINT resumed · Field state stayed current"
          : "LIVE PAINT paused · Field state still updating",
      );
    });
    const relationToggle = document.getElementById("tRelations");
    relationToggle.addEventListener("click", () => {
      relationsVisible = !relationsVisible;
      relationToggle.classList.toggle("on", relationsVisible);
      relationToggle.setAttribute("aria-pressed", String(relationsVisible));
      stateDirty = true;
      saveViewPreferences();
      flashTicker(relationsVisible ? "typed relation layer enabled" : "typed relation layer hidden");
    });
    document.getElementById("relationLifecycle").addEventListener("change", (event) => {
      relationLifecycle = event.target.value;
      stateDirty = true;
      saveViewPreferences();
      flashTicker(`relation lifecycle · ${relationLifecycle}`);
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
    });
    const rotateToggle = document.getElementById("tRot");
    rotateToggle.addEventListener("click", () => {
      autoRotate = !autoRotate;
      rotateToggle.classList.toggle("on", autoRotate);
      rotateToggle.setAttribute("aria-pressed", String(autoRotate));
      saveViewPreferences();
    });
    const soundToggle = document.getElementById("tSnd");
    soundToggle.addEventListener("click", () => {
      soundOn = !soundOn;
      soundToggle.classList.toggle("on", soundOn);
      soundToggle.setAttribute("aria-pressed", String(soundOn));
      saveViewPreferences();
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
      const value = event.target.value;
      followLatestSession = value === LIVE_SESSION_VALUE;
      loadField(followLatestSession ? "" : value);
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
    mode = nextMode === "cluster" ? "cluster" : "organic";
    reheat(0.7);
    document
      .getElementById("mOrganic")
      .classList.toggle("on", mode === "organic");
    document
      .getElementById("mCluster")
      .classList.toggle("on", mode === "cluster");
    saveViewPreferences();
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
    const liveOption = document.createElement("option");
    liveOption.value = LIVE_SESSION_VALUE;
    liveOption.textContent = "LIVE · follow activity";
    liveOption.selected = followLatestSession;
    selectElement.appendChild(liveOption);
    if (!fieldState.sessions.length) {
      const option = document.createElement("option");
      option.value = "";
      option.textContent = "telemetry fallback";
      option.selected = !followLatestSession;
      selectElement.appendChild(option);
      return;
    }
    fieldState.sessions.forEach((session, index) => {
      const option = document.createElement("option");
      option.value = session.session_hash;
      const age = Math.max(0, Date.now() / 1000 - Number(session.updated_at_epoch || 0));
      const host = session.host || "unknown";
      option.textContent =
        `${host} · ${session.session_hash} · ${index ? `${Math.round(age / 60)}m` : "latest"}`;
      option.selected =
        !followLatestSession && session.session_hash === fieldState.sessionHash;
      selectElement.appendChild(option);
    });
  }

  function processingEffectPhase(step) {
    const normalized = String(step || "").toLowerCase();
    if (
      ["raw", "triage", "search", "select", "discover", "detect"].includes(
        normalized,
      )
    ) {
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
    if (
      ["consensus", "tie_break", "verify", "evaluate", "escalate"].includes(
        normalized,
      )
    ) {
      return "consensus";
    }
    if (["apply", "commit", "report", "consolidate", "promote"].includes(normalized)) {
      return "apply";
    }
    return "generate";
  }

  function processingLaneEvent(lane, phase = processingEffectPhase(lane.current_step)) {
    const laneKey = String(lane.key || "process");
    const step = String(lane.current_step || lane.phase || "work");
    const label = `${lane.label || laneKey} · ${step}`;
    return {
      kind: "processing",
      phase,
      source: "processing-activity",
      lane_key: laneKey,
      step,
      channel_key: `processing:${laneKey}`,
      label,
      model: String(lane.model || ""),
      role: String(lane.role || ""),
      page_ids: [],
    };
  }

  function pulseProcessingLane(state, now = performance.now()) {
    if (!liveEventsEnabled || document.hidden) return;
    visualizeTransportEvent(processingLaneEvent(state.lane, state.phase));
    state.lastPulseAt = now;
  }

  function applyProcessingActivity(snapshot) {
    if (!Array.isArray(snapshot?.lanes)) return;
    const now = performance.now();
    const activeKeys = new Set();
    snapshot.lanes.forEach((lane) => {
      const laneKey = String(lane.key || "process");
      if (lane.state !== "active") return;
      activeKeys.add(laneKey);
      const phase = processingEffectPhase(lane.current_step);
      const signature = [
        lane.work_item,
        lane.current_step,
        lane.model,
        lane.role,
      ].map((value) => String(value || "")).join(":");
      const previous = activeProcessingLanes.get(laneKey);
      if (!previous || previous.signature !== signature) {
        const state = { lane, phase, signature, lastPulseAt: 0 };
        activeProcessingLanes.set(laneKey, state);
        pulseProcessingLane(state, now);
      } else {
        previous.lane = lane;
        previous.phase = phase;
      }
    });
    [...activeProcessingLanes.entries()].forEach(([laneKey, state]) => {
      if (activeKeys.has(laneKey)) return;
      if (liveEventsEnabled) {
        visualizeTransportEvent(processingLaneEvent(state.lane, "complete"));
      }
      activeProcessingLanes.delete(laneKey);
    });
    cortexMetrics.activeProcessingLanes = activeProcessingLanes.size;
    publishCortexMetrics();
  }

  function pulseActiveProcessingLanes() {
    const now = performance.now();
    activeProcessingLanes.forEach((state) => {
      if (now - state.lastPulseAt >= PROCESSING_EFFECT_PULSE_MS) {
        pulseProcessingLane(state, now);
      }
    });
  }

  function connectProcessingActivity() {
    if (!("EventSource" in window)) return;
    if (processingActivitySource) processingActivitySource.close();
    processingActivitySource = new EventSource("/api/activity-stream");
    processingActivitySource.addEventListener("activity", (message) => {
      try {
        applyProcessingActivity(JSON.parse(message.data));
      } catch (_error) {
        return;
      }
    });
    processingActivitySource.onopen = () => {
      cortexMetrics.processingActivityConnected = true;
      publishCortexMetrics();
    };
    processingActivitySource.onerror = () => {
      cortexMetrics.processingActivityConnected = false;
      publishCortexMetrics();
    };
    window.clearInterval(processingPulseTimer);
    processingPulseTimer = window.setInterval(
      pulseActiveProcessingLanes,
      250,
    );
    document.addEventListener("visibilitychange", pulseActiveProcessingLanes);
  }

  async function loadField(sessionHash = "", replayEvents = []) {
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
      if (liveEventsEnabled && replayEvents.length) {
        replayEvents
          .filter((event) => event.session_hash === fieldState.sessionHash)
          .forEach(visualizeFieldEvent);
      }
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
    const queryString = followLatestSession
      ? "?follow=latest"
      : sessionHash
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
      const telemetryEvents = payload.events.filter(
        (event) => event.source === "telemetry-fallback",
      );
      const TRANSPORT_KINDS = new Set(["save", "ingest", "recall", "auto_recall", "read", "search", "used"]);
      const transportEvents = telemetryEvents.filter(
        (event) => TRANSPORT_KINDS.has(event.kind),
      );
      const fieldEvents = payload.events.filter(
        (event) => event.source !== "telemetry-fallback",
      );
      if (liveEventsEnabled) transportEvents.forEach(visualizeTransportEvent);
      const latestFallback = telemetryEvents
        .filter((event) => !TRANSPORT_KINDS.has(event.kind))
        .at(-1);
      if (latestFallback) {
        flashTicker(
          `TELEMETRY · ${latestFallback.label || latestFallback.kind || "activity"} · no synthetic firing`,
        );
      }
      const latestEvent = fieldEvents.at(-1);
      if (
        followLatestSession
        && latestEvent?.session_hash
        && latestEvent.session_hash !== fieldState.sessionHash
      ) {
        const targetSession = latestEvent.session_hash;
        const replayEvents = fieldEvents.filter(
          (event) => event.session_hash === targetSession,
        );
        loadField(targetSession, replayEvents);
        return;
      }
      if (!fieldEvents.length) return;
      if (!fieldState.sessionHash) {
        const event = latestEvent;
        flashTicker(
          `TELEMETRY FALLBACK · ${event?.label || event?.kind || "activity"} · no synthetic firing`,
        );
        return;
      }
      const accepted = window.CortexField.applyEvents(fieldState, fieldEvents);
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
      applyViewPreferences(loadViewPreferences());
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
      connectProcessingActivity();
      await loadField();
    } catch (error) {
      showBootError(error);
    }
  }

  boot();
})();
