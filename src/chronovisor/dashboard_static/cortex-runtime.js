"use strict";

((root, factory) => {
  const api = factory();
  if (typeof module === "object" && module.exports) module.exports = api;
  if (root) root.CortexRuntime = api;
})(typeof window === "undefined" ? globalThis : window, () => {
  const DEFAULT_SAMPLE_CAPACITY = 240;
  const GOLDEN_ANGLE = Math.PI * (3 - Math.sqrt(5));
  const SPHERE_FOG_OPACITIES = Object.freeze([0.875, 0.625, 0.375, 0.2]);

  function clamp(value, minimum, maximum) {
    return Math.max(minimum, Math.min(maximum, value));
  }

  function deterministicUnit(value, salt = 0) {
    let hash = 2166136261 ^ Number(salt || 0);
    const text = String(value);
    for (let index = 0; index < text.length; index += 1) {
      hash ^= text.charCodeAt(index);
      hash = Math.imul(hash, 16777619);
    }
    return (hash >>> 0) / 4294967295;
  }

  function normalizeLayoutMode(value) {
    return value === "sphere" || value === "cluster" ? value : "organic";
  }

  function sphereFogBand(depthFade) {
    const fade = Number(depthFade);
    const boundedFade = Number.isFinite(fade) ? clamp(fade, 0, 1) : 0;
    return Math.min(3, Math.floor((1 - boundedFade) * 4));
  }

  function sphereFogOpacity(depthBand) {
    const band = clamp(Math.trunc(Number(depthBand) || 0), 0, 3);
    return SPHERE_FOG_OPACITIES[band];
  }

  function fitSphereCamera(options = {}) {
    const viewportWidth = Math.max(1, Number(options.viewportWidth) || 1);
    const viewportHeight = Math.max(1, Number(options.viewportHeight) || 1);
    const topInset = clamp(
      Number(options.topInset) || 0,
      0,
      Math.max(0, viewportHeight - 1),
    );
    const usableWidth = viewportWidth;
    const usableHeight = Math.max(1, viewportHeight - topInset);
    const maximumPadding = Math.max(
      0,
      Math.min(usableWidth, usableHeight) / 2 - 1,
    );
    const padding = clamp(
      Number(options.padding) || 0,
      0,
      maximumPadding,
    );
    const safeRadius = Math.max(
      1,
      Math.min(usableWidth, usableHeight) / 2 - padding,
    );
    const sphereRadius = Math.max(0, Number(options.sphereRadius) || 0);
    const focalLength = Math.max(1, Number(options.focalLength) || 1);
    const minimumDistance = Math.max(
      0,
      Number(options.minimumDistance) || 0,
    );
    const tangentDistance = sphereRadius > 0
      ? sphereRadius * Math.sqrt(1 + (focalLength / safeRadius) ** 2)
      : 0;
    const distance = Math.max(
      minimumDistance,
      tangentDistance,
      sphereRadius > 0 ? sphereRadius + 1e-6 : 0,
    );
    const projectedRadius = sphereRadius > 0
      ? focalLength * sphereRadius
        / Math.sqrt(Math.max(1e-12, distance ** 2 - sphereRadius ** 2))
      : 0;
    const centerX = viewportWidth / 2;
    const centerY = topInset + usableHeight / 2;
    return {
      distance,
      focalLength,
      padding,
      projectedRadius,
      safeRadius,
      sphereRadius,
      topInset,
      usableHeight,
      usableWidth,
      center: { x: centerX, y: centerY },
      projectedBounds: {
        left: centerX - projectedRadius,
        right: centerX + projectedRadius,
        top: centerY - projectedRadius,
        bottom: centerY + projectedRadius,
      },
      safeBounds: {
        left: padding,
        right: viewportWidth - padding,
        top: topInset + padding,
        bottom: viewportHeight - padding,
      },
    };
  }

  function compareText(left, right) {
    const leftText = String(left);
    const rightText = String(right);
    return leftText < rightText ? -1 : leftText > rightText ? 1 : 0;
  }

  function fibonacciSpherePoint(index, count, radii = 1) {
    const total = Math.max(1, Number(count) || 1);
    const offset = Math.max(0, Number(index) || 0) + 0.5;
    const y = 1 - (2 * offset) / total;
    const planar = Math.sqrt(Math.max(0, 1 - y * y));
    const angle = GOLDEN_ANGLE * offset;
    const scale = typeof radii === "number"
      ? { x: radii, y: radii, z: radii }
      : {
          x: Number(radii?.x) || 1,
          y: Number(radii?.y) || 1,
          z: Number(radii?.z) || 1,
        };
    return {
      x: Math.cos(angle) * planar * scale.x,
      y: y * scale.y,
      z: Math.sin(angle) * planar * scale.z,
    };
  }

  function normalizeVector(vector) {
    const length = Math.hypot(vector.x, vector.y, vector.z) || 1;
    return {
      x: vector.x / length,
      y: vector.y / length,
      z: vector.z / length,
    };
  }

  function createSphereTargets(nodes, options = {}) {
    const source = Array.isArray(nodes) ? nodes : [];
    const radii = {
      core: Math.max(40, Number(options.coreRadius) || 180),
      middle: Math.max(80, Number(options.middleRadius) || 350),
      outer: Math.max(120, Number(options.outerRadius) || 540),
    };
    const coreIds = new Set([
      "claude-code",
      "current-state",
      "lessons-learned",
      "user-profile",
    ]);
    const coreIndexes = new Set();
    source.forEach((node, index) => {
      if (
        node?.entrypoint
        || node?.packageName === "system"
        || coreIds.has(String(node?.id || ""))
      ) coreIndexes.add(index);
    });
    const hubCandidates = source
      .map((node, index) => ({
        index,
        id: String(node?.id || index),
        connectivity: Math.max(
          0,
          Number(node?.fanIn || 0) + Number(node?.fanOut || 0),
        ),
      }))
      .filter((row) => !coreIndexes.has(row.index))
      .sort(
        (left, right) =>
          right.connectivity - left.connectivity
          || compareText(left.id, right.id),
      );
    const hubCount = hubCandidates.length
      ? Math.max(1, Math.ceil(source.length * 0.06))
      : 0;
    const hubIndexes = new Set(
      hubCandidates.slice(0, hubCount).map((row) => row.index),
    );
    const packageNames = [...new Set(
      source.map((node) => String(node?.packageName || "unclassified")),
    )].sort(compareText);
    const membersByPackage = new Map(
      packageNames.map((packageName) => [packageName, []]),
    );
    source.forEach((node, index) => {
      const packageName = String(node?.packageName || "unclassified");
      membersByPackage.get(packageName).push({
        index,
        id: String(node?.id || index),
      });
    });
    membersByPackage.forEach((members) => {
      members.sort((left, right) => compareText(left.id, right.id));
    });
    const packageAnchors = new Map(
      packageNames.map((packageName, index) => [
        packageName,
        normalizeVector(fibonacciSpherePoint(index, packageNames.length)),
      ]),
    );

    const targets = new Array(source.length);
    packageNames.forEach((packageName) => {
      const anchor = packageAnchors.get(packageName);
      const reference = Math.abs(anchor.y) < 0.9
        ? { x: 0, y: 1, z: 0 }
        : { x: 1, y: 0, z: 0 };
      const tangent = normalizeVector({
        x: reference.y * anchor.z - reference.z * anchor.y,
        y: reference.z * anchor.x - reference.x * anchor.z,
        z: reference.x * anchor.y - reference.y * anchor.x,
      });
      const bitangent = normalizeVector({
        x: anchor.y * tangent.z - anchor.z * tangent.y,
        y: anchor.z * tangent.x - anchor.x * tangent.z,
        z: anchor.x * tangent.y - anchor.y * tangent.x,
      });
      const members = membersByPackage.get(packageName);
      const packageFraction = members.length / Math.max(1, source.length);
      const capRadius = clamp(
        Math.acos(clamp(1 - 2 * packageFraction * 0.92, -1, 1)),
        0.12,
        1.45,
      );
      const phase = deterministicUnit(packageName, 73) * Math.PI * 2;
      members.forEach((member, memberOffset) => {
        const tier = coreIndexes.has(member.index)
          ? "core"
          : hubIndexes.has(member.index)
            ? "middle"
            : "outer";
        const baseRadius = radii[tier];
        const jitterRange = tier === "outer" ? 22 : tier === "middle" ? 12 : 7;
        const radius = baseRadius
          + (deterministicUnit(member.id, 211) - 0.5) * jitterRange * 2;
        const radialFraction = Math.sqrt(
          (memberOffset + 0.5) / Math.max(1, members.length),
        );
        const angularRadius = capRadius * radialFraction;
        const angle = phase + GOLDEN_ANGLE * memberOffset;
        const tangentX =
          tangent.x * Math.cos(angle) + bitangent.x * Math.sin(angle);
        const tangentY =
          tangent.y * Math.cos(angle) + bitangent.y * Math.sin(angle);
        const tangentZ =
          tangent.z * Math.cos(angle) + bitangent.z * Math.sin(angle);
        const direction = normalizeVector({
          x: anchor.x * Math.cos(angularRadius)
            + tangentX * Math.sin(angularRadius),
          y: anchor.y * Math.cos(angularRadius)
            + tangentY * Math.sin(angularRadius),
          z: anchor.z * Math.cos(angularRadius)
            + tangentZ * Math.sin(angularRadius),
        });
        targets[member.index] = {
          x: direction.x * radius,
          y: direction.y * radius,
          z: direction.z * radius,
          radius,
          tier,
          packageName,
        };
      });
    });
    // A few deterministic recentering passes remove residual mass bias from
    // uneven package sizes while retaining each node's tier radius.
    for (let iteration = 0; iteration < 4 && targets.length > 1; iteration += 1) {
      const centroid = targets.reduce(
        (total, target) => ({
          x: total.x + target.x / targets.length,
          y: total.y + target.y / targets.length,
          z: total.z + target.z / targets.length,
        }),
        { x: 0, y: 0, z: 0 },
      );
      if (Math.hypot(centroid.x, centroid.y, centroid.z) < 0.01) break;
      targets.forEach((target) => {
        const shifted = {
          x: target.x - centroid.x,
          y: target.y - centroid.y,
          z: target.z - centroid.z,
        };
        const shiftedLength = Math.hypot(shifted.x, shifted.y, shifted.z);
        if (shiftedLength < 1e-9) return;
        const direction = normalizeVector(shifted);
        target.x = direction.x * target.radius;
        target.y = direction.y * target.radius;
        target.z = direction.z * target.radius;
      });
    }
    return targets;
  }

  function measureSphereQuality(nodes, targets) {
    const source = Array.isArray(nodes) ? nodes : [];
    const goals = Array.isArray(targets) ? targets : [];
    const tiers = { core: 0, middle: 0, outer: 0 };
    let finiteNodes = 0;
    let targetMinimum = Infinity;
    let targetMaximum = 0;
    let targetTotal = 0;
    let actualMinimum = Infinity;
    let actualMaximum = 0;
    let actualTotal = 0;
    let errorTotal = 0;
    let errorSquaredTotal = 0;
    let errorMaximum = 0;
    let radialErrorTotal = 0;
    let radialErrorSquaredTotal = 0;
    let radialErrorMaximum = 0;
    let targetCount = 0;
    const targetCentroid = { x: 0, y: 0, z: 0 };
    const targetOctants = new Array(8).fill(0);
    for (let index = 0; index < goals.length; index += 1) {
      const target = goals[index];
      const node = source[index];
      if (!target) continue;
      targetCount += 1;
      targetCentroid.x += Number(target.x) || 0;
      targetCentroid.y += Number(target.y) || 0;
      targetCentroid.z += Number(target.z) || 0;
      const octant = (target.x >= 0 ? 1 : 0)
        | (target.y >= 0 ? 2 : 0)
        | (target.z >= 0 ? 4 : 0);
      targetOctants[octant] += 1;
      if (Object.hasOwn(tiers, target.tier)) tiers[target.tier] += 1;
      const targetRadius = Number(target.radius) || 0;
      targetMinimum = Math.min(targetMinimum, targetRadius);
      targetMaximum = Math.max(targetMaximum, targetRadius);
      targetTotal += targetRadius;
      if (
        !node
        || !Number.isFinite(node.x)
        || !Number.isFinite(node.y)
        || !Number.isFinite(node.z)
      ) continue;
      const actualRadius = Math.hypot(node.x, node.y, node.z);
      const radialError = Math.abs(actualRadius - targetRadius);
      const error = Math.hypot(
        node.x - target.x,
        node.y - target.y,
        node.z - target.z,
      );
      finiteNodes += 1;
      actualMinimum = Math.min(actualMinimum, actualRadius);
      actualMaximum = Math.max(actualMaximum, actualRadius);
      actualTotal += actualRadius;
      errorTotal += error;
      errorSquaredTotal += error * error;
      errorMaximum = Math.max(errorMaximum, error);
      radialErrorTotal += radialError;
      radialErrorSquaredTotal += radialError * radialError;
      radialErrorMaximum = Math.max(radialErrorMaximum, radialError);
    }
    if (targetCount) {
      targetCentroid.x /= targetCount;
      targetCentroid.y /= targetCount;
      targetCentroid.z /= targetCount;
    }
    const targetCentroidOffset = Math.hypot(
      targetCentroid.x,
      targetCentroid.y,
      targetCentroid.z,
    );
    return {
      nodeCount: targetCount,
      finiteNodes,
      tiers,
      targetRadius: {
        min: targetCount ? targetMinimum : 0,
        max: targetCount ? targetMaximum : 0,
        mean: targetCount ? targetTotal / targetCount : 0,
      },
      targetCentroid: {
        ...targetCentroid,
        offset: targetCentroidOffset,
        normalizedOffset: targetMaximum
          ? targetCentroidOffset / targetMaximum
          : 0,
      },
      targetOctants,
      occupiedOctants: targetOctants.filter((count) => count > 0).length,
      actualRadius: {
        min: finiteNodes ? actualMinimum : 0,
        max: finiteNodes ? actualMaximum : 0,
        mean: finiteNodes ? actualTotal / finiteNodes : 0,
      },
      targetError: {
        mean: finiteNodes ? errorTotal / finiteNodes : 0,
        rms: finiteNodes ? Math.sqrt(errorSquaredTotal / finiteNodes) : 0,
        max: errorMaximum,
      },
      radialError: {
        mean: finiteNodes ? radialErrorTotal / finiteNodes : 0,
        rms: finiteNodes
          ? Math.sqrt(radialErrorSquaredTotal / finiteNodes)
          : 0,
        max: radialErrorMaximum,
      },
    };
  }

  function percentile(sorted, fraction) {
    if (!sorted.length) return 0;
    const index = Math.min(
      sorted.length - 1,
      Math.max(0, Math.ceil(sorted.length * fraction) - 1),
    );
    return sorted[index];
  }

  function createDurationRing(capacity = DEFAULT_SAMPLE_CAPACITY) {
    const values = new Float64Array(Math.max(1, capacity));
    let cursor = 0;
    let count = 0;
    let maximum = 0;
    let total = 0;
    return {
      record(value) {
        const duration = Number(value);
        if (!Number.isFinite(duration) || duration < 0) return;
        if (count === values.length) total -= values[cursor];
        values[cursor] = duration;
        cursor = (cursor + 1) % values.length;
        count = Math.min(values.length, count + 1);
        maximum = Math.max(maximum, duration);
        total += duration;
      },
      snapshot() {
        const sample = new Array(count);
        const start = (cursor - count + values.length) % values.length;
        for (let index = 0; index < count; index += 1) {
          sample[index] = values[(start + index) % values.length];
        }
        sample.sort((left, right) => left - right);
        return {
          count,
          p50: percentile(sample, 0.5),
          p95: percentile(sample, 0.95),
          p99: percentile(sample, 0.99),
          max: sample.length ? sample[sample.length - 1] : 0,
          lifetimeMax: maximum,
          mean: count ? total / count : 0,
          samples: sample,
        };
      },
    };
  }

  function createStageMetrics(
    names = ["simulation", "projectionBase", "overlayEffects", "domEventFlush"],
    capacity = DEFAULT_SAMPLE_CAPACITY,
  ) {
    const rings = new Map(
      names.map((name) => [name, createDurationRing(capacity)]),
    );
    return {
      record(name, duration) {
        if (!rings.has(name)) rings.set(name, createDurationRing(capacity));
        rings.get(name).record(duration);
      },
      snapshot() {
        return Object.fromEntries(
          [...rings.entries()].map(([name, ring]) => [name, ring.snapshot()]),
        );
      },
    };
  }

  function createRenderScheduler(options) {
    const requestFrame = options.requestFrame;
    const cancelFrame = options.cancelFrame || (() => {});
    const isHidden = options.isHidden || (() => false);
    const onFrame = options.onFrame;
    const hasWork = options.hasWork || (() => false);
    const dirtyReasons = new Set();
    // requestAnimationFrame identifiers are allowed to wrap to zero, so zero
    // cannot double as the "nothing scheduled" sentinel.
    let pendingFrame = null;
    let disposed = false;
    let running = false;
    let pendingContinuation = false;
    let scheduledCount = 0;
    let renderedCount = 0;
    let lastRenderedAt = null;

    function schedule(continuation = false) {
      if (disposed || pendingFrame !== null || isHidden()) return false;
      pendingContinuation = Boolean(continuation);
      pendingFrame = requestFrame(run);
      scheduledCount += 1;
      return true;
    }

    function invalidate(reason = "unknown") {
      if (disposed) return false;
      dirtyReasons.add(String(reason));
      return running ? true : schedule(false);
    }

    function run(now) {
      pendingFrame = null;
      const continuation = pendingContinuation;
      pendingContinuation = false;
      if (disposed || isHidden()) return;
      const reasons = [...dirtyReasons];
      dirtyReasons.clear();
      renderedCount += 1;
      const sinceLastRender = lastRenderedAt === null
        ? null
        : Math.max(0, Number(now) - lastRenderedAt);
      lastRenderedAt = Number(now);
      running = true;
      try {
        onFrame(now, reasons, { continuation, sinceLastRender });
      } finally {
        running = false;
      }
      if (dirtyReasons.size || hasWork()) schedule(true);
    }

    function visibilityChanged() {
      if (disposed) return;
      if (isHidden()) {
        if (pendingFrame !== null) cancelFrame(pendingFrame);
        pendingFrame = null;
        pendingContinuation = false;
        return;
      }
      invalidate("visibility");
    }

    return {
      invalidate,
      visibilityChanged,
      dispose() {
        disposed = true;
        if (pendingFrame !== null) cancelFrame(pendingFrame);
        pendingFrame = null;
        pendingContinuation = false;
        running = false;
        dirtyReasons.clear();
      },
      state() {
        return {
          pending: pendingFrame === null ? 0 : 1,
          dirtyReasons: [...dirtyReasons],
          scheduledCount,
          renderedCount,
          lastRenderedAt,
          running,
          hidden: isHidden(),
        };
      },
    };
  }

  function createCooldownWake(options) {
    const now = options.now;
    const setTimer = options.setTimer;
    const clearTimer = options.clearTimer;
    const readyAt = options.readyAt;
    const isEnabled = options.isEnabled;
    const isHidden = options.isHidden || (() => false);
    const onReady = options.onReady;
    let timer = null;
    let scheduledFor = 0;
    let generation = 0;
    let disposed = false;

    function cancel() {
      if (timer === null && !scheduledFor) return;
      generation += 1;
      if (timer !== null) clearTimer(timer);
      timer = null;
      scheduledFor = 0;
    }

    function ready() {
      return !disposed
        && isEnabled()
        && !isHidden()
        && now() >= Number(readyAt());
    }

    function sync() {
      if (disposed || !isEnabled() || isHidden()) {
        cancel();
        return false;
      }
      const observed = now();
      const target = Number(readyAt());
      if (!Number.isFinite(target) || observed >= target) {
        cancel();
        return true;
      }
      if (timer !== null && scheduledFor === target) return false;
      cancel();
      const token = ++generation;
      scheduledFor = target;
      timer = setTimer(() => {
        if (disposed || token !== generation) return;
        timer = null;
        scheduledFor = 0;
        if (!isEnabled() || isHidden()) return;
        if (now() < Number(readyAt())) {
          sync();
          return;
        }
        onReady();
      }, Math.max(0, target - observed));
      return false;
    }

    return {
      ready,
      sync,
      cancel,
      dispose() {
        if (disposed) return;
        cancel();
        disposed = true;
      },
      state() {
        return {
          pending: timer === null ? 0 : 1,
          scheduledFor,
          generation,
          disposed,
        };
      },
    };
  }

  function createGenerationGate() {
    let observedGeneration = null;
    let observedKey = "";
    return {
      accept(generation, key = "") {
        const normalizedKey = String(key);
        if (
          observedGeneration === generation
          && observedKey === normalizedKey
        ) return false;
        observedGeneration = generation;
        observedKey = normalizedKey;
        return true;
      },
      clear() {
        observedGeneration = null;
        observedKey = "";
      },
      state() {
        return { generation: observedGeneration, key: observedKey };
      },
    };
  }

  // The pulse queue has exactly one structural owner. Completion callbacks run
  // only after the queue has been compacted, so callbacks cannot invalidate an
  // in-progress index walk.
  function drainExpiredPulses(queue, isExpired, onComplete) {
    const completed = [];
    let writeIndex = 0;
    const initialLength = queue.length;
    for (let readIndex = 0; readIndex < initialLength; readIndex += 1) {
      const pulse = queue[readIndex];
      if (isExpired(pulse)) completed.push(pulse);
      else {
        queue[writeIndex] = pulse;
        writeIndex += 1;
      }
    }
    queue.length = writeIndex;
    for (let index = 0; index < completed.length; index += 1) {
      onComplete(completed[index]);
    }
    return completed.length;
  }

  function createSpatialIndex(cellSize = 48) {
    const size = Math.max(8, Number(cellSize) || 48);
    const cells = new Map();
    let maximumHitRadius = 0;
    function key(column, row) {
      return `${column}:${row}`;
    }
    return {
      rebuild(nodes, visible = () => true, hitRadius = () => 0) {
        cells.clear();
        maximumHitRadius = 0;
        for (let index = 0; index < nodes.length; index += 1) {
          const node = nodes[index];
          if (!node || !visible(node, index)) continue;
          maximumHitRadius = Math.max(
            maximumHitRadius,
            Math.max(0, Number(hitRadius(node, index)) || 0),
          );
          const column = Math.floor(node.screenX / size);
          const row = Math.floor(node.screenY / size);
          const cellKey = key(column, row);
          let bucket = cells.get(cellKey);
          if (!bucket) {
            bucket = [];
            cells.set(cellKey, bucket);
          }
          bucket.push(index);
        }
      },
      pick(x, y, nodes, hitDistance) {
        const column = Math.floor(x / size);
        const row = Math.floor(y / size);
        const searchRadius = Math.max(
          1,
          Math.ceil((maximumHitRadius + 4) / size),
        );
        let best = -1;
        let bestDistance = Infinity;
        for (let offsetY = -searchRadius; offsetY <= searchRadius; offsetY += 1) {
          for (let offsetX = -searchRadius; offsetX <= searchRadius; offsetX += 1) {
            const bucket = cells.get(key(column + offsetX, row + offsetY));
            if (!bucket) continue;
            for (let index = 0; index < bucket.length; index += 1) {
              const nodeIndex = bucket[index];
              const distance = hitDistance(nodes[nodeIndex], x, y);
              if (distance < 4 && distance < bestDistance) {
                best = nodeIndex;
                bestDistance = distance;
              }
            }
          }
        }
        return best;
      },
      cellCount() {
        return cells.size;
      },
      maximumHitRadius() {
        return maximumHitRadius;
      },
    };
  }

  function createEventQueue(options = {}) {
    const maximum = Math.max(1, Number(options.maximum) || 256);
    const protectedEvent = options.protectedEvent || (() => false);
    const coalesceKey = options.coalesceKey || (() => "");
    const overflowCoalesceKey = options.overflowCoalesceKey || (() => "");
    const mergeOverflow = options.mergeOverflow || ((_previous, event) => event);
    const recoverableEvent = options.recoverableEvent || (() => false);
    const onOverflow = options.onOverflow || (() => {});
    const queue = [];
    let overflowCount = 0;
    let droppedCount = 0;
    let coalescedCount = 0;

    function evictStandard() {
      const index = queue.findIndex((event) => !protectedEvent(event));
      if (index < 0) return false;
      queue.splice(index, 1);
      droppedCount += 1;
      return true;
    }

    return {
      push(events) {
        for (let index = 0; index < events.length; index += 1) {
          const event = events[index];
          if (!event) continue;
          const key = coalesceKey(event);
          if (key) {
            const existing = queue.findIndex(
              (candidate) => coalesceKey(candidate) === key,
            );
            if (existing >= 0) {
              queue[existing] = event;
              continue;
            }
          }
          if (queue.length >= maximum && !evictStandard()) {
            const overflowKey = overflowCoalesceKey(event);
            const existing = overflowKey
              ? queue.findIndex(
                  (candidate) => overflowCoalesceKey(candidate) === overflowKey,
                )
              : -1;
            if (existing >= 0) {
              const merged = mergeOverflow(queue[existing], event);
              queue.splice(existing, 1);
              queue.push(merged);
              overflowCount += 1;
              coalescedCount += 1;
              continue;
            }
            const recoverable = queue.findIndex(recoverableEvent);
            if (recoverable >= 0) {
              const [evicted] = queue.splice(recoverable, 1);
              overflowCount += 1;
              droppedCount += 1;
              onOverflow(evicted);
              queue.push(event);
              continue;
            }
            overflowCount += 1;
            droppedCount += 1;
            onOverflow(event);
            continue;
          }
          queue.push(event);
        }
      },
      drain(limit = 32) {
        return queue.splice(0, Math.max(0, Math.min(queue.length, limit)));
      },
      clear() {
        queue.length = 0;
      },
      state() {
        return {
          length: queue.length,
          maximum,
          overflowCount,
          droppedCount,
          coalescedCount,
        };
      },
    };
  }

  function relationBatchKey(link, edgeState, active, depthBand) {
    return [
      link.lifecycle || "proposed",
      edgeState === 3 ? "focused" : "normal",
      active ? "active" : "inactive",
      depthBand,
      link.lifecycle === "authoritative" ? "solid" : "dashed",
    ].join(":");
  }

  return {
    createSphereTargets,
    createDurationRing,
    createCooldownWake,
    createEventQueue,
    createGenerationGate,
    createRenderScheduler,
    createSpatialIndex,
    createStageMetrics,
    deterministicUnit,
    drainExpiredPulses,
    fibonacciSpherePoint,
    fitSphereCamera,
    measureSphereQuality,
    normalizeLayoutMode,
    relationBatchKey,
    sphereFogBand,
    sphereFogOpacity,
  };
});
