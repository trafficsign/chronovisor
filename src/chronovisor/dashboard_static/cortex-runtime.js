"use strict";

((root, factory) => {
  const api = factory();
  if (typeof module === "object" && module.exports) module.exports = api;
  if (root) root.CortexRuntime = api;
})(typeof window === "undefined" ? globalThis : window, () => {
  const DEFAULT_SAMPLE_CAPACITY = 240;

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
              queue[existing] = mergeOverflow(queue[existing], event);
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
    createDurationRing,
    createCooldownWake,
    createEventQueue,
    createGenerationGate,
    createRenderScheduler,
    createSpatialIndex,
    createStageMetrics,
    drainExpiredPulses,
    relationBatchKey,
  };
});
