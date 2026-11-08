"use strict";

((root, factory) => {
  const api = factory();
  if (typeof module === "object" && module.exports) module.exports = api;
  if (root) root.CortexWebGL = api;
})(typeof window === "undefined" ? globalThis : window, () => {
  const FLOATS_PER_VERTEX = 7;

  function nextCapacity(current, needed) {
    let capacity = Math.max(64, current || 0);
    while (capacity < needed) capacity *= 2;
    return capacity;
  }

  function compile(gl, type, source) {
    const shader = gl.createShader(type);
    gl.shaderSource(shader, source);
    gl.compileShader(shader);
    if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) {
      const message = gl.getShaderInfoLog(shader) || "shader compile failed";
      gl.deleteShader(shader);
      throw new Error(message);
    }
    return shader;
  }

  function createProgram(gl) {
    const vertex = compile(gl, gl.VERTEX_SHADER, `#version 300 es
      in vec2 a_position;
      in vec4 a_color;
      in float a_size;
      out vec4 v_color;
      void main() {
        gl_Position = vec4(a_position, 0.0, 1.0);
        gl_PointSize = a_size;
        v_color = a_color;
      }
    `);
    const fragment = compile(gl, gl.FRAGMENT_SHADER, `#version 300 es
      precision mediump float;
      in vec4 v_color;
      uniform bool u_points;
      out vec4 outputColor;
      void main() {
        if (u_points) {
          vec2 offset = gl_PointCoord - vec2(0.5);
          float radius = length(offset);
          if (radius > 0.5) discard;
          outputColor = vec4(
            v_color.rgb,
            v_color.a * (1.0 - smoothstep(0.28, 0.5, radius))
          );
        } else {
          outputColor = v_color;
        }
      }
    `);
    const program = gl.createProgram();
    gl.attachShader(program, vertex);
    gl.attachShader(program, fragment);
    gl.linkProgram(program);
    gl.deleteShader(vertex);
    gl.deleteShader(fragment);
    if (!gl.getProgramParameter(program, gl.LINK_STATUS)) {
      const message = gl.getProgramInfoLog(program) || "program link failed";
      gl.deleteProgram(program);
      throw new Error(message);
    }
    return program;
  }

  function createRenderer(canvas, options = {}) {
    const onStateChange = options.onStateChange || (() => {});
    const contextAttributes = {
      alpha: true,
      antialias: false,
      depth: false,
      preserveDrawingBuffer: false,
      powerPreference: "high-performance",
    };
    let gl = null;
    let program = null;
    let buffer = null;
    let positionLocation = -1;
    let colorLocation = -1;
    let sizeLocation = -1;
    let pointsLocation = null;
    let cpuData = new Float32Array(64 * FLOATS_PER_VERTEX);
    let gpuCapacity = 0;
    let lost = false;
    let fallbackLatched = false;
    let disposed = false;
    let failure = "";
    let frameCount = 0;
    let edgeCount = 0;
    let nodeCount = 0;

    function emitState(state) {
      const visible = state === "ready" || state === "restored";
      if (canvas && canvas.style) canvas.style.opacity = visible ? "1" : "0";
      if (canvas && canvas.dataset) canvas.dataset.rendererState = state;
      onStateChange(state);
    }

    function initialize() {
      if (disposed || fallbackLatched) return false;
      try {
        gl = canvas && canvas.getContext
          ? canvas.getContext("webgl2", contextAttributes)
          : null;
        if (!gl) {
          failure = "webgl2-unavailable";
          fallbackLatched = true;
          emitState("fallback");
          return false;
        }
        program = createProgram(gl);
        buffer = gl.createBuffer();
        positionLocation = gl.getAttribLocation(program, "a_position");
        colorLocation = gl.getAttribLocation(program, "a_color");
        sizeLocation = gl.getAttribLocation(program, "a_size");
        pointsLocation = gl.getUniformLocation(program, "u_points");
        gl.useProgram(program);
        gl.bindBuffer(gl.ARRAY_BUFFER, buffer);
        const stride = FLOATS_PER_VERTEX * Float32Array.BYTES_PER_ELEMENT;
        gl.enableVertexAttribArray(positionLocation);
        gl.vertexAttribPointer(positionLocation, 2, gl.FLOAT, false, stride, 0);
        gl.enableVertexAttribArray(colorLocation);
        gl.vertexAttribPointer(
          colorLocation,
          4,
          gl.FLOAT,
          false,
          stride,
          2 * Float32Array.BYTES_PER_ELEMENT,
        );
        gl.enableVertexAttribArray(sizeLocation);
        gl.vertexAttribPointer(
          sizeLocation,
          1,
          gl.FLOAT,
          false,
          stride,
          6 * Float32Array.BYTES_PER_ELEMENT,
        );
        gl.enable(gl.BLEND);
        gl.blendFunc(gl.SRC_ALPHA, gl.ONE_MINUS_SRC_ALPHA);
        lost = false;
        fallbackLatched = false;
        failure = "";
        emitState("ready");
        return true;
      } catch (error) {
        failure = error && error.message ? error.message : String(error);
        gl = null;
        program = null;
        buffer = null;
        fallbackLatched = true;
        emitState("fallback");
        return false;
      }
    }

    function ensureCpuCapacity(vertices) {
      const needed = vertices * FLOATS_PER_VERTEX;
      if (needed <= cpuData.length) return;
      cpuData = new Float32Array(nextCapacity(cpuData.length, needed));
    }

    function writeVertex(offset, x, y, size, color, opacity, width, height) {
      cpuData[offset] = (x / Math.max(1, width)) * 2 - 1;
      cpuData[offset + 1] = 1 - (y / Math.max(1, height)) * 2;
      cpuData[offset + 2] = color[0] / 255;
      cpuData[offset + 3] = color[1] / 255;
      cpuData[offset + 4] = color[2] / 255;
      cpuData[offset + 5] = opacity;
      cpuData[offset + 6] = size;
      return offset + FLOATS_PER_VERTEX;
    }

    function upload(vertexCount) {
      gl.bindBuffer(gl.ARRAY_BUFFER, buffer);
      const floatCount = vertexCount * FLOATS_PER_VERTEX;
      if (floatCount > gpuCapacity) {
        gpuCapacity = nextCapacity(gpuCapacity, floatCount);
        gl.bufferData(
          gl.ARRAY_BUFFER,
          gpuCapacity * Float32Array.BYTES_PER_ELEMENT,
          gl.DYNAMIC_DRAW,
        );
      }
      gl.bufferSubData(gl.ARRAY_BUFFER, 0, cpuData.subarray(0, floatCount));
    }

    function render(scene) {
      if (
        disposed
        || lost
        || fallbackLatched
        || (!gl && !initialize())
      ) return false;
      const width = Math.max(1, Number(scene.width) || 1);
      const height = Math.max(1, Number(scene.height) || 1);
      const pixelRatio = Math.max(1, Number(scene.pixelRatio) || 1);
      const nodes = scene.nodes || [];
      const links = scene.links || [];
      const nodeState = scene.nodeState || [];
      const edgeState = scene.edgeState || [];
      const fog = scene.fog || (() => 1);
      try {
        gl.viewport(0, 0, Math.round(width * pixelRatio), Math.round(height * pixelRatio));
        gl.clearColor(0, 0, 0, 0);
        gl.clear(gl.COLOR_BUFFER_BIT);
        gl.useProgram(program);

        ensureCpuCapacity(links.length * 2);
        let offset = 0;
        edgeCount = 0;
        for (let index = 0; index < links.length; index += 1) {
          const link = links[index];
          const state = edgeState[index] || 0;
          if (!link || !state) continue;
          const source = nodes[link.source];
          const target = nodes[link.target];
          if (!source || !target || source.viewDepth > 9e8 || target.viewDepth > 9e8) continue;
          if (link.expiresAt && link.expiresAt <= scene.now) continue;
          const depthFade = fog((source.viewDepth + target.viewDepth) / 2);
          let color = state === 3
            ? scene.selectedEdgeColor || [255, 180, 84]
            : scene.baseEdgeColor || [125, 146, 181];
          let opacity = state === 3 ? 0.42 : state === 2 ? 0.075 : 0.025;
          if (state !== 3) {
            const normalEdgeScale = Number(scene.normalEdgeScale);
            opacity *= Number.isFinite(normalEdgeScale)
              ? Math.max(0, normalEdgeScale)
              : 1;
          }
          opacity = Math.min(0.9, opacity * depthFade * (scene.edgeVisibility || 1));
            offset = writeVertex(offset, source.screenX, source.screenY, 1, color, opacity, width, height);
            offset = writeVertex(offset, target.screenX, target.screenY, 1, color, opacity, width, height);
            edgeCount += 1;
        }
        upload(edgeCount * 2);
        gl.uniform1i(pointsLocation, 0);
        gl.drawArrays(gl.LINES, 0, edgeCount * 2);

        ensureCpuCapacity(nodes.length);
        offset = 0;
        nodeCount = 0;
        for (let index = 0; index < nodes.length; index += 1) {
          const node = nodes[index];
          const state = nodeState[index] || 0;
          if (!node || !state || node.viewDepth > 9e8) continue;
          if (node.screenX < -40 || node.screenX > width + 40 || node.screenY < -40 || node.screenY > height + 40) continue;
          const color = node.base || [125, 146, 181];
          const coreOpacity = state === 1 ? 0.22 : state === 3 ? 1 : 0.62;
          const opacity = fog(node.viewDepth) * coreOpacity;
          const size = Math.max(1.5, node.radius * node.screenScale * 2 * pixelRatio);
          offset = writeVertex(offset, node.screenX, node.screenY, size, color, opacity, width, height);
          nodeCount += 1;
        }
        upload(nodeCount);
        gl.uniform1i(pointsLocation, 1);
        gl.drawArrays(gl.POINTS, 0, nodeCount);
        frameCount += 1;
        return true;
      } catch (error) {
        failure = error && error.message ? error.message : String(error);
        fallbackLatched = true;
        emitState("fallback");
        return false;
      }
    }

    function contextLost(event) {
      if (event && event.preventDefault) event.preventDefault();
      if (lost || disposed) return;
      lost = true;
      emitState("lost");
    }

    function contextRestored() {
      if (disposed) return;
      gl = null;
      program = null;
      buffer = null;
      gpuCapacity = 0;
      fallbackLatched = false;
      if (initialize()) emitState("restored");
    }

    if (canvas && canvas.addEventListener) {
      canvas.addEventListener("webglcontextlost", contextLost);
      canvas.addEventListener("webglcontextrestored", contextRestored);
    }
    initialize();

    return {
      render,
      resize(width, height, pixelRatio = 1) {
        if (!canvas) return;
        canvas.width = Math.max(1, Math.round(width * pixelRatio));
        canvas.height = Math.max(1, Math.round(height * pixelRatio));
      },
      snapshot() {
        return {
          mode: gl && !lost && !fallbackLatched ? "webgl2" : "canvas2d",
          lost,
          fallbackLatched,
          failure,
          frameCount,
          edgeCount,
          nodeCount,
          gpuCapacity,
        };
      },
      retry() {
        if (disposed || lost) return false;
        if (gl && buffer) gl.deleteBuffer(buffer);
        if (gl && program) gl.deleteProgram(program);
        if (canvas && canvas.style) canvas.style.opacity = "0";
        gl = null;
        program = null;
        buffer = null;
        gpuCapacity = 0;
        fallbackLatched = false;
        return initialize();
      },
      dispose() {
        if (disposed) return;
        disposed = true;
        if (canvas && canvas.style) canvas.style.opacity = "0";
        if (canvas && canvas.removeEventListener) {
          canvas.removeEventListener("webglcontextlost", contextLost);
          canvas.removeEventListener("webglcontextrestored", contextRestored);
        }
        if (gl && buffer) gl.deleteBuffer(buffer);
        if (gl && program) gl.deleteProgram(program);
        gl = null;
        program = null;
        buffer = null;
      },
    };
  }

  return { createRenderer, nextCapacity };
});
