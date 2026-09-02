from __future__ import annotations

import argparse
import json
from http import HTTPStatus
from http.server import ThreadingHTTPServer
from typing import Any
from urllib.parse import urlsplit

from chronovisor.ops import dashboard
from chronovisor.ops.decision_trace_projection import (
    TRACE_PROJECTION_SCHEMA,
    TRACE_WORKFLOWS,
)


def decision_trace_step_scenarios() -> list[dict[str, Any]]:
    scenarios: list[dict[str, Any]] = []
    for pipeline, workflow in TRACE_WORKFLOWS.items():
        for route_name, route in workflow["routes"].items():
            frames = []
            for cursor, milestone in enumerate(route):
                terminal = cursor == len(route) - 1
                target_state = (
                    "error"
                    if terminal and milestone == "hold"
                    else "done"
                    if terminal
                    else "active"
                )
                projection = {
                    "schema": TRACE_PROJECTION_SCHEMA,
                    "execution_id": f"stepper-{pipeline}-{route_name}",
                    "graph_id": f"workflow:{pipeline}:v2",
                    "revision": str(cursor),
                    "pipeline": pipeline,
                    "trace_state": "quarantined"
                    if target_state == "error"
                    else "agreed"
                    if terminal
                    else "active",
                    "outcome_kind": "hold" if target_state == "error" else "active",
                    "mode": "single",
                    "single_model": True,
                    "authority_kind": "single_model_v1",
                    "context": {
                        "selected_tokens": 65_536,
                        "options": [
                            {"tokens": 32_768, "label": "32K", "selected": False},
                            {"tokens": 65_536, "label": "65K", "selected": True},
                            {"tokens": 98_304, "label": "98K", "selected": False},
                            {"tokens": 131_072, "label": "131K", "selected": False},
                        ],
                        "label": "required 55K → selected 65K",
                    },
                    "reasoning": {
                        "selected": "low",
                        "options": ["off", "low", "medium", "high"],
                    },
                    "workflow": {
                        "nodes": [dict(node) for node in workflow["nodes"]],
                        "edges": [dict(edge) for edge in workflow["edges"]],
                        "route": route_name,
                        "route_node_ids": list(route),
                        "target_cursor": cursor,
                        "target_node": milestone,
                        "target_state": target_state,
                    },
                    "labels": {
                        "validation": "Held"
                        if target_state == "error"
                        else "Validated"
                        if terminal
                        else "Validating",
                        "hold": "Test hold",
                    },
                    "authority": {
                        "kind": "single_model_v1",
                        "label": "Single Authority",
                        "model": "Qwen",
                        "revision": "fixture",
                        "target": 1,
                        "validated": terminal and target_state == "done",
                        "repair_is_vote": False,
                    },
                }
                frames.append(
                    {
                        "cursor": cursor,
                        "milestone": milestone,
                        "trace": {
                            "state": projection["trace_state"],
                            "active": not terminal,
                            "request_sha256": projection["execution_id"],
                            "task_role": pipeline,
                            "summary": f"{pipeline} · {milestone}",
                            "events": [],
                            "lanes": [],
                            "projection": projection,
                        },
                    }
                )
            scenarios.append(
                {
                    "id": f"{pipeline}-{route_name}",
                    "pipeline": pipeline,
                    "route": route_name,
                    "frames": frames,
                }
            )
    return scenarios


_STEPPER_HARNESS = r"""
const scenarios = __SCENARIOS__;
const api = window.__chronovisorDashboardTest;
const query = new URLSearchParams(location.search);
const selectedId = query.get("scenario");
const selectedStep = Number(query.get("step") || 0);
const selectedContext = Number(query.get("context") || 0);
const selectedReasoning = query.get("reasoning");

document.documentElement.dataset.traceStepper = "true";
document.head.insertAdjacentHTML("beforeend", `<style>
  html[data-trace-stepper="true"] body { padding: 0; }
  html[data-trace-stepper="true"] .activity-bar,
  html[data-trace-stepper="true"] .topbar,
  html[data-trace-stepper="true"] .metrics-grid,
  html[data-trace-stepper="true"] .processing-lanes-card,
  html[data-trace-stepper="true"] .decision-summary,
  html[data-trace-stepper="true"] .trace-legacy-data,
  html[data-trace-stepper="true"] .dashboard-grid { display: none; }
  html[data-trace-stepper="true"] .shell { margin: 0; max-width: none; padding: 16px; }
  html[data-trace-stepper="true"] .processing-panel { margin: 0; }
</style>`);

const nextPaint = () => new Promise((resolve) => setTimeout(resolve, 0));

function renderFrame(scenario, frame) {
  const trace = structuredClone(frame.trace);
  if (selectedContext) {
    trace.projection.context.selected_tokens = selectedContext;
    trace.projection.context.options.forEach((option) => {
      option.selected = option.tokens === selectedContext;
    });
    const option = trace.projection.context.options.find((item) => item.selected);
    trace.projection.context.label = `required 55K → selected ${option?.label || "—"}`;
  }
  if (selectedReasoning) trace.projection.reasoning.selected = selectedReasoning;
  const workflow = trace.projection.workflow;
  api.updateDecisionSvgHarness(trace, null, {
    ...workflow,
    cursor: frame.cursor,
  });
  document.getElementById("decision-elapsed").textContent = scenario.id;
  document.getElementById("decision-context").textContent = `Step ${frame.cursor + 1}/${scenario.frames.length}`;
  document.getElementById("decision-trace-caption").textContent = frame.milestone;
  document.body.dataset.traceReady = "true";
  document.body.dataset.scenario = scenario.id;
  document.body.dataset.step = String(frame.cursor);
}

function captureFrame(scenario, frame) {
  const harness = document.getElementById("decision-trace-harness");
  const paths = [...harness.querySelectorAll("[data-workflow-edge]")].map((group) => {
    const path = group.querySelector("path.selected") || group.querySelector("path");
    return {
      id: group.dataset.workflowEdge,
      source: group.dataset.source,
      target: group.dataset.target,
      kind: group.dataset.kind,
      state: group.dataset.state,
      d: path.getAttribute("d"),
    };
  });
  const guides = [...harness.querySelectorAll(".trace-context-guide, .trace-reasoning-guide")].map((guide) => ({
    className: guide.getAttribute("class"),
    d: guide.getAttribute("d"),
  }));
  const sharpCorners = [...harness.querySelectorAll(".trace-path, .trace-context-guide, .trace-reasoning-guide")].flatMap((path) => {
    const d = path.getAttribute("d") || "";
    const commands = d.match(/[LHVQ]/g) || [];
    return commands.length > 1 && !commands.includes("Q") ? [d] : [];
  });
  const cornerRadii = paths.flatMap((path) => {
    const tokens = path.d.match(/[MLHVQ]|-?\d+(?:\.\d+)?/g) || [];
    const radii = [];
    let command = "";
    let x = 0;
    let y = 0;
    for (let index = 0; index < tokens.length;) {
      if (/^[MLHVQ]$/.test(tokens[index])) command = tokens[index++];
      if (command === "M" || command === "L") {
        x = Number(tokens[index++]);
        y = Number(tokens[index++]);
      } else if (command === "H") {
        x = Number(tokens[index++]);
      } else if (command === "V") {
        y = Number(tokens[index++]);
      } else if (command === "Q") {
        const controlX = Number(tokens[index++]);
        const controlY = Number(tokens[index++]);
        const nextX = Number(tokens[index++]);
        const nextY = Number(tokens[index++]);
        radii.push({
          edge: path.id,
          incoming: Math.hypot(controlX - x, controlY - y),
          outgoing: Math.hypot(controlX - nextX, controlY - nextY),
        });
        x = nextX;
        y = nextY;
      }
      command = "";
    }
    return radii;
  });
  const nodes = [...harness.querySelectorAll("[data-workflow-node]")].map((node) => ({
    id: node.dataset.workflowNode,
    state: node.dataset.state,
    transform: node.getAttribute("transform"),
    shape: node.querySelector(":scope > path") ? "decision" : "circle",
  }));
  const labels = [...harness.querySelectorAll(".trace-node text, .trace-branch-label")].map((label) => {
    const rect = label.getBoundingClientRect();
    return {
      text: label.textContent,
      rect: { left: rect.left, right: rect.right, top: rect.top, bottom: rect.bottom },
    };
  });
  const pathLabelCollisions = [];
  const endpointErrors = [];
  const decisionCenterEndpoints = [];
  for (const group of harness.querySelectorAll("[data-workflow-edge]")) {
    const path = group.querySelector("path.selected") || group.querySelector("path");
    const matrix = path.getScreenCTM();
    const length = path.getTotalLength();
    for (let distance = 0; distance <= length; distance += 3) {
      const rawPoint = path.getPointAtLength(distance);
      const point = new DOMPoint(rawPoint.x, rawPoint.y).matrixTransform(matrix);
      for (const label of labels) {
        if (
          point.x > label.rect.left - 1 && point.x < label.rect.right + 1
          && point.y > label.rect.top - 1 && point.y < label.rect.bottom + 1
        ) {
          pathLabelCollisions.push([group.dataset.workflowEdge, label.text]);
        }
      }
    }
    for (const [end, nodeId, distance] of [
      ["source", group.dataset.source, 0],
      ["target", group.dataset.target, length],
    ]) {
      const rawPoint = path.getPointAtLength(distance);
      const point = new DOMPoint(rawPoint.x, rawPoint.y).matrixTransform(matrix);
      const shape = harness.querySelector(`[data-workflow-node="${nodeId}"] > circle, [data-workflow-node="${nodeId}"] > path`);
      const rect = shape.getBoundingClientRect();
      const center = { x: rect.left + rect.width / 2, y: rect.top + rect.height / 2 };
      const vertices = [
        { x: center.x, y: rect.top },
        { x: rect.right, y: center.y },
        { x: center.x, y: rect.bottom },
        { x: rect.left, y: center.y },
      ];
      const distanceToSegment = (candidate, first, second) => {
        const dx = second.x - first.x;
        const dy = second.y - first.y;
        const lengthSquared = dx * dx + dy * dy;
        const ratio = Math.max(0, Math.min(1,
          ((candidate.x - first.x) * dx + (candidate.y - first.y) * dy) / lengthSquared,
        ));
        return Math.hypot(
          candidate.x - (first.x + ratio * dx),
          candidate.y - (first.y + ratio * dy),
        );
      };
      const centerDistance = Math.hypot(point.x - center.x, point.y - center.y);
      const endpointDistance = shape.tagName.toLowerCase() === "path"
        ? Math.min(...vertices.map((vertex, index) => distanceToSegment(
            point,
            vertex,
            vertices[(index + 1) % vertices.length],
          )))
        : Math.abs(centerDistance - rect.width / 2);
      endpointErrors.push({
        edge: group.dataset.workflowEdge,
        end,
        distance: endpointDistance,
      });
      if (
        shape.tagName.toLowerCase() === "path"
        && Math.hypot(point.x - center.x, point.y - center.y) < 0.75
      ) {
        decisionCenterEndpoints.push({ edge: group.dataset.workflowEdge, end, node: nodeId });
      }
    }
  }
  const segments = (d) => {
    const tokens = d.match(/[MLHVQ]|-?\d+(?:\.\d+)?/g) || [];
    const rows = [];
    let command = "";
    let x = 0;
    let y = 0;
    for (let index = 0; index < tokens.length;) {
      if (/^[MLHVQ]$/.test(tokens[index])) command = tokens[index++];
      if (command === "M") {
        x = Number(tokens[index++]);
        y = Number(tokens[index++]);
        command = "";
      } else if (command === "L") {
        const nextX = Number(tokens[index++]);
        const nextY = Number(tokens[index++]);
        rows.push({ x1: x, y1: y, x2: nextX, y2: nextY });
        x = nextX;
        y = nextY;
      } else if (command === "H") {
        const nextX = Number(tokens[index++]);
        rows.push({ x1: x, y1: y, x2: nextX, y2: y });
        x = nextX;
      } else if (command === "V") {
        const nextY = Number(tokens[index++]);
        rows.push({ x1: x, y1: y, x2: x, y2: nextY });
        y = nextY;
      } else if (command === "Q") {
        index += 2;
        const nextX = Number(tokens[index++]);
        const nextY = Number(tokens[index++]);
        rows.push({ x1: x, y1: y, x2: nextX, y2: nextY });
        x = nextX;
        y = nextY;
      }
    }
    return rows;
  };
  const milestoneTurns = [];
  const route = frame.trace.projection.workflow.route_node_ids;
  route.slice(1, -1).forEach((nodeId, index) => {
    const node = nodes.find((item) => item.id === nodeId);
    if (node?.shape !== "circle") return;
    const previousId = route[index];
    const nextId = route[index + 2];
    const incoming = paths.find((path) => (
      path.source === previousId && path.target === nodeId
    ));
    const outgoing = paths.find((path) => (
      path.source === nodeId && path.target === nextId
    ));
    if (incoming?.kind !== "main" || outgoing?.kind !== "main") return;
    const tangent = (edge, atEnd) => {
      if (!edge) return null;
      const group = harness.querySelector(`[data-workflow-edge="${edge.id}"]`);
      const path = group?.querySelector("path.selected") || group?.querySelector("path");
      const length = path?.getTotalLength() || 0;
      if (!length) return null;
      const first = path.getPointAtLength(atEnd ? Math.max(0, length - 1) : 0);
      const second = path.getPointAtLength(atEnd ? length : Math.min(1, length));
      const dx = second.x - first.x;
      const dy = second.y - first.y;
      return Math.abs(dx) >= Math.abs(dy)
        ? [Math.sign(dx), 0]
        : [0, Math.sign(dy)];
    };
    const incomingDirection = tangent(incoming, true);
    const outgoingDirection = tangent(outgoing, false);
    if (
      incomingDirection && outgoingDirection
      && (incomingDirection[0] !== outgoingDirection[0]
        || incomingDirection[1] !== outgoingDirection[1])
    ) {
      milestoneTurns.push({
        node: nodeId,
        incoming: incoming.id,
        outgoing: outgoing.id,
        incomingDirection,
        outgoingDirection,
      });
    }
  });
  const between = (value, first, second) =>
    value >= Math.min(first, second) && value <= Math.max(first, second);
  const intersects = (left, right) => {
    const leftVertical = left.x1 === left.x2;
    const rightVertical = right.x1 === right.x2;
    const leftHorizontal = left.y1 === left.y2;
    const rightHorizontal = right.y1 === right.y2;
    if ((!leftVertical && !leftHorizontal) || (!rightVertical && !rightHorizontal)) {
      return false;
    }
    if (leftVertical !== rightVertical) {
      const vertical = leftVertical ? left : right;
      const horizontal = leftVertical ? right : left;
      return between(vertical.x1, horizontal.x1, horizontal.x2)
        && between(horizontal.y1, vertical.y1, vertical.y2);
    }
    if (leftVertical) {
      return left.x1 === right.x1
        && Math.max(Math.min(left.y1, left.y2), Math.min(right.y1, right.y2))
          <= Math.min(Math.max(left.y1, left.y2), Math.max(right.y1, right.y2));
    }
    return left.y1 === right.y1
      && Math.max(Math.min(left.x1, left.x2), Math.min(right.x1, right.x2))
      <= Math.min(Math.max(left.x1, left.x2), Math.max(right.x1, right.x2));
  };
  const overlapLength = (left, right) => {
    const leftVertical = left.x1 === left.x2;
    const rightVertical = right.x1 === right.x2;
    const leftHorizontal = left.y1 === left.y2;
    const rightHorizontal = right.y1 === right.y2;
    if ((!leftVertical && !leftHorizontal) || (!rightVertical && !rightHorizontal)) {
      return 0;
    }
    if (leftVertical !== rightVertical) return 0;
    if (leftVertical) {
      if (left.x1 !== right.x1) return 0;
      return Math.max(0,
        Math.min(Math.max(left.y1, left.y2), Math.max(right.y1, right.y2))
          - Math.max(Math.min(left.y1, left.y2), Math.min(right.y1, right.y2))
      );
    }
    if (left.y1 !== right.y1) return 0;
    return Math.max(0,
      Math.min(Math.max(left.x1, left.x2), Math.max(right.x1, right.x2))
        - Math.max(Math.min(left.x1, left.x2), Math.min(right.x1, right.x2))
    );
  };
  const pathIntersections = [];
  const pathOverlaps = [];
  paths.forEach((left, index) => paths.slice(index + 1).forEach((right) => {
    const overlap = Math.max(
      ...segments(left.d).flatMap((a) => segments(right.d).map((b) => overlapLength(a, b))),
    );
    if (overlap > 0.5) pathOverlaps.push([left.id, right.id, overlap]);
    if (
      left.source === right.source || left.source === right.target
      || left.target === right.source || left.target === right.target
    ) return;
    if (segments(left.d).some((a) => segments(right.d).some((b) => intersects(a, b)))) {
      pathIntersections.push([left.id, right.id]);
    }
  }));
  const guideOverlaps = [];
  for (const guide of guides) {
    const guideSegments = segments(guide.d);
    guideSegments.forEach((left, index) => guideSegments.slice(index + 1).forEach((right) => {
      const overlap = overlapLength(left, right);
      if (overlap > 0.5) guideOverlaps.push([guide.className, overlap]);
    }));
  }
  const textOverlaps = [];
  labels.forEach((left, index) => labels.slice(index + 1).forEach((right) => {
    if (
      left.rect.left < right.rect.right && left.rect.right > right.rect.left
      && left.rect.top < right.rect.bottom && left.rect.bottom > right.rect.top
    ) textOverlaps.push([left.text, right.text]);
  }));
  const reasoningLabelClearances = [...harness.querySelectorAll(".trace-reasoning")].map((group) => {
    const circleRect = group.querySelector("circle").getBoundingClientRect();
    const labelRect = group.querySelector("[data-reasoning-label]").getBoundingClientRect();
    return {
      key: group.dataset.reasoningKey,
      clearance: labelRect.top - (circleRect.top + circleRect.height / 2),
    };
  });
  const decisionQuestions = [...harness.querySelectorAll(
    "[data-workflow-nodes] .trace-node-decision",
  )].map((node) => {
    const shape = node.querySelector(":scope > path");
    const shapeRect = shape.getBoundingClientRect();
    const question = node.querySelector(":scope > text");
    const questionRect = question.getBoundingClientRect();
    return {
      id: node.dataset.workflowNode,
      fill: getComputedStyle(shape).fill,
      text: question.textContent,
      inside: questionRect.left >= shapeRect.left
        && questionRect.right <= shapeRect.right
        && questionRect.top >= shapeRect.top
        && questionRect.bottom <= shapeRect.bottom,
    };
  });
  const decisionBranchLabels = [...harness.querySelectorAll(
    "[data-workflow-edge]",
  )].flatMap((group) => {
    const source = harness.querySelector(
      `[data-workflow-node="${group.dataset.source}"]`,
    );
    const label = group.querySelector(".trace-branch-label");
    if (!source?.classList.contains("trace-node-decision") || !label) return [];
    const shapeRect = source.querySelector(":scope > path").getBoundingClientRect();
    const labelRect = label.getBoundingClientRect();
    const dx = Math.max(shapeRect.left - labelRect.right, 0, labelRect.left - shapeRect.right);
    const dy = Math.max(shapeRect.top - labelRect.bottom, 0, labelRect.top - shapeRect.bottom);
    return [{
      id: group.dataset.workflowEdge,
      text: label.textContent,
      state: group.dataset.state,
      labelState: label.dataset.state,
      distance: Math.hypot(dx, dy),
    }];
  });
  return {
    scenario: scenario.id,
    pipeline: scenario.pipeline,
    route: scenario.route,
    cursor: frame.cursor,
    milestone: frame.milestone,
    graphId: harness.dataset.graphId,
    visibleCursor: Number(harness.dataset.visibleCursor),
    geometry: JSON.stringify({
      paths: paths.map(({ id, d }) => [id, d]),
      guides,
      nodes: nodes.map(({ id, transform }) => [id, transform]),
    }),
    paths,
    guides,
    sharpCorners,
    cornerRadii,
    milestoneTurns,
    nodes,
    pathLabelCollisions: [...new Set(pathLabelCollisions.map(JSON.stringify))].map(JSON.parse),
    pathIntersections,
    pathOverlaps,
    guideOverlaps,
    endpointErrors,
    decisionCenterEndpoints,
    textOverlaps,
    reasoningLabelClearances,
    decisionQuestions,
    decisionBranchLabels,
  };
}

addEventListener("DOMContentLoaded", async () => {
  await document.fonts.ready;
  if (selectedId) {
    const scenario = scenarios.find((item) => item.id === selectedId);
    const frame = scenario?.frames[Math.min(selectedStep, scenario.frames.length - 1)];
    if (!scenario || !frame) throw new Error("unknown stepper scenario");
    renderFrame(scenario, frame);
    window.__traceStepperCapture = captureFrame(scenario, frame);
    return;
  }

  const results = [];
  for (const scenario of scenarios) {
    for (const frame of scenario.frames) {
      renderFrame(scenario, frame);
      await nextPaint();
      results.push(captureFrame(scenario, frame));
    }
  }
  await fetch("/trace-stepper-result", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(results),
  });
});
"""


def stepper_harness(scenarios: list[dict[str, Any]]) -> str:
    return _STEPPER_HARNESS.replace(
        "__SCENARIOS__", json.dumps(scenarios, ensure_ascii=False)
    )


def stepper_page() -> str:
    return (
        (dashboard.STATIC_DIR / "index.html")
        .read_text(encoding="utf-8")
        .replace(
            '<script src="/static/app-client.js"></script>',
            '<script src="/trace-stepper.js"></script>',
        )
    )


def _send_bytes(
    handler: dashboard.DashboardHandler,
    body: bytes,
    content_type: str,
) -> None:
    handler.send_response(HTTPStatus.OK)
    handler.send_header("Content-Type", content_type)
    handler.send_header("Content-Length", str(len(body)))
    handler.send_header("Cache-Control", "no-store")
    handler.end_headers()
    handler.wfile.write(body)


def stepper_handler(
    scenarios: list[dict[str, Any]],
    results: list[dict[str, Any]] | None = None,
) -> type[dashboard.DashboardHandler]:
    page = stepper_page().encode()
    harness = stepper_harness(scenarios).encode()

    class Handler(dashboard.DashboardHandler):
        def do_GET(self) -> None:
            request_path = urlsplit(self.path).path
            if request_path in {"/", "/trace-stepper.js"}:
                if self._browser_boundary_allows():
                    _send_bytes(
                        self,
                        page if request_path == "/" else harness,
                        "text/html; charset=utf-8"
                        if request_path == "/"
                        else "text/javascript; charset=utf-8",
                    )
                return
            super().do_GET()

        def do_POST(self) -> None:
            if urlsplit(self.path).path != "/trace-stepper-result":
                self.send_error(HTTPStatus.NOT_FOUND)
                return
            if not self._browser_boundary_allows():
                return
            length = int(self.headers.get("Content-Length", "0"))
            if not 0 < length <= 4_194_304:
                self.send_error(HTTPStatus.BAD_REQUEST)
                return
            payload = json.loads(self.rfile.read(length))
            if results is not None:
                results.extend(payload)
            self.send_response(HTTPStatus.NO_CONTENT)
            self.end_headers()

        def log_message(self, _format: str, *_args: object) -> None:
            return

    return Handler


def main() -> None:
    parser = argparse.ArgumentParser(description="Deterministic Decision Trace stepper")
    parser.add_argument("--port", type=int, default=8877)
    args = parser.parse_args()
    scenarios = decision_trace_step_scenarios()
    server = ThreadingHTTPServer(("127.0.0.1", args.port), stepper_handler(scenarios))
    print(
        f"http://127.0.0.1:{server.server_port}/?scenario=ingest-success&step=0",
        flush=True,
    )
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
