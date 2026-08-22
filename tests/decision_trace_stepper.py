from __future__ import annotations

import argparse
import json
from http import HTTPStatus
from http.server import ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

from chronovisor.ops import dashboard

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "tests/fixtures/dashboard_decision_trace_states.json"
MODELS = {
    "primary": "qwen3.8:27b-axq4",
    "challenger": "challenger:model",
    "tie_break": "tie:model",
}
WORKFLOW_ROLES = {
    "ingest": "ingest_reconciliation",
    "recall": "recall_auto_apply",
    "audit": "content_correction_classification",
    "improve": "model_eval",
    "repair": "local_repair",
    "typed_graph": "relation_extract",
}
ACTIVE_PHASES = {"A": "generate", "B": "validate", "D": "vote"}


def decision_trace_step_scenarios() -> list[dict[str, Any]]:
    """Build deterministic A-F traces from one observed transition at a time."""

    cases = json.loads(FIXTURE.read_text(encoding="utf-8"))["cases"][:6]
    cases = [
        {**case, "lane_context_tokens": [131_072, None, None]}
        if case["id"] == "A"
        else dict(case)
        for case in cases
    ]
    reasoning_modes = {"A": "off", "B": "low", "D": "high"}
    cases = [
        {
            **case,
            "lane_think": [
                reasoning_modes.get(case["id"], think) if think is not None else None
                for think in case["lane_think"]
            ],
        }
        for case in cases
    ]

    original_models = dashboard._decision_trace_models
    dashboard._decision_trace_models = lambda: dict(MODELS)
    try:
        return [_build_scenario(case, index) for index, case in enumerate(cases)]
    finally:
        dashboard._decision_trace_models = original_models


def _build_scenario(case: dict[str, Any], case_index: int) -> dict[str, Any]:
    request = str(case_index + 1) * 64
    role = WORKFLOW_ROLES[case["workflow"]]
    active: dict[str, dict[str, Any]] = {}
    history: list[dict[str, Any]] = []
    trace_events: list[dict[str, Any]] = []
    frames: list[dict[str, Any]] = []
    latest_decision: dict[str, Any] | None = None
    event_index = 0

    def timestamp() -> str:
        nonlocal event_index
        event_index += 1
        return f"2026-08-15T00:00:{event_index:02d}Z"

    def capture(label: str) -> None:
        trace = dashboard._decision_trace_snapshot(
            list(active.values()),
            history,
            latest_decision,
            trace_events,
            preferred_request_sha256=request,
        )
        frames.append({"index": len(frames), "label": label, "trace": trace})

    capture("idle")
    for lane_index, lane in enumerate(dashboard._DECISION_TRACE_ROLES):
        final_state = case["lane_states"][lane_index]
        if final_state not in {"active", "done", "error"}:
            continue
        tokens = case["lane_context_tokens"][lane_index]
        think = case["lane_think"][lane_index]
        observed = {
            "request_sha256": request,
            "role": f"{role}:{lane}",
            "model": MODELS[lane],
            "think": False if think == "off" else think,
            "required_context_tokens": tokens,
            "requested_context_tokens": tokens,
            "context_tokens": tokens,
        }
        final_phase = (
            ACTIVE_PHASES[case["id"]]
            if final_state == "active"
            else dashboard._DECISION_TRACE_PHASES[-1]
        )
        started_at: str | None = None
        for phase in dashboard._DECISION_TRACE_PHASES:
            observed_at = timestamp()
            started_at = started_at or observed_at
            active[lane] = {
                **observed,
                "phase": phase,
                "attempt": 0,
                "elapsed_seconds": len(frames),
                "started_at": started_at,
                "updated_at": observed_at,
            }
            trace_events.append(
                {
                    **observed,
                    "kind": "phase",
                    "lane": lane,
                    "phase": phase,
                    "status": "active",
                    "attempt": 0,
                    "timestamp": observed_at,
                }
            )
            capture(f"{lane}-{phase}")
            if phase == final_phase:
                break
        if final_state == "active":
            continue

        active.pop(lane)
        observed_at = timestamp()
        session = {
            **observed,
            "kind": "session",
            "timestamp": observed_at,
            "ok": final_state == "done",
            "first_pass_valid": final_state == "done",
            "repair_turns": 0,
        }
        history.append(session)
        trace_events.append(
            {
                **session,
                "lane": lane,
                "phase": "vote",
                "status": "done" if final_state == "done" else "error",
            }
        )
        capture(f"{lane}-complete" if final_state == "done" else f"{lane}-error")

    if case["state"] in {"agreed", "quarantined"}:
        tie_used = case["branch"] in {"tie-break-agreement", "no-safe-quorum"}
        latest_decision = {
            "kind": "decision",
            "timestamp": timestamp(),
            "request_sha256": request,
            "role": role,
            "status": case["state"],
            "artifact_expected": True,
            "pair_agreement": case["branch"] == "pair-agreement",
            "tie_break_used": tie_used,
            "vote_count": 3 if tie_used else 2,
            "valid_votes": 1 if case["state"] == "quarantined" else 2,
            "vote_roles": list(MODELS) if tie_used else ["primary", "challenger"],
            "models": list(MODELS.values()),
        }
        if case["state"] == "quarantined":
            latest_decision.update(
                {
                    "failure_class": "local_consensus_failed",
                    "quarantine_reason": "local_models_did_not_reach_two_vote_quorum",
                }
            )
        history.append(latest_decision)
        trace_events.append({**latest_decision, "phase": "decision"})
        capture("decision")

        if case["state"] == "agreed":
            artifact = {
                "kind": "decision_artifact",
                "timestamp": timestamp(),
                "request_sha256": request,
                "role": role,
                "artifact_status": "sealed",
                "status": "done",
                "phase": "artifact",
            }
            history.append(artifact)
            trace_events.append(artifact)
            capture("artifact-sealed")

    return {"case": case, "request_sha256": request, "frames": frames}


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
    handler: dashboard.DashboardHandler, body: bytes, content_type: str
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
            if self.path != "/trace-stepper-result":
                self.send_error(HTTPStatus.NOT_FOUND)
                return
            if not self._browser_boundary_allows():
                return
            length = int(self.headers.get("Content-Length", "0"))
            if not 0 < length <= 1_048_576:
                self.send_error(HTTPStatus.BAD_REQUEST)
                return
            body = self.rfile.read(length)
            if results is not None:
                results.extend(json.loads(body))
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
    print(f"http://127.0.0.1:{server.server_port}/?case=A&step=0", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


_STEPPER_HARNESS = r"""
const scenarios = __SCENARIOS__;
const scenarioById = new Map(scenarios.map((scenario) => [scenario.case.id, scenario]));
const workflowKeys = [...new Set(scenarios.map((scenario) => scenario.case.workflow))];
const params = new URLSearchParams(location.search);
const api = window.__chronovisorDashboardTest;
let currentScenario = scenarioById.get(params.get("case")) || scenarios[0];
let currentStep = Math.min(
  Math.max(Number.parseInt(params.get("step") || "0", 10) || 0, 0),
  currentScenario.frames.length - 1,
);

function renderCurrent() {
  const frame = currentScenario.frames[currentStep];
  const trace = frame.trace;
  api.renderProcessingActivity({
    generated_at: `2026-08-15T00:01:${String(currentStep).padStart(2, "0")}Z`,
    revision: `${currentScenario.case.id}-${currentStep}`,
    active_count: trace.active ? 1 : 0,
    lanes: workflowKeys.map((key) => ({
      key,
      label: key,
      state: key === currentScenario.case.workflow && trace.active ? "active" : "idle",
      current_step: key === currentScenario.case.workflow
        ? currentScenario.case.processing_stage
        : null,
      steps: [],
    })),
  });
  api.renderDecisionTrace({ decision_trace: trace });
  document.documentElement.dataset.traceStepperReady = "true";
  document.documentElement.dataset.traceStepperCase = currentScenario.case.id;
  document.documentElement.dataset.traceStepperStep = String(currentStep);
  document.documentElement.dataset.traceStepperLabel = frame.label;
  document.querySelector("#trace-stepper-position").textContent =
    `${currentScenario.case.id} · ${currentStep + 1}/${currentScenario.frames.length} · ${frame.label}`;
  document.querySelector("#trace-stepper-prev").disabled = currentStep === 0;
  document.querySelector("#trace-stepper-next").disabled =
    currentStep === currentScenario.frames.length - 1;
  return frame;
}

function selectScenario(id) {
  const selected = scenarioById.get(id);
  if (!selected) throw new Error(`unknown Decision Trace scenario: ${id}`);
  currentScenario = selected;
  currentStep = 0;
  document.querySelector("#trace-stepper-case").value = id;
  return renderCurrent();
}

function gotoStep(index) {
  if (!Number.isInteger(index) || index < 0 || index >= currentScenario.frames.length) {
    return false;
  }
  currentStep = index;
  renderCurrent();
  return true;
}

function nextStep() {
  return gotoStep(currentStep + 1);
}

function previousStep() {
  return gotoStep(currentStep - 1);
}

function inspectCurrent() {
  const frame = currentScenario.frames[currentStep];
  const pathState = (node) => {
    const style = getComputedStyle(node);
    return {
      state: node.dataset.state,
      dash: style.strokeDasharray,
      length: node.getTotalLength(),
    };
  };
  const scroller = document.querySelector("#decision-trace-scroll");
  const harness = document.querySelector("#decision-trace-harness");
  const selectedContext = document.querySelector("[data-context-option].selected");
  const selectedReasoning = document.querySelector("[data-reasoning-key].selected");
  const opacity = (node) => node ? getComputedStyle(node).opacity : null;
  const paths = Object.fromEntries([
    ...[...document.querySelectorAll("[data-path-key]")]
      .map((node) => [node.dataset.pathKey, node]),
    ...[...document.querySelectorAll("[data-reasoning-output]")]
      .map((node) => [`reasoning-output-${node.dataset.reasoningOutput}`, node]),
  ].map(([key, node]) => [key, pathState(node)]));
  const rails = Object.fromEntries([...document.querySelectorAll("[data-decision-lane]")]
    .map((lane) => [lane.dataset.decisionLane, [...lane.querySelectorAll("[data-lane-path]")]
      .map((node) => ({ key: node.dataset.lanePath, ...pathState(node) }))]));
  const scrollerBounds = scroller.getBoundingClientRect();
  const harnessBounds = harness.getBoundingClientRect();
  return {
    case: currentScenario.case.id,
    step: currentStep,
    label: frame.label,
    projectionStatus: harness.dataset.projectionStatus,
    selectedContext: selectedContext?.dataset.contextTokens ?? null,
    selectedReasoning: selectedReasoning?.dataset.reasoningKey ?? null,
    contextCoreOpacities: [...document.querySelectorAll(
      "[data-context-option] .trace-selected-core"
    )].map(opacity),
    reasoningCoreOpacities: [...document.querySelectorAll(
      "[data-reasoning-key] .trace-selected-core"
    )].map(opacity),
    nodes: Object.fromEntries([...document.querySelectorAll("[data-trace-key]")]
      .map((node) => [node.dataset.traceKey, node.dataset.state])),
    paths,
    rails,
    layout: {
      clientWidth: scroller.clientWidth,
      scrollWidth: scroller.scrollWidth,
      fitsWidth: Math.abs(harnessBounds.width - scrollerBounds.width) <= 1,
      rightReachable: harnessBounds.right <= scrollerBounds.right + 1,
    },
  };
}

function installControls() {
  const controls = document.createElement("div");
  controls.id = "trace-stepper-controls";
  controls.style.cssText = [
    "position:fixed", "z-index:1000", "right:16px", "top:16px", "display:flex",
    "gap:8px", "align-items:center", "padding:8px", "border:1px solid #334155",
    "border-radius:8px", "background:#061724", "color:#dbeafe", "font:12px monospace",
  ].join(";");
  controls.innerHTML = `
    <select id="trace-stepper-case" aria-label="Decision Trace case"></select>
    <button id="trace-stepper-prev" type="button">Previous</button>
    <button id="trace-stepper-next" type="button">Next step</button>
    <span id="trace-stepper-position"></span>`;
  document.body.append(controls);
  const select = document.querySelector("#trace-stepper-case");
  for (const scenario of scenarios) {
    const option = document.createElement("option");
    option.value = scenario.case.id;
    option.textContent = `${scenario.case.id} · ${scenario.case.branch}`;
    select.append(option);
  }
  select.value = currentScenario.case.id;
  select.addEventListener("change", () => selectScenario(select.value));
  document.querySelector("#trace-stepper-prev").addEventListener("click", previousStep);
  document.querySelector("#trace-stepper-next").addEventListener("click", nextStep);
  if (params.has("capture") || params.has("audit")) controls.style.opacity = "0";
  if (params.has("capture")) {
    document.documentElement.style.zoom = "0.75";
    document.querySelectorAll(
      "main > .topbar, main > .metrics-grid, .processing-lanes-card, main > .dashboard-grid"
    ).forEach((node) => { node.style.display = "none"; });
  }
}

async function auditAllFrames() {
  const results = [];
  for (const scenario of scenarios) {
    selectScenario(scenario.case.id);
    results.push(inspectCurrent());
    while (nextStep()) results.push(inspectCurrent());
  }
  await fetch("/trace-stepper-result", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(results),
  });
  document.documentElement.dataset.traceStepperAudit = "done";
}

addEventListener("DOMContentLoaded", async () => {
  installControls();
  renderCurrent();
  document.querySelector("#decision-trace-panel")?.scrollIntoView({ block: "start" });
  window.__traceStepper = {
    scenarios: () => scenarios.map(({ case: fixture, frames }) => ({
      id: fixture.id,
      branch: fixture.branch,
      steps: frames.map(({ index, label }) => ({ index, label })),
    })),
    current: () => ({
      case: currentScenario.case.id,
      step: currentStep,
      label: currentScenario.frames[currentStep].label,
    }),
    inspect: inspectCurrent,
    next: nextStep,
    previous: previousStep,
    select: selectScenario,
    goto: gotoStep,
  };
  if (params.has("audit")) await auditAllFrames();
});
"""


if __name__ == "__main__":
    main()
