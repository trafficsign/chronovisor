from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path
from typing import Any

import pytest

from chronovisor.ops import decision_trace_projection as projection

ROOT = Path(__file__).resolve().parents[1]

EXPECTED_WORKFLOW_MILESTONES = {
    "ingest": (
        "raw",
        "triage",
        "target",
        "generate",
        "authority",
        "result",
        "change",
        "apply",
        "publish",
        "readback",
        "complete",
        "hold",
    ),
    "recall": (
        "search",
        "rerank",
        "authority",
        "result",
        "commit",
        "readback",
        "complete",
        "hold",
    ),
    "audit": ("select", "inspect", "consensus", "result", "report", "complete", "hold"),
    "improve": (
        "discover",
        "generate",
        "verify",
        "result",
        "apply",
        "readback",
        "complete",
        "hold",
    ),
    "repair": (
        "detect",
        "local_fix",
        "verify",
        "result",
        "escalate",
        "readback",
        "complete",
        "hold",
    ),
    "typed_graph": (
        "discover",
        "extract",
        "verify",
        "consolidate",
        "evaluate",
        "result",
        "promote",
        "readback",
        "complete",
        "hold",
    ),
}

COMMON_PLAN_MILESTONES = (
    "packet",
    "preflight",
    "execution_plan",
    "context_choice",
    "headroom",
    "reasoning_choice",
    "fit",
)


def _trace(**extra: Any) -> dict[str, Any]:
    return {
        "state": "active",
        "active": True,
        "request_sha256": "child-request",
        "task_role": "ingest_triage",
        "event_count": 4,
        "lanes": [
            {
                "key": "primary",
                "state": "active",
                "phase": "generate",
                "think": "low",
                "model": "Qwen",
                "context_tokens": 65_536,
            }
        ],
        **extra,
    }


def test_six_workflow_patterns_inventory_every_milestone() -> None:
    assert set(projection.TRACE_WORKFLOWS) == set(EXPECTED_WORKFLOW_MILESTONES)
    for pipeline, expected in EXPECTED_WORKFLOW_MILESTONES.items():
        workflow = projection.TRACE_WORKFLOWS[pipeline]
        all_expected = COMMON_PLAN_MILESTONES + expected
        assert tuple(node["id"] for node in workflow["nodes"]) == all_expected
        node_ids = set(all_expected)
        assert len(node_ids) == len(all_expected)
        assert all(
            edge["source"] in node_ids and edge["target"] in node_ids
            for edge in workflow["edges"]
        )
        exits: dict[str, int] = {}
        for edge in workflow["edges"]:
            exits[edge["source"]] = exits.get(edge["source"], 0) + 1
        assert all(
            exits[node["id"]] >= 2
            for node in workflow["nodes"]
            if node["type"] == "decision"
        )
        assert all(
            node["label"].endswith("?")
            for node in workflow["nodes"]
            if node["type"] == "decision"
        )
        decision_ids = {
            node["id"] for node in workflow["nodes"] if node["type"] == "decision"
        }
        assert all(
            edge.get("label")
            for edge in workflow["edges"]
            if edge["source"] in decision_ids
        )
        assert all(
            tuple(route[: len(COMMON_PLAN_MILESTONES)]) == COMMON_PLAN_MILESTONES
            and route[-1] in {"complete", "hold"}
            for route in workflow["routes"].values()
        )


def test_projection_restores_context_and_reasoning_choices() -> None:
    result = projection.project_decision_trace(
        _trace(
            lanes=[
                {
                    "key": "primary",
                    "state": "active",
                    "phase": "generate",
                    "think": "low",
                    "model": "Qwen",
                    "context_tokens": 65_536,
                    "required_context_tokens": 55_000,
                }
            ]
        ),
        pipeline="ingest",
        processing_lane={"work_item": "child"},
    )

    assert result["context"] == {
        "selected_tokens": 65_536,
        "options": [
            {"tokens": 32_768, "label": "32K", "selected": False},
            {"tokens": 65_536, "label": "65K", "selected": True},
            {"tokens": 98_304, "label": "98K", "selected": False},
            {"tokens": 131_072, "label": "131K", "selected": False},
        ],
        "label": "required 55K → selected 65K",
    }
    assert result["reasoning"] == {
        "selected": "low",
        "options": ["off", "low", "medium", "high"],
    }


@pytest.mark.parametrize(
    ("pipeline", "current_step", "expected"),
    [
        ("recall", "primary", "authority"),
        ("audit", "inspect", "inspect"),
        ("improve", "verify", "verify"),
        ("repair", "local_fix", "local_fix"),
        ("typed_graph", "extract", "extract"),
    ],
)
def test_each_pipeline_projects_its_observed_stage(
    pipeline: str,
    current_step: str,
    expected: str,
) -> None:
    result = projection.project_decision_trace(
        _trace(task_role=pipeline),
        pipeline=pipeline,
        processing_lane={"work_item": f"{pipeline}-job", "current_step": current_step},
    )

    assert result["graph_id"] == f"workflow:{pipeline}:v2"
    assert result["execution_id"] == f"{pipeline}-job"
    assert result["workflow"]["target_node"] == expected
    assert (
        result["workflow"]["route_node_ids"][result["workflow"]["target_cursor"]]
        == expected
    )
    assert "processing" not in result
    assert "paths" not in result
    assert "lanes" not in result


@pytest.mark.parametrize(
    ("stage", "target", "route"),
    [
        ("triage", "triage", "success"),
        ("target-resolution", "target", "success"),
        ("generate", "generate", "success"),
        ("authorization", "authority", "success"),
        ("local-regenerate", "generate", "retry"),
        ("apply", "apply", "success"),
        ("semantic-publish", "publish", "success"),
        ("read-back", "readback", "success"),
        ("complete", "complete", "success"),
        ("semantic-noop", "complete", "noop"),
        ("failed", "hold", "hold"),
    ],
)
def test_ingest_runtime_stage_maps_to_one_workflow_arrival(
    stage: str,
    target: str,
    route: str,
) -> None:
    result = projection.project_decision_trace(
        _trace(),
        pipeline="ingest",
        processing_lane={"work_item": "child"},
        runtime_status={
            "current_job_id": "parent-job",
            "stage": stage,
            "state": "error" if stage == "failed" else "running",
        },
    )

    assert result["execution_id"] == "parent-job"
    assert result["workflow"]["target_node"] == target
    assert result["workflow"]["route"] == route
    assert result["workflow"]["target_state"] == (
        "error" if stage == "failed" else "done" if target == "complete" else "active"
    )


def test_running_ingest_is_authoritative_over_a_terminal_child_decision() -> None:
    result = projection.project_decision_trace(
        _trace(state="quarantined", active=False),
        pipeline="ingest",
        runtime_status={
            "current_job_id": "parent-job",
            "stage": "local-regenerate",
            "state": "running",
        },
    )

    assert result["trace_state"] == "active"
    assert result["workflow"]["route"] == "retry"
    assert result["workflow"]["target_node"] == "generate"
    assert result["workflow"]["target_state"] == "active"


@pytest.mark.parametrize(
    "pipeline", ["recall", "audit", "improve", "repair", "typed_graph"]
)
def test_terminal_non_ingest_decision_reaches_the_workflow_terminal(
    pipeline: str,
) -> None:
    result = projection.project_decision_trace(
        _trace(state="agreed", active=False, task_role=pipeline),
        pipeline=pipeline,
        processing_lane={"work_item": f"{pipeline}-job"},
    )

    assert result["workflow"]["target_node"] == "complete"
    assert result["workflow"]["target_state"] == "done"


def test_repair_escalation_uses_the_observed_lane_branch() -> None:
    result = projection.project_decision_trace(
        _trace(task_role="repair"),
        pipeline="repair",
        processing_lane={
            "work_item": "repair-job",
            "current_step": "escalate",
            "phase": "pending_frontier_review",
        },
    )

    assert result["workflow"]["route"] == "escalate"
    assert result["workflow"]["target_node"] == "escalate"


def test_single_authority_metadata_survives_without_becoming_a_second_graph() -> None:
    result = projection.project_decision_trace(
        _trace(
            authority_kind="single_model_v1",
            task_role="recall_authority",
            lanes=[
                {
                    "key": "primary",
                    "state": "active",
                    "phase": "validate",
                    "model": "Qwen",
                    "revision": "abc",
                }
            ],
        ),
        pipeline="recall",
        processing_lane={"current_step": "primary"},
    )

    assert result["single_model"] is True
    assert result["authority"] == {
        "kind": "single_model_v1",
        "label": "Single Authority",
        "model": "Qwen",
        "revision": "abc",
        "target": 1,
        "validated": False,
        "repair_is_vote": False,
    }
    assert set(result["workflow"]) == {
        "nodes",
        "edges",
        "route",
        "route_node_ids",
        "target_cursor",
        "target_node",
        "target_state",
    }


def test_renderer_owns_geometry_only_when_mounting_a_graph() -> None:
    renderer = (ROOT / "src/chronovisor/dashboard_static/app-renderer.js").read_text(
        encoding="utf-8"
    )
    update = renderer.split("function updateDecisionSvgHarness", 1)[1].split(
        "function decisionEventText", 1
    )[0]

    assert "updateIngestJobTrace" not in renderer
    assert "projection.processing" not in renderer
    assert "function advanceDecisionTraceOneStep" in renderer
    assert "DECISION_PROGRESS_INTERVAL_MS = 500" in renderer
    assert 'setAttribute("d"' not in renderer
    assert not re.search(
        r'setAttribute\(\s*["\'](?:d|transform|viewBox|height)["\']',
        update,
    )


def test_browser_stepper_advances_exactly_one_and_ignores_a_lower_target() -> None:
    renderer = (ROOT / "src/chronovisor/dashboard_static/app-renderer.js").read_text(
        encoding="utf-8"
    )
    helper = (
        "function synchronizeDecisionTraceTarget"
        + renderer.split("function synchronizeDecisionTraceTarget", 1)[1].split(
            "function renderDecisionTraceNow", 1
        )[0]
    )
    scenario = f"""
const vm = require("node:vm");
const sandbox = {{ fmt: (value, fallback) => value ?? fallback }};
vm.createContext(sandbox);
vm.runInContext({json.dumps(helper)}, sandbox);
const playback = {{ frame: null }};
const projection = {{ workflow: {{
  route: "success",
  route_node_ids: ["a", "b", "c", "d"],
  target_cursor: 3,
  target_state: "active",
}} }};
const cursors = [];
while (sandbox.advanceDecisionTraceOneStep(playback, projection)) {{
  cursors.push(playback.frame.cursor);
}}
projection.workflow.target_cursor = 1;
const changed = sandbox.advanceDecisionTraceOneStep(playback, projection);
process.stdout.write(JSON.stringify({{ cursors, changed, cursor: playback.frame.cursor }}));
"""
    completed = subprocess.run(
        ["node", "-e", scenario],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )

    assert json.loads(completed.stdout) == {
        "cursors": [0, 1, 2, 3],
        "changed": False,
        "cursor": 3,
    }


def test_browser_stepper_drains_one_milestone_every_half_second() -> None:
    renderer = (ROOT / "src/chronovisor/dashboard_static/app-renderer.js").read_text(
        encoding="utf-8"
    )
    pacing = (
        "const DECISION_PROGRESS_INTERVAL_MS"
        + renderer.split("const DECISION_PROGRESS_INTERVAL_MS", 1)[1].split(
            "window.__chronovisorDashboardTest", 1
        )[0]
    ).replace(
        """function renderDecisionTraceNow(trace) {
  renderDecisionTraceFrame(trace, null, decisionTracePlayback.frame);
  renderDecisionTransitionFeed(trace);
  setDecisionTransitionState(trace);
}""",
        """function renderDecisionTraceNow(_trace) {
  frames.push(decisionTracePlayback.frame.cursor);
}""",
    )
    scenario = f"""
const vm = require("node:vm");
const scheduled = [];
const frames = [];
const delays = [];
let clock = 0;
const sandbox = {{
  Date: {{ now: () => clock, parse: Date.parse }},
  document: {{ visibilityState: "visible" }},
  frames,
  window: {{
    matchMedia: () => ({{ matches: false }}),
    clearTimeout: () => {{}},
    setTimeout: (callback, delay) => {{
      scheduled.push({{ callback, delay }});
      return scheduled.length;
    }},
  }},
}};
vm.createContext(sandbox);
vm.runInContext(
  `function fmt(value, fallback) {{ return value ?? fallback; }}
   function decisionTraceProjection(trace) {{ return trace?.projection || null; }}
   {pacing}
   this.__test = {{ renderDecisionTrace }};`,
  sandbox,
);
const workflow = {{
  nodes: [], edges: [], route: "success",
  route_node_ids: ["a", "b", "c", "d"],
  target_cursor: 3, target_node: "d", target_state: "active",
}};
sandbox.__test.renderDecisionTrace({{ decision_trace: {{
  active: true,
  request_sha256: "execution",
  events: [],
  projection: {{
    schema: "chronovisor.decision-trace-projection.v3",
    execution_id: "execution",
    graph_id: "workflow:test:v1",
    revision: "3",
    workflow,
  }},
}} }});
while (scheduled.length) {{
  const task = scheduled.shift();
  delays.push(task.delay);
  clock += task.delay;
  task.callback();
}}
process.stdout.write(JSON.stringify({{ frames, delays }}));
"""
    completed = subprocess.run(
        ["node", "-e", scenario],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )

    assert json.loads(completed.stdout) == {
        "frames": [0, 1, 2, 3],
        "delays": [500, 500, 500],
    }


def test_browser_stepper_keeps_a_selected_retry_rail_after_backend_route_settles() -> (
    None
):
    renderer = (ROOT / "src/chronovisor/dashboard_static/app-renderer.js").read_text(
        encoding="utf-8"
    )
    helper = (
        "function synchronizeDecisionTraceTarget"
        + renderer.split("function synchronizeDecisionTraceTarget", 1)[1].split(
            "function renderDecisionTraceNow", 1
        )[0]
    )
    scenario = f"""
const vm = require("node:vm");
const sandbox = {{ fmt: (value, fallback) => value ?? fallback }};
vm.createContext(sandbox);
vm.runInContext({json.dumps(helper)}, sandbox);
const success = ["raw", "triage", "target", "generate", "authority", "result", "change", "apply", "publish", "readback", "complete"];
const retry = ["raw", "triage", "target", "generate", "authority", "result", "generate", "authority", "result", "change", "apply", "publish", "readback", "complete"];
const playback = {{ frame: null }};
const projection = {{ workflow: {{ route: "success", route_node_ids: success, target_cursor: 5, target_state: "active" }} }};
while (sandbox.advanceDecisionTraceOneStep(playback, projection)) {{}}
projection.workflow = {{ route: "retry", route_node_ids: retry, target_cursor: 6, target_state: "active" }};
sandbox.advanceDecisionTraceOneStep(playback, projection);
projection.workflow = {{ route: "success", route_node_ids: success, target_cursor: 7, target_state: "active" }};
const observed = [];
while (sandbox.advanceDecisionTraceOneStep(playback, projection)) {{
  observed.push(playback.frame.route_node_ids[playback.frame.cursor]);
}}
process.stdout.write(JSON.stringify({{ route: playback.frame.route, observed }}));
"""
    completed = subprocess.run(
        ["node", "-e", scenario],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )

    assert json.loads(completed.stdout) == {
        "route": "retry",
        "observed": ["authority", "result", "change", "apply"],
    }


def test_browser_stepper_selects_noop_before_the_branch_is_crossed() -> None:
    renderer = (ROOT / "src/chronovisor/dashboard_static/app-renderer.js").read_text(
        encoding="utf-8"
    )
    helper = (
        "function synchronizeDecisionTraceTarget"
        + renderer.split("function synchronizeDecisionTraceTarget", 1)[1].split(
            "function renderDecisionTraceNow", 1
        )[0]
    )
    scenario = f"""
const vm = require("node:vm");
const sandbox = {{ fmt: (value, fallback) => value ?? fallback }};
vm.createContext(sandbox);
vm.runInContext({json.dumps(helper)}, sandbox);
const success = ["raw", "triage", "target", "generate", "authority", "result", "change", "apply", "publish", "readback", "complete"];
const noop = ["raw", "triage", "target", "generate", "authority", "result", "change", "readback", "complete"];
const playback = {{ frame: null }};
const projection = {{ workflow: {{ route: "success", route_node_ids: success, target_cursor: 4, target_state: "active" }} }};
while (sandbox.advanceDecisionTraceOneStep(playback, projection)) {{}}
projection.workflow = {{ route: "noop", route_node_ids: noop, target_cursor: 8, target_state: "done" }};
const observed = [];
while (sandbox.advanceDecisionTraceOneStep(playback, projection)) {{
  observed.push(playback.frame.route_node_ids[playback.frame.cursor]);
}}
process.stdout.write(JSON.stringify({{ route: playback.frame.route, observed }}));
"""
    completed = subprocess.run(
        ["node", "-e", scenario],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )

    assert json.loads(completed.stdout) == {
        "route": "noop",
        "observed": ["result", "change", "readback", "complete"],
    }


def test_markup_restores_the_fixed_execution_plan_before_the_dynamic_workflow() -> None:
    page = (ROOT / "src/chronovisor/dashboard_static/index.html").read_text(
        encoding="utf-8"
    )

    assert 'id="decision-trace-harness" viewBox="0 0 1500 840"' in page
    assert page.count("data-workflow-edges") == 1
    assert page.count("data-workflow-nodes") == 1
    assert page.count("data-context-option") == 4
    assert page.count("data-reasoning-key") == 4
    assert "CONTEXT WINDOW" in page
    assert "REASONING BUDGET" in page
    assert "data-ingest-job-step" not in page
    assert "data-decision-lane-step" not in page
