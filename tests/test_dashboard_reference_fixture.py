from __future__ import annotations

import json
import shutil
import subprocess
import threading
import time
from http.server import ThreadingHTTPServer
from pathlib import Path

from chronovisor.ops import dashboard
from chronovisor.ops.decision_trace_projection import TRACE_WORKFLOWS
from tests.decision_trace_stepper import (
    decision_trace_step_scenarios,
    stepper_handler,
)

ROOT = Path(__file__).resolve().parents[1]


def _chrome() -> str:
    candidate = next(
        (
            value
            for value in (
                shutil.which("google-chrome"),
                shutil.which("chromium"),
                shutil.which("chromium-browser"),
                "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
            )
            if value and Path(value).is_file()
        ),
        None,
    )
    assert candidate is not None, "headless Chrome is required for Dashboard DOM tests"
    return candidate


def test_stepper_inventory_covers_all_six_workflows_and_every_route_step() -> None:
    scenarios = decision_trace_step_scenarios()
    expected_ids = {
        f"{pipeline}-{route}"
        for pipeline, workflow in TRACE_WORKFLOWS.items()
        for route in workflow["routes"]
    }

    assert {scenario["id"] for scenario in scenarios} == expected_ids
    assert {scenario["pipeline"] for scenario in scenarios} == set(TRACE_WORKFLOWS)
    for scenario in scenarios:
        route = TRACE_WORKFLOWS[scenario["pipeline"]]["routes"][scenario["route"]]
        assert [frame["cursor"] for frame in scenario["frames"]] == list(
            range(len(route))
        )
        assert [frame["milestone"] for frame in scenario["frames"]] == list(route)


def test_dashboard_keeps_one_fixed_plan_and_one_dynamic_pipeline_mount() -> None:
    page = (dashboard.STATIC_DIR / "index.html").read_text(encoding="utf-8")
    renderer = (dashboard.STATIC_DIR / "app-renderer.js").read_text(encoding="utf-8")

    assert page.count("data-workflow-edges") == 1
    assert page.count("data-workflow-nodes") == 1
    assert page.count("data-context-option") == 4
    assert page.count("data-reasoning-key") == 4
    assert "data-ingest-job-step" not in page
    assert "data-decision-lane-step" not in page
    assert "function mountDecisionWorkflow" in renderer
    assert "function workflowFrameStates" in renderer
    assert "function advanceDecisionTraceOneStep" in renderer
    assert "updateIngestJobTrace" not in renderer
    assert "project_processing_trace" not in renderer


def test_dashboard_css_has_one_state_system_for_nodes_and_edges() -> None:
    style = (dashboard.STATIC_DIR / "style.css").read_text(encoding="utf-8")

    for selector in (
        ".decision-trace-harness .trace-path.done",
        ".decision-trace-harness .trace-path.active",
        ".decision-trace-harness .trace-path.error",
        ".decision-trace-harness .trace-path.skipped",
        ".decision-trace-harness .trace-node.done > circle",
        ".decision-trace-harness .trace-node.active > circle",
        ".decision-trace-harness .trace-node.error > circle",
    ):
        assert selector in style
    assert ".trace-ingest-job" not in style
    assert ".decision-lane-step" not in style
    assert ".trace-repair-loop" not in style


def test_all_workflow_frames_keep_geometry_fixed_and_advance_one_node(
    tmp_path: Path,
) -> None:
    scenarios = decision_trace_step_scenarios()
    browser_results: list[dict[str, object]] = []
    server = ThreadingHTTPServer(
        ("127.0.0.1", 0),
        stepper_handler(scenarios, browser_results),
    )
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    process = None
    stderr = ""
    try:
        process = subprocess.Popen(
            [
                _chrome(),
                "--headless=new",
                "--no-sandbox",
                "--disable-dev-shm-usage",
                "--disable-gpu",
                "--disable-background-networking",
                "--disable-component-update",
                "--no-first-run",
                "--force-prefers-reduced-motion=reduce",
                "--run-all-compositor-stages-before-draw",
                "--virtual-time-budget=15000",
                "--window-size=1586,1200",
                f"--user-data-dir={tmp_path / 'trace-profile'}",
                f"http://127.0.0.1:{server.server_port}/?audit=1",
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            text=True,
        )
        expected = sum(len(scenario["frames"]) for scenario in scenarios)
        deadline = time.monotonic() + 35
        while len(browser_results) < expected and time.monotonic() < deadline:
            if process.poll() is not None:
                break
            time.sleep(0.05)
        if process.poll() is None:
            process.terminate()
        try:
            _, stderr = process.communicate(timeout=3)
        except subprocess.TimeoutExpired:
            process.kill()
            _, stderr = process.communicate(timeout=3)
    finally:
        if process is not None and process.poll() is None:
            process.kill()
            process.wait(timeout=3)
        server.shutdown()
        server.server_close()
        thread.join(timeout=3)

    expected_frames = [
        (scenario, frame) for scenario in scenarios for frame in scenario["frames"]
    ]
    assert len(browser_results) == len(expected_frames), stderr[-4000:]

    geometry_by_pipeline: dict[str, str] = {}
    for result, (scenario, frame) in zip(
        browser_results,
        expected_frames,
        strict=True,
    ):
        assert result["scenario"] == scenario["id"]
        assert result["cursor"] == frame["cursor"]
        assert result["visibleCursor"] == frame["cursor"]
        assert result["milestone"] == frame["milestone"]
        assert result["graphId"] == f"workflow:{scenario['pipeline']}:v2"

        geometry = str(result["geometry"])
        previous_geometry = geometry_by_pipeline.setdefault(
            scenario["pipeline"],
            geometry,
        )
        assert geometry == previous_geometry

        paths = result["paths"]
        nodes = {node["id"]: node["state"] for node in result["nodes"]}
        assert all(path["d"] for path in paths)
        assert result["pathLabelCollisions"] == []
        assert result["pathIntersections"] == []
        assert result["textOverlaps"] == []
        assert max(error["distance"] for error in result["endpointErrors"]) < 0.75
        final = frame["cursor"] == len(scenario["frames"]) - 1
        expected_state = (
            "error"
            if final and frame["milestone"] == "hold"
            else "done"
            if final
            else "active"
        )
        assert nodes[frame["milestone"]] == expected_state
        route = TRACE_WORKFLOWS[scenario["pipeline"]]["routes"][scenario["route"]]
        assert all(
            nodes[node] == "done"
            for node in set(route[: frame["cursor"]])
            if node != frame["milestone"]
        )


def test_stepper_scenarios_are_json_serializable() -> None:
    payload = json.dumps(decision_trace_step_scenarios(), sort_keys=True)

    assert '"pipeline": "ingest"' in payload
    assert '"pipeline": "typed_graph"' in payload
