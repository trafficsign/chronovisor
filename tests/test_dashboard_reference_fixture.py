from __future__ import annotations

import json
import os
import shutil
import signal
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
    assert 'class="trace-diamond trace-node-plan" data-workflow-node="fit"' not in page
    assert 'class="trace-node trace-node-plan" data-workflow-node="fit"' in page
    assert 'class="trace-branch-label trace-fit-outcome"' in page
    assert 'class="trace-context-guide" d="M458 132 H536' in page
    for guide_class in ("trace-context-guide", "trace-reasoning-guide"):
        guide_path = page.split(f'class="{guide_class}" d="', 1)[1].split('"', 1)[0]
        assert guide_path.count("Q") == 4


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


def _capture_stepper_visuals(
    scenarios: list[dict[str, object]], visual_dir: Path, tmp_path: Path
) -> None:
    server = ThreadingHTTPServer(("127.0.0.1", 0), stepper_handler(scenarios))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        visual_dir.mkdir(parents=True, exist_ok=True)
        image_index = 0
        for scenario in scenarios:
            for frame in scenario["frames"]:
                screenshot = visual_dir / (
                    f"{scenario['id']}-{frame['cursor']:02d}-{frame['milestone']}.png"
                )
                if (
                    screenshot.is_file()
                    and screenshot.stat().st_size > 0
                    and screenshot.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"
                ):
                    image_index += 1
                    continue
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
                        "--virtual-time-budget=1000",
                        "--hide-scrollbars",
                        "--window-size=2048,1800",
                        f"--screenshot={screenshot}",
                        f"--user-data-dir={tmp_path / f'trace-visual-profile-{image_index}'}",
                        (
                            f"http://127.0.0.1:{server.server_port}/"
                            f"?scenario={scenario['id']}&step={frame['cursor']}"
                        ),
                    ],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
                deadline = time.monotonic() + 15
                while not screenshot.is_file() and time.monotonic() < deadline:
                    if process.poll() is not None:
                        break
                    time.sleep(0.05)
                if process.poll() is None:
                    process.terminate()
                try:
                    process.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=10)
                assert screenshot.is_file() and screenshot.stat().st_size > 0
                assert process.returncode in {0, -signal.SIGTERM, -signal.SIGKILL}
                assert screenshot.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"
                image_index += 1
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=3)


def test_all_decision_inputs_keep_real_dashboard_paths_connected(
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
    plan_guide_styles: dict[str, dict[str, object]] = {}
    skipped_elements_seen = 0
    layout_checked: set[str] = set()
    overlaps: list[tuple[str, object]] = []
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
        path_states = {
            (path["source"], path["target"]): path["state"] for path in paths
        }
        nodes = {node["id"]: node["state"] for node in result["nodes"]}
        if scenario["pipeline"] not in layout_checked:
            workflow = TRACE_WORKFLOWS[scenario["pipeline"]]
            main = [
                node["id"]
                for node in workflow["nodes"]
                if node["type"] != "plan" and node["id"] not in {"hold", "escalate"}
            ]
            positions = {}
            for node in result["nodes"]:
                if node["transform"]:
                    raw = (
                        str(node["transform"])
                        .removeprefix("translate(")
                        .removesuffix(")")
                    )
                    positions[node["id"]] = tuple(
                        map(float, raw.replace(",", " ").split())
                    )
            result_index = main.index("result")
            first_row = main[: result_index + 1]
            second_row = main[result_index + 1 :]
            assert positions.get(first_row[0]) == (1260.0, 416.0), (
                scenario["id"],
                first_row[0],
                positions,
            )
            dispatch = next(path for path in paths if path["source"] == "fit")
            assert dispatch["d"] == (
                "M1338 132 H1358 Q1368 132 1368 142 "
                "V406 Q1368 416 1358 416 H1270"
            )
            first_xs = [positions[node][0] for node in first_row]
            assert {positions[node][1] for node in first_row} == {416.0}
            assert first_xs == sorted(first_xs, reverse=True)
            assert all(
                0 < left - right <= 230
                for left, right in zip(first_xs, first_xs[1:], strict=False)
            )
            if second_row:
                second_xs = [positions[node][0] for node in second_row]
                assert {positions[node][1] for node in second_row} == {576.0}
                assert second_xs == sorted(second_xs)
                assert all(
                    0 < right - left <= 230
                    for left, right in zip(second_xs, second_xs[1:], strict=False)
                )
                first_after_result = next(
                    node for node in workflow["nodes"] if node["id"] == second_row[0]
                )
                expected_offset = 0 if first_after_result["type"] == "decision" else 80
                assert positions[second_row[0]][0] - positions["result"][0] == expected_offset
            layout_checked.add(scenario["pipeline"])
        assert all(path["d"] for path in paths)
        guides = result["guides"]
        assert all(guide["d"].count("Q") == 4 for guide in guides)
        for guide in guides:
            class_name = str(guide["className"])
            style = {
                key: guide[key]
                for key in (
                    "display",
                    "opacity",
                    "stroke",
                    "strokeDasharray",
                    "visibility",
                )
            }
            assert style == {
                "display": "inline",
                "opacity": "0.5",
                "stroke": "rgb(125, 137, 146)",
                "strokeDasharray": "4px, 5px",
                "visibility": "visible",
            }, (scenario["id"], frame["cursor"], class_name, style)
            assert style == plan_guide_styles.setdefault(class_name, style), (
                scenario["id"],
                frame["cursor"],
                class_name,
                style,
            )
        skipped_elements_seen += len(result["skippedElements"])
        assert all(
            element["opacity"] == "0.24"
            for element in result["skippedElements"]
        ), (scenario["id"], frame["cursor"], result["skippedElements"])
        assert result["sharpCorners"] == [], (
            scenario["id"],
            frame["cursor"],
            result["sharpCorners"],
        )
        bad_corner_radii = [
            radius
            for radius in result["cornerRadii"]
            if any(abs(radius[side] - 10) >= 0.01 for side in ("incoming", "outgoing"))
        ]
        assert bad_corner_radii == [], (
            scenario["id"],
            frame["cursor"],
            bad_corner_radii,
        )
        assert result["milestoneTurns"] == [], (
            scenario["id"],
            frame["cursor"],
            result["milestoneTurns"],
        )
        assert result["pathLabelCollisions"] == [], (
            scenario["id"],
            frame["cursor"],
            result["pathLabelCollisions"],
        )
        assert result["pathIntersections"] == []
        assert result["guideOverlaps"] == []
        assert result["textOverlaps"] == [], (
            scenario["id"],
            frame["cursor"],
            result["textOverlaps"],
        )
        assert all(
            item["clearance"] > 1.5
            for item in result["reasoningLabelClearances"]
        ), (
            scenario["id"],
            frame["cursor"],
            result["reasoningLabelClearances"],
        )
        assert result["decisionCenterEndpoints"] == []
        assert all(question["inside"] for question in result["decisionQuestions"]), (
            scenario["id"],
            frame["cursor"],
            result["decisionQuestions"],
        )
        assert all(
            question["fill"] == "rgb(6, 23, 36)"
            for question in result["decisionQuestions"]
        ), (
            scenario["id"],
            frame["cursor"],
            result["decisionQuestions"],
        )
        assert all(
            branch["labelState"] == branch["state"]
            and branch["distance"] <= 48
            for branch in result["decisionBranchLabels"]
        ), (
            scenario["id"],
            frame["cursor"],
            result["decisionBranchLabels"],
        )
        worst_endpoint = max(
            result["endpointErrors"],
            key=lambda error: error["distance"],
        )
        assert worst_endpoint["distance"] < 0.75, (
            scenario["id"],
            frame["cursor"],
            worst_endpoint,
        )
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
        if frame["cursor"]:
            incoming = (route[frame["cursor"] - 1], route[frame["cursor"]])
            assert path_states[incoming] == (
                "active" if expected_state == "active" else "done"
            ), (scenario["id"], frame["cursor"], incoming, path_states[incoming])
        assert all(
            nodes[node] == "done"
            for node in set(route[: frame["cursor"]])
            if node != frame["milestone"]
        )
        if result["pathOverlaps"]:
            overlaps.append((scenario["id"], result["pathOverlaps"]))

    assert overlaps == []
    assert skipped_elements_seen > 0
    if visual_dir := os.environ.get("CHRONOVISOR_DASHBOARD_VISUAL_DIR"):
        _capture_stepper_visuals(scenarios, Path(visual_dir), tmp_path)


def test_stepper_scenarios_are_json_serializable() -> None:
    payload = json.dumps(decision_trace_step_scenarios(), sort_keys=True)

    assert '"pipeline": "ingest"' in payload
    assert '"pipeline": "typed_graph"' in payload
