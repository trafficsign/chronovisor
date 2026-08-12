from __future__ import annotations

import hashlib
import json
import struct
from html.parser import HTMLParser
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "tests/fixtures/dashboard_decision_trace_states.json"


class _DashboardMarkup(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.elements: list[tuple[str, dict[str, str | None]]] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        self.elements.append((tag, dict(attrs)))


def test_dashboard_reference_contract_and_state_matrix() -> None:
    contract = json.loads(FIXTURE.read_text(encoding="utf-8"))
    reference = contract["reference"]
    image = ROOT / reference["path"]
    raw = image.read_bytes()

    assert hashlib.sha256(raw).hexdigest() == reference["sha256"]
    assert raw[:8] == b"\x89PNG\r\n\x1a\n"
    assert struct.unpack(">II", raw[16:24]) == (
        reference["width"],
        reference["height"],
    )
    assert reference["viewport"] == {"width": 1586, "height": 992}
    assert reference["capture_selector"] == "#processing-panel"

    cases = {row["id"]: row for row in contract["cases"]}
    assert list(cases) == list("ABCDEFGH")
    assert contract["topology"]["artifact_nodes"] == 1
    assert contract["topology"]["decision_nodes"] == 1
    assert contract["topology"]["dynamic_paths"] == ["processing-entry"]

    assert cases["A"]["active_lane"] == "primary"
    assert cases["B"]["active_lane"] == "challenger"
    assert cases["C"]["branch"] == "pair-agreement"
    assert cases["C"]["lane_states"][2] == "skipped"
    assert cases["D"]["active_lane"] == "tie_break"
    assert cases["E"]["branch"] == "tie-break-agreement"
    assert cases["F"]["hold"] == "before-artifact"
    assert cases["F"]["artifact"] == "skipped"
    assert cases["G"]["hold"] == "seal-failed"
    assert cases["G"]["artifact"] == "error"
    assert cases["G"]["decision"] == "skipped"
    assert cases["H"]["state"] == "idle"
    assert cases["H"]["active_lane"] is None

    for row in cases.values():
        assert len(row["lane_states"]) == 3
        assert len(row["lane_think"]) == 3
        assert len(row["lane_context_tokens"]) == 3


def test_dashboard_reference_svg_has_one_fixed_safe_topology() -> None:
    parser = _DashboardMarkup()
    parser.feed(
        (ROOT / "src/chronovisor/dashboard_static/index.html").read_text(
            encoding="utf-8"
        )
    )
    elements = parser.elements

    def matching(key: str, value: str) -> list[dict[str, str | None]]:
        return [attrs for _tag, attrs in elements if attrs.get(key) == value]

    assert len(matching("id", "processing-panel")) == 1
    assert len(matching("class", "processing-lanes-card")) == 1
    assert len(matching("id", "decision-trace-panel")) == 1
    assert len(matching("id", "processing-trace-connector-path")) == 1
    assert len(matching("id", "decision-trace-harness")) == 1
    assert matching("id", "decision-trace-harness")[0]["viewbox"] == ("0 0 1500 650")
    assert matching("id", "decision-trace-harness")[0]["height"] == "607"

    assert len(matching("data-trace-key", "artifact")) == 1
    assert len(matching("data-trace-key", "decision")) == 1
    assert len(matching("data-trace-key", "hold")) == 1
    assert [
        attrs.get("data-overall-key")
        for _tag, attrs in elements
        if attrs.get("data-overall-key")
    ] == [
        "packet",
        "preflight",
        "execution_plan",
    ]
    assert len(matching("data-context-tokens", "32768")) == 1
    assert len(matching("data-context-tokens", "65536")) == 1
    assert len(matching("data-context-tokens", "98304")) == 1
    assert len(matching("data-context-tokens", "131072")) == 1
    assert {
        attrs["data-decision-lane"]: attrs.get("transform")
        for _tag, attrs in elements
        if attrs.get("data-decision-lane")
    } == {
        "primary": "translate(0 325)",
        "challenger": "translate(0 450)",
        "tie_break": "translate(0 575)",
    }
    assert len(matching("data-seal-label", "true")) == 1

    paths = {
        attrs["data-path-key"]: attrs.get("d")
        for tag, attrs in elements
        if tag == "path" and attrs.get("data-path-key")
    }
    assert {
        "single-artifact",
        "pair-artifact",
        "pair-tie_break",
        "tie_break-artifact",
        "pair-hold",
        "tie_break-hold",
        "artifact-decision",
        "artifact-hold",
    }.issubset(paths)
    assert all(paths[key] for key in paths)
    assert paths["pair-artifact"] != paths["tie_break-artifact"]
    assert paths["artifact-hold"] == "M1460 415 V492"
    assert paths["pair-hold"] == "M1208 490 V494 Q1208 504 1218 504 H1448"
    assert paths["tie_break-hold"] == (
        "M1106 575 H1408 Q1418 575 1418 565 "
        "V514 Q1418 504 1428 504 H1448"
    )
    assert any(
        tag == "path" and attrs.get("d") == "M1208 410 1248 450 1208 490 1168 450Z"
        for tag, attrs in elements
    )

    expected_boundary_paths = {
        "packet-preflight": "M105 10 H289",
        "preflight-execution_plan": "M307 10 H501",
        "plan-dispatch": (
            "M1380 164 H1466 Q1476 164 1476 174 V265 Q1476 275 1466 275 "
            "H190 Q180 275 180 285 V315 Q180 325 190 325 H220"
        ),
        "primary-challenger": (
            "M1096 335 V377 Q1096 387 1086 387 H190 Q180 387 180 397 "
            "V440 Q180 450 190 450 H220"
        ),
        "single-artifact": "M1106 325 H1332 Q1342 325 1342 335 V393",
        "challenger-agree": "M1106 450 H1168",
        "pair-artifact": (
            "M1248 450 H1266 Q1276 450 1276 440 V414 Q1276 404 1286 404 H1331"
        ),
        "pair-tie_break": (
            "M1208 490 V520 Q1208 530 1198 530 "
            "H190 Q180 530 180 540 V565 Q180 575 190 575 H220"
        ),
        "tie_break-artifact": "M1106 575 H1332 Q1342 575 1342 565 V415",
        "artifact-decision": "M1353 404 H1449",
    }
    assert {key: paths[key] for key in expected_boundary_paths} == (
        expected_boundary_paths
    )

    repair_paths = [
        attrs
        for tag, attrs in elements
        if tag == "path" and attrs.get("data-repair-lane")
    ]
    assert len(repair_paths) == 3
    assert all(
        attrs.get("d") == "M743 12 V30 Q743 40 753 40 H900 Q910 40 910 30 V12"
        and attrs.get("marker-start") == "url(#trace-arrow)"
        and attrs.get("marker-end") == "url(#trace-arrow)"
        for attrs in repair_paths
    )

    reasoning_guides = [
        attrs
        for tag, attrs in elements
        if tag == "path" and attrs.get("class") == "trace-reasoning-guide"
    ]
    assert len(reasoning_guides) == 3
    assert [attrs.get("d") for attrs in reasoning_guides] == [
        "M858 164 H882 Q892 164 892 154 V116 Q892 106 902 106 "
        "H1245 Q1255 106 1255 116 V154 Q1255 164 1265 164 H1270",
        "M858 164 H1270",
        "M858 164 H882 Q892 164 892 174 V212 Q892 222 902 222 "
        "H1245 Q1255 222 1255 212 V174 Q1255 164 1265 164 H1270",
    ]
    assert [
        (attrs.get("data-reasoning-output"), attrs.get("marker-end"))
        for tag, attrs in elements
        if tag == "path" and attrs.get("data-reasoning-output")
    ] == [("low", None), ("medium", None), ("high", None)]
    assert all(
        attrs.get("marker-end") == "url(#trace-arrow)"
        for tag, attrs in elements
        if tag == "path"
        and attrs.get("data-path-key")
        in {"reasoning-low", "reasoning-medium", "reasoning-high", "plan-fit"}
    )
    assert all(
        attrs.get("y") == "28"
        for tag, attrs in elements
        if tag == "text" and attrs.get("data-lane-result")
    )
    assert len(matching("class", "trace-label-backdrop")) == 1
    assert len(matching("class", "trace-selected-core")) == 3
    assert {
        attrs["data-reasoning-key"]: attrs.get("transform")
        for _tag, attrs in elements
        if attrs.get("data-reasoning-key")
    } == {
        "low": "translate(912 106)",
        "medium": "translate(912 164)",
        "high": "translate(912 222)",
    }
    assert all(
        attrs.get("y") == "34"
        for tag, attrs in elements
        if tag == "text" and attrs.get("class") == "trace-reasoning-detail"
    )
    assert matching("data-plan-value", "fit")[0]["y"] == "181"
    assert matching("data-plan-fit-pass", "true")[0]["y"] == "181"
    assert matching("class", "trace-branch-node-label")[0]["x"] == "16"
    assert matching("class", "trace-artifact-label")[0]["x"] == "16"
    assert matching("class", "trace-artifact-label")[0]["text-anchor"] == "start"
    assert [
        attrs.get("y")
        for _tag, attrs in elements
        if attrs.get("class") == "trace-path-label"
    ] == ["271", "381"]
    assert matching("class", "trace-branch-label trace-yes-label")[0]["y"] == "431"
    assert matching("class", "trace-branch-label trace-no-label")[0]["y"] == "514"
    assert matching("data-seal-label", "true")[0]["x"] == "1448"
    assert paths["plan-context"] == (
        "M470 174 V178 Q470 182 474 182 H490 Q498 182 498 190 "
        "V226 Q498 236 508 236 H724 Q734 236 734 226 V174"
    )
    context_guides = {
        attrs["data-context-guide"]: attrs
        for tag, attrs in elements
        if tag == "path" and attrs.get("data-context-guide")
    }
    assert {key: attrs.get("d") for key, attrs in context_guides.items()} == {
        "32768": (
            "M510 19 V76 Q510 86 500 86 H352 Q342 86 342 96 V155 "
            "M342 174 V178 Q342 182 346 182 H362 Q370 182 370 190 "
            "V226 Q370 236 380 236 H724 Q734 236 734 226 V174 "
            "Q734 164 744 164 H766"
        ),
        "65536": (
            "M510 19 V76 Q510 86 500 86 H480 Q470 86 470 96 V155 "
            "M470 174 V178 Q470 182 474 182 H490 Q498 182 498 190 "
            "V226 Q498 236 508 236 H724 Q734 236 734 226 V174 "
            "Q734 164 744 164 H766"
        ),
        "98304": (
            "M510 19 V76 Q510 86 520 86 H545 Q555 86 555 96 V155 "
            "M555 174 V178 Q555 182 559 182 H575 Q583 182 583 190 "
            "V226 Q583 236 593 236 H724 Q734 236 734 226 V174 "
            "Q734 164 744 164 H766"
        ),
        "131072": (
            "M510 19 V76 Q510 86 520 86 H632 Q642 86 642 96 V155 "
            "M642 174 V178 Q642 182 646 182 H662 Q670 182 670 190 "
            "V226 Q670 236 680 236 H724 Q734 236 734 226 V174 "
            "Q734 164 744 164 H766"
        ),
    }
    assert all(attrs.get("marker-end") is None for attrs in context_guides.values())
    assert not matching("class", "trace-context-rail")
    assert len(matching("class", "trace-label-backdrop trace-context-kicker-backdrop")) == 1


def test_dashboard_reference_keeps_selection_and_bucket_truth() -> None:
    static = ROOT / "src/chronovisor/dashboard_static"
    page = (static / "index.html").read_text(encoding="utf-8")
    style = (static / "style.css").read_text(encoding="utf-8")
    renderer = (static / "app-renderer.js").read_text(encoding="utf-8")

    assert [
        token
        for token in ("32768", "65536", "98304", "131072")
        if f'data-context-tokens="{token}"' in page
    ] == ["32768", "65536", "98304", "131072"]
    assert '<p id="work-summary">Waiting · 0 / 3</p>' in page
    assert (
        page.index('data-context-guide="131072"')
        < page.index('data-path-key="execution-plan-context"')
        < page.index("trace-context-kicker-backdrop")
        < page.index(">CONTEXT WINDOW<")
    )
    assert (
        page.index('data-context-guide="131072"')
        < page.index('data-path-key="plan-context"')
        < page.index('data-plan-value="context-selection"')
    )
    assert (
        '.processing-lane[aria-expanded="false"] .processing-track {\n'
        "  visibility: hidden;"
    ) in style
    assert (
        ".decision-trace-harness .trace-repair-loop.pending {\n"
        "  stroke-dasharray: 4 4;\n"
        "  opacity: 0.18;"
    ) in style
    assert style.count("height: 1030px;") == 3
    assert style.count("height: 802px;") == 3
    assert style.count("height: 607px;") == 2
    assert (
        ".decision-trace-harness .done.decision-lane-step circle,\n"
        ".decision-trace-harness .decision-lane-step.done circle {\n"
        "  fill: url(#trace-active-core);\n"
        "  stroke: #9296ff;"
    ) in style
    assert ".trace-shared-node.done circle" not in style
    assert (
        ".decision-trace-harness .trace-single-path.pending,\n"
        ".decision-trace-harness .trace-hold-path.pending,\n"
        ".decision-trace-harness .trace-hold-node.pending {\n"
        "  opacity: 0.18;"
    ) in style
    assert (
        ".decision-trace-harness .trace-context-guide,\n"
        ".decision-trace-harness .trace-reasoning-guide {\n"
        "  fill: none;\n"
        "  stroke: #566470;"
    ) in style
    assert (
        '.decision-trace-harness .trace-diamond [data-plan-value="fit"] {'
        "\n  text-anchor: end;"
    ) in style
    active_path_style = style.split(
        ".decision-trace-harness .active.trace-path,", 1
    )[1].split("}", 1)[0]
    assert "filter: drop-shadow(" in active_path_style
    assert "filter: url(" not in active_path_style
    assert "stroke-dasharray: none;" in active_path_style
    for state in ("done", "error"):
        reached_path_style = style.split(
            f".decision-trace-harness .{state}.trace-path,", 1
        )[1].split("}", 1)[0]
        assert "stroke-dasharray: none;" in reached_path_style
    assert (
        ".decision-trace-harness .trace-path {\n"
        "  stroke-dasharray: 4 5;\n"
        "}"
    ) in style
    assert ".decision-trace-harness .trace-tie-path {" not in style
    frame = renderer.split("function renderDecisionTraceFrame", 1)[1].split(
        "function setDecisionTransitionState", 1
    )[0]
    assert "updateProcessingTraceSelection(trace);" in frame
    harness = renderer.split("function updateDecisionSvgHarness", 1)[1].split(
        "function decisionEventText", 1
    )[0]
    assert 'decision: decisionOverallState(overall, "decision")' in harness
    assert 'decision: noSafeQuorum ? "skipped"' not in harness
    assert 'harness.querySelector("[data-seal-label]")' in harness
    assert 'sealFailure ? "error" : "pending"' in harness
    assert 'const artifactReached = traceState === "agreed" || sealFailure;' in harness
    assert '["artifact-decision", traceState === "agreed" ? "done" : "pending"]' in harness
    assert '["challenger-agree", completedStepState("challenger", "vote")]' in harness
    assert 'const fitState = reasoningSelected ? planState : "pending";' in harness
    assert 'querySelector("[data-trace-entry] circle")' in renderer
    assert 'getComputedStyle(source, "::before")' in renderer
    assert "const y2 = targetBox.top - panelBox.top;" in renderer
    assert "Math.floor(tokens / 1000)" in renderer
    assert 'data-reasoning-output="${mode}"' in harness
