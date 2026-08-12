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

    def handle_starttag(
        self, tag: str, attrs: list[tuple[str, str | None]]
    ) -> None:
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
    assert matching("id", "decision-trace-harness")[0]["viewbox"] == (
        "0 0 1500 590"
    )

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
        "dispatch",
        "local_decision",
        "artifact",
        "decision",
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
        "primary": "translate(0 329)",
        "challenger": "translate(0 436)",
        "tie_break": "translate(0 556)",
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
    assert paths["artifact-hold"].startswith("M1342 416")
    assert paths["pair-hold"].startswith("M1208 476")
    assert paths["tie_break-hold"].startswith("M1096 556")


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
    assert (
        '.processing-lane[aria-expanded="false"] .processing-track {\n'
        "  visibility: hidden;"
    ) in style
    assert (
        ".decision-trace-harness .trace-repair-loop.pending {\n"
        "  stroke-dasharray: 4 4;\n"
        "  opacity: 0.18;"
    ) in style
    assert (
        ".decision-trace-harness .trace-single-path.pending,\n"
        ".decision-trace-harness .trace-hold-path.pending,\n"
        ".decision-trace-harness .trace-hold-node.pending {\n"
        "  opacity: 0.18;"
    ) in style
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
