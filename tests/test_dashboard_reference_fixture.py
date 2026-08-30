from __future__ import annotations

import hashlib
import json
import os
import shutil
import signal
import struct
import subprocess
import threading
import time
from html.parser import HTMLParser
from http.server import ThreadingHTTPServer
from pathlib import Path
from urllib.parse import quote

import pytest

from chronovisor.ops import dashboard
from tests.decision_trace_stepper import (
    decision_trace_step_scenarios,
    stepper_handler,
)

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
    assert contract["topology"]["dynamic_paths"] == []
    assert contract["topology"]["artifact_join_state_order"] == [
        "single",
        "pair_yes",
        "tie_input",
        "gate",
        "gate_yes",
        "no",
    ]
    assert contract["topology"]["fixed_paths"][-9:] == [
        "single-artifact",
        "pair-tie-break",
        "pair-artifact-join",
        "tie_break-quorum",
        "quorum-artifact-join",
        "artifact-seal",
        "seal-decision",
        "quorum-hold",
        "seal-hold",
    ]

    assert cases["A"]["active_lane"] == "primary"
    assert cases["B"]["active_lane"] == "challenger"
    assert cases["C"]["branch"] == "pair-agreement"
    assert cases["C"]["lane_states"][2] == "skipped"
    assert cases["D"]["active_lane"] == "tie_break"
    assert cases["E"]["branch"] == "tie-break-agreement"
    assert cases["F"]["hold"] == "before-artifact"
    assert cases["F"]["artifact"] == "skipped"
    assert cases["F"]["decision"] == "skipped"
    assert cases["G"]["hold"] == "seal-failed"
    assert cases["G"]["artifact"] == "error"
    assert cases["G"]["decision"] == "skipped"
    assert cases["H"]["state"] == "idle"
    assert cases["H"]["active_lane"] is None
    assert [cases[key]["artifact_join_states"] for key in cases] == [
        ["pending", "pending", "pending", "pending", "pending", "pending"],
        ["pending", "pending", "pending", "pending", "pending", "pending"],
        ["pending", "done", "pending", "pending", "pending", "pending"],
        ["pending", "pending", "pending", "pending", "pending", "pending"],
        ["pending", "pending", "done", "done", "done", "pending"],
        ["pending", "pending", "error", "error", "pending", "error"],
        ["pending", "done", "pending", "pending", "pending", "pending"],
        ["pending", "pending", "pending", "pending", "pending", "pending"],
    ]

    for row in cases.values():
        assert len(row["lane_states"]) == 3
        assert len(row["lane_think"]) == 3
        assert len(row["lane_context_tokens"]) == 3


def test_dashboard_reference_svg_has_one_fixed_safe_topology() -> None:
    parser = _DashboardMarkup()
    markup = (ROOT / "src/chronovisor/dashboard_static/index.html").read_text(
        encoding="utf-8"
    )
    parser.feed(markup)
    elements = parser.elements

    def matching(key: str, value: str) -> list[dict[str, str | None]]:
        return [attrs for _tag, attrs in elements if attrs.get(key) == value]

    assert len(matching("id", "processing-panel")) == 1
    assert len(matching("class", "processing-lanes-card")) == 1
    assert len(matching("id", "decision-trace-panel")) == 1
    assert not matching("id", "processing-trace-connector")
    assert not matching("id", "processing-trace-connector-path")
    assert not matching("data-trace-entry", "true")
    assert len(matching("id", "decision-trace-harness")) == 1
    assert matching("id", "decision-trace-harness")[0]["viewbox"] == ("0 0 1500 650")
    assert matching("id", "decision-trace-harness")[0]["height"] == "607"
    assert matching("id", "trace-arrow")[0]["refx"] == "6"

    assert len(matching("data-trace-key", "artifact")) == 1
    assert len(matching("data-trace-key", "quorum")) == 1
    assert len(matching("data-trace-key", "seal")) == 1
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
    for key in ("packet", "preflight", "execution_plan"):
        node = markup.split(f'data-overall-key="{key}"', 1)[1].split("</g>", 1)[0]
        assert '<circle r="10"></circle>' in node
    style = (ROOT / "src/chronovisor/dashboard_static/style.css").read_text(
        encoding="utf-8"
    )
    assert ".trace-overall .trace-node circle" not in style
    assert len(matching("data-context-tokens", "32768")) == 1
    assert len(matching("data-context-tokens", "65536")) == 1
    assert len(matching("data-context-tokens", "98304")) == 1
    assert len(matching("data-context-tokens", "131072")) == 1
    for tokens in ("32768", "65536", "98304", "131072"):
        option = markup.split(f'data-context-tokens="{tokens}"', 1)[1].split("</g>", 1)[
            0
        ]
        assert '<circle r="10"></circle>' in option
    assert not matching("class", "trace-plan-frame")
    assert {
        attrs["data-decision-lane"]: attrs.get("transform")
        for _tag, attrs in elements
        if attrs.get("data-decision-lane")
    } == {
        "primary": "translate(0 325)",
        "challenger": "translate(0 450)",
        "tie_break": "translate(0 575)",
    }
    primary_lane_markup = markup.split('data-decision-lane="primary"', 1)[1].split(
        'data-decision-lane="challenger"', 1
    )[0]
    assert (
        'data-decision-lane-step="vote" transform="translate(1096 35)"><circle r="10"></circle><text x="-25" y="-17">Vote</text>'
        in primary_lane_markup
    )
    assert len(matching("data-seal-yes-label", "true")) == 1
    assert len(matching("data-seal-no-label", "true")) == 1
    assert len(matching("data-pair-no-label", "true")) == 1

    paths = {
        attrs["data-path-key"]: attrs.get("d")
        for tag, attrs in elements
        if tag == "path" and attrs.get("data-path-key")
    }
    assert {
        "single-artifact",
        "pair-tie_break",
        "pair-artifact-join",
        "tie_break-quorum",
        "quorum-artifact-join",
        "quorum-hold",
        "artifact-seal",
        "seal-decision",
        "seal-hold",
    }.issubset(paths)
    assert all(paths[key] for key in paths)
    assert (
        not {
            "pair-artifact",
            "tie_break-artifact",
            "pair-hold",
            "tie_break-hold",
            "pair-quorum",
            "quorum-artifact",
        }
        & paths.keys()
    )
    assert any(
        tag == "path" and attrs.get("d") == "M1208 410 1248 450 1208 490 1168 450Z"
        for tag, attrs in elements
    )
    context_generate_rails = matching("data-lane-path", "context-generate")
    assert len(context_generate_rails) == 3
    assert all(
        attrs.get("d") == "M575 0 H733"
        and attrs.get("marker-end") == "url(#trace-arrow)"
        for attrs in context_generate_rails
    )
    primary_validate_vote = markup.split('data-decision-lane="primary"', 1)[1].split(
        'data-decision-lane="challenger"', 1
    )[0]
    assert (
        'data-lane-path="validate-vote" '
        'd="M920 0 H1086 Q1096 0 1096 10 V25" '
        'marker-end="url(#trace-arrow)"' in primary_validate_vote
    )

    expected_boundary_paths = {
        "packet-preflight": "M106 10 H288",
        "preflight-execution_plan": "M308 10 H482 Q492 10 492 20 V36",
        "execution-plan-context": (
            "M492 56 V76 Q492 86 482 86 H452 Q442 86 442 96 V140"
        ),
        "plan-dispatch": (
            "M1380 164 H1466 Q1476 164 1476 174 V252 Q1476 262 1466 262 "
            "H190 Q180 262 180 272 V315 Q180 325 190 325 H220"
        ),
        "primary-challenger": (
            "M1096 370 V377 Q1096 387 1086 387 H190 Q180 387 180 397 "
            "V440 Q180 450 190 450 H220"
        ),
        "single-artifact": "M1106 360 H1300 Q1310 360 1310 370 V393",
        "challenger-agree": "M1106 450 H1168",
        "pair-tie_break": (
            "M1208 490 V502 Q1208 512 1198 512 "
            "H190 Q180 512 180 522 V565 Q180 575 190 575 H220"
        ),
        "pair-artifact-join": (
            "M1248 450 H1278 Q1288 450 1288 440 V414 Q1288 404 1298 404 H1299"
        ),
        "tie_break-quorum": "M1106 575 H1217",
        "quorum-artifact-join": (
            "M1248 545 V524 Q1248 514 1258 514 H1300 Q1310 514 1310 504 V415"
        ),
        "quorum-hold": (
            "M1278 575 H1408 Q1418 575 1418 585 V590 Q1418 600 1428 600 H1448"
        ),
        "artifact-seal": ("M1321 404 H1360 Q1370 404 1370 414 V470"),
        "seal-decision": "M1404 500 H1449",
        "seal-hold": "M1370 530 V540 Q1370 550 1380 550 H1450 Q1460 550 1460 560 V588",
    }
    assert {key: paths[key] for key in expected_boundary_paths} == (
        expected_boundary_paths
    )
    assert all(
        matching("data-path-key", key)[0].get("marker-end") == "url(#trace-arrow)"
        for key in (
            "single-artifact",
            "pair-artifact-join",
            "quorum-artifact-join",
        )
    )
    assert not {"quorum-artifact-trunk", "artifact-input"} & paths.keys()

    ingest_markup = markup.split(
        '<g class="trace-ingest-job" data-ingest-job-flow', 1
    )[1].split("</svg>", 1)[0]
    ingest_entries = [
        attrs
        for tag, attrs in elements
        if tag == "path" and attrs.get("data-ingest-job-entry")
    ]
    assert len(ingest_entries) == 1
    ingest_paths = {
        attrs["data-ingest-job-path"]: attrs
        for tag, attrs in elements
        if tag == "path" and attrs.get("data-ingest-job-path")
    }
    assert matching("data-ingest-job-step", "mutation")
    assert ingest_paths["accepted"].get("data-ingest-job-to") == "mutation"
    assert ingest_paths["apply"]["d"].split(" ", 2)[:2] != (
        ingest_paths["noop"]["d"].split(" ", 2)[:2]
    )
    assert "decision-role" not in ingest_markup
    assert "decision-sublabel" not in ingest_markup
    assert ">Raw</text>" not in ingest_markup
    assert ">Triage LLM</text>" not in ingest_markup
    for branch, label in {
        "accepted": "ACCEPTED",
        "apply": "APPLY",
        "noop": "NOOP",
        "retry": "RETRY",
        "hold": "HOLD",
    }.items():
        assert f'data-ingest-branch-label="{branch}"' in ingest_markup
        assert f">{label}</text>" in ingest_markup

    repair_paths = [
        attrs
        for tag, attrs in elements
        if tag == "path" and attrs.get("data-repair-lane")
    ]
    assert len(repair_paths) == 3
    assert all(
        attrs.get("d") == "M743 10 V30 Q743 40 753 40 H900 Q910 40 910 30 V10"
        and attrs.get("marker-start") == "url(#trace-arrow)"
        and attrs.get("marker-end") is None
        for attrs in repair_paths
    )

    reasoning_guides = [
        attrs
        for tag, attrs in elements
        if tag == "path" and attrs.get("class") == "trace-reasoning-guide"
    ]
    assert len(reasoning_guides) == 1
    assert [attrs.get("d") for attrs in reasoning_guides] == [
        "M812 118 V58 Q812 48 822 48 H903 "
        "M921 48 H1320 Q1330 48 1330 58 V118 "
        "M858 164 H882 Q892 164 892 154 V116 Q892 106 902 106 H903 "
        "M921 106 H1245 Q1255 106 1255 116 V154 Q1255 164 1265 164 H1270 "
        "M858 164 H903 M921 164 H1270 "
        "M858 164 H882 Q892 164 892 174 V212 Q892 222 902 222 H903 "
        "M921 222 H1245 Q1255 222 1255 212 V174 Q1255 164 1265 164 H1270",
    ]
    assert [
        (
            attrs.get("data-reasoning-output"),
            attrs.get("d"),
            attrs.get("marker-end"),
        )
        for tag, attrs in elements
        if tag == "path" and attrs.get("data-reasoning-output")
    ] == [
        (
            "off",
            "M921 48 H1320 Q1330 48 1330 58 V118",
            "url(#trace-arrow)",
        ),
        (
            "low",
            "M921 106 H1245 Q1255 106 1255 116 V154 Q1255 164 1265 164 H1270",
            None,
        ),
        ("medium", "M921 164 H1270", None),
        (
            "high",
            "M921 222 H1245 Q1255 222 1255 212 V174 Q1255 164 1265 164 H1270",
            None,
        ),
    ]
    assert paths["reasoning-off"] == "M812 118 V58 Q812 48 822 48 H903"
    assert matching("data-path-key", "plan-fit")[0].get("class") == (
        "trace-path trace-reasoning-path"
    )
    assert all(
        attrs.get("marker-end") == "url(#trace-arrow)"
        for tag, attrs in elements
        if tag == "path"
        and attrs.get("data-path-key")
        in {
            "reasoning-off",
            "reasoning-low",
            "reasoning-medium",
            "reasoning-high",
            "plan-fit",
        }
    )
    assert all(
        attrs.get("y") == "28"
        for tag, attrs in elements
        if tag == "text" and attrs.get("data-lane-result")
    )
    assert len(matching("class", "trace-label-backdrop")) == 1
    assert len(matching("class", "trace-selected-core")) == 8
    assert {
        attrs["data-reasoning-key"]: attrs.get("transform")
        for _tag, attrs in elements
        if attrs.get("data-reasoning-key")
    } == {
        "off": "translate(912 48)",
        "low": "translate(912 106)",
        "medium": "translate(912 164)",
        "high": "translate(912 222)",
    }
    assert all(
        attrs.get("y") == "34"
        for tag, attrs in elements
        if tag == "text" and attrs.get("class") == "trace-reasoning-detail"
    )
    assert matching("data-plan-value", "fit")[0]["y"] == "174"
    assert matching("data-plan-fit-pass", "true")[0]["y"] == "190"
    assert matching("class", "trace-branch-node-label")[0]["x"] == "16"
    assert matching("class", "trace-branch-node-label")[0]["y"] == "4"
    assert matching("data-overall-key", "execution_plan")[0]["transform"] == (
        "translate(492 46)"
    )
    assert matching("class", "trace-artifact-label")[0]["x"] == "-16"
    assert matching("class", "trace-artifact-label")[0]["text-anchor"] == "end"
    assert [
        attrs.get("y")
        for _tag, attrs in elements
        if "trace-path-label" in (attrs.get("class") or "").split()
    ] == ["258", "381"]
    assert 325 - 262 == 450 - 387 == 575 - 512 == 63
    assert matching("class", "trace-branch-label trace-yes-label")[0]["y"] == "438"
    assert matching("class", "trace-branch-label trace-no-label")[0]["y"] == "506"
    pair_yes_label = matching("data-pair-yes-label", "true")[0]
    assert (pair_yes_label["x"], pair_yes_label["y"]) == ("1258", "438")
    assert matching("data-pair-no-label", "true")[0]["x"] == "1180"
    assert matching("data-quorum-yes-label", "true")[0]["x"] == "1268"
    assert matching("data-quorum-no-label", "true")[0]["x"] == "1336"
    assert matching("data-seal-yes-label", "true")[0]["x"] == "1422"
    assert matching("data-seal-no-label", "true")[0]["x"] == "1382"
    assert paths["plan-context"] == (
        "M442 160 V204 Q442 214 452 214 H724 Q734 214 734 204 "
        "V174 Q734 164 744 164 H766"
    )
    assert "plan-headroom" not in paths
    assert (
        next(
            attrs.get("marker-end")
            for tag, attrs in elements
            if tag == "path" and attrs.get("data-path-key") == "plan-context"
        )
        == "url(#trace-arrow)"
    )
    context_guides = [
        attrs
        for tag, attrs in elements
        if tag == "path" and attrs.get("data-context-guide")
    ]
    assert len(context_guides) == 1
    assert context_guides[0]["data-context-guide"] == "all"
    assert context_guides[0]["d"] == (
        "M492 56 V76 Q492 86 482 86 H352 Q342 86 342 96 V140 "
        "M492 56 V76 Q492 86 482 86 H452 Q442 86 442 96 V140 "
        "M492 56 V76 Q492 86 502 86 H532 Q542 86 542 96 V140 "
        "M492 56 V76 Q492 86 502 86 H632 Q642 86 642 96 V140 "
        "M442 160 V204 Q442 214 452 214 "
        "M542 160 V204 Q542 214 552 214 "
        "M642 160 V204 Q642 214 652 214 "
        "M342 160 V204 Q342 214 352 214 H724 Q734 214 734 204 "
        "V174 Q734 164 744 164 H766"
    )
    assert context_guides[0].get("marker-end") is None
    assert not matching("class", "trace-context-rail")
    assert (
        len(matching("class", "trace-label-backdrop trace-context-kicker-backdrop"))
        == 1
    )
    assert all(
        attrs.get("x") == "15" and attrs.get("y") == "27"
        for attrs in matching("data-context-label", "true")
    )
    assert [
        attrs.get("transform") for attrs in matching("data-context-option", "true")
    ] == [
        "translate(342 150)",
        "translate(442 150)",
        "translate(542 150)",
        "translate(642 150)",
    ]
    assert matching("data-plan-value", "context-selection")[0]["y"] == "239"


@pytest.mark.parametrize("viewport_width", [640, 700, 900, 980, 1050, 2048])
def test_dashboard_css_keeps_processing_milestones_and_svg_paths_connected(
    tmp_path: Path,
    viewport_width: int,
) -> None:
    style = ROOT / "src/chronovisor/dashboard_static/style.css"
    page = tmp_path / "single-model-geometry.html"
    page.write_text(
        """<!doctype html>
<link rel="stylesheet" href="/style.css">
<div class="processing-lanes-card">
  <div class="processing-lanes">
    <section class="processing-lane" data-count="2" aria-selected="false">
      <strong class="processing-lane-label">Two</strong>
      <div class="processing-track">
        <span class="processing-step"><span>Inspect</span></span>
        <span class="processing-step"><span>Consensus</span></span>
      </div>
    </section>
    <section class="processing-lane" data-count="3" aria-selected="false">
      <strong class="processing-lane-label">Three</strong>
      <div class="processing-track">
        <span class="processing-step"><span>Local fix</span></span>
        <span class="processing-step"><span>Verify</span></span>
        <span class="processing-step"><span>Escalate</span></span>
      </div>
    </section>
    <section class="processing-lane" data-count="4" aria-selected="false">
      <strong class="processing-lane-label">Four</strong>
      <div class="processing-track">
        <span class="processing-step"><span>Search</span></span>
        <span class="processing-step"><span>Rerank</span></span>
        <span class="processing-step"><span>Authority</span></span>
        <span class="processing-step"><span>Commit</span></span>
      </div>
    </section>
    <section class="processing-lane" data-count="5" aria-selected="false">
      <strong class="processing-lane-label">Five</strong>
      <div class="processing-track">
        <span class="processing-step"><span>Raw</span></span>
        <span class="processing-step"><span>Triage</span></span>
        <span class="processing-step"><span>Generate</span></span>
        <span class="processing-step"><span>Consensus</span></span>
        <span class="processing-step"><span>Apply</span></span>
      </div>
    </section>
    <section class="processing-lane" data-count="6" aria-selected="false">
      <strong class="processing-lane-label">Six</strong>
      <div class="processing-track">
        <span class="processing-step"><span>Discover</span></span>
        <span class="processing-step"><span>Extract</span></span>
        <span class="processing-step"><span>Verify</span></span>
        <span class="processing-step"><span>Consolidate</span></span>
        <span class="processing-step"><span>Evaluate</span></span>
        <span class="processing-step"><span>Promote</span></span>
      </div>
    </section>
  </div>
</div>
<div class="decision-single-authority" id="decision-single-authority">
  <span class="decision-single-authority-title">Single Authority</span>
  <span><b>MODEL</b><strong>Qwen3.8-Flash-Next-oQ4e-mtp</strong></span>
  <span><b>REVISION</b><strong>2615fc0e976e65c2f3b55daca3a948f1cdc5b9f8</strong></span>
  <span><b>STATUS</b><strong data-single-status>Validating</strong></span>
  <span><b>TARGET</b><strong>1</strong></span>
  <span><b>REPAIR</b><strong>REPAIR ≠ VOTE</strong></span>
</div>
<svg class="decision-trace-harness single-model" viewBox="0 0 1500 550" width="1500" height="550">
  <path id="dispatch" d="M1380 164 H1466 Q1476 164 1476 174 V297 Q1476 307 1466 307 H190 Q180 307 180 317 V394 Q180 404 190 404 H220"></path>
  <path id="single-artifact" d="M920 404 H1039"></path>
  <path id="seal-decision" data-path-key="seal-decision" d="M1190 434 V483"></path>
  <path id="seal-hold" data-path-key="seal-hold" d="M1224 404 H1318"></path>
  <g class="decision-lane" data-decision-lane="primary" transform="translate(0 325)">
    <g id="trigger" transform="translate(230 0)"><circle r="10"></circle></g>
    <g id="validate" transform="translate(910 0)"><circle r="10"></circle></g>
  </g>
  <g id="artifact" transform="translate(1050 404)"><circle r="11"></circle></g>
  <g id="decision" data-trace-key="decision" transform="translate(1190 494)"><circle r="11"></circle><text>Validated</text></g>
  <g id="hold" data-trace-key="hold" transform="translate(1330 404)"><circle r="12"></circle><text>Hold</text></g>
</svg>
<script src="/geometry.js"></script>
""",
        encoding="utf-8",
    )
    script = tmp_path / "geometry.js"
    script.write_text(
        """
const screenPoint = (node, point) => point.matrixTransform(node.getScreenCTM());
const dispatch = document.querySelector("#dispatch");
const single = document.querySelector("#single-artifact");
const sealDecision = document.querySelector("#seal-decision");
const sealHold = document.querySelector("#seal-hold");
const decision = document.querySelector("#decision");
const hold = document.querySelector("#hold");
const lane = document.querySelector('[data-decision-lane="primary"]');
const trigger = screenPoint(document.querySelector("#trigger"), new DOMPoint(0, 0));
const validate = screenPoint(document.querySelector("#validate"), new DOMPoint(0, 0));
const dispatchEnd = screenPoint(dispatch, dispatch.getPointAtLength(dispatch.getTotalLength()));
const dispatchRail = screenPoint(dispatch, dispatch.getPointAtLength(dispatch.getTotalLength() / 2));
const planBottom = screenPoint(document.querySelector("svg"), new DOMPoint(0, 210));
const singleStart = screenPoint(single, single.getPointAtLength(0));
const singleEnd = screenPoint(single, single.getPointAtLength(single.getTotalLength()));
const artifact = document.querySelector("#artifact");
const artifactLeft = screenPoint(artifact, new DOMPoint(-11, 0));
const holdDisplay = getComputedStyle(hold).display;
const sealDecisionStart = screenPoint(sealDecision, sealDecision.getPointAtLength(0));
const sealDecisionEnd = screenPoint(sealDecision, sealDecision.getPointAtLength(sealDecision.getTotalLength()));
const decisionCenter = screenPoint(decision, new DOMPoint(0, 0));
const sealHoldStart = screenPoint(sealHold, sealHold.getPointAtLength(0));
const sealHoldEnd = screenPoint(sealHold, sealHold.getPointAtLength(sealHold.getTotalLength()));
const holdCenter = holdDisplay === "none" ? null : screenPoint(hold, new DOMPoint(0, 0));
const authority = document.querySelector("#decision-single-authority");
const authorityStatus = authority.querySelector("[data-single-status]");
authorityStatus.textContent = "Validating";
const shortAuthorityHeight = authority.getBoundingClientRect().height;
authorityStatus.textContent = "Triage done · Generate active";
const longAuthorityHeight = authority.getBoundingClientRect().height;
const processing = [...document.querySelectorAll(".processing-lane")].map((lane) => {
  const track = lane.querySelector(".processing-track");
  const lastStep = track.querySelector(".processing-step:last-child");
  const lastLabel = track.querySelector(".processing-step:last-child > span");
  const laneBounds = lane.getBoundingClientRect();
  const trackBounds = track.getBoundingClientRect();
  const labelBounds = lastLabel.getBoundingClientRect();
  return {
    count: Number(lane.dataset.count),
    laneRight: laneBounds.right,
    trackRight: trackBounds.right,
    lastStepLeft: lastStep.getBoundingClientRect().left,
    labelRight: labelBounds.right,
  };
});
fetch("/geometry-result", {
  method: "POST",
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify({
    transform: getComputedStyle(lane).transform,
    dispatchDeltaY: Math.abs(trigger.y - dispatchEnd.y),
    dispatchCenterDeltaY: Math.abs(dispatchRail.y - ((planBottom.y + trigger.y) / 2)),
    singleDeltaY: Math.abs(validate.y - singleStart.y),
    singleArtifactDeltaX: Math.abs(artifactLeft.x - singleEnd.x),
    singleArtifactDeltaY: Math.abs(artifactLeft.y - singleEnd.y),
    holdDisplay,
    sealDecisionStraightDeltaX: Math.abs(sealDecisionStart.x - sealDecisionEnd.x),
    sealDecisionDeltaX: Math.abs(decisionCenter.x - sealDecisionEnd.x),
    sealDecisionDeltaY: Math.abs(decisionCenter.y - sealDecisionEnd.y),
    decisionRadiusY: 11 * Math.abs(decision.getScreenCTM().d),
    sealHoldStraightDeltaY: Math.abs(sealHoldStart.y - sealHoldEnd.y),
    sealHoldDeltaX: holdCenter ? Math.abs(holdCenter.x - sealHoldEnd.x) : null,
    sealHoldDeltaY: holdCenter ? Math.abs(holdCenter.y - sealHoldEnd.y) : null,
    holdRadiusX: holdCenter ? 12 * Math.abs(hold.getScreenCTM().a) : null,
    authority: {
      shortHeight: shortAuthorityHeight,
      longHeight: longAuthorityHeight,
      clientWidth: authority.clientWidth,
      scrollWidth: authority.scrollWidth,
    },
    processing,
  }),
});
""",
        encoding="utf-8",
    )
    chrome = next(
        (
            candidate
            for candidate in (
                shutil.which("google-chrome"),
                shutil.which("chromium"),
                shutil.which("chromium-browser"),
                "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
            )
            if candidate and Path(candidate).is_file()
        ),
        None,
    )
    assert chrome is not None, "headless Chrome is required for Dashboard DOM tests"
    geometry: list[dict[str, object]] = []
    result_ready = threading.Event()

    class Handler(dashboard.DashboardHandler):
        def do_GET(self) -> None:
            if self.path in {"/", "/geometry.js", "/style.css"}:
                source = {"/": page, "/geometry.js": script, "/style.css": style}[
                    self.path
                ]
                dashboard._file_response(self, source)
                return
            self.send_error(404)

        def do_POST(self) -> None:
            if self.path != "/geometry-result":
                self.send_error(404)
                return
            length = int(self.headers.get("Content-Length", "0"))
            geometry.append(json.loads(self.rfile.read(length)))
            result_ready.set()
            self.send_response(204)
            self.end_headers()

        def log_message(self, _format: str, *_args: object) -> None:
            return

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    server_thread = threading.Thread(target=server.serve_forever, daemon=True)
    server_thread.start()
    screenshot = tmp_path / f"processing-{viewport_width}.png"
    process = subprocess.Popen(
        [
            chrome,
            "--headless=new",
            "--no-sandbox",
            "--disable-gpu",
            "--disable-background-networking",
            "--disable-component-update",
            "--no-first-run",
            f"--window-size={viewport_width},900",
            f"--screenshot={screenshot}",
            f"--user-data-dir={tmp_path / 'single-model-profile'}",
            f"http://127.0.0.1:{server.server_port}/",
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        assert result_ready.wait(10)
        deadline = time.monotonic() + 3
        while time.monotonic() < deadline and not screenshot.is_file():
            if process.poll() not in (None, 0):
                break
            time.sleep(0.05)
    finally:
        process.terminate()
        try:
            process.wait(timeout=2)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=2)
        server.shutdown()
        server.server_close()
        server_thread.join(timeout=2)
    assert screenshot.is_file() and screenshot.stat().st_size > 0
    assert struct.unpack(">II", screenshot.read_bytes()[16:24]) == (
        viewport_width,
        900,
    )
    assert geometry[0]["transform"] == "matrix(1, 0, 0, 1, 0, 404)"
    assert geometry[0]["dispatchDeltaY"] <= 0.01
    assert geometry[0]["dispatchCenterDeltaY"] <= 0.01
    assert geometry[0]["singleDeltaY"] <= 0.01
    assert geometry[0]["singleArtifactDeltaX"] <= 0.01
    assert geometry[0]["singleArtifactDeltaY"] <= 0.01
    assert geometry[0]["holdDisplay"] != "none"
    assert geometry[0]["sealDecisionStraightDeltaX"] <= 0.01
    assert geometry[0]["sealDecisionDeltaX"] <= 0.01
    assert abs(
        geometry[0]["sealDecisionDeltaY"] - geometry[0]["decisionRadiusY"]
    ) <= 0.01
    assert geometry[0]["sealHoldStraightDeltaY"] <= 0.01
    assert abs(geometry[0]["sealHoldDeltaX"] - geometry[0]["holdRadiusX"]) <= 0.01
    assert geometry[0]["sealHoldDeltaY"] <= 0.01
    assert geometry[0]["authority"]["longHeight"] == geometry[0]["authority"]["shortHeight"]
    assert geometry[0]["authority"]["scrollWidth"] <= geometry[0]["authority"]["clientWidth"]
    assert [row["count"] for row in geometry[0]["processing"]] == [2, 3, 4, 5, 6]
    for row in geometry[0]["processing"]:
        assert row["trackRight"] <= row["laneRight"]
        assert abs(row["lastStepLeft"] - row["trackRight"]) <= 0.1
        assert row["labelRight"] <= row["laneRight"]


def test_dashboard_reference_quorum_hold_paths_and_label_stay_in_bounds() -> None:
    parser = _DashboardMarkup()
    parser.feed(
        (ROOT / "src/chronovisor/dashboard_static/index.html").read_text(
            encoding="utf-8"
        )
    )
    elements = parser.elements
    paths = {
        attrs["data-path-key"]: attrs.get("d")
        for tag, attrs in elements
        if tag == "path" and attrs.get("data-path-key")
    }

    assert paths["quorum-hold"] == (
        "M1278 575 H1408 Q1418 575 1418 585 V590 Q1418 600 1428 600 H1448"
    )
    assert paths["seal-hold"] == (
        "M1370 530 V540 Q1370 550 1380 550 H1450 Q1460 550 1460 560 V588"
    )

    hold_reason = next(
        attrs for _tag, attrs in elements if attrs.get("data-hold-reason") == "true"
    )
    assert hold_reason["class"] == "trace-hold-reason"
    assert hold_reason["x"] == "28"
    assert 1460 + int(hold_reason["x"]) == 1488 < 1500
    assert hold_reason["y"] == "43"
    assert 600 + int(hold_reason["y"]) == 643 < 650

    style = (ROOT / "src/chronovisor/dashboard_static/style.css").read_text(
        encoding="utf-8"
    )
    hold_reason_style = style.split(".decision-trace-harness .trace-hold-reason {", 1)[
        1
    ].split("}", 1)[0]
    assert "text-anchor: end;" in hold_reason_style


def test_dashboard_reference_keeps_selection_and_bucket_truth() -> None:
    static = ROOT / "src/chronovisor/dashboard_static"
    page = (static / "index.html").read_text(encoding="utf-8")
    style = (static / "style.css").read_text(encoding="utf-8")
    app = (static / "app.js").read_text(encoding="utf-8")
    renderer = (static / "app-renderer.js").read_text(encoding="utf-8")

    assert "processing-trace-connector" not in page
    assert "processingTraceConnector" not in app
    assert "updateProcessingTraceConnector" not in renderer
    assert "processingTraceResizeObserver" not in renderer
    assert "ResizeObserver" not in renderer
    assert ".processing-trace-connector" not in style
    assert (
        'node.className = `processing-step ${fmt(step.status, "pending")}`;' in renderer
    )
    assert "setDecisionSvgState(node, nodes[node.dataset.overallKey]);" in renderer
    processing_active_style = style.rsplit(".processing-step.active::before {", 1)[
        1
    ].split("}", 1)[0]
    assert "border-color: #ff9d00;" in processing_active_style
    assert "background: #ff9d00;" in processing_active_style

    assert [
        token
        for token in ("32768", "65536", "98304", "131072")
        if f'data-context-tokens="{token}"' in page
    ] == ["32768", "65536", "98304", "131072"]
    assert '<p id="work-summary">Waiting · 0 / 3</p>' in page
    assert (
        page.index('data-context-guide="all"')
        < page.index('data-path-key="execution-plan-context"')
        < page.index("trace-context-kicker-backdrop")
        < page.index(">CONTEXT WINDOW<")
    )
    assert (
        page.index('data-context-guide="all"')
        < page.index('data-path-key="plan-context"')
        < page.index('data-plan-value="context-selection"')
    )
    unselected_track_style = style.split(
        '.processing-lane[aria-selected="false"] .processing-track {', 1
    )[1].split("}", 1)[0]
    assert "opacity: 0.42;" in unselected_track_style
    assert "visibility: hidden;" not in unselected_track_style
    assert '.processing-lane.active[aria-selected="false"] .processing-track {' in style
    assert "opacity: 0.72;" in style
    assert '.processing-lane[aria-selected="true"] {' in style
    assert "box-shadow: inset 2px 0 #898dff;" in style
    pending_repair_style = style.split(
        ".decision-trace-harness .trace-repair-loop.pending {", 1
    )[1].split("}", 1)[0]
    assert "stroke-dasharray: 4 5;" in pending_repair_style
    assert "opacity: 0.66;" in pending_repair_style
    assert "filter: none;" in pending_repair_style
    reference_style = style.split("/* ---------- reference-first", 1)[1]
    processing_panel_style = reference_style.split(".processing-panel {", 1)[1].split(
        "}", 1
    )[0]
    assert "height: auto;" in processing_panel_style
    assert "height: 1030px;" not in style
    assert "height: 802px;" not in style
    assert style.count("height: 607px;") == 2
    assert "trace-check" not in page
    assert ".trace-check" not in style
    assert (
        ".decision-trace-harness .trace-overall .trace-node.done circle,\n"
        ".decision-trace-harness .done.decision-lane-step circle,\n"
        ".decision-trace-harness .decision-lane-step.done circle {\n"
        "  fill: url(#trace-active-core);\n"
        "  stroke: #9296ff;\n"
        "  stroke-width: 2.1;"
    ) in style
    assert (
        ".decision-trace-harness .done.trace-shared-node circle,\n"
        ".decision-trace-harness .trace-shared-node.done circle {\n"
        "  fill: url(#trace-active-core);\n"
        "  stroke: #9296ff;"
    ) in style
    done_node_style = style.split(
        ".decision-trace-harness .done.trace-node circle,", 1
    )[1].split("}", 1)[0]
    assert "filter: url(#trace-glow-violet);" in done_node_style
    assert ".decision-trace-harness .trace-single-path.pending" not in style
    assert ".decision-trace-harness .trace-hold-path.pending" not in style
    assert ".decision-trace-harness .skipped.trace-path" not in style
    assert ".decision-trace-harness .trace-tie-path.pending" not in style
    assert ".decision-trace-harness .done .trace-path" not in style
    assert (
        ".decision-trace-harness .trace-context-guide,\n"
        ".decision-trace-harness .trace-reasoning-guide {\n"
        "  fill: none;\n"
        "  stroke: #7d8992;"
    ) in style
    assert (
        '.decision-trace-harness .trace-diamond [data-plan-value="fit"] {' not in style
    )
    assert "headroom OK ·" not in renderer
    assert 'fmt(projection.labels?.fit, "WAITING")' in renderer
    active_path_style = style.split(".decision-trace-harness .active.trace-path,", 1)[
        1
    ].split("}", 1)[0]
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
        "  stroke: #7d8992;\n"
        "  stroke-dasharray: 4 5;\n"
        "  opacity: 0.66;\n"
        "}"
    ) in style
    assert ".decision-trace-harness .trace-lane-rail.pending" not in style
    assert (
        ".decision-trace-harness .decision-lane-step.active circle {\n"
        "  fill: url(#trace-processing-core);\n"
        "  stroke: #ff9d00;\n"
        "  filter: url(#trace-glow-orange);"
    ) in style
    assert (
        ".decision-trace-harness .trace-lane-rail.active {\n"
        "  stroke: #ff8a00;\n"
        "  filter: drop-shadow(0 0 3px rgba(255, 138, 0, 0.68));"
    ) in style
    active_rail_style = style.split(
        ".decision-trace-harness .trace-lane-rail.active {", 1
    )[1].split("}", 1)[0]
    assert "filter: url(" not in active_rail_style
    assert (
        ".decision-trace-harness .decision-lane-step.active circle,\n"
        "  .decision-trace-harness .trace-lane-rail.active,\n"
        "  .decision-model-stream pre::after {\n"
        "    animation: none;"
    ) in style
    assert '<radialGradient id="trace-processing-core">' in page
    assert (
        ".decision-trace-harness .trace-context-option.selected [data-context-label] {\n"
        "  fill: #ffb340;"
    ) in style
    assert (
        ".decision-trace-harness .trace-reasoning.selected [data-reasoning-label] {\n"
        "  fill: #ffb340;"
    ) in style
    assert (
        ".decision-trace-harness .trace-yes-label,\n"
        ".decision-trace-harness .trace-no-label {\n"
        "  fill: #7d8992;\n"
        "  opacity: 0.18;"
    ) in style
    assert (
        ".decision-trace-harness .trace-yes-label.active,\n"
        ".decision-trace-harness .trace-yes-label.done {\n"
        "  fill: #74d84f;\n"
        "  opacity: 1;"
    ) in style
    assert (
        ".decision-trace-harness .trace-no-label.active,\n"
        ".decision-trace-harness .trace-no-label.done,\n"
        ".decision-trace-harness .trace-no-label.error {\n"
        "  fill: #ff3948;\n"
        "  opacity: 1;"
    ) in style
    assert ".decision-trace-harness .trace-hold-node.pending" not in style
    assert (
        ".decision-trace-harness .trace-hold-node.error text {\n  fill: #ff3948;"
    ) in style
    assert (
        "filter: url(#trace-glow-violet);"
        in style.split(".decision-trace-harness .active.trace-node circle,", 1)[
            1
        ].split("}", 1)[0]
    )
    assert ".decision-console-chrome {\n  display: grid;" in style
    assert ".decision-console-lights i:first-child {\n  background: #ff5f57;" in style
    assert "grid-template-columns: minmax(0, 2fr) minmax(0, 3fr);" in style
    assert "grid-template-rows: 32px 32px minmax(0, 1fr);" in style
    assert (
        ".decision-stream-pane {\n"
        "  grid-template-rows: 32px minmax(0, 1fr) 32px;"
    ) in style
    responsive_style = style.split("@media (max-width: 980px) {", 1)[1].split(
        "@media (prefers-reduced-motion: reduce)", 1
    )[0]
    assert (
        ".decision-stream-pane {\n"
        "    grid-template-rows: 32px minmax(0, 1fr) 55px;"
    ) in responsive_style
    assert "height: 261px;" in style
    assert "min-height: 261px;" in style
    assert ".decision-transition-feed {\n  display: grid;" in style
    assert ".decision-transition-event {\n  display: grid;" in style
    assert ".decision-generation-meter {\n  display: grid;" in style
    assert ".decision-generation-track.indeterminate i {" not in style
    assert "@keyframes decision-console-scan" not in style
    assert ".decision-events" not in page
    frame = renderer.split("function renderDecisionTraceFrame", 1)[1].split(
        "function setDecisionTransitionState", 1
    )[0]
    assert "updateProcessingTraceSelection(trace);" in frame
    harness = renderer.split("function updateDecisionSvgHarness", 1)[1].split(
        "function decisionEventText", 1
    )[0]
    assert "const projection = decisionTraceProjection(trace);" in harness
    assert (
        'harness.dataset.projectionStatus = projection ? "ok" : "missing";' in harness
    )
    assert "projection.nodes" in harness
    assert "projection.paths" in harness
    assert "projection.context" in harness
    assert "projection.reasoning" in harness
    assert "projection.lanes" in harness
    assert "decisionTracePlayback" not in renderer
    assert "decisionOverallState" not in renderer
    assert "decisionSealStates" not in renderer
    assert "decisionPairBranchStates" not in renderer
    assert "decisionReasoningPlanState" not in renderer
    assert "decisionRepairState" not in renderer
    assert '["[data-seal-yes-label]", "seal-decision"]' in harness
    assert '["[data-seal-no-label]", "seal-hold"]' in harness
    assert "quorum-artifact-trunk" not in harness
    assert "artifact-input" not in harness
    assert "Math.sign(contextX - 492)" in harness
    assert "Math.abs(contextX - 492) / 2" in harness
    assert "`M492 56 V${86 - contextRadius} Q492 86 ${492 +" in harness
    assert "${86 + contextRadius} V140`" in harness
    assert "`M${contextX} 160 V204 Q${contextX} 214 ${contextX + 10} 214 `" in harness
    assert '+ "H724 Q734 214 734 204 V174 Q734 164 744 164 H766"' in harness
    assert 'data-reasoning-output="${mode}"' in harness
    assert 'node.classList.toggle("selected", option.selected === true);' in harness
    assert 'node.classList.toggle("selected", mode === reasoning.selected);' in harness
    assert 'observedPhases.add("load")' not in renderer


def test_dashboard_reference_renderer_fails_closed_without_projection() -> None:
    renderer = (ROOT / "src/chronovisor/dashboard_static/app-renderer.js").read_text(
        encoding="utf-8"
    )
    scenario = f"""
const vm = require("node:vm");
const node = (traceKey = null) => ({{
  dataset: traceKey ? {{ traceKey }} : {{}},
  classList: {{ remove() {{}}, add() {{}}, toggle() {{}} }},
  querySelector: () => node(),
  querySelectorAll: () => [],
  setAttribute() {{}},
  textContent: "",
}});
const traceNodes = Object.fromEntries(["artifact", "decision"].map((key) => [key, node(key)]));
const harness = node();
harness.querySelectorAll = (selector) => selector.includes("[data-trace-key]")
  ? Object.values(traceNodes)
  : [];
harness.querySelector = () => node();
const sandbox = {{
  window: {{ matchMedia: () => ({{ matches: true }}) }},
  els: {{ decisionTraceHarness: harness }},
}};
vm.createContext(sandbox);
vm.runInContext({json.dumps(renderer)}, sandbox);
sandbox.updateDecisionSvgHarness({{
  state: "agreed",
  overall: [{{ key: "artifact", status: "done" }}, {{ key: "decision", status: "done" }}],
}});
process.stdout.write(JSON.stringify({{
  status: harness.dataset.projectionStatus,
  nodes: Object.fromEntries(Object.entries(traceNodes).map(([key, value]) => [key, value.dataset.state])),
}}));
"""
    completed = subprocess.run(
        ["node", "-"],
        input=scenario,
        check=True,
        capture_output=True,
        encoding="utf-8",
    )

    assert json.loads(completed.stdout) == {
        "status": "missing",
        "nodes": {"artifact": "pending", "decision": "pending"},
    }


def test_decision_trace_stepper_advances_and_checks_every_frame(tmp_path: Path) -> None:
    scenarios = decision_trace_step_scenarios()
    assert {scenario["case"]["id"] for scenario in scenarios} == set("ABCDEF")
    assert [len(scenario["frames"]) for scenario in scenarios] == [
        5,
        13,
        17,
        21,
        24,
        23,
    ]
    for scenario in scenarios:
        case = scenario["case"]
        frames = scenario["frames"]
        assert [frame["index"] for frame in frames] == list(range(len(frames)))
        assert frames[0]["label"] == "idle"
        assert all(
            frame["trace"]["projection"] != previous["trace"]["projection"]
            for previous, frame in zip(frames, frames[1:], strict=False)
        )
        final = frames[-1]["trace"]
        assert final["state"] == case["state"]
        assert [lane["state"] for lane in final["lanes"]] == case["lane_states"]
        assert [lane["context_tokens"] for lane in final["lanes"]] == case[
            "lane_context_tokens"
        ]
        for frame in frames:
            trace = frame["trace"]
            projection = trace["projection"]
            if frame["label"] == "tie_break-complete":
                assert projection["paths"]["tie_break-quorum"] == "active"
                assert projection["nodes"]["quorum"] == "active"
            if frame["label"] == "decision" and trace["state"] == "agreed":
                branch_path = (
                    "pair-artifact-join"
                    if case["branch"] == "pair-agreement"
                    else "quorum-artifact-join"
                )
                assert projection["paths"][branch_path] == "active"
                assert projection["nodes"]["artifact"] == "pending"
                assert projection["nodes"]["seal"] == "pending"
                assert projection["nodes"]["decision"] == "pending"
                assert projection["paths"]["artifact-seal"] == "pending"
                assert projection["paths"]["seal-decision"] == "pending"

    chrome = next(
        (
            candidate
            for candidate in (
                shutil.which("google-chrome"),
                shutil.which("chromium"),
                shutil.which("chromium-browser"),
                "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
            )
            if candidate and Path(candidate).is_file()
        ),
        None,
    )
    assert chrome is not None, "headless Chrome is required for Dashboard DOM tests"

    browser_results: list[dict[str, object]] = []
    server = ThreadingHTTPServer(
        ("127.0.0.1", 0), stepper_handler(scenarios, browser_results)
    )
    server_thread = threading.Thread(target=server.serve_forever, daemon=True)
    server_thread.start()
    process = None
    browser_stderr = ""
    try:
        process = subprocess.Popen(
            [
                chrome,
                "--headless=new",
                "--no-sandbox",
                "--disable-dev-shm-usage",
                "--disable-gpu",
                "--disable-background-networking",
                "--disable-component-update",
                "--no-first-run",
                "--force-prefers-reduced-motion=reduce",
                "--run-all-compositor-stages-before-draw",
                "--virtual-time-budget=3000",
                "--window-size=1280,1400",
                f"--user-data-dir={tmp_path / 'stepper-profile'}",
                f"http://127.0.0.1:{server.server_port}/?audit=1",
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            text=True,
        )
        expected_count = sum(len(scenario["frames"]) for scenario in scenarios)
        deadline = time.monotonic() + 30
        while len(browser_results) < expected_count and time.monotonic() < deadline:
            if process.poll() is not None:
                break
            time.sleep(0.05)
        process.terminate()
        try:
            _, browser_stderr = process.communicate(timeout=2)
        except subprocess.TimeoutExpired:
            process.kill()
            _, browser_stderr = process.communicate(timeout=2)
    finally:
        if process is not None and process.poll() is None:
            process.kill()
            process.wait(timeout=2)
        server.shutdown()
        server.server_close()
        server_thread.join(timeout=2)

    expected_frames = [
        (scenario, frame) for scenario in scenarios for frame in scenario["frames"]
    ]
    assert len(browser_results) == len(expected_frames), browser_stderr[-4000:]
    for result, (scenario, frame) in zip(browser_results, expected_frames, strict=True):
        projection = frame["trace"]["projection"]
        assert result["case"] == scenario["case"]["id"]
        assert result["step"] == frame["index"]
        assert result["label"] == frame["label"]
        assert result["projectionStatus"] == "ok"
        assert result["layout"] == {
            "clientWidth": result["layout"]["clientWidth"],
            "scrollWidth": result["layout"]["clientWidth"],
            "fitsWidth": True,
            "rightReachable": True,
        }
        assert result["nodes"] == {
            key: value
            for key, value in projection["nodes"].items()
            if key in result["nodes"]
        }
        expected_paths = {
            **projection["paths"],
            **{
                f"reasoning-output-{mode}": state
                for mode, state in projection["reasoning"]["options"].items()
            },
        }
        assert {
            key: path["state"] for key, path in result["paths"].items()
        } == expected_paths
        for key, path in result["paths"].items():
            if expected_paths[key] in {"pending", "skipped"}:
                continue
            assert path["dash"] in {"none", ""}
            assert path["length"] > 0
        for lane in dashboard._DECISION_TRACE_ROLES:
            expected_rails = projection["lanes"][lane]["rails"]
            assert {
                rail["key"]: rail["state"] for rail in result["rails"][lane]
            } == expected_rails
            for rail in result["rails"][lane]:
                if rail["state"] in {"pending", "skipped"}:
                    continue
                assert rail["dash"] in {"none", ""}
                assert rail["length"] > 0

        selected_context = projection["context"]["selected_tokens"]
        selected_reasoning = projection["reasoning"]["selected"]
        assert result["selectedContext"] == (
            str(selected_context) if selected_context is not None else None
        )
        assert result["selectedReasoning"] == selected_reasoning
        assert result["contextCoreOpacities"].count("1") == int(
            selected_context is not None
        )
        assert result["reasoningCoreOpacities"].count("1") == int(
            selected_reasoning is not None
        )


def test_all_decision_inputs_keep_real_dashboard_paths_connected(
    monkeypatch, tmp_path: Path
) -> None:
    cases = [
        {**case, "lane_context_tokens": [131_072, None, None]}
        if case["id"] == "A"
        else case
        for case in json.loads(FIXTURE.read_text(encoding="utf-8"))["cases"]
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
    roles = {
        "ingest": "ingest_reconciliation",
        "recall": "recall_auto_apply",
        "audit": "content_correction_classification",
        "improve": "model_eval",
        "repair": "local_repair",
        "typed_graph": "relation_extract",
    }
    models = {
        "primary": "qwen3.8:27b-axq4",
        "challenger": "challenger:model",
        "tie_break": "tie:model",
    }
    monkeypatch.setattr(dashboard, "_decision_trace_models", lambda: dict(models))

    payload = []
    for case_index, case in enumerate(cases):
        if case["id"] == "H":
            trace = dashboard._decision_trace_snapshot([], [], None)
            assert trace["state"] == "idle"
            assert [lane["state"] for lane in trace["lanes"]] == case["lane_states"]
            payload.append({"case": {**case, "kind": "decision"}, "trace": trace})
            continue
        request = str(case_index + 1) * 64
        role = roles[case["workflow"]]
        activities = []
        history = []
        active_phases = {"A": "generate", "B": "validate", "D": "vote"}
        for lane_index, lane in enumerate(dashboard._DECISION_TRACE_ROLES):
            state = case["lane_states"][lane_index]
            tokens = case["lane_context_tokens"][lane_index]
            think = case["lane_think"][lane_index]
            observed = {
                "request_sha256": request,
                "role": f"{role}:{lane}",
                "model": models[lane],
                "think": False if think == "off" else think,
                "required_context_tokens": tokens,
                "requested_context_tokens": tokens,
                "context_tokens": tokens,
            }
            timestamp = f"2026-08-15T00:00:0{lane_index + 1}Z"
            if state == "active":
                activities.append(
                    {
                        **observed,
                        "phase": active_phases[case["id"]],
                        "attempt": 0,
                        "elapsed_seconds": case["elapsed_seconds"],
                        "started_at": timestamp,
                        "updated_at": timestamp,
                    }
                )
            elif state in {"done", "error"}:
                history.append(
                    {
                        **observed,
                        "kind": "session",
                        "timestamp": timestamp,
                        "ok": state == "done",
                        "first_pass_valid": state == "done",
                        "repair_turns": 0,
                    }
                )

        decision = None
        if case["state"] in {"agreed", "quarantined"}:
            tie_used = case["branch"] in {"tie-break-agreement", "no-safe-quorum"}
            decision = {
                "kind": "decision",
                "timestamp": "2026-08-15T00:00:09Z",
                "request_sha256": request,
                "role": role,
                "status": case["state"],
                "pair_agreement": case["branch"] == "pair-agreement",
                "tie_break_used": tie_used,
                "vote_count": 3 if tie_used else 2,
                "valid_votes": 3 if tie_used else 2,
                "vote_roles": list(models) if tie_used else ["primary", "challenger"],
                "models": list(models.values()),
            }
            if case["state"] == "quarantined":
                if case["id"] == "G":
                    decision.update(
                        {
                            "artifact_expected": True,
                            "quarantine_reason": (
                                "canonical_decision_artifact_publish_failed"
                            ),
                            "failure_class": "decision_artifact_invalid",
                        }
                    )
                    history.append(
                        {
                            "kind": "decision_artifact",
                            "timestamp": "2026-08-15T00:00:10Z",
                            "request_sha256": request,
                            "artifact_status": "error",
                            "quarantine_reason": (
                                "canonical_decision_artifact_publish_failed"
                            ),
                            "failure_class": "decision_artifact_invalid",
                        }
                    )
                else:
                    decision["quarantine_reason"] = (
                        "local_models_did_not_reach_two_vote_quorum"
                    )
            history.append(decision)

        trace = dashboard._decision_trace_snapshot(
            activities,
            history,
            decision,
            preferred_request_sha256=request,
        )
        assert trace["state"] == case["state"]
        assert [lane["state"] for lane in trace["lanes"]] == case["lane_states"]
        assert [
            None if lane["think"] == "—" else lane["think"] for lane in trace["lanes"]
        ] == case["lane_think"]
        assert [lane["context_tokens"] for lane in trace["lanes"]] == case[
            "lane_context_tokens"
        ]
        payload.append({"case": {**case, "kind": "decision"}, "trace": trace})

    single_request = "9" * 64
    single_case = {
        "id": "S",
        "workflow": "ingest",
        "processing_stage": "consensus",
        "state": "agreed",
        "kind": "decision",
    }
    single_decision = {
        "kind": "decision",
        "timestamp": "2026-08-15T00:02:02Z",
        "request_sha256": single_request,
        "role": "ingest_reconciliation",
        "status": "agreed",
        "authority_kind": "single_model_v1",
        "authority_model": "Qwen3.8-Flash-Next-oQ4e-mtp",
        "authority_revision": "b" * 64,
        "quorum_flow": False,
        "vote_count": 1,
        "valid_votes": 1,
        "artifact_expected": True,
    }
    single_trace = dashboard._decision_trace_snapshot(
        [],
        [
            {
                "kind": "session",
                "timestamp": "2026-08-15T00:02:01Z",
                "request_sha256": single_request,
                "role": "ingest_reconciliation:authority",
                "model": "Qwen3.8-Flash-Next-oQ4e-mtp",
                "revision": "b" * 64,
                "think": False,
                "context_tokens": 65_536,
                "required_context_tokens": 34_028,
                "requested_context_tokens": 65_536,
                "ok": True,
                "first_pass_valid": True,
                "repair_turns": 0,
            },
            single_decision,
            {
                "kind": "decision_artifact",
                "timestamp": "2026-08-15T00:02:03Z",
                "request_sha256": single_request,
                "artifact_status": "sealed",
            },
        ],
        single_decision,
    )
    payload.append({"case": single_case, "trace": single_trace})
    ingest_visual_cases = [
        {
            "name": "target",
            "status": {
                "state": "running",
                "stage": "target-resolution",
                "current_job_id": "job",
                "host_phase": {"name": "target-resolution", "state": "active"},
            },
            "trace": {"task_role": "ingest_triage", "state": "ready"},
        },
        {
            "name": "generate",
            "status": {
                "state": "running",
                "stage": "generate",
                "current_job_id": "job",
                "llm": {"active": True, "event": "start", "phase": "generate"},
            },
            "trace": {"task_role": "ingest_recall_metadata", "state": "ready"},
        },
        {
            "name": "authority",
            "status": {
                "state": "running",
                "stage": "authorization",
                "current_job_id": "job",
            },
            "trace": {
                "task_role": "ingest_reconciliation:authority",
                "state": "active",
            },
        },
        {
            "name": "route",
            "status": {
                "state": "running",
                "stage": "authorization",
                "current_job_id": "job",
            },
            "trace": {
                "task_role": "ingest_reconciliation:authority",
                "state": "ready",
            },
        },
        {
            "name": "retry",
            "status": {
                "state": "running",
                "stage": "local-regenerate",
                "current_job_id": "job",
            },
            "trace": {
                "task_role": "ingest_reconciliation:authority",
                "state": "ready",
            },
        },
        {
            "name": "apply",
            "status": {
                "state": "running",
                "stage": "apply",
                "current_job_id": "job",
                "ingest_disposition": "apply_available",
                "host_phase": {"name": "apply", "state": "active"},
            },
            "trace": {
                "task_role": "ingest_reconciliation:authority",
                "state": "ready",
            },
        },
        {
            "name": "publish",
            "status": {
                "state": "running",
                "stage": "semantic-publish",
                "current_job_id": "job",
                "ingest_disposition": "apply_available",
                "host_phase": {"name": "semantic-publish", "state": "active"},
            },
            "trace": {
                "task_role": "ingest_reconciliation:authority",
                "state": "ready",
            },
        },
        {
            "name": "readback-apply",
            "status": {
                "state": "running",
                "stage": "read-back",
                "current_job_id": "job",
                "ingest_disposition": "apply_available",
                "host_phase": {"name": "read-back", "state": "active"},
            },
            "trace": {
                "task_role": "ingest_reconciliation:authority",
                "state": "ready",
            },
        },
        {
            "name": "readback-noop",
            "status": {
                "state": "running",
                "stage": "read-back",
                "current_job_id": "job",
                "ingest_disposition": "confirmed_noop",
                "host_phase": {"name": "read-back", "state": "active"},
            },
            "trace": {
                "task_role": "ingest_reconciliation:authority",
                "state": "ready",
            },
        },
        {
            "name": "complete-apply",
            "status": {
                "state": "idle",
                "stage": "idle",
                "last_success": {
                    "job_id": "job",
                    "local_consensus_status": "apply_available",
                },
            },
            "trace": {
                "task_role": "ingest_reconciliation:authority",
                "state": "ready",
            },
        },
        {
            "name": "complete-noop",
            "status": {
                "state": "idle",
                "stage": "idle",
                "last_success": {
                    "job_id": "job",
                    "local_consensus_status": "confirmed_noop",
                },
            },
            "trace": {
                "task_role": "ingest_reconciliation:authority",
                "state": "ready",
            },
        },
        {
            "name": "hold",
            "status": {
                "state": "error",
                "stage": "failed",
                "current_job_id": "job",
            },
            "trace": {
                "task_role": "ingest_reconciliation:authority",
                "state": "ready",
            },
        },
    ]

    harness = """
const fixtures = __FIXTURES__;
const ingestVisualCases = __INGEST_VISUAL_CASES__;
const workflowKeys = [...new Set(fixtures.map(({ case: fixture }) => fixture.workflow).filter(Boolean))];
const selectedPipelines = [];
function browserFailure(detail) {
  fetch("/fixture-error", {
    method: "POST",
    headers: { "Content-Type": "text/plain" },
    body: String(detail).slice(0, 2000),
  }).catch(() => {});
}
addEventListener("error", (event) => browserFailure(event.error?.stack || event.message));
addEventListener("unhandledrejection", (event) => browserFailure(event.reason?.stack || event.reason));
addEventListener("chronovisor:processing-lane-select", (event) => {
  selectedPipelines.push(event.detail?.pipeline || "");
});
addEventListener("DOMContentLoaded", () => {
  const api = window.__chronovisorDashboardTest;
  const routeStates = new Set(["active", "done", "error"]);
  const visible = (node) => {
    const style = getComputedStyle(node);
    return style.display !== "none" && style.visibility !== "hidden";
  };
  const screenPoint = (node, point) => point.matrixTransform(node.getScreenCTM());
  const distance = (a, b) => Math.hypot(a.x - b.x, a.y - b.y);
  const segmentDistance = (point, start, end) => {
    const dx = end.x - start.x;
    const dy = end.y - start.y;
    const lengthSquared = dx * dx + dy * dy;
    const t = lengthSquared
      ? Math.max(0, Math.min(1, ((point.x - start.x) * dx + (point.y - start.y) * dy) / lengthSquared))
      : 0;
    return distance(point, { x: start.x + t * dx, y: start.y + t * dy });
  };
  const shapeDistance = (point, group) => {
    const shape = group.querySelector(":scope > circle, :scope > path");
    if (!shape) return Infinity;
    const bounds = shape.getBoundingClientRect();
    if (shape.tagName.toLowerCase() === "circle") {
      const rx = bounds.width / 2;
      const ry = bounds.height / 2;
      const radial = Math.hypot(
        (point.x - (bounds.left + rx)) / rx,
        (point.y - (bounds.top + ry)) / ry
      );
      return Math.abs(radial - 1) * Math.max(rx, ry);
    }
    const midX = bounds.left + bounds.width / 2;
    const midY = bounds.top + bounds.height / 2;
    const vertices = [
      { x: midX, y: bounds.top },
      { x: bounds.right, y: midY },
      { x: midX, y: bounds.bottom },
      { x: bounds.left, y: midY },
      { x: midX, y: bounds.top },
    ];
    return Math.min(...vertices.slice(0, -1).map(
      (vertex, index) => segmentDistance(point, vertex, vertices[index + 1])
    ));
  };
    const disconnectedPathEnds = () => {
      const paths = [...document.querySelectorAll(
        "[data-path-key], [data-reasoning-output], [data-lane-path], [data-repair-lane], "
        + "[data-ingest-job-to], [data-ingest-job-entry]"
      )].filter((node) => visible(node) && routeStates.has(node.dataset.state));
    const ends = paths.flatMap((path, pathIndex) => {
      const length = path.getTotalLength();
      const lane = path.closest("[data-decision-lane]")?.dataset.decisionLane;
      const key = path.dataset.pathKey || path.dataset.reasoningOutput
        || (path.dataset.ingestJobEntry ? `ingest-entry:${path.dataset.ingestJobEntry}` : "")
        || (path.dataset.ingestJobTo ? `ingest:${path.dataset.ingestJobTo}` : "")
        || `${lane}:${path.dataset.lanePath || path.dataset.repairLane}`;
      return [0, length].map((offset, endIndex) => ({
        key,
        pathIndex,
        end: endIndex ? "end" : "start",
        point: screenPoint(path, path.getPointAtLength(offset)),
      }));
    });
    const nodes = [...document.querySelectorAll(
      "[data-trace-key], [data-overall-key], [data-plan-key], [data-context-option], "
        + "[data-reasoning-key], [data-decision-lane-step], [data-ingest-job-step]"
    )].filter((node) => visible(node) && routeStates.has(node.dataset.state));
    return ends.filter((entry) => (
      !nodes.some((node) => shapeDistance(entry.point, node) <= 2.5)
      && !ends.some((other) => (
        other.pathIndex !== entry.pathIndex && distance(entry.point, other.point) <= 1.5
      ))
    )).map(({ key, end }) => `${key}:${end}`);
  };
  const renderFixture = (fixture, trace, revision) => {
    api.renderProcessingActivity({
      generated_at: `2026-08-15T00:01:${revision}Z`,
      revision,
      active_count: trace.active ? 1 : 0,
      lanes: workflowKeys.map((key) => ({
        key,
        label: key,
        state: key === fixture.workflow && trace.active ? "active" : "idle",
        current_step: key === fixture.workflow ? fixture.processing_stage : null,
        steps: [],
      })),
    });
    api.renderDecisionTraceFrame(trace);
  };
  const requestedParams = new URLSearchParams(location.search);
  const requestedFixture = requestedParams.get("fixture");
  if (requestedFixture) {
    const selected = fixtures.find(({ case: fixture }) => fixture.id === requestedFixture);
    if (!selected) throw new Error(`unknown fixture: ${requestedFixture}`);
    renderFixture(selected.case, selected.trace, "screenshot");
    const requestedIngestState = requestedParams.get("ingest-state");
    if (requestedIngestState) {
      const ingestCase = ingestVisualCases.find(({ name }) => name === requestedIngestState);
      if (!ingestCase) throw new Error(`unknown ingest state: ${requestedIngestState}`);
      api.updateIngestJobTrace(
        { ...selected.trace, ...ingestCase.trace },
        ingestCase.status,
        true,
      );
    }
    document.querySelector("#decision-trace-panel")?.scrollIntoView({ block: "start" });
    document.documentElement.dataset.fixtureReady = requestedIngestState
      ? `${requestedFixture}:${requestedIngestState}`
      : requestedFixture;
    return;
  }
  const results = fixtures.map(({ case: fixture, trace }, index) => {
    const selectionStart = selectedPipelines.length;
    renderFixture(fixture, trace, `0${index}`);
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
    const singleArtifact = document.querySelector('[data-path-key="single-artifact"]');
    const artifact = document.querySelector('[data-trace-key="artifact"]');
    const singleArtifactEnd = screenPoint(
      singleArtifact,
      singleArtifact.getPointAtLength(singleArtifact.getTotalLength())
    );
    const artifactLeft = screenPoint(artifact, new DOMPoint(-11, 0));
    const shapeCenter = (node) => {
      const bounds = node.querySelector("circle, path").getBoundingClientRect();
      return { x: bounds.left + bounds.width / 2, y: bounds.top + bounds.height / 2 };
    };
    const terminalCenters = [
      document.querySelector('[data-decision-lane="primary"] [data-decision-lane-step="validate"]'),
      artifact,
      document.querySelector('[data-trace-key="seal"]'),
    ].map(shapeCenter);
    const dispatch = document.querySelector('[data-path-key="plan-dispatch"]');
    const dispatchRail = screenPoint(
      dispatch,
      dispatch.getPointAtLength(dispatch.getTotalLength() / 2)
    );
    const planBottom = screenPoint(harness, new DOMPoint(0, 210));
    const holdCenter = shapeCenter(document.querySelector('[data-trace-key="hold"]'));
    const ingestEntry = document.querySelector("[data-ingest-job-entry]");
    const ingestEntryStart = ingestEntry
      ? screenPoint(ingestEntry, ingestEntry.getPointAtLength(0))
      : null;
    const decisionNode = document.querySelector('[data-trace-key="decision"]');
    const decisionCenter = shapeCenter(decisionNode);
    const mutationGate = document.querySelector('g[data-ingest-job-step="mutation"]');
    const acceptedPath = document.querySelector('[data-ingest-job-path="accepted"]');
    const applyPath = document.querySelector('[data-ingest-job-path="apply"]');
    const noopPath = document.querySelector('[data-ingest-job-path="noop"]');
    const ingestMutationGeometry = fixture.id === "S" ? (() => {
      const applyStart = screenPoint(applyPath, applyPath.getPointAtLength(0));
      const noopStart = screenPoint(noopPath, noopPath.getPointAtLength(0));
      const acceptedEnd = screenPoint(
        acceptedPath,
        acceptedPath.getPointAtLength(acceptedPath.getTotalLength())
      );
      return {
        branchStartSeparation: distance(applyStart, noopStart),
        applyStartDelta: shapeDistance(applyStart, mutationGate),
        noopStartDelta: shapeDistance(noopStart, mutationGate),
        acceptedEndDelta: shapeDistance(acceptedEnd, mutationGate),
      };
    })() : null;
    const ingestGridGeometry = fixture.id === "S" ? (() => {
      const centers = (keys) => keys.map((key) => shapeCenter(
        document.querySelector(`g[data-ingest-job-step="${key}"]`)
      ));
      const upper = centers(["hold", "route", "authority", "generate", "target"]);
      const lower = centers(["mutation", "apply", "publish", "readback", "complete"]);
      const gapSpread = (points) => {
        const gaps = points.slice(1).map((point, index) => point.x - points[index].x);
        return Math.max(...gaps) - Math.min(...gaps);
      };
      return {
        upperGapSpread: gapSpread(upper),
        lowerGapSpread: gapSpread(lower),
        upperRailSpread: Math.max(...upper.map(({ y }) => y)) - Math.min(...upper.map(({ y }) => y)),
        lowerRailSpread: Math.max(...lower.map(({ y }) => y)) - Math.min(...lower.map(({ y }) => y)),
        decisionColumnDelta: Math.abs(upper[1].x - lower[0].x),
      };
    })() : null;
    const opacity = (node) => node ? getComputedStyle(node).opacity : null;
    scroller.scrollLeft = scroller.scrollWidth;
    const scrollerBounds = scroller.getBoundingClientRect();
    const harnessBounds = harness.getBoundingClientRect();
    const decisionBounds = document.querySelector("#decision-trace-panel").getBoundingClientRect();
    const saveHistoryBounds = document.querySelector("#save-history-panel").getBoundingClientRect();
    const scrollerRight = scrollerBounds.right;
    const harnessRight = harnessBounds.right;
    const layout = {
      clientWidth: scroller.clientWidth,
      scrollWidth: scroller.scrollWidth,
      scrollLeft: scroller.scrollLeft,
      rightReachable: harnessRight <= scrollerRight + 1,
      fitsWidth: Math.abs(harnessBounds.width - scrollerBounds.width) <= 1,
      nextPanelGap: saveHistoryBounds.top - decisionBounds.bottom,
      generationMeterHeight: document.querySelector("#decision-generation-meter")
        .getBoundingClientRect().height,
    };
    const paths = Object.fromEntries([
      ...[...document.querySelectorAll("[data-path-key]")]
        .map((node) => [node.dataset.pathKey, node]),
      ...[...document.querySelectorAll("[data-reasoning-output]")]
        .map((node) => [`reasoning-output-${node.dataset.reasoningOutput}`, node]),
    ].map(([key, node]) => [key, pathState(node)]));
    const rails = Object.fromEntries([...document.querySelectorAll("[data-decision-lane]")]
      .map((lane) => [lane.dataset.decisionLane, [...lane.querySelectorAll("[data-lane-path]")]
        .map((node) => ({ key: node.dataset.lanePath, ...pathState(node) }))]));
    scroller.scrollLeft = 0;
    const ingestBranchStates = fixture.id === "S" ? ingestVisualCases.map((item) => {
      const branchTrace = { ...trace, ...item.trace };
      const { name, status } = item;
      api.updateIngestJobTrace(branchTrace, status, true);
      return {
        name,
        branch: harness.dataset.ingestJobBranch,
        current: harness.dataset.ingestJobStep,
        disconnectedPathEnds: disconnectedPathEnds(),
      };
    }) : [];
    renderFixture(fixture, trace, `restore-${index}`);
    return {
      id: fixture.id,
      kind: fixture.kind,
      primaryPhase: trace.lanes.find((lane) => lane.key === "primary")?.phase,
      layout,
      projectionStatus: harness.dataset.projectionStatus,
      selectionEvent: selectedPipelines.length > selectionStart ? selectedPipelines.at(-1) : null,
      selected: [...document.querySelectorAll('[data-processing-lane][aria-selected="true"]')]
        .map((node) => node.dataset.processingLane),
      context: selectedContext?.dataset.contextTokens ?? null,
      contextState: selectedContext?.dataset.state ?? null,
      contextClass: selectedContext?.getAttribute("class") ?? null,
      contextCore: opacity(selectedContext?.querySelector(".trace-selected-core")),
      contextCoreOpacities: [...document.querySelectorAll(
        "[data-context-option] .trace-selected-core"
      )].map(opacity),
      reasoning: selectedReasoning?.dataset.reasoningKey ?? null,
      reasoningState: selectedReasoning?.dataset.state ?? null,
      reasoningClass: selectedReasoning?.getAttribute("class") ?? null,
      reasoningCore: opacity(selectedReasoning?.querySelector(".trace-selected-core")),
      reasoningCoreOpacities: [...document.querySelectorAll(
        "[data-reasoning-key] .trace-selected-core"
      )].map(opacity),
      reasoningLabel: document.querySelector(
        '[data-reasoning-key="medium"] [data-reasoning-label]'
      )?.textContent,
      fitLabel: document.querySelector('[data-plan-value="fit"]')?.textContent,
      ingestBranchStates,
      disconnectedPathEnds: disconnectedPathEnds(),
      singleArtifactDelta: {
        x: Math.abs(artifactLeft.x - singleArtifactEnd.x),
        y: Math.abs(artifactLeft.y - singleArtifactEnd.y),
      },
      singleLayout: {
        viewBox: harness.getAttribute("viewBox"),
        height: harness.getAttribute("height"),
        dispatchLabelY: document.querySelector("[data-dispatch-label]")?.getAttribute("y"),
        dispatchCenterDeltaY: Math.abs(
          dispatchRail.y - ((planBottom.y + terminalCenters[0].y) / 2)
        ),
        gaps: terminalCenters.slice(1).map((point, index) => (
          point.x - terminalCenters[index].x
        )),
        railSpread: Math.max(...terminalCenters.map(({ y }) => y))
          - Math.min(...terminalCenters.map(({ y }) => y)),
        decisionSealDeltaX: Math.abs(decisionCenter.x - terminalCenters.at(-1).x),
        decisionSealDrop: decisionCenter.y - terminalCenters.at(-1).y,
        holdSealOffset: holdCenter.x - terminalCenters.at(-1).x,
        holdSealDeltaY: Math.abs(holdCenter.y - terminalCenters.at(-1).y),
        ingestEntryStartDelta: ingestEntryStart
          ? shapeDistance(ingestEntryStart, decisionNode)
          : null,
        ingestMutationGeometry,
        ingestGridGeometry,
        ingestLaneLabels: [...document.querySelectorAll(
          ".trace-ingest-job .decision-role, .trace-ingest-job .decision-sublabel"
        )].filter(visible).map((node) => node.textContent.trim()),
        ingestStepLabels: [...document.querySelectorAll(
          ".trace-ingest-job [data-ingest-job-step] > text"
        )].filter(visible).map((node) => node.textContent.trim()),
        ingestBranchLabels: [...document.querySelectorAll(
          ".trace-ingest-job [data-ingest-branch-label]"
        )].filter(visible).map((node) => node.textContent.trim()),
      },
      nodes: Object.fromEntries([...document.querySelectorAll("[data-trace-key]")]
        .map((node) => [node.dataset.traceKey, node.dataset.state])),
      paths,
      rails,
    };
  });
  renderFixture(fixtures[0].case, fixtures[0].trace, "59");
  results[0].restoredLayout = {
    viewBox: document.querySelector("#decision-trace-harness").getAttribute("viewBox"),
    height: document.querySelector("#decision-trace-harness").getAttribute("height"),
    dispatchLabelY: document.querySelector("[data-dispatch-label]").getAttribute("y"),
  };
  results[0].tabClicks = workflowKeys.map((pipeline) => {
    document.querySelector(`[data-processing-lane="${pipeline}"]`).click();
    return {
      event: selectedPipelines.at(-1),
      selected: document.querySelector('[data-processing-lane][aria-selected="true"]')
        ?.dataset.processingLane,
      tabCount: document.querySelectorAll('[data-processing-lane][role="tab"]').length,
      panelRole: document.querySelector("#decision-trace-panel")?.getAttribute("role"),
    };
  });
  document.querySelector(
    `[data-processing-lane="${fixtures[0].case.workflow}"]`
  ).click();
  fetch("/fixture-result", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(results),
  }).catch(browserFailure);
});
""".replace("__FIXTURES__", json.dumps(payload, ensure_ascii=False)).replace(
        "__INGEST_VISUAL_CASES__",
        json.dumps(ingest_visual_cases, ensure_ascii=False),
    )
    page = (
        (dashboard.STATIC_DIR / "index.html")
        .read_text(encoding="utf-8")
        .replace(
            '<script src="/static/app-client.js"></script>',
            '<script src="/fixture-harness.js"></script>',
        )
    )
    page_path = tmp_path / "index.html"
    harness_path = tmp_path / "fixture-harness.js"
    screenshot_path = tmp_path / "eight-case-dashboard.png"
    page_path.write_text(page, encoding="utf-8")
    harness_path.write_text(harness, encoding="utf-8")
    result_ready = threading.Event()
    browser_results = []
    browser_errors = []

    class Handler(dashboard.DashboardHandler):
        def do_GET(self) -> None:
            request_path = self.path.split("?", 1)[0]
            if request_path in {"/", "/fixture-harness.js"}:
                if self._browser_boundary_allows():
                    dashboard._file_response(
                        self,
                        page_path if request_path == "/" else harness_path,
                    )
                return
            super().do_GET()

        def do_POST(self) -> None:
            if self.path not in {"/fixture-error", "/fixture-result"}:
                self.send_error(404)
                return
            if not self._browser_boundary_allows():
                return
            length = int(self.headers.get("Content-Length", "0"))
            if not 0 < length <= 262_144:
                self.send_error(400)
                return
            body = self.rfile.read(length)
            if self.path == "/fixture-error":
                browser_errors.append(body.decode(errors="replace"))
            else:
                browser_results.extend(json.loads(body))
                result_ready.set()
            self.send_response(204)
            self.end_headers()

        def log_message(self, _format: str, *_args: object) -> None:
            return

    chrome = next(
        (
            candidate
            for candidate in (
                shutil.which("google-chrome"),
                shutil.which("chromium"),
                shutil.which("chromium-browser"),
                "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
            )
            if candidate and Path(candidate).is_file()
        ),
        None,
    )
    assert chrome is not None, "headless Chrome is required for Dashboard DOM tests"

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    server_thread = threading.Thread(target=server.serve_forever, daemon=True)
    server_thread.start()
    process = None
    browser_exit = None
    received = False
    visual_paths = []
    try:
        process = subprocess.Popen(
            [
                chrome,
                "--headless=new",
                "--no-sandbox",
                "--disable-dev-shm-usage",
                "--disable-gpu",
                "--disable-background-networking",
                "--disable-component-update",
                "--no-first-run",
                "--force-prefers-reduced-motion=reduce",
                "--window-size=1050,992",
                f"--screenshot={screenshot_path}",
                f"--user-data-dir={tmp_path / 'chrome-profile'}",
                f"http://127.0.0.1:{server.server_port}/",
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            text=True,
        )
        received = result_ready.wait(60)
        if received and process.poll() is None:
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                pass
        browser_exit = process.poll()
        if received:
            visual_dir = Path(
                os.environ.get(
                    "CHRONOVISOR_DASHBOARD_VISUAL_DIR",
                    tmp_path / "decision-trace-visuals",
                )
            )
            visual_dir.mkdir(parents=True, exist_ok=True)
            visual_requests = [
                (item["case"]["id"], None) for item in payload
            ] + [("S", item["name"]) for item in ingest_visual_cases]
            for index, (case_id, ingest_state) in enumerate(visual_requests):
                stem = case_id.replace("::", "__")
                if ingest_state:
                    stem += f"__{ingest_state}"
                visual_path = visual_dir / f"{stem}.png"
                query = f"fixture={quote(case_id)}"
                if ingest_state:
                    query += f"&ingest-state={quote(ingest_state)}"
                visual_process = subprocess.Popen(
                    [
                        chrome,
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
                        "--window-size=1280,1400",
                        f"--screenshot={visual_path}",
                        f"--user-data-dir={tmp_path / f'visual-profile-{index}'}",
                        f"http://127.0.0.1:{server.server_port}/?{query}",
                    ],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.PIPE,
                    text=True,
                    start_new_session=True,
                )
                deadline = time.monotonic() + 10
                while time.monotonic() < deadline:
                    if visual_path.is_file() and visual_path.stat().st_size > 0:
                        break
                    if visual_process.poll() not in (None, 0):
                        break
                    time.sleep(0.05)
                try:
                    os.killpg(visual_process.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
                try:
                    _, visual_stderr = visual_process.communicate(timeout=2)
                except subprocess.TimeoutExpired:
                    try:
                        os.killpg(visual_process.pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                    try:
                        _, visual_stderr = visual_process.communicate(timeout=2)
                    except subprocess.TimeoutExpired:
                        visual_process.kill()
                        _, visual_stderr = visual_process.communicate(timeout=2)
                assert visual_path.is_file() and visual_path.stat().st_size > 0, (
                    f"chrome returncode={visual_process.returncode}\n"
                    f"{visual_stderr[-4000:]}"
                )
                assert struct.unpack(">II", visual_path.read_bytes()[16:24]) == (
                    1280,
                    1400,
                )
                visual_paths.append(visual_path)
    finally:
        if process is not None:
            process.terminate()
            try:
                process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=2)
        server.shutdown()
        server.server_close()
        server_thread.join(timeout=2)

    stderr = process.stderr.read()[-4000:] if process and process.stderr else ""
    diagnostics = json.dumps(
        {
            "browser_exit": browser_exit,
            "browser_stderr": stderr,
            "browser_errors": browser_errors,
            "browser_results": browser_results,
        },
        indent=2,
    )
    assert received, diagnostics
    assert not browser_errors, diagnostics
    assert len(browser_results) == len(payload), diagnostics
    assert screenshot_path.is_file() and screenshot_path.stat().st_size > 0

    assert len(visual_paths) == len(payload) + len(ingest_visual_cases)

    by_id = {result["id"]: result for result in browser_results}
    by_trace = {item["case"]["id"]: item["trace"] for item in payload}
    for case in cases:
        result = by_id[case["id"]]
        trace = by_trace[case["id"]]
        assert result["projectionStatus"] == "ok"
        assert result["selected"] == [case["workflow"] or "ingest"]
        assert result["selectionEvent"] is None
        assert result["layout"]["scrollWidth"] == result["layout"]["clientWidth"]
        assert result["layout"]["rightReachable"] is True
        assert result["layout"]["fitsWidth"] is True
        assert result["layout"]["nextPanelGap"] == 12
        assert result["layout"]["generationMeterHeight"] == 32
        assert result["singleLayout"]["viewBox"] == "0 0 1500 650"
        assert result["singleLayout"]["height"] == "607"
        assert result["disconnectedPathEnds"] == [], case["id"]
        projection = trace["projection"]
        assert result["nodes"] == {
            key: value
            for key, value in projection["nodes"].items()
            if key in result["nodes"]
        }
        expected_paths = {
            **projection["paths"],
            **{
                f"reasoning-output-{mode}": state
                for mode, state in projection["reasoning"]["options"].items()
            },
        }
        assert {
            key: path["state"] for key, path in result["paths"].items()
        } == expected_paths
        for path_key, state in expected_paths.items():
            if state in {"pending", "skipped"}:
                continue
            path = result["paths"][path_key]
            assert path["state"] in {"active", "done", "error"}
            assert path["dash"] in {"none", ""}
            assert path["length"] > 0

        for lane in dashboard._DECISION_TRACE_ROLES:
            expected_rails = projection["lanes"][lane]["rails"]
            assert {
                rail["key"]: rail["state"] for rail in result["rails"][lane]
            } == expected_rails
            for rail in result["rails"][lane]:
                if rail["state"] in {"pending", "skipped"}:
                    continue
                assert rail["state"] in {"active", "done", "error"}
                assert rail["dash"] in {"none", ""}
                assert rail["length"] > 0

        selected_tokens = projection["context"]["selected_tokens"]
        selected_reasoning = projection["reasoning"]["selected"]
        assert result["context"] == (
            str(selected_tokens) if selected_tokens is not None else None
        )
        assert result["reasoning"] == selected_reasoning
        assert result["contextCore"] == ("1" if selected_tokens else None), (
            case["id"],
            result["contextState"],
            result["contextClass"],
            projection["context"],
        )
        assert result["reasoningCore"] == ("1" if selected_reasoning else None), (
            case["id"],
            result["reasoningState"],
            result["reasoningClass"],
            projection["reasoning"],
        )
        assert result["contextCoreOpacities"].count("1") == int(
            selected_tokens is not None
        )
        assert result["reasoningCoreOpacities"].count("1") == int(
            selected_reasoning is not None
        )

    single_result = by_id[single_case["id"]]
    assert single_result["projectionStatus"] == "ok"
    assert single_result["context"] == "65536"
    assert single_result["reasoning"] == "off"
    assert single_result["disconnectedPathEnds"] == []
    assert single_result["singleArtifactDelta"]["x"] <= 0.01
    assert single_result["singleArtifactDelta"]["y"] <= 0.01
    assert single_result["singleLayout"]["viewBox"] == "0 0 1500 750"
    assert single_result["singleLayout"]["height"] == "700"
    assert single_result["singleLayout"]["dispatchLabelY"] == "303"
    assert single_result["singleLayout"]["dispatchCenterDeltaY"] <= 0.01
    gaps = single_result["singleLayout"]["gaps"]
    assert max(gaps) - min(gaps) <= 1
    assert single_result["singleLayout"]["railSpread"] <= 1
    assert single_result["singleLayout"]["decisionSealDeltaX"] <= 1
    assert (
        0.6
        <= single_result["singleLayout"]["decisionSealDrop"] / (sum(gaps) / 2)
        <= 0.7
    )
    assert abs(single_result["singleLayout"]["holdSealOffset"] - gaps[-1]) <= 1
    assert single_result["singleLayout"]["holdSealDeltaY"] <= 1
    assert single_result["singleLayout"]["ingestEntryStartDelta"] is not None
    assert single_result["singleLayout"]["ingestEntryStartDelta"] <= 1
    mutation_geometry = single_result["singleLayout"]["ingestMutationGeometry"]
    assert mutation_geometry is not None
    assert mutation_geometry["branchStartSeparation"] > 5
    assert mutation_geometry["applyStartDelta"] <= 1
    assert mutation_geometry["noopStartDelta"] <= 1
    assert mutation_geometry["acceptedEndDelta"] <= 1
    grid_geometry = single_result["singleLayout"]["ingestGridGeometry"]
    assert grid_geometry is not None
    assert grid_geometry["upperGapSpread"] <= 1
    assert grid_geometry["lowerGapSpread"] <= 1
    assert grid_geometry["upperRailSpread"] <= 1
    assert grid_geometry["lowerRailSpread"] <= 1
    assert grid_geometry["decisionColumnDelta"] <= 1
    assert single_result["singleLayout"]["ingestLaneLabels"] == []
    assert "Raw" not in single_result["singleLayout"]["ingestStepLabels"]
    assert "Triage LLM" not in single_result["singleLayout"]["ingestStepLabels"]
    assert set(single_result["singleLayout"]["ingestBranchLabels"]) == {
        "APPLY",
        "NOOP",
        "RETRY",
        "HOLD",
        "ACCEPTED",
    }
    assert single_result["ingestBranchStates"] == [
        {
            "name": "target",
            "branch": "waiting",
            "current": "target",
            "disconnectedPathEnds": [],
        },
        {
            "name": "generate",
            "branch": "waiting",
            "current": "generate",
            "disconnectedPathEnds": [],
        },
        {
            "name": "authority",
            "branch": "waiting",
            "current": "authority",
            "disconnectedPathEnds": [],
        },
        {
            "name": "route",
            "branch": "waiting",
            "current": "route",
            "disconnectedPathEnds": [],
        },
        {
            "name": "retry",
            "branch": "retry",
            "current": "generate",
            "disconnectedPathEnds": [],
        },
        {
            "name": "apply",
            "branch": "apply",
            "current": "apply",
            "disconnectedPathEnds": [],
        },
        {
            "name": "publish",
            "branch": "apply",
            "current": "publish",
            "disconnectedPathEnds": [],
        },
        {
            "name": "readback-apply",
            "branch": "apply",
            "current": "readback",
            "disconnectedPathEnds": [],
        },
        {
            "name": "readback-noop",
            "branch": "noop",
            "current": "readback",
            "disconnectedPathEnds": [],
        },
        {
            "name": "complete-apply",
            "branch": "apply",
            "current": "complete",
            "disconnectedPathEnds": [],
        },
        {
            "name": "complete-noop",
            "branch": "noop",
            "current": "complete",
            "disconnectedPathEnds": [],
        },
        {
            "name": "hold",
            "branch": "hold",
            "current": "hold",
            "disconnectedPathEnds": [],
        },
    ]

    assert by_id[cases[0]["id"]]["tabClicks"] == [
        {
            "event": pipeline,
            "selected": pipeline,
            "tabCount": 6,
            "panelRole": "tabpanel",
        }
        for pipeline in (
            "ingest",
            "recall",
            "audit",
            "improve",
            "repair",
            "typed_graph",
        )
    ]
    assert by_id[cases[0]["id"]]["restoredLayout"] == {
        "viewBox": "0 0 1500 650",
        "height": "607",
        "dispatchLabelY": "258",
    }
