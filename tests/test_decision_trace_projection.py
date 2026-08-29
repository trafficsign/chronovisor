from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import pytest

from chronovisor.ops import decision_trace_projection as projection

ROOT = Path(__file__).resolve().parents[1]


def _steps(state: str = "done") -> list[dict[str, str]]:
    return [{"key": phase, "status": state} for phase in projection.TRACE_PHASES]


def _lane(
    key: str,
    state: str = "pending",
    *,
    phase: str | None = None,
    think: object = None,
    requested: object = None,
    context: object = None,
    required: object = None,
    repair_turns: object = 0,
    steps: object = None,
) -> dict[str, Any]:
    return {
        "key": key,
        "state": state,
        "phase": phase,
        "think": think,
        "requested_context_tokens": requested,
        "context_tokens": context,
        "required_context_tokens": required,
        "repair_turns": repair_turns,
        "steps": _steps() if steps is None and state == "done" else steps or [],
    }


def _trace(
    *,
    state: str = "active",
    packet: str = "done",
    dispatch: str = "done",
    quorum: str = "pending",
    artifact: str = "pending",
    decision: str = "pending",
    lanes: list[dict[str, Any]] | None = None,
    **extra: Any,
) -> dict[str, Any]:
    return {
        "state": state,
        "overall": [
            {"key": "packet", "status": packet},
            {"key": "dispatch", "status": dispatch},
            {"key": "quorum", "status": quorum},
            {"key": "artifact", "status": artifact},
            {"key": "decision", "status": decision},
        ],
        "lanes": lanes or [_lane("primary"), _lane("challenger"), _lane("tie_break")],
        **extra,
    }


def test_projection_topology_matches_the_svg_contract() -> None:
    page = (ROOT / "src/chronovisor/dashboard_static/index.html").read_text(
        encoding="utf-8"
    )
    assert set(re.findall(r'data-path-key="([^"]+)"', page)) == set(
        projection.TRACE_PATH_KEYS
    )
    assert {
        "pending",
        "active",
        "done",
        "skipped",
        "error",
    } == projection.TRACE_STATES
    assert projection.TRACE_ROLES == ("primary", "challenger", "tie_break")


def test_projection_normalizes_helper_inputs() -> None:
    assert projection._state("active") == "active"
    assert projection._state("unknown", "skipped") == "skipped"
    assert projection._positive_int(1) == 1
    assert all(projection._positive_int(value) is None for value in (True, 0, -1, "1"))
    assert projection._rows_by_key(None) == {}
    assert projection._rows_by_key(
        [
            {"key": "valid", "value": 1},
            {"key": 2},
            {"value": 3},
            "not-a-row",
        ]
    ) == {"valid": {"key": "valid", "value": 1}}
    assert projection._context_label(None) == "—"
    assert projection._selected_path("pending", selected=False) == "pending"
    assert projection._selected_path("pending", selected=True) == "active"
    assert projection._selected_path("done", selected=True) == "done"
    assert projection._context_label(32_768) == "32K"
    assert [row["tokens"] for row in projection._context_options(None, "done")] == [
        32_768,
        65_536,
        98_304,
        131_072,
    ]
    custom = projection._context_options(100_000, "active")
    assert [(row["tokens"], row["state"]) for row in custom] == [
        (32_768, "pending"),
        (65_536, "pending"),
        (100_000, "active"),
        (131_072, "pending"),
    ]


def test_idle_projection_is_a_complete_pending_snapshot() -> None:
    result = projection.project_decision_trace({})

    assert result["schema"] == projection.TRACE_PROJECTION_SCHEMA
    assert set(result["paths"]) == set(projection.TRACE_PATH_KEYS)
    assert set(result["paths"].values()) == {"pending"}
    assert set(result["nodes"].values()) == {"pending"}
    assert result["model_routes"] == {
        "primary": "pending",
        "challenger": "pending",
        "tie_break": "pending",
    }
    assert result["context"]["selected_tokens"] is None
    assert result["reasoning"]["selected"] is None
    assert result["labels"] == {
        "fit": "WAITING",
        "fit_pass": False,
        "hold": "No safe quorum",
    }


def test_active_repair_projects_custom_context_and_reasoning_bypass() -> None:
    result = projection.project_decision_trace(
        _trace(
            dispatch="active",
            lanes=[
                _lane(
                    "primary",
                    "active",
                    phase="repair",
                    think="off",
                    requested=100_000,
                    context=98_304,
                    required=90_000,
                    repair_turns=True,
                    steps=[
                        {"key": "trigger", "status": "done"},
                        {"key": "load", "status": "active"},
                        {"key": "context", "status": "unknown"},
                    ],
                ),
                _lane("challenger"),
                _lane("tie_break"),
            ],
            events=[
                "invalid",
                {"lane": "challenger", "phase": "repair", "attempt": 4},
                {"lane": "primary", "phase": "generate", "attempt": 3},
                {"lane": "primary", "phase": "repair", "attempt": 0},
            ],
        )
    )

    assert result["context"]["selected_tokens"] == 100_000
    assert [row for row in result["context"]["options"] if row["selected"]] == [
        {
            "tokens": 100_000,
            "label": "100K",
            "selected": True,
            "state": "active",
        }
    ]
    assert result["context"]["label"] == "required 90K → selected 100K"
    assert result["reasoning"]["selected"] == "off"
    assert result["paths"]["reasoning-off"] == "active"
    assert result["paths"]["plan-fit"] == "pending"
    assert result["labels"]["fit"] == "BYPASS"
    assert result["lanes"]["primary"]["repair"] == "active"
    assert result["lanes"]["primary"]["repair_attempt"] == 1
    assert result["lanes"]["primary"]["steps"]["context"] == "pending"


def test_pair_agreement_projects_successful_seal_without_tie_break() -> None:
    result = projection.project_decision_trace(
        _trace(
            state="agreed",
            artifact="done",
            decision="done",
            lanes=[
                _lane(
                    "primary", "done", think="medium", context=32_768, repair_turns=2
                ),
                _lane("challenger", "done", think="medium", context=32_768),
                _lane("tie_break", "skipped"),
            ],
            pair_agreement=True,
            quorum_attempted=True,
        )
    )

    assert result["paths"]["pair-artifact-join"] == "done"
    assert result["paths"]["pair-tie_break"] == "pending"
    assert result["paths"]["artifact-seal"] == "done"
    assert result["paths"]["seal-decision"] == "done"
    assert result["nodes"]["agree"] == "done"
    assert result["nodes"]["seal"] == "done"
    assert result["labels"] == {
        "fit": "headroom OK",
        "fit_pass": True,
        "hold": "No safe quorum",
    }
    assert result["lanes"]["primary"]["repair"] == "done"
    assert result["lanes"]["primary"]["repair_attempt"] == 2


def test_tie_break_active_and_completed_paths() -> None:
    active = projection.project_decision_trace(
        _trace(
            lanes=[
                _lane("primary", "done", think="medium", context=32_768),
                _lane("challenger", "done", think="medium", context=32_768),
                _lane("tie_break", "active", think="high", context=65_536),
            ],
            pair_agreement=False,
            quorum_attempted=True,
        )
    )
    completed = projection.project_decision_trace(
        _trace(
            state="agreed",
            quorum="done",
            artifact="done",
            decision="done",
            lanes=[
                _lane("primary", "done", think="medium", context=32_768),
                _lane("challenger", "done", think="medium", context=32_768),
                _lane("tie_break", "done", think="high", context=65_536),
            ],
            pair_agreement=False,
            tie_break_used=True,
            quorum_attempted=True,
        )
    )

    assert active["paths"]["pair-tie_break"] == "active"
    assert active["nodes"]["agree"] == "active"
    assert active["labels"]["fit"] == "headroom OK"
    assert completed["paths"]["pair-tie_break"] == "done"
    assert completed["paths"]["tie_break-quorum"] == "done"
    assert completed["paths"]["quorum-artifact-join"] == "done"
    assert completed["nodes"]["quorum"] == "done"


def test_tie_break_completion_and_agreement_keep_the_selected_route_connected() -> None:
    lanes = [
        _lane("primary", "done", think="medium", context=32_768),
        _lane("challenger", "done", think="medium", context=32_768),
        _lane("tie_break", "done", think="high", context=65_536),
    ]
    evaluating = projection.project_decision_trace(
        _trace(
            state="idle",
            quorum="active",
            lanes=lanes,
            pair_agreement=False,
            quorum_attempted=True,
        )
    )
    agreed = projection.project_decision_trace(
        _trace(
            state="agreed",
            quorum="done",
            lanes=lanes,
            pair_agreement=False,
            tie_break_used=True,
            quorum_attempted=True,
        )
    )

    assert evaluating["paths"]["tie_break-quorum"] == "active"
    assert evaluating["nodes"]["quorum"] == "active"
    assert agreed["paths"]["tie_break-quorum"] == "done"
    assert agreed["paths"]["quorum-artifact-join"] == "active"
    assert agreed["nodes"]["quorum"] == "done"
    assert agreed["nodes"]["artifact"] == "pending"
    assert agreed["nodes"]["seal"] == "pending"
    assert agreed["nodes"]["decision"] == "pending"
    assert agreed["paths"]["artifact-seal"] == "pending"
    assert agreed["paths"]["seal-decision"] == "pending"


def test_pair_agreement_waits_for_the_artifact_before_lighting_the_seal() -> None:
    result = projection.project_decision_trace(
        _trace(
            state="agreed",
            quorum="done",
            lanes=[
                _lane("primary", "done", think="medium", context=32_768),
                _lane("challenger", "done", think="medium", context=32_768),
                _lane("tie_break", "skipped"),
            ],
            pair_agreement=True,
            quorum_attempted=True,
        )
    )

    assert result["paths"]["pair-artifact-join"] == "active"
    assert result["nodes"]["artifact"] == "pending"
    assert result["nodes"]["seal"] == "pending"
    assert result["paths"]["artifact-seal"] == "pending"
    assert result["paths"]["seal-decision"] == "pending"


def test_no_safe_quorum_stops_before_artifact_and_decision() -> None:
    result = projection.project_decision_trace(
        _trace(
            state="quarantined",
            quorum="error",
            artifact="skipped",
            decision="skipped",
            lanes=[
                _lane("primary", "done", think="medium", context=32_768),
                _lane("challenger", "done", think="medium", context=32_768),
                _lane("tie_break", "done", think="medium", context=65_536),
            ],
            pair_agreement=False,
            tie_break_used=True,
            quorum_attempted=True,
            summary="No safe quorum · quarantined",
        )
    )

    assert result["paths"]["tie_break-quorum"] == "error"
    assert result["paths"]["quorum-hold"] == "error"
    assert result["paths"]["artifact-seal"] == "pending"
    assert result["paths"]["seal-decision"] == "pending"
    assert {
        key: result["nodes"][key]
        for key in ("agree", "quorum", "artifact", "decision", "hold")
    } == {
        "agree": "error",
        "quorum": "error",
        "artifact": "skipped",
        "decision": "skipped",
        "hold": "error",
    }


@pytest.mark.parametrize(
    ("overall_artifact", "outcome"),
    [
        ("error", {}),
        ("pending", {"code": "canonical_seal_failure"}),
        ("pending", {"reason": "SEAL unavailable"}),
    ],
)
def test_seal_failure_projects_the_operational_hold(
    overall_artifact: str, outcome: dict[str, str]
) -> None:
    result = projection.project_decision_trace(
        _trace(
            state="quarantined",
            artifact=overall_artifact,
            decision="skipped",
            lanes=[
                _lane("primary", "done", think="medium", context=32_768),
                _lane("challenger", "done", think="medium", context=32_768),
                _lane("tie_break", "skipped"),
            ],
            pair_agreement=True,
            quorum_attempted=True,
            outcome=outcome,
        )
    )

    assert result["paths"]["artifact-seal"] in {"error", "pending"}
    assert result["paths"]["seal-hold"] == "error"
    assert result["nodes"]["artifact"] == "error"
    assert result["nodes"]["seal"] == "error"
    assert result["nodes"]["hold"] == "error"
    assert result["labels"]["hold"] == "Seal failed"


def test_single_model_and_explicitly_unattempted_quorum_paths() -> None:
    standalone = projection.project_decision_trace(
        _trace(
            state="ready",
            artifact="done",
            decision="done",
            lanes=[
                _lane("primary", "done", think="low", context=32_768),
                _lane("challenger", "skipped"),
                _lane("tie_break", "skipped"),
            ],
            quorum_flow=False,
        )
    )
    held_without_quorum = projection.project_decision_trace(
        _trace(
            state="quarantined",
            artifact="skipped",
            decision="skipped",
            lanes=[
                _lane("primary", "broken", think="unknown", context=True),
                _lane("challenger", "done", think="unknown", context=0),
                _lane("tie_break", "skipped"),
                {"key": 1},
                "invalid",
            ],
            quorum_attempted=False,
            context_tokens=65_536,
            outcome=[],
            summary="Custom hold · details",
            events="not-a-list",
        )
    )

    assert standalone["paths"]["single-artifact"] == "done"
    assert standalone["single_model"] is False
    assert list(standalone["lanes"]) == ["primary", "challenger", "tie_break"]
    assert standalone["paths"]["quorum-artifact-join"] == "pending"
    assert standalone["nodes"]["seal"] == "done"
    assert held_without_quorum["context"]["selected_tokens"] == 65_536
    assert held_without_quorum["reasoning"]["selected"] is None
    assert held_without_quorum["nodes"]["agree"] == "active"
    assert held_without_quorum["nodes"]["hold"] == "pending"
    assert held_without_quorum["labels"]["hold"] == "Custom hold"


def test_single_model_authority_projects_one_lane_and_binds_identity() -> None:
    result = projection.project_decision_trace(
        _trace(
            state="ready",
            artifact="done",
            decision="done",
            lanes=[
                {
                    **_lane("primary", "done", think="low", context=32_768),
                    "model": "Qwen3.8-Flash-Next-oQ4e-mtp",
                    "revision": "a" * 64,
                },
                _lane("challenger", "skipped"),
                _lane("tie_break", "skipped"),
            ],
            authority_kind="single_model_v1",
            quorum_flow=False,
        )
    )

    assert result["single_model"] is True
    assert result["mode"] == "single"
    assert result["authority_kind"] == "single_model_v1"
    assert result["authority"] == {
        "kind": "single_model_v1",
        "label": "Single Authority",
        "model": "Qwen3.8-Flash-Next-oQ4e-mtp",
        "revision": "a" * 64,
        "target": 1,
        "validated": True,
        "repair_is_vote": False,
    }
    assert list(result["lanes"]) == ["primary"]
    assert result["lanes"]["primary"]["label"] == "Single Authority"
    assert result["model_routes"] == {"primary": "done"}
    assert result["paths"]["single-artifact"] == "done"
    assert result["labels"]["authority"] == "Single Authority"
    assert result["labels"]["validation"] == "Validated"
    assert result["labels"]["target"] == "1"
    assert result["labels"]["repair"] == "REPAIR ≠ VOTE"


def test_active_reasoning_fit_is_checking() -> None:
    result = projection.project_decision_trace(
        _trace(
            dispatch="active",
            lanes=[
                _lane("primary", "active", think="low", context=32_768),
                _lane("challenger"),
                _lane("tie_break"),
            ],
        )
    )

    assert result["labels"] == {
        "fit": "CHECKING",
        "fit_pass": False,
        "hold": "No safe quorum",
    }
    assert result["paths"]["plan-fit"] == "active"
