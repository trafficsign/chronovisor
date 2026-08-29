"""Project one authoritative Decision Trace snapshot into SVG display state."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Final

TRACE_PROJECTION_SCHEMA: Final = "chronovisor.decision-trace-projection.v1"
TRACE_SINGLE_AUTHORITY_KIND: Final = "single_model_v1"
TRACE_STATES: Final = frozenset({"pending", "active", "done", "skipped", "error"})
TRACE_ROLES: Final = ("primary", "challenger", "tie_break")
TRACE_PHASES: Final = ("trigger", "load", "context", "generate", "validate", "vote")
TRACE_RAILS: Final = {
    "trigger-load": "load",
    "load-context": "context",
    "context-generate": "generate",
    "generate-validate": "validate",
    "validate-vote": "vote",
}
TRACE_PATH_KEYS: Final = (
    "packet-preflight",
    "preflight-execution_plan",
    "execution-plan-context",
    "plan-context",
    "reasoning-off",
    "reasoning-low",
    "reasoning-medium",
    "reasoning-high",
    "plan-fit",
    "plan-dispatch",
    "primary-challenger",
    "single-artifact",
    "challenger-agree",
    "pair-artifact-join",
    "pair-tie_break",
    "tie_break-quorum",
    "quorum-artifact-join",
    "quorum-hold",
    "artifact-seal",
    "seal-decision",
    "seal-hold",
)

_CONTEXT_OPTIONS: Final = (32_768, 65_536, 98_304, 131_072)
_REASONING_MODES: Final = ("off", "low", "medium", "high")


def _state(value: object, default: str = "pending") -> str:
    return str(value) if value in TRACE_STATES else default


def _positive_int(value: object) -> int | None:
    return (
        value
        if isinstance(value, int) and not isinstance(value, bool) and value > 0
        else None
    )


def _rows_by_key(value: object) -> dict[str, Mapping[str, Any]]:
    if not isinstance(value, list):
        return {}
    return {
        str(row["key"]): row
        for row in value
        if isinstance(row, Mapping) and isinstance(row.get("key"), str)
    }


def _context_label(value: int | None) -> str:
    return f"{value // 1000}K" if value else "—"


def _context_options(selected: int | None, state: str) -> list[dict[str, Any]]:
    displayed = list(_CONTEXT_OPTIONS)
    if selected and selected not in displayed:
        nearest = min(
            range(len(displayed)), key=lambda index: abs(displayed[index] - selected)
        )
        displayed[nearest] = selected
    return [
        {
            "tokens": tokens,
            "label": _context_label(tokens),
            "selected": tokens == selected,
            "state": state if tokens == selected else "pending",
        }
        for tokens in displayed
    ]


def _selected_path(target_state: str, *, selected: bool) -> str:
    """Keep a chosen branch solid while its next milestone is still pending."""

    if not selected:
        return "pending"
    return "active" if target_state == "pending" else target_state


def project_decision_trace(trace: Mapping[str, Any]) -> dict[str, Any]:
    """Return the complete, versioned display contract for the fixed SVG topology."""

    overall = _rows_by_key(trace.get("overall"))
    lane_rows = _rows_by_key(trace.get("lanes"))

    def overall_state(key: str) -> str:
        return _state(overall.get(key, {}).get("status"))

    trace_state = str(trace.get("state") or "idle")
    outcome_value = trace.get("outcome")
    outcome: Mapping[str, Any] = (
        outcome_value if isinstance(outcome_value, Mapping) else {}
    )
    quorum_payload = overall_state("quorum")
    artifact_payload = overall_state("artifact")
    decision_payload = overall_state("decision")
    seal_failure = trace_state == "quarantined" and (
        artifact_payload == "error"
        or "seal" in f"{outcome.get('code', '')} {outcome.get('reason', '')}".lower()
    )
    no_safe_quorum = (
        trace_state == "quarantined"
        and not seal_failure
        and trace.get("quorum_attempted") is not False
    )
    authority_kind_value = trace.get("authority_kind")
    authority_kind = (
        authority_kind_value.strip()
        if isinstance(authority_kind_value, str) and authority_kind_value.strip()
        else None
    )
    single_flow = trace.get("quorum_flow") is False
    # A single-model projection is enabled only by persisted execution facts.
    # A bare quorum_flow=False snapshot remains compatible with the legacy
    # three-lane projection used by synthetic fixtures.
    single_model = single_flow and authority_kind == TRACE_SINGLE_AUTHORITY_KIND

    lanes: dict[str, dict[str, Any]] = {}
    for role in TRACE_ROLES:
        lane = lane_rows.get(role, {})
        steps = _rows_by_key(lane.get("steps"))
        step_states = {
            phase: _state(steps.get(phase, {}).get("status")) for phase in TRACE_PHASES
        }
        repair_events = (
            [
                event
                for event in trace.get("events", [])
                if isinstance(event, Mapping)
                and event.get("lane") == role
                and event.get("phase") == "repair"
            ]
            if isinstance(trace.get("events"), list)
            else []
        )
        repair_turns = lane.get("repair_turns")
        repair_count = (
            repair_turns
            if isinstance(repair_turns, int)
            and not isinstance(repair_turns, bool)
            and repair_turns > 0
            else 0
        )
        repair_count = max(
            repair_count,
            max(
                (max(1, int(event.get("attempt") or 0)) for event in repair_events),
                default=0,
            ),
        )
        repair_state = (
            "active"
            if lane.get("state") == "active" and lane.get("phase") == "repair"
            else "done"
            if repair_count or repair_events
            else "pending"
        )
        lane_projection: dict[str, Any] = {
            "state": _state(lane.get("state")),
            "steps": step_states,
            "rails": {key: step_states[phase] for key, phase in TRACE_RAILS.items()},
            "repair": repair_state,
            "repair_attempt": repair_count,
        }
        for key in ("model", "revision"):
            value = lane.get(key)
            if isinstance(value, str) and value:
                lane_projection[key] = value
        lanes[role] = lane_projection

    selected_lane = next(
        (
            lane_rows[role]
            for role in TRACE_ROLES
            if lane_rows.get(role, {}).get("state") == "active"
        ),
        next(
            (
                lane_rows[role]
                for role in reversed(TRACE_ROLES)
                if lane_rows.get(role, {}).get("state") == "done"
            ),
            None,
        ),
    )
    selected_context = _positive_int(
        (selected_lane or {}).get("requested_context_tokens")
    )
    selected_context = selected_context or _positive_int(
        (selected_lane or {}).get("context_tokens")
    )
    selected_context = selected_context or _positive_int(trace.get("context_tokens"))
    requested_mode = str((selected_lane or {}).get("think") or "").lower()
    selected_reasoning = requested_mode if requested_mode in _REASONING_MODES else None

    packet_state = overall_state("packet")
    plan_state = overall_state("dispatch")
    fit_state = plan_state if selected_reasoning is not None else "pending"
    tie_state = lanes["tie_break"]["state"]
    tie_observed = trace.get("tie_break_used") is True or tie_state in {
        "active",
        "done",
        "error",
    }
    pair_agreement = trace.get("pair_agreement") is True
    pair_no = (
        "active"
        if tie_state == "active"
        else "done"
        if tie_observed
        or trace.get("pair_agreement") is False
        and trace.get("quorum_attempted") is True
        else "pending"
    )
    safe_no_quorum = no_safe_quorum and not single_flow
    tie_finished = tie_state in {"done", "error"}
    tie_quorum = _selected_path(
        quorum_payload,
        selected=not single_flow and tie_observed and tie_finished,
    )
    safe_tie_quorum = tie_quorum == "done" and not safe_no_quorum
    pair_yes = _selected_path(artifact_payload, selected=pair_agreement)
    quorum_yes = _selected_path(artifact_payload, selected=safe_tie_quorum)
    quorum_no = "error" if safe_no_quorum else "pending"
    seal_failed = seal_failure or artifact_payload == "error"
    seal_gate = (
        "error" if seal_failed else "done" if artifact_payload == "done" else "pending"
    )
    seal_input = seal_gate
    seal_yes = _selected_path(decision_payload, selected=seal_gate == "done")
    seal_no = "error" if seal_failed else "pending"
    artifact_state = (
        "error" if seal_failed else "skipped" if no_safe_quorum else artifact_payload
    )
    agreement_state = (
        "done"
        if trace_state == "agreed" or seal_failure
        else "error"
        if no_safe_quorum
        else "active"
        if tie_observed or lanes["challenger"]["state"] == "done"
        else "pending"
    )
    quorum_state = tie_quorum

    nodes = {
        "packet": packet_state,
        "preflight": packet_state,
        "execution_plan": plan_state,
        "context": plan_state,
        "headroom": plan_state,
        "fit": fit_state,
        "agree": agreement_state,
        "quorum": quorum_state,
        "artifact": artifact_state,
        "seal": seal_gate,
        "decision": decision_payload,
        "hold": "error" if safe_no_quorum or seal_failure else "pending",
    }
    paths = {key: "pending" for key in TRACE_PATH_KEYS}
    paths.update(
        {
            "packet-preflight": packet_state,
            "preflight-execution_plan": plan_state,
            "execution-plan-context": plan_state,
            "plan-context": plan_state,
            "plan-fit": fit_state
            if selected_reasoning not in {None, "off"}
            else "pending",
            "plan-dispatch": plan_state,
            "primary-challenger": lanes["challenger"]["steps"]["trigger"],
            "single-artifact": artifact_state if single_flow else "pending",
            "challenger-agree": (
                lanes["challenger"]["steps"]["vote"]
                if lanes["challenger"]["steps"]["vote"] in {"done", "error"}
                else "pending"
            ),
            "pair-artifact-join": pair_yes,
            "pair-tie_break": pair_no,
            "tie_break-quorum": tie_quorum,
            "quorum-artifact-join": quorum_yes,
            "quorum-hold": quorum_no,
            "artifact-seal": seal_input,
            "seal-decision": seal_yes,
            "seal-hold": seal_no,
        }
    )
    for mode in _REASONING_MODES:
        paths[f"reasoning-{mode}"] = (
            plan_state if mode == selected_reasoning else "pending"
        )

    required_context = _positive_int(
        (selected_lane or {}).get("required_context_tokens")
    )
    hold_reason = (
        "Seal failed"
        if seal_failure
        else str(trace.get("summary") or "No safe quorum").split("·", 1)[0].strip()
    )
    display_lanes = lanes
    display_model_routes = {
        role: lanes[role]["state"] for role in TRACE_ROLES
    }
    labels: dict[str, Any] = {
        "fit": "BYPASS"
        if selected_reasoning == "off" and fit_state != "pending"
        else "headroom OK"
        if fit_state == "done"
        else "CHECKING"
        if fit_state == "active"
        else "WAITING",
        "fit_pass": fit_state == "done" and selected_reasoning != "off",
        "hold": hold_reason,
    }
    authority: dict[str, Any] | None = None
    if single_model:
        primary_lane = lane_rows.get("primary", {})
        model = primary_lane.get("model") or trace.get("model")
        revision = primary_lane.get("revision") or trace.get("revision")
        model = model if isinstance(model, str) and model else None
        revision = revision if isinstance(revision, str) and revision else None
        authority = {
            "kind": TRACE_SINGLE_AUTHORITY_KIND,
            "label": "Single Authority",
            "model": model,
            "revision": revision,
            "target": 1,
            "validated": trace_state in {"ready", "agreed"}
            and artifact_payload == "done"
            and decision_payload == "done",
            "repair_is_vote": False,
        }
        display_lanes = {"primary": {**lanes["primary"], "label": "Single Authority"}}
        display_model_routes = {"primary": lanes["primary"]["state"]}
        labels.update(
            {
                "authority": "Single Authority",
                "validation": "Validated"
                if authority["validated"]
                else "Held"
                if trace_state == "quarantined"
                else "Validating",
                "target": "1",
                "repair": "REPAIR ≠ VOTE",
            }
        )

    result = {
        "schema": TRACE_PROJECTION_SCHEMA,
        "mode": "single" if single_model else "quorum",
        "single_model": single_model,
        "authority_kind": authority_kind,
        "nodes": nodes,
        "paths": paths,
        "lanes": display_lanes,
        "model_routes": display_model_routes,
        "context": {
            "selected_tokens": selected_context,
            "options": _context_options(selected_context, plan_state),
            "label": f"required {_context_label(required_context)} → selected {_context_label(selected_context)}",
        },
        "reasoning": {
            "selected": selected_reasoning,
            "options": {
                mode: plan_state if mode == selected_reasoning else "pending"
                for mode in _REASONING_MODES
            },
        },
        "labels": labels,
    }
    if authority is not None:
        result["authority"] = authority
    return result
