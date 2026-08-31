"""Project one authoritative Decision Trace snapshot into SVG display state."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Final

TRACE_PROJECTION_SCHEMA: Final = "chronovisor.decision-trace-projection.v2"
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
_INGEST_JOB_STEPS: Final = (
    "target",
    "generate",
    "authority",
    "route",
    "mutation",
    "apply",
    "publish",
    "readback",
    "complete",
    "hold",
)
_INGEST_JOB_EDGES: Final = {
    "target-generate": ("target", "generate", None),
    "generate-authority": ("generate", "authority", None),
    "authority-route": ("authority", "route", None),
    "retry": ("route", "generate", "RETRY"),
    "hold": ("route", "hold", "HOLD"),
    "accepted": ("route", "mutation", "ACCEPTED"),
    "apply": ("mutation", "apply", "APPLY"),
    "noop": ("mutation", "readback", "NOOP"),
    "apply-publish": ("apply", "publish", None),
    "publish-readback": ("publish", "readback", None),
    "readback-complete": ("readback", "complete", None),
}


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


def _ingest_processing_projection(
    status: Mapping[str, Any],
    trace: Mapping[str, Any],
    lane: Mapping[str, Any],
) -> dict[str, Any]:
    """Reduce ingest runtime facts into the fixed continuation graph."""

    stage = str(status.get("stage") or status.get("current_op") or "").lower()
    llm_value = status.get("llm")
    llm = llm_value if isinstance(llm_value, Mapping) else {}
    host_value = status.get("host_phase")
    host = host_value if isinstance(host_value, Mapping) else {}
    role = str(trace.get("task_role") or "").lower()
    batch_value = status.get("batch")
    batch = batch_value if isinstance(batch_value, Mapping) else {}
    current_job = bool(
        status.get("current_job_id")
        or status.get("current_raw")
        or batch.get("active")
        or llm.get("active")
        or (lane.get("state") == "active" and lane.get("recent") is not True)
    )
    terminal = stage in {
        "complete",
        "completion-ack",
        "projection",
        "semantic-noop",
    } or (not current_job and bool(status.get("last_success")))
    host_matches_stage = str(host.get("name") or "").lower() == stage
    host_complete = host_matches_stage and host.get("state") == "complete"
    states = dict.fromkeys(_INGEST_JOB_STEPS, "pending")
    failed = (
        str(status.get("state") or "").lower() == "error"
        or stage == "failed"
        or (host_matches_stage and host.get("state") == "error")
        or llm.get("event") == "error"
    )
    trace_ready = str(trace.get("state") or "") in {"ready", "agreed"}
    authority_ready = role.startswith("ingest_reconciliation") and trace_ready
    retrying = stage in {"local-regenerate", "frontier-regenerate"}
    success_value = status.get("last_success")
    last_success = success_value if isinstance(success_value, Mapping) else {}
    disposition = str(
        status.get("ingest_disposition")
        or last_success.get("local_consensus_status")
        or last_success.get("frontier_status")
        or ""
    ).lower()
    disposition_branch = (
        "noop"
        if disposition == "confirmed_noop"
        else "apply"
        if disposition == "apply_available"
        else ""
    )
    entry = (
        "target"
        if stage == "triage" or role.startswith("ingest_triage")
        else "authority"
        if role.startswith("ingest_recall_metadata")
        else "route"
    )

    def mark_done(*keys: str) -> None:
        for key in keys:
            states[key] = "done"

    current = ""
    branch = ""
    if terminal:
        mark_done(
            "target",
            "generate",
            "authority",
            "route",
            "mutation",
            "readback",
            "complete",
        )
        branch = disposition_branch or "apply"
        if branch == "noop":
            states["apply"] = "skipped"
            states["publish"] = "skipped"
        else:
            mark_done("apply", "publish")
        current = "complete"
    elif failed:
        states["route"] = "error"
        states["hold"] = "error"
        branch = "hold"
        current = "hold"
    elif retrying:
        mark_done("target", "authority", "route")
        states["generate"] = "active"
        branch = "retry"
        current = "generate"
    elif stage == "triage":
        states["target"] = "active" if trace_ready else "pending"
        current = "target"
    elif stage == "target-resolution":
        states["target"] = "done" if host_complete else "active"
        current = "generate" if host_complete else "target"
    elif stage == "generate":
        mark_done("target")
        generated = llm.get("active") is False and llm.get("event") == "done"
        states["generate"] = "done" if generated else "active"
        current = "authority" if generated else "generate"
    elif stage in {
        "authorization",
        "authorization-continuation",
        "local-consensus-review",
        "frontier-review",
        "locked",
    }:
        mark_done("target", "generate")
        states["authority"] = "done" if authority_ready else "active"
        states["route"] = "active" if authority_ready else "pending"
        current = "route" if authority_ready else "authority"
    elif stage == "apply":
        mark_done("target", "generate", "authority", "route")
        branch = disposition_branch
        if branch:
            mark_done("mutation")
        else:
            states["mutation"] = "active"
        if branch == "noop":
            states["apply"] = "skipped"
            states["publish"] = "skipped"
            current = "readback" if host_complete else "mutation"
        elif branch == "apply":
            states["apply"] = "done" if host_complete else "active"
            current = "publish" if host_complete else "apply"
        else:
            current = "mutation"
    elif stage in {"semantic-publish", "claim-publish", "state-register"}:
        mark_done("target", "generate", "authority", "route", "mutation", "apply")
        states["publish"] = "done" if host_complete else "active"
        branch = "apply"
        current = "readback" if host_complete else "publish"
    elif stage == "read-back":
        mark_done("target", "generate", "authority", "route", "mutation")
        branch = disposition_branch
        if branch == "apply":
            mark_done("apply", "publish")
        elif branch == "noop":
            states["apply"] = "skipped"
            states["publish"] = "skipped"
        states["readback"] = "done" if host_complete else "active"
        current = "complete" if host_complete else "readback"
    else:
        current = entry
        if trace_ready:
            states[entry] = "active"

    lane_floor = (
        {"generate": "generate", "consensus": "authority", "apply": "mutation"}.get(
            str(lane.get("current_step") or "").lower(),
            "",
        )
        if lane.get("state") == "active" and lane.get("recent") is not True
        else ""
    )
    if (
        not branch
        and lane_floor
        and _INGEST_JOB_STEPS.index(current) < _INGEST_JOB_STEPS.index(lane_floor)
    ):
        floor_index = _INGEST_JOB_STEPS.index(lane_floor)
        mark_done(*_INGEST_JOB_STEPS[:floor_index])
        states[lane_floor] = "active"
        current = lane_floor

    if branch and branch != "hold":
        states["hold"] = "skipped"

    def onward(source: str, target: str) -> str:
        return states[target] if states[source] == "done" else "pending"

    route_resolved = bool(branch) or states["mutation"] != "pending"
    paths = {
        "target-generate": "done"
        if branch == "retry"
        else onward("target", "generate"),
        "generate-authority": "done"
        if branch == "retry"
        else onward("generate", "authority"),
        "authority-route": "done"
        if branch == "retry"
        else onward("authority", "route"),
        "retry": "active"
        if branch == "retry"
        else "skipped"
        if route_resolved
        else "pending",
        "hold": "error"
        if branch == "hold"
        else "skipped"
        if route_resolved
        else "pending",
        "accepted": (
            onward("route", "mutation")
            if branch in {"apply", "noop"} or states["mutation"] != "pending"
            else "skipped"
            if route_resolved
            else "pending"
        ),
        "apply": (
            onward("mutation", "apply")
            if branch == "apply"
            else "skipped"
            if branch
            else "pending"
        ),
        "noop": (
            (
                "active"
                if states["readback"] == "pending"
                else onward("mutation", "readback")
            )
            if branch == "noop"
            else "skipped"
            if branch
            else "pending"
        ),
        "apply-publish": (
            onward("apply", "publish")
            if branch == "apply"
            else "skipped"
            if branch
            else "pending"
        ),
        "publish-readback": (
            onward("publish", "readback")
            if branch == "apply"
            else "skipped"
            if branch
            else "pending"
        ),
        "readback-complete": (
            onward("readback", "complete")
            if branch in {"apply", "noop"} or states["readback"] != "pending"
            else "skipped"
            if branch
            else "pending"
        ),
    }
    node_labels = {
        "target": "Target",
        "generate": "Generate",
        "authority": "Authority",
        "route": "Result route",
        "mutation": "Page change?",
        "apply": "Apply",
        "publish": "Publish",
        "readback": "Read-back",
        "complete": "Complete",
        "hold": "Hold",
    }
    return {
        "kind": "ingest",
        "pipeline": "ingest",
        "graph_id": "processing:ingest:v2",
        "current": current,
        "selected_branch": branch or None,
        "entry": entry,
        "states": states,
        "paths": paths,
        "nodes": [
            {
                "id": key,
                "label": node_labels[key],
                "state": states[key],
                "type": "decision" if key in {"route", "mutation"} else "milestone",
            }
            for key in _INGEST_JOB_STEPS
        ],
        "edges": [
            {
                "id": key,
                "source": source,
                "target": target,
                "label": label,
                "state": paths[key],
            }
            for key, (source, target, label) in _INGEST_JOB_EDGES.items()
        ],
    }


def _linear_processing_projection(
    pipeline: str, lane: Mapping[str, Any]
) -> dict[str, Any]:
    rows = lane.get("steps")
    steps = (
        [row for row in rows if isinstance(row, Mapping)]
        if isinstance(rows, list)
        else []
    )
    nodes = [
        {
            "id": str(row.get("key") or f"step-{index}"),
            "label": str(row.get("label") or row.get("key") or f"Step {index + 1}"),
            "state": _state(row.get("status")),
            "type": "milestone",
        }
        for index, row in enumerate(steps[:5])
    ]
    edges = []
    for source, target in zip(nodes, nodes[1:], strict=False):
        target_state = target["state"]
        state = target_state if source["state"] == "done" else "pending"
        edges.append(
            {
                "id": f"{source['id']}-{target['id']}",
                "source": source["id"],
                "target": target["id"],
                "label": None,
                "state": state,
            }
        )
    return {
        "kind": "linear",
        "pipeline": pipeline,
        "graph_id": f"processing:{pipeline}:v2",
        "current": str(lane.get("current_step") or "") or None,
        "selected_branch": None,
        "entry": nodes[0]["id"] if nodes else None,
        "nodes": nodes,
        "edges": edges,
        "states": {node["id"]: node["state"] for node in nodes},
        "paths": {edge["id"]: edge["state"] for edge in edges},
    }


def project_processing_trace(
    pipeline: str | None,
    lane: Mapping[str, Any] | None = None,
    runtime_status: Mapping[str, Any] | None = None,
    trace: Mapping[str, Any] | None = None,
) -> dict[str, Any] | None:
    """Project any Processing Lane through one display contract."""

    if not pipeline:
        return None
    lane = lane or {}
    trace = trace or {}
    if pipeline == "ingest":
        return _ingest_processing_projection(runtime_status or {}, trace, lane)
    return _linear_processing_projection(pipeline, lane)


def project_decision_trace(
    trace: Mapping[str, Any],
    *,
    pipeline: str | None = None,
    processing_lane: Mapping[str, Any] | None = None,
    runtime_status: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
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
            "label": str(lane.get("label") or role.replace("_", " ").title()),
            "think": str(lane.get("think") or "") or None,
            "result": str(lane.get("result") or "") or None,
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
        else "Validation failed"
        if single_model
        else str(trace.get("summary") or "No safe quorum").split("·", 1)[0].strip()
    )
    display_lanes = lanes
    display_model_routes = {role: lanes[role]["state"] for role in TRACE_ROLES}
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

    processing = project_processing_trace(
        pipeline,
        processing_lane,
        runtime_status,
        trace,
    )
    execution_id = str(
        trace.get("request_sha256")
        or (runtime_status or {}).get("current_job_id")
        or (runtime_status or {}).get("current_raw")
        or "idle"
    )
    revision_candidates = [
        str(value)
        for value in (
            trace.get("updated_at"),
            (runtime_status or {}).get("updated_at"),
            (runtime_status or {}).get("timestamp"),
        )
        if value
    ]
    revision = max(revision_candidates, default=str(trace.get("event_count") or "0"))
    result = {
        "schema": TRACE_PROJECTION_SCHEMA,
        "execution_id": execution_id,
        "graph_id": f"decision:{'single' if single_model else 'quorum'}:v2",
        "revision": revision,
        "pipeline": pipeline,
        "trace_state": trace_state,
        "outcome_kind": str(outcome.get("kind") or "idle"),
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
    if processing is not None:
        result["processing"] = processing
    if authority is not None:
        result["authority"] = authority
    return result
