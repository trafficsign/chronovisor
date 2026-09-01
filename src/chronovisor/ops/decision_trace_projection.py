"""Project one workflow execution onto one fixed Decision Trace rail."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, Final

TRACE_PROJECTION_SCHEMA: Final = "chronovisor.decision-trace-projection.v3"
TRACE_SINGLE_AUTHORITY_KIND: Final = "single_model_v1"
TRACE_STATES: Final = frozenset({"pending", "active", "done", "skipped", "error"})
TRACE_PHASES: Final = ("trigger", "load", "context", "generate", "validate", "vote")
_CONTEXT_OPTIONS: Final = (32_768, 65_536, 98_304, 131_072)
_REASONING_MODES: Final = ("off", "low", "medium", "high")
_PLAN_ROUTE: Final = (
    "packet",
    "preflight",
    "execution_plan",
    "context_choice",
    "headroom",
    "reasoning_choice",
    "fit",
)


def _node(key: str, label: str, kind: str = "step") -> dict[str, str]:
    return {"id": key, "label": label, "type": kind}


def _edge(
    source: str, target: str, label: str | None = None, kind: str = "main"
) -> dict[str, str]:
    row = {
        "id": f"{source}-{target}-{label or kind}".lower().replace(" ", "_"),
        "source": source,
        "target": target,
        "kind": kind,
    }
    if label:
        row["label"] = label
    return row


def _linear_workflow(
    nodes: Sequence[tuple[str, str]],
    success: Sequence[str],
) -> dict[str, Any]:
    result_index = success.index("result")
    return {
        "nodes": [
            _node(
                key,
                label,
                "decision"
                if key == "result"
                else "terminal"
                if key in {"complete", "hold"}
                else "step",
            )
            for key, label in nodes
        ],
        "edges": [
            *(
                _edge(source, target, "ACCEPTED" if source == "result" else None)
                for source, target in zip(success, success[1:], strict=False)
            ),
            _edge("result", "hold", "HOLD", "branch"),
        ],
        "routes": {
            "success": tuple(success),
            "hold": (*success[: result_index + 1], "hold"),
        },
    }


def _with_execution_plan(workflow: dict[str, Any]) -> None:
    first = workflow["routes"]["success"][0]
    workflow["nodes"] = [
        *(
            _node(key, label, "plan")
            for key, label in (
                ("packet", "Packet"),
                ("preflight", "Preflight"),
                ("execution_plan", "Execution Plan"),
                ("context_choice", "Context Window"),
                ("headroom", "Task + headroom"),
                ("reasoning_choice", "Reasoning Budget"),
                ("fit", "Fit Gate"),
            )
        ),
        *workflow["nodes"],
    ]
    workflow["edges"] = [
        *(
            _edge(source, target)
            for source, target in zip(_PLAN_ROUTE, _PLAN_ROUTE[1:], strict=False)
        ),
        _edge("fit", first, "DISPATCH"),
        *workflow["edges"],
    ]
    workflow["routes"] = {
        name: (*_PLAN_ROUTE, *route) for name, route in workflow["routes"].items()
    }


TRACE_WORKFLOWS: Final = {
    "ingest": {
        "nodes": [
            _node("raw", "Raw"),
            _node("triage", "Triage"),
            _node("target", "Target"),
            _node("generate", "Generate"),
            _node("authority", "Authority"),
            _node("result", "Result route", "decision"),
            _node("change", "Page change?", "decision"),
            _node("apply", "Apply"),
            _node("publish", "Publish"),
            _node("readback", "Read-back"),
            _node("complete", "Complete", "terminal"),
            _node("hold", "Hold", "terminal"),
        ],
        "edges": [
            _edge("raw", "triage"),
            _edge("triage", "target"),
            _edge("target", "generate"),
            _edge("generate", "authority"),
            _edge("authority", "result"),
            _edge("result", "generate", "RETRY", "loop"),
            _edge("result", "change", "ACCEPTED", "branch"),
            _edge("result", "hold", "HOLD", "branch"),
            _edge("change", "apply", "APPLY", "branch"),
            _edge("change", "readback", "NOOP", "branch"),
            _edge("apply", "publish"),
            _edge("publish", "readback"),
            _edge("readback", "complete"),
        ],
        "routes": {
            "success": (
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
            ),
            "noop": (
                "raw",
                "triage",
                "target",
                "generate",
                "authority",
                "result",
                "change",
                "readback",
                "complete",
            ),
            "hold": (
                "raw",
                "triage",
                "target",
                "generate",
                "authority",
                "result",
                "hold",
            ),
            "retry": (
                "raw",
                "triage",
                "target",
                "generate",
                "authority",
                "result",
                "generate",
                "authority",
                "result",
                "change",
                "apply",
                "publish",
                "readback",
                "complete",
            ),
        },
    },
    "recall": _linear_workflow(
        (
            ("search", "Search"),
            ("rerank", "Rerank"),
            ("authority", "Authority"),
            ("result", "Result route"),
            ("commit", "Commit"),
            ("readback", "Read-back"),
            ("complete", "Complete"),
            ("hold", "Hold"),
        ),
        ("search", "rerank", "authority", "result", "commit", "readback", "complete"),
    ),
    "audit": _linear_workflow(
        (
            ("select", "Select"),
            ("inspect", "Inspect"),
            ("consensus", "Consensus"),
            ("result", "Result route"),
            ("report", "Report"),
            ("complete", "Complete"),
            ("hold", "Hold"),
        ),
        ("select", "inspect", "consensus", "result", "report", "complete"),
    ),
    "improve": _linear_workflow(
        (
            ("discover", "Discover"),
            ("generate", "Generate"),
            ("verify", "Verify"),
            ("result", "Result route"),
            ("apply", "Apply"),
            ("readback", "Read-back"),
            ("complete", "Complete"),
            ("hold", "Hold"),
        ),
        ("discover", "generate", "verify", "result", "apply", "readback", "complete"),
    ),
    "repair": _linear_workflow(
        (
            ("detect", "Detect"),
            ("local_fix", "Local fix"),
            ("verify", "Verify"),
            ("result", "Result route"),
            ("escalate", "Escalate"),
            ("readback", "Read-back"),
            ("complete", "Complete"),
            ("hold", "Hold"),
        ),
        ("detect", "local_fix", "verify", "result", "readback", "complete"),
    ),
    "typed_graph": _linear_workflow(
        (
            ("discover", "Discover"),
            ("extract", "Extract"),
            ("verify", "Verify"),
            ("consolidate", "Consolidate"),
            ("evaluate", "Evaluate"),
            ("result", "Result route"),
            ("promote", "Promote"),
            ("readback", "Read-back"),
            ("complete", "Complete"),
            ("hold", "Hold"),
        ),
        (
            "discover",
            "extract",
            "verify",
            "consolidate",
            "evaluate",
            "result",
            "promote",
            "readback",
            "complete",
        ),
    ),
}

# Escalation is a real repair branch and returns to the same verification rail.
TRACE_WORKFLOWS["repair"]["edges"].extend(
    (
        _edge("result", "escalate", "ESCALATE", "branch"),
        _edge("escalate", "verify", "RETRY", "loop"),
    )
)
TRACE_WORKFLOWS["repair"]["routes"]["escalate"] = (
    "detect",
    "local_fix",
    "verify",
    "result",
    "escalate",
    "verify",
    "result",
    "readback",
    "complete",
)

for _workflow in TRACE_WORKFLOWS.values():
    _with_execution_plan(_workflow)

_PIPELINE_STAGE_ALIASES: Final = {
    "ingest": {"consensus": "authority", "apply": "apply"},
    "recall": {
        "primary": "authority",
        "challenger": "authority",
        "tie_break": "authority",
    },
}
_PIPELINE_DEFAULT_STAGE: Final = {
    "ingest": "triage",
    "recall": "authority",
    "audit": "consensus",
    "improve": "generate",
    "repair": "local_fix",
    "typed_graph": "verify",
}
_INGEST_RUNTIME_STAGE: Final = {
    "raw": "raw",
    "triage": "triage",
    "target-resolution": "target",
    "generate": "generate",
    "local-regenerate": "generate",
    "frontier-regenerate": "generate",
    "authorization": "authority",
    "authorization-continuation": "authority",
    "local-consensus-review": "authority",
    "frontier-review": "authority",
    "locked": "result",
    "apply": "apply",
    "semantic-publish": "publish",
    "claim-publish": "publish",
    "state-register": "publish",
    "read-back": "readback",
    "complete": "complete",
    "completion-ack": "complete",
    "projection": "complete",
    "semantic-noop": "complete",
    "failed": "hold",
}


def _positive_int(value: object) -> int | None:
    return (
        value
        if isinstance(value, int) and not isinstance(value, bool) and value > 0
        else None
    )


def _context_label(value: int | None) -> str:
    return f"{value // 1000}K" if value else "—"


def _context_options(selected: int | None) -> list[dict[str, Any]]:
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
        }
        for tokens in displayed
    ]


def _pipeline_from_trace(trace: Mapping[str, Any]) -> str:
    role = str(trace.get("task_role") or "").casefold()
    if role.startswith(("relation_", "entity_merge", "recall_rubric")):
        return "typed_graph"
    if role.startswith("recall"):
        return "recall"
    if role.startswith("ingest"):
        return "ingest"
    if role.startswith(("model_eval", "autonomy", "orphan_link")):
        return "improve"
    if "repair" in role:
        return "repair"
    return "audit"


def _current_stage(
    pipeline: str,
    trace: Mapping[str, Any],
    lane: Mapping[str, Any],
    runtime: Mapping[str, Any],
) -> str | None:
    trace_state = str(trace.get("state") or "").casefold()
    if pipeline == "ingest":
        runtime_stage = str(
            runtime.get("stage") or runtime.get("current_op") or ""
        ).casefold()
        if runtime_stage in _INGEST_RUNTIME_STAGE:
            return _INGEST_RUNTIME_STAGE[runtime_stage]
        if not (runtime.get("current_job_id") or runtime.get("current_raw")):
            if trace_state in {"ready", "agreed"}:
                return "complete"
            if trace_state in {"quarantined", "error"}:
                return "hold"
        role = str(trace.get("task_role") or "").casefold()
        if role.startswith("ingest_triage"):
            return "triage"
        if role.startswith("ingest_recall_metadata"):
            return "generate"
        if role.startswith("ingest"):
            return "authority"
    elif not trace.get("active"):
        if trace_state in {"ready", "agreed"}:
            return "complete"
        if trace_state in {"quarantined", "error"}:
            return "hold"
    lane_stage = str(lane.get("current_step") or "").casefold()
    lane_stage = _PIPELINE_STAGE_ALIASES.get(pipeline, {}).get(lane_stage, lane_stage)
    node_ids = {node["id"] for node in TRACE_WORKFLOWS[pipeline]["nodes"]}
    if lane_stage in node_ids:
        return lane_stage
    if trace.get("request_sha256") or trace.get("active"):
        return _PIPELINE_DEFAULT_STAGE[pipeline]
    return None


def _selected_route(
    pipeline: str,
    trace: Mapping[str, Any],
    lane: Mapping[str, Any],
    runtime: Mapping[str, Any],
) -> str:
    trace_state = str(trace.get("state") or "").casefold()
    runtime_state = str(runtime.get("state") or "").casefold()
    stage = str(runtime.get("stage") or runtime.get("current_op") or "").casefold()
    if pipeline == "ingest":
        if runtime_state == "error" or stage == "failed":
            return "hold"
        if stage in {"local-regenerate", "frontier-regenerate"}:
            return "retry"
        success = runtime.get("last_success")
        last_success = success if isinstance(success, Mapping) else {}
        disposition = str(
            runtime.get("ingest_disposition")
            or last_success.get("local_consensus_status")
            or last_success.get("frontier_status")
            or ""
        ).casefold()
        if disposition == "confirmed_noop" or stage == "semantic-noop":
            return "noop"
        if runtime.get("current_job_id") or runtime.get("current_raw"):
            return "success"
    lane_stage = str(lane.get("current_step") or "").casefold()
    lane_phase = str(lane.get("phase") or "").casefold()
    if pipeline == "repair" and (
        lane_stage == "escalate"
        or "frontier" in lane_phase
        or stage in {"frontier-review", "escalate"}
    ):
        return "escalate"
    if trace_state in {"quarantined", "error"}:
        return "hold"
    return "success"


def _target_cursor(route: Sequence[str], stage: str | None, *, retrying: bool) -> int:
    if stage is None:
        return -1
    positions = [index for index, node in enumerate(route) if node == stage]
    return (positions[-1] if retrying else positions[0]) if positions else -1


def _active_lane(trace: Mapping[str, Any]) -> Mapping[str, Any]:
    lanes = trace.get("lanes")
    rows = (
        [row for row in lanes if isinstance(row, Mapping)]
        if isinstance(lanes, list)
        else []
    )
    return next(
        (row for row in rows if row.get("state") == "active"), rows[0] if rows else {}
    )


def _authority(
    trace: Mapping[str, Any], lane: Mapping[str, Any]
) -> dict[str, Any] | None:
    if trace.get("authority_kind") != TRACE_SINGLE_AUTHORITY_KIND:
        return None
    return {
        "kind": TRACE_SINGLE_AUTHORITY_KIND,
        "label": "Single Authority",
        "model": lane.get("model") or trace.get("model"),
        "revision": lane.get("revision") or trace.get("revision"),
        "target": 1,
        "validated": str(trace.get("state") or "") in {"ready", "agreed"},
        "repair_is_vote": False,
    }


def project_decision_trace(
    trace: Mapping[str, Any],
    pipeline: str | None = None,
    processing_lane: Mapping[str, Any] | None = None,
    runtime_status: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Return one fixed workflow graph and the single authoritative arrival point."""

    selected_pipeline = (
        pipeline if pipeline in TRACE_WORKFLOWS else _pipeline_from_trace(trace)
    )
    workflow = TRACE_WORKFLOWS[selected_pipeline]
    lane = processing_lane or {}
    runtime = runtime_status or {}
    route_name = _selected_route(selected_pipeline, trace, lane, runtime)
    route = workflow["routes"][route_name]
    stage = _current_stage(selected_pipeline, trace, lane, runtime)
    if route_name == "hold":
        stage = "hold"
    retrying = route_name in {"retry", "escalate"}
    cursor = _target_cursor(route, stage, retrying=retrying)
    trace_state = str(trace.get("state") or "idle")
    runtime_active = bool(
        selected_pipeline == "ingest"
        and (runtime.get("current_job_id") or runtime.get("current_raw"))
        and str(runtime.get("state") or "").casefold() not in {"error", "idle"}
        and stage not in {"complete", "hold"}
    )
    target_state = (
        "error"
        if stage == "hold"
        else "done"
        if stage == "complete"
        or (
            selected_pipeline != "ingest"
            and not trace.get("active")
            and trace_state in {"ready", "agreed"}
        )
        else "active"
        if cursor >= 0
        else "pending"
    )
    active_lane = _active_lane(trace)
    selected_context = _positive_int(active_lane.get("requested_context_tokens"))
    selected_context = selected_context or _positive_int(
        active_lane.get("context_tokens") or trace.get("context_tokens")
    )
    required_context = _positive_int(
        active_lane.get("required_context_tokens")
        or trace.get("required_context_tokens")
    )
    requested_mode = str(active_lane.get("think") or "").casefold()
    selected_reasoning = (
        requested_mode if requested_mode in _REASONING_MODES else None
    )
    single_model = trace.get("authority_kind") == TRACE_SINGLE_AUTHORITY_KIND
    identity = (
        (
            runtime.get("current_job_id")
            or runtime.get("current_raw")
            or lane.get("work_item")
            or trace.get("request_sha256")
        )
        if selected_pipeline == "ingest"
        else lane.get("work_item") or trace.get("request_sha256")
    )
    revision_values = [
        str(value)
        for value in (
            runtime.get("updated_at"),
            runtime.get("timestamp"),
            lane.get("updated_at"),
            trace.get("updated_at"),
            trace.get("event_count"),
        )
        if value is not None
    ]
    outcome = trace.get("outcome") if isinstance(trace.get("outcome"), Mapping) else {}
    result = {
        "schema": TRACE_PROJECTION_SCHEMA,
        "execution_id": str(identity or "idle"),
        "graph_id": f"workflow:{selected_pipeline}:v2",
        "revision": max(revision_values, default="0"),
        "pipeline": selected_pipeline,
        "trace_state": "active" if runtime_active else trace_state,
        "outcome_kind": str(outcome.get("kind") or "idle"),
        "mode": "single" if single_model else "quorum",
        "single_model": single_model,
        "authority_kind": trace.get("authority_kind"),
        "workflow": {
            "nodes": [dict(node) for node in workflow["nodes"]],
            "edges": [dict(edge) for edge in workflow["edges"]],
            "route": route_name,
            "route_node_ids": list(route),
            "target_cursor": cursor,
            "target_node": stage,
            "target_state": target_state,
        },
        "detail": {
            "phase": active_lane.get("phase"),
            "think": active_lane.get("think"),
            "model": active_lane.get("model"),
            "context_tokens": _positive_int(
                active_lane.get("context_tokens") or trace.get("context_tokens")
            ),
        },
        "context": {
            "selected_tokens": selected_context,
            "options": _context_options(selected_context),
            "label": f"required {_context_label(required_context)} → selected {_context_label(selected_context)}",
        },
        "reasoning": {
            "selected": selected_reasoning,
            "options": list(_REASONING_MODES),
        },
        "labels": {
            "validation": "Held"
            if stage == "hold"
            else "Validated"
            if trace_state in {"ready", "agreed"}
            else "Validating"
            if trace.get("active")
            else "Waiting",
            "hold": str(trace.get("summary") or "Processing held")
            .split("·", 1)[0]
            .strip(),
        },
    }
    authority = _authority(trace, active_lane)
    if authority is not None:
        result["authority"] = authority
    return result
