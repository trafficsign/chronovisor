"""Local-only structured worker for offline Recall distillation.

Its stdin/stdout contract intentionally contains no routing knobs.  The parent
may choose an operation and a fixed local role, but model/provider selection is
always re-resolved here from the local runtime configuration.
"""

from __future__ import annotations

import json
import re
import sys
from collections.abc import Mapping
from typing import Any

from chronovisor.core import ollama
from chronovisor.decision.local_structured import LocalStructuredSession

WORKER_SCHEMA = "chronovisor.recall-distillation-worker.v1"
# Keep the full initial request plus two bounded repair turns inside the fixed
# 32K local context.  The caller may split larger evidence deterministically.
MAX_INPUT_CHARS = 12_000
MAX_OUTPUT_CHARS = 4_000
MAX_DEADLINE_MS = 660_000
MAX_TEACHER_CANDIDATES = 16
_REQUEST_FIELDS = frozenset(
    {"schema", "operation", "role", "request_id", "deadline_ms", "input"}
)
_REQUEST_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$")
_ROLE_BY_OPERATION = {
    "teacher": frozenset(
        {
            "recall.distill.teacher.a",
            "recall.distill.teacher.b",
            "recall.distill.teacher.c",
        }
    ),
    "answer": frozenset({"recall.distill.answer_generator"}),
    "utility": frozenset({"recall.distill.utility_judge"}),
}


def _schema(operation: str, *, candidate_ids: tuple[str, ...] = ()) -> dict[str, Any]:
    common = {
        "additionalProperties": False,
        "required": ["confidence", "rationale"],
        "properties": {
            "confidence": {"type": "number", "minimum": 0, "maximum": 1},
            "rationale": {"type": "string", "minLength": 1, "maxLength": 600},
        },
    }
    if operation == "teacher":
        return {
            "type": "object",
            **common,
            "required": [
                "labels",
            ],
            "properties": {
                "labels": {
                    "type": "array",
                    "minItems": len(candidate_ids),
                    "maxItems": len(candidate_ids),
                    "items": {
                        "type": "object",
                        "additionalProperties": False,
                        "required": [
                            "candidate_id",
                            "verdict",
                            "confidence",
                            "rationale",
                            "minimal_atom_ids",
                            "missing_slots",
                            "changing_claim",
                        ],
                        "properties": {
                            "candidate_id": {
                                "type": "string",
                                "enum": list(candidate_ids),
                            },
                            "verdict": {
                                "type": "string",
                                "enum": ["relevant", "irrelevant", "uncertain"],
                            },
                            **common["properties"],
                            "minimal_atom_ids": {
                                "type": "array",
                                "maxItems": 8,
                                "uniqueItems": True,
                                "items": {
                                    "type": "string",
                                    "minLength": 1,
                                    "maxLength": 160,
                                },
                            },
                            "missing_slots": {
                                "type": "array",
                                "maxItems": 5,
                                "uniqueItems": True,
                                "items": {
                                    "type": "string",
                                    "minLength": 1,
                                    "maxLength": 160,
                                },
                            },
                            "changing_claim": {
                                "type": "string",
                                "maxLength": 600,
                            },
                        },
                    },
                },
            },
        }
    if operation == "answer":
        return {
            "type": "object",
            **common,
            "required": ["answer", *common["required"]],
            "properties": {
                **common["properties"],
                "answer": {"type": "string", "minLength": 1, "maxLength": 4_000},
            },
        }
    return {
        "type": "object",
        **common,
        "required": [
            "verdict",
            "basis_atom_ids",
            "blind_order",
            "blind_choice",
            *common["required"],
        ],
        "properties": {
            **common["properties"],
            "verdict": {
                "type": "string",
                "enum": ["helpful", "neutral", "harmful", "uncertain"],
            },
            "basis_atom_ids": {
                "type": "array",
                "maxItems": 8,
                "uniqueItems": True,
                "items": {"type": "string", "minLength": 1, "maxLength": 160},
            },
            "blind_order": {"type": "string", "enum": ["a_first", "b_first"]},
            "blind_choice": {
                "type": "string",
                "enum": ["a", "b", "tie", "uncertain"],
            },
        },
    }


def _teacher_candidate_ids(request_input: Mapping[str, Any]) -> tuple[str, ...]:
    candidates = request_input.get("candidates")
    if (
        not isinstance(candidates, list)
        or not 1 <= len(candidates) <= MAX_TEACHER_CANDIDATES
    ):
        raise ValueError
    candidate_ids = tuple(
        str(row.get("candidate_id") or "") if isinstance(row, Mapping) else ""
        for row in candidates
    )
    if len(candidate_ids) != len(set(candidate_ids)) or any(
        not 0 < len(candidate_id) <= 160 for candidate_id in candidate_ids
    ):
        raise ValueError
    return candidate_ids


def _valid_teacher_labels(
    value: Mapping[str, Any], candidate_ids: tuple[str, ...]
) -> bool:
    labels = value.get("labels")
    return (
        isinstance(labels, list)
        and len(labels) == len(candidate_ids)
        and all(isinstance(label, Mapping) for label in labels)
        and {str(label.get("candidate_id") or "") for label in labels}
        == set(candidate_ids)
    )


def _system(operation: str) -> str:
    if operation == "teacher":
        return (
            "You are one local Recall relevance teacher. Judge only the supplied "
            "point-in-time evidence. Return schema-valid JSON and choose uncertain "
            "when the evidence is insufficient."
        )
    if operation == "answer":
        return (
            "You are a local counterfactual answer generator. Use only supplied "
            "evidence, do not invent facts, and return schema-valid JSON."
        )
    return (
        "You are a blind local Recall utility judge. Compare answer A and answer B "
        "using only the supplied matched point-in-time evidence. Choose a, b, tie, "
        "or uncertain without inferring which answer used the candidate. Return "
        "schema-valid JSON."
    )


def _safe_operation(value: object) -> str:
    return str(value) if value in _ROLE_BY_OPERATION else "invalid"


def _safe_role(value: object) -> str:
    return (
        str(value)
        if isinstance(value, str) and value in set().union(*_ROLE_BY_OPERATION.values())
        else "invalid"
    )


def _safe_request_id(value: object) -> str:
    return (
        str(value)
        if isinstance(value, str) and _REQUEST_ID.fullmatch(value)
        else "invalid"
    )


def _envelope(
    *,
    ok: bool,
    operation: str,
    role: str,
    request_id: str,
    route: ollama.RuntimeGenerationRoute | None = None,
    model_digest: str = "",
    result: Mapping[str, Any] | None = None,
    failure_class: str = "",
) -> dict[str, Any]:
    return {
        "schema": WORKER_SCHEMA,
        "ok": ok,
        "operation": operation,
        "role": role,
        "request_id": request_id,
        "route_identity": (
            {
                "role": route.role,
                "provider": route.provider,
                "model": route.model,
                "location": route.location,
            }
            if route is not None
            else {}
        ),
        "model_digest": model_digest,
        "result": dict(result or {}),
        "failure_class": failure_class,
    }


def _parse(payload: Mapping[str, Any]) -> tuple[str, str, str, int, Mapping[str, Any]]:
    if set(payload) != _REQUEST_FIELDS or payload.get("schema") != WORKER_SCHEMA:
        raise ValueError
    operation = _safe_operation(payload.get("operation"))
    role = _safe_role(payload.get("role"))
    request_id = _safe_request_id(payload.get("request_id"))
    deadline_ms = payload.get("deadline_ms")
    request_input = payload.get("input")
    if (
        operation == "invalid"
        or role not in _ROLE_BY_OPERATION[operation]
        or request_id == "invalid"
        or isinstance(deadline_ms, bool)
        or not isinstance(deadline_ms, int)
        or not 1_000 <= deadline_ms <= MAX_DEADLINE_MS
        or not isinstance(request_input, Mapping)
    ):
        raise ValueError
    encoded = json.dumps(
        dict(request_input), ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    if not encoded or len(encoded) > MAX_INPUT_CHARS:
        raise ValueError
    input_copy = dict(request_input)
    if operation == "teacher":
        _teacher_candidate_ids(input_copy)
    return operation, role, request_id, deadline_ms, input_copy


def _resolve_local_route(role: str) -> tuple[ollama.RuntimeGenerationRoute, str]:
    route = ollama.runtime_generation_routes((role,))[0]
    if (
        route.role != role
        or route.provider != "ollama"
        or route.location != "local"
        or not route.structured_output
    ):
        raise ValueError
    digest = ollama.model_digests([route.model]).get(route.model, "")
    if not isinstance(digest, str) or not digest:
        raise ValueError
    return route, digest


def run(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Return a fixed privacy-safe envelope; never expose raw inputs or errors."""

    operation = _safe_operation(payload.get("operation"))
    role = _safe_role(payload.get("role"))
    request_id = _safe_request_id(payload.get("request_id"))
    try:
        operation, role, request_id, deadline_ms, request_input = _parse(payload)
    except Exception:
        return _envelope(
            ok=False,
            operation=operation,
            role=role,
            request_id=request_id,
            failure_class="input_invalid",
        )
    try:
        route, digest = _resolve_local_route(role)
    except Exception:
        return _envelope(
            ok=False,
            operation=operation,
            role=role,
            request_id=request_id,
            failure_class="route_unavailable",
        )
    try:
        session = LocalStructuredSession(
            model=route.model,
            role="recall_distillation_worker",
            runtime_role=role,
            runtime_location="local",
            source_data_class="raw",
            source_sensitivity="high",
            num_ctx=32_768,
            num_predict=2_048,
            keep_alive="0",
            read_timeout_ms=deadline_ms,
            max_input_chars=MAX_INPUT_CHARS,
            max_output_chars=MAX_OUTPUT_CHARS,
            max_feedback_chars=512,
            max_responses=2,
            require_returned_model=True,
        )
        structured = session.run(
            json.dumps(
                {"operation": operation, "input": request_input},
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ),
            _schema(
                operation,
                candidate_ids=(
                    _teacher_candidate_ids(request_input)
                    if operation == "teacher"
                    else ()
                ),
            ),
            system=_system(operation),
        )
    except Exception:
        return _envelope(
            ok=False,
            operation=operation,
            role=role,
            request_id=request_id,
            route=route,
            model_digest=digest,
            failure_class="backend_error",
        )
    if (
        not structured.ok
        or not isinstance(structured.value, Mapping)
        or (
            operation == "teacher"
            and not _valid_teacher_labels(
                structured.value, _teacher_candidate_ids(request_input)
            )
        )
    ):
        return _envelope(
            ok=False,
            operation=operation,
            role=role,
            request_id=request_id,
            route=route,
            model_digest=digest,
            failure_class=(
                structured.failure_class
                or ("output_invalid" if structured.ok else "structured_failure")
            ),
        )
    return _envelope(
        ok=True,
        operation=operation,
        role=role,
        request_id=request_id,
        route=route,
        model_digest=digest,
        result=structured.value,
    )


def main() -> None:
    try:
        value = json.load(sys.stdin)
        output = run(value if isinstance(value, Mapping) else {})
    except Exception:
        output = _envelope(
            ok=False,
            operation="invalid",
            role="invalid",
            request_id="invalid",
            failure_class="input_invalid",
        )
    print(json.dumps(output, ensure_ascii=False, sort_keys=True, separators=(",", ":")))


if __name__ == "__main__":
    main()
