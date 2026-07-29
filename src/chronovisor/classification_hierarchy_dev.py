"""Development evaluation for search-free hierarchical UDC navigation."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import time
import uuid
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from chronovisor import ollama
from chronovisor.classification import ClassificationError, default_udc_package
from chronovisor.classification_fixture_set import read_jsonl, sha256_file
from chronovisor.classification_hierarchy import (
    NavigationGraph,
    build_navigation_graph,
    deterministic_evidence_capsule,
    navigation_cards,
)
from chronovisor.classification_hierarchy_worker import (
    AUDIT_SCHEMA,
    EXTRACTION_SCHEMA,
    HOLD,
    PROMPT_SHA256,
    STEP_SCHEMA,
    WORKER_SCHEMA,
)
from chronovisor.durable_state import read_sealed_json, write_sealed_json
from chronovisor.research_scheduler import (
    research_lane,
    run_cancellable_command,
    sync_pending,
)
from chronovisor.runtime_config import load_decision_router_config
from chronovisor.store import CHRONOVISOR_ROOT

EVALUATION_SCHEMA = "chronovisor.classification-hierarchy-dev.v1"
CASE_SCHEMA = "chronovisor.classification-hierarchy-case.v1"
CALL_SCHEMA = "chronovisor.classification-hierarchy-call.v1"
MAX_DEPTH = 10
MAX_BEAM = 2
_SAFE_NAME = re.compile(r"[^A-Za-z0-9._-]+")


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="milliseconds")


def _json_sha256(value: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


def output_root(root: Path, fixture_set: str) -> Path:
    return root / "classification" / f"hierarchy-navigation-v2-{fixture_set}"


def _call_path(root: Path, fixture_set: str, uid: str, call_id: str) -> Path:
    safe = _SAFE_NAME.sub("-", call_id).strip("-")[:80]
    digest = hashlib.sha256(call_id.encode()).hexdigest()[:12]
    return output_root(root, fixture_set) / "cases" / uid / "calls" / (
        f"{safe}-{digest}.json"
    )


def _call_worker(
    root: Path,
    fixture_set: str,
    uid: str,
    call_id: str,
    payload: Mapping[str, Any],
) -> dict[str, Any]:
    path = _call_path(root, fixture_set, uid, call_id)
    input_sha256 = _json_sha256(payload)
    if path.is_file():
        artifact = read_sealed_json(path)
        if (
            artifact.get("schema") != CALL_SCHEMA
            or artifact.get("input_sha256") != input_sha256
            or artifact.get("prompt_sha256") != PROMPT_SHA256
        ):
            raise ClassificationError(
                f"sealed hierarchy call contract changed: {call_id}"
            )
        return artifact
    attempts = 0
    timeout_ms = int(payload.get("read_timeout_ms") or 660_000)
    deadline = time.monotonic() + max(60.0, timeout_ms / 1_000 + 30)
    while True:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise ClassificationError(f"hierarchy call exceeded deadline: {call_id}")
        attempts += 1
        with research_lane(
            f"hierarchy-{uid[:10]}-{uuid.uuid4().hex[:8]}",
            enabled=True,
            mode="on",
            purpose="explicit",
            needs_model=True,
        ) as lease:
            result = run_cancellable_command(
                [
                    sys.executable,
                    "-m",
                    "chronovisor.classification_hierarchy_worker",
                ],
                json.dumps(dict(payload), ensure_ascii=False, sort_keys=True),
                lease,
                timeout_seconds=remaining,
            )
        if result.status in {"cancelled", "deferred"}:
            while sync_pending():
                if time.monotonic() >= deadline:
                    raise ClassificationError(
                        f"hierarchy foreground wait exceeded deadline: {call_id}"
                    )
                time.sleep(0.05)
            continue
        if result.status != "completed" or not isinstance(result.value, Mapping):
            raise ClassificationError(result.error or f"hierarchy call failed: {call_id}")
        worker = dict(result.value)
        operation = str(payload.get("operation") or "")
        expected_result_schema = {
            "extract": EXTRACTION_SCHEMA,
            "navigate": STEP_SCHEMA,
            "audit": AUDIT_SCHEMA,
        }[operation]
        worker_result = worker.get("result")
        if (
            worker.get("schema") != WORKER_SCHEMA
            or worker.get("operation") != operation
            or worker.get("model") != payload.get("model")
            or worker.get("model_digest") != payload.get("model_digest")
            or worker.get("prompt_sha256") != PROMPT_SHA256
            or int(worker.get("model_calls") or 0) != 1
            or not isinstance(worker_result, Mapping)
            or worker_result.get("schema") != expected_result_schema
        ):
            raise ClassificationError(f"hierarchy worker contract mismatch: {call_id}")
        artifact = {
            "schema": CALL_SCHEMA,
            "created_at": _now(),
            "uid": uid,
            "call_id": call_id,
            "operation": operation,
            "model": payload.get("model"),
            "model_digest": payload.get("model_digest"),
            "prompt_sha256": PROMPT_SHA256,
            "input_sha256": input_sha256,
            "attempts": attempts,
            "model_calls": 1,
            "result": dict(worker_result),
        }
        write_sealed_json(path, artifact, backup=True)
        return read_sealed_json(path)


def _model_payload(
    *,
    operation: str,
    model: str,
    model_digest: str,
    keep_alive: str,
    read_timeout_ms: int,
    page: Mapping[str, str],
    **extra: Any,
) -> dict[str, Any]:
    return {
        "schema": WORKER_SCHEMA,
        "operation": operation,
        "model": model,
        "model_digest": model_digest,
        "keep_alive": keep_alive,
        "read_timeout_ms": read_timeout_ms,
        "page": dict(page),
        **extra,
    }


def _step_signature(step: Mapping[str, Any]) -> tuple[str, tuple[str, ...]]:
    return (
        str(step.get("action") or ""),
        tuple(sorted(str(value) for value in step.get("selected_notations") or [])),
    )


def _requires_adjudication(
    first: Mapping[str, Any],
    second: Mapping[str, Any],
) -> bool:
    return (
        _step_signature(first) != _step_signature(second)
        or first.get("action") == "stop"
        or len(first.get("selected_notations") or []) != 1
    )


def _run_navigation(
    *,
    root: Path,
    fixture_set: str,
    uid: str,
    call_prefix: str,
    graph: NavigationGraph,
    capsule: Mapping[str, str],
    subject: Mapping[str, Any],
    current_path: Sequence[str],
    options: Sequence[Mapping[str, Any]],
    primary: Mapping[str, str],
    challenger: Mapping[str, str],
    read_timeout_ms: int,
    unconditional_probe: bool,
) -> tuple[dict[str, Any], list[dict[str, Any]], int]:
    forward_options = [dict(card) for card in options]
    first_artifact = _call_worker(
        root,
        fixture_set,
        uid,
        f"{call_prefix}-primary-forward",
        _model_payload(
            operation="navigate",
            page=capsule,
            subject=dict(subject),
            current_path=list(current_path),
            options=forward_options,
            prior_attempts=[],
            read_timeout_ms=read_timeout_ms,
            **primary,
        ),
    )
    attempts = [dict(first_artifact["result"])]
    first = attempts[0]
    needs_probe = (
        unconditional_probe
        or first.get("action") == "stop"
        or len(first.get("selected_notations") or []) != 1
        or bool(first.get("invalid_reason"))
    )
    if not needs_probe:
        return first, attempts, 1
    second_artifact = _call_worker(
        root,
        fixture_set,
        uid,
        f"{call_prefix}-primary-reverse",
        _model_payload(
            operation="navigate",
            page=capsule,
            subject=dict(subject),
            current_path=list(current_path),
            options=list(reversed(forward_options)),
            prior_attempts=[],
            read_timeout_ms=read_timeout_ms,
            **primary,
        ),
    )
    attempts.append(dict(second_artifact["result"]))
    if not _requires_adjudication(attempts[0], attempts[1]):
        return attempts[0], attempts, 2
    adjudication = _call_worker(
        root,
        fixture_set,
        uid,
        f"{call_prefix}-challenger-adjudication",
        _model_payload(
            operation="navigate",
            page=capsule,
            subject=dict(subject),
            current_path=list(current_path),
            options=forward_options,
            prior_attempts=attempts,
            read_timeout_ms=read_timeout_ms,
            **challenger,
        ),
    )
    attempts.append(dict(adjudication["result"]))
    return dict(adjudication["result"]), attempts, 3


def run_case(
    root: Path,
    fixture_set: str,
    page: Mapping[str, Any],
    graph: NavigationGraph,
    *,
    primary: Mapping[str, str],
    challenger: Mapping[str, str],
    auditor: Mapping[str, str],
    read_timeout_ms: int,
) -> dict[str, Any]:
    uid = str(page.get("uid") or "")
    source_sha256 = str(page.get("source_sha256") or "")
    case_path = output_root(root, fixture_set) / "cases" / uid / "result.json"
    if case_path.is_file():
        return read_sealed_json(case_path)
    if not uid or not source_sha256:
        raise ClassificationError("hierarchy dev page requires uid and source hash")
    capsule = deterministic_evidence_capsule(page)
    extract_artifact = _call_worker(
        root,
        fixture_set,
        uid,
        "subject-primary",
        _model_payload(
            operation="extract",
            page=capsule,
            read_timeout_ms=read_timeout_ms,
            **primary,
        ),
    )
    subject = dict(extract_artifact["result"])
    total_calls = 1
    root_step, root_attempts, calls = _run_navigation(
        root=root,
        fixture_set=fixture_set,
        uid=uid,
        call_prefix="root",
        graph=graph,
        capsule=capsule,
        subject=subject,
        current_path=[],
        options=navigation_cards(graph, None),
        primary=primary,
        challenger=challenger,
        read_timeout_ms=read_timeout_ms,
        unconditional_probe=True,
    )
    total_calls += calls
    active_paths = [
        [notation]
        for notation in root_step.get("selected_notations") or []
    ][:MAX_BEAM]
    final_paths: list[list[str]] = []
    navigation_events = [
        {
            "parent": None,
            "resolved_step": root_step,
            "attempts": root_attempts,
        }
    ]
    for depth in range(1, MAX_DEPTH + 1):
        if not active_paths:
            break
        next_paths: list[list[str]] = []
        for path in active_paths:
            cards = navigation_cards(graph, [path[-1]])
            if not cards:
                final_paths.append(path)
                continue
            path_token = hashlib.sha256("/".join(path).encode()).hexdigest()[:10]
            step, attempts, calls = _run_navigation(
                root=root,
                fixture_set=fixture_set,
                uid=uid,
                call_prefix=f"d{depth}-{path_token}",
                graph=graph,
                capsule=capsule,
                subject=subject,
                current_path=path,
                options=cards,
                primary=primary,
                challenger=challenger,
                read_timeout_ms=read_timeout_ms,
                unconditional_probe=False,
            )
            total_calls += calls
            navigation_events.append(
                {
                    "parent": path[-1],
                    "resolved_step": step,
                    "attempts": attempts,
                }
            )
            selected = list(step.get("selected_notations") or [])
            if step.get("action") == "stop" or not selected:
                final_paths.append(path)
            else:
                next_paths.extend([*path, notation] for notation in selected)
        deduplicated = []
        seen = set()
        for path in next_paths:
            key = tuple(path)
            if key not in seen:
                seen.add(key)
                deduplicated.append(path)
        active_paths = deduplicated[:MAX_BEAM]
    final_paths.extend(active_paths)
    deduped_final = []
    seen_final = set()
    for path in final_paths:
        key = tuple(path)
        if key not in seen_final:
            seen_final.add(key)
            deduped_final.append(path)
    final_paths = deduped_final[:MAX_BEAM]
    if not final_paths:
        audit = {
            "schema": AUDIT_SCHEMA,
            "selected_notation": HOLD,
            "rationale": "Root navigation stopped after mandatory stability probing.",
            "invalid_reason": "root_stop",
        }
    else:
        allowed_notations = list(
            dict.fromkeys(notation for path in final_paths for notation in path)
        )
        explored_paths = [
            [
                graph.by_notation(notation).card()
                for notation in path
                if graph.by_notation(notation) is not None
            ]
            for path in final_paths
        ]
        audit_artifact = _call_worker(
            root,
            fixture_set,
            uid,
            "final-audit",
            _model_payload(
                operation="audit",
                page=capsule,
                subject=subject,
                explored_paths=explored_paths,
                allowed_notations=allowed_notations,
                root_attempts=root_attempts,
                read_timeout_ms=read_timeout_ms,
                **auditor,
            ),
        )
        total_calls += 1
        audit = dict(audit_artifact["result"])
    result = {
        "schema": CASE_SCHEMA,
        "created_at": _now(),
        "uid": uid,
        "source_sha256": source_sha256,
        "graph_schema": graph.schema,
        "graph_release": graph.release,
        "graph_checksum": graph.checksum,
        "prompt_sha256": PROMPT_SHA256,
        "capsule": capsule,
        "subject": subject,
        "root_attempts": root_attempts,
        "navigation_events": navigation_events,
        "final_paths": final_paths,
        "audit": audit,
        "selected_notation": audit["selected_notation"],
        "model_calls": total_calls,
        "page_mutations": 0,
    }
    write_sealed_json(case_path, result, backup=True)
    return read_sealed_json(case_path)


def _root_notation(graph: NavigationGraph, notation: str) -> str:
    ancestors = graph.ancestors(notation)
    return ancestors[0].notation if ancestors else ""


def score_selection(
    graph: NavigationGraph,
    selected: str,
    expected: Sequence[str],
) -> dict[str, Any]:
    expected_nodes = [
        value for value in expected if graph.by_notation(value) is not None
    ]
    if selected == HOLD:
        relation = "hold"
    elif selected in expected_nodes:
        relation = "exact"
    elif any(graph.is_ancestor(selected, value) for value in expected_nodes):
        relation = "ancestor"
    elif any(graph.is_ancestor(value, selected) for value in expected_nodes):
        relation = "overdeep-descendant"
    else:
        selected_node = graph.by_notation(selected)
        sibling = bool(
            selected_node
            and any(
                graph.by_notation(value)
                and graph.by_notation(value).parent_uri == selected_node.parent_uri
                for value in expected_nodes
            )
        )
        if sibling:
            relation = "same-parent-sibling"
        elif selected_node and any(
            _root_notation(graph, value) == _root_notation(graph, selected)
            for value in expected_nodes
        ):
            relation = "same-root-cross-branch"
        else:
            relation = "catastrophic-root"
    return {
        "relation": relation,
        "accepted": relation in {"exact", "ancestor"},
        "exact": relation == "exact",
        "ancestor": relation == "ancestor",
        "held": relation == "hold",
        "same_parent_sibling": relation == "same-parent-sibling",
        "catastrophic": relation == "catastrophic-root",
    }


def _candidate_rows(root: Path) -> dict[str, dict[str, Any]]:
    path = (
        root
        / "classification"
        / "fixtures"
        / "epochs"
        / "epoch-3-library-evidence-v1"
        / "candidates.jsonl"
    )
    return {
        str(row.get("uid") or ""): row
        for row in read_jsonl(path)
        if str(row.get("uid") or "")
    }


def load_dev_cases(root: Path, fixture_set: str) -> list[dict[str, Any]]:
    if fixture_set == "early10":
        evaluation = read_sealed_json(
            root / "classification" / "query2doc-pilot" / "evaluation.json"
        )
        candidates = _candidate_rows(root)
        output = []
        for row in evaluation.get("cases") or []:
            uid = str(row.get("uid") or "")
            page = dict(candidates[uid])
            page["expected_primary_notations"] = list(
                row.get("expected_primary_notations") or []
            )
            output.append(page)
        return output
    if fixture_set == "opened30":
        fixture = read_sealed_json(
            root / "classification" / "query2doc-v2-unseen" / "fixture.json"
        )
        return [
            dict(case)
            for case in fixture.get("cases") or []
            if isinstance(case, Mapping)
        ]
    raise ClassificationError(f"unsupported hierarchy dev fixture: {fixture_set}")


def run_dev(root: Path, fixture_set: str) -> dict[str, Any]:
    destination = output_root(root, fixture_set)
    evaluation_path = destination / "evaluation.json"
    if evaluation_path.is_file():
        return read_sealed_json(evaluation_path)
    pages = load_dev_cases(root, fixture_set)
    config = load_decision_router_config()
    model_specs = {
        "primary": {
            "model": config.primary_model,
            "keep_alive": config.primary_keep_alive,
        },
        "challenger": {
            "model": config.tie_break_model,
            "keep_alive": config.tie_break_keep_alive,
        },
        "auditor": {
            "model": config.tie_break_model,
            "keep_alive": config.tie_break_keep_alive,
        },
    }
    digests = ollama.model_digests(
        sorted({str(spec["model"]) for spec in model_specs.values()})
    )
    for spec in model_specs.values():
        spec["model_digest"] = digests.get(str(spec["model"]), "")
        if not spec["model_digest"]:
            raise ClassificationError(
                f"hierarchy model is unavailable: {spec['model']}"
            )
    graph = build_navigation_graph(default_udc_package())
    cases = []
    model_calls = 0
    for page in pages:
        result = run_case(
            root,
            fixture_set,
            page,
            graph,
            primary=model_specs["primary"],
            challenger=model_specs["challenger"],
            auditor=model_specs["auditor"],
            read_timeout_ms=config.read_timeout_ms,
        )
        expected = [
            str(value)
            for value in page.get("expected_primary_notations") or []
        ]
        score = score_selection(
            graph,
            str(result.get("selected_notation") or HOLD),
            expected,
        )
        model_calls += int(result.get("model_calls") or 0)
        cases.append(
            {
                "uid": str(page.get("uid") or ""),
                "title": str(page.get("title") or ""),
                "expected_primary_notations": expected,
                "selected_notation": result.get("selected_notation"),
                "subject": result.get("subject"),
                "final_paths": result.get("final_paths"),
                "audit": result.get("audit"),
                "model_calls": result.get("model_calls"),
                **score,
            }
        )
    metrics = {
        "case_count": len(cases),
        "accepted": sum(bool(case["accepted"]) for case in cases),
        "exact": sum(bool(case["exact"]) for case in cases),
        "ancestor": sum(bool(case["ancestor"]) for case in cases),
        "holds": sum(bool(case["held"]) for case in cases),
        "same_parent_sibling": sum(
            bool(case["same_parent_sibling"]) for case in cases
        ),
        "catastrophic": sum(bool(case["catastrophic"]) for case in cases),
        "average_selected_depth": round(
            sum(
                len(graph.ancestors(str(case["selected_notation"])))
                for case in cases
                if case["selected_notation"] != HOLD
            )
            / max(1, sum(case["selected_notation"] != HOLD for case in cases)),
            3,
        ),
    }
    minimum_accepted = 7 if fixture_set == "early10" else 24
    maximum_holds = 3 if fixture_set == "early10" else 6
    passed = (
        metrics["accepted"] >= minimum_accepted
        and metrics["holds"] <= maximum_holds
        and metrics["catastrophic"] == 0
    )
    evaluation = {
        "schema": EVALUATION_SCHEMA,
        "evaluated_at": _now(),
        "fixture_set": fixture_set,
        "graph": {
            "schema": graph.schema,
            "release": graph.release,
            "checksum": graph.checksum,
            "node_count": len(graph.nodes),
            "contracted_parent_count": graph.contracted_parent_count,
        },
        "models": model_specs,
        "prompt_sha256": PROMPT_SHA256,
        "model_calls": model_calls,
        "page_mutations": 0,
        "gate": {
            "minimum_accepted": minimum_accepted,
            "maximum_holds": maximum_holds,
            "maximum_catastrophic": 0,
        },
        "metrics": metrics,
        "cases": cases,
        "decision": (
            "qualify-hierarchy-navigation-dev"
            if passed
            else "reject-hierarchy-navigation-dev"
        ),
        "next_fixture_authorized": passed,
    }
    write_sealed_json(evaluation_path, evaluation, backup=True)
    return read_sealed_json(evaluation_path)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=CHRONOVISOR_ROOT)
    parser.add_argument(
        "--fixture-set",
        choices=("early10", "opened30"),
        default="early10",
    )
    args = parser.parse_args(argv)
    try:
        print(
            json.dumps(
                run_dev(args.root.expanduser(), args.fixture_set),
                ensure_ascii=False,
                sort_keys=True,
            )
        )
        return 0
    except (
        ClassificationError,
        OSError,
        TypeError,
        ValueError,
        json.JSONDecodeError,
    ) as exc:
        print(str(exc), file=sys.stderr)
        return 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
