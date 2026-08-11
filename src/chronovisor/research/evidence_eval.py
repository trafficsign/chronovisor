"""Deterministic paired replay and live acceptance for Campaign Y."""

from __future__ import annotations

import hashlib
import json
import re
import time
from collections.abc import Callable, Mapping, Sequence
from datetime import datetime
from fractions import Fraction
from pathlib import Path
from typing import Any
from unittest.mock import patch

from chronovisor.core import canonical_json, durable_state, store
from chronovisor.research.evidence_reconstruction import (
    EVALUATION_CONTRACT,
    EVALUATION_CONTRACT_SHA256,
    EvidenceReconstructionError,
    build_episode_projection,
    build_evidence_packet,
    committed_raw_watermark,
    compile_retrieval_program,
    evaluation_contract_bytes,
    load_episode_projection,
    physical_raw_inventory,
)

canonical_json_line_bytes_strict = canonical_json.canonical_json_line_bytes_strict
canonical_json_sha256_strict = canonical_json.canonical_json_sha256_strict
atomic_write_bytes_at = durable_state.atomic_write_bytes_at
okf_runtime_operation = store.okf_runtime_operation
okf_startup_status = store.okf_startup_status

PAIRED_EVALUATION_SCHEMA = "chronovisor.evidence-paired-evaluation.v1"
EVIDENCE_CASES_SCHEMA = "chronovisor.evidence-cases.v1"
_ATOM_ID_RE = re.compile(r"atom:[0-9a-f]{64}")
_CASE_FIELDS = {
    "case_id",
    "slice",
    "query",
    "as_of",
    "expected_answerable",
    "expected_answer_text",
    "expected_page_ids",
    "expected_atom_ids",
    "forbidden_obsolete_page_ids",
    "forbidden_obsolete_atom_ids",
    "relation_expectation",
}


def bind_recall_provider(bind: Callable[..., None]) -> None:
    """Compose the research implementation through the recall-owned port."""

    from chronovisor.research.evidence_runtime import (
        evidence_publication_payload,
        observe_projection_evidence,
    )

    bind(observe_projection_evidence, evidence_publication_payload)


def seal_paired_case(case: Mapping[str, Any]) -> dict[str, Any]:
    if set(case) != _CASE_FIELDS:
        raise EvidenceReconstructionError("paired case fields are invalid")
    payload = dict(case)
    return {**payload, "case_sha256": canonical_json_sha256_strict(payload)}


def _sealed_cases(cases: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    from chronovisor.research.evidence_runtime import (
        evidence_relation_semantics_proof,
    )

    slices = set(EVALUATION_CONTRACT.paired_slices)
    relation_proof = evidence_relation_semantics_proof()
    normalized: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    for index, value in enumerate(cases):
        if set(value) != _CASE_FIELDS | {"case_sha256"}:
            raise EvidenceReconstructionError(f"paired case {index} fields are invalid")
        case_id = value["case_id"]
        slice_name = value["slice"]
        query = value["query"]
        as_of = value["as_of"]
        answerable = value["expected_answerable"]
        answer_text = value["expected_answer_text"]
        page_ids = value["expected_page_ids"]
        atom_ids = value["expected_atom_ids"]
        forbidden_pages = value["forbidden_obsolete_page_ids"]
        forbidden_atoms = value["forbidden_obsolete_atom_ids"]
        relation_expectation = value["relation_expectation"]
        try:
            parsed_as_of = datetime.fromisoformat(str(as_of).replace("Z", "+00:00"))
        except ValueError as exc:
            raise EvidenceReconstructionError(
                f"paired case {index} as_of is invalid"
            ) from exc
        lists = (page_ids, atom_ids, forbidden_pages, forbidden_atoms)
        if (
            not isinstance(case_id, str)
            or not case_id
            or case_id in seen_ids
            or slice_name not in slices
            or not isinstance(query, str)
            or not query.strip()
            or not isinstance(as_of, str)
            or parsed_as_of.tzinfo is None
            or not isinstance(answerable, bool)
            or answerable != (slice_name not in {"contradiction", "no-answer"})
            or not isinstance(answer_text, str)
            or bool(answer_text.strip()) != answerable
            or any(not isinstance(row, list) for row in lists)
            or any(
                any(not isinstance(item, str) or not item for item in row)
                or len(row) != len(set(row))
                for row in lists
            )
            or any(_ATOM_ID_RE.fullmatch(item) is None for item in atom_ids)
            or any(_ATOM_ID_RE.fullmatch(item) is None for item in forbidden_atoms)
            or bool(page_ids) != answerable
            or bool(atom_ids) != answerable
            or (not answerable and (forbidden_pages or forbidden_atoms))
            or relation_expectation
            != (relation_proof if slice_name == "contradiction" else None)
            or value["case_sha256"]
            != canonical_json_sha256_strict({key: value[key] for key in _CASE_FIELDS})
        ):
            raise EvidenceReconstructionError(f"paired case {index} is invalid")
        seen_ids.add(case_id)
        normalized.append(dict(value))
    if len(normalized) != len(slices) or {row["slice"] for row in normalized} != slices:
        raise EvidenceReconstructionError(
            "paired replay must contain exactly one case for every sealed slice"
        )
    if len({str(row["query"]) for row in normalized}) != len(normalized):
        raise EvidenceReconstructionError("paired replay queries must be distinct")
    return sorted(normalized, key=lambda row: row["case_id"])


def _finalized(root: Path) -> bool:
    decision = okf_startup_status(root)
    return bool(
        decision.allowed
        and decision.layout == "okf_v0_2"
        and decision.state == "finalized-v2"
    )


def register_evidence_cases(
    *, root: Path, cases: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    """Seal the operator-selected seven live cases once, before acceptance."""

    from chronovisor.research.evidence_runtime import (
        evidence_authority_entry_exists,
        evidence_authority_operation,
        write_evidence_authority_at,
    )

    if not _finalized(root):
        raise EvidenceReconstructionError("Campaign X is not finalized")
    normalized = _sealed_cases(cases)
    with okf_runtime_operation(root) as locked:
        if not (
            locked.allowed
            and locked.layout == "okf_v0_2"
            and locked.state == "finalized-v2"
        ):
            raise EvidenceReconstructionError("Campaign X changed during registration")
        with evidence_authority_operation(root) as directory_fd:
            if any(
                evidence_authority_entry_exists(directory_fd, name)
                for name in ("cases.json", "acceptance.json", "promotion.json")
            ):
                raise EvidenceReconstructionError(
                    "evidence cases are already registered"
                )
            return write_evidence_authority_at(
                directory_fd,
                "cases.json",
                {
                    "schema": EVIDENCE_CASES_SCHEMA,
                    "contract_sha256": EVALUATION_CONTRACT_SHA256,
                    "case_manifest_sha256": canonical_json_sha256_strict(normalized),
                    "cases": normalized,
                },
            )


def _load_registered_cases(root: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    from chronovisor.research.evidence_runtime import (
        evidence_cases_path,
        read_evidence_authority,
    )

    try:
        payload = read_evidence_authority(evidence_cases_path(root))
    except Exception as exc:
        raise EvidenceReconstructionError(
            "registered evidence cases are invalid"
        ) from exc
    if set(payload) != {
        "schema",
        "contract_sha256",
        "case_manifest_sha256",
        "cases",
        "seal_sha256",
    } or not isinstance(payload.get("cases"), list):
        raise EvidenceReconstructionError("registered evidence cases are invalid")
    normalized = _sealed_cases(payload["cases"])
    if (
        payload["schema"] != EVIDENCE_CASES_SCHEMA
        or payload["contract_sha256"] != EVALUATION_CONTRACT_SHA256
        or payload["cases"] != normalized
        or payload["case_manifest_sha256"] != canonical_json_sha256_strict(normalized)
    ):
        raise EvidenceReconstructionError("registered evidence cases are invalid")
    return normalized, payload


def _output_text(raw: bytes, field: str) -> str:
    if not isinstance(raw, bytes):
        raise EvidenceReconstructionError(f"{field} must be bytes")
    try:
        return raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise EvidenceReconstructionError(f"{field} must be UTF-8") from exc


def _answer_score(case: Mapping[str, Any], output: bytes, abstained: bool) -> float:
    text = _output_text(output, "rendered output").casefold()
    if not case["expected_answerable"]:
        return float(abstained and not text.strip())
    return float(not abstained and str(case["expected_answer_text"]).casefold() in text)


def _baseline_scores(
    case: Mapping[str, Any], result: Any, output: bytes
) -> dict[str, float]:
    if any(
        not hasattr(result, field)
        for field in ("queries", "context", "context_items", "latency_ms")
    ):
        raise EvidenceReconstructionError("paired baseline result is invalid")
    if (
        case["query"] not in result.queries
        or output != result.context.encode("utf-8")
        or isinstance(result.latency_ms, bool)
        or not isinstance(result.latency_ms, int)
        or result.latency_ms < 0
    ):
        raise EvidenceReconstructionError(
            "paired baseline query/output binding is invalid"
        )
    page_ids = {item.page_id for item in result.context_items}
    expected = set(case["expected_page_ids"])
    forbidden = set(case["forbidden_obsolete_page_ids"])
    abstained = not result.context_items
    if case["expected_answerable"]:
        as_of = datetime.fromisoformat(str(case["as_of"]).replace("Z", "+00:00"))
        temporal = True
        for item in result.context_items:
            if item.page_id not in expected:
                continue
            try:
                observed = datetime.fromisoformat(item.updated.replace("Z", "+00:00"))
                if observed.tzinfo is None:
                    observed = observed.replace(tzinfo=as_of.tzinfo)
            except ValueError:
                temporal = False
                break
            temporal = temporal and observed <= as_of
        evidence = expected.issubset(page_ids)
        obsolete = not bool(page_ids.intersection(forbidden))
    else:
        evidence = temporal = obsolete = abstained and not output
    return {
        "answer": _answer_score(case, output, abstained),
        "evidence": float(evidence),
        "temporal": float(temporal),
        "obsolete-use": float(obsolete),
        "latency": float(result.latency_ms <= 4_000),
    }


def _candidate_scores(
    case: Mapping[str, Any], run: Any, output: bytes, measured_latency_ms: int
) -> dict[str, float]:
    from chronovisor.research.evidence_runtime import (
        EvidenceLedger,
        EvidenceRun,
        evidence_relation_semantics_proof,
    )

    if not isinstance(run, EvidenceRun):
        raise EvidenceReconstructionError("paired candidate run is invalid")
    packet = run.packet
    rebuilt_packet = build_evidence_packet(
        query=packet.query,
        as_of=packet.as_of,
        retrieval_program_id=packet.retrieval_program_id,
        atoms=packet.atoms,
        abstention_reason=packet.abstention_reason,
    )
    trace = run.trace
    program = trace.get("program") if isinstance(trace, Mapping) else None
    canonical_program = None
    if isinstance(program, Mapping) and set(program) == {
        "schema",
        "program_id",
        "query",
        "as_of",
        "claim_slots",
        "required_evidence",
        "allowed_actions",
        "stop_rules",
    }:
        try:
            canonical_program = compile_retrieval_program(
                program["query"],
                {
                    "as_of": program["as_of"],
                    "claim_slots": program["claim_slots"],
                    "required_evidence": program["required_evidence"],
                    "allowed_actions": program["allowed_actions"],
                    "stop_rules": program["stop_rules"],
                },
            )
        except (KeyError, EvidenceReconstructionError):
            pass
    if (
        rebuilt_packet != packet
        or canonical_program is None
        or canonical_program.to_dict() != program
        or canonical_program.program_id != packet.retrieval_program_id
        or canonical_program.query != packet.query
        or canonical_program.as_of != packet.as_of
        or packet.query != case["query"]
        or packet.as_of != case["as_of"]
        or trace.get("stop_reason") != run.stop_reason
        or isinstance(measured_latency_ms, bool)
        or not isinstance(measured_latency_ms, int)
        or measured_latency_ms < 0
    ):
        raise EvidenceReconstructionError("paired candidate identity is invalid")
    rendered = _output_text(output, "candidate rendered output")
    output_bound = packet.abstained and not output
    if not packet.abstained and not output:
        output_bound = True
    elif not packet.abstained:
        marker = "payload_json=\n"
        try:
            encoded = rendered.split(marker, 1)[1].rsplit("\n[/RECALL_CONTEXT]", 1)[0]
            payload = json.loads(encoded)
            published_trace = payload["trace"]["evidence_trace"]
            published_trace_sha = payload["trace"]["evidence_trace_sha256"]
            output_bound = bool(
                payload.get("evidence_packet") == json.loads(packet.canonical_bytes())
                and published_trace == trace
                and published_trace_sha == canonical_json_sha256_strict(trace)
            )
        except (IndexError, KeyError, TypeError, json.JSONDecodeError):
            output_bound = False
    if not output_bound:
        raise EvidenceReconstructionError("paired candidate output binding is invalid")
    atom_ids = {atom.atom_id for atom in packet.atoms}
    expected = set(case["expected_atom_ids"])
    forbidden = set(case["forbidden_obsolete_atom_ids"])
    if case["expected_answerable"]:
        ledger = EvidenceLedger(canonical_program)
        for slot in canonical_program.claim_slots:
            for atom in packet.atoms:
                ledger.add(slot.slot_id, atom)
        evidence = (
            not packet.abstained and expected.issubset(atom_ids) and ledger.covered()
        )
        temporal = (
            not packet.abstained
            and ledger.covered()
            and not ledger.temporal_gap()
        )
        obsolete = not bool(
            atom_ids.intersection(forbidden)
        ) and not ledger.unresolved_contradiction()
    elif case["slice"] == "contradiction":
        safe_abstention = packet.abstained and not output
        proof_valid = (
            case["relation_expectation"] == evidence_relation_semantics_proof()
        )
        evidence = temporal = obsolete = safe_abstention and proof_valid
    else:
        evidence = temporal = obsolete = packet.abstained and not output
    return {
        "answer": _answer_score(case, output, packet.abstained),
        "evidence": float(evidence),
        "temporal": float(temporal),
        "obsolete-use": float(obsolete),
        "latency": float(measured_latency_ms <= 4_000),
        "context": float(bool(output) if case["expected_answerable"] else not output),
    }


def evaluate_paired_replay(
    cases: Sequence[Mapping[str, Any]],
    *,
    baseline_results: Mapping[str, Any],
    baseline_outputs: Mapping[str, bytes],
    candidate_runs: Mapping[str, Any],
    candidate_outputs: Mapping[str, bytes],
    candidate_latency_ms: Mapping[str, int],
) -> dict[str, Any]:
    """Derive all metrics from sealed cases and actual paired arm objects."""

    evaluation_contract_bytes()
    normalized_cases = _sealed_cases(cases)
    case_ids = {str(row["case_id"]) for row in normalized_cases}
    if any(
        set(values) != case_ids
        for values in (
            baseline_results,
            baseline_outputs,
            candidate_runs,
            candidate_outputs,
            candidate_latency_ms,
        )
    ):
        raise EvidenceReconstructionError("paired actual arm cases are incomplete")
    rows: list[dict[str, Any]] = []
    for case in normalized_cases:
        case_id = str(case["case_id"])
        baseline_output = baseline_outputs[case_id]
        candidate_output = candidate_outputs[case_id]
        baseline = _baseline_scores(case, baseline_results[case_id], baseline_output)
        candidate = _candidate_scores(
            case,
            candidate_runs[case_id],
            candidate_output,
            candidate_latency_ms[case_id],
        )
        rows.append(
            {
                **case,
                "baseline": baseline,
                "candidate": candidate,
                "baseline_output_sha256": hashlib.sha256(baseline_output).hexdigest(),
                "candidate_output_sha256": hashlib.sha256(candidate_output).hexdigest(),
                "candidate_packet_sha256": candidate_runs[
                    case_id
                ].packet.packet_id.removeprefix("packet:"),
            }
        )
    gates: dict[str, bool] = {}
    slice_gates: dict[str, bool] = {}
    observations: dict[str, dict[str, Any]] = {}
    slice_observations: dict[str, dict[str, dict[str, Any]]] = {}
    for gate in EVALUATION_CONTRACT.metrics:
        values = [
            Fraction(str(row["candidate"][gate.metric]))
            if gate.metric == "latency"
            else Fraction(str(row["candidate"][gate.metric]))
            - Fraction(str(row["baseline"][gate.metric]))
            for row in rows
        ]
        observed = sum(values, Fraction()) / len(values)
        passed = observed >= Fraction(str(gate.lower_bound))
        gates[gate.metric] = passed
        observations[gate.metric] = {
            "measure": gate.measure,
            "observed": float(observed),
            "lower_bound": gate.lower_bound,
            "passed": passed,
        }
        slice_observations[gate.metric] = {}
        for row, observed_slice in zip(rows, values, strict=True):
            slice_passed = observed_slice >= Fraction(str(gate.lower_bound))
            slice_name = str(row["slice"])
            slice_gates[f"{gate.metric}:{slice_name}"] = slice_passed
            slice_observations[gate.metric][slice_name] = {
                "observed": float(observed_slice),
                "lower_bound": gate.lower_bound,
                "passed": slice_passed,
            }
    packet_ids = {run.packet.packet_id for run in candidate_runs.values()}
    program_ids = {run.packet.retrieval_program_id for run in candidate_runs.values()}
    if len(packet_ids) != len(rows) or len(program_ids) != len(rows):
        raise EvidenceReconstructionError("paired candidate runs must be distinct")
    unanswerable = [row for row in rows if not row["expected_answerable"]]
    safe_abstention = bool(unanswerable) and all(
        row["candidate"]["answer"] == 1.0 for row in unanswerable
    )
    answerable_execution = all(
        row["candidate"]["answer"] == 1.0 for row in rows if row["expected_answerable"]
    )
    bounded_context = all(
        row["candidate"]["context"] == 1.0 for row in rows if row["expected_answerable"]
    )
    sealed = {"contract_sha256": EVALUATION_CONTRACT_SHA256, "rows": rows}
    return {
        "schema": PAIRED_EVALUATION_SCHEMA,
        "evaluation_sha256": canonical_json_sha256_strict(sealed),
        "contract_sha256": EVALUATION_CONTRACT_SHA256,
        "case_manifest_sha256": canonical_json_sha256_strict(normalized_cases),
        "case_count": len(rows),
        "slice_counts": {
            name: sum(row["slice"] == name for row in rows)
            for name in sorted(EVALUATION_CONTRACT.paired_slices)
        },
        "metrics": observations,
        "gates": gates,
        "slice_metrics": slice_observations,
        "slice_gates": slice_gates,
        "abstention": {"passed": safe_abstention, "unanswerable_count": 2},
        "answerable_execution": {
            "passed": answerable_execution,
            "count": len(rows) - 2,
        },
        "bounded_context": {"passed": bounded_context, "count": len(rows) - 2},
        "all_pass": (
            all(gates.values())
            and all(slice_gates.values())
            and safe_abstention
            and answerable_execution
            and bounded_context
        ),
        "rows": rows,
        "external_backend_call_count": 0,
        "external_model_call_count": 0,
        "cloud_call_count": 0,
    }


def paired_evaluation_bytes(
    cases: Sequence[Mapping[str, Any]],
    **actual_arms: Any,
) -> bytes:
    return canonical_json_line_bytes_strict(
        evaluate_paired_replay(cases, **actual_arms)
    )


def _atomic_publication_fault_proof(directory_fd: int, before: bytes) -> str:
    failed = False
    try:
        with patch(
            "chronovisor.core.durable_state.os.replace",
            side_effect=OSError("injected before atomic replace"),
        ):
            atomic_write_bytes_at(
                directory_fd,
                "episode-projection.json",
                before + b"fault",
            )
    except OSError:
        failed = True
    from chronovisor.research.evidence_runtime import read_evidence_bytes_at

    after = read_evidence_bytes_at(directory_fd, "episode-projection.json")
    if not failed or after != before:
        raise EvidenceReconstructionError("projection changed during fault proof")
    return canonical_json_sha256_strict(
        {
            "fault_stage": "before_replace",
            "preserved": True,
            "projection_sha256": hashlib.sha256(after).hexdigest(),
        }
    )


def _run_evidence_acceptance(
    *,
    root: Path,
    page_teacher: Callable[[str], Any],
    candidate_renderer: Callable[[Any, Any], bytes],
) -> dict[str, Any]:
    """Run with trusted host adapters, never caller-controlled external input.

    Same-process hostile imports are outside this internal underscore boundary.
    """
    from chronovisor.research.evidence_runtime import (
        EVIDENCE_ACCEPTANCE_SCHEMA,
        advance_evidence_rollout_at,
        begin_evidence_refresh_at,
        compile_projection_program,
        evidence_authority_operation,
        evidence_cases_path,
        evidence_relation_semantics_sha256,
        evidence_rollout_gate_keys,
        evidence_selected,
        load_evidence_acceptance,
        load_evidence_rollout,
        raw_stat_watermark,
        read_evidence_authority,
        read_evidence_bytes_at,
        run_evidence_retrieval,
        run_projection_cycle_at,
        write_evidence_authority_at,
    )

    if not _finalized(root):
        raise EvidenceReconstructionError("Campaign X is not finalized")
    raw_dir = root / "raw"
    with okf_runtime_operation(root) as locked:
        if not (
            locked.allowed
            and locked.layout == "okf_v0_2"
            and locked.state == "finalized-v2"
        ):
            raise EvidenceReconstructionError("Campaign X changed during acceptance")
        with evidence_authority_operation(root) as directory_fd:
            cases, registered = _load_registered_cases(root)
            begin_evidence_refresh_at(directory_fd)
            raw_before = physical_raw_inventory(raw_dir)
            projection = run_projection_cycle_at(
                raw_dir=raw_dir, directory_fd=directory_fd
            )
            rebuilt = build_episode_projection(raw_dir)
            deterministic = rebuilt == projection.canonical_bytes()
            fault_sha = _atomic_publication_fault_proof(
                directory_fd, projection.canonical_bytes()
            )
            relation_sha = evidence_relation_semantics_sha256()
            projection_sha = projection.projection_id.removeprefix("projection:")
            raw_stat_before = raw_stat_watermark(raw_dir)
            provisional = write_evidence_authority_at(
                directory_fd,
                "acceptance.json",
                {
                    "schema": EVIDENCE_ACCEPTANCE_SCHEMA,
                    "contract_sha256": EVALUATION_CONTRACT_SHA256,
                    "projection_sha256": projection_sha,
                    "raw_watermark_sha256": committed_raw_watermark(raw_dir),
                    "raw_before_sha256": raw_before["inventory_sha256"],
                    "raw_after_sha256": raw_before["inventory_sha256"],
                    "raw_stat_sha256": raw_stat_before,
                    "evaluation_sha256": "0" * 64,
                    "case_manifest_sha256": registered["case_manifest_sha256"],
                    "case_count": len(cases),
                    "gates": {
                        key: key == "raw_unchanged"
                        for key in sorted(evidence_rollout_gate_keys())
                    },
                    "atomic_publication_fault_sha256": fault_sha,
                    "relation_semantics_sha256": relation_sha,
                    "raw_relation_mode": "supports-only",
                },
            )
            projection_atom_ids = {atom.atom_id for atom in projection.atoms}
            baseline_results: dict[str, Any] = {}
            baseline_outputs: dict[str, bytes] = {}
            candidate_runs: dict[str, Any] = {}
            candidate_outputs: dict[str, bytes] = {}
            candidate_latency_ms: dict[str, int] = {}
            for case in cases:
                case_id = str(case["case_id"])
                if not set(case["expected_atom_ids"]).issubset(projection_atom_ids):
                    raise EvidenceReconstructionError(
                        "paired case atom is not in current projection"
                    )
                baseline = page_teacher(str(case["query"]))
                if getattr(baseline, "evidence_packet", None) is not None:
                    raise EvidenceReconstructionError(
                        "paired baseline is not page teacher"
                    )
                started = time.monotonic_ns()
                if not _finalized(root):
                    raise EvidenceReconstructionError(
                        "Campaign X changed during replay"
                    )
                if evidence_selected(
                    root, "acceptance-probe", projection.projection_id
                ):
                    raise EvidenceReconstructionError(
                        "refresh unexpectedly selected"
                    )
                if load_evidence_acceptance(root).get(
                    "seal_sha256"
                ) != provisional.get("seal_sha256"):
                    raise EvidenceReconstructionError(
                        "provisional acceptance changed"
                    )
                observed_projection = load_episode_projection(
                    read_evidence_bytes_at(directory_fd, "episode-projection.json")
                )
                if observed_projection.projection_id != projection.projection_id:
                    raise EvidenceReconstructionError("observer projection changed")
                current_cases = read_evidence_authority(evidence_cases_path(root))
                if current_cases.get("seal_sha256") != registered.get("seal_sha256"):
                    raise EvidenceReconstructionError(
                        "registered case authority changed"
                    )
                if load_evidence_rollout(root).get("mode") != "shadow":
                    raise EvidenceReconstructionError("refresh is not fail closed")
                run = run_evidence_retrieval(
                    compile_projection_program(
                        str(case["query"]), str(case["as_of"])
                    ),
                    observed_projection,
                )
                if run.telemetry.get("projection_sha256") != projection_sha:
                    raise EvidenceReconstructionError(
                        "paired candidate projection binding is invalid"
                    )
                baseline_results[case_id] = baseline
                baseline_outputs[case_id] = baseline.context.encode("utf-8")
                candidate_runs[case_id] = run
                candidate_outputs[case_id] = candidate_renderer(baseline, run)
                elapsed_ms = max(
                    0, (time.monotonic_ns() - started) // 1_000_000
                )
                candidate_latency_ms[case_id] = elapsed_ms
            evaluation = evaluate_paired_replay(
                cases,
                baseline_results=baseline_results,
                baseline_outputs=baseline_outputs,
                candidate_runs=candidate_runs,
                candidate_outputs=candidate_outputs,
                candidate_latency_ms=candidate_latency_ms,
            )
            if (
                evaluation["case_manifest_sha256"]
                != registered["case_manifest_sha256"]
            ):
                raise EvidenceReconstructionError(
                    "registered case manifest changed"
                )
            raw_after = physical_raw_inventory(raw_dir)
            raw_stat = raw_stat_watermark(raw_dir)
            raw_unchanged = raw_before == raw_after
            cloud_egress_zero = all(
                run.trace.get("actions") == []
                and run.telemetry.get("cloud_call_count") == 0
                and run.telemetry.get("external_backend_call_count") == 0
                and run.telemetry.get("external_model_call_count") == 0
                for run in candidate_runs.values()
            )
            gates = {
                **evaluation["gates"],
                **{
                    f"slice:{key}": value
                    for key, value in evaluation["slice_gates"].items()
                },
                "safe_abstention": evaluation["abstention"]["passed"],
                "answerable_execution": evaluation["answerable_execution"][
                    "passed"
                ],
                "bounded_context": evaluation["bounded_context"]["passed"],
                "cloud_egress_zero": cloud_egress_zero,
                "projection_deterministic": deterministic,
                "raw_unchanged": raw_unchanged,
                "atomic_publication_fault": bool(fault_sha),
                "relation_semantics": bool(relation_sha),
                "activation:okf_finalized": True,
            }
            if set(gates) != evidence_rollout_gate_keys():
                raise EvidenceReconstructionError("acceptance gate keys are invalid")
            receipt = write_evidence_authority_at(
                directory_fd,
                "acceptance.json",
                {
                    "schema": EVIDENCE_ACCEPTANCE_SCHEMA,
                    "contract_sha256": EVALUATION_CONTRACT_SHA256,
                    "projection_sha256": projection_sha,
                    "raw_watermark_sha256": committed_raw_watermark(raw_dir),
                    "raw_before_sha256": raw_before["inventory_sha256"],
                    "raw_after_sha256": raw_after["inventory_sha256"],
                    "raw_stat_sha256": raw_stat,
                    "evaluation_sha256": evaluation["evaluation_sha256"],
                    "case_manifest_sha256": evaluation["case_manifest_sha256"],
                    "case_count": evaluation["case_count"],
                    "gates": gates,
                    "atomic_publication_fault_sha256": fault_sha,
                    "relation_semantics_sha256": relation_sha,
                    "raw_relation_mode": "supports-only",
                },
            )
            rollout = advance_evidence_rollout_at(
                root=root, directory_fd=directory_fd
            )
    return {
        "status": "passed" if all(gates.values()) else "held",
        "projection": {
            "projection_sha256": projection_sha,
            "byte_count": len(rebuilt),
            "receipt_count": len(projection.source_receipts),
            "atom_count": len(projection.atoms),
            "deterministic": deterministic,
        },
        "raw_inventory": {"before": raw_before, "after": raw_after},
        "evaluation": evaluation,
        "acceptance_receipt": receipt,
        "rollout": rollout,
        "cloud_call_count": 0,
        "external_backend_call_count": 0,
        "external_model_call_count": 0,
    }
