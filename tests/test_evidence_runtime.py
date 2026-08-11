from __future__ import annotations

import json
import threading
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FutureTimeoutError
from contextlib import contextmanager
from dataclasses import replace
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from zoneinfo import ZoneInfo

import pytest

from chronovisor.core.canonical_json import canonical_json_sha256_strict
from chronovisor.core.raw_segment import append_capture
from chronovisor.decision.recall_policy_contract import RecallPolicy
from chronovisor.hosts import evidence_composition
from chronovisor.hosts.evidence_composition import run_evidence_acceptance
from chronovisor.recall.recall_runtime import (
    ContextItem,
    RecallRequest,
    RecallResult,
    format_recall_context,
)
from chronovisor.research import evidence_eval
from chronovisor.research import evidence_runtime as runtime
from chronovisor.research.evidence_eval import (
    evaluate_paired_replay,
    paired_evaluation_bytes,
    register_evidence_cases,
    seal_paired_case,
)
from chronovisor.research.evidence_reconstruction import (
    EpisodeProjection,
    EvidenceReconstructionError,
    EvidenceRef,
    EvidenceRelation,
    EvidenceRelationKind,
    Provenance,
    TimeInterval,
    build_episode_projection,
    build_evidence_atom,
    compile_retrieval_program,
    load_episode_projection,
)
from chronovisor.research.evidence_runtime import (
    EvidenceLedger,
    EvidenceRun,
    advance_evidence_rollout,
    compile_bounded_evidence_context,
    compile_projection_program,
    evidence_acceptance_path,
    evidence_applied_session_count,
    evidence_selected,
    load_evidence_rollout,
    rollback_evidence_rollout,
    run_evidence_retrieval,
)
from chronovisor.research.research_tools import ToolContext
from chronovisor.search.research_types import Action, ActionType

NOW = datetime(2026, 8, 11, 9, 30, tzinfo=ZoneInfo("Asia/Tokyo"))


@pytest.fixture(autouse=True)
def bind_evidence_provider() -> None:
    evidence_composition.bind_recall_provider()


def test_acceptance_callbacks_are_not_public(tmp_path: Path) -> None:
    assert not hasattr(evidence_eval, "run_evidence_acceptance")
    with pytest.raises(TypeError):
        run_evidence_acceptance(  # type: ignore[call-arg]
            tmp_path,
            page_teacher=lambda _query: object(),
            candidate_renderer=lambda _baseline, _run: b"forged",
        )


def test_host_acceptance_normalizes_only_no_page_teacher_result(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from chronovisor.recall import recall_runtime

    no_page = RecallResult(
        status="timeout",
        decision="none",
        confidence=0.0,
        queries=[],
        reasons=[],
        matched_terms={},
        context="shared working memory",
    )
    answerable = replace(
        no_page,
        status="ok",
        decision="read",
        queries=["found"],
        context_items=[ContextItem("page", "title", "2026-08-11", 1.0)],
        context="page answer",
    )
    by_query = {"missing": no_page, "found": answerable}
    monkeypatch.setattr(
        recall_runtime,
        "run_recall",
        lambda request, _policy: by_query[request.prompt],
    )
    monkeypatch.setattr(
        evidence_eval,
        "_run_evidence_acceptance",
        lambda **callbacks: {
            query: callbacks["page_teacher"](query) for query in by_query
        },
    )

    accepted = run_evidence_acceptance(tmp_path)

    assert accepted["missing"] == replace(no_page, context="", queries=["missing"])
    assert accepted["found"] is answerable


def _projection(tmp_path: Path) -> tuple[Path, EpisodeProjection, str]:
    for name in ("index.md", "log.md", "schema.md"):
        (tmp_path / name).write_text("legacy\n", encoding="utf-8")
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir(parents=True)
    source = tmp_path / "session.jsonl"
    rows = [
        {
            "type": "response_item",
            "timestamp": "2026-08-11T09:00:00+09:00",
            "payload": {
                "type": "message",
                "role": "user",
                "content": [{"type": "input_text", "text": "Is feature enabled?"}],
            },
        },
        {
            "type": "response_item",
            "timestamp": "2026-08-11T09:01:00+09:00",
            "payload": {
                "type": "message",
                "role": "assistant",
                "content": [{"type": "output_text", "text": "The feature is enabled."}],
            },
        },
        {
            "type": "response_item",
            "timestamp": "2026-08-11T09:02:00+09:00",
            "payload": {
                "type": "message",
                "role": "assistant",
                "content": [
                    {"type": "output_text", "text": "The lease reason is alpha."}
                ],
            },
        },
        {
            "type": "response_item",
            "timestamp": "2026-08-11T09:03:00+09:00",
            "payload": {
                "type": "message",
                "role": "assistant",
                "content": [
                    {
                        "type": "output_text",
                        "text": "The configuration changed to beta.",
                    }
                ],
            },
        },
        {
            "type": "response_item",
            "timestamp": "2026-08-11T09:04:00+09:00",
            "payload": {
                "type": "message",
                "role": "assistant",
                "content": [
                    {"type": "output_text", "text": "The worker failure was gamma."}
                ],
            },
        },
        {
            "type": "response_item",
            "timestamp": "2026-08-11T09:05:00+09:00",
            "payload": {
                "type": "message",
                "role": "assistant",
                "content": [
                    {"type": "output_text", "text": "The deployment workflow is delta."}
                ],
            },
        },
    ]
    raw = b"".join(
        json.dumps(row, ensure_ascii=False, separators=(",", ":")).encode() + b"\n"
        for row in rows
    )
    source.write_bytes(raw)
    append_capture(
        raw_dir=raw_dir,
        raw_id="save-codex-evidence-runtime.md",
        idempotency_key="codex-evidence-runtime",
        host="codex",
        session_key="a" * 24,
        session_id="session-1",
        source_file=source,
        after_line=0,
        until_line=len(rows),
        source_bytes=raw,
        record_count=len(rows),
        now=NOW,
    )
    return (
        raw_dir,
        load_episode_projection(build_episode_projection(raw_dir)),
        raw.decode(),
    )


def _atom(
    relation: EvidenceRelationKind,
    claim_id: str,
    *,
    claim: str,
    when: str = "2026-08-11T09:00:00+09:00",
    end: str | None = None,
    extra: tuple[EvidenceRelation, ...] = (),
) -> object:
    return build_evidence_atom(
        episode_id="episode:test",
        claim=claim,
        entities=(),
        provenance=Provenance("test", f"event:{claim}", "assistant", 0),
        evidence=EvidenceRef(
            raw_id=f"{claim.replace(' ', '-')}.md",
            byte_start=0,
            byte_end=1,
            raw_sha256="a" * 64,
            receipt_sha256="b" * 64,
        ),
        validity=TimeInterval(when, end or when),
        relations=(EvidenceRelation(relation, claim_id), *extra),
    )


def _program(query: str = "feature enabled") -> object:
    return compile_retrieval_program(
        query,
        {
            "as_of": "2026-08-11T10:00:00+09:00",
            "claim_slots": [{"slot_id": "answer", "claim": query}],
            "required_evidence": [
                {
                    "claim_slot": "answer",
                    "minimum_atoms": 1,
                    "relations": ["supports", "contradicts"],
                    "must_match_as_of": True,
                }
            ],
            "allowed_actions": ["raw_search"],
            "stop_rules": [
                "coverage",
                "contradiction_resolved",
                "as_of_satisfied",
                "abstain_on_gap",
            ],
        },
    )


def _evaluation_fixture(
    tmp_path: Path,
) -> tuple[
    list[dict[str, object]],
    dict[str, RecallResult],
    dict[str, bytes],
    dict[str, EvidenceRun],
    dict[str, bytes],
    dict[str, int],
]:
    _raw_dir, projection, _raw = _projection(tmp_path)
    as_of = "2026-08-11T10:00:00+09:00"
    definitions = {
        "current": ("feature enabled", "feature is enabled"),
        "why": ("lease reason alpha", "lease reason is alpha"),
        "change": ("configuration changed beta", "configuration changed to beta"),
        "failure": ("worker failure gamma", "worker failure was gamma"),
        "workflow": ("deployment workflow delta", "deployment workflow is delta"),
        "contradiction": ("synthetic relation proof", ""),
        "no-answer": ("no answer zeta", ""),
    }
    cases: list[dict[str, object]] = []
    baseline_results: dict[str, RecallResult] = {}
    baseline_outputs: dict[str, bytes] = {}
    candidate_runs: dict[str, EvidenceRun] = {}
    candidate_outputs: dict[str, bytes] = {}
    candidate_latency_ms: dict[str, int] = {}
    for slice_name, (query, expected_text) in definitions.items():
        case_id = f"case-{slice_name}"
        answerable = slice_name not in {"contradiction", "no-answer"}
        expected_atom = next(
            (
                atom
                for atom in projection.atoms
                if answerable and expected_text in atom.claim.casefold()
            ),
            None,
        )
        cases.append(
            seal_paired_case(
                {
                    "case_id": case_id,
                    "slice": slice_name,
                    "query": query,
                    "as_of": as_of,
                    "expected_answerable": answerable,
                    "expected_answer_text": expected_text,
                    "expected_page_ids": []
                    if not answerable
                    else [f"page-{slice_name}"],
                    "expected_atom_ids": []
                    if expected_atom is None
                    else [expected_atom.atom_id],
                    "forbidden_obsolete_page_ids": [],
                    "forbidden_obsolete_atom_ids": [],
                    "relation_expectation": (
                        runtime.evidence_relation_semantics_proof()
                        if slice_name == "contradiction"
                        else None
                    ),
                }
            )
        )
        baseline_context = "" if not answerable else expected_text
        baseline_results[case_id] = RecallResult(
            status="ok",
            decision="skip" if not answerable else "read",
            confidence=1.0,
            queries=[query],
            reasons=[],
            matched_terms={},
            context_items=(
                []
                if not answerable
                else [
                    ContextItem(
                        page_id=f"page-{slice_name}",
                        title=slice_name,
                        updated="2026-08-11T09:01:00+09:00",
                        score=1.0,
                    )
                ]
            ),
            context=baseline_context,
            latency_ms=1,
        )
        baseline_outputs[case_id] = baseline_context.encode()
        candidate_runs[case_id] = run_evidence_retrieval(
            compile_projection_program(query, as_of), projection
        )
        candidate_result = replace(
            baseline_results[case_id],
            decision="read",
            context_items=[],
            context="",
            evidence_packet=candidate_runs[case_id].packet,
            evidence_features={
                "evidence_reconstruction": {
                    "trace": dict(candidate_runs[case_id].trace),
                    "trace_sha256": canonical_json_sha256_strict(
                        candidate_runs[case_id].trace
                    ),
                }
            },
        )
        candidate_outputs[case_id] = format_recall_context(
            candidate_result, RecallPolicy(max_context_chars=100_000)
        ).encode()
        candidate_latency_ms[case_id] = 1
    return (
        cases,
        baseline_results,
        baseline_outputs,
        candidate_runs,
        candidate_outputs,
        candidate_latency_ms,
    )


def _evaluate_fixture(
    fixture: tuple[
        list[dict[str, object]],
        dict[str, RecallResult],
        dict[str, bytes],
        dict[str, EvidenceRun],
        dict[str, bytes],
        dict[str, int],
    ],
) -> dict[str, object]:
    (
        cases,
        baseline_results,
        baseline_outputs,
        candidate_runs,
        candidate_outputs,
        candidate_latency_ms,
    ) = fixture
    return evaluate_paired_replay(
        cases,
        baseline_results=baseline_results,
        baseline_outputs=baseline_outputs,
        candidate_runs=candidate_runs,
        candidate_outputs=candidate_outputs,
        candidate_latency_ms=candidate_latency_ms,
    )


def test_projection_first_stops_before_tools_and_context_is_all_or_none(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _raw_dir, projection, _raw = _projection(tmp_path)
    program = compile_projection_program("feature enabled", "2026-08-11T10:00:00+09:00")
    monkeypatch.setattr(
        runtime,
        "execute_tool",
        lambda *_args: (_ for _ in ()).throw(AssertionError("tool must not run")),
    )

    result = run_evidence_retrieval(
        program,
        projection,
        tool_context=ToolContext(config=None, store=None),  # type: ignore[arg-type]
        actions=(("answer", Action(ActionType.RAW_SEARCH, {"query": "feature"})),),
    )

    assert result.stop_reason == "coverage"
    assert result.packet.abstained is False
    assert result.telemetry["action_count"] == 0
    assert compile_bounded_evidence_context(result.packet, max_chars=50) is None
    context = compile_bounded_evidence_context(result.packet, max_chars=100_000)
    assert context is not None and "The feature is enabled." in context
    safe_telemetry = json.dumps(result.telemetry, sort_keys=True)
    assert "raw_id" not in safe_telemetry
    assert "feature enabled" not in safe_telemetry


def test_raw_search_only_promotes_exact_reconstructed_projection_atoms(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    raw_dir, projection, raw = _projection(tmp_path)
    program = _program("feature enabled")
    valid = {
        "query": "feature",
        "scanned": 1,
        "hits": [
            {
                "raw_id": "save-codex-evidence-runtime.md",
                "captured_date": "2026/08/11",
                "excerpt": raw,
                "offset": 0,
                "citation": "raw:save-codex-evidence-runtime.md#offset=0",
            }
        ],
    }
    monkeypatch.setattr(runtime, "execute_tool", lambda *_args: valid)

    result = run_evidence_retrieval(
        program,
        projection,
        tool_context=ToolContext(config=None, store=None),  # type: ignore[arg-type]
        actions=(("answer", Action(ActionType.RAW_SEARCH, {"query": "feature"})),),
        raw_dir=raw_dir,
    )
    assert result.stop_reason == "coverage"
    assert result.telemetry["action_count"] == 0

    unrelated = run_evidence_retrieval(
        _program("feature disabled"),  # type: ignore[arg-type]
        projection,
        tool_context=ToolContext(config=None, store=None),  # type: ignore[arg-type]
        actions=(("answer", Action(ActionType.RAW_SEARCH, {"query": "feature"})),),
        raw_dir=raw_dir,
    )
    assert unrelated.stop_reason == "action_exhausted"
    assert unrelated.packet.atoms == ()

    invalid = json.loads(json.dumps(valid))
    invalid["hits"][0]["excerpt"] = "forged"
    monkeypatch.setattr(runtime, "execute_tool", lambda *_args: invalid)
    held = run_evidence_retrieval(
        _program("feature disabled"),  # type: ignore[arg-type]
        projection,
        tool_context=ToolContext(config=None, store=None),  # type: ignore[arg-type]
        actions=(("answer", Action(ActionType.RAW_SEARCH, {"query": "feature"})),),
        raw_dir=raw_dir,
    )
    assert held.stop_reason == "action_exhausted"
    assert held.packet.abstained is True
    assert held.telemetry["tool_error_count"] == 1


def test_raw_search_fills_a_real_gap_beyond_the_l2_projection_cap(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    raw_dir, _projection_before, _raw = _projection(tmp_path)
    source = tmp_path / "bounded.jsonl"
    rows = [
        {
            "type": "response_item",
            "timestamp": "2026-08-11T09:10:00+09:00",
            "payload": {
                "type": "message",
                "role": "assistant",
                "content": [
                    {
                        "type": "output_text",
                        "text": f"Bounded evidence token{index:03d}.",
                    }
                ],
            },
        }
        for index in range(140)
    ]
    raw = b"".join(
        json.dumps(row, separators=(",", ":")).encode() + b"\n" for row in rows
    )
    source.write_bytes(raw)
    append_capture(
        raw_dir=raw_dir,
        raw_id="save-codex-bounded.md",
        idempotency_key="codex-bounded",
        host="codex",
        session_key="c" * 24,
        session_id="session-bounded",
        source_file=source,
        after_line=0,
        until_line=len(rows),
        source_bytes=raw,
        record_count=len(rows),
        now=NOW,
    )
    projection = load_episode_projection(build_episode_projection(raw_dir))
    target = next(
        atom
        for atom in projection.atoms[128:]
        if atom.claim.startswith("Bounded evidence")
    )
    program = compile_projection_program(target.claim, "2026-08-11T10:00:00+09:00")
    assert run_evidence_retrieval(program, projection).packet.abstained is True
    monkeypatch.setattr(
        runtime,
        "execute_tool",
        lambda *_args: {
            "query": target.claim,
            "scanned": len(rows),
            "hits": [
                {
                    "raw_id": "save-codex-bounded.md",
                    "captured_date": "2026/08/11",
                    "excerpt": raw.decode(),
                    "offset": 0,
                    "citation": "raw:save-codex-bounded.md#offset=0",
                }
            ],
        },
    )
    result = run_evidence_retrieval(
        program,
        projection,
        tool_context=ToolContext(config=None, store=None),  # type: ignore[arg-type]
        actions=(("answer", Action(ActionType.RAW_SEARCH, {"query": target.claim})),),
        raw_dir=raw_dir,
    )
    assert result.stop_reason == "coverage"
    assert result.telemetry["action_count"] == 1
    assert target.atom_id in {atom.atom_id for atom in result.packet.atoms}


def test_ledger_fails_closed_on_contradiction_future_and_resolves_superseded() -> None:
    program = _program()
    unrelated = EvidenceLedger(program)  # type: ignore[arg-type]
    unrelated.add(
        "answer",
        _atom(
            EvidenceRelationKind.SUPPORTS,
            "claim:a",
            claim="feature enabled support-a",
        ),  # type: ignore[arg-type]
    )
    unrelated.add(
        "answer",
        _atom(
            EvidenceRelationKind.CONTRADICTS,
            "claim:b",
            claim="feature enabled contradict-b",
        ),  # type: ignore[arg-type]
    )
    assert unrelated.unresolved_contradiction() is False
    assert unrelated.covered() is True

    ledger = EvidenceLedger(program)  # type: ignore[arg-type]
    ledger.add(
        "answer",
        _atom(
            EvidenceRelationKind.SUPPORTS,
            "claim:old",
            claim="feature enabled support",
        ),  # type: ignore[arg-type]
    )
    ledger.add(
        "answer",
        _atom(
            EvidenceRelationKind.CONTRADICTS,
            "claim:old",
            claim="feature enabled contradict",
        ),  # type: ignore[arg-type]
    )
    assert ledger.unresolved_contradiction() is True
    assert ledger.covered() is False

    ledger.add(
        "answer",
        _atom(
            EvidenceRelationKind.SUPPORTS,
            "claim:new",
            claim="feature enabled replacement",
            extra=(EvidenceRelation(EvidenceRelationKind.SUPERSEDES, "claim:old"),),
        ),  # type: ignore[arg-type]
    )
    assert ledger.unresolved_contradiction() is False
    assert ledger.covered() is True

    mixed = EvidenceLedger(program)  # type: ignore[arg-type]
    mixed.add(
        "answer",
        _atom(
            EvidenceRelationKind.SUPPORTS,
            "claim:valid",
            claim="feature enabled valid",
        ),  # type: ignore[arg-type]
    )
    assert (
        mixed.add(
            "answer",
            _atom(
                EvidenceRelationKind.SUPPORTS,
                "claim:future",
                claim="feature enabled future",
                when="2026-08-12T09:00:00+09:00",
            ),  # type: ignore[arg-type]
        )
        is False
    )
    assert mixed.covered() is True
    assert mixed.slot_state("answer")["future_count"] == 1
    assert mixed.slot_state("answer")["as_of_satisfied"] is True

    expired = EvidenceLedger(program)  # type: ignore[arg-type]
    assert (
        expired.add(
            "answer",
            _atom(
                EvidenceRelationKind.SUPPORTS,
                "claim:expired",
                claim="feature enabled expired",
                when="2026-08-11T08:00:00+09:00",
                end="2026-08-11T09:00:00+09:00",
            ),  # type: ignore[arg-type]
        )
        is False
    )
    assert expired.temporal_gap() is True
    assert expired.slot_state("answer")["expired_count"] == 1


def test_deadline_and_unapproved_actions_abstain_before_execution(
    tmp_path: Path,
) -> None:
    _raw_dir, projection, _raw = _projection(tmp_path)
    program = _program()
    result = run_evidence_retrieval(program, projection, deadline_ms=0)  # type: ignore[arg-type]
    assert result.stop_reason == "deadline_exceeded"
    assert result.packet.abstained is True

    with pytest.raises(EvidenceReconstructionError, match="not allowed"):
        run_evidence_retrieval(
            program,  # type: ignore[arg-type]
            projection,
            actions=(("answer", Action(ActionType.WEB_SEARCH, {"query": "x"})),),
        )


def test_sealed_paired_eval_derives_independent_gates_from_actual_arms(
    tmp_path: Path,
) -> None:
    fixture = _evaluation_fixture(tmp_path)
    cases, baselines, baseline_outputs, runs, candidate_outputs, latencies = fixture
    actual = {
        "baseline_results": baselines,
        "baseline_outputs": baseline_outputs,
        "candidate_runs": runs,
        "candidate_outputs": candidate_outputs,
        "candidate_latency_ms": latencies,
    }
    first = paired_evaluation_bytes(cases, **actual)
    assert first == paired_evaluation_bytes(list(reversed(cases)), **actual)
    assert _evaluate_fixture(fixture)["all_pass"] is True

    current = "case-current"
    answer_cases = json.loads(json.dumps(cases))
    index = next(i for i, row in enumerate(answer_cases) if row["case_id"] == current)
    answer_cases[index]["expected_answer_text"] = "teacher-only answer"
    answer_cases[index] = seal_paired_case(
        {
            key: value
            for key, value in answer_cases[index].items()
            if key != "case_sha256"
        }
    )
    answer_baselines = dict(baselines)
    answer_baselines[current] = replace(
        baselines[current], context="teacher-only answer"
    )
    answer_baseline_outputs = dict(baseline_outputs)
    answer_baseline_outputs[current] = b"teacher-only answer"
    answer_result = evaluate_paired_replay(
        answer_cases,
        baseline_results=answer_baselines,
        baseline_outputs=answer_baseline_outputs,
        candidate_runs=runs,
        candidate_outputs=candidate_outputs,
        candidate_latency_ms=latencies,
    )
    assert answer_result["gates"]["answer"] is False
    assert answer_result["slice_gates"]["answer:current"] is False
    assert all(
        passed is True
        for name, passed in answer_result["gates"].items()
        if name != "answer"
    )

    evidence_cases = json.loads(json.dumps(cases))
    index = next(i for i, row in enumerate(evidence_cases) if row["case_id"] == current)
    evidence_cases[index]["expected_atom_ids"] = ["atom:" + "f" * 64]
    evidence_cases[index] = seal_paired_case(
        {
            key: value
            for key, value in evidence_cases[index].items()
            if key != "case_sha256"
        }
    )
    evidence_result = evaluate_paired_replay(
        evidence_cases,
        baseline_results=baselines,
        baseline_outputs=baseline_outputs,
        candidate_runs=runs,
        candidate_outputs=candidate_outputs,
        candidate_latency_ms=latencies,
    )
    assert evidence_result["slice_gates"]["evidence:current"] is False

    obsolete_cases = json.loads(json.dumps(cases))
    index = next(i for i, row in enumerate(obsolete_cases) if row["case_id"] == current)
    obsolete_cases[index]["forbidden_obsolete_atom_ids"] = list(
        obsolete_cases[index]["expected_atom_ids"]
    )
    obsolete_cases[index] = seal_paired_case(
        {
            key: value
            for key, value in obsolete_cases[index].items()
            if key != "case_sha256"
        }
    )
    obsolete_result = evaluate_paired_replay(
        obsolete_cases,
        baseline_results=baselines,
        baseline_outputs=baseline_outputs,
        candidate_runs=runs,
        candidate_outputs=candidate_outputs,
        candidate_latency_ms=latencies,
    )
    assert obsolete_result["slice_gates"]["obsolete-use:current"] is False

    forged_runs = dict(runs)
    forged_trace = json.loads(json.dumps(runs[current].trace))
    forged_trace["ledger"]["slots"]["answer"]["as_of_satisfied"] = False
    forged_trace["ledger"]["slots"]["answer"]["contradiction"] = True
    forged_runs[current] = replace(
        runs[current],
        trace=forged_trace,
        telemetry={**runs[current].telemetry, "latency_ms": 99_999},
    )
    forged_bound_outputs = dict(candidate_outputs)
    forged_candidate = replace(
        baselines[current],
        decision="read",
        context_items=[],
        context="",
        evidence_packet=forged_runs[current].packet,
        evidence_features={
            "evidence_reconstruction": {
                "trace": dict(forged_runs[current].trace),
                "trace_sha256": canonical_json_sha256_strict(
                    forged_runs[current].trace
                ),
            }
        },
    )
    forged_bound_outputs[current] = format_recall_context(
        forged_candidate, RecallPolicy(max_context_chars=100_000)
    ).encode()
    assert forged_bound_outputs[current] == b""
    forged_metrics = evaluate_paired_replay(
        cases,
        baseline_results=baselines,
        baseline_outputs=baseline_outputs,
        candidate_runs=forged_runs,
        candidate_outputs=forged_bound_outputs,
        candidate_latency_ms=latencies,
    )
    assert forged_metrics["all_pass"] is False
    assert forged_metrics["bounded_context"]["passed"] is False

    slow = dict(latencies)
    slow[current] = 5_000
    slow_result = evaluate_paired_replay(
        cases,
        baseline_results=baselines,
        baseline_outputs=baseline_outputs,
        candidate_runs=runs,
        candidate_outputs=candidate_outputs,
        candidate_latency_ms=slow,
    )
    assert slow_result["slice_gates"]["latency:current"] is False

    masked_baselines = dict(answer_baselines)
    masked_baseline_outputs = dict(answer_baseline_outputs)
    for case_id in (
        "case-why",
        "case-change",
        "case-failure",
        "case-workflow",
    ):
        masked_baselines[case_id] = replace(
            baselines[case_id], context="wrong baseline answer"
        )
        masked_baseline_outputs[case_id] = b"wrong baseline answer"
    masked_result = evaluate_paired_replay(
        answer_cases,
        baseline_results=masked_baselines,
        baseline_outputs=masked_baseline_outputs,
        candidate_runs=runs,
        candidate_outputs=candidate_outputs,
        candidate_latency_ms=latencies,
    )
    assert masked_result["gates"]["answer"] is True
    assert masked_result["slice_gates"]["answer:current"] is False
    assert masked_result["all_pass"] is False

    unsafe_outputs = dict(candidate_outputs)
    unsafe_outputs["case-no-answer"] = b"fabricated answer bytes"
    with pytest.raises(EvidenceReconstructionError, match="output binding"):
        evaluate_paired_replay(
            cases,
            baseline_results=baselines,
            baseline_outputs=baseline_outputs,
            candidate_runs=runs,
            candidate_outputs=unsafe_outputs,
            candidate_latency_ms=latencies,
        )
    forged_outputs = dict(candidate_outputs)
    forged_outputs[current] = b"The feature is enabled."
    with pytest.raises(EvidenceReconstructionError, match="output binding"):
        evaluate_paired_replay(
            cases,
            baseline_results=baselines,
            baseline_outputs=baseline_outputs,
            candidate_runs=runs,
            candidate_outputs=forged_outputs,
            candidate_latency_ms=latencies,
        )
    tampered = json.loads(json.dumps(cases))
    tampered[0]["query"] = "substituted query"
    with pytest.raises(EvidenceReconstructionError, match="paired case 0 is invalid"):
        evaluate_paired_replay(
            tampered,
            baseline_results=baselines,
            baseline_outputs=baseline_outputs,
            candidate_runs=runs,
            candidate_outputs=candidate_outputs,
            candidate_latency_ms=latencies,
        )
    contradiction_index = next(
        i for i, row in enumerate(cases) if row["slice"] == "contradiction"
    )
    for relation_expectation in (
        None,
        {
            **runtime.evidence_relation_semantics_proof(),
            "support_atom_id": "atom:" + "f" * 64,
        },
    ):
        invalid_relation = json.loads(json.dumps(cases))
        invalid_relation[contradiction_index][
            "relation_expectation"
        ] = relation_expectation
        invalid_relation[contradiction_index] = seal_paired_case(
            {
                key: value
                for key, value in invalid_relation[contradiction_index].items()
                if key != "case_sha256"
            }
        )
        with pytest.raises(EvidenceReconstructionError, match="paired case .* invalid"):
            evaluate_paired_replay(
                invalid_relation,
                baseline_results=baselines,
                baseline_outputs=baseline_outputs,
                candidate_runs=runs,
                candidate_outputs=candidate_outputs,
                candidate_latency_ms=latencies,
            )
    no_answer = next(row for row in cases if row["slice"] == "no-answer")
    assert no_answer["relation_expectation"] is None
    substituted = dict(runs)
    substituted[current] = runs["case-no-answer"]
    with pytest.raises(EvidenceReconstructionError, match="candidate identity"):
        evaluate_paired_replay(
            cases,
            baseline_results=baselines,
            baseline_outputs=baseline_outputs,
            candidate_runs=substituted,
            candidate_outputs=candidate_outputs,
            candidate_latency_ms=latencies,
        )
    trace_substituted = dict(runs)
    bad_trace = json.loads(json.dumps(runs[current].trace))
    bad_trace["program"]["query"] = "different trace query"
    trace_substituted[current] = replace(runs[current], trace=bad_trace)
    with pytest.raises(EvidenceReconstructionError, match="candidate identity"):
        evaluate_paired_replay(
            cases,
            baseline_results=baselines,
            baseline_outputs=baseline_outputs,
            candidate_runs=trace_substituted,
            candidate_outputs=candidate_outputs,
            candidate_latency_ms=latencies,
        )

    duplicate_queries = json.loads(json.dumps(cases))
    duplicate_queries[1]["query"] = duplicate_queries[0]["query"]
    duplicate_queries[1] = seal_paired_case(
        {
            key: value
            for key, value in duplicate_queries[1].items()
            if key != "case_sha256"
        }
    )
    with pytest.raises(EvidenceReconstructionError, match="queries must be distinct"):
        evaluate_paired_replay(
            duplicate_queries,
            baseline_results=baselines,
            baseline_outputs=baseline_outputs,
            candidate_runs=runs,
            candidate_outputs=candidate_outputs,
            candidate_latency_ms=latencies,
        )


def test_raw_inventory_rejects_symlink_entries(tmp_path: Path) -> None:
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    regular = raw_dir / "regular.jsonl"
    regular.write_text("{}\n")
    (raw_dir / "linked.jsonl").symlink_to(regular)

    with pytest.raises(EvidenceReconstructionError, match="non-regular"):
        evidence_eval.physical_raw_inventory(raw_dir)


def test_public_acceptance_seals_raw_fault_and_live_sample_receipts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from chronovisor.recall import recall_runtime

    cases, baselines, _outputs, fixture_runs, _candidate_outputs, _latencies = (
        _evaluation_fixture(tmp_path)
    )
    with pytest.raises(EvidenceReconstructionError, match="Campaign X"):
        register_evidence_cases(root=tmp_path, cases=cases)
    with pytest.raises(EvidenceReconstructionError, match="Campaign X"):
        run_evidence_acceptance(root=tmp_path)
    assert not runtime.evidence_projection_path(tmp_path).exists()
    assert not evidence_acceptance_path(tmp_path).exists()

    decision = SimpleNamespace(
        allowed=True,
        layout="okf_v0_2",
        state="finalized-v2",
    )

    @contextmanager
    def operation(_root: Path):
        yield decision

    monkeypatch.setattr(evidence_eval, "okf_startup_status", lambda _root: decision)
    monkeypatch.setattr(evidence_eval, "okf_runtime_operation", operation)
    monkeypatch.setattr(runtime, "_okf_finalized", lambda _root: True)
    monkeypatch.setattr(runtime, "okf_runtime_operation", operation)
    baseline_by_query = {result.queries[0]: result for result in baselines.values()}
    monkeypatch.setattr(
        recall_runtime,
        "load_policy",
        lambda: RecallPolicy(max_context_chars=100_000),
    )

    def page_teacher(request: object, policy: RecallPolicy) -> RecallResult:
        assert policy.semantic is False
        assert policy.judge_mode == "off"
        assert policy.rewrite_enabled is False
        assert policy.processor_enabled is False
        assert policy.processor_shadow_enabled is False
        return baseline_by_query[request.prompt]  # type: ignore[attr-defined]

    monkeypatch.setattr(
        recall_runtime,
        "run_recall",
        page_teacher,
    )

    registered = register_evidence_cases(root=tmp_path, cases=cases)
    assert registered["case_manifest_sha256"]
    with pytest.raises(EvidenceReconstructionError, match="already registered"):
        register_evidence_cases(root=tmp_path, cases=cases)
    result = run_evidence_acceptance(root=tmp_path)

    assert result["status"] == "passed"
    assert result["projection"]["deterministic"] is True
    assert result["evaluation"]["all_pass"] is True
    assert result["rollout"]["canary_percent"] == 5
    assert result["rollout"]["sample_count"] == 0
    assert result["raw_inventory"]["before"] == result["raw_inventory"]["after"]
    assert result["acceptance_receipt"]["atomic_publication_fault_sha256"]
    assert result["acceptance_receipt"]["relation_semantics_sha256"]
    assert result["acceptance_receipt"]["gates"]["relation_semantics"] is True
    assert result["acceptance_receipt"]["gates"]["cloud_egress_zero"] is True

    projection_id = "projection:" + result["projection"]["projection_sha256"]
    monkeypatch.setattr(runtime, "selected_for_canary", lambda *_args: True)
    assert evidence_selected(tmp_path, "session", projection_id) is True
    assert evidence_selected(tmp_path, "session", "projection:" + "b" * 64) is False
    with monkeypatch.context() as scoped:
        scoped.setattr(
            runtime,
            "physical_raw_inventory",
            lambda *_args: (_ for _ in ()).throw(
                AssertionError("L2 must not hash full Raw")
            ),
        )
        assert evidence_selected(tmp_path, "session", projection_id) is True

    authority_dir = tmp_path / "runtime" / "evidence-reconstruction"
    cases_path = runtime.evidence_cases_path(tmp_path)
    acceptance_path = evidence_acceptance_path(tmp_path)
    promotion_path = authority_dir / "promotion.json"
    saved_cases = cases_path.read_bytes()
    cases_path.unlink()
    assert evidence_selected(tmp_path, "session", projection_id) is False
    with pytest.raises(EvidenceReconstructionError, match="registered evidence cases"):
        run_evidence_acceptance(root=tmp_path)
    cases_path.write_bytes(saved_cases)
    assert run_evidence_acceptance(root=tmp_path)["status"] == "passed"

    cases_path.write_bytes(b"{}\n")
    assert evidence_selected(tmp_path, "session", projection_id) is False
    with pytest.raises(EvidenceReconstructionError, match="registered evidence cases"):
        run_evidence_acceptance(root=tmp_path)
    cases_path.write_bytes(saved_cases)
    assert run_evidence_acceptance(root=tmp_path)["status"] == "passed"

    for authority in (cases_path, acceptance_path, promotion_path):
        saved = authority.read_bytes()
        old = authority_dir / f"old-{authority.name}"
        old.write_bytes(saved)
        authority.unlink()
        authority.symlink_to(old)
        assert evidence_selected(tmp_path, "session", projection_id) is False
        if authority == cases_path:
            with pytest.raises(
                EvidenceReconstructionError, match="registered evidence cases"
            ):
                run_evidence_acceptance(root=tmp_path)
            authority.unlink()
            authority.write_bytes(saved)
        else:
            with pytest.raises((EvidenceReconstructionError, ValueError, OSError)):
                run_evidence_acceptance(root=tmp_path)
            assert authority.is_symlink() is True
            authority.unlink()
            authority.write_bytes(saved)
            repaired = run_evidence_acceptance(root=tmp_path)
            assert repaired["rollout"]["canary_percent"] == 5

    recall_log = tmp_path / "recall" / "recall-log.jsonl"
    recall_log.parent.mkdir(parents=True)
    forged_rows = [
        {
            "stage": "injected",
            "host": "codex",
            "session_id": session,
            "evidence_features": {
                "evidence_reconstruction": {
                    "status": "active",
                    "abstained": False,
                    "projection_sha256": result["projection"]["projection_sha256"],
                    "packet_sha256": "a" * 64,
                }
            },
        }
        for session in ("one", "two")
    ]
    recall_log.write_text("".join(json.dumps(row) + "\n" for row in forged_rows))
    assert evidence_applied_session_count(
        tmp_path, result["projection"]["projection_sha256"]
    ) == 0
    recall_log.write_text("")
    monkeypatch.setattr(recall_runtime, "RECALL_LOG_FILE", recall_log)
    applied_run = fixture_runs["case-current"]

    def publish(session: str) -> None:
        applied = RecallResult(
            status="ok",
            decision="read",
            confidence=1.0,
            queries=[applied_run.packet.query],
            reasons=[],
            matched_terms={},
            decision_id=f"decision-{session}",
            session_id=session,
            context_style="cards",
            evidence_packet=applied_run.packet,
            evidence_features={
                "evidence_reconstruction": {
                    **applied_run.telemetry,
                    "status": "active",
                    "authority": "evidence_reconstruction",
                    "trace": dict(applied_run.trace),
                    "trace_sha256": canonical_json_sha256_strict(applied_run.trace),
                }
            },
        )
        applied.context = format_recall_context(
            applied, RecallPolicy(max_context_chars=100_000)
        )
        recall_runtime.append_recall_log(
            RecallRequest(
                host="codex",
                event="UserPromptSubmit",
                prompt=applied_run.packet.query,
                session_id=session,
            ),
            applied,
        )

    for session in ("one", "two"):
        publish(session)
    assert evidence_applied_session_count(
        tmp_path, result["projection"]["projection_sha256"]
    ) == 2
    advanced = advance_evidence_rollout(root=tmp_path, minimum_step_samples=1)
    assert advanced["canary_percent"] == 25
    assert advanced["sample_count"] == 2
    valid_log = recall_log.read_text()
    corrupt_rows = [json.loads(line) for line in valid_log.splitlines()]
    corrupt_rows[0]["context_receipt"]["rendered_context_sha256"] = "0" * 64
    recall_log.write_text("".join(json.dumps(row) + "\n" for row in corrupt_rows))
    assert evidence_applied_session_count(
        tmp_path, result["projection"]["projection_sha256"]
    ) == 0
    recall_log.write_text(valid_log)
    publish("three")
    active = advance_evidence_rollout(root=tmp_path, minimum_step_samples=1)
    assert active["mode"] == "active"
    assert active["canary_percent"] == 100
    active_bytes = promotion_path.read_bytes()
    old_backup = promotion_path.with_name("promotion.json.bak")
    old_backup.write_bytes(active_bytes)
    promotion_path.write_text("corrupt")
    assert load_evidence_rollout(tmp_path)["mode"] == "shadow"
    with pytest.raises(EvidenceReconstructionError, match="promotion authority"):
        rollback_evidence_rollout(root=tmp_path, reason="must not recover backup")
    assert promotion_path.read_text() == "corrupt"
    assert old_backup.read_bytes() == active_bytes
    assert run_evidence_acceptance(root=tmp_path)["rollout"]["canary_percent"] == 5

    data_path = next(
        path
        for path in (tmp_path / "raw").rglob("*")
        if path.is_file() and b"feature is enabled" in path.read_bytes()
    )
    original = data_path.read_bytes()
    data_path.write_bytes(bytes([original[0] ^ 1]) + original[1:])
    assert evidence_selected(tmp_path, "session", projection_id) is False
    with pytest.raises(EvidenceReconstructionError, match="stale"):
        advance_evidence_rollout(root=tmp_path, minimum_step_samples=1)
    data_path.write_bytes(original)

    added_source = tmp_path / "added-session.jsonl"
    added_raw = (
        json.dumps(
            {
                "type": "response_item",
                "timestamp": "2026-08-11T09:02:00+09:00",
                "payload": {
                    "type": "message",
                    "role": "assistant",
                    "content": [{"type": "output_text", "text": "Raw changed."}],
                },
            },
            separators=(",", ":"),
        ).encode()
        + b"\n"
    )
    added_source.write_bytes(added_raw)
    append_capture(
        raw_dir=tmp_path / "raw",
        raw_id="save-codex-added.md",
        idempotency_key="codex-added",
        host="codex",
        session_key="b" * 24,
        session_id="session-2",
        source_file=added_source,
        after_line=0,
        until_line=1,
        source_bytes=added_raw,
        record_count=1,
        now=NOW,
    )
    assert evidence_selected(tmp_path, "session", projection_id) is False

    refreshed = run_evidence_acceptance(root=tmp_path)
    refreshed_projection_id = (
        "projection:" + refreshed["projection"]["projection_sha256"]
    )
    assert refreshed_projection_id != projection_id
    assert refreshed["rollout"]["canary_percent"] == 5
    assert evidence_selected(tmp_path, "session", refreshed_projection_id) is True

    rolled_back = rollback_evidence_rollout(root=tmp_path, reason="fault injection")
    assert rolled_back["mode"] == "shadow"
    assert rolled_back["canary_percent"] == 0
    assert set(rolled_back["gates"]) == runtime.evidence_rollout_gate_keys()
    assert evidence_selected(tmp_path, "session", refreshed_projection_id) is False
    with pytest.raises(EvidenceReconstructionError, match="promotion authority"):
        advance_evidence_rollout(root=tmp_path, minimum_step_samples=1)

    promotion = promotion_path
    backup = promotion.with_name("promotion.json.bak")
    assert backup.exists()
    promotion.write_text("corrupt")
    assert load_evidence_rollout(tmp_path)["reason"] == "no_valid_promotion"
    assert promotion.read_text() == "corrupt"
    recovered = run_evidence_acceptance(root=tmp_path)
    assert recovered["rollout"]["canary_percent"] == 5
    assert load_evidence_rollout(tmp_path)["mode"] == "candidate"
    json.dumps(result)


def test_rollout_rejects_stale_raw_without_changing_promotion(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from chronovisor.recall import recall_runtime

    cases, baselines, *_ = _evaluation_fixture(tmp_path)
    decision = SimpleNamespace(allowed=True, layout="okf_v0_2", state="finalized-v2")

    @contextmanager
    def operation(_root: Path):
        yield decision

    monkeypatch.setattr(evidence_eval, "okf_startup_status", lambda _root: decision)
    monkeypatch.setattr(evidence_eval, "okf_runtime_operation", operation)
    monkeypatch.setattr(runtime, "_okf_finalized", lambda _root: True)
    monkeypatch.setattr(runtime, "okf_runtime_operation", operation)
    monkeypatch.setattr(
        recall_runtime,
        "load_policy",
        lambda: RecallPolicy(max_context_chars=100_000),
    )
    baseline_by_query = {result.queries[0]: result for result in baselines.values()}
    monkeypatch.setattr(
        recall_runtime,
        "run_recall",
        lambda request, _policy: baseline_by_query[request.prompt],
    )
    register_evidence_cases(root=tmp_path, cases=cases)
    accepted = run_evidence_acceptance(root=tmp_path)
    assert accepted["status"] == "passed"
    assert all(accepted["acceptance_receipt"]["gates"].values())

    promotion_path = (
        tmp_path / "runtime" / "evidence-reconstruction" / "promotion.json"
    )
    promotion_before = promotion_path.read_bytes()
    source = tmp_path / "stale-session.jsonl"
    raw = (
        json.dumps(
            {
                "type": "response_item",
                "timestamp": "2026-08-11T09:10:00+09:00",
                "payload": {
                    "type": "message",
                    "role": "assistant",
                    "content": [{"type": "output_text", "text": "Raw advanced."}],
                },
            },
            separators=(",", ":"),
        ).encode()
        + b"\n"
    )
    source.write_bytes(raw)
    append_capture(
        raw_dir=tmp_path / "raw",
        raw_id="save-codex-stale.md",
        idempotency_key="codex-stale",
        host="codex",
        session_key="c" * 24,
        session_id="stale-session",
        source_file=source,
        after_line=0,
        until_line=1,
        source_bytes=raw,
        record_count=1,
        now=NOW,
    )
    assert runtime.committed_raw_watermark(tmp_path / "raw") != accepted[
        "acceptance_receipt"
    ]["raw_watermark_sha256"]

    with pytest.raises(EvidenceReconstructionError, match="acceptance receipt is stale"):
        advance_evidence_rollout(root=tmp_path)
    assert promotion_path.read_bytes() == promotion_before


def test_held_acceptance_can_refresh_after_cold_latency_and_context_fallback(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from chronovisor.recall import recall_runtime

    cases, baselines, _outputs, _runs, _candidate_outputs, _latencies = (
        _evaluation_fixture(tmp_path)
    )
    decision = SimpleNamespace(allowed=True, layout="okf_v0_2", state="finalized-v2")

    @contextmanager
    def operation(_root: Path):
        yield decision

    monkeypatch.setattr(evidence_eval, "okf_startup_status", lambda _root: decision)
    monkeypatch.setattr(evidence_eval, "okf_runtime_operation", operation)
    monkeypatch.setattr(runtime, "_okf_finalized", lambda _root: True)
    monkeypatch.setattr(runtime, "okf_runtime_operation", operation)
    baseline_by_query = {result.queries[0]: result for result in baselines.values()}
    monkeypatch.setattr(
        recall_runtime,
        "run_recall",
        lambda request, _policy: baseline_by_query[request.prompt],
    )
    register_evidence_cases(root=tmp_path, cases=cases)

    with monkeypatch.context() as scoped:
        scoped.setattr(
            runtime,
            "run_projection_cycle_at",
            lambda **_kwargs: (_ for _ in ()).throw(OSError("refresh fault")),
        )
        with pytest.raises(OSError, match="refresh fault"):
            run_evidence_acceptance(root=tmp_path)
    assert load_evidence_rollout(tmp_path)["mode"] == "shadow"

    monkeypatch.setattr(
        recall_runtime,
        "load_policy",
        lambda: RecallPolicy(max_context_chars=100_000),
    )
    with monkeypatch.context() as scoped:
        ticks = iter([value for _ in range(7) for value in (0, 5_000_000_000)])
        scoped.setattr(evidence_eval.time, "monotonic_ns", lambda: next(ticks))
        cold = run_evidence_acceptance(root=tmp_path)
    assert cold["status"] == "held"
    assert cold["evaluation"]["gates"]["latency"] is False
    assert cold["rollout"]["mode"] == "shadow"

    monkeypatch.setattr(
        recall_runtime,
        "load_policy",
        lambda: RecallPolicy(max_context_chars=400),
    )
    oversized = run_evidence_acceptance(root=tmp_path)
    assert oversized["status"] == "held"
    assert oversized["evaluation"]["bounded_context"]["passed"] is False
    assert oversized["rollout"]["mode"] == "shadow"

    held_receipt = dict(oversized["acceptance_receipt"])
    held_receipt["raw_before_sha256"] = "f" * 64
    held_receipt["gates"] = {
        **held_receipt["gates"],
        "projection_deterministic": False,
        "raw_unchanged": False,
    }
    for raw_before, raw_unchanged in (
        (held_receipt["raw_after_sha256"], False),
        (held_receipt["raw_before_sha256"], True),
    ):
        conflicting = {
            **held_receipt,
            "raw_before_sha256": raw_before,
            "gates": {
                **held_receipt["gates"],
                "raw_unchanged": raw_unchanged,
            },
        }
        with runtime.evidence_authority_operation(tmp_path) as directory_fd:
            runtime.write_evidence_authority_at(
                directory_fd, "acceptance.json", conflicting
            )
        with pytest.raises(EvidenceReconstructionError, match="receipt is invalid"):
            runtime.load_evidence_acceptance(tmp_path)
    with runtime.evidence_authority_operation(tmp_path) as directory_fd:
        runtime.write_evidence_authority_at(
            directory_fd, "acceptance.json", held_receipt
        )
    loaded_held = runtime.load_evidence_acceptance(tmp_path)
    assert loaded_held["raw_before_sha256"] != loaded_held["raw_after_sha256"]
    held_rollout = advance_evidence_rollout(root=tmp_path)
    assert held_rollout["mode"] == "shadow"
    assert held_rollout["reason"] == "gate_failed"
    projection_id = "projection:" + oversized["projection"]["projection_sha256"]
    assert evidence_selected(tmp_path, "held-session", projection_id) is False

    monkeypatch.setattr(
        recall_runtime,
        "load_policy",
        lambda: RecallPolicy(max_context_chars=100_000),
    )
    recovered = run_evidence_acceptance(root=tmp_path)
    assert recovered["status"] == "passed"
    assert recovered["rollout"]["canary_percent"] == 5


def test_authority_parent_symlink_cannot_escape_root(tmp_path: Path) -> None:
    root = tmp_path / "root"
    outside = tmp_path / "outside"
    root.mkdir()
    outside.mkdir()
    (root / "runtime").symlink_to(outside, target_is_directory=True)

    with pytest.raises(OSError):
        with runtime.evidence_authority_operation(root) as directory_fd:
            runtime.write_evidence_authority_at(
                directory_fd, "cases.json", {"unexpected": True}
            )

    assert list(outside.iterdir()) == []


def test_case_registration_is_one_shot_under_concurrency(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cases, *_ = _evaluation_fixture(tmp_path)
    decision = SimpleNamespace(allowed=True, layout="okf_v0_2", state="finalized-v2")

    @contextmanager
    def operation(_root: Path):
        yield decision

    monkeypatch.setattr(evidence_eval, "okf_startup_status", lambda _root: decision)
    monkeypatch.setattr(evidence_eval, "okf_runtime_operation", operation)
    alternate = [
        seal_paired_case(
            {
                **{key: value for key, value in case.items() if key != "case_sha256"},
                "query": f"{case['query']} alternate",
            }
        )
        for case in cases
    ]
    barrier = threading.Barrier(2)

    def register(rows: list[dict[str, object]]) -> object:
        barrier.wait()
        try:
            return register_evidence_cases(root=tmp_path, cases=rows)
        except EvidenceReconstructionError as exc:
            return exc

    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(register, (cases, alternate)))

    winners = [result for result in results if isinstance(result, dict)]
    failures = [result for result in results if isinstance(result, Exception)]
    assert len(winners) == len(failures) == 1
    first_bytes = runtime.evidence_cases_path(tmp_path).read_bytes()
    with pytest.raises(EvidenceReconstructionError, match="already registered"):
        register_evidence_cases(root=tmp_path, cases=cases)
    assert runtime.evidence_cases_path(tmp_path).read_bytes() == first_bytes


def test_acceptance_and_rollback_are_linearized(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from chronovisor.recall import recall_runtime

    cases, baselines, *_ = _evaluation_fixture(tmp_path)
    decision = SimpleNamespace(allowed=True, layout="okf_v0_2", state="finalized-v2")

    @contextmanager
    def operation(_root: Path):
        yield decision

    monkeypatch.setattr(evidence_eval, "okf_startup_status", lambda _root: decision)
    monkeypatch.setattr(evidence_eval, "okf_runtime_operation", operation)
    monkeypatch.setattr(runtime, "_okf_finalized", lambda _root: True)
    monkeypatch.setattr(runtime, "okf_runtime_operation", operation)
    monkeypatch.setattr(
        recall_runtime, "load_policy", lambda: RecallPolicy(max_context_chars=100_000)
    )
    baseline_by_query = {result.queries[0]: result for result in baselines.values()}
    monkeypatch.setattr(
        recall_runtime,
        "run_recall",
        lambda request, _policy: baseline_by_query[request.prompt],
    )
    register_evidence_cases(root=tmp_path, cases=cases)
    run_evidence_acceptance(root=tmp_path)

    entered = threading.Event()
    release = threading.Event()
    original = runtime.run_projection_cycle_at

    def paused_projection(**kwargs: object) -> EpisodeProjection:
        entered.set()
        assert release.wait(timeout=5)
        return original(**kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(runtime, "run_projection_cycle_at", paused_projection)
    with ThreadPoolExecutor(max_workers=2) as pool:
        acceptance = pool.submit(run_evidence_acceptance, root=tmp_path)
        assert entered.wait(timeout=5)
        rollback = pool.submit(
            rollback_evidence_rollout, root=tmp_path, reason="concurrent rollback"
        )
        with pytest.raises(FutureTimeoutError):
            rollback.result(timeout=0.05)
        release.set()
        assert acceptance.result(timeout=5)["rollout"]["canary_percent"] == 5
        assert rollback.result(timeout=5)["mode"] == "shadow"

    assert load_evidence_rollout(tmp_path)["mode"] == "shadow"
