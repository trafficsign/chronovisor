"""Candidate-blind query2doc retrieval gate for the fixed ten diagnostics."""

from __future__ import annotations

from chronovisor.core.timeutil import utc_iso_milliseconds as _now

import argparse
import json
import sys
import time
import uuid
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from chronovisor.core import ollama
from chronovisor.classification.classification import ClassificationError, default_udc_package
from chronovisor.classification.classification_engine import CandidateIndex
from chronovisor.lab.classification_fixture_set import read_jsonl, sha256_file
from chronovisor.lab.classification_profile_pilot import (
    FIXTURE_EPOCH,
    notation_matches,
    profile_pilot_root,
    query_profile_index,
)
from chronovisor.classification.classification_query_worker import (
    QUERY_POLICY,
    QUERY_PROMPT_SHA256,
    QUERY_SCHEMA,
    WORKER_SCHEMA,
)
from chronovisor.core.durable_state import read_sealed_json, write_sealed_json
from chronovisor.research.research_scheduler import (
    research_lane,
    run_cancellable_command,
    sync_pending,
)
from chronovisor.core.runtime_config import load_decision_router_config
from chronovisor.core.store import CHRONOVISOR_ROOT

ARTIFACT_SCHEMA = "chronovisor.classification-query2doc-artifact.v1"
EVALUATION_SCHEMA = "chronovisor.classification-query2doc-evaluation.v1"
MANIFEST_SCHEMA = "chronovisor.classification-query2doc-manifest.v1"
STATE_SCHEMA = "chronovisor.classification-query2doc-pilot-state.v1"
CHANNEL_LIMIT = 12
FUSED_LIMIT = 12
RRF_K = 60
PASS_HITS = 8
CHANNELS = (
    "raw_lexical",
    "raw_dense",
    "query2doc_lexical",
    "query2doc_dense",
)




def query2doc_root(root: Path) -> Path:
    return root / "classification" / "query2doc-pilot"


def _fixture_path(root: Path) -> Path:
    return (
        root
        / "classification"
        / "fixtures"
        / "epochs"
        / FIXTURE_EPOCH
        / "adjudication.jsonl"
    )


def _review_path(root: Path) -> Path:
    return root / "classification" / "annif-pilot" / "early-council-review.json"


def _write_state(
    root: Path,
    *,
    status: str,
    stage: str,
    **detail: Any,
) -> dict[str, Any]:
    return write_sealed_json(
        query2doc_root(root) / "state.json",
        {
            "schema": STATE_SCHEMA,
            "status": status,
            "stage": stage,
            "updated_at": _now(),
            **detail,
        },
        backup=True,
    )


def candidate_blind_page(page: Mapping[str, Any]) -> dict[str, str]:
    """Project a fixture row without candidates, gold, tags or case metadata."""

    return {
        "uid": str(page.get("uid") or ""),
        "title": str(page.get("title") or ""),
        "summary": str(page.get("summary") or ""),
        "excerpt": str(page.get("excerpt") or "")[:2_400],
    }


def _artifact_path(root: Path, uid: str) -> Path:
    return query2doc_root(root) / "queries" / f"{uid}.json"


def _cached_artifact(
    path: Path,
    *,
    source_sha256: str,
    model: str,
    model_digest: str,
) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    artifact = read_sealed_json(path)
    query = artifact.get("query")
    if (
        artifact.get("schema") == ARTIFACT_SCHEMA
        and artifact.get("source_sha256") == source_sha256
        and artifact.get("model") == model
        and artifact.get("model_digest") == model_digest
        and artifact.get("prompt_sha256") == QUERY_PROMPT_SHA256
        and isinstance(query, Mapping)
        and query.get("schema") == QUERY_SCHEMA
    ):
        return artifact
    return None


def generate_query_artifact(
    root: Path,
    page: Mapping[str, Any],
    *,
    model: str,
    model_digest: str,
    keep_alive: str,
    read_timeout_ms: int,
    artifact_path: Path | None = None,
) -> dict[str, Any]:
    source_sha256 = str(page.get("source_sha256") or "")
    projected = candidate_blind_page(page)
    uid = projected["uid"]
    if not source_sha256 or not uid:
        raise ClassificationError("query2doc source identity is incomplete")
    path = artifact_path or _artifact_path(root, uid)
    cached = _cached_artifact(
        path,
        source_sha256=source_sha256,
        model=model,
        model_digest=model_digest,
    )
    if cached is not None:
        return cached
    payload = {
        "schema": WORKER_SCHEMA,
        "model": model,
        "model_digest": model_digest,
        "keep_alive": keep_alive,
        "read_timeout_ms": read_timeout_ms,
        "page": projected,
    }
    attempts = 0
    deadline = time.monotonic() + max(60.0, read_timeout_ms / 1_000 + 30)
    while True:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise ClassificationError("query2doc generation exceeded its deadline")
        attempts += 1
        run_id = f"query2doc-{uid[:12]}-{uuid.uuid4().hex[:8]}"
        with research_lane(
            run_id,
            enabled=True,
            mode="on",
            purpose="explicit",
            needs_model=True,
        ) as lease:
            result = run_cancellable_command(
                [
                    sys.executable,
                    "-m",
                    "chronovisor.classification.classification_query_worker",
                ],
                json.dumps(payload, ensure_ascii=False, sort_keys=True),
                lease,
                timeout_seconds=remaining,
            )
        if result.status in {"cancelled", "deferred"}:
            while sync_pending():
                if time.monotonic() >= deadline:
                    raise ClassificationError(
                        "query2doc foreground wait exceeded deadline"
                    )
                time.sleep(0.05)
            continue
        if result.status != "completed" or not isinstance(result.value, Mapping):
            raise ClassificationError(result.error or "query2doc worker failed")
        worker = dict(result.value)
        query = worker.get("query")
        if (
            worker.get("schema") != WORKER_SCHEMA
            or worker.get("uid") != uid
            or worker.get("model") != model
            or worker.get("model_digest") != model_digest
            or worker.get("prompt_sha256") != QUERY_PROMPT_SHA256
            or not isinstance(query, Mapping)
            or query.get("schema") != QUERY_SCHEMA
            or int(worker.get("model_calls") or 0) != 1
        ):
            raise ClassificationError("query2doc worker contract mismatch")
        artifact = {
            "schema": ARTIFACT_SCHEMA,
            "created_at": _now(),
            "uid": uid,
            "source_sha256": source_sha256,
            "model": model,
            "model_digest": model_digest,
            "prompt_sha256": QUERY_PROMPT_SHA256,
            "input_fields": list(QUERY_POLICY["input_fields"]),
            "forbidden_input_fields": list(QUERY_POLICY["forbidden_input_fields"]),
            "model_calls": 1,
            "model_attempts": attempts,
            "query": dict(query),
        }
        write_sealed_json(path, artifact, backup=True)
        return read_sealed_json(path)


def query_text(artifact: Mapping[str, Any]) -> str:
    query = artifact.get("query")
    if not isinstance(query, Mapping):
        raise ClassificationError("query2doc artifact has no query")
    headings = [
        str(value).strip()
        for field in ("subject_headings_ja", "subject_headings_en")
        for value in query.get(field) or []
        if str(value).strip()
    ]
    if len(headings) < 4:
        raise ClassificationError("query2doc artifact has too few headings")
    return "\n".join(headings)


def reciprocal_rank_fusion(
    channel_rows: Mapping[str, Sequence[Mapping[str, Any]]],
    *,
    limit: int = FUSED_LIMIT,
    k: int = RRF_K,
) -> list[dict[str, Any]]:
    scores: dict[str, float] = {}
    ranks: dict[str, dict[str, int]] = {}
    metadata: dict[str, dict[str, Any]] = {}
    for channel in CHANNELS:
        rows = channel_rows.get(channel) or []
        for rank, row in enumerate(rows[:CHANNEL_LIMIT], start=1):
            notation = str(row.get("notation") or "")
            if not notation:
                continue
            scores[notation] = scores.get(notation, 0.0) + 1.0 / (k + rank)
            ranks.setdefault(notation, {})[channel] = rank
            metadata.setdefault(
                notation,
                {
                    "label_en": str(row.get("label_en") or ""),
                    "label_ja": str(row.get("label_ja") or ""),
                },
            )
    order = sorted(
        scores,
        key=lambda notation: (
            -scores[notation],
            min(ranks[notation].values()),
            notation,
        ),
    )
    return [
        {
            "rank": rank,
            "notation": notation,
            **metadata[notation],
            "rrf_score": round(scores[notation], 9),
            "channel_ranks": ranks[notation],
        }
        for rank, notation in enumerate(order[: max(1, limit)], start=1)
    ]


def _matching_rank(
    rows: Sequence[Mapping[str, Any]],
    expected: Sequence[str],
) -> int | None:
    return next(
        (
            rank
            for rank, row in enumerate(rows, start=1)
            if notation_matches(str(row.get("notation") or ""), expected)
        ),
        None,
    )


def _channel_metrics(cases: Sequence[Mapping[str, Any]], channel: str) -> dict[str, Any]:
    key = f"{channel}_rank"
    ranks = [
        int(case[key])
        for case in cases
        if isinstance(case.get(key), int)
    ]
    return {
        "hit_count": len(ranks),
        "recall_at_12": len(ranks) / max(1, len(cases)),
        "mrr": sum(1 / rank for rank in ranks) / max(1, len(cases)),
    }


def evaluate_fixed_cases(root: Path) -> dict[str, Any]:
    review = read_sealed_json(_review_path(root))
    prior = read_sealed_json(profile_pilot_root(root) / "evaluation.json")
    source_rows = read_jsonl(_fixture_path(root))
    source_by_uid = {str(row.get("uid") or ""): row for row in source_rows}
    prior_by_uid = {
        str(row.get("uid") or ""): row for row in prior.get("cases") or []
    }
    package = default_udc_package()
    lexical = CandidateIndex(package)
    cases = []
    total_model_calls = 0
    total_model_attempts = 0
    config = load_decision_router_config()
    model = config.primary_model
    model_digest = ollama.model_digests([model]).get(model, "")
    if not model_digest:
        raise ClassificationError(f"query2doc model is not installed: {model}")
    for offset, reviewed in enumerate(review.get("cases") or [], start=1):
        uid = str(reviewed.get("uid") or "")
        source = source_by_uid.get(uid)
        prior_case = prior_by_uid.get(uid)
        if source is None or prior_case is None:
            raise ClassificationError(f"query2doc source row missing for {uid}")
        _write_state(
            root,
            status="running",
            stage="generating-candidate-blind-queries",
            generated_count=offset - 1,
            case_count=len(review.get("cases") or []),
            model=model,
        )
        artifact = generate_query_artifact(
            root,
            source,
            model=model,
            model_digest=model_digest,
            keep_alive=config.primary_keep_alive,
            read_timeout_ms=config.read_timeout_ms,
        )
        total_model_calls += int(artifact.get("model_calls") or 0)
        total_model_attempts += int(artifact.get("model_attempts") or 0)
        abstract_query = query_text(artifact)
        query_page = {
            "title": abstract_query,
            "summary": "",
            "tags": [],
            "raw_keywords": [],
            "excerpt": "",
        }
        channel_rows = {
            "raw_lexical": [
                dict(row)
                for row in prior_case.get("baseline_candidates") or []
            ][:CHANNEL_LIMIT],
            "raw_dense": [
                dict(row)
                for row in prior_case.get("profile_candidates") or []
            ][:CHANNEL_LIMIT],
            "query2doc_lexical": lexical.candidates(
                query_page,
                limit=CHANNEL_LIMIT,
            ),
            "query2doc_dense": query_profile_index(
                root,
                query_page,
                limit=CHANNEL_LIMIT,
            ),
        }
        fused = reciprocal_rank_fusion(channel_rows)
        expected = [
            str(value)
            for value in reviewed.get("expected_primary_notations") or []
            if str(value)
        ]
        case = {
            "case_number": int(reviewed.get("case_number") or 0),
            "uid": uid,
            "title": str(source.get("title") or ""),
            "expected_primary_notations": expected,
            "query_artifact_path": str(_artifact_path(root, uid)),
            "query_artifact_sha256": sha256_file(_artifact_path(root, uid)),
            "query": artifact["query"],
            "fused_candidates": fused,
        }
        for channel, rows in channel_rows.items():
            case[f"{channel}_candidates"] = rows
            case[f"{channel}_rank"] = _matching_rank(rows, expected)
            case[f"{channel}_hit"] = case[f"{channel}_rank"] is not None
        case["fused_rank"] = _matching_rank(fused, expected)
        case["fused_hit"] = case["fused_rank"] is not None
        cases.append(case)
        _write_state(
            root,
            status="running",
            stage="retrieving-query2doc-channels",
            generated_count=offset,
            evaluated_count=offset,
            case_count=len(review.get("cases") or []),
            model=model,
        )
    metrics = {
        channel: _channel_metrics(cases, channel)
        for channel in (*CHANNELS, "fused")
    }
    decision = (
        "qualify-query2doc-retrieval"
        if metrics["fused"]["hit_count"] >= PASS_HITS
        else "reject-query2doc-retrieval"
    )
    receipt = {
        "schema": EVALUATION_SCHEMA,
        "evaluated_at": _now(),
        "source_review_path": str(_review_path(root)),
        "source_review_sha256": sha256_file(_review_path(root)),
        "source_fixture_path": str(_fixture_path(root)),
        "source_fixture_sha256": sha256_file(_fixture_path(root)),
        "source_profile_evaluation_path": str(
            profile_pilot_root(root) / "evaluation.json"
        ),
        "source_profile_evaluation_sha256": sha256_file(
            profile_pilot_root(root) / "evaluation.json"
        ),
        "input_contract": {
            "included": list(QUERY_POLICY["input_fields"]),
            "forbidden": list(QUERY_POLICY["forbidden_input_fields"]),
            "candidates_exposed_to_model": False,
            "expected_notations_exposed_to_model": False,
        },
        "model": model,
        "model_digest": model_digest,
        "prompt_sha256": QUERY_PROMPT_SHA256,
        "model_calls": total_model_calls,
        "model_attempts": total_model_attempts,
        "case_count": len(cases),
        "channel_limit": CHANNEL_LIMIT,
        "fused_limit": FUSED_LIMIT,
        "fusion": {
            "algorithm": "equal-weight reciprocal-rank-fusion",
            "k": RRF_K,
            "channels": list(CHANNELS),
            "weights": {channel: 1 for channel in CHANNELS},
            "tuned_on_fixed_ten": False,
        },
        "match_policy": "exact target, target descendant, or explicit UDC range",
        "metrics": metrics,
        "minimum_fused_hits": PASS_HITS,
        "decision": decision,
        "unseen_evaluation_authorized": decision == "qualify-query2doc-retrieval",
        "larger_corpus_evaluation_authorized": False,
        "classification_judge_calls": 0,
        "page_mutations": 0,
        "cases": cases,
    }
    output_root = query2doc_root(root)
    write_sealed_json(output_root / "evaluation.json", receipt, backup=True)
    manifest = {
        "schema": MANIFEST_SCHEMA,
        "created_at": _now(),
        "evaluation_path": str(output_root / "evaluation.json"),
        "evaluation_sha256": sha256_file(output_root / "evaluation.json"),
        "query_count": len(cases),
        "model": model,
        "model_digest": model_digest,
        "prompt_sha256": QUERY_PROMPT_SHA256,
        "query_artifacts_separate_from_udc_profiles": True,
        "udc_profile_mutations": 0,
        "external_library_records_used": 0,
        "classification_judge_calls": 0,
        "page_mutations": 0,
    }
    write_sealed_json(output_root / "manifest.json", manifest, backup=True)
    _write_state(
        root,
        status=(
            "qualified"
            if decision == "qualify-query2doc-retrieval"
            else "rejected"
        ),
        stage="fixed-ten-query2doc-complete",
        decision=decision,
        fused_hit_count=metrics["fused"]["hit_count"],
        case_count=len(cases),
        minimum_fused_hits=PASS_HITS,
        unseen_evaluation_authorized=receipt["unseen_evaluation_authorized"],
        larger_corpus_evaluation_authorized=False,
        model_calls=total_model_calls,
        classification_judge_calls=0,
        page_mutations=0,
    )
    return read_sealed_json(output_root / "evaluation.json")


def run_pilot(root: Path) -> dict[str, Any]:
    try:
        _write_state(
            root,
            status="running",
            stage="preparing-fixed-ten-query2doc",
            generated_count=0,
            evaluated_count=0,
        )
        return evaluate_fixed_cases(root)
    except Exception as exc:
        _write_state(
            root,
            status="failed",
            stage="query2doc-pilot-failed",
            error=f"{type(exc).__name__}: {exc}",
        )
        raise


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Run the candidate-blind fixed-ten query2doc retrieval gate"
    )
    parser.add_argument("--root", type=Path, default=CHRONOVISOR_ROOT)
    arguments = parser.parse_args(argv)
    result = run_pilot(arguments.root)
    summary = {
        "schema": result["schema"],
        "case_count": result["case_count"],
        "model": result["model"],
        "model_calls": result["model_calls"],
        "metrics": result["metrics"],
        "minimum_fused_hits": result["minimum_fused_hits"],
        "decision": result["decision"],
        "unseen_evaluation_authorized": result[
            "unseen_evaluation_authorized"
        ],
    }
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True, indent=2))


if __name__ == "__main__":
    main()
