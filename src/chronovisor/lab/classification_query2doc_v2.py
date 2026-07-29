"""Coverage-first Query2doc v2 development evaluator."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
import uuid
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from chronovisor.classification.classification import (
    ClassificationError,
    default_udc_package,
)
from chronovisor.classification.classification_engine import CandidateIndex
from chronovisor.classification.classification_query_worker_v2 import (
    HEADING_ROLES,
    QUERY_POLICY,
    QUERY_PROMPT_SHA256,
    QUERY_SCHEMA,
    WORKER_SCHEMA,
)
from chronovisor.core import ollama
from chronovisor.core.durable_state import read_sealed_json, write_sealed_json
from chronovisor.core.runtime_config import load_decision_router_config
from chronovisor.core.store import CHRONOVISOR_ROOT
from chronovisor.core.timeutil import utc_iso_milliseconds as _now
from chronovisor.lab.classification_fixture_set import sha256_file
from chronovisor.lab.classification_profile_pilot import (
    notation_matches,
    query_profile_index,
)
from chronovisor.lab.classification_query2doc_pilot import candidate_blind_page
from chronovisor.research.research_scheduler import (
    research_lane,
    run_cancellable_command,
    sync_pending,
)

ARTIFACT_SCHEMA = "chronovisor.classification-query2doc-artifact.v2"
DEV_EVALUATION_SCHEMA = "chronovisor.classification-query2doc-v2-dev-evaluation.v1"
STATE_SCHEMA = "chronovisor.classification-query2doc-v2-state.v1"
CHANNEL_LIMIT = 12
FUSED_LIMIT = 12
RRF_K = 60
DEV_TARGET_HITS = 24
CHANNELS = (
    "raw_lexical",
    "raw_dense",
    "query2doc_lexical",
    "query2doc_dense",
    "role_lexical",
    "role_dense",
)
CHANNEL_QUOTAS = {
    "query2doc_dense": 4,
    "query2doc_lexical": 5,
    "role_dense": 1,
    "raw_lexical": 2,
}
CHANNEL_WEIGHTS = {
    "query2doc_lexical": 4.0,
    "role_lexical": 3.0,
    "raw_lexical": 3.0,
    "query2doc_dense": 2.0,
    "role_dense": 2.0,
    "raw_dense": 1.0,
}


def retrieval_contract() -> dict[str, Any]:
    return {
        "version": "coverage-first-v2.2",
        "artifact_schema": ARTIFACT_SCHEMA,
        "query_schema": QUERY_SCHEMA,
        "prompt_sha256": QUERY_PROMPT_SHA256,
        "channel_limit": CHANNEL_LIMIT,
        "fused_limit": FUSED_LIMIT,
        "channels": list(CHANNELS),
        "channel_quotas": dict(CHANNEL_QUOTAS),
        "channel_weights": dict(CHANNEL_WEIGHTS),
        "rrf_k": RRF_K,
        "independent_heading_roles": list(HEADING_ROLES),
        "broad_heading_search": "joined-bilingual",
        "role_heading_search": "independent-bilingual-role-round-robin",
        "fusion": "reserved-union-with-weighted-rrf-backfill",
    }


def retrieval_contract_sha256() -> str:
    payload = json.dumps(
        retrieval_contract(),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return "sha256:" + hashlib.sha256(payload).hexdigest()




def v2_dev_root(root: Path) -> Path:
    return root / "classification" / "query2doc-v2-dev"


def _write_state(
    output_root: Path,
    *,
    status: str,
    stage: str,
    **detail: Any,
) -> dict[str, Any]:
    return write_sealed_json(
        output_root / "state.json",
        {
            "schema": STATE_SCHEMA,
            "status": status,
            "stage": stage,
            "updated_at": _now(),
            **detail,
        },
        backup=True,
    )


def _artifact_path(output_root: Path, uid: str) -> Path:
    return output_root / "queries" / f"{uid}.json"


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
    output_root: Path,
    page: Mapping[str, Any],
    *,
    model: str,
    model_digest: str,
    keep_alive: str,
    read_timeout_ms: int,
) -> dict[str, Any]:
    source_sha256 = str(page.get("source_sha256") or "")
    projected = candidate_blind_page(page)
    uid = projected["uid"]
    if not source_sha256 or not uid:
        raise ClassificationError("query2doc v2 source identity is incomplete")
    path = _artifact_path(output_root, uid)
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
            raise ClassificationError("query2doc v2 generation exceeded its deadline")
        attempts += 1
        run_id = f"query2doc-v2-{uid[:12]}-{uuid.uuid4().hex[:8]}"
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
                    "chronovisor.classification.classification_query_worker_v2",
                ],
                json.dumps(payload, ensure_ascii=False, sort_keys=True),
                lease,
                timeout_seconds=remaining,
            )
        if result.status in {"cancelled", "deferred"}:
            while sync_pending():
                if time.monotonic() >= deadline:
                    raise ClassificationError(
                        "query2doc v2 foreground wait exceeded deadline"
                    )
                time.sleep(0.05)
            continue
        if result.status != "completed" or not isinstance(result.value, Mapping):
            raise ClassificationError(result.error or "query2doc v2 worker failed")
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
            raise ClassificationError("query2doc v2 worker contract mismatch")
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


def heading_query_pages(
    artifact: Mapping[str, Any],
) -> list[tuple[str, dict[str, Any]]]:
    query = artifact.get("query")
    if not isinstance(query, Mapping):
        raise ClassificationError("query2doc v2 artifact has no query")
    raw_headings = query.get("headings")
    if not isinstance(raw_headings, list):
        raise ClassificationError("query2doc v2 headings are missing")
    by_role = {
        str(heading.get("role") or ""): heading
        for heading in raw_headings
        if isinstance(heading, Mapping)
    }
    if set(by_role) != set(HEADING_ROLES):
        raise ClassificationError("query2doc v2 heading roles are incomplete")
    output = []
    for role in HEADING_ROLES:
        heading = by_role[role]
        ja = str(heading.get("ja") or "").strip()
        en = str(heading.get("en") or "").strip()
        if not ja or not en:
            raise ClassificationError("query2doc v2 bilingual heading is incomplete")
        output.append(
            (
                role,
                {
                    "title": f"{ja}\n{en}",
                    "summary": "",
                    "tags": [],
                    "raw_keywords": [],
                    "excerpt": "",
                },
            )
        )
    return output


def broad_query_page(artifact: Mapping[str, Any]) -> dict[str, Any]:
    query = artifact.get("query")
    if not isinstance(query, Mapping):
        raise ClassificationError("query2doc v2 artifact has no query")
    headings = [
        str(value).strip()
        for field in ("broad_headings_ja", "broad_headings_en")
        for value in query.get(field) or []
        if str(value).strip()
    ]
    if len(headings) < 4:
        raise ClassificationError("query2doc v2 broad headings are incomplete")
    return {
        "title": "\n".join(headings),
        "summary": "",
        "tags": [],
        "raw_keywords": [],
        "excerpt": "",
    }


def interleave_heading_candidates(
    rows_by_role: Mapping[str, Sequence[Mapping[str, Any]]],
    *,
    limit: int = CHANNEL_LIMIT,
) -> list[dict[str, Any]]:
    """Round-robin role searches so one poisoned heading cannot occupy all slots."""

    metadata: dict[str, dict[str, Any]] = {}
    heading_ranks: dict[str, dict[str, int]] = {}
    for role in HEADING_ROLES:
        for rank, row in enumerate(
            (rows_by_role.get(role) or [])[:CHANNEL_LIMIT],
            start=1,
        ):
            notation = str(row.get("notation") or "")
            if not notation:
                continue
            metadata.setdefault(
                notation,
                {
                    "notation": notation,
                    "label_en": str(row.get("label_en") or ""),
                    "label_ja": str(row.get("label_ja") or ""),
                },
            )
            heading_ranks.setdefault(notation, {})[role] = rank
    order: list[str] = []
    for depth in range(CHANNEL_LIMIT):
        for role in HEADING_ROLES:
            rows = rows_by_role.get(role) or []
            if depth >= len(rows):
                continue
            notation = str(rows[depth].get("notation") or "")
            if notation and notation not in order:
                order.append(notation)
            if len(order) >= max(1, limit):
                break
        if len(order) >= max(1, limit):
            break
    return [
        {
            "rank": rank,
            **metadata[notation],
            "heading_ranks": heading_ranks[notation],
        }
        for rank, notation in enumerate(order, start=1)
    ]


def _weighted_rrf(
    channel_rows: Mapping[str, Sequence[Mapping[str, Any]]],
    *,
    k: int = RRF_K,
) -> list[dict[str, Any]]:
    scores: dict[str, float] = {}
    ranks: dict[str, dict[str, int]] = {}
    metadata: dict[str, dict[str, Any]] = {}
    for channel in CHANNELS:
        rows = channel_rows.get(channel) or []
        weight = float(CHANNEL_WEIGHTS[channel])
        for rank, row in enumerate(rows[:CHANNEL_LIMIT], start=1):
            notation = str(row.get("notation") or "")
            if not notation:
                continue
            scores[notation] = scores.get(notation, 0.0) + weight / (k + rank)
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
            "notation": notation,
            **metadata[notation],
            "weighted_rrf_score": round(scores[notation], 9),
            "channel_ranks": ranks[notation],
        }
        for notation in order
    ]


def coverage_first_fusion(
    channel_rows: Mapping[str, Sequence[Mapping[str, Any]]],
    *,
    limit: int = FUSED_LIMIT,
) -> list[dict[str, Any]]:
    """Reserve reliable channel coverage, then fill deduplicated gaps by weighted RRF."""

    weighted = _weighted_rrf(channel_rows)
    weighted_by_notation = {row["notation"]: row for row in weighted}
    reserved_by: dict[str, list[str]] = {}
    selected: set[str] = set()
    for channel, quota in CHANNEL_QUOTAS.items():
        for row in (channel_rows.get(channel) or [])[:quota]:
            notation = str(row.get("notation") or "")
            if not notation:
                continue
            selected.add(notation)
            reserved_by.setdefault(notation, []).append(channel)
    for row in weighted:
        if len(selected) >= max(1, limit):
            break
        selected.add(str(row["notation"]))
    selected_order = sorted(
        selected,
        key=lambda notation: (
            -float(weighted_by_notation[notation]["weighted_rrf_score"]),
            min(weighted_by_notation[notation]["channel_ranks"].values()),
            notation,
        ),
    )[: max(1, limit)]
    return [
        {
            "rank": rank,
            **weighted_by_notation[notation],
            "reserved_by": reserved_by.get(notation, []),
            "selection": "reserved" if notation in reserved_by else "backfill",
        }
        for rank, notation in enumerate(selected_order, start=1)
    ]


def _matching_rank(
    candidates: Sequence[Mapping[str, Any]],
    expected: Sequence[str],
) -> int | None:
    for rank, row in enumerate(candidates, start=1):
        notation = str(row.get("notation") or "")
        if notation and notation_matches(notation, expected):
            return rank
    return None


def _metrics(cases: Sequence[Mapping[str, Any]], channel: str) -> dict[str, Any]:
    ranks = [
        int(case.get(f"{channel}_rank") or 0)
        for case in cases
        if case.get(f"{channel}_rank") is not None
    ]
    total = len(cases)
    return {
        "hit_count": len(ranks),
        "recall_at_12": len(ranks) / total if total else 0.0,
        "mrr": sum(1.0 / rank for rank in ranks) / total if total else 0.0,
    }


def evaluate_fixture(
    root: Path,
    *,
    fixture: Mapping[str, Any],
    output_root: Path,
    prior_cases: Mapping[str, Mapping[str, Any]] | None = None,
    state_stage_prefix: str,
) -> dict[str, Any]:
    raw_cases = fixture.get("cases")
    if not isinstance(raw_cases, list) or not raw_cases:
        raise ClassificationError("query2doc v2 fixture has no cases")
    package = default_udc_package()
    lexical = CandidateIndex(package)
    config = load_decision_router_config()
    model = config.primary_model
    model_digest = ollama.model_digests([model]).get(model, "")
    if not model_digest:
        raise ClassificationError(f"query2doc v2 model is not installed: {model}")
    cases: list[dict[str, Any]] = []
    total_model_calls = 0
    total_model_attempts = 0
    for offset, source in enumerate(raw_cases, start=1):
        uid = str(source.get("uid") or "")
        expected = [
            str(value)
            for value in source.get("expected_primary_notations") or []
            if str(value)
        ]
        if not uid or not expected:
            raise ClassificationError("query2doc v2 fixture case is incomplete")
        _write_state(
            output_root,
            status="running",
            stage=f"{state_stage_prefix}-generating-role-separated-queries",
            generated_count=offset - 1,
            evaluated_count=offset - 1,
            case_count=len(raw_cases),
            model=model,
        )
        artifact = generate_query_artifact(
            output_root,
            source,
            model=model,
            model_digest=model_digest,
            keep_alive=config.primary_keep_alive,
            read_timeout_ms=config.read_timeout_ms,
        )
        total_model_calls += int(artifact.get("model_calls") or 0)
        total_model_attempts += int(artifact.get("model_attempts") or 0)
        heading_pages = heading_query_pages(artifact)
        broad_page = broad_query_page(artifact)
        lexical_by_role: dict[str, list[dict[str, Any]]] = {}
        dense_by_role: dict[str, list[dict[str, Any]]] = {}
        for role, query_page in heading_pages:
            lexical_by_role[role] = lexical.candidates(
                query_page,
                limit=CHANNEL_LIMIT,
            )
            dense_by_role[role] = query_profile_index(
                root,
                query_page,
                limit=CHANNEL_LIMIT,
            )
        prior = (prior_cases or {}).get(uid)
        if prior is None:
            raw_page = {
                "title": str(source.get("title") or ""),
                "summary": str(source.get("summary") or ""),
                "tags": list(source.get("tags") or []),
                "raw_keywords": list(source.get("raw_keywords") or []),
                "excerpt": str(source.get("excerpt") or ""),
            }
            raw_lexical = lexical.candidates(raw_page, limit=CHANNEL_LIMIT)
            raw_dense = query_profile_index(root, raw_page, limit=CHANNEL_LIMIT)
        else:
            raw_lexical = [
                dict(row)
                for row in prior.get("raw_lexical_candidates") or []
            ][:CHANNEL_LIMIT]
            raw_dense = [
                dict(row)
                for row in prior.get("raw_dense_candidates") or []
            ][:CHANNEL_LIMIT]
        channel_rows = {
            "raw_lexical": raw_lexical,
            "raw_dense": raw_dense,
            "query2doc_lexical": lexical.candidates(
                broad_page,
                limit=CHANNEL_LIMIT,
            ),
            "query2doc_dense": query_profile_index(
                root,
                broad_page,
                limit=CHANNEL_LIMIT,
            ),
            "role_lexical": interleave_heading_candidates(
                lexical_by_role,
                limit=CHANNEL_LIMIT,
            ),
            "role_dense": interleave_heading_candidates(
                dense_by_role,
                limit=CHANNEL_LIMIT,
            ),
        }
        fused = coverage_first_fusion(channel_rows)
        case: dict[str, Any] = {
            "position": int(source.get("position") or offset),
            "uid": uid,
            "source_sha256": str(source.get("source_sha256") or ""),
            "title": str(source.get("title") or ""),
            "expected_primary_notations": expected,
            "gold_rationale": str(source.get("gold_rationale") or ""),
            "query_artifact_path": str(_artifact_path(output_root, uid)),
            "query_artifact_sha256": sha256_file(_artifact_path(output_root, uid)),
            "query": artifact["query"],
            "broad_query_page": broad_page,
            "heading_lexical_candidates": lexical_by_role,
            "heading_dense_candidates": dense_by_role,
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
            output_root,
            status="running",
            stage=f"{state_stage_prefix}-retrieving-independent-headings",
            generated_count=offset,
            evaluated_count=offset,
            case_count=len(raw_cases),
            model=model,
        )
    metrics = {
        channel: _metrics(cases, channel)
        for channel in (*CHANNELS, "fused")
    }
    return {
        "evaluated_at": _now(),
        "model": model,
        "model_digest": model_digest,
        "prompt_sha256": QUERY_PROMPT_SHA256,
        "model_calls": total_model_calls,
        "model_attempts": total_model_attempts,
        "case_count": len(cases),
        "channel_limit": CHANNEL_LIMIT,
        "fused_limit": FUSED_LIMIT,
        "fusion": {
            "algorithm": "coverage-first-reserved-union-with-weighted-rrf-backfill",
            "channel_quotas": dict(CHANNEL_QUOTAS),
            "channel_weights": dict(CHANNEL_WEIGHTS),
            "rrf_k": RRF_K,
            "independent_heading_roles": list(HEADING_ROLES),
            "heading_merge": "role-round-robin",
        },
        "retrieval_contract": retrieval_contract(),
        "retrieval_contract_sha256": retrieval_contract_sha256(),
        "metrics": metrics,
        "classification_judge_calls": 0,
        "page_mutations": 0,
        "cases": cases,
    }


def evaluate_dev(root: Path) -> dict[str, Any]:
    output_root = v2_dev_root(root)
    prior_path = root / "classification" / "query2doc-unseen" / "evaluation.json"
    prior = read_sealed_json(prior_path)
    fixture_path = Path(str(prior.get("fixture_path") or ""))
    fixture = read_sealed_json(fixture_path)
    prior_by_uid = {
        str(case.get("uid") or ""): case
        for case in prior.get("cases") or []
        if isinstance(case, Mapping)
    }
    evaluation = evaluate_fixture(
        root,
        fixture=fixture,
        output_root=output_root,
        prior_cases=prior_by_uid,
        state_stage_prefix="dev",
    )
    fused_hits = int((evaluation["metrics"]["fused"] or {}).get("hit_count") or 0)
    prior_fused_hits = int(
        ((prior.get("metrics") or {}).get("fused") or {}).get("hit_count") or 0
    )
    receipt = {
        "schema": DEV_EVALUATION_SCHEMA,
        "development_data_only": True,
        "source_unseen_evaluation_path": str(prior_path),
        "source_unseen_evaluation_sha256": sha256_file(prior_path),
        "source_fixture_path": str(fixture_path),
        "source_fixture_sha256": sha256_file(fixture_path),
        **evaluation,
        "prior_equal_rrf_hit_count": prior_fused_hits,
        "development_target_hits": DEV_TARGET_HITS,
        "development_target_met": fused_hits >= DEV_TARGET_HITS,
        "decision": (
            "freeze-v2-for-new-unseen-evaluation"
            if fused_hits >= DEV_TARGET_HITS
            else "revise-v2-before-new-unseen-evaluation"
        ),
        "new_unseen_evaluation_authorized": fused_hits >= DEV_TARGET_HITS,
    }
    evaluation_path = output_root / "evaluation.json"
    write_sealed_json(evaluation_path, receipt, backup=True)
    _write_state(
        output_root,
        status="qualified" if fused_hits >= DEV_TARGET_HITS else "rejected",
        stage="dev-query2doc-v2-complete",
        decision=receipt["decision"],
        case_count=receipt["case_count"],
        fused_hit_count=fused_hits,
        prior_equal_rrf_hit_count=prior_fused_hits,
        development_target_hits=DEV_TARGET_HITS,
        new_unseen_evaluation_authorized=receipt[
            "new_unseen_evaluation_authorized"
        ],
        model_calls=receipt["model_calls"],
        classification_judge_calls=0,
        page_mutations=0,
    )
    return read_sealed_json(evaluation_path)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=CHRONOVISOR_ROOT)
    args = parser.parse_args(argv)
    try:
        result = evaluate_dev(args.root.expanduser())
    except (
        ClassificationError,
        OSError,
        TypeError,
        ValueError,
        json.JSONDecodeError,
    ) as exc:
        _write_state(
            v2_dev_root(args.root.expanduser()),
            status="failed",
            stage="dev-query2doc-v2-failed",
            error=str(exc),
        )
        print(str(exc), file=sys.stderr)
        return 1
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
