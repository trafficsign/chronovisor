"""Search ranking evaluation for Chronovisor.

This is intentionally separate from ``recall_eval.py``. Recall eval measures
whether the synchronous gate injects useful context; this module measures the
ranking quality of search candidates.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import math
import os
import statistics
import tempfile
import time
from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass, replace
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from chronovisor.core.canonical_json import (
    canonical_json_sha256_stringifying as _canonical_json_sha256,
)
from chronovisor.core.feedback_ledger import active_feedback_rows
from chronovisor.core.negative_feedback import apply_penalties, penalties_for_query
from chronovisor.core.page_mutation import decision_authority_lock
from chronovisor.core.pipeline import (
    PipelineConfig,
    PipelineDependencies,
    apply_negative_feedback_stage,
    apply_rerank_stage,
    production_pipeline_config,
    run_search_pipeline,
)
from chronovisor.core.reranker import rerank_results
from chronovisor.core.runtime_config import (
    load_negative_feedback_config,
    load_reranker_config,
    load_search_embedding_config,
    runtime_repo_root,
)
from chronovisor.core.store import CHRONOVISOR_ROOT, find_page
from chronovisor.decision import decision_authority
from chronovisor.decision import decision_lane_prompts as _decision_lane_prompts
from chronovisor.decision.decision_schema_manifest import FRONTIER_LABEL_SCHEMA
from chronovisor.decision.frontier_guard import is_human_required_result
from chronovisor.decision.semantic_hold import (
    LOCAL_SEMANTIC_NO_QUORUM,
    build_semantic_no_quorum_hold,
    canonical_sha256,
    frontier_failure_class,
    is_local_semantic_no_quorum,
    persisted_semantic_no_quorum_hold,
    semantic_no_quorum_hold_error,
)
from chronovisor.search.search import (
    ACTIVE_SEARCH_POLICY_FILE,
    apply_filters,
    context_seed_results,
    fuse_results,
    get_bm25,
    graph_expand_results,
    graph_query_context,
    last_search_trace,
    load_active_fusion_weights,
    semantic_search,
    semantic_verify,
    usage_prior_results,
)
from chronovisor.search.search import (
    DEFAULT_FUSION_WEIGHTS as DEFAULT_FUSION_WEIGHTS,
)

REPO_ROOT = runtime_repo_root()
RECALL_DIR = CHRONOVISOR_ROOT / "recall"
RECALL_FEEDBACK_FILE = RECALL_DIR / "feedback.jsonl"
RECALL_LOG_FILE = RECALL_DIR / "recall-log.jsonl"
GOLDEN_FILE = RECALL_DIR / "search-golden.jsonl"
LABEL_QUEUE_FILE = RECALL_DIR / "search-label-queue.jsonl"
FAILURE_INDEX_FILE = RECALL_DIR / "search-failures.jsonl"
BASELINE_DIR = CHRONOVISOR_ROOT / "runtime" / "search-eval"
MANUAL_MANIFEST_FILE = BASELINE_DIR / "manual-94-manifest.json"
LOCKED_E2E_ARTIFACT = BASELINE_DIR / "recall-field-locked-e2e.json"
SELF_TUNE_HISTORY_FILE = BASELINE_DIR / "self-tune-history.jsonl"
SELF_TUNE_ATTEMPT_FILE = BASELINE_DIR / "self-tune-attempt.json"
SEALED_MANIFEST_SCHEMA_VERSION = 2

DEFAULT_VARIANTS = (
    "bm25",
    "semantic",
    "hybrid-current",
    "hybrid-plain-rrf",
    "hybrid-graph",
)

FRONTIER_PENDING_STATUSES = {
    "",
    "pending_review",
    "pending_frontier_review",
    "frontier_retry",
    "frontier_uncertain",
}

FRONTIER_TERMINAL_STATUSES = {
    "frontier_approved",
    "frontier_rejected",
    "frontier_quarantined",
    "human_required",
}
DEFAULT_QUARANTINE_RETRY_SECONDS = 6 * 60 * 60
SEARCH_LABEL_LANE = "search_label"
SEARCH_LABEL_PROMPT_POLICY_VERSION = (
    _decision_lane_prompts.SEARCH_LABEL_PROMPT_POLICY_VERSION
)
_str_tuple = _decision_lane_prompts._str_tuple
_str_list = _decision_lane_prompts._str_list
_page_for_label = _decision_lane_prompts._page_for_label
_page_excerpt = _decision_lane_prompts._page_excerpt
_candidate_label_pages = _decision_lane_prompts._candidate_label_pages
build_frontier_label_prompt = _decision_lane_prompts.build_frontier_label_prompt
SEARCH_SELF_TUNE_LANE = "search_self_tune"
SEARCH_REVIEW_ARTIFACT_SCHEMA_VERSION = 2
RQ_PROJECTION_POLICY_SHA256 = canonical_sha256(
    {
        "version": 1,
        "maximum_page_bytes": 12_000,
        "maximum_total_bytes": 32_000,
        "format": "[PAGE <page_id>]\\n<utf8-prefix>",
        "source": "independent_page_snapshot",
    }
)
ALREADY_APPLIED_RECOVERY = "already_applied_exact_postimage"

FrontierLabelReviewer = Callable[[dict[str, Any]], dict[str, Any]]


@dataclass(frozen=True)
class SearchExample:
    query: str
    expected_pages: tuple[str, ...] = ()
    negative_pages: tuple[str, ...] = ()
    stale_pages: tuple[str, ...] = ()
    split: str = "dev"
    language: str = "unknown"
    kind: str = "manual"
    source: str = "manual"
    ref: str = ""
    ts: str = ""
    reviewed: bool = False

    @property
    def positive(self) -> bool:
        return bool(self.expected_pages)

    @property
    def bad_pages(self) -> tuple[str, ...]:
        return tuple(dict.fromkeys(self.negative_pages + self.stale_pages))


def _sealed_manifest_entry(example: SearchExample) -> dict[str, Any]:
    entry = {
        "query_sha256": hashlib.sha256(example.query.encode("utf-8")).hexdigest(),
        "ref": example.ref,
        "source": example.source,
        "split": example.split,
        "language": example.language,
        "kind": example.kind,
        "reviewed": example.reviewed,
        "expected_pages": list(example.expected_pages),
        "negative_pages": list(example.negative_pages),
        "stale_pages": list(example.stale_pages),
    }
    return {**entry, "entry_sha256": _canonical_json_sha256(entry)}


sealed_manifest_entry = _sealed_manifest_entry


def write_sealed_manifest(
    examples: list[SearchExample],
    output_file: Path,
    *,
    review_ledger_file: Path | None = None,
) -> dict[str, Any]:
    """Freeze a reviewed cohort before evaluation without storing query text."""

    entries = sorted(
        (_sealed_manifest_entry(example) for example in examples),
        key=lambda row: (str(row["query_sha256"]), str(row["ref"])),
    )
    ledger_path = review_ledger_file.expanduser().resolve(strict=False) if review_ledger_file else None
    try:
        ledger_bytes = ledger_path.read_bytes() if ledger_path else b""
        ledger_sha256 = hashlib.sha256(ledger_bytes).hexdigest() if ledger_path else ""
        frozen_at = datetime.fromtimestamp(ledger_path.stat().st_mtime, UTC).isoformat(
            timespec="microseconds"
        ).replace("+00:00", "Z") if ledger_path else "1970-01-01T00:00:00Z"
    except OSError:
        ledger_sha256 = ""
        frozen_at = ""
    review_ledger = {
        "path": str(ledger_path) if ledger_path else "",
        "file_sha256": ledger_sha256,
        "head_sha256": _canonical_json_sha256(
            {
                "file_sha256": ledger_sha256,
                "entry_sha256": [str(entry["entry_sha256"]) for entry in entries],
            }
        ),
    }
    unsigned = {
        "schema_version": SEALED_MANIFEST_SCHEMA_VERSION,
        "examples": len(entries),
        "frozen_at": frozen_at,
        "review_ledger": review_ledger,
        "entries": entries,
    }
    payload = {**unsigned, "manifest_sha256": _canonical_json_sha256(unsigned)}
    _atomic_write_text(
        output_file,
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    )
    return {
        "output_file": str(output_file),
        "examples": len(entries),
        "manifest_sha256": payload["manifest_sha256"],
    }


def _wilson_lower_bound(successes: int, total: int, z: float = 1.96) -> float | None:
    if total <= 0:
        return None
    proportion = successes / total
    denominator = 1.0 + (z * z / total)
    centre = proportion + (z * z / (2.0 * total))
    margin = z * math.sqrt(
        (proportion * (1.0 - proportion) / total) + (z * z / (4.0 * total * total))
    )
    return round((centre - margin) / denominator, 6)


def write_locked_e2e_artifact(
    payload: dict[str, Any],
    examples: list[SearchExample],
    *,
    path: Path = LOCKED_E2E_ARTIFACT,
    variant: str = "hybrid-rerank",
    frozen_manifest: Path = MANUAL_MANIFEST_FILE,
) -> dict[str, Any]:
    """Seal the full retrieval→certificate→commit locked evaluation gate."""

    variant_payload = payload.get("variants", {}).get(variant, {})
    metrics = variant_payload.get("metrics", {})
    processor = metrics.get("processor", {}) if isinstance(metrics, dict) else {}
    evidence = processor.get("evidence_kind", {}) if isinstance(processor, dict) else {}
    evaluated_entries = sorted(
        (_sealed_manifest_entry(example) for example in examples),
        key=lambda row: (str(row["query_sha256"]), str(row["ref"])),
    )
    try:
        frozen_manifest_bytes = frozen_manifest.read_bytes()
        manifest = json.loads(frozen_manifest_bytes.decode("utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        frozen_manifest_bytes = b""
        manifest = {}
    manifest_unsigned = {
        key: value for key, value in manifest.items() if key != "manifest_sha256"
    } if isinstance(manifest, dict) else {}
    manifest_sha = str(manifest.get("manifest_sha256") or "") if isinstance(manifest, dict) else ""
    review_ledger = manifest.get("review_ledger") if isinstance(manifest, dict) else None
    try:
        ledger_path = Path(str(review_ledger.get("path") or "")).expanduser().resolve(strict=False)
        ledger_sha256 = hashlib.sha256(ledger_path.read_bytes()).hexdigest()
    except (AttributeError, OSError):
        ledger_sha256 = ""
    expected_review_head = _canonical_json_sha256(
        {
            "file_sha256": ledger_sha256,
            "entry_sha256": [str(entry["entry_sha256"]) for entry in evaluated_entries],
        }
    )
    generated_raw = str(payload.get("generated_at") or "")
    try:
        frozen_dt = datetime.fromisoformat(
            str(manifest.get("frozen_at") or "").replace("Z", "+00:00")
        )
        generated_dt = datetime.fromisoformat(generated_raw.replace("Z", "+00:00"))
        preregistered = bool(
            frozen_dt.tzinfo is not None
            and generated_dt.tzinfo is not None
            and frozen_dt.utcoffset() == timedelta(0)
            and generated_dt.utcoffset() == timedelta(0)
            and frozen_dt < generated_dt
        )
    except (TypeError, ValueError):
        preregistered = False
    frozen_manifest_valid = bool(
        isinstance(manifest, dict)
        and manifest.get("schema_version") == SEALED_MANIFEST_SCHEMA_VERSION
        and manifest.get("examples") == 94
        and manifest.get("entries") == evaluated_entries
        and len({str(entry.get("entry_sha256") or "") for entry in evaluated_entries}) == 94
        and all(entry.get("reviewed") is True for entry in evaluated_entries)
        and manifest_sha == _canonical_json_sha256(manifest_unsigned)
        and isinstance(review_ledger, dict)
        and review_ledger.get("file_sha256") == ledger_sha256
        and len(ledger_sha256) == 64
        and review_ledger.get("head_sha256") == expected_review_head
        and preregistered
    )
    precision = processor.get("precision") if isinstance(processor, dict) else None
    related_recall = (
        float(processor.get("related_recall") or 0.0)
        if isinstance(processor, dict)
        else 0.0
    )
    labeled = int(processor.get("labeled_selected_pages") or 0)
    true_positive = int(processor.get("true_positive_pages") or 0)
    rich_precision = evidence.get("rich", {}).get("precision")
    pointer_precision = evidence.get("pointer", {}).get("precision")
    authority_cases = variant_payload.get("authority_cases")
    cases = authority_cases if isinstance(authority_cases, list) else []
    gates = {
        "sealed_manual_94": (
            len(examples) == 94
            and len(cases) == 94
            and frozen_manifest_valid
        ),
        "rerank_recall_at_5": float(metrics.get("recall_at_5") or 0.0) >= 0.535,
        "negative_hit_rate": (
            float(metrics.get("negative_hit_rate_at_20") or 0.0) <= 0.20
        ),
        "processor_precision": isinstance(precision, int | float)
        and float(precision) >= 0.90,
        "processor_related_recall": related_recall >= 0.535,
        "rich_precision": isinstance(rich_precision, int | float)
        and float(rich_precision) >= 0.90,
        "pointer_precision": isinstance(pointer_precision, int | float)
        and float(pointer_precision) >= 0.90,
        "latency": float(metrics.get("latency_ms", {}).get("max") or 0.0) <= 4_000,
    }
    from chronovisor.recall.recall_answer_eval import (
        builtin_field_environment_identity,
    )

    environment_epoch = builtin_field_environment_identity()
    authority_metrics = {
        "recall_at_5": metrics.get("recall_at_5"),
        "negative_hit_rate_at_20": metrics.get("negative_hit_rate_at_20"),
        "latency_ms": {
            "max": metrics.get("latency_ms", {}).get("max")
            if isinstance(metrics.get("latency_ms"), dict)
            else None
        },
        "processor": {
            "precision": precision,
            "related_recall": related_recall,
            "evidence_kind": {
                "rich": {"precision": rich_precision},
                "pointer": {"precision": pointer_precision},
            },
        },
    }
    unsigned = {
        "schema_version": 2,
        "generated_at": generated_raw,
        "variant": variant,
        "manifest_sha256": manifest_sha,
        "manifest": manifest,
        "frozen_manifest_sha256": (
            hashlib.sha256(frozen_manifest_bytes).hexdigest()
            if frozen_manifest_bytes
            else ""
        ),
        "examples": len(examples),
        "frozen_at": str(manifest.get("frozen_at") or "") if isinstance(manifest, dict) else "",
        "environment_epoch": environment_epoch,
        "environment_epoch_sha256": _canonical_json_sha256(environment_epoch),
        "cases": cases,
        "status": "passed" if all(gates.values()) else "failed",
        "gates": gates,
        "metrics": authority_metrics,
        "precision_lower_95": _wilson_lower_bound(true_positive, labeled),
        "precision_delta_points": (
            round((float(precision) - 1.0) * 100.0, 6)
            if isinstance(precision, int | float)
            else None
        ),
        "recall_delta_points": round(
            (related_recall - float(metrics.get("recall_at_5") or 0.0)) * 100.0,
            6,
        ),
    }
    artifact = {**unsigned, "snapshot_sha256": _canonical_json_sha256(unsigned)}
    _atomic_write_text(
        path,
        json.dumps(artifact, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    )
    return artifact


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    try:
        lines = path.read_text(encoding="utf-8").split("\n")
    except OSError:
        return []
    rows: list[dict[str, Any]] = []
    for line in lines:
        try:
            parsed = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            rows.append(parsed)
    return rows


def _atomic_write_text(path: Path, payload: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            tmp = Path(handle.name)
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, path)
        try:
            directory_fd = os.open(path.parent, os.O_RDONLY)
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
        except OSError:
            pass
    finally:
        if tmp is not None:
            with contextlib.suppress(FileNotFoundError):
                tmp.unlink()


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    payload = "".join(
        json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows
    )
    _atomic_write_text(path, payload)


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def _label_candidate_payload(row: Mapping[str, Any]) -> dict[str, Any]:
    """Return the stable label claim authorized by one semantic verdict."""

    payload = {
        "query": str(row.get("query") or ""),
        "expected_pages": _str_list(row.get("expected_pages")),
        "negative_pages": _str_list(row.get("negative_pages")),
        "stale_pages": _str_list(row.get("stale_pages")),
        "split": str(row.get("split") or ""),
        "language": str(row.get("language") or ""),
        "kind": str(row.get("kind") or ""),
        "source": str(row.get("source") or ""),
        "ref": str(row.get("ref") or ""),
        "ts": str(row.get("ts") or ""),
    }
    if payload["source"] in {"recall_questions", "auto", "generated"}:
        payload["candidate_preregistration"] = {
            "candidate_sha256": str(row.get("candidate_sha256") or ""),
            "preregistered_at": str(row.get("preregistered_at") or ""),
            "source_page": str(row.get("source_page") or ""),
            "page_uid": str(row.get("page_uid") or ""),
            "content_sha256": str(row.get("content_sha256") or ""),
            "content_byte_length": row.get("content_byte_length"),
            "projection_policy_sha256": str(
                row.get("projection_policy_sha256") or ""
            ),
            "split_role": str(row.get("split_role") or ""),
        }
    return payload


label_candidate_payload = _label_candidate_payload


def _auto_candidate_preregistration_error(row: Mapping[str, Any]) -> str:
    from chronovisor.recall.recall_runtime import page_uid_for_id

    query = str(row.get("query") or "")
    pages = _str_list(row.get("expected_pages"))
    page_id = str(row.get("source_page") or "")
    path = find_page(page_id) if page_id else None
    try:
        content = path.read_bytes() if path else b""
    except OSError:
        content = b""
    identity = {
        "query": query,
        "expected_pages": pages,
        "source": str(row.get("source") or ""),
        "page_uid": str(row.get("page_uid") or ""),
        "content_sha256": str(row.get("content_sha256") or ""),
        "content_byte_length": row.get("content_byte_length"),
        "projection_policy_sha256": str(
            row.get("projection_policy_sha256") or ""
        ),
        "search_eval_split": str(row.get("split") or ""),
    }
    candidate_sha = _canonical_json_sha256(identity)
    try:
        preregistered = datetime.fromisoformat(
            str(row.get("preregistered_at") or "").replace("Z", "+00:00")
        )
        preregistered_valid = (
            preregistered.tzinfo is not None
            and preregistered.utcoffset() is not None
        )
    except ValueError:
        preregistered_valid = False
    if (
        len(pages) != 1
        or pages != [page_id]
        or not query
        or not page_id
        or row.get("candidate_sha256") != candidate_sha
        or row.get("projection_policy_sha256") != RQ_PROJECTION_POLICY_SHA256
        or row.get("split_role") != "search_eval_only_not_answer_benchmark"
        or not preregistered_valid
        or not content
        or row.get("content_sha256") != hashlib.sha256(content).hexdigest()
        or row.get("content_byte_length") != len(content)
        or row.get("page_uid") != page_uid_for_id(page_id)
    ):
        return "search label candidate preregistration is stale or invalid"
    return ""


auto_candidate_preregistration_error = _auto_candidate_preregistration_error


def _has_semantic_no_quorum_marker(value: object) -> bool:
    if not isinstance(value, Mapping):
        return False
    return (
        "semantic_hold" in value
        or value.get("last_failure_class") == LOCAL_SEMANTIC_NO_QUORUM
        or frontier_failure_class(value) == LOCAL_SEMANTIC_NO_QUORUM
    )


def _label_semantic_epoch(row: Mapping[str, Any]) -> dict[str, Any]:
    evidence = _label_candidate_payload(row)
    return {
        "artifact_schema_version": SEARCH_REVIEW_ARTIFACT_SCHEMA_VERSION,
        "prompt_policy_version": SEARCH_LABEL_PROMPT_POLICY_VERSION,
        "evidence_sha256": _canonical_json_sha256(evidence),
        "review_schema_sha256": canonical_sha256(FRONTIER_LABEL_SCHEMA),
    }


def _restore_label_semantic_hold(
    row: dict[str, Any],
    *,
    hold: Mapping[str, Any] | None,
    reviewed_at: str,
    malformed: bool = False,
) -> None:
    row["queue_status"] = "frontier_quarantined"
    row["promoted_to_golden"] = False
    row["last_failure_class"] = LOCAL_SEMANTIC_NO_QUORUM
    row["review_note"] = (
        "malformed local semantic no-quorum hold; refusing resample"
        if malformed
        else "local semantic models did not reach a safe quorum"
    )
    row["quarantined_at"] = row.get("quarantined_at") or reviewed_at
    row.pop("next_attempt_at", None)
    row.pop("decision_artifact", None)
    row.pop("frontier_review", None)
    if hold is not None:
        existing = persisted_semantic_no_quorum_hold(row, lane=SEARCH_LABEL_LANE)
        if existing is not None and existing.get("hold_sha256") != hold.get(
            "hold_sha256"
        ):
            history = [
                item
                for item in row.get("semantic_hold_history", [])
                if isinstance(item, Mapping)
            ]
            if not any(
                item.get("hold_sha256") == existing.get("hold_sha256")
                for item in history
            ):
                history.append(existing)
            # This is a durable identity ledger rather than a bounded retry
            # cache.  Retaining every distinct hold guarantees that returning
            # to any prior epoch/authority restores it before model execution.
            row["semantic_hold_history"] = history
        row["semantic_hold"] = dict(hold)


def _label_review_claim_error(
    review: object,
    evidence: object,
) -> str | None:
    """Bind an approved label action to the exact candidate buckets."""

    if not isinstance(review, Mapping) or not isinstance(evidence, Mapping):
        return "search label verdict evidence is missing"
    if review.get("decision") != "approved":
        return None
    fields = ("expected_pages", "negative_pages", "stale_pages")
    reviewed = tuple(review.get(field) for field in fields)
    candidate = tuple(evidence.get(field) for field in fields)
    if any(
        not isinstance(value, list)
        or any(not isinstance(item, str) or not item for item in value)
        or len(value) != len(set(value))
        for value in (*reviewed, *candidate)
    ):
        return "approved search label verdict arrays are invalid"
    if reviewed != candidate:
        return "approved search label verdict changed candidate buckets"
    if not any(bool(value) for value in candidate):
        return "approved search label verdict has no candidate page"
    return None


def _seal_search_review(
    *,
    kind: str,
    lane: str,
    evidence: Mapping[str, Any],
    review: Mapping[str, Any],
    authority: Mapping[str, Any],
) -> dict[str, Any]:
    evidence_payload = dict(evidence)
    return decision_authority.seal_semantic_artifact(
        {
            "schema_version": SEARCH_REVIEW_ARTIFACT_SCHEMA_VERSION,
            "kind": kind,
            "evidence": evidence_payload,
            "evidence_sha256": _canonical_json_sha256(evidence_payload),
            "review": dict(review),
        },
        authority=authority,
        lane=lane,
    )


def _search_review_artifact_error(
    artifact: object,
    *,
    kind: str,
    lane: str,
    evidence: Mapping[str, Any] | None = None,
    current_authority: object | None = None,
) -> str | None:
    if not isinstance(artifact, Mapping):
        return "semantic review artifact is missing"
    if (
        artifact.get("schema_version") != SEARCH_REVIEW_ARTIFACT_SCHEMA_VERSION
        or artifact.get("kind") != kind
    ):
        return "semantic review artifact identity is invalid"
    stored_evidence = artifact.get("evidence")
    if (
        not isinstance(stored_evidence, Mapping)
        or artifact.get("evidence_sha256")
        != _canonical_json_sha256(dict(stored_evidence))
        or (evidence is not None and dict(stored_evidence) != dict(evidence))
    ):
        return "semantic review artifact evidence is invalid"
    review = artifact.get("review")
    authority = artifact.get("authority")
    error = decision_authority.semantic_verdict_authority_error(
        review,
        authority,
        lane=lane,
    )
    if error is not None:
        return error
    if current_authority is not None:
        return decision_authority.compare_semantic_authority(
            authority,
            current_authority,
            lane=lane,
        )
    return None


def _label_review_artifact_error(
    artifact: object,
    *,
    evidence: Mapping[str, Any] | None = None,
    current_authority: object | None = None,
) -> str | None:
    error = _search_review_artifact_error(
        artifact,
        kind="search_label_verdict",
        lane=SEARCH_LABEL_LANE,
        evidence=evidence,
        current_authority=current_authority,
    )
    if error is not None:
        return error
    assert isinstance(artifact, Mapping)
    return _label_review_claim_error(
        artifact.get("review"),
        artifact.get("evidence"),
    )


label_review_artifact_error = _label_review_artifact_error


def authoritative_search_label_error(row: Mapping[str, Any]) -> str | None:
    """Rejoin a golden row to its exact approved current-authority artifact."""

    preregistration_error = _auto_candidate_preregistration_error(row)
    if preregistration_error:
        return preregistration_error
    artifact = row.get("decision_artifact")
    evidence = _label_candidate_payload(row)
    authority, authority_error = decision_authority.current_semantic_authority(
        SEARCH_LABEL_LANE
    )
    if authority_error is not None or authority is None:
        return authority_error or "search label authority unavailable"
    error = _label_review_artifact_error(
        artifact,
        evidence=evidence,
        current_authority=authority,
    )
    if error is not None:
        return error
    review = artifact.get("review") if isinstance(artifact, Mapping) else None
    if not isinstance(review, Mapping) or review.get("decision") != "approved":
        return "search label artifact is not approved"
    if _label_tuple_from_review(dict(review)) != (
        tuple(evidence["expected_pages"]),
        tuple(evidence["negative_pages"]),
        tuple(evidence["stale_pages"]),
    ):
        return "search label artifact postimage changed"
    return None


def _top_page_ids(value: Any) -> tuple[str, ...]:
    if not isinstance(value, list):
        return ()
    out: list[str] = []
    for item in value:
        if isinstance(item, dict) and isinstance(item.get("page_id"), str):
            out.append(item["page_id"])
        elif isinstance(item, str):
            out.append(item)
    return tuple(dict.fromkeys(out))


def language_bucket(text: str) -> str:
    has_cjk = any(
        ("\u3040" <= ch <= "\u30ff")
        or ("\u3400" <= ch <= "\u4dbf")
        or ("\u4e00" <= ch <= "\u9fff")
        or ("\uff66" <= ch <= "\uff9f")
        for ch in text
    )
    has_ascii_word = any(("a" <= ch.lower() <= "z") for ch in text)
    if has_cjk and has_ascii_word:
        return "mixed"
    if has_cjk:
        return "ja"
    if has_ascii_word:
        return "en"
    return "unknown"


def query_kind(text: str) -> str:
    compact = text.strip()
    if len(compact) <= 24:
        return "short"
    if "?" in compact or "？" in compact:
        return "question"
    if any(
        token in compact
        for token in ("```", "def ", "class ", "import ", "pytest", "uv run")
    ):
        return "code"
    return "statement"


def assign_split(seed: str) -> str:
    bucket = int(hashlib.sha1(seed.encode("utf-8")).hexdigest()[:8], 16) % 10
    if bucket < 2:
        return "locked-test"
    if bucket < 4:
        return "dev"
    return "train"


def build_candidates(
    *,
    feedback_file: Path = RECALL_FEEDBACK_FILE,
    log_file: Path = RECALL_LOG_FILE,
    limit: int = 100,
) -> list[SearchExample]:
    logs_by_id = {
        str(row.get("decision_id", "")): row
        for row in read_jsonl(log_file)
        if row.get("decision_id")
    }
    positive_examples: list[SearchExample] = []
    negative_examples: list[SearchExample] = []
    seen: set[tuple[str, tuple[str, ...], tuple[str, ...]]] = set()

    for feedback in active_feedback_rows(feedback_file):
        kind = str(feedback.get("kind", ""))
        if kind not in {
            "missed",
            "missed_candidate",
            "injection_used",
            "injection_ignored",
            "false-positive",
            "page_ignored",
        }:
            continue

        ref = str(feedback.get("ref", ""))
        snapshot = (
            feedback.get("snapshot")
            if isinstance(feedback.get("snapshot"), dict)
            else {}
        )
        record = logs_by_id.get(ref) or snapshot or {}
        query = str(
            feedback.get("prompt") or record.get("prompt_preview") or ""
        ).strip()
        if not query:
            continue

        raw_expected = _str_tuple(feedback.get("expected_pages"))
        raw_negative = _str_tuple(feedback.get("negative_pages"))
        raw_injected = (
            _str_tuple(feedback.get("injected_pages"))
            or _str_tuple(record.get("pages"))
            or _top_page_ids(feedback.get("top_pages"))
        )

        expected: tuple[str, ...] = ()
        negative: tuple[str, ...] = ()
        if kind in {"missed", "missed_candidate"} or kind == "injection_used":
            expected = raw_expected or raw_injected
        elif kind == "page_ignored":
            # Page-scoped feedback must never turn every injected page into a
            # negative example when only one candidate was rejected.
            negative = raw_negative
        else:
            # Prefer the explicit page-scoped field when present while
            # retaining compatibility with legacy prompt-scoped feedback.
            negative = raw_negative or raw_injected or raw_expected

        # A reviewed search label may carry both relevant and irrelevant
        # candidates. Preserve that mixed supervision for ranking evaluation.
        if raw_negative:
            negative = raw_negative

        if not expected and not negative:
            continue
        key = (query, expected, negative)
        if key in seen:
            continue
        seen.add(key)

        seed = json.dumps(
            [query, expected, negative], ensure_ascii=False, sort_keys=True
        )
        example = SearchExample(
            query=query,
            expected_pages=expected,
            negative_pages=negative,
            split=assign_split(seed),
            language=language_bucket(query),
            kind=kind,
            source=str(feedback.get("source") or "feedback"),
            ref=ref,
            ts=str(feedback.get("ts") or record.get("ts") or ""),
            reviewed=False,
        )
        if negative:
            negative_examples.append(example)
        else:
            positive_examples.append(example)

    if limit <= 0:
        return []
    if not negative_examples:
        return positive_examples[:limit]
    negative_quota = min(len(negative_examples), max(1, limit // 5))
    positive_quota = max(0, limit - negative_quota)
    return positive_examples[:positive_quota] + negative_examples[:negative_quota]


def _source_allowed(source: str, source_filter: str) -> bool:
    if source_filter == "all":
        return True
    is_auto = source in {"recall_questions", "auto", "generated"}
    if source_filter == "auto":
        return is_auto
    if source_filter == "manual":
        return not is_auto
    return True


def load_examples(
    path: Path = GOLDEN_FILE,
    *,
    limit: int = 0,
    source_filter: str = "all",
    reviewed_only: bool = True,
) -> list[SearchExample]:
    examples: list[SearchExample] = []
    for row in read_jsonl(path):
        # Active evaluation and self-tune must never consume a locally
        # generated label. Candidate rows live in the label queue until a
        # frontier reviewer promotes them with reviewed=true.
        if reviewed_only and row.get("reviewed") is not True:
            continue
        query = str(row.get("query", "")).strip()
        if not query:
            continue
        source = str(row.get("source") or "manual")
        if not _source_allowed(source, source_filter):
            continue
        if source in {
            "recall_questions",
            "auto",
            "generated",
        } and authoritative_search_label_error(row) is not None:
            continue
        expected = _str_tuple(row.get("expected_pages"))
        negative = _str_tuple(row.get("negative_pages"))
        stale = _str_tuple(row.get("stale_pages"))
        if not expected and not negative and not stale:
            continue
        examples.append(
            SearchExample(
                query=query,
                expected_pages=expected,
                negative_pages=negative,
                stale_pages=stale,
                split=str(row.get("split") or assign_split(query)),
                language=str(row.get("language") or language_bucket(query)),
                kind=str(row.get("kind") or query_kind(query)),
                source=source,
                ref=str(row.get("ref") or ""),
                ts=str(row.get("ts") or ""),
                reviewed=bool(row.get("reviewed", False)),
            )
        )
        if limit > 0 and len(examples) >= limit:
            break
    return examples


def examples_to_rows(examples: list[SearchExample]) -> list[dict[str, Any]]:
    return [
        {
            **asdict(example),
            "expected_pages": list(example.expected_pages),
            "negative_pages": list(example.negative_pages),
            "stale_pages": list(example.stale_pages),
        }
        for example in examples
    ]


def _pipeline_dependencies() -> PipelineDependencies:
    return PipelineDependencies(
        get_bm25=get_bm25,
        context_seed_results=context_seed_results,
        semantic_search=semantic_search,
        semantic_verify=semantic_verify,
        graph_expand_results=graph_expand_results,
        usage_prior_results=usage_prior_results,
        fuse_results=fuse_results,
        apply_filters=apply_filters,
        apply_sort=lambda results, sort_by="relevance": results,
        load_negative_feedback_config=load_negative_feedback_config,
        penalties_for_query=penalties_for_query,
        apply_penalties=apply_penalties,
    )


def _variant_pipeline_config(
    variant: str, *, top_n: int
) -> tuple[PipelineConfig, bool]:
    weights = dict(load_active_fusion_weights())
    search_embedding = load_search_embedding_config()
    if (
        search_embedding.enabled
        and search_embedding.backend == "nemotron_service"
        and search_embedding.rollout_mode in {"canary", "on"}
    ):
        weights.update(
            {
                "semantic": search_embedding.fusion_weight,
                "semantic_min_top_score": search_embedding.min_top_score,
                "semantic_min_margin": search_embedding.min_margin,
                "semantic_low_confidence_weight": (
                    search_embedding.low_confidence_weight
                ),
            }
        )
    if variant == "bm25":
        return (
            PipelineConfig(
                top_n=top_n,
                semantic=False,
                fusion_weights=weights,
                result_strategy="bm25",
                graph_strategy="disabled",
                usage_strategy="disabled",
            ),
            False,
        )
    if variant == "semantic":
        return (
            PipelineConfig(
                top_n=top_n,
                semantic=True,
                fusion_weights=weights,
                result_strategy="semantic",
                graph_strategy="disabled",
                usage_strategy="disabled",
            ),
            False,
        )
    if variant == "hybrid-plain-rrf":
        return (
            PipelineConfig(
                top_n=top_n,
                semantic=True,
                fusion_weights=weights,
                result_strategy="plain_rrf",
                graph_strategy="disabled",
                usage_strategy="disabled",
                plain_rrf_weights={"bm25": 1.0, "semantic": 1.0},
            ),
            False,
        )
    if variant == "hybrid-graph":
        return (
            PipelineConfig(
                top_n=top_n,
                semantic=True,
                fusion_weights={**weights, "graph": 0.5, "usage_prior": 0.0},
                result_strategy="weighted_fusion",
                graph_strategy="fixed",
                graph_decay=0.5,
                usage_strategy="disabled",
            ),
            False,
        )
    if variant == "hybrid-usage":
        return (
            PipelineConfig(
                top_n=top_n,
                semantic=True,
                fusion_weights={**weights, "graph": 0.0, "usage_prior": 0.2},
                result_strategy="weighted_fusion",
                graph_strategy="disabled",
                usage_strategy="always",
                usage_include_graph=False,
            ),
            False,
        )
    if variant == "hybrid-current":
        return (
            production_pipeline_config(top_n=top_n, fusion_weights=weights),
            False,
        )
    if variant == "hybrid-rerank":
        return (
            replace(
                production_pipeline_config(top_n=top_n, fusion_weights=weights),
                apply_negative_feedback=False,
                filter_results=False,
                sort_results=False,
                truncate_results=False,
            ),
            True,
        )
    raise ValueError(f"unknown search eval variant: {variant}")


def _stage_page_ids(pages: list[Any], *, limit: int) -> list[str]:
    return [
        page.page_id
        for page in pages[: max(0, limit)]
        if isinstance(getattr(page, "page_id", None), str)
    ]


def _candidate_union(
    pipeline_result: Any,
    *,
    limit: int,
) -> list[str]:
    ordered: dict[str, None] = {}
    for pages in (
        pipeline_result.anchor_results,
        pipeline_result.bm25_results,
        pipeline_result.semantic_results,
        pipeline_result.graph_results,
        pipeline_result.context_results,
        pipeline_result.usage_results,
    ):
        for page_id in _stage_page_ids(pages, limit=limit):
            ordered.setdefault(page_id, None)
    return list(ordered)[: max(0, limit)]


def _eval_rerank_results(
    query: str,
    candidates: list[Any],
    *,
    config: Any,
) -> Any:
    """Use the production resident service, with local inference as fallback."""

    if config.service.enabled and config.service.mode in {"shadow", "canary", "on"}:
        try:
            from chronovisor.core import reranker_client

            return reranker_client.rerank(
                query,
                candidates,
                config=config,
                timeout_ms=config.service.timeout_ms,
            )
        except Exception:
            pass
    return rerank_results(query, candidates, config=config)


def run_variant(
    query: str,
    variant: str,
    *,
    top_n: int = 20,
    typed_retrieval_mode: str | None = None,
    calibrated_judge: bool = False,
) -> dict[str, Any]:
    started = time.perf_counter()
    config, needs_rerank = _variant_pipeline_config(variant, top_n=top_n)
    deps = _pipeline_dependencies()
    from chronovisor.knowledge_graph.retrieval import retrieval_mode_override

    typed_mode = typed_retrieval_mode or "shadow"
    with retrieval_mode_override(typed_mode), graph_query_context(query):
        pipeline_result = run_search_pipeline(query, config=config, deps=deps)
        typed_search_trace = last_search_trace()
    results = pipeline_result.results
    fused_pages = _stage_page_ids(apply_filters(results), limit=top_n)
    reranker_meta: dict[str, Any] = {"status": "not_requested"}
    negative_meta = pipeline_result.negative_feedback
    if needs_rerank:
        rerank_stage = apply_rerank_stage(
            query,
            apply_filters(results),
            reranker_config=load_reranker_config(),
            rerank_results=_eval_rerank_results,
        )
        results = rerank_stage.results
        reranker_meta = rerank_stage.metadata
        results, negative_meta = apply_negative_feedback_stage(
            query, results, deps=deps
        )

    elapsed_ms = int(round((time.perf_counter() - started) * 1000))
    out = apply_filters(results)[:top_n]
    result_pages = [page.page_id for page in out]
    from chronovisor.recall.recall_processor import select_certified_candidates
    from chronovisor.recall.recall_runtime import load_policy

    policy = load_policy()
    selections, processor = select_certified_candidates(
        query,
        out,
        reranker_metadata=reranker_meta,
        max_candidates=policy.processor_max_candidates,
        max_pointer_cards=policy.processor_max_pointer_cards,
        max_rich_evidence=policy.processor_max_rich_evidence,
        injection_token_budget=policy.processor_injection_token_budget,
        certificate_required=policy.processor_certificate_required,
        ledger=False,
        judge_policy=policy if calibrated_judge else None,
        judge_timeout_ms=(policy.processor_judge_timeout_ms if calibrated_judge else 0),
    )
    page_gate = [
        str(row.get("page_id"))
        for row in processor.get("certificates", [])
        if isinstance(row, dict) and row.get("outcome") == "pass" and row.get("page_id")
    ]
    committed = [str(selection.candidate.page_id) for selection in selections]
    processor["selected"] = [
        {
            "page_id": str(selection.candidate.page_id),
            "evidence_kind": selection.evidence_kind,
            "certificate_id": selection.certificate.certificate_id,
        }
        for selection in selections
    ]
    return {
        "variant": variant,
        "results": out,
        "latency_ms": elapsed_ms,
        "stages": {
            "candidate_union": _candidate_union(pipeline_result, limit=top_n),
            "fused": fused_pages,
            "reranked": result_pages if needs_rerank else fused_pages,
            "page_gate": page_gate,
            "committed": committed,
            "host_used": None,
            "observed": {
                "candidate_union": True,
                "fused": True,
                "reranked": True,
                "page_gate": True,
                "committed": True,
                "host_used": False,
            },
        },
        "processor": processor,
        "search_trace": typed_search_trace,
        "channels": {
            "anchor": [page.page_id for page in pipeline_result.anchor_results[:top_n]],
            "bm25": [page.page_id for page in pipeline_result.bm25_results[:top_n]],
            "semantic": [
                page.page_id for page in pipeline_result.semantic_results[:top_n]
            ],
            "graph": [page.page_id for page in pipeline_result.graph_results[:top_n]],
            "context": [
                page.page_id for page in pipeline_result.context_results[:top_n]
            ],
            "usage_prior": [
                page.page_id for page in pipeline_result.usage_results[:top_n]
            ],
            "reranker": reranker_meta,
            "negative_feedback": negative_meta,
        },
    }


def _dcg(ranks: list[int], *, k: int) -> float:
    return sum(1.0 / math.log2(rank + 1) for rank in ranks if rank <= k)


def _ideal_dcg(relevant_count: int, *, k: int) -> float:
    return sum(
        1.0 / math.log2(rank + 1) for rank in range(1, min(relevant_count, k) + 1)
    )


def percentile(values: list[int], pct: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    idx = int(round((len(ordered) - 1) * pct))
    return float(ordered[max(0, min(idx, len(ordered) - 1))])


def _metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    positives = [row for row in rows if row["expected_pages"]]
    negative_labeled = [row for row in rows if row["negative_pages"]]
    stale_labeled = [row for row in rows if row["stale_pages"]]
    latencies = [int(row["latency_ms"]) for row in rows]

    def recall_at(k: int) -> float:
        if not positives:
            return 0.0
        return sum(
            bool(set(row["expected_pages"]) & set(row["result_pages"][:k]))
            for row in positives
        ) / len(positives)

    reciprocal_ranks: list[float] = []
    ndcgs: list[float] = []
    for row in positives:
        expected = set(row["expected_pages"])
        ranks = [
            idx
            for idx, page_id in enumerate(row["result_pages"], start=1)
            if page_id in expected
        ]
        reciprocal_ranks.append((1.0 / ranks[0]) if ranks and ranks[0] <= 10 else 0.0)
        ideal = _ideal_dcg(len(expected), k=10)
        ndcgs.append((_dcg(ranks, k=10) / ideal) if ideal else 0.0)

    negative_hits = 0
    for row in negative_labeled:
        if set(row["negative_pages"]) & set(row["result_pages"][:20]):
            negative_hits += 1

    stale_hits = 0
    for row in stale_labeled:
        if set(row["stale_pages"]) & set(row["result_pages"][:20]):
            stale_hits += 1

    processor_rows = [
        row
        for row in rows
        if isinstance(row.get("stages"), dict)
        and isinstance(row["stages"].get("committed"), list)
    ]
    committed_pages = sum(len(row["stages"]["committed"]) for row in processor_rows)
    committed_expected = sum(
        len(set(row.get("expected_pages", [])) & set(row["stages"]["committed"]))
        for row in processor_rows
    )
    committed_negative = sum(
        len(set(row.get("bad_pages", [])) & set(row["stages"]["committed"]))
        for row in processor_rows
    )
    processor_positives = [row for row in processor_rows if row["expected_pages"]]
    processor_hits = sum(
        bool(set(row["expected_pages"]) & set(row["stages"]["committed"]))
        for row in processor_positives
    )
    host_rows = [
        row
        for row in processor_rows
        if bool(row["stages"].get("observed", {}).get("host_used"))
        and isinstance(row["stages"].get("host_used"), list)
    ]
    evidence_quality: dict[str, dict[str, int | float | None]] = {}
    for evidence_kind in ("rich", "pointer"):
        selected_ids: list[tuple[dict[str, Any], str]] = []
        for row in processor_rows:
            processor = row.get("processor")
            selected = processor.get("selected") if isinstance(processor, dict) else []
            if not isinstance(selected, list):
                continue
            selected_ids.extend(
                (row, str(item.get("page_id")))
                for item in selected
                if isinstance(item, dict)
                and item.get("evidence_kind") == evidence_kind
                and item.get("page_id")
            )
        true_positive = sum(
            page_id in set(row.get("expected_pages", []))
            for row, page_id in selected_ids
        )
        false_positive = sum(
            page_id in set(row.get("bad_pages", [])) for row, page_id in selected_ids
        )
        labeled = true_positive + false_positive
        evidence_quality[evidence_kind] = {
            "selected": len(selected_ids),
            "labeled": labeled,
            "true_positive": true_positive,
            "false_positive": false_positive,
            "precision": true_positive / labeled if labeled else None,
        }
    return {
        "examples": len(rows),
        "positives": len(positives),
        "negative_label_examples": len(negative_labeled),
        "stale_label_examples": len(stale_labeled),
        "recall_at_5": recall_at(5),
        "recall_at_20": recall_at(20),
        "mrr_at_10": statistics.mean(reciprocal_ranks) if reciprocal_ranks else 0.0,
        "ndcg_at_10": statistics.mean(ndcgs) if ndcgs else 0.0,
        "negative_hit_rate_at_20": (negative_hits / len(negative_labeled))
        if negative_labeled
        else 0.0,
        "stale_hit_rate_at_20": (stale_hits / len(stale_labeled))
        if stale_labeled
        else 0.0,
        "processor": {
            "evaluated_examples": len(processor_rows),
            "selected_pages": committed_pages,
            "labeled_selected_pages": committed_expected + committed_negative,
            "true_positive_pages": committed_expected,
            "false_positive_pages": committed_negative,
            "precision": (
                committed_expected / (committed_expected + committed_negative)
                if committed_expected + committed_negative
                else None
            ),
            "related_recall": (
                processor_hits / len(processor_positives)
                if processor_positives
                else 0.0
            ),
            "abstention_rate": (
                sum(not row["stages"]["committed"] for row in processor_rows)
                / len(processor_rows)
                if processor_rows
                else 0.0
            ),
            "host_used_observations": len(host_rows),
            "evidence_kind": evidence_quality,
        },
        "latency_ms": {
            "p50": float(statistics.median(latencies)) if latencies else 0.0,
            "p95": percentile(latencies, 0.95),
            "max": float(max(latencies)) if latencies else 0.0,
        },
    }


def _bucket_metrics(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    buckets: dict[str, list[dict[str, Any]]] = {"all": rows}
    for row in rows:
        for key in (
            f"split:{row['split']}",
            f"language:{row['language']}",
            f"kind:{row['kind']}",
        ):
            buckets.setdefault(key, []).append(row)
    return {key: _metrics(value) for key, value in sorted(buckets.items())}


def _manual94_authority_cases(
    examples: list[SearchExample], rows: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """Seal reviewed case-level evidence used by the manual-94 authority gate."""

    from chronovisor.recall.recall_runtime import page_uid_for_id

    cases: list[dict[str, Any]] = []
    for example, row in zip(examples, rows, strict=True):
        manifest_entry = _sealed_manifest_entry(example)
        bindings: list[dict[str, Any]] = []
        for rank, page_id in enumerate(row.get("result_pages", []), start=1):
            path = find_page(str(page_id))
            try:
                content_sha = hashlib.sha256(path.read_bytes()).hexdigest() if path else ""
            except OSError:
                content_sha = ""
            bindings.append(
                {
                    "page_id": str(page_id),
                    "page_uid": page_uid_for_id(str(page_id)),
                    "content_sha256": content_sha,
                    "rank": rank,
                }
            )
        processor = row.get("processor") if isinstance(row.get("processor"), dict) else {}
        selected_value = processor.get("selected") if isinstance(processor, dict) else []
        selected = selected_value if isinstance(selected_value, list) else []
        selected_rows = [dict(item) for item in selected if isinstance(item, dict)]
        committed = row.get("stages", {}).get("committed", [])
        committed_ids = [str(value) for value in committed if isinstance(value, str)]
        certificate_ids = [
            str(item.get("certificate_id") or "")
            for item in selected_rows
            if str(item.get("certificate_id") or "")
        ]
        commit_ids = [
            _canonical_json_sha256(
                {
                    "query_sha256": manifest_entry["query_sha256"],
                    "committed_page_ids": committed_ids,
                    "certificate_ids": certificate_ids,
                }
            )
        ] if committed_ids and certificate_ids else []
        case = {
            "manifest_entry_sha256": manifest_entry["entry_sha256"],
            "review_receipt_sha256": _canonical_json_sha256(
                {
                    "kind": "manual94-human-review-v1",
                    "entry_sha256": manifest_entry["entry_sha256"],
                    "ref": example.ref,
                    "source": example.source,
                    "reviewed": example.reviewed,
                }
            ),
            "query_sha256": manifest_entry["query_sha256"],
            "expected_pages": list(example.expected_pages),
            "bad_pages": list(example.bad_pages),
            "reviewed": example.reviewed,
            "ranked_page_bindings": bindings,
            "committed_page_ids": committed_ids,
            "certificate_ids": certificate_ids,
            "commit_ids": commit_ids,
            "selected_evidence": selected_rows,
            "latency_ms": row.get("latency_ms"),
        }
        case["case_sha256"] = _canonical_json_sha256(case)
        cases.append(case)
    return cases


def evaluate_examples(
    examples: list[SearchExample],
    *,
    variants: tuple[str, ...] = DEFAULT_VARIANTS,
    top_n: int = 20,
) -> dict[str, Any]:
    host_usage: dict[str, list[str]] = {}
    try:
        from chronovisor.recall.recall_runtime import RECALL_PULL_LOG_FILE

        for feedback in read_jsonl(RECALL_PULL_LOG_FILE):
            if feedback.get("type") != "used":
                continue
            decision_id = str(feedback.get("decision_id") or "")
            if not decision_id:
                continue
            pages = [
                str(page)
                for page in feedback.get("page_ids", [])
                if isinstance(page, str) and page
            ]
            host_usage[decision_id] = list(
                dict.fromkeys([*host_usage.get(decision_id, []), *pages])
            )
    except Exception:
        host_usage = {}
    by_variant: dict[str, Any] = {}
    debug_rows: list[dict[str, Any]] = []
    for variant in variants:
        rows: list[dict[str, Any]] = []
        for example in examples:
            result = run_variant(example.query, variant, top_n=top_n)
            stages = result.get("stages", {})
            if isinstance(stages, dict) and example.ref in host_usage:
                stages["host_used"] = host_usage[example.ref]
                observed = stages.get("observed")
                if isinstance(observed, dict):
                    observed["host_used"] = True
            result_pages = [page.page_id for page in result["results"]]
            row = {
                "query": example.query,
                "split": example.split,
                "language": example.language,
                "kind": example.kind,
                "source": example.source,
                "reviewed": example.reviewed,
                "expected_pages": list(example.expected_pages),
                "negative_pages": list(example.negative_pages),
                "stale_pages": list(example.stale_pages),
                "bad_pages": list(example.bad_pages),
                "result_pages": result_pages,
                "latency_ms": result["latency_ms"],
                "stages": stages,
                "processor": result.get("processor", {}),
            }
            rows.append(row)
            debug_rows.append(
                {
                    **row,
                    "variant": variant,
                    "channels": result["channels"],
                    "stages": stages,
                    "processor": result.get("processor", {}),
                }
            )
        by_variant[variant] = {
            "metrics": _metrics(rows),
            "by_bucket": _bucket_metrics(rows),
            "authority_cases": (
                _manual94_authority_cases(examples, rows)
                if len(examples) == 94 and all(example.reviewed for example in examples)
                else []
            ),
        }
    return {"variants": by_variant, "debug_rows": debug_rows}


def save_baseline(
    payload: dict[str, Any], *, baseline_dir: Path = BASELINE_DIR
) -> Path:
    baseline_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    path = baseline_dir / f"search-baseline-{stamp}.json"
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=str) + "\n",
        encoding="utf-8",
    )
    return path


def build_golden(
    *,
    feedback_file: Path = RECALL_FEEDBACK_FILE,
    log_file: Path = RECALL_LOG_FILE,
    output_file: Path = LABEL_QUEUE_FILE,
    limit: int = 100,
) -> dict[str, Any]:
    # Compatibility wrapper: the old command name may remain in scripts, but
    # local candidates can no longer overwrite the authoritative golden set.
    target = LABEL_QUEUE_FILE if output_file == GOLDEN_FILE else output_file
    queued = build_label_queue(
        feedback_file=feedback_file,
        log_file=log_file,
        output_file=target,
        limit=limit,
    )
    return {
        **queued,
        "status": (
            queued.get("status")
            if queued.get("status") != "ok"
            else "queued_for_frontier_review"
        ),
        "legacy_command": "build-golden",
        "authoritative_golden_unchanged": True,
    }


def build_label_queue(
    *,
    feedback_file: Path = RECALL_FEEDBACK_FILE,
    log_file: Path = RECALL_LOG_FILE,
    output_file: Path = LABEL_QUEUE_FILE,
    limit: int = 100,
    dry_run: bool = False,
    budget: Any | None = None,
) -> dict[str, Any]:
    try:
        output_preimage = output_file.read_bytes()
    except FileNotFoundError:
        output_preimage = None
    examples = build_candidates(
        feedback_file=feedback_file, log_file=log_file, limit=limit
    )
    existing_rows = read_jsonl(output_file)
    existing_by_key = {_golden_key(row): row for row in existing_rows}
    rows = []
    seen: set[tuple[str, tuple[str, ...], tuple[str, ...], tuple[str, ...]]] = set()
    for row in examples_to_rows(examples):
        key = _golden_key(row)
        seen.add(key)
        previous = existing_by_key.get(key, {})
        rows.append(
            {
                **row,
                "queue_status": previous.get("queue_status", "pending_frontier_review"),
                "promoted_to_golden": bool(previous.get("promoted_to_golden", False)),
                "reviewer": previous.get("reviewer", ""),
                "review_confidence": previous.get("review_confidence"),
                "review_note": previous.get("review_note", ""),
                **{
                    key_: value
                    for key_, value in previous.items()
                    if key_
                    in {
                        "frontier_attempts",
                        "frontier_review",
                        "decision_artifact",
                        "authority_recovery",
                        "decision_authority_error",
                        "last_authority_error_at",
                        "last_attempt_at",
                        "next_attempt_at",
                        "reviewed",
                        "reviewed_at",
                    }
                },
            }
        )
    # A refresh must never erase a row that fell outside the latest candidate
    # window. It is still subject to the same authority check below.
    rows.extend(row for row in existing_rows if _golden_key(row) not in seen)
    terminal_rows = [
        row
        for row in rows
        if str(row.get("queue_status") or "") in FRONTIER_TERMINAL_STATUSES
        and not (
            isinstance(row.get("authority_recovery"), Mapping)
            and row["authority_recovery"].get("kind") == ALREADY_APPLIED_RECOVERY
        )
    ]
    if terminal_rows:
        with decision_authority_lock():
            current_authority, current_error = (
                decision_authority.current_semantic_authority(SEARCH_LABEL_LANE)
            )
        for row in terminal_rows:
            artifact = row.get("decision_artifact")
            authority_error = current_error or _label_review_artifact_error(
                artifact,
                evidence=_label_candidate_payload(row),
                current_authority=current_authority,
            )
            if authority_error is None:
                continue
            row["queue_status"] = "frontier_retry"
            row["promoted_to_golden"] = False
            row["decision_authority_error"] = authority_error
            row["last_authority_error_at"] = datetime.now().isoformat(
                timespec="seconds"
            )
            row.pop("decision_artifact", None)
            row.pop("next_attempt_at", None)
    changed = rows != existing_rows
    if not dry_run and changed:
        if budget is not None:
            allowed, reason = budget.consume("mutation")
            if not allowed:
                return {
                    "status": "budget_deferred",
                    "reason": reason,
                    "output_file": str(output_file),
                    "examples": len(existing_rows),
                    "reviewed": sum(
                        1
                        for row in existing_rows
                        if str(row.get("queue_status") or "")
                        in FRONTIER_TERMINAL_STATUSES
                    ),
                    "preserved": len(existing_rows),
                    "dry_run": False,
                }
        with decision_authority_lock(), _search_label_queue_lock(output_file):
            try:
                current_output_preimage = output_file.read_bytes()
            except FileNotFoundError:
                current_output_preimage = None
            if current_output_preimage != output_preimage:
                return {
                    "status": "concurrent_update_deferred",
                    "reason": "search label queue changed before refresh",
                    "output_file": str(output_file),
                    "examples": len(existing_rows),
                    "reviewed": sum(
                        1
                        for row in existing_rows
                        if str(row.get("queue_status") or "")
                        in FRONTIER_TERMINAL_STATUSES
                    ),
                    "preserved": len(existing_rows),
                    "dry_run": False,
                }
            final_terminal = [
                row
                for row in rows
                if str(row.get("queue_status") or "") in FRONTIER_TERMINAL_STATUSES
                and not (
                    isinstance(row.get("authority_recovery"), Mapping)
                    and row["authority_recovery"].get("kind")
                    == ALREADY_APPLIED_RECOVERY
                )
            ]
            if final_terminal:
                current_authority, current_error = (
                    decision_authority.current_semantic_authority(SEARCH_LABEL_LANE)
                )
                for row in final_terminal:
                    authority_error = current_error or _label_review_artifact_error(
                        row.get("decision_artifact"),
                        evidence=_label_candidate_payload(row),
                        current_authority=current_authority,
                    )
                    if authority_error is None:
                        continue
                    row["queue_status"] = "frontier_retry"
                    row["promoted_to_golden"] = False
                    row["decision_authority_error"] = authority_error
                    row["last_authority_error_at"] = datetime.now().isoformat(
                        timespec="seconds"
                    )
                    row.pop("decision_artifact", None)
                    row.pop("next_attempt_at", None)
            write_jsonl(output_file, rows)
    return {
        "status": "ok",
        "output_file": str(output_file),
        "examples": len(rows),
        "reviewed": sum(
            1
            for row in rows
            if str(row.get("queue_status") or "") in FRONTIER_TERMINAL_STATUSES
        ),
        "preserved": len(existing_rows),
        "dry_run": dry_run,
        "changed": changed,
        "note": "Candidates are not added to search-golden.jsonl until trusted frontier review.",
    }


def _frontier_label_failure(
    summary: str, *, output: str = "", failure: dict[str, Any] | None = None
) -> dict[str, Any]:
    return {
        "decision": "needs_retry",
        "confidence": 0.0,
        "expected_pages": [],
        "negative_pages": [],
        "stale_pages": [],
        "summary": summary,
        "notes": None,
        "reviewer": "frontier",
        "frontier_failure": failure,
        "raw_output": output[-4000:] if output else "",
    }


def _extract_json_object(text: str) -> dict[str, Any] | None:
    if not text:
        return None
    decoder = json.JSONDecoder()
    for idx, char in enumerate(text):
        if char != "{":
            continue
        try:
            value, _end = decoder.raw_decode(text, idx)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            return value
    return None


def _normalize_frontier_label_result(
    raw: dict[str, Any], *, raw_output: str = ""
) -> dict[str, Any]:
    decision = raw.get("decision")
    if decision not in {"approved", "rejected", "uncertain", "needs_retry"}:
        return _frontier_label_failure(
            "frontier label JSON failed schema validation", output=raw_output
        )
    confidence = raw.get("confidence")
    if (
        isinstance(confidence, bool)
        or not isinstance(confidence, (int, float))
        or not math.isfinite(float(confidence))
        or not 0.0 <= float(confidence) <= 1.0
    ):
        return _frontier_label_failure(
            "frontier label confidence metadata failed schema validation",
            output=raw_output,
        )
    confidence_value = float(confidence)
    summary = raw.get("summary")
    label_fields = ("expected_pages", "negative_pages", "stale_pages")
    normalized_labels = {field: _str_list(raw.get(field)) for field in label_fields}
    if decision == "approved" and any(
        raw.get(field) != normalized_labels[field] for field in label_fields
    ):
        return _frontier_label_failure(
            "approved frontier label arrays failed exact schema validation",
            output=raw_output,
        )
    normalized = {
        "decision": decision,
        "confidence": confidence_value,
        **normalized_labels,
        "summary": summary
        if isinstance(summary, str) and summary.strip()
        else decision,
        "notes": raw.get("notes") if isinstance(raw.get("notes"), str) else None,
        "reviewer": str(raw.get("reviewer") or "frontier"),
    }
    raw_text = raw_output or str(raw.get("raw_output") or "")
    if raw_text and decision == "needs_retry":
        normalized["raw_output"] = raw_text[-4000:]
    for key in (
        "frontier_failure",
        "access_repair",
        "votes",
        "decision_policy",
        "local_consensus",
        "schema_valid",
        "validation_errors",
    ):
        if key in raw:
            normalized[key] = raw[key]
    if any(key in raw for key in ("frontier_failure", "human_required", "notify_user")):
        needs_human = is_human_required_result(raw)
        normalized["human_required"] = needs_human
        normalized["notify_user"] = needs_human
    return normalized


def _parse_frontier_label_output(output: str) -> dict[str, Any]:
    parsed = _extract_json_object(output)
    if parsed is None:
        return _frontier_label_failure(
            "frontier label output did not contain JSON", output=output
        )
    return _normalize_frontier_label_result(parsed, raw_output=output)


def run_frontier_label_review(
    row: dict[str, Any],
    *,
    repo_root: Path = REPO_ROOT,
    timeout: int | None = None,
) -> dict[str, Any]:
    from chronovisor.decision import routine_review

    prompt = build_frontier_label_prompt(row)
    timeout_seconds = timeout or int(
        os.environ.get("CHRONOVISOR_FRONTIER_TIMEOUT_SECONDS", "3600")
    )
    raw = routine_review.run_structured_review(
        prompt,
        FRONTIER_LABEL_SCHEMA,
        repo_root=repo_root,
        timeout=timeout_seconds,
        execute_patch=False,
        command_env="CHRONOVISOR_LABEL_REVIEW_CMD",
        decision_lane="search_label",
    )
    return _normalize_frontier_label_result(raw)


def _label_tuple_from_review(
    review: dict[str, Any],
) -> tuple[tuple[str, ...], tuple[str, ...], tuple[str, ...]]:
    return (
        tuple(_str_list(review.get("expected_pages"))),
        tuple(_str_list(review.get("negative_pages"))),
        tuple(_str_list(review.get("stale_pages"))),
    )


label_tuple_from_review = _label_tuple_from_review


def _combine_frontier_label_reviews(
    reviews: list[dict[str, Any]],
    *,
    min_confidence: float,
) -> dict[str, Any]:
    # Kept for API compatibility only. Confidence is diagnostic metadata and
    # never participates in consensus or promotion.
    del min_confidence
    if not reviews:
        return _frontier_label_failure("no frontier label reviews were attempted")
    if len(reviews) == 1:
        return reviews[0]

    if any(is_human_required_result(review) for review in reviews):
        first = next(review for review in reviews if is_human_required_result(review))
        return {
            **first,
            "summary": f"frontier label review needs human action: {first.get('summary', '')}",
        }

    retry = [review for review in reviews if review.get("decision") == "needs_retry"]
    if retry:
        return {
            **retry[0],
            "summary": f"frontier label consensus needs retry: {retry[0].get('summary', '')}",
            "votes": reviews,
        }

    approvals = [review for review in reviews if review.get("decision") == "approved"]
    label_sets = {_label_tuple_from_review(review) for review in approvals}
    if len(approvals) == len(reviews) and len(label_sets) == 1:
        agreed = approvals[0]
        return {
            **agreed,
            "reviewer": "frontier_consensus",
            "summary": f"frontier consensus approved: {agreed.get('summary', '')}",
            "votes": reviews,
        }

    if any(review.get("decision") == "rejected" for review in reviews):
        return {
            **next(
                review for review in reviews if review.get("decision") == "rejected"
            ),
            "reviewer": "frontier_consensus",
            "summary": "frontier consensus rejected or disagreed on labels",
            "votes": reviews,
        }

    return {
        "decision": "uncertain",
        "confidence": 0.0,
        "expected_pages": [],
        "negative_pages": [],
        "stale_pages": [],
        "summary": "frontier reviewers did not agree on one label action",
        "notes": None,
        "reviewer": "frontier_consensus",
        "votes": reviews,
        **(
            {"decision_policy": reviews[0]["decision_policy"]}
            if isinstance(reviews[0].get("decision_policy"), Mapping)
            else {}
        ),
        **(
            {"local_consensus": reviews[0]["local_consensus"]}
            if isinstance(reviews[0].get("local_consensus"), Mapping)
            else {}
        ),
    }


def _golden_key(
    row: dict[str, Any],
) -> tuple[str, tuple[str, ...], tuple[str, ...], tuple[str, ...]]:
    return (
        str(row.get("query") or ""),
        tuple(_str_list(row.get("expected_pages"))),
        tuple(_str_list(row.get("negative_pages"))),
        tuple(_str_list(row.get("stale_pages"))),
    )


def _golden_row_from_review(
    row: dict[str, Any],
    review: dict[str, Any],
    *,
    reviewed_at: str,
    decision_artifact: Mapping[str, Any],
) -> dict[str, Any]:
    expected = _str_list(review.get("expected_pages"))
    negative = _str_list(review.get("negative_pages"))
    stale = _str_list(review.get("stale_pages"))
    out = {
        "query": str(row.get("query") or ""),
        "expected_pages": expected,
        "negative_pages": negative,
        "stale_pages": stale,
        "split": str(row.get("split") or assign_split(str(row.get("query") or ""))),
        "language": str(
            row.get("language") or language_bucket(str(row.get("query") or ""))
        ),
        "kind": str(row.get("kind") or query_kind(str(row.get("query") or ""))),
        "source": str(row.get("source") or "frontier_label_review"),
        "ref": str(row.get("ref") or ""),
        "ts": str(row.get("ts") or ""),
        "reviewed": True,
        "reviewer": str(review.get("reviewer") or "frontier"),
        "review_confidence": float(review.get("confidence") or 0.0),
        "reviewed_at": reviewed_at,
        "review_note": str(review.get("summary") or ""),
        "decision_artifact": dict(decision_artifact),
    }
    if str(row.get("source") or "") in {"recall_questions", "auto", "generated"}:
        for field in (
            "candidate_sha256",
            "preregistered_at",
            "source_page",
            "page_uid",
            "content_sha256",
            "content_byte_length",
            "projection_policy_sha256",
            "split_role",
        ):
            out[field] = row.get(field)
    return out


def _queue_status_for_review(review: dict[str, Any], *, min_confidence: float) -> str:
    # Deprecated compatibility input; decision + exact label action determine
    # queue state.
    del min_confidence
    if is_human_required_result(review):
        return "human_required"
    decision = review.get("decision")
    if decision == "approved" and any(_label_tuple_from_review(review)):
        return "frontier_approved"
    if decision == "approved":
        return "frontier_uncertain"
    if decision == "rejected":
        return "frontier_rejected"
    if decision == "needs_retry":
        return "frontier_retry"
    return "frontier_uncertain"


def review_label_queue_with_frontier(
    *,
    queue_file: Path = LABEL_QUEUE_FILE,
    golden_file: Path = GOLDEN_FILE,
    limit: int = 100,
    min_confidence: float = 0.8,
    votes: int = 1,
    timeout: int | None = None,
    repo_root: Path = REPO_ROOT,
    reviewer: FrontierLabelReviewer | None = None,
    max_attempts: int = 3,
    backoff_base_seconds: int = 6 * 60 * 60,
    dry_run: bool = False,
    now: datetime | None = None,
    budget: Any | None = None,
) -> dict[str, Any]:
    def optional_bytes(path: Path) -> bytes | None:
        try:
            return path.read_bytes()
        except FileNotFoundError:
            return None

    queue_preimage = optional_bytes(queue_file)
    golden_preimage = optional_bytes(golden_file)
    rows = read_jsonl(queue_file)
    original_rows = json.loads(json.dumps(rows, ensure_ascii=False, default=str))
    golden_rows = read_jsonl(golden_file)
    golden_keys = {_golden_key(row) for row in golden_rows}
    current_time = now or datetime.now()
    reviewed_at = current_time.isoformat(timespec="seconds")
    attempted = 0
    promoted = 0
    status_counts: dict[str, int] = {}
    updated_rows: list[dict[str, Any]] = []
    max_votes = max(1, votes)
    attempts_cap = max(1, max_attempts)
    budget_exhausted = False
    commit_error: str | None = None
    mutation_reserved = False
    injected_reviewer = reviewer is not None or (
        getattr(run_frontier_label_review, "__module__", None) != __name__
    )
    effect_rows: dict[int, dict[str, Any]] = {}
    staged_golden: list[tuple[int, dict[str, Any], bool]] = []

    def authority_retry(
        row: dict[str, Any], error: str, *, preserve_attempts: bool = True
    ) -> dict[str, Any]:
        updated = dict(row)
        updated["queue_status"] = "frontier_retry"
        updated["promoted_to_golden"] = False
        updated["decision_authority_error"] = error
        updated["last_authority_error_at"] = reviewed_at
        updated.pop("decision_artifact", None)
        updated.pop("next_attempt_at", None)
        if not preserve_attempts:
            updated["frontier_attempts"] = 0
        return updated

    def label_artifact_error(
        artifact: object,
        *,
        current_authority: object | None = None,
        evidence: Mapping[str, Any] | None = None,
    ) -> str | None:
        return _label_review_artifact_error(
            artifact,
            evidence=evidence,
            current_authority=current_authority,
        )

    def artifact_review(artifact: object) -> dict[str, Any] | None:
        review_value = artifact.get("review") if isinstance(artifact, Mapping) else None
        return dict(review_value) if isinstance(review_value, Mapping) else None

    semantic_rows = [row for row in rows if _has_semantic_no_quorum_marker(row)]
    semantic_authority: dict[str, Any] | None = None
    semantic_authority_error: str | None = None
    if semantic_rows:
        with decision_authority_lock():
            semantic_authority, semantic_authority_error = (
                decision_authority.current_semantic_authority(
                    SEARCH_LABEL_LANE,
                    injected_reviewer=injected_reviewer,
                )
            )
    for row in semantic_rows:
        hold = persisted_semantic_no_quorum_hold(row, lane=SEARCH_LABEL_LANE)
        if hold is None:
            _restore_label_semantic_hold(
                row, hold=None, reviewed_at=reviewed_at, malformed=True
            )
            continue
        if semantic_authority_error is not None or semantic_authority is None:
            _restore_label_semantic_hold(row, hold=hold, reviewed_at=reviewed_at)
            continue
        hold_error = semantic_no_quorum_hold_error(
            hold,
            SEARCH_LABEL_LANE,
            epoch=_label_semantic_epoch(row),
            authority=semantic_authority,
        )
        if hold_error is None:
            _restore_label_semantic_hold(row, hold=hold, reviewed_at=reviewed_at)
        elif hold_error in {
            "semantic hold epoch changed",
            "semantic hold authority changed",
        }:
            historical_hold: dict[str, Any] | None = None
            history = row.get("semantic_hold_history")
            if isinstance(history, list):
                for candidate in reversed(history):
                    historical_hold = persisted_semantic_no_quorum_hold(
                        candidate,
                        lane=SEARCH_LABEL_LANE,
                        epoch=_label_semantic_epoch(row),
                        authority=semantic_authority,
                    )
                    if historical_hold is not None:
                        break
            if historical_hold is not None:
                _restore_label_semantic_hold(
                    row,
                    hold=historical_hold,
                    reviewed_at=reviewed_at,
                )
                continue
            # Keep the old hold attached while granting this exact new epoch
            # one ordinary review opportunity.  A rollback restores it before
            # another model call.
            row["queue_status"] = "frontier_retry"
            row["promoted_to_golden"] = False
            row.pop("decision_artifact", None)
        else:
            _restore_label_semantic_hold(
                row, hold=None, reviewed_at=reviewed_at, malformed=True
            )

    for row in rows:
        if _has_semantic_no_quorum_marker(row):
            continue
        review = row.get("frontier_review")
        if (
            row.get("queue_status") == "human_required"
            and isinstance(review, dict)
            and not is_human_required_result(review)
        ):
            attempts = int(row.get("frontier_attempts") or 0)
            row["queue_status"] = (
                "frontier_quarantined" if attempts >= attempts_cap else "frontier_retry"
            )
            row["human_boundary_reclassified_at"] = reviewed_at
            if row["queue_status"] == "frontier_quarantined":
                row["quarantined_at"] = reviewed_at
            row.pop("next_attempt_at", None)

    try:
        quarantine_retry_seconds = max(
            0,
            int(
                os.getenv(
                    "CHRONOVISOR_CONVERGENCE_QUARANTINE_RETRY_SECONDS",
                    str(DEFAULT_QUARANTINE_RETRY_SECONDS),
                )
            ),
        )
    except (TypeError, ValueError):
        quarantine_retry_seconds = DEFAULT_QUARANTINE_RETRY_SECONDS
    for row in rows:
        if row.get("queue_status") != "frontier_quarantined":
            continue
        if _has_semantic_no_quorum_marker(row):
            continue
        raw_quarantined_at = row.get("quarantined_at") or row.get("last_attempt_at")
        try:
            quarantined_at = (
                datetime.fromisoformat(raw_quarantined_at)
                if isinstance(raw_quarantined_at, str) and raw_quarantined_at
                else None
            )
        except ValueError:
            quarantined_at = None
        compare_now = current_time
        if quarantined_at is not None:
            if quarantined_at.tzinfo is None and compare_now.tzinfo is not None:
                compare_now = compare_now.replace(tzinfo=None)
            elif quarantined_at.tzinfo is not None and compare_now.tzinfo is None:
                quarantined_at = quarantined_at.replace(tzinfo=None)
        if (
            quarantined_at is not None
            and (compare_now - quarantined_at).total_seconds()
            < quarantine_retry_seconds
        ):
            continue
        row["queue_status"] = "frontier_retry"
        row["frontier_attempts"] = 0
        row["quarantine_reopened_at"] = reviewed_at
        row["quarantine_reopen_count"] = (
            int(row.get("quarantine_reopen_count") or 0) + 1
        )
        row.pop("next_attempt_at", None)

    stale_terminal_indices = [
        index
        for index, row in enumerate(rows)
        if str(row.get("queue_status") or "") in FRONTIER_TERMINAL_STATUSES
        and row.get("queue_status") != "frontier_approved"
        and not _has_semantic_no_quorum_marker(row)
        and not (
            isinstance(row.get("authority_recovery"), Mapping)
            and row["authority_recovery"].get("kind") == ALREADY_APPLIED_RECOVERY
        )
    ]
    if stale_terminal_indices:
        with decision_authority_lock():
            terminal_authority, terminal_authority_error = (
                decision_authority.current_semantic_authority(
                    SEARCH_LABEL_LANE,
                    injected_reviewer=injected_reviewer,
                )
            )
        for row_index in stale_terminal_indices:
            row = rows[row_index]
            artifact = row.get("decision_artifact")
            authority_error = terminal_authority_error or label_artifact_error(
                artifact,
                current_authority=terminal_authority,
                evidence=_label_candidate_payload(row),
            )
            if authority_error is not None:
                rows[row_index] = authority_retry(row, authority_error)

    # Golden-first recovery is bookkeeping only: the exact semantic postimage
    # is already durable. It may therefore acknowledge a queue row even after
    # the authority epoch changes, but it is explicitly marked and can never
    # install another label. Queue-first recovery still performs a semantic
    # effect and must revalidate current authority below.
    reviewed_golden: list[dict[str, Any]] = []
    for golden_row in golden_rows:
        if golden_row.get("reviewed") is not True:
            continue
        artifact = golden_row.get("decision_artifact")
        review_value = artifact_review(artifact)
        evidence = artifact.get("evidence") if isinstance(artifact, Mapping) else None
        if (
            review_value is None
            or review_value.get("decision") != "approved"
            or label_artifact_error(artifact) is not None
            or not isinstance(evidence, Mapping)
            or str(evidence.get("query") or "") != str(golden_row.get("query") or "")
            or _label_tuple_from_review(review_value)
            != (
                tuple(_str_list(golden_row.get("expected_pages"))),
                tuple(_str_list(golden_row.get("negative_pages"))),
                tuple(_str_list(golden_row.get("stale_pages"))),
            )
        ):
            continue
        reviewed_golden.append(golden_row)
    golden_by_key = {_golden_key(row): row for row in reviewed_golden}
    golden_by_ref = {
        (str(row.get("query") or ""), str(row.get("ref") or "")): row
        for row in reviewed_golden
        if str(row.get("ref") or "")
    }
    recovered_queue = 0
    for row in rows:
        if (
            bool(row.get("promoted_to_golden"))
            or row.get("queue_status") == "frontier_approved"
        ):
            continue
        golden_match = golden_by_key.get(_golden_key(row))
        ref = str(row.get("ref") or "")
        if golden_match is None and ref:
            golden_match = golden_by_ref.get((str(row.get("query") or ""), ref))
        if golden_match is None:
            continue
        artifact = golden_match.get("decision_artifact")
        review = artifact_review(artifact)
        evidence = artifact.get("evidence") if isinstance(artifact, Mapping) else None
        if (
            review is None
            or not isinstance(evidence, Mapping)
            or dict(evidence) != _label_candidate_payload(row)
        ):
            continue
        row.update(
            {
                "queue_status": "frontier_approved",
                "promoted_to_golden": True,
                "reviewed": True,
                "reviewed_at": str(golden_match.get("reviewed_at") or reviewed_at),
                "reviewer": review["reviewer"],
                "review_confidence": review["confidence"],
                "review_note": review["summary"],
                "frontier_review": review,
                "decision_artifact": artifact,
                "authority_recovery": {
                    "kind": ALREADY_APPLIED_RECOVERY,
                    "source": "golden_label",
                    "recovered_at": reviewed_at,
                },
            }
        )
        recovered_queue += 1

    recovered_golden = 0
    for row_index, row in enumerate(rows):
        recovery = row.get("authority_recovery")
        if (
            isinstance(recovery, Mapping)
            and recovery.get("kind") == ALREADY_APPLIED_RECOVERY
        ):
            continue
        if (
            not bool(row.get("promoted_to_golden"))
            and row.get("queue_status") != "frontier_approved"
        ):
            continue
        artifact = row.get("decision_artifact")
        review = artifact_review(artifact) or row.get("frontier_review")
        if not isinstance(review, dict):
            rows[row_index] = authority_retry(
                row, "approved queue row has no sealed semantic verdict"
            )
            continue
        if (
            _queue_status_for_review(review, min_confidence=min_confidence)
            != "frontier_approved"
        ):
            rows[row_index] = authority_retry(
                row, "approved queue row has no approved semantic verdict"
            )
            continue
        with decision_authority_lock():
            current_authority, current_error = (
                decision_authority.current_semantic_authority(
                    SEARCH_LABEL_LANE,
                    injected_reviewer=injected_reviewer,
                )
            )
        authority_error = current_error or label_artifact_error(
            artifact,
            current_authority=current_authority,
            evidence=_label_candidate_payload(row),
        )
        if authority_error is not None:
            rows[row_index] = authority_retry(row, authority_error)
            continue
        golden_row = _golden_row_from_review(
            row,
            review,
            reviewed_at=str(row.get("reviewed_at") or reviewed_at),
            decision_artifact=artifact,
        )
        key = _golden_key(golden_row)
        if key not in golden_keys:
            golden_keys.add(key)
            staged_golden.append((row_index, golden_row, False))
            recovered_golden += 1
        row["promoted_to_golden"] = True
        row["reviewed"] = True
        row["reviewed_at"] = str(row.get("reviewed_at") or reviewed_at)
        effect_rows[row_index] = {
            "artifact": artifact,
            "evidence": _label_candidate_payload(row),
            "status": "frontier_approved",
            "promoted": False,
        }

    if (recovered_queue or recovered_golden) and not dry_run and budget is not None:
        allowed, reason = budget.consume("mutation")
        if not allowed:
            return {
                "status": "budget_deferred",
                "reason": reason,
                "queue_file": str(queue_file),
                "golden_file": str(golden_file),
                "attempted": 0,
                "promoted": 0,
                "remaining": sum(
                    1
                    for row in original_rows
                    if str(row.get("queue_status") or "") in FRONTIER_PENDING_STATUSES
                    and not bool(row.get("promoted_to_golden"))
                ),
                "dry_run": False,
                "budget_exhausted": True,
                "recovered": 0,
            }
        mutation_reserved = True

    def retry_due(row: dict[str, Any]) -> bool:
        raw = row.get("next_attempt_at")
        if not isinstance(raw, str) or not raw:
            return True
        try:
            return datetime.fromisoformat(raw) <= current_time
        except ValueError:
            return True

    for row_index, row in enumerate(rows):
        status = str(row.get("queue_status") or "")
        if (
            attempted >= limit
            or bool(row.get("promoted_to_golden"))
            or status not in FRONTIER_PENDING_STATUSES
            or not retry_due(row)
        ):
            updated_rows.append(row)
            continue

        if dry_run:
            updated_rows.append(row)
            attempted += 1
            continue

        if budget is not None:
            frontier_allowed, _frontier_reason = budget.can_consume(
                "frontier", max_votes
            )
            mutation_allowed, _mutation_reason = (
                budget.can_consume("mutation")
                if not mutation_reserved
                else (True, "ok")
            )
            if not frontier_allowed or not mutation_allowed:
                budget_exhausted = True
                updated_rows.append(row)
                continue
            budget.consume("frontier", max_votes)
            if not mutation_reserved:
                budget.consume("mutation")
                mutation_reserved = True

        with decision_authority_lock():
            authority, authority_error = decision_authority.current_semantic_authority(
                SEARCH_LABEL_LANE,
                injected_reviewer=injected_reviewer,
            )
        if authority_error is not None or authority is None:
            updated_rows.append(
                authority_retry(
                    row, authority_error or "search label authority unavailable"
                )
            )
            continue

        reviews: list[dict[str, Any]] = []
        semantic_review: dict[str, Any] | None = None
        try:
            for _idx in range(max_votes):
                review = (
                    reviewer(row)
                    if reviewer is not None
                    else run_frontier_label_review(
                        row, repo_root=repo_root, timeout=timeout
                    )
                )
                normalized_review = (
                    dict(review)
                    if is_local_semantic_no_quorum(review)
                    else (
                        _normalize_frontier_label_result(review)
                        if "decision" in review
                        else review
                    )
                )
                reviews.append(normalized_review)
                if is_local_semantic_no_quorum(normalized_review):
                    semantic_review = normalized_review
                    break
        except Exception as exc:
            reviews = [
                _frontier_label_failure(
                    f"frontier label reviewer raised {exc.__class__.__name__}: {exc}"
                )
            ]
        if semantic_review is not None:
            hold: dict[str, Any] | None = None
            hold_error: str | None = None
            try:
                with decision_authority_lock():
                    current_authority, current_error = (
                        decision_authority.current_semantic_authority(
                            SEARCH_LABEL_LANE,
                            injected_reviewer=injected_reviewer,
                        )
                    )
                    hold_error = current_error or (
                        decision_authority.compare_semantic_authority(
                            authority,
                            current_authority,
                            lane=SEARCH_LABEL_LANE,
                        )
                    )
                    if hold_error is not None:
                        raise ValueError(hold_error)
                    hold = build_semantic_no_quorum_hold(
                        SEARCH_LABEL_LANE,
                        _label_semantic_epoch(row),
                        authority,
                        semantic_review,
                    )
            except (TypeError, ValueError) as exc:
                hold_error = str(exc)
            attempted += 1
            if hold is None:
                updated_rows.append(
                    authority_retry(
                        row,
                        hold_error or "search label semantic hold authority invalid",
                    )
                )
                status_counts["frontier_retry"] = (
                    status_counts.get("frontier_retry", 0) + 1
                )
                continue
            updated = dict(row)
            _restore_label_semantic_hold(
                updated,
                hold=hold,
                reviewed_at=reviewed_at,
            )
            updated["frontier_attempts"] = int(row.get("frontier_attempts") or 0) + 1
            updated["last_attempt_at"] = reviewed_at
            updated_rows.append(updated)
            status_counts["semantic_hold"] = status_counts.get("semantic_hold", 0) + 1
            continue
        combined = _combine_frontier_label_reviews(
            reviews, min_confidence=min_confidence
        )
        with decision_authority_lock():
            current_authority, current_error = (
                decision_authority.current_semantic_authority(
                    SEARCH_LABEL_LANE,
                    injected_reviewer=injected_reviewer,
                )
            )
            authority_error = current_error or (
                decision_authority.compare_semantic_authority(
                    authority,
                    current_authority,
                    lane=SEARCH_LABEL_LANE,
                )
            )
            if authority_error is None:
                for review_value in [*reviews, combined]:
                    authority_error = (
                        decision_authority.semantic_verdict_authority_error(
                            review_value,
                            authority,
                            lane=SEARCH_LABEL_LANE,
                        )
                    )
                    if authority_error is not None:
                        break
            label_evidence = _label_candidate_payload(row)
            authority_error = authority_error or _label_review_claim_error(
                combined,
                label_evidence,
            )
            decision_artifact = (
                _seal_search_review(
                    kind="search_label_verdict",
                    lane=SEARCH_LABEL_LANE,
                    evidence=label_evidence,
                    review=combined,
                    authority=authority,
                )
                if authority_error is None
                else None
            )
        if authority_error is not None or decision_artifact is None:
            attempted += 1
            updated_rows.append(
                authority_retry(
                    row, authority_error or "search label verdict authority invalid"
                )
            )
            status_counts["frontier_retry"] = status_counts.get("frontier_retry", 0) + 1
            continue
        next_status = _queue_status_for_review(combined, min_confidence=min_confidence)
        frontier_attempts = int(row.get("frontier_attempts") or 0) + 1
        if (
            next_status in {"frontier_retry", "frontier_uncertain"}
            and frontier_attempts >= attempts_cap
        ):
            next_status = "frontier_quarantined"
        attempted += 1
        status_counts[next_status] = status_counts.get(next_status, 0) + 1

        updated = {
            **row,
            "queue_status": next_status,
            "reviewer": combined.get("reviewer") or "frontier",
            "review_confidence": float(combined.get("confidence") or 0.0),
            "review_note": combined.get("summary") or "",
            "frontier_review": combined,
            "decision_artifact": decision_artifact,
            "frontier_attempts": frontier_attempts,
            "last_attempt_at": reviewed_at,
        }
        if next_status in {"frontier_approved", "frontier_rejected"}:
            updated.pop("semantic_hold", None)
            updated.pop("semantic_hold_history", None)
            updated.pop("last_failure_class", None)

        if next_status in {"frontier_retry", "frontier_uncertain"}:
            delay = max(0, backoff_base_seconds) * (2 ** max(0, frontier_attempts - 1))
            updated["next_attempt_at"] = (
                current_time + timedelta(seconds=delay)
            ).isoformat(timespec="seconds")
        else:
            updated.pop("next_attempt_at", None)
        if next_status == "frontier_quarantined":
            updated["quarantined_at"] = reviewed_at

        if next_status == "frontier_approved":
            golden_row = _golden_row_from_review(
                row,
                combined,
                reviewed_at=reviewed_at,
                decision_artifact=decision_artifact,
            )
            key = _golden_key(golden_row)
            if key not in golden_keys:
                golden_keys.add(key)
                staged_golden.append((row_index, golden_row, True))
                promoted += 1
            updated["promoted_to_golden"] = True
            updated["reviewed"] = True
            updated["reviewed_at"] = reviewed_at
        else:
            updated["promoted_to_golden"] = False
        updated_rows.append(updated)
        effect_rows[row_index] = {
            "artifact": decision_artifact,
            "evidence": _label_candidate_payload(row),
            "status": next_status,
            "promoted": next_status == "frontier_approved",
        }

    if not dry_run:
        # Keep the final authority stable through the semantic postimage and
        # its queue acknowledgement. A review whose epoch changed after the
        # model returned is reopened and cannot mutate either durable file.
        with decision_authority_lock(), _search_label_queue_lock(queue_file):
            if (
                optional_bytes(queue_file) != queue_preimage
                or optional_bytes(golden_file) != golden_preimage
            ):
                commit_error = "search label files changed before effect"
            current_authority: dict[str, Any] | None = None
            current_error: str | None = None
            if effect_rows and commit_error is None:
                current_authority, current_error = (
                    decision_authority.current_semantic_authority(
                        SEARCH_LABEL_LANE,
                        injected_reviewer=injected_reviewer,
                    )
                )
            invalid_effects: dict[int, str] = {}
            for row_index, effect in effect_rows.items():
                error = (
                    commit_error
                    or current_error
                    or label_artifact_error(
                        effect["artifact"],
                        current_authority=current_authority,
                        evidence=effect["evidence"],
                    )
                )
                if error is not None:
                    invalid_effects[row_index] = error
            for row_index, error in invalid_effects.items():
                prior_status = str(effect_rows[row_index].get("status") or "")
                if prior_status in status_counts:
                    status_counts[prior_status] -= 1
                    if status_counts[prior_status] <= 0:
                        status_counts.pop(prior_status, None)
                if effect_rows[row_index].get("promoted"):
                    promoted = max(0, promoted - 1)
                recovered_golden -= sum(
                    1
                    for staged_index, _golden_row, new_promotion in staged_golden
                    if staged_index == row_index and not new_promotion
                )
                updated_rows[row_index] = authority_retry(
                    original_rows[row_index], error
                )
                status_counts["frontier_retry"] = (
                    status_counts.get("frontier_retry", 0) + 1
                )
            valid_staged = [
                golden_row
                for row_index, golden_row, _new_promotion in staged_golden
                if row_index not in invalid_effects
            ]
            if valid_staged and commit_error is None:
                # Golden is the semantic postimage and is committed first. A
                # crash after this point is reconciled only by the explicitly
                # marked exact-postimage recovery path above.
                write_jsonl(golden_file, [*golden_rows, *valid_staged])
            if commit_error is None and updated_rows != original_rows:
                write_jsonl(queue_file, updated_rows)
    return {
        "status": "concurrent_update_deferred" if commit_error else "ok",
        "queue_file": str(queue_file),
        "golden_file": str(golden_file),
        "attempted": attempted,
        "promoted": promoted,
        "remaining": sum(
            1
            for row in updated_rows
            if str(row.get("queue_status") or "") in FRONTIER_PENDING_STATUSES
            and not bool(row.get("promoted_to_golden"))
        ),
        "status_counts": dict(sorted(status_counts.items())),
        "min_confidence": min_confidence,
        "votes": max_votes,
        "dry_run": dry_run,
        "max_attempts": attempts_cap,
        "budget_exhausted": budget_exhausted,
        "recovered": 0 if commit_error else recovered_queue + recovered_golden,
        "reason": commit_error or "",
    }


def _count_by(examples: list[SearchExample], field: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    for example in examples:
        value = str(getattr(example, field))
        counts[value] = counts.get(value, 0) + 1
    return dict(sorted(counts.items()))


def _failure_index_rows(debug_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in debug_rows:
        expected = [
            page for page in row.get("expected_pages", []) if isinstance(page, str)
        ]
        if not expected:
            continue
        result_pages = [
            page for page in row.get("result_pages", []) if isinstance(page, str)
        ]
        expected_set = set(expected)
        channels = row.get("channels") if isinstance(row.get("channels"), dict) else {}
        channel_candidates: dict[str, list[str]] = {}
        channel_hit = False
        for name, values in channels.items():
            if not isinstance(values, list):
                continue
            pages = [page for page in values[:20] if isinstance(page, str)]
            channel_candidates[str(name)] = pages
            if set(expected) & set(pages):
                channel_hit = True

        failed_stage = "fusion" if channel_hit else "retrieval"
        reason_code = "fusion_missed" if channel_hit else "retrieval_missed"
        fix_kind = "fusion" if channel_hit else "data_or_rewrite"
        stages = row.get("stages") if isinstance(row.get("stages"), dict) else {}
        observed_stages = (
            stages.get("observed") if isinstance(stages.get("observed"), dict) else {}
        )
        ordered_stages = (
            (
                "candidate_union",
                "retrieval",
                "candidate_union_missed",
                "data_or_rewrite",
            ),
            ("fused", "fusion", "fusion_missed", "fusion"),
            ("reranked", "reranker", "reranker_missed", "reranker"),
            ("page_gate", "page_gate", "page_gate_rejected", "selector"),
            ("committed", "commit", "commit_missed", "selector"),
            ("host_used", "host_used", "host_did_not_use", "host_or_card"),
        )
        trace_resolved = False
        observed_trace = False
        if stages:
            for stage_key, stage_name, stage_reason, stage_fix in ordered_stages:
                if observed_stages.get(stage_key) is False:
                    continue
                values = stages.get(stage_key)
                if values is None:
                    break
                if not isinstance(values, list):
                    break
                observed_trace = True
                stage_pages = [page for page in values if isinstance(page, str)]
                stage_limit = 20 if stage_key == "reranked" else len(stage_pages)
                if not (expected_set & set(stage_pages[:stage_limit])):
                    if stage_key == "reranked" and expected_set & set(stage_pages):
                        failed_stage = "rank_cutoff"
                        reason_code = "below_evaluation_cutoff"
                        fix_kind = "ranking"
                    else:
                        failed_stage = stage_name
                        reason_code = stage_reason
                        fix_kind = stage_fix
                    trace_resolved = True
                    break
            if (
                not trace_resolved
                and isinstance(stages.get("reranked"), list)
                and not (expected_set & set(result_pages[:20]))
                and expected_set & set(stages["reranked"])
            ):
                failed_stage = "rank_cutoff"
                reason_code = "below_evaluation_cutoff"
                fix_kind = "ranking"
                trace_resolved = True
        if not trace_resolved and (
            observed_trace or expected_set & set(result_pages[:20])
        ):
            continue
        rows.append(
            {
                "ts": datetime.now().isoformat(timespec="seconds"),
                "variant": row.get("variant", ""),
                "query": row.get("query", ""),
                "split": row.get("split", ""),
                "language": row.get("language", ""),
                "kind": row.get("kind", ""),
                "expected_pages": expected,
                "result_pages": result_pages[:20],
                "channel_candidates": channel_candidates,
                "stages": stages,
                "observed_stages": observed_stages,
                "failed_stage": failed_stage,
                "reason_code": reason_code,
                "fix_kind": fix_kind,
            }
        )
    return rows


def write_failure_index(
    debug_rows: list[dict[str, Any]], path: Path = FAILURE_INDEX_FILE
) -> dict[str, Any]:
    rows = _failure_index_rows(debug_rows)
    write_jsonl(path, rows)
    return {"path": str(path), "failures": len(rows)}


def run_weighted_hybrid(
    query: str, weights: dict[str, float], *, top_n: int = 20
) -> dict[str, Any]:
    started = time.perf_counter()
    pipeline_result = run_search_pipeline(
        query,
        config=production_pipeline_config(
            top_n=top_n,
            semantic=True,
            fusion_weights=dict(weights),
        ),
        deps=_pipeline_dependencies(),
    )
    elapsed_ms = int(round((time.perf_counter() - started) * 1000))
    return {"results": pipeline_result.results, "latency_ms": elapsed_ms}


def _rows_for_weight_eval(
    examples: list[SearchExample], weights: dict[str, float]
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for example in examples:
        result = run_weighted_hybrid(example.query, weights, top_n=20)
        rows.append(
            {
                "query": example.query,
                "split": example.split,
                "language": example.language,
                "kind": example.kind,
                "source": example.source,
                "reviewed": example.reviewed,
                "expected_pages": list(example.expected_pages),
                "negative_pages": list(example.negative_pages),
                "stale_pages": list(example.stale_pages),
                "bad_pages": list(example.bad_pages),
                "result_pages": [page.page_id for page in result["results"]],
                "latency_ms": result["latency_ms"],
            }
        )
    return rows


def _self_tune_review_evidence(record: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "baseline": record.get("baseline", {}),
        "best": record.get("best", {}),
        "guardrails": record.get("guardrails", {}),
        "previous_sha256": record.get("previous_policy_sha256"),
        "previous_summary": record.get("previous_policy_summary", {}),
    }


def _self_tune_semantic_epoch(evidence: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "artifact_schema_version": SEARCH_REVIEW_ARTIFACT_SCHEMA_VERSION,
        "evidence_sha256": _canonical_json_sha256(dict(evidence)),
    }


def _persisted_self_tune_hold(
    history: list[dict[str, Any]],
    *,
    epoch: Mapping[str, Any],
    authority: Mapping[str, Any],
) -> tuple[str, dict[str, Any] | None]:
    """Find an exact old epoch in append-only history, including A-B-A."""

    saw_marker = False
    for row in reversed(history):
        if not _has_semantic_no_quorum_marker(row):
            continue
        saw_marker = True
        hold = persisted_semantic_no_quorum_hold(row, lane=SEARCH_SELF_TUNE_LANE)
        if hold is None:
            return "malformed", None
        error = semantic_no_quorum_hold_error(
            hold,
            SEARCH_SELF_TUNE_LANE,
            epoch=epoch,
            authority=authority,
        )
        if error is None:
            return "same", hold
        if error not in {
            "semantic hold epoch changed",
            "semantic hold authority changed",
        }:
            return "malformed", None
    return ("changed", None) if saw_marker else ("none", None)


def _self_tune_artifact_error(
    artifact: object,
    *,
    evidence: Mapping[str, Any] | None = None,
    current_authority: object | None = None,
) -> str | None:
    return _search_review_artifact_error(
        artifact,
        kind="search_self_tune_verdict",
        lane=SEARCH_SELF_TUNE_LANE,
        evidence=evidence,
        current_authority=current_authority,
    )


def _self_tune_policy_id(policy: Mapping[str, Any]) -> str:
    canonical = dict(policy)
    canonical.pop("policy_id", None)
    return _canonical_json_sha256(canonical)


def _self_tune_previous_summary(previous: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "version": previous.get("version"),
        "source": previous.get("source"),
        "policy_id": previous.get("policy_id"),
        "weights": previous.get("weights"),
        "holdout": previous.get("holdout"),
    }


def _self_tune_policy_error(
    policy: object,
    *,
    current_authority: object | None = None,
) -> str | None:
    if not isinstance(policy, Mapping):
        return "search self-tune policy is missing"
    artifact = policy.get("decision_artifact")
    evidence = artifact.get("evidence") if isinstance(artifact, Mapping) else None
    review = artifact.get("review") if isinstance(artifact, Mapping) else None
    previous = policy.get("previous")
    weights = policy.get("weights")
    holdout = policy.get("holdout")
    if (
        set(policy)
        != {
            "version",
            "created_at",
            "source",
            "weights",
            "holdout",
            "previous",
            "decision_artifact",
            "policy_id",
        }
        or policy.get("version") != 1
        or policy.get("source") != "search_eval.self_tune"
        or not isinstance(policy.get("created_at"), str)
        or not isinstance(previous, Mapping)
        or not isinstance(weights, Mapping)
        or not isinstance(holdout, Mapping)
        or not isinstance(evidence, Mapping)
        or not isinstance(review, Mapping)
        or review.get("decision") != "approved"
    ):
        return "search self-tune policy shape is invalid"
    artifact_error = _self_tune_artifact_error(
        artifact,
        evidence=evidence,
        current_authority=current_authority,
    )
    if artifact_error is not None:
        return artifact_error
    best = evidence.get("best")
    if (
        not isinstance(best, Mapping)
        or best.get("weights") != weights
        or best.get("locked-test") != holdout
        or evidence.get("previous_sha256") != _canonical_json_sha256(previous)
        or evidence.get("previous_summary") != _self_tune_previous_summary(previous)
    ):
        return "search self-tune policy is not linked to reviewed evidence"
    policy_id = policy.get("policy_id")
    if not isinstance(policy_id, str) or policy_id != _self_tune_policy_id(policy):
        return "search self-tune policy id is invalid"
    return None


def _recover_applied_self_tune_receipt(
    *,
    policy_file: Path,
    history_file: Path,
) -> dict[str, Any] | None:
    """Acknowledge only an exact policy postimage left by a prior crash."""

    with decision_authority_lock():
        # Read both sides only after acquiring the same lease used by policy
        # mutation and rollback. This prevents a stale pre-lock snapshot from
        # creating a duplicate or misattributed receipt.
        try:
            policy_preimage = policy_file.read_bytes()
            policy = json.loads(policy_preimage)
        except (OSError, json.JSONDecodeError, UnicodeDecodeError):
            return None
        if _self_tune_policy_error(policy) is not None:
            return None
        assert isinstance(policy, dict)
        artifact = policy["decision_artifact"]
        policy_id = policy["policy_id"]
        try:
            history_preimage = history_file.read_bytes()
        except FileNotFoundError:
            history_preimage = None
        except OSError:
            return None
        prior = read_jsonl(history_file)
        lifecycle = [
            row
            for row in prior
            if row.get("status") in {"applied", "applied_recovered", "rolled_back"}
        ]
        if any(
            row.get("policy_id") == policy_id
            and row.get("status") in {"applied", "applied_recovered"}
            for row in lifecycle
        ):
            return None
        latest = lifecycle[-1] if lifecycle else None
        if isinstance(latest, Mapping):
            if (
                latest.get("status") == "rolled_back"
                and latest.get("restored_policy_id") == policy_id
            ):
                return None
            latest_ts = str(latest.get("ts") or "")
            created_at = str(policy.get("created_at") or "")
            if (
                latest.get("policy_id") != policy_id
                and latest_ts
                and created_at
                and latest_ts >= created_at
            ):
                return None
        recovered = {
            "ts": datetime.now().isoformat(timespec="seconds"),
            "status": "applied_recovered",
            "applied": True,
            "policy_id": policy_id,
            "policy": policy,
            "decision_artifact": artifact,
            "authority_recovery": {
                "kind": ALREADY_APPLIED_RECOVERY,
                "source": "active_search_policy",
            },
        }
        # This is audit reconciliation only; the exact policy bytes are already
        # installed and are never rewritten on this path.
        try:
            if policy_file.read_bytes() != policy_preimage:
                return None
        except OSError:
            return None
        try:
            current_history_preimage = history_file.read_bytes()
        except FileNotFoundError:
            current_history_preimage = None
        except OSError:
            return None
        if current_history_preimage != history_preimage:
            return None
        append_jsonl(history_file, recovered)
        return recovered


def _self_tune_recovery_paths_aligned(history_file: Path, policy_file: Path) -> bool:
    return (history_file == SELF_TUNE_HISTORY_FILE) == (
        policy_file == ACTIVE_SEARCH_POLICY_FILE
    )


def self_tune(
    *,
    golden_file: Path = GOLDEN_FILE,
    history_file: Path = SELF_TUNE_HISTORY_FILE,
    policy_file: Path = ACTIVE_SEARCH_POLICY_FILE,
    apply: bool = False,
    dry_run: bool = False,
    frontier_mode: str = "off",
    frontier_reviewer: Callable[[dict[str, Any]], dict[str, Any]] | None = None,
    budget: Any | None = None,
    max_examples: int = 200,
    max_elapsed_seconds: float = 10 * 60,
) -> dict[str, Any]:
    if not dry_run and _self_tune_recovery_paths_aligned(history_file, policy_file):
        recovered = _recover_applied_self_tune_receipt(
            policy_file=policy_file,
            history_file=history_file,
        )
        if recovered is not None:
            return recovered
    examples = load_examples(golden_file)
    dev = [example for example in examples if example.split == "dev"]
    locked = [example for example in examples if example.split == "locked-test"]
    example_cap = max(2, int(max_examples))
    if len(dev) + len(locked) > example_cap:
        locked_quota = min(len(locked), max(1, example_cap // 5))
        dev_quota = min(len(dev), example_cap - locked_quota)
        if dev_quota < example_cap - locked_quota:
            locked_quota = min(len(locked), example_cap - dev_quota)
        dev = dev[-dev_quota:] if dev_quota else []
        locked = locked[-locked_quota:] if locked_quota else []
    if not dev or not locked:
        record = {
            "ts": datetime.now().isoformat(timespec="seconds"),
            "status": "blocked",
            "applied": False,
            "reason": "independent dev and locked-test examples are required",
            "dataset": {"dev": len(dev), "locked-test": len(locked)},
        }
        if not dry_run:
            append_jsonl(history_file, record)
        return record
    deadline = time.monotonic() + max(0.0, float(max_elapsed_seconds))

    def evaluate_bounded(
        items: list[SearchExample], weights: dict[str, float]
    ) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        # A ten-example batch can take several minutes and prevented the
        # deadline from being observed until the whole batch returned. Keep
        # the expensive unit to one query so the overrun is bounded by one
        # search rather than an entire evaluation shard.
        for item in items:
            if time.monotonic() >= deadline:
                raise TimeoutError("search self-tune runtime budget exhausted")
            rows.extend(_rows_for_weight_eval([item], weights))
        return rows

    baseline_weights = load_active_fusion_weights(policy_file)
    try:
        baseline_dev = _metrics(evaluate_bounded(dev, baseline_weights))
        baseline_locked = _metrics(evaluate_bounded(locked, baseline_weights))

        candidates = []
        for semantic_weight in (0.4, 0.5, 0.6, 0.7, 0.8):
            weights = {**baseline_weights, "semantic": semantic_weight}
            dev_metrics = _metrics(evaluate_bounded(dev, weights))
            candidates.append({"weights": weights, "dev": dev_metrics})
    except TimeoutError as exc:
        return {
            "ts": datetime.now().isoformat(timespec="seconds"),
            "status": "budget_deferred",
            "applied": False,
            "reason": str(exc),
            "dataset": {
                "dev": len(dev),
                "locked-test": len(locked),
                "max_examples": example_cap,
            },
        }
    best = max(
        candidates,
        key=lambda item: (
            item["dev"]["mrr_at_10"],
            item["dev"]["recall_at_5"],
            item["dev"]["ndcg_at_10"],
        ),
    )
    try:
        locked_metrics = _metrics(evaluate_bounded(locked, best["weights"]))
    except TimeoutError as exc:
        return {
            "ts": datetime.now().isoformat(timespec="seconds"),
            "status": "budget_deferred",
            "applied": False,
            "reason": str(exc),
            "dataset": {
                "dev": len(dev),
                "locked-test": len(locked),
                "max_examples": example_cap,
            },
        }
    locked_ok = (
        locked_metrics["recall_at_5"] >= baseline_locked["recall_at_5"]
        and locked_metrics["mrr_at_10"] >= baseline_locked["mrr_at_10"]
        and locked_metrics["stale_hit_rate_at_20"]
        <= baseline_locked["stale_hit_rate_at_20"]
        and locked_metrics["negative_hit_rate_at_20"]
        <= baseline_locked["negative_hit_rate_at_20"]
    )
    dev_improved = best["dev"]["mrr_at_10"] > baseline_dev["mrr_at_10"]
    status = "shadow_pass" if dev_improved and locked_ok else "blocked"
    record = {
        "ts": datetime.now().isoformat(timespec="seconds"),
        "status": status,
        "applied": False,
        "reason": ""
        if status == "shadow_pass"
        else "dev improvement or locked-test guard failed",
        "baseline": {"dev": baseline_dev, "locked-test": baseline_locked},
        "best": {
            "weights": best["weights"],
            "dev": best["dev"],
            "locked-test": locked_metrics,
        },
        "guardrails": {
            "dev_improved": dev_improved,
            "locked_non_degrading": locked_ok,
            "apply_policy": "validated_auto" if apply else "shadow_only",
        },
    }
    semantic_artifact: dict[str, Any] | None = None
    pending_policy: dict[str, Any] | None = None
    pending_policy_preimage: bytes | None = None
    previous_policy: dict[str, Any] = {}
    semantic_evidence: dict[str, Any] | None = None
    injected_reviewer = frontier_reviewer is not None or (
        getattr(review_search_policy_with_frontier, "__module__", None) != __name__
    )
    if status == "shadow_pass" and apply and not dry_run:
        try:
            pending_policy_preimage = policy_file.read_bytes()
            parsed_previous = json.loads(pending_policy_preimage)
            if not isinstance(parsed_previous, dict):
                raise ValueError("active search policy must be an object")
            previous_policy = parsed_previous
        except FileNotFoundError:
            pending_policy_preimage = None
        except (OSError, json.JSONDecodeError, ValueError) as exc:
            record["status"] = "frontier_retry"
            record["reason"] = f"active search policy preimage is invalid: {exc}"
        record["previous_policy_sha256"] = _canonical_json_sha256(previous_policy)
        record["previous_policy_summary"] = _self_tune_previous_summary(previous_policy)
    if record["status"] == "shadow_pass" and apply:
        if dry_run:
            record["status"] = "dry_run"
        elif frontier_mode != "auto":
            record["status"] = "frontier_retry"
            record["reason"] = (
                "search policy mutation requires local semantic authority"
            )
        else:
            semantic_evidence = _self_tune_review_evidence(record)
            semantic_epoch = _self_tune_semantic_epoch(semantic_evidence)
            prior_history = read_jsonl(history_file)
            with decision_authority_lock():
                authority, authority_error = (
                    decision_authority.current_semantic_authority(
                        SEARCH_SELF_TUNE_LANE,
                        injected_reviewer=injected_reviewer,
                    )
                )
            if authority_error is not None or authority is None:
                prior_markers = [
                    row
                    for row in reversed(prior_history)
                    if _has_semantic_no_quorum_marker(row)
                ]
                if prior_markers:
                    prior_hold = persisted_semantic_no_quorum_hold(
                        prior_markers[0], lane=SEARCH_SELF_TUNE_LANE
                    )
                    record["status"] = (
                        "semantic_hold"
                        if prior_hold is not None
                        else "semantic_hold_malformed"
                    )
                    record["semantic_hold"] = prior_hold
                    record["last_failure_class"] = LOCAL_SEMANTIC_NO_QUORUM
                    record["reason"] = (
                        "semantic authority unavailable; retaining durable hold"
                    )
                else:
                    record["status"] = "frontier_retry"
                    record["reason"] = (
                        authority_error or "search self-tune authority unavailable"
                    )
            else:
                hold_state, prior_hold = _persisted_self_tune_hold(
                    prior_history,
                    epoch=semantic_epoch,
                    authority=authority,
                )
                if hold_state in {"same", "malformed"}:
                    record["status"] = (
                        "semantic_hold"
                        if hold_state == "same"
                        else "semantic_hold_malformed"
                    )
                    record["semantic_hold"] = prior_hold
                    record["last_failure_class"] = LOCAL_SEMANTIC_NO_QUORUM
                    record["reason"] = (
                        "local semantic models did not reach a safe quorum"
                        if hold_state == "same"
                        else "malformed local semantic no-quorum hold; refusing resample"
                    )
                else:
                    allowed = True
                    if budget is not None:
                        allowed, _reason = budget.consume("frontier")
                    if not allowed:
                        record["status"] = "budget_deferred"
                        record["reason"] = "frontier cycle budget exhausted"
                    else:
                        try:
                            frontier = (
                                frontier_reviewer(record)
                                if frontier_reviewer is not None
                                else review_search_policy_with_frontier(record)
                            )
                        except Exception as exc:
                            frontier = {
                                "decision": "needs_retry",
                                "summary": (
                                    "search policy reviewer failed: "
                                    f"{exc.__class__.__name__}: {exc}"
                                ),
                            }
                        if is_local_semantic_no_quorum(frontier):
                            try:
                                with decision_authority_lock():
                                    current_authority, current_error = (
                                        decision_authority.current_semantic_authority(
                                            SEARCH_SELF_TUNE_LANE,
                                            injected_reviewer=injected_reviewer,
                                        )
                                    )
                                    hold_error = current_error or (
                                        decision_authority.compare_semantic_authority(
                                            authority,
                                            current_authority,
                                            lane=SEARCH_SELF_TUNE_LANE,
                                        )
                                    )
                                    if hold_error is not None:
                                        raise ValueError(hold_error)
                                    semantic_hold = build_semantic_no_quorum_hold(
                                        SEARCH_SELF_TUNE_LANE,
                                        semantic_epoch,
                                        authority,
                                        frontier,
                                    )
                            except (TypeError, ValueError) as exc:
                                record["status"] = "frontier_retry"
                                record["reason"] = str(exc)
                            else:
                                record["status"] = "semantic_hold"
                                record["reason"] = (
                                    "local semantic models did not reach a safe quorum"
                                )
                                record["semantic_hold"] = semantic_hold
                                record["last_failure_class"] = LOCAL_SEMANTIC_NO_QUORUM
                        else:
                            record["frontier_review"] = frontier
                            with decision_authority_lock():
                                current_authority, current_error = (
                                    decision_authority.current_semantic_authority(
                                        SEARCH_SELF_TUNE_LANE,
                                        injected_reviewer=injected_reviewer,
                                    )
                                )
                                authority_error = current_error or (
                                    decision_authority.compare_semantic_authority(
                                        authority,
                                        current_authority,
                                        lane=SEARCH_SELF_TUNE_LANE,
                                    )
                                )
                                authority_error = authority_error or (
                                    decision_authority.semantic_verdict_authority_error(
                                        frontier,
                                        authority,
                                        lane=SEARCH_SELF_TUNE_LANE,
                                    )
                                )
                                if authority_error is None:
                                    semantic_artifact = _seal_search_review(
                                        kind="search_self_tune_verdict",
                                        lane=SEARCH_SELF_TUNE_LANE,
                                        evidence=semantic_evidence,
                                        review=frontier,
                                        authority=authority,
                                    )
                            if authority_error is not None or semantic_artifact is None:
                                record["status"] = "frontier_retry"
                                record["reason"] = (
                                    authority_error
                                    or "search self-tune verdict authority invalid"
                                )
                            else:
                                record["decision_artifact"] = semantic_artifact
                                if is_human_required_result(frontier):
                                    record["status"] = "human_required"
                                    record["reason"] = str(
                                        frontier.get("summary")
                                        or "frontier access requires external authority"
                                    )
                                elif frontier.get("decision") != "approved":
                                    record["status"] = (
                                        "frontier_rejected"
                                        if frontier.get("decision")
                                        in {"rejected", "quarantined"}
                                        else "frontier_retry"
                                    )
                                    record["reason"] = str(
                                        frontier.get("summary")
                                        or "frontier did not approve search policy"
                                    )
        if record["status"] == "shadow_pass" and semantic_artifact is not None:
            artifact = {
                "version": 1,
                "created_at": record["ts"],
                "source": "search_eval.self_tune",
                "weights": best["weights"],
                "holdout": locked_metrics,
                "previous": previous_policy,
                "decision_artifact": semantic_artifact,
            }
            artifact["policy_id"] = _self_tune_policy_id(artifact)
            policy_error = _self_tune_policy_error(artifact)
            if policy_error is not None:
                record["status"] = "frontier_retry"
                record["reason"] = policy_error
            if record["status"] == "shadow_pass":
                record["policy"] = artifact
                record["policy_id"] = artifact["policy_id"]
                mutation_allowed = True
                mutation_reason = "ok"
                if budget is not None:
                    mutation_allowed, mutation_reason = budget.consume("mutation")
                if not mutation_allowed:
                    record["status"] = "budget_deferred"
                    record["reason"] = mutation_reason
                else:
                    pending_policy = artifact
    if record.get("status") == "frontier_retry":
        candidate_payload = {
            "weights": (record.get("best") or {}).get("weights", {}),
            "dev": (record.get("best") or {}).get("dev", {}),
            "locked-test": (record.get("best") or {}).get("locked-test", {}),
        }
        candidate_hash = hashlib.sha256(
            json.dumps(
                candidate_payload, ensure_ascii=False, sort_keys=True, default=str
            ).encode("utf-8")
        ).hexdigest()
        prior_history = read_jsonl(history_file)
        prior = prior_history[-1] if prior_history else {}
        attempts = (
            int(prior.get("frontier_attempts") or 0) + 1
            if prior.get("candidate_hash") == candidate_hash
            else 1
        )
        record["candidate_hash"] = candidate_hash
        record["frontier_attempts"] = attempts
        if attempts >= 3:
            record["status"] = "frontier_quarantined"
            record["reason"] = (
                f"{record.get('reason', '')}; frontier retry limit exhausted"
            )
            record["next_attempt_at"] = None
        else:
            record["next_attempt_at"] = (
                datetime.now() + timedelta(minutes=15 * (2 ** max(0, attempts - 1)))
            ).isoformat(timespec="seconds")
    if not dry_run and record.get("status") != "budget_deferred":
        if semantic_artifact is not None and semantic_evidence is not None:
            with decision_authority_lock():
                current_authority, current_error = (
                    decision_authority.current_semantic_authority(
                        SEARCH_SELF_TUNE_LANE,
                        injected_reviewer=injected_reviewer,
                    )
                )
                authority_error = current_error or (
                    _self_tune_policy_error(
                        pending_policy,
                        current_authority=current_authority,
                    )
                    if pending_policy is not None
                    else _self_tune_artifact_error(
                        semantic_artifact,
                        evidence=semantic_evidence,
                        current_authority=current_authority,
                    )
                )
                if authority_error is None and pending_policy is not None:
                    try:
                        current_policy_preimage = policy_file.read_bytes()
                    except FileNotFoundError:
                        current_policy_preimage = None
                    except OSError as exc:
                        authority_error = (
                            f"active search policy preimage check failed: {exc}"
                        )
                        current_policy_preimage = None
                    if (
                        authority_error is None
                        and current_policy_preimage != pending_policy_preimage
                    ):
                        authority_error = "active search policy changed before effect"
                if authority_error is None:
                    if pending_policy is not None:
                        _atomic_write_text(
                            policy_file,
                            json.dumps(
                                pending_policy,
                                ensure_ascii=False,
                                indent=2,
                                sort_keys=True,
                            )
                            + "\n",
                        )
                        record["status"] = "applied"
                        record["applied"] = True
                    append_jsonl(history_file, record)
                else:
                    # A stale verdict is returned for observability but is not
                    # installed and does not become durable terminal history.
                    record["status"] = "frontier_retry"
                    record["applied"] = False
                    record["reason"] = authority_error
                    record["decision_authority_error"] = authority_error
        else:
            append_jsonl(history_file, record)
    return record


def rollback_search_policy(
    *,
    policy_file: Path = ACTIVE_SEARCH_POLICY_FILE,
    history_file: Path = SELF_TUNE_HISTORY_FILE,
    expected_policy_id: str | None = None,
    injected_reviewer: bool = False,
) -> dict[str, Any]:
    """Restore one approved policy's exact preimage under its authority lease."""

    try:
        expected_bytes = policy_file.read_bytes()
        policy = json.loads(expected_bytes)
    except (OSError, json.JSONDecodeError) as exc:
        return {
            "status": "rollback_blocked",
            "rolled_back": False,
            "reason": f"active search policy is unreadable: {exc}",
        }
    policy_id = policy.get("policy_id") if isinstance(policy, dict) else None
    previous = policy.get("previous") if isinstance(policy, dict) else None
    if (
        not isinstance(policy_id, str)
        or not policy_id
        or (expected_policy_id is not None and policy_id != expected_policy_id)
        or not isinstance(previous, Mapping)
        or _self_tune_policy_error(policy) is not None
    ):
        return {
            "status": "rollback_blocked",
            "rolled_back": False,
            "reason": "active search policy has no matching sealed approval",
        }

    with decision_authority_lock():
        try:
            if policy_file.read_bytes() != expected_bytes:
                return {
                    "status": "rollback_blocked",
                    "rolled_back": False,
                    "reason": "active search policy changed before rollback",
                }
        except OSError as exc:
            return {
                "status": "rollback_blocked",
                "rolled_back": False,
                "reason": f"active search policy disappeared before rollback: {exc}",
            }
        current_authority, current_error = (
            decision_authority.current_semantic_authority(
                SEARCH_SELF_TUNE_LANE,
                injected_reviewer=injected_reviewer,
            )
        )
        authority_error = current_error or _self_tune_policy_error(
            policy,
            current_authority=current_authority,
        )
        if authority_error is not None:
            return {
                "status": "rollback_blocked",
                "rolled_back": False,
                "reason": authority_error,
            }
        restored = dict(previous)
        _atomic_write_text(
            policy_file,
            json.dumps(restored, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        )
        record = {
            "ts": datetime.now().isoformat(timespec="seconds"),
            "status": "rolled_back",
            "rolled_back": True,
            "policy_id": policy_id,
            "restored_policy_id": (
                restored.get("policy_id")
                if isinstance(restored.get("policy_id"), str)
                else None
            ),
            "restored_sha256": _canonical_json_sha256(restored),
            "decision_artifact": policy["decision_artifact"],
        }
        append_jsonl(history_file, record)
    return record


def review_search_policy_with_frontier(
    record: dict[str, Any],
    *,
    repo_root: Path = REPO_ROOT,
    timeout: int | None = None,
) -> dict[str, Any]:
    """Ask the frontier model for the final veto on a validated policy."""
    from chronovisor.decision import routine_review
    from chronovisor.decision.decision_lane_prompts import build_search_self_tune_prompt

    prompt = build_search_self_tune_prompt(record)
    return routine_review.run_structured_review(
        prompt,
        routine_review.FRONTIER_DECISION_SCHEMA,
        repo_root=repo_root,
        timeout=timeout
        or int(os.environ.get("CHRONOVISOR_FRONTIER_TIMEOUT_SECONDS", "3600")),
        execute_patch=False,
        decision_lane="search_self_tune",
    )


def run_self_tune_due(
    *,
    golden_file: Path = GOLDEN_FILE,
    history_file: Path = SELF_TUNE_HISTORY_FILE,
    attempt_file: Path = SELF_TUNE_ATTEMPT_FILE,
    policy_file: Path = ACTIVE_SEARCH_POLICY_FILE,
    min_interval_hours: float = 7 * 24,
    apply: bool = True,
    dry_run: bool = False,
    frontier_mode: str = "auto",
    budget: Any | None = None,
    max_examples: int = 200,
    max_elapsed_seconds: float = 10 * 60,
) -> dict[str, Any]:
    attempt: dict[str, Any] = {}
    try:
        loaded_attempt = json.loads(attempt_file.read_text(encoding="utf-8"))
        if isinstance(loaded_attempt, dict):
            attempt = loaded_attempt
    except (FileNotFoundError, OSError, json.JSONDecodeError):
        pass
    next_attempt_at = str(attempt.get("next_attempt_at") or "")
    if next_attempt_at:
        try:
            next_attempt = datetime.fromisoformat(next_attempt_at)
            if datetime.now(next_attempt.tzinfo) < next_attempt:
                return {
                    "status": "skipped",
                    "reason": "budget_backoff",
                    "last_attempt_at": attempt.get("ts"),
                    "next_attempt_at": next_attempt_at,
                }
        except ValueError:
            pass

    history = read_jsonl(history_file)
    latest = history[-1] if history else {}
    last_ts = str(latest.get("ts") or "")
    due = True
    retry_pending = latest.get("status") == "frontier_retry"
    retry_at = str(latest.get("next_attempt_at") or "")
    if retry_pending and retry_at:
        try:
            due = datetime.now() >= datetime.fromisoformat(retry_at)
        except ValueError:
            due = True
    elif last_ts:
        try:
            due = datetime.now() - datetime.fromisoformat(last_ts) >= timedelta(
                hours=max(0.0, min_interval_hours)
            )
        except ValueError:
            due = True
    if not due:
        return {
            "status": "skipped",
            "reason": "interval_not_due",
            "last_run_at": last_ts,
        }
    result = self_tune(
        golden_file=golden_file,
        history_file=history_file,
        policy_file=policy_file,
        apply=apply,
        dry_run=dry_run,
        frontier_mode=frontier_mode,
        budget=budget,
        max_examples=max_examples,
        max_elapsed_seconds=max_elapsed_seconds,
    )
    if not dry_run:
        now = datetime.now()
        attempt_record = {
            "ts": now.isoformat(timespec="seconds"),
            "status": result.get("status"),
            "next_attempt_at": (
                now + timedelta(hours=max(0.0, min_interval_hours))
            ).isoformat(timespec="seconds")
            if result.get("status") == "budget_deferred"
            else None,
        }
        _atomic_write_text(
            attempt_file,
            json.dumps(attempt_record, ensure_ascii=False, indent=2, sort_keys=True)
            + "\n",
        )
    return result


def run_report(
    *,
    golden_file: Path = GOLDEN_FILE,
    variants: tuple[str, ...] = DEFAULT_VARIANTS,
    top_n: int = 20,
    limit: int = 0,
    source_filter: str = "all",
    save: bool = False,
    debug_dump: Path | None = None,
    failure_index: Path | None = None,
    sealed_manifest: Path | None = None,
    locked_e2e_artifact: Path | None = None,
) -> dict[str, Any]:
    examples = load_examples(
        golden_file, limit=max(0, limit), source_filter=source_filter
    )
    result = evaluate_examples(examples, variants=variants, top_n=top_n)
    payload = {
        "status": "ok",
        "dataset": {
            "golden_file": str(golden_file),
            "examples": len(examples),
            "reviewed": sum(1 for example in examples if example.reviewed),
            "splits": _count_by(examples, "split"),
            "languages": _count_by(examples, "language"),
            "kinds": _count_by(examples, "kind"),
            "sources": _count_by(examples, "source"),
            "source_filter": source_filter,
            "limit": max(0, limit),
        },
        "top_n": top_n,
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds").replace(
            "+00:00", "Z"
        ),
        "variants": result["variants"],
    }
    if debug_dump is not None:
        write_jsonl(debug_dump, result["debug_rows"])
        payload["debug_dump"] = str(debug_dump)
    if failure_index is not None:
        payload["failure_index"] = write_failure_index(
            result["debug_rows"], failure_index
        )
    if sealed_manifest is not None and locked_e2e_artifact is None:
        payload["sealed_manifest"] = write_sealed_manifest(
            examples,
            sealed_manifest,
            review_ledger_file=golden_file,
        )
    if locked_e2e_artifact is not None:
        payload["locked_e2e"] = write_locked_e2e_artifact(
            payload,
            examples,
            path=locked_e2e_artifact,
            frozen_manifest=sealed_manifest or MANUAL_MANIFEST_FILE,
        )
    if save:
        payload["baseline_file"] = str(save_baseline(payload))
    return payload


def _parse_variants(raw: str | None) -> tuple[str, ...]:
    if not raw:
        return DEFAULT_VARIANTS
    variants = tuple(item.strip() for item in raw.split(",") if item.strip())
    return variants or DEFAULT_VARIANTS


def print_report(payload: dict[str, Any]) -> None:
    dataset = payload["dataset"]
    print(f"dataset\t{dataset['examples']} examples\t{dataset['golden_file']}")
    print(f"reviewed\t{dataset['reviewed']}")
    for variant, data in payload["variants"].items():
        metrics = data["metrics"]
        print(
            "\t".join(
                [
                    variant,
                    f"recall@5={metrics['recall_at_5']:.3f}",
                    f"recall@20={metrics['recall_at_20']:.3f}",
                    f"mrr@10={metrics['mrr_at_10']:.3f}",
                    f"ndcg@10={metrics['ndcg_at_10']:.3f}",
                    f"negative@20={metrics['negative_hit_rate_at_20']:.3f}",
                    f"stale@20={metrics['stale_hit_rate_at_20']:.3f}",
                    f"p95={metrics['latency_ms']['p95']:.0f}ms",
                ]
            )
        )
    if payload.get("debug_dump"):
        print(f"debug_dump\t{payload['debug_dump']}")
    if payload.get("baseline_file"):
        print(f"baseline\t{payload['baseline_file']}")


def ci_gate(
    payload: dict[str, Any],
    *,
    variant: str = "hybrid-current",
    min_recall_at_5: float = 0.0,
    min_mrr_at_10: float = 0.0,
    max_negative_hit_rate_at_20: float = 1.0,
) -> dict[str, Any]:
    variants = (
        payload.get("variants") if isinstance(payload.get("variants"), dict) else {}
    )
    selected = variants.get(variant) if isinstance(variants, dict) else None
    if selected is None and variants:
        variant, selected = next(iter(variants.items()))
    if not isinstance(selected, dict):
        return {"status": "failed", "reason": "no variant metrics", "variant": variant}
    metrics = selected.get("metrics")
    if not isinstance(metrics, dict):
        return {"status": "failed", "reason": "missing metrics", "variant": variant}

    def metric(name: str, default: float) -> float:
        try:
            return float(metrics.get(name, default))
        except (TypeError, ValueError):
            return default

    failures = []
    if metric("recall_at_5", 0.0) < min_recall_at_5:
        failures.append("recall_at_5")
    if metric("mrr_at_10", 0.0) < min_mrr_at_10:
        failures.append("mrr_at_10")
    if metric("negative_hit_rate_at_20", 1.0) > max_negative_hit_rate_at_20:
        failures.append("negative_hit_rate_at_20")
    return {
        "status": "passed" if not failures else "failed",
        "variant": variant,
        "failures": failures,
        "thresholds": {
            "min_recall_at_5": min_recall_at_5,
            "min_mrr_at_10": min_mrr_at_10,
            "max_negative_hit_rate_at_20": max_negative_hit_rate_at_20,
        },
        "metrics": metrics,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate Chronovisor search ranking quality."
    )
    parser.add_argument("--golden-file", default=str(GOLDEN_FILE))
    parser.add_argument("--label-queue-file", default=str(LABEL_QUEUE_FILE))
    parser.add_argument("--feedback-file", default=str(RECALL_FEEDBACK_FILE))
    parser.add_argument("--log-file", default=str(RECALL_LOG_FILE))
    parser.add_argument("--output-file", default=str(GOLDEN_FILE))
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument(
        "--source-filter", choices=("all", "manual", "auto"), default="all"
    )
    parser.add_argument("--top-n", type=int, default=20)
    parser.add_argument("--variants", help="Comma-separated variants to evaluate.")
    parser.add_argument(
        "--debug-dump", help="Write per-query channel/result rows as JSONL."
    )
    parser.add_argument(
        "--failure-index",
        nargs="?",
        const=str(FAILURE_INDEX_FILE),
        help="Write failed query index JSONL.",
    )
    parser.add_argument(
        "--sealed-manifest",
        nargs="?",
        const=str(MANUAL_MANIFEST_FILE),
        help="Write a deterministic manifest with query hashes, never query text.",
    )
    parser.add_argument(
        "--locked-e2e-artifact",
        nargs="?",
        const=str(LOCKED_E2E_ARTIFACT),
        help="Seal the manual-94 retrieval through commit promotion gate.",
    )
    parser.add_argument("--build-golden", action="store_true")
    parser.add_argument("--build-label-queue", action="store_true")
    parser.add_argument(
        "--frontier-review-labels",
        action="store_true",
        help="Use a frontier model to promote trusted label-queue rows into the golden set.",
    )
    parser.add_argument(
        "--frontier-min-confidence",
        type=float,
        default=0.8,
        help="Deprecated no-op; confidence is retained only as review metadata.",
    )
    parser.add_argument(
        "--frontier-votes",
        type=int,
        default=1,
        help="Number of frontier votes required to agree before promotion.",
    )
    parser.add_argument("--frontier-timeout", type=int, default=None)
    parser.add_argument(
        "--self-tune",
        action="store_true",
        help="Run dev-only shadow self-tune with locked-test guard.",
    )
    parser.add_argument("--self-tune-history", default=str(SELF_TUNE_HISTORY_FILE))
    parser.add_argument("--report", action="store_true")
    parser.add_argument("--save-baseline", action="store_true")
    parser.add_argument(
        "--ci", action="store_true", help="Fail non-zero when metrics miss thresholds."
    )
    parser.add_argument("--ci-variant", default="hybrid-current")
    parser.add_argument("--min-recall-at-5", type=float, default=0.0)
    parser.add_argument("--min-mrr-at-10", type=float, default=0.0)
    parser.add_argument("--max-negative-hit-rate-at-20", type=float, default=1.0)
    parser.add_argument("--json", action="store_true")
    return parser


@contextlib.contextmanager
def _search_label_queue_lock(path: Path):
    import fcntl

    lock_path = path.with_suffix(path.suffix + ".lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a+", encoding="utf-8") as handle:
        os.chmod(lock_path, 0o600)
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def main(argv: list[str] | None = None) -> int:
    """Run the ``chronovisor-eval`` command-line entry point."""
    args = build_parser().parse_args(argv)
    if args.build_golden:
        payload = build_golden(
            feedback_file=Path(args.feedback_file).expanduser(),
            log_file=Path(args.log_file).expanduser(),
            output_file=Path(args.output_file).expanduser(),
            limit=args.limit,
        )
        if args.json:
            print(json.dumps(payload, ensure_ascii=False, indent=2, default=str))
        else:
            print(f"golden_file\t{payload['output_file']}")
            print(f"examples\t{payload['examples']}")
            print(f"reviewed\t{payload['reviewed']}")
        return 0
    if args.build_label_queue:
        queue_output = (
            Path(args.label_queue_file).expanduser()
            if args.label_queue_file != str(LABEL_QUEUE_FILE)
            else (
                Path(args.output_file).expanduser()
                if args.output_file != str(GOLDEN_FILE)
                else LABEL_QUEUE_FILE
            )
        )
        payload = build_label_queue(
            feedback_file=Path(args.feedback_file).expanduser(),
            log_file=Path(args.log_file).expanduser(),
            output_file=queue_output,
            limit=args.limit,
        )
        if args.json:
            print(json.dumps(payload, ensure_ascii=False, indent=2, default=str))
        else:
            print(f"label_queue\t{payload['output_file']}")
            print(f"examples\t{payload['examples']}")
            print(payload["note"])
        return 0
    if args.frontier_review_labels:
        payload = review_label_queue_with_frontier(
            queue_file=Path(args.label_queue_file).expanduser(),
            golden_file=Path(args.golden_file).expanduser(),
            limit=args.limit,
            min_confidence=args.frontier_min_confidence,
            votes=args.frontier_votes,
            timeout=args.frontier_timeout,
        )
        if args.json:
            print(json.dumps(payload, ensure_ascii=False, indent=2, default=str))
        else:
            print(f"frontier_label_review\t{payload['status']}")
            print(f"attempted\t{payload['attempted']}")
            print(f"promoted\t{payload['promoted']}")
            print(f"remaining\t{payload['remaining']}")
            print(f"golden_file\t{payload['golden_file']}")
        return 0
    if args.self_tune:
        payload = self_tune(
            golden_file=Path(args.golden_file).expanduser(),
            history_file=Path(args.self_tune_history).expanduser(),
        )
        if args.json:
            print(json.dumps(payload, ensure_ascii=False, indent=2, default=str))
        else:
            print(f"self_tune\t{payload['status']}")
            print(f"applied\t{payload['applied']}")
            print(f"history\t{args.self_tune_history}")
        return 0

    payload = run_report(
        golden_file=Path(args.golden_file).expanduser(),
        variants=_parse_variants(args.variants),
        top_n=args.top_n,
        limit=args.limit,
        source_filter="manual"
        if args.ci and args.source_filter == "all"
        else args.source_filter,
        save=args.save_baseline,
        debug_dump=Path(args.debug_dump).expanduser() if args.debug_dump else None,
        failure_index=Path(args.failure_index).expanduser()
        if args.failure_index
        else None,
        sealed_manifest=Path(args.sealed_manifest).expanduser()
        if args.sealed_manifest
        else None,
        locked_e2e_artifact=Path(args.locked_e2e_artifact).expanduser()
        if args.locked_e2e_artifact
        else None,
    )
    if args.ci:
        payload["ci_gate"] = ci_gate(
            payload,
            variant=args.ci_variant,
            min_recall_at_5=max(0.0, args.min_recall_at_5),
            min_mrr_at_10=max(0.0, args.min_mrr_at_10),
            max_negative_hit_rate_at_20=max(0.0, args.max_negative_hit_rate_at_20),
        )
    if args.json:
        print(json.dumps(payload, ensure_ascii=False, indent=2, default=str))
    else:
        print_report(payload)
        if args.ci:
            print(f"ci_gate\t{payload['ci_gate']['status']}")
    if args.ci and payload["ci_gate"]["status"] != "passed":
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
