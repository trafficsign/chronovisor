# ruff: noqa: F401
#!/usr/bin/env python3
"""Discover and enforce ownership of Chronovisor runtime state resources."""

from __future__ import annotations

import argparse
import ast
import copy
import hashlib
import io
import json
import plistlib
import re
import subprocess
import tarfile
import tomllib
from collections import Counter
from pathlib import Path, PurePosixPath
from typing import Any

REGISTRY_PATH = Path("docs/refactoring/runtime-state-owners.json")
BASELINE_PATH = Path("docs/refactoring/runtime-state-baseline.json")
FROZEN_SOURCE_HEAD = "fec76ac919b1cb0f64e772f85ceda46163df309c"
REGISTRY_SCHEMA_VERSION = 1
BASELINE_SCHEMA_VERSION = 1
RESOURCE_KINDS = frozenset({"artifact", "queue", "lock", "socket", "schema", "worker"})
BASELINE_ID_FIELDS = ("discovery_ids", "resource_ids")
NAME_SUFFIXES = (
    "FILE",
    "DIR",
    "ROOT",
    "PATH",
    "QUEUE",
    "LEDGER",
    "LOCK",
    "SOCKET",
    "SOCK",
    "STATE",
    "STATUS",
    "MANIFEST",
    "ARTIFACT",
    "CACHE",
    "INDEX",
    "DB",
    "SCHEMA",
    "SCHEMA_VERSION",
)
SOURCE_OR_DEPLOYMENT_NAMES = frozenset(
    {
        "PROJECT_ROOT",
        "REPO_ROOT",
        "STATIC_DIR",
        "LAUNCH_AGENT_DIR",
        "WRAPPER_DIR",
    }
)
UNKNOWN_VALUES = frozenset({"", "*", "any", "none", "tbd", "unknown", "wildcard"})
MANUAL_RESOURCE_FIELDS = (
    "owner_package",
    "owner_symbol",
    "writers",
    "readers",
    "coordination",
    "lifecycle",
    "compatibility",
)
BASELINE_ROOT_KEYS = (
    "schema_version",
    "source_baseline_head",
    "semantic_identity",
    "discovery_ids",
    "resource_ids",
    "counts",
)
REGISTRY_ROOT_KEYS = (
    "schema_version",
    "source_baseline_head",
    "baseline_sha256",
    "policy",
    "counts",
    "resources",
    "exclusions",
    "lock_protocol_sites",
)
RESOURCE_BASE_KEYS = (
    "id",
    "kind",
    "locator",
    "owner_package",
    "owner_symbol",
    "writers",
    "readers",
    "coordination",
    "lifecycle",
    "evidence",
    "compatibility",
    "discovery_ids",
)
BASELINE_SEMANTIC_IDENTITY = (
    "line-independent discovery and resource IDs with monotonic active/retired sets"
)
REGISTRY_POLICY = {
    "discovery": (
        "module-level durable path origins and string schema identities; "
        "explicit sockets, entrypoints, and launchd workers"
    ),
    "exclusions": (
        "prompt/in-memory schemas, version-only constants, status enums, "
        "process-local locks/caches, and source/fixture/deployment paths"
    ),
    "new_resources": "fail closed against the frozen active/retired ceiling",
}

# These are ownership facts that cannot be inferred from lexical declaration
# order.  They are deliberately keyed by the stable locator instead of source
# line so regeneration preserves the reviewed contract.
RESOURCE_OWNERSHIP_OVERRIDES: dict[tuple[str, str], dict[str, Any]] = {
    (
        "artifact",
        "$CHRONOVISOR_ROOT/runtime/recall-field/promotion.json",
    ): {
        "owner_symbol": "chronovisor.recall.recall_growth:PROMOTION_ARTIFACT",
        "writers": ["chronovisor.recall.recall_growth:_persist_growth_artifacts"],
        "readers": ["chronovisor.recall.recall_field_candidate:authority_allowed"],
        "format": {
            "status": "versioned",
            "schema_id": "chronovisor.recall-field-promotion",
            "version": 4,
            "source": "chronovisor.recall.recall_growth:_persist_growth_artifacts",
        },
    },
    (
        "artifact",
        "$CHRONOVISOR_ROOT/runtime/search-eval/recall-field-locked-e2e.json",
    ): {
        "owner_symbol": "chronovisor.search.search_eval:LOCKED_E2E_ARTIFACT",
        "writers": ["chronovisor.search.search_eval:write_locked_e2e_artifact"],
        "readers": ["chronovisor.recall.recall_growth:retrieval_locked_e2e_status"],
        "format": {
            "status": "versioned",
            "schema_id": "chronovisor.search.locked-e2e",
            "version": 2,
            "source": "chronovisor.search.search_eval:write_locked_e2e_artifact",
        },
    },
    (
        "artifact",
        "$CHRONOVISOR_ROOT/runtime/typed-graph/consensus-receipts.jsonl",
    ): {
        "owner_symbol": "chronovisor.knowledge_graph.consensus:RECEIPT_LEDGER",
        "writers": ["chronovisor.knowledge_graph.consensus:verify_pending_relations"],
        "readers": ["chronovisor.recall.recall_label_factory:build_label_ledger"],
    },
    (
        "artifact",
        "$CHRONOVISOR_ROOT/runtime/typed-graph/candidate-trace.jsonl",
    ): {
        "owner_symbol": "chronovisor.recall.recall_runtime:TYPED_GRAPH_TRACE_FILE",
        "writers": ["chronovisor.recall.recall_runtime:search_candidates"],
        "readers": [
            "chronovisor.knowledge_graph.runtime:run_graph_maintenance",
            "chronovisor.recall.recall_label_factory:build_label_ledger",
        ],
    },
    ("artifact", "$CHRONOVISOR_ROOT/.index/embeddings.sqlite"): {
        "owner_symbol": "chronovisor.search.search:EMBEDDINGS_DB",
        "writers": ["chronovisor.search.search:EMBEDDINGS_DB"],
        "readers": ["chronovisor.search.semantic_index:archive_legacy_search_index"],
    },
    ("artifact", "$CHRONOVISOR_ROOT/.embeddings.json"): {
        "owner_symbol": "chronovisor.search.search:JSON_EMBEDDINGS_FILE",
        "writers": ["chronovisor.search.search:JSON_EMBEDDINGS_FILE"],
        "readers": ["chronovisor.search.search:EMBEDDINGS_FILE"],
    },
    ("queue", "$CHRONOVISOR_ROOT/review/raw-replay-queue.jsonl"): {
        "owner_symbol": "chronovisor.raw.raw_replay:QUEUE_FILE",
        "writers": [
            "chronovisor.raw.raw_replay:build_queue",
            "chronovisor.raw.raw_replay:run_pending_queue",
        ],
        "readers": ["chronovisor.raw.raw_replay:_read_jsonl"],
        "coordination": {"protocol": "flock-then-atomic-replace"},
        "format": {
            "status": "versioned",
            "schema_id": "chronovisor.raw-replay-queue",
            "version": 2,
            "source": "chronovisor.raw.raw_replay:SCHEMA_VERSION",
        },
    },
    ("queue", "$CHRONOVISOR_ROOT/runtime/semantic-jobs.sqlite"): {
        "owner_symbol": "chronovisor.search.semantic_jobs:SEMANTIC_JOBS_DB",
        "writers": [
            "chronovisor.search.semantic_jobs:enqueue_page",
            "chronovisor.search.semantic_jobs:enqueue_pages",
            "chronovisor.search.semantic_jobs:enqueue_rebuild",
            "chronovisor.search.semantic_jobs:claim_next",
            "chronovisor.search.semantic_jobs:complete",
            "chronovisor.search.semantic_jobs:fail",
            "chronovisor.search.semantic_jobs:prune_completed_jobs",
        ],
        "readers": ["chronovisor.search.semantic_jobs:job_status"],
        "coordination": {
            "protocol": "sqlite-wal-transactions",
            "transaction": "BEGIN IMMEDIATE",
        },
        "format": {
            "status": "unversioned",
            "rationale": "SQLite application schema has no declared version constant",
            "migration_owner": "chronovisor.search.semantic_jobs:SEMANTIC_JOBS_DB",
        },
        "compatibility": ["sqlite-wal", "transaction:BEGIN IMMEDIATE"],
    },
    ("artifact", "$CHRONOVISOR_ROOT/claims/claims.jsonl"): {
        "owner_symbol": "chronovisor.recall.claims:CLAIMS_FILE",
        "writers": [
            "chronovisor.recall.claims:append_page_claims",
            "chronovisor.recall.claims:sanitize_claim_ledger",
        ],
        "readers": ["chronovisor.raw.raw_replay:CLAIMS_FILE"],
        "coordination": {
            "protocol": "shared-sidecar-flock",
        },
    },
    ("queue", "$CHRONOVISOR_ROOT/recall/search-label-queue.jsonl"): {
        "owner_symbol": "chronovisor.search.search_eval:LABEL_QUEUE_FILE",
        "writers": [
            "chronovisor.ops.golden_expand:expand_golden_from_recall_questions",
            "chronovisor.search.search_eval:build_label_queue",
            "chronovisor.search.search_eval:review_label_queue_with_frontier",
        ],
        "readers": [
            "chronovisor.search.search_eval:build_label_queue",
            "chronovisor.search.search_eval:review_label_queue_with_frontier",
        ],
        "coordination": {"protocol": "shared-sidecar-flock"},
    },
    ("artifact", "$CHRONOVISOR_ROOT/recall/feedback.jsonl"): {
        "owner_symbol": "chronovisor.recall.recall_runtime:RECALL_FEEDBACK_FILE",
        "writers": ["chronovisor.recall.recall_runtime:append_feedback"],
    },
    ("artifact", "$CHRONOVISOR_ROOT/recall/recall-log.jsonl"): {
        "owner_symbol": "chronovisor.recall.recall_runtime:RECALL_LOG_FILE",
        "writers": ["chronovisor.recall.recall_runtime:append_recall_log"],
    },
    ("artifact", "$CHRONOVISOR_ROOT/recall/pull-log.jsonl"): {
        "owner_symbol": "chronovisor.recall.recall_runtime:RECALL_PULL_LOG_FILE",
        "writers": ["chronovisor.hosts.server:_append_pull_log"],
    },
    (
        "artifact",
        "$CHRONOVISOR_ROOT/runtime/search-eval/manual-94-manifest.json",
    ): {
        "owner_symbol": "chronovisor.search.search_eval:MANUAL_MANIFEST_FILE",
        "writers": ["chronovisor.search.search_eval:write_sealed_manifest"],
    },
    (
        "artifact",
        "$CHRONOVISOR_ROOT/runtime/recall-field/candidate-trace.jsonl",
    ): {
        "owner_symbol": "chronovisor.recall.recall_field_candidate:CANDIDATE_TRACE_FILE",
        "writers": ["chronovisor.recall.recall_field_candidate:append_candidate_trace"],
    },
    (
        "artifact",
        "$CHRONOVISOR_ROOT/runtime/recall-compiler/shadow-trace.jsonl",
    ): {
        "owner_symbol": "chronovisor.recall.recall_compiler:SHADOW_TRACE_FILE",
        "writers": ["chronovisor.recall.recall_compiler:append_shadow_trace"],
    },
    ("artifact", "$CHRONOVISOR_ROOT/runtime/status.json"): {
        "owner_symbol": "chronovisor.ops.runtime_status:STATUS_FILE",
        "writers": ["chronovisor.ops.runtime_status:write_status"],
    },
    ("schema", "chronovisor.ops.deadman-heartbeat.v1"): {
        "owner_symbol": "chronovisor.ops.deadman:HEARTBEAT_SCHEMA",
        "writers": ["chronovisor.ops.deadman:HEARTBEAT_SCHEMA"],
    },
    ("schema", "chronovisor.classification-disabled-baseline.v1"): {
        "owner_symbol": (
            "chronovisor.lab.classification_fixture_set:DISABLED_BASELINE_SCHEMA"
        ),
        "writers": [
            "chronovisor.lab.classification_fixture_set:DISABLED_BASELINE_SCHEMA"
        ],
        "readers": [
            "chronovisor.lab.classification_fixture_set:DISABLED_BASELINE_SCHEMA"
        ],
    },
    ("schema", "chronovisor.classification-inference-dto.v1"): {
        "owner_symbol": (
            "chronovisor.lab.classification_fixture_set:INFERENCE_DTO_SCHEMA"
        ),
        "writers": [
            "chronovisor.lab.classification_fixture_set:INFERENCE_DTO_SCHEMA"
        ],
        "readers": [
            "chronovisor.lab.classification_fixture_set:INFERENCE_DTO_SCHEMA"
        ],
    },
    ("artifact", "$CHRONOVISOR_ROOT/runtime"): {
        "owner_symbol": "chronovisor.ops.runtime_status:RUNTIME_DIR",
        "writers": ["chronovisor.ops.runtime_status:RUNTIME_DIR"],
    },
    ("artifact", "$CHRONOVISOR_ROOT/.index/semantic/generations"): {
        "owner_symbol": "chronovisor.search.semantic_index:GENERATIONS_DIR",
        "writers": ["chronovisor.search.semantic_index:build_generation"],
        "format": {
            "status": "versioned",
            "schema_id": "chronovisor.semantic-index-generation",
            "version": 3,
            "source": "chronovisor.search.semantic_index:INDEX_SCHEMA_VERSION",
        },
    },
    ("artifact", "$CHRONOVISOR_ROOT/runtime/dashboard-access-token"): {
        "owner_symbol": "chronovisor.ops.dashboard:serve",
        "writers": ["chronovisor.ops.dashboard:serve"],
        "readers": ["chronovisor.ops.dashboard:serve"],
        "compatibility": ["secret", "mode:0600", "rotation-owned-by:dashboard"],
        "lifecycle": {
            "retention": "until explicit dashboard token rotation",
            "recovery_owner": "chronovisor.ops.dashboard:serve",
            "recovery_contract": "Regenerate with mode 0600; never log token bytes.",
        },
    },
    ("artifact", "$CHRONOVISOR_ROOT/runtime/dashboard-credentials.json"): {
        "owner_symbol": "chronovisor.ops.dashboard:serve",
        "writers": ["chronovisor.ops.dashboard:serve"],
        "readers": ["chronovisor.ops.dashboard:serve"],
        "compatibility": ["secret", "mode:0600", "rotation-owned-by:dashboard"],
        "lifecycle": {
            "retention": "until explicit dashboard credential rotation",
            "recovery_owner": "chronovisor.ops.dashboard:serve",
            "recovery_contract": "Regenerate with mode 0600; never log credentials.",
        },
    },
    ("artifact", "$HOME/.chronovisor/runtime/searxng/secret"): {
        "owner_symbol": "script:scripts/chronovisor-searxng",
        "writers": ["script:scripts/chronovisor-searxng"],
        "readers": ["script:scripts/chronovisor-searxng"],
        "compatibility": ["secret", "umask:077", "rotation-owned-by:searxng"],
        "lifecycle": {
            "retention": "until explicit SearXNG secret rotation",
            "recovery_owner": "script:scripts/chronovisor-searxng",
            "recovery_contract": "Regenerate under umask 077; never log secret bytes.",
        },
    },
}


LOCK_PROTECTS_LOCATORS: dict[str, tuple[tuple[str, str], ...]] = {
    "$CHRONOVISOR_ROOT/review/raw-replay-queue.jsonl.lock": (
        ("queue", "$CHRONOVISOR_ROOT/review/raw-replay-queue.jsonl"),
    ),
    "$CHRONOVISOR_ROOT/claims/claims.jsonl.lock": (
        ("artifact", "$CHRONOVISOR_ROOT/claims/claims.jsonl"),
    ),
    "$CHRONOVISOR_ROOT/recall/search-label-queue.jsonl.lock": (
        ("queue", "$CHRONOVISOR_ROOT/recall/search-label-queue.jsonl"),
    ),
    "$CHRONOVISOR_ROOT/.index/semantic/activation.lock": (
        ("artifact", "$CHRONOVISOR_ROOT/.index/semantic/active.json"),
    ),
    "$CHRONOVISOR_ROOT/recall/evidence-certificate-ledger.jsonl.lock": (
        ("artifact", "$CHRONOVISOR_ROOT/recall/evidence-certificate-ledger.jsonl"),
    ),
    "$CHRONOVISOR_ROOT/runtime/background-jobs/state.lock": (
        ("artifact", "$CHRONOVISOR_ROOT/runtime/background-jobs/state.json"),
    ),
    "$CHRONOVISOR_ROOT/runtime/model-lab/model-lab.lock": (
        ("artifact", "$CHRONOVISOR_ROOT/runtime/model-lab/active-policy.json"),
        ("artifact", "$CHRONOVISOR_ROOT/runtime/model-lab/state.json"),
        ("artifact", "$CHRONOVISOR_ROOT/runtime/model-lab/replay.jsonl"),
        ("artifact", "$CHRONOVISOR_ROOT/runtime/model-lab/history.jsonl"),
    ),
    "$CHRONOVISOR_ROOT/runtime/convergence/state.lock": (
        ("artifact", "$CHRONOVISOR_ROOT/runtime/convergence/state.json"),
        ("artifact", "$CHRONOVISOR_ROOT/runtime/convergence/events.jsonl"),
    ),
    "$CHRONOVISOR_ROOT/runtime/librarian/page-registry.lock": (
        ("artifact", "$CHRONOVISOR_ROOT/runtime/librarian/page-registry.json"),
        (
            "artifact",
            "$CHRONOVISOR_ROOT/runtime/librarian/page-registry-events.jsonl",
        ),
    ),
    "$CHRONOVISOR_ROOT/runtime/librarian/collection-registry.lock": (
        (
            "artifact",
            "$CHRONOVISOR_ROOT/runtime/librarian/collection-registry.json",
        ),
    ),
    "$CHRONOVISOR_ROOT/runtime/managed-holds/state.json.lock": (
        ("artifact", "$CHRONOVISOR_ROOT/runtime/managed-holds/state.json"),
    ),
    "$CHRONOVISOR_ROOT/knowledge-graph/store.lock": (
        ("artifact", "$CHRONOVISOR_ROOT/knowledge-graph/relation-events.jsonl"),
        ("artifact", "$CHRONOVISOR_ROOT/knowledge-graph/relation-snapshot.json"),
        ("artifact", "$CHRONOVISOR_ROOT/knowledge-graph/entity-snapshot.json"),
        ("artifact", "$CHRONOVISOR_ROOT/knowledge-graph/community-snapshot.json"),
        ("artifact", "$CHRONOVISOR_ROOT/knowledge-graph/builder-state.json"),
        (
            "artifact",
            "$CHRONOVISOR_ROOT/knowledge-graph/community-summary-state.json",
        ),
    ),
    "$CHRONOVISOR_ROOT/runtime/research/consolidation.lock": (
        ("artifact", "$CHRONOVISOR_ROOT/runtime/research/consolidation-state.json"),
    ),
    "$CHRONOVISOR_ROOT/runtime/failures/state.lock": (
        ("artifact", "$CHRONOVISOR_ROOT/runtime/failures/state.json"),
    ),
    "$CHRONOVISOR_ROOT/recall/audit.lock": (
        ("artifact", "$CHRONOVISOR_ROOT/recall/audit-state.json"),
        ("artifact", "$CHRONOVISOR_ROOT/recall/pull-consumed.jsonl"),
        ("worker", "chronovisor-recall-audit"),
    ),
    "$CHRONOVISOR_ROOT/runtime/sleep-cycle.lock": (
        ("artifact", "$CHRONOVISOR_ROOT/runtime/sleep-cycle-history.jsonl"),
        ("artifact", "$CHRONOVISOR_ROOT/runtime/sleep-cycle-active-lane.json"),
        ("worker", "chronovisor-sleep"),
    ),
    "$CHRONOVISOR_ROOT/runtime/research/research-generation.lock": (
        ("artifact", "$CHRONOVISOR_ROOT/runtime/research/active-research.json"),
        ("worker", "chronovisor-research"),
    ),
    "$CHRONOVISOR_ROOT/runtime/chronovisor-mutation.lock": (
        ("artifact", "$CHRONOVISOR_ROOT/pages"),
        ("artifact", "$CHRONOVISOR_ROOT/system"),
    ),
    "$CHRONOVISOR_ROOT/runtime/decision-authority.lock": (
        ("artifact", "$CHRONOVISOR_ROOT/pages"),
        ("artifact", "$CHRONOVISOR_ROOT/system"),
    ),
    "$CHRONOVISOR_ROOT/runtime/recall-improvement/run-due.lock": (
        ("worker", "chronovisor-recall-improve"),
    ),
    "$CHRONOVISOR_ROOT/runtime/accelerator-inference.lock": (
        ("worker", "chronovisor-recall"),
        ("worker", "chronovisor-research"),
        ("worker", "chronovisor-semantic-service"),
    ),
}


def _run_git(root: Path, *args: str, check: bool = True) -> str:
    completed = subprocess.run(
        ["git", *args],
        cwd=root,
        text=True,
        capture_output=True,
        check=False,
    )
    if check and completed.returncode != 0:
        raise RuntimeError(completed.stderr.strip() or "git command failed")
    return completed.stdout


def _canonical_sha256(value: Any) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _json_document_bytes(value: Any) -> bytes:
    try:
        return (
            json.dumps(
                value,
                ensure_ascii=False,
                indent=2,
                allow_nan=False,
            )
            + "\n"
        ).encode("utf-8")
    except (TypeError, ValueError, UnicodeEncodeError, RecursionError) as exc:
        raise ValueError("runtime ownership document is not canonical JSON") from exc


def _decode_json_document(raw: bytes) -> dict[str, Any]:
    def reject_constant(value: str) -> None:
        raise ValueError(f"non-finite JSON constant is forbidden: {value}")

    def unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON key is forbidden: {key}")
            result[key] = value
        return result

    try:
        decoded = raw.decode("utf-8")
        value = json.loads(
            decoded,
            parse_constant=reject_constant,
            object_pairs_hook=unique_object,
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError, RecursionError) as exc:
        raise ValueError(f"runtime ownership document is invalid: {exc}") from exc
    if type(value) is not dict:
        raise ValueError("runtime ownership document root must be an object")
    if _json_document_bytes(value) != raw:
        raise ValueError("runtime ownership document is not in canonical byte form")
    return value


def _same_json_value(left: Any, right: Any) -> bool:
    if type(left) is not type(right):
        return False
    if isinstance(left, dict) and isinstance(right, dict):
        return tuple(left) == tuple(right) and all(
            _same_json_value(left[key], right[key]) for key in left
        )
    if isinstance(left, (list, tuple)) and isinstance(right, (list, tuple)):
        return len(left) == len(right) and all(
            _same_json_value(left_item, right_item)
            for left_item, right_item in zip(left, right, strict=True)
        )
    try:
        return _json_document_bytes(left) == _json_document_bytes(right)
    except ValueError:
        return False


def _semantic_id(prefix: str, value: Any) -> str:
    return f"{prefix}:{_canonical_sha256(value)}"


__all__ = [
    "REGISTRY_PATH",
    "BASELINE_PATH",
    "FROZEN_SOURCE_HEAD",
    "REGISTRY_SCHEMA_VERSION",
    "BASELINE_SCHEMA_VERSION",
    "RESOURCE_KINDS",
    "BASELINE_ID_FIELDS",
    "NAME_SUFFIXES",
    "SOURCE_OR_DEPLOYMENT_NAMES",
    "UNKNOWN_VALUES",
    "MANUAL_RESOURCE_FIELDS",
    "BASELINE_ROOT_KEYS",
    "REGISTRY_ROOT_KEYS",
    "RESOURCE_BASE_KEYS",
    "BASELINE_SEMANTIC_IDENTITY",
    "REGISTRY_POLICY",
    "RESOURCE_OWNERSHIP_OVERRIDES",
    "LOCK_PROTECTS_LOCATORS",
    "_run_git",
    "_canonical_sha256",
    "_json_document_bytes",
    "_decode_json_document",
    "_same_json_value",
    "_semantic_id",
]
