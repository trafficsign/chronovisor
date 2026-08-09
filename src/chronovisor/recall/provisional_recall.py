"""Projection-only recall namespace for semantic no-quorum raw units.

The namespace never indexes a transcript raw directly.  It accepts only
verified deterministic semantic projection children, marks every hit as
unintegrated/untrusted, caps ranking, and cannot authorize wiki mutation.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections import Counter
from pathlib import Path
from typing import Any

from chronovisor.core.durable_state import (
    DurableStateError,
    canonical_bytes,
    read_sealed_json,
    write_sealed_json,
)
from chronovisor.core.sealed_artifact_decoder import schema_matches
from chronovisor.ingest.raw_semantic_projection import (
    PROJECTION_CHILD_SCHEMA,
    PROJECTION_POLICY_VERSION,
    verify_projection_child,
)
from chronovisor.search.search_types import tokenize

PROVISIONAL_SCHEMA_VERSION = 1
MAX_HITS = 3
RANK_CAP = 0.25
MAX_SNIPPET_CHARS = 600
_TOKEN_RE = re.compile(r"[\w\-]{2,}", re.UNICODE)
_SECRET_RE = re.compile(
    r"(?:api[_-]?key|authorization|bearer|password|secret|private[_-]?key)"
    r"\s*[:=]\s*[^\s]{6,}",
    re.IGNORECASE,
)


class ProvisionalRecallError(RuntimeError):
    pass


def _tokens(text: str) -> set[str]:
    # Keep the lightweight word tokens used by the original namespace and add
    # the production search tokenizer's CJK bigrams. Japanese prompts must not
    # collapse into one giant token between punctuation marks.
    tokens = {match.group(0).casefold() for match in _TOKEN_RE.finditer(text)}
    tokens.update(tokenize(text))
    return tokens


def _token_density(query_tokens: set[str], text: str) -> float:
    ordered = tokenize(text)
    positions = [idx for idx, token in enumerate(ordered) if token in query_tokens]
    if not positions:
        return 0.0
    distinct = len({ordered[idx] for idx in positions})
    span = max(positions) - min(positions) + 1
    return min(1.0, distinct / max(1, span))


def _source_host(path: Path) -> str:
    name = path.name.casefold()
    if "codex" in name:
        return "codex"
    if "claude" in name:
        return "claude"
    return "unknown"


def _projection_entry(path: Path) -> dict[str, Any] | None:
    try:
        verified_child = verify_projection_child(path)
    except Exception:
        return None
    try:
        raw = path.read_bytes()
        payload = json.loads(raw.decode("utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return None
    if (
        not isinstance(payload, dict)
        or not schema_matches(payload.get("schema"), PROJECTION_CHILD_SCHEMA)
        or payload.get("kind") != "semantic_projection_child"
        or payload.get("projection_policy_version") != PROJECTION_POLICY_VERSION
    ):
        return None
    records = payload.get("records")
    if not isinstance(records, list):
        return None
    safe_records: list[dict[str, Any]] = []
    for row in records:
        if not isinstance(row, dict):
            continue
        text = row.get("text")
        role = row.get("role")
        if not isinstance(text, str) or role not in {"user", "assistant"}:
            continue
        encoded = text.encode("utf-8")
        if row.get("segment_sha256") != hashlib.sha256(encoded).hexdigest():
            return None
        if _SECRET_RE.search(text):
            continue
        safe_records.append(
            {
                "source_record_index": row.get("source_record_index"),
                "role": role,
                "text": text,
                "text_sha256": hashlib.sha256(encoded).hexdigest(),
            }
        )
    if not safe_records:
        return None
    projection_id = payload.get("projection_id")
    child_id = payload.get("child_id")
    if not isinstance(projection_id, str) or not isinstance(child_id, str):
        return None
    return {
        "provisional_id": hashlib.sha256(
            canonical_bytes(
                {
                    "raw_file": path.name,
                    "projection_id": projection_id,
                    "child_id": child_id,
                }
            )
        ).hexdigest(),
        "raw_file": path.name,
        "raw_sha256": hashlib.sha256(raw).hexdigest(),
        "projection_id": projection_id,
        "child_id": child_id,
        "records": safe_records,
        "unintegrated": True,
        "content_is_untrusted": True,
        "mutation_evidence_allowed": False,
        "host_boundary": "citation_only",
        "source_host": _source_host(path),
        "sensitivity_filter": "secret_like_records_omitted",
        "verified_child_file_sha256": verified_child.file_sha256,
    }


def sync_index(*, chronovisor_root: Path) -> dict[str, Any]:
    from chronovisor.ingest.failure_supervisor import (
        SEMANTIC_NO_QUORUM_DEFER_REASON,
        operational_deferred_raw_files,
    )
    from chronovisor.ingest.raw_replay import is_raw_retracted

    raw_dir = chronovisor_root / "raw"
    artifact_dir = chronovisor_root / "runtime" / "raw-projections" / "artifacts"
    raw_paths = sorted(raw_dir.glob("*.md"))
    if artifact_dir.exists():
        raw_paths.extend(sorted(artifact_dir.glob("*.md")))
        raw_paths = sorted(dict.fromkeys(raw_paths), key=lambda path: path.name)
    deferred = operational_deferred_raw_files(raw_paths)
    entries: list[dict[str, Any]] = []
    for raw_path in raw_paths:
        if deferred.get(raw_path.name) != SEMANTIC_NO_QUORUM_DEFER_REASON:
            continue
        if is_raw_retracted(raw_path):
            continue
        entry = _projection_entry(raw_path)
        if entry is not None:
            entries.append(entry)
    entries.sort(key=lambda row: str(row["provisional_id"]))
    path = chronovisor_root / "runtime" / "provisional-recall" / "index.json"
    return write_sealed_json(
        path,
        {
            "schema_version": PROVISIONAL_SCHEMA_VERSION,
            "namespace": "unintegrated_semantic_projection",
            "projection_policy_version": PROJECTION_POLICY_VERSION,
            "rank_cap": RANK_CAP,
            "mutation_evidence_allowed": False,
            "entries": entries,
        },
        backup=True,
    )


def search_provisional(
    query: str,
    *,
    chronovisor_root: Path,
    limit: int = MAX_HITS,
) -> list[dict[str, Any]]:
    try:
        index = sync_index(chronovisor_root=chronovisor_root)
    except Exception:
        # Stale provisional data is less safe than returning no provisional hit;
        # forget/retract/integration removal must take effect immediately.
        return []
    query_tokens = _tokens(query)
    if not query_tokens:
        return []
    documents: list[tuple[dict[str, Any], dict[str, Any], str, set[str]]] = []
    for entry in index.get("entries", []):
        if not isinstance(entry, dict):
            continue
        for record in entry.get("records", []):
            if not isinstance(record, dict) or not isinstance(record.get("text"), str):
                continue
            text = record["text"]
            documents.append((entry, record, text, _tokens(text)))

    document_frequency: Counter[str] = Counter()
    for _entry, _record, _text, tokens in documents:
        document_frequency.update(tokens)
    document_count = max(1, len(documents))
    query_weights = {
        token: math.log((document_count + 1) / (document_frequency.get(token, 0) + 1))
        + 1.0
        for token in query_tokens
    }
    total_query_weight = max(1.0, sum(query_weights.values()))
    normalized_query = " ".join(query.casefold().split())

    ranked: list[tuple[tuple[float, float, float, int, str], dict[str, Any]]] = []
    for entry, record, text, text_tokens in documents:
        overlap = query_tokens & text_tokens
        if not overlap:
            continue
        weighted_coverage = (
            sum(query_weights[token] for token in overlap) / total_query_weight
        )
        density = _token_density(overlap, text)
        normalized_text = " ".join(text.casefold().split())
        phrase_bonus = (
            0.04
            if len(normalized_query) >= 4 and normalized_query in normalized_text
            else 0.0
        )
        score = min(
            RANK_CAP,
            0.03 + (weighted_coverage * 0.15) + (density * 0.03) + phrase_bonus,
        )
        snippet = text[:MAX_SNIPPET_CHARS]
        source_record_index = record.get("source_record_index")
        stable_record_index = (
            int(source_record_index) if isinstance(source_record_index, int) else -1
        )
        ranked.append(
            (
                (
                    score,
                    weighted_coverage,
                    density,
                    stable_record_index,
                    str(entry.get("raw_file") or ""),
                ),
                {
                    "provisional_id": entry["provisional_id"],
                    "raw_file": entry["raw_file"],
                    "score": round(score, 4),
                    "rank_cap": RANK_CAP,
                    "matched_terms": sorted(overlap),
                    "weighted_coverage": round(weighted_coverage, 4),
                    "match_density": round(density, 4),
                    "ranking_basis": "idf_weighted_coverage+density+phrase",
                    "snippet": snippet,
                    "citation": (
                        f"raw:{entry['raw_file']}#record="
                        f"{record.get('source_record_index')}"
                    ),
                    "unintegrated": True,
                    "content_is_untrusted": True,
                    "mutation_evidence_allowed": False,
                    "prompt_injection_treatment": "quote_only_never_instructions",
                    "host_boundary": entry.get("host_boundary"),
                    "source_host": entry.get("source_host"),
                },
            )
        )
    ranked.sort(
        key=lambda item: (
            -item[0][0],
            -item[0][1],
            -item[0][2],
            -item[0][3],
            item[0][4],
            str(item[1]["provisional_id"]),
        )
    )
    return [row for _rank, row in ranked[: max(0, min(MAX_HITS, int(limit)))]]


def snapshot(*, chronovisor_root: Path) -> dict[str, Any]:
    path = chronovisor_root / "runtime" / "provisional-recall" / "index.json"
    try:
        index = read_sealed_json(path)
    except DurableStateError:
        return {"status": "unavailable", "entries": 0}
    entries = index.get("entries")
    return {
        "status": "ok",
        "entries": len(entries) if isinstance(entries, list) else 0,
        "namespace": index.get("namespace"),
        "rank_cap": index.get("rank_cap"),
        "mutation_evidence_allowed": False,
    }
