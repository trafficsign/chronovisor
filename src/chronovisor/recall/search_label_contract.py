"""Canonical search-label and golden-corpus contract owned by Recall."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from chronovisor.core.canonical_json import (
    canonical_json_sha256_stringifying as _canonical_json_sha256,
)
from chronovisor.core.store import CHRONOVISOR_ROOT, find_page
from chronovisor.decision import decision_authority
from chronovisor.decision.semantic_hold import canonical_sha256

RECALL_DIR = CHRONOVISOR_ROOT / "recall"
GOLDEN_FILE = RECALL_DIR / "search-golden.jsonl"
BASELINE_DIR = CHRONOVISOR_ROOT / "runtime" / "search-eval"
MANUAL_MANIFEST_FILE = BASELINE_DIR / "manual-94-manifest.json"
SEALED_MANIFEST_SCHEMA_VERSION = 2
SEARCH_LABEL_LANE = "search_label"
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


def _str_tuple(value: Any) -> tuple[str, ...]:
    if not isinstance(value, list):
        return ()
    return tuple(
        dict.fromkeys(item for item in value if isinstance(item, str) and item)
    )


def _str_list(value: Any) -> list[str]:
    return list(_str_tuple(value))


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
            "projection_policy_sha256": str(row.get("projection_policy_sha256") or ""),
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
        "projection_policy_sha256": str(row.get("projection_policy_sha256") or ""),
        "search_eval_split": str(row.get("split") or ""),
    }
    candidate_sha = _canonical_json_sha256(identity)
    try:
        preregistered = datetime.fromisoformat(
            str(row.get("preregistered_at") or "").replace("Z", "+00:00")
        )
        preregistered_valid = (
            preregistered.tzinfo is not None and preregistered.utcoffset() is not None
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


label_review_claim_error = _label_review_claim_error


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


search_review_artifact_error = _search_review_artifact_error


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
        if (
            source
            in {
                "recall_questions",
                "auto",
                "generated",
            }
            and authoritative_search_label_error(row) is not None
        ):
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


def _label_tuple_from_review(
    review: dict[str, Any],
) -> tuple[tuple[str, ...], tuple[str, ...], tuple[str, ...]]:
    return (
        tuple(_str_list(review.get("expected_pages"))),
        tuple(_str_list(review.get("negative_pages"))),
        tuple(_str_list(review.get("stale_pages"))),
    )


label_tuple_from_review = _label_tuple_from_review
