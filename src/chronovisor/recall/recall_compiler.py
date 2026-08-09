"""Shadow meaning-address compiler over deterministic page claims."""

from __future__ import annotations

import hashlib
import json
import os
import re
from pathlib import Path
from typing import Any

from chronovisor.core.durable_state import exclusive_text_file_lock
from chronovisor.core.store import CHRONOVISOR_ROOT, find_page
from chronovisor.search.claims import CLAIM_INDEX_FILE
from chronovisor.search.search_types import tokenize

_STRUCTURED_INTENT_RE = re.compile(
    r"(?:価格|値段|容量|状態|ステータス|モデル|何GB|何TB|何%|"
    r"price|capacity|status|model|how many)",
    re.IGNORECASE,
)
_YEAR_RE = re.compile(r"(?:19|20)\d{2}")
SHADOW_TRACE_FILE = (
    CHRONOVISOR_ROOT / "runtime" / "recall-compiler" / "shadow-trace.jsonl"
)


def _intent_predicates(query: str) -> set[str]:
    folded = query.casefold()
    predicates: set[str] = set()
    if re.search(r"価格|値段|price|cost", folded):
        predicates.add("fact.price")
    if re.search(r"容量|何gb|何tb|capacity|memory size", folded):
        predicates.add("fact.capacity")
    if re.search(r"状態|ステータス|status", folded):
        predicates.add("fact.status")
    if re.search(r"モデル|model", folded):
        predicates.add("fact.model")
    if re.search(r"何%|how many|件数|数量", folded):
        predicates.update({"fact.ratio", "fact.quantity"})
    return predicates


def _meaning_address_exact(query: str, row: dict[str, Any]) -> bool:
    folded = query.casefold()
    subject = str(row.get("subject") or "").strip().casefold()
    page_key = (
        str(row.get("source_page") or "")
        .replace("-", " ")
        .replace("_", " ")
        .strip()
        .casefold()
    )
    subject_exact = len(subject) >= 3 and subject in folded
    page_exact = len(page_key) >= 5 and page_key in folded
    return subject_exact or page_exact


def _read_claims(path: Path) -> list[dict[str, Any]]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return []
    rows: list[dict[str, Any]] = []
    for line in lines:
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(row, dict):
            rows.append(row)
    return rows


def _source_valid(row: dict[str, Any]) -> bool:
    page_id = str(row.get("source_page") or "")
    path = find_page(page_id) if page_id else None
    if path is None:
        return False
    expected = str(row.get("source_sha256") or "")
    if not expected:
        return False
    try:
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError:
        return False
    return actual == expected


def compile_query(
    query: str,
    *,
    claims_path: Path = CLAIM_INDEX_FILE,
    limit: int = 10,
) -> dict[str, Any]:
    """Resolve only exact structured claims; otherwise require full retrieval."""

    if not _STRUCTURED_INTENT_RE.search(query):
        return {"status": "fallback", "reason": "unstructured_query", "page_ids": []}
    allowed_predicates = _intent_predicates(query)
    if not allowed_predicates:
        return {"status": "fallback", "reason": "untyped_intent", "page_ids": []}
    query_tokens = set(tokenize(query))
    years = set(_YEAR_RE.findall(query))
    candidates: list[tuple[int, dict[str, Any]]] = []
    for row in _read_claims(claims_path):
        if row.get("status") in {"superseded", "expired"} or row.get("valid_to"):
            continue
        page_id = str(row.get("source_page") or "")
        subject = str(row.get("subject") or "")
        predicate = str(row.get("predicate") or "")
        value = str(row.get("value") or "")
        slot = str(row.get("semantic_slot") or "")
        if predicate not in allowed_predicates or not _meaning_address_exact(
            query, row
        ):
            continue
        address_tokens = set(tokenize(" ".join([page_id, subject, slot])))
        overlap = len(query_tokens & address_tokens)
        if overlap <= 0:
            continue
        valid_from = str(row.get("valid_from") or "")
        if years and not any(year in valid_from for year in years):
            continue
        score = (overlap * 4) + len(query_tokens & set(tokenize(value)))
        candidates.append((score, row))
    candidates.sort(
        key=lambda item: (
            -item[0],
            str(item[1].get("source_page") or ""),
            str(item[1].get("claim_id") or ""),
        )
    )
    top = [row for _score, row in candidates[: max(1, limit)]]
    if not top:
        return {"status": "fallback", "reason": "no_meaning_address", "page_ids": []}
    grouped: dict[tuple[str, str, str], set[str]] = {}
    for row in top:
        key = (
            str(row.get("subject") or ""),
            str(row.get("predicate") or ""),
            str(row.get("semantic_slot") or ""),
        )
        grouped.setdefault(key, set()).add(str(row.get("value") or ""))
    if any(len(values) > 1 for values in grouped.values()):
        return {
            "status": "fallback",
            "reason": "active_claim_conflict",
            "page_ids": [],
        }
    valid = [row for row in top if _source_valid(row)]
    if not valid:
        return {
            "status": "fallback",
            "reason": "source_digest_unverified",
            "page_ids": [],
        }
    page_ids = list(
        dict.fromkeys(str(row.get("source_page") or "") for row in valid)
    )
    return {
        "status": "exact",
        "reason": "verified_meaning_address",
        "page_ids": page_ids,
        "claims": [
            {
                "claim_id": str(row.get("claim_id") or ""),
                "source_page": str(row.get("source_page") or ""),
                "predicate": str(row.get("predicate") or ""),
                "source_line": int(row.get("source_line") or 0),
                "source_sha256": str(row.get("source_sha256") or ""),
            }
            for row in valid
        ],
        "coverage": len(valid),
        "authority": "shadow",
    }


def append_shadow_trace(
    *,
    prompt: str,
    compiler: dict[str, Any],
    teacher_page_ids: list[str],
    committed_page_ids: list[str],
    path: Path = SHADOW_TRACE_FILE,
) -> dict[str, Any]:
    """Persist fast/teacher disagreement without raw prompt or claim values."""

    compiler_ids = list(
        dict.fromkeys(
            str(value)
            for value in compiler.get("page_ids", [])
            if isinstance(value, str) and value
        )
    )
    teacher_ids = list(dict.fromkeys(teacher_page_ids))
    committed_ids = list(dict.fromkeys(committed_page_ids))
    record = {
        "schema_version": 1,
        "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
        "compiler_status": str(compiler.get("status") or ""),
        "compiler_reason": str(compiler.get("reason") or ""),
        "compiler_page_ids": compiler_ids,
        "teacher_page_ids": teacher_ids,
        "committed_page_ids": committed_ids,
        "teacher_overlap": len(set(compiler_ids) & set(teacher_ids)),
        "commit_overlap": len(set(compiler_ids) & set(committed_ids)),
        "authority": "teacher",
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    lock_path = path.with_suffix(path.suffix + ".lock")
    with exclusive_text_file_lock(lock_path):
        with path.open("a", encoding="utf-8") as handle:
            handle.write(
                json.dumps(
                    record,
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                )
                + "\n"
            )
    os.chmod(path, 0o600)
    return record
