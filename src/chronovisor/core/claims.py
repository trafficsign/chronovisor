"""Append-only claim ledger seed for future event-sourced memory."""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Iterator
from datetime import date, datetime
from pathlib import Path
from typing import Any

from chronovisor.core.frontmatter import parse
from chronovisor.core.index_store import get_store
from chronovisor.core.store import CHRONOVISOR_ROOT, find_page

CLAIMS_DIR = CHRONOVISOR_ROOT / "claims"
CLAIMS_FILE = CLAIMS_DIR / "claims.jsonl"
CLAIM_INDEX_FILE = CLAIMS_DIR / "claims-index.jsonl"
CLAIM_CONFLICT_FILE = CLAIMS_DIR / "claim-conflicts.jsonl"
CLAIM_REVIEW_FILE = CLAIMS_DIR / "claim-conflict-reviews.jsonl"

FACT_VALUE_RE = re.compile(
    r"(?:[¥￥$]\s*)?\d[\d,]*(?:\.\d+)?\s*(?:円|万円|GB|GP|TB|枚|台|件|個|回|人|%)",
    re.IGNORECASE,
)
MODEL_VALUE_RE = re.compile(r"\b[A-Z][A-Z0-9-]{2,}(?:\s*\d+(?:GB|GP|TB))?\b")
STATUS_VALUE_RE = re.compile(r"(?:未確定|確定|予定|到着済み|設置済み|完了|保留|却下|廃止|有効|無効)")
DATE_HEADING_RE = re.compile(r"(?:19|20)\d{2}[-/.年]\d{1,2}(?:[-/.月]\d{1,2}日?)?")
SLOT_LABEL_RE = re.compile(
    r"^(?:[-*+]\s*)?(?:\d+[.)]\s*)?(?:\*\*)?([^:：|]{2,80}?)(?:\*\*)?\s*[:：]"
)


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False, sort_keys=True, default=str) + "\n")


def claim_from_page(page_id: str, *, source_raw: str = "", op: str = "upsert") -> dict[str, Any] | None:
    path = find_page(page_id)
    if path is None:
        return None
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        return None
    meta, _body = parse(text)
    title = meta.get("title")
    summary = meta.get("summary")
    if not isinstance(summary, str) or not summary.strip():
        summary = title if isinstance(title, str) else page_id
    entities = meta.get("entities")
    page_type = meta.get("type") if isinstance(meta.get("type"), str) else "knowledge"
    return {
        "claim_id": f"{datetime.now().strftime('%Y%m%dT%H%M%S%f')}-{page_id}",
        "recorded_at": datetime.now().isoformat(timespec="seconds"),
        "subject": page_id,
        "predicate": "page.summary",
        "value": summary,
        "source_page": page_id,
        "source_raw": source_raw,
        "op": op,
        "page_type": page_type,
        "entities": entities if isinstance(entities, list) else [],
        "valid_from": str(meta.get("updated") or date.today().isoformat()),
        "valid_to": None,
    }


def append_page_claims(page_ids: list[str], *, source_raw: str = "", op: str = "upsert") -> dict[str, Any]:
    if not source_raw.strip():
        return {
            "status": "skipped",
            "reason": "source_raw required for append-only claim ledger",
            "claims_file": str(CLAIMS_FILE),
            "written": 0,
            "skipped": list(page_ids),
        }
    skipped, pending = list[str](), list[dict[str, Any]]()
    for page_id in page_ids:
        rows = page_claims(page_id, source_raw=source_raw, op=op)
        if rows:
            pending.extend(rows)
        else:
            skipped.append(page_id)
    if pending:
        with _claims_ledger_lock(CLAIMS_FILE):
            _append_claim_rows(CLAIMS_FILE, pending)
    return {
        "status": "ok",
        "claims_file": str(CLAIMS_FILE),
        "written": len(pending),
        "skipped": skipped,
    }


def page_claims(page_id: str, *, source_raw: str = "", op: str = "index") -> list[dict[str, Any]]:
    path = find_page(page_id)
    if path is None:
        return []
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        return []
    meta, body = parse(text)
    source_sha256 = hashlib.sha256(text.encode()).hexdigest()
    title = meta.get("title") if isinstance(meta.get("title"), str) else page_id
    summary_value = meta.get("summary")
    summary = summary_value if isinstance(summary_value, str) else ""
    updated = str(meta.get("updated") or date.today().isoformat())
    entities_value = meta.get("entities")
    entities = entities_value if isinstance(entities_value, list) else []
    page_type = meta.get("type") if isinstance(meta.get("type"), str) else "knowledge"
    status = meta.get("status") if isinstance(meta.get("status"), str) else "active"
    base = {
        "recorded_at": datetime.now().isoformat(timespec="seconds"),
        "source_page": page_id,
        "source_raw": source_raw,
        "op": op,
        "page_type": page_type,
        "entities": entities,
        "valid_from": updated,
        "valid_to": None if status == "active" else updated,
        "status": "active" if status == "active" else "expired",
        "source_sha256": source_sha256,
    }
    claims: list[dict[str, Any]] = [
        {
            **base,
            "claim_id": f"{page_id}:title",
            "subject": page_id,
            "predicate": "page.title",
            "value": title,
        }
    ]
    if summary.strip():
        claims.append(
            {
                **base,
                "claim_id": f"{page_id}:summary",
                "subject": page_id,
                "predicate": "page.summary",
                "value": summary.strip(),
            }
        )
    for entity in entities:
        if isinstance(entity, str) and entity.strip():
            claims.append(
                {
                    **base,
                    "claim_id": f"{page_id}:entity:{entity}",
                    "subject": page_id,
                    "predicate": "page.entity",
                    "value": entity.strip(),
                }
            )
    first_line = next((line.strip(" #-\t") for line in body.splitlines() if line.strip(" #-\t")), "")
    if first_line and first_line != summary:
        claims.append(
            {
                **base,
                "claim_id": f"{page_id}:body.lead",
                "subject": page_id,
                "predicate": "body.lead",
                "value": first_line[:280],
            }
        )
    claims.extend(_fact_claims(
        page_id=page_id,
        body=body,
        base=base,
        source_raw=source_raw or str(meta.get("source_raw") or meta.get("raw_source") or ""),
        default_date=updated,
    ))
    return claims


def _fact_claims(
    *,
    page_id: str,
    body: str,
    base: dict[str, Any],
    source_raw: str,
    default_date: str,
) -> list[dict[str, Any]]:
    """Extract high-value atomic facts while retaining exact evidence spans."""
    rows: list[dict[str, Any]] = []
    current_date = default_date
    page_subject = next((str(v) for v in base.get("entities", []) if isinstance(v, str) and v), page_id)
    for line_no, raw_line in enumerate(body.splitlines(), start=1):
        line = raw_line.strip(" #-\t")
        if not line:
            continue
        heading_date = DATE_HEADING_RE.search(line)
        if raw_line.lstrip().startswith("#") and heading_date:
            current_date = heading_date.group(0).replace("年", "-").replace("月", "-").replace("日", "").replace("/", ".")
        line_models = [match.group(0).strip() for match in MODEL_VALUE_RE.finditer(line)]
        line_subject = line_models[0] if len(line_models) == 1 else page_subject
        semantic_slot = _fact_slot(line)
        candidates: list[tuple[str, str, str]] = []
        for match in FACT_VALUE_RE.finditer(line):
            value = match.group(0).strip()
            if "%" in value:
                predicate = "fact.ratio"
            elif re.search(r"(?:円|万円|[¥￥$])", value):
                predicate = "fact.price"
            elif re.search(r"(?:GB|GP|TB)", value, re.IGNORECASE):
                predicate = "fact.capacity"
            else:
                predicate = "fact.quantity"
            candidates.append((predicate, value, line_subject))
        candidates.extend(("fact.model", value, page_subject) for value in line_models)
        candidates.extend(("fact.status", match.group(0).strip(), line_subject) for match in STATUS_VALUE_RE.finditer(line))
        seen: set[tuple[str, str]] = set()
        for predicate, value, subject in candidates:
            key = (predicate, value.casefold())
            if key in seen:
                continue
            seen.add(key)
            digest = hashlib.sha256(f"{page_id}:{line_no}:{predicate}:{value}".encode()).hexdigest()[:16]
            rows.append({
                **base,
                "claim_id": f"{page_id}:fact:{digest}",
                "subject": subject,
                "predicate": predicate,
                "value": value,
                "evidence_span": line[:500],
                "source_line": line_no,
                "source_raw": source_raw,
                "valid_from": current_date,
                "valid_to": None,
                "confidence": "deterministic_candidate",
                "semantic_slot": semantic_slot,
            })
    return rows


def _fact_slot(line: str) -> str:
    """Return an explicit property label; unlabeled prose is not conflict-safe."""
    match = SLOT_LABEL_RE.search(line.strip())
    if not match:
        return ""
    label = re.sub(r"[*_`\[\]()#]", "", match.group(1)).strip().casefold()
    label = re.sub(r"\s+", " ", label)
    return label[:80]


def claim_conflicts(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
    for row in rows:
        predicate = str(row.get("predicate") or "")
        if not predicate.startswith("fact.") or predicate == "fact.model":
            continue
        if row.get("status") == "superseded" or row.get("valid_to"):
            continue
        slot = str(row.get("semantic_slot") or "").strip()
        if not slot:
            continue
        key = (str(row.get("source_page") or ""), str(row.get("subject") or ""), f"{predicate}:{slot}")
        grouped.setdefault(key, []).append(row)
    conflicts: list[dict[str, Any]] = []
    for (page_id, subject, predicate_slot), claims_ in grouped.items():
        predicate, slot = predicate_slot.split(":", 1)
        values = {str(row.get("value") or "").casefold() for row in claims_}
        source_lines = {row.get("source_line") for row in claims_}
        if len(values) < 2 or len(source_lines) < 2:
            continue
        conflict_id = hashlib.sha256(
            json.dumps(sorted(str(row.get("claim_id")) for row in claims_)).encode("utf-8")
        ).hexdigest()
        conflicts.append({
            "conflict_id": conflict_id,
            "page_id": page_id,
            "subject": subject,
            "predicate": predicate,
            "semantic_slot": slot,
            "claims": claims_,
            "status": "preserve_conflict",
        })
    return conflicts


def rebuild_claim_index(*, limit: int = 0, path: Path = CLAIM_INDEX_FILE, write: bool = True) -> dict[str, Any]:
    store = get_store()
    store.refresh()
    metas = [meta for meta in store.all_pages_meta(include_system=False) if meta.get("page_type") != "reference"]
    if limit:
        metas = metas[:limit]
    rows: list[dict[str, Any]] = []
    for meta in metas:
        page_id = str(meta.get("page_id") or "")
        if page_id:
            rows.extend(page_claims(page_id))
    review_state = _reviewed_claim_state()
    for row in rows:
        state = review_state.get(str(row.get("claim_id") or ""))
        if state:
            row.update(state)
    if write:
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = "".join(
            json.dumps(row, ensure_ascii=False, sort_keys=True, default=str) + "\n"
            for row in rows
        )
        path.write_text(payload, encoding="utf-8")
        conflicts = claim_conflicts(rows)
        CLAIM_CONFLICT_FILE.write_text(
            "".join(json.dumps(row, ensure_ascii=False, sort_keys=True, default=str) + "\n" for row in conflicts),
            encoding="utf-8",
        )
    else:
        conflicts = claim_conflicts(rows)
    return {"status": "ok", "path": str(path), "pages": len(metas), "claims": len(rows), "conflicts": len(conflicts), "write": write}


def _reviewed_claim_state() -> dict[str, dict[str, Any]]:
    """Materialize only explicit user corrections into the derived view.

    Model consensus may classify a conflict, but it is not authority to erase
    either provenance branch.  Destructive supersession requires an explicit
    user correction artifact.
    """
    from chronovisor.core.jsonl import read_jsonl

    state: dict[str, dict[str, Any]] = {}
    for result in read_jsonl(CLAIM_REVIEW_FILE):
        review = result.get("review")
        if (
            not result.get("valid")
            or result.get("authority") != "user"
            or not isinstance(review, dict)
            or review.get("decision") != "approved"
        ):
            continue
        reviewed_at = str(result.get("reviewed_at") or datetime.now().isoformat(timespec="seconds"))
        for claim_id in review.get("invalidated_claim_ids") or []:
            if isinstance(claim_id, str) and claim_id:
                state[claim_id] = {
                    "status": "superseded",
                    "valid_to": reviewed_at,
                    "superseded_by_review": result.get("conflict_id"),
                }
        for claim_id in review.get("preferred_claim_ids") or []:
            if isinstance(claim_id, str) and claim_id and claim_id not in state:
                state[claim_id] = {"status": "active"}
    return state


def _claim_tokens(row: dict[str, Any]) -> set[str]:
    text = " ".join(str(row.get(key) or "") for key in ("subject", "predicate", "value", "entities"))
    return {
        token.lower()
        for token in re.findall(r"[a-z0-9_.+-]{2,}|[\u3040-\u30ff\u3400-\u9fff]{2,}", text.lower())
    }


def search_claims(query: str, *, limit: int = 10, path: Path = CLAIM_INDEX_FILE) -> list[dict[str, Any]]:
    query_tokens = {
        token.lower()
        for token in re.findall(r"[a-z0-9_.+-]{2,}|[\u3040-\u30ff\u3400-\u9fff]{2,}", query.lower())
    }
    if not query_tokens:
        return []
    try:
        lines = [line for line in path.read_text(encoding="utf-8").split("\n") if line.strip()]
    except OSError:
        return []
    scored: list[tuple[int, dict[str, Any]]] = []
    for line in lines:
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(row, dict):
            continue
        if row.get("status") == "superseded" or row.get("valid_to"):
            continue
        score = len(query_tokens & _claim_tokens(row))
        if score:
            scored.append((score, row))
    scored.sort(key=lambda item: item[0], reverse=True)
    return [{**row, "score": score} for score, row in scored[:limit]]


def _is_placeholder_claim(row: dict[str, Any]) -> bool:
    source_raw = row.get("source_raw")
    source_page = row.get("source_page")
    value = str(row.get("value") or "").strip().lower()
    if not isinstance(source_raw, str) or not source_raw.strip():
        return True
    if not isinstance(source_page, str) or not source_page.strip():
        return True
    if re.fullmatch(r"p\d*|foo|bar|baz", source_page.strip()):
        return True
    if value in {"", "body", "test", "placeholder"}:
        return True
    return find_page(source_page) is None


def _sanitize_claim_ledger_unlocked(
    *, path: Path = CLAIMS_FILE, write: bool = True
) -> dict[str, Any]:
    try:
        lines = [line for line in path.read_text(encoding="utf-8").split("\n") if line.strip()]
    except OSError:
        return {"status": "missing", "path": str(path), "kept": 0, "dropped": 0, "write": write}
    kept: list[dict[str, Any]] = []
    dropped = 0
    for line in lines:
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            dropped += 1
            continue
        if not isinstance(row, dict) or _is_placeholder_claim(row):
            dropped += 1
            continue
        kept.append(row)
    if write:
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = "".join(json.dumps(row, ensure_ascii=False, sort_keys=True, default=str) + "\n" for row in kept)
        path.write_text(payload, encoding="utf-8")
    return {"status": "ok", "path": str(path), "kept": len(kept), "dropped": dropped, "write": write}


def sanitize_claim_ledger(
    *, path: Path = CLAIMS_FILE, write: bool = True
) -> dict[str, Any]:
    if not write:
        return _sanitize_claim_ledger_unlocked(path=path, write=False)
    with _claims_ledger_lock(path):
        return _sanitize_claim_ledger_unlocked(path=path, write=True)


def _claims_ledger_lock(path: Path) -> Any:
    import contextlib
    import fcntl
    import os

    @contextlib.contextmanager
    def locked() -> Iterator[None]:
        lock_path = path.with_suffix(path.suffix + ".lock")
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        with lock_path.open("a+", encoding="utf-8") as handle:
            os.chmod(lock_path, 0o600)
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)

    return locked()


def _append_claim_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    for row in rows:
        append_jsonl(path, row)
