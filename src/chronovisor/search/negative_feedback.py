"""Query-conditioned negative feedback penalties for search ranking.

Recall feedback already records pages that were injected for a prompt and then
ignored (``injection_ignored``), explicitly flagged (``false-positive``), or
rejected individually (``page_ignored``).
The recall hook consumes those entries as injection suppressions, but the
search ranking itself never learned from them, so the same pages keep
surfacing in the top-20 for the same kind of vague prompts.

This module turns those feedback entries into fusion-stage penalties: when an
incoming query is lexically similar to a feedback prompt, the pages recorded
on that entry are demoted (never removed) in the fused ranking. Similarity is
Jaccard over the shared search tokenizer, so the penalty stays conditioned on
the query and does not suppress a page globally.
"""

from __future__ import annotations

import hashlib
import json
import os
import threading
from collections import Counter
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path

from chronovisor.core import feedback_ledger
from chronovisor.core.feedback_ledger import trusted_negative_feedback_rows
from chronovisor.core.link_fix import atomic_write
from chronovisor.core.page_mutation import find_mutation_page
from chronovisor.core.runtime_config import (
    NegativeFeedbackConfig,
    load_negative_feedback_config,
)
from chronovisor.core.search_types import ScoredPage, tokenize
from chronovisor.core.store import CHRONOVISOR_ROOT

# Test seams: when set, bypass the recall_runtime/golden default paths.
FEEDBACK_FILE_OVERRIDE: Path | None = None
GOLDEN_FILE_OVERRIDE: Path | None = None
RECALL_LOG_FILE_OVERRIDE: Path | None = None
PERSISTENT_CACHE_FILE = (
    CHRONOVISOR_ROOT / "runtime" / "search" / "negative-feedback-cache.json"
)
PERSISTENT_CACHE_SCHEMA = 1


@dataclass(frozen=True)
class _FeedbackEntry:
    tokens: frozenset[str]
    pages: tuple[str, ...]
    ts: datetime | None = None
    kind: str = ""
    frontier_confirmed: bool = False
    page_hashes: tuple[tuple[str, str], ...] = ()
    label_quality: str = ""


@dataclass
class _Cache:
    key: tuple | None = None
    entries: list[_FeedbackEntry] = field(default_factory=list)


_CACHE = _Cache()
_PROTECT_CACHE = _Cache()
_GOLD_NEGATIVE_CACHE = _Cache()
_CACHE_LOCK = threading.Lock()


def _feedback_file() -> Path:
    if FEEDBACK_FILE_OVERRIDE is not None:
        return FEEDBACK_FILE_OVERRIDE
    return feedback_ledger.RECALL_DIR / "feedback.jsonl"


def _golden_file() -> Path:
    if GOLDEN_FILE_OVERRIDE is not None:
        return GOLDEN_FILE_OVERRIDE
    return feedback_ledger.RECALL_DIR / "search-golden.jsonl"


def _recall_log_file() -> Path:
    if RECALL_LOG_FILE_OVERRIDE is not None:
        return RECALL_LOG_FILE_OVERRIDE
    return feedback_ledger.RECALL_DIR / "recall-log.jsonl"


def _tokenize(text: str) -> frozenset[str]:
    return frozenset(tokenize(text))


def _parse_ts(value: object) -> datetime | None:
    if not isinstance(value, str) or not value:
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)


def _ts_rank(value: datetime | None) -> float:
    """Return a stable ordering rank; undated legacy evidence is oldest."""

    return value.timestamp() if value is not None else float("-inf")


def _persistent_key(
    path: Path,
    *,
    mtime_ns: int,
    size: int,
    config: NegativeFeedbackConfig,
    recall_log_file: Path,
) -> dict[str, object]:
    try:
        recall_stat = recall_log_file.stat()
    except OSError:
        recall_mtime_ns, recall_size = 0, 0
    else:
        recall_mtime_ns, recall_size = recall_stat.st_mtime_ns, recall_stat.st_size
    return {
        "path": str(path),
        "mtime_ns": mtime_ns,
        "size": size,
        "recall_log_path": str(recall_log_file),
        "recall_log_mtime_ns": recall_mtime_ns,
        "recall_log_size": recall_size,
        "kinds": list(config.kinds),
        "max_age_days": config.max_age_days,
        "max_entries": config.max_entries,
        "utc_date": (
            datetime.now(UTC).date().isoformat() if config.max_age_days > 0 else ""
        ),
    }


def _entry_payload(entry: _FeedbackEntry) -> dict[str, object]:
    return {
        "tokens": sorted(entry.tokens),
        "pages": list(entry.pages),
        "ts": entry.ts.isoformat() if entry.ts is not None else None,
        "kind": entry.kind,
        "frontier_confirmed": entry.frontier_confirmed,
        "page_hashes": [list(item) for item in entry.page_hashes],
        "label_quality": entry.label_quality,
    }


def _entry_from_payload(value: object) -> _FeedbackEntry | None:
    if not isinstance(value, dict):
        return None
    tokens = value.get("tokens")
    pages = value.get("pages")
    hashes = value.get("page_hashes")
    if not isinstance(tokens, list) or not isinstance(pages, list):
        return None
    page_hashes: list[tuple[str, str]] = []
    if isinstance(hashes, list):
        for item in hashes:
            if (
                isinstance(item, list)
                and len(item) == 2
                and all(isinstance(part, str) for part in item)
            ):
                page_hashes.append((item[0], item[1]))
    return _FeedbackEntry(
        tokens=frozenset(str(token) for token in tokens if isinstance(token, str)),
        pages=tuple(str(page) for page in pages if isinstance(page, str)),
        ts=_parse_ts(value.get("ts")),
        kind=str(value.get("kind") or ""),
        frontier_confirmed=value.get("frontier_confirmed") is True,
        page_hashes=tuple(page_hashes),
        label_quality=str(value.get("label_quality") or ""),
    )


def _read_persistent_entries(key: dict[str, object]) -> list[_FeedbackEntry] | None:
    if FEEDBACK_FILE_OVERRIDE is not None:
        return None
    try:
        payload = json.loads(PERSISTENT_CACHE_FILE.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if (
        not isinstance(payload, dict)
        or payload.get("schema_version") != PERSISTENT_CACHE_SCHEMA
        or payload.get("source") != key
    ):
        return None
    raw_entries = payload.get("entries")
    if not isinstance(raw_entries, list):
        return None
    entries = [_entry_from_payload(value) for value in raw_entries]
    if any(entry is None for entry in entries):
        return None
    return [entry for entry in entries if entry is not None]


def _write_persistent_entries(
    key: dict[str, object],
    entries: list[_FeedbackEntry],
) -> None:
    if FEEDBACK_FILE_OVERRIDE is not None:
        return
    try:
        PERSISTENT_CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
        atomic_write(
            PERSISTENT_CACHE_FILE,
            json.dumps(
                {
                    "schema_version": PERSISTENT_CACHE_SCHEMA,
                    "source": key,
                    "entries": [_entry_payload(entry) for entry in entries],
                },
                ensure_ascii=False,
                separators=(",", ":"),
                sort_keys=True,
            )
            + "\n",
        )
        os.chmod(PERSISTENT_CACHE_FILE, 0o600)
    except OSError:
        pass


def _load_entries(config: NegativeFeedbackConfig) -> list[_FeedbackEntry]:
    path = _feedback_file()
    try:
        stat = path.stat()
    except OSError:
        return []
    cache_key = (
        str(path),
        stat.st_mtime_ns,
        stat.st_size,
        config.kinds,
        config.max_age_days,
        config.max_entries,
        (datetime.now(UTC).date().isoformat() if config.max_age_days > 0 else ""),
    )
    recall_log_file = _recall_log_file()
    try:
        recall_stat = recall_log_file.stat()
        recall_identity = (recall_stat.st_mtime_ns, recall_stat.st_size)
    except OSError:
        recall_identity = (0, 0)
    cache_key = (*cache_key, str(recall_log_file), *recall_identity)
    with _CACHE_LOCK:
        if _CACHE.key == cache_key:
            return _CACHE.entries
    persistent_key = _persistent_key(
        path,
        mtime_ns=stat.st_mtime_ns,
        size=stat.st_size,
        config=config,
        recall_log_file=recall_log_file,
    )
    persisted = _read_persistent_entries(persistent_key)
    if persisted is not None:
        with _CACHE_LOCK:
            _CACHE.key = cache_key
            _CACHE.entries = persisted
        return persisted

    cutoff = (
        datetime.now(UTC) - timedelta(days=config.max_age_days)
        if config.max_age_days > 0
        else None
    )
    entries: list[_FeedbackEntry] = []
    for row in trusted_negative_feedback_rows(
        path, recall_log_file=recall_log_file
    ):
        if not isinstance(row, dict) or row.get("kind") not in config.kinds:
            continue
        kind = str(row.get("kind") or "")
        # Exposure outcomes are not relevance labels. Only a reviewed false
        # positive or a hash-bound, strong contradiction may suppress ranking.
        # ignored/page_ignored/certificate rejects remain label-factory inputs,
        # never standalone production negatives.
        label_quality = str(row.get("label_quality") or "")
        reviewed_false_positive = (
            kind == "false-positive" and row.get("frontier_reviewed") is True
        )
        strong_contradiction = (
            kind in {"strong-contradiction", "contradiction", "page_ignored"}
            and row.get("frontier_reviewed") is True
            and label_quality == "strong"
            and isinstance(row.get("negative_page_hashes"), dict)
        )
        if not (reviewed_false_positive or strong_contradiction):
            continue
        prompt = row.get("prompt")
        negative_pages = row.get("negative_pages")
        pages = (
            negative_pages
            if isinstance(negative_pages, list) and negative_pages
            else None
        )
        if pages is None and row.get("kind") != "page_ignored":
            # Backward compatibility: the legacy prompt-scoped labels stored
            # injected pages in ``expected_pages``.
            pages = row.get("expected_pages")
        if not isinstance(prompt, str) or not prompt.strip():
            continue
        if not isinstance(pages, list):
            continue
        page_ids = tuple(p for p in pages if isinstance(p, str) and p)
        if not page_ids:
            continue
        if cutoff is not None:
            ts = _parse_ts(row.get("ts"))
            if ts is not None and ts < cutoff:
                continue
        tokens = _tokenize(prompt)
        if not tokens:
            continue
        page_hashes_value = row.get("negative_page_hashes")
        page_hashes = (
            tuple(
                sorted(
                    (str(page_id), str(digest))
                    for page_id, digest in page_hashes_value.items()
                    if isinstance(page_id, str) and isinstance(digest, str)
                )
            )
            if isinstance(page_hashes_value, dict)
            else ()
        )
        entries.append(
            _FeedbackEntry(
                tokens=tokens,
                pages=page_ids,
                ts=_parse_ts(row.get("ts")),
                kind=kind,
                frontier_confirmed=(row.get("frontier_reviewed") is True),
                page_hashes=page_hashes,
                label_quality=label_quality,
            )
        )

    # Keep the most recent entries when the file grows large.
    if len(entries) > config.max_entries:
        entries = entries[-config.max_entries :]

    with _CACHE_LOCK:
        _CACHE.key = cache_key
        _CACHE.entries = entries
    _write_persistent_entries(persistent_key, entries)
    return entries


def _load_protections() -> list[_FeedbackEntry]:
    """Reviewed positive labels veto auto-penalties for similar queries.

    ``injection_ignored`` only means "the assistant did not use the page",
    not "the page is irrelevant".  When a reviewer has since confirmed the
    page as relevant for a similar query, demoting it would reintroduce the
    very recall miss the review fixed, so those pages are protected.
    """
    path = _golden_file()
    try:
        mtime = path.stat().st_mtime
    except OSError:
        return []
    cache_key = (str(path), mtime)
    with _CACHE_LOCK:
        if _PROTECT_CACHE.key == cache_key:
            return _PROTECT_CACHE.entries

    entries: list[_FeedbackEntry] = []
    try:
        lines = path.read_text(encoding="utf-8").split("\n")
    except OSError:
        return []
    for line in lines:
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(row, dict) or row.get("reviewed") is not True:
            continue
        if str(row.get("split") or "train") != "train":
            continue
        query = row.get("query")
        pages = row.get("expected_pages")
        if (
            not isinstance(query, str)
            or not query.strip()
            or not isinstance(pages, list)
        ):
            continue
        page_ids = tuple(p for p in pages if isinstance(p, str) and p)
        if not page_ids:
            continue
        tokens = _tokenize(query)
        if not tokens:
            continue
        entries.append(
            _FeedbackEntry(
                tokens=tokens,
                pages=page_ids,
                ts=_parse_ts(row.get("ts") or row.get("reviewed_at")),
                kind="reviewed_positive",
            )
        )

    with _CACHE_LOCK:
        _PROTECT_CACHE.key = cache_key
        _PROTECT_CACHE.entries = entries
    return entries


def _load_reviewed_train_negatives() -> list[_FeedbackEntry]:
    """Load only reviewed train-split negatives; holdout/locked stay unseen."""

    path = _golden_file()
    try:
        stat = path.stat()
    except OSError:
        return []
    cache_key = (str(path), stat.st_mtime_ns, stat.st_size, "train-negatives")
    with _CACHE_LOCK:
        if _GOLD_NEGATIVE_CACHE.key == cache_key:
            return _GOLD_NEGATIVE_CACHE.entries
    entries: list[_FeedbackEntry] = []
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return []
    for line in lines:
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if (
            not isinstance(row, dict)
            or row.get("reviewed") is not True
            or str(row.get("split") or "train") != "train"
        ):
            continue
        query = row.get("query")
        pages = row.get("negative_pages")
        if (
            not isinstance(query, str)
            or not query.strip()
            or not isinstance(pages, list)
        ):
            continue
        page_ids = tuple(page for page in pages if isinstance(page, str) and page)
        tokens = _tokenize(query)
        if not page_ids or not tokens:
            continue
        entries.append(
            _FeedbackEntry(
                tokens=tokens,
                pages=page_ids,
                ts=_parse_ts(row.get("ts") or row.get("reviewed_at")),
                kind="explicit_eval_negative",
                frontier_confirmed=True,
                label_quality="gold",
            )
        )
    with _CACHE_LOCK:
        _GOLD_NEGATIVE_CACHE.key = cache_key
        _GOLD_NEGATIVE_CACHE.entries = entries
    return entries


def contextual_negative_trace(
    query: str,
    config: NegativeFeedbackConfig | None = None,
) -> dict[str, dict[str, object]]:
    """Build a query-conditioned centroid trace from strong negative evidence."""

    config = config or load_negative_feedback_config()
    if not config.enabled:
        return {}
    query_tokens = _tokenize(query)
    if not query_tokens:
        return {}
    matched_entries: dict[str, list[tuple[_FeedbackEntry, float]]] = {}
    # A frontier-confirmed page-scoped rejection is stronger than an older
    # reviewed positive label for the same query/page. Keep its newest
    # timestamp separately from penalty magnitude so a slightly different
    # wording cannot let stale golden data veto the newer correction.
    newest_confirmed_ignored: dict[str, datetime | None] = {}
    for entry in [*_load_entries(config), *_load_reviewed_train_negatives()]:
        union = len(query_tokens | entry.tokens)
        if union == 0:
            continue
        jaccard = len(query_tokens & entry.tokens) / union
        if jaccard < config.similarity_threshold:
            continue
        expected_hashes = dict(entry.page_hashes)
        for page_id in entry.pages:
            expected_hash = expected_hashes.get(page_id)
            if expected_hash:
                path = find_mutation_page(page_id)
                try:
                    current_hash = (
                        hashlib.sha256(path.read_bytes()).hexdigest()
                        if path is not None
                        else ""
                    )
                except OSError:
                    current_hash = ""
                if current_hash != expected_hash:
                    continue
            matched_entries.setdefault(page_id, []).append((entry, jaccard))
            if entry.frontier_confirmed:
                previous = newest_confirmed_ignored.get(page_id)
                if page_id not in newest_confirmed_ignored or _ts_rank(
                    entry.ts
                ) > _ts_rank(previous):
                    newest_confirmed_ignored[page_id] = entry.ts
    if not matched_entries:
        return {}

    trace: dict[str, dict[str, object]] = {}
    for page_id, matches in matched_entries.items():
        token_weights: Counter[str] = Counter()
        strongest_jaccard = 0.0
        newest_ts: datetime | None = None
        evidence_kinds: set[str] = set()
        for entry, jaccard in matches:
            strongest_jaccard = max(strongest_jaccard, jaccard)
            evidence_kinds.add(entry.kind)
            if _ts_rank(entry.ts) > _ts_rank(newest_ts):
                newest_ts = entry.ts
            for token in entry.tokens:
                token_weights[token] += jaccard
        centroid_tokens = frozenset(
            token for token, _weight in token_weights.most_common(48)
        )
        centroid_union = len(query_tokens | centroid_tokens)
        centroid_similarity = (
            len(query_tokens & centroid_tokens) / centroid_union
            if centroid_union
            else 0.0
        )
        similarity = max(strongest_jaccard, centroid_similarity)
        trace[page_id] = {
            "penalty": min(config.penalty, config.penalty * similarity),
            "similarity": round(similarity, 6),
            "centroid_terms": len(centroid_tokens),
            "evidence_count": len(matches),
            "evidence_kinds": sorted(evidence_kinds),
            "newest_ts": newest_ts,
        }

    protected: dict[str, tuple[float, datetime | None]] = {}
    for entry in _load_protections():
        if not set(entry.pages) & set(trace):
            continue
        union = len(query_tokens | entry.tokens)
        if union == 0:
            continue
        jaccard = len(query_tokens & entry.tokens) / union
        if jaccard < config.similarity_threshold:
            continue
        for page_id in entry.pages:
            current = protected.get(page_id)
            if (
                current is None
                or jaccard > current[0]
                or (jaccard == current[0] and _ts_rank(entry.ts) > _ts_rank(current[1]))
            ):
                protected[page_id] = (jaccard, entry.ts)
    for page_id, (pos_jaccard, positive_ts) in protected.items():
        row = trace.get(page_id)
        penalty = float(row.get("penalty") or 0.0) if row else None
        confirmed_ts = newest_confirmed_ignored.get(page_id)
        confirmed_is_newer = page_id in newest_confirmed_ignored and _ts_rank(
            confirmed_ts
        ) >= _ts_rank(positive_ts)
        if (
            penalty is not None
            and not confirmed_is_newer
            and pos_jaccard * config.penalty >= penalty
        ):
            del trace[page_id]
    return trace


def penalties_for_query(
    query: str,
    config: NegativeFeedbackConfig | None = None,
) -> dict[str, float]:
    """Return strong contextual penalties only; never infer from exposure."""

    return {
        page_id: float(row["penalty"])
        for page_id, row in contextual_negative_trace(query, config).items()
    }


def apply_penalties(results, penalties: dict[str, float]):
    """Demote penalized pages by scaling their fused score, then re-sort."""
    if not penalties:
        return results

    adjusted = []
    for page in results:
        penalty = penalties.get(page.page_id, 0.0)
        if penalty <= 0.0:
            adjusted.append(page)
            continue
        adjusted.append(
            ScoredPage(
                page_id=page.page_id,
                title=page.title,
                folder=page.folder,
                updated=page.updated,
                score=float(page.score) * max(0.0, 1.0 - penalty),
                status=page.status,
                superseded_by=page.superseded_by,
                page_type=page.page_type,
                sensitivity=page.sensitivity,
            )
        )
    adjusted.sort(key=lambda x: x.score, reverse=True)
    return adjusted
