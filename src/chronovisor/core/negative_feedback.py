"""Query-conditioned negative feedback penalties for core search ranking.

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
from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, cast

from chronovisor.core import feedback_ledger
from chronovisor.core.feedback_ledger import trusted_negative_feedback_rows
from chronovisor.core.link_fix import atomic_write
from chronovisor.core.page_mutation import find_mutation_page
from chronovisor.core.runtime_config import (
    NegativeFeedbackConfig,
    load_negative_feedback_config,
)
from chronovisor.core.search_types import TOKENIZER_VERSION, ScoredPage, tokenize
from chronovisor.core.store import CHRONOVISOR_ROOT

# Test seams: when set, bypass the recall_runtime/golden default paths.
FEEDBACK_FILE_OVERRIDE: Path | None = None
GOLDEN_FILE_OVERRIDE: Path | None = None
RECALL_LOG_FILE_OVERRIDE: Path | None = None
PERSISTENT_CACHE_FILE = (
    CHRONOVISOR_ROOT / "runtime" / "search" / "negative-feedback-cache.json"
)
PERSISTENT_CACHE_SCHEMA = 2
GOLDEN_CACHE_FILE = CHRONOVISOR_ROOT / "runtime" / "search" / "golden-feedback-cache.json"
# Schema 2 stores the source stat signature and tail-append state beside the
# content hash, so a hook process can skip re-hashing an unchanged golden file.
GOLDEN_CACHE_SCHEMA = 2


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
    key: tuple[object, ...] | None = None
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
) -> dict[str, object]:
    return {
        "path": str(path),
        "mtime_ns": mtime_ns,
        "size": size,
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
    if not isinstance(payload, dict) or payload.get("schema_version") != PERSISTENT_CACHE_SCHEMA:
        return None
    previous = payload.get("source")
    if not isinstance(previous, dict) or any(
        previous.get(name) != value
        for name, value in key.items()
        if name not in {"mtime_ns", "size"}
    ):
        return None
    old_size = previous.get("size")
    if not isinstance(old_size, int) or isinstance(old_size, bool) or old_size < 0:
        return None
    path = Path(str(key["path"]))
    try:
        signature = _golden_signature(path)
        if signature[2:4] != (key["size"], key["mtime_ns"]) or old_size > signature[2]:
            return None
        with path.open("rb") as stream:
            digest = hashlib.sha256()
            remaining = old_size
            last = b""
            while remaining:
                chunk = stream.read(min(1024 * 1024, remaining))
                if not chunk:
                    return None
                digest.update(chunk)
                remaining -= len(chunk)
                last = chunk[-1:]
            if digest.hexdigest() != payload.get("source_sha256"):
                return None
            tail = stream.read()
        if tail:
            # A partial old/new row could become a target when completed later.
            if (old_size and last != b"\n") or not tail.endswith(b"\n"):
                return None
            targets = set(cast(list[str], key["kinds"])) | {
                "false-positive", "strong-contradiction", "contradiction",
                "page_ignored", feedback_ledger.PAGE_IGNORED_RETRACTION_KIND,
            }
            for line in tail.decode("utf-8").split("\n"):
                if not line.strip():
                    continue
                row = json.loads(line)
                if not isinstance(row, dict) or not isinstance(row.get("kind"), str):
                    return None
                if row["kind"] in targets:
                    return None
            digest.update(tail)
        if signature != _golden_signature(path):
            return None
    except (OSError, ValueError):
        return None
    raw_entries = payload.get("entries")
    if not isinstance(raw_entries, list):
        return None
    entries = [_entry_from_payload(value) for value in raw_entries]
    if any(entry is None for entry in entries):
        return None
    decoded = [entry for entry in entries if entry is not None]
    if previous != key:
        _write_persistent_entries(key, decoded, source_sha256=digest.hexdigest())
    return decoded


def _write_persistent_entries(
    key: dict[str, object],
    entries: list[_FeedbackEntry],
    *,
    source_sha256: str = "",
) -> None:
    if FEEDBACK_FILE_OVERRIDE is not None or os.environ.get("CHRONOVISOR_READ_ONLY") == "1":
        return
    try:
        PERSISTENT_CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
        atomic_write(
            PERSISTENT_CACHE_FILE,
            json.dumps(
                {
                    "schema_version": PERSISTENT_CACHE_SCHEMA,
                    "source": key,
                    "source_sha256": source_sha256,
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
    cache_key: tuple[object, ...] = (
        str(path),
        stat.st_mtime_ns,
        stat.st_size,
        stat.st_ctime_ns,
        stat.st_ino,
        config.kinds,
        config.max_age_days,
        config.max_entries,
        (datetime.now(UTC).date().isoformat() if config.max_age_days > 0 else ""),
    )
    recall_log_file = _recall_log_file()
    with _CACHE_LOCK:
        if _CACHE.key == cache_key:
            return _CACHE.entries
    persistent_key = _persistent_key(
        path,
        mtime_ns=stat.st_mtime_ns,
        size=stat.st_size,
        config=config,
    )
    persisted = _read_persistent_entries(persistent_key)
    if persisted is not None:
        with _CACHE_LOCK:
            _CACHE.key = cache_key
            _CACHE.entries = persisted
        return persisted

    # Bind a rebuilt cache to the exact source validated below, even if a writer races it.
    try:
        source_signature = _golden_signature(path)
        with path.open("rb") as stream:
            source_sha256 = hashlib.file_digest(stream, "sha256").hexdigest()
    except OSError:
        source_signature = ()
        source_sha256 = ""
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

    try:
        stable = (
            source_signature == _golden_signature(path)
            and source_signature[2:4] == (stat.st_size, stat.st_mtime_ns)
            and bool(source_sha256)
        )
    except OSError:
        stable = False
    if stable:
        with _CACHE_LOCK:
            _CACHE.key = cache_key
            _CACHE.entries = entries
        _write_persistent_entries(persistent_key, entries, source_sha256=source_sha256)
    return entries


def _golden_signature(path: Path) -> tuple[int, ...]:
    stat = path.stat()
    return (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns)


def _parse_golden_entries(text: str) -> list[dict[str, object]]:
    """Project both train-label views in one pass, preserving their row order."""
    entries: list[dict[str, object]] = []

    def append(line: str, *, positive: bool, negative: bool) -> None:
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            return
        if (
            not isinstance(row, dict)
            or row.get("reviewed") is not True
            or str(row.get("split") or "train") != "train"
        ):
            return
        query = row.get("query")
        if not isinstance(query, str) or not query.strip():
            return
        tokens: list[str] | None = None
        for enabled, field_name, kind in (
            (positive, "expected_pages", "reviewed_positive"),
            (negative, "negative_pages", "explicit_eval_negative"),
        ):
            pages = row.get(field_name)
            if not enabled or not isinstance(pages, list):
                continue
            page_ids = [page for page in pages if isinstance(page, str) and page]
            if not page_ids:
                continue
            if tokens is None:
                tokens = tokenize(query)
            if not tokens:
                continue
            ts = _parse_ts(row.get("ts") or row.get("reviewed_at"))
            entries.append({
                # Preserve tokenizer insertion order, including duplicates:
                # frozenset iteration can affect equal-weight centroid ties.
                "tokens": tokens,
                "pages": page_ids,
                "ts": ts.isoformat() if ts is not None else None,
                "kind": kind,
            })

    # read_text's universal newline behavior, before the two legacy split rules.
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    for line in text.split("\n"):
        fragments = line.splitlines()
        append(line, positive=True, negative=fragments == [line])
        if fragments != [line]:
            # The negative view historically uses splitlines(), the positive
            # view split("\n"). Preserve literal Unicode separator behavior.
            for fragment in fragments:
                append(fragment, positive=False, negative=True)
    return entries


def _decode_golden_entries(
    values: object,
) -> tuple[list[_FeedbackEntry], list[_FeedbackEntry]]:
    if not isinstance(values, list):
        raise ValueError("invalid golden cache entries")
    negatives: list[_FeedbackEntry] = []
    protections: list[_FeedbackEntry] = []
    for value in values:
        if not isinstance(value, dict):
            raise ValueError("invalid golden cache entry")
        for name in ("tokens", "pages"):
            items = value.get(name)
            if not isinstance(items, list) or not items or not all(
                isinstance(item, str) and item for item in items
            ):
                raise ValueError("invalid golden cache strings")
        kind = value.get("kind")
        ts = value.get("ts")
        if kind not in {"reviewed_positive", "explicit_eval_negative"} or (
            ts is not None and (not isinstance(ts, str) or _parse_ts(ts) is None)
        ):
            raise ValueError("invalid golden cache metadata")
        negative = kind == "explicit_eval_negative"
        entry = _FeedbackEntry(
            tokens=frozenset(value["tokens"]),
            pages=tuple(value["pages"]),
            ts=_parse_ts(ts),
            kind=kind,
            frontier_confirmed=negative,
            label_quality="gold" if negative else "",
        )
        (negatives if negative else protections).append(entry)
    return negatives, protections


def _golden_source_key(
    source_sha256: str,
    signature: tuple[int, ...],
    *,
    ends_with_newline: bool,
) -> dict[str, object]:
    return {
        "source_sha256": source_sha256,
        # The full stat signature (dev, ino, size, mtime_ns, ctime_ns) is stored
        # so an in-place rewrite that keeps the first four still moves ctime_ns.
        "signature": list(signature),
        "size": signature[2],
        "ends_with_newline": ends_with_newline,
        "schema_version": GOLDEN_CACHE_SCHEMA,
        "tokenizer_version": TOKENIZER_VERSION,
    }


def _read_golden_cache(cache_file: Path) -> dict[str, Any] | None:
    """Load a self-consistent sidecar envelope; currency is the caller's call."""

    try:
        cached = json.loads(cache_file.read_bytes())
    except (OSError, ValueError):
        return None
    if not isinstance(cached, dict):
        return None
    source = cached.get("source")
    payload = cached.get("entries_json")
    if not isinstance(source, dict) or not isinstance(payload, str):
        return None
    if cached.get("entries_sha256") != hashlib.sha256(payload.encode()).hexdigest():
        return None
    try:
        values = json.loads(payload)
    except ValueError:
        return None
    if not isinstance(values, list):
        return None
    return {"source": source, "values": values}


def _cached_golden_source(source: Mapping[str, Any]) -> dict[str, Any] | None:
    if (
        source.get("schema_version") != GOLDEN_CACHE_SCHEMA
        or source.get("tokenizer_version") != TOKENIZER_VERSION
    ):
        return None
    digest = source.get("source_sha256")
    size = source.get("size")
    signature = source.get("signature")
    ends = source.get("ends_with_newline")
    if (
        not isinstance(digest, str)
        or len(digest) != 64
        or any(char not in "0123456789abcdef" for char in digest)
        or not isinstance(size, int)
        or isinstance(size, bool)
        or size < 0
        or not isinstance(signature, list)
        or len(signature) != 5
        or not all(
            isinstance(value, int) and not isinstance(value, bool)
            for value in signature
        )
        or not isinstance(ends, bool)
        or size != signature[2]
    ):
        return None
    return {
        "source_sha256": digest,
        "size": size,
        "signature": signature,
        "ends_with_newline": ends,
    }


def _extend_golden_entries(
    path: Path,
    values: list[Any],
    stored: Mapping[str, Any],
) -> tuple[
    tuple[list[_FeedbackEntry], list[_FeedbackEntry]] | None,
    tuple[list[Any], str, bool] | None,
]:
    """Parse only the appended tail after verifying the cached prefix hash.

    The cached source must end with b"\n", so the split is always a line
    boundary: every cached line is complete and no "\r\n" can straddle it.
    """

    try:
        with path.open("rb") as stream:
            digest = hashlib.sha256()
            remaining = int(stored["size"])
            while remaining:
                chunk = stream.read(min(1 << 20, remaining))
                if not chunk:
                    return None, None
                digest.update(chunk)
                remaining -= len(chunk)
            if digest.hexdigest() != stored["source_sha256"]:
                return None, None
            tail = stream.read()
        if not tail:
            return None, None
        tail_text = tail.decode("utf-8")
    except (OSError, UnicodeDecodeError):
        return None, None
    digest.update(tail)
    combined = [*values, *_parse_golden_entries(tail_text)]
    try:
        entries = _decode_golden_entries(combined)
    except ValueError:
        return None, None
    return entries, (combined, digest.hexdigest(), tail.endswith(b"\n"))


def _write_golden_cache(
    cache_file: Path,
    key: dict[str, object],
    values: list[Any],
) -> None:
    payload = json.dumps(values, ensure_ascii=False, separators=(",", ":"))
    cache_file.parent.mkdir(parents=True, exist_ok=True)
    atomic_write(cache_file, json.dumps({
        "source": key,
        "entries_json": payload,
        "entries_sha256": hashlib.sha256(payload.encode()).hexdigest(),
    }, ensure_ascii=False, separators=(",", ":")))
    os.chmod(cache_file, 0o600)


def _load_golden_entries() -> tuple[list[_FeedbackEntry], list[_FeedbackEntry]]:
    path = _golden_file()
    cache_file = (
        path.with_suffix(".tokens.json")
        if GOLDEN_FILE_OVERRIDE is not None
        else GOLDEN_CACHE_FILE
    )
    try:
        signature = _golden_signature(path)
        memory_key = (str(path), signature, GOLDEN_CACHE_SCHEMA, TOKENIZER_VERSION)
        with _CACHE_LOCK:
            if _GOLD_NEGATIVE_CACHE.key == _PROTECT_CACHE.key == memory_key:
                return _GOLD_NEGATIVE_CACHE.entries, _PROTECT_CACHE.entries
        stable = signature == _golden_signature(path)
    except OSError:
        return [], []
    entries: tuple[list[_FeedbackEntry], list[_FeedbackEntry]] | None = None
    pending: tuple[list[Any], str, bool] | None = None
    if stable:
        cached = _read_golden_cache(cache_file)
        stored = None if cached is None else _cached_golden_source(cached["source"])
        if cached is not None and stored is not None:
            if stored["signature"] == list(signature):
                # ponytail: unchanged-stat fast path, no source read or hash.
                # Ceiling: an in-place rewrite that preserves size, inode,
                # mtime_ns and ctime_ns would be missed (upgrade path: hash the
                # source on a timer). The signature above already includes
                # ctime_ns on top of the size/inode/mtime_ns identity.
                try:
                    entries = _decode_golden_entries(cached["values"])
                except ValueError:
                    entries = None
            elif stored["ends_with_newline"] and signature[2] > stored["size"]:
                entries, pending = _extend_golden_entries(
                    path, cached["values"], stored
                )
    if entries is None:
        try:
            source = path.read_bytes()
            stable = signature == _golden_signature(path)
            if not stable:
                # Never publish or reuse a result from a concurrent source update.
                source = path.read_bytes()
        except OSError:
            return [], []
        values = _parse_golden_entries(source.decode("utf-8"))
        entries = _decode_golden_entries(values)
        pending = (values, hashlib.sha256(source).hexdigest(), source.endswith(b"\n"))
    if pending is not None and os.environ.get("CHRONOVISOR_READ_ONLY") != "1":
        values, source_sha256, ends_with_newline = pending
        try:
            if stable and signature == _golden_signature(path):
                _write_golden_cache(
                    cache_file,
                    _golden_source_key(
                        source_sha256, signature, ends_with_newline=ends_with_newline
                    ),
                    values,
                )
        except OSError:
            pass
    try:
        if stable and signature != _golden_signature(path):
            return _decode_golden_entries(
                _parse_golden_entries(path.read_bytes().decode("utf-8"))
            )
        if stable:
            with _CACHE_LOCK:
                _GOLD_NEGATIVE_CACHE.key = _PROTECT_CACHE.key = memory_key
                _GOLD_NEGATIVE_CACHE.entries, _PROTECT_CACHE.entries = entries
    except OSError:
        pass
    return entries


def _load_protections() -> list[_FeedbackEntry]:
    return _load_golden_entries()[1]


def _load_reviewed_train_negatives() -> list[_FeedbackEntry]:
    return _load_golden_entries()[0]


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
        token_weights: dict[str, float] = {}
        strongest_jaccard = 0.0
        newest_ts: datetime | None = None
        evidence_kinds: set[str] = set()
        for entry, jaccard in matches:
            strongest_jaccard = max(strongest_jaccard, jaccard)
            evidence_kinds.add(entry.kind)
            if _ts_rank(entry.ts) > _ts_rank(newest_ts):
                newest_ts = entry.ts
            for token in entry.tokens:
                token_weights[token] = token_weights.get(token, 0.0) + jaccard
        centroid_tokens = frozenset(
            sorted(token_weights, key=token_weights.__getitem__, reverse=True)[:48]
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
        penalty = (
            float(cast(Any, row.get("penalty") or 0.0)) if row else None
        )
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
        page_id: float(cast(Any, row["penalty"]))
        for page_id, row in contextual_negative_trace(query, config).items()
    }


def apply_penalties(
    results: list[ScoredPage], penalties: dict[str, float]
) -> list[ScoredPage]:
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
            replace(
                page,
                score=float(page.score) * max(0.0, 1.0 - penalty),
            )
        )
    adjusted.sort(key=lambda x: x.score, reverse=True)
    return adjusted
