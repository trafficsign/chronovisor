"""Atomic raw publication and optional ingest triggering."""

from __future__ import annotations

import contextlib
import os
import re
import secrets
from datetime import datetime
from pathlib import Path
from typing import Any

from chronovisor.core.durable_state import fsync_directory
from chronovisor.core.frontmatter import patch as patch_frontmatter
from chronovisor.core.save_transaction import parse_save_transaction_receipt
from chronovisor.core.store import RAW_DIR

RAW_PREFIX_RE = re.compile(r"[^a-zA-Z0-9_-]+")
RAW_SLUG_TOKEN_RE = re.compile(r"[a-zA-Z0-9]+")
RAW_UUIDISH_RE = re.compile(r"^[0-9a-f]{8,}(?:-[0-9a-f]{4,})*$", re.IGNORECASE)
RAW_TOPIC_STOPWORDS = {
    "and",
    "code",
    "codex",
    "claude",
    "memory",
    "save",
    "session",
    "the",
}
RAW_SOURCE_PREFIXES = ("claude-code", "codex", "ingest")
RAW_ALLOC_MAX_RETRIES = 32
RAW_IDEMPOTENCY_RE = re.compile(r"^[a-zA-Z0-9][a-zA-Z0-9_-]{0,159}$")
RAW_KEYWORD_FORBIDDEN_CHARS = frozenset(",[]:#{}\n\r")


def sanitize_raw_prefix(prefix: str) -> str:
    """Sanitize a caller-supplied prefix so it cannot break out of RAW_DIR."""

    if not prefix:
        return ""
    cleaned = RAW_PREFIX_RE.sub("-", prefix.strip())[:64]
    cleaned = cleaned.strip("-_")
    return f"-{cleaned}" if cleaned else ""


def sanitize_raw_component(value: str, *, max_len: int = 64) -> str:
    cleaned = RAW_PREFIX_RE.sub("-", value.strip().lower())
    cleaned = re.sub(r"-{2,}", "-", cleaned)
    return cleaned.strip("-_")[:max_len].strip("-_")


def raw_source_label(session_id: str | None) -> str:
    if not session_id:
        return ""
    cleaned = sanitize_raw_component(session_id, max_len=64)
    for source in RAW_SOURCE_PREFIXES:
        if cleaned == source or cleaned.startswith(f"{source}-"):
            return source
    if RAW_UUIDISH_RE.match(cleaned):
        return "session"
    return cleaned[:28].strip("-_")


def raw_topic_slug(
    content: str, keywords: list[str] | None = None, *, max_len: int = 56
) -> str:
    """Create a readable raw filename slug while keeping ASCII-only safety."""

    parts: list[str] = []
    if keywords:
        for keyword in keywords:
            for match in RAW_SLUG_TOKEN_RE.finditer(keyword.lower()):
                token = match.group(0)
                if (
                    len(token) < 2
                    or token in RAW_TOPIC_STOPWORDS
                    or RAW_UUIDISH_RE.match(token)
                ):
                    continue
                parts.append(token)
                slug = "-".join(parts)
                if len(slug) >= max_len:
                    return slug[:max_len].strip("-")
        if parts:
            return "-".join(parts)[:max_len].strip("-")

    candidates: list[str] = []
    for line in content.splitlines():
        stripped = line.strip(" #-\t")
        if not stripped:
            continue
        lower = stripped.lower()
        if lower in {
            "codex session memory save",
            "claude code session memory save",
            "memory",
            "writer reason",
            "rejected keywords",
        }:
            continue
        if lower.startswith(
            (
                "source:",
                "session id:",
                "cwd:",
                "session file:",
                "lines:",
                "memory writer model:",
                "generated at:",
                "raw_keywords:",
            )
        ):
            continue
        candidates.append(stripped)
        break

    for candidate in candidates:
        for match in RAW_SLUG_TOKEN_RE.finditer(candidate.lower()):
            token = match.group(0)
            if (
                len(token) < 2
                or token in RAW_TOPIC_STOPWORDS
                or RAW_UUIDISH_RE.match(token)
            ):
                continue
            parts.append(token)
            slug = "-".join(parts)
            if len(slug) >= max_len:
                return slug[:max_len].strip("-")
    return "-".join(parts)[:max_len].strip("-")


def raw_readable_component(prefix: str, topic_slug: str) -> str:
    source = raw_source_label(prefix)
    topic = sanitize_raw_component(topic_slug, max_len=56)
    name_parts = [part for part in (source, topic) if part]
    return f"-{'-'.join(name_parts)}" if name_parts else sanitize_raw_prefix(prefix)


def raw_candidate_path(prefix: str = "", topic_slug: str = "") -> Path:
    from chronovisor.core.raw_segment import capture_date
    from chronovisor.core.raw_store import raw_layout_mode

    readable = raw_readable_component(prefix, topic_slug)
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    suffix = secrets.token_hex(4)
    if raw_layout_mode(chronovisor_root=RAW_DIR.parent) == "v2":
        day_dir = RAW_DIR / capture_date()
        day_dir.mkdir(parents=True, exist_ok=True)
        return day_dir / f"manual-{ts}{readable}-{suffix}.md"
    return RAW_DIR / f"{ts}{readable}-{suffix}.md"


def allocate_raw_path(prefix: str = "", topic_slug: str = "") -> Path:
    """Reserve a unique, non-ingestable staging path for a raw publish."""

    readable = raw_readable_component(prefix, topic_slug)
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    last_err: Exception | None = None
    for _ in range(RAW_ALLOC_MAX_RETRIES):
        ts = datetime.now().strftime("%Y%m%d-%H%M%S")
        suffix = secrets.token_hex(4)
        path = RAW_DIR / f".{ts}{readable}-{suffix}.tmp"
        try:
            fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
            os.close(fd)
            return path
        except FileExistsError as exc:
            last_err = exc
    raise RuntimeError(
        f"could not allocate unique raw path after "
        f"{RAW_ALLOC_MAX_RETRIES} retries: {last_err}"
    )


def link_raw_no_replace(staging: Path, target: Path) -> None:
    """Atomically publish ``staging`` at ``target`` without replacement."""

    os.link(staging, target)


def publish_raw(content: str, *, prefix: str = "", topic_slug: str = "") -> Path:
    """Write a complete raw entry, then atomically expose its final name."""

    staging = allocate_raw_path(prefix=prefix, topic_slug=topic_slug)
    published: Path | None = None
    try:
        with staging.open("w", encoding="utf-8") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())

        last_err: Exception | None = None
        for _ in range(RAW_ALLOC_MAX_RETRIES):
            target = raw_candidate_path(prefix=prefix, topic_slug=topic_slug)
            try:
                link_raw_no_replace(staging, target)
                fsync_directory(target.parent)
                published = target
                break
            except FileExistsError as exc:
                last_err = exc
        if published is None:
            raise RuntimeError(
                "could not publish unique raw path after "
                f"{RAW_ALLOC_MAX_RETRIES} retries: {last_err}"
            )
        return published
    finally:
        with contextlib.suppress(FileNotFoundError):
            staging.unlink()


def publish_raw_idempotent(
    content: str,
    *,
    idempotency_key: str,
    prefix: str = "",
    topic_slug: str = "",
) -> tuple[Path, bool]:
    """Atomically publish one complete raw per idempotency key."""

    if not RAW_IDEMPOTENCY_RE.fullmatch(idempotency_key):
        raise ValueError(
            "idempotency_key must contain only ASCII letters, digits, dash, "
            "or underscore and be at most 160 characters"
        )
    staging = allocate_raw_path(prefix=prefix, topic_slug=topic_slug)
    target = RAW_DIR / f"save-{idempotency_key}.md"
    try:
        with staging.open("w", encoding="utf-8") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            link_raw_no_replace(staging, target)
        except FileExistsError as exc:
            try:
                existing = target.read_text(encoding="utf-8")
            except (OSError, UnicodeError) as exc:
                raise RuntimeError(
                    "idempotent raw target exists but cannot be verified"
                ) from exc
            if existing != content:
                incoming_receipt = parse_save_transaction_receipt(content)
                existing_receipt = parse_save_transaction_receipt(existing)
                if (
                    incoming_receipt is None
                    or existing_receipt is None
                    or incoming_receipt.transaction != existing_receipt.transaction
                    or incoming_receipt.transaction.idempotency_key != idempotency_key
                ):
                    raise RuntimeError(
                        "idempotency key collision with different or corrupt raw content"
                    ) from exc
            return target, True
        fsync_directory(RAW_DIR)
        return target, False
    finally:
        with contextlib.suppress(FileNotFoundError):
            staging.unlink()


def validate_raw_keyword(keyword: object) -> bool:
    """Return whether a keyword is safe to serialize as an inline-list item."""

    if not isinstance(keyword, str):
        return False
    if not keyword or not keyword.strip():
        return False
    for character in keyword:
        if character in RAW_KEYWORD_FORBIDDEN_CHARS or ord(character) < 0x20:
            return False
    return True


def ingest_raw(content: str, *, force: bool = True) -> dict[str, Any]:
    from chronovisor.ingest.orchestrator import run_pending_ingest

    path = publish_raw(content, prefix="ingest", topic_slug=raw_topic_slug(content))
    return {"saved": path.name, "ingest": run_pending_ingest(force=force)}


def record_raw(
    content: str,
    *,
    session_id: str | None = None,
    keywords: list[str] | None = None,
    trigger_ingest: bool = True,
    idempotency_key: str | None = None,
) -> dict[str, Any]:
    accepted: list[str] = []
    rejected: list[str] = []
    if keywords:
        for keyword in keywords:
            if validate_raw_keyword(keyword):
                accepted.append(keyword)
            else:
                rejected.append(
                    keyword if isinstance(keyword, str) else repr(keyword)
                )

    body = patch_frontmatter(content, {"raw_keywords": accepted}) if accepted else content
    raw_slug = raw_topic_slug(body, accepted)
    if idempotency_key:
        path, deduplicated = publish_raw_idempotent(
            body,
            idempotency_key=idempotency_key,
            prefix=session_id or "",
            topic_slug=raw_slug,
        )
    else:
        path = publish_raw(body, prefix=session_id or "", topic_slug=raw_slug)
        deduplicated = False

    from chronovisor.ingest.orchestrator import run_pending_ingest, should_ingest

    should, reason = should_ingest()
    result: dict[str, Any] = {
        "saved": path.name,
        "path": str(path),
        "raw_slug": raw_slug,
        "ingest_pending": should,
        "ingest_reason": reason,
    }
    if idempotency_key:
        result["deduplicated"] = deduplicated
    if rejected:
        result["rejected_keywords"] = rejected
    if should and trigger_ingest:
        result["ingest_triggered"] = run_pending_ingest()
    elif should:
        result["ingest_deferred"] = True
    result["accepted_keywords"] = accepted
    return result
