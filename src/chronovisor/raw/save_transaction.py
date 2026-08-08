"""Crash-safe transaction primitives for session-to-raw save workers.

The state cursor and the raw file cannot be committed in one filesystem
transaction.  A published raw is therefore the durable transaction receipt:
its deterministic name identifies the source interval and an embedded marker
lets a later worker verify the receipt before advancing a stale cursor.
"""

from __future__ import annotations

import fcntl
import hashlib
import json
import re
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from chronovisor.raw.raw_store import RawStore

_MARKER_PREFIX = "<!-- chronovisor-save-transaction:"
_MARKER_SUFFIX = "-->"
_HOST_RE = re.compile(r"^[a-z0-9][a-z0-9-]{0,31}$")
_SESSION_KEY_RE = re.compile(r"^[0-9a-f]{24}$")


@dataclass(frozen=True)
class SaveTransaction:
    """Identity of one contiguous transcript delta."""

    host: str
    session_key: str
    after_line: int
    until_line: int
    idempotency_key: str


@dataclass(frozen=True)
class PublishedSaveTransaction:
    """A verified raw receipt and its source interval."""

    transaction: SaveTransaction
    path: Path


@dataclass(frozen=True)
class SaveTransactionReceipt:
    """Self-verifying transaction marker plus the payload digest it protects."""

    transaction: SaveTransaction
    payload_sha256: str


def _normalized_host(host: str) -> str:
    normalized = host.strip().lower()
    if not _HOST_RE.fullmatch(normalized):
        raise ValueError(f"invalid save transaction host: {host!r}")
    return normalized


def save_session_key(*, host: str, session_file: Path, session_id: str | None) -> str:
    """Return a stable, non-sensitive identity for a host transcript."""
    normalized_host = _normalized_host(host)
    resolved = session_file.expanduser().resolve(strict=False)
    material = f"{normalized_host}\0{resolved}\0{session_id or ''}".encode()
    return hashlib.sha256(material).hexdigest()[:24]


def make_save_transaction(
    *,
    host: str,
    session_file: Path,
    session_id: str | None,
    after_line: int,
    until_line: int,
) -> SaveTransaction:
    """Build the deterministic identity used by raw publication and recovery."""
    normalized_host = _normalized_host(host)
    if after_line < 0 or until_line <= after_line:
        raise ValueError(
            f"invalid save transaction interval: {after_line}-{until_line}"
        )
    session_key = save_session_key(
        host=normalized_host,
        session_file=session_file,
        session_id=session_id,
    )
    idempotency_key = (
        f"{normalized_host}-{session_key}-from{after_line}-to{until_line}"
    )
    return SaveTransaction(
        host=normalized_host,
        session_key=session_key,
        after_line=after_line,
        until_line=until_line,
        idempotency_key=idempotency_key,
    )


def save_transaction_marker(
    transaction: SaveTransaction,
    *,
    payload_sha256: str,
) -> str:
    """Serialize a receipt marker that survives raw frontmatter patching."""
    if not re.fullmatch(r"[0-9a-f]{64}", payload_sha256):
        raise ValueError("payload_sha256 must be a lowercase SHA-256 digest")
    payload = {
        "version": 1,
        "host": transaction.host,
        "session_key": transaction.session_key,
        "after_line": transaction.after_line,
        "until_line": transaction.until_line,
        "idempotency_key": transaction.idempotency_key,
        "payload_sha256": payload_sha256,
    }
    return (
        f"{_MARKER_PREFIX} "
        f"{json.dumps(payload, ensure_ascii=True, sort_keys=True, separators=(',', ':'))} "
        f"{_MARKER_SUFFIX}"
    )


def attach_save_transaction_marker(transaction: SaveTransaction, content: str) -> str:
    """Attach a receipt whose digest covers every byte after its marker line."""
    protected_payload = "\n" + content
    digest = hashlib.sha256(protected_payload.encode("utf-8")).hexdigest()
    marker = save_transaction_marker(transaction, payload_sha256=digest)
    return f"{marker}\n{protected_payload}"


def publish_transcript_capture(
    *,
    raw_dir: Path,
    host: str,
    session_key: str,
    session_id: str | None,
    session_file: Path,
    after_line: int,
    until_line: int,
    idempotency_key: str,
    source_bytes: bytes,
    record_count: int,
    legacy_content: str,
    legacy_session_id: str,
    keywords: list[str],
    trigger_ingest: bool,
    legacy_publisher: Callable[..., dict[str, Any]],
) -> dict[str, Any]:
    """Publish according to the reversible legacy/shadow/v2 feature flag."""

    from chronovisor.raw.raw_segment import append_capture
    from chronovisor.raw.raw_store import raw_layout_mode

    mode = raw_layout_mode(chronovisor_root=raw_dir.parent)
    if mode == "legacy":
        return legacy_publisher(
            legacy_content,
            session_id=legacy_session_id,
            keywords=keywords,
            trigger_ingest=trigger_ingest,
            idempotency_key=idempotency_key,
        )

    if mode == "shadow":
        authority = legacy_publisher(
            legacy_content,
            session_id=legacy_session_id,
            keywords=keywords,
            trigger_ingest=trigger_ingest,
            idempotency_key=idempotency_key,
        )
        try:
            shadow = append_capture(
                raw_dir=raw_dir,
                raw_id=f"save-{idempotency_key}.md",
                idempotency_key=idempotency_key,
                host=host,
                session_key=session_key,
                session_id=session_id,
                source_file=session_file,
                after_line=after_line,
                until_line=until_line,
                source_bytes=source_bytes,
                record_count=record_count,
            )
        except Exception as exc:
            # Legacy is the explicit authority in shadow mode.  Its durable
            # receipt may advance the cursor, while the mismatch stays visible
            # for the adoption gate instead of taking down Stop capture.
            return {
                **authority,
                "layout": "shadow",
                "shadow_error": f"{type(exc).__name__}: {exc}",
            }
        return {
            **authority,
            "layout": "shadow",
            "shadow_result": shadow.to_result(),
            "shadow_comparison": _compare_shadow_capture(
                legacy_content=legacy_content,
                source_bytes=source_bytes,
                after_line=after_line,
                until_line=until_line,
            ),
        }

    receipt = append_capture(
        raw_dir=raw_dir,
        raw_id=f"save-{idempotency_key}.md",
        idempotency_key=idempotency_key,
        host=host,
        session_key=session_key,
        session_id=session_id,
        source_file=session_file,
        after_line=after_line,
        until_line=until_line,
        source_bytes=source_bytes,
        record_count=record_count,
    )
    return {**receipt.to_result(), "layout": "v2"}


def _compare_shadow_capture(
    *,
    legacy_content: str,
    source_bytes: bytes,
    after_line: int,
    until_line: int,
) -> dict[str, Any]:
    """Compare logical source records without requiring byte-identical envelopes."""

    try:
        source_rows = [json.loads(line) for line in source_bytes.splitlines()]
        payload_text = legacy_content.split("```json\n", 1)[1].split("\n```", 1)[0]
        legacy_rows = json.loads(payload_text)
        if not isinstance(legacy_rows, list) or any(
            not isinstance(row, dict) for row in legacy_rows
        ):
            raise ValueError("legacy transcript payload is not an object array")
        legacy_events = [row.get("event") for row in legacy_rows]
        legacy_lines = [row.get("line") for row in legacy_rows]
        expected_lines = list(range(after_line + 1, until_line + 1))
        matched = sum(
            source == legacy
            for source, legacy in zip(source_rows, legacy_events, strict=False)
        )
        duplicate_lines = len(legacy_lines) - len(set(legacy_lines))
        missing = max(0, len(source_rows) - matched)
        extra = max(0, len(legacy_events) - matched)
        status = (
            "match"
            if source_rows == legacy_events
            and legacy_lines == expected_lines
            and duplicate_lines == 0
            else "mismatch"
        )
        return {
            "status": status,
            "source_records": len(source_rows),
            "legacy_records": len(legacy_events),
            "matched_records": matched,
            "missing": missing,
            "extra": extra,
            "duplicate_lines": duplicate_lines,
            "line_identity_match": legacy_lines == expected_lines,
            "source_sha256": hashlib.sha256(source_bytes).hexdigest(),
        }
    except Exception as exc:
        return {
            "status": "error",
            "error": f"{type(exc).__name__}: {exc}",
        }


def publish_oversized_shadow(
    *,
    raw_dir: Path,
    host: str,
    session_file: Path,
    session_id: str | None,
    source_line: int,
) -> dict[str, Any]:
    """Mirror a legacy fragment set as one native source record in shadow mode."""

    from chronovisor.raw.raw_segment import append_capture, copy_source_interval
    from chronovisor.raw.raw_store import raw_layout_mode

    if raw_layout_mode(chronovisor_root=raw_dir.parent) != "shadow":
        return {}
    transaction = make_save_transaction(
        host=host,
        session_file=session_file,
        session_id=session_id,
        after_line=source_line - 1,
        until_line=source_line,
    )
    source_bytes = copy_source_interval(
        session_file,
        after_line=source_line - 1,
        until_line=source_line,
    )
    try:
        receipt = append_capture(
            raw_dir=raw_dir,
            raw_id=f"save-{transaction.idempotency_key}.md",
            idempotency_key=transaction.idempotency_key,
            host=host,
            session_key=transaction.session_key,
            session_id=session_id,
            source_file=session_file,
            after_line=transaction.after_line,
            until_line=transaction.until_line,
            source_bytes=source_bytes,
            record_count=1,
        )
    except Exception as exc:
        return {"shadow_error": f"{type(exc).__name__}: {exc}"}
    return {
        "shadow_result": receipt.to_result(),
        "shadow_comparison": {
            "status": "match",
            "source_records": 1,
            "legacy_fragment_count": None,
            "missing": 0,
            "extra": 0,
            "duplicate_lines": 0,
            "source_sha256": hashlib.sha256(source_bytes).hexdigest(),
        },
    }


def parse_save_transaction_receipt(content: str) -> SaveTransactionReceipt | None:
    """Parse a marker and reject it if any protected payload byte changed."""
    offset = 0
    for line in content.splitlines(keepends=True):
        stripped = line.strip()
        if not stripped.startswith(_MARKER_PREFIX) or not stripped.endswith(
            _MARKER_SUFFIX
        ):
            offset += len(line)
            continue
        raw_payload = stripped[len(_MARKER_PREFIX) : -len(_MARKER_SUFFIX)].strip()
        try:
            payload = json.loads(raw_payload)
        except json.JSONDecodeError:
            return None
        if not isinstance(payload, dict) or payload.get("version") != 1:
            return None
        host = payload.get("host")
        session_key = payload.get("session_key")
        after_line = payload.get("after_line")
        until_line = payload.get("until_line")
        idempotency_key = payload.get("idempotency_key")
        payload_sha256 = payload.get("payload_sha256")
        if (
            not isinstance(host, str)
            or not _HOST_RE.fullmatch(host)
            or not isinstance(session_key, str)
            or not _SESSION_KEY_RE.fullmatch(session_key)
            or not isinstance(after_line, int)
            or isinstance(after_line, bool)
            or after_line < 0
            or not isinstance(until_line, int)
            or isinstance(until_line, bool)
            or until_line <= after_line
            or not isinstance(idempotency_key, str)
            or not isinstance(payload_sha256, str)
            or not re.fullmatch(r"[0-9a-f]{64}", payload_sha256)
        ):
            return None
        expected = f"{host}-{session_key}-from{after_line}-to{until_line}"
        if idempotency_key != expected:
            return None
        protected_payload = content[offset + len(line) :]
        actual_digest = hashlib.sha256(protected_payload.encode("utf-8")).hexdigest()
        if actual_digest != payload_sha256:
            return None
        transaction = SaveTransaction(
            host=host,
            session_key=session_key,
            after_line=after_line,
            until_line=until_line,
            idempotency_key=idempotency_key,
        )
        return SaveTransactionReceipt(
            transaction=transaction,
            payload_sha256=payload_sha256,
        )
    return None


def parse_save_transaction_marker(content: str) -> SaveTransaction | None:
    """Return the transaction only when its complete payload receipt verifies."""
    receipt = parse_save_transaction_receipt(content)
    return receipt.transaction if receipt is not None else None


def validate_published_save_receipt(
    *,
    raw_dir: Path,
    save_result: dict[str, object],
    expected: SaveTransaction,
) -> Path:
    """Validate the exact file returned by raw publication before state commit."""
    storage = save_result.get("storage")
    if storage in {"segment_open", "segment_sealed"}:
        expected_raw_id = f"save-{expected.idempotency_key}.md"
        raw_id = save_result.get("raw_id") or save_result.get("saved")
        if raw_id != expected_raw_id:
            raise ValueError("segment publisher returned the wrong logical Raw ID")
        unit = RawStore(raw_dir, mode="v2").resolve_segment(expected_raw_id)
        if unit is None or unit.commit is None:
            raise ValueError("segment publisher receipt cannot be resolved")
        commit = unit.commit
        if (
            commit.host != expected.host
            or commit.session_key != expected.session_key
            or commit.after_line != expected.after_line
            or commit.until_line != expected.until_line
            or commit.idempotency_key != expected.idempotency_key
        ):
            raise ValueError("segment publisher receipt transaction is invalid")
        # read_bytes verifies the exact committed range against its digest.
        RawStore(raw_dir, mode="v2").read_bytes(unit)
        return unit.path

    raw_path = save_result.get("path")
    if isinstance(raw_path, str) and raw_path:
        path = Path(raw_path).expanduser()
    else:
        saved = save_result.get("saved")
        if not isinstance(saved, str) or not saved:
            raise ValueError("raw publisher did not return a receipt path")
        path = raw_dir / saved
    resolved_dir = raw_dir.expanduser().resolve(strict=False)
    resolved_path = path.resolve(strict=False)
    if resolved_path.parent != resolved_dir:
        raise ValueError("raw publisher receipt path escaped RAW_DIR")
    if resolved_path.name != f"save-{expected.idempotency_key}.md":
        raise ValueError("raw publisher returned the wrong transaction filename")
    try:
        content = resolved_path.read_text(encoding="utf-8")
    except (OSError, UnicodeError) as exc:
        raise ValueError(f"raw publisher receipt is unreadable: {exc}") from exc
    receipt = parse_save_transaction_receipt(content)
    if receipt is None or receipt.transaction != expected:
        raise ValueError("raw publisher receipt marker or payload is invalid")
    return resolved_path


def find_published_save_transaction(
    *,
    raw_dir: Path,
    host: str,
    session_file: Path,
    session_id: str | None,
    after_line: int,
) -> PublishedSaveTransaction | None:
    """Find a verified raw published from the current (possibly stale) cursor.

    The glob is scoped to a hashed session identity and the exact starting
    cursor, so recovery remains cheap even when the wiki contains many raws.
    If an earlier bug produced more than one receipt, advancing to the widest
    verified interval avoids publishing that overlap yet again.
    """
    normalized_host = _normalized_host(host)
    session_key = save_session_key(
        host=normalized_host,
        session_file=session_file,
        session_id=session_id,
    )
    prefix = f"save-{normalized_host}-{session_key}-from{after_line}-to"
    candidates: list[PublishedSaveTransaction] = []
    try:
        segment_units = RawStore(raw_dir, mode="v2").iter_segment_units()
        for unit in segment_units:
            commit = unit.commit
            if (
                commit is None
                or commit.host != normalized_host
                or commit.session_key != session_key
                or commit.after_line != after_line
            ):
                continue
            RawStore(raw_dir, mode="v2").read_bytes(unit)
            candidates.append(
                PublishedSaveTransaction(
                    transaction=SaveTransaction(
                        host=commit.host,
                        session_key=commit.session_key,
                        after_line=commit.after_line,
                        until_line=commit.until_line,
                        idempotency_key=commit.idempotency_key,
                    ),
                    path=unit.path,
                )
            )
    except FileNotFoundError:
        pass
    search_roots = (
        raw_dir,
        raw_dir / ".dead-letter",
        raw_dir.parent / "runtime" / "failures" / "quarantined-raw",
    )
    for root in search_roots:
        for path in root.glob(f"{prefix}*.md"):
            try:
                transaction = parse_save_transaction_marker(
                    path.read_text(encoding="utf-8")
                )
            except (OSError, UnicodeError):
                continue
            if transaction is None:
                continue
            if (
                transaction.host != normalized_host
                or transaction.session_key != session_key
                or transaction.after_line != after_line
                or path.name != f"save-{transaction.idempotency_key}.md"
            ):
                continue
            candidates.append(
                PublishedSaveTransaction(transaction=transaction, path=path)
            )
    from chronovisor.raw.legacy_archive import iter_legacy_members, read_legacy_member

    for member in iter_legacy_members(raw_dir):
        if not member.raw_id.startswith(prefix):
            continue
        try:
            transaction = parse_save_transaction_marker(
                read_legacy_member(member).decode("utf-8")
            )
        except (OSError, UnicodeError):
            continue
        if (
            transaction is None
            or transaction.host != normalized_host
            or transaction.session_key != session_key
            or transaction.after_line != after_line
            or member.raw_id != f"save-{transaction.idempotency_key}.md"
        ):
            continue
        candidates.append(
            PublishedSaveTransaction(
                transaction=transaction,
                path=member.archive_path,
            )
        )
    if not candidates:
        return None
    return max(candidates, key=lambda item: item.transaction.until_line)


def state_lock_path(state_file: Path) -> Path:
    """Return the one lock shared by every transaction using ``state_file``.

    The state document contains entries for many sessions, so a per-session
    lock is insufficient: two different sessions could both load an old
    document and then replace it, losing one entry.  The state file is the
    coarsest correctness boundary and is host-specific under the defaults.
    """
    resolved = state_file.expanduser().resolve(strict=False)
    return resolved.with_name(f".{resolved.name}.lock")


@contextmanager
def save_transaction_lock(
    *,
    host: str,
    session_file: Path,
    state_file: Path,
) -> Iterator[Path]:
    """Serialize load -> extract -> writer -> raw publish -> state commit.

    ``host`` and ``session_file`` deliberately remain explicit at call sites
    to make the protected transaction identity reviewable.  Lock scope is the
    whole state file because it is a shared multi-session document.
    """
    del host, session_file
    lock_path = state_lock_path(state_file)
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a+", encoding="utf-8") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        try:
            yield lock_path
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
