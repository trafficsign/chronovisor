"""Durable acknowledgement between page apply and raw retirement.

Page mutation and ``processed_raw_files`` live in different durable stores.  A
crash between them used to replay the same raw through inference and apply the
same semantic mutation twice.  This module publishes a content-bound receipt
for the completed job before the orchestrator retires any source raw.  A later
process can validate the receipt and finish only the state transition.
"""

from __future__ import annotations

from chronovisor.hashutil import sha256_bytes as _sha256

import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Sequence

from chronovisor import store as chronovisor_store
from chronovisor.canonical_json import (
    canonical_json_bytes_strict as _canonical_bytes,
)
from chronovisor.link_fix import atomic_write
from chronovisor.sealed_artifact_decoder import schema_matches


RECEIPT_SCHEMA = "chronovisor.raw-completion-ack.v1"


class RawCompletionAckError(RuntimeError):
    """Base class for durable raw-completion acknowledgement failures."""


class RawCompletionReceiptInvalid(RawCompletionAckError):
    """An existing receipt cannot safely authorize raw retirement."""


class RawCompletionReceiptPublishError(RawCompletionAckError):
    """A completed mutation could not publish its durable receipt."""


class RawCompletionStatePending(RawCompletionAckError):
    """The receipt is durable, but processed-state acknowledgement is pending."""




def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _source_paths(paths: Iterable[Path]) -> tuple[Path, ...]:
    normalized = sorted(
        (Path(path) for path in paths),
        key=lambda path: path.name,
    )
    if not normalized:
        raise RawCompletionReceiptInvalid("source set is empty")
    filenames = [path.name for path in normalized]
    if len(set(filenames)) != len(filenames):
        raise RawCompletionReceiptInvalid("source filenames are not unique")
    return tuple(normalized)


def source_set_key(paths: Iterable[Path]) -> str:
    """Return the stable discovery key for one logical source set."""

    normalized = _source_paths(paths)
    return _sha256(_canonical_bytes([path.name for path in normalized]))


def receipt_path(paths: Iterable[Path]) -> Path:
    """Return the one receipt slot for a logical source filename set."""

    key = source_set_key(paths)
    return chronovisor_store.CHRONOVISOR_ROOT / "runtime" / "raw-completion-acks" / f"{key}.json"


def _source_evidence(paths: Sequence[Path]) -> list[dict[str, object]]:
    evidence: list[dict[str, object]] = []
    for path in paths:
        try:
            raw = path.read_bytes()
        except OSError as exc:
            raise RawCompletionReceiptInvalid(
                f"source unreadable: {path.name}: {type(exc).__name__}: {exc}"
            ) from exc
        evidence.append(
            {
                "filename": path.name,
                "bytes": len(raw),
                "sha256": _sha256(raw),
            }
        )
    return evidence


def _page_postimages(page_ids: Sequence[str]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for page_id in sorted(set(page_ids)):
        if not isinstance(page_id, str) or not page_id:
            raise RawCompletionReceiptPublishError("completed job has invalid page id")
        path = chronovisor_store.find_page(page_id)
        if path is None or not path.is_file():
            raise RawCompletionReceiptPublishError(
                f"completed page postimage is missing: {page_id}"
            )
        try:
            raw = path.read_bytes()
            relative_path = path.relative_to(chronovisor_store.CHRONOVISOR_ROOT).as_posix()
        except (OSError, ValueError) as exc:
            raise RawCompletionReceiptPublishError(
                f"completed page postimage is unreadable: {page_id}: "
                f"{type(exc).__name__}: {exc}"
            ) from exc
        rows.append(
            {
                "page_id": page_id,
                "path": relative_path,
                "bytes": len(raw),
                "sha256": _sha256(raw),
            }
        )
    return rows


def _job_outcome(job: Any) -> dict[str, object]:
    status = getattr(getattr(job, "status", None), "value", None)
    if status != "completed":
        raise RawCompletionReceiptPublishError(
            f"completion callback observed non-completed job status: {status!r}"
        )
    pages_created = sorted(set(getattr(job, "pages_created", None) or []))
    pages_updated = sorted(set(getattr(job, "pages_updated", None) or []))
    if not all(isinstance(value, str) for value in pages_created + pages_updated):
        raise RawCompletionReceiptPublishError(
            "completed job page outcome contains a non-string id"
        )
    try:
        result_sha256 = _sha256(_canonical_bytes(getattr(job, "result", None)))
    except (TypeError, ValueError) as exc:
        raise RawCompletionReceiptPublishError(
            f"completed job result is not canonical JSON: {type(exc).__name__}: {exc}"
        ) from exc
    outcome: dict[str, object] = {
        "job_id": str(getattr(job, "job_id", "")),
        "status": "completed",
        "processor": str(getattr(job, "processor", "")),
        "completed_at": str(getattr(job, "completed_at", "") or ""),
        "pages_created": pages_created,
        "pages_updated": pages_updated,
        "result_sha256": result_sha256,
        "page_postimages": _page_postimages(pages_created + pages_updated),
    }
    return outcome


def _receipt_payload(paths: Sequence[Path], job: Any) -> dict[str, object]:
    key = source_set_key(paths)
    outcome = _job_outcome(job)
    payload: dict[str, object] = {
        "schema": RECEIPT_SCHEMA,
        "source_set_key": key,
        "sources": _source_evidence(paths),
        "outcome": outcome,
        "outcome_sha256": _sha256(_canonical_bytes(outcome)),
    }
    payload["receipt_sha256"] = _sha256(_canonical_bytes(payload))
    return payload


def _invalid(reason: str) -> RawCompletionReceiptInvalid:
    return RawCompletionReceiptInvalid(f"raw completion receipt invalid: {reason}")


def _verify_page_postimages(
    rows: object,
    *,
    verify_current: bool,
) -> None:
    if not isinstance(rows, list):
        raise _invalid("page_postimages is not a list")
    previous: tuple[str, str] | None = None
    for row in rows:
        if not isinstance(row, dict) or set(row) != {
            "page_id",
            "path",
            "bytes",
            "sha256",
        }:
            raise _invalid("page postimage row schema mismatch")
        page_id = row.get("page_id")
        relative = row.get("path")
        size = row.get("bytes")
        digest = row.get("sha256")
        if (
            not isinstance(page_id, str)
            or not page_id
            or not isinstance(relative, str)
            or not relative
            or not isinstance(size, int)
            or size < 0
            or not _is_sha256(digest)
        ):
            raise _invalid("page postimage row value mismatch")
        relative_path = Path(relative)
        if relative_path.is_absolute() or ".." in relative_path.parts:
            raise _invalid("page postimage path escapes the wiki")
        path = chronovisor_store.CHRONOVISOR_ROOT / relative_path
        try:
            path.relative_to(chronovisor_store.PAGES_DIR)
        except ValueError as exc:
            raise _invalid(f"page postimage path is outside pages: {page_id}") from exc
        if path.stem != page_id:
            raise _invalid(f"page postimage identity mismatch: {page_id}")
        if verify_current:
            try:
                raw = path.read_bytes()
            except OSError as exc:
                raise _invalid(
                    f"page postimage unavailable: {page_id}: {type(exc).__name__}"
                ) from exc
            if len(raw) != size or _sha256(raw) != digest:
                raise _invalid(f"page postimage changed during publication: {page_id}")
        order_key = (page_id, relative)
        if previous is not None and order_key <= previous:
            raise _invalid("page postimages are not canonical")
        previous = order_key


def _validate_receipt_payload(
    payload: object,
    paths: Sequence[Path],
    *,
    verify_current_postimages: bool,
) -> dict[str, Any]:
    if not isinstance(payload, dict) or set(payload) != {
        "schema",
        "source_set_key",
        "sources",
        "outcome",
        "outcome_sha256",
        "receipt_sha256",
    }:
        raise _invalid("top-level schema mismatch")
    if not schema_matches(payload.get("schema"), RECEIPT_SCHEMA):
        raise _invalid("schema version mismatch")
    key = source_set_key(paths)
    if payload.get("source_set_key") != key:
        raise _invalid("source set key mismatch")
    if payload.get("sources") != _source_evidence(paths):
        raise _invalid("source content changed")

    outcome = payload.get("outcome")
    if not isinstance(outcome, dict) or set(outcome) != {
        "job_id",
        "status",
        "processor",
        "completed_at",
        "pages_created",
        "pages_updated",
        "result_sha256",
        "page_postimages",
    }:
        raise _invalid("job outcome schema mismatch")
    if (
        outcome.get("status") != "completed"
        or not isinstance(outcome.get("job_id"), str)
        or not outcome.get("job_id")
        or not isinstance(outcome.get("processor"), str)
        or not isinstance(outcome.get("completed_at"), str)
        or not isinstance(outcome.get("pages_created"), list)
        or not isinstance(outcome.get("pages_updated"), list)
        or not _is_sha256(outcome.get("result_sha256"))
    ):
        raise _invalid("job outcome value mismatch")
    page_ids = outcome["pages_created"] + outcome["pages_updated"]
    if (
        not all(isinstance(value, str) and value for value in page_ids)
        or outcome["pages_created"] != sorted(set(outcome["pages_created"]))
        or outcome["pages_updated"] != sorted(set(outcome["pages_updated"]))
    ):
        raise _invalid("job page ids are not canonical")
    postimages = outcome.get("page_postimages")
    _verify_page_postimages(
        postimages,
        verify_current=verify_current_postimages,
    )
    if not isinstance(postimages, list) or {
        str(row.get("page_id")) for row in postimages if isinstance(row, dict)
    } != set(page_ids):
        raise _invalid("page outcome and postimages disagree")
    if payload.get("outcome_sha256") != _sha256(_canonical_bytes(outcome)):
        raise _invalid("job outcome digest mismatch")

    receipt_digest = payload.get("receipt_sha256")
    if not _is_sha256(receipt_digest):
        raise _invalid("receipt self-hash is malformed")
    without_hash = dict(payload)
    without_hash.pop("receipt_sha256")
    if receipt_digest != _sha256(_canonical_bytes(without_hash)):
        raise _invalid("receipt self-hash mismatch")
    return payload


def load_valid_receipt(
    paths: Iterable[Path],
    *,
    verify_current_postimages: bool = False,
) -> dict[str, Any] | None:
    """Load and fully validate an ACK receipt, or return ``None`` if absent.

    An existing but malformed or source-mismatched receipt raises.  Recorded
    page postimages are validated structurally and by their receipt digest, but
    are compared with live pages only during publication readback.  A later raw
    may legitimately update the same page before this ACK resumes; that must not
    replay the earlier semantic work.
    """

    normalized = _source_paths(paths)
    path = receipt_path(normalized)
    if not path.exists():
        return None
    try:
        raw = path.read_bytes()
        payload = json.loads(raw)
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise _invalid(f"receipt unreadable: {type(exc).__name__}: {exc}") from exc
    verified = _validate_receipt_payload(
        payload,
        normalized,
        verify_current_postimages=verify_current_postimages,
    )
    expected_bytes = _canonical_bytes(verified) + b"\n"
    if raw != expected_bytes:
        raise _invalid("receipt bytes are not canonical")
    return verified


def publish_receipt(paths: Iterable[Path], job: Any) -> tuple[Path, dict[str, Any]]:
    """Atomically publish and read back a completed-job ACK receipt."""

    normalized = _source_paths(paths)
    path = receipt_path(normalized)
    try:
        existing = load_valid_receipt(
            normalized,
            verify_current_postimages=True,
        )
        if existing is not None:
            return path, existing
        payload = _receipt_payload(normalized, job)
        path.parent.mkdir(parents=True, exist_ok=True)
        try:
            atomic_write(path, (_canonical_bytes(payload) + b"\n").decode("utf-8"))
        except Exception as write_exc:
            # atomic_write may report a directory-fsync error after os.replace
            # has already made the receipt durable.  Accept only an exact,
            # fully verified readback; otherwise preserve the original failure.
            try:
                committed = load_valid_receipt(
                    normalized,
                    verify_current_postimages=True,
                )
            except RawCompletionReceiptInvalid:
                committed = None
            if committed == payload:
                return path, committed
            raise RawCompletionReceiptPublishError(
                "raw completion receipt publish failed: "
                f"{type(write_exc).__name__}: {write_exc}"
            ) from write_exc
        verified = load_valid_receipt(
            normalized,
            verify_current_postimages=True,
        )
        if verified is None or verified != payload:
            raise RawCompletionReceiptPublishError(
                "receipt disappeared or changed during readback"
            )
        return path, verified
    except RawCompletionReceiptPublishError as exc:
        if str(exc).startswith("raw completion receipt publish failed:"):
            raise
        raise RawCompletionReceiptPublishError(
            f"raw completion receipt publish failed: {type(exc).__name__}: {exc}"
        ) from exc
    except Exception as exc:
        raise RawCompletionReceiptPublishError(
            f"raw completion receipt publish failed: {type(exc).__name__}: {exc}"
        ) from exc


def receipt_summary(path: Path, payload: dict[str, Any]) -> dict[str, object]:
    """Return safe, compact metadata for job and dashboard results."""

    return {
        "path": str(path),
        "source_set_key": payload.get("source_set_key"),
        "receipt_sha256": payload.get("receipt_sha256"),
        "outcome_sha256": payload.get("outcome_sha256"),
        "source_files": [
            row.get("filename")
            for row in payload.get("sources", [])
            if isinstance(row, dict)
        ],
        "resumed": False,
    }


__all__ = [
    "RECEIPT_SCHEMA",
    "RawCompletionAckError",
    "RawCompletionReceiptInvalid",
    "RawCompletionReceiptPublishError",
    "RawCompletionStatePending",
    "load_valid_receipt",
    "publish_receipt",
    "receipt_path",
    "receipt_summary",
    "source_set_key",
]
