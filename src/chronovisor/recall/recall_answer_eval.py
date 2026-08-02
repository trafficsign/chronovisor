"""Answer-level Recall episodes and offline field-on/field-off evaluation.

Stop hooks only capture privacy-safe references after the host transcript has
been durably saved. Model replay and scoring are explicit offline operations.
No production or generated answer body is written to the Recall log or to the
episode/evaluation artifacts. The separately authored gold manifest remains an
explicit scorer input and is embedded so its reference/evidence binding can be
revalidated offline.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import importlib
import inspect
import json
import math
import os
import re
import sys
import tempfile
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Protocol

from chronovisor.core.durable_state import (
    DurableStateError,
    read_sealed_json,
    seal_object,
    sidecar_exclusive_lock,
    verify_sealed_object,
    write_sealed_json,
)
from chronovisor.core.jsonl_write import append_jsonl_durable
from chronovisor.core.store import (
    CHRONOVISOR_ROOT,
    INDEX_FILE,
    PAGES_DIR,
    find_page,
    init_chronovisor,
)
from chronovisor.raw.raw_segment import copy_source_interval
from chronovisor.raw.raw_store import RawStore
from chronovisor.raw.save_transaction import (
    find_published_save_transaction,
    make_save_transaction,
    parse_save_transaction_receipt,
    save_session_key,
    validate_published_save_receipt,
)
from chronovisor.recall.content_correction import complete_turns, source_recall_record
from chronovisor.recall.recall_confidence import (
    cluster_bootstrap_interval,
    cluster_rate_wilson_interval,
    manifest_sha256,
    wilson_interval,
)
from chronovisor.recall.recall_log_schema import (
    join_used_recall_episodes,
    page_ids_from_record,
)
from chronovisor.recall.recall_runtime import RECALL_LOG_FILE, RECALL_PULL_LOG_FILE
from chronovisor.recall.recall_runtime_paths import RECALL_DIR

ANSWER_EPISODE_LEDGER = RECALL_DIR / "answer-episodes.jsonl"
ANSWER_CAPTURE_CURSOR = (
    CHRONOVISOR_ROOT / "runtime" / "recall-answer-eval" / "capture-cursors.json"
)
ANSWER_SPLIT_MANIFEST = (
    CHRONOVISOR_ROOT / "runtime" / "recall-answer-eval" / "split-manifest.json"
)
LOCKED_ANSWER_EVAL_ARTIFACT = (
    CHRONOVISOR_ROOT / "runtime" / "recall-field" / "locked-answer-eval.json"
)
TRAIN_ANSWER_EVAL_ARTIFACT = (
    CHRONOVISOR_ROOT / "runtime" / "recall-answer-eval" / "train-answer-eval.json"
)
ANSWER_REVIEW_LEDGER = RECALL_DIR / "answer-review-receipts.jsonl"
ANSWER_EXECUTION_LEDGER = RECALL_DIR / "answer-execution-receipts.jsonl"
ANSWER_ADAPTER_REGISTRY = (
    CHRONOVISOR_ROOT / "runtime" / "recall-answer-eval" / "adapter-registry.json"
)
HOOK_ENABLE_ENV = "CHRONOVISOR_RECALL_ANSWER_CAPTURE_ENABLED"
ANSWER_EPISODE_SCHEMA_VERSION = 1
ANSWER_SPLIT_SCHEMA_VERSION = 2
ANSWER_EVAL_SCHEMA_VERSION = 3
ANSWER_DIMENSIONS = ("correctness", "grounding", "citation")
ANSWER_AUTHORITY_CONFIDENCE = 0.95
ANSWER_AUTHORITY_SEED = 1729
AUTHORITY_EMBARGO_SECONDS = 86_400
SCORER_CALIBRATION_SCHEMA_VERSION = 1
SCORER_CALIBRATION_CONFIDENCE = 0.95
SCORER_CALIBRATION_MIN_CASES = 20
SCORER_CALIBRATION_MIN_SESSIONS = 10
SCORER_CALIBRATION_MIN_CLUSTERS = 10
SCORER_CALIBRATION_COVERAGE_POINT_FLOOR = 0.95
SCORER_CALIBRATION_COVERAGE_LCB_FLOOR = 0.80
SCORER_CALIBRATION_AGREEMENT_POINT_FLOOR = 0.90
SCORER_CALIBRATION_AGREEMENT_LCB_FLOOR = 0.80
SCORER_CALIBRATION_SCORE_TOLERANCE = 0.10
SCORER_CALIBRATION_MAE_CEILING = 0.10
SCORER_CALIBRATION_ABS_BIAS_CEILING = 0.05
SCORER_CALIBRATION_WITHIN_TOLERANCE_LCB_FLOOR = 0.80
SCORER_CALIBRATION_PREFERENCE_LCB_FLOOR = 0.80
_REQUIRED_RUNNER_IDENTITY = (
    "runner_id",
    "model",
    "system_sha256",
    "sampler_sha256",
    "policy_sha256",
)
_REQUIRED_SCORER_IDENTITY = (
    "scorer_id",
    "version",
    "model",
    "system_sha256",
    "sampler_sha256",
    "policy_sha256",
    "rubric_sha256",
    "evidence_manifest_sha256",
    "calibration_protocol_sha256",
)
_REQUIRED_CALIBRATION_SCORER_IDENTITY = (
    "scorer_id",
    "version",
    "model",
    "system_sha256",
    "sampler_sha256",
    "policy_sha256",
    "rubric_sha256",
    "calibration_protocol_sha256",
)
_ALLOWED_GOLD_SOURCE_KINDS = frozenset(
    {"human_review", "adjudicated_benchmark"}
)


class AnswerRunner(Protocol):
    """Injected answer generator used identically for both paired arms."""

    def __call__(
        self, prompt: str, context: str, generation: Mapping[str, Any]
    ) -> Mapping[str, Any]: ...


class AnswerScorer(Protocol):
    """Independent scorer for one generated answer and immutable reference."""

    def __call__(
        self,
        prompt: str,
        answer: str,
        gold: Mapping[str, Any],
        scoring: Mapping[str, Any],
    ) -> Mapping[str, Any]: ...


class FieldEnvironmentReplay(Protocol):
    """Registered replay adapter; generator/scorer never receive arm labels."""

    def __call__(
        self, prompt: str, episode: Mapping[str, Any], seed: int
    ) -> Mapping[str, Any]: ...


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
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


def _sha_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _canonical_sha(value: object) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _valid_sha(value: object) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(
        char in "0123456789abcdef" for char in value.casefold()
    )


def _strict_utc(value: object) -> str:
    if not isinstance(value, str) or not value.strip():
        return ""
    try:
        parsed = datetime.fromisoformat(value.strip().replace("Z", "+00:00"))
    except ValueError:
        return ""
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        return ""
    return parsed.astimezone(UTC).isoformat().replace("+00:00", "Z")


def _now_utc() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def _verified_save_receipt(
    *,
    host: str,
    save_output: Mapping[str, Any],
    session_file: Path,
    session_id: str,
    raw_dir: Path,
) -> tuple[dict[str, Any], str]:
    """Revalidate the exact durable Raw transaction emitted by the saver."""

    declared_file = Path(str(save_output.get("session_file") or "")).expanduser()
    if declared_file.resolve(strict=False) != session_file.resolve(strict=False):
        return {}, "save_session_file_mismatch"
    if str(save_output.get("session_id") or "") != session_id:
        return {}, "save_session_id_mismatch"
    status = str(save_output.get("status") or "")
    try:
        if status == "saved":
            after_line = save_output.get("after_line")
            until_line = save_output.get("scanned_until_line")
            if (
                not isinstance(after_line, int)
                or isinstance(after_line, bool)
                or not isinstance(until_line, int)
                or isinstance(until_line, bool)
            ):
                return {}, "save_receipt_shape_invalid"
            results_value = save_output.get("save_results")
            if results_value is None:
                results_value = [save_output.get("save_result")]
            if (
                not isinstance(results_value, list)
                or not results_value
                or any(not isinstance(result, dict) for result in results_value)
                or save_output.get("chunk_count", len(results_value))
                != len(results_value)
            ):
                return {}, "save_receipt_shape_invalid"
            transactions = []
            expected_after = after_line
            recovered_value = save_output.get("recovered_save")
            if recovered_value is not None:
                if not isinstance(recovered_value, Mapping):
                    return {}, "recovered_receipt_missing"
                recovered_identity = str(
                    recovered_value.get("idempotency_key") or ""
                )
                recovered_match = re.fullmatch(
                    rf"{re.escape(host)}-([0-9a-f]{{24}})-from(\d+)-to(\d+)",
                    recovered_identity,
                )
                if recovered_match is None:
                    return {}, "recovered_receipt_identity_invalid"
                recovered_transaction = make_save_transaction(
                    host=host,
                    session_file=session_file,
                    session_id=session_id,
                    after_line=int(recovered_match.group(2)),
                    until_line=int(recovered_match.group(3)),
                )
                recovered = find_published_save_transaction(
                    raw_dir=raw_dir,
                    host=host,
                    session_file=session_file,
                    session_id=session_id,
                    after_line=recovered_transaction.after_line,
                )
                if (
                    recovered is None
                    or recovered.transaction != recovered_transaction
                    or recovered_transaction.until_line != after_line
                ):
                    return {}, "recovered_receipt_not_found"
                transactions.append(
                    _durable_receipt_chunk(
                        transaction=recovered_transaction,
                        receipt_path=recovered.path,
                        raw_dir=raw_dir,
                    )
                )
            for result in results_value:
                raw_id = str(result.get("raw_id") or result.get("saved") or "")
                match = re.fullmatch(
                    rf"save-{re.escape(host)}-([0-9a-f]{{24}})-from(\d+)-to(\d+)\.md",
                    raw_id,
                )
                if match is None:
                    return {}, "save_chunk_identity_invalid"
                transaction = make_save_transaction(
                    host=host,
                    session_file=session_file,
                    session_id=session_id,
                    after_line=int(match.group(2)),
                    until_line=int(match.group(3)),
                )
                if (
                    transaction.session_key != match.group(1)
                    or transaction.after_line != expected_after
                ):
                    return {}, "save_chunks_not_contiguous"
                receipt_path = validate_published_save_receipt(
                    raw_dir=raw_dir,
                    save_result=result,
                    expected=transaction,
                )
                transactions.append(
                    _durable_receipt_chunk(
                        transaction=transaction,
                        receipt_path=receipt_path,
                        raw_dir=raw_dir,
                    )
                )
                expected_after = transaction.until_line
            if expected_after != until_line:
                return {}, "save_chunks_terminal_mismatch"
        elif status == "recovered":
            recovered = save_output.get("recovered_save")
            if not isinstance(recovered, Mapping):
                return {}, "recovered_receipt_missing"
            identity = str(recovered.get("idempotency_key") or "")
            match = re.fullmatch(
                rf"{re.escape(host)}-([0-9a-f]{{24}})-from(\d+)-to(\d+)",
                identity,
            )
            if match is None:
                return {}, "recovered_receipt_identity_invalid"
            if match.group(1) != save_session_key(
                host=host,
                session_file=session_file,
                session_id=session_id,
            ):
                return {}, "recovered_receipt_session_mismatch"
            published = find_published_save_transaction(
                raw_dir=raw_dir,
                host=host,
                session_file=session_file,
                session_id=session_id,
                after_line=int(match.group(2)),
            )
            if (
                published is None
                or published.transaction.idempotency_key != identity
                or published.transaction.until_line != int(match.group(3))
            ):
                return {}, "recovered_receipt_not_found"
            transaction = published.transaction
            transactions = [
                _durable_receipt_chunk(
                    transaction=transaction,
                    receipt_path=published.path,
                    raw_dir=raw_dir,
                )
            ]
        else:
            return {}, "save_status_not_publishable"
    except (OSError, TypeError, ValueError):
        return {}, "save_receipt_validation_failed"
    receipt = {
        "status": status,
        "session_key": transactions[0]["session_key"],
        "after_line": transactions[0]["after_line"],
        "until_line": transactions[-1]["until_line"],
        "chunks": transactions,
    }
    receipt["receipt_manifest_sha256"] = _canonical_sha(receipt)
    return receipt, ""


def _durable_receipt_chunk(
    *,
    transaction: Any,
    receipt_path: Path,
    raw_dir: Path,
) -> dict[str, Any]:
    raw_id = f"save-{transaction.idempotency_key}.md"
    store = RawStore(raw_dir, mode="v2")
    unit = store.resolve(raw_id)
    if unit is not None:
        logical = store.read_bytes(unit)
        return {
            "idempotency_key": transaction.idempotency_key,
            "session_key": transaction.session_key,
            "after_line": transaction.after_line,
            "until_line": transaction.until_line,
            "raw_id": raw_id,
            "raw_dir": str(raw_dir.resolve(strict=False)),
            "storage": "segment-v2",
            "segment_storage": unit.storage,
            "commit": unit.commit.to_dict() if unit.commit is not None else None,
            "offset": unit.offset,
            "length": unit.length,
            "logical_sha256": hashlib.sha256(logical).hexdigest(),
            "source_interval_sha256": hashlib.sha256(logical).hexdigest(),
            "path": str(unit.path.resolve(strict=False)),
        }
    content = receipt_path.read_text(encoding="utf-8")
    parsed = parse_save_transaction_receipt(content)
    if parsed is None or parsed.transaction != transaction:
        raise ValueError("legacy receipt is not transaction-bound")
    logical = content.encode("utf-8")
    return {
        "idempotency_key": transaction.idempotency_key,
        "session_key": transaction.session_key,
        "after_line": transaction.after_line,
        "until_line": transaction.until_line,
        "raw_id": raw_id,
        "raw_dir": str(raw_dir.resolve(strict=False)),
        "storage": "legacy_file",
        "offset": 0,
        "length": len(logical),
        "logical_sha256": hashlib.sha256(logical).hexdigest(),
        # Legacy Raw wraps the source records, so the exact saved source
        # interval cannot be reconstructed.  It is intentionally ineligible
        # for answer-authority capture.
        "source_interval_sha256": "",
        "path": str(receipt_path.resolve(strict=False)),
    }


def _receipt_error(
    receipt: Mapping[str, Any],
    *,
    host: str,
    session_file: Path,
    session_id: str,
    user_line: int,
    assistant_line: int,
    require_live_source: bool = True,
) -> str:
    if not isinstance(receipt, Mapping):
        return "missing_save_receipt"
    observed_manifest = receipt.get("receipt_manifest_sha256")
    unsigned_receipt = {
        key: value
        for key, value in receipt.items()
        if key != "receipt_manifest_sha256"
    }
    if not _valid_sha(observed_manifest) or observed_manifest != _canonical_sha(
        unsigned_receipt
    ):
        return "save_receipt_manifest_mismatch"
    try:
        after_line = int(receipt["after_line"])
        until_line = int(receipt["until_line"])
    except (KeyError, TypeError, ValueError):
        return "save_receipt_shape_invalid"
    expected_key = save_session_key(
        host=host, session_file=session_file, session_id=session_id
    )
    if receipt.get("session_key") != expected_key:
        return "save_receipt_session_mismatch"
    if not (after_line < user_line <= assistant_line <= until_line):
        return "turn_outside_save_receipt"
    chunks = receipt.get("chunks")
    if not isinstance(chunks, list) or not chunks:
        return "save_receipt_chunks_missing"
    cursor = after_line
    for chunk in chunks:
        if not isinstance(chunk, Mapping):
            return "save_receipt_chunk_invalid"
        chunk_after = chunk.get("after_line")
        chunk_until = chunk.get("until_line")
        if (
            not isinstance(chunk_after, int)
            or isinstance(chunk_after, bool)
            or not isinstance(chunk_until, int)
            or isinstance(chunk_until, bool)
            or chunk_after != cursor
            or chunk_until <= chunk_after
            or chunk.get("session_key") != expected_key
        ):
            return "save_receipt_chunks_not_contiguous"
        try:
            expected = make_save_transaction(
                host=host,
                session_file=session_file,
                session_id=session_id,
                after_line=chunk_after,
                until_line=chunk_until,
            )
        except ValueError:
            return "save_receipt_chunk_invalid"
        if (
            chunk.get("idempotency_key") != expected.idempotency_key
            or chunk.get("raw_id") != f"save-{expected.idempotency_key}.md"
            or _durable_receipt_chunk_error(chunk, expected=expected)
        ):
            return "save_receipt_chunk_invalid"
        if chunk.get("storage") != "segment-v2":
            return "save_receipt_source_binding_unavailable"
        source_sha = chunk.get("source_interval_sha256")
        if not _valid_sha(source_sha) or source_sha != chunk.get("logical_sha256"):
            return "save_receipt_source_binding_missing"
        if require_live_source:
            try:
                source_interval = copy_source_interval(
                    session_file,
                    after_line=chunk_after,
                    until_line=chunk_until,
                )
            except (OSError, ValueError):
                return "save_receipt_source_interval_unavailable"
            if hashlib.sha256(source_interval).hexdigest() != source_sha:
                return "save_receipt_source_interval_mismatch"
        cursor = chunk_until
    if cursor != until_line:
        return "turn_not_bound_to_exact_save_chain"
    return ""


def _receipt_source_slice(
    receipt: Mapping[str, Any], *, start_line: int, end_line: int
) -> tuple[bytes, str]:
    """Restore one exact line interval from immutable Raw v2 chunks."""

    chunks = receipt.get("chunks")
    if not isinstance(chunks, list) or start_line <= 0 or end_line < start_line:
        return b"", "save_receipt_chunks_missing"
    selected: list[bytes] = []
    expected_line = start_line
    for chunk in chunks:
        if not isinstance(chunk, Mapping):
            return b"", "save_receipt_chunk_invalid"
        after_line = chunk.get("after_line")
        until_line = chunk.get("until_line")
        if (
            not isinstance(after_line, int)
            or isinstance(after_line, bool)
            or not isinstance(until_line, int)
            or isinstance(until_line, bool)
            or until_line < expected_line
            or after_line >= end_line
        ):
            continue
        try:
            store = RawStore(Path(str(chunk.get("raw_dir") or "")), mode="v2")
            unit = store.resolve(str(chunk.get("raw_id") or ""))
            if unit is None:
                return b"", "save_receipt_raw_missing"
            logical = store.read_bytes(unit)
        except (OSError, UnicodeError, ValueError):
            return b"", "save_receipt_raw_unavailable"
        lines = logical.splitlines(keepends=True)
        if len(lines) != until_line - after_line or any(
            not line.endswith(b"\n") for line in lines
        ):
            return b"", "save_receipt_raw_line_count_mismatch"
        slice_start = max(expected_line, after_line + 1)
        slice_end = min(end_line, until_line)
        if slice_start > slice_end:
            continue
        if slice_start != expected_line:
            return b"", "save_receipt_turn_gap"
        selected.extend(
            lines[slice_start - after_line - 1 : slice_end - after_line]
        )
        expected_line = slice_end + 1
        if expected_line > end_line:
            break
    if expected_line != end_line + 1:
        return b"", "save_receipt_turn_gap"
    return b"".join(selected), ""


def _durable_receipt_chunk_error(
    chunk: Mapping[str, Any],
    *,
    expected: Any,
) -> str:
    raw_id = str(chunk.get("raw_id") or "")
    raw_dir = Path(str(chunk.get("raw_dir") or ""))
    expected_sha = chunk.get("logical_sha256")
    expected_length = chunk.get("length")
    if (
        not raw_id
        or not str(raw_dir)
        or chunk.get("storage") != "segment-v2"
        or chunk.get("segment_storage") not in {"segment_open", "segment_sealed"}
        or not isinstance(chunk.get("commit"), Mapping)
        or not _valid_sha(expected_sha)
        or not isinstance(expected_length, int)
        or isinstance(expected_length, bool)
        or expected_length <= 0
    ):
        return "shape_invalid"
    try:
        store = RawStore(raw_dir, mode="v2")
        unit = store.resolve(raw_id)
        if (
            unit is None
            or not unit.is_segment
            or unit.commit is None
            or unit.storage != chunk.get("segment_storage")
            or unit.commit.to_dict() != chunk.get("commit")
            or unit.offset != chunk.get("offset")
            or unit.length != expected_length
            or unit.path.resolve(strict=False)
            != Path(str(chunk.get("path") or "")).resolve(strict=False)
        ):
            return "segment_identity_mismatch"
        logical = store.read_bytes(unit)
        if (
            unit.commit.host != expected.host
            or unit.commit.session_key != expected.session_key
            or unit.commit.after_line != expected.after_line
            or unit.commit.until_line != expected.until_line
            or unit.commit.idempotency_key != expected.idempotency_key
        ):
            return "transaction_mismatch"
    except (OSError, UnicodeError, ValueError):
        return "unavailable"
    if (
        len(logical) != expected_length
        or hashlib.sha256(logical).hexdigest() != expected_sha
    ):
        return "logical_digest_mismatch"
    return ""


def _manifest_digest_valid(manifest: Mapping[str, Any]) -> bool:
    observed = manifest.get("manifest_sha256")
    unsigned = {
        str(key): value
        for key, value in manifest.items()
        if key != "manifest_sha256"
    }
    return _valid_sha(observed) and observed == _canonical_sha(unsigned)


def _identity_error(identity: Mapping[str, Any], required: Sequence[str]) -> str:
    for key in required:
        value = identity.get(key)
        if not isinstance(value, str) or not value.strip():
            return f"missing_{key}"
        if key.endswith("_sha256") and not _valid_sha(value):
            return f"invalid_{key}"
    return ""


def append_answer_review_receipt(
    *,
    kind: str,
    payload: Mapping[str, Any],
    reviewer_kind: str,
    reviewed_at: str,
    protocol_sha256: str,
    ledger_file: Path = ANSWER_REVIEW_LEDGER,
) -> dict[str, Any]:
    """Append one hash-chained human review receipt."""

    if (
        kind not in {"gold_entry_review", "scorer_calibration_case_review"}
        or reviewer_kind not in {"human_reviewer", "expert_adjudicator"}
        or not _strict_utc(reviewed_at)
        or not _valid_sha(protocol_sha256)
    ):
        raise ValueError("invalid answer review receipt authority")
    with sidecar_exclusive_lock(ledger_file):
        rows, error = _validated_review_receipts(ledger_file)
        if error:
            raise ValueError(error)
        previous = str(rows[-1]["receipt_sha256"]) if rows else "0" * 64
        row = {
            "schema_version": 1,
            "kind": kind,
            "reviewer_kind": reviewer_kind,
            "reviewed_at": _strict_utc(reviewed_at),
            "protocol_sha256": protocol_sha256,
            "payload": dict(payload),
            "payload_sha256": _canonical_sha(dict(payload)),
            "previous_receipt_sha256": previous,
        }
        row["receipt_sha256"] = _canonical_sha(row)
        append_jsonl_durable(ledger_file, [row], sort_keys=True)
        return row


def _validated_review_receipts(
    ledger_file: Path,
) -> tuple[list[dict[str, Any]], str]:
    rows = _read_jsonl(ledger_file)
    previous = "0" * 64
    seen: set[str] = set()
    for row in rows:
        receipt_sha = row.get("receipt_sha256")
        payload = row.get("payload")
        unsigned = {
            key: value for key, value in row.items() if key != "receipt_sha256"
        }
        if (
            row.get("schema_version") != 1
            or row.get("kind")
            not in {"gold_entry_review", "scorer_calibration_case_review"}
            or row.get("reviewer_kind")
            not in {"human_reviewer", "expert_adjudicator"}
            or not _strict_utc(row.get("reviewed_at"))
            or not _valid_sha(row.get("protocol_sha256"))
            or not isinstance(payload, Mapping)
            or row.get("payload_sha256") != _canonical_sha(dict(payload))
            or row.get("previous_receipt_sha256") != previous
            or receipt_sha != _canonical_sha(unsigned)
            or receipt_sha in seen
        ):
            return [], "review_ledger_chain_invalid"
        seen.add(str(receipt_sha))
        previous = str(receipt_sha)
    return rows, ""


def _review_receipt_error(
    *,
    receipt_sha256: object,
    expected_kind: str,
    expected_payload: Mapping[str, Any],
    expected_protocol_sha256: str,
    ledger_file: Path,
    frozen_at: str,
    expected_reviewed_at: str = "",
) -> str:
    rows, error = _validated_review_receipts(ledger_file)
    if error:
        return error
    matching = [
        row for row in rows if row.get("receipt_sha256") == receipt_sha256
    ]
    if len(matching) != 1:
        return "review_receipt_missing"
    row = matching[0]
    reviewed_at = _strict_utc(row.get("reviewed_at"))
    normalized_frozen_at = _strict_utc(frozen_at)
    if (
        row.get("kind") != expected_kind
        or row.get("payload") != dict(expected_payload)
        or row.get("protocol_sha256") != expected_protocol_sha256
        or not reviewed_at
        or not normalized_frozen_at
        or (expected_reviewed_at and reviewed_at != _strict_utc(expected_reviewed_at))
        or datetime.fromisoformat(reviewed_at.replace("Z", "+00:00"))
        > datetime.fromisoformat(normalized_frozen_at.replace("Z", "+00:00"))
    ):
        return "review_receipt_binding_invalid"
    return ""


def append_answer_execution_receipt(
    *,
    kind: str,
    adapter_identity_sha256: str,
    parent_run_id: str,
    input_payload: Mapping[str, Any],
    output_payload: Mapping[str, Any],
    started_at: str,
    completed_at: str,
    ledger_file: Path = ANSWER_EXECUTION_LEDGER,
) -> dict[str, Any]:
    """Append a hash-chained adapter execution receipt without answer bodies."""

    allowed = {
        "answer_runner_call",
        "answer_scorer_call",
        "calibration_scorer_call",
        "field_environment_replay",
    }
    normalized_started = _strict_utc(started_at)
    normalized_completed = _strict_utc(completed_at)
    if (
        kind not in allowed
        or not _valid_sha(adapter_identity_sha256)
        or not _valid_sha(parent_run_id)
        or not normalized_started
        or not normalized_completed
        or datetime.fromisoformat(normalized_started.replace("Z", "+00:00"))
        > datetime.fromisoformat(normalized_completed.replace("Z", "+00:00"))
    ):
        raise ValueError("invalid answer execution receipt authority")
    with sidecar_exclusive_lock(ledger_file):
        rows, error = _validated_execution_receipts(ledger_file)
        if error:
            raise ValueError(error)
        for existing in rows:
            if (
                existing.get("kind") == kind
                and existing.get("adapter_identity_sha256")
                == adapter_identity_sha256
                and existing.get("parent_run_id") == parent_run_id
                and existing.get("input_payload") == dict(input_payload)
                and existing.get("output_payload") == dict(output_payload)
            ):
                return existing
        previous = str(rows[-1]["receipt_sha256"]) if rows else "0" * 64
        row = {
            "schema_version": 1,
            "kind": kind,
            "adapter_identity_sha256": adapter_identity_sha256,
            "parent_run_id": parent_run_id,
            "input_payload": dict(input_payload),
            "input_sha256": _canonical_sha(dict(input_payload)),
            "output_payload": dict(output_payload),
            "output_sha256": _canonical_sha(dict(output_payload)),
            "started_at": normalized_started,
            "completed_at": normalized_completed,
            "previous_receipt_sha256": previous,
        }
        row["receipt_sha256"] = _canonical_sha(row)
        append_jsonl_durable(ledger_file, [row], sort_keys=True)
        return row


def _validated_execution_receipts(
    ledger_file: Path,
) -> tuple[list[dict[str, Any]], str]:
    rows = _read_jsonl(ledger_file)
    previous = "0" * 64
    seen: set[str] = set()
    allowed = {
        "answer_runner_call",
        "answer_scorer_call",
        "calibration_scorer_call",
        "field_environment_replay",
    }
    for row in rows:
        receipt_sha = row.get("receipt_sha256")
        input_payload = row.get("input_payload")
        output_payload = row.get("output_payload")
        unsigned = {
            key: value for key, value in row.items() if key != "receipt_sha256"
        }
        started_at = _strict_utc(row.get("started_at"))
        completed_at = _strict_utc(row.get("completed_at"))
        if (
            row.get("schema_version") != 1
            or row.get("kind") not in allowed
            or not _valid_sha(row.get("adapter_identity_sha256"))
            or not _valid_sha(row.get("parent_run_id"))
            or not isinstance(input_payload, Mapping)
            or not isinstance(output_payload, Mapping)
            or row.get("input_sha256") != _canonical_sha(dict(input_payload))
            or row.get("output_sha256") != _canonical_sha(dict(output_payload))
            or not started_at
            or not completed_at
            or datetime.fromisoformat(started_at.replace("Z", "+00:00"))
            > datetime.fromisoformat(completed_at.replace("Z", "+00:00"))
            or row.get("previous_receipt_sha256") != previous
            or receipt_sha != _canonical_sha(unsigned)
            or receipt_sha in seen
        ):
            return [], "execution_ledger_chain_invalid"
        seen.add(str(receipt_sha))
        previous = str(receipt_sha)
    return rows, ""


def _execution_receipt_error(
    *,
    receipt_sha256: object,
    expected_kind: str,
    expected_adapter_identity_sha256: str,
    expected_parent_run_id: str,
    expected_input_payload: Mapping[str, Any],
    expected_output_payload: Mapping[str, Any],
    ledger_file: Path,
    completed_before: str = "",
) -> str:
    rows, error = _validated_execution_receipts(ledger_file)
    if error:
        return error
    matching = [
        row for row in rows if row.get("receipt_sha256") == receipt_sha256
    ]
    if len(matching) != 1:
        return "execution_receipt_missing"
    row = matching[0]
    normalized_before = _strict_utc(completed_before) if completed_before else ""
    completed_at = _strict_utc(row.get("completed_at"))
    if (
        row.get("kind") != expected_kind
        or row.get("adapter_identity_sha256")
        != expected_adapter_identity_sha256
        or row.get("parent_run_id") != expected_parent_run_id
        or row.get("input_payload") != dict(expected_input_payload)
        or row.get("output_payload") != dict(expected_output_payload)
        or (normalized_before and completed_at > normalized_before)
    ):
        return "execution_receipt_binding_invalid"
    return ""


def _page_hashes(
    page_ids: Sequence[str], source: Mapping[str, Any]
) -> tuple[dict[str, str], dict[str, str], dict[str, str]]:
    known: dict[str, str] = {}
    historical_uids: dict[str, str] = {}
    context_items_value = source.get("context_items", [])
    context_items = (
        context_items_value if isinstance(context_items_value, list) else []
    )
    malformed_context_items = not isinstance(context_items_value, list)
    for item in context_items:
        if not isinstance(item, Mapping):
            continue
        page_id = str(item.get("page_id") or "")
        digest = item.get("content_sha256") or item.get("page_sha256")
        if page_id and _valid_sha(digest):
            known[page_id] = str(digest)
        page_uid = str(item.get("page_uid") or "")
        if page_id and page_uid:
            historical_uids[page_id] = page_uid
    hashes: dict[str, str] = {}
    uids: dict[str, str] = {}
    errors: dict[str, str] = {}
    try:
        from chronovisor.ingest.page_registry import PageRegistry

        registry = PageRegistry(CHRONOVISOR_ROOT)
    except Exception:
        registry = None
    for page_id in page_ids:
        path = find_page(page_id)
        if path is not None:
            try:
                current = hashlib.sha256(path.read_bytes()).hexdigest()
            except OSError:
                current = ""
            # Authority requires the recall-time binding. Current bytes are
            # only an equality check, never a historical-hash fallback.
            if current and page_id in known and known[page_id] == current:
                hashes[page_id] = current
        if page_id not in known:
            errors[page_id] = (
                "malformed_recall_context_items"
                if malformed_context_items
                else "missing_recall_time_content_sha256"
            )
        elif page_id not in hashes:
            errors[page_id] = "page_content_changed_since_recall"
        uid = ""
        if registry is not None:
            try:
                uid = str((registry.resolve(page_id) or {}).get("uid") or "")
            except Exception:
                uid = ""
        historical_uid = historical_uids.get(page_id, "")
        if historical_uid:
            uids[page_id] = historical_uid
            if uid and uid != historical_uid:
                errors[page_id] = "page_uid_changed_since_recall"
        elif uid:
            uids[page_id] = uid
            errors[page_id] = "missing_recall_time_page_uid"
        else:
            errors[page_id] = "missing_recall_time_page_uid"
    return hashes, uids, errors


def _context_receipt_error(
    value: object,
    *,
    expected_page_bindings: Sequence[Mapping[str, Any]] | None = None,
) -> str:
    if not isinstance(value, Mapping):
        return "missing_exact_context_receipt"
    rendered = value.get("rendered_context")
    bindings = value.get("page_bindings")
    receipt_sha = value.get("receipt_sha256")
    unsigned = {key: item for key, item in value.items() if key != "receipt_sha256"}
    if (
        value.get("schema_version") != 1
        or value.get("renderer_protocol") != "recall-result-context-v1"
        or not isinstance(value.get("context_style"), str)
        or not isinstance(rendered, str)
        or value.get("rendered_context_sha256") != _sha_text(rendered)
        or not isinstance(bindings, list)
        or any(not isinstance(binding, Mapping) for binding in bindings)
        or receipt_sha != _canonical_sha(unsigned)
    ):
        return "exact_context_receipt_invalid"
    if expected_page_bindings is not None and [dict(item) for item in bindings] != [
        dict(item) for item in expected_page_bindings
    ]:
        return "exact_context_receipt_page_binding_mismatch"
    return ""


def _raw_slice(path: Path, start_line: int, end_line: int) -> tuple[bytes, str]:
    try:
        lines = path.read_bytes().splitlines(keepends=True)
    except OSError:
        return b"", ""
    if start_line <= 0 or end_line < start_line or end_line > len(lines):
        return b"", ""
    raw = b"".join(lines[start_line - 1 : end_line])
    return raw, hashlib.sha256(raw).hexdigest() if raw else ""


def _extract(host: str, session_file: Path) -> Any:
    if host == "codex":
        from chronovisor.hosts.codex_record import extract_transcript_slice
    elif host == "claude-code":
        from chronovisor.hosts.claude_code_record import extract_transcript_slice
    else:
        raise ValueError(f"unsupported answer capture host: {host}")
    return extract_transcript_slice(session_file, after_line=0)


def _resolve_session_file(
    host: str,
    *,
    session_file: str | Path | None,
    session_id: str,
    cwd: str,
    hints: Mapping[str, str],
) -> Path:
    explicit = session_file or hints.get("session_file") or hints.get("transcript_path")
    if not explicit:
        raise ValueError("exact save-confirmed session_file is required")
    resolved = Path(explicit).expanduser().resolve(strict=False)
    if not resolved.exists():
        raise ValueError("save-confirmed session_file is unavailable")
    return resolved


def _capture_key(host: str, session_file: Path, session_id: str) -> str:
    return _sha_text(f"{host}:{session_id}:{session_file.resolve(strict=False)}")


def _read_cursor(path: Path, key: str) -> int:
    try:
        value = read_sealed_json(path)
    except DurableStateError:
        return 0
    cursors = value.get("cursors")
    if not isinstance(cursors, Mapping):
        return 0
    line = cursors.get(key)
    return max(0, line) if isinstance(line, int) and not isinstance(line, bool) else 0


def _write_cursor(path: Path, key: str, line: int) -> None:
    try:
        current = read_sealed_json(path)
    except DurableStateError:
        current = {}
    cursors_value = current.get("cursors")
    cursors = dict(cursors_value) if isinstance(cursors_value, Mapping) else {}
    prior = cursors.get(key)
    previous = prior if isinstance(prior, int) and not isinstance(prior, bool) else 0
    cursors[key] = max(previous, max(0, int(line)))
    write_sealed_json(
        path,
        {"schema_version": 1, "cursors": cursors},
        backup=True,
    )


def capture_session_answer_episodes(
    *,
    host: str,
    session_file: Path,
    session_id_hint: str = "",
    cwd_hint: str = "",
    episode_file: Path = ANSWER_EPISODE_LEDGER,
    cursor_file: Path = ANSWER_CAPTURE_CURSOR,
    recall_log_file: Path = RECALL_LOG_FILE,
    pull_log_file: Path = RECALL_PULL_LOG_FILE,
    save_output: Mapping[str, Any] | None = None,
    raw_dir: Path | None = None,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Capture every newly complete turn after a durable host transcript save."""

    session_file = session_file.resolve(strict=False)
    transcript = _extract(host, session_file)
    canonical_session = str(transcript.session_id or "").strip()
    declared_session = str((save_output or {}).get("session_id") or "").strip()
    if (
        not canonical_session
        or not declared_session
        or canonical_session != declared_session
        or (session_id_hint and session_id_hint != canonical_session)
    ):
        return {
            "status": "held",
            "reason": "save_transcript_session_mismatch",
            "captured": 0,
        }
    verified_receipt, receipt_error = _verified_save_receipt(
        host=host,
        save_output=save_output or {},
        session_file=session_file,
        session_id=canonical_session,
        raw_dir=raw_dir or CHRONOVISOR_ROOT / "raw",
    )
    if receipt_error:
        return {"status": "held", "reason": receipt_error, "captured": 0}
    turns = complete_turns(
        transcript.records,
        host=host,
        session_file=session_file.resolve(strict=False),
        session_id=canonical_session,
        cwd=str(transcript.cwd or cwd_hint or ""),
    )
    key = _capture_key(host, session_file, canonical_session)
    with sidecar_exclusive_lock(cursor_file):
        cursor = _read_cursor(cursor_file, key)
        existing_rows = _read_jsonl(episode_file)
        latest_by_id = {
            str(row.get("episode_id") or ""): row
            for row in existing_rows
            if str(row.get("episode_id") or "")
        }
        recall_rows = _read_jsonl(recall_log_file)
        pull_rows = _read_jsonl(pull_log_file)
        joined = join_used_recall_episodes(recall_rows, pull_rows)
        episodes_by_key = {
            (str(row.get("decision_id") or ""), str(row.get("session_id") or "")): row
            for row in joined["episodes"]
        }
        captured: list[dict[str, Any]] = []
        advance_line = cursor
        blocked = False
        for turn in turns:
            if turn.assistant_line <= cursor:
                continue
            source = source_recall_record(turn, log_file=recall_log_file)
            decision_id = str((source or {}).get("decision_id") or "")
            used = episodes_by_key.get((decision_id, canonical_session))
            injected = page_ids_from_record(source or {})
            used_pages = [
                str(page_id)
                for page_id in (used or {}).get("page_ids", [])
                if isinstance(page_id, str) and page_id
            ]
            all_pages = list(dict.fromkeys([*injected, *used_pages]))
            hashes, uids, uid_errors = _page_hashes(all_pages, source or {})
            context_receipt_value = (source or {}).get("context_receipt")
            context_receipt = (
                dict(context_receipt_value)
                if isinstance(context_receipt_value, Mapping)
                else {}
            )
            context_receipt_error = _context_receipt_error(context_receipt)
            _raw, raw_sha = _raw_slice(
                session_file, turn.user_line, turn.assistant_line
            )
            stable = {
                "host": host,
                "session_id": canonical_session,
                "session_file": str(session_file),
                "user_line": turn.user_line,
                "assistant_line": turn.assistant_line,
                "prompt_sha256": turn.prompt_hash,
                "raw_sha256": raw_sha,
            }
            episode_id = _canonical_sha(stable)[:32]
            prior = latest_by_id.get(episode_id)
            prior_receipt = (
                prior.get("raw_ref", {}).get("save_receipt", {})
                if isinstance(prior, Mapping)
                and isinstance(prior.get("raw_ref"), Mapping)
                else {}
            )
            selected_receipt = verified_receipt
            selected_receipt_error = _receipt_error(
                selected_receipt,
                host=host,
                session_file=session_file,
                session_id=canonical_session,
                user_line=turn.user_line,
                assistant_line=turn.assistant_line,
            )
            if selected_receipt_error and prior_receipt:
                selected_receipt = dict(prior_receipt)
                selected_receipt_error = _receipt_error(
                    selected_receipt,
                    host=host,
                    session_file=session_file,
                    session_id=canonical_session,
                    user_line=turn.user_line,
                    assistant_line=turn.assistant_line,
                )
            exact_used_subset = bool(
                used is not None
                and used_pages
                and set(used_pages).issubset(injected)
                and all(page_id in hashes for page_id in used_pages)
            )
            binding_reasons: list[str] = []
            if not canonical_session:
                binding_reasons.append("missing_canonical_session")
            if source is None:
                binding_reasons.append("missing_or_ambiguous_recall")
            elif not decision_id:
                binding_reasons.append("missing_decision_id")
            if not raw_sha:
                binding_reasons.append("missing_raw_binding")
            if selected_receipt_error:
                binding_reasons.append(selected_receipt_error)
            assistant_observed_at = _strict_utc(
                getattr(turn, "assistant_timestamp", "")
            )
            if not assistant_observed_at:
                binding_reasons.append("missing_actual_assistant_timestamp")
            if used_pages and not exact_used_subset:
                binding_reasons.append("used_subset_or_hash_mismatch")
            if uid_errors:
                binding_reasons.append("historical_page_uid_mismatch")
            if context_receipt_error:
                binding_reasons.append(context_receipt_error)
            source_generator = {
                "model": str((source or {}).get("model") or ""),
                "system_sha256": str((source or {}).get("system_sha256") or ""),
                "sampler_sha256": str((source or {}).get("sampler_sha256") or ""),
            }
            row = {
                "schema_version": ANSWER_EPISODE_SCHEMA_VERSION,
                "episode_id": episode_id,
                "captured_at": datetime.now(UTC).isoformat(),
                "observed_at": assistant_observed_at,
                "host": host,
                "canonical_session_id": canonical_session,
                "session_hash": _sha_text(f"{host}:{canonical_session}")[:16]
                if canonical_session
                else "",
                "turn_ref": {
                    **turn.turn_ref(),
                    "session_file": str(session_file),
                },
                "prompt_sha256": turn.prompt_hash,
                "answer_sha256": _sha_text(turn.assistant_response),
                "answer_chars": len(turn.assistant_response),
                "raw_ref": {
                    "transcript_path": str(session_file),
                    "start_line": turn.user_line,
                    "end_line": turn.assistant_line,
                    "transcript_slice_sha256": raw_sha,
                    "save_receipt": dict(selected_receipt),
                },
                "decision_id": decision_id,
                "injected_page_ids": injected,
                "used_page_ids": used_pages,
                "page_content_sha256": hashes,
                "page_uids": uids,
                "page_uid_binding_errors": uid_errors,
                "context_receipt": context_receipt,
                "exact_used_subset": exact_used_subset,
                "policy_identity": {
                    "policy_sha256": str((source or {}).get("policy_sha256") or ""),
                    "policy_version": str((source or {}).get("policy_version") or ""),
                    "generator": source_generator,
                },
                "production_replayable": all(
                    _valid_sha(source_generator[key])
                    for key in ("system_sha256", "sampler_sha256")
                )
                and bool(source_generator["model"]),
                "binding_status": "verified" if not binding_reasons else "unknown",
                "binding_reasons": binding_reasons,
                "join_event_ids": list((used or {}).get("event_ids", [])),
                "split": "unassigned",
            }
            row["episode_sha256"] = _canonical_sha(row)
            prior_verified = bool(
                isinstance(prior, Mapping)
                and prior.get("binding_status") == "verified"
            )
            current_verified = row["binding_status"] == "verified"
            if prior is None or (not prior_verified and current_verified):
                captured.append(row)
                latest_by_id[episode_id] = row
            if not blocked:
                if prior_verified or current_verified:
                    advance_line = int(turn.assistant_line)
                else:
                    blocked = True
        if not dry_run:
            append_jsonl_durable(episode_file, captured, sort_keys=True)
            _write_cursor(cursor_file, key, advance_line)
    return {
        "status": "ok",
        "complete_turns": len(turns),
        "captured": len(captured),
        "cursor_line": advance_line,
        "verified": sum(row["binding_status"] == "verified" for row in captured),
        "unknown": sum(row["binding_status"] != "verified" for row in captured),
        "join": {key: value for key, value in joined.items() if key != "episodes"},
        "episode_file": str(episode_file),
    }


def _load_bound_turn(episode: Mapping[str, Any]) -> tuple[Any | None, str]:
    raw_ref = episode.get("raw_ref")
    if not isinstance(raw_ref, Mapping):
        return None, "missing_raw_ref"
    path = Path(str(raw_ref.get("transcript_path") or ""))
    start = raw_ref.get("start_line")
    end = raw_ref.get("end_line")
    if not isinstance(start, int) or not isinstance(end, int):
        return None, "invalid_raw_ref"
    receipt = raw_ref.get("save_receipt", {})
    receipt_error = _receipt_error(
        receipt,
        host=str(episode.get("host") or ""),
        session_file=path,
        session_id=str(episode.get("canonical_session_id") or ""),
        user_line=start,
        assistant_line=end,
        require_live_source=False,
    )
    if receipt_error:
        return None, receipt_error
    raw, raw_error = _receipt_source_slice(
        receipt if isinstance(receipt, Mapping) else {},
        start_line=start,
        end_line=end,
    )
    if raw_error:
        return None, raw_error
    digest = hashlib.sha256(raw).hexdigest() if raw else ""
    if not digest or digest != raw_ref.get("transcript_slice_sha256"):
        return None, "raw_digest_mismatch"
    # Host extractors operate on paths.  Rebuild the exact turn at its original
    # line offsets from sealed Raw bytes; the mutable host transcript path is
    # provenance only after capture.
    descriptor, temporary_name = tempfile.mkstemp(prefix="chronovisor-answer-", suffix=".jsonl")
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(b"\n" * (start - 1))
            handle.write(raw)
        transcript = _extract(
            str(episode.get("host") or ""), Path(temporary_name)
        )
    except (OSError, ValueError, UnicodeError):
        return None, "raw_unavailable"
    finally:
        Path(temporary_name).unlink(missing_ok=True)
    turns = complete_turns(
        transcript.records,
        host=str(episode.get("host") or ""),
        session_file=path,
        session_id=str(episode.get("canonical_session_id") or ""),
        cwd=str(transcript.cwd or ""),
    )
    matching = [
        turn
        for turn in turns
        if turn.user_line == start
        and turn.assistant_line == end
        and turn.prompt_hash == episode.get("prompt_sha256")
        and _sha_text(turn.assistant_response) == episode.get("answer_sha256")
    ]
    return (matching[0], "") if len(matching) == 1 else (None, "turn_binding_mismatch")


def _context_for_episode(episode: Mapping[str, Any]) -> tuple[str, str]:
    hashes = episode.get("page_content_sha256")
    uids = episode.get("page_uids")
    page_ids = episode.get("injected_page_ids")
    if (
        not isinstance(hashes, Mapping)
        or not isinstance(uids, Mapping)
        or not isinstance(page_ids, list)
    ):
        return "", "missing_context_binding"
    expected_bindings: list[dict[str, str]] = []
    for page_id in page_ids:
        if (
            not isinstance(page_id, str)
            or not _valid_sha(hashes.get(page_id))
            or not isinstance(uids.get(page_id), str)
            or not str(uids.get(page_id) or "")
        ):
            return "", "missing_page_hash"
        expected_bindings.append(
            {
                "page_id": page_id,
                "page_uid": str(uids[page_id]),
                "content_sha256": str(hashes[page_id]),
            }
        )
    receipt = episode.get("context_receipt")
    error = _context_receipt_error(
        receipt,
        expected_page_bindings=expected_bindings,
    )
    if error:
        return "", error
    return str(receipt.get("rendered_context") or ""), ""  # type: ignore[union-attr]


def _latest_episode_rows(path: Path) -> list[dict[str, Any]]:
    latest: dict[str, dict[str, Any]] = {}
    for row in _read_jsonl(path):
        episode_id = str(row.get("episode_id") or "")
        if episode_id:
            latest[episode_id] = row
    return [latest[key] for key in sorted(latest)]


def _episode_page_bindings(episode: Mapping[str, Any]) -> list[dict[str, str]]:
    page_ids = list(
        dict.fromkeys(
            str(value)
            for field in ("injected_page_ids", "used_page_ids")
            for value in episode.get(field, [])
            if isinstance(value, str) and value
        )
    )
    hashes = episode.get("page_content_sha256")
    uids = episode.get("page_uids")
    hash_map = hashes if isinstance(hashes, Mapping) else {}
    uid_map = uids if isinstance(uids, Mapping) else {}
    return [
        {
            "page_id": page_id,
            "page_uid": str(uid_map.get(page_id) or ""),
            "content_sha256": str(hash_map.get(page_id) or ""),
        }
        for page_id in page_ids
    ]


def _episode_manifest_entry(episode: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "episode_id": str(episode.get("episode_id") or ""),
        "episode_sha256": str(episode.get("episode_sha256") or ""),
        "observed_at": str(episode.get("observed_at") or ""),
        "session_hash": str(episode.get("session_hash") or ""),
        "query_sha256": str(episode.get("prompt_sha256") or ""),
        "page_bindings": _episode_page_bindings(episode),
    }


def _authority_eligible_episode_entry(
    episode: Mapping[str, Any],
) -> tuple[dict[str, Any] | None, str]:
    """Return the fixed authority population entry or a fail-closed reason."""

    if (
        episode.get("binding_status") != "verified"
        or episode.get("exact_used_subset") is not True
    ):
        return None, "not_authority_eligible"
    unsigned = {
        key: value for key, value in episode.items() if key != "episode_sha256"
    }
    entry = _episode_manifest_entry(episode)
    bindings = entry.get("page_bindings")
    if (
        episode.get("schema_version") != ANSWER_EPISODE_SCHEMA_VERSION
        or not _valid_sha(episode.get("episode_sha256"))
        or episode.get("episode_sha256") != _canonical_sha(unsigned)
        or not entry.get("episode_id")
        or not _strict_utc(entry.get("observed_at"))
        or not entry.get("session_hash")
        or not _valid_sha(entry.get("query_sha256"))
        or not isinstance(bindings, list)
        or not bindings
        or any(
            not isinstance(binding, Mapping)
            or not str(binding.get("page_id") or "")
            or not _valid_sha(binding.get("content_sha256"))
            for binding in bindings
        )
    ):
        return None, "authority_eligible_episode_invalid"
    return entry, ""


def _authority_eligible_entries(
    episode_file: Path,
) -> tuple[list[dict[str, Any]], str]:
    entries: list[dict[str, Any]] = []
    for episode in _latest_episode_rows(episode_file):
        entry, reason = _authority_eligible_episode_entry(episode)
        if reason == "not_authority_eligible":
            continue
        if reason:
            return [], reason
        if entry is not None:
            entries.append(entry)
    return sorted(entries, key=lambda row: str(row["episode_id"])), ""


def _assign_episode_splits(
    entries: Sequence[Mapping[str, Any]],
) -> dict[str, str]:
    """Assign the fixed authority split over connected episode identities."""

    parent: dict[str, str] = {}

    def find(value: str) -> str:
        parent.setdefault(value, value)
        while parent[value] != value:
            parent[value] = parent[parent[value]]
            value = parent[value]
        return value

    def union(left: str, right: str) -> None:
        left_root, right_root = find(left), find(right)
        if left_root != right_root:
            parent[max(left_root, right_root)] = min(left_root, right_root)

    row_nodes: dict[str, list[str]] = {}
    epochs: dict[str, float] = {}
    assignments: dict[str, str] = {}
    for entry in entries:
        episode_id = str(entry.get("episode_id") or "")
        observed = _strict_utc(entry.get("observed_at"))
        session = str(entry.get("session_hash") or "")
        query = str(entry.get("query_sha256") or "")
        if (
            not episode_id
            or not observed
            or not session
            or not _valid_sha(query)
            or not _valid_sha(entry.get("episode_sha256"))
        ):
            assignments[episode_id] = "unassigned"
            continue
        nodes = [f"session:{session}", f"query:{query}"]
        bindings = entry.get("page_bindings")
        if not isinstance(bindings, list):
            assignments[episode_id] = "unassigned"
            continue
        valid_bindings = True
        for binding in bindings:
            if not isinstance(binding, Mapping):
                valid_bindings = False
                break
            page_id = str(binding.get("page_id") or "")
            page_uid = str(binding.get("page_uid") or "")
            content_sha = str(binding.get("content_sha256") or "")
            if not page_id or not _valid_sha(content_sha):
                valid_bindings = False
                break
            # Page ID is always included. UID augments rather than replaces it,
            # joining UID-present rows to legacy UID-missing aliases.
            nodes.extend((f"page:{page_id}", f"content:{content_sha}"))
            if page_uid:
                nodes.append(f"uid:{page_uid}")
        if not valid_bindings:
            assignments[episode_id] = "unassigned"
            continue
        nodes = list(dict.fromkeys(nodes))
        for node in nodes:
            find(node)
        for node in nodes[1:]:
            union(nodes[0], node)
        row_nodes[episode_id] = nodes
        epochs[episode_id] = datetime.fromisoformat(
            observed.replace("Z", "+00:00")
        ).timestamp()

    components: dict[str, list[str]] = {}
    for episode_id, nodes in row_nodes.items():
        components.setdefault(find(nodes[0]), []).append(episode_id)
    ordered = sorted(
        components.values(),
        key=lambda ids: (max(epochs[value] for value in ids), min(ids)),
    )
    count = len(ordered)
    train_end = max(1, (count * 70 + 99) // 100) if count else 0
    holdout_end = max(train_end, (count * 90 + 99) // 100)
    for index, episode_ids in enumerate(ordered):
        split = (
            "train"
            if index < train_end
            else "holdout"
            if index < holdout_end
            else "locked-test"
        )
        for episode_id in episode_ids:
            assignments[episode_id] = split
    boundaries: list[float] = []
    for left, right in zip(ordered, ordered[1:], strict=False):
        if assignments[left[0]] != assignments[right[0]]:
            boundaries.append(min(epochs[value] for value in right))
    for episode_ids in ordered:
        if any(
            abs(epochs[episode_id] - boundary) < AUTHORITY_EMBARGO_SECONDS
            for episode_id in episode_ids
            for boundary in boundaries
        ):
            for episode_id in episode_ids:
                assignments[episode_id] = "embargo"
    return assignments


def write_answer_split_manifest(
    *,
    episode_file: Path = ANSWER_EPISODE_LEDGER,
    output_file: Path = ANSWER_SPLIT_MANIFEST,
) -> dict[str, Any]:
    entries, error = _authority_eligible_entries(episode_file)
    if error:
        raise ValueError(error)
    assignments = _assign_episode_splits(entries)
    manifest_entries = [
        {**entry, "split": assignments.get(str(entry["episode_id"]), "unassigned")}
        for entry in entries
    ]
    ledger_manifest_sha = manifest_sha256(
        [str(entry["episode_sha256"]) for entry in manifest_entries]
    )
    payload = {
        "schema_version": ANSWER_SPLIT_SCHEMA_VERSION,
        "artifact_kind": "answer-preregistered-split-manifest",
        "frozen_at": _now_utc(),
        "embargo_seconds": AUTHORITY_EMBARGO_SECONDS,
        "strategy": "connected-components-chronological-70-20-10",
        "component_keys": [
            "session_hash",
            "query_sha256",
            "page_id",
            "page_uid",
            "content_sha256",
        ],
        "episode_ledger_manifest_sha256": ledger_manifest_sha,
        "entries": manifest_entries,
    }
    payload["epoch_id"] = _split_epoch_id(payload)
    with sidecar_exclusive_lock(output_file):
        if output_file.exists():
            existing = validate_split_manifest(output_file)
            if (
                existing.get("passed") is True
                and existing.get("manifest", {}).get("epoch_id")
                == payload["epoch_id"]
            ):
                return dict(existing["manifest"])
            raise FileExistsError(
                "split manifest is an immutable epoch; choose a new epoch path"
            )
        return write_sealed_json(output_file, payload, backup=False)


def _split_epoch_id(manifest: Mapping[str, Any]) -> str:
    return _canonical_sha(
        {
            "artifact_kind": "answer-preregistered-split-manifest",
            "embargo_seconds": AUTHORITY_EMBARGO_SECONDS,
            "strategy": "connected-components-chronological-70-20-10",
            "component_keys": [
                "session_hash",
                "query_sha256",
                "page_id",
                "page_uid",
                "content_sha256",
            ],
            "episode_ledger_manifest_sha256": manifest.get(
                "episode_ledger_manifest_sha256"
            ),
            "entries": manifest.get("entries"),
        }
    )


def validate_split_manifest(value: Path | Mapping[str, Any]) -> dict[str, Any]:
    try:
        manifest = (
            read_sealed_json(value)
            if isinstance(value, Path)
            else verify_sealed_object(dict(value))
        )
    except (DurableStateError, TypeError, ValueError):
        return {"passed": False, "reason": "split_manifest_seal_invalid"}
    entries = manifest.get("entries")
    if (
        manifest.get("schema_version") != ANSWER_SPLIT_SCHEMA_VERSION
        or manifest.get("artifact_kind") != "answer-preregistered-split-manifest"
        or not _strict_utc(manifest.get("frozen_at"))
        or manifest.get("embargo_seconds") != AUTHORITY_EMBARGO_SECONDS
        or manifest.get("strategy")
        != "connected-components-chronological-70-20-10"
        or manifest.get("component_keys")
        != [
            "session_hash",
            "query_sha256",
            "page_id",
            "page_uid",
            "content_sha256",
        ]
        or manifest.get("epoch_id") != _split_epoch_id(manifest)
        or not isinstance(entries, list)
        or not entries
        or any(not isinstance(entry, Mapping) for entry in entries)
    ):
        return {"passed": False, "reason": "split_manifest_shape_invalid"}
    ids = [str(entry.get("episode_id") or "") for entry in entries]
    if not all(ids) or len(ids) != len(set(ids)):
        return {"passed": False, "reason": "split_manifest_episode_ids_invalid"}
    expected = _assign_episode_splits(entries)
    if any(entry.get("split") != expected.get(str(entry["episode_id"])) for entry in entries):
        return {"passed": False, "reason": "split_manifest_assignment_invalid"}
    episode_shas = [str(entry.get("episode_sha256") or "") for entry in entries]
    if (
        not all(_valid_sha(value) for value in episode_shas)
        or manifest.get("episode_ledger_manifest_sha256")
        != manifest_sha256(episode_shas)
    ):
        return {"passed": False, "reason": "split_manifest_ledger_invalid"}
    return {
        "passed": True,
        "reason": "verified",
        "manifest": manifest,
        "manifest_sha256": str(manifest["seal_sha256"]),
        "entries": entries,
    }


def adapter_callable_sha256(adapter: Any) -> str:
    """Derive a callable identity from exact source and qualified name."""

    try:
        source = inspect.getsource(adapter)
    except (OSError, TypeError):
        return ""
    return _canonical_sha(
        {
            "module": str(getattr(adapter, "__module__", "")),
            "qualname": str(getattr(adapter, "__qualname__", "")),
            "source": source,
        }
    )


def _adapter_registry_identity(
    kind: str, identity: Mapping[str, Any]
) -> dict[str, Any]:
    """Return the split-invariant adapter identity frozen in the registry."""

    normalized = dict(identity)
    if kind == "scorer":
        normalized.pop("evidence_manifest_sha256", None)
    return normalized


def validate_adapter_registry(
    value: Path | Mapping[str, Any],
    *,
    required: Sequence[tuple[str, Any, Mapping[str, Any]]],
    evaluated_at: str,
) -> dict[str, Any]:
    """Resolve authority adapters through one frozen, sealed allowlist."""

    try:
        payload = (
            read_sealed_json(value)
            if isinstance(value, Path)
            else verify_sealed_object(dict(value))
        )
    except (DurableStateError, TypeError, ValueError):
        return {"passed": False, "reason": "adapter_registry_seal_invalid"}
    entries = payload.get("entries")
    frozen_at = _strict_utc(payload.get("frozen_at"))
    normalized_evaluated_at = _strict_utc(evaluated_at)
    if (
        payload.get("schema_version") != 1
        or payload.get("artifact_kind") != "answer-authority-adapter-registry"
        or not frozen_at
        or not normalized_evaluated_at
        or datetime.fromisoformat(frozen_at.replace("Z", "+00:00"))
        >= datetime.fromisoformat(normalized_evaluated_at.replace("Z", "+00:00"))
        or not isinstance(entries, list)
        or any(not isinstance(entry, Mapping) for entry in entries)
    ):
        return {"passed": False, "reason": "adapter_registry_shape_invalid"}
    by_key: dict[tuple[str, str], Mapping[str, Any]] = {}
    for entry in entries:
        kind = str(entry.get("kind") or "")
        adapter_id = str(entry.get("adapter_id") or "")
        unsigned = {key: item for key, item in entry.items() if key != "entry_sha256"}
        if (
            kind not in {"runner", "scorer", "field_environment"}
            or not adapter_id
            or not _valid_sha(entry.get("callable_sha256"))
            or not _valid_sha(entry.get("identity_sha256"))
            or entry.get("entry_sha256") != _canonical_sha(unsigned)
            or (kind, adapter_id) in by_key
        ):
            return {"passed": False, "reason": "adapter_registry_entry_invalid"}
        by_key[(kind, adapter_id)] = entry
    for kind, adapter, identity in required:
        identity_id_key = {
            "runner": "runner_id",
            "scorer": "scorer_id",
            "field_environment": "adapter_id",
        }[kind]
        adapter_id = str(identity.get(identity_id_key) or "")
        entry = by_key.get((kind, adapter_id))
        if (
            entry is None
            or entry.get("callable_sha256") != adapter_callable_sha256(adapter)
            or entry.get("identity_sha256")
            != _canonical_sha(_adapter_registry_identity(kind, identity))
        ):
            return {"passed": False, "reason": "adapter_not_registered"}
    return {
        "passed": True,
        "reason": "verified",
        "manifest_sha256": str(payload.get("seal_sha256") or ""),
        "payload": payload,
    }


def _adapter_registry_binding_error(
    embedded: object,
    *,
    live_registry: Path | Mapping[str, Any],
    identities: Sequence[tuple[str, Mapping[str, Any]]],
    evaluated_at: str,
) -> str:
    try:
        embedded_payload = verify_sealed_object(dict(embedded))  # type: ignore[arg-type]
        live_payload = (
            read_sealed_json(live_registry)
            if isinstance(live_registry, Path)
            else verify_sealed_object(dict(live_registry))
        )
    except (DurableStateError, TypeError, ValueError):
        return "adapter_registry_seal_invalid"
    if embedded_payload != live_payload:
        return "adapter_registry_live_mismatch"
    entries = live_payload.get("entries")
    frozen_at = _strict_utc(live_payload.get("frozen_at"))
    normalized_evaluated_at = _strict_utc(evaluated_at)
    if (
        live_payload.get("schema_version") != 1
        or live_payload.get("artifact_kind")
        != "answer-authority-adapter-registry"
        or not frozen_at
        or not normalized_evaluated_at
        or datetime.fromisoformat(frozen_at.replace("Z", "+00:00"))
        >= datetime.fromisoformat(normalized_evaluated_at.replace("Z", "+00:00"))
        or not isinstance(entries, list)
    ):
        return "adapter_registry_shape_invalid"
    seen: set[tuple[str, str]] = set()
    for entry in entries:
        if not isinstance(entry, Mapping):
            return "adapter_registry_entry_invalid"
        kind = str(entry.get("kind") or "")
        adapter_id = str(entry.get("adapter_id") or "")
        key = (kind, adapter_id)
        unsigned = {name: item for name, item in entry.items() if name != "entry_sha256"}
        if (
            kind not in {"runner", "scorer", "field_environment"}
            or not adapter_id
            or key in seen
            or not _valid_sha(entry.get("callable_sha256"))
            or not _valid_sha(entry.get("identity_sha256"))
            or entry.get("entry_sha256") != _canonical_sha(unsigned)
        ):
            return "adapter_registry_entry_invalid"
        seen.add(key)
    for kind, identity in identities:
        id_key = {
            "runner": "runner_id",
            "scorer": "scorer_id",
            "field_environment": "adapter_id",
        }[kind]
        matches = [
            entry
            for entry in entries
            if isinstance(entry, Mapping)
            and entry.get("kind") == kind
            and entry.get("adapter_id") == identity.get(id_key)
            and entry.get("identity_sha256")
            == _canonical_sha(_adapter_registry_identity(kind, identity))
        ]
        if len(matches) != 1:
            return "adapter_registry_identity_mismatch"
    return ""


_REQUIRED_FIELD_ENVIRONMENT_IDENTITY = (
    "adapter_id",
    "version",
    "model_sha256",
    "policy_sha256",
    "config_sha256",
    "corpus_sha256",
    "index_sha256",
    "clone_protocol_sha256",
    "lkg_base_artifact_sha256",
    "lkg_base_snapshot_sha256",
    "effective_field_config_sha256",
    "candidate_policy_delta_sha256",
)
BUILTIN_FIELD_ENVIRONMENT_ADAPTER_ID = "chronovisor-field-e2e-v1"


def builtin_field_environment_identity() -> dict[str, Any]:
    """Derive the exact live implementation/config/corpus identity for replay."""

    from chronovisor.core.runtime_config import load_search_embedding_config
    from chronovisor.recall.recall_field import _effective_config
    from chronovisor.recall.recall_field_schema import load_recall_field_config
    from chronovisor.recall.recall_learning import load_last_known_good
    from chronovisor.recall.recall_runtime import load_policy

    field_config = load_recall_field_config()
    effective_field_config = _effective_config(field_config)
    lkg_path = (
        CHRONOVISOR_ROOT
        / "runtime"
        / "recall-field"
        / "last-known-good-policy.json"
    )
    lkg = load_last_known_good(lkg_path)
    try:
        lkg_artifact_sha = hashlib.sha256(lkg_path.read_bytes()).hexdigest()
    except OSError:
        lkg_artifact_sha = _canonical_sha({"status": "missing"})
    lkg_snapshot_sha = str(lkg.get("snapshot_sha256") or "")
    if not _valid_sha(lkg_snapshot_sha):
        lkg_snapshot_sha = _canonical_sha({"status": "missing"})
    policy = load_policy()
    search_config = load_search_embedding_config()
    corpus_rows: list[str] = []
    for path in sorted(PAGES_DIR.rglob("*.md")):
        try:
            corpus_rows.append(
                f"{path.relative_to(PAGES_DIR)}:{hashlib.sha256(path.read_bytes()).hexdigest()}"
            )
        except OSError:
            continue
    try:
        index_sha = hashlib.sha256(INDEX_FILE.read_bytes()).hexdigest()
    except OSError:
        index_sha = _canonical_sha({"missing": str(INDEX_FILE)})
    candidate_delta = {
        "base_lkg_artifact_sha256": lkg_artifact_sha,
        "effective_field_config": asdict(effective_field_config),
        "field_algorithm_sha256": adapter_callable_sha256(
            builtin_field_environment_replay
        ),
    }
    return {
        "adapter_id": BUILTIN_FIELD_ENVIRONMENT_ADAPTER_ID,
        "version": "1",
        "model_sha256": _canonical_sha(asdict(search_config)),
        "policy_sha256": _canonical_sha(asdict(policy)),
        "config_sha256": _canonical_sha(asdict(field_config)),
        "corpus_sha256": manifest_sha256(corpus_rows),
        "index_sha256": index_sha,
        "clone_protocol_sha256": _canonical_sha(
            {"protocol": "in-memory-field-state-clone-rollback-v1"}
        ),
        "lkg_base_artifact_sha256": lkg_artifact_sha,
        "lkg_base_snapshot_sha256": lkg_snapshot_sha,
        "effective_field_config_sha256": _canonical_sha(
            asdict(effective_field_config)
        ),
        "candidate_policy_delta_sha256": _canonical_sha(candidate_delta),
    }


def _field_replay_arm(
    *,
    arm_name: str,
    items: Sequence[Any],
    context: str,
    base_state_sha256: str,
    post_state_sha256: str,
    topic_epoch: int,
    identity: Mapping[str, Any],
) -> dict[str, Any]:
    from chronovisor.recall.recall_runtime import page_uid_for_id

    bindings: list[dict[str, Any]] = []
    certificate_ids: list[str] = []
    for rank, item in enumerate(items, start=1):
        page_id = str(getattr(item, "page_id", "") or "")
        path = find_page(page_id)
        try:
            content_sha = hashlib.sha256(path.read_bytes()).hexdigest() if path else ""
        except OSError:
            content_sha = ""
        certificate_id = str(getattr(item, "certificate_id", "") or "")
        page_uid = str(getattr(item, "uid", "") or page_uid_for_id(page_id))
        if not page_id or not page_uid or not content_sha or not certificate_id:
            continue
        bindings.append(
            {
                "page_id": page_id,
                "page_uid": page_uid,
                "content_sha256": content_sha,
                "rank": rank,
            }
        )
        certificate_ids.append(certificate_id)
    effective_policy_sha = (
        _canonical_sha(
            {
                "base_policy_sha256": identity["policy_sha256"],
                "candidate_policy_delta_sha256": identity[
                    "candidate_policy_delta_sha256"
                ],
            }
        )
        if arm_name == "candidate_field"
        else str(identity["policy_sha256"])
    )
    commit_id = _field_replay_commit_id(
        arm_name=arm_name,
        base_state_sha256=base_state_sha256,
        post_state_sha256=post_state_sha256,
        topic_epoch=topic_epoch,
        bindings=bindings,
        certificate_ids=certificate_ids,
        effective_policy_sha256=effective_policy_sha,
    )
    return {
        "context": context,
        "context_sha256": _sha_text(context),
        "pre_state_sha256": base_state_sha256,
        "post_state_sha256": post_state_sha256,
        "rollback_state_sha256": base_state_sha256,
        "clone_sha256": _canonical_sha(
            {"base": base_state_sha256, "arm": arm_name}
        ),
        "topic_epoch": topic_epoch,
        "policy_sha256": identity["policy_sha256"],
        "effective_policy_sha256": effective_policy_sha,
        "config_sha256": identity["config_sha256"],
        "corpus_sha256": identity["corpus_sha256"],
        "index_sha256": identity["index_sha256"],
        "retrieved_page_bindings": bindings,
        "certificate_ids": certificate_ids,
        "commit_ids": [commit_id] if bindings and certificate_ids else [],
    }


def _field_replay_commit_id(
    *,
    arm_name: str,
    base_state_sha256: str,
    post_state_sha256: str,
    topic_epoch: int,
    bindings: Sequence[Mapping[str, Any]],
    certificate_ids: Sequence[str],
    effective_policy_sha256: str,
) -> str:
    return _canonical_sha(
        {
            "arm": arm_name,
            "pre_state_sha256": base_state_sha256,
            "post_state_sha256": post_state_sha256,
            "topic_epoch": topic_epoch,
            "bindings": bindings,
            "certificate_ids": certificate_ids,
            "effective_policy_sha256": effective_policy_sha256,
        }
    )


def builtin_field_environment_replay(
    prompt: str, episode: Mapping[str, Any], seed: int
) -> Mapping[str, Any]:
    """Run current Field and full-search teacher from one in-memory state clone."""

    from chronovisor.recall.recall_field import (
        _effective_config,
        queue_teacher_commits,
        run_field_turn,
    )
    from chronovisor.recall.recall_field_candidate import _verify
    from chronovisor.recall.recall_field_schema import load_recall_field_config
    from chronovisor.recall.recall_field_store import RecallFieldStore
    from chronovisor.recall.recall_runtime import (
        RecallRequest,
        RecallResult,
        _retained_context_page_ids,
        collect_certified_context,
        format_recall_context,
        load_policy,
        search_candidates,
    )

    del seed  # State and search identity, not sampling order, determine retrieval.
    identity = builtin_field_environment_identity()
    config = _effective_config(load_recall_field_config())
    policy = replace(
        load_policy(),
        log_decisions=False,
        processor_judge_enabled=False,
    )
    session_hash = str(episode.get("session_hash") or "")
    base_state = RecallFieldStore(config=config).load(session_hash)
    base_sha = _canonical_sha(base_state.to_dict())

    class _ClonedFieldStore:
        def __init__(self, state: Any) -> None:
            self.state = copy.deepcopy(state)

        def transact(
            self,
            claimed_session_hash: str,
            mutate: Callable[[Any], tuple[Any, list[Any]]],
            *,
            now: float,
        ) -> tuple[Any, list[Any]]:
            if claimed_session_hash != session_hash:
                raise ValueError("field replay session hash mismatch")
            self.state, events = mutate(self.state)
            return self.state, events

    observed = max(time.time(), base_state.updated_at_epoch)
    candidate_store = _ClonedFieldStore(base_state)
    teacher_store = _ClonedFieldStore(base_state)
    candidate_turn = run_field_turn(
        host="answer-eval",
        session_id="replay",
        session_hash_override=session_hash,
        prompt=prompt,
        config=config,
        store=candidate_store,  # type: ignore[arg-type]
        now=observed,
    )
    teacher_turn = run_field_turn(
        host="answer-eval",
        session_id="replay",
        session_hash_override=session_hash,
        prompt=prompt,
        config=config,
        store=teacher_store,  # type: ignore[arg-type]
        now=observed,
    )
    candidate_ids = [
        str(page_id)
        for page_id in candidate_turn.get("candidate_page_ids", [])
        if isinstance(page_id, str) and page_id
    ]
    candidate_results, verify_meta = _verify(prompt, candidate_ids, timeout_ms=650)
    if verify_meta.get("status") != "verified":
        candidate_results = []
    request = RecallRequest(
        host="answer-eval",
        event="field-e2e-replay",
        prompt=prompt,
        session_id="",
    )
    deadline = time.monotonic() + max(4.0, policy.total_timeout_ms / 1_000)
    teacher_results, _mode = search_candidates(
        [prompt], policy, request=request, deadline_at=deadline
    )
    candidate_items, _candidate_meta = collect_certified_context(
        prompt,
        policy,
        request=request,
        session_state=None,
        candidates=list(candidate_results),
        reranker_metadata=None,
        deadline_at=deadline,
    )
    teacher_items, _teacher_meta = collect_certified_context(
        prompt,
        policy,
        request=request,
        session_state=None,
        candidates=list(teacher_results),
        reranker_metadata=None,
        deadline_at=deadline,
    )

    def render(items: list[Any]) -> tuple[str, list[Any]]:
        result = RecallResult(
            status="ok",
            decision="search",
            confidence=1.0,
            queries=[prompt],
            reasons=["registered field environment replay"],
            matched_terms={},
            context_items=items,
            context_style=policy.context_style,
        )
        context = format_recall_context(result, policy)
        retained = set(_retained_context_page_ids(context))
        return context, [item for item in items if item.page_id in retained]

    candidate_context, candidate_items = render(candidate_items)
    teacher_context, teacher_items = render(teacher_items)

    def queue(items: list[Any], store: _ClonedFieldStore) -> dict[str, Any]:
        return queue_teacher_commits(
            host="answer-eval",
            session_id="replay",
            session_hash_override=session_hash,
            page_ids=[str(item.page_id) for item in items],
            certificate_ids={
                str(item.page_id): str(item.certificate_id)
                for item in items
                if str(item.certificate_id or "")
            },
            config=config,
            store=store,  # type: ignore[arg-type]
            now=observed,
        )

    candidate_queue = queue(candidate_items, candidate_store)
    teacher_queue = queue(teacher_items, teacher_store)
    if (
        candidate_queue.get("queued") != len(candidate_items)
        or teacher_queue.get("queued") != len(teacher_items)
    ):
        raise ValueError("field replay teacher commit queue mismatch")
    candidate_post_sha = _canonical_sha(candidate_store.state.to_dict())
    teacher_post_sha = _canonical_sha(teacher_store.state.to_dict())
    return {
        "identity": identity,
        "base_state_sha256": base_sha,
        "arms": {
            "candidate_field": _field_replay_arm(
                arm_name="candidate_field",
                items=candidate_items,
                context=candidate_context,
                base_state_sha256=base_sha,
                post_state_sha256=candidate_post_sha,
                topic_epoch=int(candidate_turn.get("topic_epoch") or 0),
                identity=identity,
            ),
            "production_teacher": _field_replay_arm(
                arm_name="production_teacher",
                items=teacher_items,
                context=teacher_context,
                base_state_sha256=base_sha,
                post_state_sha256=teacher_post_sha,
                topic_epoch=int(teacher_turn.get("topic_epoch") or 0),
                identity=identity,
            ),
        },
    }


def _field_environment_contexts(
    adapter: FieldEnvironmentReplay,
    *,
    prompt: str,
    episode: Mapping[str, Any],
    pair_seed: int,
    identity: Mapping[str, Any],
    parent_run_id: str,
    execution_ledger_file: Path,
) -> tuple[dict[str, str], dict[str, Any], str]:
    from chronovisor.recall.recall_runtime import _retained_context_page_ids

    started_at = _now_utc()
    try:
        result = adapter(prompt, episode, pair_seed)
    except Exception as exc:
        return {}, {}, f"field_environment_error:{type(exc).__name__}"
    if not isinstance(result, Mapping) or result.get("identity") != dict(identity):
        return {}, {}, "field_environment_identity_mismatch"
    if _identity_error(identity, _REQUIRED_FIELD_ENVIRONMENT_IDENTITY):
        return {}, {}, "field_environment_identity_invalid"
    base_state_sha = result.get("base_state_sha256")
    arms_value = result.get("arms")
    if not _valid_sha(base_state_sha) or not isinstance(arms_value, Mapping):
        return {}, {}, "field_environment_receipt_invalid"
    if set(arms_value) != {"candidate_field", "production_teacher"}:
        return {}, {}, "field_environment_arms_invalid"
    contexts: dict[str, str] = {}
    sealed_arms: dict[str, Any] = {}
    for arm_name in ("candidate_field", "production_teacher"):
        arm_value = arms_value.get(arm_name)
        arm = arm_value if isinstance(arm_value, Mapping) else {}
        context = arm.get("context")
        bindings = arm.get("retrieved_page_bindings")
        certificates = arm.get("certificate_ids")
        commits = arm.get("commit_ids")
        retained_page_ids = (
            _retained_context_page_ids(context) if isinstance(context, str) else []
        )
        binding_page_ids = [
            str(binding.get("page_id") or "")
            for binding in bindings
            if isinstance(binding, Mapping)
        ] if isinstance(bindings, list) else []
        expected_effective_policy_sha = (
            _canonical_sha(
                {
                    "base_policy_sha256": identity.get("policy_sha256"),
                    "candidate_policy_delta_sha256": identity.get(
                        "candidate_policy_delta_sha256"
                    ),
                }
            )
            if arm_name == "candidate_field"
            else str(identity.get("policy_sha256") or "")
        )
        if (
            not isinstance(context, str)
            or arm.get("context_sha256") != _sha_text(context)
            or arm.get("pre_state_sha256") != base_state_sha
            or not _valid_sha(arm.get("post_state_sha256"))
            or arm.get("rollback_state_sha256") != base_state_sha
            or not _valid_sha(arm.get("clone_sha256"))
            or not isinstance(arm.get("topic_epoch"), int)
            or isinstance(arm.get("topic_epoch"), bool)
            or int(arm.get("topic_epoch")) < 0
            or arm.get("policy_sha256") != identity.get("policy_sha256")
            or arm.get("effective_policy_sha256")
            != expected_effective_policy_sha
            or arm.get("config_sha256") != identity.get("config_sha256")
            or arm.get("corpus_sha256") != identity.get("corpus_sha256")
            or arm.get("index_sha256") != identity.get("index_sha256")
            or not isinstance(bindings, list)
            or not bindings
            or binding_page_ids != retained_page_ids
            or any(
                not isinstance(binding, Mapping)
                or not str(binding.get("page_id") or "")
                or not str(binding.get("page_uid") or "")
                or not _valid_sha(binding.get("content_sha256"))
                or not isinstance(binding.get("rank"), int)
                or isinstance(binding.get("rank"), bool)
                or int(binding.get("rank")) <= 0
                for binding in bindings
            )
            or not isinstance(certificates, list)
            or not certificates
            or len(certificates) != len(bindings)
            or not all(isinstance(value, str) and value for value in certificates)
            or not isinstance(commits, list)
            or not commits
            or not all(isinstance(value, str) and value for value in commits)
            or commits
            != [
                _field_replay_commit_id(
                    arm_name=arm_name,
                    base_state_sha256=str(base_state_sha),
                    post_state_sha256=str(arm.get("post_state_sha256") or ""),
                    topic_epoch=int(arm.get("topic_epoch") or 0),
                    bindings=[dict(value) for value in bindings],
                    certificate_ids=[str(value) for value in certificates],
                    effective_policy_sha256=expected_effective_policy_sha,
                )
            ]
            or _field_environment_live_binding_error(bindings)
        ):
            return {}, {}, "field_environment_arm_receipt_invalid"
        contexts[arm_name] = context
        sealed_arms[arm_name] = {
            key: value for key, value in arm.items() if key != "context"
        }
    if (
        sealed_arms["candidate_field"]["topic_epoch"]
        != sealed_arms["production_teacher"]["topic_epoch"]
    ):
        return {}, {}, "field_environment_topic_epoch_mismatch"
    evidence = {
        "base_state_sha256": base_state_sha,
        "pair_seed": pair_seed,
        "arms": sealed_arms,
    }
    try:
        receipt = append_answer_execution_receipt(
            kind="field_environment_replay",
            adapter_identity_sha256=_canonical_sha(dict(identity)),
            parent_run_id=parent_run_id,
            input_payload={
                "episode_id": episode.get("episode_id"),
                "prompt_sha256": _sha_text(prompt),
                "pair_seed": pair_seed,
            },
            output_payload=evidence,
            started_at=started_at,
            completed_at=_now_utc(),
            ledger_file=execution_ledger_file,
        )
    except (OSError, ValueError):
        return {}, {}, "field_environment_execution_receipt_failed"
    evidence["execution_receipt_sha256"] = receipt["receipt_sha256"]
    return contexts, evidence, ""


def _field_environment_evidence_error(
    evidence: object,
    *,
    identity: Mapping[str, Any],
    parent_run_id: str,
    episode_id: object,
    prompt_sha256: str,
    pair_seed: int,
    execution_ledger_file: Path,
) -> str:
    if not isinstance(evidence, Mapping):
        return "field_environment_evidence_missing"
    base_state_sha = evidence.get("base_state_sha256")
    arms = evidence.get("arms")
    if (
        not _valid_sha(base_state_sha)
        or evidence.get("pair_seed") != pair_seed
        or not isinstance(arms, Mapping)
        or set(arms) != {"candidate_field", "production_teacher"}
    ):
        return "field_environment_evidence_invalid"
    topic_epochs: set[int] = set()
    for arm_name in ("candidate_field", "production_teacher"):
        arm_value = arms.get(arm_name)
        arm = arm_value if isinstance(arm_value, Mapping) else {}
        bindings = arm.get("retrieved_page_bindings")
        topic_epoch = arm.get("topic_epoch")
        expected_effective_policy_sha = (
            _canonical_sha(
                {
                    "base_policy_sha256": identity.get("policy_sha256"),
                    "candidate_policy_delta_sha256": identity.get(
                        "candidate_policy_delta_sha256"
                    ),
                }
            )
            if arm_name == "candidate_field"
            else str(identity.get("policy_sha256") or "")
        )
        if (
            not _valid_sha(arm.get("context_sha256"))
            or arm.get("pre_state_sha256") != base_state_sha
            or not _valid_sha(arm.get("post_state_sha256"))
            or arm.get("rollback_state_sha256") != base_state_sha
            or not _valid_sha(arm.get("clone_sha256"))
            or not isinstance(topic_epoch, int)
            or isinstance(topic_epoch, bool)
            or topic_epoch < 0
            or arm.get("policy_sha256") != identity.get("policy_sha256")
            or arm.get("effective_policy_sha256")
            != expected_effective_policy_sha
            or arm.get("config_sha256") != identity.get("config_sha256")
            or arm.get("corpus_sha256") != identity.get("corpus_sha256")
            or arm.get("index_sha256") != identity.get("index_sha256")
            or not isinstance(bindings, list)
            or not bindings
            or any(
                not isinstance(binding, Mapping)
                or not str(binding.get("page_id") or "")
                or not str(binding.get("page_uid") or "")
                or not _valid_sha(binding.get("content_sha256"))
                or not isinstance(binding.get("rank"), int)
                or isinstance(binding.get("rank"), bool)
                or int(binding.get("rank")) <= 0
                for binding in bindings
            )
            or not isinstance(arm.get("certificate_ids"), list)
            or not arm.get("certificate_ids")
            or not isinstance(arm.get("commit_ids"), list)
            or not arm.get("commit_ids")
            or any(
                not isinstance(value, str) or not value
                for value in [
                    *arm.get("certificate_ids", []),
                    *arm.get("commit_ids", []),
                ]
            )
            or _field_environment_live_binding_error(bindings)
            or arm.get("commit_ids")
            != [
                _field_replay_commit_id(
                    arm_name=arm_name,
                    base_state_sha256=str(base_state_sha),
                    post_state_sha256=str(arm.get("post_state_sha256") or ""),
                    topic_epoch=int(topic_epoch or 0),
                    bindings=[dict(value) for value in bindings],
                    certificate_ids=[str(value) for value in arm.get("certificate_ids", [])],
                    effective_policy_sha256=expected_effective_policy_sha,
                )
            ]
        ):
            return "field_environment_evidence_invalid"
        topic_epochs.add(topic_epoch)
    if len(topic_epochs) != 1:
        return "field_environment_topic_epoch_mismatch"
    unsigned_evidence = {
        key: value
        for key, value in evidence.items()
        if key != "execution_receipt_sha256"
    }
    return _execution_receipt_error(
        receipt_sha256=evidence.get("execution_receipt_sha256"),
        expected_kind="field_environment_replay",
        expected_adapter_identity_sha256=_canonical_sha(dict(identity)),
        expected_parent_run_id=parent_run_id,
        expected_input_payload={
            "episode_id": episode_id,
            "prompt_sha256": prompt_sha256,
            "pair_seed": pair_seed,
        },
        expected_output_payload=unsigned_evidence,
        ledger_file=execution_ledger_file,
    )


def _field_environment_live_binding_error(bindings: Sequence[object]) -> str:
    """Bind replay ranks to the current registered page UID and exact bytes."""

    from chronovisor.recall.recall_runtime import page_uid_for_id

    seen_ranks: set[int] = set()
    for value in bindings:
        if not isinstance(value, Mapping):
            return "field_environment_binding_invalid"
        page_id = str(value.get("page_id") or "")
        rank = value.get("rank")
        path = find_page(page_id)
        try:
            live_sha = hashlib.sha256(path.read_bytes()).hexdigest() if path else ""
        except OSError:
            return "field_environment_binding_unreadable"
        if (
            not isinstance(rank, int)
            or isinstance(rank, bool)
            or rank in seen_ranks
            or value.get("page_uid") != page_uid_for_id(page_id)
            or not value.get("page_uid")
            or value.get("content_sha256") != live_sha
        ):
            return "field_environment_binding_live_mismatch"
        seen_ranks.add(rank)
    return ""


def _preregistered_pair_protocol(
    *,
    seed: int,
    episode_id: object,
    episode_sha256: object,
    split_manifest_sha256: object,
    gold_manifest_sha256: object,
    adapter_registry_sha256: object,
    evaluation_kind: str,
) -> dict[str, Any]:
    """Derive pair randomness only from immutable preregistered evidence."""

    protocol_sha256 = _canonical_sha(
        {
            "protocol": "answer-paired-replay-v1",
            "seed": seed,
            "episode_id": episode_id,
            "episode_sha256": episode_sha256,
            "split_manifest_sha256": split_manifest_sha256,
            "gold_manifest_sha256": gold_manifest_sha256,
            "adapter_registry_sha256": adapter_registry_sha256,
            "evaluation_kind": evaluation_kind,
        }
    )
    pair_seed = int(protocol_sha256[:16], 16)
    arm_order = ["field_on", "field_off"]
    if pair_seed % 2:
        arm_order.reverse()
    return {
        "protocol_sha256": protocol_sha256,
        "pair_seed": pair_seed,
        "arm_order": arm_order,
        "generation": {
            "seed": pair_seed,
            "base_state_sha256": _canonical_sha(
                {"pair_protocol_sha256": protocol_sha256, "role": "answer_runner"}
            ),
        },
        "scoring": {
            "seed": pair_seed,
            "base_state_sha256": _canonical_sha(
                {"pair_protocol_sha256": protocol_sha256, "role": "answer_scorer"}
            ),
        },
    }


def _runner_answer(
    runner: AnswerRunner,
    prompt: str,
    context: str,
    generation: Mapping[str, Any],
    identity: Mapping[str, Any],
    *,
    parent_run_id: str,
    execution_ledger_file: Path,
) -> tuple[str, str, str]:
    started_at = _now_utc()
    try:
        result = runner(prompt, context, generation)
    except Exception as exc:
        return "", f"runner_error:{type(exc).__name__}", ""
    if not isinstance(result, Mapping):
        return "", "invalid_runner_result", ""
    answer = result.get("answer")
    returned_identity = result.get("identity")
    reset_receipt = result.get("reset_receipt")
    if not isinstance(answer, str) or not answer:
        return "", "empty_runner_answer", ""
    if not isinstance(returned_identity, Mapping) or dict(returned_identity) != dict(
        identity
    ):
        return "", "runner_identity_mismatch", ""
    expected_reset = {
        "seed": generation.get("seed"),
        "base_state_sha256": generation.get("base_state_sha256"),
        "reset_protocol_sha256": identity.get("policy_sha256"),
    }
    if not isinstance(reset_receipt, Mapping) or dict(reset_receipt) != expected_reset:
        return "", "runner_reset_receipt_invalid", ""
    input_payload = {
        "prompt_sha256": _sha_text(prompt),
        "context_sha256": _sha_text(context),
        "generation": dict(generation),
    }
    output_payload = {
        "answer_sha256": _sha_text(answer),
        "answer_chars": len(answer),
        "reset_receipt": expected_reset,
    }
    try:
        receipt = append_answer_execution_receipt(
            kind="answer_runner_call",
            adapter_identity_sha256=_canonical_sha(dict(identity)),
            parent_run_id=parent_run_id,
            input_payload=input_payload,
            output_payload=output_payload,
            started_at=started_at,
            completed_at=_now_utc(),
            ledger_file=execution_ledger_file,
        )
    except (OSError, ValueError):
        return "", "runner_execution_receipt_failed", ""
    return answer, "", str(receipt["receipt_sha256"])


def _score_answer(
    scorer: AnswerScorer,
    prompt: str,
    answer: str,
    gold: Mapping[str, Any],
    identity: Mapping[str, Any],
    scoring: Mapping[str, Any],
    *,
    parent_run_id: str,
    execution_ledger_file: Path,
) -> tuple[dict[str, float] | None, str, str]:
    started_at = _now_utc()
    try:
        result = scorer(prompt, answer, gold, scoring)
    except Exception as exc:
        return None, f"scorer_error:{type(exc).__name__}", ""
    if not isinstance(result, Mapping):
        return None, "invalid_scorer_result", ""
    returned_identity = result.get("identity")
    dimensions = result.get("dimensions")
    evidence_sha = result.get("evidence_sha256")
    expected_reset = {
        "seed": scoring.get("seed"),
        "base_state_sha256": scoring.get("base_state_sha256"),
        "reset_protocol_sha256": identity.get("policy_sha256"),
    }
    reset_receipt = result.get("reset_receipt")
    if (
        not isinstance(returned_identity, Mapping)
        or dict(returned_identity) != dict(identity)
        or evidence_sha != gold.get("evidence_sha256")
        or not isinstance(reset_receipt, Mapping)
        or dict(reset_receipt) != expected_reset
        or not isinstance(dimensions, Mapping)
        or set(dimensions) != set(ANSWER_DIMENSIONS)
    ):
        return None, "scorer_identity_or_evidence_mismatch", ""
    scores: dict[str, float] = {}
    for dimension in ANSWER_DIMENSIONS:
        raw = dimensions.get(dimension)
        if (
            isinstance(raw, bool)
            or not isinstance(raw, int | float)
            or not 0.0 <= float(raw) <= 1.0
        ):
            return None, "invalid_scorer_dimensions", ""
        scores[dimension] = round(float(raw), 9)
    try:
        receipt = append_answer_execution_receipt(
            kind="answer_scorer_call",
            adapter_identity_sha256=_canonical_sha(dict(identity)),
            parent_run_id=parent_run_id,
            input_payload={
                "prompt_sha256": _sha_text(prompt),
                "answer_sha256": _sha_text(answer),
                "gold_evidence_sha256": gold.get("evidence_sha256"),
                "scoring": dict(scoring),
            },
            output_payload={"dimensions": scores, "reset_receipt": expected_reset},
            started_at=started_at,
            completed_at=_now_utc(),
            ledger_file=execution_ledger_file,
        )
    except (OSError, ValueError):
        return None, "scorer_execution_receipt_failed", ""
    return scores, "", str(receipt["receipt_sha256"])


def validate_gold_manifest(
    value: Path | Mapping[str, Any],
    *,
    required_episode_ids: Sequence[str],
    review_ledger_file: Path = ANSWER_REVIEW_LEDGER,
) -> dict[str, Any]:
    try:
        payload = (
            read_sealed_json(value)
            if isinstance(value, Path)
            else verify_sealed_object(dict(value))
        )
    except (DurableStateError, TypeError, ValueError):
        return {"passed": False, "reason": "gold_manifest_seal_invalid"}
    entries = payload.get("entries")
    rubric_sha = payload.get("rubric_sha256")
    if (
        payload.get("schema_version") != 1
        or payload.get("artifact_kind") != "immutable-answer-gold-manifest"
        or not _strict_utc(payload.get("frozen_at"))
        or not isinstance(payload.get("gold_id"), str)
        or not str(payload.get("gold_id") or "")
        or not isinstance(payload.get("version"), str)
        or not str(payload.get("version") or "")
        or not isinstance(payload.get("gold_family_id"), str)
        or not str(payload.get("gold_family_id") or "")
        or not _valid_sha(payload.get("review_protocol_sha256"))
        or not _valid_sha(rubric_sha)
        or not isinstance(entries, list)
    ):
        return {"passed": False, "reason": "gold_manifest_shape_invalid"}
    by_id: dict[str, dict[str, Any]] = {}
    seen_review_receipts: set[str] = set()
    frozen_at = _strict_utc(payload.get("frozen_at"))
    review_protocol_sha = str(payload.get("review_protocol_sha256") or "")
    for raw_entry in entries:
        if not isinstance(raw_entry, dict):
            return {"passed": False, "reason": "gold_entry_invalid"}
        entry = dict(raw_entry)
        episode_id = str(entry.get("episode_id") or "")
        gold_answer = entry.get("gold_answer")
        evidence = entry.get("evidence")
        evidence_sha = entry.get("evidence_sha256")
        review_value = entry.get("review_provenance")
        review = review_value if isinstance(review_value, Mapping) else {}
        expected_evidence_sha = _canonical_sha(
            {
                "episode_id": episode_id,
                "gold_answer": gold_answer,
                "evidence": evidence,
                "rubric_sha256": rubric_sha,
            }
        )
        receipt_sha = str(review.get("reviewer_receipt_sha256") or "")
        review_error = _review_receipt_error(
            receipt_sha256=receipt_sha,
            expected_kind="gold_entry_review",
            expected_payload={
                "episode_id": episode_id,
                "gold_answer_sha256": _sha_text(gold_answer)
                if isinstance(gold_answer, str)
                else "",
                "evidence_sha256": evidence_sha,
                "rubric_sha256": rubric_sha,
            },
            expected_protocol_sha256=review_protocol_sha,
            ledger_file=review_ledger_file,
            frozen_at=frozen_at,
            expected_reviewed_at=str(review.get("reviewed_at") or ""),
        )
        if (
            not episode_id
            or episode_id in by_id
            or not isinstance(gold_answer, str)
            or not gold_answer
            or not isinstance(evidence, Mapping)
            or evidence_sha != expected_evidence_sha
            or review.get("source_kind") not in _ALLOWED_GOLD_SOURCE_KINDS
            or not _valid_sha(review.get("reviewer_receipt_sha256"))
            or not _strict_utc(review.get("reviewed_at"))
            or receipt_sha in seen_review_receipts
            or review_error
        ):
            return {"passed": False, "reason": "gold_entry_invalid"}
        seen_review_receipts.add(receipt_sha)
        by_id[episode_id] = entry
    required = list(required_episode_ids)
    if len(required) != len(set(required)) or set(by_id) != set(required):
        return {"passed": False, "reason": "gold_episode_membership_mismatch"}
    return {
        "passed": True,
        "reason": "verified",
        "manifest_sha256": str(payload["seal_sha256"]),
        "rubric_sha256": str(rubric_sha),
        "gold_id": str(payload["gold_id"]),
        "version": str(payload["version"]),
        "gold_family_id": str(payload["gold_family_id"]),
        "review_protocol_sha256": str(payload["review_protocol_sha256"]),
        "payload": payload,
        "entries": by_id,
    }


def _scorer_calibration_policy() -> dict[str, Any]:
    """Return the fixed authority policy; callers cannot loosen these gates."""

    return {
        "confidence": SCORER_CALIBRATION_CONFIDENCE,
        "minimum_cases": SCORER_CALIBRATION_MIN_CASES,
        "minimum_sessions": SCORER_CALIBRATION_MIN_SESSIONS,
        "minimum_clusters": SCORER_CALIBRATION_MIN_CLUSTERS,
        "coverage_point_floor": SCORER_CALIBRATION_COVERAGE_POINT_FLOOR,
        "coverage_lcb_floor": SCORER_CALIBRATION_COVERAGE_LCB_FLOOR,
        "agreement_point_floor": SCORER_CALIBRATION_AGREEMENT_POINT_FLOOR,
        "agreement_lcb_floor": SCORER_CALIBRATION_AGREEMENT_LCB_FLOOR,
        "score_tolerance": SCORER_CALIBRATION_SCORE_TOLERANCE,
        "mae_ceiling": SCORER_CALIBRATION_MAE_CEILING,
        "absolute_bias_ceiling": SCORER_CALIBRATION_ABS_BIAS_CEILING,
        "within_tolerance_lcb_floor": SCORER_CALIBRATION_WITHIN_TOLERANCE_LCB_FLOOR,
        "pairwise_preference_lcb_floor": SCORER_CALIBRATION_PREFERENCE_LCB_FLOOR,
    }


def _calibration_scorer_identity(identity: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: identity.get(key) for key in _REQUIRED_CALIBRATION_SCORER_IDENTITY
    }


def _calibration_case_error(
    case: Mapping[str, Any],
    *,
    scorer_identity_sha256: str,
    review_protocol_sha256: str,
    review_ledger_file: Path,
    execution_ledger_file: Path,
    calibration_run_id: str,
    frozen_at: str,
) -> str:
    unsigned = {key: value for key, value in case.items() if key != "case_sha256"}
    labels = case.get("human_reviewed")
    scores = case.get("scorer_scores")
    nodes = case.get("cluster_nodes")
    expected_nodes = _calibration_cluster_nodes(case)
    review_error = _review_receipt_error(
        receipt_sha256=case.get("review_receipt_sha256"),
        expected_kind="scorer_calibration_case_review",
        expected_payload={
            "case_id": case.get("case_id"),
            "session_hash": case.get("session_hash"),
            "query_sha256": case.get("query_sha256"),
            "evidence_sha256": case.get("evidence_sha256"),
            "human_reviewed": labels,
        },
        expected_protocol_sha256=review_protocol_sha256,
        ledger_file=review_ledger_file,
        frozen_at=frozen_at,
    )
    execution_error = _execution_receipt_error(
        receipt_sha256=case.get("execution_receipt_sha256"),
        expected_kind="calibration_scorer_call",
        expected_adapter_identity_sha256=scorer_identity_sha256,
        expected_parent_run_id=calibration_run_id,
        expected_input_payload={
            "case_id": case.get("case_id"),
            "query_sha256": case.get("query_sha256"),
            "evidence_sha256": case.get("evidence_sha256"),
        },
        expected_output_payload={"dimensions": scores},
        ledger_file=execution_ledger_file,
        completed_before=frozen_at,
    )
    if (
        not isinstance(case.get("case_id"), str)
        or not str(case.get("case_id") or "").strip()
        or not isinstance(case.get("session_hash"), str)
        or len(str(case.get("session_hash") or "").strip()) < 8
        or not _valid_sha(case.get("query_sha256"))
        or not _valid_sha(case.get("evidence_sha256"))
        or case.get("scorer_identity_sha256") != scorer_identity_sha256
        or not _valid_sha(case.get("review_receipt_sha256"))
        or review_error
        or not _valid_sha(case.get("execution_receipt_sha256"))
        or execution_error
        or nodes != expected_nodes
        or not isinstance(case.get("pair_id"), str)
        or not str(case.get("pair_id") or "")
        or case.get("pair_arm") not in {"a", "b"}
        or not isinstance(labels, Mapping)
        or set(labels) != set(ANSWER_DIMENSIONS)
        or any(
            not _finite_unit_score(labels.get(dimension))
            for dimension in ANSWER_DIMENSIONS
        )
        or not isinstance(scores, Mapping)
        or set(scores) != set(ANSWER_DIMENSIONS)
        or any(
            scores.get(dimension) is not None
            and not _finite_unit_score(scores.get(dimension))
            for dimension in ANSWER_DIMENSIONS
        )
        or case.get("case_sha256") != _canonical_sha(unsigned)
    ):
        return "calibration_case_invalid"
    return ""


def _finite_unit_score(value: object) -> bool:
    return bool(
        not isinstance(value, bool)
        and isinstance(value, int | float)
        and math.isfinite(float(value))
        and 0.0 <= float(value) <= 1.0
    )


def _calibration_cluster_nodes(case: Mapping[str, Any]) -> list[str]:
    """Derive independence identities; callers cannot invent unique clusters."""

    return [
        f"session:{str(case.get('session_hash') or '')}",
        f"query:{str(case.get('query_sha256') or '')}",
        f"evidence:{str(case.get('evidence_sha256') or '')}",
    ]


def _calibration_split_overlap(
    cases: Sequence[Mapping[str, Any]],
    split_manifest: Path | Mapping[str, Any],
) -> str:
    split_check = validate_split_manifest(split_manifest)
    if split_check.get("passed") is not True:
        return "calibration_answer_split_invalid"
    split_sessions: set[str] = set()
    split_queries: set[str] = set()
    split_contents: set[str] = set()
    for entry in split_check.get("entries", []):
        split_sessions.add(str(entry.get("session_hash") or ""))
        split_queries.add(str(entry.get("query_sha256") or ""))
        for binding in entry.get("page_bindings", []):
            if isinstance(binding, Mapping):
                split_contents.add(str(binding.get("content_sha256") or ""))
    calibration_sessions = {str(case["session_hash"]) for case in cases}
    calibration_queries = {str(case["query_sha256"]) for case in cases}
    calibration_evidence = {str(case["evidence_sha256"]) for case in cases}
    if (
        calibration_sessions & split_sessions
        or calibration_queries & split_queries
        or calibration_evidence & split_contents
    ):
        return "calibration_answer_split_overlap"
    return ""


def _calibration_metrics(cases: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    sessions = {
        str(case.get("session_hash") or "")
        for case in cases
        if str(case.get("session_hash") or "")
    }
    coverage: dict[str, dict[str, Any]] = {}
    within_tolerance: dict[str, dict[str, Any]] = {}
    mean_absolute_error: dict[str, float] = {}
    bias: dict[str, float] = {}
    abstentions: dict[str, int] = {}
    for dimension in ANSWER_DIMENSIONS:
        coverage_rows: list[dict[str, Any]] = []
        tolerance_rows: list[dict[str, Any]] = []
        signed_errors: list[float] = []
        for case in cases:
            decision = case["scorer_scores"][dimension]
            common = {
                "session_hash": case["session_hash"],
                "query_sha256": case["query_sha256"],
                "cluster_nodes": case["cluster_nodes"],
            }
            covered = decision is not None
            coverage_rows.append({**common, "covered": 1.0 if covered else 0.0})
            tolerance_rows.append(
                {
                    **common,
                    "within": 1.0
                    if covered
                    and abs(float(decision) - float(case["human_reviewed"][dimension]))
                    <= SCORER_CALIBRATION_SCORE_TOLERANCE
                    else 0.0,
                }
            )
            if covered:
                signed_errors.append(
                    float(decision) - float(case["human_reviewed"][dimension])
                )
        coverage[dimension] = cluster_rate_wilson_interval(
            coverage_rows,
            value_key="covered",
            success_threshold=1.0,
            confidence=SCORER_CALIBRATION_CONFIDENCE,
        )
        within_tolerance[dimension] = cluster_rate_wilson_interval(
            tolerance_rows,
            value_key="within",
            success_threshold=1.0,
            confidence=SCORER_CALIBRATION_CONFIDENCE,
        )
        abstentions[dimension] = sum(
            case["scorer_scores"][dimension] is None for case in cases
        )
        mean_absolute_error[dimension] = round(
            sum(abs(value) for value in signed_errors) / max(1, len(signed_errors)),
            9,
        )
        bias[dimension] = round(
            sum(signed_errors) / max(1, len(signed_errors)),
            9,
        )
    pairs: dict[str, list[Mapping[str, Any]]] = {}
    for case in cases:
        pairs.setdefault(str(case.get("pair_id") or ""), []).append(case)
    preference_rows: list[dict[str, Any]] = []
    invalid_pairs = 0
    for pair_id, pair_cases in sorted(pairs.items()):
        by_arm = {str(case.get("pair_arm") or ""): case for case in pair_cases}
        if set(by_arm) != {"a", "b"} or len(pair_cases) != 2:
            invalid_pairs += 1
            continue
        left, right = by_arm["a"], by_arm["b"]
        scorer_values = [
            left["scorer_scores"][dimension] for dimension in ANSWER_DIMENSIONS
        ] + [right["scorer_scores"][dimension] for dimension in ANSWER_DIMENSIONS]
        covered = all(value is not None for value in scorer_values)
        human_delta = sum(
            float(left["human_reviewed"][dimension])
            - float(right["human_reviewed"][dimension])
            for dimension in ANSWER_DIMENSIONS
        )
        scorer_delta = (
            sum(
                float(left["scorer_scores"][dimension])
                - float(right["scorer_scores"][dimension])
                for dimension in ANSWER_DIMENSIONS
            )
            if covered
            else 0.0
        )
        preference_rows.append(
            {
                "pair_id": pair_id,
                "cluster_nodes": list(
                    dict.fromkeys(
                        [
                            *left["cluster_nodes"],
                            *right["cluster_nodes"],
                            f"pair:{pair_id}",
                        ]
                    )
                ),
                "correct": 1.0
                if covered
                and human_delta != 0.0
                and scorer_delta != 0.0
                and human_delta * scorer_delta > 0.0
                else 0.0,
            }
        )
    preference = cluster_rate_wilson_interval(
        preference_rows,
        value_key="correct",
        success_threshold=1.0,
        confidence=SCORER_CALIBRATION_CONFIDENCE,
    )
    cluster_counts = {
        bound.get("clusters")
        for bound in coverage.values()
        if isinstance(bound.get("clusters"), int)
    }
    return {
        "cases": len(cases),
        "sessions": len(sessions),
        "clusters": next(iter(cluster_counts)) if len(cluster_counts) == 1 else 0,
        "abstentions": abstentions,
        "dimension_coverage": coverage,
        "dimension_within_tolerance": within_tolerance,
        "dimension_mae": mean_absolute_error,
        "dimension_bias": bias,
        "pairwise_preference": preference,
        "pairs": len(preference_rows),
        "invalid_pairs": invalid_pairs,
    }


def _calibration_gates(metrics: Mapping[str, Any]) -> dict[str, bool]:
    coverage = metrics.get("dimension_coverage")
    within_tolerance = metrics.get("dimension_within_tolerance")
    coverage_map = coverage if isinstance(coverage, Mapping) else {}
    tolerance_map = (
        within_tolerance if isinstance(within_tolerance, Mapping) else {}
    )
    mae_map = metrics.get("dimension_mae")
    bias_map = metrics.get("dimension_bias")
    preference = metrics.get("pairwise_preference")
    preference_map = preference if isinstance(preference, Mapping) else {}
    return {
        "minimum_cases": isinstance(metrics.get("cases"), int)
        and metrics["cases"] >= SCORER_CALIBRATION_MIN_CASES,
        "minimum_sessions": isinstance(metrics.get("sessions"), int)
        and metrics["sessions"] >= SCORER_CALIBRATION_MIN_SESSIONS,
        "minimum_clusters": isinstance(metrics.get("clusters"), int)
        and metrics["clusters"] >= SCORER_CALIBRATION_MIN_CLUSTERS,
        "coverage_point": all(
            isinstance(coverage_map.get(dimension), Mapping)
            and isinstance(coverage_map[dimension].get("point"), int | float)
            and float(coverage_map[dimension]["point"])
            >= SCORER_CALIBRATION_COVERAGE_POINT_FLOOR
            for dimension in ANSWER_DIMENSIONS
        ),
        "coverage_lower_bound": all(
            isinstance(coverage_map.get(dimension), Mapping)
            and isinstance(coverage_map[dimension].get("lower"), int | float)
            and float(coverage_map[dimension]["lower"])
            >= SCORER_CALIBRATION_COVERAGE_LCB_FLOOR
            for dimension in ANSWER_DIMENSIONS
        ),
        "mean_absolute_error": isinstance(mae_map, Mapping)
        and all(
            _finite_unit_score(mae_map.get(dimension))
            and float(mae_map[dimension]) <= SCORER_CALIBRATION_MAE_CEILING
            for dimension in ANSWER_DIMENSIONS
        ),
        "absolute_bias": isinstance(bias_map, Mapping)
        and all(
            isinstance(bias_map.get(dimension), int | float)
            and not isinstance(bias_map.get(dimension), bool)
            and math.isfinite(float(bias_map[dimension]))
            and abs(float(bias_map[dimension]))
            <= SCORER_CALIBRATION_ABS_BIAS_CEILING
            for dimension in ANSWER_DIMENSIONS
        ),
        "within_tolerance_lower_bound": all(
            isinstance(tolerance_map.get(dimension), Mapping)
            and isinstance(tolerance_map[dimension].get("lower"), int | float)
            and float(tolerance_map[dimension]["lower"])
            >= SCORER_CALIBRATION_WITHIN_TOLERANCE_LCB_FLOOR
            for dimension in ANSWER_DIMENSIONS
        ),
        "pairwise_preference_lower_bound": isinstance(
            preference_map.get("lower"), int | float
        )
        and float(preference_map["lower"])
        >= SCORER_CALIBRATION_PREFERENCE_LCB_FLOOR,
        "paired_cases_complete": metrics.get("invalid_pairs") == 0
        and isinstance(metrics.get("pairs"), int)
        and metrics["pairs"] >= SCORER_CALIBRATION_MIN_CLUSTERS,
    }


def build_scorer_calibration_artifact(
    *,
    cases: Sequence[Mapping[str, Any]],
    scorer_identity: Mapping[str, Any],
    frozen_at: str,
    review_protocol_sha256: str,
    review_ledger_file: Path = ANSWER_REVIEW_LEDGER,
    execution_ledger_file: Path = ANSWER_EXECUTION_LEDGER,
) -> dict[str, Any]:
    """Seal case-level reviewed labels and scorer decisions for preregistration."""

    identity = _calibration_scorer_identity(scorer_identity)
    identity_sha = _canonical_sha(identity)
    calibration_run_id = _canonical_sha(
        {
            "artifact_kind": "preregistered-answer-scorer-calibration",
            "frozen_at": _strict_utc(frozen_at),
            "scorer_identity_sha256": identity_sha,
            "review_protocol_sha256": review_protocol_sha256,
        }
    )
    normalized_cases = []
    for value in cases:
        case = {
            **dict(value),
            "scorer_identity_sha256": identity_sha,
        }
        case["cluster_nodes"] = _calibration_cluster_nodes(case)
        case.pop("case_sha256", None)
        case["case_sha256"] = _canonical_sha(case)
        normalized_cases.append(case)
    error = _identity_error(identity, _REQUIRED_CALIBRATION_SCORER_IDENTITY)
    if (
        not _valid_sha(review_protocol_sha256)
        or identity.get("calibration_protocol_sha256") != review_protocol_sha256
    ):
        error = error or "calibration_review_protocol_invalid"
    seen_ids: set[str] = set()
    seen_bindings: set[tuple[str, str, str]] = set()
    seen_review_receipts: set[str] = set()
    seen_execution_receipts: set[str] = set()
    for case in normalized_cases:
        error = error or _calibration_case_error(
            case,
            scorer_identity_sha256=identity_sha,
            review_protocol_sha256=review_protocol_sha256,
            review_ledger_file=review_ledger_file,
            execution_ledger_file=execution_ledger_file,
            calibration_run_id=calibration_run_id,
            frozen_at=frozen_at,
        )
        case_id = str(case.get("case_id") or "")
        binding = (
            str(case.get("session_hash") or ""),
            str(case.get("query_sha256") or ""),
            str(case.get("evidence_sha256") or ""),
        )
        receipt_sha = str(case.get("review_receipt_sha256") or "")
        execution_receipt_sha = str(case.get("execution_receipt_sha256") or "")
        if (
            case_id in seen_ids
            or binding in seen_bindings
            or receipt_sha in seen_review_receipts
            or execution_receipt_sha in seen_execution_receipts
        ):
            error = error or "calibration_case_identity_duplicate"
        seen_ids.add(case_id)
        seen_bindings.add(binding)
        seen_review_receipts.add(receipt_sha)
        seen_execution_receipts.add(execution_receipt_sha)
    if not _strict_utc(frozen_at):
        error = error or "calibration_frozen_at_invalid"
    metrics = _calibration_metrics(normalized_cases) if not error else {}
    gates = _calibration_gates(metrics) if not error else {}
    passed = bool(gates and all(gates.values()))
    return seal_object(
        {
            "schema_version": SCORER_CALIBRATION_SCHEMA_VERSION,
            "artifact_kind": "preregistered-answer-scorer-calibration",
            "status": "passed" if passed else "held",
            "reason": error or ("gate_failed" if not passed else "verified"),
            "frozen_at": _strict_utc(frozen_at),
            "review_protocol_sha256": review_protocol_sha256,
            "calibration_run_id": calibration_run_id,
            "scorer_identity": identity,
            "scorer_identity_sha256": identity_sha,
            "policy": _scorer_calibration_policy(),
            "cases": normalized_cases,
            "metrics": metrics,
            "gates": gates,
        }
    )


def validate_scorer_calibration_artifact(
    value: Path | Mapping[str, Any],
    *,
    scorer_identity: Mapping[str, Any],
    answer_split_manifest: Path | Mapping[str, Any] | None = None,
    evaluated_at: str = "",
    review_ledger_file: Path = ANSWER_REVIEW_LEDGER,
    execution_ledger_file: Path = ANSWER_EXECUTION_LEDGER,
) -> dict[str, Any]:
    """Recompute case-level agreement; a metrics-only envelope fails closed."""

    try:
        payload = (
            read_sealed_json(value)
            if isinstance(value, Path)
            else verify_sealed_object(dict(value))
        )
    except (DurableStateError, TypeError, ValueError):
        return {"passed": False, "reason": "calibration_seal_invalid"}
    identity = _calibration_scorer_identity(scorer_identity)
    identity_sha = _canonical_sha(identity)
    cases = payload.get("cases")
    review_protocol_sha = str(payload.get("review_protocol_sha256") or "")
    expected_calibration_run_id = _canonical_sha(
        {
            "artifact_kind": "preregistered-answer-scorer-calibration",
            "frozen_at": _strict_utc(payload.get("frozen_at")),
            "scorer_identity_sha256": identity_sha,
            "review_protocol_sha256": review_protocol_sha,
        }
    )
    if (
        payload.get("schema_version") != SCORER_CALIBRATION_SCHEMA_VERSION
        or payload.get("artifact_kind")
        != "preregistered-answer-scorer-calibration"
        or payload.get("scorer_identity") != identity
        or payload.get("scorer_identity_sha256") != identity_sha
        or _identity_error(identity, _REQUIRED_CALIBRATION_SCORER_IDENTITY)
        or not _valid_sha(review_protocol_sha)
        or identity.get("calibration_protocol_sha256") != review_protocol_sha
        or payload.get("calibration_run_id") != expected_calibration_run_id
        or not _strict_utc(payload.get("frozen_at"))
        or payload.get("policy") != _scorer_calibration_policy()
        or not isinstance(cases, list)
        or not cases
        or any(not isinstance(case, Mapping) for case in cases)
    ):
        return {"passed": False, "reason": "calibration_shape_or_identity_invalid"}
    seen_ids: set[str] = set()
    seen_bindings: set[tuple[str, str, str]] = set()
    seen_review_receipts: set[str] = set()
    seen_execution_receipts: set[str] = set()
    normalized_cases: list[Mapping[str, Any]] = []
    for case in cases:
        error = _calibration_case_error(
            case,
            scorer_identity_sha256=identity_sha,
            review_protocol_sha256=review_protocol_sha,
            review_ledger_file=review_ledger_file,
            execution_ledger_file=execution_ledger_file,
            calibration_run_id=expected_calibration_run_id,
            frozen_at=str(payload.get("frozen_at") or ""),
        )
        case_id = str(case.get("case_id") or "")
        binding = (
            str(case.get("session_hash") or ""),
            str(case.get("query_sha256") or ""),
            str(case.get("evidence_sha256") or ""),
        )
        receipt_sha = str(case.get("review_receipt_sha256") or "")
        execution_receipt_sha = str(case.get("execution_receipt_sha256") or "")
        if (
            error
            or case_id in seen_ids
            or binding in seen_bindings
            or receipt_sha in seen_review_receipts
            or execution_receipt_sha in seen_execution_receipts
        ):
            return {
                "passed": False,
                "reason": error or "calibration_case_identity_duplicate",
            }
        seen_ids.add(case_id)
        seen_bindings.add(binding)
        seen_review_receipts.add(receipt_sha)
        seen_execution_receipts.add(execution_receipt_sha)
        normalized_cases.append(case)
    metrics = _calibration_metrics(normalized_cases)
    gates = _calibration_gates(metrics)
    contextual_error = ""
    if answer_split_manifest is not None:
        contextual_error = _calibration_split_overlap(
            normalized_cases, answer_split_manifest
        )
    if evaluated_at:
        frozen_at = _strict_utc(payload.get("frozen_at"))
        normalized_evaluated_at = _strict_utc(evaluated_at)
        if (
            not normalized_evaluated_at
            or not frozen_at
            or datetime.fromisoformat(frozen_at.replace("Z", "+00:00"))
            >= datetime.fromisoformat(normalized_evaluated_at.replace("Z", "+00:00"))
        ):
            contextual_error = contextual_error or "calibration_not_frozen_before_evaluation"
    passed = bool(
        payload.get("metrics") == metrics
        and payload.get("gates") == gates
        and gates
        and all(gates.values())
        and payload.get("status") == "passed"
        and payload.get("reason") == "verified"
        and not contextual_error
    )
    return {
        "passed": passed,
        "reason": "verified"
        if passed
        else contextual_error or "calibration_evidence_incomplete",
        "manifest_sha256": str(payload.get("seal_sha256") or ""),
        "scorer_identity_sha256": identity_sha,
        "metrics": metrics,
        "payload": payload,
    }


def evaluate_answer_episodes(
    *,
    runner: AnswerRunner | None,
    scorer: AnswerScorer | None,
    runner_identity: Mapping[str, Any] | None,
    scorer_identity: Mapping[str, Any] | None,
    field_environment_replay: FieldEnvironmentReplay | None = None,
    field_environment_identity: Mapping[str, Any] | None = None,
    episode_file: Path = ANSWER_EPISODE_LEDGER,
    output_file: Path | None = LOCKED_ANSWER_EVAL_ARTIFACT,
    split_manifest: Path | Mapping[str, Any] = ANSWER_SPLIT_MANIFEST,
    gold_manifest: Path | Mapping[str, Any] | None = None,
    scorer_calibration: Path | Mapping[str, Any] | None = None,
    review_ledger_file: Path = ANSWER_REVIEW_LEDGER,
    execution_ledger_file: Path = ANSWER_EXECUTION_LEDGER,
    adapter_registry: Path | Mapping[str, Any] = ANSWER_ADAPTER_REGISTRY,
    confidence: float = ANSWER_AUTHORITY_CONFIDENCE,
    seed: int = ANSWER_AUTHORITY_SEED,
    min_independent_samples: int = 20,
    improvement_point_floor: float = 0.02,
    improvement_lcb_floor: float = 0.0,
    split: str = "locked-test",
    evaluation_kind: str = "historical-context-utility",
) -> dict[str, Any]:
    """Run a preregistered paired replay and seal answer-level evidence."""
    evaluated_at = _now_utc()
    if split not in {"train", "holdout", "locked-test"}:
        raise ValueError("answer evaluation split must be preregistered")
    if evaluation_kind not in {
        "historical-context-utility",
        "field-e2e-replay",
    }:
        raise ValueError("answer evaluation kind is not registered")
    if confidence != ANSWER_AUTHORITY_CONFIDENCE:
        raise ValueError("answer authority confidence is fixed at 0.95")
    if seed != ANSWER_AUTHORITY_SEED:
        raise ValueError("answer authority bootstrap seed is fixed at 1729")
    split_check = validate_split_manifest(split_manifest)
    split_value = split_check.get("manifest", {})
    split_entries = split_check.get("entries", [])
    required_entries = sorted(
        [
            dict(entry)
            for entry in split_entries
            if isinstance(entry, Mapping) and entry.get("split") == split
        ],
        key=lambda entry: str(entry.get("episode_id") or ""),
    )
    selected_ids = [str(entry.get("episode_id") or "") for entry in required_entries]
    latest = {
        str(row.get("episode_id") or ""): row
        for row in _latest_episode_rows(episode_file)
    }
    selected = [latest[episode_id] for episode_id in selected_ids if episode_id in latest]
    eligible_entries, eligible_error = _authority_eligible_entries(episode_file)
    declared_entries = sorted(
        [dict(entry) for entry in split_entries if isinstance(entry, Mapping)],
        key=lambda entry: str(entry.get("episode_id") or ""),
    )
    declared_unsigned = [
        {key: value for key, value in entry.items() if key != "split"}
        for entry in declared_entries
    ]
    ledger_exact = bool(
        split_check.get("passed")
        and not eligible_error
        and declared_unsigned == eligible_entries
        and len(selected) == len(selected_ids)
    )
    if ledger_exact:
        ledger_exact = all(
            row.get("episode_sha256") == entry.get("episode_sha256")
            and row.get("episode_sha256")
            == _canonical_sha(
                {key: value for key, value in row.items() if key != "episode_sha256"}
            )
            for row, entry in zip(selected, required_entries, strict=True)
        )
    gold_check = (
        validate_gold_manifest(
            gold_manifest,
            required_episode_ids=selected_ids,
            review_ledger_file=review_ledger_file,
        )
        if gold_manifest is not None
        else {"passed": False, "reason": "missing_gold_manifest"}
    )
    calibration_check = (
        validate_scorer_calibration_artifact(
            scorer_calibration,
            scorer_identity=scorer_identity or {},
            answer_split_manifest=split_value,
            evaluated_at=evaluated_at,
            review_ledger_file=review_ledger_file,
            execution_ledger_file=execution_ledger_file,
        )
        if scorer_calibration is not None
        else {"passed": False, "reason": "missing_scorer_calibration"}
    )
    evaluation_epoch = datetime.fromisoformat(evaluated_at.replace("Z", "+00:00"))
    preregistration_valid = all(
        normalized
        and datetime.fromisoformat(normalized.replace("Z", "+00:00"))
        < evaluation_epoch
        for normalized in (
            _strict_utc(split_value.get("frozen_at")),
            _strict_utc(gold_check.get("payload", {}).get("frozen_at")),
        )
    )
    required_adapters: list[tuple[str, Any, Mapping[str, Any]]] = []
    if runner is not None:
        required_adapters.append(("runner", runner, runner_identity or {}))
    if scorer is not None:
        required_adapters.append(("scorer", scorer, scorer_identity or {}))
    if field_environment_replay is not None:
        required_adapters.append(
            (
                "field_environment",
                field_environment_replay,
                field_environment_identity or {},
            )
        )
    adapter_check = validate_adapter_registry(
        adapter_registry,
        required=required_adapters,
        evaluated_at=evaluated_at,
    )
    run_receipt = {
        "evaluated_at": evaluated_at,
        "split": split,
        "episode_manifest_sha256": manifest_sha256(
            sorted(str(entry.get("episode_sha256") or "") for entry in required_entries)
        ),
        "episode_ledger_full_set_sha256": manifest_sha256(
            [str(entry.get("episode_sha256") or "") for entry in eligible_entries]
        )
        if not eligible_error
        else "",
        "split_manifest_sha256": str(split_check.get("manifest_sha256") or ""),
        "gold_manifest_sha256": str(gold_check.get("manifest_sha256") or ""),
        "scorer_calibration_sha256": str(
            calibration_check.get("manifest_sha256") or ""
        ),
        "runner_identity_sha256": _canonical_sha(dict(runner_identity or {})),
        "scorer_identity_sha256": _canonical_sha(dict(scorer_identity or {})),
        "field_environment_identity_sha256": _canonical_sha(
            dict(field_environment_identity or {})
        ),
        "adapter_registry_sha256": str(adapter_check.get("manifest_sha256") or ""),
        "evaluation_kind": evaluation_kind,
    }
    run_receipt["receipt_sha256"] = _canonical_sha(run_receipt)
    manifest = {
        "schema_version": ANSWER_EVAL_SCHEMA_VERSION,
        "split": split,
        "evaluation_kind": evaluation_kind,
        "episode_ids": sorted(selected_ids),
        "episode_manifest_sha256": manifest_sha256(
            sorted(str(entry.get("episode_sha256") or "") for entry in required_entries)
        ),
        "episode_ledger_full_set_sha256": run_receipt[
            "episode_ledger_full_set_sha256"
        ],
        "split_manifest_sha256": str(split_check.get("manifest_sha256") or ""),
        "gold_manifest_sha256": str(gold_check.get("manifest_sha256") or ""),
        "scorer_calibration_sha256": str(
            calibration_check.get("manifest_sha256") or ""
        ),
        "runner_identity_sha256": _canonical_sha(dict(runner_identity or {})),
        "scorer_identity_sha256": _canonical_sha(dict(scorer_identity or {})),
        "field_environment_identity_sha256": _canonical_sha(
            dict(field_environment_identity or {})
        ),
        "adapter_registry_sha256": str(adapter_check.get("manifest_sha256") or ""),
        "evaluated_at": evaluated_at,
        "evaluation_run_receipt_sha256": run_receipt["receipt_sha256"],
        "confidence": confidence,
        "seed": seed,
        "minimum_independent_samples": min_independent_samples,
        "improvement_point_floor": improvement_point_floor,
        "improvement_lcb_floor": improvement_lcb_floor,
    }
    manifest["manifest_sha256"] = _canonical_sha(manifest)
    identity_error = (
        str(split_check.get("reason") or "split_manifest_invalid")
        if not split_check.get("passed")
        else "split_has_no_episodes"
        if not selected_ids
        else "episode_ledger_mismatch"
        if not ledger_exact
        else str(gold_check.get("reason") or "gold_manifest_invalid")
        if not gold_check.get("passed")
        else str(calibration_check.get("reason") or "scorer_calibration_invalid")
        if not calibration_check.get("passed")
        else "preregistered_artifact_not_frozen_before_evaluation"
        if not preregistration_valid
        else "historical_context_utility_is_train_only"
        if evaluation_kind == "historical-context-utility" and split != "train"
        else "field_e2e_replay_is_locked_only"
        if evaluation_kind == "field-e2e-replay" and split != "locked-test"
        else "missing_field_environment_replay"
        if evaluation_kind == "field-e2e-replay" and field_environment_replay is None
        else "field_environment_adapter_not_builtin"
        if evaluation_kind == "field-e2e-replay"
        and field_environment_replay is not builtin_field_environment_replay
        else "field_environment_live_identity_mismatch"
        if evaluation_kind == "field-e2e-replay"
        and dict(field_environment_identity or {})
        != builtin_field_environment_identity()
        else str(adapter_check.get("reason") or "adapter_registry_invalid")
        if not adapter_check.get("passed")
        else "missing_runner"
        if runner is None
        else "missing_scorer"
        if scorer is None
        else _identity_error(runner_identity or {}, _REQUIRED_RUNNER_IDENTITY)
        or _identity_error(scorer_identity or {}, _REQUIRED_SCORER_IDENTITY)
        or (
            "scorer_gold_identity_mismatch"
            if (scorer_identity or {}).get("rubric_sha256")
            != gold_check.get("rubric_sha256")
            or (scorer_identity or {}).get("evidence_manifest_sha256")
            != gold_check.get("manifest_sha256")
            else ""
        )
    )
    results: list[dict[str, Any]] = []
    if not identity_error:
        for episode in selected:
            reason = ""
            if episode.get("binding_status") != "verified":
                reason = "episode_binding_unknown"
            elif episode.get("exact_used_subset") is not True:
                reason = "used_subset_unverified"
            elif episode.get("episode_sha256") != _canonical_sha(
                {key: value for key, value in episode.items() if key != "episode_sha256"}
            ):
                reason = "episode_digest_mismatch"
            turn, turn_error = _load_bound_turn(episode)
            reason = reason or turn_error
            if reason or turn is None:
                results.append(
                    {"episode_id": episode.get("episode_id"), "status": "unknown", "reason": reason}
                )
                continue
            pair_protocol = _preregistered_pair_protocol(
                seed=seed,
                episode_id=episode.get("episode_id"),
                episode_sha256=episode.get("episode_sha256"),
                split_manifest_sha256=split_check.get("manifest_sha256"),
                gold_manifest_sha256=gold_check.get("manifest_sha256"),
                adapter_registry_sha256=adapter_check.get("manifest_sha256"),
                evaluation_kind=evaluation_kind,
            )
            pair_seed = int(pair_protocol["pair_seed"])
            generation = dict(pair_protocol["generation"])
            arm_order = list(pair_protocol["arm_order"])
            environment_evidence: dict[str, Any] = {}
            if evaluation_kind == "historical-context-utility":
                context, context_error = _context_for_episode(episode)
                if context_error:
                    results.append(
                        {
                            "episode_id": episode.get("episode_id"),
                            "status": "unknown",
                            "reason": context_error,
                        }
                    )
                    continue
                contexts = {"field_on": context, "field_off": ""}
            else:
                replay_contexts, environment_evidence, replay_error = (
                    _field_environment_contexts(
                        field_environment_replay,  # type: ignore[arg-type]
                        prompt=turn.prompt,
                        episode=episode,
                        pair_seed=pair_seed,
                        identity=field_environment_identity or {},
                        parent_run_id=run_receipt["receipt_sha256"],
                        execution_ledger_file=execution_ledger_file,
                    )
                )
                if replay_error:
                    results.append(
                        {
                            "episode_id": episode.get("episode_id"),
                            "status": "unknown",
                            "reason": replay_error,
                        }
                    )
                    continue
                contexts = {
                    "field_on": replay_contexts["candidate_field"],
                    "field_off": replay_contexts["production_teacher"],
                }
            generated: dict[str, tuple[str, str, str]] = {}
            for arm in arm_order:
                generated[arm] = _runner_answer(
                    runner,
                    turn.prompt,
                    contexts[arm],
                    generation,
                    runner_identity or {},
                    parent_run_id=run_receipt["receipt_sha256"],
                    execution_ledger_file=execution_ledger_file,
                )
            on_answer, on_error, on_runner_receipt = generated["field_on"]
            off_answer, off_error, off_runner_receipt = generated["field_off"]
            gold = gold_check["entries"][str(episode["episode_id"])]
            if _sha_text(str(gold["gold_answer"])) == episode.get("answer_sha256"):
                results.append(
                    {
                        "episode_id": episode.get("episode_id"),
                        "status": "unknown",
                        "reason": "gold_reuses_production_answer",
                    }
                )
                continue
            scoring = dict(pair_protocol["scoring"])
            scored: dict[str, tuple[dict[str, float] | None, str, str]] = {}
            for arm in arm_order:
                answer, runner_error, _runner_receipt = generated[arm]
                scored[arm] = (
                    _score_answer(
                        scorer,
                        turn.prompt,
                        answer,
                        gold,
                        scorer_identity or {},
                        scoring,
                        parent_run_id=run_receipt["receipt_sha256"],
                        execution_ledger_file=execution_ledger_file,
                    )
                    if not runner_error
                    else (None, runner_error, "")
                )
            on_score, on_score_error, on_scorer_receipt = scored["field_on"]
            off_score, off_score_error, off_scorer_receipt = scored["field_off"]
            error = on_error or off_error or on_score_error or off_score_error
            if error or on_score is None or off_score is None:
                results.append(
                    {"episode_id": episode.get("episode_id"), "status": "unknown", "reason": error}
                )
                continue
            used_hashes = {
                page_id: episode["page_content_sha256"][page_id]
                for page_id in episode.get("used_page_ids", [])
            }
            bindings = [
                binding
                for binding in _episode_page_bindings(episode)
                if binding["page_id"] in used_hashes
            ]
            dimensions = {
                dimension: round(on_score[dimension] - off_score[dimension], 9)
                for dimension in ANSWER_DIMENSIONS
            }
            score_delta = round(
                sum(dimensions.values()) / len(ANSWER_DIMENSIONS), 9
            )
            result_row = {
                    "episode_id": episode["episode_id"],
                    "status": "verified",
                    "session_hash": episode["session_hash"],
                    "query_sha256": episode["prompt_sha256"],
                    "decision_id": episode["decision_id"],
                    "used_page_ids": episode["used_page_ids"],
                    "used_page_hashes": used_hashes,
                    "used_page_bindings": bindings,
                    "observed_at": episode["observed_at"],
                    "gold_evidence_sha256": gold["evidence_sha256"],
                    "production_answer_sha256": episode["answer_sha256"],
                    "pair_seed": pair_seed,
                    "pair_protocol_sha256": pair_protocol["protocol_sha256"],
                    "arm_order": arm_order,
                    "scorer_arm_order": arm_order,
                    "environment_evidence": environment_evidence,
                    "field_on": {
                        "answer_sha256": _sha_text(on_answer),
                        "answer_chars": len(on_answer),
                        "dimensions": on_score,
                        "runner_execution_receipt_sha256": on_runner_receipt,
                        "scorer_execution_receipt_sha256": on_scorer_receipt,
                    },
                    "field_off": {
                        "answer_sha256": _sha_text(off_answer),
                        "answer_chars": len(off_answer),
                        "dimensions": off_score,
                        "runner_execution_receipt_sha256": off_runner_receipt,
                        "scorer_execution_receipt_sha256": off_scorer_receipt,
                    },
                    "dimension_deltas": dimensions,
                    "score_delta": score_delta,
                }
            result_row["cluster_nodes"] = _answer_result_cluster_nodes(result_row)
            results.append(result_row)
    verified = [row for row in results if row.get("status") == "verified"]
    bound = cluster_bootstrap_interval(
        verified,
        value_key="score_delta",
        confidence=confidence,
        seed=seed,
    )
    cluster_count = int(bound.get("clusters") or 0)
    dimension_bounds = {
        dimension: cluster_bootstrap_interval(
            [
                {**row, "dimension_delta": row["dimension_deltas"][dimension]}
                for row in verified
            ],
            value_key="dimension_delta",
            confidence=confidence,
            seed=seed,
        )
        for dimension in ANSWER_DIMENSIONS
    }
    gates = {
        "sealed_split_manifest": split_check.get("passed") is True,
        "immutable_gold_manifest": gold_check.get("passed") is True,
        "scorer_calibration": calibration_check.get("passed") is True,
        "preregistered_before_run": preregistration_valid,
        "episode_ledger_exact": ledger_exact,
        "registered_adapters": adapter_check.get("passed") is True,
        "evaluation_kind_authorized": (
            evaluation_kind == "historical-context-utility" and split == "train"
        )
        or (evaluation_kind == "field-e2e-replay" and split == "locked-test"),
        "runner_identity": not bool(identity_error),
        "all_pairs_verified": bool(selected) and len(verified) == len(selected),
        "minimum_independent_samples": cluster_count >= min_independent_samples,
        "valid_confidence_bound": bound.get("valid") is True,
        "authority_protocol_fixed": confidence == ANSWER_AUTHORITY_CONFIDENCE
        and seed == ANSWER_AUTHORITY_SEED,
        "improvement_point": isinstance(bound.get("point"), int | float)
        and float(bound["point"]) >= improvement_point_floor,
        "improvement_lower_bound": isinstance(bound.get("lower"), int | float)
        and float(bound["lower"]) >= improvement_lcb_floor,
        "non_degradation": isinstance(bound.get("lower"), int | float)
        and float(bound["lower"]) >= 0.0,
        "no_leakage": len({str(row.get("episode_id") or "") for row in selected})
        == len(selected_ids)
        == len(selected),
        "dimension_bounds_valid": all(
            item.get("valid") is True for item in dimension_bounds.values()
        ),
    }
    passed = bool(gates and all(value is True for value in gates.values()))
    page_rewards: list[dict[str, Any]] = []
    page_penalties: list[dict[str, Any]] = []
    if all(
        gates[key]
        for key in (
            "sealed_split_manifest",
            "immutable_gold_manifest",
            "scorer_calibration",
            "preregistered_before_run",
            "episode_ledger_exact",
            "registered_adapters",
            "evaluation_kind_authorized",
            "runner_identity",
            "all_pairs_verified",
            "no_leakage",
        )
    ):
        for row in verified:
            delta = float(row["score_delta"])
            if delta == 0.0:
                continue
            for binding in row["used_page_bindings"]:
                common = {
                    "episode_id": row["episode_id"],
                    "decision_id": row["decision_id"],
                    "page_id": binding["page_id"],
                    "page_uid": binding["page_uid"],
                    "content_sha256": binding["content_sha256"],
                    "producer": "verified_answer_pair_v2",
                    "session_hash": row["session_hash"],
                    "query_sha256": row["query_sha256"],
                    "observed_at": row["observed_at"],
                }
                if delta > 0.0:
                    page_rewards.append({**common, "reward": round(delta, 9)})
                else:
                    page_penalties.append({**common, "penalty": round(-delta, 9)})
    payload = {
        "schema_version": ANSWER_EVAL_SCHEMA_VERSION,
        "artifact_kind": "answer-on-off-evaluation",
        "status": "passed" if passed else "held",
        "reason": identity_error or ("gate_failed" if not passed else "verified"),
        "manifest": manifest,
        "split_manifest": split_value,
        "gold_manifest": gold_check.get("payload", {}),
        "scorer_calibration": calibration_check.get("payload", {}),
        "adapter_registry": adapter_check.get("payload", {}),
        "runner_identity": dict(runner_identity or {}),
        "scorer_identity": dict(scorer_identity or {}),
        "field_environment_identity": dict(field_environment_identity or {}),
        "evaluation_run_receipt": run_receipt,
        "production_host_exact_replay_claimed": False,
        "samples": len(selected),
        "verified_samples": len(verified),
        "unknown_samples": len(selected) - len(verified),
        "distinct_clusters": cluster_count,
        "confidence_bound": bound,
        "dimension_bounds": dimension_bounds,
        "pair_success_bound": wilson_interval(len(verified), len(selected), confidence=confidence),
        "gates": gates,
        "results": results,
        "page_rewards": page_rewards,
        "page_penalties": page_penalties,
    }
    sealed = seal_object(payload)
    if output_file is not None:
        write_sealed_json(output_file, payload, backup=True)
    return sealed


def validate_locked_answer_artifact(
    value: Path | Mapping[str, Any],
    *,
    minimum_independent_samples: int = 20,
    episode_file: Path = ANSWER_EPISODE_LEDGER,
    review_ledger_file: Path = ANSWER_REVIEW_LEDGER,
    execution_ledger_file: Path = ANSWER_EXECUTION_LEDGER,
    adapter_registry: Path | Mapping[str, Any] = ANSWER_ADAPTER_REGISTRY,
) -> dict[str, Any]:
    """Validate new answer evidence; old point-only artifacts fail closed."""

    return validate_answer_outcome_artifact(
        value,
        required_split="locked-test",
        minimum_independent_samples=minimum_independent_samples,
        episode_file=episode_file,
        review_ledger_file=review_ledger_file,
        execution_ledger_file=execution_ledger_file,
        adapter_registry=adapter_registry,
    )


def _outcome_page_effects(
    rows: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rewards: list[dict[str, Any]] = []
    penalties: list[dict[str, Any]] = []
    for row in rows:
        delta = row.get("score_delta")
        bindings = row.get("used_page_bindings")
        if (
            isinstance(delta, bool)
            or not isinstance(delta, int | float)
            or float(delta) == 0.0
            or not isinstance(bindings, list)
        ):
            continue
        for binding in bindings:
            if not isinstance(binding, Mapping):
                continue
            common = {
                "episode_id": row.get("episode_id"),
                "decision_id": row.get("decision_id"),
                "page_id": binding.get("page_id"),
                "page_uid": binding.get("page_uid"),
                "content_sha256": binding.get("content_sha256"),
                "producer": "verified_answer_pair_v2",
                "session_hash": row.get("session_hash"),
                "query_sha256": row.get("query_sha256"),
                "observed_at": row.get("observed_at"),
            }
            if float(delta) > 0.0:
                rewards.append({**common, "reward": round(float(delta), 9)})
            else:
                penalties.append({**common, "penalty": round(-float(delta), 9)})
    return rewards, penalties


def _answer_result_cluster_nodes(row: Mapping[str, Any]) -> list[str]:
    bindings_value = row.get("used_page_bindings")
    bindings = bindings_value if isinstance(bindings_value, list) else []
    nodes = [
        f"session:{str(row.get('session_hash') or '')}",
        f"query:{str(row.get('query_sha256') or '')}",
    ]
    for binding in bindings:
        if not isinstance(binding, Mapping):
            continue
        nodes.extend(
            (
                f"page:{str(binding.get('page_id') or '')}",
                f"content:{str(binding.get('content_sha256') or '')}",
            )
        )
        page_uid = str(binding.get("page_uid") or "")
        if page_uid:
            nodes.append(f"uid:{page_uid}")
    environment = row.get("environment_evidence")
    environment_arms = (
        environment.get("arms") if isinstance(environment, Mapping) else None
    )
    if isinstance(environment_arms, Mapping):
        for arm_name in ("candidate_field", "production_teacher"):
            arm = environment_arms.get(arm_name)
            retrieved = (
                arm.get("retrieved_page_bindings")
                if isinstance(arm, Mapping)
                else None
            )
            if not isinstance(retrieved, list):
                continue
            for binding in retrieved:
                if not isinstance(binding, Mapping):
                    continue
                page_id = str(binding.get("page_id") or "")
                page_uid = str(binding.get("page_uid") or "")
                content_sha = str(binding.get("content_sha256") or "")
                if page_id:
                    nodes.append(f"page:{page_id}")
                if page_uid:
                    nodes.append(f"uid:{page_uid}")
                if content_sha:
                    nodes.append(f"content:{content_sha}")
    return list(dict.fromkeys(nodes))


def validate_answer_outcome_artifact(
    value: Path | Mapping[str, Any],
    *,
    required_split: str,
    minimum_independent_samples: int = 20,
    episode_file: Path = ANSWER_EPISODE_LEDGER,
    review_ledger_file: Path = ANSWER_REVIEW_LEDGER,
    execution_ledger_file: Path = ANSWER_EXECUTION_LEDGER,
    adapter_registry: Path | Mapping[str, Any] = ANSWER_ADAPTER_REGISTRY,
) -> dict[str, Any]:
    """Validate one preregistered split without exposing another to learning."""

    try:
        payload = (
            read_sealed_json(value)
            if isinstance(value, Path)
            else verify_sealed_object(dict(value))
        )
    except (DurableStateError, TypeError, ValueError):
        return {"status": "invalid", "passed": False, "reason": "seal_invalid"}
    bound = payload.get("confidence_bound")
    manifest = payload.get("manifest")
    gates = payload.get("gates")
    split_check = validate_split_manifest(payload.get("split_manifest", {}))
    split_entries = split_check.get("entries", [])
    live_entries, live_episode_error = _authority_eligible_entries(episode_file)
    live_episodes = {
        str(row.get("episode_id") or ""): row
        for row in _latest_episode_rows(episode_file)
    }
    declared_entries = sorted(
        [dict(entry) for entry in split_entries if isinstance(entry, Mapping)],
        key=lambda entry: str(entry.get("episode_id") or ""),
    )
    declared_unsigned = [
        {key: value for key, value in entry.items() if key != "split"}
        for entry in declared_entries
    ]
    live_full_set_sha = manifest_sha256(
        [str(entry.get("episode_sha256") or "") for entry in live_entries]
    ) if not live_episode_error else ""
    expected_ids = sorted(
        str(entry.get("episode_id") or "")
        for entry in split_entries
        if isinstance(entry, Mapping) and entry.get("split") == required_split
    )
    split_by_id = {
        str(entry.get("episode_id") or ""): entry
        for entry in split_entries
        if isinstance(entry, Mapping)
    }
    gold_check = validate_gold_manifest(
        payload.get("gold_manifest", {}),
        required_episode_ids=expected_ids,
        review_ledger_file=review_ledger_file,
    )
    results = payload.get("results")
    result_ids = [
        str(row.get("episode_id") or "")
        for row in results
        if isinstance(row, Mapping)
    ] if isinstance(results, list) else []
    runner_value = payload.get("runner_identity")
    scorer_value = payload.get("scorer_identity")
    runner_mapping = runner_value if isinstance(runner_value, Mapping) else {}
    scorer_mapping = scorer_value if isinstance(scorer_value, Mapping) else {}
    field_environment_value = payload.get("field_environment_identity")
    field_environment_mapping = (
        field_environment_value
        if isinstance(field_environment_value, Mapping)
        else {}
    )
    evaluation_kind = (
        str(manifest.get("evaluation_kind") or "")
        if isinstance(manifest, Mapping)
        else ""
    )
    registry_identities: list[tuple[str, Mapping[str, Any]]] = [
        ("runner", runner_mapping),
        ("scorer", scorer_mapping),
    ]
    if evaluation_kind == "field-e2e-replay":
        registry_identities.append(
            ("field_environment", field_environment_mapping)
        )
    adapter_registry_error = _adapter_registry_binding_error(
        payload.get("adapter_registry"),
        live_registry=adapter_registry,
        identities=registry_identities,
        evaluated_at=str(manifest.get("evaluated_at") or "")
        if isinstance(manifest, Mapping)
        else "",
    )
    evaluated_at = (
        str(manifest.get("evaluated_at") or "")
        if isinstance(manifest, Mapping)
        else ""
    )
    calibration_check = validate_scorer_calibration_artifact(
        payload.get("scorer_calibration", {}),
        scorer_identity=scorer_mapping,
        answer_split_manifest=payload.get("split_manifest", {}),
        evaluated_at=evaluated_at,
        review_ledger_file=review_ledger_file,
        execution_ledger_file=execution_ledger_file,
    )
    normalized_evaluated_at = _strict_utc(evaluated_at)
    split_frozen_at = _strict_utc(split_check.get("manifest", {}).get("frozen_at"))
    gold_frozen_at = _strict_utc(gold_check.get("payload", {}).get("frozen_at"))
    preregistration_valid = bool(
        normalized_evaluated_at
        and split_frozen_at
        and gold_frozen_at
        and datetime.fromisoformat(split_frozen_at.replace("Z", "+00:00"))
        < datetime.fromisoformat(normalized_evaluated_at.replace("Z", "+00:00"))
        and datetime.fromisoformat(gold_frozen_at.replace("Z", "+00:00"))
        < datetime.fromisoformat(normalized_evaluated_at.replace("Z", "+00:00"))
    )
    identities_valid = (
        bool(runner_mapping)
        and not _identity_error(runner_mapping, _REQUIRED_RUNNER_IDENTITY)
        and bool(scorer_mapping)
        and not _identity_error(scorer_mapping, _REQUIRED_SCORER_IDENTITY)
        and scorer_mapping.get("rubric_sha256")
        == gold_check.get("rubric_sha256")
        and scorer_mapping.get("evidence_manifest_sha256")
        == gold_check.get("manifest_sha256")
        and not adapter_registry_error
        and (
            not field_environment_mapping
            if evaluation_kind == "historical-context-utility"
            else (
                not _identity_error(
                    field_environment_mapping,
                    _REQUIRED_FIELD_ENVIRONMENT_IDENTITY,
                )
                and dict(field_environment_mapping)
                == builtin_field_environment_identity()
                and any(
                    isinstance(entry, Mapping)
                    and entry.get("kind") == "field_environment"
                    and entry.get("adapter_id")
                    == BUILTIN_FIELD_ENVIRONMENT_ADAPTER_ID
                    and entry.get("callable_sha256")
                    == adapter_callable_sha256(builtin_field_environment_replay)
                    for entry in (
                        payload.get("adapter_registry", {}).get("entries", [])
                        if isinstance(payload.get("adapter_registry"), Mapping)
                        else []
                    )
                )
            )
        )
    )
    raw_confidence = manifest.get("confidence") if isinstance(manifest, Mapping) else None
    confidence_value = (
        ANSWER_AUTHORITY_CONFIDENCE
        if raw_confidence == ANSWER_AUTHORITY_CONFIDENCE
        and not isinstance(raw_confidence, bool)
        else -1.0
    )
    raw_seed = manifest.get("seed") if isinstance(manifest, Mapping) else None
    seed_value = (
        ANSWER_AUTHORITY_SEED
        if raw_seed == ANSWER_AUTHORITY_SEED and not isinstance(raw_seed, bool)
        else 0
    )
    expected_episode_manifest_sha = manifest_sha256(
        sorted(
            str(entry.get("episode_sha256") or "")
            for entry in split_entries
            if isinstance(entry, Mapping) and entry.get("split") == required_split
        )
    )
    expected_run_receipt = {
        "evaluated_at": evaluated_at,
        "split": required_split,
        "episode_manifest_sha256": expected_episode_manifest_sha,
        "episode_ledger_full_set_sha256": live_full_set_sha,
        "split_manifest_sha256": str(split_check.get("manifest_sha256") or ""),
        "gold_manifest_sha256": str(gold_check.get("manifest_sha256") or ""),
        "scorer_calibration_sha256": str(
            calibration_check.get("manifest_sha256") or ""
        ),
        "runner_identity_sha256": _canonical_sha(dict(runner_mapping)),
        "scorer_identity_sha256": _canonical_sha(dict(scorer_mapping)),
        "field_environment_identity_sha256": _canonical_sha(
            dict(field_environment_mapping)
        ),
        "adapter_registry_sha256": str(
            payload.get("adapter_registry", {}).get("seal_sha256")
            if isinstance(payload.get("adapter_registry"), Mapping)
            else ""
        ),
        "evaluation_kind": evaluation_kind,
    }
    expected_run_receipt["receipt_sha256"] = _canonical_sha(expected_run_receipt)
    structure_valid = bool(
        payload.get("schema_version") == ANSWER_EVAL_SCHEMA_VERSION
        and payload.get("artifact_kind") == "answer-on-off-evaluation"
        and payload.get("production_host_exact_replay_claimed") is False
        and isinstance(manifest, Mapping)
        and manifest.get("split") == required_split
        and evaluation_kind
        in {"historical-context-utility", "field-e2e-replay"}
        and (
            (evaluation_kind == "historical-context-utility" and required_split == "train")
            or (evaluation_kind == "field-e2e-replay" and required_split == "locked-test")
        )
        and _manifest_digest_valid(manifest)
        and manifest.get("split_manifest_sha256") == split_check.get("manifest_sha256")
        and manifest.get("gold_manifest_sha256") == gold_check.get("manifest_sha256")
        and manifest.get("scorer_calibration_sha256")
        == calibration_check.get("manifest_sha256")
        and manifest.get("runner_identity_sha256")
        == _canonical_sha(dict(runner_mapping))
        and manifest.get("scorer_identity_sha256")
        == _canonical_sha(dict(scorer_mapping))
        and manifest.get("field_environment_identity_sha256")
        == _canonical_sha(dict(field_environment_mapping))
        and manifest.get("adapter_registry_sha256")
        == expected_run_receipt["adapter_registry_sha256"]
        and _strict_utc(evaluated_at) == evaluated_at
        and manifest.get("evaluation_run_receipt_sha256")
        == expected_run_receipt["receipt_sha256"]
        and confidence_value == ANSWER_AUTHORITY_CONFIDENCE
        and seed_value == ANSWER_AUTHORITY_SEED
        and payload.get("evaluation_run_receipt") == expected_run_receipt
        and manifest.get("episode_ids") == expected_ids
        and not live_episode_error
        and declared_unsigned == live_entries
        and manifest.get("episode_ledger_full_set_sha256") == live_full_set_sha
        and split_check.get("passed") is True
        and gold_check.get("passed") is True
        and calibration_check.get("passed") is True
        and preregistration_valid
        and identities_valid
        and isinstance(results, list)
        and len(results) == len(expected_ids)
        and all(isinstance(row, Mapping) for row in results)
        and result_ids == expected_ids
        and len(result_ids) == len(set(result_ids))
        and isinstance(bound, Mapping)
        and bound.get("valid") is True
        and bound.get("method") == "connected-cluster-bootstrap-percentile"
        and isinstance(gates, Mapping)
        and gates
        and all(value is True for value in gates.values())
    )
    numeric_valid = structure_valid
    verified_rows: list[dict[str, Any]] = []
    if numeric_valid:
        for row in results:
            if row.get("status") != "verified":
                numeric_valid = False
                break
            on_value = row.get("field_on")
            off_value = row.get("field_off")
            on = on_value.get("dimensions") if isinstance(on_value, Mapping) else None
            off = off_value.get("dimensions") if isinstance(off_value, Mapping) else None
            deltas = row.get("dimension_deltas")
            if not isinstance(on, Mapping) or not isinstance(off, Mapping) or not isinstance(deltas, Mapping):
                numeric_valid = False
                break
            expected_delta: dict[str, float] = {}
            for dimension in ANSWER_DIMENSIONS:
                left, right = on.get(dimension), off.get(dimension)
                if (
                    isinstance(left, bool)
                    or isinstance(right, bool)
                    or not isinstance(left, int | float)
                    or not isinstance(right, int | float)
                    or not 0.0 <= float(left) <= 1.0
                    or not 0.0 <= float(right) <= 1.0
                ):
                    numeric_valid = False
                    break
                expected_delta[dimension] = round(float(left) - float(right), 9)
            expected_point = round(sum(expected_delta.values()) / len(ANSWER_DIMENSIONS), 9)
            if not numeric_valid or dict(deltas) != expected_delta or row.get("score_delta") != expected_point:
                numeric_valid = False
                break
            if row.get("cluster_nodes") != _answer_result_cluster_nodes(row):
                numeric_valid = False
                break
            gold_entry = gold_check.get("entries", {}).get(row.get("episode_id"), {})
            bindings = row.get("used_page_bindings")
            used_ids = row.get("used_page_ids")
            used_hashes = row.get("used_page_hashes")
            split_entry = split_by_id.get(str(row.get("episode_id") or ""), {})
            split_bindings = split_entry.get("page_bindings")
            live_episode = live_episodes.get(str(row.get("episode_id") or ""), {})
            turn, turn_error = _load_bound_turn(live_episode)
            parent_run_id = expected_run_receipt["receipt_sha256"]
            pair_protocol = _preregistered_pair_protocol(
                seed=seed_value,
                episode_id=row.get("episode_id"),
                episode_sha256=split_entry.get("episode_sha256"),
                split_manifest_sha256=split_check.get("manifest_sha256"),
                gold_manifest_sha256=gold_check.get("manifest_sha256"),
                adapter_registry_sha256=expected_run_receipt[
                    "adapter_registry_sha256"
                ],
                evaluation_kind=evaluation_kind,
            )
            pair_seed = int(pair_protocol["pair_seed"])
            generation = dict(pair_protocol["generation"])
            arm_order = list(pair_protocol["arm_order"])
            context_error = ""
            environment_error = ""
            if evaluation_kind == "historical-context-utility":
                context, context_error = _context_for_episode(live_episode)
                context_shas = {
                    "field_on": _sha_text(context),
                    "field_off": _sha_text(""),
                }
                if row.get("environment_evidence") != {}:
                    environment_error = "unexpected_field_environment_evidence"
            else:
                evidence = row.get("environment_evidence")
                environment_error = _field_environment_evidence_error(
                    evidence,
                    identity=field_environment_mapping,
                    parent_run_id=parent_run_id,
                    episode_id=row.get("episode_id"),
                    prompt_sha256=_sha_text(turn.prompt) if turn is not None else "",
                    pair_seed=pair_seed,
                    execution_ledger_file=execution_ledger_file,
                )
                evidence_mapping = evidence if isinstance(evidence, Mapping) else {}
                evidence_arms_value = evidence_mapping.get("arms")
                evidence_arms = (
                    evidence_arms_value
                    if isinstance(evidence_arms_value, Mapping)
                    else {}
                )
                candidate_value = evidence_arms.get("candidate_field")
                teacher_value = evidence_arms.get("production_teacher")
                candidate = (
                    candidate_value if isinstance(candidate_value, Mapping) else {}
                )
                teacher = teacher_value if isinstance(teacher_value, Mapping) else {}
                context_shas = {
                    "field_on": str(candidate.get("context_sha256") or ""),
                    "field_off": str(teacher.get("context_sha256") or ""),
                }
            reset_receipt = {
                "seed": pair_seed,
                "base_state_sha256": generation["base_state_sha256"],
                "reset_protocol_sha256": runner_mapping.get("policy_sha256"),
            }
            scoring = dict(pair_protocol["scoring"])
            scorer_reset_receipt = {
                "seed": pair_seed,
                "base_state_sha256": scoring["base_state_sha256"],
                "reset_protocol_sha256": scorer_mapping.get("policy_sha256"),
            }
            execution_errors: list[str] = []
            if turn is None or turn_error or context_error or environment_error:
                execution_errors.append(
                    turn_error
                    or context_error
                    or environment_error
                    or "turn_unavailable"
                )
            else:
                for arm_name, arm_value in (
                    ("field_on", on_value),
                    ("field_off", off_value),
                ):
                    arm_mapping = arm_value if isinstance(arm_value, Mapping) else {}
                    execution_errors.append(
                        _execution_receipt_error(
                            receipt_sha256=arm_mapping.get(
                                "runner_execution_receipt_sha256"
                            ),
                            expected_kind="answer_runner_call",
                            expected_adapter_identity_sha256=_canonical_sha(
                                dict(runner_mapping)
                            ),
                            expected_parent_run_id=parent_run_id,
                            expected_input_payload={
                                "prompt_sha256": _sha_text(turn.prompt),
                                "context_sha256": context_shas[arm_name],
                                "generation": generation,
                            },
                            expected_output_payload={
                                "answer_sha256": arm_mapping.get("answer_sha256"),
                                "answer_chars": arm_mapping.get("answer_chars"),
                                "reset_receipt": reset_receipt,
                            },
                            ledger_file=execution_ledger_file,
                        )
                    )
                    execution_errors.append(
                        _execution_receipt_error(
                            receipt_sha256=arm_mapping.get(
                                "scorer_execution_receipt_sha256"
                            ),
                            expected_kind="answer_scorer_call",
                            expected_adapter_identity_sha256=_canonical_sha(
                                dict(scorer_mapping)
                            ),
                            expected_parent_run_id=parent_run_id,
                            expected_input_payload={
                                "prompt_sha256": _sha_text(turn.prompt),
                                "answer_sha256": arm_mapping.get("answer_sha256"),
                                "gold_evidence_sha256": gold_entry.get(
                                    "evidence_sha256"
                                ),
                                "scoring": scoring,
                            },
                            expected_output_payload={
                                "dimensions": arm_mapping.get("dimensions"),
                                "reset_receipt": scorer_reset_receipt,
                            },
                            ledger_file=execution_ledger_file,
                        )
                    )
            live_used_ids = live_episode.get("used_page_ids")
            live_hashes_value = live_episode.get("page_content_sha256")
            live_hashes = (
                live_hashes_value if isinstance(live_hashes_value, Mapping) else {}
            )
            expected_used_hashes = {
                page_id: live_hashes.get(page_id)
                for page_id in live_used_ids
            } if isinstance(live_used_ids, list) else {}
            expected_bindings = [
                binding
                for binding in _episode_page_bindings(live_episode)
                if isinstance(live_used_ids, list)
                and binding["page_id"] in live_used_ids
            ]
            if (
                row.get("gold_evidence_sha256") != gold_entry.get("evidence_sha256")
                or row.get("pair_seed") != pair_seed
                or row.get("pair_protocol_sha256")
                != pair_protocol["protocol_sha256"]
                or row.get("arm_order") != arm_order
                or row.get("scorer_arm_order") != arm_order
                or any(execution_errors)
                or _sha_text(str(gold_entry.get("gold_answer") or ""))
                == row.get("production_answer_sha256")
                or not _valid_sha(row.get("production_answer_sha256"))
                or not _strict_utc(row.get("observed_at"))
                or not isinstance(bindings, list)
                or not isinstance(used_ids, list)
                or not isinstance(used_hashes, Mapping)
                or row.get("session_hash") != split_entry.get("session_hash")
                or row.get("query_sha256") != split_entry.get("query_sha256")
                or not isinstance(split_bindings, list)
                or any(binding not in split_bindings for binding in bindings)
                or live_episode.get("exact_used_subset") is not True
                or row.get("decision_id") != live_episode.get("decision_id")
                or row.get("observed_at") != live_episode.get("observed_at")
                or row.get("production_answer_sha256")
                != live_episode.get("answer_sha256")
                or row.get("session_hash") != live_episode.get("session_hash")
                or row.get("query_sha256")
                != live_episode.get("prompt_sha256")
                or used_ids != live_used_ids
                or dict(used_hashes) != expected_used_hashes
                or bindings != expected_bindings
                or [binding.get("page_id") for binding in bindings if isinstance(binding, Mapping)]
                != used_ids
                or any(
                    not isinstance(binding, Mapping)
                    or not binding.get("page_id")
                    or not _valid_sha(binding.get("content_sha256"))
                    or used_hashes.get(binding.get("page_id"))
                    != binding.get("content_sha256")
                    for binding in bindings
                )
            ):
                numeric_valid = False
                break
            verified_rows.append(dict(row))
    recomputed = cluster_bootstrap_interval(
        verified_rows,
        value_key="score_delta",
        confidence=confidence_value,
        seed=seed_value,
    )
    bound_valid = numeric_valid and dict(bound) == recomputed
    expected_dimension_bounds = {
        dimension: cluster_bootstrap_interval(
            [
                {**row, "dimension_delta": row["dimension_deltas"][dimension]}
                for row in verified_rows
            ],
            value_key="dimension_delta",
            confidence=confidence_value,
            seed=seed_value,
        )
        for dimension in ANSWER_DIMENSIONS
    } if isinstance(manifest, Mapping) else {}
    dimension_bounds_valid = payload.get("dimension_bounds") == expected_dimension_bounds
    cluster_value = bound.get("clusters") if isinstance(bound, Mapping) else None
    sample_value = payload.get("samples")
    count_valid = (
        isinstance(cluster_value, int)
        and not isinstance(cluster_value, bool)
        and cluster_value >= minimum_independent_samples
        and isinstance(sample_value, int)
        and not isinstance(sample_value, bool)
        and sample_value == len(expected_ids)
        and payload.get("verified_samples") == len(expected_ids)
        and payload.get("unknown_samples") == 0
    )
    expected_rewards, expected_penalties = _outcome_page_effects(verified_rows)
    effects_valid = (
        payload.get("page_rewards") == expected_rewards
        and payload.get("page_penalties") == expected_penalties
    )
    manifest_counts_valid = bool(
        isinstance(manifest, Mapping)
        and manifest.get("episode_manifest_sha256")
        == expected_episode_manifest_sha
        and isinstance(manifest.get("minimum_independent_samples"), int)
        and not isinstance(manifest.get("minimum_independent_samples"), bool)
        and manifest["minimum_independent_samples"] >= minimum_independent_samples
        and isinstance(manifest.get("improvement_point_floor"), int | float)
        and not isinstance(manifest.get("improvement_point_floor"), bool)
        and isinstance(manifest.get("improvement_lcb_floor"), int | float)
        and not isinstance(manifest.get("improvement_lcb_floor"), bool)
    )
    expected_gates = {
        "sealed_split_manifest": split_check.get("passed") is True,
        "immutable_gold_manifest": gold_check.get("passed") is True,
        "scorer_calibration": calibration_check.get("passed") is True,
        "preregistered_before_run": preregistration_valid,
        "episode_ledger_exact": True,
        "registered_adapters": not bool(adapter_registry_error),
        "evaluation_kind_authorized": (
            evaluation_kind == "historical-context-utility"
            and required_split == "train"
        )
        or (
            evaluation_kind == "field-e2e-replay"
            and required_split == "locked-test"
        ),
        "runner_identity": identities_valid,
        "all_pairs_verified": len(verified_rows) == len(expected_ids) and bool(expected_ids),
        "minimum_independent_samples": isinstance(cluster_value, int)
        and isinstance(manifest, Mapping)
        and isinstance(manifest.get("minimum_independent_samples"), int)
        and not isinstance(manifest.get("minimum_independent_samples"), bool)
        and cluster_value >= manifest["minimum_independent_samples"],
        "valid_confidence_bound": recomputed.get("valid") is True,
        "authority_protocol_fixed": confidence_value
        == ANSWER_AUTHORITY_CONFIDENCE
        and seed_value == ANSWER_AUTHORITY_SEED,
        "improvement_point": isinstance(recomputed.get("point"), int | float)
        and isinstance(manifest, Mapping)
        and isinstance(manifest.get("improvement_point_floor"), int | float)
        and not isinstance(manifest.get("improvement_point_floor"), bool)
        and float(recomputed["point"]) >= float(manifest["improvement_point_floor"]),
        "improvement_lower_bound": isinstance(recomputed.get("lower"), int | float)
        and isinstance(manifest, Mapping)
        and isinstance(manifest.get("improvement_lcb_floor"), int | float)
        and not isinstance(manifest.get("improvement_lcb_floor"), bool)
        and float(recomputed["lower"]) >= float(manifest["improvement_lcb_floor"]),
        "non_degradation": isinstance(recomputed.get("lower"), int | float)
        and float(recomputed["lower"]) >= 0.0,
        "no_leakage": len(result_ids) == len(set(result_ids)) == len(expected_ids),
        "dimension_bounds_valid": all(
            item.get("valid") is True for item in expected_dimension_bounds.values()
        ),
    }
    gates_valid = gates == expected_gates
    valid = bool(
        structure_valid
        and numeric_valid
        and bound_valid
        and dimension_bounds_valid
        and count_valid
        and effects_valid
        and manifest_counts_valid
        and gates_valid
        and payload.get("pair_success_bound")
        == wilson_interval(len(verified_rows), len(expected_ids), confidence=confidence_value)
        and payload.get("status") == "passed"
    )
    return {
        "status": str(payload.get("status") or "invalid"),
        "passed": valid,
        "reason": "verified" if valid else "answer_evidence_incomplete",
        "manifest_sha256": str((manifest or {}).get("manifest_sha256") or ""),
        "samples": sample_value if isinstance(sample_value, int) else 0,
        "distinct_clusters": cluster_value if isinstance(cluster_value, int) else 0,
        "method": str(bound.get("method") or "") if isinstance(bound, Mapping) else "",
        "confidence": bound.get("confidence") if isinstance(bound, Mapping) else None,
        "seed": bound.get("seed") if isinstance(bound, Mapping) else None,
        "point": bound.get("point") if isinstance(bound, Mapping) else None,
        "lower": bound.get("lower") if isinstance(bound, Mapping) else None,
        "upper": bound.get("upper") if isinstance(bound, Mapping) else None,
        "page_rewards": payload.get("page_rewards") if valid else [],
        "page_penalties": payload.get("page_penalties") if valid else [],
        "split_manifest_sha256": str(split_check.get("manifest_sha256") or ""),
        "environment_epoch_sha256": (
            _canonical_sha(dict(field_environment_mapping))
            if evaluation_kind == "field-e2e-replay" and field_environment_mapping
            else ""
        ),
    }


def validate_answer_artifact_set(
    *,
    train: Path | Mapping[str, Any],
    locked: Path | Mapping[str, Any],
    minimum_independent_samples: int = 20,
    episode_file: Path = ANSWER_EPISODE_LEDGER,
    review_ledger_file: Path = ANSWER_REVIEW_LEDGER,
    execution_ledger_file: Path = ANSWER_EXECUTION_LEDGER,
    adapter_registry: Path | Mapping[str, Any] = ANSWER_ADAPTER_REGISTRY,
) -> dict[str, Any]:
    """Validate exact, disjoint train/locked subsets of one sealed manifest."""

    train_check = validate_answer_outcome_artifact(
        train,
        required_split="train",
        minimum_independent_samples=minimum_independent_samples,
        episode_file=episode_file,
        review_ledger_file=review_ledger_file,
        execution_ledger_file=execution_ledger_file,
        adapter_registry=adapter_registry,
    )
    locked_check = validate_answer_outcome_artifact(
        locked,
        required_split="locked-test",
        minimum_independent_samples=minimum_independent_samples,
        episode_file=episode_file,
        review_ledger_file=review_ledger_file,
        execution_ledger_file=execution_ledger_file,
        adapter_registry=adapter_registry,
    )
    try:
        train_payload = (
            read_sealed_json(train)
            if isinstance(train, Path)
            else verify_sealed_object(dict(train))
        )
        locked_payload = (
            read_sealed_json(locked)
            if isinstance(locked, Path)
            else verify_sealed_object(dict(locked))
        )
    except (DurableStateError, TypeError, ValueError):
        return {"passed": False, "reason": "artifact_set_seal_invalid"}
    same_manifest = (
        train_check.get("split_manifest_sha256")
        and train_check.get("split_manifest_sha256")
        == locked_check.get("split_manifest_sha256")
    )
    train_manifest_value = train_payload.get("manifest")
    locked_manifest_value = locked_payload.get("manifest")
    train_manifest = (
        train_manifest_value if isinstance(train_manifest_value, Mapping) else {}
    )
    locked_manifest = (
        locked_manifest_value if isinstance(locked_manifest_value, Mapping) else {}
    )
    train_values = train_manifest.get("episode_ids")
    locked_values = locked_manifest.get("episode_ids")
    train_values_valid = isinstance(train_values, list) and all(
        isinstance(value, str) and value for value in train_values
    )
    locked_values_valid = isinstance(locked_values, list) and all(
        isinstance(value, str) and value for value in locked_values
    )
    train_ids = set(train_values) if train_values_valid else set()
    locked_ids = set(locked_values) if locked_values_valid else set()
    train_runner = train_payload.get("runner_identity")
    locked_runner = locked_payload.get("runner_identity")
    train_scorer_value = train_payload.get("scorer_identity")
    locked_scorer_value = locked_payload.get("scorer_identity")
    train_scorer = (
        train_scorer_value if isinstance(train_scorer_value, Mapping) else {}
    )
    locked_scorer = (
        locked_scorer_value if isinstance(locked_scorer_value, Mapping) else {}
    )
    scorer_protocol_keys = _REQUIRED_CALIBRATION_SCORER_IDENTITY
    train_gold_value = train_payload.get("gold_manifest")
    locked_gold_value = locked_payload.get("gold_manifest")
    train_gold = train_gold_value if isinstance(train_gold_value, Mapping) else {}
    locked_gold = locked_gold_value if isinstance(locked_gold_value, Mapping) else {}
    train_calibration_value = train_payload.get("scorer_calibration")
    locked_calibration_value = locked_payload.get("scorer_calibration")
    train_calibration = (
        train_calibration_value
        if isinstance(train_calibration_value, Mapping)
        else {}
    )
    locked_calibration = (
        locked_calibration_value
        if isinstance(locked_calibration_value, Mapping)
        else {}
    )
    train_registry_value = train_payload.get("adapter_registry")
    locked_registry_value = locked_payload.get("adapter_registry")
    train_registry = (
        train_registry_value if isinstance(train_registry_value, Mapping) else {}
    )
    locked_registry = (
        locked_registry_value if isinstance(locked_registry_value, Mapping) else {}
    )
    evaluation_protocol_keys = (
        "schema_version",
        "confidence",
        "seed",
        "minimum_independent_samples",
        "improvement_point_floor",
        "improvement_lcb_floor",
    )
    harness_same = bool(
        isinstance(train_runner, Mapping)
        and isinstance(locked_runner, Mapping)
        and dict(train_runner) == dict(locked_runner)
        and all(
            train_scorer.get(key) == locked_scorer.get(key)
            for key in scorer_protocol_keys
        )
        and train_calibration.get("seal_sha256")
        == locked_calibration.get("seal_sha256")
        and bool(train_calibration.get("seal_sha256"))
        and train_gold.get("gold_family_id") == locked_gold.get("gold_family_id")
        and train_gold.get("review_protocol_sha256")
        == locked_gold.get("review_protocol_sha256")
        and train_manifest.get("evaluation_kind")
        == "historical-context-utility"
        and locked_manifest.get("evaluation_kind") == "field-e2e-replay"
        and not train_payload.get("field_environment_identity")
        and isinstance(locked_payload.get("field_environment_identity"), Mapping)
        and train_registry.get("seal_sha256")
        == locked_registry.get("seal_sha256")
        and bool(train_registry.get("seal_sha256"))
        and all(train_manifest.get(key) == locked_manifest.get(key) for key in evaluation_protocol_keys)
    )
    passed = bool(
        train_check.get("passed")
        and locked_check.get("passed")
        and same_manifest
        and train_values_valid
        and locked_values_valid
        and train_ids.isdisjoint(locked_ids)
        and harness_same
    )
    return {
        "passed": passed,
        "reason": "verified" if passed else "cross_split_validation_failed",
        "split_manifest_sha256": train_check.get("split_manifest_sha256")
        if same_manifest
        else "",
        "train_samples": len(train_ids),
        "locked_samples": len(locked_ids),
        "harness_same": harness_same,
    }


def _load_adapter(spec: str) -> Any:
    module_name, separator, attribute = spec.partition(":")
    if not separator or not module_name or not attribute:
        raise ValueError("adapter must use module:callable syntax")
    adapter = getattr(importlib.import_module(module_name), attribute)
    if not callable(adapter):
        raise TypeError("adapter target is not callable")
    return adapter


def _json_mapping(value: str, *, label: str) -> dict[str, Any]:
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError as exc:
        raise ValueError(f"{label} must be a JSON object") from exc
    if not isinstance(parsed, dict):
        raise ValueError(f"{label} must be a JSON object")
    return parsed


def capture_hook_only(
    *,
    host: str,
    stdin_text: str = "",
    session_file: str | Path | None = None,
    session_id: str = "",
    cwd: str = "",
    dry_run: bool = False,
) -> dict[str, Any]:
    save_output: dict[str, Any] = {}
    for line in reversed(stdin_text.splitlines()):
        try:
            candidate = json.loads(line)
        except (json.JSONDecodeError, TypeError):
            continue
        if isinstance(candidate, dict):
            save_output = candidate
            break
    if save_output.get("status") not in {"saved", "recovered"}:
        return {"status": "held", "reason": "exact_save_output_required", "captured": 0}
    exact_file = str(save_output.get("session_file") or "")
    if session_file and str(Path(session_file).expanduser().resolve(strict=False)) != str(
        Path(exact_file).expanduser().resolve(strict=False)
    ):
        return {"status": "held", "reason": "save_session_file_mismatch", "captured": 0}
    resolved = _resolve_session_file(
        host,
        session_file=exact_file,
        session_id=str(save_output.get("session_id") or ""),
        cwd=str(save_output.get("cwd") or ""),
        hints={},
    )
    return capture_session_answer_episodes(
        host=host,
        session_file=resolved,
        session_id_hint=session_id or str(save_output.get("session_id") or ""),
        cwd_hint=cwd or str(save_output.get("cwd") or ""),
        save_output=save_output,
        dry_run=dry_run,
    )


def status(
    *,
    episode_file: Path = ANSWER_EPISODE_LEDGER,
    artifact_file: Path = LOCKED_ANSWER_EVAL_ARTIFACT,
) -> dict[str, Any]:
    rows = _read_jsonl(episode_file)
    return {
        "status": "ok",
        "episodes": len(rows),
        "verified_bindings": sum(row.get("binding_status") == "verified" for row in rows),
        "unknown_bindings": sum(row.get("binding_status") != "verified" for row in rows),
        "locked_evaluation": validate_locked_answer_artifact(artifact_file),
        "production_exact_replay_available": False,
    }


def main(argv: list[str] | None = None) -> int:
    """Run the explicit answer capture, manifest, evaluation, or status command."""

    parser = argparse.ArgumentParser(description="Capture/evaluate Recall answer episodes.")
    parser.add_argument("--host", choices=["codex", "claude-code"], default="codex")
    parser.add_argument("--hook", action="store_true")
    parser.add_argument("--capture-only", action="store_true")
    parser.add_argument("--session-file")
    parser.add_argument("--session-id", default="")
    parser.add_argument("--cwd", default="")
    parser.add_argument("--status", action="store_true")
    parser.add_argument("--build-split-manifest", action="store_true")
    parser.add_argument("--split-manifest", default=str(ANSWER_SPLIT_MANIFEST))
    parser.add_argument("--gold-manifest")
    parser.add_argument("--scorer-calibration")
    parser.add_argument("--evaluate", action="store_true")
    parser.add_argument("--split", choices=["train", "holdout", "locked-test"], default="locked-test")
    parser.add_argument("--runner-adapter")
    parser.add_argument("--scorer-adapter")
    parser.add_argument("--field-environment-adapter")
    parser.add_argument("--runner-identity-json", default="{}")
    parser.add_argument("--scorer-identity-json", default="{}")
    parser.add_argument("--field-environment-identity-json", default="{}")
    parser.add_argument(
        "--evaluation-kind",
        choices=["historical-context-utility", "field-e2e-replay"],
        default="historical-context-utility",
    )
    parser.add_argument("--adapter-registry", default=str(ANSWER_ADAPTER_REGISTRY))
    parser.add_argument("--review-ledger", default=str(ANSWER_REVIEW_LEDGER))
    parser.add_argument("--execution-ledger", default=str(ANSWER_EXECUTION_LEDGER))
    parser.add_argument("--output")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    if not args.dry_run:
        init_chronovisor()
    if args.build_split_manifest:
        result = write_answer_split_manifest(
            output_file=Path(args.split_manifest).expanduser()
        )
    elif args.evaluate:
        runner = _load_adapter(args.runner_adapter) if args.runner_adapter else None
        scorer = _load_adapter(args.scorer_adapter) if args.scorer_adapter else None
        field_environment = (
            _load_adapter(args.field_environment_adapter)
            if args.field_environment_adapter
            else None
        )
        result = evaluate_answer_episodes(
            runner=runner,
            scorer=scorer,
            runner_identity=_json_mapping(args.runner_identity_json, label="runner identity"),
            scorer_identity=_json_mapping(args.scorer_identity_json, label="scorer identity"),
            field_environment_replay=field_environment,
            field_environment_identity=_json_mapping(
                args.field_environment_identity_json,
                label="field environment identity",
            ),
            split_manifest=Path(args.split_manifest).expanduser(),
            gold_manifest=Path(args.gold_manifest).expanduser() if args.gold_manifest else None,
            scorer_calibration=Path(args.scorer_calibration).expanduser()
            if args.scorer_calibration
            else None,
            split=args.split,
            evaluation_kind=args.evaluation_kind,
            adapter_registry=Path(args.adapter_registry).expanduser(),
            review_ledger_file=Path(args.review_ledger).expanduser(),
            execution_ledger_file=Path(args.execution_ledger).expanduser(),
            output_file=Path(args.output).expanduser() if args.output else None,
        )
    elif args.status:
        result = status()
    else:
        if args.hook and os.environ.get(HOOK_ENABLE_ENV) not in {"1", "true", "True"}:
            result = {"status": "disabled", "reason": f"{HOOK_ENABLE_ENV}=1 is required"}
        else:
            stdin_text = sys.stdin.read() if args.hook else ""
            result = capture_hook_only(
                host=args.host,
                stdin_text=stdin_text,
                session_file=args.session_file,
                session_id=args.session_id,
                cwd=args.cwd,
                dry_run=args.dry_run,
            )
    print(json.dumps(result, ensure_ascii=False, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
