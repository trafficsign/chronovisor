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
from datetime import UTC, datetime, timedelta
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
from chronovisor.core.durable_state import (
    canonical_sha256 as _sealed_canonical_sha,
)
from chronovisor.core.index_store import canonical_document_paths
from chronovisor.core.jsonl_write import append_jsonl_durable
from chronovisor.core.raw_segment import copy_source_interval
from chronovisor.core.raw_store import RawStore
from chronovisor.core.recall_log_schema import (
    join_used_recall_episodes,
    page_ids_from_record,
)
from chronovisor.core.recall_runtime_paths import RECALL_DIR
from chronovisor.core.save_transaction import (
    find_published_save_transaction,
    make_save_transaction,
    parse_save_transaction_receipt,
    save_session_key,
    validate_published_save_receipt,
)
from chronovisor.core.store import (
    CHRONOVISOR_ROOT,
    PAGES_DIR,
    find_page,
    init_chronovisor,
    okf_runtime_operation,
)
from chronovisor.decision.graph_decisions import (
    RECALL_ANSWER_ADJUDICATION_SCHEMA,
    build_recall_answer_adjudication_prompt,
)
from chronovisor.decision.machine_consensus_receipt import (
    GOLD_ENTRY_PRODUCER_POLICY_SHA256,
    SCORER_CALIBRATION_PRODUCER_POLICY_SHA256,
    SEARCH_LABEL_CANDIDATE_PRODUCER_POLICY_SHA256,
    append_machine_consensus_receipt,
    list_machine_consensus_receipts,
    load_machine_consensus_receipt,
    search_label_candidate_packet_error,
    validate_machine_consensus_receipt,
)
from chronovisor.recall.content_correction import complete_turns, source_recall_record
from chronovisor.recall.recall_confidence import (
    cluster_bootstrap_interval,
    cluster_rate_wilson_interval,
    manifest_sha256,
    wilson_interval,
)
from chronovisor.recall.recall_runtime import RECALL_LOG_FILE, RECALL_PULL_LOG_FILE

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
ANSWER_CONSENSUS_LEDGER = RECALL_DIR / "answer-consensus-receipts.jsonl"
ANSWER_EXECUTION_LEDGER = RECALL_DIR / "answer-execution-receipts.jsonl"
ANSWER_ADAPTER_REGISTRY = (
    CHRONOVISOR_ROOT / "runtime" / "recall-answer-eval" / "adapter-registry.json"
)
INDEPENDENT_ANSWER_BENCHMARK = (
    CHRONOVISOR_ROOT
    / "runtime"
    / "recall-answer-eval"
    / "independent-benchmark.json"
)
ANSWER_BENCHMARK_SOURCE_LEDGER_DIR = (
    CHRONOVISOR_ROOT
    / "runtime"
    / "recall-answer-eval"
    / "benchmark-source-ledgers"
)
SEARCH_GOLDEN_FILE = CHRONOVISOR_ROOT / "recall" / "search-golden.jsonl"
SEARCH_MANUAL94_MANIFEST = (
    CHRONOVISOR_ROOT / "runtime" / "search-eval" / "manual-94-manifest.json"
)
HOOK_ENABLE_ENV = "CHRONOVISOR_RECALL_ANSWER_CAPTURE_ENABLED"
ANSWER_EPISODE_SCHEMA_VERSION = 2
LEGACY_ANSWER_EPISODE_SCHEMA_VERSION = 1
ANSWER_SPLIT_SCHEMA_VERSION = 2
ANSWER_EVAL_SCHEMA_VERSION = 3
ANSWER_BENCHMARK_EVAL_SCHEMA_VERSION = 1
ANSWER_DIMENSIONS = ("correctness", "grounding", "citation")
ANSWER_AUTHORITY_CONFIDENCE = 0.95
ANSWER_AUTHORITY_SEED = 1729
AUTHORITY_EMBARGO_SECONDS = 86_400
ANSWER_BENCHMARK_MIN_TRAIN_CLUSTERS = 20
ANSWER_BENCHMARK_MIN_LOCKED_CLUSTERS = 20
SCORER_CALIBRATION_SCHEMA_VERSION = 1
MACHINE_SCORER_CALIBRATION_SCHEMA_VERSION = 2
MACHINE_SCORER_CALIBRATION_MIN_CASES = 40
MACHINE_SCORER_CALIBRATION_MIN_PAIRS = 20
MACHINE_SCORER_CALIBRATION_MIN_CLUSTERS = 20
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
ANSWER_ADAPTER_IDENTITY_SCHEMA = "chronovisor.recall-answer-adapter-identity.v2"
_ANSWER_ROUTE_IDENTITY_KEYS = {
    "role",
    "provider",
    "model",
    "location",
    "model_digest",
}
_REQUIRED_RUNNER_IDENTITY = (
    "identity_schema",
    "runner_id",
    "route_identity",
    "model",
    "model_digest",
    "system_sha256",
    "sampler_sha256",
    "policy_sha256",
)
_REQUIRED_SCORER_IDENTITY = (
    "identity_schema",
    "scorer_id",
    "version",
    "route_identity",
    "model",
    "model_digest",
    "system_sha256",
    "sampler_sha256",
    "policy_sha256",
    "rubric_sha256",
    "evidence_manifest_sha256",
    "calibration_protocol_sha256",
)
_REQUIRED_CALIBRATION_SCORER_IDENTITY = (
    "identity_schema",
    "scorer_id",
    "version",
    "route_identity",
    "model",
    "model_digest",
    "system_sha256",
    "sampler_sha256",
    "policy_sha256",
    "rubric_sha256",
    "calibration_protocol_sha256",
)
_ALLOWED_GOLD_SOURCE_KINDS = frozenset(
    {"human_review", "adjudicated_benchmark"}
)
ANSWER_ADJUDICATION_LANE = "recall_answer_adjudication"


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


def _read_jsonl_strict(path: Path) -> list[dict[str, Any]]:
    """Read a durable JSONL projection without hiding physical corruption."""

    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except FileNotFoundError:
        return []
    except (OSError, UnicodeError) as exc:
        raise DurableStateError("jsonl read failed") from exc
    rows: list[dict[str, Any]] = []
    for line in lines:
        if not line.strip():
            raise DurableStateError("jsonl blank row")
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise DurableStateError("jsonl row invalid") from exc
        if not isinstance(row, dict):
            raise DurableStateError("jsonl row must be an object")
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


DETERMINISTIC_GOLD_PROJECTION_POLICY_SHA256 = GOLD_ENTRY_PRODUCER_POLICY_SHA256
BOUNDED_EVIDENCE_PROJECTION_POLICY_SHA256 = _canonical_sha(
    {
        "version": 1,
        "maximum_page_bytes": 12_000,
        "maximum_total_bytes": 32_000,
        "format": "[PAGE <page_id>]\\n<utf8-prefix>",
        "source": "independent_page_snapshot",
    }
)


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


def _utc_order_key(value: object) -> str:
    normalized = _strict_utc(value)
    if not normalized:
        return ""
    parsed = datetime.fromisoformat(normalized.replace("Z", "+00:00"))
    return parsed.astimezone(UTC).isoformat(timespec="microseconds").replace(
        "+00:00", "Z"
    )


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
        if key == "route_identity":
            continue
        if key == "model_digest":
            if key not in identity or (
                value is not None
                and (not isinstance(value, str) or not value.strip())
            ):
                return f"invalid_{key}"
            continue
        if not isinstance(value, str) or not value.strip():
            return f"missing_{key}"
        if key.endswith("_sha256") and not _valid_sha(value):
            return f"invalid_{key}"
    if "route_identity" in required:
        route = identity.get("route_identity")
        expected_role = (
            "recall.answer.runner"
            if "runner_id" in required
            else "recall.answer.scorer"
        )
        if (
            identity.get("identity_schema") != ANSWER_ADAPTER_IDENTITY_SCHEMA
            or not isinstance(route, Mapping)
            or set(route) != _ANSWER_ROUTE_IDENTITY_KEYS
            or route.get("role") != expected_role
            or not isinstance(route.get("provider"), str)
            or not str(route["provider"]).strip()
            or not isinstance(route.get("model"), str)
            or not str(route["model"]).strip()
            or route.get("location") not in {"local", "remote"}
            or route.get("model") != identity.get("model")
            or route.get("model_digest") != identity.get("model_digest")
            or (
                route.get("provider") == "ollama"
                and route.get("location") == "local"
                and (
                    not isinstance(route.get("model_digest"), str)
                    or not str(route["model_digest"]).strip()
                )
            )
            or (
                (route.get("provider") != "ollama" or route.get("location") != "local")
                and route.get("model_digest") is not None
            )
        ):
            return "invalid_route_identity"
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
        from chronovisor.core.codex_transcript import extract_transcript_slice
    elif host == "claude-code":
        from chronovisor.core.claude_code_transcript import extract_transcript_slice
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
                "prompt_content_sha256": _sha_text(turn.prompt),
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
            if (
                prior is None
                or (not prior_verified and current_verified)
                or (
                    current_verified
                    and isinstance(prior, Mapping)
                    and prior.get("schema_version")
                    == LEGACY_ANSWER_EPISODE_SCHEMA_VERSION
                )
            ):
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
        and (
            not episode.get("prompt_content_sha256")
            or _sha_text(turn.prompt) == episode.get("prompt_content_sha256")
        )
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
    query_sha256 = str(episode.get("prompt_content_sha256") or "")
    if not _valid_sha(query_sha256):
        # V1 persisted the parser's short prompt join hash.  Recover the full
        # content digest from the immutable Raw turn; never reinterpret the
        # short join key as a SHA-256 authority value.
        turn, error = _load_bound_turn(episode)
        query_sha256 = _sha_text(turn.prompt) if turn is not None and not error else ""
    return {
        "episode_id": str(episode.get("episode_id") or ""),
        "episode_sha256": str(episode.get("episode_sha256") or ""),
        "observed_at": str(episode.get("observed_at") or ""),
        "session_hash": str(episode.get("session_hash") or ""),
        "prompt_hash": str(episode.get("prompt_sha256") or ""),
        "query_sha256": query_sha256,
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
        episode.get("schema_version")
        not in {LEGACY_ANSWER_EPISODE_SCHEMA_VERSION, ANSWER_EPISODE_SCHEMA_VERSION}
        or not _valid_sha(episode.get("episode_sha256"))
        or episode.get("episode_sha256") != _canonical_sha(unsigned)
        or not entry.get("episode_id")
        or not _strict_utc(entry.get("observed_at"))
        or not entry.get("session_hash")
        or not entry.get("prompt_hash")
        or not _valid_sha(entry.get("query_sha256"))
        or (
            episode.get("schema_version") == ANSWER_EPISODE_SCHEMA_VERSION
            and episode.get("prompt_content_sha256") != entry.get("query_sha256")
        )
        or not isinstance(bindings, list)
        or not bindings
        or any(
            not isinstance(binding, Mapping)
            or not str(binding.get("page_id") or "")
            or not str(binding.get("page_uid") or "")
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
    for resolved in canonical_document_paths(PAGES_DIR, require_stable=True):
        try:
            corpus_rows.append(
                f"{resolved.relative_to(PAGES_DIR.resolve())}:"
                f"{hashlib.sha256(resolved.read_bytes()).hexdigest()}"
            )
        except OSError:
            continue
    try:
        index_path = CHRONOVISOR_ROOT / ".index" / "pages.json"
        index_sha = hashlib.sha256(index_path.read_bytes()).hexdigest()
    except OSError:
        index_sha = _canonical_sha({"missing": str(index_path)})
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
    """Run current Field and full-search teacher from one in-memory state clone.

    Independent benchmark replay always starts from a new empty state.  Historical
    production-episode replay may still clone the corresponding live session.
    """

    from chronovisor.recall.recall_field import (
        _effective_config,
        queue_teacher_commits,
        run_field_turn,
    )
    from chronovisor.recall.recall_field_candidate import _verify
    from chronovisor.recall.recall_field_schema import (
        RecallFieldState,
        load_recall_field_config,
    )
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

    identity = builtin_field_environment_identity()
    config = _effective_config(load_recall_field_config())
    policy = replace(
        load_policy(),
        log_decisions=False,
        processor_judge_enabled=False,
    )
    session_hash = str(episode.get("session_hash") or "")
    independent_replay = (
        episode.get("evaluation_kind")
        == "independent-benchmark-field-replay"
    )
    if independent_replay:
        base_state = RecallFieldState(
            session_hash=session_hash,
            host="answer-eval",
        )
        observed = float(1_700_000_000 + (seed % 1_000_000))
    else:
        base_state = RecallFieldStore(config=config).load(session_hash)
        observed = max(time.time(), base_state.updated_at_epoch)
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


def _contains_forbidden_gold_input(value: object) -> bool:
    forbidden = {
        "production_answer",
        "production_answer_sha256",
        "field_arm",
        "field_outcome",
        "scorer_output",
        "scorer_scores",
    }
    if isinstance(value, Mapping):
        return any(
            str(key) in forbidden or _contains_forbidden_gold_input(item)
            for key, item in value.items()
        )
    if isinstance(value, list | tuple):
        return any(_contains_forbidden_gold_input(item) for item in value)
    return False


def _project_gold_for_scorer(gold: Mapping[str, Any]) -> dict[str, Any]:
    """Project scorer input to remove forbidden keys recursively.

    We keep all top-level scorer keys while scrubbing forbidden fields from all
    nested objects. This prevents accidentally passing raw treatment payloads
    (such as production answers) into the scorer while still preserving the
    evidence fields used by existing scorers.
    """

    forbidden = {
        "production_answer",
        "production_answer_sha256",
        "field_arm",
        "field_outcome",
        "scorer_output",
        "scorer_scores",
    }

    def _project(value: Any) -> Any:
        if isinstance(value, Mapping):
            return {
                str(key): _project(item)
                for key, item in value.items()
                if str(key) not in forbidden
            }
        if isinstance(value, list | tuple):
            return [_project(item) for item in value]
        return value

    return _project(dict(gold))


def _gold_machine_subject(
    entry: Mapping[str, Any],
    *,
    rubric_sha256: str,
    gold_family_id: str,
    expected_split: str,
    split_epoch_id: str,
) -> dict[str, Any]:
    evidence = entry.get("evidence")
    evidence_map = dict(evidence) if isinstance(evidence, Mapping) else {}
    source_packet = evidence_map.get("source_packet")
    return {
        "schema_version": 1,
        "subject_kind": "gold_entry",
        "episode_id": str(entry.get("episode_id") or ""),
        "gold_answer_sha256": _sha_text(str(entry.get("gold_answer") or "")),
        "evidence_sha256": str(entry.get("evidence_sha256") or ""),
        "source_packet_sha256": str(
            evidence_map.get("source_packet_sha256") or ""
        ),
        "source_packet": dict(source_packet)
        if isinstance(source_packet, Mapping)
        else {},
        "reference_answer_sha256": _sha_text(str(entry.get("gold_answer") or "")),
        "source_frozen_at": str(evidence_map.get("source_frozen_at") or ""),
        "rubric_sha256": rubric_sha256,
        "gold_family_id": gold_family_id,
        "split": expected_split,
        "split_epoch_id": split_epoch_id,
        "producer_kind": "deterministic_evidence_projection",
        "producer_model": None,
        "producer_policy_sha256": DETERMINISTIC_GOLD_PROJECTION_POLICY_SHA256,
        "production_answer_used": False,
    }


def _independent_gold_source_packet_error(packet: object) -> str:
    """Validate a gold packet without consulting Recall treatment output."""

    if not isinstance(packet, Mapping):
        return "independent_gold_source_packet_invalid"
    expected_keys = {
        "schema_version",
        "source_kind",
        "case_id",
        "prompt",
        "prompt_content_sha256",
        "evidence_chunks",
        "reference_evidence_sha256",
        "page_bindings",
        "source_authority_sha256",
        "source_entry_sha256",
        "split",
        "split_epoch_id",
        "component_sha256",
        "source_frozen_at",
        "projection_policy_sha256",
    }
    prompt = packet.get("prompt")
    chunks = packet.get("evidence_chunks")
    bindings = packet.get("page_bindings")
    if (
        set(packet) != expected_keys
        or
        packet.get("schema_version") != 1
        or not isinstance(packet.get("source_kind"), str)
        or packet.get("source_kind")
        not in {"manual94_candidate_seed", "machine_search_label_consensus", "synthetic_fixture"}
        or not isinstance(packet.get("case_id"), str)
        or not str(packet.get("case_id") or "")
        or not isinstance(prompt, str)
        or not prompt
        or packet.get("prompt_content_sha256") != _sha_text(prompt)
        or packet.get("projection_policy_sha256")
        != BOUNDED_EVIDENCE_PROJECTION_POLICY_SHA256
        or not _valid_sha(packet.get("source_authority_sha256"))
        or not _valid_sha(packet.get("source_entry_sha256"))
        or not _valid_sha(packet.get("split_epoch_id"))
        or not isinstance(packet.get("split"), str)
        or packet.get("split") not in {"train", "holdout", "locked-test"}
        or not _strict_utc(packet.get("source_frozen_at"))
        or not isinstance(bindings, list)
        or not bindings
        or not isinstance(chunks, list)
        or len(chunks) != len(bindings)
        or any(
            not isinstance(binding, Mapping)
            or set(binding)
            != {"page_id", "page_uid", "content_sha256", "content_byte_length"}
            or not str(binding.get("page_id") or "")
            or not str(binding.get("page_uid") or "")
            or not _valid_sha(binding.get("content_sha256"))
            or not isinstance(binding.get("content_byte_length"), int)
            or isinstance(binding.get("content_byte_length"), bool)
            or int(binding.get("content_byte_length") or 0) <= 0
            for binding in bindings
        )
        or not _valid_sha(packet.get("component_sha256"))
        or _contains_forbidden_gold_input(packet)
    ):
        return "independent_gold_source_packet_invalid"
    assert isinstance(chunks, list) and isinstance(bindings, list)
    binding_page_ids = [str(binding.get("page_id") or "") for binding in bindings]
    chunk_page_ids = [
        str(chunk.get("page_id") or "") if isinstance(chunk, Mapping) else ""
        for chunk in chunks
    ]
    if (
        len(set(binding_page_ids)) != len(binding_page_ids)
        or len(set(chunk_page_ids)) != len(chunk_page_ids)
        or binding_page_ids != chunk_page_ids
    ):
        return "independent_gold_source_packet_invalid"
    rendered: list[str] = []
    for binding, chunk in zip(bindings, chunks, strict=True):
        if (
            not isinstance(binding, Mapping)
            or not isinstance(chunk, Mapping)
            or set(chunk)
            != {
                "page_id",
                "content_sha256",
                "byte_start",
                "byte_end",
                "excerpt",
                "excerpt_sha256",
                "truncated",
            }
            or chunk.get("page_id") != binding.get("page_id")
            or chunk.get("content_sha256") != binding.get("content_sha256")
            or chunk.get("byte_start") != 0
            or not isinstance(chunk.get("byte_end"), int)
            or isinstance(chunk.get("byte_end"), bool)
            or not 0
            < int(chunk["byte_end"])
            <= min(12_000, int(binding.get("content_byte_length") or 0))
            or not isinstance(chunk.get("excerpt"), str)
            or len(str(chunk["excerpt"]).encode("utf-8")) != chunk.get("byte_end")
            or chunk.get("excerpt_sha256") != _sha_text(str(chunk["excerpt"]))
            or not isinstance(chunk.get("truncated"), bool)
            or chunk.get("truncated")
            is not (
                int(chunk["byte_end"])
                < int(binding.get("content_byte_length") or 0)
            )
        ):
            return "independent_gold_source_packet_invalid"
        rendered.append(f"[PAGE {chunk['page_id']}]\n{chunk['excerpt']}")
    evidence = "\n\n".join(rendered)
    if (
        len(evidence.encode("utf-8")) > 32_000
        or packet.get("reference_evidence_sha256") != _sha_text(evidence)
    ):
        return "independent_gold_source_packet_invalid"
    return ""


def _packet_reference_evidence(packet: Mapping[str, Any]) -> str:
    chunks = packet.get("evidence_chunks")
    if _independent_gold_source_packet_error(packet) or not isinstance(chunks, list):
        raise ValueError("independent gold source packet is invalid")
    return "\n\n".join(
        f"[PAGE {chunk['page_id']}]\n{chunk['excerpt']}"
        for chunk in chunks
        if isinstance(chunk, Mapping)
    )


def _legacy_machine_source_ledger_entry_error(entry: object) -> str:
    if (
        not isinstance(entry, Mapping)
        or set(entry) != {"case_id", "source_entry_sha256", "search_row"}
        or not isinstance(entry.get("case_id"), str)
        or not str(entry.get("case_id") or "")
        or not _valid_sha(entry.get("source_entry_sha256"))
        or not isinstance(entry.get("search_row"), Mapping)
        or entry.get("source_entry_sha256")
        != _canonical_sha(dict(entry["search_row"]))
    ):
        return "machine_answer_source_entry_invalid"
    row = dict(entry["search_row"])
    try:
        from chronovisor.decision.decision_schema_manifest import FRONTIER_LABEL_SCHEMA
        from chronovisor.recall.search_label_contract import (
            label_candidate_payload,
            label_review_artifact_error,
            label_tuple_from_review,
        )

        artifact = row.get("decision_artifact")
        artifact_map = artifact if isinstance(artifact, Mapping) else {}
        authority = artifact_map.get("authority")
        evidence = label_candidate_payload(row)
        review = artifact_map.get("review")
        error = label_review_artifact_error(
            artifact,
            evidence=evidence,
            current_authority=authority,
        )
        expected = (
            tuple(evidence["expected_pages"]),
            tuple(evidence["negative_pages"]),
            tuple(evidence["stale_pages"]),
        )
        execution = review.get("decision_execution") if isinstance(review, Mapping) else None
        fingerprint = (
            execution.get("execution_fingerprint")
            if isinstance(execution, Mapping)
            else None
        )
        artifact_seal = (
            execution.get("decision_artifact_seal_sha256")
            if isinstance(execution, Mapping)
            else None
        )
        from chronovisor.decision.decision_artifact import (
            DecisionArtifactStore,
            default_store_root,
        )

        execution_artifact = (
            DecisionArtifactStore(default_store_root(CHRONOVISOR_ROOT)).load(
                fingerprint
            )
            if isinstance(fingerprint, str)
            else None
        )
        execution_identity = (
            execution_artifact.get("execution_identity")
            if isinstance(execution_artifact, Mapping)
            else None
        )
        execution_provenance = (
            execution_artifact.get("provenance")
            if isinstance(execution_artifact, Mapping)
            else None
        )
        decision = (
            execution_artifact.get("decision")
            if isinstance(execution_artifact, Mapping)
            else None
        )
        schema_fields = set(FRONTIER_LABEL_SCHEMA.get("properties", {}))
        review_decision = (
            {key: review.get(key) for key in schema_fields}
            if isinstance(review, Mapping)
            else None
        )
    except (KeyError, TypeError, ValueError):
        return "machine_answer_source_entry_invalid"
    if (
        error is not None
        or not isinstance(authority, Mapping)
        or authority.get("source") != "configured_runtime_consensus"
        or not isinstance(execution_artifact, Mapping)
        or execution_artifact.get("seal_sha256") != artifact_seal
        or not isinstance(execution_identity, Mapping)
        or execution_identity.get("lane") != "search_label"
        or execution_identity.get("authority_sha256")
        != _sealed_canonical_sha(authority)
        or not isinstance(execution_provenance, Mapping)
        or execution_provenance.get("router_policy") != authority.get("router")
        or decision != review_decision
        or not isinstance(review, Mapping)
        or review.get("decision") != "approved"
        or label_tuple_from_review(dict(review)) != expected
        or row.get("reviewed") is not True
        or row.get("source") not in {"recall_questions", "recall_question"}
        or not isinstance(row.get("query"), str)
        or not str(row.get("query") or "").strip()
        or not isinstance(row.get("expected_pages"), list)
        or not row.get("expected_pages")
        or entry.get("case_id")
        != "search-machine-" + str(entry["source_entry_sha256"])[:32]
    ):
        return "machine_answer_source_entry_invalid"
    return ""


def _legacy_validate_machine_answer_source_ledger(
    value: Path | Mapping[str, Any],
) -> dict[str, Any]:
    try:
        payload = (
            read_sealed_json(value)
            if isinstance(value, Path)
            else verify_sealed_object(dict(value))
        )
    except (DurableStateError, TypeError, ValueError):
        return {"passed": False, "reason": "machine_answer_source_ledger_seal_invalid"}
    entries = payload.get("entries")
    if (
        set(payload)
        != {
            "schema_version",
            "artifact_kind",
            "frozen_at",
            "entries",
            "entries_sha256",
            "seal_sha256",
        }
        or payload.get("schema_version") != 1
        or payload.get("artifact_kind")
        != "machine-search-label-answer-source-ledger"
        or not _strict_utc(payload.get("frozen_at"))
        or not isinstance(entries, list)
        or not entries
        or payload.get("entries_sha256") != _canonical_sha(entries)
    ):
        return {"passed": False, "reason": "machine_answer_source_ledger_invalid"}
    by_sha: dict[str, dict[str, Any]] = {}
    case_ids: set[str] = set()
    for entry in entries:
        error = _legacy_machine_source_ledger_entry_error(entry)
        entry_map = dict(entry) if isinstance(entry, Mapping) else {}
        source_sha = str(entry_map.get("source_entry_sha256") or "")
        case_id = str(entry_map.get("case_id") or "")
        if error or source_sha in by_sha or case_id in case_ids:
            return {
                "passed": False,
                "reason": error or "machine_answer_source_entry_duplicate",
            }
        by_sha[source_sha] = entry_map
        case_ids.add(case_id)
    return {
        "passed": True,
        "reason": "verified_machine_answer_source_ledger",
        "payload": payload,
        "entries": by_sha,
        "manifest_sha256": str(payload["seal_sha256"]),
    }


def _legacy_packet_from_machine_source_entry(
    entry: Mapping[str, Any],
    *,
    source_authority_sha256: str,
    frozen_at: str,
) -> dict[str, Any]:
    from chronovisor.recall.recall_runtime import page_uid_for_id

    row = entry.get("search_row")
    if not isinstance(row, Mapping):
        raise ValueError("machine source row is missing")
    expected_pages = [
        str(page_id)
        for page_id in row.get("expected_pages", [])
        if isinstance(page_id, str) and page_id
    ]
    if not expected_pages:
        raise ValueError("machine source row has no positive page")
    chunks: list[dict[str, Any]] = []
    bindings: list[dict[str, Any]] = []
    for page_id in expected_pages:
        path = find_page(page_id)
        try:
            content = path.read_text(encoding="utf-8") if path else ""
        except (OSError, UnicodeError):
            content = ""
        page_uid = page_uid_for_id(page_id)
        if not content or not page_uid:
            raise ValueError("machine source page binding is unavailable")
        content_bytes = content.encode("utf-8")
        excerpt = content_bytes[:12_000].decode("utf-8", errors="ignore")
        excerpt_bytes = excerpt.encode("utf-8")
        content_sha = _sha_text(content)
        bindings.append(
            {
                "page_id": page_id,
                "page_uid": page_uid,
                "content_sha256": content_sha,
                "content_byte_length": len(content_bytes),
            }
        )
        chunks.append(
            {
                "page_id": page_id,
                "content_sha256": content_sha,
                "byte_start": 0,
                "byte_end": len(excerpt_bytes),
                "excerpt": excerpt,
                "excerpt_sha256": _sha_text(excerpt),
                "truncated": len(excerpt_bytes) < len(content_bytes),
            }
        )
    while len(
        "\n\n".join(
            f"[PAGE {chunk['page_id']}]\n{chunk['excerpt']}" for chunk in chunks
        ).encode("utf-8")
    ) > 32_000:
        chunks.pop()
        bindings.pop()
    if not chunks:
        raise ValueError("machine source evidence projection is empty")
    reference = "\n\n".join(
        f"[PAGE {chunk['page_id']}]\n{chunk['excerpt']}" for chunk in chunks
    )
    prompt = str(row.get("query") or "")
    return {
        "schema_version": 1,
        "source_kind": "machine_search_label_consensus",
        "case_id": str(entry["case_id"]),
        "prompt": prompt,
        "prompt_content_sha256": _sha_text(prompt),
        "evidence_chunks": chunks,
        "reference_evidence_sha256": _sha_text(reference),
        "page_bindings": bindings,
        "source_authority_sha256": source_authority_sha256,
        "source_entry_sha256": str(entry["source_entry_sha256"]),
        "split": "",
        "split_epoch_id": "",
        "component_sha256": "",
        "source_frozen_at": frozen_at,
        "projection_policy_sha256": BOUNDED_EVIDENCE_PROJECTION_POLICY_SHA256,
    }


def _search_label_candidate_subject(packet: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "subject_kind": "search_label_candidate",
        "candidate_preregistration_sha256": str(
            packet.get("candidate_preregistration_sha256") or ""
        ),
        "source_packet_sha256": _sealed_canonical_sha(packet),
        "source_packet": copy.deepcopy(dict(packet)),
        "evidence_sha256": str(packet.get("reference_evidence_sha256") or ""),
        "producer_kind": "deterministic_evidence_projection",
        "producer_model": None,
        "producer_policy_sha256": (
            SEARCH_LABEL_CANDIDATE_PRODUCER_POLICY_SHA256
        ),
        "production_answer_used": False,
    }


def _freeze_search_label_candidate_packet(row: Mapping[str, Any]) -> dict[str, Any]:
    """Freeze one preregistered RQ only while its exact page bytes still match."""

    from chronovisor.recall.search_label_contract import (
        auto_candidate_preregistration_error,
    )

    preregistration_error = auto_candidate_preregistration_error(row)
    if preregistration_error:
        raise ValueError(preregistration_error)
    preregistered_at = _strict_utc(row.get("preregistered_at"))
    if (
        not preregistered_at
        or datetime.fromisoformat(preregistered_at.replace("Z", "+00:00"))
        > datetime.now(UTC) + timedelta(minutes=5)
    ):
        raise ValueError("candidate preregistration time is invalid")
    page_id = str(row.get("source_page") or "")
    path = find_page(page_id)
    try:
        content_bytes = path.read_bytes() if path else b""
    except OSError as exc:
        raise ValueError("candidate page bytes unavailable") from exc
    if (
        not content_bytes
        or hashlib.sha256(content_bytes).hexdigest()
        != row.get("content_sha256")
        or len(content_bytes) != row.get("content_byte_length")
    ):
        raise ValueError("candidate preregistration page drift")
    excerpt = content_bytes[:12_000].decode("utf-8", errors="ignore")
    excerpt_bytes = excerpt.encode("utf-8")
    reference = f"[PAGE {page_id}]\n{excerpt}"
    candidate = {
        "query": str(row.get("query") or ""),
        "expected_pages": [str(value) for value in row.get("expected_pages", [])],
        "negative_pages": [],
        "stale_pages": [],
        "source": str(row.get("source") or ""),
        "source_page": page_id,
        "search_eval_split": str(row.get("split") or ""),
        "split_role": str(row.get("split_role") or ""),
        "language": str(row.get("language") or ""),
        "kind": str(row.get("kind") or ""),
        "preregistered_at": preregistered_at,
        "candidate_preregistration_sha256": str(
            row.get("candidate_sha256") or ""
        ),
        "page_uid": str(row.get("page_uid") or ""),
        "content_sha256": str(row.get("content_sha256") or ""),
        "content_byte_length": row.get("content_byte_length"),
        "projection_policy_sha256": str(
            row.get("projection_policy_sha256") or ""
        ),
    }
    packet = {
        "schema_version": 1,
        "packet_kind": "preregistered_rq_page_evidence",
        "candidate_preregistration_sha256": str(
            row.get("candidate_sha256") or ""
        ),
        "candidate": candidate,
        "page_binding": {
            "page_id": page_id,
            "page_uid": str(row.get("page_uid") or ""),
            "content_sha256": hashlib.sha256(content_bytes).hexdigest(),
            "content_byte_length": len(content_bytes),
        },
        "evidence_chunk": {
            "page_id": page_id,
            "content_sha256": hashlib.sha256(content_bytes).hexdigest(),
            "byte_start": 0,
            "byte_end": len(excerpt_bytes),
            "excerpt": excerpt,
            "excerpt_sha256": hashlib.sha256(excerpt_bytes).hexdigest(),
            "truncated": len(excerpt_bytes) < len(content_bytes),
        },
        "reference_evidence_sha256": _sha_text(reference),
        "projection_policy_sha256": BOUNDED_EVIDENCE_PROJECTION_POLICY_SHA256,
    }
    packet_error = search_label_candidate_packet_error(packet)
    if packet_error:
        raise ValueError(packet_error)
    return packet


def _validated_search_label_candidate_receipt(
    receipt_sha256: object,
    *,
    packet: Mapping[str, Any],
    consensus_ledger_file: Path,
    chronovisor_root: Path,
) -> dict[str, Any]:
    loaded = load_machine_consensus_receipt(
        receipt_sha256, ledger_file=consensus_ledger_file
    )
    if loaded.get("passed") is not True:
        return loaded
    receipt = loaded.get("receipt")
    authority = receipt.get("authority") if isinstance(receipt, Mapping) else None
    if (
        not isinstance(authority, Mapping)
        or authority.get("source") != "configured_runtime_consensus"
    ):
        return {"passed": False, "reason": "machine_source_authority_invalid"}
    subject = _search_label_candidate_subject(packet)
    prompt = build_recall_answer_adjudication_prompt(
        {"subject": subject, "subject_sha256": _sealed_canonical_sha(subject)}
    )
    return validate_machine_consensus_receipt(
        receipt_sha256,
        expected_kind="search_label_candidate_review",
        expected_subject=subject,
        expected_producer_policy_sha256=(
            SEARCH_LABEL_CANDIDATE_PRODUCER_POLICY_SHA256
        ),
        prompt=prompt,
        schema=RECALL_ANSWER_ADJUDICATION_SCHEMA,
        system=None,
        lane=ANSWER_ADJUDICATION_LANE,
        ledger_file=consensus_ledger_file,
        chronovisor_root=chronovisor_root,
        current_authority=authority,
    )


def adjudicate_machine_search_label_candidates(
    *,
    candidate_file: Path,
    consensus_ledger_file: Path = ANSWER_CONSENSUS_LEDGER,
    chronovisor_root: Path = CHRONOVISOR_ROOT,
    max_items: int = 1,
    dry_run: bool = False,
    router_factory: Callable[[str], Any] | None = None,
) -> dict[str, Any]:
    """Freeze and locally adjudicate preregistered RQ candidates incrementally."""

    chain = list_machine_consensus_receipts(ledger_file=consensus_ledger_file)
    if chain.get("passed") is not True:
        return {"status": "held", "reason": str(chain.get("reason") or "ledger_invalid")}
    existing = {
        str(receipt.get("subject", {}).get("candidate_preregistration_sha256") or ""):
        receipt
        for receipt in chain.get("receipts", [])
        if isinstance(receipt, Mapping)
        and receipt.get("kind") == "search_label_candidate_review"
        and isinstance(receipt.get("subject"), Mapping)
    }
    raw_candidates = [
        row
        for row in _read_jsonl(candidate_file)
        if row.get("source") == "recall_questions"
        and isinstance(row.get("candidate_sha256"), str)
    ]
    latest_by_logical_key: dict[tuple[str, str], dict[str, Any]] = {}
    future_candidates = 0
    now_limit = datetime.now(UTC) + timedelta(minutes=5)
    for row in raw_candidates:
        preregistered = _utc_order_key(row.get("preregistered_at"))
        if (
            not preregistered
            or datetime.fromisoformat(preregistered.replace("Z", "+00:00"))
            > now_limit
        ):
            future_candidates += 1
            continue
        logical_key = (
            str(row.get("query") or ""),
            str(row.get("page_uid") or ""),
        )
        incumbent = latest_by_logical_key.get(logical_key)
        rank = (
            _utc_order_key(row.get("preregistered_at")),
            str(row.get("candidate_sha256") or ""),
        )
        incumbent_rank = (
            _utc_order_key(incumbent.get("preregistered_at")),
            str(incumbent.get("candidate_sha256") or ""),
        ) if incumbent is not None else ("", "")
        if rank > incumbent_rank:
            latest_by_logical_key[logical_key] = row
    candidates = sorted(
        latest_by_logical_key.values(),
        key=lambda row: str(row.get("candidate_sha256") or ""),
    )
    superseded = len(raw_candidates) - len(candidates) - future_candidates
    accepted = 0
    already = 0
    stale = future_candidates
    for row in candidates:
        candidate_sha = str(row.get("candidate_sha256") or "")
        prior = existing.get(candidate_sha)
        if isinstance(prior, Mapping):
            subject = prior.get("subject")
            packet = subject.get("source_packet") if isinstance(subject, Mapping) else None
            checked = (
                _validated_search_label_candidate_receipt(
                    prior.get("receipt_sha256"),
                    packet=packet,
                    consensus_ledger_file=consensus_ledger_file,
                    chronovisor_root=chronovisor_root,
                )
                if isinstance(packet, Mapping)
                else {"passed": False, "reason": "machine_source_packet_missing"}
            )
            if checked.get("passed") is not True:
                return {"status": "held", "reason": str(checked.get("reason"))}
            already += 1
            continue
        try:
            packet = _freeze_search_label_candidate_packet(row)
        except (OSError, TypeError, ValueError):
            stale += 1
            continue
        if dry_run or accepted >= max(0, max_items):
            continue
        subject = _search_label_candidate_subject(packet)
        prompt = build_recall_answer_adjudication_prompt(
            {"subject": subject, "subject_sha256": _sealed_canonical_sha(subject)}
        )
        result = append_machine_consensus_receipt(
            kind="search_label_candidate_review",
            subject=subject,
            producer_policy_sha256=SEARCH_LABEL_CANDIDATE_PRODUCER_POLICY_SHA256,
            prompt=prompt,
            schema=RECALL_ANSWER_ADJUDICATION_SCHEMA,
            system=None,
            lane=ANSWER_ADJUDICATION_LANE,
            ledger_file=consensus_ledger_file,
            chronovisor_root=chronovisor_root,
            router_factory=router_factory,
        )
        if result.get("status") != "accepted":
            return {
                "status": str(result.get("status") or "held"),
                "reason": str(result.get("reason") or "candidate_consensus_held"),
                "accepted": accepted,
                "already": already,
                "stale": stale,
            }
        accepted += 1
    pending = max(0, len(candidates) - already - accepted - stale)
    return {
        "status": "complete" if pending == 0 and stale == 0 else "waiting",
        "reason": (
            "verified_machine_search_label_consensus"
            if pending == 0 and stale == 0
            else "candidate_re_preregistration_required"
            if stale
            else "candidate_consensus_pending"
        ),
        "accepted": accepted,
        "already": already,
        "pending": pending,
        "stale": stale,
        "superseded": superseded,
        "dry_run": dry_run,
    }


def _machine_source_entries_from_receipts(
    receipts: Sequence[Mapping[str, Any]],
    *,
    consensus_ledger_file: Path,
    chronovisor_root: Path,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    grouped: dict[tuple[str, str], list[tuple[str, Mapping[str, Any], Mapping[str, Any]]]] = {}
    for receipt in receipts:
        if receipt.get("kind") != "search_label_candidate_review":
            continue
        subject = receipt.get("subject")
        packet = subject.get("source_packet") if isinstance(subject, Mapping) else None
        if not isinstance(packet, Mapping) or search_label_candidate_packet_error(packet):
            raise ValueError("machine source packet is invalid")
        checked = _validated_search_label_candidate_receipt(
            receipt.get("receipt_sha256"),
            packet=packet,
            consensus_ledger_file=consensus_ledger_file,
            chronovisor_root=chronovisor_root,
        )
        if checked.get("passed") is not True:
            raise ValueError(str(checked.get("reason") or "machine source receipt invalid"))
        candidate = packet["candidate"]
        logical_key = (str(candidate["query"]), str(candidate["page_uid"]))
        rank = _utc_order_key(candidate["preregistered_at"])
        if not rank:
            raise ValueError("machine source preregistration timestamp invalid")
        grouped.setdefault(logical_key, []).append((rank, packet, receipt))
    entries: list[dict[str, Any]] = []
    retirements: list[dict[str, Any]] = []
    for logical_key, versions in sorted(grouped.items()):
        versions.sort(
            key=lambda row: (
                row[0],
                str(row[1].get("candidate_preregistration_sha256") or ""),
            )
        )
        _active_rank, active_packet, active_receipt = versions[-1]
        identity = {
            "candidate_preregistration_sha256": active_packet.get(
                "candidate_preregistration_sha256"
            ),
            "source_packet_sha256": _sealed_canonical_sha(active_packet),
            "consensus_receipt_sha256": active_receipt.get("receipt_sha256"),
        }
        source_sha = _canonical_sha(identity)
        entries.append(
            {
                "case_id": "search-machine-" + source_sha[:32],
                "source_entry_sha256": source_sha,
                **identity,
                "source_packet": copy.deepcopy(dict(active_packet)),
            }
        )
        logical_key_sha = _canonical_sha(
            {"query": logical_key[0], "page_uid": logical_key[1]}
        )
        for old_rank, old_packet, old_receipt in versions[:-1]:
            retirements.append(
                {
                    "logical_key_sha256": logical_key_sha,
                    "superseded_candidate_preregistration_sha256": old_packet.get(
                        "candidate_preregistration_sha256"
                    ),
                    "superseded_source_packet_sha256": _sealed_canonical_sha(
                        old_packet
                    ),
                    "superseded_consensus_receipt_sha256": old_receipt.get(
                        "receipt_sha256"
                    ),
                    "superseded_preregistered_at": old_rank,
                    "superseded_by_candidate_preregistration_sha256": (
                        active_packet.get("candidate_preregistration_sha256")
                    ),
                    "superseded_by_source_packet_sha256": _sealed_canonical_sha(
                        active_packet
                    ),
                    "superseded_by_consensus_receipt_sha256": active_receipt.get(
                        "receipt_sha256"
                    ),
                    "superseded_by_preregistered_at": _utc_order_key(
                        active_packet["candidate"]["preregistered_at"]
                    ),
                }
            )
    entries.sort(key=lambda entry: str(entry["source_entry_sha256"]))
    retirements.sort(
        key=lambda row: (
            str(row["logical_key_sha256"]),
            str(row["superseded_preregistered_at"]),
        )
    )
    return entries, retirements


def _machine_source_ledger_entry_error(
    entry: object,
    *,
    consensus_ledger_file: Path,
    chronovisor_root: Path,
    ledger_frozen_at: str,
    receipt_positions: Mapping[str, int],
    frozen_head_position: int,
) -> str:
    if not isinstance(entry, Mapping) or set(entry) != {
        "case_id",
        "source_entry_sha256",
        "candidate_preregistration_sha256",
        "source_packet_sha256",
        "source_packet",
        "consensus_receipt_sha256",
    }:
        return "machine_answer_source_entry_invalid"
    packet = entry.get("source_packet")
    identity = {
        "candidate_preregistration_sha256": entry.get(
            "candidate_preregistration_sha256"
        ),
        "source_packet_sha256": entry.get("source_packet_sha256"),
        "consensus_receipt_sha256": entry.get("consensus_receipt_sha256"),
    }
    source_sha = _canonical_sha(identity)
    if (
        not isinstance(packet, Mapping)
        or search_label_candidate_packet_error(packet)
        or entry.get("candidate_preregistration_sha256")
        != packet.get("candidate_preregistration_sha256")
        or entry.get("source_packet_sha256") != _sealed_canonical_sha(packet)
        or entry.get("source_entry_sha256") != source_sha
        or entry.get("case_id") != f"search-machine-{source_sha[:32]}"
        or not _valid_sha(entry.get("consensus_receipt_sha256"))
    ):
        return "machine_answer_source_entry_invalid"
    checked = _validated_search_label_candidate_receipt(
        entry.get("consensus_receipt_sha256"),
        packet=packet,
        consensus_ledger_file=consensus_ledger_file,
        chronovisor_root=chronovisor_root,
    )
    if checked.get("passed") is not True:
        return str(checked.get("reason") or "machine_answer_source_receipt_invalid")
    receipt = checked.get("receipt")
    candidate = packet.get("candidate") if isinstance(packet, Mapping) else None
    preregistered_at = (
        _utc_order_key(candidate.get("preregistered_at"))
        if isinstance(candidate, Mapping)
        else ""
    )
    created_at = (
        _utc_order_key(receipt.get("created_at"))
        if isinstance(receipt, Mapping)
        else ""
    )
    receipt_position = receipt_positions.get(
        str(entry.get("consensus_receipt_sha256") or ""), -1
    )
    if (
        not preregistered_at
        or not created_at
        or not ledger_frozen_at
        or not preregistered_at < created_at <= ledger_frozen_at
        or receipt_position < 0
        or receipt_position > frozen_head_position
    ):
        return "machine_answer_source_receipt_epoch_invalid"
    return ""


def validate_machine_answer_source_ledger(
    value: Path | Mapping[str, Any],
    *,
    consensus_ledger_file: Path | None = None,
    chronovisor_root: Path = CHRONOVISOR_ROOT,
) -> dict[str, Any]:
    """Validate a source epoch entirely from sealed packet and CAS receipts."""

    try:
        payload = (
            read_sealed_json(value)
            if isinstance(value, Path)
            else verify_sealed_object(dict(value))
        )
    except (DurableStateError, OSError, TypeError, ValueError):
        return {"passed": False, "reason": "machine_answer_source_ledger_seal_invalid"}
    entries = payload.get("entries")
    if (
        set(payload)
        != {
            "schema_version",
            "artifact_kind",
            "frozen_at",
            "entries",
            "entries_sha256",
            "retirements",
            "retirements_sha256",
            "consensus_ledger_path",
            "consensus_ledger_head_sha256",
            "seal_sha256",
        }
        or payload.get("schema_version") != 2
        or payload.get("artifact_kind")
        != "machine-search-label-answer-source-ledger"
        or not _strict_utc(payload.get("frozen_at"))
        or not isinstance(entries, list)
        or not entries
        or payload.get("entries_sha256") != _canonical_sha(entries)
        or not isinstance(payload.get("retirements"), list)
        or payload.get("retirements_sha256")
        != _canonical_sha(payload.get("retirements"))
    ):
        return {"passed": False, "reason": "machine_answer_source_ledger_invalid"}
    bound_consensus_ledger = Path(str(payload.get("consensus_ledger_path") or ""))
    expected_consensus_ledger = (
        chronovisor_root / "recall" / "answer-consensus-receipts.jsonl"
    ).expanduser().resolve(strict=False)
    if (
        not str(payload.get("consensus_ledger_path") or "")
        or bound_consensus_ledger.expanduser().resolve(strict=False)
        != expected_consensus_ledger
        or (
            consensus_ledger_file is not None
            and bound_consensus_ledger.expanduser().resolve(strict=False)
            != consensus_ledger_file.expanduser().resolve(strict=False)
        )
    ):
        return {"passed": False, "reason": "machine_answer_source_ledger_invalid"}
    chain = list_machine_consensus_receipts(ledger_file=bound_consensus_ledger)
    receipts = chain.get("receipts")
    receipt_shas = (
        [str(receipt.get("receipt_sha256") or "") for receipt in receipts]
        if isinstance(receipts, list)
        else []
    )
    frozen_head = str(payload.get("consensus_ledger_head_sha256") or "")
    receipt_positions = {receipt_sha: index for index, receipt_sha in enumerate(receipt_shas)}
    frozen_head_position = receipt_positions.get(frozen_head, -1)
    if (
        chain.get("passed") is not True
        or not _valid_sha(frozen_head)
        or frozen_head == "0" * 64
        or frozen_head_position < 0
    ):
        return {"passed": False, "reason": "machine_answer_consensus_ledger_invalid"}
    try:
        expected_entries, expected_retirements = _machine_source_entries_from_receipts(
            [
                receipt
                for receipt in receipts[: frozen_head_position + 1]
                if isinstance(receipt, Mapping)
            ],
            consensus_ledger_file=bound_consensus_ledger,
            chronovisor_root=chronovisor_root,
        )
    except (OSError, TypeError, ValueError):
        return {"passed": False, "reason": "machine_answer_source_receipt_invalid"}
    if entries != expected_entries or payload.get("retirements") != expected_retirements:
        return {"passed": False, "reason": "machine_answer_source_epoch_invalid"}
    by_sha: dict[str, dict[str, Any]] = {}
    case_ids: set[str] = set()
    for entry in entries:
        error = _machine_source_ledger_entry_error(
            entry,
            consensus_ledger_file=bound_consensus_ledger,
            chronovisor_root=chronovisor_root,
            ledger_frozen_at=_utc_order_key(payload.get("frozen_at")),
            receipt_positions=receipt_positions,
            frozen_head_position=frozen_head_position,
        )
        entry_map = dict(entry) if isinstance(entry, Mapping) else {}
        source_sha = str(entry_map.get("source_entry_sha256") or "")
        case_id = str(entry_map.get("case_id") or "")
        if error or source_sha in by_sha or case_id in case_ids:
            return {
                "passed": False,
                "reason": error or "machine_answer_source_entry_duplicate",
            }
        by_sha[source_sha] = entry_map
        case_ids.add(case_id)
    return {
        "passed": True,
        "reason": "verified_machine_answer_source_ledger",
        "payload": payload,
        "entries": by_sha,
        "manifest_sha256": str(payload["seal_sha256"]),
    }


def _packet_from_machine_source_entry(
    entry: Mapping[str, Any],
    *,
    source_authority_sha256: str,
    frozen_at: str,
) -> dict[str, Any]:
    """Project benchmark input from the frozen receipt packet; never read live pages."""

    source_packet = entry.get("source_packet")
    if not isinstance(source_packet, Mapping) or search_label_candidate_packet_error(
        source_packet
    ):
        raise ValueError("machine source packet is invalid")
    candidate = source_packet["candidate"]
    binding = source_packet["page_binding"]
    chunk = source_packet["evidence_chunk"]
    return {
        "schema_version": 1,
        "source_kind": "machine_search_label_consensus",
        "case_id": str(entry["case_id"]),
        "prompt": str(candidate["query"]),
        "prompt_content_sha256": _sha_text(str(candidate["query"])),
        "evidence_chunks": [copy.deepcopy(dict(chunk))],
        "reference_evidence_sha256": str(
            source_packet["reference_evidence_sha256"]
        ),
        "page_bindings": [copy.deepcopy(dict(binding))],
        "source_authority_sha256": source_authority_sha256,
        "source_entry_sha256": str(entry["source_entry_sha256"]),
        "split": "",
        "split_epoch_id": "",
        "component_sha256": "",
        "source_frozen_at": frozen_at,
        "projection_policy_sha256": BOUNDED_EVIDENCE_PROJECTION_POLICY_SHA256,
    }


def _benchmark_component_assignments(
    packets: Sequence[Mapping[str, Any]],
    *,
    source_authority_sha256: str,
    predecessor_entries: Sequence[Mapping[str, Any]] | None = None,
    predecessor_epoch_id: str = "",
) -> tuple[dict[str, tuple[str, str]], str, dict[str, int]]:
    case_ids = [str(packet.get("case_id") or "") for packet in packets]
    if not case_ids or any(not case_id for case_id in case_ids) or len(case_ids) != len(
        set(case_ids)
    ):
        raise ValueError("benchmark case identity is invalid")
    parent = {case_id: case_id for case_id in case_ids}

    def find(case_id: str) -> str:
        while parent[case_id] != case_id:
            parent[case_id] = parent[parent[case_id]]
            case_id = parent[case_id]
        return case_id

    def union(left: str, right: str) -> None:
        left_root, right_root = find(left), find(right)
        if left_root != right_root:
            parent[max(left_root, right_root)] = min(left_root, right_root)

    identity_nodes: dict[str, list[str]] = {}
    for packet in packets:
        case_id = str(packet["case_id"])
        bindings = packet.get("page_bindings")
        if not isinstance(bindings, list):
            raise ValueError("benchmark page bindings are invalid")
        for binding in bindings:
            if not isinstance(binding, Mapping):
                raise ValueError("benchmark page binding is invalid")
            for field in ("page_id", "page_uid", "content_sha256"):
                identity_nodes.setdefault(
                    f"{field}:{str(binding.get(field) or '')}", []
                ).append(case_id)
    for members in identity_nodes.values():
        for case_id in members[1:]:
            union(members[0], case_id)
    components: dict[str, list[Mapping[str, Any]]] = {}
    for packet in packets:
        components.setdefault(find(str(packet["case_id"])), []).append(packet)
    normalized: list[tuple[str, list[str], list[str]]] = []
    for members in components.values():
        member_ids = sorted(str(packet["case_id"]) for packet in members)
        identity_values = sorted(
            {
                f"{field}:{str(binding.get(field) or '')}"
                for packet in members
                for binding in packet.get("page_bindings", [])
                if isinstance(binding, Mapping)
                for field in ("page_id", "page_uid", "content_sha256")
            }
        )
        component_sha = _canonical_sha(
            {"case_ids": member_ids, "page_identity_nodes": identity_values}
        )
        normalized.append((component_sha, member_ids, identity_values))
    normalized.sort(key=lambda row: row[0])
    predecessor_node_splits: dict[str, str] = {}
    for packet in predecessor_entries or ():
        split = packet.get("split")
        bindings = packet.get("page_bindings")
        if split not in {"train", "holdout", "locked-test"} or not isinstance(
            bindings, list
        ):
            raise ValueError("benchmark predecessor split evidence is invalid")
        for binding in bindings:
            if not isinstance(binding, Mapping):
                raise ValueError("benchmark predecessor binding is invalid")
            for field in ("page_id", "page_uid", "content_sha256"):
                node = f"{field}:{str(binding.get(field) or '')}"
                previous = predecessor_node_splits.setdefault(node, str(split))
                if previous != split:
                    raise ValueError("benchmark predecessor already leaks splits")
    inherited: dict[str, str] = {}
    new_components: list[str] = []
    normalized_by_sha = {row[0]: row for row in normalized}
    for component_sha, _members, identity_values in normalized:
        previous_splits = {
            predecessor_node_splits[node]
            for node in identity_values
            if node in predecessor_node_splits
        }
        if len(previous_splits) > 1:
            raise ValueError("benchmark successor would merge predecessor splits")
        if previous_splits:
            inherited[component_sha] = next(iter(previous_splits))
        else:
            new_components.append(component_sha)
    new_components.sort()
    count = len(new_components)
    train_count = max(1, int(count * 0.70)) if count else 0
    holdout_count = max(1, int(count * 0.20)) if count > 1 else 0
    if train_count + holdout_count >= count:
        holdout_count = max(0, count - train_count - 1)
    new_split: dict[str, str] = {}
    for index, component_sha in enumerate(new_components):
        new_split[component_sha] = (
            "train"
            if index < train_count
            else "holdout"
            if index < train_count + holdout_count
            else "locked-test"
        )
    assignments: dict[str, tuple[str, str]] = {}
    component_manifest: list[dict[str, Any]] = []
    for component_sha in sorted(normalized_by_sha):
        _sha, members, identity_values = normalized_by_sha[component_sha]
        split = inherited.get(component_sha) or new_split[component_sha]
        component_manifest.append(
            {
                "component_sha256": component_sha,
                "case_ids": members,
                "page_identity_nodes": identity_values,
                "split": split,
            }
        )
        assignments.update({case_id: (component_sha, split) for case_id in members})
    split_epoch_id = _canonical_sha(
        {
            "source_authority_sha256": source_authority_sha256,
            "policy": (
                "sticky-page-identity-connected-components-70-20-10-v3"
                if predecessor_entries is not None or predecessor_epoch_id
                else "page-identity-connected-components-70-20-10-v2"
            ),
            **(
                {"predecessor_epoch_id": predecessor_epoch_id or None}
                if predecessor_entries is not None or predecessor_epoch_id
                else {}
            ),
            "components": component_manifest,
        }
    )
    cluster_counts = {
        split: sum(1 for row in component_manifest if row["split"] == split)
        for split in ("train", "holdout", "locked-test")
    }
    return assignments, split_epoch_id, cluster_counts


def validate_independent_answer_benchmark(
    value: Path | Mapping[str, Any],
    *,
    chronovisor_root: Path = CHRONOVISOR_ROOT,
) -> dict[str, Any]:
    try:
        payload = (
            read_sealed_json(value)
            if isinstance(value, Path)
            else verify_sealed_object(dict(value))
        )
    except (DurableStateError, TypeError, ValueError):
        return {"passed": False, "reason": "independent_benchmark_seal_invalid"}
    if payload.get("artifact_kind") == "independent-answer-benchmark-active-pointer":
        if not isinstance(value, Path):
            return {"passed": False, "reason": "independent_benchmark_pointer_invalid"}
        epoch_sha = str(payload.get("epoch_sha256") or "")
        manifest_sha = str(payload.get("manifest_sha256") or "")
        expected_path = value.parent / "benchmarks" / f"{epoch_sha}.json"
        if (
            payload.get("schema_version") != 1
            or not _valid_sha(epoch_sha)
            or not _valid_sha(manifest_sha)
            or payload.get("manifest_path") != str(expected_path)
            or not _strict_utc(payload.get("updated_at"))
        ):
            return {"passed": False, "reason": "independent_benchmark_pointer_invalid"}
        nested = validate_independent_answer_benchmark(
            expected_path, chronovisor_root=chronovisor_root
        )
        if (
            nested.get("passed") is not True
            or nested.get("manifest_sha256") != manifest_sha
            or nested.get("split_epoch_id") != epoch_sha
        ):
            return {"passed": False, "reason": "independent_benchmark_pointer_drift"}
        return nested
    entries = payload.get("entries")
    split_epoch_id = payload.get("split_epoch_id")
    schema_version = payload.get("schema_version")
    source_kind = payload.get("source_kind")
    if (
        schema_version not in {1, 2}
        or payload.get("artifact_kind") != "independent-answer-benchmark-manifest"
        or not isinstance(source_kind, str)
        or source_kind
        not in {"manual94_candidate_seed", "machine_search_label_consensus", "synthetic_fixture"}
        or not _strict_utc(payload.get("frozen_at"))
        or not _valid_sha(payload.get("source_authority_sha256"))
        or not _valid_sha(split_epoch_id)
        or not isinstance(entries, list)
        or not entries
    ):
        return {"passed": False, "reason": "independent_benchmark_shape_invalid"}
    if (
        schema_version == 1
        and source_kind == "machine_search_label_consensus"
    ) or (schema_version == 2 and source_kind != "machine_search_label_consensus"):
        return {"passed": False, "reason": "independent_benchmark_source_version_invalid"}
    if (
        source_kind == "synthetic_fixture"
        and isinstance(value, Path)
        and value.expanduser().resolve(strict=False).is_relative_to(
            CHRONOVISOR_ROOT.expanduser().resolve(strict=False)
        )
    ):
        return {"passed": False, "reason": "synthetic_benchmark_forbidden_in_production"}
    source_ledger_check: dict[str, Any] = {}
    predecessor_check: dict[str, Any] = {}
    predecessor_entries: list[Mapping[str, Any]] | None = (
        [] if schema_version == 2 else None
    )
    predecessor_epoch_id = ""
    if schema_version == 2:
        ledger_sha = str(payload.get("source_ledger_sha256") or "")
        ledger_path = Path(str(payload.get("source_ledger_path") or ""))
        expected_ledger_path = ledger_path.parent / f"{ledger_sha}.json"
        if (
            not _valid_sha(ledger_sha)
            or payload.get("source_authority_sha256") != ledger_sha
            or ledger_path != expected_ledger_path
        ):
            return {"passed": False, "reason": "machine_benchmark_source_ledger_invalid"}
        source_ledger_check = validate_machine_answer_source_ledger(
            ledger_path, chronovisor_root=chronovisor_root
        )
        if (
            source_ledger_check.get("passed") is not True
            or source_ledger_check.get("manifest_sha256") != ledger_sha
        ):
            return {"passed": False, "reason": "machine_benchmark_source_ledger_invalid"}
        predecessor = payload.get("predecessor")
        if predecessor is not None:
            if not isinstance(predecessor, Mapping):
                return {"passed": False, "reason": "benchmark_predecessor_invalid"}
            predecessor_path = Path(str(predecessor.get("manifest_path") or ""))
            predecessor_check = validate_independent_answer_benchmark(
                predecessor_path, chronovisor_root=chronovisor_root
            )
            if (
                predecessor_check.get("passed") is not True
                or predecessor.get("split_epoch_id")
                != predecessor_check.get("split_epoch_id")
                or predecessor.get("manifest_sha256")
                != predecessor_check.get("manifest_sha256")
                or predecessor_path.name
                != f"{predecessor_check.get('split_epoch_id')}.json"
            ):
                return {"passed": False, "reason": "benchmark_predecessor_invalid"}
            predecessor_entries = list(
                predecessor_check.get("entries", {}).values()
            )
            predecessor_epoch_id = str(predecessor_check["split_epoch_id"])
    by_id: dict[str, dict[str, Any]] = {}
    for raw in entries:
        packet = dict(raw) if isinstance(raw, Mapping) else {}
        if (
            _independent_gold_source_packet_error(packet)
            or packet.get("source_kind") != payload.get("source_kind")
            or packet.get("source_authority_sha256")
            != payload.get("source_authority_sha256")
            or packet.get("split_epoch_id") != split_epoch_id
            or packet.get("source_frozen_at") != payload.get("frozen_at")
            or packet.get("case_id") in by_id
        ):
            return {"passed": False, "reason": "independent_benchmark_entry_invalid"}
        by_id[str(packet["case_id"])] = packet
    if schema_version == 2:
        ledger_entries = source_ledger_check.get("entries", {})
        for packet in by_id.values():
            ledger_entry = ledger_entries.get(packet.get("source_entry_sha256"))
            source_packet = (
                ledger_entry.get("source_packet")
                if isinstance(ledger_entry, Mapping)
                else None
            )
            candidate = (
                source_packet.get("candidate")
                if isinstance(source_packet, Mapping)
                else None
            )
            source_binding = (
                source_packet.get("page_binding")
                if isinstance(source_packet, Mapping)
                else None
            )
            source_chunk = (
                source_packet.get("evidence_chunk")
                if isinstance(source_packet, Mapping)
                else None
            )
            if (
                not isinstance(ledger_entry, Mapping)
                or ledger_entry.get("case_id") != packet.get("case_id")
                or not isinstance(candidate, Mapping)
                or candidate.get("query") != packet.get("prompt")
                or not isinstance(source_binding, Mapping)
                or packet.get("page_bindings") != [source_binding]
                or not isinstance(source_chunk, Mapping)
                or packet.get("evidence_chunks") != [source_chunk]
                or packet.get("reference_evidence_sha256")
                != source_packet.get("reference_evidence_sha256")
            ):
                return {"passed": False, "reason": "machine_benchmark_source_join_invalid"}
        if predecessor_check.get("payload", {}).get("source_kind") == (
            "machine_search_label_consensus"
        ):
            previous_sources = {
                str(packet.get("source_entry_sha256") or "")
                for packet in predecessor_entries or []
            }
            current_sources = {
                str(packet.get("source_entry_sha256") or "")
                for packet in by_id.values()
            }
            retired_sources = {
                _canonical_sha(
                    {
                        "candidate_preregistration_sha256": retirement.get(
                            "superseded_candidate_preregistration_sha256"
                        ),
                        "source_packet_sha256": retirement.get(
                            "superseded_source_packet_sha256"
                        ),
                        "consensus_receipt_sha256": retirement.get(
                            "superseded_consensus_receipt_sha256"
                        ),
                    }
                )
                for retirement in source_ledger_check.get("payload", {}).get(
                    "retirements", []
                )
                if isinstance(retirement, Mapping)
            }
            if not (previous_sources - current_sources).issubset(retired_sources):
                return {"passed": False, "reason": "machine_benchmark_source_shrank"}
    try:
        expected_assignments, expected_epoch, expected_cluster_counts = (
            _benchmark_component_assignments(
                list(by_id.values()),
                source_authority_sha256=str(payload["source_authority_sha256"]),
                predecessor_entries=predecessor_entries,
                predecessor_epoch_id=predecessor_epoch_id,
            )
        )
    except (TypeError, ValueError):
        return {"passed": False, "reason": "independent_benchmark_components_invalid"}
    if (
        split_epoch_id != expected_epoch
        or any(
            (packet.get("component_sha256"), packet.get("split"))
            != expected_assignments[case_id]
            for case_id, packet in by_id.items()
        )
    ):
        return {"passed": False, "reason": "independent_benchmark_component_drift"}
    cluster_sets = {
        split: {
            str(packet["component_sha256"])
            for packet in by_id.values()
            if packet.get("split") == split
        }
        for split in ("train", "holdout", "locked-test")
    }
    if any(
        cluster_sets[left] & cluster_sets[right]
        for left, right in (
            ("train", "holdout"),
            ("train", "locked-test"),
            ("holdout", "locked-test"),
        )
    ):
        return {"passed": False, "reason": "independent_benchmark_split_leakage"}
    cluster_counts = {key: len(value) for key, value in cluster_sets.items()}
    if cluster_counts != expected_cluster_counts:
        return {"passed": False, "reason": "independent_benchmark_component_drift"}
    gates = {
        "train_cluster_floor": (
            cluster_counts["train"] >= ANSWER_BENCHMARK_MIN_TRAIN_CLUSTERS
        ),
        "locked_cluster_floor": (
            cluster_counts["locked-test"]
            >= ANSWER_BENCHMARK_MIN_LOCKED_CLUSTERS
        ),
    }
    if (
        payload.get("cluster_counts") != cluster_counts
        or payload.get("promotion_gates") != gates
        or payload.get("promotion_status")
        != ("promotion_ready" if all(gates.values()) else "waiting_for_machine_expansion")
    ):
        return {"passed": False, "reason": "independent_benchmark_gate_invalid"}
    return {
        "passed": True,
        "reason": "verified_independent_benchmark",
        "payload": payload,
        "entries": by_id,
        "manifest_sha256": str(payload["seal_sha256"]),
        "split_epoch_id": str(split_epoch_id),
        "cluster_counts": cluster_counts,
        "promotion_ready": all(gates.values()),
        "manifest_path": str(value) if isinstance(value, Path) else "",
    }


def _create_once_sealed(path: Path, payload: Mapping[str, Any]) -> dict[str, Any]:
    """Create one immutable sealed artifact, or return the identical object."""

    proposed = seal_object(payload)
    with sidecar_exclusive_lock(path):
        if path.exists():
            existing = read_sealed_json(path)
            if existing != proposed:
                raise DurableStateError("immutable artifact conflict")
            return existing
        return write_sealed_json(path, payload, backup=False)


def _publish_versioned_benchmark(
    pointer_path: Path,
    payload: Mapping[str, Any],
    *,
    chronovisor_root: Path = CHRONOVISOR_ROOT,
) -> dict[str, Any]:
    """Publish an immutable epoch, then atomically advance its active pointer."""

    epoch_sha = str(payload.get("split_epoch_id") or "")
    if not _valid_sha(epoch_sha):
        raise DurableStateError("benchmark epoch identity is invalid")
    epoch_path = pointer_path.parent / "benchmarks" / f"{epoch_sha}.json"
    sealed = _create_once_sealed(epoch_path, payload)
    pointer = {
        "schema_version": 1,
        "artifact_kind": "independent-answer-benchmark-active-pointer",
        "epoch_sha256": epoch_sha,
        "manifest_sha256": str(sealed["seal_sha256"]),
        "manifest_path": str(epoch_path),
        "updated_at": _now_utc(),
    }
    with sidecar_exclusive_lock(pointer_path):
        if pointer_path.exists():
            current = read_sealed_json(pointer_path)
            if (
                current.get("epoch_sha256") == epoch_sha
                and current.get("manifest_sha256") == sealed["seal_sha256"]
            ):
                return sealed
        write_sealed_json(pointer_path, pointer, backup=True)
    check = validate_independent_answer_benchmark(
        pointer_path, chronovisor_root=chronovisor_root
    )
    if check.get("passed") is not True:
        raise DurableStateError("benchmark active pointer read-back failed")
    return sealed


def build_independent_answer_benchmark(
    *,
    golden_file: Path = SEARCH_GOLDEN_FILE,
    manual94_manifest: Path = SEARCH_MANUAL94_MANIFEST,
    output_file: Path = INDEPENDENT_ANSWER_BENCHMARK,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Freeze reviewed search truth before any Recall answer evaluation.

    Legacy manual94 human review remains readable, while new machine-reviewed
    rows must first enter the independently sealed search authority surface.
    No captured Recall context, generated answer, or field outcome is read.
    """

    if output_file.exists():
        check = validate_independent_answer_benchmark(output_file)
        return {
            "status": "complete" if check.get("passed") else "held",
            "reason": (
                "promotion_ready"
                if check.get("promotion_ready")
                else "waiting_for_machine_expansion"
                if check.get("passed")
                else str(check.get("reason") or "independent_benchmark_invalid")
            ),
            "artifact": str(output_file),
            "entries": len(check.get("entries", {})),
            "manifest_sha256": str(check.get("manifest_sha256") or ""),
        }
    try:
        manual = json.loads(manual94_manifest.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return {"status": "waiting", "reason": "manual94_manifest_missing"}
    frozen_at = _now_utc()
    manual_unsigned = {
        key: value for key, value in manual.items() if key != "manifest_sha256"
    } if isinstance(manual, Mapping) else {}
    manual_entries = manual.get("entries") if isinstance(manual, Mapping) else None
    if (
        not isinstance(manual_entries, list)
        or len(manual_entries) != 94
        or manual.get("examples") != 94
        or manual.get("manifest_sha256") != _canonical_sha(manual_unsigned)
        or len({str(row.get("entry_sha256") or "") for row in manual_entries if isinstance(row, Mapping)}) != 94
    ):
        return {"status": "held", "reason": "manual94_authority_invalid"}
    from chronovisor.recall.search_label_contract import (
        load_examples,
        sealed_manifest_entry,
    )

    golden_by_entry: dict[str, list[Any]] = {}
    for example in load_examples(golden_file, reviewed_only=True):
        sealed_entry = sealed_manifest_entry(example)
        golden_by_entry.setdefault(str(sealed_entry["entry_sha256"]), []).append(
            example
        )
    source_authority_sha = _canonical_sha(manual)
    from chronovisor.recall.recall_runtime import page_uid_for_id

    packets: list[dict[str, Any]] = []
    for raw_entry in sorted(
        (dict(row) for row in manual_entries if isinstance(row, Mapping)),
        key=lambda row: str(row.get("entry_sha256") or ""),
    ):
        entry_sha = str(raw_entry.get("entry_sha256") or "")
        entry_unsigned = {key: value for key, value in raw_entry.items() if key != "entry_sha256"}
        joined = golden_by_entry.get(entry_sha, [])
        example = joined[0] if len(joined) == 1 else None
        if (
            not _valid_sha(entry_sha)
            or entry_sha != _canonical_sha(entry_unsigned)
            or raw_entry.get("reviewed") is not True
            or example is None
            or sealed_manifest_entry(example) != raw_entry
        ):
            return {"status": "held", "reason": "manual94_search_join_invalid"}
        expected_pages = [
            str(page_id)
            for page_id in raw_entry.get("expected_pages", [])
            if isinstance(page_id, str) and page_id
        ]
        if not expected_pages:
            continue
        evidence_chunks: list[dict[str, Any]] = []
        bindings: list[dict[str, str]] = []
        for page_id in expected_pages:
            path = find_page(page_id)
            try:
                content = path.read_text(encoding="utf-8") if path else ""
            except (OSError, UnicodeError):
                content = ""
            if not content:
                return {"status": "waiting", "reason": "benchmark_page_unavailable"}
            content_sha = _sha_text(content)
            page_uid = page_uid_for_id(page_id)
            if not page_uid:
                return {"status": "waiting", "reason": "benchmark_page_uid_missing"}
            bindings.append(
                {
                    "page_id": page_id,
                    "page_uid": page_uid,
                    "content_sha256": content_sha,
                    "content_byte_length": len(content.encode("utf-8")),
                }
            )
            content_bytes = content.encode("utf-8")
            excerpt = content_bytes[:12_000].decode("utf-8", errors="ignore")
            excerpt_bytes = excerpt.encode("utf-8")
            evidence_chunks.append(
                {
                    "page_id": page_id,
                    "content_sha256": content_sha,
                    "byte_start": 0,
                    "byte_end": len(excerpt_bytes),
                    "excerpt": excerpt,
                    "excerpt_sha256": _sha_text(excerpt),
                    "truncated": len(excerpt_bytes) < len(content_bytes),
                }
            )
        prompt = str(example.query)
        while len(
            "\n\n".join(
                f"[PAGE {chunk['page_id']}]\n{chunk['excerpt']}"
                for chunk in evidence_chunks
            ).encode("utf-8")
        ) > 32_000:
            evidence_chunks.pop()
        if not evidence_chunks:
            return {"status": "held", "reason": "benchmark_projection_empty"}
        reference_evidence = "\n\n".join(
            f"[PAGE {chunk['page_id']}]\n{chunk['excerpt']}"
            for chunk in evidence_chunks
        )
        retained_page_ids = {str(chunk["page_id"]) for chunk in evidence_chunks}
        bindings = [
            binding for binding in bindings if binding["page_id"] in retained_page_ids
        ]
        packet = {
            "schema_version": 1,
            "source_kind": "manual94_candidate_seed",
            "case_id": "search-" + entry_sha[:32],
            "prompt": prompt,
            "prompt_content_sha256": _sha_text(prompt),
            "evidence_chunks": evidence_chunks,
            "reference_evidence_sha256": _sha_text(reference_evidence),
            "page_bindings": bindings,
            "source_authority_sha256": source_authority_sha,
            "source_entry_sha256": entry_sha,
            "split": "",
            "split_epoch_id": "",
            "component_sha256": "",
            "source_frozen_at": frozen_at,
            "projection_policy_sha256": (
                BOUNDED_EVIDENCE_PROJECTION_POLICY_SHA256
            ),
        }
        packets.append(packet)
    if not packets:
        return {"status": "held", "reason": "independent_benchmark_empty"}

    assignments, split_epoch_id, cluster_counts = _benchmark_component_assignments(
        packets, source_authority_sha256=source_authority_sha
    )
    for packet in packets:
        packet["component_sha256"], packet["split"] = assignments[
            str(packet["case_id"])
        ]
        packet["split_epoch_id"] = split_epoch_id
        if _independent_gold_source_packet_error(packet):
            return {"status": "held", "reason": "independent_benchmark_entry_invalid"}
    payload = {
        "schema_version": 1,
        "artifact_kind": "independent-answer-benchmark-manifest",
        "source_kind": "manual94_candidate_seed",
        "frozen_at": frozen_at,
        "source_authority_sha256": source_authority_sha,
        "split_epoch_id": split_epoch_id,
        "cluster_counts": cluster_counts,
        "entries": packets,
    }
    payload["promotion_gates"] = {
        "train_cluster_floor": (
            payload["cluster_counts"]["train"]
            >= ANSWER_BENCHMARK_MIN_TRAIN_CLUSTERS
        ),
        "locked_cluster_floor": (
            payload["cluster_counts"]["locked-test"]
            >= ANSWER_BENCHMARK_MIN_LOCKED_CLUSTERS
        ),
    }
    payload["promotion_status"] = (
        "promotion_ready"
        if all(payload["promotion_gates"].values())
        else "waiting_for_machine_expansion"
    )
    if dry_run:
        return {"status": "waiting", "reason": "dry_run", "entries": len(packets)}
    try:
        sealed = _publish_versioned_benchmark(output_file, payload)
    except (DurableStateError, OSError, ValueError):
        return {"status": "held", "reason": "independent_benchmark_conflict"}
    return {
        "status": "complete",
        "reason": payload["promotion_status"],
        "artifact": str(output_file),
        "entries": len(packets),
        "manifest_sha256": str(sealed["seal_sha256"]),
    }


def build_machine_answer_benchmark_epoch(
    *,
    output_file: Path = INDEPENDENT_ANSWER_BENCHMARK,
    source_ledger_dir: Path = ANSWER_BENCHMARK_SOURCE_LEDGER_DIR,
    consensus_ledger_file: Path = ANSWER_CONSENSUS_LEDGER,
    chronovisor_root: Path = CHRONOVISOR_ROOT,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Freeze exact locally-adjudicated RQ packets into one successor epoch."""

    chain = list_machine_consensus_receipts(ledger_file=consensus_ledger_file)
    if chain.get("passed") is not True:
        return {"status": "held", "reason": str(chain.get("reason") or "ledger_invalid")}
    receipts = chain.get("receipts")
    if not isinstance(receipts, list):
        return {"status": "held", "reason": "machine_consensus_ledger_invalid"}
    try:
        entries, retirements = _machine_source_entries_from_receipts(
            [receipt for receipt in receipts if isinstance(receipt, Mapping)],
            consensus_ledger_file=consensus_ledger_file,
            chronovisor_root=chronovisor_root,
        )
    except (OSError, TypeError, ValueError) as exc:
        return {
            "status": "held",
            "reason": f"machine_source_receipt_invalid:{type(exc).__name__}",
        }
    if len(entries) != len(
        {str(entry["source_entry_sha256"]) for entry in entries}
    ):
        return {"status": "held", "reason": "machine_source_entry_duplicate"}
    if not entries:
        return {"status": "waiting", "reason": "machine_search_labels_pending"}
    frozen_at = _now_utc()
    consensus_head = (
        str(receipts[-1].get("receipt_sha256") or "") if receipts else "0" * 64
    )
    ledger_payload = {
        "schema_version": 2,
        "artifact_kind": "machine-search-label-answer-source-ledger",
        "frozen_at": frozen_at,
        "entries": entries,
        "entries_sha256": _canonical_sha(entries),
        "retirements": retirements,
        "retirements_sha256": _canonical_sha(retirements),
        "consensus_ledger_path": str(consensus_ledger_file),
        "consensus_ledger_head_sha256": consensus_head,
    }
    sealed_ledger = seal_object(ledger_payload)
    ledger_sha = str(sealed_ledger["seal_sha256"])
    ledger_path = source_ledger_dir / f"{ledger_sha}.json"
    ledger_check = validate_machine_answer_source_ledger(
        sealed_ledger,
        consensus_ledger_file=consensus_ledger_file,
        chronovisor_root=chronovisor_root,
    )
    if ledger_check.get("passed") is not True:
        return {"status": "held", "reason": str(ledger_check.get("reason"))}
    predecessor_check: dict[str, Any] = {}
    predecessor_entries: list[Mapping[str, Any]] = []
    predecessor: dict[str, Any] | None = None
    if output_file.exists():
        predecessor_check = validate_independent_answer_benchmark(
            output_file, chronovisor_root=chronovisor_root
        )
        if predecessor_check.get("passed") is not True:
            return {
                "status": "held",
                "reason": str(
                    predecessor_check.get("reason") or "benchmark_predecessor_invalid"
                ),
            }
        predecessor_entries = list(predecessor_check.get("entries", {}).values())
        predecessor_path = str(predecessor_check.get("manifest_path") or "")
        if not predecessor_path:
            return {"status": "held", "reason": "benchmark_predecessor_path_missing"}
        predecessor = {
            "split_epoch_id": str(predecessor_check["split_epoch_id"]),
            "manifest_sha256": str(predecessor_check["manifest_sha256"]),
            "manifest_path": predecessor_path,
        }
        current_payload = predecessor_check.get("payload", {})
        if (
            isinstance(current_payload, Mapping)
            and current_payload.get("source_kind")
            == "machine_search_label_consensus"
            and current_payload.get("source_authority_sha256") == ledger_sha
        ):
            return {
                "status": "complete",
                "reason": str(current_payload.get("promotion_status") or "verified"),
                "artifact": str(output_file),
                "entries": len(predecessor_entries),
                "manifest_sha256": str(predecessor_check["manifest_sha256"]),
            }
    packets: list[dict[str, Any]] = []
    try:
        for entry in entries:
            packets.append(
                _packet_from_machine_source_entry(
                    entry,
                    source_authority_sha256=ledger_sha,
                    frozen_at=frozen_at,
                )
            )
        assignments, split_epoch_id, cluster_counts = (
            _benchmark_component_assignments(
                packets,
                source_authority_sha256=ledger_sha,
                predecessor_entries=predecessor_entries,
                predecessor_epoch_id=(
                    str(predecessor["split_epoch_id"]) if predecessor else ""
                ),
            )
        )
    except (OSError, TypeError, ValueError):
        return {"status": "held", "reason": "machine_benchmark_projection_invalid"}
    for packet in packets:
        packet["component_sha256"], packet["split"] = assignments[
            str(packet["case_id"])
        ]
        packet["split_epoch_id"] = split_epoch_id
        if _independent_gold_source_packet_error(packet):
            return {"status": "held", "reason": "machine_benchmark_packet_invalid"}
    payload: dict[str, Any] = {
        "schema_version": 2,
        "artifact_kind": "independent-answer-benchmark-manifest",
        "source_kind": "machine_search_label_consensus",
        "frozen_at": frozen_at,
        "source_authority_sha256": ledger_sha,
        "source_ledger_sha256": ledger_sha,
        "source_ledger_path": str(ledger_path),
        "predecessor": predecessor,
        "split_epoch_id": split_epoch_id,
        "cluster_counts": cluster_counts,
        "entries": packets,
    }
    payload["promotion_gates"] = {
        "train_cluster_floor": (
            cluster_counts["train"] >= ANSWER_BENCHMARK_MIN_TRAIN_CLUSTERS
        ),
        "locked_cluster_floor": (
            cluster_counts["locked-test"]
            >= ANSWER_BENCHMARK_MIN_LOCKED_CLUSTERS
        ),
    }
    payload["promotion_status"] = (
        "promotion_ready"
        if all(payload["promotion_gates"].values())
        else "waiting_for_machine_expansion"
    )
    if dry_run:
        return {
            "status": "waiting",
            "reason": "dry_run",
            "entries": len(entries),
            "cluster_counts": cluster_counts,
        }
    try:
        _create_once_sealed(ledger_path, ledger_payload)
        sealed = _publish_versioned_benchmark(
            output_file, payload, chronovisor_root=chronovisor_root
        )
    except (DurableStateError, OSError, ValueError):
        return {"status": "held", "reason": "machine_benchmark_publish_conflict"}
    return {
        "status": "complete",
        "reason": str(payload["promotion_status"]),
        "artifact": str(output_file),
        "entries": len(packets),
        "manifest_sha256": str(sealed["seal_sha256"]),
    }


def build_machine_gold_cycle(
    *,
    split: str,
    benchmark_manifest: Path | Mapping[str, Any] = INDEPENDENT_ANSWER_BENCHMARK,
    candidate_file: Path | None = None,
    output_file: Path | None = None,
    consensus_ledger_file: Path = ANSWER_CONSENSUS_LEDGER,
    chronovisor_root: Path = CHRONOVISOR_ROOT,
    router_factory: Callable[[str], Any] | None = None,
    max_items: int = 1,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Incrementally create source-backed gold without any answer/outcome input."""

    if split not in {"train", "holdout", "locked-test"}:
        return {"status": "held", "reason": "machine_gold_split_invalid"}
    if not isinstance(benchmark_manifest, Path):
        return {"status": "held", "reason": "machine_gold_pinned_benchmark_required"}
    benchmark_check = validate_independent_answer_benchmark(
        benchmark_manifest, chronovisor_root=chronovisor_root
    )
    if benchmark_check.get("passed") is not True:
        return {
            "status": "waiting",
            "reason": str(
                benchmark_check.get("reason") or "independent_benchmark_missing"
            ),
        }
    benchmark = benchmark_check["payload"]
    if benchmark.get("source_kind") != "machine_search_label_consensus":
        return {"status": "waiting", "reason": "machine_benchmark_authority_required"}
    if benchmark_check.get("promotion_ready") is not True:
        return {"status": "waiting", "reason": "waiting_for_machine_expansion"}
    split_epoch_id = str(benchmark_check["split_epoch_id"])
    source_frozen_at = _strict_utc(benchmark.get("frozen_at"))
    selected = [
        dict(entry)
        for entry in benchmark_check.get("entries", {}).values()
        if entry.get("split") == split
    ]
    selected.sort(key=lambda entry: str(entry.get("case_id") or ""))
    if not selected:
        return {"status": "waiting", "reason": "machine_gold_split_empty"}
    runtime_root = chronovisor_root / "runtime" / "recall-answer-eval"
    pointer_path = output_file or runtime_root / f"{split}-gold-manifest.json"
    candidates_path = candidate_file or (
        runtime_root / "gold-candidates" / split / f"{split_epoch_id}.jsonl"
    )
    artifact_path = runtime_root / "gold" / split / f"{split_epoch_id}.json"
    rubric_sha = _canonical_sha(
        {
            "version": 1,
            "dimensions": list(ANSWER_DIMENSIONS),
            "reference": "deterministic_source_evidence_projection",
        }
    )
    gold_family_id = "machine-gold-" + split_epoch_id[:24]
    if pointer_path.exists():
        existing_artifact = validate_gold_manifest(
            pointer_path,
            required_episode_ids=[str(entry["case_id"]) for entry in selected],
            consensus_ledger_file=consensus_ledger_file,
            chronovisor_root=chronovisor_root,
            expected_split=split,
            split_epoch_id=split_epoch_id,
            benchmark_manifest=benchmark_manifest,
        )
        if existing_artifact.get("passed") is True:
            return {
                "status": "complete",
                "reason": "verified_machine_consensus",
                "artifact": str(existing_artifact.get("manifest_path") or artifact_path),
                "pointer": str(pointer_path),
                "entries": len(selected),
                "manifest_sha256": str(existing_artifact.get("manifest_sha256") or ""),
            }
    existing: dict[str, dict[str, Any]] = {}
    selected_ids = {str(entry["case_id"]) for entry in selected}
    for row in _read_jsonl(candidates_path):
        entry = row.get("entry")
        if not isinstance(entry, Mapping) or row.get("entry_sha256") != _canonical_sha(
            dict(entry)
        ):
            continue
        candidate = dict(entry)
        episode_id = str(candidate.get("episode_id") or "")
        if episode_id not in selected_ids:
            continue
        subject = _gold_machine_subject(
            candidate,
            rubric_sha256=rubric_sha,
            gold_family_id=gold_family_id,
            expected_split=split,
            split_epoch_id=split_epoch_id,
        )
        prompt = build_recall_answer_adjudication_prompt(
            {"subject": subject, "subject_sha256": _sealed_canonical_sha(subject)}
        )
        provenance = candidate.get("review_provenance")
        receipt_sha = (
            provenance.get("consensus_receipt_sha256")
            if isinstance(provenance, Mapping)
            else None
        )
        candidate_check = validate_machine_consensus_receipt(
            receipt_sha,
            expected_kind="gold_entry_review",
            expected_subject=subject,
            expected_producer_policy_sha256=(
                DETERMINISTIC_GOLD_PROJECTION_POLICY_SHA256
            ),
            prompt=prompt,
            schema=RECALL_ANSWER_ADJUDICATION_SCHEMA,
            system=None,
            lane=ANSWER_ADJUDICATION_LANE,
            ledger_file=consensus_ledger_file,
            chronovisor_root=chronovisor_root,
        )
        if candidate_check.get("passed") is True:
            existing[episode_id] = candidate
    accepted = 0
    waiting_reason = ""
    for split_entry in selected:
        episode_id = str(split_entry.get("case_id") or "")
        if episode_id in existing or accepted >= max(0, max_items):
            continue
        source_packet = dict(split_entry)
        source_error = _independent_gold_source_packet_error(source_packet)
        if source_error:
            return {"status": "held", "reason": source_error}
        reference_answer = _packet_reference_evidence(source_packet)
        evidence: dict[str, Any] = {
            "source_packet": source_packet,
            "source_packet_sha256": _sealed_canonical_sha(source_packet),
            "source_frozen_at": source_frozen_at,
            "reference_policy_sha256": DETERMINISTIC_GOLD_PROJECTION_POLICY_SHA256,
        }
        evidence_sha = _canonical_sha(
            {
                "episode_id": episode_id,
                "gold_answer": reference_answer,
                "evidence": evidence,
                "rubric_sha256": rubric_sha,
            }
        )
        entry = {
            "episode_id": episode_id,
            "gold_answer": reference_answer,
            "evidence": evidence,
            "evidence_sha256": evidence_sha,
        }
        subject = _gold_machine_subject(
            entry,
            rubric_sha256=rubric_sha,
            gold_family_id=gold_family_id,
            expected_split=split,
            split_epoch_id=split_epoch_id,
        )
        prompt = build_recall_answer_adjudication_prompt(
            {"subject": subject, "subject_sha256": _sealed_canonical_sha(subject)}
        )
        if dry_run:
            return {
                "status": "waiting",
                "reason": "dry_run",
                "pending": len(selected) - len(existing),
            }
        adjudication = append_machine_consensus_receipt(
            kind="gold_entry_review",
            subject=subject,
            producer_policy_sha256=DETERMINISTIC_GOLD_PROJECTION_POLICY_SHA256,
            prompt=prompt,
            schema=RECALL_ANSWER_ADJUDICATION_SCHEMA,
            system=None,
            lane=ANSWER_ADJUDICATION_LANE,
            ledger_file=consensus_ledger_file,
            chronovisor_root=chronovisor_root,
            router_factory=router_factory,
        )
        if adjudication.get("status") != "accepted":
            waiting_reason = str(adjudication.get("reason") or "machine_gold_held")
            return {
                "status": str(adjudication.get("status") or "held"),
                "reason": waiting_reason,
                "accepted": accepted,
            }
        receipt = adjudication["receipt"]
        entry["review_provenance"] = {
            "source_kind": "adjudicated_benchmark",
            "authority_kind": "adopted_local_consensus",
            "consensus_receipt_sha256": receipt["receipt_sha256"],
            "subject_sha256": _sealed_canonical_sha(subject),
            "reviewed_at": receipt["created_at"],
        }
        append_jsonl_durable(
            candidates_path,
            [{"entry": entry, "entry_sha256": _canonical_sha(entry)}],
            sort_keys=True,
        )
        existing[episode_id] = entry
        accepted += 1
    if len(existing) < len(selected):
        return {
            "status": "waiting",
            "reason": waiting_reason or "machine_gold_adjudication_pending",
            "accepted": accepted,
            "completed": len(existing),
            "required": len(selected),
        }
    payload = {
        "schema_version": 1,
        "artifact_kind": "immutable-answer-gold-manifest",
        "frozen_at": _now_utc(),
        "gold_id": f"{gold_family_id}-{split}",
        "gold_family_id": gold_family_id,
        "version": "machine-consensus-v1",
        "review_protocol_sha256": DETERMINISTIC_GOLD_PROJECTION_POLICY_SHA256,
        "rubric_sha256": rubric_sha,
        "split": split,
        "split_epoch_id": split_epoch_id,
        "benchmark_manifest_path": str(benchmark_check.get("manifest_path") or ""),
        "benchmark_manifest_sha256": str(benchmark_check["manifest_sha256"]),
        "entries": [existing[str(entry["case_id"])] for entry in selected],
    }
    sealed = seal_object(payload)
    check = validate_gold_manifest(
        sealed,
        required_episode_ids=[str(entry["case_id"]) for entry in selected],
        consensus_ledger_file=consensus_ledger_file,
        chronovisor_root=chronovisor_root,
        expected_split=split,
        split_epoch_id=split_epoch_id,
        benchmark_manifest=benchmark_manifest,
    )
    if check.get("passed") is not True:
        return {
            "status": "held",
            "reason": str(check.get("reason") or "machine_gold_validation_failed"),
        }
    try:
        sealed = _create_once_sealed(artifact_path, payload)
        pointer = {
            "schema_version": 1,
            "artifact_kind": "immutable-answer-gold-active-pointer",
            "split": split,
            "split_epoch_id": split_epoch_id,
            "manifest_path": str(artifact_path),
            "manifest_sha256": str(sealed["seal_sha256"]),
            "updated_at": _now_utc(),
        }
        with sidecar_exclusive_lock(pointer_path):
            write_sealed_json(pointer_path, pointer, backup=True)
        pointer_check = validate_gold_manifest(
            pointer_path,
            required_episode_ids=[str(entry["case_id"]) for entry in selected],
            consensus_ledger_file=consensus_ledger_file,
            chronovisor_root=chronovisor_root,
            expected_split=split,
            split_epoch_id=split_epoch_id,
            benchmark_manifest=benchmark_manifest,
        )
        if pointer_check.get("passed") is not True:
            raise DurableStateError("machine gold pointer read-back failed")
    except (DurableStateError, OSError, ValueError):
        return {"status": "held", "reason": "machine_gold_immutable_conflict"}
    return {
        "status": "complete",
        "reason": "verified_machine_consensus",
        "artifact": str(artifact_path),
        "pointer": str(pointer_path),
        "entries": len(selected),
        "manifest_sha256": sealed["seal_sha256"],
    }


def validate_gold_manifest(
    value: Path | Mapping[str, Any],
    *,
    required_episode_ids: Sequence[str],
    review_ledger_file: Path = ANSWER_REVIEW_LEDGER,
    consensus_ledger_file: Path = ANSWER_CONSENSUS_LEDGER,
    chronovisor_root: Path = CHRONOVISOR_ROOT,
    expected_split: str = "",
    split_epoch_id: str = "",
    benchmark_manifest: Path | Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    try:
        payload = (
            read_sealed_json(value)
            if isinstance(value, Path)
            else verify_sealed_object(dict(value))
        )
    except (DurableStateError, TypeError, ValueError):
        return {"passed": False, "reason": "gold_manifest_seal_invalid"}
    if payload.get("artifact_kind") == "immutable-answer-gold-active-pointer":
        if not isinstance(value, Path):
            return {"passed": False, "reason": "machine_gold_pointer_invalid"}
        pointer_split = str(payload.get("split") or "")
        pointer_epoch = str(payload.get("split_epoch_id") or "")
        manifest_path = Path(str(payload.get("manifest_path") or ""))
        expected_path = (
            chronovisor_root
            / "runtime"
            / "recall-answer-eval"
            / "gold"
            / pointer_split
            / f"{pointer_epoch}.json"
        )
        if (
            set(payload)
            != {
                "schema_version",
                "artifact_kind",
                "split",
                "split_epoch_id",
                "manifest_path",
                "manifest_sha256",
                "updated_at",
                "seal_sha256",
            }
            or payload.get("schema_version") != 1
            or pointer_split not in {"train", "holdout", "locked-test"}
            or not _valid_sha(pointer_epoch)
            or not _valid_sha(payload.get("manifest_sha256"))
            or manifest_path.expanduser().resolve(strict=False)
            != expected_path.expanduser().resolve(strict=False)
            or not _strict_utc(payload.get("updated_at"))
            or (expected_split and pointer_split != expected_split)
            or (split_epoch_id and pointer_epoch != split_epoch_id)
        ):
            return {"passed": False, "reason": "machine_gold_pointer_invalid"}
        nested = validate_gold_manifest(
            manifest_path,
            required_episode_ids=required_episode_ids,
            review_ledger_file=review_ledger_file,
            consensus_ledger_file=consensus_ledger_file,
            chronovisor_root=chronovisor_root,
            expected_split=expected_split or pointer_split,
            split_epoch_id=split_epoch_id or pointer_epoch,
            benchmark_manifest=benchmark_manifest,
        )
        if (
            nested.get("passed") is not True
            or nested.get("manifest_sha256") != payload.get("manifest_sha256")
        ):
            return {"passed": False, "reason": "machine_gold_pointer_drift"}
        return nested
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
    if payload.get("version") == "machine-consensus-v1" and set(payload) != {
        "schema_version",
        "artifact_kind",
        "frozen_at",
        "gold_id",
        "gold_family_id",
        "version",
        "review_protocol_sha256",
        "rubric_sha256",
        "split",
        "split_epoch_id",
        "benchmark_manifest_path",
        "benchmark_manifest_sha256",
        "entries",
        "seal_sha256",
    }:
        return {"passed": False, "reason": "machine_gold_shape_invalid"}
    if (
        payload.get("version") == "machine-consensus-v1"
        and payload.get("review_protocol_sha256")
        != DETERMINISTIC_GOLD_PROJECTION_POLICY_SHA256
    ):
        return {"passed": False, "reason": "machine_gold_protocol_invalid"}
    benchmark_entries: Mapping[str, Mapping[str, Any]] = {}
    if payload.get("version") == "machine-consensus-v1":
        pinned_benchmark_path = Path(
            str(payload.get("benchmark_manifest_path") or "")
        )
        expected_benchmark_path = (
            chronovisor_root
            / "runtime"
            / "recall-answer-eval"
            / "benchmarks"
            / f"{split_epoch_id}.json"
        )
        if (
            pinned_benchmark_path.expanduser().resolve(strict=False)
            != expected_benchmark_path.expanduser().resolve(strict=False)
        ):
            return {"passed": False, "reason": "machine_gold_benchmark_invalid"}
        benchmark_check = validate_independent_answer_benchmark(
            pinned_benchmark_path, chronovisor_root=chronovisor_root
        )
        if (
            benchmark_check.get("passed") is not True
            or benchmark_check.get("split_epoch_id") != split_epoch_id
            or benchmark_check.get("manifest_sha256")
            != payload.get("benchmark_manifest_sha256")
            or payload.get("split") != expected_split
            or payload.get("split_epoch_id") != split_epoch_id
        ):
            return {"passed": False, "reason": "machine_gold_benchmark_invalid"}
        benchmark_entries = benchmark_check.get("entries", {})
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
        if payload.get("version") == "machine-consensus-v1" and (
            set(entry)
            != {
                "episode_id",
                "gold_answer",
                "evidence",
                "evidence_sha256",
                "review_provenance",
            }
            or not isinstance(evidence, Mapping)
            or set(evidence)
            != {
                "source_packet",
                "source_packet_sha256",
                "source_frozen_at",
                "reference_policy_sha256",
            }
            or set(review)
            != {
                "source_kind",
                "authority_kind",
                "consensus_receipt_sha256",
                "subject_sha256",
                "reviewed_at",
            }
            or review.get("source_kind") != "adjudicated_benchmark"
            or review.get("authority_kind") != "adopted_local_consensus"
        ):
            return {"passed": False, "reason": "machine_gold_entry_shape_invalid"}
        expected_evidence_sha = _canonical_sha(
            {
                "episode_id": episode_id,
                "gold_answer": gold_answer,
                "evidence": evidence,
                "rubric_sha256": rubric_sha,
            }
        )
        receipt_sha = str(review.get("reviewer_receipt_sha256") or "")
        source_kind = str(review.get("source_kind") or "")
        machine_subject: dict[str, Any] = {}
        if source_kind == "adjudicated_benchmark":
            source_packet = (
                evidence.get("source_packet") if isinstance(evidence, Mapping) else None
            )
            source_packet_sha = (
                evidence.get("source_packet_sha256")
                if isinstance(evidence, Mapping)
                else None
            )
            machine_source_error = _independent_gold_source_packet_error(
                source_packet
            )
            expected_packet = benchmark_entries.get(episode_id)
            if (
                not machine_source_error
                and (
                    not isinstance(expected_packet, Mapping)
                    or not isinstance(source_packet, Mapping)
                    or dict(source_packet) != dict(expected_packet)
                )
            ):
                machine_source_error = "machine_gold_benchmark_packet_mismatch"
            if (
                not machine_source_error
                and isinstance(source_packet, Mapping)
                and source_packet_sha != _sealed_canonical_sha(source_packet)
            ):
                machine_source_error = "machine_gold_source_packet_digest_invalid"
            if (
                not machine_source_error
                and isinstance(source_packet, Mapping)
                and (
                    gold_answer != _packet_reference_evidence(source_packet)
                    or evidence.get("source_frozen_at")
                    != source_packet.get("source_frozen_at")
                    or source_packet.get("split") != expected_split
                    or source_packet.get("split_epoch_id") != split_epoch_id
                )
            ):
                machine_source_error = "machine_gold_reference_evidence_mismatch"
            machine_subject = _gold_machine_subject(
                entry,
                rubric_sha256=str(rubric_sha or ""),
                gold_family_id=str(payload.get("gold_family_id") or ""),
                expected_split=expected_split,
                split_epoch_id=split_epoch_id,
            )
            machine_prompt = build_recall_answer_adjudication_prompt(
                {
                    "subject": machine_subject,
                    "subject_sha256": _sealed_canonical_sha(machine_subject),
                }
            )
            loaded_machine_receipt = load_machine_consensus_receipt(
                review.get("consensus_receipt_sha256"),
                ledger_file=consensus_ledger_file,
            )
            historical_authority = (
                loaded_machine_receipt.get("receipt", {}).get("authority")
                if loaded_machine_receipt.get("passed") is True
                and isinstance(loaded_machine_receipt.get("receipt"), Mapping)
                else None
            )
            machine_check = validate_machine_consensus_receipt(
                review.get("consensus_receipt_sha256"),
                expected_kind="gold_entry_review",
                expected_subject=machine_subject,
                expected_producer_policy_sha256=(
                    DETERMINISTIC_GOLD_PROJECTION_POLICY_SHA256
                ),
                prompt=machine_prompt,
                schema=RECALL_ANSWER_ADJUDICATION_SCHEMA,
                system=None,
                lane=ANSWER_ADJUDICATION_LANE,
                ledger_file=consensus_ledger_file,
                chronovisor_root=chronovisor_root,
                current_authority=(
                    historical_authority
                    if isinstance(historical_authority, Mapping)
                    else None
                ),
            )
            machine_receipt = machine_check.get("receipt", {})
            source_frozen_at = _strict_utc(machine_subject.get("source_frozen_at"))
            receipt_created_at = _strict_utc(
                machine_receipt.get("created_at")
                if isinstance(machine_receipt, Mapping)
                else ""
            )
            causal_order = bool(
                source_frozen_at
                and receipt_created_at
                and frozen_at
                and datetime.fromisoformat(source_frozen_at.replace("Z", "+00:00"))
                < datetime.fromisoformat(receipt_created_at.replace("Z", "+00:00"))
                <= datetime.fromisoformat(frozen_at.replace("Z", "+00:00"))
            )
            if machine_source_error:
                review_error = machine_source_error
            elif machine_check.get("passed") is not True:
                review_error = str(
                    machine_check.get("reason") or "machine_consensus_invalid"
                )
            elif not causal_order:
                review_error = "machine_gold_causal_order_invalid"
            elif (
                review.get("subject_sha256")
                != _sealed_canonical_sha(machine_subject)
                or review.get("reviewed_at") != receipt_created_at
            ):
                review_error = "machine_gold_subject_binding_invalid"
            else:
                review_error = ""
            receipt_sha = str(review.get("consensus_receipt_sha256") or "")
        else:
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
            or source_kind not in _ALLOWED_GOLD_SOURCE_KINDS
            or not _valid_sha(receipt_sha)
            or not _strict_utc(review.get("reviewed_at"))
            or receipt_sha in seen_review_receipts
            or review_error
            or (
                source_kind == "adjudicated_benchmark"
                and (
                    not expected_split
                    or not _valid_sha(split_epoch_id)
                    or not _valid_sha(machine_subject.get("source_packet_sha256"))
                    or _contains_forbidden_gold_input(evidence)
                )
            )
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
    consensus_ledger_file: Path = ANSWER_CONSENSUS_LEDGER,
    execution_ledger_file: Path = ANSWER_EXECUTION_LEDGER,
    chronovisor_root: Path = CHRONOVISOR_ROOT,
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
    independent_benchmark_manifest: Path | Mapping[str, Any] | None = None,
    evaluated_at: str = "",
    review_ledger_file: Path = ANSWER_REVIEW_LEDGER,
    consensus_ledger_file: Path = ANSWER_CONSENSUS_LEDGER,
    execution_ledger_file: Path = ANSWER_EXECUTION_LEDGER,
    chronovisor_root: Path = CHRONOVISOR_ROOT,
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
    if payload.get("schema_version") == MACHINE_SCORER_CALIBRATION_SCHEMA_VERSION:
        if independent_benchmark_manifest is None:
            return {"passed": False, "reason": "machine_calibration_benchmark_missing"}
        return validate_machine_scorer_calibration_artifact(
            payload,
            scorer_identity=scorer_identity,
            benchmark_manifest=independent_benchmark_manifest,
            consensus_ledger_file=consensus_ledger_file,
            execution_ledger_file=execution_ledger_file,
            chronovisor_root=chronovisor_root,
            evaluated_at=evaluated_at,
        )
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


def _machine_calibration_policy() -> dict[str, Any]:
    return {
        "version": MACHINE_SCORER_CALIBRATION_SCHEMA_VERSION,
        "minimum_cases": MACHINE_SCORER_CALIBRATION_MIN_CASES,
        "minimum_pairs": MACHINE_SCORER_CALIBRATION_MIN_PAIRS,
        "minimum_clusters": MACHINE_SCORER_CALIBRATION_MIN_CLUSTERS,
        "control_variants": [
            "positive",
            "wrong_fact",
            "missing_citation",
            "wrong_page",
        ],
        "source_split": "holdout",
        "scorer_role": "tie_break",
        "consensus_roles": ["primary", "challenger"],
        "quality_policy": _scorer_calibration_policy(),
    }


def _machine_calibration_policy_sha256() -> str:
    policy_sha = _canonical_sha(_machine_calibration_policy())
    if policy_sha != SCORER_CALIBRATION_PRODUCER_POLICY_SHA256:
        raise ValueError("machine calibration producer policy allowlist drift")
    return policy_sha


def _machine_calibration_answer(
    packet: Mapping[str, Any], variant: str
) -> tuple[str, dict[str, float]]:
    reference = _packet_reference_evidence(packet)
    chunks = packet.get("evidence_chunks")
    first_page = (
        str(chunks[0].get("page_id") or "")
        if isinstance(chunks, list) and chunks and isinstance(chunks[0], Mapping)
        else "source"
    )
    if variant == "positive":
        return reference, {dimension: 1.0 for dimension in ANSWER_DIMENSIONS}
    if variant == "wrong_fact":
        return (
            f"The frozen evidence states the opposite of its actual fact. [PAGE {first_page}]",
            {dimension: 0.0 for dimension in ANSWER_DIMENSIONS},
        )
    if variant == "missing_citation":
        text = "\n\n".join(
            str(chunk.get("excerpt") or "")
            for chunk in chunks or []
            if isinstance(chunk, Mapping)
        )
        return text, {"correctness": 1.0, "grounding": 1.0, "citation": 0.0}
    if variant == "wrong_page":
        text = reference.replace(f"[PAGE {first_page}]", "[PAGE wrong-page]")
        return text, {"correctness": 1.0, "grounding": 0.0, "citation": 0.0}
    raise ValueError("unknown machine calibration variant")


def _machine_calibration_controls(
    benchmark_check: Mapping[str, Any],
) -> list[dict[str, Any]]:
    packets = [
        dict(packet)
        for packet in benchmark_check.get("entries", {}).values()
        if packet.get("split") == "holdout"
    ]
    packets.sort(
        key=lambda packet: (
            str(packet.get("component_sha256") or ""),
            str(packet.get("case_id") or ""),
        )
    )
    by_component: dict[str, dict[str, Any]] = {}
    for packet in packets:
        by_component.setdefault(str(packet.get("component_sha256") or ""), packet)
    selected = [by_component[key] for key in sorted(by_component)[:20] if key]
    if len(selected) < MACHINE_SCORER_CALIBRATION_MIN_CLUSTERS:
        return []
    policy_sha = _machine_calibration_policy_sha256()
    negatives = ("wrong_fact", "missing_citation", "wrong_page")
    controls: list[dict[str, Any]] = []
    for index, packet in enumerate(selected):
        packet_sha = _sealed_canonical_sha(packet)
        pair_id = _canonical_sha(
            {
                "policy_sha256": policy_sha,
                "component_sha256": packet["component_sha256"],
                "source_entry_sha256": packet["source_entry_sha256"],
            }
        )
        for arm, variant in (("a", "positive"), ("b", negatives[index % 3])):
            answer, expected_scores = _machine_calibration_answer(packet, variant)
            control_id = _canonical_sha(
                {
                    "policy_sha256": policy_sha,
                    "pair_id": pair_id,
                    "pair_arm": arm,
                    "variant": variant,
                    "source_packet_sha256": packet_sha,
                }
            )
            controls.append(
                {
                    "control_id": control_id,
                    "pair_id": pair_id,
                    "pair_arm": arm,
                    "variant": variant,
                    "answer": answer,
                    "answer_sha256": _sha_text(answer),
                    "expected_scores": expected_scores,
                    "expected_scores_sha256": _canonical_sha(expected_scores),
                    "query_sha256": packet["prompt_content_sha256"],
                    "evidence_sha256": packet["reference_evidence_sha256"],
                    "component_sha256": packet["component_sha256"],
                    "source_packet": packet,
                    "source_packet_sha256": packet_sha,
                }
            )
    return controls


def _machine_calibration_subject(
    control: Mapping[str, Any], *, benchmark_manifest_sha256: str
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "subject_kind": "scorer_calibration_case",
        "control": copy.deepcopy(dict(control)),
        "control_sha256": _canonical_sha(dict(control)),
        "control_id": control.get("control_id"),
        "pair_id": control.get("pair_id"),
        "pair_arm": control.get("pair_arm"),
        "answer_variant": control.get("variant"),
        "answer_sha256": control.get("answer_sha256"),
        "expected_scores_sha256": control.get("expected_scores_sha256"),
        "source_packet_sha256": control.get("source_packet_sha256"),
        "benchmark_manifest_sha256": benchmark_manifest_sha256,
        "component_sha256": control.get("component_sha256"),
        "evidence_complete": True,
        "reference_independent": True,
        "preregistered_before_evaluation": True,
        "split_safe": True,
        "producer_kind": "deterministic_evidence_projection",
        "producer_model": None,
        "producer_policy_sha256": _machine_calibration_policy_sha256(),
        "production_answer_used": False,
    }


def _machine_calibration_case_metrics(
    cases: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, Any], dict[str, bool]]:
    metric_cases = [
        {
            "session_hash": str(case["control"]["component_sha256"])[:16],
            "query_sha256": case["control"]["query_sha256"],
            "evidence_sha256": case["control"]["evidence_sha256"],
            "cluster_nodes": [
                f"component:{case['control']['component_sha256']}"
            ],
            "pair_id": case["control"]["pair_id"],
            "pair_arm": case["control"]["pair_arm"],
            "human_reviewed": case["control"]["expected_scores"],
            "scorer_scores": case["scorer_scores"],
        }
        for case in cases
    ]
    metrics = _calibration_metrics(metric_cases)
    quality_gates = _calibration_gates(metrics)
    gates = {
        **quality_gates,
        "machine_case_count": metrics.get("cases")
        == MACHINE_SCORER_CALIBRATION_MIN_CASES,
        "machine_pair_count": metrics.get("pairs")
        == MACHINE_SCORER_CALIBRATION_MIN_PAIRS,
        "machine_cluster_count": metrics.get("clusters")
        == MACHINE_SCORER_CALIBRATION_MIN_CLUSTERS,
    }
    return metrics, gates


_MACHINE_CALIBRATION_CASE_KEYS = {
    "control_id",
    "control",
    "consensus_receipt_sha256",
    "scoring",
    "scorer_scores",
    "execution_receipt_sha256",
}
_MACHINE_CALIBRATION_CONTROL_KEYS = {
    "control_id",
    "pair_id",
    "pair_arm",
    "variant",
    "answer",
    "answer_sha256",
    "expected_scores",
    "expected_scores_sha256",
    "query_sha256",
    "evidence_sha256",
    "component_sha256",
    "source_packet",
    "source_packet_sha256",
}


def _machine_calibration_case_error(
    case: Mapping[str, Any],
    *,
    control: Mapping[str, Any],
    benchmark: Mapping[str, Any],
    scorer_identity: Mapping[str, Any],
    consensus_ledger_file: Path,
    execution_ledger_file: Path,
    chronovisor_root: Path,
    completed_before: str = "",
) -> str:
    if set(case) != _MACHINE_CALIBRATION_CASE_KEYS:
        return "machine_calibration_case_shape_invalid"
    if set(control) != _MACHINE_CALIBRATION_CONTROL_KEYS:
        return "machine_calibration_control_shape_invalid"
    if case.get("control_id") != control.get("control_id") or case.get(
        "control"
    ) != dict(control):
        return "machine_calibration_control_drift"
    scorer_identity_sha = _canonical_sha(dict(scorer_identity))
    benchmark_sha = str(benchmark.get("manifest_sha256") or "")
    policy_sha = _machine_calibration_policy_sha256()
    calibration_run_id = _canonical_sha(
        {
            "kind": "machine-scorer-calibration-v2",
            "benchmark_manifest_sha256": benchmark_sha,
            "scorer_identity_sha256": scorer_identity_sha,
            "policy_sha256": policy_sha,
        }
    )
    pair_protocol = _preregistered_pair_protocol(
        seed=ANSWER_AUTHORITY_SEED,
        episode_id=str(control["control_id"]),
        episode_sha256=str(control["source_packet_sha256"]),
        split_manifest_sha256=benchmark_sha,
        gold_manifest_sha256=benchmark_sha,
        adapter_registry_sha256=scorer_identity_sha,
        evaluation_kind="machine-scorer-calibration-v2",
    )
    expected_scoring = {
        **dict(pair_protocol["scoring"]),
        "evidence_manifest_sha256": benchmark_sha,
        "rubric_sha256": str(scorer_identity.get("rubric_sha256") or ""),
    }
    scoring = case.get("scoring")
    scores = case.get("scorer_scores")
    if scoring != expected_scoring:
        return "machine_calibration_scoring_drift"
    if (
        not isinstance(scores, Mapping)
        or set(scores) != set(ANSWER_DIMENSIONS)
        or any(not _finite_unit_score(scores.get(name)) for name in ANSWER_DIMENSIONS)
    ):
        return "machine_calibration_scores_invalid"
    subject = _machine_calibration_subject(
        control, benchmark_manifest_sha256=benchmark_sha
    )
    prompt = build_recall_answer_adjudication_prompt(
        {"subject": subject, "subject_sha256": _sealed_canonical_sha(subject)}
    )
    loaded = load_machine_consensus_receipt(
        case.get("consensus_receipt_sha256"), ledger_file=consensus_ledger_file
    )
    receipt = loaded.get("receipt") if loaded.get("passed") is True else None
    historical_authority = (
        receipt.get("authority") if isinstance(receipt, Mapping) else None
    )
    receipt_check = validate_machine_consensus_receipt(
        case.get("consensus_receipt_sha256"),
        expected_kind="scorer_calibration_case_review",
        expected_subject=subject,
        expected_producer_policy_sha256=policy_sha,
        prompt=prompt,
        schema=RECALL_ANSWER_ADJUDICATION_SCHEMA,
        system=None,
        lane=ANSWER_ADJUDICATION_LANE,
        ledger_file=consensus_ledger_file,
        chronovisor_root=chronovisor_root,
        current_authority=(
            historical_authority
            if isinstance(historical_authority, Mapping)
            else None
        ),
    )
    artifact = receipt_check.get("artifact", {})
    votes = (
        artifact.get("provenance", {}).get("vote_manifest", [])
        if isinstance(artifact, Mapping)
        else []
    )
    models = (
        receipt.get("authority", {}).get("router", {}).get("models", [])
        if isinstance(receipt, Mapping)
        else []
    )
    receipt_created_at = _strict_utc(
        receipt.get("created_at") if isinstance(receipt, Mapping) else ""
    )
    benchmark_frozen_at = _strict_utc(
        benchmark.get("payload", {}).get("frozen_at")
        if isinstance(benchmark.get("payload"), Mapping)
        else ""
    )
    normalized_before = _strict_utc(completed_before) if completed_before else ""
    causal_error = bool(
        not receipt_created_at
        or not benchmark_frozen_at
        or datetime.fromisoformat(receipt_created_at.replace("Z", "+00:00"))
        <= datetime.fromisoformat(benchmark_frozen_at.replace("Z", "+00:00"))
        or (
            normalized_before
            and datetime.fromisoformat(receipt_created_at.replace("Z", "+00:00"))
            > datetime.fromisoformat(normalized_before.replace("Z", "+00:00"))
        )
    )
    if (
        receipt_check.get("passed") is not True
        or not isinstance(votes, list)
        or len(votes) != 2
        or [vote.get("role") for vote in votes if isinstance(vote, Mapping)]
        != ["primary", "challenger"]
        or not isinstance(models, list)
        or len(models) != 3
        or causal_error
    ):
        return "machine_calibration_consensus_invalid"
    expected_reset = {
        "seed": expected_scoring["seed"],
        "base_state_sha256": expected_scoring["base_state_sha256"],
        "reset_protocol_sha256": scorer_identity.get("policy_sha256"),
    }
    return _execution_receipt_error(
        receipt_sha256=case.get("execution_receipt_sha256"),
        expected_kind="answer_scorer_call",
        expected_adapter_identity_sha256=scorer_identity_sha,
        expected_parent_run_id=calibration_run_id,
        expected_input_payload={
            "prompt_sha256": control["query_sha256"],
            "answer_sha256": control["answer_sha256"],
            "gold_evidence_sha256": control["evidence_sha256"],
            "scoring": expected_scoring,
        },
        expected_output_payload={
            "dimensions": dict(scores),
            "reset_receipt": expected_reset,
        },
        ledger_file=execution_ledger_file,
        completed_before=completed_before,
    )


def _machine_calibration_paths(
    *,
    chronovisor_root: Path,
    benchmark_epoch_sha256: str,
    scorer_identity_sha256: str,
) -> tuple[Path, Path]:
    runtime_root = chronovisor_root / "runtime" / "recall-answer-eval"
    return (
        runtime_root
        / "scorer-calibration-candidates"
        / benchmark_epoch_sha256
        / f"{scorer_identity_sha256}.jsonl",
        runtime_root
        / "scorer-calibration"
        / benchmark_epoch_sha256
        / f"{scorer_identity_sha256}.json",
    )


def _publish_machine_calibration_pointer(
    *,
    pointer_path: Path,
    artifact_path: Path,
    artifact: Mapping[str, Any],
    benchmark_epoch_sha256: str,
    scorer_identity_sha256: str,
    scorer_identity: Mapping[str, Any],
    consensus_ledger_file: Path,
    execution_ledger_file: Path,
    chronovisor_root: Path,
) -> dict[str, Any]:
    pointer = {
        "schema_version": 1,
        "artifact_kind": "machine-answer-scorer-calibration-active-pointer",
        "benchmark_epoch_sha256": benchmark_epoch_sha256,
        "scorer_identity_sha256": scorer_identity_sha256,
        "manifest_path": str(artifact_path),
        "manifest_sha256": str(artifact["seal_sha256"]),
        "updated_at": _now_utc(),
    }
    with sidecar_exclusive_lock(pointer_path):
        write_sealed_json(pointer_path, pointer, backup=True)
    return validate_machine_scorer_calibration_artifact(
        pointer_path,
        scorer_identity=scorer_identity,
        benchmark_manifest=None,
        consensus_ledger_file=consensus_ledger_file,
        execution_ledger_file=execution_ledger_file,
        chronovisor_root=chronovisor_root,
    )


def build_machine_scorer_calibration_cycle(
    *,
    scorer: AnswerScorer,
    scorer_identity: Mapping[str, Any],
    benchmark_manifest: Path | Mapping[str, Any] = INDEPENDENT_ANSWER_BENCHMARK,
    output_file: Path | None = None,
    candidate_file: Path | None = None,
    consensus_ledger_file: Path = ANSWER_CONSENSUS_LEDGER,
    execution_ledger_file: Path = ANSWER_EXECUTION_LEDGER,
    chronovisor_root: Path = CHRONOVISOR_ROOT,
    router_factory: Callable[[str], Any] | None = None,
    max_items: int = 1,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Incrementally score one immutable 40-control calibration epoch."""

    from chronovisor.recall.recall_answer_adapters import (
        builtin_ollama_answer_scorer,
    )

    if not isinstance(benchmark_manifest, Path):
        return {"status": "held", "reason": "machine_calibration_pinned_benchmark_required"}
    benchmark = validate_independent_answer_benchmark(
        benchmark_manifest, chronovisor_root=chronovisor_root
    )
    if (
        benchmark.get("passed") is not True
        or benchmark.get("payload", {}).get("source_kind")
        != "machine_search_label_consensus"
    ):
        return {"status": "waiting", "reason": "machine_benchmark_authority_required"}
    controls = _machine_calibration_controls(benchmark)
    if len(controls) != MACHINE_SCORER_CALIBRATION_MIN_CASES:
        return {"status": "waiting", "reason": "calibration_holdout_clusters_pending"}
    if scorer is not builtin_ollama_answer_scorer:
        return {"status": "held", "reason": "builtin_scorer_required"}
    identity_error = _identity_error(scorer_identity, _REQUIRED_SCORER_IDENTITY)
    if identity_error:
        return {"status": "held", "reason": identity_error}
    scorer_identity_sha = _canonical_sha(dict(scorer_identity))
    benchmark_epoch_sha = str(benchmark["split_epoch_id"])
    canonical_candidate, artifact_path = _machine_calibration_paths(
        chronovisor_root=chronovisor_root,
        benchmark_epoch_sha256=benchmark_epoch_sha,
        scorer_identity_sha256=scorer_identity_sha,
    )
    if candidate_file is not None and candidate_file.resolve(strict=False) != (
        canonical_candidate.resolve(strict=False)
    ):
        return {"status": "held", "reason": "machine_calibration_candidate_path_invalid"}
    candidates_path = canonical_candidate
    pointer_path = output_file or (
        chronovisor_root
        / "runtime"
        / "recall-answer-eval"
        / "scorer-calibration-active.json"
    )
    if pointer_path.exists():
        pointer_check = validate_machine_scorer_calibration_artifact(
            pointer_path,
            scorer_identity=scorer_identity,
            benchmark_manifest=None,
            consensus_ledger_file=consensus_ledger_file,
            execution_ledger_file=execution_ledger_file,
            chronovisor_root=chronovisor_root,
        )
        if pointer_check.get("passed") is not True:
            return {"status": "held", "reason": "machine_calibration_pointer_invalid"}
        if pointer_check.get("benchmark_epoch_sha256") == benchmark_epoch_sha:
            return {
                "status": "complete",
                "reason": "verified",
                "artifact": str(pointer_check.get("manifest_path") or artifact_path),
                "pointer": str(pointer_path),
                "manifest_sha256": str(pointer_check.get("manifest_sha256") or ""),
            }
    if artifact_path.exists():
        artifact_check = validate_machine_scorer_calibration_artifact(
            artifact_path,
            scorer_identity=scorer_identity,
            benchmark_manifest=None,
            consensus_ledger_file=consensus_ledger_file,
            execution_ledger_file=execution_ledger_file,
            chronovisor_root=chronovisor_root,
        )
        if artifact_check.get("passed") is not True:
            return {"status": "held", "reason": "machine_calibration_immutable_conflict"}
        pointer_check = _publish_machine_calibration_pointer(
            pointer_path=pointer_path,
            artifact_path=artifact_path,
            artifact=artifact_check["payload"],
            benchmark_epoch_sha256=benchmark_epoch_sha,
            scorer_identity_sha256=scorer_identity_sha,
            scorer_identity=scorer_identity,
            consensus_ledger_file=consensus_ledger_file,
            execution_ledger_file=execution_ledger_file,
            chronovisor_root=chronovisor_root,
        )
        if pointer_check.get("passed") is not True:
            return {"status": "held", "reason": "machine_calibration_pointer_write_failed"}
        return {
            "status": "complete",
            "reason": "verified",
            "artifact": str(artifact_path),
            "pointer": str(pointer_path),
            "manifest_sha256": str(artifact_check["manifest_sha256"]),
        }
    control_by_id = {str(control["control_id"]): control for control in controls}
    cases: dict[str, dict[str, Any]] = {}
    try:
        candidate_rows = _read_jsonl_strict(candidates_path)
    except DurableStateError:
        return {"status": "held", "reason": "machine_calibration_candidates_invalid"}
    for row in candidate_rows:
        case = row.get("case")
        if (
            set(row) != {"case", "case_sha256"}
            or not isinstance(case, Mapping)
            or row.get("case_sha256") != _canonical_sha(dict(case))
        ):
            return {"status": "held", "reason": "machine_calibration_candidates_invalid"}
        control_id = str(case.get("control_id") or "")
        control = control_by_id.get(control_id)
        if control is None or control_id in cases:
            return {"status": "held", "reason": "machine_calibration_candidates_invalid"}
        case_error = _machine_calibration_case_error(
            case,
            control=control,
            benchmark=benchmark,
            scorer_identity=scorer_identity,
            consensus_ledger_file=consensus_ledger_file,
            execution_ledger_file=execution_ledger_file,
            chronovisor_root=chronovisor_root,
        )
        if case_error:
            return {"status": "held", "reason": case_error}
        cases[control_id] = dict(case)
    policy_sha = _machine_calibration_policy_sha256()
    benchmark_sha = str(benchmark["manifest_sha256"])
    calibration_run_id = _canonical_sha(
        {
            "kind": "machine-scorer-calibration-v2",
            "benchmark_manifest_sha256": benchmark_sha,
            "scorer_identity_sha256": scorer_identity_sha,
            "policy_sha256": policy_sha,
        }
    )
    accepted = 0
    for control in controls:
        control_id = str(control["control_id"])
        if control_id in cases or accepted >= max(0, max_items):
            continue
        subject = _machine_calibration_subject(
            control, benchmark_manifest_sha256=benchmark_sha
        )
        prompt = build_recall_answer_adjudication_prompt(
            {"subject": subject, "subject_sha256": _sealed_canonical_sha(subject)}
        )
        if dry_run:
            return {
                "status": "waiting",
                "reason": "dry_run",
                "pending": len(controls) - len(cases),
            }
        adjudication = append_machine_consensus_receipt(
            kind="scorer_calibration_case_review",
            subject=subject,
            producer_policy_sha256=policy_sha,
            prompt=prompt,
            schema=RECALL_ANSWER_ADJUDICATION_SCHEMA,
            system=None,
            lane=ANSWER_ADJUDICATION_LANE,
            ledger_file=consensus_ledger_file,
            chronovisor_root=chronovisor_root,
            router_factory=router_factory,
        )
        if adjudication.get("status") != "accepted":
            return {
                "status": str(adjudication.get("status") or "held"),
                "reason": str(adjudication.get("reason") or "control_consensus_held"),
                "completed": len(cases),
            }
        receipt = adjudication["receipt"]
        pair_protocol = _preregistered_pair_protocol(
            seed=ANSWER_AUTHORITY_SEED,
            episode_id=control_id,
            episode_sha256=str(control["source_packet_sha256"]),
            split_manifest_sha256=benchmark_sha,
            gold_manifest_sha256=benchmark_sha,
            adapter_registry_sha256=scorer_identity_sha,
            evaluation_kind="machine-scorer-calibration-v2",
        )
        scoring = {
            **dict(pair_protocol["scoring"]),
            "evidence_manifest_sha256": benchmark_sha,
            "rubric_sha256": str(scorer_identity.get("rubric_sha256") or ""),
        }
        gold = {
            "evidence": {"source_packet": control["source_packet"]},
            "evidence_sha256": control["evidence_sha256"],
            "rubric_sha256": str(scorer_identity.get("rubric_sha256") or ""),
        }
        scores, score_error, execution_receipt = _score_answer(
            scorer,
            str(control["source_packet"]["prompt"]),
            str(control["answer"]),
            gold,
            scorer_identity,
            scoring,
            parent_run_id=calibration_run_id,
            execution_ledger_file=execution_ledger_file,
        )
        if score_error or scores is None:
            return {"status": "waiting", "reason": score_error or "scorer_failed"}
        case = {
            "control_id": control_id,
            "control": control,
            "consensus_receipt_sha256": receipt["receipt_sha256"],
            "scoring": scoring,
            "scorer_scores": scores,
            "execution_receipt_sha256": execution_receipt,
        }
        case_error = _machine_calibration_case_error(
            case,
            control=control,
            benchmark=benchmark,
            scorer_identity=scorer_identity,
            consensus_ledger_file=consensus_ledger_file,
            execution_ledger_file=execution_ledger_file,
            chronovisor_root=chronovisor_root,
        )
        if case_error:
            return {"status": "held", "reason": case_error}
        append_jsonl_durable(
            candidates_path,
            [{"case": case, "case_sha256": _canonical_sha(case)}],
            sort_keys=True,
        )
        cases[control_id] = case
        accepted += 1
    if len(cases) < len(controls):
        return {
            "status": "waiting",
            "reason": "machine_calibration_pending",
            "accepted": accepted,
            "completed": len(cases),
            "required": len(controls),
        }
    ordered = [cases[str(control["control_id"])] for control in controls]
    metrics, gates = _machine_calibration_case_metrics(ordered)
    if not all(gates.values()):
        return {"status": "held", "reason": "machine_calibration_gate_failed"}
    benchmark_payload = benchmark["payload"]
    benchmark_path = Path(str(benchmark.get("manifest_path") or ""))
    source_ledger_path = str(benchmark_payload.get("source_ledger_path") or "")
    source_ledger_sha = str(benchmark_payload.get("source_ledger_sha256") or "")
    payload = {
        "schema_version": MACHINE_SCORER_CALIBRATION_SCHEMA_VERSION,
        "artifact_kind": "machine-answer-scorer-calibration",
        "status": "passed",
        "reason": "verified",
        "frozen_at": _now_utc(),
        "benchmark_manifest_path": str(benchmark_path),
        "benchmark_manifest_sha256": benchmark_sha,
        "benchmark_epoch_sha256": benchmark_epoch_sha,
        "benchmark_source_ledger_path": source_ledger_path,
        "benchmark_source_ledger_sha256": source_ledger_sha,
        "policy": _machine_calibration_policy(),
        "policy_sha256": policy_sha,
        "scorer_identity": dict(scorer_identity),
        "scorer_identity_sha256": scorer_identity_sha,
        "calibration_run_id": calibration_run_id,
        "cases": ordered,
        "metrics": metrics,
        "gates": gates,
    }
    check = validate_machine_scorer_calibration_artifact(
        seal_object(payload),
        scorer_identity=scorer_identity,
        benchmark_manifest=None,
        consensus_ledger_file=consensus_ledger_file,
        execution_ledger_file=execution_ledger_file,
        chronovisor_root=chronovisor_root,
    )
    if check.get("passed") is not True:
        return {"status": "held", "reason": str(check.get("reason"))}
    try:
        artifact = _create_once_sealed(artifact_path, payload)
        pointer_check = _publish_machine_calibration_pointer(
            pointer_path=pointer_path,
            artifact_path=artifact_path,
            artifact=artifact,
            benchmark_epoch_sha256=benchmark_epoch_sha,
            scorer_identity_sha256=scorer_identity_sha,
            scorer_identity=scorer_identity,
            consensus_ledger_file=consensus_ledger_file,
            execution_ledger_file=execution_ledger_file,
            chronovisor_root=chronovisor_root,
        )
    except (DurableStateError, OSError, ValueError):
        return {"status": "held", "reason": "machine_calibration_immutable_conflict"}
    if pointer_check.get("passed") is not True:
        return {"status": "held", "reason": "machine_calibration_pointer_write_failed"}
    return {
        "status": "complete",
        "reason": "verified",
        "artifact": str(artifact_path),
        "pointer": str(pointer_path),
        "manifest_sha256": str(artifact["seal_sha256"]),
    }


def validate_machine_scorer_calibration_artifact(
    value: Path | Mapping[str, Any],
    *,
    scorer_identity: Mapping[str, Any],
    benchmark_manifest: Path | Mapping[str, Any],
    consensus_ledger_file: Path = ANSWER_CONSENSUS_LEDGER,
    execution_ledger_file: Path = ANSWER_EXECUTION_LEDGER,
    chronovisor_root: Path = CHRONOVISOR_ROOT,
    evaluated_at: str = "",
) -> dict[str, Any]:
    try:
        payload = (
            read_sealed_json(value)
            if isinstance(value, Path)
            else verify_sealed_object(dict(value))
        )
    except (DurableStateError, TypeError, ValueError):
        return {"passed": False, "reason": "machine_calibration_seal_invalid"}
    benchmark = validate_independent_answer_benchmark(benchmark_manifest)
    controls = _machine_calibration_controls(benchmark)
    cases = payload.get("cases")
    scorer_identity_sha = _canonical_sha(dict(scorer_identity))
    policy_sha = _machine_calibration_policy_sha256()
    calibration_run_id = _canonical_sha(
        {
            "kind": "machine-scorer-calibration-v2",
            "benchmark_manifest_sha256": str(benchmark.get("manifest_sha256") or ""),
            "scorer_identity_sha256": scorer_identity_sha,
            "policy_sha256": policy_sha,
        }
    )
    if (
        benchmark.get("passed") is not True
        or benchmark.get("payload", {}).get("source_kind")
        != "machine_search_label_consensus"
        or len(controls) != MACHINE_SCORER_CALIBRATION_MIN_CASES
        or payload.get("schema_version") != MACHINE_SCORER_CALIBRATION_SCHEMA_VERSION
        or payload.get("artifact_kind") != "machine-answer-scorer-calibration"
        or payload.get("benchmark_manifest_sha256")
        != benchmark.get("manifest_sha256")
        or payload.get("benchmark_epoch_sha256") != benchmark.get("split_epoch_id")
        or payload.get("policy") != _machine_calibration_policy()
        or payload.get("policy_sha256") != policy_sha
        or payload.get("scorer_identity") != dict(scorer_identity)
        or payload.get("scorer_identity_sha256") != scorer_identity_sha
        or payload.get("calibration_run_id") != calibration_run_id
        or not _strict_utc(payload.get("frozen_at"))
        or not isinstance(cases, list)
        or len(cases) != MACHINE_SCORER_CALIBRATION_MIN_CASES
    ):
        return {"passed": False, "reason": "machine_calibration_identity_invalid"}
    expected_by_id = {str(control["control_id"]): control for control in controls}
    observed: list[Mapping[str, Any]] = []
    seen: set[str] = set()
    for case in cases:
        if not isinstance(case, Mapping):
            return {"passed": False, "reason": "machine_calibration_case_invalid"}
        control_id = str(case.get("control_id") or "")
        control = expected_by_id.get(control_id)
        if control is None or control_id in seen or case.get("control") != control:
            return {"passed": False, "reason": "machine_calibration_control_drift"}
        subject = _machine_calibration_subject(
            control,
            benchmark_manifest_sha256=str(benchmark["manifest_sha256"]),
        )
        prompt = build_recall_answer_adjudication_prompt(
            {"subject": subject, "subject_sha256": _sealed_canonical_sha(subject)}
        )
        receipt_check = validate_machine_consensus_receipt(
            case.get("consensus_receipt_sha256"),
            expected_kind="scorer_calibration_case_review",
            expected_subject=subject,
            expected_producer_policy_sha256=policy_sha,
            prompt=prompt,
            schema=RECALL_ANSWER_ADJUDICATION_SCHEMA,
            system=None,
            lane=ANSWER_ADJUDICATION_LANE,
            ledger_file=consensus_ledger_file,
            chronovisor_root=chronovisor_root,
        )
        receipt = receipt_check.get("receipt", {})
        artifact = receipt_check.get("artifact", {})
        votes = artifact.get("provenance", {}).get("vote_manifest", []) if isinstance(artifact, Mapping) else []
        models = receipt.get("authority", {}).get("router", {}).get("models", []) if isinstance(receipt, Mapping) else []
        scoring = case.get("scoring")
        scores = case.get("scorer_scores")
        expected_reset = {
            "seed": scoring.get("seed") if isinstance(scoring, Mapping) else None,
            "base_state_sha256": scoring.get("base_state_sha256") if isinstance(scoring, Mapping) else None,
            "reset_protocol_sha256": scorer_identity.get("policy_sha256"),
        }
        execution_error = _execution_receipt_error(
            receipt_sha256=case.get("execution_receipt_sha256"),
            expected_kind="answer_scorer_call",
            expected_adapter_identity_sha256=scorer_identity_sha,
            expected_parent_run_id=calibration_run_id,
            expected_input_payload={
                "prompt_sha256": control["query_sha256"],
                "answer_sha256": control["answer_sha256"],
                "gold_evidence_sha256": control["evidence_sha256"],
                "scoring": scoring,
            },
            expected_output_payload={
                "dimensions": scores,
                "reset_receipt": expected_reset,
            },
            ledger_file=execution_ledger_file,
            completed_before=str(payload.get("frozen_at") or ""),
        )
        if (
            receipt_check.get("passed") is not True
            or not isinstance(votes, list)
            or len(votes) != 2
            or [vote.get("role") for vote in votes if isinstance(vote, Mapping)]
            != ["primary", "challenger"]
            or not isinstance(models, list)
            or len(models) != 3
            or not isinstance(scores, Mapping)
            or set(scores) != set(ANSWER_DIMENSIONS)
            or any(not _finite_unit_score(scores.get(name)) for name in ANSWER_DIMENSIONS)
            or execution_error
        ):
            return {"passed": False, "reason": "machine_calibration_case_invalid"}
        seen.add(control_id)
        observed.append(case)
    if set(seen) != set(expected_by_id):
        return {"passed": False, "reason": "machine_calibration_control_set_invalid"}
    metrics, gates = _machine_calibration_case_metrics(observed)
    frozen_at = _strict_utc(payload.get("frozen_at"))
    time_error = bool(
        evaluated_at
        and (
            not _strict_utc(evaluated_at)
            or datetime.fromisoformat(frozen_at.replace("Z", "+00:00"))
            >= datetime.fromisoformat(_strict_utc(evaluated_at).replace("Z", "+00:00"))
        )
    )
    passed = bool(
        payload.get("metrics") == metrics
        and payload.get("gates") == gates
        and all(gates.values())
        and payload.get("status") == "passed"
        and payload.get("reason") == "verified"
        and not time_error
    )
    return {
        "passed": passed,
        "reason": "verified" if passed else "machine_calibration_evidence_incomplete",
        "manifest_sha256": str(payload.get("seal_sha256") or ""),
        "scorer_identity_sha256": scorer_identity_sha,
        "metrics": metrics,
        "payload": payload,
    }


def _answer_evaluation_gates(
    *,
    sealed_split_manifest: bool,
    immutable_gold_manifest: bool,
    scorer_calibration: bool,
    preregistered_before_run: bool,
    episode_ledger_exact: bool,
    registered_adapters: bool,
    evaluation_kind_authorized: bool,
    runner_identity: bool,
    all_pairs_verified: bool,
    minimum_independent_samples: bool,
    valid_confidence_bound: bool,
    authority_protocol_fixed: bool,
    improvement_point: bool,
    improvement_lower_bound: bool,
    non_degradation: bool,
    no_leakage: bool,
    dimension_bounds_valid: bool,
) -> dict[str, bool]:
    """Project the stable, byte-significant answer authority gate envelope."""

    return {
        "sealed_split_manifest": sealed_split_manifest,
        "immutable_gold_manifest": immutable_gold_manifest,
        "scorer_calibration": scorer_calibration,
        "preregistered_before_run": preregistered_before_run,
        "episode_ledger_exact": episode_ledger_exact,
        "registered_adapters": registered_adapters,
        "evaluation_kind_authorized": evaluation_kind_authorized,
        "runner_identity": runner_identity,
        "all_pairs_verified": all_pairs_verified,
        "minimum_independent_samples": minimum_independent_samples,
        "valid_confidence_bound": valid_confidence_bound,
        "authority_protocol_fixed": authority_protocol_fixed,
        "improvement_point": improvement_point,
        "improvement_lower_bound": improvement_lower_bound,
        "non_degradation": non_degradation,
        "no_leakage": no_leakage,
        "dimension_bounds_valid": dimension_bounds_valid,
    }


def _validated_answer_outcome_gates(
    *,
    split_valid: bool,
    gold_valid: bool,
    calibration_valid: bool,
    preregistration_valid: bool,
    adapter_registry_valid: bool,
    evaluation_kind: str,
    required_split: str,
    identities_valid: bool,
    verified_count: int,
    expected_count: int,
    cluster_value: Any,
    manifest: Mapping[str, Any] | None,
    recomputed: Mapping[str, Any],
    confidence_value: float,
    seed_value: int,
    result_ids: Sequence[str],
    dimension_bounds: Mapping[str, Mapping[str, Any]],
) -> dict[str, bool]:
    """Recompute the authority gate envelope from validated artifact evidence."""

    minimum_samples = (
        isinstance(cluster_value, int)
        and isinstance(manifest, Mapping)
        and isinstance(manifest.get("minimum_independent_samples"), int)
        and not isinstance(manifest.get("minimum_independent_samples"), bool)
        and cluster_value >= manifest["minimum_independent_samples"]
    )
    return _answer_evaluation_gates(
        sealed_split_manifest=split_valid,
        immutable_gold_manifest=gold_valid,
        scorer_calibration=calibration_valid,
        preregistered_before_run=preregistration_valid,
        episode_ledger_exact=True,
        registered_adapters=adapter_registry_valid,
        evaluation_kind_authorized=(
            evaluation_kind == "historical-context-utility"
            and required_split == "train"
        )
        or (
            evaluation_kind == "field-e2e-replay"
            and required_split == "locked-test"
        ),
        runner_identity=identities_valid,
        all_pairs_verified=verified_count == expected_count and bool(expected_count),
        minimum_independent_samples=minimum_samples,
        valid_confidence_bound=recomputed.get("valid") is True,
        authority_protocol_fixed=confidence_value
        == ANSWER_AUTHORITY_CONFIDENCE
        and seed_value == ANSWER_AUTHORITY_SEED,
        improvement_point=isinstance(recomputed.get("point"), int | float)
        and isinstance(manifest, Mapping)
        and isinstance(manifest.get("improvement_point_floor"), int | float)
        and not isinstance(manifest.get("improvement_point_floor"), bool)
        and float(recomputed["point"]) >= float(manifest["improvement_point_floor"]),
        improvement_lower_bound=isinstance(recomputed.get("lower"), int | float)
        and isinstance(manifest, Mapping)
        and isinstance(manifest.get("improvement_lcb_floor"), int | float)
        and not isinstance(manifest.get("improvement_lcb_floor"), bool)
        and float(recomputed["lower"]) >= float(manifest["improvement_lcb_floor"]),
        non_degradation=isinstance(recomputed.get("lower"), int | float)
        and float(recomputed["lower"]) >= 0.0,
        no_leakage=len(result_ids) == len(set(result_ids)) == expected_count,
        dimension_bounds_valid=all(
            item.get("valid") is True for item in dimension_bounds.values()
        ),
    )


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
    consensus_ledger_file: Path = ANSWER_CONSENSUS_LEDGER,
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
            consensus_ledger_file=consensus_ledger_file,
            expected_split=split,
            split_epoch_id=_split_epoch_id(split_value),
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
            consensus_ledger_file=consensus_ledger_file,
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
            safe_gold = _project_gold_for_scorer(gold)
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
                        safe_gold,
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
                    "query_sha256": _sha_text(turn.prompt),
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
    gates = _answer_evaluation_gates(
        sealed_split_manifest=split_check.get("passed") is True,
        immutable_gold_manifest=gold_check.get("passed") is True,
        scorer_calibration=calibration_check.get("passed") is True,
        preregistered_before_run=preregistration_valid,
        episode_ledger_exact=ledger_exact,
        registered_adapters=adapter_check.get("passed") is True,
        evaluation_kind_authorized=(
            evaluation_kind == "historical-context-utility" and split == "train"
        )
        or (evaluation_kind == "field-e2e-replay" and split == "locked-test"),
        runner_identity=not bool(identity_error),
        all_pairs_verified=bool(selected) and len(verified) == len(selected),
        minimum_independent_samples=cluster_count >= min_independent_samples,
        valid_confidence_bound=bound.get("valid") is True,
        authority_protocol_fixed=confidence == ANSWER_AUTHORITY_CONFIDENCE
        and seed == ANSWER_AUTHORITY_SEED,
        improvement_point=isinstance(bound.get("point"), int | float)
        and float(bound["point"]) >= improvement_point_floor,
        improvement_lower_bound=isinstance(bound.get("lower"), int | float)
        and float(bound["lower"]) >= improvement_lcb_floor,
        non_degradation=isinstance(bound.get("lower"), int | float)
        and float(bound["lower"]) >= 0.0,
        no_leakage=len({str(row.get("episode_id") or "") for row in selected})
        == len(selected_ids)
        == len(selected),
        dimension_bounds_valid=all(
            item.get("valid") is True for item in dimension_bounds.values()
        ),
    )
    passed = bool(gates and all(value is True for value in gates.values()))
    effects_authorized = all(
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
    )
    page_rewards, page_penalties = (
        _strict_outcome_page_effects(verified) if effects_authorized else ([], [])
    )
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


def evaluate_independent_answer_benchmark(
    *,
    runner: AnswerRunner | None,
    scorer: AnswerScorer | None,
    runner_identity: Mapping[str, Any] | None,
    scorer_identity: Mapping[str, Any] | None,
    benchmark_manifest: Path | Mapping[str, Any] = INDEPENDENT_ANSWER_BENCHMARK,
    gold_manifest: Path | Mapping[str, Any] | None = None,
    scorer_calibration: Path | Mapping[str, Any] | None = None,
    field_environment_replay: FieldEnvironmentReplay | None = None,
    field_environment_identity: Mapping[str, Any] | None = None,
    consensus_ledger_file: Path = ANSWER_CONSENSUS_LEDGER,
    review_ledger_file: Path = ANSWER_REVIEW_LEDGER,
    execution_ledger_file: Path = ANSWER_EXECUTION_LEDGER,
    adapter_registry: Path | Mapping[str, Any] = ANSWER_ADAPTER_REGISTRY,
    chronovisor_root: Path = CHRONOVISOR_ROOT,
    output_file: Path | None = None,
    split: str = "train",
    min_independent_samples: int = 20,
) -> dict[str, Any]:
    """Evaluate fresh Recall replay on independent benchmark episodes only."""

    evaluated_at = _now_utc()
    if min_independent_samples != 20:
        raise ValueError("benchmark minimum independent samples is fixed at 20")
    if split not in {"train", "locked-test"}:
        raise ValueError("benchmark evaluation split is invalid")
    benchmark_check = validate_independent_answer_benchmark(benchmark_manifest)
    benchmark_payload = benchmark_check.get("payload", {})
    selected = sorted(
        [
            dict(packet)
            for packet in benchmark_check.get("entries", {}).values()
            if packet.get("split") == split
        ],
        key=lambda packet: str(packet.get("case_id") or ""),
    )
    selected_ids = [str(packet["case_id"]) for packet in selected]
    machine_benchmark_authority = bool(
        benchmark_payload.get("schema_version") == 2
        and benchmark_payload.get("source_kind")
        == "machine_search_label_consensus"
    )
    gold_check = (
        validate_gold_manifest(
            gold_manifest,
            required_episode_ids=selected_ids,
            consensus_ledger_file=consensus_ledger_file,
            expected_split=split,
            split_epoch_id=str(benchmark_check.get("split_epoch_id") or ""),
            benchmark_manifest=benchmark_manifest,
        )
        if gold_manifest is not None
        else {"passed": False, "reason": "missing_gold_manifest"}
    )
    calibration_check = (
        validate_scorer_calibration_artifact(
            scorer_calibration,
            scorer_identity=scorer_identity or {},
            answer_split_manifest=None,
            independent_benchmark_manifest=benchmark_manifest,
            evaluated_at=evaluated_at,
            review_ledger_file=review_ledger_file,
            consensus_ledger_file=consensus_ledger_file,
            execution_ledger_file=execution_ledger_file,
            chronovisor_root=chronovisor_root,
        )
        if scorer_calibration is not None
        else {"passed": False, "reason": "missing_scorer_calibration"}
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
        adapter_registry, required=required_adapters, evaluated_at=evaluated_at
    )
    preregistered = all(
        timestamp
        and datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
        < datetime.fromisoformat(evaluated_at.replace("Z", "+00:00"))
        for timestamp in (
            _strict_utc(benchmark_payload.get("frozen_at")),
            _strict_utc(gold_check.get("payload", {}).get("frozen_at")),
        )
    )
    identity_error = (
        str(benchmark_check.get("reason") or "benchmark_invalid")
        if benchmark_check.get("passed") is not True
        else "machine_benchmark_authority_required"
        if not machine_benchmark_authority
        else "waiting_for_machine_expansion"
        if benchmark_check.get("promotion_ready") is not True
        else "benchmark_split_empty"
        if not selected
        else str(gold_check.get("reason") or "gold_manifest_invalid")
        if gold_check.get("passed") is not True
        else str(calibration_check.get("reason") or "scorer_calibration_invalid")
        if calibration_check.get("passed") is not True
        else "benchmark_not_preregistered"
        if not preregistered
        else str(adapter_check.get("reason") or "adapter_registry_invalid")
        if adapter_check.get("passed") is not True
        else "missing_runner"
        if runner is None
        else "missing_scorer"
        if scorer is None
        else "missing_field_environment_replay"
        if field_environment_replay is None
        else "field_environment_adapter_not_builtin"
        if field_environment_replay is not builtin_field_environment_replay
        else "field_environment_identity_mismatch"
        if dict(field_environment_identity or {})
        != builtin_field_environment_identity()
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
    run_receipt = {
        "evaluated_at": evaluated_at,
        "split": split,
        "benchmark_manifest_sha256": str(
            benchmark_check.get("manifest_sha256") or ""
        ),
        "benchmark_epoch_sha256": str(
            benchmark_check.get("split_epoch_id") or ""
        ),
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
    }
    run_receipt["receipt_sha256"] = _canonical_sha(run_receipt)
    results: list[dict[str, Any]] = []
    if not identity_error:
        assert runner is not None and scorer is not None
        assert field_environment_replay is not None
        for packet in selected:
            case_id = str(packet["case_id"])
            pair_protocol = _preregistered_pair_protocol(
                seed=ANSWER_AUTHORITY_SEED,
                episode_id=case_id,
                episode_sha256=packet["source_entry_sha256"],
                split_manifest_sha256=benchmark_check["manifest_sha256"],
                gold_manifest_sha256=gold_check["manifest_sha256"],
                adapter_registry_sha256=adapter_check["manifest_sha256"],
                evaluation_kind="independent-benchmark-field-replay",
            )
            pair_seed = int(pair_protocol["pair_seed"])
            contexts, environment, environment_error = _field_environment_contexts(
                field_environment_replay,
                prompt=str(packet["prompt"]),
                episode={
                    "episode_id": case_id,
                    "session_hash": _sha_text(f"benchmark:{case_id}")[:16],
                    "evaluation_kind": "independent-benchmark-field-replay",
                },
                pair_seed=pair_seed,
                identity=field_environment_identity or {},
                parent_run_id=run_receipt["receipt_sha256"],
                execution_ledger_file=execution_ledger_file,
            )
            if environment_error:
                results.append(
                    {"case_id": case_id, "status": "unknown", "reason": environment_error}
                )
                continue
            arm_contexts = {
                "field_on": contexts["candidate_field"],
                "field_off": contexts["production_teacher"],
            }
            generation = dict(pair_protocol["generation"])
            generated = {
                arm: _runner_answer(
                    runner,
                    str(packet["prompt"]),
                    arm_contexts[arm],
                    generation,
                    runner_identity or {},
                    parent_run_id=run_receipt["receipt_sha256"],
                    execution_ledger_file=execution_ledger_file,
                )
                for arm in pair_protocol["arm_order"]
            }
            scoring = {
                **dict(pair_protocol["scoring"]),
                "evidence_manifest_sha256": gold_check["manifest_sha256"],
                "rubric_sha256": gold_check["rubric_sha256"],
            }
            gold = gold_check["entries"][case_id]
            safe_gold = _project_gold_for_scorer(gold)
            scored: dict[str, tuple[dict[str, float] | None, str, str]] = {}
            for arm in pair_protocol["arm_order"]:
                answer, runner_error, _runner_receipt = generated[arm]
                scored[arm] = (
                    _score_answer(
                        scorer,
                        str(packet["prompt"]),
                        answer,
                        safe_gold,
                        scorer_identity or {},
                        scoring,
                        parent_run_id=run_receipt["receipt_sha256"],
                        execution_ledger_file=execution_ledger_file,
                    )
                    if not runner_error
                    else (None, runner_error, "")
                )
            on_answer, on_error, on_runner_receipt = generated["field_on"]
            off_answer, off_error, off_runner_receipt = generated["field_off"]
            on_score, on_score_error, on_scorer_receipt = scored["field_on"]
            off_score, off_score_error, off_scorer_receipt = scored["field_off"]
            error = on_error or off_error or on_score_error or off_score_error
            if error or on_score is None or off_score is None:
                results.append({"case_id": case_id, "status": "unknown", "reason": error})
                continue
            deltas = {
                dimension: round(on_score[dimension] - off_score[dimension], 9)
                for dimension in ANSWER_DIMENSIONS
            }
            result = {
                "case_id": case_id,
                "status": "verified",
                "component_sha256": packet["component_sha256"],
                "query_sha256": packet["prompt_content_sha256"],
                "source_entry_sha256": packet["source_entry_sha256"],
                "source_page_bindings": packet["page_bindings"],
                "pair_seed": pair_seed,
                "pair_protocol_sha256": pair_protocol["protocol_sha256"],
                "arm_order": pair_protocol["arm_order"],
                "environment_evidence": environment,
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
                "dimension_deltas": deltas,
                "score_delta": round(sum(deltas.values()) / len(deltas), 9),
                "cluster_nodes": [f"component:{packet['component_sha256']}"],
            }
            results.append(result)
    verified = [row for row in results if row.get("status") == "verified"]
    bound = cluster_bootstrap_interval(
        verified,
        value_key="score_delta",
        confidence=ANSWER_AUTHORITY_CONFIDENCE,
        seed=ANSWER_AUTHORITY_SEED,
    )
    gates = {
        "identity": not identity_error,
        "all_cases_verified": bool(selected) and len(verified) == len(selected),
        "minimum_samples": len(verified) >= min_independent_samples,
        "minimum_clusters": int(bound.get("clusters") or 0)
        >= min_independent_samples,
        "improvement_point": float(bound.get("point") or 0.0) >= 0.02,
        "improvement_lcb": float(bound.get("lower") or -1.0) > 0.0,
    }
    passed = bool(gates and all(gates.values()))
    payload = {
        "schema_version": ANSWER_BENCHMARK_EVAL_SCHEMA_VERSION,
        "artifact_kind": "independent-answer-benchmark-evaluation",
        "status": "passed" if passed else "held",
        "reason": identity_error or ("verified" if passed else "gate_failed"),
        "split": split,
        "benchmark_manifest": benchmark_payload,
        "gold_manifest": gold_check.get("payload", {}),
        "scorer_calibration": calibration_check.get("payload", {}),
        "adapter_registry": adapter_check.get("payload", {}),
        "runner_identity": dict(runner_identity or {}),
        "scorer_identity": dict(scorer_identity or {}),
        "field_environment_identity": dict(field_environment_identity or {}),
        "run_receipt": run_receipt,
        "minimum_independent_samples": min_independent_samples,
        "case_ids": selected_ids,
        "results": results,
        "confidence_bound": bound,
        "gates": gates,
        "page_rewards": [],
        "page_penalties": [],
        "learning_effects": [],
        "production_episode_ledger_used": False,
    }
    sealed = seal_object(payload)
    if output_file is not None:
        _create_once_sealed(output_file, payload)
    return sealed


def validate_locked_answer_artifact(
    value: Path | Mapping[str, Any],
    *,
    minimum_independent_samples: int = 20,
    episode_file: Path = ANSWER_EPISODE_LEDGER,
    review_ledger_file: Path = ANSWER_REVIEW_LEDGER,
    consensus_ledger_file: Path = ANSWER_CONSENSUS_LEDGER,
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
        consensus_ledger_file=consensus_ledger_file,
        execution_ledger_file=execution_ledger_file,
        adapter_registry=adapter_registry,
    )


def _strict_outcome_page_effects(
    rows: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Project trusted evaluator rows without tolerating malformed evidence."""

    rewards: list[dict[str, Any]] = []
    penalties: list[dict[str, Any]] = []
    for row in rows:
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
                rewards.append({**common, "reward": round(delta, 9)})
            else:
                penalties.append({**common, "penalty": round(-delta, 9)})
    return rewards, penalties


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
    consensus_ledger_file: Path = ANSWER_CONSENSUS_LEDGER,
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
        consensus_ledger_file=consensus_ledger_file,
        expected_split=required_split,
        split_epoch_id=_split_epoch_id(split_check.get("manifest", {})),
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
                != split_entry.get("query_sha256")
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
    expected_gates = _validated_answer_outcome_gates(
        split_valid=split_check.get("passed") is True,
        gold_valid=gold_check.get("passed") is True,
        calibration_valid=calibration_check.get("passed") is True,
        preregistration_valid=preregistration_valid,
        adapter_registry_valid=not bool(adapter_registry_error),
        evaluation_kind=evaluation_kind,
        required_split=required_split,
        identities_valid=identities_valid,
        verified_count=len(verified_rows),
        expected_count=len(expected_ids),
        cluster_value=cluster_value,
        manifest=manifest if isinstance(manifest, Mapping) else None,
        recomputed=recomputed,
        confidence_value=confidence_value,
        seed_value=seed_value,
        result_ids=result_ids,
        dimension_bounds=expected_dimension_bounds,
    )
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
    if args.status or args.dry_run:
        return _main_locked(args)
    from chronovisor.core.okf_cutover import OKFStartupBlocked

    try:
        with okf_runtime_operation(CHRONOVISOR_ROOT):
            return _main_locked(args)
    except OKFStartupBlocked:
        print(json.dumps({"status": "blocked", "category": "okf_startup_blocked"}))
        return 75


def _main_locked(args: argparse.Namespace) -> int:
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
