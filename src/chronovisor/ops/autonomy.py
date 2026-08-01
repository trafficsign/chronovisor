"""Autonomous operation layer for Chronovisor.

This module replaces human review queues with reversible machine decisions:
safe items are applied, uncertain items are deferred for the next cycle, and
health regressions quarantine the batch instead of waiting for a person.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import os
import plistlib
import re
import shlex
import shutil
import subprocess
import tempfile
from collections.abc import Callable, Mapping
from contextlib import contextmanager, suppress
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

from chronovisor.core.frontmatter import parse as parse_frontmatter
from chronovisor.core.frontmatter import patch as patch_frontmatter
from chronovisor.core.runtime_config import (
    runtime_identity,
    runtime_repo_root,
    uvx_runtime_command,
)
from chronovisor.core.store import CHRONOVISOR_ROOT, find_page
from chronovisor.decision.decision_authority import (
    compare_semantic_authority,
    current_semantic_authority,
    seal_semantic_artifact,
    semantic_authority_shape_error,
    semantic_verdict_authority_error,
)
from chronovisor.ingest.page_mutation import (
    chronovisor_mutation_lock,
    decision_authority_lock,
)
from chronovisor.ops.convergence import (
    ConvergenceStateError,
    ConvergenceStore,
    CycleBudget,
    InvalidTransition,
    is_human_required_result,
    stable_item_key,
)
from chronovisor.search.semantic_hold import (
    LOCAL_SEMANTIC_NO_QUORUM,
    canonical_sha256,
    is_local_semantic_no_quorum,
    persisted_semantic_no_quorum_hold,
    semantic_no_quorum_hold_error,
)

AUTONOMY_DIR = CHRONOVISOR_ROOT / "autonomy"
DECISIONS_FILE = AUTONOMY_DIR / "decisions.jsonl"
LATEST_FILE = AUTONOMY_DIR / "latest.json"
WATCHDOG_FILE = AUTONOMY_DIR / "watchdog-latest.json"
WATCHDOG_HISTORY = AUTONOMY_DIR / "watchdog-history.jsonl"
WATCHDOG_HEARTBEAT_FILE = AUTONOMY_DIR / "watchdog-heartbeat.json"
OBSERVER_HEARTBEAT_FILE = AUTONOMY_DIR / "observer-heartbeat.json"
WATCHDOG_NOTIFICATION_REMINDER_HOURS = 6.0
DIGEST_FILE = AUTONOMY_DIR / "digest-latest.md"
QUARANTINE_FILE = AUTONOMY_DIR / "quarantine.json"
PROJECT_ROOT = runtime_repo_root()

SLEEP_LABEL = "com.trafficsign.chronovisor-sleep"
CONVERGE_LABEL = "com.trafficsign.chronovisor-converge"
WATCHDOG_LABEL = "com.trafficsign.chronovisor-watchdog"
DEADMAN_LABEL = "com.trafficsign.chronovisor-deadman-observer"
SOAK_LABEL = "com.trafficsign.chronovisor-soak"
LAUNCH_AGENT_DIR = Path.home() / "Library" / "LaunchAgents"
WRAPPER_DIR = CHRONOVISOR_ROOT / "bin"
DUPLICATE_FRONTIER_LANE = "autonomy_duplicate_resolution"
RETENTION_FRONTIER_LANE = "autonomy_retention"
CONTENT_CORRECTION_LANE = "content_correction"
DUPLICATE_FRONTIER_RESOLVER_VERSION = "duplicate-frontier-v3-exact-postimage"
RETENTION_FRONTIER_RESOLVER_VERSION = "retention-frontier-v3-exact-postimage"
LEGACY_SEMANTIC_HOLD_KIND = "legacy_local_semantic_no_quorum_fail_closed"
DUPLICATE_FRONTIER_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["decision", "confidence", "summary"],
    "properties": {
        "decision": {
            "type": "string",
            "enum": ["supersede_left", "supersede_right", "keep_both", "needs_retry"],
        },
        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
        "summary": {"type": "string"},
    },
}
RETENTION_FRONTIER_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["decision", "confidence", "summary"],
    "properties": {
        "decision": {
            "type": "string",
            "enum": ["archive", "keep_active", "needs_retry"],
        },
        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
        "summary": {"type": "string"},
    },
}


def _now() -> str:
    return datetime.now().isoformat(timespec="seconds")


def _append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=str) + "\n",
        encoding="utf-8",
    )


def _read_json(path: Path) -> dict[str, Any]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return data if isinstance(data, dict) else {}


def _latest_jsonl(path: Path) -> dict[str, Any]:
    try:
        with path.open(encoding="utf-8") as f:
            lines = [line for line in f if line.strip()]
    except OSError:
        return {}
    if not lines:
        return {}
    try:
        row = json.loads(lines[-1])
    except json.JSONDecodeError:
        return {}
    return row if isinstance(row, dict) else {}


def _parse_dt(value: object) -> datetime | None:
    if not isinstance(value, str) or not value:
        return None
    try:
        return datetime.fromisoformat(value)
    except ValueError:
        return None


def _norm(text: object) -> str:
    if not isinstance(text, str):
        return ""
    text = text.casefold()
    text = re.sub(r"[\W_]+", " ", text, flags=re.UNICODE)
    return re.sub(r"\s+", " ", text).strip()


def _git(args: list[str], *, cwd: Path = CHRONOVISOR_ROOT) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", *args],
        cwd=cwd,
        text=True,
        capture_output=True,
        check=False,
    )


def _page_meta(page_id: str) -> dict[str, Any]:
    from chronovisor.search.index_store import get_store

    store = get_store()
    store.refresh()
    return store.meta(page_id) or {}


def _page_quality(page_id: str, meta: dict[str, Any] | None = None) -> float:
    from chronovisor.search.index_store import get_store

    meta = meta or _page_meta(page_id)
    if not meta:
        return -1.0
    score = 0.0
    if str(meta.get("summary") or "").strip():
        score += 3.0
    questions = meta.get("recall_questions")
    if isinstance(questions, list):
        score += min(3.0, len(questions) * 0.75)
    path_value = meta.get("path")
    if isinstance(path_value, str):
        try:
            size = Path(path_value).stat().st_size
            score += min(3.0, size / 3000.0)
        except OSError:
            pass
    try:
        store = get_store()
        score += min(2.0, len(store.backlinks(page_id)) * 0.3)
        score += min(1.0, len(store.outlinks(page_id)) * 0.1)
    except Exception:
        pass
    updated = _parse_dt(str(meta.get("updated") or ""))
    if updated and datetime.now() - updated <= timedelta(days=90):
        score += 0.5
    return score


def _chronovisor_root_for_page(path: Path) -> Path:
    """Infer the owning Wiki root without consulting a process-global path."""

    for parent in (path.parent, *path.parents):
        if parent.name == "pages":
            return parent.parent
    # Unit tests and one-off stores often place pages directly in a temporary
    # directory. Keeping their convergence lock beside that directory avoids
    # touching the live Wiki while preserving the production layout below it.
    return path.parent


def _content_correction_store_for_page(path: Path) -> ConvergenceStore:
    runtime = _chronovisor_root_for_page(path) / "runtime" / "convergence"
    return ConvergenceStore(
        runtime / "state.json",
        events_file=runtime / "events.jsonl",
        lock_file=runtime / "state.lock",
    )


def _pending_content_correction_targets(
    state: dict[str, Any],
) -> dict[str, set[str]]:
    targets: dict[str, set[str]] = {}
    items = state.get("items") if isinstance(state, dict) else None
    if not isinstance(items, dict):
        return targets
    for key, item in items.items():
        if (
            not isinstance(item, dict)
            or item.get("lane") != CONTENT_CORRECTION_LANE
            # Non-human quarantine is a cooldown, not a terminal abandonment:
            # content_correction will reopen it automatically.  Keep its pages
            # lifecycle-protected until the correction is actually resolved.
            or item.get("status") in {"applied", "rejected"}
        ):
            continue
        metadata = (
            item.get("metadata") if isinstance(item.get("metadata"), dict) else {}
        )
        pages = metadata.get("candidate_pages")
        if not isinstance(pages, list):
            event = (
                metadata.get("event") if isinstance(metadata.get("event"), dict) else {}
            )
            pages = event.get("candidate_pages")
        if not isinstance(pages, list):
            continue
        for page_id in pages:
            if isinstance(page_id, str) and page_id:
                targets.setdefault(page_id, set()).add(str(key))
    return targets


@contextmanager
def _lifecycle_mutation_guard(
    page_ids: list[str],
    *,
    page_path: Path,
    correction_store: ConvergenceStore | None = None,
):
    """Serialize lifecycle changes after excluding pending corrections.

    Lock order is always convergence state, then Wiki mutation. Correction
    workers never hold the Wiki lock while taking the convergence lock, so
    this closes the enqueue-to-lifecycle race without an inverse-order
    deadlock.
    """

    store = correction_store or _content_correction_store_for_page(page_path)
    with store._exclusive_lock():
        targets = _pending_content_correction_targets(store._load_unlocked())
        blocked = sorted(page_id for page_id in page_ids if page_id in targets)
        if blocked:
            keys = sorted({key for page_id in blocked for key in targets[page_id]})
            yield {
                "allowed": False,
                "reason": "pending_content_correction",
                "blocked_pages": blocked,
                "correction_keys": keys,
            }
            return
        with chronovisor_mutation_lock():
            yield {"allowed": True, "blocked_pages": [], "correction_keys": []}


def _write_unique_temp(path: Path, payload: bytes, *, token: str) -> Path:
    """Write an fsynced sibling temp file with a collision-proof name."""

    with tempfile.NamedTemporaryFile(
        mode="wb",
        dir=path.parent,
        prefix=f".{path.name}.{os.getpid()}.{token}.",
        suffix=".tmp",
        delete=False,
    ) as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
        return Path(handle.name)


def _bytes_receipt(payload: bytes) -> dict[str, Any]:
    """Return the exact byte identity used by durable effect receipts."""

    return {
        "sha256": hashlib.sha256(payload).hexdigest(),
        "size_bytes": len(payload),
    }


def _bytes_receipt_error(receipt: object) -> str | None:
    if not isinstance(receipt, dict) or set(receipt) != {"sha256", "size_bytes"}:
        return "effect byte receipt is invalid"
    digest = receipt.get("sha256")
    size = receipt.get("size_bytes")
    if (
        not isinstance(digest, str)
        or re.fullmatch(r"[0-9a-f]{64}", digest) is None
        or isinstance(size, bool)
        or not isinstance(size, int)
        or size < 0
    ):
        return "effect byte receipt is invalid"
    return None


def _effect_receipt_shape_error(receipt: object) -> str | None:
    """Validate the durable, pre-CAS description of one semantic effect."""

    required = {
        "receipt_version",
        "operation",
        "decision",
        "decision_at",
        "targets",
        "preimages",
        "postimages",
    }
    if not isinstance(receipt, dict) or set(receipt) != required:
        return "semantic effect receipt is invalid"
    if receipt.get("receipt_version") != 1:
        return "semantic effect receipt version is invalid"
    if not all(
        isinstance(receipt.get(field), str) and bool(receipt[field])
        for field in ("operation", "decision", "decision_at")
    ):
        return "semantic effect receipt identity is invalid"
    targets = receipt.get("targets")
    if (
        not isinstance(targets, dict)
        or not targets
        or any(
            not isinstance(name, str)
            or not name
            or not isinstance(page_id, str)
            or not page_id
            for name, page_id in targets.items()
        )
    ):
        return "semantic effect targets are invalid"
    for field in ("preimages", "postimages"):
        images = receipt.get(field)
        if not isinstance(images, dict):
            return f"semantic effect {field} are invalid"
        for page_id, byte_receipt in images.items():
            if not isinstance(page_id, str) or not page_id:
                return f"semantic effect {field} page identity is invalid"
            error = _bytes_receipt_error(byte_receipt)
            if error is not None:
                return error
    if set(receipt["postimages"]) - set(receipt["preimages"]):
        return "semantic effect postimage is missing a preimage"
    return None


def _render_frontmatter_postimage(
    original: bytes,
    updates: dict[str, Any],
) -> tuple[bytes | None, str | None]:
    """Render a lifecycle postimage without performing any filesystem write."""

    try:
        text = original.decode("utf-8")
        _meta, body = parse_frontmatter(text)
        updated = patch_frontmatter(text, updates)
        _updated_meta, updated_body = parse_frontmatter(updated)
    except (UnicodeDecodeError, ValueError) as exc:
        return None, f"postimage_render_failed:{exc}"
    if updated_body != body:
        return None, "body_change_refused"
    return updated.encode("utf-8"), None


def _prepare_duplicate_effect_receipt(
    *,
    decision: str,
    left: str,
    right: str,
    page_hashes: dict[str, str],
    decision_at: str,
    approval_key: str,
) -> tuple[dict[str, Any] | None, str | None]:
    if decision not in {"supersede_left", "supersede_right", "keep_both"}:
        return None, "duplicate effect decision is invalid"
    left_path = find_page(left)
    right_path = find_page(right)
    if left_path is None or right_path is None:
        return None, "winner_or_loser_missing"
    try:
        raw_by_page = {left: left_path.read_bytes(), right: right_path.read_bytes()}
    except OSError as exc:
        return None, f"effect_preimage_read_error:{exc}"
    if any(
        hashlib.sha256(raw_by_page[page_id]).hexdigest() != page_hashes.get(page_id)
        for page_id in (left, right)
    ):
        return None, "content_changed_before_approval"

    if decision == "keep_both":
        return {
            "receipt_version": 1,
            "operation": "keep_both",
            "decision": decision,
            "decision_at": decision_at,
            "targets": {"left": left, "right": right},
            "preimages": {
                page_id: _bytes_receipt(raw_by_page[page_id])
                for page_id in (left, right)
            },
            "postimages": {},
        }, None

    loser, winner = (left, right) if decision == "supersede_left" else (right, left)
    try:
        loser_meta, _body = parse_frontmatter(raw_by_page[loser].decode("utf-8"))
        winner_meta, _winner_body = parse_frontmatter(
            raw_by_page[winner].decode("utf-8")
        )
    except (UnicodeDecodeError, ValueError) as exc:
        return None, f"effect_preimage_parse_error:{exc}"
    if loser_meta.get("status") == "deprecated":
        return None, "loser_already_superseded"
    if winner_meta.get("status") in {"deprecated", "archived"}:
        return None, "winner_is_not_active"
    postimage, render_error = _render_frontmatter_postimage(
        raw_by_page[loser],
        {
            "status": "deprecated",
            "superseded_by": winner,
            "autonomy_decision": "duplicate_frontier_supersede",
            "autonomy_decision_at": decision_at,
            "frontier_approval_key": approval_key,
        },
    )
    if render_error is not None or postimage is None:
        return None, render_error or "postimage_render_failed"
    return {
        "receipt_version": 1,
        "operation": "soft_supersede",
        "decision": decision,
        "decision_at": decision_at,
        "targets": {"loser": loser, "winner": winner},
        "preimages": {
            page_id: _bytes_receipt(raw_by_page[page_id]) for page_id in (loser, winner)
        },
        "postimages": {loser: _bytes_receipt(postimage)},
    }, None


def _prepare_retention_effect_receipt(
    *,
    decision: str,
    page_id: str,
    page_hash: str,
    decision_at: str,
    approval_key: str,
) -> tuple[dict[str, Any] | None, str | None]:
    if decision not in {"archive", "keep_active"}:
        return None, "retention effect decision is invalid"
    path = find_page(page_id)
    if path is None:
        return None, "page_not_found"
    try:
        original = path.read_bytes()
    except OSError as exc:
        return None, f"effect_preimage_read_error:{exc}"
    if hashlib.sha256(original).hexdigest() != page_hash:
        return None, "content_changed_before_approval"
    receipt = {
        "receipt_version": 1,
        "operation": "keep_active" if decision == "keep_active" else "soft_archive",
        "decision": decision,
        "decision_at": decision_at,
        "targets": {"page_id": page_id},
        "preimages": {page_id: _bytes_receipt(original)},
        "postimages": {},
    }
    if decision == "keep_active":
        return receipt, None
    postimage, render_error = _render_frontmatter_postimage(
        original,
        {
            "status": "archived",
            "autonomy_decision": "retention_frontier_archive",
            "autonomy_decision_at": decision_at,
            "frontier_approval_key": approval_key,
            "archive_reason": "frontier_approved_reversible_soft_archive",
        },
    )
    if render_error is not None or postimage is None:
        return None, render_error or "postimage_render_failed"
    receipt["postimages"] = {page_id: _bytes_receipt(postimage)}
    return receipt, None


def _frontier_approval_path(
    store: ConvergenceStore,
    *,
    lane: str,
    key: str,
) -> Path:
    digest = hashlib.sha256(key.encode("utf-8")).hexdigest()
    return store.state_file.parent / "approvals" / lane / f"{digest}.json"


def _current_autonomy_authority(
    lane: str,
    *,
    reviewer: Callable[[dict[str, Any]], dict[str, Any]] | None,
) -> tuple[dict[str, Any] | None, str | None]:
    """Resolve the exact semantic epoch allowed to affect an autonomy lane."""

    review_fn = (
        _review_deferred_duplicate
        if lane == DUPLICATE_FRONTIER_LANE
        else _review_retention_candidate
    )
    return current_semantic_authority(
        lane,
        injected_reviewer=(
            reviewer is not None or getattr(review_fn, "__module__", None) != __name__
        ),
    )


def _autonomy_authority_epoch_error(
    expected: dict[str, Any],
    review: dict[str, Any],
    *,
    lane: str,
    reviewer: Callable[[dict[str, Any]], dict[str, Any]] | None,
) -> str | None:
    current, current_error = _current_autonomy_authority(lane, reviewer=reviewer)
    return (
        current_error
        or compare_semantic_authority(expected, current, lane=lane)
        or semantic_verdict_authority_error(review, expected, lane=lane)
    )


def _without_observation_timestamps(value: Any) -> Any:
    """Remove timestamps that are not part of a semantic review request."""

    if isinstance(value, dict):
        return {
            str(key): _without_observation_timestamps(item)
            for key, item in value.items()
            if str(key) not in {"ts", "timestamp", "observed_at"}
        }
    if isinstance(value, list):
        return [_without_observation_timestamps(item) for item in value]
    return value


def _duplicate_semantic_epoch(
    item: Mapping[str, Any],
    *,
    candidate: Mapping[str, Any],
    local_decision: Mapping[str, Any],
    page_hashes: Mapping[str, str],
) -> dict[str, Any]:
    return {
        "resolver_version": DUPLICATE_FRONTIER_RESOLVER_VERSION,
        "input_hash": str(item.get("input_hash") or ""),
        "candidate": dict(candidate),
        "local_proposal": _without_observation_timestamps(dict(local_decision)),
        "page_hashes": dict(sorted(page_hashes.items())),
    }


def _retention_semantic_epoch(
    item: Mapping[str, Any],
    *,
    page_id: str,
    page_hash: str,
    retention: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "resolver_version": RETENTION_FRONTIER_RESOLVER_VERSION,
        "input_hash": str(item.get("input_hash") or ""),
        "page_id": page_id,
        "page_hash": page_hash,
        "retention": dict(retention),
        "local_recommendation": "archive",
    }


def _legacy_semantic_hold(
    *,
    lane: str,
    epoch: Mapping[str, Any],
    authority: Mapping[str, Any],
    item: Mapping[str, Any],
) -> dict[str, Any]:
    """Bind an old retry-exhausted no-quorum row without inventing a verdict."""

    payload: dict[str, Any] = {
        "schema_version": 1,
        "kind": LEGACY_SEMANTIC_HOLD_KIND,
        "lane": lane,
        "epoch": dict(epoch),
        "epoch_sha256": canonical_sha256(epoch),
        "authority": dict(authority),
        "authority_sha256": canonical_sha256(authority),
        "migrated_from": {
            "quarantine_reason": str(item.get("quarantine_reason") or ""),
            "frontier_attempts": int(item.get("frontier_attempts") or 0),
            "last_error": str(item.get("last_error") or ""),
        },
    }
    payload["hold_sha256"] = canonical_sha256(payload)
    return payload


def _legacy_semantic_hold_error(
    hold: object,
    *,
    lane: str,
    epoch: Mapping[str, Any] | None = None,
    authority: Mapping[str, Any] | None = None,
) -> str | None:
    if not isinstance(hold, Mapping):
        return "legacy semantic hold is missing"
    expected_fields = {
        "schema_version",
        "kind",
        "lane",
        "epoch",
        "epoch_sha256",
        "authority",
        "authority_sha256",
        "migrated_from",
        "hold_sha256",
    }
    if set(hold) != expected_fields:
        return "legacy semantic hold fields are invalid"
    if (
        hold.get("schema_version") != 1
        or hold.get("kind") != LEGACY_SEMANTIC_HOLD_KIND
        or hold.get("lane") != lane
        or not isinstance(hold.get("epoch"), Mapping)
        or not isinstance(hold.get("authority"), Mapping)
        or not isinstance(hold.get("migrated_from"), Mapping)
    ):
        return "legacy semantic hold identity is invalid"
    stored_epoch = hold["epoch"]
    stored_authority = hold["authority"]
    authority_error = semantic_authority_shape_error(stored_authority, lane=lane)
    if authority_error is not None:
        return authority_error
    if hold.get("epoch_sha256") != canonical_sha256(stored_epoch):
        return "legacy semantic hold epoch digest is invalid"
    if hold.get("authority_sha256") != canonical_sha256(stored_authority):
        return "legacy semantic hold authority digest is invalid"
    unsigned = dict(hold)
    unsigned.pop("hold_sha256", None)
    if hold.get("hold_sha256") != canonical_sha256(unsigned):
        return "legacy semantic hold self digest is invalid"
    if epoch is not None and dict(stored_epoch) != dict(epoch):
        return "legacy semantic hold epoch changed"
    if authority is not None and dict(stored_authority) != dict(authority):
        return "legacy semantic hold authority changed"
    return None


def _resume_or_migrate_autonomy_semantic_hold(
    store: ConvergenceStore,
    item: Mapping[str, Any],
    *,
    lane: str,
    epoch: Mapping[str, Any],
    authority: Mapping[str, Any],
    now: datetime | None,
    dry_run: bool,
) -> tuple[dict[str, Any], bool]:
    """Return ``(item, held)`` after exact semantic-hold validation."""

    key = str(item.get("key") or "")
    common_hold = persisted_semantic_no_quorum_hold(item, lane=lane)
    if common_hold is not None:
        hold_error = semantic_no_quorum_hold_error(
            common_hold,
            lane,
            epoch=epoch,
            authority=authority,
        )
        if hold_error is None:
            return dict(item), True
        resumed = store.resume_quarantined(
            key,
            stage="frontier",
            reason=hold_error,
            resume_context={
                "reason": "semantic_hold_epoch_changed",
                "decision_lane": lane,
                "invalidated_semantic_hold": common_hold,
                "invalidated_hold_sha256": str(common_hold["hold_sha256"]),
                "expected_epoch": dict(epoch),
                "expected_epoch_sha256": canonical_sha256(epoch),
                "expected_authority": dict(authority),
            },
            now=now,
            dry_run=dry_run,
        )
        return dict(resumed["item"]), False

    result = item.get("result") if isinstance(item.get("result"), Mapping) else {}
    legacy_hold = result.get("legacy_semantic_hold")
    if _legacy_semantic_hold_error(legacy_hold, lane=lane) is None:
        hold_error = _legacy_semantic_hold_error(
            legacy_hold,
            lane=lane,
            epoch=epoch,
            authority=authority,
        )
        if hold_error is None:
            return dict(item), True
        resumed = store.resume_quarantined(
            key,
            stage="frontier",
            reason=hold_error,
            resume_context={
                "reason": "legacy_semantic_hold_epoch_changed",
                "decision_lane": lane,
                "invalidated_semantic_hold": dict(legacy_hold),
                "invalidated_hold_sha256": str(legacy_hold["hold_sha256"]),
                "expected_epoch": dict(epoch),
                "expected_epoch_sha256": canonical_sha256(epoch),
                "expected_authority": dict(authority),
            },
            now=now,
            dry_run=dry_run,
        )
        return dict(resumed["item"]), False

    if str(item.get("last_failure_class") or "") != LOCAL_SEMANTIC_NO_QUORUM:
        return dict(item), True
    marker = _legacy_semantic_hold(
        lane=lane,
        epoch=epoch,
        authority=authority,
        item=item,
    )
    migrated = store.quarantine(
        key,
        reason=f"semantic_no_quorum_legacy:{lane}",
        error=(
            str(item.get("last_error") or "")
            or "legacy local semantic no-quorum preserved fail-closed"
        ),
        failure_class=LOCAL_SEMANTIC_NO_QUORUM,
        result={
            "terminal_reason": "semantic_no_quorum",
            "legacy_semantic_hold": marker,
        },
        now=now,
        dry_run=dry_run,
    )
    return dict(migrated["item"]), True


def _persist_frontier_approval(
    store: ConvergenceStore,
    *,
    lane: str,
    key: str,
    input_hash: str,
    page_hashes: dict[str, str],
    review: dict[str, Any],
    authority: dict[str, Any],
    effect_receipt: dict[str, Any],
) -> dict[str, Any]:
    """Atomically persist a frontier decision before any lifecycle write."""

    payload = seal_semantic_artifact(
        {
            "schema_version": 3,
            "lane": lane,
            "key": key,
            "input_hash": input_hash,
            "page_hashes": dict(sorted(page_hashes.items())),
            "review": review,
            "effect_receipt": effect_receipt,
        },
        authority=authority,
        lane=lane,
    )
    path = _frontier_approval_path(store, lane=lane, key=key)
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = (
        json.dumps(
            payload,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            default=str,
        )
        + "\n"
    ).encode("utf-8")
    tmp: Path | None = None
    try:
        tmp = _write_unique_temp(
            path,
            encoded,
            token=hashlib.sha256(encoded).hexdigest()[:12],
        )
        os.replace(tmp, path)
        # Persist the directory entry as well as the file bytes. Failure is
        # reported as a retry; a lifecycle mutation must never race ahead of
        # an approval whose rename is not durably visible.
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        if tmp is not None:
            tmp.unlink(missing_ok=True)
    return payload


def _load_frontier_approval(
    store: ConvergenceStore,
    *,
    lane: str,
    key: str,
    input_hash: str,
    page_hashes: dict[str, str] | None = None,
) -> dict[str, Any] | None:
    path = _frontier_approval_path(store, lane=lane, key=key)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if (
        not isinstance(payload, dict)
        or payload.get("schema_version") != 3
        or payload.get("lane") != lane
        or payload.get("key") != key
        or payload.get("input_hash") != input_hash
        or not isinstance(payload.get("review"), dict)
        or _effect_receipt_shape_error(payload.get("effect_receipt")) is not None
    ):
        return None
    stored_hashes = payload.get("page_hashes")
    if not isinstance(stored_hashes, dict) or any(
        not isinstance(page_id, str) or not isinstance(value, str)
        for page_id, value in stored_hashes.items()
    ):
        return None
    if page_hashes is not None and stored_hashes != dict(sorted(page_hashes.items())):
        return None
    if semantic_authority_shape_error(payload.get("authority"), lane=lane) is not None:
        return None
    return payload


def _trusted_approval_review(
    approval: dict[str, Any] | None,
    *,
    lane: str,
    current_authority: dict[str, Any],
    allowed_decisions: set[str],
) -> tuple[dict[str, Any] | None, str | None]:
    review = approval.get("review") if isinstance(approval, dict) else None
    if not isinstance(review, dict) or review.get("schema_valid") is not True:
        return None, None
    decision = review.get("decision")
    confidence = review.get("confidence")
    if (
        decision not in allowed_decisions
        or isinstance(confidence, bool)
        or not isinstance(confidence, (int, float))
        or not math.isfinite(float(confidence))
        or not 0.0 <= float(confidence) <= 1.0
    ):
        return None, None
    artifact_authority = (
        approval.get("authority") if isinstance(approval, dict) else None
    )
    authority_error = compare_semantic_authority(
        artifact_authority,
        current_authority,
        lane=lane,
    ) or semantic_verdict_authority_error(review, artifact_authority, lane=lane)
    if authority_error is not None:
        return None, authority_error
    return review, None


def _duplicate_effect_receipt_error(
    receipt: object,
    *,
    decision: str,
    left: str,
    right: str,
    page_hashes: dict[str, str],
) -> str | None:
    error = _effect_receipt_shape_error(receipt)
    if error is not None:
        return error
    assert isinstance(receipt, dict)
    if receipt.get("decision") != decision:
        return "duplicate effect receipt decision mismatch"
    preimages = receipt["preimages"]
    if set(preimages) != {left, right} or any(
        preimages[page_id].get("sha256") != page_hashes.get(page_id)
        for page_id in (left, right)
    ):
        return "duplicate effect receipt preimage mismatch"
    if decision == "keep_both":
        if (
            receipt.get("operation") != "keep_both"
            or receipt.get("targets") != {"left": left, "right": right}
            or receipt.get("postimages") != {}
        ):
            return "duplicate keep receipt mismatch"
        return None
    expected_loser, expected_winner = (
        (left, right) if decision == "supersede_left" else (right, left)
    )
    if (
        receipt.get("operation") != "soft_supersede"
        or receipt.get("targets")
        != {"loser": expected_loser, "winner": expected_winner}
        or set(receipt.get("postimages", {})) != {expected_loser}
    ):
        return "duplicate supersede receipt mismatch"
    return None


def _retention_effect_receipt_error(
    receipt: object,
    *,
    decision: str,
    page_id: str,
    page_hash: str,
) -> str | None:
    error = _effect_receipt_shape_error(receipt)
    if error is not None:
        return error
    assert isinstance(receipt, dict)
    if (
        receipt.get("decision") != decision
        or receipt.get("targets") != {"page_id": page_id}
        or set(receipt.get("preimages", {})) != {page_id}
        or receipt["preimages"][page_id].get("sha256") != page_hash
    ):
        return "retention effect receipt identity mismatch"
    if decision == "keep_active":
        if receipt.get("operation") != "keep_active" or receipt.get("postimages") != {}:
            return "retention keep receipt mismatch"
        return None
    if receipt.get("operation") != "soft_archive" or set(
        receipt.get("postimages", {})
    ) != {page_id}:
        return "retention archive receipt mismatch"
    return None


def _current_pages_match_postimages(receipt: dict[str, Any]) -> bool:
    """Require every mutated page to equal its durable reviewed postimage."""

    postimages = receipt.get("postimages")
    if not isinstance(postimages, dict) or not postimages:
        return False
    for page_id, expected in postimages.items():
        path = find_page(page_id)
        if path is None:
            return False
        try:
            current = path.read_bytes()
        except OSError:
            return False
        if _bytes_receipt(current) != expected:
            return False
    return True


def _finalize_frontier_receipt(
    store: ConvergenceStore,
    *,
    lane: str,
    key: str,
    expected_decision: str,
    receipt: dict[str, Any],
    now: datetime | None,
) -> dict[str, Any] | None:
    """Complete convergence after a crash left an approved page write behind."""

    item = store.get(key)
    if not isinstance(item, dict) or item.get("lane") != lane:
        return None
    artifact = _load_frontier_approval(
        store,
        lane=lane,
        key=key,
        input_hash=str(item.get("input_hash") or ""),
    )
    review = artifact.get("review") if isinstance(artifact, dict) else None
    artifact_authority = (
        artifact.get("authority") if isinstance(artifact, dict) else None
    )
    artifact_hashes = (
        artifact.get("page_hashes") if isinstance(artifact, dict) else None
    )
    effect_receipt = (
        artifact.get("effect_receipt") if isinstance(artifact, dict) else None
    )
    confidence = review.get("confidence") if isinstance(review, dict) else None
    if (
        not isinstance(review, dict)
        or not isinstance(artifact_hashes, dict)
        or review.get("schema_valid") is not True
        or review.get("decision") != expected_decision
        or isinstance(confidence, bool)
        or not isinstance(confidence, (int, float))
        or not math.isfinite(float(confidence))
        or not 0.0 <= float(confidence) <= 1.0
        or semantic_verdict_authority_error(
            review,
            artifact_authority,
            lane=lane,
        )
        is not None
        or _effect_receipt_shape_error(effect_receipt) is not None
    ):
        return None
    assert isinstance(effect_receipt, dict)
    if lane == DUPLICATE_FRONTIER_LANE:
        receipt_pages = {
            str(receipt.get("loser") or ""),
            str(receipt.get("winner") or ""),
        }
        effect_targets = effect_receipt.get("targets")
        if (
            "" in receipt_pages
            or receipt_pages != set(artifact_hashes)
            or effect_targets
            != {
                "loser": str(receipt.get("loser") or ""),
                "winner": str(receipt.get("winner") or ""),
            }
            or effect_receipt.get("decision") != expected_decision
            or effect_receipt.get("operation") != "soft_supersede"
        ):
            return None
    elif lane == RETENTION_FRONTIER_LANE:
        page_id = str(receipt.get("page_id") or "")
        if (
            set(artifact_hashes) != {page_id}
            or effect_receipt.get("targets") != {"page_id": page_id}
            or effect_receipt.get("decision") != expected_decision
            or effect_receipt.get("operation") != "soft_archive"
        ):
            return None
    if not _current_pages_match_postimages(effect_receipt):
        return None
    status = str(item.get("status") or "")
    if status in {"applied", "rejected", "quarantined", "human_required"}:
        return item
    return store.complete(
        key,
        "applied",
        result={
            "frontier": review,
            "apply": receipt,
            "recovered": True,
            "recovery_only": True,
            "semantic_effect": False,
        },
        owner=(
            str(item.get("lease_owner"))
            if isinstance(item.get("lease_owner"), str)
            else None
        ),
        now=now,
    )["item"]


def _patch_page_status(
    page_id: str,
    updates: dict[str, Any],
    *,
    expected_hash: str | None = None,
    effect_receipt: dict[str, Any] | None = None,
    correction_store: ConvergenceStore | None = None,
) -> dict[str, Any]:
    path = find_page(page_id)
    if path is None:
        return {"status": "skipped", "reason": "page_not_found", "page_id": page_id}
    try:
        original = path.read_bytes()
        text = original.decode("utf-8")
    except (OSError, UnicodeDecodeError) as exc:
        return {"status": "skipped", "reason": f"read_error: {exc}", "page_id": page_id}
    observed_hash = hashlib.sha256(original).hexdigest()
    if expected_hash is not None and observed_hash != expected_hash:
        return {
            "status": "retry",
            "reason": "page_changed_before_apply",
            "page_id": page_id,
        }
    if effect_receipt is not None and (
        _effect_receipt_shape_error(effect_receipt) is not None
        or effect_receipt.get("targets") != {"page_id": page_id}
        or effect_receipt.get("preimages", {}).get(page_id)
        != _bytes_receipt(original)
    ):
        return {
            "status": "retry",
            "reason": "effect_receipt_preimage_mismatch",
            "page_id": page_id,
        }
    new_text = patch_frontmatter(text, updates)
    if new_text == text:
        return {"status": "unchanged", "page_id": page_id, "path": str(path)}
    written = new_text.encode("utf-8")
    if effect_receipt is not None and effect_receipt.get("postimages") != {
        page_id: _bytes_receipt(written)
    }:
        return {
            "status": "retry",
            "reason": "effect_receipt_postimage_mismatch",
            "page_id": page_id,
        }
    tmp: Path | None = None
    try:
        tmp = _write_unique_temp(path, written, token=observed_hash[:12])
        with _lifecycle_mutation_guard(
            [page_id],
            page_path=path,
            correction_store=correction_store,
        ) as guard:
            if not guard["allowed"]:
                return {
                    "status": "retry",
                    "reason": guard["reason"],
                    "page_id": page_id,
                    "correction_keys": guard["correction_keys"],
                }
            if path.read_bytes() != original:
                return {
                    "status": "retry",
                    "reason": "page_changed_before_replace",
                    "page_id": page_id,
                }
            os.replace(tmp, path)
            if path.read_bytes() != written:
                return {
                    "status": "retry",
                    "reason": "post_write_verification_failed",
                    "page_id": page_id,
                }
    except (OSError, ConvergenceStateError) as exc:
        return {"status": "retry", "reason": f"write_error: {exc}", "page_id": page_id}
    finally:
        if tmp is not None:
            tmp.unlink(missing_ok=True)
    return {
        "status": "applied",
        "page_id": page_id,
        "path": str(path),
        "updates": updates,
        **({"effect_receipt": effect_receipt} if effect_receipt is not None else {}),
    }


def decide_duplicate(record: dict[str, Any]) -> dict[str, Any]:
    """Return a reversible autonomous decision for one duplicate candidate."""
    left = str(record.get("left") or "")
    right = str(record.get("right") or "")
    score = float(record.get("score") or 0.0)
    left_title = str(record.get("left_title") or "")
    right_title = str(record.get("right_title") or "")
    decision = {
        "type": "duplicate_decision",
        "ts": _now(),
        "left": left,
        "right": right,
        "score": score,
        "method": record.get("method"),
        "action": "defer",
        "apply": False,
        "reason": "insufficient_confidence",
    }
    if not left or not right or left == right:
        decision["reason"] = "invalid_pair"
        return decision
    if _norm(left_title) != _norm(right_title):
        decision["reason"] = "title_mismatch"
        return decision
    if score < 0.995:
        decision["reason"] = "score_below_auto_supersede_threshold"
        return decision
    left_meta = _page_meta(left)
    right_meta = _page_meta(right)
    if not left_meta or not right_meta:
        decision["reason"] = "missing_page"
        return decision
    if (
        left_meta.get("sensitivity") == "high"
        or right_meta.get("sensitivity") == "high"
    ):
        decision["reason"] = "high_sensitivity_page"
        return decision
    left_q = _page_quality(left, left_meta)
    right_q = _page_quality(right, right_meta)
    if abs(left_q - right_q) < 1.0:
        decision["reason"] = "quality_tie"
        decision["quality"] = {"left": round(left_q, 3), "right": round(right_q, 3)}
        return decision
    winner, loser = (left, right) if left_q > right_q else (right, left)
    decision.update(
        {
            "action": "supersede",
            "apply": True,
            "winner": winner,
            "loser": loser,
            "reason": "exact_title_high_score_quality_gap",
            "quality": {"left": round(left_q, 3), "right": round(right_q, 3)},
        }
    )
    return decision


def resolve_duplicate_candidates(
    records: list[dict[str, Any]],
    *,
    apply: bool = True,
    write: bool = True,
    budget: CycleBudget | None = None,
    correction_store: ConvergenceStore | None = None,
) -> dict[str, Any]:
    """Record deterministic duplicate proposals without mutating page state.

    Exact-title and quality heuristics are useful routing evidence, but they
    are not a semantic authority.  The frontier convergence lane is the only
    caller allowed to turn such a proposal into a lifecycle mutation.
    """

    decisions: list[dict[str, Any]] = []
    applied = 0
    deferred = 0
    for record in records:
        decision = decide_duplicate(record)
        if decision.get("apply"):
            decision["proposal"] = {
                "action": "supersede",
                "winner": decision.get("winner"),
                "loser": decision.get("loser"),
                "reason": decision.get("reason"),
            }
            decision["apply"] = False
            decision["action"] = "defer"
            decision["reason"] = "frontier_approval_required"
            decision["result"] = {
                "status": "pending_frontier",
                "reason": "deterministic_heuristic_is_proposal_only",
            }
            deferred += 1
        else:
            deferred += 1
        decisions.append(decision)
        if write:
            _append_jsonl(DECISIONS_FILE, decision)
    payload = {
        "status": "ok",
        "candidates": len(records),
        "applied": applied,
        "deferred": deferred,
        "decisions": decisions[:20],
    }
    if budget is not None:
        payload["budget"] = budget.snapshot()
    return payload


def _canonical_duplicate_record(record: dict[str, Any]) -> dict[str, Any] | None:
    original_left = str(record.get("left") or "").strip()
    original_right = str(record.get("right") or "").strip()
    if not original_left or not original_right or original_left == original_right:
        return None
    left, right = sorted((original_left, original_right))
    title_by_page = {
        original_left: str(record.get("left_title") or ""),
        original_right: str(record.get("right_title") or ""),
    }
    try:
        score = float(record.get("score") or 0.0)
    except (TypeError, ValueError):
        score = 0.0
    return {
        "left": left,
        "right": right,
        "left_title": title_by_page.get(left, ""),
        "right_title": title_by_page.get(right, ""),
        "score": score,
        "method": str(record.get("method") or ""),
    }


def _duplicate_page_snapshot(
    page_id: str, *, excerpt_chars: int = 5000
) -> dict[str, Any]:
    path = find_page(page_id)
    if path is None:
        return {
            "page_id": page_id,
            "status": "missing",
            "content_hash": "missing",
            "path": None,
            "meta": {},
            "excerpt": "",
        }
    try:
        raw = path.read_bytes()
        text = raw.decode("utf-8")
    except (OSError, UnicodeDecodeError) as exc:
        return {
            "page_id": page_id,
            "status": "unreadable",
            "content_hash": "unreadable",
            "path": str(path),
            "meta": {},
            "excerpt": "",
            "error": str(exc),
        }
    meta, body = parse_frontmatter(text)
    excerpt = re.sub(r"\s+", " ", body).strip()[:excerpt_chars]
    return {
        "page_id": page_id,
        "status": "ok",
        "content_hash": hashlib.sha256(raw).hexdigest(),
        "path": str(path),
        "meta": meta,
        "excerpt": excerpt,
    }


def _existing_duplicate_resolution(
    left: dict[str, Any],
    right: dict[str, Any],
) -> dict[str, Any] | None:
    left_meta = left.get("meta") if isinstance(left.get("meta"), dict) else {}
    right_meta = right.get("meta") if isinstance(right.get("meta"), dict) else {}
    if (
        left_meta.get("status") == "deprecated"
        and left_meta.get("superseded_by") == right["page_id"]
    ):
        return {
            "decision": "supersede_left",
            "winner": right["page_id"],
            "loser": left["page_id"],
            "approval_key": left_meta.get("frontier_approval_key"),
        }
    if (
        right_meta.get("status") == "deprecated"
        and right_meta.get("superseded_by") == left["page_id"]
    ):
        return {
            "decision": "supersede_right",
            "winner": left["page_id"],
            "loser": right["page_id"],
            "approval_key": right_meta.get("frontier_approval_key"),
        }
    return None


def _rollback_owned_page_write_locked(
    path: Path,
    *,
    expected_written: bytes,
    original: bytes,
) -> bool:
    """Restore ``original`` while holding the shared Wiki mutation lock."""
    rollback: Path | None = None
    try:
        if path.read_bytes() != expected_written:
            return False
        rollback = path.with_name(f".{path.name}.{os.getpid()}.rollback.tmp")
        rollback.write_bytes(original)
        if path.read_bytes() != expected_written:
            return False
        os.replace(rollback, path)
        return path.read_bytes() == original
    except OSError:
        return False
    finally:
        if rollback is not None:
            with suppress(OSError):
                rollback.unlink()


def _soft_supersede_page(
    *,
    loser: str,
    winner: str,
    expected_loser_hash: str,
    expected_winner_hash: str,
    decision_at: str,
    autonomy_decision: str = "duplicate_frontier_supersede",
    frontier_approval_key: str | None = None,
    effect_receipt: dict[str, Any] | None = None,
    correction_store: ConvergenceStore | None = None,
) -> dict[str, Any]:
    """Soft-supersede one page with content CAS; never delete or merge bodies."""
    loser_path = find_page(loser)
    winner_path = find_page(winner)
    if loser_path is None or winner_path is None:
        return {"status": "retry", "reason": "winner_or_loser_missing"}
    try:
        loser_raw = loser_path.read_bytes()
        winner_raw = winner_path.read_bytes()
        loser_text = loser_raw.decode("utf-8")
        winner_text = winner_raw.decode("utf-8")
    except (OSError, UnicodeDecodeError) as exc:
        return {"status": "retry", "reason": f"read_error:{exc}"}
    if hashlib.sha256(loser_raw).hexdigest() != expected_loser_hash:
        return {"status": "retry", "reason": "loser_content_changed"}
    if hashlib.sha256(winner_raw).hexdigest() != expected_winner_hash:
        return {"status": "retry", "reason": "winner_content_changed"}
    if effect_receipt is not None and (
        _effect_receipt_shape_error(effect_receipt) is not None
        or effect_receipt.get("operation") != "soft_supersede"
        or effect_receipt.get("targets") != {"loser": loser, "winner": winner}
        or effect_receipt.get("decision_at") != decision_at
        or effect_receipt.get("preimages")
        != {
            loser: _bytes_receipt(loser_raw),
            winner: _bytes_receipt(winner_raw),
        }
    ):
        return {"status": "retry", "reason": "effect_receipt_preimage_mismatch"}

    loser_meta, loser_body = parse_frontmatter(loser_text)
    winner_meta, _winner_body = parse_frontmatter(winner_text)
    if loser_meta.get("status") == "deprecated":
        if loser_meta.get("superseded_by") == winner:
            return {"status": "already_applied", "loser": loser, "winner": winner}
        return {"status": "retry", "reason": "loser_already_superseded_elsewhere"}
    if winner_meta.get("status") in {"deprecated", "archived"}:
        return {"status": "retry", "reason": "winner_is_not_active"}

    updates = {
        "status": "deprecated",
        "superseded_by": winner,
        "autonomy_decision": autonomy_decision,
        "autonomy_decision_at": decision_at,
    }
    if frontier_approval_key:
        updates["frontier_approval_key"] = frontier_approval_key
    updated = patch_frontmatter(loser_text, updates)
    _updated_meta, updated_body = parse_frontmatter(updated)
    if updated_body != loser_body:
        return {"status": "retry", "reason": "body_change_refused"}
    updated_raw = updated.encode("utf-8")
    if effect_receipt is not None and effect_receipt.get("postimages") != {
        loser: _bytes_receipt(updated_raw)
    }:
        return {"status": "retry", "reason": "effect_receipt_postimage_mismatch"}
    tmp: Path | None = None
    replaced = False
    try:
        with _lifecycle_mutation_guard(
            [loser],
            page_path=loser_path,
            correction_store=correction_store,
        ) as guard:
            if not guard["allowed"]:
                return {
                    "status": "retry",
                    "reason": guard["reason"],
                    "loser": loser,
                    "winner": winner,
                    "correction_keys": guard["correction_keys"],
                }
            try:
                if (
                    loser_path.read_bytes() != loser_raw
                    or winner_path.read_bytes() != winner_raw
                ):
                    return {"status": "retry", "reason": "content_changed_before_apply"}
                tmp = _write_unique_temp(
                    loser_path,
                    updated_raw,
                    token=expected_loser_hash[:12],
                )
                # Keep the CAS immediately adjacent to the replace.  The
                # earlier snapshot check alone is insufficient when a content
                # correction lands while this temporary file is prepared.
                if (
                    loser_path.read_bytes() != loser_raw
                    or winner_path.read_bytes() != winner_raw
                ):
                    return {
                        "status": "retry",
                        "reason": "content_changed_before_replace",
                    }
                os.replace(tmp, loser_path)
                replaced = True
                written_raw = loser_path.read_bytes()
                winner_after = winner_path.read_bytes()
            except OSError as exc:
                rolled_back = (
                    _rollback_owned_page_write_locked(
                        loser_path,
                        expected_written=updated_raw,
                        original=loser_raw,
                    )
                    if replaced
                    else False
                )
                return {
                    "status": "retry",
                    "reason": f"write_error:{exc}",
                    "rolled_back": rolled_back,
                }
            try:
                written = written_raw.decode("utf-8")
                written_meta, written_body = parse_frontmatter(written)
            except (UnicodeDecodeError, ValueError) as exc:
                rolled_back = _rollback_owned_page_write_locked(
                    loser_path,
                    expected_written=updated_raw,
                    original=loser_raw,
                )
                return {
                    "status": "retry",
                    "reason": f"post_write_parse_failed:{exc}",
                    "rolled_back": rolled_back,
                }
            verified = (
                written_raw == updated_raw
                and written_body == loser_body
                and written_meta.get("status") == "deprecated"
                and written_meta.get("superseded_by") == winner
                and written_meta.get("autonomy_decision") == autonomy_decision
                and written_meta.get("autonomy_decision_at") == decision_at
                and (
                    not frontier_approval_key
                    or written_meta.get("frontier_approval_key")
                    == frontier_approval_key
                )
                and hashlib.sha256(winner_after).hexdigest() == expected_winner_hash
            )
            if not verified:
                rolled_back = _rollback_owned_page_write_locked(
                    loser_path,
                    expected_written=updated_raw,
                    original=loser_raw,
                )
                return {
                    "status": "retry",
                    "reason": "post_write_verification_failed",
                    "rolled_back": rolled_back,
                }
    except (OSError, ConvergenceStateError) as exc:
        return {"status": "retry", "reason": f"write_error:{exc}", "rolled_back": False}
    finally:
        if tmp is not None:
            with suppress(OSError):
                tmp.unlink()
    return {
        "status": "applied",
        "loser": loser,
        "winner": winner,
        "path": str(loser_path),
        **({"effect_receipt": effect_receipt} if effect_receipt is not None else {}),
    }


def _duplicate_frontier_failure_class(review: dict[str, Any]) -> str | None:
    failure = review.get("frontier_failure")
    if isinstance(failure, dict) and isinstance(failure.get("failure_class"), str):
        return failure["failure_class"]
    return None


def _normalize_duplicate_frontier_review(review: object) -> dict[str, Any]:
    def invalid(summary: str) -> dict[str, Any]:
        normalized = {
            "decision": "needs_retry",
            "confidence": 0.0,
            "summary": summary,
            "schema_valid": False,
            "raw_review": review,
        }
        if isinstance(review, dict):
            failure = review.get("frontier_failure")
            if isinstance(failure, dict):
                normalized["frontier_failure"] = failure
            normalized["human_required"] = is_human_required_result(review)
        return normalized

    if not isinstance(review, dict):
        return invalid("frontier result is not an object")
    required = set(DUPLICATE_FRONTIER_SCHEMA["required"])
    missing = sorted(required - set(review))
    if missing:
        return invalid(
            f"frontier result is missing required fields: {', '.join(missing)}"
        )
    decision = review.get("decision")
    summary = review.get("summary")
    confidence = review.get("confidence")
    valid_decisions = set(DUPLICATE_FRONTIER_SCHEMA["properties"]["decision"]["enum"])
    if not isinstance(decision, str) or decision not in valid_decisions:
        return invalid("frontier result has an invalid decision")
    if not isinstance(summary, str):
        return invalid("frontier result is missing a string summary")
    if (
        isinstance(confidence, bool)
        or not isinstance(confidence, (int, float))
        or not math.isfinite(float(confidence))
        or not 0.0 <= float(confidence) <= 1.0
    ):
        return invalid("frontier result confidence is not a finite number in [0, 1]")

    schema_fields = set(DUPLICATE_FRONTIER_SCHEMA["properties"])
    diagnostic_fields = {"frontier_failure", "human_required"}
    authority_fields = {"decision_policy", "local_consensus"}
    allowed_fields = (
        schema_fields
        | {"reviewer"}
        | authority_fields
        | (diagnostic_fields if decision == "needs_retry" else set())
    )
    extras = sorted(set(review) - allowed_fields)
    if extras:
        return invalid(f"frontier result has unexpected fields: {', '.join(extras)}")
    normalized = {
        **review,
        "decision": decision,
        "summary": summary,
        "confidence": float(confidence),
        "schema_valid": True,
    }
    if decision == "needs_retry":
        normalized["human_required"] = is_human_required_result(review)
    return normalized


def _review_deferred_duplicate(
    candidate: dict[str, Any],
    *,
    timeout: int | None = None,
) -> dict[str, Any]:
    from chronovisor.decision.decision_lane_prompts import (
        build_autonomy_duplicate_review_prompt,
    )
    from chronovisor.decision.frontier_review import run_structured_review

    prompt = build_autonomy_duplicate_review_prompt(candidate)
    return run_structured_review(
        prompt,
        DUPLICATE_FRONTIER_SCHEMA,
        repo_root=PROJECT_ROOT,
        timeout=timeout,
        execute_patch=False,
        command_env="CHRONOVISOR_DUPLICATE_REVIEW_CMD",
        decision_lane="autonomy_duplicate_resolution",
    )


def resolve_deferred_duplicates_with_frontier(
    records: list[dict[str, Any]],
    *,
    convergence_store: ConvergenceStore | None = None,
    budget: CycleBudget | None = None,
    reviewer: Callable[[dict[str, Any]], dict[str, Any]] | None = None,
    timeout: int | None = None,
    now: datetime | None = None,
    dry_run: bool = False,
    write: bool = True,
    eligible_keys: set[str] | None = None,
    inventory_complete: bool = False,
) -> dict[str, Any]:
    """Boundedly converge deterministic duplicate deferrals via a frontier judge."""
    state = convergence_store or ConvergenceStore()
    cycle_budget = budget or CycleBudget(
        max_local_calls=20,
        max_frontier_calls=3,
        max_mutations=3,
        max_elapsed_seconds=900,
    )
    decision_at = (now or datetime.now().astimezone()).isoformat(timespec="seconds")
    retired_stale = (
        state.retire_stale(
            lane=DUPLICATE_FRONTIER_LANE,
            reason="duplicate_candidate_expired",
            now=now,
            dry_run=dry_run,
        )
        if eligible_keys is None
        else {"status": "skipped", "reason": "targeted_allowlist", "retired": []}
    )
    active_source_ids = {
        f"{candidate['left']}<->{candidate['right']}"
        for record in records
        if (candidate := _canonical_duplicate_record(record)) is not None
    }
    retired_absent = (
        state.retire_absent_sources(
            lane=DUPLICATE_FRONTIER_LANE,
            active_source_ids=active_source_ids,
            reason="duplicate_pair_no_longer_detected",
            now=now,
            dry_run=dry_run,
        )
        if inventory_complete and eligible_keys is None
        else {"status": "skipped", "reason": "partial_inventory", "retired": []}
    )
    frontier_remaining = int(cycle_budget.snapshot()["remaining"]["frontier"])
    seen_keys: set[str] = set()
    results: list[dict[str, Any]] = []
    deferred_seen = 0
    frontier_calls = 0
    applied = 0
    kept_both = 0
    for record in records:
        local_decision = decide_duplicate(record)
        deferred_seen += 1
        candidate = _canonical_duplicate_record(record)
        if candidate is None:
            results.append(
                {"status": "invalid_pair", "reason": local_decision.get("reason")}
            )
            continue
        left_snapshot = _duplicate_page_snapshot(candidate["left"])
        right_snapshot = _duplicate_page_snapshot(candidate["right"])
        input_data = {
            "pair": [candidate["left"], candidate["right"]],
            "content_hashes": {
                candidate["left"]: left_snapshot["content_hash"],
                candidate["right"]: right_snapshot["content_hash"],
            },
        }
        candidate_key = stable_item_key(
            DUPLICATE_FRONTIER_LANE,
            f"{candidate['left']}<->{candidate['right']}",
            input_data,
            resolver_version=DUPLICATE_FRONTIER_RESOLVER_VERSION,
        )
        if eligible_keys is not None and candidate_key not in eligible_keys:
            continue
        existing = _existing_duplicate_resolution(left_snapshot, right_snapshot)
        if existing is not None:
            approval_key = existing.get("approval_key")
            recovered = None
            if (
                isinstance(approval_key, str)
                and approval_key
                and (eligible_keys is None or approval_key in eligible_keys)
            ):
                recovered = _finalize_frontier_receipt(
                    state,
                    lane=DUPLICATE_FRONTIER_LANE,
                    key=approval_key,
                    expected_decision=str(existing["decision"]),
                    receipt={"status": "already_applied", **existing},
                    now=now,
                )
            results.append(
                {
                    "status": "already_applied",
                    "recovery_only": True,
                    "semantic_effect": False,
                    **existing,
                    **(
                        {
                            "convergence_status": recovered.get("status"),
                            "recovery_only": True,
                            "semantic_effect": False,
                        }
                        if isinstance(recovered, dict)
                        else {}
                    ),
                }
            )
            continue
        merged = state.merge_item(
            lane=DUPLICATE_FRONTIER_LANE,
            source_id=f"{candidate['left']}<->{candidate['right']}",
            input_data=input_data,
            resolver_version=DUPLICATE_FRONTIER_RESOLVER_VERSION,
            metadata={
                "candidate": candidate,
                "local_action": local_decision.get("action"),
                "local_reason": local_decision.get("reason"),
                "local_proposal": local_decision,
            },
            now=now,
            dry_run=dry_run,
            supersede_eligible_keys=eligible_keys,
        )
        item = merged["item"]
        if item is None:
            results.append(
                {
                    "status": "out_of_scope_source_changed",
                    "key": candidate_key,
                    "blocked_by_out_of_scope": merged.get(
                        "blocked_by_out_of_scope", []
                    ),
                }
            )
            continue
        key = str(item["key"])
        if key in seen_keys:
            results.append({"status": "duplicate_in_cycle", "key": key})
            continue
        seen_keys.add(key)
        result: dict[str, Any] = {"key": key, "pair": input_data["pair"]}
        item_result = (
            item.get("result") if isinstance(item.get("result"), Mapping) else {}
        )
        semantic_terminal = bool(
            item.get("status") == "quarantined"
            and (
                str(item.get("last_failure_class") or "") == LOCAL_SEMANTIC_NO_QUORUM
                or persisted_semantic_no_quorum_hold(
                    item,
                    lane=DUPLICATE_FRONTIER_LANE,
                )
                is not None
                or isinstance(item_result.get("legacy_semantic_hold"), Mapping)
            )
        )
        if semantic_terminal:
            current_authority, authority_error = _current_autonomy_authority(
                DUPLICATE_FRONTIER_LANE,
                reviewer=reviewer,
            )
            if authority_error is not None or current_authority is None:
                results.append(
                    {
                        **result,
                        "status": "quarantined",
                        "cached": True,
                        "semantic_deferred": True,
                        "reason": authority_error or "decision_authority_unavailable",
                    }
                )
                continue
            semantic_epoch = _duplicate_semantic_epoch(
                item,
                candidate=candidate,
                local_decision=local_decision,
                page_hashes=input_data["content_hashes"],
            )
            try:
                item, held = _resume_or_migrate_autonomy_semantic_hold(
                    state,
                    item,
                    lane=DUPLICATE_FRONTIER_LANE,
                    epoch=semantic_epoch,
                    authority=current_authority,
                    now=now,
                    dry_run=dry_run,
                )
            except (TypeError, ValueError, InvalidTransition) as exc:
                results.append(
                    {
                        **result,
                        "status": "quarantined",
                        "cached": True,
                        "semantic_deferred": True,
                        "reason": f"semantic hold validation failed: {exc}",
                    }
                )
                continue
            if held:
                results.append(
                    {
                        **result,
                        "status": "quarantined",
                        "cached": True,
                        "semantic_deferred": True,
                    }
                )
                continue
        if item.get("status") in {
            "applied",
            "rejected",
            "quarantined",
            "human_required",
        }:
            results.append({**result, "status": item.get("status"), "cached": True})
            continue
        snapshots_ok = (
            left_snapshot["status"] == "ok" and right_snapshot["status"] == "ok"
        )
        if dry_run:
            if not snapshots_ok:
                results.append(
                    {
                        **result,
                        "status": "would_retry",
                        "reason": "page_snapshot_unavailable",
                    }
                )
                continue
            if item.get("status") in {
                "pending_frontier",
                "frontier_retry",
                "frontier_running",
            }:
                projected = state.claim_attempt(key, "frontier", now=now, dry_run=True)
                if not projected["claimed"]:
                    results.append({**result, "status": projected["reason"]})
                    continue
            if frontier_remaining <= 0 or not cycle_budget.can_consume("frontier")[0]:
                results.append({**result, "status": "frontier_budget_exhausted"})
                continue
            frontier_remaining -= 1
            results.append({**result, "status": "would_review"})
            continue

        if item.get("status") in {"pending_local", "local_retry"}:
            escalated = state.escalate(
                key,
                reason=str(
                    local_decision.get("reason") or "deterministic duplicate defer"
                ),
                now=now,
            )
            item = escalated["item"]
        if item.get("status") not in {
            "pending_frontier",
            "frontier_retry",
            "frontier_running",
        }:
            results.append(
                {**result, "status": item.get("status") or "not_frontier_pending"}
            )
            continue

        if snapshots_ok:
            claim = state.claim_attempt(key, "frontier", budget=cycle_budget, now=now)
        else:
            allowed, reason = cycle_budget.consume("local")
            if not allowed:
                results.append({**result, "status": reason})
                continue
            claim = state.claim_attempt(key, "frontier", now=now)
        if not claim["claimed"]:
            results.append(
                {
                    **result,
                    "status": claim["item"].get("status"),
                    "reason": claim["reason"],
                }
            )
            continue
        owner = claim["owner"]
        if not snapshots_ok:
            transition = state.fail_attempt(
                key,
                "frontier",
                error="page snapshot unavailable",
                owner=owner,
                now=now,
            )
            results.append({**result, "status": transition["item"]["status"]})
            continue

        page_hashes = dict(input_data["content_hashes"])
        authority, authority_error = _current_autonomy_authority(
            DUPLICATE_FRONTIER_LANE,
            reviewer=reviewer,
        )
        if authority_error is not None or authority is None:
            transition = state.fail_attempt(
                key,
                "frontier",
                error=authority_error or "duplicate decision authority is missing",
                failure_class="review_artifact_invalid",
                owner=owner,
                now=now,
            )
            results.append(
                {
                    **result,
                    "status": transition["item"]["status"],
                    "reason": authority_error or "decision_authority_missing",
                }
            )
            continue
        restored_hold = state.restore_semantic_no_quorum_hold(
            key,
            lane=DUPLICATE_FRONTIER_LANE,
            epoch=_duplicate_semantic_epoch(
                item,
                candidate=candidate,
                local_decision=local_decision,
                page_hashes=page_hashes,
            ),
            authority=authority,
            owner=owner,
            now=now,
        )
        if restored_hold is not None:
            results.append(
                {
                    **result,
                    "status": "quarantined",
                    "cached": True,
                    "semantic_deferred": True,
                    "restored_semantic_hold": True,
                }
            )
            continue
        approval = _load_frontier_approval(
            state,
            lane=DUPLICATE_FRONTIER_LANE,
            key=key,
            input_hash=str(item.get("input_hash") or ""),
            page_hashes=page_hashes,
        )
        review, approval_authority_error = _trusted_approval_review(
            approval,
            lane=DUPLICATE_FRONTIER_LANE,
            current_authority=authority,
            allowed_decisions={"supersede_left", "supersede_right", "keep_both"},
        )
        if approval_authority_error is not None:
            transition = state.fail_attempt(
                key,
                "frontier",
                error=approval_authority_error,
                failure_class="review_artifact_invalid",
                owner=owner,
                now=now,
            )
            results.append(
                {
                    **result,
                    "status": transition["item"]["status"],
                    "reason": approval_authority_error,
                }
            )
            continue
        if isinstance(review, dict) and isinstance(approval, dict):
            receipt_error = _duplicate_effect_receipt_error(
                approval.get("effect_receipt"),
                decision=str(review.get("decision") or ""),
                left=str(candidate["left"]),
                right=str(candidate["right"]),
                page_hashes=page_hashes,
            )
            if receipt_error is not None:
                transition = state.fail_attempt(
                    key,
                    "frontier",
                    error=receipt_error,
                    failure_class="review_artifact_invalid",
                    owner=owner,
                    now=now,
                )
                results.append(
                    {
                        **result,
                        "status": transition["item"]["status"],
                        "reason": receipt_error,
                    }
                )
                continue
        if review is None:
            approval = None
        if not isinstance(review, dict):
            frontier_calls += 1
            review_candidate = {
                **candidate,
                "left_content_hash": left_snapshot["content_hash"],
                "right_content_hash": right_snapshot["content_hash"],
                "left_meta": left_snapshot["meta"],
                "right_meta": right_snapshot["meta"],
                "left_excerpt": left_snapshot["excerpt"],
                "right_excerpt": right_snapshot["excerpt"],
                "local_reason": local_decision.get("reason"),
                "local_proposal": local_decision,
            }
            try:
                raw_review = (
                    reviewer(review_candidate)
                    if reviewer is not None
                    else _review_deferred_duplicate(review_candidate, timeout=timeout)
                )
            except Exception as exc:
                from chronovisor.decision.frontier_review import (
                    classify_frontier_failure,
                )

                failure = classify_frontier_failure(str(exc)).to_dict()
                raw_review = {
                    "decision": "needs_retry",
                    "confidence": 0.0,
                    "summary": str(exc),
                    "frontier_failure": failure,
                }
            review = _normalize_duplicate_frontier_review(raw_review)
        frontier_decision = str(review.get("decision"))
        if frontier_decision != "needs_retry" and approval is None:
            try:
                with decision_authority_lock():
                    epoch_error = _autonomy_authority_epoch_error(
                        authority,
                        review,
                        lane=DUPLICATE_FRONTIER_LANE,
                        reviewer=reviewer,
                    )
                    if epoch_error is not None:
                        raise ValueError(epoch_error)
                    effect_receipt, receipt_error = _prepare_duplicate_effect_receipt(
                        decision=frontier_decision,
                        left=str(candidate["left"]),
                        right=str(candidate["right"]),
                        page_hashes=page_hashes,
                        decision_at=decision_at,
                        approval_key=key,
                    )
                    if receipt_error is not None or effect_receipt is None:
                        raise ValueError(
                            receipt_error or "duplicate effect receipt is missing"
                        )
                    _persist_frontier_approval(
                        state,
                        lane=DUPLICATE_FRONTIER_LANE,
                        key=key,
                        input_hash=str(item.get("input_hash") or ""),
                        page_hashes=page_hashes,
                        review=review,
                        authority=authority,
                        effect_receipt=effect_receipt,
                    )
                    approval = _load_frontier_approval(
                        state,
                        lane=DUPLICATE_FRONTIER_LANE,
                        key=key,
                        input_hash=str(item.get("input_hash") or ""),
                        page_hashes=page_hashes,
                    )
                    if (
                        not isinstance(approval, dict)
                        or approval.get("review") != review
                        or approval.get("authority") != authority
                        or approval.get("effect_receipt") != effect_receipt
                    ):
                        raise OSError("frontier approval readback mismatch")
            except (OSError, ValueError) as exc:
                review = {
                    "decision": "needs_retry",
                    "confidence": 0.0,
                    "summary": f"frontier approval persistence failed: {exc}",
                    "schema_valid": False,
                }
                frontier_decision = "needs_retry"
        effect_receipt = (
            approval.get("effect_receipt") if isinstance(approval, dict) else None
        )
        transition: dict[str, Any]
        apply_result: dict[str, Any] | None = None
        if frontier_decision == "needs_retry":
            if is_local_semantic_no_quorum(review):
                try:
                    with decision_authority_lock():
                        current_authority, current_error = _current_autonomy_authority(
                            DUPLICATE_FRONTIER_LANE,
                            reviewer=reviewer,
                        )
                        epoch_error = current_error or compare_semantic_authority(
                            authority,
                            current_authority,
                            lane=DUPLICATE_FRONTIER_LANE,
                        )
                        if epoch_error is not None or current_authority is None:
                            raise ValueError(
                                epoch_error or "decision authority is unavailable"
                            )
                        transition = state.hold_semantic_no_quorum(
                            key,
                            lane=DUPLICATE_FRONTIER_LANE,
                            stage="frontier",
                            review=review,
                            epoch=_duplicate_semantic_epoch(
                                item,
                                candidate=candidate,
                                local_decision=local_decision,
                                page_hashes=page_hashes,
                            ),
                            authority=current_authority,
                            owner=owner,
                            error=str(
                                review.get("summary") or "local semantic no quorum"
                            ),
                            now=now,
                        )
                except (InvalidTransition, TypeError, ValueError) as exc:
                    transition = state.fail_attempt(
                        key,
                        "frontier",
                        error=f"semantic hold rejected: {exc}",
                        failure_class="review_artifact_invalid",
                        owner=owner,
                        now=now,
                    )
            else:
                transition = state.fail_attempt(
                    key,
                    "frontier",
                    error=str(review.get("summary") or "frontier needs retry"),
                    failure_class=_duplicate_frontier_failure_class(review),
                    owner=owner,
                    now=now,
                )
        elif frontier_decision == "keep_both":
            with decision_authority_lock():
                epoch_error = _autonomy_authority_epoch_error(
                    authority,
                    review,
                    lane=DUPLICATE_FRONTIER_LANE,
                    reviewer=reviewer,
                )
                if epoch_error is not None:
                    transition = state.fail_attempt(
                        key,
                        "frontier",
                        error=epoch_error,
                        failure_class="review_artifact_invalid",
                        owner=owner,
                        now=now,
                    )
                else:
                    transition = state.complete(
                        key,
                        "rejected",
                        result={"decision": "keep_both", "frontier": review},
                        owner=owner,
                        now=now,
                    )
                    kept_both += 1
        else:
            allowed, reason = cycle_budget.consume("mutation")
            if not allowed:
                transition = state.fail_attempt(
                    key,
                    "frontier",
                    error=reason,
                    owner=owner,
                    now=now,
                )
            else:
                loser_snapshot = (
                    left_snapshot
                    if frontier_decision == "supersede_left"
                    else right_snapshot
                )
                winner_snapshot = (
                    right_snapshot
                    if frontier_decision == "supersede_left"
                    else left_snapshot
                )
                with decision_authority_lock():
                    epoch_error = _autonomy_authority_epoch_error(
                        authority,
                        review,
                        lane=DUPLICATE_FRONTIER_LANE,
                        reviewer=reviewer,
                    )
                    if epoch_error is not None:
                        apply_result = {"status": "retry", "reason": epoch_error}
                        transition = state.fail_attempt(
                            key,
                            "frontier",
                            error=epoch_error,
                            failure_class="review_artifact_invalid",
                            owner=owner,
                            now=now,
                        )
                    else:
                        assert isinstance(effect_receipt, dict)
                        apply_result = _soft_supersede_page(
                            loser=str(loser_snapshot["page_id"]),
                            winner=str(winner_snapshot["page_id"]),
                            expected_loser_hash=str(loser_snapshot["content_hash"]),
                            expected_winner_hash=str(winner_snapshot["content_hash"]),
                            decision_at=str(effect_receipt["decision_at"]),
                            frontier_approval_key=key,
                            effect_receipt=effect_receipt,
                            correction_store=state,
                        )
                        if apply_result.get("status") in {
                            "applied",
                            "already_applied",
                        }:
                            transition = state.complete(
                                key,
                                "applied",
                                result={"frontier": review, "apply": apply_result},
                                owner=owner,
                                now=now,
                            )
                            applied += 1
                        else:
                            transition = state.fail_attempt(
                                key,
                                "frontier",
                                error=str(
                                    apply_result.get("reason")
                                    or apply_result.get("status")
                                ),
                                owner=owner,
                                now=now,
                            )
        final_status = str(transition["item"]["status"])
        audit = {
            "type": "duplicate_frontier_decision",
            "ts": decision_at,
            "key": key,
            "left": candidate["left"],
            "right": candidate["right"],
            "decision": frontier_decision,
            "review": review,
            "result": apply_result,
            "status": final_status,
        }
        if write:
            _append_jsonl(DECISIONS_FILE, audit)
        results.append(
            {
                **result,
                "status": final_status,
                "decision": frontier_decision,
                "apply": apply_result,
            }
        )

    status_counts: dict[str, int] = {}
    for result in results:
        status = str(result.get("status") or "unknown")
        status_counts[status] = status_counts.get(status, 0) + 1
    return {
        "status": "ok",
        "dry_run": dry_run,
        "candidates": len(records),
        "deferred_seen": deferred_seen,
        "unique_items": len(seen_keys),
        "frontier_calls": frontier_calls,
        "applied": applied,
        "kept_both": kept_both,
        "retired": sorted(
            {
                *retired_stale.get("retired", []),
                *retired_absent.get("retired", []),
            }
        ),
        "retired_absent": retired_absent.get("retired", []),
        "status_counts": dict(sorted(status_counts.items())),
        "budget": cycle_budget.snapshot(),
        "results": results,
    }


def _normalize_retention_frontier_review(review: object) -> dict[str, Any]:
    def invalid(summary: str) -> dict[str, Any]:
        normalized: dict[str, Any] = {
            "decision": "needs_retry",
            "confidence": 0.0,
            "summary": summary,
            "schema_valid": False,
            "raw_review": review,
        }
        if isinstance(review, dict):
            failure = review.get("frontier_failure")
            if isinstance(failure, dict):
                normalized["frontier_failure"] = failure
            normalized["human_required"] = is_human_required_result(review)
        return normalized

    if not isinstance(review, dict):
        return invalid("frontier result is not an object")
    required = set(RETENTION_FRONTIER_SCHEMA["required"])
    missing = sorted(required - set(review))
    if missing:
        return invalid(
            f"frontier result is missing required fields: {', '.join(missing)}"
        )
    decision = review.get("decision")
    confidence = review.get("confidence")
    summary = review.get("summary")
    valid_decisions = set(RETENTION_FRONTIER_SCHEMA["properties"]["decision"]["enum"])
    if not isinstance(decision, str) or decision not in valid_decisions:
        return invalid("frontier result has an invalid decision")
    if not isinstance(summary, str):
        return invalid("frontier result is missing a string summary")
    if (
        isinstance(confidence, bool)
        or not isinstance(confidence, (int, float))
        or not math.isfinite(float(confidence))
        or not 0.0 <= float(confidence) <= 1.0
    ):
        return invalid("frontier result confidence is not a finite number in [0, 1]")
    schema_fields = set(RETENTION_FRONTIER_SCHEMA["properties"])
    diagnostics = {"frontier_failure", "human_required"}
    authority_fields = {"decision_policy", "local_consensus"}
    allowed = (
        schema_fields
        | {"reviewer"}
        | authority_fields
        | (diagnostics if decision == "needs_retry" else set())
    )
    extras = sorted(set(review) - allowed)
    if extras:
        return invalid(f"frontier result has unexpected fields: {', '.join(extras)}")
    normalized = {
        **review,
        "decision": decision,
        "confidence": float(confidence),
        "summary": summary,
        "schema_valid": True,
    }
    if decision == "needs_retry":
        normalized["human_required"] = is_human_required_result(review)
    return normalized


def _review_retention_candidate(
    candidate: dict[str, Any],
    *,
    timeout: int | None = None,
) -> dict[str, Any]:
    from chronovisor.decision.decision_lane_prompts import (
        build_autonomy_retention_review_prompt,
    )
    from chronovisor.decision.frontier_review import run_structured_review

    prompt = build_autonomy_retention_review_prompt(candidate)
    return run_structured_review(
        prompt,
        RETENTION_FRONTIER_SCHEMA,
        repo_root=PROJECT_ROOT,
        timeout=timeout,
        execute_patch=False,
        command_env="CHRONOVISOR_RETENTION_REVIEW_CMD",
        decision_lane="autonomy_retention",
    )


def apply_retention_archives(
    retention_payload: dict[str, Any],
    *,
    apply: bool = True,
    write: bool = True,
    limit: int = 25,
    budget: CycleBudget | None = None,
    correction_store: ConvergenceStore | None = None,
    convergence_store: ConvergenceStore | None = None,
    reviewer: Callable[[dict[str, Any]], dict[str, Any]] | None = None,
    timeout: int | None = None,
    now: datetime | None = None,
    eligible_keys: set[str] | None = None,
) -> dict[str, Any]:
    """Frontier-review retention proposals before reversible soft archival."""
    candidates = retention_payload.get("archive_candidates")
    if not isinstance(candidates, list):
        candidates = []
    pages = retention_payload.get("pages")
    pages = pages if isinstance(pages, dict) else {}
    state = convergence_store or correction_store or ConvergenceStore()
    lifecycle_store = correction_store or state
    cycle_budget = budget or CycleBudget(
        max_local_calls=0,
        max_frontier_calls=max(1, limit),
        max_mutations=max(1, limit),
        max_elapsed_seconds=900,
    )
    dry_run = not apply
    decision_at = (now or datetime.now().astimezone()).isoformat(timespec="seconds")
    retired_stale = (
        state.retire_stale(
            lane=RETENTION_FRONTIER_LANE,
            reason="retention_candidate_expired",
            now=now,
            dry_run=dry_run,
        )
        if eligible_keys is None
        else {"status": "skipped", "reason": "targeted_allowlist", "retired": []}
    )
    decisions: list[dict[str, Any]] = []
    applied = 0
    frontier_calls = 0
    actionable_seen = 0
    seen_keys: set[str] = set()
    for page_id in [str(item) for item in candidates if isinstance(item, str)]:
        row = pages.get(page_id) if isinstance(pages.get(page_id), dict) else {}
        snapshot = _duplicate_page_snapshot(page_id)
        decision: dict[str, Any] = {
            "type": "archive_frontier_decision",
            "ts": decision_at,
            "page_id": page_id,
            "action": "defer",
            "apply": False,
            "score": row.get("score"),
            "reason": "frontier_approval_required",
        }
        if snapshot.get("status") != "ok":
            decision["reason"] = "archive_page_snapshot_unavailable"
            decisions.append(decision)
            if write and not dry_run and eligible_keys is None:
                _append_jsonl(DECISIONS_FILE, decision)
            continue
        input_data = {"page_id": page_id, "content_hash": snapshot["content_hash"]}
        candidate_key = stable_item_key(
            RETENTION_FRONTIER_LANE,
            page_id,
            input_data,
            resolver_version=RETENTION_FRONTIER_RESOLVER_VERSION,
        )
        if eligible_keys is not None and candidate_key not in eligible_keys:
            continue
        snapshot_meta = (
            snapshot.get("meta") if isinstance(snapshot.get("meta"), dict) else {}
        )
        if snapshot_meta.get("status") == "archived":
            approval_key = snapshot_meta.get("frontier_approval_key")
            recovered = None
            if (
                isinstance(approval_key, str)
                and approval_key
                and (eligible_keys is None or approval_key in eligible_keys)
            ):
                recovered = _finalize_frontier_receipt(
                    state,
                    lane=RETENTION_FRONTIER_LANE,
                    key=approval_key,
                    expected_decision="archive",
                    receipt={"status": "already_applied", "page_id": page_id},
                    now=now,
                )
            decision.update(
                {
                    "action": "already_archived",
                    "reason": "retention_archive_already_applied",
                    "result": {"status": "already_applied", "page_id": page_id},
                    "recovery_only": True,
                    "semantic_effect": False,
                    **(
                        {
                            "convergence_status": recovered.get("status"),
                            "recovery_only": True,
                            "semantic_effect": False,
                        }
                        if isinstance(recovered, dict)
                        else {}
                    ),
                }
            )
            decisions.append(decision)
            if write and not dry_run:
                _append_jsonl(DECISIONS_FILE, decision)
            continue
        if actionable_seen >= max(0, limit):
            break
        actionable_seen += 1
        merged = state.merge_item(
            lane=RETENTION_FRONTIER_LANE,
            source_id=page_id,
            input_data=input_data,
            resolver_version=RETENTION_FRONTIER_RESOLVER_VERSION,
            metadata={
                "page_id": page_id,
                "retention": row,
                "local_recommendation": "archive",
            },
            now=now,
            dry_run=dry_run,
            supersede_eligible_keys=eligible_keys,
        )
        item = merged["item"]
        if item is None:
            decision.update(
                {
                    "reason": "out_of_scope_source_changed",
                    "blocked_by_out_of_scope": merged.get(
                        "blocked_by_out_of_scope", []
                    ),
                }
            )
            decisions.append(decision)
            continue
        key = str(item["key"])
        decision["key"] = key
        if key in seen_keys:
            decision["reason"] = "duplicate_in_cycle"
            decisions.append(decision)
            continue
        seen_keys.add(key)
        status = str(item.get("status") or "")
        item_result = (
            item.get("result") if isinstance(item.get("result"), Mapping) else {}
        )
        semantic_terminal = bool(
            status == "quarantined"
            and (
                str(item.get("last_failure_class") or "") == LOCAL_SEMANTIC_NO_QUORUM
                or persisted_semantic_no_quorum_hold(
                    item,
                    lane=RETENTION_FRONTIER_LANE,
                )
                is not None
                or isinstance(item_result.get("legacy_semantic_hold"), Mapping)
            )
        )
        if semantic_terminal:
            current_authority, authority_error = _current_autonomy_authority(
                RETENTION_FRONTIER_LANE,
                reviewer=reviewer,
            )
            if authority_error is not None or current_authority is None:
                decision.update(
                    {
                        "reason": authority_error or "decision_authority_unavailable",
                        "status": "quarantined",
                        "cached": True,
                        "semantic_deferred": True,
                    }
                )
                decisions.append(decision)
                continue
            semantic_epoch = _retention_semantic_epoch(
                item,
                page_id=page_id,
                page_hash=str(snapshot["content_hash"]),
                retention=row,
            )
            try:
                item, held = _resume_or_migrate_autonomy_semantic_hold(
                    state,
                    item,
                    lane=RETENTION_FRONTIER_LANE,
                    epoch=semantic_epoch,
                    authority=current_authority,
                    now=now,
                    dry_run=dry_run,
                )
            except (TypeError, ValueError, InvalidTransition) as exc:
                decision.update(
                    {
                        "reason": f"semantic hold validation failed: {exc}",
                        "status": "quarantined",
                        "cached": True,
                        "semantic_deferred": True,
                    }
                )
                decisions.append(decision)
                continue
            if held:
                decision.update(
                    {
                        "reason": "cached_semantic_no_quorum",
                        "status": "quarantined",
                        "cached": True,
                        "semantic_deferred": True,
                    }
                )
                decisions.append(decision)
                continue
            status = str(item.get("status") or "")
        if status in {"applied", "rejected", "quarantined", "human_required"}:
            decision.update(
                {
                    "action": "archive"
                    if status == "applied"
                    else "keep_active"
                    if status == "rejected"
                    else "defer",
                    "reason": f"cached_{status}",
                    "status": status,
                    "cached": True,
                }
            )
            decisions.append(decision)
            continue
        if dry_run:
            frontier_allowed, frontier_reason = cycle_budget.can_consume("frontier")
            decision["reason"] = "would_review" if frontier_allowed else frontier_reason
            decision["status"] = decision["reason"]
            decisions.append(decision)
            continue
        if status in {"pending_local", "local_retry"}:
            item = state.escalate(
                key,
                reason="retention heuristic requires frontier approval",
                now=now,
            )["item"]
        claim = state.claim_attempt(key, "frontier", budget=cycle_budget, now=now)
        if not claim.get("claimed"):
            claimed_item = (
                claim.get("item") if isinstance(claim.get("item"), dict) else item
            )
            decision.update(
                {
                    "reason": str(claim.get("reason") or "frontier_not_claimed"),
                    "status": claimed_item.get("status"),
                }
            )
            decisions.append(decision)
            continue
        owner = claim.get("owner")
        page_hashes = {page_id: str(snapshot["content_hash"])}
        authority, authority_error = _current_autonomy_authority(
            RETENTION_FRONTIER_LANE,
            reviewer=reviewer,
        )
        if authority_error is not None or authority is None:
            transition = state.fail_attempt(
                key,
                "frontier",
                error=authority_error or "retention decision authority is missing",
                failure_class="review_artifact_invalid",
                owner=owner,
                now=now,
            )
            decision.update(
                {
                    "reason": authority_error or "decision_authority_missing",
                    "status": transition["item"]["status"],
                }
            )
            decisions.append(decision)
            continue
        restored_hold = state.restore_semantic_no_quorum_hold(
            key,
            lane=RETENTION_FRONTIER_LANE,
            epoch=_retention_semantic_epoch(
                item,
                page_id=page_id,
                page_hash=str(snapshot["content_hash"]),
                retention=row,
            ),
            authority=authority,
            owner=owner,
            now=now,
        )
        if restored_hold is not None:
            decision.update(
                {
                    "reason": "semantic hold epoch restored before reevaluation",
                    "status": "quarantined",
                    "cached": True,
                    "semantic_deferred": True,
                    "restored_semantic_hold": True,
                }
            )
            decisions.append(decision)
            continue
        approval = _load_frontier_approval(
            state,
            lane=RETENTION_FRONTIER_LANE,
            key=key,
            input_hash=str(item.get("input_hash") or ""),
            page_hashes=page_hashes,
        )
        review, approval_authority_error = _trusted_approval_review(
            approval,
            lane=RETENTION_FRONTIER_LANE,
            current_authority=authority,
            allowed_decisions={"archive", "keep_active"},
        )
        if approval_authority_error is not None:
            transition = state.fail_attempt(
                key,
                "frontier",
                error=approval_authority_error,
                failure_class="review_artifact_invalid",
                owner=owner,
                now=now,
            )
            decision.update(
                {
                    "reason": approval_authority_error,
                    "status": transition["item"]["status"],
                }
            )
            decisions.append(decision)
            continue
        if isinstance(review, dict) and isinstance(approval, dict):
            receipt_error = _retention_effect_receipt_error(
                approval.get("effect_receipt"),
                decision=str(review.get("decision") or ""),
                page_id=page_id,
                page_hash=str(snapshot["content_hash"]),
            )
            if receipt_error is not None:
                transition = state.fail_attempt(
                    key,
                    "frontier",
                    error=receipt_error,
                    failure_class="review_artifact_invalid",
                    owner=owner,
                    now=now,
                )
                decision.update(
                    {
                        "reason": receipt_error,
                        "status": transition["item"]["status"],
                    }
                )
                decisions.append(decision)
                continue
        if review is None:
            approval = None
        if not isinstance(review, dict):
            frontier_calls += 1
            review_candidate = {
                "page_id": page_id,
                "content_hash": snapshot["content_hash"],
                "meta": snapshot["meta"],
                "excerpt": snapshot["excerpt"],
                "retention": row,
                "local_recommendation": "archive",
            }
            try:
                raw_review = (
                    reviewer(review_candidate)
                    if reviewer is not None
                    else _review_retention_candidate(review_candidate, timeout=timeout)
                )
            except Exception as exc:
                from chronovisor.decision.frontier_review import (
                    classify_frontier_failure,
                )

                raw_review = {
                    "decision": "needs_retry",
                    "confidence": 0.0,
                    "summary": str(exc),
                    "frontier_failure": classify_frontier_failure(str(exc)).to_dict(),
                }
            review = _normalize_retention_frontier_review(raw_review)
        frontier_decision = str(review.get("decision") or "needs_retry")
        if frontier_decision != "needs_retry" and approval is None:
            try:
                with decision_authority_lock():
                    epoch_error = _autonomy_authority_epoch_error(
                        authority,
                        review,
                        lane=RETENTION_FRONTIER_LANE,
                        reviewer=reviewer,
                    )
                    if epoch_error is not None:
                        raise ValueError(epoch_error)
                    effect_receipt, receipt_error = _prepare_retention_effect_receipt(
                        decision=frontier_decision,
                        page_id=page_id,
                        page_hash=str(snapshot["content_hash"]),
                        decision_at=decision_at,
                        approval_key=key,
                    )
                    if receipt_error is not None or effect_receipt is None:
                        raise ValueError(
                            receipt_error or "retention effect receipt is missing"
                        )
                    _persist_frontier_approval(
                        state,
                        lane=RETENTION_FRONTIER_LANE,
                        key=key,
                        input_hash=str(item.get("input_hash") or ""),
                        page_hashes=page_hashes,
                        review=review,
                        authority=authority,
                        effect_receipt=effect_receipt,
                    )
                    approval = _load_frontier_approval(
                        state,
                        lane=RETENTION_FRONTIER_LANE,
                        key=key,
                        input_hash=str(item.get("input_hash") or ""),
                        page_hashes=page_hashes,
                    )
                    if (
                        not isinstance(approval, dict)
                        or approval.get("review") != review
                        or approval.get("authority") != authority
                        or approval.get("effect_receipt") != effect_receipt
                    ):
                        raise OSError("frontier approval readback mismatch")
            except (OSError, ValueError) as exc:
                review = {
                    "decision": "needs_retry",
                    "confidence": 0.0,
                    "summary": f"frontier approval persistence failed: {exc}",
                    "schema_valid": False,
                }
                frontier_decision = "needs_retry"
        effect_receipt = (
            approval.get("effect_receipt") if isinstance(approval, dict) else None
        )

        result: dict[str, Any] | None = None
        if frontier_decision == "needs_retry":
            if is_local_semantic_no_quorum(review):
                try:
                    with decision_authority_lock():
                        current_authority, current_error = _current_autonomy_authority(
                            RETENTION_FRONTIER_LANE,
                            reviewer=reviewer,
                        )
                        epoch_error = current_error or compare_semantic_authority(
                            authority,
                            current_authority,
                            lane=RETENTION_FRONTIER_LANE,
                        )
                        if epoch_error is not None or current_authority is None:
                            raise ValueError(
                                epoch_error or "decision authority is unavailable"
                            )
                        transition = state.hold_semantic_no_quorum(
                            key,
                            lane=RETENTION_FRONTIER_LANE,
                            stage="frontier",
                            review=review,
                            epoch=_retention_semantic_epoch(
                                item,
                                page_id=page_id,
                                page_hash=str(snapshot["content_hash"]),
                                retention=row,
                            ),
                            authority=current_authority,
                            owner=owner,
                            error=str(
                                review.get("summary") or "local semantic no quorum"
                            ),
                            now=now,
                        )
                except (InvalidTransition, TypeError, ValueError) as exc:
                    transition = state.fail_attempt(
                        key,
                        "frontier",
                        error=f"semantic hold rejected: {exc}",
                        failure_class="review_artifact_invalid",
                        owner=owner,
                        now=now,
                    )
            else:
                transition = state.fail_attempt(
                    key,
                    "frontier",
                    error=str(review.get("summary") or "frontier needs retry"),
                    failure_class=_duplicate_frontier_failure_class(review),
                    owner=owner,
                    now=now,
                )
        elif frontier_decision == "keep_active":
            with decision_authority_lock():
                epoch_error = _autonomy_authority_epoch_error(
                    authority,
                    review,
                    lane=RETENTION_FRONTIER_LANE,
                    reviewer=reviewer,
                )
                if epoch_error is not None:
                    transition = state.fail_attempt(
                        key,
                        "frontier",
                        error=epoch_error,
                        failure_class="review_artifact_invalid",
                        owner=owner,
                        now=now,
                    )
                else:
                    transition = state.complete(
                        key,
                        "rejected",
                        result={"decision": "keep_active", "frontier": review},
                        owner=owner,
                        now=now,
                    )
        else:
            mutation_allowed, mutation_reason = cycle_budget.consume("mutation")
            if not mutation_allowed:
                transition = state.fail_attempt(
                    key,
                    "frontier",
                    error=mutation_reason,
                    owner=owner,
                    now=now,
                )
            else:
                with decision_authority_lock():
                    epoch_error = _autonomy_authority_epoch_error(
                        authority,
                        review,
                        lane=RETENTION_FRONTIER_LANE,
                        reviewer=reviewer,
                    )
                    if epoch_error is not None:
                        result = {"status": "retry", "reason": epoch_error}
                        transition = state.fail_attempt(
                            key,
                            "frontier",
                            error=epoch_error,
                            failure_class="review_artifact_invalid",
                            owner=owner,
                            now=now,
                        )
                    else:
                        assert isinstance(effect_receipt, dict)
                        receipt_decision_at = str(effect_receipt["decision_at"])
                        result = _patch_page_status(
                            page_id,
                            {
                                "status": "archived",
                                "autonomy_decision": "retention_frontier_archive",
                                "autonomy_decision_at": receipt_decision_at,
                                "frontier_approval_key": key,
                                "archive_reason": (
                                    "frontier_approved_reversible_soft_archive"
                                ),
                            },
                            expected_hash=str(snapshot["content_hash"]),
                            effect_receipt=effect_receipt,
                            correction_store=lifecycle_store,
                        )
                        if result.get("status") in {"applied", "unchanged"}:
                            transition = state.complete(
                                key,
                                "applied",
                                result={"frontier": review, "apply": result},
                                owner=owner,
                                now=now,
                            )
                            applied += 1
                        else:
                            transition = state.fail_attempt(
                                key,
                                "frontier",
                                error=str(result.get("reason") or result.get("status")),
                                owner=owner,
                                now=now,
                            )
        final_status = str(transition["item"].get("status") or "")
        decision.update(
            {
                "action": (
                    "archive"
                    if frontier_decision == "archive"
                    else "keep_active"
                    if frontier_decision == "keep_active"
                    else "defer"
                ),
                "apply": bool(
                    result and result.get("status") in {"applied", "unchanged"}
                ),
                "reason": (
                    str(result.get("reason") or result.get("status"))
                    if result
                    else str(review.get("summary") or frontier_decision)
                ),
                "status": final_status,
                "review": review,
                "result": result,
            }
        )
        decisions.append(decision)
        if write:
            _append_jsonl(DECISIONS_FILE, decision)

    status_counts: dict[str, int] = {}
    for decision in decisions:
        status = str(decision.get("status") or decision.get("reason") or "unknown")
        status_counts[status] = status_counts.get(status, 0) + 1
    return {
        "status": "ok",
        "dry_run": dry_run,
        "candidates": len(candidates),
        "considered": len(decisions),
        "frontier_calls": frontier_calls,
        "applied": applied,
        "retired": retired_stale.get("retired", []),
        "status_counts": dict(sorted(status_counts.items())),
        "budget": cycle_budget.snapshot(),
        "decisions": decisions[:20],
    }


def _capture_rate(health: dict[str, Any]) -> float | None:
    memory = health.get("memory_integrity")
    if not isinstance(memory, dict):
        return None
    value = memory.get("capture_rate")
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _queue_value(health: dict[str, Any], key: str) -> int:
    queues = health.get("queues")
    if not isinstance(queues, dict):
        return 0
    try:
        return int(queues.get(key) or 0)
    except (TypeError, ValueError):
        return 0


def _watchdog_history_summary(payload: dict[str, Any]) -> dict[str, Any]:
    health = payload.get("health") if isinstance(payload.get("health"), dict) else {}
    latest_sleep = (
        payload.get("latest_sleep")
        if isinstance(payload.get("latest_sleep"), dict)
        else {}
    )
    alerts = payload.get("alerts") if isinstance(payload.get("alerts"), list) else []
    if alerts:
        alert_types = [
            str(alert.get("type") or "unknown")[:200]
            for alert in alerts[:32]
            if isinstance(alert, dict)
        ]
    else:
        existing_types = payload.get("alert_types")
        alert_types = (
            [str(value)[:200] for value in existing_types[:32]]
            if isinstance(existing_types, list)
            else []
        )
    if health:
        capture_rate = _capture_rate(health)
        queues = {
            "duplicate_candidates": _queue_value(health, "duplicate_candidates"),
            "lint_repair": _queue_value(health, "lint_repair"),
        }
    else:
        try:
            capture_rate = float(payload.get("capture_rate"))
        except (TypeError, ValueError):
            capture_rate = None
        existing_queues = payload.get("queues")
        existing_queues = existing_queues if isinstance(existing_queues, dict) else {}
        compact_health = {"queues": existing_queues}
        queues = {
            "duplicate_candidates": _queue_value(
                compact_health, "duplicate_candidates"
            ),
            "lint_repair": _queue_value(compact_health, "lint_repair"),
        }
    try:
        alert_count = max(0, int(payload.get("alert_count") or 0))
    except (TypeError, ValueError):
        alert_count = 0
    return {
        "ts": str(payload.get("ts"))[:200] if payload.get("ts") is not None else None,
        "status": (
            str(payload.get("status"))[:200]
            if payload.get("status") is not None
            else None
        ),
        "alert_count": len(alerts) if alerts else alert_count,
        "alert_types": alert_types,
        "capture_rate": capture_rate,
        "queues": queues,
        "latest_sleep": _sleep_run_summary(latest_sleep),
    }


def _sleep_run_summary(payload: dict[str, Any]) -> dict[str, Any]:
    if not payload:
        return {}
    return {
        "status": str(payload.get("status"))[:200]
        if payload.get("status") is not None
        else None,
        "started_at": (
            str(payload.get("started_at"))[:200]
            if payload.get("started_at") is not None
            else None
        ),
        "finished_at": (
            str(payload.get("finished_at"))[:200]
            if payload.get("finished_at") is not None
            else None
        ),
        "run_id": str(payload.get("run_id"))[:200]
        if payload.get("run_id") is not None
        else None,
    }


def _compact_watchdog_alerts(alerts: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Persist only bounded scalar alert state needed for notification decisions."""

    compact: list[dict[str, Any]] = []
    for alert in alerts[:32]:
        row: dict[str, Any] = {}
        for key, value in alert.items():
            if isinstance(value, (str, int, float, bool)) or value is None:
                row[str(key)[:100]] = value[:500] if isinstance(value, str) else value
        if row.get("type"):
            compact.append(row)
    return compact


def _watchdog_alert_types(alerts: list[dict[str, Any]]) -> list[str]:
    return sorted(
        {
            str(alert.get("type") or "unknown")[:200]
            for alert in alerts
            if isinstance(alert, dict)
        }
    )


def _threshold_bucket(value: object, thresholds: tuple[float, ...]) -> int:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return 0
    return sum(number >= threshold for threshold in thresholds)


def _watchdog_alert_severity(alert: dict[str, Any]) -> int:
    """Return a coarse severity rank so small counter drift stays quiet."""

    alert_type = str(alert.get("type") or "")
    if alert_type == "lint_backlog_high":
        return _threshold_bucket(alert.get("value"), (250, 500, 1000, 2000))
    if alert_type == "background_jobs_quarantined":
        return _threshold_bucket(alert.get("value"), (1, 5, 10, 25, 50))
    if alert_type == "convergence_slo_missed":
        age = _threshold_bucket(alert.get("oldest_age_hours"), (24, 72, 168))
        actionable = _threshold_bucket(alert.get("actionable"), (1, 5, 10, 25, 50))
        return age * 10 + actionable
    if alert_type in {"capture_rate_low", "capture_rate_regression"}:
        value = alert.get("value", alert.get("after"))
        try:
            rate = float(value)
        except (TypeError, ValueError):
            return 1
        return sum(rate <= threshold for threshold in (0.8, 0.6, 0.4, 0.2))
    for key in ("value", "pending", "actionable", "age_seconds"):
        if key in alert:
            return _threshold_bucket(
                alert.get(key),
                (1, 5, 10, 25, 50, 100, 250, 500, 1000),
            )
    return 1


def _watchdog_alert_identity(alert: dict[str, Any]) -> tuple[tuple[str, str], ...]:
    """Return identity fields whose change should notify without waiting."""

    keys_by_type = {
        "component_error": (
            "component",
            "exception_type",
            "diagnostic_hash",
            "fingerprint",
            "incident_status",
        ),
        "runtime_commit_drift": ("runtime_commit", "expected_commit"),
        "runtime_commit_unknown": ("archive_path",),
        "sleep_status_not_ok": ("status",),
        "deadman_observer_unhealthy": ("status",),
        "quality_probe_error": ("error_type",),
    }
    alert_type = str(alert.get("type") or "")
    return tuple(
        (key, str(alert.get(key)))
        for key in keys_by_type.get(alert_type, ())
        if key in alert
    )


def _watchdog_alert_map(
    alerts: list[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    return {
        str(alert.get("type") or "unknown"): alert
        for alert in alerts
        if isinstance(alert, dict)
    }


def _watchdog_notification_plan(
    alerts: list[dict[str, Any]],
    previous_notification: dict[str, Any],
    *,
    now: datetime | None = None,
    reminder_hours: float = WATCHDOG_NOTIFICATION_REMINDER_HOURS,
) -> dict[str, Any]:
    """Decide whether the current watchdog transition deserves a notification."""

    current = _compact_watchdog_alerts(alerts)
    previous_raw = previous_notification.get("last_notified_alerts")
    previous = (
        _compact_watchdog_alerts(previous_raw) if isinstance(previous_raw, list) else []
    )
    has_baseline = isinstance(previous_raw, list)
    current_types = _watchdog_alert_types(current)
    previous_types = _watchdog_alert_types(previous)

    if not has_baseline:
        return {
            "send": bool(current),
            "reason": "new_alerts" if current else "unchanged",
        }
    if not current:
        return {
            "send": bool(previous),
            "reason": "recovered" if previous else "unchanged",
        }
    if current_types != previous_types:
        return {"send": True, "reason": "alert_set_changed"}

    current_map = _watchdog_alert_map(current)
    previous_map = _watchdog_alert_map(previous)
    if any(
        _watchdog_alert_identity(current_map[alert_type])
        != _watchdog_alert_identity(previous_map[alert_type])
        for alert_type in current_types
    ):
        return {"send": True, "reason": "alert_identity_changed"}
    if any(
        _watchdog_alert_severity(current_map[alert_type])
        > _watchdog_alert_severity(previous_map[alert_type])
        for alert_type in current_types
    ):
        return {"send": True, "reason": "severity_increased"}

    last_sent = _parse_dt(previous_notification.get("last_sent_at"))
    current_time = now or datetime.now(last_sent.tzinfo if last_sent else None)
    if last_sent is not None:
        if current_time.tzinfo is None and last_sent.tzinfo is not None:
            current_time = current_time.replace(tzinfo=last_sent.tzinfo)
        elif current_time.tzinfo is not None and last_sent.tzinfo is None:
            last_sent = last_sent.replace(tzinfo=current_time.tzinfo)
    if last_sent is not None and current_time - last_sent >= timedelta(
        hours=max(0.0, reminder_hours)
    ):
        return {"send": True, "reason": "reminder"}
    return {"send": False, "reason": "unchanged"}


def _watchdog_notification_body(alerts: list[dict[str, Any]], reason: str) -> str:
    if reason == "recovered":
        return "Autonomy recovered"
    alert_types = _watchdog_alert_types(alerts)
    visible = ", ".join(alert_types[:3])
    if len(alert_types) > 3:
        visible = f"{visible} +{len(alert_types) - 3}"
    prefix = "Reminder" if reason == "reminder" else "Alert"
    return f"{prefix}: {visible or 'unknown'}"


def _watchdog_notification_state(
    alerts: list[dict[str, Any]],
    previous_notification: dict[str, Any],
    *,
    enabled: bool,
) -> dict[str, Any]:
    """Send a transition notification and return durable notification state."""

    current = _compact_watchdog_alerts(alerts)
    state = {
        "status": "disabled" if not enabled else "suppressed",
        "reason": "notifications_disabled" if not enabled else "unchanged",
        "reminder_hours": WATCHDOG_NOTIFICATION_REMINDER_HOURS,
        "active_alert_types": _watchdog_alert_types(current),
        "last_sent_at": previous_notification.get("last_sent_at"),
        "last_notified_alerts": previous_notification.get("last_notified_alerts"),
    }
    if not enabled:
        return state

    plan = _watchdog_notification_plan(current, previous_notification)
    state["reason"] = plan["reason"]
    if not plan["send"]:
        return state

    delivery = _send_notification(
        "Chronovisor watchdog",
        _watchdog_notification_body(current, str(plan["reason"])),
    )
    state["delivery"] = delivery
    if delivery.get("sent") is True:
        state.update(
            {
                "status": "sent",
                "last_sent_at": _now(),
                "last_notified_alerts": current,
            }
        )
    else:
        state["status"] = "failed"
    return state


def _write_watchdog_history(
    payload: dict[str, Any],
    *,
    path: Path | None = None,
    max_lines: int = 1000,
) -> None:
    target = path or WATCHDOG_HISTORY
    max_lines = max(1, int(max_lines))
    previous: list[dict[str, Any]] = []
    try:
        lines = [
            line
            for line in target.read_text(encoding="utf-8").split("\n")
            if line.strip()
        ]
    except OSError:
        lines = []
    keep_previous = max_lines - 1
    recent_lines = lines[-keep_previous:] if keep_previous else []
    for line in recent_lines:
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(row, dict):
            continue
        previous.append(_watchdog_history_summary(row))
    rows = [*previous, _watchdog_history_summary(payload)][-max_lines:]
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_name(f".{target.name}.{os.getpid()}.tmp")
    try:
        tmp.write_text(
            "".join(
                json.dumps(row, ensure_ascii=False, default=str) + "\n" for row in rows
            ),
            encoding="utf-8",
        )
        os.replace(tmp, target)
    finally:
        with suppress(OSError):
            tmp.unlink()


def _run_watchdog_quality_probe(*, write: bool) -> dict[str, Any]:
    """Run the local quality probe only for the installed watchdog writer."""

    if not write:
        return {"status": "read_only_skipped"}
    try:
        from chronovisor.core.runtime_config import load_decision_router_config
        from chronovisor.decision.quality_guard import run_quality_probe

        adoption_artifact = load_decision_router_config().adoption_artifact.strip()
        if not adoption_artifact:
            return {"status": "not_configured", "frontier_calls": 0}
        result = run_quality_probe(
            root=CHRONOVISOR_ROOT / "runtime" / "quality",
            adoption_artifact=Path(adoption_artifact).expanduser(),
        )
        return {
            "status": "ok",
            "artifact_sha256": result.get("artifact_sha256"),
            "frozen_lanes": result.get("frozen_lanes"),
            "frontier_calls": 0,
        }
    except Exception as exc:
        return {
            "status": "error",
            "error_type": exc.__class__.__name__,
            "frontier_calls": 0,
        }


def _read_watchdog_health(
    health_snapshot: Callable[[], dict[str, Any]],
    *,
    write: bool,
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    """Read health or project a privacy-bounded component incident."""

    try:
        return health_snapshot(), None
    except Exception as exc:
        from chronovisor.ops.system_incident_supervisor import (
            safe_exception_diagnostic,
            supervise_health_snapshot_exception,
        )

        diagnostic = safe_exception_diagnostic(exc)
        incident_run_id = (
            f"watchdog:{datetime.now().isoformat(timespec='microseconds')}:"
            f"{os.getpid()}"
        )
        try:
            incident = supervise_health_snapshot_exception(
                exc,
                run_id=incident_run_id,
                runner=lambda _attempt: health_snapshot(),
                dry_run=not write,
            )
        except Exception as supervisor_exc:
            incident = {
                "status": "supervisor_error",
                "supervisor_error_type": supervisor_exc.__class__.__name__,
            }
        alert = {
            "type": "component_error",
            "component": "watchdog.health_snapshot",
            "exception_type": diagnostic.exception_type,
            "diagnostic_hash": diagnostic.diagnostic_hash,
            "incident_status": incident.get("status"),
            "fingerprint": incident.get("fingerprint"),
            "occurrence_count": incident.get("occurrence_count"),
            "distinct_input_count": incident.get("distinct_input_count"),
            "packet_path": incident.get("packet_path"),
            "supervisor_error_type": incident.get("supervisor_error_type"),
        }
        return {
            "status": "error",
            "component": "watchdog.health_snapshot",
        }, alert


def watchdog_snapshot(
    *,
    before_health: dict[str, Any] | None = None,
    write: bool = True,
    notify: bool = False,
    max_sleep_age_hours: float = 30.0,
    min_capture_rate: float = 0.80,
) -> dict[str, Any]:
    from chronovisor.ops.deadman import inspect_heartbeat, write_heartbeat
    from chronovisor.ops.health import health_snapshot
    from chronovisor.ops.sleep_cycle import HISTORY_FILE

    quality_probe = _run_watchdog_quality_probe(write=write)
    health, component_alert = _read_watchdog_health(
        health_snapshot,
        write=write,
    )
    latest_sleep = _latest_jsonl(HISTORY_FILE)
    alerts: list[dict[str, Any]] = []
    observer = inspect_heartbeat(
        OBSERVER_HEARTBEAT_FILE,
        expected_role="independent_observer",
        max_age_seconds=10 * 60,
    )
    observer_alert = (
        {
            "type": "deadman_observer_unhealthy",
            "status": observer.get("status"),
            "age_seconds": observer.get("age_seconds"),
            "sequence": observer.get("sequence"),
        }
        if observer.get("status") != "ok"
        else None
    )
    if component_alert is not None:
        alerts.append(component_alert)
    if quality_probe.get("status") == "error":
        alerts.append(
            {
                "type": "quality_probe_error",
                "error_type": quality_probe.get("error_type"),
                "frontier_allowed": False,
            }
        )
    capture = _capture_rate(health)
    if capture is not None and capture < min_capture_rate:
        alerts.append(
            {
                "type": "capture_rate_low",
                "value": capture,
                "threshold": min_capture_rate,
            }
        )
    sleep_started = _parse_dt(latest_sleep.get("started_at")) if latest_sleep else None
    if not latest_sleep:
        alerts.append({"type": "sleep_never_ran"})
    elif latest_sleep.get("status") != "ok":
        alerts.append(
            {"type": "sleep_status_not_ok", "status": latest_sleep.get("status")}
        )
    elif sleep_started and datetime.now() - sleep_started > timedelta(
        hours=max_sleep_age_hours
    ):
        alerts.append(
            {"type": "sleep_stale", "started_at": latest_sleep.get("started_at")}
        )
    if _queue_value(health, "duplicate_candidates") > 25:
        alerts.append(
            {
                "type": "duplicate_backlog_high",
                "value": _queue_value(health, "duplicate_candidates"),
            }
        )
    queues = health.get("queues") if isinstance(health.get("queues"), dict) else {}
    lint_total = _queue_value(health, "lint_repair")
    # Once a lint candidate is represented in the convergence ledger it is
    # durably owned by the bounded repair worker.  Alerting on that managed
    # catch-up work duplicates the convergence SLO below and makes every
    # healthy drain cycle look broken.  Keep the legacy total fallback for
    # older health payloads that do not expose the managed/untracked split.
    lint_untracked = (
        _queue_value(health, "lint_repair_untracked")
        if "lint_repair_untracked" in queues
        else lint_total
    )
    if lint_untracked > 250:
        alerts.append(
            {
                "type": "lint_backlog_high",
                "value": lint_untracked,
                "total": lint_total,
                "managed": max(0, lint_total - lint_untracked),
            }
        )
    convergence = (
        health.get("convergence") if isinstance(health.get("convergence"), dict) else {}
    )
    if int(convergence.get("expired_running") or 0) > 0:
        alerts.append(
            {
                "type": "convergence_expired_leases",
                "value": convergence.get("expired_running"),
            }
        )
    if int(convergence.get("quarantined") or 0) > 0:
        alerts.append(
            {
                "type": "convergence_quarantined",
                "value": convergence.get("quarantined"),
            }
        )
    if float(convergence.get("oldest_actionable_age_hours") or 0.0) > 24.0:
        alerts.append(
            {
                "type": "convergence_slo_missed",
                "oldest_age_hours": convergence.get("oldest_actionable_age_hours"),
                "actionable": convergence.get("actionable"),
            }
        )
    capture_pipeline = (
        health.get("capture_pipeline")
        if isinstance(health.get("capture_pipeline"), dict)
        else {}
    )
    background = (
        capture_pipeline.get("background_jobs")
        if isinstance(capture_pipeline.get("background_jobs"), dict)
        else {}
    )
    background_status = (
        background.get("by_status")
        if isinstance(background.get("by_status"), dict)
        else {}
    )
    retained_quarantined = int(background_status.get("quarantined") or 0)
    recent_quarantined_raw = background.get("quarantined_24h")
    recent_quarantined = (
        retained_quarantined
        if recent_quarantined_raw is None
        else int(recent_quarantined_raw or 0)
    )
    if recent_quarantined > 0:
        alert = {
            "type": "background_jobs_quarantined",
            "value": recent_quarantined,
        }
        if recent_quarantined_raw is not None:
            alert.update(
                {
                    "retained_total": retained_quarantined,
                    "window_hours": 24,
                }
            )
        alerts.append(alert)
    if int(background_status.get("retry_wait") or 0) > 0:
        alerts.append(
            {
                "type": "background_jobs_retrying",
                "value": background_status.get("retry_wait"),
            }
        )
    sweeper = (
        capture_pipeline.get("session_sweeper")
        if isinstance(capture_pipeline.get("session_sweeper"), dict)
        else {}
    )
    if sweeper.get("status") == "attention":
        alerts.append(
            {"type": "session_sweeper_attention", "pending": sweeper.get("pending")}
        )
    runtime = health.get("runtime") if isinstance(health.get("runtime"), dict) else {}
    if runtime.get("drift") is True:
        alerts.append(
            {
                "type": "runtime_commit_drift",
                "runtime_commit": runtime.get("commit_id"),
                "expected_commit": runtime.get("expected_commit"),
            }
        )
    if not runtime.get("commit_id"):
        alerts.append(
            {
                "type": "runtime_commit_unknown",
                "archive_path": runtime.get("archive_path"),
            }
        )

    read_back = health.get("read_back") if isinstance(health.get("read_back"), dict) else {}
    read_back_integrity = (
        read_back.get("derived_view_integrity")
        if isinstance(read_back.get("derived_view_integrity"), dict)
        else {}
    )
    if read_back_integrity.get("status") == "invalid":
        alerts.append({"type": "read_back_derived_view_invalid"})
    hardening = (
        health.get("autonomy_hardening")
        if isinstance(health.get("autonomy_hardening"), dict)
        else {}
    )
    quality = hardening.get("quality") if isinstance(hardening.get("quality"), dict) else {}
    if int(quality.get("frozen") or 0) > 0:
        alerts.append(
            {
                "type": "quality_lanes_frozen",
                "value": int(quality.get("frozen") or 0),
                "frontier_allowed": False,
            }
        )

    if before_health:
        before_capture = _capture_rate(before_health)
        if (
            before_capture is not None
            and capture is not None
            and capture + 0.15 < before_capture
        ):
            alerts.append(
                {
                    "type": "capture_rate_regression",
                    "before": before_capture,
                    "after": capture,
                }
            )

    # A read-only watchdog preview must not report deployment state from the
    # caller's test/sandbox namespace. The installed writer is the authority
    # that cross-checks the independent observer.
    if write and observer_alert is not None:
        alerts.append(observer_alert)
    previous_watchdog = _read_json(WATCHDOG_FILE) if write else {}
    previous_notification = (
        previous_watchdog.get("notification")
        if isinstance(previous_watchdog.get("notification"), dict)
        else {}
    )
    payload = {
        "status": "alert" if alerts else "ok",
        "ts": _now(),
        "alerts": alerts,
        "health": health,
        "latest_sleep": _sleep_run_summary(latest_sleep),
        "deadman": {
            "threshold_schema_version": 1,
            "observer": observer,
        },
        "quality_probe": quality_probe,
    }
    if write:
        heartbeat = write_heartbeat(
            WATCHDOG_HEARTBEAT_FILE,
            role="main_watchdog",
            reported_status=str(payload["status"]),
            peer=observer,
        )
        payload["deadman"]["main_sequence"] = heartbeat["sequence"]
        payload["notification"] = _watchdog_notification_state(
            alerts,
            previous_notification,
            enabled=notify,
        )
        _write_json(WATCHDOG_FILE, payload)
        _write_watchdog_history(payload)
    return payload


def _send_notification(title: str, body: str) -> dict[str, Any]:
    script = f"display notification {json.dumps(body)} with title {json.dumps(title)}"
    try:
        proc = subprocess.run(
            ["osascript", "-e", script], text=True, capture_output=True, timeout=5
        )
    except Exception as exc:
        return {"sent": False, "error": str(exc)}
    return {"sent": proc.returncode == 0, "stderr": proc.stderr.strip()}


def regression_guard(
    *,
    before_health: dict[str, Any],
    after_watchdog: dict[str, Any],
    snapshot: dict[str, Any],
    auto_revert: bool,
    write: bool = True,
) -> dict[str, Any]:
    alerts = [
        item
        for item in after_watchdog.get("alerts", [])
        if isinstance(item, dict) and item.get("type") in {"capture_rate_regression"}
    ]
    payload = {
        "status": "ok" if not alerts else "regression",
        "ts": _now(),
        "alerts": alerts,
        "auto_revert": auto_revert,
        "auto_revert_effective": False,
        "reverted": False,
        "rollback_scope": "mutation_cas_only",
        "global_reset_disabled": True,
    }
    head = str(snapshot.get("head") or "")
    if alerts:
        quarantine_action = {
            "ts": payload["ts"],
            "reason": "capture_rate_regression",
            "scope": "cycle",
            "snapshot_head": head or None,
            "note": "Global git reset is disabled; each mutation owns its CAS rollback.",
        }
        payload["quarantine"] = quarantine_action
        if write:
            quarantine = _read_json(QUARANTINE_FILE)
            quarantined = quarantine.get("actions")
            if not isinstance(quarantined, list):
                quarantined = []
            quarantined.append(quarantine_action)
            _write_json(QUARANTINE_FILE, {"actions": quarantined[-100:]})
    if write:
        _append_jsonl(DECISIONS_FILE, {"type": "regression_guard", **payload})
    return payload


def build_digest(payload: dict[str, Any], *, write: bool = True) -> dict[str, Any]:
    watchdog = (
        payload.get("watchdog") if isinstance(payload.get("watchdog"), dict) else {}
    )
    duplicate = (
        payload.get("duplicates") if isinstance(payload.get("duplicates"), dict) else {}
    )
    retention = (
        payload.get("archives") if isinstance(payload.get("archives"), dict) else {}
    )
    guard = (
        payload.get("regression_guard")
        if isinstance(payload.get("regression_guard"), dict)
        else {}
    )
    lines = [
        "# Chronovisor Autonomy Digest",
        "",
        f"- Generated: {_now()}",
        f"- Status: {watchdog.get('status', payload.get('status', 'unknown'))}",
        f"- Alerts: {len(watchdog.get('alerts', []) or [])}",
        f"- Duplicate decisions: {duplicate.get('applied', 0)} applied, {duplicate.get('deferred', 0)} deferred",
        f"- Retention archives: {retention.get('applied', 0)} applied",
        f"- Regression guard: {guard.get('status', 'unknown')}",
    ]
    if watchdog.get("alerts"):
        lines.extend(["", "## Alerts"])
        for alert in watchdog["alerts"]:
            lines.append(
                f"- `{alert.get('type', 'unknown')}` {json.dumps(alert, ensure_ascii=False)}"
            )
    text = "\n".join(lines).rstrip() + "\n"
    if write:
        DIGEST_FILE.parent.mkdir(parents=True, exist_ok=True)
        DIGEST_FILE.write_text(text, encoding="utf-8")
    return {"status": "ok", "path": str(DIGEST_FILE), "text": text}


def run_autonomy_cycle(
    *,
    duplicates: list[dict[str, Any]],
    retention: dict[str, Any],
    before_health: dict[str, Any],
    snapshot: dict[str, Any],
    dry_run: bool = False,
    auto_revert: bool = True,
    budget: CycleBudget | None = None,
    retention_budget: CycleBudget | None = None,
) -> dict[str, Any]:
    duplicate_result = resolve_duplicate_candidates(
        duplicates,
        apply=not dry_run,
        write=not dry_run,
        budget=budget,
    )
    archive_result = apply_retention_archives(
        retention,
        apply=not dry_run,
        write=not dry_run,
        budget=retention_budget or budget,
    )
    watchdog = watchdog_snapshot(before_health=before_health, write=not dry_run)
    guard = regression_guard(
        before_health=before_health,
        after_watchdog=watchdog,
        snapshot=snapshot,
        auto_revert=(not dry_run and auto_revert),
        write=not dry_run,
    )
    payload = {
        "status": "ok"
        if watchdog.get("status") == "ok" and guard.get("status") == "ok"
        else "attention",
        "ts": _now(),
        "dry_run": dry_run,
        "duplicates": duplicate_result,
        "archives": archive_result,
        "watchdog": watchdog,
        "regression_guard": guard,
    }
    digest = build_digest(payload, write=not dry_run)
    payload["digest"] = {k: v for k, v in digest.items() if k != "text"}
    if not dry_run:
        _write_json(LATEST_FILE, payload)
    return payload


def _uvx_path() -> str:
    return shutil.which("uvx") or str(Path.home() / ".local/bin/uvx")


def _plist(
    label: str,
    args: list[str],
    *,
    stdout: Path,
    stderr: Path,
    start_interval: int | None = None,
    calendar: dict[str, int] | None = None,
    environment: dict[str, str] | None = None,
) -> dict[str, Any]:
    data: dict[str, Any] = {
        "Label": label,
        "ProgramArguments": args,
        "WorkingDirectory": str(PROJECT_ROOT),
        "StandardOutPath": str(stdout),
        "StandardErrorPath": str(stderr),
        "RunAtLoad": False,
        "EnvironmentVariables": {
            "PATH": "/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin",
            "PYTHONUNBUFFERED": "1",
            "CHRONOVISOR_REPO_ROOT": str(PROJECT_ROOT),
        },
    }
    if environment:
        data["EnvironmentVariables"].update(environment)
    if start_interval is not None:
        data["StartInterval"] = start_interval
    if calendar is not None:
        data["StartCalendarInterval"] = calendar
    return data


def _write_plist(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    buf = io.BytesIO()
    plistlib.dump(data, buf, sort_keys=False)
    path.write_bytes(buf.getvalue())


def _write_wrapper(path: Path, command: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    script = "#!/bin/sh\nexec " + shlex.join(command) + "\n"
    path.write_text(script, encoding="utf-8")
    path.chmod(0o755)


def install_launchd(*, dry_run: bool = False, load: bool = False) -> dict[str, Any]:
    logs = CHRONOVISOR_ROOT / "logs"
    uvx = _uvx_path()
    sleep_path = LAUNCH_AGENT_DIR / f"{SLEEP_LABEL}.plist"
    converge_path = LAUNCH_AGENT_DIR / f"{CONVERGE_LABEL}.plist"
    watchdog_path = LAUNCH_AGENT_DIR / f"{WATCHDOG_LABEL}.plist"
    deadman_path = LAUNCH_AGENT_DIR / f"{DEADMAN_LABEL}.plist"
    soak_path = LAUNCH_AGENT_DIR / f"{SOAK_LABEL}.plist"
    sleep_wrapper = WRAPPER_DIR / "chronovisor-sleep"
    converge_wrapper = WRAPPER_DIR / "chronovisor-converge"
    watchdog_wrapper = WRAPPER_DIR / "chronovisor-watchdog"
    deadman_script = WRAPPER_DIR / "chronovisor-deadman-observer"
    soak_wrapper = WRAPPER_DIR / "chronovisor-soak"
    legacy_deadman_script = WRAPPER_DIR / "chronovisor-deadman-observer.py"
    sleep_command = [
        *uvx_runtime_command("chronovisor", executable=uvx, refresh=True),
        "sleep",
        "--raw-limit",
        "200",
        "--eval-limit",
        "150",
        "--duplicate-limit",
        "300",
    ]
    watchdog_command = [
        *uvx_runtime_command("chronovisor", executable=uvx, refresh=True),
        "autonomy",
        "watchdog",
        "--notify",
        "--json",
    ]
    converge_command = [
        *uvx_runtime_command("chronovisor-converge", executable=uvx, refresh=True),
        "--session-limit",
        "8",
        "--job-limit",
        "8",
        "--lint-limit",
        "50",
        "--orphan-limit",
        "8",
        "--maintenance-max-elapsed-seconds",
        "900",
        "--no-sleep",
    ]
    deadman_command = [
        str(deadman_script),
        "--chronovisor-root",
        str(CHRONOVISOR_ROOT),
        "--max-main-age-seconds",
        "1200",
    ]
    expected_commit = str(runtime_identity().get("expected_commit") or "")
    soak_command = [
        *uvx_runtime_command("chronovisor-burn-monitor", executable=uvx, refresh=True),
        "--duration-seconds",
        "604800",
        "--sample-seconds",
        "60",
        "--probe-seconds",
        "5",
        "--preflight-wait-seconds",
        "604800",
        "--final-idle-wait-seconds",
        "1800",
        "--expected-commit",
        expected_commit,
    ]
    sleep_plist = _plist(
        SLEEP_LABEL,
        [str(sleep_wrapper)],
        stdout=logs / "sleep-cycle.launchd.out.log",
        stderr=logs / "sleep-cycle.launchd.err.log",
        calendar={"Hour": 3, "Minute": 40},
    )
    # A newly installed or renamed agent may be loaded after today's calendar
    # boundary.  Run once at bootstrap so the watchdog always gets a durable
    # sleep receipt instead of reporting ``sleep_never_ran`` until tomorrow.
    sleep_plist["RunAtLoad"] = True
    watchdog_plist = _plist(
        WATCHDOG_LABEL,
        [str(watchdog_wrapper)],
        stdout=Path(os.devnull),
        stderr=logs / "watchdog.launchd.err.log",
        start_interval=900,
    )
    converge_plist = _plist(
        CONVERGE_LABEL,
        [str(converge_wrapper)],
        stdout=logs / "converge.launchd.out.log",
        stderr=logs / "converge.launchd.err.log",
        start_interval=1800,
    )
    deadman_plist = _plist(
        DEADMAN_LABEL,
        deadman_command,
        stdout=logs / "deadman-observer.launchd.out.log",
        stderr=logs / "deadman-observer.launchd.err.log",
        start_interval=300,
    )
    deadman_plist["RunAtLoad"] = True
    soak_plist = _plist(
        SOAK_LABEL,
        [str(soak_wrapper)],
        stdout=logs / "soak-7d.launchd.out.log",
        stderr=logs / "soak-7d.launchd.err.log",
    )
    soak_plist["RunAtLoad"] = True
    soak_plist["KeepAlive"] = {"SuccessfulExit": False}
    soak_plist["ThrottleInterval"] = 60
    payload: dict[str, Any] = {
        "status": "ok",
        "dry_run": dry_run,
        "load": load,
        "plists": [
            {
                "label": SLEEP_LABEL,
                "path": str(sleep_path),
                "program": sleep_plist["ProgramArguments"],
                "stdout": sleep_plist["StandardOutPath"],
                "run_at_load": bool(sleep_plist.get("RunAtLoad")),
            },
            {
                "label": CONVERGE_LABEL,
                "path": str(converge_path),
                "program": converge_plist["ProgramArguments"],
                "stdout": converge_plist["StandardOutPath"],
            },
            {
                "label": WATCHDOG_LABEL,
                "path": str(watchdog_path),
                "program": watchdog_plist["ProgramArguments"],
                "stdout": watchdog_plist["StandardOutPath"],
            },
            {
                "label": DEADMAN_LABEL,
                "path": str(deadman_path),
                "program": deadman_plist["ProgramArguments"],
                "stdout": deadman_plist["StandardOutPath"],
            },
            {
                "label": SOAK_LABEL,
                "path": str(soak_path),
                "program": soak_plist["ProgramArguments"],
                "stdout": soak_plist["StandardOutPath"],
            },
        ],
        "wrappers": [
            {"path": str(sleep_wrapper), "command": sleep_command},
            {"path": str(converge_wrapper), "command": converge_command},
            {"path": str(watchdog_wrapper), "command": watchdog_command},
            {"path": str(soak_wrapper), "command": soak_command},
        ],
    }
    if not dry_run:
        logs.mkdir(parents=True, exist_ok=True)
        _write_wrapper(sleep_wrapper, sleep_command)
        _write_wrapper(converge_wrapper, converge_command)
        _write_wrapper(watchdog_wrapper, watchdog_command)
        _write_wrapper(soak_wrapper, soak_command)
        # The package-independent observer lives at the package root.  The
        # operations namespace only contains a compatibility shim that imports
        # Chronovisor, which would defeat this external failure boundary if it
        # were copied as the standalone executable.
        observer_source = Path(__file__).parents[1] / "deadman_observer.py"
        deadman_script.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(observer_source, deadman_script)
        deadman_script.chmod(0o755)
        legacy_deadman_script.unlink(missing_ok=True)
        _write_plist(sleep_path, sleep_plist)
        _write_plist(converge_path, converge_plist)
        _write_plist(watchdog_path, watchdog_plist)
        _write_plist(deadman_path, deadman_plist)
        _write_plist(soak_path, soak_plist)
    if load and not dry_run:
        uid = os.getuid()
        loads = []
        for path in (
            sleep_path,
            converge_path,
            watchdog_path,
            deadman_path,
            soak_path,
        ):
            subprocess.run(
                ["launchctl", "bootout", f"gui/{uid}", str(path)],
                text=True,
                capture_output=True,
            )
            proc = subprocess.run(
                ["launchctl", "bootstrap", f"gui/{uid}", str(path)],
                text=True,
                capture_output=True,
            )
            loads.append(
                {
                    "path": str(path),
                    "returncode": proc.returncode,
                    "stderr": proc.stderr.strip(),
                }
            )
        payload["launchctl"] = loads
        if any(item["returncode"] != 0 for item in loads):
            payload["status"] = "error"
    return payload


def uninstall_launchd(*, dry_run: bool = False, unload: bool = False) -> dict[str, Any]:
    """Unload and remove all generated autonomous service artifacts."""

    labels = (SLEEP_LABEL, CONVERGE_LABEL, WATCHDOG_LABEL, DEADMAN_LABEL, SOAK_LABEL)
    plists = [LAUNCH_AGENT_DIR / f"{label}.plist" for label in labels]
    wrappers = [
        WRAPPER_DIR / "chronovisor-sleep",
        WRAPPER_DIR / "chronovisor-converge",
        WRAPPER_DIR / "chronovisor-watchdog",
        WRAPPER_DIR / "chronovisor-deadman-observer",
        WRAPPER_DIR / "chronovisor-deadman-observer.py",
        WRAPPER_DIR / "chronovisor-soak",
    ]
    payload: dict[str, Any] = {
        "status": "ok",
        "dry_run": dry_run,
        "unload": unload,
        "plists": [str(path) for path in plists],
        "wrappers": [str(path) for path in wrappers],
    }
    if dry_run:
        return payload
    unloads: list[dict[str, Any]] = []
    if unload:
        uid = os.getuid()
        for path in reversed(plists):
            proc = subprocess.run(
                ["launchctl", "bootout", f"gui/{uid}", str(path)],
                text=True,
                capture_output=True,
            )
            unloads.append(
                {
                    "path": str(path),
                    "returncode": proc.returncode,
                    "stderr": proc.stderr.strip(),
                }
            )
    for path in (*plists, *wrappers):
        path.unlink(missing_ok=True)
    payload["launchctl"] = unloads
    return payload


def status() -> dict[str, Any]:
    return {
        "status": "ok",
        "latest": _read_json(LATEST_FILE),
        "watchdog": _read_json(WATCHDOG_FILE),
        "digest_path": str(DIGEST_FILE),
        "launchd": {
            "sleep": str(LAUNCH_AGENT_DIR / f"{SLEEP_LABEL}.plist"),
            "converge": str(LAUNCH_AGENT_DIR / f"{CONVERGE_LABEL}.plist"),
            "watchdog": str(LAUNCH_AGENT_DIR / f"{WATCHDOG_LABEL}.plist"),
            "deadman": str(LAUNCH_AGENT_DIR / f"{DEADMAN_LABEL}.plist"),
            "soak": str(LAUNCH_AGENT_DIR / f"{SOAK_LABEL}.plist"),
        },
    }


def main(argv: list[str] | None = None) -> int:
    """Run the ``chronovisor-autonomy`` command-line entry point."""
    parser = argparse.ArgumentParser(
        description="Run or install Chronovisor autonomous operation."
    )
    sub = parser.add_subparsers(dest="command", required=True)
    status_parser = sub.add_parser("status")
    status_parser.add_argument("--json", action="store_true")
    watchdog_parser = sub.add_parser("watchdog")
    watchdog_parser.add_argument("--notify", action="store_true")
    watchdog_parser.add_argument("--json", action="store_true")
    digest_parser = sub.add_parser("digest")
    digest_parser.add_argument("--json", action="store_true")
    install_parser = sub.add_parser("install-launchd")
    install_parser.add_argument("--dry-run", action="store_true")
    install_parser.add_argument("--load", action="store_true")
    install_parser.add_argument("--json", action="store_true")
    uninstall_parser = sub.add_parser("uninstall-launchd")
    uninstall_parser.add_argument("--dry-run", action="store_true")
    uninstall_parser.add_argument("--unload", action="store_true")
    uninstall_parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)
    if args.command == "status":
        payload = status()
    elif args.command == "watchdog":
        payload = watchdog_snapshot(notify=args.notify)
    elif args.command == "digest":
        payload = build_digest(_read_json(LATEST_FILE))
    elif args.command == "install-launchd":
        payload = install_launchd(dry_run=args.dry_run, load=args.load)
    else:
        payload = uninstall_launchd(dry_run=args.dry_run, unload=args.unload)
    if getattr(args, "json", False):
        print(json.dumps(payload, ensure_ascii=False, indent=2, default=str))
    else:
        print(
            "\t".join(
                f"{key}={value}" for key, value in payload.items() if key != "latest"
            )
        )
    return 0 if payload.get("status") in {"ok", "alert"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
