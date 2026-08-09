"""Crash-safe targeted drain for an existing convergence backlog.

Unlike the daily sleep cycle, this runner never discovers or enqueues broad
new work.  ``start`` freezes the currently active supported keys in a durable
manifest.  Every later transition is constrained to that exact allowlist.
"""

from __future__ import annotations

import fcntl
import hashlib
import json
import math
import os
import tempfile
import time
import uuid
from collections import Counter, defaultdict
from collections.abc import Iterable, Iterator, Mapping
from contextlib import contextmanager, suppress
from dataclasses import asdict, dataclass
from dataclasses import field as dataclass_field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from chronovisor.core import store as chronovisor_store
from chronovisor.core.canonical_json import (
    canonical_json_bytes_stringifying as _canonical_bytes,
)
from chronovisor.core.timeutil import utc_now as _now
from chronovisor.ingest.convergence import (
    FRONTIER_STATUSES,
    LOCAL_STATUSES,
    TERMINAL_STATUSES,
    ConvergenceStore,
    CycleBudget,
    input_fingerprint,
    stable_item_key,
)

SCHEMA_VERSION = 1
ACTIVE_STATUSES = LOCAL_STATUSES | FRONTIER_STATUSES
PROCESSOR_LANES = (
    "content_correction",
    "autonomy_duplicate_resolution",
    "lint_repair",
    "orphan_link",
    "autonomy_retention",
)
SUPPORTED_LANES = (
    *PROCESSOR_LANES,
    "duplicate_frontier",
    "retention_frontier",
)
LANE_LIMITS = {
    "content_correction": 6,
    "autonomy_duplicate_resolution": 3,
    # Deterministic observations and orphan routing do not consume model-call
    # or mutation authority. Keep their throughput separate from the bounded
    # semantic tag-review budget so cheap rows cannot accumulate forever.
    "lint_repair": 200,
    "orphan_link": 2,
    "autonomy_retention": 3,
}
DECISION_POLICY_LANES = (
    "autonomy_duplicate_resolution",
    "autonomy_retention",
    "content_correction_classification",
    "content_correction_review",
    "exact_user_correction",
    "lint_tag_repair",
    "orphan_link",
)


class DrainError(RuntimeError):
    """The targeted drain cannot safely continue."""


@dataclass
class Inventory:
    """Read-only producer inventory and lane payloads for one observation."""

    keys_by_source: dict[str, dict[str, set[str]]]
    payloads: dict[str, Any]
    indeterminate_sources: set[tuple[str, str]]
    derived_items: list[dict[str, Any]]
    non_actionable_keys: set[str] = dataclass_field(default_factory=set)

    def canonical_projection(self) -> dict[str, Any]:
        return {
            "keys_by_source": {
                lane: {source: sorted(keys) for source, keys in sorted(sources.items())}
                for lane, sources in sorted(self.keys_by_source.items())
            },
            "derived_items": self.derived_items,
            "non_actionable_keys": sorted(self.non_actionable_keys),
        }




def _iso(value: datetime | None = None) -> str:
    return (value or _now()).astimezone(UTC).isoformat(timespec="seconds")


def _sha256_value(value: Any) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _file_fingerprint(path: Path) -> dict[str, Any]:
    try:
        raw = path.read_bytes()
    except FileNotFoundError:
        return {"status": "absent", "sha256": None, "bytes": 0}
    except OSError as exc:
        return {"status": "error", "error": str(exc), "sha256": None, "bytes": 0}
    return {
        "status": "present",
        "sha256": hashlib.sha256(raw).hexdigest(),
        "bytes": len(raw),
    }


def _repair_active_projection(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return {"status": "absent", "active": False, "incident_id": None}
    except (OSError, json.JSONDecodeError) as exc:
        return {
            "status": "indeterminate",
            "active": None,
            "incident_id": None,
            "error": str(exc),
        }
    if not isinstance(payload, Mapping):
        return {
            "status": "indeterminate",
            "active": None,
            "incident_id": None,
            "error": "frontier repair state is not an object",
        }
    incident_id = payload.get("active_incident_id")
    if not isinstance(incident_id, str) or not incident_id:
        return {"status": "ok", "active": False, "incident_id": None}
    incidents = payload.get("incidents")
    incident = incidents.get(incident_id) if isinstance(incidents, Mapping) else None
    if not isinstance(incident, Mapping):
        return {
            "status": "indeterminate",
            "active": None,
            "incident_id": incident_id,
            "error": "active frontier repair incident is missing",
        }
    incident_status = str(incident.get("status") or "")
    return {
        "status": "ok",
        "active": incident_status in {"reserved", "started"},
        "incident_id": incident_id,
        "incident_status": incident_status,
    }


def _frontier_fingerprint() -> dict[str, Any]:
    runtime_root = chronovisor_store.CHRONOVISOR_ROOT / "runtime"
    repair_root = runtime_root / "frontier-repair"
    frontier_events: list[dict[str, Any]] = []
    events_path = runtime_root / "events.jsonl"
    event_status = "ok"
    event_error: str | None = None
    try:
        lines = events_path.read_text(encoding="utf-8").splitlines()
    except FileNotFoundError:
        lines = []
    except OSError as exc:
        lines = []
        event_status = "error"
        event_error = str(exc)
    invalid_frontier_lines = 0
    if event_error is None:
        for line in lines:
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                if '"source"' in line and "frontier" in line:
                    invalid_frontier_lines += 1
                continue
            if isinstance(row, dict) and row.get("source") == "frontier":
                frontier_events.append(row)
    frontier_starts = [
        row
        for row in frontier_events
        if "frontier | review started" in str(row.get("message") or "")
    ]
    frontier_event_projection: dict[str, Any] = {
        "status": event_status,
        "count": len(frontier_events),
        "invalid_frontier_lines": invalid_frontier_lines,
        "sha256": _sha256_value(frontier_events) if event_error is None else None,
        "start_count": len(frontier_starts),
        "start_sha256": (
            _sha256_value(frontier_starts) if event_error is None else None
        ),
    }
    if event_error is not None:
        frontier_event_projection["error"] = event_error

    active_root = runtime_root / "frontier-reviews" / "active"
    active_records = []
    active_scan_status = "ok"
    active_scan_error: str | None = None
    try:
        active_paths = sorted(active_root.glob("*.json"))
    except OSError as exc:
        active_paths = []
        active_scan_status = "error"
        active_scan_error = str(exc)
    for path in active_paths:
        active_records.append({"name": path.name, **_file_fingerprint(path)})
    return {
        "repair": {
            "state": _file_fingerprint(repair_root / "state.json"),
            "events": _file_fingerprint(repair_root / "events.jsonl"),
            "active": _repair_active_projection(repair_root / "state.json"),
        },
        "frontier_events": frontier_event_projection,
        "frontier_active": {
            "status": active_scan_status,
            "count": len(active_records),
            "sha256": _sha256_value(active_records),
            "records": active_records,
            **({"error": active_scan_error} if active_scan_error is not None else {}),
        },
    }


def _frontier_baseline_error(value: object) -> str | None:
    if not isinstance(value, Mapping):
        return "frontier_baseline_missing"
    repair = value.get("repair")
    if not isinstance(repair, Mapping):
        return "frontier_repair_baseline_missing"
    for name in ("state", "events"):
        file_state = repair.get(name)
        if not isinstance(file_state, Mapping) or file_state.get("status") == "error":
            return f"frontier_repair_{name}_unreadable"
    repair_active = repair.get("active")
    if not isinstance(repair_active, Mapping) or repair_active.get("active") is None:
        return "frontier_repair_activity_indeterminate"
    if repair_active.get("active") is True:
        return "frontier_repair_already_active"
    frontier_events = value.get("frontier_events")
    if (
        not isinstance(frontier_events, Mapping)
        or frontier_events.get("status") != "ok"
    ):
        return "frontier_event_ledger_unreadable"
    if int(frontier_events.get("invalid_frontier_lines") or 0) > 0:
        return "frontier_event_ledger_malformed"
    active = value.get("frontier_active")
    if not isinstance(active, Mapping) or active.get("status") != "ok":
        return "frontier_active_markers_unreadable"
    if int(active.get("count") or 0) > 0:
        return "frontier_review_already_active"
    records = active.get("records")
    if not isinstance(records, list) or any(
        not isinstance(record, Mapping) or record.get("status") == "error"
        for record in records
    ):
        return "frontier_active_marker_unreadable"
    return None


def _frontier_start_delta(before: object, after: object) -> int | None:
    if not isinstance(before, Mapping) or not isinstance(after, Mapping):
        return None
    before_events = before.get("frontier_events")
    after_events = after.get("frontier_events")
    if not isinstance(before_events, Mapping) or not isinstance(after_events, Mapping):
        return None
    before_count = before_events.get("start_count")
    after_count = after_events.get("start_count")
    if (
        isinstance(before_count, bool)
        or not isinstance(before_count, int)
        or isinstance(after_count, bool)
        or not isinstance(after_count, int)
    ):
        return None
    return max(0, after_count - before_count)


def _decision_policy_fingerprint() -> dict[str, Any]:
    """Resolve every policy that can affect this drain's local decisions."""

    from chronovisor.decision.decision_policy import resolve_decision_policy

    lanes: dict[str, Any] = {}
    for name in DECISION_POLICY_LANES:
        try:
            policy, mode, error = resolve_decision_policy(name)
        except Exception as exc:
            lanes[name] = {
                "kind": None,
                "schema_name": None,
                "mode": "off",
                "error": f"{exc.__class__.__name__}: {exc}",
            }
            continue
        lanes[name] = {
            "kind": policy.kind if policy is not None else None,
            "schema_name": policy.schema_name if policy is not None else None,
            "mode": mode,
            "error": error,
        }
    return {"lanes": lanes, "sha256": _sha256_value(lanes)}


def _adoption_artifact_fingerprint() -> dict[str, Any]:
    """Bind config, lane policy, artifact bytes, and live router resolution."""

    from chronovisor.core.runtime_config import load_decision_router_config
    from chronovisor.decision.decision_router import (
        config_error,
        resolve_router_policy,
    )

    policies = _decision_policy_fingerprint()
    try:
        configured = load_decision_router_config()
    except Exception as exc:
        return {
            "path": None,
            "status": "error",
            "sha256": None,
            "bytes": 0,
            "decision_policies": policies,
            "configured_router": None,
            "configured_router_sha256": None,
            "resolved_router_policy": {
                "status": "error",
                "error": f"{exc.__class__.__name__}: {exc}",
            },
        }

    configured_projection = asdict(configured)
    configured_error = config_error(configured)
    nominated = configured.adoption_artifact.strip()
    if nominated:
        path = Path(nominated).expanduser()
        artifact = {"path": str(path), **_file_fingerprint(path)}
    else:
        artifact = {
            "path": None,
            "status": "not_nominated",
            "sha256": None,
            "bytes": 0,
        }

    try:
        resolution = resolve_router_policy(configured)
        resolved_projection = asdict(resolution.config)
        resolved_config_error = config_error(resolution.config)
        resolved = {
            "status": (
                "ok"
                if resolution.error is None and resolved_config_error is None
                else "error"
            ),
            "source": resolution.source,
            "artifact_path": resolution.artifact_path,
            "artifact_sha256": resolution.artifact_sha256,
            "error": resolution.error,
            "config_error": resolved_config_error,
            "config": resolved_projection,
            "config_sha256": _sha256_value(resolved_projection),
            "audit": resolution.audit_record(),
        }
    except Exception as exc:
        resolved = {
            "status": "error",
            "source": None,
            "artifact_path": str(Path(nominated).expanduser()) if nominated else None,
            "artifact_sha256": None,
            "error": f"{exc.__class__.__name__}: {exc}",
            "config_error": None,
            "config": None,
            "config_sha256": None,
            "audit": None,
        }
    return {
        **artifact,
        "decision_policies": policies,
        "configured_router": configured_projection,
        "configured_router_sha256": _sha256_value(configured_projection),
        "configured_router_error": configured_error,
        "resolved_router_policy": resolved,
    }


def _adoption_baseline_error(value: object) -> str | None:
    """Reject indeterminate policy or a nominated artifact not truly adopted."""

    if not isinstance(value, Mapping):
        return "adoption_baseline_missing"
    policies = value.get("decision_policies")
    lanes = policies.get("lanes") if isinstance(policies, Mapping) else None
    if not isinstance(lanes, Mapping):
        return "decision_policy_snapshot_missing"
    if policies.get("sha256") != _sha256_value(lanes):
        return "decision_policy_snapshot_digest_mismatch"
    for name in DECISION_POLICY_LANES:
        row = lanes.get(name)
        if not isinstance(row, Mapping):
            return f"decision_policy_missing:{name}"
        if row.get("error") is not None:
            return f"decision_policy_invalid:{name}"
        if row.get("mode") not in {"off", "shadow", "enabled"}:
            return f"decision_policy_mode_invalid:{name}"
        if not isinstance(row.get("kind"), str) or not row.get("kind"):
            return f"decision_policy_kind_missing:{name}"

    configured = value.get("configured_router")
    if not isinstance(configured, Mapping):
        return "configured_router_missing"
    if value.get("configured_router_sha256") != _sha256_value(configured):
        return "configured_router_digest_mismatch"
    path = value.get("path")
    if path is None:
        if value.get("status") != "not_nominated":
            return "adoption_artifact_state_invalid"
        if value.get("configured_router_error") is not None:
            return "configured_router_invalid"
    elif not isinstance(path, str) or not path:
        return "adoption_artifact_path_invalid"
    elif value.get("status") != "present" or not isinstance(value.get("sha256"), str):
        return "nominated_adoption_artifact_unreadable"
    resolved = value.get("resolved_router_policy")
    if not isinstance(resolved, Mapping) or resolved.get("status") != "ok":
        return "resolved_router_policy_invalid"
    if resolved.get("config_error") is not None:
        return "resolved_router_policy_config_invalid"
    audit = resolved.get("audit")
    if not isinstance(audit, Mapping):
        return "resolved_router_policy_audit_missing"
    resolved_config = resolved.get("config")
    if not isinstance(resolved_config, Mapping) or resolved.get(
        "config_sha256"
    ) != _sha256_value(resolved_config):
        return "resolved_router_policy_config_invalid"
    if dict(audit) != {
        "source": resolved.get("source"),
        "artifact_sha256": resolved.get("artifact_sha256"),
        "error": resolved.get("error"),
        "models": [
            resolved_config.get("primary_model"),
            resolved_config.get("challenger_model"),
            resolved_config.get("tie_break_model"),
        ],
    }:
        return "resolved_router_policy_audit_mismatch"

    if path is None:
        if resolved.get("source") != "bootstrap_current_policy":
            return "resolved_router_policy_source_invalid"
        enabled_model_lanes = sorted(
            name
            for name, row in lanes.items()
            if isinstance(row, Mapping)
            and row.get("kind") in {"consensus", "local_batch"}
            and row.get("mode") == "enabled"
        )
        if enabled_model_lanes:
            return "adoption_artifact_required:" + ",".join(enabled_model_lanes)
        return None
    if (
        resolved.get("source") != "adopted_artifact"
        or resolved.get("artifact_path") != path
        or resolved.get("artifact_sha256") != value.get("sha256")
        or resolved.get("error") is not None
    ):
        return "nominated_adoption_artifact_not_adopted"
    return None


def _runtime_commit() -> str | None:
    from chronovisor.core.runtime_config import runtime_identity

    identity = runtime_identity()
    value = identity.get("commit_id")
    return str(value) if value else None


def _runtime_adoption_observation(manifest: Mapping[str, Any]) -> dict[str, Any]:
    """Compare live execution authority with the sealed manifest baseline."""

    try:
        runtime_commit = _runtime_commit()
        adoption_artifact = _adoption_artifact_fingerprint()
    except Exception as exc:
        return {
            "status": "indeterminate",
            "runtime_commit": None,
            "runtime_changed": None,
            "adoption_sha256": None,
            "adoption_changed": None,
            "error": f"{exc.__class__.__name__}: {exc}",
        }
    runtime_changed = runtime_commit != manifest.get("runtime_commit")
    adoption_changed = adoption_artifact != manifest.get("adoption_artifact")
    return {
        "status": "changed" if runtime_changed or adoption_changed else "unchanged",
        "runtime_commit": runtime_commit,
        "runtime_changed": runtime_changed,
        "adoption_sha256": _sha256_value(adoption_artifact),
        "adoption_changed": adoption_changed,
        "error": None,
    }


def _drain_dir() -> Path:
    return chronovisor_store.CHRONOVISOR_ROOT / "runtime" / "convergence" / "drains"


def _manifest_path(run_id: str) -> Path:
    if not run_id or any(
        char not in "abcdefghijklmnopqrstuvwxyz0123456789-" for char in run_id
    ):
        raise ValueError(
            "run_id must contain only lowercase letters, digits, and hyphens"
        )
    return _drain_dir() / f"{run_id}.json"


def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    tmp_path = Path(temporary)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(
                payload,
                handle,
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
                default=str,
            )
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_path, path)
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    except Exception:
        with suppress(OSError):
            tmp_path.unlink()
        raise


def _manifest_projection(manifest: Mapping[str, Any]) -> dict[str, Any]:
    """Return the complete durable record excluding its own checksum."""

    return {
        str(key): value
        for key, value in manifest.items()
        if str(key) != "manifest_sha256"
    }


def _manifest_sha256(manifest: Mapping[str, Any]) -> str:
    return _sha256_value(_manifest_projection(manifest))


def _write_manifest(run_id: str, manifest: dict[str, Any]) -> None:
    """Seal and atomically replace one manifest after an audited transition."""

    manifest["manifest_sha256"] = _manifest_sha256(manifest)
    _atomic_write_json(_manifest_path(run_id), manifest)


def _read_manifest(run_id: str) -> dict[str, Any]:
    path = _manifest_path(run_id)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise DrainError(f"unknown convergence drain run: {run_id}") from exc
    except (OSError, json.JSONDecodeError) as exc:
        raise DrainError(f"cannot read convergence drain manifest: {exc}") from exc
    if not isinstance(payload, dict) or payload.get("schema_version") != SCHEMA_VERSION:
        raise DrainError("unsupported convergence drain manifest")
    if payload.get("run_id") != run_id:
        raise DrainError(
            "convergence drain manifest run_id does not match its filename"
        )
    return payload


@contextmanager
def _run_lock(run_id: str) -> Iterator[None]:
    path = _manifest_path(run_id).with_suffix(".lock")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+", encoding="utf-8") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


@contextmanager
def _local_only_environment() -> Iterator[None]:
    forced = {
        "CHRONOVISOR_DECISION_POLICY_SYSTEM_CODE_REPAIR": "off",
        "CHRONOVISOR_SELF_HEAL_AUTORUN": "0",
    }
    previous = {name: os.environ.get(name) for name in forced}
    os.environ.update(forced)
    try:
        yield
    finally:
        for name, value in previous.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


@contextmanager
def _read_only_environment() -> Iterator[None]:
    """Suppress derived-index persistence while building a plan snapshot."""

    previous = os.environ.get("CHRONOVISOR_READ_ONLY")
    os.environ["CHRONOVISOR_READ_ONLY"] = "1"
    try:
        yield
    finally:
        if previous is None:
            os.environ.pop("CHRONOVISOR_READ_ONLY", None)
        else:
            os.environ["CHRONOVISOR_READ_ONLY"] = previous


def _all_active_items(store: ConvergenceStore) -> list[dict[str, Any]]:
    return store.list_items(statuses=ACTIVE_STATUSES)


def _active_items(store: ConvergenceStore) -> list[dict[str, Any]]:
    return [
        item
        for item in _all_active_items(store)
        if str(item.get("lane") or "") in SUPPORTED_LANES
    ]


def _state_items_snapshot(store: ConvergenceStore) -> list[dict[str, Any]]:
    """Read one internally consistent convergence-state projection."""

    payload = store.load()
    raw_items = payload.get("items") if isinstance(payload, Mapping) else None
    if not isinstance(raw_items, Mapping):
        raise DrainError("convergence state snapshot is missing items")
    return sorted(
        [dict(item) for item in raw_items.values() if isinstance(item, Mapping)],
        key=lambda item: (str(item.get("lane") or ""), str(item.get("key") or "")),
    )


def _duplicate_inventory(
    current_sources: set[str],
    legacy_sources: set[str],
) -> tuple[
    dict[str, set[str]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    set[tuple[str, str]],
]:
    from chronovisor.ops import autonomy
    from chronovisor.recall.duplicate_review import build_duplicate_review_queue

    try:
        records = build_duplicate_review_queue(limit=1_000_000, strict=True)
    except Exception:
        uncertain = {
            ("autonomy_duplicate_resolution", source) for source in current_sources
        } | {("duplicate_frontier", source) for source in legacy_sources}
        return {}, [], [], uncertain
    current: dict[str, set[str]] = defaultdict(set)
    scoped_records: list[dict[str, Any]] = []
    derived_items: list[dict[str, Any]] = []
    all_sources = current_sources | legacy_sources
    for record in records:
        candidate = autonomy._canonical_duplicate_record(record)
        if candidate is None:
            continue
        source = f"{candidate['left']}<->{candidate['right']}"
        if source not in all_sources:
            continue
        left = autonomy._duplicate_page_snapshot(candidate["left"])
        right = autonomy._duplicate_page_snapshot(candidate["right"])
        input_data = {
            "pair": [candidate["left"], candidate["right"]],
            "content_hashes": {
                candidate["left"]: left["content_hash"],
                candidate["right"]: right["content_hash"],
            },
        }
        key = stable_item_key(
            autonomy.DUPLICATE_FRONTIER_LANE,
            source,
            input_data,
            resolver_version=autonomy.DUPLICATE_FRONTIER_RESOLVER_VERSION,
        )
        current[source].add(key)
        scoped_records.append(record)
        if source in legacy_sources:
            local_decision = autonomy.decide_duplicate(record)
            derived_items.append(
                {
                    "key": key,
                    "lane": autonomy.DUPLICATE_FRONTIER_LANE,
                    "source_id": source,
                    "input_hash": input_fingerprint(input_data),
                    "input_data": input_data,
                    "resolver_version": autonomy.DUPLICATE_FRONTIER_RESOLVER_VERSION,
                    "metadata": {
                        "candidate": candidate,
                        "local_action": local_decision.get("action"),
                        "local_reason": local_decision.get("reason"),
                        "local_proposal": local_decision,
                    },
                    "derived_from_lane": "duplicate_frontier",
                }
            )
    return dict(current), scoped_records, derived_items, set()


def _lint_inventory(
    sources: set[str],
) -> tuple[dict[str, set[str]], Path, set[tuple[str, str]]]:
    from chronovisor.ops import lint_repair

    path = chronovisor_store.CHRONOVISOR_ROOT / "review" / "lint-repair-queue.jsonl"
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return {}, path, {("lint_repair", source) for source in sources}
    rows: list[dict[str, Any]] = []
    invalid = 0
    for line in lines:
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            invalid += 1
            continue
        if isinstance(value, dict):
            rows.append(value)
        else:
            invalid += 1
    current: dict[str, set[str]] = defaultdict(set)
    uncertain: set[tuple[str, str]] = set()
    for row in rows:
        page_id = str(row.get("page") or "")
        source, _missing_input = lint_repair._candidate_identity(row, None)
        if source not in sources:
            continue
        try:
            page_path = chronovisor_store.find_page(page_id) if page_id else None
            page_text = page_path.read_text(encoding="utf-8") if page_path else None
        except OSError:
            uncertain.add(("lint_repair", source))
            continue
        source, input_data = lint_repair._candidate_identity(row, page_text)
        current[source].add(
            stable_item_key(
                "lint_repair",
                source,
                input_data,
                resolver_version=lint_repair.REPAIR_RESOLVER_VERSION,
            )
        )
    if invalid:
        uncertain.update(("lint_repair", source) for source in sources)
    return dict(current), path, uncertain


def _orphan_inventory(
    sources: set[str],
) -> tuple[dict[str, set[str]], set[tuple[str, str]]]:
    from chronovisor.core.index_store import get_store
    from chronovisor.decision.decision_authority import current_semantic_authority
    from chronovisor.ops import orphan_link
    from chronovisor.search.search import semantic_search

    try:
        index = get_store()
        index.refresh()
        orphan_ids = set(index.orphans(include_system=False))
    except Exception:
        return {}, {("orphan_link", source) for source in sources}
    current: dict[str, set[str]] = defaultdict(set)
    indeterminate: set[tuple[str, str]] = set()
    for source in sources:
        orphan_id = source.removeprefix("orphan:")
        if orphan_id not in orphan_ids:
            continue
        try:
            candidates = orphan_link.gather_candidates(
                orphan_id,
                index,
                max_candidates=3,
                semantic_search_fn=lambda query, top_n: semantic_search(
                    query,
                    top_n,
                    strict=True,
                ),
            )
            authority, error = current_semantic_authority(orphan_link.DECISION_LANE)
        except Exception:
            candidates, authority, error = [], None, "inventory_error"
        if authority is None or error is not None:
            indeterminate.add(("orphan_link", source))
            continue
        orphan_hash = orphan_link._content_hash(orphan_id)
        candidate_hashes = {
            candidate_id: orphan_link._content_hash(candidate_id)
            for candidate_id in candidates
        }
        if orphan_hash in {"missing", "unreadable"} or any(
            value in {"missing", "unreadable"} for value in candidate_hashes.values()
        ):
            indeterminate.add(("orphan_link", source))
            continue
        input_data = {
            "orphan": orphan_id,
            "orphan_hash": orphan_hash,
            "decision_authority": authority,
            "candidates": [
                {
                    "source": candidate_id,
                    "source_hash": candidate_hashes[candidate_id],
                }
                for candidate_id in candidates
            ],
        }
        current[source].add(
            stable_item_key(
                "orphan_link",
                source,
                input_data,
                resolver_version=orphan_link.RESOLVER_VERSION,
            )
        )
    return dict(current), indeterminate


def _retention_inventory(
    current_sources: set[str],
    legacy_sources: set[str],
) -> tuple[
    dict[str, set[str]],
    dict[str, Any],
    list[dict[str, Any]],
    set[tuple[str, str]],
]:
    from chronovisor.core.retention import build_retention_scores
    from chronovisor.ops import autonomy

    all_sources = current_sources | legacy_sources
    try:
        payload = build_retention_scores(write=False)
    except Exception:
        uncertain = {("autonomy_retention", source) for source in current_sources} | {
            ("retention_frontier", source) for source in legacy_sources
        }
        return {}, {}, [], uncertain
    uncertain_all = {("autonomy_retention", source) for source in current_sources} | {
        ("retention_frontier", source) for source in legacy_sources
    }
    pages_value = payload.get("pages")
    candidates_value = payload.get("archive_candidates")
    counts_value = payload.get("counts")
    total_candidates = (
        counts_value.get("archive_candidates")
        if isinstance(counts_value, Mapping)
        else None
    )
    if (
        payload.get("status") != "ok"
        or not isinstance(pages_value, dict)
        or not isinstance(candidates_value, list)
        or any(not isinstance(value, str) for value in candidates_value)
        or isinstance(total_candidates, bool)
        or not isinstance(total_candidates, int)
        or total_candidates < len(candidates_value)
    ):
        return {}, {}, [], uncertain_all
    pages = pages_value
    candidates = candidates_value
    scoped_candidates: list[str] = []
    current: dict[str, set[str]] = defaultdict(set)
    derived_items: list[dict[str, Any]] = []
    uncertain: set[tuple[str, str]] = set()
    candidate_sources = {str(value) for value in candidates if isinstance(value, str)}
    if total_candidates > len(candidate_sources):
        for source in all_sources - candidate_sources:
            if source in current_sources:
                uncertain.add(("autonomy_retention", source))
            if source in legacy_sources:
                uncertain.add(("retention_frontier", source))
    for source in [str(value) for value in candidates if isinstance(value, str)]:
        if source not in all_sources:
            continue
        scoped_candidates.append(source)
        snapshot = autonomy._duplicate_page_snapshot(source)
        if snapshot.get("status") != "ok":
            if source in current_sources:
                uncertain.add(("autonomy_retention", source))
            if source in legacy_sources:
                uncertain.add(("retention_frontier", source))
            continue
        input_data = {"page_id": source, "content_hash": snapshot["content_hash"]}
        key = stable_item_key(
            autonomy.RETENTION_FRONTIER_LANE,
            source,
            input_data,
            resolver_version=autonomy.RETENTION_FRONTIER_RESOLVER_VERSION,
        )
        current[source].add(key)
        if source in legacy_sources:
            row = pages.get(source) if isinstance(pages.get(source), dict) else {}
            derived_items.append(
                {
                    "key": key,
                    "lane": autonomy.RETENTION_FRONTIER_LANE,
                    "source_id": source,
                    "input_hash": input_fingerprint(input_data),
                    "input_data": input_data,
                    "resolver_version": autonomy.RETENTION_FRONTIER_RESOLVER_VERSION,
                    "metadata": {
                        "page_id": source,
                        "retention": row,
                        "local_recommendation": "archive",
                    },
                    "derived_from_lane": "retention_frontier",
                }
            )
    return (
        dict(current),
        {
            **payload,
            "pages": pages,
            "archive_candidates": scoped_candidates,
        },
        derived_items,
        uncertain,
    )


def _build_inventory(items: Iterable[Mapping[str, Any]]) -> Inventory:
    from chronovisor.recall import content_correction

    sources: dict[str, set[str]] = defaultdict(set)
    keys_by_source: dict[str, dict[str, set[str]]] = defaultdict(
        lambda: defaultdict(set)
    )
    indeterminate: set[tuple[str, str]] = set()
    non_actionable_keys: set[str] = set()
    for item in items:
        lane = str(item.get("lane") or "")
        source = str(item.get("source_id") or "")
        sources[lane].add(source)
        if lane == "content_correction":
            try:
                actionable, _reason = content_correction.correction_item_actionability(
                    item
                )
            except Exception:
                actionable = None
            if actionable is True:
                keys_by_source[lane][source].add(str(item.get("key") or ""))
            elif actionable is False:
                non_actionable_keys.add(str(item.get("key") or ""))
            else:
                indeterminate.add((lane, source))

    payloads: dict[str, Any] = {}
    derived_items: list[dict[str, Any]] = []
    if sources["autonomy_duplicate_resolution"] or sources["duplicate_frontier"]:
        duplicate, payloads["autonomy_duplicate_resolution"], derived, uncertain = (
            _duplicate_inventory(
                sources["autonomy_duplicate_resolution"],
                sources["duplicate_frontier"],
            )
        )
        indeterminate.update(uncertain)
        keys_by_source["autonomy_duplicate_resolution"].update(duplicate)
        for source in sources["duplicate_frontier"]:
            if source in duplicate:
                keys_by_source["duplicate_frontier"][source].update(duplicate[source])
        derived_items.extend(derived)
    if sources["lint_repair"]:
        lint, payloads["lint_repair"], uncertain = _lint_inventory(
            sources["lint_repair"]
        )
        keys_by_source["lint_repair"].update(lint)
        indeterminate.update(uncertain)
    if sources["orphan_link"]:
        orphan, uncertain = _orphan_inventory(sources["orphan_link"])
        keys_by_source["orphan_link"].update(orphan)
        indeterminate.update(uncertain)
    if sources["autonomy_retention"] or sources["retention_frontier"]:
        retention, payloads["autonomy_retention"], derived, uncertain = (
            _retention_inventory(
                sources["autonomy_retention"],
                sources["retention_frontier"],
            )
        )
        indeterminate.update(uncertain)
        keys_by_source["autonomy_retention"].update(retention)
        for source in sources["retention_frontier"]:
            if source in retention:
                keys_by_source["retention_frontier"][source].update(retention[source])
        derived_items.extend(derived)
    return Inventory(
        keys_by_source={lane: dict(values) for lane, values in keys_by_source.items()},
        payloads=payloads,
        indeterminate_sources=indeterminate,
        derived_items=derived_items,
        non_actionable_keys=non_actionable_keys,
    )


def _classify_item(item: Mapping[str, Any], inventory: Inventory) -> str:
    lane = str(item.get("lane") or "")
    source = str(item.get("source_id") or "")
    key = str(item.get("key") or "")
    if key in inventory.non_actionable_keys:
        return "non_actionable"
    if (lane, source) in inventory.indeterminate_sources:
        return "indeterminate"
    current = inventory.keys_by_source.get(lane, {}).get(source, set())
    if not current:
        return "source_absent"
    if key not in current:
        return "source_superseded"
    return "current"


def plan(*, store: ConvergenceStore | None = None) -> dict[str, Any]:
    """Build a byte-for-byte read-only targeted drain plan."""

    state = store or ConvergenceStore()
    state_items = _state_items_snapshot(state)
    all_active = [
        item for item in state_items if str(item.get("status") or "") in ACTIVE_STATUSES
    ]
    items = [
        item for item in all_active if str(item.get("lane") or "") in SUPPORTED_LANES
    ]
    runtime_before = _runtime_commit()
    adoption_before = _adoption_artifact_fingerprint()
    frontier_before = _frontier_fingerprint()
    with _read_only_environment():
        inventory = _build_inventory(items)
    runtime_after = _runtime_commit()
    adoption_after = _adoption_artifact_fingerprint()
    frontier_after = _frontier_fingerprint()
    if runtime_after != runtime_before or adoption_after != adoption_before:
        raise DrainError("runtime or adoption changed while building drain inventory")
    if frontier_after != frontier_before:
        raise DrainError("frontier activity changed while building drain inventory")
    derived_items: list[dict[str, Any]] = []
    for raw_row in inventory.derived_items:
        row = dict(raw_row)
        lane = str(row.get("lane") or "")
        source_id = str(row.get("source_id") or "")
        row["source_key_baseline"] = sorted(
            str(item.get("key") or "")
            for item in state_items
            if str(item.get("lane") or "") == lane
            and str(item.get("source_id") or "") == source_id
            and str(item.get("key") or "")
        )
        derived_items.append(row)
    inventory_projection = inventory.canonical_projection()
    inventory_projection["derived_items"] = derived_items
    planned = [
        {
            "key": str(item.get("key") or ""),
            "lane": str(item.get("lane") or ""),
            "source_id": str(item.get("source_id") or ""),
            "input_hash": str(item.get("input_hash") or ""),
            "resolver_version": str(item.get("resolver_version") or ""),
            "initial_status": str(item.get("status") or ""),
            "attempts": {
                "local": int(item.get("local_attempts") or 0),
                "frontier": int(item.get("frontier_attempts") or 0),
            },
            "source_state": _classify_item(item, inventory),
        }
        for item in items
    ]
    counts = Counter(str(item["source_state"]) for item in planned)
    lane_counts = Counter(str(item["lane"]) for item in planned)
    return {
        "status": "planned",
        "dry_run": True,
        "active_keys": len(planned),
        "unsupported_active_keys": len(all_active) - len(items),
        "counts": dict(sorted(counts.items())),
        "lanes": dict(sorted(lane_counts.items())),
        "inventory_sha256": _sha256_value(inventory_projection),
        "runtime_commit": runtime_after,
        "adoption_artifact": adoption_after,
        "frontier_repair": frontier_after,
        "items": planned,
        "derived_items": derived_items,
        "derived_keys": len(derived_items),
    }


def _allowlist_projection(manifest: Mapping[str, Any]) -> dict[str, Any]:
    projection: dict[str, Any] = {}
    for field in ("items", "derived_items"):
        rows = manifest.get(field)
        if not isinstance(rows, list):
            raise DrainError(f"manifest {field} must be a list")
        projection[field] = sorted(
            [row for row in rows if isinstance(row, Mapping)],
            key=lambda row: str(row.get("key") or ""),
        )
    return projection


def _allowlist_sha256(manifest: Mapping[str, Any]) -> str:
    return _sha256_value(_allowlist_projection(manifest))


def _manifest_integrity_error(manifest: Mapping[str, Any]) -> str | None:
    """Validate immutable scope first, then the complete durable record."""

    try:
        if manifest.get("allowlist_sha256") != _allowlist_sha256(manifest):
            return "manifest_allowlist_digest_mismatch"
    except (DrainError, TypeError, ValueError):
        return "manifest_allowlist_digest_mismatch"
    recorded = manifest.get("manifest_sha256")
    if not isinstance(recorded, str) or recorded != _manifest_sha256(manifest):
        return "manifest_integrity_mismatch"
    return None


def _new_manifest(
    planned: Mapping[str, Any],
    *,
    run_id: str,
    max_elapsed_seconds: float,
) -> dict[str, Any]:
    now = _iso()
    try:
        normalized_elapsed = float(max_elapsed_seconds)
    except (TypeError, ValueError) as exc:
        raise DrainError(
            "max_elapsed_seconds must be a finite positive number"
        ) from exc
    if (
        isinstance(max_elapsed_seconds, bool)
        or not math.isfinite(normalized_elapsed)
        or normalized_elapsed <= 0.0
    ):
        raise DrainError("max_elapsed_seconds must be a finite positive number")
    runtime_commit = _runtime_commit()
    adoption_artifact = _adoption_artifact_fingerprint()
    if not runtime_commit or not planned.get("runtime_commit"):
        raise DrainError(
            "targeted convergence drain requires an installed runtime commit identity"
        )
    if runtime_commit != planned.get(
        "runtime_commit"
    ) or adoption_artifact != planned.get("adoption_artifact"):
        raise DrainError("runtime or adoption changed before manifest persistence")
    adoption_error = _adoption_baseline_error(adoption_artifact)
    if adoption_error is not None:
        raise DrainError(
            f"targeted convergence drain adoption precondition: {adoption_error}"
        )
    planned_frontier = planned.get("frontier_repair")
    frontier_error = _frontier_baseline_error(planned_frontier)
    if frontier_error is not None:
        raise DrainError(
            f"targeted convergence drain frontier precondition: {frontier_error}"
        )
    current_frontier = _frontier_fingerprint()
    if current_frontier != planned_frontier:
        raise DrainError("frontier activity changed before manifest persistence")
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "run_id": run_id,
        "status": "created",
        "created_at": now,
        "updated_at": now,
        "runtime_commit": runtime_commit,
        "adoption_artifact": adoption_artifact,
        "frontier_repair_baseline": current_frontier,
        "frontier_repair_postcondition": "unchanged",
        "inventory_sha256": planned["inventory_sha256"],
        "batch_budget": {
            "max_elapsed_seconds": normalized_elapsed,
            "max_local_generation_calls": 30,
            "max_local_consensus_calls": 24,
            "max_model_effect_mutations": 60,
            "subscription_frontier_calls": 0,
            "legacy_cycle_budget_mapping": {
                "local": "local_generation",
                "frontier": "local_consensus",
                "mutation": "model_effect_mutation",
            },
            "lane_limits": dict(LANE_LIMITS),
        },
        "items": list(planned["items"]),
        "derived_items": list(planned.get("derived_items", [])),
        "attempts": 0,
        "last_run": None,
        "next_retry_at": None,
        "out_of_scope_active": [],
        "history": [],
    }
    manifest["allowlist_sha256"] = _allowlist_sha256(manifest)
    manifest["manifest_sha256"] = _manifest_sha256(manifest)
    return manifest


def start(
    *,
    store: ConvergenceStore | None = None,
    max_elapsed_seconds: float = 1_800.0,
    dry_run: bool = False,
    run_once: bool = True,
) -> dict[str, Any]:
    """Freeze a manifest before any claim, then optionally run one batch."""

    state = store or ConvergenceStore()
    planned = plan(store=state)
    if dry_run:
        return {**planned, "command": "start", "manifest_written": False}
    run_id = f"{_now().strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:10]}"
    manifest = _new_manifest(
        planned,
        run_id=run_id,
        max_elapsed_seconds=max_elapsed_seconds,
    )
    _write_manifest(run_id, manifest)
    if not run_once:
        return status(run_id=run_id, store=state)
    return resume(run_id=run_id, store=state)


def _manifest_item_keys(manifest: Mapping[str, Any], field: str) -> set[str]:
    items = manifest.get(field)
    if not isinstance(items, list):
        raise DrainError(f"manifest {field} must be a list")
    return {
        str(item.get("key") or "")
        for item in items
        if isinstance(item, Mapping) and str(item.get("key") or "")
    }


def _manifest_keys(manifest: Mapping[str, Any]) -> set[str]:
    return _manifest_item_keys(manifest, "items") | _manifest_item_keys(
        manifest, "derived_items"
    )


def _merge_derived_items(
    manifest: Mapping[str, Any], store: ConvergenceStore
) -> dict[str, Any]:
    rows = manifest.get("derived_items")
    if not isinstance(rows, list):
        raise DrainError("manifest derived_items must be a list")
    allowlist = _manifest_keys(manifest)
    created: list[str] = []
    existing: list[str] = []
    blocked: dict[str, list[str]] = {}
    prepared: list[tuple[Mapping[str, Any], set[str]]] = []
    for row in rows:
        if not isinstance(row, Mapping):
            raise DrainError("manifest derived item must be an object")
        required = {
            "key",
            "lane",
            "source_id",
            "input_hash",
            "input_data",
            "resolver_version",
            "metadata",
            "source_key_baseline",
        }
        if not required.issubset(row):
            raise DrainError("manifest derived item is incomplete")
        source_key_baseline = row.get("source_key_baseline")
        if not isinstance(source_key_baseline, list) or any(
            not isinstance(value, str) or not value for value in source_key_baseline
        ):
            raise DrainError("manifest derived source history is invalid")
        observed_source_keys = {
            str(item.get("key") or "")
            for item in store.list_items(lane=str(row["lane"]))
            if str(item.get("source_id") or "") == str(row["source_id"])
            and str(item.get("key") or "")
        }
        baseline_source_keys = set(source_key_baseline)
        new_source_keys = (
            observed_source_keys - baseline_source_keys - {str(row["key"])}
        )
        missing_source_keys = baseline_source_keys - observed_source_keys
        if new_source_keys or missing_source_keys:
            blocked[str(row["key"])] = sorted(new_source_keys) + [
                f"missing:{key}" for key in sorted(missing_source_keys)
            ]
            continue
        expected_key = stable_item_key(
            str(row["lane"]),
            str(row["source_id"]),
            row["input_data"],
            resolver_version=row["resolver_version"],
        )
        if (
            expected_key != row["key"]
            or input_fingerprint(row["input_data"]) != row["input_hash"]
        ):
            raise DrainError("manifest derived item identity mismatch")
        prepared.append((row, baseline_source_keys))

    # Validate every frozen source history before creating any derived item.
    # Per-row store guards below repeat this check under the mutation lock to
    # close races after the optimistic batch preflight.
    if blocked:
        return {"created": [], "existing": [], "blocked": blocked}

    batch = store.merge_items_atomically(
        [
            {
                "lane": str(row["lane"]),
                "source_id": str(row["source_id"]),
                "input_data": row["input_data"],
                "resolver_version": row["resolver_version"],
                "metadata": (
                    row["metadata"] if isinstance(row["metadata"], Mapping) else {}
                ),
                "update_metadata": False,
                "supersede_eligible_keys": allowlist,
                "source_history_eligible_keys": baseline_source_keys
                | {str(row["key"])},
                "source_history_required_keys": baseline_source_keys,
            }
            for row, baseline_source_keys in prepared
        ]
    )
    if not batch.get("committed"):
        batch_blocked = batch.get("blocked_by_key")
        if not isinstance(batch_blocked, Mapping) or not batch_blocked:
            raise DrainError("atomic derived merge failed without blockers")
        return {
            "created": [],
            "existing": [],
            "blocked": {
                str(key): [str(value) for value in values]
                for key, values in batch_blocked.items()
                if isinstance(values, list)
            },
        }
    results = batch.get("results")
    if not isinstance(results, list) or len(results) != len(prepared):
        raise DrainError("atomic derived merge result count mismatch")
    for (row, _baseline_source_keys), merged in zip(prepared, results, strict=False):
        item = merged.get("item") if isinstance(merged, Mapping) else None
        if not isinstance(item, Mapping) or item.get("key") != row["key"]:
            raise DrainError("derived convergence merge readback mismatch")
        (created if merged.get("created") else existing).append(str(row["key"]))
    return {"created": created, "existing": existing, "blocked": blocked}


def _state_identity_errors(
    manifest: Mapping[str, Any],
    store: ConvergenceStore,
    *,
    fields: tuple[str, ...] = ("items", "derived_items"),
    allow_missing: bool = False,
) -> list[str]:
    errors: list[str] = []
    for field in fields:
        rows = manifest.get(field)
        if not isinstance(rows, list):
            return [f"manifest_{field}_invalid"]
        for row in rows:
            if not isinstance(row, Mapping):
                errors.append(f"{field}:non_object")
                continue
            key = str(row.get("key") or "")
            item = store.get(key)
            if item is None:
                if not allow_missing:
                    errors.append(f"{key}:missing")
                continue
            for name in ("lane", "source_id", "input_hash", "resolver_version"):
                if str(item.get(name) or "") != str(row.get(name) or ""):
                    errors.append(f"{key}:{name}_mismatch")
    return errors


def _current_scoped_items(
    manifest: Mapping[str, Any], store: ConvergenceStore
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    keys = _manifest_keys(manifest)
    scoped: list[dict[str, Any]] = []
    for key in sorted(keys):
        item = store.get(key)
        if item is not None:
            scoped.append(item)
    outside = [
        item
        for item in _all_active_items(store)
        if str(item.get("key") or "") not in keys
    ]
    return scoped, outside


def _next_retry_at(items: Iterable[Mapping[str, Any]]) -> str | None:
    values: list[str] = []
    for item in items:
        if str(item.get("status") or "") not in ACTIVE_STATUSES:
            continue
        for field in ("next_attempt_at", "lease_expires_at"):
            value = item.get(field)
            if isinstance(value, str) and value:
                values.append(value)
    return min(values) if values else None


def _run_lanes(
    *,
    store: ConvergenceStore,
    eligible_keys: set[str],
    inventory: Inventory,
    max_elapsed_seconds: float,
) -> dict[str, Any]:
    from chronovisor.ops import autonomy, lint_repair, orphan_link
    from chronovisor.recall import content_correction

    budget = CycleBudget(
        max_local_calls=30,
        max_frontier_calls=24,
        max_mutations=60,
        max_elapsed_seconds=max_elapsed_seconds,
    )
    by_lane: dict[str, set[str]] = defaultdict(set)
    for key in eligible_keys:
        item = store.get(key)
        if item is not None and str(item.get("status") or "") in ACTIVE_STATUSES:
            by_lane[str(item.get("lane") or "")].add(key)
    results: dict[str, Any] = {}
    if keys := by_lane.get("content_correction"):
        results["content_correction"] = content_correction.run_pending_corrections(
            max_items=LANE_LIMITS["content_correction"],
            store=store,
            budget=budget.slice(
                max_local_calls=6,
                max_frontier_calls=6,
                max_mutations=3,
            ),
            eligible_keys=set(keys),
        )
    if keys := by_lane.get("autonomy_duplicate_resolution"):
        records = inventory.payloads.get("autonomy_duplicate_resolution", [])
        results["autonomy_duplicate_resolution"] = (
            autonomy.resolve_deferred_duplicates_with_frontier(
                records if isinstance(records, list) else [],
                convergence_store=store,
                budget=budget.slice(max_frontier_calls=3, max_mutations=3),
                dry_run=False,
                eligible_keys=set(keys),
            )
        )
    if keys := by_lane.get("lint_repair"):
        queue_file = inventory.payloads.get("lint_repair")
        results["lint_repair"] = lint_repair.run_lint_repair(
            queue_file=queue_file if isinstance(queue_file, Path) else None,
            store=store,
            budget=budget.slice(
                max_local_calls=10,
                max_frontier_calls=3,
                max_mutations=6,
            ),
            max_items=LANE_LIMITS["lint_repair"],
            eligible_keys=set(keys),
        )
    if keys := by_lane.get("orphan_link"):
        results["orphan_link"] = orphan_link.run_autonomous(
            orphan_limit=LANE_LIMITS["orphan_link"],
            convergence_store=store,
            budget=budget.slice(
                max_local_calls=8,
                max_frontier_calls=3,
                max_mutations=3,
            ),
            eligible_keys=set(keys),
        )
    if keys := by_lane.get("autonomy_retention"):
        payload = inventory.payloads.get("autonomy_retention")
        results["autonomy_retention"] = autonomy.apply_retention_archives(
            payload if isinstance(payload, dict) else {},
            limit=LANE_LIMITS["autonomy_retention"],
            budget=budget.slice(max_frontier_calls=3, max_mutations=3),
            convergence_store=store,
            eligible_keys=set(keys),
        )
    legacy_budget = budget.snapshot()
    limits = legacy_budget.get("limits") or {}
    consumed = legacy_budget.get("consumed") or {}
    remaining = legacy_budget.get("remaining") or {}
    return {
        "lanes": results,
        "local_execution_budget": {
            "elapsed_seconds": legacy_budget.get("elapsed_seconds"),
            "limits": {
                "local_generation": limits.get("local"),
                "local_consensus": limits.get("frontier"),
                "model_effect_mutation": limits.get("mutation"),
            },
            "consumed": {
                "local_generation": consumed.get("local"),
                "local_consensus": consumed.get("frontier"),
                "model_effect_mutation": consumed.get("mutation"),
            },
            "remaining": {
                "local_generation": remaining.get("local"),
                "local_consensus": remaining.get("frontier"),
                "model_effect_mutation": remaining.get("mutation"),
            },
        },
        "subscription_frontier_calls": 0,
    }


def _status_payload(
    manifest: Mapping[str, Any], store: ConvergenceStore
) -> dict[str, Any]:
    integrity_error = _manifest_integrity_error(manifest)
    if integrity_error is not None:
        run_id = str(manifest.get("run_id") or "")
        try:
            manifest_path: str | None = str(_manifest_path(run_id))
        except ValueError:
            manifest_path = None
        return {
            "status": "failed",
            "run_id": run_id or None,
            "manifest_path": manifest_path,
            "target_keys": 0,
            "derived_keys": 0,
            "allowlist_keys": 0,
            "target_active": 0,
            "target_terminal": 0,
            "target_missing": 0,
            "status_counts": {},
            "out_of_scope_active": [],
            "next_retry_at": None,
            "frontier_repair_postcondition": "indeterminate_manifest",
            "failure_reason": integrity_error,
            "attempts": manifest.get("attempts", 0),
            "last_run": manifest.get("last_run"),
        }
    scoped, outside = _current_scoped_items(manifest, store)
    statuses = Counter(str(item.get("status") or "missing") for item in scoped)
    active = [
        item for item in scoped if str(item.get("status") or "") in ACTIVE_STATUSES
    ]
    missing = len(_manifest_keys(manifest)) - len(scoped)
    if missing:
        statuses["missing"] += missing
    run_status = str(manifest.get("status") or "unknown")
    failure_reason = manifest.get("failure_reason")
    reported_postcondition = manifest.get("frontier_repair_postcondition")
    live_frontier = _frontier_fingerprint()
    if live_frontier != manifest.get("frontier_repair_baseline"):
        run_status = "failed_frontier_activity"
        if reported_postcondition in {"unchanged", None}:
            reported_postcondition = "changed_live"
    else:
        try:
            runtime_changed = _runtime_commit() != manifest.get("runtime_commit")
            adoption_changed = _adoption_artifact_fingerprint() != manifest.get(
                "adoption_artifact"
            )
        except Exception:
            runtime_changed = True
            adoption_changed = True
            failure_reason = "runtime_or_adoption_unreadable"
        if runtime_changed or adoption_changed:
            run_status = "failed"
            failure_reason = failure_reason or "runtime_or_adoption_drift"
        elif run_status in {"failed_frontier_activity", "failed"}:
            pass
        else:
            run_status = "completed" if not active and not missing else "running"
            if outside and run_status == "completed":
                run_status = "attention"
    return {
        "status": run_status,
        "run_id": manifest.get("run_id"),
        "manifest_path": str(_manifest_path(str(manifest.get("run_id") or ""))),
        "target_keys": len(_manifest_item_keys(manifest, "items")),
        "derived_keys": len(_manifest_item_keys(manifest, "derived_items")),
        "allowlist_keys": len(_manifest_keys(manifest)),
        "target_active": len(active),
        "target_terminal": sum(statuses.get(value, 0) for value in TERMINAL_STATUSES),
        "target_missing": missing,
        "status_counts": dict(sorted(statuses.items())),
        "out_of_scope_active": [
            {
                "key": item.get("key"),
                "lane": item.get("lane"),
                "status": item.get("status"),
            }
            for item in outside
        ],
        "next_retry_at": _next_retry_at(active),
        "frontier_repair_postcondition": reported_postcondition,
        "failure_reason": failure_reason,
        "attempts": manifest.get("attempts", 0),
        "last_run": manifest.get("last_run"),
    }


def resume(
    *,
    run_id: str,
    store: ConvergenceStore | None = None,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Run at most one bounded batch for the immutable manifest allowlist."""

    state = store or ConvergenceStore()
    if dry_run:
        manifest = _read_manifest(run_id)
        return {**_status_payload(manifest, state), "dry_run": True}
    with _run_lock(run_id):
        invocation_started = time.monotonic()
        manifest = _read_manifest(run_id)
        integrity_error = _manifest_integrity_error(manifest)
        if integrity_error is not None:
            return _status_payload(manifest, state)
        if manifest.get("status") in {"failed_frontier_activity", "failed"}:
            return _status_payload(manifest, state)
        baseline = manifest.get("frontier_repair_baseline")
        frontier_before = _frontier_fingerprint()
        if frontier_before != baseline:
            manifest["status"] = "failed_frontier_activity"
            manifest["frontier_repair_postcondition"] = "changed_before_resume"
            manifest["updated_at"] = _iso()
            _write_manifest(run_id, manifest)
            return _status_payload(manifest, state)
        try:
            runtime_commit = _runtime_commit()
            adoption_artifact = _adoption_artifact_fingerprint()
        except Exception as exc:
            manifest["status"] = "failed"
            manifest["failure_reason"] = "runtime_or_adoption_unreadable"
            manifest["runtime_guard_error"] = f"{exc.__class__.__name__}: {exc}"
            manifest["updated_at"] = _iso()
            _write_manifest(run_id, manifest)
            return _status_payload(manifest, state)
        if runtime_commit != manifest.get("runtime_commit") or adoption_artifact != (
            manifest.get("adoption_artifact")
        ):
            manifest["status"] = "failed"
            manifest["failure_reason"] = (
                "runtime_commit_changed"
                if runtime_commit != manifest.get("runtime_commit")
                else "adoption_artifact_changed"
            )
            manifest["updated_at"] = _iso()
            _write_manifest(run_id, manifest)
            return _status_payload(manifest, state)

        eligible = _manifest_keys(manifest)
        premerge_identity_errors = _state_identity_errors(
            manifest,
            state,
            fields=("items",),
        ) + _state_identity_errors(
            manifest,
            state,
            fields=("derived_items",),
            allow_missing=True,
        )
        if premerge_identity_errors:
            manifest["status"] = "failed"
            manifest["failure_reason"] = "manifest_state_identity_mismatch"
            manifest["state_identity_errors"] = premerge_identity_errors
            manifest["updated_at"] = _iso()
            _write_manifest(run_id, manifest)
            return _status_payload(manifest, state)
        derived_merge = _merge_derived_items(manifest, state)
        if derived_merge["blocked"]:
            manifest["status"] = "failed"
            manifest["failure_reason"] = "derived_source_changed_before_merge"
            manifest["derived_merge"] = derived_merge
            manifest["updated_at"] = _iso()
            _write_manifest(run_id, manifest)
            return _status_payload(manifest, state)
        scoped, _outside = _current_scoped_items(manifest, state)
        identity_errors = _state_identity_errors(manifest, state)
        if len(scoped) != len(eligible) or identity_errors:
            manifest["status"] = "failed"
            manifest["failure_reason"] = "manifest_state_identity_mismatch"
            manifest["state_identity_errors"] = identity_errors
            manifest["updated_at"] = _iso()
            _write_manifest(run_id, manifest)
            return _status_payload(manifest, state)
        active = [
            item for item in scoped if str(item.get("status") or "") in ACTIVE_STATUSES
        ]
        content_keys = {
            str(item.get("key") or "")
            for item in active
            if str(item.get("lane") or "") == "content_correction"
            and str(item.get("key") or "")
        }
        content_migration: dict[str, Any] = {
            "status": "ok",
            "requested": 0,
            "completed": 0,
        }
        try:
            if content_keys:
                from chronovisor.recall import content_correction

                content_migration = (
                    content_correction.retire_non_actionable_corrections(
                        store=state,
                        eligible_keys=content_keys,
                        dry_run=False,
                    )
                )
        except BaseException as exc:
            frontier_after_migration = _frontier_fingerprint()
            manifest["status"] = (
                "failed_frontier_activity"
                if frontier_after_migration != baseline
                else "failed"
            )
            manifest["frontier_repair_postcondition"] = (
                "changed" if frontier_after_migration != baseline else "unchanged"
            )
            manifest["failure_reason"] = (
                "frontier_activity_during_content_migration"
                if frontier_after_migration != baseline
                else "content_false_positive_migration_error"
            )
            manifest["content_migration_error"] = f"{exc.__class__.__name__}: {exc}"
            manifest["updated_at"] = _iso()
            _write_manifest(run_id, manifest)
            if not isinstance(exc, Exception):
                raise
            return _status_payload(manifest, state)

        # The deterministic migration above can terminalize most stale
        # content-correction false positives without any model call. Refresh
        # before producer classification so generic source retirement cannot
        # overwrite its dedicated audit reason.
        scoped, _outside = _current_scoped_items(manifest, state)
        active = [
            item for item in scoped if str(item.get("status") or "") in ACTIVE_STATUSES
        ]
        try:
            inventory = _build_inventory(active)
        except Exception as exc:
            frontier_after_inventory = _frontier_fingerprint()
            manifest["status"] = (
                "failed_frontier_activity"
                if frontier_after_inventory != baseline
                else "failed"
            )
            manifest["frontier_repair_postcondition"] = (
                "changed" if frontier_after_inventory != baseline else "unchanged"
            )
            manifest["failure_reason"] = (
                "frontier_activity_during_inventory"
                if frontier_after_inventory != baseline
                else "producer_inventory_error"
            )
            manifest["inventory_error"] = f"{exc.__class__.__name__}: {exc}"
            manifest["updated_at"] = _iso()
            _write_manifest(run_id, manifest)
            return _status_payload(manifest, state)
        absent: list[str] = []
        superseded: list[str] = []
        indeterminate: list[str] = []
        non_actionable: list[str] = []
        for item in active:
            classification = _classify_item(item, inventory)
            key = str(item.get("key") or "")
            if classification == "source_absent":
                absent.append(key)
            elif classification == "source_superseded":
                superseded.append(key)
            elif classification == "indeterminate":
                indeterminate.append(key)
            elif classification == "non_actionable":
                non_actionable.append(key)

        # The manifest is already durable at this point.  These migrations can
        # therefore never race ahead of allowlist persistence.
        state.reap_expired_leases(eligible_keys=eligible)
        retired_absent = state.complete_many(
            absent,
            "rejected",
            result={"reason": "targeted_drain_source_absent"},
        )
        retired_superseded = state.complete_many(
            superseded,
            "rejected",
            result={"reason": "targeted_drain_source_superseded"},
        )

        configured_elapsed = (manifest.get("batch_budget") or {}).get(
            "max_elapsed_seconds"
        )
        max_elapsed = (
            float(configured_elapsed)
            if isinstance(configured_elapsed, (int, float))
            and not isinstance(configured_elapsed, bool)
            else 1_800.0
        )
        lane_exception: BaseException | None = None
        from chronovisor.core.page_mutation import decision_authority_lock

        with decision_authority_lock(
            chronovisor_store.CHRONOVISOR_ROOT / "runtime" / "decision-authority.lock"
        ):
            authority_before_lane = _runtime_adoption_observation(manifest)
            try:
                if authority_before_lane["status"] != "unchanged":
                    lane_result = {
                        "status": "blocked",
                        "error": "runtime_or_adoption_changed_before_lane",
                    }
                else:
                    # Lock contention and live adoption validation consume the
                    # same hard elapsed budget. Recompute only after both so a
                    # stale pre-lock allowance cannot start model work late.
                    lane_elapsed_budget = max(
                        0.0,
                        max_elapsed - (time.monotonic() - invocation_started),
                    )
                    if lane_elapsed_budget <= 0.0:
                        lane_result = {
                            "status": "budget_exhausted",
                            "max_elapsed_seconds": max_elapsed,
                        }
                    else:
                        with _local_only_environment():
                            lane_result = _run_lanes(
                                store=state,
                                eligible_keys=eligible,
                                inventory=inventory,
                                max_elapsed_seconds=lane_elapsed_budget,
                            )
            except BaseException as exc:
                lane_exception = exc
                lane_result = {
                    "status": "error",
                    "error": f"{exc.__class__.__name__}: {exc}",
                }
            finally:
                # Cooperative adoption writers share this lock. Direct config
                # edits and runtime replacement are still detected and sealed
                # before the lock is released, even when a lane raises.
                authority_after_lane = _runtime_adoption_observation(manifest)
                frontier_after = _frontier_fingerprint()
        postcondition = "unchanged" if frontier_after == baseline else "changed"
        attempt = int(manifest.get("attempts") or 0) + 1
        current_status = _status_payload(manifest, state)
        history = manifest.get("history")
        history = history if isinstance(history, list) else []
        run_record = {
            "attempt": attempt,
            "finished_at": _iso(),
            "retired_source_absent": retired_absent.get("completed", 0),
            "retired_source_superseded": retired_superseded.get("completed", 0),
            "indeterminate_sources": indeterminate,
            "non_actionable": non_actionable,
            "content_false_positive_migration": content_migration,
            "target_active": current_status["target_active"],
            "next_retry_at": current_status["next_retry_at"],
            "frontier_repair_postcondition": postcondition,
            "runtime_adoption_before_lane": authority_before_lane,
            "runtime_adoption_postcondition": authority_after_lane,
            "authority_sealed_effects": (
                authority_before_lane["status"] == "unchanged"
                and authority_after_lane["status"] == "unchanged"
            ),
            "subscription_frontier_starts": _frontier_start_delta(
                baseline, frontier_after
            ),
            "elapsed_seconds": round(time.monotonic() - invocation_started, 3),
            "lane_result": lane_result,
            "lane_exception": (
                f"{lane_exception.__class__.__name__}: {lane_exception}"
                if lane_exception is not None
                else None
            ),
        }
        manifest["attempts"] = attempt
        manifest["last_run"] = run_record
        manifest["history"] = [*history[-19:], run_record]
        manifest["updated_at"] = _iso()
        manifest["next_retry_at"] = current_status["next_retry_at"]
        manifest["out_of_scope_active"] = current_status["out_of_scope_active"]
        manifest["frontier_repair_postcondition"] = postcondition
        authority_failed = (
            authority_before_lane["status"] != "unchanged"
            or authority_after_lane["status"] != "unchanged"
        )
        manifest["status"] = (
            "failed_frontier_activity"
            if postcondition != "unchanged"
            else "failed"
            if lane_exception is not None or authority_failed
            else "completed"
            if current_status["target_active"] == 0
            and current_status["target_missing"] == 0
            else "running"
        )
        if postcondition != "unchanged":
            manifest["failure_reason"] = "frontier_activity_during_lane"
        elif authority_failed:
            manifest["failure_reason"] = (
                "runtime_or_adoption_changed_before_lane"
                if authority_before_lane["status"] != "unchanged"
                else "runtime_or_adoption_changed_during_lane"
            )
        elif lane_exception is not None:
            manifest["failure_reason"] = "lane_exception"
        _write_manifest(run_id, manifest)
        if lane_exception is not None and not isinstance(lane_exception, Exception):
            raise lane_exception
        return _status_payload(manifest, state)


def status(
    *,
    run_id: str,
    store: ConvergenceStore | None = None,
) -> dict[str, Any]:
    return _status_payload(_read_manifest(run_id), store or ConvergenceStore())


__all__ = [
    "ACTIVE_STATUSES",
    "LANE_LIMITS",
    "PROCESSOR_LANES",
    "SUPPORTED_LANES",
    "plan",
    "resume",
    "start",
    "status",
]
