"""Autonomous self-healing loop for LLM Wiki failure packets."""
from __future__ import annotations

import argparse
import fcntl
import json
import os
import shutil
import subprocess
import sys
import tempfile
import uuid
from contextlib import contextmanager
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

from llm_wiki_mcp import runtime_status, wiki
from llm_wiki_mcp.alias_store import add_alias
from llm_wiki_mcp.local_repair import LocalRepairDecision, propose_repair


SELF_HEAL_STATUSES = {
    "pending_local_repair",
    "local_repair_failed",
    "pending_frontier",
    "frontier_retry",
    "frontier_preflight_failed",
    "pending_frontier_review",
}

RUNNING_STATUSES = {
    "local_repairing",
    "frontier_running",
}

DEFAULT_RUNNING_LEASE_SECONDS = 2 * 60 * 60

HUMAN_REQUIRED_STATUSES = {
    "human_required",
}

PENDING_REVIEW_STATUSES = {
    "frontier_preflight_failed",
    "pending_frontier_review",
}

FRONTIER_ONLY_STATUSES = {
    "pending_frontier",
    "frontier_retry",
    "frontier_preflight_failed",
    "pending_frontier_review",
    "frontier_running",
}

MAC_NOTIFICATION_TITLE = "LLM Wiki 自己修復"
MAC_NOTIFICATION_COOLDOWN_SECONDS = 3600


def _repo_root() -> Path:
    configured = os.environ.get("LLM_WIKI_REPO_ROOT")
    if configured:
        return Path(configured)
    return Path(__file__).resolve().parents[2]


def _failures_dir() -> Path:
    return wiki.WIKI_ROOT / "runtime" / "failures"


def _packet_dir() -> Path:
    return _failures_dir() / "packets"


def _local_repair_dir() -> Path:
    return _failures_dir() / "local-repair"


def _frontier_queue_dir() -> Path:
    return _failures_dir() / "frontier-queue"


def _frontier_decision_dir() -> Path:
    return _failures_dir() / "frontier-decisions"


def _pending_frontier_review_dir() -> Path:
    return _failures_dir() / "pending-frontier-review"


def _applied_actions_dir() -> Path:
    return _failures_dir() / "applied-actions"


def _rejected_actions_dir() -> Path:
    return _failures_dir() / "rejected-actions"


def _registry_file() -> Path:
    return _failures_dir() / "failure-registry.jsonl"


def _notification_file() -> Path:
    return _failures_dir() / "notifications.json"


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
    )
    tmp = Path(tmp_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(data, handle, indent=2, ensure_ascii=False, default=str)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, path)
    except Exception:
        try:
            tmp.unlink()
        except OSError:
            pass
        raise


@contextmanager
def _packet_lock(packet_path: Path):
    """Acquire a non-blocking process lock for one failure packet."""

    lock_dir = _failures_dir() / "locks"
    lock_dir.mkdir(parents=True, exist_ok=True)
    lock_path = lock_dir / f"{packet_path.name}.lock"
    with lock_path.open("a+", encoding="utf-8") as handle:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            yield False
            return
        try:
            yield True
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def _append_registry(record: dict[str, Any]) -> None:
    path = _registry_file()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")


def _read_notification_state() -> dict[str, Any]:
    path = _notification_file()
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {"notifications": {}}
    if not isinstance(data, dict) or not isinstance(data.get("notifications"), dict):
        return {"notifications": {}}
    return data


def _write_notification_state(state: dict[str, Any]) -> None:
    _write_json(_notification_file(), state)


def _parse_iso(value: object) -> datetime | None:
    if not isinstance(value, str):
        return None
    try:
        return datetime.fromisoformat(value)
    except ValueError:
        return None


def _comparable_datetimes(left: datetime, right: datetime) -> tuple[datetime, datetime]:
    """Normalize legacy naive timestamps against newer aware timestamps."""

    if left.tzinfo is not None and right.tzinfo is None:
        left = left.replace(tzinfo=None)
    elif left.tzinfo is None and right.tzinfo is not None:
        right = right.replace(tzinfo=None)
    return left, right


def _running_lease_seconds() -> int:
    try:
        return max(
            0,
            int(
                os.environ.get(
                    "LLM_WIKI_SELF_HEAL_RUNNING_LEASE_SECONDS",
                    DEFAULT_RUNNING_LEASE_SECONDS,
                )
            ),
        )
    except ValueError:
        return DEFAULT_RUNNING_LEASE_SECONDS


def _lease_updates(owner: str, *, now: datetime | None = None) -> dict[str, Any]:
    current = now or datetime.now()
    return {
        "lease_owner": owner,
        "lease_expires_at": (
            current + timedelta(seconds=_running_lease_seconds())
        ).isoformat(timespec="seconds"),
    }


def _running_lease_expired(
    packet: dict[str, Any],
    *,
    now: datetime | None = None,
) -> bool:
    current = now or datetime.now()
    expiry = _parse_iso(packet.get("lease_expires_at"))
    if expiry is None:
        started = _parse_iso(packet.get("updated_at")) or _parse_iso(
            packet.get("last_attempt_at")
        )
        if started is None:
            return True
        expiry = started + timedelta(seconds=_running_lease_seconds())
    expiry, current = _comparable_datetimes(expiry, current)
    return expiry <= current


def _notification_cooldown_seconds() -> int:
    try:
        return int(
            os.environ.get(
                "LLM_WIKI_MAC_NOTIFICATION_COOLDOWN_SECONDS",
                MAC_NOTIFICATION_COOLDOWN_SECONDS,
            )
        )
    except ValueError:
        return MAC_NOTIFICATION_COOLDOWN_SECONDS


def _notification_key(packet: dict[str, Any], frontier_result: dict[str, Any]) -> str:
    failure = (
        frontier_result.get("frontier_failure")
        if isinstance(frontier_result.get("frontier_failure"), dict)
        else {}
    )
    return ":".join(
        str(part or "unknown")
        for part in (
            packet.get("fingerprint") or packet.get("failure_id") or packet.get("raw_file"),
            failure.get("failure_class") or frontier_result.get("rescue_status"),
        )
    )


def _human_notification_body(packet: dict[str, Any], frontier_result: dict[str, Any]) -> str:
    failure = (
        frontier_result.get("frontier_failure")
        if isinstance(frontier_result.get("frontier_failure"), dict)
        else {}
    )
    failure_class = failure.get("failure_class") or "frontier"
    if failure_class in {"auth_required", "oauth_required"}:
        return "Codex の認証が切れている可能性があります。ログイン確認が必要です。"
    if failure_class == "quota_or_billing_required":
        return "Codex の quota または billing の確認が必要です。"
    if failure_class == "keychain_permission_required":
        return "Codex の Keychain アクセス許可が必要です。"
    if failure_class == "frontier_tool_unavailable":
        return "Codex CLI が見つかりません。インストール状態の確認が必要です。"
    if failure_class == "both_frontiers_unavailable":
        return "Codex と Claude Code の両方を呼び出せません。手動確認が必要です。"
    raw_file = packet.get("raw_file")
    if isinstance(raw_file, str) and raw_file:
        return f"自己修復に人間の確認が必要です: {Path(raw_file).name}"
    return "自己修復に人間の確認が必要です。"


def _send_mac_notification(title: str, body: str) -> dict[str, Any]:
    if os.environ.get("LLM_WIKI_MAC_NOTIFICATIONS", "1") in {"0", "false", "False"}:
        return {"sent": False, "reason": "disabled"}
    osascript = "/usr/bin/osascript"
    if not Path(osascript).exists():
        found = shutil.which("osascript")
        if found is None:
            return {"sent": False, "reason": "osascript not found"}
        osascript = found
    script = f"display notification {json.dumps(body)} with title {json.dumps(title)}"
    try:
        completed = subprocess.run(
            [osascript, "-e", script],
            text=True,
            capture_output=True,
            timeout=5,
        )
    except Exception as exc:
        return {"sent": False, "reason": str(exc)}
    return {
        "sent": completed.returncode == 0,
        "exit_code": completed.returncode,
        "stderr": (completed.stderr or "")[-500:],
    }


def maybe_notify_human_required(
    packet: dict[str, Any],
    frontier_result: dict[str, Any],
    *,
    now: datetime | None = None,
) -> dict[str, Any]:
    if not frontier_result.get("human_required") and not frontier_result.get("notify_user"):
        return {"sent": False, "reason": "not human required"}
    now = now or datetime.now()
    key = _notification_key(packet, frontier_result)
    state = _read_notification_state()
    notifications = state.setdefault("notifications", {})
    prior = notifications.get(key) if isinstance(notifications.get(key), dict) else {}
    prior_at = _parse_iso(prior.get("last_notified_at"))
    cooldown = _notification_cooldown_seconds()
    if prior_at is not None and prior_at.tzinfo is not None and now.tzinfo is None:
        prior_at = prior_at.replace(tzinfo=None)
    if prior_at is not None and prior_at.tzinfo is None and now.tzinfo is not None:
        now = now.replace(tzinfo=None)
    if prior_at is not None and (now - prior_at).total_seconds() < cooldown:
        return {
            "sent": False,
            "reason": "cooldown",
            "key": key,
            "last_notified_at": prior_at.isoformat(),
            "cooldown_seconds": cooldown,
        }
    body = _human_notification_body(packet, frontier_result)
    delivery = _send_mac_notification(MAC_NOTIFICATION_TITLE, body)
    record = {
        "last_notified_at": now.isoformat(),
        "title": MAC_NOTIFICATION_TITLE,
        "body": body,
        "delivery": delivery,
    }
    notifications[key] = record
    _write_notification_state(state)
    return {"key": key, **record}


def _update_packet(path: Path, packet: dict[str, Any], **updates: Any) -> None:
    next_status = updates.get("status", packet.get("status"))
    if next_status not in RUNNING_STATUSES:
        updates.setdefault("lease_owner", None)
        updates.setdefault("lease_expires_at", None)
    packet.update(updates)
    packet["updated_at"] = datetime.now().isoformat()
    _write_json(path, packet)


def pending_packets(*, now: datetime | None = None) -> list[Path]:
    if not _packet_dir().exists():
        return []
    current = now or datetime.now()
    out: list[Path] = []
    for path in sorted(_packet_dir().glob("*.json")):
        try:
            packet = _read_json(path)
        except Exception:
            continue
        next_attempt = _parse_iso(packet.get("next_attempt_at"))
        due = True
        if next_attempt is not None:
            next_attempt, comparable_now = _comparable_datetimes(next_attempt, current)
            due = next_attempt <= comparable_now
        status = packet.get("status")
        if status in SELF_HEAL_STATUSES and due:
            out.append(path)
        elif status in RUNNING_STATUSES and _running_lease_expired(packet, now=current):
            out.append(path)
    return out


def _raw_candidate_paths(packet: dict[str, Any]) -> list[Path]:
    raw_file = packet.get("raw_file")
    if not isinstance(raw_file, str) or not raw_file:
        return []
    return [
        _failures_dir() / "quarantined-raw" / raw_file,
        wiki.RAW_DIR / raw_file,
    ]


def _restore_quarantined_raw(packet: dict[str, Any], *, dry_run: bool) -> dict[str, Any]:
    candidates = _raw_candidate_paths(packet)
    if not candidates:
        return {"restored": False, "reason": "packet has no raw_file"}
    quarantine_path, raw_path = candidates[0], candidates[1]
    if raw_path.exists():
        return {"restored": False, "reason": "raw already pending", "path": str(raw_path)}
    if not quarantine_path.exists():
        return {"restored": False, "reason": "quarantined raw not found"}
    if dry_run:
        return {
            "restored": False,
            "dry_run": True,
            "source": str(quarantine_path),
            "target": str(raw_path),
        }
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(quarantine_path), str(raw_path))
    return {"restored": True, "source": str(quarantine_path), "target": str(raw_path)}


def _retry_ingest(*, dry_run: bool) -> dict[str, Any]:
    if dry_run:
        return {"triggered": False, "dry_run": True}
    from llm_wiki_mcp import orchestrator

    return orchestrator.run_pending_ingest(force=True)


def apply_local_decision(
    packet: dict[str, Any],
    decision: LocalRepairDecision,
    *,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Apply a whitelisted local repair action."""

    if decision.action == "resolve_update_target":
        if not decision.requested_page_id or not decision.target_page_id:
            raise ValueError("resolve_update_target requires requested and target ids")
        if not dry_run:
            add_alias(
                decision.requested_page_id,
                decision.target_page_id,
                source=packet.get("failure_id"),
            )
        restore = _restore_quarantined_raw(packet, dry_run=dry_run)
        retry = _retry_ingest(dry_run=dry_run)
        return {
            "action": decision.action,
            "alias": {
                "requested": decision.requested_page_id,
                "target": decision.target_page_id,
                "dry_run": dry_run,
            },
            "restore": restore,
            "retry": retry,
        }

    if decision.action == "retry_raw":
        restore = _restore_quarantined_raw(packet, dry_run=dry_run)
        retry = _retry_ingest(dry_run=dry_run)
        return {"action": decision.action, "restore": restore, "retry": retry}

    if decision.action == "quarantine_raw":
        return {"action": decision.action, "kept_quarantined": True}

    raise ValueError(f"local action requires frontier or is not directly applicable: {decision.action}")


def _save_local_decision(packet_path: Path, decision: LocalRepairDecision) -> Path:
    path = _local_repair_dir() / packet_path.name
    _write_json(path, decision.to_dict())
    return path


def _save_action(packet_path: Path, action: dict[str, Any], *, applied: bool) -> Path:
    target_dir = _applied_actions_dir() if applied else _rejected_actions_dir()
    path = target_dir / packet_path.name
    _write_json(path, action)
    return path


def _queue_frontier(packet_path: Path, packet: dict[str, Any], decision: dict[str, Any] | None) -> Path:
    target = _frontier_queue_dir() / packet_path.name
    payload = {
        "queued_at": datetime.now().isoformat(),
        "packet_path": str(packet_path),
        "packet": packet,
        "local_decision": decision,
    }
    _write_json(target, payload)
    return target


def _run_frontier(
    packet_path: Path,
    packet: dict[str, Any],
    local_decision: dict[str, Any] | None,
    *,
    execute_patch: bool,
) -> dict[str, Any]:
    from llm_wiki_mcp.frontier_review import run_frontier_review

    result = run_frontier_review(
        packet,
        local_decision,
        repo_root=_repo_root(),
        execute_patch=execute_patch,
    )
    payload = result.to_dict()
    _write_json(_frontier_decision_dir() / packet_path.name, payload)
    return payload


def _save_pending_frontier_review(
    packet_path: Path,
    packet: dict[str, Any],
    local_decision: dict[str, Any] | None,
    frontier_result: dict[str, Any],
    *,
    status: str,
) -> Path:
    payload = {
        "queued_at": datetime.now().isoformat(),
        "status": status,
        "packet_path": str(packet_path),
        "packet": packet,
        "local_decision": local_decision,
        "frontier_result": frontier_result,
        "access_repair": frontier_result.get("access_repair"),
        "rescue_attempt": frontier_result.get("rescue_attempt"),
    }
    path = _pending_frontier_review_dir() / packet_path.name
    _write_json(path, payload)
    return path


def _budget_deferred_result(
    packet_path: Path,
    packet: dict[str, Any],
    *,
    kind: str,
    reason: str,
    local_decision: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Return a no-progress budget deferral for pre-attempt gates."""

    result: dict[str, Any] = {
        "packet": str(packet_path),
        "failure_id": packet.get("failure_id"),
        "status": "budget_deferred",
        "budget_kind": kind,
        "reason": reason,
    }
    if local_decision is not None:
        result["local_decision"] = local_decision
    return result


def _frontier_final_status(frontier_result: dict[str, Any]) -> str:
    if frontier_result.get("decision") == "approved":
        return "frontier_approved"
    if frontier_result.get("human_required"):
        return "human_required"
    rescue_status = frontier_result.get("rescue_status")
    if rescue_status in PENDING_REVIEW_STATUSES:
        return str(rescue_status)
    if frontier_result.get("decision") == "needs_retry":
        return "frontier_retry"
    if frontier_result.get("decision") == "quarantined":
        return "frontier_quarantined"
    return "frontier_rejected"


def _handle_packet_unlocked(
    packet_path: Path,
    *,
    use_qwen: bool = True,
    enable_frontier: bool = True,
    execute_frontier_patch: bool = True,
    dry_run: bool = False,
    max_attempts: int = 3,
    backoff_base_seconds: int = 6 * 60 * 60,
    frontier_budget: Any | None = None,
) -> dict[str, Any]:
    packet = _read_json(packet_path)
    frontier_only = packet.get("status") in FRONTIER_ONLY_STATUSES
    decision: LocalRepairDecision | None = None
    persisted_decision = packet.get("local_decision")
    local_decision = dict(persisted_decision) if isinstance(persisted_decision, dict) else None
    prior_frontier_attempts = int(
        packet.get("frontier_attempts")
        if packet.get("frontier_attempts") is not None
        else packet.get("self_heal_attempts")
        or 0
    )
    max_frontier_attempts = max(1, max_attempts)
    will_apply_local = False
    frontier_reserved = False
    mutation_reserved = False

    if frontier_only:
        if dry_run:
            return {
                "packet": str(packet_path),
                "failure_id": packet.get("failure_id"),
                "status": "dry_run",
                "local_decision": local_decision,
                "frontier_only": True,
            }
    else:
        if not dry_run and frontier_budget is not None:
            allowed, reason = frontier_budget.consume("local")
            if not allowed:
                return _budget_deferred_result(
                    packet_path,
                    packet,
                    kind="local",
                    reason=reason,
                )
        decision = propose_repair(packet, use_qwen=use_qwen)
        local_decision = decision.to_dict()
        if dry_run:
            return {
                "packet": str(packet_path),
                "failure_id": packet.get("failure_id"),
                "status": "dry_run",
                "local_decision": local_decision,
            }

        will_apply_local = decision.status == "resolved" and decision.action in {
            "resolve_update_target",
            "retry_raw",
            "quarantine_raw",
        }
        if will_apply_local and frontier_budget is not None:
            allowed, reason = frontier_budget.consume("mutation")
            if not allowed:
                return _budget_deferred_result(
                    packet_path,
                    packet,
                    kind="mutation",
                    reason=reason,
                    local_decision=local_decision,
                )
            mutation_reserved = True

    routes_directly_to_frontier = frontier_only or not will_apply_local
    if (
        routes_directly_to_frontier
        and enable_frontier
        and prior_frontier_attempts < max_frontier_attempts
        and frontier_budget is not None
    ):
        frontier_allowed, frontier_reason = frontier_budget.can_consume("frontier")
        needs_mutation = execute_frontier_patch and not dry_run and not mutation_reserved
        mutation_allowed, mutation_reason = (
            frontier_budget.can_consume("mutation")
            if needs_mutation
            else (True, "ok")
        )
        if not frontier_allowed or not mutation_allowed:
            return _budget_deferred_result(
                packet_path,
                packet,
                kind="frontier" if not frontier_allowed else "mutation",
                reason=frontier_reason if not frontier_allowed else mutation_reason,
                local_decision=local_decision,
            )
        frontier_budget.consume("frontier")
        if needs_mutation:
            frontier_budget.consume("mutation")
            mutation_reserved = True
        frontier_reserved = True

    lease_owner = uuid.uuid4().hex
    result: dict[str, Any] = {
        "packet": str(packet_path),
        "failure_id": packet.get("failure_id"),
        "local_decision": local_decision,
    }

    if not frontier_only:
        assert decision is not None
        local_attempt = int(packet.get("local_repair_attempts") or 0) + 1
        _update_packet(
            packet_path,
            packet,
            status="local_repairing",
            local_repair_attempts=local_attempt,
            local_decision=local_decision,
            last_attempt_at=datetime.now().isoformat(timespec="seconds"),
            next_attempt_at=None,
            **_lease_updates(lease_owner),
        )
        decision_path = _save_local_decision(packet_path, decision)
        result["local_decision_path"] = str(decision_path)

        try:
            if will_apply_local:
                action = apply_local_decision(packet, decision, dry_run=False)
                action_path = _save_action(packet_path, action, applied=True)
                _update_packet(
                    packet_path,
                    packet,
                    status="local_repair_applied",
                    local_decision=local_decision,
                    applied_action_path=str(action_path),
                )
                _append_registry({
                    "timestamp": datetime.now().isoformat(),
                    "failure_id": packet.get("failure_id"),
                    "raw_file": packet.get("raw_file"),
                    "failure_class": packet.get("failure_class"),
                    "fingerprint": packet.get("fingerprint"),
                    "resolution": "local",
                    "decision": local_decision,
                    "action": action,
                })
                runtime_status.safe_append_event(
                    "success",
                    f"self-heal | local repair applied for {packet.get('raw_file')}",
                    source="self-heal",
                    packet=str(packet_path),
                    action=decision.action,
                )
                result["status"] = "local_repair_applied"
                result["action"] = action
                return result
        except Exception as exc:
            action = {
                "action": decision.action,
                "error": str(exc),
                "decision": local_decision,
            }
            _save_action(packet_path, action, applied=False)
            _update_packet(
                packet_path,
                packet,
                status="local_repair_failed",
                local_decision=local_decision,
                local_error=str(exc),
            )
            result["local_error"] = str(exc)

    queue_path = _queue_frontier(packet_path, packet, local_decision)
    result["frontier_queue_path"] = str(queue_path)
    if not enable_frontier:
        delay = max(0, backoff_base_seconds) * (2 ** max(0, prior_frontier_attempts))
        _update_packet(
            packet_path,
            packet,
            status="pending_frontier",
            local_decision=local_decision,
            frontier_queue_path=str(queue_path),
            next_attempt_at=(datetime.now() + timedelta(seconds=delay)).isoformat(timespec="seconds"),
        )
        result["status"] = "pending_frontier"
        return result

    if prior_frontier_attempts >= max_frontier_attempts:
        _update_packet(
            packet_path,
            packet,
            status="frontier_quarantined",
            local_decision=local_decision,
            frontier_queue_path=str(queue_path),
            next_attempt_at=None,
            frontier_error="frontier attempt limit reached before execution",
        )
        result["status"] = "frontier_quarantined"
        result["reason"] = "frontier_attempt_limit_reached"
        return result

    if frontier_budget is not None and not frontier_reserved:
        frontier_allowed, frontier_reason = frontier_budget.can_consume("frontier")
        needs_mutation = execute_frontier_patch and not dry_run and not mutation_reserved
        mutation_allowed, mutation_reason = (
            frontier_budget.can_consume("mutation")
            if needs_mutation
            else (True, "ok")
        )
        if not frontier_allowed or not mutation_allowed:
            return _budget_deferred_result(
                packet_path,
                packet,
                kind="frontier" if not frontier_allowed else "mutation",
                reason=frontier_reason if not frontier_allowed else mutation_reason,
                local_decision=local_decision,
            )
        frontier_budget.consume("frontier")
        if needs_mutation:
            frontier_budget.consume("mutation")
            mutation_reserved = True

    attempt = prior_frontier_attempts + 1
    _update_packet(
        packet_path,
        packet,
        status="frontier_running",
        frontier_attempts=attempt,
        # Keep the legacy aggregate field in sync for old dashboards and
        # packets, but never increment it before a real frontier execution.
        self_heal_attempts=attempt,
        local_decision=local_decision,
        frontier_queue_path=str(queue_path),
        **_lease_updates(lease_owner),
    )
    try:
        frontier_result = _run_frontier(
            packet_path,
            packet,
            local_decision,
            execute_patch=execute_frontier_patch and not dry_run,
        )
    except Exception as exc:
        final_status = (
            "frontier_quarantined"
            if attempt >= max_frontier_attempts
            else "frontier_retry"
        )
        delay = max(0, backoff_base_seconds) * (2 ** max(0, attempt - 1))
        next_attempt_at = (
            None
            if final_status == "frontier_quarantined"
            else (datetime.now() + timedelta(seconds=delay)).isoformat(timespec="seconds")
        )
        frontier_error = {
            "exception_type": exc.__class__.__name__,
            "message": str(exc),
        }
        _update_packet(
            packet_path,
            packet,
            status=final_status,
            local_decision=local_decision,
            frontier_queue_path=str(queue_path),
            frontier_error=frontier_error,
            next_attempt_at=next_attempt_at,
        )
        _append_registry(
            {
                "timestamp": datetime.now().isoformat(),
                "failure_id": packet.get("failure_id"),
                "raw_file": packet.get("raw_file"),
                "failure_class": packet.get("failure_class"),
                "fingerprint": packet.get("fingerprint"),
                "resolution": "frontier_error",
                "decision": local_decision,
                "frontier_error": frontier_error,
                "status": final_status,
            }
        )
        runtime_status.safe_append_event(
            "warn",
            f"self-heal | frontier exception for {packet.get('raw_file')}",
            source="self-heal",
            packet=str(packet_path),
            frontier_status=final_status,
            frontier_error=frontier_error,
        )
        result["status"] = final_status
        result["frontier_error"] = frontier_error
        return result
    final_status = _frontier_final_status(frontier_result)
    next_attempt_at = None
    if final_status in {"frontier_retry", *PENDING_REVIEW_STATUSES}:
        if attempt >= max(1, max_attempts):
            final_status = "frontier_quarantined"
        else:
            delay = max(0, backoff_base_seconds) * (2 ** max(0, attempt - 1))
            next_attempt_at = (datetime.now() + timedelta(seconds=delay)).isoformat(timespec="seconds")
    human_notification = None
    pending_review_path = None
    if final_status == "human_required" and not dry_run:
        human_notification = maybe_notify_human_required(packet, frontier_result)
    if final_status in PENDING_REVIEW_STATUSES:
        pending_review_path = _save_pending_frontier_review(
            packet_path,
            packet,
            local_decision,
            frontier_result,
            status=final_status,
        )
    _update_packet(
        packet_path,
        packet,
        status=final_status,
        frontier_result=frontier_result,
        human_notification=human_notification,
        pending_frontier_review_path=str(pending_review_path) if pending_review_path else None,
        next_attempt_at=next_attempt_at,
    )
    _append_registry({
        "timestamp": datetime.now().isoformat(),
        "failure_id": packet.get("failure_id"),
        "raw_file": packet.get("raw_file"),
        "failure_class": packet.get("failure_class"),
        "fingerprint": packet.get("fingerprint"),
        "resolution": "frontier",
        "decision": local_decision,
        "frontier": frontier_result,
        "human_notification": human_notification,
        "pending_frontier_review_path": str(pending_review_path) if pending_review_path else None,
    })
    event_level = (
        "success"
        if final_status == "frontier_approved"
        else "error"
        if final_status in HUMAN_REQUIRED_STATUSES
        else "warn"
    )
    event_message = (
        f"self-heal | human required for {packet.get('raw_file')}"
        if final_status == "human_required"
        else f"self-heal | frontier {frontier_result.get('decision')} for {packet.get('raw_file')}"
    )
    runtime_status.safe_append_event(
        event_level,
        event_message,
        source="self-heal",
        packet=str(packet_path),
        frontier_status=final_status,
        human_required=final_status == "human_required",
    )
    result["status"] = final_status
    result["frontier_result"] = frontier_result
    result["human_notification"] = human_notification
    result["pending_frontier_review_path"] = str(pending_review_path) if pending_review_path else None
    return result


def handle_packet(
    packet_path: Path,
    *,
    use_qwen: bool = True,
    enable_frontier: bool = True,
    execute_frontier_patch: bool = True,
    dry_run: bool = False,
    max_attempts: int = 3,
    backoff_base_seconds: int = 6 * 60 * 60,
    frontier_budget: Any | None = None,
) -> dict[str, Any]:
    """Handle one packet with process-safe single-flight execution.

    Dry runs intentionally avoid even creating the sidecar lock file, keeping
    the established byte-for-byte read-only contract. ``frontier_budget``
    retains its public name for compatibility but is the shared cycle budget
    for local, frontier, and mutation work.
    """

    kwargs = {
        "use_qwen": use_qwen,
        "enable_frontier": enable_frontier,
        "execute_frontier_patch": execute_frontier_patch,
        "dry_run": dry_run,
        "max_attempts": max_attempts,
        "backoff_base_seconds": backoff_base_seconds,
        "frontier_budget": frontier_budget,
    }
    if dry_run:
        return _handle_packet_unlocked(packet_path, **kwargs)
    with _packet_lock(packet_path) as acquired:
        if not acquired:
            return {
                "packet": str(packet_path),
                "status": "busy",
                "reason": "packet_already_running",
            }
        current = _read_json(packet_path)
        current_status = current.get("status")
        if current_status in RUNNING_STATUSES and not _running_lease_expired(current):
            return {
                "packet": str(packet_path),
                "failure_id": current.get("failure_id"),
                "status": "busy",
                "reason": "running_lease_active",
            }
        if (
            current_status
            and current_status not in SELF_HEAL_STATUSES
            and current_status not in RUNNING_STATUSES
        ):
            return {
                "packet": str(packet_path),
                "failure_id": current.get("failure_id"),
                "status": current_status,
                "cached": True,
            }
        # The packet is read inside the lock by the implementation.  This is
        # the CAS boundary that prevents a stale pre-lock snapshot from being
        # applied after another worker completes.
        return _handle_packet_unlocked(packet_path, **kwargs)


def run_pending(
    *,
    max_packets: int = 3,
    use_qwen: bool = True,
    enable_frontier: bool = True,
    execute_frontier_patch: bool = True,
    dry_run: bool = False,
    max_attempts: int = 3,
    backoff_base_seconds: int = 6 * 60 * 60,
    frontier_budget: Any | None = None,
) -> dict[str, Any]:
    packets = pending_packets()[:max_packets]
    results: list[dict[str, Any]] = []
    for packet in packets:
        try:
            result = handle_packet(
                packet,
                use_qwen=use_qwen,
                enable_frontier=enable_frontier,
                execute_frontier_patch=execute_frontier_patch,
                dry_run=dry_run,
                max_attempts=max_attempts,
                backoff_base_seconds=backoff_base_seconds,
                frontier_budget=frontier_budget,
            )
        except Exception as exc:
            # One corrupt or externally-failing packet must not abort the
            # bounded drain of unrelated packets.
            result = {
                "packet": str(packet),
                "status": "error",
                "error": f"{exc.__class__.__name__}: {exc}",
            }
        results.append(result)
    return {
        "status": "ok",
        "packets_seen": len(packets),
        "results": results,
    }


def run_auto_apply_error_self_heal(
    *,
    threshold: int = 3,
    max_packets: int = 3,
    use_qwen: bool = True,
    enable_frontier: bool = True,
    execute_frontier_patch: bool = True,
    dry_run: bool = False,
) -> dict[str, Any]:
    from llm_wiki_mcp.auto_apply_error_supervisor import (
        pending_auto_apply_error_packets,
        supervise_auto_apply_log,
    )

    supervision = supervise_auto_apply_log(
        threshold=threshold,
        start_background=False,
        dry_run=dry_run,
    )
    created = [Path(path) for path in supervision.get("packets_created", []) if isinstance(path, str)]
    packets = created or pending_auto_apply_error_packets()
    packets = packets[:max_packets]
    results: list[dict[str, Any]] = []
    for packet in packets:
        try:
            result = handle_packet(
                packet,
                use_qwen=use_qwen,
                enable_frontier=enable_frontier,
                execute_frontier_patch=execute_frontier_patch,
                dry_run=dry_run,
            )
        except Exception as exc:
            result = {
                "packet": str(packet),
                "status": "error",
                "error": f"{exc.__class__.__name__}: {exc}",
            }
        results.append(result)
    return {
        "status": "ok",
        "supervision": supervision,
        "packets_seen": len(packets),
        "results": results,
    }


def start_background(packet_path: Path) -> None:
    """Launch self-heal asynchronously after quarantine."""

    if os.environ.get("LLM_WIKI_SELF_HEAL_AUTORUN", "1") in {"0", "false", "False"}:
        return
    if os.environ.get("PYTEST_CURRENT_TEST"):
        return
    subprocess.Popen(
        [
            sys.executable,
            "-m",
            "llm_wiki_mcp.self_heal",
            "--packet",
            str(packet_path),
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
    )


def drill_packet() -> dict[str, Any]:
    return {
        "failure_id": "drill-update-target-not-found",
        "created_at": datetime.now().isoformat(),
        "raw_file": "drill.md",
        "job_id": "drill",
        "failure_class": "apply.update_target_not_found",
        "fingerprint": (
            "apply.update_target_not_found:"
            "opus-4-7-evaluation-and-industry-geopolitics"
        ),
        "attempts": 3,
        "error": (
            "update target not found for page_id "
            "'opus-4-7-evaluation-and-industry-geopolitics'"
        ),
        "requested_page_id": "opus-4-7-evaluation-and-industry-geopolitics",
        "similar_existing_pages": [
            "ai/opus-4.7-evaluation-and-industry-geopolitics"
        ],
        "status": "pending_local_repair",
    }


def run_drill(*, use_qwen: bool = True) -> dict[str, Any]:
    packet = drill_packet()
    decision = propose_repair(packet, use_qwen=use_qwen)
    return {"packet": packet, "decision": decision.to_dict()}


def _patch_wiki_paths(wiki_root: Path) -> dict[str, Any]:
    """Point path globals at a sandbox wiki for an end-to-end drill."""

    pages = wiki_root / "pages"
    raw = wiki_root / "raw"
    system = wiki_root / "system"
    runtime = wiki_root / "runtime"
    for path in (pages, raw, system, runtime):
        path.mkdir(parents=True, exist_ok=True)

    from llm_wiki_mcp import ingest, orchestrator

    snapshot = {
        "wiki": {
            "WIKI_ROOT": wiki.WIKI_ROOT,
            "PAGES_DIR": wiki.PAGES_DIR,
            "RAW_DIR": wiki.RAW_DIR,
            "SYSTEM_DIR": wiki.SYSTEM_DIR,
            "INDEX_FILE": wiki.INDEX_FILE,
            "LOG_FILE": wiki.LOG_FILE,
        },
        "ingest": {
            "PAGES_DIR": ingest.PAGES_DIR,
            "INDEX_FILE": ingest.INDEX_FILE,
            "LOG_FILE": ingest.LOG_FILE,
        },
        "orchestrator": {
            "RAW_DIR": orchestrator.RAW_DIR,
            "WIKI_ROOT": orchestrator.WIKI_ROOT,
            "LOG_FILE": orchestrator.LOG_FILE,
            "STATE_FILE": orchestrator.STATE_FILE,
        },
        "runtime_status": {
            "RUNTIME_DIR": runtime_status.RUNTIME_DIR,
            "STATUS_FILE": runtime_status.STATUS_FILE,
            "EVENTS_FILE": runtime_status.EVENTS_FILE,
            "METRICS_FILE": runtime_status.METRICS_FILE,
        },
    }

    wiki.WIKI_ROOT = wiki_root
    wiki.PAGES_DIR = pages
    wiki.RAW_DIR = raw
    wiki.SYSTEM_DIR = system
    wiki.INDEX_FILE = wiki_root / "index.md"
    wiki.LOG_FILE = wiki_root / "log.md"

    ingest.PAGES_DIR = pages
    ingest.INDEX_FILE = wiki_root / "index.md"
    ingest.LOG_FILE = wiki_root / "log.md"
    orchestrator.RAW_DIR = raw
    orchestrator.WIKI_ROOT = wiki_root
    orchestrator.LOG_FILE = wiki_root / "log.md"
    orchestrator.STATE_FILE = wiki_root / ".orchestrator_state.json"

    runtime_status.RUNTIME_DIR = runtime
    runtime_status.STATUS_FILE = runtime / "status.json"
    runtime_status.EVENTS_FILE = runtime / "events.jsonl"
    runtime_status.METRICS_FILE = runtime / "metrics.jsonl"
    return snapshot


def _restore_wiki_paths(snapshot: dict[str, Any]) -> None:
    """Restore path globals after a sandbox drill."""

    from llm_wiki_mcp import ingest, orchestrator

    for name, value in snapshot["wiki"].items():
        setattr(wiki, name, value)
    for name, value in snapshot["ingest"].items():
        setattr(ingest, name, value)
    for name, value in snapshot["orchestrator"].items():
        setattr(orchestrator, name, value)
    for name, value in snapshot["runtime_status"].items():
        setattr(runtime_status, name, value)


def run_sandbox_drill(*, use_qwen: bool = True) -> dict[str, Any]:
    """Exercise pending raw -> failure packet -> self-heal -> retry success."""

    sandbox_root = Path(tempfile.mkdtemp(prefix="llm-wiki-self-heal-drill-"))
    path_snapshot = _patch_wiki_paths(sandbox_root)

    page = sandbox_root / "pages" / "ai" / "opus-4.7-evaluation-and-industry-geopolitics.md"
    page.parent.mkdir(parents=True, exist_ok=True)
    page.write_text("---\ntitle: Opus\nupdated: 2026-01-01\n---\nold\n", encoding="utf-8")
    raw_path = sandbox_root / "raw" / "broken.md"
    raw_path.write_text("sandbox drill raw\n", encoding="utf-8")

    old_autorun = os.environ.get("LLM_WIKI_SELF_HEAL_AUTORUN")
    os.environ["LLM_WIKI_SELF_HEAL_AUTORUN"] = "0"

    from llm_wiki_mcp import ingest as ingest_mod, orchestrator
    from llm_wiki_mcp.alias_store import load_aliases
    from llm_wiki_mcp.jobs import JobStatus, job_store

    original_run_ingest = ingest_mod.run_ingest

    def fake_run_ingest(
        content, job_id, on_complete=None, on_finally=None, *, metadata=None
    ):
        aliases = load_aliases()
        if (
            aliases.get("opus-4-7-evaluation-and-industry-geopolitics")
            == "ai/opus-4.7-evaluation-and-industry-geopolitics"
        ):
            job_store.update(job_id, status=JobStatus.COMPLETED)
            if on_complete:
                on_complete()
            if on_finally:
                on_finally(failed=False, triage_failed=False)
            return
        job_store.update(
            job_id,
            status=JobStatus.FAILED,
            error=(
                "update target not found for page_id "
                "'opus-4-7-evaluation-and-industry-geopolitics'"
            ),
        )
        if on_finally:
            on_finally(failed=True, triage_failed=False)

    ingest_mod.run_ingest = fake_run_ingest
    try:
        batches = [orchestrator.run_pending_ingest(force=True) for _ in range(3)]
        packet_paths = sorted((_packet_dir()).glob("*.json"))
        heal_result = None
        if packet_paths:
            heal_result = handle_packet(
                packet_paths[0],
                use_qwen=use_qwen,
                enable_frontier=False,
                dry_run=False,
            )
        pending_after = [p.name for p in orchestrator.get_pending_raw_files()]
        aliases = load_aliases()
    finally:
        ingest_mod.run_ingest = original_run_ingest
        if old_autorun is None:
            os.environ.pop("LLM_WIKI_SELF_HEAL_AUTORUN", None)
        else:
            os.environ["LLM_WIKI_SELF_HEAL_AUTORUN"] = old_autorun
        _restore_wiki_paths(path_snapshot)

    return {
        "status": "ok",
        "sandbox_root": str(sandbox_root),
        "batches": batches,
        "packet_paths": [str(p) for p in packet_paths],
        "heal_result": heal_result,
        "pending_after": pending_after,
        "aliases": aliases,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run LLM Wiki self-healing.")
    parser.add_argument("--packet", type=Path, help="Process one failure packet.")
    parser.add_argument("--max-packets", type=int, default=3)
    parser.add_argument(
        "--auto-apply-errors",
        action="store_true",
        help="Promote repeated recall auto-apply errors into self-heal packets.",
    )
    parser.add_argument("--auto-apply-error-threshold", type=int, default=3)
    parser.add_argument("--no-qwen", action="store_true")
    parser.add_argument("--no-frontier", action="store_true")
    parser.add_argument("--review-only", action="store_true", help="Frontier may review but not patch.")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--drill", action="store_true", help="Run a synthetic local repair drill.")
    parser.add_argument(
        "--sandbox-drill",
        action="store_true",
        help="Run a sandbox pending-raw self-heal drill without touching production wiki.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.sandbox_drill:
        print(
            json.dumps(
                run_sandbox_drill(use_qwen=not args.no_qwen),
                ensure_ascii=False,
                indent=2,
                default=str,
            )
        )
        return 0
    if args.drill:
        print(json.dumps(run_drill(use_qwen=not args.no_qwen), ensure_ascii=False, indent=2))
        return 0
    if args.auto_apply_errors:
        result = run_auto_apply_error_self_heal(
            threshold=args.auto_apply_error_threshold,
            max_packets=args.max_packets,
            use_qwen=not args.no_qwen,
            enable_frontier=not args.no_frontier,
            execute_frontier_patch=not args.review_only,
            dry_run=args.dry_run,
        )
        print(json.dumps(result, ensure_ascii=False, indent=2, default=str))
        return 0
    if args.packet:
        result = handle_packet(
            args.packet,
            use_qwen=not args.no_qwen,
            enable_frontier=not args.no_frontier,
            execute_frontier_patch=not args.review_only,
            dry_run=args.dry_run,
        )
    else:
        result = run_pending(
            max_packets=args.max_packets,
            use_qwen=not args.no_qwen,
            enable_frontier=not args.no_frontier,
            execute_frontier_patch=not args.review_only,
            dry_run=args.dry_run,
        )
    print(json.dumps(result, ensure_ascii=False, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
