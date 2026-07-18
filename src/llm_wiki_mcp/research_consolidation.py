"""Receipt-gated, proposal-only research consolidation for Sleep."""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping

from llm_wiki_mcp.jsonl_write import append_jsonl_durable
from llm_wiki_mcp.research_config import ResearchConfig, load_research_config
from llm_wiki_mcp.research_store import ResearchStore
from llm_wiki_mcp.wiki import WIKI_ROOT

STATE_FILE = WIKI_ROOT / "runtime" / "research" / "consolidation-state.json"
LOCK_FILE = WIKI_ROOT / "runtime" / "research" / "consolidation.lock"
PROPOSAL_FILE = WIKI_ROOT / "review" / "research-improvement-proposals.jsonl"
ALLOWED_OPERATIONS = frozenset({"query_hint", "alias", "tag"})


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}-", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(dict(payload), handle, ensure_ascii=False, sort_keys=True, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        try:
            os.unlink(temporary)
        except OSError:
            pass


def _load_state(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        return payload if isinstance(payload, dict) else {}
    except (OSError, json.JSONDecodeError):
        return {}


def _receipt_runs(store: ResearchStore, cursor: str) -> list[str]:
    rows: list[tuple[str, str]] = []
    try:
        run_dirs = list(store.runs.iterdir())
    except OSError:
        run_dirs = []
    for run_dir in run_dirs:
        if not run_dir.is_dir():
            continue
        receipts = [event for event in store.events(run_dir.name) if event.get("kind") == "durable_receipt"]
        if not receipts:
            continue
        stamp = str(receipts[-1].get("ts") or "")
        identity = f"{stamp}|{run_dir.name}"
        if identity > cursor:
            rows.append((identity, run_dir.name))
    return [run_id for _identity, run_id in sorted(rows)]


def _existing_latest(path: Path) -> dict[str, str]:
    latest: dict[str, str] = {}
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        lines = []
    for line in lines[-20_000:]:
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(row, dict) and row.get("coalesce_key") and row.get("proposal_id"):
            latest[str(row["coalesce_key"])] = str(row["proposal_id"])
    return latest


def _proposal_rows(
    run_ids: Iterable[str],
    store: ResearchStore,
    *,
    max_jobs: int,
    existing: Mapping[str, str],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for run_id in run_ids:
        audit = next(
            (event.get("audit") for event in reversed(store.events(run_id)) if event.get("kind") == "post_answer_audit"),
            {},
        )
        if not isinstance(audit, Mapping):
            continue
        candidates: list[tuple[str, str, str]] = []
        for claim in audit.get("missing_evidence", []):
            if isinstance(claim, str) and claim.strip():
                candidates.append(("query_hint", "", claim.strip()[:500]))
        for claim in audit.get("unsupported_claim", []):
            if isinstance(claim, str) and claim.strip():
                candidates.append(("tag", "", "needs-verification"))
        for operation, page_id, value in candidates:
            if operation not in ALLOWED_OPERATIONS:
                continue
            coalesce_key = hashlib.sha256(f"{operation}\0{page_id}\0{value}".encode("utf-8")).hexdigest()
            proposal_id = "proposal:" + hashlib.sha256(f"{run_id}\0{coalesce_key}".encode("utf-8")).hexdigest()
            rows.append(
                {
                    "schema_version": 1,
                    "proposal_id": proposal_id,
                    "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                    "research_run_id": run_id,
                    "operation": operation,
                    "page_id": page_id,
                    "value": value,
                    "coalesce_key": coalesce_key,
                    "supersedes": existing.get(coalesce_key),
                    "mutation_mode": "proposal_only",
                    "status": "pending_replay_holdout_consensus",
                }
            )
            if len(rows) >= max_jobs:
                return rows
    return rows


def run_consolidation(
    *,
    config: ResearchConfig | None = None,
    store: ResearchStore | None = None,
    dry_run: bool = False,
    force: bool = False,
    state_path: Path = STATE_FILE,
    lock_path: Path = LOCK_FILE,
    proposal_path: Path = PROPOSAL_FILE,
) -> dict[str, Any]:
    config = config or load_research_config()
    store = store or ResearchStore()
    if not config.consolidation_enabled:
        return {"status": "disabled", "reason": "kill_switch"}
    if config.consolidation_mutation_mode != "proposal_only":
        return {"status": "blocked", "reason": "mutation_mode_not_allowlisted"}
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    handle = lock_path.open("a+")
    try:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return {"status": "busy", "reason": "durable_lease_held"}
        state = _load_state(state_path)
        last_completed = str(state.get("completed_at") or "")
        if last_completed and not force:
            try:
                elapsed = (datetime.now(timezone.utc) - datetime.fromisoformat(last_completed)).total_seconds()
            except ValueError:
                elapsed = config.consolidation_min_interval_seconds
            if elapsed < config.consolidation_min_interval_seconds:
                return {"status": "not_due", "reason": "interval", "elapsed_seconds": round(elapsed)}
        cursor = str(state.get("cursor") or "")
        run_ids = _receipt_runs(store, cursor)
        if not run_ids:
            return {"status": "not_due", "reason": "no_durable_receipts", "new_sessions": 0}
        if len(run_ids) < config.consolidation_min_new_sessions and not force:
            return {"status": "not_due", "reason": "new_sessions", "new_sessions": len(run_ids)}
        rows = _proposal_rows(
            run_ids,
            store,
            max_jobs=config.consolidation_max_jobs,
            existing=_existing_latest(proposal_path),
        )
        if not rows:
            rows = [
                {
                    "schema_version": 1,
                    "proposal_id": f"noop:{hashlib.sha256('|'.join(run_ids).encode()).hexdigest()}",
                    "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                    "operation": "query_hint",
                    "value": "",
                    "coalesce_key": "noop",
                    "mutation_mode": "proposal_only",
                    "status": "no_improvement_needed",
                    "research_run_ids": run_ids,
                }
            ]
        if dry_run:
            return {"status": "dry_run", "new_sessions": len(run_ids), "proposals": len(rows), "rows": rows}
        # Cursor is deliberately written only after the durable proposal receipt.
        append_jsonl_durable(proposal_path, rows, sort_keys=True)
        receipt_id = max(
            f"{str(event.get('ts') or '')}|{run_id}"
            for run_id in run_ids
            for event in store.events(run_id)
            if event.get("kind") == "durable_receipt"
        )
        completed = datetime.now(timezone.utc).isoformat(timespec="seconds")
        _atomic_json(
            state_path,
            {
                "schema_version": 1,
                "cursor": receipt_id,
                "completed_at": completed,
                "new_sessions": len(run_ids),
                "proposals": len(rows),
            },
        )
        return {
            "status": "ok",
            "new_sessions": len(run_ids),
            "proposals": len(rows),
            "cursor": receipt_id,
            "mutation_mode": "proposal_only",
        }
    finally:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        finally:
            handle.close()
