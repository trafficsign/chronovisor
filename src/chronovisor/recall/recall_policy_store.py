"""Policy persistence helpers for recall self-improvement."""

from __future__ import annotations

import json
import os
from collections.abc import MutableMapping
from pathlib import Path
from typing import Any

from chronovisor.core.jsonl import read_jsonl as _strict_read_jsonl
from chronovisor.core.store import CHRONOVISOR_ROOT
from chronovisor.decision.recall_policy_contract import (
    ALLOWED_POLICY_FIELDS as ALLOWED_POLICY_FIELDS,
)
from chronovisor.decision.recall_policy_contract import FALSE_VALUES as FALSE_VALUES
from chronovisor.decision.recall_policy_contract import (
    apply_policy_overrides as apply_policy_overrides,
)
from chronovisor.decision.recall_policy_contract import (
    normalize_policy_overrides as normalize_policy_overrides,
)
from chronovisor.decision.recall_policy_contract import (
    policy_snapshot as policy_snapshot,
)

IMPROVEMENT_DIR = CHRONOVISOR_ROOT / "runtime" / "recall-improvement"
ACTIVE_POLICY_FILE = IMPROVEMENT_DIR / "active-policy.json"
REGISTRY_FILE = IMPROVEMENT_DIR / "policy-registry.jsonl"
EPISODES_FILE = IMPROVEMENT_DIR / "recall-episodes.jsonl"
LIVE_EPISODES_FILE = IMPROVEMENT_DIR / "live-episodes.jsonl"
SCHEDULE_FILE = IMPROVEMENT_DIR / "schedule-state.json"
RUNS_DIR = IMPROVEMENT_DIR / "runs"
FRONTIER_AUDIT_DIR = IMPROVEMENT_DIR / "frontier-audits"

def improvement_policy_enabled() -> bool:
    try:
        from chronovisor.recall.recall_distillation import distillation_enabled
    except ImportError:
        distillation_enabled = None
    if distillation_enabled is not None:
        try:
            if distillation_enabled():
                return False
        except Exception:
            # A broken cutover config must not re-enable a retired policy.
            return False
    value = os.environ.get("CHRONOVISOR_RECALL_IMPROVEMENT_POLICY")
    return value not in FALSE_VALUES


def read_json_file(path: Path) -> dict[str, Any]:
    try:
        if not path.exists():
            return {}
        parsed = json.loads(path.read_text(encoding="utf-8"))
        return parsed if isinstance(parsed, dict) else {}
    except (OSError, json.JSONDecodeError):
        return {}


def read_active_policy(path: Path = ACTIVE_POLICY_FILE) -> dict[str, Any]:
    return read_json_file(path)


def active_policy_overrides(path: Path = ACTIVE_POLICY_FILE) -> dict[str, Any]:
    data = read_active_policy(path)
    return normalize_policy_overrides(data.get("overrides"))


def apply_active_policy(policy: Any, path: Path = ACTIVE_POLICY_FILE) -> list[str]:
    if not improvement_policy_enabled():
        return []
    return apply_policy_overrides(policy, active_policy_overrides(path))


def atomic_write_json(path: Path, payload: MutableMapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    tmp.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )
    os.replace(tmp, path)


def append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload, ensure_ascii=False, sort_keys=True, default=str) + "\n")


def read_jsonl(path: Path, *, limit: int | None = None) -> list[dict[str, Any]]:
    return _strict_read_jsonl(path, limit=limit)


def append_live_episode(record: dict[str, Any], *, path: Path = LIVE_EPISODES_FILE) -> None:
    episode = {
        "schema_version": 1,
        "ts": record.get("ts"),
        "decision_id": record.get("decision_id"),
        "host": record.get("host"),
        "event": record.get("event"),
        "cwd": record.get("cwd"),
        "session_id": record.get("session_id"),
        "prompt_hash": record.get("prompt_hash"),
        "prompt_chars": record.get("prompt_chars"),
        "prompt_preview": record.get("prompt_preview"),
        "decision": record.get("decision"),
        "confidence": record.get("confidence"),
        "queries": record.get("queries") if isinstance(record.get("queries"), list) else [],
        "pages": record.get("pages") if isinstance(record.get("pages"), list) else [],
        "used_judge": bool(record.get("used_judge")),
        "search_mode": record.get("search_mode"),
        "context_style": record.get("context_style"),
        "latency_ms": record.get("latency_ms"),
        "status": record.get("status"),
        "error": record.get("error"),
        "quality": {
            "expected_pages": [],
            "negative_pages": [],
            "source": "unlabeled-live",
            "usefulness": "unknown",
        },
    }
    append_jsonl(path, episode)
