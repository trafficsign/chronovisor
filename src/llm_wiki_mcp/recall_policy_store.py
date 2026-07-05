"""Policy persistence helpers for recall self-improvement."""

from __future__ import annotations

import json
import os
from collections.abc import MutableMapping
from pathlib import Path
from typing import Any

from llm_wiki_mcp.wiki import WIKI_ROOT


IMPROVEMENT_DIR = WIKI_ROOT / "runtime" / "recall-improvement"
ACTIVE_POLICY_FILE = IMPROVEMENT_DIR / "active-policy.json"
REGISTRY_FILE = IMPROVEMENT_DIR / "policy-registry.jsonl"
EPISODES_FILE = IMPROVEMENT_DIR / "recall-episodes.jsonl"
RUNS_DIR = IMPROVEMENT_DIR / "runs"

FALSE_VALUES = {"0", "false", "False", "no", "NO", "off", "OFF"}

ALLOWED_POLICY_FIELDS: dict[str, dict[str, Any]] = {
    "search_threshold": {"type": float, "min": 0.05, "max": 0.9},
    "read_threshold": {"type": float, "min": 0.1, "max": 0.95},
    "max_context_chars": {"type": int, "min": 400, "max": 3000},
    "max_pages": {"type": int, "min": 1, "max": 6},
    "max_queries": {"type": int, "min": 1, "max": 6},
    "semantic": {"type": bool},
    "rewrite_enabled": {"type": bool},
    "fusion_bm25": {"type": float, "min": 0.0, "max": 2.0},
    "fusion_semantic": {"type": float, "min": 0.0, "max": 2.0},
    "fusion_usage_prior": {"type": float, "min": 0.0, "max": 1.0},
    "fusion_bm25_score_bonus": {"type": float, "min": 0.0, "max": 0.05},
    "fusion_bm25_rank_bonus": {"type": float, "min": 0.0, "max": 0.05},
    "fusion_bm25_rank_decay": {"type": float, "min": 0.0, "max": 0.05},
    "fusion_semantic_min_top_score": {"type": float, "min": 0.0, "max": 1.0},
    "fusion_semantic_min_margin": {"type": float, "min": 0.0, "max": 0.05},
    "fusion_semantic_low_confidence_weight": {"type": float, "min": 0.0, "max": 1.0},
}


def improvement_policy_enabled() -> bool:
    value = os.environ.get("LLM_WIKI_RECALL_IMPROVEMENT_POLICY")
    return value not in FALSE_VALUES


def _coerce_value(field: str, value: Any) -> Any:
    spec = ALLOWED_POLICY_FIELDS[field]
    wanted = spec["type"]
    if wanted is bool:
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            if value in {"1", "true", "True", "yes", "YES", "on", "ON"}:
                return True
            if value in FALSE_VALUES:
                return False
        raise ValueError(f"{field} must be boolean")
    if wanted is int:
        if isinstance(value, bool):
            raise ValueError(f"{field} must be int")
        coerced = int(value)
    else:
        if isinstance(value, bool):
            raise ValueError(f"{field} must be number")
        coerced = float(value)
    minimum = spec.get("min")
    maximum = spec.get("max")
    if minimum is not None and coerced < minimum:
        raise ValueError(f"{field} below minimum {minimum}")
    if maximum is not None and coerced > maximum:
        raise ValueError(f"{field} above maximum {maximum}")
    return coerced


def normalize_policy_overrides(raw: Any) -> dict[str, Any]:
    if not isinstance(raw, dict):
        return {}
    overrides: dict[str, Any] = {}
    for field, value in raw.items():
        key = str(field).strip().replace("-", "_")
        if key not in ALLOWED_POLICY_FIELDS:
            continue
        try:
            overrides[key] = _coerce_value(key, value)
        except (TypeError, ValueError):
            continue
    search = overrides.get("search_threshold")
    read = overrides.get("read_threshold")
    if isinstance(search, int | float) and isinstance(read, int | float) and read <= search:
        overrides["read_threshold"] = min(0.95, float(search) + 0.05)
    return overrides


def apply_policy_overrides(policy: Any, overrides: dict[str, Any]) -> list[str]:
    applied: list[str] = []
    normalized = normalize_policy_overrides(overrides)
    for field, value in normalized.items():
        if hasattr(policy, field):
            setattr(policy, field, value)
            applied.append(field)
    if getattr(policy, "read_threshold", 1.0) <= getattr(policy, "search_threshold", 0.0):
        policy.read_threshold = min(0.95, float(policy.search_threshold) + 0.05)
    return applied


def policy_snapshot(policy: Any) -> dict[str, Any]:
    return {
        field: getattr(policy, field)
        for field in ALLOWED_POLICY_FIELDS
        if hasattr(policy, field)
    }


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
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return []
    if limit is not None:
        lines = lines[-max(1, limit):]
    rows: list[dict[str, Any]] = []
    for line in lines:
        try:
            parsed = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            rows.append(parsed)
    return rows
