"""Shared runtime configuration helpers.

The old runtime grew separate knobs around ``recall.toml``.  Keep that file
working, but prefer ``config.toml`` when it exists so hook, recall, audit, and
auto-apply settings can converge on one public shape.
"""

from __future__ import annotations

import os
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from llm_wiki_mcp.wiki import WIKI_ROOT

CONFIG_FILE = WIKI_ROOT / "config.toml"
LEGACY_RECALL_CONFIG_FILE = WIKI_ROOT / "recall.toml"

FALSE_VALUES = {"0", "false", "False", "no", "NO", "off", "OFF"}
TRUE_VALUES = {"1", "true", "True", "yes", "YES", "on", "ON"}
DEFAULT_EMBEDDING_MODEL = "nomic-embed-text"


@dataclass(frozen=True)
class HookPolicy:
    user_prompt_recall: bool = True
    stop_save: bool = True
    stop_audit: bool = True


@dataclass(frozen=True)
class EmbeddingConfig:
    model: str = DEFAULT_EMBEDDING_MODEL
    document_prefix: str = ""
    query_prefix: str = ""


def active_config_file(path: Path | str | None = None) -> Path:
    if path:
        return Path(path).expanduser()
    if CONFIG_FILE.exists():
        return CONFIG_FILE
    return LEGACY_RECALL_CONFIG_FILE


def load_toml_file(path: Path | str | None = None) -> dict[str, Any]:
    resolved = active_config_file(path)
    try:
        if not resolved.exists():
            return {}
        data = tomllib.loads(resolved.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except (OSError, tomllib.TOMLDecodeError):
        return {}


def env_flag(name: str) -> bool | None:
    value = os.environ.get(name)
    if value is None:
        return None
    if value in FALSE_VALUES:
        return False
    if value in TRUE_VALUES:
        return True
    return bool(value)


def nested_bool(data: dict[str, Any], path: tuple[str, ...], default: bool) -> bool:
    cur: Any = data
    for key in path:
        if not isinstance(cur, dict):
            return default
        cur = cur.get(key)
    return cur if isinstance(cur, bool) else default


def load_hook_policy(path: Path | str | None = None) -> HookPolicy:
    data = load_toml_file(path)
    return HookPolicy(
        user_prompt_recall=nested_bool(data, ("hooks", "user_prompt", "recall"), True),
        stop_save=nested_bool(data, ("hooks", "stop", "save"), True),
        stop_audit=nested_bool(data, ("hooks", "stop", "audit"), True),
    )


def load_embedding_config(path: Path | str | None = None) -> EmbeddingConfig:
    data = load_toml_file(path)
    embedding = data.get("embedding")
    if not isinstance(embedding, dict):
        return EmbeddingConfig()
    model = embedding.get("model")
    document_prefix = embedding.get("document_prefix")
    query_prefix = embedding.get("query_prefix")
    return EmbeddingConfig(
        model=model if isinstance(model, str) and model.strip() else DEFAULT_EMBEDDING_MODEL,
        document_prefix=document_prefix if isinstance(document_prefix, str) else "",
        query_prefix=query_prefix if isinstance(query_prefix, str) else "",
    )


def normalize_recall_config(data: dict[str, Any]) -> dict[str, Any]:
    """Return a config shape accepted by ``recall_runtime._apply_config``.

    Legacy ``recall.toml`` already uses top-level sections like ``[gate]`` and
    ``[thresholds]``.  New ``config.toml`` nests those under ``[recall.*]``.
    """
    if not isinstance(data, dict):
        return {}
    out: dict[str, Any] = dict(data)
    recall = data.get("recall")
    if not isinstance(recall, dict):
        return out

    if isinstance(recall.get("enabled"), bool):
        out["enabled"] = recall["enabled"]
    if isinstance(recall.get("model"), str):
        out["model"] = recall["model"]

    for section in ("thresholds", "budgets", "gate", "policy", "rewrite", "fusion", "calibration"):
        if isinstance(recall.get(section), dict):
            out[section] = recall[section]

    recall_options: dict[str, Any] = {}
    for key in ("semantic", "judge_mode", "gate_mode", "context_style", "max_context_chars", "session_ttl_seconds"):
        if key in recall:
            recall_options[key] = recall[key]
    if recall_options:
        out["recall"] = recall_options
    return out


def normalize_audit_config(data: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(data, dict):
        return {}
    out: dict[str, Any] = dict(data)
    audit = data.get("audit")
    if isinstance(audit, dict):
        out["auditor"] = audit
    return out


def config_summary(path: Path | str | None = None) -> dict[str, Any]:
    resolved = active_config_file(path)
    data = load_toml_file(resolved)
    return {
        "path": str(resolved),
        "exists": resolved.exists(),
        "mode": "unified" if resolved == CONFIG_FILE else "legacy-recall",
        "hook_policy": load_hook_policy(resolved).__dict__,
        "sections": sorted(data.keys()),
    }
