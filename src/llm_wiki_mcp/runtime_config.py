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
DEFAULT_INGEST_MODEL = "qwen3.6:35b-a3b-mxfp8"


@dataclass(frozen=True)
class HookPolicy:
    user_prompt_recall: bool = True
    stop_save: bool = True
    stop_audit: bool = True
    stop_content_correction: bool = True
    stop_recall_improve: bool = True


@dataclass(frozen=True)
class EmbeddingConfig:
    model: str = DEFAULT_EMBEDDING_MODEL
    document_prefix: str = ""
    query_prefix: str = ""


@dataclass(frozen=True)
class RerankerConfig:
    enabled: bool = False
    model: str = "BAAI/bge-reranker-v2-m3"
    backend: str = "transformers"
    top_n: int = 20
    max_length: int = 1024
    batch_size: int = 8
    device: str = ""
    weight: float = 0.25


@dataclass(frozen=True)
class IngestConfig:
    model: str = DEFAULT_INGEST_MODEL
    keep_alive: str = "5m"
    temperature: float = 0.3
    num_ctx: int = 65_536
    max_num_ctx: int = 262_144
    num_predict: int = 8_192
    read_timeout_ms: int = 660_000


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
        stop_content_correction=nested_bool(
            data, ("hooks", "stop", "content_correction"), True
        ),
        stop_recall_improve=nested_bool(data, ("hooks", "stop", "recall_improve"), True),
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


def _positive_int(value: Any, default: int, *, minimum: int = 1) -> int:
    if isinstance(value, bool):
        return default
    if isinstance(value, int):
        return value if value >= minimum else default
    return default


def _nonnegative_float(value: Any, default: float) -> float:
    if isinstance(value, bool):
        return default
    if isinstance(value, (int, float)):
        return float(value) if value >= 0 else default
    return default


def _bounded_float(value: Any, default: float, *, minimum: float, maximum: float) -> float:
    if isinstance(value, bool):
        return default
    if isinstance(value, (int, float)):
        number = float(value)
        return number if minimum <= number <= maximum else default
    return default


def load_ingest_config(path: Path | str | None = None) -> IngestConfig:
    data = load_toml_file(path)
    section = data.get("ingest")
    if not isinstance(section, dict):
        return IngestConfig()

    model = section.get("model")
    keep_alive = section.get("keep_alive")
    max_num_ctx = _positive_int(section.get("max_num_ctx"), IngestConfig.max_num_ctx, minimum=2_048)
    num_ctx = _positive_int(section.get("num_ctx"), IngestConfig.num_ctx, minimum=2_048)
    if num_ctx > max_num_ctx:
        num_ctx = max_num_ctx

    return IngestConfig(
        model=model if isinstance(model, str) and model.strip() else IngestConfig.model,
        keep_alive=keep_alive if isinstance(keep_alive, str) and keep_alive.strip() else IngestConfig.keep_alive,
        temperature=_bounded_float(section.get("temperature"), IngestConfig.temperature, minimum=0.0, maximum=2.0),
        num_ctx=num_ctx,
        max_num_ctx=max_num_ctx,
        num_predict=_positive_int(section.get("num_predict"), IngestConfig.num_predict, minimum=128),
        read_timeout_ms=_positive_int(section.get("read_timeout_ms"), IngestConfig.read_timeout_ms, minimum=1_000),
    )


def load_reranker_config(path: Path | str | None = None) -> RerankerConfig:
    data = load_toml_file(path)
    search = data.get("search")
    reranker: Any = data.get("reranker")
    if isinstance(search, dict) and isinstance(search.get("reranker"), dict):
        reranker = search["reranker"]
    if not isinstance(reranker, dict):
        return RerankerConfig()

    model = reranker.get("model")
    backend = reranker.get("backend")
    device = reranker.get("device")
    return RerankerConfig(
        enabled=reranker.get("enabled") is True,
        model=model if isinstance(model, str) and model.strip() else RerankerConfig.model,
        backend=backend if isinstance(backend, str) and backend.strip() else RerankerConfig.backend,
        top_n=_positive_int(reranker.get("top_n"), RerankerConfig.top_n),
        max_length=_positive_int(reranker.get("max_length"), RerankerConfig.max_length),
        batch_size=_positive_int(reranker.get("batch_size"), RerankerConfig.batch_size),
        device=device if isinstance(device, str) else "",
        weight=_nonnegative_float(reranker.get("weight"), RerankerConfig.weight),
    )


@dataclass(frozen=True)
class NegativeFeedbackConfig:
    enabled: bool = False
    kinds: tuple[str, ...] = ("page_ignored", "injection_ignored", "false-positive")
    similarity_threshold: float = 0.35
    penalty: float = 0.85
    max_age_days: int = 180
    max_entries: int = 500


def load_negative_feedback_config(path: Path | str | None = None) -> NegativeFeedbackConfig:
    data = load_toml_file(path)
    search = data.get("search")
    section: Any = None
    if isinstance(search, dict) and isinstance(search.get("negative_feedback"), dict):
        section = search["negative_feedback"]
    if not isinstance(section, dict):
        return NegativeFeedbackConfig()

    kinds_value = section.get("kinds")
    kinds = NegativeFeedbackConfig.kinds
    if isinstance(kinds_value, list):
        cleaned = tuple(k for k in kinds_value if isinstance(k, str) and k.strip())
        if cleaned:
            kinds = cleaned
    threshold = section.get("similarity_threshold")
    penalty = section.get("penalty")
    return NegativeFeedbackConfig(
        enabled=section.get("enabled") is True,
        kinds=kinds,
        similarity_threshold=(
            float(threshold)
            if isinstance(threshold, (int, float)) and not isinstance(threshold, bool)
            and 0.0 < float(threshold) <= 1.0
            else NegativeFeedbackConfig.similarity_threshold
        ),
        penalty=(
            float(penalty)
            if isinstance(penalty, (int, float)) and not isinstance(penalty, bool)
            and 0.0 < float(penalty) <= 1.0
            else NegativeFeedbackConfig.penalty
        ),
        max_age_days=_positive_int(section.get("max_age_days"), NegativeFeedbackConfig.max_age_days),
        max_entries=_positive_int(section.get("max_entries"), NegativeFeedbackConfig.max_entries),
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
