"""Fail-safe configuration for the optional research lane."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from llm_wiki_mcp.research_types import ResearchBudget
from llm_wiki_mcp.runtime_config import load_toml_file


def _int(data: dict[str, Any], key: str, default: int, minimum: int = 0) -> int:
    value = data.get(key)
    return max(minimum, value) if isinstance(value, int) and not isinstance(value, bool) else default


def _float(data: dict[str, Any], key: str, default: float, minimum: float = 0.0) -> float:
    value = data.get(key)
    return max(minimum, float(value)) if isinstance(value, (int, float)) and not isinstance(value, bool) else default


def _bool(data: dict[str, Any], key: str, default: bool) -> bool:
    value = data.get(key)
    return value if isinstance(value, bool) else default


@dataclass(frozen=True)
class ResourceConfig:
    scheduler: str = "sync_first"
    max_concurrent_generations: int = 1
    preempt_on_sync: bool = True
    preempt_grace_ms: int = 250
    protected_models: tuple[str, ...] = ("ornith:9b-q4_K_M", "bge-m3")
    require_protected_residency: bool = True
    sync_reserved_headroom_gib: int = 16
    sync_lease_wait_limit_ms: int = 50
    coordinate_ollama: bool = True
    coordinate_mps_reranker: bool = True


@dataclass(frozen=True)
class WebConfig:
    adapter_enabled: bool = False
    live_egress_enabled: bool = False
    provider: str = ""
    endpoint: str = ""
    api_key_env: str = ""
    max_searches: int = 8
    max_fetches: int = 5
    cache_ttl_seconds: int = 900
    allow_private_network: bool = False
    max_fetch_bytes: int = 2_000_000


@dataclass(frozen=True)
class CompactionConfig:
    enabled: bool = False
    checkpoint_enabled: bool = False
    checkpoint_ttl_seconds: int = 604_800
    checkpoint_max_total_bytes: int = 536_870_912
    gc_on_durable_receipt: bool = True


@dataclass(frozen=True)
class ResearchConfig:
    enabled: bool = False
    mode: str = "off"
    planner_model: str = "maxwell1500/ornith-35b:Q5_K_M"
    challenge_model: str = "gpt-oss:20b"
    tie_break_model: str = "gemma4:26b"
    max_depth: int = 1
    budgets: ResearchBudget = field(default_factory=ResearchBudget)
    resources: ResourceConfig = field(default_factory=ResourceConfig)
    web: WebConfig = field(default_factory=WebConfig)
    compaction: CompactionConfig = field(default_factory=CompactionConfig)
    consolidation_enabled: bool = False
    egress_guard: bool = True
    external_content_trust: str = "untrusted"


def load_research_config(path: Path | str | None = None) -> ResearchConfig:
    root = load_toml_file(path).get("research")
    data = root if isinstance(root, dict) else {}
    budget_data = data.get("budgets") if isinstance(data.get("budgets"), dict) else {}
    resource_data = data.get("resources") if isinstance(data.get("resources"), dict) else {}
    web_data = data.get("web") if isinstance(data.get("web"), dict) else {}
    compaction_data = data.get("compaction") if isinstance(data.get("compaction"), dict) else {}
    consolidation = data.get("consolidation") if isinstance(data.get("consolidation"), dict) else {}
    security = data.get("security") if isinstance(data.get("security"), dict) else {}

    mode = str(os.getenv("LLM_WIKI_RESEARCH_MODE") or data.get("mode") or "off")
    if mode not in {"off", "trace", "explicit", "idle", "sleep", "shadow", "auto"}:
        mode = "off"
    enabled_value = os.getenv("LLM_WIKI_RESEARCH_ENABLED")
    enabled = _bool(data, "enabled", False)
    if enabled_value is not None:
        enabled = enabled_value.casefold() in {"1", "true", "yes", "on"}
    budget = ResearchBudget(
        max_iterations=_int(budget_data, "max_iterations", 5, 1),
        max_total_wall_seconds=_float(budget_data, "max_total_wall_seconds", 90.0, 1.0),
        max_single_generation_seconds=_float(budget_data, "max_single_generation_seconds", 30.0, 1.0),
        max_single_generation_tokens=_int(budget_data, "max_single_generation_tokens", 512, 1),
        max_planner_calls=_int(budget_data, "max_planner_calls", 5),
        max_challenge_calls=_int(budget_data, "max_challenge_calls", 2),
        max_tie_break_calls=_int(budget_data, "max_tie_break_calls", 1),
        max_repair_calls=_int(budget_data, "max_repair_calls", 2),
        max_total_model_calls=_int(budget_data, "max_total_model_calls", 10),
        max_searches=_int(web_data, "max_searches", 8),
        max_fetches=_int(web_data, "max_fetches", 5),
        max_observation_bytes=_int(budget_data, "max_observation_bytes", 200_000),
    )
    protected = resource_data.get("protected_models")
    protected_models = tuple(item for item in protected if isinstance(item, str) and item) if isinstance(protected, list) else ResourceConfig().protected_models
    resources = ResourceConfig(
        scheduler=str(resource_data.get("scheduler") or "sync_first"),
        max_concurrent_generations=1,
        preempt_on_sync=_bool(resource_data, "preempt_on_sync", True),
        preempt_grace_ms=_int(resource_data, "preempt_grace_ms", 250),
        protected_models=protected_models,
        require_protected_residency=_bool(resource_data, "require_protected_residency", True),
        sync_reserved_headroom_gib=_int(resource_data, "sync_reserved_headroom_gib", 16),
        sync_lease_wait_limit_ms=_int(resource_data, "sync_lease_wait_limit_ms", 50),
        coordinate_ollama=_bool(resource_data, "coordinate_ollama", True),
        coordinate_mps_reranker=_bool(resource_data, "coordinate_mps_reranker", True),
    )
    web = WebConfig(
        adapter_enabled=_bool(web_data, "adapter_enabled", False),
        live_egress_enabled=_bool(web_data, "live_egress_enabled", False),
        provider=str(web_data.get("provider") or ""),
        endpoint=str(web_data.get("endpoint") or ""),
        api_key_env=str(web_data.get("api_key_env") or ""),
        max_searches=budget.max_searches,
        max_fetches=budget.max_fetches,
        cache_ttl_seconds=_int(web_data, "cache_ttl_seconds", 900),
        allow_private_network=_bool(web_data, "allow_private_network", False),
        max_fetch_bytes=_int(web_data, "max_fetch_bytes", 2_000_000, 1),
    )
    compaction = CompactionConfig(
        enabled=_bool(compaction_data, "enabled", False),
        checkpoint_enabled=_bool(compaction_data, "checkpoint_enabled", False),
        checkpoint_ttl_seconds=_int(compaction_data, "checkpoint_ttl_seconds", 604_800),
        checkpoint_max_total_bytes=_int(compaction_data, "checkpoint_max_total_bytes", 536_870_912),
        gc_on_durable_receipt=_bool(compaction_data, "gc_on_durable_receipt", True),
    )
    return ResearchConfig(
        enabled=enabled,
        mode=mode,
        planner_model=str(data.get("planner_model") or ResearchConfig.planner_model),
        challenge_model=str(data.get("challenge_model") or ResearchConfig.challenge_model),
        tie_break_model=str(data.get("tie_break_model") or ResearchConfig.tie_break_model),
        max_depth=min(1, _int(data, "max_depth", 1)),
        budgets=budget,
        resources=resources,
        web=web,
        compaction=compaction,
        consolidation_enabled=_bool(consolidation, "enabled", False),
        egress_guard=_bool(security, "egress_guard", True),
        external_content_trust=str(security.get("external_content_trust") or "untrusted"),
    )
