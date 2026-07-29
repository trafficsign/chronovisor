"""Fail-safe configuration for the optional research lane."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from chronovisor.research.research_types import ResearchBudget
from chronovisor.core.runtime_config import load_toml_file


def _int(data: dict[str, Any], key: str, default: int, minimum: int = 0) -> int:
    value = data.get(key)
    return (
        max(minimum, value)
        if isinstance(value, int) and not isinstance(value, bool)
        else default
    )


def _float(
    data: dict[str, Any], key: str, default: float, minimum: float = 0.0
) -> float:
    value = data.get(key)
    return (
        max(minimum, float(value))
        if isinstance(value, (int, float)) and not isinstance(value, bool)
        else default
    )


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
    source_packs: tuple[str, ...] = (
        "general",
        "code",
        "academic",
        "encyclopedia",
    )
    searxng_endpoint: str = ""
    github_endpoint: str = "https://api.github.com/search/repositories"
    github_token_env: str = "GITHUB_TOKEN"
    arxiv_endpoint: str = "https://export.arxiv.org/api/query"
    crossref_endpoint: str = "https://api.crossref.org/works"
    mediawiki_endpoint: str = "https://ja.wikipedia.org/w/api.php"
    allow_local_search_backend: bool = False
    max_provider_calls: int = 4
    per_provider_limit: int = 3
    provider_timeout_seconds: float = 8.0
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
    consolidation_mutation_mode: str = "proposal_only"
    consolidation_min_interval_seconds: int = 86_400
    consolidation_min_new_sessions: int = 5
    consolidation_max_jobs: int = 20
    egress_guard: bool = True
    external_content_trust: str = "untrusted"


def load_research_config(path: Path | str | None = None) -> ResearchConfig:
    root = load_toml_file(path).get("research")
    data = root if isinstance(root, dict) else {}
    budget_data = data.get("budgets") if isinstance(data.get("budgets"), dict) else {}
    resource_data = (
        data.get("resources") if isinstance(data.get("resources"), dict) else {}
    )
    web_data = data.get("web") if isinstance(data.get("web"), dict) else {}
    compaction_data = (
        data.get("compaction") if isinstance(data.get("compaction"), dict) else {}
    )
    consolidation = (
        data.get("consolidation") if isinstance(data.get("consolidation"), dict) else {}
    )
    security = data.get("security") if isinstance(data.get("security"), dict) else {}

    mode = str(os.getenv("CHRONOVISOR_RESEARCH_MODE") or data.get("mode") or "off")
    if mode not in {"off", "trace", "explicit", "idle", "sleep", "shadow", "auto"}:
        mode = "off"
    enabled_value = os.getenv("CHRONOVISOR_RESEARCH_ENABLED")
    enabled = _bool(data, "enabled", False)
    if enabled_value is not None:
        enabled = enabled_value.casefold() in {"1", "true", "yes", "on"}
    budget = ResearchBudget(
        max_iterations=_int(budget_data, "max_iterations", 5, 1),
        max_total_wall_seconds=_float(budget_data, "max_total_wall_seconds", 90.0, 1.0),
        max_single_generation_seconds=_float(
            budget_data, "max_single_generation_seconds", 30.0, 1.0
        ),
        max_single_generation_tokens=_int(
            budget_data, "max_single_generation_tokens", 256, 1
        ),
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
    protected_models = (
        tuple(item for item in protected if isinstance(item, str) and item)
        if isinstance(protected, list)
        else ResourceConfig().protected_models
    )
    configured_source_packs = web_data.get("source_packs")
    source_packs = (
        tuple(
            item for item in configured_source_packs if isinstance(item, str) and item
        )
        if isinstance(configured_source_packs, list)
        else WebConfig().source_packs
    )
    resources = ResourceConfig(
        scheduler=str(resource_data.get("scheduler") or "sync_first"),
        max_concurrent_generations=1,
        preempt_on_sync=_bool(resource_data, "preempt_on_sync", True),
        preempt_grace_ms=_int(resource_data, "preempt_grace_ms", 250),
        protected_models=protected_models,
        require_protected_residency=_bool(
            resource_data, "require_protected_residency", True
        ),
        sync_reserved_headroom_gib=_int(
            resource_data, "sync_reserved_headroom_gib", 16
        ),
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
        source_packs=source_packs,
        searxng_endpoint=str(web_data.get("searxng_endpoint") or ""),
        github_endpoint=str(
            web_data.get("github_endpoint") or WebConfig.github_endpoint
        ),
        github_token_env=str(
            web_data.get("github_token_env") or WebConfig.github_token_env
        ),
        arxiv_endpoint=str(web_data.get("arxiv_endpoint") or WebConfig.arxiv_endpoint),
        crossref_endpoint=str(
            web_data.get("crossref_endpoint") or WebConfig.crossref_endpoint
        ),
        mediawiki_endpoint=str(
            web_data.get("mediawiki_endpoint") or WebConfig.mediawiki_endpoint
        ),
        allow_local_search_backend=_bool(web_data, "allow_local_search_backend", False),
        max_provider_calls=min(4, _int(web_data, "max_provider_calls", 4, 1)),
        per_provider_limit=min(5, _int(web_data, "per_provider_limit", 3, 1)),
        provider_timeout_seconds=min(
            15.0,
            _float(web_data, "provider_timeout_seconds", 8.0, 0.5),
        ),
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
        checkpoint_max_total_bytes=_int(
            compaction_data, "checkpoint_max_total_bytes", 536_870_912
        ),
        gc_on_durable_receipt=_bool(compaction_data, "gc_on_durable_receipt", True),
    )
    return ResearchConfig(
        enabled=enabled,
        mode=mode,
        planner_model=str(data.get("planner_model") or ResearchConfig.planner_model),
        challenge_model=str(
            data.get("challenge_model") or ResearchConfig.challenge_model
        ),
        tie_break_model=str(
            data.get("tie_break_model") or ResearchConfig.tie_break_model
        ),
        max_depth=min(1, _int(data, "max_depth", 1)),
        budgets=budget,
        resources=resources,
        web=web,
        compaction=compaction,
        consolidation_enabled=_bool(consolidation, "enabled", False),
        # Preserve an invalid value so the consolidation runner can reject it
        # explicitly instead of silently turning it into an allowed mutation
        # mode. The default remains proposal-only.
        consolidation_mutation_mode=str(
            consolidation.get("mutation_mode") or "proposal_only"
        ),
        consolidation_min_interval_seconds=_int(
            consolidation, "min_interval_seconds", 86_400
        ),
        consolidation_min_new_sessions=_int(consolidation, "min_new_sessions", 5, 1),
        consolidation_max_jobs=_int(consolidation, "max_jobs", 20, 1),
        egress_guard=_bool(security, "egress_guard", True),
        external_content_trust=str(
            security.get("external_content_trust") or "untrusted"
        ),
    )
