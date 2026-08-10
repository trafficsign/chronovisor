"""Pure policy contracts shared by recall execution and decision fixtures."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

FALSE_VALUES = {"0", "false", "False", "no", "NO", "off", "OFF"}

ALLOWED_POLICY_FIELDS: dict[str, dict[str, Any]] = {
    "search_threshold": {"type": float, "min": 0.05, "max": 0.9},
    "read_threshold": {"type": float, "min": 0.1, "max": 0.95},
    "max_context_chars": {"type": int, "min": 400, "max": 3000},
    "max_pages": {"type": int, "min": 1, "max": 6},
    "max_queries": {"type": int, "min": 1, "max": 6},
    "semantic": {"type": bool},
    "rewrite_enabled": {"type": bool},
    "fusion_anchor": {"type": float, "min": 0.0, "max": 2.0},
    "fusion_bm25": {"type": float, "min": 0.0, "max": 2.0},
    "fusion_semantic": {"type": float, "min": 0.0, "max": 2.0},
    "fusion_graph": {"type": float, "min": 0.0, "max": 1.0},
    "fusion_context": {"type": float, "min": 0.0, "max": 1.0},
    "fusion_usage_prior": {"type": float, "min": 0.0, "max": 1.0},
    "fusion_bm25_score_bonus": {"type": float, "min": 0.0, "max": 0.05},
    "fusion_bm25_rank_bonus": {"type": float, "min": 0.0, "max": 0.05},
    "fusion_bm25_rank_decay": {"type": float, "min": 0.0, "max": 0.05},
    "fusion_semantic_min_top_score": {"type": float, "min": 0.0, "max": 1.0},
    "fusion_semantic_min_margin": {"type": float, "min": 0.0, "max": 0.05},
    "fusion_semantic_low_confidence_weight": {
        "type": float,
        "min": 0.0,
        "max": 1.0,
    },
}


@dataclass
class RecallPolicy:
    enabled: bool = True
    search_threshold: float = 0.35
    read_threshold: float = 0.65
    # Keep the contract-fixture fallback stable. Deployments that opt into a
    # larger recall envelope must set this explicitly in [recall.budgets].
    max_context_chars: int = 600
    max_state_context_chars: int = 1600
    max_total_context_chars: int = 2402
    max_pages: int = 3
    max_queries: int = 3
    total_timeout_ms: int = 4000
    deterministic_fallback_reserve_ms: int = 600
    circuit_breaker_failures: int = 2
    circuit_breaker_cooldown_seconds: int = 60
    semantic: bool = True
    gate_mode: str = "evidence"  # legacy | evidence
    context_style: str = "cards"  # legacy | cards
    log_decisions: bool = True
    avoid_heavy_personal_context_in_chitchat: bool = True
    use_feedback_suppressions: bool = True
    fail_silent_on_judge_unavailable: bool = True
    judge_mode: str = "auto"  # off | auto | always
    judge_think: bool = False
    judge_timeout_ms: int = 2000
    judge_num_ctx: int = 4096
    judge_num_predict: int = 64
    judge_keep_alive: str = "24h"
    warmup_timeout_ms: int = 15000
    judge_include_queries: bool = False
    rewrite_enabled: bool = True
    rewrite_timeout_ms: int = 3000
    fusion_anchor: float = 0.9
    fusion_bm25: float = 1.0
    fusion_semantic: float = 0.6
    fusion_graph: float = 0.3
    fusion_context: float = 0.25
    fusion_usage_prior: float = 0.0
    fusion_bm25_score_bonus: float = 0.005
    fusion_bm25_rank_bonus: float = 0.006
    fusion_bm25_rank_decay: float = 0.006
    fusion_semantic_min_top_score: float = 0.45
    fusion_semantic_min_margin: float = 0.002
    fusion_semantic_low_confidence_weight: float = 0.25
    fusion_usage_prior_decay: float = 0.98
    fusion_usage_prior_cap: float = 3.0
    calibration_enabled: bool = True
    calibration_min_samples: int = 500
    calibration_holdout_ratio: float = 0.2
    calibration_min_improvement: float = 0.02
    session_ttl_seconds: int = 7 * 24 * 60 * 60
    processor_enabled: bool = False
    processor_shadow_enabled: bool = False
    processor_auto_enable: bool = False
    processor_max_candidates: int = 10
    processor_max_pointer_cards: int = 6
    processor_max_rich_evidence: int = 2
    processor_injection_token_budget: int = 1200
    processor_certificate_required: bool = True
    processor_judge_enabled: bool = True
    processor_judge_timeout_ms: int = 900
    processor_escalation_timeout_ms: int = 900


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
    coerced: int | float
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
    if getattr(policy, "read_threshold", 1.0) <= getattr(
        policy, "search_threshold", 0.0
    ):
        policy.read_threshold = min(0.95, float(policy.search_threshold) + 0.05)
    return applied


def policy_snapshot(policy: Any) -> dict[str, Any]:
    return {
        field: getattr(policy, field)
        for field in ALLOWED_POLICY_FIELDS
        if hasattr(policy, field)
    }
