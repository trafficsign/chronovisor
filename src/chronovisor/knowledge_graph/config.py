"""Feature flags and bounded resource policy for the typed graph."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from chronovisor.core.runtime_config import active_config_file, load_toml_file

VALID_MODES = frozenset({"off", "shadow", "candidate", "active"})
LOCAL_MODEL_PREFIXES = ("", "ollama://", "local://")


@dataclass(frozen=True)
class GraphRetrievalConfig:
    mode: str = "shadow"
    max_hops: int = 2
    max_relations_per_node: int = 12
    max_candidate_pages: int = 50
    per_predicate_cap: int = 4
    hub_penalty: float = 0.15


@dataclass(frozen=True)
class KnowledgeGraphConfig:
    enabled: bool = True
    mode: str = "shadow"
    external_models_allowed: bool = False
    extractor_model: str = "gemma4:26b"
    community_summary_model: str = "gemma4:26b"
    local_extraction_enabled: bool = True
    max_changed_pages_per_cycle: int = 25
    max_model_seconds_per_day: int = 7_200
    max_queue_size: int = 500
    max_community_summaries_per_cycle: int = 2
    min_relation_sessions: int = 5
    min_relation_strong: int = 20
    min_entity_sessions: int = 5
    min_entity_strong: int = 20
    min_rubric_gold: int = 30
    retrieval: GraphRetrievalConfig = field(default_factory=GraphRetrievalConfig)


def _integer(value: Any, default: int, low: int, high: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        return default
    return max(low, min(high, value))


def _number(value: Any, default: float, low: float, high: float) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        return default
    return max(low, min(high, float(value)))


def _mode(value: Any, default: str) -> str:
    candidate = str(value or default).strip().lower()
    return candidate if candidate in VALID_MODES else "off"


def model_is_local(model: str) -> bool:
    """Reject HTTP/provider model identifiers at the configuration boundary."""

    value = model.strip().lower()
    if "://" not in value:
        return bool(value)
    return value.startswith(("ollama://", "local://"))


def load_config(path: Path | str | None = None) -> KnowledgeGraphConfig:
    data = load_toml_file(active_config_file(Path(path)) if path else None)
    section = data.get("knowledge_graph") if isinstance(data, dict) else None
    if not isinstance(section, dict):
        return KnowledgeGraphConfig()
    retrieval_value = section.get("retrieval")
    retrieval_section = retrieval_value if isinstance(retrieval_value, dict) else {}
    model = str(section.get("extractor_model") or "gemma4:26b").strip()
    summary_model = str(
        section.get("community_summary_model") or model or "gemma4:26b"
    ).strip()
    external_allowed = section.get("external_models_allowed") is True
    if not model_is_local(model) or not model_is_local(summary_model):
        raise ValueError("knowledge graph local model requirement violated")
    if external_allowed:
        raise ValueError("knowledge graph external models are not supported")
    return KnowledgeGraphConfig(
        enabled=section.get("enabled") is not False,
        mode=_mode(section.get("mode"), "shadow"),
        external_models_allowed=external_allowed,
        extractor_model=model,
        community_summary_model=summary_model,
        local_extraction_enabled=section.get("local_extraction_enabled") is not False,
        max_changed_pages_per_cycle=_integer(
            section.get("max_changed_pages_per_cycle"), 25, 1, 1_000
        ),
        max_model_seconds_per_day=_integer(
            section.get("max_model_seconds_per_day"), 7_200, 0, 86_400
        ),
        max_queue_size=_integer(section.get("max_queue_size"), 500, 1, 50_000),
        max_community_summaries_per_cycle=_integer(
            section.get("max_community_summaries_per_cycle"), 2, 0, 50
        ),
        min_relation_sessions=_integer(
            section.get("min_relation_sessions"), 5, 1, 10_000
        ),
        min_relation_strong=_integer(
            section.get("min_relation_strong"), 20, 1, 1_000_000
        ),
        min_entity_sessions=_integer(section.get("min_entity_sessions"), 5, 1, 10_000),
        min_entity_strong=_integer(section.get("min_entity_strong"), 20, 1, 1_000_000),
        min_rubric_gold=_integer(section.get("min_rubric_gold"), 30, 1, 10_000),
        retrieval=GraphRetrievalConfig(
            mode=_mode(retrieval_section.get("mode"), "shadow"),
            max_hops=_integer(retrieval_section.get("max_hops"), 2, 1, 4),
            max_relations_per_node=_integer(
                retrieval_section.get("max_relations_per_node"), 12, 1, 100
            ),
            max_candidate_pages=_integer(
                retrieval_section.get("max_candidate_pages"), 50, 1, 200
            ),
            per_predicate_cap=_integer(
                retrieval_section.get("per_predicate_cap"), 4, 1, 50
            ),
            hub_penalty=_number(retrieval_section.get("hub_penalty"), 0.15, 0.0, 1.0),
        ),
    )
