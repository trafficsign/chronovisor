"""Schema and configuration for the stateful Recall Field."""

from __future__ import annotations

import hashlib
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from chronovisor.core.runtime_config import active_config_file, load_toml_file
from chronovisor.search.search_types import tokenize

FIELD_SCHEMA_VERSION = 2
FIELD_EVENT_KINDS = frozenset(
    {
        "stimulus",
        "spread",
        "inhibit",
        "reject",
        "commit_queued",
        "commit_applied",
        "topic_reset",
        "snapshot",
        "fault",
    }
)


@dataclass(frozen=True)
class RecallFieldConfig:
    mode: str = "shadow"
    canary_percent: int = 0
    auto_growth: bool = False
    auto_promote: bool = False
    working_set_size: int = 30
    max_active_nodes: int = 128
    max_active_edges: int = 256
    positive_learning: bool = False
    wall_half_life_seconds: int = 300
    turn_decay: float = 0.82
    spread_gain: float = 0.35
    max_hops: int = 2
    global_inhibition: float = 0.08
    refractory_turns: int = 1
    topic_reset_similarity: float = 0.15
    session_ttl_seconds: int = 7 * 24 * 60 * 60
    event_retention: int = 2_000


@dataclass
class ActivationNode:
    activation: float = 0.0
    direct: float = 0.0
    spread: float = 0.0
    negative: float = 0.0
    inhibition: float = 0.0
    anti_index: float = 0.0
    hub_penalty: float = 0.0
    last_turn: int = 0
    last_seq: int = 0

    @classmethod
    def from_dict(cls, value: Any) -> ActivationNode:
        if not isinstance(value, dict):
            return cls()
        return cls(
            activation=_bounded_float(value.get("activation"), 0.0, -1.0, 1.0),
            direct=_bounded_float(value.get("direct"), 0.0, 0.0, 1.0),
            spread=_bounded_float(value.get("spread"), 0.0, 0.0, 1.0),
            negative=_bounded_float(value.get("negative"), 0.0, 0.0, 1.0),
            inhibition=_bounded_float(value.get("inhibition"), 0.0, 0.0, 1.0),
            anti_index=_bounded_float(value.get("anti_index"), 0.0, 0.0, 1.0),
            hub_penalty=_bounded_float(value.get("hub_penalty"), 0.0, 0.0, 1.0),
            last_turn=_bounded_int(value.get("last_turn"), 0, 0, 1_000_000_000),
            last_seq=_bounded_int(value.get("last_seq"), 0, 0, 10_000_000_000),
        )


@dataclass
class RecallFieldState:
    session_hash: str
    host: str = ""
    topic_epoch: int = 0
    turn: int = 0
    seq: int = 0
    created_at_epoch: float = 0.0
    updated_at_epoch: float = 0.0
    topic_signature: tuple[str, ...] = ()
    topic_prompt_hash: str = ""
    active: dict[str, ActivationNode] = field(default_factory=dict)
    shadow: dict[str, ActivationNode] = field(default_factory=dict)
    pending_teacher_commits: list[dict[str, Any]] = field(default_factory=list)
    negative_contributions: dict[str, dict[str, Any]] = field(default_factory=dict)
    full_search_fallback: bool = True

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": FIELD_SCHEMA_VERSION,
            "session_hash": self.session_hash,
            "host": self.host,
            "topic_epoch": self.topic_epoch,
            "turn": self.turn,
            "seq": self.seq,
            "created_at_epoch": self.created_at_epoch,
            "updated_at_epoch": self.updated_at_epoch,
            "topic_signature": list(self.topic_signature),
            "topic_prompt_hash": self.topic_prompt_hash,
            "active": {
                page_id: asdict(node) for page_id, node in sorted(self.active.items())
            },
            "shadow": {
                page_id: asdict(node) for page_id, node in sorted(self.shadow.items())
            },
            "pending_teacher_commits": list(self.pending_teacher_commits),
            "negative_contributions": dict(self.negative_contributions),
            "full_search_fallback": self.full_search_fallback,
        }

    @classmethod
    def from_dict(cls, value: Any, *, session_hash: str) -> RecallFieldState:
        if not isinstance(value, dict):
            raise ValueError("field snapshot must be an object")
        if value.get("schema_version") != FIELD_SCHEMA_VERSION:
            raise ValueError("unsupported field snapshot schema")
        if value.get("session_hash") != session_hash:
            raise ValueError("field snapshot session mismatch")

        def nodes(name: str) -> dict[str, ActivationNode]:
            rows = value.get(name)
            if not isinstance(rows, dict):
                return {}
            return {
                str(page_id): ActivationNode.from_dict(node)
                for page_id, node in rows.items()
                if isinstance(page_id, str) and page_id
            }

        pending = value.get("pending_teacher_commits")
        negative_contributions = value.get("negative_contributions")
        if not isinstance(negative_contributions, dict) or any(
            not isinstance(key, str)
            or not key
            or not isinstance(row, dict)
            or row.get("buffer") not in {"active", "shadow"}
            or not isinstance(row.get("page_weights"), dict)
            or any(
                not isinstance(page_id, str)
                or not page_id
                or isinstance(weight, bool)
                or not isinstance(weight, int | float)
                or not 0.0 <= float(weight) <= 1.0
                for page_id, weight in row.get("page_weights", {}).items()
            )
            for key, row in negative_contributions.items()
        ):
            raise ValueError("invalid field negative contributions")
        return cls(
            session_hash=session_hash,
            host=str(value.get("host") or "").strip().casefold(),
            topic_epoch=_bounded_int(value.get("topic_epoch"), 0, 0, 1_000_000),
            turn=_bounded_int(value.get("turn"), 0, 0, 1_000_000_000),
            seq=_bounded_int(value.get("seq"), 0, 0, 10_000_000_000),
            created_at_epoch=_bounded_float(
                value.get("created_at_epoch"), 0.0, 0.0, 10_000_000_000.0
            ),
            updated_at_epoch=_bounded_float(
                value.get("updated_at_epoch"), 0.0, 0.0, 10_000_000_000.0
            ),
            topic_signature=tuple(
                str(item)
                for item in value.get("topic_signature", [])
                if isinstance(item, str) and re.fullmatch(r"[0-9a-f]{12}", item)
            )[:64],
            topic_prompt_hash=str(value.get("topic_prompt_hash") or "")[:64],
            active=nodes("active"),
            shadow=nodes("shadow"),
            pending_teacher_commits=[
                row
                for row in pending
                if isinstance(row, dict)
                and isinstance(row.get("page_id"), str)
                and isinstance(row.get("available_turn"), int)
            ]
            if isinstance(pending, list)
            else [],
            negative_contributions={
                str(key): dict(row)
                for key, row in negative_contributions.items()
                if isinstance(key, str) and key and isinstance(row, dict)
            },
            full_search_fallback=value.get("full_search_fallback") is not False,
        )


@dataclass(frozen=True)
class FieldStimulus:
    page_id: str
    kind: str
    weight: float
    negative: bool = False
    reason_code: str = ""
    certificate_id: str = ""
    components: dict[str, float] = field(default_factory=dict)


@dataclass(frozen=True)
class FieldEvent:
    seq: int
    timestamp_epoch: float
    session_hash: str
    topic_epoch: int
    kind: str
    page_id: str = ""
    source_page_id: str = ""
    target_page_id: str = ""
    edge_type: str = ""
    delta: float = 0.0
    activation: float = 0.0
    reason_code: str = ""
    certificate_id: str = ""
    components: dict[str, float | str] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def session_hash(host: str, session_id: str) -> str:
    if not host.strip() or not session_id.strip():
        return ""
    return hashlib.sha256(
        f"{host.strip().casefold()}:{session_id.strip()}".encode()
    ).hexdigest()[:16]


def topic_signature(text: str) -> tuple[str, ...]:
    tokens = set(tokenize(text))
    return tuple(
        sorted(
            hashlib.sha256(token.encode("utf-8")).hexdigest()[:12] for token in tokens
        )
    )[:64]


_PRONOUN_CONTINUATION_RE = re.compile(
    r"(?:それ|その|これ|この|あれ|前の|さっきの|続き|"
    r"\b(?:it|that|this|those|them|same)\b)",
    re.IGNORECASE,
)
_ABRUPT_SWITCH_RE = re.compile(
    r"(?:話(?:は|を)?変|別件|ところで|unrelated|new topic|switch topic)",
    re.IGNORECASE,
)


def topic_transition(
    previous: tuple[str, ...],
    current: tuple[str, ...],
    *,
    prompt: str = "",
    reset_similarity: float,
) -> tuple[str, float]:
    """Classify stable, pronoun continuation, or abrupt topic reset."""

    previous_set = set(previous)
    current_set = set(current)
    if not previous_set or not current_set:
        similarity = 1.0 if previous_set == current_set else 0.0
        return "stable", similarity
    similarity = len(previous_set & current_set) / len(previous_set | current_set)
    if _ABRUPT_SWITCH_RE.search(prompt):
        return "reset", similarity
    if similarity >= reset_similarity:
        return "stable", similarity
    if _PRONOUN_CONTINUATION_RE.search(prompt) and len(current_set) <= 12:
        return "continuation", similarity
    return "reset", similarity


def load_recall_field_config(
    path: Path | str | None = None,
) -> RecallFieldConfig:
    data = load_toml_file(active_config_file(Path(path)) if path else None)
    recall = data.get("recall")
    section = recall.get("field") if isinstance(recall, dict) else None
    if not isinstance(section, dict):
        return RecallFieldConfig()
    growth = section.get("growth")
    growth = growth if isinstance(growth, dict) else {}
    mode = str(section.get("mode") or "shadow").strip().lower()
    if mode not in {"off", "shadow", "candidate", "active"}:
        mode = "off"
    return RecallFieldConfig(
        mode=mode,
        canary_percent=_bounded_int(section.get("canary_percent"), 0, 0, 100),
        auto_growth=growth.get("enabled") is True,
        auto_promote=growth.get("auto_promote") is True,
        working_set_size=_bounded_int(section.get("working_set_size"), 30, 1, 100),
        max_active_nodes=_bounded_int(section.get("max_active_nodes"), 128, 1, 1_000),
        max_active_edges=_bounded_int(section.get("max_active_edges"), 256, 1, 5_000),
        positive_learning=section.get("positive_learning") is True,
        wall_half_life_seconds=_bounded_int(
            section.get("wall_half_life_seconds"), 300, 10, 86_400
        ),
        turn_decay=_bounded_float(section.get("turn_decay"), 0.82, 0.0, 1.0),
        spread_gain=_bounded_float(section.get("spread_gain"), 0.35, 0.0, 1.0),
        max_hops=_bounded_int(section.get("max_hops"), 2, 0, 4),
        global_inhibition=_bounded_float(
            section.get("global_inhibition"), 0.08, 0.0, 1.0
        ),
        refractory_turns=_bounded_int(section.get("refractory_turns"), 1, 0, 10),
        topic_reset_similarity=_bounded_float(
            section.get("topic_reset_similarity"), 0.15, 0.0, 1.0
        ),
        session_ttl_seconds=_bounded_int(
            section.get("session_ttl_seconds"),
            7 * 24 * 60 * 60,
            60,
            90 * 24 * 60 * 60,
        ),
        event_retention=_bounded_int(
            section.get("event_retention"), 2_000, 100, 20_000
        ),
    )


def _bounded_int(value: Any, default: int, minimum: int, maximum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        return default
    return max(minimum, min(maximum, value))


def _bounded_float(
    value: Any,
    default: float,
    minimum: float,
    maximum: float,
) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        return default
    return max(minimum, min(maximum, float(value)))
