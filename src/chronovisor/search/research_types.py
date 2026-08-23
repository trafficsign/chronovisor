"""Typed contracts for bounded agentic memory research runs."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from enum import StrEnum
from typing import Any


class ActionType(StrEnum):
    WIKI_SEARCH = "chronovisor_search"
    WIKI_READ = "chronovisor_read"
    WIKI_NEIGHBORS = "wiki_neighbors"
    VERIFIED_CLAIMS = "verified_claims"
    RAW_SEARCH = "raw_search"
    WEB_SEARCH = "web_search"
    WEB_FETCH = "web_fetch"
    FINISH = "finish"


class StopReason(StrEnum):
    COMPLETED = "completed"
    ADMISSION_DENIED = "admission_denied"
    BUDGET_EXHAUSTED = "budget_exhausted"
    CANCELLED_FOR_SYNC = "cancelled_for_sync"
    DUPLICATE_ACTION = "duplicate_action"
    MALFORMED_ACTION = "malformed_action"
    MODEL_TIMEOUT = "model_timeout"
    TOOL_ERROR = "tool_error"
    NO_PROGRESS = "no_progress"
    INTERRUPTED = "interrupted"


class ClaimKind(StrEnum):
    STABLE = "stable"
    FRESHNESS_SENSITIVE = "freshness-sensitive"
    USER_REPORTED = "user-reported"


class ClaimStatus(StrEnum):
    SUPPORTED = "supported"
    CONTRADICTED = "contradicted"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class Action:
    type: ActionType
    arguments: dict[str, Any] = field(default_factory=dict)
    rationale: str = ""
    epoch: int = 0

    def canonical_key(self) -> str:
        import json

        return json.dumps(
            {"type": self.type.value, "arguments": self.arguments},
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            default=str,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": self.type.value,
            "arguments": self.arguments,
            "rationale": self.rationale,
            "epoch": self.epoch,
        }


@dataclass(frozen=True)
class Observation:
    action: Action
    status: str
    preview: str
    artifact_id: str = ""
    bytes: int = 0
    citations: tuple[str, ...] = ()
    latency_ms: int = 0
    error: str = ""
    terminal: bool = True
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["action"] = self.action.to_dict()
        payload["citations"] = list(self.citations)
        return payload


@dataclass(frozen=True)
class EvidenceArtifact:
    artifact_id: str
    source_type: str
    source_uri: str
    retrieved_at: str
    sha256: str
    byte_length: int
    preview: str
    trust: str = "untrusted"
    title: str = ""
    mime_type: str = "text/plain"
    citation: str = ""
    durable: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ResearchBudget:
    max_iterations: int = 5
    max_total_wall_seconds: float = 90.0
    max_single_generation_seconds: float = 30.0
    max_single_generation_tokens: int = 512
    max_planner_calls: int = 5
    max_challenge_calls: int = 2
    max_tie_break_calls: int = 1
    max_repair_calls: int = 2
    max_total_model_calls: int = 10
    max_searches: int = 8
    max_fetches: int = 5
    max_observation_bytes: int = 200_000


@dataclass
class BudgetUsage:
    iterations: int = 0
    planner_calls: int = 0
    challenge_calls: int = 0
    tie_break_calls: int = 0
    repair_calls: int = 0
    searches: int = 0
    fetches: int = 0
    observation_bytes: int = 0

    @property
    def total_model_calls(self) -> int:
        return (
            self.planner_calls
            + self.challenge_calls
            + self.tie_break_calls
            + self.repair_calls
        )

    def can_consume(self, budget: ResearchBudget, kind: str, amount: int = 1) -> bool:
        if amount < 0:
            return False
        limits = {
            "iterations": budget.max_iterations,
            "planner_calls": budget.max_planner_calls,
            "challenge_calls": budget.max_challenge_calls,
            "tie_break_calls": budget.max_tie_break_calls,
            "repair_calls": budget.max_repair_calls,
            "searches": budget.max_searches,
            "fetches": budget.max_fetches,
            "observation_bytes": budget.max_observation_bytes,
        }
        if kind not in limits:
            raise KeyError(kind)
        if (
            kind.endswith("_calls")
            and self.total_model_calls + amount > budget.max_total_model_calls
        ):
            return False
        return int(getattr(self, kind)) + amount <= limits[kind]

    def consume(self, budget: ResearchBudget, kind: str, amount: int = 1) -> bool:
        if not self.can_consume(budget, kind, amount):
            return False
        setattr(self, kind, int(getattr(self, kind)) + amount)
        return True

    def to_dict(self) -> dict[str, int]:
        payload = asdict(self)
        payload["total_model_calls"] = self.total_model_calls
        return payload


@dataclass(frozen=True)
class ParsedAction:
    action: Action | None
    error: str = ""


def parse_action(value: Any, *, epoch: int) -> ParsedAction:
    """Validate the strict planner protocol without executing malformed JSON."""

    if not isinstance(value, Mapping):
        return ParsedAction(None, "action must be an object")
    allowed = {"type", "arguments", "rationale"}
    unknown = sorted(set(value) - allowed)
    if unknown:
        return ParsedAction(None, f"unknown action fields: {', '.join(unknown)}")
    kind = value.get("type")
    try:
        action_type = ActionType(kind)
    except (TypeError, ValueError):
        return ParsedAction(None, f"unknown action type: {kind!r}")
    arguments = value.get("arguments", {})
    if not isinstance(arguments, Mapping):
        return ParsedAction(None, "action arguments must be an object")
    argument_contracts: dict[ActionType, tuple[frozenset[str], str]] = {
        ActionType.WIKI_SEARCH: (frozenset({"query", "limit", "semantic"}), "query"),
        ActionType.WIKI_READ: (frozenset({"page_id", "max_chars"}), "page_id"),
        ActionType.WIKI_NEIGHBORS: (frozenset({"page_id"}), "page_id"),
        ActionType.VERIFIED_CLAIMS: (frozenset({"query"}), "query"),
        ActionType.RAW_SEARCH: (frozenset({"query", "limit", "scan_limit"}), "query"),
        ActionType.WEB_SEARCH: (frozenset({"query", "limit"}), "query"),
        ActionType.WEB_FETCH: (frozenset({"url"}), "url"),
        ActionType.FINISH: (frozenset({"answer"}), "answer"),
    }
    allowed_arguments, required_argument = argument_contracts[action_type]
    unknown_arguments = sorted(set(arguments) - allowed_arguments)
    if unknown_arguments:
        return ParsedAction(
            None,
            f"unknown {action_type.value} arguments: {', '.join(unknown_arguments)}",
        )
    required_value = arguments.get(required_argument)
    if not isinstance(required_value, str) or not required_value.strip():
        return ParsedAction(
            None,
            f"{action_type.value} requires non-empty {required_argument}",
        )
    rationale = value.get("rationale", "")
    if not isinstance(rationale, str):
        return ParsedAction(None, "action rationale must be a string")
    return ParsedAction(
        Action(
            type=action_type,
            arguments=dict(arguments),
            rationale=rationale[:500],
            epoch=max(0, int(epoch)),
        )
    )


ACTION_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["type", "arguments", "rationale"],
    "properties": {
        "type": {"type": "string", "enum": [item.value for item in ActionType]},
        "arguments": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "query": {"type": "string", "maxLength": 500},
                "page_id": {"type": "string", "maxLength": 500},
                "url": {"type": "string", "maxLength": 2_000},
                "answer": {"type": "string", "maxLength": 4_000},
                "limit": {"type": "integer", "minimum": 1, "maximum": 20},
                "max_chars": {"type": "integer", "minimum": 100, "maximum": 50_000},
                "scan_limit": {"type": "integer", "minimum": 10, "maximum": 5_000},
                "semantic": {"type": "boolean"},
            },
        },
        "rationale": {"type": "string", "maxLength": 500},
    },
}


def _ollama_action_format_schema(value: Any) -> Any:
    """Drop grammar-only length bounds unsupported by some GGUF runners."""

    if isinstance(value, dict):
        return {
            key: _ollama_action_format_schema(item)
            for key, item in value.items()
            if key != "maxLength"
        }
    if isinstance(value, list):
        return [_ollama_action_format_schema(item) for item in value]
    return value


# Client validation continues to use ACTION_SCHEMA. This relaxed copy is only
# sent to Ollama's grammar compiler; the session repairs or rejects any value
# that violates the full schema above.
ACTION_FORMAT_SCHEMA: dict[str, Any] = _ollama_action_format_schema(ACTION_SCHEMA)

CHALLENGE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": [
        "verdict",
        "unsupported_claims",
        "contradictions",
        "injection_detected",
        "rationale",
    ],
    "properties": {
        "verdict": {
            "type": "string",
            "enum": ["confirm", "reject", "inconclusive"],
        },
        "unsupported_claims": {
            "type": "array",
            "maxItems": 20,
            "items": {"type": "string", "maxLength": 500},
        },
        "contradictions": {
            "type": "array",
            "maxItems": 20,
            "items": {"type": "string", "maxLength": 500},
        },
        "injection_detected": {"type": "boolean"},
        "rationale": {"type": "string", "maxLength": 1_000},
    },
}

TIE_BREAK_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["choice", "rationale"],
    "properties": {
        "choice": {
            "type": "string",
            "enum": ["planner", "challenger", "unknown"],
        },
        "rationale": {"type": "string", "maxLength": 1_000},
    },
}
