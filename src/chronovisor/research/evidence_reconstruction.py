"""Strict contracts and deterministic Raw evidence reconstruction."""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum
from pathlib import Path
from typing import Any

from chronovisor.core.canonical_json import (
    canonical_json_line_bytes_strict,
    canonical_json_sha256_strict,
)
from chronovisor.core.durable_state import atomic_write_bytes
from chronovisor.core.raw_store import RawStore
from chronovisor.search.research_types import ActionType

EVALUATION_CONTRACT_SCHEMA = "chronovisor.evidence-evaluation-contract.v1"
EVIDENCE_PACKET_SCHEMA = "chronovisor.evidence-packet.v1"
EPISODE_PROJECTION_SCHEMA = "chronovisor.episode-evidence-projection.v1"
RETRIEVAL_PROGRAM_SCHEMA = "chronovisor.retrieval-program.v1"

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


class EvidenceReconstructionError(ValueError):
    """Evidence reconstruction input is invalid or incomplete."""


class EvidenceRelationKind(StrEnum):
    SUPPORTS = "supports"
    CONTRADICTS = "contradicts"
    SUPERSEDES = "supersedes"


class RetrievalStopRule(StrEnum):
    COVERAGE = "coverage"
    CONTRADICTION_RESOLVED = "contradiction_resolved"
    AS_OF_SATISFIED = "as_of_satisfied"
    ABSTAIN_ON_GAP = "abstain_on_gap"


@dataclass(frozen=True)
class MetricGate:
    metric: str
    measure: str
    lower_bound: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "metric": self.metric,
            "measure": self.measure,
            "lower_bound": self.lower_bound,
        }


@dataclass(frozen=True)
class EvaluationContract:
    contract_id: str
    candidate: str
    baseline: str
    metrics: tuple[MetricGate, ...]
    paired_slices: tuple[str, ...]
    abstention_conditions: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": EVALUATION_CONTRACT_SCHEMA,
            "contract_id": self.contract_id,
            "comparison": {"candidate": self.candidate, "baseline": self.baseline},
            "metrics": [metric.to_dict() for metric in self.metrics],
            "paired_slices": list(self.paired_slices),
            "abstention_conditions": list(self.abstention_conditions),
        }


EVALUATION_CONTRACT = EvaluationContract(
    contract_id="campaign-y-preregistered-v1",
    candidate="evidence_reconstruction",
    baseline="page_teacher",
    metrics=(
        MetricGate("answer", "paired_accuracy_delta", 0.0),
        MetricGate("evidence", "paired_evidence_coverage_delta", 0.0),
        MetricGate("temporal", "paired_temporal_accuracy_delta", 0.0),
        MetricGate("obsolete-use", "paired_obsolete_avoidance_delta", 0.0),
        MetricGate("latency", "l2_4000ms_deadline_pass_rate", 0.99),
    ),
    paired_slices=(
        "current",
        "why",
        "change",
        "failure",
        "workflow",
        "contradiction",
        "no-answer",
    ),
    abstention_conditions=(
        "missing_required_evidence",
        "unresolved_contradiction",
        "as_of_unsatisfied",
    ),
)

# This literal is the preregistration seal. Contract edits must deliberately update it.
EVALUATION_CONTRACT_SHA256 = (
    "d30047004970fb7b1fc9316ef4c1e191851e1c95afd6014edce14c89785d9e79"
)


def evaluation_contract_bytes() -> bytes:
    raw = canonical_json_line_bytes_strict(EVALUATION_CONTRACT.to_dict())
    if hashlib.sha256(raw).hexdigest() != EVALUATION_CONTRACT_SHA256:
        raise EvidenceReconstructionError("evaluation contract seal mismatch")
    return raw


def _nonempty(value: str, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise EvidenceReconstructionError(f"{field} must be a non-empty string")
    return value.strip()


def _sha256(value: str, field: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise EvidenceReconstructionError(f"{field} must be a lowercase SHA-256")
    return value


def _timestamp(value: str, field: str) -> datetime:
    _nonempty(value, field)
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise EvidenceReconstructionError(f"{field} must be ISO-8601") from exc
    if parsed.tzinfo is None:
        raise EvidenceReconstructionError(f"{field} must include a timezone")
    return parsed


@dataclass(frozen=True)
class EvidenceRef:
    raw_id: str
    byte_start: int
    byte_end: int
    raw_sha256: str
    receipt_sha256: str

    def __post_init__(self) -> None:
        if Path(self.raw_id).name != self.raw_id or not self.raw_id:
            raise EvidenceReconstructionError("raw_id must be a basename")
        if (
            isinstance(self.byte_start, bool)
            or isinstance(self.byte_end, bool)
            or not isinstance(self.byte_start, int)
            or not isinstance(self.byte_end, int)
            or self.byte_start < 0
            or self.byte_end <= self.byte_start
        ):
            raise EvidenceReconstructionError("raw byte range is invalid")
        _sha256(self.raw_sha256, "raw_sha256")
        _sha256(self.receipt_sha256, "receipt_sha256")

    def to_dict(self) -> dict[str, Any]:
        return {
            "raw_id": self.raw_id,
            "byte_range": [self.byte_start, self.byte_end],
            "byte_coordinate_space": "logical_raw",
            "raw_sha256": self.raw_sha256,
            "receipt_sha256": self.receipt_sha256,
        }


@dataclass(frozen=True)
class TimeInterval:
    start: str
    end: str

    def __post_init__(self) -> None:
        if _timestamp(self.end, "time interval end") < _timestamp(
            self.start, "time interval start"
        ):
            raise EvidenceReconstructionError("time interval end precedes start")

    def to_dict(self) -> dict[str, str]:
        return {"start": self.start, "end": self.end}


@dataclass(frozen=True)
class Provenance:
    producer: str
    source_event_id: str
    source_role: str
    event_index: int

    def __post_init__(self) -> None:
        _nonempty(self.producer, "provenance producer")
        _nonempty(self.source_event_id, "source_event_id")
        _nonempty(self.source_role, "source_role")
        if (
            isinstance(self.event_index, bool)
            or not isinstance(self.event_index, int)
            or self.event_index < 0
        ):
            raise EvidenceReconstructionError("event_index must be non-negative")

    def to_dict(self) -> dict[str, Any]:
        return {
            "producer": self.producer,
            "source_event_id": self.source_event_id,
            "source_role": self.source_role,
            "event_index": self.event_index,
        }


@dataclass(frozen=True)
class EvidenceRelation:
    kind: EvidenceRelationKind
    claim_id: str

    def __post_init__(self) -> None:
        if not isinstance(self.kind, EvidenceRelationKind):
            raise EvidenceReconstructionError("relation kind is invalid")
        _nonempty(self.claim_id, "relation claim_id")

    def to_dict(self) -> dict[str, str]:
        return {"kind": self.kind.value, "claim_id": self.claim_id}


@dataclass(frozen=True)
class EvidenceAtom:
    atom_id: str
    episode_id: str
    claim: str
    entities: tuple[str, ...]
    provenance: Provenance
    evidence: EvidenceRef
    validity: TimeInterval
    relations: tuple[EvidenceRelation, ...]

    def __post_init__(self) -> None:
        if not self.atom_id.startswith("atom:"):
            raise EvidenceReconstructionError("atom_id is invalid")
        _sha256(self.atom_id.removeprefix("atom:"), "atom_id")
        _nonempty(self.episode_id, "episode_id")
        _nonempty(self.claim, "claim")
        if (
            not isinstance(self.entities, tuple)
            or any(
                not isinstance(entity, str) or not entity.strip()
                for entity in self.entities
            )
            or len(set(self.entities)) != len(self.entities)
        ):
            raise EvidenceReconstructionError(
                "entities must be unique non-empty strings"
            )
        if (
            not isinstance(self.provenance, Provenance)
            or not isinstance(self.evidence, EvidenceRef)
            or not isinstance(self.validity, TimeInterval)
            or not isinstance(self.relations, tuple)
            or not self.relations
            or any(
                not isinstance(relation, EvidenceRelation)
                for relation in self.relations
            )
            or len(set(self.relations)) != len(self.relations)
        ):
            raise EvidenceReconstructionError("relations must be non-empty and unique")

    def to_dict(self) -> dict[str, Any]:
        return {
            "atom_id": self.atom_id,
            "episode_id": self.episode_id,
            "claim": self.claim,
            "entities": list(self.entities),
            "provenance": self.provenance.to_dict(),
            "evidence": self.evidence.to_dict(),
            "validity": self.validity.to_dict(),
            "relations": [relation.to_dict() for relation in self.relations],
        }


def _identity(prefix: str, value: object) -> str:
    digest = hashlib.sha256(canonical_json_line_bytes_strict(value)).hexdigest()
    return f"{prefix}:{digest}"


def build_evidence_atom(
    *,
    episode_id: str,
    claim: str,
    entities: Sequence[str],
    provenance: Provenance,
    evidence: EvidenceRef,
    validity: TimeInterval,
    relations: Sequence[EvidenceRelation],
) -> EvidenceAtom:
    episode_id = _nonempty(episode_id, "episode_id")
    claim = _nonempty(claim, "claim")
    if any(not isinstance(entity, str) for entity in entities):
        raise EvidenceReconstructionError("entities must be strings")
    if any(not isinstance(relation, EvidenceRelation) for relation in relations):
        raise EvidenceReconstructionError("relations are invalid")
    rows = tuple(sorted(relations, key=lambda row: (row.kind.value, row.claim_id)))
    entity_rows = tuple(sorted(dict.fromkeys(entities)))
    unsigned = {
        "episode_id": episode_id,
        "claim": claim,
        "entities": list(entity_rows),
        "provenance": provenance.to_dict(),
        "evidence": evidence.to_dict(),
        "validity": validity.to_dict(),
        "relations": [relation.to_dict() for relation in rows],
    }
    return EvidenceAtom(
        atom_id=_identity("atom", unsigned),
        episode_id=episode_id,
        claim=claim,
        entities=entity_rows,
        provenance=provenance,
        evidence=evidence,
        validity=validity,
        relations=rows,
    )


@dataclass(frozen=True)
class EvidencePacket:
    packet_id: str
    query: str
    as_of: str
    retrieval_program_id: str
    atoms: tuple[EvidenceAtom, ...]
    abstained: bool = False
    abstention_reason: str = ""

    def __post_init__(self) -> None:
        if not self.packet_id.startswith("packet:"):
            raise EvidenceReconstructionError("packet_id is invalid")
        _sha256(self.packet_id.removeprefix("packet:"), "packet_id")
        _nonempty(self.query, "query")
        _timestamp(self.as_of, "as_of")
        if not self.retrieval_program_id.startswith("program:"):
            raise EvidenceReconstructionError("retrieval_program_id is invalid")
        _sha256(
            self.retrieval_program_id.removeprefix("program:"),
            "retrieval_program_id",
        )
        if (
            not isinstance(self.atoms, tuple)
            or any(not isinstance(atom, EvidenceAtom) for atom in self.atoms)
            or len({atom.atom_id for atom in self.atoms}) != len(self.atoms)
        ):
            raise EvidenceReconstructionError("packet atoms must be unique")
        if not isinstance(self.abstained, bool):
            raise EvidenceReconstructionError("abstained must be boolean")
        if self.abstained != bool(self.abstention_reason):
            raise EvidenceReconstructionError("abstention requires exactly one reason")
        if not self.atoms and not self.abstained:
            raise EvidenceReconstructionError("empty evidence requires abstention")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": EVIDENCE_PACKET_SCHEMA,
            "packet_id": self.packet_id,
            "query": self.query,
            "as_of": self.as_of,
            "retrieval_program_id": self.retrieval_program_id,
            "atoms": [atom.to_dict() for atom in self.atoms],
            "abstained": self.abstained,
            "abstention_reason": self.abstention_reason,
        }

    def canonical_bytes(self) -> bytes:
        return canonical_json_line_bytes_strict(self.to_dict())


def build_evidence_packet(
    *,
    query: str,
    as_of: str,
    retrieval_program_id: str,
    atoms: Sequence[EvidenceAtom],
    abstention_reason: str = "",
) -> EvidencePacket:
    query = _nonempty(query, "query")
    abstention_reason = abstention_reason.strip()
    atom_rows = tuple(sorted(atoms, key=lambda atom: atom.atom_id))
    unsigned = {
        "query": query,
        "as_of": as_of,
        "retrieval_program_id": retrieval_program_id,
        "atoms": [atom.to_dict() for atom in atom_rows],
        "abstained": bool(abstention_reason),
        "abstention_reason": abstention_reason,
    }
    return EvidencePacket(
        packet_id=_identity("packet", unsigned),
        query=query,
        as_of=as_of,
        retrieval_program_id=retrieval_program_id,
        atoms=atom_rows,
        abstained=bool(abstention_reason),
        abstention_reason=abstention_reason,
    )


def _event_semantics(host: str, event: Mapping[str, Any]) -> tuple[str, str]:
    event_type = event.get("type")
    if host == "codex":
        from chronovisor.core.codex_transcript import codex_semantic_view

        return codex_semantic_view(event_type, event.get("payload"))
    if host == "claude-code":
        from chronovisor.core.claude_code_transcript import claude_semantic_view

        message = event.get("message")
        content = message.get("content") if isinstance(message, dict) else None
        return claude_semantic_view(event_type, content)
    raise EvidenceReconstructionError(f"unsupported committed Raw host: {host}")


def _event_entities(event: Mapping[str, Any]) -> tuple[str, ...]:
    candidates: object = event.get("entities")
    payload = event.get("payload")
    if candidates is None and isinstance(payload, Mapping):
        candidates = payload.get("entities")
    if not isinstance(candidates, list):
        return ()
    return tuple(
        sorted(
            {
                item.strip()
                for item in candidates
                if isinstance(item, str) and item.strip()
            }
        )
    )


def _event_time(event: Mapping[str, Any], captured_at: str) -> str:
    if "timestamp" not in event:
        _timestamp(captured_at, "receipt captured_at")
        return captured_at
    timestamp = event["timestamp"]
    if not isinstance(timestamp, str):
        raise EvidenceReconstructionError("event timestamp must be a string")
    _timestamp(timestamp, "event timestamp")
    return timestamp


def build_episode_projection(raw_dir: Path) -> bytes:
    """Rebuild canonical episode evidence from committed Raw v2 receipts only."""

    store = RawStore(raw_dir, mode="v2")
    receipts: list[dict[str, Any]] = []
    atoms: list[EvidenceAtom] = []
    for unit in store.iter_segment_units():
        commit = unit.commit
        if commit is None or unit.sha256 is None or unit.captured_at is None:
            raise EvidenceReconstructionError("Raw unit has no committed receipt")
        raw = store.read_bytes(unit)
        lines = raw.splitlines(keepends=True)
        if not raw.endswith(b"\n") or len(lines) != commit.record_count:
            raise EvidenceReconstructionError("committed Raw record count is invalid")
        receipt_sha256 = canonical_json_sha256_strict(commit.to_dict())
        receipts.append(
            {
                "raw_id": unit.raw_id,
                "byte_range": [0, unit.length],
                "byte_coordinate_space": "logical_raw",
                "raw_sha256": unit.sha256,
                "receipt_sha256": receipt_sha256,
                "captured_at": unit.captured_at,
                "host": commit.host,
                "session_key": commit.session_key,
                "source_line_range": [commit.after_line, commit.until_line],
            }
        )
        cursor = 0
        episode_id = _identity(
            "episode", {"host": commit.host, "session_key": commit.session_key}
        )
        for index, line in enumerate(lines):
            try:
                event = json.loads(line)
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise EvidenceReconstructionError(
                    f"committed Raw event {unit.raw_id}:{index} is invalid JSON"
                ) from exc
            if not isinstance(event, dict):
                raise EvidenceReconstructionError(
                    "committed Raw event must be an object"
                )
            instant = _event_time(event, commit.captured_at)
            role, text = _event_semantics(commit.host, event)
            claim = text.strip()
            if claim:
                start = cursor
                evidence = EvidenceRef(
                    raw_id=unit.raw_id,
                    byte_start=start,
                    byte_end=start + len(line),
                    raw_sha256=commit.sha256,
                    receipt_sha256=receipt_sha256,
                )
                claim_id = _identity(
                    "claim", {"episode_id": episode_id, "claim": claim}
                )
                atoms.append(
                    build_evidence_atom(
                        episode_id=episode_id,
                        claim=claim,
                        entities=_event_entities(event),
                        provenance=Provenance(
                            producer="committed-raw-receipt",
                            source_event_id=f"{unit.raw_id}:{index}",
                            source_role=role,
                            event_index=index,
                        ),
                        evidence=evidence,
                        validity=TimeInterval(start=instant, end=instant),
                        relations=(
                            EvidenceRelation(EvidenceRelationKind.SUPPORTS, claim_id),
                        ),
                    )
                )
            cursor += len(line)
    unsigned = {
        "schema": EPISODE_PROJECTION_SCHEMA,
        "source_receipts": receipts,
        "atoms": [
            atom.to_dict() for atom in sorted(atoms, key=lambda row: row.atom_id)
        ],
    }
    return canonical_json_line_bytes_strict(
        {"projection_id": _identity("projection", unsigned), **unsigned}
    )


def rebuild_episode_projection(raw_dir: Path, output_path: Path) -> bytes:
    raw_root = raw_dir.expanduser().resolve(strict=False)
    target = output_path.expanduser().resolve(strict=False)
    if target == raw_root or target.is_relative_to(raw_root):
        raise EvidenceReconstructionError("derived projection must be outside raw/")
    raw = build_episode_projection(raw_root)
    atomic_write_bytes(target, raw, backup=False, min_free_bytes=0)
    return raw


@dataclass(frozen=True)
class ClaimSlot:
    slot_id: str
    claim: str

    def to_dict(self) -> dict[str, str]:
        return {"slot_id": self.slot_id, "claim": self.claim}


@dataclass(frozen=True)
class RequiredEvidence:
    claim_slot: str
    minimum_atoms: int
    relations: tuple[EvidenceRelationKind, ...]
    must_match_as_of: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "claim_slot": self.claim_slot,
            "minimum_atoms": self.minimum_atoms,
            "relations": [relation.value for relation in self.relations],
            "must_match_as_of": self.must_match_as_of,
        }


@dataclass(frozen=True)
class RetrievalProgram:
    program_id: str
    query: str
    as_of: str
    claim_slots: tuple[ClaimSlot, ...]
    required_evidence: tuple[RequiredEvidence, ...]
    allowed_actions: tuple[ActionType, ...]
    stop_rules: tuple[RetrievalStopRule, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": RETRIEVAL_PROGRAM_SCHEMA,
            "program_id": self.program_id,
            "query": self.query,
            "as_of": self.as_of,
            "claim_slots": [slot.to_dict() for slot in self.claim_slots],
            "required_evidence": [row.to_dict() for row in self.required_evidence],
            "allowed_actions": [action.value for action in self.allowed_actions],
            "stop_rules": [rule.value for rule in self.stop_rules],
        }

    def canonical_bytes(self) -> bytes:
        return canonical_json_line_bytes_strict(self.to_dict())


def _strict_fields(value: object, required: set[str], field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise EvidenceReconstructionError(f"{field} must be an object")
    if any(not isinstance(key, str) for key in value):
        raise EvidenceReconstructionError(f"{field} keys must be strings")
    observed = set(value)
    if observed != required:
        missing = sorted(required - observed)
        unknown = sorted(observed - required)
        raise EvidenceReconstructionError(
            f"{field} fields are invalid: missing={missing}, unknown={unknown}"
        )
    return value


def compile_retrieval_program(query: str, plan: Mapping[str, Any]) -> RetrievalProgram:
    """Compile and validate a complete plan before any research action runs."""

    query = _nonempty(query, "query")
    root = _strict_fields(
        plan,
        {"as_of", "claim_slots", "required_evidence", "allowed_actions", "stop_rules"},
        "retrieval plan",
    )
    as_of = root["as_of"]
    if not isinstance(as_of, str):
        raise EvidenceReconstructionError("as_of must be a string")
    _timestamp(as_of, "as_of")

    raw_slots = root["claim_slots"]
    if not isinstance(raw_slots, list) or not raw_slots:
        raise EvidenceReconstructionError("claim_slots must be a non-empty list")
    slots: list[ClaimSlot] = []
    for index, raw_slot in enumerate(raw_slots):
        slot = _strict_fields(raw_slot, {"slot_id", "claim"}, f"claim_slots[{index}]")
        slot_id = slot["slot_id"]
        claim = slot["claim"]
        if not isinstance(slot_id, str) or not isinstance(claim, str):
            raise EvidenceReconstructionError("claim slot values must be strings")
        slots.append(
            ClaimSlot(_nonempty(slot_id, "slot_id"), _nonempty(claim, "claim"))
        )
    slot_ids = [slot.slot_id for slot in slots]
    if len(set(slot_ids)) != len(slot_ids):
        raise EvidenceReconstructionError("claim slot IDs must be unique")
    slots.sort(key=lambda slot: slot.slot_id)

    raw_requirements = root["required_evidence"]
    if not isinstance(raw_requirements, list) or not raw_requirements:
        raise EvidenceReconstructionError("required_evidence must be non-empty")
    requirements: list[RequiredEvidence] = []
    for index, raw_requirement in enumerate(raw_requirements):
        requirement = _strict_fields(
            raw_requirement,
            {"claim_slot", "minimum_atoms", "relations", "must_match_as_of"},
            f"required_evidence[{index}]",
        )
        claim_slot = requirement["claim_slot"]
        minimum_atoms = requirement["minimum_atoms"]
        raw_relations = requirement["relations"]
        must_match_as_of = requirement["must_match_as_of"]
        if not isinstance(claim_slot, str) or claim_slot not in slot_ids:
            raise EvidenceReconstructionError(
                "required evidence has unknown claim_slot"
            )
        if (
            isinstance(minimum_atoms, bool)
            or not isinstance(minimum_atoms, int)
            or minimum_atoms < 1
        ):
            raise EvidenceReconstructionError("minimum_atoms must be positive")
        if not isinstance(raw_relations, list) or not raw_relations:
            raise EvidenceReconstructionError("relations must be non-empty")
        try:
            relations = tuple(
                sorted(
                    (EvidenceRelationKind(value) for value in raw_relations),
                    key=lambda relation: relation.value,
                )
            )
        except (TypeError, ValueError) as exc:
            raise EvidenceReconstructionError("required relation is invalid") from exc
        if len(set(relations)) != len(relations):
            raise EvidenceReconstructionError("required relations must be unique")
        if not isinstance(must_match_as_of, bool):
            raise EvidenceReconstructionError("must_match_as_of must be boolean")
        requirements.append(
            RequiredEvidence(
                claim_slot=claim_slot,
                minimum_atoms=minimum_atoms,
                relations=relations,
                must_match_as_of=must_match_as_of,
            )
        )
    if sorted(row.claim_slot for row in requirements) != sorted(slot_ids):
        raise EvidenceReconstructionError(
            "required_evidence must cover every claim slot exactly once"
        )
    requirements.sort(key=lambda row: row.claim_slot)

    raw_actions = root["allowed_actions"]
    if not isinstance(raw_actions, list) or not raw_actions:
        raise EvidenceReconstructionError("allowed_actions must be non-empty")
    try:
        actions = tuple(
            sorted(
                (ActionType(value) for value in raw_actions),
                key=lambda action: action.value,
            )
        )
    except (TypeError, ValueError) as exc:
        raise EvidenceReconstructionError("allowed action is invalid") from exc
    if ActionType.FINISH in actions or len(set(actions)) != len(actions):
        raise EvidenceReconstructionError(
            "allowed actions must be unique retrieval actions"
        )

    raw_rules = root["stop_rules"]
    if not isinstance(raw_rules, list):
        raise EvidenceReconstructionError("stop_rules must be a list")
    try:
        parsed_rules = tuple(RetrievalStopRule(value) for value in raw_rules)
    except (TypeError, ValueError) as exc:
        raise EvidenceReconstructionError("stop rule is invalid") from exc
    if set(parsed_rules) != set(RetrievalStopRule) or len(parsed_rules) != len(
        RetrievalStopRule
    ):
        raise EvidenceReconstructionError("stop_rules are incomplete or duplicated")
    rules = tuple(RetrievalStopRule)

    unsigned = {
        "query": query,
        "as_of": as_of,
        "claim_slots": [slot.to_dict() for slot in slots],
        "required_evidence": [row.to_dict() for row in requirements],
        "allowed_actions": [action.value for action in actions],
        "stop_rules": [rule.value for rule in rules],
    }
    return RetrievalProgram(
        program_id=_identity("program", unsigned),
        query=query,
        as_of=as_of,
        claim_slots=tuple(slots),
        required_evidence=tuple(requirements),
        allowed_actions=actions,
        stop_rules=rules,
    )
