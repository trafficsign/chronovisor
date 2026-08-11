"""Local-only evidence reconstruction runtime and rollout seam."""

from __future__ import annotations

import hashlib
import json
import os
import re
import stat
import time
from collections.abc import Iterator, Mapping, Sequence
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from chronovisor.core.canonical_json import (
    canonical_json_line_bytes_strict,
    canonical_json_sha256_strict,
)
from chronovisor.core.durable_state import (
    atomic_write_bytes_at,
    canonical_bytes,
    file_lock,
    open_directory_nofollow,
    open_regular_nofollow,
    seal_object,
    verify_sealed_object,
)
from chronovisor.core.knowledge_graph_rollout import (
    CANARY_SAMPLE_UNIT,
    CANARY_STEPS,
    selected_for_canary,
)
from chronovisor.core.raw_store import RawStore
from chronovisor.core.store import okf_runtime_operation, okf_startup_status
from chronovisor.research.evidence_reconstruction import (
    EVALUATION_CONTRACT,
    EVALUATION_CONTRACT_SHA256,
    EVIDENCE_AUTHORITY_ROLES,
    EVIDENCE_PACKET_SCHEMA,
    EpisodeProjection,
    EvidenceAtom,
    EvidencePacket,
    EvidenceReconstructionError,
    EvidenceRef,
    EvidenceRelation,
    EvidenceRelationKind,
    Provenance,
    RequiredEvidence,
    RetrievalProgram,
    TimeInterval,
    build_episode_projection,
    build_evidence_atom,
    build_evidence_packet,
    committed_raw_watermark,
    compile_retrieval_program,
    load_episode_projection,
    physical_raw_inventory,
    verify_projection_atom,
)
from chronovisor.research.research_tools import ToolContext, execute_tool
from chronovisor.search.research_types import Action, ActionType

_TOKEN_RE = re.compile(r"[a-z0-9]{3,}|[\u3040-\u30ff\u3400-\u9fff]{2,}")
_LOCAL_ACTIONS = frozenset({ActionType.RAW_SEARCH})
_L2_PROJECTION_SCAN_LIMIT = 128
_AUTHORITY_LOCK = "authority.lock"
EVIDENCE_ACCEPTANCE_SCHEMA = "chronovisor.evidence-acceptance.v1"
_PROMOTION_FIELDS = {
    "schema_version",
    "generated_at",
    "mode",
    "canary_percent",
    "stage_started_sample_count",
    "sample_count",
    "sample_unit",
    "gates",
    "reason",
    "rollback_reason",
    "rollback_teacher",
    "manifest_sha256",
    "relation_snapshot_sha256",
    "rubric_sha256",
    "model_manifest_sha256",
    "seal_sha256",
}
_ACCEPTANCE_FIELDS = {
    "schema",
    "contract_sha256",
    "projection_sha256",
    "raw_watermark_sha256",
    "raw_before_sha256",
    "raw_after_sha256",
    "raw_stat_sha256",
    "evaluation_sha256",
    "case_manifest_sha256",
    "case_count",
    "gates",
    "atomic_publication_fault_sha256",
    "relation_semantics_sha256",
    "raw_relation_mode",
    "seal_sha256",
}


def evidence_projection_path(root: Path) -> Path:
    return root / "runtime" / "evidence-reconstruction" / "episode-projection.json"


def _evidence_promotion_path(root: Path) -> Path:
    return root / "runtime" / "evidence-reconstruction" / "promotion.json"


def evidence_acceptance_path(root: Path) -> Path:
    return root / "runtime" / "evidence-reconstruction" / "acceptance.json"


def evidence_cases_path(root: Path) -> Path:
    return root / "runtime" / "evidence-reconstruction" / "cases.json"


def _open_or_create_directory(parent_fd: int, name: str) -> int:
    try:
        os.mkdir(name, mode=0o700, dir_fd=parent_fd)
    except FileExistsError:
        pass
    return os.open(
        name,
        os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | getattr(os, "O_CLOEXEC", 0),
        dir_fd=parent_fd,
    )


@contextmanager
def evidence_authority_operation(root: Path) -> Iterator[int]:
    """Serialize evidence authority writes below pinned, nofollow directories."""

    with open_directory_nofollow(root) as root_fd:
        runtime_fd = _open_or_create_directory(root_fd, "runtime")
        try:
            authority_fd = _open_or_create_directory(
                runtime_fd, "evidence-reconstruction"
            )
            try:
                with ExitStack() as stack:
                    for attempt in range(2):
                        try:
                            stack.enter_context(
                                file_lock(Path(_AUTHORITY_LOCK), dir_fd=authority_fd)
                            )
                            break
                        except FileNotFoundError:
                            if attempt:
                                raise
                    yield authority_fd
            finally:
                os.close(authority_fd)
        finally:
            os.close(runtime_fd)


def evidence_authority_entry_exists(directory_fd: int, name: str) -> bool:
    try:
        os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
    except FileNotFoundError:
        return False
    return True


def read_evidence_bytes_at(directory_fd: int, name: str) -> bytes:
    descriptor = os.open(
        name,
        os.O_RDONLY | os.O_NOFOLLOW | getattr(os, "O_CLOEXEC", 0),
        dir_fd=directory_fd,
    )
    try:
        if not stat.S_ISREG(os.fstat(descriptor).st_mode):
            raise EvidenceReconstructionError("evidence authority is not regular")
        chunks: list[bytes] = []
        while chunk := os.read(descriptor, 1024 * 1024):
            chunks.append(chunk)
        return b"".join(chunks)
    finally:
        os.close(descriptor)


def write_evidence_authority_at(
    directory_fd: int, name: str, payload: Mapping[str, Any]
) -> dict[str, Any]:
    """Seal and atomically publish one authority object under an existing lease."""

    sealed = seal_object(payload)
    atomic_write_bytes_at(directory_fd, name, canonical_bytes(sealed))
    return sealed


def read_evidence_authority(path: Path) -> dict[str, Any]:
    """Read one canonical sealed authority file without following symlinks."""

    try:
        with open_regular_nofollow(path) as stream:
            raw = stream.read()
        payload = verify_sealed_object(json.loads(raw))
    except Exception as exc:
        raise EvidenceReconstructionError("evidence authority is invalid") from exc
    if canonical_json_line_bytes_strict(payload) != raw:
        raise EvidenceReconstructionError("evidence authority is not canonical")
    return payload


def raw_stat_watermark(raw_dir: Path) -> str:
    """Return a cheap nofollow tamper signal without reading Raw content."""

    try:
        root_mode = raw_dir.lstat().st_mode
    except OSError as exc:
        raise EvidenceReconstructionError("Raw stat root is invalid") from exc
    if not stat.S_ISDIR(root_mode) or stat.S_ISLNK(root_mode):
        raise EvidenceReconstructionError("Raw stat root is invalid")
    rows: list[dict[str, Any]] = []
    for path in sorted(raw_dir.rglob("*")):
        try:
            observed = path.lstat()
        except OSError as exc:
            raise EvidenceReconstructionError("Raw stat entry is invalid") from exc
        if stat.S_ISDIR(observed.st_mode):
            continue
        if not stat.S_ISREG(observed.st_mode) or stat.S_ISLNK(observed.st_mode):
            raise EvidenceReconstructionError("Raw stat entry is non-regular")
        rows.append(
            {
                "path": path.relative_to(raw_dir).as_posix(),
                "size": observed.st_size,
                "mtime_ns": observed.st_mtime_ns,
                "ctime_ns": observed.st_ctime_ns,
                "inode": observed.st_ino,
            }
        )
    return canonical_json_sha256_strict(rows)


def evidence_rollout_gate_keys() -> frozenset[str]:
    metrics = tuple(gate.metric for gate in EVALUATION_CONTRACT.metrics)
    return frozenset(
        {
            *metrics,
            *(
                f"slice:{metric}:{slice_name}"
                for metric in metrics
                for slice_name in EVALUATION_CONTRACT.paired_slices
            ),
            "safe_abstention",
            "answerable_execution",
            "bounded_context",
            "cloud_egress_zero",
            "projection_deterministic",
            "raw_unchanged",
            "atomic_publication_fault",
            "relation_semantics",
            "activation:okf_finalized",
        }
    )


def run_projection_cycle_at(*, raw_dir: Path, directory_fd: int) -> EpisodeProjection:
    """Build and publish a projection while holding the authority lease."""

    raw = build_episode_projection(raw_dir)
    projection = load_episode_projection(raw)
    atomic_write_bytes_at(directory_fd, "episode-projection.json", raw)
    if projection.canonical_bytes() != raw:
        raise EvidenceReconstructionError("projection read-back mismatch")
    return projection


def run_projection_cycle(*, raw_dir: Path, output_path: Path) -> EpisodeProjection:
    """Build and strictly publish the fixed committed-Raw projection."""

    target = output_path.expanduser().absolute()
    root = target.parents[2]
    if target != evidence_projection_path(root) or raw_dir.expanduser().absolute() != (
        root / "raw"
    ):
        raise EvidenceReconstructionError("projection path is not authoritative")
    if not _okf_finalized(root):
        raise EvidenceReconstructionError("Campaign X is not finalized")
    with okf_runtime_operation(root) as locked:
        if not (
            locked.allowed
            and locked.layout == "okf_v0_2"
            and locked.state == "finalized-v2"
        ):
            raise EvidenceReconstructionError("Campaign X changed during projection")
        with evidence_authority_operation(root) as directory_fd:
            return run_projection_cycle_at(raw_dir=raw_dir, directory_fd=directory_fd)


def compile_projection_program(query: str, as_of: str) -> RetrievalProgram:
    """Compile the deterministic one-slot local projection program."""

    return compile_retrieval_program(
        query,
        {
            "as_of": as_of,
            "claim_slots": [{"slot_id": "answer", "claim": query}],
            "required_evidence": [
                {
                    "claim_slot": "answer",
                    "minimum_atoms": 1,
                    "relations": ["supports"],
                    "must_match_as_of": True,
                }
            ],
            "allowed_actions": [action.value for action in sorted(_LOCAL_ACTIONS)],
            "stop_rules": [
                "coverage",
                "contradiction_resolved",
                "as_of_satisfied",
                "abstain_on_gap",
            ],
        },
    )


class EvidenceLedger:
    """Track exact atoms against required claim slots without mutation authority."""

    def __init__(
        self,
        program: RetrievalProgram,
        authority_roles: Sequence[str] = EVIDENCE_AUTHORITY_ROLES,
    ):
        self.program = program
        self.authority_roles = frozenset(authority_roles)
        self._atoms: dict[str, dict[str, EvidenceAtom]] = {
            slot.slot_id: {} for slot in program.claim_slots
        }
        self._future: dict[str, int] = dict.fromkeys(self._atoms, 0)
        self._expired: dict[str, int] = dict.fromkeys(self._atoms, 0)

    def add(self, slot_id: str, atom: EvidenceAtom) -> bool:
        if slot_id not in self._atoms:
            raise EvidenceReconstructionError("ledger claim slot is invalid")
        requirement = self._requirement(slot_id)
        slot = next(row for row in self.program.claim_slots if row.slot_id == slot_id)
        if not _atom_matches_slot(
            claim=slot.claim,
            requirement=requirement,
            atom=atom,
            authority_roles=self.authority_roles,
        ):
            return False
        if requirement.must_match_as_of:
            start = datetime.fromisoformat(atom.validity.start.replace("Z", "+00:00"))
            end = datetime.fromisoformat(atom.validity.end.replace("Z", "+00:00"))
            as_of = datetime.fromisoformat(self.program.as_of.replace("Z", "+00:00"))
            if start > as_of:
                self._future[slot_id] += 1
                return False
            if start < end < as_of:
                self._expired[slot_id] += 1
                return False
        self._atoms[slot_id][atom.atom_id] = atom
        return True

    def _requirement(self, slot_id: str) -> RequiredEvidence:
        return next(
            row for row in self.program.required_evidence if row.claim_slot == slot_id
        )

    def slot_state(self, slot_id: str) -> dict[str, Any]:
        requirement = self._requirement(slot_id)
        atoms = tuple(self._atoms[slot_id].values())
        superseded = {
            relation.claim_id
            for atom in atoms
            for relation in atom.relations
            if relation.kind == EvidenceRelationKind.SUPERSEDES
        }
        active_relations: dict[str, set[EvidenceRelationKind]] = {}
        for atom in atoms:
            for relation in atom.relations:
                if (
                    relation.kind != EvidenceRelationKind.SUPERSEDES
                    and relation.claim_id not in superseded
                ):
                    active_relations.setdefault(relation.claim_id, set()).add(
                        relation.kind
                    )
        contradiction = any(
            {
                EvidenceRelationKind.SUPPORTS,
                EvidenceRelationKind.CONTRADICTS,
            }.issubset(kinds)
            for kinds in active_relations.values()
        )
        covered_atoms = {
            atom.atom_id
            for atom in atoms
            if any(
                relation.kind in requirement.relations
                and (
                    relation.kind == EvidenceRelationKind.SUPERSEDES
                    or relation.claim_id not in superseded
                )
                for relation in atom.relations
            )
        }
        covered = len(covered_atoms) >= requirement.minimum_atoms and not contradiction
        return {
            "covered": covered,
            "contradiction": contradiction,
            "as_of_satisfied": covered,
            "atom_count": len(atoms),
            "covered_atom_count": len(covered_atoms),
            "future_count": self._future[slot_id],
            "expired_count": self._expired[slot_id],
            "superseded_count": len(superseded),
        }

    def gaps(self) -> tuple[str, ...]:
        return tuple(
            slot.slot_id
            for slot in self.program.claim_slots
            if not self.slot_state(slot.slot_id)["covered"]
        )

    def covered(self) -> bool:
        return not self.gaps() and all(
            self.slot_state(slot.slot_id)["as_of_satisfied"]
            for slot in self.program.claim_slots
        )

    def unresolved_contradiction(self) -> bool:
        return any(
            self.slot_state(slot.slot_id)["contradiction"]
            for slot in self.program.claim_slots
        )

    def temporal_gap(self) -> bool:
        return any(
            (
                self.slot_state(slot.slot_id)["future_count"] > 0
                or self.slot_state(slot.slot_id)["expired_count"] > 0
            )
            and self.slot_state(slot.slot_id)["covered_atom_count"] == 0
            for slot in self.program.claim_slots
        )

    def atoms(self) -> tuple[EvidenceAtom, ...]:
        return tuple(
            sorted(
                {
                    atom.atom_id: atom
                    for rows in self._atoms.values()
                    for atom in rows.values()
                }.values(),
                key=lambda atom: atom.atom_id,
            )
        )

    def safe_snapshot(self) -> dict[str, Any]:
        slots = {
            slot.slot_id: self.slot_state(slot.slot_id)
            for slot in self.program.claim_slots
        }
        identity = canonical_json_sha256_strict(
            {
                "program_id": self.program.program_id,
                "slots": slots,
                "atom_ids": [atom.atom_id for atom in self.atoms()],
            }
        )
        return {"ledger_sha256": identity, "slots": slots}


def evidence_relation_semantics_proof() -> dict[str, Any]:
    """Return the sealed contradiction and supersession expectation."""

    query = "synthetic relation proof"
    program = compile_retrieval_program(
        query,
        {
            "as_of": "2026-08-11T09:30:00+09:00",
            "claim_slots": [{"slot_id": "answer", "claim": query}],
            "required_evidence": [
                {
                    "claim_slot": "answer",
                    "minimum_atoms": 1,
                    "relations": ["supports", "contradicts"],
                    "must_match_as_of": True,
                }
            ],
            "allowed_actions": ["raw_search"],
            "stop_rules": [
                "coverage",
                "contradiction_resolved",
                "as_of_satisfied",
                "abstain_on_gap",
            ],
        },
    )

    def atom(index: int, relations: tuple[EvidenceRelation, ...]) -> EvidenceAtom:
        return build_evidence_atom(
            episode_id="episode:synthetic-relation-proof",
            claim=query,
            entities=(),
            provenance=Provenance(
                "synthetic-acceptance-proof",
                f"event:{index}",
                "assistant",
                index,
            ),
            evidence=EvidenceRef(
                f"synthetic-relation-{index}.md",
                index,
                index + 1,
                f"{index + 1:064x}",
                f"{index + 11:064x}",
            ),
            validity=TimeInterval(
                "2026-08-11T09:00:00+09:00", "2026-08-11T09:00:00+09:00"
            ),
            relations=relations,
        )

    old_claim = "claim:synthetic-old"
    support = atom(
        0, (EvidenceRelation(EvidenceRelationKind.SUPPORTS, old_claim),)
    )
    contradict = atom(
        1, (EvidenceRelation(EvidenceRelationKind.CONTRADICTS, old_claim),)
    )
    replacement = atom(
        2,
        (
            EvidenceRelation(EvidenceRelationKind.SUPPORTS, "claim:synthetic-new"),
            EvidenceRelation(EvidenceRelationKind.SUPERSEDES, old_claim),
        ),
    )
    ledger = EvidenceLedger(program)
    ledger.add("answer", support)
    ledger.add("answer", contradict)
    contradictory = ledger.slot_state("answer")
    ledger.add("answer", replacement)
    resolved = ledger.slot_state("answer")
    if not (
        contradictory["contradiction"] is True
        and contradictory["covered"] is False
        and resolved["contradiction"] is False
        and resolved["covered"] is True
        and resolved["superseded_count"] == 1
    ):
        raise EvidenceReconstructionError("relation semantics proof failed")
    proof = {
        "conflicting_claim_id": old_claim,
        "support_atom_id": support.atom_id,
        "contradict_atom_id": contradict.atom_id,
        "superseding_atom_id": replacement.atom_id,
        "before": contradictory,
        "after": resolved,
    }
    return {"proof_sha256": canonical_json_sha256_strict(proof), **proof}


def evidence_relation_semantics_sha256() -> str:
    return str(evidence_relation_semantics_proof()["proof_sha256"])


@dataclass(frozen=True)
class EvidenceRun:
    packet: EvidencePacket
    stop_reason: str
    trace: Mapping[str, Any]
    telemetry: Mapping[str, Any]


def _tokens(value: str) -> frozenset[str]:
    return frozenset(token.casefold() for token in _TOKEN_RE.findall(value))


def _atom_matches_slot(
    *,
    claim: str,
    requirement: RequiredEvidence,
    atom: EvidenceAtom,
    authority_roles: frozenset[str],
) -> bool:
    wanted = _tokens(claim)
    return bool(
        wanted
        and atom.provenance.source_role in authority_roles
        and wanted.issubset(_tokens(atom.claim))
        and any(relation.kind in requirement.relations for relation in atom.relations)
    )


def _projection_candidates(
    program: RetrievalProgram,
    projection: EpisodeProjection,
    ledger: EvidenceLedger,
    deadline: float,
) -> bool:
    for slot in program.claim_slots:
        # ponytail: fixed L2 scan cap; raise only after measured recall/latency need.
        for index, atom in enumerate(projection.atoms[:_L2_PROJECTION_SCAN_LIMIT]):
            if index % 128 == 0 and time.monotonic() >= deadline:
                return False
            ledger.add(slot.slot_id, atom)
    return True


def _raw_search_atoms(
    payload: Mapping[str, Any], projection: EpisodeProjection, raw_dir: Path
) -> tuple[EvidenceAtom, ...]:
    hits = payload.get("hits")
    if not isinstance(hits, list):
        raise EvidenceReconstructionError("RAW_SEARCH hits are invalid")
    store = RawStore(raw_dir, mode="v2")
    selected: dict[str, EvidenceAtom] = {}
    atoms_by_raw: dict[str, list[EvidenceAtom]] = {}
    for atom in projection.atoms:
        atoms_by_raw.setdefault(atom.evidence.raw_id, []).append(atom)
    for hit in hits:
        if not isinstance(hit, Mapping) or set(hit) != {
            "raw_id",
            "captured_date",
            "excerpt",
            "offset",
            "citation",
        }:
            raise EvidenceReconstructionError("RAW_SEARCH hit is invalid")
        raw_id = hit["raw_id"]
        offset = hit["offset"]
        excerpt = hit["excerpt"]
        if (
            not isinstance(raw_id, str)
            or isinstance(offset, bool)
            or not isinstance(offset, int)
            or offset < 0
            or not isinstance(excerpt, str)
        ):
            raise EvidenceReconstructionError("RAW_SEARCH hit fields are invalid")
        unit = store.resolve_segment(raw_id)
        if unit is None:
            raise EvidenceReconstructionError("RAW_SEARCH hit has no committed Raw")
        raw = store.read_bytes(unit)
        try:
            text = raw.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise EvidenceReconstructionError("RAW_SEARCH Raw is not UTF-8") from exc
        if offset > len(text) or text[offset : offset + len(excerpt)] != excerpt:
            raise EvidenceReconstructionError("RAW_SEARCH excerpt is not exact")
        byte_start = len(text[:offset].encode("utf-8"))
        byte_end = byte_start + len(excerpt.encode("utf-8"))
        for atom in atoms_by_raw.get(raw_id, []):
            if (
                atom.evidence.byte_start < byte_end
                and byte_start < atom.evidence.byte_end
            ):
                verify_projection_atom(raw_dir, atom)
                selected[atom.atom_id] = atom
    return tuple(sorted(selected.values(), key=lambda atom: atom.atom_id))


def run_evidence_retrieval(
    program: RetrievalProgram,
    projection: EpisodeProjection,
    *,
    tool_context: ToolContext | None = None,
    actions: Sequence[tuple[str, Action]] = (),
    raw_dir: Path | None = None,
    deadline_ms: int = 4_000,
) -> EvidenceRun:
    """Run projection first, then local tools only for remaining claim slots."""

    if not isinstance(program, RetrievalProgram) or not isinstance(
        projection, EpisodeProjection
    ):
        raise EvidenceReconstructionError("runtime contract identity is invalid")
    rebuilt_program = compile_retrieval_program(
        program.query,
        {
            "as_of": program.as_of,
            "claim_slots": [row.to_dict() for row in program.claim_slots],
            "required_evidence": [row.to_dict() for row in program.required_evidence],
            "allowed_actions": [row.value for row in program.allowed_actions],
            "stop_rules": [row.value for row in program.stop_rules],
        },
    )
    if rebuilt_program != program:
        raise EvidenceReconstructionError("runtime contract identity is invalid")
    if (
        isinstance(deadline_ms, bool)
        or not isinstance(deadline_ms, int)
        or deadline_ms < 0
    ):
        raise EvidenceReconstructionError("deadline_ms must be non-negative")
    slot_ids = {slot.slot_id for slot in program.claim_slots}
    for slot_id, action in actions:
        if (
            slot_id not in slot_ids
            or not isinstance(action, Action)
            or action.type not in program.allowed_actions
            or action.type not in _LOCAL_ACTIONS
        ):
            raise EvidenceReconstructionError("evidence action is not allowed")
    started = time.monotonic()
    deadline = started + (deadline_ms / 1000)
    ledger = EvidenceLedger(program, projection.evidence_authority_roles)
    trace_actions: list[dict[str, Any]] = []
    projection_completed = _projection_candidates(program, projection, ledger, deadline)
    tool_errors = 0
    executed = 0
    if projection_completed and not ledger.covered():
        for slot_id, action in actions:
            if time.monotonic() >= deadline:
                projection_completed = False
                break
            if slot_id not in ledger.gaps():
                continue
            if tool_context is None:
                break
            executed += 1
            action_sha = canonical_json_sha256_strict(action.to_dict())
            try:
                payload = execute_tool(action, tool_context)
                if not isinstance(payload, Mapping):
                    raise EvidenceReconstructionError("tool observation is invalid")
                observation_sha = canonical_json_sha256_strict(payload)
                if action.type == ActionType.RAW_SEARCH and raw_dir is not None:
                    for atom in _raw_search_atoms(payload, projection, raw_dir):
                        ledger.add(slot_id, atom)
                status = "ok"
            except Exception:
                tool_errors += 1
                observation_sha = canonical_json_sha256_strict(
                    {"status": "error", "action_sha256": action_sha}
                )
                status = "error"
            trace_actions.append(
                {
                    "slot_id": slot_id,
                    "action": action.type.value,
                    "action_sha256": action_sha,
                    "observation_sha256": observation_sha,
                    "status": status,
                    "ledger_sha256": ledger.safe_snapshot()["ledger_sha256"],
                }
            )
            if ledger.covered():
                break
    if ledger.covered():
        stop_reason = "coverage"
        abstention_reason = ""
    elif not projection_completed or time.monotonic() >= deadline:
        stop_reason = abstention_reason = "deadline_exceeded"
    elif ledger.unresolved_contradiction():
        stop_reason = abstention_reason = "unresolved_contradiction"
    elif ledger.temporal_gap():
        stop_reason = abstention_reason = "as_of_unsatisfied"
    elif actions and executed:
        stop_reason = abstention_reason = "action_exhausted"
    else:
        stop_reason = abstention_reason = "missing_required_evidence"
    packet = build_evidence_packet(
        query=program.query,
        as_of=program.as_of,
        retrieval_program_id=program.program_id,
        atoms=ledger.atoms(),
        abstention_reason=abstention_reason,
    )
    latency_ms = max(0, int((time.monotonic() - started) * 1000))
    ledger_snapshot = ledger.safe_snapshot()
    trace = {
        "program": program.to_dict(),
        "projection_sha256": projection.projection_id.removeprefix("projection:"),
        "actions": trace_actions,
        "ledger": ledger_snapshot,
        "packet_sha256": packet.packet_id.removeprefix("packet:"),
        "stop_reason": stop_reason,
    }
    telemetry = {
        "program_sha256": program.program_id.removeprefix("program:"),
        "projection_sha256": projection.projection_id.removeprefix("projection:"),
        "packet_sha256": packet.packet_id.removeprefix("packet:"),
        "ledger_sha256": ledger_snapshot["ledger_sha256"],
        "stop_reason": stop_reason,
        "abstained": packet.abstained,
        "atom_count": len(packet.atoms),
        "action_count": executed,
        "tool_error_count": tool_errors,
        "latency_ms": latency_ms,
        "cloud_call_count": 0,
        "external_backend_call_count": 0,
        "external_model_call_count": 0,
        "raw_relation_mode": "supports-only",
    }
    return EvidenceRun(packet, stop_reason, trace, telemetry)


def compile_bounded_evidence_context(
    packet: EvidencePacket, *, max_chars: int
) -> str | None:
    """Return the complete packet or nothing; never inject partial evidence."""

    if isinstance(max_chars, bool) or not isinstance(max_chars, int) or max_chars < 1:
        raise EvidenceReconstructionError("max_chars must be positive")
    context = packet.canonical_bytes().decode("utf-8").rstrip("\n")
    return None if packet.abstained or len(context) > max_chars else context


def _okf_finalized(root: Path) -> bool:
    decision = okf_startup_status(root)
    return bool(
        decision.allowed
        and decision.layout == "okf_v0_2"
        and decision.state == "finalized-v2"
    )


def load_evidence_acceptance(root: Path) -> dict[str, Any]:
    payload = read_evidence_authority(evidence_acceptance_path(root))
    if (
        set(payload) != _ACCEPTANCE_FIELDS
        or payload.get("schema") != EVIDENCE_ACCEPTANCE_SCHEMA
        or payload.get("contract_sha256") != EVALUATION_CONTRACT_SHA256
        or payload.get("raw_relation_mode") != "supports-only"
        or any(
            re.fullmatch(r"[0-9a-f]{64}", str(payload.get(field) or "")) is None
            for field in (
                "projection_sha256",
                "raw_watermark_sha256",
                "raw_before_sha256",
                "raw_after_sha256",
                "raw_stat_sha256",
                "evaluation_sha256",
                "case_manifest_sha256",
                "atomic_publication_fault_sha256",
                "relation_semantics_sha256",
                "seal_sha256",
            )
        )
        or payload.get("raw_before_sha256") != payload.get("raw_after_sha256")
        or payload.get("relation_semantics_sha256")
        != evidence_relation_semantics_sha256()
        or payload.get("case_count") != len(EVALUATION_CONTRACT.paired_slices)
        or not isinstance(payload.get("gates"), Mapping)
        or set(payload["gates"]) != evidence_rollout_gate_keys()
        or any(not isinstance(value, bool) for value in payload["gates"].values())
    ):
        raise EvidenceReconstructionError("evidence acceptance receipt is invalid")
    return payload


def _published_evidence_payload(rendered: str) -> dict[str, Any] | None:
    from chronovisor.recall.recall_publication import _render_recall_payload

    opening = "[RECALL_CONTEXT]"
    closing = "[/RECALL_CONTEXT]"
    if rendered.count(opening) != 1 or rendered.count(closing) != 1:
        return None
    start = rendered.index(opening)
    end = rendered.index(closing, start) + len(closing)
    block = rendered[start:end]
    marker = "payload_json=\n"
    try:
        encoded = block.split(marker, 1)[1].rsplit(f"\n{closing}", 1)[0]
        payload = json.loads(encoded)
    except (IndexError, TypeError, json.JSONDecodeError):
        return None
    if (
        not isinstance(payload, dict)
        or payload.get("authority") != "evidence_reconstruction"
        or _render_recall_payload(dict(payload), len(block)) != block
    ):
        return None
    return payload


def _verified_applied_session(
    row: Mapping[str, Any], projection: EpisodeProjection, projection_sha256: str
) -> str | None:
    evidence_features = row.get("evidence_features")
    evidence = (
        evidence_features.get("evidence_reconstruction")
        if isinstance(evidence_features, Mapping)
        else None
    )
    receipt = row.get("context_receipt")
    host = row.get("host")
    session_id = row.get("session_id")
    if (
        row.get("schema_version") != 2
        or row.get("stage") != "injected"
        or not isinstance(evidence, Mapping)
        or evidence.get("status") != "active"
        or evidence.get("authority") != "evidence_reconstruction"
        or evidence.get("abstained") is not False
        or not isinstance(host, str)
        or not host
        or not isinstance(session_id, str)
        or not session_id
        or not isinstance(receipt, Mapping)
        or set(receipt)
        != {
            "schema_version",
            "renderer_protocol",
            "context_style",
            "rendered_context",
            "rendered_context_sha256",
            "page_bindings",
            "receipt_sha256",
        }
    ):
        return None
    rendered = receipt.get("rendered_context")
    unsigned_receipt = {
        key: value for key, value in receipt.items() if key != "receipt_sha256"
    }
    if (
        receipt.get("schema_version") != 1
        or receipt.get("renderer_protocol") != "recall-result-context-v1"
        or receipt.get("context_style") != row.get("context_style")
        or not isinstance(rendered, str)
        or receipt.get("rendered_context_sha256")
        != hashlib.sha256(rendered.encode("utf-8")).hexdigest()
        or receipt.get("receipt_sha256")
        != canonical_json_sha256_strict(unsigned_receipt)
        or receipt.get("page_bindings") != []
        or row.get("pages") != []
        or row.get("context_items") != []
    ):
        return None
    payload = _published_evidence_payload(rendered)
    if payload is None or payload.get("decision") != row.get("decision"):
        return None
    packet = payload.get("evidence_packet")
    outer_trace = payload.get("trace")
    if (
        not isinstance(packet, Mapping)
        or set(packet)
        != {
            "schema",
            "packet_id",
            "query",
            "as_of",
            "retrieval_program_id",
            "atoms",
            "abstained",
            "abstention_reason",
        }
        or packet.get("schema") != EVIDENCE_PACKET_SCHEMA
        or packet.get("abstained") is not False
        or packet.get("abstention_reason") != ""
        or not isinstance(packet.get("atoms"), list)
        or not packet["atoms"]
        or not isinstance(outer_trace, Mapping)
        or set(outer_trace)
        != {
            "decision_id",
            "session_id",
            "evidence_trace",
            "evidence_trace_sha256",
        }
        or outer_trace.get("decision_id") != row.get("decision_id")
        or outer_trace.get("session_id") != session_id
    ):
        return None
    trace = outer_trace.get("evidence_trace")
    trace_sha = outer_trace.get("evidence_trace_sha256")
    if (
        not isinstance(trace, Mapping)
        or set(trace)
        != {
            "program",
            "projection_sha256",
            "actions",
            "ledger",
            "packet_sha256",
            "stop_reason",
        }
        or trace_sha != canonical_json_sha256_strict(trace)
        or evidence.get("trace") != trace
        or evidence.get("trace_sha256") != trace_sha
        or trace.get("projection_sha256") != projection_sha256
        or evidence.get("projection_sha256") != projection_sha256
        or trace.get("stop_reason") != "coverage"
        or trace.get("actions") != []
    ):
        return None
    program_row = trace.get("program")
    if not isinstance(program_row, Mapping):
        return None
    try:
        program = compile_retrieval_program(
            str(program_row.get("query") or ""),
            {
                "as_of": program_row.get("as_of"),
                "claim_slots": program_row.get("claim_slots"),
                "required_evidence": program_row.get("required_evidence"),
                "allowed_actions": program_row.get("allowed_actions"),
                "stop_rules": program_row.get("stop_rules"),
            },
        )
    except (TypeError, EvidenceReconstructionError):
        return None
    if program.to_dict() != program_row:
        return None
    atoms_by_id = {atom.atom_id: atom for atom in projection.atoms}
    packet_atoms: list[EvidenceAtom] = []
    for atom_row in packet["atoms"]:
        if not isinstance(atom_row, Mapping):
            return None
        atom = atoms_by_id.get(str(atom_row.get("atom_id") or ""))
        if atom is None or atom.to_dict() != atom_row:
            return None
        packet_atoms.append(atom)
    if [atom.atom_id for atom in packet_atoms] != sorted(
        {atom.atom_id for atom in packet_atoms}
    ):
        return None
    for atom in packet_atoms:
        if not any(
            _atom_matches_slot(
                claim=slot.claim,
                requirement=next(
                    item
                    for item in program.required_evidence
                    if item.claim_slot == slot.slot_id
                ),
                atom=atom,
                authority_roles=frozenset(projection.evidence_authority_roles),
            )
            for slot in program.claim_slots
        ):
            return None
    unsigned_packet = {
        "query": packet.get("query"),
        "as_of": packet.get("as_of"),
        "retrieval_program_id": packet.get("retrieval_program_id"),
        "atoms": packet["atoms"],
        "abstained": False,
        "abstention_reason": "",
    }
    expected_packet_id = "packet:" + hashlib.sha256(
        canonical_json_line_bytes_strict(unsigned_packet)
    ).hexdigest()
    packet_sha = expected_packet_id.removeprefix("packet:")
    ledger = trace.get("ledger")
    rebuilt_ledger = EvidenceLedger(program, projection.evidence_authority_roles)
    for slot in program.claim_slots:
        for atom in packet_atoms:
            rebuilt_ledger.add(slot.slot_id, atom)
    if (
        packet.get("packet_id") != expected_packet_id
        or packet.get("retrieval_program_id") != program.program_id
        or packet.get("query") != program.query
        or packet.get("as_of") != program.as_of
        or trace.get("packet_sha256") != packet_sha
        or evidence.get("packet_sha256") != packet_sha
        or evidence.get("program_sha256")
        != program.program_id.removeprefix("program:")
        or not isinstance(ledger, Mapping)
        or ledger != rebuilt_ledger.safe_snapshot()
        or not rebuilt_ledger.covered()
    ):
        return None
    return hashlib.sha256(f"{host}\0{session_id}".encode()).hexdigest()


def evidence_applied_session_count(root: Path, projection_sha256: str) -> int:
    """Count only exact, actually published bounded evidence receipts."""

    if re.fullmatch(r"[0-9a-f]{64}", projection_sha256) is None:
        return 0
    try:
        projection = load_episode_projection(evidence_projection_path(root))
        if projection.projection_id.removeprefix("projection:") != projection_sha256:
            return 0
        with open_regular_nofollow(root / "recall" / "recall-log.jsonl") as stream:
            lines = stream.read().decode("utf-8").splitlines()
    except Exception:
        return 0
    sessions: set[str] = set()
    for line in lines:
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            return 0
        if not isinstance(row, Mapping):
            return 0
        evidence_features = row.get("evidence_features")
        evidence = (
            evidence_features.get("evidence_reconstruction")
            if isinstance(evidence_features, Mapping)
            else None
        )
        if not isinstance(evidence, Mapping) or evidence.get("status") != "active":
            continue
        session = _verified_applied_session(row, projection, projection_sha256)
        if session is None:
            return 0
        sessions.add(session)
    return len(sessions)


def load_evidence_rollout(root: Path) -> dict[str, Any]:
    try:
        payload = read_evidence_authority(_evidence_promotion_path(root))
    except Exception:
        return {"mode": "shadow", "canary_percent": 0, "reason": "no_valid_promotion"}
    percent = payload.get("canary_percent")
    mode = payload.get("mode")
    gates = payload.get("gates")
    sample_count = payload.get("sample_count")
    stage_started = payload.get("stage_started_sample_count")
    if (
        set(payload) != _PROMOTION_FIELDS
        or payload.get("schema_version") != 1
        or (mode, percent)
        not in {
            ("shadow", 0),
            ("candidate", 5),
            ("candidate", 25),
            ("active", 100),
        }
        or isinstance(percent, bool)
        or not isinstance(percent, int)
        or payload.get("sample_unit") != CANARY_SAMPLE_UNIT
        or payload.get("rollback_teacher") != "current"
        or not isinstance(payload.get("rollback_reason"), str)
        or not isinstance(payload.get("generated_at"), str)
        or not payload.get("generated_at")
        or isinstance(sample_count, bool)
        or not isinstance(sample_count, int)
        or sample_count < 0
        or isinstance(stage_started, bool)
        or not isinstance(stage_started, int)
        or not 0 <= stage_started <= sample_count
        or any(
            re.fullmatch(r"[0-9a-f]{64}", str(payload.get(field) or "")) is None
            for field in (
                "manifest_sha256",
                "relation_snapshot_sha256",
                "rubric_sha256",
                "model_manifest_sha256",
                "seal_sha256",
            )
        )
        or not isinstance(gates, Mapping)
        or set(gates) != evidence_rollout_gate_keys()
        or any(not isinstance(value, bool) for value in gates.values())
        or (mode != "shadow" and not all(gates.values()))
    ):
        return {"mode": "shadow", "canary_percent": 0, "reason": "invalid_promotion"}
    return payload


def begin_evidence_refresh_at(directory_fd: int) -> dict[str, Any]:
    return write_evidence_authority_at(
        directory_fd,
        "promotion.json",
        {
            "schema_version": 1,
            "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
            "mode": "shadow",
            "canary_percent": 0,
            "stage_started_sample_count": 0,
            "sample_count": 0,
            "sample_unit": CANARY_SAMPLE_UNIT,
            "gates": {key: False for key in sorted(evidence_rollout_gate_keys())},
            "reason": "acceptance_refresh",
            "rollback_reason": "acceptance_refresh",
            "rollback_teacher": "current",
            "manifest_sha256": "0" * 64,
            "relation_snapshot_sha256": "0" * 64,
            "rubric_sha256": EVALUATION_CONTRACT_SHA256,
            "model_manifest_sha256": "0" * 64,
        },
    )


def begin_evidence_refresh(root: Path) -> dict[str, Any]:
    """Atomically force shadow before replacing projection or acceptance."""

    if not _okf_finalized(root):
        raise EvidenceReconstructionError("Campaign X is not finalized")
    with okf_runtime_operation(root) as locked:
        if not (
            locked.allowed
            and locked.layout == "okf_v0_2"
            and locked.state == "finalized-v2"
        ):
            raise EvidenceReconstructionError("Campaign X changed during refresh")
        with evidence_authority_operation(root) as directory_fd:
            return begin_evidence_refresh_at(directory_fd)


def _rollout_transition(
    *,
    previous: Mapping[str, Any],
    gates: Mapping[str, bool],
    sample_count: int,
    minimum_step_samples: int,
) -> tuple[str, int, int, str]:
    prior_percent = int(previous.get("canary_percent") or 0)
    prior_started = int(previous.get("stage_started_sample_count") or 0)
    if not gates or not all(value is True for value in gates.values()):
        return "shadow", 0, sample_count, "gate_failed"
    if prior_percent == 0:
        return "candidate", 5, sample_count, "sealed_gate_passed"
    if sample_count - prior_started >= minimum_step_samples:
        percent = next((step for step in CANARY_STEPS if step > prior_percent), 100)
        return (
            "active" if percent == 100 else "candidate",
            percent,
            sample_count,
            "canary_advanced",
        )
    return (
        "active" if prior_percent == 100 else "candidate",
        prior_percent,
        prior_started,
        "collecting_canary_samples",
    )


def _write_evidence_rollout(
    *,
    directory_fd: int,
    previous: Mapping[str, Any],
    gates: Mapping[str, bool],
    sample_count: int,
    minimum_step_samples: int,
    manifest_sha256: str,
    relation_snapshot_sha256: str,
    rubric_sha256: str,
    model_manifest_sha256: str,
    rollback_reason: str = "",
) -> dict[str, Any]:
    mode, percent, started, reason = _rollout_transition(
        previous=previous,
        gates=gates,
        sample_count=sample_count,
        minimum_step_samples=minimum_step_samples,
    )
    return write_evidence_authority_at(
        directory_fd,
        "promotion.json",
        {
            "schema_version": 1,
            "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
            "mode": mode,
            "canary_percent": percent,
            "stage_started_sample_count": started,
            "sample_count": sample_count,
            "sample_unit": CANARY_SAMPLE_UNIT,
            "gates": dict(sorted(gates.items())),
            "reason": reason,
            "rollback_reason": rollback_reason[:160],
            "rollback_teacher": "current",
            "manifest_sha256": manifest_sha256,
            "relation_snapshot_sha256": relation_snapshot_sha256,
            "rubric_sha256": rubric_sha256,
            "model_manifest_sha256": model_manifest_sha256,
        },
    )


def evidence_selected(root: Path, session_id: str, projection_id: str) -> bool:
    if (
        not isinstance(projection_id, str)
        or re.fullmatch(r"projection:[0-9a-f]{64}", projection_id) is None
    ):
        return False
    try:
        projection = load_episode_projection(evidence_projection_path(root))
        acceptance = load_evidence_acceptance(root)
        cases = read_evidence_authority(evidence_cases_path(root))
        raw_watermark = committed_raw_watermark(root / "raw")
        raw_stat = raw_stat_watermark(root / "raw")
    except Exception:
        return False
    if projection.projection_id != projection_id:
        return False
    rollout = load_evidence_rollout(root)
    gates = rollout.get("gates")
    projection_sha = projection_id.removeprefix("projection:")
    bound = bool(
        rollout.get("manifest_sha256") == projection_sha
        and rollout.get("relation_snapshot_sha256") == raw_watermark
        and rollout.get("rubric_sha256") == EVALUATION_CONTRACT_SHA256
        and rollout.get("model_manifest_sha256") == acceptance["seal_sha256"]
        and acceptance["projection_sha256"] == projection_sha
        and acceptance["raw_watermark_sha256"] == raw_watermark
        and acceptance["raw_stat_sha256"] == raw_stat
        and set(cases)
        == {
            "schema",
            "contract_sha256",
            "case_manifest_sha256",
            "cases",
            "seal_sha256",
        }
        and cases.get("schema") == "chronovisor.evidence-cases.v1"
        and cases.get("contract_sha256") == EVALUATION_CONTRACT_SHA256
        and cases.get("case_manifest_sha256") == acceptance["case_manifest_sha256"]
        and isinstance(gates, Mapping)
        and set(gates) == evidence_rollout_gate_keys()
        and all(value is True for value in gates.values())
    )
    return (
        bound
        and _okf_finalized(root)
        and selected_for_canary(session_id, int(rollout["canary_percent"]))
    )


def advance_evidence_rollout_at(
    *,
    root: Path,
    directory_fd: int,
    minimum_step_samples: int = 100,
) -> dict[str, Any]:
    if (
        isinstance(minimum_step_samples, bool)
        or not isinstance(minimum_step_samples, int)
        or minimum_step_samples < 1
    ):
        raise EvidenceReconstructionError("rollout gates are invalid")
    receipt = load_evidence_acceptance(root)
    cases = read_evidence_authority(evidence_cases_path(root))
    projection = load_episode_projection(evidence_projection_path(root))
    if (
        projection.projection_id.removeprefix("projection:")
        != receipt["projection_sha256"]
        or committed_raw_watermark(root / "raw") != receipt["raw_watermark_sha256"]
        or physical_raw_inventory(root / "raw")["inventory_sha256"]
        != receipt["raw_after_sha256"]
        or raw_stat_watermark(root / "raw") != receipt["raw_stat_sha256"]
        or cases.get("contract_sha256") != EVALUATION_CONTRACT_SHA256
        or cases.get("case_manifest_sha256") != receipt["case_manifest_sha256"]
        or set(cases)
        != {
            "schema",
            "contract_sha256",
            "case_manifest_sha256",
            "cases",
            "seal_sha256",
        }
        or cases.get("schema") != "chronovisor.evidence-cases.v1"
    ):
        raise EvidenceReconstructionError("acceptance receipt is stale")
    previous = load_evidence_rollout(root)
    if (
        set(previous) != _PROMOTION_FIELDS
        or previous.get("rollback_reason") not in {"", "acceptance_refresh"}
    ):
        raise EvidenceReconstructionError("promotion authority is invalid")
    sample_count = evidence_applied_session_count(
        root, str(receipt["projection_sha256"])
    )
    return _write_evidence_rollout(
        directory_fd=directory_fd,
        previous=previous,
        gates=receipt["gates"],
        sample_count=sample_count,
        minimum_step_samples=minimum_step_samples,
        manifest_sha256=str(receipt["projection_sha256"]),
        relation_snapshot_sha256=str(receipt["raw_watermark_sha256"]),
        rubric_sha256=EVALUATION_CONTRACT_SHA256,
        model_manifest_sha256=str(receipt["seal_sha256"]),
    )


def advance_evidence_rollout(
    *,
    root: Path,
    minimum_step_samples: int = 100,
) -> dict[str, Any]:
    if not _okf_finalized(root):
        raise EvidenceReconstructionError("Campaign X is not finalized")
    with okf_runtime_operation(root) as locked:
        if not (
            locked.allowed
            and locked.layout == "okf_v0_2"
            and locked.state == "finalized-v2"
        ):
            raise EvidenceReconstructionError("Campaign X changed during rollout")
        with evidence_authority_operation(root) as directory_fd:
            return advance_evidence_rollout_at(
                root=root,
                directory_fd=directory_fd,
                minimum_step_samples=minimum_step_samples,
            )


def rollback_evidence_rollout_at(
    *, root: Path, directory_fd: int, reason: str
) -> dict[str, Any]:
    if not isinstance(reason, str) or not reason.strip():
        raise EvidenceReconstructionError("rollback reason is invalid")
    previous = load_evidence_rollout(root)
    if set(previous) != _PROMOTION_FIELDS:
        raise EvidenceReconstructionError("promotion authority is invalid")
    return _write_evidence_rollout(
        directory_fd=directory_fd,
        previous=previous,
        gates={key: False for key in evidence_rollout_gate_keys()},
        sample_count=int(previous.get("sample_count") or 0),
        minimum_step_samples=1,
        manifest_sha256=str(previous.get("manifest_sha256") or ""),
        relation_snapshot_sha256=str(previous.get("relation_snapshot_sha256") or ""),
        rubric_sha256=str(previous.get("rubric_sha256") or ""),
        model_manifest_sha256=str(previous.get("model_manifest_sha256") or ""),
        rollback_reason=f"manual:{reason.strip()}",
    )


def rollback_evidence_rollout(*, root: Path, reason: str) -> dict[str, Any]:
    with okf_runtime_operation(root):
        with evidence_authority_operation(root) as directory_fd:
            return rollback_evidence_rollout_at(
                root=root, directory_fd=directory_fd, reason=reason
            )
