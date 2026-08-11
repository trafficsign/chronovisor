from __future__ import annotations

import hashlib
import json
from dataclasses import FrozenInstanceError
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import pytest

from chronovisor.core.canonical_json import canonical_json_sha256_strict
from chronovisor.core.raw_segment import append_capture, seal_segment
from chronovisor.research.evidence_reconstruction import (
    EVALUATION_CONTRACT,
    EvidenceReconstructionError,
    EvidenceRef,
    EvidenceRelation,
    EvidenceRelationKind,
    Provenance,
    TimeInterval,
    build_episode_projection,
    build_evidence_atom,
    build_evidence_packet,
    compile_retrieval_program,
    evaluation_contract_bytes,
    load_episode_projection,
)

NOW = datetime(2026, 8, 11, 9, 30, tzinfo=ZoneInfo("Asia/Tokyo"))
MISSING = object()


def test_y1_evaluation_contract_is_preregistered_and_immutable() -> None:
    payload = json.loads(evaluation_contract_bytes())

    assert {row["metric"] for row in payload["metrics"]} == {
        "answer",
        "evidence",
        "temporal",
        "obsolete-use",
        "latency",
    }
    assert all("lower_bound" in row for row in payload["metrics"])
    assert set(payload["paired_slices"]) == {
        "current",
        "why",
        "change",
        "failure",
        "workflow",
        "contradiction",
        "no-answer",
    }
    assert payload["abstention_conditions"]
    with pytest.raises(FrozenInstanceError):
        EVALUATION_CONTRACT.contract_id = "changed"  # type: ignore[misc]


def test_y2_packet_has_strict_exact_typed_evidence() -> None:
    evidence = EvidenceRef(
        raw_id="save-codex-a.md",
        byte_start=12,
        byte_end=34,
        raw_sha256="a" * 64,
        receipt_sha256="b" * 64,
    )
    atom = build_evidence_atom(
        episode_id="episode:one",
        claim="The current setting was changed.",
        entities=("chronovisor",),
        provenance=Provenance("committed-raw-receipt", "raw:1", "assistant", 1),
        evidence=evidence,
        validity=TimeInterval("2026-08-11T09:00:00+09:00", "2026-08-11T09:01:00+09:00"),
        relations=(
            EvidenceRelation(EvidenceRelationKind.SUPPORTS, "claim:current"),
            EvidenceRelation(EvidenceRelationKind.CONTRADICTS, "claim:old"),
            EvidenceRelation(EvidenceRelationKind.SUPERSEDES, "claim:prior"),
        ),
    )
    packet = build_evidence_packet(
        query="What changed?",
        as_of="2026-08-11T10:00:00+09:00",
        retrieval_program_id="program:" + "c" * 64,
        atoms=(atom,),
    )
    payload = json.loads(packet.canonical_bytes())

    assert payload["schema"] == "chronovisor.evidence-packet.v1"
    assert payload["atoms"][0]["evidence"]["byte_range"] == [12, 34]
    assert payload["atoms"][0]["evidence"]["byte_coordinate_space"] == "logical_raw"
    assert "raw_path" not in payload["atoms"][0]["evidence"]
    assert {row["kind"] for row in payload["atoms"][0]["relations"]} == {
        "supports",
        "contradicts",
        "supersedes",
    }
    with pytest.raises(EvidenceReconstructionError, match="SHA-256"):
        EvidenceRef("a", 0, 1, "bad", "b" * 64)


def _committed_raw(
    raw_dir: Path,
    *,
    first_timestamp: object = "2026-08-11T09:00:00+09:00",
    first_text: str = "Why did it fail?",
) -> tuple[Path, bytes, dict[str, object]]:
    raw_dir.parent.mkdir(parents=True, exist_ok=True)
    for name in ("index.md", "log.md", "schema.md"):
        (raw_dir.parent / name).write_text("legacy\n", encoding="utf-8")
    source = raw_dir.parent / "session.jsonl"
    rows: list[dict[str, object]] = [
        {
            "type": "response_item",
            "payload": {
                "type": "message",
                "role": "user",
                "content": [{"type": "input_text", "text": first_text}],
            },
        },
        {
            "type": "response_item",
            "timestamp": "2026-08-11T09:01:00+09:00",
            "entities": ["chronovisor"],
            "payload": {
                "type": "message",
                "role": "assistant",
                "content": [{"type": "output_text", "text": "The lease expired."}],
            },
        },
    ]
    if first_timestamp is not MISSING:
        rows[0]["timestamp"] = first_timestamp
    raw = b"".join(
        json.dumps(row, ensure_ascii=False, separators=(",", ":")).encode() + b"\n"
        for row in rows
    )
    source.write_bytes(raw)
    receipt = append_capture(
        raw_dir=raw_dir,
        raw_id="save-codex-reconstruction.md",
        idempotency_key="codex-reconstruction",
        host="codex",
        session_key="a" * 24,
        session_id="session-1",
        source_file=source,
        after_line=0,
        until_line=2,
        source_bytes=raw,
        record_count=2,
        now=NOW,
    )
    receipt.data_path.write_bytes(
        receipt.data_path.read_bytes() + b'{"uncommitted":true}\n'
    )
    return receipt.data_path, raw, receipt.commit.to_dict()


def test_y3_projection_uses_committed_receipts_and_rebuilds_identically(
    tmp_path: Path,
) -> None:
    raw_dir = tmp_path / "raw"
    data_path, committed_raw, commit = _committed_raw(raw_dir)
    before = {path: path.read_bytes() for path in raw_dir.rglob("*") if path.is_file()}

    first = build_episode_projection(raw_dir)
    second = build_episode_projection(raw_dir)
    payload = json.loads(first)

    assert first == second
    assert {atom["claim"] for atom in payload["atoms"]} == {
        "Why did it fail?",
        "The lease expired.",
    }
    assert payload["evidence_authority_roles"] == ["assistant"]
    expected_raw_sha256 = hashlib.sha256(committed_raw).hexdigest()
    expected_receipt_sha256 = canonical_json_sha256_strict(commit)
    assert {atom["evidence"]["raw_sha256"] for atom in payload["atoms"]} == {
        expected_raw_sha256
    }
    assert {atom["evidence"]["receipt_sha256"] for atom in payload["atoms"]} == {
        expected_receipt_sha256
    }
    assert payload["source_receipts"][0]["raw_sha256"] == expected_raw_sha256
    assert payload["source_receipts"][0]["receipt_sha256"] == expected_receipt_sha256
    assert {
        path: path.read_bytes() for path in raw_dir.rglob("*") if path.is_file()
    } == before
    assert hashlib.sha256(first).hexdigest() == hashlib.sha256(second).hexdigest()
    assert load_episode_projection(first).canonical_bytes() == first
    seal_segment(data_path, remove_open=True)
    assert build_episode_projection(raw_dir) == first
    projection_dir = tmp_path / "projection"
    projection_dir.mkdir()
    projection_path = projection_dir / "episode.json"
    projection_path.write_bytes(first)
    assert load_episode_projection(projection_path).canonical_bytes() == first
    linked_dir = tmp_path / "linked-projection"
    linked_dir.symlink_to(projection_dir)
    with pytest.raises(EvidenceReconstructionError, match="cannot be read"):
        load_episode_projection(linked_dir / "episode.json")

    tampered = json.loads(first)
    tampered["atoms"][0]["claim"] = "tampered"
    with pytest.raises(EvidenceReconstructionError, match="atom identity mismatch"):
        load_episode_projection(
            json.dumps(
                tampered, ensure_ascii=False, sort_keys=True, separators=(",", ":")
            ).encode()
            + b"\n"
        )

    authority_tampered = json.loads(first)
    authority_tampered["evidence_authority_roles"] = ["user"]
    with pytest.raises(EvidenceReconstructionError, match="evidence authority"):
        load_episode_projection(
            json.dumps(
                authority_tampered,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode()
            + b"\n"
        )


def test_y3_event_timestamp_falls_back_only_when_missing(tmp_path: Path) -> None:
    missing_raw = tmp_path / "missing" / "raw"
    _committed_raw(missing_raw, first_timestamp=MISSING)
    payload = json.loads(build_episode_projection(missing_raw))
    first = next(
        atom for atom in payload["atoms"] if atom["claim"] == "Why did it fail?"
    )
    assert first["validity"] == {"start": NOW.isoformat(), "end": NOW.isoformat()}

    invalid_raw = tmp_path / "invalid" / "raw"
    _committed_raw(invalid_raw, first_timestamp="not-a-timestamp")
    with pytest.raises(EvidenceReconstructionError, match="event timestamp"):
        build_episode_projection(invalid_raw)

    empty_raw = tmp_path / "invalid-empty" / "raw"
    _committed_raw(
        empty_raw,
        first_timestamp="not-a-timestamp",
        first_text="",
    )
    with pytest.raises(EvidenceReconstructionError, match="event timestamp"):
        build_episode_projection(empty_raw)


def _valid_plan() -> dict[str, object]:
    return {
        "as_of": "2026-08-11T10:00:00+09:00",
        "claim_slots": [{"slot_id": "cause", "claim": "What caused the failure?"}],
        "required_evidence": [
            {
                "claim_slot": "cause",
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
    }


def test_y4_compiler_is_strict_complete_and_fail_closed() -> None:
    program = compile_retrieval_program("Why did it fail?", _valid_plan())
    assert program.program_id.startswith("program:")
    assert json.loads(program.canonical_bytes())["claim_slots"][0]["slot_id"] == "cause"

    incomplete = _valid_plan()
    incomplete["stop_rules"] = ["coverage"]
    with pytest.raises(EvidenceReconstructionError, match="incomplete"):
        compile_retrieval_program("Why did it fail?", incomplete)

    invalid = _valid_plan()
    invalid["allowed_actions"] = ["finish"]
    with pytest.raises(EvidenceReconstructionError, match="local evidence actions"):
        compile_retrieval_program("Why did it fail?", invalid)

    cloud = _valid_plan()
    cloud["allowed_actions"] = ["web_search"]
    with pytest.raises(EvidenceReconstructionError, match="local evidence actions"):
        compile_retrieval_program("Why did it fail?", cloud)

    hint_only = _valid_plan()
    hint_only["allowed_actions"] = ["chronovisor_search"]
    with pytest.raises(EvidenceReconstructionError, match="local evidence actions"):
        compile_retrieval_program("Why did it fail?", hint_only)

    unknown = _valid_plan()
    unknown["unexpected"] = True
    with pytest.raises(EvidenceReconstructionError, match="unknown"):
        compile_retrieval_program("Why did it fail?", unknown)
