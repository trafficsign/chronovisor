from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from contextlib import nullcontext
from dataclasses import replace
from datetime import datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import pytest

import chronovisor.core.store as core_store
from chronovisor.core.canonical_json import (
    canonical_json_line_bytes_strict,
    canonical_json_sha256_strict,
)
from chronovisor.core.raw_segment import append_capture
from chronovisor.core.raw_store import RawStore
from chronovisor.ingest.ingest_schemas import TRIAGE_C2_SCHEMA
from chronovisor.ingest.raw_semantic_projection import (
    project_native_transcript,
    read_native_c2_source_records,
)
from chronovisor.research.c2_semantic_projection import (
    C2SemanticProjectionError,
    build_c2_semantic_projection,
    load_c2_semantic_envelope,
    materialize_c2_semantic_envelope,
    run_c2_semantic_retrieval,
    store_c2_semantic_envelope,
)
from chronovisor.research.evidence_reconstruction import (
    EPISODE_PROJECTION_C2_SCHEMA,
    EpisodeProjection,
    EvidenceRef,
    EvidenceRelation,
    EvidenceRelationKind,
    Provenance,
    TimeInterval,
    build_episode_projection_c2,
    build_evidence_atom_c2,
    compile_retrieval_program,
    load_episode_projection_c2,
)
from chronovisor.research.evidence_runtime import compile_bounded_evidence_context
from chronovisor.search.research_store import ResearchStore


def _span(text: str, quote: str) -> dict[str, object]:
    start = text.index(quote)
    byte_start = len(text[:start].encode("utf-8"))
    return {
        "byte_range": [byte_start, byte_start + len(quote.encode("utf-8"))],
        "byte_coordinate_space": "decoded_source_text_utf8",
        "span_sha256": hashlib.sha256(quote.encode("utf-8")).hexdigest(),
    }


def _fixture(
    *,
    unknown_validity: bool = False,
) -> tuple[dict[str, Any], list[dict[str, str]], list[dict[str, object]], EpisodeProjection]:
    validity = (
        None
        if unknown_validity
        else TimeInterval(
            "2026-09-09T00:00:00+09:00", "2026-09-09T01:00:00+09:00"
        )
    )
    text = "決定: feature enabled for alpha if approved."
    source_records = [{"record_id": "save-demo.md#0", "text": text}]
    text_sha256 = hashlib.sha256(text.encode("utf-8")).hexdigest()
    semantic = {
        "record_id": "save-demo.md#0",
        "quote": "feature enabled",
        "kind": "decision",
        "subject_quote": "feature",
        "scope_quotes": ["alpha"],
        "condition_quotes": ["approved"],
        "source_text_sha256": text_sha256,
        "byte_coordinate_space": "decoded_source_text_utf8",
        "quote_span": _span(text, "feature enabled"),
        "subject_span": _span(text, "feature"),
        "scope_spans": [_span(text, "alpha")],
        "condition_spans": [_span(text, "approved")],
    }
    source_records_sha256 = canonical_json_sha256_strict(source_records)
    triage = {
        "operations": [],
        "semantic_evidence": [semantic],
        "audit": {
            "schema_version": "chronovisor.ingest-triage-c2.v1",
            "schema_sha256": canonical_json_sha256_strict(TRIAGE_C2_SCHEMA),
            "prompt_sha256": "f" * 64,
            "request_sha256": "1" * 64,
            "source_records_sha256": source_records_sha256,
            "source_record_count": 1,
            "semantic_evidence_authority": "model_judgment",
            "local_structured": {"ok": True, "attempts": []},
        },
    }
    raw_sha256 = "a" * 64
    receipt_sha256 = "b" * 64
    native_text_sha256 = text_sha256
    binding = {
        "record_id": "save-demo.md#0",
        # Deliberately differs from the projection atom event index.  The
        # native byte range is the join key across decoded/source indices.
        "source_record_index": 7,
        "raw_id": "save-demo.md",
        "raw_sha256": raw_sha256,
        "receipt_sha256": receipt_sha256,
        "source_record_sha256": "d" * 64,
        "source_text_sha256": text_sha256,
        "native_text_sha256": native_text_sha256,
        "range_sha256": "e" * 64,
        "byte_range": [10, 110],
        "byte_coordinate_space": "logical_raw",
        "source_line": 11,
        "source_character_start": 0,
        "source_character_end": len(text),
    }
    base_atom = build_evidence_atom_c2(
        episode_id="episode:demo",
        claim=text,
        entities=(),
        provenance=Provenance(
            "committed-raw-receipt", "save-demo.md:42", "assistant", 42
        ),
        evidence=EvidenceRef(
            "save-demo.md", 10, 110, raw_sha256, receipt_sha256
        ),
        event_time=None,
        recorded_at="2026-09-09T00:00:00+09:00",
        validity=validity,
        event_type="message",
        role="assistant",
        phase="answer",
        relations=(
            EvidenceRelation(EvidenceRelationKind.SUPPORTS, "claim:base"),
        ),
    )
    base_projection = EpisodeProjection(
        projection_id="projection:" + "9" * 64,
        schema=EPISODE_PROJECTION_C2_SCHEMA,
        evidence_authority_roles=("assistant",),
        source_receipts=(
            {
                "raw_id": "save-demo.md",
                "byte_range": [0, 200],
                "byte_coordinate_space": "logical_raw",
                "raw_sha256": raw_sha256,
                "receipt_sha256": receipt_sha256,
                "captured_at": "2026-09-09T00:00:00+09:00",
                "host": "codex",
                "session_key": "s" * 24,
                "source_line_range": [0, 100],
            },
        ),
        atoms=(base_atom,),
    )
    return triage, source_records, [binding], base_projection


def _real_fixture(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    unknown_validity: bool = False,
    long_context: bool = False,
) -> tuple[
    dict[str, Any],
    list[dict[str, str]],
    list[dict[str, Any]],
    EpisodeProjection,
    Path,
    Path,
]:
    """Create one committed Raw event and its opt-in native source map."""

    monkeypatch.setattr(core_store, "okf_runtime_operation", lambda _root: nullcontext())
    raw_dir = tmp_path / "raw"
    source_file = tmp_path / "native-session.jsonl"
    text = (
        "決定: feature enabled for alpha if approved."
        if not long_context
        else "決定: feature enabled " + ("context " * 500) + "for alpha if approved."
    )
    event: dict[str, Any] = {
        "type": "assistant",
        "timestamp": "2026-09-09T00:15:00+09:00",
        "message": {"content": [{"type": "text", "text": text}]},
    }
    if not unknown_validity:
        event["validity"] = {
            "start": "2026-09-09T00:00:00+09:00",
            "end": "2026-09-09T01:00:00+09:00",
        }
    raw_bytes = (
        json.dumps(event, ensure_ascii=False, separators=(",", ":")) + "\n"
    ).encode("utf-8")
    source_file.write_bytes(raw_bytes)
    idempotency_key = "claude-code-" + "a" * 24 + "-from0-to1"
    raw_id = f"save-{idempotency_key}.md"
    receipt = append_capture(
        raw_dir=raw_dir,
        raw_id=raw_id,
        idempotency_key=idempotency_key,
        host="claude-code",
        session_key="a" * 24,
        session_id="session-1",
        source_file=source_file,
        after_line=0,
        until_line=1,
        source_bytes=raw_bytes,
        record_count=1,
        now=datetime(2026, 9, 9, tzinfo=ZoneInfo("Asia/Tokyo")),
    )
    store = RawStore(raw_dir, mode="v2")
    unit = store.resolve(raw_id)
    assert unit is not None and unit.commit is not None
    reference = store.materialize_ingest(
        unit, tmp_path / "runtime" / "raw-projections" / "parents"
    )
    projected = project_native_transcript(
        reference,
        raw_bytes,
        receipt.commit,
        output_dir=tmp_path / "projection",
        max_child_bytes=12_000 if long_context else 4_000,
        include_source_bindings=True,
    )
    assert projected.child_paths
    sources = read_native_c2_source_records(
        projected.child_paths[0], raw_store=store, raw_dir=raw_dir
    )
    records = sources["source_records"]
    assert len(records) == 1
    record_text = records[0]["text"]
    text_sha256 = hashlib.sha256(record_text.encode("utf-8")).hexdigest()
    semantic = {
        "record_id": records[0]["record_id"],
        "quote": "feature enabled",
        "kind": "decision",
        "subject_quote": "feature",
        "scope_quotes": ["alpha"],
        "condition_quotes": ["approved"],
        "source_text_sha256": text_sha256,
        "byte_coordinate_space": "decoded_source_text_utf8",
        "quote_span": _span(record_text, "feature enabled"),
        "subject_span": _span(record_text, "feature"),
        "scope_spans": [_span(record_text, "alpha")],
        "condition_spans": [_span(record_text, "approved")],
    }
    source_records_sha256 = canonical_json_sha256_strict(records)
    triage = {
        "operations": [],
        "semantic_evidence": [semantic],
        "audit": {
            "schema_version": "chronovisor.ingest-triage-c2.v1",
            "schema_sha256": canonical_json_sha256_strict(TRIAGE_C2_SCHEMA),
            "prompt_sha256": "f" * 64,
            "request_sha256": "1" * 64,
            "source_records_sha256": source_records_sha256,
            "source_record_count": 1,
            "semantic_evidence_authority": "model_judgment",
            "local_structured": {"ok": True, "attempts": []},
        },
    }
    base = load_episode_projection_c2(build_episode_projection_c2(raw_dir))
    assert len(base.atoms) == 1
    return triage, records, sources["bindings"], base, raw_dir, projected.child_paths[0]


def test_materialize_and_cas_round_trip_without_raw_event_duplication(tmp_path: Path) -> None:
    triage, records, bindings, _projection = _fixture()
    envelope = materialize_c2_semantic_envelope(
        triage,
        records,
        bindings,
        projection_id="2" * 64,
        source_sha256="3" * 64,
    )
    assert envelope["semantic_evidence_authority"] == "model_judgment"
    assert envelope["triage_result"]["semantic_evidence"][0]["condition_quotes"] == [
        "approved"
    ]
    assert envelope["source_records"][0] == {
        "record_id": records[0]["record_id"],
        "source_text_sha256": hashlib.sha256(records[0]["text"].encode()).hexdigest(),
        "text_bytes": len(records[0]["text"].encode()),
    }
    encoded = canonical_json_line_bytes_strict(envelope)
    assert records[0]["text"] not in json.loads(encoded)["source_records"]

    store = ResearchStore(tmp_path / "research")
    store.durable_cas = tmp_path / "durable-cas"
    store.durable_manifests = tmp_path / "durable-manifests"
    artifact = store_c2_semantic_envelope(
        store, envelope, source_uri="projection:" + "2" * 64
    )
    restored = load_c2_semantic_envelope(store, artifact.artifact_id)
    assert restored == envelope
    assert artifact.sha256 == hashlib.sha256(encoded).hexdigest()


def test_consumer_rebinds_decoded_quote_to_native_range_and_ledger(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    triage, records, bindings, base_projection, raw_dir, _child = _real_fixture(
        tmp_path, monkeypatch
    )
    envelope = materialize_c2_semantic_envelope(
        triage,
        records,
        bindings,
        projection_id="2" * 64,
        source_sha256="3" * 64,
    )
    projection = build_c2_semantic_projection(
        envelope,
        base_projection,
        source_records=records,
        raw_dir=raw_dir,
    )
    assert len(projection.atoms) == 1
    assert load_episode_projection_c2(projection.canonical_bytes()) == projection
    atom = projection.atoms[0]
    assert atom.claim == "決定: feature enabled for alpha if approved."
    assert "if approved" in atom.claim
    assert atom.event_time == "2026-09-09T00:15:00+09:00"
    assert atom.validity is not None
    assert atom.evidence.byte_start == bindings[0]["byte_range"][0]
    assert atom.evidence.byte_end == bindings[0]["byte_range"][1]
    assert all(
        relation.kind == EvidenceRelationKind.SUPPORTS for relation in atom.relations
    )
    program = compile_retrieval_program(
        "feature enabled",
        {
            "as_of": "2026-09-09T00:30:00+09:00",
            "claim_slots": [{"slot_id": "answer", "claim": "feature enabled"}],
            "required_evidence": [
                {
                    "claim_slot": "answer",
                    "minimum_atoms": 1,
                    "relations": ["supports"],
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
    run = run_c2_semantic_retrieval(
        envelope,
        base_projection,
        source_records=records,
        program=program,
        raw_dir=raw_dir,
    )
    assert run.packet.schema == "chronovisor.evidence-packet.v2"
    assert run.packet.abstained is False
    assert run.stop_reason == "coverage"
    assert "if approved" in run.packet.canonical_bytes().decode("utf-8")


def test_unknown_validity_is_held_and_source_tampering_fails_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    triage, records, bindings, base_projection, raw_dir, _child = _real_fixture(
        tmp_path, monkeypatch, unknown_validity=True
    )
    envelope = materialize_c2_semantic_envelope(
        triage,
        records,
        bindings,
        projection_id="2" * 64,
        source_sha256="3" * 64,
    )
    program = compile_retrieval_program(
        "feature enabled",
        {
            "as_of": "2026-09-09T00:30:00+09:00",
            "claim_slots": [{"slot_id": "answer", "claim": "feature enabled"}],
            "required_evidence": [
                {
                    "claim_slot": "answer",
                    "minimum_atoms": 1,
                    "relations": ["supports"],
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
    run = run_c2_semantic_retrieval(
        envelope,
        base_projection,
        source_records=records,
        program=program,
        raw_dir=raw_dir,
    )
    assert run.packet.abstained is True
    assert run.stop_reason == "as_of_unsatisfied"

    tampered = json.loads(json.dumps(envelope, ensure_ascii=False))
    tampered["triage_result"]["semantic_evidence"][0]["quote_span"]["byte_range"][0] += 1
    with pytest.raises(C2SemanticProjectionError, match="span is invalid"):
        build_c2_semantic_projection(
            tampered,
            base_projection,
            source_records=records,
            raw_dir=raw_dir,
        )


def test_forged_decoded_source_with_real_raw_identity_is_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    triage, records, bindings, base_projection, raw_dir, _child = _real_fixture(
        tmp_path, monkeypatch
    )
    forged_text = records[0]["text"].replace("feature", "changed")
    forged_records = [{"record_id": records[0]["record_id"], "text": forged_text}]
    forged_triage = json.loads(json.dumps(triage, ensure_ascii=False))
    semantic = forged_triage["semantic_evidence"][0]
    semantic["quote"] = "changed enabled"
    semantic["subject_quote"] = "changed"
    semantic["scope_quotes"] = ["alpha"]
    semantic["condition_quotes"] = ["approved"]
    semantic["source_text_sha256"] = hashlib.sha256(forged_text.encode()).hexdigest()
    semantic["quote_span"] = _span(forged_text, "changed enabled")
    semantic["subject_span"] = _span(forged_text, "changed")
    semantic["scope_spans"] = [_span(forged_text, "alpha")]
    semantic["condition_spans"] = [_span(forged_text, "approved")]
    forged_triage["audit"]["source_records_sha256"] = canonical_json_sha256_strict(
        forged_records
    )
    forged_binding = dict(bindings[0])
    forged_binding["source_text_sha256"] = hashlib.sha256(
        forged_text.encode()
    ).hexdigest()
    envelope = materialize_c2_semantic_envelope(
        forged_triage,
        forged_records,
        [forged_binding],
        projection_id="2" * 64,
        source_sha256="3" * 64,
    )
    # The Raw/receipt identity is still the real base atom, but the decoded
    # child text is not the text reconstructed from that Raw event.
    with pytest.raises(C2SemanticProjectionError, match="native Raw"):
        build_c2_semantic_projection(
            envelope,
            base_projection,
            source_records=forged_records,
            raw_dir=raw_dir,
        )


def test_native_atom_metadata_forgery_is_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    triage, records, bindings, base_projection, raw_dir, _child = _real_fixture(
        tmp_path, monkeypatch
    )
    envelope = materialize_c2_semantic_envelope(
        triage,
        records,
        bindings,
        projection_id="2" * 64,
        source_sha256="3" * 64,
    )
    forged_atom = replace(
        base_projection.atoms[0],
        event_time="2026-09-09T00:16:00+09:00",
    )
    forged_projection = replace(base_projection, atoms=(forged_atom,))
    with pytest.raises(C2SemanticProjectionError, match="atom metadata"):
        build_c2_semantic_projection(
            envelope,
            forged_projection,
            source_records=records,
            raw_dir=raw_dir,
        )


def test_oversized_native_packet_is_rejected_as_a_whole_context(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    triage, records, bindings, base_projection, raw_dir, _child = _real_fixture(
        tmp_path, monkeypatch, long_context=True
    )
    envelope = materialize_c2_semantic_envelope(
        triage,
        records,
        bindings,
        projection_id="2" * 64,
        source_sha256="3" * 64,
    )
    program = compile_retrieval_program(
        "feature enabled",
        {
            "as_of": "2026-09-09T00:30:00+09:00",
            "claim_slots": [{"slot_id": "answer", "claim": "feature enabled"}],
            "required_evidence": [
                {
                    "claim_slot": "answer",
                    "minimum_atoms": 1,
                    "relations": ["supports"],
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
    run = run_c2_semantic_retrieval(
        envelope,
        base_projection,
        source_records=records,
        program=program,
        raw_dir=raw_dir,
    )
    assert run.packet.abstained is False
    assert len(run.packet.canonical_bytes()) > 3_000
    # The bounded compiler returns no partial claim, so the trailing condition
    # cannot be injected without the complete native event context.
    assert compile_bounded_evidence_context(run.packet, max_chars=3_000) is None


def test_cli_fixed_triage_round_trip_preserves_raw_and_keeps_stdout_metadata_only(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    triage, records, bindings, _base_projection, raw_dir, child = _real_fixture(
        tmp_path, monkeypatch
    )
    triage_path = tmp_path / "triage-result.json"
    triage_path.write_text(
        json.dumps(triage, ensure_ascii=False, sort_keys=True), encoding="utf-8"
    )
    isolated_root = tmp_path / "isolated"
    isolated_root.mkdir()
    (isolated_root / "config.toml").write_text(
        "[raw]\nlayout = \"v2\"\n", encoding="utf-8"
    )
    raw_store = RawStore(raw_dir, mode="v2")
    unit = raw_store.resolve(bindings[0]["raw_id"])
    assert unit is not None
    before_raw = raw_store.read_bytes(unit)
    repo = Path(__file__).resolve().parents[1]
    completed = subprocess.run(
        [
            sys.executable,
            "scripts/project_c2_evidence.py",
            "--root",
            str(isolated_root),
            "--raw-dir",
            str(raw_dir),
            "--child",
            str(child),
            "--triage-result",
            str(triage_path),
        ],
        cwd=repo,
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    result = json.loads(completed.stdout)
    assert result["status"] == "stored"
    assert result["model_called"] is False
    assert result["page_operations_applied"] is False
    assert records[0]["text"] not in completed.stdout
    assert raw_store.read_bytes(unit) == before_raw


def test_cli_rejects_output_root_inside_raw_before_import_or_write(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    triage, _records, _bindings, _base_projection, raw_dir, child = _real_fixture(
        tmp_path, monkeypatch
    )
    triage_path = tmp_path / "triage-result.json"
    triage_path.write_text(
        json.dumps(triage, ensure_ascii=False, sort_keys=True), encoding="utf-8"
    )
    forbidden_root = raw_dir / "isolated"
    repo = Path(__file__).resolve().parents[1]
    completed = subprocess.run(
        [
            sys.executable,
            "scripts/project_c2_evidence.py",
            "--root",
            str(forbidden_root),
            "--raw-dir",
            str(raw_dir),
            "--child",
            str(child),
            "--triage-result",
            str(triage_path),
        ],
        cwd=repo,
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 2
    assert not forbidden_root.exists()
    assert "C2" not in completed.stdout


def test_cas_manifest_tampering_is_rejected(tmp_path: Path) -> None:
    triage, records, bindings, _projection = _fixture()
    envelope = materialize_c2_semantic_envelope(
        triage,
        records,
        bindings,
        projection_id="2" * 64,
        source_sha256="3" * 64,
    )
    store = ResearchStore(tmp_path / "research")
    store.durable_cas = tmp_path / "durable-cas"
    store.durable_manifests = tmp_path / "durable-manifests"
    artifact = store_c2_semantic_envelope(
        store, envelope, source_uri="projection:" + "2" * 64
    )
    manifest_path = store.durable_manifests / (artifact.sha256 + ".json")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["metadata"]["source_sha256"] = "4" * 64
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(C2SemanticProjectionError, match="manifest"):
        load_c2_semantic_envelope(store, artifact.artifact_id)

    manifest_path.unlink()
    with pytest.raises(C2SemanticProjectionError, match="manifest is missing"):
        load_c2_semantic_envelope(store, artifact.artifact_id)
