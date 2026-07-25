from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from chronovisor.classification import (
    ClassificationError,
    ControlledSubject,
    classification_authority_status,
    classification_source_sha256,
    default_udc_package,
    load_ndc_overlay,
    propose_from_legacy_metadata,
    render_call_number,
    validate_controlled_subject,
    validate_record,
)
from chronovisor import classification_model_worker
from chronovisor.librarian import capture_baseline, run_shadow
from chronovisor.librarian_status import build_librarian_status


def _write_page(path: Path, *, tags: str = "") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tag_line = f"tags: [{tags}]\n" if tags else ""
    path.write_text(
        "---\n"
        f"title: {path.stem}\n"
        "updated: 2026-07-25\n"
        f"{tag_line}"
        "---\n\n"
        f"# {path.stem}\n",
        encoding="utf-8",
    )


def test_bootstrap_classification_cannot_become_authority() -> None:
    package = replace(default_udc_package(), complete=False)
    record = propose_from_legacy_metadata(
        tags=["d/ai", "t/architecture"],
        page_type="architecture",
        lifecycle="active",
        sensitivity="normal",
        evidence_ref="page-sha256:abc",
        package=package,
    )
    assert record.primary.notation == "0"
    assert render_call_number(record, project="chronovisor") == (
        "UDCS 0 · CHRONOVISOR · ARC"
    )
    with pytest.raises(ClassificationError, match="complete UDC package"):
        validate_record(record, package=package, require_complete_package=True)
    authority = classification_authority_status(Path("/nonexistent"), package=package)
    assert authority["active"] is False
    assert authority["reason"] == (
        "complete_udc_package_missing,locked_calibration_not_adopted"
    )


def test_cvo_subject_and_ndc_overlay_are_explicitly_versioned(
    tmp_path: Path,
) -> None:
    package = default_udc_package()
    broader = package.by_notation("0")
    assert broader is not None
    subject = ControlledSubject(
        concept_uri="cvo:subject/agent-memory",
        broader_udc_uri=str(broader["uri"]),
        label="Agent memory",
        definition="Long-term memory systems for software agents.",
        inclusion_examples=("retrieval memory",),
        exclusion_examples=("human clinical memory",),
        version="1.0.0",
        authority_epoch=1,
    )
    validate_controlled_subject(subject, package=package)
    assert load_ndc_overlay(tmp_path) is None


def test_bundled_udc_snapshot_is_complete_licensed_and_japanese() -> None:
    package = default_udc_package()
    concepts = list(package.concepts.values())

    assert package.complete is True
    assert package.license == "CC BY-SA 3.0"
    assert len(concepts) >= 2_500
    assert (
        sum(bool(row.get("label_ja")) for row in concepts) / len(concepts)
        >= 0.95
    )
    artificial_intelligence = package.by_notation("004.8")
    assert artificial_intelligence is not None
    assert artificial_intelligence["label_en"]
    assert artificial_intelligence["label_ja"]


def test_classification_source_hash_ignores_adopted_metadata() -> None:
    before = (
        "---\n"
        "title: Memory\n"
        "tags: [d/ai]\n"
        "---\n\n"
        "# Memory\n\n"
        "Stable body.\n"
    )
    after = (
        "---\n"
        "title: Memory\n"
        "tags: [d/ai]\n"
        "uid: 019f0000-0000-7000-8000-000000000000\n"
        "call_number: UDCS 004.8\n"
        "classification_status: adopted\n"
        'classification_json: {"status":"adopted"}\n'
        "---\n\n"
        "# Memory\n\n"
        "Stable body.\n"
    )

    assert classification_source_sha256(before) == classification_source_sha256(after)


def test_librarian_shadow_progress_is_visible_but_not_false_green(
    tmp_path: Path,
) -> None:
    _write_page(tmp_path / "pages" / "ai-memory.md", tags="d/ai, t/architecture")
    _write_page(tmp_path / "pages" / "unknown.md")

    result = run_shadow(root=tmp_path, full_sweep=True)
    repeated = run_shadow(root=tmp_path, full_sweep=True)
    status = build_librarian_status(tmp_path)

    assert result["status"] == "ok"
    assert result["remaining"] == 0
    assert repeated["selected"] == 0
    assert repeated["registry"]["updated"] == 0
    assert repeated["scope_generation"] == result["scope_generation"]
    assert status["state"] == "NOT_READY"
    assert status["authority"]["active"] is False
    assert status["progress"]["uid"]["numerator"] == 2
    assert status["progress"]["classification_shadow"]["numerator"] == 2
    assert status["progress"]["full_sweep"]["current"] is True
    assert status["queue"]["held"] == 1

    _write_page(tmp_path / "pages" / "late-arrival.md", tags="d/ai")
    drifted = build_librarian_status(tmp_path)
    assert drifted["queue"]["actionable"] == 1
    assert drifted["progress"]["uid"]["numerator"] == 2
    assert drifted["progress"]["uid"]["denominator"] == 3
    assert drifted["progress"]["full_sweep"]["current"] is False
    assert drifted["debts"]["scope_unregistered"] == 1


def test_librarian_dry_run_is_byte_for_byte_read_only(tmp_path: Path) -> None:
    _write_page(tmp_path / "pages" / "alpha.md", tags="d/ai")
    before = {
        path.relative_to(tmp_path): path.read_bytes()
        for path in tmp_path.rglob("*")
        if path.is_file()
    }

    result = run_shadow(root=tmp_path, full_sweep=True, dry_run=True)

    after = {
        path.relative_to(tmp_path): path.read_bytes()
        for path in tmp_path.rglob("*")
        if path.is_file()
    }
    assert result["status"] == "dry_run"
    assert before == after


def test_baseline_records_missing_locked_fixture_without_wiki_mutation(
    tmp_path: Path,
) -> None:
    _write_page(tmp_path / "pages" / "alpha.md", tags="d/ai")

    baseline = capture_baseline(root=tmp_path, write=False)

    assert baseline["pages"] == 1
    assert baseline["fixtures"] == {
        "dev_200": False,
        "holdout_100": False,
        "locked": False,
    }
    assert baseline["wiki_mutated"] is False
    assert not (tmp_path / "runtime").exists()


def test_model_worker_splits_a_truncated_json_batch(monkeypatch) -> None:
    pages = [
        {
            "uid": f"uid-{index}",
            "title": f"Page {index}",
            "candidates": [{"notation": "0"}],
        }
        for index in range(2)
    ]
    calls: list[int] = []

    def fake_chat(*args, **kwargs):
        count = kwargs["format"]["properties"]["decisions"]["minItems"]
        calls.append(count)
        if count == 2:
            return '{"decisions":[{"uid":"uid-0"'
        uid = "uid-0" if len(calls) == 2 else "uid-1"
        return (
            '{"decisions":[{"uid":"'
            + uid
            + '","primary_notation":"0","secondary_notations":[],'
            '"confidence":0.9,"rationale":"ok"}]}'
        )

    monkeypatch.setattr(classification_model_worker.ollama, "chat", fake_chat)

    decisions, model_calls = classification_model_worker._call(
        model="test",
        keep_alive="0",
        pages=pages,
        role="primary-proposer",
    )

    assert [row["uid"] for row in decisions] == ["uid-0", "uid-1"]
    assert model_calls == 3
    assert calls == [2, 1, 1]
