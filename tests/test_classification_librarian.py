from __future__ import annotations

from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from chronovisor import classification_engine, classification_model_worker
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


def test_invalid_optional_secondary_does_not_discard_valid_primary(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        classification_model_worker.ollama,
        "chat",
        lambda *args, **kwargs: (
            '{"decisions":[{"uid":"uid-1",'
            '"primary_notation":"004.8",'
            '"secondary_notations":["999"],'
            '"confidence":0.9,"rationale":"ok"}]}'
        ),
    )

    decisions, _calls = classification_model_worker._call(
        model="test",
        keep_alive="0",
        pages=[
            {
                "uid": "uid-1",
                "title": "AI",
                "candidates": [{"notation": "004.8"}],
            }
        ],
        role="primary-proposer",
    )

    assert decisions[0]["primary_notation"] == "004.8"
    assert decisions[0]["secondary_notations"] == []
    assert decisions[0]["_rejected_secondary_notations"] == ["999"]
    assert "_invalid_reason" not in decisions[0]


def test_tie_break_candidates_are_limited_to_independent_proposals() -> None:
    pages = [
        {
            "uid": "uid-1",
            "candidates": [
                {"notation": "004.8"},
                {"notation": "51"},
                {"notation": "62"},
            ],
        }
    ]

    narrowed = classification_model_worker._tie_candidate_pages(
        pages,
        [{"primary_notation": "004.8"}],
        [{"primary_notation": "51"}],
    )

    assert [row["notation"] for row in narrowed[0]["candidates"]] == [
        "004.8",
        "51",
    ]


def test_consensus_batch_retries_foreground_preemption(monkeypatch) -> None:
    class FakeStore:
        def __init__(self) -> None:
            self.failures: list[dict] = []
            self.completed: list[dict] = []

        def merge_item(self, **kwargs):
            return {"item": {"key": "batch-key", "status": "pending_local"}}

        def claim_attempt(self, *args, **kwargs):
            return {"claimed": True}

        def fail_attempt(self, *args, **kwargs):
            self.failures.append(kwargs)

        def complete(self, *args, **kwargs):
            self.completed.append(kwargs)

    store = FakeStore()
    results = iter(
        [
            SimpleNamespace(
                status="cancelled",
                value=None,
                error="cancelled for foreground sync",
            ),
            SimpleNamespace(
                status="completed",
                value={
                    "decisions": [
                        {
                            "uid": "uid-1",
                            "primary_notation": "004.8",
                            "status": "proposed",
                        }
                    ],
                    "model_calls": 2,
                },
                error="",
            ),
        ]
    )

    @contextmanager
    def fake_lane(*args, **kwargs):
        yield object()

    monkeypatch.setattr(
        classification_engine,
        "librarian_convergence_store",
        lambda _root: store,
    )
    monkeypatch.setattr(
        classification_engine,
        "load_udc_package",
        lambda _root: SimpleNamespace(checksum="sha256:test"),
    )
    monkeypatch.setattr(classification_engine, "research_lane", fake_lane)
    monkeypatch.setattr(
        classification_engine,
        "run_cancellable_command",
        lambda *args, **kwargs: next(results),
    )
    monkeypatch.setattr(classification_engine, "sync_pending", lambda: False)

    decisions = classification_engine.run_consensus_batches(
        [
            {
                "uid": "uid-1",
                "source_sha256": "sha256:source",
                "candidates": [{"notation": "004.8"}],
            }
        ],
        root=Path("/tmp/test-chronovisor"),
    )

    assert decisions[0]["uid"] == "uid-1"
    assert store.failures[0]["consume_attempt"] is False
    assert len(store.completed) == 1


def test_model_stage_cache_resumes_from_last_complete_chunk(
    tmp_path: Path,
    monkeypatch,
) -> None:
    pages = [
        {
            "uid": f"uid-{index}",
            "source_sha256": f"sha256:{index}",
            "candidates": [{"notation": "004.8"}],
        }
        for index in range(6)
    ]
    cache_path = tmp_path / "stage-cache.json"
    cache = {
        "schema": classification_model_worker.STAGE_CACHE_SCHEMA,
        "cache_key": "cache-key",
        "stages": {},
    }
    first_calls: list[list[str]] = []

    def interrupted_call(**kwargs):
        first_calls.append([str(row["uid"]) for row in kwargs["pages"]])
        if len(first_calls) == 2:
            raise RuntimeError("foreground killed child")
        return (
            [
                {
                    "uid": row["uid"],
                    "primary_notation": "004.8",
                    "secondary_notations": [],
                    "confidence": 0.9,
                    "rationale": "ok",
                }
                for row in kwargs["pages"]
            ],
            1,
        )

    monkeypatch.setattr(classification_model_worker, "_call", interrupted_call)
    with pytest.raises(RuntimeError, match="foreground"):
        classification_model_worker._cached_stage_call(
            cache=cache,
            cache_path=cache_path,
            stage="primary",
            model="test",
            keep_alive="0",
            pages=pages,
            role="primary-proposer",
        )

    resumed_cache = classification_model_worker._load_stage_cache(
        cache_path,
        "cache-key",
    )
    resumed_calls: list[list[str]] = []

    def resumed_call(**kwargs):
        resumed_calls.append([str(row["uid"]) for row in kwargs["pages"]])
        return (
            [
                {
                    "uid": row["uid"],
                    "primary_notation": "004.8",
                    "secondary_notations": [],
                    "confidence": 0.9,
                    "rationale": "ok",
                }
                for row in kwargs["pages"]
            ],
            1,
        )

    monkeypatch.setattr(classification_model_worker, "_call", resumed_call)
    decisions, model_calls = classification_model_worker._cached_stage_call(
        cache=resumed_cache,
        cache_path=cache_path,
        stage="primary",
        model="test",
        keep_alive="0",
        pages=pages,
        role="primary-proposer",
    )

    assert len(decisions) == 6
    assert model_calls == 1
    assert resumed_calls == [["uid-5"]]
