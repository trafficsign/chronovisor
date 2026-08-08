from __future__ import annotations

import hashlib
import json
from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from chronovisor.classification import (
    classification_calibration,
    classification_engine,
    classification_model_worker,
)
from chronovisor.classification.classification import (
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
from chronovisor.librarian.librarian import (
    capture_baseline,
    run_legacy_udc_shadow,
)
from chronovisor.librarian.librarian_status import build_librarian_status


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

    result = run_legacy_udc_shadow(root=tmp_path, full_sweep=True)
    repeated = run_legacy_udc_shadow(root=tmp_path, full_sweep=True)
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

    result = run_legacy_udc_shadow(
        root=tmp_path,
        full_sweep=True,
        dry_run=True,
    )

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


def test_dev_audit_corrects_reviewed_labels_without_opening_holdout(
    tmp_path: Path,
) -> None:
    fixture_root = tmp_path / "classification" / "fixtures"
    fixture_root.mkdir(parents=True)
    dev_path = fixture_root / "classification-dev-200.jsonl"
    holdout_path = fixture_root / "classification-holdout-100.jsonl"
    manifest_path = fixture_root / "manifest.json"
    row = {
        "uid": "uid-1",
        "source_sha256": "sha256:source",
        "gold_primary_notation": "004.3",
        "gold_allowed_primary_notations": ["004.3"],
        "gold_rationale": "Original local consensus.",
        "candidates": [
            {"notation": "004.3"},
            {"notation": "62"},
            {"notation": "6"},
        ],
    }
    dev_path.write_text(json.dumps(row) + "\n", encoding="utf-8")
    holdout_before = b'{"uid":"sealed"}\n'
    holdout_path.write_bytes(holdout_before)
    manifest_path.write_text(
        json.dumps(
            {
                "dev": {
                    "path": str(dev_path),
                    "count": 1,
                    "sha256": "sha256:before",
                },
                "holdout": {
                    "path": str(holdout_path),
                    "count": 1,
                    "sha256": "sha256:sealed",
                    "opened_at": None,
                },
            }
        ),
        encoding="utf-8",
    )
    audit_path = tmp_path / "audit.json"
    audit_path.write_text(
        json.dumps(
            {
                "schema": classification_calibration.DEV_AUDIT_SCHEMA,
                "audit_id": "epoch2-dev-review-v1",
                "reviewed_at": "2026-07-26T06:30:00+09:00",
                "corrections": [
                    {
                        "uid": "uid-1",
                        "source_sha256": "sha256:source",
                        "original_gold_primary_notation": "004.3",
                        "gold_primary_notation": "62",
                        "gold_allowed_primary_notations": ["62", "6"],
                        "gold_rationale": "Automotive body specification is engineering.",
                        "reason": "Computer hardware was a false lexical match.",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    receipt = classification_calibration.apply_dev_audit(
        tmp_path,
        audit_path,
    )

    updated = json.loads(dev_path.read_text(encoding="utf-8"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert receipt["status"] == "verified"
    assert receipt["correction_count"] == 1
    assert updated["gold_primary_notation"] == "62"
    assert updated["gold_allowed_primary_notations"] == ["62", "6"]
    assert holdout_path.read_bytes() == holdout_before
    assert manifest["holdout"]["opened_at"] is None
    assert manifest["dev"]["sha256"] == (
        "sha256:" + hashlib.sha256(dev_path.read_bytes()).hexdigest()
    )
    assert (
        fixture_root
        / "epochs"
        / "epoch2-dev-review-v1-pre-audit"
        / dev_path.name
    ).is_file()


def test_calibration_resumes_opened_holdout_from_sealed_preregistration(
    tmp_path: Path,
    monkeypatch,
) -> None:
    fixture_root = tmp_path / "classification" / "fixtures"
    fixture_root.mkdir(parents=True)
    dev_path = fixture_root / "classification-dev-200.jsonl"
    holdout_path = fixture_root / "classification-holdout-100.jsonl"
    manifest_path = fixture_root / "manifest.json"
    dev_path.write_text('{"uid":"dev"}\n', encoding="utf-8")
    holdout_row = {
        "uid": "holdout-1",
        "gold_primary_notation": "004.8",
        "gold_allowed_primary_notations": ["004.8"],
        "gold_expected_status": "proposed",
    }
    holdout_path.write_text(json.dumps(holdout_row) + "\n", encoding="utf-8")
    manifest_path.write_text(
        json.dumps(
            {
                "dev": {
                    "sha256": (
                        "sha256:"
                        + hashlib.sha256(dev_path.read_bytes()).hexdigest()
                    )
                },
                "holdout": {
                    "sha256": (
                        "sha256:"
                        + hashlib.sha256(holdout_path.read_bytes()).hexdigest()
                    ),
                    "opened_at": "2026-07-26T07:11:35+00:00",
                },
            }
        ),
        encoding="utf-8",
    )
    (tmp_path / "classification" / "calibration.json").write_text(
        '{"status":"rejected"}\n',
        encoding="utf-8",
    )
    fingerprint = {
        "dev_fixture_sha256": "sha256:dev",
        "package_checksum": "sha256:package",
        "config_digest": "sha256:config",
    }
    classification_calibration.write_sealed_json(
        tmp_path / "classification" / "calibration-preregistration.json",
        {
            "schema": classification_calibration.PREREGISTRATION_SCHEMA,
            "input_fingerprint": fingerprint,
            "thresholds": {"minimum_confidence": 0.7},
            "dev_metrics": {"forced_misclassification_rate": 0.005},
        },
    )
    monkeypatch.setattr(
        classification_calibration,
        "calibration_input_fingerprint",
        lambda _root: fingerprint,
    )
    calls: list[str] = []

    def consensus(rows, **kwargs):
        calls.append(kwargs["run_namespace"])
        assert rows == [holdout_row]
        return [
            {
                "uid": "holdout-1",
                "primary_notation": "004.8",
                "quorum": 2,
                "confidence": 0.9,
            }
        ]

    monkeypatch.setattr(
        classification_calibration,
        "run_consensus_batches",
        consensus,
    )
    adopted: dict[str, object] = {}

    def adopt(_root, **kwargs):
        adopted.update(kwargs)
        return {"status": "adopted"}

    monkeypatch.setattr(
        classification_calibration,
        "adopt_calibration",
        adopt,
    )

    result = classification_calibration.calibrate(tmp_path)

    assert result["status"] == "adopted"
    assert calls == ["calibration-holdout-epoch2-v2"]
    assert adopted["dev_metrics"] == {
        "forced_misclassification_rate": 0.005
    }
    assert adopted["holdout_metrics"]["exact_match_rate"] == 1.0


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
    assert model_calls == 5
    assert resumed_calls == [
        ["uid-1"],
        ["uid-2"],
        ["uid-3"],
        ["uid-4"],
        ["uid-5"],
    ]


def test_model_stage_cache_resumes_after_safely_marked_invalid_output(
    tmp_path: Path,
    monkeypatch,
) -> None:
    pages = [
        {
            "uid": f"uid-{index}",
            "source_sha256": f"sha256:{index}",
            "candidates": [{"notation": "004.8"}],
        }
        for index in range(2)
    ]
    invalid = {
        "uid": "uid-0",
        "primary_notation": "999",
        "secondary_notations": [],
        "confidence": 0.0,
        "rationale": "model left the host candidate set",
        "_invalid_reason": "notation_outside_host_candidates",
    }
    cache_path = tmp_path / "stage-cache.json"
    cache = {
        "schema": classification_model_worker.STAGE_CACHE_SCHEMA,
        "cache_key": "cache-key",
        "stages": {"primary": [invalid]},
    }
    calls: list[list[str]] = []

    def resumed_call(**kwargs):
        calls.append([str(row["uid"]) for row in kwargs["pages"]])
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
        cache=cache,
        cache_path=cache_path,
        stage="primary",
        model="test",
        keep_alive="0",
        pages=pages,
        role="primary-proposer",
    )

    assert decisions[0] == invalid
    assert decisions[1]["uid"] == "uid-1"
    assert model_calls == 1
    assert calls == [["uid-1"]]
    unsafe = {**invalid}
    unsafe.pop("_invalid_reason")
    assert (
        classification_model_worker._valid_cached_stage([unsafe], pages[:1])
        is None
    )


def test_expected_hold_is_a_separate_safety_gate() -> None:
    fixtures = [
        {
            "uid": "assignable",
            "gold_primary_notation": "004.8",
            "gold_expected_status": "proposed",
        },
        {
            "uid": "safety-hold",
            "gold_primary_notation": "62",
            "gold_expected_status": "held",
        },
    ]
    decisions = [
        {
            "uid": "assignable",
            "primary_notation": "004.8",
            "status": "proposed",
        },
        {
            "uid": "safety-hold",
            "primary_notation": "62",
            "status": "held",
        },
    ]

    metrics = classification_engine.evaluate_predictions(fixtures, decisions)

    assert metrics["primary_assignment_rate"] == 1.0
    assert metrics["exact_match_rate"] == 1.0
    assert metrics["hold_rate"] == 0.5
    assert metrics["expected_hold_recall"] == 1.0
    assert metrics["expected_hold_escape_rate"] == 0.0
    assert metrics["forced_misclassification_rate"] == 0.0


def test_expected_hold_escape_is_forced_misclassification() -> None:
    fixtures = [
        {
            "uid": "safety-hold",
            "gold_primary_notation": "62",
            "gold_expected_status": "held",
        }
    ]
    decisions = [
        {
            "uid": "safety-hold",
            "primary_notation": "62",
            "status": "proposed",
        }
    ]

    metrics = classification_engine.evaluate_predictions(fixtures, decisions)

    assert metrics["expected_hold_escape_rate"] == 1.0
    assert metrics["forced_misclassification_rate"] == 1.0
