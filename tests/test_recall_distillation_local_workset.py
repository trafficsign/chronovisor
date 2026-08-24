from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from chronovisor.core import canonical_json
from chronovisor.recall import recall_distillation as distill
from chronovisor.recall import recall_distillation_store as store
from chronovisor.recall.recall_distillation_workset import DistillationWorkset


def _sealed_counterfactual_exposure(
    root: Path, *, candidate_ids: list[object] | None = None
) -> dict[str, object]:
    """Create the immutable receipt required by the real CF producer."""

    artifact_id, _, _ = store.write_immutable(
        store.distillation_dir(root) / "exposures",
        {
            "kind": "exact-rendered-exposure",
            "candidate_refs": [],
            "candidate_pool_refs": [],
            "candidate_feature_snapshot": [],
        },
        schema="chronovisor.recall-exact-exposure.v1",
    )
    return {
        "exposure_artifact_id": artifact_id,
        "candidate_ids": candidate_ids or [],
    }


def _record_counterfactual_exposure(
    root: Path,
    *,
    host: str,
    session_id_sha256: str,
    query_sha256: str,
    candidate_ids: list[object] | None = None,
) -> dict[str, object]:
    """Create the canonical receipt projection used by root lineage checks."""

    candidate_ids = [7] if candidate_ids is None else candidate_ids
    candidate_refs = (
        [
            {
                "candidate_id": 7,
                "content_sha256": "text",
                "rendered_context": "candidate",
            }
        ]
        if candidate_ids == [7]
        else []
    )
    binding = {
        "decision_id": "decision-1",
        "host": host,
        "session_id_sha256": session_id_sha256,
        "query_semantic_sha256": query_sha256,
        "policy_id": "a" * 64,
        "candidate_ids": candidate_ids,
        "candidate_refs_sha256": "b" * 64,
        "candidate_pool_refs_sha256": "c" * 64,
        "candidate_feature_snapshot_sha256": "d" * 64,
        "runtime_observation_sha256": "e" * 64,
        "render_sha256": "f" * 64,
        "renderer_revision": "test-renderer-v1",
        "context_style": "test",
        "candidate_snapshot_sha256": "1" * 64,
        "observed_at": "2026-01-01T00:00:00Z",
    }
    artifact_id, _, _ = store.write_immutable(
        store.distillation_dir(root) / "exposures",
        {
            "kind": "exact-rendered-exposure",
            **binding,
            "candidate_refs": candidate_refs,
            "candidate_pool_refs": [],
            "candidate_feature_snapshot": [],
            "runtime_observation": {},
        },
        schema="chronovisor.recall-exact-exposure.v1",
    )
    receipt = {
        "kind": "prospective-exact-exposure-v1",
        **binding,
        "binding_sha256": canonical_json.canonical_json_sha256_strict(binding),
        "exposure_artifact_id": artifact_id,
    }
    store.append_chain(
        store.distillation_dir(root) / "exposure-receipts.jsonl", receipt
    )
    return {"exposure_artifact_id": artifact_id, "candidate_ids": candidate_ids}


def test_local_workset_reconciles_appended_label_without_repeat_call(
    tmp_path: Path, monkeypatch
) -> None:
    class Teacher:
        local = True

        def __init__(self) -> None:
            self.calls = 0

        def evaluate(self, payload: object) -> dict[str, object]:
            self.calls += 1
            assert isinstance(payload, dict)
            return {
                "labels": [
                    {
                        "candidate_id": item["candidate_id"],
                        "verdict": "relevant",
                    }
                    for item in payload["candidates"]
                ]
            }

    raw_paths: list[Path] = []
    alternate_raw = tmp_path / "alternate-raw"
    monkeypatch.setattr(
        distill,
        "committed_raw_watermark",
        lambda path: raw_paths.append(path) or "raw-1",
    )
    monkeypatch.setattr(
        distill,
        "teacher_assignment",
        lambda *_args: {
            "revision": distill.ASSIGNMENT_REVISION,
            "owner": distill.TEACHER_ROLES[0],
            "probe_revision": distill.PROBE_REVISION,
            "probe": False,
            "routes": [distill.TEACHER_ROLES[0]],
        },
    )
    teacher = Teacher()
    teachers = {role: teacher for role in distill.TEACHER_ROLES}
    rally = {"rally_id": "rally-1", "query_sha256": "query", "context_refs": []}
    snapshots = {
        "rally-1": {
            "candidates": [{"candidate_id": "candidate-1", "text_sha256": "text"}]
        }
    }
    label_path = store.distillation_dir(tmp_path) / "label-ledger.jsonl"
    config = distill.DistillationConfig(enabled=True, max_input_bytes=4_096)

    first = distill._run_local_teacher_batch(
        root=tmp_path,
        raw_dir=alternate_raw,
        config=config,
        teachers=teachers,
        snapshots=snapshots,
        rally_by_id={"rally-1": rally},
        texts={"query": "question", "text": "evidence"},
        label_path=label_path,
        label_rows=[],
        structural_verifier=lambda *_args: None,
    )
    assert first.labels_written == 1
    assert teacher.calls == 1
    assert raw_paths and all(path == alternate_raw for path in raw_paths)

    resumed = distill._run_local_teacher_batch(
        root=tmp_path,
        raw_dir=alternate_raw,
        config=config,
        teachers=teachers,
        snapshots=snapshots,
        rally_by_id={"rally-1": rally},
        texts={"query": "question", "text": "evidence"},
        label_path=label_path,
        label_rows=store.read_chain(label_path),
        structural_verifier=lambda *_args: None,
    )
    assert resumed.labels_written == 0
    assert teacher.calls == 1
    assert (
        DistillationWorkset(
            store.distillation_dir(tmp_path) / "local-workset.sqlite3"
        ).status()["completed"]
        == 1
    )

    legacy_rows = [
        {key: value for key, value in row.items() if key != "work_id"}
        for row in store.read_chain(label_path)
    ]
    changed = distill._run_local_teacher_batch(
        root=tmp_path,
        raw_dir=alternate_raw,
        config=config,
        teachers=teachers,
        snapshots={
            "rally-1": {
                "candidates": [{"candidate_id": "candidate-1", "text_sha256": "text-2"}]
            }
        },
        rally_by_id={"rally-1": rally},
        texts={"query": "question", "text-2": "changed evidence"},
        label_path=label_path,
        label_rows=legacy_rows,
        structural_verifier=lambda *_args: None,
    )
    assert changed.labels_written == 1
    assert teacher.calls == 2


def test_local_workset_rejects_same_count_head_rewrite() -> None:
    class Workset:
        def watermark(self) -> dict[str, object]:
            return {
                "candidate_count": 3,
                "candidate_head": "a" * 64,
                "label_count": 2,
                "label_head": "b" * 64,
            }

        def advance(self, *_args: object) -> None:
            raise AssertionError("must fail before advance")

    with pytest.raises(distill.DistillationError, match="watermark regressed"):
        distill._advance_local_workset(
            Workset(),
            [],
            {
                "candidate_count": 3,
                "candidate_head": "c" * 64,
                "label_count": 2,
                "label_head": "b" * 64,
            },
        )


def test_counterfactual_payload_change_creates_new_work(
    tmp_path: Path, monkeypatch
) -> None:
    class Counterfactual:
        local = True

        def __init__(self) -> None:
            self.calls = 0

        def compare(self, _payload: object) -> dict[str, object]:
            self.calls += 1
            return {
                "verdict": "helpful",
                "order_agreement": True,
                "blind_orders": ["a0_first", "a1_first"],
                "a0_sha256": "c" * 64,
                "a1_sha256": "d" * 64,
                "generator_route_identity": {
                    "role": "generator",
                    "provider": "test",
                    "model": "generator",
                    "location": "local",
                },
                "judge_route_identity": {
                    "role": "judge",
                    "provider": "test",
                    "model": "judge",
                    "location": "local",
                },
                "generator_model_digest": "a" * 64,
                "judge_model_digest": "b" * 64,
            }

    raw_dir = tmp_path / "raw-alt"
    monkeypatch.setattr(distill, "committed_raw_watermark", lambda _path: "raw-1")
    counterfactual = Counterfactual()
    rally = {
        "rally_id": "rally-1",
        "query_sha256": "query",
        "context_refs": [],
        "actual_answer_refs": [{"semantic_sha256": "answer"}],
        "exposure_receipts": [_sealed_counterfactual_exposure(tmp_path)],
    }
    label_path = store.distillation_dir(tmp_path) / "label-ledger.jsonl"
    config = distill.DistillationConfig(enabled=True, max_input_bytes=4_096)
    first = distill._run_counterfactual_block(
        execute=True,
        root=tmp_path,
        raw_dir=raw_dir,
        config=config,
        counterfactual=counterfactual,
        snapshots={
            "rally-1": {
                "snapshot_sha256": "one",
                "candidates": [
                    {"candidate_id": "candidate-1", "text_sha256": "text-1"}
                ],
            }
        },
        rally_by_id={"rally-1": rally},
        texts={"query": "question", "answer": "answer", "text-1": "one"},
        label_path=label_path,
        label_rows=[],
    )
    assert first.written == 1
    second = distill._run_counterfactual_block(
        execute=True,
        root=tmp_path,
        raw_dir=raw_dir,
        config=config,
        counterfactual=counterfactual,
        snapshots={
            "rally-1": {
                "snapshot_sha256": "two",
                "candidates": [
                    {"candidate_id": "candidate-1", "text_sha256": "text-2"}
                ],
            }
        },
        rally_by_id={"rally-1": rally},
        texts={"query": "question", "answer": "answer", "text-2": "two"},
        label_path=label_path,
        label_rows=store.read_chain(label_path),
    )
    assert second.written == 1
    assert counterfactual.calls == 2


def test_counterfactual_without_sealed_exposure_defers_without_label_or_attempt(
    tmp_path: Path, monkeypatch
) -> None:
    class Counterfactual:
        local = True

        def __init__(self) -> None:
            self.calls = 0

        def compare(self, _payload: object) -> dict[str, object]:
            self.calls += 1
            raise AssertionError("no exposure receipt must not reach the producer")

    monkeypatch.setattr(distill, "committed_raw_watermark", lambda _path: "raw-1")
    counterfactual = Counterfactual()
    label_path = store.distillation_dir(tmp_path) / "label-ledger.jsonl"
    result = distill._run_counterfactual_block(
        execute=True,
        root=tmp_path,
        raw_dir=tmp_path / "raw-alt",
        config=distill.DistillationConfig(enabled=True, max_input_bytes=4_096),
        counterfactual=counterfactual,
        snapshots={
            "rally-1": {
                "snapshot_sha256": "snapshot",
                "candidates": [
                    {"candidate_id": "candidate-1", "text_sha256": "text"}
                ],
            }
        },
        rally_by_id={
            "rally-1": {
                "rally_id": "rally-1",
                "query_sha256": "query",
                "context_refs": [],
                "actual_answer_refs": [{"semantic_sha256": "answer"}],
            }
        },
        texts={"query": "question", "answer": "answer", "text": "candidate"},
        label_path=label_path,
        label_rows=[],
    )

    status = DistillationWorkset(
        store.distillation_dir(tmp_path) / "local-workset.sqlite3"
    ).status("local-counterfactual")
    assert result.written == 0
    assert result.deferred is True
    assert counterfactual.calls == 0
    assert not label_path.exists()
    assert status["leased"] == 0
    assert status["ready"] == 1
    assert status["last_durable_receipt"]["generation"] >= 3
    assert status["last_durable_progress"]["progress_kind"] == "local-workset-v2"


def test_local_teacher_defers_cross_rally_duplicate_candidate_id(
    tmp_path: Path, monkeypatch
) -> None:
    class Teacher:
        local = True

        def __init__(self) -> None:
            self.calls: list[list[str]] = []

        def evaluate(self, payload: object) -> dict[str, object]:
            assert isinstance(payload, dict)
            candidate_ids = [
                str(item["candidate_id"])
                for item in payload["candidates"]
                if isinstance(item, dict)
            ]
            self.calls.append(candidate_ids)
            return {
                "labels": [
                    {"candidate_id": candidate_id, "verdict": "relevant"}
                    for candidate_id in candidate_ids
                ]
            }

    monkeypatch.setattr(distill, "committed_raw_watermark", lambda _path: "raw-1")
    monkeypatch.setattr(
        distill,
        "teacher_assignment",
        lambda *_args: {
            "revision": distill.ASSIGNMENT_REVISION,
            "owner": distill.TEACHER_ROLES[0],
            "probe_revision": distill.PROBE_REVISION,
            "probe": False,
            "routes": [distill.TEACHER_ROLES[0]],
        },
    )
    teacher = Teacher()
    teachers = {role: teacher for role in distill.TEACHER_ROLES}
    snapshots = {
        "rally-1": {
            "candidates": [{"candidate_id": "shared", "text_sha256": "text-1"}]
        },
        "rally-2": {
            "candidates": [{"candidate_id": "shared", "text_sha256": "text-2"}]
        },
    }
    rallies = {
        rally_id: {"rally_id": rally_id, "query_sha256": "query", "context_refs": []}
        for rally_id in snapshots
    }
    label_path = store.distillation_dir(tmp_path) / "label-ledger.jsonl"
    kwargs = {
        "root": tmp_path,
        "raw_dir": tmp_path / "raw-alt",
        "config": distill.DistillationConfig(enabled=True, max_input_bytes=4_096),
        "teachers": teachers,
        "snapshots": snapshots,
        "rally_by_id": rallies,
        "texts": {"query": "question", "text-1": "one", "text-2": "two"},
        "label_path": label_path,
        "structural_verifier": lambda *_args: None,
    }

    first = distill._run_local_teacher_batch(label_rows=[], **kwargs)
    second = distill._run_local_teacher_batch(
        label_rows=store.read_chain(label_path), **kwargs
    )

    assert (first.labels_written, second.labels_written) == (1, 1)
    assert teacher.calls == [["shared"], ["shared"]]
    labels = store.read_chain(label_path)
    assert {str(row["rally_id"]) for row in labels} == {"rally-1", "rally-2"}
    assert {str(row["teacher_profile"]) for row in labels} == {
        distill.LOCAL_TRIAD_PROFILE
    }
    assert (
        DistillationWorkset(
            store.distillation_dir(tmp_path) / "local-workset.sqlite3"
        ).status()["completed"]
        == 2
    )


def test_counterfactual_retries_missing_answer_digest_or_route_identity(
    tmp_path: Path, monkeypatch
) -> None:
    class Counterfactual:
        local = True

        def compare(self, _payload: object) -> dict[str, object]:
            return {
                "verdict": "helpful",
                "order_agreement": True,
                "blind_orders": ["a0_first", "a1_first"],
                "a0_sha256": "",
                "a1_sha256": "b" * 64,
                "generator_route_identity": {},
                "judge_route_identity": {
                    "role": "judge",
                    "provider": "test",
                    "model": "judge",
                    "location": "local",
                },
                "generator_model_digest": "a" * 64,
                "judge_model_digest": "b" * 64,
            }

    monkeypatch.setattr(distill, "committed_raw_watermark", lambda _path: "raw-1")
    result = distill._run_counterfactual_block(
        execute=True,
        root=tmp_path,
        raw_dir=tmp_path / "raw-alt",
        config=distill.DistillationConfig(enabled=True, max_input_bytes=4_096),
        counterfactual=Counterfactual(),
        snapshots={
            "rally-1": {
                "snapshot_sha256": "snapshot",
                "candidates": [{"candidate_id": "candidate-1", "text_sha256": "text"}],
            }
        },
        rally_by_id={
            "rally-1": {
                "rally_id": "rally-1",
                "query_sha256": "query",
                "context_refs": [],
                "actual_answer_refs": [{"semantic_sha256": "answer"}],
                "exposure_receipts": [_sealed_counterfactual_exposure(tmp_path)],
            }
        },
        texts={"query": "question", "answer": "answer", "text": "candidate"},
        label_path=store.distillation_dir(tmp_path) / "label-ledger.jsonl",
        label_rows=[],
    )

    assert result.written == 0
    assert result.deferred is True


def test_counterfactual_exposure_mismatch_precedes_malformed_pool(
    tmp_path: Path, monkeypatch
) -> None:
    class Counterfactual:
        local = True

        def compare(self, _payload: object) -> dict[str, object]:
            raise AssertionError("mismatched exposure must not reach the teacher")

    monkeypatch.setattr(distill, "committed_raw_watermark", lambda _path: "raw-1")
    monkeypatch.setattr(
        store,
        "read_sealed",
        lambda *_args, **_kwargs: {
            "artifact_id": "e" * 64,
            "candidate_refs": [
                {
                    "candidate_id": "candidate-1",
                    "content_sha256": "text",
                    "rendered_context": "candidate",
                }
            ],
            "candidate_pool_refs": [{"selected": False}],
        },
    )
    result = distill._run_counterfactual_block(
        execute=True,
        root=tmp_path,
        config=distill.DistillationConfig(enabled=True, max_input_bytes=4_096),
        counterfactual=Counterfactual(),
        snapshots={
            "rally-1": {
                "snapshot_sha256": "snapshot",
                "candidates": [{"candidate_id": "candidate-1", "text_sha256": "text"}],
            }
        },
        rally_by_id={
            "rally-1": {
                "rally_id": "rally-1",
                "query_sha256": "query",
                "context_refs": [],
                "actual_answer_refs": [{"semantic_sha256": "answer"}],
                "exposure_receipts": [
                    {
                        "exposure_artifact_id": "e" * 64,
                        "candidate_ids": ["other-candidate"],
                    }
                ],
            }
        },
        texts={"query": "question", "answer": "answer", "text": "candidate"},
        label_path=store.distillation_dir(tmp_path) / "label-ledger.jsonl",
        label_rows=[],
    )

    assert result.deferred is True
    assert result.written == 0
    assert (
        DistillationWorkset(
            store.distillation_dir(tmp_path) / "local-workset.sqlite3"
        ).status()["leased"]
        == 0
    )


def test_counterfactual_preserves_raw_candidate_id_in_label(
    tmp_path: Path, monkeypatch
) -> None:
    class Counterfactual:
        local = True

        def compare(self, _payload: object) -> dict[str, object]:
            return {
                "verdict": "helpful",
                "order_agreement": True,
                "blind_orders": ["a0_first", "a1_first"],
                "a0_sha256": "c" * 64,
                "a1_sha256": "d" * 64,
                "generator_route_identity": {
                    "role": "recall.distill.answer_generator",
                    "provider": "test",
                    "model": "generator",
                    "location": "local",
                },
                "judge_route_identity": {
                    "role": "recall.distill.utility_judge",
                    "provider": "test",
                    "model": "judge",
                    "location": "local",
                },
                "generator_model_digest": "a" * 64,
                "judge_model_digest": "b" * 64,
            }

    monkeypatch.setattr(distill, "committed_raw_watermark", lambda _path: "raw-1")
    from chronovisor.core import ollama

    roles = (
        *distill.TEACHER_ROLES,
        "recall.distill.answer_generator",
        "recall.distill.utility_judge",
    )
    routes = tuple(
        SimpleNamespace(
            role=role,
            provider="test",
            model=("generator" if role.endswith("answer_generator") else "judge"),
            location="local",
        )
        for role in roles
    )
    digests = {
        role: (
            "a" * 64
            if role == "recall.distill.answer_generator"
            else "b" * 64
            if role == "recall.distill.utility_judge"
            else f"{index + 1:064x}"
        )
        for index, role in enumerate(roles)
    }
    monkeypatch.setattr(ollama, "runtime_generation_routes", lambda _roles: routes)
    monkeypatch.setattr(
        ollama, "runtime_generation_route_fingerprints", lambda _routes: digests
    )
    label_path = store.distillation_dir(tmp_path) / "label-ledger.jsonl"
    features = distill.build_fast_features(
        query_chargram_coverage=1, candidate_chargram_precision=1
    )
    rally = {
        "rally_id": "rally-1",
        "host": "test-host",
        "session_id_sha256": "s" * 64,
        "session_cluster_id": "session-1",
        "as_of": "2026-01-01T00:00:00Z",
        "query_sha256": "query",
        "context_refs": [],
        "actual_answer_refs": [{"semantic_sha256": "answer"}],
        "exposure_receipts": [
            _record_counterfactual_exposure(
                tmp_path,
                host="test-host",
                session_id_sha256="s" * 64,
                query_sha256="query",
            )
        ],
    }
    snapshots = {
        "rally-1": {
            "as_of": rally["as_of"],
            "snapshot_sha256": "e" * 64,
            "feature_revision": distill.TEXT_FEATURE_REVISION,
            "candidates": [
                {"candidate_id": 7, "text_sha256": "text", "features": features}
            ],
        }
    }
    distill._ensure_split_plan(
        tmp_path,
        [rally],
        raw_watermark="a" * 64,
        model_cohort_sha256="b" * 64,
    )
    result = distill._run_counterfactual_block(
        execute=True,
        root=tmp_path,
        config=distill.DistillationConfig(enabled=True, max_input_bytes=4_096),
        counterfactual=Counterfactual(),
        snapshots=snapshots,
        rally_by_id={"rally-1": rally},
        texts={"query": "question", "answer": "answer", "text": "candidate"},
        label_path=label_path,
        label_rows=[],
    )

    assert result.written == 1
    labels = store.read_chain(label_path)
    assert labels[0]["candidate_id"] == 7
    assert labels[0]["assignment"]["revision"] == distill.ASSIGNMENT_REVISION
    assert labels[0]["assignment_revision"] == distill.ASSIGNMENT_REVISION
    assert labels[0]["identity_revision"] == "local-blind-counterfactual-v1"
    monkeypatch.setattr(
        distill,
        "_materialization_rallies",
        lambda _root, _supplied: {"rally-1": rally},
    )
    monkeypatch.setattr(
        distill,
        "_materialization_snapshots",
        lambda _root, _supplied: snapshots,
    )
    materialized = distill.materialize_training_rows(tmp_path)["rows"]
    assert materialized[0]["assignment_revision"] == distill.ASSIGNMENT_REVISION
    assert materialized[0]["identity_revision"] == "local-blind-counterfactual-v1"
    assert distill._materialized_row_integrity(materialized[0]) is True
    assert distill._sealed_counterfactual_exposure_binding(
        tmp_path, materialized[0], labels[0], rally
    ) is True
    assert distill._configured_local_route_binding(materialized[0]) is True
    assert distill._materialized_row_integrity(materialized[0], root=tmp_path) is True
    missing = {**materialized[0], "counterfactual_ref": "c" * 64}
    assert distill._materialized_row_integrity(missing, root=tmp_path) is False
    forged_label = store.append_chain(
        label_path,
        {
            key: value
            for key, value in labels[0].items()
            if key not in {"schema", "namespace", "previous_sha256", "record_sha256"}
        }
        | {"group_id": "evil-group", "split_plan_id": "f" * 64},
    )
    forged_row = next(
        row
        for row in distill.materialize_training_rows(tmp_path)["rows"]
        if row["label_record_sha256"] == forged_label["record_sha256"]
    )
    assert distill._materialized_row_integrity(forged_row, root=tmp_path) is False
    wrong_candidate_row = {**materialized[0], "candidate_id": "missing"}
    wrong_candidate_label = {**labels[0], "candidate_id": "missing"}
    assert distill._sealed_counterfactual_exposure_binding(
        tmp_path, wrong_candidate_row, wrong_candidate_label, rally
    ) is False
    empty_receipt = _record_counterfactual_exposure(
        tmp_path,
        host="test-host",
        session_id_sha256="s" * 64,
        query_sha256="query",
        candidate_ids=[],
    )
    empty_row = {
        **materialized[0],
        "counterfactual_ref": empty_receipt["exposure_artifact_id"],
    }
    empty_label = {
        **labels[0],
        "exposure_artifact_id": empty_row["counterfactual_ref"],
    }
    empty_rally = {**rally, "exposure_receipts": [empty_receipt]}
    assert distill._sealed_counterfactual_exposure_binding(
        tmp_path, empty_row, empty_label, empty_rally
    ) is False
    bad_assignment = {
        **materialized[0],
        "assignment_authority": {
            **materialized[0]["assignment_authority"],
            "kind": "forged",
        },
    }
    assert distill._materialized_row_integrity(bad_assignment) is False
    exposure_path = (
        store.distillation_dir(tmp_path)
        / "exposures"
        / f"{materialized[0]['counterfactual_ref']}.json"
    )
    tampered = json.loads(exposure_path.read_text(encoding="utf-8"))
    tampered["candidate_ids"] = ["evil"]
    exposure_path.write_text(json.dumps(tampered), encoding="utf-8")
    assert distill._materialized_row_integrity(materialized[0], root=tmp_path) is False
