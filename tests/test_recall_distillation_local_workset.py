from __future__ import annotations

from pathlib import Path

import pytest

from chronovisor.recall import recall_distillation as distill
from chronovisor.recall import recall_distillation_store as store
from chronovisor.recall.recall_distillation_workset import DistillationWorkset


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
                "generator_route_identity": {"role": "generator"},
                "judge_route_identity": {"role": "judge"},
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
                "judge_route_identity": {"role": "judge"},
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
            }
        },
        texts={"query": "question", "answer": "answer", "text": "candidate"},
        label_path=store.distillation_dir(tmp_path) / "label-ledger.jsonl",
        label_rows=[],
    )

    assert result.written == 0
    assert result.deferred is True
