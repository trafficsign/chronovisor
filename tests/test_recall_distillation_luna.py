from __future__ import annotations

import json
from pathlib import Path

import pytest

from chronovisor.recall.recall_distillation_luna import (
    NATIVE_OUTPUT_SCHEMA,
    NATIVE_OUTPUT_SCHEMA_SHA256,
    NATIVE_TEACHER_PROMPT,
    NATIVE_TEACHER_PROMPT_SHA256,
    LunaBulkBundle,
    LunaBulkError,
    ResumeMismatch,
    bundle_status,
    canonical_native_response_bytes,
    export_gate_rows,
    export_labels,
    finalize_bundle,
    native_output_sha256,
    next_batch,
    open_bundle,
    submit_batch,
)


def _rows(count: int = 5) -> list[dict[str, object]]:
    return [
        {
            "rally_id": f"rally-{index // 2}",
            "candidate_id": f"candidate-{index}",
            "as_of_us": 1_000 + index,
            "input": {
                "query": f"query {index}",
                "context": [f"context {index}"],
                "candidate": {"text": f"candidate text {index}"},
            },
            "split": "train" if index < 3 else "validation",
            "group_id": f"group-{index // 2}",
        }
        for index in range(count)
    ]


def _label(
    request_row: dict[str, object], *, verdict: str, assignment: str, run: str
) -> dict[str, object]:
    task_id = str(request_row["task_id"])
    label: dict[str, object] = {
        "task_id": task_id,
        "input_sha256": request_row["input_sha256"],
        "teacher_profile": "native-luna-single-v1",
        "native_luna": True,
        "verdict": verdict,
        "confidence": 0.9,
        "reason_code": "direct_match" if verdict == "relevant" else "not_relevant",
        "rationale": None,
    }
    label["native_receipt"] = {
        "task_id": task_id,
        "output_sha256": native_output_sha256(label),
        "assignment_id": assignment,
        "run_id": run,
    }
    return label


def test_bundle_freezes_snapshot_and_resumes_per_shard(tmp_path: Path) -> None:
    bundle = LunaBulkBundle.create(
        tmp_path / "bundle",
        _rows(),
        frozen_as_of_us=2_000,
        shard_count=2,
        batch_size=2,
    )

    status = bundle.status()
    assert status["status"] == "running"
    assert status["completed"] == 0
    assert status["record_count"] == 5
    request = next_batch(bundle.root, 0)
    assert request["teacher_profile"] == "native-luna-single-v1"
    assert request["teacher_contract"]["runtime_role"] == "luna_worker"
    assert request["prompt"] == NATIVE_TEACHER_PROMPT
    assert request["prompt_sha256"] == NATIVE_TEACHER_PROMPT_SHA256
    assert request["prompt_template_sha256"] == NATIVE_TEACHER_PROMPT_SHA256
    assert request["output_schema"] == NATIVE_OUTPUT_SCHEMA
    assert request["output_schema_sha256"] == NATIVE_OUTPUT_SCHEMA_SHA256
    assert request["cursor"] == 0
    assert len(request["records"]) == 2

    label = _label(
        request["records"][0],
        verdict="relevant",
        assignment="assignment-0",
        run="run-0",
    )
    submit_batch(bundle.root, 0, cursor=0, records=[label])
    assert bundle.next_batch(0)["cursor"] == 1
    with pytest.raises(ResumeMismatch, match="cursor"):
        bundle.submit(0, cursor=0, records=[label])


def test_observers_never_rewrite_another_shards_checkpoint(tmp_path: Path) -> None:
    first = LunaBulkBundle.create(tmp_path / "bundle", _rows(4), shard_count=2)
    checkpoints = sorted(first.root.glob("shards/*/checkpoint.json"))
    before = {path: path.read_bytes() for path in checkpoints}
    second = LunaBulkBundle.open(first.root)
    second.status()
    second.next_batch(1, limit=1)
    assert {path: path.read_bytes() for path in checkpoints} == before

    request = second.next_batch(1, limit=1)
    second.submit(1, cursor=0, records=[_label(
        request["records"][0], verdict="relevant", assignment="second", run="second",
    )])
    other_before = checkpoints[1].read_bytes()
    request = first.next_batch(0, limit=1)
    first.submit(0, cursor=0, records=[_label(
        request["records"][0], verdict="irrelevant", assignment="first", run="first",
    )])
    assert checkpoints[1].read_bytes() == other_before
    assert json.loads(checkpoints[0].read_text())["cursor"] == 1


def test_writer_recovers_a_checkpoint_from_a_batch_committed_before_a_crash(tmp_path: Path) -> None:
    bundle = LunaBulkBundle.create(tmp_path / "bundle", _rows(1), shard_count=1)
    checkpoint = bundle.root / "shards/00/checkpoint.json"
    before = checkpoint.read_bytes()
    request = bundle.next_batch(0, limit=1)
    bundle.submit(0, cursor=0, records=[_label(
        request["records"][0], verdict="relevant", assignment="writer", run="writer",
    )])
    checkpoint.write_bytes(before)
    resumed = LunaBulkBundle.open(bundle.root)
    assert checkpoint.read_bytes() == before
    resumed.refresh_checkpoint(0)
    assert json.loads(checkpoint.read_text())["cursor"] == 1
    assert resumed.next_batch(0, limit=1)["records"] == []


def test_bundle_rejects_non_native_or_wrong_profile_output(tmp_path: Path) -> None:
    bundle = LunaBulkBundle.create(tmp_path / "bundle", _rows(1), shard_count=1)
    request = bundle.next_batch(0)
    label = _label(
        request["records"][0], verdict="relevant", assignment="assignment", run="run"
    )
    label["native_luna"] = False
    with pytest.raises(LunaBulkError, match="native Luna"):
        bundle.submit(0, cursor=0, records=[label])
    label["native_luna"] = True
    label["teacher_profile"] = "other-teacher"
    with pytest.raises(LunaBulkError, match="profile"):
        bundle.submit(0, cursor=0, records=[label])


def test_native_output_hash_excludes_receipt_to_avoid_circular_binding() -> None:
    label: dict[str, object] = {
        "task_id": "a" * 64,
        "teacher_profile": "native-luna-single-v1",
        "native_luna": True,
        "verdict": "relevant",
        "confidence": 0.9,
        "reason_code": "direct_match",
        "rationale": None,
    }
    digest = native_output_sha256(label)
    label["native_receipt"] = {
        "task_id": label["task_id"],
        "output_sha256": digest,
        "assignment_id": "assignment",
        "run_id": "run",
    }
    assert native_output_sha256(label) == digest
    assert b"native_receipt" not in canonical_native_response_bytes(label)


def test_native_output_hash_excludes_bound_source_provenance() -> None:
    label: dict[str, object] = {
        "task_id": "a" * 64,
        "input_sha256": "b" * 64,
        "teacher_profile": "native-luna-single-v1",
        "native_luna": True,
        "verdict": "uncertain",
        "confidence": 0.5,
        "reason_code": "insufficient_evidence",
        "rationale": None,
    }
    digest = native_output_sha256(label)
    label.update(
        {
            "rally_id": "rally-1",
            "candidate_id": "candidate-1",
            "as_of_us": 1_000,
            "split": "validation",
            "group_id": "group-1",
            "assignment": {"revision": "native-luna-bulk-v1"},
        }
    )
    assert native_output_sha256(label) == digest


def test_receipt_hash_must_match_canonical_native_response(tmp_path: Path) -> None:
    bundle = LunaBulkBundle.create(tmp_path / "bundle", _rows(1), shard_count=1)
    request = bundle.next_batch(0)
    label = _label(
        request["records"][0], verdict="relevant", assignment="assignment", run="run"
    )
    label["native_receipt"] = {
        **label["native_receipt"],
        "output_sha256": "a" * 64,
    }
    with pytest.raises(LunaBulkError, match="does not match response"):
        bundle.submit(0, cursor=0, records=[label])


def test_rationale_is_always_null_in_native_output(tmp_path: Path) -> None:
    bundle = LunaBulkBundle.create(tmp_path / "bundle", _rows(1), shard_count=1)
    request = bundle.next_batch(0)
    label = _label(
        request["records"][0], verdict="uncertain", assignment="assignment", run="run"
    )
    label["rationale"] = "free form"
    label["native_receipt"] = {
        **label["native_receipt"],
        "output_sha256": native_output_sha256(label),
    }
    with pytest.raises(LunaBulkError, match="rationale must be null"):
        bundle.submit(0, cursor=0, records=[label])


def test_blind_repeat_requires_two_independent_native_assignments(
    tmp_path: Path,
) -> None:
    rows = []
    for order in ("a_first", "b_first"):
        rows.append(
            {
                "rally_id": "rally-repeat",
                "candidate_id": "candidate-repeat",
                "as_of_us": 1_000,
                "input": {"query": "same point in time"},
                "repeat_pair_id": "pair-1",
                "fixed_repeat": True,
                "order_swap": True,
                "blind_order": order,
            }
        )
    bundle = LunaBulkBundle.create(tmp_path / "bundle", rows, shard_count=2)
    requests = [bundle.next_batch(shard) for shard in range(2)]
    assert {bundle._source_rows(shard)[0]["blind_order"] for shard in range(2)} == {
        "a_first",
        "b_first",
    }
    assert all(request["blind_metadata_hidden"] is True for request in requests)
    assert all("blind_order" not in request["records"][0] for request in requests)
    assert all("group_id" not in request["records"][0] for request in requests)
    labels = [
        _label(
            requests[0]["records"][0],
            verdict="relevant",
            assignment="same",
            run="run-a",
        ),
        _label(
            requests[1]["records"][0],
            verdict="relevant",
            assignment="same",
            run="run-a",
        ),
    ]
    bundle.submit(0, cursor=0, records=[labels[0]])
    bundle.submit(1, cursor=0, records=[labels[1]])
    with pytest.raises(LunaBulkError, match="separate native assignments"):
        bundle.finalize()

    labels[0]["native_receipt"] = {
        **labels[0]["native_receipt"],
        "assignment_id": "assignment-a",
    }
    labels[1]["native_receipt"] = {
        **labels[1]["native_receipt"],
        "assignment_id": "assignment-b",
        "run_id": "run-b",
    }
    # A batch is immutable, so a corrected native response starts a fresh bundle.
    corrected = LunaBulkBundle.create(tmp_path / "corrected", rows, shard_count=2)
    corrected.submit(0, cursor=0, records=[labels[0]])
    corrected.submit(1, cursor=0, records=[labels[1]])
    completion = finalize_bundle(corrected.root)
    assert completion["blind_repeat"] == {"pairs": 1, "stable": 1, "unstable": 0}
    assert (
        finalize_bundle(corrected.root)["labels_sha256"] == completion["labels_sha256"]
    )
    assert corrected.gate_rows()[0]["teacher_only"] is True
    assert corrected.gate_rows()[0]["identity_revision"] == (
        "native-luna-single-v1-fixed-identity-v2"
    )


def test_open_rechecks_frozen_hash_and_atomic_outputs(tmp_path: Path) -> None:
    root = tmp_path / "bundle"
    LunaBulkBundle.create(root, _rows(1), shard_count=1)
    source = root / "snapshot" / "source.jsonl"
    source.write_bytes(source.read_bytes().replace(b"candidate-0", b"changed-00"))
    with pytest.raises(LunaBulkError, match="hash"):
        open_bundle(root)


def test_open_rechecks_snapshot_cardinality(tmp_path: Path) -> None:
    root = tmp_path / "bundle"
    LunaBulkBundle.create(root, _rows(2), shard_count=1)
    snapshot = root / "snapshot" / "manifest.json"
    payload = json.loads(snapshot.read_text())
    payload["candidate_count"] = 999
    snapshot.write_text(json.dumps(payload) + "\n")
    with pytest.raises(ResumeMismatch, match="cardinality"):
        open_bundle(root)


def test_status_wrapper_reports_class_ratio(tmp_path: Path) -> None:
    bundle = LunaBulkBundle.create(tmp_path / "bundle", _rows(2), shard_count=1)
    request = bundle.next_batch(0)
    labels = [
        _label(request["records"][0], verdict="relevant", assignment="a", run="r1"),
        _label(request["records"][1], verdict="irrelevant", assignment="b", run="r2"),
    ]
    submit_batch(bundle.root, 0, cursor=0, records=labels)
    assert bundle_status(bundle.root)["class_ratio"] == {
        "relevant": 1,
        "irrelevant": 1,
        "labeled": 2,
    }
    assert {row["verdict"] for row in export_labels(bundle.root)} == {
        "relevant",
        "irrelevant",
    }
    assert all(row["teacher_only"] is True for row in export_gate_rows(bundle.root))
