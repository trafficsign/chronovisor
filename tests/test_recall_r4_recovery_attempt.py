"""R4 recovery keeps label-generation attempts distinct from settlement retries."""

from pathlib import Path
from typing import Any

import pytest

from chronovisor.core import canonical_json
from chronovisor.recall import recall_distillation as distill
from chronovisor.recall import recall_distillation_store as store
from chronovisor.recall.recall_distillation_workset import DistillationWorkset

SOURCE = {
    "source_commit": "a" * 40,
    "source_tree_sha256": "b" * 64,
    "source_ox_identity_sha256": "c" * 64,
}


def _reclaimed_label(
    root: Path, profile: str, *, label_patch: dict[str, Any] | None = None,
) -> tuple[DistillationWorkset, Any, dict[str, Any]]:
    distill.bootstrap_r4_distillation_root_authority(root)
    now = [1000.0]
    queue = DistillationWorkset(
        distill._local_teacher_workset_path(root, profile), clock=lambda: now[0],
    )
    rally, candidate = {"rally_id": "synthetic-rally"}, {"candidate_id": "synthetic-candidate"}
    assignment = distill.teacher_assignment(rally["rally_id"], candidate["candidate_id"])
    if profile == distill.LOCAL_SINGLE_PROFILE:
        assignment = {
            "revision": "local-single-assignment-v1", "owner": distill.TEACHER_ROLES[0],
            "probe_revision": distill.PROBE_REVISION, "probe": False,
            "routes": [distill.TEACHER_ROLES[0]],
        }
    route = str(assignment["owner"])
    digest = canonical_json.canonical_json_sha256_strict({"rally": rally, "candidate": candidate})
    work_id = distill._local_work_id(digest, route, profile)
    task = {"rally": rally, "candidate": candidate, "assignment": assignment, "route": route}
    identity = {"role": route, "provider": "ollama", "model": "synthetic-local", "location": "local"}
    queue.advance([{
        "work_id": work_id, "kind": f"local-teacher:{route}",
        "payload_ref": "candidate-snapshot:synthetic-rally:synthetic-candidate",
        "payload_digest": digest, "priority": 0, "temporal_split": {"split": "train"},
        "provenance": {"route": route},
    }], {"source": 1})
    original = queue.claim(f"local-teacher:{route}", 1, route, 1)[0]
    label = store.append_chain(store.distillation_dir(root) / "label-ledger.jsonl", {
        "kind": "teacher-label", "status": "completed", "teacher_profile": profile,
        "profile": profile,
        "work_id": work_id, "attempt": original.attempt, "payload_digest": digest,
        "rally_id": rally["rally_id"], "candidate_id": candidate["candidate_id"],
        "route": route, "captured_at": "2026-10-20T00:00:00Z", "assignment": assignment,
        "route_identity": identity, **SOURCE, **(label_patch or {}),
    })
    # Simulate interruption after durable append and before workset completion.
    now[0] += 2
    reclaimed = queue.claim(f"local-teacher:{route}", 1, route, 60)[0]
    assert original.attempt == 1 and reclaimed.attempt == 2
    return queue, reclaimed, {
        "work_id": work_id, "attempt": label["attempt"], "task": task,
        "label": label, "route_identity": identity,
    }


@pytest.mark.parametrize("profile", [distill.LOCAL_TRIAD_PROFILE, distill.LOCAL_SINGLE_PROFILE])
def test_reclaimed_label_publishes_r4_without_rewriting_or_regenerating(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, profile: str,
) -> None:
    monkeypatch.setattr(distill, "ox_alpha_source_binding", lambda: dict(SOURCE))
    queue, claim, entry = _reclaimed_label(tmp_path, profile)
    label_path = store.distillation_dir(tmp_path) / "label-ledger.jsonl"
    before = label_path.read_bytes()
    assert distill._reconcile_local_teacher_claims(
        workset=queue, claims=[claim], label_rows=[entry["label"]],
        tasks={claim.work_id: entry["task"]}, route=entry["task"]["route"],
        root=tmp_path, config=distill.DistillationConfig(teacher_profile=profile),
    ) == []
    assert label_path.read_bytes() == before
    assert queue.completion_identities([claim.work_id])[claim.work_id]["attempt"] == 2
    paths = list((store.distillation_dir(tmp_path) / "r4-receipts").rglob("*.json"))
    assert len(paths) == 1
    receipt = store.read_sealed(paths[0], schema=distill.R4_RECEIPT_SCHEMA)
    assert receipt["attempt"] == 1
    assert receipt["label_record_sha256"] == entry["label"]["record_sha256"]
    assert receipt["workset_receipt"]["head_sha256"] == queue.recent_transition_receipts(1)[0]["receipt_sha256"]
    assert queue.audit_transition_receipts()["status"] == "verified"


def test_previously_settled_recovery_can_publish_its_missing_receipt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(distill, "ox_alpha_source_binding", lambda: dict(SOURCE))
    monkeypatch.setattr(distill.runtime_config, "runtime_identity", lambda: {"commit_id": SOURCE["source_commit"]})
    profile = distill.LOCAL_SINGLE_PROFILE
    queue, claim, entry = _reclaimed_label(tmp_path, profile)
    label = entry["label"]
    queue.commit([claim], [{
        "status": "completed", "completion_ref": f"label-ledger:{label['record_sha256']}",
        "completion_digest": label["record_sha256"],
    }])
    label_path = store.distillation_dir(tmp_path) / "label-ledger.jsonl"
    before, transition = label_path.read_bytes(), queue.recent_transition_receipts(1)
    distill._repair_r4_local_receipts(
        root=tmp_path, config=distill.DistillationConfig(teacher_profile=profile),
        workset=queue, tasks={claim.work_id: entry["task"]},
        label_path=label_path, label_rows=store.read_chain(label_path),
    )
    assert len(list((store.distillation_dir(tmp_path) / "r4-receipts").rglob("*.json"))) == 1
    assert label_path.read_bytes() == before
    assert queue.recent_transition_receipts(1) == transition
    assert queue.completion_identities([claim.work_id])[claim.work_id]["attempt"] == 2

    receipts = list((store.distillation_dir(tmp_path) / "r4-receipts").rglob("*.json"))
    receipt_bytes = receipts[0].read_bytes()
    distill._repair_r4_local_receipts(
        root=tmp_path, config=distill.DistillationConfig(teacher_profile=profile),
        workset=queue, tasks={claim.work_id: entry["task"]},
        label_path=label_path, label_rows=store.read_chain(label_path),
    )
    assert receipts[0].read_bytes() == receipt_bytes
    assert queue.recent_transition_receipts(1) == transition


@pytest.mark.parametrize("patch", [
    {"payload_digest": "f" * 64}, {"attempt": 3}, {"attempt": True},
    {"work_id": "local-teacher-" + "d" * 64}, {"payload_digest": None},
])
def test_recovery_does_not_accept_foreign_or_future_label_provenance(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, patch: dict[str, Any],
) -> None:
    monkeypatch.setattr(distill, "ox_alpha_source_binding", lambda: dict(SOURCE))
    profile = distill.LOCAL_SINGLE_PROFILE
    queue, claim, entry = _reclaimed_label(tmp_path, profile, label_patch=patch)
    label = entry["label"]
    queue.commit([claim], [{
        "status": "completed", "completion_ref": f"label-ledger:{label['record_sha256']}",
        "completion_digest": label["record_sha256"],
    }])
    assert not distill._write_r4_local_receipts(
        root=tmp_path, config=distill.DistillationConfig(teacher_profile=profile),
        workset=queue, entries=[entry],
    )
    assert not list((store.distillation_dir(tmp_path) / "r4-receipts").rglob("*.json"))


@pytest.mark.parametrize("patch", [
    {"payload_digest": "f" * 64}, {"attempt": 3}, {"attempt": True},
    {"work_id": "local-teacher-" + "d" * 64}, {"payload_digest": None},
])
def test_reconciliation_rejects_conflicting_label_before_settlement(
    tmp_path: Path, patch: dict[str, Any],
) -> None:
    profile = distill.LOCAL_SINGLE_PROFILE
    queue, claim, entry = _reclaimed_label(tmp_path, profile, label_patch=patch)
    transition = queue.recent_transition_receipts(1)
    with pytest.raises(distill.DistillationError, match="recovered label identity"):
        distill._reconcile_local_teacher_claims(
            workset=queue, claims=[claim], label_rows=[entry["label"]],
            tasks={claim.work_id: entry["task"]}, route=entry["task"]["route"],
            root=tmp_path, config=distill.DistillationConfig(teacher_profile=profile),
        )
    assert queue.completion_identities([claim.work_id]) == {}
    assert queue.recent_transition_receipts(1) == transition


def test_recovered_label_keeps_the_current_source_gate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    profile = distill.LOCAL_SINGLE_PROFILE
    queue, claim, entry = _reclaimed_label(tmp_path, profile)
    label = entry["label"]
    queue.commit([claim], [{
        "status": "completed", "completion_ref": f"label-ledger:{label['record_sha256']}",
        "completion_digest": label["record_sha256"],
    }])
    monkeypatch.setattr(distill, "ox_alpha_source_binding", lambda: {
        **SOURCE, "source_commit": "e" * 40,
    })
    label_path = store.distillation_dir(tmp_path) / "label-ledger.jsonl"
    before, transition = label_path.read_bytes(), queue.recent_transition_receipts(1)
    assert not distill._write_r4_local_receipts(
        root=tmp_path, config=distill.DistillationConfig(teacher_profile=profile),
        workset=queue, entries=[entry],
    )
    assert not list((store.distillation_dir(tmp_path) / "r4-receipts").rglob("*.json"))
    assert label_path.read_bytes() == before
    assert queue.recent_transition_receipts(1) == transition


@pytest.mark.parametrize("key,value", [
    ("source_commit", "e" * 40), ("source_tree_sha256", "f" * 64),
])
def test_reconciliation_rejects_source_drift_before_settlement(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, key: str, value: str,
) -> None:
    profile = distill.LOCAL_SINGLE_PROFILE
    queue, claim, entry = _reclaimed_label(tmp_path, profile)
    monkeypatch.setattr(distill, "ox_alpha_source_binding", lambda: {**SOURCE, key: value})
    transition = queue.recent_transition_receipts(1)
    with pytest.raises(distill.DistillationError, match="recovered label source"):
        distill._reconcile_local_teacher_claims(
            workset=queue, claims=[claim], label_rows=[entry["label"]],
            tasks={claim.work_id: entry["task"]}, route=entry["task"]["route"],
            root=tmp_path, config=distill.DistillationConfig(teacher_profile=profile),
        )
    assert queue.completion_identities([claim.work_id]) == {}
    assert queue.recent_transition_receipts(1) == transition
