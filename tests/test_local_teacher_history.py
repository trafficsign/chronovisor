"""Local collection must retain and verify the complete prior teacher ledger."""

from __future__ import annotations

from pathlib import Path

import pytest

from chronovisor.recall import recall_distillation as distill
from chronovisor.recall import recall_distillation_store as store


def _history(root: Path) -> Path:
    path = store.distillation_dir(root) / "label-ledger.jsonl"
    store.append_chain_batch(path, [
        {"kind": "teacher-label", "profile": distill.NATIVE_LUNA_SINGLE_PROFILE,
         "teacher_profile": distill.NATIVE_LUNA_SINGLE_PROFILE},
        {"kind": "teacher-label", "profile": distill.LOCAL_SINGLE_PROFILE,
         "teacher_profile": distill.LOCAL_SINGLE_PROFILE, "source_commit": "b" * 40},
        {"kind": "counterfactual-label", "verdict": "uncertain"},
    ])
    return path


def _repair(root: Path, path: Path, rows: list[dict[str, object]]) -> None:
    distill._repair_r4_local_receipts(
        root=root, config=distill.DistillationConfig(teacher_profile=distill.LOCAL_SINGLE_PROFILE),
        workset=object(), tasks={}, label_path=path, label_rows=rows,
    )


def test_local_receipt_repair_accepts_exact_subset_of_verified_full_history(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(distill.runtime_config, "runtime_identity", lambda: {"commit_id": "a" * 40})
    path = _history(tmp_path)
    before = path.read_bytes()
    rows = distill._local_single_label_rows(path)
    assert len(rows) == 2
    _repair(tmp_path, path, rows)
    assert path.read_bytes() == before
    assert len(store.read_chain(path)) == 3


@pytest.mark.parametrize("change", ["missing", "altered", "extra"])
def test_local_receipt_repair_rejects_changed_subset(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, change: str,
) -> None:
    monkeypatch.setattr(distill.runtime_config, "runtime_identity", lambda: {"commit_id": "a" * 40})
    path = _history(tmp_path)
    before = path.read_bytes()
    rows = distill._local_single_label_rows(path)
    if change == "missing":
        rows.pop()
    elif change == "altered":
        rows[0]["verdict"] = "relevant"
    else:
        rows.insert(0, store.read_chain(path)[0])
    with pytest.raises(distill.DistillationError, match="ledger identity changed"):
        _repair(tmp_path, path, rows)
    assert path.read_bytes() == before


def test_local_receipt_repair_rejects_full_ledger_advance_during_repair(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(distill.runtime_config, "runtime_identity", lambda: {"commit_id": "a" * 40})
    path = _history(tmp_path)
    rows = distill._local_single_label_rows(path)
    monkeypatch.setattr(
        distill, "_ensure_r4_local_receipts",
        lambda **_kwargs: store.append_chain(path, {"kind": "teacher-label", "profile": distill.NATIVE_LUNA_SINGLE_PROFILE}),
    )
    with pytest.raises(distill.DistillationError, match="ledger changed"):
        _repair(tmp_path, path, rows)
