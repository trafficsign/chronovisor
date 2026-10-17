from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pytest

from chronovisor.recall import recall_distillation as distill
from chronovisor.recall import recall_distillation_catalog as catalog
from chronovisor.recall import recall_distillation_ledger as ledger
from chronovisor.recall import recall_distillation_materialization as materialization
from chronovisor.recall import recall_distillation_native as native
from chronovisor.recall import recall_distillation_store as store
from chronovisor.recall import recall_distillation_validation as validation


class _SelectionView:
    def __init__(self, root: Path, path: Path) -> None:
        self.root = root
        self.path = path
        self.metadata: dict[str, object] = {}
        self.state = ledger.file_state(path)
        self.connection = ledger.connect(path, readonly=True)


@pytest.fixture(autouse=True)
def _clear_views() -> None:
    materialization._VIEWS.clear()
    yield
    materialization._VIEWS.clear()


def _selection_rows() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    # Four old and two recent unlocked rows, followed by three locked rows.
    for index, rally_id in enumerate(("old-a", "old-b", "old-c", "old-d"), 1):
        rows.append(
            {
                "record_sha256": f"record-{rally_id}",
                "rally_id": rally_id,
                "as_of": str(index),
                "epoch": float(index),
                "profile": distill.NATIVE_LUNA_SINGLE_PROFILE,
                "label_split_plan_id": "plan",
                "locked_test_read_only": False,
            }
        )
    for index, rally_id in enumerate(("recent-a", "recent-b"), 5):
        rows.append(
            {
                "record_sha256": f"record-{rally_id}",
                "rally_id": rally_id,
                "as_of": str(index),
                "epoch": float(index),
                "profile": distill.NATIVE_LUNA_SINGLE_PROFILE,
                "label_split_plan_id": "plan",
                "locked_test_read_only": False,
            }
        )
    for index, rally_id in enumerate(("locked-a", "locked-b", "locked-c"), 0):
        rows.append(
            {
                "record_sha256": f"record-{rally_id}",
                "rally_id": rally_id,
                "as_of": str(index),
                "epoch": float(index),
                "profile": distill.NATIVE_LUNA_SINGLE_PROFILE,
                "label_split_plan_id": "plan",
                "locked_test_read_only": True,
            }
        )
    return rows


def _selection_view(tmp_path: Path, rows: list[dict[str, object]]) -> _SelectionView:
    path = tmp_path / "materialization.sqlite3"
    with ledger.connect(path) as connection:
        materialization._schema(connection)
        connection.executemany(
            "INSERT INTO rows(sequence,record_sha256,rally_id,receipt_sha256,profile,"
            "profile_contract_id,label_plan,epoch,dirty,row_json,excluded_prior,"
            "excluded_unsupported) VALUES(?,?,?,?,?,?,?,?,0,?,0,0)",
            (
                (
                    index,
                    str(row["record_sha256"]),
                    str(row["rally_id"]),
                    "",
                    str(row["profile"]),
                    "contract",
                    str(row.get("label_split_plan_id") or "plan"),
                    float(row["epoch"]),
                    json.dumps(row, separators=(",", ":")),
                )
                for index, row in enumerate(rows, 1)
            ),
        )
    return _SelectionView(tmp_path, path)


def _stub_build_dependencies(
    monkeypatch: pytest.MonkeyPatch,
    *,
    materialize_calls: list[set[str]],
) -> None:
    monkeypatch.setattr(
        materialization,
        "_contract_dependencies",
        lambda _root, _labels: {
            "code": "code",
            "native": "native",
            "ox": "ox",
            "files": {},
        },
    )
    monkeypatch.setattr(
        catalog,
        "rally_changes",
        lambda _root, after_revision=0: {
            "catalog_lineage": "lineage",
            "revision": int(after_revision),
            "rally_ids": [],
        },
    )
    monkeypatch.setattr(
        catalog,
        "candidate_index_state",
        lambda _root: {"record_count": 0, "ledger_ino": None, "head_sha256": ""},
    )
    monkeypatch.setattr(
        catalog,
        "candidate_prefix_matches",
        lambda _root, records=0, head_sha256="": records == 0 and head_sha256 == "",
    )
    monkeypatch.setattr(catalog, "candidate_rally_ids", lambda _root, after_seq=0: [])
    monkeypatch.setattr(materialization, "_source_state", lambda _root, _contracts: ("source",))

    def fake_materialize(
        connection: object,
        _root: Path,
        _labels: object,
        rally_ids: list[str] | tuple[str, ...],
        _ox: str,
    ) -> None:
        ids = {str(value) for value in rally_ids}
        materialize_calls.append(ids)
        run = len(materialize_calls)
        for rally_id in ids:
            rows = connection.execute(
                "SELECT sequence,record_sha256,rally_id FROM rows WHERE rally_id=? AND dirty=1",
                (rally_id,),
            ).fetchall()
            for sequence, record_sha256, value in rows:
                projected = {
                    "record_sha256": str(record_sha256),
                    "rally_id": str(value),
                    "as_of": "1",
                    "locked_test_read_only": False,
                    "materialize_run": run,
                }
                connection.execute(
                    "UPDATE rows SET epoch=1,row_json=?,dirty=0 WHERE sequence=?",
                    (json.dumps(projected, separators=(",", ":")), sequence),
                )

    monkeypatch.setattr(materialization, "_materialize_batch", fake_materialize)


def _append_label(
    root: Path,
    rally_id: str,
    candidate_id: str,
    *,
    receipt_sha256: str = "",
) -> dict[str, object]:
    path = store.distillation_dir(root) / store.LABEL_LEDGER_FILE
    return store.append_chain(
        path,
        {
            "kind": "teacher-label",
            "rally_id": rally_id,
            "candidate_id": candidate_id,
            "profile": distill.NATIVE_LUNA_SINGLE_PROFILE,
            "profile_contract_id": "contract",
            "split_plan_id": "plan",
            "provider_receipt_sha256": receipt_sha256,
        },
    )


def _canonical_projection_fixture(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> dict[str, object]:
    """Create a real sealed projection whose row can be source-bound."""

    label = _append_label(tmp_path, "rally-a", "candidate-a")
    calls: list[set[str]] = []
    _stub_build_dependencies(monkeypatch, materialize_calls=calls)
    monkeypatch.setattr(
        catalog,
        "rallies",
        lambda _root, rally_ids: [{"rally_id": rally_id} for rally_id in rally_ids],
    )
    record = str(label["record_sha256"])
    row: dict[str, object] = {
        "rally_id": "rally-a",
        "candidate_id": "candidate-a",
        "source": "teacher-label",
        "profile": distill.NATIVE_LUNA_SINGLE_PROFILE,
        "label_record_sha256": record,
        "split_plan_id": "plan",
        "split": "train",
        "locked_test_read_only": False,
        "locked_test_evidence_ref": "",
    }

    def write_projection(
        connection: object,
        _root: Path,
        _labels: object,
        rally_ids: list[str] | tuple[str, ...],
        _ox: str,
    ) -> None:
        for rally_id in rally_ids:
            entries = connection.execute(
                "SELECT sequence FROM rows WHERE rally_id=? AND dirty=1", (rally_id,)
            ).fetchall()
            for (sequence,) in entries:
                connection.execute(
                    "UPDATE rows SET epoch=1,row_json=?,dirty=0 WHERE sequence=?",
                    (json.dumps(row, separators=(",", ":")), sequence),
                )

    monkeypatch.setattr(materialization, "_materialize_batch", write_projection)
    initial = materialization._build(tmp_path)
    initial.close()
    return row


def test_build_reuses_clean_projection_without_rematerializing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _append_label(tmp_path, "rally-a", "candidate-a")
    calls: list[set[str]] = []
    _stub_build_dependencies(monkeypatch, materialize_calls=calls)

    first = materialization._build(tmp_path)
    assert calls == [{"rally-a"}]
    first.connection.close()

    second = materialization._build(tmp_path)
    assert calls == [{"rally-a"}]
    assert second.binding("not-present") is None
    second.connection.close()


def test_build_only_materializes_appended_rally_and_rebuilds_tampered_cache(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _append_label(tmp_path, "rally-a", "candidate-a")
    _append_label(tmp_path, "rally-b", "candidate-b")
    calls: list[set[str]] = []
    _stub_build_dependencies(monkeypatch, materialize_calls=calls)

    first = materialization._build(tmp_path)
    first.connection.close()
    assert calls == [{"rally-a", "rally-b"}]

    _append_label(tmp_path, "rally-c", "candidate-c")
    second = materialization._build(tmp_path)
    second.connection.close()
    assert calls[-1] == {"rally-c"}

    projection_path = materialization._path(tmp_path)
    with sqlite3.connect(projection_path) as connection:
        connection.execute(
            "UPDATE rows SET row_json=? WHERE rally_id='rally-a'",
            (json.dumps({"forged": True}, separators=(",", ":")),),
        )
    rebuilt = materialization._build(tmp_path)
    rebuilt.connection.close()
    assert calls[-1] == {"rally-a", "rally-b", "rally-c"}


def test_catalog_lineage_change_invalidates_only_changed_rally(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _append_label(tmp_path, "rally-a", "candidate-a")
    _append_label(tmp_path, "rally-b", "candidate-b")
    calls: list[set[str]] = []
    _stub_build_dependencies(monkeypatch, materialize_calls=calls)
    first = materialization._build(tmp_path)
    first.connection.close()
    with sqlite3.connect(materialization._path(tmp_path)) as connection:
        before = {row[0]: json.loads(row[1]) for row in connection.execute(
            "SELECT rally_id,row_json FROM rows"
        )}

    monkeypatch.setattr(
        catalog,
        "rally_changes",
        lambda _root, after_revision=0: {
            "catalog_lineage": "lineage",
            "revision": int(after_revision) + 1,
            "rally_ids": ["rally-a"],
        },
    )
    second = materialization._build(tmp_path)
    second.connection.close()
    assert calls[-1] == {"rally-a"}
    with sqlite3.connect(materialization._path(tmp_path)) as connection:
        after = {row[0]: json.loads(row[1]) for row in connection.execute(
            "SELECT rally_id,row_json FROM rows"
        )}
    assert after["rally-a"]["materialize_run"] == before["rally-a"]["materialize_run"] + 1
    assert after["rally-b"] == before["rally-b"]


def test_new_receipt_invalidates_label_that_was_missing_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _append_label(tmp_path, "rally-a", "candidate-a", receipt_sha256="receipt-a")
    calls: list[set[str]] = []
    _stub_build_dependencies(monkeypatch, materialize_calls=calls)
    first = materialization._build(tmp_path)
    first.connection.close()
    receipt_path = native._receipt_path(tmp_path)
    store.append_chain(receipt_path, {"kind": "native-receipt", "receipt_sha256": "receipt-a"})

    second = materialization._build(tmp_path)
    second.connection.close()
    assert calls[-1] == {"rally-a"}


def test_contract_file_change_invalidates_all_cached_rows(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _append_label(tmp_path, "rally-a", "candidate-a")
    _append_label(tmp_path, "rally-b", "candidate-b")
    calls: list[set[str]] = []
    _stub_build_dependencies(monkeypatch, materialize_calls=calls)
    first = materialization._build(tmp_path)
    first.connection.close()
    monkeypatch.setattr(
        materialization,
        "_contract_dependencies",
        lambda _root, _labels: {
            "code": "code",
            "native": "native",
            "ox": "ox",
            "files": {"contract.json": {"size_bytes": 2}},
        },
    )

    second = materialization._build(tmp_path)
    second.connection.close()
    assert calls[-1] == {"rally-a", "rally-b"}


def test_candidate_prefix_rewrite_same_count_invalidates_all_rows(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _append_label(tmp_path, "rally-a", "candidate-a")
    _append_label(tmp_path, "rally-b", "candidate-b")
    calls: list[set[str]] = []
    _stub_build_dependencies(monkeypatch, materialize_calls=calls)
    monkeypatch.setattr(
        catalog,
        "candidate_index_state",
        lambda _root: {"record_count": 1, "ledger_ino": 17, "head_sha256": "head"},
    )
    monkeypatch.setattr(catalog, "candidate_prefix_matches", lambda *_a, **_k: True)
    first = materialization._build(tmp_path)
    first.connection.close()
    monkeypatch.setattr(catalog, "candidate_prefix_matches", lambda *_a, **_k: False)

    second = materialization._build(tmp_path)
    second.connection.close()
    assert calls[-1] == {"rally-a", "rally-b"}


def test_late_exposure_arrival_and_update_dirty_its_rally(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _append_label(tmp_path, "rally-a", "candidate-a")
    calls: list[set[str]] = []
    _stub_build_dependencies(monkeypatch, materialize_calls=calls)
    first = materialization._build(tmp_path)
    first.connection.close()

    exposure = store.distillation_dir(tmp_path) / "exposures" / "exposure-a.json"
    with sqlite3.connect(materialization._path(tmp_path)) as connection:
        connection.execute(
            "INSERT INTO dependencies(path,rally_id,state) VALUES(?,?,?)",
            (str(exposure), "rally-a", "null"),
        )
    exposure.parent.mkdir(parents=True, exist_ok=True)
    exposure.write_text("first", encoding="utf-8")
    arrived = materialization._build(tmp_path)
    arrived.connection.close()
    assert calls[-1] == {"rally-a"}

    exposure.write_text("updated", encoding="utf-8")
    updated = materialization._build(tmp_path)
    updated.connection.close()
    assert calls[-1] == {"rally-a"}


def test_materialize_rejects_exposure_mutation_during_finalize(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _append_label(tmp_path, "rally-a", "candidate-a")
    calls: list[set[str]] = []
    _stub_build_dependencies(monkeypatch, materialize_calls=calls)
    first = materialization._build(tmp_path)
    first.connection.close()

    exposure = store.distillation_dir(tmp_path) / "exposures" / "exposure-a.json"
    exposure.parent.mkdir(parents=True, exist_ok=True)
    exposure.write_text("before", encoding="utf-8")
    projection_path = materialization._path(tmp_path)
    with sqlite3.connect(projection_path) as connection:
        connection.execute(
            "INSERT INTO dependencies(path,rally_id,state) VALUES(?,?,?)",
            (
                str(exposure),
                "rally-a",
                materialization._encoded(ledger.file_state(exposure)),
            ),
        )
        metadata = materialization._read_metadata(connection)
    # Keep the manually seeded dependency part of the sealed projection view.
    ledger.seal_checkpoint(
        projection_path,
        kind=materialization._VERSION,
        metadata=metadata,
    )
    monkeypatch.setattr(catalog, "rallies", lambda _root, _ids: [{"rally_id": "rally-a"}])

    def mutate_finalize(_root: Path, rows: list[dict[str, object]], _limit: int, **_kwargs: object) -> dict[str, object]:
        exposure.write_text("changed", encoding="utf-8")
        return {"rows": rows}

    monkeypatch.setattr(distill, "_finalize_materialized_training_rows", mutate_finalize)
    with pytest.raises(store.DistillationStoreError, match="canonical materialization projection changed"):
        materialization._materialize(tmp_path, limit=1)


def test_materialize_rejects_split_change_during_finalize(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _append_label(tmp_path, "rally-a", "candidate-a")
    calls: list[set[str]] = []
    _stub_build_dependencies(monkeypatch, materialize_calls=calls)
    first = materialization._build(tmp_path)
    first.connection.close()

    contexts = iter(
        (
            ("plan-a", {"rally-a": "train"}, {"plan-a"}),
            ("plan-b", {"rally-a": "test"}, {"plan-b"}),
        )
    )
    monkeypatch.setattr(
        distill,
        "_materialization_split_context",
        lambda _root, _plans: next(contexts),
    )
    monkeypatch.setattr(
        distill,
        "_finalize_materialized_training_rows",
        lambda _root, rows, _limit, **_kwargs: {"rows": rows},
    )
    with pytest.raises(store.DistillationStoreError, match="split changed during selection"):
        materialization._materialize(tmp_path, limit=1)


def test_materialize_selection_uses_registered_projection_for_source_binding(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    row = _canonical_projection_fixture(tmp_path, monkeypatch)
    root = tmp_path.resolve()
    fallback_calls: list[Path] = []

    def forbidden_fallback(root: Path) -> object:
        fallback_calls.append(root)
        raise AssertionError("selection must use the registered projection")

    monkeypatch.setattr(validation, "binding_inputs", forbidden_fallback)
    plan = {"artifact_id": "plan", "assignments": {"rally-a": "train"}}
    selected_views: list[materialization.Projection] = []

    def select(view: materialization.Projection, limit: int) -> list[dict[str, object]]:
        assert limit == 1
        selected_views.append(view)
        assert materialization._VIEWS[root] is view
        assert distill._authoritative_materialized_row_binding(tmp_path, row, plan)
        assert materialization.canonical_binding(
            tmp_path, str(row["label_record_sha256"])
        ) is not None
        return [row]

    monkeypatch.setattr(materialization, "_selection", select)
    monkeypatch.setattr(
        distill,
        "_finalize_materialized_training_rows",
        lambda _root, rows, _limit, **_kwargs: {"rows": rows},
    )

    artifact = materialization._materialize(tmp_path, limit=1)

    assert artifact["rows"] == [row]
    assert len(selected_views) == 1
    assert fallback_calls == []
    published = materialization._VIEWS[root]
    assert published is selected_views[0]
    published.close()


def test_materialize_selection_rejects_projection_source_drift(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    row = _canonical_projection_fixture(tmp_path, monkeypatch)
    state = ["source"]
    monkeypatch.setattr(
        materialization, "_source_state", lambda _root, _contracts: (state[0],)
    )
    seen: list[materialization.Projection] = []

    def select(view: materialization.Projection, _limit: int) -> list[dict[str, object]]:
        seen.append(view)
        state[0] = "changed"
        # This is the same source check used by the authoritative binding
        # path; it must reject before any legacy source rebuild can occur.
        materialization.canonical_binding(tmp_path, str(row["label_record_sha256"]))
        return [row]

    monkeypatch.setattr(materialization, "_selection", select)
    with pytest.raises(
        store.DistillationStoreError,
        match="canonical materialization projection changed",
    ):
        materialization._materialize(tmp_path, limit=1)

    assert seen
    with pytest.raises(sqlite3.ProgrammingError):
        seen[0].connection.execute("SELECT 1")
    assert tmp_path.resolve() not in materialization._VIEWS


@pytest.mark.parametrize("with_prior", [False, True])
def test_materialize_selection_exception_restores_previous_view_and_closes_new_view(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, with_prior: bool,
) -> None:
    _canonical_projection_fixture(tmp_path, monkeypatch)
    old_view = materialization._build(tmp_path) if with_prior else None
    new_view = materialization._build(tmp_path)
    root = tmp_path.resolve()
    if old_view is not None:
        materialization._VIEWS[root] = old_view

    monkeypatch.setattr(materialization, "_build", lambda _root: new_view)

    def fail(view: materialization.Projection, _limit: int) -> list[dict[str, object]]:
        assert materialization._VIEWS[root] is view
        raise RuntimeError("selection failed")

    monkeypatch.setattr(materialization, "_selection", fail)
    try:
        with pytest.raises(RuntimeError, match="selection failed"):
            materialization._materialize(tmp_path, limit=1)
        with pytest.raises(sqlite3.ProgrammingError):
            new_view.connection.execute("SELECT 1")
        if old_view is None:
            assert root not in materialization._VIEWS
        else:
            assert materialization._VIEWS[root] is old_view
            old_view.connection.execute("SELECT 1").fetchone()
    finally:
        if old_view is not None:
            old_view.close()


def test_materialize_success_closes_previous_view_and_publishes_new_view(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    _canonical_projection_fixture(tmp_path, monkeypatch)
    old_view = materialization._build(tmp_path)
    new_view = materialization._build(tmp_path)
    root = tmp_path.resolve()
    materialization._VIEWS[root] = old_view
    selected_during_selection: list[materialization.Projection] = []

    def select(view: materialization.Projection, _limit: int) -> list[dict[str, object]]:
        selected_during_selection.append(view)
        assert materialization._VIEWS[root] is view
        return []

    monkeypatch.setattr(materialization, "_build", lambda _root: new_view)
    monkeypatch.setattr(materialization, "_selection", select)
    monkeypatch.setattr(
        distill,
        "_finalize_materialized_training_rows",
        lambda _root, rows, _limit, **_kwargs: {"rows": rows},
    )

    try:
        assert materialization._materialize(tmp_path, limit=1) == {"rows": []}
        assert selected_during_selection == [new_view]
        assert materialization._VIEWS[root] is new_view
        with pytest.raises(sqlite3.ProgrammingError):
            old_view.connection.execute("SELECT 1")
        assert tuple(new_view.connection.execute("SELECT 1").fetchone()) == (1,)
    finally:
        new_view.close()


def test_source_change_during_sync_rolls_back_dirty_marks(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _append_label(tmp_path, "rally-a", "candidate-a")
    calls: list[set[str]] = []
    _stub_build_dependencies(monkeypatch, materialize_calls=calls)
    first = materialization._build(tmp_path)
    first.connection.close()
    projection = materialization._path(tmp_path)
    checkpoint = projection.with_suffix(projection.suffix + ".checkpoint.json").read_bytes()
    with sqlite3.connect(projection) as connection:
        before = connection.execute(
            "SELECT dirty,row_json FROM rows WHERE rally_id='rally-a'"
        ).fetchone()

    label_path = store.distillation_dir(tmp_path) / store.LABEL_LEDGER_FILE
    original_current = ledger.LedgerIndex.current

    def changed_source(self: ledger.LedgerIndex) -> bool:
        if self.path == label_path:
            return False
        return original_current(self)

    monkeypatch.setattr(ledger.LedgerIndex, "current", changed_source)
    with pytest.raises(store.DistillationStoreError, match="source changed"):
        materialization._build(tmp_path)

    with sqlite3.connect(projection) as connection:
        after = connection.execute(
            "SELECT dirty,row_json FROM rows WHERE rally_id='rally-a'"
        ).fetchone()
    assert after == before
    assert projection.with_suffix(projection.suffix + ".checkpoint.json").read_bytes() == checkpoint


def test_source_change_during_batch_does_not_publish_partial_projection(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _append_label(tmp_path, "rally-a", "candidate-a")
    calls: list[set[str]] = []
    _stub_build_dependencies(monkeypatch, materialize_calls=calls)
    first = materialization._build(tmp_path)
    first.connection.close()
    projection = materialization._path(tmp_path)
    with sqlite3.connect(projection) as connection:
        before = connection.execute(
            "SELECT dirty,row_json FROM rows WHERE rally_id='rally-a'"
        ).fetchone()

    state = ["source"]
    monkeypatch.setattr(materialization, "_source_state", lambda _root, _contracts: (state[0],))
    monkeypatch.setattr(
        catalog,
        "rally_changes",
        lambda _root, after_revision=0: {
            "catalog_lineage": "lineage",
            "revision": int(after_revision) + 1,
            "rally_ids": ["rally-a"],
        },
    )
    real_materialize = materialization._materialize_batch

    def mutate_after_batch(*args: object, **kwargs: object) -> None:
        real_materialize(*args, **kwargs)
        state[0] = "changed"

    monkeypatch.setattr(materialization, "_materialize_batch", mutate_after_batch)
    with pytest.raises(store.DistillationStoreError, match="source changed"):
        materialization._build(tmp_path)

    with sqlite3.connect(projection) as connection:
        after = connection.execute(
            "SELECT dirty,row_json FROM rows WHERE rally_id='rally-a'"
        ).fetchone()
    # The synchronization marker is a committed, resumable pending state.  The
    # attempted join itself must not publish its changed row_json.
    assert after == (1, before[1])

    state[0] = "source"
    monkeypatch.setattr(materialization, "_materialize_batch", real_materialize)
    resumed = materialization._build(tmp_path)
    resumed.connection.close()
    with sqlite3.connect(projection) as connection:
        clean = connection.execute(
            "SELECT dirty,row_json FROM rows WHERE rally_id='rally-a'"
        ).fetchone()
    assert clean[0] == 0
    assert clean[1] != before[1]


def test_failed_partial_batch_reuses_rows_already_cleaned(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # One materialization call handles at most 100 rallies.  A failure in the
    # second call therefore leaves the first committed batch reusable.
    for index in range(101):
        rally_id = f"rally-{index:03d}"
        _append_label(tmp_path, rally_id, f"candidate-{index:03d}")
    calls: list[set[str]] = []
    _stub_build_dependencies(monkeypatch, materialize_calls=calls)
    first = materialization._build(tmp_path)
    first.connection.close()
    monkeypatch.setattr(
        materialization,
        "_contract_dependencies",
        lambda _root, _labels: {
            "code": "changed-code",
            "native": "native",
            "ox": "ox",
            "files": {},
        },
    )
    real_materialize = materialization._materialize_batch
    attempts: list[set[str]] = []

    def fail_on_second_batch(
        connection: object,
        _root: Path,
        _labels: object,
        rally_ids: list[str] | tuple[str, ...],
        _ox: str,
    ) -> None:
        ids = {str(value) for value in rally_ids}
        attempts.append(ids)
        if len(attempts) == 2:
            raise store.DistillationStoreError("injected partial materialization failure")
        real_materialize(connection, _root, _labels, rally_ids, _ox)

    monkeypatch.setattr(materialization, "_materialize_batch", fail_on_second_batch)
    with pytest.raises(store.DistillationStoreError, match="partial materialization"):
        materialization._build(tmp_path)
    with sqlite3.connect(materialization._path(tmp_path)) as connection:
        state = {
            row[0]: row[1]
            for row in connection.execute("SELECT rally_id,dirty FROM rows ORDER BY rally_id")
        }
    assert sum(value == 0 for value in state.values()) == 100
    assert state["rally-100"] == 1

    monkeypatch.setattr(materialization, "_materialize_batch", real_materialize)
    resumed = materialization._build(tmp_path)
    resumed.connection.close()
    assert calls[-1] == {"rally-100"}


def test_validated_sources_route_uses_canonical_materialization(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    called: list[tuple[Path, int]] = []
    monkeypatch.setattr(catalog, "_read_catalog_checkpoint", lambda _root: {"ready": True})
    monkeypatch.setattr(catalog, "_catalog_lineage", lambda _checkpoint: "lineage")

    def canonical(root: Path, limit: int) -> dict[str, object]:
        called.append((root, limit))
        return {"rows": []}

    monkeypatch.setattr(materialization, "materialize", canonical)
    assert distill.materialize_training_rows(tmp_path, limit=17, _validated_sources=True) == {
        "rows": []
    }
    assert called == [(tmp_path.resolve(), 17)]
    with pytest.raises(distill.DistillationError, match="cannot accept supplied"):
        distill.materialize_training_rows(
            tmp_path,
            _validated_sources=True,
            _label_rows=[],
        )


def test_selection_matches_legacy_old_recent_locked_order(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    rows = _selection_rows()
    assignments = {
        str(row["rally_id"]): ("test" if row["locked_test_read_only"] else "train")
        for row in rows
    }
    monkeypatch.setattr(
        distill,
        "_materialization_split_context",
        lambda _root, _plans: ("plan", assignments, {"plan"}),
    )
    view = _selection_view(tmp_path, rows)
    selected = materialization._selection(view, limit=4)
    selected_final = distill._allocate_materialized_rows(selected, 4)
    expected = distill._allocate_materialized_rows(rows, 4)

    assert [row["rally_id"] for row in selected] == [
        "old-a",
        "recent-a",
        "locked-a",
        "old-b",
    ]

    assert [row["record_sha256"] for row in selected_final] == [
        row["record_sha256"] for row in expected
    ]
    assert [row["rally_id"] for row in selected_final] == [
        "locked-a",
        "old-a",
        "old-b",
        "recent-a",
    ]
    view.connection.close()


def test_selection_tie_breaks_by_rally_then_sequence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    rows = _selection_rows()
    rows[0]["as_of"] = rows[1]["as_of"] = rows[0]["epoch"] = rows[1]["epoch"] = 1
    assignments = {
        str(row["rally_id"]): ("test" if row["locked_test_read_only"] else "train")
        for row in rows
    }
    monkeypatch.setattr(
        distill,
        "_materialization_split_context",
        lambda _root, _plans: ("plan", assignments, {"plan"}),
    )
    view = _selection_view(tmp_path, rows)
    selected = materialization._selection(view, limit=4)
    selected_final = distill._allocate_materialized_rows(selected, 4)
    expected = distill._allocate_materialized_rows(rows, 4)

    assert [row["record_sha256"] for row in selected_final] == [
        row["record_sha256"] for row in expected
    ]
    view.connection.close()


def test_counterfactual_reservation_requires_structure_without_verdict_filter(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    rows = [
        {
            "record_sha256": "record-cf-uncertain",
            "rally_id": "cf-uncertain",
            "epoch": 1.0,
            "profile": distill.NATIVE_LUNA_SINGLE_PROFILE,
            "source": "counterfactual-label",
            "label_record_sha256": "a" * 64,
            "label_split_plan_id": "plan",
            "verdict": "uncertain",
            "structurally_valid": True,
        },
        {
            "record_sha256": "record-cf-helpful",
            "rally_id": "cf-helpful",
            "epoch": 2.0,
            "profile": distill.NATIVE_LUNA_SINGLE_PROFILE,
            "source": "counterfactual-label",
            "label_record_sha256": "b" * 64,
            "label_split_plan_id": "plan",
            "verdict": "helpful",
            "structurally_valid": False,
        },
    ]
    view = _selection_view(tmp_path, rows)
    monkeypatch.setattr(
        distill,
        "_read_split_plan_artifact",
        lambda _root, _plan_id: {"artifact_id": "plan", "assignments": {
            "cf-uncertain": "test", "cf-helpful": "test"
        }},
    )
    monkeypatch.setattr(
        distill,
        "_materialized_row_integrity",
        lambda row, **_kwargs: bool(row.get("structurally_valid")),
    )

    reserved = materialization._counterfactual_reservations(view, limit=4)

    assert [row["rally_id"] for row in reserved] == ["cf-uncertain"]
    view.connection.close()


def test_locked_classification_survives_restart_and_only_updates_new_labels(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    _append_label(tmp_path, "rally-a", "candidate-a")
    calls: list[set[str]] = []
    _stub_build_dependencies(monkeypatch, materialize_calls=calls)
    view = materialization._build(tmp_path)
    materialization._classify_locked_rows(view, "plan", {"rally-a": "test"}, {"plan"})
    assert view.connection.execute("SELECT locked FROM rows").fetchone()[0] == 1
    view.close()

    reopened = materialization._build(tmp_path)
    before = ledger.file_state(reopened.path)
    materialization._classify_locked_rows(reopened, "plan", {"rally-a": "test"}, {"plan"})
    assert ledger.file_state(reopened.path) == before
    reopened.close()

    _append_label(tmp_path, "rally-a", "candidate-b")
    appended = materialization._build(tmp_path)
    with ledger.connect(appended.path) as connection:
        connection.execute("CREATE TABLE observed(sequence INTEGER)")
        connection.execute(
            "CREATE TRIGGER observe_locked AFTER UPDATE OF locked ON rows "
            "BEGIN INSERT INTO observed VALUES(new.sequence); END"
        )
    materialization._classify_locked_rows(appended, "plan", {"rally-a": "test"}, {"plan"})
    assert [tuple(row) for row in appended.connection.execute("SELECT sequence,locked FROM rows")] == [(1, 1), (2, 1)]
    assert [row[0] for row in appended.connection.execute("SELECT sequence FROM observed")] == [2]
    appended.close()


@pytest.mark.parametrize("changed", ["assignment", "compatibility", "regressed_cursor"])
def test_locked_classification_recomputes_when_its_verified_context_changes(
    tmp_path: Path, changed: str,
) -> None:
    view = _selection_view(tmp_path, _selection_rows())
    assignments = {"locked-a": "test"}
    materialization._classify_locked_rows(view, "plan", assignments, {"plan"})
    assert view.connection.execute("SELECT locked FROM rows WHERE rally_id='locked-a'").fetchone()[0] == 1
    if changed == "assignment":
        assignments = {"old-a": "test"}
    elif changed == "regressed_cursor":
        view.metadata["selection"]["sequence"] = 100
    materialization._classify_locked_rows(
        view, "plan", assignments, set() if changed == "compatibility" else {"plan"},
    )
    actual = {row[0] for row in view.connection.execute("SELECT rally_id FROM rows WHERE locked=1")}
    assert actual == ({"old-a"} if changed == "assignment" else set() if changed == "compatibility" else {"locked-a"})
    view.connection.close()


def test_split_growth_does_not_rewrite_existing_label_classifications(tmp_path: Path) -> None:
    view = _selection_view(tmp_path, _selection_rows())
    assignments = {"locked-a": "test", "old-a": "train"}
    materialization._classify_locked_rows(view, "plan", assignments, {"plan"})
    with ledger.connect(view.path) as connection:
        connection.execute("CREATE TABLE observed(rally_id TEXT)")
        connection.execute(
            "CREATE TRIGGER observe_locked AFTER UPDATE OF locked ON rows "
            "BEGIN INSERT INTO observed VALUES(new.rally_id); END"
        )
    assignments = {**assignments, "new-rally": "embargo"}
    materialization._classify_locked_rows(view, "growth", assignments, {"plan", "growth"})
    assert view.connection.execute("SELECT COUNT(*) FROM observed").fetchone()[0] == 0
    materialization._classify_locked_rows(
        view, "growth", {**assignments, "old-a": "test"}, {"plan", "growth"},
    )
    assert [row[0] for row in view.connection.execute("SELECT rally_id FROM observed")] == ["old-a"]
    assert view.connection.execute("SELECT locked FROM rows WHERE rally_id='old-a'").fetchone()[0] == 1
    view.connection.close()


def test_certification_selection_precedes_limit_and_keeps_adverse_rows(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    rows = _selection_rows()
    for row in rows:
        row.update(source="teacher-label", verdict="relevant")
    rows[-2].update(label_split_plan_id="certification", verdict="irrelevant")
    rows[-1].update(label_split_plan_id="certification", probe=True, repeat_pair_id="incomplete")
    assignments = {str(row["rally_id"]): "test" for row in rows[-2:]}
    monkeypatch.setattr(distill, "_read_certification_split_plan", lambda _root: {
        "artifact_id": "certification", "assignments": assignments,
    })
    monkeypatch.setattr(distill, "_materialization_split_context", lambda _root, _plans: (
        "certification", assignments, {"certification"},
    ))
    view = _selection_view(tmp_path, rows)
    try:
        selected = materialization._selection(view, limit=3)
        assert [row["rally_id"] for row in selected] == ["locked-b", "locked-c"]
        assert selected[0]["verdict"] == "irrelevant"
        assert selected[1]["repeat_pair_id"] == "incomplete"
        # Selection must neither alter nor remove historical audit rows.
        assert view.connection.execute("SELECT COUNT(*) FROM rows").fetchone()[0] == len(rows)
        with pytest.raises(distill.DistillationError, match="exceeds training snapshot limit"):
            materialization._selection(view, limit=1)
    finally:
        view.connection.close()
