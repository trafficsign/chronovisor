"""Resumable canonical label joins, invalidated by verified source changes.

Only derived data lives here. Ledger cursors verify the chain, the catalog's
change journal invalidates affected rallies, and every new/changed label uses
the original materialization validator. A sealed physical checkpoint binds
the SQLite projection to those sources; an interrupted or changed checkpoint
cannot authorize an old projection.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
import threading
from collections import deque
from collections.abc import Mapping, Sequence
from contextlib import closing
from pathlib import Path
from typing import Any

from chronovisor.recall import recall_distillation_catalog as catalog
from chronovisor.recall import recall_distillation_ledger as ledger
from chronovisor.recall import recall_distillation_store as store
from chronovisor.recall.recall_distillation_single_teacher_gate import (
    NATIVE_LUNA_SINGLE_PROFILE,
)

_VERSION = "canonical-materialization-v1"
_VIEWS: dict[Path, Projection] = {}
_VIEW_LOCK = threading.RLock()


def _path(root: Path) -> Path:
    return store.distillation_dir(root) / "materialization.sqlite3"


def _encoded(value: Any) -> str:
    return store.projection_json(value)


def _schema(connection: sqlite3.Connection) -> None:
    connection.executescript("""
        CREATE TABLE IF NOT EXISTS metadata(key TEXT PRIMARY KEY,value TEXT NOT NULL);
        CREATE TABLE IF NOT EXISTS rows(
            sequence INTEGER PRIMARY KEY, record_sha256 TEXT NOT NULL UNIQUE,
            rally_id TEXT NOT NULL, receipt_sha256 TEXT NOT NULL,
            profile TEXT NOT NULL, profile_contract_id TEXT NOT NULL,
            label_plan TEXT NOT NULL, epoch REAL NOT NULL DEFAULT 0,
            dirty INTEGER NOT NULL DEFAULT 1, row_json TEXT,
            excluded_prior INTEGER NOT NULL, excluded_unsupported INTEGER NOT NULL,
            locked INTEGER NOT NULL DEFAULT 0
        );
        CREATE INDEX IF NOT EXISTS materialization_rally ON rows(rally_id,dirty);
        CREATE INDEX IF NOT EXISTS materialization_dirty ON rows(dirty,rally_id);
        CREATE INDEX IF NOT EXISTS materialization_receipt ON rows(receipt_sha256);
        CREATE INDEX IF NOT EXISTS materialization_plan ON rows(label_plan);
        CREATE INDEX IF NOT EXISTS materialization_present_plan ON rows(label_plan)
            WHERE row_json IS NOT NULL;
        CREATE INDEX IF NOT EXISTS materialization_order ON rows(locked,epoch,rally_id,sequence)
            WHERE row_json IS NOT NULL;
        CREATE TABLE IF NOT EXISTS dependencies(
            path TEXT NOT NULL, rally_id TEXT NOT NULL, state TEXT NOT NULL,
            PRIMARY KEY(path,rally_id)
        );
        CREATE TABLE IF NOT EXISTS assignments(rally_id TEXT PRIMARY KEY,split TEXT NOT NULL);
        CREATE TABLE IF NOT EXISTS compatible(plan_id TEXT PRIMARY KEY);
    """)


def _read_metadata(connection: sqlite3.Connection) -> dict[str, Any]:
    return {row[0]: json.loads(row[1]) for row in connection.execute("SELECT * FROM metadata")}


def _save(connection: sqlite3.Connection, path: Path, metadata: Mapping[str, Any]) -> None:
    connection.executemany(
        "INSERT OR REPLACE INTO metadata VALUES(?,?)",
        ((key, _encoded(value)) for key, value in metadata.items()),
    )
    connection.commit()
    ledger.seal_checkpoint(path, kind=_VERSION, metadata=dict(metadata))


def _code_identity() -> str:
    digest = hashlib.sha256()
    for name in (
        "recall_distillation.py", "recall_distillation_native.py",
        "recall_distillation_catalog.py", "recall_distillation_store.py",
        "recall_distillation_ledger.py", "recall_distillation_materialization.py",
        "recall_distillation_validation.py",
    ):
        digest.update(Path(__file__).with_name(name).read_bytes())
    return digest.hexdigest()


def _contract_dependencies(root: Path, labels: ledger.LedgerIndex) -> dict[str, Any]:
    from chronovisor.recall import recall_distillation as distill
    from chronovisor.recall import recall_distillation_native as native

    base = store.distillation_dir(root)
    paths: set[Path] = set()
    for row in labels.query("SELECT DISTINCT profile,profile_contract_id FROM rows"):
        profile, contract = str(row[0]), str(row[1])
        if not contract or len(contract) != 64 or any(c not in "0123456789abcdef" for c in contract):
            continue
        if profile == NATIVE_LUNA_SINGLE_PROFILE:
            paths.add(native._contract_dir(root) / f"{contract}.json")
        elif profile == distill.OX_SINGLE_PROFILE:
            paths.add(base / "ox-profile-contracts" / f"{contract}.json")
    return {
        "code": _code_identity(),
        "native": native.current_contract_id(root),
        "ox": distill._current_ox_profile_contract_id(root),
        "files": {str(path): ledger.file_state(path) for path in sorted(paths)},
    }


def _prefix_matches(index: ledger.LedgerIndex, old: Mapping[str, Any]) -> bool:
    count = old.get("records", 0)
    if isinstance(count, bool) or not isinstance(count, int) or count < 0:
        return False
    if not count:
        return bool(old.get("head_sha256", "") == "")
    rows = index.query("SELECT record_sha256 FROM rows WHERE sequence=?", (count,))
    return bool(rows and rows[0][0] == old.get("head_sha256"))


def _dirty_rallies(connection: sqlite3.Connection, ids: Sequence[str]) -> None:
    connection.executemany(
        "UPDATE rows SET dirty=1 WHERE rally_id=?", ((value,) for value in ids)
    )


def _add_labels(
    connection: sqlite3.Connection, labels: ledger.LedgerIndex, after: int, ox: str,
) -> None:
    from chronovisor.recall import recall_distillation as distill

    while True:
        batch = labels.query(
            "SELECT sequence,row_json FROM rows WHERE sequence>? ORDER BY sequence LIMIT 500",
            (after,),
        )
        if not batch:
            return
        values = []
        for entry in batch:
            label = json.loads(entry[1])
            rally_id = label.get("rally_id")
            candidate_id = label.get("candidate_id")
            if (
                not isinstance(rally_id, str) or not rally_id
                or (candidate_id is not None and (
                    isinstance(candidate_id, bool)
                    or not isinstance(candidate_id, (str, int)) or candidate_id == ""
                ))
            ):
                raise store.DistillationStoreError("materialization label identity is invalid")
            values.append((
                entry[0], label["record_sha256"], rally_id,
                str(label.get("provider_receipt_sha256") or ""),
                str(label.get("profile") or ""), str(label.get("profile_contract_id") or ""),
                str(label.get("split_plan_id") or ""),
                int(label.get("profile") == distill.OX_SINGLE_PROFILE
                    and label.get("profile_contract_id") != ox),
                int(distill._unsupported_materialization_profile(label)),
            ))
        connection.executemany(
            "INSERT INTO rows(sequence,record_sha256,rally_id,receipt_sha256,profile,"
            "profile_contract_id,label_plan,excluded_prior,excluded_unsupported) "
            "VALUES(?,?,?,?,?,?,?,?,?)", values,
        )
        after = int(batch[-1][0])


def _sync_inputs(
    connection: sqlite3.Connection, root: Path, metadata: dict[str, Any],
    labels: ledger.LedgerIndex, receipts: ledger.LedgerIndex,
    contracts: dict[str, Any],
) -> dict[str, Any]:
    journal = catalog.rally_changes(root, after_revision=int(metadata.get("revision", 0)))
    candidates = catalog.candidate_index_state(root)
    reset = not _prefix_matches(labels, metadata.get("labels", {}))
    if reset:
        connection.execute("DELETE FROM rows")
        connection.execute("DELETE FROM dependencies")
        metadata = {}
    old_labels = int(metadata.get("labels", {}).get("records", 0))
    _add_labels(connection, labels, old_labels, str(contracts["ox"]))
    if (
        metadata.get("contracts") != contracts
        or metadata.get("lineage") != journal["catalog_lineage"]
        or not _prefix_matches(receipts, metadata.get("receipts", {}))
        or int(metadata.get("candidate_count", 0)) > int(candidates["record_count"])
        or metadata.get("candidate_ino") != candidates.get("ledger_ino")
        or not catalog.candidate_prefix_matches(
            root, records=int(metadata.get("candidate_count", 0)),
            head_sha256=str(metadata.get("candidate_head", "")),
        )
    ):
        connection.execute("UPDATE rows SET dirty=1")
    else:
        _dirty_rallies(connection, journal["rally_ids"])
        _dirty_rallies(connection, sorted(catalog.candidate_rally_ids(
            root, after_seq=int(metadata.get("candidate_count", 0)),
        )))
        for receipt in receipts.query(
            "SELECT receipt_sha256 FROM rows WHERE sequence>?",
            (int(metadata.get("receipts", {}).get("records", 0)),),
        ):
            connection.execute("UPDATE rows SET dirty=1 WHERE receipt_sha256=?", (receipt[0],))
    for dependency in connection.execute("SELECT * FROM dependencies").fetchall():
        if _encoded(ledger.file_state(Path(dependency["path"]))) != dependency["state"]:
            _dirty_rallies(connection, [str(dependency["rally_id"])])
    return {
        "labels": labels.cursor, "receipts": receipts.cursor, "contracts": contracts,
        "lineage": journal["catalog_lineage"], "revision": journal["revision"],
        "candidate_count": candidates["record_count"], "candidate_ino": candidates.get("ledger_ino"),
        "candidate_head": candidates["head_sha256"],
        "selection": metadata.get("selection", {}),
    }


def _materialize_batch(
    connection: sqlite3.Connection, root: Path, labels: ledger.LedgerIndex,
    rally_ids: Sequence[str], ox: str,
) -> None:
    from chronovisor.recall import recall_distillation as distill
    from chronovisor.recall.recall_distillation_validation import (
        _catalog_sources,
        _label_source_claim,
    )

    pending = []
    for rally_id in rally_ids:
        pending.extend(connection.execute(
            "SELECT sequence,record_sha256 FROM rows WHERE rally_id=? AND dirty=1", (rally_id,)
        ).fetchall())
    source_rows = [labels.lookup("record_sha256", str(row[1])) for row in pending]
    if any(row is None for row in source_rows):
        raise store.DistillationStoreError("materialization label disappeared")
    rows = [row for row in source_rows if row is not None]
    claims = [_label_source_claim(row) for row in rows]
    sources = _catalog_sources(root, rally_ids, claims)
    if sources is None:
        raise store.DistillationStoreError("materialization catalog is unavailable")
    rallies, snapshots = sources
    dependencies: list[tuple[str, str, str]] = []
    for rally_id, rally in rallies.items():
        for receipt in rally.get("exposure_receipts", []):
            if isinstance(receipt, Mapping):
                path = store.distillation_dir(root) / "exposures" / f"{receipt.get('exposure_artifact_id') or ''}.json"
                dependencies.append((str(path), rally_id, _encoded(ledger.file_state(path))))
    features, contracts = distill._materialization_feature_pairs(root, rallies, snapshots)
    updates = []
    for entry, label in zip(pending, rows, strict=True):
        projected = distill._materialization_label_row(label, root, rallies, features, contracts, ox)
        updates.append((
            distill._source_epoch(projected) if projected is not None else 0,
            _encoded(projected) if projected is not None else None,
            int(label.get("profile") == distill.OX_SINGLE_PROFILE
                and label.get("profile_contract_id") != ox),
            int(distill._unsupported_materialization_profile(label)),
            entry[0],
        ))
    # A reference that is absent today is also tracked, so its later arrival
    # invalidates the rally rather than freezing an earlier negative result.
    for rally_id in rallies:
        connection.execute("DELETE FROM dependencies WHERE rally_id=?", (rally_id,))
    for path_text, _rally_id, state in dependencies:
        if _encoded(ledger.file_state(Path(path_text))) != state:
            raise store.DistillationStoreError("materialization exposure changed during join")
    connection.executemany("INSERT OR REPLACE INTO dependencies VALUES(?,?,?)", dependencies)
    connection.executemany(
        "UPDATE rows SET epoch=?,row_json=?,excluded_prior=?,excluded_unsupported=?,dirty=0 WHERE sequence=?", updates,
    )


def _source_state(root: Path, contracts: Mapping[str, Any]) -> tuple[Any, ...]:
    from chronovisor.recall import recall_distillation_native as native
    from chronovisor.recall.recall_distillation_validation import _runtime_paths

    paths = (*_runtime_paths(root), native._receipt_path(root), native._contract_pointer_path(root),
             store.distillation_dir(root) / "ox-profile-contract.json")
    paths += tuple(store._chain_checkpoint_path(path) for path in (
        store.distillation_dir(root) / store.LABEL_LEDGER_FILE, native._receipt_path(root),
        store.distillation_dir(root) / "candidate-ledger.jsonl",
    ))
    return tuple(ledger.file_state(path) for path in paths) + tuple(
        ledger.file_state(Path(path)) for path in contracts["files"]
    )


def _build(root: Path) -> Projection:
    from chronovisor.recall import recall_distillation_native as native

    path = _path(root)
    labels = ledger.ledger_index(store.distillation_dir(root) / store.LABEL_LEDGER_FILE)
    receipts = ledger.ledger_index(native._receipt_path(root))
    with store._locked(path.with_suffix(".lock")), closing(ledger.connect(path)) as connection:
        saved = ledger.checkpoint(path)
        _schema(connection)
        metadata = _read_metadata(connection)
        if saved is None or saved.get("kind") != _VERSION or saved.get("metadata") != metadata:
            connection.execute("DELETE FROM rows")
            connection.execute("DELETE FROM dependencies")
            connection.execute("DELETE FROM metadata")
            metadata = {}
        contracts = _contract_dependencies(root, labels)
        before = _source_state(root, contracts)
        metadata = _sync_inputs(connection, root, metadata, labels, receipts, contracts)
        if (before != _source_state(root, contracts) or not labels.current() or not receipts.current()):
            connection.rollback()
            raise store.DistillationStoreError("materialization source changed during synchronization")
        _save(connection, path, metadata)
        while True:
            ids = [str(row[0]) for row in connection.execute(
                "SELECT DISTINCT rally_id FROM rows WHERE dirty=1 ORDER BY rally_id LIMIT 100"
            )]
            if not ids:
                break
            _materialize_batch(connection, root, labels, ids, metadata["contracts"]["ox"])
            if (before != _source_state(root, metadata["contracts"])
                    or not labels.current() or not receipts.current()):
                connection.rollback()
                raise store.DistillationStoreError("materialization source changed during join")
            _save(connection, path, metadata)
        if before != _source_state(root, metadata["contracts"]):
            raise store.DistillationStoreError("materialization source changed during completion")
    return Projection(root, metadata, before)


class Projection:
    def __init__(self, root: Path, metadata: dict[str, Any], source_state: tuple[Any, ...]) -> None:
        self.root = root
        self.metadata = metadata
        self.source_state = source_state
        self.path = _path(root)
        self.state = ledger.file_state(self.path)
        self.connection = ledger.connect(self.path, readonly=True)
        self.rallies: dict[str, Mapping[str, Any]] = {}
        self.dependencies: dict[str, list[tuple[Path, Any]]] = {}
        self.selection_plans: set[str] = set()
        self.split_context: tuple[str, dict[str, str], set[str]] | None = None
        for row in self.connection.execute("SELECT * FROM dependencies"):
            self.dependencies.setdefault(row["rally_id"], []).append((Path(row["path"]), json.loads(row["state"])))
        self.check()

    def check(self, rally_id: str = "", *, all_dependencies: bool = False) -> None:
        dependencies = (
            (entry for values in self.dependencies.values() for entry in values)
            if all_dependencies else iter(self.dependencies.get(rally_id, []))
        )
        saved = ledger.checkpoint(self.path)
        if (
            ledger.file_state(self.path) != self.state or saved is None
            or saved.get("metadata") != self.metadata
            or _source_state(self.root, self.metadata["contracts"]) != self.source_state
            or any(ledger.file_state(path) != state for path, state in dependencies)
        ):
            raise store.DistillationStoreError("canonical materialization projection changed")

    def close(self) -> None:
        self.connection.close()

    def binding(self, record: str) -> tuple[dict[str, Any] | None, dict[str, Any], Mapping[str, Any]] | None:
        self.check()
        value = self.connection.execute(
            "SELECT rally_id,row_json FROM rows WHERE record_sha256=? AND dirty=0", (record,)
        ).fetchone()
        if value is None:
            return None
        rally_id = str(value[0])
        self.check(rally_id)
        label = ledger.ledger_index(store.distillation_dir(self.root) / store.LABEL_LEDGER_FILE).lookup("record_sha256", record)
        if label is None:
            raise store.DistillationStoreError("materialization label binding disappeared")
        if rally_id not in self.rallies:
            self.rallies.update((str(row["rally_id"]), row) for row in catalog.rallies(self.root, [rally_id]))
        self.check(rally_id)
        return (json.loads(value[1]) if value[1] is not None else None, label, self.rallies.get(rally_id, {}))


def _update_selection_tables(
    connection: sqlite3.Connection, assignments: Mapping[str, str], compatible: set[str],
) -> None:
    old_assignments = dict(connection.execute("SELECT rally_id,split FROM assignments"))
    changed = {
        rally_id for rally_id in old_assignments.keys() | assignments.keys()
        if old_assignments.get(rally_id) != assignments.get(rally_id)
    }
    old_compatible = {str(row[0]) for row in connection.execute("SELECT plan_id FROM compatible")}
    connection.executemany("INSERT INTO changed_rallies VALUES(?)", ((value,) for value in changed))
    connection.executemany(
        "INSERT INTO changed_plans VALUES(?)", ((value,) for value in old_compatible ^ compatible),
    )
    connection.executemany("DELETE FROM assignments WHERE rally_id=?", ((value,) for value in changed))
    connection.executemany(
        "INSERT INTO assignments VALUES(?,?)", ((value, assignments[value]) for value in changed if value in assignments),
    )
    connection.executemany("DELETE FROM compatible WHERE plan_id=?", ((value,) for value in old_compatible - compatible))
    connection.executemany("INSERT INTO compatible VALUES(?)", ((value,) for value in compatible - old_compatible))


def _classify_locked_rows(
    view: Projection, plan_id: str, assignments: Mapping[str, str], compatible: set[str],
) -> None:
    from chronovisor.recall import recall_distillation as distill

    profiles = tuple(sorted(distill.SINGLE_TEACHER_PROFILES))
    context = hashlib.sha256(_encoded(
        [plan_id, assignments, sorted(compatible), profiles]
    ).encode()).hexdigest()
    rules = _encoded(["locked-selection-v1", bool(plan_id), profiles])
    last = int(view.connection.execute("SELECT COALESCE(MAX(sequence),0) FROM rows").fetchone()[0])
    previous = view.metadata.get("selection", {})
    after = previous.get("sequence", -1) if isinstance(previous, Mapping) else -1
    reusable = (
        isinstance(previous, Mapping) and previous.get("rules") == rules
        and isinstance(after, int) and not isinstance(after, bool) and 0 <= after <= last
    )
    if reusable and previous.get("context") == context and after == last:
        return
    with closing(ledger.connect(view.path)) as connection:
        connection.execute("CREATE TEMP TABLE changed_rallies(rally_id TEXT PRIMARY KEY)")
        connection.execute("CREATE TEMP TABLE changed_plans(plan_id TEXT PRIMARY KEY)")
        if not reusable or previous.get("context") != context:
            _update_selection_tables(connection, assignments, compatible)
        connection.execute(
            "UPDATE rows SET locked=CASE WHEN ?<>'' AND EXISTS "
            "(SELECT 1 FROM assignments WHERE assignments.rally_id=rows.rally_id AND split='test') "
            f"AND (profile NOT IN ({','.join('?' for _ in profiles)}) OR label_plan IN "
            "(SELECT plan_id FROM compatible)) THEN 1 ELSE 0 END WHERE sequence>? "
            "OR rally_id IN (SELECT rally_id FROM changed_rallies) "
            "OR label_plan IN (SELECT plan_id FROM changed_plans)",
            (plan_id, *profiles, after if reusable else 0),
        )
        updated = {**view.metadata, "selection": {"context": context, "rules": rules, "sequence": last}}
        _save(connection, view.path, updated)
    view.metadata = updated
    view.state = ledger.file_state(view.path)


def _counterfactual_reservations(
    view: Projection, limit: int
) -> list[dict[str, Any]]:
    """Reserve structurally valid CF rows before bounded band filling.

    The training snapshot is bounded, while the generic locked and unlocked
    bands can be much larger than that bound.  A counterfactual row is
    reserved only after the same materialized-row integrity and exact-exposure
    checks used by the gate; verdict is deliberately not consulted here. Each
    row is validated against its own sealed source split plan so a compatible
    historical test plan is not rewritten during selection.
    """

    from chronovisor.recall import recall_distillation as distill

    if limit <= 0:
        return []
    try:
        source_rows = view.connection.execute(
            "SELECT row_json FROM rows WHERE row_json IS NOT NULL "
            "AND json_extract(row_json, '$.source') = ? "
            "ORDER BY epoch,rally_id,sequence",
            ("counterfactual-label",),
        )
    except sqlite3.OperationalError:
        # Older SQLite builds may not expose JSON1.  The row-level source
        # check below remains authoritative in that case.
        source_rows = view.connection.execute(
            "SELECT row_json FROM rows WHERE row_json IS NOT NULL "
            "ORDER BY epoch,rally_id,sequence"
        )
    plans: dict[str, Mapping[str, Any] | None] = {}
    reserved: list[dict[str, Any]] = []
    reserved_ids: set[str] = set()
    for entry in source_rows:
        try:
            row = json.loads(entry[0])
        except (TypeError, ValueError, json.JSONDecodeError):
            continue
        if not isinstance(row, Mapping) or row.get("source") != "counterfactual-label":
            continue
        label_plan_id = str(row.get("label_split_plan_id") or "")
        if label_plan_id not in plans:
            try:
                plans[label_plan_id] = distill._read_split_plan_artifact(
                    view.root, label_plan_id
                )
            except (KeyError, distill.DistillationError, store.DistillationStoreError):
                plans[label_plan_id] = None
        plan = plans[label_plan_id]
        if plan is None:
            continue
        rally_id = str(row.get("rally_id") or "")
        split = plan.get("assignments", {}).get(rally_id)
        if split != "test":
            continue
        candidate = {
            **row,
            "split": "test",
            "split_plan_id": label_plan_id,
            "locked_test_read_only": True,
            "locked_test_evidence_ref": f"split-plan:{label_plan_id}",
        }
        try:
            valid = distill._materialized_row_integrity(
                candidate, root=view.root, split_plan=plan
            )
        except (KeyError, TypeError, ValueError, distill.DistillationError,
                store.DistillationStoreError):
            valid = False
        if not valid:
            continue
        record = str(candidate.get("label_record_sha256") or "")
        if not record or record in reserved_ids:
            continue
        reserved_ids.add(record)
        reserved.append(candidate)
        if len(reserved) >= limit:
            break
    return reserved


def _selection(view: Projection, limit: int) -> list[dict[str, Any]]:
    from chronovisor.recall import recall_distillation as distill

    certification = distill._read_certification_split_plan(view.root)
    plans = {str(row[0]) for row in view.connection.execute("SELECT DISTINCT label_plan FROM rows WHERE row_json IS NOT NULL")}
    try:
        plan_id, assignments, compatible = distill._materialization_split_context(view.root, plans)
        view.selection_plans = plans
        view.split_context = (plan_id, assignments, compatible)
    except (KeyError, distill.DistillationError, store.DistillationStoreError):
        if certification:
            raise
        # Legacy snapshots without a sealed split use the original full-cohort
        # grouped split; the normal fixed-plan path always stays bounded.
        return [json.loads(row[0]) for row in view.connection.execute("SELECT row_json FROM rows WHERE row_json IS NOT NULL ORDER BY sequence")]
    _classify_locked_rows(view, plan_id, assignments, compatible)
    if certification:
        # Apply the preregistered plan identity before bounding the snapshot.
        # Every projected teacher row in this cohort is retained; neither its
        # verdict nor the gate result participates in selection.
        teachers = [json.loads(row[0]) for row in view.connection.execute(
            "SELECT row_json FROM rows WHERE row_json IS NOT NULL AND label_plan=? "
            "AND json_extract(row_json, '$.source') <> 'counterfactual-label' "
            "ORDER BY sequence LIMIT ?", (plan_id, limit + 1),
        )]
        if len(teachers) > limit:
            raise distill.DistillationError("certification inventory exceeds training snapshot limit")
        reserved = _counterfactual_reservations(view, limit - len(teachers))
        return [*teachers, *reserved]
    count = int(view.connection.execute("SELECT COUNT(*) FROM rows WHERE row_json IS NOT NULL").fetchone()[0])
    if count <= limit:
        return [json.loads(row[0]) for row in view.connection.execute("SELECT row_json FROM rows WHERE row_json IS NOT NULL ORDER BY sequence")]
    reserved = _counterfactual_reservations(view, limit)
    reserved_ids = {
        str(row.get("label_record_sha256") or "") for row in reserved
    }
    unlocked = int(view.connection.execute("SELECT COUNT(*) FROM rows WHERE row_json IS NOT NULL AND locked=0").fetchone()[0])
    old_count = max(0, unlocked - max(1, unlocked // 3))
    bands = []
    for locked, offset, take in ((0, 0, min(limit, old_count)), (0, old_count, limit), (1, 0, limit)):
        bands.append(deque(json.loads(row[0]) for row in view.connection.execute(
            "SELECT row_json FROM rows WHERE row_json IS NOT NULL AND locked=? "
            "ORDER BY epoch,rally_id,sequence LIMIT ? OFFSET ?", (locked, take, offset),
        )))
    selected: list[dict[str, Any]] = list(reserved)
    selected_ids = set(reserved_ids)
    while len(selected) < limit and any(bands):
        for band in bands:
            while band and len(selected) < limit:
                candidate = band.popleft()
                record = str(candidate.get("label_record_sha256") or "")
                if record and record in selected_ids:
                    continue
                selected.append(candidate)
                if record:
                    selected_ids.add(record)
                break
    return selected


def _materialize(root: Path, limit: int) -> dict[str, Any]:
    from chronovisor.recall import recall_distillation as distill

    if isinstance(limit, bool) or not isinstance(limit, int) or not 0 < limit <= 10_000:
        raise distill.DistillationError("training materialization limit is invalid")
    root = root.expanduser().resolve(strict=False)
    view = _build(root)
    try:
        with store._locked(view.path.with_suffix(".lock")):
            view.check(all_dependencies=True)
            # The caller holds _VIEW_LOCK; selection must bind rows through
            # this verified projection instead of rebuilding all source inputs.
            prior = _VIEWS.get(root)
            _VIEWS[root] = view
            try:
                selected = _selection(view, limit)
            finally:
                if prior is None:
                    _VIEWS.pop(root, None)
                else:
                    _VIEWS[root] = prior
            excluded = view.connection.execute(
                "SELECT COALESCE(SUM(excluded_prior),0),COALESCE(SUM(excluded_unsupported),0) FROM rows"
            ).fetchone()
            view.check(all_dependencies=True)
            artifact = distill._finalize_materialized_training_rows(
                root, selected, limit, excluded_prior_contract_rows=int(excluded[0]),
                excluded_unsupported_profile_rows=int(excluded[1]),
            )
            if view.split_context is not None and view.split_context != distill._materialization_split_context(
                root, view.selection_plans,
            ):
                raise store.DistillationStoreError("materialization split changed during selection")
            ids = sorted({str(row["rally_id"]) for row in selected})
            view.rallies.update((str(row["rally_id"]), row) for row in catalog.rallies(root, ids))
            view.check(all_dependencies=True)
        previous = _VIEWS.pop(root, None)
        if previous is not None:
            previous.close()
        if len(_VIEWS) >= 16:
            _VIEWS.pop(next(iter(_VIEWS))).close()
        _VIEWS[root] = view
        return artifact
    except BaseException:
        view.close()
        raise


def materialize(root: Path, limit: int) -> dict[str, Any]:
    with _VIEW_LOCK:
        return _materialize(root, limit)


def canonical_binding(root: Path, record: str) -> tuple[dict[str, Any] | None, dict[str, Any], Mapping[str, Any]] | None:
    """Use only an already-built canonical view, never caller-supplied rows."""

    with _VIEW_LOCK:
        view = _VIEWS.get(root.expanduser().resolve(strict=False))
        return view.binding(record) if view is not None else None
