from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import sqlite3
import sys
from contextlib import contextmanager
from pathlib import Path

import pytest

from chronovisor.core import canonical_json
from chronovisor.core.durable_state import okf_writer_lock
from chronovisor.recall import recall_distillation as distill
from chronovisor.recall import recall_distillation_store as store
from chronovisor.recall.recall_distillation_workset import DistillationWorkset

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "recall_r4_workset_cutover_test_module",
    ROOT / "scripts" / "recall_r4_workset_cutover.py",
)
assert SPEC is not None and SPEC.loader is not None
CUTOVER = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = CUTOVER
SPEC.loader.exec_module(CUTOVER)
REAL_VALIDATE_R0 = CUTOVER._validate_r0_evidence

SOURCE = {
    "source_commit": "a" * 40,
    "source_tree_sha256": "b" * 64,
    "source_ox_identity_sha256": "c" * 64,
}


def _state(path: Path) -> dict[str, int | str] | None:
    if not path.exists():
        return None
    observed = path.stat()
    return {
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "size_bytes": observed.st_size,
        "st_dev": observed.st_dev,
        "st_ino": observed.st_ino,
        "st_mtime_ns": observed.st_mtime_ns,
        "st_ctime_ns": observed.st_ctime_ns,
    }


def _offline_evidence(root: Path, path: Path) -> None:
    directory = store.distillation_dir(root)
    paths = {
        "candidate": directory / "candidate-ledger.jsonl",
        "candidate_anchor": directory / distill.R4_CANDIDATE_ANCHOR_FILE,
        "candidate_checkpoint": directory / "candidate-ledger.jsonl.head.json",
        "config": root / "config.toml",
        "distillation_lock": directory / "distillation-worker.lock",
        "workset": directory / "ox-workset.sqlite3",
        "workset_journal": directory / "ox-workset.sqlite3-journal",
        "workset_shm": directory / "ox-workset.sqlite3-shm",
        "workset_wal": directory / "ox-workset.sqlite3-wal",
    }
    after = {name: _state(item) for name, item in paths.items()}
    unsigned = {
        "captured_at_unix": 1,
        "kind": "r4-offline-bootstrap-receipt",
        "namespace": "recall-distillation",
        "production": {
            "root": str(root),
            "unchanged": True,
            "before": after,
            "after": after,
        },
        "schema": distill.R4_OFFLINE_BOOTSTRAP_SCHEMA,
        "scope": {
            "provider_calls": 0,
            "ox_enabled": False,
            "owned_clone_only": True,
            "production_certification": False,
            "r4_checkbox_complete": False,
        },
        "source": {"binding": SOURCE, "commit": SOURCE["source_commit"]},
        "verdict": "passed",
    }
    payload = {
        "artifact_id": canonical_json.canonical_json_sha256_strict(unsigned),
        **unsigned,
    }
    payload["seal_sha256"] = canonical_json.canonical_json_sha256_strict(payload)
    path.write_bytes(canonical_json.canonical_json_bytes_strict(payload) + b"\n")


def _fixture(tmp_path: Path) -> tuple[Path, Path, Path]:
    root = tmp_path / "root"
    with okf_writer_lock(root):
        pass
    directory = store.distillation_dir(root)
    directory.mkdir()
    workset = directory / "ox-workset.sqlite3"
    with sqlite3.connect(workset) as connection:
        connection.executescript(
            """
            CREATE TABLE work_items (
                sequence INTEGER PRIMARY KEY AUTOINCREMENT, work_id TEXT NOT NULL UNIQUE,
                kind TEXT NOT NULL, payload_ref TEXT NOT NULL, payload_digest TEXT NOT NULL,
                temporal_split_json TEXT NOT NULL, provenance_json TEXT NOT NULL,
                priority INTEGER NOT NULL, watermark_json TEXT NOT NULL, state TEXT NOT NULL,
                attempt_count INTEGER NOT NULL, last_error_class TEXT NOT NULL, lease_id TEXT,
                lease_owner TEXT, lease_expires_at REAL, completion_ref TEXT NOT NULL,
                completion_digest TEXT NOT NULL, created_at REAL NOT NULL, updated_at REAL NOT NULL
            );
            CREATE INDEX work_items_claim_order ON work_items(kind, state, priority DESC, sequence ASC);
            CREATE INDEX work_items_expiry ON work_items(state, lease_expires_at);
            CREATE TABLE workset_state (key TEXT PRIMARY KEY, value_json TEXT NOT NULL);
            """
        )
        connection.execute(
            "INSERT INTO work_items VALUES (1, 'w1', 'ox', 'candidate-snapshot:w1', ?, '{}', '{}', 0, '{}', "
            "'ready', 0, '', NULL, NULL, NULL, '', '', 1, 1)",
            ("a" * 64,),
        )
    workset.with_name(workset.name + "-wal").touch()
    (directory / "distillation-worker.lock").touch()
    (directory / "candidate-ledger.jsonl").write_text("candidate\n")
    store.write_sealed_state(
        directory / "candidate-ledger.jsonl.head.json",
        {
            "head_sha256": "candidate-head",
            "records": 1,
            "file_state": {"size_bytes": 10},
        },
    )
    (root / "config.toml").write_text("[recall_distillation]\nenabled=true\n")
    offline = tmp_path / "offline.json"
    _offline_evidence(root, offline)
    r0 = tmp_path / "r0.json"
    store.write_sealed_state(
        r0,
        {
            "kind": "r0",
            "production": {
                "ledgers": {
                    "candidate-ledger.jsonl": {
                        "records": 1,
                        "head_sha256": "candidate-head",
                    }
                }
            },
        },
    )
    return root, offline, r0


@pytest.fixture(autouse=True)
def _binding_and_anchor(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        CUTOVER.distill, "ox_alpha_source_binding", lambda: dict(SOURCE)
    )
    monkeypatch.setattr(
        CUTOVER.distill, "_r4_critical_module_sha256", lambda: {"x": "y"}
    )
    # R0's fixed historical artifact id is intentionally unrelated to this
    # isolated queue test; assert the real cutover calls the anchor seam.
    monkeypatch.setattr(
        CUTOVER.distill,
        "bootstrap_r4_candidate_anchor",
        lambda **_kwargs: {"anchor": True},
    )
    monkeypatch.setattr(
        CUTOVER,
        "_validate_r0_evidence",
        lambda _path: {"artifact_id": distill.R4_R0_EVIDENCE_ID},
    )


def _run(root: Path, offline: Path, r0: Path, **kwargs: object) -> dict[str, object]:
    return CUTOVER.cutover(
        root=root,
        offline_evidence=offline,
        r0_evidence=r0,
        source_commit=SOURCE["source_commit"],
        output=None,
        execute=True,
        **kwargs,
    )


def test_archives_legacy_and_atomically_installs_verified_empty(tmp_path: Path) -> None:
    root, offline, r0 = _fixture(tmp_path)
    before = (root / "config.toml").read_bytes()

    result = _run(root, offline, r0)

    archive = Path(str(result["archive"]))
    assert (archive / "snapshot" / "ox-workset.sqlite3").exists()
    assert (
        CUTOVER._sqlite_identity(archive / "snapshot" / "ox-workset.sqlite3")["rows"]
        == 1
    )
    fresh = CUTOVER._fresh_identity(store.distillation_dir(root) / "ox-workset.sqlite3")
    assert fresh["audit"]["status"] == "verified-empty"
    assert (root / "config.toml").read_bytes() == before
    assert result["provider_calls"] == 0 and result["ox_enabled"] is False
    assert result["production_certification"] is False


def test_completed_cutover_output_seals_noncertifying_scope(tmp_path: Path) -> None:
    root, offline, r0 = _fixture(tmp_path)
    output = tmp_path / "cutover-completed.json"

    result = CUTOVER.cutover(
        root=root,
        offline_evidence=offline,
        r0_evidence=r0,
        source_commit=SOURCE["source_commit"],
        output=output,
        execute=True,
    )

    assert result["verdict"] == "completed"
    _assert_preflight_output(result, output)


def test_default_preflight_does_not_adopt_the_legacy_root(tmp_path: Path) -> None:
    root, offline, r0 = _fixture(tmp_path)

    result = CUTOVER.cutover(
        root=root,
        offline_evidence=offline,
        r0_evidence=r0,
        source_commit=SOURCE["source_commit"],
        output=None,
    )

    assert result["verdict"] == "preflight"
    assert result["authority"] is None
    assert not list(root.glob("*.json"))


def _assert_preflight_output(result: dict[str, object], output: Path) -> None:
    assert result["provider_calls"] == 0
    assert result["ox_enabled"] is False
    assert result["production_certification"] is False
    receipt = CUTOVER._read_sealed_regular(output, schema=CUTOVER.CUTOVER_SCHEMA)
    assert result["output"] == receipt
    assert {
        key: value
        for key, value in receipt.items()
        if key not in {"schema", "namespace", "artifact_id", "seal_sha256"}
    } == {key: value for key, value in result.items() if key != "output"}


def test_preflight_variants_persist_the_required_output(tmp_path: Path) -> None:
    root, offline, r0 = _fixture(tmp_path)
    initial_output = tmp_path / "initial-preflight.json"
    initial = CUTOVER.cutover(
        root=root,
        offline_evidence=offline,
        r0_evidence=r0,
        source_commit=SOURCE["source_commit"],
        output=initial_output,
    )
    assert initial["verdict"] == "preflight"
    _assert_preflight_output(initial, initial_output)

    completed = _run(root, offline, r0)
    resume_output = tmp_path / "resume-preflight.json"
    resumed = CUTOVER.cutover(
        root=root,
        offline_evidence=offline,
        r0_evidence=r0,
        source_commit=SOURCE["source_commit"],
        output=resume_output,
    )
    assert resumed["verdict"] == "resume-preflight"
    _assert_preflight_output(resumed, resume_output)

    rollback_output = tmp_path / "rollback-preflight.json"
    rolled = CUTOVER.rollback(
        root=root,
        operation_id=Path(str(completed["archive"])).name,
        output=rollback_output,
    )
    assert rolled["verdict"] == "rollback-preflight"
    _assert_preflight_output(rolled, rollback_output)


def test_cli_preflight_persists_required_output(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    root, offline, r0 = _fixture(tmp_path)
    output = tmp_path / "cli-preflight.json"

    exit_code = CUTOVER.main(
        [
            "--root",
            str(root),
            "--offline-evidence",
            str(offline),
            "--r0-evidence",
            str(r0),
            "--source-commit",
            SOURCE["source_commit"],
            "--output",
            str(output),
        ]
    )

    assert exit_code == 0
    result = json.loads(capsys.readouterr().out)
    assert result["verdict"] == "preflight"
    _assert_preflight_output(result, output)


def test_output_inside_root_is_rejected_for_preflight_and_execute(
    tmp_path: Path,
) -> None:
    root, offline, r0 = _fixture(tmp_path)
    output = root / "output.json"

    for execute in (False, True):
        with pytest.raises(CUTOVER.CutoverError, match="outside"):
            CUTOVER.cutover(
                root=root,
                offline_evidence=offline,
                r0_evidence=r0,
                source_commit=SOURCE["source_commit"],
                output=output,
                execute=execute,
            )
    assert not output.exists()


@pytest.mark.parametrize("input_name", ("offline", "r0"))
def test_output_overlapping_cutover_input_is_rejected_before_mutation(
    tmp_path: Path, input_name: str
) -> None:
    root, offline, r0 = _fixture(tmp_path)
    output = offline if input_name == "offline" else r0
    workset = store.distillation_dir(root) / "ox-workset.sqlite3"
    before = CUTOVER._sha256(workset)
    input_before = output.read_bytes()

    for execute in (False, True):
        with pytest.raises(CUTOVER.CutoverError, match="overlap"):
            CUTOVER.cutover(
                root=root,
                offline_evidence=offline,
                r0_evidence=r0,
                source_commit=SOURCE["source_commit"],
                output=output,
                execute=execute,
            )
    assert CUTOVER._sha256(workset) == before
    assert output.read_bytes() == input_before


@pytest.mark.parametrize("point", ("before-swap", "after-swap"))
def test_crash_is_idempotently_resumed(tmp_path: Path, point: str) -> None:
    root, offline, r0 = _fixture(tmp_path)

    with pytest.raises(RuntimeError, match=point):
        _run(
            root,
            offline,
            r0,
            fault=lambda seen: (
                (_ for _ in ()).throw(RuntimeError(seen)) if seen == point else None
            ),
        )

    result = _run(root, offline, r0)
    assert result["verdict"] == "completed"
    assert (
        CUTOVER._fresh_identity(store.distillation_dir(root) / "ox-workset.sqlite3")[
            "audit"
        ]["status"]
        == "verified-empty"
    )


def test_rollback_restores_legacy_identity(tmp_path: Path) -> None:
    root, offline, r0 = _fixture(tmp_path)
    legacy = CUTOVER._sqlite_identity(
        store.distillation_dir(root) / "ox-workset.sqlite3"
    )
    result = _run(root, offline, r0)
    operation_id = Path(str(result["archive"])).name

    rolled = CUTOVER.rollback(
        root=root, operation_id=operation_id, output=None, execute=True
    )

    assert rolled["verdict"] == "rolled-back"
    assert (
        CUTOVER._sqlite_identity(store.distillation_dir(root) / "ox-workset.sqlite3")
        == legacy
    )


def test_idempotent_rollback_persists_function_and_cli_receipts(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    root, offline, r0 = _fixture(tmp_path)
    workset = store.distillation_dir(root) / "ox-workset.sqlite3"
    legacy = CUTOVER._sqlite_identity(workset)
    completed = _run(root, offline, r0)
    operation_id = Path(str(completed["archive"])).name

    first_output = tmp_path / "rollback.json"
    first = CUTOVER.rollback(
        root=root,
        operation_id=operation_id,
        output=first_output,
        execute=True,
    )
    assert first["verdict"] == "rolled-back"
    assert CUTOVER._sqlite_identity(workset) == legacy
    before_retry = CUTOVER._sha256(workset)

    retry_output = tmp_path / "rollback-noop.json"
    retry = CUTOVER.rollback(
        root=root,
        operation_id=operation_id,
        output=retry_output,
        execute=True,
    )
    assert retry["verdict"] == "rollback-noop"
    _assert_preflight_output(retry, retry_output)
    assert CUTOVER._sha256(workset) == before_retry
    assert CUTOVER._sqlite_identity(workset) == legacy

    cli_output = tmp_path / "rollback-noop-cli.json"
    exit_code = CUTOVER.main(
        [
            "--root",
            str(root),
            "--rollback",
            operation_id,
            "--execute",
            "--output",
            str(cli_output),
        ]
    )
    assert exit_code == 0
    cli_result = json.loads(capsys.readouterr().out)
    assert cli_result["verdict"] == "rollback-noop"
    _assert_preflight_output(cli_result, cli_output)
    assert CUTOVER._sha256(workset) == before_retry
    assert CUTOVER._sqlite_identity(workset) == legacy


@pytest.mark.parametrize("fault", ("lease", "wal", "symlink", "sidecar"))
def test_rejects_unsafe_legacy_state(tmp_path: Path, fault: str) -> None:
    root, offline, r0 = _fixture(tmp_path)
    workset = store.distillation_dir(root) / "ox-workset.sqlite3"
    if fault == "lease":
        with sqlite3.connect(workset) as connection:
            connection.execute("UPDATE work_items SET lease_id='x'")
    elif fault == "wal":
        workset.with_name(workset.name + "-wal").write_bytes(b"not-empty")
    elif fault == "symlink":
        workset.unlink()
        workset.symlink_to(tmp_path / "target")
    else:
        workset.with_name(workset.name + "-unknown").write_text("unsafe")
    _offline_evidence(root, offline)

    with pytest.raises((CUTOVER.CutoverError, distill.DistillationError)):
        _run(root, offline, r0)


def test_rejects_busy_worker_after_authority_preflight(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root, offline, r0 = _fixture(tmp_path)
    original = CUTOVER.store.acquire_nonblocking_lock
    calls = 0

    def acquire(path: Path):
        nonlocal calls
        calls += 1
        return original(path) if calls == 1 else None

    monkeypatch.setattr(CUTOVER.store, "acquire_nonblocking_lock", acquire)

    with pytest.raises(CUTOVER.CutoverError, match="worker is busy"):
        _run(root, offline, r0)
    assert not (store.distillation_dir(root) / "workset-archives").exists()


def test_streaming_digest_never_calls_read_bytes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "large"
    path.write_bytes(b"x" * (2 * 1024 * 1024 + 3))
    expected = hashlib.sha256(b"x" * (2 * 1024 * 1024 + 3)).hexdigest()
    monkeypatch.setattr(Path, "read_bytes", lambda _self: pytest.fail("read_bytes"))

    assert CUTOVER._sha256(path) == expected


def test_partial_archive_and_prepared_temp_resume(tmp_path: Path) -> None:
    root, offline, r0 = _fixture(tmp_path)
    with pytest.raises(RuntimeError, match="before-swap"):
        _run(
            root,
            offline,
            r0,
            fault=lambda point: (
                (_ for _ in ()).throw(RuntimeError(point))
                if point == "before-swap"
                else None
            ),
        )
    archive = next((store.distillation_dir(root) / "workset-archives").iterdir())
    (archive / "archive-manifest.json").unlink()

    result = _run(root, offline, r0)

    assert result["verdict"] == "completed"
    assert (
        CUTOVER._fresh_identity(store.distillation_dir(root) / "ox-workset.sqlite3")[
            "audit"
        ]["status"]
        == "verified-empty"
    )


def test_empty_owned_archive_directory_is_resumed(tmp_path: Path) -> None:
    root, offline, r0 = _fixture(tmp_path)
    workset = store.distillation_dir(root) / "ox-workset.sqlite3"
    archive = (
        store.distillation_dir(root)
        / "workset-archives"
        / f"legacy-v1-{CUTOVER._sha256(workset)}"
    )
    archive.mkdir(parents=True)

    result = _run(root, offline, r0)

    assert result["verdict"] == "completed"
    assert (archive / "archive-manifest.json").exists()


def test_empty_snapshot_after_crash_is_rebuilt(tmp_path: Path) -> None:
    root, offline, r0 = _fixture(tmp_path)
    with pytest.raises(RuntimeError, match="after-snapshot-mkdir"):
        _run(
            root,
            offline,
            r0,
            fault=lambda point: (
                (_ for _ in ()).throw(RuntimeError(point))
                if point == "after-snapshot-mkdir"
                else None
            ),
        )

    result = _run(root, offline, r0)

    assert result["verdict"] == "completed"


def test_partial_snapshot_completes_missing_source_sidecars(tmp_path: Path) -> None:
    root, offline, r0 = _fixture(tmp_path)
    workset = store.distillation_dir(root) / "ox-workset.sqlite3"
    workset.with_name(workset.name + "-shm").write_bytes(b"legacy-shm")
    _offline_evidence(root, offline)
    with pytest.raises(RuntimeError, match="before-swap"):
        _run(
            root,
            offline,
            r0,
            fault=lambda point: (
                (_ for _ in ()).throw(RuntimeError(point))
                if point == "before-swap"
                else None
            ),
        )
    archive = next((store.distillation_dir(root) / "workset-archives").iterdir())
    snapshot_shm = archive / "snapshot" / "ox-workset.sqlite3-shm"
    snapshot_shm.unlink()
    (archive / "snapshot" / "ox-workset.sqlite3-wal").unlink()
    (archive / "archive-manifest.json").unlink()

    result = _run(root, offline, r0)

    assert result["verdict"] == "completed"
    assert snapshot_shm.read_bytes() == b"legacy-shm"
    assert not list(store.distillation_dir(root).glob(".ox-workset.sqlite3.r4-*.tmp*"))


def test_manifest_resume_rejects_tampered_snapshot_sidecar(tmp_path: Path) -> None:
    root, offline, r0 = _fixture(tmp_path)
    with pytest.raises(RuntimeError, match="before-swap"):
        _run(
            root,
            offline,
            r0,
            fault=lambda point: (
                (_ for _ in ()).throw(RuntimeError(point))
                if point == "before-swap"
                else None
            ),
        )
    archive = next((store.distillation_dir(root) / "workset-archives").iterdir())
    (archive / "snapshot" / "ox-workset.sqlite3-shm").write_bytes(b"forged")

    with pytest.raises(CUTOVER.CutoverError, match="snapshot|completion"):
        _run(root, offline, r0)


def test_rejects_tampered_prepared_fresh_temp(tmp_path: Path) -> None:
    root, offline, r0 = _fixture(tmp_path)
    with pytest.raises(RuntimeError, match="before-swap"):
        _run(
            root,
            offline,
            r0,
            fault=lambda point: (
                (_ for _ in ()).throw(RuntimeError(point))
                if point == "before-swap"
                else None
            ),
        )
    temp = next(store.distillation_dir(root).glob(".ox-workset.sqlite3.r4-*.tmp"))
    temp.write_bytes(b"tampered")

    with pytest.raises(CUTOVER.CutoverError):
        _run(root, offline, r0)


def test_rejects_orphaned_prepared_sidecar(tmp_path: Path) -> None:
    root, offline, r0 = _fixture(tmp_path)
    workset = store.distillation_dir(root) / "ox-workset.sqlite3"
    temp = workset.with_name(f".{workset.name}.r4-{CUTOVER._sha256(workset)}.tmp")
    temp.with_name(temp.name + "-wal").write_bytes(b"orphan")

    with pytest.raises(CUTOVER.CutoverError, match="orphaned"):
        _run(root, offline, r0)


def test_existing_anchor_missing_field_is_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root, _offline, r0 = _fixture(tmp_path)
    directory = store.distillation_dir(root)
    anchor = directory / distill.R4_CANDIDATE_ANCHOR_FILE
    anchor.write_text("anchor")
    critical = {"x": "y"}
    monkeypatch.setattr(CUTOVER.distill, "_r4_critical_module_sha256", lambda: critical)
    checkpoint = {
        "head_sha256": "d" * 64,
        "records": 1,
        "file_state": {"size_bytes": 10},
    }
    current = {
        "schema": distill.R4_CANDIDATE_ANCHOR_SCHEMA,
        "namespace": "recall-distillation",
        "artifact_id": "a" * 64,
        "seal_sha256": "b" * 64,
        "kind": "r4-candidate-anchor",
        "r0_artifact_id": distill.R4_R0_EVIDENCE_ID,
        "r0_file_sha256": "c" * 64,
        # bootstrap_source_commit intentionally missing
        "candidate_checkpoint": {
            "head_sha256": "d" * 64,
            "records": 1,
            "bytes": 10,
            "file_state": {"size_bytes": 10},
        },
        "critical_module_sha256": critical,
    }
    original = CUTOVER._read_sealed_regular

    def read(path: Path, *, schema: str | None = None):
        if path == anchor:
            return current
        if path.name.endswith("head.json"):
            return checkpoint
        return original(path, schema=schema)

    monkeypatch.setattr(CUTOVER, "_read_sealed_regular", read)
    monkeypatch.setattr(
        CUTOVER,
        "_sha256",
        lambda path: (
            "c" * 64 if path == r0 else hashlib.sha256(path.read_bytes()).hexdigest()
        ),
    )
    monkeypatch.setattr(
        CUTOVER,
        "_validate_r0_evidence",
        lambda _path: {
            "artifact_id": distill.R4_R0_EVIDENCE_ID,
            "production": {
                "ledgers": {
                    "candidate-ledger.jsonl": {
                        "head_sha256": "d" * 64,
                        "records": 1,
                        "bytes": 10,
                    }
                }
            },
        },
    )

    with pytest.raises(CUTOVER.CutoverError, match="anchor"):
        CUTOVER._verify_or_anchor(root, r0, SOURCE)


def test_r0_requires_its_canonical_artifact_id(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _root, _offline, r0 = _fixture(tmp_path)
    payload = {
        "schema": "chronovisor.recall-r0.v1",
        "namespace": "recall-distillation",
        "artifact_id": distill.R4_R0_EVIDENCE_ID,
    }
    payload["seal_sha256"] = canonical_json.canonical_json_sha256_strict(payload)
    monkeypatch.setattr(CUTOVER.store, "read_sealed", lambda *_args, **_kwargs: payload)
    monkeypatch.setattr(CUTOVER, "_validate_r0_evidence", REAL_VALIDATE_R0)

    with pytest.raises(CUTOVER.CutoverError, match="R0"):
        CUTOVER._validate_r0_evidence(r0)


def test_rollback_recovers_after_main_swap_before_sidecars(tmp_path: Path) -> None:
    root, offline, r0 = _fixture(tmp_path)
    result = _run(root, offline, r0)
    operation_id = Path(str(result["archive"])).name

    with pytest.raises(RuntimeError, match="after-rollback-main-swap"):
        CUTOVER.rollback(
            root=root,
            operation_id=operation_id,
            output=None,
            execute=True,
            fault=lambda point: (_ for _ in ()).throw(RuntimeError(point)),
        )
    recovered = CUTOVER.rollback(
        root=root, operation_id=operation_id, output=None, execute=True
    )

    assert recovered["verdict"] == "rollback-noop"
    assert (
        CUTOVER._sqlite_identity(store.distillation_dir(root) / "ox-workset.sqlite3")[
            "rows"
        ]
        == 1
    )


def test_rollback_rejects_resealed_manifest_with_wrong_snapshot(tmp_path: Path) -> None:
    root, offline, r0 = _fixture(tmp_path)
    result = _run(root, offline, r0)
    archive = Path(str(result["archive"]))
    manifest_path = archive / "archive-manifest.json"
    original = json.loads(manifest_path.read_text())
    unsigned = {
        key: value
        for key, value in original.items()
        if key not in {"artifact_id", "seal_sha256", "schema", "namespace"}
    }
    unsigned["snapshot"] = {
        "main": original["snapshot"]["main"],
        "-wal": None,
        "-shm": {"sha256": "0" * 64, "size_bytes": 1},
    }
    forged = CUTOVER._sealed(unsigned, schema=CUTOVER.ARCHIVE_SCHEMA)
    os.chmod(manifest_path, 0o600)
    manifest_path.write_bytes(
        canonical_json.canonical_json_bytes_strict(forged) + b"\n"
    )
    fresh = store.distillation_dir(root) / "ox-workset.sqlite3"
    before = CUTOVER._sha256(fresh)

    with pytest.raises(CUTOVER.CutoverError, match="snapshot|completion"):
        CUTOVER.rollback(
            root=root, operation_id=archive.name, output=None, execute=True
        )
    assert CUTOVER._sha256(fresh) == before


def test_completion_binding_rejects_preflight_and_rollback(tmp_path: Path) -> None:
    root, offline, r0 = _fixture(tmp_path)
    result = _run(root, offline, r0)
    archive = Path(str(result["archive"]))
    completion_path = archive / "cutover-completion.json"
    original = json.loads(completion_path.read_text())
    unsigned = {
        key: value
        for key, value in original.items()
        if key not in {"artifact_id", "seal_sha256", "schema", "namespace"}
    }
    unsigned["legacy_main_sha256"] = "0" * 64
    forged = CUTOVER._sealed(unsigned, schema=CUTOVER.CUTOVER_SCHEMA)
    os.chmod(completion_path, 0o600)
    completion_path.write_bytes(
        canonical_json.canonical_json_bytes_strict(forged) + b"\n"
    )
    fresh = store.distillation_dir(root) / "ox-workset.sqlite3"
    before = CUTOVER._sha256(fresh)

    with pytest.raises(CUTOVER.CutoverError, match="completion"):
        CUTOVER.cutover(
            root=root,
            offline_evidence=offline,
            r0_evidence=r0,
            source_commit=SOURCE["source_commit"],
            output=None,
            execute=False,
        )
    with pytest.raises(CUTOVER.CutoverError, match="completion"):
        CUTOVER.rollback(
            root=root, operation_id=archive.name, output=None, execute=False
        )
    with pytest.raises(CUTOVER.CutoverError, match="completion"):
        CUTOVER.rollback(
            root=root, operation_id=archive.name, output=None, execute=True
        )
    assert CUTOVER._sha256(fresh) == before


@pytest.mark.parametrize("name", ("archive-manifest.json", "cutover-completion.json"))
def test_rollback_output_overlapping_archive_input_is_rejected(
    tmp_path: Path, name: str
) -> None:
    root, offline, r0 = _fixture(tmp_path)
    result = _run(root, offline, r0)
    archive = Path(str(result["archive"]))
    fresh = store.distillation_dir(root) / "ox-workset.sqlite3"
    before = CUTOVER._sha256(fresh)

    with pytest.raises(CUTOVER.CutoverError):
        CUTOVER.rollback(
            root=root, operation_id=archive.name, output=archive / name, execute=True
        )
    assert CUTOVER._sha256(fresh) == before


def test_resume_and_rollback_reject_unexpected_archive_entry(tmp_path: Path) -> None:
    root, offline, r0 = _fixture(tmp_path)
    result = _run(root, offline, r0)
    archive = Path(str(result["archive"]))
    (archive / "unexpected").write_text("unsafe")
    fresh = store.distillation_dir(root) / "ox-workset.sqlite3"
    before = CUTOVER._sha256(fresh)

    with pytest.raises(CUTOVER.CutoverError, match="shape"):
        CUTOVER.cutover(
            root=root,
            offline_evidence=offline,
            r0_evidence=r0,
            source_commit=SOURCE["source_commit"],
            output=None,
            execute=False,
        )
    with pytest.raises(CUTOVER.CutoverError, match="shape"):
        _run(root, offline, r0)
    with pytest.raises(CUTOVER.CutoverError, match="shape"):
        CUTOVER.rollback(
            root=root, operation_id=archive.name, output=None, execute=False
        )
    with pytest.raises(CUTOVER.CutoverError, match="shape"):
        CUTOVER.rollback(
            root=root, operation_id=archive.name, output=None, execute=True
        )
    assert CUTOVER._sha256(fresh) == before


def test_manifest_resume_rejects_symlinked_worker_lock(tmp_path: Path) -> None:
    root, offline, r0 = _fixture(tmp_path)
    result = _run(root, offline, r0)
    workset = store.distillation_dir(root) / "ox-workset.sqlite3"
    before = CUTOVER._sha256(workset)
    lock = store.distillation_dir(root) / "distillation-worker.lock"
    lock.unlink()
    lock.symlink_to(tmp_path / "outside-lock")

    with pytest.raises(CUTOVER.CutoverError, match="unsafe file"):
        _run(root, offline, r0)
    assert CUTOVER._sha256(workset) == before == result["manifest"]["fresh_main_sha256"]


def test_existing_anchor_rejects_symlinked_checkpoint(tmp_path: Path) -> None:
    root, _offline, r0 = _fixture(tmp_path)
    directory = store.distillation_dir(root)
    anchor = directory / distill.R4_CANDIDATE_ANCHOR_FILE
    anchor.write_bytes(
        canonical_json.canonical_json_bytes_strict(
            CUTOVER._sealed(
                {"kind": "test-anchor"}, schema=distill.R4_CANDIDATE_ANCHOR_SCHEMA
            )
        )
        + b"\n"
    )
    checkpoint = directory / "candidate-ledger.jsonl.head.json"
    redirected = tmp_path / "redirected-checkpoint.json"
    store.write_sealed_state(redirected, {"head_sha256": "d" * 64, "records": 1})
    checkpoint.unlink()
    checkpoint.symlink_to(redirected)

    with pytest.raises(CUTOVER.CutoverError, match="unsafe file"):
        CUTOVER._verify_or_anchor(root, r0, SOURCE)


def test_new_anchor_rejects_symlinked_checkpoint_before_bootstrap(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root, _offline, r0 = _fixture(tmp_path)
    directory = store.distillation_dir(root)
    checkpoint = directory / "candidate-ledger.jsonl.head.json"
    redirected = tmp_path / "redirected-checkpoint.json"
    store.write_sealed_state(redirected, {"head_sha256": "d" * 64, "records": 1})
    checkpoint.unlink()
    checkpoint.symlink_to(redirected)
    monkeypatch.setattr(
        CUTOVER.distill,
        "bootstrap_r4_candidate_anchor",
        lambda **_kwargs: pytest.fail("bootstrap must not follow checkpoint symlink"),
    )

    with pytest.raises(CUTOVER.CutoverError, match="unsafe file"):
        CUTOVER._verify_or_anchor(root, r0, SOURCE)


def test_existing_anchor_rejects_same_size_candidate_replacement(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root, _offline, r0 = _fixture(tmp_path)
    directory = store.distillation_dir(root)
    candidate = directory / "candidate-ledger.jsonl"
    observed = candidate.stat()
    file_state = {
        "size_bytes": observed.st_size,
        "st_dev": observed.st_dev,
        "st_ino": observed.st_ino,
        "st_mtime_ns": observed.st_mtime_ns,
        "st_ctime_ns": observed.st_ctime_ns,
    }
    checkpoint = directory / "candidate-ledger.jsonl.head.json"
    store.write_sealed_state(
        checkpoint,
        {"head_sha256": "d" * 64, "records": 1, "file_state": file_state},
    )
    r0_payload = {
        "artifact_id": distill.R4_R0_EVIDENCE_ID,
        "production": {
            "ledgers": {
                "candidate-ledger.jsonl": {
                    "head_sha256": "d" * 64,
                    "records": 1,
                    "bytes": observed.st_size,
                    "file_state": file_state,
                }
            }
        },
    }
    monkeypatch.setattr(CUTOVER, "_validate_r0_evidence", lambda _path: r0_payload)
    anchor = directory / distill.R4_CANDIDATE_ANCHOR_FILE
    anchor_payload = CUTOVER._sealed(
        {
            "kind": "r4-candidate-anchor",
            "r0_artifact_id": distill.R4_R0_EVIDENCE_ID,
            "r0_file_sha256": CUTOVER._sha256(r0),
            "bootstrap_source_commit": SOURCE["source_commit"],
            "candidate_checkpoint": {
                "head_sha256": "d" * 64,
                "records": 1,
                "bytes": observed.st_size,
                "file_state": file_state,
            },
            "critical_module_sha256": {"x": "y"},
        },
        schema=distill.R4_CANDIDATE_ANCHOR_SCHEMA,
    )
    anchor.write_bytes(
        canonical_json.canonical_json_bytes_strict(anchor_payload) + b"\n"
    )
    binding = CUTOVER._candidate_binding(root)
    # The R0/Checkpoint state is from the production source.  An APFS clone
    # deliberately has a different inode/ctime, but is accepted before change.
    CUTOVER._verify_or_anchor(root, r0, SOURCE)
    replacement = tmp_path / "same-bytes-replacement"
    replacement.write_bytes(candidate.read_bytes())
    os.replace(replacement, candidate)
    assert candidate.stat().st_size == observed.st_size
    assert CUTOVER._sha256(candidate) == binding["sha256"]

    with pytest.raises(CUTOVER.CutoverError, match="candidate changed"):
        CUTOVER._validate_candidate_binding(root, binding)


def test_locked_config_ox_race_rejects_before_archive(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root, offline, r0 = _fixture(tmp_path)
    original = CUTOVER._cutover_locks

    @contextmanager
    def changed_config(locked_root: Path):
        with original(locked_root):
            (locked_root / "config.toml").write_text(
                "[recall.distillation]\nox_enabled=true\n"
            )
            yield

    monkeypatch.setattr(CUTOVER, "_cutover_locks", changed_config)

    with pytest.raises(CUTOVER.CutoverError, match="OX"):
        _run(root, offline, r0)
    assert not (store.distillation_dir(root) / "workset-archives").exists()


def test_locked_source_binding_race_rejects_before_archive(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root, offline, r0 = _fixture(tmp_path)
    initial = _run(root, offline, r0)
    fresh = store.distillation_dir(root) / "ox-workset.sqlite3"
    before = CUTOVER._sha256(fresh)
    calls = 0

    def binding() -> dict[str, str]:
        nonlocal calls
        calls += 1
        if calls == 1:
            return dict(SOURCE)
        changed = dict(SOURCE)
        changed["source_tree_sha256"] = "e" * 64
        return changed

    monkeypatch.setattr(CUTOVER.distill, "ox_alpha_source_binding", binding)

    with pytest.raises(CUTOVER.CutoverError, match="source binding changed"):
        _run(root, offline, r0)
    assert CUTOVER._sha256(fresh) == before == initial["manifest"]["fresh_main_sha256"]


def test_anchor_artifact_id_must_be_canonical() -> None:
    artifact = CUTOVER._sealed({"kind": "r4-candidate-anchor"}, schema="test")
    artifact["artifact_id"] = "0" * 64

    assert not CUTOVER._canonical_artifact_id(artifact)


def test_clone_durability_and_fresh_reopen(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root, offline, r0 = _fixture(tmp_path)
    calls: list[Path] = []
    original = CUTOVER._fsync_regular
    monkeypatch.setattr(
        CUTOVER, "_fsync_regular", lambda path: (calls.append(path), original(path))[1]
    )

    _run(root, offline, r0)

    fresh = store.distillation_dir(root) / "ox-workset.sqlite3"
    assert any("snapshot" in str(path) for path in calls)
    assert (
        DistillationWorkset(fresh, migrate=False).audit_transition_receipts()["status"]
        == "verified-empty"
    )
