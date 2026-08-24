from __future__ import annotations

import argparse
import contextlib
import itertools
import os
import socket
import sqlite3
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from chronovisor.recall import recall_distillation_catalog as catalog
from chronovisor.recall import recall_distillation_store as store

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))
import recall_r1_harness as harness  # noqa: E402


def _rusage(resident: int = 10) -> dict[str, int | str]:
    return {
        "rusage_uuid": "a" * 32,
        "resident_bytes": resident,
        "footprint_bytes": resident * 2,
        "disk_read_bytes": 100,
        "disk_write_bytes": 200,
    }


def _patch_rusage(monkeypatch: pytest.MonkeyPatch, count: int = 2) -> None:
    values = itertools.cycle(_rusage(10 + index) for index in range(count))
    monkeypatch.setattr(harness, "_proc_pid_rusage_v2", lambda: next(values))


def _patch_run_dependencies(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, *, bad_stage: bool = False
) -> argparse.Namespace:
    production = tmp_path / "production"
    source = tmp_path / "source"
    clone = tmp_path / "clone"
    output = tmp_path / "evidence"
    production.mkdir()
    source.mkdir()
    clone.mkdir()
    monkeypatch.setattr(harness.sys, "platform", "darwin")

    class Parity:
        @staticmethod
        def _runtime_identity(_root: Path, _commit: str) -> dict[str, str]:
            return {"source_commit": "a" * 40, "source_tree": "b" * 40}

    distill = SimpleNamespace(_default_workers=lambda: None)
    baseline = {
        "ledgers": {
            name: {
                **dict(value),
                "file_state": {
                    "size_bytes": value["bytes"],
                    "st_dev": index + 1,
                    "st_ino": index + 10,
                    "st_mtime_ns": index + 20,
                    "st_ctime_ns": index + 30,
                },
            }
            for index, (name, value) in enumerate(harness.EXPECTED_BASELINE.items())
        },
        "raw_watermark": "c" * 64,
        "fts": {
            "content_sha256": "d" * 64,
            "atom_count": 2,
            "fts_count": 2,
            "checkpoint_seal_sha256": "g" * 64,
            "file_state": {
                "size_bytes": 3,
                "st_dev": 1,
                "st_ino": 2,
                "st_mtime_ns": 4,
                "st_ctime_ns": 4,
            },
        },
    }
    anchor = {
        "artifact_id": harness.R0_EVIDENCE_ID,
        "file_sha256": "e" * 64,
        "seal_sha256": "f" * 64,
        "production": baseline,
    }

    monkeypatch.setattr(
        harness,
        "_load",
        lambda _source: (Parity(), distill, store, None, None),
    )
    env_values: dict[str, str] = {}
    real_env = harness._env

    @contextlib.contextmanager
    def capture_env(values: dict[str, str]):
        env_values.update(values)
        with real_env(values):
            yield

    monkeypatch.setattr(harness, "_env", capture_env)
    monkeypatch.setattr(
        harness,
        "_production_snapshot",
        lambda *_args: {
            "ledgers": {
                name: dict(value) for name, value in baseline["ledgers"].items()
            },
            "raw_watermark": baseline["raw_watermark"],
            "fts": baseline["fts"],
        },
    )
    monkeypatch.setattr(harness, "_load_r0_anchor", lambda *_args: anchor)
    monkeypatch.setattr(
        harness, "_head", lambda *_args: {"records": 0, "head_sha256": ""}
    )

    @contextlib.contextmanager
    def fake_clone_context(
        _production: Path,
        evidence: list[dict[str, object]] | None = None,
        _protected_roots: tuple[Path, ...] = (),
    ):
        if evidence is not None:
            evidence.append(
                {
                    "source_filesystem": "apfs",
                    "destination_filesystem": "apfs",
                    "same_volume": True,
                    "copy_backend": "copyfile(3)",
                    "copy_flags": harness.COPYFILE_CLONE_FORCE,
                    "clone_force": True,
                    "files_cloned": 1,
                    "clone_filesystem": "apfs",
                    "clone_copy_verified": True,
                }
            )
        yield clone

    monkeypatch.setattr(harness, "_clone_context", fake_clone_context)

    def fake_steady(
        _store: object, _root: Path, name: str, _payload: object
    ) -> dict[str, object]:
        if bad_stage:
            socket.create_connection(("127.0.0.1", 1))
        return {"name": name, "measurement": {"status": "ok"}}

    monkeypatch.setattr(harness, "_steady_ledger", fake_steady)
    monkeypatch.setattr(
        harness, "_label_projection_bootstrap", lambda *_args: {"bootstrap": True}
    )
    monkeypatch.setattr(
        harness, "_candidate_hot_path", lambda *_args: {"candidate": True}
    )
    monkeypatch.setattr(
        harness,
        "_recovery",
        lambda *_args: {"before": {}, "after": {}, "measurement": {"status": "ok"}},
    )
    monkeypatch.setattr(
        harness,
        "_unique_duplicate",
        lambda *_args, **_kwargs: {"measurement": {"status": "ok"}},
    )
    monkeypatch.setattr(
        harness, "_unique_crash", lambda *_args: {"measurement": {"status": "ok"}}
    )
    args = argparse.Namespace(
        production_root=production,
        source_root=source,
        source_commit="a" * 40,
        dashboard_url="http://127.0.0.1:1",
        output=output,
    )
    args._r0_anchor = anchor
    args._env_values = env_values
    return args


def test_measure_records_disk_wall_and_rss(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_rusage(monkeypatch)

    measured = harness._measure("unit", lambda: {"ok": True})

    assert measured["status"] == "ok"
    assert measured["value"] == {"ok": True}
    assert measured["metrics"]["disk_read_bytes"] == 0
    assert measured["metrics"]["disk_write_bytes"] == 0
    assert measured["metrics"]["resident_peak_bytes"] == 11
    assert measured["metrics"]["rusage_sample_count"] >= 2
    assert measured["metrics"]["wall_time_ns"] >= 0


def test_measure_samples_interior_peak(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = itertools.count()

    def rusage() -> dict[str, int | str]:
        index = next(calls)
        return _rusage(100 if 0 < index < 4 else 10)

    monkeypatch.setattr(harness, "_proc_pid_rusage_v2", rusage)
    monkeypatch.setattr(harness, "RUSAGE_SAMPLE_INTERVAL_SECONDS", 0.001)

    measured = harness._measure("slow", lambda: time.sleep(0.02))

    assert measured["metrics"]["resident_peak_bytes"] >= 100
    assert measured["metrics"]["rusage_sample_count"] >= 3


def test_body_guard_rejects_old_prefix_and_records_tail(tmp_path: Path) -> None:
    path = tmp_path / "ledger.jsonl"
    path.write_bytes(b"prefix\ntail\n")
    size = path.stat().st_size

    with harness._body_guard(path, allowed_ranges=[(size - 1, 1)]) as guard:
        with path.open("rb") as handle:
            handle.seek(size - 1)
            assert handle.read(1) == b"\n"
        with pytest.raises(harness.R1Error, match="escaped"):
            with path.open("rb") as handle:
                handle.read(1)

    evidence = harness._guard_evidence(guard)
    assert evidence["bytes_read"] == 1
    assert evidence["old_prefix_bytes"] == 0
    assert evidence["old_prefix_scanned"] is False


def test_label_health_readonly_does_not_change_file_state(tmp_path: Path) -> None:
    path = store.distillation_dir(tmp_path) / store.LABEL_LEDGER_FILE
    store.append_chain_batch(
        path,
        [
            {"authority": "teacher-only", "assignment": {"probe": True}},
            {"authority": "verified"},
        ],
    )
    before = path.stat().st_mtime_ns, path.stat().st_ctime_ns, path.stat().st_size

    health = harness._label_health_readonly(store, path)

    after = path.stat().st_mtime_ns, path.stat().st_ctime_ns, path.stat().st_size
    assert before == after
    assert health["records"] == 2
    assert health["counts"] == {
        "teacher_only": 1,
        "verified_truth": 1,
        "probe_not_truth": 1,
    }


def test_partial_tail_recovery_is_measured_without_duplicate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = store.distillation_dir(tmp_path) / "candidate-ledger.jsonl"
    store.append_chain(path, {"index": 0})
    _patch_rusage(monkeypatch)

    result = harness._recovery(store, tmp_path, path.name, harness._append_partial)

    assert result["measurement"]["status"] == "ok"
    assert result["measurement"]["read_guard"]["old_prefix_scanned"] is True
    assert result["before"]["head"] == result["after"]["head"]
    assert (
        result["before"]["file_state"]["size_bytes"]
        == result["after"]["file_state"]["size_bytes"]
    )


def test_stale_checkpoint_recovery_preserves_exact_head_and_count(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = store.distillation_dir(tmp_path) / "candidate-ledger.jsonl"
    store.append_chain(path, {"index": 0})
    _patch_rusage(monkeypatch)

    result = harness._recovery(
        store,
        tmp_path,
        path.name,
        lambda ledger, head: harness._mismatch_checkpoint(store, ledger, head),
    )

    assert result["injection"]["kind"] == "stale_checkpoint_file_state"
    assert result["before"]["head"] == result["after"]["head"]
    assert result["before"]["head"]["records"] == 1


def test_steady_append_write_guard_is_native_append(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = store.distillation_dir(tmp_path) / "rally-manifest.jsonl"
    store.append_chain(path, {"index": 0})
    _patch_rusage(monkeypatch)

    result = harness._steady_ledger(store, tmp_path, path.name, {"kind": "r1-probe"})
    write_guard = result["measurement"]["write_guard"]

    assert write_guard["native_o_append"] is True
    assert write_guard["old_prefix_write_bytes"] == 0
    assert write_guard["append_bytes"] == (
        result["after"]["file_state"]["size_bytes"]
        - result["before"]["file_state"]["size_bytes"]
    )
    assert write_guard["inode_before"] == write_guard["inode_after"]


def test_write_guard_rejects_non_append_high_level_mutation(tmp_path: Path) -> None:
    path = tmp_path / "ledger.jsonl"
    path.write_bytes(b"seed\n")

    with pytest.raises(harness.R1Error, match="append-only"):
        with harness._write_guard(path):
            path.open("r+b")
    assert path.read_bytes() == b"seed\n"


def test_label_projection_bootstrap_repairs_missing_clone_projection(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = store.distillation_dir(tmp_path) / store.LABEL_LEDGER_FILE
    store.append_chain(path, {"authority": "verified"})
    projection = store._label_health_projection_path(path)
    projection.unlink()
    _patch_rusage(monkeypatch)

    result = harness._label_projection_bootstrap(store, tmp_path)

    assert result["before"] is None
    assert result["after"] is not None
    assert result["value"]["label_records"] == 1
    assert result["measurement"]["status"] == "ok"


def test_apfs_preflight_rejects_non_apfs_before_clone(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    destination_parent = tmp_path / "clone-temp"
    destination_parent.mkdir()
    production = tmp_path / "production"
    production.mkdir()
    monkeypatch.setattr(harness.tempfile, "gettempdir", lambda: str(destination_parent))
    monkeypatch.setattr(harness.sys, "platform", "darwin")
    monkeypatch.setattr(harness, "_filesystem_type", lambda _path: "hfs")
    clone_calls: list[Path] = []

    def unexpected_clone(path: Path) -> tuple[Path, bool, dict[str, object]]:
        clone_calls.append(path)
        raise AssertionError("clone must not start before APFS preflight")

    monkeypatch.setattr(harness, "_forced_clone", unexpected_clone)

    with pytest.raises(harness.R1Error, match="APFS"):
        with harness._clone_context(production):
            pass
    assert clone_calls == []


def test_bytecode_suppression_is_process_and_runtime_env_contract() -> None:
    assert harness.sys.dont_write_bytecode is True
    previous = os.environ.get("PYTHONDONTWRITEBYTECODE")
    with harness._env({"PYTHONDONTWRITEBYTECODE": "1"}):
        assert os.environ["PYTHONDONTWRITEBYTECODE"] == "1"
    assert os.environ.get("PYTHONDONTWRITEBYTECODE") == previous


def test_run_rejects_reverse_parent_overlap_before_clone(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    args = _patch_run_dependencies(monkeypatch, tmp_path)
    args.output = tmp_path

    with pytest.raises(harness.R1Error, match="output overlaps protected root"):
        harness._run(args)


def test_clone_preflight_rejects_temp_parent_collision_in_both_directions(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    production = tmp_path / "production"
    production.mkdir()
    destination_parent = tmp_path / "temp-parent"
    destination_parent.mkdir()
    monkeypatch.setattr(harness.sys, "platform", "darwin")
    monkeypatch.setattr(harness.tempfile, "gettempdir", lambda: str(destination_parent))

    with pytest.raises(harness.R1Error, match="clone temp destination"):
        harness._apfs_clone_preflight(production, (tmp_path,))

    nested_production = destination_parent / "nested-production"
    nested_production.mkdir()
    with pytest.raises(harness.R1Error, match="clone temp destination"):
        harness._apfs_clone_preflight(nested_production)


def _clone_fixture(root: Path) -> Path:
    production = root / "production"
    (production / "raw").mkdir(parents=True)
    (production / "runtime" / "recall-distillation").mkdir(parents=True)
    (production / "recall").mkdir()
    (production / "raw" / "event.bin").write_bytes(b"raw")
    (production / "runtime" / "recall-distillation" / "state.json").write_text(
        "state", encoding="utf-8"
    )
    (production / "recall" / "recall-log.jsonl").write_text("log", encoding="utf-8")
    return production


def _patch_clone_tempdir(monkeypatch: pytest.MonkeyPatch, root: Path) -> None:
    destination_parent = root / "clone-temp"
    destination_parent.mkdir()
    monkeypatch.setattr(harness.tempfile, "gettempdir", lambda: str(destination_parent))


def test_forced_clone_requires_each_copyfile_call(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    production = _clone_fixture(tmp_path)
    _patch_clone_tempdir(monkeypatch, tmp_path)
    monkeypatch.setattr(harness.sys, "platform", "darwin")
    calls: list[tuple[Path, Path, int]] = []

    def fake_copy(source: Path, destination: Path, flags: int) -> None:
        calls.append((source, destination, flags))
        if len(calls) == 2:
            raise harness.R1Error("mock clone failure")
        destination.write_bytes(source.read_bytes())

    monkeypatch.setattr(harness, "_copyfile_clone", fake_copy)
    clone_root: list[Path] = []

    def capture_copy(source: Path, destination: Path, flags: int) -> None:
        clone_root.append(destination.parents[1])
        fake_copy(source, destination, flags)

    monkeypatch.setattr(harness, "_copyfile_clone", capture_copy)
    with pytest.raises(harness.R1Error, match="mock clone failure"):
        harness._forced_clone(production)
    assert calls
    assert all(flags & harness.COPYFILE_CLONE_FORCE for _, _, flags in calls)
    assert clone_root and not clone_root[0].exists()


def test_forced_clone_success_records_copyfile_backend(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    production = _clone_fixture(tmp_path)
    _patch_clone_tempdir(monkeypatch, tmp_path)
    monkeypatch.setattr(harness.sys, "platform", "darwin")
    calls: list[tuple[Path, Path, int]] = []

    def fake_copy(source: Path, destination: Path, flags: int) -> None:
        calls.append((source, destination, flags))
        destination.write_bytes(source.read_bytes())

    monkeypatch.setattr(harness, "_copyfile_clone", fake_copy)
    clone, temporary, evidence = harness._forced_clone(production)
    try:
        assert temporary is True
        assert evidence["copy_backend"] == "copyfile(3)"
        assert evidence["clone_force"] is True
        assert evidence["files_cloned"] == len(calls) == 3
        assert all(flags & harness.COPYFILE_CLONE_FORCE for _, _, flags in calls)
    finally:
        harness._cleanup_clone(clone)


def test_stage_guards_reject_provider_and_socket_calls() -> None:
    distill = SimpleNamespace(_default_workers=lambda: None)

    with harness._stage_guards(distill):
        with pytest.raises(harness.R0Error, match="provider/OX"):
            distill._default_workers()
        with pytest.raises(harness.R0Error, match="provider/OX"):
            socket.create_connection(("127.0.0.1", 1))


def test_run_integration_writes_sealed_artifact_and_clone_contract(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    args = _patch_run_dependencies(monkeypatch, tmp_path)

    _artifact_id, artifact_path, artifact = harness._run(args)

    assert artifact_path.is_file()
    sealed = store.read_sealed(artifact_path, schema=harness.SCHEMA)
    assert sealed["artifact_id"] == artifact["artifact_id"]
    assert sealed["baseline_reference"]["r0_evidence"] == harness.R0_EVIDENCE_ID
    assert len(sealed["baseline_reference"]["r0_anchor_file_sha256"]) == 64
    assert len(sealed["production"]["anchor_sha256"]) == 64
    assert sealed["contract"]["offline_stage_guards"] is True
    assert sealed["contract"]["acceptance_scope"] == "R1 storage gates only"
    assert sealed["contract"]["advance_latency_gates"] == {
        "status": "not_certified",
        "authority": "R2",
    }
    assert args._env_values["PYTHONDONTWRITEBYTECODE"] == "1"
    assert sealed["contract"]["clones"]
    assert all(
        clone["clone_copy_verified"]
        and clone["copy_backend"] == "copyfile(3)"
        and clone["clone_force"] is True
        and clone["source_filesystem"] == "apfs"
        and clone["destination_filesystem"] == "apfs"
        for clone in sealed["contract"]["clones"]
    )


def test_r0_anchor_rejects_production_watermark_mismatch() -> None:
    ledgers = {name: dict(value) for name, value in harness.EXPECTED_BASELINE.items()}
    fts = {
        "content_sha256": "d" * 64,
        "atom_count": 2,
        "fts_count": 2,
        "checkpoint_seal_sha256": "g" * 64,
        "file_state": {
            "size_bytes": 3,
            "st_dev": 1,
            "st_ino": 2,
            "st_mtime_ns": 4,
            "st_ctime_ns": 4,
        },
    }
    snapshot = {"ledgers": ledgers, "raw_watermark": "c" * 64, "fts": fts}
    anchor = {
        "production": {
            "ledgers": ledgers,
            "raw_watermark": "x" * 64,
            "fts": fts,
        }
    }

    with pytest.raises(harness.R1Error, match="raw watermark"):
        harness._assert_r0_anchor(snapshot, anchor)


def test_r0_anchor_rejects_ledger_file_state_mismatch() -> None:
    file_state = {
        "size_bytes": 10,
        "st_dev": 1,
        "st_ino": 2,
        "st_mtime_ns": 3,
        "st_ctime_ns": 4,
    }
    anchor_ledgers = {
        name: {
            **dict(value),
            "head_sha256": "a" * 64,
            "file_state": dict(file_state),
        }
        for name, value in harness.EXPECTED_BASELINE.items()
    }
    snapshot_ledgers = {
        name: {**value, "file_state": dict(value["file_state"])}
        for name, value in anchor_ledgers.items()
    }
    snapshot_ledgers["candidate-ledger.jsonl"]["file_state"]["st_ctime_ns"] += 1
    fts = {
        "content_sha256": "d" * 64,
        "atom_count": 2,
        "fts_count": 2,
        "checkpoint_seal_sha256": "g" * 64,
        "file_state": dict(file_state),
    }
    snapshot = {
        "ledgers": snapshot_ledgers,
        "raw_watermark": "c" * 64,
        "fts": fts,
    }
    anchor = {
        "production": {
            "ledgers": anchor_ledgers,
            "raw_watermark": "c" * 64,
            "fts": fts,
        }
    }

    with pytest.raises(
        harness.R1Error, match="ledger mismatch: candidate-ledger.jsonl.file_state"
    ):
        harness._assert_r0_anchor(snapshot, anchor)


def test_production_snapshot_uses_checkpoints_without_ledger_body_reads(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    directory = store.distillation_dir(tmp_path)
    heads: dict[str, dict[str, object]] = {}
    ledger_paths: set[Path] = set()
    for name, payload in (
        ("rally-manifest.jsonl", {"kind": "rally"}),
        ("candidate-ledger.jsonl", {"kind": "candidate"}),
        ("label-ledger.jsonl", {"authority": "verified"}),
    ):
        path = directory / name
        store.append_chain(path, payload)
        heads[name] = store.verify_chain(path)
        ledger_paths.add(path.resolve())
    store.write_sealed_state(directory / store.STATE_FILE, {})

    class Raw:
        @staticmethod
        def committed_raw_watermark(_path: Path) -> str:
            return "a" * 64

    monkeypatch.setattr(
        harness,
        "_fts",
        lambda *_args, **_kwargs: {
            "content_sha256": "b" * 64,
            "atom_count": 0,
            "fts_count": 0,
            "file_state": {"size_bytes": 1},
            "checkpoint_seal_sha256": "c" * 64,
        },
    )
    monkeypatch.setattr(
        harness,
        "_loopback_json",
        lambda *_args: (401, b"", None),
    )
    monkeypatch.setattr(
        harness,
        "_label_health_readonly",
        lambda _store, path: {
            "records": heads[path.name]["records"],
            "head_sha256": heads[path.name]["head_sha256"],
            "bytes": path.stat().st_size,
            "counts": {
                "teacher_only": 0,
                "verified_truth": 1,
                "probe_not_truth": 0,
            },
            "file_state": harness._stat(path),
            "source": "test",
        },
    )
    monkeypatch.setattr(
        store,
        "verify_chain",
        lambda _path: (_ for _ in ()).throw(
            AssertionError("production snapshot must not verify ledger bodies")
        ),
    )
    original_read_bytes = Path.read_bytes

    def forbid_ledger_body(self: Path) -> bytes:
        if self.resolve() in ledger_paths:
            raise AssertionError("production snapshot read a ledger body")
        return original_read_bytes(self)

    monkeypatch.setattr(Path, "read_bytes", forbid_ledger_body)
    snapshot = harness._production_snapshot(
        store, None, Raw(), tmp_path, "http://127.0.0.1:1"
    )

    for name, expected in heads.items():
        assert snapshot["ledgers"][name]["records"] == expected["records"]
        assert snapshot["ledgers"][name]["head_sha256"] == expected["head_sha256"]
        assert snapshot["ledgers"][name]["bytes"] == (directory / name).stat().st_size
    harness._assert_r0_anchor(snapshot, {"production": snapshot})


def test_run_integration_aborts_on_accidental_socket_before_artifact(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    args = _patch_run_dependencies(monkeypatch, tmp_path, bad_stage=True)

    with pytest.raises(harness.R0Error, match="provider/OX"):
        harness._run(args)
    assert not args.output.exists()


def test_candidate_snapshot_seals_its_own_digest() -> None:
    snapshot = harness._candidate_snapshot(catalog, "r1-test")
    unsigned = {
        key: value for key, value in snapshot.items() if key != "snapshot_sha256"
    }

    assert snapshot["snapshot_sha256"] == catalog.canonical_json_sha256_strict(unsigned)


def test_unique_duplicate_rebuilds_on_missing_sidecar_and_appends_zero(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = store.distillation_dir(tmp_path) / "exposure-receipts.jsonl"
    store.append_chain_unique(
        path,
        {"decision_id": "decision-1", "idempotency_sha256": "binding-1"},
        unique_field="decision_id",
        binding_field="idempotency_sha256",
    )
    _patch_rusage(monkeypatch)
    index = store._unique_index_path(path)
    index.unlink()
    store._unique_index_checkpoint_path(index).unlink()

    result = harness._unique_duplicate(store, tmp_path, "sidecar_deleted")

    assert result["measurement"]["status"] == "ok"
    assert result["measurement"]["result"]["duplicate_appends"] == 0
    assert result["before"]["head"] == result["after"]["head"]


def test_unique_duplicate_recovers_logical_delete_tamper_without_append(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = store.distillation_dir(tmp_path) / "exposure-receipts.jsonl"
    store.append_chain_unique(
        path,
        {"decision_id": "decision-1", "idempotency_sha256": "binding-1"},
        unique_field="decision_id",
        binding_field="idempotency_sha256",
    )
    _patch_rusage(monkeypatch)

    result = harness._unique_duplicate(
        store, tmp_path, "logical_delete", harness._logical_delete_tamper
    )

    assert result["measurement"]["status"] == "ok"
    assert result["measurement"]["result"]["duplicate_appends"] == 0
    assert result["before"]["head"] == result["after"]["head"]
    with sqlite3.connect(store._unique_index_path(path)) as connection:
        metadata = dict(connection.execute("SELECT key,value FROM metadata"))
    assert metadata["records"] == "1"
    assert metadata["head_sha256"] == result["after"]["head"]["head_sha256"]


def test_unique_crash_retry_commits_one_row(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    path = store.distillation_dir(tmp_path) / "exposure-receipts.jsonl"
    store.append_chain_unique(
        path,
        {"decision_id": "decision-1", "idempotency_sha256": "binding-1"},
        unique_field="decision_id",
        binding_field="idempotency_sha256",
    )
    values = itertools.cycle((_rusage(10), _rusage(11)))
    monkeypatch.setattr(harness, "_proc_pid_rusage_v2", lambda: next(values))

    result = harness._unique_crash(store, tmp_path)

    assert result["first"]["status"] == "error"
    assert result["retry"]["status"] == "ok"
    assert result["retry_result"]["duplicate_appends"] == 0
    assert result["after"]["head"]["records"] == result["before"]["head"]["records"] + 1


def test_static_snapshot_drops_live_payloads() -> None:
    snapshot = {
        "ledgers": {},
        "fast_snapshot": {"events": 1},
        "live_health": {"status": "ok"},
    }

    assert harness._static(snapshot) == {"ledgers": {}}
    redacted = harness._redact_operations(
        {"measurement": {"value": [{"secret": "raw"}]}}
    )
    assert redacted["measurement"]["value"]["shape"] == "list"
    assert "secret" not in redacted["measurement"]["value"]
    index_state = harness._redact_index_state(
        {"ledger_path": "/private/clone/runtime/candidate-ledger.jsonl", "count": 1}
    )
    assert "ledger_path" not in index_state
    assert index_state["ledger_path_basename"] == "candidate-ledger.jsonl"
    assert len(index_state["ledger_path_sha256"]) == 64
