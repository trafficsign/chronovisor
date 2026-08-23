from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from chronovisor.core import raw_store as raw_store_module
from chronovisor.core import runtime_config
from chronovisor.core.legacy_archive import migrate_processed_legacy
from chronovisor.core.save_transaction import (
    attach_save_transaction_marker,
    make_save_transaction,
)
from chronovisor.ingest import orchestrator
from chronovisor.ingest.raw_semantic_projection import project_parent_raw


def _state(**overrides) -> dict:
    state = {
        "last_ingest": None,
        "last_lint": None,
        "processed_raw_files": [],
        "ollama_health": {"status": None, "checked_at": None},
        "current_job_id": None,
        "current_job_pid": None,
        "current_job_started_at": None,
    }
    state.update(overrides)
    return state


def _write_transcript(raw_dir: Path, filename: str, text: str) -> Path:
    transaction = make_save_transaction(
        host="codex",
        session_file=raw_dir / "session.jsonl",
        session_id=filename,
        after_line=0,
        until_line=1,
    )
    content = "\n".join(
        [
            "# Codex Session Transcript Delta",
            "",
            "- Source: Codex",
            "- Capture mode: deterministic-lossless",
            "",
            "## Transcript Delta",
            "",
            "```json",
            json.dumps(
                [{"line": 1, "role": "user", "text": text}],
                ensure_ascii=False,
                indent=2,
            ),
            "```",
            "",
        ]
    )
    path = raw_dir / filename
    path.write_text(
        attach_save_transaction_marker(transaction, content),
        encoding="utf-8",
    )
    return path


def _configure_reconciler(
    tmp_path: Path,
    monkeypatch,
    *,
    processed: list[str],
) -> tuple[Path, Path]:
    root = tmp_path / "wiki"
    raw_dir = root / "raw"
    raw_dir.mkdir(parents=True)
    state_path = root / ".orchestrator_state.json"
    state_path.write_text(
        json.dumps(_state(processed_raw_files=processed)),
        encoding="utf-8",
    )
    monkeypatch.setattr(orchestrator, "CHRONOVISOR_ROOT", root)
    monkeypatch.setattr(orchestrator, "RAW_DIR", raw_dir)
    monkeypatch.setattr(orchestrator, "STATE_FILE", state_path)
    monkeypatch.setenv("CHRONOVISOR_RAW_LAYOUT", "v2")
    monkeypatch.setattr(
        runtime_config,
        "load_ingest_config",
        lambda _path=None: runtime_config.IngestConfig(
            semantic_projection_max_child_bytes=2_000,
        ),
    )
    return root, raw_dir


def test_load_state_drops_unsafe_legacy_batch_failure_counter(
    tmp_path: Path, monkeypatch
) -> None:
    path = tmp_path / "state.json"
    path.write_text(
        json.dumps(_state(triage_failure_count=9, current_job_id="job-a")),
        encoding="utf-8",
    )
    monkeypatch.setattr(orchestrator, "STATE_FILE", path)

    loaded = orchestrator._load_state()

    assert "triage_failure_count" not in loaded
    assert loaded["current_job_id"] == "job-a"
    assert "triage_failure_count" in json.loads(path.read_text(encoding="utf-8"))


def test_reset_stale_pending_reservation_clears_all_owner_fields(
    tmp_path: Path, monkeypatch
) -> None:
    path = tmp_path / "state.json"
    path.write_text(
        json.dumps(
            _state(
                current_job_id="__pending__",
                current_job_pid=123,
                current_job_started_at="2026-07-17T12:00:00",
            )
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(orchestrator, "STATE_FILE", path)
    monkeypatch.setattr(orchestrator, "_lock_is_fresh_in_live_process", lambda _state: False)

    orchestrator.reset_stale_lock()

    persisted = json.loads(path.read_text(encoding="utf-8"))
    assert persisted["current_job_id"] is None
    assert persisted["current_job_pid"] is None
    assert persisted["current_job_started_at"] is None
    assert path.read_text(encoding="utf-8").endswith("\n")


def test_reset_stale_lock_preserves_live_cross_process_owner(
    tmp_path: Path, monkeypatch
) -> None:
    path = tmp_path / "state.json"
    original = _state(
        current_job_id="job-a",
        current_job_pid=456,
        current_job_started_at="2026-07-17T12:00:00",
    )
    path.write_text(json.dumps(original), encoding="utf-8")
    monkeypatch.setattr(orchestrator, "STATE_FILE", path)
    monkeypatch.setattr(orchestrator, "_lock_is_fresh_in_live_process", lambda _state: True)

    orchestrator.reset_stale_lock()

    assert json.loads(path.read_text(encoding="utf-8")) == original


def test_mark_one_raw_processed_preserves_batch_reservation(
    tmp_path: Path, monkeypatch
) -> None:
    path = tmp_path / "state.json"
    path.write_text(
        json.dumps(_state(current_job_id="job-a", current_job_pid=456)),
        encoding="utf-8",
    )
    monkeypatch.setattr(orchestrator, "STATE_FILE", path)

    orchestrator._mark_one_raw_processed("raw-a.md")

    persisted = json.loads(path.read_text(encoding="utf-8"))
    assert persisted["processed_raw_files"] == ["raw-a.md"]
    assert persisted["current_job_id"] == "job-a"
    assert persisted["current_job_pid"] == 456


def test_reconcile_processed_projection_uses_raw_id_cursor_and_repairs_idempotently(
    tmp_path: Path, monkeypatch
) -> None:
    root, raw_dir = _configure_reconciler(
        tmp_path,
        monkeypatch,
        processed=["z.md", "a.md"],
    )
    (root / "config.toml").write_text("[raw]\nlayout = 'v2'\n", encoding="utf-8")
    first = _write_transcript(raw_dir, "z.md", "z source")
    second = _write_transcript(raw_dir, "a.md", "a source")
    source_bytes = {first.name: first.read_bytes(), second.name: second.read_bytes()}

    first_result = orchestrator.reconcile_processed_projections(max_parents=1)
    assert [row["raw_id"] for row in first_result["processed"]] == ["a.md"]
    assert first_result["cursor"] == "a.md"
    first_row = first_result["processed"][0]
    assert len(first_row["raw_sha256"]) == 64
    assert first_row["manifest"].startswith("runtime/raw-projections/artifacts/")
    assert all(
        not Path(child).is_absolute() for child in first_row["children"]
    )
    assert first.read_bytes() == source_bytes[first.name]
    assert second.read_bytes() == source_bytes[second.name]

    second_result = orchestrator.reconcile_processed_projections(max_parents=1)
    assert [row["raw_id"] for row in second_result["processed"]] == ["z.md"]
    assert second_result["cursor"] == "z.md"

    artifact_dir = root / "runtime" / "raw-projections" / "artifacts"
    before = sorted(path.read_bytes() for path in artifact_dir.iterdir())
    wrapped = orchestrator.reconcile_processed_projections(max_parents=1)
    assert wrapped["processed"] == []
    assert wrapped["wrapped"] is True
    again = orchestrator.reconcile_processed_projections(max_parents=1)
    assert again["processed"][0]["status"] == "completed"
    after = sorted(path.read_bytes() for path in artifact_dir.iterdir())
    assert before == after
    assert first.read_bytes() == source_bytes[first.name]
    assert second.read_bytes() == source_bytes[second.name]


def test_reconcile_projection_holds_tampered_child_without_touching_raw(
    tmp_path: Path, monkeypatch
) -> None:
    _root, raw_dir = _configure_reconciler(
        tmp_path,
        monkeypatch,
        processed=["parent.md"],
    )
    parent = _write_transcript(raw_dir, "parent.md", "immutable source")
    output = raw_dir.parent / "runtime" / "raw-projections" / "artifacts"
    projected = project_parent_raw(parent, output_dir=output, max_child_bytes=2_000)
    child = projected.child_paths[0]
    child.write_text("tampered", encoding="utf-8")
    source_bytes = parent.read_bytes()

    result = orchestrator.reconcile_processed_projections()

    assert result["held"][0]["raw_id"] == "parent.md"
    assert result["held"][0]["status"] == "hold"
    assert parent.read_bytes() == source_bytes
    assert child.read_text(encoding="utf-8") == "tampered"


def test_reconcile_projection_repairs_missing_child_from_existing_manifest(
    tmp_path: Path, monkeypatch
) -> None:
    _root, raw_dir = _configure_reconciler(
        tmp_path,
        monkeypatch,
        processed=["parent.md"],
    )
    parent = _write_transcript(raw_dir, "parent.md", "repair missing child")
    output = raw_dir.parent / "runtime" / "raw-projections" / "artifacts"
    projected = project_parent_raw(parent, output_dir=output, max_child_bytes=2_000)
    child = projected.child_paths[0]
    child_bytes = child.read_bytes()
    child.unlink()
    source_bytes = parent.read_bytes()

    def forbidden_llm(*_args, **_kwargs):
        raise AssertionError("processed projection repair must not call LLM ingest")

    monkeypatch.setattr("chronovisor.ingest.ingest.run_ingest", forbidden_llm)

    result = orchestrator.reconcile_processed_projections()

    assert result["processed"][0]["state_before"] == "incomplete"
    assert result["processed"][0]["status"] == "repaired"
    assert child.read_bytes() == child_bytes
    assert parent.read_bytes() == source_bytes


def test_reconcile_disables_after_three_slow_passes(
    tmp_path: Path, monkeypatch
) -> None:
    _root, raw_dir = _configure_reconciler(
        tmp_path,
        monkeypatch,
        processed=["parent.md"],
    )
    _write_transcript(raw_dir, "parent.md", "slow pass")
    ticks = iter((0.0, 0.300, 1.0, 1.300, 2.0, 2.300, 3.0))
    monkeypatch.setattr(orchestrator.time, "perf_counter", lambda: next(ticks))

    first = orchestrator.reconcile_processed_projections()
    second = orchestrator.reconcile_processed_projections()
    third = orchestrator.reconcile_processed_projections()

    assert first["slow_streak"] == 1
    assert second["slow_streak"] == 2
    assert third["status"] == "disabled"
    assert third["slow_streak"] == 3
    assert orchestrator.reconcile_processed_projections()["reason"] == (
        "slow_pass_kill_switch"
    )


def test_reconcile_p95_uses_nearest_rank_for_two_samples(
    tmp_path: Path, monkeypatch
) -> None:
    _root, _raw_dir = _configure_reconciler(
        tmp_path,
        monkeypatch,
        processed=[],
    )
    state_path = orchestrator.STATE_FILE
    state = _state(
        processed_raw_files=[],
        processed_projection_reconciler_timings_ms=[1.0],
    )
    state_path.write_text(json.dumps(state), encoding="utf-8")
    ticks = iter((0.0, 0.300))
    monkeypatch.setattr(orchestrator.time, "perf_counter", lambda: next(ticks))

    result = orchestrator.reconcile_processed_projections()

    assert result["p95_ms"] == 300.0
    assert result["max_ms"] == 300.0


def test_reconcile_checkpoint_merges_latest_non_owned_state_fields(
    tmp_path: Path, monkeypatch
) -> None:
    _root, raw_dir = _configure_reconciler(
        tmp_path,
        monkeypatch,
        processed=["parent.md"],
    )
    _write_transcript(raw_dir, "parent.md", "merge checkpoint")
    initial = _state(processed_raw_files=["parent.md"])
    latest = _state(
        processed_raw_files=["newer-parent.md"],
        current_job_id="job-new",
        unknown_field={"keep": True},
    )
    loads = iter((initial, latest))
    monkeypatch.setattr(orchestrator, "_load_state", lambda: next(loads))

    orchestrator.reconcile_processed_projections(max_parents=1)

    persisted = json.loads(orchestrator.STATE_FILE.read_text(encoding="utf-8"))
    assert persisted["processed_raw_files"] == ["newer-parent.md"]
    assert persisted["current_job_id"] == "job-new"
    assert persisted["unknown_field"] == {"keep": True}
    assert persisted["processed_projection_reconciler_cursor"] == "parent.md"


def test_reconcile_rejects_path_traversal_processed_id_without_echo(
    tmp_path: Path, monkeypatch
) -> None:
    _root, _raw_dir = _configure_reconciler(
        tmp_path,
        monkeypatch,
        processed=["../outside.md", "/absolute/outside.md"],
    )

    result = orchestrator.reconcile_processed_projections()

    assert len(result["held"]) == 2
    assert all("raw_id" not in row for row in result["held"])
    assert all("outside" not in json.dumps(row) for row in result["held"])


def test_reconcile_128_parent_batch_stays_within_wall_time_budget(
    tmp_path: Path, monkeypatch
) -> None:
    raw_ids = [f"parent-{index:03d}.md" for index in range(128)]
    root, raw_dir = _configure_reconciler(
        tmp_path,
        monkeypatch,
        processed=raw_ids,
    )
    units = [
        SimpleNamespace(
            raw_id=raw_id,
            storage="legacy_file",
            path=tmp_path / raw_id,
            commit=None,
        )
        for raw_id in raw_ids
    ]
    for raw_id in raw_ids:
        (raw_dir / raw_id).write_bytes(b"ordinary raw\n")

    class FakeRawStore:
        def __init__(self, _raw_dir: Path) -> None:
            pass

        def iter_units(self):
            return iter(units)

        def read_bytes(self, _unit):
            return b"ordinary raw\n"

    monkeypatch.setattr(raw_store_module, "RawStore", FakeRawStore)
    artifact_dir = root / "runtime" / "raw-projections" / "artifacts"
    artifact_dir.mkdir(parents=True)
    manifests: dict[Path, str] = {}
    for index in range(128):
        projection_id = f"{index:064x}"
        manifest = artifact_dir / f"semantic-{projection_id}.manifest.json"
        noop = artifact_dir / f"semantic-{projection_id}.noop.json"
        manifest.write_bytes(b"manifest")
        noop.write_bytes(b"noop")
        manifests[manifest] = noop.name

    def verify_existing(manifest: Path) -> dict[str, str]:
        return {
            "status": "noop",
            "projection_id": manifest.stem.removeprefix("semantic-").removesuffix(
                ".manifest"
            ),
            "noop_receipt_filename": manifests[manifest],
        }

    def project_existing(path: Path, *, output_dir: Path, **_kwargs):
        index = int(path.stem.removeprefix("parent-"))
        return SimpleNamespace(
            kind="noop",
            manifest_path=output_dir
            / f"semantic-{index:064x}.manifest.json",
            child_paths=(),
        )

    monkeypatch.setattr(
        "chronovisor.ingest.raw_semantic_projection.project_parent_raw",
        project_existing,
    )
    monkeypatch.setattr(
        "chronovisor.ingest.raw_semantic_projection.verify_projection_bundle",
        verify_existing,
    )

    result = orchestrator.reconcile_processed_projections()

    assert len(result["processed"]) == 128
    assert result["cursor"] == raw_ids[-1]
    assert result["max_ms"] >= result["p95_ms"]


def test_reconcile_budget_is_not_spent_listing_all_projection_artifacts(
    tmp_path: Path, monkeypatch
) -> None:
    root, raw_dir = _configure_reconciler(
        tmp_path,
        monkeypatch,
        processed=["a.md", "b.md"],
    )
    _write_transcript(raw_dir, "a.md", "first source")
    _write_transcript(raw_dir, "b.md", "second source")
    artifact_dir = root / "runtime" / "raw-projections" / "artifacts"
    artifact_dir.mkdir(parents=True)
    (artifact_dir / "existing-artifact").write_bytes(b"existing")

    elapsed = 0.0
    original_iterdir = Path.iterdir

    def expensive_iterdir(path: Path):
        nonlocal elapsed
        if path == artifact_dir:
            elapsed += 0.300
        return original_iterdir(path)

    monkeypatch.setattr(Path, "iterdir", expensive_iterdir)
    monkeypatch.setattr(orchestrator.time, "perf_counter", lambda: elapsed)

    result = orchestrator.reconcile_processed_projections(max_parents=2)

    assert result["cursor"] == "b.md"
    assert len(result["processed"]) == 2
    assert result["budget_exhausted"] is False
    assert result["elapsed_ms"] < 250


def test_reconcile_checkpoints_before_candidate_discovery_exhausts_budget(
    tmp_path: Path, monkeypatch
) -> None:
    raw_ids = [f"parent-{index:03d}.md" for index in range(20)]
    _root, raw_dir = _configure_reconciler(
        tmp_path,
        monkeypatch,
        processed=raw_ids,
    )
    for raw_id in raw_ids:
        _write_transcript(raw_dir, raw_id, "ordinary source")

    elapsed = 0.0
    original_regular_unit = orchestrator._processed_projection_regular_unit

    def slow_regular_unit(path: Path, raw_id: str, source_dir: Path):
        nonlocal elapsed
        elapsed += 0.210 if raw_id == raw_ids[0] else 0.012
        return original_regular_unit(path, raw_id, source_dir)

    monkeypatch.setattr(
        orchestrator,
        "_processed_projection_regular_unit",
        slow_regular_unit,
    )
    monkeypatch.setattr(orchestrator.time, "perf_counter", lambda: elapsed)
    monkeypatch.setattr(
        "chronovisor.ingest.raw_semantic_projection.project_parent_raw",
        lambda *_args, **_kwargs: SimpleNamespace(
            kind="passthrough",
            manifest_path=None,
            child_paths=(),
        ),
    )

    result = orchestrator.reconcile_processed_projections(max_parents=20)

    assert result["cursor"] == raw_ids[0]
    assert len(result["processed"]) == 1
    assert result["budget_exhausted"] is True
    assert result["elapsed_ms"] < 250


def test_reconcile_rejects_symlink_swap_before_builder_receives_bytes(
    tmp_path: Path, monkeypatch
) -> None:
    _root, raw_dir = _configure_reconciler(
        tmp_path,
        monkeypatch,
        processed=["parent.md"],
    )
    raw_path = _write_transcript(raw_dir, "parent.md", "inside Raw root")
    outside = tmp_path / "outside.md"
    outside.write_text("must never reach projection", encoding="utf-8")
    real_open = raw_store_module.os.open
    swapped = False

    def swap_final_open(path, flags, mode=0o777, *, dir_fd=None):
        nonlocal swapped
        if path == raw_path.name and dir_fd is not None and not swapped:
            swapped = True
            raw_path.unlink()
            raw_path.symlink_to(outside)
        return real_open(path, flags, mode, dir_fd=dir_fd)

    monkeypatch.setattr(raw_store_module.os, "open", swap_final_open)
    monkeypatch.setattr(
        "chronovisor.ingest.raw_semantic_projection.project_parent_raw",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("builder received swapped Raw bytes")
        ),
    )

    result = orchestrator.reconcile_processed_projections(max_parents=1)

    assert result["processed"] == []
    assert result["held"][0]["raw_id"] == "parent.md"
    assert "must never reach projection" not in json.dumps(result)


def test_reconcile_resolves_ref_missing_legacy_archive_without_full_inventory(
    tmp_path: Path, monkeypatch
) -> None:
    _root, raw_dir = _configure_reconciler(
        tmp_path,
        monkeypatch,
        processed=["archived.md"],
    )
    _write_transcript(raw_dir, "archived.md", "archived Raw")
    migrate_processed_legacy(
        raw_dir,
        processed_raw_ids={"archived.md"},
        before="9999/12/31",
        dry_run=False,
        remove_source=True,
    )
    monkeypatch.setattr(
        raw_store_module.RawStore,
        "iter_units",
        lambda _store: (_ for _ in ()).throw(
            AssertionError("legacy archive lookup scanned full inventory")
        ),
    )

    result = orchestrator.reconcile_processed_projections(max_parents=1)

    assert result["held"] == []
    assert result["processed"][0]["raw_id"] == "archived.md"
    assert result["processed"][0]["resolution"] == "legacy_archive"


def test_reconcile_projects_verified_archived_markdown_as_parent_raw(
    tmp_path: Path, monkeypatch
) -> None:
    root, raw_dir = _configure_reconciler(
        tmp_path,
        monkeypatch,
        processed=["archived.md"],
    )
    reference = root / "runtime" / "raw-projections" / "parents" / "archived.md"
    reference.parent.mkdir(parents=True)
    reference.write_text("{}", encoding="utf-8")
    raw_bytes = b"verified archived Markdown\n"
    unit = SimpleNamespace(
        raw_id="archived.md",
        storage="segment_sealed",
        path=raw_dir / "segment.jsonl.zst",
        sha256="a" * 64,
        commit=SimpleNamespace(sha256="a" * 64),
    )

    class FakeRawStore:
        def __init__(self, _raw_dir: Path) -> None:
            pass

        def resolve_reference(self, _path: Path):
            return unit

        def materialize_ingest(self, _unit, _reference_dir: Path) -> Path:
            return reference

        def read_bytes(self, _unit) -> bytes:
            return raw_bytes

        def is_archived_legacy_markdown(self, _unit, value: bytes) -> bool:
            assert value == raw_bytes
            return True

    monkeypatch.setattr(raw_store_module, "RawStore", FakeRawStore)
    monkeypatch.setattr(
        "chronovisor.ingest.raw_semantic_projection.project_native_transcript",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("verified archived Markdown is not native JSONL")
        ),
    )

    def project_archived_markdown(*_args, **kwargs):
        assert kwargs["allow_verified_legacy_markdown"] is True
        return SimpleNamespace(
            kind="passthrough",
            manifest_path=None,
            child_paths=(),
        )

    monkeypatch.setattr(
        "chronovisor.ingest.raw_semantic_projection.project_parent_raw",
        project_archived_markdown,
    )

    result = orchestrator.reconcile_processed_projections(max_parents=1)

    assert result["held"] == []
    assert result["processed"][0]["raw_id"] == "archived.md"
    assert result["processed"][0]["status"] == "passthrough"


def test_reconcile_skips_unresolved_id_without_full_inventory(
    tmp_path: Path, monkeypatch
) -> None:
    _configure_reconciler(
        tmp_path,
        monkeypatch,
        processed=["removed.md"],
    )
    monkeypatch.setattr(
        raw_store_module.RawStore,
        "iter_units",
        lambda _store: (_ for _ in ()).throw(
            AssertionError("unresolved lookup scanned full inventory")
        ),
    )

    result = orchestrator.reconcile_processed_projections(max_parents=1)

    assert result["processed"] == []
    assert result["held"] == []
    assert result["cursor"] == "removed.md"
