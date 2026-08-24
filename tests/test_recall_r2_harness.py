from __future__ import annotations

import importlib.util
import shutil
import subprocess
import sys
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "recall_r2_harness_test_module", ROOT / "scripts" / "recall_r2_harness.py"
)
assert SPEC is not None and SPEC.loader is not None
HARNESS = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = HARNESS
SPEC.loader.exec_module(HARNESS)


def test_p95_uses_nearest_rank_without_float_rounding() -> None:
    assert HARNESS._p95(list(range(1, 21))) == 19
    assert HARNESS._p95([10, 1, 5, 2, 9]) == 10


def test_read_counters_separate_logical_and_physical_old_bytes() -> None:
    path = Path("/tmp/r2-segment")
    counters = HARNESS.ReadCounters(
        old_ids=frozenset({"old"}), old_ranges={path.resolve(): ((0, 10),)}
    )
    counters.record_logical("new", 4)
    counters.record_logical("old", 6)
    counters.record_range(path, 0, 20)
    assert counters.logical_old_bytes == 6
    assert counters.logical_new_bytes == 4
    assert counters.physical_old_bytes == 10
    assert counters.range_overlaps[0]["old_overlap_bytes"] == 10


def test_instrument_counts_sealed_prefix_as_physical_io() -> None:
    path = Path("/tmp/r2-sealed-segment")
    counters = HARNESS.ReadCounters(
        old_ids=frozenset(), old_ranges={path.resolve(): ((0, 10),)}
    )

    class FakeRawStore:
        def read_bytes(self, raw: object) -> bytes:
            return b""

        def iter_segment_bytes(self, raw_ids: object = None):
            return iter(())

    raw_store_module = SimpleNamespace(
        RawStore=FakeRawStore,
        read_open_range=lambda _path, _offset, length: b"x" * length,
        read_sealed_range=lambda _path, _offset, length: b"x" * length,
    )
    catalog = SimpleNamespace(sqlite3=HARNESS.sqlite3)
    with HARNESS._instrument(
        catalog, SimpleNamespace(), raw_store_module, SimpleNamespace(), counters
    ):
        raw_store_module.read_sealed_range(path, 10, 5)

    assert counters.physical_bytes == 15
    assert counters.physical_old_bytes == 10
    assert counters.range_overlaps == [
        {
            "path": path.name,
            "offset": 0,
            "length": 15,
            "old_overlap_bytes": 10,
        }
    ]


def test_evidence_digest_does_not_depend_on_row_order() -> None:
    rows = [("events", (2, "b")), ("events", (1, "a"))]
    reversed_rows = list(reversed(rows))
    assert HARNESS._canonical_digest(iter(rows)) != HARNESS._canonical_digest(
        iter(reversed_rows)
    )


def test_non_darwin_is_explicitly_unsupported(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(HARNESS.sys, "platform", "linux")
    with pytest.raises(HARNESS.R2Error, match="Darwin/APFS"):
        HARNESS._require_supported_environment(Path("/tmp"))


def test_r2_schema_is_versioned_and_evidence_limit_is_bounded() -> None:
    assert HARNESS.R2_SCHEMA == "chronovisor.recall-r2.v1"
    assert HARNESS.MAX_EVIDENCE_BYTES == 2 * 1024 * 1024
    assert HARNESS.DEFAULT_DELTA_SAMPLES >= 20


def test_root_matrix_rejects_every_protected_overlap(tmp_path: Path) -> None:
    production = tmp_path / "production"
    source = tmp_path / "source"
    production.mkdir()
    source.mkdir()
    output = tmp_path / "evidence.json"
    HARNESS._assert_root_matrix(production, source, output)
    with pytest.raises(HARNESS.R2Error, match="overlap"):
        HARNESS._assert_root_matrix(production, production / "nested", output)


def test_root_matrix_rejects_symlink_entry_points(tmp_path: Path) -> None:
    target = tmp_path / "target"
    target.mkdir()
    link = tmp_path / "link"
    link.symlink_to(target, target_is_directory=True)
    with pytest.raises(HARNESS.R2Error, match="symlink"):
        HARNESS._assert_root_matrix(link, target, tmp_path / "evidence.json")


def test_repair_parity_ignores_file_state_but_checks_inventory_and_duplicates() -> None:
    expected = {
        "catalog": {
            "exists": True,
            "rows": {"raw_units": 2},
            "duplicates": {"raw_units": 0},
            "columns": {"raw_units": ["raw_id"]},
            "digest": "catalog",
            "file_state": {"st_ino": 1},
        },
        "fts": {"exists": True, "rows": 1, "digest": "fts", "file_state": {}},
        "inventory": {"count": 2, "ids_sha256": "ids", "status_counts": {"indexed": 2}},
    }
    repaired = {
        **expected,
        "catalog": {**expected["catalog"], "file_state": {"st_ino": 2}},
        "fts": {**expected["fts"], "file_state": {"st_ino": 3}},
    }
    HARNESS._assert_repair_parity(repaired, expected, "repair")
    repaired["inventory"] = {**expected["inventory"], "count": 1}
    with pytest.raises(HARNESS.R2Error, match="inventory"):
        HARNESS._assert_repair_parity(repaired, expected, "repair")


def test_warm_rejects_assistant_full_scan() -> None:
    metrics = {
        "raw": {
            "logical_old_reads": 0,
            "logical_old_bytes": 0,
            "logical_new_reads": 0,
            "logical_new_bytes": 0,
            "physical_old_bytes": 0,
            "full_raw_scans": 0,
        },
        "scans": {
            "full_event_scans": 0,
            "full_rally_scans": 0,
            "full_session_scans": 0,
            "full_fts_scans": 0,
            "assistant_scans": 1,
            "full_fts_rebuilds": 0,
            "fts_scan_statements": 0,
        },
    }
    with pytest.raises(HARNESS.R2Error, match="scanned"):
        HARNESS._assert_warm(metrics)


def test_delta_read_failure_reports_bounded_safe_counters() -> None:
    metrics = {
        "raw": {
            "logical_old_reads": 1,
            "logical_old_bytes": 6,
            "logical_new_reads": 1,
            "logical_new_bytes": 4,
            "physical_old_bytes": 6,
            "full_raw_scans": 0,
            "logical_old_id_sha256": ["a" * 64],
            "range_overlaps": [
                {
                    "path": "segment.open",
                    "offset": 2,
                    "length": 8,
                    "old_overlap_bytes": 6,
                }
            ],
        }
    }
    with pytest.raises(HARNESS.R2Error) as failure:
        HARNESS._assert_delta(metrics, "new")
    message = str(failure.value)
    assert "logical_old_reads=1" in message
    assert f"old_id_sha256={'a' * 64}" in message
    assert "overlap_path=segment.open" in message


def test_clone_checkpoint_rebind_keeps_delta_incremental(tmp_path: Path) -> None:
    from chronovisor.core import raw_segment
    from chronovisor.core.raw_store import RawStore
    from chronovisor.core.store import RuntimeContext, init_chronovisor
    from chronovisor.recall import recall_distillation as distill
    from chronovisor.recall import recall_distillation_catalog as catalog
    from chronovisor.recall import recall_distillation_store as store

    base = (tmp_path / "base").resolve()
    clone = (tmp_path / "clone").resolve()
    init_chronovisor(RuntimeContext(base))
    HARNESS._append_events(
        raw_segment,
        base,
        session_key="a" * 24,
        after_line=0,
        events=[
            HARNESS._message("user", "baseline query", 0),
            HARNESS._message("assistant", "baseline answer", 1),
        ],
        tag="baseline",
    )
    catalog.advance(base / "raw", base, 4096)
    catalog.sync_historical_index(base / "raw", base)
    shutil.copytree(base, clone, copy_function=shutil.copy2)

    source_catalog = catalog._read_catalog_checkpoint(base)
    assert source_catalog is not None
    assert catalog._read_catalog_checkpoint(clone) is None
    HARNESS._rebind_clone_checkpoints(catalog, base, clone)
    clone_catalog = catalog._read_catalog_checkpoint(clone)
    clone_index = catalog._read_index_checkpoint(catalog.historical_index_path(clone))
    assert clone_catalog is not None and clone_index is not None
    assert clone_catalog["catalog_lineage"] != source_catalog["catalog_lineage"]
    assert clone_index["catalog_lineage"] == clone_catalog["catalog_lineage"]

    source_index_path = catalog._index_checkpoint_path(
        catalog.historical_index_path(base)
    )
    source_index_checkpoint = store.read_sealed(
        source_index_path, schema=store.DISTILLATION_SCHEMA
    )
    mismatched = dict(source_index_checkpoint)
    source_lineage = str(source_index_checkpoint["catalog_lineage"])
    mismatched["catalog_lineage"] = (
        ("0" if source_lineage[0] != "0" else "1") + source_lineage[1:]
    )
    store.write_sealed_state(
        source_index_path,
        {
            key: value
            for key, value in mismatched.items()
            if key not in {"schema", "namespace", "seal_sha256"}
        },
    )
    with pytest.raises(HARNESS.R2Error, match="lineages differ"):
        HARNESS._rebind_clone_checkpoints(catalog, base, clone)
    store.write_sealed_state(
        source_index_path,
        {
            key: value
            for key, value in source_index_checkpoint.items()
            if key not in {"schema", "namespace", "seal_sha256"}
        },
    )

    raw_store_module = sys.modules[RawStore.__module__]
    old_units = HARNESS._raw_units(raw_store_module, clone / "raw")
    new_id, _receipt_sha256 = HARNESS._append_events(
        raw_segment,
        clone,
        session_key="b" * 24,
        after_line=0,
        events=[
            HARNESS._message("user", "delta query", 0),
            HARNESS._message("assistant", "delta answer", 1),
        ],
        tag="delta",
    )
    result, metrics = HARNESS._measure(
        "clone-delta",
        lambda: (
            catalog.advance(clone / "raw", clone, 4096),
            catalog.sync_historical_index(clone / "raw", clone),
        ),
        catalog=catalog,
        distill=distill,
        raw_store_module=raw_store_module,
        store=store,
        old_units=old_units,
    )

    HARNESS._assert_delta(metrics, new_id)
    assert result[0].status == "advanced"

    source_checkpoint = store.read_sealed(
        catalog._catalog_checkpoint_path(base), schema=store.DISTILLATION_SCHEMA
    )
    source_checkpoint.pop("catalog_lineage")
    store.write_sealed_state(
        catalog._catalog_checkpoint_path(base),
        {
            key: value
            for key, value in source_checkpoint.items()
            if key not in {"schema", "namespace", "seal_sha256"}
        },
    )
    legacy_clone = (tmp_path / "legacy-clone").resolve()
    shutil.copytree(base, legacy_clone, copy_function=shutil.copy2)
    HARNESS._rebind_clone_checkpoints(catalog, base, legacy_clone)
    legacy_catalog = catalog._read_catalog_checkpoint(legacy_clone)
    legacy_index = catalog._read_index_checkpoint(catalog.historical_index_path(legacy_clone))
    assert legacy_catalog is not None and legacy_index is not None
    assert legacy_catalog["catalog_lineage"] == legacy_index["catalog_lineage"]


def test_post_commit_crash_recovers_adversarial_catalog_lineage(tmp_path: Path) -> None:
    from chronovisor.core import raw_segment
    from chronovisor.core.raw_store import RawStore
    from chronovisor.core.store import RuntimeContext, init_chronovisor
    from chronovisor.recall import recall_distillation as distill
    from chronovisor.recall import recall_distillation_catalog as catalog
    from chronovisor.recall import recall_distillation_store as store

    base = (tmp_path / "base").resolve()
    init_chronovisor(RuntimeContext(base))
    for session_key in ("c" * 24, "d" * 24, "e" * 24, "f" * 24):
        HARNESS._append_events(
            raw_segment,
            base,
            session_key=session_key,
            after_line=0,
            events=[HARNESS._message("assistant", session_key, 0)],
            tag=f"baseline-{session_key[0]}",
        )
    catalog.advance(base / "raw", base, 4096)
    catalog.sync_historical_index(base / "raw", base)
    raw_store_module = sys.modules[RawStore.__module__]
    root, receipt, clean_root = HARNESS._run_post_commit_crash(
        base=base,
        source_root=ROOT,
        raw_segment=raw_segment,
        catalog=catalog,
        distill=distill,
        store=store,
        raw_store_module=raw_store_module,
        context_bytes=4096,
        old_units=HARNESS._raw_units(raw_store_module, base / "raw"),
    )
    try:
        assert receipt["catalog_child_returncode"] == 137
        assert receipt["fts_child_returncode"] == 137
        assert receipt["parity"] is True
    finally:
        HARNESS._cleanup_clone(root)
        HARNESS._cleanup_clone(clean_root)


def test_clone_cleanup_is_verified(tmp_path: Path) -> None:
    clone = tmp_path / "clone"
    clone.mkdir()
    (clone / "marker").write_text("x", encoding="utf-8")
    HARNESS._cleanup_clone(clone)
    assert not clone.exists()


def test_paired_append_uses_path_neutral_receipt_identity(tmp_path: Path) -> None:
    class FakeRawSegment:
        def __init__(self) -> None:
            self.calls: list[dict[str, object]] = []

        def append_capture(self, **kwargs: object) -> object:
            self.calls.append(kwargs)
            return SimpleNamespace(
                commit=SimpleNamespace(
                    sha256="a" * 64,
                    to_dict=lambda: {
                        "raw_id": kwargs["raw_id"],
                        "session_key": kwargs["session_key"],
                        "source_file": str(kwargs["source_file"]),
                    },
                ),
            )

    first_root = tmp_path / "first"
    second_root = tmp_path / "second"
    first_root.mkdir()
    second_root.mkdir()
    raw_segment = FakeRawSegment()
    events = [HARNESS._message("assistant", "same", 1)]
    first = HARNESS._append_events(
        raw_segment,
        first_root,
        session_key="paired",
        after_line=0,
        events=events,
        tag="paired",
        logical_source_file=Path("r2-paired.jsonl"),
    )
    second = HARNESS._append_events(
        raw_segment,
        second_root,
        session_key="paired",
        after_line=0,
        events=events,
        tag="paired",
        logical_source_file=Path("r2-paired.jsonl"),
    )
    assert first == second
    assert raw_segment.calls[0]["source_file"] == raw_segment.calls[1]["source_file"]
    assert raw_segment.calls[0]["source_bytes"] == raw_segment.calls[1]["source_bytes"]


def test_post_commit_child_receives_context_bytes() -> None:
    assert "context_bytes = int(sys.argv[4])" in HARNESS._POST_COMMIT_CHILD
    assert 'catalog.advance(root / "raw", root, context_bytes)' in (
        HARNESS._POST_COMMIT_CHILD
    )


def test_source_digest_catches_dirty_tracked_and_untracked_bytes(
    tmp_path: Path,
) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "src").mkdir()
    (repo / "scripts").mkdir()
    tracked = repo / "tracked.txt"
    untracked = repo / "tests-dirty.txt"
    tracked.write_text("before", encoding="utf-8")
    untracked.write_text("one", encoding="utf-8")
    subprocess.run(["git", "init", "-q"], cwd=repo, check=True)
    subprocess.run(["git", "add", "tracked.txt"], cwd=repo, check=True)
    tracked.write_text("dirty-before", encoding="utf-8")
    before = HARNESS._source_tree_digest(repo)
    tracked.write_text("dirty-after", encoding="utf-8")
    untracked.write_text("two", encoding="utf-8")
    after = HARNESS._source_tree_digest(repo)
    assert before["git_status_sha256"] == after["git_status_sha256"]
    assert before["repo"]["content_sha256"] != after["repo"]["content_sha256"]


def test_bounded_chain_never_reads_ledger_body(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    ledger = tmp_path / "candidate-ledger.jsonl"
    ledger.write_bytes(b"x" * 128)
    state = HARNESS.R0._stat(ledger)
    assert state is not None

    class BoundedStore:
        DISTILLATION_SCHEMA = "schema"

        @staticmethod
        def _chain_checkpoint_path(path: Path) -> Path:
            return path.with_suffix(path.suffix + ".checkpoint.json")

        @staticmethod
        def read_sealed(_path: Path, *, schema: str) -> dict[str, object]:
            assert schema == "schema"
            return {
                "kind": "ledger-chain-checkpoint",
                "ledger_name": ledger.name,
                "records": 1,
                "head_sha256": "a" * 64,
                "file_state": state,
            }

        @staticmethod
        def verify_chain(_path: Path) -> None:
            raise AssertionError("bounded snapshot must not replay the ledger")

    def fail_read_bytes(*_args: object, **_kwargs: object) -> bytes:
        raise AssertionError("bounded snapshot must not read ledger bytes")

    monkeypatch.setattr(Path, "read_bytes", fail_read_bytes)
    result = HARNESS._bounded_chain(
        BoundedStore(), ledger, require_checkpoint_file_state=True
    )
    assert result["records"] == 1
    assert result["bytes"] == 128


def test_raw_state_digest_does_not_read_file_bodies(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    raw = tmp_path / "raw"
    raw.mkdir()
    (raw / "segment.jsonl.open").write_bytes(b"payload")

    def fail_open(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("Raw state digest must not open bodies")

    monkeypatch.setattr(Path, "open", fail_open)
    monkeypatch.setattr(Path, "read_bytes", fail_open)
    result = HARNESS._raw_tree_state_digest(tmp_path)
    assert result["file_count"] == 1
    assert result["bytes"] == 7


def test_clone_temp_preflight_rejects_source_as_temp_parent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "source"
    source.mkdir()
    monkeypatch.setattr(HARNESS.sys, "platform", "darwin")
    monkeypatch.setattr(HARNESS.tempfile, "gettempdir", lambda: str(source))
    with pytest.raises(HARNESS.R2Error, match="overlaps source"):
        HARNESS._clone_from_root(source)


def test_clone_uses_the_resolved_temp_parent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from chronovisor.core.store import RuntimeContext, init_chronovisor

    source = tmp_path / "source"
    init_chronovisor(RuntimeContext(source))
    (source / "runtime" / "recall-distillation").mkdir()
    (source / "raw" / "event.bin").write_bytes(b"raw")
    real_temp = tmp_path / "real-temp"
    real_temp.mkdir()
    temp_alias = tmp_path / "temp-alias"
    temp_alias.symlink_to(real_temp, target_is_directory=True)
    monkeypatch.setattr(HARNESS.sys, "platform", "darwin")
    monkeypatch.setattr(HARNESS.tempfile, "gettempdir", lambda: str(temp_alias))

    def copyfile(source_path: Path, destination: Path, _flags: int) -> None:
        destination.write_bytes(source_path.read_bytes())

    monkeypatch.setattr(HARNESS, "_copyfile_clone", copyfile)

    clone = HARNESS._clone_from_root(source)
    try:
        assert not HARNESS._has_symlink_component(clone)
        assert clone.parent == real_temp.resolve()
        assert (clone / "pages").stat().st_mode & 0o777 == 0o700
    finally:
        HARNESS._cleanup_clone(clone)


def test_clone_preserves_okf_startup_proof_for_raw_append(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from chronovisor.core import raw_segment
    from chronovisor.core.okf_cutover import discover_okf_startup
    from chronovisor.core.store import RuntimeContext, init_chronovisor

    source = tmp_path / "source"
    init_chronovisor(RuntimeContext(source))
    (source / "runtime" / "recall-distillation").mkdir()
    (source / "runtime" / "recall-distillation" / "state.json").write_text(
        "{}", encoding="utf-8"
    )
    temp_parent = tmp_path / "temp"
    temp_parent.mkdir()
    monkeypatch.setattr(HARNESS.sys, "platform", "darwin")
    monkeypatch.setattr(HARNESS.tempfile, "gettempdir", lambda: str(temp_parent))

    def copyfile(source_path: Path, destination: Path, _flags: int) -> None:
        destination.write_bytes(source_path.read_bytes())

    monkeypatch.setattr(HARNESS, "_copyfile_clone", copyfile)

    clone = HARNESS._clone_from_root(source)
    try:
        assert (clone / "runtime" / "okf-writer.lock").is_file()
        startup = discover_okf_startup(clone, clone / "runtime")
        assert startup.allowed
        raw_id, _receipt_sha256 = HARNESS._append_events(
            raw_segment,
            clone,
            session_key="0123456789abcdef01234567",
            after_line=0,
            events=[HARNESS._message("user", "probe", 0)],
            tag="okf-proof",
        )
        assert raw_id.startswith("save-codex-")
    finally:
        HARNESS._cleanup_clone(clone)


def test_clone_rejects_symlinked_okf_layout_parent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from chronovisor.core.store import RuntimeContext, init_chronovisor

    source = tmp_path / "source"
    init_chronovisor(RuntimeContext(source))
    (source / "runtime" / "recall-distillation").mkdir()
    real_pages = tmp_path / "real-pages"
    (source / "pages").rename(real_pages)
    (source / "pages").symlink_to(real_pages, target_is_directory=True)
    temp_parent = tmp_path / "temp"
    temp_parent.mkdir()
    monkeypatch.setattr(HARNESS.sys, "platform", "darwin")
    monkeypatch.setattr(HARNESS.tempfile, "gettempdir", lambda: str(temp_parent))

    with pytest.raises(HARNESS.R2Error, match="unsafe"):
        HARNESS._clone_from_root(source)


@pytest.mark.parametrize(
    "invalid_entry", ("migrations-file", "root-marker", "pages-symlink")
)
def test_clone_rejects_source_state_that_okf_blocks(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, invalid_entry: str
) -> None:
    from chronovisor.core.okf_cutover import discover_okf_startup
    from chronovisor.core.store import RuntimeContext, init_chronovisor

    source = tmp_path / "source"
    init_chronovisor(RuntimeContext(source))
    (source / "runtime" / "recall-distillation").mkdir()
    if invalid_entry == "migrations-file":
        (source / "runtime" / "migrations").write_text("invalid", encoding="utf-8")
    elif invalid_entry == "root-marker":
        (source / "index.md").write_text("invalid", encoding="utf-8")
    else:
        (source / "pages" / "extra").symlink_to(source / "pages" / "index.md")
    assert not discover_okf_startup(source, source / "runtime").allowed
    temp_parent = tmp_path / "temp"
    temp_parent.mkdir()
    monkeypatch.setattr(HARNESS.sys, "platform", "darwin")
    monkeypatch.setattr(HARNESS.tempfile, "gettempdir", lambda: str(temp_parent))

    def copyfile(source_path: Path, destination: Path, _flags: int) -> None:
        destination.write_bytes(source_path.read_bytes())

    monkeypatch.setattr(HARNESS, "_copyfile_clone", copyfile)
    try:
        clone = HARNESS._clone_from_root(source)
    except HARNESS.R2Error:
        return
    HARNESS._cleanup_clone(clone)
    pytest.fail("OKF-blocked source was normalized into a clone")


def test_clone_holds_okf_layout_lease_while_copying(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from chronovisor.core import durable_state
    from chronovisor.core.store import RuntimeContext, init_chronovisor

    source = tmp_path / "source"
    init_chronovisor(RuntimeContext(source))
    (source / "runtime" / "recall-distillation").mkdir()
    temp_parent = tmp_path / "temp"
    temp_parent.mkdir()
    monkeypatch.setattr(HARNESS.sys, "platform", "darwin")
    monkeypatch.setattr(HARNESS.tempfile, "gettempdir", lambda: str(temp_parent))
    lease_held = False

    @contextmanager
    def writer_lock(root: Path, **kwargs: object) -> object:
        nonlocal lease_held
        assert root == source
        assert kwargs == {"allow_create": False}
        lease_held = True
        try:
            yield
        finally:
            lease_held = False

    def copyfile(source_path: Path, destination: Path, _flags: int) -> None:
        assert lease_held
        destination.write_bytes(source_path.read_bytes())

    monkeypatch.setattr(durable_state, "okf_writer_lock", writer_lock)
    monkeypatch.setattr(HARNESS, "_copyfile_clone", copyfile)
    clone = HARNESS._clone_from_root(source)
    try:
        assert not lease_held
    finally:
        HARNESS._cleanup_clone(clone)


def test_clone_walk_errors_fail_closed_and_cleanup(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from chronovisor.core.store import RuntimeContext, init_chronovisor

    source = tmp_path / "source"
    init_chronovisor(RuntimeContext(source))
    (source / "runtime" / "recall-distillation").mkdir()
    temp_parent = tmp_path / "temp"
    temp_parent.mkdir()
    monkeypatch.setattr(HARNESS.sys, "platform", "darwin")
    monkeypatch.setattr(HARNESS.tempfile, "gettempdir", lambda: str(temp_parent))

    def failing_walk(
        _root: Path, *, followlinks: bool, onerror: object = None
    ) -> object:
        assert not followlinks
        if callable(onerror):
            onerror(PermissionError("denied"))
        return iter(())

    monkeypatch.setattr(HARNESS.os, "walk", failing_walk)
    try:
        clone = HARNESS._clone_from_root(source)
    except HARNESS.R2Error:
        assert not tuple(temp_parent.iterdir())
        return
    HARNESS._cleanup_clone(clone)
    pytest.fail("clone source walk error was ignored")
