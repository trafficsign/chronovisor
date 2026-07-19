from __future__ import annotations

import json
import fcntl
import os
from pathlib import Path

import pytest

from chronovisor import brand_migration, schema_compat, store
from chronovisor.brand_audit import audit


def test_resolve_root_prefers_legacy_before_migration(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    legacy = tmp_path / ".wiki"
    canonical = tmp_path / ".chronovisor"
    legacy.mkdir()
    monkeypatch.delenv("CHRONOVISOR_ROOT", raising=False)
    monkeypatch.setattr(store, "LEGACY_ROOT", legacy)
    monkeypatch.setattr(store, "DEFAULT_ROOT", canonical)

    assert store.resolve_root() == legacy


def test_resolve_root_rejects_independent_trees(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    legacy = tmp_path / ".wiki"
    canonical = tmp_path / ".chronovisor"
    legacy.mkdir()
    canonical.mkdir()
    monkeypatch.delenv("CHRONOVISOR_ROOT", raising=False)
    monkeypatch.setattr(store, "LEGACY_ROOT", legacy)
    monkeypatch.setattr(store, "DEFAULT_ROOT", canonical)

    with pytest.raises(RuntimeError, match="split-brain"):
        store.resolve_root()


def test_legacy_environment_alias_rejects_conflicts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import chronovisor

    monkeypatch.setenv("LLM_WIKI_TEST_FLAG", "legacy")
    monkeypatch.setenv("CHRONOVISOR_TEST_FLAG", "canonical")

    with pytest.raises(RuntimeError, match="conflicting compatibility"):
        chronovisor._alias_environment(
            "LLM_WIKI_TEST_FLAG", "CHRONOVISOR_TEST_FLAG"
        )


def test_schema_compatibility_is_dual_read_and_canonical_write() -> None:
    current = "chronovisor.example.v1"

    assert schema_compat.schema_matches(current, current)
    assert schema_compat.schema_matches("llm-wiki.example.v1", current)
    assert schema_compat.canonical_schema("llm-wiki.example.v1") == current
    assert schema_compat.legacy_schema(current) == "llm-wiki.example.v1"


@pytest.mark.parametrize("suffix", schema_compat.MIGRATED_SCHEMA_SUFFIXES)
def test_all_persistent_schema_ids_accept_legacy_and_canonical_reads(
    suffix: str,
) -> None:
    current = f"chronovisor.{suffix}"
    legacy = f"llm-wiki.{suffix}"

    assert schema_compat.schema_matches(current, current)
    assert schema_compat.schema_matches(legacy, current)
    assert schema_compat.canonical_schema(legacy) == current


def test_brand_migration_round_trip_preserves_inventory_and_compat_link(
    tmp_path: Path,
) -> None:
    legacy = tmp_path / ".wiki"
    canonical = tmp_path / ".chronovisor"
    (legacy / "raw").mkdir(parents=True)
    (legacy / "pages" / "ai").mkdir(parents=True)
    (legacy / "runtime").mkdir(parents=True)
    (legacy / "raw" / "record.md").write_text("raw", encoding="utf-8")
    (legacy / "pages" / "ai" / "memory.md").write_text(
        "memory", encoding="utf-8"
    )
    (legacy / "runtime" / "wiki-mutation.lock").write_text("", encoding="utf-8")
    (legacy / "config.toml").write_text(
        f'root = "{legacy}"\n', encoding="utf-8"
    )

    assert brand_migration.inspect(
        legacy_root=legacy, canonical_root=canonical
    )["state"] == "ready"
    applied = brand_migration.apply(
        legacy_root=legacy, canonical_root=canonical
    )

    assert applied["status"] == "applied"
    assert legacy.is_symlink()
    assert legacy.resolve() == canonical
    assert not (canonical / "runtime" / "wiki-mutation.lock").exists()
    assert (canonical / "runtime" / "chronovisor-mutation.lock").exists()
    assert str(canonical) in (canonical / "config.toml").read_text(encoding="utf-8")
    verified = brand_migration.verify(
        legacy_root=legacy, canonical_root=canonical
    )
    assert verified["status"] == "verified"
    assert verified["inventory"] == applied["inventory"]

    rolled_back = brand_migration.rollback(
        legacy_root=legacy, canonical_root=canonical
    )
    assert rolled_back["status"] == "rolled_back"
    assert legacy.is_dir() and not legacy.is_symlink()
    assert not canonical.exists()
    assert (legacy / "runtime" / "wiki-mutation.lock").exists()
    assert not (legacy / "runtime" / "chronovisor-mutation.lock").exists()
    assert str(legacy) in (legacy / "config.toml").read_text(encoding="utf-8")


def test_brand_migration_preflight_rejects_a_live_writer_lock(
    tmp_path: Path,
) -> None:
    legacy = tmp_path / ".wiki"
    canonical = tmp_path / ".chronovisor"
    lock_path = legacy / "runtime" / "wiki-mutation.lock"
    lock_path.parent.mkdir(parents=True)
    descriptor = os.open(lock_path, os.O_RDWR | os.O_CREAT, 0o600)
    fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
    try:
        with pytest.raises(brand_migration.BrandMigrationError, match="writer lock"):
            brand_migration.preflight(
                legacy_root=legacy, canonical_root=canonical
            )
    finally:
        fcntl.flock(descriptor, fcntl.LOCK_UN)
        os.close(descriptor)


def test_brand_migration_reconciles_empty_transition_lock_duplicates(
    tmp_path: Path,
) -> None:
    legacy = tmp_path / ".wiki"
    canonical = tmp_path / ".chronovisor"
    runtime = legacy / "runtime"
    runtime.mkdir(parents=True)
    (runtime / "wiki-mutation.lock").touch()
    (runtime / "chronovisor-mutation.lock").touch()

    preflight = brand_migration.preflight(
        legacy_root=legacy, canonical_root=canonical
    )
    applied = brand_migration.apply(
        legacy_root=legacy, canonical_root=canonical
    )

    assert preflight["duplicate_internal_paths"] == [
        "runtime/chronovisor-mutation.lock"
    ]
    assert applied["status"] == "applied"
    assert (canonical / "runtime" / "chronovisor-mutation.lock").exists()
    assert not (canonical / "runtime" / "wiki-mutation.lock").exists()


def test_public_mcp_surface_contains_only_chronovisor_tools() -> None:
    from chronovisor.server import mcp

    names = set(mcp._tool_manager._tools)
    assert names == {
        "chronovisor_init",
        "chronovisor_search",
        "chronovisor_read",
        "chronovisor_recall_used",
        "chronovisor_index",
        "chronovisor_log",
        "chronovisor_status",
        "chronovisor_reindex",
        "chronovisor_ingest",
        "chronovisor_check",
        "chronovisor_apply",
        "chronovisor_jobs",
        "chronovisor_deep_dive",
        "chronovisor_research",
        "chronovisor_provenance",
        "chronovisor_record",
        "chronovisor_tick",
    }
    assert all(not name.startswith("wiki_") for name in names)


def test_brand_audit_has_no_unclassified_legacy_tokens() -> None:
    repo_root = Path(__file__).resolve().parents[1]

    result = audit(repo_root)

    assert result["status"] == "ok", json.dumps(
        result["violations"], ensure_ascii=False, indent=2
    )
