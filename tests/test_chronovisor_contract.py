from __future__ import annotations

import json
from pathlib import Path

import pytest

from chronovisor.core import sealed_artifact_decoder, store


def test_resolve_root_uses_only_canonical_root(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    canonical = tmp_path / ".chronovisor"
    monkeypatch.delenv("CHRONOVISOR_ROOT", raising=False)
    monkeypatch.setattr(store, "DEFAULT_ROOT", canonical)

    assert store.resolve_root() == canonical


def test_resolve_root_honors_canonical_override(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    configured = tmp_path / "external" / "chronovisor"
    monkeypatch.setenv("CHRONOVISOR_ROOT", str(configured))

    assert store.resolve_root() == configured


def test_runtime_context_derives_paths_and_preserves_default_aliases(
    tmp_path: Path,
) -> None:
    context = store.RuntimeContext(tmp_path)

    assert context.raw_dir == tmp_path / "raw"
    assert context.pages_dir == tmp_path / "pages"
    assert context.system_dir == tmp_path / "system"
    assert context.config_file == tmp_path / "config.toml"
    assert context.codex_state_file == tmp_path / "codex-save-state.json"
    assert context.model_lab_replay_file == (
        tmp_path / "runtime" / "model-lab" / "replay.jsonl"
    )
    assert context.index_file == tmp_path / "index.md"
    assert context.log_file == tmp_path / "log.md"
    assert context.schema_file == tmp_path / "schema.md"
    assert store.DEFAULT_CONTEXT.root == store.CHRONOVISOR_ROOT
    assert store.DEFAULT_CONTEXT.raw_dir == store.RAW_DIR
    assert store.DEFAULT_CONTEXT.pages_dir == store.PAGES_DIR
    assert store.DEFAULT_CONTEXT.system_dir == store.SYSTEM_DIR
    assert store.DEFAULT_CONTEXT.model_lab_replay_file == store.MODEL_LAB_REPLAY_FILE
    assert store.DEFAULT_CONTEXT.index_file == store.INDEX_FILE
    assert store.DEFAULT_CONTEXT.log_file == store.LOG_FILE
    assert store.DEFAULT_CONTEXT.schema_file == store.SCHEMA_FILE


def test_init_chronovisor_no_arg_preserves_legacy_path_seams(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(store, "RAW_DIR", tmp_path / "legacy-raw")
    monkeypatch.setattr(store, "PAGES_DIR", tmp_path / "legacy-pages")
    monkeypatch.setattr(store, "SYSTEM_DIR", tmp_path / "legacy-system")
    monkeypatch.setattr(store, "INDEX_FILE", tmp_path / "legacy-index.md")
    monkeypatch.setattr(store, "LOG_FILE", tmp_path / "legacy-log.md")
    monkeypatch.setattr(store, "SCHEMA_FILE", tmp_path / "legacy-schema.md")

    store.init_chronovisor()

    assert store.RAW_DIR.is_dir()
    assert store.PAGES_DIR.is_dir()
    assert store.SYSTEM_DIR.is_dir()
    assert store.INDEX_FILE.is_file()
    assert store.LOG_FILE.is_file()
    assert store.SCHEMA_FILE.is_file()


def test_schema_decoder_accepts_only_exact_canonical_schema() -> None:
    current = "chronovisor.raw-reference.v1"
    foreign = "precutover.raw-reference.v1"

    assert sealed_artifact_decoder.schema_matches(current, current)
    assert sealed_artifact_decoder.schema_matches(foreign, current) is False
    assert sealed_artifact_decoder.canonical_schema(current) == current
    with pytest.raises(ValueError, match="canonical Chronovisor"):
        sealed_artifact_decoder.canonical_schema(foreign)


def test_schema_decoder_rejects_noncanonical_current_contract() -> None:
    current = "chronovisor.unregistered-runtime-contract.v1"

    assert sealed_artifact_decoder.schema_matches(current, current)
    assert sealed_artifact_decoder.schema_matches(current, "foreign.schema.v1") is False


def test_public_mcp_surface_contains_only_chronovisor_tools() -> None:
    from chronovisor.hosts.server import mcp

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
    assert all(name.startswith("chronovisor_") for name in names)


def test_server_read_path_resolves_durable_page_alias(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from chronovisor.core import alias_store
    from chronovisor.hosts import server

    target = tmp_path / "pages" / "chronovisor" / "chronovisor-system.md"
    target.parent.mkdir(parents=True)
    target.write_text("# Chronovisor\n", encoding="utf-8")
    monkeypatch.setattr(server, "find_page", lambda _page_id: None)
    monkeypatch.setattr(alias_store, "resolve_alias_path", lambda _page_id: target)

    assert server._find_page_with_alias("previous-page-id") == target


def test_server_alias_read_returns_and_traces_only_canonical_page_id(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from chronovisor.core import alias_store
    from chronovisor.hosts import server

    target = tmp_path / "pages" / "chronovisor" / "chronovisor-system.md"
    target.parent.mkdir(parents=True)
    target.write_text("# Chronovisor\n", encoding="utf-8")
    calls: list[tuple[str, str]] = []
    pull_rows: list[dict] = []

    class FakeStore:
        def refresh(self) -> None:
            return None

        def outlinks(self, page_id: str) -> list[str]:
            calls.append(("outlinks", page_id))
            return ["canonical-outlink"]

        def backlinks(self, page_id: str) -> list[str]:
            calls.append(("backlinks", page_id))
            return ["canonical-backlink"]

    monkeypatch.setattr(server, "get_store", FakeStore)
    monkeypatch.setattr(server, "CHRONOVISOR_ROOT", tmp_path)
    monkeypatch.setattr(server, "find_page", lambda _page_id: None)
    monkeypatch.setattr(alias_store, "resolve_alias_path", lambda _page_id: target)
    monkeypatch.setattr(server, "_append_pull_log", pull_rows.append)

    read_tool = getattr(server.chronovisor_read, "fn", server.chronovisor_read)
    result = json.loads(
        read_tool(
            "previous-page-id",
            session_id="session-1",
            decision_id="decision-1",
        )
    )

    assert result["page_id"] == "chronovisor-system"
    assert result["alias"] == {
        "requested": "previous-page-id",
        "target": "chronovisor-system",
    }
    assert result["outlinks"] == ["canonical-outlink"]
    assert result["backlinks"] == ["canonical-backlink"]
    assert calls == [
        ("outlinks", "chronovisor-system"),
        ("backlinks", "chronovisor-system"),
    ]
    assert pull_rows == [
        {
            "type": "read",
            "stage": "read",
            "session_id": "session-1",
            "decision_id": "decision-1",
            "page_id": "chronovisor-system",
            "requested_page_id": "previous-page-id",
        }
    ]
