from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from chronovisor.core import runtime_config, sealed_artifact_decoder, store


def test_direct_search_revalidates_exact_stable_namespace_path(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from chronovisor.hosts import server

    root = tmp_path / "wiki"
    pages = root / "pages"
    system = root / "system"
    pages.mkdir(parents=True)
    system.mkdir()
    (pages / "same.md").write_text(
        "---\ntitle: Wrong\nstatus: deprecated\ntype: knowledge\n---\nWRONG PAGE\n",
        encoding="utf-8",
    )
    stable_system = system / "same.md"
    stable_system.write_text(
        "---\ntitle: Same\nstatus: stable\ntype: knowledge\n---\nSYSTEM NEEDLE\n",
        encoding="utf-8",
    )
    drifted = pages / "drifted.md"
    drifted.write_text(
        "---\ntitle: Drifted\nstatus: deprecated\ntype: knowledge\n---\nDRIFT NEEDLE\n",
        encoding="utf-8",
    )

    class FakeStore:
        def meta(self, page_id: str):
            if page_id == "same":
                return {
                    "status": "stable",
                    "path": str(stable_system),
                    "relative_path": "same.md",
                    "is_system": True,
                }
            return {
                "status": "stable",
                "path": str(drifted),
                "relative_path": "drifted.md",
                "is_system": False,
            }

        def tags(self, _page_id: str) -> list[str]:
            return []

    monkeypatch.setattr(server, "CHRONOVISOR_ROOT", root)
    hits = server._direct_search_hits(
        [
            SimpleNamespace(page_id="same", title="Same", updated="now", score=1.0),
            SimpleNamespace(
                page_id="drifted", title="Drifted", updated="now", score=0.9
            ),
        ],
        query="needle",
        store=FakeStore(),
        registry_row=lambda _page_id: {},
    )

    assert [hit["page_id"] for hit in hits] == ["same"]
    assert "SYSTEM NEEDLE" in hits[0]["snippets"][0]
    assert "WRONG PAGE" not in json.dumps(hits)


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
    assert context.claude_code_state_file == tmp_path / "claude-code-save-state.json"
    assert context.model_lab_replay_file == (
        tmp_path / "runtime" / "model-lab" / "replay.jsonl"
    )
    assert context.index_file == tmp_path / "pages" / "index.md"
    assert context.log_file == tmp_path / "pages" / "log.md"
    assert context.schema_file == tmp_path / "system" / "schema.md"
    assert context.activity_file == tmp_path / "runtime" / "activity.jsonl"
    assert store.DEFAULT_CONTEXT.root == store.CHRONOVISOR_ROOT
    assert store.DEFAULT_CONTEXT.raw_dir == store.RAW_DIR
    assert store.DEFAULT_CONTEXT.pages_dir == store.PAGES_DIR
    assert store.DEFAULT_CONTEXT.system_dir == store.SYSTEM_DIR
    assert store.DEFAULT_CONTEXT.model_lab_replay_file == store.MODEL_LAB_REPLAY_FILE
    assert store.DEFAULT_CONTEXT.index_file == store.INDEX_FILE
    assert store.DEFAULT_CONTEXT.log_file == store.LOG_FILE
    assert store.DEFAULT_CONTEXT.schema_file == store.SCHEMA_FILE
    assert store.DEFAULT_CONTEXT.activity_file == store.ACTIVITY_FILE


def test_init_chronovisor_no_arg_uses_final_reserved_paths(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "wiki"
    monkeypatch.setattr(store, "RAW_DIR", root / "raw")
    monkeypatch.setattr(store, "PAGES_DIR", root / "pages")
    monkeypatch.setattr(store, "SYSTEM_DIR", root / "system")
    monkeypatch.setattr(store, "INDEX_FILE", root / "pages" / "index.md")
    monkeypatch.setattr(store, "LOG_FILE", root / "pages" / "log.md")
    monkeypatch.setattr(store, "SCHEMA_FILE", root / "system" / "schema.md")
    monkeypatch.setattr(store, "ACTIVITY_FILE", root / "runtime" / "activity.jsonl")

    store.init_chronovisor()

    assert store.RAW_DIR.is_dir()
    assert store.PAGES_DIR.is_dir()
    assert store.SYSTEM_DIR.is_dir()
    assert store.INDEX_FILE.is_file()
    assert store.LOG_FILE.is_file()
    assert store.SCHEMA_FILE.is_file()
    assert store.ACTIVITY_FILE.is_file()


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
    target.write_text(
        "---\ntitle: Chronovisor\nstatus: stable\ntype: knowledge\n---\n# Chronovisor\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(server, "CHRONOVISOR_ROOT", tmp_path)
    monkeypatch.setattr(alias_store, "resolve_alias_path", lambda _page_id: target)

    assert server._find_page_with_alias("previous-page-id") == target
    target.write_text(
        "---\ntitle: Chronovisor\nstatus: draft\ntype: knowledge\n---\n# Chronovisor\n",
        encoding="utf-8",
    )
    assert (
        server._find_page_with_alias("previous-page-id", require_stable=False) is None
    )


def test_provenance_stringifies_typed_yaml_and_guards_non_string_title(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from chronovisor.hosts import server

    page = tmp_path / "pages" / "typed.md"
    page.parent.mkdir(parents=True)
    page.write_text(
        "---\ntitle:\n  nested: value\nupdated: 2026-08-11\n---\nBody.\n",
        encoding="utf-8",
    )
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    monkeypatch.setattr(server, "RAW_DIR", raw_dir)
    monkeypatch.setattr(server, "_find_page_with_alias", lambda _page: page)
    monkeypatch.setattr(server.activity_log, "iter_activity", lambda _path: [])
    provenance = getattr(server.chronovisor_provenance, "fn", server.chronovisor_provenance)

    payload = json.loads(provenance("typed"))

    assert payload["page_updated"] == "2026-08-11"
    assert payload["raw_sources"] == []


def test_mcp_log_raw_content_requires_both_opt_ins_and_redacts_secret_paths(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from chronovisor.hosts import server

    row = {
        "event_id": "event-1",
        "timestamp": "2026-08-23T20:00:00+09:00",
        "level": "info",
        "source": "mcp",
        "message": "api_key=super-secret /Users/test/.chronovisor/raw/one.md",
    }
    monkeypatch.setattr(server.activity_log, "read_activity", lambda *_args, **_kw: [row])
    monkeypatch.setattr(
        runtime_config,
        "load_mcp_config",
        lambda *_args, **_kwargs: runtime_config.McpConfig(expose_raw_content=False),
    )
    log = getattr(server.chronovisor_log, "fn", server.chronovisor_log)

    default = json.loads(log(limit=1))
    one_sided = json.loads(log(limit=1, include_raw_content=True))
    assert "message" not in default["entries"][0]
    assert "message" not in one_sided["entries"][0]

    monkeypatch.setattr(
        runtime_config,
        "load_mcp_config",
        lambda *_args, **_kwargs: runtime_config.McpConfig(expose_raw_content=True),
    )
    opted_in = json.loads(log(limit=1, include_raw_content=True))
    assert "super-secret" not in opted_in["entries"][0]["message"]
    assert "/Users/test" not in opted_in["entries"][0]["message"]
    assert "[REDACTED]" in opted_in["entries"][0]["message"]


def test_mcp_record_and_ingest_are_metadata_only_even_when_exposure_is_enabled(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from chronovisor.hosts import server

    monkeypatch.setattr(server, "CHRONOVISOR_ROOT", tmp_path)
    monkeypatch.setattr(
        runtime_config,
        "load_mcp_config",
        lambda *_args, **_kwargs: runtime_config.McpConfig(expose_raw_content=True),
    )
    monkeypatch.setattr(
        server,
        "record_raw",
        lambda *_args, **_kwargs: {
            "saved": "raw-safe.md",
            "path": str(tmp_path / "raw" / "raw-safe.md"),
            "raw_slug": "top-secret-slug",
            "rejected_keywords": ["credential=secret"],
            "accepted_keywords": ["also-secret"],
            "content": "raw body secret",
            "ingest_pending": False,
        },
    )
    monkeypatch.setattr(
        server,
        "ingest_raw",
        lambda *_args, **_kwargs: {
            "saved": "ingest-safe.md",
            "path": str(tmp_path / "raw" / "ingest-safe.md"),
            "content": "ingest body secret",
            "raw_slug": "ingest-secret-slug",
            "ingest": {
                "status": "ok",
                "path": str(tmp_path / "runtime" / "result.json"),
                "manifest": str(tmp_path / "runtime" / "manifest.json"),
                "reason": f"read failed: {tmp_path}/private api_key=reason-secret",
                "preview": "nested secret",
            },
        },
    )
    record = getattr(server.chronovisor_record, "fn", server.chronovisor_record)
    ingest = getattr(server.chronovisor_ingest, "fn", server.chronovisor_ingest)

    record_payload = json.loads(record("raw body secret", trigger_ingest=False))
    ingest_payload = json.loads(ingest("ingest body secret", force=False))

    for payload in (record_payload, ingest_payload):
        encoded = json.dumps(payload, ensure_ascii=False)
        assert "secret" not in encoded
        assert str(tmp_path) not in encoded
        assert "raw_slug" not in encoded
        assert "rejected_keywords" not in encoded
        assert "content" not in encoded
    assert record_payload["path"] == "raw/raw-safe.md"
    assert ingest_payload["ingest"]["path"] == "runtime/result.json"
    assert ingest_payload["ingest"]["manifest"] == "runtime/manifest.json"


def test_provenance_not_found_does_not_echo_requested_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from chronovisor.hosts import server

    monkeypatch.setattr(server, "_find_page_with_alias", lambda _page: None)
    provenance = getattr(server.chronovisor_provenance, "fn", server.chronovisor_provenance)

    payload = json.loads(provenance("/Users/test/.chronovisor/secret-page"))

    assert payload == {"error": "page not found"}


def test_server_alias_read_returns_and_traces_only_canonical_page_id(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from chronovisor.core import alias_store
    from chronovisor.hosts import server

    target = tmp_path / "pages" / "chronovisor" / "chronovisor-system.md"
    target.parent.mkdir(parents=True)
    target.write_text(
        "---\ntitle: Chronovisor\nstatus: stable\ntype: knowledge\n---\n# Chronovisor\n",
        encoding="utf-8",
    )
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


def test_server_read_uses_existing_registry_as_fail_closed_authority(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from chronovisor.hosts import server
    from chronovisor.ingest.page_registry import PageRegistry

    root = tmp_path / "wiki"
    pages = root / "pages"
    system = root / "system"
    pages.mkdir(parents=True)
    system.mkdir()

    def write_page(path: Path, status: str, body: str) -> None:
        path.write_text(
            f"---\ntitle: {path.stem}\nstatus: {status}\ntype: knowledge\n---\n{body}\n",
            encoding="utf-8",
        )

    for page_id, status in (
        ("stable", "stable"),
        ("draft", "draft"),
        ("deprecated", "deprecated"),
        ("old", "stable"),
        ("redirect-draft", "stable"),
    ):
        write_page(pages / f"{page_id}.md", status, page_id)
    registry = PageRegistry(root)
    registry.ensure_manifest()
    registry.add_redirect(registry.resolve("old")["uid"], registry.resolve("stable")["uid"])
    registry.add_redirect(
        registry.resolve("redirect-draft")["uid"], registry.resolve("draft")["uid"]
    )
    write_page(pages / "unregistered.md", "stable", "unregistered")
    write_page(system / "unregistered-system.md", "stable", "system")

    class FakeStore:
        def refresh(self) -> None:
            return None

        def meta(self, _page_id: str) -> None:
            return None

        def outlinks(self, _page_id: str) -> list[str]:
            return []

        def backlinks(self, _page_id: str) -> list[str]:
            return []

    monkeypatch.setattr(server, "CHRONOVISOR_ROOT", root)
    monkeypatch.setattr(server, "get_store", FakeStore)
    monkeypatch.setattr(server, "_append_pull_log", lambda _row: None)
    read = getattr(server.chronovisor_read, "fn", server.chronovisor_read)

    redirected = json.loads(read("old"))
    assert redirected["page_id"] == "stable"
    assert "stable" in redirected["content"]
    for page_id in ("draft", "deprecated"):
        uid = registry.resolve(page_id)["uid"]
        assert registry.path_for(page_id) is None
        expected = (pages / f"{page_id}.md").resolve()
        assert registry.path_for(page_id, require_stable=False) == expected
        assert server._find_page_with_alias(page_id) is None
        assert server._find_page_with_alias(page_id, require_stable=False) == expected
        for exact_key in (page_id, uid):
            payload = json.loads(read(exact_key))
            assert payload["page_id"] == page_id
            assert page_id in payload["content"]
        for nonexact_key in (f"{page_id}.md", f"pages/{page_id}"):
            assert json.loads(read(nonexact_key)) == {
                "error": f"Page '{nonexact_key}' not found"
            }
    for page_id in ("redirect-draft", "unregistered", "unregistered-system"):
        assert json.loads(read(page_id)) == {
            "error": f"Page '{page_id}' not found"
        }
