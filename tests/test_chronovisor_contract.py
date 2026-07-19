from __future__ import annotations

import json
from pathlib import Path

import pytest

from chronovisor import sealed_artifact_decoder, store
from chronovisor.contract_audit import audit


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


def test_schema_decoder_is_dual_read_and_canonical_write() -> None:
    current = "chronovisor.raw-reference.v1"
    previous = sealed_artifact_decoder.previous_schema(current)

    assert sealed_artifact_decoder.schema_matches(current, current)
    assert sealed_artifact_decoder.schema_matches(previous, current)
    assert sealed_artifact_decoder.canonical_schema(previous) == current


def test_schema_decoder_rejects_unregistered_previous_contract() -> None:
    current = "chronovisor.unregistered-runtime-contract.v1"
    previous = "llm-wiki.unregistered-runtime-contract.v1"

    assert sealed_artifact_decoder.schema_matches(current, current)
    assert sealed_artifact_decoder.schema_matches(previous, current) is False
    with pytest.raises(ValueError, match="sealed-artifact allowlist"):
        sealed_artifact_decoder.previous_schema(current)
    with pytest.raises(ValueError, match="sealed-artifact allowlist"):
        sealed_artifact_decoder.canonical_schema(previous)


@pytest.mark.parametrize("suffix", sealed_artifact_decoder.SEALED_SCHEMA_SUFFIXES)
def test_persistent_schema_decoder_accepts_sealed_previous_artifacts(
    suffix: str,
) -> None:
    current = f"chronovisor.{suffix}"
    previous = sealed_artifact_decoder.previous_schema(current)

    assert sealed_artifact_decoder.schema_matches(current, current)
    assert sealed_artifact_decoder.schema_matches(previous, current)
    assert sealed_artifact_decoder.canonical_schema(previous) == current


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
    assert all(name.startswith("chronovisor_") for name in names)


def test_server_read_path_resolves_durable_previous_page_alias(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from chronovisor import alias_store, server

    target = tmp_path / "pages" / "chronovisor" / "chronovisor-system.md"
    target.parent.mkdir(parents=True)
    target.write_text("# Chronovisor\n", encoding="utf-8")
    monkeypatch.setattr(server, "find_page", lambda _page_id: None)
    monkeypatch.setattr(alias_store, "resolve_alias_path", lambda _page_id: target)

    assert server._find_page_with_alias("llm-wiki-mcp") == target


def test_contract_audit_has_no_unclassified_previous_tokens() -> None:
    repo_root = Path(__file__).resolve().parents[1]

    result = audit(repo_root)

    assert result["status"] == "ok", json.dumps(
        result["violations"], ensure_ascii=False, indent=2
    )
