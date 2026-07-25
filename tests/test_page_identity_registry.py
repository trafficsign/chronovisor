from __future__ import annotations

from pathlib import Path

import pytest

from chronovisor.page_identity import (
    new_page_uid,
    normalize_page_uid,
    page_uid_timestamp_ms,
)
from chronovisor.page_registry import PageRegistry, PageRegistryError
from chronovisor.uid_link_index import build_uid_link_index


def _page(path: Path, title: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        f"---\ntitle: {title}\nupdated: 2026-07-25\n---\n\n# {title}\n",
        encoding="utf-8",
    )


def test_uuidv7_is_stable_and_semantically_opaque() -> None:
    uid = new_page_uid(timestamp_ms=1_725_000_000_123, random_bits=12345)
    assert normalize_page_uid(uid) == uid
    assert page_uid_timestamp_ms(uid) == 1_725_000_000_123


def test_registry_manifest_is_idempotent_and_redirect_read_is_pure(
    tmp_path: Path,
) -> None:
    _page(tmp_path / "pages" / "alpha.md", "Alpha")
    _page(tmp_path / "pages" / "beta.md", "Beta")
    registry = PageRegistry(tmp_path)

    first = registry.ensure_manifest()
    second = registry.ensure_manifest()

    assert first["created"] == 2
    assert second["created"] == 0
    assert second["updated"] == 0
    assert first["generation"] == second["generation"]
    alpha = registry.resolve("alpha")
    beta = registry.resolve("beta")
    assert alpha and beta
    registry.add_redirect(alpha["uid"], beta["uid"], anchor_map={"old": "new"})
    before = registry.path.stat().st_mtime_ns
    resolved = registry.resolve("alpha")
    after = registry.path.stat().st_mtime_ns
    assert resolved and resolved["uid"] == beta["uid"]
    assert resolved["anchor_map"] == {"old": "new"}
    assert before == after


def test_registry_rejects_duplicate_uid(tmp_path: Path) -> None:
    uid = new_page_uid(timestamp_ms=1_725_000_000_123, random_bits=456)
    for name in ("alpha", "beta"):
        path = tmp_path / "pages" / f"{name}.md"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            f"---\ntitle: {name}\nupdated: 2026-07-25\nuid: {uid}\n---\n",
            encoding="utf-8",
        )
    with pytest.raises(PageRegistryError, match="duplicate page UID"):
        PageRegistry(tmp_path).ensure_manifest()


def test_registry_imports_legacy_alias_keys(tmp_path: Path) -> None:
    _page(tmp_path / "pages" / "folder" / "canonical.md", "Canonical")
    alias_path = tmp_path / "runtime" / "page-aliases.json"
    alias_path.parent.mkdir(parents=True)
    alias_path.write_text(
        '{"aliases":{"old-name":{"target":"folder/canonical"}}}',
        encoding="utf-8",
    )

    registry = PageRegistry(tmp_path)
    registry.ensure_manifest()

    canonical = registry.resolve("canonical")
    legacy = registry.resolve("old-name")
    assert canonical and legacy
    assert legacy["uid"] == canonical["uid"]


def test_uid_link_index_reports_missing_anchor(tmp_path: Path) -> None:
    _page(tmp_path / "pages" / "target.md", "Target")
    source = tmp_path / "pages" / "source.md"
    _page(source, "Source")
    source.write_text(
        source.read_text(encoding="utf-8") + "\n[[target#Missing]]\n",
        encoding="utf-8",
    )

    index = build_uid_link_index(tmp_path)

    assert index["edge_count"] == 0
    assert index["unresolved_count"] == 1
    assert index["unresolved"][0]["reason"] == "missing_anchor"


def test_duplicate_stem_requires_uid_or_relative_path(tmp_path: Path) -> None:
    _page(tmp_path / "pages" / "one" / "same.md", "One")
    _page(tmp_path / "pages" / "two" / "same.md", "Two")
    registry = PageRegistry(tmp_path)

    manifest = registry.ensure_manifest()

    assert manifest["observed"] == 2
    with pytest.raises(PageRegistryError, match="ambiguous page key"):
        registry.resolve("same")
    one = registry.resolve("pages/one/same")
    two = registry.resolve("pages/two/same")
    assert one and two
    assert one["uid"] != two["uid"]


def test_exact_page_id_wins_over_normalized_dot_md_collision(
    tmp_path: Path,
) -> None:
    _page(tmp_path / "pages" / "item.md", "Canonical")
    _page(tmp_path / "pages" / "item.md.md", "Literal dot md")
    registry = PageRegistry(tmp_path)
    registry.ensure_manifest()

    canonical = registry.resolve("item")
    literal = registry.resolve("item.md")

    assert canonical and canonical["page_id"] == "item"
    assert literal and literal["page_id"] == "item.md"


def test_server_does_not_bypass_ambiguous_registry_key(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from chronovisor import server

    first = tmp_path / "pages" / "one" / "same.md"
    second = tmp_path / "pages" / "two" / "same.md"
    _page(first, "One")
    _page(second, "Two")
    PageRegistry(tmp_path).ensure_manifest()
    monkeypatch.setattr(server, "CHRONOVISOR_ROOT", tmp_path)
    monkeypatch.setattr(server, "find_page", lambda _page_id: first)

    assert server._find_page_with_alias("same") is None
    assert server._find_page_with_alias("pages/two/same") == second
