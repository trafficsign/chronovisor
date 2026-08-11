from __future__ import annotations

import json
from pathlib import Path

import pytest

from chronovisor.core.index_store import canonical_document_paths
from chronovisor.ingest.page_registry import PageRegistry, PageRegistryError
from chronovisor.ingest.uid_link_index import build_uid_link_index


def _page(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        f"---\ntitle: {path.stem}\nstatus: stable\ntype: knowledge\n---\n",
        encoding="utf-8",
    )


def test_page_paths_exclude_reserved_documents_and_symlinks(tmp_path: Path) -> None:
    for relative in (
        "pages/index.md",
        "pages/log.md",
        "pages/schema.md",
        "pages/nested/index.md",
        "pages/nested/log.md",
        "pages/nested/schema.md",
        "system/index.md",
        "system/log.md",
    ):
        _page(tmp_path / relative)
    _page(tmp_path / "pages" / "concept-index.md")
    _page(tmp_path / "system" / "schema.md")
    outside = tmp_path / "outside.md"
    _page(outside)
    (tmp_path / "pages" / "outside-link.md").symlink_to(outside)
    (tmp_path / "pages" / "inside-link.md").symlink_to(
        tmp_path / "pages" / "concept-index.md"
    )

    paths = canonical_document_paths(
        tmp_path / "pages",
        system_dir=tmp_path / "system",
    )

    assert [path.relative_to(tmp_path).as_posix() for path in paths] == [
        "pages/concept-index.md",
        "system/schema.md",
    ]
    with pytest.raises(PageRegistryError, match="invalid canonical document"):
        PageRegistry._page_paths(tmp_path, include_system=True)


def test_registry_refreshes_stable_page_to_deprecated(tmp_path: Path) -> None:
    page = tmp_path / "pages" / "page.md"
    _page(page)
    registry = PageRegistry(tmp_path)
    registry.ensure_manifest()

    page.write_text(
        "---\ntitle: page\nstatus: deprecated\ntype: knowledge\n---\n",
        encoding="utf-8",
    )
    refreshed = registry.ensure_manifest()

    assert refreshed["registry"]["pages"][registry.resolve("page")["uid"]][
        "status"
    ] == "deprecated"


@pytest.mark.parametrize(
    ("registered_status", "document_status"),
    [("draft", "stable"), ("stable", "draft")],
)
def test_path_for_rejects_registry_document_status_drift(
    tmp_path: Path,
    registered_status: str,
    document_status: str,
) -> None:
    page = tmp_path / "pages" / "page.md"
    page.parent.mkdir(parents=True)
    page.write_text(
        f"---\ntitle: page\nstatus: {registered_status}\ntype: knowledge\n---\n",
        encoding="utf-8",
    )
    registry = PageRegistry(tmp_path)
    registry.ensure_manifest()
    page.write_text(
        f"---\ntitle: page\nstatus: {document_status}\ntype: knowledge\n---\n",
        encoding="utf-8",
    )

    assert registry.path_for("page", require_stable=False) is None


@pytest.mark.parametrize(
    "invalid",
    [
        "---\ntitle: page\nstatus: stable\n---\n",
        "---\ntitle: page\nstatus: active\ntype: knowledge\n---\n",
        "---\ntitle: page\nstatus: stable\ntype: knowledge\n---\n[[legacy]]\n",
    ],
)
def test_registry_refresh_fails_closed_on_present_invalid_page(
    tmp_path: Path,
    invalid: str,
) -> None:
    page = tmp_path / "pages" / "page.md"
    _page(page)
    registry = PageRegistry(tmp_path)
    state = registry.ensure_manifest()["registry"]
    uid = next(iter(state["pages"]))
    page.write_text(invalid, encoding="utf-8")

    with pytest.raises(PageRegistryError, match="invalid canonical document"):
        registry.ensure_manifest()
    loaded = registry.load()
    assert loaded["pages"][uid]["status"] == "stable"
    assert registry.resolve("page")["uid"] == uid
    assert registry.stable_pages(loaded) == {}
    assert registry.path_for("page") is None
    with pytest.raises(PageRegistryError, match="invalid canonical document"):
        build_uid_link_index(tmp_path, registry=registry, write=False)


def test_registry_stale_stable_symlink_is_never_adopted(tmp_path: Path) -> None:
    page = tmp_path / "pages" / "page.md"
    _page(page)
    registry = PageRegistry(tmp_path)
    state = registry.ensure_manifest()["registry"]
    uid = next(iter(state["pages"]))
    outside = tmp_path / "outside.md"
    _page(outside)
    page.unlink()
    page.symlink_to(outside)

    with pytest.raises(PageRegistryError, match="invalid canonical document"):
        registry.ensure_manifest()
    loaded = registry.load()
    assert loaded["pages"][uid]["status"] == "stable"
    assert registry.stable_pages(loaded) == {}
    assert registry.path_for(uid) is None
    assert registry.path_for(uid, require_stable=False) is None


def test_v1_active_current_row_refreshes_from_frontmatter(tmp_path: Path) -> None:
    page = tmp_path / "pages" / "page.md"
    _page(page)
    registry = PageRegistry(tmp_path)
    state = registry.ensure_manifest()["registry"]
    uid = next(iter(state["pages"]))
    state["pages"][uid]["status"] = "active"
    registry.path.write_text(json.dumps(state), encoding="utf-8")

    assert registry.load()["pages"][uid]["status"] == "active"
    refreshed = registry.ensure_manifest()["registry"]
    assert refreshed["pages"][uid]["status"] == "stable"


def test_stable_pages_rejects_inverse_redirect_mismatch(tmp_path: Path) -> None:
    source = tmp_path / "pages" / "source.md"
    target = tmp_path / "pages" / "target.md"
    _page(source)
    _page(target)
    registry = PageRegistry(tmp_path)
    registry.ensure_manifest()
    source_uid = registry.resolve("source")["uid"]
    target_uid = registry.resolve("target")["uid"]
    registry.add_redirect(source_uid, target_uid)
    state = registry.load()
    state["pages"][source_uid]["canonical_uid"] = None
    registry.path.write_text(json.dumps(state), encoding="utf-8")

    loaded = registry.load()
    stable = registry.stable_pages(loaded)

    assert source_uid not in stable
    assert target_uid in stable
    assert registry.path_for(source_uid) == target.resolve()


def test_v1_registry_rows_are_canonicalized_inside_ensure(tmp_path: Path) -> None:
    source = tmp_path / "pages" / "source.md"
    target = tmp_path / "pages" / "target.md"
    _page(source)
    _page(target)
    registry = PageRegistry(tmp_path)
    state = registry.ensure_manifest()["registry"]
    source_uid = registry.resolve("source")["uid"]
    target_uid = registry.resolve("target")["uid"]
    registry.add_redirect(source_uid, target_uid)
    source.unlink()
    state = registry.load()
    state["pages"][source_uid]["status"] = "superseded"
    registry.path.write_text(json.dumps(state), encoding="utf-8")

    refreshed = registry.ensure_manifest()["registry"]

    assert refreshed["schema"] == "chronovisor.page-registry.v1"
    assert refreshed["pages"][source_uid]["status"] == "deprecated"
    assert refreshed["pages"][source_uid]["canonical_uid"] == target_uid


def test_current_v1_row_uses_document_status_and_preserves_redirect(
    tmp_path: Path,
) -> None:
    source = tmp_path / "pages" / "source.md"
    target = tmp_path / "pages" / "target.md"
    _page(source)
    _page(target)
    registry = PageRegistry(tmp_path)
    registry.ensure_manifest()
    source_uid = registry.resolve("source")["uid"]
    target_uid = registry.resolve("target")["uid"]
    registry.add_redirect(source_uid, target_uid)
    state = registry.load()
    state["pages"][source_uid]["status"] = "active"
    registry.path.write_text(json.dumps(state), encoding="utf-8")
    source.write_text(
        "---\ntitle: source\nstatus: draft\ntype: knowledge\n---\n",
        encoding="utf-8",
    )

    refreshed = registry.ensure_manifest()["registry"]

    assert refreshed["pages"][source_uid]["status"] == "draft"
    assert refreshed["pages"][source_uid]["canonical_uid"] == target_uid
