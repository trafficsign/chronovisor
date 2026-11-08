from __future__ import annotations

import json
from pathlib import Path

import pytest

from chronovisor.core import page_registry
from chronovisor.core.index_store import IndexStore
from chronovisor.hosts import server


def _page(root: Path, name: str, *, status: str = "stable") -> Path:
    path = root / "pages" / f"{name}.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        f"---\ntitle: {name}\nstatus: {status}\ntype: knowledge\n---\n{name} body\n",
        encoding="utf-8",
    )
    return path


def _registry(
    root: Path, state: dict[str, object] | None = None
) -> page_registry.PageRegistry:
    registry = page_registry.PageRegistry(root)
    registry.path.parent.mkdir(parents=True, exist_ok=True)
    registry.path.write_text(
        json.dumps(state if state is not None else registry.empty()),
        encoding="utf-8",
    )
    return registry


@pytest.fixture
def isolated_read(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[Path, IndexStore]:
    root = tmp_path / "wiki"
    for directory in (root / "pages", root / "system", root / ".index"):
        directory.mkdir(parents=True)
    store = IndexStore(root)
    monkeypatch.setattr(server, "CHRONOVISOR_ROOT", root)
    monkeypatch.setattr(server, "get_store", lambda: store)
    return root, store


def _read(page_id: str) -> dict[str, object]:
    return json.loads(
        getattr(server.chronovisor_read, "fn", server.chronovisor_read)(page_id)
    )


def test_frozen_registry_falls_back_to_unknown_canonical_page(
    isolated_read: tuple[Path, IndexStore],
) -> None:
    root, store = isolated_read
    path = _page(root, "missing-from-registry")
    _registry(root)
    store.refresh()

    result = _read("missing-from-registry")

    assert result["page_id"] == "missing-from-registry"
    assert result["content"] == path.read_text(encoding="utf-8")


def test_fallback_preserves_require_stable(
    isolated_read: tuple[Path, IndexStore],
) -> None:
    root, _store = isolated_read
    draft = _page(root, "unregistered-draft", status="draft")
    _registry(root)

    assert (
        server._find_page_with_alias("unregistered-draft", require_stable=True) is None
    )
    assert (
        server._find_page_with_alias("unregistered-draft", require_stable=False)
        == draft
    )


def test_known_registry_alias_and_uid_remain_authoritative(
    isolated_read: tuple[Path, IndexStore],
) -> None:
    root, store = isolated_read
    target = _page(root, "current")
    alias_path = _page(root, "old", status="deprecated")
    uid = "019f0000-0000-7000-8000-000000000001"
    retired_uid = "019f0000-0000-7000-8000-000000000002"
    state = page_registry.PageRegistry.empty()
    state["pages"] = {
        uid: {
            "uid": uid,
            "page_id": "current",
            "path": "pages/current.md",
            "status": "stable",
        },
        retired_uid: {
            "uid": retired_uid,
            "page_id": "old",
            "path": "pages/old.md",
            "status": "deprecated",
            "canonical_uid": uid,
        },
    }
    state["keys"] = {"current": uid, "old": retired_uid}
    state["redirects"] = {retired_uid: {"to_uid": uid, "anchor_map": {}}}
    _registry(root, state)
    store.refresh()

    assert _read("old")["page_id"] == "current"
    assert _read(uid)["page_id"] == "current"
    assert _read("old")["content"] == target.read_text(encoding="utf-8")
    assert alias_path.read_text(encoding="utf-8") != _read("old")["content"]


def test_known_registry_draft_and_ambiguous_keys_do_not_fallback(
    isolated_read: tuple[Path, IndexStore],
) -> None:
    root, _store = isolated_read
    draft = _page(root, "registered-draft", status="draft")
    _page(root, "registered-missing")
    _page(root, "one")
    _page(root, "two")
    uid = "019f0000-0000-7000-8000-000000000003"
    one_uid = "019f0000-0000-7000-8000-000000000004"
    two_uid = "019f0000-0000-7000-8000-000000000005"
    missing_uid = "019f0000-0000-7000-8000-000000000006"
    missing_target_uid = "019f0000-0000-7000-8000-000000000007"
    state = page_registry.PageRegistry.empty()
    state["pages"] = {
        uid: {
            "uid": uid,
            "page_id": "registered-draft",
            "path": "pages/registered-draft.md",
            "status": "draft",
        },
        one_uid: {
            "uid": one_uid,
            "page_id": "one",
            "path": "pages/one.md",
            "status": "stable",
        },
        two_uid: {
            "uid": two_uid,
            "page_id": "two",
            "path": "pages/two.md",
            "status": "stable",
        },
    }
    state["keys"] = {
        "registered-draft": uid,
        "registered-draft-alias": uid,
        "registered-missing": missing_uid,
    }
    state["ambiguous_keys"] = {"collision": [one_uid, two_uid]}
    state["redirects"] = {missing_uid: {"to_uid": missing_target_uid, "anchor_map": {}}}
    _registry(root, state)

    assert _read("registered-draft")["page_id"] == "registered-draft"
    assert _read("registered-draft-alias") == {
        "error": "Page 'registered-draft-alias' not found"
    }
    assert _read("registered-missing") == {
        "error": "Page 'registered-missing' not found"
    }
    assert _read("collision") == {"error": "Page 'collision' not found"}
    assert draft.exists()


def test_invalid_canonical_candidates_remain_invisible_with_frozen_registry(
    isolated_read: tuple[Path, IndexStore],
) -> None:
    root, _store = isolated_read
    _page(root, "normal")
    (root / "pages" / "index.md").write_text(
        "---\ntitle: Reserved\nstatus: stable\ntype: knowledge\n---\nreserved\n",
        encoding="utf-8",
    )
    outside = root.parent / "outside.md"
    outside.write_text("secret", encoding="utf-8")
    (root / "pages" / "page-link.md").symlink_to(outside)
    _registry(root)

    for page_id in ("index", "../outside", "page-link"):
        assert _read(page_id) == {"error": f"Page '{page_id}' not found"}


def test_corrupt_registry_is_a_safe_error_not_a_not_found_fallback(
    isolated_read: tuple[Path, IndexStore],
) -> None:
    root, _store = isolated_read
    _page(root, "present")
    registry = page_registry.PageRegistry(root)
    registry.path.parent.mkdir(parents=True, exist_ok=True)
    registry.path.write_text("not-json", encoding="utf-8")

    result = _read("present")

    assert result == {"error": "Page registry unavailable"}
