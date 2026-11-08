from __future__ import annotations

import json
from pathlib import Path

import pytest

from chronovisor.index import index_store
from chronovisor.index.index_store import IndexStore
from chronovisor.search import search as search_module
from chronovisor.surface import mcp_server as server


def _page(root: Path, name: str, body: str) -> Path:
    path = root / "pages" / f"{name}.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        f"---\ntitle: {name}\nstatus: stable\ntype: knowledge\n---\n{body}\n",
        encoding="utf-8",
    )
    return path


def test_mcp_tools_do_not_scan_disk_when_the_snapshot_is_fresh(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "wiki"
    _page(root, "alpha", "alpha needle")
    (root / "system").mkdir()

    store = IndexStore(root)
    store.refresh()

    monkeypatch.setattr(server, "CHRONOVISOR_ROOT", root)
    monkeypatch.setattr(server, "get_store", lambda: store)
    monkeypatch.setattr(index_store, "get_store", lambda: store)
    monkeypatch.setattr(server, "snapshot", lambda: {"by_status": {}})
    monkeypatch.setattr(
        search_module, "_BM25_CACHE_FILE", root / ".index" / "lexical.sqlite"
    )
    monkeypatch.setattr(search_module, "_BM25_SINGLETON", None)

    scans: list[Path] = []
    real_scan = IndexStore._scan_disk

    def scan(this: IndexStore) -> list[object]:
        scans.append(this._root)
        return real_scan(this)

    monkeypatch.setattr(IndexStore, "_scan_disk", scan)

    read = getattr(server.chronovisor_read, "fn", server.chronovisor_read)
    search = getattr(server.chronovisor_search, "fn", server.chronovisor_search)
    init = getattr(server.chronovisor_init, "fn", server.chronovisor_init)
    status = getattr(server.chronovisor_status, "fn", server.chronovisor_status)

    assert json.loads(read("alpha"))["page_id"] == "alpha"
    assert [hit["page_id"] for hit in json.loads(search("needle"))["direct_hits"]] == [
        "alpha"
    ]
    init()
    status()

    assert scans == []
