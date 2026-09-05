#!/usr/bin/env python3
"""Small model-free smoke check for a deployed Chronovisor archive.

All state is created below a temporary directory.  The checks cover the
runtime contracts affected by the fixed-model optimization without starting a
service or reading the production root.
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path

from chronovisor.core.index_store import IndexStore
from chronovisor.ingest.convergence import ConvergenceStore
from chronovisor.recall.recall_auditor import read_jsonl_tail


def _check_jsonl_tail(root: Path) -> None:
    path = root / "mixed.jsonl"
    path.write_bytes(
        b'{"id":"lf"}\n'
        b'{"id":"crlf"}\r\n'
        b'{"id":"cr"}\r'
        b'{"id":"tail"}'
    )
    assert read_jsonl_tail(path, limit=4) == [
        {"id": "tail"},
        {"id": "cr"},
        {"id": "crlf"},
        {"id": "lf"},
    ]


def _check_convergence(root: Path) -> None:
    state_file = root / "convergence" / "state.json"
    store = ConvergenceStore(
        state_file,
        events_file=state_file.with_name("events.jsonl"),
        lock_file=state_file.with_name("state.lock"),
    )
    created = store.merge_item(
        lane="smoke",
        source_id="item-1",
        input_data={"value": 1},
        metadata={"nested": {"count": 1}},
    )
    key = str(created["item"]["key"])
    before = state_file.read_bytes()

    loaded = store.load()
    loaded["items"][key]["metadata"]["nested"]["count"] = 2
    selected = store.get(key)
    assert selected is not None
    selected["metadata"]["nested"]["count"] = 3
    listed = store.list_items()
    assert len(listed) == 1
    listed[0]["metadata"]["nested"]["count"] = 4

    assert state_file.read_bytes() == before
    current = store.get(key)
    assert current is not None
    assert current["metadata"]["nested"]["count"] == 1


def _write_page(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def _check_index_store(root: Path) -> None:
    pages = root / "pages"
    system = root / "system"
    pages.mkdir(parents=True)
    system.mkdir()
    for name in ("index.md", "log.md", "schema.md"):
        _write_page(root / name, "legacy\n")

    target = pages / "target.md"
    source = pages / "source.md"
    _write_page(
        target,
        "---\n"
        "title: Target\n"
        "updated: 2026-01-01\n"
        "status: stable\n"
        "type: knowledge\n"
        "---\nbody\n",
    )
    _write_page(
        source,
        "---\n"
        "title: Source\n"
        "updated: 2026-01-01\n"
        "status: stable\n"
        "type: knowledge\n"
        "tags: [topic]\n"
        "entities: [Chronovisor]\n"
        "---\n[Target](target.md)\n",
    )

    store = IndexStore(root)
    store.refresh()
    expected = {
        "outlinks": ["target"],
        "backlinks": ["source"],
        "tag_pages": ["source"],
        "entity_pages": ["source"],
    }
    assert store.outlinks("source") == expected["outlinks"]
    assert store.backlinks("target") == expected["backlinks"]
    assert store.pages_for_tag("topic") == expected["tag_pages"]
    assert store.pages_for_entity("chronovisor") == expected["entity_pages"]

    _write_page(
        source,
        "---\n"
        "title: Source Revised\n"
        "updated: 2026-01-02\n"
        "status: stable\n"
        "type: knowledge\n"
        "tags: [topic]\n"
        "entities: [Chronovisor]\n"
        "---\n[Target](target.md)\n",
    )
    store.apply_changes([source])
    assert store.meta("source")["title"] == "Source Revised"
    assert store.outlinks("source") == expected["outlinks"]
    assert store.backlinks("target") == expected["backlinks"]
    assert store.pages_for_tag("topic") == expected["tag_pages"]
    assert store.pages_for_entity("chronovisor") == expected["entity_pages"]
    assert (root / ".index" / "pages.json").is_file()
    assert (root / ".index" / "backlinks.json").is_file()

    reloaded = IndexStore(root)
    reloaded.ensure_loaded()
    assert reloaded.meta("source")["title"] == "Source Revised"
    assert reloaded.outlinks("source") == expected["outlinks"]
    assert reloaded.backlinks("target") == expected["backlinks"]
    assert reloaded.pages_for_tag("topic") == expected["tag_pages"]
    assert reloaded.pages_for_entity("chronovisor") == expected["entity_pages"]


def main() -> None:
    previous_read_only = os.environ.pop("CHRONOVISOR_READ_ONLY", None)
    try:
        with tempfile.TemporaryDirectory(
            prefix="chronovisor-fixed-model-live-smoke-"
        ) as name:
            root = Path(name).resolve()
            _check_jsonl_tail(root)
            _check_convergence(root)
            _check_index_store(root)
    finally:
        if previous_read_only is not None:
            os.environ["CHRONOVISOR_READ_ONLY"] = previous_read_only
    print(
        json.dumps(
            {
                "status": "ok",
                "checks": [
                    "read_jsonl_tail_mixed_newline_order",
                    "convergence_read_helpers_are_copy_safe",
                    "index_metadata_update_persist_reload_and_derived_results",
                ],
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(
            json.dumps(
                {"status": "error", "error_type": type(exc).__name__},
                ensure_ascii=False,
                sort_keys=True,
            ),
            file=sys.stderr,
        )
        raise
