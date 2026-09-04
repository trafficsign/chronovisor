#!/usr/bin/env python3
"""Measure IndexStore metadata-only updates before and after A3.

The ``legacy`` store binds the actual ``_refresh_locked`` and ``apply_changes``
methods extracted from commit ``7b872fa``. Both stores use the same generated
corpus, and each update pair alternates old/new call order. Only a synthetic
local corpus is used; no model or application service starts.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
import shutil
import statistics
import subprocess
import tempfile
import time
from pathlib import Path
from types import MethodType
from typing import Any

from chronovisor.core import index_store as index_store_mod
from chronovisor.core.index_store import IndexStore


def _seed(root: Path, page_count: int) -> Path:
    pages = root / "pages"
    (pages).mkdir(parents=True)
    (root / "system").mkdir()
    for name in ("index.md", "log.md", "schema.md"):
        (root / name).write_text("legacy\n", encoding="utf-8")
    for index in range(page_count):
        (pages / f"{index:05d}.md").write_text(
            "---\n"
            f"title: Page {index}\n"
            "updated: 2026-01-01\n"
            "status: stable\n"
            "type: knowledge\n"
            f"tags: [tag-{index % 10}]\n"
            f"entities: [entity-{index % 20}]\n"
            "---\n"
            "body\n",
            encoding="utf-8",
        )
    return pages / "00001.md"


def _load_legacy_methods() -> dict[str, Any]:
    repo_root = Path(__file__).resolve().parents[3]
    source = subprocess.run(
        [
            "git",
            "show",
            "7b872fa:src/chronovisor/core/index_store.py",
        ],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    tree = ast.parse(source)
    index_store_class = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "IndexStore"
    )
    future_annotations = ast.ImportFrom(
        module="__future__", names=[ast.alias(name="annotations")], level=0
    )
    methods: dict[str, Any] = {}
    for method_name in ("_refresh_locked", "apply_changes"):
        method = next(
            node
            for node in index_store_class.body
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            and node.name == method_name
        )
        module = ast.Module(body=[future_annotations, method], type_ignores=[])
        ast.fix_missing_locations(module)
        namespace: dict[str, Any] = {}
        exec(
            compile(module, f"{method_name}@7b872fa", "exec"),
            index_store_mod.__dict__,
            namespace,
        )
        methods[method_name] = namespace[method_name]
    return methods


def _digest(value: object) -> str:
    encoded = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _sample_mapping(mapping: dict[str, list[str]]) -> list[list[object]]:
    return [
        [key, values[:2] + values[-2:]]
        for key, values in list(mapping.items())[:2]
    ]


def _snapshot(store: IndexStore) -> dict[str, Any]:
    metadata = store.all_pages_meta(include_system=True)
    canonical = sorted(store.all_canonical_page_keys(include_system=True))
    with store._lock:
        backlinks = {
            target: list(sources)
            for target, sources in sorted(store._backlinks.items())
        }
        tag_pages = {
            tag: list(page_ids) for tag, page_ids in sorted(store._tag_pages.items())
        }
        entity_pages = {
            entity: list(page_ids)
            for entity, page_ids in sorted(store._entity_pages.items())
        }
    components = {
        "all_pages_meta": metadata,
        "canonical_keys": canonical,
        "backlinks": backlinks,
        "tag_pages": tag_pages,
        "entity_pages": entity_pages,
    }
    return {
        "digests": {name: _digest(value) for name, value in components.items()},
        "counts": {
            "all_pages_meta": len(metadata),
            "canonical_keys": len(canonical),
            "backlinks": len(backlinks),
            "tag_pages": len(tag_pages),
            "entity_pages": len(entity_pages),
        },
        "samples": {
            "all_pages_meta": metadata[:2] + metadata[-2:],
            "canonical_keys": canonical[:2] + canonical[-2:],
            "backlinks": _sample_mapping(backlinks),
            "tag_pages": _sample_mapping(tag_pages),
            "entity_pages": _sample_mapping(entity_pages),
        },
    }


def _run_pair(
    *,
    legacy_methods: dict[str, Any],
    page_count: int,
    updates: int,
    operation: str,
    persist: bool,
) -> list[dict[str, Any]]:
    with (
        tempfile.TemporaryDirectory(prefix="chronovisor-a3-") as name,
        tempfile.TemporaryDirectory(prefix="chronovisor-a3-peer-") as peer_name,
    ):
        root = Path(name).resolve()
        changed_path = _seed(root, page_count)
        peer_root = Path(peer_name).resolve() / "corpus"
        shutil.copytree(root, peer_root, copy_function=shutil.copy2)
        old_store = IndexStore(root)
        new_store = IndexStore(peer_root)
        old_store._refresh_locked = MethodType(
            legacy_methods["_refresh_locked"], old_store
        )
        old_store.apply_changes = MethodType(
            legacy_methods["apply_changes"], old_store
        )
        old_path = changed_path
        new_path = peer_root / "pages" / changed_path.name
        old_store._refresh_locked()
        new_store._refresh_locked()

        calls = {
            mode: {
                "_rebuild_canonical_entries": 0,
                "_rebuild_backlinks": 0,
                "_rebuild_associations": 0,
            }
            for mode in ("legacy", "optimized")
        }
        for mode, candidate in (("legacy", old_store), ("optimized", new_store)):
            for method_name in calls[mode]:
                original = getattr(candidate, method_name)

                def counted(
                    *args: object,
                    _calls=calls[mode],
                    _name=method_name,
                    _original=original,
                    **kwargs: object,
                ) -> None:
                    _calls[_name] += 1
                    _original(*args, **kwargs)

                setattr(candidate, method_name, counted)

        durations_ms: dict[str, list[float]] = {"legacy": [], "optimized": []}
        pair_equivalence: list[bool] = []
        final_snapshots: dict[str, dict[str, Any]] = {}
        for update in range(updates):
            content = (
                "---\n"
                f"title: Changed {update}\n"
                "updated: 2026-01-01\n"
                "status: stable\n"
                "type: knowledge\n"
                "tags: [tag-1]\n"
                "entities: [entity-1]\n"
                "---\n"
                "body\n"
            )
            old_path.write_text(content, encoding="utf-8")
            new_path.write_text(content, encoding="utf-8")
            old_first = update % 2 == 0
            ordered_stores = (
                (old_store, old_path, "legacy"), (new_store, new_path, "optimized")
            ) if old_first else (
                (new_store, new_path, "optimized"), (old_store, old_path, "legacy")
            )
            for candidate, candidate_path, mode in ordered_stores:
                started = time.perf_counter_ns()
                if operation == "refresh":
                    candidate._refresh_locked()
                else:
                    candidate.apply_changes([candidate_path])
                elapsed_ms = (time.perf_counter_ns() - started) / 1e6
                durations_ms[mode].append(elapsed_ms)
            old_snapshot = _snapshot(old_store)
            new_snapshot = _snapshot(new_store)
            final_snapshots = {"legacy": old_snapshot, "optimized": new_snapshot}
            pair_equivalence.append(old_snapshot["digests"] == new_snapshot["digests"])

        runs: list[dict[str, Any]] = []
        for mode, candidate in (("legacy", old_store), ("optimized", new_store)):
            samples = durations_ms[mode]
            ordered = sorted(samples)
            p95_index = min(
                len(ordered) - 1, max(0, int(len(ordered) * 0.95) - 1)
            )
            runs.append(
                {
                    "mode": mode,
                    "operation": operation,
                    "persist": persist,
                    "page_count": page_count,
                    "updates": updates,
                    "update_ms": samples,
                    "median_ms": statistics.median(samples),
                    "p95_ms": ordered[p95_index],
                    "derived_rebuild_calls": calls[mode],
                    "result_title": candidate.meta("00001")["title"],
                    "result_page_count": candidate.page_count(),
                    "snapshot": final_snapshots[mode],
                    "pair_equivalence": pair_equivalence,
                }
            )
        return runs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pages", type=int, default=1000)
    parser.add_argument("--updates", type=int, default=30)
    parser.add_argument(
        "--operation", choices=("refresh", "apply_changes"), default="refresh"
    )
    parser.add_argument(
        "--persist",
        action="store_true",
        help="keep the normal atomic index persistence path enabled",
    )
    args = parser.parse_args()
    if args.pages < 2 or args.updates < 1:
        parser.error("--pages must be >= 2 and --updates must be >= 1")

    previous = os.environ.get("CHRONOVISOR_READ_ONLY")
    if args.persist:
        os.environ.pop("CHRONOVISOR_READ_ONLY", None)
    else:
        os.environ["CHRONOVISOR_READ_ONLY"] = "1"
    try:
        runs = _run_pair(
            legacy_methods=_load_legacy_methods(),
            page_count=args.pages,
            updates=args.updates,
            operation=args.operation,
            persist=args.persist,
        )
        legacy_digests = runs[0]["snapshot"]["digests"]
        optimized_digests = runs[1]["snapshot"]["digests"]
        results = {
            "baseline_reference": "7b872fa:src/chronovisor/core/index_store.py",
            "python": os.sys.version.split()[0],
            "page_count": args.pages,
            "updates": args.updates,
            "paired_order": "old,new then new,old alternating per update",
            "runs": runs,
            "equivalence": {
                "pairs_checked": args.updates,
                "all_pairs": all(runs[0]["pair_equivalence"]),
                "components": {
                    name: legacy_digests[name] == optimized_digests[name]
                    for name in legacy_digests
                },
            },
        }
    finally:
        if previous is None:
            os.environ.pop("CHRONOVISOR_READ_ONLY", None)
        else:
            os.environ["CHRONOVISOR_READ_ONLY"] = previous
    print(json.dumps(results, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
