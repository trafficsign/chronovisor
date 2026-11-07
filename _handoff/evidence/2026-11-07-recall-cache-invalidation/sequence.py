"""CPU-only real-data sequence; all mutations are confined to a copied store."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import shutil
import sqlite3
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, cast


def bind_frozen_pages(root: Path) -> None:
    from chronovisor.core import feedback_ledger as ledger
    from chronovisor.core import negative_feedback as nf

    bindings = json.loads((root / "ledger-pages.json").read_text())
    for name in ("find_page", "find_mutation_page"):
        def resolve(page_id: str, name: str = name) -> Path | None:
            value = bindings.get(page_id, {}).get(name)
            return Path(value) if value else None
        setattr(ledger, name, resolve)
    pages = json.loads((root / "bound-pages.json").read_text())
    cast(Any, nf).find_mutation_page = lambda page_id: Path(pages[page_id]) if page_id in pages else None


def child(root: Path, mode: str) -> dict[str, Any]:
    from chronovisor.core import negative_feedback as nf
    from chronovisor.core import search
    from chronovisor.core.jsonl_write import append_jsonl_durable
    from chronovisor.core.lexical_index import LexicalIndex
    from chronovisor.core.runtime_config import load_negative_feedback_config

    assert root != Path.home() / ".chronovisor"
    bind_frozen_pages(root)
    path = root / ".index/lexical.sqlite"
    if mode == "prime":
        entries = nf._load_entries(load_negative_feedback_config())
        nf._load_golden_entries()
        index = LexicalIndex(path=path, pages=lambda: [])
        assert index.load_existing()
        index.close()
        return {"entries": len(entries), "primed": True}
    if mode == "append":
        append_jsonl_durable(root / "recall/feedback.jsonl", [{
            "kind": "missed_candidate", "source": "auditor", "host": "codex",
            "prompt": "StartLux recall filter threshold", "expected_pages": [],
        }])
        return {"appended": 1}
    if mode == "open":
        # Both service writers and normal search share LexicalIndex._open.
        keeper = sqlite3.connect(path)
        keeper.execute("PRAGMA user_version").fetchone()
        wal = Path(f"{path}-wal")
        before = hashlib.sha256(wal.read_bytes()).hexdigest()
        index = LexicalIndex(path=path, pages=lambda: [])
        index._open()
        index.stats()
        index.close()
        after = hashlib.sha256(wal.read_bytes()).hexdigest()
        assert before == after
        keeper.close()
        return {"wal_unchanged": True}
    if mode == "search":
        original = nf._read_persistent_entries
        hits = []

        def cache(key: dict[str, object]) -> Any:
            value = original(key)
            hits.append(value is not None)
            return value

        def miss(*_args: Any, **_kwargs: Any) -> Any:
            raise AssertionError("unexpected cache miss in real sequence")

        nf._read_persistent_entries = cache
        cast(Any, nf).trusted_negative_feedback_rows = miss
        nf._parse_golden_entries = miss
        cast(Any, LexicalIndex)._run_quick_check = staticmethod(miss)
        caches = [nf.PERSISTENT_CACHE_FILE, nf.GOLDEN_CACHE_FILE, path.with_suffix(".quick-check.json")]
        cache_before = [(p.read_bytes(), p.stat().st_mtime_ns) for p in caches]
        query = "StartLux の recall filter の閾値はいくつに凍結した？"
        started = time.perf_counter()
        rows, strategy = search.search(query, semantic=False, read_only_snapshot=True)
        elapsed = (time.perf_counter() - started) * 1000
        stages = cast(dict[str, int], search.last_search_trace()["stage_timings_ms"])
        assert hits == [True], hits
        assert cache_before == [(p.read_bytes(), p.stat().st_mtime_ns) for p in caches]
        assert stages["negative_feedback"] < 250, stages
        assert stages["bm25_snapshot"] < 400, stages
        return {"cache_hits": {"feedback": True, "golden": True, "lexical": True},
                "stages_ms": stages, "total_ms": round(elapsed, 3),
                "results": len(rows), "strategy": strategy}
    if mode == "feedback-equivalence":
        # Compare full validation and actual incremental cache paths, not injected entries.
        current_entries = nf._load_entries(load_negative_feedback_config())
        replay_path = Path(__file__).parents[1] / "2026-11-07-recall-cache/replay.py"
        spec = importlib.util.spec_from_file_location("previous_replay", replay_path)
        assert spec is not None and spec.loader is not None
        replay: Any = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(replay)
        replay.BASE = "7e9d4fee"
        old = replay.old_module("negative_feedback")
        old.PERSISTENT_CACHE_FILE = root / "unused-old-cache.json"
        old.find_mutation_page = cast(Any, nf).find_mutation_page
        old_entries = old._load_entries(old.load_negative_feedback_config())
        assert [nf._entry_payload(x) for x in current_entries] == [old._entry_payload(x) for x in old_entries]
        queries = json.loads((root / "queries.json").read_text())
        queries.append(json.loads((root / "benchmark-query.json").read_text()))
        mismatches = 0
        for query in queries:
            mismatches += nf.contextual_negative_trace(query) != old.contextual_negative_trace(query)
            mismatches += nf.penalties_for_query(query) != old.penalties_for_query(query)
        assert mismatches == 0
        return {"queries": len(queries), "entries": len(current_entries), "mismatches": mismatches}
    raise ValueError(mode)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("--source", type=Path)
    parser.add_argument("--child")
    args = parser.parse_args()
    root = args.root.resolve()
    if args.child:
        print(json.dumps(child(root, args.child)), flush=True)
        return
    assert args.source is not None and not root.exists()
    shutil.copytree(args.source, root)
    os.chmod(root, 0o700)
    for name in ("pages.json", "backlinks.json"):
        path = root / ".index" / name
        path.write_text(path.read_text().replace(str(args.source.resolve()), str(root)))
    (root / "config.toml").write_text(
        "[search.negative_feedback]\nenabled = true\nsimilarity_threshold = 0.35\n"
        "penalty = 0.85\nmax_age_days = 180\n"
    )
    results: dict[str, Any] = {}
    for mode in ("prime", "append", "open", "search", "search", "feedback-equivalence"):
        env = os.environ.copy()
        env.update(CHRONOVISOR_ROOT=str(root), CHRONOVISOR_READ_ONLY="1" if mode in {"search", "feedback-equivalence"} else "0")
        output = subprocess.check_output([sys.executable, __file__, str(root), "--child", mode], env=env, text=True)
        value = json.loads(output)
        results.setdefault(mode, []).append(value)
        print(mode, output.strip(), flush=True)
    (root / "sequence-result.json").write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
