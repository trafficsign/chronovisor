"""CPU-only equivalence replay; production inputs are copied, never modified."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sqlite3
import subprocess
import sys
import time
import types
from collections.abc import Callable
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[3]
BASE = "2f5cb8be"


def old_module(name: str) -> Any:
    module_name = "chronovisor.core._baseline_" + name
    module = types.ModuleType(module_name)
    sys.modules[module_name] = module
    source = subprocess.check_output(
        ["git", "show", f"{BASE}:src/chronovisor/core/{name}.py"], cwd=REPO
    )
    exec(compile(source, f"{BASE}/{name}.py", "exec"), module.__dict__)
    return module


def timed[T](fn: Callable[[], T]) -> tuple[T, float]:
    started = time.perf_counter()
    result = fn()
    return result, round((time.perf_counter() - started) * 1000, 3)


def digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def capture(destination: Path) -> None:
    from chronovisor.core import negative_feedback as nf
    from chronovisor.core.page_mutation import find_mutation_page
    from chronovisor.core.runtime_config import load_negative_feedback_config

    production = Path.home() / ".chronovisor"
    destination.mkdir(mode=0o700)
    for folder in ["pages", "system", ".index", "recall"]:
        (destination / folder).mkdir()
    for name in ["feedback.jsonl", "search-golden.jsonl", "recall-log.jsonl"]:
        source = production / "recall" / name
        before = source.stat()
        shutil.copyfile(source, destination / "recall" / name)
        after = source.stat()
        assert (before.st_size, before.st_mtime_ns) == (after.st_size, after.st_mtime_ns)
    source_feedback = production / "recall/feedback.jsonl"
    stat = source_feedback.stat()
    cache = json.loads(nf.PERSISTENT_CACHE_FILE.read_text())
    assert cache["source"] == nf._persistent_key(source_feedback, mtime_ns=stat.st_mtime_ns, size=stat.st_size, config=load_negative_feedback_config())
    assert digest(source_feedback) == digest(destination / "recall/feedback.jsonl")
    (destination / "accepted-feedback.json").write_text(json.dumps(cache["entries"]))
    bound_pages = {}
    for entry in cache["entries"]:
        for page_id, _hash in entry["page_hashes"]:
            path = find_mutation_page(page_id)
            if path is not None:
                target = destination / "pages" / f"{page_id}.md"
                shutil.copyfile(path, target)
                bound_pages[page_id] = str(target)
    (destination / "bound-pages.json").write_text(json.dumps(bound_pages))
    for _attempt in range(10):
        docs = {name: (production / ".index" / name).read_bytes() for name in ["pages.json", "backlinks.json"]}
        target = destination / ".index/lexical.sqlite"
        target.unlink(missing_ok=True)
        source_db = sqlite3.connect(f"file:{production / '.index/lexical.sqlite'}?mode=ro", uri=True)
        copied = sqlite3.connect(target)
        source_db.backup(copied)
        generation = copied.execute("SELECT value FROM projection_meta WHERE key='generation'").fetchone()[0]
        copied.close()
        source_db.close()
        pages = json.loads(docs["pages.json"])
        links = json.loads(docs["backlinks.json"])
        if str(pages["generation"]) == str(links["generation"]) == generation:
            break
    else:
        raise RuntimeError("could not capture a coherent lexical generation")
    # Only canonical absolute path roots change; all ranking metadata is retained.
    for entry in pages["entries"].values():
        folder = "system" if entry["is_system"] else "pages"
        entry["path"] = str(destination / folder / entry["relative_path"])
    (destination / ".index/pages.json").write_text(json.dumps(pages, ensure_ascii=False))
    (destination / ".index/backlinks.json").write_bytes(docs["backlinks.json"])
    queries: list[str] = []
    with (destination / "recall/recall-log.jsonl").open() as stream:
        for line in stream:
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            queries.extend(q for q in row.get("queries", []) if isinstance(q, str) and q.strip())
    queries = list(dict.fromkeys(reversed(queries)))[:500]
    assert len(queries) >= 500, len(queries)
    (destination / "queries.json").write_text(json.dumps(queries, ensure_ascii=False))
    inputs = {name: digest(destination / "recall" / name) for name in ["feedback.jsonl", "search-golden.jsonl", "recall-log.jsonl"]}
    (destination / "capture.json").write_text(json.dumps({"input_sha256": inputs, "lexical_generation": generation, "query_count": len(queries)}, indent=2))


def configure(root: Path, baseline: bool) -> tuple[Any, Any]:
    from chronovisor.core import negative_feedback
    from chronovisor.core.lexical_index import LexicalIndex

    nf: Any = old_module("negative_feedback") if baseline else negative_feedback
    entries = [nf._entry_from_payload(x) for x in json.loads((root / "accepted-feedback.json").read_text())]
    # Both paths receive identical already-validated feedback; separately compare
    # full ledger validation below, including exact ref and page hash checks.
    nf._load_entries = lambda _config: entries
    nf.GOLDEN_FILE_OVERRIDE = root / "recall/search-golden.jsonl"
    pages = json.loads((root / "bound-pages.json").read_text())
    nf.find_mutation_page = lambda page_id: Path(pages[page_id]) if page_id in pages else None
    cls = old_module("lexical_index").LexicalIndex if baseline else LexicalIndex
    index = cls(path=root / ".index/lexical.sqlite", pages=lambda: [])
    return nf, index


def freeze_ledger_pages(root: Path) -> None:
    from chronovisor.core import feedback_ledger as ledger
    from chronovisor.core import negative_feedback as nf

    target_tokens = set(json.loads((root / "accepted-feedback.json").read_text())[0]["tokens"])
    page_ids: set[str] = set()
    with (root / "recall/feedback.jsonl").open() as stream:
        for line in stream:
            row = json.loads(line)
            if row.get("frontier_reviewed") and isinstance(row.get("prompt"), str) and nf._tokenize(row["prompt"]) == target_tokens:
                (root / "benchmark-query.json").write_text(json.dumps(row["prompt"]))
            if row.get("frontier_reviewed") is True and row.get("label_quality") == "strong":
                page_ids.update(p for p in row.get("negative_pages", []) if isinstance(p, str))
    copies = root / "ledger-pages"
    copies.mkdir(exist_ok=True)
    bindings: dict[str, dict[str, str | None]] = {}
    for page_id in sorted(page_ids):
        bindings[page_id] = {}
        for name in ["find_page", "find_mutation_page"]:
            source = getattr(ledger, name)(page_id)
            target = None
            if source is not None:
                before = source.stat()
                target = copies / (hashlib.sha256(str(source).encode()).hexdigest() + ".md")
                shutil.copyfile(source, target)
                after = source.stat()
                assert (before.st_size, before.st_mtime_ns) == (after.st_size, after.st_mtime_ns)
            bindings[page_id][name] = str(target) if target else None
    (root / "ledger-pages.json").write_text(json.dumps(bindings))


def benchmark(root: Path, baseline: bool) -> dict[str, float]:
    nf, index = configure(root, baseline)
    query = json.loads((root / "benchmark-query.json").read_text())
    _, negative_cold = timed(lambda: nf.contextual_negative_trace(query))
    _, negative_warm = timed(lambda: nf.contextual_negative_trace(query))
    ok, lexical_cold = timed(index.load_existing)
    assert ok
    ok, lexical_warm = timed(index.load_existing)
    assert ok
    index.close()
    return {"negative_first_ms": negative_cold, "negative_warm_ms": negative_warm, "lexical_first_ms": lexical_cold, "lexical_warm_ms": lexical_warm}


def replay(root: Path) -> dict[str, Any]:
    os.environ["CHRONOVISOR_READ_ONLY"] = "1"
    old, old_index = configure(root, True)
    new, new_index = configure(root, False)
    assert old_index.load_existing() and new_index.load_existing()
    queries = json.loads((root / "queries.json").read_text())
    # Add real reviewed negative prompts to avoid an all-empty negative cohort.
    extra = []
    with (root / "recall/search-golden.jsonl").open() as stream:
        for line in stream:
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if row.get("reviewed") is True and str(row.get("split") or "train") == "train" and row.get("negative_pages") and isinstance(row.get("query"), str):
                extra.append(row["query"])
    counts = dict.fromkeys(["trace", "penalties", "bm25_order_scores", "anchor_order_scores", "penalized_order_scores"], 0)
    nonempty = 0
    for i, query in enumerate(queries + extra):
        left = old.contextual_negative_trace(query)
        right = new.contextual_negative_trace(query)
        nonempty += bool(left)
        counts["trace"] += left != right
        lp = old.penalties_for_query(query)
        rp = new.penalties_for_query(query)
        counts["penalties"] += lp != rp
        before = old_index.query_existing(query, top_n=20)
        after = new_index.query_existing(query, top_n=20)
        counts["bm25_order_scores"] += before != after
        counts["anchor_order_scores"] += old_index.anchor_query_existing(query, top_n=20) != new_index.anchor_query_existing(query, top_n=20)
        # Real lexical candidates plus actual negative page IDs exercise demotion
        # even when a negatively labelled page isn't in today's top 20.
        from chronovisor.core.search_types import ScoredPage
        candidates = before + [ScoredPage(page_id=k, title=k, folder="", updated="", score=1.0) for k in left]
        counts["penalized_order_scores"] += old.apply_penalties(candidates, lp) != new.apply_penalties(candidates, rp)
        if (i + 1) % 100 == 0:
            print(f"replayed {i + 1}", flush=True)
    old_index.close()
    new_index.close()
    result = {"real_recall_queries": len(queries), "reviewed_negative_queries": len(extra), "nonempty_negative_traces": nonempty, "mismatches": counts}
    assert not any(counts.values()), result
    return result


def ledger(root: Path) -> dict[str, Any]:
    from chronovisor.core import feedback_ledger

    pages = json.loads((root / "ledger-pages.json").read_text())
    results = []
    times = []
    for module in [old_module("feedback_ledger"), feedback_ledger]:
        for name in ["find_page", "find_mutation_page"]:
            def resolve(page_id: str, name: str = name) -> Path | None:
                path = pages.get(page_id, {}).get(name)
                return Path(path) if path else None
            setattr(module, name, resolve)
        def validate(module: Any = module) -> list[dict[str, Any]]:
            result: list[dict[str, Any]] = module.trusted_negative_feedback_rows(root / "recall/feedback.jsonl", recall_log_file=root / "recall/recall-log.jsonl")
            return result
        value, elapsed = timed(validate)
        results.append(value)
        times.append(elapsed)
        print(f"ledger {module.__name__}: {elapsed} ms, {len(value)} rows", flush=True)
    assert results[0] == results[1]
    return {"before_ms": times[0], "after_ms": times[1], "mismatches": 0, "rows": len(results[0])}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["capture", "bench-old", "bench-new", "replay", "ledger"])
    parser.add_argument("root", type=Path)
    args = parser.parse_args()
    if args.mode == "capture":
        capture(args.root)
        freeze_ledger_pages(args.root)
    elif args.mode.startswith("bench-"):
        print(json.dumps(benchmark(args.root, args.mode == "bench-old")), flush=True)
    else:
        result = replay(args.root) if args.mode == "replay" else ledger(args.root)
        (args.root / f"{args.mode}-result.json").write_text(json.dumps(result, indent=2))
        print(json.dumps(result), flush=True)
