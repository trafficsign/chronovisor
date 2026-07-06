from __future__ import annotations

import json
from pathlib import Path

from llm_wiki_mcp.prefetch import build_prefetch_cache, prefetch_page_ids


def test_prefetch_cache_matches_project_bucket_and_tokens(tmp_path: Path) -> None:
    log_file = tmp_path / "recall-log.jsonl"
    out_file = tmp_path / "prefetch.json"
    row = {
        "host": "codex",
        "cwd": "/Users/trafficsign/projects/personal/llm-wiki-mcp",
        "prompt": "recall hook",
        "queries": ["recall runtime"],
        "context_items": [{"page_id": "recall-runtime"}],
    }
    log_file.write_text(json.dumps(row) + "\n", encoding="utf-8")

    payload = build_prefetch_cache(log_file=log_file, output_file=out_file)
    pages = prefetch_page_ids(
        host="codex",
        cwd="/Users/trafficsign/projects/personal/llm-wiki-mcp",
        queries=["runtime"],
        path=out_file,
    )

    assert payload["episodes"] == 1
    assert pages == ["recall-runtime"]
