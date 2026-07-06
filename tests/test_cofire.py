from __future__ import annotations

import json
from pathlib import Path

from llm_wiki_mcp.cofire import build_cofire_graph, neighbors


def test_build_cofire_graph_counts_repeated_context_pairs(tmp_path: Path) -> None:
    log_file = tmp_path / "recall-log.jsonl"
    out_file = tmp_path / "cofire.json"
    rows = [
        {"context_items": [{"page_id": "a"}, {"page_id": "b"}]},
        {"context_items": [{"page_id": "a"}, {"page_id": "b"}, {"page_id": "c"}]},
    ]
    log_file.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")

    payload = build_cofire_graph(log_file=log_file, output_file=out_file, min_count=2)

    assert payload["episodes"] == 2
    assert payload["edges"] == 2
    assert neighbors("a", path=out_file)[0]["page_id"] == "b"
