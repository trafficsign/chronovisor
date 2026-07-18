from __future__ import annotations

import json
from pathlib import Path

from llm_wiki_mcp.cofire import build_cofire_graph, neighbors


def test_build_cofire_graph_counts_repeated_context_pairs(tmp_path: Path) -> None:
    log_file = tmp_path / "recall-log.jsonl"
    out_file = tmp_path / "cofire.json"
    rows = [
        {"pages": ["a", "b"]},
        {"context_items": [{"page_id": "a"}, {"page_id": "b"}, {"page_id": "c"}]},
    ]
    log_file.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")

    payload = build_cofire_graph(
        log_file=log_file,
        pull_log_file=tmp_path / "missing-pull-log.jsonl",
        output_file=out_file,
        min_count=2,
    )

    assert payload["episodes"] == 2
    assert payload["edges"] == 2
    assert neighbors("a", path=out_file)[0]["page_id"] == "b"


def test_cofire_keeps_used_graph_separate_from_exposure_graph(tmp_path: Path) -> None:
    log_file = tmp_path / "recall-log.jsonl"
    pull_file = tmp_path / "pull-log.jsonl"
    out_file = tmp_path / "cofire.json"
    log_file.write_text(
        json.dumps(
            {
                "decision_id": "decision-1",
                "session_id": "session-1",
                "pages": ["exposure-a", "exposure-b"],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    pull_file.write_text(
        json.dumps(
            {
                "type": "used",
                "event_id": "event-1",
                "decision_id": "decision-1",
                "session_id": "session-1",
                "page_ids": ["used-a", "used-b"],
            }
        )
        + "\n",
        encoding="utf-8",
    )

    payload = build_cofire_graph(
        log_file=log_file,
        pull_log_file=pull_file,
        output_file=out_file,
        min_count=1,
    )

    assert payload["graphs"]["positive_used"]["graph"]["used-a"][0]["page_id"] == "used-b"
    assert payload["graphs"]["exposure"]["graph"]["exposure-a"][0]["page_id"] == "exposure-b"
