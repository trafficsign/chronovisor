from __future__ import annotations

import json
from pathlib import Path

from chronovisor.recall.recall_prefetch import build_prefetch_cache
from chronovisor.search.prefetch import prefetch_page_ids


def test_prefetch_cache_matches_project_bucket_and_tokens(tmp_path: Path) -> None:
    log_file = tmp_path / "recall-log.jsonl"
    out_file = tmp_path / "prefetch.json"
    row = {
        "host": "codex",
        "cwd": "/Users/trafficsign/projects/personal/chronovisor",
        "prompt_preview": "recall hook",
        "queries": ["recall runtime"],
        "pages": ["recall-runtime"],
    }
    log_file.write_text(json.dumps(row) + "\n", encoding="utf-8")

    payload = build_prefetch_cache(
        log_file=log_file,
        pull_log_file=tmp_path / "missing-pull-log.jsonl",
        output_file=out_file,
    )
    pages = prefetch_page_ids(
        host="codex",
        cwd="/Users/trafficsign/projects/personal/chronovisor",
        queries=["runtime"],
        path=out_file,
    )

    assert payload["episodes"] == 1
    assert payload["positive_episodes"] == 0
    assert pages == ["recall-runtime"]


def test_prefetch_keeps_used_supervision_separate_from_exposure(tmp_path: Path) -> None:
    log_file = tmp_path / "recall-log.jsonl"
    pull_file = tmp_path / "pull-log.jsonl"
    out_file = tmp_path / "prefetch.json"
    log_file.write_text(
        json.dumps(
            {
                "host": "codex",
                "cwd": "/tmp/project",
                "decision_id": "decision-1",
                "session_id": "session-1",
                "prompt_preview": "recall runtime",
                "pages": ["exposed-page"],
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
                "page_ids": ["used-page"],
            }
        )
        + "\n",
        encoding="utf-8",
    )

    payload = build_prefetch_cache(
        log_file=log_file,
        pull_log_file=pull_file,
        output_file=out_file,
    )

    positive = payload["features"]["positive_used"]["buckets"]["codex|project"]
    exposure = payload["features"]["exposure"]["buckets"]["codex|project"]
    assert positive == [{"page_id": "used-page", "count": 1}]
    assert exposure == [{"page_id": "exposed-page", "count": 1}]
