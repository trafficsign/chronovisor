from __future__ import annotations

import json
from pathlib import Path

import chronovisor.core.prefetch as prefetch
from chronovisor.core.prefetch import build_prefetch_cache, prefetch_page_ids


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
    assert "positive_used" not in payload.get("features", {})
    assert pages == ["recall-runtime"]


def test_prefetch_ignores_used_receipts_and_keeps_exposure_only(
    tmp_path: Path, monkeypatch
) -> None:
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

    exposure = payload["features"]["exposure"]["buckets"]["codex|project"]
    assert exposure == [{"page_id": "exposed-page", "count": 1}]
    assert "positive_used" not in payload["features"]
    assert "used-page" not in json.dumps(payload)
    assert (
        prefetch_page_ids(
            host="codex",
            cwd="/tmp/project",
            queries=["recall"],
            path=out_file,
            positive_weight=100,
            exposure_weight=0,
        )
        == []
    )

    db_file = tmp_path / "prefetch.sqlite"
    monkeypatch.setattr(prefetch, "PREFETCH_FILE", out_file)
    monkeypatch.setattr(prefetch, "PREFETCH_DB_FILE", db_file)
    build_prefetch_cache(
        log_file=log_file,
        pull_log_file=pull_file,
        output_file=out_file,
    )
    assert (
        prefetch_page_ids(
            host="codex",
            cwd="/tmp/project",
            queries=["recall"],
            path=out_file,
            positive_weight=100,
            exposure_weight=0,
        )
        == []
    )
    assert prefetch_page_ids(
        host="codex",
        cwd="/tmp/project",
        queries=["recall"],
        path=out_file,
        positive_weight=100,
        exposure_weight=1,
    ) == ["exposed-page"]
