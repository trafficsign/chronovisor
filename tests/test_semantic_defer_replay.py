from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import pytest

from llm_wiki_mcp import raw_replay


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def _isolate_paths(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict[str, Path]:
    paths = {
        "raw": tmp_path / "raw",
        "queue": tmp_path / "review" / "raw-replay-queue.jsonl",
        "history": tmp_path / "runtime" / "raw-replay-history.jsonl",
        "completions": tmp_path / "runtime" / "raw-replay-completions.jsonl",
        "memory": tmp_path / "eval" / "memory-integrity-latest.json",
        "claims": tmp_path / "claims" / "claims.jsonl",
        "failure_log": tmp_path / "runtime" / "ingest-read-back-failures.jsonl",
        "runtime_status": tmp_path / "runtime" / "status.json",
        "packets": tmp_path / "runtime" / "failures" / "packets",
        "quarantine": tmp_path / "runtime" / "failures" / "quarantined-raw",
    }
    paths["raw"].mkdir(parents=True)
    monkeypatch.setattr(raw_replay, "RAW_DIR", paths["raw"])
    monkeypatch.setattr(raw_replay, "QUEUE_FILE", paths["queue"])
    monkeypatch.setattr(raw_replay, "HISTORY_FILE", paths["history"])
    monkeypatch.setattr(raw_replay, "COMPLETIONS_FILE", paths["completions"])
    monkeypatch.setattr(raw_replay, "MEMORY_INTEGRITY_FILE", paths["memory"])
    monkeypatch.setattr(raw_replay, "CLAIMS_FILE", paths["claims"])
    monkeypatch.setattr(raw_replay, "INGEST_FAILURE_LOG_FILE", paths["failure_log"])
    monkeypatch.setattr(raw_replay, "RUNTIME_STATUS_FILE", paths["runtime_status"])
    monkeypatch.setattr(raw_replay, "FAILURE_PACKETS_DIR", paths["packets"])
    monkeypatch.setattr(raw_replay, "QUARANTINED_RAW_DIR", paths["quarantine"])
    return paths


def test_auto_signals_skip_semantic_defer_until_authority_artifact_changes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from llm_wiki_mcp import (
        decision_router,
        failure_supervisor,
        runtime_config,
        runtime_status,
        wiki,
    )

    paths = _isolate_paths(tmp_path, monkeypatch)
    monkeypatch.setattr(wiki, "WIKI_ROOT", tmp_path)
    monkeypatch.setattr(wiki, "RAW_DIR", paths["raw"])
    monkeypatch.setattr(runtime_status, "RUNTIME_DIR", tmp_path / "runtime")
    monkeypatch.setattr(
        runtime_status,
        "STATUS_FILE",
        tmp_path / "runtime" / "status.json",
    )
    monkeypatch.setattr(
        runtime_status,
        "EVENTS_FILE",
        tmp_path / "runtime" / "events.jsonl",
    )
    monkeypatch.setattr(
        runtime_status,
        "METRICS_FILE",
        tmp_path / "runtime" / "metrics.jsonl",
    )
    raw = paths["raw"] / "20260714-semantic-split.md"
    raw.write_text("immutable source", encoding="utf-8")

    paths["memory"].parent.mkdir(parents=True)
    paths["memory"].write_text(
        json.dumps(
            {
                "rows": [
                    {
                        "raw": raw.name,
                        "path": str(raw),
                        "status": "miss",
                        "query": "semantic split",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    paths["runtime_status"].parent.mkdir(parents=True, exist_ok=True)
    paths["runtime_status"].write_text(
        json.dumps(
            {
                "last_success": {
                    "source_raw": raw.name,
                    "failed_ops": [{"filename": "memory/split.md"}],
                }
            }
        ),
        encoding="utf-8",
    )

    authority_artifact = tmp_path / "adopted-router.json"
    authority_artifact.write_bytes(b"authority epoch one")
    monkeypatch.setattr(
        runtime_config,
        "load_decision_router_config",
        lambda: SimpleNamespace(adoption_artifact=str(authority_artifact)),
    )
    monkeypatch.setattr(
        decision_router,
        "resolve_router_policy",
        lambda config: SimpleNamespace(
            source="adopted_artifact",
            error=None,
            artifact_sha256=hashlib.sha256(
                Path(config.adoption_artifact).read_bytes()
            ).hexdigest(),
        ),
    )
    deferred_authority_sha256 = hashlib.sha256(
        authority_artifact.read_bytes()
    ).hexdigest()
    supervision = failure_supervisor.record_raw_failure(
        raw_path=raw,
        error=(
            "local consensus semantic no quorum "
            f"[authority_sha256={deferred_authority_sha256}]: "
            "local_models_did_not_reach_two_vote_quorum"
        ),
    )
    assert supervision.terminal_deferred is True

    held = raw_replay.build_queue(
        path=paths["queue"],
        include_migration=True,
        include_auto_signals=True,
    )

    assert held["candidates"] == 0
    assert held["candidate_keys"] == []
    assert held["skipped_semantic_deferred"] == 1
    assert paths["queue"].read_text(encoding="utf-8") == ""

    authority_artifact.write_bytes(b"authority epoch two")
    released = raw_replay.build_queue(
        path=paths["queue"],
        include_migration=True,
        include_auto_signals=True,
    )

    assert released["candidates"] == 1
    assert released["candidate_keys"] == [raw_replay.stable_key(raw)]
    assert released["skipped_semantic_deferred"] == 0


@pytest.mark.parametrize(
    ("queue_status", "attempts", "extra_fields"),
    [
        ("pending", 0, {}),
        (
            "quarantined",
            3,
            {"quarantined_at": "2000-01-01T00:00:00+00:00"},
        ),
    ],
    ids=("pending", "cooldown-quarantined"),
)
def test_existing_queue_row_does_not_cooldown_reopen_or_launch_while_deferred(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    queue_status: str,
    attempts: int,
    extra_fields: dict[str, str],
) -> None:
    paths = _isolate_paths(tmp_path, monkeypatch)
    raw = paths["raw"] / "20260714-semantic-split.md"
    raw.write_text("immutable source", encoding="utf-8")
    _write_jsonl(
        paths["queue"],
        [
            {
                "key": raw_replay.stable_key(raw),
                "raw": raw.name,
                "path": str(raw),
                "status": queue_status,
                "attempts": attempts,
                "sources": ["ingest_failure"],
                "priority": 300,
                "bytes": raw.stat().st_size,
                "date": "20260714",
                **extra_fields,
            }
        ],
    )
    deferred = {raw.name}
    monkeypatch.setattr(
        raw_replay,
        "_active_terminal_semantic_deferred_raw_names",
        lambda: frozenset(deferred),
    )
    monkeypatch.setenv("LLM_WIKI_CONVERGENCE_QUARANTINE_RETRY_SECONDS", "0")
    monkeypatch.setattr(
        "llm_wiki_mcp.ingest.run_ingest",
        lambda *_args, **_kwargs: pytest.fail("semantic defer must not launch ingest"),
    )

    held = raw_replay.run_pending_queue(
        path=paths["queue"],
        history_file=paths["history"],
        claims_file=paths["claims"],
        completions_file=paths["completions"],
        max_runs=1,
        max_bytes=100,
        now=datetime(2026, 7, 14, tzinfo=timezone.utc),
    )

    assert held["count"] == 0
    [held_row] = [
        json.loads(line)
        for line in paths["queue"].read_text(encoding="utf-8").splitlines()
    ]
    assert held_row["status"] == queue_status
    assert "quarantine_resumed_at" not in held_row

    deferred.clear()
    released = raw_replay.run_pending_queue(
        path=paths["queue"],
        history_file=paths["history"],
        claims_file=paths["claims"],
        completions_file=paths["completions"],
        max_runs=1,
        max_bytes=100,
        dry_run=True,
        now=datetime(2026, 7, 14, tzinfo=timezone.utc),
    )

    assert released["count"] == 1
    assert released["planned"][0]["raw"] == raw.name


def test_semantic_defer_skips_indeterminate_frontier_reconciliation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths = _isolate_paths(tmp_path, monkeypatch)
    raw = paths["raw"] / "20260714-indeterminate-split.md"
    raw.write_text("immutable source", encoding="utf-8")
    _write_jsonl(
        paths["queue"],
        [
            {
                "key": raw_replay.stable_key(raw),
                "raw": raw.name,
                "path": str(raw),
                "status": "indeterminate",
                "attempts": 1,
                "sources": ["ingest_failure"],
                "priority": 300,
                "bytes": raw.stat().st_size,
                "date": "20260714",
            }
        ],
    )
    monkeypatch.setattr(
        raw_replay,
        "_active_terminal_semantic_deferred_raw_names",
        lambda: frozenset({raw.name}),
    )

    result = raw_replay.run_pending_queue(
        path=paths["queue"],
        history_file=paths["history"],
        claims_file=paths["claims"],
        completions_file=paths["completions"],
        max_runs=1,
        max_bytes=100,
        frontier_reviewer=lambda *_args, **_kwargs: pytest.fail(
            "semantic defer must not invoke frontier reconciliation"
        ),
    )

    assert result["count"] == 0
    assert result["frontier_reconciliation"]["reviewed"] == 0


def test_semantic_defer_published_during_budget_preflight_blocks_launch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths = _isolate_paths(tmp_path, monkeypatch)
    raw = paths["raw"] / "20260714-budget-race.md"
    raw.write_text("immutable source", encoding="utf-8")
    key = raw_replay.stable_key(raw)
    _write_jsonl(
        paths["queue"],
        [
            {
                "key": key,
                "raw": raw.name,
                "path": str(raw),
                "status": "pending",
                "attempts": 0,
                "sources": ["ingest_failure"],
                "priority": 300,
                "bytes": raw.stat().st_size,
                "date": "20260714",
            }
        ],
    )
    deferred: set[str] = set()
    monkeypatch.setattr(
        raw_replay,
        "_active_terminal_semantic_deferred_raw_names",
        lambda: frozenset(deferred),
    )
    consume_calls: list[tuple[str, int]] = []

    class PublishingBudget:
        def can_consume(self, kind: str, amount: int = 1) -> tuple[bool, str]:
            if kind == "raw_bytes":
                deferred.add(raw.name)
            return True, "ok"

        def consume(self, kind: str, amount: int = 1) -> tuple[bool, str]:
            consume_calls.append((kind, amount))
            return True, "ok"

    monkeypatch.setattr(
        raw_replay,
        "job_store",
        SimpleNamespace(
            create=lambda **_kwargs: pytest.fail(
                "semantic defer must not create a replay job"
            )
        ),
    )
    monkeypatch.setattr(
        raw_replay,
        "_run_candidate",
        lambda *_args, **_kwargs: pytest.fail("semantic defer must not invoke replay"),
    )

    result = raw_replay.run_pending_queue(
        path=paths["queue"],
        history_file=paths["history"],
        claims_file=paths["claims"],
        completions_file=paths["completions"],
        max_runs=1,
        max_bytes=100,
        budget=PublishingBudget(),
        now=datetime(2026, 7, 14, tzinfo=timezone.utc),
    )

    assert result["count"] == 0
    assert result["budget_deferred"] == []
    assert result["semantic_deferred"] == [
        {"key": key, "raw": raw.name, "reason": "semantic_no_quorum"}
    ]
    assert consume_calls == []
    [queue_row] = [
        json.loads(line)
        for line in paths["queue"].read_text(encoding="utf-8").splitlines()
    ]
    assert queue_row["status"] == "pending"
    assert "job_id" not in queue_row
