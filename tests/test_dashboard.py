"""Tests for dashboard data assembly."""

from __future__ import annotations

import json
import os
from datetime import date, datetime
from pathlib import Path
from types import SimpleNamespace

from llm_wiki_mcp import dashboard, runtime_status


def test_mark_batch_activity_requires_a_running_batch_job() -> None:
    completed = {
        "state": "idle",
        "stage": "idle",
        "current_job_id": None,
        "batch": {"index": 2, "total": 2, "succeeded": 2, "failed": 0},
    }
    dashboard._mark_batch_activity(completed)
    assert completed["batch"]["active"] is False

    running = {
        "state": "running",
        "stage": "raw",
        "current_job_id": "job-1",
        "batch": {"index": 1, "total": 2, "succeeded": 0, "failed": 0},
    }
    dashboard._mark_batch_activity(running)
    assert running["batch"]["active"] is True

    waiting = {
        "state": "running",
        "stage": "waiting",
        "current_job_id": None,
        "batch": {"index": 2, "total": 2, "succeeded": 2, "failed": 0},
    }
    dashboard._mark_batch_activity(waiting)
    assert waiting["batch"]["active"] is False


def test_build_snapshot_combines_runtime_and_queue(tmp_path: Path, monkeypatch) -> None:
    wiki_root = tmp_path / "wiki"
    raw_dir = wiki_root / "raw"
    runtime_dir = wiki_root / "runtime"
    raw_dir.mkdir(parents=True)
    (wiki_root / "pages").mkdir()
    (wiki_root / "system").mkdir()
    (raw_dir / "r1.md").write_text("raw")

    monkeypatch.setattr(dashboard, "WIKI_ROOT", wiki_root)
    monkeypatch.setattr(dashboard, "LOG_FILE", wiki_root / "log.md")
    monkeypatch.setattr(runtime_status, "RUNTIME_DIR", runtime_dir)
    monkeypatch.setattr(runtime_status, "STATUS_FILE", runtime_dir / "status.json")
    monkeypatch.setattr(runtime_status, "EVENTS_FILE", runtime_dir / "events.jsonl")
    monkeypatch.setattr(runtime_status, "METRICS_FILE", runtime_dir / "metrics.jsonl")

    from llm_wiki_mcp import orchestrator, wiki

    monkeypatch.setattr(wiki, "WIKI_ROOT", wiki_root)
    monkeypatch.setattr(wiki, "RAW_DIR", raw_dir)
    monkeypatch.setattr(wiki, "PAGES_DIR", wiki_root / "pages")
    monkeypatch.setattr(wiki, "SYSTEM_DIR", wiki_root / "system")
    monkeypatch.setattr(wiki, "INDEX_FILE", wiki_root / "index.md")
    monkeypatch.setattr(wiki, "LOG_FILE", wiki_root / "log.md")
    monkeypatch.setattr(orchestrator, "RAW_DIR", raw_dir)
    monkeypatch.setattr(orchestrator, "STATE_FILE", wiki_root / ".orchestrator_state.json")
    monkeypatch.setattr(dashboard, "_ollama_snapshot", lambda: {"available": False, "models": []})
    monkeypatch.setattr(
        dashboard,
        "_model_status_snapshot",
        lambda ollama=None: {"available": False, "models": [], "summary": {"installed": 0, "loaded": 0}},
    )
    monkeypatch.setattr(
        dashboard,
        "_frontier_preflight_snapshot",
        lambda: {"ok": True, "checked_at": "2026-06-01T12:00:00"},
    )

    runtime_status.write_status({"state": "running", "stage": "generate"})
    runtime_status.append_event("info", "ingest | stage 1: triage started")
    runtime_status.append_metric("batch", pending_before=2, pending_after=1, files_processed=1)
    frontier_activity_dir = runtime_dir / "frontier-reviews" / "active"
    frontier_activity_dir.mkdir(parents=True)
    (frontier_activity_dir / "review-1.json").write_text(
        json.dumps(
            {
                "review_id": "review-1",
                "active": True,
                "kind": "semantic_judge",
                "reviewer": "codex",
                "model": "gpt-5.5",
                "pid": os.getpid(),
                "started_at": datetime.now().isoformat(timespec="seconds"),
            }
        ),
        encoding="utf-8",
    )
    failures_dir = runtime_dir / "failures"
    packets_dir = failures_dir / "packets"
    packets_dir.mkdir(parents=True)
    packet = {
        "failure_id": "f1",
        "created_at": "2026-06-01T12:00:00",
        "raw_file": "broken.md",
        "failure_class": "apply.update_target_not_found",
        "status": "local_repair_applied",
    }
    (packets_dir / "f1.json").write_text(json.dumps(packet), encoding="utf-8")
    registry_record = {
        "timestamp": "2026-06-01T12:01:00",
        "failure_id": "f1",
        "raw_file": "broken.md",
        "failure_class": "apply.update_target_not_found",
        "resolution": "local",
        "decision": {
            "action": "resolve_update_target",
            "requested_page_id": "missing",
            "target_page_id": "ai/target",
            "source": "qwen",
        },
        "action": {
            "alias": {"requested": "missing", "target": "ai/target"},
            "retry": {"files_processed": ["broken.md"]},
        },
    }
    (failures_dir / "failure-registry.jsonl").write_text(
        json.dumps(registry_record) + "\n",
        encoding="utf-8",
    )

    snapshot = dashboard.build_snapshot()

    assert snapshot["status"]["pending"] == 1
    assert snapshot["status"]["state"] == "running"
    assert snapshot["events"]
    assert snapshot["metrics"][0]["pending_after"] == 1
    assert snapshot["self_heal"]["status"] == "resolved"
    assert snapshot["self_heal"]["counts"]["resolved"] == 1
    assert snapshot["self_heal"]["latest"]["raw_file"] == "broken.md"
    assert "alias missing -> ai/target" in snapshot["self_heal"]["latest"]["detail"]
    assert snapshot["self_heal"]["latest"]["details"]["failure"]["packet_status"] == "local_repair_applied"
    assert snapshot["self_heal"]["latest"]["details"]["decision"]["source"] == "qwen"
    assert snapshot["self_heal"]["latest"]["details"]["action"]["retry"]["files_processed"] == ["broken.md"]
    assert snapshot["self_heal"]["watch"]["packets"]["total"] == 1
    assert snapshot["self_heal"]["watch"]["frontier_preflight"]["ok"] is True
    assert snapshot["frontier_review"]["active"] is True
    assert snapshot["frontier_review"]["count"] == 1
    assert snapshot["status"]["frontier_review"]["latest"]["model"] == "gpt-5.5"


def test_build_snapshot_surfaces_frontier_human_required(
    tmp_path: Path, monkeypatch
) -> None:
    wiki_root = tmp_path / "wiki"
    raw_dir = wiki_root / "raw"
    runtime_dir = wiki_root / "runtime"
    raw_dir.mkdir(parents=True)
    (wiki_root / "pages").mkdir()
    (wiki_root / "system").mkdir()

    monkeypatch.setattr(dashboard, "WIKI_ROOT", wiki_root)
    monkeypatch.setattr(dashboard, "LOG_FILE", wiki_root / "log.md")
    monkeypatch.setattr(runtime_status, "RUNTIME_DIR", runtime_dir)
    monkeypatch.setattr(runtime_status, "STATUS_FILE", runtime_dir / "status.json")
    monkeypatch.setattr(runtime_status, "EVENTS_FILE", runtime_dir / "events.jsonl")
    monkeypatch.setattr(runtime_status, "METRICS_FILE", runtime_dir / "metrics.jsonl")

    from llm_wiki_mcp import orchestrator, wiki

    monkeypatch.setattr(wiki, "WIKI_ROOT", wiki_root)
    monkeypatch.setattr(wiki, "RAW_DIR", raw_dir)
    monkeypatch.setattr(wiki, "PAGES_DIR", wiki_root / "pages")
    monkeypatch.setattr(wiki, "SYSTEM_DIR", wiki_root / "system")
    monkeypatch.setattr(wiki, "INDEX_FILE", wiki_root / "index.md")
    monkeypatch.setattr(wiki, "LOG_FILE", wiki_root / "log.md")
    monkeypatch.setattr(orchestrator, "RAW_DIR", raw_dir)
    monkeypatch.setattr(orchestrator, "STATE_FILE", wiki_root / ".orchestrator_state.json")
    monkeypatch.setattr(dashboard, "_ollama_snapshot", lambda: {"available": False, "models": []})
    monkeypatch.setattr(
        dashboard,
        "_model_status_snapshot",
        lambda ollama=None: {"available": False, "models": [], "summary": {"installed": 0, "loaded": 0}},
    )
    monkeypatch.setattr(
        dashboard,
        "_frontier_preflight_snapshot",
        lambda: {"ok": False, "checked_at": "2026-06-04T22:00:00"},
    )

    runtime_status.write_status({"state": "idle"})
    failures_dir = runtime_dir / "failures"
    packets_dir = failures_dir / "packets"
    packets_dir.mkdir(parents=True)
    packet = {
        "failure_id": "auth1",
        "created_at": "2026-06-04T22:00:00",
        "updated_at": "2026-06-04T22:01:00",
        "raw_file": "auth-broken.md",
        "failure_class": "triage.parse_failed",
        "fingerprint": "triage.parse_failed",
        "status": "human_required",
        "frontier_result": {
            "decision": "needs_retry",
            "summary": "codex auth missing",
            "rescue_status": "human_required",
            "human_required": True,
            "frontier_failure": {"failure_class": "auth_required"},
            "access_repair": {"applied": True, "repairs": [{"type": "cli_option_adapted"}]},
        },
        "human_notification": {
            "body": "Codex の認証が切れている可能性があります。ログイン確認が必要です。",
            "delivery": {"sent": True},
        },
        "pending_frontier_review_path": str(failures_dir / "pending-frontier-review" / "auth1.json"),
    }
    (packets_dir / "auth1.json").write_text(json.dumps(packet), encoding="utf-8")
    registry_record = {
        "timestamp": "2026-06-04T22:02:00",
        "failure_id": "auth1",
        "raw_file": "auth-broken.md",
        "failure_class": "triage.parse_failed",
        "fingerprint": "triage.parse_failed",
        "resolution": "frontier",
        "decision": {"status": "escalate", "action": "escalate_to_frontier"},
        "frontier": packet["frontier_result"],
        "human_notification": packet["human_notification"],
        "pending_frontier_review_path": packet["pending_frontier_review_path"],
    }
    (failures_dir / "failure-registry.jsonl").write_text(
        json.dumps(registry_record, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    snapshot = dashboard.build_snapshot()

    self_heal = snapshot["self_heal"]
    assert self_heal["status"] == "failed"
    assert self_heal["counts"]["human_required"] == 1
    assert self_heal["counts"]["failed"] == 1
    assert self_heal["latest"]["title"] == "Human required"
    assert self_heal["latest"]["details"]["frontier"]["human_required"] is True
    assert self_heal["latest"]["details"]["frontier"]["access_repair"]["applied"] is True
    assert self_heal["latest"]["details"]["human_notification"]["delivery"]["sent"] is True
    assert self_heal["latest"]["details"]["pending_frontier_review_path"] == packet["pending_frontier_review_path"]
    assert self_heal["watch"]["packets"]["failed"] == 1
    assert self_heal["watch"]["frontier_preflight"]["ok"] is False


def test_model_status_snapshot_combines_ollama_and_config(monkeypatch) -> None:
    monkeypatch.setattr(dashboard, "ingest_model", lambda: "qwen3.6:35b-a3b-mxfp8")
    monkeypatch.setattr(
        dashboard,
        "_ollama_snapshot",
        lambda: {
            "available": True,
            "models": [
                {
                    "name": "qwen3.6:35b-a3b-mxfp8",
                    "model": "qwen3.6:35b-a3b-mxfp8",
                    "size": 39_000,
                    "size_vram": 39_000,
                    "context_length": 32768,
                    "expires_at": "2026-07-05T20:11:28+09:00",
                    "details": {"format": "safetensors", "quantization_level": "mxfp8"},
                },
                {
                    "name": "bge-m3:latest",
                    "model": "bge-m3:latest",
                    "size": 664,
                    "size_vram": 664,
                    "context_length": 8192,
                    "details": {"format": "gguf", "quantization_level": "F16"},
                },
            ],
        },
    )
    monkeypatch.setattr(
        dashboard,
        "_ollama_tags_snapshot",
        lambda: {
            "available": True,
            "models": [
                {
                    "name": "qwen3.6:35b-a3b-mxfp8",
                    "model": "qwen3.6:35b-a3b-mxfp8",
                    "size": 37_000,
                    "modified_at": "2026-07-05T18:04:01+09:00",
                    "details": {"format": "safetensors", "quantization_level": "mxfp8"},
                    "capabilities": ["completion", "tools"],
                },
                {
                    "name": "gemma4:26b-mxfp8",
                    "model": "gemma4:26b-mxfp8",
                    "size": 27_000,
                    "details": {"format": "safetensors", "quantization_level": "mxfp8"},
                    "capabilities": ["completion"],
                },
                {
                    "name": "bge-m3:latest",
                    "model": "bge-m3:latest",
                    "size": 1_157,
                    "details": {"format": "gguf", "quantization_level": "F16"},
                    "capabilities": ["embedding"],
                },
                {
                    "name": "gpt-oss:20b",
                    "model": "gpt-oss:20b",
                    "size": 13_000,
                    "details": {"format": "gguf", "quantization_level": "MXFP4"},
                    "capabilities": ["completion"],
                },
            ],
        },
    )
    monkeypatch.setattr(
        dashboard,
        "load_audit_policy",
        lambda: SimpleNamespace(enabled=True, model="qwen3.6:35b-a3b-mxfp8"),
    )
    monkeypatch.setattr(
        dashboard.recall_runtime,
        "load_policy",
        lambda: SimpleNamespace(
            judge_mode="auto",
            judge_model="qwen3.5:4b-mlx",
            rewrite_enabled=True,
            rewrite_model="qwen3.5:4b-mlx",
        ),
    )
    monkeypatch.setattr(
        dashboard,
        "configured_models",
        lambda models=None: ("qwen3.6:35b-a3b-mxfp8", "gemma4:26b-mxfp8"),
    )
    monkeypatch.setattr(dashboard, "embedding_model", lambda: "bge-m3")
    monkeypatch.setattr(
        dashboard,
        "load_reranker_config",
        lambda: SimpleNamespace(enabled=False, model="BAAI/bge-reranker-v2-m3"),
    )

    snapshot = dashboard._model_status_snapshot()
    by_name = {row["name"]: row for row in snapshot["models"]}

    assert snapshot["summary"]["installed"] == 3
    assert snapshot["summary"]["all_installed"] == 4
    assert snapshot["summary"]["unused_installed"] == 1
    assert snapshot["summary"]["loaded"] == 2
    assert snapshot["summary"]["configured"] == 4
    assert by_name["qwen3.6:35b-a3b-mxfp8"]["status"] == "loaded"
    assert by_name["qwen3.6:35b-a3b-mxfp8"]["roles"] == ["ingest", "audit", "improve"]
    assert by_name["gemma4:26b-mxfp8"]["status"] == "ready"
    assert by_name["gemma4:26b-mxfp8"]["roles"] == ["improve"]
    assert by_name["bge-m3:latest"]["roles"] == ["embed"]
    assert "bge-m3" not in by_name
    assert "gpt-oss:20b" not in by_name
    assert by_name["qwen3.5:4b-mlx"]["status"] == "missing"
    assert by_name["qwen3.5:4b-mlx"]["roles"] == ["gate", "rewrite"]


def test_self_heal_snapshot_surfaces_watch_status(tmp_path: Path, monkeypatch) -> None:
    wiki_root = tmp_path / "wiki"
    failures_dir = wiki_root / "runtime" / "failures"
    packets_dir = failures_dir / "packets"
    logs_dir = wiki_root / "logs"
    packets_dir.mkdir(parents=True)
    logs_dir.mkdir(parents=True)

    monkeypatch.setattr(dashboard, "WIKI_ROOT", wiki_root)
    monkeypatch.setattr(
        dashboard,
        "_frontier_preflight_snapshot",
        lambda: {
            "ok": True,
            "checked_at": "2026-07-04T19:30:00",
            "codex_version_ok": True,
            "exec_help_ok": True,
        },
    )

    pending_packet = {
        "failure_id": "queued1",
        "created_at": "2026-07-04T19:00:00",
        "raw_file": "queued.md",
        "failure_class": "triage.parse_failed",
        "status": "pending_frontier",
    }
    approved_packet = {
        "failure_id": "ok1",
        "created_at": "2026-07-04T18:00:00",
        "raw_file": "ok.md",
        "failure_class": "apply.update_target_not_found",
        "status": "frontier_approved",
    }
    (packets_dir / "queued1.json").write_text(json.dumps(pending_packet), encoding="utf-8")
    (packets_dir / "ok1.json").write_text(json.dumps(approved_packet), encoding="utf-8")
    (logs_dir / "ingest-drain-20260704.jsonl").write_text(
        json.dumps(
            {
                "timestamp": "2026-07-04T19:31:00",
                "self_heal": {"status": "ok", "packets_seen": 1, "results": [{"status": "pending_frontier"}]},
            }
        )
        + "\n",
        encoding="utf-8",
    )

    self_heal = dashboard._self_heal_snapshot()

    watch = self_heal["watch"]
    assert watch["last_checked"]["timestamp"] == "2026-07-04T19:31:00"
    assert watch["last_checked"]["packets_seen"] == 1
    assert watch["last_checked"]["results"] == 1
    assert watch["packets"]["total"] == 2
    assert watch["packets"]["pending"] == 1
    assert watch["packets"]["failed"] == 0
    assert watch["packets"]["status_counts"]["pending_frontier"] == 1
    assert watch["frontier_preflight"]["ok"] is True


def test_recall_snapshot_reads_logs_and_eval(tmp_path: Path, monkeypatch) -> None:
    wiki_root = tmp_path / "wiki"
    recall_dir = wiki_root / "recall"
    eval_dir = wiki_root / "runtime" / "eval"
    recall_dir.mkdir(parents=True)
    eval_dir.mkdir(parents=True)

    monkeypatch.setattr(dashboard, "WIKI_ROOT", wiki_root)

    records = [
        {
            "ts": "2026-06-11T10:00:00",
            "decision": "search",
            "latency_ms": 120,
            "used_judge": False,
            "evidence_features": {"rewrite_confidence": 0.8},
            "pages": ["a", "b"],
            "prompt_preview": "あれどうなった?",
            "host": "codex",
        },
        {
            "ts": "2026-06-11T10:01:00",
            "decision": "none",
            "latency_ms": 40,
            "used_judge": True,
            "evidence_features": {},
            "pages": [],
            "prompt_preview": "こんにちは",
            "host": "claude-code",
        },
    ]
    (recall_dir / "recall-log.jsonl").write_text(
        "\n".join(json.dumps(r) for r in records) + "\n", encoding="utf-8"
    )
    (recall_dir / "pull-log.jsonl").write_text(
        json.dumps({"type": "read", "page_id": "a"}) + "\n", encoding="utf-8"
    )
    (recall_dir / "calibration-history.jsonl").write_text(
        json.dumps({"ts": "2026-06-10T03:00:00", "reason": "improved"}) + "\n",
        encoding="utf-8",
    )
    (eval_dir / "baseline-20260611.json").write_text(
        json.dumps(
            {
                "metrics": {
                    "examples": 816,
                    "recall_at_1": 0.71,
                    "recall_at_3": 0.83,
                    "waste_injection_rate": 0.33,
                    "latency_ms": {"p50": 95, "p95": 137},
                },
                "policy": {"gate_mode": "evidence"},
            }
        ),
        encoding="utf-8",
    )

    recall = dashboard._recall_snapshot()

    assert recall["samples"] == 2
    assert recall["decisions"] == {"none": 1, "search": 1, "read": 0}
    assert recall["judge_used"] == 1
    assert recall["rewrite_used"] == 1
    assert recall["latency_ms"]["p50"] == 40
    assert recall["pulls"]["total"] == 1
    assert recall["pulls"]["counts"]["read"] == 1
    assert recall["latest_eval"]["recall_at_3"] == 0.83
    assert recall["calibration"]["last_applied"]["reason"] == "improved"
    assert recall["recent"][-1]["decision"] == "none"


def test_recall_snapshot_empty_wiki(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(dashboard, "WIKI_ROOT", tmp_path / "wiki")

    recall = dashboard._recall_snapshot()

    assert recall["samples"] == 0
    assert recall["decisions"] == {"none": 0, "search": 0, "read": 0}
    assert recall["latency_ms"]["p50"] is None
    assert recall["latest_eval"] is None
    assert recall["calibration"]["last_applied"] is None


def test_save_history_snapshot_combines_raw_drain_and_log(tmp_path: Path, monkeypatch) -> None:
    wiki_root = tmp_path / "wiki"
    raw_dir = wiki_root / "raw"
    logs_dir = wiki_root / "logs"
    raw_dir.mkdir(parents=True)
    logs_dir.mkdir(parents=True)
    log_file = wiki_root / "log.md"

    monkeypatch.setattr(dashboard, "WIKI_ROOT", wiki_root)
    monkeypatch.setattr(dashboard, "LOG_FILE", log_file)

    raw_names = [
        "20260701-120000-codex-dashboard-save-history-aaaaaaaa.md",
        "20260701-121000-claude-code-dashboard-save-history-bbbbbbbb.md",
        "vestige-20260702-event-cccccccc.md",
    ]
    for name in raw_names:
        (raw_dir / name).write_text("raw", encoding="utf-8")

    drain_record = {
        "timestamp": "2026-07-01T12:30:00",
        "files_processed": 1,
        "result": {
            "files_attempted": [raw_names[0], "20260701-122000-failed-dddddddd.md"],
            "files_processed": [raw_names[0]],
            "per_raw": [
                {"filename": raw_names[0], "succeeded": True},
                {"filename": "20260701-122000-failed-dddddddd.md", "succeeded": False},
            ],
        },
    }
    (logs_dir / "ingest-drain-20260701.jsonl").write_text(
        json.dumps(drain_record) + "\n",
        encoding="utf-8",
    )
    log_file.write_text(
        "\n".join(
            [
                "# Change Log",
                "- [2026-07-01 12:31] ingest | created save-history-dashboard",
                "- [2026-07-01 12:32] ingest | updated llm-wiki-dashboard",
                "- [2026-07-02 12:32] ingest | updated save-history-dashboard",
            ]
        ),
        encoding="utf-8",
    )

    history = dashboard._save_history_snapshot(days=4, today=date(2026, 7, 4))
    by_date = {row["date"]: row for row in history["days"]}

    assert history["totals"]["raw_saved"] == 3
    assert history["totals"]["raw_bytes"] == 9
    assert history["totals"]["processed_bytes"] == 3
    assert history["totals"]["pending_bytes"] == 6
    assert history["totals"]["failed_bytes"] == 0
    assert history["totals"]["processed"] == 1
    assert history["totals"]["attempted"] == 2
    assert history["totals"]["succeeded"] == 1
    assert history["totals"]["failed"] == 1
    assert history["totals"]["pages_created"] == 1
    assert history["totals"]["pages_updated"] == 2
    assert by_date["2026-07-01"]["raw_saved"] == 2
    assert by_date["2026-07-01"]["raw_bytes"] == 6
    assert by_date["2026-07-01"]["processed_bytes"] == 3
    assert by_date["2026-07-01"]["pending_bytes"] == 3
    assert by_date["2026-07-01"]["processed"] == 1
    assert by_date["2026-07-01"]["failed"] == 1
    assert by_date["2026-07-01"]["raw_segments"] == [
        {
            "name": raw_names[0],
            "bytes": 3,
            "status": "processed",
            "source": "codex",
        },
        {
            "name": raw_names[1],
            "bytes": 3,
            "status": "pending",
            "source": "claude-code",
        },
    ]
    assert by_date["2026-07-02"]["raw_saved"] == 1
    assert by_date["2026-07-02"]["pending_bytes"] == 3
    assert by_date["2026-07-02"]["raw_segments"][0]["status"] == "pending"
    assert by_date["2026-07-02"]["pages_updated"] == 1
    assert by_date["2026-07-01"]["sources"] == [
        {"name": "claude-code", "count": 1},
        {"name": "codex", "count": 1},
    ]
    assert history["sources"] == [
        {"name": "claude-code", "count": 1},
        {"name": "codex", "count": 1},
        {"name": "vestige", "count": 1},
    ]
    assert history["recent"][-1]["date"] == "2026-07-02"


def test_save_history_snapshot_empty_wiki(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(dashboard, "WIKI_ROOT", tmp_path / "wiki")
    monkeypatch.setattr(dashboard, "LOG_FILE", tmp_path / "wiki" / "log.md")

    history = dashboard._save_history_snapshot(days=2, today=date(2026, 7, 4))

    assert [row["date"] for row in history["days"]] == ["2026-07-03", "2026-07-04"]
    assert history["recent"] == []
    assert history["totals"]["raw_saved"] == 0
    assert history["totals"]["raw_bytes"] == 0
    assert history["totals"]["pending_bytes"] == 0
    assert history["days"][0]["raw_segments"] == []


def test_knowledge_mix_snapshot_groups_pages_by_category(tmp_path: Path, monkeypatch) -> None:
    wiki_root = tmp_path / "wiki"
    pages_dir = wiki_root / "pages"
    (pages_dir / "ai").mkdir(parents=True)
    (pages_dir / "macos").mkdir()
    (pages_dir / "ai" / "agent-memory.md").write_text("a" * 20, encoding="utf-8")
    (pages_dir / "ai" / "evals.md").write_text("b" * 10, encoding="utf-8")
    (pages_dir / "macos" / "display.md").write_text("c" * 15, encoding="utf-8")

    monkeypatch.setattr(dashboard, "WIKI_ROOT", wiki_root)

    mix = dashboard._knowledge_mix_snapshot()

    assert mix["total_pages"] == 3
    assert mix["total_bytes"] == 45
    assert [row["id"] for row in mix["categories"]] == ["ai", "macos"]
    assert mix["categories"][0]["label"] == "AI"
    assert mix["categories"][0]["pages"] == 2
    assert mix["categories"][0]["bytes"] == 30
    assert round(mix["categories"][0]["share"], 3) == 0.667
    assert "ai/agent-memory.md" in mix["categories"][0]["samples"]
