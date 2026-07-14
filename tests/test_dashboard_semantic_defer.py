from __future__ import annotations

import json
from datetime import date
from pathlib import Path
from types import SimpleNamespace

import pytest


def test_save_load_segments_semantic_defer_returns_to_pending_after_release(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from llm_wiki_mcp import dashboard

    wiki_root = tmp_path / "wiki"
    raw_dir = wiki_root / "raw"
    logs_dir = wiki_root / "logs"
    raw_dir.mkdir(parents=True)
    logs_dir.mkdir()
    names = {
        "processed": "20260714-100000-codex-processed-aaaaaaaa.md",
        "pending": "20260714-100100-codex-pending-bbbbbbbb.md",
        "deferred": "20260714-100200-codex-deferred-cccccccc.md",
        "failed": "20260714-100300-codex-failed-dddddddd.md",
    }
    for name in names.values():
        (raw_dir / name).write_text("raw", encoding="utf-8")
    (logs_dir / "ingest-drain-20260714.jsonl").write_text(
        json.dumps(
            {
                "timestamp": "2026-07-14T10:10:00",
                "result": {
                    "files_attempted": [
                        names["processed"],
                        names["deferred"],
                        names["failed"],
                    ],
                    "files_processed": [names["processed"]],
                    "files_deferred": [names["deferred"]],
                    "per_raw": [
                        {"filename": names["processed"], "succeeded": True},
                        {
                            "filename": names["deferred"],
                            "succeeded": False,
                            "deferred": True,
                            "supervision": {"terminal_deferred": True},
                        },
                        {"filename": names["failed"], "succeeded": False},
                    ],
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(dashboard, "WIKI_ROOT", wiki_root)
    monkeypatch.setattr(dashboard, "LOG_FILE", wiki_root / "log.md")
    active_deferred = {names["deferred"]: "semantic_no_quorum"}
    monkeypatch.setattr(
        dashboard,
        "_operational_deferred_raw_statuses",
        lambda _paths: dict(active_deferred),
    )

    history = dashboard._save_history_snapshot(days=1, today=date(2026, 7, 14))

    totals = history["totals"]
    assert totals["raw_bytes"] == 12
    assert totals["processed_bytes"] == 3
    assert totals["pending_bytes"] == 3
    assert totals["deferred_bytes"] == 3
    assert totals["failed_bytes"] == 3
    assert totals["raw_bytes"] == sum(
        totals[key]
        for key in (
            "processed_bytes",
            "pending_bytes",
            "deferred_bytes",
            "failed_bytes",
        )
    )
    assert totals["deferred"] == 1
    assert totals["failed"] == 1
    assert {
        segment["name"]: segment["status"]
        for segment in history["days"][0]["raw_segments"]
    } == {name: status for status, name in names.items()}

    active_deferred.clear()
    released = dashboard._save_history_snapshot(days=1, today=date(2026, 7, 14))

    released_totals = released["totals"]
    assert released_totals["processed_bytes"] == 3
    assert released_totals["pending_bytes"] == 6
    assert released_totals["deferred_bytes"] == 0
    assert released_totals["failed_bytes"] == 3
    assert released_totals["deferred"] == 1
    assert released_totals["failed"] == 1
    assert {
        segment["name"]: segment["status"]
        for segment in released["days"][0]["raw_segments"]
    } == {
        names["processed"]: "processed",
        names["pending"]: "pending",
        names["deferred"]: "pending",
        names["failed"]: "failed",
    }


def test_snapshot_separates_semantic_and_operational_holds_once(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from llm_wiki_mcp import dashboard, orchestrator, runtime_status
    from llm_wiki_mcp import decision_policy, runtime_config

    wiki_root = tmp_path / "wiki"
    raw_dir = wiki_root / "raw"
    raw_dir.mkdir(parents=True)
    for name in ("semantic.md", "operational.md"):
        (raw_dir / name).write_text(name, encoding="utf-8")

    scans: list[list[str]] = []

    def deferred_statuses(paths: list[Path]) -> dict[str, str]:
        scans.append([path.name for path in paths])
        return {
            "semantic.md": "semantic_no_quorum",
            "operational.md": "self_heal_pending",
        }

    monkeypatch.setattr(dashboard, "WIKI_ROOT", wiki_root)
    monkeypatch.setattr(dashboard, "init_wiki", lambda: None)
    monkeypatch.setattr(
        dashboard, "_operational_deferred_raw_statuses", deferred_statuses
    )
    monkeypatch.setattr(orchestrator, "_load_state", lambda: {})
    monkeypatch.setattr(
        orchestrator,
        "get_pending_raw_files",
        lambda: [raw_dir / "pending-a.md", raw_dir / "pending-b.md"],
    )
    monkeypatch.setattr(runtime_status, "read_status", lambda: {})
    monkeypatch.setattr(runtime_status, "read_metrics", lambda limit: [])
    monkeypatch.setattr(runtime_status, "read_events", lambda limit: [])
    monkeypatch.setattr(dashboard, "_drain_history", lambda limit: [])
    monkeypatch.setattr(dashboard, "_recent_log_events", lambda limit: [])
    monkeypatch.setattr(dashboard, "_ollama_snapshot", lambda: {})
    monkeypatch.setattr(dashboard, "_model_status_snapshot", lambda _ollama: {})
    monkeypatch.setattr(
        dashboard,
        "_local_consensus_snapshot",
        lambda: {"active": False, "activities": [], "summary": {}, "history": []},
    )
    monkeypatch.setattr(
        dashboard,
        "_frontier_activity_snapshot",
        lambda: {"active": False, "reviews": [], "latest": None},
    )
    monkeypatch.setattr(
        dashboard,
        "_frontier_repair_snapshot",
        lambda: {"active": False, "summary": {}, "recent": [], "events": []},
    )
    self_heal = {"status": "quiet", "history": [], "watch": {}}
    for name, value in (
        ("_self_heal_snapshot", self_heal),
        ("_recall_snapshot", {}),
        ("_recall_improvement_snapshot", {}),
        ("_model_lab_snapshot", {}),
        ("_save_history_snapshot", {}),
        ("_knowledge_mix_snapshot", {}),
        ("health_snapshot", {}),
    ):
        monkeypatch.setattr(dashboard, name, lambda value=value: value)
    monkeypatch.setattr(runtime_config, "runtime_identity", lambda: {})
    monkeypatch.setattr(decision_policy, "decision_policy_snapshot", lambda: {})

    snapshot = dashboard.build_snapshot()

    assert scans == [["operational.md", "semantic.md"]]
    assert snapshot["status"]["pending"] == 2
    assert snapshot["status"]["semantic_deferred"] == {
        "count": 1,
        "samples": ["semantic.md"],
    }
    assert snapshot["status"]["operational_deferred"] == {
        "count": 1,
        "samples": ["operational.md"],
    }
    assert snapshot["status"]["raw_outstanding"] == 4
    assert snapshot["self_heal"] is self_heal


def test_semantic_defer_packets_are_absent_from_self_heal_history_and_watch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from llm_wiki_mcp import dashboard

    failures = tmp_path / "runtime" / "failures"
    packets = failures / "packets"
    packets.mkdir(parents=True)
    semantic = {
        "failure_id": "semantic-1",
        "failure_class": "ingest.semantic_no_quorum",
        "terminal_deferred": True,
        "status": "local_quarantined",
        "raw_file": "semantic.md",
    }
    operational = {
        "failure_id": "operational-1",
        "failure_class": "ingest.runtime_transport_error",
        "status": "pending_local_repair",
        "raw_file": "operational.md",
    }
    superseded_operational = {
        "failure_id": "operational-superseded",
        "failure_class": "ingest.runtime_local_consensus_authority_unavailable",
        "status": "superseded_semantic_defer",
        "raw_file": "semantic.md",
    }
    released_semantic = {
        **semantic,
        "failure_id": "semantic-released",
        "terminal_deferred": False,
        "status": "released",
    }
    superseded_semantic = {
        **semantic,
        "failure_id": "semantic-superseded",
        "terminal_deferred": False,
        "status": "superseded",
    }
    (packets / "semantic.json").write_text(json.dumps(semantic), encoding="utf-8")
    (packets / "semantic-released.json").write_text(
        json.dumps(released_semantic), encoding="utf-8"
    )
    (packets / "semantic-superseded.json").write_text(
        json.dumps(superseded_semantic), encoding="utf-8"
    )
    (packets / "operational.json").write_text(json.dumps(operational), encoding="utf-8")
    (packets / "operational-superseded.json").write_text(
        json.dumps(superseded_operational), encoding="utf-8"
    )
    (failures / "failure-registry.jsonl").write_text(
        "".join(
            json.dumps({**row, "resolution": "local"}) + "\n"
            for row in (semantic, released_semantic, superseded_semantic)
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(dashboard, "WIKI_ROOT", tmp_path)
    monkeypatch.setattr(
        dashboard,
        "_frontier_preflight_snapshot",
        lambda: {"state": "standby"},
    )

    snapshot = dashboard._self_heal_snapshot()

    assert [row["failure_id"] for row in snapshot["history"]] == ["operational-1"]
    assert snapshot["watch"]["packets"]["total"] == 1
    assert snapshot["watch"]["packets"]["pending"] == 1
    assert "local_quarantined" not in snapshot["watch"]["packets"]["status_counts"]


def test_orchestrator_reports_terminal_semantic_defer_without_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from llm_wiki_mcp import (
        failure_supervisor,
        ingest,
        jobs,
        orchestrator,
        raw_semantic_projection,
        runtime_config,
        runtime_status,
    )

    wiki_root = tmp_path / "wiki"
    raw_dir = wiki_root / "raw"
    raw_dir.mkdir(parents=True)
    raw = raw_dir / "semantic.md"
    raw.write_text("semantic source", encoding="utf-8")
    deferred = False
    statuses: list[dict] = []
    metrics: list[dict] = []
    events: list[dict] = []

    monkeypatch.setattr(orchestrator, "WIKI_ROOT", wiki_root)
    monkeypatch.setattr(orchestrator, "RAW_DIR", raw_dir)
    monkeypatch.setattr(orchestrator, "LOG_FILE", wiki_root / "log.md")
    monkeypatch.setattr(
        orchestrator, "STATE_FILE", wiki_root / ".orchestrator_state.json"
    )
    monkeypatch.setattr(orchestrator, "is_available", lambda: True)
    monkeypatch.setattr(
        orchestrator,
        "get_ollama_status",
        lambda: {"available": True, "processor": "ollama"},
    )
    monkeypatch.setattr(
        orchestrator,
        "get_pending_raw_files",
        lambda: [] if deferred else [raw],
    )
    monkeypatch.setattr(
        runtime_status,
        "safe_write_status",
        lambda **values: statuses.append(values),
    )
    monkeypatch.setattr(
        runtime_status,
        "safe_append_metric",
        lambda kind, **values: metrics.append({"kind": kind, **values}),
    )
    monkeypatch.setattr(
        runtime_status,
        "safe_append_event",
        lambda level, message, **values: events.append(
            {"level": level, "message": message, **values}
        ),
    )
    isolated_jobs = jobs.JobStore()
    monkeypatch.setattr(jobs, "job_store", isolated_jobs)

    def fail_ingest(_raw_text: str, job_id: str, **_kwargs: object) -> None:
        isolated_jobs.update(
            job_id,
            status=jobs.JobStatus.FAILED,
            error="local consensus semantic no quorum",
        )

    monkeypatch.setattr(ingest, "run_ingest", fail_ingest)
    monkeypatch.setattr(
        runtime_config,
        "load_ingest_config",
        lambda: SimpleNamespace(semantic_projection_max_child_bytes=1_000_000),
    )
    monkeypatch.setattr(
        raw_semantic_projection,
        "project_parent_raw",
        lambda *_args, **_kwargs: raw_semantic_projection.ProjectionArtifacts(
            kind="passthrough",
            parent_paths=(raw,),
            parent_sha256="a" * 64,
            projection_sha256=None,
            manifest_path=None,
            projection_paths=(),
            child_paths=(),
            noop_receipt_path=None,
            role_counts={},
            record_count=0,
            selected_record_count=0,
            child_count=0,
            children=(),
        ),
    )

    def record_defer(**_kwargs: object) -> failure_supervisor.SupervisionResult:
        nonlocal deferred
        deferred = True
        return failure_supervisor.SupervisionResult(
            raw_file=raw.name,
            failure_class="ingest.semantic_no_quorum",
            fingerprint="semantic",
            attempts=1,
            terminal_deferred=True,
        )

    monkeypatch.setattr(failure_supervisor, "record_raw_failure", record_defer)

    result = orchestrator.run_pending_ingest(force=True)

    assert result["files_deferred"] == [raw.name]
    assert result["files_failed"] == 0
    assert result["per_raw"][0]["deferred"] is True
    batch_metric = next(row for row in metrics if row["kind"] == "batch")
    assert batch_metric["files_deferred"] == 1
    assert batch_metric["files_failed"] == 0
    assert statuses[-1]["batch"]["deferred"] == 1
    assert statuses[-1]["batch"]["failed"] == 0
    raw_event = next(row for row in events if row.get("raw_file") == raw.name)
    assert raw_event["message"].endswith("semantic deferred")
    assert raw_event["level"] == "info"


def test_dashboard_static_contract_exposes_deferred_without_pending_dashes() -> None:
    root = Path(__file__).parents[1] / "src" / "llm_wiki_mcp" / "dashboard_static"
    html = (root / "index.html").read_text(encoding="utf-8")
    js = (root / "app.js").read_text(encoding="utf-8")

    assert 'id="pending-deferred"' in html
    assert 'id="batch-deferred"' in html
    assert "semantic deferred" in html
    assert 'document.getElementById("pending-deferred")' in js
    assert 'document.getElementById("batch-deferred")' in js
    assert 'dashed: segment.status === "pending"' in js
    assert 'dashed: segment.status === "deferred"' not in js
    assert "row.files_deferred" in js
    assert "countParts.push(`${row.deferred} defer`)" in js
