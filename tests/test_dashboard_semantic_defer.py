from __future__ import annotations

import hashlib
import json
import os
from datetime import date, datetime
from pathlib import Path
from types import SimpleNamespace

import pytest


def test_save_load_segments_semantic_defer_returns_to_pending_after_release(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from chronovisor.ops import dashboard

    chronovisor_root = tmp_path / "wiki"
    raw_dir = chronovisor_root / "raw"
    logs_dir = chronovisor_root / "logs"
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
    monkeypatch.setattr(dashboard, "CHRONOVISOR_ROOT", chronovisor_root)
    monkeypatch.setattr(dashboard, "LOG_FILE", chronovisor_root / "log.md")
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


def test_save_load_shard_continuation_is_pending_not_failed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from chronovisor.ops import dashboard

    chronovisor_root = tmp_path / "wiki"
    raw_dir = chronovisor_root / "raw"
    logs_dir = chronovisor_root / "logs"
    raw_dir.mkdir(parents=True)
    logs_dir.mkdir()
    name = "20260714-101000-codex-continued-eeeeeeee.md"
    (raw_dir / name).write_text("raw", encoding="utf-8")
    (logs_dir / "ingest-drain-20260714.jsonl").write_text(
        json.dumps(
            {
                "timestamp": "2026-07-14T10:10:00",
                "result": {
                    "files_attempted": [name],
                    "files_processed": [],
                    "files_deferred": [],
                    "files_continued": [name],
                    "files_failed": 0,
                    "per_raw": [
                        {
                            "filename": name,
                            "succeeded": False,
                            "deferred": False,
                            "continued": True,
                        }
                    ],
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(dashboard, "CHRONOVISOR_ROOT", chronovisor_root)
    monkeypatch.setattr(dashboard, "LOG_FILE", chronovisor_root / "log.md")
    monkeypatch.setattr(
        dashboard,
        "_operational_deferred_raw_statuses",
        lambda _paths: {},
    )

    history = dashboard._save_history_snapshot(days=1, today=date(2026, 7, 14))

    assert history["totals"]["continued"] == 1
    assert history["totals"]["failed"] == 0
    assert history["totals"]["pending_bytes"] == 3
    assert history["totals"]["failed_bytes"] == 0
    assert history["days"][0]["raw_segments"] == [
        {
            "name": name,
            "bytes": 3,
            "status": "pending",
            "source": "codex",
        }
    ]
    assert dashboard._drain_history(limit=10)[0]["files_continued"] == 1


def test_metric_history_deduplicates_runtime_and_drain_records_for_same_batch() -> None:
    from chronovisor.ops import dashboard

    result = {
        "pending_before": 113,
        "pending_after": 103,
        "files_attempted": 10,
        "files_processed": 0,
        "files_deferred": 0,
        "files_continued": 0,
        "files_failed": 10,
        "elapsed_seconds": 94.57,
    }
    runtime = {
        **result,
        "timestamp": "2026-07-19T22:26:15",
        "kind": "batch",
        "processor": "ollama",
    }
    drain = {
        **result,
        "timestamp": "2026-07-19T22:26:26",
        "kind": "drain_batch",
        "batch": 5,
    }

    merged = dashboard._merge_metric_history([runtime], [drain], limit=240)

    assert merged == [runtime]


def test_save_load_attributes_held_projection_child_to_saved_parent(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from chronovisor.ingest import raw_semantic_projection
    from chronovisor.ops import dashboard

    chronovisor_root = tmp_path / "wiki"
    raw_dir = chronovisor_root / "raw"
    raw_dir.mkdir(parents=True)
    session_key = "c" * 24
    idempotency_key = f"claude-code-{session_key}-from1-to2"
    parent_name = f"save-{idempotency_key}.md"
    parent_bytes = b"lossless saved transcript"
    projection_id = "a" * 64
    child_name = f"semantic-{projection_id}-child-00000001-{'b' * 64}.md"
    processed_child_name = f"semantic-{projection_id}-child-00000002-{'d' * 64}.md"
    parent_path = raw_dir / parent_name
    parent_path.write_bytes(parent_bytes)
    saved_at = datetime(2026, 7, 14, 12, 0, 0).timestamp()
    os.utime(parent_path, (saved_at, saved_at))
    (raw_dir / child_name).write_text("derived semantic child", encoding="utf-8")
    (raw_dir / processed_child_name).write_text(
        "already processed semantic child", encoding="utf-8"
    )
    (raw_dir / f"semantic-{projection_id}.manifest.json").write_text(
        "placeholder", encoding="utf-8"
    )
    (chronovisor_root / ".orchestrator_state.json").write_text(
        json.dumps({"processed_raw_files": [parent_name, processed_child_name]}),
        encoding="utf-8",
    )
    manifest = {
        "children": [
            {"filename": child_name},
            {"filename": processed_child_name},
        ],
        "source": {
            "parents": [
                {
                    "raw_sha256": hashlib.sha256(parent_bytes).hexdigest(),
                    "receipt": {
                        "host": "claude-code",
                        "session_key": session_key,
                        "after_line": 1,
                        "until_line": 2,
                        "idempotency_key": idempotency_key,
                    },
                }
            ]
        },
    }
    active_deferred = {child_name: "semantic_no_quorum"}
    monkeypatch.setattr(dashboard, "CHRONOVISOR_ROOT", chronovisor_root)
    monkeypatch.setattr(dashboard, "LOG_FILE", chronovisor_root / "log.md")
    monkeypatch.setattr(
        dashboard,
        "_operational_deferred_raw_statuses",
        lambda _paths: dict(active_deferred),
    )
    monkeypatch.setattr(
        raw_semantic_projection,
        "verify_projection_bundle",
        lambda _path: manifest,
    )

    history = dashboard._save_history_snapshot(days=1, today=date(2026, 7, 14))

    assert history["totals"]["raw_bytes"] == len(parent_bytes)
    assert history["totals"]["processed_bytes"] == 0
    assert history["totals"]["pending_bytes"] == 0
    assert history["totals"]["deferred_bytes"] == len(parent_bytes)
    assert history["totals"]["failed_bytes"] == 0
    assert history["days"][0]["raw_segments"] == [
        {
            "name": parent_name,
            "bytes": len(parent_bytes),
            "status": "deferred",
            "source": "claude-code",
        }
    ]

    active_deferred.clear()
    released = dashboard._save_history_snapshot(days=1, today=date(2026, 7, 14))

    assert released["totals"]["processed_bytes"] == 0
    assert released["totals"]["pending_bytes"] == len(parent_bytes)
    assert released["totals"]["deferred_bytes"] == 0
    assert released["days"][0]["raw_segments"][0]["status"] == "pending"

    (chronovisor_root / ".orchestrator_state.json").write_text(
        json.dumps(
            {
                "processed_raw_files": [
                    parent_name,
                    child_name,
                    processed_child_name,
                ]
            }
        ),
        encoding="utf-8",
    )
    processed = dashboard._save_history_snapshot(days=1, today=date(2026, 7, 14))

    assert processed["totals"]["processed_bytes"] == len(parent_bytes)
    assert processed["totals"]["pending_bytes"] == 0
    assert processed["totals"]["deferred_bytes"] == 0
    assert processed["days"][0]["raw_segments"][0]["status"] == "processed"


def test_projection_parent_resolution_rejects_unbound_receipt_and_symlink(
    tmp_path: Path,
) -> None:
    from chronovisor.ops import dashboard

    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    session_key = "a" * 24
    parent_bytes = b"source"
    source_parent = {
        "raw_sha256": hashlib.sha256(parent_bytes).hexdigest(),
        "receipt": {
            "host": "codex",
            "session_key": session_key,
            "after_line": 3,
            "until_line": 4,
            "idempotency_key": f"codex-{session_key}-from3-to4",
        },
    }
    outside = tmp_path / "outside.md"
    outside.write_bytes(parent_bytes)
    parent_path = raw_dir / f"save-codex-{session_key}-from3-to4.md"
    parent_path.symlink_to(outside)

    assert dashboard._projection_parent_name(raw_dir, source_parent) is None

    parent_path.unlink()
    parent_path.write_bytes(parent_bytes)
    source_parent["receipt"]["idempotency_key"] = "../outside"
    assert dashboard._projection_parent_name(raw_dir, source_parent) is None


def test_projection_parent_resolution_reuses_one_raw_store_snapshot(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from chronovisor.core import raw_store
    from chronovisor.ingest import raw_semantic_projection
    from chronovisor.ops import dashboard

    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    projection_id = "a" * 64
    child_name = f"semantic-{projection_id}-child-00000001-{'b' * 64}.md"
    source_parents = []
    expected_names = set()
    for index in range(2):
        session_key = f"{index + 1:024x}"
        idempotency_key = f"codex-{session_key}-from{index}-to{index + 1}"
        parent_name = f"save-{idempotency_key}.md"
        parent_bytes = f"source-{index}".encode()
        (raw_dir / parent_name).write_bytes(parent_bytes)
        expected_names.add(parent_name)
        source_parents.append(
            {
                "raw_sha256": hashlib.sha256(parent_bytes).hexdigest(),
                "receipt": {
                    "host": "codex",
                    "session_key": session_key,
                    "after_line": index,
                    "until_line": index + 1,
                    "idempotency_key": idempotency_key,
                },
            }
        )

    monkeypatch.setattr(
        raw_semantic_projection,
        "verify_projection_bundle",
        lambda _path: {
            "children": [{"filename": child_name}],
            "source": {"parents": source_parents},
        },
    )
    original = raw_store.RawStore
    constructions = 0

    class CountingRawStore(original):
        def __init__(self, *args, **kwargs):
            nonlocal constructions
            constructions += 1
            super().__init__(*args, **kwargs)

    monkeypatch.setattr(raw_store, "RawStore", CountingRawStore)

    parents = dashboard._projection_parent_raw_names_by_child(
        raw_dir,
        {child_name},
    )

    assert parents == {child_name: expected_names}
    assert constructions == 1


def test_projection_parent_resolution_verifies_archive_once_without_member_reads(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from chronovisor.core import legacy_archive
    from chronovisor.ops import dashboard

    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    session_key = "a" * 24
    expected_sha256 = "b" * 64
    parent_name = f"save-codex-{session_key}-from3-to4.md"
    source_parent = {
        "raw_sha256": expected_sha256,
        "receipt": {
            "host": "codex",
            "session_key": session_key,
            "after_line": 3,
            "until_line": 4,
            "idempotency_key": f"codex-{session_key}-from3-to4",
        },
    }
    archive_path = raw_dir / "legacy-part-0001.tar.zst"
    manifest_path = raw_dir / "legacy-part-0001.manifest.json"
    unit = SimpleNamespace(
        storage="legacy_archive",
        sha256=expected_sha256,
        archive_member=SimpleNamespace(
            archive_path=archive_path,
            manifest_path=manifest_path,
        ),
    )
    verified_calls: list[Path] = []
    monkeypatch.setattr(
        legacy_archive,
        "verify_legacy_manifest",
        lambda path, *, full: verified_calls.append(path),
    )

    class IndexedRawStore:
        def resolve(self, raw_id: str) -> object | None:
            return unit if raw_id == parent_name else None

        def read_bytes(self, _unit: object) -> bytes:
            raise AssertionError("indexed immutable Raw must not be reopened")

    verified_archives: set[Path] = set()
    for _index in range(2):
        assert (
            dashboard._projection_parent_name(
                raw_dir,
                source_parent,
                raw_store=IndexedRawStore(),
                verified_archives=verified_archives,
            )
            == parent_name
        )
    assert verified_calls == [manifest_path]

    unit.sha256 = "c" * 64
    assert (
        dashboard._projection_parent_name(
            raw_dir,
            source_parent,
            raw_store=IndexedRawStore(),
            verified_archives=verified_archives,
        )
        is None
    )


def test_snapshot_separates_semantic_and_operational_holds_once(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from chronovisor.core import runtime_config, runtime_status
    from chronovisor.decision import decision_policy
    from chronovisor.ingest import orchestrator
    from chronovisor.ops import dashboard

    chronovisor_root = tmp_path / "wiki"
    raw_dir = chronovisor_root / "raw"
    raw_dir.mkdir(parents=True)
    for name in ("semantic.md", "operational.md"):
        (raw_dir / name).write_text(name, encoding="utf-8")
    artifact_dir = chronovisor_root / "runtime" / "raw-projections" / "artifacts"
    artifact_dir.mkdir(parents=True)
    (artifact_dir / "semantic.md").write_text(
        "same logical semantic child",
        encoding="utf-8",
    )

    scans: list[list[str]] = []

    def deferred_statuses(paths: list[Path]) -> dict[str, str]:
        scans.append([path.name for path in paths])
        return {
            "semantic.md": "semantic_no_quorum",
            "operational.md": "self_heal_pending",
        }

    monkeypatch.setattr(dashboard, "CHRONOVISOR_ROOT", chronovisor_root)
    monkeypatch.setattr(dashboard, "init_chronovisor", lambda: None)
    monkeypatch.setattr(
        dashboard, "_operational_deferred_raw_statuses", deferred_statuses
    )
    monkeypatch.setattr(orchestrator, "_load_state", lambda: {})
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
    assert snapshot["status"]["pending"] == 0
    assert snapshot["status"]["semantic_deferred"] == {
        "count": 1,
        "samples": ["semantic.md"],
    }
    assert snapshot["status"]["operational_deferred"] == {
        "count": 1,
        "samples": ["operational.md"],
    }
    assert snapshot["status"]["raw_outstanding"] == 2
    assert snapshot["self_heal"] is self_heal


def test_semantic_defer_packets_are_absent_from_self_heal_history_and_watch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from chronovisor.ops import dashboard

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
    monkeypatch.setattr(dashboard, "CHRONOVISOR_ROOT", tmp_path)
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
    from chronovisor.core import jobs, runtime_config, runtime_status
    from chronovisor.ingest import (
        failure_supervisor,
        ingest,
        orchestrator,
        raw_semantic_projection,
    )

    chronovisor_root = tmp_path / "wiki"
    raw_dir = chronovisor_root / "raw"
    raw_dir.mkdir(parents=True)
    raw = raw_dir / "semantic.md"
    raw.write_text("semantic source", encoding="utf-8")
    deferred = False
    statuses: list[dict] = []
    metrics: list[dict] = []
    events: list[dict] = []

    monkeypatch.setattr(orchestrator, "CHRONOVISOR_ROOT", chronovisor_root)
    monkeypatch.setattr(orchestrator, "RAW_DIR", raw_dir)
    monkeypatch.setattr(orchestrator, "LOG_FILE", chronovisor_root / "log.md")
    monkeypatch.setattr(
        orchestrator, "STATE_FILE", chronovisor_root / ".orchestrator_state.json"
    )
    monkeypatch.setattr(
        orchestrator,
        "get_ollama_status",
        lambda: {"available": True, "processor": "ollama"},
    )
    monkeypatch.setattr(
        orchestrator,
        "ingest_authority_preflight",
        lambda **_kwargs: {
            "ok": True,
            "status": "ready",
            "blocked_by": None,
            "retryable": False,
            "error": None,
            "artifact_sha256": "a" * 64,
        },
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
    root = Path(__file__).parents[1] / "src" / "chronovisor" / "dashboard_static"
    html = (root / "index.html").read_text(encoding="utf-8")
    store_js = (root / "app.js").read_text(encoding="utf-8")
    renderer_js = (root / "app-renderer.js").read_text(encoding="utf-8")
    client_js = (root / "app-client.js").read_text(encoding="utf-8")

    assert 'id="pending-deferred"' in html
    assert 'id="batch-deferred"' in html
    assert 'id="batch-continued"' in html
    assert "semantic deferred" in html
    assert 'document.getElementById("pending-deferred")' in store_js
    assert 'document.getElementById("batch-deferred")' in store_js
    assert 'document.getElementById("batch-continued")' in store_js
    assert 'dashed: segment.status === "pending"' in renderer_js
    assert 'dashed: segment.status === "deferred"' not in renderer_js
    assert "row.files_deferred" in renderer_js
    assert "row.files_continued" in renderer_js
    assert "function batchCountLabel(row)" in renderer_js
    assert "parts.push(`${row.deferred} defer`)" in renderer_js
    assert "parts.push(`${row.continued} continue`)" in renderer_js
    assert "function fitCanvasText(ctx, text, maxWidth)" in renderer_js
    assert "function drawBatchLegend(ctx, width, left, y)" in renderer_js
    assert "ctx.measureText(batchCountLabel(row)).width" in renderer_js
    assert "Math.ceil(ctx.measureText(label).width) + textGap" in renderer_js
    assert "Math.max(40, pad.right - 16)" in renderer_js
    assert "let refreshInFlight = null" in client_js
    assert "if (refreshInFlight !== null) return refreshInFlight" in client_js
    assert "window.setTimeout(refreshLoop, nextRefreshDelayMs)" in client_js
    assert "const ACTIVE_REFRESH_DELAY_MS = 5000" in client_js
    assert "const IDLE_REFRESH_DELAY_MS = 10000" in client_js
    assert "setInterval(refresh" not in client_js
    assert "const SNAPSHOT_TIMEOUT_MS = 180000" in client_js
    assert "const controller = new AbortController()" in client_js
    assert "signal: controller.signal" in client_js
    assert "controller.abort()" in client_js
    assert "window.clearTimeout(timeoutId)" in client_js
    assert (
        "finally {\n    window.setTimeout(refreshLoop, nextRefreshDelayMs)" in client_js
    )
