from __future__ import annotations

import json
import os
import plistlib
import subprocess
from datetime import UTC, datetime
from pathlib import Path

import pytest

from chronovisor.core import capture_status
from chronovisor.core.runtime_config import RuntimeConfigError
from chronovisor.ops import capture

ROOT = Path(__file__).resolve().parents[1]
DEADMAN = ROOT / "scripts" / "chronovisor-deadman"
CAPTURE_PLIST = ROOT / "launchd" / "com.trafficsign.chronovisor-capture.plist"


def _now() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


@pytest.fixture
def status_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    path = tmp_path / "runtime" / "capture-status.json"
    monkeypatch.setattr(capture, "STATUS_FILE", path)
    return path


def _install_fakes(
    monkeypatch: pytest.MonkeyPatch,
    *,
    job_results: list[dict[str, object]],
    sweep_results: list[dict[str, object]],
    by_status: dict[str, int] | None = None,
    oldest_pending_at: str | None = None,
    calls: list[tuple[str, int]] | None = None,
) -> None:
    def retry_due(*, limit: int) -> dict[str, object]:
        if calls is not None:
            calls.append(("retry", limit))
        return {"results": job_results}

    def run_sweeper(*, limit: int) -> dict[str, object]:
        if calls is not None:
            calls.append(("sweep", limit))
        return {"results": sweep_results}

    monkeypatch.setattr(capture, "retry_due", retry_due)
    monkeypatch.setattr(capture, "run_sweeper", run_sweeper)
    monkeypatch.setattr(
        capture,
        "snapshot",
        lambda: {
            "status": "ok",
            "by_status": by_status or {},
            "oldest_pending_at": oldest_pending_at,
        },
    )


def test_capture_retries_then_sweeps_in_one_process(
    status_file: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[tuple[str, int]] = []
    _install_fakes(
        monkeypatch,
        job_results=[
            {"status": "completed", "output_tail": '{"status": "saved"}\n'},
            {"status": "retry_wait"},
        ],
        sweep_results=[{"status": "skipped"}],
        by_status={"queued": 2, "retry_wait": 1},
        oldest_pending_at="2026-11-08T12:00:00+00:00",
        calls=calls,
    )

    result = capture.run_capture()

    assert calls == [("retry", 8), ("sweep", 8)]
    assert result["ok"] is True
    payload = json.loads(status_file.read_text(encoding="utf-8"))
    assert payload["version"] == 1
    assert payload["pending"] == 3
    assert payload["failed"] == 0
    assert payload["last_error"] == "transient_failure"
    assert payload["oldest_pending_at"] == "2026-11-08T12:00:00Z"
    assert payload["last_attempt_at"].endswith("Z")
    assert payload["last_success_at"].endswith("Z")
    assert payload["last_commit_at"].endswith("Z")


def test_capture_preserves_last_commit_when_no_raw_is_written(
    status_file: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    status_file.parent.mkdir(parents=True)
    status_file.write_text(
        json.dumps(
            {
                "version": 1,
                "last_attempt_at": "2026-11-01T00:00:00Z",
                "last_success_at": "2026-11-01T00:00:00Z",
                "last_commit_at": "2026-11-01T00:00:00+00:00",
                "oldest_pending_at": None,
                "pending": 0,
                "failed": 0,
                "last_error": "none",
            }
        ),
        encoding="utf-8",
    )
    _install_fakes(
        monkeypatch,
        job_results=[],
        sweep_results=[{"status": "skipped"}],
    )

    capture.run_capture()

    payload = json.loads(status_file.read_text(encoding="utf-8"))
    assert payload["last_commit_at"] == "2026-11-01T00:00:00Z"
    assert payload["last_error"] == "none"
    assert payload["last_success_at"] != "2026-11-01T00:00:00Z"


def test_capture_reports_permanent_job_failure(
    status_file: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _install_fakes(
        monkeypatch,
        job_results=[{"status": "failed"}],
        sweep_results=[],
        by_status={"failed": 1, "quarantined": 1},
    )

    result = capture.run_capture()

    payload = json.loads(status_file.read_text(encoding="utf-8"))
    assert result["ok"] is True
    assert payload["failed"] == 2
    assert payload["last_error"] == "permanent_failure"


def test_capture_reports_config_error_without_losing_last_success(
    status_file: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    status_file.parent.mkdir(parents=True)
    status_file.write_text(
        json.dumps(
            {
                "version": 1,
                "last_attempt_at": "2026-11-01T00:00:00Z",
                "last_success_at": "2026-11-01T00:00:00Z",
                "last_commit_at": "2026-11-01T00:00:00Z",
                "oldest_pending_at": None,
                "pending": 0,
                "failed": 0,
                "last_error": "none",
            }
        ),
        encoding="utf-8",
    )

    def fail(*, limit: int) -> dict[str, object]:
        raise RuntimeConfigError()

    monkeypatch.setattr(capture, "retry_due", fail)
    monkeypatch.setattr(capture, "run_sweeper", lambda *, limit: {"results": []})
    monkeypatch.setattr(capture, "snapshot", lambda: {"status": "ok", "by_status": {}})

    result = capture.run_capture()

    payload = json.loads(status_file.read_text(encoding="utf-8"))
    assert result["ok"] is False
    assert payload["last_error"] == "config_invalid"
    assert payload["last_success_at"] == "2026-11-01T00:00:00Z"


def test_capture_status_snapshot_sanitizes_and_rejects_links(tmp_path: Path) -> None:
    path = tmp_path / "runtime" / "capture-status.json"
    assert capture_status.snapshot(path) == {
        "available": False,
        "error": "capture_status_missing",
    }

    capture_status.write(
        path,
        {
            "last_success_at": "2026-11-08T12:00:00+00:00",
            "pending": -4,
            "failed": "many",
            "last_error": "SECRET-TRANSCRIPT",
        },
    )
    observed = capture_status.snapshot(path)
    assert observed["available"] is True
    assert observed["last_success_at"] == "2026-11-08T12:00:00Z"
    assert observed["pending"] == 0
    assert observed["failed"] == 0
    assert observed["last_error"] == "capture_failed"

    path.write_text("{not json", encoding="utf-8")
    assert capture_status.snapshot(path)["error"] == "capture_status_invalid"

    target = tmp_path / "elsewhere.json"
    target.write_text("{}", encoding="utf-8")
    path.unlink()
    path.symlink_to(target)
    assert capture_status.snapshot(path)["error"] == "capture_status_invalid"


def test_deadman_accepts_capture_written_status(tmp_path: Path) -> None:
    root = tmp_path / "root"
    now = _now()
    capture_status.write(
        capture_status.status_path(root),
        {
            "last_attempt_at": now,
            "last_success_at": now,
            "last_commit_at": now,
            "oldest_pending_at": None,
            "pending": 0,
            "failed": 0,
            "last_error": "none",
        },
    )
    stub_bin = tmp_path / "bin"
    stub_bin.mkdir()
    calls = tmp_path / "osascript.calls"
    stub = stub_bin / "osascript"
    stub.write_text(
        '#!/bin/sh\nprintf \'%s\\n\' "$*" >> "$OSASCRIPT_LOG"\n', encoding="utf-8"
    )
    stub.chmod(0o700)
    completed = subprocess.run(
        [str(DEADMAN)],
        capture_output=True,
        text=True,
        timeout=10,
        env={
            **os.environ,
            "CHRONOVISOR_ROOT": str(root),
            "OSASCRIPT_LOG": str(calls),
            "PATH": f"{stub_bin}{os.pathsep}{os.environ['PATH']}",
        },
    )

    assert completed.returncode == 0, completed.stderr
    observed = json.loads(
        (root / "runtime" / "deadman-status.json").read_text(encoding="utf-8")
    )
    assert observed["condition"] == "healthy"
    assert not calls.exists()


def test_capture_launchd_template_runs_the_canonical_wrapper() -> None:
    result = subprocess.run(
        ["plutil", "-lint", str(CAPTURE_PLIST)], capture_output=True, text=True
    )
    assert result.returncode == 0, result.stderr
    payload = plistlib.loads(CAPTURE_PLIST.read_bytes())
    assert payload["Label"] == "@@LAUNCHD_LABEL_PREFIX@@capture"
    assert payload["ProgramArguments"] == [
        "@@PROJECT_ROOT@@/scripts/chronovisor-capture"
    ]
    assert payload["StartInterval"] == 120
    assert payload["RunAtLoad"] is True


def test_chronovisor_status_reports_capture_status(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from chronovisor.core import reranker_client, semantic_client
    from chronovisor.hosts import server

    monkeypatch.setattr(server, "CHRONOVISOR_ROOT", tmp_path)
    now = _now()
    capture_status.write(
        capture_status.status_path(tmp_path),
        {
            "last_attempt_at": now,
            "last_success_at": now,
            "last_commit_at": now,
            "oldest_pending_at": None,
            "pending": 3,
            "failed": 1,
            "last_error": "permanent_failure",
        },
    )

    class Store:
        def refresh(self) -> None:
            return None

        def page_count(self, *, include_system: bool) -> int:
            assert include_system is False
            return 0

    monkeypatch.setattr(server, "get_store", lambda: Store())
    monkeypatch.setattr(server, "snapshot", lambda: {"by_status": {}})
    monkeypatch.setattr(
        semantic_client, "health", lambda _config: {"status": "unavailable"}
    )
    monkeypatch.setattr(
        reranker_client, "health", lambda _config: {"status": "unavailable"}
    )

    read = getattr(server.chronovisor_status, "fn", server.chronovisor_status)
    payload = json.loads(read())

    assert payload["capture_status"]["available"] is True
    assert payload["capture_status"]["pending"] == 3
    assert payload["capture_status"]["failed"] == 1
    assert payload["capture_status"]["last_error"] == "permanent_failure"
    assert payload["capture_status"]["last_success_at"] == now


def test_project_declares_capture_entry_point() -> None:
    import tomllib

    scripts = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))[
        "project"
    ]["scripts"]
    assert scripts["chronovisor-capture"] == "chronovisor.ops.capture:main"
