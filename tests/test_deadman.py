from __future__ import annotations

import json
import os
import plistlib
import stat
import subprocess
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "chronovisor-deadman"
PLIST = ROOT / "launchd" / "com.trafficsign.chronovisor-deadman.plist"


def _timestamp(delta: timedelta = timedelta()) -> str:
    return (
        (datetime.now(UTC) + delta).isoformat(timespec="seconds").replace("+00:00", "Z")
    )


def _status(
    *,
    last_success_at: str | None = None,
    failed: int = 0,
    pending: int = 0,
    last_error: str = "none",
) -> dict[str, object]:
    now = _timestamp()
    return {
        "version": 1,
        "last_attempt_at": now,
        "last_success_at": last_success_at,
        "last_commit_at": now,
        "oldest_pending_at": None,
        "pending": pending,
        "failed": failed,
        "last_error": last_error,
    }


@pytest.fixture
def monitor(tmp_path: Path) -> tuple[Path, Path, Path]:
    root = tmp_path / "chronovisor root"
    stub_bin = tmp_path / "bin"
    stub_bin.mkdir()
    calls = tmp_path / "osascript.calls"
    stub = stub_bin / "osascript"
    stub.write_text(
        '#!/bin/sh\nprintf \'%s\\n\' "$*" >> "$OSASCRIPT_LOG"\n',
        encoding="utf-8",
    )
    stub.chmod(0o700)
    return root, stub_bin, calls


def _run(
    monitor: tuple[Path, Path, Path],
    *,
    status: dict[str, object] | None = None,
) -> tuple[Path, dict[str, object]]:
    root, stub_bin, calls = monitor
    runtime = root / "runtime"
    runtime.mkdir(parents=True, exist_ok=True)
    capture = runtime / "capture-status.json"
    if status is None:
        capture.unlink(missing_ok=True)
    else:
        capture.write_text(json.dumps(status) + "\n", encoding="utf-8")
        capture.chmod(0o600)
    env = {
        **os.environ,
        "CHRONOVISOR_ROOT": str(root),
        "OSASCRIPT_LOG": str(calls),
        "PATH": f"{stub_bin}{os.pathsep}{os.environ['PATH']}",
    }
    completed = subprocess.run(
        [str(SCRIPT)],
        env=env,
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert completed.returncode == 0, completed.stderr
    output = runtime / "deadman-status.json"
    assert output.is_file()
    return calls, json.loads(output.read_text(encoding="utf-8"))


def _calls(path: Path) -> list[str]:
    return path.read_text(encoding="utf-8").splitlines() if path.exists() else []


def test_valid_now_is_healthy_without_notification(monitor) -> None:
    calls, observed = _run(monitor, status=_status(last_success_at=_timestamp()))

    assert _calls(calls) == []
    assert observed["version"] == 1
    assert observed["status"] == "healthy"
    assert observed["condition"] == "healthy"


def test_missing_status_alerts_once_and_writes_private_state(monitor) -> None:
    calls, observed = _run(monitor)
    root, _stub_bin, _calls_path = monitor

    assert len(_calls(calls)) == 1
    assert "display notification" in _calls(calls)[0]
    assert observed["status"] == "alert"
    assert observed["condition"] == "missing"
    state = root / "runtime" / "deadman-state"
    output = root / "runtime" / "deadman-status.json"
    assert stat.S_IMODE(state.stat().st_mode) == 0o600
    assert stat.S_IMODE(output.stat().st_mode) == 0o600


def test_null_success_is_stale(monitor) -> None:
    calls, observed = _run(monitor, status=_status())

    assert observed["condition"] == "stale"
    assert len(_calls(calls)) == 1


def test_stale_and_failed_conditions_alert(monitor) -> None:
    calls, stale = _run(
        monitor,
        status=_status(last_success_at=_timestamp(timedelta(minutes=-16))),
    )
    assert stale["condition"] == "stale"
    assert len(_calls(calls)) == 1

    calls, failed = _run(
        monitor,
        status=_status(
            last_success_at=_timestamp(), failed=1, last_error="capture_failed"
        ),
    )
    assert failed["condition"] == "failed"
    assert len(_calls(calls)) == 2


def test_repeated_condition_is_throttled_and_recovery_resets(monitor) -> None:
    calls, first = _run(
        monitor,
        status=_status(
            last_success_at=_timestamp(), failed=1, last_error="capture_failed"
        ),
    )
    assert first["condition"] == "failed"
    assert len(_calls(calls)) == 1

    calls, second = _run(
        monitor,
        status=_status(
            last_success_at=_timestamp(), failed=1, last_error="capture_failed"
        ),
    )
    assert second["condition"] == "failed"
    assert len(_calls(calls)) == 1

    calls, recovered = _run(monitor, status=_status(last_success_at=_timestamp()))
    assert recovered["status"] == "healthy"
    assert len(_calls(calls)) == 1

    calls, repeated = _run(
        monitor,
        status=_status(
            last_success_at=_timestamp(), failed=1, last_error="capture_failed"
        ),
    )
    assert repeated["condition"] == "failed"
    assert len(_calls(calls)) == 2


def test_invalid_status_alert_is_privacy_safe(monitor) -> None:
    root, stub_bin, calls = monitor
    runtime = root / "runtime"
    runtime.mkdir(parents=True, exist_ok=True)
    marker = "SECRET-TRANSCRIPT-CONTENT"
    (runtime / "capture-status.json").write_text(
        '{"version":1,"failed":"oops","last_error":"' + marker + '"}\n',
        encoding="utf-8",
    )
    env = {
        **os.environ,
        "CHRONOVISOR_ROOT": str(root),
        "OSASCRIPT_LOG": str(calls),
        "PATH": f"{stub_bin}{os.pathsep}{os.environ['PATH']}",
    }
    completed = subprocess.run(
        [str(SCRIPT)],
        env=env,
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert completed.returncode == 0, completed.stderr
    observed = json.loads((runtime / "deadman-status.json").read_text())
    assert observed["condition"] == "invalid"
    assert len(_calls(calls)) == 1
    assert marker not in (runtime / "deadman-status.json").read_text()
    assert marker not in "\n".join(_calls(calls))
    assert marker not in completed.stdout
    assert marker not in completed.stderr


def test_malformed_status_alerts_without_details(monitor) -> None:
    root, stub_bin, calls = monitor
    runtime = root / "runtime"
    runtime.mkdir(parents=True, exist_ok=True)
    (runtime / "capture-status.json").write_text('{"version":1,\n', encoding="utf-8")
    completed = subprocess.run(
        [str(SCRIPT)],
        env={
            **os.environ,
            "CHRONOVISOR_ROOT": str(root),
            "OSASCRIPT_LOG": str(calls),
            "PATH": f"{stub_bin}{os.pathsep}{os.environ['PATH']}",
        },
        capture_output=True,
        text=True,
        timeout=10,
    )

    assert completed.returncode == 0, completed.stderr
    observed = json.loads((runtime / "deadman-status.json").read_text())
    assert observed["condition"] == "invalid"
    assert len(_calls(calls)) == 1


def test_launchd_template_is_valid_and_runs_every_five_minutes() -> None:
    result = subprocess.run(
        ["plutil", "-lint", str(PLIST)], capture_output=True, text=True
    )
    assert result.returncode == 0, result.stderr
    payload = plistlib.loads(PLIST.read_bytes())
    assert payload["Label"] == "@@LAUNCHD_LABEL_PREFIX@@deadman"
    assert payload["ProgramArguments"] == [
        "@@PROJECT_ROOT@@/scripts/chronovisor-deadman"
    ]
    assert payload["StartInterval"] == 300
    assert payload["RunAtLoad"] is True
    assert "@@UVX@@" not in PLIST.read_text(encoding="utf-8")
    assert "@@PYTHON@@" not in PLIST.read_text(encoding="utf-8")
