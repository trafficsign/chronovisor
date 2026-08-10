from __future__ import annotations

import json
import os
import shutil
from pathlib import Path

import pytest

from chronovisor.core.canonical_json import canonical_json_line_bytes_strict
from chronovisor.core.okf_cutover import (
    CUTOVER_FAULT_POINTS,
    execute_okf_cutover,
    okf_startup_allowed,
    recover_okf_cutover,
)
from chronovisor.core.okf_workspace import (
    RESTART_REFUSAL_FILENAME,
    prepare_okf_workspace,
)

FIXTURE = Path(__file__).parent / "fixtures" / "okf_workspace" / "source"


class InjectedCrash(RuntimeError):
    pass


def _setup(
    tmp_path: Path, *, old_activity_present: bool = True
) -> tuple[Path, Path, Path]:
    source = tmp_path / "source"
    shutil.copytree(FIXTURE, source)
    runtime = tmp_path / "runtime"
    runtime.mkdir()
    if old_activity_present:
        (runtime / "activity.jsonl").write_bytes(b'{"legacy":true}\n')
    workspace = prepare_okf_workspace(source, runtime, "run-001")
    return source, runtime, workspace


def _tree(root: Path) -> dict[str, bytes]:
    return {
        path.relative_to(root).as_posix(): path.read_bytes()
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def _live(
    source: Path, runtime: Path
) -> tuple[dict[str, bytes], dict[str, bytes], dict[str, bytes | None]]:
    activity = runtime / "activity.jsonl"
    return (
        _tree(source / "pages"),
        _tree(source / "system"),
        {"activity.jsonl": activity.read_bytes() if activity.exists() else None},
    )


def _static_source(source: Path) -> tuple[dict[str, bytes], dict[str, bytes]]:
    return (
        _tree(source / "raw"),
        {
            name: (source / name).read_bytes()
            for name in ("index.md", "log.md", "schema.md")
        },
    )


def _state(workspace: Path) -> str:
    state = json.loads((workspace / "journal.json").read_bytes())["state"]
    assert isinstance(state, str)
    return state


def test_cutover_publishes_all_assets_and_keeps_rollback_backup(tmp_path: Path) -> None:
    source, runtime, workspace = _setup(tmp_path)
    old = _live(source, runtime)
    new = (
        _tree(workspace / "staging" / "pages"),
        _tree(workspace / "staging" / "system"),
        {
            "activity.jsonl": (workspace / "staging" / "activity.jsonl").read_bytes()
        },
    )
    static = _static_source(source)

    assert execute_okf_cutover(
        source, runtime, "run-001", is_quiescent=lambda: True
    ) == "committed"

    assert _live(source, runtime) == new
    assert (
        _tree(workspace / "rollback-backup" / "pages"),
        _tree(workspace / "rollback-backup" / "system"),
        {
            "activity.jsonl": (
                workspace / "rollback-backup" / "activity.jsonl"
            ).read_bytes()
        },
    ) == old
    assert _static_source(source) == static
    assert _state(workspace) == "committed"
    assert json.loads((workspace / "journal.json").read_bytes())["old_activity"][
        "present"
    ] is True
    assert not (workspace / RESTART_REFUSAL_FILENAME).exists()
    assert okf_startup_allowed(source, runtime, "run-001")


def test_cutover_accepts_absent_old_activity_and_records_skipped_backup(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source, runtime, workspace = _setup(tmp_path, old_activity_present=False)
    new_activity = (workspace / "staging" / "activity.jsonl").read_bytes()
    destinations: list[Path] = []
    real_rename = os.rename

    def tracked_rename(
        source_path: str | os.PathLike[str],
        destination_path: str | os.PathLike[str],
    ) -> None:
        destinations.append(Path(destination_path))
        real_rename(source_path, destination_path)

    monkeypatch.setattr("chronovisor.core.okf_cutover.os.rename", tracked_rename)

    assert execute_okf_cutover(
        source, runtime, "run-001", is_quiescent=lambda: True
    ) == "committed"

    journal = json.loads((workspace / "journal.json").read_bytes())
    assert journal["old_activity"] == {"present": False}
    assert "backup-activity:skipped" in journal["completed"]
    assert not (workspace / "rollback-backup" / "activity.jsonl").exists()
    assert workspace / "rollback-backup" / "activity.jsonl" not in destinations
    assert (runtime / "activity.jsonl").read_bytes() == new_activity
    assert okf_startup_allowed(source, runtime, "run-001")


def test_quiescence_failure_performs_no_rename(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source, runtime, workspace = _setup(tmp_path)
    renames = 0
    real_rename = os.rename

    def tracked_rename(
        source_path: str | os.PathLike[str],
        destination_path: str | os.PathLike[str],
    ) -> None:
        nonlocal renames
        renames += 1
        real_rename(source_path, destination_path)

    monkeypatch.setattr("chronovisor.core.okf_cutover.os.rename", tracked_rename)
    with pytest.raises(RuntimeError, match="quiescent"):
        execute_okf_cutover(source, runtime, "run-001", is_quiescent=lambda: False)

    assert renames == 0
    assert _state(workspace) == "prepared"
    assert not okf_startup_allowed(source, runtime, "run-001")


@pytest.mark.parametrize("drift", ["raw", "source", "staged"])
def test_manifest_drift_is_rejected_before_first_rename(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, drift: str
) -> None:
    source, runtime, _workspace = _setup(tmp_path)
    target = {
        "raw": source / "raw" / "sessions" / "session.jsonl",
        "source": source / "pages" / "notes" / "source.md",
        "staged": _workspace / "staging" / "pages" / "notes" / "source.md",
    }[drift]
    target.write_bytes(target.read_bytes() + b"drift")
    renames = 0

    def unexpected_rename(_source: object, _destination: object) -> None:
        nonlocal renames
        renames += 1

    monkeypatch.setattr("chronovisor.core.okf_cutover.os.rename", unexpected_rename)
    with pytest.raises(ValueError, match="hash inventory mismatch"):
        execute_okf_cutover(source, runtime, "run-001", is_quiescent=lambda: True)
    assert renames == 0


@pytest.mark.parametrize(
    "old_activity_present", [True, False], ids=["old-activity", "old-absent"]
)
@pytest.mark.parametrize("fault_point", CUTOVER_FAULT_POINTS)
def test_every_cutover_boundary_recovers_to_all_old_or_all_new(
    tmp_path: Path, fault_point: str, old_activity_present: bool
) -> None:
    source, runtime, workspace = _setup(
        tmp_path, old_activity_present=old_activity_present
    )
    old = _live(source, runtime)
    new = (
        _tree(workspace / "staging" / "pages"),
        _tree(workspace / "staging" / "system"),
        {
            "activity.jsonl": (workspace / "staging" / "activity.jsonl").read_bytes()
        },
    )
    static = _static_source(source)

    def crash(point: str) -> None:
        if point == fault_point:
            raise InjectedCrash(point)

    with pytest.raises(InjectedCrash, match=fault_point):
        execute_okf_cutover(
            source,
            runtime,
            "run-001",
            is_quiescent=lambda: True,
            fault_inject=crash,
        )

    assert _static_source(source) == static
    if fault_point != "after-sentinel-remove":
        assert not okf_startup_allowed(source, runtime, "run-001")
    terminal = recover_okf_cutover(
        source, runtime, "run-001", is_quiescent=lambda: True
    )
    observed = _live(source, runtime)
    assert (terminal == "rollback-complete" and observed == old) or (
        terminal == "committed" and observed == new
    )
    assert _static_source(source) == static
    assert not (workspace / RESTART_REFUSAL_FILENAME).exists()
    assert okf_startup_allowed(source, runtime, "run-001")


def test_startup_gate_rejects_unknown_state_and_terminal_hash_mismatch(
    tmp_path: Path,
) -> None:
    source, runtime, workspace = _setup(tmp_path)
    assert not okf_startup_allowed(source, runtime, "run-001")
    execute_okf_cutover(source, runtime, "run-001", is_quiescent=lambda: True)
    assert okf_startup_allowed(source, runtime, "run-001")

    journal_path = workspace / "journal.json"
    journal = json.loads(journal_path.read_bytes())
    journal["state"] = "unknown"
    journal_path.write_bytes(canonical_json_line_bytes_strict(journal))
    assert not okf_startup_allowed(source, runtime, "run-001")

    journal["state"] = "committed"
    journal_path.write_bytes(canonical_json_line_bytes_strict(journal))
    activity = runtime / "activity.jsonl"
    activity.write_bytes(activity.read_bytes() + b"tamper")
    assert not okf_startup_allowed(source, runtime, "run-001")


def test_unsafe_run_id_and_symlinked_staging_are_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source, runtime, workspace = _setup(tmp_path)
    with pytest.raises(ValueError, match="safe single path component"):
        execute_okf_cutover(source, runtime, "../escape", is_quiescent=lambda: True)

    staged = workspace / "staging" / "pages" / "notes" / "source.md"
    staged.unlink()
    staged.symlink_to(source / "pages" / "notes" / "source.md")
    renames = 0

    def unexpected_rename(_source: object, _destination: object) -> None:
        nonlocal renames
        renames += 1

    monkeypatch.setattr("chronovisor.core.okf_cutover.os.rename", unexpected_rename)
    with pytest.raises(ValueError, match="symlink"):
        execute_okf_cutover(source, runtime, "run-001", is_quiescent=lambda: True)
    assert renames == 0


def test_recovery_rejects_tampered_backup_and_keeps_gate_closed(
    tmp_path: Path,
) -> None:
    source, runtime, workspace = _setup(tmp_path)

    def crash(point: str) -> None:
        if point == "backup-pages:after-rename":
            raise InjectedCrash(point)

    with pytest.raises(InjectedCrash):
        execute_okf_cutover(
            source,
            runtime,
            "run-001",
            is_quiescent=lambda: True,
            fault_inject=crash,
        )
    backup_page = workspace / "rollback-backup" / "pages" / "notes" / "source.md"
    backup_page.write_bytes(backup_page.read_bytes() + b"tamper")

    with pytest.raises(ValueError, match="hash inventory mismatch"):
        recover_okf_cutover(
            source, runtime, "run-001", is_quiescent=lambda: True
        )
    assert not okf_startup_allowed(source, runtime, "run-001")
    assert (workspace / RESTART_REFUSAL_FILENAME).exists()
