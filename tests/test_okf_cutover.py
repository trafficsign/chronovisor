from __future__ import annotations

import json
import os
import shutil
import stat
from pathlib import Path

import pytest

from chronovisor.core.canonical_document import parse_document
from chronovisor.core.canonical_json import (
    canonical_json_line_bytes_strict,
    canonical_json_sha256_strict,
)
from chronovisor.core.durable_state import seal_object
from chronovisor.core.okf_cutover import (
    CLEANUP_FAULT_POINTS,
    CUTOVER_FAULT_POINTS,
    RECEIPT_FILENAME,
    RECEIPT_SCHEMA,
    OKFStartupDecision,
    cleanup_okf_cutover,
    discover_okf_startup,
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


def test_system_identity_hashes_survive_interrupted_cutover_recovery(
    tmp_path: Path,
) -> None:
    source, runtime, workspace = _setup(tmp_path)
    manifest_path = workspace / "dry-run-manifest.json"
    manifest_raw = manifest_path.read_bytes()
    system_documents = json.loads(manifest_raw)["system_documents"]

    def crash(point: str) -> None:
        if point == "publish-system:after-rename":
            raise InjectedCrash(point)

    with pytest.raises(InjectedCrash, match="publish-system:after-rename"):
        execute_okf_cutover(
            source,
            runtime,
            "run-001",
            is_quiescent=lambda: True,
            fault_inject=crash,
        )
    assert recover_okf_cutover(
        source, runtime, "run-001", is_quiescent=lambda: True
    ) == "rollback-complete"
    assert manifest_path.read_bytes() == manifest_raw

    staged_system = workspace / "staging" / "system"
    for item in system_documents:
        document = parse_document(
            (staged_system / item["relative_path"]).read_bytes()
        )
        identity_source = item["identity_source"]
        identity = (
            item["relative_path"]
            if identity_source == "relative_path"
            else document.metadata[identity_source]
        )
        assert item["identity_sha256"] == canonical_json_sha256_strict(
            {"source": identity_source, "value": identity}
        )
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


def test_committed_cleanup_publishes_content_free_receipt_and_allows_later_writes(
    tmp_path: Path,
) -> None:
    source, runtime, workspace = _setup(tmp_path)
    execute_okf_cutover(source, runtime, "run-001", is_quiescent=lambda: True)
    live_before = _live(source, runtime)
    raw_before = _tree(source / "raw")
    unrelated = runtime / "status.json"
    unrelated.write_text('{"private":"runtime-canary"}\n', encoding="utf-8")

    assert cleanup_okf_cutover(
        source, runtime, "run-001", is_quiescent=lambda: True
    ) == "committed"

    assert [path.name for path in workspace.iterdir()] == [RECEIPT_FILENAME]
    receipt_raw = (workspace / RECEIPT_FILENAME).read_bytes()
    receipt = json.loads(receipt_raw)
    assert set(receipt) == {
        "schema",
        "version",
        "run_id",
        "state",
        "manifest_sha256",
        "seal_sha256",
    }
    assert receipt["schema"] == RECEIPT_SCHEMA
    assert receipt["state"] == "committed"
    assert receipt_raw == canonical_json_line_bytes_strict(receipt)
    assert len(receipt["manifest_sha256"]) == 64
    assert not any(str(value).startswith("/") for value in receipt.values())
    assert stat.S_IMODE((workspace / RECEIPT_FILENAME).stat().st_mode) == 0o600
    assert "runtime-canary" not in receipt_raw.decode()
    assert "Private state summary" not in receipt_raw.decode()
    assert _live(source, runtime) == live_before
    assert _tree(source / "raw") == raw_before
    assert unrelated.read_text(encoding="utf-8") == '{"private":"runtime-canary"}\n'

    page = source / "pages" / "notes" / "source.md"
    system = source / "system" / "current-state.md"
    activity = runtime / "activity.jsonl"
    for path in (page, system, activity):
        path.write_bytes(path.read_bytes() + b"\n")
    assert discover_okf_startup(source, runtime) == OKFStartupDecision(
        True, "okf_v0_2", "committed", "ok", "run-001"
    )
    assert okf_startup_allowed(source, runtime, "run-001")
    before_idempotent = _tree(workspace)
    assert cleanup_okf_cutover(
        source, runtime, "run-001", is_quiescent=lambda: False
    ) == "committed"
    assert _tree(workspace) == before_idempotent


def test_rollback_cleanup_publishes_legacy_receipt(tmp_path: Path) -> None:
    source, runtime, workspace = _setup(tmp_path)
    assert recover_okf_cutover(
        source, runtime, "run-001", is_quiescent=lambda: True
    ) == "rollback-complete"

    assert cleanup_okf_cutover(
        source, runtime, "run-001", is_quiescent=lambda: True
    ) == "rollback-complete"

    assert [path.name for path in workspace.iterdir()] == [RECEIPT_FILENAME]
    assert discover_okf_startup(source, runtime) == OKFStartupDecision(
        True, "legacy", "rollback-complete", "ok", "run-001"
    )
    assert okf_startup_allowed(source, runtime, "run-001")


@pytest.mark.parametrize("fault_point", CLEANUP_FAULT_POINTS)
def test_every_cleanup_boundary_remains_blocked_or_safely_complete(
    tmp_path: Path, fault_point: str
) -> None:
    source, runtime, workspace = _setup(tmp_path)
    execute_okf_cutover(source, runtime, "run-001", is_quiescent=lambda: True)

    def crash(point: str) -> None:
        if point == fault_point:
            if point == "after-remove-cutover-lock":
                assert not (workspace / "cutover.lock").exists()
                assert (workspace / "journal.json").is_file()
                assert (workspace / RECEIPT_FILENAME).is_file()
            raise InjectedCrash(point)

    with pytest.raises(InjectedCrash, match=fault_point):
        cleanup_okf_cutover(
            source,
            runtime,
            "run-001",
            is_quiescent=lambda: True,
            fault_inject=crash,
        )

    decision = discover_okf_startup(source, runtime)
    if fault_point == "after-cleanup-journal":
        assert json.loads((workspace / "journal.json").read_bytes())[
            "cleanup_in_progress"
        ] is True
        assert decision.category == "cleanup_in_progress"
    if fault_point == "after-remove-cutover-lock":
        assert not (workspace / "cutover.lock").exists()
        assert (workspace / "journal.json").is_file()
        assert (workspace / RECEIPT_FILENAME).is_file()
    if fault_point == "after-journal-remove":
        assert decision.allowed
    else:
        assert not decision.allowed
    assert cleanup_okf_cutover(
        source, runtime, "run-001", is_quiescent=lambda: True
    ) == "committed"
    assert [path.name for path in workspace.iterdir()] == [RECEIPT_FILENAME]
    assert discover_okf_startup(source, runtime).allowed


def test_committed_receipt_accepts_removed_legacy_root_documents(
    tmp_path: Path,
) -> None:
    source, runtime, _workspace = _setup(tmp_path)
    execute_okf_cutover(source, runtime, "run-001", is_quiescent=lambda: True)
    cleanup_okf_cutover(source, runtime, "run-001", is_quiescent=lambda: True)
    for name in ("index.md", "log.md", "schema.md"):
        (source / name).unlink()

    assert discover_okf_startup(source, runtime) == OKFStartupDecision(
        True, "okf_v0_2", "committed", "ok", "run-001"
    )
    assert okf_startup_allowed(source, runtime, "run-001")


@pytest.mark.parametrize(
    "mutation",
    ["partial-root", "missing-reserved", "pages-symlink", "activity-symlink"],
)
def test_committed_receipt_rejects_unsafe_or_incomplete_live_layout(
    tmp_path: Path, mutation: str
) -> None:
    source, runtime, _workspace = _setup(tmp_path)
    execute_okf_cutover(source, runtime, "run-001", is_quiescent=lambda: True)
    cleanup_okf_cutover(source, runtime, "run-001", is_quiescent=lambda: True)

    if mutation == "partial-root":
        (source / "index.md").unlink()
    elif mutation == "missing-reserved":
        (source / "pages" / "index.md").unlink()
    elif mutation == "pages-symlink":
        shutil.rmtree(source / "pages")
        (source / "pages").symlink_to(source / "system")
    else:
        (runtime / "activity.jsonl").unlink()
        (runtime / "activity.jsonl").symlink_to(source / "index.md")

    assert not discover_okf_startup(source, runtime).allowed
    assert not okf_startup_allowed(source, runtime, "run-001")


def test_rollback_receipt_requires_all_legacy_root_documents(tmp_path: Path) -> None:
    source, runtime, _workspace = _setup(tmp_path)
    recover_okf_cutover(source, runtime, "run-001", is_quiescent=lambda: True)
    cleanup_okf_cutover(source, runtime, "run-001", is_quiescent=lambda: True)
    (source / "index.md").unlink()

    assert not discover_okf_startup(source, runtime).allowed
    assert not okf_startup_allowed(source, runtime, "run-001")


@pytest.mark.parametrize(
    "mutation",
    ["malformed", "run", "hash", "state", "extra-field", "extra-file", "symlink"],
)
def test_receipt_only_startup_rejects_invalid_or_ambiguous_state(
    tmp_path: Path, mutation: str
) -> None:
    source, runtime, workspace = _setup(tmp_path)
    execute_okf_cutover(source, runtime, "run-001", is_quiescent=lambda: True)
    cleanup_okf_cutover(source, runtime, "run-001", is_quiescent=lambda: True)
    receipt_path = workspace / RECEIPT_FILENAME
    receipt = json.loads(receipt_path.read_bytes())

    if mutation == "malformed":
        receipt_path.write_text("private-content-canary", encoding="utf-8")
    elif mutation == "extra-file":
        (workspace / "private-name-canary").write_text(
            "private-content-canary", encoding="utf-8"
        )
    elif mutation == "symlink":
        receipt_path.unlink()
        receipt_path.symlink_to(source / "index.md")
    else:
        if mutation == "run":
            receipt["run_id"] = "other-run"
        elif mutation == "hash":
            receipt["manifest_sha256"] = (
                ("0" if receipt["manifest_sha256"][0] != "0" else "1")
                + receipt["manifest_sha256"][1:]
            )
        elif mutation == "state":
            receipt["state"] = "prepared"
        else:
            receipt["private"] = "private-content-canary"
        receipt_path.write_bytes(canonical_json_line_bytes_strict(receipt))

    decision = discover_okf_startup(source, runtime)

    assert not decision.allowed
    assert not okf_startup_allowed(source, runtime, "run-001")
    assert "private" not in decision.category


def test_cleanup_resume_rejects_receipt_hash_not_bound_to_terminal_journal(
    tmp_path: Path,
) -> None:
    source, runtime, workspace = _setup(tmp_path)
    execute_okf_cutover(source, runtime, "run-001", is_quiescent=lambda: True)

    def crash(point: str) -> None:
        if point == "after-receipt-write":
            raise InjectedCrash(point)

    with pytest.raises(InjectedCrash):
        cleanup_okf_cutover(
            source,
            runtime,
            "run-001",
            is_quiescent=lambda: True,
            fault_inject=crash,
        )
    receipt_path = workspace / RECEIPT_FILENAME
    receipt = json.loads(receipt_path.read_bytes())
    receipt["manifest_sha256"] = "f" * 64
    receipt_path.write_bytes(canonical_json_line_bytes_strict(seal_object(receipt)))

    with pytest.raises(ValueError, match="cleanup journal"):
        cleanup_okf_cutover(
            source, runtime, "run-001", is_quiescent=lambda: True
        )
    assert not discover_okf_startup(source, runtime).allowed


def test_cleanup_rejects_unknown_artifact_without_mutation_or_content_leak(
    tmp_path: Path,
) -> None:
    source, runtime, workspace = _setup(tmp_path)
    execute_okf_cutover(source, runtime, "run-001", is_quiescent=lambda: True)
    unknown = workspace / "private-name-canary"
    unknown.write_text("private-content-canary", encoding="utf-8")
    before = _tree(workspace)

    with pytest.raises(ValueError) as raised:
        cleanup_okf_cutover(
            source, runtime, "run-001", is_quiescent=lambda: True
        )

    assert "private-name-canary" not in str(raised.value)
    assert "private-content-canary" not in str(raised.value)
    assert _tree(workspace) == before
    assert not (workspace / RECEIPT_FILENAME).exists()


def test_cleanup_rejects_symlinked_artifact_without_mutation(tmp_path: Path) -> None:
    source, runtime, workspace = _setup(tmp_path)
    execute_okf_cutover(source, runtime, "run-001", is_quiescent=lambda: True)
    link = workspace / "rollback-backup" / "private-link"
    link.symlink_to(source / "index.md")
    before = _tree(workspace)

    with pytest.raises(ValueError, match="unsafe"):
        cleanup_okf_cutover(
            source, runtime, "run-001", is_quiescent=lambda: True
        )

    assert link.is_symlink()
    assert _tree(workspace) == before
    assert not (workspace / RECEIPT_FILENAME).exists()


def test_cleanup_quiescence_failure_does_not_issue_receipt(tmp_path: Path) -> None:
    source, runtime, workspace = _setup(tmp_path)
    execute_okf_cutover(source, runtime, "run-001", is_quiescent=lambda: True)
    journal_before = (workspace / "journal.json").read_bytes()

    with pytest.raises(RuntimeError, match="quiescent"):
        cleanup_okf_cutover(
            source, runtime, "run-001", is_quiescent=lambda: False
        )

    assert (workspace / "journal.json").read_bytes() == journal_before
    assert not (workspace / RECEIPT_FILENAME).exists()
    assert okf_startup_allowed(source, runtime, "run-001")
