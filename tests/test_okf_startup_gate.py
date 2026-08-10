from __future__ import annotations

import json
import shutil
import stat
from pathlib import Path

import pytest

from chronovisor.core import okf_cutover, store
from chronovisor.core.okf_cutover import (
    OKFStartupBlocked,
    OKFStartupDecision,
    discover_okf_startup,
    execute_okf_cutover,
    recover_okf_cutover,
    require_okf_startup_allowed,
)
from chronovisor.core.okf_workspace import prepare_okf_workspace
from chronovisor.hosts import cli

FIXTURE = Path(__file__).parent / "fixtures" / "okf_workspace" / "source"


def _snapshot(root: Path) -> dict[str, tuple[int, bytes | None]]:
    if not root.exists():
        return {}
    return {
        path.relative_to(root).as_posix(): (
            stat.S_IMODE(path.lstat().st_mode),
            path.read_bytes() if path.is_file() else None,
        )
        for path in (root, *sorted(root.rglob("*")))
    }


def _legacy(tmp_path: Path) -> tuple[Path, Path]:
    root = tmp_path / "wiki"
    shutil.copytree(FIXTURE, root)
    runtime = root / "runtime"
    runtime.mkdir()
    return root, runtime


@pytest.mark.parametrize("shape", ["absent", "empty", "skeleton", "config"])
def test_bootstrap_discovery_is_read_only(tmp_path: Path, shape: str) -> None:
    root = tmp_path / "wiki"
    if shape != "absent":
        root.mkdir()
    if shape == "skeleton":
        for name in ("raw", "pages", "system", "runtime", "logs"):
            (root / name).mkdir()
    if shape == "config":
        (root / "config.toml").write_text("[hooks]\n", encoding="utf-8")
    before = _snapshot(root)

    decision = discover_okf_startup(root, root / "runtime")

    assert decision == OKFStartupDecision(True, "bootstrap", "uninitialized", "ok")
    assert _snapshot(root) == before
    assert root.exists() is (shape != "absent")


@pytest.mark.parametrize("shape", ["partial", "content", "symlink", "residue"])
def test_unsafe_unmigrated_layout_blocks_without_mutation(
    tmp_path: Path, shape: str
) -> None:
    root = tmp_path / "wiki"
    root.mkdir()
    if shape == "partial":
        (root / "index.md").write_text("partial", encoding="utf-8")
    elif shape == "content":
        (root / "pages").mkdir()
        (root / "pages" / "page.md").write_text("content", encoding="utf-8")
    elif shape == "symlink":
        outside = tmp_path / "outside"
        outside.mkdir()
        (root / "pages").symlink_to(outside)
    else:
        (root / "runtime" / "migrations").mkdir(parents=True)
    before = _snapshot(tmp_path)

    decision = discover_okf_startup(root, root / "runtime")

    assert not decision.allowed
    assert _snapshot(tmp_path) == before
    with pytest.raises(OKFStartupBlocked, match=decision.category):
        require_okf_startup_allowed(root, root / "runtime")
    assert _snapshot(tmp_path) == before
    with pytest.raises(OKFStartupBlocked, match=decision.category):
        store.init_chronovisor(store.RuntimeContext(root))
    assert _snapshot(tmp_path) == before


def test_absent_root_below_symlink_is_not_a_bootstrap_target(tmp_path: Path) -> None:
    outside = tmp_path / "outside"
    outside.mkdir()
    link = tmp_path / "link"
    link.symlink_to(outside)

    decision = discover_okf_startup(link / "wiki", link / "wiki" / "runtime")

    assert not decision.allowed
    assert not (outside / "wiki").exists()


def test_migration_residue_precedes_absent_source_bootstrap(tmp_path: Path) -> None:
    source = tmp_path / "absent-source"
    runtime = tmp_path / "runtime"
    (runtime / "migrations").mkdir(parents=True)

    decision = discover_okf_startup(source, runtime)

    assert not decision.allowed
    assert decision.category == "migration_residue"
    assert not source.exists()


def test_full_legacy_layout_is_allowed_with_existing_runtime_content(
    tmp_path: Path,
) -> None:
    root, runtime = _legacy(tmp_path)
    (runtime / "status.json").write_text('{"safe":true}\n', encoding="utf-8")
    (root / "other-state.json").write_text("{}\n", encoding="utf-8")

    assert discover_okf_startup(root, runtime) == OKFStartupDecision(
        True, "legacy", "unmigrated", "ok"
    )


def test_terminal_committed_and_rollback_proofs_select_exact_layout(
    tmp_path: Path,
) -> None:
    committed_root, committed_runtime = _legacy(tmp_path / "committed")
    prepare_okf_workspace(committed_root, committed_runtime, "commit-run")
    execute_okf_cutover(
        committed_root, committed_runtime, "commit-run", is_quiescent=lambda: True
    )

    assert discover_okf_startup(committed_root, committed_runtime) == (
        OKFStartupDecision(True, "okf_v0_2", "committed", "ok", "commit-run")
    )

    rollback_root, rollback_runtime = _legacy(tmp_path / "rollback")
    prepare_okf_workspace(rollback_root, rollback_runtime, "rollback-run")
    assert (
        recover_okf_cutover(
            rollback_root,
            rollback_runtime,
            "rollback-run",
            is_quiescent=lambda: True,
        )
        == "rollback-complete"
    )
    assert discover_okf_startup(rollback_root, rollback_runtime) == (
        OKFStartupDecision(True, "legacy", "rollback-complete", "ok", "rollback-run")
    )


@pytest.mark.parametrize(
    ("mutation", "category"),
    [
        ("tamper", "migration_proof_invalid"),
        ("multiple", "multiple_migrations"),
        ("sentinel", "restart_refusal_active"),
        ("receipt", "receipt_validation_unsupported"),
        ("unknown", "unsafe_migration_workspace"),
        ("nonterminal", "migration_nonterminal"),
        ("malformed", "migration_proof_invalid"),
        ("missing", "migration_proof_invalid"),
        ("symlink", "unsafe_migration_workspace"),
    ],
)
def test_migration_ambiguity_and_incomplete_proof_fail_closed(
    tmp_path: Path, mutation: str, category: str
) -> None:
    root, runtime = _legacy(tmp_path)
    workspace = prepare_okf_workspace(root, runtime, "run-001")
    if mutation == "tamper":
        execute_okf_cutover(root, runtime, "run-001", is_quiescent=lambda: True)
        activity = runtime / "activity.jsonl"
        activity.write_bytes(activity.read_bytes() + b"tamper")
    elif mutation == "multiple":
        (runtime / "migrations" / "run-002").mkdir()
    elif mutation == "receipt":
        shutil.rmtree(workspace)
        workspace.mkdir()
        (workspace / "receipt.json").write_text('{"private":"canary"}\n')
    elif mutation == "unknown":
        (workspace / "private-canary.txt").write_text("secret", encoding="utf-8")
    elif mutation == "nonterminal":
        (workspace / "restart-refusal.json").unlink()
    elif mutation == "malformed":
        (workspace / "restart-refusal.json").unlink()
        (workspace / "journal.json").write_text("private-canary", encoding="utf-8")
    elif mutation == "missing":
        (workspace / "restart-refusal.json").unlink()
        (workspace / "dry-run-manifest.json").unlink()
    elif mutation == "symlink":
        target = workspace / "staging" / "activity.jsonl"
        (workspace / "staging" / "private-link").symlink_to(target)

    decision = discover_okf_startup(root, runtime)

    assert not decision.allowed
    assert decision.category == category


def test_committed_store_initialization_never_writes_legacy_documents(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "wiki"
    monkeypatch.setattr(
        okf_cutover,
        "require_okf_startup_allowed",
        lambda *_args: OKFStartupDecision(
            True, "okf_v0_2", "committed", "ok", "run-001"
        ),
    )

    store.init_chronovisor(store.RuntimeContext(root))

    assert all((root / name).is_dir() for name in ("raw", "pages", "system"))
    assert not any(
        (root / name).exists() for name in ("index.md", "log.md", "schema.md")
    )


def test_okf_status_cli_is_read_only_and_content_free(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    root = tmp_path / "wiki"
    root.mkdir()
    (root / "private-canary.txt").write_text("secret-canary", encoding="utf-8")
    before = _snapshot(root)

    assert cli.main(["okf", "status", "--root", str(root), "--json"]) == 75
    output = capsys.readouterr().out

    assert json.loads(output) == {
        "allowed": False,
        "category": "unsafe_bootstrap_layout",
        "layout": "blocked",
        "run_id": None,
        "state": "blocked",
    }
    assert "secret-canary" not in output
    assert str(root) not in output
    assert _snapshot(root) == before


def test_okf_prepare_cli_is_single_use_and_private_mode(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    root, runtime = _legacy(tmp_path)

    assert (
        cli.main(
            ["okf", "prepare", "--run-id", "run-001", "--root", str(root), "--json"]
        )
        == 0
    )
    output = json.loads(capsys.readouterr().out)
    assert output == {
        "category": "ok",
        "prepared": True,
        "run_id": "run-001",
        "workspace": "runtime/migrations/run-001",
    }
    migration_dirs = [
        path for path in (runtime / "migrations").rglob("*") if path.is_dir()
    ]
    migration_files = [
        path for path in (runtime / "migrations").rglob("*") if path.is_file()
    ]
    assert stat.S_IMODE((runtime / "migrations").stat().st_mode) == 0o700
    assert all(stat.S_IMODE(path.stat().st_mode) == 0o700 for path in migration_dirs)
    assert all(stat.S_IMODE(path.stat().st_mode) == 0o600 for path in migration_files)

    assert (
        cli.main(
            ["okf", "prepare", "--run-id", "run-001", "--root", str(root), "--json"]
        )
        == 75
    )
    duplicate = json.loads(capsys.readouterr().out)
    assert duplicate["prepared"] is False
    assert duplicate["category"] == "restart_refusal_active"

    assert (
        cli.main(
            [
                "okf",
                "prepare",
                "--run-id",
                "private\ncanary",
                "--root",
                str(root),
                "--json",
            ]
        )
        == 75
    )
    rejected = capsys.readouterr().out
    assert "private" not in rejected
    assert "canary" not in rejected
