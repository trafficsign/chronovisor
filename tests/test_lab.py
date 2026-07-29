from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from chronovisor.lab import cli
from chronovisor.lab.harness import (
    LabHarness,
    aggregate_channel_metrics,
    require_contract,
    require_file_hashes,
)


def test_harness_seals_selection_and_preregistration(tmp_path: Path) -> None:
    harness = LabHarness(tmp_path, "fixed-gate")

    selection = harness.seal_selection({"schema": "selection.v1"})
    preregistration = harness.lock_preregistration({"schema": "prereg.v1"})

    assert selection["schema"] == "selection.v1"
    assert selection["seal_sha256"]
    assert preregistration["schema"] == "prereg.v1"
    assert preregistration["seal_sha256"]
    assert harness.output_root == tmp_path / "classification" / "fixed-gate"


def test_harness_validates_contract_hashes_and_metrics(tmp_path: Path) -> None:
    artifact = tmp_path / "artifact"
    artifact.write_text("fixed", encoding="utf-8")

    require_contract(
        {"schema": "gate.v1", "sample_size": 2},
        schema="gate.v1",
        exact={"sample_size": 2},
    )
    require_file_hashes(
        {artifact: "fixed"},
        digest=lambda path: path.read_text(encoding="utf-8"),
    )
    assert aggregate_channel_metrics(
        [{"ok": True}],
        ["raw", "fused"],
        lambda cases, channel: {"channel": channel, "case_count": len(cases)},
    ) == {
        "raw": {"channel": "raw", "case_count": 1},
        "fused": {"channel": "fused", "case_count": 1},
    }


def test_lab_cli_forwards_arguments(monkeypatch) -> None:
    calls: list[list[str]] = []
    monkeypatch.setitem(cli.COMMANDS, "fake", ("fake.module", True))
    monkeypatch.setattr(
        cli.importlib,
        "import_module",
        lambda _name: SimpleNamespace(main=lambda args: calls.append(args) or 0),
    )

    assert cli.main(["fake", "--root", "/tmp/example"]) == 0
    assert calls == [["--root", "/tmp/example"]]


def test_lab_cli_rejects_unknown_command(capsys) -> None:
    assert cli.main(["missing"]) == 2
    assert "unknown command: missing" in capsys.readouterr().err
