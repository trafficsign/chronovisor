from __future__ import annotations

import ast
from pathlib import Path

import tomllib

from chronovisor.decision import failure_supervisor
from chronovisor.raw import raw_replay


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_every_console_entry_point_has_a_docstring() -> None:
    project = tomllib.loads((PROJECT_ROOT / "pyproject.toml").read_text())

    missing: list[str] = []
    for command, target in project["project"]["scripts"].items():
        module_name, function_name = target.split(":", 1)
        path = (
            PROJECT_ROOT
            / "src"
            / Path(*module_name.split("."))
        ).with_suffix(".py")
        tree = ast.parse(path.read_text(encoding="utf-8"))
        function = next(
            node
            for node in tree.body
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            and node.name == function_name
        )
        if ast.get_docstring(function) is None:
            missing.append(command)

    assert missing == []


def test_quality_tools_are_scoped_to_the_staged_baseline() -> None:
    project = tomllib.loads((PROJECT_ROOT / "pyproject.toml").read_text())

    assert project["tool"]["ruff"]["target-version"] == "py311"
    assert project["tool"]["ruff"]["lint"]["select"] == [
        "E9",
        "F63",
        "F7",
        "F82",
    ]
    assert project["tool"]["mypy"]["files"] == ["src/chronovisor/core"]
    assert project["tool"]["mypy"]["strict"] is True


def test_semantic_defer_packet_evidence_returns_validated_authority(
    monkeypatch,
    tmp_path: Path,
) -> None:
    authority = "a" * 64
    raw_sha256 = "b" * 64
    raw_name = "raw.jsonl"
    packet_path = tmp_path / "packet.json"
    packet = {
        "authority_epoch": authority,
        "source_raws": [{"filename": raw_name, "sha256": raw_sha256}],
        "error": "local semantic no quorum",
        "job_id": "job-1",
    }
    monkeypatch.setattr(
        failure_supervisor,
        "_current_adopted_authority_epoch",
        lambda: authority,
    )
    monkeypatch.setattr(
        failure_supervisor,
        "_semantic_defer_packet_records",
        lambda *, verify_sources: [(packet_path, packet, {raw_name})],
    )

    evidence = raw_replay._active_semantic_defer_packet_evidence(
        {"raw_file": raw_name, "raw_sha256": raw_sha256},
        active_raws=frozenset({raw_name}),
    )

    assert evidence is not None
    assert evidence["authority_sha256"] == authority
    assert evidence["packet_path"] == str(packet_path)
