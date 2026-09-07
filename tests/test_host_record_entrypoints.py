from __future__ import annotations

import importlib
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"

HOST_RECORD_MODULES = (
    "chronovisor.hosts.codex_record",
    "chronovisor.hosts.claude_code_record",
    "chronovisor.hosts.pi_record",
    "chronovisor.hosts.hermes_record",
)


def _child_env() -> dict[str, str]:
    environment = os.environ.copy()
    environment["PYTHONPATH"] = os.pathsep.join(
        path for path in (str(SRC), environment.get("PYTHONPATH")) if path
    )
    return environment


def _legacy_root(path: Path) -> Path:
    root = path.resolve()
    root.mkdir()
    for name in ("index.md", "log.md", "schema.md"):
        (root / name).write_text("legacy\n", encoding="utf-8")
    return root


@pytest.mark.parametrize("module", HOST_RECORD_MODULES)
def test_host_record_module_entrypoints_show_help(module: str, tmp_path: Path) -> None:
    environment = _child_env()
    environment["CHRONOVISOR_ROOT"] = str(_legacy_root(tmp_path / "chronovisor"))
    completed = subprocess.run(
        [sys.executable, "-m", module, "--help"],
        capture_output=True,
        text=True,
        env=environment,
        timeout=30,
    )

    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.strip()


@pytest.mark.parametrize("host_name", HOST_RECORD_MODULES)
def test_host_record_imports_preserve_raw_module_identity(host_name: str) -> None:
    raw_name = host_name.replace("chronovisor.hosts.", "chronovisor.raw.")

    host = importlib.import_module(host_name)
    raw = importlib.import_module(raw_name)

    assert host is raw
    assert sys.modules[host_name] is raw


def test_codex_host_entrypoint_saves_hook_in_isolated_root(tmp_path: Path) -> None:
    root = _legacy_root(tmp_path / "chronovisor")

    session_id = "019e5ec3-42fe-7f70-9402-7ff20da6be69"
    session = (tmp_path / "session.jsonl").resolve()
    session.write_text(
        "".join(
            json.dumps(row, ensure_ascii=False) + "\n"
            for row in (
                {
                    "type": "session_meta",
                    "payload": {
                        "id": session_id,
                        "cwd": "/tmp/chronovisor-entrypoint-test",
                    },
                },
                {
                    "type": "response_item",
                    "payload": {
                        "type": "message",
                        "role": "user",
                        "content": [
                            {"type": "input_text", "text": "entrypoint regression"}
                        ],
                    },
                },
            )
        ),
        encoding="utf-8",
    )

    environment = _child_env()
    environment.update(
        {
            "CHRONOVISOR_ROOT": str(root),
            "CODEX_CHRONOVISOR_RECORD_ENABLED": "1",
        }
    )
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "chronovisor.hosts.codex_record",
            "--hook",
            "--save",
        ],
        input=json.dumps(
            {"session_id": session_id, "session_file": str(session)},
            ensure_ascii=False,
        ),
        capture_output=True,
        text=True,
        env=environment,
        timeout=30,
    )

    assert completed.returncode == 0, completed.stderr
    result = json.loads(completed.stdout)
    assert result["status"] == "saved"
    assert result["session_file"] == str(session)
    assert Path(result["save_result"]["path"]).exists()
    state = json.loads((root / "codex-save-state.json").read_text(encoding="utf-8"))
    assert state["files"][str(session)]["last_saved_line"] == 2
