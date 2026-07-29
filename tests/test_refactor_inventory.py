from __future__ import annotations

import importlib.util
from pathlib import Path


def _inventory_module():
    path = Path(__file__).parents[1] / "scripts" / "refactor_inventory.py"
    spec = importlib.util.spec_from_file_location("refactor_inventory", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_scan_repository_reports_contract_candidates_and_ingest_seams(tmp_path: Path) -> None:
    (tmp_path / "src" / "chronovisor").mkdir(parents=True)
    (tmp_path / "tests").mkdir()
    (tmp_path / "scripts").mkdir()
    (tmp_path / "docs").mkdir()
    (tmp_path / "pyproject.toml").write_text(
        """
[project]
name = "sample"
version = "0.0.0"
[project.scripts]
sample-tool = "chronovisor.sample:main"
""".strip()
        + "\n",
        encoding="utf-8",
    )
    (tmp_path / "src" / "chronovisor" / "sample.py").write_text(
        """
import fcntl
import hashlib
import json
import os

def canonical_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()

def write_jsonl(path, value):
    with path.open("a") as handle:
        handle.write(json.dumps(value) + "\\n")

def atomic_write(path, tmp):
    os.replace(tmp, path)
""".strip()
        + "\n",
        encoding="utf-8",
    )
    (tmp_path / "tests" / "test_sample.py").write_text(
        """
from chronovisor.ingest import ingest as ingest_mod

def test_patch(monkeypatch):
    monkeypatch.setattr(ingest_mod, "PAGES_DIR", object())
    assert ingest_mod.run_ingest
""".strip()
        + "\n",
        encoding="utf-8",
    )
    script = tmp_path / "scripts" / "old_task.py"
    script.write_text("print('old')\n", encoding="utf-8")
    (tmp_path / "docs" / "operations.md").write_text(
        "Run old_task.py and sample-tool.\n", encoding="utf-8"
    )

    inventory = _inventory_module().scan_repository(tmp_path)

    assert inventory["source"]["python_modules"] == 1
    assert len(inventory["candidate_signals"]["canonical-json-hash"]) == 1
    assert len(inventory["candidate_signals"]["jsonl-append"]) == 1
    assert len(inventory["candidate_signals"]["replace-call"]) == 1
    assert inventory["ingest_seams"]["monkeypatch_targets"] == {"PAGES_DIR": 1}
    assert inventory["ingest_seams"]["attribute_references"]["run_ingest"] == 1
    assert inventory["console_entrypoints"][0]["repository_references"] == 1
    assert inventory["scripts"][0]["basename_references"] == 1
