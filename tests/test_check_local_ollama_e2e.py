from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType

import pytest

from chronovisor.core.llm_config import load_llm_config
from chronovisor.core.runtime_config import load_ingest_config


def _script() -> ModuleType:
    path = Path(__file__).parents[1] / "scripts" / "check_local_ollama_e2e.py"
    spec = importlib.util.spec_from_file_location("check_local_ollama_e2e", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_config_covers_required_local_workflow_roles(tmp_path: Path) -> None:
    module = _script()
    args = module._parse_args(
        [
            "--generation-model",
            "generation:test",
            "--embedding-model",
            "embedding:test",
            "--reranker-model",
            "reranker:test",
            "--primary-model",
            "primary:test",
            "--challenger-model",
            "challenger:test",
            "--tie-break-model",
            "tie:test",
        ]
    )
    path = tmp_path / "config.toml"
    module._write_config(path, args)
    roles = load_llm_config(path).roles
    ingest = load_ingest_config(path)

    assert path.stat().st_mode & 0o777 == 0o600
    assert set(roles) == set(module.GENERATION_ROLES) | set(module.EMBEDDING_ROLES) | {
        module.RERANK_ROLE
    }
    assert roles["classification.primary"].model == "primary:test"
    assert roles["classification.challenger"].model == "challenger:test"
    assert roles["classification.tie_break"].model == "tie:test"
    assert ingest.num_ctx == ingest.max_num_ctx == 32_768


def test_isolated_root_requires_empty_temporary_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _script()
    monkeypatch.setenv("CHRONOVISOR_ROOT", str(tmp_path))
    assert module._isolated_root() == tmp_path.resolve()

    (tmp_path / "owned").write_text("data", encoding="utf-8")
    with pytest.raises(ValueError, match="isolated_root_not_empty"):
        module._isolated_root()


def test_rejects_non_temporary_root_and_duplicate_decision_models(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _script()
    monkeypatch.setenv("CHRONOVISOR_ROOT", str(Path(__file__).parents[1]))
    with pytest.raises(ValueError, match="isolated_temporary_root_required"):
        module._isolated_root()
    with pytest.raises(SystemExit):
        module._parse_args(["--primary-model", "same", "--challenger-model", "same"])
