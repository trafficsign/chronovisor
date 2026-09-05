from __future__ import annotations

import json
import stat
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
RUNTIME_COMMAND = 'chronovisor_exec_uvx --python "$CHRONOVISOR_PYTHON"'
UVX_WRAPPERS = (
    "chronovisor-dashboard",
    "chronovisor-ingest-drain",
    "chronovisor-library-evidence",
    "chronovisor-librarian-review",
    "chronovisor-reranker-service",
    "chronovisor-semantic-service",
)


def _text(name: str) -> str:
    return (SCRIPTS / name).read_text(encoding="utf-8")


def test_exactly_six_production_uvx_wrappers_use_shared_runtime_resolution() -> None:
    wrappers = {
        path.name
        for path in SCRIPTS.glob("chronovisor-*")
        if RUNTIME_COMMAND in path.read_text(encoding="utf-8")
    }

    assert wrappers == set(UVX_WRAPPERS)
    for name in UVX_WRAPPERS:
        wrapper = _text(name)
        assert '. "$CHRONOVISOR_WRAPPER_DIR/chronovisor-runtime-env"' in wrapper
        assert "/Users/" not in wrapper
        assert "/opt/homebrew/bin/python" not in wrapper
        assert "3.14t" not in wrapper
        assert "--python 3.14" not in wrapper
        assert "uv python find" not in wrapper


def test_each_uvx_call_uses_resolved_python_immediately_and_before_from() -> None:
    for name in UVX_WRAPPERS:
        wrapper = _text(name)
        expected_calls = 1

        assert wrapper.count(RUNTIME_COMMAND) == expected_calls
        assert wrapper.count("--from") == expected_calls
        for segment in wrapper.split(RUNTIME_COMMAND)[1:]:
            assert segment.index("--from") >= 0


def test_shared_runtime_resolver_uses_safe_portable_defaults() -> None:
    resolver = _text("chronovisor-runtime-env")

    assert 'chronovisor_resolve_executable "${CHRONOVISOR_UVX:-}" uvx' in resolver
    assert (
        'chronovisor_resolve_executable "${CHRONOVISOR_PYTHON:-}" python3.14'
        in resolver
    )
    assert "CHRONOVISOR_ROOT=${CHRONOVISOR_ROOT:-$HOME/.chronovisor}" in resolver
    assert "chronovisor-contract-manifest.json" in resolver
    assert 'runtime.get("source")' in resolver
    assert "github.com/trafficsign/chronovisor" not in resolver
    assert "git+ssh://" not in resolver
    assert "/Users/" not in resolver


def test_shared_runtime_resolver_reads_configured_source(tmp_path: Path) -> None:
    home = tmp_path / "home"
    root = home / ".chronovisor with spaces"
    root.mkdir(parents=True)
    (root / "config.toml").write_text(
        '[runtime]\nsource = "git+https://github.com/example/fork.git"\n',
        encoding="utf-8",
    )
    uvx = tmp_path / "uvx"
    uvx.write_text(
        f"#!{sys.executable}\nimport json, os, sys\n"
        "print(json.dumps({'args': sys.argv[1:], 'constraints': os.getenv('UV_CONSTRAINT')}))\n",
        encoding="utf-8",
    )
    uvx.chmod(0o755)
    constraints = root / "runtime/dependency-constraints/chronovisor-dashboard.txt"
    for use_constraints, override in ((False, None), (True, None), (True, "custom.txt")):
        if use_constraints:
            constraints.parent.mkdir(parents=True, exist_ok=True)
            constraints.write_text("anyio==4.14.2\n", encoding="utf-8")
        environment = {
            "HOME": str(home),
            "PATH": "/usr/bin:/bin",
            "CHRONOVISOR_REPO_ROOT": str(ROOT),
            "CHRONOVISOR_ROOT": str(root),
            "CHRONOVISOR_UVX": str(uvx),
            "CHRONOVISOR_PYTHON": sys.executable,
        }
        if override is not None:
            environment["UV_CONSTRAINT"] = override
        completed = subprocess.run(
            [str(SCRIPTS / "chronovisor-dashboard")],
            check=True,
            capture_output=True,
            text=True,
            env=environment,
        )
        result = json.loads(completed.stdout)
        args = result["args"]
        assert args[args.index("--from") + 1] == "git+https://github.com/example/fork.git"
        assert args[args.index("--python") + 1] == sys.executable
        if use_constraints and override is None:
            assert args[args.index("--constraints") + 1] == str(constraints)
        else:
            assert "--constraints" not in args
        assert result["constraints"] == override


def test_library_evidence_invokes_canonical_module_directly() -> None:
    wrapper = _text("chronovisor-library-evidence")

    assert "python -m chronovisor.lab.classification_library_pilot" in wrapper
    assert "chronovisor-lab classification-library-pilot" not in wrapper


def test_shared_runtime_rejects_other_versions_and_disabled_gil(tmp_path: Path) -> None:
    python = tmp_path / "python"
    for version, gil_enabled in (((3, 13), True), ((3, 14), False)):
        python.write_text(
            f"#!{sys.executable}\nimport sys\n"
            f"sys.version_info = {version!r}\n"
            f"sys._is_gil_enabled = lambda: {gil_enabled!r}\n"
            "exec(sys.argv[2])\n",
            encoding="utf-8",
        )
        python.chmod(0o755)
        result = subprocess.run(
            [str(SCRIPTS / "chronovisor-dashboard")],
            capture_output=True,
            text=True,
            env={
                "HOME": str(tmp_path),
                "PATH": "/usr/bin:/bin",
                "CHRONOVISOR_REPO_ROOT": str(ROOT),
                "CHRONOVISOR_UVX": sys.executable,
                "CHRONOVISOR_PYTHON": str(python),
                "CHRONOVISOR_RUNTIME_SOURCE": "unused",
            },
        )
        assert result.returncode == 64
        assert "standard GIL Python 3.14 executable required" in result.stderr


def test_uvx_wrappers_remain_executable() -> None:
    executable_bits = stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH

    for name in UVX_WRAPPERS:
        assert (SCRIPTS / name).stat().st_mode & executable_bits == executable_bits


def test_searxng_uses_resolved_python_314_and_annif_keeps_supported_runtime() -> None:
    assert "chronovisor-searxng" not in UVX_WRAPPERS
    assert RUNTIME_COMMAND not in _text("chronovisor-searxng")
    installer = _text("install-searxng")
    assert '. "$SCRIPT_DIR/chronovisor-runtime-env"' in installer
    assert '--python "$CHRONOVISOR_PYTHON" "$VENV"' in installer
    assert 'readonly VENV="$RUNTIME_ROOT/.venv-3.14"' in installer
    assert 'readonly VENV="$RUNTIME_ROOT/.venv-3.14"' in _text("chronovisor-searxng")
    assert "dependency-constraints/chronovisor-searxng.txt" in installer

    annif = (ROOT / "src/chronovisor/lab/classification_annif.py").read_text(
        encoding="utf-8"
    )
    assert 'ANNIF_PYTHON = "3.13"' in annif
    assert 'ANNIF_PYTHON = "3.14"' not in annif
