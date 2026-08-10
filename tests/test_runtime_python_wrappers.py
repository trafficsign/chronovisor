from __future__ import annotations

import stat
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
RUNTIME_COMMAND = '"$CHRONOVISOR_UVX" --python "$CHRONOVISOR_PYTHON"'
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
        expected_calls = 3 if name == "chronovisor-librarian-review" else 1

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
    assert "git+https://github.com/trafficsign/chronovisor.git" in resolver
    assert "git+ssh://" not in resolver
    assert "/Users/" not in resolver


def test_library_evidence_invokes_canonical_module_directly() -> None:
    wrapper = _text("chronovisor-library-evidence")

    assert "python -m chronovisor.lab.classification_library_pilot" in wrapper
    assert "chronovisor-lab classification-library-pilot" not in wrapper


def test_uvx_wrappers_remain_executable() -> None:
    executable_bits = stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH

    for name in UVX_WRAPPERS:
        assert (SCRIPTS / name).stat().st_mode & executable_bits == executable_bits


def test_searxng_and_annif_remain_on_their_intentional_python_313_paths() -> None:
    assert "chronovisor-searxng" not in UVX_WRAPPERS
    assert RUNTIME_COMMAND not in _text("chronovisor-searxng")
    assert '--python 3.13 "$VENV"' in _text("install-searxng")

    annif = (ROOT / "src/chronovisor/lab/classification_annif.py").read_text(
        encoding="utf-8"
    )
    assert 'ANNIF_PYTHON = "3.13"' in annif
    assert 'ANNIF_PYTHON = "3.14"' not in annif
