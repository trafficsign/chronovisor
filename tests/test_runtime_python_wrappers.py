from __future__ import annotations

import re
import stat
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
PINNED_WRAPPERS = (
    "chronovisor-dashboard",
    "chronovisor-ingest-drain",
    "chronovisor-library-evidence",
    "chronovisor-librarian-review",
    "chronovisor-reranker-service",
    "chronovisor-semantic-service",
)


def _text(name: str) -> str:
    return (SCRIPTS / name).read_text(encoding="utf-8")


def _uvx_segments(wrapper: str) -> list[str]:
    starts = [match.start() for match in re.finditer(r"\buvx\b", wrapper)]
    return [
        wrapper[start : starts[index + 1] if index + 1 < len(starts) else None]
        for index, start in enumerate(starts)
    ]


def test_exactly_six_production_uvx_wrappers_pin_standard_python_314() -> None:
    pinned = {
        path.name
        for path in SCRIPTS.glob("chronovisor-*")
        if "--python 3.14" in path.read_text(encoding="utf-8")
    }

    assert pinned == set(PINNED_WRAPPERS)
    assert all("3.14t" not in _text(name) for name in PINNED_WRAPPERS)


def test_each_uvx_call_pins_python_immediately_and_before_from() -> None:
    for name in PINNED_WRAPPERS:
        wrapper = _text(name)
        segments = _uvx_segments(wrapper)
        expected_calls = 3 if name == "chronovisor-librarian-review" else 1

        assert len(segments) == expected_calls
        assert wrapper.count("--python 3.14") == expected_calls
        assert wrapper.count("--from") == expected_calls
        for segment in segments:
            assert segment.startswith("uvx --python 3.14")
            assert segment.index("--python 3.14") < segment.index("--from")


def test_pinned_wrappers_remain_executable() -> None:
    executable_bits = stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH

    for name in PINNED_WRAPPERS:
        assert (SCRIPTS / name).stat().st_mode & executable_bits == executable_bits


def test_searxng_and_annif_remain_on_their_intentional_python_313_paths() -> None:
    assert "chronovisor-searxng" not in PINNED_WRAPPERS
    assert "--python 3.14" not in _text("chronovisor-searxng")
    assert '--python 3.13 "$VENV"' in _text("install-searxng")

    annif = (ROOT / "src/chronovisor/lab/classification_annif.py").read_text(
        encoding="utf-8"
    )
    assert 'ANNIF_PYTHON = "3.13"' in annif
    assert 'ANNIF_PYTHON = "3.14"' not in annif
