from __future__ import annotations

from importlib.machinery import SourceFileLoader
from importlib.util import module_from_spec, spec_from_loader
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "chronovisor-macos-services"


def _module():
    loader = SourceFileLoader("chronovisor_macos_services", str(SCRIPT))
    spec = spec_from_loader(loader.name, loader)
    assert spec is not None and spec.loader is not None
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_bundled_plist_keeps_service_boundary_and_uses_one_app(tmp_path: Path) -> None:
    module = _module()
    executable = tmp_path / "chronovisor-dashboard"
    executable.write_text("#!/bin/sh\n", encoding="utf-8")
    executable.chmod(0o755)
    payload = {
        "Label": "com.trafficsign.chronovisor-lan-dashboard",
        "ProgramArguments": [str(executable), "--lan"],
        "KeepAlive": True,
        "EnvironmentVariables": {"HOME": str(tmp_path)},
    }

    bundled = module.bundled_plist(payload, "lan-dashboard")

    assert "Program" not in bundled
    assert bundled["BundleProgram"] == "Contents/MacOS/Chronovisor"
    assert bundled["ProgramArguments"] == [
        "Contents/MacOS/Chronovisor",
        "run",
        str(executable),
        "--lan",
    ]
    assert bundled["Label"] == (
        "com.trafficsign.chronovisor-lan-dashboard.managed"
    )
    assert "AssociatedBundleIdentifiers" not in bundled
    assert bundled["KeepAlive"] is True

    assert module.bundled_plist(bundled, "lan-dashboard") == bundled


def test_bundled_plist_rejects_wrong_label_and_relative_program(tmp_path: Path) -> None:
    module = _module()
    with pytest.raises(ValueError, match="unexpected LaunchAgent label"):
        module.bundled_plist(
            {
                "Label": "com.example.wrong",
                "ProgramArguments": [str(tmp_path / "missing")],
            },
            "dashboard",
        )
    with pytest.raises(ValueError, match="service executable is unavailable"):
        module.bundled_plist(
            {
                "Label": "com.trafficsign.chronovisor-dashboard",
                "ProgramArguments": ["relative-command"],
            },
            "dashboard",
        )


def test_retired_experiments_are_not_bundled() -> None:
    module = _module()
    assert set(module.RETIRED_SUFFIXES).isdisjoint(module.SERVICE_SUFFIXES)
    assert set(module.RETIRED_SUFFIXES) == {
        "librarian-rollout",
        "library-evidence",
        "soak",
    }
