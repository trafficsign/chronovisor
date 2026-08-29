from __future__ import annotations

import plistlib
from importlib.machinery import SourceFileLoader
from importlib.util import module_from_spec, spec_from_loader
from pathlib import Path
from types import SimpleNamespace

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


def test_signing_requires_a_real_identity(monkeypatch: pytest.MonkeyPatch) -> None:
    module = _module()
    monkeypatch.delenv(module.SIGNING_IDENTITY_ENV, raising=False)

    assert module._signing_identity() == "Chronovisor Local Code Signing"
    assert module._signing_identity("Developer ID Application: Example") == (
        "Developer ID Application: Example"
    )
    with pytest.raises(ValueError, match="non-ad-hoc"):
        module._signing_identity("-")
    with pytest.raises(RuntimeError, match="signed ad-hoc"):
        module._signature_authority("Signature=adhoc")
    assert module._signature_authority(
        "Authority=Chronovisor Local Code Signing\n"
    ) == "Chronovisor Local Code Signing"


def test_app_metadata_uses_bundled_icon(tmp_path: Path) -> None:
    module = _module()
    app = tmp_path / module.APP_NAME
    (app / "Contents").mkdir(parents=True)

    module._write_info_plist(app)

    payload = plistlib.loads((app / "Contents" / "Info.plist").read_bytes())
    assert payload["CFBundleIconFile"] == "Chronovisor.icns"
    assert module.ICON_SOURCE.is_file()


def test_refresh_reenrolls_after_replacing_signed_bundle(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _module()
    app = tmp_path / module.APP_NAME
    app.mkdir()
    events: list[str] = []
    launchctl_commands: list[list[str]] = []

    monkeypatch.setattr(
        module,
        "_managed_service",
        lambda _app, _suffix: ("service.plist", "service.label"),
    )
    monkeypatch.setattr(
        module,
        "_manager",
        lambda _app, command, *_args: (
            events.append(command)
            or SimpleNamespace(returncode=0, stdout="", stderr="")
        ),
    )
    monkeypatch.setattr(
        module,
        "build_app",
        lambda *_args: events.append("build") or {},
    )
    monkeypatch.setattr(
        module.time,
        "sleep",
        lambda seconds: events.append(f"sleep:{seconds}"),
    )
    monkeypatch.setattr(
        module.subprocess,
        "run",
        lambda command, **_kwargs: (
            launchctl_commands.append(command)
            or SimpleNamespace(returncode=0, stderr="")
        ),
    )

    result = module.refresh_service(tmp_path, app, "dashboard")

    assert result["status"] == "ok"
    assert events == [
        "unregister-one",
        "sleep:3",
        "build",
        "register-one",
        "sleep:2",
        "unregister-one",
        "sleep:3",
        "register-one",
    ]
    assert launchctl_commands == []
