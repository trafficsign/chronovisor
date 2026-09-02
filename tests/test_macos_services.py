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
    assert bundled["BundleProgram"] == "Contents/Helpers/ChronovisorServiceRunner"
    assert bundled["ProgramArguments"] == [
        "Contents/Helpers/ChronovisorServiceRunner",
        "run",
        str(executable),
        "--lan",
    ]
    assert bundled["Label"] == "com.trafficsign.chronovisor-lan-dashboard.managed-v2"
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
    assert (
        module._signature_authority("Authority=Chronovisor Local Code Signing\n")
        == "Chronovisor Local Code Signing"
    )


def test_app_metadata_uses_bundled_icon(tmp_path: Path) -> None:
    module = _module()
    app = tmp_path / module.APP_NAME
    (app / "Contents").mkdir(parents=True)

    module._write_info_plist(app)

    payload = plistlib.loads((app / "Contents" / "Info.plist").read_bytes())
    assert payload["CFBundleIconFile"] == "Chronovisor.icns"
    assert module.ICON_SOURCE.is_file()
    assert module.RUNNER_SOURCE.is_file()


@pytest.mark.parametrize(
    ("service_status", "expected_events"),
    [
        (
            "enabled",
            [
                "status",
                "unregister-one",
                "sleep:3",
                "build",
                "register-one",
                "sleep:2",
                "unregister-one",
                "sleep:3",
                "register-one",
            ],
        ),
        ("not_registered", ["status", "build", "register-one"]),
    ],
)
def test_refresh_only_reenrolls_registered_service(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    service_status: str,
    expected_events: list[str],
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

    def manager(_app, command, *_args):
        events.append(command)
        stdout = (
            f'{{"services":[{{"plist":"service.plist","status":"{service_status}"}}]}}'
            if command == "status"
            else ""
        )
        return SimpleNamespace(returncode=0, stdout=stdout, stderr="")

    monkeypatch.setattr(module, "_manager", manager)
    monkeypatch.setattr(module, "_retire_legacy_service", lambda *_args: None)
    monkeypatch.setattr(module, "_verify_managed_service", lambda *_args: None)
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
    assert events == expected_events
    assert launchctl_commands == []


def test_refresh_retires_standalone_legacy_service_after_managed_registration(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _module()
    app = tmp_path / module.APP_NAME
    app.mkdir()
    legacy = (
        tmp_path
        / "Library"
        / "LaunchAgents"
        / "com.trafficsign.chronovisor-dashboard.plist"
    )
    legacy.parent.mkdir(parents=True)
    legacy.write_text("legacy", encoding="utf-8")
    legacy_v1 = legacy.with_name("com.trafficsign.chronovisor-dashboard.managed.plist")
    legacy_v1.write_text("legacy-v1", encoding="utf-8")
    launchctl_commands: list[list[str]] = []

    monkeypatch.setattr(
        module,
        "_managed_service",
        lambda _app, _suffix: (
            "com.trafficsign.chronovisor-dashboard.managed-v2.plist",
            "com.trafficsign.chronovisor-dashboard.managed-v2",
        ),
    )
    monkeypatch.setattr(module, "_service_is_registered", lambda *_args: False)
    monkeypatch.setattr(
        module,
        "_manager",
        lambda *_args: SimpleNamespace(returncode=0, stdout="", stderr=""),
    )
    monkeypatch.setattr(module, "build_app", lambda *_args: {})
    monkeypatch.setattr(module.time, "sleep", lambda _seconds: None)

    legacy_registered = {
        "com.trafficsign.chronovisor-dashboard",
        "com.trafficsign.chronovisor-dashboard.managed",
    }

    def run(command, **_kwargs):
        launchctl_commands.append(command)
        if command[1] == "kickstart":
            return SimpleNamespace(returncode=0, stdout="", stderr="")
        if command[1] == "bootout":
            legacy_registered.discard(command[-1].rsplit("/", 1)[-1])
            return SimpleNamespace(returncode=0, stdout="", stderr="")
        if command[-1].endswith(".managed-v2"):
            return SimpleNamespace(
                returncode=0,
                stdout="state = running\nruns = 1\n",
                stderr="",
            )
        return SimpleNamespace(
            returncode=0 if command[-1].rsplit("/", 1)[-1] in legacy_registered else 113,
            stdout=(
                "state = not running\n"
                if command[-1].rsplit("/", 1)[-1] in legacy_registered
                else ""
            ),
            stderr="",
        )

    monkeypatch.setattr(module.subprocess, "run", run)

    result = module.refresh_service(tmp_path, app, "dashboard")

    target = f"gui/{module.os.getuid()}/com.trafficsign.chronovisor-dashboard"
    target_v1 = f"{target}.managed"
    assert result["status"] == "ok"
    assert launchctl_commands == [
        ["/bin/launchctl", "print", target],
        ["/bin/launchctl", "print", target_v1],
        [
            "/bin/launchctl",
            "kickstart",
            f"gui/{module.os.getuid()}/"
            "com.trafficsign.chronovisor-dashboard.managed-v2",
        ],
        [
            "/bin/launchctl",
            "print",
            f"gui/{module.os.getuid()}/"
            "com.trafficsign.chronovisor-dashboard.managed-v2",
        ],
        ["/bin/launchctl", "bootout", target],
        ["/bin/launchctl", "print", target],
        ["/bin/launchctl", "bootout", target_v1],
        ["/bin/launchctl", "print", target_v1],
    ]
    assert not legacy.exists()
    assert not legacy_v1.exists()


def test_refresh_keeps_legacy_service_when_managed_launch_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _module()
    app = tmp_path / module.APP_NAME
    app.mkdir()
    legacy = (
        tmp_path
        / "Library"
        / "LaunchAgents"
        / "com.trafficsign.chronovisor-dashboard.plist"
    )
    legacy.parent.mkdir(parents=True)
    legacy.write_text("legacy", encoding="utf-8")

    monkeypatch.setattr(
        module,
        "_managed_service",
        lambda _app, _suffix: (
            "com.trafficsign.chronovisor-dashboard.managed-v2.plist",
            "com.trafficsign.chronovisor-dashboard.managed-v2",
        ),
    )
    monkeypatch.setattr(module, "_service_is_registered", lambda *_args: False)
    monkeypatch.setattr(
        module,
        "_manager",
        lambda *_args: SimpleNamespace(returncode=0, stdout="", stderr=""),
    )
    monkeypatch.setattr(module, "build_app", lambda *_args: {})
    monkeypatch.setattr(module.time, "sleep", lambda _seconds: None)

    def run(command, **_kwargs):
        if command[1:] == [
            "print",
            f"gui/{module.os.getuid()}/com.trafficsign.chronovisor-dashboard",
        ]:
            return SimpleNamespace(
                returncode=0, stdout="state = not running\n", stderr=""
            )
        if command[1] == "kickstart":
            return SimpleNamespace(returncode=0, stdout="", stderr="")
        return SimpleNamespace(
            returncode=0,
            stdout="state = not running\nruns = 1\njob state = spawn failed\n",
            stderr="",
        )

    monkeypatch.setattr(module.subprocess, "run", run)

    result = module.refresh_service(tmp_path, app, "dashboard")

    assert result["status"] == "error"
    assert result["phase"] == "verify-managed"
    assert legacy.exists()


def test_legacy_plist_is_kept_when_standalone_service_cannot_be_retired(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _module()
    legacy = (
        tmp_path
        / "Library"
        / "LaunchAgents"
        / "com.trafficsign.chronovisor-dashboard.plist"
    )
    legacy.parent.mkdir(parents=True)
    legacy.write_text("legacy", encoding="utf-8")
    monkeypatch.setattr(
        module.subprocess,
        "run",
        lambda command, **_kwargs: SimpleNamespace(
            returncode=1 if command[1] == "bootout" else 0,
            stderr="bootout failed",
        ),
    )

    with pytest.raises(RuntimeError, match="bootout failed"):
        module._retire_legacy_service(
            tmp_path,
            "dashboard",
        )

    assert legacy.exists()
