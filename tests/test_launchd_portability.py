from __future__ import annotations

import json
import os
import plistlib
import re
import stat
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
LAUNCHD = ROOT / "launchd"
RENDERER = ROOT / "scripts" / "chronovisor-render-launchd"
SERVICES = {
    "com.trafficsign.chronovisor-dashboard.plist": "chronovisor-dashboard",
    "com.trafficsign.chronovisor-lan-dashboard.plist": "chronovisor-dashboard",
    "com.trafficsign.chronovisor-ingest-drain.plist": "chronovisor-ingest-drain",
    "com.trafficsign.chronovisor-librarian-review.plist": (
        "chronovisor-librarian-review"
    ),
    "com.trafficsign.chronovisor-library-evidence.plist": (
        "chronovisor-library-evidence"
    ),
    "com.trafficsign.chronovisor-reranker.plist": "chronovisor-reranker-service",
    "com.trafficsign.chronovisor-searxng.plist": "chronovisor-searxng",
    "com.trafficsign.chronovisor-semantic.plist": "chronovisor-semantic-service",
}
GENERIC_SERVICES = {
    "dashboard": "chronovisor-dashboard",
    "lan-dashboard": "chronovisor-dashboard",
    "ingest-drain": "chronovisor-ingest-drain",
    "librarian-review": "chronovisor-librarian-review",
    "library-evidence": "chronovisor-library-evidence",
}
UVX_SERVICES = set(SERVICES) - {"com.trafficsign.chronovisor-searxng.plist"}
PERSONAL_HOME = re.compile(rb"/Users/trafficsign(?:/|$)")


def _executable(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    path.chmod(0o755)
    return path


@pytest.mark.parametrize(("plist_name", "wrapper"), SERVICES.items())
def test_launchd_template_renders_portable_absolute_paths(
    tmp_path: Path,
    plist_name: str,
    wrapper: str,
) -> None:
    project_root = tmp_path / "portable-project"
    home = tmp_path / "portable-home"
    output = tmp_path / "rendered" / plist_name
    project_root.mkdir()
    home.mkdir()
    uvx = _executable(tmp_path / "tools" / "uvx")
    python = _executable(tmp_path / "tools" / "python3.14")

    subprocess.run(
        [
            sys.executable,
            str(RENDERER),
            str(LAUNCHD / plist_name),
            str(output),
            "--project-root",
            str(project_root),
            "--home",
            str(home),
            "--uvx",
            str(uvx),
            "--python",
            str(python),
        ],
        check=True,
    )

    rendered_text = output.read_text(encoding="utf-8")
    payload = plistlib.loads(output.read_bytes())
    environment = payload["EnvironmentVariables"]

    assert "@@" not in rendered_text
    assert "/Users/trafficsign" not in rendered_text
    assert payload["ProgramArguments"][0] == str(project_root / "scripts" / wrapper)
    assert payload["WorkingDirectory"] == str(project_root)
    assert environment["HOME"] == str(home)
    assert Path(payload["StandardOutPath"]).is_relative_to(home / ".chronovisor")
    assert Path(payload["StandardErrorPath"]).is_relative_to(home / ".chronovisor")
    assert stat.S_IMODE(output.stat().st_mode) == 0o644
    if plist_name in UVX_SERVICES:
        assert str(uvx.parent) in environment["PATH"].split(os.pathsep)
        assert environment["CHRONOVISOR_UVX"] == str(uvx)
        assert environment["CHRONOVISOR_PYTHON"] == str(python)
    else:
        assert "CHRONOVISOR_UVX" not in environment
        assert "CHRONOVISOR_PYTHON" not in environment
    if plist_name == "com.trafficsign.chronovisor-library-evidence.plist":
        arguments = payload["ProgramArguments"]
        assert arguments[arguments.index("--repo-root") + 1] == str(project_root)
    if plist_name == "com.trafficsign.chronovisor-dashboard.plist":
        assert payload["ProgramArguments"] == [
            str(project_root / "scripts" / wrapper),
        ]
    if plist_name == "com.trafficsign.chronovisor-lan-dashboard.plist":
        assert payload["ProgramArguments"] == [
            str(project_root / "scripts" / wrapper),
            "--lan",
        ]


def test_launchd_label_prefix_reads_runtime_config(tmp_path: Path) -> None:
    project_root = tmp_path / "portable-project"
    home = tmp_path / "portable-home"
    output = tmp_path / "rendered.plist"
    project_root.mkdir()
    (home / ".chronovisor").mkdir(parents=True)
    (home / ".chronovisor" / "config.toml").write_text(
        '[runtime]\nlaunchd_label_prefix = "org.example.chronovisor-"\n',
        encoding="utf-8",
    )
    uvx = _executable(tmp_path / "tools" / "uvx")
    python = _executable(tmp_path / "tools" / "python3.14")

    subprocess.run(
        [
            sys.executable,
            str(RENDERER),
            str(LAUNCHD / "com.trafficsign.chronovisor-dashboard.plist"),
            str(output),
            "--project-root",
            str(project_root),
            "--home",
            str(home),
            "--uvx",
            str(uvx),
            "--python",
            str(python),
        ],
        check=True,
    )

    assert plistlib.loads(output.read_bytes())["Label"] == (
        "org.example.chronovisor-dashboard"
    )


def test_launchd_sources_and_installers_have_no_personal_checkout_path() -> None:
    for path in LAUNCHD.glob("*.plist"):
        source = path.read_text(encoding="utf-8")
        assert "/Users/trafficsign" not in source
        assert "/projects/personal/chronovisor" not in source

    for name in (
        "install-reranker-service",
        "install-searxng",
        "install-semantic-service",
    ):
        source = (ROOT / "scripts" / name).read_text(encoding="utf-8")
        assert "chronovisor-render-launchd" in source
        assert "/Users/trafficsign" not in source


@pytest.mark.parametrize(("service", "wrapper"), GENERIC_SERVICES.items())
def test_generic_installer_renders_every_supported_service_without_launchctl(
    tmp_path: Path,
    service: str,
    wrapper: str,
) -> None:
    home = tmp_path / "home"
    home.mkdir()
    uvx = _executable(tmp_path / "tools" / "uvx")
    launchctl_log = tmp_path / "launchctl.log"
    launchctl = tmp_path / "tools" / "launchctl"
    launchctl.write_text(
        '#!/bin/sh\nprintf "%s\\n" "$*" >> "$CHRONOVISOR_TEST_LAUNCHCTL_LOG"\n',
        encoding="utf-8",
    )
    launchctl.chmod(0o755)
    env = os.environ | {
        "HOME": str(home),
        "CHRONOVISOR_UVX": str(uvx),
        "CHRONOVISOR_PYTHON": sys.executable,
        "CHRONOVISOR_LAUNCHCTL": str(launchctl),
        "CHRONOVISOR_LAUNCHD_LABEL_PREFIX": "org.example.chronovisor-",
        "CHRONOVISOR_TEST_LAUNCHCTL_LOG": str(launchctl_log),
    }

    subprocess.run(
        [str(ROOT / "scripts" / "install-launchd-service"), service],
        check=True,
        env=env,
    )

    output = (
        home
        / "Library"
        / "LaunchAgents"
        / f"com.trafficsign.chronovisor-{service}.plist"
    )
    payload = plistlib.loads(output.read_bytes())
    label = f"org.example.chronovisor-{service}"
    domain = f"gui/{os.getuid()}"

    assert payload["Label"] == label
    assert payload["ProgramArguments"][0] == str(ROOT / "scripts" / wrapper)
    assert "@@" not in output.read_text(encoding="utf-8")
    assert launchctl_log.read_text(encoding="utf-8").splitlines() == [
        f"bootout {domain}/{label}",
        f"bootstrap {domain} {output}",
        f"print {domain}/{label}",
        f"kickstart -k {domain}/{label}",
    ]


def test_generic_installer_retries_bootstrap_after_async_bootout(tmp_path: Path) -> None:
    home = tmp_path / "home"
    home.mkdir()
    uvx = _executable(tmp_path / "tools" / "uvx")
    launchctl_log = tmp_path / "launchctl.log"
    launchctl_state = tmp_path / "launchctl-state"
    launchctl = tmp_path / "tools" / "launchctl"
    launchctl.write_text(
        """#!/bin/sh
printf '%s\\n' \"$*\" >> \"$CHRONOVISOR_TEST_LAUNCHCTL_LOG\"
if [ \"$1\" = bootstrap ]; then
  count=0
  [ -f \"$CHRONOVISOR_TEST_LAUNCHCTL_STATE\" ] && IFS= read -r count < \"$CHRONOVISOR_TEST_LAUNCHCTL_STATE\"
  count=$((count + 1))
  printf '%s\\n' \"$count\" > \"$CHRONOVISOR_TEST_LAUNCHCTL_STATE\"
  [ \"$count\" -lt 3 ] && exit 37
fi
exit 0
""",
        encoding="utf-8",
    )
    launchctl.chmod(0o755)
    env = os.environ | {
        "HOME": str(home),
        "CHRONOVISOR_UVX": str(uvx),
        "CHRONOVISOR_PYTHON": sys.executable,
        "CHRONOVISOR_LAUNCHCTL": str(launchctl),
        "CHRONOVISOR_LAUNCHD_LABEL_PREFIX": "org.example.chronovisor-",
        "CHRONOVISOR_TEST_LAUNCHCTL_LOG": str(launchctl_log),
        "CHRONOVISOR_TEST_LAUNCHCTL_STATE": str(launchctl_state),
    }

    subprocess.run(
        [str(ROOT / "scripts" / "install-launchd-service"), "dashboard"],
        check=True,
        env=env,
    )

    bootstrap_calls = [
        line for line in launchctl_log.read_text(encoding="utf-8").splitlines()
        if line.startswith("bootstrap ")
    ]
    assert len(bootstrap_calls) == 3


def test_contract_manifest_and_production_files_have_no_personal_home() -> None:
    manifest = json.loads(
        (ROOT / "chronovisor-contract-manifest.json").read_text(encoding="utf-8")
    )
    assert "~/projects/plan/**" in manifest["durable_history"]

    tracked = subprocess.run(
        [
            "git",
            "ls-files",
            "-z",
            "--",
            ".github",
            "deploy",
            "docs",
            "launchd",
            "scripts",
            "src",
            "tools",
            "README.md",
            "chronovisor-contract-manifest.json",
            "pyproject.toml",
            "uv.lock",
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
    ).stdout.split(b"\0")
    offenders = [
        raw_path.decode()
        for raw_path in tracked
        if raw_path
        if PERSONAL_HOME.search((ROOT / raw_path.decode()).read_bytes())
    ]

    assert offenders == []
