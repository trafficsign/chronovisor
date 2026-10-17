from __future__ import annotations

import json
import os
import plistlib
import re
import shutil
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
    "com.trafficsign.chronovisor-search-label-drain.plist": (
        "chronovisor-search-label-drain"
    ),
    "com.trafficsign.chronovisor-semantic.plist": "chronovisor-semantic-service",
}
GENERIC_SERVICES = {
    "dashboard": "chronovisor-dashboard",
    "lan-dashboard": "chronovisor-dashboard",
    "ingest-drain": "chronovisor-ingest-drain",
    "librarian-review": "chronovisor-librarian-review",
    "search-label-drain": "chronovisor-search-label-drain",
}
UVX_SERVICES = set(SERVICES) - {"com.trafficsign.chronovisor-searxng.plist"}
PERSONAL_HOME = re.compile(rb"/Users/trafficsign(?:/|$)")


def _executable(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    path.chmod(0o755)
    return path


def _git(repo: Path, *arguments: str, capture_output: bool = False) -> str:
    result = subprocess.run(
        ["git", "-C", str(repo), *arguments],
        check=True,
        capture_output=capture_output,
        text=True,
    )
    return result.stdout.strip() if capture_output else ""


def _git_commit(repo: Path, message: str) -> None:
    subprocess.run(
        [
            "git",
            "-C",
            str(repo),
            "-c",
            "user.name=Chronovisor test",
            "-c",
            "user.email=chronovisor-test@example.invalid",
            "commit",
            "-m",
            message,
        ],
        check=True,
        capture_output=True,
        text=True,
    )


def _search_label_wrapper_fixture(tmp_path: Path) -> tuple[Path, Path, dict[str, str], Path]:
    repo = tmp_path / "repo"
    scripts = repo / "scripts"
    source = repo / "src" / "chronovisor"
    scripts.mkdir(parents=True)
    source.mkdir(parents=True)
    shutil.copy2(ROOT / "scripts" / "chronovisor-runtime-env", scripts)
    shutil.copy2(ROOT / "scripts" / "chronovisor-search-label-drain", scripts)
    shutil.copy2(ROOT / "scripts" / "search_label_drain.py", scripts)
    (source / "__init__.py").write_text("", encoding="utf-8")
    (source / "core.py").write_text("fixture = True\n", encoding="utf-8")
    (repo / "pyproject.toml").write_text("[project]\nname = 'fixture'\n", encoding="utf-8")
    (repo / "uv.lock").write_text("version = 1\n", encoding="utf-8")
    _git(repo, "init", "-q")
    _git(repo, "add", ".")
    _git_commit(repo, "fixture base")
    pin = _git(repo, "rev-parse", "HEAD", capture_output=True)

    home = tmp_path / "home"
    recall = home / ".chronovisor" / "recall"
    recall.mkdir(parents=True)
    (recall / "search-label-queue.jsonl").write_text("", encoding="utf-8")
    (recall / "search-golden.jsonl").write_text("", encoding="utf-8")
    marker = tmp_path / "uvx-called"
    uvx = tmp_path / "uvx"
    uvx.write_text(
        "#!/bin/sh\n"
        'printf \'called\' > "$UVX_MARKER"\n'
        "printf '%s\\n' '{\"status\":\"stubbed\"}'\n",
        encoding="utf-8",
    )
    uvx.chmod(0o755)
    environment = os.environ.copy()
    environment.update(
        {
            "CHRONOVISOR_REPO_ROOT": str(repo),
            "CHRONOVISOR_ROOT": str(home / ".chronovisor"),
            "CHRONOVISOR_RUNTIME_SOURCE": f"git+file://{repo}@{pin}",
            "CHRONOVISOR_PYTHON": sys.executable,
            "CHRONOVISOR_UVX": str(uvx),
            "HOME": str(home),
            "UVX_MARKER": str(marker),
        }
    )
    return repo, scripts / "chronovisor-search-label-drain", environment, marker


def _run_search_label_wrapper(
    wrapper: Path, environment: dict[str, str]
) -> tuple[subprocess.CompletedProcess[str], dict[str, str]]:
    result = subprocess.run(
        [str(wrapper)],
        check=False,
        capture_output=True,
        text=True,
        env=environment,
    )
    return result, json.loads(result.stdout)


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
    if plist_name == "com.trafficsign.chronovisor-search-label-drain.plist":
        assert payload["RunAtLoad"] is False
        assert payload["StartInterval"] == 300
        assert payload["ThrottleInterval"] == 60
        assert payload["LowPriorityIO"] is True
        assert payload["ProcessType"] == "Background"


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


def test_search_label_drain_wrapper_is_bounded_and_pin_guarded() -> None:
    source = (ROOT / "scripts" / "chronovisor-search-label-drain").read_text(
        encoding="utf-8"
    )
    assert (
        'git -C "$CHRONOVISOR_REPO_ROOT" diff --quiet HEAD -- \\\n  src scripts pyproject.toml uv.lock'
        in source
    )
    assert "ls-files --others --exclude-standard -- src scripts" in source
    assert "repository_untracked_code" in source
    assert '"$EXPECTED_COMMIT:$input_path"' in source
    assert '"$HEAD:$input_path"' in source
    assert '"$EXPECTED_COMMIT:scripts/search_label_drain.py"' in source
    assert '"$SCRIPT_HASH" ] || skip drain_script_drift' in source
    assert '"--batch-size",\n    "20"' in source
    assert '"--max-elapsed-seconds",\n    "300"' in source
    assert '"--review-order",\n    "balanced"' in source
    assert "native_package_commit_drift" in source

    for name in (
        "install-reranker-service",
        "install-searxng",
        "install-semantic-service",
    ):
        source = (ROOT / "scripts" / name).read_text(encoding="utf-8")
        assert "chronovisor-render-launchd" in source
        assert "/Users/trafficsign" not in source


@pytest.mark.parametrize(
    ("scenario", "expected_reason", "runs_uvx"),
    [
        ("pinned", None, True),
        ("docs_commit", None, True),
        ("handoff_untracked", None, True),
        ("tracked_source_edit", "repository_tracked_drift", False),
        ("scripts_commit", "source_input_drift", False),
        ("pyproject_commit", "source_input_drift", False),
        ("uv_lock_commit", "source_input_drift", False),
        ("untracked_code", "repository_untracked_code", False),
    ],
)
def test_search_label_drain_wrapper_source_pin_gate(
    tmp_path: Path,
    scenario: str,
    expected_reason: str | None,
    runs_uvx: bool,
) -> None:
    repo, wrapper, environment, marker = _search_label_wrapper_fixture(tmp_path)
    if scenario == "docs_commit":
        docs = repo / "docs"
        docs.mkdir()
        (docs / "daily.md").write_text("docs-only\n", encoding="utf-8")
        _git(repo, "add", "docs/daily.md")
        _git_commit(repo, "docs-only automation update")
    elif scenario == "handoff_untracked":
        handoff = repo / "_handoff"
        handoff.mkdir()
        (handoff / "evidence.md").write_text("metadata-only\n", encoding="utf-8")
    elif scenario == "tracked_source_edit":
        (repo / "src" / "chronovisor" / "core.py").write_text(
            "fixture = False\n", encoding="utf-8"
        )
    elif scenario == "scripts_commit":
        script = repo / "scripts" / "search_label_drain.py"
        script.write_text(script.read_text(encoding="utf-8") + "\n# changed\n", encoding="utf-8")
        _git(repo, "add", "scripts/search_label_drain.py")
        _git_commit(repo, "script change")
    elif scenario == "pyproject_commit":
        pyproject = repo / "pyproject.toml"
        pyproject.write_text(
            pyproject.read_text(encoding="utf-8") + "requires-python = '>=3.14'\n",
            encoding="utf-8",
        )
        _git(repo, "add", "pyproject.toml")
        _git_commit(repo, "package metadata change")
    elif scenario == "uv_lock_commit":
        lockfile = repo / "uv.lock"
        lockfile.write_text(lockfile.read_text(encoding="utf-8") + "revision = 2\n", encoding="utf-8")
        _git(repo, "add", "uv.lock")
        _git_commit(repo, "lockfile change")
    elif scenario == "untracked_code":
        (repo / "src" / "untracked.py").write_text("untracked = True\n", encoding="utf-8")

    result, payload = _run_search_label_wrapper(wrapper, environment)
    assert result.returncode == 0
    if expected_reason is None:
        assert payload == {"status": "stubbed"}
    else:
        assert payload == {"reason": expected_reason, "status": "skipped"}
    assert marker.exists() is runs_uvx


@pytest.mark.parametrize(("service", "wrapper"), GENERIC_SERVICES.items())
def test_generic_installer_renders_every_supported_service_for_native_manager(
    tmp_path: Path,
    service: str,
    wrapper: str,
) -> None:
    home = tmp_path / "home"
    home.mkdir()
    uvx = _executable(tmp_path / "tools" / "uvx")
    manager_log = tmp_path / "manager.log"
    manager = tmp_path / "tools" / "service-manager"
    manager.write_text(
        '#!/bin/sh\nprintf "%s\\n" "$*" >> "$CHRONOVISOR_TEST_MANAGER_LOG"\n',
        encoding="utf-8",
    )
    manager.chmod(0o755)
    env = os.environ | {
        "HOME": str(home),
        "CHRONOVISOR_UVX": str(uvx),
        "CHRONOVISOR_PYTHON": sys.executable,
        "CHRONOVISOR_SERVICE_MANAGER": str(manager),
        "CHRONOVISOR_LAUNCHD_LABEL_PREFIX": "org.example.chronovisor-",
        "CHRONOVISOR_TEST_MANAGER_LOG": str(manager_log),
    }

    subprocess.run(
        [str(ROOT / "scripts" / "install-launchd-service"), service],
        check=True,
        env=env,
    )

    output = (
        home
        / ".chronovisor"
        / "runtime"
        / "launchd-sources"
        / f"com.trafficsign.chronovisor-{service}.plist"
    )
    payload = plistlib.loads(output.read_bytes())
    label = f"org.example.chronovisor-{service}"

    assert payload["Label"] == label
    assert payload["ProgramArguments"][0] == str(ROOT / "scripts" / wrapper)
    assert "@@" not in output.read_text(encoding="utf-8")
    assert manager_log.read_text(encoding="utf-8").splitlines() == [
        f"refresh {service}"
    ]


def test_generic_installer_propagates_native_manager_failure(tmp_path: Path) -> None:
    home = tmp_path / "home"
    home.mkdir()
    uvx = _executable(tmp_path / "tools" / "uvx")
    manager = _executable(tmp_path / "tools" / "service-manager")
    manager.write_text("#!/bin/sh\nexit 37\n", encoding="utf-8")
    env = os.environ | {
        "HOME": str(home),
        "CHRONOVISOR_UVX": str(uvx),
        "CHRONOVISOR_PYTHON": sys.executable,
        "CHRONOVISOR_SERVICE_MANAGER": str(manager),
        "CHRONOVISOR_LAUNCHD_LABEL_PREFIX": "org.example.chronovisor-",
    }

    with pytest.raises(subprocess.CalledProcessError):
        subprocess.run(
            [str(ROOT / "scripts" / "install-launchd-service"), "dashboard"],
            check=True,
            env=env,
        )


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
    paths = [ROOT / raw_path.decode() for raw_path in tracked if raw_path]
    offenders = [
        str(path.relative_to(ROOT))
        for path in paths
        if path.is_file() and PERSONAL_HOME.search(path.read_bytes())
    ]

    assert offenders == []
