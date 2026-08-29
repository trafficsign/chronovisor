from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
INSTALLER = ROOT / "scripts" / "install-omlx-service"
RUNTIME_ENV = ROOT / "scripts" / "omlx-runtime-env"
START = ROOT / "scripts" / "start-omlx-service"
STOP = ROOT / "scripts" / "stop-omlx-service"
APP_CLI = Path("/Applications/oMLX.app/Contents/MacOS/omlx-cli")
MODEL_ID = "Qwen3.8-Flash-Next-oQ4e-mtp"


def _install_env(home: Path) -> dict[str, str]:
    return os.environ | {
        "HOME": str(home),
        # A lifecycle binary is injected for staging-only tests. Production
        # resolves ~/.omlx/bin/omlx and the app bundle, never PATH/Homebrew.
        "OMLX_CLI": "/usr/bin/true",
        "CHRONOVISOR_PYTHON": "python3",
    }


def test_repo_uses_official_app_lifecycle_only() -> None:
    assert not (ROOT / "config/omlx/com.trafficsign.omlx-qwen.plist").exists()
    assert not (ROOT / "scripts/omlx-render-launchd").exists()
    for path in (RUNTIME_ENV, INSTALLER, START, STOP):
        text = path.read_text(encoding="utf-8")
        assert "launchctl" not in text
        assert "omlx-render-launchd" not in text
    assert "--launch-agent" in INSTALLER.read_text(encoding="utf-8")


def test_installer_never_overwrites_existing_settings_or_auth(tmp_path: Path) -> None:
    home = tmp_path / "home"
    base = home / ".omlx"
    model_dir = base / "models"
    home.mkdir()
    args = [
        str(INSTALLER),
        "--base-path",
        str(base),
        "--model-dir",
        str(model_dir),
    ]
    subprocess.run(args, check=True, env=_install_env(home))

    settings = base / "settings.json"
    model_settings = base / "model_settings.json"
    settings_payload = {
        "version": "1.0",
        "auth": {"api_key": "fixture-auth-value", "skip_api_key_verification": False},
        "server": {"port": 19999},
    }
    model_payload = {
        "version": 1,
        "models": {"Ornith-1.5-9B-MLX-4bit": {"is_pinned": True}},
    }
    settings.write_text(json.dumps(settings_payload), encoding="utf-8")
    model_settings.write_text(json.dumps(model_payload), encoding="utf-8")
    settings_before = settings.read_bytes()
    model_settings_before = model_settings.read_bytes()

    subprocess.run([*args, "--force"], check=True, env=_install_env(home))

    assert settings.read_bytes() == settings_before
    assert model_settings.read_bytes() == model_settings_before


def test_installer_dry_run_does_not_create_operator_paths(tmp_path: Path) -> None:
    home = tmp_path / "home"
    base = home / ".omlx"
    result = subprocess.run(
        [str(INSTALLER), "--base-path", str(base), "--dry-run"],
        check=True,
        capture_output=True,
        text=True,
        env=_install_env(home),
    )
    assert not base.exists()
    assert "18125" in result.stdout
    assert "omlx.app" in result.stdout.lower()


def test_examples_use_model_id_discovered_from_stable_nested_layout() -> None:
    config_text = (ROOT / "config.toml.example").read_text(encoding="utf-8")
    model_settings = json.loads(
        (ROOT / "config/omlx/model_settings.json.example").read_text(encoding="utf-8")
    )
    assert MODEL_ID in config_text
    assert MODEL_ID in model_settings["models"]
    assert "Jundot/Qwen3.8-Flash-Next-oQ4e-mtp" not in config_text


def test_settings_example_keeps_profile_switches_explicit() -> None:
    settings = json.loads(
        (ROOT / "config/omlx/settings.json.example").read_text(encoding="utf-8")
    )
    memory = settings["memory"]
    cache = settings["cache"]
    sampling = settings["sampling"]
    assert settings["server"]["host"] == "127.0.0.1"
    assert settings["server"]["port"] == 18125
    assert settings["server"]["auto_start_on_launch"] is True
    assert memory["memory_guard_tier"] == "custom"
    assert memory["memory_guard_custom_ceiling_gb"] == 104.0
    assert cache["gdn_snapshot_storage"] == "embedded"
    assert cache["gdn_ssd_split_enabled"] is False
    assert cache["ssd_cache_max_size"] == "5GB"
    assert cache["hot_cache_max_size"] == "2GB"
    assert cache["hot_cache_write_through"] is False
    assert cache["ane_compile_cache"] is True
    assert sampling["max_context_window"] == 114688
    assert sampling["max_context_window_policy"] == 114688


def test_runtime_env_prefers_official_app_over_path_or_homebrew(tmp_path: Path) -> None:
    if not APP_CLI.is_file() or not os.access(APP_CLI, os.X_OK):
        return
    env = os.environ.copy()
    env.pop("OMLX_CLI", None)
    env["HOME"] = str(tmp_path)
    # A misleading PATH entry must not win over the app bundle.
    fake_homebrew = tmp_path / "opt/homebrew/bin/omlx"
    fake_homebrew.parent.mkdir(parents=True)
    fake_homebrew.write_text("#!/bin/sh\nexit 99\n", encoding="utf-8")
    fake_homebrew.chmod(0o755)
    env["PATH"] = str(fake_homebrew.parent) + ":/usr/bin:/bin"
    wrapper = tmp_path / "print-omlx-cli.sh"
    wrapper.write_text(
        f'#!/bin/sh\n. "{RUNTIME_ENV}"\nprintf "%s\\n%s" "$OMLX_CLI" "$OMLX_APP_CLI"\n',
        encoding="utf-8",
    )
    wrapper.chmod(0o755)
    selected = subprocess.run(
        [str(wrapper)],
        check=True,
        capture_output=True,
        text=True,
        env=env,
    ).stdout.splitlines()
    assert selected == [str(APP_CLI), str(APP_CLI)]


def test_runtime_env_prefers_user_omlx_wrapper_when_present(tmp_path: Path) -> None:
    if not APP_CLI.is_file() or not os.access(APP_CLI, os.X_OK):
        return
    home = tmp_path / "home"
    user_cli = home / ".omlx/bin/omlx"
    user_cli.parent.mkdir(parents=True)
    user_cli.write_text("#!/bin/sh\nexec /usr/bin/true\n", encoding="utf-8")
    user_cli.chmod(0o755)
    wrapper = tmp_path / "print-omlx-cli.sh"
    wrapper.write_text(
        f'#!/bin/sh\n. "{RUNTIME_ENV}"\nprintf "%s" "$OMLX_CLI"\n',
        encoding="utf-8",
    )
    wrapper.chmod(0o755)
    selected = subprocess.run(
        [str(wrapper)],
        check=True,
        capture_output=True,
        text=True,
        env=os.environ | {"HOME": str(home), "PATH": "/usr/bin:/bin"},
    ).stdout
    assert selected == str(user_cli)


def test_lifecycle_helpers_delegate_to_official_cli(tmp_path: Path) -> None:
    fake_cli = tmp_path / "omlx-cli"
    log = tmp_path / "argv.log"
    fake_cli.write_text(
        "#!/bin/sh\n"
        "if [ \"$1\" = \"--version\" ]; then echo 0.6.3; exit 0; fi\n"
        "printf '%s\\n' \"$*\" > \"$OMLX_TEST_ARGV_LOG\"\n",
        encoding="utf-8",
    )
    fake_cli.chmod(0o755)
    env = os.environ | {
        "HOME": str(tmp_path / "home"),
        "OMLX_CLI": str(fake_cli),
        "OMLX_TEST_ARGV_LOG": str(log),
    }
    subprocess.run([str(START), "--timeout", "0.5", "--no-wait"], check=True, env=env)
    assert log.read_text(encoding="utf-8").strip() == "start --timeout 0.5 --no-wait"
    subprocess.run([str(STOP), "--timeout", "2"], check=True, env=env)
    assert log.read_text(encoding="utf-8").strip() == "stop --timeout 2"


def test_official_cli_parser_smoke() -> None:
    if not APP_CLI.is_file() or not os.access(APP_CLI, os.X_OK):
        return
    version = subprocess.run(
        [str(APP_CLI), "--version"], check=True, capture_output=True, text=True
    )
    assert version.stdout.strip() == "0.6.3"
    for command in ("start", "stop", "restart"):
        result = subprocess.run(
            [str(APP_CLI), command, "--help"],
            check=True,
            capture_output=True,
            text=True,
        )
        assert "--timeout" in result.stdout
        assert f"usage: cli.py {command}" in result.stdout
        assert "usage: cli.py serve" not in result.stdout
