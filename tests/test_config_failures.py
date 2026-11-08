from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import pytest

from chronovisor.core import runtime_config
from chronovisor.core.raw_store import raw_layout_mode
from chronovisor.core.runtime_config import RuntimeConfigError


def _assert_safe_config_error(exc: BaseException) -> None:
    assert getattr(exc, "category", None) == "config_error"
    assert getattr(exc, "permanent", False) is True
    assert str(exc) == "config.toml is unreadable or invalid"


@pytest.mark.parametrize(
    "loader",
    (
        runtime_config.load_toml_file,
        runtime_config.load_runtime_settings,
        runtime_config.load_hook_policy,
    ),
)
def test_missing_config_uses_reader_defaults(
    loader: Callable[..., object], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("CHRONOVISOR_RAW_LAYOUT", raising=False)
    missing = tmp_path / "config.toml"

    result = loader(missing)

    if loader is runtime_config.load_toml_file:
        assert result == {}
    elif loader is runtime_config.load_runtime_settings:
        assert result == runtime_config.RuntimeSettings()
    else:
        assert result == runtime_config.HookPolicy()

    assert raw_layout_mode(chronovisor_root=tmp_path) == "v2"


@pytest.mark.parametrize(
    "loader",
    (
        runtime_config.load_toml_file,
        runtime_config.load_runtime_settings,
        runtime_config.load_hook_policy,
    ),
)
def test_invalid_config_is_a_safe_permanent_error(
    loader: Callable[..., object], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("CHRONOVISOR_RAW_LAYOUT", raising=False)
    config = tmp_path / "config.toml"
    config.write_bytes(b'transcript_secret = "unterminated\n')

    with pytest.raises(RuntimeConfigError) as caught:
        loader(config)

    _assert_safe_config_error(caught.value)
    assert "transcript_secret" not in str(caught.value)
    with pytest.raises(RuntimeConfigError) as raw_caught:
        raw_layout_mode(chronovisor_root=tmp_path)
    _assert_safe_config_error(raw_caught.value)


def test_unreadable_config_is_a_safe_permanent_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("CHRONOVISOR_RAW_LAYOUT", raising=False)
    config = tmp_path / "config.toml"
    config.write_text('transcript_secret = "hidden"\n', encoding="utf-8")

    for reader in ("read_bytes", "read_text"):
        original = getattr(Path, reader)

        def fail(path: Path, *args: object, _original=original, **kwargs: object) -> object:
            if path == config:
                raise PermissionError("transcript_secret")
            return _original(path, *args, **kwargs)

        monkeypatch.setattr(Path, reader, fail)

    with pytest.raises(RuntimeConfigError) as caught:
        runtime_config.load_toml_file(config)

    _assert_safe_config_error(caught.value)
    assert "transcript_secret" not in str(caught.value)
    with pytest.raises(RuntimeConfigError) as raw_caught:
        raw_layout_mode(chronovisor_root=tmp_path)
    _assert_safe_config_error(raw_caught.value)


def test_explicit_raw_layout_override_does_not_read_config(
    tmp_path: Path,
) -> None:
    (tmp_path / "config.toml").write_bytes(b"not valid toml = [")

    assert raw_layout_mode("shadow", chronovisor_root=tmp_path) == "shadow"

