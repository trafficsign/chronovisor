from __future__ import annotations

import os
from pathlib import Path
from types import SimpleNamespace
from typing import cast

import pytest

from chronovisor.recall import recall_file_identity as file_identity


@pytest.mark.parametrize(
    ("device", "inode", "current_device", "parent_device", "birth", "accepted"),
    [
        (11, 42, 13, 13, 100, True),
        (11, 42, 13, 13, 201, False),
        (11, 43, 13, 13, 100, False),
        (11, 42, 13, 14, 100, False),
        (13, 42, 13, 13, 201, False),
        (11, 42, 13, 13, None, False),
        (13, 42, 13, 13, None, True),
        (True, 42, 13, 13, 100, False),
    ],
)
def test_persisted_identity_requires_volume_inode_and_birth(
    monkeypatch: pytest.MonkeyPatch, device: int, inode: int,
    current_device: int, parent_device: int, birth: int | None, accepted: bool,
) -> None:
    monkeypatch.setattr(file_identity.sys, "platform", "darwin")
    current = SimpleNamespace(st_dev=current_device, st_ino=42, st_birthtime_ns=birth)
    volume = SimpleNamespace(st_dev=parent_device)
    assert file_identity.persisted_identity_matches(
        device, inode, cast(os.stat_result, current),
        volume=cast(os.stat_result, volume), recorded_at_ns=200,
    ) is accepted


def test_device_change_without_darwin_is_not_assumed_to_be_a_remount(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(file_identity.sys, "platform", "linux")
    current = cast(os.stat_result, SimpleNamespace(st_dev=13, st_ino=42, st_birthtime_ns=100))
    assert not file_identity.persisted_identity_matches(
        11, 42, current, volume=current, recorded_at_ns=200,
    )


def test_float_birthtime_uses_the_stat_result_precision(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(file_identity.sys, "platform", "darwin")
    captured = 1792410811834966654
    seconds, nanoseconds = divmod(captured, 1_000_000_000)
    current = cast(os.stat_result, SimpleNamespace(
        st_dev=13, st_ino=42, st_birthtime=seconds + nanoseconds * 1e-9,
    ))
    assert file_identity.persisted_identity_matches(
        11, 42, current, volume=current, recorded_at_ns=captured,
    )
    assert not file_identity.persisted_identity_matches(
        11, 42, current, volume=current, recorded_at_ns=captured - 1_000_000,
    )


def test_preserved_mtime_is_bounded_by_captured_ctime(tmp_path: Path) -> None:
    path = tmp_path / "copied-file"
    path.write_bytes(b"unchanged")
    os.utime(path, ns=(1_000_000_000, 1_000_000_000))
    observed = path.stat()
    state = {key: getattr(observed, key) for key in (
        "st_dev", "st_ino", "st_mtime_ns", "st_ctime_ns", "st_size",
    )}
    assert file_identity.persisted_file_state_matches(state, state, path=path)
