"""Persisted filesystem identity without trusting mount-local device numbers."""

from __future__ import annotations

import os
import stat
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any


def birthtime_ns(value: os.stat_result) -> int | None:
    precise = getattr(value, "st_birthtime_ns", None)
    if precise is not None:
        return int(precise)
    seconds = getattr(value, "st_birthtime", None)
    return int(seconds * 1_000_000_000) if seconds is not None else None


def persisted_identity_matches(
    device: Any,
    inode: Any,
    current: os.stat_result,
    *,
    volume: os.stat_result,
    recorded_at_ns: int,
) -> bool:
    """Verify durable identity; callers keep exact stat checks during an operation.

    Darwin assigns st_dev at mount time. A changed number is admissible only
    on the current authority volume, with the same inode born before capture.
    Platforms without birth time retain the exact-device requirement.
    """
    if (
        any(isinstance(v, bool) or not isinstance(v, int) or v < 0 for v in (device, inode, recorded_at_ns))
        or inode != current.st_ino
        or current.st_dev != volume.st_dev
    ):
        return False
    born = birthtime_ns(current)
    if born is not None:
        # macOS Python exposes float birth seconds; compare at that precision.
        seconds, nanoseconds = divmod(recorded_at_ns, 1_000_000_000)
        birth_seconds = getattr(current, "st_birthtime", None)
        later = (
            born > recorded_at_ns
            if getattr(current, "st_birthtime_ns", None) is not None or birth_seconds is None
            else birth_seconds > seconds + nanoseconds * 1e-9
        )
        if later:
            return False
    return device == current.st_dev or (sys.platform == "darwin" and born is not None)


def persisted_file_state_matches(
    recorded: Any,
    current: Any,
    *,
    path: Path,
    device_key: str = "st_dev",
    inode_key: str = "st_ino",
    time_key: str = "st_mtime_ns",
    identity_only: bool = False,
) -> bool:
    """Compare a saved file state, preserving every non-device content field."""
    if recorded is None or current is None:
        return recorded is None and current is None
    if not isinstance(recorded, Mapping) or not isinstance(current, Mapping):
        return False
    if not identity_only and (
        set(recorded) != set(current)
        or any(recorded[key] != current[key] for key in recorded if key != device_key)
    ):
        return False
    captured = recorded.get(time_key)
    if isinstance(captured, bool) or not isinstance(captured, int) or captured < 0:
        return False
    # A copied file can retain an mtime older than its birth; ctime also
    # predates the saved snapshot and bounds that legitimate case.
    changed = recorded.get(time_key.replace("mtime", "ctime"), captured)
    if isinstance(changed, bool) or not isinstance(changed, int) or changed < 0:
        return False
    captured = max(captured, changed)
    try:
        observed = path.lstat()
        parent = path.parent.stat()
    except OSError:
        return False
    return (
        stat.S_ISREG(observed.st_mode)
        and (observed.st_dev, observed.st_ino)
        == (current.get(device_key), current.get(inode_key))
        and persisted_identity_matches(
            recorded.get(device_key), recorded.get(inode_key), observed,
            volume=parent, recorded_at_ns=captured,
        )
    )


def stable_file_state(value: Any) -> dict[str, int]:
    """Project the existing R4 content-bound anchor identity across mounts."""
    stable = {"size_bytes", "st_ino", "st_mtime_ns", "st_ctime_ns"}
    if not isinstance(value, Mapping) or set(value) not in (stable, stable | {"st_dev"}):
        return {}
    result = {key: value[key] for key in stable}
    if any(isinstance(item, bool) or not isinstance(item, int) or item < 0 for item in result.values()):
        return {}
    return result
