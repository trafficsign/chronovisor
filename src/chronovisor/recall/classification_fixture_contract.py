"""Published production contract shared with classification fixture tooling.

The fixture implementation and this contract are classification-owned. This
module owns the small, stable boundary needed by callers while keeping the
legacy byte formats unchanged.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from chronovisor.recall.classification import ClassificationError

DISABLED_BASELINE_SCHEMA = "chronovisor.classification-disabled-baseline.v1"
INFERENCE_DTO_SCHEMA = "chronovisor.classification-inference-dto.v1"
GOLD_FIELD_PREFIXES = ("gold_", "adjudication_")

__all__ = [
    "DISABLED_BASELINE_SCHEMA",
    "GOLD_FIELD_PREFIXES",
    "INFERENCE_DTO_SCHEMA",
    "inference_dto",
    "sha256_bytes",
    "sha256_file",
    "write_jsonl",
]


def sha256_file(path: Path) -> str:
    """Return the legacy prefixed SHA-256 digest of exact file bytes."""

    return sha256_bytes(path.read_bytes())


def sha256_bytes(value: bytes) -> str:
    """Return the established ``sha256:``-prefixed digest."""

    return f"sha256:{hashlib.sha256(value).hexdigest()}"


def _encode_jsonl(
    rows: Iterable[Mapping[str, Any]], *, sort_keys: bool
) -> bytes:
    return "".join(
        json.dumps(dict(row), ensure_ascii=False, sort_keys=sort_keys) + "\n"
        for row in rows
    ).encode("utf-8")


def _atomic_replace_bytes(path: Path, data: bytes, *, mode: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, raw_temp = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temp = Path(raw_temp)
    try:
        os.fchmod(descriptor, mode)
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp, path)
        os.chmod(path, mode)
    finally:
        temp.unlink(missing_ok=True)


def write_jsonl(
    path: Path,
    rows: Iterable[Mapping[str, Any]],
    *,
    sort_keys: bool = True,
    mode: int = 0o600,
) -> None:
    """Atomically write deterministic JSONL using the established byte format."""

    _atomic_replace_bytes(path, _encode_jsonl(rows, sort_keys=sort_keys), mode=mode)


def inference_dto(row: Mapping[str, Any]) -> dict[str, Any]:
    """Strip labels and adjudication state before model/provider execution."""

    output = {
        key: value
        for key, value in row.items()
        if not key.startswith(GOLD_FIELD_PREFIXES)
        and key not in {"fixture_split", "fixture_rank"}
    }
    output["schema"] = INFERENCE_DTO_SCHEMA
    leaked = [key for key in output if key.startswith(GOLD_FIELD_PREFIXES)]
    if leaked:
        raise ClassificationError(f"gold fields crossed inference boundary: {leaked}")
    return output
