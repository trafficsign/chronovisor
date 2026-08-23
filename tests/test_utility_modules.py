from __future__ import annotations

import hashlib
from datetime import UTC, datetime, timedelta
from pathlib import Path

from chronovisor.core.hashutil import (
    is_sha256,
    sha256_bytes,
    sha256_file,
    sha256_prefixed_bytes,
    sha256_prefixed_text,
    sha256_text,
)
from chronovisor.core.jsonl import encode_jsonl, read_jsonl, write_jsonl
from chronovisor.core.jsonl_write import (
    append_jsonl_durable,
    atomic_replace_bytes,
    atomic_write_json_payload,
    write_jsonl_atomic,
)
from chronovisor.core.timeutil import (
    ensure_utc,
    iso_milliseconds,
    iso_seconds,
    utc_iso_milliseconds,
    utc_iso_seconds,
)


def test_time_helpers_preserve_precision_and_normalize_utc() -> None:
    local = datetime(2026, 7, 29, 12, 34, 56, 789123, tzinfo=UTC)
    naive = local.replace(tzinfo=None)

    assert iso_seconds(local) == "2026-07-29T12:34:56+00:00"
    assert iso_milliseconds(local) == "2026-07-29T12:34:56.789+00:00"
    assert ensure_utc(naive) == local
    assert ensure_utc(local + timedelta(hours=1)).tzinfo is UTC
    assert ".000+00:00" in utc_iso_milliseconds() or "+" in utc_iso_milliseconds()
    assert utc_iso_seconds().endswith("+00:00")


def test_hash_helpers_keep_prefixed_and_unprefixed_contracts(tmp_path: Path) -> None:
    payload = b"chronovisor"
    expected = hashlib.sha256(payload).hexdigest()
    path = tmp_path / "payload"
    path.write_bytes(payload)

    assert sha256_bytes(payload) == expected
    assert sha256_text("chronovisor") == expected
    assert sha256_file(path) == expected
    assert sha256_prefixed_bytes(payload) == f"sha256:{expected}"
    assert sha256_prefixed_text("chronovisor") == f"sha256:{expected}"
    assert is_sha256(expected)
    assert not is_sha256(f"sha256:{expected}")


def test_jsonl_helpers_preserve_exact_lf_bytes(tmp_path: Path) -> None:
    rows = [{"z": 1, "a": "日本語"}, {"z": 2}]
    expected = '{"a": "日本語", "z": 1}\n{"z": 2}\n'
    direct = tmp_path / "direct.jsonl"
    atomic = tmp_path / "atomic.jsonl"

    assert encode_jsonl(rows) == expected
    write_jsonl(direct, rows)
    write_jsonl_atomic(atomic, rows)

    assert direct.read_bytes() == expected.encode("utf-8")
    assert atomic.read_bytes() == direct.read_bytes()
    assert read_jsonl(atomic) == [{"a": "日本語", "z": 1}, {"z": 2}]


def test_jsonl_tail_reads_only_bounded_physical_lines(tmp_path: Path) -> None:
    path = tmp_path / "tail.jsonl"
    path.write_bytes(
        b'{"ignored":true}\n'
        + '{"text":"before\u2028after"}\n'.encode()
        + b'{"value":3}\r\n'
        + b"torn"
    )

    assert read_jsonl(path, limit=3) == [
        {"text": "before\u2028after"},
        {"value": 3},
    ]


def test_atomic_replace_bytes_sets_private_mode(tmp_path: Path) -> None:
    path = tmp_path / "artifact"

    atomic_replace_bytes(path, b"fixed")

    assert path.read_bytes() == b"fixed"
    assert path.stat().st_mode & 0o777 == 0o600


def test_atomic_write_json_payload_preserves_exact_contract(tmp_path: Path) -> None:
    path = tmp_path / "payload.json"

    atomic_write_json_payload(path, {"z": 1, "a": "日本語"})

    assert path.read_bytes() == '{"a": "日本語", "z": 1}\n'.encode()
    assert path.stat().st_mode & 0o777 == 0o600
    assert not list(tmp_path.glob(f".{path.name}.*.tmp"))


def test_append_jsonl_durable_creates_private_file(tmp_path: Path) -> None:
    path = tmp_path / "append.jsonl"

    append_jsonl_durable(path, [{"event": "created"}])

    assert path.read_text(encoding="utf-8") == '{"event": "created"}\n'
    assert path.stat().st_mode & 0o777 == 0o600


def test_append_jsonl_durable_repairs_existing_file_mode(tmp_path: Path) -> None:
    path = tmp_path / "append-existing.jsonl"
    path.write_text('{"event": "old"}\n', encoding="utf-8")
    path.chmod(0o644)

    append_jsonl_durable(path, [{"event": "new"}])

    assert path.read_text(encoding="utf-8") == (
        '{"event": "old"}\n{"event": "new"}\n'
    )
    assert path.stat().st_mode & 0o777 == 0o600
