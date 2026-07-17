from __future__ import annotations

from pathlib import Path

from llm_wiki_mcp.durable_state import (
    exclusive_text_file_lock,
    sidecar_exclusive_lock,
)


def test_exclusive_text_file_lock_creates_the_exact_lock_path(tmp_path: Path) -> None:
    lock_path = tmp_path / "state.lock"

    with exclusive_text_file_lock(lock_path):
        assert lock_path.is_file()
        assert lock_path.read_bytes() == b""


def test_sidecar_exclusive_lock_uses_suffix_plus_lock(tmp_path: Path) -> None:
    target = tmp_path / "queue.jsonl"

    with sidecar_exclusive_lock(target):
        assert (tmp_path / "queue.jsonl.lock").is_file()
