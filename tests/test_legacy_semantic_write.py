from __future__ import annotations

from pathlib import Path

import pytest

from chronovisor.raw.legacy_semantic_write import (
    LegacySemanticMutationDisabled,
    block_legacy_semantic_mutation,
)


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_guard_fails_closed_with_frontier_replacement() -> None:
    with pytest.raises(LegacySemanticMutationDisabled) as excinfo:
        block_legacy_semantic_mutation(
            tool="legacy.py",
            replacement="chronovisor-sleep",
        )

    message = str(excinfo.value)
    assert "semantic writes are disabled" in message
    assert "frontier-model final decision" in message
    assert "chronovisor-sleep" in message


@pytest.mark.parametrize(
    "relative_path",
    [
        "scripts/cleanup_garbage.py",
        "scripts/tag_backfill_apply.py",
        "scripts/fix_broken_links.py",
        "scripts/backfill_links.py",
        "scripts/ollama_backfill.py",
        "scripts/backfill_recall_questions.py",
        "scripts/migrate_folders.py",
        "scripts/migrate_basic_memory.py",
        "scripts/tag_backfill_retry.py",
    ],
)
def test_obsolete_semantic_writers_import_and_call_guard(relative_path: str) -> None:
    source = (REPO_ROOT / relative_path).read_text(encoding="utf-8")

    assert "block_legacy_semantic_mutation" in source
    assert "replacement=" in source
