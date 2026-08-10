from __future__ import annotations

import hashlib
from contextlib import contextmanager
from pathlib import Path

import pytest

from chronovisor.core import page_mutation


@pytest.fixture(autouse=True)
def _valid_okf_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    root = tmp_path / "wiki-root"
    root.mkdir()
    for name in ("index.md", "log.md", "schema.md"):
        (root / name).write_text("legacy\n", encoding="utf-8")
    monkeypatch.setattr(page_mutation, "CHRONOVISOR_ROOT", root)


def _page(path: Path, *, title: str, body: str) -> None:
    path.write_text(
        f"---\ntitle: {title}\nupdated: 2026-07-01\nstatus: stable\ntype: knowledge\n---\n{body}",
        encoding="utf-8",
    )


def _patch_pages(monkeypatch, pages: Path) -> None:
    monkeypatch.setattr(page_mutation, "PAGES_DIR", pages)
    monkeypatch.setattr(
        page_mutation,
        "CHRONOVISOR_MUTATION_LOCK",
        pages.parent / "wiki-mutation.lock",
    )
    monkeypatch.setattr(
        page_mutation,
        "find_page",
        lambda page_id: (
            (pages / f"{page_id}.md") if (pages / f"{page_id}.md").exists() else None
        ),
    )


@pytest.mark.parametrize(
    ("lock_name", "path_name"),
    [
        ("chronovisor_mutation_lock", "wiki.lock"),
        ("decision_authority_lock", "authority.lock"),
    ],
)
def test_file_locks_reuse_same_thread_outer_lease(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    lock_name: str,
    path_name: str,
) -> None:
    calls: list[int] = []
    with page_mutation.okf_runtime_operation(page_mutation.CHRONOVISOR_ROOT):
        pass
    monkeypatch.setattr(
        page_mutation.fcntl,
        "flock",
        lambda _fd, operation: calls.append(operation),
    )
    lock = getattr(page_mutation, lock_name)

    with lock(tmp_path / path_name), lock(tmp_path / path_name):
        pass

    expected = [page_mutation.fcntl.LOCK_EX, page_mutation.fcntl.LOCK_UN]
    if lock_name == "chronovisor_mutation_lock":
        expected = [
            page_mutation.fcntl.LOCK_SH,
            page_mutation.fcntl.LOCK_SH,
            page_mutation.fcntl.LOCK_SH,
            page_mutation.fcntl.LOCK_EX,
            page_mutation.fcntl.LOCK_UN,
            page_mutation.fcntl.LOCK_UN,
            page_mutation.fcntl.LOCK_UN,
            page_mutation.fcntl.LOCK_UN,
        ]
    assert calls == expected


def test_prepare_and_apply_exact_replacement_adds_marker(
    tmp_path: Path, monkeypatch
) -> None:
    pages = tmp_path / "pages"
    pages.mkdir()
    path = pages / "memory.md"
    _page(path, title="Memory", body="# Memory\n\nInstalled RAM is 16GB.\n")
    _patch_pages(monkeypatch, pages)

    prepared = page_mutation.prepare_page_mutation(
        "memory",
        [{"old_text": "Installed RAM is 16GB.", "new_text": "Installed RAM is 32GB."}],
        correction_id="corr-1",
        summary="The machine has 32GB RAM.",
        recall_questions=["How much RAM is installed?"],
    )

    assert path.read_text(encoding="utf-8").endswith("Installed RAM is 16GB.\n")
    result = page_mutation.apply_prepared_mutations([prepared])

    assert result["status"] == "applied"
    written = path.read_text(encoding="utf-8")
    assert "Installed RAM is 16GB." not in written
    assert "Installed RAM is 32GB." in written
    assert "applied_corrections:\n- corr-1" in written
    assert "summary: The machine has 32GB RAM." in written

    registry = page_mutation.correction_constraints_file()
    assert registry.exists()
    constrained, applied = page_mutation.enforce_correction_constraints(
        "memory",
        written,
        written + "Legacy replay: Installed RAM is 16GB.\n",
    )
    assert "Installed RAM is 16GB." not in constrained
    assert constrained.count("Installed RAM is 32GB.") == 2
    assert applied[0]["correction_id"] == "corr-1"


def test_allowlisted_memory_system_page_can_be_corrected(
    tmp_path: Path,
    monkeypatch,
) -> None:
    pages = tmp_path / "pages"
    system = tmp_path / "system"
    pages.mkdir()
    system.mkdir()
    path = system / "user-profile.md"
    _page(path, title="User Profile", body="Preferred editor is Vim.\n")
    _patch_pages(monkeypatch, pages)
    monkeypatch.setattr(page_mutation, "SYSTEM_DIR", system)

    prepared = page_mutation.prepare_page_mutation(
        "user-profile",
        [
            {
                "old_text": "Preferred editor is Vim.",
                "new_text": "Preferred editor is Helix.",
            }
        ],
        correction_id="corr-system-memory",
    )
    result = page_mutation.apply_prepared_mutations([prepared])

    assert result["status"] == "applied"
    assert "Preferred editor is Helix." in path.read_text(encoding="utf-8")


def test_operational_system_page_remains_outside_correction_boundary(
    tmp_path: Path,
    monkeypatch,
) -> None:
    pages = tmp_path / "pages"
    system = tmp_path / "system"
    pages.mkdir()
    system.mkdir()
    _page(system / "claude-code.md", title="Claude Code", body="Old setting.\n")
    _patch_pages(monkeypatch, pages)
    monkeypatch.setattr(page_mutation, "SYSTEM_DIR", system)

    with pytest.raises(page_mutation.PageMutationError, match="page not found"):
        page_mutation.prepare_page_mutation(
            "claude-code",
            [{"old_text": "Old setting.", "new_text": "New setting."}],
            correction_id="corr-forbidden-system",
        )


def test_constraint_registry_row_is_inert_without_applied_marker(
    tmp_path: Path,
    monkeypatch,
) -> None:
    pages = tmp_path / "pages"
    pages.mkdir()
    path = pages / "memory.md"
    _page(path, title="Memory", body="Old fact.\n")
    _patch_pages(monkeypatch, pages)
    registry = page_mutation.correction_constraints_file()
    registry.parent.mkdir(parents=True, exist_ok=True)
    registry.write_text(
        '{"kind":"content_correction_constraint","correction_id":"pending",'
        '"page_id":"memory","action":"replace","old_text":"Old fact.",'
        '"new_text":"New fact."}\n',
        encoding="utf-8",
    )
    current = path.read_text(encoding="utf-8")

    constrained, applied = page_mutation.enforce_correction_constraints(
        "memory", current, current + "Old fact.\n"
    )

    assert constrained.endswith("Old fact.\n")
    assert applied == []


def test_new_correction_never_evicts_older_constraint_markers(
    tmp_path: Path,
    monkeypatch,
) -> None:
    pages = tmp_path / "pages"
    pages.mkdir()
    path = pages / "memory.md"
    prior = ", ".join(f"corr-{index}" for index in range(60))
    path.write_text(
        "---\ntitle: Memory\nupdated: 2026-07-01\n"
        "status: stable\ntype: knowledge\n"
        f"applied_corrections: [{prior}]\n---\nOld fact.\n",
        encoding="utf-8",
    )
    _patch_pages(monkeypatch, pages)

    prepared = page_mutation.prepare_page_mutation(
        "memory",
        [{"old_text": "Old fact.", "new_text": "New fact."}],
        correction_id="corr-new",
    )
    assert page_mutation.apply_prepared_mutations([prepared])["status"] == "applied"

    written = path.read_text(encoding="utf-8")
    assert "corr-0" in written
    assert "corr-59" in written
    assert "corr-new" in written


def test_constraint_is_fsynced_before_page_marker_is_written(
    tmp_path: Path,
    monkeypatch,
) -> None:
    pages = tmp_path / "pages"
    pages.mkdir()
    path = pages / "memory.md"
    _page(path, title="Memory", body="Old fact.\n")
    _patch_pages(monkeypatch, pages)
    prepared = page_mutation.prepare_page_mutation(
        "memory",
        [{"old_text": "Old fact.", "new_text": "New fact."}],
        correction_id="corr-order",
    )
    real_atomic_write = page_mutation.atomic_write
    observed = {"registry_before_page": False}

    def inspect_registry_before_write(target: Path, content: str) -> None:
        registry = page_mutation.correction_constraints_file()
        observed["registry_before_page"] = (
            registry.exists()
            and '"correction_id": "corr-order"' in registry.read_text()
        )
        real_atomic_write(target, content)

    monkeypatch.setattr(page_mutation, "atomic_write", inspect_registry_before_write)

    assert page_mutation.apply_prepared_mutations([prepared])["status"] == "applied"
    assert observed["registry_before_page"] is True


def test_torn_constraint_tail_is_delimited_and_retry_constraint_stays_active(
    tmp_path: Path,
    monkeypatch,
) -> None:
    pages = tmp_path / "pages"
    pages.mkdir()
    path = pages / "memory.md"
    _page(path, title="Memory", body="Old fact.\n")
    _patch_pages(monkeypatch, pages)
    prepared = page_mutation.prepare_page_mutation(
        "memory",
        [{"old_text": "Old fact.", "new_text": "New fact."}],
        correction_id="corr-after-torn-tail",
    )
    registry = page_mutation.correction_constraints_file()
    registry.parent.mkdir(parents=True, exist_ok=True)
    registry.write_bytes(
        b'{"kind":"content_correction_constraint","correction_id":"torn"'
    )
    real_atomic_write = page_mutation.atomic_write
    failed_once = False

    def fail_first_page_write(target: Path, content: str) -> None:
        nonlocal failed_once
        if target == path and not failed_once:
            failed_once = True
            raise OSError("simulated page write failure")
        real_atomic_write(target, content)

    monkeypatch.setattr(page_mutation, "atomic_write", fail_first_page_write)
    first = page_mutation.apply_prepared_mutations([prepared])
    assert first["status"] == "retry"
    assert "applied_corrections" not in path.read_text(encoding="utf-8")

    second = page_mutation.apply_prepared_mutations([prepared])
    assert second["status"] == "applied"
    written = path.read_text(encoding="utf-8")
    active = page_mutation.active_correction_constraints("memory", written)

    assert registry.read_bytes().startswith(
        b'{"kind":"content_correction_constraint","correction_id":"torn"\n'
    )
    assert [row["correction_id"] for row in active] == ["corr-after-torn-tail"]
    assert "Old fact." not in written
    assert "New fact." in written


def test_apply_refuses_concurrent_change_without_overwrite(
    tmp_path: Path, monkeypatch
) -> None:
    pages = tmp_path / "pages"
    pages.mkdir()
    path = pages / "memory.md"
    _page(path, title="Memory", body="Old fact.\n")
    _patch_pages(monkeypatch, pages)
    prepared = page_mutation.prepare_page_mutation(
        "memory",
        [{"old_text": "Old fact.", "new_text": "New fact."}],
        correction_id="corr-2",
    )
    path.write_text(
        path.read_text(encoding="utf-8") + "Foreign edit.\n", encoding="utf-8"
    )

    result = page_mutation.apply_prepared_mutations([prepared])

    assert result["status"] == "retry"
    assert "Foreign edit." in path.read_text(encoding="utf-8")
    assert "New fact." not in path.read_text(encoding="utf-8")


def test_second_write_failure_rolls_back_first_owned_write(
    tmp_path: Path, monkeypatch
) -> None:
    pages = tmp_path / "pages"
    pages.mkdir()
    first = pages / "first.md"
    second = pages / "second.md"
    _page(first, title="First", body="First old.\n")
    _page(second, title="Second", body="Second old.\n")
    _patch_pages(monkeypatch, pages)
    first_before = first.read_bytes()
    second_before = second.read_bytes()
    prepared = [
        page_mutation.prepare_page_mutation(
            "first",
            [{"old_text": "First old.", "new_text": "First new."}],
            correction_id="corr-3",
        ),
        page_mutation.prepare_page_mutation(
            "second",
            [{"old_text": "Second old.", "new_text": "Second new."}],
            correction_id="corr-3",
        ),
    ]
    real_atomic_write = page_mutation.atomic_write

    def fail_second(path: Path, content: str) -> None:
        if path.name == "second.md":
            raise OSError("simulated disk failure")
        real_atomic_write(path, content)

    monkeypatch.setattr(page_mutation, "atomic_write", fail_second)

    result = page_mutation.apply_prepared_mutations(prepared)

    assert result["status"] == "retry"
    assert result["rolled_back"] == {"first": True}
    assert first.read_bytes() == first_before
    assert second.read_bytes() == second_before


def test_rejects_duplicate_or_protected_old_span(tmp_path: Path, monkeypatch) -> None:
    pages = tmp_path / "pages"
    pages.mkdir()
    duplicate = pages / "duplicate.md"
    protected = pages / "protected.md"
    _page(duplicate, title="Duplicate", body="Wrong. Wrong.\n")
    _page(protected, title="Protected", body="```\nWrong.\n```\n")
    _patch_pages(monkeypatch, pages)

    for page_id in ("duplicate", "protected"):
        try:
            page_mutation.prepare_page_mutation(
                page_id,
                [{"old_text": "Wrong.", "new_text": "Right."}],
                correction_id=f"corr-{page_id}",
            )
        except page_mutation.PageMutationError:
            pass
        else:
            raise AssertionError(f"unsafe mutation unexpectedly prepared for {page_id}")


def test_each_page_is_cas_checked_immediately_before_write(
    tmp_path: Path,
    monkeypatch,
) -> None:
    pages = tmp_path / "pages"
    pages.mkdir()
    first = pages / "first.md"
    second = pages / "second.md"
    _page(first, title="First", body="First old.\n")
    _page(second, title="Second", body="Second old.\n")
    _patch_pages(monkeypatch, pages)
    first_before = first.read_bytes()
    prepared = [
        page_mutation.prepare_page_mutation(
            "first",
            [{"old_text": "First old.", "new_text": "First new."}],
            correction_id="corr-per-write-cas",
        ),
        page_mutation.prepare_page_mutation(
            "second",
            [{"old_text": "Second old.", "new_text": "Second new."}],
            correction_id="corr-per-write-cas",
        ),
    ]
    real_atomic_write = page_mutation.atomic_write

    def change_second_after_first_write(path: Path, content: str) -> None:
        real_atomic_write(path, content)
        if path == first:
            second.write_text(
                second.read_text(encoding="utf-8") + "Foreign edit.\n", encoding="utf-8"
            )

    monkeypatch.setattr(page_mutation, "atomic_write", change_second_after_first_write)

    result = page_mutation.apply_prepared_mutations(prepared)

    assert result["status"] == "retry"
    assert result["rolled_back"] == {"first": True}
    assert first.read_bytes() == first_before
    assert "Foreign edit." in second.read_text(encoding="utf-8")
    assert "Second new." not in second.read_text(encoding="utf-8")


def test_rollback_does_not_replace_foreign_postwrite_bytes(
    tmp_path: Path,
    monkeypatch,
) -> None:
    pages = tmp_path / "pages"
    pages.mkdir()
    path = pages / "memory.md"
    _page(path, title="Memory", body="Old fact.\n")
    _patch_pages(monkeypatch, pages)
    prepared = page_mutation.prepare_page_mutation(
        "memory",
        [{"old_text": "Old fact.", "new_text": "New fact."}],
        correction_id="corr-foreign-postwrite",
    )
    real_atomic_write = page_mutation.atomic_write

    def foreign_write_after_owned_write(target: Path, content: str) -> None:
        real_atomic_write(target, content)
        target.write_text("Foreign replacement.\n", encoding="utf-8")

    monkeypatch.setattr(page_mutation, "atomic_write", foreign_write_after_owned_write)

    result = page_mutation.apply_prepared_mutations([prepared])

    assert result["status"] == "retry"
    assert result["rolled_back"] == {"memory": False}
    assert path.read_text(encoding="utf-8") == "Foreign replacement.\n"


def test_prepare_requires_old_claim_removed_from_active_frontmatter(
    tmp_path: Path,
    monkeypatch,
) -> None:
    pages = tmp_path / "pages"
    pages.mkdir()
    path = pages / "memory.md"
    path.write_text(
        "---\n"
        "title: Memory\n"
        "summary: Installed RAM is 16GB.\n"
        'recall_questions: ["Is installed RAM 16GB?"]\n'
        "updated: 2026-07-01\n"
        "status: stable\ntype: knowledge\n"
        "---\n"
        "Installed RAM is 16GB.\n",
        encoding="utf-8",
    )
    _patch_pages(monkeypatch, pages)

    with pytest.raises(
        page_mutation.PageMutationError, match="frontmatter fields: summary"
    ):
        page_mutation.prepare_page_mutation(
            "memory",
            [
                {
                    "old_text": "Installed RAM is 16GB.",
                    "new_text": "Installed RAM is 32GB.",
                }
            ],
            correction_id="corr-active-meta",
        )

    prepared = page_mutation.prepare_page_mutation(
        "memory",
        [{"old_text": "Installed RAM is 16GB.", "new_text": "Installed RAM is 32GB."}],
        correction_id="corr-active-meta",
        summary="Installed RAM is 32GB.",
        recall_questions=["Is installed RAM 32GB?"],
    )
    assert page_mutation.apply_prepared_mutations([prepared])["status"] == "applied"
    written = path.read_text(encoding="utf-8")
    assert "Installed RAM is 16GB." not in written
    assert "Installed RAM is 32GB." in written


def test_prepare_requires_old_claim_removed_from_description(
    tmp_path: Path,
    monkeypatch,
) -> None:
    pages = tmp_path / "pages"
    pages.mkdir()
    path = pages / "memory.md"
    path.write_text(
        "---\n"
        "title: Memory\n"
        "description: Installed RAM is 16GB.\n"
        "updated: 2026-07-01\n"
        "status: stable\ntype: knowledge\n"
        "---\n"
        "Installed RAM is 16GB.\n",
        encoding="utf-8",
    )
    _patch_pages(monkeypatch, pages)

    with pytest.raises(
        page_mutation.PageMutationError, match="frontmatter fields: description"
    ):
        page_mutation.prepare_page_mutation(
            "memory",
            [
                {
                    "old_text": "Installed RAM is 16GB.",
                    "new_text": "Installed RAM is 32GB.",
                }
            ],
            correction_id="corr-active-description",
        )


def test_review_payload_includes_middle_claim_context_and_bound_diff(
    tmp_path: Path,
    monkeypatch,
) -> None:
    pages = tmp_path / "pages"
    pages.mkdir()
    path = pages / "memory.md"
    _page(
        path,
        title="Memory",
        body=("A" * 20_000) + "\nOld middle claim.\n" + ("B" * 20_000) + "\n",
    )
    _patch_pages(monkeypatch, pages)
    prepared = page_mutation.prepare_page_mutation(
        "memory",
        [{"old_text": "Old middle claim.", "new_text": "New middle claim."}],
        correction_id="corr-review-context",
    )

    payload = prepared.review_payload(preview_chars=1_000)
    replacement = payload["replacements"][0]

    assert "Old middle claim." not in payload["before_preview"]
    assert "Old middle claim." in replacement["before_context"]["context"]
    assert "New middle claim." in replacement["after_context"]["context"]
    assert "-Old middle claim." in replacement["unified_diff_hunk"]
    assert "+New middle claim." in replacement["unified_diff_hunk"]
    assert (
        payload["unified_diff_sha256"]
        == hashlib.sha256(payload["unified_diff"].encode("utf-8")).hexdigest()
    )


def test_already_applied_review_payload_keeps_after_context(
    tmp_path: Path,
    monkeypatch,
) -> None:
    pages = tmp_path / "pages"
    pages.mkdir()
    path = pages / "memory.md"
    _page(path, title="Memory", body="Old fact.\n")
    _patch_pages(monkeypatch, pages)
    prepared = page_mutation.prepare_page_mutation(
        "memory",
        [{"old_text": "Old fact.", "new_text": "New fact."}],
        correction_id="corr-already-applied",
    )
    assert page_mutation.apply_prepared_mutations([prepared])["status"] == "applied"

    recovered = page_mutation.prepare_page_mutation(
        "memory",
        [{"old_text": "Old fact.", "new_text": "New fact."}],
        correction_id="corr-already-applied",
    )
    payload = recovered.review_payload()

    assert recovered.already_applied is True
    assert payload["replacements"][0]["preimage_available"] is False
    assert "New fact." in payload["replacements"][0]["after_context"]["context"]


def test_lock_failure_returns_retry_without_touching_page(
    tmp_path: Path,
    monkeypatch,
) -> None:
    pages = tmp_path / "pages"
    pages.mkdir()
    path = pages / "memory.md"
    _page(path, title="Memory", body="Old fact.\n")
    _patch_pages(monkeypatch, pages)
    before = path.read_bytes()
    prepared = page_mutation.prepare_page_mutation(
        "memory",
        [{"old_text": "Old fact.", "new_text": "New fact."}],
        correction_id="corr-lock-failure",
    )

    @contextmanager
    def fail_lock():
        raise OSError("lock unavailable")
        yield

    monkeypatch.setattr(page_mutation, "chronovisor_mutation_lock", fail_lock)

    result = page_mutation.apply_prepared_mutations([prepared])

    assert result["status"] == "retry"
    assert "lock unavailable" in result["reason"]
    assert path.read_bytes() == before


@pytest.mark.parametrize("status", ["draft", "deprecated"])
def test_mutation_rejects_non_stable_lifecycle(
    tmp_path: Path, monkeypatch, status: str
) -> None:
    pages = tmp_path / "pages"
    pages.mkdir()
    path = pages / "memory.md"
    path.write_text(
        f"---\ntitle: Memory\nstatus: {status}\n---\nOld fact.\n",
        encoding="utf-8",
    )
    _patch_pages(monkeypatch, pages)

    with pytest.raises(page_mutation.PageMutationError, match="not mutable"):
        page_mutation.prepare_page_mutation(
            "memory",
            [{"old_text": "Old fact.", "new_text": "New fact."}],
            correction_id="corr-lifecycle",
        )


def test_mutation_preserves_nested_unknown_yaml_and_unrelated_body_bytes(
    tmp_path: Path, monkeypatch
) -> None:
    pages = tmp_path / "pages"
    pages.mkdir()
    path = pages / "memory.md"
    path.write_bytes(
        b"---\ntitle: Memory\nstatus: stable\ntype: knowledge\nextension:\n  nested:\n    keep: true\n"
        b"---\nPrefix  bytes\r\nOld fact.\r\nSuffix  bytes\r\n"
    )
    _patch_pages(monkeypatch, pages)

    prepared = page_mutation.prepare_page_mutation(
        "memory",
        [{"old_text": "Old fact.", "new_text": "New fact."}],
        correction_id="corr-nested",
    )
    result = page_mutation.apply_prepared_mutations([prepared])

    assert result["status"] == "applied"
    written = path.read_bytes()
    assert b"nested:\n    keep: true" in written
    assert b"Prefix  bytes\r\nNew fact.\r\nSuffix  bytes\r\n" in written
