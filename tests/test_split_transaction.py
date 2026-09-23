from __future__ import annotations

from pathlib import Path

import pytest

from chronovisor.core import frontmatter
from chronovisor.ingest.page_registry import PageRegistry
from chronovisor.recall import split_transaction
from chronovisor.recall.merge_ledger import MergeLedger
from chronovisor.recall.split_transaction import (
    SplitPlanError,
    apply_split_plan,
    prepare_split_plan,
    split_children,
)


@pytest.fixture(autouse=True)
def _valid_okf_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    for name in ("index.md", "log.md", "schema.md"):
        (tmp_path / name).write_text("legacy\n", encoding="utf-8")
    from chronovisor.core import page_mutation

    monkeypatch.setattr(page_mutation, "CHRONOVISOR_ROOT", tmp_path)


def _write(path: Path, title: str, body: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        f"---\ntitle: {title}\nstatus: stable\ntype: knowledge\n"
        "sensitivity: high\ntags:\n- d/test\n---\n" + body,
        encoding="utf-8",
    )


def _fixture(root: Path) -> tuple[Path, Path, str]:
    sections = "".join(
        f"## {index}. Topic {index}\n" + f"fact {index}\n" * 20 for index in range(1, 7)
    )
    duplicate = "## 2. Topic 2\n" + "fact 2\n" * 20
    body = (
        "# Big\n\nintro\n\n"
        + sections
        + duplicate
        + "## 7. Links\nsee [four](big.md#4) and [same](#1)\n"
        + "```\n## Topic 3\n```\n"
    )
    big = root / "pages" / "topic" / "big.md"
    _write(big, "Big", body)
    other = root / "pages" / "topic" / "other.md"
    _write(
        other,
        "Other",
        "# Other\n\n[five](big.md#5) [hub](big.md) [three](big.md#topic-3) [slug](big.md#3-topic-3)\n",
    )
    PageRegistry(root).ensure_manifest()
    return big, other, body


def test_split_keeps_every_section_once_and_retargets_anchors(tmp_path: Path) -> None:
    big, other, _body = _fixture(tmp_path)
    uid = PageRegistry(tmp_path).resolve("big")["uid"]

    plan = prepare_split_plan(tmp_path, page_key="big", target_bytes=300)

    receipt = plan["verification_receipt"]
    assert receipt["unique_sections"] == 8  # "# Big" H1 + 7 H2
    assert receipt["duplicate_sections_removed"] == 1
    assert apply_split_plan(tmp_path, plan)["status"] == "blocked"
    assert not (big.parent / "big-part-01.md").exists()

    result = apply_split_plan(tmp_path, plan, activate=True)

    assert result["status"] == "committed", result
    hub = big.read_text(encoding="utf-8")
    children = split_children(hub)
    assert len(children) >= 2 and "## 1. Topic 1" not in hub
    meta, _ = frontmatter.parse(hub)
    assert meta["title"] == "Big" and meta["sensitivity"] == "high"
    registry = PageRegistry(tmp_path)
    assert registry.resolve("big")["uid"] == uid
    texts = {name: (big.parent / name).read_text(encoding="utf-8") for name in children}
    for name, text in texts.items():
        child_meta, _ = frontmatter.parse(text)
        assert child_meta["split_from"] == uid
        assert child_meta["sensitivity"] == "high"
        assert registry.resolve(name.removesuffix(".md"))["status"] == "stable"
    joined = "".join(texts.values())
    for index in range(1, 7):
        assert joined.count(f"## {index}. Topic {index}\n") == 1
    holder = next(name for name, text in texts.items() if "## 5. Topic 5" in text)
    assert f"(<{holder}#5>)" in other.read_text(encoding="utf-8")
    assert "(big.md)" in other.read_text(encoding="utf-8")
    one = next(name for name, text in texts.items() if "## 1. Topic 1" in text)
    links_child = next(text for text in texts.values() if "## 7. Links" in text)
    assert f"[same](<{one}#1>)" in links_child
    three = next(name for name, text in texts.items() if "## 3. Topic 3\n" in text)
    assert "[three](big.md#topic-3)" in other.read_text(encoding="utf-8")  # fenced
    assert f"[slug](<{three}#3-topic-3>)" in other.read_text(encoding="utf-8")
    assert MergeLedger(tmp_path).recent(limit=1)[0]["status"] == "committed"

    with pytest.raises(SplitPlanError, match="already a split hub"):
        prepare_split_plan(tmp_path, page_key="big")


def test_split_rolls_back_every_owned_file_on_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    big, other, body = _fixture(tmp_path)
    before_other = other.read_bytes()
    plan = prepare_split_plan(tmp_path, page_key="big", target_bytes=300)

    def fail(*_args, **_kwargs):
        raise RuntimeError("registry down")

    monkeypatch.setattr(PageRegistry, "ensure_manifest", fail)
    result = apply_split_plan(tmp_path, plan, activate=True)

    assert result["status"] == "rolled_back"
    assert big.read_text(encoding="utf-8").endswith(body)
    assert other.read_bytes() == before_other
    assert not list(big.parent.glob("big-part-*.md"))


def test_split_restores_registry_written_before_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    big, other, body = _fixture(tmp_path)
    before_other = other.read_bytes()
    registry_path = PageRegistry(tmp_path).path
    before_registry = registry_path.read_bytes()
    plan = prepare_split_plan(tmp_path, page_key="big", target_bytes=300)
    real = PageRegistry.ensure_manifest

    def write_then_fail(self, *args, **kwargs):
        real(self, *args, **kwargs)
        raise RuntimeError("event log down")

    monkeypatch.setattr(PageRegistry, "ensure_manifest", write_then_fail)
    result = apply_split_plan(tmp_path, plan, activate=True)

    assert result["status"] == "rolled_back"
    assert registry_path.read_bytes() == before_registry
    assert big.read_text(encoding="utf-8").endswith(body)
    assert other.read_bytes() == before_other
    assert not list(big.parent.glob("big-part-*.md"))


def test_split_refuses_stale_source(tmp_path: Path) -> None:
    big, _other, _body = _fixture(tmp_path)
    plan = prepare_split_plan(tmp_path, page_key="big", target_bytes=300)
    big.write_text(
        big.read_text(encoding="utf-8") + "## 8. Late\nx\n", encoding="utf-8"
    )

    result = apply_split_plan(tmp_path, plan, activate=True)

    assert result["status"] == "rolled_back"
    assert "CAS mismatch" in result["error"]
    assert not list(big.parent.glob("big-part-*.md"))


def test_invalid_groups_are_rejected(tmp_path: Path) -> None:
    _fixture(tmp_path)
    with pytest.raises(SplitPlanError, match="exactly once"):
        prepare_split_plan(
            tmp_path,
            page_key="big",
            groups=[{"sections": [0, 1]}, {"sections": [1, 2]}],
        )
    assert split_transaction.group_sections([]) == []


def test_split_cli_dry_run_then_activate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from chronovisor.librarian.librarian_merge import run_page_splits
    from chronovisor.recall import collection_authority

    big, _other, body = _fixture(tmp_path)
    monkeypatch.setattr(collection_authority, "OVERSIZED_PAGE_BYTES", 500)
    monkeypatch.setattr(split_transaction, "CHILD_TARGET_BYTES", 300)

    dry = run_page_splits(tmp_path, limit=5)
    assert dry["mode"] == "dry_run" and dry["results"][0]["status"] == "planned"
    assert big.read_text(encoding="utf-8").endswith(body)

    active = run_page_splits(tmp_path, limit=5, activate=True)
    assert [row["status"] for row in active["results"]] == ["committed"]
    assert split_children(big.read_text(encoding="utf-8"))
