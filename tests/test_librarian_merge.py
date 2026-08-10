from __future__ import annotations

import json
from pathlib import Path

import pytest

from chronovisor.core.legacy_archive import migrate_processed_legacy
from chronovisor.ingest.page_registry import PageRegistry
from chronovisor.librarian.librarian_merge import prepare_cluster_plan
from chronovisor.recall.merge_transaction import (
    apply_merge_plan,
    cleanup_expired_preimages,
)


@pytest.fixture(autouse=True)
def _valid_okf_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    for name in ("index.md", "log.md", "schema.md"):
        (tmp_path / name).write_text("legacy\n", encoding="utf-8")
    from chronovisor.core import page_mutation

    monkeypatch.setattr(page_mutation, "CHRONOVISOR_ROOT", tmp_path)


def test_discover_main_blocks_before_mutation_on_unsafe_root(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    root = tmp_path / "blocked"
    root.mkdir()
    (root / "private.txt").write_text("canary", encoding="utf-8")

    from chronovisor.librarian import librarian_merge

    assert librarian_merge.main(["discover", "--root", str(root)]) == 75
    assert json.loads(capsys.readouterr().out) == {
        "status": "blocked",
        "category": "okf_startup_blocked",
    }
    assert [path.name for path in root.iterdir()] == ["private.txt"]


def _page(
    path: Path,
    title: str,
    body: str,
    *,
    sensitivity: str = "normal",
    recall_questions: list[str] | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "---\n"
        f"title: {title}\n"
        "status: stable\n"
        "type: knowledge\n"
        "updated: 2026-07-25\n"
        f"sensitivity: {sensitivity}\n"
        + (
            "recall_questions: " + json.dumps(recall_questions) + "\n"
            if recall_questions
            else ""
        )
        + "---\n\n"
        f"# {title}\n\n{body}\n",
        encoding="utf-8",
    )


def test_cluster_plan_rejects_stale_noncanonical_registry_source(
    tmp_path: Path,
) -> None:
    alpha = tmp_path / "pages" / "alpha.md"
    beta = tmp_path / "pages" / "beta.md"
    _page(alpha, "Alpha", "Alpha fact.")
    _page(beta, "Beta", "Beta fact.")
    registry = PageRegistry(tmp_path)
    registry.ensure_manifest()
    beta.write_text(
        beta.read_text(encoding="utf-8").replace(
            "status: stable", "status: deprecated"
        ),
        encoding="utf-8",
    )

    with pytest.raises(KeyError, match="beta"):
        prepare_cluster_plan(tmp_path, page_keys=["alpha", "beta"])


def test_cluster_plan_and_transaction_preserve_links_provenance_and_sensitivity(
    tmp_path: Path,
) -> None:
    _page(
        tmp_path / "pages" / "alpha.md",
        "Alpha",
        "Alpha fact 2026-07-25.\n\n## Details\n\nURL https://example.com/a",
    )
    _page(
        tmp_path / "pages" / "beta.md",
        "Beta",
        "Beta fact 42GB.\n\n## Details\n\n`beta.symbol` remains.",
        sensitivity="restricted",
        recall_questions=["What is the beta fact?"],
    )
    _page(
        tmp_path / "pages" / "incoming.md",
        "Incoming",
        "[Alpha](<alpha.md>) [Alpha again](<alpha.md>) "
        "[legacy detail](<beta.md#Details>)",
    )
    for name in ("alpha-source.md", "beta-source.md"):
        raw = tmp_path / "raw" / name
        raw.parent.mkdir(parents=True, exist_ok=True)
        raw.write_text(f"raw provenance for {name}\n", encoding="utf-8")
    claims = tmp_path / "claims" / "claims.jsonl"
    claims.parent.mkdir(parents=True, exist_ok=True)
    claims.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "source_page": "alpha",
                        "source_raw": "alpha-source.md",
                    }
                ),
                json.dumps(
                    {
                        "source_page": "beta",
                        "source_raw": "beta-source.md",
                    }
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    registry = PageRegistry(tmp_path)
    registry.ensure_manifest()

    plan = prepare_cluster_plan(
        tmp_path,
        page_keys=["alpha", "beta"],
    )

    assert plan["status"] == "prepared"
    assert plan["output"]["sensitivity"] == "restricted"
    assert plan["verification_receipt"]["status"] == "verified"
    assert "What is the beta fact?" in plan["output"]["content"]
    assert plan["link_rewrites"]
    result = apply_merge_plan(
        tmp_path,
        plan,
        activate=True,
        preimage_ttl_days=7,
    )

    assert result["status"] == "committed"
    canonical_uid = plan["output"]["uid"]
    loser_uid = next(
        row["uid"] for row in plan["inputs"] if row["uid"] != canonical_uid
    )
    loser_page_id = next(
        Path(row["path"]).stem for row in plan["inputs"] if row["uid"] == loser_uid
    )
    assert registry.resolve(loser_uid)["uid"] == canonical_uid
    incoming = (tmp_path / "pages" / "incoming.md").read_text(encoding="utf-8")
    assert f"<{loser_page_id}.md#" not in incoming
    assert "legacy detail" in incoming
    assert Path(result["preimage"]).is_dir()
    cleanup = cleanup_expired_preimages(tmp_path, force=True)
    assert cleanup["deleted"] == [plan["transaction_id"]]
    assert cleanup["retained"] == []
    assert not Path(result["preimage"]).exists()


def test_cluster_plan_resolves_raw_provenance_inside_legacy_archives(
    tmp_path: Path,
) -> None:
    _page(tmp_path / "pages" / "alpha.md", "Alpha", "Alpha archived fact.")
    _page(tmp_path / "pages" / "beta.md", "Beta", "Beta archived fact.")
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    for name in ("alpha-source.md", "beta-source.md"):
        (raw_dir / name).write_text(f"raw {name}\n", encoding="utf-8")
    claims = tmp_path / "claims" / "claims.jsonl"
    claims.parent.mkdir(parents=True)
    claims.write_text(
        "\n".join(
            json.dumps({"source_page": page, "source_raw": f"{page}-source.md"})
            for page in ("alpha", "beta")
        )
        + "\n",
        encoding="utf-8",
    )
    migrate_processed_legacy(
        raw_dir,
        processed_raw_ids={"alpha-source.md", "beta-source.md"},
        before="2100/01/01",
        remove_source=True,
        dry_run=False,
    )
    registry = PageRegistry(tmp_path)
    registry.ensure_manifest()

    plan = prepare_cluster_plan(tmp_path, page_keys=["alpha", "beta"])

    assert plan["status"] == "prepared"
    assert plan["verification_receipt"]["raw_ref_count"] > 0


def test_missing_raw_provenance_is_terminal_keep_both_not_human_hold(
    tmp_path: Path,
) -> None:
    _page(tmp_path / "pages" / "alpha.md", "Alpha", "Alpha orphan fact.")
    _page(tmp_path / "pages" / "beta.md", "Beta", "Beta orphan fact.")
    PageRegistry(tmp_path).ensure_manifest()

    plan = prepare_cluster_plan(tmp_path, page_keys=["alpha", "beta"])

    assert plan["status"] == "kept"
    assert plan["reason"] == "raw_provenance_unrecoverable_keep_both"
    assert len(plan["uids"]) == 2


def test_transaction_restores_owned_pages_when_registry_commit_fails(
    tmp_path: Path,
    monkeypatch,
) -> None:
    _page(tmp_path / "pages" / "alpha.md", "Alpha", "Alpha fact 2026.")
    _page(
        tmp_path / "pages" / "beta.md",
        "Beta",
        "Beta fact 42GB.",
        sensitivity="restricted",
    )
    _page(
        tmp_path / "pages" / "incoming.md",
        "Incoming",
        "[legacy](<beta.md>)",
    )
    for name in ("alpha-source.md", "beta-source.md"):
        raw = tmp_path / "raw" / name
        raw.parent.mkdir(parents=True, exist_ok=True)
        raw.write_text(f"raw {name}\n", encoding="utf-8")
    claims = tmp_path / "claims" / "claims.jsonl"
    claims.parent.mkdir(parents=True, exist_ok=True)
    claims.write_text(
        "\n".join(
            json.dumps({"source_page": page, "source_raw": f"{page}-source.md"})
            for page in ("alpha", "beta")
        )
        + "\n",
        encoding="utf-8",
    )
    registry = PageRegistry(tmp_path)
    registry.ensure_manifest()
    plan = prepare_cluster_plan(
        tmp_path,
        page_keys=["alpha", "beta"],
    )
    before = {path: path.read_bytes() for path in (tmp_path / "pages").glob("*.md")}

    def fail_redirects(*args, **kwargs):
        raise RuntimeError("injected registry failure")

    monkeypatch.setattr(PageRegistry, "add_redirects", fail_redirects)
    result = apply_merge_plan(
        tmp_path,
        plan,
        activate=True,
        preimage_ttl_days=7,
    )

    assert result["status"] == "rolled_back"
    assert "injected registry failure" in result["error"]
    assert all(path.read_bytes() == content for path, content in before.items())
    assert registry.resolve("alpha")["status"] == "stable"
    assert registry.resolve("beta")["status"] == "stable"
