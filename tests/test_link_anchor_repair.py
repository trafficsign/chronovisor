from pathlib import Path

from chronovisor.librarian.link_anchor_repair import repair_known_anchors


def _page(path: Path, title: str, body: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        f"---\ntitle: {title}\n---\n\n# {title}\n\n{body}\n",
        encoding="utf-8",
    )


def test_reviewed_anchor_repair_is_snapshot_backed_and_code_safe(
    tmp_path: Path,
) -> None:
    _page(
        tmp_path / "pages" / "target.md",
        "Target",
        "## New heading\n",
    )
    _page(
        tmp_path / "pages" / "source.md",
        "Source",
        "Read [[target#old|label]].\n\n"
        "```md\n[[target#old|example]]\n```\n",
    )

    result = repair_known_anchors(
        tmp_path,
        repairs={("target", "old"): "New heading"},
    )

    text = (tmp_path / "pages" / "source.md").read_text(encoding="utf-8")
    assert result["status"] == "committed"
    assert result["before_unresolved"] == 1
    assert result["after_unresolved"] == 0
    assert result["links_changed"] == 1
    assert "[[target#New heading|label]]" in text
    assert "[[target#old|example]]" in text
    restore = (
        tmp_path
        / "runtime"
        / "librarian"
        / "migration-restore-points"
        / result["restore_id"]
    )
    assert (restore / "manifest.json").is_file()
