from pathlib import Path

from chronovisor.librarian.link_anchor_repair import (
    DEFAULT_REPAIRS,
    repair_known_anchors,
)


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
        "Read [[target#old|label]].\n\n```md\n[[target#old|example]]\n```\n",
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


def test_default_repairs_cover_reviewed_live_anchor_debt() -> None:
    expected = {
        (
            "local-ai-hardware-strategy-2026",
            "section_22",
        ): "22. M4 Max 128GB の「物理的再現不可能性」と供給逼迫の構造的終着点",
        (
            "child-record-segment-structure",
            "Phase Field Details",
        ): "Segment Phase Field Flexibility",
        (
            "child-record-policy-v2-production-instance",
            "eleventh-instance-boundary-hardening",
        ): "Campaign Boundary Hardening Instance (2026-08-07)",
        (
            "child-record-policy-v2-production-instance",
            "twelfth-instance-v2-only-migration",
        ): "Campaign v2-Only Migration Instance (2026-08-07)",
        (
            "child-record-policy-v2-production-instance",
            "sixteenth-instance-parallelization-strategy",
        ): "Campaign Parallelization Strategy Instance (2026-08-07)",
        (
            "child-record-policy-v2-production-instance",
            "fourth-instance-final-review",
        ): "Campaign Final Review Instance (2026-08-06)",
        (
            "child-record-policy-v2-production-instance",
            "fifteenth-instance-p1-analysis",
        ): "Campaign P1 Analysis Instance (2026-08-07)",
        (
            "child-record-policy-v2-production-instance",
            "ninth-instance-frozen-dependency-hardening",
        ): "Campaign Frozen Dependency Reference Hardening Instance (2026-08-06)",
    }

    assert expected.items() <= DEFAULT_REPAIRS.items()
