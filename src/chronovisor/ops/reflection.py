"""Generate durable reflection pages from health metrics."""

from __future__ import annotations

import argparse
import json
from datetime import date, datetime
from pathlib import Path
from typing import Any

from chronovisor.core.store import PAGES_DIR, find_page, okf_runtime_operation
from chronovisor.ingest.page_write import apply_page_writes, prepare_page_write
from chronovisor.ops.health import health_snapshot

INSIGHTS_DIR = PAGES_DIR / "insights"


def _fmt(value: object) -> str:
    if isinstance(value, float):
        return f"{value:.3f}"
    if value is None:
        return "unknown"
    return str(value)


def build_reflection_markdown(
    snapshot: dict[str, Any], *, today: date | None = None
) -> str:
    today = today or date.today()
    coverage = (
        snapshot.get("coverage", {})
        if isinstance(snapshot.get("coverage"), dict)
        else {}
    )
    memory = (
        snapshot.get("memory_integrity", {})
        if isinstance(snapshot.get("memory_integrity"), dict)
        else {}
    )
    queues = (
        snapshot.get("queues", {}) if isinstance(snapshot.get("queues"), dict) else {}
    )
    feedback = (
        snapshot.get("recall_feedback", {})
        if isinstance(snapshot.get("recall_feedback"), dict)
        else {}
    )
    title = f"Memory Reflection {today.isoformat()}"
    return "\n".join(
        [
            "---",
            f"title: {title}",
            f"updated: {today.isoformat()}",
            "status: stable",
            "type: semantic",
            "tags: [d/chronovisor, t/reflection]",
            f"description: Sleep-cycle reflection generated from dashboard health metrics on {today.isoformat()}.",
            "---",
            "",
            f"# {title}",
            "",
            "## Signals",
            f"- Knowledge pages: {_fmt(coverage.get('knowledge_pages'))}",
            f"- Summary coverage: {_fmt(coverage.get('summary_coverage'))}",
            f"- Recall question coverage: {_fmt(coverage.get('recall_question_coverage'))}",
            f"- Memory capture rate: {_fmt(memory.get('capture_rate'))}",
            f"- Recall precision proxy: {_fmt(feedback.get('precision_proxy'))}",
            "",
            "## Queues",
            f"- Duplicate candidates: {_fmt(queues.get('duplicate_candidates'))}",
            f"- Lint repair queue: {_fmt(queues.get('lint_repair'))}",
            f"- Search golden examples: {_fmt(queues.get('search_golden'))}",
            "",
            "## Next Checks",
            "- Review deprecation candidates from retention output before excluding anything.",
            "- Promote high-confidence duplicate candidates through the review lane.",
            "- Keep generated reflections cited by metric source, not as standalone facts.",
            "",
        ]
    )


def write_reflection_page(
    *,
    output_dir: Path | None = None,
    write: bool = True,
) -> dict[str, Any]:
    if write:
        with okf_runtime_operation(PAGES_DIR.parent):
            return _write_reflection_page_locked(output_dir=output_dir, write=True)
    return _write_reflection_page_locked(output_dir=output_dir, write=False)


def _write_reflection_page_locked(
    *, output_dir: Path | None, write: bool
) -> dict[str, Any]:
    snapshot = health_snapshot()
    today = date.today()
    page_id = f"memory-reflection-{today.isoformat()}"
    if output_dir is None:
        # Generated page IDs are global even when an operator reorganizes the
        # pages tree. Keep updating the existing page at its current location
        # instead of recreating the same ID in the generator's preferred
        # directory. Explicit output_dir remains isolated for tests/exports.
        path = find_page(page_id) or INSIGHTS_DIR / f"{page_id}.md"
    else:
        path = output_dir / f"{page_id}.md"
    markdown = build_reflection_markdown(snapshot, today=today)
    mutation: dict[str, Any] | None = None
    if write:
        try:
            source_path = (
                path.resolve(strict=False)
                .relative_to(PAGES_DIR.resolve(strict=False))
                .as_posix()
            )
        except ValueError:
            source_path = f"insights/{path.name}"
        mutation = apply_page_writes(
            [
                prepare_page_write(
                    path,
                    markdown,
                    namespace="pages",
                    source_path=source_path,
                )
            ]
        )
    return {
        "status": (
            "ok"
            if mutation is None or mutation["status"] in {"applied", "unchanged"}
            else "retry"
        ),
        "path": str(path),
        "write": write,
        "mutation": mutation,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
    }


def main(argv: list[str] | None = None) -> int:
    """Run the ``chronovisor-reflect`` command-line entry point."""
    parser = argparse.ArgumentParser(description="Generate a memory reflection page.")
    parser.add_argument("--no-write", action="store_true")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)
    from chronovisor.core.okf_cutover import OKFStartupBlocked

    try:
        data = write_reflection_page(write=not args.no_write)
    except OKFStartupBlocked:
        print(json.dumps({"status": "blocked", "category": "okf_startup_blocked"}))
        return 75
    if args.json:
        print(json.dumps(data, ensure_ascii=False, indent=2))
    else:
        print(f"path\t{data['path']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
