"""Generate durable reflection pages from health metrics."""

from __future__ import annotations

import argparse
import json
from datetime import date, datetime
from pathlib import Path
from typing import Any

from llm_wiki_mcp.health import health_snapshot
from llm_wiki_mcp.wiki import PAGES_DIR, find_page
from llm_wiki_mcp.wiki_write import apply_wiki_writes, prepare_wiki_write

INSIGHTS_DIR = PAGES_DIR / "insights"


def _fmt(value: object) -> str:
    if isinstance(value, float):
        return f"{value:.3f}"
    if value is None:
        return "unknown"
    return str(value)


def build_reflection_markdown(snapshot: dict[str, Any], *, today: date | None = None) -> str:
    today = today or date.today()
    coverage = snapshot.get("coverage", {}) if isinstance(snapshot.get("coverage"), dict) else {}
    memory = snapshot.get("memory_integrity", {}) if isinstance(snapshot.get("memory_integrity"), dict) else {}
    queues = snapshot.get("queues", {}) if isinstance(snapshot.get("queues"), dict) else {}
    feedback = snapshot.get("recall_feedback", {}) if isinstance(snapshot.get("recall_feedback"), dict) else {}
    title = f"Memory Reflection {today.isoformat()}"
    return "\n".join(
        [
            "---",
            f"title: {title}",
            f"updated: {today.isoformat()}",
            "type: semantic",
            "tags: [d/llm-wiki, t/reflection]",
            f"summary: Sleep-cycle reflection generated from dashboard health metrics on {today.isoformat()}.",
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
            "- Review archive candidates from retention output before excluding anything.",
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
        mutation = apply_wiki_writes([prepare_wiki_write(path, markdown)])
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
    parser = argparse.ArgumentParser(description="Generate a memory reflection page.")
    parser.add_argument("--no-write", action="store_true")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)
    data = write_reflection_page(write=not args.no_write)
    if args.json:
        print(json.dumps(data, ensure_ascii=False, indent=2))
    else:
        print(f"path\t{data['path']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
