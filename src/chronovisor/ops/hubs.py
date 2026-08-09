"""Auto-maintained map-of-content hub pages."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter, defaultdict
from datetime import date, datetime
from pathlib import Path
from typing import Any

from chronovisor.core.index_store import get_store
from chronovisor.core.store import PAGES_DIR
from chronovisor.ingest.page_write import apply_page_writes, prepare_page_write

HUBS_DIR = PAGES_DIR / "hubs"


def _slug(value: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")
    if slug:
        return slug
    digest = hashlib.sha1(value.encode("utf-8")).hexdigest()[:10]
    return f"hub-{digest}"


def _folder_for_meta(meta: dict[str, Any]) -> str:
    path = meta.get("path")
    if isinstance(path, str):
        parent = Path(path).parent
        if parent != PAGES_DIR:
            return parent.name
    return ""


def _hub_markdown(kind: str, name: str, pages: list[dict[str, Any]], *, today: date) -> str:
    title = f"{name} Hub"
    links = []
    for meta in pages[:80]:
        page_id = str(meta.get("page_id") or "")
        page_title = str(meta.get("title") or page_id)
        updated = str(meta.get("updated") or "unknown")
        links.append(f"- [[{page_id}]] - {page_title} ({updated})")
    return "\n".join(
        [
            "---",
            f"title: {title}",
            f"updated: {today.isoformat()}",
            "type: semantic",
            "tags: [d/chronovisor, t/hub]",
            f"summary: Auto-maintained {kind} map of content for {name}.",
            "---",
            "",
            f"# {title}",
            "",
            f"Auto-maintained {kind} hub. Regenerate with `chronovisor hubs`.",
            "",
            "## Pages",
            *links,
            "",
        ]
    )


def _existing_hub_path(store: Any, page_id: str) -> Path | None:
    """Keep an existing hub at its indexed path instead of duplicating its ID."""

    if not hasattr(store, "meta"):
        return None
    meta = store.meta(page_id)
    value = meta.get("path") if isinstance(meta, dict) else None
    if not isinstance(value, str) or not value:
        return None
    candidate = Path(value).expanduser().resolve(strict=False)
    pages_root = PAGES_DIR.expanduser().resolve(strict=False)
    try:
        candidate.relative_to(pages_root)
    except ValueError:
        return None
    if not candidate.is_file() or candidate.is_symlink():
        return None
    return candidate


def build_hub_pages(
    *,
    output_dir: Path = HUBS_DIR,
    min_pages: int = 3,
    max_hubs: int = 20,
    write: bool = True,
) -> dict[str, Any]:
    store = get_store()
    store.refresh()
    metas = [
        meta
        for meta in store.all_pages_meta(include_system=False)
        if meta.get("page_type") != "reference"
    ]
    folders: dict[str, list[dict[str, Any]]] = defaultdict(list)
    entities: dict[str, list[dict[str, Any]]] = defaultdict(list)
    entity_counts: Counter[str] = Counter()
    for meta in metas:
        page_id = str(meta.get("page_id") or "")
        full = store.meta(page_id) if page_id and hasattr(store, "meta") else None
        if isinstance(full, dict):
            meta = {**meta, **full}
        folder = _folder_for_meta(meta)
        if folder and folder != "hubs":
            folders[folder].append(meta)
        for entity in meta.get("entities", []) or []:
            if isinstance(entity, str) and entity.strip():
                entities[entity].append(meta)
                entity_counts[entity] += 1

    selected: list[tuple[str, str, list[dict[str, Any]]]] = []
    for folder, pages in sorted(folders.items(), key=lambda item: len(item[1]), reverse=True):
        if len(pages) >= min_pages:
            selected.append(("folder", folder, pages))
    for entity, _count in entity_counts.most_common(max_hubs):
        pages = entities[entity]
        if len(pages) >= min_pages:
            selected.append(("entity", entity, pages))
    selected = selected[:max_hubs]

    written: list[str] = []
    plans = []
    today = date.today()
    for kind, name, pages in selected:
        page_id = f"{kind}-{_slug(name)}-hub"
        path = _existing_hub_path(store, page_id) or output_dir / f"{page_id}.md"
        if write:
            plans.append(
                prepare_page_write(
                    path,
                    _hub_markdown(kind, name, pages, today=today),
                )
            )
        written.append(str(path))
    mutation = apply_page_writes(plans) if write else None
    return {
        "status": (
            "ok"
            if mutation is None or mutation["status"] in {"applied", "unchanged"}
            else "retry"
        ),
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "write": write,
        "hubs": len(selected),
        "paths": written,
        "mutation": mutation,
    }


def main(argv: list[str] | None = None) -> int:
    """Run the ``chronovisor-hubs`` command-line entry point."""
    parser = argparse.ArgumentParser(description="Generate auto-maintained hub pages.")
    parser.add_argument("--min-pages", type=int, default=3)
    parser.add_argument("--max-hubs", type=int, default=20)
    parser.add_argument("--no-write", action="store_true")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)
    data = build_hub_pages(
        min_pages=max(1, args.min_pages),
        max_hubs=max(1, args.max_hubs),
        write=not args.no_write,
    )
    if args.json:
        print(json.dumps(data, ensure_ascii=False, indent=2))
    else:
        print(f"hubs\t{data['hubs']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
