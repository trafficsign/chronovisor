"""Entity registry and lightweight alias extraction."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

from llm_wiki_mcp.frontmatter import parse, patch
from llm_wiki_mcp.link_fix import atomic_write
from llm_wiki_mcp.wiki import WIKI_ROOT, all_pages, page_id_from_path

ENTITY_DIR = WIKI_ROOT / "entities"
ENTITY_REGISTRY_FILE = ENTITY_DIR / "registry.json"

DEFAULT_ALIASES: dict[str, list[str]] = {
    "llm-wiki": ["LLM Wiki", "llm wiki", "LLMウィキ", "ウィキ"],
    "codex": ["Codex", "コードエクス", "コーデックス"],
    "claude-code": ["Claude Code", "クラウドコード"],
    "ollama": ["Ollama"],
    "qwen": ["Qwen", "クエン"],
    "gemma": ["Gemma", "ジェンマ"],
    "mhi": ["MHI", "三菱重工", "三菱重工業"],
    "khi": ["KHI", "川崎重工", "川重"],
    "mazda": ["Mazda", "マツダ"],
}

ENTITY_ID_RE = re.compile(r"[^a-z0-9]+")


def normalize_entity_id(value: str) -> str:
    normalized = value.strip().casefold()
    normalized = ENTITY_ID_RE.sub("-", normalized).strip("-")
    return normalized[:80]


def load_registry(path: Path = ENTITY_REGISTRY_FILE) -> dict[str, list[str]]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return dict(DEFAULT_ALIASES)
    aliases: dict[str, list[str]] = dict(DEFAULT_ALIASES)
    if isinstance(data, dict):
        raw_entities = data.get("entities", data)
        if isinstance(raw_entities, dict):
            for key, value in raw_entities.items():
                entity_id = normalize_entity_id(str(key))
                if not entity_id:
                    continue
                values: list[str] = []
                if isinstance(value, list):
                    values = [v for v in value if isinstance(v, str) and v.strip()]
                elif isinstance(value, dict):
                    raw_aliases = value.get("aliases")
                    if isinstance(raw_aliases, list):
                        values = [
                            v for v in raw_aliases
                            if isinstance(v, str) and v.strip()
                        ]
                    label = value.get("label")
                    if isinstance(label, str) and label.strip():
                        values.insert(0, label)
                aliases[entity_id] = list(dict.fromkeys([entity_id, *values]))
    return aliases


def write_default_registry(path: Path = ENTITY_REGISTRY_FILE) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        path.write_text(
            json.dumps({"entities": DEFAULT_ALIASES}, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    return path


def extract_entities(text: str, *, registry: dict[str, list[str]] | None = None) -> list[str]:
    registry = registry or load_registry()
    haystack = text.casefold()
    found: list[str] = []
    for entity_id, aliases in registry.items():
        for alias in aliases:
            if alias and alias.casefold() in haystack:
                found.append(entity_id)
                break
    return list(dict.fromkeys(found))[:20]


def patch_entities_frontmatter(
    text: str,
    *,
    registry: dict[str, list[str]] | None = None,
) -> str:
    meta, body = parse(text)
    title = meta.get("title")
    current = meta.get("entities")
    existing = current if isinstance(current, list) else []
    extracted = extract_entities(
        "\n".join([title if isinstance(title, str) else "", body]),
        registry=registry,
    )
    merged = list(dict.fromkeys([*existing, *extracted]))
    if not merged or merged == existing:
        return text
    return patch(text, {"entities": merged})


def backfill_entities(
    *,
    limit: int = 0,
    dry_run: bool = False,
    include_reference: bool = False,
) -> dict[str, Any]:
    registry = load_registry()
    scanned = 0
    updated = 0
    skipped_reference = 0
    pages: list[str] = []
    for path in all_pages():
        scanned += 1
        try:
            text = path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            continue
        meta, _body = parse(text)
        if (
            not include_reference
            and (meta.get("type") == "reference" or path.parent.name == "car-spec")
        ):
            skipped_reference += 1
            continue
        new_text = patch_entities_frontmatter(text, registry=registry)
        if new_text == text:
            continue
        updated += 1
        pages.append(page_id_from_path(path))
        if not dry_run:
            atomic_write(path, new_text)
        if limit and updated >= limit:
            break
    return {
        "status": "ok",
        "dry_run": dry_run,
        "scanned": scanned,
        "skipped_reference": skipped_reference,
        "updated": updated,
        "pages": pages,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Maintain LLM Wiki entity frontmatter.")
    sub = parser.add_subparsers(dest="command", required=True)
    init_cmd = sub.add_parser("init", help="Write a default entity registry if missing.")
    init_cmd.add_argument("--json", action="store_true")
    backfill = sub.add_parser("backfill", help="Backfill entities on existing pages.")
    backfill.add_argument("--limit", type=int, default=0)
    backfill.add_argument("--dry-run", action="store_true")
    backfill.add_argument("--include-reference", action="store_true")
    backfill.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)

    if args.command == "init":
        path = write_default_registry()
        payload = {"status": "ok", "path": str(path)}
    else:
        payload = backfill_entities(
            limit=max(0, args.limit),
            dry_run=args.dry_run,
            include_reference=args.include_reference,
        )

    if getattr(args, "json", False):
        print(json.dumps(payload, ensure_ascii=False))
    else:
        print("\t".join(f"{key}={value}" for key, value in payload.items() if key != "pages"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
