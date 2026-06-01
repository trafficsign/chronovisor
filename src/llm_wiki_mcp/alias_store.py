"""Persistent page-id aliases for self-healing update target repair."""
from __future__ import annotations

import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any

from llm_wiki_mcp import wiki


def _alias_file() -> Path:
    return wiki.WIKI_ROOT / "runtime" / "page-aliases.json"


def _normalize_page_id(value: str) -> str:
    text = value.strip()
    if text.endswith(".md"):
        text = text[:-3]
    if "/" in text:
        text = text.rsplit("/", 1)[-1]
    return text


def _valid_page_id(value: str) -> bool:
    return bool(re.fullmatch(r"[A-Za-z0-9._-]+", value))


def _load() -> dict[str, Any]:
    path = _alias_file()
    if not path.exists():
        return {"aliases": {}}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {"aliases": {}}
    if not isinstance(data, dict):
        return {"aliases": {}}
    aliases = data.get("aliases")
    if not isinstance(aliases, dict):
        data["aliases"] = {}
    return data


def _save(data: dict[str, Any]) -> None:
    path = _alias_file()
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp")
    tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    tmp.replace(path)


def load_aliases() -> dict[str, str]:
    data = _load()
    aliases = data.get("aliases", {})
    out: dict[str, str] = {}
    if isinstance(aliases, dict):
        for alias, record in aliases.items():
            if isinstance(record, dict) and isinstance(record.get("target"), str):
                out[str(alias)] = record["target"]
            elif isinstance(record, str):
                out[str(alias)] = record
    return out


def add_alias(alias: str, target: str, *, source: str | None = None) -> None:
    """Persist an alias after validating that target resolves to exactly one page."""

    alias_id = _normalize_page_id(alias)
    target_id = target.strip()
    if not alias_id or not _valid_page_id(alias_id):
        raise ValueError(f"invalid alias page_id: {alias!r}")
    target_path = resolve_target_path(target_id)
    if target_path is None:
        raise ValueError(f"alias target does not exist: {target!r}")

    try:
        target_ref = str(target_path.relative_to(wiki.PAGES_DIR).with_suffix(""))
    except ValueError:
        target_ref = target_path.stem

    data = _load()
    aliases = data.setdefault("aliases", {})
    if not isinstance(aliases, dict):
        aliases = {}
        data["aliases"] = aliases
    existing = aliases.get(alias_id)
    if isinstance(existing, dict) and existing.get("target") not in (None, target_ref):
        raise ValueError(
            f"alias {alias_id!r} already points at {existing.get('target')!r}"
        )
    aliases[alias_id] = {
        "target": target_ref,
        "source": source,
        "updated_at": datetime.now().isoformat(),
    }
    _save(data)


def resolve_target_path(target: str) -> Path | None:
    ref = target.strip()
    if not ref:
        return None
    if ref.endswith(".md"):
        ref = ref[:-3]
    if "/" in ref:
        path = (wiki.PAGES_DIR / f"{ref}.md").resolve()
        try:
            path.relative_to(wiki.PAGES_DIR.resolve())
        except ValueError:
            return None
        return path if path.exists() else None

    flat = wiki.PAGES_DIR / f"{ref}.md"
    if flat.exists():
        return flat
    matches = list(wiki.PAGES_DIR.rglob(f"{ref}.md"))
    return matches[0] if len(matches) == 1 else None


def resolve_alias_path(page_id: str) -> Path | None:
    aliases = load_aliases()
    target = aliases.get(_normalize_page_id(page_id))
    if not target:
        return None
    return resolve_target_path(target)
