"""Persistent page-id aliases for self-healing update target repair."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from chronovisor.common import paths as chronovisor_paths


def _alias_file() -> Path:
    return chronovisor_paths.CHRONOVISOR_ROOT / "runtime" / "page-aliases.json"


def _normalize_page_id(value: str) -> str:
    text = value.strip()
    if text.endswith(".md"):
        text = text[:-3]
    if "/" in text:
        text = text.rsplit("/", 1)[-1]
    return text


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


def resolve_target_path(target: str) -> Path | None:
    ref = target.strip()
    if not ref:
        return None
    if ref.endswith(".md"):
        ref = ref[:-3]
    from chronovisor.pages.store import (
        PAGE_RESERVED_FILENAMES,
        canonical_document_path,
    )

    if "/" in ref:
        return canonical_document_path(
            chronovisor_paths.PAGES_DIR / f"{ref}.md",
            chronovisor_paths.PAGES_DIR,
            namespace="pages",
            reserved_filenames=PAGE_RESERVED_FILENAMES,
            require_stable=True,
        )

    matches = [
        resolved
        for path in chronovisor_paths.PAGES_DIR.rglob(f"{ref}.md")
        if (
            resolved := canonical_document_path(
                path,
                chronovisor_paths.PAGES_DIR,
                namespace="pages",
                reserved_filenames=PAGE_RESERVED_FILENAMES,
                require_stable=True,
            )
        )
        is not None
    ]
    return matches[0] if len(matches) == 1 else None


def resolve_alias_path(page_id: str) -> Path | None:
    aliases = load_aliases()
    target = aliases.get(_normalize_page_id(page_id))
    if not target:
        return None
    return resolve_target_path(target)
