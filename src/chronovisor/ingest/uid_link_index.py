"""Derived UID edge index for wiki links.

Markdown remains slug-oriented for Obsidian.  This projection resolves link
targets to stable UIDs and keeps classification concepts out of the ordinary
wikilink namespace.
"""

from __future__ import annotations

import hashlib
import json
import re
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from chronovisor.core.durable_state import write_sealed_json
from chronovisor.core.link_fix import extract_wiki_links
from chronovisor.ingest.page_registry import PageRegistry

SCHEMA = "chronovisor.uid-link-index.v1"


def _now_iso() -> str:
    return datetime.now(UTC).isoformat(timespec="milliseconds")


def _heading_anchors(text: str) -> set[str]:
    anchors: set[str] = set()
    for line in text.splitlines():
        if not line.startswith("#"):
            continue
        heading = line.lstrip("#").strip().casefold()
        anchors.add(heading)
        anchors.add(heading.replace(" ", "-"))
        slug = re.sub(
            r"[^0-9a-z\u3040-\u30ff\u3400-\u9fff]+",
            "-",
            heading,
        ).strip("-")
        if slug:
            anchors.add(slug)
        numbered = re.match(r"^(?:section[- ]*)?(\d+(?:\.\d+)*)", heading)
        if numbered:
            section = numbered.group(1)
            anchors.add(section)
            anchors.add(f"section-{section}")
            anchors.add(f"section{section}")
    return anchors


def build_uid_link_index(
    root: Path,
    *,
    registry: PageRegistry | None = None,
    write: bool = True,
) -> dict[str, Any]:
    registry = registry or PageRegistry(root)
    manifest = registry.ensure_manifest(write=write)
    state = manifest["registry"]
    edges: list[dict[str, str]] = []
    unresolved: list[dict[str, str]] = []
    reverse: dict[str, list[str]] = {}
    anchors: dict[str, list[str]] = {}
    heading_index: dict[str, set[str]] = {}
    for uid, row in state["pages"].items():
        if not isinstance(row, dict) or row.get("status") == "superseded":
            continue
        path = root / str(row.get("path") or "")
        if path.exists():
            heading_index[str(uid)] = _heading_anchors(path.read_text(encoding="utf-8"))

    for source_uid, row in sorted(state["pages"].items()):
        if not isinstance(row, dict) or row.get("status") == "superseded":
            continue
        path = root / str(row.get("path") or "")
        if not path.exists():
            continue
        text = path.read_text(encoding="utf-8")
        for raw_target in extract_wiki_links(text, strip=True):
            target_text = str(raw_target).split("|", 1)[0].strip()
            page_key, _, anchor = target_text.partition("#")
            target_uid = state.get("keys", {}).get(page_key.casefold())
            if target_uid is None:
                ambiguous = state.get("ambiguous_keys", {}).get(page_key.casefold())
                if isinstance(ambiguous, list):
                    exact = [
                        uid
                        for uid in ambiguous
                        if str(
                            (state.get("pages", {}).get(uid) or {}).get("page_id")
                            or ""
                        ).casefold()
                        == page_key.casefold()
                    ]
                    if len(exact) == 1:
                        target_uid = exact[0]
            target = (
                state.get("pages", {}).get(target_uid)
                if isinstance(target_uid, str)
                else None
            )
            if target is None:
                unresolved.append(
                    {
                        "source_uid": source_uid,
                        "target": page_key,
                        "anchor": anchor,
                        "reason": "missing_target",
                    }
                )
                continue
            target_uid = str(target["uid"])
            if anchor and anchor.casefold() not in heading_index.get(target_uid, set()):
                unresolved.append(
                    {
                        "source_uid": source_uid,
                        "target": page_key,
                        "target_uid": target_uid,
                        "anchor": anchor,
                        "reason": "missing_anchor",
                    }
                )
                continue
            edge = {
                "source_uid": source_uid,
                "target_uid": target_uid,
                "anchor": anchor,
            }
            edges.append(edge)
            reverse.setdefault(target_uid, []).append(source_uid)
            if anchor:
                anchors.setdefault(target_uid, []).append(anchor)

    payload: dict[str, Any] = {
        "schema": SCHEMA,
        "generated_at": _now_iso(),
        "registry_generation": int(state.get("generation") or 0),
        "edge_count": len(edges),
        "unresolved_count": len(unresolved),
        "edges": edges,
        "reverse": {
            key: sorted(set(values)) for key, values in sorted(reverse.items())
        },
        "anchors": {
            key: sorted(set(values)) for key, values in sorted(anchors.items())
        },
        "unresolved": unresolved,
    }
    canonical = json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    payload["content_sha256"] = hashlib.sha256(canonical).hexdigest()
    if write:
        path = root / "runtime" / "librarian" / "uid-link-index.json"
        write_sealed_json(path, payload, backup=True)
    return payload
