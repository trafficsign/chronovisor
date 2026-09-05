#!/usr/bin/env python3.14
"""Build a complete, licensed UDC Summary package from the official website.

The legacy UDC Summary linked-data endpoint is temporarily unavailable, but the
official multilingual browser still renders the complete public schedule as a
deterministic JavaScript tree.  This importer snapshots that public schedule,
preserves its hierarchy and English/Japanese labels, and emits the package
schema consumed by Chronovisor.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import html
import json
import re
import sys
import urllib.parse
import urllib.request
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

SCHEMA = "chronovisor.udcs-package.v1"
BASE_URL = "https://udcsummary.info/php/index.php"
ROOT_TAGS = ("--", "---", "0", "1", "2", "3", "5", "6", "7", "8", "9")
MINIMUM_COMPLETE_CONCEPTS = 2_500
ATTRIBUTION = (
    "Multilingual Universal Decimal Classification Summary "
    "(UDCC Publication No. 088), © UDC Consortium, 2013"
)
LICENSE = "CC BY-SA 3.0"
_TAG_RE = re.compile(r"<[^>]+>")


class ImportError(RuntimeError):
    """Raised when the official export cannot be parsed completely."""


def _fetch(tag: str, language: str) -> tuple[str, str]:
    query = urllib.parse.urlencode({"tag": tag, "lang": language, "pr": "Y"})
    url = f"{BASE_URL}?{query}"
    request = urllib.request.Request(
        url,
        headers={"User-Agent": "Chronovisor-UDCS-Importer/1.0"},
    )
    with urllib.request.urlopen(request, timeout=60) as response:
        if response.status != 200:
            raise ImportError(f"{url} returned HTTP {response.status}")
        return url, response.read().decode("utf-8")


def _split_javascript_arguments(line: str) -> list[str]:
    start = line.find("d.add(")
    end = line.rfind(");")
    if start < 0 or end < 0:
        raise ImportError("malformed d.add row")
    payload = line[start + len("d.add(") : end]
    parts: list[str] = []
    current: list[str] = []
    quoted = False
    escaped = False
    for char in payload:
        if escaped:
            current.append(char)
            escaped = False
            continue
        if char == "\\" and quoted:
            current.append(char)
            escaped = True
            continue
        if char == "'":
            quoted = not quoted
            current.append(char)
            continue
        if char == "," and not quoted:
            parts.append("".join(current).strip())
            current = []
            continue
        current.append(char)
    parts.append("".join(current).strip())
    if quoted or len(parts) < 6:
        raise ImportError("unterminated or incomplete d.add row")
    return parts


def _javascript_string(token: str) -> str:
    try:
        value = ast.literal_eval(token)
    except (SyntaxError, ValueError) as exc:
        raise ImportError(f"invalid JavaScript string {token[:80]!r}") from exc
    if not isinstance(value, str):
        raise ImportError("expected a JavaScript string")
    return html.unescape(value).strip()


def _fallback_label(markup: str, notation: str) -> str:
    plain = html.unescape(_TAG_RE.sub("", markup)).replace("\u00a0", " ").strip()
    if plain.startswith(notation):
        plain = plain[len(notation) :].strip()
    return re.sub(r"\s+", " ", plain)


def parse_tree(document: str) -> list[dict[str, Any]]:
    parsed: list[dict[str, Any]] = []
    for line in document.splitlines():
        if "d.add(" not in line:
            continue
        parts = _split_javascript_arguments(line)
        try:
            node_id = int(parts[0])
            parent_id = int(parts[1])
        except ValueError as exc:
            raise ImportError("non-integer UDC tree node") from exc
        notation = _javascript_string(parts[4])
        markup = _javascript_string(parts[3])
        label = _javascript_string(parts[5]) or _fallback_label(markup, notation)
        if not notation or not label:
            raise ImportError(f"UDC row {node_id} is missing notation/label")
        parsed.append(
            {
                "node_id": node_id,
                "parent_id": parent_id,
                "notation": notation,
                "label": label,
            }
        )
    if not parsed:
        raise ImportError("official UDC page contained no tree rows")
    ids = {int(row["node_id"]) for row in parsed}
    if len(ids) != len(parsed):
        raise ImportError("duplicate tree node id within UDC page")
    return parsed


def _concept_uri(notation: str) -> str:
    query = urllib.parse.urlencode({"tag": notation, "lang": "en", "pr": "Y"})
    return f"{BASE_URL}?{query}"


def build_package(*, release: str | None = None) -> dict[str, Any]:
    concepts: dict[str, dict[str, Any]] = {}
    sources: list[dict[str, str]] = []
    for root_tag in ROOT_TAGS:
        english_url, english_document = _fetch(root_tag, "en")
        japanese_url, japanese_document = _fetch(root_tag, "ja")
        english = parse_tree(english_document)
        japanese = parse_tree(japanese_document)
        japanese_labels = {
            str(row["notation"]): str(row["label"]) for row in japanese
        }
        notation_by_node = {
            int(row["node_id"]): str(row["notation"]) for row in english
        }
        for row in english:
            notation = str(row["notation"])
            parent_id = int(row["parent_id"])
            broader_notation = notation_by_node.get(parent_id)
            concept = {
                "uri": _concept_uri(notation),
                "notation": notation,
                "label_en": str(row["label"]),
                "label_ja": japanese_labels.get(notation) or None,
                "label_ja_source": (
                    "udcsummary.info"
                    if japanese_labels.get(notation)
                    else "missing"
                ),
                "broader_uri": (
                    _concept_uri(broader_notation) if broader_notation else None
                ),
                "source_url": _concept_uri(notation),
            }
            existing = concepts.get(notation)
            if existing is not None and existing != concept:
                raise ImportError(f"conflicting duplicate notation {notation!r}")
            concepts[notation] = concept
        sources.append(
            {
                "tag": root_tag,
                "english_url": english_url,
                "english_sha256": hashlib.sha256(
                    english_document.encode("utf-8")
                ).hexdigest(),
                "japanese_url": japanese_url,
                "japanese_sha256": hashlib.sha256(
                    japanese_document.encode("utf-8")
                ).hexdigest(),
            }
        )

    if len(concepts) < MINIMUM_COMPLETE_CONCEPTS:
        raise ImportError(
            f"incomplete UDC Summary: {len(concepts)} concepts, "
            f"expected at least {MINIMUM_COMPLETE_CONCEPTS}"
        )
    uris = [str(row["uri"]) for row in concepts.values()]
    if len(set(uris)) != len(uris):
        raise ImportError("UDC concept URIs are not unique")
    known_uris = set(uris)
    missing_parents = sorted(
        {
            str(row["broader_uri"])
            for row in concepts.values()
            if row["broader_uri"] and row["broader_uri"] not in known_uris
        }
    )
    if missing_parents:
        raise ImportError(f"UDC hierarchy has missing parents: {missing_parents[:5]}")

    japanese_count = sum(
        bool(row.get("label_ja")) for row in concepts.values()
    )
    return {
        "schema": SCHEMA,
        "release": release
        or f"udcs-official-web-{datetime.now(UTC).date().isoformat()}",
        "source_url": "https://udcsummary.info/",
        "source_kind": "official-multilingual-browser-snapshot",
        "license": LICENSE,
        "license_url": "https://creativecommons.org/licenses/by-sa/3.0/",
        "attribution": ATTRIBUTION,
        "complete": True,
        "concept_count": len(concepts),
        "japanese_label_count": japanese_count,
        "sources": sources,
        "concepts": [concepts[key] for key in sorted(concepts)],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    parser.add_argument("--release")
    args = parser.parse_args()
    try:
        payload = build_package(release=args.release)
    except Exception as exc:
        print(f"UDC Summary import failed: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
    args.output.parent.mkdir(parents=True, exist_ok=True)
    encoded = (
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")
    args.output.write_bytes(encoded)
    print(
        json.dumps(
            {
                "output": str(args.output),
                "sha256": hashlib.sha256(encoded).hexdigest(),
                "concepts": payload["concept_count"],
                "japanese_labels": payload["japanese_label_count"],
                "release": payload["release"],
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
