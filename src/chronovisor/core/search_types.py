"""Shared search data types and tokenization helpers."""

from __future__ import annotations

import re
from dataclasses import dataclass

from chronovisor.core.canonical_document import CanonicalDocumentError
from chronovisor.core.frontmatter import parse as parse_frontmatter


@dataclass
class ScoredPage:
    page_id: str
    title: str
    folder: str
    updated: str
    score: float
    snippet: str = ""
    status: str = "stable"
    superseded_by: str = ""
    page_type: str = "knowledge"
    sensitivity: str = "normal"
    content_sha256: str = ""
    uid: str = ""


_CJK_RANGES = (
    ("\u3040", "\u309f"),  # Hiragana
    ("\u30a0", "\u30ff"),  # Katakana
    ("\u4e00", "\u9fff"),  # CJK Unified Ideographs
    ("\u3400", "\u4dbf"),  # CJK Extension A
    ("\uff66", "\uff9f"),  # Halfwidth Katakana
)

def _is_cjk(ch: str) -> bool:
    return any(lo <= ch <= hi for lo, hi in _CJK_RANGES)


def tokenize(text: str) -> list[str]:
    """Tokenize text: ASCII words + CJK character bigrams."""
    try:
        _meta, text = parse_frontmatter(text)
    except (CanonicalDocumentError, UnicodeError):
        # Search queries are untrusted text, not canonical page documents.
        # A query that happens to start with ``---`` must remain searchable.
        pass
    text_lower = text.lower()

    tokens: list[str] = []
    for match in re.finditer(r"[a-z0-9_]+", text_lower):
        word = match.group()
        if len(word) >= 2:
            tokens.append(word)

    cjk_ranges = "".join(f"{lo}-{hi}" for lo, hi in _CJK_RANGES)
    for run in re.findall(rf"[{cjk_ranges}]+", text):
        if len(run) == 1:
            tokens.append(run)
        for idx in range(len(run) - 1):
            tokens.append(run[idx] + run[idx + 1])

    return tokens
