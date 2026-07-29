"""Shared search data types and tokenization helpers."""

from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass
class ScoredPage:
    page_id: str
    title: str
    folder: str
    updated: str
    score: float
    snippet: str = ""
    status: str = "active"
    superseded_by: str = ""
    page_type: str = "knowledge"
    sensitivity: str = "normal"


_CJK_RANGES = (
    ("\u3040", "\u309f"),  # Hiragana
    ("\u30a0", "\u30ff"),  # Katakana
    ("\u4e00", "\u9fff"),  # CJK Unified Ideographs
    ("\u3400", "\u4dbf"),  # CJK Extension A
    ("\uff66", "\uff9f"),  # Halfwidth Katakana
)

_FRONTMATTER_RE = re.compile(r"^---\n.*?\n---\n", re.DOTALL)


def _is_cjk(ch: str) -> bool:
    for lo, hi in _CJK_RANGES:
        if lo <= ch <= hi:
            return True
    return False


def tokenize(text: str) -> list[str]:
    """Tokenize text: ASCII words + CJK character bigrams."""
    text = _FRONTMATTER_RE.sub("", text)
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
