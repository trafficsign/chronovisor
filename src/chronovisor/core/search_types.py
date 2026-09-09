"""Shared search data types and tokenization helpers."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass

from chronovisor.core.canonical_document import CanonicalDocumentError
from chronovisor.core.frontmatter import parse as parse_frontmatter
from chronovisor.core.page_identity import normalize_page_uid


@dataclass(frozen=True)
class SemanticEvidence:
    """Identity of a scored document; resolve text only against this source digest."""

    page_id: str
    doc_id: str
    kind: str
    ordinal: int
    source_sha256: str
    page_uid: str
    generation_id: str
    score: float


def parse_semantic_evidence(
    values: object, *, page_id: str, generation_id: str | None = None
) -> tuple[SemanticEvidence, ...]:
    """Validate bounded wire identities without treating search keys as source text."""
    if not isinstance(values, list) or len(values) > 3:
        raise ValueError("invalid semantic evidence")
    result: list[SemanticEvidence] = []
    seen: set[str] = set()
    for value in values:
        try:
            if not isinstance(value, dict):
                raise ValueError
            item = SemanticEvidence(**value)
            if (
                not isinstance(item.page_id, str) or not item.page_id or item.page_id != page_id
                or not isinstance(item.doc_id, str) or item.doc_id in seen
                or not isinstance(item.page_uid, str)
                or (item.page_uid and normalize_page_uid(item.page_uid) != item.page_uid)
                or not isinstance(item.generation_id, str) or not item.generation_id.strip()
                or (generation_id is not None and item.generation_id != generation_id)
                or not isinstance(item.source_sha256, str)
                or not re.fullmatch(r"[0-9a-f]{64}", item.source_sha256)
                or type(item.ordinal) is not int
                or item.kind not in {"page", "question", "chunk", "section-v1"}
                or (item.kind == "page" and item.ordinal != -1)
                or (
                    item.kind in {"question", "chunk"}
                    and not 0 <= item.ordinal < 8
                )
                or (item.kind == "section-v1" and item.ordinal < 0)
                or type(item.score) not in {float, int} or not math.isfinite(item.score)
            ):
                raise ValueError
            if item.kind == "section-v1":
                if re.fullmatch(r"page-record:[0-9a-f]{64}", item.doc_id) is None:
                    raise ValueError
            else:
                suffix = (
                    ""
                    if item.kind == "page"
                    else f"#{'c' if item.kind == 'chunk' else 'q'}{item.ordinal}"
                )
                if item.doc_id != (item.page_uid or item.page_id) + suffix:
                    raise ValueError
            if result and (item.page_uid, item.source_sha256, item.generation_id) != (
                result[0].page_uid, result[0].source_sha256, result[0].generation_id
            ):
                raise ValueError
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError("invalid semantic evidence") from exc
        result.append(item)
        seen.add(item.doc_id)
    return tuple(result)


def merge_evidence(
    *groups: tuple[SemanticEvidence, ...],
) -> tuple[SemanticEvidence, ...]:
    """Keep a bounded, coherent set of document matches across retrieval channels."""
    ranked = sorted(
        (item for group in groups for item in group),
        key=lambda x: x.score,
        reverse=True,
    )
    if not ranked:
        return ()
    first = ranked[0]
    unique: dict[str, SemanticEvidence] = {}
    for item in ranked:
        if (item.page_id, item.page_uid, item.source_sha256, item.generation_id) == (
            first.page_id,
            first.page_uid,
            first.source_sha256,
            first.generation_id,
        ):
            unique.setdefault(item.doc_id, item)
    rows = list(unique.values())
    # Page/question documents are retrieval keys. Keep the winning key, then
    # prefer scored source chunks over redundant generated questions.
    return tuple(
        [rows[0], *(item for item in rows[1:] if item.kind in {"chunk", "section-v1"})]
    )[:3]


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
    evidence: tuple[SemanticEvidence, ...] = ()


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
