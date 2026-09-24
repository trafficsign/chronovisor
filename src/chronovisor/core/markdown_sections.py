"""Lossless H1/H2 Markdown section partitioning shared by ingest and librarian."""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass


@dataclass(frozen=True)
class MarkdownSection:
    """One byte-complete Markdown section from an existing page."""

    start_line: int
    end_line: int
    heading: str | None
    content: str
    sha256: str


MARKDOWN_SECTION_HEADING_RE = re.compile(
    r"^ {0,3}(?P<marks>#{1,2})[\t ]+(?P<title>.*?)(?:[\t ]+#+)?[\t ]*(?:\n)?$"
)
MARKDOWN_FENCE_RE = re.compile(r"^ {0,3}(?P<marker>`{3,}|~{3,})")


def markdown_sections(text: str, *, max_level: int = 2) -> tuple[MarkdownSection, ...]:
    """Split Markdown at H1/H2 boundaries outside fenced code blocks.

    Sections are a lossless partition of ``text``: concatenating their
    ``content`` fields reproduces the exact page bytes.  A pre-heading region
    (normally frontmatter) is represented as a section with ``heading=None``.
    Lower-level headings remain inside their enclosing H2 section so selection
    never detaches a subsection from its top-level semantic unit.
    ``max_level=3`` also splits at H3 (used to break one oversized section).
    """

    if not text:
        return ()
    heading_re = (
        MARKDOWN_SECTION_HEADING_RE
        if max_level == 2
        else re.compile(
            MARKDOWN_SECTION_HEADING_RE.pattern.replace("#{1,2}", f"#{{1,{max_level}}}")
        )
    )
    lines = text.splitlines(keepends=True)
    heading_rows: list[tuple[int, str]] = []
    fence_char: str | None = None
    fence_width = 0
    for index, line in enumerate(lines):
        fence_match = MARKDOWN_FENCE_RE.match(line)
        if fence_match is not None:
            marker = fence_match.group("marker")
            if fence_char is None:
                fence_char = marker[0]
                fence_width = len(marker)
            elif (
                marker[0] == fence_char
                and len(marker) >= fence_width
                and re.fullmatch(
                    r"[\t ]*(?:\r?\n)?",
                    line[fence_match.end() :],
                )
                is not None
            ):
                fence_char = None
                fence_width = 0
            continue
        if fence_char is not None:
            continue
        heading_match = heading_re.match(line)
        if heading_match is not None:
            heading_rows.append(
                (
                    index,
                    f"{heading_match.group('marks')} "
                    f"{heading_match.group('title').strip()}",
                )
            )

    boundaries = [index for index, _heading in heading_rows]
    if not boundaries or boundaries[0] != 0:
        boundaries.insert(0, 0)
    boundaries.append(len(lines))
    headings_by_index = dict(heading_rows)
    sections: list[MarkdownSection] = []
    for start, end in zip(boundaries[:-1], boundaries[1:], strict=True):
        content = "".join(lines[start:end])
        if not content:
            continue
        sections.append(
            MarkdownSection(
                start_line=start + 1,
                end_line=end,
                heading=headings_by_index.get(start),
                content=content,
                sha256=hashlib.sha256(content.encode("utf-8")).hexdigest(),
            )
        )
    return tuple(sections)


def split_children(text: str) -> list[str]:
    """Return a hub page's child filenames (empty for ordinary pages)."""

    from chronovisor.core import frontmatter

    try:
        meta, _body = frontmatter.parse(text)
    except Exception:
        return []
    value = meta.get("split_children")
    return [str(item) for item in value] if isinstance(value, list) else []
