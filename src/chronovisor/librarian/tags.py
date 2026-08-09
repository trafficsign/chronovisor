"""Tag taxonomy v0.1 + tag generation rules v1.0.

This module owns the tag lifecycle:

    raw page text  -- LLM generation -->  candidate tags
                                            |
                                            v
                                    validate_tag (form rules)
                                            |
                                            v
                                  dedupe_with_existing  (>= 0.80 cosine similarity
                                            |            to an existing tag)
                                            v
                                  validate_axis_counts (d/ 1-3, t/ 1, s/ 1)
                                            |
                                            v
                                    record_new_tag (append to tag-changelog.md
                                                      for newly accepted tags)

Failures are *soft* by default — a malformed tag drops itself rather than
killing the whole page. Ingest can't easily replay an LLM call, and tags
are a best-effort overlay on top of the body. Strict enforcement (count
violations, prefix violations) is handled by ``chronovisor_check`` lint, not by
the apply path.
"""

from __future__ import annotations

import os
from datetime import date

from chronovisor.core.link_fix import atomic_write
from chronovisor.core.page_mutation import chronovisor_mutation_lock
from chronovisor.core.store import SYSTEM_DIR
from chronovisor.core.tag_rules import (
    AXIS_LIMITS,
    SEED_TAGS,
    VALID_PREFIXES,
    parse_tags,
    validate_axis_counts,
    validate_tag,
)

__all__ = [
    "AXIS_LIMITS",
    "SEED_TAGS",
    "VALID_PREFIXES",
    "dedupe_with_existing",
    "parse_tags",
    "record_new_tag",
    "validate_axis_counts",
    "validate_tag",
]


# ---------------------------------------------------------------------------
# Existing-tag preference (rule 7)
# ---------------------------------------------------------------------------


def dedupe_with_existing(
    new_tag: str,
    existing: list[str],
    threshold: float = 0.80,
) -> str:
    """Return an existing tag if its similarity to ``new_tag`` is >= threshold,
    else return ``new_tag`` unchanged.

    Comparison only considers tags from the SAME axis as ``new_tag``: a
    domain tag should never collapse into a type tag even if their bodies
    happen to embed close together.

    Soft-fails to ``new_tag`` if the embedding service is unreachable —
    we'd rather over-create tags than abort an ingest because Ollama
    blinked.
    """
    valid, _reason = validate_tag(new_tag)
    if not valid:
        return new_tag  # let the caller's form validator handle it

    # Restrict candidates to the same axis as new_tag.
    matched_prefix: str | None = None
    for prefix in VALID_PREFIXES:
        if new_tag.startswith(prefix):
            matched_prefix = prefix
            break
    if matched_prefix is None:
        return new_tag
    same_axis = [t for t in existing if t.startswith(matched_prefix) and t != new_tag]
    if not same_axis:
        return new_tag

    try:
        from chronovisor.search.embedding import most_similar
        result = most_similar(new_tag, same_axis, threshold=threshold)
    except Exception:
        return new_tag
    if result is None:
        return new_tag
    return result[0]


# ---------------------------------------------------------------------------
# Audit log: ~/.chronovisor/system/tag-changelog.md
# ---------------------------------------------------------------------------


_TAG_CHANGELOG_HEADER = """\
---
title: Tag Changelog
updated: {today}
---

# Tag Changelog

Append-only audit log of every newly minted tag, recorded at the moment
ingest first accepts it (after dedupe_with_existing fails to find a
sufficiently similar existing tag).

Format: `YYYY-MM-DD | <tag> | <reason>`
"""


def _changelog_path():
    return SYSTEM_DIR / "tag-changelog.md"


def record_new_tag(tag: str, reason: str = "ingest auto-gen") -> None:
    """Append ``tag`` to the changelog. Best-effort — silently skips on IO error.

    Idempotent within a single day: if the same tag already appears in
    today's lines, we don't add a duplicate. Across days the same tag
    can re-appear if it was somehow dropped from the corpus and
    re-introduced (a separate, deliberate event).
    """
    try:
        path = _changelog_path()
        today = date.today().isoformat()
        line = f"- {today} | {tag} | {reason}"
        with chronovisor_mutation_lock():
            path.parent.mkdir(parents=True, exist_ok=True)
            if not path.exists():
                atomic_write(path, _TAG_CHANGELOG_HEADER.format(today=today))

            existing = path.read_text(encoding="utf-8")
            # Header creation, same-day dedup, and append share the Wiki writer
            # lock. The final append is one O_APPEND write, so even an older
            # non-cooperating process cannot partially overwrite an entry.
            prefix = f"- {today} | {tag} |"
            if any(ln.startswith(prefix) for ln in existing.splitlines()[-50:]):
                return
            payload = ("" if existing.endswith("\n") else "\n") + line + "\n"
            with path.open("a", encoding="utf-8") as handle:
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
    except Exception:
        # Audit failure must never break ingest. chronovisor_check can re-derive
        # the tag set from page frontmatter if the changelog gets out of
        # sync.
        pass
