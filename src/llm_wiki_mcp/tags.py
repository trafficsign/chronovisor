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
violations, prefix violations) is handled by ``wiki_check`` lint, not by
the apply path.
"""

from __future__ import annotations

import os
import re
from datetime import date

from llm_wiki_mcp.link_fix import atomic_write
from llm_wiki_mcp.page_mutation import wiki_mutation_lock
from llm_wiki_mcp.wiki import SYSTEM_DIR


# ---------------------------------------------------------------------------
# Taxonomy v0.1 — seed tags
# ---------------------------------------------------------------------------


SEED_TAGS: dict[str, list[str]] = {
    "d/": [
        "d/ai-industry",
        "d/hardware",
        "d/geopolitics",
        "d/health",
        "d/finance",
        "d/personal-strategy",
        "d/tools-config",
        "d/japan",
        "d/theory",
        "d/paranormal",
    ],
    "t/": [
        "t/analysis",
        "t/chat-log",
        "t/howto",
        "t/reference",
        "t/decision",
        "t/scenario",
        "t/news-summary",
    ],
    "s/": [
        "s/2026",
        "s/evergreen",
        "s/historical",
    ],
}


# Axis -> (min_count, max_count). ``s/`` uses None upper bound conventions
# but here we keep an explicit (1, 1) so all four limits are uniform.
AXIS_LIMITS: dict[str, tuple[int, int]] = {
    "d/": (1, 3),
    "t/": (1, 1),
    "s/": (1, 1),
}


VALID_PREFIXES: tuple[str, ...] = ("d/", "t/", "s/")


# ``s/`` is the only axis where a digit-leading body (years) is allowed.
# Other axes must start with a letter.
_AXIS_BODY_REGEX = {
    "d/": re.compile(r"^[a-z][a-z0-9-]*$"),
    "t/": re.compile(r"^[a-z][a-z0-9-]*$"),
    "s/": re.compile(r"^[a-z0-9][a-z0-9-]*$"),
}

_MAX_WORDS_PER_TAG = 2  # rule 3: 3+ words go to keywords, not tags


# ---------------------------------------------------------------------------
# Form validation
# ---------------------------------------------------------------------------


def validate_tag(tag: str) -> tuple[bool, str]:
    """Validate a single tag against form rules (1-6 from generation rules v1.0).

    Returns ``(is_valid, reason)``. ``reason`` is empty for valid tags.
    """
    if not isinstance(tag, str):
        return False, "tag must be a string"

    if not tag or tag != tag.strip():
        return False, "empty or has surrounding whitespace"

    matched_prefix: str | None = None
    for prefix in VALID_PREFIXES:
        if tag.startswith(prefix):
            matched_prefix = prefix
            break
    if matched_prefix is None:
        return False, f"missing required prefix (one of {VALID_PREFIXES})"

    body = tag[len(matched_prefix):]
    if not body:
        return False, "empty body after prefix"

    if not _AXIS_BODY_REGEX[matched_prefix].fullmatch(body):
        if matched_prefix == "s/":
            return False, "s/ body must be ASCII kebab-case (digits ok)"
        return False, f"{matched_prefix} body must be ASCII kebab-case starting with a letter"

    # Word count: hyphens delimit words. ``personal-strategy`` is 2 words.
    word_count = len([w for w in body.split("-") if w])
    if word_count > _MAX_WORDS_PER_TAG:
        return False, f"too many words (max {_MAX_WORDS_PER_TAG}); use keywords instead"

    return True, ""


def parse_tags(tags: list[str]) -> dict[str, list[str]]:
    """Group tags by their axis prefix.

    Tags that don't match any known prefix go under the empty-string key
    so callers can surface them as form violations without losing the
    raw value.
    """
    out: dict[str, list[str]] = {p: [] for p in VALID_PREFIXES}
    out[""] = []
    for tag in tags:
        if not isinstance(tag, str):
            out[""].append(repr(tag))
            continue
        for prefix in VALID_PREFIXES:
            if tag.startswith(prefix):
                out[prefix].append(tag)
                break
        else:
            out[""].append(tag)
    return out


def validate_axis_counts(parsed: dict[str, list[str]]) -> list[str]:
    """Return a list of human-readable violation messages.

    Empty list = the tag set satisfies axis count constraints. The check
    is informational at apply time — wiki_check lint enforces it.
    """
    violations: list[str] = []
    for prefix, (min_n, max_n) in AXIS_LIMITS.items():
        n = len(parsed.get(prefix, []))
        if n < min_n:
            violations.append(
                f"{prefix} has {n} tag(s); requires at least {min_n}"
            )
        if n > max_n:
            violations.append(
                f"{prefix} has {n} tag(s); allows at most {max_n}"
            )
    unknown = parsed.get("", [])
    if unknown:
        violations.append(
            f"unknown prefix: {', '.join(unknown[:5])}"
            + ("..." if len(unknown) > 5 else "")
        )
    return violations


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
        from llm_wiki_mcp.embedding import most_similar
        result = most_similar(new_tag, same_axis, threshold=threshold)
    except Exception:
        return new_tag
    if result is None:
        return new_tag
    return result[0]


# ---------------------------------------------------------------------------
# Audit log: ~/.wiki/system/tag-changelog.md
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
        with wiki_mutation_lock():
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
        # Audit failure must never break ingest. wiki_check can re-derive
        # the tag set from page frontmatter if the changelog gets out of
        # sync.
        pass
