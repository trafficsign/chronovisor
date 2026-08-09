"""Pure tag taxonomy and validation rules."""

from __future__ import annotations

import re

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

AXIS_LIMITS: dict[str, tuple[int, int]] = {
    "d/": (1, 3),
    "t/": (1, 1),
    "s/": (1, 1),
}

VALID_PREFIXES: tuple[str, ...] = ("d/", "t/", "s/")

_AXIS_BODY_REGEX = {
    "d/": re.compile(r"^[a-z][a-z0-9-]*$"),
    "t/": re.compile(r"^[a-z][a-z0-9-]*$"),
    "s/": re.compile(r"^[a-z0-9][a-z0-9-]*$"),
}

_MAX_WORDS_PER_TAG = 2


def validate_tag(tag: str) -> tuple[bool, str]:
    """Validate a single tag against form rules."""
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

    body = tag[len(matched_prefix) :]
    if not body:
        return False, "empty body after prefix"

    if not _AXIS_BODY_REGEX[matched_prefix].fullmatch(body):
        if matched_prefix == "s/":
            return False, "s/ body must be ASCII kebab-case (digits ok)"
        return False, f"{matched_prefix} body must be ASCII kebab-case starting with a letter"

    word_count = len([word for word in body.split("-") if word])
    if word_count > _MAX_WORDS_PER_TAG:
        return False, f"too many words (max {_MAX_WORDS_PER_TAG}); use keywords instead"

    return True, ""


def parse_tags(tags: list[str]) -> dict[str, list[str]]:
    """Group tags by their axis prefix."""
    out: dict[str, list[str]] = {prefix: [] for prefix in VALID_PREFIXES}
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
    """Return human-readable axis-count violations."""
    violations: list[str] = []
    for prefix, (min_n, max_n) in AXIS_LIMITS.items():
        count = len(parsed.get(prefix, []))
        if count < min_n:
            violations.append(f"{prefix} has {count} tag(s); requires at least {min_n}")
        if count > max_n:
            violations.append(f"{prefix} has {count} tag(s); allows at most {max_n}")
    unknown = parsed.get("", [])
    if unknown:
        violations.append(
            f"unknown prefix: {', '.join(unknown[:5])}"
            + ("..." if len(unknown) > 5 else "")
        )
    return violations
