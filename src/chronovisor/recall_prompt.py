"""Normalize host transport envelopes into the user's effective recall prompt.

Host applications may prepend ambient UI state or append system notifications
to the string delivered to a UserPromptSubmit hook.  Recall must operate on the
effective request, not on those transport details.  The allowlist below stays
deliberately narrow so user-authored XML or Markdown is not stripped merely
because it looks structured.
"""

from __future__ import annotations

import re


SYSTEM_ENVELOPE_RE = re.compile(
    r"^\s*<(task-notification|system-reminder|system-notification)\b",
    re.IGNORECASE,
)
SYSTEM_BLOCK_RE = re.compile(
    r"(?ims)(^|\n)\s*"
    r"<(task-notification|system-reminder|system-notification)\b.*?</\2>\s*"
)
SYSTEM_BLOCK_TO_END_RE = re.compile(
    r"(?ims)(^|\n)\s*"
    r"<(task-notification|system-reminder|system-notification)\b.*\Z"
)
IN_APP_BROWSER_BLOCK_RE = re.compile(
    r"(?ims)(^|\n)\s*"
    r"<in-app-browser-context\b.*?</in-app-browser-context>\s*"
)
IN_APP_BROWSER_BLOCK_TO_END_RE = re.compile(
    r"(?ims)(^|\n)\s*<in-app-browser-context\b.*\Z"
)
RECALL_CONTEXT_BLOCK_RE = re.compile(
    r"(?ms)(^|\n)\s*\[RECALL_CONTEXT\].*?\[/RECALL_CONTEXT\]\s*"
)
RECALL_CONTEXT_TO_END_RE = re.compile(
    r"(?ms)(^|\n)\s*\[RECALL_CONTEXT\].*\Z"
)
WORKING_MEMORY_BLOCK_RE = re.compile(
    r"(?ms)(^|\n)\s*\[WORKING_MEMORY\].*?\[/WORKING_MEMORY\]\s*"
)
WORKING_MEMORY_TO_END_RE = re.compile(
    r"(?ms)(^|\n)\s*\[WORKING_MEMORY\].*\Z"
)
CODEX_INTERNAL_SUGGESTION_RE = re.compile(
    r"^\s*#\s*Overview\s+Generate\s+0\s+to\s+3\s+"
    r"hyperpersonalized\s+suggestions\b",
    re.IGNORECASE,
)
CODEX_INTERNAL_SUGGESTION_BLOCK_RE = re.compile(
    r"(?ims)(^|\n)\s*#\s*Overview\s+Generate\s+0\s+to\s+3\s+"
    r"hyperpersonalized\s+suggestions\b.*\Z"
)
CODEX_USER_REQUEST_MARKER_RE = re.compile(
    r"(?im)^\s*##\s*My request for Codex:\s*$"
)
HEARTBEAT_BLOCK_RE = re.compile(
    r"(?ims)^\s*<heartbeat\b[^>]*>.*?"
    r"<instructions\b[^>]*>(?P<instructions>.*?)</instructions>.*?"
    r"</heartbeat>\s*$"
)
HEARTBEAT_ENVELOPE_RE = re.compile(r"^\s*<heartbeat\b", re.IGNORECASE)


def normalize_recall_prompt(prompt: str) -> tuple[str, list[str]]:
    """Return the effective user request and an auditable list of removals."""

    cleaned = prompt
    reasons: list[str] = []

    heartbeat = HEARTBEAT_BLOCK_RE.fullmatch(cleaned)
    if heartbeat:
        cleaned = heartbeat.group("instructions")
        reasons.append("extracted automation instructions")
    elif HEARTBEAT_ENVELOPE_RE.match(cleaned):
        # Session query history stores compacted strings.  An old heartbeat can
        # therefore be missing its closing tag and must not survive forever as
        # conversational context.
        return "", ["stripped incomplete automation heartbeat"]

    cleaned, removed = _strip_block(cleaned, SYSTEM_BLOCK_RE)
    if removed:
        reasons.append("stripped system notification block")
    cleaned, removed_to_end = _strip_block(cleaned, SYSTEM_BLOCK_TO_END_RE)
    if removed_to_end and "stripped system notification block" not in reasons:
        reasons.append("stripped system notification block")

    cleaned, removed = _strip_block(cleaned, IN_APP_BROWSER_BLOCK_RE)
    if removed:
        reasons.append("stripped in-app browser context")
    cleaned, removed_to_end = _strip_block(
        cleaned, IN_APP_BROWSER_BLOCK_TO_END_RE
    )
    if removed_to_end and "stripped in-app browser context" not in reasons:
        reasons.append("stripped in-app browser context")

    cleaned, removed = _strip_block(cleaned, RECALL_CONTEXT_BLOCK_RE)
    if removed:
        reasons.append("stripped recall context block")
    cleaned, removed_to_end = _strip_block(cleaned, RECALL_CONTEXT_TO_END_RE)
    if removed_to_end and "stripped recall context block" not in reasons:
        reasons.append("stripped recall context block")

    cleaned, removed = _strip_block(cleaned, WORKING_MEMORY_BLOCK_RE)
    if removed:
        reasons.append("stripped working memory block")
    cleaned, removed_to_end = _strip_block(cleaned, WORKING_MEMORY_TO_END_RE)
    if removed_to_end and "stripped working memory block" not in reasons:
        reasons.append("stripped working memory block")

    cleaned, removed = _strip_block(
        cleaned, CODEX_INTERNAL_SUGGESTION_BLOCK_RE
    )
    if removed:
        reasons.append("stripped codex internal suggestion block")

    markers = list(CODEX_USER_REQUEST_MARKER_RE.finditer(cleaned))
    if markers:
        cleaned = cleaned[markers[-1].end() :]
        reasons.append("extracted codex user request")

    return re.sub(r"\n{3,}", "\n\n", cleaned).strip(), reasons


def _strip_block(text: str, pattern: re.Pattern[str]) -> tuple[str, bool]:
    cleaned, count = pattern.subn("\n", text)
    return cleaned, count > 0
