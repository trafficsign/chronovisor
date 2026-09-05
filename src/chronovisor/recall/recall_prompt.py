"""Normalize host transport envelopes into the user's effective recall prompt.

Host applications may prepend ambient UI state or append system notifications
to the string delivered to a UserPromptSubmit hook.  Recall must operate on the
effective request, not on those transport details.  The allowlist below stays
deliberately narrow so user-authored XML or Markdown is not stripped merely
because it looks structured.
"""

from __future__ import annotations

import json
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

QUESTION_REPLY_OPEN = "<send_user_message_question_reply>"
QUESTION_REPLY_CLOSE = "</send_user_message_question_reply>"
# ponytail: bounded host payload; raise caps only after observing a larger
# authenticated host schema.
QUESTION_REPLY_MAX_CHARS = 16_384
QUESTION_REPLY_MAX_ITEMS = 8
QUESTION_REPLY_MAX_FIELD_CHARS = 8_192
QUESTION_REPLY_KEYS = frozenset({"questionItemId", "question", "answer"})


def normalize_recall_prompt(prompt: str) -> tuple[str, list[str]]:
    """Return the effective user request and an auditable list of removals.

    The one recognized question-reply envelope is projected to its question
    and answer text before the generic transport cleanup runs.
    """

    question_reply = _normalize_question_reply_envelope(prompt)
    if question_reply is not None:
        return question_reply, ["extracted question reply envelope"]

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


def _normalize_question_reply_envelope(prompt: str) -> str | None:
    """Extract human text from the one known question-reply transport shape."""

    if len(prompt) > QUESTION_REPLY_MAX_CHARS:
        return None
    envelope = prompt.strip()
    if not (
        envelope.startswith(QUESTION_REPLY_OPEN)
        and envelope.endswith(QUESTION_REPLY_CLOSE)
    ):
        return None
    payload = envelope[
        len(QUESTION_REPLY_OPEN) : -len(QUESTION_REPLY_CLOSE)
    ].strip()
    if not (payload.startswith("[") and payload.endswith("]")):
        return None
    try:
        items = json.loads(payload, object_pairs_hook=_reject_duplicate_json_keys)
    except (json.JSONDecodeError, RecursionError, ValueError):
        return None
    if not isinstance(items, list) or not items or len(items) > QUESTION_REPLY_MAX_ITEMS:
        return None

    parts: list[str] = []
    for item in items:
        if not isinstance(item, dict) or set(item) != QUESTION_REPLY_KEYS:
            return None
        question_item_id = item["questionItemId"]
        question = item["question"]
        answer = item["answer"]
        if not (
            isinstance(question_item_id, str)
            and isinstance(question, str)
            and isinstance(answer, str)
        ):
            return None
        if not question_item_id.strip():
            return None
        question = question.strip()
        answer = answer.strip()
        if not question or not answer:
            return None
        if (
            len(question_item_id) > QUESTION_REPLY_MAX_FIELD_CHARS
            or len(question) > QUESTION_REPLY_MAX_FIELD_CHARS
            or len(answer) > QUESTION_REPLY_MAX_FIELD_CHARS
        ):
            return None
        parts.extend((question, answer))
    return "\n".join(parts)


def _reject_duplicate_json_keys(
    pairs: list[tuple[str, object]],
) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate JSON object key")
        result[key] = value
    return result
