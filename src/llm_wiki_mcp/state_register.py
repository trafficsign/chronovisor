"""Always-on working-memory state register."""

from __future__ import annotations

import re
from pathlib import Path

from llm_wiki_mcp.frontmatter import parse
from llm_wiki_mcp.wiki import SYSTEM_DIR

STATE_PAGE_ID = "current-state"
STATE_PAGE = SYSTEM_DIR / f"{STATE_PAGE_ID}.md"


def _strip_heading_noise(text: str) -> str:
    lines: list[str] = []
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            if lines and lines[-1] != "":
                lines.append("")
            continue
        if stripped.startswith("#"):
            continue
        lines.append(line.rstrip())
    return re.sub(r"\n{3,}", "\n\n", "\n".join(lines)).strip()


def load_state_register(path: Path = STATE_PAGE, *, max_chars: int = 1600) -> str:
    """Return compact current-state text, or an empty string if unavailable."""
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        return ""
    _meta, body = parse(text)
    body = _strip_heading_noise(body)
    if not body:
        return ""
    if len(body) > max_chars:
        body = body[:max_chars].rstrip() + "..."
    return body


def format_state_context(
    *,
    host: str,
    cwd: str = "",
    max_chars: int = 1600,
    path: Path = STATE_PAGE,
) -> str:
    """Build a small context block injected outside the recall gate."""
    body = load_state_register(path, max_chars=max_chars)
    if not body:
        return ""
    lines = [
        "[WORKING_MEMORY]",
        "Current state from LLM Wiki. Use only when relevant; do not overfit casual chatter.",
        f"source={STATE_PAGE_ID}",
    ]
    if host:
        lines.append(f"host={host}")
    if cwd:
        lines.append(f"cwd={cwd}")
    lines.append("content:")
    lines.append(body)
    lines.append("[/WORKING_MEMORY]")
    return "\n".join(lines)


def should_inject_state(host: str) -> bool:
    return host in {"codex", "claude-code", "generic"}
