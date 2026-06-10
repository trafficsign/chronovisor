"""Small path module shared by recall components without import cycles."""

from __future__ import annotations

from llm_wiki_mcp.wiki import WIKI_ROOT


RECALL_DIR = WIKI_ROOT / "recall"
