"""Wiki link 抽出・正規化・fuzzy match のユーティリティ。

Canonical documents and capture state share these helpers.
"""

import re

from chronovisor.pages.frontmatter import parse as parse_frontmatter

# [[target]] / [[target|label]] / [[target#section]] にマッチ。
# 改行・ネスト・空リンクは除外。
WIKI_LINK_RE = re.compile(r"\[\[([^\[\]\n]+?)\]\]")

# fenced code block (``` ... ```), closed blocks only. Use
# fenced_code_spans() when unclosed/truncated fences must be protected too.
_FENCED_CODE_RE = re.compile(r"```.*?```", re.DOTALL)
# inline code (`...`)
_INLINE_CODE_RE = re.compile(r"`[^`\n]*`")


def normalize_link_target(link: str) -> str:
    """[[page|label]] / [[page#section]] から target page_id を抽出。

    ``[[foo|Display]]`` → ``foo``
    ``[[foo#heading]]`` → ``foo``
    ``[[foo|label#sec]]`` → ``foo``
    """
    return link.split("|")[0].split("#")[0].strip()


def strip_fences_and_frontmatter(text: str) -> str:
    """frontmatter + code fence + inline code を取り除いた body を返す。

    auto-fix で code 内のリンクを誤って書き換えないために必須。
    """
    _meta, text = parse_frontmatter(text)
    for start, end in reversed(fenced_code_spans(text)):
        text = text[:start] + text[end:]
    text = _INLINE_CODE_RE.sub("", text)
    return text


def fenced_code_spans(text: str) -> list[tuple[int, int]]:
    """Return spans for fenced code blocks, including an unclosed EOF fence.

    LLM output is often truncated. Treating a dangling opening fence as prose
    makes link extraction / auto-fix corrupt examples like ``data[[1]]``.
    """
    spans: list[tuple[int, int]] = []
    pos = 0
    open_at: int | None = None
    n = len(text)
    while pos < n:
        idx = text.find("```", pos)
        if idx == -1:
            break
        if open_at is None:
            open_at = idx
        else:
            spans.append((open_at, idx + 3))
            open_at = None
        pos = idx + 3
    if open_at is not None:
        spans.append((open_at, n))
    return spans


def extract_wiki_links(text: str, strip: bool = True) -> list[str]:
    """``[[...]]`` の中身をそのまま抽出 (``id|label`` / ``id#sec`` 形式を含む)。

    Args:
        text: markdown 本文
        strip: True なら frontmatter / code fence / inline code を除外してから抽出
    """
    body = strip_fences_and_frontmatter(text) if strip else text
    return WIKI_LINK_RE.findall(body)
