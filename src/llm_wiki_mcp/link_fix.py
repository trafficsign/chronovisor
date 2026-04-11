"""Wiki link 抽出・正規化・fuzzy match のユーティリティ。

server.py / lint.py / scripts/fix_broken_links.py から共通で利用される。
"""

import difflib
import os
import re
import tempfile
from pathlib import Path


# [[target]] / [[target|label]] / [[target#section]] にマッチ。
# 改行・ネスト・空リンクは除外。
WIKI_LINK_RE = re.compile(r"\[\[([^\[\]\n]+?)\]\]")

# frontmatter delimiter
_FRONTMATTER_RE = re.compile(r"^---\n.*?\n---\n", re.DOTALL)
# fenced code block (``` ... ```)
_FENCED_CODE_RE = re.compile(r"```.*?```", re.DOTALL)
# inline code (`...`)
_INLINE_CODE_RE = re.compile(r"`[^`\n]*`")

FUZZY_THRESHOLD = 0.88


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
    text = _FRONTMATTER_RE.sub("", text, count=1)
    text = _FENCED_CODE_RE.sub("", text)
    text = _INLINE_CODE_RE.sub("", text)
    return text


def extract_wiki_links(text: str, strip: bool = True) -> list[str]:
    """``[[...]]`` の中身をそのまま抽出 (``id|label`` / ``id#sec`` 形式を含む)。

    Args:
        text: markdown 本文
        strip: True なら frontmatter / code fence / inline code を除外してから抽出
    """
    body = strip_fences_and_frontmatter(text) if strip else text
    return WIKI_LINK_RE.findall(body)


def extract_targets(text: str, strip: bool = True) -> list[str]:
    """``[[...]]`` を normalize し、target page_id のリストで返す。

    重複は保持 (一ページ内での複数参照をカウント可能にするため)。
    """
    return [normalize_link_target(link) for link in extract_wiki_links(text, strip=strip)]


def find_fuzzy_match(
    target: str, all_ids: set[str], threshold: float = FUZZY_THRESHOLD
) -> str | None:
    """target page_id に近い実在の page_id を返す (prefix-aware)。

    優先順位:
    1. target + "-..." で candidate にマッチ (target が省略形、candidate が拡張版)
    2. candidate + "-..." で target にマッチ (逆)
    3. strict ratio (>= threshold) でマッチ
    """
    candidates = difflib.get_close_matches(target, all_ids, n=5, cutoff=0.6)
    if not candidates:
        return None

    # Case 1: target ⊂ candidate (e.g., "lazarus" → "lazarus-refactor")
    for c in candidates:
        if c == target + "-" or c.startswith(target + "-"):
            return c
    # Case 2: candidate ⊂ target (e.g., "lazarus-refactor" → "lazarus")
    for c in candidates:
        if target == c + "-" or target.startswith(c + "-"):
            return c
    # Case 3: strict ratio
    top = candidates[0]
    ratio = difflib.SequenceMatcher(None, target, top).ratio()
    if ratio >= threshold:
        return top
    return None


def atomic_write(path: Path, content: str) -> None:
    """tempfile + os.replace による atomic write。

    同一ディレクトリに dot-prefix の一時ファイルを作り、fsync 後に rename で置換。
    書き込み失敗時は tempfile をクリーンアップする。
    """
    tmp = tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=str(path.parent),
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    )
    tmp_path = tmp.name
    try:
        tmp.write(content)
        tmp.flush()
        os.fsync(tmp.fileno())
        tmp.close()
        os.replace(tmp_path, path)
    except Exception:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise
