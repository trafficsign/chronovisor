"""Lint engine - detect and fix wiki quality issues."""

import re
from datetime import date
from pathlib import Path

from llm_wiki_mcp.wiki import SYSTEM_DIR, all_pages, find_page
from llm_wiki_mcp.index_store import get_store
from llm_wiki_mcp.link_fix import (
    atomic_write,
    extract_targets,
    find_fuzzy_match,
)


STALE_DAYS = 90  # Pages not updated in this many days are flagged


def _collect_all_page_ids() -> set[str]:
    """pages/ + system/ の全 page_id を返す。broken_link 判定の母集合。

    Backed by the IndexStore. Refresh is the caller's responsibility.
    """
    return get_store().all_page_ids(include_system=True)


def check() -> list[dict]:
    """Run all lint checks and return a list of issues."""
    store = get_store()
    store.refresh()

    issues = []

    # System pages are part of the broken_link universe but are not
    # themselves linted (they're treated as a fixed reference set).
    all_page_ids = store.all_page_ids(include_system=True)
    pages_meta = store.all_pages_meta(include_system=False)
    page_count = len(pages_meta)

    for meta in pages_meta:
        page_id = meta["page_id"]

        # 1. Broken links — outlinks come pre-normalized + code-fence-stripped
        #    from the index (same `extract_targets(strip=True)` semantics).
        seen_broken: set[str] = set()
        for target in store.outlinks(page_id):
            if target in all_page_ids or target in seen_broken:
                continue
            seen_broken.add(target)
            issues.append({
                "type": "broken_link",
                "severity": "high",
                "page": page_id,
                "detail": f"Link [[{target}]] points to non-existent page",
                "auto_fixable": True,
            })

        # 2. Stale pages
        updated_str = meta["updated"]
        if updated_str and updated_str != "unknown":
            try:
                updated_date = date.fromisoformat(updated_str)
                if (date.today() - updated_date).days > STALE_DAYS:
                    issues.append({
                        "type": "stale",
                        "severity": "low",
                        "page": page_id,
                        "detail": f"Last updated {updated_str} ({(date.today() - updated_date).days} days ago)",
                        "auto_fixable": False,
                    })
            except ValueError:
                pass

        # 3. Orphan pages (no backlinks)
        if not store.backlinks(page_id) and page_count > 1:
            issues.append({
                "type": "orphan",
                "severity": "medium",
                "page": page_id,
                "detail": "No other pages link to this page",
                "auto_fixable": False,
            })

    # 4. Duplicate detection (pages with very similar titles)
    titles: dict[str, str] = {}
    for meta in pages_meta:
        title = meta["title"].lower().strip()
        page_id = meta["page_id"]
        if title in titles:
            issues.append({
                "type": "duplicate",
                "severity": "medium",
                "page": page_id,
                "detail": f"Possible duplicate of '{titles[title]}' (same title)",
                "auto_fixable": False,
            })
        else:
            titles[title] = page_id

    # 5. Contradiction detection placeholder
    # Full contradiction detection requires LLM; we check for simple cases
    # like same entity with conflicting facts across linked pages
    # TODO: LLM-based contradiction detection in future iteration

    return issues


def _broken_link_target(issue: dict) -> str | None:
    """issue["detail"] から target page_id を取り出す。"""
    m = re.search(r"\[\[([^\]]+)\]\]", issue.get("detail", ""))
    if not m:
        return None
    return m.group(1).strip()


def _replace_link_in_content(content: str, target: str, replacement: str | None) -> tuple[str, int]:
    """``[[target]]`` / ``[[target|label]]`` / ``[[target#sec]]`` を置換。

    Args:
        content: 対象ファイルの本文
        target: normalize 済みの target page_id
        replacement: fuzzy match で見つかった置換先 page_id。None か同値なら plaintext 化。

    Returns:
        (new_content, count) — count は置換された箇所数
    """
    # frontmatter / code fence / inline code を壊さないため、本文だけ置換する。
    # ここでは全文で regex を走らせるが、pattern が [[...]] 限定なので
    # code 内の同パターン文字列だけは副次的に書き換わるリスクが残る。
    # lint.check() は strip した body で検出するが、apply はファイル全体を触る。
    # そのため pattern を保守的に定義する。
    pattern = re.compile(
        r"\[\[" + re.escape(target) + r"(?P<tail>\|[^\]\n]*|#[^\]\n]*)?\]\]"
    )

    def _repl(m: re.Match) -> str:
        tail = m.group("tail") or ""
        if replacement and replacement != target:
            return f"[[{replacement}{tail}]]"
        # plaintext fallback
        if tail.startswith("|"):
            return tail[1:]
        return target

    new_content, count = pattern.subn(_repl, content)
    return new_content, count


def apply_safe_fixes(
    issues: list[dict],
    dry_run: bool = False,
    fuzzy: bool = True,
) -> list[str]:
    """Apply safe auto-fixes. Returns list of actions taken.

    Args:
        issues: Issue list from check()
        dry_run: True なら書き込まず actions のプレビューだけ返す
        fuzzy: True なら broken_link を fuzzy match で救い、fallback で plaintext 化する。
               False なら broken_link は放置 (既存の挙動より安全側)。
    """
    store = get_store()
    actions: list[str] = []
    all_page_ids = store.all_page_ids(include_system=True)

    for issue in issues:
        if not issue.get("auto_fixable"):
            continue
        if issue["type"] != "broken_link":
            continue
        if not fuzzy:
            continue

        page_id = issue["page"]
        path = find_page(page_id)
        if not path:
            continue

        target = _broken_link_target(issue)
        if not target:
            continue

        # system/ 配下に実在するなら書き換え不要 (lint false positive のガード)
        if (SYSTEM_DIR / f"{target}.md").exists():
            continue

        replacement = find_fuzzy_match(target, all_page_ids)

        try:
            content = path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            continue

        new_content, count = _replace_link_in_content(content, target, replacement)
        if count == 0 or new_content == content:
            continue

        if replacement and replacement != target:
            label = f"[{page_id}] [[{target}]] → [[{replacement}]] ({count}x)"
        else:
            label = f"[{page_id}] [[{target}]] → plaintext ({count}x)"

        if dry_run:
            actions.append(f"[dry-run] {label}")
        else:
            atomic_write(path, new_content)
            actions.append(label)

    # If we mutated pages, the index is now stale — refresh once at the end
    # so subsequent reads see consistent backlinks/outlinks.
    if actions and not dry_run:
        store.refresh()

    return actions
