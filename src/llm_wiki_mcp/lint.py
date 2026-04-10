"""Lint engine - detect and fix wiki quality issues."""

import re
from datetime import date, datetime, timedelta
from pathlib import Path

from llm_wiki_mcp.wiki import PAGES_DIR, all_pages, find_page
from llm_wiki_mcp.server import _parse_frontmatter, _extract_wiki_links, _find_backlinks


STALE_DAYS = 90  # Pages not updated in this many days are flagged


def check() -> list[dict]:
    """Run all lint checks and return a list of issues."""
    issues = []
    pages = all_pages()

    all_page_ids = {p.stem for p in pages}

    for path in pages:
        page_id = path.stem
        content = path.read_text()
        fm = _parse_frontmatter(content)

        # 1. Broken links
        outlinks = _extract_wiki_links(content)
        for link in outlinks:
            if link not in all_page_ids:
                issues.append({
                    "type": "broken_link",
                    "severity": "high",
                    "page": page_id,
                    "detail": f"Link [[{link}]] points to non-existent page",
                    "auto_fixable": True,
                })

        # 2. Stale pages
        updated_str = fm.get("updated", "")
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
        backlinks = _find_backlinks(page_id)
        if not backlinks and len(pages) > 1:
            issues.append({
                "type": "orphan",
                "severity": "medium",
                "page": page_id,
                "detail": "No other pages link to this page",
                "auto_fixable": False,
            })

    # 4. Duplicate detection (pages with very similar titles)
    titles = {}
    for path in pages:
        fm = _parse_frontmatter(path.read_text())
        title = fm.get("title", path.stem).lower().strip()
        if title in titles:
            issues.append({
                "type": "duplicate",
                "severity": "medium",
                "page": path.stem,
                "detail": f"Possible duplicate of '{titles[title]}' (same title)",
                "auto_fixable": False,
            })
        else:
            titles[title] = path.stem

    # 5. Contradiction detection placeholder
    # Full contradiction detection requires LLM; we check for simple cases
    # like same entity with conflicting facts across linked pages
    # TODO: LLM-based contradiction detection in future iteration

    return issues


def apply_safe_fixes(issues: list[dict]) -> list[str]:
    """Apply only safe, auto-fixable issues. Returns list of actions taken."""
    actions = []

    for issue in issues:
        if not issue.get("auto_fixable"):
            continue

        if issue["type"] == "broken_link":
            page_id = issue["page"]
            path = find_page(page_id)
            if not path:
                continue

            content = path.read_text()
            # Extract the broken link target from detail
            link_match = re.search(r"\[\[([^\]]+)\]\]", issue["detail"])
            if link_match:
                broken_link = link_match.group(1)
                # Remove the broken link, keeping the text
                content = content.replace(f"[[{broken_link}]]", broken_link)
                path.write_text(content)
                actions.append(f"Removed broken link [[{broken_link}]] from {page_id}")

    return actions
