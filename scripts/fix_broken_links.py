#!/usr/bin/env python3
"""broken_link の修復。

lint が検出した broken_link について、以下を試行:
1. fuzzy match (difflib ratio >= 0.7) で近い page_id が見つかれば → 置換
2. マッチしなかったら → `[[link]]` を `link` (プレーンテキスト) に変換
3. `system/` 配下のページへの参照 (例: [[claude-code]] → ~/.chronovisor/system/claude-code.md)
   は system パス経由で存在するので、プレーンテキスト化で妥協する

実行:
    python3 fix_broken_links.py --dry-run   # プレビュー
    chronovisor-sleep                          # frontier 審査後に自動適用

この旧スクリプトの直接書き込み関数は fail-closed。
"""

import argparse
import difflib
import json
import os
import re
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from chronovisor.store import PAGES_DIR, CHRONOVISOR_ROOT  # noqa: E402
from chronovisor.lint import check  # noqa: E402
from chronovisor.legacy_semantic_write import (  # noqa: E402
    block_legacy_semantic_mutation,
)

SYSTEM_DIR = CHRONOVISOR_ROOT / "system"

# fuzzy match threshold (0.0-1.0)
FUZZY_THRESHOLD = 0.70


def normalize_link_target(link: str) -> str:
    """[[page|label]] / [[page#section]] から target page_id を抽出。"""
    return link.split("|")[0].split("#")[0].strip()


def get_all_page_ids() -> set[str]:
    """pages/ + system/ 配下の全 page_id を取得。"""
    ids: set[str] = set()
    for p in PAGES_DIR.rglob("*.md"):
        ids.add(p.stem)
    if SYSTEM_DIR.exists():
        for p in SYSTEM_DIR.rglob("*.md"):
            ids.add(p.stem)
    return ids


def find_fuzzy_match(
    target: str, all_ids: set[str], threshold: float = FUZZY_THRESHOLD
) -> str | None:
    """target page_id に近い実在の page_id を返す。

    優先順位:
    1. target が candidate の prefix (target + "-..." = candidate) → 置換 OK
    2. candidate が target の prefix → 置換 OK
    3. どちらでもない場合は ratio >= 0.88 のときだけ置換
    """
    candidates = difflib.get_close_matches(target, all_ids, n=5, cutoff=0.6)
    if not candidates:
        return None

    # Case 1: target + "-..." で candidate にマッチ (補完関係、target の省略形)
    for c in candidates:
        if c == target + "-" or c.startswith(target + "-"):
            return c
    # Case 2: candidate + "-..." で target にマッチ (target が candidate の拡張版)
    for c in candidates:
        if target == c + "-" or target.startswith(c + "-"):
            return c
    # Case 3: strict ratio でマッチ
    top = candidates[0]
    ratio = difflib.SequenceMatcher(None, target, top).ratio()
    if ratio >= 0.88:
        return top
    return None


def atomic_write(path: Path, content: str) -> None:
    block_legacy_semantic_mutation(
        tool="fix_broken_links.py",
        replacement="chronovisor-sleep",
    )
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


def process_broken_link(issue: dict, all_ids: set[str], threshold: float = FUZZY_THRESHOLD) -> dict:
    """1 件の broken_link を分析。

    Returns:
        {
            'page': source page_id,
            'source_link': '[[link]]' (元の形, label 含む可能性),
            'target': 'target page_id' (normalized),
            'action': 'replace' | 'plaintext',
            'replacement': page_id or None (replace 時のみ),
        }
    """
    page = issue["page"]
    detail = issue["detail"]
    m = re.search(r"\[\[([^\]]+)\]\]", detail)
    if not m:
        return {"page": page, "action": "skip", "reason": "unparseable detail"}
    source_link = m.group(1)
    target = normalize_link_target(source_link).lower()

    # 1. system/ 配下に実在する場合は、リンクを保持する (lint の false positive)
    #    lint は pages/ のみ走査するので system/ ファイルを検出できない
    if (SYSTEM_DIR / f"{target}.md").exists():
        return {
            "page": page,
            "source_link": source_link,
            "target": target,
            "action": "skip",
            "reason": "target exists in system/ (lint false positive)",
        }

    # 2. fuzzy match (prefix-aware)
    match = find_fuzzy_match(target, all_ids, threshold)
    if match and match != target:
        return {
            "page": page,
            "source_link": source_link,
            "target": target,
            "action": "replace",
            "replacement": match,
        }

    # 3. マッチなし → プレーンテキスト化
    return {
        "page": page,
        "source_link": source_link,
        "target": target,
        "action": "plaintext",
        "reason": "no fuzzy match",
    }


def apply_fix(path: Path, source_link: str, action: str, replacement: str | None) -> bool:
    """対象ファイルに修正を適用。

    - replace: [[old|label]] → [[new|label]] or [[new]]
    - plaintext: [[old|label]] → label (or old)
    """
    try:
        content = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return False

    old_full = f"[[{source_link}]]"
    if old_full not in content:
        return False

    if action == "replace":
        # [[old|label]] → [[new|label]], [[old]] → [[new]], [[old#sec]] → [[new#sec]]
        if "|" in source_link:
            _, _, label = source_link.partition("|")
            new_full = f"[[{replacement}|{label}]]"
        elif "#" in source_link:
            _, _, section = source_link.partition("#")
            new_full = f"[[{replacement}#{section}]]"
        else:
            new_full = f"[[{replacement}]]"
    elif action == "plaintext":
        # [[old|label]] → label, [[old]] → old, [[old#sec]] → old
        if "|" in source_link:
            _, _, label = source_link.partition("|")
            new_full = label
        else:
            new_full = normalize_link_target(source_link)
    else:
        return False

    new_content = content.replace(old_full, new_full)
    if new_content == content:
        return False

    atomic_write(path, new_content)
    return True


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true", help="プレビューのみ")
    parser.add_argument(
        "--threshold",
        type=float,
        default=FUZZY_THRESHOLD,
        help="fuzzy match threshold (default 0.70)",
    )
    args = parser.parse_args()

    threshold = args.threshold

    print("lint 実行中...")
    issues = check()
    broken_links = [i for i in issues if i["type"] == "broken_link"]
    print(f"  broken_link: {len(broken_links)} 件")

    all_ids = get_all_page_ids()
    print(f"  全 page_id (pages + system): {len(all_ids)}")

    print("\n=== 分析 ===")
    plans: list[dict] = []
    stats = {"replace": 0, "plaintext": 0, "skip": 0}
    for issue in broken_links:
        plan = process_broken_link(issue, all_ids, threshold)
        plans.append(plan)
        stats[plan.get("action", "skip")] = stats.get(plan.get("action", "skip"), 0) + 1

    print(f"  replace (fuzzy match): {stats.get('replace', 0)}")
    print(f"  plaintext (削除):      {stats.get('plaintext', 0)}")
    print(f"  skip:                  {stats.get('skip', 0)}")

    print("\n=== 詳細 (最初の 30 件) ===")
    for plan in plans[:30]:
        page = plan["page"]
        action = plan.get("action", "skip")
        source = plan.get("source_link", "?")
        if action == "replace":
            repl = plan["replacement"]
            print(f"  [REPLACE]   {page:45s} [[{source}]] → [[{repl}]]")
        elif action == "plaintext":
            target = plan["target"]
            reason = plan.get("reason", "")
            print(f"  [PLAINTEXT] {page:45s} [[{source}]] → {target}  ({reason})")
        else:
            print(f"  [SKIP]      {page:45s} {plan.get('reason', '')}")

    if len(plans) > 30:
        print(f"  ... 他 {len(plans) - 30} 件")

    if args.dry_run:
        print("\n[dry-run] 適用はスキップ")
        return

    print("\n=== 適用 ===")
    applied = 0
    failed = 0
    for plan in plans:
        action = plan.get("action", "skip")
        if action == "skip":
            continue
        page = plan["page"]
        source_link = plan["source_link"]
        # page file を特定
        matches = list(PAGES_DIR.rglob(f"{page}.md"))
        if not matches:
            failed += 1
            continue
        path = matches[0]
        ok = apply_fix(
            path,
            source_link,
            action,
            plan.get("replacement"),
        )
        if ok:
            applied += 1
        else:
            failed += 1

    print(f"  適用: {applied}")
    print(f"  失敗: {failed}")


if __name__ == "__main__":
    main()
