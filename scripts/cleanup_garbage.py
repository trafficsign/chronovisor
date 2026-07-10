#!/usr/bin/env python3
"""ゴミページのクリーンアップ。

GitHub テンプレ由来や generic なプロジェクト管理ファイルが
raw 取り込み時にフィルタされず流入したものを削除する。

判定基準（保守的、曖昧なら残す）:
- filename がブラックリスト (authors, license, contributing 等)
- 本文が極端に短い (< 200B) + filename が generic パターン
- 一般的な title (authors, license 等) + 本文が短い (< 500B)

実行モード:
- 引数なし / --dry-run: 一覧表示のみ、削除しない
- --apply: 廃止済み。削除提案は frontier-managed sleep lane が処理する

ログ: ~/.wiki/backups/<latest>/cleanup.log に記録
"""

import argparse
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from llm_wiki_mcp.legacy_semantic_write import (  # noqa: E402
    block_legacy_semantic_mutation,
)

WIKI_ROOT = Path.home() / ".wiki"
WIKI_PAGES = WIKI_ROOT / "pages"
BACKUPS_DIR = WIKI_ROOT / "backups"

# GitHub テンプレ / プロジェクト管理ファイル (page_id lowercase)
GARBAGE_FILENAMES = frozenset({
    "authors",
    "best-practices",
    "bestpractices",
    "privacy",
    "pullrequesttemplate",
    "pull-request-template",
    "contributing",
    "contribution-guide",
    "code-of-conduct",
    "codeofconduct",
    "license",
    "license-mit",
    "license-apache",
    "security",
    "security-policy",
    "support",
    "issue-template",
    "issuetemplate",
    "funding",
    "codeowners",
    "stale",
    "dependabot",
    "manifest",
    "package",
    "package-lock",
    "tsconfig",
    "eslintrc",
    "eslint-config",
    "prettierrc",
    "prettier-config",
    "gitignore",
    "dockerignore",
    "editorconfig",
    "browserslistrc",
    "release-notes",
    "releasenotes",
    "credits",
    "acknowledgments",
    "acknowledgements",
})


def parse_frontmatter_and_body(text: str) -> tuple[dict, str]:
    """frontmatter と body を分離。"""
    fm: dict[str, str] = {}
    body = text
    if text.startswith("---"):
        end = text.find("---", 3)
        if end != -1:
            for line in text[3:end].strip().splitlines():
                if ":" in line:
                    key, _, value = line.partition(":")
                    fm[key.strip()] = value.strip()
            body = text[end + 3:].strip()
    return fm, body


def is_garbage(path: Path) -> tuple[bool, str]:
    """ゴミ判定。(is_garbage, reason) を返す。"""
    page_id = path.stem.lower()

    # 1. filename ブラックリスト直接マッチ
    if page_id in GARBAGE_FILENAMES:
        return True, f"GitHub template filename: '{page_id}'"

    # 2. 内容ベース判定
    try:
        text = path.read_text()
    except OSError as e:
        return False, f"read error: {e}"

    fm, body = parse_frontmatter_and_body(text)
    title = fm.get("title", path.stem).lower()
    body_len = len(body)

    # 3. 短い本文 + generic filename パターン
    generic_patterns = ["template", "manifest", "schema-template"]
    if body_len < 200 and any(g in page_id for g in generic_patterns):
        return True, f"Short ({body_len}B) + generic pattern in '{page_id}'"

    # 4. generic な title + 短い本文
    title_norm = title.replace(" ", "").replace("_", "").replace("-", "")
    generic_titles = {
        "authors", "license", "contributing", "readme",
        "changelog", "support", "credits", "acknowledgments",
        "codeofconduct", "security", "privacy",
    }
    if title_norm in generic_titles and body_len < 500:
        return True, f"Generic title '{title}' + short body ({body_len}B)"

    return False, ""


def find_log_target() -> Path:
    """ログ書き込み先 = 最新の backup ディレクトリ。"""
    if not BACKUPS_DIR.exists():
        target = BACKUPS_DIR / datetime.now().strftime("%Y%m%d_%H%M%S")
        target.mkdir(parents=True, exist_ok=True)
        return target
    backups = sorted(
        [p for p in BACKUPS_DIR.iterdir() if p.is_dir()],
        reverse=True,
    )
    if backups:
        return backups[0]
    target = BACKUPS_DIR / datetime.now().strftime("%Y%m%d_%H%M%S")
    target.mkdir(parents=True, exist_ok=True)
    return target


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true",
                        help="プレビューのみ、削除しない")
    parser.add_argument(
        "--apply",
        action="store_true",
        help="廃止済み。frontier-managed sleep lane を使用する",
    )
    args = parser.parse_args()

    if args.apply:
        block_legacy_semantic_mutation(
            tool="cleanup_garbage.py",
            replacement="llm-wiki-sleep",
        )

    pages = sorted(WIKI_PAGES.rglob("*.md"))
    print(f"スキャン対象: {len(pages)} ページ")

    garbage = []
    for path in pages:
        is_g, reason = is_garbage(path)
        if is_g:
            garbage.append((path, reason))

    print(f"ゴミ判定: {len(garbage)} ページ\n")

    if not garbage:
        print("ゴミページなし。終了。")
        return

    # 一覧表示
    for path, reason in garbage:
        rel = path.relative_to(WIKI_PAGES)
        size = path.stat().st_size
        print(f"  {rel} ({size}B) — {reason}")
    print()

    print("[dry-run] 削除はスキップ。適用は llm-wiki-sleep が frontier review 後に行います。")


if __name__ == "__main__":
    main()
