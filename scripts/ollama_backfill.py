#!/usr/bin/env python3
"""Ollama (Gemma 4 26B MoE) でリンクバックフィル。

backfill_links.py の Sonnet 版と同じ機能をローカル LLM で実行。
1 ページずつ独立 query で Ollama に問い合わせて、最大 5 件の関連リンクを取得し、
backfill_links.py の merge_related_section を流用して書き込む。

引数: バッチ JSON ファイルパス (1 バッチ = ~20 ページ程度)

この旧ローカルモデル実行経路は fail-closed。リンク提案と適用には
``chronovisor-sleep`` の frontier-managed orphan-link lane を使う。
"""

import json
import re
import sys
import time
from pathlib import Path

# chronovisor パッケージから既存ロジックを import
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent))

# AUTO_MARKER を Ollama 版に上書きするため import 後に再定義
import backfill_links
from backfill_links import atomic_write, merge_related_section

from chronovisor.core.ollama import generate, is_available
from chronovisor.raw.legacy_semantic_write import (
    block_legacy_semantic_mutation,
)

backfill_links.AUTO_MARKER = "<!-- auto-linked: backfill (ollama gemma4) -->"


PROMPT_TEMPLATE = """あなたは Wiki ページの関連リンクを判定する LLM です。

# 対象ページ
ID: {page_id}
タイトル: {title}

## 本文 (抜粋)
{body_excerpt}

# 候補リンク
{candidates_text}

# 指示
上記の対象ページと意味的に関連する候補を **最大 5 件** 選んでください。

判定基準:
- **関連あり**: 同じ概念領域 / プロジェクト / 技術スタック / 参照関係 / 依存関係
- **関連なし**: 共通語があるだけ / 別ドメイン / 部品番号シリーズ違い等

迷ったら付けない (0 件も OK)。

# 出力形式
JSON 配列のみで page_id を列挙してください。説明文は不要。
例: ["page-a", "page-b", "page-c"]
関連なし: []
"""


def parse_llm_output(output: str, valid_ids: set[str]) -> list[str]:
    """LLM 出力から page_id リストを抽出 + valid 候補のみ保持。"""
    # JSON 配列パターンをマッチ
    m = re.search(r"\[\s*((?:\"[^\"]*\"\s*,?\s*)*)\]", output)
    if not m:
        return []
    try:
        result = json.loads("[" + m.group(1) + "]")
        if isinstance(result, list):
            # valid な候補のみ
            return [str(x) for x in result if isinstance(x, str) and x in valid_ids][:5]
    except json.JSONDecodeError:
        pass
    return []


def get_body_excerpt(path: Path, max_chars: int = 3000) -> str:
    """ファイルから body 部分を抽出 (frontmatter 除去)。"""
    try:
        text = path.read_text(encoding="utf-8", errors="strict")
    except (OSError, UnicodeDecodeError):
        return ""

    # frontmatter 除去
    if text.startswith("---"):
        end = text.find("---", 3)
        if end != -1:
            text = text[end + 3:].strip()

    if len(text) <= max_chars:
        return text
    return text[:max_chars] + "\n...[truncated]"


def main() -> None:
    block_legacy_semantic_mutation(
        tool="ollama_backfill.py",
        replacement="chronovisor-sleep",
    )
    if len(sys.argv) != 2:
        print("Usage: python3 ollama_backfill.py <batch.json>", file=sys.stderr)
        sys.exit(1)

    batch_file = Path(sys.argv[1])
    if not batch_file.exists():
        print(f"Error: {batch_file} not found", file=sys.stderr)
        sys.exit(1)

    if not is_available():
        print("Error: Ollama is not running on localhost:11434", file=sys.stderr)
        sys.exit(1)

    with open(batch_file, encoding="utf-8") as f:
        batch_data = json.load(f)

    pages = batch_data["pages"]
    batch_id = batch_data.get("batch_id", "?")
    batch_label = f"{batch_id:02d}" if isinstance(batch_id, int) else str(batch_id)
    print(f"=== Batch {batch_label}: {len(pages)} pages ===")

    written = 0
    skipped = 0
    errors: list[tuple[str, str]] = []
    total_links = 0
    start_time = time.time()

    for i, page in enumerate(pages):
        page_id = page["page_id"]
        path = Path(page["path"])
        title = page["title"]
        candidates = page["candidates"]
        valid_ids = {c["page_id"] for c in candidates}

        elapsed = time.time() - start_time
        avg = elapsed / max(i, 1)
        eta = avg * (len(pages) - i - 1)
        print(
            f"[{i + 1:3d}/{len(pages)}] {page_id[:50]:50s} "
            f"({len(candidates)} cand) elapsed={elapsed:.0f}s eta={eta:.0f}s"
        )

        body_excerpt = get_body_excerpt(path)
        if not body_excerpt:
            print("    skip: cannot read body")
            errors.append((page_id, "read failed"))
            continue

        candidates_text = "\n".join(
            f"- {c['page_id']}: {c['title']}" for c in candidates
        )

        prompt = PROMPT_TEMPLATE.format(
            page_id=page_id,
            title=title,
            body_excerpt=body_excerpt,
            candidates_text=candidates_text,
        )

        try:
            output = generate(prompt)
        except Exception as e:
            print(f"    ollama error: {e}")
            errors.append((page_id, f"ollama: {e}"))
            continue

        selected = parse_llm_output(output, valid_ids)
        if not selected:
            print("    → 0 selected")
            skipped += 1
            continue

        # 書き込み (backfill_links.py の merge_related_section を流用)
        try:
            content = path.read_text(encoding="utf-8")
            new_content = merge_related_section(content, selected)
            if new_content == content:
                print("    → no change (already linked)")
                skipped += 1
                continue
            atomic_write(path, new_content)
            written += 1
            total_links += len(selected)
            print(f"    ✓ {len(selected)} links: {selected}")
        except Exception as e:
            print(f"    write error: {e}")
            errors.append((page_id, f"write: {e}"))

    elapsed_total = time.time() - start_time
    print(f"\n=== Batch {batch_id} 完了 ({elapsed_total:.0f}s) ===")
    print(f"  書き込み: {written}")
    print(f"  スキップ: {skipped}")
    print(f"  エラー: {len(errors)}")
    print(f"  追加リンク総数: {total_links}")
    if errors:
        print("\n  エラー詳細 (最大 5 件):")
        for pid, err in errors[:5]:
            print(f"    {pid}: {err}")


if __name__ == "__main__":
    main()
