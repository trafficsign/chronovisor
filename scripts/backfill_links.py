#!/usr/bin/env python3
"""Chronovisor に既存ページのリンクをバックフィル。

古いシステムから取り込んだ orphan ページに対して機械的に「## 関連」セクションを追加する。
Phase 2 (機械バックフィル) の本体。Phase 3 (Sonnet 精査) の前段。

Codex レビュー対応:
- 2-pass + inverted index で feedback loop と O(N²) を回避
- 既存「## 関連」セクションは完全保持、auto-linked マーカー以下に追記のみ
- コードブロック追跡 + 厳密 frontmatter parse
- 日本語対応キーワード抽出 (漢字 2+, カタカナ 3+, 英大文字始まり 4+)
- atomic write (tempfile + os.replace)
- リンク正規化 ([[page|label]], [[page#section]] → page)
- encoding strict (失敗は個別ログ)

実行:
- python3 backfill_links.py --dry-run    # プレビュー
- chronovisor-sleep                         # frontier 審査後に自動適用

この旧スクリプトの直接書き込み関数は fail-closed。
"""

import argparse
import json
import os
import re
import sys
import tempfile
from collections import defaultdict
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from chronovisor.raw.legacy_semantic_write import (
    block_legacy_semantic_mutation,
)

# chronovisor パッケージから既存定数を import
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
import contextlib

from chronovisor.core.store import PAGES_DIR

CHRONOVISOR_ROOT = Path.home() / ".chronovisor"
BACKUPS_DIR = CHRONOVISOR_ROOT / "backups"

# サイズ閾値 (バイト)
SIZE_MED = 30_000     # 30KB 未満は全文
SIZE_LARGE = 100_000  # 100KB 以上は警告

# キーワード抽出パターン
KANJI_PATTERN = re.compile(r"[\u4e00-\u9fff]{2,}")
KATAKANA_PATTERN = re.compile(r"[\u30a0-\u30ff\u31f0-\u31ff]{3,}")
ASCII_WORD_PATTERN = re.compile(r"\b[A-Z][a-zA-Z0-9_]{3,}\b")  # 大文字始まり 4+
WIKI_LINK_PATTERN = re.compile(r"\[\[([^\]]+)\]\]")
CODE_FENCE_PATTERN = re.compile(r"^\s*```")

# セクション検出
SECTION_HEADER_PATTERN = re.compile(r"^##\s+関連\s*$")
NEXT_SECTION_PATTERN = re.compile(r"^##\s+")
AUTO_MARKER = "<!-- auto-linked: backfill_links.py -->"

# 設定
TOP_K = 8
MIN_SCORE = 0.5
MIN_TERM_LEN = 2  # inverted index に登録する term の最小長
MIN_BODY_SIZE = 300  # これ未満のページは関連検索をスキップ (キーワード貧弱)
MAX_TERM_FREQ_RATIO = 0.30  # 全ページの 30% 以上に出現する term は generic として除外

# 関連候補から除外する generic な page_id (lint/raw 取り込みノイズ)
GENERIC_BLACKLIST = frozenset({
    "architecture",
    "architecture-1",
    "history",
    "manifest",
    "readme",
    "changelog",
    "support",
    "credits",
    "acknowledgments",
    "acknowledgements",
    "authors",
    "license",
    "privacy",
    "security",
    "contributing",
    "code-of-conduct",
    "release-notes",
    "issue-template",
    "pullrequesttemplate",
    "agents",  # AGENTS.md は GitHub の generic file
    "foundry",
    "sdksintro",
    "workflowtaskchunking",
    "best-practices",
    "bestpractices",
})


def normalize_link(link: str) -> str:
    """[[page|label]] や [[page#section]] を 'page' に正規化。"""
    return link.split("|")[0].split("#")[0].strip().lower()


def parse_frontmatter_strict(text: str) -> tuple[dict, int]:
    """frontmatter を厳密 parse。(fm_dict, body_offset) を返す。"""
    if not text.startswith("---"):
        return {}, 0
    end = text.find("---", 3)
    if end == -1:
        return {}, 0
    fm: dict[str, str] = {}
    for line in text[3:end].strip().splitlines():
        if ":" in line:
            key, _, value = line.partition(":")
            fm[key.strip()] = value.strip()
    return fm, end + 3


def strip_code_blocks(content: str) -> str:
    """コードブロックを除去。fence (```) で囲まれた範囲を削除。"""
    lines = []
    in_block = False
    for line in content.splitlines():
        if CODE_FENCE_PATTERN.match(line):
            in_block = not in_block
            continue
        if not in_block:
            lines.append(line)
    return "\n".join(lines)


def extract_body_for_keywords(content: str, body_start: int, file_size: int) -> str:
    """サイズ別に keyword 抽出用テキストを返す。

    大ファイルは header bias を避けて head/middle/tail から分散取得。
    """
    body = content[body_start:]
    body_len = len(body)

    if file_size < SIZE_MED:
        return body

    chunk = 2000
    if body_len <= chunk * 3:
        return body

    head = body[:chunk]
    mid_start = body_len // 2 - chunk // 2
    mid = body[mid_start:mid_start + chunk]
    tail = body[-chunk:]
    return f"{head}\n\n{mid}\n\n{tail}"


def extract_keywords(title: str, page_id: str, body_text: str) -> set[str]:
    """ページからキーワード集合を抽出。"""
    keywords: set[str] = set()

    # タイトル
    if title:
        keywords.add(title.lower().strip())

    # page_id を words 分解 (4 文字以上)
    for w in re.split(r"[-_]", page_id):
        if len(w) >= 4:
            keywords.add(w.lower())

    # コードブロック除去
    text = strip_code_blocks(body_text)

    # 漢字 2 文字以上の連続
    for m in KANJI_PATTERN.findall(text):
        keywords.add(m.lower())

    # カタカナ 3 文字以上
    for m in KATAKANA_PATTERN.findall(text):
        keywords.add(m.lower())

    # 英大文字始まり 4 文字以上
    for m in ASCII_WORD_PATTERN.findall(text):
        keywords.add(m.lower())

    # 既存の [[link]] 参照先 (これらは確実な関連シグナル)
    for m in WIKI_LINK_PATTERN.findall(text):
        normalized = normalize_link(m)
        if normalized:
            keywords.add(normalized)

    return keywords


def find_related_section(content: str) -> tuple[int, int] | None:
    """既存「## 関連」セクションの (start_line, end_line) を返す。

    code block 内は無視。次の「## 」ヘッダで終わる。
    section_start <= i < section_end の範囲が関連セクション。
    """
    lines = content.splitlines()
    in_block = False
    section_start: int | None = None

    for i, line in enumerate(lines):
        if CODE_FENCE_PATTERN.match(line):
            in_block = not in_block
            continue
        if in_block:
            continue

        if section_start is None:
            if SECTION_HEADER_PATTERN.match(line):
                section_start = i
        else:
            if NEXT_SECTION_PATTERN.match(line):
                return (section_start, i)

    if section_start is not None:
        return (section_start, len(lines))
    return None


def build_inverted_index(
    pages: list[Path],
) -> tuple[dict[str, set[str]], dict[str, dict]]:
    """全ページを 1 回スキャンして inverted index を構築。

    高頻度 term (全ページの MAX_TERM_FREQ_RATIO 以上に出現) は generic として除外。

    Returns:
        (term_to_page_ids, page_id_to_meta)
    """
    index: dict[str, set[str]] = defaultdict(set)
    meta: dict[str, dict] = {}
    read_errors: list[tuple[Path, str]] = []
    large_files: list[tuple[Path, int]] = []

    for path in pages:
        try:
            content = path.read_text(encoding="utf-8", errors="strict")
        except (OSError, UnicodeDecodeError) as e:
            read_errors.append((path, str(e)))
            continue

        fm, body_start = parse_frontmatter_strict(content)
        title = fm.get("title", path.stem)
        page_id = path.stem
        file_size = len(content)
        body_size = file_size - body_start

        if file_size >= SIZE_LARGE:
            large_files.append((path, file_size))

        kw_text = extract_body_for_keywords(content, body_start, file_size)
        keywords = extract_keywords(title, page_id, kw_text)

        meta[page_id] = {
            "path": path,
            "title": title,
            "keywords": keywords,
            "content": content,
            "file_size": file_size,
            "body_size": body_size,
        }

        for kw in keywords:
            if len(kw) >= MIN_TERM_LEN:
                index[kw].add(page_id)

    if read_errors:
        print(f"  [warn] read failed: {len(read_errors)} files")
        for path, err in read_errors[:5]:
            print(f"    {path.relative_to(PAGES_DIR)}: {err}")
    if large_files:
        print(f"  [info] large files (>= {SIZE_LARGE}B): {len(large_files)}")

    # 高頻度 term を drop (TF-IDF 風)
    total_pages = len(meta)
    threshold = max(2, int(total_pages * MAX_TERM_FREQ_RATIO))
    generic_terms = {term for term, pages_set in index.items() if len(pages_set) >= threshold}
    for term in generic_terms:
        del index[term]
    print(f"  generic terms dropped (>= {threshold} pages): {len(generic_terms)}")

    return index, meta


def find_related_pages(
    page_id: str,
    page_meta: dict,
    index: dict[str, set[str]],
    meta: dict[str, dict],
) -> list[tuple[str, float]]:
    """1 ページに対する関連ページを返す。

    inverted index から候補を取得し、候補ごとにスコアリング。
    小ページ (本文 < MIN_BODY_SIZE) はスキップ。
    GENERIC_BLACKLIST 該当ページは候補から除外。
    """
    # 小ページはスキップ (キーワード貧弱で誤マッチが多い)
    if page_meta["body_size"] < MIN_BODY_SIZE:
        return []

    self_keywords = page_meta["keywords"]
    scores: dict[str, float] = defaultdict(float)

    for kw in self_keywords:
        candidates = index.get(kw, set())
        for cand_id in candidates:
            if cand_id == page_id:
                continue
            if cand_id.lower() in GENERIC_BLACKLIST:
                continue
            cand_meta = meta[cand_id]
            cand_title = cand_meta["title"].lower()
            cand_stem = cand_id.lower().replace("-", " ")

            score_inc = 0.0
            if kw in cand_title:
                score_inc += 0.5
            if kw in cand_stem:
                score_inc += 0.3
            score_inc += 0.1  # 共通 term があるだけで base score
            scores[cand_id] += score_inc

    # 既存リンク先を除外
    page_content = page_meta["content"]
    existing_links = {
        normalize_link(m) for m in WIKI_LINK_PATTERN.findall(page_content)
    }

    filtered = [
        (pid, sc)
        for pid, sc in scores.items()
        if sc >= MIN_SCORE and pid.lower() not in existing_links
    ]
    filtered.sort(key=lambda x: x[1], reverse=True)
    return filtered[:TOP_K]


def merge_related_section(content: str, new_links: list[str]) -> str:
    """既存「## 関連」セクションは保持し、AUTO_MARKER 以下に新規追記。

    既存セクションがなければ末尾に新規追加。
    """
    if not new_links:
        return content

    lines = content.splitlines()
    section = find_related_section(content)
    new_link_lines = [f"- [[{lnk}]]" for lnk in new_links]

    if section is None:
        # 末尾に新規セクション追加
        # 末尾の空行を整える
        while lines and not lines[-1].strip():
            lines.pop()
        addition = ["", "## 関連", "", AUTO_MARKER, *new_link_lines]
        return "\n".join(lines + addition) + "\n"

    start, end = section
    section_lines = lines[start:end]

    # AUTO_MARKER の位置を探す
    marker_idx: int | None = None
    for i, line in enumerate(section_lines):
        if line.strip() == AUTO_MARKER:
            marker_idx = i
            break

    if marker_idx is not None:
        # marker 以降の既存 auto-linked リンクを取得
        existing_auto: set[str] = set()
        for line in section_lines[marker_idx + 1:]:
            for m in WIKI_LINK_PATTERN.findall(line):
                existing_auto.add(normalize_link(m))

        truly_new = [lnk for lnk in new_links if lnk.lower() not in existing_auto]
        if not truly_new:
            return content
        new_link_lines = [f"- [[{lnk}]]" for lnk in truly_new]

        # marker 以下の auto-linked 部分を新規で置換 (既存 marker 以下も保持)
        new_section = (
            section_lines[: marker_idx + 1]
            + new_link_lines
            + section_lines[marker_idx + 1:]
        )
    else:
        # marker なし: 既存内容を保持し、末尾に marker + 新規リンクを追加
        existing_in_section: set[str] = set()
        for line in section_lines:
            for m in WIKI_LINK_PATTERN.findall(line):
                existing_in_section.add(normalize_link(m))
        truly_new = [
            lnk for lnk in new_links if lnk.lower() not in existing_in_section
        ]
        if not truly_new:
            return content
        new_link_lines = [f"- [[{lnk}]]" for lnk in truly_new]

        # 既存セクションの末尾の空行を削る
        while section_lines and not section_lines[-1].strip():
            section_lines.pop()
        new_section = section_lines + ["", AUTO_MARKER, *new_link_lines]

    new_lines = lines[:start] + new_section + lines[end:]
    result = "\n".join(new_lines)
    if not result.endswith("\n"):
        result += "\n"
    return result


def atomic_write(path: Path, content: str) -> None:
    """tempfile + os.replace で atomic 書き込み。"""
    block_legacy_semantic_mutation(
        tool="backfill_links.py",
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
        with contextlib.suppress(OSError):
            os.unlink(tmp_path)
        raise


def get_sample_pages(meta: dict, decisions: dict, n: int = 14) -> list[str]:
    """サイズ別 + decisions ありから代表サンプルを選ぶ。"""
    with_decisions = [pid for pid in decisions]
    if not with_decisions:
        return []

    sorted_by_size = sorted(
        with_decisions, key=lambda pid: meta[pid]["file_size"]
    )
    n_each = max(1, n // 5)
    samples = []

    # 小・中小・中・中大・大 から各 n_each
    L = len(sorted_by_size)
    if n >= L:
        return sorted_by_size

    indices = [
        0,                          # 最小
        L // 4,                     # 小寄り
        L // 2,                     # 中央
        L * 3 // 4,                 # 大寄り
        L - 1,                      # 最大
    ]
    seen = set()
    for base in indices:
        for offset in range(n_each):
            idx = base + offset
            if 0 <= idx < L and idx not in seen:
                seen.add(idx)
                samples.append(sorted_by_size[idx])
                if len(samples) >= n:
                    return samples
    return samples


def find_log_target() -> Path:
    """最新の backups/<timestamp>/ を返す。"""
    if BACKUPS_DIR.exists():
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
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="変更プレビューのみ、書き込まない",
    )
    parser.add_argument(
        "--sample",
        type=int,
        default=14,
        help="dry-run 時のサンプル数",
    )
    parser.add_argument(
        "--export-candidates",
        type=str,
        default=None,
        help="候補を JSON にエクスポート (Sonnet サブエージェント連携用)",
    )
    parser.add_argument(
        "--num-batches",
        type=int,
        default=6,
        help="--export-candidates 時のバッチ分割数",
    )
    args = parser.parse_args()

    pages = sorted(PAGES_DIR.rglob("*.md"))
    print(f"スキャン対象: {len(pages)} ページ")

    print("\nPass 1: inverted index 構築...")
    index, meta = build_inverted_index(pages)
    print(f"  unique terms: {len(index)}")
    print(f"  pages indexed: {len(meta)}")

    print("\nPass 2: 関連ページ判定...")
    decisions: dict[str, list[str]] = {}
    skipped_processed = 0
    for page_id, page_meta in meta.items():
        # AUTO_MARKER 済みのページはスキップ (backfill 再実行時)
        if "auto-linked: backfill" in page_meta["content"]:
            skipped_processed += 1
            continue
        related = find_related_pages(page_id, page_meta, index, meta)
        if related:
            decisions[page_id] = [pid for pid, _ in related]

    print(f"  処理済みスキップ: {skipped_processed}")
    print(f"  関連ありページ: {len(decisions)} / {len(meta)}")
    print(f"  追加リンク総数: {sum(len(v) for v in decisions.values())}")

    if args.dry_run:
        sample_ids = get_sample_pages(meta, decisions, n=args.sample)
        print(f"\n[dry-run] サンプル {len(sample_ids)} 件:")
        for pid in sample_ids:
            page_meta = meta[pid]
            print(f"\n  --- {pid} ---")
            print(f"  title: {page_meta['title']}")
            print(f"  size: {page_meta['file_size']}B")
            print(f"  追加リンク ({len(decisions[pid])}):")
            for r in decisions[pid]:
                r_title = meta[r]["title"] if r in meta else r
                print(f"    - [[{r}]]  {r_title}")
        return

    if args.export_candidates:
        # JSON 形式で候補をエクスポート + バッチ分割
        out_dir = Path(args.export_candidates)
        out_dir.mkdir(parents=True, exist_ok=True)

        # ページ ID リストを sort して安定化
        page_ids = sorted(decisions.keys())
        n_batches = args.num_batches
        batches: list[list[str]] = [[] for _ in range(n_batches)]
        for i, pid in enumerate(page_ids):
            batches[i % n_batches].append(pid)

        # 各バッチを JSON ファイルに保存
        for bi, batch in enumerate(batches):
            batch_data = {
                "batch_id": bi,
                "total_batches": n_batches,
                "pages": [],
            }
            for pid in batch:
                page_meta = meta[pid]
                candidates = []
                for cand_id in decisions[pid]:
                    cand_meta = meta.get(cand_id, {})
                    candidates.append({
                        "page_id": cand_id,
                        "title": cand_meta.get("title", cand_id),
                        "path": str(cand_meta.get("path", "")),
                    })
                batch_data["pages"].append({
                    "page_id": pid,
                    "title": page_meta["title"],
                    "path": str(page_meta["path"]),
                    "file_size": page_meta["file_size"],
                    "candidates": candidates,
                })
            out_file = out_dir / f"batch_{bi:02d}.json"
            with open(out_file, "w", encoding="utf-8") as f:
                json.dump(batch_data, f, ensure_ascii=False, indent=2)
            print(f"  batch_{bi:02d}.json: {len(batch)} pages")

        print(f"\nエクスポート完了: {out_dir}")
        print(f"バッチ数: {n_batches}")
        print(f"総ページ数: {len(page_ids)}")
        print(f"平均ページ/バッチ: {len(page_ids) // n_batches}")
        return

    print("\nPass 3: 書き込み...")
    written = 0
    skipped = 0
    errors: list[tuple[str, str]] = []
    for page_id, related_ids in decisions.items():
        page_meta = meta[page_id]
        path = page_meta["path"]
        try:
            new_content = merge_related_section(
                page_meta["content"], related_ids
            )
            if new_content == page_meta["content"]:
                skipped += 1
                continue
            atomic_write(path, new_content)
            written += 1
        except Exception as e:
            errors.append((page_id, str(e)))

    print(f"  書き込み: {written}")
    print(f"  スキップ (変更なし): {skipped}")
    print(f"  エラー: {len(errors)}")

    log_target = find_log_target()
    log_file = log_target / "backfill.log"
    with open(log_file, "w") as f:
        f.write("# Backfill Links Log\n")
        f.write(f"Date: {datetime.now().isoformat()}\n")
        f.write(f"Total pages: {len(pages)}\n")
        f.write(f"Pages indexed: {len(meta)}\n")
        f.write(f"Pages with related: {len(decisions)}\n")
        f.write(f"Pages written: {written}\n")
        f.write(f"Pages skipped: {skipped}\n")
        f.write(f"Errors: {len(errors)}\n\n")
        if errors:
            f.write("## Errors\n\n")
            for pid, err in errors:
                f.write(f"- {pid}: {err}\n")

    print(f"\nログ: {log_file}")


if __name__ == "__main__":
    main()
