---
task_id: repo_refactoring_campaign_b_20260717
created_at: 2026-07-17T20:57:01+09:00
状態: complete
branch: codex/refactor-campaign-b-deletions
baseline_commit: 3b7cad0
---

# Campaign B: safe legacy removal

## Implemented

- `normalize_recall_config()`を削除し、`recall_runtime._apply_config()`がunified `[recall.*]` とlegacy top-level sectionを直接読むようにした
- unified / legacy configから生成される`RecallPolicy`全fieldのparity testを追加した
- embedding SQLite connectionをraw openとone-shot legacy migration orchestrationへ分離した
- read pathの重複migration callを削除し、migration自身はraw connectionだけを使うためlock再入を起こさない
- rollback用`.embeddings.json`は削除せず、従来どおり残す

## Intentionally retained

- legacy `recall.toml` fallbackとsemantic-hold observation: in-flight authority fingerprint互換のため保持
- deprecated hook `audit` / `improve`: documented removal date 2026-10-01前のため保持
- console entrypoints: replacement CLIが未整備のものは保持

## Verification

- targeted config/search/hook/hold tests: 121 passed
- `normalize_recall_config` / `_maybe_migrate_legacy_json` references: zero
- `git diff --check`: pass

## Rollback

Campaign B commitをrevertする。legacy files、persistent SQLite schema、semantic-hold fingerprintは変更していない。
