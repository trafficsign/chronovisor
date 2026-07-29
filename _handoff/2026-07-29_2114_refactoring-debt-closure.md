---
task_id: repo_refactoring_debt_closure_20260729
created_at: 2026-07-29T21:14:40+09:00
状態: complete
branch: codex/refactor-finish-all
baseline_commit: 211a2f1
---

# Chronovisor refactoring debt closure

Campaign H-L の完了後に残っていた明示的なリファクタリング負債を閉じた。

## 完了内容

- Campaign G:
  incident ledger の実装済み契約と 66 本の回帰テストを監査し、RFCを完了扱いにした。
- 大型関数:
  当初の 300 行以上 38 本を全数棚卸しした。7 本を 300 行未満へ分解し、
  残る 31 本は I/O順序、authority境界、rollback範囲などを理由に維持した。
  `large-function-policy.toml` とASTテストが、新規未審査関数・上限増加・
  分解済み関数の再肥大を拒否する。
- 旧モジュールshim:
  219 本のtop-level forwarding shimと`core.compat`を削除した。
  旧durable module path 223件はcanonical pathへ移行してから実行されるため、
  保存済みbackground jobは継続できる。
- 静的品質:
  Ruffを`E4,E7,E9,F,I,UP,B,SIM,RUF100`へ拡張し、`src/scripts/tests`
  全域の既存違反を解消した。意味が変わり得る4規則だけを理由付きで除外した。
  Mypyは計画どおり17 core moduleでstrictを維持し、外側を広域ignoreで
  見かけ上通す変更はしていない。
- 公開API:
  49 console entry pointのdocstring契約を維持した。

## コミット

- `ceb5c5b` Close implemented incident ledger campaign
- `7c56402` Retire legacy module compatibility shims
- `f031a62` Close the large-function refactor inventory
- `180b8e2` Expand the repository-wide Ruff quality gate

## 最終検証

- 本番Ingest安全ゲート:
  `status=idle`, `pending_raws=0`, `ollama_available=true`,
  `authority_available=true`, `alert=false`, `retryable=false`
- 隔離全スイート:
  `2821 passed, 1 skipped in 579.10s`
- Ruff: 全設定ルール合格
- Mypy: strict core 17 module、0 issue
- Import Linter: 241 files / 1,314 dependencies、1 contract kept / 0 broken
- Python実装モジュール: 228 names、順方向・逆方向import成功
- Console entry points: 49 targets、全てcallable
- Launchd plist: 6 files、全て`plutil -lint`成功
- Compileall / whitespace: 成功
- Authority hash: lane contract、lane case、structured generation policy、
  production schema manifestの4件が変更前と同一

## Workspace

既存の未追跡ファイル
`_handoff/2026-06-11_0042_recall-redesign.md`と`logs/`は変更・追加していない。
作業用codemodは削除済み。

## Deployment

この記録の次に、完成済みHEADを`origin/main`へpushし、
GitHub-backed Dashboard / Ingest-drain / Semanticを再起動する。
runtime archive commit、Dashboard health、Semantic ready/self-test、
Ingest livenessを照合して本番完了とする。
