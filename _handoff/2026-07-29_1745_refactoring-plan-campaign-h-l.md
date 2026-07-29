---
task_id: repo_refactoring_plan_20260729
created_at: 2026-07-29T17:45:00+09:00
状態: plan
baseline: main (c60f773)
前回完了: campaign A–F (2026-07-17, `_handoff/2026-07-17_2348_refactoring-completion.md`)
---

# Chronovisor リファクタリング計画(第2期: Campaign H–L)

## 経緯

前回リファクタ(A–F)完了後の 128 コミットで src に新規 107 ファイルが追加され、
classification / librarian / collection 系を中心に急拡大した。現状:

| 指標 | 値 |
|---|---|
| モジュール数(src/chronovisor 直下、フラット) | 234 |
| 総行数 | 171,356 |
| console entry points | 62 |
| 関数総数 / うち public | 4,225 / 1,952 |
| public 関数の型ヒント完備率 | 99%(良好) |
| public 関数の docstring 率 | 30% |
| 300 行超の関数 | 38 本(最長 896 行) |
| 関数内遅延 import(内部モジュール間) | 344 箇所 |
| top-level import 循環 | 0(規律は維持されている) |

## 全体診断(俯瞰)

### D1. 実験コードと本番コードの混在 — 最重要

classification 系だけで 53 ファイル / 27,251 行。うち pilot / dev / unseen / trial
系は一回性の preregistered 評価ゲートで、結論確定後もパッケージ内に残留している。

- どこからも参照されない真の孤児: 11 モジュール / 約 4,500 行
  (`classification_anchor_complement_dev`, `classification_intent_unseen`,
  `classification_cvo_ab_unseen`, `classification_hierarchy_dev`,
  `classification_query2doc_v2_unseen`, `capability_recovery`, `contract_audit` ほか)
- 実験足場の重複: `_validate_preregistration`×7, `lock_preregistration`×6,
  `run_pilot`×5, `evaluate_unseen`×5, `output_root`×6 が各ファイルにコピーされている
- 近似重複ペア(diff 類似度 0.5 超)が 3 組
- entry points 62 本のうち 10 本以上が実験用

注: `*_worker` 系(`research_model_worker`, `classification_model_worker`,
`deep_retrieval_worker`, `classification_resource_probe`)は `python -m` の
サブプロセス起動、`deadman_observer` はファイルコピー配備で使われており孤児ではない。
モデル常駐メモリ制御のための正当なプロセス分離パターンとして維持する。

### D2. フラット構造とレイヤの暗黙化

サブパッケージゼロの 234 モジュール。依存の集中自体は健全
(`store` fan-in 99, `runtime_config` 72, `durable_state` 61)だが、
機能モジュール間は 344 箇所の関数内遅延 import で結ばれ、実質 68 モジュールの
相互依存圏を形成。レイヤ境界がコード上に存在せず、規約と記憶で維持されている。

### D3. 巨大モジュール・巨大関数

4,000 行超のモジュールが 6 本(content_correction 5,532 / ingest 5,279 /
dashboard 4,627 / autonomy 4,323 / self_heal 4,268 / decision_router 4,208)。
300 行超の関数 38 本。上位:

| 関数 | 行数 |
|---|---|
| self_heal._handle_packet_unlocked | 896 |
| content_correction._process_frontier_item | 876 |
| orphan_link.run_autonomous | 846 |
| ingest_review_apply.review_and_apply_ingest_operations | 781 |
| orchestrator.run_pending_ingest | 769 |
| search_eval.review_label_queue_with_frontier | 727 |
| read_back_repair.run_read_back_repair | 722 |
| autonomy.resolve_deferred_duplicates_with_frontier | 696 |
| ingest.run_ingest | 675 |

直近 128 コミットの churn は collection_authority(2,817)、
classification_library_pilot、librarian_status、classification_engine に集中。
今後も触る場所と巨大関数が重なっている。

### D4. ユーティリティ重複

`_now`×42, `_iso`×11, `_utc_now`×5, JSONL 読み書き×8+, sha256 系×19,
`_atomic_write_json`×5, `_extract_json_object`×5。`jsonl.py` / `jsonl_write.py` /
`canonical_json.py` が既にあるのに各モジュールが私製ヘルパーを持つ。

### D5. 品質基盤のばらつき

型ヒント 99% は優秀。docstring 30%。ruff は運用実績があるが pyproject に
`[tool.ruff]` 設定がなく規約が暗黙。mypy / pyright なし。

## 計画

前回同様のキャンペーン方式。各キャンペーンは独立ブランチ・独立コミットで、
挙動変更ゼロ(バイトレベルの永続データ・authority hash 不変)を原則とする。
Campaign G(incident-ledger RFC)は既存ロードマップのまま据え置き。

### Campaign H: 実験コードの退役と隔離(低リスク・高効果)— 最優先

1. 退役: 結論が確定した preregistered ゲート(真の孤児 11 本+対応テスト)を削除。
   git 履歴とアーティファクト(docs/decisions, マニフェスト)が保存庫。
   判断基準: (a) src/scripts/launchd/entry point から参照なし
   (b) 実験結果が採択済みまたは棄却済み (c) unseen/dev ゲートは再実行の予定なし。
2. 隔離: 継続中・再実行可能性のある pilot / eval 系を `chronovisor/lab/`
   サブパッケージへ移動。entry points を `chronovisor-lab <subcommand>` 1 本に
   集約し、62 本 → 40 本台へ。
3. 共有ハーネス抽出: preregistration ロック、選択固定、出力 root、メトリクス
   集計を `lab/harness.py` に一本化(×7 の重複を解消)。

効果見込み: 約 8,000〜10,000 行の削減・隔離。以後のキャンペーンの対象が減る。

### Campaign I: ユーティリティ統合(機械的・低リスク)

`_now`/`_iso`/`_utc_now` → `timeutil.py`、sha256 系 → `hashutil.py`、
JSONL・atomic write → 既存 `jsonl.py`/`jsonl_write.py` へ集約。
純粋な置換のみ。フォーマット差異(タイムゾーン・ミリ秒有無)は置換前に
全数調査し、差異があるものは触らず記録に残す。

### Campaign J: パッケージ再編とレイヤ契約の機械化(最大リスク)

フラット 234 → ドメインサブパッケージ:

```text
chronovisor/
  core/       store, runtime_config, durable_state, canonical_json, jsonl, timeutil...
  raw/        raw_store, raw_segment, raw_replay, raw_semantic_projection...
  ingest/     ingest, ingest_review_*, ingest_transport, triage...
  recall/     recall_runtime, recall_auditor, recall_auto_apply, recall_breaker...
  search/     search, lexical_index, semantic_index, reranker, embedding...
  classification/  engine, anchor, hierarchy, resolver, workers...
  librarian/  librarian, merge, release, rollout, collection_authority...
  decision/   decision_router, decision_authority, lane contracts/prompts...
  research/   service, orchestrator, tools, verification...
  ops/        dashboard, autonomy, self_heal, burn_monitor, supervisors...
  hosts/      hook_dispatcher, codex_record, claude_code_record...
  lab/        (Campaign H で作成済み)
```

- 後方互換: 旧モジュールパスに再エクスポート shim を残し、entry points(62)・
  launchd plist・scripts/ のフルパス参照を段階的に切替。shim は次期で削除。
- 遅延 import 344 箇所を棚卸し: 循環回避に必要なもの(共有型を `*_types.py` に
  抽出して top-level 化)と、起動コスト回避目的のもの(維持・コメント明記)に分類。
- import-linter を導入し `core → ドメイン → ops/hosts/lab` のレイヤ契約を
  CI 相当のチェックで機械化。手動規約からの卒業が本キャンペーンの本丸。

### Campaign K: 巨大関数の分解(前回 F 方式の継続)

pure seam 方式(判定・変換の純粋関数を抽出し、I/O とオーケストレーションを
細く残す)。全 38 本は狙わず、churn×サイズで上位を選定:

1. `ingest.run_ingest` + `orchestrator.run_pending_ingest`(中核パス)
2. `self_heal._handle_packet_unlocked`(最長)
3. `content_correction._process_frontier_item`
4. `ingest_review_apply.review_and_apply_ingest_operations`
5. collection_authority / classification_engine の直近肥大分

各分解の前に characterization テストの存在を確認し、なければ先に追加。

### Campaign L: 品質基盤の明文化

- `[tool.ruff]` を pyproject に明文化(現行コードが通る最小集合から開始)
- 公開 API(entry point から到達する public 関数優先)へ Google スタイル
  docstring を整備、30% → 段階的に引き上げ
- mypy を core/ から段階導入(strict は core のみ、外側は緩め)

## 順序と検証

順序は H → I → J → K → L。J は H で対象が減ってから行うのが効率的。
K は J の後(移動とシンボル分解を同時にやらない)。

検証は前回 completion のプロトコルを踏襲:

- 全スイート(2,362+ passed 基準)、compileall、whitespace チェック
- lane contract / schema manifest の SHA 照合(挙動不変の証明)
- 最終スイートは production ingest drain が `state=idle, pending=0` に
  自然到達してから実行。稼働中バッチを殺さない
- J のみ追加で: 全 62 entry points の起動スモーク、launchd plist の参照先検証、
  `scripts/` 内 import の順方向・逆方向チェック

## リスクと撤退

- H: 低。削除はコミット単位で revert 可能。退役判断に迷うゲートは削除せず lab/ へ。
- I: 低。ただしタイムスタンプ書式の暗黙差異にだけ注意(事前全数調査で対処)。
- J: 高。shim で旧パスを一世代維持し、launchd / hooks の切替は明示的な
  デプロイ手順として分離(前回同様、push・再起動は完了コミットに含めない)。
- K: 中。characterization テスト先行を厳守。
- 永続データと authority hash は全キャンペーンで不変のため、データ移行・
  rehash は不要。撤退は該当コミットの逆順 revert。
