---
task_id: plan_c71a785041df458c83361eb12cba88f4
created_at: 2026-04-10T17:18:31+09:00
状態: running
---

## 目的
- Claude の記憶システムを0ベースで再設計し、「使えば使うほど Claude が賢くなる」状態を実現する
- 理想像: ジャービス（アイアンマン）— 優秀な執事であり友達

## 設計要件

### ゴール
「使えば使うほど Claude が賢くなって、ユーザーは楽をする」

### 記憶システムの利用者
- 主ユーザーは **Claude（AI側）**。ユーザーは何もしなくても精度が上がるのが理想
- ユーザーに記憶の管理・整理を一切させない

### 設計方針（技術調査を経て確定）

**責務の分離:**
- **ユーザーモデル（好み・性格・行動パターン）→ フロンティア AI に任せる**
  - Anthropic が memory 機能・パーソナライゼーションとして積極的に解決する領域
  - 自前で投資する必要なし。世代ごとに勝手に良くなる
- **知識の構造化（技術知見・調査結果・プロジェクト固有データ）→ ローカルで自前構築**
  - セキュリティ・プライバシー上、外に出せないデータ
  - フロンティア AI が代替できない領域。ここが自分で構築する価値がある

**アーキテクチャ方針: カーパシー LLM Wiki パターン + ローカル LLM / Sonnet フォールバック**

```
[Claude Code セッション（Opus）]
  ↓ 会話・実装・判断
[保存 hooks トリガー]
  ↓
[Sonnet（バックグラウンド）= 司令塔]
  ├→ 1. raw/ にセッションログ書き出し
  ├→ 2. 「Ingest すべきか？」を判定（内容の量・重要度に基づく）
  └→ 3. Yes の場合:
       ├→ Ollama ヘルスチェック（キャッシュ済みなら再利用、失敗は1時間キャッシュ）
       ├→ Ollama 生存 → ローカル LLM で Ingest
       └→ Ollama 死亡 → Sonnet 自身で Ingest + Opus に報告（「Ollama停止中、直しますか？」）

[次のセッション開始時]
  → Claude Code が pages/ から Wiki 読み込み → 賢くなってる
```

**コンポーネント:**
| 役割 | 担当 | 備考 |
|---|---|---|
| 会話・実装・判断 | Claude Code (Opus) | 生データの生産者であり、Wiki の消費者 |
| 司令塔（保存判定・キック） | Sonnet（バックグラウンド） | 会話をブロックしない。保存 + 判定 + キック |
| Ingest（構造化） | ローカル LLM（優先） / Sonnet（フォールバック） | 生データ → 構造化 Wiki ページに変換 |
| Lint（品質管理） | ローカル LLM / Sonnet | 矛盾検出、古い情報フラグ、孤立ノート統合。前回から24時間以上で実行 |
| Compile（知識育成） | ローカル LLM / Sonnet | 関連ページの相互参照、知識の統合 |
| Query（検索） | Claude Code | セッション中に Wiki を検索・参照 |
| ユーザーモデル | Anthropic（フロンティアAI） | 自前で作らない |

**処理エンジン切り替え:**
- Ollama（ローカル LLM）が使えるなら優先。API 代ゼロ、データが外に出ない
- Ollama が停止/未インストールの環境では Sonnet にフォールバック
- ヘルスチェック結果を1時間キャッシュ（失敗時に毎回リトライしない）
- フォールバック時は Opus 経由でユーザーに即時報告（次のセッションまで放置しない）
- インターフェース統一: 裏が Ollama でも Sonnet でも Wiki 操作は同じ

**Ingest トリガー:**
- 固定間隔ではなく **Sonnet が文脈を見て判断**
- 保存 hooks のタイミングで Sonnet が「今 Ingest すべきか」を判定
- 重要な技術的決定が続いた → すぐ Ingest
- 雑談が続いてるだけ → まだいい
- セッション中にローカル LLM が走っても Ollama の keep_alive=0 で処理後即アンロード

**ローカル LLM 候補:**
- Gemma 4 26B MoE — 256K コンテキスト、Apache 2.0、M4 Max で余裕で動作
- Qwen 3.6 系のローカル版（サイズ次第）
- Ollama 0.19 + MLX バックエンド（Apple Silicon 最適化済み）

### Wiki Schema（確定）

**ページ粒度:** 1エンティティ1ページ（カーパシー準拠）

**命名規則:** kebab-case.md（英語）
例: `jt-v10-probability-contexts.md`, `studio-display-xdr-multi-monitor.md`

**フロントマター:** 最小限（AI-first、タグ等は本文から LLM が判断）
```yaml
---
title: ページタイトル
updated: 2026-04-10
---
```

**相互参照:** `[[wiki-link]]` 記法
例: `[[jt-v10-probability-contexts]]`

**更新ルール:** LLM 判断に委ねる（ガイドラインのみ）
- 明確な誤りの訂正 → 上書き
- 状態の変化 → 経緯を残して更新
- 迷ったら追記して Lint に任せる

### ディレクトリ構造（確定）

```
~/.wiki/
├── raw/          ← 永続。セッションごとの生データ（原本として遡れるよう保持）
├── pages/        ← 永続。構造化された Wiki ページ
├── index.md      ← カタログ（LLM が自動更新）
├── log.md        ← 時系列ログ（追記のみ）
└── schema.md     ← Wiki のルール（上記の内容）
```

### MCP エンドポイント（確定、Codex レビュー反映済み）

| エンドポイント | 用途 |
|---|---|
| `wiki.search(query, depth=1)` | チェーン検索。direct_hits / expanded_hits / edges を区別して返す |
| `wiki.read(page)` | ページ内容 + outlinks + backlinks |
| `wiki.index(limit, cursor)` | 構造化メタデータ（JSON、ページネーション対応） |
| `wiki.ingest(content)` | 非同期。job_id を返す |
| `wiki.jobs(job_id)` | ジョブ進捗確認 |
| `wiki.check()` | Lint 検出のみ。問題リストを返す |
| `wiki.apply()` | 安全な修正のみ自動実行（壊れたリンク、孤立ページ等）。矛盾はフラグのみ |
| `wiki.log()` | 変更履歴 |
| `wiki.status()` | Wiki 状態 + Ollama 状態 + 基本統計 |
| `wiki.provenance(page)` | ページの根拠（元セッション・raw データへのリンク） |

**search の返り値フォーマット:**
```json
{
  "direct_hits": [
    {"page_id": "...", "title": "...", "updated": "...", "score": 0.93, "snippets": ["..."]}
  ],
  "expanded_hits": [
    {"page_id": "...", "title": "...", "distance": 1, "via": ["..."], "score": 0.58}
  ],
  "edges": [
    {"from": "...", "to": "...", "type": "wikilink"}
  ]
}
```

**Codex レビューからの採用事項:**
- オーケストレーションをコードに移す（Sonnet は構造化の中身だけ担当、制御フローは決定的コード）
- Ingest トリガー: 決定的ルール（未処理 raw が N 件溜まったら）+ 重要度ランキング
- Lint を check / apply に分離（矛盾は検出のみ、安全なものだけ自動修正）
- ingest は非同期 + job tracking
- related 廃止 → read に backlinks/outlinks 統合
- index を構造化メタデータ（JSON）に変更
- raw を provenance に変更（ページ単位で根拠を辿る）
- stats は status に統合

**Codex レビューから不採用としたもの:**
- サイドカー SQLite — シンプルに Markdown + ファイルシステムで始める。必要になったら足す
- Ollama キャッシュの指数バックオフ — 15分固定キャッシュで十分

### Lint 詳細（確定）

**実行タイミング:** 前回から24時間以上経過で実行

**チェック項目（wiki.check）:**
1. 矛盾検出 — リンクで繋がったページ同士の内容食い違い
2. 陳腐化検出 — updated が古いページをフラグ
3. 孤立ページ — どこからもリンクされていないページ
4. 壊れたリンク — 存在しないページへの `[[wiki-link]]`
5. 重複検出 — 同じトピックの複数ページ

**自動修正（wiki.apply）:**
- 壊れたリンク → 修正 or 削除
- 孤立ページ → 関連ページとのリンク追加 or 統合
- 重複ページ → 統合
- **矛盾はフラグのみ** → 次に Claude が該当ページを読んだときに判断・解消

### Query 設計（確定）

**セッション開始時:** wiki.index() でカタログ読み込み（全体像把握、コンテキスト消費最小限）

**会話中:** ユーザーの発言から関連トピック判断 → wiki.search() で該当ページ特定 → wiki.read() で個別読み込み

**スケーリング:** ページ数が1000超えたらセマンティック検索を検討（現時点では YAGNI）

### 必須機能（カーパシー LLM Wiki パターンより）
1. **全データ蓄積（Ingest）** — 会話・Web検索・コード・判断、全部取りこぼさない
2. **構造化（Schema）** — 何をどこにどういう粒度で保存するかのルール
3. **品質管理（Lint）** — 古い情報・矛盾・重複の検出と安全な自動修正
4. **検索性（Query）** — 必要な時に必要な知識が出てくる
5. **複利効果（Compile）** — 溜まったデータが整理されて知識に育つ

### 技術調査で得た取り込むべきアイデア
- **A-MEM（NeurIPS 2025）**: 新記憶追加時に既存記憶も連鎖更新（Zettelkasten 方式）
- **TiMem**: 生体験 → セッション要約 → 確定知識の3層抽象化
- **PersistBench**: 技術/雑談のドメイン分離（cross-domain leakage 防止）
- **Letta Sleep-time Compute**: アイドル中の記憶再構成
- **MehmetGoekce L1/L2**: 「この知識なしにミスしたら危険→L1常時、不便なだけ→L2オンデマンド」

### 廃止するもの
- **Vestige** — FSRS 忘却曲線は AI には不要（人間の忘却はキャパ制約の最適化、AI にはその制約がない）。raw/ + pages/ + 賢い検索で代替
- **Basic Memory** — pages/ に統合
- **auto memory (MEMORY.md)** — pages/ に統合

### 現行システムの問題点（刷新理由）
- Ingest は回っているが Schema / Lint が不在
- 保存はしているが体系化されていない
- 読み出しにルールがなく、ぐちゃぐちゃなデータが溜まるだけ
- セッション開始時に読み込むが実際の会話に活きてない実感
- ユーザーが意識的にアクセスしなきゃいけないツールは結局使わなくなる（Letta の教訓）

### 参考資料
- [Karpathy LLM Wiki gist](https://gist.github.com/karpathy/442a6bf555914893e9891c11519de94f)
- [LLM Wiki vs Wikidata: ナラティブ×セマンティック](https://zenn.dev/knowledge_graph/articles/llm-wiki-wikidata-narrative-semantic)
- [Karpathy LLM Knowledge Base — クラスメソッド](https://dev.classmethod.jp/articles/karpathy-llm-knowledge-base/)
- [A-MEM (NeurIPS 2025)](https://arxiv.org/abs/2502.12110)
- [TiMem](https://arxiv.org/abs/2601.02845)
- [PersistBench](https://arxiv.org/abs/2602.01146)
- [MehmetGoekce/llm-wiki](https://github.com/MehmetGoekce/llm-wiki)
- [Gemma 4 — Google DeepMind](https://deepmind.google/models/gemma/gemma-4/)

### 制約・環境
- Mac Studio M4 Max 128GB RAM
- ローカル完結志向（プライバシー重視）
- PC はスリープ運用（常時起動しない）→ ローカル LLM は PC 起動中のみ稼働
- Anthropic サーバー側の scheduled trigger も補助的に使える

## やってはいけないこと（必須）
- 現行システム（Vestige / Basic Memory / auto memory）の前提に引きずられない — 0ベース設計
- ツール選定から入らない — 要件が固まってから技術を選ぶ
- ユーザーに記憶管理の手間を負わせる設計にしない

## 前提・仮定
- Claude Code のセッションは毎回リセットされる（コンテキストウィンドウは持ち越せない）
- 記憶の読み書きは MCP ツール経由で行う
- セッション間の記憶共有は外部ストレージに依存する
- 「ハズレ」モデルが来てもシステムとして最低限機能する必要がある

## 実行手順
- [x] 理想像の言語化
- [x] 設計要件の定義
- [x] 技術調査（既存ツール・研究論文・コミュニティ実践）
- [x] 設計方針の確定（ユーザーモデル→フロンティアAI / 知識構造化→ローカルLLM）
- [x] アーキテクチャ詳細設計（Schema 定義・データフロー・ディレクトリ構造）
- [x] MCP インターフェース設計（10エンドポイント、Codex レビュー済み）
- [x] Lint / Query ワークフロー詳細設計
- [x] ローカル LLM 選定・検証（Gemma 4 26B MoE、Ollama 0.20.5 + MLX で検証済み）
- [x] 実装計画

## 実装計画

### 技術スタック
- **言語:** Python（ユーザーが読める、MCP SDK あり、プロトタイプ速度重視。必要なら後で Rust 移植）
- **ローカル LLM:** Gemma 4 26B MoE via Ollama 0.20.5（MLX バックエンド）
- **フォールバック:** Sonnet API
- **MCP:** Python MCP SDK

### Phase 1: Wiki MCP サーバー（コア）✅
- [x] 1-1. プロジェクト骨格（pyproject.toml、MCP SDK セットアップ、リポジトリ作成）
- [x] 1-2. `~/.wiki/` ディレクトリ構造の初期化 + schema.md 作成
- [x] 1-3. 基本エンドポイント: wiki.read, wiki.index, wiki.log, wiki.status
- [x] 1-4. wiki.ingest（非同期、Ollama API 連携、job_id 返却）— E2Eテスト済み、3ページ自動生成確認
- [x] 1-5. wiki.jobs（ジョブ進捗確認）
- [x] 1-6. wiki.search（チェーン検索、direct_hits / expanded_hits / edges）— テスト済み
- [x] 1-7. wiki.check / wiki.apply（Lint 検出 + 安全な自動修正）— broken link修正テスト済み
- [x] 1-8. wiki.provenance（ページの根拠追跡）
- 追加: wiki.save_raw（raw/ への書き出し + 閾値チェック）
- 追加: wiki.tick（手動オーケストレーション）

### Phase 2: オーケストレーター（決定的コード）✅
- [x] 2-1. raw/ 未処理ファイル監視 + N 件トリガー（閾値5件、テスト済み）
- [x] 2-2. Ollama ヘルスチェック + 15分キャッシュ
- [x] 2-3. Sonnet フォールバック（TODO: API 実装）+ Opus への即時報告（設計済み）
- [x] 2-4. Lint 24時間トリガー
- [x] 2-5. 状態永続化（.orchestrator_state.json）

### Phase 3: Claude Code 統合（進行中）
- [ ] 3-1. hooks で raw/ への書き出し（保存トリガー時）
- [ ] 3-2. セッション開始フロー変更（Vestige/Basic Memory → wiki.index + wiki.search）
- [ ] 3-3. CLAUDE.md / rules/ 更新
- [x] 3-4. claude_desktop_config.json に MCP サーバー登録済み

### Phase 4: 移行・廃止 ✅
- [x] 4-1. 既存データ移行: Basic Memory 1369ファイル → pages/、Vestige 356ノード → raw/
- [x] 4-2. 旧システム廃止: Vestige/Basic Memory を claude_desktop_config.json から削除、CLAUDE.md/rules/ 更新、permissions 整理
- [ ] 4-3. 動作検証（3セッション以上で実運用テスト）— 次のセッションから開始

## 実行中更新ルール
- 実行手順のチェックは、各手順の完了直後に更新する（まとめて更新しない）。
- チェックを更新したら、同時に `実行ログ` へ時刻付きで追記する。
- 中断時は `状態` を `blocked` にし、理由を `ブロッカー` に記入する。

## 完了条件
- 記憶システムの全体アーキテクチャが設計書として定義されている
- Schema（何をどこにどう保存するか）が具体的に定義されている
- Ingest / Query / Lint のワークフローが定義されている
- 実装に着手できるレベルの具体性がある

## 検証手順
- 設計が全設計要件を満たしているか照合
- 「ハズレモデルでも最低限機能するか」のシナリオ検証
- ユーザーの手間がゼロであるか確認

## ブロッカー（発生時のみ）
- なし

## 受け渡しメモ（次担当へ）
- 設計は全て確定済み。このPlanの「設計要件」セクションが設計書本体
- 実装は Phase 1 から順に。Phase 1 完了時点で単体テスト可能
- Ollama 0.20.5 + Gemma 4 26B は既にインストール・検証済み（`ollama list` で確認）
- Codex のレビュー結果は「Codex レビューからの採用事項」に反映済み
- 迷ったらこの Plan の設計要件に立ち返ること


## 実行ログ
- 2026-04-10 17:18 Plan 作成、理想像の言語化と設計要件の定義完了
- 2026-04-10 18:00 技術調査完了（Mem0, Letta, Zep, A-MEM, TiMem, PersistBench, Karpathy派生等）
- 2026-04-10 18:15 設計方針確定: ユーザーモデル→フロンティアAI任せ / 知識構造化→ローカルLLM+カーパシーWikiパターン
- 2026-04-10 18:20 Plan 更新: アーキテクチャ方針・コンポーネント・参考資料・実行手順を反映
- 2026-04-10 19:00 Schema 確定: 1エンティティ1ページ、kebab-case、最小フロントマター(title+updated)、[[wiki-link]]、更新はLLM判断
- 2026-04-10 19:10 ディレクトリ構造確定: ~/.wiki/ (raw/ + pages/ + index.md + log.md + schema.md)、両方永続保存
- 2026-04-10 19:20 Vestige 廃止決定: AI に忘却は不要。全データ永続 + 検索で代替
- 2026-04-10 19:30 データフロー確定: Sonnet=司令塔（保存+判定+キック）、Ingest はローカルLLM優先/Sonnetフォールバック、ヘルスチェック1時間キャッシュ、フォールバック時は即時ユーザー報告
- 2026-04-10 19:35 Ingest トリガー確定: 固定間隔ではなくSonnetが文脈判断。Ollama keep_alive=0 で処理後即アンロード
- 2026-04-10 19:50 Query 設計確定: セッション開始時に index.md、会話中に都度 search+read
- 2026-04-10 20:00 MCP エンドポイント設計完了（10個）。Codex レビュー実施
- 2026-04-10 20:10 Codex フィードバック反映: オーケストレーションをコードに移す、Lint を check/apply 分離、ingest 非同期化、search に理由含める、related→read統合、index→JSON化、raw→provenance変更
- 2026-04-10 20:15 Lint 詳細設計確定: 5種チェック、安全なものだけ自動修正、矛盾はフラグのみ
- 2026-04-10 20:30 ローカル LLM 確定: Gemma 4 26B MoE。Ollama 0.16.1→0.20.5 アップデート（Homebrew 移行）、構造化テスト合格
- 2026-04-10 20:45 実装計画確定: Python、4 Phase 16 ステップ。状態を ready に移行
- 2026-04-10 21:00 Phase 1 完了: MCP サーバー全12エンドポイント実装・テスト済み。初回コミット
- 2026-04-10 21:15 Phase 2 完了: オーケストレーター（閾値ベース Ingest + 24h Lint）実装・テスト済み
- 2026-04-10 21:30 Phase 3 進行中: claude_desktop_config.json に llm-wiki MCP 登録済み。hooks 書き換え・CLAUDE.md 更新は残
- 2026-04-10 21:45 Phase 3 完了: wiki-save.py hook 作成、CLAUDE.md/rules/memory.md 更新、settings.json 更新
- 2026-04-10 22:00 Phase 4-1 完了: Basic Memory 1369ファイル→pages/、Vestige 356ノード→raw/ 移行。index 再構築（1373ページ）
- 2026-04-10 22:10 Phase 4-2 完了: Vestige/Basic Memory を MCP 設定から削除。データは保持（ロールバック可能）
- 2026-04-10 22:15 全Phase実装完了。残りは 4-3 動作検証（3セッション以上）のみ
