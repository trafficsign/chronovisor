# Codex 引き継ぎブリーフ: Chronovisor の LLM バックエンドを Ollama → oMLX へ切り替え

状態: **REPO-IMPLEMENTED**(2026-08-20、Hermes が実装。Codex 復活までの暫定)
→ 本番カットオーバー(production config の provider 切替 + サービス再起動)が残作業。
作成: 2026-08-20 / Hermes(oMLX 実測・検証済み)
実装者: Hermes(Codex 復活までの暫定担当)
前提プラン: `_handoff/2026-08-19_2258_model-stack-unify-omlx.md`

## 0b. 実装状況(2026-08-20)

- ✅ `src/chronovisor/core/omlx_adapter.py` 新規(OMLXAdapter / 写像は本ブリーフ §2 の実測どおり)
- ✅ `src/chronovisor/core/llm_config.py` に provider kind `omlx` 追加(gen+embed+structured)
- ✅ `tests/test_omlx_adapter.py` 新規 13 件 + `test_llm_config/llm_runtime/ollama` 合計 174 件 green
- ✅ `scripts/check_local_omlx_e2e.py` 新規、実サーバーで **PASS**(gen 6 役割 + gate 1.34s + embed 1024d)
- ⏳ 残り:**本番 `~/.chronovisor/config.toml` の provider 切替**(roles を omlx に + モデルID 書換)+
  サービス再起動での確認 → その後 Ollama 退役。実装以降の手順: 下記 §4.3 / DoD。

---

## 0. ゴール

Ollama を完全退役し、Chronovisor が使う全モデルを **oMLX(z-lab DFlash fork、localhost:8000、
OpenAI 互換 API)** から提供する。`ollama_adapter.py` 相当の **`omlx_adapter.py`** を実装し、
`llm_runtime.py` のルーティングを oMLX へ切替える。

モデル側の検証は**既に全て通過**(下表)。残りは本ブリーフのアダプタ実装 + config 写像 + 検証。

## 1. 対象モデルと実測値(2026-08-19/20、M4 Max)

| config上の名前(旧) | oMLX モデル ID | 役割 | 実測 |
| --- | --- | --- | --- |
| qwen3.8:27b-axq4 | `Qwen3.8-27B-4bit` | ingest / decision.primary / audit / planner | DFlash2, 45.8 t/s(rc1) |
| gemma4:26b-optiq4 | `gemma-4-26b-a4b-it-4bit` | decision.tie_break / research.tie_break | DFlash v1, 140.8 t/s(rc1) |
| muse-glimmer:30b-q4k-dynamic | `Muse-Glimmer-30B-4bit` | decision.challenger | **DFlash2(rc1)24.1** / v1 20.2 t/s |
| ornith:9b-q4_K_M | **`Ornith-1.5-9B-MLX-4bit`** | recall.gate / recall.processor judge | ウォーム ~0.2s(1.0 と同じ。誤発火も改善傾向) |
| bge-m3:latest | `bge-m3-mlx-fp16` | [embedding] 埋め込み | /v1/embeddings 1024d・正規化 |
| (gpt-oss:20b) | — | research.challenge 等 6 割当 | **不使用**(削除対象、挙動影響なし) |

**バージョン**: デプロイ対象は **oMLX 0.6.3(rc1 で検証済み、クリーン実測: Qwen 45.8 / Gemma 140.8 /
Muse+DFlash2 24.1)**。0.6.3rc1 で Muse の DFlash2 drafter(z-lab/Muse-Glimmer-30B-DFlash2)が
実動する。Muse の model_settings の `dflash_draft_model` は **`z-lab/Muse-Glimmer-30B-DFlash2` を
デフォルト推奨**(v1 の `meta-models/Muse-Glimmer-30B-assistant` はフォールバック)。
Ornith は **1.5(MLX-4bit)を採用**(1.0-OptiQ-6bit は残していてよい/削除可)。

Ollama 非依存(変更不要): `nvidia/Nemotron-3-Embed-1B-BF16`(semantic-service / Unix socket)、
`BAAI/bge-reranker-v2-m3`(プロセス内 transformers / MPS)。

## 2. oMLX API 仕様(実測済み)

- base: `http://localhost:8000/v1`、認証: ヘッダ **`x-api-key: omlx-local`**、`Content-Type: application/json` 必須
- `GET /v1/models` → 登録モデル一覧
- `POST /v1/chat/completions` → 生成(OpenAI 形式)
- `POST /v1/embeddings` → 埋め込み(**検証済み**: 200 / 1024 次元 / L2 正規化済み)
- パラメータ写像(実測ベース):
  - `max_tokens` ✓ / `temperature` ✓ / `seed` ✓ / `messages` ✓
  - **`keep_alive`・`num_ctx`・`think`・`enable_thinking`(トップレベル)は全て無視される**(422 にならず黙殺)。写像ミスが無音で起きる
  - **思考制御は `chat_template_kwargs: {"enable_thinking": false}` で効く**(実測: ornith の Thinking Process が抑制された)← recall.gate の think=false はこれに写像
  - `response_format: {"type":"json_object"}` … **未確認**(要実装時プローブ、ornith で素早く試せる)
- **DFlash エンジンは同時 1 つのみアクティブ**(現ビルド)。非 pinned モデルは自動スワップで切替可(実測: Qwen→ornith→Qwen すべて成功)。**`is_pinned: true` の DFlash モデルが他モデルのロードをブロックする**(実測で遭遇、ornith は現在非 pinned)。

## 3. 対象コード(リポジトリ = ~/projects/personal/chronovisor)

- `src/chronovisor/core/ollama_adapter.py`(171 行)— `OllamaAdapter`(generate/embed/resource_lease/resident_models/unload)+ `compose_ollama_runtime()`
- `src/chronovisor/core/ollama.py`(796 行)/ `ollama_transport.py`(624 行)— Ollama クライアント・チャンク埋め込み等
- `src/chronovisor/core/ollama_lease.py`(215 行)— **プロバイダ中立**のファイル/プロセスロック + admission 制御。このまま流用可(ただし内容は Ollama 非依存なのでロックファイル名 `runtime/ollama-resource.lock` などの表記を汎用化するか判断)
- `src/chronovisor/core/llm_runtime.py`(951 行)— `LLMRuntime` / `GenerationRoute` / `EmbeddingRoute` / 各プロトコル(変更原則不要、アダプタだけ追加)
- 利用側(アダプタ交換で済む想定): `decision_router` / `ingest_generation` / `recall_*` / `research_*` / `classification_*`
- `scripts/check_local_ollama_e2e.py` — e2e 検証の雛形(複製して oMLX 版を)
- 実 config: `~/.chronovisor/config.toml`(root 直下にあるので `CHRONOVISOR_ROOT/runtime` 等の参照に注意)

## 4. 実装方針

### 4.1 `src/chronovisor/core/omlx_adapter.py`(新規)
`OllamaAdapter` と同じインターフェースを実装:
- `generate(request, *, model)` → `POST /v1/chat/completions`
  - `MessageGenerationRequest` → `messages`(role/content)。`GenerationInput`(prompt+system)は単一メッセージへ
  - `max_output_tokens` → `max_tokens`、`temperature`→温度、`seed`→`seed`、`timeout_ms`→ httpx.Timeout
  - `max_output_chars` → クライアント側 truncation
  - **`think`(=False のケース: recall.gate) → `chat_template_kwargs: {"enable_thinking": false}`**(実測済み)
  - `num_ctx` → oMLX には送らない(無視される)。許容: モデル側デフォルト ctx 依存か、`model_settings.json` での設定へ移行
  - `keep_alive` → 送らない。常駐制御は oMLX(`model_settings.is_pinned`)+ Chronovisor リースに委譲
  - `format` / structured_output → 実装時に `response_format` プローブで確認
  - エラー写像: httpx タイムアウト/接続 → `SafeBackendError("timeout"/"transport_error", transient=True)`(OllamaAdapter と同型)
- `embed(request, *, model)` → `POST /v1/embeddings`。`EmbeddingResult(vectors=tuple(tuple(...)), provider="omlx", model=...)`
- `resource_lease(...)` → `ollama_lease.model_resource_lease` を再利用(プロバイダ中立)
- `resident_models()` / `unload()` → oMLX API は residency/unload を公開していない可能性が高い。方針:
  - `resident_models` を実装できないなら `local_controls` から外し、`adaptive_residency` 関連は `config で無効化 or oMLX 側の管理に委譲`(下記 4.3)
  - `unload` は実装不可なら no-op(真偽を実装時プローブで確認)
- 接続先: `OMLX_BASE_URL`(+ 既定 `http://localhost:8000/v1`)と `OMLX_API_KEY` を config/env から注入。**ハードコード禁止**

### 4.2 `compose_omlx_runtime()`
`compose_ollama_runtime` と同型で、generation_roles / embedding_roles → oMLX モデル ID のマップ(§1 の表)。
`BackendCapabilities`: ollama 版と同様(`structured_output=True` は response_format プローブ後に調整)。

### 4.3 config 写像(`~/.chronovisor/config.toml` + コード)
- 全 `model = "xxx"` / `*_model = "xxx"` を §1 の oMLX ID へ置換(例: `[ingest] model = "Qwen3.8-27B-4bit"` …)
- バックエンド選択: provider を設定で切替(`[llm] provider = "omlx"` 相当)。compose 関数の選択は既存呼び出し箇所で
- `[decision_router]` / `[research]` の `keep_alive`/`num_ctx` → oMLX では意味をなさないため整理(これらは Ollama 固有の residency 制御)。`adaptive_residency` / `max_resident_models` / `coordinate_ollama` / `sync_reserved_headroom_gib` 等は oMLX モデルで妥当な値を再設計
- research の `protected_models` / `require_protected_residency` → oMLX の residency モデルに合わせて見直し(ornith=gate は非 pinned でスワップ許容の判断もあり得る)
- gpt-oss 割当 6 箇所(`research.challenge` / `librarian.review.challenger` / `classification.anchor[.challenger]` / `recall.distill.teacher.c` / line 292 等)を削除 or 有効モデルへ統合(実使用ゼロのため挙動影響なし)
- 埋め込み役割の `bge-m3` → `bge-m3-mlx-fp16`。埋め込み API の呼び出し箇所(ollama.py / ollama_transport.py 内の embed)は `/v1/embeddings` へ

### 4.4 e2e / テスト / 検証
- `scripts/check_local_omlx_e2e.py`(ollama 版を複製、oMLX 接続に差し替え)
- adapter 単体テスト: `httpx.MockTransport` で chat/embeddings 応答を固定し、写像(think→chat_template_kwargs など)をテストで固定
- 本番スモーク(converge / semantic-service 稼働下): ingest + decision_router + recall.gate が oMLX 経由で通ること
- **recall.gate のレイテンシ**: 設定 1.5s 以内であること(本番スモークで計測。DFlash スワップが絡むと 1 発目が遅い可能性 → 計測・判断)

## 5. 注意・制約(実測で判明)

1. **DFlash エンジンは同時 1 つのみアクティブ**。非 pinned なら自動スワップ(実測 OK)。**`is_pinned: true` DFlash モデルは他モデルのロードをブロック** — pinned 使用は慎重に。Chronovisor の既存リース(直列化)は維持・活用が必須(D Flash 有効モデルが 5 つに増えるため重要度↑)
1b. **Muse + DFlash2 は 0.6.3 で実動**(rc1 クリーン実測 24.1 tok/s)。`dflash_draft_model` を `z-lab/Muse-Glimmer-30B-DFlash2` に設定すれば Muse target でも投機が効く。
2. **未知パラメータは黙って無視される**(keep_alive/num_ctx top-level think 等)。写像漏れが無音 = 動作違いになるので、テストで挙動を固定すること
3. 思考抑制は `chat_template_kwargs.enable_thinking: false`(トップレベル enable_thinking は効かない)
4. 埋め込みは 1024 次元・正規化済み(既存 bge-m3 と同じ次元。Ollama 版の前処理(prefix)は不要前提で確認)
5. `response_format: json_object` は未確認。実装時に ornith を使って 1 プローブで確定
6. Ollama 非依存サービス(Nemotron / reranker)は変更しない

## 6. 完了条件(Definition of Done)

- [ ] `check_local_omlx_e2e.py` が green
- [ ] adapter 単体テスト(写像含む)追加・green
- [ ] 3 先生 + ornith の切り替え生成が Chronovisor 経由で通る(手動スモーク 1 回以上)
- [ ] 埋め込み(分類/検索)が oMLX 経由で通る
- [ ] recall.gate 実測レイテンシ <= 1.5s
- [ ] 関連テストスイート regression なし
- [ ] 検証した意味のある単位で commit(push は明示依頼時のみ)

## 7. 参考

- 前提プラン: `_handoff/2026-08-19_2258_model-stack-unify-omlx.md`(commit c172080 まで)
- oMLX 運用: `~/.omlx/`(settings.json に API キー、model_settings.json にモデル設定、models/ は HF キャッシュへの symlink)
- モデル登録ツール: `~/projects/sandbox/tools/register_omlx_ornith_bge3.py`(symlink + model_settings 注入の雛形)
- スキル: `dflash2-mlx-mac`(oMLX セットアップ・計測・メモリ運用の教訓)
- 不明点は推測せず、oMLX に 1 プローブ(curl)を打って実測してから実装を進めること
