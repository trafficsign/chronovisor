# Codex 引き継ぎブリーフ: Chronovisor の LLM バックエンドを Ollama → oMLX へ切り替え

状態: **CUTOVER-COMPLETE / KNOWN SCHEDULER LIMIT**(2026-08-20 21:43 JST)
→ GitHub-backed runtime、production config、Ollama 退役まで完了。並行 gate SLA だけ既知制約として残す。
作成: 2026-08-20 / Hermes(oMLX 実測・検証済み)
実装者: Hermes(Codex 復活までの暫定担当)
前提プラン: `_handoff/2026-08-19_2258_model-stack-unify-omlx.md`

## 0a. Codex 再検証の訂正結論(2026-08-20)

**0c2 / 0c3 の「oMLX は約 10K token 超を処理できない」という結論は撤回。**
旧プローブは文字列反復回数を token 数と誤認し、短い client timeout を server failure とした。
target tokenizer で chat-template 後の総 token 数を厳密に作った clean・serial 試験では、
Qwen3.8-27B-4bit + DFlash2 が 16,384 / 32,768(cold) / 65,536 token をそれぞれ
69.34 / 149.40 / 339.98 秒で server completion した。

- 実 ingest unit: `files_processed=1`、`files_failed=0`、`processor=omlx`、wall 127.6 秒。
- 実 runtime e2e: Qwen 4 role、Muse challenger、Gemma tie-break、Ornith gate、bge-m3 が成功。
  HTTP 409 は 0、gate wall 0.708 秒、embedding 1024d。
- 根因修正: string think → `reasoning_effort`、raw prompt の thinking OFF、DFlash model の
  cross-process lease、oMLX 3 teacher の formatless-first + client validation。
- oMLX 設定: teacher 3モデルは DFlash ON、Ornith 1.5 は DFlash OFF + pinned。
- 常駐 Chronovisor / Ollama を止めると 16K は 30.2%、32K は 19.0%短縮した。
  競合は実在するが、clean でも 64K prefill は約 340 秒かかる。
- gate は serial では 0.708 秒だが、15,024-token Qwen prefill と同時では 9.33 秒。
  oMLX に request priority / preemption はなく、1.5 秒 SLA の production canary は未完了。

production config と GitHub-backed runtime は oMLX へ切替済み。最終証拠は §0d を参照。

## 0b. 実装状況(2026-08-20)

- ✅ `src/chronovisor/core/omlx_adapter.py` 新規(OMLXAdapter / 写像は本ブリーフ §2 の実測どおり)
- ✅ `src/chronovisor/core/llm_config.py` に provider kind `omlx` 追加(gen+embed+structured)
- ✅ `tests/test_omlx_adapter.py` 15 件 + focused suite 合計 213 件 green
- ✅ `scripts/check_local_omlx_e2e.py`、実サーバーで **PASS**
  (Qwen 4 role + Muse + Gemma + gate 0.708s + embed 1024d)
- ✅ 本番 `~/.chronovisor/config.toml`、GitHub-backed services、Ollama 退役まで完了。残件は
  background prefill と同時の gate 1.5 秒 SLA、および既存 broad-suite 不一致のみ。

## 0d. 最終 cutover 証拠(2026-08-20 21:43 JST)

- oMLX cutover code: `a97fa170df88bc8d8d1146117537167d8b548585`。本記録を含む最終 `main` で
  dashboard health の `commit_id` / `expected_commit` は一致し `drift=false`。semantic archive も
  最終 `main`、`ready=true`。
- production 44 role: 41 oMLX + local reranker 1 + Nemotron semantic 2。decision は
  Qwen / Muse / Gemma、librarian / research は Qwen / Muse。distillation は Qwen / Muse / Gemma の
  pinned revision fingerprint 3 本で稼働可能。
- Ollama LaunchAgent は disabled + bootout、port 11434 / process なし。rollback config は
  `~/.chronovisor/config.toml.bak-before-omlx-cutover-20260820-6175212`。
- Ollama 停止下の clean E2E は全 role PASS。Qwen 4 role、Muse、Gemma、Ornith gate 0.553 秒、
  bge-m3 2 x 1024d。ingest liveness は `ready` / `alert=false` / pending 1773。
- 強制停止で実行中 DFlash request を abort した直後だけ model-switch 409 が failure cache に入った。
  admin model reload 後は再び全 role PASS。更新時は共有 lease 解放後に agent を停止する。

## 0c. 本番カットオーバー試行結果(2026-08-20) — 未成立 → ロールバック

- 実施: config を omlx 切替(41 role)+ エージェント再起動 + push(remote HEAD に反映)。
- **結果: ingest worker が「ingest ランタイム待ち」のまま進行せず → カットオーバー未成立と判断し、
  config をバックアップから Ollama へ復元・再起動でロールバック完了。**
- 発見した真因(注): 停滞の直接原因は omlx 切替ではなく、**当該日のクリーンベンチ
  (SIGSTOP + ollama bootout)の副次損傷**で、ingest の子 Python が `ollama-resource.lock` /
  `ingest-orchestrator.lock` を掴んだまま待機(自己デッドロック)。**ロック保持者を kill して再起動で解除済み。**
- **残る統合ギャップ(本番適用の真の障壁)**: ingest / decision の「ランタイム獲得・リース・
  residency」は Ollama 結合(keep_alive / adaptive_residency / ollama-resource lock 連携)。
  OMLXAdapter の `resident_models→{}` / `unload→False` だけではこの結合を満たせない。
  → アダプタ実装は完成だが、**本番適用にはこの結合を解く統合作業が別途必要**(Codex 復活後 or 専用セッション)。
- 教訓(スキル/メンテ): クリーンベンチの SIGSTOP 運用は学校のロック状態を壊し得る。
  停止は「launchd でサービス一括停止 → 計測 → 一括復帰」へ変更し、ロック保持者は lsof で確認。

## 0c2. 旧カットオーバー診断(大入力ハング判定は撤回)

**前回 0c の「統合ギャップ = リース/residency の Ollama 結合」は誤診。真因は oMLX 側。**

- 再試行で立証できた事実:
  - config→omlx(41 role)は route 解決・authority preflight(`ingest_runtime_available:true`/
    `authority_available:true`)とも **omlx で正常**。エージェント子は起動時に oMLX(:8000)へ接続。
  - 本番経路 `runtime_generate`(ingest role)直接呼び出しは **API キー無しでも 0.7 秒成功**
    (アダプタの `OMLX_API_KEY` デフォルト `omlx-local` が有効)。→ 認証・config・アダプタは全て正常。
  - **ただし per-unit バッチ(転写サイズの大入力)は oMLX 0.6.3rc1 の HTTP サーバーがハング**:
    モデルをロードせず無応答(直接テスト: 120KB 入力 → 90秒+無応答 / 小入力は 0.7s 応答)。
    サーバーは後続要求を全て塞ぐ = **単一エンジンの大入力デッドロック**。
  - drain の各ユニットは 180 秒 read_timeout(バックエンドの既定)を消化 → `runtime_backend_error`
    → リースを掴んだまま全ユニット繰り返し = **実質永久停滞(provider 無関係の見え方になる)**。
- 学校の停滞全体像: 06:43 の `ConnectError` ブロックはベンチの Ollama bootout の巻き添え。
  その後 Ollama 復帰後も停滞が続いたのは、**初回カットオーバー(07:18)の oMLX リクエストで
  サーバーが既に貼り付いていたため**(config ロールバックではサーバーは復帰しない)。
- **現状**: config は Ollama にロールバック、ingest-drain 復帰、14 エージェント生存、
  pending 1838→1788 と消化再開。oMLX 0.6.3rc1 は起動したまま(未使用、要再起動で復帰)。
- **残タスク(oMLX 本番適用の真の障壁)**: oMLX 0.6.3rc1 の大入力ハング(z-lab 側 rc バグ)。
  選択肢: ①上流(z-lab)へ再現報告 & 次リリース待ち ②model_settings 等で入力上限を稼働側で抑える調査
  ③アダプタ側で大入力を分割送信する暫定回避(本質解ではない)。Ollama 完全退役はハング解消後。

## 0c3. 旧境界判定(文字数推定のため撤回)

**結論: oMLX 0.6.3rc1 は「~10K トークン超の入力」を実用時間内に処理できない。DFlash 有無・
コンテキスト窓設定は無関係。→ Chronovisor の ingest(32K〜256K)・decision(16K〜114K)は
両方 oMLX 射程外。Ollama との「二本立て」が現実解。**

- 実測(M4 Max・Qwen3.8-27B-4bit・max_tokens=4):
  - 4K chars(≈1.3K tok): OK 12s / 10K chars(≈3.3K tok): OK 9.7s(ウォーム)
  - 30K chars(≈10-18K tok): **90s+ 無応答** / 120K chars(≈40K tok): **180-240s 無応答**
  - DFlash ON/OFF で挙動不変(問題は prefill/サーバー側で decode 加速の DFlash は無関係)
  - `sampling.max_context_window` を 32768→131072 に上げても **40K tok は 180s+ 無応答**
    = コンテキスト窓設定では解決しない(サーバー処理の実質限界/潜在バグ)
- **config 実要件との突合**: `[ingest] num_ctx=32768, max_num_ctx=262144` /
  `[decision_router] num_ctx=114688, min_num_ctx=16384` → いずれも oMLX 実用限界を超える。
  → **decision レーンも oMLX 移行不可**(前述の「decision は oMLX へ」は取り消し)。
- **運用指針(二本立て)**: テキスト LLM レーンは Ollama 継続。oMLX は
  **gate(Ornith・小コンテキスト)・embedding(bge-m3)など小入力レーンのみ**先行移行が可能。
  価値は限定的(これらは元々 Ollama でも満足性能)。「全部 oMLX 統一」は
  **oMLX が大コンテキスト処理を実装/修正(上流)して初めて成立**。
- 環境復元済み: settings.json(max_context_window=32768)・model_settings(DFlash ON)はセッション前状態に復元。
  oMLX 0.6.2 退避コピーはディスク上から消えており、比較テストには再ダウンロードが必要。

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
  - 文字列 reasoning level はトップレベル `reasoning_effort` へ写像する。
  - DFlash engine の `response_format` strict schema は未対応で best-effort に落ちるため、
    formatless-first + client validation を使う。
- **DFlash エンジンは同時 1 つのみアクティブ**(現ビルド)。teacher は既存 lease で直列化し、
  Ornith は DFlash OFF + pinned として別 engine で常駐する。

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
  - `format` / structured_output → DFlash teacher は formatless-first + client validation
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
5. DFlash teacher の strict `response_format` は未対応。formatless-first + client validation を維持
6. Ollama 非依存サービス(Nemotron / reranker)は変更しない

## 6. 完了条件(Definition of Done)

- [x] `check_local_omlx_e2e.py` が green
- [x] adapter 単体テスト(写像含む)追加・green
- [x] 3 先生 + ornith の切り替え生成が Chronovisor 経由で通る(手動スモーク 1 回以上)
- [x] 埋め込み(分類/検索)が oMLX 経由で通る
- [x] recall.gate の idle・serial 実測 <= 1.5s(0.708s)
- [ ] background teacher 実行中の recall.gate <= 1.5s(現状 9.33s)
- [x] focused suite 213 passed
- [ ] 広い関連 suite の既存不一致 12件を別タスクで解消(880 passed / 12 failed)
- [x] 検証した意味のある単位で commit、明示依頼に基づき push
- [x] push 後、本番 cutover、runtime archive / role provenance、Ollama 退役を確認

## 7. 参考

- 前提プラン: `_handoff/2026-08-19_2258_model-stack-unify-omlx.md`(commit c172080 まで)
- oMLX 運用: `~/.omlx/`(settings.json に API キー、model_settings.json にモデル設定、models/ は HF キャッシュへの symlink)
- モデル登録ツール: `~/projects/sandbox/tools/register_omlx_ornith_bge3.py`(symlink + model_settings 注入の雛形)
- スキル: `dflash2-mlx-mac`(oMLX セットアップ・計測・メモリ運用の教訓)
- 不明点は推測せず、oMLX に 1 プローブ(curl)を打って実測してから実装を進めること
