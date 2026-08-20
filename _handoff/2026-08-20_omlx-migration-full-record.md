# oMLX 移行 全記録(2026-08-20 版)

本ドキュメントは、Mac Studio M4 Max でのローカル LLM 運用の
**Ollama → oMLX(MLX ネイティブ)完全移行**プロジェクトの全検証・実装・
失敗・教訓の決定版アーカイブ。今後の再挑戦・引き継ぎの一次ソースとする。

- 並行資料:
  - `_handoff/2026-08-19_2258_model-stack-unify-omlx.md`(計画書)
  - `_handoff/2026-08-20_0610_omlx-migration-codex-handoff.md`(Codex 引き継ぎブリーフ、0c/0c2/0c3 節が本記録の要約)
  - `dflash2-mlx-mac` スキル(Hermes)
- 更新: 2026-08-20

---

## 1. 目的と背景

- **ゴール**: Chronovisor の全 5 モデル(qwen / gemma / muse / ornith / bge-m3)を
  Ollama から **oMLX(z-lab DFlash fork)+ dflash CLI** の MLX ネイティブ構成へ統一し、
  Ollama を退役させる。DFlash/DFlash2 投機デコードで 3 教師モデルを高速化する。
- **ユーザーの確定方針**: 「オーラ(維持)はない。」「全部 MLX で統一。」速度優先。
  ただし**物理的制約(大コンテキスト)が判明した場合は方針調整を許容**(2026-08-20 本記録)。
- **ハード**: Mac Studio M4 Max / 128GB。**クリーン計測が必須**(swap 9GB 超で全て 5-9 tok/s に落ちる)。

## 2. 環境

- **oMLX 0.6.3rc1**(z-lab DFlash2 fork):`/Applications/oMLX.app`。
  0.6.2 は退避コピーしていたが **2026-08-20 時点でディスク上から消滅**(比較テスト不可)。
- **dflash CLI**(uv tool、python 3.13): `~/.local/bin/dflash`。計測・生成ツール。
- **mlx_lm 0.31.3**(dflash 同梱): Muse 非対応(`muse_glimmer not supported`)。
- oMLX 設定: `~/.omlx/settings.json`(auth.api_key / sampling.max_context_window=32768)、
  `~/.omlx/model_settings.json`(モデル別 DFlash 設定)、`~/.omlx/models/`(HF symlink)。
- HF キャッシュ: mlx-community の各 4bit + z-lab / meta-models / ornith-ai / incoai のドラフター類。

## 3. モデルスタックと採用標準

| 役割 | モデル(Ollama) | oMLX 移行先 | 備考 |
|---|---|---|---|
| ingest / primary / audit / planner / anchor 等 | `qwen3.8:27b-axq4` | `Qwen3.8-27B-4bit` + **DFlash2**(incoai/Qwen3.8-27B-DFlash2) | 速度採用 Q4 |
| tie_break / scorer | `gemma4:26b-optiq4` | `gemma-4-26b-a4b-it-4bit` + DFlash v1(z-lab drafter) | |
| challenger | `muse-glimmer:30b-q4k-dynamic` | `Muse-Glimmer-30B-4bit` + **DFlash2**(z-lab/Muse-Glimmer-30B-DFlash2) | rc1 でのみ実動 |
| recall.gate / judge | `ornith:9b-q4_K_M` | `Ornith-1.5-9B-MLX-4bit` | ウォーム 0.2s、1.5 は誤 YES が正しく NO |
| embeddings | `bge-m3:latest` | `bge-m3-mlx-fp16` | /v1/embeddings 1024d・正規化済み |
| — | `gpt-oss:20b` | **移植対象外** | 実生成ゼロ(ログで裏取り) |

- AXQ(5.2bpw)は Ollama blob から再構築 → `.weight.scale/.bias`→`.scales/.biases` リネームで
  MLX ロード可(17.3GB・バイト同一)。`+DFlash2 = 28.7 tok/s`(2.03x)だが stock カーネル遅く
  現行 Ollama 26.5 と同等 → **精度重視時の保険として sandbox に保存**(Q4+DFlash2 が本命)。

## 4. 検証済み実測値(M4 Max・クリーン状態)

| 構成 | 実測 |
|---|---|
| dflash CLI: Qwen3.8-27B-4bit + DFlash2 | **56.3 tok/s(1.93x)** / AR 14.2 |
| dflash CLI: Gemma4-26B + DFlash v1 | **54.4 tok/s(1.76x)** / 受容長 7.24 |
| oMLX 0.6.2: Qwen / Gemma / Muse | 43.8 / 112.4 / 16.4 |
| oMLX **0.6.3rc1**(クリーン): Qwen / Gemma / Muse | **45.8 / 140.8 / 24.1**(Muse+DFlash2、v1 20.2・0.6.2 16.4・Ollama 20.9 を全て超え) |
| Ornith-1.5-9B-MLX-4bit ゲート | ウォーム **0.20s**(予算 1.5s 内)、判定は 1.0 より正確 |
| llama.cpp DFlash2 PR(M5 Pro) | AR 10.42 / DFlash2 19.31(1.85x)。MLX の絶対速度は約 3 倍速い |
| 現行 Ollama(参考) | qwen 26.5 / gemma4 4.9 / muse-glimmer 20.9 tok/s |

**oMLX 運用上の実測制約**:
- DFlash エンジンは**同時 1 つ**(pinned は他をブロック、非 pinned は自動スワップ)。
  `is_pinned: true` の DFlash モデル使用は慎重に。
- 未知パラメータ(keep_alive / num_ctx / トップレベル think)は黙殺 → 写像ミスが無音になる罠。
- 思考制御は `chat_template_kwargs.enable_thinking` でのみ効く(gate の think=false に必須)。
- Muse の空 content は思考モデル仕様(reasoning_content のみ返す)で実害なし。

## 5. 実装物(コード)とコミット

- `src/chronovisor/core/omlx_adapter.py`(280 行): **OMLXAdapter**。
  provider="omlx" / RouteLocation.LOCAL。
  - generate() → POST /v1/chat/completions(Content-Type: application/json 必須)
  - embed() → POST /v1/embeddings
  - think → `chat_template_kwargs:{"enable_thinking":bool}`
  - num_ctx/keep_alive は送らない(黙殺されるため honest に drop)
  - エラー allowlist(http_401/429/5xx/invalid_request)へ分類
  - reasoning-only フォールバック、`unload()` は no-op(False を正直報告)
- `src/chronovisor/core/llm_config.py`: kind "omlx" 分岐を 2 箇所(`_provider()` / `build_llm_runtime`)。
- `tests/test_omlx_adapter.py`(300 行): 13 件新規(MockTransport)。
- `scripts/check_local_omlx_e2e.py`(167 行): 実サーバー相手 e2e(6 役割生成+gate+埋め込み)。
- テスト実績: `pytest` **174 件全パス・回帰ゼロ**、e2e PASS。
- コミット(chronovisor/main、全て push 済み):
  - `42f9318` 計画・`8fcd3e3` モデル棚卸・`15ca044` gpt-oss 除外・`c172080` ornith/bge 検証
  - `f6c2b08` Codex ブリーフ・`3952988` rc1 検証(Muse DFlash2)・`5af6bc8` decision 思考レベル低
  - `8570d72` **OMLXAdapter + tests + e2e**・`e6a6e0d` ブリーフ REPO-IMPLEMENTED
  - `a88ea0a` カットオーバー1 回目記録・`7bcedd6` 2 回目 真因記録・`3907c80` 0c3 境界実測

## 6. 本番カットオーバー試行(1 回目)

- `[llm]` 領域のみ書き換え(cutover_omlx.py: 41 role flip + REMAP、スコープ検証 OK)。
  バックアップ→`install -m 600` 原子入れ替え→launchd 再起動(converge/ingest-drain/library-evidence/soak/semantic)。
- **結果: 未成立 → ロールバック**。原因の第一報は「旧バイナリ + kind omlx の不整合」
  (エージェントは毎回 Git remote からインストール、push 前に旧コードだったため)→ push 解決。
- それでも ingest が「待ち」のまま → 一時 config を Ollama に復元。

## 7. 学校停滞の調査と真因(2 回目・ここが核心)

**症状**: 06:43 以降、ingest がバッチを 1 件も消化せず「待ち」。ロールバック後も同じ。

調査で判明した構造:
1. **リース自己保持(実は副次)**: ingest-drain 子が
   `ingest-orchestrator.lock` + `ollama-resource.lock` を握ったまま待機。
   手動ドレインは「another ingest process holds the cross-process lease」でブロック。
   → 朝のクリーンベンチ(SIGSTOP + ollama bootout)が 06:43 の `ConnectError` ブロックを生んだ。
2. **フラグメント regex 説は否定**: `get_pending_raw_files()`=1838 件 0.89s /
   `_prepare_pending_raw_units()`=0.14s(隔離実行)。sample で regex 連打が見えたのは
   生成処理中であって犯人ではなかった。
3. **真因: oMLX 0.6.3rc1 が大入力で応答しない**。per-unit バッチの子は
   `:8000`(oMLX)へ ESTABLISHED のまま 0%CPU で待機。直接テストで再現:
   - 本番経路 `runtime_generate`(ingest role・短入力)は **0.7s 成功**(API キー無しでも通り、
     認証・アダプタ・config・route 解決・provenance は**全て正常**と立証)。
   - **120KB(≈40K token)入力 → 90〜240s 超無応答**。
   - 各ユニットは config の read_timeout_ms=660000(11 分)を消化 → `backend_error`
     → リース保持のまま全ユニット繰り返し = 実質永久停滞(provider 無関係に見える)。
- **誤診の訂正(正直な記録)**: 1 回目はこれを「リース/residency の Ollama 結合ギャップ」と
  誤診した。真因は oMLX の大入力処理限界。ブリーフ 0c2/0c3 で訂正済み。

## 8. 大入力境界の実測(決定版・2026-08-20)

oMLX 0.6.3rc1・Qwen3.8-27B-4bit・max_tokens=4 で境界測定(M4 Max):

| 入力 | 結果 |
|---|---|
| ≤10K chars(≈3.3K tok) | ✅ 4〜16s で応答 |
| 30K chars(≈10-18K tok) | ❌ 90s 超無応答 |
| 120K chars(≈40K tok) | ❌ 180〜240s 超無応答 |

- **DFlash ON/OFF で挙動不変**(問題は prefill/サーバー側。decode しか加速しない DFlash は無関係)。
- **`sampling.max_context_window` 32768→131072 に上げても 40K tok は 180s+ 無応答**
  = コンテキスト窓設定では解決しない(サーバー処理の実質限界/潜在バグ)。

**config 実要件との突合**(これが決定的):
- `[ingest]`: num_ctx=32768 / max_num_ctx=262144 / read_timeout_ms=660000
- `[decision_router]`: num_ctx=114688 / min_num_ctx=16384 / read_timeout_ms=660000
- → **ingest も decision も oMLX の実用限界(~10K tok)を超える。両レーンとも oMLX 移行不可。**

## 8b. 実 unit での本番パス検証(決定打・2026-08-20)

- サーバーをクリーン起動 → config を一時 omlx に切替 → 実 pending unit 1 件を
  `run_pending_ingest(force=True, max_units=1)` で実行(220s バウンド、実行後に config 復元)。
- **結果: `triggered=True, proc=0, failed=1, failure=ingest.runtime_backend_error`(221.0s)**
- OMLXAdapter.generate をラップして実ペイロード計測:
  - **payload: Qwen3.8-27B-4bit, messages=2, chars=36,245 ≈ 12K トークン**
  - (生ファイルは ~2KB だが、システムメッセージ+構造化コンテキストで **12K トークンに膨張**)
- **oMLX は Qwen をロードして(server RSS 18.6GB)処理を開始したが、220 秒で応答なし**。
- → 「窓が小さい」でも「完全ハング」でもなく、**oMLX の prefill/エンジンは
  ~12K トークンという中規模入力でも実用不能な速度**。既知上流 Issue(#2179 prefill guard /
  #2624 engine wedge 家族)と整合。
- **決定**: ingest・decision とも oMLX 0.6.3rc1 では**不可(実データで確定)**。

### ユーザー提示ダッシュボードとの整合(重要な解釈)
- ダッシュボードの「CONTEXT WINDOW: 32K/65K/98K/131K」「required 35K → selected 65K」は
  **Chronovisor の構造化セッションが現行 Ollama で使うコンテキスト実績**であり、
  「ワークロードが 35K〜131K トークンを要求する」ことの証拠。**oMLX の達成ではない**。
- 「131K が光ってた」= Ollama バックエンドで 131K バケットを実際に選択した実績。
  逆に言うと、**このワークロード規模(35K 起点)は oMLX の実用限界(~12K)の約 3 倍**。

## 9. 結論: 運用方針(Ollama 二本立て)

- **Ollama**: テキスト LLM 全レーン(ingest / decision / challenger)— 大コンテキスト必須のため継続。
- **oMLX**: gate(Ornith・小入力)・embedding(bge-m3)など**小入力レーンのみ**先行移行可能。
  実益は限定的(元々 Ollama でも満足性能)。
- 「**全部 MLX 統一**」は方針として維持。**oMLX が大コンテキスト処理を実装/修正した後に再挑戦。**
  (上流未報告: ユーザー判断で z-lab へのレポートは出さない。)

## 10. 上流 Issue マッピング(jundot/omlx・既知バグクラス)

今回の症状と一致する既存 Issue(調査日 2026-08-20、state は未確認):

| # | 日付 | 内容 |
|---|---|---|
| #2179 | 7/10 | BLOCKER: prefill token guard 満杯でクリア不能 → 後続全要求停止(≒本症状) |
| #2624 | 8/13 | engine が 6h 停止・全スレッド park で dispatch しない |
| #2701 | 8/16 | scheduler がエンジン auto-load 中のリクエストで wedges |
| #2235 | 7/14 | decode 途中キャンセルで engine 恒久 wedge |
| #2909 | 8/20 | 30K ctx で MTP drafter 再プリム ≈7.5s(大コンテキスト遅延) |
| #2689 | 8/16 | Qwen3.8-27B oQ8e MTP、16K ctx で decode 崩落(4-5.5 tok/s) |
| #2641 | 8/13 | Glimmer + dflash が極端に遅い |
| #2632 | 8/13 | DFlash: リクエスト競合で `runtime_context is required` |

→ **次版リリースノートで #2179 / #2624 / #2701 / #2909 のクローズ/修正を確認できたら
「大コンテキスト対応が入った」の判定に使う。**

## 11. 最終状態(2026-08-20)

- **学校**: config = Ollama(検証済みの既知状態)。14 エージェント生存。
  ingest status=ready、pending=約 1788(消化中・低速)。
- **oMLX**: rc1 サーバー起動中・23 モデル登録。設定はセッション前へ復元
  (model_settings: Qwen DFlash ON、settings: max_context_window=32768)。
  ※今後の利用前は再起動を推奨(大入力で一度詰まるため)。
- 残存ファイル:
  - バックアップ: `~/.chronovisor/config.toml.bak-ollama-before-cutover2-20260820`、
    `config.toml.bak-omlx-cutover-20260820`
  - ステージ: `/tmp/chronovisor-cfg-omlx.toml`(omlx config 完成品・検証済み)
  - スクリプト: `/tmp/cutover_omlx.py`、`/tmp/rc1_bench.sh`
  - tools: `~/projects/sandbox/tools/{reconstruct_ollama_mlx.py, rename_axquant_to_mlx.py,
    ocr_vision.swift, omlx_3model_bench.sh, register_omlx_ornith_bge3.py}`
  - 境界テスト用プローブは本記録 §14 に掲載(ツール化は保留)

## 12. 今後の動き方

1. **oMLX 更新の見張り**: リリースが出たら §14 の境界プローブを流し、
   §10 の Issue クローズと併せて「大コンテキスト対応」を判定。
2. **二本立ての実装(任意)**: gate / embedding の oMLX 先行移行(小規模)。
3. **全部 MLX 統一**: oMLX の大コンテキスト対応後の再挑戦(手順はブリーフ 0〜7 節)。

## 13. 教訓・運用ルール

1. **クリーン計測必須**: swap >9GB だと全数値が無意味。Ollama 47GB の同居は不可。
   計測は「Ollama 停止(bootout)→計測→load 復帰」で。SIGSTOP は学校のロックを壊し得るので
   **launchd 一括停止方式に変更**。ロック保持者は lsof で確認。
2. **oMLX の大入力**は「ハング」ではなく「実用時間外」— バウンド 15s だと誤判する。
   大小2点(小=通る基準・大=目標)で測れ。
3. **launchd エージェントは毎回 Git remote からインストール** → ローカル commit は push まで効かない。
4. 症状の「待ち」は provider/原因を見誤りやすい: config ロールバックではサーバー状態は戻らない
   (oMLX は再起動しない限り詰まったまま)。
5. 既知バグは上流に既にあることが多い → 深追いの前に Issue 検索を。

## 14. 付録: 境界テストプローブ(再利用用)

```bash
# oMLX が新しいバージョンになったら、これを流して大小2点で判定する
KEY=$(python3 -c "import json,os;print(json.load(open(os.path.expanduser('~/.omlx/settings.json')))['auth']['api_key'])")
python3 - "$KEY" <<'PY'
import json,sys,urllib.request,time
key, models = sys.argv[1], {m['id'] for m in json.load(urllib.request.urlopen(
    urllib.request.Request('http://127.0.0.1:8000/v1/models',headers={'x-api-key':key})))['data']}
print('models:', len(models))
for label, n in [('SMALL(3K tok)','転写データです。'*3000), ('BIG(40K tok)','転写データです。'*24000)]:
    p=json.dumps({'model':'Qwen3.8-27B-4bit','messages':[{'role':'user','content':n}],'max_tokens':4}).encode()
    req=urllib.request.Request('http://127.0.0.1:8000/v1/chat/completions',data=p,
        headers={'Content-Type':'application/json','x-api-key':key})
    t=time.monotonic()
    try:
        urllib.request.urlopen(req,timeout=240); print(f'{label}: OK {(time.monotonic()-t):.1f}s')
    except Exception as e:
        print(f'{label}: FAIL {type(e).__name__}')
        subprocess.run(['pkill','-9','-f','omlx-server'])
PY
```
判定: SMALL は十数秒で OK、BIG も OK なら大コンテキスト対応が入った。
BIG だけ 240s 超なら従来通り(二本立て維持)。
