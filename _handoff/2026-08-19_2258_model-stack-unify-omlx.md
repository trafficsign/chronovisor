# 計画: ローカルLLMスタックの oMLX 統一(3モデル)

状態: **PLAN(方向性確定 / 実装詳細は未着手)**
作成: 2026-08-19 / Hermes 相談セッション(実測済み)
実装者(想定): ①退役・常駐化 = Hermes / ②Chronovisor 付け替え = Codex

## 背景

- 現行は Ollama(launchd 常駐)が 3 モデルを提供。AXQ/mixed-bit の MLX runner バグ(PR 側)、
  メモリ圧迫(47GB + swap 9GB)、速度頭打ちが課題。
- **方針決定(2026-08-19): マルチランナー運用をやめ、3モデル全部を MLX ネイティブの
  oMLX に統一する。「1個だけ Ollama に残す」運用はしない。**
- oMLX は z-lab DFlash/DΔFlash fork。OpenAI 互換サーバー(localhost:8000)内蔵。
  Qwen=DFlash2 / Gemma・Muse=DFlash v1 対応。3 モデル登録・動作済み。

## 実測(2026-08-19、クリーン状態 = Ollama 停止必須)

| モデル | oMLX + DFlash | dflash CLI 参考 | 現行 Ollama |
| --- | --- | --- | --- |
| Qwen3.8-27B Q4 + DFlash2 | **43.8 t/s** | 56.3 | 26.1 |
| Gemma4-26B Q4 + DFlash v1 | **112.4 t/s** | 54.4 | 4.9 |
| Muse-Glimmer-30B Q4 + DFlash v1 | **16.4 t/s**(動作) | —(CLI不可) | 20.9 |

注: swap 9GB 超で何でも 5-9 t/s に落ちるため、計測・常駐は Ollama 退役が前提。

## ターゲット構成

```
oMLX server (localhost:8000, OpenAI互換, x-api-key)
  ├─ Qwen3.8-27B  Q4 + DFlash2  … 43.8 t/s (先生・高速)
  ├─ Gemma4-26B   Q4 + DFlash v1 …112.4 t/s (先生・爆速)
  └─ Muse-30B     Q4 + DFlash v1 … 16.4 t/s (先生・高級)
        ▲ /v1/chat/completions
   Chronovisor (ollama_lease.py → oMLX クライアントへ置換)
```

- oMLX サーバー = 唯一の推論サーバー(Ollama の launchd「席」を引き継ぐ)
- Chronovisor は Ollama ではなく oMLX の OpenAI 互換 API を叩く
- Ollama = launchd 無効化・完全退役(99GB blobs も道連れに解放)

## 対象モデル全量(実運用 config = ~/.chronovisor/config.toml に基づく)

画像で「3モデル」に見えても、Chronovisor 本番は **計 6 モデルが Ollama 依存**。
統一するには以下全部の移行が必要:

| モデル | 役割(実 config) | 現バックエンド | oMLX 移行可否 |
| --- | --- | --- | --- |
| qwen3.8:27b-axq4 | ingest / decision.primary / audit / research.planner / recall escalation | Ollama | ✅ 実施済み(Qwen3.8-27B-4bit + DFlash2, 43.8) |
| gemma4:26b-optiq4 | decision.tie_break / research.tie_break | Ollama | ✅ 実施済み(gemma-4-26b-a4b-it-4bit + DFlash, 112.4) |
| muse-glimmer:30b-q4k-dynamic | decision.challenger | Ollama | ✅ 実施済み(Muse-30B-4bit + DFlash v1, 16.4) |
| **ornith:9b-q4_K_M** | recall.gate(1.5s 判定)/ recall.processor judge / 低遅延クリティカル | Ollama | ✅ **有望**: Ornith-1.0-9B(Qwen3.5 ベース)の MLX quant あり + oMLX は Qwen3.5-9B を DFlash drafter 付きで対応。要計測 |
| **gpt-oss:20b** | research.challenge(明示モードのみ) | Ollama | ⚠️ MLX 4bit quant あり(majentik/gpt-oss-20b-TurboQuant-MLX-4bit) だが oMLX DFlash レジストリ外。素朴ロード要検証、または支持モデルへの振替 |
| **bge-m3:latest** | [embedding] 埋め込み(Ollama API 経由) | Ollama | ⚠️ mlx-community/bge-m3-mlx-fp16 + oMLX の /v1/embeddings + mlx_embeddings 同梱までは確認。BERT 埋め込みの実際の登録・提供可否は未検証 |

**Ollama 非依存 = 統一対象外(独立して継続):**
- nvidia/Nemotron-3-Embed-1B-BF16 … [search.embedding] semantic-service(Unix socket / MPS)
- BAAI/bge-reranker-v2-m3 … [search.reranker] プロセス内 transformers(MPS)

方針補足:
- ornith は「3モデル以外で唯一の本命移行ターゲット」。Qwen3.5-9B 対応があるので DFlash 高速化も狙える。
- bge-m3 が oMLX で提供できない場合の代替:①独立 MLX 埋め込みサービス(mlx_embeddings)化、②Nemotron 側へ一本化(bge-m3 廃止、次元 1024→2048・再埋め込みの挙動変化あり)。
- 6 モデルすべて oMLX(または独立サービス)に載れば Ollama 退役が完成する。

## 作業フェーズ(概略)

1. **oMLX 常駐化**: launchd エージェントで `~/.omlx/bin/omlx start`(KeepAlive、自動起動)。
   GUI(oMLX.app)はモデル管理ツールとして併用(ポート 8000 の二重起動を避ける)。
2. **Ollama 退役**: launchd エージェント無効化(bootout + plist リネーム)→ swap 根絶。
3. **Chronovisor 付け替え(本格作業)**: `ollama_lease.py` を oMLX API クライアントに置換。
   リース = プロセス spawn/kill ではなく「モデル枠予約 + 生存チェック」に簡略化。
   3 モデル名を oMLX モデル ID にマップ。[Codex 引き継ぎ想定]
4. **検証**: 3 先生すべて oMLX 経由で生成できること、swap 監視(>4GB で警告)。

## 未決 / 注意

- Muse 16.4 t/s(現 Ollama 20.9 より遅い)は統一の代償として許容。底上げ余地:
  `mlx-community/Muse-Glimmer-30B-OptiQ-4bit`(混合 bit、oMLX 内蔵 mlx_lm 0.31.3 で理論上ロード可)への差し替え比較。
- Muse DFlash2 は oMLX 未対応(imcoai/z-lab の DFlash2 ドラフターは実在するが、oMLX の
  Muse target 検証パスは DFlash v1 止まり。llama.cpp PR #27342 は Qwen のみ実測)。
- 移行対象の残り 3 モデル(ornith / gpt-oss / bge-m3)は oMLX への実装確認待ち:
  - ornith:計測必須(recall.gate は 1.5s タイムアウトの低遅延パス)
  - gpt-oss:oMLX DFlash レジストリ外。素朴ロード試行 or 役割振替の判断
  - bge-m3:oMLX の BERT 埋め込み提供可否を実証する必要あり
- Chronovisor 側は Ollama 固有設定が多い(keep_alive / num_ctx / coordinate_ollama /
  adaptive_residency / ollama_lease)ため、oMLX へのパラメータ写像の設計が本格作業。
- oMLX の 3 モデル同時リクエストのキューイング挙動、全モデル常駐 vs 遅延ロードは要検証。
- 付随ツール: `~/projects/sandbox/tools/{omlx_3model_bench.sh, reconstruct_ollama_mlx.py,
  rename_axquant_to_mlx.py, ocr_vision.swift}`、スキル `dflash2-mlx-mac`。

## 関連

- 本スレッドの実測・協議(Hermes、2026-08-19)
- r/LocalLLaMA スレッド「DFlash 2 available for Qwen 3.8 27B and Muse Glimmer」
- llama.cpp PR #27342(spec: add DFlash2 support / Apple M5 Pro 実測あり・未マージ)
