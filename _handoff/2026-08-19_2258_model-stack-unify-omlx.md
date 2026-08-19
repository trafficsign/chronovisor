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
- oMLX の 3 モデル同時リクエストのキューイング挙動、全モデル常駐 vs 遅延ロードは要検証。
- 付随ツール: `~/projects/sandbox/tools/{omlx_3model_bench.sh, reconstruct_ollama_mlx.py,
  rename_axquant_to_mlx.py, ocr_vision.swift}`、スキル `dflash2-mlx-mac`。

## 関連

- 本スレッドの実測・協議(Hermes、2026-08-19)
- r/LocalLLaMA スレッド「DFlash 2 available for Qwen 3.8 27B and Muse Glimmer」
- llama.cpp PR #27342(spec: add DFlash2 support / Apple M5 Pro 実測あり・未マージ)
