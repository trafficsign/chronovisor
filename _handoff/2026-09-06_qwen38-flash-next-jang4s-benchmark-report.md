# Qwen3.8-Flash-Next JANG_4S benchmark

実施日時: 2026-09-06 JST

マシン: Mac Studio / Apple M4 Max (40-core GPU) / 128 GB unified memory

## 重要な訂正（2026-09-06 10:50 JST）

当初の「約18.36 GiBのメモリ削減」は、oMLXとvMLXが返す互換性のない内部指標を比較したもので、macOS全体のRAM削減量ではなかった。同一boot・no-model基準のOS実測では、旧oQ4eが約69.39 GiB、JANG_4Sが約69.01 GiBを消費し、差は約0.38 GiBに留まった。JANG_4S側では、内部指標に入らないclean mapped weights約11.7 GiBとretained cache約4.5 GiBが主な未計上分だった。

したがってメモリ優位の採用判定を撤回する。JANG_4Sは汎用品質が7.7 percentage points低下し、32K prefillも77.8%遅い一方、期待した実RAM削減が得られなかったため、本番は `Jundot/Qwen3.8-Flash-Next-oQ4e-mtp` / oMLXへロールバックした。以下は当初の比較結果を、誤判定の経緯が分かるよう履歴として残す。

## 当初の結論（撤回済み）

当初は `JANGQ-AI/Qwen3.8-Flash-Next-JANG_4S` を「メモリ優先の条件付き採用」と判定したが、この判定は上記のOS実測により撤回した。

- Chronovisor 実コーパスでは、構造妥当性 19/19、意味的効果 18/19 で現行と同率。中央値は 8.73 秒から 6.32 秒へ 27.7%短縮、全体も 14.1%短縮した。
- runtime内部のclean-load指標は現行約72.65 GiB、候補約54.29 GiBだったが、定義が異なるためRAM使用量として比較できなかった。同一bootのOS実測差は約0.38 GiBだった。
- 256-token decode は 62.44 から 71.85 tok/sへ 15.1%向上。4K は 12.7%高速、16K は同等。
- 犠牲は汎用推論と超長文。汎用 exact score は 24/26 から 22/26へ 7.7 percentage points低下し、32K prefill は 33.86 秒から 60.21 秒へ 77.8%悪化した。

速度差は参考値として残るが、実RAM削減がほぼなく品質・長文性能の低下だけが残るため、Chronovisor本番へ採用する根拠にはならない。

## 比較結果

| 指標 | 現行 oQ4e-mtp / oMLX 0.6.4 | JANG_4S / vMLX 1.6.53 | 候補の差 |
|---|---:|---:|---:|
| 汎用品質 | 24/26 (92.3%) | 22/26 (84.6%) | -7.7 pp |
| JSON schema | 19/19 (100%) | 19/19 (100%) | 同率 |
| Chronovisor semantic effect | 18/19 (94.7%) | 18/19 (94.7%) | 同率 |
| Chronovisor 中央値 | 8.73 s | 6.32 s | -27.7% |
| Chronovisor p95 | 27.06 s | 24.99 s | -7.6% |
| Chronovisor 全19件 | 243.32 s | 209.04 s | -14.1% |
| Chronovisor 最大 | 45.06 s | 48.81 s | +8.3% |
| 4K needle | 6.90 s / 100% | 6.02 s / 100% | -12.7% |
| 16K needle | 26.18 s / 100% | 26.27 s / 100% | +0.3% |
| 32K needle | 33.86 s / 100% | 60.21 s / 100% | +77.8% |
| 256-token decode | 62.44 tok/s | 71.85 tok/s | +15.1% |
| runtime内部のclean-load指標（非比較可能） | 約72.65 GiB | 約54.29 GiB | 見かけ上 -18.36 GiB |
| 同一boot・no-model基準のOS消費量 | 約69.39 GiB | 約69.01 GiB | 約 -0.38 GiB |
| 候補 post-run / peak | — | 55.79 / 58.89 GiB | — |

当初はruntimeごとの公開値に加えて候補の `footprint` だけを確認したが、両者を同じno-model基準で差分測定していなかった。再測定で候補はclean mapped weights約11.7 GiBを別に保持しており、runtime内部値だけではプロセス全体もOS全体も表せないと確定した。

## 品質差

汎用26件で候補だけが追加で落としたのは `truth`（期待 A、応答 B）と `code`（期待 30、応答 55）。`remainder` と `precedence` は両モデルとも失敗した。日本語の計算・順序・抽出3件、および6件のJSON課題は候補もすべて成功した。

Chronovisor で唯一不一致だった `lane-contract-v28:recall_improvement:3` は両モデルで同一だった。入力の top-level `status` が `candidate` なのに評価条件が `candidate_pass` を要求しているため、両方とも `needs_retry` / `hold` を選んだ。したがって今回検出された19レーン上の不一致は JANG_4S 固有の退行ではない。

## 実行条件と観測事項

- Chronovisor managed-v2 の登録済み9サービスを停止し、現行・候補を一モデルずつ、同一マシン・single sequence・greedy・最大114,688 prompt tokensで測定した。
- prefix/paged/block/disk/vision cache は候補側で無効化し、キャッシュ利益を含めなかった。
- 候補は vMLX native MTP D3 が実動。全走行の集計は prompt 225,160 tokens / 521.4 tok/s、generation 4,159 tokens / 51.2 tok/s。短い structured JSON ではMTP受理率が自由生成より下がり、少なくとも1回 vMLX のJSON correction retryを観測したが、最終schema成功率は100%だった。
- 16K以降は vMLX が Metal single-buffer limit 回避の chunked prefillへ移行し、32Kで現行より遅くなった。
- M4 Maxでは vMLXのMetal NA host最適化は無効。mixed 3-bit layoutのため affine MoE pair fusionも `refused_or_fallback` だった。この構成で71.85 tok/sを得たので、宣伝値ではなく本機実測を判定に使った。
- vMLXの専用 `qwen4_exp` VLM loaderなら起動するが、`--text-only` はgeneric mlx-lmへ落ちて `qwen4_exp` unsupportedとなる。テキスト用途でも full VLM pathが必要。
- tool auto-choice、vision、同時実行、長時間thermal soakは対象外。Chronovisor mainでほぼ使わないtool callは今回の採用判定に含めていない。

## 再現証跡

- `baseline-generic.json`: 現行26件 + 4K/16K/32K + decode
- `candidate-generic.json`: 候補26件 + 4K/16K/32K + decode
- `baseline-chronovisor.json`: 現行19レーン
- `candidate-chronovisor.json`: 候補19レーン
- `adoption-corpus.jsonl`: canonical 100 cases / 19 lanes
- `chronovisor_corpus_benchmark.py`: 既存評価関数をHTTP transportへつないだ最小adapter（`--self-check` あり）

- モデル: [JANGQ-AI/Qwen3.8-Flash-Next-JANG_4S](https://huggingface.co/JANGQ-AI/Qwen3.8-Flash-Next-JANG_4S)
- runtime: [jjang-ai/vmlx](https://github.com/jjang-ai/vmlx)
- upstream: [QwenLM/Qwen3.8-Flash-Next](https://github.com/QwenLM/Qwen3.8-Flash-Next)
