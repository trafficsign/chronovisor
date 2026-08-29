# Qwen3.8-Flash-Next / MLX 量子化調査

調査基準日: 2026-08-29 (公開 API の日時は UTC)。対象は公開一次情報と公開モデル・リポジトリのメタデータだけである。ローカルのダウンロード状態、既存の会話、モデル起動結果は根拠にしていない。調査中の重みダウンロード・モデル起動は行っていない。

## 結論（選定前の仮説）

Qwen の公式モデルは `Qwen/Qwen3.8-Flash-Next` で、125B（6B activated）、51B の N-gram embedding、4B MTP を含む MoE である。[Qwen 公式 GitHub](https://github.com/QwenLM/Qwen3.8-Flash-Next) と[公式 HF model card](https://huggingface.co/Qwen/Qwen3.8-Flash-Next) が一次情報である。

oMLX の調査基準版は `v0.6.3`（2026-08-27 公開、commit `85708e4b9a585df42241c826b6be2b4dba018406`）。この版は `qwen4_exp`、Qwen3.8 の text/image、structured tool call、continuous batching、prefix-cache state、oQ 変換、Lightning MTP を正式に追加した。[v0.6.3 release](https://github.com/jundot/omlx/releases/tag/v0.6.3) の記載上、最初に検証する候補は oMLX が生成した `Jundot/Qwen3.8-Flash-Next-oQ4e-mtp` である。ただし「最速・最高品質」とはまだ結論できない。M4 Max 128GB の同一条件測定と、後述の open issue を通過する必要がある。

特に、(a) BF16 から直接変換した一部の community checkpoint が現行 oMLX で決定的な garbage を生成するという [#3181](https://github.com/jundot/omlx/issues/3181)、(b) Hugging Face cache の symlink で PLE shard の remap/drop に失敗するという [#3239](https://github.com/jundot/omlx/issues/3239) は、いずれも 2026-08-29 時点で open の報告である。したがってカード記載の速度・品質を採用条件にせず、実ファイル配置、生成 smoke、品質 gate を先に置く。

## 公式モデルと公式 BF16 基準値

| 項目 | 公式情報 |
|---|---|
| 公開 | Qwen 公式 repo README は 2026-08-26 公開と記載。公式 HF repo は [`Qwen/Qwen3.8-Flash-Next`](https://huggingface.co/Qwen/Qwen3.8-Flash-Next)。 |
| 構成 | 125B total、6B activated、51B N-gram embedding、4B MTP、48 layers、512 experts（10 routed + 1 shared）、native context 262,144（1M まで拡張可能）、MTP 1 layer。出典: [公式 card](https://huggingface.co/Qwen/Qwen3.8-Flash-Next)。 |
| 公式ベンチマーク | BF16/thinking 設定で Qwen3.8-27B と比較。DeepSWE **58.7 vs 42.2**、SWE-bench Pro **62.5 vs 61.7**、SWE-bench Multilingual **81.0 vs 73.8**、NL2Repo **48.1 vs 42.3**、CoWorkBench **73.9 vs 70.7**、JobBench **55.7 vs 33.4**、Toolathlon **73.5 vs 67.1**、IFBench **81.3 vs 79.5**、GPQA **91.7 vs 89.2**、HLE **35.9 vs 30.8**、LiveCodeBench v6 **91.9 vs 90.3**。出典: [公式 card](https://huggingface.co/Qwen/Qwen3.8-Flash-Next)。 |
| 解釈上の注意 | 上記は公式 BF16 の基準値であり、以下の MLX/oQ 量子化の測定値ではない。量子化の品質保持を直接証明しない。 |

## oMLX 0.6.3 の互換性と前提

- リリースは [oMLX v0.6.3](https://github.com/jundot/omlx/releases/tag/v0.6.3)、公開時刻は `2026-08-27T17:04:00Z`、対象 commit は `85708e4b9a585df42241c826b6be2b4dba018406`。Qwen3.8 の実装・oQ PLE・Lightning MTP が入った版として固定する。
- [v0.6.3 `pyproject.toml`](https://github.com/jundot/omlx/blob/v0.6.3/pyproject.toml) の実行条件は Python `>=3.11,<3.14`、macOS 15+ / Apple Silicon、`mlx==0.32.0`、`transformers>=5.12.1,<5.13`、`huggingface-hub>=1.19.0`。関連固定値は `mlx-lm` commit `ab1806e8f5d6aa035973af194a1b9198ab4754dc`、`mlx-vlm` commit `78b96eb5462141447b9a6b4943ef553891da56dd`、`dflash-mlx` commit `c55324c86540c369f6818a0f47eae544d405475b`。
- [v0.6.3 の model loading 実装](https://github.com/jundot/omlx/blob/v0.6.3/omlx/utils/model_loading.py) は `model_type=="qwen4_exp"` を検出し、実際の index に `mtp.*` tensor があるときだけ MTP を有効化する。MTP 無し checkpoint に MTP を要求しないこと、MTP off は重みをロードしても decode は行わないことを smoke test で確認する。
- [PR #3174](https://github.com/jundot/omlx/pull/3174) は QSA/recurrent cache state、adaptive Lightning MTP、group32 の oQ PLE shard、oQ2/oQ3/oQ4/oQ5/oQ6/oQ8 の PLE mmap を実装し、Qwen3.8 API/cache/oQ のテストを追加した。

### 公式 oMLX の速度資料（別ハードウェア）

[v0.6.3 release](https://github.com/jundot/omlx/releases/tag/v0.6.3) と [PR #3174](https://github.com/jundot/omlx/pull/3174) の M3 Ultra 512GB、`Qwen3.8-Flash-Next-oQ4e-mtp`、fresh process、SSD PLE mmap、prefix cache 無効、同一 prompt/seed、128 output token、temperature 1 / top-p 1、adaptive max depth 3 の結果は次の通りである（PP=prompt processing、TG=token generation、単位 tok/s）。

| context | MTP | PP | TG | MLX peak | process peak | acceptance |
|---:|:---:|---:|---:|---:|---:|---:|
| 4K | off | 1062.5 | 22.2 | 70.88 GiB | 73.39 GiB | — |
| 4K | on | 1035.9 | 58.1 | 72.43 GiB | 75.00 GiB | 96.8% |
| 16K | off | 964.6 | 21.9 | 72.32 GiB | 75.55 GiB | — |
| 16K | on | 934.1 | 53.7 | 73.95 GiB | 77.07 GiB | 97.9% |
| 32K | off | 840.8 | 21.0 | 74.44 GiB | 79.08 GiB | — |
| 32K | on | 809.4 | 49.0 | 76.18 GiB | 80.77 GiB | 97.8% |

これは M4 Max 128GB の結果でも、他の量子化との比較でもない。従って、MTP の仮説（TG 2.33–2.62x、PP 2.5–3.7%低下、MLX peak +1.6–1.8 GiB）を得る資料としてのみ使う。

## 公開 MLX/oQ 候補（revision 固定）

サイズは HF tree の `safetensors` 合計（GiB = bytes / 2^30）。`createdAt` と revision SHA は [HF model API](https://huggingface.co/docs/huggingface_hub/guides/integrations) の各 repo 応答、tensor 数・サイズは各 repo の revision 固定 tree から取得したメタデータである。いずれも Qwen 公式ではなく community artifact なので、カードの速度・品質主張は仮説扱いとする。

| repo ID（固定 revision / 公開日時） | 容量・tensor 数 | quantization / MTP（raw config） | 扱い |
|---|---:|---|---|
| [`Jundot/Qwen3.8-Flash-Next-oQ4e-mtp`](https://huggingface.co/Jundot/Qwen3.8-Flash-Next-oQ4e-mtp/tree/2615fc0e976e65c2f3b55daca3a948f1cdc5b9f8) / `2615fc0e976e65c2f3b55daca3a948f1cdc5b9f8`, 2026-08-27 08:56:41Z | 21 safetensors、106,294,664,646 B = **99.00 GiB**（repo total 約99.02 GiB） | [`config.json`](https://huggingface.co/Jundot/Qwen3.8-Flash-Next-oQ4e-mtp/raw/2615fc0e976e65c2f3b55daca3a948f1cdc5b9f8/config.json): `qwen4_exp`、base 4-bit、group64、affine; overrides 4-bit/128、5-bit/181、6-bit/99、8-bit/223; `mtp_num_hidden_layers=1`、MTP hybrid/full-attention 1 layer、index に `mtp.*` 76 keys。README は oMLX v0.6.3rc3 の oQ 変換と `oqe_code_multilingual` 1,024 sample calibration metadata を記載。 | **第一候補（検証優先）**。oMLX 公式 PR/release の benchmark 名と一致し、oQ 経路の greedy parity が報告される。ただし本番受入は smoke/quality gate 後。 |
| [`Vontra/Qwen3.8-Flash-Next-MLX-oQ3-MTP`](https://huggingface.co/Vontra/Qwen3.8-Flash-Next-MLX-oQ3-MTP/tree/dbcf52e7921f4ed441f7bb46ff3b440d25c19744) / `dbcf52e7921f4ed441f7bb46ff3b440d25c19744`, 2026-08-26 22:02:12Z | 19、92,505,200,266 B = **86.15 GiB**（repo total 約86.17 GiB） | [`config.json`](https://huggingface.co/Vontra/Qwen3.8-Flash-Next-MLX-oQ3-MTP/raw/dbcf52e7921f4ed441f7bb46ff3b440d25c19744/config.json): 3-bit、group32、affine; overrides 4-bit/314、5-bit/82、6-bit/127、8-bit/223; MTP 1 layer、`mtp.*` 76 keys、`qwen4_exp`。 | 3-bit mixed 候補。カード（[README](https://huggingface.co/Vontra/Qwen3.8-Flash-Next-MLX-oQ3-MTP/blob/dbcf52e7921f4ed441f7bb46ff3b440d25c19744/README.md)）の M3 Studio 主張は MTP off 26.5352 / on 29.0822 tok/s、acceptance 68.83%だが未検証。#3239 の symlink 問題に注意。 |
| [`Vontra/Qwen3.8-Flash-Next-MLX-3bit-MTP`](https://huggingface.co/Vontra/Qwen3.8-Flash-Next-MLX-3bit-MTP/tree/051d36f1a21aa85403ca52cba889dcea6c94c498) / `051d36f1a21aa85403ca52cba889dcea6c94c498`, 2026-08-26 22:02:03Z | 18、90,774,153,499 B = **84.54 GiB**（repo total 約84.56 GiB） | [`config.json`](https://huggingface.co/Vontra/Qwen3.8-Flash-Next-MLX-3bit-MTP/raw/051d36f1a21aa85403ca52cba889dcea6c94c498/config.json): uniform 3-bit、group32、affine、overrides 無し; MTP 1 layer、`mtp.*` 76 keys。 | uniform 3-bit 候補。カード（[README](https://huggingface.co/Vontra/Qwen3.8-Flash-Next-MLX-3bit-MTP/blob/051d36f1a21aa85403ca52cba889dcea6c94c498/README.md)）の off 28.0408 / on 30.5895 tok/s、acceptance 65.4% は未検証。 |
| [`Vontra/Qwen3.8-Flash-Next-MLX-oQ4-MTP`](https://huggingface.co/Vontra/Qwen3.8-Flash-Next-MLX-oQ4-MTP/tree/43a82b3f0ff64fa417fd09ca046580f08d19b0d6) / `43a82b3f0ff64fa417fd09ca046580f08d19b0d6`, 2026-08-26 16:23:54Z | 22、113,325,274,612 B = **105.54 GiB**（repo total 約105.56 GiB） | [`config.json`](https://huggingface.co/Vontra/Qwen3.8-Flash-Next-MLX-oQ4-MTP/raw/43a82b3f0ff64fa417fd09ca046580f08d19b0d6/config.json): mixed oQ 4-bit、group32、affine; overrides 5-bit/36、8-bit/196; MTP 1 layer、76 keys。 | mixed 4-bit候補。カード（[README](https://huggingface.co/Vontra/Qwen3.8-Flash-Next-MLX-oQ4-MTP/blob/43a82b3f0ff64fa417fd09ca046580f08d19b0d6/README.md)）は direct BF16 conversion と M3 Studio の 17.4–39.7 tok/s 等を主張するが、#3181 の direct-BF16 garbage 報告と同じリスク領域。 |
| [`Vontra/Qwen3.8-Flash-Next-MLX-4bit-MTP`](https://huggingface.co/Vontra/Qwen3.8-Flash-Next-MLX-4bit-MTP/tree/327c8a604de613b42f84ba5e6b796c0931e8aa3b) / `327c8a604de613b42f84ba5e6b796c0931e8aa3b`, 2026-08-26 18:33:58Z | 22、113,209,682,735 B = **105.43 GiB**（repo total 約105.45 GiB） | [`config.json`](https://huggingface.co/Vontra/Qwen3.8-Flash-Next-MLX-4bit-MTP/raw/327c8a604de613b42f84ba5e6b796c0931e8aa3b/config.json): uniform 4-bit、group32、affine; overrides 無し; MTP 1 layer、76 keys。 | uniform 4-bit候補。カード（[README](https://huggingface.co/Vontra/Qwen3.8-Flash-Next-MLX-4bit-MTP/blob/327c8a604de613b42f84ba5e6b796c0931e8aa3b/README.md)）の off 27.8314 / on 31.4418 tok/s、acceptance 68.23% は未検証。 |
| [`Vontra/Qwen3.8-Flash-Next-MLX-oQ4`](https://huggingface.co/Vontra/Qwen3.8-Flash-Next-MLX-oQ4/tree/fc2ad5d7afed7664fcefb10ead1749d62eda763c) / `fc2ad5d7afed7664fcefb10ead1749d62eda763c`, 2026-08-26 12:51:20Z | 22、111,691,934,009 B = **104.02 GiB**（repo total 約104.04 GiB） | [`config.json`](https://huggingface.co/Vontra/Qwen3.8-Flash-Next-MLX-oQ4/raw/fc2ad5d7afed7664fcefb10ead1749d62eda763c/config.json): mixed oQ 4-bit、group32、affine; overrides 5-bit/36、8-bit/192; `mtp_num_hidden_layers=0`、MTP keys 無し。 | MTP off の mixed 4-bit control。カードの M3 Studio warm 27–27.9 tok/s は未検証。 |
| [`Vontra/Qwen3.8-Flash-Next-MLX-4bit`](https://huggingface.co/Vontra/Qwen3.8-Flash-Next-MLX-4bit/tree/de597762aa61387c89590a46582222a261ce0387) / `de597762aa61387c89590a46582222a261ce0387`, 2026-08-26 12:51:18Z | 22、111,578,339,151 B = **103.91 GiB**（repo total 約103.93 GiB） | [`config.json`](https://huggingface.co/Vontra/Qwen3.8-Flash-Next-MLX-4bit/raw/de597762aa61387c89590a46582222a261ce0387/config.json): uniform 4-bit、group32、affine; overrides 無し; MTP 0、MTP keys 無し。 | uniform 4-bit MTP-off control。カードの速度は未検証で、#3181 の direct-BF16 conversion リスクがある。 |
| [`Vontra/Qwen3.8-Flash-Next-MLX-oQ2-MTP`](https://huggingface.co/Vontra/Qwen3.8-Flash-Next-MLX-oQ2-MTP/tree/8d07d17c7b66dbbbd9bd10abc98d21bc8ed26dcd) / `8d07d17c7b66dbbbd9bd10abc98d21bc8ed26dcd`, 2026-08-27 09:11:45Z | 15、70,782,136,152 B = **65.92 GiB**（repo total 約65.94 GiB） | [`config.json`](https://huggingface.co/Vontra/Qwen3.8-Flash-Next-MLX-oQ2-MTP/raw/8d07d17c7b66dbbbd9bd10abc98d21bc8ed26dcd/config.json): 2-bit、group32、affine; overrides 3-bit/297、4-bit/60、5-bit/36、6-bit/129、8-bit/224; MTP 1 layer、76 keys。 | 任意ではなく memory-stress 用の optional。カードは MTP off 26.1398 / on 25.9290 tok/s と報告し、速度上の優位を主張していない。品質 risk が大きいため primary matrix 外。 |

### 128GB では除外する容量帯

同じ Vontra 系の `MLX-6bit-MTP`（revision `b2fb2dddc3fcfd2429c680f32d884deb2f6c1338`、公開 2026-08-26 18:53:34Z、約147.22 GiB）、`MLX-oQ6-MTP`（`8a0b097ef61c31389bcec2f0192f542d2c644d24`、公開 2026-08-26 15:48:48Z、約147.21 GiB）、`MLX-oQ8-MTP`（`981fe13e691afb99d6768f50330ee64c11972984`、公開 2026-08-26 17:11:17Z、約188.82 GiB）は 128GB M4 Max の実用測定候補から外す。実効メモリ、KV cache、OS headroom を足す前に容量が大きすぎるためである。各 revision は [HF API](https://huggingface.co/api/models/Vontra/Qwen3.8-Flash-Next-MLX-6bit-MTP)、[HF API](https://huggingface.co/api/models/Vontra/Qwen3.8-Flash-Next-MLX-oQ6-MTP)、[HF API](https://huggingface.co/api/models/Vontra/Qwen3.8-Flash-Next-MLX-oQ8-MTP) で固定する。

## oMLX primary matrix（M4 Max 128GB）

### 対象

1. `Jundot/Qwen3.8-Flash-Next-oQ4e-mtp`（公式 oMLX 生成 oQ4e）
2. `Vontra/Qwen3.8-Flash-Next-MLX-oQ3-MTP`（mixed 3-bit）
3. `Vontra/Qwen3.8-Flash-Next-MLX-3bit-MTP`（uniform 3-bit）
4. `Vontra/Qwen3.8-Flash-Next-MLX-oQ4-MTP`（mixed 4-bit）
5. `Vontra/Qwen3.8-Flash-Next-MLX-4bit-MTP`（uniform 4-bit）
6. `Vontra/Qwen3.8-Flash-Next-MLX-oQ4`（同じ mixed 4-bit の MTP-off control）
7. `Vontra/Qwen3.8-Flash-Next-MLX-4bit`（uniform 4-bit の MTP-off control）

MTP 付き各モデルは同一重み・同一 process で **MTP off / on (max draft depth 3)** の両方を実行する。MTP 無し control は off のみ。oQ2 は primary から外し、memory-pressure が許す場合だけ品質 triage として追加する。

### 実行前の再現条件

- Chronovisor、他の推論サーバー、常駐モデルを停止した後、oMLX `v0.6.3` commit と上記依存を固定する。main/rc と混ぜない。
- いったん HF cache の symlink を使わず、各 revision を通常の実ファイル directory に materialize する。#3239 が解決済みと確認できるまでは、この条件を gate とする。
- `config.json` の `model_type=qwen4_exp`、index の shard 数/`mtp.*` key の有無、quantization bits/group/mode、revision SHA を read-back する。MTP 無しに MTP を要求しない。
- fresh process、warm-up 1 回を捨て、prefix-cache reuse を無効化（または `cached_tokens=0`）、同じ prompt/seed/sampling を使う。公式形式に合わせた throughput は `temperature=1, top_p=1, max_new_tokens=128`、context 4K/16K/32K、各 3 測定（ノイズが大きければ 5）とする。
- 記録: load time、PP/TG、TTFT/E2E、p50/p95、MLX peak、process RSS peak、memory pressure、swap/pageout、MTP draft/accepted/acceptance。128GB では measured working set に安全 headroom を残す（暫定目安 ≤110 GiB、最終閾値は実機で合意）。
- Chronovisor の実ワークロードは throughput 設定と分け、temperature 0 の固定 prompt corpus を使う。性能と品質の数字を混ぜない。

### 品質 battery と gate

固定した 100–200 件の held-out ケース（code edit/patch、JSON schema・structured tool call、retrieval/fact extraction、日本語/英語、4K/16K/32K needle、multi-turn）を同一 evaluator で実行する。記録対象は schema/tool parse validity、指示遵守、code test pass、omission/unsupported claim、output hash、garbage/repetition、MTP off/on の greedy 同値性、sandbox 外書き込み試行である。カードの自己申告値を独立 ground truth としない。

**Hard fail**:

- load error、`qwen4_exp` 以外への fallback、NaN/garbage/repetition、schema/tool parse 失敗、sandbox 外への unsafe write。
- 同一条件の MTP off と on で、verifier が同値を期待する greedy 出力が変わる。
- 測定中の swapout/pageout、または 128GB の実用 headroom を失う常時 memory pressure。

**Pass floor**:

- 固定 baseline（少なくとも Qwen3.8-27B の同一 corpus）に対する held-out task score の下側信頼限界が **−2 percentage points より悪化しない**（Wilson または bootstrap を事前固定）。
- schema/tool validity は baseline 以上、unsafe write は 0、retry/hold は別記する。

**選定**: gate を通った候補だけで、steady TG 最大、p95 E2E 最小、PP/TTFT とメモリ余裕を併記して決める。速度だけで候補を選ばず、品質 gate 不合格・swap 発生・garbage を出すモデルは除外する。第一走は Jundot oQ4e-mtp の smoke（MTP off/on）だが、これは優先順位であって合格判定ではない。

## 参考扱い（oMLX primary matrix 外）

- [`pipenetwork/Qwen3.8-Flash-Next-MLX-mixed-4_8bit`](https://huggingface.co/pipenetwork/Qwen3.8-Flash-Next-MLX-mixed-4_8bit) は routed experts 4-bit + attention/DeltaNet/hyperconnections/embedding 8-bit、MTP 無しの独自 runtime。README の wikitext paired NLL は BF16 4.4708、mixed4/8 4.5286（Δ +0.0128）、uniform4 5.3914（Δ +0.1872）と品質資料として有用だが、oMLX と同じ load/decode 経路ではない。
- [`txgsync/Qwen3.8-Flash-Next-MXFP4-MLX`](https://huggingface.co/txgsync/Qwen3.8-Flash-Next-MXFP4-MLX) は native MXFP4/Q4/Q5/Q6/Q8 の別プロファイル。experimental で throughput/quality benchmark 無し、MLX-VLM `>=0.6.17` 前提のため、oMLX v0.6.3 primary matrix には入れない。
- [`rapid-mlx/Qwen3.8-Flash-Next-4bit`](https://huggingface.co/rapid-mlx/Qwen3.8-Flash-Next-4bit)、[`ddalcu/Qwen3.8-Flash-Next-MLX-Serve-4bit`](https://huggingface.co/ddalcu/Qwen3.8-Flash-Next-MLX-Serve-4bit)、[`labhraighlep/Qwen3.8-Flash-Next-MLX-Serve-4bit`](https://huggingface.co/labhraighlep/Qwen3.8-Flash-Next-MLX-Serve-4bit) は custom server / external n-gram mmap / 別 batching 実装で、oMLX 比較ではない。
- [`sh0wie/Qwen3.8-Flash-Next-REAP-288-MLX-4bit`](https://huggingface.co/sh0wie/Qwen3.8-Flash-Next-REAP-288-MLX-4bit) は 512→288 expert pruning を含むため量子化だけの比較ではない。

以上は候補を絞るための調査であり、M4 Max 128GB 上の性能・品質・本番適合性を確定する測定結果ではない。
