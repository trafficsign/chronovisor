# Qwen3.8-Flash-Next 125B / oMLX 0.6.3 実機比較

測定日: 2026-08-29。対象は Qwen3.8-Flash-Next 125B の MLX 量子化だけであり、27B は比較表から除外した。候補調査時の 100–200 問計画は実行せず、ユーザー指定に従い 26 問の短い品質 smoke と速度測定へ縮小した。本書の実行条件が、先行する量子化調査文書の実行計画を置き換える。

## 結論

**24/26 対 23/26 の1問差から、oQ4eの方が高品質とは判定できない。** 差が出たのは `sum(i*i for i in range(5))` の1問だけで、逆方向の差は0問。exact McNemar両側検定は p=1.0、95% Wilson区間も大きく重なる。26問smoke上は、oQ4eとoQ3の品質を実質同等として扱う。

同じSSD mmap条件ではoQ3が63.06 GB、oQ4eが69.93 GBで、oQ3の方が6.87 GB軽い。oQ3はprefillも約1%速い。一方、同一の実コードpromptではoQ4eが38.2 tok/s、oQ3 mmapが34.0 tok/sで、256-token込みE2EはoQ4eが0.1–0.8秒だけ速かった。したがって、**短中程度の実コード応答を含む最速構成は暫定的にoQ4e mmap、最小メモリ構成はoQ3 mmap、長い出力のdecode最速はoQ3 resident** となる。oQ3 residentは47.8 tok/sだが86.90 GBを使い、一時soft memory pressureがある。

oQ4はoQ4eより実コード生成が約6%速いだけで品質が22/26、uniform 3bitは実コード生成47.1 tok/sでも品質21/26かつ32K throttleのため、どちらも選ばない。Chronovisorのroutingはまだ変更せず、選定結果だけを確定した。

## 固定条件

- Mac Studio `Mac16,9`、Apple M4 Max、128 GB unified memory、16 CPU cores。
- oMLX app/CLI final `0.6.3` build `260828014302-macos26-27`、bundled MLX `0.32.0`。
- localhost、1 concurrent request、prefix/HF cache 無効、memory guard 104 GB。品質・needle・synthetic generation は temperature 0、実コード生成は temperature 1 / top-p 1 / seed 6330。
- MTP は adaptive depth 3。主比較のoQ4e/oQ3はPLEをSSD mmapに揃え、oQ3だけresident構成も別枠で測った。不安定だった設定も結果として記録した。
- 重みは revision を固定し、通常ファイルへ materialize。symlink と incomplete shard は 0。
- 品質は exact-answer 20 問 + JSON 6 問。性能は 4K/16K/32K needle 各 3 回の中央値、256-token synthetic generation 3 回、および実用コード生成 3 回。
- PP/effective は API の総時間から算出した effective prompt tok/s で、oMLX 内部 profiler の純 PP と同一ではない。

固定revisionは oQ4e `2615fc0e976e65c2f3b55daca3a948f1cdc5b9f8`、oQ3 `dbcf52e7921f4ed441f7bb46ff3b440d25c19744`、uniform 3bit `051d36f1a21aa85403ca52cba889dcea6c94c498`、oQ4 `43a82b3f0ff64fa417fd09ca046580f08d19b0d6`。

## ローカル実測

| 125B model | quant / PLE | 重み | 実ロード | 品質 | JSON | 4K PP/effective | 16K | 32K | synthetic TG | 実コード TG | 判定 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| Jundot oQ4e MTP | mixed 4-bit / SSD mmap | 99.00 GiB | 69.93 GB | **24/26** | **6/6** | 574.3 | 554.7 | 488.5 | 61.7 | **38.2** | mmap E2E僅差1位 |
| Vontra oQ3 MTP | mixed 3-bit / SSD mmap | 86.15 GiB | **63.06 GB** | 23/26 | **6/6** | **580.0** | **559.1** | 493.7 | 64.0 | 34.0 | 最小メモリ、prefill僅差1位 |
| Vontra oQ3 MTP | mixed 3-bit / resident | 86.15 GiB | 86.90 GB | 23/26 | **6/6** | 324.3 | 484.5 | **508.9** | **68.6** | **47.8** | decode最速、長文向け |
| Vontra uniform 3bit MTP | uniform 3-bit / resident | 84.54 GiB | 85.26 GB | 21/26 | **6/6** | 約434 | 約606 | 1回目134秒、以後throttle | 58.2 | **47.1** | 不合格 |
| Vontra oQ4 MTP | mixed 4-bit / SSD mmap | 105.54 GiB | 76.49 GB | 22/26 | **6/6** | 572.6 | **554.9** | 490.8 | **68.6** | 40.5 | 品質差に見合わず除外 |

単位は容量を除き tok/s。oQ4e / oQ3 / oQ4 の needle は全context 3/3。oQ3 mmapはoQ4e mmapより実ロードが6.87 GB（9.8%）小さく、4K/16K/32K prefillがそれぞれ1.0%/0.8%/1.1%速い。uniform 3bit の 32K は最初の正答 1 回後に memory guard が継続作動したため中央値を作らず失格扱いとした。実コードTGは同一コードpromptの生成速度であり、生成物のtest pass率ではない。

### 256-token生成込みの概算E2E

各contextのmedian秒に、実コード生成256 tokenのmedian秒を加えた単純比較である。

| model | 4K | 16K | 32K |
|---|---:|---:|---:|
| Jundot oQ4e | 13.74秒 | 36.14秒 | 73.67秒 |
| Vontra oQ3 / SSD mmap | 14.49秒 | 36.73秒 | 73.79秒 |
| Vontra oQ3 / resident | 17.81秒 | 39.06秒 | **69.64秒** |
| Vontra oQ4 | **13.37秒** | **35.74秒** | 72.96秒 |

oQ4はE2EだけならoQ4eを0.4秒ほど上回るが、品質が2件落ちるため採らない。mmap同士ではoQ4eとoQ3はほぼ同率で、実コードdecodeの差によりoQ4eが0.12–0.75秒先行した。resident oQ3は32Kで約4秒勝ち、4K/16Kでも出力が概ね1000/800 tokenを超えるとdecode優位が初期差を回収する計算になる。

### MTP on/off control

oQ3 の同一重みでは品質は on/off とも 23/26。residentでのMTP onはsynthetic TG 68.6 tok/s、offは28.1 tok/s。一方、短いneedleのeffective PPはメモリ圧力の影響が大きく、onが4K 324.3 / 16K 484.5 / 32K 508.9、offが558.4 / 612.8 / 494.1 tok/sだった。SSD mmap + MTP onでは4K 580.0 / 16K 559.1 / 32K 493.7、実コードTG 34.0 tok/s。速度目的ならMTP onを維持するが、resident/mmapは典型prompt長とoutput長で選ぶ。

## 品質差

| model | 失敗した exact case | JSON |
|---|---|---:|
| Jundot oQ4e | remainder、Python precedence | 6/6 |
| Vontra oQ3 | remainder、sum-of-squares code、Python precedence | 6/6 |
| Vontra uniform 3bit | remainder、work、set intersection、Python precedence、日本語 order | 6/6 |
| Vontra oQ4 | remainder、sum-of-squares code、Python precedence、日本語 order | 6/6 |

26 問は量子化候補を落とす smoke であり、一般能力の統計的ベンチマークではない。ただし全モデルへ同一 deterministic prompt を投げているため、この選定内の相対 gate として使う。

oQ4eとoQ3のpaired比較では、oQ4eだけが通ったcaseは1、oQ3だけが通ったcaseは0。oQ4eの正答率92.3%の95% Wilson区間は75.9–97.9%、oQ3の88.5%は71.0–96.0%で大きく重なる。exact McNemar両側 p=1.0 なので、この1問差を量子化bit数による品質差とは解釈しない。

## 公開情報との照合

- [Qwen 公式 card](https://huggingface.co/Qwen/Qwen3.8-Flash-Next-FP8) は 125B total / 6B activated、51B N-gram embedding、MTP 構成を示す。
- [oMLX 0.6.3 release](https://github.com/jundot/omlx/releases/tag/v0.6.3) は `qwen4_exp`、PLE SSD mmap、Lightning MTP depth 3 を正式対応し、M3 Ultra 512 GB の oQ4e で 4K/16K/32K TG 58.1/53.7/49.0 tok/s を報告する。別ハードなので順位には使わない。
- [Jundot oQ4e](https://huggingface.co/Jundot/Qwen3.8-Flash-Next-oQ4e-mtp) と [Vontra oQ3](https://huggingface.co/Vontra/Qwen3.8-Flash-Next-MLX-oQ3-MTP)、[uniform 3bit](https://huggingface.co/Vontra/Qwen3.8-Flash-Next-MLX-3bit-MTP)、[oQ4](https://huggingface.co/Vontra/Qwen3.8-Flash-Next-MLX-oQ4-MTP) はすべて公開 revision を SHA 固定して取得した。
- 公開 oMLX 測定では [Jundot oQ4e / M4 Max 128 GB / 8K](https://omlx.ai/benchmarks/performance/pcqhpxqy) が PP 294.7 / TG 23.1 / peak footprint 77.12 GB、[Vontra oQ4 / M5 Max 128 GB](https://omlx.ai/benchmarks/performance/4lrsbs9m) が 4K/8K TG 47.1/47.9、32K 30.8 tok/s を記録する。sampling と SoC が異なるため、これも候補選定の裏付けに限る。

## 運用状態

- Chronovisor、teacher、semantic/reranker/dashboard/ingest、関連 LaunchAgent、MCP 接続は停止済み。Codex、Claude Code、共通MCP設定からChronovisorの自動起動定義も除去し、ベンチマーク中に再起動していない。
- 旧 oMLX 0.6.2 / 0.6.3rc2 の実行物、2本のRC2実験runtime、uv package archive、旧cache、log、LaunchAgent は recoverable quarantine へ退避し、final 0.6.3 だけで測定した。`~/.omlx/bin/omlx` と app CLI はともに0.6.3。
- ユーザーが比較対象から外した27B/100問の結果と、旧symlink/materialize test runtimeも同じrecoverable quarantineへ退避した。125Bの重みと本測定結果は保持した。
- cutover合意前なので Chronovisor のモデル設定・routing は変更していない。

## 成果物

- harness: `scripts/qwen_next_benchmark.py`
- raw results: `/Users/trafficsign/projects/sandbox/chronovisor-qwen-next-bench/results/`
- model/runtime workspace: `/Users/trafficsign/projects/sandbox/chronovisor-qwen-next-bench/`
- recoverable quarantine: `/Users/trafficsign/.Trash/omlx-old-runtime-20260829-T46eRS/`
