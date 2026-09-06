# Flash Next VQ 3.2bpw 実機ベンチマーク

2026-09-06 / Apple M4 Max / 128GiB RAM。

## 結論

今回の構成では切替を推奨しない。「実メモリ10GB以上削減・現行4bitに近い品質・速度」の同時達成には届かなかった。

- 速い方のMTP有効構成でも、physical footprintの削減は約3.1GiB。OSへの解放量の差も約3.68GiBで、10GB削減には足りない。
- 日本語生成は40.26 → 30.92 tok/s（約23%低下）。32K読込は46.71 → 96.55秒（約2.07倍）。
- 小規模な汎用テストは24/26 → 21/26。追加日本語12問と、曖昧な設問を除いたChronovisor判断は良好だったが、一般品質の同等性は確認できない。
- MTPなしでは約6.3GiB減るが、生成は19.51 tok/sまで低下。32K読込も改善しなかった。

本番への切替は行っていない。モデル・依存関係は実験用ディレクトリに隔離した。

## 正式測定の対象と条件

| 項目 | 現行 | VQ候補 |
|---|---|---|
| モデル | Jundot/Qwen3.8-Flash-Next-oQ4e-mtp | TheDrainFlorist/Qwen3.8-Flash-Next-VQ-3.2bpw |
| revision | `2615fc0e976e65c2f3b55daca3a948f1cdc5b9f8` | `b3a40c3590785c7276e6d51bda97486d0b179f0e` |
| エンジン | oMLX 0.6.4 | VQLab `36f700cf8a335a760561321d5e8136fac4e189e2` |
| MLX-LM | oMLX同梱版 | `8a36d1ece51b34a2e76d9f7064a7f090334f39c4` |
| 最適化 | 現行設定由来、MTP、PLE mmap、ANE | 専用VQカーネル、MTP sidecar |

VQはPython 3.12.12 / MLX 0.32.2。MLX-LMの表示バージョンは両調査版とも0.32.0なので、再現にはcommit固定が必要。正式測定は上記互換版のstock VQLabで行い、ローカル補正コードは使っていない。

Chronovisorの11サービスと本番oMLXを停止し、port 18136に一モデルずつ起動。本番port 18125と実験portの解放を確認してから各試行を開始した。現行の測定中はダウンロードも停止、候補の測定はダウンロード完了後。temperature=0、thinking無効、単一リクエスト、prefix cacheなし。32Kまでの各長文テストと日本語384-token生成は3回の中央値。MTPなしの32Kのみ1回の補足値。

量子化だけを切り替えた比較ではなく、このMac上で利用できるエンジン込みの実用比較である。エンジン差と量子化自体の影響を完全には分離していない。

## 結果

| 指標 | 現行/MTP | VQ/MTP | VQ/MTPなし・補足 |
|---|---:|---:|---:|
| 汎用26問 | 24/26 | 21/26 | 全件再測定なし |
| 追加の日本語12問 | 12/12 | 12/12 | — |
| 汎用内の日本語を含む合計 | 16/16 | 15/16 | — |
| Chronovisor JSON schema valid | 18/19 | 19/19 | — |
| Chronovisor判断一致 | 17/19 | 18/19 | — |
| 既知の曖昧な1件を除外 | 17/18 | 18/18 | — |
| Chronovisor処理時間中央値 | 5.56秒 | 7.32秒 | — |
| 4K読込＋短い回答 | 6.29秒 | 11.65秒 | — |
| 16K読込＋短い回答 | 23.50秒 | 44.20秒 | — |
| 32K読込＋短い回答 | 46.71秒 | 96.55秒 | 96.66秒・1回 |
| 長文内の情報抽出 | 9/9 | 9/9 | 1/1 |
| 日本語生成 | 40.26 tok/s | 30.92 tok/s | 19.51 tok/s |
| 処理後physical footprint | 75.7GiB | 72.6GiB | 69.4GiB |
| physical footprintピーク | 81.4GiB | 82.5GiB | 78.6GiB |
| 処理後TOTAL resident | 84.1GiB | 72.9GiB | 69.7GiB |
| プロセス終了時のOS使用量減少 | 76.77GiB | 73.09GiB | 70.32GiB |
| 測定開始→終了のSwapouts増加 | 0 | 0 | 0 |

長文はalpha反復文の先頭・中央・末尾に置いた情報を抽出する課題。自然な長文の理解力全般を保証するテストではない。日本語生成は同一説明課題に試行番号を付け、384 tokensで打ち切る。内容の違いによる実行量の差は残る。

## メモリを過大評価しないための解釈

TOTAL residentだけを見ると、MTP有効でも11.2GiB減っている。しかし現行側には7.3GiBのclean mapped fileが含まれる一方、VQ側は約4.5MiBだった。この種のページはphysical footprintと同じ負担ではなく、プロセス終了後もファイルキャッシュとして残り得る。

実際、physical footprint差は3.1GiBで、独立に測ったOSの解放量差も3.68GiBだった。両者のgraphics領域のresidentはどちらも約72.1GiB。**TOTAL resident差をそのまま「10GB以上の実用的なメモリ余裕」として宣伝しない。** MTPなしでもphysical差6.3GiB、OS解放量差6.44GiBに留まる。MTP有効のピークはむしろ現行より約1.1GiB大きい。

OS使用量は `page_size × (active + inactive + wired + compressor)`。停止直前→直後は、現行131,062,874,112 → 48,634,462,208 bytes、VQ/MTP124,094,545,920 → 45,616,857,088 bytes、VQ/MTPなし127,841,353,728 → 52,333,133,824 bytes。OS全体には他アプリやキャッシュの変動があるので厳密な所有者別会計ではないが、physical footprintと規模が整合する。

1秒ごとのvm_statとRSS、ロード後・処理後のvmmapを保存した。MLX内部のactive-memory表示やダウンロード容量だけで判定していない。

## 品質・JSON互換性

現行は正解し、VQが追加で落とした問題は以下の3件。

- 作業量の比例計算: 正解120に対して150。
- 真偽の論理問題: 正解Aに対してB。
- 日本語の新旧比較: 正解「乙」に対して「甲」。

余りの計算で数字だけの指示を守らず128-token上限に達する問題、Python演算優先順位の問題は両者とも不合格。MTPなしでも作業量150・新旧比較「甲」を再現したので、その2件はMTPの有効化だけでは説明できない。小規模テストのため広範な品質差の推定には限界がある。

VQLab側のAPIは`response_format.json_schema`を扱わない。専用キー`probe=NATIVE_SCHEMA`をschemaだけで要求する試験では、現行は指定どおり返すが、VQは無関係なJSONを生成した。このAPI互換性差を量子化の品質差に混ぜないため、正式なペア比較では**両者ともschemaを同じ文章で追加し、response_formatを削除**した。

従って表のChronovisor結果は、現行の通常運用そのものではない。現行が落とした1件は`ingest_reconciliation:4`のJSON解析エラー。VQはここを通過した。両者が落とした`lane-contract-v28:recall_improvement:3`は既知の曖昧なfixtureで、除外値も併記した。前回の現行・schema強制ありの結果は、曖昧な1件を除くと18/18だった。VQがChronovisorで明確に優位とは結論しない。

ツール呼び出し全般やoMLXへの直接組込みは未評価。VQLabをそのまま既存APIの完全互換代替とは扱えない。

## 発見したランタイム不整合と無効試行

作者の案内どおりPR #1788の最新head `2196836` を使うと、`Return only READY.`に対して`ECHECHECH...`が返った。MTPなし、fused生成カーネルを迂回した場合も同じ。151ファイルのSHA256/git-blob hashは配布元と全て一致し、付属カーネルの10検査もPASSした。

原因は[MLX-LMの9月2日の変更ac83bb4](https://github.com/ml-explore/mlx-lm/commit/ac83bb43ed08ff3996fe93d8c41475ac59ec4590)。RMSNormの`1 + weight`をロード時に折り込むようになったが、その判定はraw HFのキー名を前提にする。旧仕様で変換されたVQのflat `model.*`キーでは折り込みが抜ける。VQLabのMTP headも旧来のruntime加算を前提にしていた。

本体の148本だけを一時的にin-memory補正するとREADYと正常回答へ回復したが、MTP受理率は0%のままだった。この試行は原因切り分け用で、最終比較には使用していない。ディスク重みは一切変更していない。

最終的に[変更前の互換commit 8a36d1e](https://github.com/ml-explore/mlx-lm/commit/8a36d1ece51b34a2e76d9f7064a7f090334f39c4)へ依存関係を固定し、補正なしのstock VQLabで全測定を実施。日本語生成時のMTP受理率は79.7%、74.5%、74.0%へ戻った。初期の破損出力や補正途中の速度を品質・速度の最終結果に混ぜていない。診断初期に約425MiBのSwapouts増加があったが、正式な3構成の各測定区間では全て増加0だった。

## 保存・再現

- 結果、生応答、全ファイルhash、OSメモリ、試行の有効/無効区分: [evidence/2026-09-06-vq32-benchmark](evidence/2026-09-06-vq32-benchmark)
- 測定コード: `scripts/vq32_benchmark.py`。既存の`qwen_next_benchmark.py`、`sawfwair_quant_benchmark.py`とcanonical corpusを再利用した。
- `.venv/bin/python scripts/vq32_benchmark.py self-check`、対象コードのruffチェックを実行。
- モデルと隔離venv: `/Users/trafficsign/.omlx/experiments/vq32/`。151ファイル合計76,996,925,441 bytes（約71.7GiB）を保持。本番モデルへの登録はしていない。
- 再測定のarmは`baseline`、`vq-compatible`、`vq-compatible-no-mtp`。先にサービスを停止して両ポートを空ける必要があり、このスクリプト自体はサービス復帰を担当しない。runtime-compatible.jsonのcommitへ固定すること。
- 本体のみの補正コードは`diagnostic_norm_shim.py`として証跡内に隔離。これは互換性診断用であり、最終ランタイムでも推奨運用でもない。

## 本番復帰

`production-restored.json`で以下を確認済み。

- 現行oQ4eがdefaultのoMLXはhealthy。実APIで`RESTORED`の生成に成功。
- Chronovisorの11サービスがenabled、DashboardはHTTP 200、Semantic/Rerankerはstatus=okかつready=true、Ingestとauthority preflightはready。
- 実験port 18136は閉鎖。設定3ファイル（Chronovisor config、oMLX settings/model_settings）のSHA256は開始前と完全一致。
- VQを本番に登録・切替していない。隔離venvと約72GiBのモデルは実験用として保持。pushはしていない。
