# Sawfwair Flash Next mixed Q3/Q4 benchmark

実施: 2026-09-06 / Apple M4 Max 40-core GPU / 128GB / oMLX 0.6.4。

## 結論

今回は約15GiBのモデル常駐メモリ削減を確認できた。ただし汎用26問は現行24/26から候補21/26へ低下し、日本語生成もMTP無効で約9%、MTP有効で約26%遅かった。「10GB以上削減・現行4bitに近い品質・現行同等以上の速度」の完全達成とは判断しない。本番への切替は行わない。

この評価は、このMacとoMLXでの実測である。作者が推奨するmere.runでの速度、一般的な日本語品質、BF16との品質差は未検証。

## 対象

- 現行: `Jundot/Qwen3.8-Flash-Next-oQ4e-mtp` / revision `2615fc0e976e65c2f3b55daca3a948f1cdc5b9f8`
- 候補: `Sawfwair/Qwen3.8-Flash-Next-MLX-Activation-3bit-Native-PLE` / revision `1cee9301c745836e0abb8933e89cf27a38b98125`
- 候補は150ファイルの存在・サイズと3,817テンソルの索引を検証。144個のrouted-expert投影がQ3/group-64、その他の対象行列は主にQ4、PLEはQ4/group-32。ダウンロードは89,667,182,797 bytes。
- 実験モデルは `~/.omlx/experiments/sawfwair-q3-native-ple/model` に保存。本番モデル登録・本番設定への追加は行っていない。

## 結果

| 指標 | 現行 oQ4e / MTP有効 | 候補 / MTP有効 |
|---|---:|---:|
| 汎用 exact / 26問 | 24/26 | 21/26 |
| 追加の日本語12問 | 12/12 | 12/12 |
| 日本語課題合計（汎用内4問＋追加12問） | 16/16 | 15/16 |
| Chronovisor schema | 19/19 | 19/19 |
| Chronovisor semantic effect（既存採点そのまま） | 18/19 | 19/19 |
| 曖昧な1件を除いたsemantic effect | 18/18 | 18/18 |
| Chronovisor処理時間中央値 | 9.40秒 | 7.69秒 |
| 4K読込＋短い回答・中央値 | 6.46秒 | 6.60秒 |
| 16K読込＋短い回答・中央値 | 24.34秒 | 24.03秒 |
| 32K読込＋短い回答・中央値 | 48.79秒 | 47.64秒 |
| 長文内の情報抽出 | 9/9 | 9/9 |
| 日本語384-token生成・中央値 | 37.50 tok/s | 27.60 tok/s |
| 同生成・候補MTP無効の補足測定 | — | **34.10 tok/s** |
| ロード後physical footprint | 73.6GiB | 58.5GiB |
| 一連の処理後physical footprint | 74.9GiB | 59.1GiB |
| 処理後vmmap TOTAL resident（clean mappedも含む） | 82.9GiB | 67.5GiB |
| physical footprintピーク | 81.5GiB | 66.2GiB |
| 停止直前→停止直後のOSページ使用量減少 | 74.93GiB | 60.35GiB |
| 主比較中のSwapouts増加 | 0 | 0 |

速度は3回の中央値。4K/16K/32Kの実入力token数はそれぞれ4,050 / 16,338 / 32,722。反復alpha文中のneedle抽出であり、自然な長文全般の速度や理解力を代表するものではない。日本語生成は3種類の試行番号を付けた同じ説明課題を384 tokensで打ち切り、最初のcontentから生成終了までの速度を測定した。生成内容が異なるので演算量が完全に同一という意味ではない。

## メモリの解釈

今回の省メモリ判定はoMLX内部のmodel-memory表示だけに依存していない。処理後の全resident差は15.4GiB、physical footprint差は15.8GiBであり、同じプロセスを終了した際のOS使用量の減り方でも14.58GiBの差を確認した。

OSページ使用量は `page_size × (active + inactive + wired + compressor)` と定義し、生のvm_stat全項目を1秒間隔で保存した。OS全体の停止前の絶対値は、現行130.27GB、候補129.84GBとほぼ同じだった。これはファイルキャッシュ・他アプリを含むため、その絶対値をモデルの消費量と同一視しない。また、実行前のno-model値との差だけも、ロードによるファイルキャッシュ変化を含むため削減量として採用しない。各armの停止直前／直後と、clean mappedを含むプロセスresidentを併記する。

主比較の停止直前／直後は、現行130,267,627,520 → 49,816,813,568 bytes、候補129,838,792,704 → 65,042,415,616 bytes。OS全体の値には外部変動があり、これらは厳密なページ所有者別会計ではないが、プロセス側の独立した観測と方向・規模が一致している。

## 品質差とMTP

候補だけが追加で落とした問題:

- `work`: 数字だけの回答指定に反し説明を始め、128-token上限までに答えを返さない。
- `code`: `sum(i*i for i in range(5))` の期待値30に対し55。
- `jp_order`: 「甲は乙より古く、丙は甲より古い。最も新しいもの」の期待値「乙」に対し「甲」。

これら3問は、競合を除いたMTP無効の再測定でも同じ失敗だった。対照の`mul_sub`は正解。したがって、この追加退行をMTP有効化だけの問題として解消することはできなかった。ただし量子化そのものとloaderの影響を完全に分離した評価ではない。

候補のMTPは起動したが、needleや日本語生成で受理率0%の区間と、自動的に通常生成へ退避するログが繰り返された。現行の日本語生成では受理率61.6〜67.0%。候補ではMTPを切ると27.60 → 34.10 tok/sへ改善したが、現行37.50 tok/sには届かなかった。MTPの互換性や低受理率の根本原因を修正する作業は今回の範囲外。

`lane-contract-v28:recall_improvement:3` は既知の曖昧なfixtureである。入力のtop-level statusは`candidate`、best.statusは`candidate_pass`。指示は実際のproposal recordの`candidate_pass`を要求する。現行は不足と解釈してhold、候補はapprovedと解釈して既存期待値に一致した。この1点を候補の品質優位の証拠にはせず、他の18件は両者とも一致と報告する。fixtureは変更していない。

## 条件、無効試行、再現

- 11個のChronovisor managed-v2サービスを一時停止。現行と候補を専用port 18136で一モデルずつ起動。基準測定中はダウンロードも停止。
- 同じoMLX 0.6.4、production由来のモデル設定、PLE mmap、ANE最適化、MTP depth 3、temperature 0、thinking無効、single sequence。両者ともpaged cacheとHF cache discoveryを無効化。速度測定のusageでcached_tokens=0。
- 26問の既存汎用テスト、追加日本語12問、既存canonical corpusから19レーン各1件を使用。小規模な選抜テストであり、一般品質の同等性の証明ではない。
- 最初のMTP無効補足試行ではoMLXアプリの管理状態と残存serverがずれ、`stop`成功表示後も本番port 18125が開いていた。この試行中にswap usedが約12GBから約54GBへ増えた。**この試行の全測定を無効**にし、`invalid-overlap/`へ保存した。
- 補足を再試行する際はport 18125の解放を必須にし、残存を検出した2回は新モデルの起動前に拒否。残存本番PIDを確認してSIGTERMで正常終了後、サービス11個と本番ポートが停止した状態でMTP無効補足を実行し直した。採用した補足値はトップ階層の`candidate-no-mtp-performance.json`。初回の重複試行と、その後に拒否したサービス復帰ログは最終稼働状態の証拠として使用しない。
- 最終補足開始時のswap usedは13,638.5MiBまで戻っており、ロード後は13,630.5MiB、終了後は13,415.94MiB。ただし補足前後のSwapoutsは6,754,715 → 6,787,004（約504.5MiB相当）増加した。二重モデル常駐はないが、主比較と違ってOSのページ移動が残るため、MTP無効の速度は補足参考値として扱う。
- `scripts/sawfwair_quant_benchmark.py self-check` とruffを実行。測定時のserverログは実験ディレクトリ内の各armの`console.log`、結果とOS生データは本報告に対応するevidenceディレクトリ。

再現コマンド: `.venv/bin/python scripts/sawfwair_quant_benchmark.py baseline`、同`candidate`。サービス停止・復帰を含むため、作業中の本番に対して無断で実行しない。`candidate-no-mtp`は追加の失敗問題と生成速度だけを測定する。

証跡: [`evidence/2026-09-06-sawfwair-q3-benchmark`](evidence/2026-09-06-sawfwair-q3-benchmark)

## 最終復帰確認

- 現行oQ4eがdefaultのoMLXはhealthy。実APIで`RESTORED`の生成に成功。
- Chronovisor managed-v2の11サービスがenabled。Dashboard HTTP 200、SemanticとRerankerはok/ready、Ingest authority preflightはready。
- 実験port 18136は閉鎖済み。
- `~/.chronovisor/config.toml`、`~/.omlx/settings.json`、`~/.omlx/model_settings.json`は開始前とSHA256が完全一致。
- 最終確認時swap usedは13,303.94MiB、memory_pressureのfree表示は31%。重複試行中の約54GBからは戻っている。
- 候補の約84GiBのファイルは実験用ディレクトリに保持。本番への切替、モデルの恒常的な追加登録、pushは行っていない。

## ベンチ後の片付け（2026-09-06追記）

- 上記の保持状態は測定終了時点の記録。ユーザー指示により、使用中プロセスと本番設定からの参照がないことを確認し、`/Users/trafficsign/.omlx/experiments/sawfwair-q3-native-ple/`（モデルと実験設定）を削除した。
- レポート、証跡、測定スクリプトは保持。再測定には記載revisionのモデル再取得と実験設定の再作成が必要。
- VQ実験環境と合わせ、`df`の空き容量は約155.3GiB増加。現行oQ4eのoMLXは削除後もhealthyで、停止・設定変更はしていない。
