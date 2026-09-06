# DwarfStar Qwen3.8 Flash Next Q4: local benchmark

測定日: 2026-09-06。対象: Apple M4 Max / 128 GiB / macOS 26.6.2。
本番切替ではなく、専用ポート18136での一時的な比較実験。

## 判定

**このMacではプリフィルの大幅高速化はなかった。一方、MTPありの日本語生成は現行比約37%速く、会話履歴再利用時の待ち時間は大きく短縮した。**
簡易品質は3構成とも30/32。本番切替はしていない。メモリ会計の異なる実装なので、低いphysical footprintだけで必要RAM削減量を断定しない。

## 対象と再現条件

- 現行側: oMLX 0.6.4 / Jundot `Qwen3.8-Flash-Next-oQ4e-mtp`。既存モデルへのsymlinkを使い、既存モデル設定を隔離領域にコピー。MTP draft=3、PLE SSD offload、ANE関連設定を維持。設定上のcontext上限114688。
- 候補側: [ivanfioravanti/ds4-metal](https://github.com/ivanfioravanti/ds4-metal/tree/236cb2a549d05ea1941a19bd4f154e0abc858661)、commit `236cb2a549d05ea1941a19bd4f154e0abc858661`。`make -j2`でビルド。Metal / context上限65536 / power100 / prefill chunk既定8192。
- 重み: [ivanfioravanti/Qwen3.8-Flash-Next-DS4-Q4](https://huggingface.co/ivanfioravanti/Qwen3.8-Flash-Next-DS4-Q4/tree/59a55fb819c82be7b162948282b50bd1a1e290b7)、revision `59a55fb819c82be7b162948282b50bd1a1e290b7`。main GGUF 74,879,771,648 bytes + external PLE Q4_1 GGUF 32,000,157,440 bytes。両方SHA256をHub値と照合済み。mainにMTP重みを含む。MLXの重みをそのまま使ったものではない。
- DwarfStarのMTP armは `--mtp --mtp-exact-sampling`。両DwarfStar armは `qwen3.8-flash-next-chat` alias。
- 全armでthinking無効、temperature=0、seed=42、逐次リクエスト。Chronovisorのenabledな11サービスと本番oMLXを一時停止。実験終了後に復帰。
- baseline中はダウンロードをSIGSTOP。候補測定中にダウンロード・ハッシュ計算・ビルドを重ねていない。

本番モデルは変更していない。これは**量子化方式だけではなく、runtime・キャッシュ・モデルの組み合わせの比較**。

## 測定方法

`scripts/ds4_qwen_benchmark.py`が既存の`qwen_next_benchmark.py` / `sawfwair_quant_benchmark.py`の公開・合成テストを再利用。

- warmup後、4K/16K/32K相当の入力を各3回。実際のprompt tokensはそれぞれ4048 / 16338 / 32722。先頭を試行別に変え、`cached_tokens=0`を全件確認。針となる値を先頭・中間・末尾に配置。これは**モデル起動後のキャッシュミス入力**であり、cold model loadやOSキャッシュを消した測定ではない。
- 16K入力の直後に同一履歴を送るfollow-upを各3回。キャッシュ再利用数を保存。
- 同じ日本語説明課題を各3回、最大384生成tokens。クライアントの最初の可視出力までをTTFTとし、`(completion_tokens-1)/(終了時刻-最初の可視chunk時刻)`をdecode速度の推定値とする。SSE chunkが1tokenとは限らないため、全wall timeによる出力速度も併記する。
- 品質は既存のexact-match20問+日本語12問。最大128生成tokens、追加のhidden reasoningは認めない。長文針テスト9件とfollow-up3件も別集計。
- 1秒間隔のvm_stat / swap / psと、warmup後・測定後のvmmap summaryを保存。
- 各armを順番に測定した3回中央値。無作為化・交互実行・信頼区間はなく、他のpromptや長時間負荷への一般化はしない。

## 速度

各3回の中央値。時間は小さい方、速度は大きい方がよい。

| 測定項目 | 現行oMLX（MTPあり） | DS4 MTPなし | DS4 MTPあり |
|---|---:|---:|---:|
| 新規4048 tokens: TTFT | 6.372秒 | 6.345秒 | 6.333秒 |
| 新規16338 tokens: TTFT | 23.971秒 | 25.476秒 | 25.565秒 |
| 新規32722 tokens: TTFT | 47.536秒 | 51.293秒 | 51.386秒 |
| 16K履歴follow-up: TTFT | 6.524秒 | 0.266秒 | 0.270秒 |
| 日本語384 tokens: decode推定 | 35.81 tok/s | 41.30 tok/s | 49.07 tok/s |
| 同上: 全wall timeでの出力速度 | 34.18 tok/s | 39.91 tok/s | 46.93 tok/s |
| 簡易品質 | 30/32 | 30/32 | 30/32 |

新規長文入力はDS4 MTPありの方が16Kで約7%、32Kで約8%遅い。DS4ログの長文純prefill速度は約640 tok/s。
スクリーンショットのM3 Ultra / 512 GiBでの約1000 tok/sを、このM4 Maxの期待値にはしない。ログ上、M4 MaxではMetal 4 tensor APIも無効。

follow-upの入力は全構成16369 tokensで同一履歴。現行oMLXは12288 tokensを再利用、DS4は16349前後を再利用した。この**キャッシュ動作込み**で約24倍のTTFT差が出た。全新規入力を24倍速く読めるという意味ではなく、別内容の独立ジョブにはそのまま適用できない。baselineの隔離キャッシュ上限はSSD5GB / hot2GB、DS4はdisk KV cacheなしのlive prefix reuse。

生成は全試行384 tokens、finish=length。DS4 MTPありは現行比約37%速く、DS4 MTPなし比約19%速い。各arm内のdecode速度範囲は現行34.51–40.02、DS4 MTPなし41.12–41.64、MTPあり47.40–50.18 tok/s。baseline JSONにない`output_tokens_per_second`は保存済みusage/wallから再計算した。

## 品質と適用限界

3構成とも30/32。同じ2問を失敗した。

- `remainder`: 答え19だけを返さず解説し、128tokens上限に達した。
- `precedence`: Python式の正解53に対し14。

日本語12問、長文針9件、follow-up3件は全構成で全件成功。
少数の簡単なexact-match課題なので、同点は4bit品質同等の証明ではない。
実際のChronovisorの構造化出力・ツール呼出し・同時処理・長時間運用・262K入力は未評価。
入力token数が一致しても、各runtimeのテンプレート処理や生成結果まで同じとは限らない。

## メモリ

測定後のvmmap summaryによるスナップショット。単位はvmmapのG表示。

| 指標 | 現行oMLX | DS4 MTPなし | DS4 MTPあり |
|---|---:|---:|---:|
| Physical footprint | 76.0G | 8.4G | 8.6G |
| Footprint peak | 83.7G | 8.4G | 8.7G |
| TOTAL resident | 78.7G | 30.5G | 29.8G |
| Mapped-file clean resident | 1.5G | 22.0G | 21.0G |

**76Gから8Gで済むようになった、という意味ではない。** DwarfStarはmain+PLE計99.6Gのvirtual file mappingを持ち、clean file-backed pagesはfootprintと別に扱われる。起動ログ自体もresident model69.73GiB + KV/buffersを含め79.72GiB plannedと報告する。

この限られた負荷の測定時点でresident量は低いが、繰り返し語の長文・少数promptしか試しておらず、必要なexpertやページの範囲が偏る。未参照ページ、OS側のfile cache、再読込、将来のworking setを含めた必要RAM量や他プロセスに返る容量は確定できない。context上限もbaseline114688とDS4 65536で異なる。**実運用で10GiB以上削減という条件は未判定**。

全構成で測定中のSwapouts増分は0。既存swapは存在し、初期量も同じではない。baseline / DS4 MTPなし / MTPありのPageouts増分は2730 / 832 / 752、swap開始量は約7698 / 12138 / 11248MiB。ゼロswap環境とは呼ばない。

## 復帰・後片付け

MTP初回準備では、oMLX CLIが停止成功を返した後も孤立した`omlx-server`が18125を保持。ガードで測定前に拒否したため、その試行に性能結果はない。原因はアプリのcontrol socket不在時にCLI stopが成功を返す経路。失敗時の復帰記録は`failed-start/`へ保存。

再試行は18125の唯一のlistener・PPID=1・comm=omlx-serverを照合し、そのPIDだけSIGTERMしてから開始。インストール済みoMLXのコードは改変していない。

全3構成の実験プロセスは終了、18136は閉鎖。本番oMLX18125はhealthy、defaultは元の`Qwen3.8-Flash-Next-oQ4e-mtp`。Chronovisor11サービスは全てenabledへ復帰し、3つの本番設定ファイルのSHA256は実験前後で一致。

実験用baseline領域約4.9GiB（KV cache・隔離設定・ログ・元モデルへのsymlink）を、本番参照とopen fileがないことを確認して削除。通常の削除であり、その一時キャッシュ自体は復元対象外だが再生成可能。元の本番モデルは保持を確認済み。
候補重み106.88GB（約99.54GiB）と専用runtime cloneは、切替判断用に保持。重みの削除/保持質問には回答待ちであり、勝手に削除していない。証跡は全て保持。

検証: `ruff format --check`、`ruff check`、runner self-check成功。self-checkはSSE usage/出力解析・error拒否・孤立プロセス識別と想定外listenerの拒否を確認。結果JSONの再集計でも全3構成の15リクエストのmessages/入力・出力token数一致、cold cache=0、針/follow-up全成功、品質30/32、本番hashとサービス復帰を確認。実credentialの既知パターンは証跡から検出されず、保存された「secret」は合成針テスト値。

## 証跡と再実行

- 証跡: `_handoff/evidence/2026-09-06-ds4-qwen38-benchmark/`。リクエスト・結果・ログ・vmmap・artifact hash・本番設定hash・復帰記録を保持。
- ランナー: `scripts/ds4_qwen_benchmark.py`。固定パスのこのMac用。実行すると本番サービスを一時停止する。
- チェック: `.venv/bin/python scripts/ds4_qwen_benchmark.py self-check`
- 各arm: `.venv/bin/python -u scripts/ds4_qwen_benchmark.py baseline` / `ds4` / `ds4-mtp`
- 再実行すると同名証跡を更新するので、既存証跡を別フォルダへ保存してから実行する。
