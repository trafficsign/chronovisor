# DwarfStar: 実資料リプレイとメモリ判定の訂正

測定日: 2026-09-06。M4 Max / 128 GiB。本番切替は未実施。

## 判定

実資料4件と最大資料の再送1件では、合計所要時間が329.00秒から249.43秒へ約24%短縮した。一方、**wiredが同程度だから省メモリではない、という会話中の結論は撤回する**。wiredは全使用済みメモリではない。プロセスresident/footprintの差をそのままRAM削減量とした先の説明も正しくない。

ユーザーはアクティビティモニタでDwarfStar側が約10GB低いと観察した。これは今回のwired比較では否定できない。両armの同条件・同時点のアクティビティモニタ記録はなく、**10GB以上削減の確定値はまだない**。採用候補としては有望だが、本番品質同等と移行完了は未確認。

## 条件と速度

既存wiki実資料をサイズ別に4件スナップショットし、productionのページ生成prompt/validatorを使用。資料原文と出力はgit外のprivate領域に保持し、外部推論には送っていない。wikiやDBは変更していない。完全な本番パイプライン再現ではなく、生成要求の隔離リプレイである。

runtime・重みrevisionは[初回報告](2026-09-06_ds4-qwen38-benchmark-report.md)と同じ。thinking off、temperature 0、seed 42、最大出力2048 tokens。入力messagesのhashとtoken数は両armで一致。各ケース1回、逐次実行。baseline context上限114688、DS4 65536に対して実入力最大20792 tokens。OS cache purgeは権限エラーで失敗し、両方ともcache-coldではない。

| 入力tokens | oMLX 所要秒 | DS4+MTP 所要秒 | oMLX TTFT秒 | DS4 TTFT秒 |
|---:|---:|---:|---:|---:|
| 2435 | 35.23 | 26.34 | 13.76 | 4.81 |
| 5179 | 66.92 | 44.64 | 21.87 | 8.94 |
| 10641 | 71.18 | 48.48 | 36.97 | 18.27 |
| 20792 | 106.88 | 67.03 | 59.68 | 35.08 |
| 20792 再送 | 48.72 | 62.94 | 2.18 | 31.33 |

初回4件の合計は280.22→186.49秒。decode推計中央値44.52→56.03 tok/s。出力長は同一でないため所要時間差を純粋なエンジン速度比としない。再送ではbaseline cached_tokens=20480、DS4=0でbaselineが速い。初回4件はいずれも両arm cached_tokens=0。

## メモリ: 指標を分離する

保存したvm_statから下記の**探索的な代理指標**を再計算した。これはアクティビティモニタの「使用済みメモリ」と一致を検証した式ではなく、必要RAMや解放可能容量でもない。

`(Anonymous pages - Pages purgeable + Pages wired down + Pages occupied by compressor) × page_size / 2^30`

| 処理後snapshot | baseline GiB | DS4+MTP GiB | 差 GiB |
|---|---:|---:|---:|
| 資料1 | 101.31 | 98.70 | 2.60 |
| 資料2 | 101.22 | 98.80 | 2.42 |
| 資料3 | 102.27 | 99.26 | 3.00 |
| 資料4 | 105.62 | 99.55 | 6.07 |
| 資料4再送 | 106.31 | 99.32 | 6.99 |

DS4でwiredが高めでも、匿名ページや圧縮メモリを合わせたこの値は低い。**wiredだけで全体の増減を判定したことが誤り**。ただし順次測定でモデルなし時点の値も23.89対22.94 GiBと異なり、背景アプリ・圧縮・cacheの寄与を完全には分離していない。代理指標の2–7GiBを「Activity Monitorで実測した削減量」と言い換えない。

復帰後のActivity Monitorを直接確認した時点では、oMLX行78.98GB、画面下の使用済みメモリ109.96GB、wired83.74GB、アプリメモリ12.91GB、圧縮12.01GBと表示された。表示内訳の和は総量と完全には一致せず、近接したvm_stat採取とも値が動いている。UI更新時差を含む可能性があり、この単一観測で代理式を校正したとは扱わない。復帰後は補助モデルやChronovisorも動くため、隔離DS4測定と直接差し引けない。

本実資料runのSwapouts増分はbaseline 0.382GiB、DS4 0。初期swap/圧縮量は異なるため、恒常的な優位やゼロswap環境とはしない。プロセスfootprint、wired、使用済みメモリ、swapは別指標として残す。

確定に必要な追加確認は、同じ他サービス状態・同じ入力・同じsettle時間で、両armのActivity Monitor下部「使用済みメモリ」をvm_statとともに直接採取すること。今回は再度のサービス停止・追加負荷を行わず、保存値の診断まで。

## 品質判定の限界

ページ形式検証はbaseline 3/5、DS4 4/5。失敗はいずれも2048tokens上限到達による終端マーカー欠落。productionの8192tokens上限とは異なる。形式検証は内容の正確性・網羅性の評価ではない。

別途19laneのcanonical fixtureを旧corpus runnerでsingle-shot実行したが、**保存JSONのschema/effect集計は本番採否に使用しない**。既存runnerには以下の問題がある。

- `validate_json`が返す違反listを無視しており、schema成功に偽陽性がある。DS4 orphan_link出力で必要キー欠落とenum違反を確認した。
- ingest_reconciliationはproductionでplain-choice selectionを受け取りhost側でmaterializeするが、runnerは変換promptに元schemaを送りJSONとしてparseする。この失敗をモデル能力低下と判定できない。
- productionのrepair経路を再現していない。single-shot失敗率と本番最終失敗率は別物。

従って「品質が同等」「DS4の本番成功率が低い」のどちらも未確定。production adapterに合わせた対象lane再検証は未実施。既存生結果は誤集計を含む観測記録として改変せず保存した。

## 復帰・後片付け・証跡

両arm終了時に本番oMLX healthy、元のdefaultモデルへ復帰。Chronovisorの11サービスを復帰し、config.toml / oMLX settings.json / model_settings.jsonの実験前後hashは一致。候補ポート18136は停止済み。

実験専用baseline領域約3.2GiBを、参照・open fileなしを確認して削除した。一時cacheは再生成可能。元のモデルは削除していない。候補重み・runtime・private実資料スナップショットと出力は再検証用に保持。本番切替、モデル削除、pushはしていない。

- 生証跡: `evidence/2026-09-06-ds4-real-workload/`
- private資料と出力: `/Users/trafficsign/.omlx/experiments/ds4-qwen38-q4/real-workload-private/`（git外）
- runner: `scripts/ds4_real_workload.py`。実行は本番サービスを一時停止する。既存結果があれば再実行を拒否する。
- metadataのcold/warmラベルはcaller-definedへ訂正。生計測値は変更していない。
