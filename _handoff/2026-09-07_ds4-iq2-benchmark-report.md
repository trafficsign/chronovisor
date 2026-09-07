# DS4 IQ2: 現行4bitとの隔離比較

2026-09-07、Apple M4 Max / 128 GiB / macOS 26.6.2。本番切替なし。

## 結論

**省メモリと生成速度は有望。ただし今回の小テストでは品質低下が見えたため、「4bitとほぼ同品質」は確認できない。** 実ルータの代表ケースは通ったが、前回の本番失敗原因の解消やIngest全体の安定運用を証明する試験ではない。

| 指標 | 現行oMLX・oQ4e 4bit | DS4・IQ2＋MTP |
|---|---:|---:|
| ロード直後のMac全体使用済みメモリ（Activity Monitor表示） | 97.19 GB | 75.85 GB |
| 実資料decode中央値 | 44.60 tok/s | 57.19 tok/s |
| 20,792 token実資料・初回応答開始 | 67.67秒 | 37.41秒 |
| 同資料・初回全体所要時間 | 122.03秒 | 73.14秒 |
| 同資料・再送応答開始 | 2.18秒 | 37.30秒 |
| 日本語を含む短問・厳密出力テスト | 30/32 | 27/32 |
| 実資料のページ形式検証 | 3/3 | 3/3 |
| 本体のルータ／検証器を使う代表ケース | 4/4 | 4/4 |

## 条件

- Chronovisorの11 managed servicesと本番oMLXを一時停止。既存の他アプリは終了していない。ダウンロードとハッシュ検査は測定開始前に終了。
- baseline→IQ2の順に1サーバずつloopback port 18136で実行。両方64K上限、既存4bitはMTP有効、IQ2は`--mtp --mtp-exact-sampling --prefill-chunk 1024`。モデルなしで15秒待機してから観測。
- 現行モデルは`Qwen3.8-Flash-Next-oQ4e-mtp`、oMLX 0.6.4。モデル原本をsymlink参照し、専用の設定／cacheだけを作成。設定の有効値は生集計の`effective_baseline_settings`に保存。元設定の初期ダンプとは異なり、実際の上限は65536。
- IQ2 runtimeは`c64bdffc1028b4c30f1b48a325b3b7f4243a8bde`。IQ2重みrevisionは`672c52bbea7865352c8f0aa766c43939acf0f0d5`、PLEはQ4 releaseの`59a55fb819c82be7b162948282b50bd1a1e290b7`。
- 本体50,343,093,376 B＋PLE32,000,157,440 B、合計82,343,250,816 B。公開SHA-256に一致。本体`31f1e193771a5f3fdaa7af5865e0417513e8e8cac37579d388052ca54c889d4b`、PLE`66db3ab390f4dd5063ecc89cc180f4713898577682347001bf64ab8e328527a1`。
- 性能／短問／実資料生成はtemperature 0、seed 42、thinking off。実資料は既存private snapshotの短・長2件＋長資料再送、最大出力8192。両armの入力messages hashと入力token数は一致。形式検証は本体のページvalidatorであり、内容の正確性の自動採点ではない。

## メモリ

ロード直後の全体表示差は**21.34 GB**。モデルなし時はbaseline24.11 GB、IQ2側25.64 GBと背景条件に1.53 GBの差があるため、小数点以下まで厳密な必要RAM差としない。それでも、この条件で10GB以上削減するという目標には十分な差が出た。

実資料後の観測はbaseline約98.03 GB、IQ2約77.30 GB。ただしUI取得の50秒待機が切れ、代表ケース検証が始まっていた可能性がある。厳密に同一時点の比較ではなく、ロード直後の結果を補う観測として扱う。baseline全チェック後97.82 GB、IQ2全チェック後のUI値はアンロード前に取得できていない。ピーク全体メモリは未測定。

プロセス行はbaseline72.85 GB／DS4 3.25 GBだが、DS4のモデル全体が3GBで動くという意味ではない。wired、RSS、プロセスfootprint、Mac全体使用済みメモリを混同しない。Activity MonitorのGB表記をそのまま使用（搭載128GiBを128GBと表示）。

1秒周期のvm_stat採取はbaseline530件、IQ2 419件。両armともSwapouts増分は0。開始前から約15GBのswapが残っており、ゼロswap環境の試験ではない。OS cache purgeや再起動は行っていない。

## 速度の内訳

| 入力／試行 | 4bit応答開始 / 全体秒 | IQ2応答開始 / 全体秒 |
|---|---:|---:|
| 実資料2,435 tokens | 11.88 / 33.72 | 4.48 / 29.70 |
| 実資料20,792 tokens | 67.67 / 122.03 | 37.41 / 73.14 |
| 同じ実資料を再送 | 2.18 / 52.96 | 37.30 / 73.48 |

全体所要時間の合計は208.71→176.32秒（約15.5%短縮）。出力token数が違うので純粋なエンジン速度比ではない。再送ではoMLXが20,480 tokensのprefix cacheを再利用し、DS4は0。反復が多い用途ではoMLXが有利になりうる。

単純反復文字列中の検索では、4K/16K/32Kの応答開始は4bitが7.14/23.90/47.53秒、IQ2が7.00/26.50/52.99秒。両方3/3正解で、16K/32KではIQ2が約11%遅い。すべてのprefillが速くなるわけではない。日本語の384token生成2回は4bit33.75/36.84、IQ2 55.96/51.13 tok/s。

oMLXログでは後半の代表ケース検証時にメモリ余裕確保のためANE prefill banksの解放が起きている。これは今回の設定でのスタック比較であり、全設定の最高性能探索ではない。

## 品質と互換性

IQ2で追加失敗した3問は、840の15%引きを1176と回答（正解714）、集合の共通要素だけを返す指示に説明を始め上限到達、日本語の新旧順序で「乙」でなく「甲」を選択。両方とも余り算で説明が長く上限到達し、演算子優先順位の計算を誤った。

各問1回の小規模試験であり、30→27を一般能力の9.4%低下と読み替えない。またエンジンと量子化を同時に変更しているため、差をIQ2量子化だけに因果帰属しない。MTP offとの切り分けは未実施。

代表ケースは`ingest_triage`（挨拶のNOOP）、`ingest_reconciliation`、`orphan_link`、`local_repair`各1件。native `OMLXAdapter`→`LLMRuntime`→`DecisionRouter`／`LocalStructuredSession`に通し、実schema validator・plain-choice decoder・semantic effect照合で両方4/4、repair 0。判断処理は本体のpolicyに従いthink=lowなどを使用し、性能用thinking-off試験とは別条件。

本番データの変更は行っていない。adoption gateや常駐制御は隔離し、stream callbackを無効化。**本番Ingestの一連の書込み処理、長時間運転、262K、画像、並行要求、全laneは未検証。前回の本番障害が解消したとは結論しない。**

## 復帰と後片付け

本番oMLXは元モデルをdefaultとしてhealthy。Chronovisorの11サービスも元どおりenabled、実験port18136は停止。config.toml／oMLX settings.json／model_settings.jsonの前後SHA-256は全一致。

比較用baselineディレクトリ約3.2GiB（生成cache、専用設定、モデルへのsymlink）を、open fileなし・モデル原本の参照先を確認して削除。元モデルの原本は保持。候補の重み約82.34GBは保持可否をユーザーに確認中のため暫定保持し、本番には登録していない。

- [公開可能な集計](evidence/2026-09-07-ds4-iq2-benchmark/summary.json)
- private入力／生成結果／console・vm_stat・vmmap証跡: `/Users/trafficsign/projects/sandbox/ds4-iq2-bench-20260907/results`（0700）
- 再実行用runner: `/Users/trafficsign/projects/sandbox/ds4-iq2-bench-20260907/run_bench.py`。既存結果の上書きを拒否する。補助runnerはGitの`a9f3218`から復元しIQ2向けに隔離設定したもの。
- [IQ2モデルカード](https://huggingface.co/ivanfioravanti/Qwen3.8-Flash-Next-DS4-IQ2)
