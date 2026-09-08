# 日本語Recall rerankerのMac実測 — 2026-09-08

公開済みの3候補は、M4 Max・128GBで追加学習・CUDA・外部推論APIなしに動作した。
今回の小規模診断では、既存BGEを置き換える精度上の根拠は得られなかった。
速度重視の次候補は日本語xsmall。MemReranker-4Bは長い入力で重く、LycheeMem v1は日本語の選別で誤りが多かった。

## 結果

各問題の候補は6件。正解あり20問、正解なし4問の合成診断。
「必要資料を上位3件に収容」は、複数資料が必要な問題で両方を含むことを要求する。

| モデル | 正解を1位にした問題 | 必要資料を上位3件に収容 | 短文中央値 | 短文p95 | 6候補×512トークン中央値 |
|---|---:|---:|---:|---:|---:|
| BGE-v2-m3（比較基準） | 20/20 | 20/20 | 34.8ms | 89.8ms | 201.0ms |
| japanese-reranker-xsmall-v2 | 20/20 | 19/20 | 11.6ms | 19.9ms | 19.9ms |
| LycheeMem Reranker v1 | 16/20 | 20/20 | 83.9ms | 121.1ms | 430.6ms |
| MemReranker-4B MLX量子化版 | 20/20 | 20/20 | 436.7ms | 545.8ms | 1838.9ms |

数値は最終実行の値。短文は24問を2周した48リクエスト、長文は2回warmup後の3リクエスト。
精度の分母は20個の独立した設問であり、繰り返し実行を問題数に加算していない。
初回の短文実行でも全モデルの全ランキングが同一だった。初回の短文中央値は順に38.6/12.0/85.8/431.4ms。

**これは常駐モデルの選別処理時間。** Tokenize、データ転送、モデル計算、結果取得・同期を含む。
モデル起動、候補検索、サービス待ち行列、回答生成は含まない。実サービス全体の待ち時間ではない。
長文負荷は同一の日本語の繰り返し文を各モデルの512トークン上限まで切り詰めた速度専用試験で、長文の回答精度試験ではない。

## 個別の観察

- 日本語xsmallは、可用性と関連性の両方の根拠が必要な`me02`で、片方を4位に落とした。1位正解だけを見ればこの不足を見逃す。
- LycheeMemは`td01`（判断方針）、`tu02`（現行方式）、`cr04`（省略された対象の過去決定）、`ts01`（話題転換）の4問で1位を誤った。`tu02`では現行方式より初期方式を優先した。
- MemRerankerのNDCG@3は0.996、BGEは0.988、日本語xsmallは0.981、LycheeMemは0.920。小さく簡単な集合なので、MemRerankerとBGEの差を一般的な優劣とは解釈しない。
- 文脈を外す追加試験は`coreference`/`topic_shift`の8問だけに実施。文脈ありでBGEは1問、日本語xsmallは2問、MemRerankerは1問改善。LycheeMemは2問改善した一方、話題転換の1問で悪化した。主集計と追加試験の時間は分離した。
- 正解なし4問は、未調整のスコアを保存しただけで、棄却精度を評価していない。MemRerankerでも正解なし候補の最大スコアが0.562となる例があり、単純な0.5閾値を採用する根拠にはならない。モデル間で生スコアを直接比較しない。

## 実行条件と正しい読み込み

- ハードウェア: Apple M4 Max、128GiB。OS・Python・依存バージョンは各結果JSONに記録。
- BGEと日本語xsmall: 既存の`chronovisor.core.reranker._transformer_scores`を直接使用。MPS、float32、batch size 6、max length 512。本番の配信ルート・設定を呼ばず、独立プロセスで計算した。
- BGEの通常の設定上限は384だが比較では512に統一した。短文問題は全モデルで切り詰め0件。長文試験は512に統一した比較条件であり、通常設定の本番ベンチマークではない。
- LycheeMem v1: Qwen3-Reranker-0.6Bに公開済みPEFTアダプターを適用したSequenceClassificationモデル。MPS/float32。学習済み`score.weight`が保存重みと完全一致することを確認し、ランダムな分類ヘッドのまま測定しない。
- MemReranker: 第三者のMLX変換は名称が4bitだが、実際は4bit/8bit混在。重み3,033,491,348bytes。Apple GPUを使用。
- MemRerankerの公開configは`rope_parameters.rope_theta`、MLX LM 0.31.3はトップレベル`rope_theta`を要求したため、同値の1000000をメモリ上で渡した。ダウンロード済みconfigや本番ファイルは変更していない。
- MemRerankerは公式方式のyes/no logitsで採点し、文章生成しない。明示的なpadding/causal maskと最終位置だけの出力計算を用いた。異なる長さの2入力で、通常の非paddingモデル呼び出しとの確率差は0だった。
- ネットワークは公開重みの取得のみ。実測プロセスはHugging Face offline設定で動かし、合成質問・候補を外部推論サービスへ送っていない。

## 入力と評価の限界

24問は、話題だけが近い候補、時間による更新、省略参照、話題転換、複数資料、根拠なしの6カテゴリ各4問。
候補は今回作成した短い合成文で、実際のChronovisorページを検索して得た候補ではない。正例は明示的な属性語や「旧／最新」などで区別できる問題が多い。
候補内に正解がある場合の選別を測っており、候補検索の漏れ、長文の推論、日本語一般での精度、完全なRecall品質は判定していない。

別エージェントがラベルを査読し、`tu03/cr01/cr04/ts01`の曖昧さを最初の採点前に修正した。
最終実行前には`cr02/cr03`の説明文にある候補IDの誤記だけを修正した。採点に使う質問・文脈・候補・正解ラベルは初回から不変であることを照合した。
初回のソースとfixtureは当時のSHA256との一致を確認し、`initial-short-run/`に保存した。

この結果で本番採用は決めない。次に比較するなら既存BGEと日本語xsmallを、実際の失敗会話と同じ候補集合で比較する。
特に、既存BGEが短文診断で全問正解だったことは、本番の候補取得や会話入力まで正常である証拠にはならない。

## 保存物

- `models.json`: モデルの固定revision、ローカルsnapshotパス、取得時間。
- `protocol.json`: 入力・runner・本番設定のSHA256、評価範囲。
- `bge.json` / `japanese.json` / `lychee.json` / `mem.json`: 候補ごとのスコア、順位、時間、token数、環境、読み込み検証。
- 同名`.log`: 各プロセスの出力。Lycheeのbase分類ヘッド初期化警告は、その後の学習済みヘッド一致確認で解消したことをJSONにも記録。
- `validation.json`: テスト・lint、初回と最終の順位一致、本番設定が変わっていないことの照合。
- `initial-short-run/`: 最初の短文実測と、対応する凍結ソース。
- `../../../scripts/benchmark_recall_rerankers.py`: 再実行用スクリプト（repository rootから実行）。
- `../../../tests/fixtures/recall_reranker_japanese.json`: 保守用の合成診断データ。

## 再実行

モデルは通常のHugging Faceキャッシュに保持した。追加したPEFT等は実験専用ディレクトリに分離し、本番venvにはインストールしていない。
再実行するときはrepository rootで、実験用の依存を用意する。

```sh
uv pip install --python .venv/bin/python --target /tmp/chronovisor-reranker-deps --no-deps peft==0.20.0 accelerate==1.14.0 psutil==7.2.2
PYTHONPATH=/tmp/chronovisor-reranker-deps .venv/bin/python scripts/benchmark_recall_rerankers.py --model mem --models _handoff/evidence/2026-09-08-reranker-comparison/models.json --output /tmp/chronovisor-mem-reranker-recheck.json
```

`--model`は`bge`、`japanese`、`lychee`、`mem`から選択。１プロセスずつ実行する。
本番の自動注入設定、モデルの採用、サービス再起動は実施していない。

## 公開元

- [MemReranker作者のモデル](https://huggingface.co/IAAR-Shanghai/MemReranker-4B)
- [今回使用した第三者MLX変換](https://huggingface.co/nisavid/MemReranker-4B-OptiQ-4bit)
- [LycheeMem v1アダプター](https://huggingface.co/fuhao23/reranker_v1)
- [LycheeMemのベースモデル](https://huggingface.co/Qwen/Qwen3-Reranker-0.6B)
- [日本語xsmall](https://huggingface.co/hotchpotch/japanese-reranker-xsmall-v2)
- [比較基準BGE](https://huggingface.co/BAAI/bge-reranker-v2-m3)
