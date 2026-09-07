# DS4 IQ2 latest: 本番切替の検証

2026-09-07、Apple M4 Max / 128 GiB。ユーザー承認: 合格したら必要変更をpushして本番反映。

## 採用構成

- Chronovisor公開コミット: `aad10d7b58f514756df5bcc055cf8a2f7f404c6e`。変更はモデル識別子、設定例、設定の回帰テストの3ファイルだけ。
- DwarfStar: `ffd85d426313ace6dae805e9f2fb4424d5b427fd` を再ビルド。常駐先は `/Users/trafficsign/.local/share/dwarfstar/runtime`。
- IQ2 revision: `672c52bbea7865352c8f0aa766c43939acf0f0d5`。IQ2本体＋Q4_1 PLE計82,343,250,816 B。前段のハッシュ検証済みファイルを移設し、重複ダウンロードしていない。
- `--metal --ctx 262144 --prefill-chunk 1024 --mtp --mtp-exact-sampling`。loopback `18136`、LaunchAgent `com.trafficsign.dwarfstar`。
- 37生成roleを切替、8補助roleは不変。oMLX `18125` はOrnithとbge-m3を担当。旧4bitはdefault/pinnedを解除し非表示、ロードなし。自動フォールバックではない。
- 本番の `ingest.num_ctx=32768`、`max_num_ctx=262144`、`decision_router.num_ctx=114688` は維持。262K全長の入力を性能試験したという意味ではない。

## 前回の全件失敗の原因

9月6日21:59〜22:12の23バッチ、230試行すべてが `ingest.runtime_context_window_exceeded`。設定を64Kに下げた一方、修復ターン用予約を含む必要枠は73,784〜88,726だった。1289 Bの原文でも74,479枠を要求していた。実際の入力が74K tokensあったわけではない。

したがって当該全件失敗は「DS4が回答を全件誤った」証拠ではなく、こちらのコンテキスト設定と受入試験の不足。今回は262Kを維持し、当時失敗した原文と現在のページ集合で検証した。

## 今回の受入検証

- ビルド成功。Qwen Metal kernelテスト成功、server unit成功、Qwen acceptance 24/24、benchmarkテスト16/16。
- Chronovisor設定回帰: 56 passed、`git diff --check`成功。正しいモデル/revisionのみを受理し、旧4bit・旧DS4 Q4識別子は拒否する。
- 20,792-token実資料、262K設定: 2回ともページ形式検証成功。全体76.72秒 / 76.29秒、出力各2045 tokens。再送のprefix cacheは両方0。
- 隔離したnative IngestのCREATE→UPDATE→NOOP成功。訂正後の値の追加質問は `2026-09-02|14 tok/s` と正答。模擬validatorやauthority迂回は使用していない。
- 現在の7,624ページをコピーし、前回失敗原文のnative triageが成功。これは計画検証で、本番への書込みとは別。
- 本番canary `46a5e5e5`: 同原文を実処理して既存ページを更新、73.52秒、処理/triage失敗なし。初回の読み戻しはSemantic未再開のためsocket missingで失敗した。その記録は改変していない。
- Semantic再開後、対象1ページを既存APIで同期再索引し、読み戻し **1/1成功**。Ornith応答と1024次元embeddingも確認。
- Dashboard/LAN Dashboard/Ingest Drain/Semantic/Rerankerの実行archiveは全て公開コミットに一致。11 managed servicesを元どおり再開。Dashboardはidle、mutation_ready=true、active_failure=null。
- 全サービス再開後、公開archiveから同原文を再投入したjob `2048948c`も23.50秒で完了。ただし `exact_postimages_already_applied` による既適用検出でmodel_calls=0。新たな生成/書込み試験や読み戻し成功件数には加算しない。

## メモリ

Activity Monitorの「使用済みメモリ」で、切替前の通常稼働 **110.83 GB**、IQ2＋全サービス再開後 **100.03 GB**。観測差は **10.80 GB減**。他アプリを含む異時点の値であり、同一負荷の厳密な因果差やピーク保証ではない。

その後の再確認では82.17 GBまで低下（cached 43.96 GB、swap 14.41 GB）。OSの回収・他アプリ・負荷状態を固定していないため、「モデル変更で28.66GB必ず減る」とは解釈しない。短い終了確認区間ではSwapoutsカウンタは11,762,653のまま増分0。

IQ2だけの隔離試験では89.73 GBだったが、補助サービス停止中なので上記と混同しない。DS4プロセス行9.77 GBもモデル全体のメモリ量ではない。262K用GPUバッファは9929.01 MiB、旧64K試験の3146.01 MiBより大きい。以前の約21GB削減値を、そのまま今回の本番構成へ転用しない。

## 限界・既知事項

- 量子化による品質低下は許容するというユーザー判断で採用。旧小テストは4bit 30/32、IQ2 27/32であり、同品質とは主張しない。
- 同じ長文を繰り返す用途はoMLXのprefix cacheが有利だった。すべての処理が高速化するわけではない。
- 上流全テストはgreenではない。generic Metal frontendは29 assertion失敗、Qwen packテストは1 fail/3 errors（無効化されたQ4_1 encoder期待）。今回のQwen推論kernelの成功と区別する。公開GGUFを利用し量子化作成はしていない。
- Semanticは初回起動でaccelerator lease timeoutが発生し、自動再起動後ready/読み戻し成功。切替前からの索引backlog/staleページは別課題で、この切替で全解消したとは扱わない。
- 数時間〜数日の長期運転、全lane、262K実入力、画像、最大並行負荷は未検証。

## 証跡と復帰

- private試験証跡: `/Users/trafficsign/projects/sandbox/ds4-iq2-bench-20260907/results-latest`。原文・ページ内容はGitへ追加しない。
- 本番検証/設定バックアップ: `/Users/trafficsign/.local/share/dwarfstar/backups`。`live-canary.json`と`live-canary-readback-recovery.json`で初回失敗と復旧を分けて記録。
- 復帰用helper: `/Users/trafficsign/.local/share/dwarfstar/cutover_services.py restore`。実行時は現在の処理を安全に止めてから使う。旧設定と旧GitHub pinを戻し、旧4bitを再起動するためのもので、自動fallbackではない。旧4bit原本は復帰用に未ロードで保持。
- 再実験runnerと証跡は保持。今回の一時ページ複製2組（shadow/shadow2のchronovisor-root、合計約313MB）は本番参照・open fileがないことを確認しゴミ箱へ移動済み。復元可能。詳細は `backups/cleanup.json`。本番ページ・モデル原本は削除していない。
