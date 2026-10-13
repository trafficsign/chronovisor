---
task_id: plan_f1af8dd28d254ae0a5fa2c431b1dfb3f
created_at: 2026-10-13T20:52:29+09:00
状態: running
---

# Recall：過去分を Luna で一括ラベル付けし、無人で回り続ける状態にする

実行役：Astra（Codex）。判定の実行は Luna。

## 目的
- recall の教師ラベルは 2026-08-29 から止まっている。過去の会話に Luna で一括ラベルを付けて蒸留の本線を再開し、ユーザーが何もしなくても学習と昇格が回り続ける状態にする。
- 止まったときは、理由つきでユーザーに届くようにする。今回は 6 週間止まっていたのに health は ok で、ユーザーが尋ねるまで誰にも見えなかった。
- 既存 Plan との関係：本線 `2026-08-14_1846_autonomous-recall-offline-distillation.html` と `2026-10-09_1011_recall-compounding-fitness-first.html` は置き換えない。両者の「教師評価の local-only 条件」と、10/9 の承認範囲だけをこの Plan で改める。

## ユーザーの決定（2026-10-13）
1. 過去分の relevance 判定を外部モデルで一括ラベル付けする。判定役は Luna（10/9 と同じ native Luna worker 方式）。
2. Luna の使用量に上限は無い。使えるだけ使い、並列で回す。
3. 外部に出す範囲は制限なし。発話の生データ、文脈、候補ページの本文、sensitivity が high のものを含む。10/9 の「質問生成に限る」承認を、relevance label と P1/R5 の受入まで広げる。
4. 教師は複数でよい。過去分が揃ったら、日々の増分はローカル教師で処理する。
5. 実行中、ユーザーに許可を求めない。ユーザーは別の作業に入っており、確認には答えられない。判断が要る点は自分で決め、理由を実行ログに残して進む。
6. 作ったあと、ユーザーが何もしなくても回り続けること。

決定 5 と 6 から、設定の変更（`~/.chronovisor/config.toml`）と配備もこの Plan の範囲に含める。これは Plan を書いた Claude の解釈で、ユーザーの言葉は「実装してほしい」「とにかく仕上げてくれ」「何もしなくても回るように」である。配備は 10/9 と同じ手順で行う。

## やってはいけないこと（必須）
- ゲートの条件を緩めない。件数や根拠が足りないときは「昇格待ち」とし、下限や閾値を下げて通さない。
- ラベルを verified truth として扱わない。teacher_only のままにする。
- 回答への効き目（反実仮想）に、10/9 より前の記録を使わない。過去分を現在のページで穴埋めしない。
- P0 で止めた旧レーン（auto_apply、旧較正、query hints）を戻さない。
- 本線と 10/9 Plan の禁止事項は、上で改めた 2 点を除いてそのまま守る。
- 生の本文・発話・質問を、リポジトリや Plan に書かない。ここに残すのは件数・hash・検証結果だけにする。生データはこれまでどおり private な bundle に置く。
- 新しい通知の仕組みを作らない。既存の health と `chronovisor_init` を使う。
- 止まったまま黙らない。進めなくなったら `状態` を blocked にし、理由を書く。

## 前提・仮定
2026-10-13 夜に Claude が読み取りだけで確認した事実。

- 実行版は `754f0754`、drift なし、health は ok、page_count は 10,012。最後のコミットは 10/10 09:05。
- `~/.chronovisor/config.toml` の `[recall.distillation]`：`enabled = false`、`capture_enabled = true`、`ox_enabled = false`、`teacher_profile = "deepseek-v4-flash-single-v1"`。下限は teacher_labels 500、per_class 100、probe_pairs 100、counterfactual_pairs 100。`rollout_stages = [5, 25, 100]`、`canary_min_days = 7`。
- status の `recall_distillation`：capture_only、rollout 0%、`hold_reason = blind_repeat_pairs_below_floor`、`last_success_at = 2026-08-29`、teacher_only 2103、verified_truth 0、probe_not_truth 1422、paired_denominator 0、alert false。
- 蒸留ワーカーは sleep サイクル（`ops/sleep_cycle.py:420`）と converge ワーカー（`ops/converge_worker.py:152`）から今も呼ばれている。`_prepare_distillation_chunk` が `if not config.enabled` で即終了している（`recall/recall_distillation.py:15202`）。
- blind repeat のペアを作るのは `_ox_prepare_tasks`（同 `:9098`）。対象は split が test で候補が 2 件以上の rally（同 `:9213-9227`）。
- 過去の発話に当時時点の索引で候補を作る関数は既にある（`candidate_snapshot`、同 `:3536`。`search_historical_index(as_of_us=...)` を使う）。rally の eligibility は、relevance が常に True、answer_utility は回答と exact exposure が揃ったときだけ（同 `:3443-3452`）。
- 全体の health は ingest_liveness・runtime_status・semantic_index だけで決まり、蒸留は入っていない（`ops/health.py:1094`）。
- 10/9 Plan の記録：Raw は 14,956 rally / 87 日。9 月以降の 5,982 件は answer_utility の eligible が 0。8/29 の training snapshot は 505 行で train のみ。旧外部教師の contract は期限切れ。local 教師は generator / judge と同一モデル。

仮定（未確認）。

- 10/9 Plan の言う「独立性」は、Luna を教師にすれば満たせる。
- R6 と R7 は 10/9 Plan の本文では未着手。同日 22:38 のログに R6 の修正があるので、現状は手順 0-4 で確かめる。

## 実行手順

### Phase 0：調査（止まらず、結果を実行ログに書いて進む）
- [x] 0-1 「local 教師と generator/judge は同一モデルで独立性要件未達」が、どの検査を止めているかを特定する。Luna を教師にして解消するか確かめる。Luna は recall_questions の生成役でもあるので、P2b の A〜E 評価が自己評価にならない構成にする（その評価だけ別モデルで判定する、など）。
- [x] 0-2 8/29 の snapshot で probe・validation・locked test が 0 件だった原因を特定する。split の割当か、そこへ進む前に止まったのか。直す必要があれば直す。
- [x] 0-3 一括ラベル付けの件数と所要時間を見積もる。
- [x] 0-4 ラベルが揃ったあと recall の挙動が変わるまでに残っているもの（R6 の student、R7 の shadow / canary）の現状を棚卸しする。

### Phase 1：承認の記録と契約
- [x] 1-1 10/9 Plan と本線 Plan に、上の決定（承認の拡張、local-only 条件の解除）を日付つきで記録する。
- [x] 1-2 Luna を教師とする profile と contract を用意する。既存の single-teacher の経路を使う。期限を設けるなら、切れる前に 4-4 の経路で知らせる。

### Phase 2：過去分の一括ラベル付け
- [ ] 2-1 過去の全 rally に当時時点の索引で候補を作り、Luna で relevance を判定する。native Luna worker を並列で、使えるだけ使う。凍結 snapshot・batch・resume cursor で再開できる形にする。10/9 の生成と同じ検査（ID、件数、hash、単一 writer、atomic 保存）を通す。
- [ ] 2-2 blind repeat（順序を入れ替えた 2 回）と、時系列の train / validation / locked test を、既存の single-teacher gate が受け取れる形で揃える。
- [ ] 2-3 relevant / irrelevant の比を記録する。8 月の較正は正例が 93% で「常に想起」を学習した。同じ偏りが無いことを確かめる。偏っていたら、学習に使う前に対処を決めて記録する。
- [ ] 2-4 数百件を別モデル（ローカル教師。可能ならもう一つ）にも判定させ、Luna との一致率を記録する。P1 の「判定モデルの種類で変わらないこと」の根拠にする。

### Phase 3：本線の再開と日次運転
- [ ] 3-1 `recall.distillation.enabled = true` にして本線を再開する。R5 の残り（blind repeat、反実仮想、validation、locked test、壊した対照が通らないこと）を受入まで進める。反実仮想が下限に届かない間は、その理由を「データ待ち」として表示する。
- [ ] 3-2 日々の増分をローカル教師で処理する経路を動かす。2-4 の一致率が低くてローカルに任せられないときは、日次も Luna で回し、その判断を記録する。
- [ ] 3-3 R6（student 候補）と R7（shadow → canary 5 / 25 / 100、自動ロールバック）を、本線の設計どおり無人で進むところまで実装して配備する。昇格するかどうかは既存のゲートだけで決める。

### Phase 4：止まったら知らせる
- [ ] 4-1 蒸留が進んでいないことを、health の全体判定に入れる（`ops/health.py:1094`）。
- [ ] 4-2 止まった理由を「データ待ち」と「人の判断待ち」に分けて出す。判断待ち（教師の契約切れ、承認の範囲、enabled=false、教師がいない）は、それと分かる理由にする。今回の表示は `blind_repeat_pairs_below_floor` で、真因（教師がいない）が見えなかった。
- [ ] 4-3 判断待ちを `chronovisor_init` の結果に出す。セッション開始時に必ず読まれる場所である。
- [ ] 4-4 期限のあるもの（教師の contract など）は、切れる前に 4-3 の経路で知らせる。
- [ ] 4-5 同じ「止まっているのに ok」が他に無いか調べる。分かっている候補は、search_labels の待ち行列 68,241 件（ほぼ `pending_review`）、lint_repair 444 件、session_sweeper の attention。原因と扱いを記録し、直せるものは直す。

### Phase 5：別件の修理
- [x] 5-1 session_sweeper：2 セッションで `TypeError: Object of type bytes is not JSON serializable`。
- [x] 5-2 ingest：`apple-watch-band-material-tradeoffs/bandwork-leather-band-experience.md` が `missing_create_frontmatter` で失敗している（2026-10-13 19:45、1/3 回目）。再試行で直っていなければ原因を見る。

## 実行中更新ルール
- 実行手順のチェックは、各手順の完了直後に更新する（まとめて更新しない）。
- チェックを更新したら、同時に `実行ログ` へ時刻付きで追記する。
- 中断時は `状態` を `blocked` にし、理由を `ブロッカー` に記入する。
- 着手したら `状態` を `running` にする。

## 完了条件
- 過去分のラベルが揃い、single-teacher gate の blind repeat と時系列分割の項目が、条件を緩めずに通っている。
- `enabled = true` にしたあと、ユーザーの操作なしで 7 日、日次のラベルが増え続けている。
- 教師をわざと使えなくしたとき、全体の health が alert になり、`chronovisor_init` に「人の判断待ち」として出る。
- R6 と R7 が無人で進む経路が配備されている。昇格したかどうかは問わない（ゲートが決める）。昇格していないときは、理由が「データ待ち」か「人の判断待ち」かで表示されている。
- この Plan の状態と実態が一致している。

7 日の確認は 1 回の作業では終わらない。既存の毎朝 09:15 の確認に載せ、結果を実行ログに追記する。7 日が満ちるまで `状態` は running のままにする。

## 検証手順
- status の `recall_distillation.last_success_at` が毎日進み、教師ラベルの件数が増えていること。
- gate の出力の reasons を読み、何で止まり、何が通ったかを記録すること。
- 故障注入：教師を無効にして health と `chronovisor_init` の表示を確かめ、元に戻す。
- テストと CI は既存の基準（Quality、Python 3.14）を通す。
- 配備後、runtime の drift が false であること。

## ブロッカー（発生時のみ）
- なし

## 受け渡しメモ（次担当へ）
- ユーザーは「前のも、作って結局動いていない。これを何回も繰り返している」と言っている。今回の成否は、実装とテストが通ることではなく、無人で回り続けることで決まる。
- 「前提・仮定」は 2026-10-13 夜の読み取り結果。Claude はリポジトリを変更していない（この Plan ファイルの追加を除く）。
- 報告は、件数・hash・検証結果・判断とその理由を、この Plan の実行ログに書く。

## 実行ログ
- 2026-10-13 20:52 JST　Plan 作成（Claude）。ユーザーの口頭の承認と指示にもとづき、状態を ready とした。
- 2026-10-13 20:57 JST　Astra が着手。承認範囲・禁止事項・完了条件を全文確認。7 日の無人継続確認が終わるまで running を維持する。
- 2026-10-13 21:03 JST　5-2 完了。対象 ingest は同一ジョブの targeted repair で回復済み。バッチ 6/6 成功、失敗・deferred・pending 各 0。canonical frontmatter parse と registry hash 一致を確認（3,880 bytes、SHA256 6bdd6658d16ff0c49674141081aa81e719613dd93d8d78e166565134dec825f7）。不要な再試行・ソース変更は行わない。
- 2026-10-13 21:05 JST　5-1 完了。Hermes SQLite の BLOB を JSON 化する境界で bytes 例外を再現し、内容を損失なくタグ付き hex へ直列化。synthetic 回帰を含む関連 64 tests、Ruff、mypy 通過。実 2 セッションの 301 records を保存し、pending=0 / sweeper=ok / errors=0、再実行は no_pending_sessions を確認。配備は統合後に行う。
- 2026-10-13 21:05 JST　1-1 完了。10/9 と本線の実行ログに承認の拡張と local-only 解除を記録。既存の禁止事項・教師ラベルの非正解扱い・昇格条件は維持した。
- 2026-10-13 21:09 JST　0-1 完了。local triad は 3 教師の model digest 不一致を要求し、現在の同一モデル 3 route は教師無しになる。Luna は独立三教師へ偽装せず native-luna-single-v1 の単教師契約と blind repeat で評価する。CF の generator/judge 独立性は別条件として維持。P2b A〜E は Luna 生成物の自己採点を避け、別 identity の local judge を使う。独立実クエリ不足なら採用待ち。
- 2026-10-13 21:09 JST　0-2 完了。8/29 snapshot は 505 行・215 rally・2 session cluster、7/18〜19 の train 群だけ。固定 split 自体は存在し、test 群を処理していないため probe / validation / locked test が 0。既存 train を test に付け替えず、全時系列群を凍結し未処理群から揃える。
- 2026-10-13 21:09 JST　0-4 完了。R6 の train/validation/test→tiny student→candidate、R7 の shadow/canary/rollback は既に実装済み。enabled=false と教師不在で本線が走らず candidate=0。native profile の契約・materialization・gate・workerへの一貫接続、日次の実行主体を追加し、既存 gate のみで進行させる。

- 2026-10-13 21:40 JST　0-3 完了。現行 Raw から 15,800 rally を確認。候補は各 rally の上位 3 件＋末尾 1 件を relevance 判定対象とし、owner 上限 63,200 件、blind repeat は test 100 rally × 2 候補 × 2 順序の 400 件を予定。候補 0 件・payload 不成立・embargo は件数を別記し、正解扱い／学習件数へ加算しない。候補履歴の追加 7,750 件は 16 並列・100 rally 約 86 秒の実測から約 2 時間、Luna は 10 並列で数時間を見込むが relevance の実処理速度は未測定のため初回バッチ後に補正する。凍結済み履歴を A/B の不変範囲に分け、A の判定と B の候補生成を並行する。
- 2026-10-13 21:40 JST　日次実行の監査結果を反映中。既存 09:15 automation は metadata-only で教師を実行しなかったため、未完バッチの再開・新規入力の凍結・native Luna 判定・import・実行ログ更新まで行う。local triad は同一 model digest のままで有効な独立三教師を作れず、300 件の別モデル一致率も未測定なので、検証が揃うまで日次も native Luna を継続する。これは local 教師が使えるという未検証の前提で再停止させないためであり、独立性検査は維持する。
- 2026-10-13 22:01 JST　1-2 完了。native-luna-single-v1 の closed identity v2 / request v9、実 prompt/schema hash、native worker assignment/run と出力 hash の契約・importerを実装。永久契約を採用し、明示的な enabled/native_luna_enabled を停止スイッチとする。契約 ID 0b518d029e368fd356951eebd10bdb1ea0139660b78c29d98a2fd28de36d22a4、cohort SHA b4a02fbb41a48e0c9d04ce37765e702170e2725d26a65a270dd2ef1642cce5b5。旧 DeepSeek の期限条件は変更なし。native 統合・hash 改竄・Oct9 境界の検証通過。新 split 6b335e8a… は 15,800 rally（train 7,998 / validation 2,057 / test 5,681 / embargo 64）を同一 session 群のまま凍結し、旧教師の train を転用しない。

- 2026-10-13 22:35 JST　一括 A を凍結し native Luna 10 worker の実判定・保存を開始。A は 8,391 snapshot の不変範囲から 32,332 tasks（8,114 rally）、input 超過／本文不足 328 件を別記。manifest SHA256 2fa768d2d9960209738bd1f9b43ca48b555c3db3b8d71f4aa1773e6e3f96a0e9。800 件保存時点で relevant 269 / irrelevant 450 / uncertain 81。残りの候補履歴は再開中。SQLite の export/capture 競合は読取終了後に tail 同期し復帰、途中成果を保持した。
- 2026-10-13 22:35 JST　既存 09:15 automation recall-p2b を ACTIVE・同じ thread/RRULE のまま更新。未完バッチを再開し、翌日の capture・凍結・Luna 判定・import まで実行する。prompt SHA256 42f77233cf379bf3eeabcf4bac30bdd1dbe77fd386618f97dcb174c32a911966。日次 runner は private helper に依存せず、capture を内蔵し stale PID/lock を検知・再開する。7 日継続はまだ未達。
- 2026-10-13 22:35 JST　wheel/sdist build 成功。隔離 full suite は 6,777 passed / 2 failed。整数 candidate ID の既存契約を cache が誤拒否した不具合と、health の明示 root 参照・hermetic fixture を修正し、関連 47 tests 通過。実 TTY で 8 KiB 超の JSON を送信する回帰も通過。input/output hash、gate 閾値は変更なし。
