# 実行記録: レーン別 conservative-veto ポリシー導入と hold 滞留解消

原計画実装者: GPT-5.3
作成: 2026-08-03 / Claude セッション(hold 実測 + コード調査済み)
実行記録更新: 2026-08-04 21:43 JST / Codex documentation writer
状態: **DONE (implementation scope) / VERIFIED WITH BASELINE EXCEPTION**。

ここでの DONE は、Workstreams A/B/C、必要なテスト実装、指定文書の更新が
worktree 上で確認できたことを指す。deployment、push、本番 hold drain は完了条件に
含めない。最終検証は完了し、focused suitesはすべて通過した。isolated-root full suiteは
既存baselineのlarge-function policy違反1件だけ失敗したため、本書はfull suite all-greenを
主張しない。

## 実行チェックリスト

### Workstream A: quorum safety policy v2

- [x] `QUORUM_SAFETY_POLICY_VERSION = 2` と5免除レーンを実装。
- [x] tie-break 後の2対1 mutating 多数派だけをレーン別に分岐し、pair 2対0を維持。
- [x] `ingest_reconciliation`、`decision_lane=None`、未知レーンをfail-closedで維持。
- [x] `False`を`conservative`、`None`を`unclassifiable`として監査し、どちらも
  非免除レーンではvetoにする。
- [x] decision/per-vote監査フィールドを追加し、prompt、生出力、decision payloadを
  永続化しない既存のredaction境界を維持。
- [x] lane-contract case identityをv27へ更新し、5免除レーンとingest vetoの
  6件の決定的policy fixtureをmanifestへ追加。
- [x] quorum policy versionをauthority/hold検証へ束縛し、旧epoch cacheを
  自動的に非再利用化。

### Workstream B: 計測

- [x] `chronovisor hold-report` と `--json` を追加。
- [x] structured-review cacheとmanaged-hold stateを既存の共有ロック規約で読み、
  lane × reason × artifact prefix、日時範囲、active/resolvedを集計。
- [x] hold-reportが入力を変更しないことをテストで固定。
- [x] Dashboard local-consensus summaryをschema v3へ更新し、veto発動数、免除通過数、
  dissent effect内訳、モデル別conservative票率を表示。
- [x] Decision Traceでlane-policy bypassを通常のquorum合意と区別して表示。

### Workstream C: bounded evidence / config hardening

- [x] content-correction classificationを、全文review payloadではなく、完全な
  replacement identity manifestのdigestと総量制限付きreplacement detailを持つschema v2の
  bounded projectionへ変更。
- [x] 巨大な単一mutationと多数replacementの両方について、provenanceを保持しつつ
  router preflight上限内に収めるテストを追加。
- [x] TOML readerを1回の`read_bytes()`による単一snapshotへ変更し、atomic replace
  中もold/newのどちらか一世代だけを読むテストを追加。
- [x] 1回のstructured reviewでinitial authority sealとDecisionRouterが同じ
  `DecisionRouterConfig` snapshotを共有し、後段のlive guardは再読して変更を
  fail-closed検出する構造をテスト。
- [x] repository内にconfig writer pathは確立できなかったため、推測に基づくwriterや
  新しい書き込み方式は追加しない。

### Tests

- [x] Router: 5免除レーン、unclassifiable dissent、ingest/None/未知lane、pair 2対0、
  non-mutating majority、真の3-way no-quorumをカバー。
- [x] Semantic hold: quorum policy epoch差によるcache missをカバー。
- [x] Lane contracts: v27 identity、6 quorum-veto policy cases、manifest hashをカバー。
- [x] CLI/Dashboard/audit summary: read-only reportとveto metricsをカバー。
- [x] Workstream C: bounded classification evidence、single-snapshot TOML、
  same-review config共有をカバー。
- [x] **最終検証完了（baseline exceptionあり）**。
  - 最新の関連9ファイルfocused integration: **535 passed in 240.13s**。
  - authority/adoption/artifact: **115 passed in 44.14s**。
  - post-refactor DecisionRouter: **117 passed in 59.64s**。
  - CLI: **15 passed in 0.56s**。read-only no-lock regression: **1 passed**。
  - stale ingest v1 assertion修正後のtargeted test: **1 passed**。
  - classification aggregate budget regressions: **2 passed**。
  - Ruff check on changed Python: passed。`node --check app.js`: passed。
    `git diff --check`: passed。
  - isolated-root full suite: **3052 passed, 1 skipped, 1 failed in 1645.08s**。
    sole failureは未変更baseline
    `src/chronovisor/lab/recall_answer_eval.py::evaluate_answer_episodes`
    **521 > 515**による`tests/test_large_function_policy.py`。同inventoryには
    `validate_answer_outcome_artifact` **610 > 606**も残る。したがってfull suiteは
    all-greenではない。
  - Task起因のlarge-function overageは構造的に解消済み:
    `DecisionRouter._decide_locked` **497 <= 506**、CLI `build_parser`
    **308 <= 308**、`dispatch` **496 <= 496**。各focused testは通過。
  - Formatter checkはchanged Python 19ファイルをreformat対象として報告。repositoryの
    changed baselineがformatter-cleanではないためbulk formatは適用せず、Ruff lint通過を
    検証根拠とした。

### Docs

- [x] `docs/architecture.md`: policy v2、5免除、fail-closed境界、監査、hold再評価、
  bounded classification evidence。
- [x] `docs/operations.md`: Dashboard metrics、hold-report、automatic re-evaluation、
  current v2/v27 artifact contract。
- [x] `docs/config.md`: v2/v27 current identity、code-owned allowlist、single snapshot。
- [x] 本handoffを計画書から実行記録へ更新。

## タイムスタンプ付き実行ログ

- 2026-08-03 19:43 JST: 原計画作成。hold実測と変更前のquorum-v1挙動を記録。
- 2026-08-04 20:35 JST: 並行writerのworktreeをread-onlyで点検。A/B/Cの実装差分、
  追加テスト、既存の未関連変更を確認。
- 2026-08-04 20:40 JST: 指定4文書だけを更新。テスト、commit、push、deployment、
  live hold drainは実行せず。
- 2026-08-04 21:43 JST: validation evidenceを反映。focused suites、静的check、
  isolated-root full suiteを記録し、full suiteのsole baseline exceptionを明示。
  Read-only live local command `uv run chronovisor hold-report --json`はexit 0、
  `errors=[]`、total 1352 / active 140 / resolved 1212 / structured 1100 /
  managed 252 / 40 groupsを確認。push、deploy、live drainは未実施。

## 完了根拠

- Routerはpolicy v2と5レーンの集合をコード定数として持ち、tie-break後のveto条件を
  `conservative_veto_fired`として保持したまま、許可レーンだけを`agreed`へ通す。
- Ingestと非許可レーンは従来の
  `mutating_local_majority_vetoed_by_conservative_vote`を維持し、unknown public laneは
  model call前にもlane-contract validationで停止する。
- Decision/hold auditにはveto/bypass/dissentと各valid voteのlabel/effect classだけが
  追加され、Dashboard集計とTraceへ伝播する。
- Hold reportは2つのdurable inputをread-onlyで横断し、quorum-policy epoch mismatchを
  resolved/non-reusableとして可視化する。
- Classification evidenceは完全なmutation本文を複製せず、pre/post hash、byte長、
  完全なreplacement identity manifestのhash、bounded excerpts、diff/context hashを
  保持する。
- Configは1回のbyte readでparseされ、同一reviewのauthorityとrouterが同一objectを
  共有する。後段guardはcurrent authorityを再観測するため、mid-review driftを隠さない。
- 最新の関連9ファイルfocused integrationは**535 passed in 240.13s**。補助focused suitesは
  authority/adoption/artifact **115 passed in 44.14s**、post-refactor DecisionRouter
  **117 passed in 59.64s**、CLI **15 passed in 0.56s**、read-only no-lock regression
  **1 passed**、stale ingest v1 targeted **1 passed**、classification aggregate budget
  regressions **2 passed**。
- Ruff lint、`node --check app.js`、`git diff --check`は通過。Task起因のlarge-function
  overageも`_decide_locked` 497 <= 506、`build_parser` 308 <= 308、`dispatch`
  496 <= 496へ解消し、focused testsが通過した。
- Isolated-root full suiteは**3052 passed, 1 skipped, 1 failed in 1645.08s**。
  sole failureは未変更baselineの`recall_answer_eval.py::evaluate_answer_episodes`
  521 > 515であり、同inventoryには`validate_answer_outcome_artifact` 610 > 606もある。
  したがって検証状態はbaseline exception付きであり、all-tests-greenではない。
- Read-only live local `uv run chronovisor hold-report --json`はexit 0、`errors=[]`、
  total 1352、active 140、resolved 1212、structured 1100、managed 252、40 groups。

## 残存不確実性と運用フォロー

- 旧`router_config_invalid` runtime record 6件は既にrotationされており、今回の
  single-snapshot修正との直接因果は証明できない。
- Config readerは単一snapshotになり、同一reviewのauthority/routerもconfigを共有する。
  ただし、これは観測されたrace候補を閉じるhardeningであり、上記旧6件の原因確定ではない。
- Repository内のconfig writer pathは確立できなかった。したがって、speculativeな
  writerは追加していない。外部operatorはatomic replacementを使う必要がある。
- Rollout/deployとhold drainは運用フォローであり、このtaskはpush禁止のため未実施。
  本番反映後に`chronovisor hold-report`を継続比較し、免除5レーンの旧veto hold減少と、
  ingest/真のno-quorumの再holdを確認する。
- Full suiteのsole failureは本taskで未変更のbaseline large-function policy違反であり、
  repository全体のall-greenは未達。Task起因overageとfocused regressionsは解消済み。
- Formatter checkはchanged Python 19ファイルをreformat対象としたが、changed baselineが
  formatter-cleanではないためbulk formatは行っていない。Ruff lintは通過。
- Push、deploy、live drainは実施していない。

---

## 原計画（履歴、2026-08-03時点）

以下のv1/v26参照と件数は変更前調査の履歴であり、現行挙動の説明でも現在の
operational instructionでもない。

---

## 0. 決定事項(ユーザー承認済み・変更禁止)

1. 低リスク5レーン(`lint_tag_repair`, `recall_auto_apply`, `orphan_link`,
   `metadata_backfill`, `search_label`)は、tie-break 後の 2対1 mutating 多数派を
   2対0 と同格に扱い、conservative veto を **発動しない**。veto 相当の状況は
   監査記録のみ残す。
2. `ingest_reconciliation` は **veto 維持**(現状どおり quarantine)。
3. 上記5レーン以外のすべてのレーン(未知レーン・`decision_lane=None` 含む)も
   **veto 維持**。fail-closed を崩さない。
4. GLM 5.2 / nvidia build API / フロンティアフォールバックは **本計画に含めない**。
   ローカル三者合議の構造・モデル・sealed sampler は一切変更しない。

## 1. 背景(実測値。2026-08-03 時点の `~/.chronovisor`)

`runtime/semantic-holds/structured-review/entries/` に no-quorum hold が 1,017 件。

| quarantine_reason | 件数 |
|---|---|
| `mutating_local_majority_vetoed_by_conservative_vote` | 860 (85%) |
| `local_models_did_not_reach_two_vote_quorum` | 157 (15%) |

veto 860 件のレーン内訳: lint_tag_repair 309 / recall_auto_apply 272 /
ingest_reconciliation 149 / orphan_link 91 / metadata_backfill 23 /
search_label 8 / local_repair 6。

直近 `runtime/local-consensus/audit.jsonl`(decision 128 行)では
agreed 100 / quarantined 28。quarantine 内訳: veto 13、
`structured_request_preflight_failed:input_too_large` 6、
`router_config_invalid` 6、真の no-quorum 3。

要点: 滞留の主因はモデル能力ではなく、**2票が mutation で一致済みなのに
3人目の保守票が拒否権を持つ**変更前 quorum 安全ポリシー(v1)であった。

## 2. 変更前実装の正確な把握(2026-08-03記録)

対象リポジトリ: このリポジトリ。主要箇所:

- `src/chronovisor/decision/decision_router.py`
  - `QUORUM_SAFETY_POLICY_VERSION = 1`(L89 付近)
  - `_winner()`(L2643 付近): valid 票の signature が2票以上で勝者。
  - `route()` 末尾(L4125 付近と L4194–4218):
    - **pair(primary+challenger)が 2対0 一致 → veto 検査なしで `_agreed`**。
    - tie-break 後に 2対1 → `_mutating_majority_has_conservative_veto()` が
      True なら `_quarantined(votes, "mutating_local_majority_vetoed_by_conservative_vote")`。
  - `_mutating_majority_has_conservative_veto()`(L2712 付近):
    勝者票の効果が `_decision_mutates_durable_state(...) is True` のとき、
    反対側 valid 票の効果が `is not True`(= False **または None**)なら veto。
    **None(分類不能)も veto になる**点に注意。
  - `_decision_mutates_durable_state()`(L327 付近): True=永続 mutation /
    False=hold・no-op / None=不明。レーン契約の effect 定義に従う。
- `src/chronovisor/decision/decision_policy.py`: `DECISION_POLICIES`(全29レーン)。
- `src/chronovisor/search/semantic_hold.py`:
  - `_SEMANTIC_REASONS`(L58)に veto 文字列が含まれる。**文字列は変更禁止**。
  - hold は `epoch_sha256` + `authority_sha256` をキーに永続キャッシュされ、
    authority observation に `quorum_safety_policy_version` が含まれる
    (実 hold JSON で `hold.authority.quorum_safety_policy_version: "1"` を確認済み)。
    → **バージョンを上げると authority が変わり、既存 hold はキャッシュミスとして
    次回 drain 時に自動再評価される。移行スクリプトは書かないこと。**
- `src/chronovisor/ops/dashboard.py` L2592 付近: veto 理由の表示ラベル。
- ストア側(コード変更対象ではないが検証に使う):
  `runtime/semantic-holds/structured-review/entries/`,
  `runtime/managed-holds/state.json`(ingest レーンの hold 台帳),
  `runtime/local-consensus/audit.jsonl`。

## 3. Workstream A: レーン別 veto 免除(本丸)

### A-1. ポリシー定数の新設

`decision_router.py` に追加:

```python
# Lanes whose contracts define additive/reversible effects. A tie-break
# 2-1 mutating majority resolves like a 2-0 pair agreement; the dissenting
# conservative vote is recorded as audit evidence, never as a veto.
TIE_BREAK_MUTATING_MAJORITY_LANES = frozenset({
    "lint_tag_repair",
    "recall_auto_apply",
    "orphan_link",
    "metadata_backfill",
    "search_label",
})
QUORUM_SAFETY_POLICY_VERSION = 2  # 1 -> 2
```

制約:

- `QUORUM_SAFETY_POLICY_VERSION` を 1→2 に上げる。免除レーン集合を将来変更する
  場合も必ず同時にバージョンを上げる、という不変条件をコメントで明記する。
- 可能であればレーン集合そのものを quorum 安全ポリシーのハッシュ対象
  (authority observation / signature_policy 等、`quorum_safety_policy_version` が
  現在流れている経路)に含める。経路上バージョン整数しか流せない構造なら、
  上記コメント規約のみで可とする。**どちらにしたかを PR 説明に書くこと。**

### A-2. route() の分岐変更

L4194 付近の tie-break 後ブロックを次の意味に変更する:

```python
winner = self._winner(votes)
if winner is not None:
    veto = self._mutating_majority_has_conservative_veto(
        votes, winner, schema, prompt=prompt, decision_lane=decision_lane
    )
    if veto and decision_lane in TIE_BREAK_MUTATING_MAJORITY_LANES:
        # audit にのみ記録して通す(A-3)
        return finalize(self._agreed(votes, winner, schema))
    if veto:
        return finalize(self._quarantined(
            votes, "mutating_local_majority_vetoed_by_conservative_vote"))
    return finalize(self._agreed(votes, winner, schema))
```

守ること:

- **pair 2対0 経路(L4125 付近)には一切触れない。**
- `decision_lane` が None・空・未知文字列なら必ず veto 側(fail-closed)。
- quarantine 理由文字列・`failure_class` は既存のまま。
  `semantic_hold._SEMANTIC_REASONS` と dashboard が exact match しているため。
- `_mutating_majority_has_conservative_veto` 自体のロジック
  (`is True` / `is not True` の非対称性)は変更しない。

### A-3. 監査フィールドの追加(犯人特定を可能にする)

`kind="decision"` の audit 行(`runtime/local-consensus/audit.jsonl` への書き出し
箇所を `decision_router.py` / 呼び出し側から特定すること)に追加:

- `conservative_veto_fired: bool` — veto 条件が成立したか(免除で通した場合も True)
- `conservative_veto_bypassed_by_lane_policy: bool`
- `dissent_effect_class: "conservative" | "unclassifiable" | null`
  — 反対票の `_decision_mutates_durable_state` 結果(False / None)を区別する。
  860 件のうち「本気の reject」と「分類不能」の比率を初めて計測できるようにする。
- per-vote 配列(既存 votes 概要があればそこに): `model`, `role`, `valid`,
  `decision_label`(スキーマ enum の文字列のみ), `effect_class`("mutating"/"conservative"/"unclassifiable")。
  どのモデルが慢性的保守票かをレーン別に集計可能にする。

**プライバシー制約(絶対)**: prompt・モデル生出力・decision payload 本体を
audit / hold に保存しない(`semantic_hold.py` docstring の既存規約)。
enum ラベルと分類クラスのみ許可。

### A-4. 影響範囲の検証(実装タスク)

1. `decision_authority.py` / 採択アーティファクト検証が
   `quorum_safety_policy_version` を束縛しているか確認する。
   - 束縛している場合: 現採択アーティファクト(`0ef9c98b…`)が v2 で
     無効化されないか、`decision_artifact_replay` 経路で再検証が通るかを確認し、
     必要なら replay を計画に含める。**ここで手を抜くと全レーン fail-closed で
     停止する**ので、ローカルで sleep cycle 1周を回して agreed が出ることを確認。
2. `decision_lane_contract_cases.py`(`LANE_CONTRACT_CASE_VERSION = 26`)に
   veto 挙動を固定しているケースがあるか確認。フィクスチャを追加・変更したら
   バージョンを 27 に上げる。免除5レーンと ingest の両方に
   「2対1 mutating 多数派」ケースを追加すること。
3. `semantic_hold.py` の `_local_consensus_error` は
   `quorum_safety_policy_version >= 1` の int 検証のみ(L434 付近)なので
   v2 で壊れないはずだが、テストで確認。

### A-5. 既存 hold の再評価(コードを書かない)

バージョンアップにより authority_sha256 が変わるため、既存 1,017 件は
convergence / sleep cycle の bounded drain が触った時点で自動再評価される。

- 免除5レーンの veto 品(703件)→ 再実行され、モデルが同じ投票をすれば agreed。
- ingest の veto 品(149件)+ 全レーンの真の no-quorum 品(157件)→ 再実行
  コストを払って同じ hold に戻る。**これは想定内のコスト**。sleep cycle の
  既存予算内で数日かけて捌ける想定。予算を一時的に増やすなら設定で行い、
  コードのループ上限は変更しない。
- `runtime/managed-holds/state.json`(ingest)は `authority_epochs` の変化で
  既存機構が処理する。手動で state.json を書き換えないこと。

## 4. Workstream B: 計測

### B-1. hold レポート CLI

`chronovisor-hold-report`(または既存 ops CLI のサブコマンド)を追加:

- 入力: `runtime/semantic-holds/structured-review/entries/` と
  `runtime/managed-holds/state.json`
- 出力: レーン × quarantine_reason × artifact_sha256(8桁) のクロス集計、
  作成日 min/max、active/resolved 数。JSON と人間可読テキストの両方。
- 読み取り専用。ロック取得は既存の読み取り規約に従う。

### B-2. ダッシュボード

`dashboard.py` の local-consensus 表示に以下を追加:

- veto 発動数 / 免除通過数(`conservative_veto_bypassed_by_lane_policy` 集計)
- dissent_effect_class の内訳
- モデル別 conservative 票率(per-vote 監査から)

表示のみ。ダッシュボードから mutation を起こす導線は作らない(既存方針)。

## 5. Workstream C: 運用系 quarantine 2種の調査・修正

veto とは独立した普通のバグ。別コミットで:

1. `structured_request_preflight_failed:input_too_large:initial`(直近6件)
   - 発生レーンと入力サイズ分布を audit から特定。
   - 対処候補: 上流での evidence packet の決定的縮約、またはより大きい
     コンテキストバケット(32K/64K/96K/112K)の選択条件見直し。
   - `local_structured.py` の固定上限(fail-closed)の意味を変えないこと。
     上限緩和ではなく「上限内に収まる入力を作る」方向で直す。
2. `router_config_invalid:primary, challenger, and tie-break mod...`(直近6件)
   - メッセージ全文と発生条件(モデル入れ替え中のレースか、設定不整合か)を特定。
   - 一時的な設定状態で quarantine を作らないよう、設定解決を読み取り時に
     アトミックにする方向で修正。

## 6. テスト(最低限、これを全部書く)

`tests/test_decision_router.py`:

1. 免除レーン: tie-break 2対1 mutating 多数派 + conservative 反対票 → `agreed`、
   audit に `conservative_veto_fired=True`, `bypassed=True`。
2. 免除レーン: 反対票の効果が None(分類不能)→ 同上 `agreed`、
   `dissent_effect_class="unclassifiable"`。
3. `ingest_reconciliation`: 同条件 → 従来どおり
   `mutating_local_majority_vetoed_by_conservative_vote` で quarantine。
4. `decision_lane=None` / 未知レーン → quarantine(fail-closed)。
5. pair 2対0 一致 → veto 検査が呼ばれない(従来どおり即 agreed)。
6. 免除レーンでも勝者が非 mutating(reject/hold 多数派)なら veto 概念自体が
   適用されず agreed(現行と同じ)。
7. 真の no-quorum(3票バラバラ)→ 従来どおり
   `local_models_did_not_reach_two_vote_quorum`。

`tests/test_semantic_hold.py`:

8. `quorum_safety_policy_version` 変更で authority_sha256 が変わり、
   既存 hold がキャッシュヒットしない(再評価パスに乗る)こと。

`tests/test_decision_lane_contract_cases.py`:

9. 追加フィクスチャがマニフェスト検証(集合一致・ハッシュ)を通ること。

既存テストがすべて緑のまま。とくに adoption / artifact replay 系
(`test_decision_authority.py`)の回帰に注意。

## 7. ロールアウト手順

1. Workstream A + テストを1コミット(または1PR)。C は別コミット。
2. デプロイ前に B-1 レポートで現状スナップショットを保存。
3. デプロイ後、sleep cycle 2〜3周ごとに B-1 を再実行し以下を確認:
   - 免除5レーンの veto 滞留(703件)が減少し、最終的にほぼ 0。
   - agreed へ流れた mutation が下流検証(CAS、read-back、semantic_checks の
     AND マージ)で異常な失敗率を出していない。
   - `content_correction` 系のユーザー訂正イベントが有意に増えていない
     (増えたら誤 mutation が増えた兆候。ロールバック検討)。
4. ロールバック手順: コミット revert(バージョンは 2 のまま 3 に上げて
   veto 復活でもよいが、単純 revert で v1 に戻る方が事故が少ない。
   hold キャッシュはどちらでも整合する)。

## 8. やらないこと(スコープ外・実装禁止)

- GLM 5.2 / nvidia build / クラウド API の追加。フロンティアフォールバックの新設。
- 三者モデル(`maxwell1500/ornith-35b:Q5_K_M`, `gpt-oss:20b`, `gemma4:26b`)や
  sealed sampler(`temperature=0, seed=0`)の変更。
- `ingest_reconciliation` の veto 緩和。
- pair 2対0 経路、`_winner`、`_agreed` の意味変更。
- quarantine 理由文字列のリネーム。
- hold の手動削除・`state.json` の手動移行スクリプト。
- 真の no-quorum 157 件への対処(将来の別計画)。

## 9. ドキュメント更新

- `docs/architecture.md` Decision router 節: veto がレーンスコープになったこと、
  免除5レーンと根拠(additive/reversible な effect 契約)、v2 への昇格を追記。
- `docs/operations.md`: `chronovisor-hold-report` の使い方と、
  デプロイ後に滞留が自動再評価で捌ける挙動を追記。
