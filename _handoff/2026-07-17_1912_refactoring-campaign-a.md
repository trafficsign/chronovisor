---
task_id: repo_refactoring_campaign_a_20260717
created_at: 2026-07-17T19:12:34+09:00
状態: complete
baseline_commit: 5d54bc7b744e53a758a7e254e9fef4a3c42a7671
branch: codex/refactor-campaign-a-safety
---

# Repo-wide refactoring Campaign A: safety baseline

## 目的

後続の削除・共通化・ingest 分割を、行数や static grep ではなく現在の契約から判定できるようにする。Campaign A は production logic を意図的に変更せず、再現可能な inventory、characterization tests、runtime / authority baseline を追加する。

## 保護する契約

- persistent JSON / JSONL bytes、digest、path、schema、authority observation
- adopted artifact と lane / case / structured policy manifest
- fail-open / fail-closed、timeout、lock、fsync、readback、CAS
- hook event / JSON I/O、CLI exit code、runtime source
- current soak pin と継続中の burn-in output
- user-owned `~/.wiki`、既存 worktree の未追跡ファイル

## Baseline snapshot

### Git / tests

| Item | Baseline |
|---|---|
| HEAD / origin/main | `5d54bc7b744e53a758a7e254e9fef4a3c42a7671` |
| Isolated worktree | `/Users/trafficsign/projects/personal/llm-wiki-mcp-refactor-a` |
| Branch | `codex/refactor-campaign-a-safety` |
| Original worktree | user-owned untracked `_handoff/2026-06-11_0042_recall-redesign.md` and `logs/`; untouched |
| Baseline full suite | 2,328 passed / 1 failed in 928.87s。casefold result identityの既存bugを検出 |
| New targeted characterization | 16 passed |
| Final repository suite | 2,345 passed in 882.22s |

### Source inventory

再生成コマンド:

```bash
uv run python scripts/refactor_inventory.py --output /tmp/llm-wiki-refactor-inventory.json
```

| Metric | Count |
|---|---:|
| Python modules under `src/llm_wiki_mcp` | 115 |
| Python lines | 116,779 |
| Function / method definitions | 2,875 |
| Functions >= 200 lines | 63 |
| Functions >= 300 lines | 33 |

最大候補は `self_heal._handle_packet_unlocked` 908 行、`content_correction._process_frontier_item` 888 行、`orphan_link.run_autonomous` 793 行、`ingest._review_and_apply_ingest_operations` 786 行、`search_eval.review_label_queue_with_frontier` 728 行、`orchestrator.run_pending_ingest` 725 行。300 行未満を完了条件にせず、churn、persistent mutation、fan-in、test seam を加えて Campaign F の上位 5 件を選ぶ。

### Candidate signal inventory

以下は削除・統合の許可数ではなく、契約分類前の候補数である。

| Signal | Candidates | 判定上の注意 |
|---|---:|---|
| atomic helper name | 18 | backup、disk floor、readback、permission を比較する |
| replace call | 97 | atomic publication、CAS、単純 rename を分離する |
| JSONL append | 23 | caller lock、fsync、torn-tail、telemetry を分離する |
| JSON dump + SHA-256 | 40 | dump options と producer / consumer を分類する |
| flock in function | 40 | EX / SH / NB、reentrant、fork safety を分類する |

完了条件は candidate zero ではなく、unclassified exact duplicate zero。差分のある実装は allowlist と parity test を持つ。

### Ingest seam inventory

| Metric | Count |
|---|---:|
| `ingest.<symbol>` attribute symbols | 78 |
| Attribute references | 390 |
| Monkeypatch target symbols | 34 |
| Monkeypatch calls | 302 |

この数字により「外部 4 symbols だけ」という前提は棄却する。`PAGES_DIR` 等を抽出先へ値 import すると isolated fixture の patch が外れ得る。Campaign E は `__module__` authority 廃止、explicit reviewer mode、path dependency、patch-owner migration を先行させる。

## Authority / adoption baseline

| Contract | SHA / result |
|---|---|
| lane contract manifest | `afcda164e055f994b7c439ea78518dd754a352cb9da581642c6c792e138ca43f` |
| lane contract case manifest | `ae33c88e4eb08e3b771db21ee5d42904420c1672a38ba026e0d55f437b8312f6` |
| structured generation policy | `8add0f29a5d56c2abdad7a78b806f16b9e58e57e49a7c9b18a7bf60e45f4afa8` |
| production schema manifest | `1541981873a0669f5ef7234c9b4490fe3c3f00872d1a584b182a3a33799fbea2` |
| adopted artifact | `bb12cab91b222d0a992e8df49066d19b5fab3ec0b0c4503fc44d7d3e3e289a6c` |
| production validator | pass |
| quality probe | `ok`; frozen lanes 0 |

後続 campaign は開始直前と deploy tree 上で同じ validator を再実行する。manifest hash が意図せず変わった時点で deploy を止める。

## Runtime baseline

取得値は command / source / path / SHA / PID / status の allowlist のみを記録した。token、credential、header、cookie、env value、raw prompt / page content は記録していない。

| Consumer | Source / state | Gate |
|---|---|---|
| Live dashboard | GitHub uvx archive; commit `5d54bc7`; drift false; PID 72629 | long-lived。package変更後はkickstartとarchive commit確認 |
| Live ingest-drain | GitHub uvx archive; commit `5d54bc7`; drift false; PID 72631; state idle / active batch false | `current_job_pid=null` / `current_raw=null` を確認してからのみkickstart |
| Codex MCP | GitHub `llm-wiki-mcp[reranker]` | fresh host / MCP canaryで確認 |
| Claude Code MCP | GitHub `llm-wiki-mcp[reranker]` | fresh host / MCP canaryで確認 |
| Claude Desktop MCP | GitHub `llm-wiki-mcp[reranker]`; commit `5d54bc7`; archive `Wq4ID0BlUuVhsfNl0TTIV`; PID 22065 | local processを停止しfresh appでGitHub sourceを再検証。torch / transformers present |
| dashboard launcher | local repo script SHA `1027c10e5f6312eab4004330668d7f3a5253c64bce8b862d18ace94c0488f9f9` | package archiveとは別にSHA確認 |
| ingest-drain launcher | local repo script SHA `c32c24c0fdf5e769d522f1a3e8a9375ad2d93bffb668b7774e66ecd13a5cd8cc` | package archiveとは別にSHA確認 |
| watchdog / converge / sleep | scheduled; last exit 0 | 強制通常runをしない。fresh import/helpと次回自然runを確認 |
| soak | PID 42326; pin `3215af701c07ecc7515fbbda6cd65e26ca14a65f` | wrapper / plist / PID / start / pin / outputを変更・再起動しない |
| deadman | copied stdlib-only observer | package runtimeと同一視しない。heartbeat continuityを確認 |

Live runtime は `state=idle`、`pending=0`、`current_raw=null`、`current_job_pid=null`、`batch.active=false`。これは取得時点の quiescence であり、deploy直前に再取得する。

## Queue / quality delta baseline

絶対ゼロではなく、後続 campaign 直前の snapshot との差分で判定する。

| Metric | Baseline |
|---|---:|
| semantic deferred | 142 |
| operational deferred | 3 |
| raw outstanding | 145 |
| duplicate candidates | 10 |
| lint repair | 700 |
| search labels pending | 146 |
| raw replay pending | 2 |
| quality frozen lanes | 0 |
| quality corpus status | ok |

## Deletion manifest: initial classification

| Candidate | Classification | Replacement / compatibility | Earliest action |
|---|---|---|---|
| `normalize_recall_config()` | replacement first | unified nested configをruntime policyへ変換する現役処理 | direct unified load + policy snapshot parity後 |
| legacy `recall.toml` fallback | migration RFC | semantic hold authority fingerprintにobservationが参加 | dual-read / old-reader rollback fixture後 |
| deprecated hook `audit` / `improve` no-op | retain | documented compatibility window | 2026-10-01以降 |
| `backfill_recall_questions.py` | retain | operations docsの診断runbookから参照 | replacementとmanual canary後 |
| tag backfill apply / dry-run | retain / classify | tests、read-only diagnostics、model previewを分離 | capability map完成後 |
| session-sweeper / background-jobs / model-lab / page-normalize / metadata-backfill / repair-runbook entrypoints | retain | unified CLI replacement未確認 | replacement subcommand + deprecation後 |
| `_maybe_migrate_legacy_json()` call sites | replacement first | non-reentrant migration lock内でconnect再入禁止 | raw connect + ensure migration分離後 |
| `gate_mode` / `context_style` | separate compatibility removal | fixed production valueだけではAPI capability削除を正当化しない | explicit migration decision後 |

static zero reference、docs/tests同時削除、entrypoint usage不明のいずれも削除証明に使わない。

## Characterization added

- `tests/test_recall_session.py`: path sanitization、corrupt fail-open、bounded state、cleanup fallback
- `tests/test_orchestrator_characterization.py`: legacy counter removal、stale reservation、live lease、durable per-raw mark
- `tests/test_failure_supervisor_characterization.py`: corrupt state、UTF-8 bytes、lock-free snapshot
- `tests/test_burn_monitor_characterization.py`: exclusive evidence writer、fsync、append / truncate delta、bounded JSONL fail-closed、runtime identity
- `tests/test_refactor_inventory.py`: deterministic candidate / seam / reference inventory

## Campaign A exit gate

- [x] baseline full suite result and exact count recorded。既存failureは別commit `0f6dd7b` で修正し、対象test pass
- [x] targeted characterization pass
- [x] source / large function inventory reproducible
- [x] durability / canonical / ingest seam candidate counts reproducible
- [x] deletion candidates classified as safe / replacement-first / retain / RFC
- [x] authority / adoption / quality baseline recorded
- [x] live runtime source、launcher SHA、soak invariant、queue delta baseline recorded
- [x] Claude Desktop local-worktree driftを独立したenvironment remediationで解消し、GitHub + reranker process / archive / commitを確認
- [x] fix + Campaign A追加test + E0 authority hardeningを含むfinal full suite pass (`2,345 passed in 882.22s`)

## 実行ログ

- 19:02: `origin/main` をfetchし、baseline `5d54bc7`を確認
- 19:03: isolated worktree / branchを作成。original worktreeのuser-owned filesを保護
- 19:08: deterministic AST / reference inventoryを追加。self-test 1 passed
- 19:10: characterization 15 testsを追加し、誤った期待値1件を現行contractへ修正。16 passed
- 19:12: live runtime、authority hashes、quality / queue baselineをredacted snapshotとして記録
- 19:18: Claude Desktop configをGitHub + rerankerへ復元。local MCP processを終了し、fresh archive `Wq4ID0BlUuVhsfNl0TTIV` / commit `5d54bc7` / reranker dependenciesを確認
- 19:23: baseline full suiteは2,328 passed / 1 failed。case-insensitive filesystemで`Foo`を`foo`として返す既存bugを再現
- 19:25: filesystem directory entryの綴りを保持する修正をcommit `0f6dd7b`として分離。対象test pass
- 20:27: Campaign E0としてingest authorityの`__module__`暗黙判定を廃止し、明示reviewer injectionへ移行。`tests/test_ingest.py`は410 passed
- 20:42: final repository suiteは2,345 passed in 882.22s。Frontier test cache isolationを含め全件pass

## Rollback

Campaign A は production logic / runtime / persistent data を変更しない。rollback はこのbranchのcommit revertで完結する。host config、LaunchAgents、`~/.wiki`、soakには触れない。
