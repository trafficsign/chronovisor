# Operations

## Status

```sh
llm-wiki status
llm-wiki status --json
```

Shows wiki counts, active config, recall decision counts, feedback counts, and
runtime status.

## Doctor

```sh
llm-wiki doctor
llm-wiki doctor --json
```

Runs lightweight operational checks for wiki directories, config, and detected
host hooks.

## Hook Install

```sh
llm-wiki hooks install --host all
llm-wiki hooks inspect --json
```

Use the installer after changing host hook topology. It keeps non-wiki hooks in
place, replaces legacy LLM Wiki script wrappers with direct dispatcher commands,
and refreshes Codex trusted hashes.

## Recall Logs

```sh
llm-wiki-recall --recent 20
llm-wiki-recall --feedback missed --prompt "..." --note "..." --ref <decision_id>
llm-wiki recall-eval --json
llm-wiki recall-eval --save-baseline
```

Manual feedback remains useful for false negatives that the auditor cannot
observe confidently.

`recall-eval` builds a replay dataset from `recall/recall-log.jsonl` and
`recall/feedback.jsonl`, then reruns the current gate without writing new
decision logs. Use it before and after changing recall thresholds, fusion
weights, rewrite settings, or context style.

## Audit and Auto-Apply

```sh
llm-wiki-recall-audit --host codex --hook --audit-read
llm-wiki-recall-auto-apply --dry-run
llm-wiki-self-heal --auto-apply-errors --auto-apply-error-threshold 3 --dry-run
```

Auditor feedback uses `kind = "missed_candidate"` and source `auditor` for
false negatives. Precision labels use `kind = "injection_used"` or
`kind = "injection_ignored"`. The Stop dispatcher passes `--audit-read` so read
decisions can be precision-audited without changing the auditor CLI default.

Repeated `recall/auto-apply.jsonl` errors are promoted into self-heal packets
after the configured threshold. The live auto-apply path accumulates repeated
errors across runs, starts the existing local Qwen repair loop, and requires the
frontier reviewer before code or policy fixes are applied. The `--dry-run`
self-heal command reads the log and reports candidate clusters without writing
packets or state.

## Search Ranking Review

```sh
llm-wiki-eval --build-label-queue
llm-wiki-eval --report --failure-index
llm-wiki-eval --self-tune
```

`--build-label-queue` writes auditor/search candidates to
`recall/search-label-queue.jsonl`; it does not promote rows into
`search-golden.jsonl`. Promote only after human review. `--failure-index`
records missed expected pages with channel candidates and a reason code.
`--self-tune` is shadow-only: it searches dev-set weights and checks locked-test
guardrails, but never edits config by itself.

## Calibration

```sh
llm-wiki-recall-calibrate --dry-run
llm-wiki-recall-calibrate
llm-wiki-recall-calibrate --rollback
```

Calibration trains on older labeled rows and validates on the newest holdout
slice. It writes `recall/calibration.json` only when holdout improvement exceeds
the configured minimum, and records the old artifact in
`recall/calibration-history.jsonl` for rollback.

## Recall Question Backfill

```sh
scripts/backfill_recall_questions.py --dry-run
scripts/backfill_recall_questions.py --limit 50
llm-wiki-reindex
```

The backfill adds `summary` and `recall_questions` frontmatter to existing
pages. Re-run `llm-wiki-reindex` after a large backfill so semantic search sees
question vectors.

## Troubleshooting

- If Codex hooks appear disabled, inspect `~/.config/codex/config.toml` trusted
  hash entries.
- If hooks look stale, check whether host settings call local scripts or a
  package entry point.
- If local models remain loaded after tests, run `ollama ps` and stop them.
- If a hook still appears to use old recall behavior after a GitHub package
  update, check the running `uvx` process and cache before changing local code.
