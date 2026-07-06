# Operations

## Status

```sh
llm-wiki status
llm-wiki status --json
llm-wiki health
```

Shows wiki counts, active config, recall decision counts, feedback counts, and
runtime status. `health` focuses on knowledge KPIs: summary coverage,
recall-question coverage, read-back pass rate, duplicate candidates, lint
repair queue size, and golden-set size.

## Doctor

```sh
llm-wiki doctor
llm-wiki doctor --json
```

Runs lightweight operational checks for wiki directories, config, and detected
host hooks.

## Dashboard

```sh
llm-wiki-dashboard --host 127.0.0.1 --port 8765
```

The local dashboard is the primary live operations view. `Current Work` shows
the active ingest stage (`Raw -> Triage -> Generate -> Apply -> Index`), the
current raw/job if one is running, and the last completed raw while idle. `Model
Fleet` combines configured roles with Ollama installed/loaded state, so unused
local models should not appear once they are removed from config and from the
local model store.

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

## Ingest Model

The page-generation path reads `[ingest]` from `~/.wiki/config.toml`. Keep the
default `num_ctx` smaller than the model ceiling for faster MLX warm runs; the
client automatically grows the request context up to `max_num_ctx` when a raw
transcript is unusually long. Changing the ingest model does not require a
semantic reindex unless `[embedding].model` also changes.

Before generation, ingest now runs a conservative search-before-create gate.
High-confidence duplicate `create` ops are rewritten to `update` ops when an
existing active knowledge page has the same page id, same title, near-identical
title/page id, or a matching search result. Reference pages are not considered
update targets.

After apply and embedding refresh, ingest read-backs changed pages with their
`recall_questions`, `summary`, or title. Failures are non-fatal and are logged
to `~/.wiki/runtime/ingest-read-back-failures.jsonl`.

Successful ingest also appends a lightweight claim seed to
`~/.wiki/claims/claims.jsonl`. The current page files remain the source of
truth, but the append-only ledger gives future event-sourced memory work a
machine-checkable trail.

## Working Memory

`system/current-state.md` is treated as a state register. Codex/Claude Code
prompt hooks inject it as a small `[WORKING_MEMORY]` block even when the normal
recall gate decides `none`. System notifications and internal prompts remain
filtered before this path.

## Entity Registry

```sh
llm-wiki entities init
llm-wiki entities backfill --dry-run
llm-wiki entities backfill --limit 100
```

The registry lives at `~/.wiki/entities/registry.json`. Ingest patches
`entities: [...]` frontmatter on created/updated knowledge pages using known
aliases such as MHI/三菱重工, KHI/川崎重工, Codex, Ollama, Qwen, and Gemma.
Entity backfill skips reference pages by default.

## Knowledge Quality Queues

```sh
llm-wiki-duplicate-review --write
wiki_check
wiki_apply
```

Pages with `type: reference` are excluded from default search, lint, duplicate
review, and recall metadata backfill. `car-spec/` pages infer this type even if
older files are missing the field; explicit `folder="car-spec"` searches still
include them.

`wiki_check` returns a compact issue summary plus a bounded sample instead of
dumping every issue. `wiki_apply` writes remaining non-auto-fixable lint work to
`~/.wiki/review/lint-repair-queue.jsonl`, split into safe-auto-fix,
heavy-model-batch, review, and monitor lanes.

`llm-wiki-duplicate-review --write` builds
`~/.wiki/review/duplicate-candidates.jsonl` from title and embedding similarity.
The queue is review-only; merge decisions should mark the losing page with
`status: deprecated` and `superseded_by: <winner>`.

## Raw Replay

```sh
llm-wiki raw-replay --since 2026-07-01 --limit 100
llm-wiki raw-replay --since 2026-07-01 --limit 1 --run
```

Without `--run`, replay writes `~/.wiki/review/raw-replay-queue.jsonl`.
With `--run`, selected raw files go back through the normal ingest path, so
search-before-create and read-back verification still apply.

## Wiki Snapshots

```sh
llm-wiki wiki-snapshot "before manual repair"
llm-wiki-snapshot "before manual repair"
```

`~/.wiki` is initialized as its own git repository on first snapshot. Scheduled
lint auto-fix and MCP `wiki_apply` snapshot before changing files, giving
self-heal and repair work a rollback point independent of the code repository.

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
llm-wiki-eval --ci --ci-variant hybrid-current --min-recall-at-5 0.80
```

`--build-label-queue` writes auditor/search candidates to
`recall/search-label-queue.jsonl`; it does not promote rows into
`search-golden.jsonl`. Promote only after human review. `--failure-index`
records missed expected pages with channel candidates and a reason code.
`--self-tune` is shadow-only: it searches dev-set weights and checks locked-test
guardrails, but never edits config by itself.

The optional Hugging Face reranker is disabled in the normal local profile.
Ranking still runs through BM25 + semantic fusion; enable `[search.reranker]`
only for explicit MCP `wiki.search` or search-eval reranker experiments.

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
knowledge pages and skips reference pages unless `--include-reference` is
passed. Re-run `llm-wiki-reindex` after a large backfill so semantic search
sees question vectors.

## Troubleshooting

- If Codex hooks appear disabled, inspect `~/.config/codex/config.toml` trusted
  hash entries.
- If hooks look stale, check whether host settings call local scripts or a
  package entry point.
- If local models remain loaded after tests, run `ollama ps` and stop them.
- If a hook still appears to use old recall behavior after a GitHub package
  update, check the running `uvx` process and cache before changing local code.
