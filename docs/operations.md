# Operations

## Status

```sh
llm-wiki status
llm-wiki status --json
llm-wiki health
```

Shows wiki counts, active config, recall decision counts, feedback counts, and
runtime status. `health` focuses on knowledge KPIs: summary coverage,
recall-question coverage, raw-to-claim capture coverage, sensitivity-tier
distribution, read-back pass rate, duplicate candidates, lint repair queue
size, and golden-set size.

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

Explicit feedback is an optional diagnostic input, not an operating gate. The
auditor, pull-log attribution, and frontier-reviewed label path discover and
close normal false negatives automatically.

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

## Sensitivity Tiers

Pages can set `sensitivity: high` in frontmatter. Career-folder pages infer
`high` in the index even before frontmatter is backfilled. Recall cards show
the sensitivity annotation next to the freshness annotation, and `llm-wiki
health` reports the tier distribution. In work-project CWDs, high-sensitivity
pages are filtered unless the prompt explicitly asks for career/interview style
context.

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
The file is an observable candidate ledger. Sleep first handles deterministic
safe cases, then sends ambiguous pairs to the frontier model; approved
supersession atomically marks the loser `status: deprecated` with
`superseded_by: <winner>`. No human review queue is required.

## Raw Replay

```sh
llm-wiki raw-replay --since 2026-07-01 --limit 100
llm-wiki raw-replay --since 2026-07-01 --limit 1 --run
```

Without `--run`, replay writes `~/.wiki/review/raw-replay-queue.jsonl`.
With `--run`, selected raw files go back through the normal ingest path, so
search-before-create and read-back verification still apply.
Read-back misses caused only by ranking (`not-in-top-results`) stay in the
lighter query-hint repair lane; raw replay is reserved for structural ingest,
metadata, quarantine, and integrity failures.

Before ingest starts, replay durably records a `running` row with job,
attempt, content hash, and start time. The ingest `on_complete` callback then
fsyncs a whole-raw completion journal before queue acknowledgement. Partial
ingest is terminal `completed_partial` so already-successful operations are
never replayed. If a process dies in the narrow unprovable window, the row
becomes `indeterminate`: the frontier model must choose processed, safe replay,
or quarantine. It is never blindly retried and never becomes a human content
decision.

## Memory Integrity Eval

```sh
llm-wiki memory-integrity --limit 100
llm-wiki-memory-integrity --limit 100 --json
```

This is the first E1/W7 write-side eval. It samples raw captures, derives a
deterministic expected-term query, checks the claim ledger and search footprint,
and writes `~/.wiki/eval/memory-integrity-latest.json`. The dashboard health
panel uses this when available.

## Cofire Graph

```sh
llm-wiki cofire --limit 5000
llm-wiki-cofire --min-count 2 --json
llm-wiki prefetch --limit 5000
```

Recall logs now build a co-fire graph at `~/.wiki/recall/cofire.json`.
Search graph expansion consumes those edges alongside wikilinks/backlinks, so
pages that repeatedly appear together can reinforce each other before a
human-curated graph exists. Prefetch cache writes
`~/.wiki/recall/prefetch.json` from recent recall episodes and is checked
before normal search context assembly.

## Sleep Cycle

```sh
llm-wiki sleep --dry-run --json
llm-wiki-sleep --raw-limit 100 --eval-limit 100
```

The sleep cycle is the single bounded convergence driver. It snapshots
`~/.wiki`, rebuilds co-fire/prefetch/retention artifacts, runs memory integrity,
and then drains small batches from lint repair, raw replay, read-back repair,
search-label review, recall auto-apply/self-heal, duplicate, and orphan-link
lanes. Weekly calibration and search self-tune also run here. Every decision or
queue lane has a stable key, retry/backoff limits, a terminal quarantine, and a
shared cycle budget with a reserved frontier slot per decision lane; artifact
writes are charged to the same mutation/time budget. One lane failure
produces `status=partial` while the others continue. A single-flight lock
prevents overlapping scheduled/manual cycles. `--dry-run` is byte-for-byte
read-only, including search indexes and caches, and does not invoke frontier
reviewers. A zero `--eval-limit` skips integrity and label evaluation instead
of expanding to an unbounded corpus scan.

Sleep history is stored as a non-recursive, 1,000-row summary rather than full
nested cycle payloads. Scheduled sleep writes a compact text report, while the
15-minute watchdog keeps its latest state and bounded history in `autonomy/`
and sends routine stdout to `/dev/null`; stderr remains logged.

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
`search-golden.jsonl`. Sleep sends a bounded batch to a frontier reviewer;
approved labels are promoted automatically, rejections are terminal, and
uncertain/retry results back off before quarantine after three passes.
`--failure-index` records missed expected pages with channel candidates and a
reason code. Weekly self-tune evaluates dev weights against an independent
locked-test set, asks the frontier model for the final veto, and atomically
writes `recall/search-policy.json` only after both gates pass.

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
`recall/calibration-history.jsonl` for rollback. Sleep schedules this weekly
with bounded samples/recomputed features and a frontier final veto.

## Human Boundary

Normal content, ranking, repair, and policy decisions converge without a human.
`human_required` is reserved for deterministic external-authority failures:
OAuth/authentication, billing or quota changes, Keychain permission, or a
missing frontier tool. Ambiguity, low confidence, schema errors, and model
disagreement use bounded retry and terminal quarantine instead.

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
