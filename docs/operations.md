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
The query hint is still a production ranking change: the local failure signal
only creates an exact proposal, and a frontier approval bound to the page hash
is durably persisted before the hint is written. Rejection is terminal for the
same evidence; transient or low-confidence decisions retry autonomously.

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

Legacy maintenance scripts may still produce read-only diagnostics, but their
heuristic/local-model page mutation paths are fail-closed. Garbage cleanup,
tag/link/recall-metadata backfill, broken-link rewriting, and model-selected
folder moves must enter the sleep/frontier lanes; they cannot write knowledge
pages directly.

## Wiki Snapshots

```sh
llm-wiki wiki-snapshot "before manual repair"
llm-wiki-snapshot "before manual repair"
```

`~/.wiki` is initialized as its own git repository on first snapshot. Scheduled
lint auto-fix and MCP `wiki_apply` snapshot before changing files, giving
self-heal and repair work a rollback point independent of the code repository.

## User Content Corrections

```sh
llm-wiki-content-correction --host codex --session-file /path/to/session.jsonl --capture --run-due
llm-wiki-content-correction --host claude-code --session-file /path/to/session.jsonl --capture --run-due
```

The Stop hook runs this lane automatically. An explicit user correction is
bound to the preceding complete turn. Recall provenance must match the exact
prompt hash, host, session, and turn-time window; injected pages and pages read
during that recall decision form the only mutation candidates. A durable
cursor keyed by host/session/transcript tracks the last completed assistant
line. Later Stop runs therefore pick up delayed transcript appends and failed
captures without replaying already-enqueued correction turns.

The local model proposes page error, outdated claim, wrong retrieval,
assistant misquote, ambiguity, unattributed, or no correction, but never makes
the terminal semantic decision. Every classification is checked by the
frontier model. A frontier-confirmed wrong retrieval writes `kind =
"page_ignored"` with only its explicit `negative_pages` subset; the remaining
pages from the recall decision are not demoted. Non-page classifications do not
mutate wiki content.

Normal pages plus the user-memory system pages `user-profile`, `current-state`,
and `lessons-learned` are correctable when exact recall provenance names them.
Operational system files remain outside the mutation boundary.

Content mutations require unique exact old spans, verbatim user evidence,
protected literal grounding, frontier approval of immutable before/after
hashes, and a per-page CAS immediately before each replace. The CAS runs under
the writer lock shared by correction, ingest, lint, entity, and orphan-link
repairs; a partial multi-page failure rolls back only bytes still owned by the
correction.

The approved frontier payload is persisted as a review artifact before page
bytes change. On restart, a matching correction marker and artifact allow the
lane to resume refresh/audit work without asking the frontier model to decide
the same patch again. Terminal `applied` additionally requires successful
refresh of the page store, BM25, changed-page embeddings, claim graph, and
generated index, followed by semantic search read-back of every changed page
and verification that old spans are inactive and new spans are present.
Refresh or read-back failure remains retryable rather than being reported as a
successful correction.

Audit rows go to `recall/content-feedback.jsonl`; capture cursors, proposals,
and review artifacts live under `runtime/content-correction/`, while lease and
retry state lives under `runtime/convergence/`. Exhausted autonomous failures
enter quarantine for a cooldown and are then reopened automatically. An invalid
review artifact is preserved under `invalid-artifacts/` and replaced by a fresh
frontier decision; it is never trusted or silently discarded.

When a historical raw capture is known to be false, keep its body for audit and
set `raw_status: retracted` in frontmatter. Normal ingest, explicit replay,
automatic replay signals, and already-queued replay all exclude it.

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

Auto-apply treats the local auditor as a proposal source only. Alias, query
hint, and page-tag actions (including few-shot-derived hints) require a
frontier approval bound to the exact proposal and current page hash. The
approval artifact is persisted and read back before the mutation, then reused
after budget deferral or a crash. Active recall-policy candidates likewise
require a frontier final approval even when legacy `frontier_mode` is `off`.

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
The legacy `--build-golden` spelling is a compatibility alias for building the
same label queue; it can no longer overwrite the authoritative golden file.
Evaluation, CI, and self-tune load only rows carrying `reviewed: true`.
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
with bounded samples/recomputed features and a frontier final veto. The public
calibration CLI uses the same mandatory frontier gate even if a legacy caller
passes `frontier_mode=off`; an approval bound to the exact active-policy hash is
persisted before a CAS-protected policy write.

## Human Boundary

Normal content, ranking, repair, and policy decisions converge without a human.
`human_required` is reserved for deterministic external-authority failures:
OAuth/authentication, billing or quota changes, or Keychain/secret-store
permission. Missing tools or models, ambiguity, low confidence, schema errors,
and model disagreement use autonomous retry and cooldown quarantine instead.

## Recall Question Backfill

```sh
scripts/backfill_recall_questions.py --dry-run
llm-wiki-sleep --raw-limit 100 --eval-limit 100
```

The legacy script is diagnostic-only. Recall-metadata proposals now enter the
scheduled sleep pipeline, where a frontier reviewer binds any accepted change
to the exact page preimage and the shared writer performs refresh/read-back.
Reference pages remain excluded by default.

## Troubleshooting

- If Codex hooks appear disabled, inspect `~/.config/codex/config.toml` trusted
  hash entries.
- If hooks look stale, check whether host settings call local scripts or a
  package entry point.
- If local models remain loaded after tests, run `ollama ps` and stop them.
- If a hook still appears to use old recall behavior after a GitHub package
  update, check the running `uvx` process and cache before changing local code.
