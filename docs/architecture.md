# Architecture

LLM Wiki has three routine loops and one exceptional repair plane around one
wiki store.

```text
UserPromptSubmit
  -> llm-wiki-hook
  -> strip non-user blocks and trivial prompts
  -> load recall/sessions/<session_id>.json
  -> build queries, rewriting ambiguous references when needed
  -> BM25 + semantic + graph-expanded search
  -> evidence gate (features -> none/cards/read)
  -> optional thin RECALL_CONTEXT cards

Stop
  -> llm-wiki-hook
  -> durably enqueue one coalesced save job for the session
  -> optionally enqueue one coalesced correction-capture job for the session
  -> return immediately; no model, ingest, mutation, audit, or improvement

Capture workers
  -> save appends lossless bounded raw/ chunks with an exact cursor and receipt
  -> correction capture queues only explicit correction signals
  -> ordinary follow-ups advance the correction cursor without queue work
  -> neither capture path performs semantic inference

Background convergence
  -> refresh derived recall/search artifacts and integrity signals
  -> bounded consumers drain lint/raw/read-back/label/self-heal queues
  -> deterministic gate -> local structured consensus when semantics are needed
  -> apply atomically or reach rejected/quarantined/human_required terminal state
  -> weekly search self-tune and recall calibration with locked holdouts

Exceptional system repair
  -> system incident supervisor proves a repeated code failure
  -> require >=3 occurrences across >=2 distinct inputs
  -> require >=2 failed local repair/recheck attempts plus reproduction evidence
  -> durable guard enforces single flight and the 24-hour budget
  -> run at most one Codex code-repair attempt
```

## Store

`~/.wiki` is the source of memory truth.

- `raw/`: append-only session captures.
- `pages/`: structured pages created by ingest/lint workflows.
- `system/`: privileged pages such as profile, current state, and lessons.
- `recall/`: recall decisions, feedback, query hints, and auto-apply logs.
- `recall/feedback.jsonl`: append-only recall supervision. Exact historical
  mistakes are disabled by digest-bound retraction rows, never by deletion or
  broad prompt/page matching.
- `recall/sessions/`: lightweight recent query/topic/page state for an active host session.
- `recall/pull-log.jsonl`: wiki.search/wiki.read calls used as implicit pull feedback.
- `recall/content-feedback.jsonl`: immutable audit records for applied content corrections.
- `recall/calibration.json`: validated evidence-gate weights.
- `runtime/`: status, events, and metrics for observability.
- `runtime/convergence/`: exact-once state, leases, retry/backoff, and terminal decisions.
- `runtime/content-correction/`: per-session capture cursors plus immutable
  proposal and structured-review artifacts used for crash recovery. Some files
  retain `frontier_*` names for compatibility; those names do not identify a
  frontier-model call.
- `runtime/local-consensus/`: bounded, redacted local structured-session and
  quorum summaries. Prompts and raw model output are not persisted here.
- `runtime/frontier-repair/`: durable guard state for exceptional system-code
  repair only.

## Runtime Roles

- **MCP server**: exposes wiki tools to hosts.
- **Recall gate**: fast evidence-based search gate. BM25 is always attempted;
  semantic, graph expansion, rewrite, and calibrated weights fail open.
- **Save harness**: host-specific transcript parser plus a deterministic,
  lossless delta writer. It uses durable cursors/receipts, chunks oversized
  deltas, and performs no LLM inference.
- **Local structured session**: sends a schema-constrained Ollama chat request.
  If validation fails, it returns the exact schema errors to the same chat and
  allows at most two repair turns. Input, output, feedback, timeout, and context
  limits are fixed; exhaustion fails closed.
- **Decision router**: asks `maxwell1500/ornith-35b:Q5_K_M` and `gpt-oss:20b`
  for independent routine votes. Matching votes finish immediately. Otherwise
  `gemma4:26b` is used as a tie-breaker and any two matching votes form the
  quorum. The complete structured-session token budget selects the smallest
  executable configured context bucket; buckets below the lightest production
  lane envelope are omitted. The current 2KB feedback policy uses 32K, 64K,
  96K, and 112K. Measured footprints at that bucket then determine whether one,
  two, or three runners may remain resident. For ingest repairs, each model
  selects only one host-hashed `repair_option_id`; the router materializes its
  trusted exact arrays and revalidates them before action signatures and quorum.
  The host supplies no regex-derived semantic verdict: every structurally valid
  tag option remains available and both local models judge the exact raw and
  proposed page independently.
  Every structured vote uses the sealed sampler policy (`temperature=0`,
  `seed=0`, no thinking, JSON Schema output). Its policy hash is part of both
  the effective-request fingerprint and adoption identity, so an unseeded or
  differently seeded artifact cannot authorize production.
  No local failure or disagreement has a frontier fallback. Frontier execution
  exists only in the separately guarded system-code-repair plane.
- **Content correction**: binds explicit user corrections to the preceding
  complete turn by exact prompt hash, host, session, and timestamped recall
  provenance. Stop only schedules its capture-only worker; the sleep/local
  convergence worker later resolves the queued item. Page error, outdated
  claim, wrong retrieval, assistant misquote,
  ambiguity, no attributable page, and no correction are resolved by the local
  decision router. Wrong retrieval writes `page_ignored` feedback only for the
  locally agreed `negative_pages`; it never penalizes every recalled page.

  Exact-provenance corrections may target ordinary pages or the allowlisted
  user-memory system pages `user-profile`, `current-state`, and
  `lessons-learned`; operational system files remain forbidden. Approved
  content changes persist a review artifact before writing, then use correction
  markers, per-page CAS under the writer lock shared with ingest and repair
  lanes, and owned-byte rollback. Recovery can reuse that artifact when the
  page marker proves the approved bytes were already applied, avoiding a
  duplicate local decision.

  A content change is not terminally applied until the page store, BM25,
  embeddings, claim graph, and generated index all refresh successfully and a
  semantic search read-back finds each changed page with the old/new span
  postconditions satisfied. Refresh or read-back failure stays in bounded retry.
  Raw captures marked `raw_status: retracted` remain as audit evidence but are
  excluded from normal ingest and replay.
- **Auditor**: heavy asynchronous judge that looks for missed recall and whether
  injected pages were used.
- **Auto-apply**: applies additive actions only (`query_hint`, `alias`, `page_tag`).
- **Calibration**: pure-Python logistic calibration on recall-log features and
  feedback labels, applied only after a time-ordered holdout and local consensus.
- **Sleep convergence driver**: gives every autonomous lane a shared time/call/
  mutation budget, uses a single-flight process lock, and isolates lane failures
  so unrelated work keeps draining. Dry-run also suppresses index/cache
  persistence and model calls.
- **System incident supervisor**: is the only automatic producer allowed to
  request frontier code repair. It excludes normal JSON, content, semantic,
  model-disagreement, and external-authority failures. A strict evidence
  envelope and durable guard admit at most one Codex attempt; the default guard
  also enforces one started repair globally per 24 hours.
- **Dashboard**: local browser observability for ingest work, self-heal status,
  local-consensus activity/summaries, guarded frontier-repair state, recall
  improvement runs, save history, knowledge mix, and model fleet roles. A dead
  worker PID is rendered as idle rather than as live work.

Production entry points load `llm-wiki-mcp` from the pushed GitHub source via
`uvx`. The checkout selected by `LLM_WIKI_REPO_ROOT` is only exceptional
code-repair context and the destination for an approved patch, so an unpushed
worktree cannot silently become the running memory policy.

`few_shot` feedback is materialized through safe query hints and the locally
reviewed golden-label path. `threshold` feedback is routed into Recall Lab,
where replay/holdout gates and local consensus decide adoption; neither action
waits for a human review queue.

The human boundary is external authority: authentication/OAuth, billing or
quota, and Keychain/secret-store permission. These failures are never sent to a
model. Missing tools or models and all other routine uncertainty use bounded
retry, rejection, or cooldown quarantine that is reopened autonomously.

## Compatibility Names

Older modules, functions, environment variables, schemas, and artifacts may
still contain `frontier`, `frontier_mode`, or `frontier_*` in their names.
Routine callers enter `run_structured_review()`, which is now a local-consensus
compatibility boundary and cannot launch Codex, Claude, or a custom review
command. Only `run_frontier_review()` with validated `RepairIncidentEvidence`
can start an actual frontier process, and that entry point is reserved for
system code repair.

## Search Pipeline

`llm_wiki_mcp.pipeline` owns the shared ranking orchestration. Production
`search.search()`, search evaluation variants, and self-tune weight trials all
enter through `run_search_pipeline()` with different `PipelineConfig` values:

```text
BM25
  -> optional semantic search
  -> graph expansion (production config calls it even when graph weight is 0)
  -> usage prior (production config only when usage_prior weight > 0)
  -> weighted fusion or plain RRF
  -> negative feedback demotion
  -> filter / sort / truncate
```

The MCP `wiki.search` tool adds one post-search stage: exact tag filtering,
then `apply_rerank_stage()` only when the optional reranker is enabled for
relevance-sorted queries. The synchronous recall hook keeps using the faster
fused search path and does not call the reranker.

## Host Boundary

Codex and Claude Code enter through `llm-wiki-hook`. Legacy scripts remain as
wrappers so existing hook settings continue to work while new deployments use
the single dispatcher directly. The Stop dispatcher schedules save capture
only; asynchronous semantic work belongs to convergence workers rather than the
host's session boundary.
