# Architecture

LLM Wiki has three runtime loops around one wiki store.

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
  -> save session delta to raw/
  -> run asynchronous recall auditor with precision checks
  -> record missed_candidate and injection_used/injection_ignored feedback
  -> auto-apply safe additive actions

Nightly sleep
  -> refresh derived recall/search artifacts and integrity signals
  -> bounded consumers drain lint/raw/read-back/label/self-heal queues
  -> local proposal -> deterministic gate -> frontier final veto when risky
  -> apply atomically or reach rejected/quarantined/human_required terminal state
  -> weekly search self-tune and recall calibration with locked holdouts
```

## Store

`~/.wiki` is the source of memory truth.

- `raw/`: append-only session captures.
- `pages/`: structured pages created by ingest/lint workflows.
- `system/`: privileged pages such as profile, current state, and lessons.
- `recall/`: recall decisions, feedback, query hints, and auto-apply logs.
- `recall/sessions/`: lightweight recent query/topic/page state for an active host session.
- `recall/pull-log.jsonl`: wiki.search/wiki.read calls used as implicit pull feedback.
- `recall/calibration.json`: validated evidence-gate weights.
- `runtime/`: status, events, and metrics for observability.
- `runtime/convergence/`: exact-once state, leases, retry/backoff, and terminal decisions.

## Runtime Roles

- **MCP server**: exposes wiki tools to hosts.
- **Recall gate**: fast evidence-based search gate. BM25 is always attempted;
  semantic, graph expansion, rewrite, and calibrated weights fail open.
- **Save harness**: host-specific session parser plus memory writer.
- **Auditor**: heavy asynchronous judge that looks for missed recall and whether
  injected pages were used.
- **Auto-apply**: applies additive actions only (`query_hint`, `alias`, `page_tag`).
- **Calibration**: pure-Python logistic calibration on recall-log features and
  feedback labels, applied only after time-ordered holdout and frontier veto.
- **Sleep convergence driver**: gives every autonomous lane a shared time/call/
  mutation budget, reserves one frontier slot per decision lane, uses a
  single-flight process lock, and isolates lane failures so unrelated work
  keeps draining. Dry-run also suppresses index/cache persistence.
- **Dashboard**: local browser observability for ingest work, self-heal status,
  recall improvement runs, save history, knowledge mix, and model fleet roles.

`few_shot` feedback is materialized through safe query hints and the
frontier-reviewed golden-label path. `threshold` feedback is routed into Recall
Lab, where replay/holdout gates and a frontier veto decide adoption; neither
action waits for a human review queue.

The only human boundary is external authority: authentication/OAuth, billing or
quota, Keychain permission, or installing/restoring a missing frontier tool.
All other uncertainty ends through bounded retry, rejection, or quarantine.

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

Codex and Claude Code now enter through `llm-wiki-hook`. Legacy scripts remain
as wrappers so existing hook settings continue to work while new deployments use
the single dispatcher directly.
