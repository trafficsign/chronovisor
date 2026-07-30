# Architecture

Chronovisor has three routine loops and one exceptional repair plane around one
wiki store.

```text
UserPromptSubmit
  -> chronovisor-hook
  -> inject bounded allowlisted core memory as non-executable JSON
  -> strip non-user blocks and trivial prompts
  -> load recall/sessions/<session_id>.json
  -> update the private session/topic Recall Field and emit ordered events
  -> build queries, rewriting ambiguous references when needed
  -> exact anchors + inverted BM25 + semantic ANN + positive-use context seeds
  -> bounded two-hop associative expansion and full-vector verification
  -> resident BGE reranker shadow/canary (when configured)
  -> per-page Evidence Certificate and dynamic 0..6 pointer / 0..2 rich selection
  -> evidence gate (features -> none/cards/read)
  -> queue teacher commits for the next Field turn; never current-turn leakage
  -> optional bounded RECALL_CONTEXT as untrusted JSON
  -> reserve a final model-free BM25 fallback inside the total deadline
  -> fail open to the host; breaker preserves BM25 degradation

Stop
  -> chronovisor-hook
  -> durably enqueue one coalesced save job for the session
  -> optionally enqueue one coalesced correction-capture job for the session
  -> return immediately; no model, ingest, mutation, audit, or improvement

Capture workers
  -> save appends lossless bounded raw/ chunks with an exact cursor and receipt
  -> successful save receipt atomically queues one Recall audit candidate
  -> correction capture queues explicit or provenance-qualified signals
  -> ordinary follow-ups advance the correction cursor without queue work
  -> neither capture path performs semantic inference

Background convergence
  -> refresh derived recall/search artifacts and integrity signals
  -> bounded consumers drain lint/raw/read-back/label/self-heal queues
  -> deterministic gate -> local structured consensus when semantics are needed
  -> apply atomically or reach a terminal state:
     rejected / quarantined / semantic_deferred / human_required (external only)
  -> weekly search self-tune and recall calibration with locked holdouts

Exceptional system repair
  -> system incident supervisor proves a repeated code failure
  -> require >=3 occurrences across >=2 distinct inputs
  -> require >=2 failed local repair/recheck attempts plus reproduction evidence
  -> durable guard enforces single flight and the 24-hour budget
  -> run at most one Codex code-repair attempt
```

## Store

`~/.chronovisor` is the source of memory truth.

- `raw/`: immutable Raw archive. Legacy flat Markdown remains readable during
  rollout. Native transcript capture is stored directly under
  `YYYY/MM/DD/<host>-<session-key>-part-NNN.jsonl.open`; a fsynced adjacent
  commit journal maps each stable `save-...md` logical Raw ID to an exact byte
  range. Sealed parts use `.jsonl.zst` plus a full-restore-verified manifest.
  There is deliberately no `hot/` or `archive/` tier: date is the physical
  grouping and the suffix/manifest is lifecycle state.
- `pages/`: structured pages created by ingest/lint workflows. Knowledge pages
  always live at `pages/<top-level-folder>/<page-id>.md`; direct Markdown files
  under `pages/` are forbidden. Triage prefers the best matching existing
  folder and may create one specific kebab-case folder when none fits. The
  host validator repairs invalid model output, and the final prepare boundary
  rejects root-level creates even for replayed or previously reviewed plans.
- `system/`: privileged pages such as profile, current state, and lessons.
- `recall/`: recall decisions, feedback, query hints, and auto-apply logs.
- `recall/feedback.jsonl`: append-only recall supervision. Exact historical
  mistakes are disabled by digest-bound retraction rows, never by deletion or
  broad prompt/page matching.
- `recall/sessions/`: lightweight recent query/topic/page state for an active host session.
- `recall/pull-log.jsonl`: returned/read/injected/used decision-trace events;
  only explicit `used` events are positive Recall feedback.
- `recall/content-feedback.jsonl`: immutable audit records for applied content corrections.
- `recall/calibration.json`: validated evidence-gate weights.
- `recall/evidence-certificate-ledger.jsonl`: query/content/policy-bound private
  page certificates. Raw prompts are not stored.
- `recall/field/`: sealed sparse session snapshots plus ordered Field events.
  Sessions are keyed by a host/session hash and never merged implicitly.
- `runtime/`: status, events, and metrics for observability.
- `runtime/recall-labels/ledger.jsonl`: derived silver/strong/gold supervision.
  Certificate pass is silver, explicit `recall_used` is strong, reviewed eval
  is gold, and read/reject exposure is never silently converted to a negative.
- `runtime/recall-field/candidate-trace.jsonl`: privacy-safe Field/teacher
  disagreement and fallback evidence.
- `runtime/raw-projections/parents/`: small deterministic logical references
  used by Path-oriented queues. They contain stable Raw IDs, hashes, and commit
  evidence, not a physical locator or second transcript copy. Semantic projection resolves and verifies the
  authoritative Raw bytes before use.
- `runtime/ingest-liveness.json`: Ollama readiness, pending Raw count,
  outage duration, and recovery transition for the persistent drain worker.
- `runtime/provisional-recall/`: a capped, citation-only search namespace for
  verified projections of semantically deferred Raw units. Ranking uses
  IDF-weighted coverage, match density, and exact-phrase evidence inside this
  namespace only; it can never authorize mutation.
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
- **Recall gate**: fast evidence-based search gate with separate L1 core-memory and
  L2 page-card budgets. The complete hook shares one deadline and an outer hard
  timer. The primary path reserves time for an L1 + BM25 fallback; semantic,
  graph expansion, rewrite, and calibrated weights may degrade without losing
  deterministic retrieval. Repeated failure opens a cooldown breaker that
  keeps only the cheap BM25 path active. Injected material is data, not an
  instruction channel.
- **Resident reranker**: a mode-0600 Unix-socket BGE service keeps model load,
  page text cache, and MPS warmup outside short-lived prompt hooks. `shadow`
  records before/after/latency only; `canary`/`on` still fail open to the fused
  teacher path. A shared foreground accelerator lease prevents Nemotron and BGE
  from silently overcommitting MPS.
- **Recall Processor**: owns candidate reranking, supporting-span selection,
  Evidence Certificates, bounded judge escalation, marginal-utility stopping,
  and commit. It is the retire/commit boundary: Field candidates can never
  bypass certificates.
- **Stateful Recall Field**: maintains deterministic sparse activation per
  session/topic epoch. Exact prompt/entity hits and prior-turn teacher commits
  provide direct stimulus; typed graph edges spread activation with decay,
  refractory periods, capacity, and inhibition. Exposure co-fire is excluded
  from authority. Modes are `off`, `shadow`, `candidate`, and `active`, with
  deterministic session canaries. Missing/failed promotion evidence always
  rolls candidate authority back to the full teacher.
- **Recall Compiler**: a shadow-only meaning-address lookup over typed Claims.
  It accepts only structured intents with an exact subject/page address, one
  non-conflicting active value, matching time, and a current source digest.
  Everything else falls back to Nemotron+BGE.
- **Save harness**: host-specific transcript parser plus a deterministic,
  lossless delta writer. `legacy` keeps the Markdown writer, `shadow` keeps it
  authoritative while mirroring exact source lines, and `v2` appends original
  newline-terminated JSONL line bytes without parse/re-serialization drift.
  Data fsync precedes commit-journal fsync and receipt read-back; only then may
  the cursor advance. A single oversized source record remains one committed
  range in v2. The save path performs no compression or LLM inference.
- **Raw archiver**: a bounded local maintenance lane, independent of Ollama.
  It seals only pre-today open segments, verifies a streamed full restore and
  every logical range digest, atomically publishes the manifest, and only then
  removes the open data/journal. Existing processed flat Raw can separately be
  shadow-packed as byte-exact `legacy-part-NNN.tar.zst`; unprocessed, held,
  quarantined, and current-day files are excluded.
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
  Ingest treats one narrowly defined outcome differently from an operational
  failure: when all three voters return valid, pairwise-distinct decisions under
  the currently validated adopted-artifact SHA, the immutable source raw enters
  terminal semantic defer. It remains in `raw/` and is excluded from self-heal,
  frontier repair, and time-based replay. It reopens only after the router fully
  validates a different adopted-artifact SHA; a changed but invalid nomination
  fails closed. Runtime, transport, capacity, and other operational failures
  remain in the separate repair queue.
- **Content correction**: binds explicit user corrections, plus narrowly
  qualified bare denial/contrast signals, to the preceding complete turn by
  exact prompt hash, host, session, and timestamped Recall provenance. Bare
  signals require a real provenance candidate. Stop only schedules its
  capture-only worker; the sleep/local
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
- **Auditor**: heavy asynchronous judge that looks for missed Recall. It is
  queued only after a successful durable save receipt. Search `returned` and
  page `read` events are telemetry, not positive supervision; only an explicit
  `chronovisor_recall_used` event says a page influenced the answer.
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
  improvement runs, save history, knowledge mix, and model fleet roles. Ingest
  semantic defers are reported separately from pending and failed work. A dead
  worker PID is rendered as idle rather than as live work.
- **Classification/Librarian plane**: assigns stable UUIDv7 identities in a
  metadata registry, projects wikilinks to UID edges, and produces bounded
  local-only shadow classification proposals. The bundled top-level UDC
  bootstrap cannot become authority; activation is fail-closed until a complete
  licensed package and locked calibration artifact exist. Merge application is
  explicit-only and requires deterministic span, fingerprint, provenance,
  sensitivity, link, redirect, and CAS gates. See
  [Classification and Librarian](librarian.md).

Production entry points load `chronovisor` from the pushed GitHub source via
`uvx`. The checkout selected by `CHRONOVISOR_REPO_ROOT` is only exceptional
code-repair context and the destination for an approved patch, so an unpushed
worktree cannot silently become the running memory policy.

`few_shot` feedback is materialized through safe query hints and the locally
reviewed golden-label path. `threshold` feedback is routed into Recall Lab,
where replay/holdout gates and local consensus decide adoption; neither action
waits for a human review queue.

The human boundary is external authority: authentication/OAuth, billing or
quota, and Keychain/secret-store permission. These failures are never sent to a
model. Missing tools or models and all other routine uncertainty use bounded
retry, rejection, or cooldown quarantine that is reopened autonomously. The
artifact-bound ingest semantic defer described above is the exception: it is
not a timed quarantine, and operational failures never enter that state.

## Compatibility Names

Older modules, functions, environment variables, schemas, and artifacts may
still contain `frontier`, `frontier_mode`, or `frontier_*` in their names.
Routine callers enter `run_structured_review()`, which is now a local-consensus
compatibility boundary and cannot launch Codex, Claude, or a custom review
command. Only `run_frontier_review()` with validated `RepairIncidentEvidence`
can start an actual frontier process, and that entry point is reserved for
system code repair.

## Search Pipeline

`chronovisor.search.pipeline` owns the shared ranking orchestration. Production
`search.search()`, search evaluation variants, and self-tune weight trials all
enter through `run_search_pipeline()` with different `PipelineConfig` values:

```text
exact page/title/tag/entity anchors + SQLite inverted BM25
  -> optional 512d HNSW semantic candidates, exactly rescored at 2048d
  -> explicit-used context seeds
  -> weighted seed merge
  -> typed graph expansion (links, backlinks, entities, tags, positive co-fire)
     with at most 2 hops and 50 output nodes
  -> cached full-vector verification of graph candidates
  -> usage prior (production config only when usage_prior weight > 0)
  -> weighted fusion or plain RRF
  -> strong contextual Anti-Index evidence
  -> filter / sort / truncate
```

The MCP `chronovisor_search` tool adds exact tag filtering and the configured
rerank stage for relevance-sorted queries. Automatic Recall may call the
resident reranker in `shadow`, `canary`, or `on` mode within its remaining
deadline; service failure keeps the fused result unchanged. Contextual
Anti-Index uses only reviewed false-positive or hash-bound strong contradiction
evidence. Hub suppression stays diagnostic/shadow while its hard-94 gate is
unmet: degree alone never penalizes a page, exact matches are protected, and
query specificity plus supporting-span coverage reduce the component.

The five Recall layers, their budgets, and decision-trace semantics are defined
in [Recall Orchestration](recall-orchestration.md).

## Host Boundary

Codex and Claude Code enter through `chronovisor-hook`. Legacy scripts remain as
wrappers so existing hook settings continue to work while new deployments use
the single dispatcher directly. The Stop dispatcher schedules deterministic
save/correction capture only; successful save completion can enqueue an audit
candidate, but asynchronous semantic work belongs to convergence workers rather
than the host's session boundary.
