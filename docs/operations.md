# Operations

## Status

```sh
chronovisor status
chronovisor status --json
chronovisor health
```

Shows wiki counts, active config, recall decision counts, feedback counts, and
runtime status. `health` focuses on knowledge KPIs: summary coverage,
recall-question coverage, raw-to-claim capture coverage, sensitivity-tier
distribution, read-back pass rate, duplicate candidates, lint repair queue
size, and golden-set size.

Research health adds run/claim counts, source provider and cache use,
first-pass malformed vs repair counts, role budgets, stop reasons, and Decision
Trace coverage. The same fields appear in the dashboard Memory Health panel.

## Doctor

```sh
chronovisor doctor
chronovisor doctor --json
```

Runs lightweight operational checks for wiki directories, config, and detected
host hooks.

## Dashboard

```sh
chronovisor-dashboard --host 127.0.0.1 --port 8765
```

### Semantic retrieval service

Install the pinned Nemotron service after its model snapshot is present in the
local Hugging Face cache:

```sh
scripts/install-semantic-service
chronovisor-semantic-service status
chronovisor-semantic-service rebuild
chronovisor-semantic-service archive-legacy
```

The service listens only on the mode-0600 Unix socket
`~/.chronovisor/runtime/semantic.sock`. Rebuilds publish an immutable
generation atomically; query errors and rebuild windows return the search
pipeline to BM25 rather than starting Ollama BGE on the synchronous path.
`chronovisor-semantic-service rollback` switches to the previous verified
generation without re-embedding. Run `archive-legacy` only after the new
runtime is active: it refuses incomplete/stale coverage, writes a mode-0600
zstd archive, verifies a decompression checksum, and then removes the mutable
BGE SQLite file. The service deletes that retirement archive after 14 days.

The local dashboard is the primary live operations view. `Current Work` shows
the active ingest stage (`Raw -> Triage -> Generate -> Apply -> Index`), the
current raw/job if one is running, and the last completed raw while idle. `Model
Fleet` combines configured roles with Ollama installed/loaded state, so unused
local models should not appear once they are removed from config and from the
local model store. Local review activity is labeled as local consensus, with
bounded completion counts for first-pass validity, repaired responses, repair
turns, pair agreement, tie-break use, and unresolved quarantine. Save Load and
Batch Yield render artifact-bound ingest semantic defers separately from
pending and failed work. Guarded Codex repair has a separate incident/budget
view. A missing or dead worker PID is idle, not live work.

Dashboard Decision Trace is event-backed, not a simulated progress animation.
Each local structured session writes redacted phase transitions to the bounded
`runtime/local-consensus/trace-events.jsonl` journal. The dashboard reads only
the current request through `/api/local-consensus`, polls that lightweight view
independently of the full snapshot, and replays unseen event IDs in durable
order. The journal never contains prompts, schemas, raw model output, or vote
payloads. Reduced-motion and background tabs skip animation and render the
latest authoritative state immediately.

## Evidence Research

`chronovisor_research` is an MCP tool and is asynchronous by default. Inspect the
returned durable job with `chronovisor_jobs(job_id)`:

```text
chronovisor_research(query="current status of the target", claims=["..."])
chronovisor_jobs(job_id="...")
```

MCP hosts freeze their tool schema when a conversation is created. If
`chronovisor_research` was added after that conversation started, reconnecting only
the process is insufficient in some hosts: start a new conversation and verify
that `chronovisor_research` appears in the advertised tool list. The server publishes
it unconditionally; a publication regression is covered by the test suite.

When the optional reranker is enabled, the MCP process warms its locally cached
model in parallel with index startup and waits at most eight seconds before
advertising tools. Model loading prefers the complete local Hugging Face
snapshot and only permits a network fallback on first install, so remote
metadata checks do not inflate the first interactive search.

The GitHub-installed package exposes the same background-first contract for
deployment canaries and shell use:

```sh
chronovisor-research "current status of the target" --claim "..." --json
# Explicit live diagnostic only; normal use stays queued.
chronovisor-research "current status of the target" --sync --no-challenge --json
```

The tool follows Wiki -> verified claims -> Raw -> Web. Web search and fetch
have separate permission checks and independent kill switches. Fetch accepts
only public, non-authenticated URLs returned by the search action. Results are
untrusted data, never instructions. Supported claims receive citations derived
from immutable Evidence Artifacts; contradicted and unknown claims remain
explicit.

Production Web search uses four adopted source packs rather than an open-ended
provider list: local SearXNG for general discovery, GitHub for code/releases,
arXiv plus Crossref for academic metadata, and MediaWiki for encyclopedic
queries or bounded fallback. Install the pinned loopback-only SearXNG service
before enabling the federation:

```sh
scripts/install-searxng
curl --fail --silent \
  'http://127.0.0.1:8888/search?q=Chronovisor&format=json'
```

GitHub works without a token at the public unauthenticated rate limit. Set
`GITHUB_TOKEN` in the MCP process only if a higher authenticated limit is
needed; the token is never written to research traces.

Run the mutation-free adversarial verifier before rollout or after changing
budgets/providers:

```sh
chronovisor-research-verify --json
```

Research, Web, checkpoint compaction, and Sleep consolidation are separately
reversible. `mode = "explicit"` keeps research usable through the MCP tool
without allowing automatic 35B work. Do not use `auto` or `shadow` until the
protected 9B/embedding residency and foreground latency gate are proven.

## Hook Install

```sh
chronovisor hooks install --host all
chronovisor hooks inspect --json
```

Use the installer after changing host hook topology. It keeps non-wiki hooks in
place, replaces legacy Chronovisor script wrappers with direct dispatcher commands,
and refreshes Codex trusted hashes.

## Recall Logs

```sh
chronovisor-recall --recent 20
chronovisor-recall --feedback missed --prompt "..." --note "..." --ref <decision_id>
chronovisor recall-eval --json
chronovisor recall-eval --save-baseline
```

Explicit feedback is an optional diagnostic input, not an operating gate. The
auditor, pull-log attribution, and locally reviewed label path discover and
close normal false negatives automatically.

`recall-eval` builds a replay dataset from `recall/recall-log.jsonl` and
`recall/feedback.jsonl`, then reruns the current gate without writing new
decision logs. Use it before and after changing recall thresholds, fusion
weights, rewrite settings, or context style.

## Local Decision Replay Gate

```sh
# Make one immutable candidate config first. Its [decision_router] section
# must contain the intended production values and adoption_artifact = "".
CANDIDATE_CONFIG="$HOME/.chronovisor/runtime/model-lab/decision-router-candidate.toml"
CORPUS="$HOME/.chronovisor/runtime/model-lab/adoption-corpus-v64-lane26-20260723.jsonl"
ARTIFACT="$HOME/.chronovisor/runtime/model-lab/local-eval/adoption-v64-evaluator21-20260723.json"
chmod 600 "$CANDIDATE_CONFIG"

# Preflight the deterministic selection without replacing the durable corpus.
chronovisor-adoption-corpus \
  --config "$CANDIDATE_CONFIG" \
  --output "$CORPUS" \
  --dry-run

# Freeze the production-representative corpus at an explicit durable path.
chronovisor-adoption-corpus \
  --config "$CANDIDATE_CONFIG" \
  --output "$CORPUS"
chmod 600 "$CORPUS"

# Read-only inspection performs no inference.
chronovisor-local-model-eval \
  --input "$CORPUS" \
  --config "$CANDIDATE_CONFIG" \
  --dry-run
chronovisor-local-model-eval \
  --input "$CORPUS" \
  --config "$CANDIDATE_CONFIG" \
  --list \
  --limit 20

# A full run evaluates every exact context bucket in ascending order.
chronovisor-local-model-eval \
  --input "$CORPUS" \
  --config "$CANDIDATE_CONFIG" \
  --output "$ARTIFACT"
chmod 600 "$ARTIFACT"

# Resume only with the exact same candidate config, corpus, and output path.
chronovisor-local-model-eval \
  --input "$CORPUS" \
  --config "$CANDIDATE_CONFIG" \
  --output "$ARTIFACT" \
  --resume
```

The compiler reads the append-only
`~/.chronovisor/runtime/model-lab/replay.jsonl` without modifying it, adds only
deterministic production-contract cases needed for complete schema and context
coverage, validates the exact output bytes, and atomically installs the frozen
corpus with mode `0600`. Use a new versioned corpus path whenever replay source
selection, the schema/signature manifest, decision-semantics policy, effective
request-fingerprint policy, or candidate config changes; do not overwrite a
corpus after an evaluation has started. The same immutable candidate config
must be supplied to both the compiler and every fresh or resumed evaluator
invocation. A config change is a new evaluation identity, never a resumable
continuation. The current v64 identity uses artifact schema 12, evaluator
policy 21, decision-semantics policy 12, quorum-safety policy 1, action-
signature policy 5, effective-request-fingerprint policy 4, structured-
generation policy 3, lane-contract registry policy 10, lane-contract case policy
26 with source `deterministic_lane_contract_v26`, residency policy 2, and
`num_predict = 3072`. Evaluator policy 21 seals explicit deterministic seed 0,
hash-bound ingest repair option selection, host-only byte-exact materialization
before action signatures/quorum, and repair-attempt accounting into the
artifact identity.
Registry policy 10 is aggregate artifact/run identity only and is not
rendered into model requests. The model-visible prompt-contract version is
lane-scoped: 16 lanes are at version 8, `ingest_reconciliation` is at version
16, and `raw_replay_reconciliation` plus `recall_auto_apply` are at version 9.
Unchanged lane prompt
bytes remain stable, but effective-request fingerprint policy 4 intentionally
reseals every lane identity and the aggregate canonical case manifest.

When one lane's prompt, system policy, effect semantics, or request envelope
changes, increment only that lane's prompt-contract version, compile a new
versioned corpus, and run a fresh evaluator artifact. The changed per-lane hash
updates the aggregate manifest and artifact identity automatically; do not put
the registry version into every model request, because that would invalidate
all 19 lanes. Increment the registry policy itself only when the registry or
artifact identity contract changes. The fixed v64 baseline contains 100
canonical cases spanning all 19 model-backed lanes and all four executable
context buckets. Previous replay rows are included only when their independent
provenance, contract identity, expected effect, and action signature all match
the current policy. During compilation only, non-deterministic historical rows
with stale lane or request identity are counted and excluded; deterministic
contract fixtures, the frozen corpus, evaluation, and runtime loading remain
strict. The v64 corpus admits no stale historical rows.

Canonical fixtures must be reachable through the same deterministic preflight
as production. Entity-backfill missing, malformed, truncated, or alias-
incomplete proposals are therefore tested before inference; its model corpus
covers only semantic approval and rejection. Other page-mutation retry cases
use actual bounded/truncated request payloads rather than availability claims
embedded in untrusted page prose.

Ingest repair choices are host-bound selectors, not model-authored page bytes.
The preflight hashes each exact deterministic or semantic repair into one
`repair_option_id`; a model may return only that ID with `retry` and
`retry_required`. After schema and lane validation, the router resolves the ID
against the same preflight, removes it, materializes the exact trusted
`invalid_tags` and `replacement_operations`, and validates the result again
before its action signature enters quorum. Unknown IDs, mixed selector/arrays,
ambiguous options, terminal decisions with an ID, or any post-materialization
drift fail closed.
Repair-option policy 2 deliberately exposes no regex-derived or host-authored
semantic contradiction receipt. Every structurally valid one-tag option stays
visible, and the two local models must independently choose from the exact raw
and page bytes. Extra legacy receipt fields fail closed before inference.

The ingest review projection never sends full unchanged page bodies. It emits
every exact changed byte, hashes every omitted equal span, and binds each
generated operation bijectively to its prepared CAS postimage. Body hunks carry
up to 256 exact UTF-8 bytes of surrounding context on each side. Frontmatter-
only hunks do not duplicate large metadata such as `raw_keywords`; instead,
`page_identity` exposes complete exact identity nodes (`title`, aliases,
permalink, canonical id/slug) once and seals the full pre/post frontmatter by
byte length and SHA-256. A changed frontmatter node up to 512 bytes is shown
whole. Larger nodes retain the exact field name plus 96 bytes of local context
on both sides of the exact changed span, without copying neighboring metadata.
Repair projections refer to that hash-bound review operation rather than
copying the same context into every selectable repair.
If this complete deterministic projection still exceeds the fixed input or
context ceiling, the request fails closed before inference.

`chronovisor-local-model-eval --dry-run` validates and counts the compiled cases,
while `--list` prints redacted case metadata; neither performs inference. A
full evaluation uses the candidate local decision router and atomically
checkpoints a resumable, redacted artifact containing hashes, labels,
validation diagnostics, latency, and aggregate metrics, never prompts or
literal model responses. It starts from cold candidate runners and processes
the required 32K, 64K, 96K, and 112K context buckets in
ascending order with larger-context reuse disabled. A resume revalidates the
candidate config and corpus identities and preserves this exact-bucket mode;
identity drift fails closed. Adoption
remains false unless all usable cases (at least 100), every usable role and
decision class, every current production schema (at least five cases per schema
hash), schema-validity, pair-validity, agreement, majority resolution,
historical-effect match, and unsafe-flip thresholds all pass. Pair agreement
means the first two models produced the same decision signature; a
deterministic safety lower-bound is reported separately and never inflates
pair or three-model majority quality. The production
schema manifest is code-defined, so adding a new local decision schema blocks
future model adoption until replay evidence exists for it. `--offset` or
`--limit` is therefore a smoke-test facility only and cannot produce an adopted
artifact. Legacy rows whose prompts are exactly 50,000 characters without a
truncation marker, plus all rows explicitly marked `prompt_truncated=true`, are
excluded and counted by reason because their leading instructions cannot be
proven intact.

Every successful routine `DecisionRouter` result appends one replay case using
the already-completed local votes; this collection step performs no additional
inference. The fixed 50,000-character evidence cap applies to the exact prompt
sent to the model. For ingest repair, a larger full host-bound prompt is still
retained losslessly when only its sealed host-only sidecar exceeds the cap; the
row records the effective model prompt's length and SHA-256 separately and is
not marked truncated. A model-visible prompt over the cap, or an over-cap
system prompt, is retained only as a marked tail with
`prompt_truncated=true` and is excluded from adoption evaluation. Production
rows retain the lane-bound pre-semantic system for exact rebinding, plus the
effective model system, lane effect, request fingerprint, and prompt/system
lengths and SHA-256 values actually sent. The strict loader recomputes each
present field; stale or tampered evidence is rejected. Only the adoption-corpus
compiler may count and exclude non-deterministic historical rows whose old
request identity can no longer be reconstructed, and an all-stale historical
file still compiles the canonical contract-only corpus. Production quorum,
replay recording, and evaluation all use the same schema-derived action
signature. Exact approved
mutation targets remain action-bearing fields. `semantic_checks`, however, are
diagnostic authorization evidence and are intentionally excluded from that
signature: after action agreement, the agreeing votes are conservatively
AND-merged per check. Any false check changes an otherwise approved result to
`needs_retry` and clears approved mutations, so matching action signatures can
never hide a failed semantic precondition.

The model triplet in `[decision_router]` remains the explicit
bootstrap/current policy. To nominate a replacement after a full passing run,
set `decision_router.adoption_artifact` to that artifact. Runtime revalidates
the artifact schema, evaluator policy, sealed per-case evidence, recomputed
metrics/gate, full-corpus coverage, fixed minimum thresholds,
all gate checks, and evaluated model digests before switching all three roles
atomically. It also reopens the immutable source corpus and compares the exact
Ollama engine version, model digests, and quantization identities. A missing,
modified, or drifted nominated artifact never partially switches roles:
enabled semantic lanes quarantine before inference, while explicit shadow
evaluation may continue on the bootstrap triplet without creating trusted
adoption labels.

Decision inference also uses memory-aware residency. The router computes the
full structured-session token requirement (system prompt, user prompt, schema,
two possible JSON-repair turns, output reservation, and safety margin), rounds
it to the smallest executable configured context bucket, and uses
calibrated Ollama `/api/ps` footprints measured at that bucket plus host
reclaimable memory to choose whether one, two, or three runners fit.
Production applies monotonic context hysteresis: a larger runner already in
memory is reused when its measured footprint still fits reserved capacity and
its context is within the configured ceiling. It grows when necessary but is
not shrunk merely because the next request is smaller, avoiding runner flap.
After every vote, the router records the actual runner size and context reported
by `/api/ps`; admission therefore counts the larger measured allocation rather
than the smaller requested bucket. Missing observations do not overturn an
otherwise safe decision, but they fail closed as adoption evidence. Likewise,
an observed larger context cannot manufacture coverage for a requested smaller
bucket. The evaluator starts cold and executes cases in ascending bucket order,
then explicitly disables larger-context reuse so it measures every exact bucket
once. Production remains grow-only; evaluation's exact mode is isolated and
resets surviving candidate runners on both fresh and resumed runs. The corpus
compiler and evaluator both reject a full gate before inference when any bucket
lacks a planned case. The evaluator also preserves Ollama's per-turn
prompt/output token counts, so production truncation checks and replay-gate
checks use the same transport contract.
Measurements persist by exact model digest, context bucket, Ollama version,
platform, and daemon process epoch; a daemon restart invalidates stale
measurements, while caller-shell environment differences do not. An
uncalibrated model/context pair bootstraps exactly one runner only when its
conservative 2x estimate fits currently reclaimable capacity after reserve.
Unrelated resident models are never counted as reclaimable; a failed resource
probe or a role that cannot fit alone quarantines before inference. Runner eviction is
verified under a thread/process-shared lease; a failed eviction quarantines the
decision rather than skipping a vote. Adoption artifacts must contain passing
evidence from every context bucket before this policy can become active.
Increasing the resident count also requires spare capacity of at least 2 GiB or
10% of the proposed resident set, whichever is larger, so small memory changes
do not flap repeatedly between two and three runners.

After the v64/evaluator-policy-21 artifact reports `adopted=true`, nominate it in
`decision_router.adoption_artifact`, revalidate it through a fresh runtime, and
promote all 19 model-backed semantic lanes from `shadow` to `enabled`. Together
with the five deterministic/guarded lanes, the post-adoption production state
is 24 enabled and 0 shadow. If artifact validation later fails, enabled semantic
lanes quarantine before inference rather than falling back to bootstrap models.

## Ingest Model

The page-generation path reads `[ingest]` from `~/.chronovisor/config.toml`. The
production profile selects the smallest safe 32K/64K/128K/256K context bucket
for the complete request envelope. A compatible larger resident Ornith 35B
runner is reused, so backlog processing grows monotonically rather than
shrinking and reloading between raws. The 256K bucket evicts unrelated Ollama
runners before admission. Inputs that cannot fit fail closed; deterministic
transcript captures are projected to complete, byte-exact user/assistant text
while the lossless raw remains on disk. Every recognized transcript projection
is first materialized as one verified content-addressed child; oversized
projections fan out on record boundaries, with only a single oversized record
split at UTF-8 boundaries. Tool/event-only captures receive a durable
content-addressed no-op receipt. A parent is retired only after every child or
no-op artifact passes exact read-back verification.
All default-transport `LocalStructuredSession` lanes use the same measured
residency broker as ingest. The broker holds one exclusive process-wide lease
for the complete initial-plus-repair session, selects the smallest complete
context bucket, and reuses a compatible larger resident runner. Lint, tagging,
recall, and correction therefore cannot race the same Ornith tag at different
`num_ctx` values and force runner reloads.

Page apply and raw retirement are joined by a durable completion ACK under
`runtime/raw-completion-acks/`. The ACK binds every logical source filename and
byte hash to the completed job outcome and observed page postimages. It is
published and read back before `processed_raw_files` changes. If the state write
is interrupted, the next tick validates the ACK and performs only raw retirement;
it does not repeat projection, model inference, or page mutation. Recorded
postimages prove the publication-time outcome but are not a lock on later,
legitimate updates to the same page.
If the process stops after a reviewed page effect but before the ACK exists,
the next attempt checks the content-bound terminal proposal/review before
triage. A fully applied page postimage, or a current-authority confirmed no-op,
can then complete the job and publish the ACK with zero model calls. Incomplete,
legacy, stale-authority, malformed, or changed proofs re-enter the ordinary
fail-closed path and can never receive this shortcut.

One ingest consensus outcome is terminal without becoming an operational
failure. If all three configured voters return schema-valid, pairwise-distinct
decisions under the currently validated adopted-artifact SHA, no two-vote quorum
exists and the exact source raw enters semantic defer. Its bytes remain
unchanged in `raw/`; it is excluded from self-heal, frontier repair,
explicit/automatic raw replay, and cooldown reopening. It becomes runnable
exactly once when the router fully validates a different adopted-artifact SHA.
A merely byte-different, partial, stale, or otherwise invalid nominated artifact
fails closed. Runtime, transport, capacity, schema, and other operational
failures do not use semantic defer and continue through the separate bounded
repair queue.

Changing the ingest model does not require a semantic reindex unless
`[embedding].model` also changes.

Before generation, ingest now runs a conservative search-before-create gate.
High-confidence duplicate `create` ops are rewritten to `update` ops when an
existing active knowledge page has the same page id, same title, near-identical
title/page id, or a matching search result. Reference pages are not considered
update targets.

After apply and embedding refresh, ingest read-backs changed pages with their
`recall_questions`, `summary`, or title. Failures are non-fatal and are logged
to `~/.chronovisor/runtime/ingest-read-back-failures.jsonl`.

Successful ingest also appends a lightweight claim seed to
`~/.chronovisor/claims/claims.jsonl`. The current page files remain the source of
truth, but the append-only ledger gives future event-sourced memory work a
machine-checkable trail.

## Working Memory

`system/current-state.md`, `system/user-profile.md`, and
`system/lessons-learned.md` form a fixed core-memory allowlist. Codex/Claude
Code prompt hooks inject bounded excerpts as one `[WORKING_MEMORY]` block even
when the normal recall gate decides `none`. Arbitrary pages cannot enter this
layer. System notifications and internal prompts remain filtered before this
path.

The persistent ingest drain probes Ollama before starting semantic work. If it
is unavailable, Raw capture remains durable, the drain writes
`runtime/ingest-liveness.json`, health becomes `alert`, and no failing ingest
job is started. The watcher retries on its normal interval and records the
recovery transition before automatically draining the backlog.

## Sensitivity Tiers

Pages can set `sensitivity: high` in frontmatter. Career-folder pages infer
`high` in the index even before frontmatter is backfilled. Recall cards show
the sensitivity annotation next to the freshness annotation, and `chronovisor
health` reports the tier distribution. In work-project CWDs, high-sensitivity
pages are filtered unless the prompt explicitly asks for career/interview style
context.

## Entity Registry

```sh
chronovisor entities init
chronovisor entities backfill --dry-run
chronovisor entities backfill --limit 100
```

The registry lives at `~/.chronovisor/entities/registry.json`. Ingest patches
`entities: [...]` frontmatter on created/updated knowledge pages using known
aliases such as MHI/三菱重工, KHI/川崎重工, Codex, Ollama, Qwen, and Gemma.
Entity backfill skips reference pages by default.

## Knowledge Quality Queues

```sh
chronovisor-duplicate-review --write
chronovisor_check
chronovisor_apply
```

Pages with `type: reference` are excluded from default search, lint, duplicate
review, and recall metadata backfill. `car-spec/` pages infer this type even if
older files are missing the field; explicit `folder="car-spec"` searches still
include them.

`chronovisor_check` returns a compact issue summary plus a bounded sample instead of
dumping every issue. `chronovisor_apply` writes remaining non-auto-fixable lint work to
`~/.chronovisor/review/lint-repair-queue.jsonl`, split into safe-auto-fix,
heavy-model-batch, review, and monitor lanes.

`chronovisor-duplicate-review --write` builds
`~/.chronovisor/review/duplicate-candidates.jsonl` from title and embedding similarity.
The file is an observable candidate ledger. Sleep first handles deterministic
safe cases, then sends ambiguous pairs to the local decision router; agreed
supersession atomically marks the loser `status: deprecated` with
`superseded_by: <winner>`. Model disagreement is quarantined, and no human
review queue or frontier fallback is required.

## Raw Replay

```sh
chronovisor raw-replay --since 2026-07-01 --limit 100
chronovisor raw-replay --since 2026-07-01 --limit 1 --run
```

Without `--run`, replay writes `~/.chronovisor/review/raw-replay-queue.jsonl`.
With `--run`, selected raw files go back through the normal ingest path, so
search-before-create and read-back verification still apply.
Active artifact-bound semantic defers are excluded from explicit replay,
automatic replay signals, existing queue rows, cooldown reopening, and crash
reconciliation. A different adoption-artifact SHA releases one deduplicated
candidate back to the ordinary ingest queue only after full router validation;
elapsed time or an invalid file change alone never releases it.
Read-back misses caused only by ranking (`not-in-top-results`) stay in the
lighter query-hint repair lane; raw replay is reserved for structural ingest,
metadata, quarantine, and integrity failures.
The query hint is still a production ranking change: the local failure signal
only creates an exact proposal, and local consensus bound to the page hash
is durably persisted before the hint is written. Rejection is terminal for the
same evidence; transient or low-confidence decisions retry autonomously.

Before ingest starts, replay durably records a `running` row with job,
attempt, content hash, and start time. The ingest `on_complete` callback then
fsyncs a whole-raw completion journal before queue acknowledgement. Partial
ingest is terminal `completed_partial` so already-successful operations are
never replayed. If a process dies in the narrow unprovable window, the row
becomes `indeterminate`: local consensus chooses processed, safe replay,
or quarantine. It is never blindly retried and never becomes a human content
decision.

## Raw Archive

Raw is the immutable evidence/rebuild layer, not the normal Recall search
database. Pages and verified semantic projections remain the synchronous
search path. The archive has one date-based hierarchy and no hot/archive tier.

```sh
# Cheap inventory; reports logical/stored bytes and open/sealed counts
chronovisor raw status --json

# Commit/range verification; --full streams every sealed byte back
chronovisor raw verify --full --json

# Preview yesterday-and-older v2 segments, then seal a bounded batch
chronovisor raw seal --limit 4 --json
chronovisor raw seal --limit 4 --apply --json

# Restore one logical Raw or a whole sealed segment without mutating the store
chronovisor raw export save-<transaction>.md /safe/output/raw.jsonl
chronovisor raw restore ~/.chronovisor/raw/YYYY/MM/DD/<segment>.manifest.json /safe/output/segment.jsonl
```

Existing flat transcript Raw uses a separate two-step migration. Eligibility
requires the logical Raw ID to be in the durable processed ledger and its file
date to be older than today. The first apply is shadow-only and retains every
flat source:

```sh
chronovisor raw migrate --json
chronovisor raw migrate --shadow --json
chronovisor raw verify --full --json
```

After an observation window and restore drill, the same command may remove
only sources already reproduced byte-for-byte from a verified archive:

```sh
chronovisor raw migrate --apply --remove-source --json
```

Never use `--remove-source` as a backup substitute. The zstd object and the
flat file share the same machine until a separate backup copies them elsewhere.
Manifests contain relative locators, so moving the complete Raw root or sealed
date folders does not depend on the old absolute path; run `raw verify --full`
after any move.

## Memory Integrity Eval

```sh
chronovisor memory-integrity --limit 100
chronovisor-memory-integrity --limit 100 --json
```

This is the first E1/W7 write-side eval. It samples raw captures, derives a
deterministic expected-term query, checks the claim ledger and search footprint,
and writes `~/.chronovisor/eval/memory-integrity-latest.json`. The dashboard health
panel uses this when available.

## Cofire Graph

```sh
chronovisor cofire --limit 5000
chronovisor-cofire --min-count 2 --json
chronovisor prefetch --limit 5000
```

Recall logs now build a co-fire graph at `~/.chronovisor/recall/cofire.json`.
Search graph expansion consumes those edges alongside wikilinks/backlinks, so
pages that repeatedly appear together can reinforce each other before a
human-curated graph exists. Prefetch cache writes
`~/.chronovisor/recall/prefetch.json` from recent recall episodes and is checked
before normal search context assembly.

## Sleep Cycle

```sh
chronovisor sleep --dry-run --json
chronovisor-sleep --raw-limit 100 --eval-limit 100
```

The sleep cycle is the single bounded convergence driver. It snapshots
`~/.chronovisor`, rebuilds co-fire/prefetch/retention artifacts, runs memory integrity,
and then drains small batches from lint repair, raw replay, read-back repair,
search-label review, recall auto-apply/self-heal, duplicate, and orphan-link
lanes. Weekly calibration and search self-tune also run here. Every decision or
queue lane has a stable key, retry/backoff limits, a terminal quarantine, and a
shared cycle time/call/mutation budget; artifact writes are charged to the same
budget. Legacy budget fields named `frontier` count local structured-review
calls for compatibility and do not authorize a frontier process. One lane failure
produces `status=partial` while the others continue. A single-flight lock
prevents overlapping scheduled/manual cycles. `--dry-run` is byte-for-byte
read-only, including search indexes and caches, and does not invoke model
reviewers. A zero `--eval-limit` skips integrity and label evaluation instead
of expanding to an unbounded corpus scan.

The daily Sleep LaunchAgent also runs once when it is newly loaded, so an
installation or product rename after the calendar boundary cannot leave the
watchdog without an execution receipt until the following day. The compact
sleep history row is operational state rather than a page mutation and is
always written even when semantic/artifact mutation budgets are exhausted.

The 30-minute `chronovisor-converge` worker remains lighter than Sleep: it does
not rebuild derived artifacts, run broad evaluation, or seal Raw. It drains
bounded existing correction, duplicate, lint, and orphan-link work under a
shared 15-minute local-model budget. Durable orphan work is ordered oldest
first, preventing terminal items at the front of the corpus from collapsing
throughput. Watchdog lint backlog counts unresolved detector `issue_key`
values; completed append-only convergence history is reported separately and
does not keep the alert permanently active.

Installed MCP, hook, dashboard, ingest-drain, sleep, and watchdog entry points
resolve the pushed GitHub package through `uvx`; the local checkout remains the
explicit code-repair target, not an implicit production import path. Long-lived
services refresh the package on restart.

Sleep history is stored as a non-recursive, 1,000-row summary rather than full
nested cycle payloads. Scheduled sleep writes a compact text report, while the
15-minute watchdog keeps its latest state and bounded history in `autonomy/`
and sends routine stdout to `/dev/null`; stderr remains logged.

Legacy maintenance scripts may still produce read-only diagnostics, but their
heuristic/local-model page mutation paths are fail-closed. Garbage cleanup,
tag/link/recall-metadata backfill, broken-link rewriting, and model-selected
folder moves must enter bounded convergence lanes; they cannot write knowledge
pages directly.

## Wiki Snapshots

```sh
chronovisor snapshot "before manual repair"
chronovisor-snapshot "before manual repair"
```

`~/.chronovisor` is initialized as its own git repository on first snapshot. Scheduled
lint auto-fix and MCP `chronovisor_apply` snapshot before changing files, giving
self-heal and repair work a rollback point independent of the code repository.

## User Content Corrections

```sh
chronovisor-content-correction --host codex --session-file /path/to/session.jsonl --capture --run-due
chronovisor-content-correction --host claude-code --session-file /path/to/session.jsonl --capture --run-due
chronovisor-content-correction --host codex --hook --capture-only
```

When enabled, the Stop hook durably enqueues the dedicated `--capture-only`
worker alongside deterministic raw capture. That worker only binds completed
turns with an explicit deterministic correction signal and appends convergence
items; normal adjacent turns only advance its cursor. It never calls a model,
ingest, mutation, or frontier process. The existing `run_due` path is drained
later by the bounded sleep/local convergence worker. An explicit user
correction is bound to the preceding complete turn. Legacy
`unfiltered_completed_turn` backlog is rejected in one deterministic bulk
migration before any model work. If that legacy path already produced an
applied `page_ignored` row, the migration preserves the append-only history and
adds a `page_ignored_retracted` record bound to both the exact convergence item
key and the exact feedback-row SHA-256. Ranking and replay consumers exclude
only that bound row; prompt- or page-name matching is never used.
Recall provenance must match the exact
prompt hash, host, session, and turn-time window; injected pages and pages read
during that recall decision form the only mutation candidates. A durable
cursor keyed by host/session/transcript tracks the last completed assistant
line so retries do not replay already-enqueued correction turns.

The local router classifies page error, outdated claim, wrong retrieval,
assistant misquote, ambiguity, unattributed, or no correction. Ornith 35B and
GPT-OSS 20B must agree, or Gemma 4 26B supplies a tie-break vote. A locally
confirmed wrong retrieval writes `kind =
"page_ignored"` with only its explicit `negative_pages` subset; the remaining
pages from the recall decision are not demoted. Non-page classifications do not
mutate wiki content. Invalid structured output receives at most two targeted
repair turns; exhaustion or disagreement is quarantined without frontier
escalation.

Normal pages plus the user-memory system pages `user-profile`, `current-state`,
and `lessons-learned` are correctable when exact recall provenance names them.
Operational system files remain outside the mutation boundary.

Content mutations require unique exact old spans, verbatim user evidence,
protected literal grounding, local agreement on immutable before/after
hashes, and a per-page CAS immediately before each replace. The CAS runs under
the writer lock shared by correction, ingest, lint, entity, and orphan-link
repairs; a partial multi-page failure rolls back only bytes still owned by the
correction.

The agreed structured payload is persisted as a review artifact before page
bytes change. On restart, a matching correction marker and artifact allow the
lane to resume refresh/audit work without repeating the local decision for
the same patch. Terminal `applied` additionally requires successful
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
local decision; it is never trusted or silently discarded. Historical artifact
fields containing `frontier_*` are compatibility names only.

When a historical raw capture is known to be false, keep its body for audit and
set `raw_status: retracted` in frontmatter. Normal ingest, explicit replay,
automatic replay signals, and already-queued replay all exclude it.

## Audit and Auto-Apply

```sh
chronovisor-recall-audit --host codex --hook --audit-read
chronovisor-recall-auto-apply --dry-run
chronovisor-self-heal --auto-apply-errors --auto-apply-error-threshold 3 --dry-run
```

Auditor feedback uses `kind = "missed_candidate"` and source `auditor` for
false negatives. Precision labels use `kind = "injection_used"` or
`kind = "injection_ignored"`. Audit is invoked explicitly or by bounded
convergence; the Stop dispatcher no longer schedules it.

Auto-apply treats the local auditor as a proposal source only. Alias, query
hint, and page-tag actions (including few-shot-derived hints) require a
local-consensus approval bound to the exact proposal and current page hash. The
approval artifact is persisted and read back before the mutation, then reused
after budget deferral or a crash. Active recall-policy candidates likewise
require local consensus. Legacy `frontier_mode` and `frontier_*` fields preserve
old queue/schema compatibility but do not select a frontier model.

Repeated `recall/auto-apply.jsonl` errors are promoted into self-heal packets
after the configured threshold. The live auto-apply path accumulates repeated
errors across runs and starts the local repair loop. Routine JSON, content,
semantic, and policy failures remain local and are rejected, retried, or
quarantined; they cannot enter frontier code repair. The `--dry-run`
self-heal command reads the log and reports candidate clusters without writing
packets or state.

## Search Ranking Review

```sh
chronovisor-eval --build-label-queue
chronovisor-eval --report --failure-index
chronovisor-eval --self-tune
chronovisor-eval --ci --ci-variant hybrid-current --min-recall-at-5 0.80
```

`--build-label-queue` writes auditor/search candidates to
`recall/search-label-queue.jsonl`; it does not promote rows into
`search-golden.jsonl`. Sleep sends a bounded batch to the local decision router;
approved labels are promoted automatically, rejections are terminal, and
uncertain/retry results back off before quarantine after three passes.
The legacy `--build-golden` spelling is a compatibility alias for building the
same label queue; it can no longer overwrite the authoritative golden file.
Evaluation, CI, and self-tune load only rows carrying `reviewed: true`.
`--failure-index` records missed expected pages with channel candidates and a
reason code. Weekly self-tune evaluates dev weights against an independent
locked-test set, asks local consensus for the final veto, and atomically
writes `recall/search-policy.json` only after both gates pass.

The optional Hugging Face reranker is disabled in the normal local profile.
Ranking still runs through BM25 + semantic fusion; enable `[search.reranker]`
only for explicit MCP `chronovisor_search` after a reviewed locked-holdout experiment.
The tuned starting point is `top_n = 10`, `max_length = 384`,
`batch_size = 10`, and `weight = 1.0`; the hook path remains unchanged.

## Calibration

```sh
chronovisor-recall-calibrate --dry-run
chronovisor-recall-calibrate
chronovisor-recall-calibrate --rollback
```

Calibration trains on older labeled rows and validates on the newest holdout
slice. It writes `recall/calibration.json` only when holdout improvement exceeds
the configured minimum, and records the old artifact in
`recall/calibration-history.jsonl` for rollback. Sleep schedules this weekly
with bounded samples/recomputed features and a local-consensus veto. The public
calibration CLI uses the same local decision boundary even if a legacy caller
passes `frontier_mode=off`; an approval bound to the exact active-policy hash is
persisted before a CAS-protected policy write.

## Human Boundary

Normal content, ranking, JSON repair, and policy decisions converge without a
human or frontier model.
`human_required` is reserved for deterministic external-authority failures:
OAuth/authentication, billing or quota changes, or Keychain/secret-store
permission. These failures go directly to the user boundary and are not sent to
any model. Missing tools or models, ambiguity, low confidence, schema errors,
and model disagreement use autonomous retry and cooldown quarantine instead.
The exact ingest three-valid/three-distinct outcome is handled separately as an
artifact-bound semantic defer: it is not a cooldown quarantine and is reopened
only by a different fully validated adopted-artifact SHA. Operational runtime
failures remain in their repair queue.

## Exceptional System-Code Repair

Frontier/Codex execution is a separate repair plane, not the highest tier of a
routine review ladder. The system incident supervisor can create an eligible
packet only for a true system-code failure with all of the following evidence:

- a supervisor-owned deterministic reproduction receipt, regardless of how many
  logical inputs share the failure fingerprint; cross-input clustering remains
  observability and is never Frontier eligibility by itself;
- at least two failed deterministic local repair/recheck attempts;
- a bound failing pytest node and reproduction artifact; the executable command
  is derived by the host as `uv run pytest -q <nodeid>` and arbitrary receipt
  argv is rejected;
- no authentication, billing/quota, Keychain, credential, content, semantic,
  structured-output, or model-disagreement failure class.

No operational runtime failure qualifies by raw-count clustering alone.
Artifact conflict, capacity, internal, schema, and legacy classes all require a
deterministic reproduction receipt. Packet, supervisor state, artifact, command,
failing test, and reproduction digest are cross-bound before the repair job can
enter the frontier guard.

`chronovisor-self-heal` additionally requires the explicit
`--enable-frontier-repair` capability. `RepairIncidentEvidence` then passes a
durable, process-wide single-flight guard with fingerprint cooldown and a
default global limit of one started attempt per 24 hours. Only after that guard
does `run_frontier_review()` start one Codex repair attempt. Starting the
process consumes the budget; inspection and reservation do not. There is no
rescue fan-out and no second remote attempt.

## Recall Question Backfill

```sh
scripts/backfill_recall_questions.py --dry-run
chronovisor-sleep --raw-limit 100 --eval-limit 100
```

The legacy script is diagnostic-only. Recall-metadata proposals now enter the
scheduled sleep pipeline, where local consensus binds any accepted change
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
