# Hooks

`llm-wiki-hook` is the public hook entry point.

## User Prompt

```sh
llm-wiki-hook --host codex --event UserPromptSubmit --hook
llm-wiki-hook --host claude-code --event UserPromptSubmit --hook
```

This runs the recall gate synchronously and prints host-native hook output. The
entire path has one wall-clock deadline (`recall.total_timeout_ms`, 4000 ms by
default), not a separate unbounded timeout per stage. The remaining budget is
propagated to query rewrite, semantic embedding, search, and the evidence
judge. A process-level timer is the final boundary around the complete hook.
The dispatcher reserves 250 ms inside that configured deadline for
preemption, process cleanup, rendering, and logging, so the Recall engine gets
3750 ms under the default contract. The installed host command allows 7000 ms,
leaving separate bounded startup/cache resolution headroom around the 4000 ms
hook deadline.

Recall is strictly fail-open for the host. The primary path reserves 600 ms of
its deadline for a model-free fallback. A soft deadline first degrades to
allowlisted L1 memory plus BM25; if a non-cooperative call reaches the outer
timer, no second pass is started and the hook immediately fails open. Thus the
primary path plus its one internal fallback stays inside the same 4000 ms
Recall budget. Config corruption or a fallback failure still prints a host
no-op, exits successfully, and lets the user prompt continue.
After two degraded or failed Recall runs, the default circuit breaker disables
rewrite, semantic search, and the local judge for 60 seconds while BM25 remains
available. A successful normal run resets the breaker.

The synchronous hook never starts `wiki_research`, Deep Recall, Web search,
Web fetch, the 35B planner, or the cross-encoder reranker. Those lanes are
explicit/background or Sleep work. Foreground Recall announces a short-lived
sync marker before local inference; a running research child observes that
marker and is cancelled/deferred within its preemption budget. The scheduler
tracks the isolated model worker separately from non-model research phases: a
foreground prompt kills only an overlapping stateless model worker, while
search/checkpoint work never consumes the synchronous resource-wait budget.
Automatic and shadow 35B research are rejected unless protected model capacity
has been proved explicitly.

Always-on memory and automatic Recall have independent budgets. L1 admits only
the fixed system-page allowlist `current-state`, `user-profile`, and
`lessons-learned`; arbitrary pages cannot enter it. Both layers are rendered as
non-executable JSON data, with memory content quoted as JSON strings rather
than executable-looking prose. Delimiter-like text inside a page is
neutralized. The combined
context is assembled from whole blocks only; a block that does not fit is
omitted rather than cut into ambiguous partial syntax. See
[Recall Orchestration](recall-orchestration.md) for the full contract.

## Stop

```sh
llm-wiki-hook --host codex --event Stop --hook
llm-wiki-hook --host claude-code --event Stop --hook
```

Stop durably enqueues a host save job and, when
`hooks.stop.content_correction = true`, one coalesced content-correction
capture job for the same session. It immediately prints `{}` and never starts
a detached process, model, ingest, mutation, frontier review, audit, or recall
improvement inside the hook process.

Repeated Stop events coalesce only while an equivalent capture job is active
and the payload contains the same stable session ID or transcript path. A Stop
payload without stable identity is never coalesced with another session. If a
Stop arrives while the worker is running, the job is marked for one more pass;
after completion a new Stop may enqueue again. Duplicate bytes are still
prevented by the save cursor and durable receipts, not by assuming the host
fires Stop exactly once.

The save worker deterministically captures every uncaptured transcript delta
after an exact per-session cursor. A delayed append or a previously failed job
is picked up on the next run. In the legacy layout, oversized deltas retain the
compatible reassemblable fragment format. In v2, original complete JSONL lines
are appended directly to a date-partitioned segment, so even one oversized
record needs no base64 fragmentation. Data and commit journal are separately
fsynced and read back; the cursor advances only with that durable logical Raw
receipt. Compression is asynchronous and can never run in the hook or save
critical path. The normal save path does not ask a model whether a turn is worth
saving and does not call any local or frontier model. Only after the save job
records a successful durable receipt does the same background-state transaction
enqueue one Recall audit candidate. The audit is therefore downstream of
durable capture and never runs inside Stop.

The correction capture worker appends turns with either a deterministic
explicit-correction signal or a narrow bare denial/contrast signal backed by
exact preceding-turn Recall provenance and a real candidate page. Ordinary
follow-ups advance the durable cursor without entering the semantic queue.
Classification and resolution happen later in the bounded local sleep worker.
Detection quality is measured with a versioned golden/holdout corpus:

```sh
llm-wiki-content-correction-eval
```

Semantic work is handled later by bounded convergence workers. When a routine
lane needs a structured decision it uses local consensus: Ornith 35B primary,
GPT-OSS 20B challenger, and Gemma 4 26B only as a tie-breaker. Invalid JSON gets
at most two targeted repair turns in the same local session. Failure or
disagreement is never escalated to a frontier model. For ingest specifically,
three valid, pairwise-distinct decisions bound to the currently validated
adopted-artifact SHA terminally defer the immutable source raw. That raw
receives no self-heal or cooldown replay and is eligible again only after the
router fully validates a different adopted-artifact SHA. A changed but invalid
nomination fails closed. Operational runtime failures stay in the separate
repair queue.

## Legacy Wrappers

These scripts remain for existing host settings:

- `scripts/codex_recall_hook.sh`
- `scripts/codex_wiki_save_hook.sh`
- `scripts/codex_recall_audit_hook.sh`
- `scripts/claude_code_recall_hook.sh`
- `scripts/claude_code_wiki_save_hook.sh`
- `scripts/claude_code_recall_audit_hook.sh`

Wrappers may still call the dispatcher with `--only save`, `--only audit`,
`--only correction`, or `--only improve`. The save selection enqueues raw
capture, and the correction selection enqueues only the dedicated
`--capture-only` worker. Audit and improve are deprecated compatibility no-ops;
their replacement is the save receipt's asynchronous audit candidate and the
bounded convergence worker. These no-op selections are scheduled for removal
after 2026-10-01. No selection starts semantic review in the Stop process.

## Install

```sh
llm-wiki hooks install --host codex
llm-wiki hooks install --host claude-code
llm-wiki hooks install --host all
llm-wiki hooks install --host all --dry-run --json
```

The installer replaces only existing LLM Wiki hook commands, preserves unrelated
host hooks, and writes direct `llm-wiki-hook` entries. For Codex it also updates
the matching trusted hash entries in `~/.config/codex/config.toml` and removes
stale LLM Wiki trust entries from the old split Stop hooks.

Names containing `frontier` in older hook settings or queued artifacts are
compatibility names, not proof that a frontier model is running. Actual Codex
execution exists only in the exceptional system-code-repair plane after strict
incident evidence passes its durable guard; a Stop hook can never enter it.

## Inspect

```sh
llm-wiki hooks inspect
llm-wiki hooks inspect --json
```

This lists detected Codex and Claude Code hook entries, labels current,
legacy-wrapper, and deprecated-no-op commands, emits migration warnings for the
last category, and computes Codex-style canonical hook hashes for inspection.
