# Recall Orchestration

Chronovisor uses a layered Recall path. The layers increase cost and authority;
they do not weaken the store's provenance, review, or adoption gates.

## Layers

1. **Always-on state (L1)** injects bounded user profile, current-state, and
   rule material when the prompt needs it. It has its own character budget.
2. **Automatic Recall (L2)** searches before the answer and injects only
   high-confidence page cards. It is bounded by the synchronous hook deadline
   and a separate context budget.
3. **Model-directed Recall (L3)** uses `chronovisor_search` and `chronovisor_read` when the
   initial context is insufficient. The model should forward the injected
   `decision_id` and `session_id`, then, before finishing the answer, call
   `chronovisor_recall_used` only for pages that materially influenced it. The
   tool validates the turn identity immediately, treats retries as idempotent,
   and merges later newly used pages into the same decision episode, so an
   acknowledged receipt is guaranteed to reach the label factory exactly once.
4. **Deep Recall (L4)** is an explicit, asynchronous investigation across
   related pages, prior conversations, and lossless Raw evidence. It is not
   allowed on the prompt hook's critical path. The current synchronous
   implementation stops at L3; Raw-crossing Deep Recall remains a distinct
   bounded worker rather than an implicit search fallback.
5. **Post-answer improvement (L5)** begins only after a save job has committed
   its durable receipt. That success transaction enqueues an audit candidate;
   later convergence workers may propose additive query hints, aliases, or
   tags, all behind Replay/Holdout and local-consensus adoption gates.

## Synchronous Contract

The default total deadline is 4000 ms. The primary path shares 3400 ms across
rewrite, embedding, search, and judging, reserving the final 600 ms for
allowlisted L1 memory plus deterministic BM25 fallback. The outer hook timer
covers code that cannot cooperate with remaining-budget checks; deployment
headroom allows one separately bounded fallback after a hard timeout. A durable
circuit breaker opens after two degraded or failed runs, disables
rewrite/semantic/judge for 60 seconds, and keeps BM25 available.

L1 and L2 are serialized as non-executable JSON strings. L1 is restricted to
`current-state`, `user-profile`, and `lessons-learned`. Page text cannot issue
commands, close its own context block, or silently consume the other layer's
budget. The combined output includes only complete blocks.

## Decision Trace

Recall telemetry uses four different stages:

- `returned`: a page appeared in search results.
- `read`: the model fetched a page.
- `injected`: the automatic hook included a page card.
- `used`: the model explicitly reported that a page materially informed the
  answer through `chronovisor_recall_used`.

Only `used` is positive supervision. Search and read activity alone never teach
the system that a page was helpful. Exact `decision_id` matching is preferred;
legacy fallback requires the same session and a bounded time window.

## Safety Boundary

Raw capture remains lossless. Content changes remain provenance-bound and use
CAS, read-back, local consensus, and fail-closed semantic lanes. Recall policy
changes remain subject to replay, locked holdout, and adoption checks. Frontier
execution remains limited to validated system-code incidents and is not a
Recall fallback.
