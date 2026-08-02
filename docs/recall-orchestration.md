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
   acknowledged receipt is guaranteed to reach the telemetry ledger exactly
   once. A `used` receipt proves use, not answer improvement, so it is never a
   positive learning label by itself.
4. **Deep Recall (L4)** is an explicit, asynchronous investigation across
   related pages, prior conversations, and lossless Raw evidence. It is not
   allowed on the prompt hook's critical path. The current synchronous
   implementation stops at L3; Raw-crossing Deep Recall remains a distinct
   bounded worker rather than an implicit search fallback.
5. **Post-answer improvement (L5)** begins only after a save job has committed
   its durable receipt. A fresh `saved` receipt queues both an audit candidate
   and privacy-safe answer capture; a `recovered` receipt queues answer capture
   only and does not duplicate the audit. In both cases the exact worker
   receipt, rather than the original hook payload or a latest-session lookup,
   is forwarded to answer capture. Stop never runs a model, scorer, or replay.
   Later offline workers may evaluate preregistered field-on/field-off pairs or
   propose additive query hints, aliases, or tags, all behind Replay/Holdout and
   local-consensus adoption gates.

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

`used` is diagnostic usage telemetry, not positive supervision. Search, read,
injection, and use activity alone never teach the system that a page improved
the answer. Positive page rewards require a sealed, preregistered answer-level
field-on/field-off evaluation against an independently sealed gold, rubric,
and evidence manifest. The scorer reports correctness, grounding, and citation;
every pair is verified, point and connected-cluster bootstrap lower bounds pass
separately, and exact used page/content identities are preserved. The production
answer is never the scoring reference. Learning consumes only the `train`
artifact; holdout and locked-test artifacts never enter training.

Answer episodes contain prompt/answer digests and character counts, exact Raw
save-transaction receipt/range/digest references, canonical session and decision
identities, and exact page/content bindings. They never contain answer bodies. If the host generator
and sampler identity was not durably captured, production-exact replay remains
explicitly unclaimed; offline paired evaluation can still use an injected,
sealed runner identity.

The append-only inputs are `$CHRONOVISOR_ROOT/recall/answer-episodes.jsonl`,
`answer-review-receipts.jsonl`, and `answer-execution-receipts.jsonl`; answer
episodes use schema 1. Capture cursors are stored at
`runtime/recall-answer-eval/capture-cursors.json`, and the preregistered split
manifest at `runtime/recall-answer-eval/split-manifest.json` uses schema 2.
Sealed evaluation artifacts use schema 3: train output is
`runtime/recall-answer-eval/train-answer-eval.json`, and locked Field evidence
is `runtime/recall-field/locked-answer-eval.json`.

Temporal splitting is sealed before evaluation and chronological over connected
components of session, query, page ID, page UID, and content digest. Timestamps
must be timezone-aware, the 24-hour split-boundary embargo is fixed, and legacy/invalid timestamps are
quarantined as `unassigned`. No unassigned, embargo, holdout, or locked-test
row can unlock Field learning.

Every offline answer evaluation also requires an independently sealed scorer
calibration artifact, frozen before evaluation and bound to the exact scorer,
review protocol, review receipts, execution receipts, and disjoint calibration
cases. Locked `field-e2e-replay` has a stronger implementation boundary: it must
use the built-in `builtin_field_environment_replay` adapter and exactly match
`builtin_field_environment_identity()` for the live model, policy, config,
corpus, index, clone protocol, last-known-good base, effective Field config, and
candidate policy delta. Supplying a custom equivalent or stale identity cannot
create locked authority.

## Safety Boundary

Raw capture remains lossless. Content changes remain provenance-bound and use
CAS, read-back, local consensus, and fail-closed semantic lanes. Recall policy
changes remain subject to the independent sealed manual-94 retrieval gate,
locked answer evaluation, connected-cluster confidence bounds, and adoption
checks. Production authority requires retrieval, locked-answer, train-outcome,
and cross-split gates together, plus point and lower-bound floors for answer
reward, candidate coverage/precision,
and Processor coverage/precision; missing identities, legacy point-only
artifacts, or fewer than the required independent clusters hold authority.

The canonical retrieval manifest is
`$CHRONOVISOR_ROOT/runtime/search-eval/manual-94-manifest.json`. Schema 2 must be
frozen before evaluation and contain exactly 94 unique reviewed entries, with
exact entry and manifest seals plus the review-ledger byte length, line count,
file hash, and hash-chain head. At evaluation time every ranked page is rebound
to its live page UID and content digest; a changed page or concurrently assembled
cohort fails closed.

A passing train-only historical-context evaluation may enable positive learning
into candidate Field state, but it is not locked authority. Until manual-94,
the locked built-in field-e2e replay, exact cross-split validation, confidence,
freshness, and promotion gates all pass, production ranking remains
teacher-owned/full-search fallback and Field is candidate/shadow. Active Field
authority begins only after the sealed promotion boundary validates.

Recall Field persistence has its own versioning and is not the Raw archive v2
layout. Legacy `recall/field/sessions/` schema-1 snapshots are read-only
migration sources. On first read they are converted into sealed schema-2
snapshots in `recall/field/sessions-v2/`; current code writes only that v2
namespace and `events-v2/`, never the legacy source.

Reviewed negative feedback has one direct path. Only a strong Frontier-reviewed
`page_ignored`/contradiction row with an exact decision/session reference and
review-time and current page-content hashes may inhibit Recall Field state or
subtract that exact episode contribution from positive co-fire. Application is idempotent by the
producer-key/feedback-digest pair. Inhibition is compositional rather than a
destructive activation rewrite; it suppresses pending and new teacher commits
for the same page and is reversible only by deleting that exact contribution
after a digest-bound append-only
retraction. Passive exposure and malformed/stale feedback remain non-negative.

Frontier execution remains limited to validated system-code incidents and is
not a Recall fallback.
