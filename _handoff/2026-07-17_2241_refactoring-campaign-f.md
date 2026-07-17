---
task_id: repo_refactoring_campaign_f_20260717
created_at: 2026-07-17T22:41:00+09:00
状態: complete
branch: codex/refactor-campaign-f-function-decomposition
baseline_commit: d5589ae
---

# Campaign F: high-ROI function decomposition

## Selection

The refreshed AST inventory was ranked by persistent mutation impact,
retry/rollback coupling, incident history, and available characterization
tests. The first five selected functions were:

1. `ingest_review_apply.review_and_apply_ingest_operations`
2. `ingest.run_ingest`
3. `orchestrator.run_pending_ingest`
4. `self_heal._handle_packet_unlocked`
5. `content_correction._process_frontier_item`

High line count alone was not used as the gate. Transaction, lock, CAS, queue,
and mutation boundaries remain in the parent functions.

## Extracted pure seams

- durable ingest review artifacts are projected into a typed structural state
  before any live authority or filesystem decision
- raw metadata is normalized into the two supported ingest side channels
- pending raw units own deterministic keyword fallback and event classification
- self-heal read-back packets are classified through one ordered retirement rule
- correction queue metadata and page evidence hashes are projected into strict
  provenance inputs

Each seam has direct characterization tests, including malformed input and
precedence cases. The implementation commits are `c62573d`, `c8b77c8`,
`2c103b2`, `dfa4bfe`, and `1995f7d`.

## Verification

- combined self-heal, content-correction, ingest-drain, and pure seam suite:
  242 passed in 25.49s
- focused ingest metadata and exact recovery suite: 7 passed in 13.79s
- prior Campaign E ingest/module regression remains 413 passed
- no persistent JSON, digest, queue transition, lock scope, CAS, or operator
  event text was intentionally changed

## Rollback

Revert the five Campaign F commits in reverse order. The helpers are pure and
do not require a data migration or runtime state repair.
