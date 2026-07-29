# Campaign K function-decomposition audit

Campaign K continued the characterization-first, pure-seam approach from
Campaign F. Existing end-to-end tests were identified before each extraction;
`tests/test_campaign_k_seams.py` adds direct coverage for the new pure
projections and guards the reduced orchestrator sizes.

## Selected functions

| Function | Before | After | Extracted responsibility |
|---|---:|---:|---|
| `ingest.run_ingest` | 676 | 514 | body-safe job result and terminal authority/ACK effect |
| `orchestrator.run_pending_ingest` | 770 | 737 | projection, raw-row, and batch-envelope projections |
| `self_heal._handle_packet_unlocked` | 897 | 766 | retry outcome and terminal frontier effect |
| `content_correction._process_frontier_item` | 877 | 846 | audit projection and convergence commit |
| `ingest_review_apply.review_and_apply_ingest_operations` | 782 | 686 | authority-locked final effect |
| `collection_authority.review_collection_queue` | 236 | 210 | worker contract and queue-row transition |
| `classification_engine.run_consensus_batches` | 166 | 145 | batch identity and applied-cache projection |

The selected orchestrators lost 500 lines in total. Fourteen named helpers now
separate pure decisions/projections from authority leases, page mutation,
durable writes, callbacks, and operator status publication.

## Characterization boundaries

- Ingest terminal ordering: confirmed-noop authority revalidation, callback
  failure, partial generation, and non-fatal derived-index failure.
- Orchestrator: batch, per-raw, projection, continuation, and fragment paths.
- Self-heal: local/frontier routing, retries, human boundaries, cancellation,
  action application, and incident persistence.
- Content correction: classification, review, CAS apply, rollback, read-back,
  and audit commit.
- Review/apply: recovered artifacts, authority replacement, sharded review,
  confirmed no-op, and exact-postimage reuse.
- Librarian/classification: non-mutating review contracts, challenger
  transitions, worker lifecycle, and cached consensus output.

## Verification

- New pure-seam and size-cap tests: 8 passed.
- Related regression suite: 420 passed.
- Compileall and whitespace checks: passed.
- Import Linter: one contract kept, none broken.
- Lane contract, lane case, structured-generation, and production-schema
  hashes match the Campaign I/J baseline exactly.
