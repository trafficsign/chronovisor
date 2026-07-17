---
task_id: repo_refactoring_campaign_e_20260717
created_at: 2026-07-17T22:19:05+09:00
状態: complete
branch: codex/refactor-campaign-e-ingest-modules
baseline_commit: 7c50f0a
---

# Campaign E: ingest modularization

## Result

`ingest.py` was reduced from roughly 9,100 lines at the roadmap baseline to
about 5,100 lines without changing its compatibility facade. The extracted
modules now own these boundaries:

- `ingest_schemas.py`: triage, recall-metadata, and frontier artifact schemas
- `ingest_transport.py`: bounded generate/chat transports and progress callback
- `ingest_review_plan.py`: review-shard dataclasses, request measurement, and planning
- `ingest_review.py`: verdict normalization and standard review execution
- `ingest_review_authority.py`: authority shape and aggregate shard-proof validation
- `ingest_review_store.py`: caller-rooted paths, JSON codec, proposal/review readback
- `ingest_review_recovery.py`: continuation, exact-repair, and no-progress records
- `ingest_review_execution.py`: shard manifests and sharded review execution
- `ingest_review_apply.py`: authority-gated review/apply transaction
- `ingest_triage.py`, `ingest_generation.py`, `ingest_prepare.py`,
  `ingest_apply.py`, `ingest_readback.py`: individual pipeline stages
- `ingest_recovery_runtime.py`: terminal and partial-shard recovery

## Compatibility and safety

- `ingest.<private symbol>` facades remain available for current tests and
  operators.
- Dynamic facades resolve patched functions at call time. Function identity
  checks inspect the original ingest symbol, not a generic forwarding wrapper.
- Artifact paths receive the caller-owned `PAGES_DIR` at call time; extracted
  modules do not capture the real Wiki path during import.
- The review/apply transaction keeps the authority lock, proof verification,
  durable readback, page CAS, and raw acknowledgement order unchanged.
- New import-order tests load every extracted module in forward and reverse
  order and verify patched Wiki roots.

## Verification

- final ingest and module-boundary regression: 413 passed in 751.40s
- the preceding run exposed three compatibility-seam regressions; all three
  were fixed and are included in the green final run
- focused authority, shard, recovery, apply, triage, generation, and readback
  tests: green
- module-boundary tests: 3 passed
- `python -m compileall`: pass
- `git diff --check`: pass
- lane contract manifest: `afcda164e055f994b7c439ea78518dd754a352cb9da581642c6c792e138ca43f`
- case manifest: `ae33c88e4eb08e3b771db21ee5d42904420c1672a38ba026e0d55f437b8312f6`
- structured policy: `8add0f29a5d56c2abdad7a78b806f16b9e58e57e49a7c9b18a7bf60e45f4afa8`
- production schema manifest: `1541981873a0669f5ef7234c9b4490fe3c3f00872d1a584b182a3a33799fbea2`

## Rollback

Revert the Campaign E commit. No artifact path, JSON bytes, schema digest,
authority source, review decision, page mutation ordering, or raw retirement
contract was intentionally changed.
