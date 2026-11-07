# Recall cache invalidation fix — 2026-11-07

Implemented on feat/decision-reranker-20261107. Not pushed or deployed. No production settings, services, stores or caches changed.

## Behavior

- Feedback sidecar schema 2 stores the validated source SHA-256. Fresh processes verify the cached prefix before reusing entries. Only complete appended non-target JSONL rows can extend the cache. Target negatives, exact-row retractions, malformed/partial tails, truncation or changed prefixes fall back to full validation. Stable-source checks prevent publishing results from a concurrent rebuild. Read-only processes never update sidecars; the same-process key also includes inode/ctime.
- Matching lexical schema/table/index names bypass schema DDL and user_version writes. Search, stats and service writers share the same opener. The only lexical user_version assignment found in src is now conditional. A clean opener also avoids chmod and explicit checkpoint writes.
- Actual writer commits refresh the integrity receipt; dirty writer close checkpoints and refreshes it again under the publication lease. Generation/schema/page-metadata validation remains in load_existing. Concurrent byte changes prevent publishing a receipt.

## Acceptance

200 tests passed across test_negative_feedback, test_feedback_ledger, test_lexical_index, test_index_store, test_search_types, test_search_fusion, test_search_pipeline_characterization, test_architecture_baseline. Ruff passed for the two production modules, two tests and sequence.py; mypy passed for the production modules and sequence.py.

Exact replay of 500 recorded queries plus 31 real reviewed negative queries: **0 mismatches** for trace, penalties, BM25 order/scores, anchor order/scores, penalized order/scores; **31 nonempty negative traces**. Compared to pre-cache baseline 2f5cb8be. A second comparison actually loads the incrementally reused feedback cache versus full validation at 7e9d4fee: 501 queries, 1 accepted feedback entry, **0 mismatches**. The original replay freezes accepted feedback, so this second comparison specifically covers the changed cache path.

The first isolated attempt used the default disabled negative-feedback setting; its empty traces/timings were rejected. Final runs explicitly enable the production negative-feedback parameters.

## Real sequence on isolated production-data snapshot

Prime caches -> append missed_candidate through append_jsonl_durable -> service/search shared LexicalIndex opener + stats + close -> fresh-process production search pipeline with read_only_snapshot=True. No-op open leaves WAL bytes/mtime unchanged (unit test also checks DB bytes). Miss handlers raise in the acceptance process, proving feedback/golden/integrity cache hits. Sidecar bytes/mtime remain unchanged during read-only search.

| Stage (ms) | Fresh process 1 | Fresh process 2 | Limit |
|---|---:|---:|---:|
| negative_feedback | 197 | 195 | <250 |
| bm25_snapshot | 338 | 345 | <400 |
| CPU search total | 770.895 | 644.700 | — |

Both searches returned 20 results. This is the production lexical/graph/feedback pipeline with semantic=False; no GPU service or end-to-end recall/filter latency is claimed. The search pipeline's `filter` timing means ordinary result filtering, not StartLux recall_filter.

## Reproduction

Private frozen data: /private/tmp/chronovisor-recall-incremental-sequence2-20261107 (0700). No raw query or page content is committed. The earlier source snapshot is /private/tmp/chronovisor-recall-cache-20261107.

```sh
PYTHONDONTWRITEBYTECODE=1 PYTHONHASHSEED=0 PYTHONPATH=src .venv/bin/python _handoff/evidence/2026-11-07-recall-cache-invalidation/sequence.py /private/tmp/NEW-SEQUENCE --source /private/tmp/chronovisor-recall-cache-20261107
CHRONOVISOR_ROOT=/private/tmp/NEW-SEQUENCE CHRONOVISOR_READ_ONLY=1 PYTHONDONTWRITEBYTECODE=1 PYTHONHASHSEED=0 PYTHONPATH=src .venv/bin/python _handoff/evidence/2026-11-07-recall-cache/replay.py replay /private/tmp/NEW-SEQUENCE
```

## Limits

The 21:10 WAL writer PID could not be identified cheaply from existing semantic/ingest logs; the diagnosis reproduced the unconditional header write, not its original process identity. Old running archives will still use the old opener until a later approved rollout. Schema-1 feedback caches need one writable rebuild. Integrity checking adds work to real writer commits/checkpoints, while removing it from the following cache-hit reader. Live hook acceptance remains for deployment.
