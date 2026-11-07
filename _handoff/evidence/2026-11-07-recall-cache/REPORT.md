# Recall cache acceptance — 2026-11-07

Implemented on `feat/decision-reranker-20261107`; not deployed. Baseline: `2f5cb8bec032d30baff9e5d632d6c59e868a7067`.

## Correctness

- 500 distinct, complete queries stored in the real recall log, plus 31 reviewed train-negative queries.
- Exact equality: full contextual trace, penalties, BM25 rows/order/scores, anchor rows/order/scores, and penalized rows/order/scores. **0 mismatches** in every category; 31 nonempty negative traces.
- All feedback rows validated against a fixed recall log and fixed page bytes: old/new both admit 1 row, **0 mismatches**. Duplicate exact refs remain rejected.
- Sidecars produced under PYTHONHASHSEED=0 also reproduce all 31 negative traces under seeds 1 and 42.
- Source hash, schema/tokenizer version, same-size preserved-mtime changes, malformed/checksum-invalid sidecars, concurrent source changes, WAL changes, schema/page metadata checks, and read-only behavior covered by unit tests.

## Timings (ms)

CPU-only, fixed production-data snapshot, same Python/host. Three fresh processes per hit condition; median below. Existing background jobs were not stopped. Imports/process startup are excluded. Feedback entries already accepted by the existing persistent cache were held constant for negative/lexical measurements; full ledger miss is measured separately.

| Stage | Old fresh process | New fresh process, sidecar hit | New sidecar miss (one run) | Old same-process warm | New same-process warm |
|---|---:|---:|---:|---:|---:|
| Matching negative trace | 1909.070 | 157.842 | 1049.328 | 3.979 | 3.614 |
| Lexical snapshot validation | 738.215 | 348.599 | 885.737 | 0.108 | 0.058 |

Full feedback validation miss: 74961.626 -> 2441.682 ms (one paired run). Each validation run reads the recall log once instead of 44 times.

The first old lexical warm measurement repeated validation after SQLite created WAL/SHM files (759.567 ms); the other old warm measurements were 0.071/0.108 ms. All raw timings are retained in results.json. A new lexical cache miss pays the extra content-hash cost; the speedup requires a valid receipt.

## Cache contract

- Golden: `~/.chronovisor/runtime/search/golden-feedback-cache.json`, source SHA-256 + schema + tokenizer version; token sequence/order retained. One parse produces both negative and positive views. A concurrent update discards cached data; corruption or mismatch reparses the source.
- Lexical: `~/.chronovisor/.index/lexical.quick-check.json`, full DB and WAL SHA-256 + SQLite version; only a successful quick_check is cached. Generation, schema, page identities and metadata checks still execute. A concurrent writer forces the full check and prevents publishing a receipt for the changed content.
- Atomic writes, mode 0600. `CHRONOVISOR_READ_ONLY=1` reads but never creates or updates sidecars, including the pre-existing feedback cache. A writable caller must prime a missing cache before read-only hooks benefit.

## Validation and reproduction

185 tests passed: test_negative_feedback, test_feedback_ledger, test_lexical_index, test_index_store, test_search_types, test_search_fusion, test_search_pipeline_characterization, test_architecture_baseline. Ruff passed for changed Python files; mypy passed for four production modules and replay.py.

Private fixed inputs remain at `/private/tmp/chronovisor-recall-cache-20261107` (directory mode 0700); no raw query text is committed. The SQLite backup retains the production generation and all ranking data; only absolute metadata path roots are relocated. Page bytes used for feedback binding are copied too. Input hashes are in results.json.

Run with the worktree .venv Python and PYTHONDONTWRITEBYTECODE=1:

```sh
python replay.py capture /private/tmp/another-recall-cache-snapshot
CHRONOVISOR_READ_ONLY=1 PYTHONHASHSEED=0 python replay.py bench-old SNAPSHOT
CHRONOVISOR_READ_ONLY=0 PYTHONHASHSEED=0 python replay.py bench-new SNAPSHOT
CHRONOVISOR_READ_ONLY=1 PYTHONHASHSEED=0 python replay.py bench-new SNAPSHOT
CHRONOVISOR_READ_ONLY=1 PYTHONHASHSEED=0 python replay.py replay SNAPSHOT
CHRONOVISOR_READ_ONLY=1 PYTHONHASHSEED=0 python replay.py ledger SNAPSHOT
```

No service/config changes or production cache writes. GPU, semantic retrieval, full hook latency, and deployment acceptance are not measured here.
