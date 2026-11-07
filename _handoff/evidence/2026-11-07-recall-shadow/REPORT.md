# Recall shadow isolation — 2026-11-07

Implemented on feat/decision-reranker-20261107; not deployed or pushed.

## Behavior

- Search commits its completed results and copies pipeline timing in `finally`, before graph diagnostics. Pipeline sub-stages already use finally timing; the existing interruption test is retained.
- Graph shadow traversal and observation-only compiler execution move to `chronovisor.recall.recall_shadow`, using the existing durable background-jobs queue. `converge_worker` consumes that queue with `retry_due`; there is no new daemon or detached hook thread.
- Only after context selection, enqueue plus remaining inline field/reranker/processor shadows get at most 100 ms, bounded by the search deadline minus 25 ms. That deadline already excludes filter and finalization reserves. Timeout/error records dropped observation and preserves results/timings; non-main-thread callers skip optional observation because SIGALRM cannot enforce a bound there.
- Graph and compiler observations retain their trace contracts and independent error handling. Compiler output is now in its background trace, rather than synchronous recall evidence. The recall row records queue status/job ID. Background snapshot data may be newer than foreground selection; these lanes remain observation-only.
- Read-only execution neither enqueues nor executes these two background observers.
- Verified, parsed relation records are shared between store instances and seeds in a bounded process cache. Device/inode/size/mtime_ns/ctime_ns changes invalidate it; each new version still passes sealed-object digest verification. Public lists are fresh, records are frozen. No disk cache or validation bypass.

## Verification

372 tests passed across recall_runtime, recall_shadow, recall_compiler, recall_filter, hook_dispatcher, knowledge_graph, knowledge_graph_runtime, search_pipeline_characterization and architecture_baseline. Includes slow enqueue/inline observer preserving results, stage timings retained on interruption, deferred worker execution, graph failure not blocking compiler, read-only suppression, and snapshot hit/invalidation/corruption recovery.

Ruff and git diff --check passed. Mypy passed for the new worker and graph store. recall_runtime has exactly the same 11 errors as HEAD before this change (normalized messages compared); zero new errors. A fresh `python -m chronovisor.recall.recall_shadow` process against a temporary root returned `observed` and wrote its compiler trace.

Architecture ledger: three new import sites, one retired foreground graph site; existing package-edge count remains 44. The exact retirement-history test was updated, without loosening gates.

## Exact hook replay

Command is copied verbatim from Codex hooks.json (see results.json). Same uvx source, interpreter (Python 3.14.7), cwd and payload/session. `PYTHONPATH` adds candidate src plus throwaway instrumentation, so this is candidate-code validation using production dependencies/data, not a deployed test. Module __file__ is asserted; the archive's direct_url is production a03eb80c, not a claim that candidate code has been deployed.

CHRONOVISOR_READ_ONLY=1; recall-log appends are intercepted, circuit-breaker state is copied per run to the private evidence directory, and diagnostic JSONL writes are suppressed. Timers and stack capture live only in that directory. Each command starts a fresh process and the six commands run sequentially.

Successful batch (all status ok, error empty, semantic enabled, breaker closed):

| Prompt | recall log ms | hook dispatcher ms | uvx/process wall ms | evidence_search ms |
|---|---:|---:|---:|---:|
| threshold | 1442 | 1566 | 2447 | 1154 |
| recall budget | 1246 | 1322 | 2119 | 986 |
| Ornith roles | 1220 | 1502 | 2241 | 961 |
| 4B choice | 1311 | 1562 | 2287 | 1051 |
| cache change | 1062 | 1300 | 1944 | 813 |
| original long English-annotated prompt | 1338 | 1405 | 2059 | 1087 |

The long prompt previously had 3204 ms logged recall and failed after a successful 1099 ms search plus 1919 ms synchronous shadow traversal. Candidate: 1338 ms logged recall, 1087 ms evidence search, no synchronous graph/compiler, no error. Remaining inline observers hit their ~100 ms limit and were dropped without discarding search results. No recall-filter invocation in these six turns (no published candidates); filter functionality is covered by the existing filter tests, not claimed from this replay. Reported recall latency precedes some finalization, so dispatcher/process measurements are included rather than presented as interchangeable totals.

**Separate production instability, not hidden:** The initial six and the final recheck six all returned `SemanticServiceUnavailable: lexical snapshot unavailable` before observers ran. An unmodified production-code control reproduced the error. Temporary instrumentation identified `projection metadata mismatch`; a later read-only comparison found zero mismatches after natural ingest publication, allowing the successful batch above. The final recheck again encountered the same mismatch. Final logged latencies: 544/479/481/472/469/475 ms; these are failure times, not speedup results. No database, config or service repair was attempted. The successful batch preceded the final mechanical tuple cleanup/timing mutex and background error-isolation cleanup; final-code regression tests pass but the last live-data recheck is blocked at lexical snapshot validation.

Private raw evidence: `/private/tmp/chronovisor-shadow-candidate-20261107`; previous diagnosis: `/private/tmp/chronovisor-hook-stacks-20261107`. Unit tests use an isolated CHRONOVISOR_ROOT. The first test run exposed a missing fixture isolation for the new queue path and created four synthetic queued jobs in the real ledger; all four were verified not running and removed, and the fixture now isolates queue files. No launchd/config changes, no deployment. Existing unrelated StartLux judge design file untouched.
