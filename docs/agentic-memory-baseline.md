# Agentic memory baseline

Frozen before the research-lane implementation on 2026-07-18 (JST), at
`bb947d396e68234501a729fd827f2344e73b6b30`.

## Storage and regression baseline

- `llm-wiki raw verify --full --json`: status `ok`, 10,848 logical units,
  62 open segments, 75 legacy archives, zero errors.
- Full pytest: 2,391 passed and four pre-existing failures in
  `test_semantic_defer_replay.py` after 666.85 seconds.
- The four failures were caused by isolated queue runs scanning production Raw
  and materializing global projection files. The Phase 0 prerequisite fix
  scopes operational-hold lookup to the queue rows being processed.

## Synchronous recall baseline

- Current hard total budget: 4,000 ms.
- Last 100 real `codex`/`claude-code` recall decisions: p50 3,283 ms,
  p95 4,000 ms, max 4,000 ms.
- Last 50: p50 3,294 ms, p95 4,000 ms, max 4,000 ms.
- The 500-row window contains historical unbounded outliers and is not used as
  the adoption reference: p50 3,886 ms, p95 32,792 ms, max 338,240 ms.

## Supervision baseline

- Recall log: 8,235 rows; 8,174 non-empty decision IDs and 8,182 non-empty
  session IDs.
- Pull log: 468 rows: 416 search, 52 read, zero explicit `used` receipts.
- Pull IDs: only one non-empty decision ID and one non-empty session ID.
- Strict explicit-used join: zero accepted samples. This is `no signal`, not a
  zero-percent success rate.
- Existing prefetch/co-fire artifacts were exposure-derived. Schema v2 keeps
  exposure and explicit-used supervision as separately named features.

These values are a frozen comparison point. Runtime logs continue to grow and
must not be substituted into the baseline after implementation begins.

## Phase 0 adopted re-baseline

- Full pytest after the scoped Raw fix and supervision schema v2: 2,399 passed
  in 668.12 seconds, zero failures.
- Targeted supervision/Raw regression: 31 passed.
- Rebuilt prefetch v2: 3,419 exposure episodes, zero promoted positive-used
  episodes, strict join accepted/rejected 0/0.
- Rebuilt co-fire v2: 3,145 exposure episodes, zero promoted positive-used
  episodes, strict join accepted/rejected 0/0.
- Positive promotion remains intentionally held until a production
  `wiki_recall_used` durable receipt joins exactly to one decision/session.
