# Campaign M large-function closure

Campaign M processes every function in the 300-line inventory rather than
silently leaving the 31 functions that Campaign K did not select.

## Result

- baseline inventory: 38 functions at or above 300 lines;
- decomposed below the threshold: 7;
- retained with a concrete responsibility/order rationale: 31;
- unreviewed current functions at or above 300 lines: 0.

The seven new decompositions are:

| Function | Before | After | Extracted responsibility |
|---|---:|---:|---|
| `LocalStructuredSession._run_impl` | 309 | 299 | initial context-capacity admission |
| `hosts.server.chronovisor_search` | 315 | 208 | metadata filters, direct-hit projection, link expansion, filter report |
| `ingest_prepare.prepare_operations` | 349 | 203 | exact per-operation preimage/postimage construction |
| `librarian_status.build_librarian_status` | 314 | 277 | operator reason/detail projection |
| `autonomy.watchdog_snapshot` | 328 | 264 | quality probe and bounded health-incident projection |
| `recall_auto_apply._frontier_gate` | 308 | 291 | authority/proposal preflight |
| `recall_runtime._run_recall_impl` | 346 | 285 | initial skip, budget fallback, final context/session publication |

The retained functions are not blanket exceptions. The policy records an
individual maximum and rationale for each one. The recurring valid cases are:

- declarative breadth with low or zero branch depth (`build_parser`,
  `dispatch`, sleep-lane wiring, dashboard schema projections);
- a single authority or file lock whose visible scope is the safety boundary;
- a per-item retry/recovery state machine sharing attempt history, budget,
  rollback, and durable convergence state;
- preregistered experiment pipelines whose sequential exclusion reasons are
  part of reproducibility.

## Executable policy

`large-function-policy.toml` is the source of truth.
`tests/test_large_function_policy.py` reparses every source function and fails
when:

- a new 300-line function is not reviewed;
- a retained function exceeds its individual cap;
- a decomposed function regresses to 300 lines;
- a rationale or referenced characterization test disappears;
- the original 38-function inventory is no longer fully represented.

This makes “all functions processed” a maintained repository invariant, not a
one-time report.
