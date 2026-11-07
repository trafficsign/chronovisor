# Decision reranker pre-deployment evidence

`replay_decision.py` is an unchanged archive of the script used for the P3 latency replay on 2026-11-07. The original benchmark directory is not a Git repository. This copy preserves the measured procedure; private inputs, model weights, and operator logs are not included.

Original location: `~/projects/sandbox/laya-recall-filter-bench-20261004/run3/prod/replay_decision.py`

The script resolves `../model_downloads.json` and `private/search_cands.jsonl` relative to its location. To rerun, place this copy at the original benchmark location (preserve any newer local edits), and use the pinned reranker-service environment with this branch on `PYTHONPATH`. It reads the local recall log and live index, and takes the real accelerator lease. Run only one replay at a time. `PROBE=0` disables the concurrent semantic probe during replay; the initial semantic-only probe still runs.

The measured replay uses a 10-second service timeout. Its latency measurements are not proof of acceptance at the proposed production search timeout (3.5 seconds) or the remaining recall-filter deadline. Timeout/abstention rates must be checked during rollout. Only the legacy `items` payload shape is replayed; production tests cover other publication paths separately.

Original script SHA-256: `6069e6f89676c32a222ea95ccf8edb9398ca4befd0786d25b7a3263e722145f7`

## Test comparison

- Original branch full suite at `c0cb6576`: 50 failed, 7,560 passed, 1 skipped (the handoff said 49).
- Base `5010e542`, all 10 affected test files, branch `.venv` and base `PYTHONPATH`: 40 failed, 415 passed in 420.81 seconds.
- Each original failure has a matching error in this base run or the inherited `failed-on-base.log`; temporary pytest directory counters are normalized for comparison. The current run matches 39 original error signatures, the inherited run matches 48, and their union covers all 50.
- The 10 originally failing tests that passed in the current base run also passed on the branch (10/10). The current base's training/artifact test failed at an earlier assertion; the inherited base log reproduces its original `KeyError: gates`.
- No new pytest regression was identified. This does not mean the full suite is green. Operator policy/process state affects these existing tests.
- `bb59f842` fixes new Ruff import ordering and mypy errors for the service-only StartLux dependency, following the existing optional-MLX import pattern. Ruff passed; mypy passed for 99 files; 96 provider/config/architecture tests passed; the standalone architecture baseline check passed.
- `test-comparison.json` contains the 50-node comparison and hashes of external logs. Full logs remain in the original scratchpad, without copying operator data into Git. The clean, unused baseline worktree was removed after verification.

## Deployment waiting for approval

The latest handoff explicitly requires the user to say OK in this pane before publication or production changes. No push, model installation, configuration edit, or service restart was performed during this continuation.

Read-only preflight confirmed remote `main` and configured runtime source both at `6bb1b0f74b3c4af238620cc14094807e0e26c5fa`. The feature branch descends from this commit. The candidate dependency constraints are archived alongside this report, SHA-256 `f6ed7701123a0f55777b33fccba4a7ecdf1090a79d7dd95dfda4dc57a5a59623`.

After approval, deploy the approved branch HEAD, copy the 4B snapshot `56f18acdc9502c44fdb80234c09a2276e3398916` (5,136,696,107 safetensor bytes) into the managed models directory, install the archived constraints, switch `search.rerank` to `local_decision`, set service mode `off` / timeout 3500 ms, recall filter `shadow`, and recall total timeout 4800 ms. Restart only `com.trafficsign.chronovisor-reranker.managed-v2` and verify the live search/filter paths and runtime identity. The unpinned recall hook also changes when main is pushed, as noted in the handoff. Refresh remote state and capture rollback files before executing.

Keep BGE weights/environment for rollback. Do not enable the recall filter's `on` mode until fresh shadow samples meet the stated acceptance criteria. The final 4.8-second recall path and proposed production timeouts still need live verification.
