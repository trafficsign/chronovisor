---
task_id: repo_refactoring_completion_20260729
created_at: 2026-07-29T19:32:35+09:00
状態: complete
branch: codex/refactor-campaign-h-l
baseline_commit: c60f773
final_commit_before_handoff: 3650e4d
---

# Chronovisor refactoring Campaign H-L completion

## Completed campaigns

| Campaign | Result | Commit |
|---|---|---|
| Baseline | preserved the classification experiment baseline and H-L plan | `8f7f8d1` |
| H | retired 11 concluded orphan experiments, moved 27 reproducible experiments into `chronovisor.lab`, introduced a shared harness and one lab CLI, and reduced console entry points from 62 to 49 | `e48e478` |
| I | consolidated byte-stable time, hash, JSONL, and atomic-write helpers after auditing formatting and durability differences | `ee95523` |
| J | moved 197 implementations behind domain packages, retained legacy import shims, rewired entry points and scripts, and made the `core` dependency boundary executable with Import Linter | `b20df94` |
| K | extracted 14 pure seams from seven high-churn orchestrators, removing 500 lines from those orchestration functions while preserving I/O order and authority checks | `3fe8a55` |
| L | established the fatal Ruff baseline, strict mypy for all 17 `core` modules, documented all 49 entry-point callables, and added regression guards for the quality contract | `3650e4d` |

Campaign G remains a separate incident-ledger RFC and was not changed.

Campaign L's initial fatal lint pass also exposed and fixed two existing bugs:

- legacy semantic-defer recovery referenced an undefined
  `authority_sha256` instead of the already validated `authority_epoch`;
- `_ollama_engine_identity()` had its result-building code after another
  function's return and therefore always returned `None`.

Both paths now have regression coverage.

## Final verification

- production ingest safety gate:
  `status=idle`, `pending_raws=0`, `ollama_available=true`,
  `authority_available=true`, `alert=false`, `retryable=false`
  (observed at `2026-07-29T19:31:45`)
- isolated repository suite:
  `2819 passed, 1 skipped in 562.98s`
- Ruff: `uv run ruff check src scripts tests` passed
- mypy: strict check passed for all 17 `core` source files
- Import Linter: 460 files, 1,770 dependencies, one contract kept and zero
  broken
- compileall and whitespace checks: passed
- all 460 Python import names: forward and reverse order passed in separate
  processes
- all 49 console entry points: resolved to callable targets
- all six launchd plists: `plutil -lint` passed
- live adopted artifact validation: source `adopted_artifact`, SHA
  `0ef9c98b840b5677b02e4f9e75189f327793f1a783b88f094b22f9eac495eefd`
- authority hashes remain byte-identical:
  - lane contract manifest:
    `9af11b07dc833561096127839ecafd9057390180ab0db7d531ec5d0953fdfcfe`
  - lane case manifest:
    `3789c8571536dd26feb8492a0727713b5f2658fedc48ecb33f2ccf5848e00e54`
  - structured generation policy:
    `8afa74215bf6217f2a6b0de916beca77df2f48e66c241145512fb35a2d8f8915`
  - production schema manifest:
    `1541981873a0669f5ef7234c9b4490fe3c3f00872d1a584b182a3a33799fbea2`

The final suite started only after the production ingest drain naturally
reached idle with no pending raws. The persistent launchd watcher remained
running; no active batch was killed or restarted.

## Workspace and deployment boundary

- implementation branch: `codex/refactor-campaign-h-l`
- the two pre-existing untracked paths were preserved unchanged:
  `_handoff/2026-06-11_0042_recall-redesign.md` and `logs/`
- no push, merge, launcher update, or runtime restart is part of this
  completion
- the running GitHub-backed runtime remains on its previously deployed
  revision until a separately authorized deployment

## Rollback

Revert the affected campaign commits in reverse order:

`3650e4d` -> `3fe8a55` -> `b20df94` -> `ee95523` -> `e48e478`

Persistent bytes and authority hashes were preserved, so rollback requires no
data migration or historical artifact rehash.
