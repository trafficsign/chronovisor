---
task_id: repo_refactoring_campaign_d_20260717
created_at: 2026-07-17T21:33:37+09:00
状態: complete
branch: codex/refactor-campaign-d-structure
baseline_commit: aad307e
---

# Campaign D: small structural refactoring

## Save adapter protocol

`agent_save_base.py` now owns only behavior that was identical in Claude Code
and Codex adapters: tolerant JSONL iteration, capture-payload detection, prompt
trimming, JSON extraction, keyword validation, cursor state read/write,
process decision, raw publication, and hook JSON parsing.

Host-specific discovery, transcript dataclasses/serialization, source marker,
session id, oversized fragmentation, save transaction identity, parser, output
JSON, and exit code remain in each adapter. Existing module-level imported
aliases preserve monkeypatch seams for `save_raw` and `write_state`.

## CLI dispatch

- `build_parser()` owns the complete stable parser tree.
- `dispatch(args)` owns handler routing and exit codes.
- `main(argv)` is parse then dispatch only.
- a recursive contract test executes `--help` for every registered top-level
  and nested command.

## Ollama facade

Non-streaming generate/chat calls with identical timeout, HTTP detail, and JSON
decode behavior now pass through `_post_json()`. Streaming generate and embed
remain separate because their status/error contracts differ.

## Semantic epoch

Plaintext-free structured-review epoch construction and validation moved to
`semantic_epoch.py`. `semantic_hold.py` retains its public compatibility
functions and resolver default while delegating pure digest/shape logic.
Filesystem authority observation and durable cache mutation remain outside the
pure module.

## Verification

- Claude/Codex save adapters: 57 passed
- CLI/Ollama: 50 passed
- combined semantic hold/CLI/save/Ollama regression: 124 passed
- every registered command help: covered recursively
- `python -m compileall`: pass
- `git diff --check`: pass
- lane/case/structured/schema hashes: unchanged from Campaign A baseline

## Rollback

Revert the Campaign D commit. No persisted state shape, hook output, authority
digest, model request payload, timeout, keep-alive, or exit code was changed.
