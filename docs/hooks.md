# Hooks

`llm-wiki-hook` is the public hook entry point.

## User Prompt

```sh
llm-wiki-hook --host codex --event UserPromptSubmit --hook
llm-wiki-hook --host claude-code --event UserPromptSubmit --hook
```

This runs the recall gate synchronously and prints host-native hook output.

## Stop

```sh
llm-wiki-hook --host codex --event Stop --hook
llm-wiki-hook --host claude-code --event Stop --hook
```

Stop durably enqueues a host save job and, when
`hooks.stop.content_correction = true`, one coalesced content-correction
capture job for the same session. It immediately prints `{}` and never starts
a detached process, model, ingest, mutation, frontier review, audit, or recall
improvement inside the hook process.

The save worker deterministically captures every uncaptured transcript delta
after an exact per-session cursor. A delayed append or a previously failed job
is picked up on the next run. Oversized deltas are split into lossless bounded
raw chunks, and the cursor advances only with the corresponding durable
receipt. The normal save path does not ask a model whether a turn is worth
saving and does not call any local or frontier model. The correction capture
worker only appends turns with a deterministic explicit-correction signal;
ordinary follow-ups advance the durable cursor without entering the queue.
Classification and resolution happen later in the bounded local sleep worker.

Semantic work is handled later by bounded convergence workers. When a routine
lane needs a structured decision it uses local consensus: Ornith 35B primary,
GPT-OSS 20B challenger, and Gemma 4 26B only as a tie-breaker. Invalid JSON gets
at most two targeted repair turns in the same local session. Failure or
disagreement is quarantined rather than escalated to a frontier model.

## Legacy Wrappers

These scripts remain for existing host settings:

- `scripts/codex_recall_hook.sh`
- `scripts/codex_wiki_save_hook.sh`
- `scripts/codex_recall_audit_hook.sh`
- `scripts/claude_code_recall_hook.sh`
- `scripts/claude_code_wiki_save_hook.sh`
- `scripts/claude_code_recall_audit_hook.sh`

Wrappers may still call the dispatcher with `--only save`, `--only audit`,
`--only correction`, or `--only improve`. The save selection enqueues raw
capture, and the correction selection enqueues only the dedicated
`--capture-only` worker. Audit and improve remain compatibility no-ops. No
selection starts semantic review in the Stop process.

## Install

```sh
llm-wiki hooks install --host codex
llm-wiki hooks install --host claude-code
llm-wiki hooks install --host all
llm-wiki hooks install --host all --dry-run --json
```

The installer replaces only existing LLM Wiki hook commands, preserves unrelated
host hooks, and writes direct `llm-wiki-hook` entries. For Codex it also updates
the matching trusted hash entries in `~/.config/codex/config.toml` and removes
stale LLM Wiki trust entries from the old split Stop hooks.

Names containing `frontier` in older hook settings or queued artifacts are
compatibility names, not proof that a frontier model is running. Actual Codex
execution exists only in the exceptional system-code-repair plane after strict
incident evidence passes its durable guard; a Stop hook can never enter it.

## Inspect

```sh
llm-wiki hooks inspect
llm-wiki hooks inspect --json
```

This lists detected Codex and Claude Code hook entries and computes Codex-style
canonical hook hashes for inspection.
