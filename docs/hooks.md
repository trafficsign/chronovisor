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

This schedules content-correction, save, recall-audit, and recall-improvement
work in background and immediately prints `{}`. Content correction keeps a
durable cursor keyed by host, session, and transcript file. Each Stop scans all
newly completed turn pairs after that cursor, so a delayed transcript append or
a previously failed Stop is captured on the next run instead of being limited
to the newest turn. Re-capturing an existing turn preserves its immutable root
identity. The lane accepts recall provenance only when prompt hash, host,
session, and turn time match; repeated ambiguous prompts are left unattributed.

Detection and local classification do not authorize a side effect. Every
classification goes to the frontier model for the final decision in the
separate content-correction convergence lane.

## Legacy Wrappers

These scripts remain for existing host settings:

- `scripts/codex_recall_hook.sh`
- `scripts/codex_wiki_save_hook.sh`
- `scripts/codex_recall_audit_hook.sh`
- `scripts/claude_code_recall_hook.sh`
- `scripts/claude_code_wiki_save_hook.sh`
- `scripts/claude_code_recall_audit_hook.sh`

Save and audit wrappers call the dispatcher with `--only save` or
`--only audit` to avoid duplicate work while current host settings still invoke
two separate Stop hooks. The legacy `--only save` path also schedules content
correction so existing installations gain the new behavior without another
host hook entry.

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

## Inspect

```sh
llm-wiki hooks inspect
llm-wiki hooks inspect --json
```

This lists detected Codex and Claude Code hook entries and computes Codex-style
canonical hook hashes for inspection.
