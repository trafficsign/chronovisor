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

This schedules save and audit work in background and immediately prints `{}`.

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
two separate Stop hooks.

## Inspect

```sh
llm-wiki hooks inspect
llm-wiki hooks inspect --json
```

This lists detected Codex and Claude Code hook entries and computes Codex-style
canonical hook hashes for inspection.
