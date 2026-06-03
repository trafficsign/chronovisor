# Operations

## Status

```sh
llm-wiki status
llm-wiki status --json
```

Shows wiki counts, active config, recall decision counts, feedback counts, and
runtime status.

## Doctor

```sh
llm-wiki doctor
llm-wiki doctor --json
```

Runs lightweight operational checks for wiki directories, config, and detected
host hooks.

## Recall Logs

```sh
llm-wiki-recall --recent 20
llm-wiki-recall --feedback missed --prompt "..." --note "..." --ref <decision_id>
```

Manual feedback remains useful for false negatives that the auditor cannot
observe confidently.

## Audit and Auto-Apply

```sh
llm-wiki-recall-audit --host codex --hook
llm-wiki-recall-auto-apply --dry-run
```

Auditor feedback uses `kind = "missed_candidate"` and source `auditor`. Additive
auto-lane actions can be applied automatically; global behavior changes remain
review-only.

## Troubleshooting

- If Codex hooks appear disabled, inspect `~/.config/codex/config.toml` trusted
  hash entries.
- If hooks look stale, check whether host settings call local scripts or a
  package entry point.
- If local models remain loaded after tests, run `ollama ps` and stop them.
