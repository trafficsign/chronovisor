# LLM Wiki MCP

LLM Wiki MCP is a local-first memory runtime for LLM agents. It stores durable
conversation knowledge under `~/.wiki`, serves it through an MCP server, and can
wire host hooks for automatic save, recall, audit, and safe self-improvement.

## Core Pieces

- `llm-wiki-mcp`: MCP server.
- `llm-wiki-hook`: single hook dispatcher for Codex, Claude Code, and future hosts.
- `llm-wiki`: operational CLI (`status`, `doctor`, `hooks inspect`).
- `llm-wiki-recall`: synchronous recall gate.
- `llm-wiki-recall-audit`: asynchronous missed-recall auditor.
- `llm-wiki-recall-auto-apply`: applies safe auto-lane recall improvements.
- `llm-wiki-codex-save` / `llm-wiki-claude-code-save`: host session save harnesses.

## Storage Layout

```text
~/.wiki/
  raw/       # durable raw session captures
  pages/     # structured wiki pages
  system/    # privileged user/profile/state pages
  recall/    # recall log, feedback, query hints, auto-apply log
  runtime/   # observable runtime status/events/metrics
```

## Hook Entry Point

Host-specific scripts are compatibility wrappers. New integrations should call:

```sh
llm-wiki-hook --host codex --event UserPromptSubmit --hook
llm-wiki-hook --host codex --event Stop --hook
llm-wiki-hook --host claude-code --event UserPromptSubmit --hook
llm-wiki-hook --host claude-code --event Stop --hook
```

See `docs/architecture.md`, `docs/config.md`, `docs/hooks.md`, and
`docs/operations.md` for the operational model.
