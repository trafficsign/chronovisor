# Architecture

LLM Wiki has three runtime loops around one wiki store.

```text
UserPromptSubmit
  -> llm-wiki-hook
  -> recall gate
  -> optional RECALL_CONTEXT

Stop
  -> llm-wiki-hook
  -> save session delta to raw/
  -> run asynchronous recall auditor
  -> record missed_candidate feedback
  -> auto-apply safe additive actions
```

## Store

`~/.wiki` is the source of memory truth.

- `raw/`: append-only session captures.
- `pages/`: structured pages created by ingest/lint workflows.
- `system/`: privileged pages such as profile, current state, and lessons.
- `recall/`: recall decisions, feedback, query hints, and auto-apply logs.
- `runtime/`: status, events, and metrics for observability.

## Runtime Roles

- **MCP server**: exposes wiki tools to hosts.
- **Recall gate**: fast synchronous classifier, usually a small local model.
- **Save harness**: host-specific session parser plus memory writer.
- **Auditor**: heavy asynchronous judge that looks for missed recall.
- **Auto-apply**: applies additive actions only (`query_hint`, `alias`, `page_tag`).

`few_shot` and `threshold` actions are review-lane only because they alter the
gate's decision behavior globally.

## Host Boundary

Codex and Claude Code now enter through `llm-wiki-hook`. Legacy scripts remain
as wrappers so existing hook settings continue to work while new deployments use
the single dispatcher directly.
