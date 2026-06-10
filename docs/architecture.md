# Architecture

LLM Wiki has three runtime loops around one wiki store.

```text
UserPromptSubmit
  -> llm-wiki-hook
  -> strip non-user blocks and trivial prompts
  -> load recall/sessions/<session_id>.json
  -> build queries, rewriting ambiguous references when needed
  -> BM25 + semantic + graph-expanded search
  -> evidence gate (features -> none/cards/read)
  -> optional thin RECALL_CONTEXT cards

Stop
  -> llm-wiki-hook
  -> save session delta to raw/
  -> run asynchronous recall auditor with precision checks
  -> record missed_candidate and injection_used/injection_ignored feedback
  -> auto-apply safe additive actions

Nightly / manual
  -> llm-wiki recall-eval --save-baseline
  -> llm-wiki-recall-calibrate
  -> apply validated-auto calibration only after holdout improvement
```

## Store

`~/.wiki` is the source of memory truth.

- `raw/`: append-only session captures.
- `pages/`: structured pages created by ingest/lint workflows.
- `system/`: privileged pages such as profile, current state, and lessons.
- `recall/`: recall decisions, feedback, query hints, and auto-apply logs.
- `recall/sessions/`: lightweight recent query/topic/page state for an active host session.
- `recall/pull-log.jsonl`: wiki.search/wiki.read calls used as implicit pull feedback.
- `recall/calibration.json`: validated evidence-gate weights.
- `runtime/`: status, events, and metrics for observability.

## Runtime Roles

- **MCP server**: exposes wiki tools to hosts.
- **Recall gate**: fast evidence-based search gate. BM25 is always attempted;
  semantic, graph expansion, rewrite, and calibrated weights fail open.
- **Save harness**: host-specific session parser plus memory writer.
- **Auditor**: heavy asynchronous judge that looks for missed recall and whether
  injected pages were used.
- **Auto-apply**: applies additive actions only (`query_hint`, `alias`, `page_tag`).
- **Calibration**: pure-Python logistic calibration on recall-log features and
  feedback labels, applied as `validated-auto` only after time-ordered holdout.

`few_shot` and `threshold` actions are review-lane only because they alter the
gate's decision behavior globally.

## Host Boundary

Codex and Claude Code now enter through `llm-wiki-hook`. Legacy scripts remain
as wrappers so existing hook settings continue to work while new deployments use
the single dispatcher directly.
