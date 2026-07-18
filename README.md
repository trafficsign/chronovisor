# LLM Wiki MCP

LLM Wiki MCP is a local-first memory runtime for LLM agents. It stores durable
conversation knowledge under `~/.wiki`, serves it through an MCP server, and can
wire host hooks for automatic recall and lossless transcript capture. Routine
semantic decisions stay on local Ollama models; a frontier model is reserved for
exceptional, repeatedly reproduced system-code repair incidents.

## Core Pieces

- `llm-wiki-mcp`: MCP server.
- `llm-wiki-hook`: single hook dispatcher for Codex, Claude Code, and future hosts.
- `llm-wiki`: operational CLI (`status`, `doctor`, `hooks inspect`, `hooks install`).
- `llm-wiki-recall`: synchronous recall gate.
- `llm-wiki-recall-audit`: asynchronous missed-recall auditor.
- `llm-wiki-recall-auto-apply`: applies safe auto-lane recall improvements.
- `llm-wiki-content-correction`: compatibility CLI for binding explicit user
  corrections to exact-turn recall provenance. Structured decisions use local
  consensus and fail closed when the local quorum cannot be reached.
- `llm-wiki-codex-save` / `llm-wiki-claude-code-save`: deterministic, lossless
  host transcript-delta capture. The save path does not call an LLM.
- `llm-wiki-local-model-eval`: read-only, full-corpus replay gate for measuring
  the local decision fleet before an artifact-backed atomic policy adoption.
- `wiki_research` / `llm-wiki-research`: asynchronous-by-default, bounded
  evidence research through MCP or CLI. It
  follows Wiki -> verified claims -> Raw -> Web, persists source-backed
  Evidence Bundles, and challenges accepted evidence locally.
- `llm-wiki-research-verify`: temp-only adversarial verification with explicit
  PASS/FAIL commands.

The routine decision fleet is:

- primary: `maxwell1500/ornith-35b:Q5_K_M`
- challenger: `gpt-oss:20b`
- tie-break, only when needed: `gemma4:26b`

The primary and challenger must agree. If they do not, the tie-break model is
called and any two matching votes form the quorum. Invalid JSON gets at most two
targeted repair turns in the same local chat session. Exhausted repair and
ordinary lane disagreement fail closed; neither is escalated to a frontier
model. In ingest, an exact three-way split of three valid, pairwise-distinct
decisions under the currently validated adopted-artifact SHA becomes a terminal
semantic defer. The immutable source stays in `raw/`, receives no self-heal,
frontier, or cooldown replay, and re-enters the ingest queue only when the router
fully validates a different adopted-artifact SHA. A merely changed, partial, or
invalid nominated file never releases it. Operational runtime failures remain in
their separate repair queue.

## Storage Layout

```text
~/.wiki/
  raw/       # durable raw session captures
  pages/     # structured wiki pages
  system/    # privileged user/profile/state pages
  recall/    # recall log, retrieval/content feedback, query hints, auto-apply log
  runtime/   # observable status, correction cursors/reviews, convergence state
  research/  # durable evidence CAS manifests and Evidence Bundles
```

## Hook Entry Point

Host-specific scripts are compatibility wrappers. New integrations should call:

```sh
llm-wiki-hook --host codex --event UserPromptSubmit --hook
llm-wiki-hook --host codex --event Stop --hook
llm-wiki-hook --host claude-code --event UserPromptSubmit --hook
llm-wiki-hook --host claude-code --event Stop --hook
```

`UserPromptSubmit` runs synchronous recall. `Stop` durably enqueues the
deterministic transcript-save lane and, when configured, a capture-only content
correction job. The hook starts no subprocess, model, ingest, or semantic
review. Only deterministic explicit-correction signals enter that queue;
ordinary turns advance the capture cursor and queued corrections are resolved
later by local convergence.

To update local Codex and Claude Code settings in one pass:

```sh
llm-wiki hooks install --host all
```

See `docs/architecture.md`, `docs/config.md`, `docs/hooks.md`,
`docs/research-orchestration.md`, and `docs/operations.md` for the operational
model.
