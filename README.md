# Chronovisor

Local transcript capture and search over existing knowledge pages.

- Codex and Claude Code Stop hooks enqueue durable raw saves. Missed tails are
  recovered by the session sweeper. Prompt hooks return a no-op.
- Raw layout v2 and existing save-state files remain compatible.
- Search combines BM25 and semantic retrieval, StartLux reranking, and wikilinks.
  Existing lexical, page metadata and semantic indexes remain readable.
- MCP exposes `chronovisor_search`, `chronovisor_read`, `chronovisor_init`,
  `chronovisor_status`, and `chronovisor_record`. Initialization reads
  `user-profile` and `lessons-learned`; explicit "remember this" requests save
  raw content through `chronovisor_record`.
- Pages are frozen: there is no ingestion, generation, recall injection,
  learning, research, decision workflow, or dashboard.

Requires CPython 3.14. Install the core with `uv sync --frozen`. Local model
services use the pinned dependencies in their wrappers; configure their local
model artifacts and routes before enabling semantic search or reranking.
Existing configurations are accepted; retired sections and roles are ignored.

Run `chronovisor-mcp` for MCP access. `scripts/chronovisor-capture` runs capture
job retries and the session sweeper, eight items each. Its launchd template runs
every 120 seconds. The macOS service installer manages semantic and reranker
services only; capture is installed separately by the operator.

For a read-only client, set `CHRONOVISOR_READ_ONLY=1`. This prevents client index
publication; already-running services still maintain their own runtime state.

```sh
PYTHONPATH=src CHRONOVISOR_ROOT=$(mktemp -d) .venv/bin/python -m pytest -q
.venv/bin/ruff check src tests
```

`_handoff/` and older design documents preserve historical context; they do not
define the retained product surface.
