# Chronovisor

Local transcript capture and BM25 search over existing knowledge pages.

- Codex and Claude Code Stop hooks enqueue durable raw saves. Missed tails are
  recovered by the session sweeper. Prompt hooks return a no-op.
- Raw layout v2 and existing save-state files remain compatible.
- Search is BM25 over frozen pages with exact page/title/tag anchors, filters,
  sorting and wikilink expansion. There is no vector channel and no reranker.
- MCP exposes `chronovisor_search`, `chronovisor_read`, `chronovisor_init`,
  `chronovisor_status`, and `chronovisor_record`. Initialization reads
  `user-profile` and `lessons-learned`; explicit "remember this" requests save
  raw content through `chronovisor_record`.
- Pages are frozen: there is no ingestion, generation, recall injection,
  learning, research, or decision workflow.

Requires CPython 3.14. Install the core with `uv sync --frozen`. Only capture,
lexical search, and the dashboard are retained. Existing configurations may
keep retired sections and roles; they are ignored.

Run `chronovisor-mcp` for MCP access. `scripts/chronovisor-capture` runs capture
job retries and the session sweeper in one process, eight items each, and writes
`runtime/capture-status.json`. Its launchd template runs every 120 seconds.
`scripts/chronovisor-deadman` alerts when capture status is missing, stale, or
reports failures; its template runs every 300 seconds. The macOS service
installer manages only the dashboard services; capture and the deadman are
installed separately by the operator.

For a read-only client, set `CHRONOVISOR_READ_ONLY=1`. This prevents client index
publication.

```sh
PYTHONPATH=src CHRONOVISOR_ROOT=$(mktemp -d) .venv/bin/python -m pytest -q
.venv/bin/ruff check src scripts tests
```

`_handoff/` and older design documents preserve historical context; they do not
define the retained product surface.
