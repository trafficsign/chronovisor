# Architecture

Chronovisor retains three surfaces: deterministic transcript capture, BM25
search over frozen pages, and a read-only dashboard.

```text
Stop hook
  -> chronovisor-hook
  -> durably enqueue one coalesced save job for the session
  -> return immediately

chronovisor-capture
  -> retry_due(limit=8): run queued host save workers
  -> session sweep: recover missed Claude Code / Codex tails
  -> write runtime/capture-status.json atomically

Save worker
  -> resolve the exact session transcript
  -> append lossless bounded raw/ v2 bytes
  -> fsync data, publish commit journal, verify receipt
  -> advance the save cursor only after receipt validation

chronovisor_search
  -> SQLite inverted BM25 over stable pages
  -> exact page/title/tag anchors + filters + sort
  -> one-hop wikilink expansion for the MCP response
  -> return direct_hits / expanded_hits / retrieval.channels.lexical
```

## Modules

- `common/`: paths, runtime config, canonical JSON/document, durable files,
  time and hash helpers.
- `raw/`: raw v2 storage, commit journals, legacy archive, save transaction.
- `capture/`: host transcripts, recorders, background jobs, sweeper and the
  capture cycle.
- `pages/`: page lookup, frontmatter, link fixes and page identity.
- `index/`: page metadata, BM25 lexical index, registry, aliases and search
  types.
- `search/`: the BM25 query API.
- `surface/`: MCP server, hook dispatcher, dashboard and search eval.
- `hosts/`: exactly two persisted-name shims for old job module strings.

## Store

`~/.chronovisor` is the source of memory truth.

- `pages/` and `system/`: frozen canonical Markdown pages.
- `.index/pages.json`: page metadata and link graph.
- `.index/lexical.sqlite`: persistent BM25 index.
- `raw/YYYY/MM/DD/`: append-only raw v2 source segments, commit journals, and
  manifests.
- `runtime/`: capture state, background-job ledger, capture status, deadman
  status, dashboard TLS/credentials, and logs.

`pages.json` schema 12 and lexical `user_version` 8 are unchanged. The raw
archive keeps the v2 source bytes; on-disk formats are not migrated by this
retained surface.

## Capture

`chronovisor-hook` only enqueues work. The background-job ledger is the durable
queue: `runtime/background-jobs/state.json`. `chronovisor-capture` runs the
retry pass and the missed-session sweep in one process. A corrupt ledger or
journal is preserved and blocks writers instead of being treated as empty.
Permanent worker errors exit once with `PERMANENT_EXIT_CODE`; transient errors
retry with bounded backoff.

`runtime/capture-status.json` separates attempt, success, commit, oldest pending
work, pending count, failed count, and a privacy-safe error token. The deadman
monitor reads it independently of Python and alerts through `osascript`.

## Search

`core.search` builds or opens the SQLite lexical index and returns
`ScoredPage` records. Ranking is the existing BM25 formula with exact anchors
for page IDs, titles, tags, entities, and raw keywords, plus CJK bigram
tokenization. Filters enforce stable lifecycle status, reference-page policy,
folder, and date ranges. The MCP tool adds tag filtering and one-hop wikilink
expansion; it does not reorder candidates after BM25.

There is no semantic channel, fusion stage, reranker, or model call in the
search path. Search evaluation uses the same in-process MCP tool under
`CHRONOVISOR_READ_ONLY=1`.

## Dashboard

The dashboard is read-only. It reads page metadata, raw save history, the
capture-job ledger, `runtime/capture-status.json`, and (on the LAN listener)
local TLS/credential files. The capture card shows last success, pending, failed,
and last error from the capture status; the queue panel shows job counts.

## Host boundary

Codex and Claude Code enter through `chronovisor-hook`. The compatibility
modules `chronovisor.hosts.codex_record` and
`chronovisor.hosts.claude_code_record` alias the raw save workers so existing
job module strings continue to execute. The Save harness performs no model
inference, ingestion, or page generation.
