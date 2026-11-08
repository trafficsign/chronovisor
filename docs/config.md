# Configuration

The runtime config is `~/.chronovisor/config.toml`. The retained runtime reads
only `[raw]`, `[hooks.stop]`, `[runtime]`, `[dashboard]`, and `[dashboard_lan]`.
Retired ingest, recall, research, semantic-search, reranker, and LLM
role/provider sections are ignored; remove them when convenient.

The reader opens the config once and parses one immutable byte snapshot. A
missing file means defaults. A file that exists but is unreadable, non-UTF-8, or
invalid TOML raises `config_error`; MCP tools and the dashboard fail explicitly
instead of silently using defaults.

## Capture

```toml
[raw]
layout = "v2"      # legacy | shadow | v2

[hooks.stop]
save = true
```

`CHRONOVISOR_RAW_LAYOUT` overrides the layout for a process; an explicit value
does not read config. Capture workers write raw v2 source bytes and advance
their cursor only after the receipt validates.

`scripts/chronovisor-capture` runs queued save jobs and the missed-session
sweep in one process, then atomically writes `runtime/capture-status.json`:

```json
{
  "version": 1,
  "last_attempt_at": "2026-11-08T05:30:00Z",
  "last_success_at": "2026-11-08T05:30:02Z",
  "last_commit_at": "2026-11-08T05:30:02Z",
  "oldest_pending_at": null,
  "pending": 0,
  "failed": 0,
  "last_error": "none"
}
```

Timestamps are UTC seconds with a `Z` suffix. `last_error` is a privacy-safe
token. `scripts/chronovisor-deadman` reads this file and writes
`runtime/deadman-status.json`; it alerts when the status is missing or invalid,
`last_success_at` is older than 15 minutes, or `failed > 0`.

## Search

There is no search section. `chronovisor_search` uses the SQLite BM25 index at
`.index/lexical.sqlite` and the frozen page metadata. `sort_by` accepts
`relevance`, `date`, or `title`; `folder`, date filters, `tags`, and `tag_match`
are applied after retrieval. The retrieval trace contains only
`channels.lexical`.

Ranking evaluation is available without writing:

```sh
CHRONOVISOR_READ_ONLY=1 PYTHONPATH=src .venv/bin/python \
  -m chronovisor.ops.search_eval --out /tmp/search-eval.json
```

## Runtime identity

```toml
[runtime]
source = "git+ssh://git@github.com/trafficsign/chronovisor@<commit>"
github_repository = "trafficsign/chronovisor"
user_agent = "Chronovisor/0.1 (+https://github.com/trafficsign/chronovisor)"
launchd_label_prefix = "com.trafficsign.chronovisor-"
```

`source` is the package source used by the shell wrappers. Pin it to a commit
for production. `CHRONOVISOR_RUNTIME_SOURCE`, `CHRONOVISOR_GITHUB_REPOSITORY`,
`CHRONOVISOR_USER_AGENT`, and `CHRONOVISOR_LAUNCHD_LABEL_PREFIX` override the
corresponding values.

## Dashboard

```toml
[dashboard]
host = "127.0.0.1"   # loopback only
port = 8765

[dashboard_lan]
host = "192.168.50.20" # exact private IPv4; no wildcard
port = 8766
tls_cert_file = "/Users/me/.chronovisor/runtime/dashboard-lan.crt"
tls_key_file = "/Users/me/.chronovisor/runtime/dashboard-lan.key"
credentials_file = "/Users/me/.chronovisor/runtime/dashboard-lan-credentials.json"
```

LAN paths must be absolute. The private key and hashed credential file must be
regular, owner-owned `0600` files and must not be symlinks. Passwords never
belong in TOML.

## Retired keys to remove from production config

- `[search.embedding]`, `[search.embedding.service]`, `[search.embedding.rollout]`
- `[search.reranker]`, `[search.reranker.service]`
- `[llm.providers.semantic_foreground]`, `[llm.providers.semantic_incremental]`,
  `[llm.providers.local_decision]`
- `[llm.roles."search.semantic.foreground"]`,
  `[llm.roles."search.semantic.incremental"]`, `[llm.roles."search.rerank"]`

If `[search]` or `[llm]` becomes empty, remove the table header too.
