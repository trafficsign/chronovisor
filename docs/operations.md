# Operations

The retained runtime is capture, BM25 search, and the read-only dashboard.
Pages are frozen; there is no ingest, generation, recall, research, or model
service to operate.

## Status

`chronovisor_status` (MCP) reports page count, index files, the capture-job
ledger, and `capture_status`. `capture-status.json` is written by
`scripts/chronovisor-capture`; `deadman-status.json` is written by
`scripts/chronovisor-deadman`.

```sh
chronovisor-background-jobs status
cat ~/.chronovisor/runtime/capture-status.json
cat ~/.chronovisor/runtime/deadman-status.json
```

A healthy capture status has a recent `last_success_at` (UTC, `Z` suffix),
numeric `pending`/`failed`, and `last_error = "none"`. The deadman alerts when
the file is missing or invalid, `last_success_at` is older than 15 minutes, or
`failed > 0`.

## Capture

Stop hooks enqueue job-module work; they do not run the save inline.
`scripts/chronovisor-capture` runs `retry_due(limit=8)`, the missed-session
sweep and the index refresh in one process, then writes
`runtime/capture-status.json`.

```sh
scripts/chronovisor-capture
chronovisor-background-jobs retry --limit 8
chronovisor-session-sweeper --limit 8
```

A corrupt `runtime/background-jobs/state.json` is renamed to
`state.json.corrupt-<UTC>` and blocks writers until repaired. A corrupt raw
journal stops the affected capture explicitly; it is never treated as empty.
Permanent worker errors (`PERMANENT_EXIT_CODE`) become terminal `failed` jobs
after one attempt; transient errors retry with bounded backoff.

Install or inspect the capture and deadman LaunchAgents with the operator
steps from the deployment report. The templates live in `launchd/` and point at
`scripts/chronovisor-capture` and `scripts/chronovisor-deadman`.

Logs:

```sh
tail -f ~/.chronovisor/logs/capture.launchd.out.log
tail -f ~/.chronovisor/logs/capture.launchd.err.log
tail -f ~/.chronovisor/logs/deadman.launchd.out.log
```

## Dashboard

The normal dashboard is loopback-only:

```sh
chronovisor-dashboard
scripts/chronovisor-macos-services build
scripts/chronovisor-macos-services register
scripts/chronovisor-macos-services status
```

The macOS installer manages the `dashboard` and `lan-dashboard` services only.
The opt-in LAN listener uses `[dashboard_lan]` from
[configuration](config.md); wildcard, public, and hostname binds fail closed.
The private key and hashed credential file must be regular owner-owned `0600`
files and must not be symlinks. Restarting the LAN service revokes sessions.

## Search

`chronovisor_search` uses BM25 over stable frozen pages. It applies exact tag
filtering and one-hop wikilink expansion. No model or service is called.

Ranking evaluation is read-only against production:

```sh
CHRONOVISOR_READ_ONLY=1 PYTHONPATH=src .venv/bin/python \
  -m chronovisor.surface.search_eval --out /tmp/search-eval.json
```

Evaluate an explicit eval set with `--set PATH`; compare two runs with
`--compare A.json B.json`. Rows contain the query id, set, rank, latency,
search mode, and page-id hits. The default sets live under
`~/.chronovisor/eval/` and are not committed.

## Raw

Raw v2 keeps original source bytes under `raw/YYYY/MM/DD/`, with commit
journals and manifests. Do not rewrite or delete raw evidence to repair a
cursor; preserve the bytes and repair the state or journal explicitly. The
dashboard's Save History and Save Load panels are read-only projections.

## Troubleshooting

- **Capture status missing**: confirm the capture LaunchAgent is loaded and
  `scripts/chronovisor-capture` exits 0. The deadman reports `missing` until the
  first run writes the status.
- **Capture reports failed**: inspect `chronovisor-background-jobs status` and
  the job's `output_tail`. Permanent failures need the input or transcript path
  repaired; they are not retried.
- **Corrupt job ledger**: keep the renamed `state.json.corrupt-*` file and
  repair or quarantine the job before restoring writes.
- **Read returns not found for a known page**: `chronovisor_read` falls back to
  canonical files when the frozen registry omits a page. Pages that fail
  canonical validation stay invisible; do not weaken validation to expose them.
- **An edited page does not show up in search**: check `index_refreshed_at` and
  `index_error` in `runtime/capture-status.json`; the cycle refreshes the index
  every 120 s.
- **Search looks empty**: check `.index/lexical.sqlite` exists and is readable.
  A BM25 database failure is surfaced as an error/unavailable channel, not as an
  empty result list.
