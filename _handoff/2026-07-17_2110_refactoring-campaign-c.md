---
task_id: repo_refactoring_campaign_c_20260717
created_at: 2026-07-17T21:10:10+09:00
状態: complete
branch: codex/refactor-campaign-c-durability
baseline_commit: 2e06683
---

# Campaign C: exact-contract infrastructure consolidation

## Consolidated exact families

| Contract family | Canonical implementation | Former copies |
|---|---|---|
| strict canonical JSON text/hash | `canonical_json.py` strict variants | adoption corpus, lane contract/cases, decision router, local eval, local structured, schema manifest, semantic hold |
| `default=str` canonical JSON text/hash | explicit stringifying variants | content correction, search eval, recall auto-apply, recall improvement |
| `default=str` + `allow_nan=False` | explicit stringifying-strict variant | ingest decision prompts |
| strict canonical bytes / line bytes | named strict byte variants | durable state, raw semantic projection, raw completion ack |
| stringifying canonical bytes without newline | `canonical_json_bytes_stringifying` | burn monitor, convergence drain |
| blocking exclusive text-file lock | `durable_state.exclusive_text_file_lock` | background jobs, model lab |
| suffix sidecar lock | `durable_state.sidecar_exclusive_lock` | raw replay, recall hints |
| directory fsync | `durable_state.fsync_directory` | durable state, orchestrator, raw semantic projection, MCP server |

Golden tests fix CJK/non-ASCII bytes, key order, trailing newline, Path
stringification, NaN rejection, SHA-256, and exact sidecar filename behavior.

## Classified retained variants

- atomic JSON writers are not exact duplicates: permission mode, backup/seal,
  directory fsync, `default=str`, sort order, cleanup, and lock boundaries differ.
- JSONL appenders are not exact duplicates: torn-tail repair, file/directory fsync,
  flock ownership, sort/default policy, and bounded-history rewrite differ.
- deadman canonical/durability code remains an intentional stdlib-only copy.
- lint's legacy NaN-permissive encoding is an explicitly named permissive
  variant rather than being silently folded into the strict artifact family.
- non-atomic telemetry writers remain behavior-preserving; hardening them is a
  separate semantic change, not this mechanical campaign.
- exact read-only JSONL readers remain local because their consumer-specific
  error/limit policy is outside the write-contract consolidation scope.
- Claude/Codex `iter_jsonl` duplication is assigned to Campaign D's save-adapter
  protocol extraction.
- read-back/self-heal reason normalization is domain policy, not durable-byte
  canonicalization, and remains locally named.
- timestamp and frontmatter candidates have no remaining exact AST-body cluster.

An AST-normalized inventory after consolidation reports no unclassified exact
duplicate in atomic write, JSONL append, lock, fsync, timestamp, frontmatter, or
durable canonical-byte writers.

## Contract preservation

| Contract | SHA after Campaign C |
|---|---|
| lane contract manifest | `afcda164e055f994b7c439ea78518dd754a352cb9da581642c6c792e138ca43f` |
| lane contract case manifest | `ae33c88e4eb08e3b771db21ee5d42904420c1672a38ba026e0d55f437b8312f6` |
| structured generation policy | `8add0f29a5d56c2abdad7a78b806f16b9e58e57e49a7c9b18a7bf60e45f4afa8` |
| production schema manifest | `1541981873a0669f5ef7234c9b4490fe3c3f00872d1a584b182a3a33799fbea2` |

## Verification

- canonical/adoption/decision/review targeted suite: 552 passed
- durability and migrated-call-site targeted suite: 237 passed
- canonical/lint/semantic-hold/ingest regression run: 341 passed before the
  live ingest-drain's legitimate Ollama lock blocked the next test; production
  was left untouched
- lock-isolated affected ingest/receipt cases: 7 passed, 403 deselected
- `python -m compileall`: pass
- `git diff --check`: pass

## Rollback

Revert the Campaign C commit. No persistent artifact was rewritten or migrated;
all consolidated functions preserve the former byte and exception contracts.
