# Classification and Librarian

Chronovisor has a low-priority Librarian control plane for stable page
identity, versioned classification, UID-normalized links, and future
claim-level consolidation. It is deliberately separate from synchronous
Recall. The default worker is deterministic, local-only, performs no model
calls, and never mutates Active Markdown.

## Current activation boundary

The bundled `udc-summary-top-level.json` is a nine-class bootstrap used only to
measure distribution and exercise the migration path. It is explicitly marked
`complete=false`. Classification authority cannot activate until an operator
installs a complete reviewed UDC Summary export, records its release, source,
license and checksum, and accepts a calibrated locked fixture. Until then:

- UUIDv7 identity and the metadata registry may operate;
- classification proposals are `shadow` only;
- existing `d/`, `t/`, and `s/` tags remain authoritative;
- classification filters report that they use non-authoritative proposals;
- the dashboard reports `NOT_READY`, even when the shadow queue is empty.

This prevents a zero-length queue or a successful bootstrap sweep from becoming
a false green.

## Runtime artifacts

All artifacts are metadata or temporary migration insurance under
`~/.chronovisor/runtime/librarian/`, outside normal page search and Obsidian:

| Artifact | Purpose |
| --- | --- |
| `page-registry.json` | UID, current path, legacy keys, status, classification proposal |
| `page-registry-events.jsonl` | Append-only registry audit |
| `uid-link-index.json` | UID outlinks, backlinks, anchors, unresolved links |
| `state.json` | Sealed `LibrarianStatusSnapshot` source |
| `events.jsonl` | Bounded-run receipts and 24h/7d flow source |
| `merge-ledger.jsonl` | Append-only claim/transaction receipts |
| `migration-restore-points/` | Checksummed, isolated migration insurance retained through final postflight |
| `transaction-preimages/` | Exact, isolated rollback bytes; seven-day TTL is an abnormal-stop upper bound |
| `soak.json` | Concurrent Phase 5–11 migration-observation state; no wall-clock release delay |

Read-time redirect resolution follows at most eight hops and never writes.
Redirect mutation eagerly flattens the target and rejects cycles.

## Commands

```sh
chronovisor-librarian --status --json
chronovisor-librarian --capture-baseline --repo-root /path/to/chronovisor --json
chronovisor-librarian --dry-run --limit 100 --json
chronovisor-librarian --limit 100 --json
chronovisor-librarian --full-sweep --json
chronovisor-librarian-release start-observation --json
```

The Sleep cycle invokes a bounded 100-page shadow batch. This lane has P3
priority and does not acquire or call Ornith, Nemotron, Gemma, GPT-OSS, or any
frontier model. Later model-backed stages must preserve P0 Recall capacity,
support cancellation/requeue, and pass the same deterministic preflight.

## Merge hard gates

`merge_transaction.py` is explicit-activation only. A plan must provide:

- every source span mapped to output, ledger history, or permitted boilerplate;
- all numbers, dates, proper names, URLs, and code identifiers preserved in
  output or explicitly disposed in the ledger;
- Raw provenance for every non-boilerplate mapping;
- sensitivity equal to the strongest input;
- exact source and affected-link preimage hashes;
- valid redirect anchors and one atomic registry generation.

A failed CAS restores only bytes owned by that transaction. Ordinary operation
deletes preimages after postflight; migration pilots may retain them for a
bounded TTL. During the initial rollout, observation starts with the full-corpus
shadow and continues through pilot and full migration. Release depends on
quality receipts, current full-sweep and terminal coverage, zero worker/link
debt, restore drills, and cleanup receipts—not on an additional fixed-duration
wait after migration.

## Dashboard states

The host derives one of:

`BLOCKED`, `FALLING_BEHIND`, `NOT_READY`, `MIGRATING`, `CATCHING_UP`,
`STEADY_WITH_HOLDS`, or `STEADY_CLEAN`.

The dashboard shows scope generation, numerator/denominator progress for UID,
shadow classification, links, migration batches and full sweeps, queue funnel,
Hold/quarantine debt, 24h/7d flow, restore points, and recent receipts.
