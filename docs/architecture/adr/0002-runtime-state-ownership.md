# ADR 0002: Runtime state ownership and coordination registry

- Status: Accepted
- Date: 2026-08-06
- Campaign: P1

## Context

Chronovisor persists queues, ledgers, snapshots, indexes, credentials, locks,
schemas, service sockets, and worker launch contracts across many packages.
Declaration order is not ownership: compatibility aliases frequently sort before
the canonical writer, dynamic class paths do not appear as module constants, and
the same path may be read by several domains. Generic file extensions also do
not prove an application schema version. Without a fail-closed inventory, a
module move can silently create a second writer, bypass a lock, lose a launchd
argument, or retire a durable format while deployed readers still use it.

## Decision

### Registry and discovery

`docs/refactoring/runtime-state-owners.json` is the generated runtime ownership
registry. Every resource has a stable semantic ID, kind, typed locator, owner
package and symbol, writers and readers, coordination protocol, lifecycle,
source evidence, and compatibility contract. Artifact and queue rows also state
either a source-backed versioned format or an explicit unversioned rationale and
migration owner. The registry never derives application schema versions from
`.json`, `.jsonl`, SQLite, or directory syntax.

The scanner inventories all module-level durable path origins and schema-like
symbols, explicit dynamic runtime resources, sockets, all 51 console entry
points, all seven launchd services, the 15 lab dispatches, and 15 Python module
worker launch sites representing 13 unique module workers. Launchd rows preserve
the real wrapper invocations and arguments. In particular, the librarian worker
retains its full-sweep, primary, and challenger calls, and library evidence links
through `chronovisor-lab` to the `classification-library-pilot` dispatch. Those
source-backed wrappers require the standard CPython executable; library evidence
also records the exact runtime source, PATH-based bare `uvx` resolution, and
source lines that establish the contract. Its complete source template, including
the `/bin/sh` shebang, blank lines, quoting, continuations, and command sequence,
is validated, so a function, alias, duplicate, or preparatory command cannot
shadow the reviewed executable. The librarian and library-evidence launchd
`ProgramArguments` arrays are also exact code-backed contracts rather than
registry-authorized forwarding inputs.

Every discovered site is represented exactly once as a resource alias, a lock
protocol site, or an explicit exclusion. Prompt-only schemas, version constants,
status enums, process-local locks and caches, and source/deployment paths are
excluded with evidence; they are not silently ignored. Dynamic dashboard and
SearXNG credentials are registered with their mode, retention, and recovery
responsibility.

### Ownership and coordination

Reviewed ownership overrides replace lexical guesses. Canonical examples include
Recall Field promotion (`recall_growth`), locked E2E search evidence
(`search_eval`), typed-graph receipts and trace ledgers, the legacy embeddings
database (`search`), claims, recall logs, and runtime status.

Locks have a mandatory reviewed scope:

- `artifact_sidecar` protects exact registered artifact or queue IDs;
- `worker_lease` protects a registered worker or its exact durable state;
- `global_protocol` protects the cross-module page/system mutation boundary.

Same-module, same-directory, and same-package proximity are not lock evidence.
The raw replay queue, claims ledger, and search label queue use explicit sidecar
lock IDs. Search-label candidate generation and frontier review occur outside
the queue lock; only final preimage validation and mutation are locked, with
decision authority acquired first. Semantic jobs are the sole reviewed
multiwriter exception without a file lock: SQLite WAL plus `BEGIN IMMEDIATE`
transactions provide the coordination contract. Any other multiwriter row
without a valid registered lock fails the gate.

Socket contracts distinguish server and clients. Unix semantic and reranker
sockets record stale-unlink and mode-0600 startup; dashboard records the launchd
LAN bind and local client alias; Ollama and SearXNG have explicit external
servers; MCP stdio names an external host client rather than the server itself.

### Frozen ceiling and generator authority

`docs/refactoring/runtime-state-baseline.json` is a separate shrink-only seed.
Its universe is produced by applying the current scanner to fixed source commit
`fec76ac919b1cb0f64e772f85ceda46163df309c`. The ceiling records the concrete
lock calls at that revision rather than reserving absent planned IDs. Current
discovery also requires those actual calls, so removing any writer call creates
drift and fails.

Monotonic seed validation walks every baseline snapshot in the Git path history,
including history before deletion and re-addition commits. Active growth is
compared with the latest prior snapshot, while retired tombstones are unioned
across every canonical historical snapshot and can never return to active state.
Malformed, noncanonical, or schema-drifted historical documents fail closed. A
historical document's count shape, integer types, ID cardinalities, and derivable
kind/reason/protocol totals must also remain internally consistent. A shallow
checkout cannot prove that history and therefore also fails closed. CI uses
checkout `fetch-depth: 0` to make the complete ancestry available.

Active and retired discovery/resource ID sets are disjoint and monotonic. Normal
registry generation cannot add IDs, reintroduce retired IDs, rewrite the fixed
source head, or authorize code growth by changing the detailed registry. A
retirement moves an absent ID from active to retired while preserving the frozen
universe. Manual ownership, reader/writer, lifecycle, compatibility, format, and
lock contracts survive deterministic regeneration, but validation still rejects
invalid symbols, stale evidence, duplicates, wildcards, and unsafe coordination.
The baseline and registry use one canonical UTF-8 JSON byte form. Loaders reject
duplicate keys, non-finite numbers, non-object roots, noncanonical whitespace,
unexpected root or row keys, and reordered schema keys instead of normalizing
ambiguous input. Reviewed owner exceptions are code-backed overrides, so an
edited registry cannot authorize its own owner.

The gate also compares generated and recorded resource contracts, discovery-ID
grouping, resource/exclusion order, worker invocations, and lock protocol sites
exactly. A launchd label with changed wrapper arguments therefore fails even
when its worker locator and aggregate count remain unchanged. Entrypoint and
launchd sets plus all detailed counts are checked independently.

The CI gate runs `scripts/runtime_state_ownership.py --check` and the focused
runtime ownership tests alongside the architecture fitness suite.

## Consequences

- Runtime state and worker/service contracts are finite, reviewable, and
  shrink-only before Campaign P moves modules.
- Aliases no longer become owners through lexical ordering.
- Unversioned state remains explicit without fabricated schema numbers.
- Queue races in claims and search-label mutation are closed with shared physical
  sidecars while preserving preimage/CAS behavior.
- Adding a resource, writer, worker invocation, endpoint, or lock call requires a
  reviewed registry and frozen-ceiling change rather than a self-authorizing
  generator run.
