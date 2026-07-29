# Campaign G incident-ledger closure

Campaign G was listed as a separate RFC in the refactoring roadmap, but the
runtime capability had already landed before Campaign H-L began. Reimplementing
it would create a second authority path, so this closure records the existing
implementation and makes its completion criteria explicit.

## Existing implementation

The incident-ledger work was introduced and hardened by the following earlier
commits:

- `a7649d2`: trusted system-incident producer, privacy-bounded durable
  observations, local rechecks, and frontier-repair admission;
- `1850975`: operational-failure linkage, authority-bound repair evidence, and
  autonomous-ingest hardening;
- `f4a6d95`: durable outcome synchronization and safe release of repaired
  operational quarantines.

The implementation now lives in
`chronovisor.ops.system_incident_supervisor`. It provides:

- an atomic, locked, schema-versioned incident ledger;
- normalized diagnostic hashes that do not persist raw exception text, paths,
  stack traces, or user identifiers;
- recurrence and distinct-input thresholds;
- two deterministic local repair/reproduction checks before packet creation;
- allowlisted trusted producers and exact fingerprint/packet binding;
- idempotent packet creation and durable background-job enqueue;
- fail-closed handling of malformed state, invalid paths, stale links, and
  cancellation races;
- synchronization of terminal repair outcomes back to every linked source
  packet.

Routine semantic disagreement, model output, authentication, billing, quota,
credentials, and other human-boundary failures remain excluded from the
system-code repair lane.

## Verification

The closure suite covers the ledger, trusted operational linkage, frontier
admission, and the routine/frontier boundary:

```text
66 passed in 1.16s
```

Covered test modules:

- `tests/test_system_incident_supervisor.py`
- `tests/test_operational_system_incident.py`
- `tests/test_frontier_guard.py`
- `tests/test_routine_frontier_contract.py`

Campaign G is therefore complete as an implemented capability. No duplicate
ledger or migration is required.
