# ADR 0001: Layering, dependency, and compatibility policy

- Status: Accepted
- Date: 2026-08-06
- Campaign: P0

## Context

Chronovisor currently contains cross-domain imports, private-symbol imports,
literal dynamic imports, production-to-`lab` imports, and schema-manifest imports
of implementation constants. Removing them is Campaign P work, but the current
sites must be made finite and enforceable before modules move. At the same time,
legacy module strings and console entry points are external compatibility
contracts and must not be mistaken for removable architecture debt.

## Decision

### Layer order and allowed dependency direction

The canonical inside-to-outside layer order is:

```text
core/contracts -> domain -> application -> infrastructure -> hosts/ops
```

The arrow records architectural order, not permission for an inner layer to
import an outer layer. Compile-time dependencies point inward: a layer may
depend only on published APIs in the same layer or a layer to its left.
`hosts/ops` composes the process, `infrastructure` implements ports,
`application` owns use-case orchestration, `domain` owns business policy, and
`core/contracts` owns durable shared contracts and pure primitives. Package
renaming alone is not an architectural improvement.

`lab` is outside the production layering chain. It may consume published
contracts and application use cases as an outbound experimental client.
Production packages must not import `lab`, including through literal dynamic
imports. Experimental artifacts enter production only through a versioned,
validated production contract.

### Public and private APIs

A cross-domain dependency is allowed only through an intentionally published
contract, port, or use-case API. A leading underscore, a private submodule, an
implementation constant, or an internal schema-manifest detail is private even
when Python permits importing it. Re-exporting an implementation detail without
documenting and testing its compatibility contract does not make it public.

Domain implementation packages do not import another domain's private symbols.
Schema registries publish stable schema identities through a contract; they do
not collect implementation constants from their consumers.
Inside the two registry scopes, every imported all-uppercase name is treated as
an implementation constant without a target-module allowlist or a `_SCHEMA`
suffix requirement. This includes names such as `SCHEMA`, `FOO_SCHEMA_VERSION`,
and `FOO_SCHEMA_V2`, including same-package imports.

### Protected compatibility surfaces

The following are compatibility contracts, not architecture exceptions:

- literal legacy module paths declared by `LEGACY_MODULE_PATHS`;
- console entry-point names and targets declared in `pyproject.toml`;
- the 15 command/module dispatch pairs declared by `lab.cli.COMMANDS`.

They are recorded separately under `compatibility_contracts` in the exception
ledger. A replacement is additive first: the old and new targets remain
callable for at least one mixed-version release generation. Retirement requires
all deployed readers and writers to use the canonical target, durable references
and queues to be drained or replayed, and a minimum seven-day observation period
with zero legacy module, command, or entry-point invocations. Before removal, an
operator verifies that the previous GitHub-backed runtime archive still starts
and passes health checks. Rollback restores the compatibility mapping or wrapper,
redeploys that recorded commit, and restarts only the affected service. A module
move must preserve the old strings until all retirement conditions are met.

### Existing exceptions and the fitness gate

`docs/refactoring/architecture-exceptions.json` is the machine-readable ledger
of every current cross-domain package edge, cross-domain private import,
literal dynamic import, and schema-manifest implementation-constant import.
Each package-edge row contains an enforced `sites` array enumerating all of its
current import statements. Package edges use one semantic identity per
source/target package pair; sites and sensitive statement exceptions use scope
and a line-independent occurrence identity plus normalized statement-content
hash. Every exception also has a source-domain/component owner, deadline,
removal campaign, and rationale.

`docs/refactoring/architecture-exception-baseline.json` is the separate frozen
ceiling. Each ID class has disjoint `active` and `retired` sets whose union must
equal the inventory produced by applying the current scanner to the fixed
Campaign O source commit
`d404a6b20d00e3bcd1d4cdb89edfa5a718c51833`. The bootstrap mode reads only that
historical commit. A valid previous seed must name that exact commit, and the
current seed must retain the previous seed's source commit; changing the module
constant, current seed, and ledger together therefore cannot move the ceiling.
The normal `--retire-missing-exceptions` mode may move IDs only from `active` to
`retired`; it cannot add an ID, reactivate a retired ID, or change the historical
source commit. `--generate-exceptions` requires the current tree to match the
active seed exactly. The detailed ledger cannot authorize itself by rewriting
this seed.

The architecture fitness gate is fail-closed. It rejects:

- a package-edge or sensitive-statement identity not present in the frozen
  baseline;
- a new site inside an existing package edge, including production-to-`lab`
  static or literal dynamic sites;
- a current package edge or sensitive statement missing from the detailed
  ledger;
- an unrecorded, stale, duplicate, identity-mismatched, content-mismatched, or
  count-mismatched site;
- an ID reintroduced after retirement or any active-to-retired history reversal;
- a baseline identity missing from the detailed ledger;
- missing exception metadata or a malformed semantic identity;
- growth of production-to-`lab` package edges or static/dynamic import sites;
- drift in protected compatibility contracts.

Existing edge and strongly connected component baselines remain no-growth
ceilings. Removals are allowed. Removing an exception requires deleting its code
dependency and detailed ledger row while moving the corresponding seed ID from
`active` to `retired` in the same change; package-edge removal also requires
retiring all sites for that edge. Ledger-only, code-only, and seed-only removal
all fail. New exceptions are not accepted as a way to make a build green.

Production-to-`lab` milestones distinguish import-site work from package-edge
retirement. P2 tracks the classification site work, while the
classification-to-`lab` package-edge row carries P3 and is retired only when the
edge count reaches zero.

The edge-and-statement ledger is the Campaign P0 enforcement mechanism. A new
hard Import Linter layering contract is deliberately deferred until Campaign
P9, after the production-to-`lab` and cross-domain debt it would prohibit has
been removed or routed through published contracts.

## Consequences

- Current dependency debt is explicit, owned, and shrink-only.
- Line-only refactors do not churn exception identities.
- Compatibility paths remain protected while implementation modules move.
- Campaign P can remove exceptions incrementally without rewriting the Campaign
  O repository baseline.
- The ledger is large by design; it is generated mechanically and reviewed by
  counts, semantic identities, and gate behavior rather than hand-maintained
  line numbers.
