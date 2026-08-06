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

### Protected compatibility surfaces

The following are compatibility contracts, not architecture exceptions:

- literal legacy module paths declared by `LEGACY_MODULE_PATHS`;
- console entry-point names and targets declared in `pyproject.toml`.

They are recorded separately under `compatibility_contracts` in the exception
ledger. Changing a name or target requires an explicit compatibility migration,
an observation period for live consumers, and a rollback path. A module move
must preserve these strings until that migration is complete.

### Existing exceptions and the fitness gate

`docs/refactoring/architecture-exceptions.json` is the machine-readable ledger
of every current cross-domain package edge, cross-domain private import,
literal dynamic import, and schema-manifest implementation-constant import.
Each package-edge row contains a `sites` diagnostic array enumerating all of its
current import statements. Package edges use one semantic identity per
source/target package pair; sensitive statement exceptions use scope and a
line-independent occurrence identity. Every exception also has a non-empty
owner, deadline, removal campaign, and rationale.

The architecture fitness gate is fail-closed. It rejects:

- a package-edge or sensitive-statement identity not present in the frozen
  baseline;
- a current package edge or sensitive statement missing from the detailed
  ledger;
- a stale ledger row whose code site no longer exists;
- a baseline identity missing from the detailed ledger;
- missing exception metadata or a malformed semantic identity;
- growth of production-to-`lab` package edges;
- drift in protected compatibility contracts.

Existing edge and strongly connected component baselines remain no-growth
ceilings. Removals are allowed. Removing an exception requires deleting its code
dependency, detailed ledger row, and frozen baseline identity in the same
change; package-edge removal also requires deleting all diagnostic sites for
that edge. Ledger-only removal and code-only removal both fail. New exceptions
are not accepted as a way to make a build green.

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
