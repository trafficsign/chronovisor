# ADR 0002: Treat MCP tool contracts as store-shaping interfaces

- Status: Accepted
- Date: 2026-08-10

## Context

An MCP tool's inputs, outputs, and mutation semantics influence how agents split,
organize, and revise the wiki. Changing the available operations can alter page
granularity, hierarchy depth, metadata completeness, link density, and whether
new evidence is merged into an existing page or written as another page.

## Decision

Changes to ingest, search, read, check, and mutation tool contracts must describe
their expected structural effect. Before rollout, compare a representative fixed
corpus before and after the change using:

- page count and size distribution;
- directory depth and pages per directory;
- required metadata presence and validation failures;
- link edge count, broken links, and orphan pages;
- create, update, merge, duplicate, and conflict outcomes.

Input and output fields retain their meaning across compatible changes. New
optional fields may be added, but existing fields or mutation behavior are not
removed or reinterpreted without a versioned contract and a migration path.
Read-only tools remain read-only; checks do not imply repair; create, update, and
merge remain distinct mutation intents.

Rollout is rejected or rolled back when the fixed-corpus comparison shows an
unexplained structural shift, schema or broken-link regression, duplicate growth,
or a create/update/merge behavior change outside the stated intent. Rollback
restores the previous tool contract and implementation; durable pages are restored
from the normal pre-mutation snapshot when the changed contract wrote invalid
structure.

## Consequences

Tool changes carry a small structural regression check. No target hierarchy,
page size, or link-density quota is introduced: measurements detect unintended
change rather than rewarding more elaborate organization.
