# ADR-0001: Collection-first classification authority

- Status: Adopted
- Date: 2026-07-27
- Authority epoch: `collection-authority-v1`
- Supersedes: mandatory page-to-UDC and page-to-CVO-anchor quality gates

## Decision

Chronovisor treats its existing logical collections as the primary
classification authority. The original `pages/<folder>/` layout is bootstrap
provenance (original order), not mutable identity. Each collection receives a
stable UUIDv7 in the sealed collection registry. Page assignments are facts in
that registry.

UDC/CVO remains as a versioned, fully audited, many-to-many crosswalk at
collection granularity. It is not predicted for every page.

The local LLM is a review-only anomaly detector. It may attach an explanation
and suggested collection to a queue item, but it cannot change a collection
assignment or page bytes. A model `no_issue` result does not close the queue
item.

## Why the previous Phase 4 gate is superseded

Eight page-level approaches showed the same failure boundary: local models
understood the page but could not reliably map modern personal documents onto
UDC's historical disciplinary layout. On the same opened 70-case development
set:

| Method | Semantically covered | Major errors | Model calls |
| --- | ---: | ---: | ---: |
| Direct local LLM, max-2 | 52/70 | 18 | model-backed |
| Controlled vocabulary | 49/70 | 18 | model-backed |
| Existing collection crosswalk | 57/70 | 2 | 0 |

The initial archive audit assigned all 3,045 active pages to 64 collections
without duplicate page identity. Before activation, a fresh dry-run observed
3,052 pages and one new collection; the audited crosswalk was expanded to all
65 live collections before the sealed evaluation opened. On 2026-07-27,
normal corpus growth added one more collection; the gate failed closed until
the semantically equivalent campaign collection was reviewed and published as
crosswalk epoch v2 with 66 entries. This drift behavior is intentional: new
collections fail the 100% crosswalk gate until reviewed.

## Invariants

1. Collection lifecycle mutations require registry-generation CAS.
2. Rename, merge, split, and logical page move create sealed receipts.
3. No collection lifecycle operation moves or edits Markdown page bytes.
4. A page outside a known collection is assigned to the logical
   `_unclassified` collection and enters the review queue.
5. The current 66-entry collection crosswalk is fully reviewed, checksum-locked, and
   has exactly one `exact` anchor per collection plus optional `broad` anchors.
6. Crosswalk changes create a new epoch.
7. Large-collection graph analysis proposes splits only; it never applies one.
8. The largest-collection share, median collection size, assignment coverage,
   crosswalk coverage, unresolved links, and review queue are dashboard
   metrics.
9. The sealed unseen fixture opens only after this ADR, its gates, and semantic
   gold are locked.

## Phase 4 replacement gates

The old `exact >= 90%`, forced-misclassification, and Hold gates do not govern
collection authority. They remain as historical evidence.

The replacement gates are:

- assignment coverage `100%`;
- duplicate logical assignment `0`;
- crosswalk audit coverage `100%`;
- unseen assignment-or-review coverage `100%`;
- unseen major error `0`;
- largest collection warning above `35%`, hard block above `60%`;
- review queue hard budget `500`, with at most `100` new items per run;
- no page mutation and no frontier calls.

The largest-collection warning does not trigger an automatic split. A
deterministic label-propagation report creates an auditable proposal.

## Rollback

The previous per-page UDC proposal is retained as non-authoritative history.
Rollback disables collection authority and returns the Librarian to shadow
mode; it does not rewrite page content or delete registry receipts.
