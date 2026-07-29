# Campaign J module-layout audit

Campaign J moved implementation code out of the flat `chronovisor` package
without changing durable schemas, authority identities, or queued worker module
names. The old import paths remain available for one compatibility generation.

## Resulting layout

| Domain | Implementation modules |
|---|---:|
| `core` | 16 |
| `raw` | 10 |
| `ingest` | 26 |
| `recall` | 21 |
| `search` | 17 |
| `classification` | 22 |
| `librarian` | 13 |
| `decision` | 14 |
| `research` | 21 |
| `ops` | 31 |
| `hosts` | 6 |
| `lab` | 29 |

The 197 non-lab implementations have top-level compatibility aliases. The
historical modules named `classification`, `ingest`, `librarian`, and `search`
collide with their new package names, so their package objects forward legacy
attribute reads and monkeypatch-style writes to the same-name implementation
module. `deadman_observer.py` is the only top-level implementation exception:
it must remain standard-library-only and installable outside the package
archive, while `ops/deadman_observer.py` aliases it.

Console scripts now target domain modules directly. Durable background-job
module strings deliberately retain their old values so already-queued jobs and
external readers remain compatible with the aliases.

## Function-scope import inventory

AST inventory found 490 function-scope Chronovisor import statements:

| Topological category | Count | Disposition |
|---|---:|---|
| Same-domain cycle seam | 155 | Retained; extract shared types only when a touched cycle can be characterized independently. |
| Core service on demand | 73 | Retained for call-time configuration/path resolution and optional heavyweight services. |
| Cross-domain on-demand boundary | 262 | Retained as explicit integration seams; Campaign K may narrow individual seams while decomposing orchestrators. |

The largest source domains are `ops` (127), `ingest` (82), `decision` (78),
`hosts` (58), and `recall` (50). The most frequent targets are
`raw.raw_store` (24), `core.runtime_config` (22),
`decision.failure_supervisor` (20), and `search.index_store` (20). There are no
remaining function-scope imports of a legacy two-segment implementation path.

This inventory classifies why the imports remain; it does not treat a delayed
import as an architecture exemption. The `core` dependency rule is enforced
separately by Import Linter.

## Mechanical checks

- Import Linter: 460 files, 1,770 dependencies, one contract kept and none
  broken (`core` cannot import domain or outer layers).
- Module import smoke: 459 modules in forward and reverse order.
- Console entry-point load smoke: 49 callable Chronovisor entry points.
- Launchd: all six checked plists pass `plutil -lint`.
- Full repository suite: 2,808 passed and 1 skipped.
- Contract strings altered by the initial path rewrite: 173 restored from the
  Campaign I baseline; follow-up AST audit found zero remaining transformed
  contract literals.
- Authority hashes remain unchanged:
  - lane contract manifest:
    `9af11b07dc833561096127839ecafd9057390180ab0db7d531ec5d0953fdfcfe`
  - lane case manifest:
    `3789c8571536dd26feb8492a0727713b5f2658fedc48ecb33f2ccf5848e00e54`
  - structured generation policy:
    `8afa74215bf6217f2a6b0de916beca77df2f48e66c241145512fb35a2d8f8915`
  - production schema manifest:
    `1541981873a0669f5ef7234c9b4490fe3c3f00872d1a584b182a3a33799fbea2`
