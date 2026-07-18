# Research implementation provenance ledger

This repository was private at the Phase 9 audit on 2026-07-18. It has no
repository license file, so no public redistribution license is asserted. Any
future visibility or licensing change requires a fresh source/provenance audit.

## Implementation sources

- Existing LLM Wiki contracts, runtime measurements, Replay/Holdout fixtures,
  and independently written tests are the primary specification.
- SearXNG's public Search API documentation informed the provider-neutral JSON
  search contract: <https://docs.searxng.org/dev/search_api.html>.
- MediaWiki's public Action API search documentation informed the keyless
  official fallback: <https://www.mediawiki.org/wiki/API:Search>.
- Python `httpx`, standard URL parsing, IP classification, and project-local
  security fixtures informed the fetch boundary.

No vendor-private or user-supplied non-public source code was copied, adapted,
or used as an implementation source. Such material was excluded from the code
and docs review boundary. The design patterns here were re-specified against
public documentation and independently verified behavior.

## Direct dependency licenses at audit time

| Package | Installed version | Declared license |
|---|---:|---|
| mcp | 1.27.0 | MIT |
| httpx | 0.28.1 | BSD-3-Clause |
| zstandard | 0.25.0 | BSD-3-Clause |
| torch (optional reranker) | 2.12.0 | BSD-3-Clause |
| transformers (optional reranker) | 5.11.0 | Apache-2.0 |
| pytest (development) | 9.0.3 | MIT |

Versions are an environment snapshot, not a lockfile guarantee. Repeat the
metadata audit before changing repository visibility or distributing a build.
