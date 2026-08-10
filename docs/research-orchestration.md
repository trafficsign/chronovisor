# Agentic Evidence and Memory Orchestration

Chronovisor keeps its existing safety layers and adds a bounded, read-only
research plane. This is an orchestration layer, not a second memory authority.

## Authority and call hierarchy

1. L1 core memory and L2 automatic Recall remain on the synchronous prompt
   path with the existing four-second, fail-open contract.
2. L3 explicit `chronovisor_search`, `chronovisor_read`, and `chronovisor_recall_used` remain the
   normal interactive tools.
3. `chronovisor_deep_dive` v2 uses the finite research kernel but is Wiki-only.
4. `chronovisor_research` follows Wiki -> verified claims -> Raw/prior conversation ->
   Web. Raw and Web are permitted only after local evidence access. A fetch URL
   must have appeared in the run's Web search result.
5. After a durable Evidence Bundle receipt, the auditor can feed proposal-only
   Sleep consolidation. It cannot edit Pages directly.

Raw remains the lossless rebuild authority. Pages remain reviewed semantic
memory. Research event logs and checkpoints are rebuildable derived state.
Evidence Artifacts and Bundles adopted by a run are durable, content-addressed
records under `~/.chronovisor/research`.

The zero-wait local prefetch counts as the first authority rung only after its
completed result has been durably added to the run trace. The host never waits
for it. If it is not ready, Raw and Web actions remain rejected rather than
bypassing local evidence access.

## Scheduler and budgets

The local planner, challenger, tie-breaker, and repair calls all use the same
cross-process sync-first scheduler. At most one research generation runs. A
foreground marker prevents new research admission and cancels a running model
subprocess within the configured grace period. Automatic/shadow model research
fails closed unless `CHRONOVISOR_RESEARCH_CAPACITY_PROVEN=1`; explicit and Sleep
work remain available with fixed per-call and total wall deadlines.

Budgets are separate for planner, challenge, tie-break, repair, total model
calls, searches, fetches, iterations, elapsed time, generation tokens, and
observation bytes. Malformed output, duplicate Action, authority violation,
timeout, interruption, and orphaned Action all receive terminal trace records.
A repeated search is never executed twice. When it already produced ranked
local candidates, the kernel may recover by reading the highest-ranked unseen
page and records that substitution as `duplicate_action_recovered`.

## Evidence contract

Claims are classified as `stable`, `freshness-sensitive`, or `user-reported`
and resolved to `supported`, `contradicted`, or `unknown`. User reports and
external verification are separate fields. Local/official sources rank before
untrusted snippets. Every Evidence Artifact records source URI, retrieval time,
SHA-256, byte length, trust, MIME type, quote range, provider/cache metadata,
and run/iteration provenance.

The primary planner gathers evidence. The challenger audits support,
contradictions, and prompt injection. The tie-break model runs only on genuine
disagreement. A reject/inconclusive challenge can narrow `supported` to
`contradicted` or `unknown`; it cannot fabricate support. Answer citations are
rendered deterministically from artifact metadata.

If the planner reaches a safe terminal state before emitting `finish`, the
service never returns an empty answer. It renders a conservative deterministic
claim assessment from the adopted artifacts, preserves the original terminal
reason, and labels the result `answer_mode=deterministic_claim_assessment`.
Both supporting and contradicting evidence receive citations; unrelated
negation elsewhere in a long artifact is not treated as a contradiction.

## Web boundary

Search and fetch are different permissions. Live egress is off unless both
`adapter_enabled` and `live_egress_enabled` are true. Queries containing
secrets, PII, private paths, invisible control characters, or excessive text
are blocked before egress. Fetch blocks credentials, localhost/private/link-
local/metadata IPs, DNS rebinding, cross-host redirects, redirect loops,
unsupported MIME types, and declared/streamed oversized bodies. Provider
outages degrade to Wiki-only research.

The production Web path is a bounded federation of four adopted source packs:
general Web through local SearXNG, code and releases through GitHub, academic
metadata through arXiv and Crossref, and encyclopedic knowledge through
MediaWiki. A deterministic router selects at most four upstream calls and
round-robin merges specialist-first results while removing duplicate URLs.
Unknown source-pack names fail closed instead of silently growing the provider
surface. Brave and Tavily remain supported as optional single-provider
adapters, but are not required by the production federation.

The local SearXNG endpoint is the only loopback exception in the search egress
guard. Normal Web fetches retain the public-address-only SSRF policy. Install or
refresh the pinned local service with `scripts/install-searxng`. Cached Web
bodies are TTL-bound derived data; durable Evidence is stored only when adopted
by a research run.

## Checkpoints and consolidation

Zero-wait prefetch is consumed at most once by the next iteration. Context
compaction keeps complete Action/Observation pairs; an unpaired Action forces
full-history fallback. Checkpoint GC can remove only inactive checkpoints with
a durable receipt, and must converge under its size cap without touching Raw or
durable Evidence.

Sleep consolidation requires a durable research receipt, elapsed-time and
new-session thresholds, and a nonblocking lease. It writes only allowlisted
`query_hint`, `alias`, or `tag` proposals with latest-wins `supersedes`
provenance. The cursor advances only after the proposal append is durable.
Replay/Holdout and local consensus remain the adoption gates.

## Recommended explicit configuration

```toml
[research]
enabled = true
mode = "explicit"

[research.web]
adapter_enabled = true
live_egress_enabled = true
provider = "federated"
source_packs = ["general", "code", "academic", "encyclopedia"]
searxng_endpoint = "http://127.0.0.1:8888"
allow_local_search_backend = true
max_provider_calls = 4
per_provider_limit = 3

[research.compaction]
enabled = true
checkpoint_enabled = true

[research.consolidation]
enabled = true
mutation_mode = "proposal_only"
```

Use `mode = "off"`, `live_egress_enabled = false`, `compaction.enabled =
false`, or `consolidation.enabled = false` as independent rollback switches.
