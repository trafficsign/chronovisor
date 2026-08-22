# Configuration

The runtime config is `~/.chronovisor/config.toml`.

The default `[dashboard]` section remains loopback-only. Secure LAN access uses
a separate section so it cannot change the existing service boundary:

```toml
[dashboard_lan]
host = "192.168.50.20" # exact private IPv4 assigned to this Mac; no wildcard
port = 8766
tls_cert_file = "/Users/me/.chronovisor/runtime/dashboard-lan.crt"
tls_key_file = "/Users/me/.chronovisor/runtime/dashboard-lan.key"
credentials_file = "/Users/me/.chronovisor/runtime/dashboard-lan-credentials.json"
```

All paths must be absolute. The private key and hashed credential file must be
regular, owner-owned `0600` files and must not be symlinks. Passwords never
belong in TOML.

The reader opens that path once and parses one immutable byte snapshot. When an
operator atomically replaces the file, a load therefore observes either the old
generation or the new generation, never an existence check from one generation
and bytes from another. One structured review resolves a single
`DecisionRouterConfig` for both its initial authority seal and the router it
constructs. Later in-flight authority guards intentionally read current state
again, so a real mid-review authority change still fails closed. Chronovisor has
no repository-owned config writer; operators should continue to publish config
changes by atomic replacement rather than in-place partial writes.

## Raw Archive rollout

Transcript storage uses one durable Wiki-wide setting so Stop hooks, MCP,
ingest, sleep, and dashboard processes cannot drift onto different layouts:

```toml
[raw]
layout = "v2" # canonical default; legacy | shadow are explicit recovery modes
```

`CHRONOVISOR_RAW_LAYOUT` remains an emergency per-process override:

```sh
CHRONOVISOR_RAW_LAYOUT=legacy  # explicit recovery: flat Markdown authority
CHRONOVISOR_RAW_LAYOUT=shadow  # flat authority plus source-native v2 mirror
CHRONOVISOR_RAW_LAYOUT=v2      # default; source-native date-partitioned authority
```

An unknown value fails before capture. `chronovisor_record` remains the compatible
manual UTF-8/Markdown API in every mode; only the Codex and Claude transcript
savers use source-native segments. Compression is never performed by Stop or
the save worker. The sleep cycle seals at most four eligible segments per run.

This Raw archive `layout = "v2"` setting is unrelated to the Recall Field store
schema. Field snapshots migrate on first read from the read-only
`recall/field/sessions/` v1 namespace into sealed schema-2 files under
`recall/field/sessions-v2/`; current code writes only `sessions-v2/` (and
`events-v2/`) and never rewrites the v1 migration source.

## Unified Shape

```toml
[hooks.user_prompt]
recall = true

[hooks.stop]
save = true
# Stop only enqueues deterministic capture jobs. After a successful durable
# `saved` receipt, the worker queues both an asynchronous Recall audit candidate
# and privacy-safe answer-episode capture. A `recovered` receipt queues answer
# capture only; it does not queue another audit candidate. Neither follow-up
# runs replay/scoring.
# Semantic work is drained by local convergence after the hook process exits.
audit = false
content_correction = true
recall_improve = false

[embedding]
# Compatibility-only keys. They are accepted but ignored by provider-neutral
# knowledge embedding; the fixed route below is the only model selector.
model = "bge-m3"
document_prefix = ""
query_prefix = ""

[llm.roles."knowledge.embedding"]
capability = "embedding"
provider = "local"
model = "bge-m3"

[ingest]
# Ingest generation routing is fixed by llm.roles."ingest.generation". Legacy
# `model` values are accepted but ignored and cannot select a runtime route.
# Select the smallest safe bucket for each complete ingest request. A larger
# resident runner is reused so a backlog grows monotonically rather than
# shrinking and reloading between raws.
keep_alive = "20m"
temperature = 0.3
num_ctx = 32768
max_num_ctx = 262144
num_predict = 8192
read_timeout_ms = 660000
memory_reserve_gib = 16
max_related_context_bytes = 8192
# Verified transcript captures are delegated as content-addressed semantic
# children. 24 KB is the hard safe ceiling for the downstream decision
# envelope; lower values create more byte-exact children and never truncate.
semantic_projection_max_child_bytes = 24000

[llm.roles."ingest.generation"]
capability = "generation"
provider = "local"
model = "qwen3.8:27b-axq4"

[llm.roles."lint.tag_repair"]
capability = "generation"
provider = "local"
model = "qwen3.8:27b-axq4"

[llm.roles."lint.orphan_link"]
capability = "generation"
provider = "local"
model = "qwen3.8:27b-axq4"

[llm.roles."recall.content_correction.proposer"]
capability = "generation"
provider = "local"
model = "qwen3.8:27b-axq4"

[llm.roles."recall.auditor"]
capability = "generation"
provider = "local"
model = "qwen3.8:27b-axq4"

[llm.roles."recall.gate"]
capability = "generation"
provider = "local"
model = "ornith:9b-q4_K_M"

[llm.roles."recall.query_rewriter"]
capability = "generation"
provider = "local"
model = "ornith:9b-q4_K_M"

[llm.roles."recall.policy_proposer.primary"]
capability = "generation"
provider = "local"
model = "qwen3.8:27b-axq4"
required_capabilities = ["structured_output"]

[llm.roles."recall.policy_proposer.challenger"]
capability = "generation"
provider = "local"
model = "gemma4:26b-optiq4"
required_capabilities = ["structured_output"]

[llm.roles."recall.rubric.variant"]
capability = "generation"
provider = "local"
model = "gemma4:26b-optiq4"
required_capabilities = ["structured_output"]

# Offline Recall distillation uses raw/high conversation inputs and is strictly
# local-only: a remote route is forbidden even when a role/data-class egress
# opt-in exists for another Recall role.
[llm.roles."recall.distill.teacher.a"]
capability = "generation"
provider = "local"
model = "qwen3.8:27b-axq4"
required_capabilities = ["structured_output"]

[llm.roles."recall.distill.teacher.b"]
capability = "generation"
provider = "local"
model = "gpt-oss:20b"
required_capabilities = ["structured_output"]

[llm.roles."recall.distill.teacher.c"]
capability = "generation"
provider = "local"
model = "gemma4:26b-optiq4"
required_capabilities = ["structured_output"]

[llm.roles."recall.distill.answer_generator"]
capability = "generation"
provider = "local"
model = "qwen3.8:27b-axq4"
required_capabilities = ["structured_output"]

[llm.roles."recall.distill.utility_judge"]
capability = "generation"
provider = "local"
model = "gemma4:26b-optiq4"
required_capabilities = ["structured_output"]

[llm.roles."research.planner"]
capability = "generation"
provider = "local"
model = "qwen3.8:27b-axq4"
required_capabilities = ["structured_output"]

[llm.roles."research.challenge"]
capability = "generation"
provider = "local"
model = "gpt-oss:20b"
required_capabilities = ["structured_output"]

[llm.roles."research.tie_break"]
capability = "generation"
provider = "local"
model = "gemma4:26b-optiq4"
required_capabilities = ["structured_output"]

[llm.roles."research.deep_retrieval_requery"]
capability = "generation"
provider = "local"
model = "qwen3.8:27b-axq4"
required_capabilities = ["structured_output"]

# All research prompts are raw/high. Remote routes require the exact
# opt-ins; denial reaches no backend and has no local fallback.
# [[llm.egress_opt_in]]
# role = "research.planner"
# data_class = "raw"
# [[llm.egress_opt_in]]
# role = "research.challenge"
# data_class = "raw"
# [[llm.egress_opt_in]]
# role = "research.tie_break"
# data_class = "raw"
# [[llm.egress_opt_in]]
# role = "research.deep_retrieval_requery"
# data_class = "raw"
# [[llm.egress_opt_in]]
# role = "recall.policy_proposer.primary"
# data_class = "raw"
# [[llm.egress_opt_in]]
# role = "recall.policy_proposer.challenger"
# data_class = "raw"
# [[llm.egress_opt_in]]
# role = "recall.rubric.variant"
# data_class = "raw"

[decision_router]
# Routine structured decisions require a two-vote configured-route quorum. The complete
# request-token budget, including both possible JSON-repair turns, selects the
# smallest executable configured context bucket. Buckets below the lightest
# full production-lane envelope are omitted; with the 2000-byte feedback budget
# the active set is 32K, 64K, 96K, and 112K. Measured /api/ps
# footprints at that exact size determine whether one, two, or three decision
# runners may remain resident. A larger measured runner is reused within the
# cap to avoid shrink/reload flap, but its actual context is recorded and never
# counted as smaller-bucket evaluation evidence. The offline adoption evaluator
# uses a separate exact-bucket mode. Production model selection comes only from
# llm.roles.classification.primary/challenger/tie_break below. Model triplets
# belong only in explicit model-evaluation/adoption candidate files.
primary_keep_alive = "20m"
challenger_keep_alive = "20m"
tie_break_keep_alive = "2m"
num_ctx = 114688
min_num_ctx = 16384
num_predict = 3072
read_timeout_ms = 660000
max_input_chars = 93000
max_output_chars = 4000
max_feedback_chars = 2000
quorum = 2
adaptive_residency = true
residency_policy_version = 2
memory_reserve_gib = 16
max_resident_models = 3
# The current runtime contract is sealed by artifact schema 12, evaluator
# policy 21, decision-semantics policy 12, quorum-safety policy 2,
# action-signature policy 5, effective-request-fingerprint policy 4,
# structured-generation policy 3, lane-contract registry policy 10 (artifact
# identity only), lane prompt policy 8 for 16 lanes, 9 for raw replay and recall
# auto-apply, 16 for ingest, and lane-contract case policy 27 (source
# deterministic_lane_contract_v27). An artifact carrying the former quorum-v1
# or lane-contract-v26 identity cannot authorize this runtime and fails closed.
# Evaluator policy 21 seals deterministic seed 0 as well as hash-bound ingest
# repair option selection and host-only byte materialization into the artifact
# identity. Repair option
# policy 2 exposes no heuristic semantic receipts; both models judge the exact
# raw and page bytes independently.
# This artifact remains offline evaluation and rollout evidence. It does not
# select production routes or gate use of a configured route authority.
adoption_artifact = "~/.chronovisor/runtime/model-lab/local-eval/adoption-quorum2-lane27-evaluator21-YYYYMMDD.json"

[decision_policies]
# Deterministic/non-model lanes and the guarded repair-only lane are live
# immediately. The repair lane still requires trusted incident evidence,
# explicit repair enablement, and the durable FrontierRepairGuard permit.
raw_capture = "enabled"
exact_user_correction = "enabled"
derived_index_rebuild = "enabled"
claims_conflict = "enabled"
system_code_repair = "enabled"

# Production has all 24 policy lanes enabled and zero in shadow: the five
# deterministic/guarded lanes above plus these 19 model-backed semantic lanes.
# Each semantic effect is bound to the exact configured route authority;
# offline adoption/re-certification is not a utilization condition.
autonomy_duplicate_resolution = "enabled"
autonomy_retention = "enabled"
content_correction_classification = "enabled"
content_correction_review = "enabled"
entity_backfill = "enabled"
ingest_reconciliation = "enabled"
lint_safe_semantic_mutation = "enabled"
lint_tag_repair = "enabled"
local_repair = "enabled"
metadata_backfill = "enabled"
orphan_link = "enabled"
page_normalize = "enabled"
raw_replay_reconciliation = "enabled"
read_back_repair = "enabled"
recall_auto_apply = "enabled"
recall_calibration = "enabled"
recall_improvement = "enabled"
search_label = "enabled"
search_self_tune = "enabled"

[ingest.audit]
# Routine quality sampling stays cheap. Mandatory correction, incomplete
# generation, privileged system/security targets, and large mutations are not
# controlled by these sampling rates.
enabled = true
sample_rate = 0.05
update_sample_rate = 0.08
noop_sample_rate = 0.05
adaptive = true
adaptive_window = 50
adaptive_min_audits = 5
elevated_reject_rate = 0.10
critical_reject_rate = 0.20
elevated_sample_rate = 0.08
critical_sample_rate = 0.10
# Hard circuit breaker for routine sampling even when historical catches spike.
max_sample_rate = 0.10
max_operations_without_audit = 4

[search.negative_feedback]
# Optional search-ranking suppressions. The separate Recall Field/co-fire path
# accepts only a strong Frontier-reviewed page_ignored/contradiction with exact
# decision/session identity and current page-content hash. Passive read,
# injection_ignored, unreviewed page_ignored, and certificate reject remain
# exposure only. An exact producer-key/feedback-digest retraction restores only
# the contribution it names.
enabled = true
similarity_threshold = 0.35
penalty = 0.85
max_age_days = 180
max_entries = 500

[search.reranker]
# Provider-independent rollout shared by MCP, eval, and the resident service.
enabled = false
top_n = 10
weight = 1.0

[search.reranker.service]
enabled = true
socket = "~/.chronovisor/runtime/reranker.sock"
timeout_ms = 1000
# off | shadow | canary | on
mode = "shadow"
canary_percent = 0
queue_size = 8

[research]
# Explicit keeps the bounded MCP/Sleep lane usable without automatic 35B work.
# auto/shadow still fail closed until protected capacity is proved.
enabled = true
mode = "explicit"
# Provider/model selection is fixed by llm.roles."research.*". Legacy model
# keys are accepted but ignored.
max_depth = 1

[research.budgets]
max_iterations = 5
max_total_wall_seconds = 90
max_single_generation_seconds = 30
max_single_generation_tokens = 256
max_planner_calls = 5
max_challenge_calls = 2
max_tie_break_calls = 1
max_repair_calls = 2
max_total_model_calls = 10
max_observation_bytes = 200000

[research.resources]
scheduler = "sync_first"
max_concurrent_generations = 1
preempt_on_sync = true
preempt_grace_ms = 250
protected_models = ["ornith:9b-q4_K_M", "bge-m3"]
require_protected_residency = true
sync_reserved_headroom_gib = 16
sync_lease_wait_limit_ms = 50
coordinate_ollama = true
coordinate_mps_reranker = true

[research.web]
adapter_enabled = true
live_egress_enabled = true
provider = "federated"
source_packs = ["general", "code", "academic", "encyclopedia"]
searxng_endpoint = "http://127.0.0.1:8888"
github_endpoint = "https://api.github.com/search/repositories"
github_token_env = "GITHUB_TOKEN"
arxiv_endpoint = "https://export.arxiv.org/api/query"
crossref_endpoint = "https://api.crossref.org/works"
mediawiki_endpoint = "https://ja.wikipedia.org/w/api.php"
allow_local_search_backend = true
max_provider_calls = 4
per_provider_limit = 3
provider_timeout_seconds = 8
max_searches = 8
max_fetches = 5
cache_ttl_seconds = 900
allow_private_network = false
max_fetch_bytes = 2000000

[research.compaction]
enabled = true
checkpoint_enabled = true
checkpoint_ttl_seconds = 604800
checkpoint_max_total_bytes = 536870912
gc_on_durable_receipt = true

[research.consolidation]
enabled = true
mutation_mode = "proposal_only"
min_interval_seconds = 86400
min_new_sessions = 5
max_jobs = 20

[research.security]
egress_guard = true
external_content_trust = "untrusted"

[recall.thresholds]
search = 0.35
read = 0.65

[recall.gate]
think = false
timeout_ms = 3000
num_ctx = 4096
num_predict = 64
include_queries = false
# How long local Ollama keeps the gate/rewrite model resident after a call.
# Default "24h" avoids cold-start timeouts on the synchronous recall path.
keep_alive = "24h"
# Used by `chronovisor-recall --warmup` before hook sessions.
warmup_timeout_ms = 15000

[recall.budgets]
# L2 automatic Recall budget.
max_context_chars = 800
# L1 always-on state budget.
max_state_context_chars = 1600
# Whole-block merge ceiling. Values below L1 + L2 + delimiters are normalized
# upward so neither layer silently steals the other's configured budget.
max_total_context_chars = 2402
max_pages = 3
max_queries = 3
# One wall-clock budget shared by rewrite, embedding, search, and judge.
total_timeout_ms = 4000
# Reserved from the total deadline for model-free L1 + BM25 degradation.
deterministic_fallback_reserve_ms = 600

[recall]
gate_mode = "evidence"
context_style = "cards"
semantic = true
judge_mode = "auto"
session_ttl_seconds = 604800

[recall.distillation]
# Offline-only. This controls bounded dataset construction, not live policy
# mutation; promotion remains gated by replay, shadow, and canary evidence.
enabled = false
chunk_size = 25
max_input_bytes = 12000
max_candidates = 200
hard_floor_rallies = 1000
hard_floor_days = 30
hard_floor_windows = 3
hard_floor_teacher_labels = 500
hard_floor_teacher_per_class = 100
hard_floor_probe_pairs = 100
hard_floor_counterfactual_pairs = 100
rollout_stages = [5, 25, 100]
canary_min_days = 7

[recall.processor]
# Keep false until pointer/rich precision and related-memory recall pass the
# locked gate. Shadow collection never changes injection, while auto_enable
# delegates future authority to the sealed autonomous rollout.
enabled = false
shadow_enabled = true
auto_enable = true
max_candidates = 10
max_pointer_cards = 6
max_rich_evidence = 2
injection_token_budget = 1200
certificate_required = true
judge_enabled = true
# Model routing is fixed by llm.roles."recall.certificate_judge.primary" and
# llm.roles."recall.certificate_judge.escalation". Legacy judge_model and
# escalation_model keys are accepted but ignored.
judge_timeout_ms = 900
escalation_timeout_ms = 900

[recall.field]
# off | shadow | candidate | active. active still requires a sealed passing
# promotion artifact and an enabled certificate boundary.
mode = "shadow"
canary_percent = 0
working_set_size = 30
max_active_nodes = 128
max_active_edges = 256
positive_learning = false
wall_half_life_seconds = 300
turn_decay = 0.82
spread_gain = 0.35
max_hops = 2
global_inhibition = 0.08
refractory_turns = 1
topic_reset_similarity = 0.15
session_ttl_seconds = 604800
event_retention = 2000

[recall.field.growth]
# Sleep-cycle supervision advances 5% -> 25% -> 100% only after every gate.
# `recall_used` is diagnostic only. Positive learning requires sealed train
# answer outcomes; production authority additionally requires the independent
# sealed manual-94 retrieval artifact, a separate sealed locked-test answer
# artifact, an exact cross-split manifest match, and connected-cluster
# confidence bounds for candidate and Processor point/LCB floors. Manual-94 is
# frozen first at `runtime/search-eval/manual-94-manifest.json`: schema 2 must
# contain exactly 94 unique reviewed entries, exact entry/manifest seals, and
# the review-ledger byte length, line count, file hash, and chain head. The
# manifest is frozen before evaluation. Evaluation then rebinds every ranked
# page to its live UID/content digest; a simultaneously generated or changed
# cohort fails closed.
enabled = true
auto_promote = true

[recall.circuit_breaker]
failures = 2
cooldown_seconds = 60

[recall.rewrite]
enabled = true
timeout_ms = 3000

[recall.fusion]
anchor = 0.9
bm25 = 1.0
semantic = 0.6
graph = 0.3
context = 0.25
usage_prior = 0.0
bm25_score_bonus = 0.005
bm25_rank_bonus = 0.006
bm25_rank_decay = 0.006
semantic_min_top_score = 0.45
semantic_min_margin = 0.002
semantic_low_confidence_weight = 0.25
usage_prior_decay = 0.98
usage_prior_cap = 3.0

[recall.calibration]
enabled = true
min_samples = 500
holdout_ratio = 0.2
min_improvement = 0.02

[recall.policy]
avoid_heavy_personal_context_in_chitchat = true
log_decisions = true
use_feedback_suppressions = true
fail_silent_on_judge_unavailable = true

[audit]
enabled = true
timeout_ms = 120000
num_ctx = 32768
keep_alive = "20m"
num_predict = 1024
top_k = 5
semantic = true
min_confidence = 0.70
max_prompt_chars = 4000
max_response_chars = 6000
recent_log_limit = 500

[auto_apply]
enabled = true
min_count = 1
actions = ["alias", "query_hint", "page_tag"]
```

Auditor model routing is fixed by `llm.roles."recall.auditor"`. Legacy
`[audit].model`, `[audit].think`, `[heavy].model`, and `[heavy].think` values
are accepted but ignored. Auditor inputs are classified as `raw/high`; remote
providers require an explicit `recall.auditor` + `raw` egress opt-in and never
fall back to another provider.

Synchronous recall routing is fixed by `llm.roles."recall.gate"` and
`llm.roles."recall.query_rewriter"`. Legacy top-level `model`,
`[recall].model`, `[recall.gate].model`, and `[recall.rewrite].model` values
are accepted but ignored. Both inputs are `raw/high`; remote providers require
an explicit role + `raw` egress opt-in and never fall back to another provider.

Recall policy proposals use the fixed, ordered roles
`llm.roles."recall.policy_proposer.primary"` and
`llm.roles."recall.policy_proposer.challenger"`. Both require structured
output and classify inputs as `raw/high`; remote providers require each exact
role + `raw` opt-in and never fall back to another provider. Legacy
`[recall_improvement].models` and `CHRONOVISOR_RECALL_IMPROVEMENT_MODELS`
cannot select runtime models, and CLI `--models` is unavailable.

Offline Recall rubric variants use only the fixed structured role
`llm.roles."recall.rubric.variant"`; there is no model fallback. Every prompt
contains the raw user query. Page-namespace cases are therefore classified as
`raw/high`, and a remote route requires the exact role + `raw` opt-in. A
system-namespace excerpt is `system/high` and currently runs locally only:
until generation supports separate raw-query and system-excerpt preflight, a
remote system case is denied before any backend call even when egress opt-ins
exist.

Offline Recall distillation is separately fixed to
`recall.distill.teacher.a`, `recall.distill.teacher.b`,
`recall.distill.teacher.c`, `recall.distill.answer_generator`, and
`recall.distill.utility_judge`. All five roles require structured output and
must resolve to local providers. Their inputs are `raw/high`; remote routing
and raw egress are forbidden rather than opt-in capable. The answer generator
and utility judge use distinct local model identities so matched answer pairs
and their blind utility verdict do not share one route.

All model labels and counterfactual outcomes are teacher-only. Probe pairs are
excluded from training and are used only for route stability and locked replay.
Authentic negative outcomes are veto-only; no human review is used. Each shadow
and canary rollout stage runs for at least seven days.

Deep Retrieval v1 requery generation is fixed by
`llm.roles."research.deep_retrieval_requery"`; Decision Router model fields do
not select it. Inputs are `raw/high`, so remote providers require the exact
role + `raw` opt-in. Denial or provider failure uses only the deterministic
query fallback, never another provider or a local model.

## Decision quorum safety

Decision effects are bound to the exact ordered runtime routes
`classification.primary`, `classification.challenger`, and
`classification.tie_break`. Durable authority and receipts record only safe
route identity: role, provider, model, local/remote location, protocol, a hash
of the configured endpoint, and the optional role `revision`; the raw endpoint
is never persisted. Local Ollama routes additionally bind the installed engine,
model digest, and quantization. Other local providers do not call Ollama
metadata or residency controls.

A remote classification voter must set an operator-owned immutable `revision`
on its existing `[llm.roles.<role>]` table. The provider response must also
report the exact configured model. A missing revision, missing/mismatched
returned model, schema/egress failure, or provider failure invalidates that
route without retrying or selecting another route. Provider-returned revision
or system-fingerprint metadata is invocation telemetry only and never supplies
authority. Other remote roles may omit `revision`. The three classification
roles must resolve to distinct provider/model identities and each selected
profile/model must advertise structured-output capability; the example shows
only one remote role in a mixed route set.

Authority, execution-fingerprint, canonical-decision-artifact, semantic-hold,
and machine-consensus receipt versions are fail-closed: older adopted-artifact
or model-triplet seals are not reinterpreted as current runtime-route authority.

The quorum-v2 lane exception is code-owned, not configurable through TOML. It
contains exactly `lint_tag_repair`, `recall_auto_apply`, `orphan_link`,
`metadata_backfill`, and `search_label`, whose reviewed effect contracts are
additive or reversible. It applies only after a tie-break produces a 2-to-1
mutating majority. `ingest_reconciliation`, every non-listed lane, and a
missing, empty, or unknown lane retain the conservative veto. An unclassifiable
(`None`) dissent also remains fail-closed outside the five named lanes.

Changing the set requires a source change, a quorum-safety policy-version bump,
updated v27-or-later lane-contract evidence, and newly generated route-bound
authority/receipt artifacts. A local config override cannot silently broaden it.

## Environment Overrides

- `CHRONOVISOR_RECALL_ENABLED=0`: disable synchronous recall.
- `CHRONOVISOR_RECALL_ANSWER_CAPTURE_ENABLED=1`: permit the capture-only answer
  episode hook. The Stop dispatcher sets this only on the post-save follow-up;
  the flag does not enable replay or scoring.
- `CODEX_CHRONOVISOR_RECORD_ENABLED=1`: enable the Codex record hook.
- `CLAUDE_CODE_CHRONOVISOR_RECORD_ENABLED=1`: enable the Claude Code record hook.
- `CHRONOVISOR_RECALL_AUDIT_ENABLED` and
  `CHRONOVISOR_CONTENT_CORRECTION_ENABLED`: legacy compatibility switches. The
  Stop dispatcher is save-only and does not schedule those lanes.
- `CHRONOVISOR_RECALL_AUTO_APPLY_FRONTIER_TIMEOUT`: legacy name for the timeout
  passed by auto-apply to the local structured-review compatibility boundary.
  It does not enable a frontier call.
- `CHRONOVISOR_CONTENT_CORRECTION_QUARANTINE_RETRY_SECONDS`: cooldown before an
  autonomous content-correction quarantine is reopened (default: `21600`).
- `CHRONOVISOR_CONVERGENCE_QUARANTINE_RETRY_SECONDS`: cooldown before autonomous
  recall, replay, repair, and self-heal quarantines are reopened (default:
  `21600`).
- `CHRONOVISOR_HUMAN_REQUIRED_RECHECK_SECONDS`: interval for automatically
  rechecking external-authority failures such as expired authentication,
  billing/quota, or keychain access (default: `3600`).
- `CHRONOVISOR_RUNTIME_SOURCE`: explicit override for the production `uvx`
  package source. The default is the pushed GitHub repository; local worktrees
  are never selected implicitly.
- `CHRONOVISOR_REPO_ROOT`: checkout used only as exceptional system-code-repair
  context and as the target of an approved repair patch. It does not control
  imported runtime code.
- `CHRONOVISOR_FRONTIER_MODEL`, `CHRONOVISOR_FRONTIER_REASONING_EFFORT`, and
  `CHRONOVISOR_FRONTIER_TIMEOUT_SECONDS`: exceptional
  code-repair settings. They are read only after validated
  `RepairIncidentEvidence` passes the durable single-flight/24-hour guard; none
  of them can turn a routine review into a frontier call.
- Arbitrary `CHRONOVISOR_FRONTIER_CMD` execution is intentionally unsupported;
  one admitted incident can start only the built-in single Codex process.

Older settings may still expose `frontier_mode`, `frontier_*`, or
`CHRONOVISOR_*FRONTIER*` names. They are schema and artifact compatibility names
unless they belong to the guarded code-repair settings listed above. Routine
`run_structured_review()` calls always use `[decision_router]`.

Direct `chronovisor-hook --event Stop` deployments may enable save and
deterministic correction capture, but never semantic work.

`chronovisor-recall-answer-eval --status` reports captured episode bindings and
the locked-test artifact. Hook mode accepts only an exact `saved`/`recovered`
worker receipt and supports capture only. Build the fixed split with
`--build-split-manifest`; offline evaluation requires `--evaluate`, a sealed
`--gold-manifest`, a separately sealed and pre-frozen `--scorer-calibration`,
and registered `module:callable` runner/scorer adapters plus their exact JSON
identities. A `field-e2e-replay` locked evaluation additionally requires the
built-in `builtin_field_environment_replay` adapter and the exact live identity
returned by `builtin_field_environment_identity()`; a look-alike adapter or
stale model, policy, config, corpus, index, last-known-good, or candidate-policy
identity fails closed.

The durable answer inputs are `recall/answer-episodes.jsonl`,
`recall/answer-review-receipts.jsonl`, and
`recall/answer-execution-receipts.jsonl`. Episode rows use schema 1. Capture
cursors and the preregistered schema-2 split live at
`runtime/recall-answer-eval/capture-cursors.json` and
`runtime/recall-answer-eval/split-manifest.json`. Sealed answer-evaluation
artifacts use schema 3: train output is
`runtime/recall-answer-eval/train-answer-eval.json`, while locked Field evidence
is `runtime/recall-field/locked-answer-eval.json`.

Training may consume only a sealed passing `train` artifact and may therefore
advance candidate-only learning before production authority exists. It never
makes train data locked authority. Until the independent manual-94, locked
field-e2e, cross-split, confidence, freshness, and promotion checks all pass,
production results remain teacher-owned/full-search fallback and Field remains
candidate/shadow. Only a valid promotion permits active Field authority.

## Optional Reranker

Install the optional local reranker dependencies with `uv sync --extra reranker`
before enabling `[search.reranker]`. The fixed `search.rerank` LLM role is the
only provider/model selector. For a local Transformers route, configure
`backend`, `device`, `dtype`, `max_length`, and `batch_size` only on its
`[llm.providers.<id>]` table. Legacy copies of those keys, or `model`, under
`[search.reranker]` are accepted but ignored.

Use `dtype = "float16"` on Apple Silicon to halve the resident reranker weights;
`float32` remains the portable default.

The default topology reranks only the first 10 fused candidates with equal
reciprocal-rank weight (`weight = 1.0`). The resident service mode is a rollout
choice, not a provider fallback: service failure preserves original order and
reports degraded metadata. Remote routes classify the query as `raw/normal`
and each candidate from canonical index metadata; raw queries, high-sensitivity
pages, and system documents require explicit role/data-class egress opt-ins.

## Recall Defaults

The production recall profile uses `gate_mode = "evidence"`,
`context_style = "cards"`, rewrite disabled, `judge_mode = "auto"`, and
calibration enabled. The `[recall].semantic` value remains the conservative
default, while `[search.embedding.rollout].sync_recall` is the authoritative
Nemotron L2 safety boundary applied after learned policy overrides. L3 can
therefore remain active independently of the hook.

The production host enabled synchronous Nemotron Recall on 2026-07-24 after
three counterbalanced 200-pair gates: idle, concurrent CPU indexing, and
concurrent Ornith 35B generation. All 600 candidate calls returned `hybrid`
results, had zero errors and zero four-second violations, and kept
`resource_wait_ms` p95 at 0 ms. Candidate wall-clock p95 was 1.13 s, 1.30 s,
and 1.30 s respectively. The complete synchronous path still has a 4000 ms
wall-clock deadline, reserves 600 ms for allowlisted L1 memory plus BM25
fallback, and remains fail-open for the host. BM25 still runs without the
semantic service; after two degraded or failed runs the breaker temporarily
disables rewrite, semantic search, and judge while keeping BM25 available.

## Search embeddings

The search encoder is independent from the fixed `knowledge.embedding` route
used for duplicate and tag workflows. Legacy `[embedding]` keys are accepted
but ignored by that provider-neutral path; `llm.roles."knowledge.embedding"`
is its only provider/model selector. Set `enabled = false` for explicit
BM25-only execution. When enabled, both semantic runtime roles must resolve to
the same provider, model, location, and vector dimensions. Service or egress
failure is reported as a semantic failure and never selects another provider
or the old SQLite embedding path. Legacy `[search.embedding]` `backend`,
`model`, and `fallback` keys are accepted but ignored; the two fixed roles are
the only provider/model selectors.

```toml
[search.embedding]
enabled = true
revision = "a5e0f804b9e90a1ca6784ecbf6e41595774fc834"
dimensions = 2048
storage_dtype = "float32"
query_prefix = "query: "
document_prefix = "passage: "
fusion_weight = 0.6
min_top_score = 0.20
min_margin = 0.001
low_confidence_weight = 0.25

[search.embedding.service]
socket = "~/.chronovisor/runtime/semantic.sock"
query_device = "mps"
query_replicas = 1
foreground_batch_window_ms = 2
foreground_max_batch = 4
incremental_device = "cpu"
incremental_enabled = true
incremental_max_batch = 1
incremental_pause_during_research = true
incremental_pause_during_ingest_generation = true
incremental_idle_unload_seconds = 300
maintenance_max_batch = 32
offline = true
query_timeout_ms = 250
interactive_timeout_ms = 1000

[search.embedding.rollout]
mode = "on"
canary_percent = 0
sync_recall = true

[llm.roles."search.semantic.foreground"]
capability = "embedding"
provider = "semantic_foreground"
model = "nvidia/Nemotron-3-Embed-1B-BF16"

[llm.roles."search.semantic.incremental"]
capability = "embedding"
provider = "semantic_incremental"
model = "nvidia/Nemotron-3-Embed-1B-BF16"
```

The service stores immutable generations below
`~/.chronovisor/.index/semantic/`. `active.json` is the atomic generation
pointer; incremental updates live in a generation-scoped delta database.
Each complete generation contains a 512-dimensional float16 HNSW candidate
index when `usearch` is available. HNSW rows are never authoritative: candidates
are rescored with the complete 2048-dimensional vectors before fusion. Graph
candidates are verified through the same full-dimensional scorer using the
10-second bounded query-vector cache, so verification does not trigger a second
model inference.
`chronovisor-semantic-service rebuild` queues a full rebuild and
`chronovisor-semantic-service upgrade-ann` clones a complete flat generation
into a sealed HNSW generation without re-embedding.
`chronovisor-semantic-service rollback` atomically returns to the previous
complete generation. The generation and query cache are sealed to the
foreground role/provider/model/location identity; identity drift requires a
rebuild. Local Nemotron keeps the MPS foreground and CPU incremental devices.
Remote providers require explicit `raw/normal`, `page/high`, and `system/high`
egress opt-ins as applicable; denied data sends no request and has no fallback.
Keep the last two generations; the old BGE SQLite file is not a runtime path.
New installations should
keep `sync_recall = false` until the same idle, CPU-indexing, and 35B-load
paired latency gate passes on their hardware.

Lexical retrieval is stored separately in
`~/.chronovisor/.index/lexical.sqlite`. It uses a persistent SQLite inverted
index that preserves the previous BM25 formula without scanning every page,
with integer term/page keys and Chronovisor's Japanese CJK-bigram
pre-tokenization. An exact anchor table covers page IDs, titles, tags,
entities, and raw keywords. The old `bm25.json` projection is deleted after
the SQLite projection is built successfully.

## Typed knowledge graph

```toml
[knowledge_graph]
enabled = true
mode = "shadow" # off | shadow | candidate | active
local_extraction_enabled = true
max_changed_pages_per_cycle = 25
max_queue_size = 500
max_community_summaries_per_cycle = 2
max_model_seconds_per_day = 7200
min_relation_strong = 20
min_relation_sessions = 5
min_entity_strong = 20
min_entity_sessions = 5
min_rubric_gold = 30

[knowledge_graph.retrieval]
mode = "shadow" # off | shadow | candidate | active
max_hops = 2
max_relations_per_node = 12
max_candidate_pages = 50
per_predicate_cap = 4
hub_penalty = 0.15
```

Provider and model selection is fixed by the shared runtime roles below; the
legacy `extractor_model`, `community_summary_model`, and
`external_models_allowed` keys are accepted but ignored.

```toml
[llm.roles."knowledge.relation_extraction"]
capability = "generation"
provider = "local"
model = "gemma4:26b-optiq4"

[llm.roles."knowledge.community_summary"]
capability = "generation"
provider = "local"
model = "gemma4:26b-optiq4"
```

Builder and community-summary budgets share the daily model-seconds ceiling.
The worker also pauses while foreground model activity holds the shared
resource lease. Page excerpts are classified before inference; `system/`
excerpts remain `system/high`, while ordinary Wiki pages are `page/high`.
Remote routes therefore require explicit role-by-data-class egress opt-in for
both classes; a denial sends no outbound request and does not fall back.

The configured mode is not authority by itself. Promotion is controlled by a
sealed evaluation artifact and proceeds automatically by session hash through
5%, 25%, and 100% canaries. Any failed evaluation, resource, model, stale-store,
or learning gate returns retrieval to shadow and preserves the existing search
teacher.
