# Configuration

The preferred runtime config is `~/.wiki/config.toml`. If it does not exist,
LLM Wiki falls back to the legacy `~/.wiki/recall.toml` shape.

## Unified Shape

```toml
[hooks.user_prompt]
recall = true

[hooks.stop]
save = true
# Stop only enqueues deterministic capture jobs. Semantic work is drained by
# local convergence after the hook process exits.
audit = false
content_correction = true
recall_improve = false

[embedding]
# Tuned search profile. If omitted, runtime falls back to nomic-embed-text.
model = "bge-m3"
document_prefix = ""
query_prefix = ""

[ingest]
# Select the smallest safe bucket for each complete ingest request. A larger
# resident runner is reused so a backlog grows monotonically rather than
# shrinking and reloading between raws.
model = "maxwell1500/ornith-35b:Q5_K_M"
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

[decision_router]
# Routine structured decisions require a two-vote local quorum. The complete
# request-token budget, including both possible JSON-repair turns, selects the
# smallest executable configured context bucket. Buckets below the lightest
# full production-lane envelope are omitted; with the 2000-byte feedback budget
# the active set is 32K, 64K, 96K, and 112K. Measured /api/ps
# footprints at that exact size determine whether one, two, or three decision
# runners may remain resident. A larger measured runner is reused within the
# cap to avoid shrink/reload flap, but its actual context is recorded and never
# counted as smaller-bucket adoption evidence. The offline adoption evaluator
# uses a separate exact-bucket mode.
primary_model = "maxwell1500/ornith-35b:Q5_K_M"
challenger_model = "gpt-oss:20b"
tie_break_model = "gemma4:26b"
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
# This is the post-adoption production shape. The v56 artifact is sealed by
# artifact schema 12, evaluator policy 20, decision-semantics policy 11,
# quorum-safety policy 1, action-signature policy 5, effective-request-
# fingerprint policy 4, structured-generation policy 1, lane-contract registry
# policy 9 (artifact identity
# only), lane prompt policy 7 for 17 lanes, 8 for raw replay, 14 for ingest, and
# lane-contract case policy 20 (source deterministic_lane_contract_v20).
# Evaluator policy 20 seals deterministic seed 0 as well as hash-bound ingest
# repair option selection and host-only byte materialization into the artifact
# identity. Repair option
# policy 2 exposes no heuristic semantic receipts; both models judge the exact
# raw and page bytes independently.
# Invalid, partial, stale, or
# engine/model-drifted artifacts make enabled semantic lanes quarantine before
# inference. Set this to "" and keep the 19 model-backed lanes in shadow only
# while compiling and evaluating a replacement candidate.
adoption_artifact = "~/.wiki/runtime/model-lab/local-eval/adoption-v56-evaluator20.json"

[decision_policies]
# Deterministic/non-model lanes and the guarded repair-only lane are live
# immediately. The repair lane still requires trusted incident evidence,
# explicit repair enablement, and the durable FrontierRepairGuard permit.
raw_capture = "enabled"
exact_user_correction = "enabled"
derived_index_rebuild = "enabled"
claims_conflict = "enabled"
system_code_repair = "enabled"

# Post-adoption production has all 24 policy lanes enabled and zero in shadow:
# the five deterministic/guarded lanes above plus these 19 model-backed
# semantic lanes. Each semantic lane still fails closed unless the nominated
# artifact validates as adopted for the exact models, engine, corpus, config,
# context buckets, and policy versions.
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
# Optional. Demotes pages recorded as page_ignored / injection_ignored /
# false-positive feedback when the incoming query is lexically similar
# (Jaccard over search tokens) to the feedback prompt. A correction classified
# as wrong retrieval supplies an explicit negative_pages subset, so unrelated
# candidates from the same recall decision are not penalized. Pages confirmed
# relevant by reviewed golden labels for a similar query remain protected.
enabled = true
similarity_threshold = 0.35
penalty = 0.85
max_age_days = 180
max_entries = 500

[search.reranker]
# Optional. Used by MCP wiki.search and the search-eval hybrid-rerank variant;
# synchronous recall hooks keep using the faster BM25/dense fusion path.
enabled = false
backend = "transformers"
model = "BAAI/bge-reranker-v2-m3"
top_n = 20
max_length = 1024
batch_size = 4
device = "mps"
weight = 0.25

[recall.thresholds]
search = 0.35
read = 0.65

[recall.gate]
model = "ornith:9b-q4_K_M"
think = false
timeout_ms = 3000
num_ctx = 4096
num_predict = 64
include_queries = false
# How long ollama keeps the gate/rewrite model resident after a call.
# Default "24h" avoids cold-start timeouts on the synchronous recall path.
keep_alive = "24h"
# Used by `llm-wiki-recall --warmup` before hook sessions.
warmup_timeout_ms = 15000

[recall.budgets]
max_context_chars = 600
max_pages = 3
max_queries = 3

[recall]
gate_mode = "evidence"
context_style = "cards"
semantic = true
judge_mode = "auto"
session_ttl_seconds = 604800

[recall.rewrite]
enabled = true
model = "ornith:9b-q4_K_M"
timeout_ms = 3000

[recall.fusion]
bm25 = 1.0
semantic = 0.6
graph = 0.0
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
model = "maxwell1500/ornith-35b:Q5_K_M"
think = false
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

[recall_improvement]
models = [
  "maxwell1500/ornith-35b:Q5_K_M",
  "gemma4:26b",
]

[auto_apply]
enabled = true
min_count = 1
actions = ["alias", "query_hint", "page_tag"]
```

## Compatibility

Existing `recall.toml` files with top-level `[gate]`, `[thresholds]`,
`[budgets]`, `[recall]`, `[policy]`, `[auditor]`, and `[auto_apply]` sections
remain supported.

## Environment Overrides

- `LLM_WIKI_RECALL_ENABLED=0`: disable synchronous recall.
- `CODEX_WIKI_SAVE_ENABLED=1`: enable Codex save hook.
- `CLAUDE_CODE_WIKI_SAVE_ENABLED=1`: enable Claude Code save hook.
- `LLM_WIKI_RECALL_AUDIT_ENABLED` and
  `LLM_WIKI_CONTENT_CORRECTION_ENABLED`: legacy compatibility switches. The
  Stop dispatcher is save-only and does not schedule those lanes.
- `LLM_WIKI_RECALL_AUTO_APPLY_FRONTIER_TIMEOUT`: legacy name for the timeout
  passed by auto-apply to the local structured-review compatibility boundary.
  It does not enable a frontier call.
- `LLM_WIKI_CONTENT_CORRECTION_QUARANTINE_RETRY_SECONDS`: cooldown before an
  autonomous content-correction quarantine is reopened (default: `21600`).
- `LLM_WIKI_CONVERGENCE_QUARANTINE_RETRY_SECONDS`: cooldown before autonomous
  recall, replay, repair, and self-heal quarantines are reopened (default:
  `21600`).
- `LLM_WIKI_HUMAN_REQUIRED_RECHECK_SECONDS`: interval for automatically
  rechecking external-authority failures such as expired authentication,
  billing/quota, or keychain access (default: `3600`).
- `LLM_WIKI_RUNTIME_SOURCE`: explicit override for the production `uvx`
  package source. The default is the pushed GitHub repository; local worktrees
  are never selected implicitly.
- `LLM_WIKI_REPO_ROOT`: checkout used only as exceptional system-code-repair
  context and as the target of an approved repair patch. It does not control
  imported runtime code.
- `LLM_WIKI_FRONTIER_MODEL`, `LLM_WIKI_FRONTIER_REASONING_EFFORT`, and
  `LLM_WIKI_FRONTIER_TIMEOUT_SECONDS`: exceptional
  code-repair settings. They are read only after validated
  `RepairIncidentEvidence` passes the durable single-flight/24-hour guard; none
  of them can turn a routine review into a frontier call.
- Arbitrary `LLM_WIKI_FRONTIER_CMD` execution is intentionally unsupported;
  one admitted incident can start only the built-in single Codex process.

Older settings may still expose `frontier_mode`, `frontier_*`, or
`LLM_WIKI_*FRONTIER*` names. They are schema and artifact compatibility names
unless they belong to the guarded code-repair settings listed above. Routine
`run_structured_review()` calls always use `[decision_router]`.

The compatibility wrappers preserve old command-line and environment parsing,
but they do not weaken the save-only Stop invariant. Direct
`llm-wiki-hook --event Stop` deployments should enable only `hooks.stop.save`.

## Optional Reranker

Install the optional local reranker dependencies with `uv sync --extra reranker`
before enabling `[search.reranker]`. The production profile keeps it disabled:
BM25 + semantic fusion is the default ranking path, and the synchronous recall
hook never calls the reranker. If enabled, the Hugging Face Transformers backend
uses `BAAI/bge-reranker-v2-m3` only for MCP `wiki.search` top candidates and
explicit search-eval reranker experiments.

## Recall Defaults

The completed recall path defaults to `gate_mode = "evidence"`,
`context_style = "cards"`, `semantic = true`, rewrite enabled, and calibration
enabled. BM25 still runs without Ollama; semantic and rewrite fail open when the
local model is unavailable.
