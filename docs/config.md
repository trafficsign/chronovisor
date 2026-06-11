# Configuration

The preferred runtime config is `~/.wiki/config.toml`. If it does not exist,
LLM Wiki falls back to the legacy `~/.wiki/recall.toml` shape.

## Unified Shape

```toml
[hooks.user_prompt]
recall = true

[hooks.stop]
save = true
audit = true

[embedding]
# Tuned search profile. If omitted, runtime falls back to nomic-embed-text.
model = "bge-m3"
document_prefix = ""
query_prefix = ""

[search.reranker]
# Optional. Used only by MCP wiki.search when enabled; synchronous recall hooks
# keep using the faster BM25/dense fusion path.
enabled = false
backend = "transformers"
model = "BAAI/bge-reranker-v2-m3"
top_n = 20
max_length = 1024
batch_size = 8
device = ""
weight = 0.25

[recall.thresholds]
search = 0.35
read = 0.65

[recall.gate]
model = "qwen3.5:4b"
think = false
timeout_ms = 2000
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
model = "qwen3.5:4b"
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
model = "qwen3.6:35b-a3b-q8_0"
think = false
timeout_ms = 120000
top_k = 5
min_confidence = 0.70

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
- `LLM_WIKI_RECALL_AUDIT_ENABLED=1`: enable recall auditor hook.

The compatibility wrappers preserve the old environment behavior. Direct
`llm-wiki-hook --event Stop` deployments can rely on `config.toml`.

## Optional Reranker

Install the optional local reranker dependencies with `uv sync --extra reranker`
before enabling `[search.reranker]`. The default backend uses Hugging Face
Transformers with `BAAI/bge-reranker-v2-m3` and reranks only the MCP
`wiki.search` top candidates. It is not called by the synchronous recall hook.

## Recall Defaults

The completed recall path defaults to `gate_mode = "evidence"`,
`context_style = "cards"`, `semantic = true`, rewrite enabled, and calibration
enabled. BM25 still runs without Ollama; semantic and rewrite fail open when the
local model is unavailable.
