# Chronovisor

Chronovisor is a local-first memory runtime for LLM agents. It stores durable
conversation knowledge on the user's filesystem, exposes it through MCP, and
can connect host hooks for automatic recall and lossless transcript capture.

## Release status

Chronovisor is pre-release software. The only production-supported deployment
is local-only. Cloud-only and hybrid routing are experimental and are not
feature-complete.

Two public-release campaigns remain open:

- Campaign W: finish provider-neutral runtime routing, OS keyring integration,
  remote adapters, and migration of every production model consumer.
- Campaign X: finish the one-shot OKF v0.2 canonical data migration and remove
  the legacy layout.

Do not treat the current `main` branch as an OSS v1 release.

## Clean local install

Prerequisites are Python 3.11 or newer, [uv](https://docs.astral.sh/uv/), and a
running [Ollama](https://ollama.com/) service.

```sh
git clone https://github.com/trafficsign/chronovisor.git
cd chronovisor
uv sync --frozen
install -d -m 700 ~/.chronovisor
test ! -e ~/.chronovisor/config.toml
install -m 600 config.toml.example ~/.chronovisor/config.toml
```

The example is the smallest supported local-only configuration. It disables
optional Web egress. Existing `~/.chronovisor/config.toml` files should be
merged manually rather than overwritten.

Install the local models named by the example:

```sh
ollama pull maxwell1500/ornith-35b:Q5_K_M
ollama pull gpt-oss:20b
ollama pull gemma4:26b
ollama pull bge-m3
ollama pull ornith:9b-q4_K_M
```

Start the MCP server from the checkout:

```sh
uv run chronovisor-mcp
```

The MCP server and host hook create the initial data directories on first use.
In another terminal, check the installation or start the loopback-only
Dashboard:

```sh
uv run chronovisor doctor
uv run chronovisor-dashboard --host 127.0.0.1 --port 8765
```

Host hook installation is optional. Review [the hook guide](docs/hooks.md)
before allowing the installer to update Codex or Claude Code settings.

## Architecture and runtime roles

```text
Host transcript -> deterministic capture -> raw/
                                      |-> ingest -> pages/ + system/
Host prompt     -> search + recall -----------------------> recalled context
                                      |-> local model runtime
Dashboard       <- redacted operational state in runtime/
```

- `chronovisor-mcp` serves the public memory tools.
- `chronovisor-hook` dispatches host events. Capture is deterministic and does
  not call an LLM.
- Ingest turns immutable raw captures into reviewed pages.
- Search combines lexical retrieval with optional local embedding and rerank
  services. Recall decides what context is safe and useful to inject.
- Structured mutation decisions use a local primary/challenger quorum and a
  tie-break model only when needed; invalid or unresolved decisions fail
  closed.
- The Dashboard is an operations view. It accepts only loopback clients and
  validates browser Host and Origin boundaries; it is not a remote admin UI.

The common `LLMRuntime` foundation normalizes generation, embedding, reranking,
source classification, and egress decisions. Campaign W has not yet moved all
production callers through it, and no supported remote-provider configuration
is shipped yet.

See [architecture](docs/architecture.md),
[configuration](docs/config.md), [operations](docs/operations.md), and the
[threat model](docs/threat-model.md) for details.

## Data and security boundary

The default data root is `~/.chronovisor`; `CHRONOVISOR_ROOT` selects a different
root for an isolated process. Important locations are:

- `config.toml`: runtime policy and model selection; keep it mode `0600`.
- `raw/`: immutable transcript captures.
- `pages/`: ordinary long-term knowledge.
- `system/`: privileged profile and state documents.
- `recall/`: recall decisions, feedback, and field state.
- `runtime/`: locks, queues, receipts, health, and redacted traces.
- `research/`, `review/`, and `claims/`: evidence, review artifacts, and derived
  claim indexes when those features are used.

Campaign X is still migrating legacy root metadata into the final canonical
layout. Back up the complete data root before testing migration work.

Chronovisor's security boundary is the local OS user, the data root, loopback
services, configured model endpoints, and explicitly enabled network egress.
It does not defend secrets or memory content from arbitrary code running as the
same user, root, or a process-memory compromise. Keep the Dashboard on
loopback, do not store plaintext credentials in the repository or config, and
leave Web/cloud egress disabled unless its data policy is understood.

Report vulnerabilities privately as described in [SECURITY.md](SECURITY.md).

## Contributing and tests

```sh
uv sync --frozen
uv run ruff check --no-cache src scripts tests
uv run mypy
uv run lint-imports --no-cache
CHRONOVISOR_ROOT="$(mktemp -d)" uv run pytest -q
```

Keep tests isolated from a live `~/.chronovisor` tree. Small changes may start
with focused tests, but the full checks above are the contribution baseline.
