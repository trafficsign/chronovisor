# DwarfStar production cutover

2026-09-06, M4 Max / 128 GiB. User explicitly approved switching the main model.

## Installed local state

- launchd: `com.trafficsign.dwarfstar`, loopback `127.0.0.1:18136`, RunAtLoad + KeepAlive, private logs (umask 077).
- runtime: `/Users/trafficsign/.local/share/dwarfstar/runtime`, commit `236cb2a549d05ea1941a19bd4f154e0abc858661`.
- weights: `/Users/trafficsign/.local/share/dwarfstar/models`, existing verified main + PLE moved from the experiment directory, no additional download.
- model artifact revision: `59a55fb819c82be7b162948282b50bd1a1e290b7`.
- flags: Metal, context 65536, power 100, MTP + exact sampling. No prompt trace, disk KV cache, or batched-session flag.
- 37 Flash Next roles now use provider ID `dwarfstar`, model `qwen3.8-flash-next-chat`. `kind = "omlx"` deliberately reuses the existing local OpenAI chat transport; runtime telemetry still labels its adapter/protocol `omlx` / `omlx-native`. This is not an assertion that the inference engine is oMLX.
- The 8 auxiliary roles are unchanged. oMLX remains at 18125 for embedding and Ornith. Old Flash Next is unpinned, not default, and hidden; its files remain on disk, not configured as a fallback.
- Local config and oMLX model_settings backups: `/Users/trafficsign/.local/share/dwarfstar/backups/pre-cutover-20260906/`.
- Existing GitHub runtime source remains pinned at `136ab8b4e5ebd91cef2137edb67e63a0a8f49eb3` pending explicit push approval. Generation services are paused because that old runtime intentionally rejects the new authority identity. Dashboard, LAN Dashboard, Semantic, Reranker and SearXNG have resumed. Do not call this completed production integration yet.

## Executed checks

`scripts/check_ds4_cutover.py` uses the updated local source with production config and the running services, not the stale installed archive. It performs no wiki mutation and removes its temporary decision audit directory on exit.

| Check | Result | Wall seconds |
|---|---|---:|
| ingest.generation: exact DS4_READY | pass, returned model matches | 1.69 |
| recall.gate: exact YES | pass, Ornith retained | 1.46 |
| classification.embedding | pass, 1024 dimensions | — |
| DecisionRouter orphan_link | schema + expected effect pass, zero repairs | 8.80 |
| DecisionRouter ingest_reconciliation | schema + expected effect pass, zero repairs | 7.61 |

Unlike the earlier single-post benchmark, these decision checks use production prompt construction, schema validation and plain-choice materialization. This resolves those two replay-path concerns, not general model quality equivalence.

`plutil -lint` passed. `launchctl kickstart -k` followed by `/v1/models` readiness passed in 20.08 seconds. oMLX `/health` healthy, default Ornith, old Qwen not loaded (loaded model memory 5,290,069,221 bytes before embedding warmup).

## Memory observation, not a controlled benchmark

Activity Monitor bottom panel before stopping services: used 116.37 GB, swap 11.94 GB. After DS4 plus auxiliary services startup: used 109.91 GB, swap 11.25 GB. Background generation was still paused in the latter observation, so the 6.46 GB display difference is not a same-workload savings estimate. Process-row DS4 8.27 GB is not its total physical RAM requirement.

## Remaining deployment gate

Scoped runtime/config/test commit: `74818b2`. Isolated `tests/test_llm_config.py`: 55 passed, including new identity acceptance and old model/revision rejection. Ruff and diff checks passed. Local generation and both production-router probes above also passed with this source.

Push only the scoped model identity/config/test change, not the earlier unpushed experimental history. Pin the GitHub runtime to that deployed commit, restart the affected services, verify runtime archive identity and actual generation/authority progress. User push approval is required by the standing global rule. Do not bypass the model identity checks or silently run the stale archive against the new model.
