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
- After explicit push approval, the GitHub runtime was pinned to `cd526bbe6c37e2e88e694c2521d7a0c97940fcb7`. All 11 managed services have resumed. Production Ingest completed a validated and sealed authority decision using the new model identity.

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

After all services resumed, Activity Monitor showed used 119.18 GB and swap 10.75 GB. These changing-workload snapshots do not establish a 10 GB saving. See `production-live.json` for the final captured state.

## Completed deployment

Scoped runtime/config/test commit: `74818b2`. Isolated `tests/test_llm_config.py`: 55 passed, including new identity acceptance and old model/revision rejection. Ruff and diff checks passed. Local generation and both production-router probes above also passed with this source.

With user approval, only `74818b2` was cherry-picked onto the previous remote main as `cd526bbe6c37e2e88e694c2521d7a0c97940fcb7` and pushed. The exact deployed source passed all 55 configuration tests. Earlier unpublished experimental history was not pushed. The temporary deployment worktree and branch were removed after publication.

Production config was then pinned to that immutable GitHub commit and services restarted. Dashboard, LAN Dashboard, Ingest, Reranker and Semantic installed archives all report the deployed commit in `direct_url.json`. Dashboard reports no runtime SHA drift and current authority ready; its overall health still contains pre-existing alerts, so this is not a claim that every unrelated health indicator is green.

Independent read-only checks found Dashboard, LAN Dashboard, Reranker, Semantic and SearXNG running. Reranker and Semantic socket health both returned `status=ok, ready=true`. SearXNG uses its separate local editable source at `ef8f6470e0473a1548f175217aaa7b9346ce6973`, not a Chronovisor archive.

Ingest PID 72932 connected directly to DwarfStar port 18136. At `2026-09-06T13:00:19.241847Z`, the production authority audit recorded `local_repair`, `agreed`, `validated`, the new model/revision, zero repair turns and no failure; the subsequent record sealed that decision. Ingest liveness was ready with authority available and no liveness alert. This verifies actual production processing, not just a synthetic request or a live PID.

The permanent runtime, verified model weights, pre-cutover configuration backups and reusable evidence remain. Old Flash Next weights remain inactive on disk; no fallback is configured. Auxiliary oMLX services remain available for Ornith and embeddings.
