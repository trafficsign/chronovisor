"""P3: production-env replay of the decision-model reranker service (worktree code, StartLux 4B 8-bit).

Runs RerankerServiceState in-process against the live index and the real accelerator lease, so the GPU is shared with the
running semantic service exactly as in production. The production status file and model telemetry are never written.
  recall_filter: the latest injected recall turns, their exact published items (title + newline + evidence), the
                 previous distinct logged turn of the session as context (the frozen cohort's rule).
  rerank:        search_cands.jsonl queries' top-10 page ids through the service's own passage resolution.
A probe thread issues semantic searches throughout (and alone, before), to see what the decision model costs it.
usage: run with the reranker-service uvx environment (see scripts/chronovisor-reranker-service), --from the worktree."""

import contextlib
import json
import os
import statistics
import sys
import tempfile
import threading
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
RUN3 = HERE.parent
MODEL = json.loads((RUN3 / "model_downloads.json").read_text())[
    "startlux-models/StartLux-Decision-v1.1-4B-Instant-MLX-8bit"
]["path"]
N_FILTER = int(sys.argv[1]) if len(sys.argv) > 1 else 60


def filter_workloads(n):
    previous, out = {}, []
    for line in open(Path.home() / ".chronovisor/recall/recall-log.jsonl"):
        r = json.loads(line)
        sid, prompt = r.get("session_id"), r.get("prompt_preview", "")
        text = (r.get("context_receipt") or {}).get("rendered_context", "")
        if r.get("context_items") and "payload_json=\n" in text and prompt.strip():
            payload = json.loads(
                text.split("payload_json=\n", 1)[1].split("\n[/RECALL_CONTEXT]", 1)[0]
            )
            texts = [
                f"{x.get('title', '')}\n{x.get('evidence', '')}"
                for x in payload.get("items", [])
            ]
            if texts:
                before = previous.get(sid, "")
                out.append(
                    {
                        "query": prompt,
                        "context": before if before != prompt else "",
                        "texts": texts,
                    }
                )
        if sid and prompt.strip():
            previous[sid] = prompt
    return out[-n:]


def pct(values, q):
    v = sorted(values)
    return round(v[min(len(v) - 1, int(round((len(v) - 1) * q)))])


class Probe(threading.Thread):
    """Semantic searches back to back (0.3 s apart) until stopped; latency and failures."""

    def __init__(self):
        super().__init__(daemon=True)
        from chronovisor.core import semantic_client
        from chronovisor.core.runtime_config import load_search_embedding_config

        self.client, self.config = semantic_client, load_search_embedding_config()
        self.ms, self.failures, self.stop = [], 0, threading.Event()

    def run(self):
        while not self.stop.is_set():
            t0 = time.perf_counter()
            try:
                self.client.search(
                    "reranker latency probe",
                    10,
                    include_reference=False,
                    config=self.config,
                    timeout_ms=800,
                )
                self.ms.append((time.perf_counter() - t0) * 1e3)
            except Exception:
                self.failures += 1
            self.stop.wait(0.3)

    def report(self):
        return {
            "n": len(self.ms),
            "failures": self.failures,
            "p50_ms": pct(self.ms, 0.5) if self.ms else None,
            "p95_ms": pct(self.ms, 0.95) if self.ms else None,
        }


def main():
    from chronovisor.core import llm_config
    from chronovisor.core.reranker import RERANK_RUNTIME_ROLE
    from chronovisor.core.runtime_config import RerankerConfig, RerankerServiceConfig
    from chronovisor.search import reranker_service

    reranker_service.SERVICE_STATUS_FILE = (
        Path(tempfile.mkdtemp()) / "status.json"
    )  # never the production status
    reranker_service.model_activity = lambda **_k: (
        contextlib.nullcontext()
    )  # keep production telemetry clean
    runtime = llm_config.build_llm_runtime(
        llm_config.parse_llm_config(
            {
                "llm": {
                    "providers": {
                        "local_decision": {"kind": "local-decision-mlx", "path": MODEL}
                    },
                    "roles": {
                        RERANK_RUNTIME_ROLE: {
                            "capability": "rerank",
                            "provider": "local_decision",
                            "model": "startlux-decision-v1.1-4b-instant-mlx-8bit",
                        }
                    },
                }
            }
        )
    )
    state = reranker_service.RerankerServiceState(
        RerankerConfig(
            enabled=True,
            top_n=10,
            service=RerankerServiceConfig(enabled=True, timeout_ms=10_000),
        ),
        llm_runtime=runtime,
    )
    print(json.dumps({"warm": state._warm.get("status")}), flush=True)

    alone = Probe()
    alone.start()
    time.sleep(15)
    alone.stop.set()
    alone.join()
    print(json.dumps({"semantic_alone": alone.report()}), flush=True)
    # per-item latencies (filter shape) and the timing of every request, for the tail
    timings = []

    work = filter_workloads(N_FILTER)
    state.handle({"method": "recall_filter", **work[0]})  # warm the filter shape
    # PROBE=0: no concurrent semantic load (production calls semantic once per turn, before the filter)
    probe = Probe()
    if os.environ.get("PROBE", "1") != "0":
        probe.start()
    ms, kept, total, sizes = [], 0, 0, []
    threshold = -4.4430
    for w in work:
        t0 = time.perf_counter()
        scores = state.handle({"method": "recall_filter", **w})["scores"]
        ms.append((time.perf_counter() - t0) * 1e3)
        timings.append((len(w["texts"]), round(ms[-1])))
        kept += sum(s >= threshold for s in scores)
        total += len(scores)
        sizes.append(len(scores))
    print(
        json.dumps(
            {
                "recall_filter": {
                    "turns": len(work),
                    "items_per_turn": dict(
                        sorted({k: sizes.count(k) for k in set(sizes)}.items())
                    ),
                    "p50_ms": pct(ms, 0.5),
                    "p95_ms": pct(ms, 0.95),
                    "max_ms": round(max(ms)),
                    "kept_fraction": round(kept / total, 3),
                    "with_context": sum(bool(w["context"]) for w in work),
                    "slowest": sorted(timings, key=lambda x: -x[1])[:5],
                    "ms_per_item_p50": round(statistics.median(t / n for n, t in timings)),
                }
            }
        ),
        flush=True,
    )

    cands = [json.loads(l) for l in open(HERE / "private/search_cands.jsonl")]
    cands = [c for c in cands if c["head_n"] == 10][:20]
    ms = []
    for c in cands:
        t0 = time.perf_counter()
        state.handle(
            {"method": "rerank", "query": c["query"], "page_ids": c["base"][:10]}
        )
        ms.append((time.perf_counter() - t0) * 1e3)
    probe.stop.set()
    if probe.is_alive():
        probe.join()
    print(
        json.dumps(
            {
                "rerank10": {
                    "queries": len(ms),
                    "p50_ms": pct(ms, 0.5),
                    "p95_ms": pct(ms, 0.95),
                    "max_ms": round(max(ms)),
                },
                "semantic_during": probe.report(),
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
