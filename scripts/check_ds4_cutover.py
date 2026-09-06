#!/usr/bin/env python3
"""Read-only local inference checks using production routing and validation.

No wiki mutations. Decision audits are isolated and removed on exit.
Run while background generation jobs are paused; this sends real requests.
"""

import json
import tempfile
import time
from pathlib import Path

from chronovisor.core.llm_config import build_llm_runtime, load_llm_config
from chronovisor.core.llm_runtime import (
    EmbeddingPurpose,
    EmbeddingRequest,
    MessageGenerationRequest,
    SourceDataClass,
    SourceDataClassification,
    SourceSensitivity,
)
from chronovisor.decision.decision_router import DecisionRouter
from chronovisor.decision.local_model_eval import replay_semantic_effect
from chronovisor.decision.local_structured import validate_json

ROOT = Path(__file__).resolve().parents[1]
CORPUS = ROOT / "_handoff/evidence/2026-09-06-jang4s-benchmark/adoption-corpus.jsonl"


def main():
    runtime = build_llm_runtime(load_llm_config())
    source = SourceDataClassification(SourceDataClass.PAGE, SourceSensitivity.NORMAL)
    for role, prompt, expected in (
        ("ingest.generation", "Reply exactly: DS4_READY", "DS4_READY"),
        ("recall.gate", "Reply exactly: YES", "YES"),
    ):
        started = time.monotonic()
        result = runtime.generate(
            role,
            MessageGenerationRequest(
                messages=({"role": "user", "content": prompt},),
                format=None,
                source=source,
                num_ctx=65536,
                keep_alive="0",
                max_output_tokens=32,
                max_output_chars=256,
                timeout_ms=60000,
                temperature=0,
                seed=42,
                think=False,
            ),
        )
        print(
            json.dumps(
                {
                    "role": role,
                    "model": result.model,
                    "seconds": time.monotonic() - started,
                    "ok": result.content.strip() == expected,
                }
            ),
            flush=True,
        )
        assert result.content.strip() == expected, role
    result = runtime.embed(
        "classification.embedding",
        EmbeddingRequest(
            texts=("local memory retrieval",),
            source=source,
            timeout_ms=60000,
            purpose=EmbeddingPurpose.DOCUMENT,
        ),
    )
    assert len(result.vectors) == 1 and len(result.vectors[0]) > 0
    print(
        json.dumps({"embedding_dimensions": len(result.vectors[0]), "ok": True}),
        flush=True,
    )
    rows = [json.loads(line) for line in CORPUS.read_text().splitlines()]
    with tempfile.TemporaryDirectory(prefix="ds4-cutover-check-") as temporary:
        for lane in ("orphan_link", "ingest_reconciliation"):
            row = next(r for r in rows if r.get("decision_lane") == lane)
            router = DecisionRouter(
                audit_root=Path(temporary),
                record_replay=False,
                artifact_replay=False,
                decision_lane=lane,
            )
            assert router.config_error is None, router.config_error
            started = time.monotonic()
            result = router.decide(
                row["prompt"], row["schema"], system=row.get("system")
            )
            value = result.value
            valid = result.ok and not validate_json(value, row["schema"])
            effect_matches = valid and replay_semantic_effect(
                value, row["schema"], prompt=row["prompt"], decision_lane=lane
            ) == replay_semantic_effect(
                row["expected"], row["schema"], prompt=row["prompt"], decision_lane=lane
            )
            print(
                json.dumps(
                    {
                        "lane": lane,
                        "valid": valid,
                        "effect_matches": effect_matches,
                        "seconds": time.monotonic() - started,
                        "failure": result.quarantine_reason,
                        "repairs": sum(v.result.repair_turns for v in result.votes),
                    }
                ),
                flush=True,
            )
            assert valid and effect_matches, lane


if __name__ == "__main__":
    main()
