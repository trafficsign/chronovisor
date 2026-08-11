"""Host composition for Campaign Y recall and acceptance."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any

from chronovisor.core.canonical_json import canonical_json_sha256_strict
from chronovisor.recall import recall_runtime
from chronovisor.research import evidence_eval


def bind_recall_provider() -> None:
    evidence_eval.bind_recall_provider(recall_runtime.bind_evidence_provider)


def run_evidence_acceptance(root: Path) -> dict[str, Any]:
    """Run acceptance with the exact production teacher and renderer arms."""

    bind_recall_provider()
    policy = replace(
        recall_runtime.load_policy(),
        semantic=False,
        judge_mode="off",
        rewrite_enabled=False,
        log_decisions=False,
        processor_enabled=False,
        processor_shadow_enabled=False,
        processor_auto_enable=False,
        processor_judge_enabled=False,
    )

    def page_teacher(query: str) -> Any:
        return recall_runtime.run_recall(
            recall_runtime.RecallRequest(
                host="codex",
                event="UserPromptSubmit",
                prompt=query,
                session_id="",
            ),
            policy,
        )

    def candidate_renderer(baseline: Any, run: Any) -> bytes:
        result = replace(
            baseline,
            decision="read",
            queries=[run.packet.query],
            context_items=[],
            context="",
            evidence_packet=run.packet,
            evidence_features={
                "evidence_reconstruction": {
                    "trace": dict(run.trace),
                    "trace_sha256": canonical_json_sha256_strict(run.trace),
                }
            },
        )
        return recall_runtime.format_recall_context(result, policy).encode("utf-8")

    return evidence_eval._run_evidence_acceptance(
        root=root,
        page_teacher=page_teacher,
        candidate_renderer=candidate_renderer,
    )
