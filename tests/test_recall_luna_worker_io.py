from __future__ import annotations

import pytest

from chronovisor.recall.recall_distillation_luna import native_output_sha256
from scripts.recall_luna_worker_io import bind_decisions


def test_native_decisions_are_bound_to_exact_frozen_ids() -> None:
    request = {
        "teacher_profile": "native-luna-single-v1",
        "records": [{"task_id": "a" * 64, "input_sha256": "b" * 64}],
    }
    decisions = [
        {
            "task_id": "a" * 64,
            "verdict": "uncertain",
            "confidence": 0.4,
            "reason_code": "insufficient_context",
        }
    ]
    row = bind_decisions(request, decisions, assignment_id="worker-0", run_id="run-0")[
        0
    ]
    assert row["verdict"] == "uncertain"
    assert row["input_sha256"] == "b" * 64
    assert row["rationale"] is None
    assert row["native_receipt"]["output_sha256"] == native_output_sha256(row)
    with pytest.raises(ValueError, match="cardinality"):
        bind_decisions(request, [], assignment_id="worker-0", run_id="run-0")
    with pytest.raises(ValueError, match="identities"):
        bind_decisions(
            request,
            [{**decisions[0], "task_id": "c" * 64}],
            assignment_id="worker-0",
            run_id="run-0",
        )
