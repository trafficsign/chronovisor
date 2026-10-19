"""CF diagnosis must preserve failure handling without exposing model content."""

import hashlib
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from chronovisor.recall import recall_distillation as distill


@pytest.mark.parametrize("attempt", [1, 3])
@pytest.mark.parametrize("error_type,failure_class", [
    (distill.DistillationError, "counterfactual_invalid"),
    (distill._CounterfactualDataRejected, "counterfactual_data_reject"),
])
def test_counterfactual_failure_logs_only_cause_digest(
    tmp_path: Path, caplog: pytest.LogCaptureFixture, attempt: int,
    error_type: type[Exception], failure_class: str,
) -> None:
    private_message = "synthetic private answer and query must not be logged"
    commits: list[Any] = []
    queue = SimpleNamespace(commit=lambda *args: commits.append(args))
    claim = SimpleNamespace(attempt=attempt)

    class InvalidCounterfactual:
        def compare(self, _payload: object) -> dict:
            raise error_type(private_message)

    result = distill._compare_and_commit_counterfactual(
        workset=queue, claim=claim, counterfactual=InvalidCounterfactual(),
        payload={}, label_path=tmp_path / "labels", root=tmp_path, raw_dir=tmp_path,
        snapshots={}, exposure={}, exact_features={}, rally_id="synthetic", candidate={},
        mode="full", config=distill.DistillationConfig(),
    )
    assert result.deferred and result.failure_class == failure_class
    assert commits[0][1] == [{
        "status": "quarantined" if attempt == 3 else "retry",
        "error_class": "counterfactual_invalid",
        "retry_after_seconds": 0 if attempt == 3 else 60,
    }]
    assert len(caplog.records) == 1
    assert caplog.records[0].getMessage() == (
        f"counterfactual_failure class={failure_class} cause_type={error_type.__name__} "
        f"cause_sha256={hashlib.sha256(private_message.encode()).hexdigest()}"
    )
    assert private_message not in caplog.text
    assert caplog.records[0].exc_info is None
    assert not (tmp_path / "labels").exists()
