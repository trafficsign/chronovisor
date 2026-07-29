from __future__ import annotations

from types import SimpleNamespace

from chronovisor.classification import classification_resource_burn


def test_resource_burn_requires_every_stage_and_persists_gate(
    tmp_path,
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        classification_resource_burn,
        "load_decision_router_config",
        lambda: SimpleNamespace(
            primary_model="ornith",
            challenger_model="gpt-oss",
            tie_break_model="gemma",
        ),
    )
    monkeypatch.setattr(
        classification_resource_burn,
        "load_embedding_config",
        lambda: SimpleNamespace(model="bge-m3"),
    )
    monkeypatch.setattr(
        classification_resource_burn.ollama,
        "generate",
        lambda *args, **kwargs: "OK",
    )
    monkeypatch.setattr(
        classification_resource_burn,
        "_recall",
        lambda _index: (
            {"latency_ms": 100, "status": "ok", "scheduler": {}},
            "same",
        ),
    )

    def overlap(**kwargs):
        return {
            "stage": kwargs["stage"],
            "kind": kwargs["kind"],
            "model": kwargs["model"],
            "recall": {"latency_ms": 105, "status": "ok", "scheduler": {}},
            "recall_fingerprint": "same",
            "cancel_ack_ms": 150,
            "cancel_to_resource_ready_ms": 500,
            "resource_ready": True,
            "protected_resident": True,
            "worker_pid_alive": False,
            "research_overlap": True,
            "research_preempted": True,
            "foreground_wait_ms": 10,
            "lease_residual": False,
            "attempts_consumed": 0,
            "requeued": True,
        }

    monkeypatch.setattr(classification_resource_burn, "_overlap_one", overlap)
    receipt = classification_resource_burn.run_resource_burn(
        tmp_path,
        samples_per_stage=50,
    )

    assert receipt["status"] == "passed"
    assert set(receipt["stages"]) == {
        "proposal",
        "audit",
        "tie_break",
        "dense_embedding",
    }
    assert all(value["sample_count"] == 50 for value in receipt["stages"].values())
    assert (
        tmp_path
        / "classification"
        / "library-evidence"
        / "evaluation"
        / "resource-gate.json"
    ).is_file()
