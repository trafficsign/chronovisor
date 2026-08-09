from __future__ import annotations

import hashlib
import json
import threading
from pathlib import Path
from types import SimpleNamespace

import pytest


@pytest.fixture()
def semantic_defer_wiki(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[Path, Path]:
    from chronovisor.core import runtime_config, store
    from chronovisor.decision import decision_router
    from chronovisor.ops import runtime_status

    chronovisor_root = tmp_path / "wiki"
    raw_dir = chronovisor_root / "raw"
    runtime_dir = chronovisor_root / "runtime"
    raw_dir.mkdir(parents=True)
    runtime_dir.mkdir(parents=True)
    artifact = tmp_path / "adoption.json"
    artifact.write_bytes(b'{"epoch":1}\n')

    monkeypatch.setattr(store, "CHRONOVISOR_ROOT", chronovisor_root)
    monkeypatch.setattr(store, "RAW_DIR", raw_dir)
    monkeypatch.setattr(runtime_status, "RUNTIME_DIR", runtime_dir)
    monkeypatch.setattr(runtime_status, "STATUS_FILE", runtime_dir / "status.json")
    monkeypatch.setattr(runtime_status, "EVENTS_FILE", runtime_dir / "events.jsonl")
    monkeypatch.setattr(runtime_status, "METRICS_FILE", runtime_dir / "metrics.jsonl")
    monkeypatch.setattr(
        runtime_config,
        "load_decision_router_config",
        lambda: SimpleNamespace(adoption_artifact=str(artifact)),
    )

    def resolve_test_artifact(config: SimpleNamespace) -> SimpleNamespace:
        try:
            artifact_sha256 = hashlib.sha256(
                Path(config.adoption_artifact).read_bytes()
            ).hexdigest()
        except OSError:
            return SimpleNamespace(
                source="bootstrap_current_policy",
                error="adoption_artifact_invalid:unreadable",
                artifact_sha256=None,
            )
        return SimpleNamespace(
            source="adopted_artifact",
            error=None,
            artifact_sha256=artifact_sha256,
        )

    monkeypatch.setattr(
        decision_router,
        "resolve_router_policy",
        resolve_test_artifact,
    )
    return chronovisor_root, artifact


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _no_quorum_error(authority_sha256: str) -> str:
    return (
        "local consensus semantic no quorum: three valid semantic votes differed "
        f"[authority_sha256={authority_sha256}]"
    )


def test_classifies_only_explicit_authority_bound_semantic_no_quorum() -> None:
    from chronovisor.raw import failure_supervisor

    authority_sha256 = "a" * 64
    semantic = failure_supervisor.classify_failure(_no_quorum_error(authority_sha256))
    missing = failure_supervisor.classify_failure(
        "local consensus semantic no quorum: marker missing"
    )
    malformed = failure_supervisor.classify_failure(
        "local consensus semantic no quorum [authority_sha256=" + "A" * 64 + "]"
    )
    legacy = failure_supervisor.classify_failure(
        "local consensus ingest review did not converge after 2 local review calls"
    )

    assert semantic.failure_class == "ingest.semantic_no_quorum"
    assert semantic.authority_artifact_sha256 == authority_sha256
    assert semantic.fingerprint == f"ingest.semantic_no_quorum:{authority_sha256}"
    assert missing.failure_class == (
        "ingest.runtime_local_consensus_authority_unavailable"
    )
    assert malformed.failure_class == (
        "ingest.runtime_local_consensus_authority_unavailable"
    )
    assert legacy.failure_class == "ingest.local_consensus_nonconvergent"


def test_terminal_semantic_defer_preserves_raw_and_never_starts_self_heal(
    semantic_defer_wiki: tuple[Path, Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from chronovisor.ops import self_heal
    from chronovisor.raw import failure_supervisor

    chronovisor_root, artifact = semantic_defer_wiki
    raw_path = chronovisor_root / "raw" / "semantic.md"
    original = "---\ntitle: Semantic\n---\nbyte exact 日本語\n".encode()
    raw_path.write_bytes(original)
    starts: list[Path] = []
    monkeypatch.setattr(failure_supervisor, "_launch_self_heal", starts.append)

    result = failure_supervisor.record_raw_failure(
        raw_path=raw_path,
        error=_no_quorum_error(_sha256(artifact)),
        job_id="job-semantic",
        raw_text=original.decode(),
    )

    assert result.quarantined is False
    assert result.terminal_deferred is True
    result_to_dict = failure_supervisor.result_to_dict(result)
    assert result_to_dict["terminal_deferred"] is True
    assert starts == []
    assert raw_path.read_bytes() == original
    assert not (chronovisor_root / "runtime" / "failures" / "quarantined-raw").exists()

    packet_path = Path(str(result.packet_path))
    packet = json.loads(packet_path.read_text(encoding="utf-8"))
    assert packet["status"] == "local_quarantined"
    assert packet["frontier_status"] == "not_requested"
    assert packet["terminal_deferred"] is True
    assert packet["self_heal_queued"] is False
    assert packet["defer_reason"] == "semantic_no_quorum"
    assert packet["authority_artifact_sha256"] == _sha256(artifact)
    assert packet["related_raw_files"] == [raw_path.name]
    assert packet["source_raws"] == [
        {
            "filename": raw_path.name,
            "bytes": len(original),
            "sha256": hashlib.sha256(original).hexdigest(),
        }
    ]
    state = json.loads((chronovisor_root / "runtime" / "failures" / "state.json").read_text())
    entry = state["failures"][raw_path.name]
    assert entry["terminal_deferred"] is True
    assert entry["self_heal_queued"] is False
    assert entry["packet_path"] == str(packet_path)
    assert failure_supervisor.operational_deferred_raw_files([raw_path]) == {
        raw_path.name: "semantic_no_quorum"
    }
    assert self_heal.pending_packets() == []


def test_related_raws_share_one_idempotent_terminal_packet_and_reset_together(
    semantic_defer_wiki: tuple[Path, Path],
) -> None:
    from chronovisor.raw import failure_supervisor

    chronovisor_root, artifact = semantic_defer_wiki
    first = chronovisor_root / "raw" / "fragment-1.md"
    second = chronovisor_root / "raw" / "fragment-2.md"
    first_bytes = b"fragment one\n"
    second_bytes = b"fragment two\n"
    first.write_bytes(first_bytes)
    second.write_bytes(second_bytes)
    error = _no_quorum_error(_sha256(artifact))

    initial = failure_supervisor.record_semantic_no_quorum_defer(
        raw_path=first,
        error=error,
        related_raw_paths=(first, second),
    )
    repeated = failure_supervisor.record_semantic_no_quorum_defer(
        raw_path=first,
        error=error,
        related_raw_paths=(second, first),
    )

    assert repeated.packet_path == initial.packet_path
    packets = list((chronovisor_root / "runtime" / "failures" / "packets").glob("*.json"))
    assert packets == [Path(str(initial.packet_path))]
    assert first.read_bytes() == first_bytes
    assert second.read_bytes() == second_bytes
    state_path = chronovisor_root / "runtime" / "failures" / "state.json"
    failures = json.loads(state_path.read_text())["failures"]
    assert set(failures) == {first.name, second.name}
    assert {entry["packet_path"] for entry in failures.values()} == {
        str(initial.packet_path)
    }
    assert all(
        entry["related_raw_files"] == [first.name, second.name]
        for entry in failures.values()
    )

    failure_supervisor.reset_raw_failure(first.name)

    assert json.loads(state_path.read_text())["failures"] == {}
    released_packet = json.loads(Path(str(initial.packet_path)).read_text())
    assert released_packet["status"] == "semantic_defer_released"
    assert released_packet["terminal_deferred"] is False
    assert released_packet["released_at"]
    assert failure_supervisor.operational_deferred_raw_files([first, second]) == {}


def test_defer_reconciliation_shares_one_raw_snapshot_across_packets(
    semantic_defer_wiki: tuple[Path, Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from chronovisor.raw import failure_supervisor, raw_store

    chronovisor_root, artifact = semantic_defer_wiki
    error = _no_quorum_error(_sha256(artifact))
    first = chronovisor_root / "raw" / "first.md"
    second = chronovisor_root / "raw" / "second.md"
    first.write_text("first source\n")
    second.write_text("second source\n")
    failure_supervisor.record_semantic_no_quorum_defer(
        raw_path=first,
        error=error,
    )
    failure_supervisor.record_semantic_no_quorum_defer(
        raw_path=second,
        error=error,
    )

    real_store = raw_store.RawStore
    constructions = 0

    class CountingRawStore(real_store):
        def __init__(self, *args, **kwargs):
            nonlocal constructions
            constructions += 1
            super().__init__(*args, **kwargs)

    monkeypatch.setattr(raw_store, "RawStore", CountingRawStore)

    assert failure_supervisor.operational_deferred_raw_files() == {
        first.name: "semantic_no_quorum",
        second.name: "semantic_no_quorum",
    }
    # One snapshot lists available raws and one verifies every packet source.
    assert constructions == 2


def test_guarded_semantic_publish_keeps_newer_operational_hold_atomic(
    semantic_defer_wiki: tuple[Path, Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from chronovisor.raw import failure_supervisor

    chronovisor_root, artifact = semantic_defer_wiki
    raw_path = chronovisor_root / "raw" / "operational-wins.md"
    raw_path.write_text("immutable source\n", encoding="utf-8")
    observed_paths: list[tuple[Path, ...]] = []

    def active_operational(paths) -> dict[str, str]:
        observed_paths.append(tuple(paths))
        return {raw_path.name: "pending_local_repair"}

    monkeypatch.setattr(
        failure_supervisor,
        "_operational_deferred_raw_files_unlocked",
        active_operational,
    )
    monkeypatch.setattr(
        failure_supervisor,
        "_record_semantic_no_quorum_defer_unlocked",
        lambda **_kwargs: pytest.fail(
            "semantic packet must not supersede an active operational hold"
        ),
    )

    result = failure_supervisor.record_semantic_no_quorum_defer_unless_operational_hold(
        raw_path=raw_path,
        error=_no_quorum_error(_sha256(artifact)),
        raw_text=raw_path.read_text(),
    )

    assert result is None
    assert observed_paths == [(raw_path,)]
    assert not (chronovisor_root / "runtime" / "failures" / "packets").exists()

    accepted = SimpleNamespace(terminal_deferred=True)
    monkeypatch.setattr(
        failure_supervisor,
        "_operational_deferred_raw_files_unlocked",
        lambda _paths: {"unrelated.md": "pending_local_repair"},
    )
    monkeypatch.setattr(
        failure_supervisor,
        "_record_semantic_no_quorum_defer_unlocked",
        lambda **_kwargs: accepted,
    )

    unrelated = (
        failure_supervisor.record_semantic_no_quorum_defer_unless_operational_hold(
            raw_path=raw_path,
            error=_no_quorum_error(_sha256(artifact)),
            raw_text=raw_path.read_text(),
        )
    )

    assert unrelated is accepted


def test_authority_artifact_change_reopens_and_unreadable_artifact_fails_closed(
    semantic_defer_wiki: tuple[Path, Path],
) -> None:
    from chronovisor.raw import failure_supervisor

    chronovisor_root, artifact = semantic_defer_wiki
    raw_path = chronovisor_root / "raw" / "epoch.md"
    raw_path.write_text("epoch-bound source\n", encoding="utf-8")
    original_sha256 = _sha256(artifact)
    failure_supervisor.record_raw_failure(
        raw_path=raw_path,
        error=_no_quorum_error(original_sha256),
    )

    assert failure_supervisor.operational_deferred_raw_files([raw_path]) == {
        raw_path.name: "semantic_no_quorum"
    }
    artifact.unlink()
    assert failure_supervisor.operational_deferred_raw_files([raw_path]) == {
        raw_path.name: "semantic_no_quorum"
    }

    artifact.write_bytes(b'{"epoch":2}\n')

    assert _sha256(artifact) != original_sha256
    assert failure_supervisor.operational_deferred_raw_files([raw_path]) == {}
    assert raw_path.is_file()


@pytest.mark.parametrize(
    ("source", "error", "artifact_sha256"),
    [
        (
            "bootstrap_current_policy",
            "adoption_artifact_invalid:incomplete",
            None,
        ),
        (
            "adopted_artifact",
            "installed model metadata differs from evaluation",
            "b" * 64,
        ),
        ("adopted_artifact", None, "not-a-sha256"),
    ],
    ids=("not-adopted", "adopted-with-error", "invalid-hash"),
)
def test_authority_change_releases_only_a_valid_adopted_artifact(
    semantic_defer_wiki: tuple[Path, Path],
    monkeypatch: pytest.MonkeyPatch,
    source: str,
    error: str | None,
    artifact_sha256: str | None,
) -> None:
    from chronovisor.decision import decision_router
    from chronovisor.raw import failure_supervisor

    chronovisor_root, artifact = semantic_defer_wiki
    raw_path = chronovisor_root / "raw" / "validated-authority.md"
    raw_path.write_text("authority-gated source\n", encoding="utf-8")
    original_sha256 = _sha256(artifact)
    failure_supervisor.record_raw_failure(
        raw_path=raw_path,
        error=_no_quorum_error(original_sha256),
    )
    artifact.write_bytes(b'{"epoch":2}\n')
    new_sha256 = _sha256(artifact)
    assert new_sha256 != original_sha256

    monkeypatch.setattr(
        decision_router,
        "resolve_router_policy",
        lambda _config: SimpleNamespace(
            source=source,
            error=error,
            artifact_sha256=(
                new_sha256 if artifact_sha256 == "b" * 64 else artifact_sha256
            ),
        ),
    )

    assert failure_supervisor.operational_deferred_raw_files([raw_path]) == {
        raw_path.name: "semantic_no_quorum"
    }

    monkeypatch.setattr(
        decision_router,
        "resolve_router_policy",
        lambda _config: SimpleNamespace(
            source="adopted_artifact",
            error=None,
            artifact_sha256=new_sha256,
        ),
    )
    assert failure_supervisor.operational_deferred_raw_files([raw_path]) == {}


@pytest.mark.parametrize(
    "state_loss",
    ["missing", "corrupt", "entry_missing"],
)
def test_packet_reconstructs_semantic_hold_after_state_publish_crash(
    semantic_defer_wiki: tuple[Path, Path],
    state_loss: str,
) -> None:
    from chronovisor.raw import failure_supervisor

    chronovisor_root, artifact = semantic_defer_wiki
    raw_path = chronovisor_root / "raw" / f"crash-{state_loss}.md"
    raw_path.write_bytes(f"crash evidence {state_loss}\n".encode())
    authority_sha256 = _sha256(artifact)
    failure_supervisor.record_raw_failure(
        raw_path=raw_path,
        error=_no_quorum_error(authority_sha256),
    )
    state_path = chronovisor_root / "runtime" / "failures" / "state.json"
    if state_loss == "missing":
        state_path.unlink()
    elif state_loss == "corrupt":
        state_path.write_text("{not-json", encoding="utf-8")
    else:
        state_path.write_text('{"failures": {}}\n', encoding="utf-8")

    assert failure_supervisor.operational_deferred_raw_files([raw_path]) == {
        raw_path.name: "semantic_no_quorum"
    }
    artifact.unlink()
    assert failure_supervisor.operational_deferred_raw_files([raw_path]) == {
        raw_path.name: "semantic_no_quorum"
    }

    artifact.write_bytes(b'{"epoch":2}\n')

    assert _sha256(artifact) != authority_sha256
    assert failure_supervisor.operational_deferred_raw_files([raw_path]) == {}


def test_packet_reconstruction_requires_matching_raw_evidence_and_active_status(
    semantic_defer_wiki: tuple[Path, Path],
) -> None:
    from chronovisor.raw import failure_supervisor

    chronovisor_root, artifact = semantic_defer_wiki
    raw_path = chronovisor_root / "raw" / "evidence.md"
    original = b"packet-bound raw\n"
    raw_path.write_bytes(original)
    result = failure_supervisor.record_raw_failure(
        raw_path=raw_path,
        error=_no_quorum_error(_sha256(artifact)),
    )
    (chronovisor_root / "runtime" / "failures" / "state.json").unlink()

    raw_path.write_bytes(b"changed bytes\n")
    assert failure_supervisor.operational_deferred_raw_files([raw_path]) == {}

    raw_path.write_bytes(original)
    assert failure_supervisor.operational_deferred_raw_files([raw_path]) == {
        raw_path.name: "semantic_no_quorum"
    }

    packet_path = Path(str(result.packet_path))
    packet = json.loads(packet_path.read_text())
    packet["status"] = "superseded_semantic_defer"
    packet_path.write_text(json.dumps(packet), encoding="utf-8")

    assert failure_supervisor.operational_deferred_raw_files([raw_path]) == {}


def test_reset_releases_orphan_semantic_packet_before_state_cleanup(
    semantic_defer_wiki: tuple[Path, Path],
) -> None:
    from chronovisor.raw import failure_supervisor

    chronovisor_root, artifact = semantic_defer_wiki
    raw_path = chronovisor_root / "raw" / "orphan-reset.md"
    raw_path.write_text("orphan packet evidence\n", encoding="utf-8")
    result = failure_supervisor.record_raw_failure(
        raw_path=raw_path,
        error=_no_quorum_error(_sha256(artifact)),
    )
    (chronovisor_root / "runtime" / "failures" / "state.json").unlink()
    assert failure_supervisor.operational_deferred_raw_files([raw_path]) == {
        raw_path.name: "semantic_no_quorum"
    }

    failure_supervisor.reset_raw_failure(raw_path.name)

    packet = json.loads(Path(str(result.packet_path)).read_text())
    assert packet["status"] == "semantic_defer_released"
    assert packet["terminal_deferred"] is False
    assert packet["released_at"]
    assert failure_supervisor.operational_deferred_raw_files([raw_path]) == {}


@pytest.mark.parametrize(
    "packet_status",
    ["semantic_defer_released", "superseded_semantic_defer"],
)
def test_released_packet_overrides_stale_terminal_state_entry(
    semantic_defer_wiki: tuple[Path, Path],
    packet_status: str,
) -> None:
    from chronovisor.raw import failure_supervisor

    chronovisor_root, artifact = semantic_defer_wiki
    raw_path = chronovisor_root / "raw" / f"stale-state-{packet_status}.md"
    raw_path.write_text("stale state evidence\n", encoding="utf-8")
    result = failure_supervisor.record_raw_failure(
        raw_path=raw_path,
        error=_no_quorum_error(_sha256(artifact)),
    )
    packet_path = Path(str(result.packet_path))
    packet = json.loads(packet_path.read_text())
    packet["status"] = packet_status
    packet["terminal_deferred"] = False
    packet_path.write_text(json.dumps(packet), encoding="utf-8")

    # Simulates a crash after packet release but before state.json cleanup.
    assert failure_supervisor.operational_deferred_raw_files([raw_path]) == {}


def test_semantic_defer_supersedes_unshared_operational_packet(
    semantic_defer_wiki: tuple[Path, Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from chronovisor.raw import failure_supervisor

    chronovisor_root, artifact = semantic_defer_wiki
    raw_path = chronovisor_root / "raw" / "upgrade.md"
    raw_path.write_text("upgrade source\n", encoding="utf-8")
    starts: list[Path] = []
    monkeypatch.setattr(failure_supervisor, "_launch_self_heal", starts.append)

    operational = failure_supervisor.record_raw_failure(
        raw_path=raw_path,
        error="local consensus semantic no quorum: authority marker missing",
    )
    state_path = chronovisor_root / "runtime" / "failures" / "state.json"
    state = json.loads(state_path.read_text(encoding="utf-8"))
    state["failures"][raw_path.name].pop("self_heal_queued")
    state["failures"][raw_path.name].pop("packet_path")
    state_path.write_text(json.dumps(state), encoding="utf-8")
    semantic = failure_supervisor.record_raw_failure(
        raw_path=raw_path,
        error=_no_quorum_error(_sha256(artifact)),
    )

    assert starts == [Path(str(operational.packet_path))]
    assert semantic.packet_path != operational.packet_path
    old_packet = json.loads(Path(str(operational.packet_path)).read_text())
    assert old_packet["status"] == "superseded_semantic_defer"
    assert old_packet["self_heal_queued"] is False
    assert old_packet["superseded_at"]
    assert old_packet["superseded_by_packet"] == semantic.packet_path
    state = json.loads((chronovisor_root / "runtime" / "failures" / "state.json").read_text())
    assert state["operational_failures"] == {}
    assert state["failures"][raw_path.name]["packet_path"] == semantic.packet_path


def test_semantic_defer_preserves_replaced_terminal_packet(
    semantic_defer_wiki: tuple[Path, Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from chronovisor.raw import failure_supervisor

    chronovisor_root, artifact = semantic_defer_wiki
    raw_path = chronovisor_root / "raw" / "terminal-upgrade.md"
    raw_path.write_text("terminal source\n", encoding="utf-8")
    monkeypatch.setattr(failure_supervisor, "_launch_self_heal", lambda _path: None)
    operational = failure_supervisor.record_raw_failure(
        raw_path=raw_path,
        error="local consensus semantic no quorum: authority marker missing",
    )
    state_path = chronovisor_root / "runtime" / "failures" / "state.json"
    state = json.loads(state_path.read_text(encoding="utf-8"))
    state["failures"][raw_path.name].pop("self_heal_queued")
    state["failures"][raw_path.name].pop("packet_path")
    state_path.write_text(json.dumps(state), encoding="utf-8")
    packet_path = Path(str(operational.packet_path))
    old_packet = json.loads(packet_path.read_text(encoding="utf-8"))
    old_packet["status"] = "local_quarantined"
    old_packet["terminal_deferred"] = True
    packet_path.write_text(json.dumps(old_packet), encoding="utf-8")
    terminal_before = packet_path.read_bytes()

    semantic = failure_supervisor.record_raw_failure(
        raw_path=raw_path,
        error=_no_quorum_error(_sha256(artifact)),
    )

    assert semantic.terminal_deferred is True
    assert packet_path.read_bytes() == terminal_before
    updated_state = json.loads(state_path.read_text(encoding="utf-8"))
    assert updated_state["failures"][raw_path.name]["packet_path"] == (
        semantic.packet_path
    )
    assert updated_state["operational_failures"] != {}


def test_in_flight_operational_worker_observes_semantic_defer_cancellation(
    semantic_defer_wiki: tuple[Path, Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from chronovisor.decision.local_repair import LocalRepairDecision
    from chronovisor.ops import self_heal
    from chronovisor.raw import failure_supervisor

    chronovisor_root, artifact = semantic_defer_wiki
    raw_path = chronovisor_root / "raw" / "in-flight-upgrade.md"
    raw_path.write_text("in-flight source\n", encoding="utf-8")
    monkeypatch.setattr(failure_supervisor, "_launch_self_heal", lambda _path: None)
    operational = failure_supervisor.record_raw_failure(
        raw_path=raw_path,
        error="local consensus semantic no quorum: authority marker missing",
    )
    operational_packet = Path(str(operational.packet_path))
    entered_model = threading.Event()
    release_model = threading.Event()
    forbidden_effects: list[str] = []

    def blocked_repair(
        _packet: dict[str, object],
        *,
        use_qwen: bool = True,
    ) -> LocalRepairDecision:
        del use_qwen
        entered_model.set()
        assert release_model.wait(timeout=5), "test did not release local model"
        return LocalRepairDecision(
            status="escalate",
            action="propose_prompt_fix",
            confidence=1.0,
            reason="would route to repair plane without cancellation",
            source="deterministic",
        )

    def record_effect(name: str):
        def inner(*_args: object, **_kwargs: object) -> object:
            forbidden_effects.append(name)
            return chronovisor_root / f"unexpected-{name}.json"

        return inner

    monkeypatch.setattr(self_heal, "propose_repair", blocked_repair)
    monkeypatch.setattr(
        self_heal,
        "_frontier_eligibility",
        lambda _packet: (SimpleNamespace(notes={}), None),
    )
    monkeypatch.setattr(self_heal, "apply_local_decision", record_effect("mutation"))
    monkeypatch.setattr(self_heal, "_queue_frontier", record_effect("queue"))
    monkeypatch.setattr(self_heal, "_run_frontier", record_effect("frontier"))
    monkeypatch.setattr(
        self_heal,
        "_promote_operational_source_packet",
        record_effect("promotion"),
    )
    outcome: dict[str, object] = {}
    worker_errors: list[BaseException] = []

    def run_worker() -> None:
        try:
            outcome.update(
                self_heal.handle_packet(
                    operational_packet,
                    use_qwen=False,
                    enable_frontier=True,
                    max_attempts=1,
                    backoff_base_seconds=0,
                )
            )
        except BaseException as exc:  # pragma: no cover - asserted below
            worker_errors.append(exc)

    worker = threading.Thread(target=run_worker, daemon=True)
    worker.start()
    assert entered_model.wait(timeout=5), "worker did not enter local model"

    semantic = failure_supervisor.record_raw_failure(
        raw_path=raw_path,
        error=_no_quorum_error(_sha256(artifact)),
    )
    superseded = json.loads(operational_packet.read_text(encoding="utf-8"))
    cancellation_path = Path(str(superseded["cancellation_path"]))
    assert cancellation_path.is_file()
    assert semantic.packet_path != operational.packet_path

    release_model.set()
    worker.join(timeout=5)

    assert not worker.is_alive()
    assert worker_errors == []
    assert outcome["status"] == "superseded_semantic_defer"
    assert outcome["cancelled"] is True
    assert forbidden_effects == []
    observed = json.loads(operational_packet.read_text(encoding="utf-8"))
    assert observed["status"] == "superseded_semantic_defer"
    assert observed["cancellation_observed_at"]
    assert observed["superseded_by_packet"] == semantic.packet_path
    assert operational_packet not in self_heal.pending_packets()


def test_cancellation_dry_run_is_byte_for_byte_read_only(
    semantic_defer_wiki: tuple[Path, Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from chronovisor.ops import self_heal
    from chronovisor.raw import failure_supervisor

    chronovisor_root, _artifact = semantic_defer_wiki
    raw_path = chronovisor_root / "raw" / "dry-run-cancel.md"
    raw_path.write_text("dry run source\n", encoding="utf-8")
    monkeypatch.setattr(failure_supervisor, "_launch_self_heal", lambda _path: None)
    operational = failure_supervisor.record_raw_failure(
        raw_path=raw_path,
        error="local consensus semantic no quorum: authority marker missing",
    )
    packet_path = Path(str(operational.packet_path))
    cancellation = self_heal.request_packet_cancellation(
        packet_path,
        reason="test_dry_run",
        superseded_by_packet=chronovisor_root / "runtime" / "semantic.json",
    )
    cancellation_path = Path(str(cancellation["cancellation_path"]))
    packet_before = packet_path.read_bytes()
    cancellation_before = cancellation_path.read_bytes()

    result = self_heal.handle_packet(packet_path, dry_run=True)

    assert result["status"] == "dry_run"
    assert result["projected_status"] == "superseded_semantic_defer"
    assert result["would_cancel"] is True
    assert packet_path.read_bytes() == packet_before
    assert cancellation_path.read_bytes() == cancellation_before


@pytest.mark.parametrize("reader_mode", ["missing", "invalid"])
def test_semantic_defer_survives_unreadable_packet_during_cancellation_publish(
    semantic_defer_wiki: tuple[Path, Path],
    monkeypatch: pytest.MonkeyPatch,
    reader_mode: str,
) -> None:
    from chronovisor.core import self_heal_cancellation
    from chronovisor.raw import failure_supervisor

    chronovisor_root, artifact = semantic_defer_wiki
    raw_path = chronovisor_root / "raw" / f"cancel-{reader_mode}.md"
    raw_path.write_text("cancellation fail-closed source\n", encoding="utf-8")
    monkeypatch.setattr(failure_supervisor, "_launch_self_heal", lambda _path: None)
    operational = failure_supervisor.record_raw_failure(
        raw_path=raw_path,
        error="local consensus semantic no quorum: authority marker missing",
    )

    if reader_mode == "missing":

        def unavailable(_path: Path) -> dict[str, object]:
            raise FileNotFoundError("simulated cancellation read race")

        monkeypatch.setattr(self_heal_cancellation, "read_json", unavailable)
    else:
        monkeypatch.setattr(self_heal_cancellation, "read_json", lambda _path: [])

    semantic = failure_supervisor.record_raw_failure(
        raw_path=raw_path,
        error=_no_quorum_error(_sha256(artifact)),
    )

    old_packet = json.loads(Path(str(operational.packet_path)).read_text())
    cancellation = json.loads(Path(old_packet["cancellation_path"]).read_text())
    assert semantic.terminal_deferred is True
    assert old_packet["status"] == "superseded_semantic_defer"
    assert cancellation["status"] == "superseded_semantic_defer"
    assert cancellation["failure_id"] is None
    assert cancellation["fingerprint"] is None


def test_semantic_defer_keeps_operational_packet_shared_by_another_raw(
    semantic_defer_wiki: tuple[Path, Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from chronovisor.raw import failure_supervisor

    chronovisor_root, artifact = semantic_defer_wiki
    first = chronovisor_root / "raw" / "shared-first.md"
    second = chronovisor_root / "raw" / "shared-second.md"
    first.write_text("first\n", encoding="utf-8")
    second.write_text("second\n", encoding="utf-8")
    monkeypatch.setattr(failure_supervisor, "_launch_self_heal", lambda _path: None)
    missing_authority = "local consensus semantic no quorum: authority marker missing"
    first_operational = failure_supervisor.record_raw_failure(
        raw_path=first,
        error=missing_authority,
    )
    second_operational = failure_supervisor.record_raw_failure(
        raw_path=second,
        error=missing_authority,
    )
    assert second_operational.packet_path == first_operational.packet_path
    state_path = chronovisor_root / "runtime" / "failures" / "state.json"
    state = json.loads(state_path.read_text(encoding="utf-8"))
    state["failures"][first.name].pop("self_heal_queued")
    state["failures"][second.name].pop("self_heal_queued")
    state["failures"][first.name].pop("packet_path")
    state["failures"][second.name].pop("packet_path")
    state_path.write_text(json.dumps(state), encoding="utf-8")

    semantic = failure_supervisor.record_raw_failure(
        raw_path=first,
        error=_no_quorum_error(_sha256(artifact)),
    )

    old_packet = json.loads(Path(str(first_operational.packet_path)).read_text())
    assert old_packet["status"] == "pending_local_repair"
    assert "superseded_by_packet" not in old_packet
    state = json.loads((chronovisor_root / "runtime" / "failures" / "state.json").read_text())
    assert state["failures"][first.name]["packet_path"] == semantic.packet_path
    assert "packet_path" not in state["failures"][second.name]
    [registry] = state["operational_failures"].values()
    assert registry["packet_path"] == first_operational.packet_path
    assert len(state["operational_failures"]) == 1


def test_semantic_defer_cancels_legacy_packet_when_every_shared_raw_is_replaced(
    semantic_defer_wiki: tuple[Path, Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from chronovisor.raw import failure_supervisor

    chronovisor_root, artifact = semantic_defer_wiki
    first = chronovisor_root / "raw" / "legacy-bundle-first.md"
    second = chronovisor_root / "raw" / "legacy-bundle-second.md"
    first.write_text("first\n", encoding="utf-8")
    second.write_text("second\n", encoding="utf-8")
    monkeypatch.setattr(failure_supervisor, "_launch_self_heal", lambda _path: None)
    missing_authority = "local consensus semantic no quorum: authority marker missing"
    operational = failure_supervisor.record_raw_failure(
        raw_path=first,
        error=missing_authority,
    )
    shared = failure_supervisor.record_raw_failure(
        raw_path=second,
        error=missing_authority,
    )
    assert shared.packet_path == operational.packet_path

    state_path = chronovisor_root / "runtime" / "failures" / "state.json"
    state = json.loads(state_path.read_text(encoding="utf-8"))
    state["failures"][first.name].pop("packet_path")
    state["failures"][second.name].pop("packet_path")
    state_path.write_text(json.dumps(state), encoding="utf-8")

    semantic = failure_supervisor.record_semantic_no_quorum_defer(
        raw_path=first,
        error=_no_quorum_error(_sha256(artifact)),
        related_raw_paths=(second,),
    )

    old_packet = json.loads(Path(str(operational.packet_path)).read_text())
    assert old_packet["status"] == "superseded_semantic_defer"
    assert old_packet["superseded_by_packet"] == semantic.packet_path
    updated_state = json.loads(state_path.read_text(encoding="utf-8"))
    assert updated_state["operational_failures"] == {}
    assert {
        updated_state["failures"][first.name]["packet_path"],
        updated_state["failures"][second.name]["packet_path"],
    } == {semantic.packet_path}


def test_semantic_defer_does_not_trust_mismatched_legacy_registry_binding(
    semantic_defer_wiki: tuple[Path, Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from chronovisor.raw import failure_supervisor

    chronovisor_root, artifact = semantic_defer_wiki
    raw_path = chronovisor_root / "raw" / "legacy-registry-mismatch.md"
    raw_path.write_text("source\n", encoding="utf-8")
    monkeypatch.setattr(failure_supervisor, "_launch_self_heal", lambda _path: None)
    operational = failure_supervisor.record_raw_failure(
        raw_path=raw_path,
        error="local consensus semantic no quorum: authority marker missing",
    )
    packet_path = Path(str(operational.packet_path))
    packet_before = packet_path.read_bytes()
    state_path = chronovisor_root / "runtime" / "failures" / "state.json"
    state = json.loads(state_path.read_text(encoding="utf-8"))
    entry = state["failures"][raw_path.name]
    fingerprint = entry["fingerprint"]
    entry.pop("packet_path")
    state["operational_failures"][fingerprint]["fingerprint"] = "different"
    state_path.write_text(json.dumps(state), encoding="utf-8")

    semantic = failure_supervisor.record_semantic_no_quorum_defer(
        raw_path=raw_path,
        error=_no_quorum_error(_sha256(artifact)),
    )

    assert semantic.terminal_deferred is True
    assert packet_path.read_bytes() == packet_before
    updated_state = json.loads(state_path.read_text(encoding="utf-8"))
    assert fingerprint in updated_state["operational_failures"]


def test_semantic_defer_preserves_direct_packet_reference_with_mismatched_fingerprint(
    semantic_defer_wiki: tuple[Path, Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from chronovisor.raw import failure_supervisor

    chronovisor_root, artifact = semantic_defer_wiki
    first = chronovisor_root / "raw" / "mismatch-shared-first.md"
    second = chronovisor_root / "raw" / "mismatch-shared-second.md"
    first.write_text("first\n", encoding="utf-8")
    second.write_text("second\n", encoding="utf-8")
    monkeypatch.setattr(failure_supervisor, "_launch_self_heal", lambda _path: None)
    missing_authority = "local consensus semantic no quorum: authority marker missing"
    operational = failure_supervisor.record_raw_failure(
        raw_path=first,
        error=missing_authority,
    )
    shared = failure_supervisor.record_raw_failure(
        raw_path=second,
        error=missing_authority,
    )
    assert shared.packet_path == operational.packet_path

    state_path = chronovisor_root / "runtime" / "failures" / "state.json"
    state = json.loads(state_path.read_text(encoding="utf-8"))
    state["failures"][second.name]["fingerprint"] = "inconsistent:fingerprint"
    state_path.write_text(json.dumps(state), encoding="utf-8")
    packet_path = Path(str(operational.packet_path))
    packet_before = packet_path.read_bytes()

    semantic = failure_supervisor.record_raw_failure(
        raw_path=first,
        error=_no_quorum_error(_sha256(artifact)),
    )

    assert semantic.terminal_deferred is True
    assert packet_path.read_bytes() == packet_before
    updated_state = json.loads(state_path.read_text(encoding="utf-8"))
    assert updated_state["failures"][second.name]["packet_path"] == str(packet_path)
