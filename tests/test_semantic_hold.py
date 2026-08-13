from __future__ import annotations

import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from chronovisor.decision import semantic_hold
from chronovisor.decision.decision_router import QUORUM_SAFETY_POLICY_VERSION
from chronovisor.decision.local_structured import (
    LocalStructuredResult,
    StructuredAttempt,
)
from tests.semantic_hold_support import (
    semantic_authority as _authority,
)
from tests.semantic_hold_support import (
    semantic_review as _semantic_review,
)
from tests.semantic_hold_support import (
    structured_review_epoch as _epoch,
)


def test_semantic_no_quorum_hold_is_strict_self_hashed_and_extractable() -> None:
    authority = _authority()
    epoch = _epoch(authority)
    review = _semantic_review(authority)

    hold = semantic_hold.build_semantic_no_quorum_hold(
        "recall_auto_apply", epoch, authority, review
    )
    assert set(hold) == {
        "schema_version",
        "kind",
        "lane",
        "epoch",
        "epoch_sha256",
        "authority",
        "authority_sha256",
        "frontier_failure",
        "local_consensus",
        "decision_policy",
        "review_sha256",
        "hold_sha256",
    }
    assert (
        semantic_hold.semantic_no_quorum_hold_error(
            hold, "recall_auto_apply", epoch, authority
        )
        is None
    )
    assert (
        semantic_hold.persisted_semantic_no_quorum_hold(
            {"result": {"semantic_hold": hold}},
            "recall_auto_apply",
            epoch,
            authority,
        )
        == hold
    )
    hold["epoch"]["prompt_sha256"] = "0" * 64
    assert "digest" in str(
        semantic_hold.semantic_no_quorum_hold_error(
            hold, "recall_auto_apply", epoch, authority
        )
    )
    assert (
        semantic_hold.persisted_semantic_no_quorum_hold(
            {"semantic_hold": hold}, "recall_auto_apply", epoch, authority
        )
        is None
    )


def test_semantic_hold_accepts_current_producer_session_audit() -> None:
    authority = _authority()
    epoch = _epoch(authority)
    review = _semantic_review(authority)
    for vote in review["local_consensus"]["votes"]:
        signature = vote["signature_sha256"]
        vote["session"] = LocalStructuredResult(
            ok=True,
            model=vote["model"],
            attempts=(StructuredAttempt(0, True, signature, 120, False, None, ()),),
            think="medium",
            ollama_think="medium",
            num_predict=256,
            think_selection_reason="medium_default",
            required_num_ctx=8_000,
            requested_num_ctx=32_768,
            effective_num_ctx=32_768,
        ).audit_record()

    hold = semantic_hold.build_semantic_no_quorum_hold(
        "recall_auto_apply", epoch, authority, review
    )

    assert (
        semantic_hold.semantic_no_quorum_hold_error(
            hold, "recall_auto_apply", epoch, authority
        )
        is None
    )
    assert (
        semantic_hold.persisted_semantic_no_quorum_hold(
            {"semantic_hold": hold}, "recall_auto_apply", epoch, authority
        )
        == hold
    )

    review["local_consensus"]["votes"][0]["session"]["context_tokens"] *= 2
    with pytest.raises(ValueError, match="session audit is invalid"):
        semantic_hold.build_semantic_no_quorum_hold(
            "recall_auto_apply", epoch, authority, review
        )


def test_authority_observation_detects_a_b_a_file_generation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from chronovisor.core import runtime_config

    config_path = tmp_path / "config.toml"
    config_path.write_text("authority = 'a'\n", encoding="utf-8")
    monkeypatch.setattr(runtime_config, "CONFIG_FILE", config_path)
    authority = _authority()
    before = semantic_hold.structured_review_authority_observation_sha256(authority)

    config_path.write_text("authority = 'b'\n", encoding="utf-8")
    config_path.write_text("authority = 'a'\n", encoding="utf-8")
    after = semantic_hold.structured_review_authority_observation_sha256(authority)

    assert before != after
    assert len(before) == len(after) == 64


def test_semantic_hold_rejects_operational_failure_and_success() -> None:
    authority = _authority()
    epoch = _epoch(authority)
    operational = _semantic_review(authority)
    operational["frontier_failure"]["failure_class"] = "local_consensus_failed"
    success = _semantic_review(authority)
    success.pop("frontier_failure")
    success["decision"] = "approved"
    plaintext = _semantic_review(authority)
    plaintext["raw_output"] = "model response must not persist"

    assert semantic_hold.is_local_semantic_no_quorum(operational) is False
    assert semantic_hold.is_local_semantic_no_quorum(success) is False
    with pytest.raises(ValueError):
        semantic_hold.build_semantic_no_quorum_hold(
            "recall_auto_apply", epoch, authority, operational
        )
    with pytest.raises(ValueError):
        semantic_hold.build_semantic_no_quorum_hold(
            "recall_auto_apply", epoch, authority, success
        )
    with pytest.raises(ValueError, match="forbids plaintext"):
        semantic_hold.build_semantic_no_quorum_hold(
            "recall_auto_apply", epoch, authority, plaintext
        )


@pytest.mark.parametrize(
    "change",
    [
        "lane",
        "authority",
        "quorum_safety_policy_version",
        "schema",
        "prompt",
        "system",
        "effective_request",
        "resolver",
    ],
)
def test_structured_review_cache_misses_every_epoch_change(
    tmp_path: Path,
    change: str,
) -> None:
    authority = _authority()
    epoch = _epoch(authority)
    cache = semantic_hold.StructuredReviewSemanticHoldCache(tmp_path / "cache")
    with cache.locked(
        lane="recall_auto_apply", epoch=epoch, authority=authority
    ) as lease:
        lease.store(_semantic_review(authority))

    changed_lane = "recall_calibration" if change == "lane" else "recall_auto_apply"
    if change == "authority":
        changed_authority = _authority(changed_lane, artifact_sha256="9" * 64)
    elif change == "quorum_safety_policy_version":
        changed_authority = _authority(
            changed_lane,
            quorum_safety_policy_version=QUORUM_SAFETY_POLICY_VERSION - 1,
        )
    else:
        changed_authority = _authority(changed_lane)
    changed_epoch = _epoch(
        changed_authority,
        lane=changed_lane,
        schema_sha256="8" * 64 if change == "schema" else "e" * 64,
        prompt="different prompt" if change == "prompt" else "private prompt",
        system=None if change == "system" else "private system",
        effective_request_sha256=(
            "7" * 64 if change == "effective_request" else "f" * 64
        ),
        resolver_sha256="6" * 64
        if change == "resolver"
        else (semantic_hold.STRUCTURED_REVIEW_HOLD_RESOLVER_SHA256),
    )
    with cache.locked(
        lane=changed_lane,
        epoch=changed_epoch,
        authority=changed_authority,
    ) as lease:
        assert lease.load() is None


def test_structured_review_v1_epoch_is_a_safe_miss(tmp_path: Path) -> None:
    authority = _authority()
    old_epoch = {
        **_epoch(authority),
        "epoch_version": 1,
        "authority_observation_sha256": "0" * 64,
    }

    assert (
        semantic_hold.structured_review_hold_epoch_error(
            old_epoch,
            lane="recall_auto_apply",
            authority=authority,
        )
        is not None
    )
    cache = semantic_hold.StructuredReviewSemanticHoldCache(tmp_path / "cache")
    with pytest.raises(ValueError), cache.locked(
        lane="recall_auto_apply",
        epoch=old_epoch,
        authority=authority,
    ):
        pass
    assert not (tmp_path / "cache" / "entries").exists()


def test_structured_review_cache_corruption_is_miss_and_atomic_rewrite(
    tmp_path: Path,
) -> None:
    authority = _authority()
    epoch = _epoch(authority)
    review = _semantic_review(authority)
    cache = semantic_hold.StructuredReviewSemanticHoldCache(tmp_path / "cache")
    with cache.locked(
        lane="recall_auto_apply", epoch=epoch, authority=authority
    ) as lease:
        lease.store(review)
        cache_path = lease.cache_path
    cache_path.write_text('{"partial":', encoding="utf-8")

    with cache.locked(
        lane="recall_auto_apply", epoch=epoch, authority=authority
    ) as lease:
        assert lease.load() is None
        lease.store(review)
        assert lease.load() == review

    parsed = json.loads(cache_path.read_text(encoding="utf-8"))
    assert parsed["result_sha256"] == semantic_hold.canonical_sha256(review)
    raw = cache_path.read_text(encoding="utf-8")
    assert "private prompt" not in raw
    assert "private system" not in raw


def test_structured_review_cache_huge_integer_json_is_a_safe_miss(
    tmp_path: Path,
) -> None:
    authority = _authority()
    epoch = _epoch(authority)
    cache = semantic_hold.StructuredReviewSemanticHoldCache(tmp_path / "cache")

    with cache.locked(
        lane="recall_auto_apply",
        epoch=epoch,
        authority=authority,
    ) as lease:
        lease.cache_path.write_text(
            '{"huge_integer":' + "9" * 10_000 + "}",
            encoding="utf-8",
        )
        assert lease.load() is None


@pytest.mark.parametrize("error_type", [RecursionError, OverflowError])
def test_structured_review_cache_decoder_runtime_failure_is_a_safe_miss(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    error_type: type[Exception],
) -> None:
    authority = _authority()
    epoch = _epoch(authority)
    cache = semantic_hold.StructuredReviewSemanticHoldCache(tmp_path / "cache")

    with cache.locked(
        lane="recall_auto_apply",
        epoch=epoch,
        authority=authority,
    ) as lease:
        lease.cache_path.write_text("{}", encoding="utf-8")
        monkeypatch.setattr(
            semantic_hold.json,
            "loads",
            lambda _raw: (_ for _ in ()).throw(error_type()),
        )
        assert lease.load() is None


def test_structured_review_cache_lock_coalesces_concurrent_writers(
    tmp_path: Path,
) -> None:
    authority = _authority()
    epoch = _epoch(authority)
    review = _semantic_review(authority)
    cache = semantic_hold.StructuredReviewSemanticHoldCache(tmp_path / "cache")
    barrier = threading.Barrier(4)
    writes = 0
    writes_lock = threading.Lock()

    def worker() -> dict[str, object]:
        nonlocal writes
        barrier.wait()
        with cache.locked(
            lane="recall_auto_apply", epoch=epoch, authority=authority
        ) as lease:
            cached = lease.load()
            if cached is None:
                with writes_lock:
                    writes += 1
                time.sleep(0.05)
                lease.store(review)
                cached = lease.load()
            assert cached is not None
            return cached

    with ThreadPoolExecutor(max_workers=4) as executor:
        results = list(executor.map(lambda _index: worker(), range(4)))

    assert writes == 1
    assert results == [review] * 4


def test_structured_review_cache_never_persists_other_results(
    tmp_path: Path,
) -> None:
    authority = _authority()
    epoch = _epoch(authority)
    cache = semantic_hold.StructuredReviewSemanticHoldCache(tmp_path / "cache")
    operational = _semantic_review(authority)
    operational["frontier_failure"]["failure_class"] = "local_consensus_failed"

    with cache.locked(
        lane="recall_auto_apply", epoch=epoch, authority=authority
    ) as lease:
        with pytest.raises(ValueError):
            lease.store(operational)
        assert lease.load() is None
        assert not lease.cache_path.exists()
