from __future__ import annotations

import hashlib
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from chronovisor.core.search_types import ScoredPage
from chronovisor.recall import recall_runtime


@pytest.fixture
def fastpath_snapshot(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    monkeypatch.setattr(recall_runtime, "CHRONOVISOR_ROOT", tmp_path)
    page = tmp_path / "page.md"
    page.write_text(
        "---\n"
        "uid: current-uid\n"
        "title: Snapshot topic\n"
        "status: stable\n"
        "updated: 2026-09-05\n"
        "---\n"
        "Current snapshot body.\n",
        encoding="utf-8",
    )
    current_hash = hashlib.sha256(page.read_bytes()).hexdigest()

    monkeypatch.setattr(
        recall_runtime,
        "find_readable_page",
        lambda page_id: page if page_id == "page" else None,
    )
    monkeypatch.setattr(
        recall_runtime,
        "_record_distilled_exposure",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        recall_runtime,
        "_observe_shadow_distillation_policy",
        lambda *_args, **_kwargs: None,
    )

    from chronovisor.recall import recall_distillation

    monkeypatch.setattr(
        recall_distillation,
        "build_text_features",
        lambda *_args, **_kwargs: {
            "query_chargram_coverage": 1.0,
            "candidate_chargram_precision": 1.0,
        },
    )
    monkeypatch.setattr(
        recall_distillation,
        "score_fast_features",
        lambda *_args, **_kwargs: 0.9,
    )

    policy = recall_runtime.RecallPolicy(
        log_decisions=False,
        max_context_chars=2_000,
    )
    distilled_policy = SimpleNamespace(
        policy_id="fast-v2",
        feature_schema="recall-distill-text-v2",
        threshold=0.6,
        margin=0.0,
        max_cards=1,
    )

    def run(candidate: ScoredPage) -> recall_runtime.RecallResult:
        monkeypatch.setattr(
            recall_runtime,
            "search_existing_lexical",
            lambda *_args, **_kwargs: ([candidate], []),
        )
        started = time.monotonic()
        return recall_runtime._run_distilled_fast_path(
            active_request=recall_runtime.RecallRequest(
                host="codex",
                event="UserPromptSubmit",
                prompt="snapshot topic",
            ),
            policy=policy,
            distilled_policy=distilled_policy,
            matched={},
            started=started,
            final_deadline_at=started + 1.0,
            telemetry=None,
        )

    return current_hash, run, page


@pytest.mark.parametrize(
    ("candidate_hash", "candidate_uid", "admitted"),
    [
        ("old", "current-uid", False),
        ("current", "wrong-uid", False),
        ("current", "current-uid", True),
    ],
)
def test_distilled_fast_path_rebinds_snapshot_identity(
    fastpath_snapshot,
    candidate_hash: str,
    candidate_uid: str,
    admitted: bool,
) -> None:
    current_hash, run, _page = fastpath_snapshot
    index_hash = (
        "0" * 64
        if candidate_hash == "old"
        else current_hash
    )
    result = run(
        ScoredPage(
            page_id="page",
            title="Snapshot topic",
            folder="",
            updated="2026-09-05",
            score=1.0,
            snippet="snapshot topic",
            content_sha256=index_hash,
            uid=candidate_uid,
        )
    )

    if admitted:
        assert result.decision == "search"
        assert [item.page_id for item in result.context_items] == ["page"]
        assert result.context_items[0].uid == "current-uid"
        assert result.page_content_hashes == {"page": current_hash}
    else:
        assert result.decision == "none"
        assert result.context_items == []
        assert getattr(result, "page_content_hashes", {}) == {}


def test_distilled_fast_path_allows_legacy_page_without_uid(fastpath_snapshot) -> None:
    current_hash, run, page = fastpath_snapshot
    page.write_text(
        page.read_text(encoding="utf-8").replace("uid: current-uid\n", ""),
        encoding="utf-8",
    )
    current_hash = hashlib.sha256(page.read_bytes()).hexdigest()

    result = run(
        ScoredPage(
            page_id="page",
            title="Snapshot topic",
            folder="",
            updated="2026-09-05",
            score=1.0,
            snippet="snapshot topic",
            content_sha256=current_hash,
            uid="",
        )
    )

    assert result.decision == "search"
    assert [item.page_id for item in result.context_items] == ["page"]
    assert result.context_items[0].uid == ""
    assert result.page_content_hashes == {"page": current_hash}


def test_distilled_fast_path_rejects_missing_uid_for_uid_page(fastpath_snapshot) -> None:
    current_hash, run, _page = fastpath_snapshot
    result = run(
        ScoredPage(
            page_id="page",
            title="Snapshot topic",
            folder="",
            updated="2026-09-05",
            score=1.0,
            snippet="snapshot topic",
            content_sha256=current_hash,
            uid="",
        )
    )

    assert result.decision == "none"
    assert result.context_items == []
    assert getattr(result, "page_content_hashes", {}) == {}
