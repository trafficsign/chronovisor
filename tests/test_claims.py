from __future__ import annotations

import json
import multiprocessing
from contextlib import contextmanager
from pathlib import Path

import pytest

from chronovisor.recall import claims


def _append_claim_worker(
    ledger_value, page_value, ready, start, attempted, done
) -> None:
    ledger = Path(ledger_value)
    page = Path(page_value)
    claims.CLAIMS_FILE = ledger
    claims.page_claims = lambda *_args, **_kwargs: [
        {
            "source_raw": "raw/new.json",
            "source_page": "new",
            "value": "new durable claim",
        }
    ]
    claims.find_page = lambda _page_id: page
    ready.set()
    if not start.wait(5):
        return
    attempted.set()
    claims.append_page_claims(["new"], source_raw="raw/new.json")
    done.set()


def _sanitize_claim_worker(
    ledger_value, page_value, ready, start, attempted, done
) -> None:
    ledger = Path(ledger_value)
    page = Path(page_value)
    claims.find_page = lambda page_id: page if page_id in {"existing", "new"} else None
    ready.set()
    if not start.wait(5):
        return
    attempted.set()
    claims.sanitize_claim_ledger(path=ledger, write=True)
    done.set()


def test_page_claims_extracts_summary_entities_and_lead(tmp_path: Path, monkeypatch) -> None:
    page = tmp_path / "alpha.md"
    page.write_text(
        "---\n"
        "title: Alpha\n"
        "updated: 2026-07-06\n"
        "summary: Alpha summary\n"
        "entities: [MHI, Codex]\n"
        "---\n"
        "# Alpha Body\n"
        "Important body lead.\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(claims, "find_page", lambda page_id: page if page_id == "alpha" else None)

    rows = claims.page_claims("alpha")

    predicates = {row["predicate"] for row in rows}
    assert {"page.title", "page.summary", "page.entity", "body.lead"} <= predicates
    assert any(row["value"] == "Alpha summary" for row in rows)


def test_search_claims_scores_token_overlap(tmp_path: Path) -> None:
    path = tmp_path / "claims.jsonl"
    path.write_text(
        json.dumps({"claim_id": "c1", "subject": "alpha", "predicate": "page.summary", "value": "MHI Codex memory"})
        + "\n"
        + json.dumps({"claim_id": "c2", "subject": "beta", "predicate": "page.summary", "value": "unrelated"})
        + "\n",
        encoding="utf-8",
    )

    rows = claims.search_claims("Codex memory", path=path)

    assert [row["claim_id"] for row in rows] == ["c1"]


def test_search_claims_excludes_superseded_rows(tmp_path: Path) -> None:
    path = tmp_path / "claims.jsonl"
    path.write_text(
        json.dumps({"claim_id": "old", "subject": "gpu", "predicate": "fact.capacity", "value": "16GB", "status": "superseded"})
        + "\n"
        + json.dumps({"claim_id": "new", "subject": "gpu", "predicate": "fact.capacity", "value": "32GB", "status": "active"})
        + "\n",
        encoding="utf-8",
    )

    rows = claims.search_claims("gpu capacity", path=path)

    assert [row["claim_id"] for row in rows] == ["new"]


def test_page_claims_extracts_evidence_backed_model_and_measurement(tmp_path: Path, monkeypatch) -> None:
    page = tmp_path / "gpu.md"
    page.write_text(
        "---\ntitle: GPU\nupdated: 2026-07-11\nentities: [q-kun]\n---\n"
        "P24U は 32GB、価格は55,399円で確定。\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(claims, "find_page", lambda page_id: page)

    rows = claims.page_claims("gpu", source_raw="raw-1.md")
    facts = [row for row in rows if str(row["predicate"]).startswith("fact.")]

    assert {"fact.model", "fact.price", "fact.capacity", "fact.status"} <= {row["predicate"] for row in facts}
    assert all(row["source_raw"] == "raw-1.md" for row in facts)
    assert all(row["evidence_span"] == "P24U は 32GB、価格は55,399円で確定。" for row in facts)


def test_claim_conflicts_require_same_explicit_semantic_slot() -> None:
    base = {"source_page": "gpu", "subject": "P24U", "predicate": "fact.price", "status": "active"}
    rows = [
        {**base, "claim_id": "unit", "value": "55,399円", "source_line": 1, "semantic_slot": "単価"},
        {**base, "claim_id": "total", "value": "114,110円", "source_line": 2, "semantic_slot": "総額"},
        {**base, "claim_id": "old", "value": "16GB", "predicate": "fact.capacity", "source_line": 3, "semantic_slot": "容量"},
        {**base, "claim_id": "new", "value": "32GB", "predicate": "fact.capacity", "source_line": 4, "semantic_slot": "容量"},
    ]

    conflicts = claims.claim_conflicts(rows)

    assert len(conflicts) == 1
    assert conflicts[0]["semantic_slot"] == "容量"
    assert {row["claim_id"] for row in conflicts[0]["claims"]} == {"old", "new"}


def test_reviewed_claim_state_materializes_approved_invalidations(tmp_path: Path, monkeypatch) -> None:
    review_file = tmp_path / "reviews.jsonl"
    review_file.write_text(
        json.dumps(
            {
                "conflict_id": "conflict-1",
                "reviewed_at": "2026-07-11T12:00:00",
                "authority": "user",
                "valid": True,
                "review": {
                    "decision": "approved",
                    "preferred_claim_ids": ["new"],
                    "invalidated_claim_ids": ["old"],
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(claims, "CLAIM_REVIEW_FILE", review_file)

    state = claims._reviewed_claim_state()

    assert state["old"]["status"] == "superseded"
    assert state["old"]["valid_to"] == "2026-07-11T12:00:00"
    assert state["new"]["status"] == "active"


def test_default_conflict_review_preserves_all_claims(
    tmp_path: Path, monkeypatch
) -> None:
    conflict_file = tmp_path / "conflicts.jsonl"
    review_file = tmp_path / "reviews.jsonl"
    conflict_file.write_text(
        json.dumps(
            {
                "conflict_id": "conflict-1",
                "claims": [
                    {"claim_id": "old", "value": "16GB", "source_raw": "a.md"},
                    {"claim_id": "new", "value": "32GB", "source_raw": "b.md"},
                ],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(claims, "CLAIM_CONFLICT_FILE", conflict_file)
    monkeypatch.setattr(claims, "CLAIM_REVIEW_FILE", review_file)

    result = claims.review_claim_conflicts()

    review = result["results"][0]["review"]
    assert review["decision"] == "preserved"
    assert review["preferred_claim_ids"] == []
    assert review["invalidated_claim_ids"] == []
    assert review_file.exists()


def test_disabled_conflict_lane_preserves_queue_without_review_or_write(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    conflict_file = tmp_path / "conflicts.jsonl"
    review_file = tmp_path / "reviews.jsonl"
    conflict_file.write_text(
        json.dumps({"conflict_id": "conflict-1", "claims": []}) + "\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(claims, "CLAIM_CONFLICT_FILE", conflict_file)
    monkeypatch.setattr(claims, "CLAIM_REVIEW_FILE", review_file)
    monkeypatch.setenv("CHRONOVISOR_DECISION_POLICY_CLAIMS_CONFLICT", "off")

    result = claims.review_claim_conflicts(
        reviewer=lambda *_args, **_kwargs: pytest.fail("disabled lane must not review")
    )

    assert result["status"] == "deferred"
    assert result["pending"] == 1
    assert result["processed"] == 0
    assert not review_file.exists()


def test_append_page_claims_requires_source_raw(tmp_path: Path, monkeypatch) -> None:
    page = tmp_path / "alpha.md"
    page.write_text("---\ntitle: Alpha\nupdated: 2026-07-06\n---\nbody", encoding="utf-8")
    ledger = tmp_path / "claims.jsonl"
    monkeypatch.setattr(claims, "find_page", lambda page_id: page if page_id == "alpha" else None)
    monkeypatch.setattr(claims, "CLAIMS_FILE", ledger)

    payload = claims.append_page_claims(["alpha"], source_raw="")

    assert payload["status"] == "skipped"
    assert not ledger.exists()


def test_sanitize_claim_ledger_drops_placeholders(tmp_path: Path, monkeypatch) -> None:
    real_page = tmp_path / "real.md"
    real_page.write_text("real", encoding="utf-8")
    path = tmp_path / "claims.jsonl"
    path.write_text(
        json.dumps({"source_raw": "", "source_page": "p0", "value": "body"}) + "\n"
        + json.dumps({"source_raw": "raw.md", "source_page": "real", "value": "useful"}) + "\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(claims, "find_page", lambda page_id: real_page if page_id == "real" else None)

    payload = claims.sanitize_claim_ledger(path=path)

    assert payload["kept"] == 1
    assert payload["dropped"] == 1
    assert "useful" in path.read_text(encoding="utf-8")


def test_claim_mutations_share_the_ledger_sidecar_lock(
    tmp_path: Path, monkeypatch
) -> None:
    ledger = tmp_path / "claims.jsonl"
    calls: list[Path] = []

    @contextmanager
    def recording_lock(path: Path):
        calls.append(path)
        yield

    monkeypatch.setattr(claims, "CLAIMS_FILE", ledger)
    monkeypatch.setattr(claims, "_claims_ledger_lock", recording_lock)
    monkeypatch.setattr(
        claims,
        "page_claims",
        lambda *_args, **_kwargs: [
            {
                "source_raw": "raw/alpha.json",
                "source_page": "alpha",
                "value": "durable fact",
            }
        ],
    )
    monkeypatch.setattr(claims, "find_page", lambda _page_id: tmp_path / "alpha.md")

    claims.append_page_claims(["alpha"], source_raw="raw/alpha.json")
    claims.sanitize_claim_ledger(path=ledger, write=True)

    assert calls == [ledger, ledger]
    assert ledger.with_suffix(".jsonl.lock") == tmp_path / "claims.jsonl.lock"


def test_empty_claim_append_does_not_create_or_enter_lock(
    tmp_path: Path, monkeypatch
) -> None:
    ledger = tmp_path / "claims.jsonl"

    @contextmanager
    def forbidden_lock(_path: Path):
        raise AssertionError("empty append must not acquire the ledger lock")
        yield

    monkeypatch.setattr(claims, "CLAIMS_FILE", ledger)
    monkeypatch.setattr(claims, "page_claims", lambda *_args, **_kwargs: [])
    monkeypatch.setattr(claims, "_claims_ledger_lock", forbidden_lock)

    payload = claims.append_page_claims(["empty"], source_raw="raw/empty.json")

    assert payload["written"] == 0
    assert not ledger.exists()
    assert not ledger.with_suffix(".jsonl.lock").exists()


def test_append_and_sanitize_serialize_across_processes(tmp_path: Path) -> None:
    ledger = tmp_path / "claims.jsonl"
    page = tmp_path / "page.md"
    page.write_text("durable", encoding="utf-8")
    write_rows = [
        {
            "source_raw": "raw/existing.json",
            "source_page": "existing",
            "value": "existing durable claim",
        },
        {"source_raw": "", "source_page": "placeholder", "value": "drop me"},
    ]
    ledger.write_text(
        "".join(json.dumps(row) + "\n" for row in write_rows),
        encoding="utf-8",
    )
    context = multiprocessing.get_context("spawn")
    append_ready = context.Event()
    sanitize_ready = context.Event()
    start = context.Event()
    append_attempted = context.Event()
    sanitize_attempted = context.Event()
    append_done = context.Event()
    sanitize_done = context.Event()
    append_process = context.Process(
        target=_append_claim_worker,
        args=(
            str(ledger),
            str(page),
            append_ready,
            start,
            append_attempted,
            append_done,
        ),
    )
    sanitize_process = context.Process(
        target=_sanitize_claim_worker,
        args=(
            str(ledger),
            str(page),
            sanitize_ready,
            start,
            sanitize_attempted,
            sanitize_done,
        ),
    )
    append_process.start()
    sanitize_process.start()
    processes = [append_process, sanitize_process]
    try:
        assert append_ready.wait(5)
        assert sanitize_ready.wait(5)
        with claims._claims_ledger_lock(ledger):
            start.set()
            assert append_attempted.wait(5)
            assert sanitize_attempted.wait(5)
            assert not append_done.wait(0.2)
            assert not sanitize_done.wait(0.2)
        assert append_done.wait(5)
        assert sanitize_done.wait(5)
        for process in processes:
            process.join(5)
            assert process.exitcode == 0
    finally:
        for process in processes:
            if process.is_alive():
                process.terminate()
                process.join(5)

    rows = [
        json.loads(line)
        for line in ledger.read_text(encoding="utf-8").splitlines()
        if line
    ]
    assert {row["value"] for row in rows} == {
        "existing durable claim",
        "new durable claim",
    }
