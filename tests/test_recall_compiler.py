from __future__ import annotations

import hashlib
import json

from chronovisor.recall import recall_compiler


def write_claims(path, rows) -> None:
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def claim(page_id: str, digest: str, value: str = "64GB") -> dict:
    return {
        "claim_id": f"{page_id}:capacity",
        "source_page": page_id,
        "subject": "Mac Studio",
        "predicate": "fact.capacity",
        "semantic_slot": "memory",
        "value": value,
        "valid_from": "2026-07-30",
        "valid_to": None,
        "status": "active",
        "source_line": 12,
        "source_sha256": digest,
    }


def test_compiler_returns_only_digest_verified_exact_meaning_address(
    tmp_path,
    monkeypatch,
) -> None:
    page = tmp_path / "mac-studio.md"
    page.write_text("Mac Studio memory 64GB", encoding="utf-8")
    digest = hashlib.sha256(page.read_bytes()).hexdigest()
    claims = tmp_path / "claims.jsonl"
    write_claims(claims, [claim("mac-studio", digest)])
    monkeypatch.setattr(
        recall_compiler,
        "find_page",
        lambda page_id: page if page_id == "mac-studio" else None,
    )

    result = recall_compiler.compile_query(
        "Mac Studio の memory 容量は何GB?",
        claims_path=claims,
    )

    assert result["status"] == "exact"
    assert result["page_ids"] == ["mac-studio"]
    assert result["authority"] == "shadow"


def test_compiler_falls_back_on_conflict_superseded_or_digest_drift(
    tmp_path,
    monkeypatch,
) -> None:
    page = tmp_path / "mac-studio.md"
    page.write_text("changed", encoding="utf-8")
    claims = tmp_path / "claims.jsonl"
    stale_digest = hashlib.sha256(b"old").hexdigest()
    write_claims(
        claims,
        [
            claim("mac-studio", stale_digest, "64GB"),
            claim("mac-studio", stale_digest, "128GB"),
            {
                **claim("old-page", stale_digest),
                "status": "superseded",
            },
        ],
    )
    monkeypatch.setattr(recall_compiler, "find_page", lambda _page_id: page)

    conflict = recall_compiler.compile_query(
        "Mac Studio の memory 容量は何GB?",
        claims_path=claims,
    )
    prose = recall_compiler.compile_query(
        "昔の思い出を詳しく説明して",
        claims_path=claims,
    )

    assert conflict["reason"] == "active_claim_conflict"
    assert prose["reason"] == "unstructured_query"


def test_compiler_rejects_source_digest_drift_and_hashes_shadow_prompt(
    tmp_path,
    monkeypatch,
) -> None:
    page = tmp_path / "mac-studio.md"
    page.write_text("changed", encoding="utf-8")
    claims = tmp_path / "claims.jsonl"
    write_claims(
        claims,
        [claim("mac-studio", hashlib.sha256(b"old").hexdigest())],
    )
    monkeypatch.setattr(recall_compiler, "find_page", lambda _page_id: page)

    result = recall_compiler.compile_query(
        "Mac Studio の memory 容量は何GB?",
        claims_path=claims,
    )
    trace_path = tmp_path / "trace.jsonl"
    recall_compiler.append_shadow_trace(
        prompt="private structured prompt",
        compiler=result,
        teacher_page_ids=["teacher"],
        committed_page_ids=[],
        path=trace_path,
    )

    assert result["reason"] == "source_digest_unverified"
    assert "private structured prompt" not in trace_path.read_text(encoding="utf-8")
