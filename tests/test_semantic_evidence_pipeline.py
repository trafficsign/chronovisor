"""The scored source survives client, fusion, penalties and actual injection."""

import hashlib
from dataclasses import asdict, replace
from types import SimpleNamespace

import pytest

from chronovisor.core import semantic_client
from chronovisor.core.negative_feedback import apply_penalties
from chronovisor.core.pipeline import plain_rrf
from chronovisor.core.recall_context import parse_recall_payload
from chronovisor.core.runtime_config import SearchEmbeddingConfig
from chronovisor.core.search import fuse_results
from chronovisor.core.search_types import ScoredPage, SemanticEvidence
from chronovisor.recall import recall_runtime as runtime

UID = "019fea6e-8a33-7401-b37b-c7afcf6711e1"


@pytest.mark.parametrize("method", ["search", "verify"])
@pytest.mark.parametrize(
    "fusion", [fuse_results, lambda a, b: plain_rrf([("bm25", a), ("semantic", b)])]
)
def test_deep_winning_chunks_reach_actual_injection(
    monkeypatch, tmp_path, method, fusion
):
    source = (
        f"---\nuid: {UID}\ntitle: Transport\nstatus: stable\nsensitivity: normal\n---\n"
        "# Obsolete\nNever use the old port.\n\n"
        "# Current\n現在の接続先は port 7443。\n\n"
        "# Condition\nOnly use it for the staging environment.\n"
    ).encode()
    path = tmp_path / "transport.md"
    path.write_bytes(source)
    digest = hashlib.sha256(source).hexdigest()
    matches = tuple(
        SemanticEvidence(
            "transport", f"{UID}#c{i}", "chunk", i, digest, UID, "gen-test", score
        )
        for i, score in [(1, 0.95), (2, 0.90)]
    )
    response = {
        "generation_id": "gen-test",
        "results": [
            {
                "page_id": "transport",
                "score": 0.95,
                "evidence": [asdict(m) for m in matches],
            }
        ],
    }
    monkeypatch.setattr(semantic_client, "request", lambda *_a, **_k: response)
    store = SimpleNamespace(
        load_existing=lambda: True,
        meta=lambda _id: {
            "title": "Transport",
            "status": "stable",
            "sensitivity": "normal",
            "uid": UID,
        },
    )
    monkeypatch.setattr("chronovisor.core.index_store.get_store", lambda: store)
    config = SearchEmbeddingConfig()
    semantic = (
        semantic_client.search(
            "staging port", 3, include_reference=False, config=config
        )
        if method == "search"
        else semantic_client.verify("staging port", ["transport"], config=config)
    )
    lexical = [
        ScoredPage("transport", "Transport", "", "", 10, content_sha256=digest, uid=UID)
    ]
    candidates = apply_penalties(fusion(lexical, semantic), {"transport": 0.1})
    assert candidates[0].evidence == matches
    monkeypatch.setattr(runtime, "find_readable_page", lambda _id: path)
    monkeypatch.setattr(runtime, "query_hint_page_ids", lambda *_a, **_k: ["transport"])
    monkeypatch.setattr(runtime, "prefetch_page_ids_for_request", lambda *_a, **_k: [])
    monkeypatch.setattr(
        runtime,
        "excerpt_page",
        lambda *_a, **_k: pytest.fail("winning chunk was replaced"),
    )
    policy = runtime.RecallPolicy(max_context_chars=3000)
    items = runtime.collect_context(
        ["staging port"], "read", policy, pre_results=candidates
    )
    assert len(items) == 1 and len(items[0].source_passages) == 2
    result = runtime.RecallResult(
        status="ok",
        decision="read",
        confidence=0.9,
        queries=["staging port"],
        reasons=[],
        matched_terms={},
        context_items=items,
    )
    rendered = runtime.format_recall_context(result, policy)
    payload = parse_recall_payload(rendered)
    assert payload is not None
    assert len(payload["items"]) == 2
    for item in payload["items"]:
        ref = item["source_ref"]
        assert ref["sha256"] == digest and ref["generation_id"] == "gen-test"
        assert source[ref["byte_start"] : ref["byte_end"]].decode() == item["evidence"]
    assert "7443" in rendered and "staging environment" in rendered
    assert "Never use" not in rendered
    # Replacement after retrieval cannot be mistaken for the indexed source.
    path.write_bytes(source.replace(b"7443", b"9999"))
    assert (
        runtime.collect_context(
            ["staging port"], "read", policy, pre_results=candidates
        )
        == []
    )
    # Under a budget too small for one complete cited passage, abstain.
    assert (
        runtime.format_recall_context(result, replace(policy, max_context_chars=400))
        == ""
    )


@pytest.mark.parametrize(
    "field,value",
    [
        ("source_sha256", "bad"),
        ("generation_id", "other"),
        ("ordinal", True),
        ("page_id", "wrong"),
        ("score", float("nan")),
    ],
)
def test_service_evidence_identity_is_validated(field, value):
    match = SemanticEvidence("page", f"{UID}#c0", "chunk", 0, "a" * 64, UID, "g", 0.9)
    row = {"page_id": "page", "evidence": [{**asdict(match), field: value}]}
    with pytest.raises(
        semantic_client.SemanticServiceUnavailable, match="invalid semantic evidence"
    ):
        semantic_client._response_evidence(row, "g")
    assert semantic_client._response_evidence({"page_id": "page"}, None) == ()
