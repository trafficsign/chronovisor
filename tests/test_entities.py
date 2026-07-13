from __future__ import annotations

from contextlib import contextmanager
import json
from pathlib import Path

from llm_wiki_mcp import entities
from llm_wiki_mcp.entities import extract_entities, patch_entities_frontmatter


def _frontier_decision(
    decision: str, summary: str = "reviewed exact entity proposal"
) -> dict:
    return {
        "decision": decision,
        "summary": summary,
        "tests_run": ["checked alias evidence and exact frontmatter diff"],
        "commit": None,
        "committed": False,
        "pushed": False,
        "risk": None,
        "notes": None,
    }


def _entity_page(title: str = "Qwen Notes", body: str = "notes") -> str:
    return f"---\ntitle: {title}\n---\n{body}\n"


def test_extract_entities_uses_alias_registry() -> None:
    registry = {"mhi": ["MHI", "三菱重工"], "llm-wiki": ["LLM Wiki"]}

    assert extract_entities("三菱重工と LLM Wiki の話", registry=registry) == [
        "mhi",
        "llm-wiki",
    ]


def test_patch_entities_frontmatter_merges_existing() -> None:
    registry = {"ollama": ["Ollama"], "qwen": ["Qwen"]}
    text = "---\ntitle: Local Models\nentities: [qwen]\n---\nOllama and Qwen notes.\n"

    out = patch_entities_frontmatter(text, registry=registry)

    assert "entities: [qwen, ollama]" in out


def test_entity_proposal_validator_proves_alias_backed_frontmatter_only_edit() -> None:
    registry = {"qwen": ["Qwen"]}
    original = _entity_page()
    updated = entities.patch_entities_frontmatter(original, registry=registry)
    proposal = entities.build_semantic_mutation_proposal(
        page_id="memory",
        operation="backfill_entities_frontmatter",
        expected_text=original,
        updated_text=updated,
        details=entities._review_evidence(original, updated, registry=registry),
    )

    assert (
        entities.validate_entity_backfill_proposal(
            proposal,
            expected_text=original,
            updated_text=updated,
            registry=registry,
        )
        is True
    )

    corrupted = json.loads(json.dumps(proposal))
    corrupted["details"]["alias_evidence"][0]["matched_aliases"] = []
    assert entities.validate_entity_backfill_proposal(corrupted) is False


def test_entity_proposal_validator_rejects_replacement_disguised_as_backfill() -> None:
    registry = {
        "qwen": ["Qwen"],
        "ollama": ["Ollama"],
        "codex": ["Codex"],
    }
    original = (
        "---\n"
        "title: Qwen and Codex Notes\n"
        "entities: [qwen, ollama]\n"
        "---\n"
        "Ollama and Codex notes.\n"
    )
    tampered = original.replace(
        "entities: [qwen, ollama]",
        "entities: [qwen, codex]",
    )
    proposal = entities.build_semantic_mutation_proposal(
        page_id="memory",
        operation="backfill_entities_frontmatter",
        expected_text=original,
        updated_text=tampered,
        details=entities._review_evidence(original, tampered, registry=registry),
    )

    assert (
        entities.validate_entity_backfill_proposal(
            proposal,
            expected_text=original,
            updated_text=tampered,
            registry=registry,
        )
        is False
    )


def test_invalid_entity_proposal_never_reaches_reviewer_or_page(
    tmp_path: Path,
    monkeypatch,
) -> None:
    page = tmp_path / "memory.md"
    original = _entity_page()
    page.write_text(original, encoding="utf-8")
    monkeypatch.setattr(entities, "all_pages", lambda: [page])
    monkeypatch.setattr(entities, "load_registry", lambda: {"qwen": ["Qwen"]})
    real_builder = entities.build_semantic_mutation_proposal

    def corrupt(**kwargs):
        proposal = real_builder(**kwargs)
        proposal["details"]["alias_evidence"][0]["matched_aliases"] = []
        return proposal

    monkeypatch.setattr(entities, "build_semantic_mutation_proposal", corrupt)
    result = entities.backfill_entities(
        reviewer=lambda _prompt, _schema: (_ for _ in ()).throw(
            AssertionError("invalid deterministic proposal must not reach a model")
        ),
        artifact_dir=tmp_path / "reviews",
    )

    assert result["invalid_proposals"] == 1
    assert result["frontier_calls"] == 0
    assert result["updated"] == 0
    assert page.read_text(encoding="utf-8") == original


def test_backfill_preserves_correction_that_lands_before_cas(
    tmp_path: Path,
    monkeypatch,
) -> None:
    page = tmp_path / "memory.md"
    original = "---\ntitle: Qwen Notes\n---\nold fact\n"
    corrected = "---\ntitle: Qwen Notes\n---\nuser-corrected fact\n"
    page.write_text(original, encoding="utf-8")
    monkeypatch.setattr(entities, "all_pages", lambda: [page])
    monkeypatch.setattr(entities, "load_registry", lambda: {"qwen": ["Qwen"]})

    @contextmanager
    def correction_wins():
        page.write_text(corrected, encoding="utf-8")
        yield

    monkeypatch.setattr(entities, "wiki_mutation_lock", correction_wins)

    result = entities.backfill_entities(
        reviewer=lambda _prompt, _schema: _frontier_decision("approved"),
        artifact_dir=tmp_path / "reviews",
    )

    assert result["updated"] == 0
    assert page.read_text(encoding="utf-8") == corrected


def test_backfill_applies_only_after_frontier_approval(
    tmp_path: Path,
    monkeypatch,
) -> None:
    page = tmp_path / "memory.md"
    page.write_text(_entity_page(), encoding="utf-8")
    monkeypatch.setattr(entities, "all_pages", lambda: [page])
    monkeypatch.setattr(entities, "load_registry", lambda: {"qwen": ["Qwen"]})

    result = entities.backfill_entities(
        reviewer=lambda prompt, _schema: (
            _frontier_decision("approved")
            if '"added_entities": [\n      "qwen"' in prompt
            else (_ for _ in ()).throw(AssertionError("missing exact entity evidence"))
        ),
        artifact_dir=tmp_path / "reviews",
    )

    assert result["updated"] == 1
    assert result["frontier_calls"] == 1
    assert "entities: [qwen]" in page.read_text(encoding="utf-8")


def test_large_entity_backfill_uses_exact_changed_spans_packet(
    tmp_path: Path,
    monkeypatch,
) -> None:
    page = tmp_path / "memory.md"
    original = _entity_page(
        title="Qwen Notes",
        body="".join(f"context line {index:05d}\n" for index in range(6_000)),
    )
    page.write_text(original, encoding="utf-8")
    monkeypatch.setattr(entities, "all_pages", lambda: [page])
    monkeypatch.setattr(entities, "load_registry", lambda: {"qwen": ["Qwen"]})
    reviews = tmp_path / "reviews"
    prompts: list[str] = []

    result = entities.backfill_entities(
        reviewer=lambda prompt, _schema: (
            prompts.append(prompt) or _frontier_decision("approved")
        ),
        artifact_dir=reviews,
    )

    proposal = json.loads(
        next((reviews / "proposals").glob("*.json")).read_text(encoding="utf-8")
    )["proposal"]
    assert proposal["review_packet"]["mode"] == "changed_spans"
    assert proposal["review_packet"]["coverage"]["all_changed_spans_rendered"] is True
    assert proposal["unified_diff"] is None
    assert '"kind": "entity_alias_semantic_evidence"' in prompts[0]
    assert result["invalid_proposals"] == 0
    assert result["updated"] == 1
    assert "entities: [qwen]" in page.read_text(encoding="utf-8")


def test_local_substring_proposal_cannot_mutate_on_frontier_retry(
    tmp_path: Path,
    monkeypatch,
) -> None:
    page = tmp_path / "memory.md"
    original = _entity_page(title="Codex Notes")
    page.write_text(original, encoding="utf-8")
    monkeypatch.setattr(entities, "all_pages", lambda: [page])
    monkeypatch.setattr(entities, "load_registry", lambda: {"codex": ["Codex"]})

    result = entities.backfill_entities(
        reviewer=lambda _prompt, _schema: _frontier_decision("needs_retry"),
        artifact_dir=tmp_path / "reviews",
    )

    assert result["updated"] == 0
    assert result["retry"] == 1
    assert page.read_text(encoding="utf-8") == original


def test_frontier_rejection_is_durable_and_non_mutating(
    tmp_path: Path,
    monkeypatch,
) -> None:
    page = tmp_path / "memory.md"
    original = _entity_page(title="Codex Notes")
    page.write_text(original, encoding="utf-8")
    monkeypatch.setattr(entities, "all_pages", lambda: [page])
    monkeypatch.setattr(entities, "load_registry", lambda: {"codex": ["Codex"]})
    calls = 0

    def reject(_prompt, _schema):
        nonlocal calls
        calls += 1
        return _frontier_decision("rejected")

    first = entities.backfill_entities(
        reviewer=reject,
        artifact_dir=tmp_path / "reviews",
    )
    second = entities.backfill_entities(
        reviewer=lambda _prompt, _schema: (_ for _ in ()).throw(
            AssertionError("durable rejection must be reused")
        ),
        artifact_dir=tmp_path / "reviews",
    )

    assert calls == 1
    assert first["rejected"] == second["rejected"] == 1
    assert second["frontier_calls"] == 0
    assert page.read_text(encoding="utf-8") == original


def test_durable_approval_is_reused_after_pre_apply_failure(
    tmp_path: Path,
    monkeypatch,
) -> None:
    page = tmp_path / "memory.md"
    page.write_text(_entity_page(), encoding="utf-8")
    monkeypatch.setattr(entities, "all_pages", lambda: [page])
    monkeypatch.setattr(entities, "load_registry", lambda: {"qwen": ["Qwen"]})
    real_apply = entities._apply_entities_cas
    monkeypatch.setattr(
        entities, "_apply_entities_cas", lambda *_args, **_kwargs: "write_error"
    )

    first = entities.backfill_entities(
        reviewer=lambda _prompt, _schema: _frontier_decision("approved"),
        artifact_dir=tmp_path / "reviews",
    )
    assert first["updated"] == 0
    assert first["retry"] == 1

    monkeypatch.setattr(entities, "_apply_entities_cas", real_apply)
    second = entities.backfill_entities(
        reviewer=lambda _prompt, _schema: (_ for _ in ()).throw(
            AssertionError("durable approval must be reused")
        ),
        artifact_dir=tmp_path / "reviews",
    )

    assert second["updated"] == 1
    assert second["frontier_calls"] == 0
    assert "entities: [qwen]" in page.read_text(encoding="utf-8")


def test_dry_run_is_fully_read_only(
    tmp_path: Path,
    monkeypatch,
) -> None:
    page = tmp_path / "memory.md"
    original = _entity_page()
    page.write_text(original, encoding="utf-8")
    monkeypatch.setattr(entities, "all_pages", lambda: [page])
    monkeypatch.setattr(entities, "load_registry", lambda: {"qwen": ["Qwen"]})
    reviews = tmp_path / "reviews"

    result = entities.backfill_entities(
        dry_run=True,
        reviewer=lambda _prompt, _schema: (_ for _ in ()).throw(
            AssertionError("dry-run must not call frontier")
        ),
        artifact_dir=reviews,
    )

    assert result["updated"] == 1
    assert result["frontier_calls"] == 0
    assert page.read_text(encoding="utf-8") == original
    assert not reviews.exists()


def test_budget_defer_persists_proposal_and_resumes_next_run(
    tmp_path: Path,
    monkeypatch,
) -> None:
    first_page = tmp_path / "first.md"
    second_page = tmp_path / "second.md"
    first_page.write_text(_entity_page(body="first"), encoding="utf-8")
    second_page.write_text(_entity_page(body="second"), encoding="utf-8")
    monkeypatch.setattr(entities, "all_pages", lambda: [first_page, second_page])
    monkeypatch.setattr(entities, "load_registry", lambda: {"qwen": ["Qwen"]})
    reviews = tmp_path / "reviews"

    first = entities.backfill_entities(
        max_frontier_calls=1,
        reviewer=lambda _prompt, _schema: _frontier_decision("approved"),
        artifact_dir=reviews,
    )

    assert first["updated"] == 1
    assert first["budget_deferred"] == 1
    assert first["frontier_calls"] == 1
    assert len(list((reviews / "proposals").glob("*.json"))) == 2

    second = entities.backfill_entities(
        max_frontier_calls=1,
        reviewer=lambda _prompt, _schema: _frontier_decision("approved"),
        artifact_dir=reviews,
    )

    assert second["updated"] == 1
    assert second["frontier_calls"] == 1
    assert "entities: [qwen]" in second_page.read_text(encoding="utf-8")


def test_both_cli_entrypoints_forward_frontier_budget(monkeypatch, capsys) -> None:
    from llm_wiki_mcp import cli

    seen: list[int] = []

    def fake_backfill(**kwargs):
        seen.append(kwargs["max_frontier_calls"])
        return {"status": "ok", "pages": []}

    monkeypatch.setattr(entities, "backfill_entities", fake_backfill)

    assert entities.main(["backfill", "--max-frontier-calls", "2", "--json"]) == 0
    assert (
        cli.main(["entities", "backfill", "--max-frontier-calls", "3", "--json"]) == 0
    )
    assert seen == [2, 3]
    capsys.readouterr()
