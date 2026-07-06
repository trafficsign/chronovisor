from __future__ import annotations

from llm_wiki_mcp.entities import extract_entities, patch_entities_frontmatter


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
