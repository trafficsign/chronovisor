from __future__ import annotations

from llm_wiki_mcp.research_config import load_research_config


def test_research_config_defaults_fail_closed(tmp_path) -> None:
    path = tmp_path / "config.toml"
    path.write_text('[research]\nenabled = true\nmode = "unknown"\n', encoding="utf-8")

    config = load_research_config(path)

    assert config.enabled is True
    assert config.mode == "off"
    assert config.web.live_egress_enabled is False
    assert config.resources.max_concurrent_generations == 1
