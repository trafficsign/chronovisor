from __future__ import annotations

from chronovisor.search.research_config import load_research_config
from chronovisor.search.research_consolidation import run_consolidation


def test_research_config_defaults_fail_closed(tmp_path) -> None:
    path = tmp_path / "config.toml"
    path.write_text('[research]\nenabled = true\nmode = "unknown"\n', encoding="utf-8")

    config = load_research_config(path)

    assert config.enabled is True
    assert config.mode == "off"
    assert config.web.live_egress_enabled is False
    assert config.resources.max_concurrent_generations == 1


def test_legacy_research_model_selectors_are_accepted_but_ignored(tmp_path) -> None:
    path = tmp_path / "config.toml"
    path.write_text(
        """
[research]
planner_model = "override-planner"
challenge_model = "override-challenge"
tie_break_model = "override-tie"
""".strip()
        + "\n",
        encoding="utf-8",
    )

    config = load_research_config(path)

    assert all(
        not hasattr(config, key)
        for key in ("planner_model", "challenge_model", "tie_break_model")
    )


def test_invalid_consolidation_mutation_mode_is_blocked(tmp_path) -> None:
    path = tmp_path / "config.toml"
    path.write_text(
        """
[research.consolidation]
enabled = true
mutation_mode = "direct_write"
""".strip()
        + "\n",
        encoding="utf-8",
    )

    config = load_research_config(path)

    assert config.consolidation_mutation_mode == "direct_write"
    assert run_consolidation(config=config) == {
        "status": "blocked",
        "reason": "mutation_mode_not_allowlisted",
    }


def test_federated_web_config_loads_only_bounded_source_pack_settings(tmp_path) -> None:
    path = tmp_path / "config.toml"
    path.write_text(
        """
[research.web]
provider = "federated"
source_packs = ["general", "code", "academic", "encyclopedia"]
searxng_endpoint = "http://127.0.0.1:8888"
allow_local_search_backend = true
max_provider_calls = 99
per_provider_limit = 99
provider_timeout_seconds = 99
""".strip()
        + "\n",
        encoding="utf-8",
    )

    config = load_research_config(path).web

    assert config.provider == "federated"
    assert config.source_packs == (
        "general",
        "code",
        "academic",
        "encyclopedia",
    )
    assert config.allow_local_search_backend is True
    assert config.max_provider_calls == 4
    assert config.per_provider_limit == 5
    assert config.provider_timeout_seconds == 15.0
