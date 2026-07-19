from __future__ import annotations

from chronovisor.research_config import load_research_config
from chronovisor.research_consolidation import run_consolidation


def test_research_config_defaults_fail_closed(tmp_path) -> None:
    path = tmp_path / "config.toml"
    path.write_text('[research]\nenabled = true\nmode = "unknown"\n', encoding="utf-8")

    config = load_research_config(path)

    assert config.enabled is True
    assert config.mode == "off"
    assert config.web.live_egress_enabled is False
    assert config.resources.max_concurrent_generations == 1


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
