from pathlib import Path


def test_periodic_review_uses_one_autonomous_full_sweep() -> None:
    wrapper = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "chronovisor-librarian-review"
    ).read_text(encoding="utf-8")

    assert wrapper.count("chronovisor_exec_uvx") == 1
    assert "--full-sweep \\\n  --json \\\n  >/dev/null" in wrapper
    assert "--review-collection-queue" not in wrapper
    assert "--review-model" not in wrapper
