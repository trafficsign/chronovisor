from pathlib import Path


def test_periodic_review_reconciles_current_scope_before_model_review() -> None:
    wrapper = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "chronovisor-librarian-review"
    ).read_text(encoding="utf-8")

    full_sweep = wrapper.index("--full-sweep")
    primary_review = wrapper.index("--review-collection-queue")

    assert full_sweep < primary_review
    assert "--full-sweep \\\n  --json \\\n  >/dev/null" in wrapper
