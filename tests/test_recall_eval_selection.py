from __future__ import annotations

from chronovisor.recall_eval import RecallExample, select_examples


def test_paired_corpus_selection_is_stable_and_kind_balanced() -> None:
    examples = [
        RecallExample(prompt=f"prompt-{index}", kind="missed" if index < 20 else "false-positive", ref=str(index))
        for index in range(40)
    ]
    first = select_examples(examples, limit=10)
    second = select_examples(list(reversed(examples)), limit=10)
    assert [(row.prompt, row.kind) for row in first] == [(row.prompt, row.kind) for row in second]
    assert {row.kind for row in first} == {"missed", "false-positive"}
    assert len(first) == 10


def test_ignored_exposure_is_not_negative_supervision() -> None:
    ignored = RecallExample(prompt="not acknowledged", kind="injection_ignored")
    explicit = RecallExample(prompt="wrong memory", kind="false-positive")

    assert ignored.is_false_positive is False
    assert explicit.is_false_positive is True
