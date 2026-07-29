from __future__ import annotations

from chronovisor.research.research_types import BudgetUsage, ResearchBudget, parse_action


def test_action_schema_rejects_unknown_fields_and_types() -> None:
    unknown = parse_action(
        {"type": "chronovisor_search", "arguments": {}, "rationale": "x", "extra": True},
        epoch=2,
    )
    invalid = parse_action({"type": "shell", "arguments": {}}, epoch=2)

    assert unknown.action is None
    assert "unknown action fields" in unknown.error
    assert invalid.action is None
    assert "unknown action type" in invalid.error


def test_role_budget_does_not_let_planner_consume_challenge_allowance() -> None:
    budget = ResearchBudget(max_planner_calls=1, max_challenge_calls=2, max_total_model_calls=2)
    usage = BudgetUsage()

    assert usage.consume(budget, "planner_calls") is True
    assert usage.consume(budget, "planner_calls") is False
    assert usage.consume(budget, "challenge_calls") is True
    assert usage.consume(budget, "challenge_calls") is False
    assert usage.total_model_calls == 2
