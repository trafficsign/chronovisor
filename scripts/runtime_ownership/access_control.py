"""Control-flow expression helpers for runtime access discovery."""

from __future__ import annotations

import ast


def match_values(pattern: ast.pattern) -> list[ast.expr]:
    values: list[ast.expr] = []
    if isinstance(pattern, ast.MatchValue):
        values.append(pattern.value)
    elif isinstance(pattern, ast.MatchSequence):
        for child in pattern.patterns:
            values.extend(match_values(child))
    elif isinstance(pattern, ast.MatchMapping):
        values.extend(pattern.keys)
        for child in pattern.patterns:
            values.extend(match_values(child))
    elif isinstance(pattern, ast.MatchClass):
        values.append(pattern.cls)
        for child in [*pattern.patterns, *pattern.kwd_patterns]:
            values.extend(match_values(child))
    elif isinstance(pattern, ast.MatchAs) and pattern.pattern is not None:
        values.extend(match_values(pattern.pattern))
    elif isinstance(pattern, ast.MatchOr):
        for child in pattern.patterns:
            values.extend(match_values(child))
    return values


__all__ = ["match_values"]
