"""Deterministic star-export metadata for module access analysis."""

from __future__ import annotations

import ast
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Literal

from .access_model import FlowValue

StarExportKind = Literal["public", "static", "dynamic"]


@dataclass(frozen=True)
class StarExportPolicy:
    kind: StarExportKind
    names: tuple[str, ...] = ()

    @classmethod
    def public(cls) -> StarExportPolicy:
        return cls("public")

    @classmethod
    def from_all_expression(cls, expression: ast.expr) -> StarExportPolicy:
        if not isinstance(expression, (ast.List, ast.Tuple)):
            return cls("dynamic")
        names: list[str] = []
        for element in expression.elts:
            if not isinstance(element, ast.Constant) or not isinstance(
                element.value, str
            ):
                return cls("dynamic")
            names.append(element.value)
        return cls("static", tuple(names))

    @classmethod
    def joined(cls, policies: Iterable[StarExportPolicy]) -> StarExportPolicy:
        materialized = list(policies)
        if not materialized:
            return cls.public()
        first = materialized[0]
        if all(policy == first for policy in materialized[1:]):
            return first
        return cls("dynamic")

    def select(self, bindings: dict[str, FlowValue]) -> dict[str, FlowValue]:
        if self.kind == "dynamic":
            names = tuple(sorted(name for name in bindings if name != "__all__"))
        elif self.kind == "static":
            names = self.names
        else:
            names = tuple(sorted(name for name in bindings if not name.startswith("_")))
        return {name: bindings[name].copy() for name in names if name in bindings}

    def select_definite_names(self, names: Iterable[str]) -> frozenset[str]:
        available = frozenset(names)
        if self.kind == "dynamic":
            return frozenset()
        if self.kind == "static":
            return frozenset(name for name in self.names if name in available)
        return frozenset(name for name in available if not name.startswith("_"))


@dataclass(frozen=True)
class ModuleExportSummary:
    bindings: dict[str, FlowValue]
    definite_bindings: frozenset[str]
    star_bindings: dict[str, FlowValue]
    star_definite: frozenset[str]
    star_policy: StarExportPolicy

    @classmethod
    def empty(cls) -> ModuleExportSummary:
        policy = StarExportPolicy.public()
        return cls({}, frozenset(), {}, frozenset(), policy)


@dataclass(frozen=True)
class ModuleExportTable:
    summaries: dict[str, ModuleExportSummary]

    @property
    def bindings(self) -> dict[str, dict[str, FlowValue]]:
        return {module: summary.bindings for module, summary in self.summaries.items()}

    @property
    def definite_bindings(self) -> dict[str, frozenset[str]]:
        return {
            module: summary.definite_bindings
            for module, summary in self.summaries.items()
        }

    @property
    def star_bindings(self) -> dict[str, dict[str, FlowValue]]:
        return {
            module: summary.star_bindings for module, summary in self.summaries.items()
        }

    @property
    def star_definite(self) -> dict[str, frozenset[str]]:
        return {
            module: summary.star_definite for module, summary in self.summaries.items()
        }

    @property
    def star_policies(self) -> dict[str, StarExportPolicy]:
        return {
            module: summary.star_policy for module, summary in self.summaries.items()
        }


__all__ = [
    "ModuleExportSummary",
    "ModuleExportTable",
    "StarExportPolicy",
]
