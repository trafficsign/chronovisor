"""Compatibility support for one-generation legacy module aliases."""

from __future__ import annotations

import sys
from importlib import import_module
from types import ModuleType
from typing import Any


def alias_legacy_module(name: str, implementation: ModuleType) -> None:
    """Expose a relocated implementation through its previous import path."""
    sys.modules[name] = implementation


def install_legacy_package(package_name: str, implementation_name: str) -> None:
    """Forward a package's legacy module attributes to one implementation.

    Four historical modules now share their name with a domain package. A
    module subclass preserves reads and monkeypatch-style writes without
    replacing the package object needed for importing domain submodules.
    """

    package = sys.modules[package_name]
    implementation_leaf = implementation_name.rsplit(".", 1)[-1]

    class LegacyPackage(ModuleType):
        def _implementation(self) -> ModuleType:
            return import_module(implementation_name)

        def __getattribute__(self, name: str) -> Any:
            if name == implementation_leaf:
                implementation = import_module(implementation_name)
                if hasattr(implementation, name):
                    return getattr(implementation, name)
            return super().__getattribute__(name)

        def __getattr__(self, name: str) -> Any:
            if name == implementation_leaf:
                return self._implementation()
            try:
                return getattr(self._implementation(), name)
            except AttributeError as exc:
                raise AttributeError(
                    f"module {package_name!r} has no attribute {name!r}"
                ) from exc

        def __setattr__(self, name: str, value: Any) -> None:
            if (
                name.startswith("__")
                or (
                    isinstance(value, ModuleType)
                    and value.__name__.startswith(f"{package_name}.")
                )
            ):
                super().__setattr__(name, value)
                return
            implementation = self._implementation()
            if hasattr(implementation, name):
                setattr(implementation, name, value)
                return
            super().__setattr__(name, value)

        def __delattr__(self, name: str) -> None:
            if name.startswith("__") or name == implementation_leaf:
                super().__delattr__(name)
                return
            implementation = self._implementation()
            if hasattr(implementation, name):
                delattr(implementation, name)
                return
            super().__delattr__(name)

        def __dir__(self) -> list[str]:
            return sorted(set(super().__dir__()) | set(dir(self._implementation())))

    package.__class__ = LegacyPackage
