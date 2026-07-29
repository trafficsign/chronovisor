"""Single command dispatcher for Chronovisor lab workloads."""

from __future__ import annotations

import importlib
import sys
from collections.abc import Callable, Sequence
from typing import Any, Final

COMMANDS: Final[dict[str, tuple[str, bool]]] = {
    "adoption-corpus": ("chronovisor.lab.adoption_corpus", True),
    "classification-annif": ("chronovisor.lab.classification_annif", False),
    "classification-calibrate": (
        "chronovisor.lab.classification_calibration",
        True,
    ),
    "classification-library-pilot": (
        "chronovisor.lab.classification_library_pilot",
        False,
    ),
    "classification-migrate": ("chronovisor.lab.classification_migration", True),
    "classification-pilot": ("chronovisor.lab.classification_pilot", False),
    "classification-pilot-v2": (
        "chronovisor.lab.classification_pilot_v2",
        False,
    ),
    "classification-profile-pilot": (
        "chronovisor.lab.classification_profile_pilot",
        True,
    ),
    "classification-query2doc-pilot": (
        "chronovisor.lab.classification_query2doc_pilot",
        True,
    ),
    "classification-query2doc-unseen": (
        "chronovisor.lab.classification_query2doc_unseen",
        True,
    ),
    "librarian-burn": ("chronovisor.lab.librarian_burn", True),
    "local-model-eval": ("chronovisor.lab.local_model_eval", True),
    "model": ("chronovisor.lab.model_lab", True),
    "research-eval": ("chronovisor.lab.research_eval", True),
}


def _help() -> str:
    commands = "\n".join(f"  {name}" for name in sorted(COMMANDS))
    return f"usage: chronovisor-lab <command> [args ...]\n\ncommands:\n{commands}"


def _invoke(module_name: str, forwards_argv: bool, args: list[str]) -> int:
    module = importlib.import_module(module_name)
    entry: Callable[..., Any] = module.main
    if forwards_argv:
        result = entry(args)
    else:
        original_argv = sys.argv
        try:
            sys.argv = [f"chronovisor-lab {args[0] if args else module_name}", *args]
            result = entry()
        finally:
            sys.argv = original_argv
    return int(result or 0)


def main(argv: Sequence[str] | None = None) -> int:
    """Dispatch one isolated lab workload and return its process status."""
    arguments = list(sys.argv[1:] if argv is None else argv)
    if not arguments or arguments[0] in {"-h", "--help"}:
        print(_help())
        return 0
    command, *forwarded = arguments
    target = COMMANDS.get(command)
    if target is None:
        print(f"chronovisor-lab: unknown command: {command}", file=sys.stderr)
        print(_help(), file=sys.stderr)
        return 2
    return _invoke(target[0], target[1], forwarded)


if __name__ == "__main__":
    raise SystemExit(main())
