"""Single command dispatcher for Chronovisor lab workloads."""

from __future__ import annotations

import sys
from collections.abc import Sequence
from typing import Final

COMMANDS: Final[dict[str, tuple[str, bool]]] = {}


def _help() -> str:
    commands = "\n".join(f"  {name}" for name in sorted(COMMANDS))
    return f"usage: chronovisor-lab <command> [args ...]\n\ncommands:\n{commands}"


def main(argv: Sequence[str] | None = None) -> int:
    """Keep the retired lab command surface available for help and errors."""
    arguments = list(sys.argv[1:] if argv is None else argv)
    if not arguments or arguments[0] in {"-h", "--help"}:
        print(_help())
        return 0
    command = arguments[0]
    print(f"chronovisor-lab: unknown command: {command}", file=sys.stderr)
    print(_help(), file=sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
