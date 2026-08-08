# ruff: noqa: F401, F403, F405
"""Runtime ownership cli layer."""

from __future__ import annotations

import argparse
import ast
import copy
import hashlib
import io
import json
import plistlib
import re
import subprocess
import tarfile
import tomllib
from collections import Counter
from pathlib import Path, PurePosixPath
from typing import Any

from .discovery import *
from .gate import *
from .model import *
from .registry import *
from .seed import *
from .source import *


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).parents[2])
    parser.add_argument("--output", type=Path)
    actions = parser.add_mutually_exclusive_group()
    actions.add_argument("--check", action="store_true")
    actions.add_argument("--generate-registry", action="store_true")
    actions.add_argument("--bootstrap-baseline", action="store_true")
    actions.add_argument("--retire-missing-resources", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.bootstrap_baseline:
        payload = build_runtime_state_baseline(args.root)
        exit_code = 0
    elif args.generate_registry:
        payload = build_runtime_state_registry(args.root)
        exit_code = 0
    elif args.retire_missing_resources:
        payload = retire_missing_runtime_state(args.root)
        exit_code = 0
    else:
        payload = runtime_state_fitness(args.root)
        exit_code = 0 if payload["passed"] else 1
    encoded = _json_document_bytes(payload).decode("utf-8")
    if args.output is None:
        print(encoded, end="")
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded, encoding="utf-8")
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())

__all__ = ["parse_args", "main"]
