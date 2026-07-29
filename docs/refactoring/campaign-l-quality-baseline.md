# Campaign L quality baseline

Campaign L makes the initial static-quality contract executable without
silently grandfathering arbitrary findings.

## Ruff

Ruff 0.16 is a locked development dependency. The repository-wide command is:

```sh
uv run ruff check src scripts tests
```

The first enforced rules are `E9`, `F63`, `F7`, and `F82`: syntax errors,
invalid control-flow constructs, and undefined names. The broader default rule
set currently reports 1,797 findings, so enabling it wholesale would either
mix a large formatting campaign into this refactor or require opaque global
ignores. Additional rule families should be enabled only with their debt fixed
in the same change.

The fatal baseline immediately found and fixed two real defects:

- legacy semantic-defer recovery returned an undefined `authority_sha256`
  instead of its already-validated `authority_epoch`;
- `_ollama_engine_identity()` placed its return logic after an unrelated
  function's return and therefore always returned `None`.

Both paths now have explicit regression assertions.

## Mypy

Mypy 2.3 is strict for all 17 modules under `src/chronovisor/core`:

```sh
uv run mypy
```

No broad error-code disables or per-module ignores are present. The initial
strict pass added concrete generic types, compatibility-proxy annotations, a
typed context-manager return, validated Ollama embedding response shape, and
the engine-identity return fix. Outer domains remain outside the configured
file set until they can be made strict without weakening `core`.

## Public API documentation

All 49 console entry-point callables now have docstrings, up from 1 of 49.
Across entry-point modules, documented public functions increased from 107 of
489 (21.9%) to 155 of 489 (31.7%). Across the full package, the public-function
rate increased from 538 of 1,466 (36.7%) to 586 of 1,466 (40.0%).

`tests/test_quality_baseline.py` prevents entry-point docstrings or tool scope
from silently regressing.

## Verification

- Ruff: all configured checks passed.
- Mypy: no issues in 17 source files.
- Quality, Ollama identity, raw replay, and module-layout tests: 48 passed.
- Compileall and whitespace checks: passed.
