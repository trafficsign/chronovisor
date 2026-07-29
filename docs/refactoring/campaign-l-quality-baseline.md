# Campaign L quality baseline

Campaign L makes the initial static-quality contract executable without
silently grandfathering arbitrary findings.

## Ruff

Ruff 0.16 is a locked development dependency. The repository-wide command is:

```sh
uv run ruff check src scripts tests
```

The enforced profile is `E4`, `E7`, `E9`, `F`, `I`, `UP`, `B`, `SIM`, and
`RUF100`. All findings in that profile were repaired across `src`, `scripts`,
and `tests`; no existing violation is grandfathered.

Four semantic-style rules remain explicitly excluded:

- `UP042`: converting `(str, Enum)` classes to `StrEnum` can change serialized
  and string representations;
- `SIM105`: explicit exception suppression often documents operational intent;
- `SIM115`: atomic temporary-file lifetimes intentionally outlive a lexical
  context manager;
- `SIM117`: nested resource acquisition makes lock and release order explicit.

The ignores are exact rule codes rather than file-wide or category-wide
waivers, and `RUF100` rejects stale suppressions.

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
- The configured Ruff profile covers 319 Python files with zero findings.
- Public module imports and durable legacy-path migrations remain executable.
- Compileall and whitespace checks: passed.
