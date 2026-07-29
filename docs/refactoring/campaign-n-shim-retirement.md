# Campaign N legacy-shim retirement

Campaign N removes the one-generation import compatibility layer created by
Campaign J. Repository code, tests, scripts, console entry points, and dynamic
module references now use domain package paths directly.

## Removed compatibility surface

- Deleted all 219 top-level forwarding modules.
- Deleted `core.compat` and the four package-object proxy installers.
- Kept only `deadman_observer.py` at package top level. It is an operational
  implementation, not a shim: launchd copies it outside the package so it can
  report failures even when the main archive cannot import.
- Replaced package-level value imports for the four same-name collisions
  (`classification`, `ingest`, `librarian`, and `search`) with explicit
  implementation-module imports.

The package now has 240 importable submodules instead of 459. Importing a
retired two-segment implementation path fails instead of silently loading a
forwarder.

## Durable path migration

Background-job state can outlive a deployed archive, so deleting Python
forwarders cannot make old queued module names unexecutable.
`core.module_paths` records the 219 retired modules and the four same-name
package collisions. `BackgroundJobStore` canonicalizes module names:

- while loading jobs, follow-ups, dedupe keys, and tombstones;
- while enqueueing, deduplicating, cancelling, and running a job.

This migration is bounded to durable module identifiers. It does not register
the old names in `sys.modules` and therefore does not restore the deleted
Python import API.

## Verification

- all 240 submodules imported in forward and reverse order;
- all 49 console entry points resolved to callable targets;
- Import Linter analyzed 241 files and 1,314 dependencies, with one contract
  kept and none broken;
- module-layout, durable-background-job, hook, inventory, and Ollama
  regressions: 89 passed;
- post-`l` test partition: 984 passed and 1 skipped;
- Ruff, compileall, and whitespace checks passed;
- lane contract, case manifest, structured-generation policy, and production
  schema hashes remained byte-identical to Campaign J.
