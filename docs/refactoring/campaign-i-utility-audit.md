# Campaign I utility consolidation audit

This audit records the byte- and precision-sensitive choices made during the
utility consolidation. Only helpers with equivalent input, output, timezone,
precision, serialization, permissions, and durability contracts were replaced.

## Consolidated groups

- `timeutil.py`
  - 19 current-UTC ISO timestamps with millisecond precision
  - 6 aware current-UTC `datetime` helpers
  - 2 current-UTC ISO timestamps with second precision
  - 5 direct `datetime.isoformat(timespec="seconds")` helpers
  - 3 naive-or-aware UTC normalization helpers
- `hashutil.py`
  - 11 unprefixed byte SHA-256 helpers
  - 4 unprefixed UTF-8 text SHA-256 helpers
  - 5 streaming 1 MiB file SHA-256 helpers
  - 3 legacy `sha256:`-prefixed byte/text helpers
- `jsonl.py` and `jsonl_write.py`
  - 2 non-atomic sorted JSONL writers
  - 2 private-mode atomic byte replacements
  - 2 private-mode atomic text replacements
  - 3 private-mode atomic sorted JSONL writers

All consolidated JSONL writers retain UTF-8, `ensure_ascii=False`,
`sort_keys=True`, one LF per record, and the prior empty-input behavior.
Atomic replacements retain same-directory temporary files, file `fsync`,
`os.replace`, and mode `0600`.

## Intentionally retained variants

- `ingest._now` remains local time with no explicit timezone.
- `autonomy._now` remains local-time ISO seconds and is a test seam.
- `classification_pilot._now` retains default ISO microsecond behavior.
- `raw_replay._now` retains the host local timezone.
- `research_scheduler._iso` and `semantic_index._utc_now` retain their
  zero-argument timestamp APIs.
- `background_jobs`, `convergence_drain`, `research_store`, and
  `semantic_jobs` retain `_iso(value=None)` wrappers because they intentionally
  compose with their module-local, patchable `_now` seams or use different
  precision/normalization.
- Atomic JSON writers remain separate: their indentation, canonical separators,
  `default=str`, directory `fsync`, sealing, and cleanup guarantees differ.
- JSONL readers remain separate when they count invalid rows, scan only a
  complete prefix, tail-limit records, or intentionally use different invalid
  record handling.
- JSONL appenders remain separate when they add file locking, `sort_keys`,
  trimming, or weaker/stronger durability guarantees.
- Canonical, authority, prefixed, identity-projection, and domain-sealed hashes
  remain in their owning modules. Equal SHA-256 syntax does not imply an equal
  semantic contract.
