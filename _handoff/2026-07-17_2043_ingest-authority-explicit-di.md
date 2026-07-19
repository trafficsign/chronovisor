---
task_id: ingest_authority_explicit_di_20260717
created_at: 2026-07-17T20:43:00+09:00
状態: complete
baseline_commit: 0f6dd7b
branch: codex/refactor-campaign-a-safety
---

# Ingest authority E0: explicit dependency injection

## Outcome

ingestのproduction authorityとtest/evaluation reviewer境界を、callableの`__module__`から完全に分離した。productionは常にadopted local-consensus authorityをpreflightし、`frontier_reviewer`が明示された呼び出しだけが`injected_reviewer_boundary`を使用する。

## Changes

- `_current_ingest_review_authority()`は`reviewer is not None`だけでinjected boundaryを選ぶ
- `run_ingest()`は`_review_and_apply_ingest_operations`の所属moduleに関係なくauthorityをpreflightする
- `run_pending_ingest()`にkeyword-onlyの`frontier_reviewer` seamを追加し、指定時だけ`run_ingest()`へ伝播する
- reviewer未指定時の既存`run_ingest()`呼び出し形は維持し、legacy test doubles / callersへ新しいkeywordを送らない
- foreign-module review adapterでもauthority preflightを迂回できない回帰testを追加
- process-global context admission cacheをFrontier test fixtureで初期化し、test order依存を除去

## Verified contracts

- `rg "__module__" src/chronovisor/ingest.py`: authority logicは0件
- invalid production authorityはtriage / generation / reviewより前にfail closed
- foreign-module callableはproduction authorityのまま
- explicit reviewerだけがinjected boundaryを得る
- shard continuationはorchestratorから明示reviewerを渡して継続できる
- reviewer未指定のorchestrator callは従来のcall signatureを保持する

## Tests

- authority / low-risk / correction targeted: pass
- explicit-DI run_ingest integration targeted: 12 passed
- `tests/test_ingest.py::TestOrchestrator`: 28 passed
- `tests/test_ingest.py`: 410 passed in 480.47s
- `tests/test_frontier_review.py`: 56 passed
- full repository: 2,345 passed in 882.22s
- `git diff --check`: pass

`ruff`はproject environmentに未導入のため実行不能だった。syntaxは`py_compile`、behaviorは上記pytestで検証した。

## Rollback

E0 commitをrevertする。persistent schema、authority manifest、adopted artifact、`~/.wiki` data、runtime configは変更していない。
