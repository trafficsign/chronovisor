# Evidence storage / Recall 運用手順（P6）

この文書は、原文を保持したまま evidence projection と semantic generation を安全に作り直すための runbook である。コードや本番状態を変更した証跡ではない。下のコマンドは「実行例」と明記したものを除き、この文書作成時点では未実行である。

親計画: [Evidence storage / Recall 実装計画](../../2026-09-08_2205_chronovisor-evidence-storage-recall.html)

## 現在の状態

既に確認できる実行済みの結果は receipt を正本にする。

- [P1 validation](p1-validation.json): offline source-integrity、対象 446 tests、production deployment なし。
- [P2 validation](p2-validation.json): `page-evidence-projection.v1` と隔離 `section-v1`、projection/adapter の原文復元検証。全件 backfill・production activation・C の品質採用は未完了。
- [P2 frozen projection](p2-frozen-projection-integrity.json): 凍結した原文の body/span SHA 検証は失敗 0。これは品質 benchmark や本番 backfill 完了の証明ではない。
- [P3 validation](p3-validation.json): active generation の extractor 追従、更新/rebuild/rollback の接続と CAS 回帰を確認。既定 extractor は 2、候補 C は 3、production 変更はなし。
- [P5 cutover](p5-runtime-cutover.json): push、対象サービス再起動、archive `direct_url` provenance、health を確認。ただし full production acceptance と live processing canary は未完了。

現在は C を採用せず、production の extractor 2 を維持する。`query_timeout_ms=800` と台帳重複読込修正を d98adcd に配布。未使用 query の実 hook 出力で source reference 2 件の byte/SHA 一致、プロセス起動から実 stdout まで 3.749 秒を確認した。これは欠落修正の限定 canary であり、別言い換えの degraded / lexical-only 結果も保持する。全体品質と cold/warm paired p95 非劣化は未測定で、C の採用条件には代用しない。

## 1. 変更しない契約と新規保存

`Raw` と canonical `Page` が原資料であり、projection・CAS・semantic index は再生成できる派生物として扱う。Raw は [`publish_raw`](../../../src/chronovisor/raw/record_raw.py) / `publish_raw_idempotent`、Page は [`apply_page_writes`](../../../src/chronovisor/ingest/page_write.py) の既存経路を使う。projection や索引の作業で `raw/`、`pages/`、`system/` を削除・上書きしない。

新しい保存は次の順序にする。

1. 既存の Raw/Page writer で canonical source を確定する。
2. 同期更新が必要な場合は `chronovisor.core.search.update_embeddings(page_ids, strict=True)`、非同期なら同じ関数の `strict=False` を使う。永続化側の worker は [`SemanticServiceState._index_page`](../../../src/chronovisor/search/semantic_service.py) で active manifest の `extractor_schema_version` を読み、同じ版で抽出する。`strict=False` 内の `extract_page_documents` は queue 用の digest 算出だけで、保存する文書を別版で確定する入口ではない。
3. active generation がない場合は `index_pages` を先に呼ばず、rebuild で generation を作る。service API の `index_pages(..., wait=True)` は active generation がないと `full semantic rebuild is required first` で停止する。

active extractor は pointer と manifest から毎回確認する（読み取り専用の実行例）。

```sh
ROOT="${CHRONOVISOR_ROOT:-$HOME/.chronovisor}"
SEMANTIC_ROOT="$ROOT/.index/semantic"
ACTIVE_ID="$(/usr/bin/jq -r '.generation_id // empty' "$SEMANTIC_ROOT/active.json")"
test -n "$ACTIVE_ID"
/usr/bin/jq '{generation_id, extractor_schema_version, repo_commit, corpus_fingerprint, metadata_sha256, vectors_sha256, ann_sha256}' \
  "$SEMANTIC_ROOT/generations/$ACTIVE_ID/manifest.json"
```

`extractor_schema_version=2` は page/question/chunk の既存経路、3 は section-v1 候補である。同一 generation に 2 と 3 を混在させない。C の採用 gate が不成立なら 2 のまま新規保存と rebuild を続け、既存の Raw/Page と v2 generation を保持する。

## 2. 既存資料を隔離して検証・generation build

本番 root を直接使わず、承認済みの凍結 snapshot を一時 root にコピーする。P0 snapshot の `source-manifest.json` と `root/` を使う例は次のとおり（未実行）。

```sh
SNAPSHOT_BASE="$HOME/.chronovisor/runtime/evidence-storage-evaluation/p0-frozen-20260909"
ISOLATED_ROOT="$(mktemp -d /private/tmp/cv-evidence.XXXXXX)"
/usr/bin/rsync -a -- "$SNAPSHOT_BASE/root/" "$ISOLATED_ROOT/"
SOCKET="$ISOLATED_ROOT/runtime/semantic.sock"
SOCKET="$SOCKET" .venv/bin/python - "$ISOLATED_ROOT/config.toml" <<'PY'
import json
import os
import re
import sys
import tomllib
from pathlib import Path

path = Path(sys.argv[1])
text = path.read_text(encoding="utf-8")
start = text.index("[search.embedding.service]")
end = text.find("\n[", start + 1)
section_end = len(text) if end < 0 else end
section, count = re.subn(
    r"(?m)^socket\s*=\s*.*$",
    "socket = " + json.dumps(os.environ["SOCKET"]),
    text[start:section_end],
    count=1,
)
if count != 1:
    raise SystemExit("[search.embedding.service].socket is missing")
path.write_text(text[:start] + section + text[section_end:], encoding="utf-8")
assert tomllib.loads(path.read_text())["search"]["embedding"]["service"]["socket"] == os.environ["SOCKET"]
PY
```

source digest と projection の read-back は [`project_page_evidence`](../../../src/chronovisor/core/page_evidence.py)、`load_page_evidence_artifact`、`reconstruct_page_body` を使う。次は snapshot の `pages/` と `system/` を走査する読み取り例で、source は変更しない。

```sh
CHRONOVISOR_ROOT="$ISOLATED_ROOT" PYTHONDONTWRITEBYTECODE=1 CHRONOVISOR_READ_ONLY=1 .venv/bin/python - <<'PY'
import hashlib
import os
from pathlib import Path

from chronovisor.core.canonical_document import parse_document
from chronovisor.core.page_evidence import (
    load_page_evidence_artifact,
    project_page_evidence,
    reconstruct_page_body,
)

root = Path(os.environ["CHRONOVISOR_ROOT"])
checked = 0
for namespace in ("pages", "system"):
    for path in sorted((root / namespace).rglob("*.md")):
        source = path.read_bytes()
        document = parse_document(source)
        if document.metadata.get("status") != "stable":
            continue
        page_id = path.stem
        projection = project_page_evidence(source, page_id)
        restored = load_page_evidence_artifact(
            projection.canonical_bytes(), source=source, page_id=page_id
        )
        assert hashlib.sha256(source).hexdigest() == restored.content_sha256
        assert reconstruct_page_body(source, restored) == document.body
        checked += 1
print({"validated_stable_pages": checked})
PY
```

generation は service の既存 rebuild 経路だけで作る。隔離 root 用 service を一つ起動し、別の shell から rebuild を要求する。
別の shell では `ISOLATED_ROOT` に同じ絶対パスを設定する。

```sh
# shell 1（未実行、終了時は Ctrl-C）
CHRONOVISOR_ROOT="$ISOLATED_ROOT" scripts/chronovisor-semantic-service serve

# shell 2（未実行）
CHRONOVISOR_ROOT="$ISOLATED_ROOT" scripts/chronovisor-semantic-service rebuild
```

この経路は `SemanticServiceState._rebuild` → `extract_all_documents(extractor_schema_version=active版)` → `build_generation` → `activate_generation(expected_current=...)` を通る。低レベル API を使う検証では [`validate_generation`](../../../src/chronovisor/core/semantic_index.py) を実行し、`COMPLETE`、manifest の extractor/model/revision、metadata/vectors/ANN SHA、行数を確認する。

```sh
CHRONOVISOR_ROOT="$ISOLATED_ROOT" .venv/bin/python - <<'PY'
from chronovisor.core.semantic_index import SEMANTIC_ROOT, read_active, validate_generation

active = read_active(root=SEMANTIC_ROOT)
generation_id = str(active.get("generation_id") or "")
assert generation_id
manifest = validate_generation(generation_id, root=SEMANTIC_ROOT)
print({"generation_id": generation_id, "extractor_schema_version": manifest.extractor_schema_version})
PY
```

既存 P2 の current-source 再検証を再実行する場合は、次の private validator を参照する。ただしこの script は inventory と production root を固定しているため、隔離 build の代わりに実行しない。既存 receipt では `source_writes=0`、`index_writes=0`、`projection_cas_writes=0` である。

```sh
uv run python "$HOME/.chronovisor/runtime/evidence-storage-evaluation/revalidate_page_projection.py"
```

## 3. projection の CAS publish と checkpoint

page-only projection の保存入口は [`store_page_evidence_projection`](../../../src/chronovisor/research/page_evidence_projection.py) である。`ResearchStore.put_artifact` は canonical bytes の SHA-256 を `sha256:<digest>` として CAS に書き、`read_artifact` で zstd 展開後の digest を再検証する。durable receipt が必要なページは `durable=True` を指定し、完了後に `checkpoint_page_evidence_projection` で artifact ID、projection ID、source digest、transform version を記録する。

```python
from chronovisor.research.page_evidence_projection import (
    checkpoint_page_evidence_projection,
    store_page_evidence_projection,
)
from chronovisor.search.research_store import ResearchStore

store = ResearchStore()
artifact = store_page_evidence_projection(store, source, page_id, durable=True)
assert store.read_artifact(artifact.artifact_id) == projection.canonical_bytes()
checkpoint_page_evidence_projection(
    store, session_id, artifact, active=False, durable_receipt=True
)
```

上の `source`、`page_id`、`projection` は同一ページについて `project_page_evidence` で作った値を渡す。既存 P2 receipt の `projection_cas_writes=0` は「この backfill を実行済み」という意味ではないため、全件処理時は manifest ごとに `validated / published / checkpointed / failed / ineligible` を集計する。

semantic generation の publish は `build_generation` が staging、manifest、`COMPLETE`、fsync、atomic rename の順で封印し、`activate_generation(..., expected_current=...)` が active pointer を CAS 更新する。active generation のディレクトリを直接編集しない。

## 4. rollback と C 不採用時の扱い

隔離 root で rollback を演習した後、想定する不良 candidate の ID を固定して既存 CAS API を使う（未実行例）。

```sh
CHRONOVISOR_ROOT="$HOME/.chronovisor" .venv/bin/python - '<failed-candidate-generation-id>' <<'PYTHON'
import sys
from chronovisor.core.semantic_index import activate_generation, read_active
candidate = sys.argv[1]
active = read_active()
assert active.get("generation_id") == candidate, "another generation is active"
previous = str(active.get("previous_generation_id") or "")
assert previous, "rollback generation is missing"
print(activate_generation(previous, expected_current=candidate))
PYTHON
```

[`activate_generation`](../../../src/chronovisor/core/semantic_index.py) は復元先を検証し、切替直前にも `expected_current` を照合する。別publisherが先に切り替えた場合は失敗し、そのpointerを維持する。serviceは次の検索でpointer変更を読み直すため、canary後にgeneration一致を確認する。既存CLIの `rollback` は呼出時のcurrentを戻す操作で、想定candidateを引数で固定できないため、この競合を含む手順では上記APIを使う。

```sh
ROOT="$HOME/.chronovisor"
SEMANTIC_ROOT="$ROOT/.index/semantic"
/usr/bin/jq '{generation_id, previous_generation_id, manifest_sha256}' "$SEMANTIC_ROOT/active.json"
CHRONOVISOR_ROOT="$HOME/.chronovisor" scripts/chronovisor-semantic-service status
```

digest mismatch、mixed extractor、CAS conflict、復元先の `validate_generation` 失敗が一つでもあれば切り替えを受け入れず、旧 generation と Raw/Page を残して停止する。旧 generation を prune するのは rollback 保持期間を過ぎ、受入 receipt が揃った後だけにする。`archive-legacy` は legacy mutable embeddings DB を退避する最後の後処理で、fresh generation の coverage/status gate 後にだけ実行する。C 不採用を理由に v2 generation や Raw/Page を削除しない。

## 5. push 後の再起動と archive SHA 照合

本番操作は [release preflight](release-preflight-20260909.json) の順序を守る。

1. clean release worktree で対象 commit と quality gate を確認し、`HEAD`、`origin/main`、`git ls-remote origin refs/heads/main` を一致させる。
2. assembled release commit を push してから restart する。push 前に restart しない。
3. `$HOME/.chronovisor/runtime/status.json` が terminal/idle になり、`current_job_id`/`current_job_pid` が null、`ingest-orchestrator.lock` に holder がないことを確認する。Raw を force-kill しない。
4. exclusive ingest lease を保持して ingest-drain を切り替える。その後、次の managed-v2 label を kickstart する。converge/librarian-review は active run が自然終了してから対象にする。

```sh
/bin/launchctl kickstart -k "gui/$(id -u)/com.trafficsign.chronovisor-ingest-drain.managed-v2"
/bin/launchctl kickstart -k "gui/$(id -u)/com.trafficsign.chronovisor-dashboard.managed-v2"
/bin/launchctl kickstart -k "gui/$(id -u)/com.trafficsign.chronovisor-lan-dashboard.managed-v2"
/bin/launchctl kickstart -k "gui/$(id -u)/com.trafficsign.chronovisor-semantic.managed-v2"
/bin/launchctl kickstart -k "gui/$(id -u)/com.trafficsign.chronovisor-reranker.managed-v2"
```

5. 各 label の登録・PID を `launchctl print` で確認し、各実プロセスが読み込んだ archive の `direct_url.json` の `vcs_info.commit_id` と raw/payload SHA を記録する。distribution metadata の所在は、既存 runtime validator と同じ `importlib.metadata.distribution("chronovisor")` で求められる。下の `DIRECT_URL` は現在の実行環境から所在を求める補助例であり、service の PID/command が示す archive と一致させてから記録する。

```sh
for label in \
  com.trafficsign.chronovisor-dashboard.managed-v2 \
  com.trafficsign.chronovisor-lan-dashboard.managed-v2 \
  com.trafficsign.chronovisor-ingest-drain.managed-v2 \
  com.trafficsign.chronovisor-semantic.managed-v2 \
  com.trafficsign.chronovisor-reranker.managed-v2
do
  /bin/launchctl print "gui/$(id -u)/$label"
done

DIRECT_URL="$(.venv/bin/python -c 'from importlib import metadata; from pathlib import Path; print(Path(metadata.distribution("chronovisor")._path) / "direct_url.json")')"
test -f "$DIRECT_URL"
/usr/bin/jq '{url, vcs_info}' "$DIRECT_URL"
.venv/bin/python - "$DIRECT_URL" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
raw = path.read_bytes()
payload = json.loads(raw)
canonical = json.dumps(
    payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
).encode("utf-8")
print({
    "direct_url_raw_sha256": hashlib.sha256(raw).hexdigest(),
    "direct_url_payload_sha256": hashlib.sha256(canonical).hexdigest(),
    "commit_id": payload.get("vcs_info", {}).get("commit_id"),
})
PY
```

`commit_id` は push 済み SHA と、次で取得した remote SHA に一致させる。

```sh
EXPECTED_SHA="<pushed-release-commit>"
test "$EXPECTED_SHA" = "$(git ls-remote origin refs/heads/main | cut -f1)"
```

最後に semantic の status/health と task-specific live query を確認する。`scripts/chronovisor-semantic-service status` の `ready` は archive provenance、generation ID、source byte/SHA、4 秒 caller budget の受入を代替しない。最新の限定 canary は起動から実 stdout まで 3.749 秒、source 2 件一致。paired 品質・p95 と本番 ingest 実処理の確認は別の受入条件として記録する。

```sh
CHRONOVISOR_ROOT="$HOME/.chronovisor" scripts/chronovisor-semantic-service status
```

## 6. receipt と停止条件

各 run では次を同じ evidence directory に記録する。

- source manifest/root SHA、対象件数、stable/ineligible/revision-changed/failed 件数。
- active/candidate/previous generation ID、manifest SHA、extractor version、model/revision、CAS artifact ID。
- `validated → published → read-back → checkpointed` の状態と未処理残件。
- push SHA、remote SHA、各 archive の `direct_url` commit/raw SHA、`launchctl print` の PID、service health/ready、live query の wall/queue/注入 source SHA。
- rollback の前後 pointer と復元先 manifest SHA。失敗時は Raw/Page を保持したまま停止した理由。

次のいずれかで publish/restart/backfill を止める。まだ pointer を切り替えていない、または CAS conflict が出た場合は pointer を触らず停止する。他の publisher が切り替えていないことを確認でき、candidate が既に active になった後の不良だけを、`expected_current=candidate` 付き rollback の対象にする。

- source digest、UID、canonical byte range、span SHA、projection transform の不一致。
- source が検証中に改訂、status が stable でない、または path が曖昧。
- extractor 2/3 の混在、manifest/model/revision 不一致、generation CAS conflict（CAS conflict 時は他 publisher の pointer を戻さない）。
- CAS read-back/checkpoint 不一致、archive `direct_url` の commit/raw SHA 不一致、PID/launchd provenance 不一致。
- ingest が safe boundary に到達していない、Raw force-kill が必要になる、または live wall time が 4 秒契約を超える。

rollback を行う場合は、まず `active.json` の `generation_id` が想定 candidate と一致することを読み取り、上記 `activate_generation(previous, expected_current=candidate)` の CAS を通す。異なる場合は pointer を変更せず、別 publisher の結果を保持したまま調査する。

### この runbook の検証範囲

参照したコードの graph coverage は、`semantic_index.py`、`page_evidence.py`、`page_evidence_projection.py`、`page_section_semantic_adapter.py`、`research_store.py`、`semantic_service.py`、`search.py`、`semantic_client.py` すべて `no_recorded_issue / metadata_match` だった（knowledge graph の best-effort 信号であり、ソース完全性の証明ではない）。この文書自体のコマンド、隔離 rebuild、全件 CAS publish、追加 restart をこの手順例で実行したわけではない。実際の配布と 4 秒 canary は上記の個別 receipt を正本とする。
