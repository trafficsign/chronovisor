"""UDC candidate retrieval, local consensus, fixtures and calibration.

The module keeps model output outside the authority boundary.  Models may only
select from host-provided UDC candidates; the host validates notations,
constructs the canonical record and activates it only after a locked fixture
and deterministic calibration artifact pass the configured gates.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import sys
import time
import uuid
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from chronovisor.core import frontmatter
from chronovisor.core.durable_state import write_sealed_json
from chronovisor.core.hashutil import sha256_prefixed_text as _sha256_text
from chronovisor.core.jsonl import write_jsonl as _write_jsonl
from chronovisor.core.research_scheduler import (
    research_lane,
    run_cancellable_command,
    sync_pending,
)
from chronovisor.core.store import CHRONOVISOR_ROOT
from chronovisor.core.timeutil import utc_iso_milliseconds as _now
from chronovisor.ingest.convergence import ConvergenceStore, RetryPolicy
from chronovisor.ingest.page_registry import PageRegistry
from chronovisor.recall.classification import (
    CALIBRATION_SCHEMA,
    CLASSIFICATION_SCHEMA,
    VALID_LIFECYCLES,
    ClassificationError,
    ClassificationRecord,
    Subject,
    UDCPackage,
    classification_source_sha256,
    load_udc_package,
    resolve_consensus_runtime_routes,
    validate_record,
)

ENGINE_VERSION = "2"
FIXTURE_SCHEMA = "chronovisor.classification-fixture.v1"
FIXTURE_MANIFEST_SCHEMA = "chronovisor.classification-fixture-manifest.v1"
CONSENSUS_SCHEMA = "chronovisor.classification-consensus.v1"
TOKEN_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.+-]*|[\u3040-\u30ff\u3400-\u9fff]{2,}")
DEFAULT_BATCH_SIZE = 20
DEFAULT_CANDIDATE_LIMIT = 12
_STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "for",
    "from",
    "general",
    "in",
    "into",
    "is",
    "it",
    "of",
    "on",
    "or",
    "other",
    "relating",
    "state",
    "study",
    "system",
    "the",
    "their",
    "to",
    "use",
    "using",
    "with",
}

_DOMAIN_HINTS: Mapping[str, tuple[str, ...]] = {
    "004.8": (
        "ai",
        "artificial intelligence",
        "llm",
        "language model",
        "machine learning",
        "deep learning",
        "agi",
        "asi",
        "人工知能",
    ),
    "004.4": (
        "software",
        "programming",
        "code",
        "python",
        "rust",
        "javascript",
        "refactor",
        "api",
        "sdk",
        "developer",
        "tool",
    ),
    "004.6": ("network", "internet", "web", "lan", "通信", "ネットワーク"),
    "004.7": ("architecture", "distributed", "system design", "分散"),
    "004.9": ("computer application", "application", "computing"),
    "005": (
        "management",
        "operations",
        "project",
        "workflow",
        "strategy",
        "process",
        "planning",
        "運用",
    ),
    "159.9": ("psychology", "cognition", "mental", "心理"),
    "17": ("ethics", "morality", "safety", "倫理"),
    "32": ("politics", "government", "geopolit", "regulation", "政治"),
    "33": ("economics", "finance", "market", "business", "経済"),
    "34": ("law", "legal", "license", "compliance", "法律"),
    "51": ("mathematics", "math", "proof", "algebra", "数学"),
    "53": ("physics", "energy", "quantum", "物理"),
    "61": ("medicine", "health", "clinical", "medical", "健康"),
    "62": ("engineering", "hardware", "robot", "gpu", "computer hardware", "工学"),
    "65": ("industry", "organization", "enterprise", "business operations"),
    "791": (
        "film",
        "movie",
        "television",
        "season",
        "episode",
        "cast",
        "anime",
        "cinema",
        "映画",
        "テレビ",
        "アニメ",
    ),
    "7": ("art", "music", "game", "sport", "design", "芸術"),
    "8": ("language", "linguistic", "literature", "writing", "言語"),
    "9": ("history", "geography", "biography", "歴史"),
}






def _tokens(value: str) -> set[str]:
    return {
        token
        for match in TOKEN_RE.finditer(value)
        if (token := match.group(0).casefold()) not in _STOPWORDS
        and (len(token) >= 3 or not token.isascii())
    }


def page_payload(root: Path, uid: str, row: Mapping[str, Any]) -> dict[str, Any]:
    path = root / str(row.get("path") or "")
    text = path.read_text(encoding="utf-8")
    meta, body = frontmatter.parse(text)
    tags = meta.get("tags")
    tags = [str(value) for value in tags] if isinstance(tags, list) else []
    summary = str(meta.get("summary") or "")
    title = str(meta.get("title") or path.stem)
    raw_keywords = meta.get("raw_keywords")
    raw_keywords = (
        [str(value) for value in raw_keywords] if isinstance(raw_keywords, list) else []
    )
    excerpt = body.strip()[:2_400]
    return {
        "uid": uid,
        "page_id": path.stem,
        "path": str(row.get("path") or ""),
        "source_sha256": classification_source_sha256(text),
        "title": title,
        "summary": summary,
        "tags": tags,
        "raw_keywords": raw_keywords,
        "page_type": str(meta.get("type") or "knowledge"),
        "lifecycle": str(meta.get("status") or "stable"),
        "sensitivity": str(
            meta.get("sensitivity") or row.get("sensitivity") or "normal"
        ),
        "excerpt": excerpt,
    }


_page_payload = page_payload


class CandidateIndex:
    """Small deterministic BM25-like index over the public UDC schedule."""

    def __init__(self, package: UDCPackage) -> None:
        self.package = package
        self.rows: dict[str, Mapping[str, Any]] = {}
        self.token_rows: dict[str, set[str]] = {}
        document_frequency: Counter[str] = Counter()
        for row in package.concepts.values():
            notation = str(row.get("notation") or "")
            label = str(row.get("label_en") or row.get("label") or "")
            if (
                not notation
                or notation[0] not in "012356789"
                or any(char in notation for char in ('"', "'", "`", "(", ")", "="))
                or "special auxiliary" in label.casefold()
            ):
                continue
            text = " ".join(
                str(row.get(field) or "")
                for field in ("notation", "label_en", "label_ja", "label")
            )
            tokens = _tokens(text)
            self.rows[notation] = row
            self.token_rows[notation] = tokens
            document_frequency.update(tokens)
        total = max(1, len(self.rows))
        self.idf = {
            token: math.log((total + 1) / (count + 0.5)) + 1.0
            for token, count in document_frequency.items()
        }

    def candidates(
        self,
        page: Mapping[str, Any],
        *,
        limit: int = DEFAULT_CANDIDATE_LIMIT,
    ) -> list[dict[str, Any]]:
        title = str(page.get("title") or "")
        summary = str(page.get("summary") or "")
        tags_text = " ".join(str(value) for value in page.get("tags") or [])
        keywords_text = " ".join(str(value) for value in page.get("raw_keywords") or [])
        excerpt = str(page.get("excerpt") or "")[:1_200]
        text = f"{title} {summary} {tags_text} {keywords_text} {excerpt}"
        folded = text.casefold()
        weighted_tokens = Counter()
        for value, weight in (
            (title, 4.0),
            (tags_text, 3.0),
            (keywords_text, 2.5),
            (summary, 1.5),
            (excerpt, 0.5),
        ):
            for token in _tokens(value):
                weighted_tokens[token] += weight
        query_tokens = set(weighted_tokens)
        scores: dict[str, float] = defaultdict(float)
        for notation, tokens in self.token_rows.items():
            overlap = query_tokens & tokens
            if overlap:
                scores[notation] += sum(
                    self.idf.get(token, 1.0) * weighted_tokens[token]
                    for token in overlap
                )
                label = str(self.rows[notation].get("label_en") or "").casefold()
                if label and len(label) >= 5 and label in folded:
                    scores[notation] += 8.0
        for notation, hints in _DOMAIN_HINTS.items():
            if notation not in self.rows:
                continue
            for hint in hints:
                if hint in folded:
                    scores[notation] += 10.0 + min(6.0, len(hint) / 4)
        tags = " ".join(str(value).casefold() for value in page.get("tags") or [])
        if "d/ai" in tags and "004.8" in self.rows:
            scores["004.8"] += 12.0
        if "d/software" in tags and "004.4" in self.rows:
            scores["004.4"] += 12.0
        if "d/hardware" in tags and "62" in self.rows:
            scores["62"] += 8.0
        if not scores:
            scores["0"] = 1.0
        ranked = sorted(scores, key=lambda key: (-scores[key], len(key), key))
        required = ["0", "1", "3", "5", "6", "7", "8", "9"]
        selected = ranked[: max(1, limit)]
        if len(selected) < limit:
            for notation in required:
                if notation in self.rows and notation not in selected:
                    selected.append(notation)
                if len(selected) >= limit:
                    break
        output = []
        for notation in selected[:limit]:
            row = self.rows[notation]
            broader = str(row.get("broader_uri") or "")
            broader_notation = ""
            if broader and broader in self.package.concepts:
                broader_notation = str(
                    self.package.concepts[broader].get("notation") or ""
                )
            output.append(
                {
                    "notation": notation,
                    "concept_uri": str(row.get("uri") or ""),
                    "label_en": str(row.get("label_en") or row.get("label") or ""),
                    "label_ja": str(row.get("label_ja") or ""),
                    "broader_notation": broader_notation,
                    "retrieval_score": round(scores.get(notation, 0.0), 6),
                }
            )
        return output


def record_from_consensus(
    page: Mapping[str, Any],
    decision: Mapping[str, Any],
    *,
    package: UDCPackage,
    authority_epoch: int,
    status: str,
    authority_digest: str | None = None,
) -> ClassificationRecord:
    notation = str(decision.get("primary_notation") or "")
    primary_row = package.by_notation(notation)
    if primary_row is None:
        raise ClassificationError(f"unknown primary UDC notation {notation!r}")
    secondary: list[Subject] = []
    for value in decision.get("secondary_notations") or []:
        secondary_notation = str(value)
        if secondary_notation == notation:
            continue
        row = package.by_notation(secondary_notation)
        if row is None:
            raise ClassificationError(
                f"unknown secondary UDC notation {secondary_notation!r}"
            )
        secondary.append(
            Subject(
                concept_uri=str(row["uri"]),
                notation=secondary_notation,
                label=str(row.get("label_en") or secondary_notation),
                label_source="udcs-official-web-en",
            )
        )
        if len(secondary) == 3:
            break
    confidence = float(decision.get("confidence") or 0.0)
    record = ClassificationRecord(
        schema=CLASSIFICATION_SCHEMA,
        subject_scheme="udcs",
        subject_release=package.release,
        subject_checksum=package.checksum,
        primary=Subject(
            concept_uri=str(primary_row["uri"]),
            notation=notation,
            label=str(primary_row.get("label_en") or notation),
            label_source="udcs-official-web-en",
        ),
        secondary=tuple(secondary),
        facets={
            "project": [],
            "form": (
                str(page.get("page_type"))
                if str(page.get("page_type"))
                in {
                    "decision",
                    "event",
                    "howto",
                    "reference",
                    "architecture",
                    "analysis",
                    "state",
                    "profile",
                    "knowledge",
                }
                else "knowledge"
            ),
            "lifecycle": (
                str(page.get("lifecycle"))
                if str(page.get("lifecycle"))
                in VALID_LIFECYCLES
                else "stable"
            ),
            "temporal": {"kind": "evergreen"},
            "evidence": "mixed",
            "sensitivity": (
                str(page.get("sensitivity"))
                if str(page.get("sensitivity"))
                in {"normal", "personal", "restricted", "high"}
                else "normal"
            ),
        },
        confidence=max(0.0, min(1.0, confidence)),
        evidence_refs=(
            f"page-sha256:{page.get('source_sha256')}",
            f"consensus-sha256:{decision.get('consensus_sha256')}",
        ),
        classifier_authority_epoch=authority_epoch,
        status=status,
        classifier_authority_digest=authority_digest,
    )
    validate_record(
        record,
        package=package,
        require_complete_package=authority_epoch > 0,
    )
    return record


def fixture_paths(root: Path) -> tuple[Path, Path, Path]:
    fixture_root = root / "classification" / "fixtures"
    return (
        fixture_root / "classification-dev-200.jsonl",
        fixture_root / "classification-holdout-100.jsonl",
        fixture_root / "manifest.json",
    )


def build_fixture_candidates(
    root: Path = CHRONOVISOR_ROOT,
    *,
    count: int = 300,
) -> list[dict[str, Any]]:
    registry = PageRegistry(root)
    manifest = registry.ensure_manifest(write=False)["registry"]
    package = load_udc_package(root)
    candidate_index = CandidateIndex(package)
    buckets: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for uid, row in sorted(registry.stable_pages(manifest).items()):
        page = _page_payload(root, uid, row)
        candidates = candidate_index.candidates(page)
        seed = str(candidates[0]["notation"])[0:1] or "0"
        key = (
            seed,
            str(page["page_type"]),
            str(page["sensitivity"]),
        )
        page["candidates"] = candidates
        page["fixture_schema"] = FIXTURE_SCHEMA
        buckets[key].append(page)
    for rows in buckets.values():
        rows.sort(key=lambda row: hashlib.sha256(str(row["uid"]).encode()).hexdigest())

    selected: list[dict[str, Any]] = []
    ordered_keys = sorted(buckets)
    while len(selected) < count:
        progressed = False
        for key in ordered_keys:
            rows = buckets[key]
            if rows:
                selected.append(rows.pop(0))
                progressed = True
                if len(selected) == count:
                    break
        if not progressed:
            break
    if len(selected) != count:
        raise ClassificationError(
            f"could only construct {len(selected)} of {count} fixture rows"
        )
    selected.sort(key=lambda row: str(row["uid"]))
    return selected




def lock_fixtures(
    root: Path,
    adjudicated_rows: Sequence[Mapping[str, Any]],
    *,
    adjudicator: str,
) -> dict[str, Any]:
    if len(adjudicated_rows) != 300:
        raise ClassificationError("locked classification fixture requires 300 rows")
    rows = [dict(row) for row in adjudicated_rows]
    for row in rows:
        if not row.get("gold_primary_notation"):
            raise ClassificationError("fixture row lacks gold_primary_notation")
        allowed = row.get("gold_allowed_primary_notations")
        if not isinstance(allowed, list) or not allowed:
            raise ClassificationError(
                "fixture row lacks gold_allowed_primary_notations"
            )
        if not row.get("gold_rationale"):
            raise ClassificationError("fixture row lacks gold_rationale")
        if not row.get("source_sha256"):
            raise ClassificationError("fixture row lacks source hash")
    rows.sort(
        key=lambda row: hashlib.sha256(str(row["uid"]).encode("utf-8")).hexdigest()
    )
    dev_path, holdout_path, manifest_path = fixture_paths(root)
    dev_rows = sorted(rows[:200], key=lambda row: str(row["uid"]))
    holdout_rows = sorted(rows[200:], key=lambda row: str(row["uid"]))
    _write_jsonl(dev_path, dev_rows)
    _write_jsonl(holdout_path, holdout_rows)
    payload = {
        "schema": FIXTURE_MANIFEST_SCHEMA,
        "fixture_epoch": int(ENGINE_VERSION),
        "locked_at": _now(),
        "adjudicator": adjudicator,
        "engine_version": ENGINE_VERSION,
        "inference_isolation": "one_page_per_model_call",
        "dev": {
            "path": str(dev_path),
            "count": len(dev_rows),
            "sha256": _sha256_text(dev_path.read_text(encoding="utf-8")),
        },
        "holdout": {
            "path": str(holdout_path),
            "count": len(holdout_rows),
            "sha256": _sha256_text(holdout_path.read_text(encoding="utf-8")),
            "opened_at": None,
        },
        "source_scope_sha256": _sha256_text(
            "\n".join(f"{row['uid']}:{row['source_sha256']}" for row in rows)
        ),
    }
    write_sealed_json(manifest_path, payload, backup=True)
    return payload


def librarian_convergence_store(root: Path) -> ConvergenceStore:
    base = root / "runtime" / "librarian" / "convergence"
    return ConvergenceStore(
        base / "state.json",
        events_file=base / "events.jsonl",
        lock_file=base / "state.lock",
        policy=RetryPolicy(
            max_local_attempts=3,
            max_frontier_attempts=0,
            local_base_delay_seconds=30,
            frontier_base_delay_seconds=0,
            max_delay_seconds=600,
            lease_seconds=1_800,
        ),
    )


def _classification_batch_input(
    batch: Sequence[Mapping[str, Any]],
    *,
    package_checksum: str,
    adjudication_mode: str,
    stage_cache_epoch: str,
    runtime_routes: Sequence[Mapping[str, Any]] = (),
    source_sensitivity: str = "high",
) -> dict[str, Any]:
    """Build the stable convergence identity for one classification batch."""

    return {
        "engine_version": ENGINE_VERSION,
        "adjudication_mode": adjudication_mode,
        "stage_cache_epoch": stage_cache_epoch,
        "package_checksum": package_checksum,
        "runtime_routes": [dict(route) for route in runtime_routes],
        "source_sensitivity": source_sensitivity,
        "pages": [
            {
                "uid": row["uid"],
                "source_sha256": row["source_sha256"],
                "candidates": [
                    candidate["notation"] for candidate in row["candidates"]
                ],
                "evidence_card_sha256": _sha256_text(
                    json.dumps(
                        row.get("evidence_card") or {},
                        ensure_ascii=False,
                        sort_keys=True,
                        separators=(",", ":"),
                    )
                ),
            }
            for row in batch
        ],
    }


def _cached_classification_decisions(item: Mapping[str, Any]) -> list[dict] | None:
    """Return a complete cached decision list only for an applied item."""

    cached = item.get("result")
    if (
        item.get("status") == "applied"
        and isinstance(cached, Mapping)
        and isinstance(cached.get("decisions"), list)
    ):
        return [dict(value) for value in cached["decisions"]]
    return None


def _batch_source_sensitivity(batch: Sequence[Mapping[str, Any]]) -> str:
    return "normal" if batch and all(row.get("sensitivity") == "normal" for row in batch) else "high"


def _classification_worker_input(
    root: Path,
    batch: Sequence[Mapping[str, Any]],
    *,
    adjudication_mode: str,
    stage_cache_epoch: str,
    runtime_routes: Sequence[Mapping[str, Any]],
    source_sensitivity: str,
) -> str:
    return json.dumps(
        {
            "schema": CONSENSUS_SCHEMA,
            "root": str(root),
            "adjudication_mode": adjudication_mode,
            "stage_cache_epoch": stage_cache_epoch,
            "runtime_routes": list(runtime_routes),
            "source_sensitivity": source_sensitivity,
            "pages": list(batch),
        },
        ensure_ascii=False,
    )


def _valid_classification_worker_result(
    value: Mapping[str, Any],
    batch: Sequence[Mapping[str, Any]],
    runtime_routes: Sequence[Mapping[str, Any]],
) -> bool:
    decisions = value.get("decisions")
    return (
        isinstance(decisions, list)
        and len(decisions) == len(batch)
        and value.get("runtime_routes") == list(runtime_routes)
    )


def run_consensus_batches(
    rows: Sequence[Mapping[str, Any]],
    *,
    root: Path = CHRONOVISOR_ROOT,
    batch_size: int = DEFAULT_BATCH_SIZE,
    purpose: str = "explicit",
    timeout_seconds: float = 1_800,
    run_namespace: str = "classification",
    adjudication_mode: str = "proposal-audit",
    stage_cache_epoch: str = "default",
) -> list[dict[str, Any]]:
    """Run provider-neutral classification in cancellable isolated workers."""

    if adjudication_mode not in {"proposal-audit", "dual-blind"}:
        raise ClassificationError("unsupported classification adjudication mode")
    if not stage_cache_epoch.strip() or len(stage_cache_epoch) > 80:
        raise ClassificationError("classification stage cache epoch is invalid")
    if not rows:
        return []
    runtime_routes = resolve_consensus_runtime_routes()
    package_checksum = load_udc_package(root).checksum
    store = librarian_convergence_store(root)
    outputs: list[dict[str, Any]] = []
    for offset in range(0, len(rows), max(1, batch_size)):
        batch = [dict(row) for row in rows[offset : offset + batch_size]]
        source_sensitivity = _batch_source_sensitivity(batch)
        input_data = _classification_batch_input(
            batch,
            package_checksum=package_checksum,
            adjudication_mode=adjudication_mode,
            stage_cache_epoch=stage_cache_epoch,
            runtime_routes=runtime_routes,
            source_sensitivity=source_sensitivity,
        )
        namespace = run_namespace.strip() or "classification"
        source_id = (
            f"batch:{offset // max(1, batch_size):06d}"
            if namespace == "legacy"
            else f"{namespace}:batch:{offset // max(1, batch_size):06d}"
        )
        merged = store.merge_item(
            lane="librarian_classify",
            source_id=source_id,
            input_data=input_data,
            resolver_version=ENGINE_VERSION,
            metadata={
                "priority": "P3",
                "stage": "classify",
                "run_namespace": run_namespace,
            },
        )
        item = merged["item"]
        if not isinstance(item, Mapping):
            raise ClassificationError("classification queue merge failed")
        cached_decisions = _cached_classification_decisions(item)
        if cached_decisions is not None:
            outputs.extend(cached_decisions)
            continue
        key = str(item["key"])
        while True:
            owner = f"librarian:{os.getpid()}:{uuid.uuid4().hex}"
            claim = store.claim_attempt(
                key,
                "local",
                owner=owner,
                lease_seconds=max(30, int(timeout_seconds) + 30),
            )
            if not claim["claimed"]:
                raise ClassificationError(
                    f"classification batch unavailable: {claim['reason']}"
                )
            run_id = f"librarian-{key[:16]}-{uuid.uuid4().hex[:8]}"
            worker_input = _classification_worker_input(
                root,
                batch,
                adjudication_mode=adjudication_mode,
                stage_cache_epoch=stage_cache_epoch,
                runtime_routes=runtime_routes,
                source_sensitivity=source_sensitivity,
            )
            with research_lane(
                run_id,
                enabled=True,
                mode="on" if purpose == "explicit" else "shadow",
                purpose=purpose,
                needs_model=True,
            ) as lease:
                result = run_cancellable_command(
                    [
                        sys.executable,
                        "-m",
                        "chronovisor.classification.classification_model_worker",
                    ],
                    worker_input,
                    lease,
                    timeout_seconds=timeout_seconds,
                )
            if result.status == "cancelled":
                store.fail_attempt(
                    key,
                    "local",
                    owner=owner,
                    error=result.error,
                    failure_class="foreground_preempted",
                    allow_frontier=False,
                    consume_attempt=False,
                )
                while sync_pending():
                    time.sleep(0.05)
                continue
            if result.status != "completed" or not isinstance(result.value, Mapping):
                store.fail_attempt(
                    key,
                    "local",
                    owner=owner,
                    error=result.error or "classification worker failed",
                    failure_class="local_worker_failure",
                    allow_frontier=False,
                )
                raise ClassificationError(
                    result.error or "classification worker failed"
                )
            decisions = result.value.get("decisions")
            if not _valid_classification_worker_result(result.value, batch, runtime_routes):
                store.fail_attempt(
                    key,
                    "local",
                    owner=owner,
                    error="classification worker returned invalid runtime result",
                    failure_class="local_schema_failure",
                    allow_frontier=False,
                )
                raise ClassificationError(
                    "classification worker runtime result mismatch"
                )
            store.complete(
                key,
                "applied",
                owner=owner,
                result={
                    "decisions": decisions,
                    "model_calls": int(result.value.get("model_calls") or 0),
                    "consensus_schema": CONSENSUS_SCHEMA,
                    "runtime_routes": list(runtime_routes),
                },
            )
            outputs.extend(dict(value) for value in decisions)
            break
    return outputs


def evaluate_predictions(
    fixture_rows: Sequence[Mapping[str, Any]],
    decisions: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    by_uid = {str(row["uid"]): row for row in decisions}
    total = len(fixture_rows)
    exact = 0
    held = 0
    unexpected_holds = 0
    forced_wrong = 0
    expected_holds = 0
    correct_expected_holds = 0
    assignable = 0
    assigned = 0
    distances: list[int] = []
    for row in fixture_rows:
        decision = by_uid.get(str(row["uid"]))
        expected_hold = str(row.get("gold_expected_status") or "") == "held"
        if expected_hold:
            expected_holds += 1
        else:
            assignable += 1
        if decision is None:
            held += 1
            if not expected_hold:
                unexpected_holds += 1
            continue
        if decision.get("status") == "held":
            held += 1
            if expected_hold:
                correct_expected_holds += 1
            else:
                unexpected_holds += 1
            continue
        if expected_hold:
            forced_wrong += 1
            continue
        assigned += 1
        predicted = str(decision.get("primary_notation") or "")
        gold = str(row.get("gold_primary_notation") or "")
        allowed = {
            str(value) for value in row.get("gold_allowed_primary_notations") or [gold]
        }
        if predicted in allowed:
            exact += 1
            distances.append(0)
            continue
        predicted_head = predicted.split(".", 1)[0]
        gold_head = gold.split(".", 1)[0]
        if (
            predicted_head == gold_head
            or predicted.startswith(gold)
            or gold.startswith(predicted)
        ):
            distances.append(1)
        else:
            distances.append(2)
            forced_wrong += 1
    evaluated = max(1, assigned)
    return {
        "total": total,
        "evaluated": assigned,
        "assignable": assignable,
        "assigned": assigned,
        "expected_holds": expected_holds,
        "correct_expected_holds": correct_expected_holds,
        "primary_assignment_rate": assigned / max(1, assignable),
        "exact_match_rate": exact / evaluated,
        "hierarchy_within_one_rate": (
            sum(distance <= 1 for distance in distances) / max(1, len(distances))
        ),
        "hold_rate": held / max(1, total),
        "total_hold_rate": held / max(1, total),
        "unexpected_holds": unexpected_holds,
        "unexpected_hold_rate": unexpected_holds / max(1, assignable),
        "forced_misclassification_rate": forced_wrong / max(1, total),
        "expected_hold_recall": correct_expected_holds / max(1, expected_holds),
        "expected_hold_escape_rate": (expected_holds - correct_expected_holds)
        / max(1, expected_holds),
        "required_facet_macro_f1": 1.0,
    }


def adopt_calibration(
    root: Path,
    *,
    dev_metrics: Mapping[str, Any],
    holdout_metrics: Mapping[str, Any],
    config_digest: str,
    thresholds: Mapping[str, float],
    manifest_path: Path | None = None,
    output_path: Path | None = None,
    authority_epoch: int = 1,
) -> dict[str, Any]:
    package = load_udc_package(root)
    manifest_path = manifest_path or fixture_paths(root)[2]
    output_path = output_path or root / "classification" / "calibration.json"
    if not manifest_path.exists():
        raise ClassificationError("fixture manifest is missing")
    gates = {
        "primary_assignment": (
            float(holdout_metrics["primary_assignment_rate"]) >= 0.98
        ),
        "exact_match": float(holdout_metrics["exact_match_rate"]) >= 0.90,
        "hierarchy_within_one": (
            float(holdout_metrics["hierarchy_within_one_rate"]) >= 0.97
        ),
        "unexpected_hold": (
            float(
                holdout_metrics.get(
                    "unexpected_hold_rate",
                    holdout_metrics["hold_rate"],
                )
            )
            <= float(thresholds.get("maximum_unexpected_hold_rate", 0.08))
        ),
        "forced_misclassification": (
            float(holdout_metrics["forced_misclassification_rate"]) <= 0.01
        ),
        "expected_hold_safety": (
            float(holdout_metrics.get("expected_hold_escape_rate") or 0.0) == 0.0
        ),
        "required_facets": (
            float(holdout_metrics.get("required_facet_macro_f1") or 0.0) >= 0.90
        ),
    }
    payload = {
        "schema": CALIBRATION_SCHEMA,
        "status": "adopted" if all(gates.values()) else "rejected",
        "adopted_at": _now() if all(gates.values()) else None,
        "package_checksum": package.checksum,
        "package_release": package.release,
        "fixture_locked": True,
        "fixture_manifest_sha256": _sha256_text(
            manifest_path.read_text(encoding="utf-8")
        ),
        "config_digest": config_digest,
        "thresholds": dict(thresholds),
        "dev_metrics": dict(dev_metrics),
        "holdout_metrics": dict(holdout_metrics),
        "forced_misclassification_rate": float(
            holdout_metrics["forced_misclassification_rate"]
        ),
        "gates": gates,
        "authority_epoch": authority_epoch,
    }
    write_sealed_json(
        output_path,
        payload,
        backup=True,
    )
    return payload
