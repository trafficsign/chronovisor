"""Read-only method pilot for hierarchical UDC classification.

The pilot intentionally stays outside the classification authority path.  It
compares candidate retrieval and decision combinations on a small, independently
reviewed diagnostic set without mutating pages, calibration, or rollout state.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import time
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

from chronovisor.core import embedding, hashutil, ollama, runtime_config
from chronovisor.recall.classification import (
    UDCPackage,
    load_udc_package,
)
from chronovisor.recall.classification_engine import CandidateIndex
from chronovisor.recall.classification_library_evidence import (
    embed_texts_cancellable,
)

DIAGNOSTIC_SCHEMA = "chronovisor.classification-diagnostic.v1"
PILOT_SCHEMA = "chronovisor.classification-method-pilot.v1"
PILOT_ENGINE_VERSION = 1
HOLD = "__HOLD__"
NONE = "__NONE__"
DEFAULT_SEMANTIC_LIMIT = 20
DEFAULT_TOTAL_LIMIT = 36
_sha256_bytes = hashutil.sha256_prefixed_bytes
load_decision_router_config = runtime_config.load_decision_router_config


def _now() -> str:
    from datetime import UTC, datetime

    return datetime.now(UTC).isoformat()




def _jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _atomic_write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(content, encoding="utf-8")
    temporary.replace(path)


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    _atomic_write(
        path,
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    )


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    content = "".join(
        json.dumps(dict(row), ensure_ascii=False, sort_keys=True) + "\n"
        for row in rows
    )
    _atomic_write(path, content)


def prepare_diagnostic_set(
    *,
    fixture_path: Path,
    baseline_results_path: Path,
    spec_path: Path,
    output_path: Path,
) -> dict[str, Any]:
    """Join frozen fixture rows, sealed baseline results, and reviewer labels."""

    fixtures = {str(row["page_id"]): row for row in _jsonl(fixture_path)}
    baseline = {str(row["uid"]): row for row in _jsonl(baseline_results_path)}
    spec = json.loads(spec_path.read_text(encoding="utf-8"))
    cases = spec.get("cases")
    if not isinstance(cases, list) or not cases:
        raise ValueError("diagnostic spec requires a non-empty cases list")
    if len(cases) != len({str(row.get("page_id") or "") for row in cases}):
        raise ValueError("diagnostic spec page_id values must be unique")

    output: list[dict[str, Any]] = []
    for case in cases:
        page_id = str(case.get("page_id") or "")
        fixture = fixtures.get(page_id)
        if fixture is None:
            raise ValueError(f"diagnostic page is absent from fixture: {page_id}")
        decision = baseline.get(str(fixture["uid"]))
        if decision is None:
            raise ValueError(f"baseline decision is absent for {page_id}")
        reference = case.get("reference")
        if not isinstance(reference, Mapping):
            raise ValueError(f"diagnostic reference is absent for {page_id}")
        expected = str(reference.get("expected_disposition") or "")
        if expected not in {"leaf", "ancestor", "hold"}:
            raise ValueError(f"invalid expected disposition for {page_id}: {expected}")
        primary = str(reference.get("primary_notation") or "")
        if expected != "hold" and not primary:
            raise ValueError(f"reference primary notation is absent for {page_id}")
        output.append(
            {
                **fixture,
                "diagnostic_schema": DIAGNOSTIC_SCHEMA,
                "diagnostic_bucket": str(case.get("bucket") or "unspecified"),
                "reference": dict(reference),
                "baseline": {
                    "status": str(decision.get("status") or "held"),
                    "primary_notation": str(
                        decision.get("primary_notation") or ""
                    ),
                    "quorum": int(decision.get("quorum") or 0),
                    "rationale": str(decision.get("rationale") or ""),
                    "consensus_sha256": str(
                        decision.get("consensus_sha256") or ""
                    ),
                },
            }
        )

    _write_jsonl(output_path, output)
    receipt = {
        "schema": DIAGNOSTIC_SCHEMA,
        "created_at": _now(),
        "fixture_path": str(fixture_path),
        "fixture_sha256": _sha256_bytes(fixture_path.read_bytes()),
        "baseline_results_path": str(baseline_results_path),
        "baseline_results_sha256": _sha256_bytes(
            baseline_results_path.read_bytes()
        ),
        "spec_path": str(spec_path),
        "spec_sha256": _sha256_bytes(spec_path.read_bytes()),
        "output_path": str(output_path),
        "output_sha256": _sha256_bytes(output_path.read_bytes()),
        "case_count": len(output),
        "page_ids": [str(row["page_id"]) for row in output],
        "authority_contract": (
            "Reference labels are an independent method-selection fixture, not "
            "classification authority or production calibration."
        ),
    }
    _write_json(output_path.with_suffix(".receipt.json"), receipt)
    return receipt


def _valid_notation(notation: str, label: str) -> bool:
    return bool(
        notation
        and notation[0] in "012356789"
        and not any(char in notation for char in ('"', "'", "`", "(", ")", "="))
        and "special auxiliary" not in label.casefold()
    )


class AuthoritativeCandidateIndex:
    """Semantic index over official UDC labels and their official lineage."""

    def __init__(
        self,
        package: UDCPackage,
        *,
        embed_many: Callable[[list[str]], list[list[float]]] | None = None,
        embedding_batch_size: int = 64,
    ) -> None:
        self.package = package
        self.lexical = CandidateIndex(package)
        self.by_notation = {
            str(row.get("notation") or ""): row
            for row in package.concepts.values()
            if _valid_notation(
                str(row.get("notation") or ""),
                str(row.get("label_en") or row.get("label") or ""),
            )
        }
        self._embed_many = embed_many
        self._embedding_batch_size = max(1, embedding_batch_size)
        self._notations = sorted(self.by_notation)
        self._card_texts = [
            self._retrieval_text(self.by_notation[notation])
            for notation in self._notations
        ]
        self._vectors: list[list[float]] | None = None

    def _path_rows(self, row: Mapping[str, Any]) -> list[Mapping[str, Any]]:
        output = [row]
        seen = {str(row.get("uri") or "")}
        current = row
        while current.get("broader_uri"):
            uri = str(current["broader_uri"])
            if uri in seen:
                break
            seen.add(uri)
            parent = self.package.concepts.get(uri)
            if parent is None:
                break
            output.append(parent)
            current = parent
        return output

    def notation_path(self, notation: str) -> list[str]:
        row = self.by_notation.get(notation)
        if row is None:
            return []
        return [
            str(value.get("notation") or "")
            for value in self._path_rows(row)
            if str(value.get("notation") or "") in self.by_notation
        ]

    def _retrieval_text(self, row: Mapping[str, Any]) -> str:
        parts = []
        for value in self._path_rows(row):
            notation = str(value.get("notation") or "")
            label_en = str(value.get("label_en") or value.get("label") or "")
            label_ja = str(value.get("label_ja") or "")
            parts.append(f"{notation} {label_en} {label_ja}".strip())
        return " | broader: ".join(parts)

    def card(self, notation: str) -> dict[str, Any] | None:
        row = self.by_notation.get(notation)
        if row is None:
            return None
        path = self._path_rows(row)
        return {
            "notation": notation,
            "label_en": str(row.get("label_en") or row.get("label") or ""),
            "label_ja": str(row.get("label_ja") or ""),
            "concept_uri": str(row.get("uri") or ""),
            "broader_notation": (
                str(path[1].get("notation") or "") if len(path) > 1 else ""
            ),
            "official_lineage": [
                {
                    "notation": str(value.get("notation") or ""),
                    "label_en": str(
                        value.get("label_en") or value.get("label") or ""
                    ),
                }
                for value in reversed(path)
            ],
        }

    def _embed_batched(
        self,
        texts: list[str],
        *,
        source_data_class: str,
        source_sensitivity: str,
        embedding_purpose: str,
    ) -> list[list[float]]:
        output: list[list[float]] = []
        for offset in range(0, len(texts), self._embedding_batch_size):
            batch = texts[offset : offset + self._embedding_batch_size]
            output.extend(
                self._embed_many(batch)
                if self._embed_many is not None
                else embed_texts_cancellable(
                    batch,
                    source_data_class=source_data_class,
                    source_sensitivity=source_sensitivity,
                    embedding_purpose=embedding_purpose,
                )[1]
            )
        if len(output) != len(texts):
            raise ValueError("embedding backend returned the wrong vector count")
        return output

    def _ensure_vectors(self) -> list[list[float]]:
        if self._vectors is None:
            self._vectors = self._embed_batched(
                self._card_texts,
                source_data_class="derived_snippet",
                source_sensitivity="normal",
                embedding_purpose="document",
            )
        return self._vectors

    @staticmethod
    def page_query(page: Mapping[str, Any]) -> str:
        """Use page content, never mined local label associations, as the query."""

        return "\n".join(
            value
            for value in (
                str(page.get("title") or ""),
                str(page.get("summary") or ""),
                str(page.get("excerpt") or "")[:2_400],
            )
            if value
        )

    def candidates(
        self,
        page: Mapping[str, Any],
        *,
        semantic_limit: int = DEFAULT_SEMANTIC_LIMIT,
        total_limit: int = DEFAULT_TOTAL_LIMIT,
    ) -> list[dict[str, Any]]:
        query_vector = self._embed_batched(
            [self.page_query(page)],
            source_data_class="page",
            source_sensitivity=(
                "normal" if page.get("sensitivity") == "normal" else "high"
            ),
            embedding_purpose="query",
        )[0]
        semantic_scores = [
            (notation, embedding.cosine(query_vector, vector))
            for notation, vector in zip(
                self._notations, self._ensure_vectors(), strict=True
            )
        ]
        semantic_scores.sort(key=lambda value: (-value[1], value[0]))
        semantic = semantic_scores[: max(1, semantic_limit)]
        semantic_by_notation = dict(semantic)

        lexical_rows = [
            dict(value) for value in page.get("candidates") or []
        ] or self.lexical.candidates(page)
        lexical_by_notation = {
            str(row.get("notation") or ""): row for row in lexical_rows
        }

        ordered: list[str] = []
        for notation, _score in semantic:
            if notation not in ordered:
                ordered.append(notation)
        for row in lexical_rows:
            notation = str(row.get("notation") or "")
            if notation in self.by_notation and notation not in ordered:
                ordered.append(notation)
        for notation, _score in semantic[:10]:
            for ancestor in self.notation_path(notation)[1:3]:
                if ancestor not in ordered:
                    ordered.append(ancestor)

        output = []
        for notation in ordered[: max(1, total_limit)]:
            card = self.card(notation)
            if card is None:
                continue
            sources = []
            if notation in semantic_by_notation:
                sources.append("official_label_semantic")
            if notation in lexical_by_notation:
                sources.append("legacy_lexical")
            output.append(
                {
                    **card,
                    "retrieval_sources": sources,
                    "semantic_score": round(
                        float(semantic_by_notation.get(notation, -1.0)), 6
                    ),
                    "legacy_score": round(
                        float(
                            lexical_by_notation.get(notation, {}).get(
                                "retrieval_score", 0.0
                            )
                        ),
                        6,
                    ),
                }
            )
        return output


def _choice_schema(candidates: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    choices = [str(row["notation"]) for row in candidates]
    return {
        "type": "object",
        "additionalProperties": False,
        "required": ["decision"],
        "properties": {
            "decision": {
                "type": "object",
                "additionalProperties": False,
                "required": [
                    "uid",
                    "central_subject",
                    "notation",
                    "runner_up_notation",
                    "disposition",
                    "rationale",
                ],
                "properties": {
                    "uid": {"type": "string"},
                    "central_subject": {"type": "string", "maxLength": 240},
                    "notation": {"type": "string", "enum": [*choices, HOLD]},
                    "runner_up_notation": {
                        "type": "string",
                        "enum": [*choices, NONE],
                    },
                    "disposition": {
                        "type": "string",
                        "enum": ["leaf", "ancestor", "hold"],
                    },
                    "rationale": {"type": "string", "maxLength": 480},
                },
            }
        },
    }


def _judge_schema(choices: Sequence[str]) -> dict[str, Any]:
    return {
        "type": "object",
        "additionalProperties": False,
        "required": ["decision"],
        "properties": {
            "decision": {
                "type": "object",
                "additionalProperties": False,
                "required": ["choice", "disposition", "rationale"],
                "properties": {
                    "choice": {"type": "string", "enum": [*choices, HOLD]},
                    "disposition": {
                        "type": "string",
                        "enum": ["leaf", "ancestor", "hold"],
                    },
                    "rationale": {"type": "string", "maxLength": 480},
                },
            }
        },
    }


def _binary_schema() -> dict[str, Any]:
    axes = {
        name: {"type": "integer", "minimum": 0, "maximum": 1}
        for name in (
            "central_subject_fit",
            "official_definition_fit",
            "non_incidental_match",
            "specificity_supported",
            "rival_not_better",
            "fatal_contradiction",
        )
    }
    return {
        "type": "object",
        "additionalProperties": False,
        "required": ["verification"],
        "properties": {
            "verification": {
                "type": "object",
                "additionalProperties": False,
                "required": [*axes, "rationale"],
                "properties": {
                    **axes,
                    "rationale": {"type": "string", "maxLength": 480},
                },
            }
        },
    }


def _prompt_page(
    page: Mapping[str, Any],
    candidates: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    return {
        "uid": str(page["uid"]),
        "title": str(page.get("title") or ""),
        "summary": str(page.get("summary") or ""),
        "excerpt": str(page.get("excerpt") or "")[:2_400],
        "official_candidates": [
            {
                "notation": str(row["notation"]),
                "label_en": str(row.get("label_en") or ""),
                "label_ja": str(row.get("label_ja") or ""),
                "official_lineage": list(row.get("official_lineage") or []),
            }
            for row in candidates
        ],
    }


class PilotRunner:
    """Run shared model stages and derive method ablations from the same calls."""

    def __init__(
        self,
        *,
        package: UDCPackage,
        cache_dir: Path,
        call_model: Callable[..., dict[str, Any]] | None = None,
        candidate_index: AuthoritativeCandidateIndex | None = None,
    ) -> None:
        self.package = package
        self.cache_dir = cache_dir
        self.config = load_decision_router_config()
        self.candidate_index = candidate_index or AuthoritativeCandidateIndex(
            package
        )
        self.call_model = call_model or self._cached_model_call

    def _cached_model_call(
        self,
        *,
        model: str,
        keep_alive: str,
        prompt: Mapping[str, Any],
        schema: Mapping[str, Any],
        stage: str,
    ) -> dict[str, Any]:
        digest = hashlib.sha256(
            json.dumps(
                {
                    "engine_version": PILOT_ENGINE_VERSION,
                    "model": model,
                    "prompt": prompt,
                    "schema": schema,
                    "stage": stage,
                },
                ensure_ascii=False,
                sort_keys=True,
            ).encode("utf-8")
        ).hexdigest()
        path = self.cache_dir / f"{digest}.json"
        if path.exists():
            cached = json.loads(path.read_text(encoding="utf-8"))
            cached["cached"] = True
            return cached

        started = time.perf_counter()
        payload: dict[str, Any] | None = None
        last_response = ""
        attempts = 0
        for attempts in range(1, 4):
            think: bool | str = (
                "low" if model.strip().lower().startswith("gpt-oss") else False
            )
            response = ollama.chat(
                [
                    {
                        "role": "system",
                        "content": (
                            "You are a conservative professional librarian. "
                            "Use only the supplied official UDC labels and lineage. "
                            "Return schema-valid JSON only."
                        ),
                    },
                    {
                        "role": "user",
                        "content": json.dumps(
                            prompt, ensure_ascii=False, sort_keys=True
                        ),
                    },
                ],
                model=model,
                format=dict(schema),
                num_ctx=min(32_768, self.config.num_ctx),
                num_predict=1_536,
                keep_alive=keep_alive,
                read_timeout_ms=self.config.read_timeout_ms,
                max_output_chars=12_000,
                temperature=0,
                seed=attempts - 1,
                think=think,
            )
            last_response = str(response).strip()
            try:
                parsed = json.loads(last_response)
            except json.JSONDecodeError:
                continue
            if isinstance(parsed, dict):
                payload = parsed
                break
        if payload is None:
            raise ValueError(
                f"{model} returned invalid JSON after {attempts} attempts "
                f"for {stage}: {last_response[:160]!r}"
            )
        result = {
            "model": model,
            "stage": stage,
            "attempts": attempts,
            "latency_seconds": round(time.perf_counter() - started, 3),
            "cached": False,
            "payload": payload,
        }
        _write_json(path, result)
        return result

    def _proposal(
        self,
        page: Mapping[str, Any],
        candidates: Sequence[Mapping[str, Any]],
        *,
        role: str,
        model: str,
        keep_alive: str,
    ) -> dict[str, Any]:
        if role == "subject-first":
            instruction = (
                "Infer the document's central subject before looking at labels. "
                "Choose the most specific supported official candidate. Exact "
                "word overlap is not evidence. Choose an ancestor only when the "
                "leaf is unsupported; hold only for unresolved cross-branch "
                "ambiguity."
            )
        else:
            instruction = (
                "Independently eliminate candidates whose label match is merely "
                "incidental, metaphorical, or based on an ambiguous word. Then "
                "choose the official path that explains the document's purpose. "
                "Prefer a justified ancestor over a guessed leaf."
            )
        result = self.call_model(
            model=model,
            keep_alive=keep_alive,
            prompt={
                "task": "zero-shot hierarchical UDC classification",
                "role": role,
                "instruction": instruction,
                "page": _prompt_page(page, candidates),
            },
            schema=_choice_schema(candidates),
            stage=role,
        )
        decision = dict(result["payload"]["decision"])
        allowed = {str(row["notation"]) for row in candidates}
        notation = str(decision.get("notation") or "")
        disposition = str(decision.get("disposition") or "")
        if notation == HOLD or disposition == "hold":
            notation = ""
            disposition = "hold"
        elif notation not in allowed:
            notation = ""
            disposition = "hold"
            decision["invalid_reason"] = "notation_outside_host_candidates"
        return {
            **decision,
            "notation": notation,
            "disposition": disposition,
            "model": result["model"],
            "latency_seconds": result["latency_seconds"],
            "cached": result["cached"],
        }

    def _prediction(self, proposal: Mapping[str, Any]) -> dict[str, Any]:
        disposition = str(proposal.get("disposition") or "hold")
        return {
            "status": (
                "held"
                if disposition == "hold"
                else "provisional"
                if disposition == "ancestor"
                else "proposed"
            ),
            "primary_notation": str(proposal.get("notation") or ""),
            "rationale": str(proposal.get("rationale") or ""),
        }

    def _common_ancestor(self, left: str, right: str) -> str:
        if not left or not right:
            return ""
        right_path = set(self.candidate_index.notation_path(right))
        for notation in self.candidate_index.notation_path(left):
            if notation in right_path:
                return notation if len(notation) > 1 else ""
        return ""

    def _path_vote(
        self,
        left: Mapping[str, Any],
        right: Mapping[str, Any],
    ) -> dict[str, Any]:
        left_notation = str(left.get("notation") or "")
        right_notation = str(right.get("notation") or "")
        if left_notation and left_notation == right_notation:
            disposition = (
                "ancestor"
                if "ancestor"
                in {
                    str(left.get("disposition") or ""),
                    str(right.get("disposition") or ""),
                }
                else "leaf"
            )
            return self._prediction(
                {
                    "notation": left_notation,
                    "disposition": disposition,
                    "rationale": "Two independent paths agree.",
                }
            )
        ancestor = self._common_ancestor(left_notation, right_notation)
        if ancestor:
            return self._prediction(
                {
                    "notation": ancestor,
                    "disposition": "ancestor",
                    "rationale": (
                        "Independent paths disagree below their deepest "
                        "authoritative common ancestor."
                    ),
                }
            )
        return self._prediction(
            {
                "notation": "",
                "disposition": "hold",
                "rationale": "Independent paths disagree across UDC branches.",
            }
        )

    def _gemma_judge(
        self,
        page: Mapping[str, Any],
        candidates: Sequence[Mapping[str, Any]],
        left: Mapping[str, Any],
        right: Mapping[str, Any],
    ) -> tuple[dict[str, Any], dict[str, Any] | None]:
        left_notation = str(left.get("notation") or "")
        right_notation = str(right.get("notation") or "")
        if left_notation and left_notation == right_notation:
            return self._path_vote(left, right), None
        common = self._common_ancestor(left_notation, right_notation)
        choices = [
            value
            for value in (left_notation, right_notation, common)
            if value
        ]
        choices = list(dict.fromkeys(choices))
        cards = [
            self.candidate_index.card(notation)
            for notation in choices
        ]
        result = self.call_model(
            model=self.config.tie_break_model,
            keep_alive=self.config.tie_break_keep_alive,
            prompt={
                "task": "path-valid UDC disagreement adjudication",
                "instruction": (
                    "Compare the two independently proposed paths against the "
                    "document's central purpose. Choose one proposal, their "
                    "deepest official common ancestor, or HOLD. A third leaf is "
                    "forbidden. Ignore incidental keyword overlap."
                ),
                "page": _prompt_page(
                    page, [card for card in cards if card is not None]
                ),
                "left_proposal": dict(left),
                "right_proposal": dict(right),
                "common_ancestor": common or None,
            },
            schema=_judge_schema(choices),
            stage="gemma-path-judge",
        )
        decision = dict(result["payload"]["decision"])
        choice = str(decision.get("choice") or HOLD)
        disposition = str(decision.get("disposition") or "hold")
        if choice == HOLD or disposition == "hold" or choice not in choices:
            choice = ""
            disposition = "hold"
        if choice == common and common not in {left_notation, right_notation}:
            disposition = "ancestor"
        prediction = self._prediction(
            {
                "notation": choice,
                "disposition": disposition,
                "rationale": str(decision.get("rationale") or ""),
            }
        )
        return prediction, {
            **decision,
            "model": result["model"],
            "latency_seconds": result["latency_seconds"],
            "cached": result["cached"],
        }

    def _binary_verify(
        self,
        page: Mapping[str, Any],
        prediction: Mapping[str, Any],
        *,
        rival_notation: str,
    ) -> tuple[dict[str, Any], dict[str, Any] | None]:
        winner = str(prediction.get("primary_notation") or "")
        if not winner or prediction.get("status") == "held":
            return dict(prediction), None
        cards = [
            card
            for card in (
                self.candidate_index.card(winner),
                self.candidate_index.card(rival_notation)
                if rival_notation
                else None,
            )
            if card is not None
        ]
        result = self.call_model(
            model=self.config.tie_break_model,
            keep_alive=self.config.tie_break_keep_alive,
            prompt={
                "task": "binary UDC winner verification",
                "instruction": (
                    "Evaluate each proposition independently as 0 or 1. Do not "
                    "invent confidence scores. The official definition is the "
                    "label and lineage supplied here. Local corpus frequency is "
                    "not evidence."
                ),
                "page": _prompt_page(page, cards),
                "winner_notation": winner,
                "rival_notation": rival_notation or None,
                "axes": {
                    "central_subject_fit": (
                        "Winner describes the document's central purpose."
                    ),
                    "official_definition_fit": (
                        "Document fits the supplied official label and lineage."
                    ),
                    "non_incidental_match": (
                        "Fit is substantive, not a keyword accident or metaphor."
                    ),
                    "specificity_supported": (
                        "Evidence supports this exact depth rather than only "
                        "its parent."
                    ),
                    "rival_not_better": (
                        "The supplied rival does not explain the document better."
                    ),
                    "fatal_contradiction": (
                        "Winner conflicts with the document's main claims."
                    ),
                },
            },
            schema=_binary_schema(),
            stage="gemma-binary-verifier",
        )
        verification = dict(result["payload"]["verification"])
        core_pass = all(
            int(verification.get(axis) or 0) == 1
            for axis in (
                "central_subject_fit",
                "official_definition_fit",
                "non_incidental_match",
                "rival_not_better",
            )
        ) and int(verification.get("fatal_contradiction") or 0) == 0
        if core_pass and int(verification.get("specificity_supported") or 0) == 1:
            revised = dict(prediction)
        elif core_pass:
            path = self.candidate_index.notation_path(winner)
            parent = path[1] if len(path) > 1 and len(path[1]) > 1 else ""
            revised = {
                "status": "provisional" if parent else "held",
                "primary_notation": parent,
                "rationale": (
                    "Binary verifier accepted the subject but not leaf "
                    "specificity."
                ),
            }
        else:
            revised = {
                "status": "held",
                "primary_notation": "",
                "rationale": "Binary verifier rejected the winning path.",
            }
        return revised, {
            **verification,
            "model": result["model"],
            "latency_seconds": result["latency_seconds"],
            "cached": result["cached"],
        }

    def run_case(self, page: Mapping[str, Any]) -> dict[str, Any]:
        candidates = self.candidate_index.candidates(page)
        left = self._proposal(
            page,
            candidates,
            role="subject-first",
            model=self.config.primary_model,
            keep_alive=self.config.primary_keep_alive,
        )
        right = self._proposal(
            page,
            candidates,
            role="exclusion-first",
            model=self.config.challenger_model,
            keep_alive=self.config.challenger_keep_alive,
        )
        single = self._prediction(left)
        path_vote = self._path_vote(left, right)
        gemma, judge = self._gemma_judge(page, candidates, left, right)
        rival = ""
        for proposal in (left, right):
            notation = str(proposal.get("notation") or "")
            if notation and notation != str(gemma.get("primary_notation") or ""):
                rival = notation
                break
        binary, verification = self._binary_verify(
            page, gemma, rival_notation=rival
        )
        baseline = dict(page.get("baseline") or {})
        return {
            "uid": str(page["uid"]),
            "page_id": str(page["page_id"]),
            "title": str(page.get("title") or ""),
            "bucket": str(page.get("diagnostic_bucket") or ""),
            "reference": dict(page["reference"]),
            "candidate_retrieval": {
                "count": len(candidates),
                "notations": [str(row["notation"]) for row in candidates],
                "cards": candidates,
            },
            "stages": {
                "ornith_subject_first": left,
                "gpt_oss_exclusion_first": right,
                "gemma_path_judge": judge,
                "gemma_binary_verifier": verification,
            },
            "variants": {
                "baseline_current": {
                    "status": str(baseline.get("status") or "held"),
                    "primary_notation": str(
                        baseline.get("primary_notation") or ""
                    ),
                    "rationale": str(baseline.get("rationale") or ""),
                },
                "semantic_ornith": single,
                "semantic_two_model_path": path_vote,
                "semantic_two_model_gemma": gemma,
                "semantic_two_model_gemma_binary": binary,
            },
        }


def _major(notation: str) -> str:
    return notation[:1] if notation[:1].isdigit() else notation


def score_prediction(
    reference: Mapping[str, Any],
    prediction: Mapping[str, Any],
) -> dict[str, Any]:
    expected = str(reference.get("expected_disposition") or "")
    primary = str(reference.get("primary_notation") or "")
    acceptable = {
        primary,
        *[
            str(value)
            for value in reference.get("acceptable_notations") or []
        ],
    }
    acceptable.discard("")
    ancestors = {
        str(value)
        for value in reference.get("acceptable_ancestor_notations") or []
    }
    status = str(prediction.get("status") or "held")
    notation = str(prediction.get("primary_notation") or "")
    assigned = status != "held" and bool(notation)

    if expected == "hold":
        accepted = status == "held"
        exact = accepted
        status_match = accepted
    else:
        accepted = assigned and (notation in acceptable or notation in ancestors)
        exact = assigned and notation == primary
        status_match = (
            (expected == "leaf" and status == "proposed")
            or (expected == "ancestor" and status == "provisional")
        )
    catastrophic = bool(
        expected != "hold"
        and assigned
        and notation not in acceptable
        and notation not in ancestors
        and _major(notation) != _major(primary)
    )
    return {
        "accepted": accepted,
        "exact": exact,
        "status_match": status_match,
        "assigned": assigned,
        "held": status == "held",
        "provisional": status == "provisional",
        "catastrophic": catastrophic,
    }


def summarize_cases(cases: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    variant_names = list(cases[0]["variants"]) if cases else []
    summaries: dict[str, Any] = {}
    for variant in variant_names:
        scored = [
            score_prediction(case["reference"], case["variants"][variant])
            for case in cases
        ]
        summaries[variant] = {
            "total": len(scored),
            "accepted": sum(bool(row["accepted"]) for row in scored),
            "exact": sum(bool(row["exact"]) for row in scored),
            "status_match": sum(bool(row["status_match"]) for row in scored),
            "assigned": sum(bool(row["assigned"]) for row in scored),
            "held": sum(bool(row["held"]) for row in scored),
            "provisional": sum(bool(row["provisional"]) for row in scored),
            "catastrophic": sum(bool(row["catastrophic"]) for row in scored),
            "case_scores": scored,
        }
    ranking = sorted(
        variant_names,
        key=lambda name: (
            summaries[name]["catastrophic"],
            -summaries[name]["accepted"],
            -summaries[name]["status_match"],
            -summaries[name]["exact"],
            summaries[name]["held"],
            name,
        ),
    )
    diagnostic_leader = ranking[0] if ranking else None
    leader_summary = summaries.get(diagnostic_leader or "", {})
    required_accepted = math.ceil(len(cases) * 0.8)
    production_qualified = bool(
        diagnostic_leader
        and leader_summary.get("catastrophic") == 0
        and leader_summary.get("accepted", 0) >= required_accepted
        and leader_summary.get("held", 0) <= math.floor(len(cases) * 0.2)
    )
    candidate_primary_recall = sum(
        str(case["reference"].get("primary_notation") or "")
        in {
            str(value)
            for value in case["candidate_retrieval"].get("notations") or []
        }
        for case in cases
    )
    return {
        "variants": summaries,
        "pilot_ranking": ranking,
        "diagnostic_leader": diagnostic_leader,
        "pilot_winner": diagnostic_leader if production_qualified else None,
        "production_qualified": production_qualified,
        "candidate_primary_recall": candidate_primary_recall,
        "candidate_primary_recall_total": len(cases),
        "selection_contract": (
            "A diagnostic leader is not a winner unless it has zero catastrophic "
            "errors, at least 80% accepted labels, and no more than 20% Hold. "
            "Any qualifying method still requires an unused 30-50 case "
            "confirmation set before production adoption."
        ),
    }


def run_pilot(
    *,
    input_path: Path,
    output_path: Path,
    root: Path,
) -> dict[str, Any]:
    rows = _jsonl(input_path)
    if not rows or any(
        row.get("diagnostic_schema") != DIAGNOSTIC_SCHEMA for row in rows
    ):
        raise ValueError("pilot input is not a diagnostic fixture")
    package = load_udc_package(root)
    cache_dir = output_path.parent / f"{output_path.stem}-stage-cache"
    runner = PilotRunner(package=package, cache_dir=cache_dir)

    existing: dict[str, Any] = {}
    if output_path.exists():
        previous = json.loads(output_path.read_text(encoding="utf-8"))
        existing = {
            str(row["uid"]): row for row in previous.get("cases") or []
        }
    cases: list[dict[str, Any]] = []
    for index, row in enumerate(rows, start=1):
        uid = str(row["uid"])
        case = existing.get(uid) or runner.run_case(row)
        cases.append(case)
        partial = {
            "schema": PILOT_SCHEMA,
            "engine_version": PILOT_ENGINE_VERSION,
            "status": "running" if index < len(rows) else "complete",
            "updated_at": _now(),
            "input_path": str(input_path),
            "input_sha256": _sha256_bytes(input_path.read_bytes()),
            "package_release": package.release,
            "package_checksum": package.checksum,
            "case_count": len(rows),
            "completed": len(cases),
            "frontier_calls": 0,
            "production_mutations": 0,
            "cases": cases,
            "summary": summarize_cases(cases),
        }
        _write_json(output_path, partial)
    return json.loads(output_path.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the read-only Chronovisor classification method pilot."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare = subparsers.add_parser("prepare")
    prepare.add_argument("--fixture", type=Path, required=True)
    prepare.add_argument("--baseline-results", type=Path, required=True)
    prepare.add_argument("--spec", type=Path, required=True)
    prepare.add_argument("--output", type=Path, required=True)

    run = subparsers.add_parser("run")
    run.add_argument("--input", type=Path, required=True)
    run.add_argument("--output", type=Path, required=True)
    run.add_argument(
        "--root",
        type=Path,
        default=Path.home() / ".chronovisor",
    )

    args = parser.parse_args()
    if args.command == "prepare":
        result = prepare_diagnostic_set(
            fixture_path=args.fixture,
            baseline_results_path=args.baseline_results,
            spec_path=args.spec,
            output_path=args.output,
        )
    else:
        result = run_pilot(
            input_path=args.input,
            output_path=args.output,
            root=args.root,
        )
    if args.command == "run":
        concise_summary = {
            key: value
            for key, value in result["summary"].items()
            if key != "variants"
        }
        concise_summary["variants"] = {
            name: {
                key: value
                for key, value in metrics.items()
                if key != "case_scores"
            }
            for name, metrics in result["summary"]["variants"].items()
        }
        result = {
            "schema": result["schema"],
            "status": result["status"],
            "output_path": str(args.output),
            "case_count": result["case_count"],
            "completed": result["completed"],
            "frontier_calls": result["frontier_calls"],
            "production_mutations": result["production_mutations"],
            "summary": concise_summary,
        }
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
