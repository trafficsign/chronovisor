"""Read-only v2 pilot for multi-path hierarchical UDC classification."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

from chronovisor.classification import UDCPackage, load_udc_package
from chronovisor.lab.classification_pilot import (
    HOLD,
    NONE,
    AuthoritativeCandidateIndex,
    PilotRunner,
    _jsonl,
    _now,
    _prompt_page,
    _sha256_bytes,
    _write_json,
    score_prediction,
)


V2_SCHEMA = "chronovisor.classification-method-pilot.v2"
V2_ENGINE_VERSION = 12
RRF_K = 60
MAX_CANDIDATES = 128
GROUP_STAGE_MODEL = "gemma4:26b"
GROUP_KEEP_ALIVE = "10m"


def _normalization_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "additionalProperties": False,
        "required": ["subject"],
        "properties": {
            "subject": {
                "type": "object",
                "additionalProperties": False,
                "required": [
                    "central_subject",
                    "query_paths",
                    "exclude_concepts",
                    "proposed_notations",
                ],
                "properties": {
                    "central_subject": {"type": "string", "maxLength": 280},
                    "query_paths": {
                        "type": "object",
                        "additionalProperties": False,
                        "required": [
                            "content_domain",
                            "document_purpose",
                            "method",
                            "broader_discipline",
                        ],
                        "properties": {
                            name: {"type": "string", "maxLength": 160}
                            for name in (
                                "content_domain",
                                "document_purpose",
                                "method",
                                "broader_discipline",
                            )
                        },
                    },
                    "exclude_concepts": {
                        "type": "array",
                        "maxItems": 5,
                        "items": {"type": "string", "maxLength": 120},
                    },
                    "proposed_notations": {
                        "type": "array",
                        "maxItems": 5,
                        "items": {"type": "string", "maxLength": 32},
                    },
                },
            }
        },
    }


def _ranking_schema(candidates: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    choices = [str(row["notation"]) for row in candidates]
    return {
        "type": "object",
        "additionalProperties": False,
        "required": ["assessment"],
        "properties": {
            "assessment": {
                "type": "object",
                "additionalProperties": False,
                "required": ["ranking", "no_fit", "certain_parent"],
                "properties": {
                    "ranking": {
                        "type": "array",
                        "minItems": min(3, len(choices)),
                        "maxItems": min(5, len(choices)),
                        "items": {
                            "type": "object",
                            "additionalProperties": False,
                            "required": [
                                "notation",
                                "incidental_match",
                                "fatal_contradiction",
                                "specificity_supported",
                                "rationale",
                            ],
                            "properties": {
                                "notation": {
                                    "type": "string",
                                    "enum": choices,
                                },
                                "incidental_match": {
                                    "type": "integer",
                                    "minimum": 0,
                                    "maximum": 1,
                                },
                                "fatal_contradiction": {
                                    "type": "integer",
                                    "minimum": 0,
                                    "maximum": 1,
                                },
                                "specificity_supported": {
                                    "type": "integer",
                                    "minimum": 0,
                                    "maximum": 1,
                                },
                                "rationale": {
                                    "type": "string",
                                    "maxLength": 320,
                                },
                            },
                        },
                    },
                    "no_fit": {
                        "type": "integer",
                        "minimum": 0,
                        "maximum": 1,
                    },
                    "certain_parent": {
                        "type": "string",
                        "enum": [*choices, NONE],
                    },
                },
            }
        },
    }


def _group_ranking_schema(
    candidates: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    choices = [str(row["notation"]) for row in candidates]
    return {
        "type": "object",
        "additionalProperties": False,
        "required": ["assessment"],
        "properties": {
            "assessment": {
                "type": "object",
                "additionalProperties": False,
                "required": ["ranked_notations", "no_fit"],
                "properties": {
                    "ranked_notations": {
                        "type": "array",
                        "minItems": min(3, len(choices)),
                        "maxItems": min(5, len(choices)),
                        "items": {"type": "string", "enum": choices},
                    },
                    "no_fit": {
                        "type": "integer",
                        "minimum": 0,
                        "maximum": 1,
                    },
                },
            }
        },
    }


def _arbitration_schema(choices: Sequence[str]) -> dict[str, Any]:
    return {
        "type": "object",
        "additionalProperties": False,
        "required": ["decision"],
        "properties": {
            "decision": {
                "type": "object",
                "additionalProperties": False,
                "required": ["choice", "specificity_supported", "rationale"],
                "properties": {
                    "choice": {"type": "string", "enum": [*choices, HOLD]},
                    "specificity_supported": {
                        "type": "integer",
                        "minimum": 0,
                        "maximum": 1,
                    },
                    "rationale": {"type": "string", "maxLength": 480},
                },
            }
        },
    }


def _veto_schema() -> dict[str, Any]:
    axes = {
        name: {"type": "integer", "minimum": 0, "maximum": 1}
        for name in (
            "central_subject_fit",
            "official_definition_fit",
            "non_incidental_match",
            "specificity_supported",
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


class MultiPathCandidateIndex(AuthoritativeCandidateIndex):
    """Union independent semantic, hierarchy, model, and legacy paths."""

    def _semantic(self, query: str, limit: int) -> list[tuple[str, float]]:
        vector = self._embed_batched([query])[0]
        scores = [
            (notation, self._cosine(vector, candidate_vector))
            for notation, candidate_vector in zip(
                self._notations, self._ensure_vectors(), strict=True
            )
        ]
        return sorted(scores, key=lambda value: (-value[1], value[0]))[:limit]

    @staticmethod
    def _cosine(left: list[float], right: list[float]) -> float:
        from chronovisor.embedding import cosine

        return cosine(left, right)

    def candidates_v2(
        self,
        page: Mapping[str, Any],
        normalizations: Sequence[Mapping[str, Any]],
        *,
        total_limit: int = MAX_CANDIDATES,
    ) -> list[dict[str, Any]]:
        query_paths: list[tuple[str, str, int]] = [
            ("page_semantic", self.page_query(page), 10)
        ]
        model_notations: list[str] = []
        for normalized in normalizations:
            central = str(normalized.get("central_subject") or "").strip()
            if central:
                query_paths.append(("central_subject", central, 5))
            paths = normalized.get("query_paths") or {}
            quotas = {
                "content_domain": 7,
                "document_purpose": 20,
                "method": 6,
                "broader_discipline": 4,
            }
            for name, quota in quotas.items():
                value = str(paths.get(name) or "").strip()
                if value:
                    query_paths.append((name, value, quota))
            model_notations.extend(
                str(value).strip()
                for value in normalized.get("proposed_notations") or []
                if str(value).strip() in self.by_notation
            )
        deduped_paths: list[tuple[str, str, int]] = []
        seen_queries: set[str] = set()
        for source, query, quota in query_paths:
            if query not in seen_queries:
                seen_queries.add(query)
                deduped_paths.append((source, query, quota))
        path_priority = {
            "document_purpose": 0,
            "page_semantic": 1,
            "content_domain": 2,
            "method": 3,
            "broader_discipline": 4,
            "central_subject": 5,
        }
        deduped_paths.sort(
            key=lambda value: (path_priority.get(value[0], 9), value[1])
        )

        score: dict[str, float] = defaultdict(float)
        sources: dict[str, set[str]] = defaultdict(set)
        semantic_value: dict[str, float] = {}
        reserved: list[str] = []
        per_query = 20
        for source, query, quota in deduped_paths:
            path = self._semantic(query, per_query)
            for rank, (notation, similarity) in enumerate(path, start=1):
                score[notation] += 1.0 / (RRF_K + rank)
                sources[notation].add(f"semantic_{source}")
                semantic_value[notation] = max(
                    semantic_value.get(notation, -1.0), similarity
                )
                if rank <= quota and notation not in reserved:
                    reserved.append(notation)

        for rank, notation in enumerate(dict.fromkeys(model_notations), start=1):
            score[notation] += 2.0 / (RRF_K + rank)
            sources[notation].add("model_free_proposal")

        legacy_rows = [dict(row) for row in page.get("candidates") or []]
        for rank, row in enumerate(legacy_rows, start=1):
            notation = str(row.get("notation") or "")
            if notation not in self.by_notation:
                continue
            score[notation] += 0.5 / (RRF_K + rank)
            sources[notation].add("legacy_lexical")

        ranked = sorted(score, key=lambda value: (-score[value], value))
        base = list(
            dict.fromkeys([*model_notations, *reserved, *ranked[:48]])
        )
        ancestors: list[str] = []
        for notation in base[:64]:
            for ancestor in self.notation_path(notation)[1:4]:
                if ancestor not in ancestors:
                    ancestors.append(ancestor)
                    sources[ancestor].add("hierarchy_ancestor")
        expanded = list(
            dict.fromkeys(
                [*model_notations, *reserved, *ancestors, *ranked[:48]]
            )
        )

        output: list[dict[str, Any]] = []
        for notation in expanded[: max(1, total_limit)]:
            card = self.card(notation)
            if card is None:
                continue
            output.append(
                {
                    **card,
                    "retrieval_sources": sorted(sources[notation]),
                    "fusion_score": round(score.get(notation, 0.0), 8),
                    "semantic_score": round(
                        semantic_value.get(notation, -1.0), 6
                    ),
                }
            )
        return output


def reciprocal_rank_fusion(
    rankings: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Fuse model-relative rankings; reject only unanimous hard negatives."""

    scores: dict[str, float] = defaultdict(float)
    evidence: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for assessment in rankings:
        for rank, item in enumerate(assessment.get("ranking") or [], start=1):
            notation = str(item.get("notation") or "")
            if not notation:
                continue
            scores[notation] += 1.0 / (RRF_K + rank)
            evidence[notation].append(dict(item))
    output = []
    for notation, value in scores.items():
        rows = evidence[notation]
        unanimously_rejected = len(rows) >= 2 and all(
            int(row.get("fatal_contradiction") or 0) == 1
            or int(row.get("incidental_match") or 0) == 1
            for row in rows
        )
        if unanimously_rejected:
            continue
        output.append(
            {
                "notation": notation,
                "rrf_score": value,
                "model_evidence": rows,
            }
        )
    return sorted(output, key=lambda row: (-row["rrf_score"], row["notation"]))


class V2PilotRunner(PilotRunner):
    def __init__(
        self,
        *,
        package: UDCPackage,
        cache_dir: Path,
        call_model: Callable[..., dict[str, Any]] | None = None,
        candidate_index: MultiPathCandidateIndex | None = None,
    ) -> None:
        super().__init__(
            package=package,
            cache_dir=cache_dir,
            call_model=call_model,
            candidate_index=candidate_index
            or MultiPathCandidateIndex(package),
        )
        self.candidate_index: MultiPathCandidateIndex

    def _normalize(
        self,
        page: Mapping[str, Any],
        *,
        role: str,
        model: str,
        keep_alive: str,
    ) -> dict[str, Any]:
        result = self.call_model(
            model=model,
            keep_alive=keep_alive,
            prompt={
                "architecture": "semantic-projection-v2",
                "task": "subject normalization before UDC candidate retrieval",
                "role": role,
                "instruction": (
                    "Describe the central subject without copying ambiguous title "
                    "words. Produce 3-5 deliberately diverse English library-search "
                    "phrases in the four named query_paths: content_domain, "
                    "document_purpose (the operational artifact or communicative "
                    "purpose, for example software implementation documentation or "
                    "interview speech strategy), method, and broader_discipline. "
                    "Do not collapse the paths into the domain object. Explicitly "
                    "list concepts that are incidental, "
                    "metaphorical, or must be excluded. You may suggest up to five "
                    "UDC Summary notations from memory; the host validates them, so "
                    "use an empty list when unsure."
                ),
                "page": {
                    "uid": str(page["uid"]),
                    "title": str(page.get("title") or ""),
                    "summary": str(page.get("summary") or ""),
                    "excerpt": str(page.get("excerpt") or "")[:2_400],
                },
            },
            schema=_normalization_schema(),
            stage=f"v2-normalize-{role}",
        )
        subject = dict(result["payload"]["subject"])
        return {
            **subject,
            "model": result["model"],
            "latency_seconds": result["latency_seconds"],
            "cached": result["cached"],
        }

    def _rank(
        self,
        page: Mapping[str, Any],
        candidates: Sequence[Mapping[str, Any]],
        normalization: Mapping[str, Any],
        *,
        role: str,
        model: str,
        keep_alive: str,
        stage_suffix: str = "",
    ) -> dict[str, Any]:
        result = self.call_model(
            model=model,
            keep_alive=keep_alive,
            prompt={
                "architecture": "semantic-projection-v2",
                "task": "independent relative ranking of official UDC candidates",
                "role": role,
                "instruction": (
                    "Return the best five candidates in order. Compare candidates "
                    "relative to each other; do not invent scores. Mark incidental "
                    "match only when a word is present but not the document purpose. "
                    "Mark fatal contradiction only for a direct subject conflict. "
                    "Use certain_parent when the branch is clear but leaf depth is "
                    "not. Local tags and corpus frequency are not authority."
                ),
                "normalized_subject": {
                    key: value
                    for key, value in normalization.items()
                    if key not in {"model", "latency_seconds", "cached"}
                },
                "page": {
                    "uid": str(page["uid"]),
                    "title": str(page.get("title") or ""),
                    "summary": str(page.get("summary") or ""),
                    "excerpt": str(page.get("excerpt") or "")[:2_400],
                    "official_candidates": [
                        {
                            "notation": str(row["notation"]),
                            "label_en": str(row.get("label_en") or ""),
                            "label_ja": str(row.get("label_ja") or ""),
                            "broader_notation": str(
                                row.get("broader_notation") or ""
                            ),
                        }
                        for row in candidates
                    ],
                },
            },
            schema=_ranking_schema(candidates),
            stage=f"v2-rank-{role}{stage_suffix}",
        )
        assessment = dict(result["payload"]["assessment"])
        allowed = {str(row["notation"]) for row in candidates}
        clean = []
        seen: set[str] = set()
        for item in assessment.get("ranking") or []:
            row = dict(item)
            notation = str(row.get("notation") or "")
            if notation in allowed and notation not in seen:
                seen.add(notation)
                clean.append(row)
        assessment["ranking"] = clean[:5]
        parent = str(assessment.get("certain_parent") or NONE)
        assessment["certain_parent"] = parent if parent in allowed else NONE
        return {
            **assessment,
            "model": result["model"],
            "latency_seconds": result["latency_seconds"],
            "cached": result["cached"],
        }

    def _rank_tournament(
        self,
        page: Mapping[str, Any],
        candidates: Sequence[Mapping[str, Any]],
        normalization: Mapping[str, Any],
        *,
        role: str,
        model: str,
        keep_alive: str,
    ) -> dict[str, Any]:
        if len(candidates) <= 40:
            return self._rank(
                page,
                candidates,
                normalization,
                role=role,
                model=model,
                keep_alive=keep_alive,
            )
        finalists, group_stage = self._shortlist_tournament(
            page,
            candidates,
            normalization,
        )
        final = self._rank(
            page,
            finalists,
            normalization,
            role=role,
            model=model,
            keep_alive=keep_alive,
            stage_suffix="-final",
        )
        final["group_stage"] = group_stage
        return final

    def _shortlist_tournament(
        self,
        page: Mapping[str, Any],
        candidates: Sequence[Mapping[str, Any]],
        normalization: Mapping[str, Any],
    ) -> tuple[list[Mapping[str, Any]], dict[str, Any]]:
        group_count = 4
        groups = [
            list(candidates[index::group_count])
            for index in range(group_count)
        ]
        group_results = []
        finalist_notations: list[str] = []
        for index, group in enumerate(groups, start=1):
            result = self._rank_group(
                page,
                group,
                normalization,
                role="shortlist-scout",
                model=GROUP_STAGE_MODEL,
                keep_alive=GROUP_KEEP_ALIVE,
                stage_suffix=f"-group-{index}",
            )
            group_results.append(result)
            finalist_notations.extend(
                str(row.get("notation") or "")
                for row in result.get("ranking") or []
            )
        by_notation = {
            str(row["notation"]): row for row in candidates
        }
        finalists = [
            by_notation[notation]
            for notation in dict.fromkeys(finalist_notations)
            if notation in by_notation
        ]
        group_stage = {
            "group_count": group_count,
            "group_size_max": max(len(group) for group in groups),
            "finalist_count": len(finalists),
            "model": GROUP_STAGE_MODEL,
            "groups": group_results,
        }
        return finalists, group_stage

    @staticmethod
    def _merge_normalizations(
        normalizations: Sequence[Mapping[str, Any]],
    ) -> dict[str, Any]:
        def joined(values: Sequence[str]) -> str:
            return " | ".join(dict.fromkeys(value for value in values if value))

        path_names = (
            "content_domain",
            "document_purpose",
            "method",
            "broader_discipline",
        )
        return {
            "central_subject": joined(
                [
                    str(item.get("central_subject") or "").strip()
                    for item in normalizations
                ]
            ),
            "query_paths": {
                name: joined(
                    [
                        str(
                            (item.get("query_paths") or {}).get(name) or ""
                        ).strip()
                        for item in normalizations
                    ]
                )
                for name in path_names
            },
            "exclude_concepts": list(
                dict.fromkeys(
                    str(value)
                    for item in normalizations
                    for value in item.get("exclude_concepts") or []
                    if str(value)
                )
            )[:10],
        }

    def _rank_group(
        self,
        page: Mapping[str, Any],
        candidates: Sequence[Mapping[str, Any]],
        normalization: Mapping[str, Any],
        *,
        role: str,
        model: str,
        keep_alive: str,
        stage_suffix: str,
    ) -> dict[str, Any]:
        result = self.call_model(
            model=model,
            keep_alive=keep_alive,
            prompt={
                "architecture": "semantic-projection-v2-group-stage",
                "task": "relative UDC group-stage ranking",
                "role": role,
                "instruction": (
                    "Return only the best five candidate notations in order. "
                    "Judge the document's central purpose, not keyword frequency. "
                    "Detailed rejection axes are deliberately deferred to the final."
                ),
                "normalized_subject": {
                    "central_subject": str(
                        normalization.get("central_subject") or ""
                    ),
                    "query_paths": dict(
                        normalization.get("query_paths") or {}
                    ),
                    "exclude_concepts": list(
                        normalization.get("exclude_concepts") or []
                    ),
                },
                "page": {
                    "uid": str(page["uid"]),
                    "title": str(page.get("title") or ""),
                    "summary": str(page.get("summary") or ""),
                    "excerpt": str(page.get("excerpt") or "")[:1_600],
                    "official_candidates": [
                        {
                            "notation": str(row["notation"]),
                            "label_en": str(row.get("label_en") or ""),
                            "label_ja": str(row.get("label_ja") or ""),
                            "broader_notation": str(
                                row.get("broader_notation") or ""
                            ),
                        }
                        for row in candidates
                    ],
                },
            },
            schema=_group_ranking_schema(candidates),
            stage=f"v2-group-rank-{role}{stage_suffix}",
        )
        assessment = dict(result["payload"]["assessment"])
        allowed = {str(row["notation"]) for row in candidates}
        ranked = []
        seen: set[str] = set()
        for value in assessment.get("ranked_notations") or []:
            notation = str(value)
            if notation in allowed and notation not in seen:
                seen.add(notation)
                ranked.append(
                    {
                        "notation": notation,
                        "incidental_match": 0,
                        "fatal_contradiction": 0,
                        "specificity_supported": 1,
                        "rationale": "Advanced from the relative group stage.",
                    }
                )
        return {
            "ranking": ranked[:5],
            "no_fit": int(assessment.get("no_fit") or 0),
            "certain_parent": NONE,
            "model": result["model"],
            "latency_seconds": result["latency_seconds"],
            "cached": result["cached"],
        }

    def _from_fusion(self, fused: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
        if not fused:
            return {
                "status": "held",
                "primary_notation": "",
                "rationale": "No candidate survived independent ranking.",
            }
        return {
            "status": "proposed",
            "primary_notation": str(fused[0]["notation"]),
            "rationale": "Host reciprocal-rank fusion winner.",
        }

    def _arbitrate_v2(
        self,
        page: Mapping[str, Any],
        fused: Sequence[Mapping[str, Any]],
        assessments: Sequence[Mapping[str, Any]],
    ) -> tuple[dict[str, Any], dict[str, Any] | None]:
        if not fused:
            return self._from_fusion(fused), None
        top_by_model = [
            str((row.get("ranking") or [{}])[0].get("notation") or "")
            for row in assessments
            if row.get("ranking")
        ]
        if len(top_by_model) >= 2 and len(set(top_by_model)) == 1:
            return self._from_fusion(fused), None
        choices = [str(row["notation"]) for row in fused[:3]]
        if len(top_by_model) >= 2:
            common = self._common_ancestor(top_by_model[0], top_by_model[1])
            if common and common not in choices:
                choices.append(common)
        cards = [
            self.candidate_index.card(notation)
            for notation in choices
        ]
        result = self.call_model(
            model=self.config.tie_break_model,
            keep_alive=self.config.tie_break_keep_alive,
            prompt={
                "architecture": "semantic-projection-v2",
                "task": "pairwise UDC arbitration",
                "instruction": (
                    "Choose only among the fused finalists, their common ancestor, "
                    "or HOLD. Compare which candidate best describes the page's "
                    "central purpose and official lineage. Do not freely reclassify."
                ),
                "page": _prompt_page(
                    page, [card for card in cards if card is not None]
                ),
                "model_assessments": [dict(value) for value in assessments],
                "fused_finalists": [dict(value) for value in fused[:3]],
            },
            schema=_arbitration_schema(choices),
            stage="v2-gemma-arbitration",
        )
        decision = dict(result["payload"]["decision"])
        choice = str(decision.get("choice") or HOLD)
        if choice == HOLD or choice not in choices:
            prediction = {
                "status": "held",
                "primary_notation": "",
                "rationale": str(decision.get("rationale") or ""),
            }
        else:
            top_paths = [
                self.candidate_index.notation_path(value)
                for value in top_by_model[:2]
            ]
            is_common_parent = (
                len(top_paths) == 2
                and choice in top_paths[0][1:]
                and choice in top_paths[1][1:]
            )
            prediction = {
                "status": (
                    "provisional"
                    if is_common_parent
                    or int(decision.get("specificity_supported") or 0) == 0
                    else "proposed"
                ),
                "primary_notation": choice,
                "rationale": str(decision.get("rationale") or ""),
            }
        return prediction, {
            **decision,
            "model": result["model"],
            "latency_seconds": result["latency_seconds"],
            "cached": result["cached"],
        }

    def _veto(
        self,
        page: Mapping[str, Any],
        prediction: Mapping[str, Any],
        fused: Sequence[Mapping[str, Any]],
    ) -> tuple[dict[str, Any], dict[str, Any] | None]:
        winner = str(prediction.get("primary_notation") or "")
        if not winner or prediction.get("status") == "held":
            return dict(prediction), None
        rival = next(
            (
                str(row["notation"])
                for row in fused
                if str(row["notation"]) != winner
            ),
            "",
        )
        cards = [
            card
            for card in (
                self.candidate_index.card(winner),
                self.candidate_index.card(rival) if rival else None,
            )
            if card is not None
        ]
        result = self.call_model(
            model=self.config.tie_break_model,
            keep_alive=self.config.tie_break_keep_alive,
            prompt={
                "architecture": "semantic-projection-v2",
                "task": "catastrophic-error veto, not reclassification",
                "instruction": (
                    "Evaluate five fact-adjacent propositions as 0 or 1. Veto is "
                    "reserved for a central-subject mismatch or fatal contradiction; "
                    "ordinary uncertainty is not fatal. Do not choose a new class."
                ),
                "page": _prompt_page(page, cards),
                "winner_notation": winner,
                "rival_notation": rival or None,
            },
            schema=_veto_schema(),
            stage="v2-gemma-veto",
        )
        check = dict(result["payload"]["verification"])
        fatal = int(check.get("fatal_contradiction") or 0) == 1
        double_miss = (
            int(check.get("central_subject_fit") or 0) == 0
            and int(check.get("official_definition_fit") or 0) == 0
        )
        incidental_miss = (
            int(check.get("central_subject_fit") or 0) == 0
            and int(check.get("non_incidental_match") or 0) == 0
        )
        revised = dict(prediction)
        if fatal or double_miss or incidental_miss:
            revised = {
                "status": "held",
                "primary_notation": "",
                "rationale": "Final verifier found a catastrophic subject mismatch.",
            }
        elif int(check.get("specificity_supported") or 0) == 0:
            path = self.candidate_index.notation_path(winner)
            parent = path[1] if len(path) > 1 and len(path[1]) > 1 else ""
            if parent:
                revised = {
                    "status": "provisional",
                    "primary_notation": parent,
                    "rationale": "Final verifier retained the branch at its parent.",
                }
        return revised, {
            **check,
            "model": result["model"],
            "latency_seconds": result["latency_seconds"],
            "cached": result["cached"],
        }

    def run_case_v2(self, page: Mapping[str, Any]) -> dict[str, Any]:
        left_subject = self._normalize(
            page,
            role="subject-first",
            model=self.config.primary_model,
            keep_alive=self.config.primary_keep_alive,
        )
        right_subject = self._normalize(
            page,
            role="exclusion-first",
            model=self.config.challenger_model,
            keep_alive=self.config.challenger_keep_alive,
        )
        candidates = self.candidate_index.candidates_v2(
            page, [left_subject, right_subject]
        )
        if len(candidates) > 40:
            finalists, group_stage = self._shortlist_tournament(
                page,
                candidates,
                self._merge_normalizations([left_subject, right_subject]),
            )
        else:
            finalists = list(candidates)
            group_stage = {
                "group_count": 0,
                "group_size_max": len(finalists),
                "finalist_count": len(finalists),
                "model": "not-required",
                "groups": [],
            }
        left_rank = self._rank(
            page,
            finalists,
            left_subject,
            role="subject-first",
            model=self.config.primary_model,
            keep_alive=self.config.primary_keep_alive,
            stage_suffix="-final",
        )
        right_rank = self._rank_group(
            page,
            finalists,
            right_subject,
            role="exclusion-first",
            model=self.config.challenger_model,
            keep_alive=self.config.challenger_keep_alive,
            stage_suffix="-compact-final",
        )
        left_rank["group_stage"] = group_stage
        right_rank["group_stage"] = group_stage
        fused = reciprocal_rank_fusion([left_rank, right_rank])
        fusion = self._from_fusion(fused)
        adjudicated, arbitration = self._arbitrate_v2(
            page, fused, [left_rank, right_rank]
        )
        verified, veto = self._veto(page, adjudicated, fused)
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
                "ornith_subject": left_subject,
                "gpt_oss_subject": right_subject,
                "ornith_ranking": left_rank,
                "gpt_oss_ranking": right_rank,
                "host_fusion": fused,
                "gemma_arbitration": arbitration,
                "gemma_veto": veto,
            },
            "variants": {
                "baseline_current": {
                    "status": str(baseline.get("status") or "held"),
                    "primary_notation": str(
                        baseline.get("primary_notation") or ""
                    ),
                    "rationale": str(baseline.get("rationale") or ""),
                },
                "v2_rank_fusion": fusion,
                "v2_ranked_gemma": adjudicated,
                "v2_ranked_gemma_veto": verified,
            },
        }

    def run_candidate_case(self, page: Mapping[str, Any]) -> dict[str, Any]:
        left_subject = self._normalize(
            page,
            role="subject-first",
            model=self.config.primary_model,
            keep_alive=self.config.primary_keep_alive,
        )
        right_subject = self._normalize(
            page,
            role="exclusion-first",
            model=self.config.challenger_model,
            keep_alive=self.config.challenger_keep_alive,
        )
        candidates = self.candidate_index.candidates_v2(
            page, [left_subject, right_subject]
        )
        return {
            "uid": str(page["uid"]),
            "page_id": str(page["page_id"]),
            "title": str(page.get("title") or ""),
            "bucket": str(page.get("diagnostic_bucket") or ""),
            "reference": dict(page["reference"]),
            "subjects": {
                "ornith": left_subject,
                "gpt_oss": right_subject,
            },
            "candidate_retrieval": {
                "count": len(candidates),
                "notations": [str(row["notation"]) for row in candidates],
                "cards": candidates,
            },
        }


def summarize_v2(cases: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    variants = list(cases[0]["variants"]) if cases else []
    metrics: dict[str, Any] = {}
    for variant in variants:
        scores = [
            score_prediction(case["reference"], case["variants"][variant])
            for case in cases
        ]
        metrics[variant] = {
            "total": len(scores),
            "accepted": sum(bool(row["accepted"]) for row in scores),
            "exact": sum(bool(row["exact"]) for row in scores),
            "catastrophic": sum(bool(row["catastrophic"]) for row in scores),
            "held": sum(bool(row["held"]) for row in scores),
            "provisional": sum(bool(row["provisional"]) for row in scores),
            "case_scores": scores,
        }
    ranking = sorted(
        variants,
        key=lambda name: (
            metrics[name]["catastrophic"],
            -metrics[name]["accepted"],
            -metrics[name]["exact"],
            metrics[name]["held"],
            name,
        ),
    )
    candidate_recall = sum(
        str(case["reference"].get("primary_notation") or "")
        in set(case["candidate_retrieval"].get("notations") or [])
        for case in cases
    )
    leader = ranking[0] if ranking else None
    leader_metrics = metrics.get(leader or "", {})
    qualified = bool(
        leader
        and candidate_recall == len(cases)
        and leader_metrics.get("catastrophic") == 0
        and leader_metrics.get("accepted", 0) >= len(cases) * 0.8
        and leader_metrics.get("held", 0) <= len(cases) * 0.2
    )
    return {
        "variants": metrics,
        "candidate_primary_recall": candidate_recall,
        "candidate_primary_recall_total": len(cases),
        "diagnostic_ranking": ranking,
        "diagnostic_leader": leader,
        "pilot_winner": leader if qualified else None,
        "production_qualified": qualified,
        "selection_contract": (
            "Diagnostic qualification requires candidate recall 100%, zero "
            "catastrophic errors, accepted >=80%, and Hold <=20%. A qualifying "
            "method still requires an unused 30-50 case confirmation set."
        ),
    }


def run_v2(*, input_path: Path, output_path: Path, root: Path) -> dict[str, Any]:
    rows = _jsonl(input_path)
    package = load_udc_package(root)
    runner = V2PilotRunner(
        package=package,
        cache_dir=output_path.parent / "v2-engine4-stage-cache",
    )
    existing: dict[str, Any] = {}
    if output_path.exists():
        previous = json.loads(output_path.read_text(encoding="utf-8"))
        existing = {
            str(row["uid"]): row for row in previous.get("cases") or []
        }
    cases = []
    for index, row in enumerate(rows, start=1):
        uid = str(row["uid"])
        case = existing.get(uid) or runner.run_case_v2(row)
        cases.append(case)
        payload = {
            "schema": V2_SCHEMA,
            "engine_version": V2_ENGINE_VERSION,
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
            "summary": summarize_v2(cases),
        }
        _write_json(output_path, payload)
    return json.loads(output_path.read_text(encoding="utf-8"))


def run_candidate_audit(
    *, input_path: Path, output_path: Path, root: Path
) -> dict[str, Any]:
    rows = _jsonl(input_path)
    package = load_udc_package(root)
    runner = V2PilotRunner(
        package=package,
        cache_dir=output_path.parent / "v2-engine4-stage-cache",
    )
    existing: dict[str, Any] = {}
    if output_path.exists():
        previous = json.loads(output_path.read_text(encoding="utf-8"))
        existing = {
            str(row["uid"]): row for row in previous.get("cases") or []
        }
    cases = []
    for index, row in enumerate(rows, start=1):
        uid = str(row["uid"])
        case = existing.get(uid) or runner.run_candidate_case(row)
        cases.append(case)
        recall = sum(
            str(value["reference"].get("primary_notation") or "")
            in set(value["candidate_retrieval"].get("notations") or [])
            for value in cases
        )
        _write_json(
            output_path,
            {
                "schema": "chronovisor.classification-candidate-audit.v2",
                "engine_version": V2_ENGINE_VERSION,
                "status": "running" if index < len(rows) else "complete",
                "updated_at": _now(),
                "input_path": str(input_path),
                "input_sha256": _sha256_bytes(input_path.read_bytes()),
                "case_count": len(rows),
                "completed": len(cases),
                "frontier_calls": 0,
                "production_mutations": 0,
                "cases": cases,
                "summary": {
                    "candidate_primary_recall": recall,
                    "candidate_primary_recall_total": len(cases),
                    "gate_passed": (
                        len(cases) == len(rows) and recall == len(cases)
                    ),
                },
            },
        )
    return json.loads(output_path.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the read-only Chronovisor v2 classification pilot."
    )
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--root", type=Path, default=Path.home() / ".chronovisor"
    )
    parser.add_argument(
        "--candidate-only",
        action="store_true",
        help="Stop after multi-path candidate recall auditing.",
    )
    args = parser.parse_args()
    if args.candidate_only:
        result = run_candidate_audit(
            input_path=args.input, output_path=args.output, root=args.root
        )
    else:
        result = run_v2(
            input_path=args.input, output_path=args.output, root=args.root
        )
    if args.candidate_only:
        print(
            json.dumps(
                {
                    key: value
                    for key, value in result.items()
                    if key
                    in {
                        "schema",
                        "status",
                        "completed",
                        "frontier_calls",
                        "production_mutations",
                        "summary",
                    }
                },
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
        )
        return
    concise = {
        key: value
        for key, value in result["summary"].items()
        if key != "variants"
    }
    concise["variants"] = {
        name: {
            key: value
            for key, value in metrics.items()
            if key != "case_scores"
        }
        for name, metrics in result["summary"]["variants"].items()
    }
    print(
        json.dumps(
            {
                "schema": result["schema"],
                "status": result["status"],
                "output_path": str(args.output),
                "completed": result["completed"],
                "frontier_calls": result["frontier_calls"],
                "production_mutations": result["production_mutations"],
                "summary": concise,
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
