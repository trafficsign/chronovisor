"""Deterministic stateful activation dynamics for precision-first Recall."""

from __future__ import annotations

import hashlib
import json
import math
import time
from collections import deque
from dataclasses import asdict, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from chronovisor.core.durable_state import sidecar_exclusive_lock
from chronovisor.core.jsonl_write import append_jsonl_durable
from chronovisor.core.store import CHRONOVISOR_ROOT
from chronovisor.recall.recall_field_schema import (
    ActivationNode,
    FieldEvent,
    FieldStimulus,
    RecallFieldConfig,
    RecallFieldState,
    load_recall_field_config,
    session_hash,
    topic_signature,
    topic_transition,
)
from chronovisor.recall.recall_field_store import RecallFieldStore
from chronovisor.search.feedback_ledger import (
    feedback_row_sha256,
    read_jsonl_rows,
    trusted_negative_feedback_row_error,
)
from chronovisor.search.graph_edges import typed_neighbors
from chronovisor.search.index_store import get_store

NEGATIVE_APPLICATION_LEDGER = (
    CHRONOVISOR_ROOT / "recall" / "field" / "negative-applications.jsonl"
)


def _clamp(value: float) -> float:
    return max(0.0, min(1.0, value))


def _negative_contribution_weight(
    state: RecallFieldState,
    *,
    page_id: str,
    buffer_name: str,
) -> float:
    total = 0.0
    for contribution in state.negative_contributions.values():
        if (
            not isinstance(contribution, dict)
            or contribution.get("buffer") != buffer_name
            or contribution.get("topic_epoch") != state.topic_epoch
            or contribution.get("prompt_hash") != state.topic_prompt_hash
        ):
            continue
        weights = contribution.get("page_weights")
        if isinstance(weights, dict):
            value = weights.get(page_id)
            if isinstance(value, int | float) and not isinstance(value, bool):
                total += max(0.0, float(value))
    return total


def effective_activation(
    state: RecallFieldState,
    *,
    page_id: str,
    node: ActivationNode,
    buffer_name: str,
) -> float:
    """Compose immutable base activation with all active negative producers."""

    return _clamp(
        node.activation
        - _negative_contribution_weight(
            state, page_id=page_id, buffer_name=buffer_name
        )
    )


def _suppressed_pages(state: RecallFieldState, *, buffer_name: str) -> set[str]:
    return {
        str(page_id)
        for contribution in state.negative_contributions.values()
        if isinstance(contribution, dict)
        and contribution.get("buffer") == buffer_name
        and contribution.get("topic_epoch") == state.topic_epoch
        and contribution.get("prompt_hash") == state.topic_prompt_hash
        for page_id, weight in (
            contribution.get("page_weights", {}).items()
            if isinstance(contribution.get("page_weights"), dict)
            else []
        )
        if isinstance(weight, int | float) and float(weight) > 0.0
    }


def _effective_config(config: RecallFieldConfig) -> RecallFieldConfig:
    """Apply each learned input only after its own supervision gate passes."""

    if not config.auto_growth:
        return config
    effective = config
    try:
        from chronovisor.recall.recall_learning import load_last_known_good

        lkg = load_last_known_good(
            CHRONOVISOR_ROOT
            / "runtime"
            / "recall-field"
            / "last-known-good-policy.json"
        )
        policy = lkg.get("policy") if isinstance(lkg, dict) else None
        if isinstance(policy, dict):
            effective = replace(
                effective,
                spread_gain=max(
                    0.0,
                    min(1.0, float(policy.get("spread_gain", effective.spread_gain))),
                ),
                global_inhibition=max(
                    0.0,
                    min(
                        1.0,
                        float(
                            policy.get("global_inhibition", effective.global_inhibition)
                        ),
                    ),
                ),
                turn_decay=max(
                    0.0,
                    min(1.0, float(policy.get("turn_decay", effective.turn_decay))),
                ),
            )
    except (TypeError, ValueError):
        effective = config
    try:
        from chronovisor.recall.recall_growth import automatic_learning_allowed

        if not effective.positive_learning and automatic_learning_allowed(enabled=True):
            effective = replace(effective, positive_learning=True)
    except Exception:
        pass
    return effective


def _event(
    state: RecallFieldState,
    now: float,
    kind: str,
    **kwargs: Any,
) -> FieldEvent:
    state.seq += 1
    return FieldEvent(
        seq=state.seq,
        timestamp_epoch=now,
        session_hash=state.session_hash,
        topic_epoch=state.topic_epoch,
        kind=kind,
        **kwargs,
    )


def prompt_stimuli(prompt: str, *, limit: int = 12) -> list[FieldStimulus]:
    """Resolve only exact hints, titles, page IDs, and entities from a prompt."""

    if not prompt.strip() or limit <= 0:
        return []
    store = get_store()
    store.refresh_if_stale()
    prompt_folded = prompt.casefold()
    stimuli: dict[str, FieldStimulus] = {}

    try:
        from chronovisor.ingest.recall_hints import matching_hint_page_ids

        for page_id in matching_hint_page_ids([prompt], limit=min(3, limit)):
            stimuli[page_id] = FieldStimulus(
                page_id=page_id,
                kind="prompt_exact",
                weight=0.92,
                reason_code="trusted_query_hint",
            )
    except Exception:
        pass

    for meta in store.all_pages_meta(include_system=True):
        page_id = str(meta.get("page_id") or "")
        if not page_id or page_id in stimuli:
            continue
        title = str(meta.get("title") or "")
        page_key = page_id.replace("-", " ").replace("_", " ").casefold()
        title_key = title.casefold().strip()
        exact_title = len(title_key) >= 6 and title_key in prompt_folded
        exact_page = len(page_key) >= 6 and page_key in prompt_folded
        entity_match = any(
            len(str(entity).strip()) >= 3
            and str(entity).strip().casefold() in prompt_folded
            for entity in meta.get("entities", [])
            if isinstance(entity, str)
        )
        if exact_title or exact_page or entity_match:
            stimuli[page_id] = FieldStimulus(
                page_id=page_id,
                kind="prompt_entity" if entity_match else "prompt_exact",
                weight=0.88 if entity_match else 1.0,
                reason_code="exact_entity" if entity_match else "exact_page",
            )
        if len(stimuli) >= limit:
            break
    return sorted(
        stimuli.values(),
        key=lambda item: (-item.weight, item.kind, item.page_id),
    )[:limit]


def _decay_buffer(
    buffer: dict[str, ActivationNode],
    *,
    elapsed_seconds: float,
    turn_delta: int,
    config: RecallFieldConfig,
) -> None:
    wall_decay = math.pow(
        0.5,
        max(0.0, elapsed_seconds) / max(1, config.wall_half_life_seconds),
    )
    turn_decay = math.pow(config.turn_decay, max(0, turn_delta))
    factor = wall_decay * turn_decay
    for node in buffer.values():
        node.activation = _clamp(node.activation * factor)
        node.direct *= factor
        node.spread *= factor
        node.negative *= factor
        node.inhibition *= factor


def update_field_state(
    state: RecallFieldState,
    *,
    stimuli: list[FieldStimulus],
    prompt_signature: tuple[str, ...],
    config: RecallFieldConfig,
    now: float,
    graph_store: Any | None = None,
    prompt_text: str = "",
) -> tuple[RecallFieldState, list[FieldEvent]]:
    """Apply one turn of deterministic sparse activation dynamics."""

    events: list[FieldEvent] = []
    previous_turn = state.turn
    state.turn += 1
    buffer_name = "shadow" if config.mode == "shadow" else "active"
    buffer = state.shadow if buffer_name == "shadow" else state.active
    elapsed = max(0.0, now - state.updated_at_epoch)
    _decay_buffer(
        state.active,
        elapsed_seconds=elapsed,
        turn_delta=state.turn - previous_turn,
        config=config,
    )
    _decay_buffer(
        state.shadow,
        elapsed_seconds=elapsed,
        turn_delta=state.turn - previous_turn,
        config=config,
    )

    transition, similarity = topic_transition(
        state.topic_signature,
        prompt_signature,
        prompt=prompt_text,
        reset_similarity=config.topic_reset_similarity,
    )
    topic_reset = bool(
        state.topic_signature and prompt_signature and transition == "reset"
    )
    if topic_reset:
        state.topic_epoch += 1
        buffer.clear()
        state.negative_contributions.clear()
        state.full_search_fallback = True
        state.pending_teacher_commits = [
            row
            for row in state.pending_teacher_commits
            if int(row.get("topic_epoch", -1)) == state.topic_epoch
        ]
        events.append(
            _event(
                state,
                now,
                "topic_reset",
                delta=round(1.0 - similarity, 6),
                reason_code=(
                    "explicit_topic_switch"
                    if prompt_text
                    and any(
                        marker in prompt_text.casefold()
                        for marker in ("別件", "ところで", "new topic", "switch topic")
                    )
                    else "topic_signature_jump"
                ),
            )
        )
    else:
        state.full_search_fallback = not bool(buffer)
    state.topic_signature = prompt_signature
    if prompt_text:
        from chronovisor.recall.recall_runtime import stable_prompt_hash

        state.topic_prompt_hash = stable_prompt_hash(prompt_text)

    applicable_commits: list[dict[str, Any]] = []
    pending: list[dict[str, Any]] = []
    suppressed = _suppressed_pages(state, buffer_name=buffer_name)
    for row in state.pending_teacher_commits:
        same_epoch = int(row.get("topic_epoch", -1)) == state.topic_epoch
        if (
            same_epoch
            and int(row.get("available_turn") or 0) <= state.turn
            and str(row.get("page_id") or "") not in suppressed
        ):
            applicable_commits.append(row)
        elif same_epoch and str(row.get("page_id") or "") not in suppressed:
            pending.append(row)
    state.pending_teacher_commits = pending
    effective_stimuli = list(stimuli)
    effective_stimuli.extend(
        FieldStimulus(
            page_id=str(row["page_id"]),
            kind="teacher_commit",
            weight=0.45,
            reason_code="prior_turn_teacher_commit",
            certificate_id=str(row.get("certificate_id") or ""),
            components={
                key: float(row["components"].get(key) or 0.0)
                for key in ("anti_index", "hub_penalty")
                if isinstance(row.get("components"), dict)
                and isinstance(row["components"].get(key), int | float)
            },
        )
        for row in applicable_commits
    )

    seen_direct: set[str] = set()
    for stimulus in sorted(
        effective_stimuli,
        key=lambda item: (item.page_id, item.kind, -item.weight),
    ):
        if not stimulus.page_id or stimulus.page_id in seen_direct:
            continue
        seen_direct.add(stimulus.page_id)
        node = buffer.setdefault(stimulus.page_id, ActivationNode())
        refractory = (
            node.last_turn > 0
            and state.turn - node.last_turn <= config.refractory_turns
        )
        delta = max(0.0, min(1.0, float(stimulus.weight)))
        if refractory and stimulus.kind != "teacher_commit":
            delta *= 0.25
        if stimulus.negative:
            node.negative = _clamp(node.negative + delta)
            node.activation = _clamp(node.activation - delta)
            kind = "inhibit"
        else:
            node.direct = _clamp(node.direct + delta)
            node.activation = _clamp(node.activation + delta)
            kind = "commit_applied" if stimulus.kind == "teacher_commit" else "stimulus"
        node.anti_index = max(
            node.anti_index,
            float(stimulus.components.get("anti_index") or 0.0),
        )
        node.hub_penalty = max(
            node.hub_penalty,
            float(stimulus.components.get("hub_penalty") or 0.0),
        )
        node.last_turn = state.turn
        node.last_seq = state.seq + 1
        events.append(
            _event(
                state,
                now,
                kind,
                page_id=stimulus.page_id,
                delta=round(-delta if stimulus.negative else delta, 6),
                activation=round(node.activation, 6),
                reason_code=stimulus.reason_code or stimulus.kind,
                certificate_id=stimulus.certificate_id,
                components={
                    "direct": round(node.direct, 6),
                    "negative": round(node.negative, 6),
                    "anti_index": round(node.anti_index, 6),
                    "hub_penalty": round(node.hub_penalty, 6),
                },
            )
        )

    graph = graph_store or get_store()
    graph.refresh_if_stale()
    frontier: deque[tuple[str, int, float]] = deque(
        (
            page_id,
            0,
            effective_activation(
                state, page_id=page_id, node=node, buffer_name=buffer_name
            ),
        )
        for page_id, node in sorted(
            buffer.items(),
            key=lambda item: (
                -effective_activation(
                    state,
                    page_id=item[0],
                    node=item[1],
                    buffer_name=buffer_name,
                ),
                item[0],
            ),
        )[: config.working_set_size]
        if effective_activation(
            state, page_id=page_id, node=node, buffer_name=buffer_name
        )
        > 0.01
    )
    traversed = 0
    best_spread: dict[str, float] = {}
    while frontier and traversed < config.max_active_edges:
        source, hop, source_activation = frontier.popleft()
        if hop >= config.max_hops:
            continue
        for edge in typed_neighbors(
            graph,
            source,
            limit=12,
            include_exposure_cofire=False,
            include_positive_cofire=config.positive_learning,
            degree_normalize=True,
            typed_relations_for_field=True,
            rollout_key=state.session_hash,
        ):
            traversed += 1
            next_hop = hop + 1
            delta = (
                source_activation
                * edge.weight
                * config.spread_gain
                * math.pow(0.72, next_hop)
            )
            if delta < 0.005 or delta <= best_spread.get(edge.target, 0.0):
                if traversed >= config.max_active_edges:
                    break
                continue
            best_spread[edge.target] = delta
            target = buffer.setdefault(edge.target, ActivationNode())
            target.spread = _clamp(target.spread + delta)
            target.activation = _clamp(target.activation + delta)
            target.last_seq = state.seq + 1
            events.append(
                _event(
                    state,
                    now,
                    "spread",
                    source_page_id=source,
                    target_page_id=edge.target,
                    edge_type=edge.edge_type,
                    delta=round(delta, 6),
                    activation=round(target.activation, 6),
                    reason_code=edge.supervision or edge.edge_type,
                    components={
                        "spread": round(target.spread, 6),
                        **(
                            {"relation_id": edge.relation_id}
                            if edge.relation_id
                            else {}
                        ),
                    },
                )
            )
            frontier.append((edge.target, next_hop, delta))
            if traversed >= config.max_active_edges:
                break

    over_capacity = max(0, len(buffer) - config.working_set_size)
    lateral = config.global_inhibition * over_capacity / max(1, config.working_set_size)
    if lateral > 0:
        for page_id, node in buffer.items():
            applied = min(node.activation, lateral)
            if applied <= 0:
                continue
            node.inhibition = _clamp(node.inhibition + applied)
            node.activation = _clamp(node.activation - applied)
            events.append(
                _event(
                    state,
                    now,
                    "inhibit",
                    page_id=page_id,
                    delta=round(-applied, 6),
                    activation=round(node.activation, 6),
                    reason_code="global_lateral_inhibition",
                    components={"inhibition": round(node.inhibition, 6)},
                )
            )

    retained = sorted(
        buffer.items(),
        key=lambda item: (
            -effective_activation(
                state,
                page_id=item[0],
                node=item[1],
                buffer_name=buffer_name,
            ),
            item[0],
        ),
    )[: config.max_active_nodes]
    buffer.clear()
    buffer.update(retained)
    state.updated_at_epoch = now
    if state.created_at_epoch <= 0:
        state.created_at_epoch = now
    return state, events


def run_field_turn(
    *,
    host: str,
    session_id: str,
    prompt: str,
    config: RecallFieldConfig | None = None,
    store: RecallFieldStore | None = None,
    now: float | None = None,
    session_hash_override: str = "",
) -> dict[str, Any]:
    """Run one private Field turn without changing Recall candidates."""

    started = time.perf_counter()
    cfg = _effective_config(config or load_recall_field_config())
    hashed_session = (
        session_hash_override
        if len(session_hash_override) == 16
        and all(char in "0123456789abcdef" for char in session_hash_override.casefold())
        else session_hash(host, session_id)
    )
    if cfg.mode == "off" or not hashed_session:
        return {
            "status": "disabled",
            "mode": cfg.mode,
            "reason": "missing_session" if not hashed_session else "mode_off",
            "latency_ms": int(round((time.perf_counter() - started) * 1_000)),
        }
    observed = time.time() if now is None else now
    field_store = store or RecallFieldStore(config=cfg)
    stimuli = prompt_stimuli(prompt)
    signature = topic_signature(prompt)

    def mutate(
        state: RecallFieldState,
    ) -> tuple[RecallFieldState, list[FieldEvent]]:
        state.host = host.strip().casefold()
        return update_field_state(
            state,
            stimuli=stimuli,
            prompt_signature=signature,
            config=cfg,
            now=observed,
            prompt_text=prompt,
        )

    state, events = field_store.transact(hashed_session, mutate, now=observed)
    buffer_name = "shadow" if cfg.mode == "shadow" else "active"
    buffer = state.shadow if buffer_name == "shadow" else state.active
    top = sorted(
        [
            item
            for item in buffer.items()
            if effective_activation(
                state,
                page_id=item[0],
                node=item[1],
                buffer_name=buffer_name,
            )
            > 0.01
        ],
        key=lambda item: (
            -effective_activation(
                state,
                page_id=item[0],
                node=item[1],
                buffer_name=buffer_name,
            ),
            item[0],
        ),
    )[: cfg.working_set_size]
    return {
        "status": "ok",
        "mode": cfg.mode,
        "session_hash": hashed_session,
        "topic_epoch": state.topic_epoch,
        "turn": state.turn,
        "seq": state.seq,
        "stimulus_count": len(stimuli),
        "event_count": len(events),
        "candidate_page_ids": [page_id for page_id, _node in top],
        "activations": [
            {
                "page_id": page_id,
                "activation": round(
                    effective_activation(
                        state,
                        page_id=page_id,
                        node=node,
                        buffer_name=buffer_name,
                    ),
                    6,
                ),
                "components": {
                    key: round(float(value), 6)
                    for key, value in asdict(node).items()
                    if key
                    in {
                        "direct",
                        "spread",
                        "negative",
                        "inhibition",
                        "anti_index",
                        "hub_penalty",
                    }
                },
            }
            for page_id, node in top
        ],
        "full_search_fallback": state.full_search_fallback,
        "latency_ms": int(round((time.perf_counter() - started) * 1_000)),
    }


def record_mcp_activity(
    *,
    host: str,
    session_id: str = "",
    page_ids: list[str],
    activity_kind: str,
    config: RecallFieldConfig | None = None,
    store: RecallFieldStore | None = None,
    now: float | None = None,
) -> dict[str, Any]:
    """Project an actual MCP search/read/record into the owning Field session."""

    started = time.perf_counter()
    cfg = _effective_config(config or load_recall_field_config())
    normalized_host = host.strip().casefold()
    observed = time.time() if now is None else now
    field_store = store or RecallFieldStore(config=cfg)
    hashed_session = (
        session_hash(normalized_host, session_id)
        if normalized_host and session_id.strip()
        else field_store.latest_session_hash(
            host=normalized_host,
            max_age_seconds=max(
                60.0,
                min(600.0, cfg.wall_half_life_seconds * 2.0),
            ),
            now=observed,
        )
    )
    unique_pages = [
        page_id
        for page_id in dict.fromkeys(str(value).strip() for value in page_ids)
        if page_id
    ][:5]
    if cfg.mode == "off" or not hashed_session or not unique_pages:
        return {
            "status": "skipped",
            "mode": cfg.mode,
            "reason": (
                "mode_off"
                if cfg.mode == "off"
                else "missing_session"
                if not hashed_session
                else "missing_pages"
            ),
            "latency_ms": int(round((time.perf_counter() - started) * 1_000)),
        }
    reason_code = {
        "read": "mcp_read",
        "record": "mcp_record",
    }.get(activity_kind, "mcp_search")
    if activity_kind == "read":
        weights = [1.0] * len(unique_pages)
    elif activity_kind == "record":
        weights = [max(0.52, 0.84 - index * 0.08) for index in range(len(unique_pages))]
    else:
        weights = [max(0.58, 0.92 - index * 0.08) for index in range(len(unique_pages))]
    stimuli = [
        FieldStimulus(
            page_id=page_id,
            kind=reason_code,
            weight=weight,
            reason_code=reason_code,
        )
        for page_id, weight in zip(unique_pages, weights, strict=True)
    ]

    def mutate(
        state: RecallFieldState,
    ) -> tuple[RecallFieldState, list[FieldEvent]]:
        if normalized_host and state.host and state.host != normalized_host:
            return state, []
        if normalized_host:
            state.host = normalized_host
        return update_field_state(
            state,
            stimuli=stimuli,
            prompt_signature=state.topic_signature,
            config=cfg,
            now=observed,
        )

    state, events = field_store.transact(hashed_session, mutate, now=observed)
    return {
        "status": "ok" if events else "skipped",
        "mode": cfg.mode,
        "session_hash": hashed_session,
        "host": state.host,
        "turn": state.turn,
        "seq": state.seq,
        "stimulus_count": sum(event.kind == "stimulus" for event in events),
        "event_count": len(events),
        "page_ids": unique_pages,
        "latency_ms": int(round((time.perf_counter() - started) * 1_000)),
    }


def _reviewed_negative_error(row: dict[str, Any]) -> str:
    """Validate the one direct negative-learning producer contract."""

    return trusted_negative_feedback_row_error(row)


def apply_reviewed_negative_feedback(
    row: dict[str, Any],
    *,
    config: RecallFieldConfig | None = None,
    store: RecallFieldStore | None = None,
    application_file: Path = NEGATIVE_APPLICATION_LEDGER,
    now: float | None = None,
) -> dict[str, Any]:
    """Apply one exact, reviewed negative directly to its owning Field session.

    Unlike ``record_mcp_activity``, this path has no unknown-positive fallback.
    The producer/digest key is stored inside the sealed session snapshot, making
    retries idempotent even if the append-only audit write was interrupted.
    """

    error = _reviewed_negative_error(row)
    if error:
        return {"status": "held", "reason": error, "applied": 0}
    cfg = _effective_config(config or load_recall_field_config())
    snapshot = row["snapshot"]
    host = str(row.get("host") or snapshot.get("host") or "")
    session_id = str(snapshot["session_id"])
    hashed_session = session_hash(host, session_id)
    if cfg.mode == "off" or not hashed_session:
        return {"status": "held", "reason": "field_unavailable", "applied": 0}
    producer_key = str(row.get("content_correction_key") or row.get("producer_key"))
    feedback_digest = feedback_row_sha256(row)
    evidence = snapshot.get("evidence_features")
    field_evidence = (
        evidence.get("field_shadow") if isinstance(evidence, dict) else None
    )
    expected_topic_epoch = (
        field_evidence.get("topic_epoch") if isinstance(field_evidence, dict) else None
    )
    expected_prompt_hash = str(snapshot.get("prompt_hash") or "")
    if (
        not isinstance(expected_topic_epoch, int)
        or isinstance(expected_topic_epoch, bool)
        or expected_topic_epoch < 0
        or str(field_evidence.get("session_hash") or "") != hashed_session
        or not expected_prompt_hash
    ):
        return {"status": "held", "reason": "field_topic_binding_missing", "applied": 0}
    contribution_key = hashlib.sha256(
        f"{producer_key}:{feedback_digest}:{expected_topic_epoch}:{expected_prompt_hash}".encode()
    ).hexdigest()
    pages = list(dict.fromkeys(str(page) for page in row["negative_pages"]))
    observed = time.time() if now is None else now
    field_store = store or RecallFieldStore(config=cfg)
    held_reason: list[str] = []

    def mutate(
        state: RecallFieldState,
    ) -> tuple[RecallFieldState, list[FieldEvent]]:
        if (
            state.topic_epoch != expected_topic_epoch
            or state.topic_prompt_hash != expected_prompt_hash
        ):
            held_reason.append("field_topic_epoch_or_query_mismatch")
            return state, []
        if contribution_key in state.negative_contributions:
            return state, []
        buffer_name = "shadow" if cfg.mode == "shadow" else "active"
        buffer = state.shadow if buffer_name == "shadow" else state.active
        state.pending_teacher_commits = [
            pending
            for pending in state.pending_teacher_commits
            if not (
                str(pending.get("page_id") or "") in pages
                and int(pending.get("topic_epoch", -1)) == expected_topic_epoch
            )
        ]
        page_weights: dict[str, float] = {}
        events: list[FieldEvent] = []
        for page_id in pages:
            node = buffer.setdefault(page_id, ActivationNode())
            composed_before = effective_activation(
                state,
                page_id=page_id,
                node=node,
                buffer_name=buffer_name,
            )
            node.last_seq = state.seq + 1
            page_weights[page_id] = 0.75
            events.append(
                _event(
                    state,
                    observed,
                    "inhibit",
                    page_id=page_id,
                    delta=-0.75,
                    activation=round(_clamp(composed_before - 0.75), 6),
                    reason_code="frontier_reviewed_negative",
                    components={
                        "producer_key": producer_key,
                        "feedback_sha256": feedback_digest,
                    },
                )
            )
        state.negative_contributions[contribution_key] = {
            "producer_key": producer_key,
            "feedback_sha256": feedback_digest,
            "page_hashes": dict(row["negative_page_hashes"]),
            "page_weights": page_weights,
            "buffer": buffer_name,
            "topic_epoch": expected_topic_epoch,
            "prompt_hash": expected_prompt_hash,
        }
        state.updated_at_epoch = observed
        return state, events

    state, events = field_store.transact(hashed_session, mutate, now=observed)
    if held_reason:
        return {"status": "held", "reason": held_reason[0], "applied": 0}
    ledger_row = {
        "schema_version": 1,
        "kind": "reviewed_negative_applied",
        "ts": datetime.fromtimestamp(observed, UTC).isoformat(),
        "session_hash": hashed_session,
        "producer_key": producer_key,
        "feedback_sha256": feedback_digest,
        "contribution_key": contribution_key,
        "page_ids": pages,
        "page_hashes": dict(row["negative_page_hashes"]),
        "applied": bool(events),
    }
    with sidecar_exclusive_lock(application_file):
        existing = {
            str(value.get("contribution_key") or "")
            for value in _read_application_rows(application_file)
            if value.get("kind") == "reviewed_negative_applied"
        }
        if contribution_key not in existing:
            append_jsonl_durable(application_file, [ledger_row], sort_keys=True)
    return {
        "status": "applied" if events else "duplicate",
        "applied": len(events),
        "session_hash": state.session_hash,
        "producer_key": producer_key,
        "feedback_sha256": feedback_digest,
        "contribution_key": contribution_key,
        "page_ids": pages,
    }


def _read_application_rows(path: Path) -> list[dict[str, Any]]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return []
    rows: list[dict[str, Any]] = []
    for line in lines:
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            rows.append(value)
    return rows


def retract_reviewed_negative_feedback(
    retraction: dict[str, Any],
    original: dict[str, Any],
    *,
    config: RecallFieldConfig | None = None,
    store: RecallFieldStore | None = None,
    application_file: Path = NEGATIVE_APPLICATION_LEDGER,
    now: float | None = None,
) -> dict[str, Any]:
    """Retract only the exact producer/digest contribution named by a tombstone."""

    producer_key = str(
        original.get("content_correction_key") or original.get("producer_key") or ""
    )
    feedback_digest = feedback_row_sha256(original)
    if (
        retraction.get("kind") != "page_ignored_retracted"
        or retraction.get("target_kind") != original.get("kind")
        or retraction.get("content_correction_key") != producer_key
        or retraction.get("target_feedback_sha256") != feedback_digest
    ):
        return {"status": "held", "reason": "retraction_binding_invalid", "restored": 0}
    snapshot = original.get("snapshot")
    if not isinstance(snapshot, dict):
        return {"status": "held", "reason": "missing_original_session", "restored": 0}
    cfg = _effective_config(config or load_recall_field_config())
    host = str(original.get("host") or snapshot.get("host") or "")
    hashed_session = session_hash(host, str(snapshot.get("session_id") or ""))
    if not hashed_session:
        return {"status": "held", "reason": "missing_original_session", "restored": 0}
    evidence = snapshot.get("evidence_features")
    field_evidence = (
        evidence.get("field_shadow") if isinstance(evidence, dict) else None
    )
    expected_topic_epoch = (
        field_evidence.get("topic_epoch") if isinstance(field_evidence, dict) else -1
    )
    expected_prompt_hash = str(snapshot.get("prompt_hash") or "")
    contribution_key = hashlib.sha256(
        f"{producer_key}:{feedback_digest}:{expected_topic_epoch}:{expected_prompt_hash}".encode()
    ).hexdigest()
    observed = time.time() if now is None else now
    field_store = store or RecallFieldStore(config=cfg)

    def mutate(
        state: RecallFieldState,
    ) -> tuple[RecallFieldState, list[FieldEvent]]:
        contribution = state.negative_contributions.get(contribution_key)
        if not isinstance(contribution, dict):
            return state, []
        if (
            contribution.get("topic_epoch") != state.topic_epoch
            or contribution.get("prompt_hash") != state.topic_prompt_hash
        ):
            del state.negative_contributions[contribution_key]
            return state, []
        buffer = (
            state.shadow if contribution.get("buffer") == "shadow" else state.active
        )
        buffer_name = "shadow" if contribution.get("buffer") == "shadow" else "active"
        del state.negative_contributions[contribution_key]
        events: list[FieldEvent] = []
        page_weights = contribution.get("page_weights", {})
        for page_id, raw_delta in page_weights.items() if isinstance(page_weights, dict) else []:
            if not isinstance(page_id, str) or not isinstance(raw_delta, int | float):
                continue
            delta = max(0.0, min(1.0, float(raw_delta)))
            node = buffer.get(page_id)
            if node is None:
                continue
            events.append(
                _event(
                    state,
                    observed,
                    "stimulus",
                    page_id=page_id,
                    delta=round(delta, 6),
                    activation=round(
                        effective_activation(
                            state,
                            page_id=page_id,
                            node=node,
                            buffer_name=buffer_name,
                        ),
                        6,
                    ),
                    reason_code="exact_negative_retraction",
                    components={
                        "producer_key": producer_key,
                        "feedback_sha256": feedback_digest,
                    },
                )
            )
        state.updated_at_epoch = observed
        return state, events

    state, events = field_store.transact(hashed_session, mutate, now=observed)
    if events:
        append_jsonl_durable(
            application_file,
            [
                {
                    "schema_version": 1,
                    "kind": "reviewed_negative_retracted",
                    "ts": datetime.fromtimestamp(observed, UTC).isoformat(),
                    "session_hash": state.session_hash,
                    "producer_key": producer_key,
                    "feedback_sha256": feedback_digest,
                    "contribution_key": contribution_key,
                    "retraction_sha256": feedback_row_sha256(retraction),
                }
            ],
            sort_keys=True,
        )
    return {
        "status": "retracted" if events else "duplicate_or_missing",
        "restored": len(events),
        "contribution_key": contribution_key,
    }


def sync_reviewed_negative_feedback(
    feedback_file: Path,
    *,
    config: RecallFieldConfig | None = None,
    store: RecallFieldStore | None = None,
) -> dict[str, Any]:
    """Replay active producers and exact tombstones into sealed Field state."""

    rows = read_jsonl_rows(feedback_file)
    originals: dict[tuple[str, str], dict[str, Any]] = {}
    tombstones: dict[tuple[str, str], dict[str, Any]] = {}
    applied = retracted = held = 0
    for row in rows:
        if row.get("kind") in {"page_ignored", "contradiction"}:
            producer = str(
                row.get("content_correction_key") or row.get("producer_key") or ""
            )
            digest = feedback_row_sha256(row)
            originals[(producer, digest)] = row
        elif row.get("kind") == "page_ignored_retracted":
            key = (
                str(row.get("content_correction_key") or ""),
                str(row.get("target_feedback_sha256") or ""),
            )
            if key in tombstones:
                held += 1
            else:
                tombstones[key] = row
    for key, original in originals.items():
        tombstone = tombstones.pop(key, None)
        if tombstone is not None:
            result = retract_reviewed_negative_feedback(
                tombstone, original, config=config, store=store
            )
            if result.get("status") in {"retracted", "duplicate_or_missing"}:
                retracted += 1
            else:
                held += 1
            continue
        result = apply_reviewed_negative_feedback(
            original, config=config, store=store
        )
        if result.get("status") in {"applied", "duplicate"}:
            applied += 1
        else:
            held += 1
    held += len(tombstones)
    return {
        "status": "ok" if held == 0 else "held",
        "applied_or_present": applied,
        "retracted_or_absent": retracted,
        "held": held,
    }


def record_mcp_content_activity(
    *,
    host: str,
    session_id: str = "",
    content: str,
    config: RecallFieldConfig | None = None,
    store: RecallFieldStore | None = None,
    now: float | None = None,
) -> dict[str, Any]:
    """Resolve a saved payload to bounded pages and emit a real record stimulus.

    Exact page/entity hints from the saved payload win.  When a Stop/save payload
    contains no exact hint, the current working set is used because that is the
    memory state the save operation is persisting; no unrelated global page is
    invented merely to create an animation.
    """

    cfg = _effective_config(config or load_recall_field_config())
    field_store = store or RecallFieldStore(config=cfg)
    observed = time.time() if now is None else now
    normalized_host = host.strip().casefold()
    hashed_session = (
        session_hash(normalized_host, session_id)
        if normalized_host and session_id.strip()
        else field_store.latest_session_hash(
            host=normalized_host,
            max_age_seconds=max(
                60.0,
                min(600.0, cfg.wall_half_life_seconds * 2.0),
            ),
            now=observed,
        )
    )
    if not hashed_session:
        return {"status": "skipped", "mode": cfg.mode, "reason": "missing_session"}
    exact_pages = [
        stimulus.page_id for stimulus in prompt_stimuli(content[-12_000:], limit=5)
    ]
    if not exact_pages:
        state = field_store.load(hashed_session, now=observed)
        buffer_name = "shadow" if cfg.mode == "shadow" else "active"
        buffer = state.shadow if buffer_name == "shadow" else state.active
        exact_pages = [
            page_id
            for page_id, _node in sorted(
                [
                    item
                    for item in buffer.items()
                    if effective_activation(
                        state,
                        page_id=item[0],
                        node=item[1],
                        buffer_name=buffer_name,
                    )
                    > 0.01
                ],
                key=lambda item: (
                    -effective_activation(
                        state,
                        page_id=item[0],
                        node=item[1],
                        buffer_name=buffer_name,
                    ),
                    item[0],
                ),
            )[:3]
        ]
    return record_mcp_activity(
        host=normalized_host,
        session_id=session_id,
        page_ids=exact_pages,
        activity_kind="record",
        config=cfg,
        store=field_store,
        now=observed,
    )


def queue_teacher_commits(
    *,
    host: str,
    session_id: str,
    page_ids: list[str],
    certificate_ids: dict[str, str] | None = None,
    ranking_components: dict[str, dict[str, Any]] | None = None,
    config: RecallFieldConfig | None = None,
    store: RecallFieldStore | None = None,
    now: float | None = None,
    session_hash_override: str = "",
) -> dict[str, Any]:
    """Queue teacher commits for the next turn without current-turn activation."""

    cfg = _effective_config(config or load_recall_field_config())
    hashed_session = (
        session_hash_override
        if len(session_hash_override) == 16
        and all(char in "0123456789abcdef" for char in session_hash_override.casefold())
        else session_hash(host, session_id)
    )
    if cfg.mode == "off" or not hashed_session or not page_ids:
        return {"status": "skipped", "queued": 0}
    observed = time.time() if now is None else now
    field_store = store or RecallFieldStore(config=cfg)
    certs = certificate_ids or {}
    components_by_page = ranking_components or {}

    def mutate(
        state: RecallFieldState,
    ) -> tuple[RecallFieldState, list[FieldEvent]]:
        buffer_name = "shadow" if cfg.mode == "shadow" else "active"
        suppressed = _suppressed_pages(state, buffer_name=buffer_name)
        existing = {
            (str(row.get("page_id") or ""), int(row.get("topic_epoch", -1)))
            for row in state.pending_teacher_commits
        }
        events: list[FieldEvent] = []
        for page_id in dict.fromkeys(page_ids):
            key = (page_id, state.topic_epoch)
            if not page_id or key in existing or page_id in suppressed:
                continue
            state.pending_teacher_commits.append(
                {
                    "page_id": page_id,
                    "certificate_id": certs.get(page_id, ""),
                    "topic_epoch": state.topic_epoch,
                    "available_turn": state.turn + 1,
                    "components": {
                        key: round(
                            float(components_by_page[page_id].get(key) or 0.0),
                            6,
                        )
                        for key in ("anti_index", "hub_penalty")
                        if isinstance(components_by_page.get(page_id), dict)
                        and isinstance(
                            components_by_page[page_id].get(key),
                            int | float,
                        )
                    },
                }
            )
            events.append(
                _event(
                    state,
                    observed,
                    "commit_queued",
                    page_id=page_id,
                    reason_code="teacher_commit_next_turn",
                    certificate_id=certs.get(page_id, ""),
                    components={
                        key: round(
                            float(components_by_page[page_id].get(key) or 0.0),
                            6,
                        )
                        for key in ("anti_index", "hub_penalty")
                        if isinstance(components_by_page.get(page_id), dict)
                        and isinstance(
                            components_by_page[page_id].get(key),
                            int | float,
                        )
                    },
                )
            )
        state.updated_at_epoch = observed
        return state, events

    state, events = field_store.transact(hashed_session, mutate, now=observed)
    return {
        "status": "ok",
        "queued": len(events),
        "session_hash": hashed_session,
        "topic_epoch": state.topic_epoch,
        "seq": state.seq,
    }
