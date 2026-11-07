"""On-device decision model as a rerank backend: StartLux-Decision Instant on MLX.

A decision model answers a yes/no question about a state in one forward pass: the full-attention layers read the prompt
bidirectionally, and the answer is read from the option-letter logits at the last token. The score of a candidate is
logit P(yes) = (z_yes - z_no) / T, so ``apply_reranker_scores`` orders by it unchanged. It is not a BGE logit: do not
feed it to heuristics calibrated on BGE.

Prompts, letters, temperatures and the model loader (with its bidirectional self-check) are the ``startlux_decide``
package's own. The package's MLX engine reads one prompt per forward; ``_readout`` reads a micro-batch per forward
instead (right padding, key-padding mask in the full-attention layers, each row read at its own last token). Batch
size 1 is identical to the package; larger batches differ by bf16 noise only. Every MLX call runs on the package's
single owner thread (MLX streams belong to the thread that made them).
"""

from __future__ import annotations

import asyncio
import math
from dataclasses import dataclass
from typing import Any

from chronovisor.core.llm_runtime import (
    RerankItem,
    RerankRequest,
    RerankResult,
    RouteLocation,
)

SEARCH_PURPOSE = "search"
RECALL_GATE_PURPOSE = "recall_gate"
QUESTIONS = {
    SEARCH_PURPOSE: "Is this memory relevant to the search query?",
    # Wording frozen with the recall-gate threshold (laya run3 dev fit, strict_en).
    RECALL_GATE_PURPOSE: (
        "Does the candidate memory contain information that would help the assistant respond to the "
        "current user turn, considering the preceding user turn? Topic or keyword overlap alone does "
        "not count."
    ),
}


@dataclass(frozen=True)
class DecisionModelConfig:
    path: str
    max_chars: int = 700
    batch_tokens: int = 8192


def decision_state(request: RerankRequest, candidate: str, max_chars: int) -> str:
    text = candidate[:max_chars]
    if request.purpose == RECALL_GATE_PURPOSE:
        sections = []
        if request.context:
            sections.append("Preceding user turn:\n" + request.context)
        sections.append("Current user turn:\n" + request.query)
        sections.append("Candidate memory:\n" + text)
        return "\n\n".join(sections)
    return f"Search query:\n{request.query}\n\nCandidate memory:\n{text}"


def micro_batches(lengths: list[int], budget: int) -> list[list[int]]:
    """Indices grouped so that each group's padded size (rows x longest row) stays within budget."""

    order = sorted(range(len(lengths)), key=lambda i: lengths[i])
    groups: list[list[int]] = []
    for index in order:
        group = groups[-1] if groups else None
        if group and (len(group) + 1) * lengths[index] <= budget:
            group.append(index)
        else:
            groups.append([index])
    return groups


class DecisionMLXBackend:
    provider = "local-decision"
    location = RouteLocation.LOCAL

    def __init__(self, config: DecisionModelConfig) -> None:
        self.config = config
        self._engine: Any = None
        self._folder: Any = None
        self._letters: list[int] = []

    def _load(self) -> None:
        from startlux_decide.backends.mlx import MLXBackend
        from startlux_decide.modelcfg import load_model_folder, load_tokenizer
        from startlux_decide.pipeline import Decider

        folder = load_model_folder(self.config.path)
        if not folder.bidirectional:
            raise RuntimeError("decision model folder is not an Instant (one-pass) model")
        engine = MLXBackend(self.config.path, load_tokenizer(self.config.path), True)
        loop = asyncio.new_event_loop()
        try:
            loop.run_until_complete(engine.start())
        finally:
            loop.close()
        self._letters = list(Decider(folder, engine).letter_ids)
        self._folder, self._engine = folder, engine

    def rerank(self, request: RerankRequest, *, model: str) -> RerankResult:
        question = QUESTIONS.get(request.purpose)
        if question is None:
            raise ValueError("unknown decision purpose")
        if self._engine is None:
            self._load()
        states = [
            decision_state(request, candidate, self.config.max_chars)
            for candidate in request.candidates
        ]
        scores = self._engine._executor.submit(self._readout, states, question).result()
        items = sorted(
            (RerankItem(index=i, score=float(s)) for i, s in enumerate(scores)),
            key=lambda item: (-item.score, item.index),
        )
        return RerankResult(items=tuple(items), provider=self.provider, model=model)

    def _readout(self, states: list[str], question: str) -> list[float]:
        """logit P(yes) per state. Runs on the engine's owner thread."""

        import mlx.core as mx
        import numpy as np
        from startlux_decide.prompt import from_systemone, messages, plain_prompt

        engine = self._engine
        tokenizer = engine.tokenizer
        rows = [from_systemone(state, {"type": "noul", "instructions": question}) for state in states]
        option_ids = [option["id"] for option in rows[0]["options"]]
        yes, no = option_ids.index("true"), option_ids.index("false")
        letters = engine._letter_rows(self._letters[: len(option_ids)])
        temperature = float(self._folder.temperature("noul"))
        pad = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
        ids = [
            np.asarray(
                tokenizer.encode(plain_prompt(tokenizer, messages(row)), add_special_tokens=False),
                dtype=np.int32,
            )
            for row in rows
        ]
        scores = [0.0] * len(ids)
        for group in micro_batches([len(x) for x in ids], self.config.batch_tokens):
            width = max(len(ids[i]) for i in group)
            tokens = np.full((len(group), width), pad, dtype=np.int32)
            valid = np.zeros((len(group), width), dtype=bool)
            for row, index in enumerate(group):
                tokens[row, : len(ids[index])] = ids[index]
                valid[row, : len(ids[index])] = True
            h = engine.body.embed_tokens(mx.array(tokens))
            positions = mx.array(
                np.broadcast_to(np.arange(width, dtype=np.int32), (3, len(group), width)).copy()
            )
            keep = mx.array(valid)
            key_mask = keep[:, None, None, :]
            first = next(layer for layer in engine.layers if not layer.is_linear)
            shared = (
                None
                if first.self_attn.rotary_emb.fused_apply
                else first.self_attn.rotary_emb(h, positions)
            )
            for layer in engine.layers:
                if layer.is_linear:  # causal recurrence; pads come after every real token
                    h = layer(h, mask=keep, cache=None)
                else:
                    h = layer(
                        h,
                        mask=key_mask,
                        cache=None,
                        position_ids=positions,
                        position_embeddings=shared,
                    )
            h = engine.body.norm(h)
            last = h[mx.arange(len(group)), mx.array([len(ids[i]) - 1 for i in group])]
            z = (last.astype(mx.float32) @ letters.T).tolist()
            for row, index in enumerate(group):
                scores[index] = (z[row][yes] - z[row][no]) / temperature
        if any(not math.isfinite(score) for score in scores):
            raise RuntimeError("decision model returned a non-finite score")
        return scores
