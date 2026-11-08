"""On-device decision model as a rerank backend: StartLux-Decision Instant on MLX.

A decision model answers a yes/no question about a state in one forward pass: the full-attention layers read the prompt
bidirectionally, and the answer is read from the option-letter logits at the last token. The score of a candidate is
logit P(yes) = (z_yes - z_no) / T, so ``apply_reranker_scores`` orders by it unchanged. It is not a BGE logit: do not
feed it to heuristics calibrated on BGE.

Prompts, letters, temperatures and the model loader (with its bidirectional self-check) are the ``startlux_decide``
package's own, and so is the forward: one prompt per forward, exactly as the package reads it. Batching was measured
and dropped: on Apple GPUs this model is compute-bound, so a padded batch is no faster than the same prompts one by one
(4B: 10 search candidates ~2 s either way). Every MLX call runs on the package's single owner thread (MLX streams
belong to the thread that made them).
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
QUESTIONS = {SEARCH_PURPOSE: "Is this memory relevant to the search query?"}


@dataclass(frozen=True)
class DecisionModelConfig:
    path: str
    # Search accuracy was measured at 400 (4B: MRR at parity with BGE); every section is cut to it, which also bounds
    # the forward (cost is linear in tokens).
    max_chars: int = 400
    # Immutable identity of the installed decision artifact; used in service identity.
    revision: str | None = None


def decision_state(request: RerankRequest, candidate: str, max_chars: int) -> str:
    # shortcut: calibrated head truncation; reassess with long-query search evaluation.
    return f"Search query:\n{request.query[:max_chars]}\n\nCandidate memory:\n{candidate[:max_chars]}"


class DecisionMLXBackend:
    provider = "local-decision"
    location = RouteLocation.LOCAL

    def __init__(self, config: DecisionModelConfig) -> None:
        self.config = config
        self._engine: Any = None
        self._folder: Any = None
        self._letters: list[int] = []

    def _load(self) -> None:
        from startlux_decide.backends.mlx import (  # type: ignore[import-not-found, unused-ignore]
            MLXBackend,
        )
        from startlux_decide.modelcfg import (  # type: ignore[import-not-found, unused-ignore]
            load_model_folder,
            load_tokenizer,
        )
        from startlux_decide.pipeline import (  # type: ignore[import-not-found, unused-ignore]
            Decider,
        )

        folder = load_model_folder(self.config.path)
        if not folder.bidirectional:
            raise RuntimeError(
                "decision model folder is not an Instant (one-pass) model"
            )
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
        """logit P(yes) per state through the package's own one-prompt read-out. Runs on the engine's owner thread."""

        from startlux_decide.prompt import (  # type: ignore[import-not-found, unused-ignore]
            from_systemone,
            messages,
            plain_prompt,
        )

        engine = self._engine
        temperature = float(self._folder.temperature("noul"))
        scores = []
        for state in states:
            row = from_systemone(state, {"type": "noul", "instructions": question})
            option_ids = [option["id"] for option in row["options"]]
            z = engine._readout(
                plain_prompt(engine.tokenizer, messages(row)),
                self._letters[: len(option_ids)],
            )
            scores.append(
                (z[option_ids.index("true")] - z[option_ids.index("false")])
                / temperature
            )
        if any(not math.isfinite(score) for score in scores):
            raise RuntimeError("decision model returned a non-finite score")
        return scores
