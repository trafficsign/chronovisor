from __future__ import annotations

import sys
import types
from concurrent.futures import ThreadPoolExecutor

import pytest

from chronovisor.core import decision_mlx
from chronovisor.core.decision_mlx import DecisionMLXBackend, DecisionModelConfig
from chronovisor.core.llm_runtime import (
    RerankRequest,
    SourceDataClass,
    SourceDataClassification,
    SourceSensitivity,
)

SOURCE = SourceDataClassification(SourceDataClass.PAGE, SourceSensitivity.NORMAL)


def test_readout_is_the_package_read_out_per_state(monkeypatch) -> None:
    backend = DecisionMLXBackend(DecisionModelConfig(path="/m"))
    prompts: list[str] = []

    class Engine:
        tokenizer = None

        def _readout(self, prompt, letters):
            prompts.append(prompt)
            # option A = yes = "true", B = no = "false" (the package's noul layout)
            return [3.0, 1.0] if "good" in prompt else [0.0, 2.0]

    class Folder:
        def temperature(self, kind):
            assert kind == "noul"
            return 2.0

    backend._engine, backend._folder, backend._letters = Engine(), Folder(), [7, 8, 9]
    # The package is a service-only dependency; its prompt helpers are faked here
    # and the real read-out is checked against the package on the model itself.
    prompt_module = types.ModuleType("startlux_decide.prompt")
    prompt_module.from_systemone = lambda state, spec: {
        "state": state,
        "q": spec["instructions"],
        "options": [{"id": "true"}, {"id": "false"}],
    }
    prompt_module.messages = lambda row: f"{row['state']}|{row['q']}"
    prompt_module.plain_prompt = lambda _tokenizer, text: text
    monkeypatch.setitem(
        sys.modules, "startlux_decide", types.ModuleType("startlux_decide")
    )
    monkeypatch.setitem(sys.modules, "startlux_decide.prompt", prompt_module)

    scores = backend._readout(["good one", "bad one"], "Q?")

    assert scores == [1.0, -1.0]
    assert len(prompts) == 2 and "good one" in prompts[0] and "Q?" in prompts[0]


def test_rerank_orders_by_score_on_the_owner_thread(monkeypatch) -> None:
    backend = DecisionMLXBackend(DecisionModelConfig(path="/m"))
    owner = ThreadPoolExecutor(1)
    backend._engine = type("Engine", (), {"_executor": owner})()
    seen: dict = {}

    def readout(states, question):
        seen.update(states=states, question=question)
        return [-1.0, 2.0, 2.0]

    monkeypatch.setattr(backend, "_readout", readout)
    result = backend.rerank(RerankRequest("q", ("a", "b", "c"), SOURCE), model="m")
    owner.shutdown()

    assert [(item.index, item.score) for item in result.items] == [
        (1, 2.0),
        (2, 2.0),
        (0, -1.0),
    ]
    assert seen["question"] == decision_mlx.QUESTIONS["search"]
    assert seen["states"][0].endswith("Candidate memory:\na")


def test_rerank_rejects_an_unknown_purpose() -> None:
    backend = DecisionMLXBackend(DecisionModelConfig(path="/m"))

    with pytest.raises(ValueError):
        backend.rerank(RerankRequest("q", ("a",), SOURCE, purpose="other"), model="m")
