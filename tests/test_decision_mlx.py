from __future__ import annotations

import sys
import types
from concurrent.futures import ThreadPoolExecutor

import pytest

from chronovisor.core import decision_mlx, llm_config
from chronovisor.core.decision_mlx import (
    ANSWER_AXES,
    ANSWER_AXIS_QUESTIONS,
    JUDGE_ANSWER_AXES,
    JUDGE_CERTIFICATE,
    JUDGE_POLICY_SHA256,
    JUDGE_ROLE_BY_KIND,
    JUDGE_UTILITY,
    DecisionMLXBackend,
    DecisionModelConfig,
    DecisionRoute,
    answer_axis_scores,
    certificate_verdict,
    decision_fingerprint,
    decision_route,
    decision_state,
    judge_state,
    logit_probability,
    utility_verdict,
)
from chronovisor.core.llm_config import (
    LLMConfigError,
    build_llm_runtime,
    parse_llm_config,
)
from chronovisor.core.llm_runtime import (
    RerankRequest,
    SourceDataClass,
    SourceDataClassification,
    SourceSensitivity,
)

SOURCE = SourceDataClassification(SourceDataClass.PAGE, SourceSensitivity.NORMAL)


def _config(provider: dict) -> dict:
    return {
        "llm": {
            "providers": {"decision": provider},
            "roles": {
                "rerank": {
                    "capability": "rerank",
                    "provider": "decision",
                    "model": "startlux-decision-4b",
                }
            },
        }
    }


def test_config_builds_a_decision_backend_with_defaults() -> None:
    config = parse_llm_config(_config({"kind": "local-decision-mlx", "path": "~/m"}))
    backend = build_llm_runtime(config)._rerank["rerank"].backend

    assert isinstance(backend, DecisionMLXBackend)
    assert backend.config.path.endswith("/m") and not backend.config.path.startswith(
        "~"
    )
    assert backend.config.max_chars == 400


@pytest.mark.parametrize(
    "provider",
    [
        {"kind": "local-decision-mlx"},
        {"kind": "local-decision-mlx", "path": ""},
        {"kind": "local-decision-mlx", "path": "/m", "device": "mps"},
        {"kind": "local-decision-mlx", "path": "/m", "batch_tokens": 8192},
    ],
)
def test_config_rejects_incomplete_or_unknown_decision_keys(provider) -> None:
    with pytest.raises(LLMConfigError):
        parse_llm_config(_config(provider))


def test_states_follow_the_purpose_and_cut_every_section() -> None:
    search = RerankRequest("dict sort", ("abcdef",), SOURCE)
    recall = RerankRequest(
        "and by value?",
        ("abcdef",),
        SOURCE,
        purpose="recall_filter",
        context="sort a dict",
    )

    assert decision_state(search, "abcdef", 3) == (
        "Search query:\ndic\n\nCandidate memory:\nabc"
    )
    assert decision_state(recall, "abcdef", 20) == (
        "Preceding user turn:\nsort a dict\n\nCurrent user turn:\nand by value?"
        "\n\nCandidate memory:\nabcdef"
    )
    assert decision_state(recall, "abcdef", 3) == (
        "Preceding user turn:\nsor\n\nCurrent user turn:\nand\n\nCandidate memory:\nabc"
    )
    no_context = RerankRequest("q", ("x",), SOURCE, purpose="recall_filter")
    assert decision_state(no_context, "x", 9).startswith("Current user turn:\nq")


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


def test_config_pins_the_decision_provider_revision() -> None:
    config = parse_llm_config(
        _config(
            {
                "kind": "local-decision-mlx",
                "path": "~/m",
                "revision": "a" * 64,
            }
        )
    )
    runtime = build_llm_runtime(config)
    route = runtime.resolve_rerank("rerank")
    backend = runtime._rerank["rerank"].backend

    assert route.revision == "a" * 64
    assert isinstance(backend, DecisionMLXBackend)
    assert backend.config.revision == "a" * 64


def test_config_rejects_a_malformed_decision_revision() -> None:
    with pytest.raises(LLMConfigError):
        parse_llm_config(
            _config(
                {
                    "kind": "local-decision-mlx",
                    "path": "/m",
                    "revision": "not-a-sha256",
                }
            )
        )


def test_judge_states_keep_labels_page_markers_and_bounds() -> None:
    answer_state = judge_state(
        JUDGE_ANSWER_AXES,
        query="q" * 401,
        answer="a" * 1301,
        reference="[PAGE page-1]\nfact one\n\n[PAGE page-2]\nfact two",
    )
    assert answer_state.startswith("Current question:\n" + "q" * 400)
    assert "Candidate answer:\n" + "a" * 1200 in answer_state
    assert "[PAGE page-1]\n[PAGE page-2]" in answer_state
    assert (
        len(
            answer_state.split("Candidate answer:\n", 1)[1].split(
                "\n\nReference evidence:", 1
            )[0]
        )
        == 1200
    )

    certificate_state = judge_state(
        JUDGE_CERTIFICATE,
        query="question",
        rubric="rubric",
        rubric_id="rubric-1",
        page_id="page-1",
        evidence="span",
    )
    assert "Adopted rubric (rubric-1):\nrubric" in certificate_state
    assert "Evidence (page page-1):\nspan" in certificate_state

    utility_state = judge_state(
        JUDGE_UTILITY,
        query="question",
        context="context",
        evidence_items=["first fact", "second fact"],
        answer="candidate",
    )
    assert utility_state == (
        "Current question:\nquestion\n\n"
        "Preceding context:\ncontext\n\n"
        "Evidence shown to the answerer:\nfirst fact\n\nsecond fact\n\n"
        "Candidate answer:\ncandidate"
    )
    no_evidence = judge_state(
        JUDGE_UTILITY, query="question", context="", answer="candidate"
    )
    assert "Preceding context:\n\n\n" in no_evidence
    assert "Evidence shown to the answerer:\n(none)\n\n" in no_evidence
    bounded = judge_state(
        JUDGE_UTILITY,
        query="q" * 2_001,
        context="c" * 401,
        evidence_items=["e" * 2_000],
        answer="a" * 801,
    )
    # Utility keeps the whole current question (2000), not the shared 400.
    assert bounded.startswith("Current question:\n" + "q" * 2_000 + "\n\n")
    assert "Preceding context:\n" + "c" * 400 + "\n\n" in bounded
    assert "Evidence shown to the answerer:\n" + "e" * 1_500 + "\n\n" in bounded
    assert bounded.endswith("Candidate answer:\n" + "a" * 800)
    with pytest.raises(ValueError):
        judge_state("unknown", query="q")


def test_utility_state_keeps_its_own_two_thousand_char_query() -> None:
    utility = judge_state(JUDGE_UTILITY, query="q" * 2_001, answer="a")
    assert utility.startswith("Current question:\n" + "q" * 2_000 + "\n\n")
    axes = judge_state(
        JUDGE_ANSWER_AXES, query="q" * 401, answer="a", reference="r"
    )
    assert axes.startswith(
        "Current question:\n" + "q" * decision_mlx.JUDGE_QUERY_CHARS + "\n\n"
    )


def test_judge_policy_and_logit_mappings_are_frozen() -> None:
    assert len(JUDGE_POLICY_SHA256) == 64
    assert decision_mlx.JUDGE_POLICY["questions"][JUDGE_ANSWER_AXES] == (
        ANSWER_AXIS_QUESTIONS
    )
    assert logit_probability(0.0) == 0.5
    assert logit_probability(60.0) > 0.999
    assert logit_probability(-60.0) < 0.001
    with pytest.raises(ValueError):
        logit_probability(float("nan"))

    axes = answer_axis_scores((2.0, -2.0, 0.0))
    assert list(axes) == list(ANSWER_AXES)
    assert axes["citation"] == 0.5
    with pytest.raises(ValueError):
        answer_axis_scores((0.0,))

    assert certificate_verdict("page-1", 2.5)["decision"] == "pass"
    assert certificate_verdict("page-1", -2.5)["decision"] == "reject"
    uncertain = certificate_verdict("page-1", 0.0)
    assert uncertain["decision"] == "uncertain"
    assert uncertain["confidence"] == 0.5

    assert decision_mlx.JUDGE_POLICY["utility_band"] == decision_mlx.UTILITY_BAND
    assert (
        decision_mlx.JUDGE_POLICY["limits"]["utility_query_chars"]
        == decision_mlx.UTILITY_QUERY_CHARS
        == 2_000
    )
    assert utility_verdict(2.5) == "helpful"
    assert utility_verdict(-2.5) == "harmful"
    assert utility_verdict(0.0) == "neutral"
    assert utility_verdict(0.5) == "neutral"
    assert utility_verdict(-0.5) == "neutral"
    assert utility_verdict(0.500001) == "helpful"
    assert utility_verdict(-0.500001) == "harmful"
    with pytest.raises(ValueError):
        utility_verdict(float("nan"))


def test_public_readout_runs_one_question_on_the_owner_thread(monkeypatch) -> None:
    backend = DecisionMLXBackend(DecisionModelConfig(path="/m"))
    owner = ThreadPoolExecutor(1)
    backend._engine = type("Engine", (), {"_executor": owner})()
    seen: dict[str, object] = {}

    def readout(states: list[str], question: str) -> list[float]:
        seen.update(states=states, question=question)
        return [1.0, 2.0]

    monkeypatch.setattr(backend, "_readout", readout)
    scores = backend.readout("Q?", ("one", "two"))
    owner.shutdown()

    assert scores == (1.0, 2.0)
    assert seen == {"states": ["one", "two"], "question": "Q?"}
    with pytest.raises(ValueError):
        backend.readout("Q?", ())


def test_decision_route_pins_revision_and_fingerprint(monkeypatch) -> None:
    route = DecisionRoute(
        role=JUDGE_ROLE_BY_KIND[JUDGE_ANSWER_AXES],
        provider="local-decision",
        model="startlux-model",
        location="local",
        revision="a" * 64,
    )

    class Runtime:
        def resolve_rerank(self, role: str):
            assert role == route.role
            return type(
                "Resolved",
                (),
                {
                    "role": route.role,
                    "provider": route.provider,
                    "model": route.model,
                    "location": decision_mlx.RouteLocation.LOCAL,
                    "revision": route.revision,
                },
            )()

    monkeypatch.setattr(llm_config, "load_default_llm_runtime", lambda: Runtime())
    assert decision_route(route.role) == route
    fingerprint = decision_fingerprint(route)
    assert len(fingerprint) == 64
    assert (
        decision_fingerprint(
            DecisionRoute(
                role=route.role,
                provider=route.provider,
                model=route.model,
                location=route.location,
                revision="b" * 64,
            )
        )
        != fingerprint
    )

    class UnpinnedRuntime(Runtime):
        def resolve_rerank(self, role: str):
            resolved = super().resolve_rerank(role)
            resolved.revision = None
            return resolved

    monkeypatch.setattr(
        llm_config, "load_default_llm_runtime", lambda: UnpinnedRuntime()
    )
    with pytest.raises(RuntimeError):
        decision_route(route.role)


@pytest.mark.parametrize(
    "capability,kind,expected",
    [
        ("generation", "mtplx", False),
        ("rerank", "local-decision-mlx", True),
        ("generation", "local-decision-mlx", False),
        ("rerank", "omlx", False),
    ],
)
def test_decision_judge_dispatch_uses_configured_role(
    monkeypatch, capability, kind, expected
):
    from types import SimpleNamespace

    from chronovisor.core import llm_config
    from chronovisor.core.decision_mlx import uses_decision_judge

    monkeypatch.setattr(
        llm_config,
        "load_llm_config",
        lambda: SimpleNamespace(
            roles={
                "judge": SimpleNamespace(capability=capability, provider_id="local")
            },
            providers={"local": SimpleNamespace(kind=kind)},
        ),
    )
    assert uses_decision_judge("judge") is expected
    assert uses_decision_judge("missing") is False
