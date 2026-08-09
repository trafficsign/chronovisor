"""Regression tests for the ingest pipeline.

Each test pins a bug we shipped at least once. Do not delete one without
replacing it with something that catches the same class of mistake.
"""

from __future__ import annotations

import base64
import hashlib
import json
import os
from contextlib import contextmanager
from datetime import date, datetime
from pathlib import Path

import pytest

from chronovisor.core import ollama
from chronovisor.ingest.ingest import (
    IngestApplyError,
    _apply_operations,
    _extract_json_array,
    _extract_page_body,
    _has_frontmatter,
    _reconcile_links,
    _strip_all_frontmatter,
)
from chronovisor.raw import record_raw as raw_record

# ---------------------------------------------------------------------------
# _extract_json_array
# ---------------------------------------------------------------------------


class TestExtractJsonArray:
    def test_plain_array(self) -> None:
        assert _extract_json_array("[]") == []
        assert _extract_json_array('[{"a":1}]') == [{"a": 1}]

    def test_preamble_stripped(self) -> None:
        assert _extract_json_array('---\n[{"type":"create"}]') == [{"type": "create"}]
        assert _extract_json_array('Here is the plan:\n[{"x":1}]') == [{"x": 1}]

    def test_postamble_with_brackets(self) -> None:
        # rfind-based extractor would grab the [done] bracket. Lexer must not.
        out = _extract_json_array('[{"x":1}]\nNote: [done]')
        assert out == [{"x": 1}]

    def test_preamble_bracket_then_array(self) -> None:
        out = _extract_json_array(
            'Note [not json]\n[{"type":"create","filename":"ok.md"}]'
        )
        assert out == [{"type": "create", "filename": "ok.md"}]

    def test_literal_close_bracket_in_string(self) -> None:
        out = _extract_json_array('[{"summary":"see [doc]"}]')
        assert out == [{"summary": "see [doc]"}]

    def test_markdown_fence(self) -> None:
        assert _extract_json_array('```json\n[{"a":1}]\n```') == [{"a": 1}]

    def test_safe_python_literal_fallback(self) -> None:
        assert _extract_json_array(
            "[{'type': 'create', 'filename': 'memory/fact.md', "
            "'keywords': ['fact'], 'enabled': True}]"
        ) == [
            {
                "type": "create",
                "filename": "memory/fact.md",
                "keywords": ["fact"],
                "enabled": True,
            }
        ]

    def test_python_only_literal_types_are_rejected(self) -> None:
        assert _extract_json_array("[{'keywords': ('not', 'json')}]") is None

    def test_object_not_array(self) -> None:
        assert _extract_json_array('{"a":1}') is None

    def test_empty_or_garbage(self) -> None:
        assert _extract_json_array("") is None
        assert _extract_json_array("I cannot help with that.") is None
        assert _extract_json_array("[not json]") is None

    def test_picks_longest_valid_array(self) -> None:
        # Two arrays in output; the second is longer and is what we want.
        out = _extract_json_array('[{"a":1}]\nthen the real plan:\n[{"a":1},{"b":2}]')
        assert out == [{"a": 1}, {"b": 2}]

    def test_truncated_outer_with_inner_keywords_returns_none(self) -> None:
        # Historical bug: when triage runs out of tokens mid-array, the
        # inner ``"keywords": [...]`` list is the longest *parseable* array
        # and used to be returned, silently routing truncated triage as
        # either "nothing wiki-worthy" (empty `[]`) or "schema invalid"
        # (string list). Both are wrong — the LLM had more to say.
        truncated_string_inner = (
            "[\n"
            '  {"type": "create", "filename": "a.md", "title": "A", '
            '"keywords": ["x", "y"], "summary": "..."},\n'
            '  {"type": "create", "filename": "b.md", "title": "B", '
            '"keywords": ['
        )
        assert _extract_json_array(truncated_string_inner) is None

        truncated_empty_inner = (
            "[\n"
            '  {"type": "create", "filename": "a.md", '
            '"keywords": []},\n'
            '  {"type": "create", "filename": "b.md", '
        )
        assert _extract_json_array(truncated_empty_inner) is None

    def test_truncated_outer_with_inner_dict_array_recovers(self) -> None:
        # Edge case: the outer array is broken but a complete dict-array
        # appears later. This *does* fit the contract, so accept it.
        text = 'broken [stuff\nlater: [{"type": "create", "filename": "a.md"}]'
        assert _extract_json_array(text) == [{"type": "create", "filename": "a.md"}]


# ---------------------------------------------------------------------------
# Frontmatter helpers
# ---------------------------------------------------------------------------


class TestFrontmatter:
    def test_has_frontmatter_positive(self) -> None:
        assert _has_frontmatter("---\ntitle: X\nupdated: 2026-04-28\n---\nbody")

    def test_has_frontmatter_negative_no_block(self) -> None:
        assert not _has_frontmatter("body without frontmatter")

    def test_has_frontmatter_negative_no_title(self) -> None:
        assert not _has_frontmatter("---\nupdated: 2026-04-28\n---\nbody")

    def test_strip_removes_all_blocks(self) -> None:
        text = "---\ntitle: A\n---\nbody1\n---\ntitle: B\n---\nbody2\n"
        assert "title:" not in _strip_all_frontmatter(text)


# ---------------------------------------------------------------------------
# _extract_page_body — op-type contracts
# ---------------------------------------------------------------------------


class TestExtractPageBody:
    def test_create_strict_with_frontmatter(self) -> None:
        out = _extract_page_body(
            "=== NEW PAGE: foo.md ===\n"
            "---\ntitle: Foo\nupdated: 2026-04-28\n---\n"
            "\nbody.\n"
            "=== END PAGE ===",
            op_type="create",
        )
        assert out is not None and out.startswith("---\ntitle: Foo")

    def test_create_lenient_with_frontmatter(self) -> None:
        # gemma-style: drops "NEW PAGE:" prefix
        out = _extract_page_body(
            "=== career/foo.md ===\n"
            "---\ntitle: Foo\nupdated: 2026-04-28\n---\n"
            "\nbody.\n"
            "=== END PAGE ===",
            op_type="create",
        )
        assert out is not None and "title: Foo" in out

    def test_create_rejects_no_frontmatter(self) -> None:
        # Wrapper present but body has no frontmatter → must reject so we
        # never persist refusals or malformed pages.
        out = _extract_page_body(
            "=== career/foo.md ===\nNo frontmatter here\n=== END PAGE ===",
            op_type="create",
        )
        assert out is None

    def test_create_rejects_empty(self) -> None:
        assert _extract_page_body("", op_type="create") is None
        assert (
            _extract_page_body(
                "=== NEW PAGE: foo.md ===\n\n=== END PAGE ===", op_type="create"
            )
            is None
        )

    def test_update_strips_stray_frontmatter(self) -> None:
        # The original Critical bug: model emits full-page output for an
        # update, which used to be appended verbatim → multi-frontmatter.
        out = _extract_page_body(
            "=== UPDATE PAGE: foo.md ===\n"
            "---\ntitle: Foo\nupdated: 2026-04-28\n---\n"
            "\n## new section\n\nnotes.\n"
            "=== END PAGE ===",
            op_type="update",
        )
        assert out is not None
        assert "title:" not in out
        assert "## new section" in out

    def test_update_rejects_frontmatter_only(self) -> None:
        out = _extract_page_body(
            "=== UPDATE PAGE: foo.md ===\n"
            "---\ntitle: Foo\nupdated: 2026-04-28\n---\n"
            "=== END PAGE ===",
            op_type="update",
        )
        assert out is None

    def test_create_truncated_no_close_rejected(self) -> None:
        out = _extract_page_body(
            "=== NEW PAGE: personal/smoking-habit-analysis.md ===\n"
            "---\ntitle: Smoking Habit Analysis\nupdated: 2026-04-29\n---\n"
            "\n# 概要\n\nニコチンの半減期は短い",
            op_type="create",
        )
        assert out is None

    def test_create_truncated_partial_close_fence_rejected(self) -> None:
        out = _extract_page_body(
            "=== NEW PAGE: foo.md ===\n"
            "---\ntitle: Foo\nupdated: 2026-04-28\n---\n"
            "\nbody text.\n=== EN",
            op_type="create",
        )
        assert out is None

    def test_create_generic_wrapper_without_close_rejected(self) -> None:
        out = _extract_page_body(
            "=== user-profile-background PAGE: user-profile-background ===\n"
            "---\ntitle: User Profile\nupdated: 2026-04-29\n---\n"
            "\nbody",
            op_type="create",
        )
        assert out is None

    def test_update_truncated_no_close_rejected(self) -> None:
        out = _extract_page_body(
            "=== UPDATE PAGE: foo.md ===\n\n## new section\n\nnotes...",
            op_type="update",
        )
        assert out is None

    def test_update_bare_markdown_without_close_rejected(self) -> None:
        out = _extract_page_body(
            "## 6.5 Stop Hook 最適化\n\n本文だけ返ってきたケース。",
            op_type="update",
        )
        assert out is None

    @pytest.mark.parametrize(
        "suffix",
        (
            "\ntrailing text after the terminal marker",
            "\n=== UPDATE PAGE: second.md ===\nsecond block\n=== END PAGE ===",
        ),
    )
    def test_terminal_marker_must_be_unique_and_final(self, suffix: str) -> None:
        out = _extract_page_body(
            "=== NEW PAGE: foo.md ===\n"
            "---\ntitle: Foo\nupdated: 2026-04-28\n---\n"
            "body\n=== END PAGE ===" + suffix,
            op_type="create",
        )
        assert out is None

    def test_create_truncated_broken_frontmatter_still_rejected(self) -> None:
        # Truncation BEFORE the closing "---" of the frontmatter cannot
        # be recovered: we'd persist a page with no proper frontmatter
        # block. The op_type contract check must still reject.
        out = _extract_page_body(
            "=== NEW PAGE: foo.md ===\n---\ntitle: Foo\nupdated: 2026",
            op_type="create",
        )
        assert out is None


def test_generate_one_supplies_current_date_and_forbids_date_inference(
    monkeypatch: pytest.MonkeyPatch,
    isolated_wiki: Path,
) -> None:
    from chronovisor.ingest import ingest

    captured: dict[str, str] = {}
    today = date.today().isoformat()

    monkeypatch.setattr(
        ingest,
        "_build_focused_context",
        lambda _op, _raw: "Focused context",
    )

    def fake_generate(prompt: str, *, system: str, progress_callback=None) -> str:
        captured["prompt"] = prompt
        captured["system"] = system
        return (
            "=== NEW PAGE: memory/date-contract.md ===\n"
            "---\n"
            "title: Date contract\n"
            f"updated: {today}\n"
            "tags: [d/tools-config, t/reference, s/2026]\n"
            "---\n\n"
            "本文\n"
            "=== END PAGE ==="
        )

    monkeypatch.setattr(ingest, "_generate_with_progress", fake_generate)

    result = ingest._generate_one(
        {
            "type": "create",
            "filename": "memory/date-contract.md",
            "title": "Date contract",
            "summary": "date handling",
        },
        "No date appears in this raw evidence.",
    )

    assert result is not None
    assert f"Current date: {today}" in captured["prompt"]
    assert "Do not add or infer any other date" in captured["prompt"]
    assert "exact current date" in captured["system"]


@pytest.mark.parametrize(
    ("op_type", "filename", "invalid", "valid"),
    [
        (
            "create",
            "memory/repaired-create.md",
            "=== NEW PAGE: memory/repaired-create.md ===\n"
            "---\ntitle: Repaired create\nupdated: 2026-07-14\n"
            "tags: [d/tools-config, t/reference, s/2026]\n---\n\n本文",
            "=== NEW PAGE: memory/repaired-create.md ===\n"
            "---\ntitle: Repaired create\nupdated: 2026-07-14\n"
            "tags: [d/tools-config, t/reference, s/2026]\n---\n\n本文\n"
            "=== END PAGE ===",
        ),
        (
            "update",
            "memory/repaired-update.md",
            "=== UPDATE PAGE: memory/repaired-update.md ===\n## 追記\n\n本文",
            "=== UPDATE PAGE: memory/repaired-update.md ===\n## 追記\n\n本文\n"
            "=== END PAGE ===",
        ),
    ],
)
def test_generate_one_repairs_missing_end_marker_in_same_logical_session(
    monkeypatch: pytest.MonkeyPatch,
    isolated_wiki: Path,
    op_type: str,
    filename: str,
    invalid: str,
    valid: str,
) -> None:
    from chronovisor.ingest import ingest

    prompts: list[str] = []
    responses = iter([invalid, valid])

    def fake_generate(prompt: str, **kwargs):
        prompts.append(prompt)
        return next(responses)

    monkeypatch.setattr(ingest, "_generate_with_progress", fake_generate)
    diagnostics: dict = {}

    result = ingest._generate_one(
        {
            "type": op_type,
            "filename": filename,
            "title": "Repaired",
            "summary": "repair the deterministic wrapper violation",
        },
        "grounded raw",
        diagnostics=diagnostics,
    )

    assert result is not None
    assert len(prompts) == 2
    assert f"<ASSISTANT>\n{invalid}" in prompts[1]
    assert "Validator errors:" in prompts[1]
    assert "code: missing_end_marker" in prompts[1]
    assert "must contain exactly one" in prompts[1]
    assert "final non-whitespace line" in prompts[1]
    if op_type == "update":
        assert "return only the new append body inside the wrapper" in prompts[1]
        assert "do not repeat, summarize, or rewrite" in prompts[1]
    assert diagnostics["attempts"] == 2
    assert diagnostics["repair_turns"] == 1


@pytest.mark.parametrize(
    ("op_type", "filename", "output"),
    [
        (
            "create",
            "memory/transport-complete-create.md",
            "=== NEW PAGE: memory/transport-complete-create.md ===\n"
            "---\ntitle: Complete create\nupdated: 2026-07-14\n"
            "tags: [d/tools-config, t/reference, s/2026]\n---\n\n本文",
        ),
        (
            "update",
            "memory/transport-complete-update.md",
            "=== UPDATE PAGE: memory/transport-complete-update.md ===\n## 追記\n\n本文",
        ),
    ],
)
def test_generate_one_restores_only_transport_attested_terminal_marker(
    monkeypatch: pytest.MonkeyPatch,
    isolated_wiki: Path,
    op_type: str,
    filename: str,
    output: str,
) -> None:
    from chronovisor.ingest import ingest

    calls = 0

    def fake_generate(_prompt: str, **kwargs):
        nonlocal calls
        calls += 1
        assert kwargs["return_metadata"] is True
        return ingest.ollama_runtime.GenerateResponse(
            content=output,
            done=True,
            done_reason="stop",
            prompt_eval_count=1_000,
            eval_count=200,
        )

    monkeypatch.setattr(ingest, "_generate_with_progress", fake_generate)
    diagnostics: dict = {}

    result = ingest._generate_one(
        {
            "type": op_type,
            "filename": filename,
            "title": "Transport complete",
            "summary": "restore one deterministic serialization boundary",
        },
        "grounded raw",
        diagnostics=diagnostics,
    )

    assert result is not None
    assert calls == 1
    assert diagnostics["attempts"] == 1
    assert diagnostics["repair_turns"] == 0
    assert diagnostics["transport_boundary_repaired"] is True


@pytest.mark.parametrize(
    "output",
    [
        ("=== UPDATE PAGE: memory/misplaced.md ===\nbody\n=== END PAGE ===\nextra"),
        "```markdown\n=== UPDATE PAGE: memory/open-fence.md ===\nbody",
        "=== UPDATE PAGE: memory/bad-wrapper.md ===\n",
        "=== UPDATE PAGE: memory/partial-marker.md ===\nbody\n=== END PA",
        (
            "=== UPDATE PAGE: memory/interior-partial-marker.md ===\n"
            "body\n=== END PA\ncontinued"
        ),
        "=== UPDATE PAGE: memory/short-fence.md ===\nbody\n== END PAGE ===",
        "=== UPDATE PAGE: memory/long-fence.md ===\nbody\n==== END PAGE ===",
        "=== UPDATE PAGE: memory/trailing-marker.md ===\nbody\n=== END PAGE === trailing",
        "=== UPDATE PAGE: memory/single-fence.md ===\nbody\n= END PAGE =",
        "=== UPDATE PAGE: memory/plural-marker.md ===\nbody\n=== END PAGES ===",
    ],
)
def test_transport_attested_boundary_repair_remains_fail_closed(
    output: str,
) -> None:
    from chronovisor.ingest import ingest

    response = ingest.ollama_runtime.GenerateResponse(
        content=output,
        done=True,
        done_reason="stop",
    )

    assert (
        ingest._repair_transport_attested_page_boundary(
            output,
            response,
            op_type="update",
        )
        is None
    )


def test_generate_one_stops_when_repair_repeats_same_invalid_output(
    monkeypatch: pytest.MonkeyPatch,
    isolated_wiki: Path,
) -> None:
    from chronovisor.ingest import ingest

    invalid = (
        "=== NEW PAGE: memory/repeated.md ===\n"
        "---\ntitle: Repeated\nupdated: 2026-07-14\n"
        "tags: [d/tools-config, t/reference, s/2026]\n---\n\n本文"
    )
    prompts: list[str] = []

    def fake_generate(prompt: str, **_kwargs) -> str:
        prompts.append(prompt)
        return invalid

    monkeypatch.setattr(ingest, "_generate_with_progress", fake_generate)
    diagnostics: dict = {}

    result = ingest._generate_one(
        {
            "type": "create",
            "filename": "memory/repeated.md",
            "title": "Repeated",
            "summary": "same output should terminate repair",
        },
        "raw",
        diagnostics=diagnostics,
    )

    assert result is None
    assert len(prompts) == 2
    assert diagnostics["failure_class"] == "repeated_output"
    assert diagnostics["attempts"] == 2


def test_generate_one_records_runtime_transport_failure_diagnostics(
    monkeypatch: pytest.MonkeyPatch,
    isolated_wiki: Path,
) -> None:
    from chronovisor.ingest import ingest

    def fake_generate(_prompt: str, **_kwargs) -> str:
        raise RuntimeError("socket reset")

    monkeypatch.setattr(ingest, "_generate_with_progress", fake_generate)
    diagnostics: dict = {}

    result = ingest._generate_one(
        {
            "type": "create",
            "filename": "memory/transport-error.md",
            "title": "Transport error",
            "summary": "Record the failed model turn",
        },
        "raw",
        diagnostics=diagnostics,
    )

    assert result is None
    assert diagnostics["failure_class"] == "transport_error"
    assert diagnostics["attempts"] == 1
    assert "socket reset" in diagnostics["reason"]


def test_generate_one_failure_logs_are_confined_to_isolated_wiki(
    monkeypatch: pytest.MonkeyPatch,
    isolated_wiki: Path,
) -> None:
    from chronovisor.core import runtime_status
    from chronovisor.ingest import ingest

    def fake_generate(_prompt: str, **_kwargs) -> str:
        raise RuntimeError("isolated transport sentinel")

    monkeypatch.setattr(ingest, "_generate_with_progress", fake_generate)
    result = ingest._generate_one(
        {
            "type": "create",
            "filename": "memory/isolated-log.md",
            "title": "Isolated log",
            "summary": "prove generation diagnostics stay in tmp wiki",
        },
        "raw",
    )

    assert result is None
    assert isolated_wiki / "log.md" == ingest.LOG_FILE
    assert isolated_wiki / "runtime" / "events.jsonl" == runtime_status.EVENTS_FILE
    assert "isolated transport sentinel" in ingest.LOG_FILE.read_text()
    assert "isolated transport sentinel" in runtime_status.EVENTS_FILE.read_text()


def test_generate_one_exhausts_after_two_distinct_targeted_repairs(
    monkeypatch: pytest.MonkeyPatch,
    isolated_wiki: Path,
) -> None:
    from chronovisor.ingest import ingest

    responses = iter(
        [
            "bare response one",
            "bare response two",
            "bare response three",
        ]
    )
    calls = 0

    def fake_generate(_prompt: str, **_kwargs) -> str:
        nonlocal calls
        calls += 1
        return next(responses)

    monkeypatch.setattr(ingest, "_generate_with_progress", fake_generate)
    diagnostics: dict = {}

    result = ingest._generate_one(
        {
            "type": "create",
            "filename": "memory/exhausted.md",
            "title": "Exhausted",
            "summary": "bounded repairs",
        },
        "raw",
        diagnostics=diagnostics,
    )

    assert result is None
    assert calls == 3
    assert diagnostics["failure_class"] == "repair_exhausted"
    assert diagnostics["attempts"] == 3


def test_generate_one_holds_one_exclusive_lease_across_repair_session(
    monkeypatch: pytest.MonkeyPatch,
    isolated_wiki: Path,
) -> None:
    from chronovisor.core.runtime_config import IngestConfig
    from chronovisor.ingest import ingest

    invalid = (
        "=== NEW PAGE: memory/leased.md ===\n"
        "本文だけでCREATE frontmatterがない\n"
        "=== END PAGE ==="
    )
    valid = (
        "=== NEW PAGE: memory/leased.md ===\n"
        "---\ntitle: Leased\nupdated: 2026-07-14\n"
        "tags: [d/tools-config, t/reference, s/2026]\n---\n\n本文\n"
        "=== END PAGE ==="
    )
    responses = iter([invalid, valid])
    events: list[tuple[str, object]] = []

    @contextmanager
    def fake_lease(*, exclusive: bool = False):
        events.append(("lease_enter", exclusive))
        try:
            yield
        finally:
            events.append(("lease_exit", exclusive))

    def fake_admit(_config, requested_num_ctx: int) -> int:
        events.append(("admit", requested_num_ctx))
        return requested_num_ctx

    def fake_generate(prompt: str, *, system: str | None = None, **_kwargs):
        events.append(("generate", prompt))
        return ingest.ollama_runtime.GenerateResponse(
            content=next(responses),
            done=True,
            done_reason="stop",
            prompt_eval_count=100,
            eval_count=100,
        )

    monkeypatch.setattr(
        ingest,
        "load_ingest_config",
        lambda: IngestConfig(
            model="ornith:test",
            num_ctx=32768,
            max_num_ctx=65536,
            num_predict=4096,
        ),
    )
    monkeypatch.setattr(ingest, "_build_focused_context", lambda *_a, **_k: "ctx")
    monkeypatch.setattr(ingest, "_admit_ingest_context", fake_admit)
    monkeypatch.setattr(ingest.ollama_runtime, "model_resource_lease", fake_lease)
    monkeypatch.setattr(ingest, "generate", fake_generate)

    assert ingest._generate_with_progress is ingest._DEFAULT_GENERATE_WITH_PROGRESS
    result = ingest._generate_one(
        {
            "type": "create",
            "filename": "memory/leased.md",
            "title": "Leased",
            "summary": "one lease must span repair turns",
        },
        "grounded raw",
    )

    assert result is not None
    assert [name for name, _value in events] == [
        "lease_enter",
        "admit",
        "generate",
        "generate",
        "lease_exit",
    ]
    assert events[0] == ("lease_enter", True)
    assert events[-1] == ("lease_exit", True)
    second_prompt = [value for name, value in events if name == "generate"][1]
    assert isinstance(second_prompt, str)
    assert f"<ASSISTANT>\n{invalid}" in second_prompt


def test_generate_one_rejects_context_accounting_at_admitted_boundary(
    monkeypatch: pytest.MonkeyPatch,
    isolated_wiki: Path,
) -> None:
    from chronovisor.core.runtime_config import IngestConfig
    from chronovisor.ingest import ingest

    calls = 0

    def fake_generate(_prompt: str, **kwargs):
        nonlocal calls
        calls += 1
        num_ctx = int(kwargs["num_ctx"])
        return ingest.ollama_runtime.GenerateResponse(
            content=(
                "=== NEW PAGE: memory/context-boundary.md ===\n"
                "---\ntitle: Context boundary\nupdated: 2026-07-14\n"
                "tags: [d/tools-config, t/reference, s/2026]\n---\n\n本文\n"
                "=== END PAGE ==="
            ),
            done=True,
            done_reason="stop",
            prompt_eval_count=num_ctx - 128,
            eval_count=64,
        )

    monkeypatch.setattr(
        ingest,
        "load_ingest_config",
        lambda: IngestConfig(
            model="ornith:test",
            num_ctx=32768,
            max_num_ctx=32768,
            num_predict=4096,
        ),
    )
    monkeypatch.setattr(ingest, "_build_focused_context", lambda *_a, **_k: "ctx")
    monkeypatch.setattr(ingest, "_generate_with_progress", fake_generate)
    diagnostics: dict = {}

    with pytest.raises(
        RuntimeError,
        match="ingest generation context_truncation_suspected",
    ):
        ingest._generate_one(
            {
                "type": "create",
                "filename": "memory/context-boundary.md",
                "title": "Context boundary",
                "summary": "valid bytes must still fail at the context boundary",
            },
            "grounded raw",
            diagnostics=diagnostics,
        )

    assert calls == 1
    assert diagnostics["failure_class"] == "context_truncation_suspected"
    assert diagnostics["attempts"] == 1


def test_generate_one_oversized_full_repair_envelope_fails_before_resources(
    monkeypatch: pytest.MonkeyPatch,
    isolated_wiki: Path,
) -> None:
    from chronovisor.core.runtime_config import IngestConfig
    from chronovisor.ingest import ingest

    def forbidden(*_args, **_kwargs):
        pytest.fail(
            "oversized generation must fail before lease, admission, or transport"
        )

    monkeypatch.setattr(
        ingest,
        "load_ingest_config",
        lambda: IngestConfig(
            model="ornith:test",
            num_ctx=32768,
            max_num_ctx=32768,
            num_predict=4096,
        ),
    )
    monkeypatch.setattr(ingest, "_build_focused_context", lambda *_a, **_k: "ctx")
    monkeypatch.setattr(ingest.ollama_runtime, "model_resource_lease", forbidden)
    monkeypatch.setattr(ingest, "_admit_ingest_context", forbidden)
    monkeypatch.setattr(ingest, "generate", forbidden)
    diagnostics: dict = {}

    with pytest.raises(RuntimeError, match="ingest generation context_window_exceeded"):
        ingest._generate_one(
            {
                "type": "create",
                "filename": "memory/oversized-envelope.md",
                "title": "Oversized envelope",
                "summary": "the complete repair envelope must be admitted up front",
            },
            "x" * 400_000,
            diagnostics=diagnostics,
        )

    assert diagnostics["failure_class"] == "context_window_exceeded"
    assert diagnostics["attempts"] == 0


# ---------------------------------------------------------------------------
# _reconcile_links — code-fence safety + resolve/rewrite/unwrap
# ---------------------------------------------------------------------------


class TestReconcileLinks:
    def setup_method(self) -> None:
        self.allowed = {"foo", "bar", "switchbot-hub-3-purchase-logic"}

    def test_resolve_intact(self) -> None:
        out, s = _reconcile_links(
            "See [[foo]] and [[bar#section|alias]].", self.allowed
        )
        assert out == "See [[foo]] and [[bar#section|alias]]."
        assert s["resolved"] == 2

    def test_folder_prefix_rewrite(self) -> None:
        out, s = _reconcile_links("Check [[personal/foo]] now.", self.allowed)
        assert out == "Check [[foo]] now."
        assert s["rewritten"] == 1

    def test_md_suffix_rewrite_with_anchor_and_alias(self) -> None:
        out, _s = _reconcile_links("[[bar.md#x|Bar]]", self.allowed)
        assert out == "[[bar#x|Bar]]"

    def test_unwrap_no_alias(self) -> None:
        out, s = _reconcile_links("Work at [[三菱重工]].", self.allowed)
        assert out == "Work at 三菱重工."
        assert s["unwrapped"] == 1

    def test_unwrap_with_alias(self) -> None:
        out, _s = _reconcile_links("[[ghost|display]]", self.allowed)
        assert out == "display"

    def test_fenced_code_is_untouched(self) -> None:
        # Critical: subscript / list indexing must not be eaten.
        text = "before\n```python\nx = data[[1]]\n```\nafter [[foo]] tail"
        out, s = _reconcile_links(text, self.allowed)
        assert "x = data[[1]]" in out
        assert "[[foo]]" in out  # outside fence still resolves
        assert s["resolved"] == 1
        assert s["unwrapped"] == 0

    def test_inline_code_is_untouched(self) -> None:
        text = "the regex `[[ghost]]` is a sample, but [[ghost]] is unresolved"
        out, _s = _reconcile_links(text, self.allowed)
        assert "`[[ghost]]`" in out  # inline code intact
        assert out.count("[[ghost]]") == 1  # only the inline one survived
        assert "is unresolved" in out and "ghost is unresolved" in out

    def test_frontmatter_is_untouched(self) -> None:
        # Frontmatter is data, not prose. We must not rewrite link-shaped values.
        text = "---\ntitle: [[ghost]]\n---\nbody [[foo]]"
        out, _s = _reconcile_links(text, self.allowed)
        assert "title: [[ghost]]" in out
        assert "[[foo]]" in out


# ---------------------------------------------------------------------------
# _apply_operations — fail-closed contracts
# ---------------------------------------------------------------------------


@pytest.fixture()
def isolated_wiki(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Point the whole package at a throw-away wiki under tmp_path.

    Every module that holds a copy of a wiki path constant gets patched.
    Without this an IndexStore.refresh() during the test would scan the
    operator's real Chronovisor corpus.
    """
    chronovisor_root = tmp_path / "wiki"
    pages = chronovisor_root / "pages"
    raw = chronovisor_root / "raw"
    system = chronovisor_root / "system"
    index_dir = chronovisor_root / ".index"
    for d in (pages, raw, system, index_dir):
        d.mkdir(parents=True, exist_ok=True)

    from chronovisor.core import (
        index_store,
        ollama,
        page_mutation,
        runtime_status,
        search,
        store,
    )
    from chronovisor.ingest import ingest, orchestrator

    monkeypatch.setattr(store, "CHRONOVISOR_ROOT", chronovisor_root)
    monkeypatch.setattr(store, "PAGES_DIR", pages)
    monkeypatch.setattr(store, "RAW_DIR", raw)
    monkeypatch.setattr(store, "SYSTEM_DIR", system)
    monkeypatch.setattr(store, "INDEX_FILE", chronovisor_root / "index.md")
    monkeypatch.setattr(store, "LOG_FILE", chronovisor_root / "log.md")
    monkeypatch.setattr(ingest, "PAGES_DIR", pages)
    monkeypatch.setattr(ingest, "INDEX_FILE", chronovisor_root / "index.md")
    monkeypatch.setattr(ingest, "LOG_FILE", chronovisor_root / "log.md")
    monkeypatch.setattr(orchestrator, "RAW_DIR", raw)
    monkeypatch.setattr(orchestrator, "CHRONOVISOR_ROOT", chronovisor_root)
    monkeypatch.setattr(
        orchestrator, "STATE_FILE", chronovisor_root / ".orchestrator_state.json"
    )
    monkeypatch.setattr(
        page_mutation,
        "CHRONOVISOR_MUTATION_LOCK",
        chronovisor_root / "runtime" / "wiki-mutation.lock",
    )
    monkeypatch.setattr(
        page_mutation,
        "DECISION_AUTHORITY_LOCK",
        chronovisor_root / "runtime" / "decision-authority.lock",
    )
    monkeypatch.setattr(page_mutation, "PAGES_DIR", pages)
    monkeypatch.setattr(runtime_status, "RUNTIME_DIR", chronovisor_root / "runtime")
    monkeypatch.setattr(
        runtime_status, "STATUS_FILE", chronovisor_root / "runtime" / "status.json"
    )
    monkeypatch.setattr(
        runtime_status, "EVENTS_FILE", chronovisor_root / "runtime" / "events.jsonl"
    )
    monkeypatch.setattr(
        runtime_status, "METRICS_FILE", chronovisor_root / "runtime" / "metrics.jsonl"
    )

    # IndexStore reads its paths from module globals AND from wiki.PAGES_DIR
    # internally; patch both layers.
    monkeypatch.setattr(index_store, "CHRONOVISOR_ROOT", chronovisor_root)
    monkeypatch.setattr(index_store, "PAGES_DIR", pages)
    monkeypatch.setattr(index_store, "SYSTEM_DIR", system)
    monkeypatch.setattr(index_store, "INDEX_DIR", index_dir)
    monkeypatch.setattr(index_store, "PAGES_INDEX_FILE", index_dir / "pages.json")
    monkeypatch.setattr(
        index_store, "BACKLINKS_INDEX_FILE", index_dir / "backlinks.json"
    )
    monkeypatch.setattr(index_store, "_store", None)
    monkeypatch.setattr(ollama, "is_available", lambda: False)
    monkeypatch.setattr(ingest, "is_available", lambda: False)
    monkeypatch.setattr(orchestrator, "is_available", lambda: False)

    real_ensure_page_metadata = ingest._ensure_page_metadata_frontmatter

    def isolated_page_metadata(
        text,
        page_id,
        parse,
        patch,
        *,
        allow_local_model=True,
        force_deterministic_rebuild=False,
    ):
        del allow_local_model
        return real_ensure_page_metadata(
            text,
            page_id,
            parse,
            patch,
            allow_local_model=False,
            force_deterministic_rebuild=force_deterministic_rebuild,
        )

    monkeypatch.setattr(
        ingest,
        "_ensure_page_metadata_frontmatter",
        isolated_page_metadata,
    )
    monkeypatch.setattr(
        search,
        "update_embeddings",
        lambda page_ids=None, *, strict=False: 0,
    )
    from chronovisor.core.runtime_config import DecisionRouterConfig

    # Review planning must not inherit the operator's live adoption artifact.
    # A production artifact can legitimately become stale after a lane-contract
    # change; isolated tests still need a deterministic bootstrap policy.
    monkeypatch.setattr(
        ingest,
        "_ingest_review_router_config",
        lambda: DecisionRouterConfig(adoption_artifact=""),
    )
    monkeypatch.setattr(
        orchestrator,
        "ingest_authority_preflight",
        lambda **_kwargs: {
            "ok": True,
            "status": "ready",
            "blocked_by": None,
            "retryable": False,
            "error": None,
            "artifact_sha256": "a" * 64,
        },
    )

    def isolated_frontier_review(proposal, *, reviewer=None):
        if reviewer is not None:
            return ingest._normalize_ingest_frontier_review(
                reviewer(proposal),
                proposal=proposal,
            )
        has_prepared = bool(proposal.get("prepared_operations"))
        has_failed = bool(proposal.get("failed_operation_specs"))
        return {
            "decision": "apply_available" if has_prepared else "confirmed_noop",
            "summary": "isolated ingest fixture disposition",
            "failed_operations_disposition": (
                "confirmed_unnecessary" if has_failed else "none"
            ),
            "tests_run": [],
            "risk": None,
            "notes": None,
        }

    monkeypatch.setattr(ingest, "_run_ingest_frontier_review", isolated_frontier_review)

    return chronovisor_root


def _seed_page(chronovisor_root: Path, rel: str, body: str) -> Path:
    path = chronovisor_root / "pages" / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(body)
    return path


def _write_legacy_v1_ingest_proposal(
    *,
    raw_content: str,
    operations: list[dict],
    raw_keywords: list[str] | None = None,
    source_raw: str | None = None,
) -> tuple[Path, Path, dict[str, object]]:
    """Write the provenance-free proposal shape emitted before schema v2."""

    from chronovisor.ingest import ingest

    planned, totals = ingest._prepare_operations(operations)
    proposal = ingest._build_ingest_frontier_proposal(
        raw_content=raw_content,
        raw_keywords=raw_keywords,
        source_raw=source_raw,
        operations=operations,
        planned=planned,
        link_totals=totals,
    )
    proposal["schema_version"] = 1
    prepared_rows = proposal["prepared_operations"]
    assert isinstance(prepared_rows, list)
    for row in prepared_rows:
        assert isinstance(row, dict)
        row.pop("source_operation_index")
        row.pop("source_operation_type")
        row.pop("source_filename")
    source_key = str(proposal["source_key"])
    proposal_path, review_path = ingest._ingest_artifact_paths(source_key)
    ingest._write_ingest_artifact(
        proposal_path,
        {
            "schema_version": 1,
            "kind": "ingest_frontier_proposal_artifact",
            "source_key": source_key,
            "proposal_sha256": ingest._canonical_json_sha256(proposal),
            "proposal": proposal,
        },
    )
    return proposal_path, review_path, proposal


def _downgrade_ingest_proposal_artifact_to_v1(
    proposal_path: Path,
) -> dict[str, object]:
    """Preserve proposal bytes except for the schema-v1 provenance omission."""

    from chronovisor.ingest import ingest

    artifact = json.loads(proposal_path.read_text(encoding="utf-8"))
    proposal = artifact["proposal"]
    proposal["schema_version"] = 1
    for row in proposal["prepared_operations"]:
        row.pop("source_operation_index")
        row.pop("source_operation_type")
        row.pop("source_filename")
    artifact["schema_version"] = 1
    artifact["proposal_sha256"] = ingest._canonical_json_sha256(proposal)
    ingest._write_ingest_artifact(proposal_path, artifact)
    return artifact


def test_ingest_artifact_schema_tracks_proposal_envelope_schema() -> None:
    from chronovisor.decision.decision_lane_prompts import (
        INGEST_PROPOSAL_SCHEMA_VERSION,
    )
    from chronovisor.ingest import ingest

    assert (
        ingest.INGEST_FRONTIER_ARTIFACT_SCHEMA_VERSION == INGEST_PROPOSAL_SCHEMA_VERSION
    )


class TestApplyOperations:
    def test_create_writes_atomically(self, isolated_wiki: Path) -> None:
        ops = [
            {
                "type": "create",
                "filename": "misc/new-page.md",
                "content": "---\ntitle: T\nupdated: 2026-04-28\n---\nhello",
            }
        ]
        created, updated = _apply_operations(ops)
        assert created == ["new-page"]
        assert updated == []
        body = (isolated_wiki / "pages" / "misc" / "new-page.md").read_text()
        assert "title: T" in body and body.endswith("\n")
        assert f"updated: {date.today().isoformat()}" in body

    @pytest.mark.parametrize("filename", ["root-page.md", "a/b/nested-page.md"])
    def test_create_refuses_wrong_path_depth_even_if_triage_was_bypassed(
        self, isolated_wiki: Path, filename: str
    ) -> None:
        ops = [
            {
                "type": "create",
                "filename": filename,
                "content": "---\ntitle: Root\nupdated: 2026-07-18\n---\nbody",
            }
        ]

        with pytest.raises(IngestApplyError, match="top-level folder"):
            _apply_operations(ops)

        assert not (isolated_wiki / "pages" / filename).exists()

    def test_create_collision_converts_to_update(self, isolated_wiki: Path) -> None:
        path = _seed_page(
            isolated_wiki,
            "a/foo.md",
            "---\ntitle: existing\nupdated: 2026-01-01\n---\nold",
        )
        ops = [
            {
                "type": "create",
                "filename": "b/foo.md",
                "content": "---\ntitle: dup\nupdated: 2026-04-28\n---\nnew",
            }
        ]
        created, updated = _apply_operations(ops)

        assert created == []
        assert updated == ["foo"]
        text = path.read_text()
        assert "old" in text
        assert "new" in text
        assert "title: dup" not in text
        assert not (isolated_wiki / "pages" / "b" / "foo.md").exists()

    def test_update_missing_target_fails(self, isolated_wiki: Path) -> None:
        ops = [
            {
                "type": "update",
                "filename": "ghost.md",
                "content": "addendum",
            }
        ]
        with pytest.raises(IngestApplyError, match="update target not found"):
            _apply_operations(ops)

    def test_update_resolves_single_loose_page_id(self, isolated_wiki: Path) -> None:
        path = _seed_page(
            isolated_wiki,
            "ai/opus-4.7-evaluation-and-industry-geopolitics.md",
            "---\ntitle: Opus\nupdated: 2026-01-01\n---\nold",
        )
        ops = [
            {
                "type": "update",
                "filename": "opus-4-7-evaluation-and-industry-geopolitics.md",
                "content": "new",
            }
        ]

        created, updated = _apply_operations(ops)

        assert created == []
        assert updated == ["opus-4.7-evaluation-and-industry-geopolitics"]
        text = path.read_text()
        assert "old" in text
        assert "new" in text

    def test_update_ambiguous_loose_page_id_fails(self, isolated_wiki: Path) -> None:
        _seed_page(
            isolated_wiki,
            "a/foo.bar.md",
            "---\ntitle: A\nupdated: 2026-01-01\n---\nold",
        )
        _seed_page(
            isolated_wiki,
            "b/foo-bar.md",
            "---\ntitle: B\nupdated: 2026-01-01\n---\nold",
        )
        ops = [
            {
                "type": "update",
                "filename": "foo_bar.md",
                "content": "new",
            }
        ]

        with pytest.raises(IngestApplyError, match="ambiguous loose page_id"):
            _apply_operations(ops)

    def test_update_resolves_page_id_alias(self, isolated_wiki: Path) -> None:
        from chronovisor.core.alias_store import add_alias

        path = _seed_page(
            isolated_wiki,
            "ai/canonical-target.md",
            "---\ntitle: Canonical\nupdated: 2026-01-01\n---\nold",
        )
        add_alias("model-made-up-target", "ai/canonical-target")
        ops = [
            {
                "type": "update",
                "filename": "model-made-up-target.md",
                "content": "new",
            }
        ]

        created, updated = _apply_operations(ops)

        assert created == []
        assert updated == ["canonical-target"]
        assert "new" in path.read_text()

    def test_update_appends_without_frontmatter_injection(
        self, isolated_wiki: Path
    ) -> None:
        # Even if the body slips a frontmatter block past _extract_page_body,
        # _apply_operations must not corrupt the existing page. Here we feed
        # a clean body (the contract): test that append works and there is
        # still exactly one frontmatter block.
        path = _seed_page(
            isolated_wiki,
            "career/x.md",
            "---\ntitle: X\nupdated: 2026-01-01\n---\noriginal\n",
        )
        ops = [
            {
                "type": "update",
                "filename": "x.md",
                "content": "## addendum\nnew lines",
            }
        ]
        created, updated = _apply_operations(ops)
        assert updated == ["x"]
        text = path.read_text()
        # Exactly one frontmatter delimiter pair.
        assert text.count("\n---\n") == 1
        assert "## addendum" in text
        assert "original" in text

    def test_stale_ingest_cannot_reintroduce_applied_content_correction(
        self, isolated_wiki: Path
    ) -> None:
        from chronovisor.core import page_mutation

        path = _seed_page(
            isolated_wiki,
            "hardware/display.md",
            "---\ntitle: Display\nupdated: 2026-07-10\n---\n"
            "The setup has two G32P 32-inch 6K displays.\n",
        )
        prepared = page_mutation.prepare_page_mutation(
            "display",
            [
                {
                    "old_text": "The setup has two G32P 32-inch 6K displays.",
                    "new_text": "The setup has one G32P 32-inch 6K display.",
                }
            ],
            correction_id="corr-display-count",
        )
        assert page_mutation.apply_prepared_mutations([prepared])["status"] == "applied"

        created, updated = _apply_operations(
            [
                {
                    "type": "update",
                    "filename": "display.md",
                    "content": (
                        "Stale replay says: The setup has two G32P 32-inch 6K displays. "
                        "The desk is 180 cm wide."
                    ),
                }
            ]
        )

        assert created == []
        assert updated == ["display"]
        written = path.read_text(encoding="utf-8")
        assert "two G32P 32-inch 6K displays" not in written
        assert written.count("one G32P 32-inch 6K display") >= 2
        assert "The desk is 180 cm wide." in written
        assert "applied_corrections: [corr-display-count]" in written

    def test_stale_ingest_cannot_resurrect_correction_under_new_slug(
        self, isolated_wiki: Path
    ) -> None:
        from chronovisor.core import page_mutation

        _seed_page(
            isolated_wiki,
            "hardware/display.md",
            "---\ntitle: Display\nupdated: 2026-07-10\n---\n"
            "The setup has two G32P 32-inch 6K displays.\n",
        )
        prepared = page_mutation.prepare_page_mutation(
            "display",
            [
                {
                    "old_text": "The setup has two G32P 32-inch 6K displays.",
                    "new_text": "The setup has one G32P 32-inch 6K display.",
                }
            ],
            correction_id="corr-display-count-global",
        )
        assert page_mutation.apply_prepared_mutations([prepared])["status"] == "applied"

        created, updated = _apply_operations(
            [
                {
                    "type": "create",
                    "filename": "hardware/alternate-display-memory.md",
                    "content": (
                        "---\ntitle: Alternate display memory\nupdated: 2026-07-11\n---\n"
                        "The setup has two G32P 32-inch 6K displays.\n"
                    ),
                }
            ]
        )

        assert updated == []
        assert created == ["alternate-display-memory"]
        alternate = (
            page_mutation.PAGES_DIR / "hardware" / "alternate-display-memory.md"
        )
        written = alternate.read_text(encoding="utf-8")
        assert "two G32P 32-inch 6K displays" not in written
        assert "one G32P 32-inch 6K display" in written
        assert "corr-display-count-global" in written

    def test_index_store_failure_raises(
        self, isolated_wiki: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Force index_store.refresh to fail and confirm we abort instead of
        # silently destroying every link with an empty allowed_ids set.
        from chronovisor.core import index_store

        def boom(*_a, **_kw):
            raise RuntimeError("simulated index failure")

        monkeypatch.setattr(index_store.IndexStore, "refresh", boom)
        ops = [
            {
                "type": "create",
                "filename": "a/x.md",
                "content": "---\ntitle: X\nupdated: 2026-04-28\n---\nbody",
            }
        ]
        with pytest.raises(IngestApplyError, match="index_store unavailable"):
            _apply_operations(ops)


class TestIngestFrontierGate:
    @staticmethod
    def _create_op() -> dict:
        return {
            "type": "create",
            "filename": "memory/frontier-only.md",
            "content": (
                "---\ntitle: Frontier only\nupdated: 2026-07-11\n---\n"
                "The exact proposed fact.\n"
            ),
        }

    @staticmethod
    def _oversized_create_ops(
        *, count: int = 8, body_bytes: int = 18_000
    ) -> list[dict]:
        return [
            {
                "type": "create",
                "filename": f"memory/sharded-{index}.md",
                "content": (
                    "---\n"
                    f"title: Sharded {index}\n"
                    "updated: 2026-07-14\n"
                    "---\n"
                    f"Grounded fact {index}.\n"
                    + (chr(ord("a") + index) * body_bytes)
                    + "\n"
                ),
            }
            for index in range(count)
        ]

    @staticmethod
    def _fixed_review_config():
        from chronovisor.core.runtime_config import DecisionRouterConfig

        return DecisionRouterConfig(
            num_ctx=114_688,
            max_input_chars=93_000,
            adoption_artifact="",
        )

    @staticmethod
    def _production_authority(marker: str) -> dict:
        return {
            "source": "adopted_local_consensus",
            "authority_version": 1,
            "lane": "ingest_reconciliation",
            "lane_contract_sha256": "1" * 64,
            "lane_contract_manifest_sha256": "2" * 64,
            "lane_contract_case_manifest_sha256": "3" * 64,
            "policy": {
                "kind": "consensus",
                "schema_name": "ingest_reconciliation",
                "mode": "enabled",
                "error": None,
            },
            "router": {
                "source": "adopted_artifact",
                "artifact_sha256": marker * 64,
                "error": None,
                "models": ["primary:model", "challenger:model", "tie:model"],
            },
        }

    @staticmethod
    def _authority_review(
        authority: dict,
        *,
        decision: str = "apply_available",
        summary: str = "authority-bound review",
    ) -> dict:
        from chronovisor.decision.decision_router import canonical_agreement_signature
        from chronovisor.decision.decision_schema_manifest import (
            production_decision_schemas,
        )

        review = {
            "decision": decision,
            "summary": summary,
            "failed_operations_disposition": "none",
            "tests_run": [],
            "risk": None,
            "notes": None,
            "decision_policy": {
                **authority["policy"],
                "router_policy": authority["router"],
            },
        }
        signature = canonical_agreement_signature(
            review,
            schema=production_decision_schemas()["ingest_reconciliation"],
        )
        agreement = hashlib.sha256(signature.encode("utf-8")).hexdigest()
        review["local_consensus"] = {
            "status": "agreed",
            "ok": True,
            "agreement_sha256": agreement,
            "failure_class": None,
            "quarantine_reason": None,
            "votes": [
                {
                    "role": "primary",
                    "model": "primary:model",
                    "valid": True,
                    "signature_sha256": agreement,
                    "invalid_reason": None,
                },
                {
                    "role": "challenger",
                    "model": "challenger:model",
                    "valid": True,
                    "signature_sha256": agreement,
                    "invalid_reason": None,
                },
            ],
        }
        return review

    def test_production_authority_binds_lane_manifests_and_adopted_models(
        self,
        isolated_wiki: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        del isolated_wiki
        from chronovisor.core import runtime_config
        from chronovisor.decision import decision_policy, decision_router
        from chronovisor.decision.decision_lane_contract_cases import (
            decision_lane_contract_case_manifest_sha256,
        )
        from chronovisor.decision.decision_lane_contracts import (
            lane_contract_manifest_sha256,
            lane_contract_sha256,
        )
        from chronovisor.decision.decision_router import (
            QUORUM_SAFETY_POLICY_VERSION,
        )
        from chronovisor.ingest import ingest

        def production_review_stub(*_args, **_kwargs):
            raise AssertionError("authority resolution must not invoke review")

        production_review_stub.__module__ = "tests.foreign_review_adapter"
        monkeypatch.setattr(
            ingest, "_run_ingest_frontier_review", production_review_stub
        )

        class Policy:
            kind = "consensus"
            schema_name = "ingest_reconciliation"

        router_audit = {
            "source": "adopted_artifact",
            "artifact_sha256": "a" * 64,
            "error": None,
            "models": ["primary:model", "challenger:model", "tie:model"],
        }

        class Resolution:
            source = "adopted_artifact"
            error = None
            artifact_sha256 = "a" * 64

            @staticmethod
            def audit_record() -> dict[str, object]:
                return router_audit

        monkeypatch.setattr(
            decision_policy,
            "resolve_decision_policy",
            lambda _lane: (Policy(), "enabled", None),
        )
        monkeypatch.setattr(
            runtime_config,
            "load_decision_router_config",
            lambda: object(),
        )
        monkeypatch.setattr(
            decision_router,
            "resolve_router_policy",
            lambda _config: Resolution(),
        )

        authority, error = ingest._current_ingest_review_authority(reviewer=None)

        assert error is None
        assert authority is not None
        assert authority["lane_contract_sha256"] == lane_contract_sha256(
            "ingest_reconciliation"
        )
        assert (
            authority["lane_contract_manifest_sha256"]
            == lane_contract_manifest_sha256()
        )
        assert (
            authority["lane_contract_case_manifest_sha256"]
            == decision_lane_contract_case_manifest_sha256()
        )
        assert authority["policy"] == {
            "kind": "consensus",
            "schema_name": "ingest_reconciliation",
            "mode": "enabled",
            "error": None,
        }
        assert authority["router"] == router_audit
        assert authority["source"] == "adopted_local_consensus"
        assert (
            authority["quorum_safety_policy_version"]
            == QUORUM_SAFETY_POLICY_VERSION
        )
        assert "tie_break_adjudication_policy_version" not in authority
        assert ingest._ingest_review_authority_shape_error(authority) is None

    @pytest.mark.parametrize("authority_state", ["stable", "drifted", "missing"])
    def test_production_bounded_nonconvergence_is_deferred_only_under_stable_authority(
        self,
        isolated_wiki: Path,
        monkeypatch: pytest.MonkeyPatch,
        authority_state: str,
    ) -> None:
        from chronovisor.core import jobs
        from chronovisor.ingest import failure_supervisor, ingest

        authority_a = self._production_authority("a")
        authority_b = self._production_authority("b")
        authority_checks = 0

        def current_authority(**_kwargs):
            nonlocal authority_checks
            authority_checks += 1
            if authority_state == "drifted" and authority_checks == 2:
                return authority_b, None
            if authority_state == "missing" and authority_checks == 2:
                return None, "decision_adoption_not_valid:ingest_reconciliation"
            return authority_a, None

        def bounded_nonconvergence(*_args, **kwargs):
            budget = kwargs["frontier_budget"]
            assert budget.consume() is True
            assert budget.consume() is True
            return {
                "status": "frontier_budget_exhausted",
                "summary": "three valid local semantic votes remained different",
                "created": [],
                "updated": [],
            }

        # Mark this injected implementation as the production boundary.  The
        # dependency-injected legacy seam intentionally keeps its old message.
        bounded_nonconvergence.__module__ = ingest.__name__
        monkeypatch.setattr(
            ingest,
            "_current_ingest_review_authority",
            current_authority,
        )
        monkeypatch.setattr(
            failure_supervisor,
            "_current_adopted_authority_sha256",
            lambda: "a" * 64,
        )
        monkeypatch.setattr(
            ingest,
            "_triage",
            lambda _content, **_kwargs: [
                {
                    "type": "create",
                    "filename": "memory/bounded-nonconvergence.md",
                    "title": "Bounded nonconvergence",
                }
            ],
        )
        monkeypatch.setattr(
            ingest,
            "_generate_one",
            lambda op, _raw, **_kwargs: {
                "type": "create",
                "filename": op["filename"],
                "content": (
                    "---\ntitle: Bounded nonconvergence\nupdated: 2026-07-15\n"
                    "---\nbody\n"
                ),
            },
        )
        monkeypatch.setattr(ingest, "is_available", lambda: True)
        monkeypatch.setattr(
            ingest,
            "_review_and_apply_ingest_operations",
            bounded_nonconvergence,
        )
        monkeypatch.setattr(
            ingest,
            "_review_exact_ingest_repair_once",
            lambda operations, result, **_kwargs: (operations, result),
        )

        raw_text = "byte-exact bounded nonconvergence source\n"
        raw_path = isolated_wiki / "raw" / "bounded-nonconvergence.md"
        raw_path.write_text(raw_text, encoding="utf-8")
        job = jobs.job_store.create(processor="ollama")
        ingest.run_ingest(raw_text, job.job_id)

        finished = jobs.job_store.get(job.job_id)
        assert finished.status == jobs.JobStatus.FAILED
        starts: list[Path] = []
        monkeypatch.setattr(failure_supervisor, "_launch_self_heal", starts.append)
        supervision = failure_supervisor.record_raw_failure(
            raw_path=raw_path,
            error=finished.error,
            job_id=job.job_id,
            raw_text=raw_text,
        )

        assert authority_checks == 2
        assert raw_path.read_text(encoding="utf-8") == raw_text
        assert supervision.quarantined is False
        if authority_state == "drifted":
            assert str(finished.error).startswith(
                "local consensus authority unavailable: decision_authority_changed:"
            )
            assert supervision.failure_class == (
                "ingest.runtime_local_consensus_authority_unavailable"
            )
            assert supervision.terminal_deferred is False
            assert starts == [Path(str(supervision.packet_path))]
        elif authority_state == "missing":
            assert str(finished.error) == (
                "local consensus authority unavailable: "
                "decision_adoption_not_valid:ingest_reconciliation"
            )
            assert supervision.failure_class == (
                "ingest.runtime_local_consensus_authority_unavailable"
            )
            assert supervision.terminal_deferred is False
            assert starts == [Path(str(supervision.packet_path))]
        else:
            assert str(finished.error) == (
                "local consensus semantic no quorum "
                f"[authority_sha256={'a' * 64}]: "
                "three valid local semantic votes remained different"
            )
            assert supervision.failure_class == "ingest.semantic_no_quorum"
            assert supervision.terminal_deferred is True
            assert starts == []
        assert not (isolated_wiki / "raw" / ".dead-letter" / raw_path.name).exists()

    @pytest.mark.parametrize("publication_authority", ["b" * 64, None])
    def test_semantic_defer_publication_rechecks_authority_after_ingest(
        self,
        isolated_wiki: Path,
        monkeypatch: pytest.MonkeyPatch,
        publication_authority: str | None,
    ) -> None:
        from chronovisor.core import jobs
        from chronovisor.ingest import failure_supervisor, ingest

        authority_a = self._production_authority("a")
        authority_sha256: list[str | None] = ["a" * 64]

        def bounded_nonconvergence(*_args, **kwargs):
            budget = kwargs["frontier_budget"]
            assert budget.consume() is True
            assert budget.consume() is True
            return {
                "status": "frontier_budget_exhausted",
                "summary": "three valid local semantic votes remained different",
                "created": [],
                "updated": [],
            }

        bounded_nonconvergence.__module__ = ingest.__name__
        monkeypatch.setattr(
            ingest,
            "_current_ingest_review_authority",
            lambda **_kwargs: (authority_a, None),
        )
        monkeypatch.setattr(
            failure_supervisor,
            "_current_adopted_authority_sha256",
            lambda: authority_sha256[0],
        )
        monkeypatch.setattr(
            ingest,
            "_triage",
            lambda _content, **_kwargs: [
                {
                    "type": "create",
                    "filename": "memory/publication-cas.md",
                    "title": "Publication CAS",
                }
            ],
        )
        monkeypatch.setattr(
            ingest,
            "_generate_one",
            lambda op, _raw, **_kwargs: {
                "type": "create",
                "filename": op["filename"],
                "content": (
                    "---\ntitle: Publication CAS\nupdated: 2026-07-15\n---\nbody\n"
                ),
            },
        )
        monkeypatch.setattr(ingest, "is_available", lambda: True)
        monkeypatch.setattr(
            ingest,
            "_review_and_apply_ingest_operations",
            bounded_nonconvergence,
        )
        monkeypatch.setattr(
            ingest,
            "_review_exact_ingest_repair_once",
            lambda operations, result, **_kwargs: (operations, result),
        )
        starts: list[Path] = []
        monkeypatch.setattr(failure_supervisor, "_launch_self_heal", starts.append)

        raw_text = "authority may change after ingest returns\n"
        raw_path = isolated_wiki / "raw" / "publication-cas.md"
        raw_path.write_text(raw_text, encoding="utf-8")
        operational = failure_supervisor.record_raw_failure(
            raw_path=raw_path,
            error="local consensus semantic no quorum: authority marker missing",
        )
        operational_packet = Path(str(operational.packet_path))
        packet_before = operational_packet.read_bytes()

        job = jobs.job_store.create(processor="ollama")
        ingest.run_ingest(raw_text, job.job_id)
        finished = jobs.job_store.get(job.job_id)
        assert finished.status == jobs.JobStatus.FAILED
        assert str(finished.error).startswith(
            "local consensus semantic no quorum [authority_sha256=" + "a" * 64
        )

        # Simulate a valid A -> B adoption after run_ingest's final check but
        # before the orchestrator publishes the failure-supervisor outcome.
        authority_sha256[0] = publication_authority
        supervision = failure_supervisor.record_raw_failure(
            raw_path=raw_path,
            error=finished.error,
            job_id=job.job_id,
            raw_text=raw_text,
        )

        assert supervision.terminal_deferred is False
        assert supervision.transient is True
        assert supervision.tracked is False
        assert supervision.failure_class == (
            "ingest.runtime_local_consensus_authority_unavailable"
        )
        assert raw_path.read_text(encoding="utf-8") == raw_text
        assert operational_packet.read_bytes() == packet_before
        assert starts == [operational_packet]
        packets = list(
            (isolated_wiki / "runtime" / "failures" / "packets").glob("*.json")
        )
        assert packets == [operational_packet]
        state = json.loads(
            (isolated_wiki / "runtime" / "failures" / "state.json").read_text(
                encoding="utf-8"
            )
        )
        assert state["failures"][raw_path.name]["packet_path"] == str(
            operational_packet
        )

    def test_production_review_requires_policy_and_exact_local_quorum(
        self,
        isolated_wiki: Path,
    ) -> None:
        del isolated_wiki
        from chronovisor.ingest import ingest

        authority = {
            "source": "adopted_local_consensus",
            "authority_version": 1,
            "lane": "ingest_reconciliation",
            "lane_contract_sha256": "b" * 64,
            "lane_contract_manifest_sha256": "c" * 64,
            "lane_contract_case_manifest_sha256": "d" * 64,
            "policy": {
                "kind": "consensus",
                "schema_name": "ingest_reconciliation",
                "mode": "enabled",
                "error": None,
            },
            "router": {
                "source": "adopted_artifact",
                "artifact_sha256": "a" * 64,
                "error": None,
                "models": ["primary:model", "challenger:model", "tie:model"],
            },
        }
        review = {
            "decision": "apply_available",
            "summary": "grounded",
            "failed_operations_disposition": "none",
            "decision_policy": {
                **authority["policy"],
                "router_policy": authority["router"],
            },
        }

        assert ingest._ingest_review_authority_error(review, authority) == (
            "decision verdict local consensus proof is missing"
        )
        from chronovisor.decision.decision_router import canonical_agreement_signature
        from chronovisor.decision.decision_schema_manifest import (
            production_decision_schemas,
        )

        signature = canonical_agreement_signature(
            review,
            schema=production_decision_schemas()["ingest_reconciliation"],
        )
        agreement = hashlib.sha256(signature.encode("utf-8")).hexdigest()
        review["local_consensus"] = {
            "status": "agreed",
            "ok": True,
            "agreement_sha256": agreement,
            "failure_class": None,
            "quarantine_reason": None,
            "votes": [
                {
                    "role": "primary",
                    "model": "primary:model",
                    "valid": True,
                    "signature_sha256": agreement,
                    "invalid_reason": None,
                },
                {
                    "role": "challenger",
                    "model": "challenger:model",
                    "valid": True,
                    "signature_sha256": agreement,
                    "invalid_reason": None,
                },
            ],
        }

        normalized = ingest._normalize_ingest_frontier_review(
            review,
            proposal={"prepared_operations": [{}], "failed_operation_specs": []},
        )
        assert normalized["decision_policy"] == review["decision_policy"]
        assert normalized["local_consensus"] == review["local_consensus"]
        assert ingest._ingest_review_authority_error(normalized, authority) is None

        normalized["failed_operations_disposition"] = "retry_required"
        assert ingest._ingest_review_authority_error(normalized, authority) == (
            "decision verdict action does not match local consensus agreement"
        )

    def test_local_prepare_alone_cannot_mutate_page(self, isolated_wiki: Path) -> None:
        from chronovisor.ingest import ingest

        planned, _totals = ingest._prepare_operations([self._create_op()])

        assert len(planned) == 1
        assert "The exact proposed fact." in planned[0].new_body
        assert not (isolated_wiki / "pages" / "memory" / "frontier-only.md").exists()

    def test_oversized_eight_operation_proposal_uses_complete_bounded_shards(
        self,
        isolated_wiki: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        from chronovisor.ingest import ingest

        config = self._fixed_review_config()
        monkeypatch.setattr(ingest, "_ingest_review_router_config", lambda: config)
        captured: list[dict] = []

        result = ingest._review_and_apply_ingest_operations(
            self._oversized_create_ops(),
            raw_content="Eight independently grounded facts.",
            reviewer=lambda proposal: (
                captured.append(proposal)
                or {
                    "decision": "apply_available",
                    "summary": "exact shard is grounded",
                    "failed_operations_disposition": "none",
                }
            ),
        )

        assert result["status"] == "apply_available"
        assert len(captured) > 1
        proof = result["review"]["review_shard_proof"]
        manifest = proof["manifest"]
        assert manifest["full_operation_count"] == 8
        assert [
            index
            for row in manifest["shards"]
            for index in row["original_operation_indices"]
        ] == list(range(8))
        assert all(
            row["effective_input_bytes"] <= config.max_input_chars
            and row["required_num_ctx"] <= config.num_ctx
            for row in manifest["shards"]
        )
        assert len(result["created"]) == 8
        assert all(
            (isolated_wiki / "pages" / "memory" / f"sharded-{index}.md").exists()
            for index in range(8)
        )

    def test_shard_budget_charges_each_new_call_and_resume_reuses_approvals(
        self,
        isolated_wiki: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        from chronovisor.ingest import ingest

        config = self._fixed_review_config()
        monkeypatch.setattr(ingest, "_ingest_review_router_config", lambda: config)
        operations = self._oversized_create_ops(body_bytes=30_000)
        calls = 0

        def reviewer(_proposal: dict) -> dict:
            nonlocal calls
            calls += 1
            return {
                "decision": "apply_available",
                "summary": "exact shard approved",
                "failed_operations_disposition": "none",
            }

        first_budget = ingest._FrontierCallBudget(limit=2)
        first = ingest._review_and_apply_ingest_operations(
            operations,
            raw_content="Four bounded review shards resume durably.",
            reviewer=reviewer,
            frontier_budget=first_budget,
        )

        assert first["status"] == "shard_continuation_pending"
        assert first["shard_continuation"]["approved_shards"] == 2
        assert first["shard_continuation"]["total_shards"] == 4
        assert first_budget.used == 2
        assert calls == 2
        assert first["created"] == []
        assert not any(
            (isolated_wiki / "pages" / "memory" / f"sharded-{index}.md").exists()
            for index in range(8)
        )

        continuation = ingest._load_pretriage_ingest_shard_continuation(
            "Four bounded review shards resume durably.",
            None,
            reviewer=reviewer,
        )
        assert continuation is not None
        assert continuation.approved_shards == 2
        second_budget = ingest._FrontierCallBudget(limit=2)
        second = ingest._review_and_apply_ingest_operations(
            operations,
            raw_content="Four bounded review shards resume durably.",
            reviewer=reviewer,
            frontier_budget=second_budget,
            shard_continuation=continuation,
        )

        assert second["status"] == "apply_available"
        assert len(second["review"]["review_shard_proof"]["manifest"]["shards"]) == 4
        assert second_budget.used == 2
        assert calls == 4
        assert len(second["created"]) == 8

    def test_standard_review_budget_charges_exactly_one_call(
        self,
        isolated_wiki: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        from chronovisor.ingest import ingest

        monkeypatch.setattr(
            ingest,
            "_ingest_review_router_config",
            lambda: self._fixed_review_config(),
        )
        calls = 0

        def reviewer(_proposal: dict) -> dict:
            nonlocal calls
            calls += 1
            return {
                "decision": "apply_available",
                "summary": "standard review approved",
                "failed_operations_disposition": "none",
            }

        budget = ingest._FrontierCallBudget(limit=1)
        result = ingest._review_and_apply_ingest_operations(
            [self._create_op()],
            raw_content="One standard review.",
            reviewer=reviewer,
            frontier_budget=budget,
        )

        assert result["status"] == "apply_available"
        assert calls == 1
        assert budget.used == 1

    def test_shard_budget_exhaustion_does_not_publish_stale_authority_continuation(
        self,
        isolated_wiki: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        from chronovisor.ingest import ingest

        monkeypatch.setattr(
            ingest,
            "_ingest_review_router_config",
            lambda: self._fixed_review_config(),
        )

        def reviewer(_proposal):
            return {
                "decision": "apply_available",
                "summary": "exact shard approved before authority race",
                "failed_operations_disposition": "none",
            }

        stable_authority, error = ingest._current_ingest_review_authority(
            reviewer=reviewer
        )
        assert error is None and stable_authority is not None
        authority_checks = 0

        def changing_authority(*, reviewer):
            nonlocal authority_checks
            del reviewer
            authority_checks += 1
            if authority_checks == 4:
                return None, "simulated authority replacement after last approval"
            return stable_authority, None

        monkeypatch.setattr(
            ingest,
            "_current_ingest_review_authority",
            changing_authority,
        )
        result = ingest._review_and_apply_ingest_operations(
            self._oversized_create_ops(body_bytes=30_000),
            raw_content="Authority changes at the shard budget boundary.",
            reviewer=reviewer,
            frontier_budget=ingest._FrontierCallBudget(limit=2),
        )

        assert authority_checks == 4
        assert result["status"] == "needs_retry"
        assert "shard_continuation" not in result
        assert "authority changed before shard continuation" in result["summary"]
        assert result["created"] == []
        assert not any(
            (isolated_wiki / "pages" / "memory" / f"sharded-{index}.md").exists()
            for index in range(8)
        )

    def test_zero_approval_manifest_without_transition_marker_is_not_resumed(
        self,
        isolated_wiki: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        from chronovisor.ingest import ingest

        monkeypatch.setattr(
            ingest,
            "_ingest_review_router_config",
            lambda: self._fixed_review_config(),
        )
        raw = "A bare manifest cannot authorize zero-progress continuation."
        dry = ingest._review_and_apply_ingest_operations(
            self._oversized_create_ops(body_bytes=30_000),
            raw_content=raw,
            reviewer=lambda _proposal: pytest.fail("dry run must not review"),
            dry_run=True,
        )
        proposal = dry["proposal"]
        source_key = dry["source_key"]
        proposal_path, _review_path = ingest._ingest_artifact_paths(source_key)
        ingest._write_ingest_artifact(
            proposal_path,
            {
                "schema_version": ingest.INGEST_FRONTIER_ARTIFACT_SCHEMA_VERSION,
                "kind": "ingest_frontier_proposal_artifact",
                "source_key": source_key,
                "proposal_sha256": dry["proposal_sha256"],
                "proposal": proposal,
            },
        )
        plan = ingest._build_ingest_review_shard_plan(
            proposal,
            force_review_unit=True,
        )
        assert plan is not None
        assert (
            ingest._persist_ingest_review_shard_manifest(
                plan,
                source_key=source_key,
            )
            is None
        )

        assert (
            ingest._load_pretriage_ingest_shard_continuation(
                raw,
                None,
                reviewer=lambda _proposal: pytest.fail("reviewer must not run"),
            )
            is None
        )

    def test_zero_approval_transition_marker_is_tamper_evident_and_claimed_resumeable(
        self,
        isolated_wiki: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        from chronovisor.ingest import ingest

        monkeypatch.setattr(
            ingest,
            "_ingest_review_router_config",
            lambda: self._fixed_review_config(),
        )
        raw = "A hash-bound exact repair continuation survives a worker crash."
        operations = self._oversized_create_ops(body_bytes=30_000)
        dry = ingest._review_and_apply_ingest_operations(
            operations,
            raw_content=raw,
            reviewer=lambda _proposal: pytest.fail("dry run must not review"),
            dry_run=True,
        )
        proposal = dry["proposal"]
        source_key = dry["source_key"]
        proposal_path, _review_path = ingest._ingest_artifact_paths(source_key)
        ingest._write_ingest_artifact(
            proposal_path,
            {
                "schema_version": ingest.INGEST_FRONTIER_ARTIFACT_SCHEMA_VERSION,
                "kind": "ingest_frontier_proposal_artifact",
                "source_key": source_key,
                "proposal_sha256": dry["proposal_sha256"],
                "proposal": proposal,
            },
        )
        plan = ingest._build_ingest_review_shard_plan(
            proposal,
            force_review_unit=True,
        )
        assert plan is not None
        assert (
            ingest._persist_ingest_review_shard_manifest(
                plan,
                source_key=source_key,
            )
            is None
        )
        authority, error = ingest._current_ingest_review_authority(
            reviewer=lambda _proposal: {}
        )
        assert error is None and authority is not None
        assert (
            ingest._persist_ingest_review_continuation_marker(
                source_key=source_key,
                plan=plan,
                reason="exact_repair_reseed",
                previous_full_proposal_sha256="f" * 64,
                previous_authority=authority,
                current_authority=authority,
            )
            is None
        )
        marker_path = ingest._ingest_review_continuation_marker_path(source_key)
        original_marker = marker_path.read_bytes()
        tampered = json.loads(original_marker)
        tampered["reason"] = "authority_epoch_reseed"
        marker_path.write_text(json.dumps(tampered), encoding="utf-8")
        with pytest.raises(IngestApplyError, match="marker is invalid"):
            ingest._load_pretriage_ingest_shard_continuation(
                raw,
                None,
                reviewer=lambda _proposal: {},
            )

        marker_path.write_bytes(original_marker)
        first = ingest._load_pretriage_ingest_shard_continuation(
            raw,
            None,
            reviewer=lambda _proposal: {},
        )
        assert first is not None and first.approved_shards == 0
        assert json.loads(marker_path.read_text())["state"] == "claimed"
        crash_recovery = ingest._load_pretriage_ingest_shard_continuation(
            raw,
            None,
            reviewer=lambda _proposal: {},
        )
        assert crash_recovery is not None
        assert crash_recovery.plan.manifest_sha256 == first.plan.manifest_sha256

    def test_exact_repair_transition_is_idempotent_but_bounded_to_one_per_raw(
        self,
        isolated_wiki: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        from chronovisor.ingest import ingest

        monkeypatch.setattr(
            ingest,
            "_ingest_review_router_config",
            lambda: self._fixed_review_config(),
        )
        raw = "Only one exact repaired postimage transition is permitted."
        original_operations = self._oversized_create_ops(body_bytes=30_000)
        original_dry = ingest._review_and_apply_ingest_operations(
            original_operations,
            raw_content=raw,
            reviewer=lambda _proposal: pytest.fail("dry run must not review"),
            dry_run=True,
        )
        first_repair = {
            "status": "needs_retry",
            "source_key": original_dry["source_key"],
            "proposal_sha256": original_dry["proposal_sha256"],
            "review": {
                "decision": "retry",
                "summary": "first exact replacement",
                "failed_operations_disposition": "retry_required",
                "replacement_operations": [
                    {
                        "filename": original_operations[0]["filename"],
                        "content": original_operations[0]["content"]
                        + "\nFIRST-EXACT-REPAIR\n",
                    }
                ],
            },
        }
        repaired, first_reseed = ingest._review_exact_ingest_repair_once(
            original_operations,
            first_repair,
            raw_content=raw,
            raw_keywords=None,
            source_raw=None,
            reviewer=lambda _proposal: pytest.fail("dry run must not review"),
            frontier_budget=ingest._FrontierCallBudget(limit=0),
        )
        assert first_reseed["status"] == "shard_continuation_pending"
        repaired_sha256 = ingest._canonical_json_sha256(repaired)
        assert (
            ingest._seal_ingest_review_repair_transition(
                source_key=original_dry["source_key"],
                previous_full_proposal_sha256=original_dry["proposal_sha256"],
                repaired_operations_sha256=repaired_sha256,
            )
            is None
        )

        second_repair = {
            "status": "needs_retry",
            "source_key": original_dry["source_key"],
            "proposal_sha256": first_reseed["proposal_sha256"],
            "review": {
                "decision": "retry",
                "summary": "different second replacement",
                "failed_operations_disposition": "retry_required",
                "replacement_operations": [
                    {
                        "filename": repaired[0]["filename"],
                        "content": repaired[0]["content"]
                        + "\nSECOND-DIFFERENT-REPAIR\n",
                    }
                ],
            },
        }
        unchanged, blocked = ingest._review_exact_ingest_repair_once(
            repaired,
            second_repair,
            raw_content=raw,
            raw_keywords=None,
            source_raw=None,
            reviewer=lambda _proposal: pytest.fail("second repair must be blocked"),
            frontier_budget=ingest._FrontierCallBudget(limit=0),
        )
        assert unchanged == repaired
        assert blocked["status"] == "needs_retry"
        assert blocked["summary"] == "exact repair transition limit exceeded"

    @pytest.mark.parametrize("drift_kind", ["authority", "router_config"])
    def test_zero_approval_marker_reseeds_once_across_identity_drift(
        self,
        isolated_wiki: Path,
        monkeypatch: pytest.MonkeyPatch,
        drift_kind: str,
    ) -> None:
        from chronovisor.core.runtime_config import DecisionRouterConfig
        from chronovisor.ingest import ingest

        config = {"value": self._fixed_review_config()}
        monkeypatch.setattr(
            ingest,
            "_ingest_review_router_config",
            lambda: config["value"],
        )
        authority_a = self._production_authority("a")
        authority_b = self._production_authority("b")
        active = {"authority": authority_a}
        monkeypatch.setattr(
            ingest,
            "_current_ingest_review_authority",
            lambda **_kwargs: (active["authority"], None),
        )
        raw = f"Zero approval marker survives {drift_kind} drift."
        dry = ingest._review_and_apply_ingest_operations(
            self._oversized_create_ops(body_bytes=30_000),
            raw_content=raw,
            reviewer=lambda _proposal: pytest.fail("dry run must not review"),
            dry_run=True,
        )
        source_key = dry["source_key"]
        proposal_path, _review_path = ingest._ingest_artifact_paths(source_key)
        ingest._write_ingest_artifact(
            proposal_path,
            {
                "schema_version": ingest.INGEST_FRONTIER_ARTIFACT_SCHEMA_VERSION,
                "kind": "ingest_frontier_proposal_artifact",
                "source_key": source_key,
                "proposal_sha256": dry["proposal_sha256"],
                "proposal": dry["proposal"],
            },
        )
        old_plan = ingest._build_ingest_review_shard_plan(
            dry["proposal"],
            force_review_unit=True,
        )
        assert old_plan is not None
        assert (
            ingest._persist_ingest_review_shard_manifest(
                old_plan,
                source_key=source_key,
            )
            is None
        )
        assert (
            ingest._persist_ingest_review_continuation_marker(
                source_key=source_key,
                plan=old_plan,
                reason="exact_repair_reseed",
                previous_full_proposal_sha256="f" * 64,
                previous_authority=authority_a,
                current_authority=authority_a,
            )
            is None
        )

        if drift_kind == "authority":
            active["authority"] = authority_b
        else:
            config["value"] = DecisionRouterConfig(
                num_ctx=1_048_576,
                min_num_ctx=16_384,
                max_input_chars=1_000_000,
                adoption_artifact="",
            )
        continuation = ingest._load_pretriage_ingest_shard_continuation(
            raw,
            None,
            reviewer=lambda _proposal: {},
        )
        assert continuation is not None and continuation.approved_shards == 0
        current_marker = json.loads(
            ingest._ingest_review_continuation_marker_path(source_key).read_text()
        )
        assert current_marker["state"] == "claimed"
        assert current_marker["manifest_sha256"] == continuation.plan.manifest_sha256
        assert current_marker["current_authority_sha256"] == (
            ingest._canonical_json_sha256(active["authority"])
        )
        assert current_marker["previous_authority_sha256"] == (
            ingest._canonical_json_sha256(authority_a)
        )

    def test_same_sha_no_progress_is_tombstoned_without_repeat_review(
        self,
        isolated_wiki: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        from chronovisor.ingest import ingest

        monkeypatch.setattr(
            ingest,
            "_ingest_review_router_config",
            lambda: self._fixed_review_config(),
        )
        calls = 0

        def reviewer(_proposal: dict) -> dict:
            nonlocal calls
            calls += 1
            return {
                "decision": "retry",
                "summary": "same postimage remains ungrounded",
                "failed_operations_disposition": "retry_required",
            }

        raw = "No verified shard approval means no continuation progress."
        first = ingest._review_and_apply_ingest_operations(
            self._oversized_create_ops(body_bytes=30_000),
            raw_content=raw,
            reviewer=reviewer,
            frontier_budget=ingest._FrontierCallBudget(limit=2),
        )
        assert first["status"] == "frontier_budget_exhausted"
        assert "shard_continuation" not in first
        assert calls == 2
        with pytest.raises(IngestApplyError, match="previously made no progress"):
            ingest._load_pretriage_ingest_shard_continuation(
                raw,
                None,
                reviewer=reviewer,
            )
        assert calls == 2

    def test_existing_shard_approval_stall_is_not_republished_as_progress(
        self,
        isolated_wiki: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        from chronovisor.ingest import ingest

        monkeypatch.setattr(
            ingest,
            "_ingest_review_router_config",
            lambda: self._fixed_review_config(),
        )
        raw = "One old approval cannot be recounted as new progress."
        calls = 0

        def first_reviewer(_proposal: dict) -> dict:
            nonlocal calls
            calls += 1
            if calls == 1:
                return {
                    "decision": "apply_available",
                    "summary": "only the first shard is approved",
                    "failed_operations_disposition": "none",
                }
            return {
                "decision": "retry",
                "summary": "remaining shard is not approved",
                "failed_operations_disposition": "retry_required",
            }

        operations = self._oversized_create_ops(body_bytes=30_000)
        first = ingest._review_and_apply_ingest_operations(
            operations,
            raw_content=raw,
            reviewer=first_reviewer,
            frontier_budget=ingest._FrontierCallBudget(limit=2),
        )
        assert first["status"] == "shard_continuation_pending"
        assert first["shard_continuation"]["approved_shards"] == 1
        continuation = ingest._load_pretriage_ingest_shard_continuation(
            raw,
            None,
            reviewer=first_reviewer,
        )
        assert continuation is not None and continuation.approved_shards == 1
        second = ingest._review_and_apply_ingest_operations(
            operations,
            raw_content=raw,
            reviewer=lambda _proposal: {
                "decision": "retry",
                "summary": "no new shard approval",
                "failed_operations_disposition": "retry_required",
            },
            frontier_budget=ingest._FrontierCallBudget(limit=2),
            shard_continuation=continuation,
        )
        assert second["status"] == "frontier_budget_exhausted"
        assert "shard_continuation" not in second
        with pytest.raises(IngestApplyError, match="previously made no progress"):
            ingest._load_pretriage_ingest_shard_continuation(
                raw,
                None,
                reviewer=first_reviewer,
            )

    def test_authority_epoch_reapproves_stale_shards_with_verified_progress(
        self,
        isolated_wiki: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        from chronovisor.ingest import ingest

        monkeypatch.setattr(
            ingest,
            "_ingest_review_router_config",
            lambda: self._fixed_review_config(),
        )
        authority_a = self._production_authority("a")
        authority_b = self._production_authority("b")
        active = {"authority": authority_a}
        monkeypatch.setattr(
            ingest,
            "_current_ingest_review_authority",
            lambda **_kwargs: (active["authority"], None),
        )
        raw = "All stale approvals must be replayed under the new authority."
        operations = self._oversized_create_ops(body_bytes=30_000)
        first = ingest._review_and_apply_ingest_operations(
            operations,
            raw_content=raw,
            reviewer=lambda _proposal: self._authority_review(authority_a),
            frontier_budget=ingest._FrontierCallBudget(limit=2),
        )
        continuation = ingest._load_pretriage_ingest_shard_continuation(
            raw,
            None,
            reviewer=lambda _proposal: self._authority_review(authority_a),
        )
        assert first["status"] == "shard_continuation_pending"
        assert continuation is not None
        for shard_index, shard in enumerate(continuation.plan.shards):
            shard_source_key, shard_path = ingest._ingest_review_shard_review_identity(
                continuation.plan,
                shard_index=shard_index,
                shard=shard,
            )
            if shard_path.exists():
                continue
            review = ingest._normalize_ingest_frontier_review(
                self._authority_review(authority_a),
                proposal=shard.proposal,
            )
            _artifact, artifact_error = (
                ingest._write_and_readback_ingest_review_artifact(
                    shard_path,
                    source_key=shard_source_key,
                    proposal_sha256=shard.proposal_sha256,
                    review=review,
                    authority=authority_a,
                    integrity=True,
                )
            )
            assert artifact_error is None

        active["authority"] = authority_b
        stale = ingest._load_pretriage_ingest_shard_continuation(
            raw,
            None,
            reviewer=lambda _proposal: self._authority_review(authority_b),
        )
        assert stale is not None and stale.approved_shards == 0
        b_calls = 0

        def reviewer_b(_proposal: dict) -> dict:
            nonlocal b_calls
            b_calls += 1
            return self._authority_review(authority_b)

        second = ingest._review_and_apply_ingest_operations(
            operations,
            raw_content=raw,
            reviewer=reviewer_b,
            frontier_budget=ingest._FrontierCallBudget(limit=2),
            shard_continuation=stale,
        )
        assert second["status"] == "shard_continuation_pending"
        assert second["shard_continuation"]["approved_shards"] == 2
        resumed = ingest._load_pretriage_ingest_shard_continuation(
            raw,
            None,
            reviewer=reviewer_b,
        )
        assert resumed is not None and resumed.approved_shards == 2
        final = ingest._review_and_apply_ingest_operations(
            operations,
            raw_content=raw,
            reviewer=reviewer_b,
            frontier_budget=ingest._FrontierCallBudget(limit=2),
            shard_continuation=resumed,
        )
        assert final["status"] == "apply_available", final
        assert b_calls == len(resumed.plan.shards)

    def test_exact_repair_can_reseed_as_one_standard_review_unit(
        self,
        isolated_wiki: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        from chronovisor.ingest import ingest

        monkeypatch.setattr(
            ingest,
            "_ingest_review_router_config",
            lambda: self._fixed_review_config(),
        )
        operations = self._oversized_create_ops(count=2, body_bytes=40_000)
        repair_marker = "REPAIRED-TO-STANDARD"

        def reviewer(proposal: dict) -> dict:
            contract = proposal["audit_decision"].get("review_shard_contract")
            generated = proposal["local_generated_operations"]
            if (
                contract
                and contract["shard_index"] == 0
                and repair_marker not in generated[0]["content"]
            ):
                header = generated[0]["content"].split("Grounded fact", 1)[0]
                return {
                    "decision": "retry",
                    "summary": "replace one oversized postimage exactly",
                    "failed_operations_disposition": "retry_required",
                    "replacement_operations": [
                        {
                            "filename": generated[0]["filename"],
                            "content": header
                            + "Grounded fact repaired."
                            + f"\n{repair_marker}\n",
                        }
                    ],
                }
            return {
                "decision": "apply_available",
                "summary": "exact review unit approved",
                "failed_operations_disposition": "none",
            }

        raw = "One shard repair shrinks the complete proposal to standard size."
        budget = ingest._FrontierCallBudget(limit=2)
        first = ingest._review_and_apply_ingest_operations(
            operations,
            raw_content=raw,
            reviewer=reviewer,
            frontier_budget=budget,
        )
        assert first["status"] == "needs_retry"
        repaired, reseeded = ingest._review_exact_ingest_repair_once(
            operations,
            first,
            raw_content=raw,
            raw_keywords=None,
            source_raw=None,
            reviewer=reviewer,
            frontier_budget=budget,
        )
        assert reseeded["status"] == "shard_continuation_pending"
        assert reseeded["shard_continuation"]["approved_shards"] == 0
        assert reseeded["shard_continuation"]["total_shards"] == 1
        assert not any((isolated_wiki / "pages" / "memory").glob("sharded-*.md"))
        continuation = ingest._load_pretriage_ingest_shard_continuation(
            raw,
            None,
            reviewer=reviewer,
        )
        assert continuation is not None and len(continuation.plan.shards) == 1
        final = ingest._review_and_apply_ingest_operations(
            repaired,
            raw_content=raw,
            reviewer=reviewer,
            frontier_budget=ingest._FrontierCallBudget(limit=2),
            shard_continuation=continuation,
        )
        assert final["status"] == "apply_available", final
        assert len(final["created"]) == 2

    def test_router_context_change_reseeds_verified_progress_without_regeneration(
        self,
        isolated_wiki: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        from chronovisor.core.runtime_config import DecisionRouterConfig
        from chronovisor.ingest import ingest

        config = {"value": self._fixed_review_config()}
        monkeypatch.setattr(
            ingest,
            "_ingest_review_router_config",
            lambda: config["value"],
        )
        operations = self._oversized_create_ops(body_bytes=30_000)
        raw = "Verified partial progress survives a router context expansion."
        calls = 0

        def reviewer(_proposal: dict) -> dict:
            nonlocal calls
            calls += 1
            return {
                "decision": "apply_available",
                "summary": "current exact review unit approved",
                "failed_operations_disposition": "none",
            }

        first = ingest._review_and_apply_ingest_operations(
            operations,
            raw_content=raw,
            reviewer=reviewer,
            frontier_budget=ingest._FrontierCallBudget(limit=2),
        )
        assert first["status"] == "shard_continuation_pending"
        old_manifest_sha256 = first["shard_continuation"]["manifest_sha256"]
        config["value"] = DecisionRouterConfig(
            num_ctx=1_048_576,
            min_num_ctx=16_384,
            max_input_chars=1_000_000,
            adoption_artifact="",
        )
        continuation = ingest._load_pretriage_ingest_shard_continuation(
            raw,
            None,
            reviewer=reviewer,
        )
        assert continuation is not None
        assert continuation.approved_shards == 0
        assert continuation.plan.manifest_sha256 != old_manifest_sha256
        assert len(continuation.plan.shards) == 1
        final = ingest._review_and_apply_ingest_operations(
            operations,
            raw_content=raw,
            reviewer=reviewer,
            frontier_budget=ingest._FrontierCallBudget(limit=2),
            shard_continuation=continuation,
        )
        assert final["status"] == "apply_available"
        assert len(final["created"]) == 8

    def test_oversized_shard_rejection_has_no_partial_page_apply(
        self,
        isolated_wiki: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        from chronovisor.ingest import ingest

        monkeypatch.setattr(
            ingest,
            "_ingest_review_router_config",
            lambda: self._fixed_review_config(),
        )
        calls = 0

        def reviewer(_proposal: dict) -> dict:
            nonlocal calls
            calls += 1
            return {
                "decision": "apply_available" if calls == 1 else "retry",
                "summary": "approved shard" if calls == 1 else "reject later shard",
                "failed_operations_disposition": (
                    "none" if calls == 1 else "retry_required"
                ),
            }

        result = ingest._review_and_apply_ingest_operations(
            self._oversized_create_ops(),
            raw_content="Eight facts, one unsafe proposed shard.",
            reviewer=reviewer,
        )

        assert calls > 1
        assert result["status"] == "needs_retry"
        assert result["created"] == []
        assert not any(
            (isolated_wiki / "pages" / "memory" / f"sharded-{index}.md").exists()
            for index in range(8)
        )

    def test_one_shard_confirmed_noop_cannot_discard_complete_raw(
        self,
        isolated_wiki: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        from chronovisor.ingest import ingest

        monkeypatch.setattr(
            ingest,
            "_ingest_review_router_config",
            lambda: self._fixed_review_config(),
        )

        def reviewer(proposal: dict) -> dict:
            shard_index = proposal["audit_decision"]["review_shard_contract"][
                "shard_index"
            ]
            return {
                "decision": "confirmed_noop" if shard_index == 0 else "apply_available",
                "summary": "one shard noop" if shard_index == 0 else "shard approved",
                "failed_operations_disposition": "none",
            }

        result = ingest._review_and_apply_ingest_operations(
            self._oversized_create_ops(),
            raw_content="A complete raw cannot be discarded by one shard.",
            reviewer=reviewer,
        )

        assert result["status"] == "needs_retry"
        assert result["created"] == []
        assert (
            result["review"]["frontier_failure"]["failure_class"]
            == "ingest_review_shard_nonapproval"
        )

    def test_oversized_terminal_reuse_recomputes_manifest_and_rejects_tamper(
        self,
        isolated_wiki: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        from chronovisor.ingest import ingest

        config = self._fixed_review_config()
        monkeypatch.setattr(ingest, "_ingest_review_router_config", lambda: config)
        operations = self._oversized_create_ops()
        calls = 0

        def reviewer(_proposal: dict) -> dict:
            nonlocal calls
            calls += 1
            return {
                "decision": "apply_available",
                "summary": "exact shard is grounded",
                "failed_operations_disposition": "none",
            }

        first = ingest._review_and_apply_ingest_operations(
            operations,
            raw_content="Eight facts with durable shard proofs.",
            reviewer=reviewer,
        )
        first_calls = calls
        recovered = ingest._review_and_apply_ingest_operations(
            operations,
            raw_content="Eight facts with durable shard proofs.",
            reviewer=reviewer,
        )
        assert recovered["status"] == "apply_available"
        assert recovered["reused_review"] is True
        assert calls == first_calls
        pretriage = ingest._load_pretriage_terminal_recovery(
            "Eight facts with durable shard proofs.",
            None,
            reviewer=reviewer,
        )
        assert pretriage is not None
        assert pretriage["status"] == "apply_available"
        assert pretriage["recovery_basis"] == "exact_postimages_already_applied"
        assert calls == first_calls

        proposal_path, review_path = ingest._ingest_artifact_paths(first["source_key"])
        proposal_bytes = proposal_path.read_bytes()
        review_bytes = review_path.read_bytes()
        corrupt_review = json.loads(review_bytes)
        corrupt_review["review"]["review_shard_proof"]["shard_reviews"].pop()
        review_path.write_text(json.dumps(corrupt_review), encoding="utf-8")
        corrupt_rejected = ingest._review_and_apply_ingest_operations(
            operations,
            raw_content="Eight facts with durable shard proofs.",
            reviewer=reviewer,
        )
        assert corrupt_rejected["status"] == "needs_retry"
        assert corrupt_rejected["created"] == []
        assert proposal_path.read_bytes() == proposal_bytes
        assert calls == first_calls
        review_path.write_bytes(review_bytes)

        plan = ingest._build_ingest_review_shard_plan(
            json.loads(proposal_path.read_text())["proposal"],
            config=config,
        )
        assert plan is not None
        manifest_path = ingest._ingest_review_shard_manifest_path(plan)
        artifact = json.loads(manifest_path.read_text())
        artifact["manifest"]["full_operation_count"] = 999
        manifest_path.write_text(json.dumps(artifact), encoding="utf-8")

        rejected = ingest._review_and_apply_ingest_operations(
            operations,
            raw_content="Eight facts with durable shard proofs.",
            reviewer=reviewer,
        )
        assert rejected["status"] == "needs_retry"
        assert rejected["created"] == []
        assert calls == first_calls
        assert (
            rejected["review"]["frontier_failure"]["failure_class"]
            == "ingest_review_shard_reuse_invalid"
        )

    def test_exact_postimage_recovery_uses_sealed_historical_shard_limits(
        self,
        isolated_wiki: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        from chronovisor.core.runtime_config import DecisionRouterConfig
        from chronovisor.ingest import ingest

        historical_config = self._fixed_review_config()
        monkeypatch.setattr(
            ingest,
            "_ingest_review_router_config",
            lambda: historical_config,
        )
        operations = self._oversized_create_ops()
        raw = "Applied shards survive a later context-window expansion."
        calls = 0

        def reviewer(_proposal: dict) -> dict:
            nonlocal calls
            calls += 1
            return {
                "decision": "apply_available",
                "summary": "historical exact shard approved",
                "failed_operations_disposition": "none",
            }

        first = ingest._review_and_apply_ingest_operations(
            operations,
            raw_content=raw,
            reviewer=reviewer,
        )
        assert first["status"] == "apply_available"
        first_calls = calls
        proposal_path, _review_path = ingest._ingest_artifact_paths(first["source_key"])
        proposal = json.loads(proposal_path.read_text(encoding="utf-8"))["proposal"]
        historical_plan = ingest._build_ingest_review_shard_plan(
            proposal,
            config=historical_config,
        )
        assert historical_plan is not None

        wide_config = DecisionRouterConfig(
            num_ctx=1_048_576,
            min_num_ctx=16_384,
            max_input_chars=1_000_000,
            adoption_artifact="",
        )
        assert (
            ingest._build_ingest_review_shard_plan(
                proposal,
                config=wide_config,
            )
            is None
        )
        monkeypatch.setattr(
            ingest,
            "_ingest_review_router_config",
            lambda: wide_config,
        )
        page_paths = sorted((isolated_wiki / "pages" / "memory").glob("sharded-*.md"))
        page_bytes = {path: path.read_bytes() for path in page_paths}
        real_apply = ingest._apply_prepared_operations
        recovery_modes: list[bool] = []

        def track_recovery(*args, **kwargs):
            recovery_modes.append(bool(kwargs.get("recovery_only")))
            return real_apply(*args, **kwargs)

        monkeypatch.setattr(ingest, "_apply_prepared_operations", track_recovery)
        recovered = ingest._review_and_apply_ingest_operations(
            operations,
            raw_content=raw,
            reviewer=reviewer,
        )
        assert recovered["status"] == "apply_available"
        assert recovered["reused_review"] is True
        assert recovered["recovery_basis"] == "exact_postimages_already_applied"
        assert recovery_modes == [True]
        assert calls == first_calls
        assert {path: path.read_bytes() for path in page_paths} == page_bytes

        pretriage = ingest._load_pretriage_terminal_recovery(
            raw,
            None,
            reviewer=reviewer,
        )
        assert pretriage is not None
        assert pretriage["recovery_basis"] == "exact_postimages_already_applied"
        assert recovery_modes == [True]
        assert calls == first_calls
        assert {path: path.read_bytes() for path in page_paths} == page_bytes

    @pytest.mark.parametrize("tamper_target", ["manifest", "shard_review"])
    def test_historical_shard_recovery_rejects_durable_proof_tamper_without_writes(
        self,
        isolated_wiki: Path,
        monkeypatch: pytest.MonkeyPatch,
        tamper_target: str,
    ) -> None:
        from chronovisor.core.runtime_config import DecisionRouterConfig
        from chronovisor.ingest import ingest

        historical_config = self._fixed_review_config()
        monkeypatch.setattr(
            ingest,
            "_ingest_review_router_config",
            lambda: historical_config,
        )
        operations = self._oversized_create_ops()
        raw = f"Historical shard tamper is fail closed: {tamper_target}."
        first = ingest._review_and_apply_ingest_operations(
            operations,
            raw_content=raw,
            reviewer=lambda _proposal: {
                "decision": "apply_available",
                "summary": "exact shard approved before tamper",
                "failed_operations_disposition": "none",
            },
        )
        assert first["status"] == "apply_available"
        proposal_path, _review_path = ingest._ingest_artifact_paths(first["source_key"])
        proposal = json.loads(proposal_path.read_text(encoding="utf-8"))["proposal"]
        plan = ingest._build_ingest_review_shard_plan(
            proposal,
            config=historical_config,
        )
        assert plan is not None
        if tamper_target == "manifest":
            target = ingest._ingest_review_shard_manifest_path(plan)
            payload = json.loads(target.read_text(encoding="utf-8"))
            payload["manifest"]["full_operation_count"] += 1
        else:
            _source_key, target = ingest._ingest_review_shard_review_identity(
                plan,
                shard_index=0,
                shard=plan.shards[0],
            )
            payload = json.loads(target.read_text(encoding="utf-8"))
            payload["review"]["summary"] = "tampered durable shard review"
        target.write_text(json.dumps(payload), encoding="utf-8")

        monkeypatch.setattr(
            ingest,
            "_ingest_review_router_config",
            lambda: DecisionRouterConfig(
                num_ctx=1_048_576,
                min_num_ctx=16_384,
                max_input_chars=1_000_000,
                adoption_artifact="",
            ),
        )
        page_paths = sorted((isolated_wiki / "pages" / "memory").glob("sharded-*.md"))
        page_bytes = {path: path.read_bytes() for path in page_paths}
        apply_calls = 0

        def forbid_apply(*_args, **_kwargs):
            nonlocal apply_calls
            apply_calls += 1
            raise AssertionError("tampered historical proof must not reach page apply")

        monkeypatch.setattr(ingest, "_apply_prepared_operations", forbid_apply)
        rejected = ingest._review_and_apply_ingest_operations(
            operations,
            raw_content=raw,
            reviewer=lambda _proposal: pytest.fail("reviewer must not run"),
        )
        assert rejected["status"] == "needs_retry"
        assert rejected["created"] == []
        assert apply_calls == 0
        assert {path: path.read_bytes() for path in page_paths} == page_bytes
        with pytest.raises(IngestApplyError, match="shard review binding is invalid"):
            ingest._load_pretriage_terminal_recovery(
                raw,
                None,
                reviewer=lambda _proposal: pytest.fail("reviewer must not run"),
            )
        assert apply_calls == 0
        assert {path: path.read_bytes() for path in page_paths} == page_bytes

    def test_standard_size_review_path_has_no_shard_contract(
        self,
        isolated_wiki: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        from chronovisor.ingest import ingest

        monkeypatch.setattr(
            ingest,
            "_ingest_review_router_config",
            lambda: self._fixed_review_config(),
        )
        captured: list[dict] = []
        result = ingest._review_and_apply_ingest_operations(
            [self._create_op()],
            raw_content="One grounded fact.",
            reviewer=lambda proposal: (
                captured.append(proposal)
                or {
                    "decision": "apply_available",
                    "summary": "standard request approved",
                    "failed_operations_disposition": "none",
                }
            ),
        )

        assert result["status"] == "apply_available"
        assert len(captured) == 1
        assert "review_shard_contract" not in captured[0]["audit_decision"]
        assert "review_shard_proof" not in result["review"]

    @pytest.mark.parametrize(
        ("operations", "failed_specs", "reason"),
        [
            (
                None,
                [],
                "one prepared ingest operation exceeds the bounded review capacity",
            ),
            (
                "eight",
                [{"filename": "memory/missing.md", "error": "generation failed"}],
                "failed_operation_specs cannot be sharded safely",
            ),
        ],
    )
    def test_unsupported_oversized_shapes_fail_closed_with_typed_capacity(
        self,
        isolated_wiki: Path,
        monkeypatch: pytest.MonkeyPatch,
        operations: str | None,
        failed_specs: list[dict],
        reason: str,
    ) -> None:
        from chronovisor.ingest import ingest

        monkeypatch.setattr(
            ingest,
            "_ingest_review_router_config",
            lambda: self._fixed_review_config(),
        )
        reviewer_calls = 0

        def reviewer(_proposal: dict) -> dict:
            nonlocal reviewer_calls
            reviewer_calls += 1
            return {
                "decision": "apply_available",
                "summary": "should not run",
                "failed_operations_disposition": "none",
            }

        selected_operations = (
            self._oversized_create_ops()
            if operations == "eight"
            else self._oversized_create_ops(count=1, body_bytes=120_000)
        )
        result = ingest._review_and_apply_ingest_operations(
            selected_operations,
            raw_content="Oversized bounded review evidence.",
            failed_operation_specs=failed_specs,
            local_disposition=(
                "partial_generation_failed" if failed_specs else "operations_available"
            ),
            reviewer=reviewer,
        )

        assert result["status"] == "needs_retry"
        assert result["created"] == []
        assert reviewer_calls == 0
        assert reason in result["review"]["summary"]
        assert result["review"]["frontier_failure"]["failure_class"] in {
            "input_too_large",
            "context_window_exceeded",
        }

    def test_multiple_shard_repairs_fail_closed_without_applying(
        self,
        isolated_wiki: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        from chronovisor.ingest import ingest

        monkeypatch.setattr(
            ingest,
            "_ingest_review_router_config",
            lambda: self._fixed_review_config(),
        )

        def reviewer(proposal: dict) -> dict:
            operation = proposal["local_generated_operations"][0]
            return {
                "decision": "retry",
                "summary": "one exact shard repair",
                "failed_operations_disposition": "retry_required",
                "replacement_operations": [
                    {
                        "filename": operation["filename"],
                        "content": operation["content"] + "repaired\n",
                    }
                ],
            }

        result = ingest._review_and_apply_ingest_operations(
            self._oversized_create_ops(),
            raw_content="Multiple shards each request a repair.",
            reviewer=reviewer,
        )

        assert result["status"] == "needs_retry"
        assert result["created"] == []
        assert (
            result["review"]["frontier_failure"]["failure_class"]
            == "ingest_review_multiple_repairs_unsupported"
        )
        assert not any(
            (isolated_wiki / "pages" / "memory" / f"sharded-{index}.md").exists()
            for index in range(8)
        )

    def test_one_exact_shard_repair_is_returned_to_existing_convergence_path(
        self,
        isolated_wiki: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        from chronovisor.ingest import ingest

        monkeypatch.setattr(
            ingest,
            "_ingest_review_router_config",
            lambda: self._fixed_review_config(),
        )

        def reviewer(proposal: dict) -> dict:
            contract = proposal["audit_decision"]["review_shard_contract"]
            if contract["shard_index"] == 0:
                operation = proposal["local_generated_operations"][0]
                return {
                    "decision": "retry",
                    "summary": "one exact shard repair",
                    "failed_operations_disposition": "retry_required",
                    "replacement_operations": [
                        {
                            "filename": operation["filename"],
                            "content": operation["content"] + "repaired\n",
                        }
                    ],
                }
            return {
                "decision": "apply_available",
                "summary": "other shard approved",
                "failed_operations_disposition": "none",
            }

        result = ingest._review_and_apply_ingest_operations(
            self._oversized_create_ops(),
            raw_content="Only one exact shard requires a repair.",
            reviewer=reviewer,
        )

        assert result["status"] == "needs_retry"
        assert len(result["review"]["replacement_operations"]) == 1
        assert result["review"]["replacement_operations"][0]["filename"].startswith(
            "memory/sharded-"
        )
        assert "frontier_failure" not in result["review"]
        assert result["created"] == []

    def test_low_risk_proposal_still_requires_semantic_consensus(
        self, isolated_wiki: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from chronovisor.ingest import ingest

        raw = next(
            candidate
            for index in range(1000)
            if (candidate := f"ordinary observation {index}")
            and int(ingest._ingest_source_key(candidate, None)[:12], 16) / float(16**12)
            >= 0.10
        )
        captured: list[dict] = []
        authority = self._production_authority("a")

        def production_reviewer(proposal, **_kwargs):
            captured.append(proposal)
            return self._authority_review(
                authority,
                summary="local consensus accepted grounded proposal",
            )

        monkeypatch.setattr(
            ingest,
            "_run_ingest_frontier_review",
            production_reviewer,
        )
        monkeypatch.setattr(
            ingest,
            "_current_ingest_review_authority",
            lambda **_kwargs: (authority, None),
        )

        result = ingest._review_and_apply_ingest_operations(
            [self._create_op()],
            raw_content=raw,
        )

        assert result["status"] == "apply_available"
        assert result["audit"]["mode"] == "local"
        assert len(captured) == 1
        assert (isolated_wiki / "pages" / "memory" / "frontier-only.md").exists()

    def test_low_risk_proposal_cannot_mutate_when_consensus_is_shadowed(
        self, isolated_wiki: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from chronovisor.ingest import ingest

        authority = self._production_authority("a")

        def production_reviewer(_proposal, **_kwargs):
            return {
                "decision": "retry",
                "summary": "decision_lane_shadow:ingest_reconciliation",
                "failed_operations_disposition": "retry_required",
            }

        monkeypatch.setattr(
            ingest,
            "_run_ingest_frontier_review",
            production_reviewer,
        )
        monkeypatch.setattr(
            ingest,
            "_current_ingest_review_authority",
            lambda **_kwargs: (authority, None),
        )

        result = ingest._review_and_apply_ingest_operations(
            [self._create_op()],
            raw_content="ordinary low-risk observation",
        )

        assert result["status"] == "needs_retry"
        assert result["audit"]["mode"] == "local"
        assert not (isolated_wiki / "pages" / "memory" / "frontier-only.md").exists()

    def test_explicit_correction_signal_requires_structured_review(
        self, isolated_wiki: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from chronovisor.ingest import ingest

        captured: list[dict] = []
        authority = self._production_authority("a")

        def production_reviewer(proposal, **_kwargs):
            captured.append(proposal)
            return self._authority_review(
                authority,
                summary="correction is grounded",
            )

        monkeypatch.setattr(
            ingest,
            "_run_ingest_frontier_review",
            production_reviewer,
        )
        monkeypatch.setattr(
            ingest,
            "_current_ingest_review_authority",
            lambda **_kwargs: (authority, None),
        )

        result = ingest._review_and_apply_ingest_operations(
            [self._create_op()],
            raw_content="その記憶は違う。訂正して。",
        )

        assert result["status"] == "apply_available"
        assert result["audit"]["mode"] == "mandatory"
        assert len(captured) == 1

    def test_frontier_confirmed_noop_is_durable_and_non_mutating(
        self, isolated_wiki: Path
    ) -> None:
        from chronovisor.ingest import ingest

        result = ingest._review_and_apply_ingest_operations(
            [self._create_op()],
            raw_content="raw evidence",
            reviewer=lambda _proposal: {
                "decision": "confirmed_noop",
                "summary": "claim is not grounded",
                "failed_operations_disposition": "none",
            },
        )

        assert result["status"] == "confirmed_noop"
        assert result["created"] == []
        assert not (isolated_wiki / "pages" / "memory" / "frontier-only.md").exists()
        proposal_path, review_path = ingest._ingest_artifact_paths(result["source_key"])
        assert proposal_path.exists()
        assert review_path.exists()
        review = json.loads(review_path.read_text(encoding="utf-8"))
        assert review["proposal_sha256"] == result["proposal_sha256"]
        assert (
            review["schema_version"]
            == ingest.INGEST_FRONTIER_REVIEW_ARTIFACT_SCHEMA_VERSION
        )
        assert review["authority"] == {
            "authority_version": 1,
            "lane": "ingest_reconciliation",
            "source": "injected_reviewer_boundary",
        }
        assert review["review"]["decision"] == "confirmed_noop"

    @pytest.mark.parametrize("terminal", ["apply_available", "confirmed_noop"])
    def test_terminal_review_with_repairs_is_normalized_to_retry(
        self,
        terminal: str,
    ) -> None:
        from chronovisor.ingest import ingest

        proposal = {
            "prepared_operations": [{"page_id": "frontier-only"}],
            "failed_operation_specs": [],
        }
        normalized = ingest._normalize_ingest_frontier_review(
            {
                "decision": terminal,
                "summary": "repair this before applying",
                "failed_operations_disposition": "none",
                "invalid_tags": ["t/zeta", "t/alpha", "t/zeta"],
                "replacement_operations": [
                    {"filename": "memory/z.md", "content": "z"},
                    {"filename": "memory/a.md", "content": "a"},
                    {"filename": "memory/a.md", "content": "a"},
                ],
            },
            proposal=proposal,
        )

        assert normalized["decision"] == "retry"
        assert normalized["failed_operations_disposition"] == "retry_required"
        assert normalized["invalid_tags"] == ["t/alpha", "t/zeta"]
        assert normalized["replacement_operations"] == [
            {"filename": "memory/a.md", "content": "a"},
            {"filename": "memory/z.md", "content": "z"},
        ]
        assert "fresh review" in normalized["summary"]

    def test_frontier_retry_keeps_proposal_but_writes_no_verdict_or_page(
        self, isolated_wiki: Path
    ) -> None:
        from chronovisor.ingest import ingest

        result = ingest._review_and_apply_ingest_operations(
            [self._create_op()],
            raw_content="retry evidence",
            reviewer=lambda _proposal: {
                "decision": "needs_retry",
                "summary": "frontier transport unavailable",
            },
        )

        assert result["status"] == "needs_retry"
        proposal_path, review_path = ingest._ingest_artifact_paths(result["source_key"])
        assert proposal_path.exists()
        assert not review_path.exists()
        assert not (isolated_wiki / "pages" / "memory" / "frontier-only.md").exists()

    def test_frontier_approval_reviews_exact_raw_preimage_and_postimage(
        self, isolated_wiki: Path
    ) -> None:
        from chronovisor.ingest import ingest

        original = "---\ntitle: Existing\nupdated: 2026-01-01\n---\nold fact\n"
        path = _seed_page(isolated_wiki, "memory/existing.md", original)
        captured: list[dict] = []

        result = ingest._review_and_apply_ingest_operations(
            [
                {
                    "type": "update",
                    "filename": "memory/existing.md",
                    "content": "## New evidence\nnew fact",
                }
            ],
            raw_content="verbatim raw evidence",
            source_raw="raw/session.md",
            reviewer=lambda proposal: (
                captured.append(proposal)
                or {
                    "decision": "apply_available",
                    "summary": "exact evidence supports it",
                    "failed_operations_disposition": "none",
                }
            ),
        )

        assert result["status"] == "apply_available"
        assert result["updated"] == ["existing"]
        assert "new fact" in path.read_text(encoding="utf-8")
        proposal = captured[0]
        assert proposal["raw_content"] == "verbatim raw evidence"
        exact = proposal["prepared_operations"][0]
        assert exact["previous_text"] == original
        assert exact["previous_sha256"] == hashlib.sha256(original.encode()).hexdigest()
        assert (
            exact["proposed_sha256"]
            == hashlib.sha256(exact["proposed_text"].encode()).hexdigest()
        )

    def test_terminal_artifact_readback_failure_blocks_page_effect(
        self,
        isolated_wiki: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        from chronovisor.ingest import ingest

        monkeypatch.setattr(
            ingest,
            "_load_ingest_review_artifact",
            lambda *_args, **_kwargs: None,
        )

        result = ingest._review_and_apply_ingest_operations(
            [self._create_op()],
            raw_content="readback must be proven",
            reviewer=lambda _proposal: {
                "decision": "apply_available",
                "summary": "injected approval",
                "failed_operations_disposition": "none",
            },
        )

        assert result["status"] == "needs_retry"
        assert result["summary"] == (
            "frontier review artifact readback verification failed"
        )
        assert not (isolated_wiki / "pages" / "memory" / "frontier-only.md").exists()

    def test_page_race_after_reviewed_prepare_fails_closed(
        self, isolated_wiki: Path
    ) -> None:
        from chronovisor.ingest import ingest

        path = _seed_page(
            isolated_wiki,
            "memory/race.md",
            "---\ntitle: Race\nupdated: 2026-01-01\n---\nold\n",
        )
        planned, totals = ingest._prepare_operations(
            [
                {
                    "type": "update",
                    "filename": "memory/race.md",
                    "content": "reviewed proposal",
                }
            ]
        )
        path.write_text(
            "---\ntitle: Race\nupdated: 2026-01-01\n---\nconcurrent correction\n",
            encoding="utf-8",
        )

        with pytest.raises(IngestApplyError, match="page changed before ingest apply"):
            ingest._apply_prepared_operations(planned, link_totals=totals)
        assert "concurrent correction" in path.read_text(encoding="utf-8")
        assert "reviewed proposal" not in path.read_text(encoding="utf-8")

    def test_recovery_only_never_reinstalls_a_reviewed_postimage(
        self, isolated_wiki: Path
    ) -> None:
        from chronovisor.ingest import ingest

        planned, totals = ingest._prepare_operations([self._create_op()])

        with pytest.raises(
            IngestApplyError,
            match="reviewed postimage no longer present during recovery",
        ):
            ingest._apply_prepared_operations(
                planned,
                link_totals=totals,
                recovery_only=True,
            )

        assert not (isolated_wiki / "pages" / "memory" / "frontier-only.md").exists()

    def test_approved_artifact_recovers_without_second_frontier_call(
        self,
        isolated_wiki: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        from chronovisor.ingest import ingest

        real_apply = ingest._apply_prepared_operations
        monkeypatch.setattr(
            ingest,
            "_apply_prepared_operations",
            lambda *_args, **_kwargs: (_ for _ in ()).throw(
                IngestApplyError("simulated power loss before page replace")
            ),
        )
        with pytest.raises(IngestApplyError, match="simulated power loss"):
            ingest._review_and_apply_ingest_operations(
                [self._create_op()],
                raw_content="recoverable raw",
                reviewer=lambda _proposal: {
                    "decision": "apply_available",
                    "summary": "approved before crash",
                    "failed_operations_disposition": "none",
                },
            )

        monkeypatch.setattr(ingest, "_apply_prepared_operations", real_apply)
        result = ingest._review_and_apply_ingest_operations(
            [self._create_op()],
            raw_content="recoverable raw",
            reviewer=lambda _proposal: (_ for _ in ()).throw(
                AssertionError("durable verdict should be reused")
            ),
        )

        assert result["status"] == "apply_available"
        assert result["recovered_artifact"] is True
        assert result["reused_review"] is True
        assert (isolated_wiki / "pages" / "memory" / "frontier-only.md").exists()

    def test_stale_unapplied_review_is_replaced_under_current_authority(
        self,
        isolated_wiki: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        from chronovisor.ingest import ingest

        authority_a = self._production_authority("a")
        authority_b = self._production_authority("b")
        active_authority = {"value": authority_a}
        monkeypatch.setattr(
            ingest,
            "_current_ingest_review_authority",
            lambda **_kwargs: (active_authority["value"], None),
        )
        real_apply = ingest._apply_prepared_operations
        monkeypatch.setattr(
            ingest,
            "_apply_prepared_operations",
            lambda *_args, **_kwargs: (_ for _ in ()).throw(
                IngestApplyError("simulated crash before first page replace")
            ),
        )
        with pytest.raises(IngestApplyError, match="simulated crash"):
            ingest._review_and_apply_ingest_operations(
                [self._create_op()],
                raw_content="stale authority raw",
                reviewer=lambda _proposal: self._authority_review(
                    authority_a,
                    summary="authority a approval",
                ),
            )

        active_authority["value"] = authority_b
        monkeypatch.setattr(ingest, "_apply_prepared_operations", real_apply)
        review_calls: list[str] = []
        result = ingest._review_and_apply_ingest_operations(
            [self._create_op()],
            raw_content="stale authority raw",
            reviewer=lambda _proposal: (
                review_calls.append("fresh")
                or self._authority_review(
                    authority_b,
                    summary="authority b approval",
                )
            ),
        )

        assert result["status"] == "apply_available"
        assert result["reused_review"] is False
        assert result["stale_review_replaced"] == (
            "ingest review authority changed before effect"
        )
        assert review_calls == ["fresh"]
        _proposal_path, review_path = ingest._ingest_artifact_paths(
            result["source_key"]
        )
        artifact = json.loads(review_path.read_text(encoding="utf-8"))
        assert artifact["authority"] == authority_b
        assert artifact["review"]["summary"] == "authority b approval"

    def test_authority_change_at_final_apply_boundary_fails_closed(
        self,
        isolated_wiki: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        from chronovisor.ingest import ingest

        authority_a = self._production_authority("a")
        authority_b = self._production_authority("b")
        resolutions = iter([authority_a, authority_a, authority_b])
        monkeypatch.setattr(
            ingest,
            "_current_ingest_review_authority",
            lambda **_kwargs: (next(resolutions), None),
        )

        result = ingest._review_and_apply_ingest_operations(
            [self._create_op()],
            raw_content="authority race raw",
            reviewer=lambda _proposal: self._authority_review(
                authority_a,
                summary="approval before authority swap",
            ),
        )

        assert result["status"] == "needs_retry"
        assert result["summary"] == "decision authority changed before effect"
        assert result["created"] == []
        assert not (isolated_wiki / "pages" / "memory" / "frontier-only.md").exists()

    def test_exact_postimage_recovery_does_not_reapply_under_stale_authority(
        self,
        isolated_wiki: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        from chronovisor.ingest import ingest

        authority_a = self._production_authority("a")
        authority_b = self._production_authority("b")
        active_authority = {"value": authority_a}
        monkeypatch.setattr(
            ingest,
            "_current_ingest_review_authority",
            lambda **_kwargs: (active_authority["value"], None),
        )
        first = ingest._review_and_apply_ingest_operations(
            [self._create_op()],
            raw_content="already applied authority raw",
            reviewer=lambda _proposal: self._authority_review(
                authority_a,
                summary="authority a approval",
            ),
        )
        assert first["status"] == "apply_available"

        active_authority["value"] = authority_b
        second = ingest._review_and_apply_ingest_operations(
            [self._create_op()],
            raw_content="already applied authority raw",
            reviewer=lambda _proposal: (_ for _ in ()).throw(
                AssertionError("exact postimage recovery must not call reviewer")
            ),
        )

        assert second["status"] == "apply_available"
        assert second["reused_review"] is True
        assert second["recovery_basis"] == "exact_postimages_already_applied"
        assert (isolated_wiki / "pages" / "memory" / "frontier-only.md").exists()

    def test_dry_run_is_completely_read_only(self, isolated_wiki: Path) -> None:
        from chronovisor.ingest import ingest

        before = {
            path.relative_to(isolated_wiki).as_posix(): path.read_bytes()
            for path in isolated_wiki.rglob("*")
            if path.is_file()
        }
        result = ingest._review_and_apply_ingest_operations(
            [self._create_op()],
            raw_content="dry-run raw",
            dry_run=True,
        )
        after = {
            path.relative_to(isolated_wiki).as_posix(): path.read_bytes()
            for path in isolated_wiki.rglob("*")
            if path.is_file()
        }

        assert result["status"] == "dry_run"
        assert result["artifact_written"] is False
        assert after == before
        assert not (isolated_wiki / "pages" / "memory" / "frontier-only.md").exists()


# ---------------------------------------------------------------------------
# Phase 4: raw_keywords frontmatter patch in apply prepare phase
# ---------------------------------------------------------------------------


class TestIngestProposalSchemaCompatibility:
    def test_v2_prepared_payload_rejects_bool_source_index(
        self, isolated_wiki: Path
    ) -> None:
        from chronovisor.ingest import ingest

        operations = [TestIngestFrontierGate._create_op()]
        planned, totals = ingest._prepare_operations(operations)
        proposal = ingest._build_ingest_frontier_proposal(
            raw_content="bool is not an operation index",
            raw_keywords=None,
            source_raw="raw/bool-index.md",
            operations=operations,
            planned=planned,
            link_totals=totals,
        )
        [row] = proposal["prepared_operations"]
        row["source_operation_index"] = True

        assert (
            ingest._prepared_from_review_payload(proposal["prepared_operations"])
            is None
        )

    def test_v1_unreviewed_five_update_incident_is_replaced_by_v2(
        self, isolated_wiki: Path
    ) -> None:
        from chronovisor.ingest import ingest

        raw_content = "legacy five-page incident pending without a review"
        raw_keywords = ["ingest-frontier", "schema-v1"]
        operations: list[dict] = []
        for index in range(5):
            _seed_page(
                isolated_wiki,
                f"memory/legacy-incident-{index}.md",
                (
                    "---\n"
                    f"title: Legacy incident {index}\n"
                    "updated: 2026-07-11\n"
                    "---\n"
                    f"preimage {index}\n"
                ),
            )
            operations.append(
                {
                    "type": "update",
                    "filename": f"memory/legacy-incident-{index}.md",
                    "content": f"grounded addition {index}\n",
                }
            )
        proposal_path, _review_path, legacy_proposal = _write_legacy_v1_ingest_proposal(
            raw_content=raw_content,
            raw_keywords=raw_keywords,
            source_raw="semantic-legacy-incident.md",
            operations=operations,
        )
        legacy_rows = legacy_proposal["prepared_operations"]
        assert isinstance(legacy_rows, list) and len(legacy_rows) == 5
        assert all("source_operation_index" not in row for row in legacy_rows)

        assert (
            ingest._load_pretriage_terminal_recovery(
                raw_content,
                raw_keywords,
                reviewer=None,
            )
            is None
        )
        legacy_bytes = proposal_path.read_bytes()
        assert (
            ingest._load_pretriage_ingest_shard_continuation(
                raw_content,
                raw_keywords,
                reviewer=lambda _proposal: pytest.fail(
                    "legacy classification must not review"
                ),
            )
            is None
        )
        assert proposal_path.read_bytes() == legacy_bytes

        replaced = ingest._review_and_apply_ingest_operations(
            operations,
            raw_content=raw_content,
            raw_keywords=raw_keywords,
            source_raw="semantic-legacy-incident.md",
            reviewer=lambda _proposal: {
                "decision": "apply_available",
                "summary": "replace unreviewed legacy proposal",
                "failed_operations_disposition": "none",
            },
        )

        assert replaced["status"] == "apply_available"
        artifact = json.loads(proposal_path.read_text(encoding="utf-8"))
        assert artifact["schema_version"] == 2
        assert artifact["proposal"]["schema_version"] == 2
        assert all(
            {
                "source_operation_index",
                "source_operation_type",
                "source_filename",
            }
            <= row.keys()
            for row in artifact["proposal"]["prepared_operations"]
        )

    @pytest.mark.parametrize(
        "corruption",
        [
            "partial_json",
            "proposal_digest",
            "raw_binding",
            "partial_continuation",
            "partial_shard_manifest",
            "legacy_review_partial_continuation",
            "legacy_review_partial_shard_manifest",
        ],
    )
    def test_v1_pretriage_regeneration_fails_closed_on_corrupt_or_partial_state(
        self,
        isolated_wiki: Path,
        corruption: str,
    ) -> None:
        from chronovisor.ingest import ingest

        raw_content = "legacy proposal must remain exactly bound before regeneration"
        proposal_path, _review_path, proposal = _write_legacy_v1_ingest_proposal(
            raw_content=raw_content,
            operations=[TestIngestFrontierGate._create_op()],
        )
        if corruption.startswith("legacy_review_"):
            ingest._write_ingest_artifact(
                _review_path,
                {
                    "schema_version": 1,
                    "kind": "ingest_frontier_review_artifact",
                    "source_key": proposal["source_key"],
                    "proposal_sha256": ingest._canonical_json_sha256(proposal),
                    "review": {"decision": "approved"},
                },
            )
        if corruption == "partial_json":
            proposal_path.write_text('{"schema_version":1', encoding="utf-8")
        elif corruption in {
            "partial_continuation",
            "legacy_review_partial_continuation",
        }:
            marker_path = ingest._ingest_review_continuation_marker_path(
                str(proposal["source_key"])
            )
            marker_path.parent.mkdir(parents=True, exist_ok=True)
            marker_path.write_text("{}", encoding="utf-8")
        elif corruption in {
            "partial_shard_manifest",
            "legacy_review_partial_shard_manifest",
        }:
            manifest_path = proposal_path.parent / (
                "review-shard-manifest-" + "f" * 24 + ".json"
            )
            manifest_path.write_text(
                json.dumps({"source_key": proposal["source_key"]}),
                encoding="utf-8",
            )
        else:
            artifact = json.loads(proposal_path.read_text(encoding="utf-8"))
            if corruption == "proposal_digest":
                artifact["proposal_sha256"] = "0" * 64
            else:
                artifact["proposal"]["raw_sha256"] = "0" * 64
                artifact["proposal_sha256"] = ingest._canonical_json_sha256(
                    artifact["proposal"]
                )
            ingest._write_ingest_artifact(proposal_path, artifact)

        with pytest.raises(
            IngestApplyError,
            match=(
                "unreadable"
                if corruption == "partial_json"
                else (
                    "partial continuation evidence"
                    if corruption
                    in {
                        "partial_continuation",
                        "partial_shard_manifest",
                        "legacy_review_partial_continuation",
                        "legacy_review_partial_shard_manifest",
                    }
                    else "binding is invalid"
                )
            ),
        ):
            ingest._load_pretriage_ingest_shard_continuation(
                raw_content,
                None,
                reviewer=lambda _proposal: pytest.fail(
                    "invalid legacy artifact must not review"
                ),
            )

    def test_v1_legacy_review_is_replaced_by_current_sealed_review(
        self, isolated_wiki: Path
    ) -> None:
        from chronovisor.ingest import ingest

        raw_content = "legacy review has no local authority seal"
        target = _seed_page(
            isolated_wiki,
            "memory/legacy-review.md",
            "---\ntitle: Legacy review\nupdated: 2026-07-11\n---\nold\n",
        )
        operations = [
            {
                "type": "update",
                "filename": "memory/legacy-review.md",
                "content": "new grounded fact\n",
            }
        ]
        proposal_path, review_path, proposal = _write_legacy_v1_ingest_proposal(
            raw_content=raw_content,
            operations=operations,
        )
        ingest._write_ingest_artifact(
            review_path,
            {
                "schema_version": 1,
                "kind": "ingest_frontier_review_artifact",
                "source_key": proposal["source_key"],
                "proposal_sha256": ingest._canonical_json_sha256(proposal),
                "review": {"decision": "approved"},
            },
        )

        assert (
            ingest._load_pretriage_terminal_recovery(
                raw_content,
                None,
                reviewer=None,
            )
            is None
        )
        replaced = ingest._review_and_apply_ingest_operations(
            operations,
            raw_content=raw_content,
            reviewer=lambda _proposal: {
                "decision": "apply_available",
                "summary": "replace legacy unsealed review",
                "failed_operations_disposition": "none",
            },
        )

        assert replaced["status"] == "apply_available"
        assert "new grounded fact" in target.read_text(encoding="utf-8")
        assert json.loads(proposal_path.read_text())["schema_version"] == 2
        assert json.loads(review_path.read_text())["schema_version"] == 2

    def test_v1_current_sealed_review_recovers_exact_postimage_without_rebind(
        self, isolated_wiki: Path
    ) -> None:
        from chronovisor.ingest import ingest

        raw_content = "legacy proposal applied before raw acknowledgement"
        seeded = ingest._review_and_apply_ingest_operations(
            [TestIngestFrontierGate._create_op()],
            raw_content=raw_content,
            reviewer=lambda _proposal: {
                "decision": "apply_available",
                "summary": "durable terminal approval",
                "failed_operations_disposition": "none",
            },
        )
        proposal_path, review_path = ingest._ingest_artifact_paths(seeded["source_key"])
        current_review = json.loads(review_path.read_text(encoding="utf-8"))
        legacy_artifact = _downgrade_ingest_proposal_artifact_to_v1(proposal_path)
        legacy_proposal = legacy_artifact["proposal"]
        legacy_sha256 = legacy_artifact["proposal_sha256"]
        sealed_review = ingest._sealed_ingest_review_artifact(
            source_key=seeded["source_key"],
            proposal_sha256=legacy_sha256,
            review=current_review["review"],
            authority=current_review["authority"],
        )
        ingest._write_ingest_artifact(review_path, sealed_review)

        recovered = ingest._load_pretriage_terminal_recovery(
            raw_content,
            None,
            reviewer=None,
        )

        assert isinstance(legacy_proposal, dict)
        assert all(
            "source_operation_index" not in row
            for row in legacy_proposal["prepared_operations"]
        )
        assert recovered is not None
        assert recovered["status"] == "apply_available"
        assert recovered["proposal_sha256"] == legacy_sha256
        assert recovered["recovery_basis"] == "exact_postimages_already_applied"
        assert json.loads(review_path.read_text())["proposal_sha256"] == legacy_sha256


class TestIngestProposalRollback:
    def test_crlf_compact_apply_and_artifact_rollback_are_byte_exact(
        self,
        isolated_wiki: Path,
    ) -> None:
        from chronovisor.ingest import ingest

        original = (
            "---\r\ntitle: CRLF compact\r\nupdated: 2026-07-11\r\n"
            "tags: [d/tools-config, t/reference, s/2026]\r\n---\r\n"
            "# Existing\r\nold fact\r\n"
            "## Product value\r\nproduct value evidence\r\n"
        )
        target = _seed_page(
            isolated_wiki,
            "memory/crlf-compact-rollback.md",
            "placeholder",
        )
        target.write_bytes(original.encode("utf-8"))
        triage_op = {
            "type": "update",
            "filename": "memory/crlf-compact-rollback.md",
            "title": "Product value",
            "keywords": ["product", "value"],
            "summary": "Add product value evidence",
        }
        compact = ingest._build_compact_update_context(
            triage_op,
            "product value evidence",
            max_selected_bytes=8_192,
        )
        assert compact is not None

        result = ingest._review_and_apply_ingest_operations(
            [
                {
                    "type": "update",
                    "filename": triage_op["filename"],
                    "content": "## Grounded append\nnew exact fact\n",
                    "_compact_update_preimage_sha256": compact.page_sha256,
                }
            ],
            raw_content="product value evidence",
            reviewer=lambda _proposal: {
                "decision": "apply_available",
                "summary": "exact CRLF update",
                "failed_operations_disposition": "none",
            },
        )
        proposal_path, _review_path = ingest._ingest_artifact_paths(
            result["source_key"]
        )
        artifact = json.loads(proposal_path.read_text(encoding="utf-8"))
        planned = ingest._prepared_from_review_payload(
            artifact["proposal"]["prepared_operations"]
        )

        assert planned is not None
        assert planned[0].previous_text == original
        assert target.read_bytes() == planned[0].new_body.encode("utf-8")
        assert ingest._prepared_plan_is_recoverable(planned) is True
        assert ingest._prepared_plan_is_fully_applied(planned) is True

        rolled_back = ingest.rollback_ingest_proposal_artifact(
            proposal_path,
            reason="verify exact CRLF restoration",
        )

        assert rolled_back["status"] == "rolled_back"
        assert target.read_bytes() == original.encode("utf-8")
        assert ingest._prepared_plan_is_recoverable(planned) is True
        assert ingest._prepared_plan_is_fully_applied(planned) is False

    def test_partial_apply_failure_restores_exact_crlf_preimage(
        self,
        isolated_wiki: Path,
    ) -> None:
        from chronovisor.ingest import ingest

        original = (
            "---\r\ntitle: CRLF rollback\r\nupdated: 2026-07-11\r\n"
            "tags: [d/tools-config, t/reference, s/2026]\r\n---\r\n"
            "# Existing\r\nold fact\r\n"
        )
        target = _seed_page(
            isolated_wiki,
            "memory/crlf-partial-rollback.md",
            "placeholder",
        )
        target.write_bytes(original.encode("utf-8"))
        planned, totals = ingest._prepare_operations(
            [
                {
                    "type": "update",
                    "filename": "memory/crlf-partial-rollback.md",
                    "content": "## First write\nnew fact",
                    "_compact_update_preimage_sha256": hashlib.sha256(
                        original.encode("utf-8")
                    ).hexdigest(),
                },
                {
                    "type": "create",
                    "filename": "memory/late-conflict.md",
                    "content": (
                        "---\ntitle: Late conflict\nupdated: 2026-07-14\n"
                        "tags: [d/tools-config, t/reference, s/2026]\n---\nbody\n"
                    ),
                },
            ]
        )
        conflict = isolated_wiki / "pages" / "memory" / "late-conflict.md"
        conflict.parent.mkdir(parents=True, exist_ok=True)
        conflict.write_text("independent writer\n", encoding="utf-8")

        with pytest.raises(
            IngestApplyError, match="page appeared before ingest create"
        ):
            ingest._apply_prepared_operations(planned, link_totals=totals)

        assert target.read_bytes() == original.encode("utf-8")
        assert conflict.read_text(encoding="utf-8") == "independent writer\n"

    def test_v1_provenance_free_update_artifact_can_be_rolled_back(
        self, isolated_wiki: Path
    ) -> None:
        from chronovisor.ingest import ingest

        target = _seed_page(
            isolated_wiki,
            "memory/legacy-rollback.md",
            "---\ntitle: Legacy rollback\nupdated: 2026-07-11\n---\nold fact\n",
        )
        previous = target.read_text(encoding="utf-8")
        applied = ingest._review_and_apply_ingest_operations(
            [
                {
                    "type": "update",
                    "filename": "memory/legacy-rollback.md",
                    "content": "new fact\n",
                }
            ],
            raw_content="legacy rollback source",
            reviewer=lambda _proposal: {
                "decision": "apply_available",
                "summary": "apply before legacy rollback",
                "failed_operations_disposition": "none",
            },
        )
        proposal_path, _review_path = ingest._ingest_artifact_paths(
            applied["source_key"]
        )
        legacy = _downgrade_ingest_proposal_artifact_to_v1(proposal_path)

        rolled_back = ingest.rollback_ingest_proposal_artifact(
            proposal_path,
            reason="recover a schema-v1 incident",
        )

        assert legacy["schema_version"] == 1
        assert rolled_back["status"] == "rolled_back"
        assert rolled_back["pages"] == ["legacy-rollback"]
        assert target.read_text(encoding="utf-8") == previous

    def test_exact_applied_postimages_can_be_rolled_back_idempotently(
        self, isolated_wiki: Path
    ) -> None:
        from chronovisor.ingest import ingest

        existing = _seed_page(
            isolated_wiki,
            "memory/existing.md",
            "---\ntitle: Existing\nupdated: 2026-07-11\n---\nold fact\n",
        )
        previous = existing.read_text(encoding="utf-8")
        operations = [
            {
                "type": "create",
                "filename": "memory/new.md",
                "content": "---\ntitle: New\nupdated: 2026-07-12\n---\nnew fact\n",
            },
            {
                "type": "update",
                "filename": "memory/existing.md",
                "content": "additional fact\n",
            },
        ]
        result = ingest._review_and_apply_ingest_operations(
            operations,
            raw_content="grounded raw",
            reviewer=lambda _proposal: {
                "decision": "apply_available",
                "summary": "test approval",
                "failed_operations_disposition": "none",
            },
        )
        proposal_path, _review_path = ingest._ingest_artifact_paths(
            result["source_key"]
        )

        rolled_back = ingest.rollback_ingest_proposal_artifact(
            proposal_path,
            reason="test incident recovery",
        )

        assert rolled_back["status"] == "rolled_back"
        assert set(rolled_back["pages"]) == {"new", "existing"}
        assert not (isolated_wiki / "pages" / "memory" / "new.md").exists()
        assert existing.read_text(encoding="utf-8") == previous
        audit_path = Path(rolled_back["audit_path"])
        audit_before = audit_path.read_bytes()

        repeated = ingest.rollback_ingest_proposal_artifact(
            proposal_path,
            reason="verify idempotence",
        )
        assert repeated["status"] == "already_rolled_back"
        assert set(repeated["already_rolled_back"]) == {"new", "existing"}
        assert repeated["audit_written"] is False
        assert audit_path.read_bytes() == audit_before

    def test_rollback_aborts_before_writes_when_any_target_diverged(
        self, isolated_wiki: Path
    ) -> None:
        from chronovisor.ingest import ingest

        result = ingest._review_and_apply_ingest_operations(
            [TestIngestFrontierGate._create_op()],
            raw_content="grounded raw with later edit",
            reviewer=lambda _proposal: {
                "decision": "apply_available",
                "summary": "test approval",
                "failed_operations_disposition": "none",
            },
        )
        target = isolated_wiki / "pages" / "memory" / "frontier-only.md"
        target.write_text("a later independent edit\n", encoding="utf-8")
        proposal_path, _review_path = ingest._ingest_artifact_paths(
            result["source_key"]
        )

        rolled_back = ingest.rollback_ingest_proposal_artifact(
            proposal_path,
            reason="must not overwrite later edit",
        )

        assert rolled_back["status"] == "conflict"
        assert rolled_back["page_id"] == "frontier-only"
        assert target.read_text(encoding="utf-8") == "a later independent edit\n"


class TestApplyRawKeywordsPatch:
    """``_apply_operations`` patches ``raw_keywords`` onto the page
    frontmatter in the prepare phase only — never in the write phase. The
    write phase keeps the existing single-atomic-write contract so partial
    failure rolls back to the pre-batch state.
    """

    def test_create_writes_raw_keywords_to_frontmatter(
        self, isolated_wiki: Path
    ) -> None:
        ops = [
            {
                "type": "create",
                "filename": "misc/p.md",
                "content": "---\ntitle: P\nupdated: 2026-04-28\n---\nbody",
                "raw_keywords": ["alpha", "beta"],
            }
        ]
        _apply_operations(ops)
        text = (isolated_wiki / "pages" / "misc" / "p.md").read_text()
        assert "raw_keywords: [alpha, beta]" in text
        assert "title: P" in text

    def test_create_without_raw_keywords_leaves_field_absent(
        self, isolated_wiki: Path
    ) -> None:
        """When the op carries no raw_keywords (e.g. raw frontmatter had
        none), the resulting page must not gain a stray empty field."""
        ops = [
            {
                "type": "create",
                "filename": "misc/q.md",
                "content": "---\ntitle: Q\nupdated: 2026-04-28\n---\nbody",
            }
        ]
        _apply_operations(ops)
        text = (isolated_wiki / "pages" / "misc" / "q.md").read_text()
        assert "raw_keywords" not in text

    def test_create_empty_list_skips_patch(self, isolated_wiki: Path) -> None:
        """Empty list = no information — don't bloat the frontmatter."""
        ops = [
            {
                "type": "create",
                "filename": "misc/r.md",
                "content": "---\ntitle: R\nupdated: 2026-04-28\n---\nbody",
                "raw_keywords": [],
            }
        ]
        _apply_operations(ops)
        text = (isolated_wiki / "pages" / "misc" / "r.md").read_text()
        assert "raw_keywords" not in text

    def test_update_unions_with_existing_preserving_order(
        self, isolated_wiki: Path
    ) -> None:
        _seed_page(
            isolated_wiki,
            "career/x.md",
            "---\ntitle: X\nupdated: 2026-01-01\nraw_keywords: [a, b]\n---\noriginal\n",
        )
        ops = [
            {
                "type": "update",
                "filename": "x.md",
                "content": "## addendum",
                "raw_keywords": ["b", "c", "d"],
            }
        ]
        _apply_operations(ops)
        text = (isolated_wiki / "pages" / "career" / "x.md").read_text()
        # Order-preserving dedupe: a, b come from existing; c, d are appended.
        assert "raw_keywords: [a, b, c, d]" in text
        # ``updated:`` was bumped to today as part of the existing contract.
        assert f"updated: {date.today().isoformat()}" in text
        # Body append still works.
        assert "## addendum" in text and "original" in text

    def test_update_recovers_from_broken_existing_value(
        self, isolated_wiki: Path
    ) -> None:
        """If a page's existing raw_keywords field is malformed (wrong
        type, manual edit, legacy), apply must self-heal by treating the
        existing value as empty rather than aborting the whole op."""
        _seed_page(
            isolated_wiki,
            "misc/y.md",
            # Scalar instead of list — broken shape.
            "---\ntitle: Y\nupdated: 2026-01-01\nraw_keywords: oops\n---\nbody\n",
        )
        ops = [
            {
                "type": "update",
                "filename": "y.md",
                "content": "## more",
                "raw_keywords": ["clean1", "clean2"],
            }
        ]
        _apply_operations(ops)
        text = (isolated_wiki / "pages" / "misc" / "y.md").read_text()
        assert "raw_keywords: [clean1, clean2]" in text

    def test_update_without_raw_keywords_leaves_existing_intact(
        self, isolated_wiki: Path
    ) -> None:
        """An update op with no raw_keywords must not erase or rewrite
        the existing field."""
        _seed_page(
            isolated_wiki,
            "misc/z.md",
            "---\ntitle: Z\nupdated: 2026-01-01\nraw_keywords: [keep, me]\n---\nbody\n",
        )
        ops = [
            {
                "type": "update",
                "filename": "z.md",
                "content": "## tail",
            }
        ]
        _apply_operations(ops)
        text = (isolated_wiki / "pages" / "misc" / "z.md").read_text()
        assert "raw_keywords: [keep, me]" in text

    def test_write_phase_rollback_restores_pre_batch_text(
        self, isolated_wiki: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Phase 4 must keep the prepare/write split intact — the
        rollback path restores the on-disk text as it was BEFORE the
        batch ran, not as it was after the in-memory raw_keywords patch.
        """
        original = (
            "---\ntitle: A\nupdated: 2026-01-01\nraw_keywords: [old]\n---\nbody\n"
        )
        path_a = _seed_page(isolated_wiki, "misc/a.md", original)

        # Make atomic_write fail on the SECOND op so the first op's write
        # is committed and then must be rolled back.
        from chronovisor.core import link_fix

        original_atomic = link_fix.atomic_write
        call_count = {"n": 0}

        def flaky_atomic(p, content):
            call_count["n"] += 1
            if call_count["n"] == 2:
                raise OSError("simulated disk full on op 2")
            return original_atomic(p, content)

        monkeypatch.setattr(link_fix, "atomic_write", flaky_atomic)

        ops = [
            {
                "type": "update",
                "filename": "a.md",
                "content": "## first",
                "raw_keywords": ["new1"],
            },
            {
                "type": "create",
                "filename": "misc/never.md",
                "content": "---\ntitle: N\nupdated: 2026-04-28\n---\nbody",
                "raw_keywords": ["nope"],
            },
        ]
        with pytest.raises(IngestApplyError):
            _apply_operations(ops)

        # The first op was rolled back to ORIGINAL — not to the
        # raw_keywords-patched intermediate.
        assert path_a.read_text() == original


# ---------------------------------------------------------------------------
# Path-traversal sanitization (R2-High)
# ---------------------------------------------------------------------------


class TestSafeResolvePagePath:
    def test_relative_ok(self, isolated_wiki: Path) -> None:
        from chronovisor.ingest.ingest import _safe_resolve_page_path

        out = _safe_resolve_page_path("ai/foo.md")
        pages = isolated_wiki / "pages"
        assert out == (pages / "ai" / "foo.md").resolve()

    def test_relative_without_md_suffix(self, isolated_wiki: Path) -> None:
        from chronovisor.ingest.ingest import _safe_resolve_page_path

        out = _safe_resolve_page_path("ai/foo")
        assert out.name == "foo.md"

    def test_absolute_rejected(self, isolated_wiki: Path) -> None:
        from chronovisor.ingest.ingest import _safe_resolve_page_path

        with pytest.raises(IngestApplyError, match="absolute filename"):
            _safe_resolve_page_path("/etc/passwd")

    def test_parent_traversal_rejected(self, isolated_wiki: Path) -> None:
        from chronovisor.ingest.ingest import _safe_resolve_page_path

        with pytest.raises(IngestApplyError, match="parent-traversal"):
            _safe_resolve_page_path("../../etc/passwd.md")
        with pytest.raises(IngestApplyError, match="parent-traversal"):
            _safe_resolve_page_path("ai/../../../etc/passwd.md")

    def test_empty_or_dot_md_rejected(self, isolated_wiki: Path) -> None:
        from chronovisor.ingest.ingest import _safe_resolve_page_path

        with pytest.raises(IngestApplyError):
            _safe_resolve_page_path("")
        with pytest.raises(IngestApplyError):
            _safe_resolve_page_path("   ")

    def test_apply_rejects_traversal_before_writing(self, isolated_wiki: Path) -> None:
        # Even a single traversal op poisons the whole batch — nothing writes.
        good = {
            "type": "create",
            "filename": "ok/safe.md",
            "content": "---\ntitle: T\nupdated: 2026-04-28\n---\nbody",
        }
        evil = {
            "type": "create",
            "filename": "../../tmp/escape.md",
            "content": "---\ntitle: E\nupdated: 2026-04-28\n---\nx",
        }
        with pytest.raises(IngestApplyError, match="parent-traversal"):
            _apply_operations([good, evil])
        # Confirm neither file was created.
        pages = isolated_wiki / "pages"
        assert not (pages / "ok" / "safe.md").exists()


# ---------------------------------------------------------------------------
# Update body — partial frontmatter rejection (R2-Medium)
# ---------------------------------------------------------------------------


class TestUpdatePartialFrontmatter:
    def test_unclosed_frontmatter_in_update_rejected(self) -> None:
        # Opening `---` with no closing — _strip_all_frontmatter can't
        # remove it, so the contract demands we reject rather than append.
        out = _extract_page_body(
            "=== UPDATE PAGE: foo.md ===\n"
            "---\ntitle: X\nupdated: 2026-04-28\n"
            "extra body but no closing fence\n"
            "=== END PAGE ===",
            op_type="update",
        )
        assert out is None

    def test_closed_frontmatter_then_body_in_update(self) -> None:
        # Closed FM followed by real body → strip the FM, keep the body.
        out = _extract_page_body(
            "=== UPDATE PAGE: foo.md ===\n"
            "---\ntitle: X\nupdated: 2026-04-28\n---\n"
            "## section\nnotes\n"
            "=== END PAGE ===",
            op_type="update",
        )
        assert out is not None
        assert "title:" not in out
        assert "## section" in out


# ---------------------------------------------------------------------------
# Unclosed fenced code preservation (R2-Medium)
# ---------------------------------------------------------------------------


class TestUnclosedFence:
    def test_unclosed_fence_protects_trailing_subscript(self) -> None:
        # Truncated LLM output: the fence opens but never closes. Everything
        # after the opener must be treated as code so we don't eat
        # `data[[1]]` -> `data1`.
        text = (
            "intro [[foo]] mid\n```python\nx = data[[1]]\ny = also[[2]]\n"
            # NOTE: no closing fence
        )
        out, stats = _reconcile_links(text, {"foo"})
        assert "x = data[[1]]" in out
        assert "y = also[[2]]" in out
        assert "[[foo]]" in out
        assert stats["resolved"] == 1
        assert stats["unwrapped"] == 0


# ---------------------------------------------------------------------------
# run_ingest integration: partial generate failure (R2-Critical)
# ---------------------------------------------------------------------------


class TestRunIngestPartialFailure:
    def test_all_generation_failure_uses_matching_representative(self) -> None:
        from chronovisor.ingest import ingest

        error = ingest._all_generation_failure_error(
            [{"filename": "memory/first.md"}, {"filename": "memory/second.md"}],
            [],
            [
                {
                    "filename": "memory/first.md",
                    "failure_class": "validation_failed",
                    "error": "first validation detail",
                },
                {
                    "filename": "memory/second.md",
                    "failure_class": "transport_error",
                    "error": "second transport detail",
                },
            ],
        )

        assert error is not None
        assert error.startswith("ingest generation transport_error:")
        assert "first=memory/second.md: second transport detail" in error

    def test_all_generation_failure_bypasses_semantic_review(
        self, isolated_wiki: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from chronovisor.core import jobs
        from chronovisor.ingest import ingest

        monkeypatch.setattr(
            ingest,
            "_triage",
            lambda _content: [
                {
                    "type": "update",
                    "filename": "memory/generation-failed.md",
                    "title": "Generation failed",
                }
            ],
        )

        def failed_generation(_op, _raw, *, diagnostics=None, **_kwargs):
            assert diagnostics is not None
            diagnostics.update(
                {
                    "failure_class": "repair_exhausted",
                    "reason": "the page boundary remained invalid",
                    "attempts": 3,
                }
            )
            return None

        monkeypatch.setattr(ingest, "_generate_one", failed_generation)
        monkeypatch.setattr(ingest, "is_available", lambda: True)
        review_calls: list[object] = []
        monkeypatch.setattr(
            ingest,
            "_review_and_apply_ingest_operations",
            lambda *_args, **_kwargs: review_calls.append(True),
        )
        job = jobs.job_store.create(processor="ollama")

        ingest.run_ingest(
            "grounded raw",
            job.job_id,
            frontier_reviewer=ingest._run_ingest_frontier_review,
        )

        finished = jobs.job_store.get(job.job_id)
        assert finished.status == jobs.JobStatus.FAILED
        assert str(finished.error).startswith(
            "ingest generation repair_exhausted: all 1 planned page operations"
        )
        assert review_calls == []

    def test_structured_review_failure_does_not_regenerate_or_exhaust_budget(
        self, isolated_wiki: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from chronovisor.core import jobs
        from chronovisor.decision import routine_review
        from chronovisor.ingest import failure_supervisor, ingest

        plan = [
            {
                "type": "create",
                "filename": "memory/reviewer-control-failure.md",
                "title": "Reviewer control failure",
            }
        ]
        monkeypatch.setattr(ingest, "_triage", lambda _content, **_kwargs: plan)
        generation_calls: list[str] = []

        def generate(op, _raw, **_kwargs):
            generation_calls.append(op["filename"])
            return {
                "type": "create",
                "filename": op["filename"],
                "content": (
                    "---\ntitle: Reviewer control failure\nupdated: 2026-07-14\n"
                    "---\nbody\n"
                ),
            }

        review_calls: list[dict] = []

        def reviewer(proposal: dict) -> dict:
            review_calls.append(proposal)
            return routine_review._validated_structured_result(
                None,
                ingest.INGEST_FRONTIER_DECISION_SCHEMA,
                reviewer="local_consensus",
            )

        monkeypatch.setattr(ingest, "_generate_one", generate)
        monkeypatch.setattr(ingest, "is_available", lambda: True)
        job = jobs.job_store.create(processor="ollama")

        ingest.run_ingest(
            "grounded source",
            job.job_id,
            frontier_reviewer=reviewer,
        )

        finished = jobs.job_store.get(job.job_id)
        assert finished.status == jobs.JobStatus.FAILED
        assert generation_calls == ["memory/reviewer-control-failure.md"]
        assert len(review_calls) == 1
        assert str(finished.error).startswith(
            "local consensus authority unavailable: schema_invalid: "
        )
        assert "did not converge" not in str(finished.error)
        classified = failure_supervisor.classify_failure(finished.error)
        assert classified.failure_class == (
            "ingest.runtime_local_consensus_authority_unavailable"
        )
        assert classified.fingerprint.endswith(":schema_invalid")

    @pytest.mark.parametrize(
        ("authority_sha256", "semantic_defer"),
        [("d" * 64, True), (None, False), ("not-a-valid-sha", False)],
        ids=("valid-authority", "missing-authority", "invalid-authority"),
    )
    def test_three_way_semantic_no_quorum_does_not_regenerate(
        self,
        isolated_wiki: Path,
        monkeypatch: pytest.MonkeyPatch,
        authority_sha256: str | None,
        semantic_defer: bool,
    ) -> None:
        from chronovisor.core import jobs
        from chronovisor.ingest import ingest

        plan = [
            {
                "type": "create",
                "filename": "memory/semantic-no-quorum.md",
                "title": "Semantic no quorum",
            }
        ]
        monkeypatch.setattr(ingest, "_triage", lambda _content, **_kwargs: plan)
        generation_calls: list[str] = []

        def generate(op, _raw, **_kwargs):
            generation_calls.append(op["filename"])
            return {
                "type": "create",
                "filename": op["filename"],
                "content": (
                    "---\ntitle: Semantic no quorum\nupdated: 2026-07-14\n---\nbody\n"
                ),
            }

        review_calls: list[dict] = []

        def reviewer(proposal: dict) -> dict:
            review_calls.append(proposal)
            review = {
                "decision": "retry",
                "summary": "local_models_did_not_reach_two_vote_quorum",
                "failed_operations_disposition": "none",
                "frontier_failure": {
                    "failure_class": "local_semantic_no_quorum",
                    "rescue_status": "local_quarantined",
                    "summary": "local_models_did_not_reach_two_vote_quorum",
                    "human_required": False,
                    "notify_user": False,
                },
                "human_required": False,
                "reviewer": "local_consensus",
                "local_consensus": {
                    "status": "quarantined",
                    "ok": False,
                    "failure_class": "local_consensus_failed",
                    "quarantine_reason": ("local_models_did_not_reach_two_vote_quorum"),
                    "votes": [
                        {"valid": True, "signature_sha256": "a" * 64},
                        {"valid": True, "signature_sha256": "b" * 64},
                        {"valid": True, "signature_sha256": "c" * 64},
                    ],
                },
            }
            if authority_sha256 is not None:
                review["decision_policy"] = {
                    "router_policy": {"artifact_sha256": authority_sha256}
                }
            return review

        monkeypatch.setattr(ingest, "_generate_one", generate)
        monkeypatch.setattr(ingest, "is_available", lambda: True)
        job = jobs.job_store.create(processor="ollama")

        ingest.run_ingest(
            "grounded source with a semantic disagreement",
            job.job_id,
            frontier_reviewer=reviewer,
        )

        finished = jobs.job_store.get(job.job_id)
        assert finished.status == jobs.JobStatus.FAILED
        if semantic_defer:
            assert str(finished.error) == (
                "local consensus semantic no quorum "
                f"[authority_sha256={authority_sha256}]: "
                "local_models_did_not_reach_two_vote_quorum"
            )
            assert "authority unavailable" not in str(finished.error)
        else:
            assert str(finished.error) == (
                "local consensus authority unavailable: "
                "local_semantic_no_quorum: "
                "local_models_did_not_reach_two_vote_quorum"
            )
        assert generation_calls == ["memory/semantic-no-quorum.md"]
        assert len(review_calls) == 1
        assert not (
            isolated_wiki / "pages" / "memory" / "semantic-no-quorum.md"
        ).exists()

    def test_structured_frontier_failure_is_not_actionable_page_feedback(
        self,
    ) -> None:
        from chronovisor.ingest import ingest

        assert (
            ingest._frontier_retry_is_actionable(
                {
                    "status": "needs_retry",
                    "review": {
                        "decision": "retry",
                        "summary": "repair the response",
                        "frontier_failure": {"failure_class": "schema_invalid"},
                    },
                }
            )
            is False
        )

    def test_completion_callback_failure_overrides_completed_job(
        self, isolated_wiki: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Applied pages remain durable, but raw ACK failure is job failure."""

        from chronovisor.core import jobs
        from chronovisor.ingest import ingest

        monkeypatch.setattr(
            ingest,
            "_triage",
            lambda _content: [
                {
                    "type": "create",
                    "filename": "memory/ack-boundary.md",
                    "title": "ACK Boundary",
                }
            ],
        )
        monkeypatch.setattr(
            ingest,
            "_generate_one",
            lambda op, _raw, **_kwargs: {
                "type": "create",
                "filename": op["filename"],
                "content": ("---\ntitle: ACK Boundary\nupdated: 2026-07-14\n---\nbody"),
            },
        )
        monkeypatch.setattr(ingest, "is_available", lambda: True)
        finally_calls: list[dict[str, bool]] = []
        job = jobs.job_store.create(processor="ollama")

        ingest.run_ingest(
            "raw completion boundary",
            job.job_id,
            on_complete=lambda: (_ for _ in ()).throw(
                RuntimeError("durable ACK write failed")
            ),
            on_finally=lambda failed, triage_failed: finally_calls.append(
                {"failed": failed, "triage_failed": triage_failed}
            ),
            frontier_reviewer=ingest._run_ingest_frontier_review,
        )

        finished = jobs.job_store.get(job.job_id)
        assert finished is not None
        assert finished.status == jobs.JobStatus.FAILED
        assert finished.error == "durable ACK write failed"
        assert finally_calls == [{"failed": True, "triage_failed": False}]
        assert (isolated_wiki / "pages" / "memory" / "ack-boundary.md").exists()

    def test_confirmed_noop_revalidates_authority_before_retiring_raw(
        self, isolated_wiki: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from chronovisor.core import jobs
        from chronovisor.ingest import ingest

        monkeypatch.setattr(ingest, "_triage", lambda _content: [])
        monkeypatch.setattr(ingest, "is_available", lambda: True)
        authority_before = TestIngestFrontierGate._production_authority("a")
        authority_after = TestIngestFrontierGate._production_authority("b")
        authority_checks = 0

        def changing_authority(**_kwargs):
            nonlocal authority_checks
            authority_checks += 1
            return (
                (authority_before, None)
                if authority_checks == 1
                else (authority_after, None)
            )

        monkeypatch.setattr(
            ingest,
            "_current_ingest_review_authority",
            changing_authority,
        )
        monkeypatch.setattr(
            ingest,
            "_review_and_apply_ingest_operations",
            lambda *_args, **_kwargs: {
                "status": "confirmed_noop",
                "source_key": "noop-source",
                "proposal_sha256": "a" * 64,
                "review": {"decision": "confirmed_noop"},
                "authority": authority_before,
                "created": [],
                "updated": [],
                "audit": {},
            },
        )
        completed: list[bool] = []
        job = jobs.job_store.create(processor="ollama")

        ingest.run_ingest(
            "raw content",
            job.job_id,
            on_complete=lambda: completed.append(True),
        )

        finished = jobs.job_store.get(job.job_id)
        assert finished.status == jobs.JobStatus.FAILED
        assert "authority changed before raw retirement" in str(finished.error)
        assert completed == []

    def test_confirmed_noop_recomputes_action_proof_before_retiring_raw(
        self, isolated_wiki: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from chronovisor.core import jobs
        from chronovisor.ingest import ingest

        authority = TestIngestFrontierGate._production_authority("a")
        review = TestIngestFrontierGate._authority_review(
            authority,
            decision="confirmed_noop",
        )
        review["failed_operations_disposition"] = "retry_required"
        monkeypatch.setattr(ingest, "_triage", lambda _content: [])
        monkeypatch.setattr(ingest, "is_available", lambda: True)
        monkeypatch.setattr(
            ingest,
            "_review_and_apply_ingest_operations",
            lambda *_args, **_kwargs: {
                "status": "confirmed_noop",
                "source_key": "noop-source",
                "proposal_sha256": "a" * 64,
                "review": review,
                "authority": authority,
                "created": [],
                "updated": [],
                "audit": {},
            },
        )
        monkeypatch.setattr(
            ingest,
            "_current_ingest_review_authority",
            lambda **_kwargs: (authority, None),
        )
        completed: list[bool] = []
        job = jobs.job_store.create(processor="ollama")

        ingest.run_ingest(
            "raw content",
            job.job_id,
            on_complete=lambda: completed.append(True),
        )

        finished = jobs.job_store.get(job.job_id)
        assert finished.status == jobs.JobStatus.FAILED
        assert "proof invalid before raw retirement" in str(finished.error)
        assert completed == []

    def test_partial_generate_applies_successful_ops(
        self, isolated_wiki: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Contract: a partial generate failure (2 of 3 ops succeed, even
        after the per-op retry) writes the 2 successful pages, marks raws
        processed (so the next tick won't re-triage and collide on stem),
        and records the failed op in ``job.result`` for autonomous follow-up.

        Replaces the prior 'discard everything on any failure' contract.
        Discarding both halved the data the wiki captured AND looped on
        raws that kept failing; partial apply + raws-processed avoids both.
        """

        from chronovisor.core import jobs
        from chronovisor.ingest import ingest

        # Stub triage → returns 3 ops.
        plan = [
            {"type": "create", "filename": f"misc/p{i}.md", "title": f"P{i}"}
            for i in range(3)
        ]
        monkeypatch.setattr(ingest, "_triage", lambda _content: plan)

        # Stub generate: always succeed for p0/p1, always fail for p2 (so
        # the retry also fails — exercises the dead-letter path).
        def fake_generate(op: dict, _raw: str, **_kw) -> dict | None:
            if op["filename"].endswith("p2.md"):
                return None
            return {
                "type": "create",
                "filename": op["filename"],
                "content": ("---\ntitle: X\nupdated: 2026-04-28\n---\nbody"),
            }

        monkeypatch.setattr(ingest, "_generate_one", fake_generate)
        monkeypatch.setattr(ingest, "is_available", lambda: True)

        on_complete_called = []
        on_finally_calls = []

        job = jobs.job_store.create(processor="ollama")
        ingest.run_ingest(
            "raw content",
            job.job_id,
            on_complete=lambda: on_complete_called.append(True),
            on_finally=lambda failed, triage_failed: on_finally_calls.append(
                {"failed": failed, "triage_failed": triage_failed}
            ),
            frontier_reviewer=ingest._run_ingest_frontier_review,
        )

        # on_complete fires → raws marked processed (no infinite retry).
        assert on_complete_called == [True]
        # on_finally fires with failed=False (we did write pages successfully).
        assert on_finally_calls == [{"failed": False, "triage_failed": False}]
        # Job COMPLETED with partial flag + failed_ops in result.
        finished = jobs.job_store.get(job.job_id)
        assert finished.status == jobs.JobStatus.COMPLETED
        assert sorted(finished.pages_created) == ["p0", "p1"]
        assert finished.pages_updated == []
        assert finished.result is not None
        assert finished.result["local_consensus"] == finished.result["frontier"]
        assert finished.result.get("partial") is True
        failed_ops = finished.result.get("failed_ops", [])
        assert len(failed_ops) == 1
        assert failed_ops[0]["filename"].endswith("p2.md")
        # Disk: p0 and p1 were written; p2 was not.
        pages = isolated_wiki / "pages"
        assert (pages / "misc" / "p0.md").exists()
        assert (pages / "misc" / "p1.md").exists()
        assert not (pages / "misc" / "p2.md").exists()

    def test_partial_generate_does_not_blindly_restart_generation_session(
        self, isolated_wiki: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The outer loop must not replay an operation without feedback.

        Real page-format repair is owned by ``_generate_one`` so it can retain
        the invalid assistant turn and validator reason.  A mocked terminal
        failure therefore crosses this boundary exactly once.
        """

        from chronovisor.core import jobs
        from chronovisor.ingest import ingest

        plan = [
            {"type": "create", "filename": "misc/p0.md", "title": "P0"},
            {"type": "create", "filename": "misc/p1.md", "title": "P1"},
        ]
        monkeypatch.setattr(ingest, "_triage", lambda _content: plan)

        attempts: dict[str, int] = {}

        def flaky_generate(op: dict, _raw: str, **_kw) -> dict | None:
            fname = op["filename"]
            attempts[fname] = attempts.get(fname, 0) + 1
            if fname.endswith("p1.md"):
                return None
            return {
                "type": "create",
                "filename": fname,
                "content": "---\ntitle: X\nupdated: 2026-04-28\n---\nbody",
            }

        monkeypatch.setattr(ingest, "_generate_one", flaky_generate)
        monkeypatch.setattr(ingest, "is_available", lambda: True)

        on_complete_called = []
        job = jobs.job_store.create(processor="ollama")
        ingest.run_ingest(
            "raw content",
            job.job_id,
            on_complete=lambda: on_complete_called.append(True),
            frontier_reviewer=ingest._run_ingest_frontier_review,
        )

        assert attempts.get("misc/p1.md") == 1
        assert attempts.get("misc/p0.md") == 1
        finished = jobs.job_store.get(job.job_id)
        assert finished.status == jobs.JobStatus.COMPLETED
        assert finished.pages_created == ["p0"]
        assert finished.result is not None and finished.result.get("partial") is True
        assert finished.result["failed_ops"][0]["attempts"] == 1
        assert on_complete_called == [True]

    def test_all_ops_fail_marks_raw_processed(
        self, isolated_wiki: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """If every generate op fails (even after retry), the wiki must
        not be mutated, but on_complete still fires so one bad model
        output does not block the pending drain forever."""

        from chronovisor.core import jobs
        from chronovisor.ingest import ingest

        plan = [
            {"type": "create", "filename": f"misc/p{i}.md", "title": f"P{i}"}
            for i in range(2)
        ]
        monkeypatch.setattr(ingest, "_triage", lambda _content: plan)
        monkeypatch.setattr(ingest, "_generate_one", lambda _op, _raw, **_kw: None)
        monkeypatch.setattr(ingest, "is_available", lambda: True)

        on_complete_called = []
        on_finally_calls = []

        job = jobs.job_store.create(processor="ollama")
        ingest.run_ingest(
            "raw content",
            job.job_id,
            on_complete=lambda: on_complete_called.append(True),
            on_finally=lambda failed, triage_failed: on_finally_calls.append(
                {"failed": failed, "triage_failed": triage_failed}
            ),
            frontier_reviewer=ingest._run_ingest_frontier_review,
        )

        # Nothing succeeded, but raws are marked processed to avoid an
        # infinite retry loop on deterministic malformed output.
        assert on_complete_called == [True]
        assert on_finally_calls == [{"failed": False, "triage_failed": False}]
        finished = jobs.job_store.get(job.job_id)
        assert finished.status == jobs.JobStatus.COMPLETED
        assert finished.pages_created == []
        assert finished.pages_updated == []
        assert finished.result is not None
        assert finished.result.get("partial") is True
        assert len(finished.result.get("failed_ops", [])) == 2
        # Disk untouched.
        pages = isolated_wiki / "pages"
        assert not (pages / "misc" / "p0.md").exists()
        assert not (pages / "misc" / "p1.md").exists()

    def test_missing_update_target_is_generated_as_create(
        self, isolated_wiki: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A model can misclassify a brand-new topic as an update.

        If the requested target is absent and the filename is safe for a new
        page, normalize the op before generation instead of letting apply
        quarantine the raw with update_target_not_found.
        """

        from chronovisor.core import jobs
        from chronovisor.ingest import ingest

        plan = [
            {
                "type": "update",
                "filename": "career/career-transition-strategy-2026.md",
                "summary": "Capture career transition strategy discussion",
                "keywords": ["career", "strategy"],
            }
        ]
        seen_ops: list[dict] = []
        monkeypatch.setattr(ingest, "_triage", lambda _content: plan)

        def fake_generate(op: dict, _raw: str, *, raw_keywords=None, **_kw) -> dict:
            seen_ops.append(op)
            generated = {
                "type": op["type"],
                "filename": op["filename"],
                "content": (
                    "---\n"
                    "title: Career Transition Strategy 2026\n"
                    "updated: 2026-06-23\n"
                    "tags: [d/personal-strategy, t/analysis, s/2026]\n"
                    "---\n"
                    "body"
                ),
            }
            if raw_keywords is not None:
                generated["raw_keywords"] = raw_keywords
            return generated

        monkeypatch.setattr(ingest, "_generate_one", fake_generate)
        monkeypatch.setattr(ingest, "is_available", lambda: True)

        on_complete_called = []
        job = jobs.job_store.create(processor="ollama")
        ingest.run_ingest(
            "raw content",
            job.job_id,
            on_complete=lambda: on_complete_called.append(True),
            frontier_reviewer=ingest._run_ingest_frontier_review,
        )

        assert seen_ops[0]["type"] == "create"
        assert seen_ops[0]["title"] == "Career Transition Strategy 2026"
        finished = jobs.job_store.get(job.job_id)
        assert finished.status == jobs.JobStatus.COMPLETED
        assert finished.pages_created == ["career-transition-strategy-2026"]
        assert finished.pages_updated == []
        assert on_complete_called == [True]
        assert (
            isolated_wiki
            / "pages"
            / "career"
            / "career-transition-strategy-2026.md"
        ).exists()

    def test_failure_packet_missing_update_target_becomes_create(
        self, isolated_wiki: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from chronovisor.core import jobs
        from chronovisor.ingest import ingest

        plan = [
            {
                "type": "update",
                "filename": "ai/claude-code-vs-claude-code-structural-analysis.md",
                "summary": "Capture Claude Code and Cursor adoption analysis",
            }
        ]
        seen_ops: list[dict] = []
        monkeypatch.setattr(ingest, "_triage", lambda _content: plan)

        def fake_generate(op: dict, _raw: str, *, raw_keywords=None, **_kw) -> dict:
            seen_ops.append(op)
            generated = {
                "type": op["type"],
                "filename": op["filename"],
                "content": (
                    "---\n"
                    "title: Claude Code Vs Claude Code Structural Analysis\n"
                    "updated: 2026-06-28\n"
                    "tags: [d/ai-tools, t/analysis, s/2026]\n"
                    "---\n"
                    "body"
                ),
            }
            if raw_keywords is not None:
                generated["raw_keywords"] = raw_keywords
            return generated

        monkeypatch.setattr(ingest, "_generate_one", fake_generate)
        monkeypatch.setattr(ingest, "is_available", lambda: True)

        job = jobs.job_store.create(processor="ollama")
        ingest.run_ingest(
            "raw content",
            job.job_id,
            metadata={"raw_keywords": ["Claude Code", "Cursor", "Mac Studio"]},
            frontier_reviewer=ingest._run_ingest_frontier_review,
        )

        assert seen_ops[0]["type"] == "create"
        assert seen_ops[0]["keywords"] == [
            "claude",
            "code",
            "vs",
            "claude",
            "code",
            "structural",
            "analysis",
        ]
        finished = jobs.job_store.get(job.job_id)
        assert finished.status == jobs.JobStatus.COMPLETED
        assert finished.pages_created == [
            "claude-code-vs-claude-code-structural-analysis"
        ]
        assert finished.pages_updated == []
        page = (
            isolated_wiki
            / "pages"
            / "ai"
            / "claude-code-vs-claude-code-structural-analysis.md"
        )
        assert page.exists()
        assert (
            'raw_keywords: ["Claude Code", Cursor, "Mac Studio"]'
            in page.read_text()
        )


class TestRunIngestFrontierDisposition:
    def test_frontier_content_replacement_is_re_reviewed_before_apply(
        self, isolated_wiki: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from chronovisor.core import jobs
        from chronovisor.ingest import ingest

        plan = [
            {
                "type": "create",
                "filename": "memory/observed-tools.md",
                "title": "Observed tools",
                "summary": "visible and responsive tools",
            }
        ]
        triage_calls: list[str] = []

        def fake_triage(content: str, **_kwargs):
            triage_calls.append(content)
            return plan

        monkeypatch.setattr(ingest, "_triage", fake_triage)
        monkeypatch.setattr(
            ingest,
            "_generate_one",
            lambda op, *_args, **_kwargs: {
                "type": "create",
                "filename": op["filename"],
                "content": (
                    "---\ntitle: Observed tools\nupdated: 2026-07-11\n---\n"
                    "Tool A was visible but failed and should not be used.\n"
                ),
            },
        )
        monkeypatch.setattr(ingest, "is_available", lambda: True)
        reviews: list[dict] = []
        replacement = (
            "---\ntitle: Observed tools\nupdated: 2026-07-11\n---\n"
            "Tool A was visible. Tool B was confirmed responsive.\n"
        )

        def reviewer(proposal: dict) -> dict:
            reviews.append(proposal)
            proposed = proposal["prepared_operations"][0]["proposed_text"]
            if "should not be used" in proposed:
                return {
                    "decision": "retry",
                    "summary": "Replace the unsupported recommendation.",
                    "failed_operations_disposition": "none",
                    "replacement_operations": [
                        {
                            "filename": "memory/observed-tools.md",
                            "content": replacement,
                        }
                    ],
                }
            return {
                "decision": "apply_available",
                "summary": "The replacement is narrowly grounded.",
                "failed_operations_disposition": "none",
                "replacement_operations": [],
            }

        job = jobs.job_store.create(processor="ollama")
        ingest.run_ingest(
            "Tool A visible; Tool B responsive.",
            job.job_id,
            frontier_reviewer=reviewer,
        )

        assert jobs.job_store.get(job.job_id).status == jobs.JobStatus.COMPLETED
        assert len(reviews) == 2
        assert len(triage_calls) == 1
        page = isolated_wiki / "pages" / "memory" / "observed-tools.md"
        written = page.read_text(encoding="utf-8")
        assert "Tool B was confirmed responsive." in written
        assert "should not be used" not in written

    def test_frontier_invalid_tag_uses_minimal_repair_before_regeneration(
        self, isolated_wiki: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from chronovisor.core import jobs
        from chronovisor.ingest import ingest

        plan = [
            {
                "type": "create",
                "filename": "memory/workflow-note.md",
                "title": "Workflow note",
                "summary": "grounded workflow",
            }
        ]
        triage_calls: list[str] = []

        def fake_triage(content: str, **_kwargs):
            triage_calls.append(content)
            return plan

        monkeypatch.setattr(ingest, "_triage", fake_triage)
        monkeypatch.setattr(
            ingest,
            "_generate_one",
            lambda op, *_args, **_kwargs: {
                "type": "create",
                "filename": op["filename"],
                "content": (
                    "---\ntitle: Workflow note\nupdated: 2026-07-11\n"
                    "tags: [d/ai-tools, d/finance, t/howto, s/2026]\n---\n"
                    "Grounded workflow fact.\n"
                ),
            },
        )
        monkeypatch.setattr(ingest, "is_available", lambda: True)
        reviews: list[dict] = []

        def reviewer(proposal: dict) -> dict:
            reviews.append(proposal)
            proposed = proposal["prepared_operations"][0]["proposed_text"]
            if "d/finance" in proposed:
                return {
                    "decision": "retry",
                    "summary": "The semantic option requires one exact postimage.",
                    "failed_operations_disposition": "none",
                    "invalid_tags": ["d/finance"],
                    "replacement_operations": [
                        {
                            "filename": "memory/workflow-note.md",
                            "content": proposed.replace(
                                "d/ai-tools, d/finance", "d/ai-tools"
                            ),
                        }
                    ],
                }
            return {
                "decision": "apply_available",
                "summary": "The metadata-only repair is grounded.",
                "failed_operations_disposition": "none",
                "invalid_tags": [],
            }

        job = jobs.job_store.create(processor="ollama")
        ingest.run_ingest(
            "raw workflow fact",
            job.job_id,
            frontier_reviewer=reviewer,
        )

        assert jobs.job_store.get(job.job_id).status == jobs.JobStatus.COMPLETED
        assert len(reviews) == 2
        assert len(triage_calls) == 1
        page = isolated_wiki / "pages" / "memory" / "workflow-note.md"
        written = page.read_text(encoding="utf-8")
        assert "d/finance" not in written
        assert "d/ai-tools" in written

    def test_tag_repair_requires_structured_filename_scoped_full_postimage(
        self,
    ) -> None:
        from chronovisor.ingest import ingest

        first = (
            "---\ntitle: First\ntags: [d/alpha, d/shared, t/reference, "
            "s/evergreen]\n---\nFirst fact.\n"
        )
        second = (
            "---\ntitle: Second\ntags: [d/beta, d/shared, t/reference, "
            "s/evergreen]\n---\nSecond fact.\n"
        )
        operations = [
            {
                "type": "create",
                "filename": "memory/first.md",
                "content": first,
            },
            {
                "type": "create",
                "filename": "memory/second.md",
                "content": second,
            },
        ]
        first_without_shared = first.replace("d/alpha, d/shared", "d/alpha")
        second_without_shared = second.replace("d/beta, d/shared", "d/beta")

        # Untrusted prose is excluded from the decision signature and cannot
        # manufacture a hidden tag deletion even when it uses old backticks.
        malicious_prose = {
            "review": {
                "decision": "retry",
                "summary": "Incorrect tag `d/shared`; remove it everywhere.",
                "risk": "Ungrounded `d/beta` too.",
                "notes": "Delete `d/shared`.",
                "replacement_operations": [
                    {
                        "filename": "memory/first.md",
                        "content": first_without_shared,
                    }
                ],
            }
        }
        repaired, replaced = ingest._apply_frontier_replacement_operations(
            operations, malicious_prose
        )
        assert repaired == operations
        assert replaced == []

        scoped = {
            "review": {
                "decision": "retry",
                "summary": "Prose still says `d/beta`, but it has no authority.",
                "invalid_tags": ["d/shared"],
                "replacement_operations": [
                    {
                        "filename": "memory/first.md",
                        "content": first_without_shared,
                    }
                ],
            }
        }
        repaired, replaced = ingest._apply_frontier_replacement_operations(
            operations, scoped
        )
        assert replaced == ["memory/first.md"]
        assert "d/shared" not in repaired[0]["content"]
        assert "d/shared" in repaired[1]["content"]
        assert "d/beta" in repaired[1]["content"]

        global_deletion = {
            "review": {
                "decision": "retry",
                "invalid_tags": ["d/shared"],
                "replacement_operations": [
                    {
                        "filename": "memory/first.md",
                        "content": first_without_shared,
                    },
                    {
                        "filename": "memory/second.md",
                        "content": second_without_shared,
                    },
                ],
            }
        }
        repaired, replaced = ingest._apply_frontier_replacement_operations(
            operations, global_deletion
        )
        assert repaired == operations
        assert replaced == []

    def test_repair_postimages_preserve_authorized_bytes_exactly(self) -> None:
        from chronovisor.ingest import ingest

        create_replacement = "---\ntitle: Exact\n---\nExact body.\n\n"
        update_replacement = "Exact addendum.  \n\n"
        operations = [
            {
                "type": "create",
                "filename": "memory/exact.md",
                "content": "---\ntitle: Exact\n---\nOld body.\n",
            },
            {
                "type": "update",
                "filename": "memory/update.md",
                "content": "Old addendum.\n",
            },
        ]
        result = {
            "review": {
                "decision": "retry",
                "invalid_tags": [],
                "replacement_operations": [
                    {
                        "filename": "memory/exact.md",
                        "content": create_replacement,
                    },
                    {
                        "filename": "memory/update.md",
                        "content": update_replacement,
                    },
                ],
            }
        }

        repaired, replaced = ingest._apply_frontier_replacement_operations(
            operations,
            result,
        )

        assert replaced == ["memory/exact.md", "memory/update.md"]
        assert repaired[0]["content"] == create_replacement
        assert repaired[1]["content"] == update_replacement

        result["review"]["replacement_operations"][1]["content"] = (
            "---\ntitle: Not body only\n---\nExact addendum.\n"
        )
        assert ingest._apply_frontier_replacement_operations(operations, result) == (
            operations,
            [],
        )

    def test_frontier_content_replacement_preserves_unrejected_taxonomy(
        self, isolated_wiki: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from chronovisor.core import jobs
        from chronovisor.ingest import ingest

        plan = [
            {
                "type": "create",
                "filename": "ai/socialization-scenario.md",
                "title": "Socialization scenario",
            }
        ]
        monkeypatch.setattr(ingest, "_triage", lambda *_args, **_kwargs: plan)
        monkeypatch.setattr(
            ingest,
            "_generate_one",
            lambda op, *_args, **_kwargs: {
                "type": "create",
                "filename": op["filename"],
                "content": (
                    "---\ntitle: Socialization scenario\nupdated: 2026-07-11\n"
                    "tags: [d/scenario, t/ai-socialization]\n---\n"
                    "Grounded fact plus unsupported claim.\n"
                ),
            },
        )
        monkeypatch.setattr(ingest, "is_available", lambda: True)
        reviews: list[dict] = []

        def reviewer(proposal: dict) -> dict:
            reviews.append(proposal)
            proposed = proposal["prepared_operations"][0]["proposed_text"]
            if "unsupported claim" in proposed:
                return {
                    "decision": "retry",
                    "summary": "Remove only the unsupported claim.",
                    "failed_operations_disposition": "none",
                    "replacement_operations": [
                        {
                            "filename": "ai/socialization-scenario.md",
                            "content": (
                                "---\ntitle: Socialization scenario\nupdated: 2026-07-11\n"
                                "tags: [d/scenario, t/ai-socialization]\n---\n"
                                "Grounded fact.\n"
                            ),
                        }
                    ],
                }
            assert "d/scenario" in proposed
            assert "d/event" not in proposed
            return {
                "decision": "apply_available",
                "summary": "The bounded repair is grounded.",
                "failed_operations_disposition": "none",
            }

        job = jobs.job_store.create(processor="ollama")
        ingest.run_ingest(
            "Grounded socialization scenario.",
            job.job_id,
            frontier_reviewer=reviewer,
        )

        assert jobs.job_store.get(job.job_id).status == jobs.JobStatus.COMPLETED
        assert len(reviews) == 2
        written = (
            isolated_wiki / "pages" / "ai" / "socialization-scenario.md"
        ).read_text(encoding="utf-8")
        assert "d/scenario" in written
        assert "d/event" not in written

    def test_frontier_rejection_regenerates_with_feedback_in_same_job(
        self, isolated_wiki: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from chronovisor.core import jobs
        from chronovisor.ingest import ingest

        plan = [
            {
                "type": "create",
                "filename": "memory/converged.md",
                "title": "Converged",
                "summary": "one explicit fact",
            }
        ]
        triage_feedback: list[str | None] = []
        generate_feedback: list[str | None] = []

        def fake_triage(_content: str, *, frontier_feedback=None):
            triage_feedback.append(frontier_feedback)
            if len(triage_feedback) > 1:
                raise AssertionError("content feedback must not re-run triage")
            return plan

        def fake_generate(
            op: dict,
            _raw: str,
            *,
            raw_keywords=None,
            progress_callback=None,
            frontier_feedback=None,
        ):
            generate_feedback.append(frontier_feedback)
            fact = "grounded fact" if frontier_feedback else "unsupported inference"
            return {
                "type": op["type"],
                "filename": op["filename"],
                "content": ("---\ntitle: Converged\nupdated: 2026-07-11\n---\n" + fact),
            }

        reviews: list[dict] = []

        def reviewer(proposal: dict) -> dict:
            reviews.append(proposal)
            if len(reviews) == 1:
                return {
                    "decision": "retry",
                    "summary": "Remove the unsupported inference; keep only the grounded fact.",
                    "failed_operations_disposition": "none",
                }
            return {
                "decision": "apply_available",
                "summary": "The regenerated proposal is grounded.",
                "failed_operations_disposition": "none",
            }

        monkeypatch.setattr(ingest, "_triage", fake_triage)
        monkeypatch.setattr(ingest, "_generate_one", fake_generate)
        monkeypatch.setattr(ingest, "is_available", lambda: True)
        status_updates: list[dict] = []
        monkeypatch.setattr(
            ingest.runtime_status,
            "safe_write_status",
            lambda **fields: status_updates.append(fields),
        )
        completed: list[bool] = []
        job = jobs.job_store.create(processor="ollama")

        ingest.run_ingest(
            "raw grounded fact",
            job.job_id,
            on_complete=lambda: completed.append(True),
            frontier_reviewer=reviewer,
        )

        finished = jobs.job_store.get(job.job_id)
        assert finished.status == jobs.JobStatus.COMPLETED
        assert completed == [True]
        assert len(reviews) == 2
        assert triage_feedback == [None]
        assert generate_feedback == [
            None,
            "Remove the unsupported inference; keep only the grounded fact.",
        ]
        assert any(
            update.get("stage") == "local-regenerate" for update in status_updates
        )
        assert any(
            isinstance(update.get("llm"), dict)
            and update["llm"].get("phase") == "local-regenerate-generate"
            for update in status_updates
        )
        assert not any(
            update.get("stage") == "frontier-regenerate"
            or (
                isinstance(update.get("llm"), dict)
                and update["llm"].get("phase") == "frontier-regenerate-triage"
            )
            for update in status_updates
        )
        page = isolated_wiki / "pages" / "memory" / "converged.md"
        assert "grounded fact" in page.read_text(encoding="utf-8")
        assert "unsupported inference" not in page.read_text(encoding="utf-8")

    def test_invalid_local_authority_fails_before_triage_or_generation(
        self, isolated_wiki: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from chronovisor.core import jobs
        from chronovisor.ingest import ingest

        calls: list[str] = []
        monkeypatch.setattr(ingest, "is_available", lambda: True)
        monkeypatch.setattr(
            ingest,
            "_current_ingest_review_authority",
            lambda **_kwargs: (
                None,
                "adoption_artifact_invalid:evaluation evidence is inconsistent",
            ),
        )
        monkeypatch.setattr(
            ingest,
            "_triage",
            lambda *_args, **_kwargs: calls.append("triage") or [],
        )
        monkeypatch.setattr(
            ingest,
            "_generate_one",
            lambda *_args, **_kwargs: calls.append("generate") or None,
        )

        def foreign_review_adapter(*_args, **_kwargs):
            calls.append("review")
            return {}

        foreign_review_adapter.__module__ = "tests.foreign_review_adapter"
        monkeypatch.setattr(
            ingest,
            "_review_and_apply_ingest_operations",
            foreign_review_adapter,
        )
        job = jobs.job_store.create(processor="ollama")

        ingest.run_ingest("grounded raw", job.job_id)

        finished = jobs.job_store.get(job.job_id)
        assert finished.status == jobs.JobStatus.FAILED
        assert calls == []
        assert finished.error == (
            "local consensus authority unavailable: "
            "adoption_artifact_invalid:evaluation evidence is inconsistent"
        )

    def test_local_noop_requires_frontier_confirmation(
        self, isolated_wiki: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from chronovisor.core import jobs
        from chronovisor.ingest import ingest

        monkeypatch.setattr(ingest, "_triage", lambda _content: [])
        monkeypatch.setattr(ingest, "is_available", lambda: True)
        captured: list[dict] = []
        completed: list[bool] = []
        job = jobs.job_store.create(processor="ollama")
        ingest.run_ingest(
            "raw must not disappear",
            job.job_id,
            on_complete=lambda: completed.append(True),
            frontier_reviewer=lambda proposal: (
                captured.append(proposal)
                or {
                    "decision": "retry",
                    "summary": "local no-op is not yet proven",
                    "failed_operations_disposition": "none",
                }
            ),
        )

        finished = jobs.job_store.get(job.job_id)
        assert finished.status == jobs.JobStatus.FAILED
        assert completed == []
        assert len(captured) == 2
        assert "did not converge after 2 local review calls" in str(finished.error)
        assert captured[0]["raw_content"] == "raw must not disappear"
        assert captured[0]["triage_plan"] == []
        assert captured[0]["local_disposition"] == "triage_no_operations"
        proposal_path, _review_path = ingest._ingest_artifact_paths(
            captured[0]["source_key"]
        )
        assert proposal_path.exists()

    def test_all_generation_failures_require_frontier_disposition(
        self, isolated_wiki: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from chronovisor.core import jobs
        from chronovisor.ingest import ingest

        plan = [
            {
                "type": "create",
                "filename": "memory/missing.md",
                "title": "Missing",
                "summary": "must be retained",
            }
        ]
        monkeypatch.setattr(ingest, "_triage", lambda _content: plan)
        monkeypatch.setattr(ingest, "_generate_one", lambda *_args, **_kwargs: None)
        monkeypatch.setattr(ingest, "is_available", lambda: True)
        captured: list[dict] = []
        completed: list[bool] = []
        job = jobs.job_store.create(processor="ollama")
        ingest.run_ingest(
            "all-failed raw",
            job.job_id,
            on_complete=lambda: completed.append(True),
            frontier_reviewer=lambda proposal: (
                captured.append(proposal)
                or {
                    "decision": "retry",
                    "summary": "regenerate the missing operation",
                    "failed_operations_disposition": "retry_required",
                }
            ),
        )

        finished = jobs.job_store.get(job.job_id)
        assert finished.status == jobs.JobStatus.FAILED
        assert completed == []
        assert captured[0]["local_disposition"] == "all_generation_failed"
        assert captured[0]["triage_plan"] == plan
        [failure] = captured[0]["failed_operation_specs"]
        assert failure["filename"] == "memory/missing.md"
        assert failure["attempts"] == 1
        assert failure["error"] == "generation validation failed"
        assert captured[0]["prepared_operations"] == []

    def test_partial_apply_without_explicit_failed_disposition_retries(
        self, isolated_wiki: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from chronovisor.core import jobs
        from chronovisor.ingest import ingest

        plan = [
            {"type": "create", "filename": "memory/ready.md", "title": "Ready"},
            {"type": "create", "filename": "memory/failed.md", "title": "Failed"},
        ]
        monkeypatch.setattr(ingest, "_triage", lambda _content: plan)

        def generate(op, _raw, **_kwargs):
            if op["filename"].endswith("failed.md"):
                return None
            return {
                "type": "create",
                "filename": op["filename"],
                "content": "---\ntitle: Ready\nupdated: 2026-07-11\n---\nready\n",
            }

        monkeypatch.setattr(ingest, "_generate_one", generate)
        monkeypatch.setattr(ingest, "is_available", lambda: True)
        completed: list[bool] = []
        job = jobs.job_store.create(processor="ollama")
        ingest.run_ingest(
            "partial raw",
            job.job_id,
            on_complete=lambda: completed.append(True),
            frontier_reviewer=lambda _proposal: {
                "decision": "apply_available",
                "summary": "ready page looks valid but failed op was not dispositioned",
                # Deliberately omit failed_operations_disposition.
            },
        )

        finished = jobs.job_store.get(job.job_id)
        assert finished.status == jobs.JobStatus.FAILED
        assert completed == []
        assert not (isolated_wiki / "pages" / "memory" / "ready.md").exists()
        assert not (isolated_wiki / "pages" / "memory" / "failed.md").exists()

    def test_complete_proposal_ignores_redundant_failed_disposition(
        self, isolated_wiki: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from chronovisor.core import jobs
        from chronovisor.ingest import ingest

        plan = [
            {"type": "create", "filename": "memory/ready.md", "title": "Ready"},
        ]
        monkeypatch.setattr(ingest, "_triage", lambda _content: plan)
        monkeypatch.setattr(
            ingest,
            "_generate_one",
            lambda op, _raw, **_kwargs: {
                "type": "create",
                "filename": op["filename"],
                "content": "---\ntitle: Ready\nupdated: 2026-07-11\n---\nready\n",
            },
        )
        monkeypatch.setattr(ingest, "is_available", lambda: True)

        job = jobs.job_store.create(processor="ollama")
        ingest.run_ingest(
            "complete raw",
            job.job_id,
            frontier_reviewer=lambda _proposal: {
                "decision": "apply_available",
                "summary": "prepared operation is grounded",
                "failed_operations_disposition": "confirmed_unnecessary",
            },
        )

        finished = jobs.job_store.get(job.job_id)
        assert finished.status == jobs.JobStatus.COMPLETED
        assert finished.pages_created == ["ready"]
        assert (isolated_wiki / "pages" / "memory" / "ready.md").exists()

    def test_retryable_partial_proposal_does_not_pin_later_complete_generation(
        self, isolated_wiki: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from chronovisor.core import jobs
        from chronovisor.ingest import ingest

        plan = [
            {"type": "create", "filename": "memory/one.md", "title": "One"},
            {"type": "create", "filename": "memory/two.md", "title": "Two"},
        ]
        monkeypatch.setattr(ingest, "_triage", lambda _content: plan)
        monkeypatch.setattr(ingest, "is_available", lambda: True)
        phase = {"complete": False}

        def generate(op, _raw, **_kwargs):
            if not phase["complete"] and op["filename"].endswith("two.md"):
                return None
            title = op["title"]
            return {
                "type": "create",
                "filename": op["filename"],
                "content": (
                    f"---\ntitle: {title}\nupdated: 2026-07-11\n---\n{title}\n"
                ),
            }

        monkeypatch.setattr(ingest, "_generate_one", generate)
        first = jobs.job_store.create(processor="ollama")
        ingest.run_ingest(
            "same replayable raw",
            first.job_id,
            frontier_reviewer=lambda _proposal: {
                "decision": "retry",
                "summary": "regenerate the missing page",
                "failed_operations_disposition": "retry_required",
            },
        )
        assert jobs.job_store.get(first.job_id).status == jobs.JobStatus.FAILED

        phase["complete"] = True
        second = jobs.job_store.create(processor="ollama")
        ingest.run_ingest(
            "same replayable raw",
            second.job_id,
            frontier_reviewer=lambda _proposal: {
                "decision": "apply_available",
                "summary": "complete regenerated proposal is grounded",
                "failed_operations_disposition": "none",
            },
        )

        finished = jobs.job_store.get(second.job_id)
        assert finished.status == jobs.JobStatus.COMPLETED
        assert sorted(finished.pages_created) == ["one", "two"]
        assert (isolated_wiki / "pages" / "memory" / "one.md").exists()
        assert (isolated_wiki / "pages" / "memory" / "two.md").exists()


# ---------------------------------------------------------------------------
# Phase 3b: raw_keywords metadata propagation through run_ingest
# ---------------------------------------------------------------------------


class TestRawKeywordsMetadataPropagation:
    """run_ingest must lift raw_keywords from metadata and ride it on
    every operation generated from this raw — without leaking into the
    triage prompt or being fabricated when absent.
    """

    @pytest.fixture(autouse=True)
    def isolate_context_admission(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from chronovisor.ingest import ingest

        monkeypatch.setattr(
            ingest,
            "_admit_ingest_context",
            lambda _config, selected: selected,
        )

    def test_metadata_raw_keywords_lands_on_every_operation(
        self, isolated_wiki: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from chronovisor.core import jobs
        from chronovisor.ingest import ingest

        plan = [
            {"type": "create", "filename": f"misc/p{i}.md", "title": f"P{i}"}
            for i in range(3)
        ]
        monkeypatch.setattr(ingest, "_triage", lambda _content: plan)

        captured_raw_keywords: list[list[str] | None] = []

        def stub_generate(op, _raw, *, raw_keywords=None):
            captured_raw_keywords.append(raw_keywords)
            return {
                "type": "create",
                "filename": op["filename"],
                "content": "---\ntitle: X\nupdated: 2026-04-28\n---\nbody",
            }

        monkeypatch.setattr(ingest, "_generate_one", stub_generate)
        monkeypatch.setattr(ingest, "is_available", lambda: True)

        job = jobs.job_store.create(processor="ollama")
        ingest.run_ingest(
            "raw content",
            job.job_id,
            metadata={"raw_keywords": ["alpha", "beta"]},
            frontier_reviewer=ingest._run_ingest_frontier_review,
        )

        # Every _generate_one call saw the same raw_keywords payload.
        assert captured_raw_keywords == [["alpha", "beta"]] * 3

    def test_no_metadata_means_no_raw_keywords_field(
        self, isolated_wiki: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Distinguishes "no propagation requested" from an empty list:
        when metadata is None the operation dict has NO ``raw_keywords``
        key at all (so the apply layer can skip patching).
        """
        from chronovisor.ingest import ingest

        op = {"type": "create", "filename": "misc/p0.md", "title": "P"}
        monkeypatch.setattr(
            ingest,
            "generate",
            lambda *_a, **_kw: (
                "=== NEW PAGE: misc/p0.md ===\n"
                "---\ntitle: P\nupdated: 2026-04-28\n---\nbody\n"
                "=== END PAGE ==="
            ),
        )

        result = ingest._generate_one(op, "raw content", raw_keywords=None)
        assert result is not None
        assert "raw_keywords" not in result

    def test_explicit_empty_list_survives(
        self, isolated_wiki: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """An empty list is intent — the apply layer should see ``[]`` and
        decide for itself, not have it elided into "no propagation".
        """
        from chronovisor.ingest import ingest

        op = {"type": "create", "filename": "misc/p0.md", "title": "P"}
        monkeypatch.setattr(
            ingest,
            "generate",
            lambda *_a, **_kw: (
                "=== NEW PAGE: misc/p0.md ===\n"
                "---\ntitle: P\nupdated: 2026-04-28\n---\nbody\n"
                "=== END PAGE ==="
            ),
        )

        result = ingest._generate_one(op, "raw content", raw_keywords=[])
        assert result is not None
        assert result.get("raw_keywords") == []

    def test_invalid_metadata_normalized_to_no_propagation(
        self, isolated_wiki: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Non-list / list with non-str items in metadata is treated as
        "no propagation". Important defensive behavior so a malformed raw
        frontmatter can't produce mojibake page metadata.
        """
        from chronovisor.core import jobs
        from chronovisor.ingest import ingest

        plan = [{"type": "create", "filename": "misc/p0.md", "title": "P"}]
        monkeypatch.setattr(ingest, "_triage", lambda _content: plan)

        captured: list[list[str] | None] = []

        def stub_generate(op, _raw, *, raw_keywords=None):
            captured.append(raw_keywords)
            return {
                "type": "create",
                "filename": op["filename"],
                "content": "---\ntitle: X\nupdated: 2026-04-28\n---\nbody",
            }

        monkeypatch.setattr(ingest, "_generate_one", stub_generate)
        monkeypatch.setattr(ingest, "is_available", lambda: True)

        for index, bad in enumerate(("not-a-list", 42, None, ["ok", 123], {"k": "v"})):
            captured.clear()
            job = jobs.job_store.create(processor="ollama")
            ingest.run_ingest(
                f"raw content {index}",
                job.job_id,
                metadata={"raw_keywords": bad},
                frontier_reviewer=ingest._run_ingest_frontier_review,
            )
            assert captured == [None], f"bad={bad!r}"


# ---------------------------------------------------------------------------
# Orchestrator (R2-High)
# ---------------------------------------------------------------------------


class TestOrchestrator:
    @staticmethod
    def _write_transcript_raw(
        raw_dir: Path,
        *,
        name: str,
        records: list[dict],
    ) -> Path:
        from chronovisor.core.save_transaction import (
            attach_save_transaction_marker,
            make_save_transaction,
        )

        body = "\n".join(
            [
                "---",
                "raw_keywords: [Codex, transcript-delta]",
                "---",
                "# Codex Session Transcript Delta",
                "",
                "## Transcript Delta",
                "",
                "```json",
                json.dumps(records, ensure_ascii=False, indent=2),
                "```",
                "",
            ]
        )
        transaction = make_save_transaction(
            host="codex",
            session_file=Path("/tmp/session.jsonl"),
            session_id=name,
            after_line=0,
            until_line=1,
        )
        path = raw_dir / name
        path.write_text(
            attach_save_transaction_marker(transaction, body),
            encoding="utf-8",
        )
        return path

    @staticmethod
    def _write_capture_fragments(
        raw_dir: Path,
        *,
        record_text: str,
        fragment_bytes: int,
        omit_indices: set[int] | None = None,
        prefix: str = "fragment",
    ) -> tuple[str, list[Path]]:
        from chronovisor.core.save_transaction import (
            attach_save_transaction_marker,
            make_save_transaction,
        )

        record = record_text.encode("utf-8")
        record_sha256 = hashlib.sha256(record).hexdigest()
        chunks = [
            record[offset : offset + fragment_bytes]
            for offset in range(0, len(record), fragment_bytes)
        ]
        paths: list[Path] = []
        for index, chunk in enumerate(chunks, start=1):
            if index in (omit_indices or set()):
                continue
            payload = {
                "schema": "chronovisor.raw-capture-fragment.v1",
                "host": "codex",
                "session_id": "fragment-session",
                "session_file": "/tmp/session.jsonl",
                "source_line": 42,
                "record_sha256": record_sha256,
                "record_bytes": len(record),
                "fragment_index": index,
                "fragment_count": len(chunks),
                "fragment_bytes": len(chunk),
                "encoding": "base64",
                "data": base64.b64encode(chunk).decode("ascii"),
            }
            path = raw_dir / f"{prefix}-{index}.md"
            body = (
                "---\nraw_keywords: [Codex, transcript-fragment]\n---\n"
                "# Codex Oversized Transcript Record Fragment\n\n"
                "```json\n"
                + json.dumps(payload, ensure_ascii=False, indent=2)
                + "\n```\n"
            )
            transaction = make_save_transaction(
                host="codex",
                session_file=Path("/tmp/session.jsonl"),
                session_id=f"{prefix}-session-{index}",
                after_line=41,
                until_line=42,
            )
            path.write_text(
                attach_save_transaction_marker(transaction, body),
                encoding="utf-8",
            )
            paths.append(path)
        return record_sha256, paths

    def test_oversized_review_continues_without_failure_or_repeating_generation(
        self,
        isolated_wiki: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        from chronovisor.core import runtime_status
        from chronovisor.core.runtime_config import DecisionRouterConfig
        from chronovisor.ingest import (
            failure_supervisor,
            ingest,
            ingest_readback,
            orchestrator,
        )

        raw_path = isolated_wiki / "raw" / "eight-shard-continuation.md"
        raw_content = "Eight grounded memories require bounded local review."
        raw_path.write_text(raw_content, encoding="utf-8")
        operations = TestIngestFrontierGate._oversized_create_ops(body_bytes=45_000)
        triage_plan = [
            {
                "type": "create",
                "filename": operation["filename"],
                "title": f"Sharded {index}",
            }
            for index, operation in enumerate(operations)
        ]
        calls = {"triage": 0, "generation": 0, "review": 0, "apply": 0}

        def triage(*_args, **_kwargs):
            calls["triage"] += 1
            return triage_plan

        def generate(*_args, **_kwargs):
            calls["generation"] += 1
            return operations, []

        def review(proposal, **_kwargs):
            calls["review"] += 1
            return ingest._normalize_ingest_frontier_review(
                {
                    "decision": "apply_available",
                    "summary": "exact shard approved",
                    "failed_operations_disposition": "none",
                },
                proposal=proposal,
            )

        real_apply = ingest._apply_prepared_operations

        def counted_apply(*args, **kwargs):
            if not kwargs.get("recovery_only"):
                calls["apply"] += 1
            return real_apply(*args, **kwargs)

        monkeypatch.setattr(orchestrator, "is_available", lambda: True)
        monkeypatch.setattr(ingest, "is_available", lambda: True)
        monkeypatch.setattr(ingest, "_triage_with_progress", triage)
        monkeypatch.setattr(ingest, "_generate_local_operations", generate)
        monkeypatch.setattr(ingest, "_run_ingest_frontier_review", review)
        monkeypatch.setattr(ingest, "_apply_prepared_operations", counted_apply)
        monkeypatch.setattr(
            ingest,
            "_ingest_review_router_config",
            lambda: DecisionRouterConfig(
                num_ctx=114_688,
                max_input_chars=93_000,
                adoption_artifact="",
            ),
        )
        monkeypatch.setattr(
            ingest_readback,
            "_refresh_ingest_derived_artifacts",
            lambda *_args, **_kwargs: {"checked": 0, "passed": 0, "failed": []},
        )

        continuation_results: list[dict] = []
        final_result: dict | None = None
        for _attempt in range(8):
            result = orchestrator.run_pending_ingest(
                force=True,
                frontier_reviewer=review,
            )
            if result["files_processed"]:
                final_result = result
                break
            continuation_results.append(result)
            row = result["per_raw"][0]
            continuation = row["continuation"]
            assert result["files_continued"] == [raw_path.name], result
            assert result["files_failed"] == 0
            assert result["files_deferred"] == []
            assert row["continued"] is True
            assert row["deferred"] is False
            assert "supervision" not in row
            assert continuation["approved_shards"] == calls["review"]
            assert continuation["approved_shards"] % 2 == 0
            assert calls["triage"] == 1
            assert calls["generation"] == 1
            assert calls["apply"] == 0
            assert not any((isolated_wiki / "pages" / "memory").glob("sharded-*.md"))
            assert failure_supervisor._load_state()["failures"] == {}
            assert failure_supervisor.operational_deferred_raw_files([raw_path]) == {}
            assert not list(
                (isolated_wiki / "runtime" / "failures" / "packets").glob("*.json")
            )
            status = runtime_status.read_status()
            assert status["batch"]["continued"] == 1
            assert status["batch"]["deferred"] == 0
            assert status["batch"]["failed"] == 0
            batch_metric = next(
                metric
                for metric in reversed(runtime_status.read_metrics(limit=20))
                if metric.get("kind") == "batch"
            )
            assert batch_metric["files_continued"] == 1
            assert batch_metric["files_deferred"] == 0
            assert batch_metric["files_failed"] == 0

        assert final_result is not None
        assert len(continuation_results) >= 3
        total_shards = continuation_results[0]["per_raw"][0]["continuation"][
            "total_shards"
        ]
        assert total_shards >= 8
        assert len(continuation_results) == (total_shards - 1) // 2
        assert final_result["files_processed"] == [raw_path.name]
        assert final_result["files_continued"] == []
        assert final_result["files_deferred"] == []
        assert final_result["files_failed"] == 0
        assert calls == {
            "triage": 1,
            "generation": 1,
            "review": total_shards,
            "apply": 1,
        }
        assert len(list((isolated_wiki / "pages" / "memory").glob("sharded-*.md"))) == 8
        assert orchestrator.get_pending_raw_files() == []
        assert failure_supervisor._load_state()["failures"] == {}
        assert failure_supervisor.operational_deferred_raw_files([raw_path]) == {}
        assert not list(
            (isolated_wiki / "runtime" / "failures" / "packets").glob("*.json")
        )

    def test_shard_repair_starts_zero_approval_continuation_without_regeneration(
        self,
        isolated_wiki: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        from chronovisor.core.runtime_config import DecisionRouterConfig
        from chronovisor.ingest import (
            failure_supervisor,
            ingest,
            ingest_readback,
            orchestrator,
        )

        raw_path = isolated_wiki / "raw" / "shard-repair-continuation.md"
        raw_content = "One exact shard needs repair before bounded review resumes."
        raw_path.write_text(raw_content, encoding="utf-8")
        operations = TestIngestFrontierGate._oversized_create_ops(body_bytes=30_000)
        triage_plan = [
            {
                "type": "create",
                "filename": operation["filename"],
                "title": f"Sharded {index}",
            }
            for index, operation in enumerate(operations)
        ]
        calls = {"triage": 0, "generation": 0, "review": 0, "apply": 0}
        repair_marker = "REPAIRED-BY-EXACT-SHARD-REVIEW"

        def triage(*_args, **_kwargs):
            calls["triage"] += 1
            return triage_plan

        def generate(*_args, **_kwargs):
            calls["generation"] += 1
            return operations, []

        def review(proposal, **_kwargs):
            calls["review"] += 1
            contract = proposal["audit_decision"]["review_shard_contract"]
            generated = proposal["local_generated_operations"]
            if (
                contract["shard_index"] == 0
                and repair_marker not in generated[0]["content"]
            ):
                return ingest._normalize_ingest_frontier_review(
                    {
                        "decision": "retry",
                        "summary": "repair the exact first shard",
                        "failed_operations_disposition": "retry_required",
                        "replacement_operations": [
                            {
                                "filename": generated[0]["filename"],
                                "content": generated[0]["content"]
                                + f"\n{repair_marker}\n",
                            }
                        ],
                    },
                    proposal=proposal,
                )
            return ingest._normalize_ingest_frontier_review(
                {
                    "decision": "apply_available",
                    "summary": "exact shard approved",
                    "failed_operations_disposition": "none",
                },
                proposal=proposal,
            )

        real_apply = ingest._apply_prepared_operations

        def counted_apply(*args, **kwargs):
            if not kwargs.get("recovery_only"):
                calls["apply"] += 1
            return real_apply(*args, **kwargs)

        monkeypatch.setattr(orchestrator, "is_available", lambda: True)
        monkeypatch.setattr(ingest, "is_available", lambda: True)
        monkeypatch.setattr(ingest, "_triage_with_progress", triage)
        monkeypatch.setattr(ingest, "_generate_local_operations", generate)
        monkeypatch.setattr(ingest, "_run_ingest_frontier_review", review)
        monkeypatch.setattr(ingest, "_apply_prepared_operations", counted_apply)
        monkeypatch.setattr(
            ingest,
            "_ingest_review_router_config",
            lambda: DecisionRouterConfig(
                num_ctx=114_688,
                max_input_chars=93_000,
                adoption_artifact="",
            ),
        )
        monkeypatch.setattr(
            ingest_readback,
            "_refresh_ingest_derived_artifacts",
            lambda *_args, **_kwargs: {"checked": 0, "passed": 0, "failed": []},
        )

        approved_progress: list[int] = []
        final_result: dict | None = None
        for _attempt in range(8):
            result = orchestrator.run_pending_ingest(
                force=True,
                frontier_reviewer=review,
            )
            if result["files_processed"]:
                final_result = result
                break
            assert result["files_continued"] == [raw_path.name], result
            assert result["files_failed"] == 0
            assert result["files_deferred"] == []
            assert result["per_raw"][0]["continued"] is True
            assert "supervision" not in result["per_raw"][0]
            approved_progress.append(
                result["per_raw"][0]["continuation"]["approved_shards"]
            )
            assert calls["triage"] == 1
            assert calls["generation"] == 1
            assert calls["apply"] == 0
            assert not any((isolated_wiki / "pages" / "memory").glob("sharded-*.md"))
            assert failure_supervisor._load_state()["failures"] == {}
            assert failure_supervisor.operational_deferred_raw_files([raw_path]) == {}
            assert not list(
                (isolated_wiki / "runtime" / "failures" / "packets").glob("*.json")
            )

        assert final_result is not None
        assert 0 in approved_progress
        assert approved_progress.index(0) > 0
        assert final_result["files_processed"] == [raw_path.name]
        assert final_result["files_continued"] == []
        assert final_result["files_deferred"] == []
        assert final_result["files_failed"] == 0
        assert calls["triage"] == 1
        assert calls["generation"] == 1
        assert calls["apply"] == 1
        assert len(list((isolated_wiki / "pages" / "memory").glob("sharded-*.md"))) == 8
        repaired_page = (isolated_wiki / "pages" / "memory" / "sharded-0.md").read_text(
            encoding="utf-8"
        )
        assert repair_marker.strip() in repaired_page
        assert orchestrator.get_pending_raw_files() == []
        assert failure_supervisor._load_state()["failures"] == {}
        assert not list(
            (isolated_wiki / "runtime" / "failures" / "packets").glob("*.json")
        )

    def test_complete_fragment_group_is_projected_then_child_is_ingested(
        self, isolated_wiki: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from chronovisor.ingest import ingest as ingest_mod
        from chronovisor.ingest import orchestrator

        record = json.dumps(
            [{"line": 42, "role": "user", "text": "remember the whole record"}],
            ensure_ascii=False,
        )
        _sha256, paths = self._write_capture_fragments(
            isolated_wiki / "raw",
            record_text=record,
            fragment_bytes=13,
        )
        observed: list[tuple[str, dict]] = []

        def fake_run_ingest(
            content, job_id, on_complete=None, on_finally=None, *, metadata=None
        ):
            observed.append((content, metadata or {}))
            if on_complete:
                on_complete()
            if on_finally:
                on_finally(failed=False, triage_failed=False)

        monkeypatch.setattr(ingest_mod, "run_ingest", fake_run_ingest)
        monkeypatch.setattr(orchestrator, "is_available", lambda: True)

        delegated = orchestrator.run_pending_ingest(force=True)

        assert observed == []
        assert delegated["files_processed"] == sorted(path.name for path in paths)
        projection = delegated["per_raw"][0]["projection"]
        assert projection["kind"] == "children"
        assert projection["child_count"] == 1
        pending_children = orchestrator.get_pending_raw_files()
        assert pending_children == [Path(projection["child_paths"][0])]
        assert all(path.exists() for path in paths)

        processed = orchestrator.run_pending_ingest(force=True)

        assert processed["files_processed"] == [pending_children[0].name]
        assert len(observed) == 1
        assert "remember the whole record" in observed[0][0]
        assert '"data"' not in observed[0][0]
        assert observed[0][1]["raw_keywords"] == ["transcript-semantic-projection"]
        assert orchestrator.get_pending_raw_files() == []

    def test_fragment_projection_resumes_group_ack_without_reprojection(
        self, isolated_wiki: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from chronovisor.ingest import ingest as ingest_mod
        from chronovisor.ingest import (
            orchestrator,
            raw_completion_ack,
            raw_semantic_projection,
        )

        record = json.dumps(
            [{"line": 7, "role": "user", "text": "remember projected record"}],
            ensure_ascii=False,
        )
        _record_sha256, paths = self._write_capture_fragments(
            isolated_wiki / "raw",
            record_text=record,
            fragment_bytes=11,
            prefix="ack-fragment",
        )
        original_project = raw_semantic_projection.project_reassembled_raws
        calls = {"projection": 0}

        def counted_projection(*args, **kwargs):
            calls["projection"] += 1
            return original_project(*args, **kwargs)

        monkeypatch.setattr(
            raw_semantic_projection,
            "project_reassembled_raws",
            counted_projection,
        )
        monkeypatch.setattr(
            ingest_mod,
            "run_ingest",
            lambda *_args, **_kwargs: pytest.fail(
                "fragment ACK recovery reached model ingest"
            ),
        )
        monkeypatch.setattr(orchestrator, "is_available", lambda: False)
        real_save_state = orchestrator._save_state
        source_names = {path.name for path in paths}
        injected = {"done": False}

        def fail_group_processed_mark(state: dict) -> None:
            processed = set(state.get("processed_raw_files", []))
            if not injected["done"] and source_names <= processed:
                injected["done"] = True
                raise OSError("injected fragment ACK state failure")
            real_save_state(state)

        monkeypatch.setattr(orchestrator, "_save_state", fail_group_processed_mark)

        first = orchestrator.run_pending_ingest(force=True)

        assert first["per_raw"][0]["succeeded"] is False
        assert first["per_raw"][0]["supervision"]["failure_class"] == (
            "ingest.raw_completion_ack_state_pending"
        )
        assert calls["projection"] == 1
        assert raw_completion_ack.receipt_path(paths).is_file()
        [child_path] = [
            path
            for path in orchestrator.get_pending_raw_files()
            if path.name.startswith("semantic-") and path.suffix == ".md"
        ]
        # The child is a separate semantic unit; remove it from this ACK-only
        # recovery assertion so no unrelated model work shares the next batch.
        child_state = orchestrator._load_state()
        child_state["processed_raw_files"] = sorted(
            set(child_state.get("processed_raw_files", [])) | {child_path.name}
        )
        orchestrator._save_state(child_state)

        second = orchestrator.run_pending_ingest(force=True)

        assert set(second["files_processed"]) == source_names
        assert second["per_raw"][0]["completion_ack"]["resumed"] is True
        assert second["per_raw"][0]["completion_ack"]["source_files"] == sorted(
            source_names
        )
        assert calls["projection"] == 1

    def test_verified_transcript_parent_delegates_byte_exact_child(
        self, isolated_wiki: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from chronovisor.ingest import ingest as ingest_mod
        from chronovisor.ingest import orchestrator

        raw_path = self._write_transcript_raw(
            isolated_wiki / "raw",
            name="verified-transcript.md",
            records=[
                {"line": 1, "role": "user", "text": "記憶して🙂\\literal"},
                {"line": 2, "role": "tool", "text": "SECRET-TOOL-PROTOCOL"},
                {"line": 3, "role": "assistant", "text": "了解した"},
            ],
        )
        original = raw_path.read_bytes()
        observed: list[tuple[str, dict]] = []

        def fake_run_ingest(
            content, job_id, on_complete=None, on_finally=None, *, metadata=None
        ):
            observed.append((content, metadata or {}))
            if on_complete:
                on_complete()
            if on_finally:
                on_finally(failed=False, triage_failed=False)

        monkeypatch.setattr(ingest_mod, "run_ingest", fake_run_ingest)
        monkeypatch.setattr(orchestrator, "is_available", lambda: True)

        delegated = orchestrator.run_pending_ingest(force=True)

        assert observed == []
        assert delegated["files_processed"] == [raw_path.name]
        projection = delegated["per_raw"][0]["projection"]
        assert projection["kind"] == "children"
        assert Path(projection["manifest_path"]).exists()
        assert raw_path.read_bytes() == original

        processed = orchestrator.run_pending_ingest(force=True)

        assert processed["files_processed"] == [Path(projection["child_paths"][0]).name]
        assert len(observed) == 1
        child = json.loads(observed[0][0])
        projected_text = [row["text"] for row in child["records"]]
        assert projected_text == ["記憶して🙂\\literal", "了解した"]
        assert "SECRET-TOOL-PROTOCOL" not in observed[0][0]
        assert orchestrator.get_pending_raw_files() == []

    def test_tool_only_transcript_uses_durable_noop_without_model(
        self, isolated_wiki: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from chronovisor.ingest import ingest as ingest_mod
        from chronovisor.ingest import orchestrator

        raw_path = self._write_transcript_raw(
            isolated_wiki / "raw",
            name="tool-only-transcript.md",
            records=[
                {"line": 1, "role": "tool", "text": "tool output"},
                {"line": 2, "role": "event", "text": "reasoning event"},
                {"line": 3, "role": "assistant", "text": "   "},
            ],
        )
        original = raw_path.read_bytes()
        monkeypatch.setattr(
            ingest_mod,
            "run_ingest",
            lambda *_args, **_kwargs: pytest.fail("tool-only raw reached model"),
        )
        monkeypatch.setattr(orchestrator, "is_available", lambda: False)

        result = orchestrator.run_pending_ingest(force=True)

        assert result["files_processed"] == [raw_path.name]
        projection = result["per_raw"][0]["projection"]
        assert projection["kind"] == "noop"
        assert Path(projection["manifest_path"]).exists()
        assert Path(projection["noop_receipt_path"]).exists()
        assert projection["selected_record_count"] == 0
        assert raw_path.read_bytes() == original
        assert orchestrator.get_pending_raw_files() == []

    def test_over_limit_fragment_group_fans_out_without_quarantine(
        self, isolated_wiki: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from chronovisor.core import runtime_config
        from chronovisor.ingest import ingest as ingest_mod
        from chronovisor.ingest import orchestrator

        record = json.dumps([{"role": "user", "text": "X" * 6_000}])
        record_sha256, paths = self._write_capture_fragments(
            isolated_wiki / "raw",
            record_text=record,
            fragment_bytes=32,
        )
        monkeypatch.setattr(
            ingest_mod,
            "run_ingest",
            lambda *_args, **_kwargs: pytest.fail("fragment leaked into model ingest"),
        )
        monkeypatch.setattr(orchestrator, "is_available", lambda: True)
        monkeypatch.setattr(
            runtime_config,
            "load_ingest_config",
            lambda: runtime_config.IngestConfig(
                semantic_projection_max_child_bytes=2_048
            ),
        )

        result = orchestrator.run_pending_ingest(force=True)

        assert result["triggered"] is True
        assert result["files_quarantined"] == []
        assert result["files_processed"] == sorted(path.name for path in paths)
        projection = result["per_raw"][0]["projection"]
        assert projection["kind"] == "children"
        assert projection["child_count"] > 1
        assert len(orchestrator.get_pending_raw_files()) == projection["child_count"]
        assert all(
            path.name.startswith("semantic-")
            for path in orchestrator.get_pending_raw_files()
        )
        dead_letter = (
            isolated_wiki
            / "raw"
            / ".dead-letter"
            / "raw-capture-fragments"
            / record_sha256
        )
        assert not dead_letter.exists()

    def test_incomplete_fragment_group_is_deferred_without_model_call(
        self, isolated_wiki: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from chronovisor.ingest import ingest as ingest_mod
        from chronovisor.ingest import orchestrator

        record = json.dumps([{"role": "user", "text": "Y" * 100}])
        _sha256, paths = self._write_capture_fragments(
            isolated_wiki / "raw",
            record_text=record,
            fragment_bytes=32,
            omit_indices={2},
        )
        monkeypatch.setattr(
            ingest_mod,
            "run_ingest",
            lambda *_args, **_kwargs: pytest.fail("incomplete fragments reached model"),
        )
        monkeypatch.setattr(orchestrator, "is_available", lambda: True)

        result = orchestrator.run_pending_ingest(force=True)

        assert result["triggered"] is False
        assert result["fragment_deferred"][0]["missing_indices"] == [2]
        assert orchestrator.get_pending_raw_files() == sorted(paths)

    def test_failed_fragment_projection_is_deferred_without_moving_group(
        self, isolated_wiki: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from chronovisor.core import background_jobs
        from chronovisor.ingest import (
            failure_supervisor,
            orchestrator,
            raw_semantic_projection,
        )
        from chronovisor.ingest import ingest as ingest_mod

        record = json.dumps([{"role": "user", "text": "grounded record"}])
        record_sha256, paths = self._write_capture_fragments(
            isolated_wiki / "raw",
            record_text=record,
            fragment_bytes=9,
        )

        projection_calls = 0

        def fail_projection(*_args, **_kwargs):
            nonlocal projection_calls
            projection_calls += 1
            raise raw_semantic_projection.ProjectionConflictError(
                "injected projection conflict"
            )

        monkeypatch.setattr(
            ingest_mod,
            "run_ingest",
            lambda *_args, **_kwargs: pytest.fail(
                "failed projection reached model ingest"
            ),
        )
        monkeypatch.setattr(orchestrator, "is_available", lambda: True)
        monkeypatch.setattr(
            raw_semantic_projection,
            "project_reassembled_raws",
            fail_projection,
        )
        monkeypatch.setattr(background_jobs, "start_self_heal_background", lambda _path: None)

        result = orchestrator.run_pending_ingest(force=True)
        second = orchestrator.run_pending_ingest(force=True)

        assert result["per_raw"][0]["succeeded"] is False
        assert result["per_raw"][0]["supervision"]["failure_class"] == (
            "ingest.runtime_semantic_projection_artifact_conflict"
        )
        assert result["per_raw"][0]["supervision"]["quarantined"] is False
        assert projection_calls == 1
        assert second["triggered"] is False
        assert second["reason"] == "no pending raws"
        assert orchestrator.get_pending_raw_files() == []
        assert all(path.exists() for path in paths)
        deferred = failure_supervisor.operational_deferred_raw_files(paths)
        assert set(deferred) == {path.name for path in paths}
        dead_letter = (
            isolated_wiki
            / "raw"
            / ".dead-letter"
            / "raw-capture-fragments"
            / record_sha256
        )
        assert not dead_letter.exists()

    def test_fragment_source_invalid_quarantines_whole_logical_group_atomically(
        self, isolated_wiki: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from chronovisor.ingest import ingest as ingest_mod
        from chronovisor.ingest import orchestrator, raw_semantic_projection

        record_sha256, paths = self._write_capture_fragments(
            isolated_wiki / "raw",
            record_text=json.dumps([{"role": "user", "text": "invalid source"}]),
            fragment_bytes=8,
        )

        def fail_source(*_args, **_kwargs):
            raise raw_semantic_projection.RawSemanticProjectionError(
                "verified fragment envelope is invalid"
            )

        monkeypatch.setattr(
            raw_semantic_projection,
            "project_reassembled_raws",
            fail_source,
        )
        monkeypatch.setattr(
            ingest_mod,
            "run_ingest",
            lambda *_args, **_kwargs: pytest.fail(
                "source-invalid fragment group reached model ingest"
            ),
        )
        monkeypatch.setattr(orchestrator, "is_available", lambda: True)

        result = orchestrator.run_pending_ingest(force=True)

        supervision = result["per_raw"][0]["supervision"]
        assert supervision["failure_class"] == (
            "raw.semantic_projection_source_invalid"
        )
        assert supervision["quarantined"] is True
        assert all(not path.exists() for path in paths)
        dead_letter = (
            isolated_wiki
            / "raw"
            / ".dead-letter"
            / "raw-capture-fragments"
            / record_sha256
        )
        manifest = json.loads(
            (dead_letter / "manifest.json").read_text(encoding="utf-8")
        )
        assert manifest["status"] == "completed"
        assert {row["source"] for row in manifest["files"]} == {
            path.name for path in paths
        }
        state = orchestrator._load_state()
        assert set(path.name for path in paths) <= set(state["processed_raw_files"])
        assert orchestrator.get_pending_raw_files() == []

    def test_invalid_fragment_group_cannot_quarantine_unrelated_group(
        self, isolated_wiki: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from chronovisor.ingest import ingest as ingest_mod
        from chronovisor.ingest import orchestrator

        bad_sha256, bad_paths = self._write_capture_fragments(
            isolated_wiki / "raw",
            record_text=json.dumps([{"role": "user", "text": "bad group"}]),
            fragment_bytes=12,
            prefix="bad",
        )
        duplicate = isolated_wiki / "raw" / "bad-duplicate.md"
        duplicate.write_bytes(bad_paths[0].read_bytes())
        _good_sha256, good_paths = self._write_capture_fragments(
            isolated_wiki / "raw",
            record_text=json.dumps([{"role": "user", "text": "good group"}]),
            fragment_bytes=11,
            prefix="good",
        )
        monkeypatch.setattr(
            ingest_mod,
            "run_ingest",
            lambda *_args, **_kwargs: pytest.fail(
                "delegated fragment parent reached model"
            ),
        )
        monkeypatch.setattr(orchestrator, "is_available", lambda: True)

        result = orchestrator.run_pending_ingest(force=True)

        assert set(result["files_quarantined"]) == {
            *(path.name for path in bad_paths),
            duplicate.name,
        }
        assert result["files_processed"] == sorted(path.name for path in good_paths)
        assert all(path.exists() for path in good_paths)
        assert not any(path.exists() for path in bad_paths)
        assert not duplicate.exists()
        bad_dead_letter = (
            isolated_wiki
            / "raw"
            / ".dead-letter"
            / "raw-capture-fragments"
            / bad_sha256
        )
        assert bad_dead_letter.exists()
        assert all(
            path.name.startswith("semantic-")
            for path in orchestrator.get_pending_raw_files()
        )

    def test_fragment_quarantine_intent_publish_failure_moves_nothing(
        self, isolated_wiki: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from chronovisor.ingest import orchestrator

        record_sha256, paths = self._write_capture_fragments(
            isolated_wiki / "raw",
            record_text=json.dumps([{"role": "user", "text": "unsafe"}]),
            fragment_bytes=12,
        )

        def fail_publish(_path: Path, _content: str) -> None:
            raise OSError("injected manifest fsync failure")

        monkeypatch.setattr(orchestrator, "atomic_write", fail_publish)

        with pytest.raises(OSError, match="manifest fsync failure"):
            orchestrator._quarantine_capture_fragment_paths(
                paths,
                record_sha256=record_sha256,
                reason="fragment_integrity_failure",
            )

        assert all(path.exists() for path in paths)
        dead_letter = (
            isolated_wiki
            / "raw"
            / ".dead-letter"
            / "raw-capture-fragments"
            / record_sha256
        )
        assert not list(dead_letter.glob("*.md"))
        assert not list(dead_letter.glob("manifest*.json"))

    def test_fragment_quarantine_partial_move_resumes_on_next_run(
        self, isolated_wiki: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from chronovisor.ingest import ingest as ingest_mod
        from chronovisor.ingest import orchestrator

        record_sha256, paths = self._write_capture_fragments(
            isolated_wiki / "raw",
            record_text=json.dumps([{"role": "user", "text": "unsafe payload"}]),
            fragment_bytes=10,
        )
        original_rename = Path.rename
        moves = 0

        def flaky_rename(path: Path, target: Path) -> Path:
            nonlocal moves
            if path in paths:
                moves += 1
                if moves == 2:
                    raise OSError("injected move crash")
            return original_rename(path, target)

        monkeypatch.setattr(Path, "rename", flaky_rename)
        with pytest.raises(OSError, match="move crash"):
            orchestrator._quarantine_capture_fragment_paths(
                paths,
                record_sha256=record_sha256,
                reason="fragment_integrity_failure",
            )

        dead_letter = (
            isolated_wiki
            / "raw"
            / ".dead-letter"
            / "raw-capture-fragments"
            / record_sha256
        )
        manifest_path = dead_letter / "manifest.json"
        prepared = json.loads(manifest_path.read_text(encoding="utf-8"))
        assert prepared["status"] == "prepared"
        assert sum(path.exists() for path in paths) == len(paths) - 1

        monkeypatch.setattr(Path, "rename", original_rename)
        monkeypatch.setattr(
            ingest_mod,
            "run_ingest",
            lambda *_args, **_kwargs: pytest.fail("resumed fragments reached model"),
        )
        result = orchestrator.run_pending_ingest(force=True)

        assert result == {"triggered": False, "reason": "no pending raws"}
        completed = json.loads(manifest_path.read_text(encoding="utf-8"))
        assert completed["status"] == "completed"
        assert isinstance(completed["completed_at"], str)
        assert not any(path.exists() for path in paths)
        assert all(
            (dead_letter / row["preserved_as"]).exists() for row in completed["files"]
        )
        state = json.loads(orchestrator.STATE_FILE.read_text(encoding="utf-8"))
        assert set(path.name for path in paths) <= set(state["processed_raw_files"])

    def test_cross_process_lease_defers_without_touching_pending_raw(
        self, isolated_wiki: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from contextlib import contextmanager

        from chronovisor.ingest import ingest as ingest_mod
        from chronovisor.ingest import orchestrator

        raw_path = isolated_wiki / "raw" / "pending.md"
        raw_path.write_text("pending evidence", encoding="utf-8")

        @contextmanager
        def busy_lease():
            yield False, "injected busy lease"

        monkeypatch.setattr(orchestrator, "_cross_process_ingest_lease", busy_lease)
        monkeypatch.setattr(
            ingest_mod,
            "run_ingest",
            lambda *_args, **_kwargs: pytest.fail("busy lease reached ingest"),
        )

        result = orchestrator.run_pending_ingest(force=True)

        assert result == {"triggered": False, "reason": "injected busy lease"}
        assert orchestrator.get_pending_raw_files() == [raw_path]

    def test_retracted_raw_is_not_pending_and_body_is_preserved(
        self, isolated_wiki: Path
    ) -> None:
        from chronovisor.ingest import orchestrator

        raw_dir = isolated_wiki / "raw"
        retracted = raw_dir / "retracted.md"
        retracted_text = (
            "---\n"
            "raw_keywords: [kuycon, p24u]\n"
            "raw_status: RETRACTED\n"
            "retraction_reason: entity_fusion\n"
            "---\n"
            "Original raw body must remain byte-for-byte unchanged.\n"
        )
        retracted.write_text(retracted_text, encoding="utf-8")
        (raw_dir / "active.md").write_text(
            "---\nraw_status: active\n---\nactive body\n",
            encoding="utf-8",
        )
        (raw_dir / "legacy.md").write_text("legacy body\n", encoding="utf-8")
        (raw_dir / "unknown.md").write_text(
            "---\nraw_status: corrected\n---\nunknown status remains compatible\n",
            encoding="utf-8",
        )

        pending = {path.name for path in orchestrator.get_pending_raw_files()}

        assert pending == {"active.md", "legacy.md", "unknown.md"}
        assert retracted.read_text(encoding="utf-8") == retracted_text
        assert retracted.name not in orchestrator._load_state()["processed_raw_files"]

    def test_force_ingest_declines_when_only_raw_is_retracted(
        self, isolated_wiki: Path
    ) -> None:
        from chronovisor.ingest import orchestrator

        (isolated_wiki / "raw" / "retracted.md").write_text(
            "---\nraw_status: retracted\n---\nbody\n",
            encoding="utf-8",
        )

        result = orchestrator.run_pending_ingest(force=True)

        assert result == {"triggered": False, "reason": "no pending raws"}

    def test_reset_stale_lock_clears_pending_sentinel(
        self, isolated_wiki: Path
    ) -> None:
        from chronovisor.ingest import orchestrator

        # Simulate a server crash mid-`run_pending_ingest`: the sentinel was
        # written but the real job_id never replaced it.
        state = orchestrator._load_state()
        state["current_job_id"] = "__pending__"
        orchestrator._save_state(state)

        orchestrator.reset_stale_lock()
        assert orchestrator._load_state()["current_job_id"] is None

    def test_reset_stale_lock_clears_unknown_job(self, isolated_wiki: Path) -> None:
        from chronovisor.ingest import orchestrator

        state = orchestrator._load_state()
        state["current_job_id"] = "no-such-job-12345"
        orchestrator._save_state(state)

        orchestrator.reset_stale_lock()
        # job_store is in-memory → after restart, the id is unknown → cleared.
        assert orchestrator._load_state()["current_job_id"] is None

    def test_reset_stale_lock_keeps_known_job(self, isolated_wiki: Path) -> None:
        from chronovisor.core import jobs
        from chronovisor.ingest import orchestrator

        job = jobs.job_store.create(processor="ollama")
        try:
            state = orchestrator._load_state()
            state["current_job_id"] = job.job_id
            orchestrator._save_state(state)

            orchestrator.reset_stale_lock()
            assert orchestrator._load_state()["current_job_id"] == job.job_id
        finally:
            # Cleanup so other tests aren't polluted.
            jobs.job_store._jobs.pop(job.job_id, None)

    def test_reset_stale_lock_keeps_live_cross_process_lock(
        self, isolated_wiki: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from chronovisor.ingest import orchestrator

        state = orchestrator._load_state()
        state["current_job_id"] = "job-in-another-process"
        state["current_job_pid"] = os.getpid()
        state["current_job_started_at"] = datetime.now().isoformat()
        orchestrator._save_state(state)
        monkeypatch.setattr(
            orchestrator,
            "ingest_process_lease_is_held",
            lambda _pid: True,
        )

        orchestrator.reset_stale_lock()
        assert orchestrator._load_state()["current_job_id"] == "job-in-another-process"

    def test_reset_stale_lock_clears_same_live_pid_without_process_lease(
        self, isolated_wiki: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from chronovisor.ingest import orchestrator

        state = orchestrator._load_state()
        state["current_job_id"] = "stranded-in-long-lived-mcp"
        state["current_job_pid"] = os.getpid()
        state["current_job_started_at"] = datetime.now().isoformat()
        orchestrator._save_state(state)
        monkeypatch.setattr(
            orchestrator,
            "ingest_process_lease_is_held",
            lambda _pid: False,
        )

        orchestrator.reset_stale_lock()

        assert orchestrator._load_state()["current_job_id"] is None

    def test_uncertain_reservation_write_clears_slot_and_next_run_succeeds(
        self, isolated_wiki: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from chronovisor.ingest import ingest as ingest_mod
        from chronovisor.ingest import orchestrator

        raw_path = isolated_wiki / "raw" / "reservation-recovery.md"
        raw_path.write_text("stable source", encoding="utf-8")
        real_save_state = orchestrator._save_state
        injected = {"done": False}

        def fail_after_reservation_replace(state: dict) -> None:
            real_save_state(state)
            if not injected["done"] and state.get("current_job_id") == "__pending__":
                injected["done"] = True
                raise OSError("injected reservation directory fsync failure")

        monkeypatch.setattr(
            orchestrator,
            "_save_state",
            fail_after_reservation_replace,
        )
        monkeypatch.setattr(orchestrator, "is_available", lambda: True)

        with pytest.raises(OSError, match="reservation directory fsync failure"):
            orchestrator.run_pending_ingest(force=True)

        stranded = orchestrator._load_state()
        assert stranded["current_job_id"] is None
        assert stranded["current_job_pid"] is None
        assert orchestrator._INGEST_PROCESS_LEASE_ACTIVE is False

        monkeypatch.setattr(orchestrator, "_save_state", real_save_state)

        def succeed(
            _content, _job_id, on_complete=None, on_finally=None, *, metadata=None
        ):
            if on_complete:
                on_complete()

        monkeypatch.setattr(ingest_mod, "run_ingest", succeed)
        recovered = orchestrator.run_pending_ingest(force=True)

        assert recovered["files_processed"] == [raw_path.name]
        assert orchestrator.get_pending_raw_files() == []

    def test_reset_stale_lock_clears_dead_cross_process_lock(
        self, isolated_wiki: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from chronovisor.ingest import orchestrator

        state = orchestrator._load_state()
        state["current_job_id"] = "job-from-dead-process"
        state["current_job_pid"] = 12345
        state["current_job_started_at"] = datetime.now().isoformat()
        orchestrator._save_state(state)
        monkeypatch.setattr(orchestrator, "_pid_is_alive", lambda _pid: False)

        orchestrator.reset_stale_lock()
        loaded = orchestrator._load_state()
        assert loaded["current_job_id"] is None
        assert loaded["current_job_pid"] is None
        assert loaded["current_job_started_at"] is None

    def test_run_pending_ingest_does_not_turn_global_authority_outage_into_raw_failures(
        self, isolated_wiki: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from chronovisor.core import runtime_status
        from chronovisor.ingest import ingest as ingest_mod
        from chronovisor.ingest import orchestrator

        raw_path = isolated_wiki / "raw" / "authority-blocked.md"
        raw_path.write_text("valid immutable source", encoding="utf-8")
        monkeypatch.setattr(orchestrator, "is_available", lambda: True)
        monkeypatch.setattr(
            orchestrator,
            "ingest_authority_preflight",
            lambda **_kwargs: {
                "ok": False,
                "status": "blocked",
                "blocked_by": "decision_authority",
                "retryable": True,
                "error": (
                    "local consensus authority unavailable: "
                    "adoption_artifact_invalid:policy version mismatch"
                ),
                "artifact_sha256": None,
            },
        )
        monkeypatch.setattr(
            ingest_mod,
            "run_ingest",
            lambda *_args, **_kwargs: (_ for _ in ()).throw(
                AssertionError("Raw processing must not start")
            ),
        )

        result = orchestrator.run_pending_ingest(force=True)

        assert result["triggered"] is False
        assert result["blocked_by"] == "decision_authority"
        assert result["files_attempted"] == []
        assert result["files_processed"] == []
        assert raw_path.exists()
        assert [path.name for path in orchestrator.get_pending_raw_files()] == [
            raw_path.name
        ]
        assert not (
            isolated_wiki / "runtime" / "failures" / "state.json"
        ).exists()
        status = runtime_status.read_status()
        assert status["state"] == "blocked"
        assert status["stage"] == "decision-authority"

    def test_run_pending_ingest_serial_then_idempotent(
        self, isolated_wiki: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Per-raw synchronous serial design: the first call ingests every
        pending raw individually; the second call sees them all marked
        processed and declines with a no-pending / threshold-not-met
        reason. Concurrency safety against true parallel callers is
        enforced by ``_INGEST_LOCK`` (in-process) and tested separately.
        """
        from chronovisor.ingest import orchestrator

        # Make 5 fake raws so should_ingest() fires.
        for i in range(5):
            (isolated_wiki / "raw" / f"r{i}.md").write_text("body")

        # Stub run_ingest to simulate full-success: invoke on_complete so
        # the orchestrator marks each raw processed individually.
        captured = {"calls": 0, "metadata_keys": []}

        def fake_run_ingest(
            content,
            job_id,
            on_complete=None,
            on_finally=None,
            *,
            metadata=None,
        ):
            captured["calls"] += 1
            captured["metadata_keys"].append(sorted((metadata or {}).keys()))
            if on_complete:
                on_complete()
            if on_finally:
                on_finally(failed=False, triage_failed=False)

        from chronovisor.ingest import ingest as ingest_mod

        monkeypatch.setattr(ingest_mod, "run_ingest", fake_run_ingest)
        monkeypatch.setattr(orchestrator, "is_available", lambda: True)

        first = orchestrator.run_pending_ingest()
        second = orchestrator.run_pending_ingest()

        assert first["triggered"] is True
        assert len(first["job_ids"]) == 5
        assert sorted(first["files_processed"]) == [f"r{i}.md" for i in range(5)]
        # Every raw fed metadata that includes the raw_keywords side channel.
        assert all("raw_keywords" in keys for keys in captured["metadata_keys"])
        assert captured["calls"] == 5

        # Second call: every raw is now marked processed → below threshold.
        assert second["triggered"] is False
        assert (
            "threshold" in second["reason"].lower()
            or "pending" in second["reason"].lower()
        )

    @pytest.mark.parametrize("lint_error", [False, True])
    def test_successful_batch_runs_bounded_post_ingest_lint_once(
        self,
        isolated_wiki: Path,
        monkeypatch: pytest.MonkeyPatch,
        lint_error: bool,
    ) -> None:
        from chronovisor.ingest import ingest as ingest_mod
        from chronovisor.ingest import lint, orchestrator

        raw_path = isolated_wiki / "raw" / "lint-after-success.md"
        raw_path.write_text("source-grounded fact", encoding="utf-8")

        def fake_run_ingest(
            _content,
            _job_id,
            on_complete=None,
            **_kwargs,
        ) -> None:
            assert on_complete is not None
            on_complete()

        calls = 0

        def check() -> list[dict]:
            nonlocal calls
            calls += 1
            if lint_error:
                raise RuntimeError("private lint detail")
            return [
                {
                    "type": "broken_link",
                    "severity": "high",
                    "page": "example",
                    "detail": "private lint detail",
                    "auto_fixable": True,
                }
            ]

        monkeypatch.setattr(ingest_mod, "run_ingest", fake_run_ingest)
        monkeypatch.setattr(orchestrator, "is_available", lambda: True)
        monkeypatch.setattr(lint, "check", check)

        result = orchestrator.run_pending_ingest(force=True)

        assert calls == 1
        assert result["files_processed"] == [raw_path.name]
        assert result["per_raw"][0]["succeeded"] is True
        if lint_error:
            assert result["post_ingest_lint"] == {
                "status": "error",
                "error_category": "RuntimeError",
            }
        else:
            assert result["post_ingest_lint"]["status"] == "ok"
            assert result["post_ingest_lint"]["summary"]["by_type"] == {
                "broken_link": 1
            }
        assert "private lint detail" not in str(result)

    def test_run_pending_ingest_can_limit_pilot_to_one_semantic_unit(
        self, isolated_wiki: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from chronovisor.ingest import ingest as ingest_mod
        from chronovisor.ingest import orchestrator

        raw_paths = [isolated_wiki / "raw" / f"r{i}.md" for i in range(3)]
        for path in raw_paths:
            path.write_text(path.stem, encoding="utf-8")
        observed: list[str] = []

        def fake_run_ingest(
            content,
            job_id,
            on_complete=None,
            on_finally=None,
            *,
            metadata=None,
        ):
            del job_id, metadata
            observed.append(content)
            if on_complete:
                on_complete()
            if on_finally:
                on_finally(failed=False, triage_failed=False)

        monkeypatch.setattr(ingest_mod, "run_ingest", fake_run_ingest)
        monkeypatch.setattr(orchestrator, "is_available", lambda: True)

        result = orchestrator.run_pending_ingest(force=True, max_units=1)

        assert observed == ["r0"]
        assert result["files_processed"] == ["r0.md"]
        assert [path.name for path in orchestrator.get_pending_raw_files()] == [
            "r1.md",
            "r2.md",
        ]

    @pytest.mark.parametrize("max_units", [0, 11])
    def test_run_pending_ingest_rejects_out_of_range_max_units(
        self,
        isolated_wiki: Path,
        max_units: int,
    ) -> None:
        from chronovisor.ingest import orchestrator

        del isolated_wiki
        with pytest.raises(ValueError, match="max_units must be between 1 and 10"):
            orchestrator.run_pending_ingest(force=True, max_units=max_units)

    def test_legacy_triage_counter_cannot_quarantine_unattempted_raws(
        self, isolated_wiki: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from chronovisor.ingest import ingest as ingest_mod
        from chronovisor.ingest import orchestrator

        paths = [isolated_wiki / "raw" / f"raw-{index}.md" for index in range(2)]
        for path in paths:
            path.write_text(path.stem, encoding="utf-8")
        orchestrator.STATE_FILE.write_text(
            json.dumps(
                {
                    "last_ingest": None,
                    "last_lint": None,
                    "processed_raw_files": [],
                    "current_job_id": None,
                    "current_job_pid": None,
                    "current_job_started_at": None,
                    "triage_failure_count": 3,
                }
            ),
            encoding="utf-8",
        )
        calls: list[str] = []

        def fake_run_ingest(
            content, job_id, on_complete=None, on_finally=None, *, metadata=None
        ):
            del job_id, on_finally, metadata
            calls.append(content)
            if on_complete:
                on_complete()

        monkeypatch.setattr(ingest_mod, "run_ingest", fake_run_ingest)
        monkeypatch.setattr(orchestrator, "is_available", lambda: True)

        result = orchestrator.run_pending_ingest(force=True)

        assert sorted(calls) == ["raw-0", "raw-1"]
        assert result["files_processed"] == [path.name for path in paths]
        assert not (isolated_wiki / "raw" / ".dead-letter").exists()
        persisted = json.loads(orchestrator.STATE_FILE.read_text(encoding="utf-8"))
        assert "triage_failure_count" not in persisted


# ---------------------------------------------------------------------------
# Phase 6: per-raw orchestrator invariants (attribution / mark / fallback)
# ---------------------------------------------------------------------------


class TestPerRawOrchestrator:
    """Verifies the Phase 2-5 contract end-to-end at the orchestrator
    level: each raw's keywords reach run_ingest as its own metadata,
    success and failure are tracked per-file, and legacy raws written
    before the field rename are still readable."""

    @staticmethod
    def _install_single_page_ingest(
        monkeypatch: pytest.MonkeyPatch,
        chronovisor_root: Path,
    ) -> dict[str, int]:
        from chronovisor.core.jobs import JobStatus, job_store
        from chronovisor.ingest import ingest as ingest_mod
        from chronovisor.ingest import orchestrator

        calls = {"ingest": 0, "mutation": 0}

        def fake_run_ingest(
            _content, job_id, on_complete=None, on_finally=None, *, metadata=None
        ):
            del metadata
            calls["ingest"] += 1
            page = chronovisor_root / "pages" / "ack-page.md"
            calls["mutation"] += 1
            page.write_text(
                f"---\ntitle: ACK Page\n---\nmutation {calls['mutation']}\n",
                encoding="utf-8",
            )
            job_store.update(
                job_id,
                status=JobStatus.COMPLETED,
                completed_at=datetime.now().isoformat(),
                pages_created=["ack-page"],
                pages_updated=[],
                result={"test_apply": True},
            )
            if on_complete:
                on_complete()
            if on_finally:
                on_finally(failed=False, triage_failed=False)

        monkeypatch.setattr(ingest_mod, "run_ingest", fake_run_ingest)
        monkeypatch.setattr(orchestrator, "is_available", lambda: True)
        return calls

    @staticmethod
    def _seed_applied_terminal_artifact(
        chronovisor_root: Path,
        *,
        raw_content: str,
        source_raw: str,
        page_id: str = "pretriage-recovery",
    ) -> tuple[Path, dict[str, object]]:
        from chronovisor.ingest import ingest

        page_path = chronovisor_root / "pages" / "memory" / f"{page_id}.md"
        page_body = (
            "---\n"
            f"title: {page_id}\n"
            "updated: 2026-07-14\n"
            "---\n"
            "durable postimage before ACK\n"
        )
        result = ingest._review_and_apply_ingest_operations(
            [
                {
                    "type": "create",
                    "filename": f"memory/{page_id}.md",
                    "content": page_body,
                }
            ],
            raw_content=raw_content,
            source_raw=source_raw,
            reviewer=lambda _proposal: {
                "decision": "apply_available",
                "summary": "terminal local approval before simulated crash",
                "failed_operations_disposition": "none",
            },
        )
        assert result["status"] == "apply_available"
        assert "durable postimage before ACK" in page_path.read_text(encoding="utf-8")
        return page_path, result

    def test_state_write_failure_resumes_durable_ack_without_duplicate_apply(
        self, isolated_wiki: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from chronovisor.core.jobs import JobStatus, job_store
        from chronovisor.ingest import orchestrator, raw_completion_ack

        raw_path = isolated_wiki / "raw" / "ack-state-failure.md"
        raw_path.write_text("stable source bytes", encoding="utf-8")
        calls = self._install_single_page_ingest(monkeypatch, isolated_wiki)
        real_save_state = orchestrator._save_state
        injected = {"done": False}

        def fail_first_processed_mark(state: dict) -> None:
            if not injected["done"] and raw_path.name in state.get(
                "processed_raw_files", []
            ):
                injected["done"] = True
                raise OSError("injected processed-state write failure")
            real_save_state(state)

        monkeypatch.setattr(orchestrator, "_save_state", fail_first_processed_mark)

        first = orchestrator.run_pending_ingest(force=True)

        assert first["per_raw"][0]["succeeded"] is False
        assert first["per_raw"][0]["supervision"]["failure_class"] == (
            "ingest.raw_completion_ack_state_pending"
        )
        first_job = job_store.get(first["per_raw"][0]["job_id"])
        assert first_job is not None and first_job.status == JobStatus.FAILED
        assert calls == {"ingest": 1, "mutation": 1}
        assert raw_path in orchestrator.get_pending_raw_files()
        assert raw_completion_ack.receipt_path([raw_path]).is_file()

        # Another raw may legitimately update the same page before this ACK is
        # retried.  The recorded postimage proves what the first job committed;
        # it is not a CAS precondition for retiring that first source later.
        page = isolated_wiki / "pages" / "ack-page.md"
        page.write_text(
            "---\ntitle: ACK Page\n---\nnewer mutation from raw B\n",
            encoding="utf-8",
        )

        second = orchestrator.run_pending_ingest(force=True)

        assert second["files_processed"] == [raw_path.name]
        assert second["per_raw"][0]["succeeded"] is True
        assert second["per_raw"][0]["completion_ack"]["resumed"] is True
        assert calls == {"ingest": 1, "mutation": 1}
        assert "newer mutation from raw B" in page.read_text(encoding="utf-8")
        assert orchestrator.get_pending_raw_files() == []

    def test_applied_artifact_recovers_before_model_and_ack_protects_later_update(
        self, isolated_wiki: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from chronovisor.ingest import (
            ingest,
            ingest_readback,
            orchestrator,
            raw_completion_ack,
        )

        raw_path = isolated_wiki / "raw" / "pretriage-crash.md"
        raw_content = "page applied, process crashed before raw ACK"
        raw_path.write_text(raw_content, encoding="utf-8")
        page_path, _seed = self._seed_applied_terminal_artifact(
            isolated_wiki,
            raw_content=raw_content,
            source_raw=raw_path.name,
        )

        model_calls = {"availability": 0, "triage": 0, "generation": 0}

        def forbidden_availability() -> bool:
            model_calls["availability"] += 1
            raise AssertionError("terminal recovery reached Ollama availability")

        def forbidden_triage(*_args, **_kwargs):
            model_calls["triage"] += 1
            raise AssertionError("terminal recovery repeated triage")

        def forbidden_generation(*_args, **_kwargs):
            model_calls["generation"] += 1
            raise AssertionError("terminal recovery repeated generation")

        monkeypatch.setattr(orchestrator, "is_available", lambda: True)
        monkeypatch.setattr(ingest, "is_available", forbidden_availability)
        monkeypatch.setattr(ingest, "_triage_with_progress", forbidden_triage)
        monkeypatch.setattr(ingest, "_generate_local_operations", forbidden_generation)
        monkeypatch.setattr(
            ingest_readback,
            "_refresh_ingest_derived_artifacts",
            lambda *_args, **_kwargs: {"checked": 0, "passed": 0, "failed": []},
        )

        recovered = orchestrator.run_pending_ingest(force=True)

        assert recovered["files_processed"] == [raw_path.name]
        assert recovered["per_raw"][0]["succeeded"] is True
        assert recovered["per_raw"][0]["completion_ack"]["resumed"] is False
        assert model_calls == {"availability": 0, "triage": 0, "generation": 0}
        assert raw_completion_ack.receipt_path([raw_path]).is_file()

        # A later raw is allowed to change the same page. Even if the first
        # processed-state mark is lost afterward, its ACK-only replay must not
        # restore the older postimage or run a model.
        later_body = (
            "---\ntitle: pretriage-recovery\nupdated: 2026-07-15\n---\n"
            "newer same-page update\n"
        )
        page_path.write_text(later_body, encoding="utf-8")
        state = orchestrator._load_state()
        state["processed_raw_files"].remove(raw_path.name)
        orchestrator._save_state(state)

        ack_only = orchestrator.run_pending_ingest(force=True)

        assert ack_only["files_processed"] == [raw_path.name]
        assert ack_only["per_raw"][0]["completion_ack"]["resumed"] is True
        assert page_path.read_text(encoding="utf-8") == later_body
        assert model_calls == {"availability": 0, "triage": 0, "generation": 0}

    def test_pretriage_terminal_recovery_rejects_tampered_current_review(
        self, isolated_wiki: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from chronovisor.core.jobs import JobStatus, job_store
        from chronovisor.ingest import ingest

        raw_content = "tampered terminal artifact must not reach a model"
        _page, seeded = self._seed_applied_terminal_artifact(
            isolated_wiki,
            raw_content=raw_content,
            source_raw="tampered-terminal.md",
            page_id="tampered-terminal",
        )
        _proposal_path, review_path = ingest._ingest_artifact_paths(
            str(seeded["source_key"])
        )
        review = json.loads(review_path.read_text(encoding="utf-8"))
        review["proposal_sha256"] = "0" * 64
        review_path.write_text(json.dumps(review), encoding="utf-8")
        model_calls: list[str] = []
        monkeypatch.setattr(
            ingest,
            "is_available",
            lambda: model_calls.append("availability") or True,
        )
        monkeypatch.setattr(
            ingest,
            "_triage_with_progress",
            lambda *_args, **_kwargs: model_calls.append("triage") or [],
        )
        job = job_store.create(processor="unavailable")

        ingest.run_ingest(raw_content, job.job_id)

        finished = job_store.get(job.job_id)
        assert finished is not None and finished.status is JobStatus.FAILED
        assert "terminal review artifact binding is invalid" in str(finished.error)
        assert model_calls == []

    def test_pretriage_proof_race_has_no_model_or_derived_side_effect(
        self, isolated_wiki: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from chronovisor.core.jobs import JobStatus, job_store
        from chronovisor.ingest import ingest, ingest_readback

        raw_content = "terminal proof changes before the recovery lock"
        self._seed_applied_terminal_artifact(
            isolated_wiki,
            raw_content=raw_content,
            source_raw="proof-race.md",
            page_id="proof-race",
        )
        real_loader = ingest._load_pretriage_terminal_recovery
        loads = 0

        def changing_proof(*args, **kwargs):
            nonlocal loads
            loads += 1
            if loads == 1:
                return real_loader(*args, **kwargs)
            return None

        derived_calls: list[str] = []
        monkeypatch.setattr(
            ingest,
            "_load_pretriage_terminal_recovery",
            changing_proof,
        )
        monkeypatch.setattr(
            ingest_readback,
            "_refresh_ingest_derived_artifacts",
            lambda *_args, **_kwargs: derived_calls.append("refresh") or {},
        )
        monkeypatch.setattr(
            ingest,
            "is_available",
            lambda: pytest.fail("proof-race recovery reached Ollama"),
        )
        job = job_store.create(processor="unavailable")

        ingest.run_ingest(raw_content, job.job_id)

        finished = job_store.get(job.job_id)
        assert finished is not None and finished.status is JobStatus.FAILED
        assert "proof changed before raw retirement" in str(finished.error)
        assert loads == 2
        assert derived_calls == []

    def test_confirmed_noop_terminal_artifact_recovers_before_model(
        self, isolated_wiki: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from chronovisor.core.jobs import JobStatus, job_store
        from chronovisor.ingest import ingest, ingest_readback

        raw_content = "durably reviewed as a semantic noop"

        def reviewer(_proposal: dict[str, object]) -> dict[str, object]:
            return {
                "decision": "confirmed_noop",
                "summary": "no durable memory effect required",
                "failed_operations_disposition": "none",
            }

        seeded = ingest._review_and_apply_ingest_operations(
            [],
            raw_content=raw_content,
            reviewer=reviewer,
            local_disposition="triage_no_operations",
        )
        assert seeded["status"] == "confirmed_noop"
        monkeypatch.setattr(
            ingest,
            "is_available",
            lambda: pytest.fail("confirmed-noop recovery reached Ollama"),
        )
        monkeypatch.setattr(
            ingest_readback,
            "_refresh_ingest_derived_artifacts",
            lambda *_args, **_kwargs: {"checked": 0, "passed": 0, "failed": []},
        )
        completed_callbacks: list[str] = []
        job = job_store.create(processor="unavailable")

        ingest.run_ingest(
            raw_content,
            job.job_id,
            on_complete=lambda: completed_callbacks.append("completed"),
            frontier_reviewer=reviewer,
        )

        finished = job_store.get(job.job_id)
        assert finished is not None and finished.status is JobStatus.COMPLETED
        assert finished.processor == "durable-ingest-recovery"
        assert finished.result["pretriage_recovery"] == {
            "basis": "durable_confirmed_noop",
            "model_calls": 0,
        }
        assert completed_callbacks == ["completed"]

    def test_unfinished_proposal_is_not_a_pretriage_terminal_recovery(
        self, isolated_wiki: Path
    ) -> None:
        from chronovisor.ingest import ingest

        raw_content = "proposal exists but no terminal review"
        result = ingest._review_and_apply_ingest_operations(
            [],
            raw_content=raw_content,
            reviewer=lambda _proposal: {
                "decision": "retry",
                "summary": "not terminal",
                "failed_operations_disposition": "none",
            },
            local_disposition="triage_no_operations",
        )
        assert result["status"] == "needs_retry"
        assert (
            ingest._load_pretriage_terminal_recovery(
                raw_content,
                None,
                reviewer=None,
            )
            is None
        )

    @pytest.mark.parametrize("corruption", ["receipt", "source"])
    def test_invalid_or_source_mismatched_receipt_fails_closed_without_model(
        self,
        isolated_wiki: Path,
        monkeypatch: pytest.MonkeyPatch,
        corruption: str,
    ) -> None:
        from chronovisor.core import background_jobs
        from chronovisor.ingest import orchestrator, raw_completion_ack

        raw_path = isolated_wiki / "raw" / f"ack-{corruption}.md"
        raw_path.write_text("original source", encoding="utf-8")
        calls = self._install_single_page_ingest(monkeypatch, isolated_wiki)
        monkeypatch.setattr(background_jobs, "start_self_heal_background", lambda _path: None)
        completed = orchestrator.run_pending_ingest(force=True)
        assert completed["files_processed"] == [raw_path.name]

        state = orchestrator._load_state()
        state["processed_raw_files"].remove(raw_path.name)
        orchestrator._save_state(state)
        receipt = raw_completion_ack.receipt_path([raw_path])
        if corruption == "receipt":
            payload = json.loads(receipt.read_text(encoding="utf-8"))
            payload["receipt_sha256"] = "0" * 64
            receipt.write_text(
                json.dumps(
                    payload,
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                )
                + "\n",
                encoding="utf-8",
            )
        else:
            raw_path.write_text("changed source", encoding="utf-8")

        failed = orchestrator.run_pending_ingest(force=True)

        assert failed["per_raw"][0]["succeeded"] is False
        assert failed["per_raw"][0]["supervision"]["failure_class"] == (
            "ingest.raw_completion_receipt_invalid"
        )
        assert calls == {"ingest": 1, "mutation": 1}
        assert orchestrator.get_pending_raw_files() == []

    def test_receipt_publication_failure_defers_without_reingest(
        self, isolated_wiki: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from chronovisor.core import background_jobs
        from chronovisor.ingest import orchestrator, raw_completion_ack

        raw_path = isolated_wiki / "raw" / "ack-publish-failure.md"
        raw_path.write_text("source survives", encoding="utf-8")
        calls = self._install_single_page_ingest(monkeypatch, isolated_wiki)
        monkeypatch.setattr(background_jobs, "start_self_heal_background", lambda _path: None)
        monkeypatch.setattr(
            raw_completion_ack,
            "atomic_write",
            lambda *_args, **_kwargs: (_ for _ in ()).throw(
                OSError("injected receipt disk failure")
            ),
        )

        first = orchestrator.run_pending_ingest(force=True)
        second = orchestrator.run_pending_ingest(force=True)

        assert first["per_raw"][0]["succeeded"] is False
        assert first["per_raw"][0]["supervision"]["failure_class"] == (
            "ingest.raw_completion_receipt_publish_failed"
        )
        assert second == {"triggered": False, "reason": "no pending raws"}
        assert calls == {"ingest": 1, "mutation": 1}
        assert raw_path.exists()

    def test_attribution_two_raws_distinct_keywords(
        self, isolated_wiki: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Each raw's frontmatter keywords must reach run_ingest as that
        raw's own metadata — no blanket copy across the batch."""
        from chronovisor.ingest import ingest as ingest_mod
        from chronovisor.ingest import orchestrator

        (isolated_wiki / "raw" / "a.md").write_text(
            "---\nraw_keywords: [alpha-1, alpha-2]\n---\nA body\n"
        )
        (isolated_wiki / "raw" / "b.md").write_text(
            "---\nraw_keywords: [beta-1]\n---\nB body\n"
        )
        # The orchestrator's threshold is 5; force=True bypasses it.
        for i in range(3):
            (isolated_wiki / "raw" / f"f{i}.md").write_text("body")

        seen: list[tuple[str, list[str] | None]] = []

        def fake_run_ingest(
            content,
            job_id,
            on_complete=None,
            on_finally=None,
            *,
            metadata=None,
        ):
            kw = (metadata or {}).get("raw_keywords")
            # Identify which raw this call belongs to by the leading body
            # line (we wrote distinct bodies above).
            tag = "a" if "A body" in content else "b" if "B body" in content else "f"
            seen.append((tag, kw))
            if on_complete:
                on_complete()
            if on_finally:
                on_finally(failed=False, triage_failed=False)

        monkeypatch.setattr(ingest_mod, "run_ingest", fake_run_ingest)
        monkeypatch.setattr(orchestrator, "is_available", lambda: True)

        result = orchestrator.run_pending_ingest(force=True)
        assert result["triggered"] is True
        # Every distinct raw_keywords payload was delivered to its OWN raw.
        a_calls = [kw for tag, kw in seen if tag == "a"]
        b_calls = [kw for tag, kw in seen if tag == "b"]
        f_calls = [kw for tag, kw in seen if tag == "f"]
        assert a_calls == [["alpha-1", "alpha-2"]]
        assert b_calls == [["beta-1"]]
        # The keyword-less raws got an empty list (not the neighbors' list).
        assert all(kw == [] for kw in f_calls)

    def test_legacy_keywords_field_falls_back(
        self, isolated_wiki: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Pre-rename raws written with ``keywords:`` (not ``raw_keywords:``)
        must still propagate via the legacy fallback path."""
        from chronovisor.ingest import ingest as ingest_mod
        from chronovisor.ingest import orchestrator

        (isolated_wiki / "raw" / "legacy.md").write_text(
            "---\nkeywords: [legacy-only]\n---\nlegacy body\n"
        )

        seen: list[list[str] | None] = []

        def fake_run_ingest(
            content, job_id, on_complete=None, on_finally=None, *, metadata=None
        ):
            seen.append((metadata or {}).get("raw_keywords"))
            if on_complete:
                on_complete()
            if on_finally:
                on_finally(failed=False, triage_failed=False)

        monkeypatch.setattr(ingest_mod, "run_ingest", fake_run_ingest)
        monkeypatch.setattr(orchestrator, "is_available", lambda: True)

        orchestrator.run_pending_ingest(force=True)
        assert seen == [["legacy-only"]]

    def test_raw_keywords_preferred_over_legacy_when_both_present(
        self, isolated_wiki: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """If a raw has both fields (transitional state), the new
        ``raw_keywords`` wins — fallback is a strict else branch."""
        from chronovisor.ingest import ingest as ingest_mod
        from chronovisor.ingest import orchestrator

        (isolated_wiki / "raw" / "both.md").write_text(
            "---\nraw_keywords: [new-name]\nkeywords: [old-name]\n---\nbody\n"
        )

        seen: list[list[str] | None] = []

        def fake_run_ingest(
            content, job_id, on_complete=None, on_finally=None, *, metadata=None
        ):
            seen.append((metadata or {}).get("raw_keywords"))
            if on_complete:
                on_complete()
            if on_finally:
                on_finally(failed=False, triage_failed=False)

        monkeypatch.setattr(ingest_mod, "run_ingest", fake_run_ingest)
        monkeypatch.setattr(orchestrator, "is_available", lambda: True)

        orchestrator.run_pending_ingest(force=True)
        assert seen == [["new-name"]]

    def test_per_raw_mark_partial_failure(
        self, isolated_wiki: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A failed raw must remain pending while its successful peers
        get marked processed individually."""
        from chronovisor.ingest import ingest as ingest_mod
        from chronovisor.ingest import orchestrator

        (isolated_wiki / "raw" / "ok.md").write_text("ok body")
        (isolated_wiki / "raw" / "broken.md").write_text("broken body")

        def fake_run_ingest(
            content, job_id, on_complete=None, on_finally=None, *, metadata=None
        ):
            # ``ok`` succeeds (calls on_complete), ``broken`` fails
            # (skips on_complete) — mirrors run_ingest's contract for
            # full success vs failure.
            if "ok body" in content:
                if on_complete:
                    on_complete()
                if on_finally:
                    on_finally(failed=False, triage_failed=False)
            else:
                if on_finally:
                    on_finally(failed=True, triage_failed=False)

        monkeypatch.setattr(ingest_mod, "run_ingest", fake_run_ingest)
        monkeypatch.setattr(orchestrator, "is_available", lambda: True)

        result = orchestrator.run_pending_ingest(force=True)

        assert result["files_processed"] == ["ok.md"]
        # ``broken.md`` is still pending for the next tick.
        pending = {p.name for p in orchestrator.get_pending_raw_files()}
        assert "broken.md" in pending
        assert "ok.md" not in pending

    def test_distinct_triage_failures_are_retried_per_raw_without_bulk_quarantine(
        self, isolated_wiki: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from chronovisor.core.jobs import JobStatus, job_store
        from chronovisor.ingest import ingest as ingest_mod
        from chronovisor.ingest import orchestrator

        paths = [isolated_wiki / "raw" / f"bad-{index}.md" for index in range(4)]
        for path in paths:
            path.write_text(path.stem, encoding="utf-8")

        def fake_run_ingest(
            content, job_id, on_complete=None, on_finally=None, *, metadata=None
        ):
            del content, on_complete, metadata
            job_store.update(
                job_id,
                status=JobStatus.FAILED,
                error="triage parse failed after convergence attempts",
            )
            if on_finally:
                on_finally(failed=True, triage_failed=True)

        monkeypatch.setattr(ingest_mod, "run_ingest", fake_run_ingest)
        monkeypatch.setattr(orchestrator, "is_available", lambda: True)

        first = orchestrator.run_pending_ingest(force=True)
        second = orchestrator.run_pending_ingest(force=True)

        assert [row["supervision"]["attempts"] for row in first["per_raw"]] == [
            1,
            1,
            1,
            1,
        ]
        assert [row["supervision"]["attempts"] for row in second["per_raw"]] == [
            2,
            2,
            2,
            2,
        ]
        assert all(path.exists() for path in paths)
        assert orchestrator.get_pending_raw_files() == paths
        dead_letter = isolated_wiki / "raw" / ".dead-letter"
        assert not dead_letter.exists() or list(dead_letter.glob("*.md")) == []

    def test_repeated_apply_failure_quarantines_raw(
        self, isolated_wiki: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Repeated deterministic apply failures leave the queue and become
        self-healing packets instead of being retried forever."""
        from chronovisor.core.jobs import JobStatus, job_store
        from chronovisor.ingest import ingest as ingest_mod
        from chronovisor.ingest import orchestrator

        raw_path = isolated_wiki / "raw" / "broken.md"
        raw_path.write_text("broken body")
        _seed_page(
            isolated_wiki,
            "ai/opus-4.7-evaluation-and-industry-geopolitics.md",
            "---\ntitle: Opus\nupdated: 2026-01-01\n---\nold",
        )

        def fake_run_ingest(
            content, job_id, on_complete=None, on_finally=None, *, metadata=None
        ):
            job_store.update(
                job_id,
                status=JobStatus.FAILED,
                error=(
                    "update target not found for page_id "
                    "'opus-4-7-evaluation-and-industry-geopolitics'"
                ),
            )
            if on_finally:
                on_finally(failed=True, triage_failed=False)

        monkeypatch.setattr(ingest_mod, "run_ingest", fake_run_ingest)
        monkeypatch.setattr(orchestrator, "is_available", lambda: True)

        first = orchestrator.run_pending_ingest(force=True)
        second = orchestrator.run_pending_ingest(force=True)
        third = orchestrator.run_pending_ingest(force=True)

        assert first["per_raw"][0]["supervision"]["attempts"] == 1
        assert second["per_raw"][0]["supervision"]["attempts"] == 2
        supervision = third["per_raw"][0]["supervision"]
        assert supervision["attempts"] == 3
        assert supervision["quarantined"] is True
        assert not raw_path.exists()
        packet_path = Path(supervision["packet_path"])
        assert packet_path.exists()
        packet = json.loads(packet_path.read_text())
        assert packet["failure_class"] == "apply.update_target_not_found"
        assert packet["similar_existing_pages"] == [
            "ai/opus-4.7-evaluation-and-industry-geopolitics"
        ]
        assert orchestrator.get_pending_raw_files() == []

    def test_frontier_nonconvergence_immediately_queues_self_heal(
        self, isolated_wiki: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from chronovisor.core import background_jobs
        from chronovisor.ingest import failure_supervisor

        raw_path = isolated_wiki / "raw" / "frontier-loop.md"
        raw_path.write_text("grounded source", encoding="utf-8")
        started: list[Path] = []
        monkeypatch.setattr(background_jobs, "start_self_heal_background", started.append)

        result = failure_supervisor.record_raw_failure(
            raw_path=raw_path,
            error=(
                "frontier ingest review did not converge after 3 attempts: "
                "taxonomy oscillated from scenario to event"
            ),
            raw_text="grounded source",
        )

        assert result.failure_class == "ingest.frontier_nonconvergent"
        assert result.attempts == 1
        assert result.quarantined is True
        assert result.packet_path is not None
        assert started == [Path(result.packet_path)]
        packet = json.loads(Path(result.packet_path).read_text(encoding="utf-8"))
        assert packet["fingerprint"] == "ingest.frontier_nonconvergent"
        assert not raw_path.exists()

    def test_local_consensus_nonconvergence_immediately_queues_self_heal(
        self, isolated_wiki: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from chronovisor.core import background_jobs
        from chronovisor.ingest import failure_supervisor

        raw_path = isolated_wiki / "raw" / "local-consensus-loop.md"
        raw_path.write_text("grounded source", encoding="utf-8")
        started: list[Path] = []
        monkeypatch.setattr(background_jobs, "start_self_heal_background", started.append)

        result = failure_supervisor.record_raw_failure(
            raw_path=raw_path,
            error=(
                "local consensus ingest review did not converge after "
                "2 local review calls: structured review budget exhausted (2/2)"
            ),
            raw_text="grounded source",
        )

        assert result.failure_class == "ingest.local_consensus_nonconvergent"
        assert result.attempts == 1
        assert result.quarantined is True
        assert result.packet_path is not None
        assert started == [Path(result.packet_path)]
        packet = json.loads(Path(result.packet_path).read_text(encoding="utf-8"))
        assert packet["fingerprint"] == "ingest.local_consensus_nonconvergent"
        assert not raw_path.exists()

    def test_invalid_local_authority_is_one_operational_failure_not_bad_raws(
        self, isolated_wiki: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from chronovisor.core import background_jobs
        from chronovisor.ingest import failure_supervisor

        first_raw = isolated_wiki / "raw" / "first-authority-failure.md"
        second_raw = isolated_wiki / "raw" / "second-authority-failure.md"
        first_raw.write_text("first grounded source", encoding="utf-8")
        second_raw.write_text("second grounded source", encoding="utf-8")
        started: list[Path] = []
        monkeypatch.setattr(background_jobs, "start_self_heal_background", started.append)
        error = (
            "local consensus authority unavailable: "
            "adoption_artifact_invalid:evaluation evidence is inconsistent"
        )

        first = failure_supervisor.record_raw_failure(
            raw_path=first_raw,
            error=error,
            raw_text="first grounded source",
        )
        second = failure_supervisor.record_raw_failure(
            raw_path=second_raw,
            error=error,
            raw_text="second grounded source",
        )

        assert first.failure_class == (
            "ingest.runtime_local_consensus_authority_unavailable"
        )
        assert first.fingerprint.endswith(":adoption_artifact_invalid")
        assert first.quarantined is False
        assert second.packet_path == first.packet_path
        assert started == [Path(str(first.packet_path))]
        assert first_raw.exists()
        assert second_raw.exists()
        deferred = failure_supervisor.operational_deferred_raw_files(
            [first_raw, second_raw]
        )
        assert set(deferred) == {first_raw.name, second_raw.name}

        completed_packet_path = Path(str(first.packet_path))
        completed_packet = json.loads(completed_packet_path.read_text(encoding="utf-8"))
        completed_packet["status"] = "local_repair_applied"
        completed_packet_path.write_text(
            json.dumps(completed_packet, ensure_ascii=False),
            encoding="utf-8",
        )
        recurring_raw = isolated_wiki / "raw" / "recurring-authority-failure.md"
        recurring_raw.write_text("third grounded source", encoding="utf-8")

        recurring = failure_supervisor.record_raw_failure(
            raw_path=recurring_raw,
            error=error,
            raw_text="third grounded source",
        )

        assert recurring.packet_path != first.packet_path
        assert started == [completed_packet_path, Path(str(recurring.packet_path))]
        assert failure_supervisor.operational_deferred_raw_files(
            [first_raw, second_raw, recurring_raw]
        ) == {recurring_raw.name: "pending_local_repair"}

    def test_valid_adopted_authority_releases_invalid_artifact_hold(
        self, isolated_wiki: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from chronovisor.core import background_jobs
        from chronovisor.ingest import failure_supervisor

        raw_path = isolated_wiki / "raw" / "authority-recovered.md"
        raw_path.write_text("grounded source", encoding="utf-8")
        monkeypatch.setattr(background_jobs, "start_self_heal_background", lambda _path: None)

        result = failure_supervisor.record_raw_failure(
            raw_path=raw_path,
            error=(
                "local consensus authority unavailable: "
                "adoption_artifact_invalid:evaluation evidence is inconsistent"
            ),
            raw_text="grounded source",
        )

        assert result.fingerprint == (
            failure_supervisor.ADOPTION_ARTIFACT_INVALID_FINGERPRINT
        )
        assert failure_supervisor.operational_deferred_raw_files([raw_path]) == {
            raw_path.name: "pending_local_repair"
        }

        monkeypatch.setattr(
            failure_supervisor,
            "_current_adopted_authority_sha256",
            lambda: "a" * 64,
        )

        assert failure_supervisor.operational_deferred_raw_files([raw_path]) == {}

    def test_semantic_hold_reenters_ingest_after_executable_policy_epoch_change(
        self, isolated_wiki: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from chronovisor.ingest import failure_supervisor

        artifact = "a" * 64
        epoch = ["b" * 64]
        monkeypatch.setattr(
            failure_supervisor,
            "_current_adopted_authority_sha256",
            lambda: artifact,
        )
        monkeypatch.setattr(
            failure_supervisor,
            "_current_adopted_authority_epoch",
            lambda: epoch[0],
        )
        raw_path = isolated_wiki / "raw" / "policy-epoch-retry.md"
        raw_path.write_text("grounded source", encoding="utf-8")

        supervision = failure_supervisor.record_raw_failure(
            raw_path=raw_path,
            error=(
                "local consensus semantic no quorum "
                f"[authority_sha256={artifact}]: two proposals disagreed"
            ),
            raw_text="grounded source",
        )

        assert supervision.terminal_deferred is True
        assert failure_supervisor.operational_deferred_raw_files([raw_path]) == {
            raw_path.name: failure_supervisor.SEMANTIC_NO_QUORUM_DEFER_REASON
        }

        epoch[0] = "c" * 64

        assert failure_supervisor.operational_deferred_raw_files([raw_path]) == {}

    def test_generation_repair_exhaustion_is_operational_not_bad_raw(
        self, isolated_wiki: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from chronovisor.core import background_jobs
        from chronovisor.ingest import failure_supervisor

        raw_path = isolated_wiki / "raw" / "generation-contract.md"
        raw_path.write_text("grounded source", encoding="utf-8")
        started: list[Path] = []
        monkeypatch.setattr(background_jobs, "start_self_heal_background", started.append)

        result = failure_supervisor.record_raw_failure(
            raw_path=raw_path,
            error=(
                "ingest generation repair_exhausted: all 1 planned page "
                "operations failed locally"
            ),
            raw_text="grounded source",
        )

        assert result.failure_class == "ingest.generation_repair_exhausted"
        assert result.quarantined is False
        assert result.packet_path is not None
        assert raw_path.exists()
        assert started == [Path(result.packet_path)]
        assert failure_supervisor.operational_deferred_raw_files([raw_path]) == {
            raw_path.name: "pending_local_repair"
        }

    def test_generation_transport_error_is_operational_not_transient(
        self, isolated_wiki: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from chronovisor.core import background_jobs
        from chronovisor.ingest import failure_supervisor

        raw_path = isolated_wiki / "raw" / "generation-transport.md"
        raw_path.write_text("grounded source", encoding="utf-8")
        started: list[Path] = []
        monkeypatch.setattr(background_jobs, "start_self_heal_background", started.append)

        result = failure_supervisor.record_raw_failure(
            raw_path=raw_path,
            error="ingest generation transport_error: Ollama connection reset",
            raw_text="grounded source",
        )

        assert result.failure_class == "ingest.generation_transport_error"
        assert result.tracked is True
        assert result.transient is False
        assert result.quarantined is False
        assert result.packet_path is not None
        assert raw_path.exists()
        assert started == [Path(result.packet_path)]
        assert failure_supervisor.operational_deferred_raw_files([raw_path]) == {
            raw_path.name: "pending_local_repair"
        }

    def test_ollama_unavailable_stays_pending_without_self_heal_packet(
        self, isolated_wiki: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A local model outage is infrastructure state, not a bad raw.

        It must not accumulate per-raw attempts or quarantine otherwise valid
        source material into the self-heal queue.
        """
        from chronovisor.ingest import ingest as ingest_mod
        from chronovisor.ingest import orchestrator

        raw_path = isolated_wiki / "raw" / "model-memory.md"
        raw_path.write_text("body")

        monkeypatch.setattr(orchestrator, "is_available", lambda: False)
        monkeypatch.setattr(ingest_mod, "is_available", lambda: False)

        result = {}
        for _ in range(3):
            result = orchestrator.run_pending_ingest(force=True)

        supervision = result["per_raw"][0]["supervision"]
        assert supervision["failure_class"] == "ingest.ollama_unavailable"
        assert supervision["attempts"] == 0
        assert supervision["tracked"] is False
        assert supervision["transient"] is True
        assert supervision["quarantined"] is False
        assert raw_path.exists()
        assert orchestrator.get_pending_raw_files() == [raw_path]
        packets_dir = isolated_wiki / "runtime" / "failures" / "packets"
        assert not packets_dir.exists() or list(packets_dir.iterdir()) == []

    def test_legacy_sonnet_fallback_error_is_not_tracked_as_raw_failure(
        self, isolated_wiki: Path
    ) -> None:
        from chronovisor.ingest import failure_supervisor

        raw_path = isolated_wiki / "raw" / "legacy.md"
        raw_path.write_text("body")

        result = failure_supervisor.record_raw_failure(
            raw_path=raw_path,
            error="Sonnet fallback not yet implemented",
            raw_text="body",
        )

        assert result.failure_class == "ingest.ollama_unavailable"
        assert result.attempts == 0
        assert result.tracked is False
        assert result.transient is True
        assert raw_path.exists()
        assert not (isolated_wiki / "runtime" / "failures" / "state.json").exists()

    @pytest.mark.parametrize(
        "error",
        [
            "triage structured failure [capacity_unavailable]: no runner fits",
            "triage structured failure [transport_error]: connection reset",
            "triage structured failure [transport_timeout]: request timed out",
            "ingest generation capacity_unavailable: no runner fits",
        ],
    )
    def test_transient_runtime_failure_stays_pending_without_quarantine(
        self, isolated_wiki: Path, error: str
    ) -> None:
        from chronovisor.ingest import failure_supervisor

        raw_path = isolated_wiki / "raw" / "large.md"
        raw_path.write_text("large source", encoding="utf-8")

        result = failure_supervisor.record_raw_failure(
            raw_path=raw_path,
            error=error,
            raw_text="large source",
        )

        assert result.attempts == 0
        assert result.tracked is False
        assert result.transient is True
        assert result.quarantined is False
        assert raw_path.exists()
        assert not (isolated_wiki / "runtime" / "failures" / "state.json").exists()

    @pytest.mark.parametrize(
        ("error", "expected_class"),
        [
            (
                "triage structured failure [schema_invalid]: malformed schema",
                "ingest.runtime_schema_invalid",
            ),
            (
                "triage structured failure [input_invalid]: prompt is not text",
                "ingest.runtime_input_invalid",
            ),
            (
                "triage structured failure [input_too_large]: input too large",
                "ingest.runtime_input_too_large",
            ),
            (
                "local consensus structured failure [input_too_large]: one op too large",
                "ingest.runtime_input_too_large",
            ),
            (
                "triage structured failure [feedback_too_large]: feedback too large",
                "ingest.runtime_feedback_too_large",
            ),
            (
                "triage structured failure [output_too_large]: output too large",
                "ingest.runtime_output_too_large",
            ),
            (
                "triage structured failure [value_validation_error]: validator failed",
                "ingest.runtime_value_validation_error",
            ),
            (
                "triage structured failure [context_truncation_suspected]: shifted",
                "ingest.runtime_context_truncation_suspected",
            ),
            (
                "triage structured failure [context_window_exceeded]: input too large",
                "ingest.runtime_context_window_exceeded",
            ),
            (
                "local consensus structured failure [context_window_exceeded]: 130000>114688",
                "ingest.runtime_context_window_exceeded",
            ),
            (
                "triage structured failure [stream_incomplete]: stream ended early",
                "ingest.runtime_stream_incomplete",
            ),
            (
                "triage structured failure [completion_incomplete]: done missing",
                "ingest.runtime_completion_incomplete",
            ),
            (
                "triage structured failure [output_truncated]: token limit",
                "ingest.runtime_output_truncated",
            ),
            (
                "triage structured failure [repair_exhausted]: invalid twice",
                "ingest.runtime_triage_repair_exhausted",
            ),
            (
                "triage structured failure [repeated_output]: same invalid JSON",
                "ingest.runtime_triage_repeated_output",
            ),
            (
                "triage structured failure [unknown]: no structured plan",
                "ingest.runtime_triage_unknown",
            ),
            (
                "ingest generation context_window_exceeded: input too large",
                "ingest.generation_context_window_exceeded",
            ),
            (
                "ingest generation context_truncation_suspected: shifted",
                "ingest.generation_context_truncation_suspected",
            ),
            (
                "ingest generation feedback_too_large: repair feedback too large",
                "ingest.generation_feedback_too_large",
            ),
            (
                "ingest generation completion_incomplete: done missing",
                "ingest.generation_completion_incomplete",
            ),
            (
                "ingest generation stream_incomplete: stream ended early",
                "ingest.generation_stream_incomplete",
            ),
            (
                "ingest generation output_truncated: token limit",
                "ingest.generation_output_truncated",
            ),
        ],
    )
    def test_operational_runtime_failure_queues_self_heal_once_without_quarantine(
        self,
        isolated_wiki: Path,
        monkeypatch: pytest.MonkeyPatch,
        error: str,
        expected_class: str,
    ) -> None:
        from chronovisor.core import background_jobs
        from chronovisor.ingest import failure_supervisor

        raw_path = isolated_wiki / "raw" / "large.md"
        raw_path.write_text("large source", encoding="utf-8")
        started: list[Path] = []
        monkeypatch.setattr(background_jobs, "start_self_heal_background", started.append)

        first = failure_supervisor.record_raw_failure(
            raw_path=raw_path,
            error=error,
            job_id="job-1",
            raw_text="large source",
        )
        second = failure_supervisor.record_raw_failure(
            raw_path=raw_path,
            error=error,
            job_id="job-2",
            raw_text="large source",
        )

        assert first.failure_class == expected_class
        assert first.attempts == 1
        assert first.tracked is True
        assert first.transient is False
        assert first.quarantined is False
        assert first.quarantine_path is None
        assert first.packet_path is not None
        assert second.packet_path == first.packet_path
        assert second.attempts == 1
        assert second.tracked is True
        assert second.quarantined is False
        assert raw_path.exists()
        assert started == [Path(first.packet_path)]

        packets_dir = isolated_wiki / "runtime" / "failures" / "packets"
        assert list(packets_dir.glob("*.json")) == [Path(first.packet_path)]
        state = json.loads(
            (isolated_wiki / "runtime" / "failures" / "state.json").read_text(
                encoding="utf-8"
            )
        )
        entry = state["failures"][raw_path.name]
        assert entry["attempts"] == 1
        assert entry["self_heal_queued"] is True
        assert entry["packet_path"] == first.packet_path
        assert entry["launch_status"] == "started"
        assert entry["launch_error"] is None

    def test_same_operational_fingerprint_across_raws_reuses_one_self_heal(
        self, isolated_wiki: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from chronovisor.core import background_jobs
        from chronovisor.ingest import failure_supervisor

        first_raw = isolated_wiki / "raw" / "first.md"
        second_raw = isolated_wiki / "raw" / "second.md"
        first_raw.write_text("first source", encoding="utf-8")
        second_raw.write_text("second source", encoding="utf-8")
        started: list[Path] = []
        monkeypatch.setattr(background_jobs, "start_self_heal_background", started.append)
        error = "triage structured failure [schema_invalid]: malformed schema"

        first = failure_supervisor.record_raw_failure(
            raw_path=first_raw,
            error=error,
            raw_text="first source",
        )
        second = failure_supervisor.record_raw_failure(
            raw_path=second_raw,
            error=error,
            raw_text="second source",
        )

        assert first.packet_path is not None
        assert second.packet_path == first.packet_path
        assert started == [Path(first.packet_path)]
        assert first_raw.exists()
        assert second_raw.exists()
        packets_dir = isolated_wiki / "runtime" / "failures" / "packets"
        assert list(packets_dir.glob("*.json")) == [Path(first.packet_path)]
        state = json.loads(
            (isolated_wiki / "runtime" / "failures" / "state.json").read_text(
                encoding="utf-8"
            )
        )
        assert state["failures"][first_raw.name]["packet_path"] == first.packet_path
        assert state["failures"][second_raw.name]["packet_path"] == first.packet_path
        assert list(state["operational_failures"]) == ["ingest.runtime_schema_invalid"]

    def test_operational_self_heal_launch_failure_is_durable_and_not_retried(
        self, isolated_wiki: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from chronovisor.core import background_jobs
        from chronovisor.ingest import failure_supervisor

        raw_path = isolated_wiki / "raw" / "system-defect.md"
        raw_path.write_text("valid source", encoding="utf-8")
        starts = 0

        def fail_start(_packet_path: Path) -> None:
            nonlocal starts
            starts += 1
            raise RuntimeError("queue ledger unavailable")

        monkeypatch.setattr(background_jobs, "start_self_heal_background", fail_start)
        error = "triage structured failure [schema_invalid]: malformed schema"

        first = failure_supervisor.record_raw_failure(
            raw_path=raw_path,
            error=error,
            raw_text="valid source",
        )
        second = failure_supervisor.record_raw_failure(
            raw_path=raw_path,
            error=error,
            raw_text="valid source",
        )

        assert starts == 1
        assert first.packet_path == second.packet_path
        assert raw_path.exists()
        state = json.loads(
            (isolated_wiki / "runtime" / "failures" / "state.json").read_text(
                encoding="utf-8"
            )
        )
        entry = state["failures"][raw_path.name]
        assert entry["self_heal_queued"] is True
        assert entry["launch_status"] == "failed"
        assert entry["launch_error"] == "RuntimeError: queue ledger unavailable"
        events = [
            json.loads(line)
            for line in (isolated_wiki / "runtime" / "events.jsonl")
            .read_text(encoding="utf-8")
            .splitlines()
        ]
        assert [
            row
            for row in events
            if row.get("outcome_kind") == "self_heal_launch_failed"
        ]

    def test_operational_defer_skips_second_run_and_repair_success_retries(
        self, isolated_wiki: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from chronovisor.core import background_jobs
        from chronovisor.core.jobs import JobStatus, job_store
        from chronovisor.ingest import ingest as ingest_mod
        from chronovisor.ingest import orchestrator

        raw_path = isolated_wiki / "raw" / "operational.md"
        raw_path.write_text("valid source", encoding="utf-8")
        calls = 0

        def run_ingest(
            _content, job_id, on_complete=None, on_finally=None, *, metadata=None
        ):
            nonlocal calls
            calls += 1
            if calls == 1:
                job_store.update(
                    job_id,
                    status=JobStatus.FAILED,
                    error=(
                        "triage structured failure [schema_invalid]: malformed schema"
                    ),
                )
                return
            if on_complete:
                on_complete()

        monkeypatch.setattr(ingest_mod, "run_ingest", run_ingest)
        monkeypatch.setattr(orchestrator, "is_available", lambda: True)
        monkeypatch.setattr(background_jobs, "start_self_heal_background", lambda _path: None)

        first = orchestrator.run_pending_ingest(force=True)
        second = orchestrator.run_pending_ingest(force=True)

        assert calls == 1
        assert second == {"triggered": False, "reason": "no pending raws"}
        packet_path = Path(first["per_raw"][0]["supervision"]["packet_path"])
        packet = json.loads(packet_path.read_text(encoding="utf-8"))
        packet["status"] = "local_repair_applied"
        packet_path.write_text(json.dumps(packet), encoding="utf-8")

        third = orchestrator.run_pending_ingest(force=True)

        assert calls == 2
        assert third["files_processed"] == [raw_path.name]
        assert raw_path.exists()
        assert orchestrator.get_pending_raw_files() == []

    def test_projection_failure_class_separates_source_from_runtime_defects(
        self,
    ) -> None:
        from chronovisor.ingest import failure_supervisor

        conflict = failure_supervisor.classify_failure(
            "raw semantic projection failed [artifact_conflict]: "
            "ProjectionConflictError: conflict"
        )
        malformed = failure_supervisor.classify_failure(
            "raw semantic projection failed [source_invalid]: "
            "RawSemanticProjectionError: malformed"
        )

        assert conflict.failure_class == (
            "ingest.runtime_semantic_projection_artifact_conflict"
        )
        assert malformed.failure_class == "raw.semantic_projection_source_invalid"
        assert ":projectionconflicterror:" in conflict.fingerprint
        assert conflict.fingerprint != malformed.fingerprint
        assert (
            malformed.failure_class
            not in failure_supervisor.OPERATIONAL_SELF_HEAL_FAILURE_CLASSES
        )

    def test_repeated_failure_after_repair_success_queues_fresh_packet(
        self, isolated_wiki: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from chronovisor.core import background_jobs
        from chronovisor.ingest import failure_supervisor

        raw_path = isolated_wiki / "raw" / "still-broken.md"
        raw_path.write_text("valid source", encoding="utf-8")
        started: list[Path] = []
        monkeypatch.setattr(background_jobs, "start_self_heal_background", started.append)
        error = "triage structured failure [schema_invalid]: malformed schema"

        first = failure_supervisor.record_raw_failure(
            raw_path=raw_path,
            error=error,
            raw_text="valid source",
        )
        assert first.packet_path is not None
        first_packet = Path(first.packet_path)
        payload = json.loads(first_packet.read_text(encoding="utf-8"))
        payload["status"] = "frontier_approved"
        first_packet.write_text(json.dumps(payload), encoding="utf-8")

        second = failure_supervisor.record_raw_failure(
            raw_path=raw_path,
            error=error,
            raw_text="valid source",
        )

        assert second.packet_path is not None
        assert second.packet_path != first.packet_path
        assert started == [first_packet, Path(second.packet_path)]
        assert raw_path.exists()

    def test_valid_projection_child_bundle_releases_old_projection_failure(
        self, isolated_wiki: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from chronovisor.core import background_jobs
        from chronovisor.ingest import (
            failure_supervisor,
            orchestrator,
            raw_semantic_projection,
        )
        from chronovisor.ingest import ingest as ingest_mod

        parent = TestOrchestrator._write_transcript_raw(
            isolated_wiki / "raw",
            name="verified-transcript.md",
            records=[{"line": 1, "role": "user", "text": "remember"}],
        )
        projection = raw_semantic_projection.project_parent_raw(
            parent,
            output_dir=isolated_wiki / "raw",
            max_child_bytes=24_000,
        )
        child = projection.child_paths[0]
        orchestrator.mark_raw_processed([parent.name])
        monkeypatch.setattr(background_jobs, "start_self_heal_background", lambda _path: None)

        recorded = failure_supervisor.record_raw_failure(
            raw_path=child,
            error=(
                "raw semantic projection failed: ProjectionConflictError: "
                "manifest-first crash"
            ),
            raw_text=child.read_text(encoding="utf-8"),
        )

        assert recorded.failure_class == ("ingest.runtime_semantic_projection_failure")
        assert ":projectionconflicterror:" in recorded.fingerprint
        assert orchestrator.get_pending_raw_files() == [child]
        # Dashboard/pending reads are intentionally read-only; successful
        # ingest owns the durable state cleanup.
        state = json.loads(
            (isolated_wiki / "runtime" / "failures" / "state.json").read_text(
                encoding="utf-8"
            )
        )
        assert child.name in state["failures"]

        def succeed(
            _content, _job_id, on_complete=None, on_finally=None, *, metadata=None
        ):
            if on_complete:
                on_complete()

        monkeypatch.setattr(ingest_mod, "run_ingest", succeed)
        monkeypatch.setattr(orchestrator, "is_available", lambda: True)
        processed = orchestrator.run_pending_ingest(force=True)

        assert processed["files_processed"] == [child.name]
        state = json.loads(
            (isolated_wiki / "runtime" / "failures" / "state.json").read_text(
                encoding="utf-8"
            )
        )
        assert child.name not in state["failures"]

    def test_tampered_projection_child_is_never_quarantined_as_source_raw(
        self, isolated_wiki: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from chronovisor.core import background_jobs
        from chronovisor.ingest import ingest as ingest_mod
        from chronovisor.ingest import orchestrator, raw_semantic_projection

        parent = TestOrchestrator._write_transcript_raw(
            isolated_wiki / "raw",
            name="tampered-child-parent.md",
            records=[{"line": 1, "role": "user", "text": "preserved parent"}],
        )
        projection = raw_semantic_projection.project_parent_raw(
            parent,
            output_dir=isolated_wiki / "raw",
            max_child_bytes=24_000,
        )
        child = projection.child_paths[0]
        orchestrator.mark_raw_processed([parent.name])
        child.write_text("tampered derived artifact", encoding="utf-8")
        monkeypatch.setattr(background_jobs, "start_self_heal_background", lambda _path: None)
        monkeypatch.setattr(orchestrator, "is_available", lambda: True)
        monkeypatch.setattr(
            ingest_mod,
            "run_ingest",
            lambda *_args, **_kwargs: pytest.fail(
                "tampered projection child reached model ingest"
            ),
        )

        first = orchestrator.run_pending_ingest(force=True)
        second = orchestrator.run_pending_ingest(force=True)
        third = orchestrator.run_pending_ingest(force=True)

        supervision = first["per_raw"][0]["supervision"]
        assert supervision["failure_class"] == (
            "ingest.runtime_semantic_projection_artifact_conflict"
        )
        assert supervision["quarantined"] is False
        assert second == {"triggered": False, "reason": "no pending raws"}
        assert third == second
        assert parent.exists()
        assert child.exists()
        assert not (isolated_wiki / "raw" / ".dead-letter" / child.name).exists()

    def test_completed_projection_parent_bundle_retries_after_commit_crash(
        self, isolated_wiki: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from chronovisor.core import background_jobs
        from chronovisor.ingest import failure_supervisor, raw_semantic_projection

        parent = TestOrchestrator._write_transcript_raw(
            isolated_wiki / "raw",
            name="completed-parent.md",
            records=[{"line": 1, "role": "user", "text": "remember exactly"}],
        )
        raw_semantic_projection.project_parent_raw(
            parent,
            output_dir=isolated_wiki / "raw",
            max_child_bytes=24_000,
        )
        monkeypatch.setattr(background_jobs, "start_self_heal_background", lambda _path: None)
        recorded = failure_supervisor.record_raw_failure(
            raw_path=parent,
            error=(
                "raw semantic projection failed [internal_error]: "
                "RuntimeError: commit callback failed"
            ),
            raw_text=parent.read_text(encoding="utf-8"),
        )

        assert recorded.failure_class == (
            "ingest.runtime_semantic_projection_internal_error"
        )
        assert raw_semantic_projection.projection_bundle_state_for_parent(parent) == (
            "completed"
        )
        assert failure_supervisor.operational_deferred_raw_files([parent]) == {}

    def test_incomplete_projection_parent_bundle_is_resumable_but_tamper_is_not(
        self, isolated_wiki: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from chronovisor.core import background_jobs
        from chronovisor.ingest import failure_supervisor, raw_semantic_projection

        parent = TestOrchestrator._write_transcript_raw(
            isolated_wiki / "raw",
            name="incomplete-parent.md",
            records=[{"line": 1, "role": "user", "text": "x" * 5_000}],
        )
        original_publish = raw_semantic_projection._atomic_create_or_verify

        def fail_first_child(target: Path, payload: bytes) -> bool:
            if "-child-" in target.name:
                raise RuntimeError("injected publication interruption")
            return original_publish(target, payload)

        monkeypatch.setattr(
            raw_semantic_projection,
            "_atomic_create_or_verify",
            fail_first_child,
        )
        with pytest.raises(RuntimeError, match="injected"):
            raw_semantic_projection.project_parent_raw(
                parent,
                output_dir=isolated_wiki / "raw",
                max_child_bytes=1_400,
            )
        monkeypatch.setattr(background_jobs, "start_self_heal_background", lambda _path: None)
        failure_supervisor.record_raw_failure(
            raw_path=parent,
            error=(
                "raw semantic projection failed [internal_error]: "
                "RuntimeError: injected publication interruption"
            ),
            raw_text=parent.read_text(encoding="utf-8"),
        )

        assert raw_semantic_projection.projection_bundle_state_for_parent(parent) == (
            "incomplete"
        )
        assert failure_supervisor.operational_deferred_raw_files([parent]) == {}

        monkeypatch.setattr(
            raw_semantic_projection,
            "_atomic_create_or_verify",
            original_publish,
        )
        resumed = raw_semantic_projection.project_parent_raw(
            parent,
            output_dir=isolated_wiki / "raw",
            max_child_bytes=3_000,
        )
        resumed.child_paths[0].write_text("tampered", encoding="utf-8")
        assert raw_semantic_projection.projection_bundle_state_for_parent(parent) == (
            "invalid"
        )
        assert failure_supervisor.operational_deferred_raw_files([parent]) != {}

    def test_projection_directory_fsync_failure_keeps_parent_pending_then_resumes(
        self, isolated_wiki: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from chronovisor.ingest import orchestrator, raw_semantic_projection

        parent = TestOrchestrator._write_transcript_raw(
            isolated_wiki / "raw",
            name="projection-fsync-parent.md",
            records=[{"line": 1, "role": "user", "text": "durable memory"}],
        )
        original_fsync_directory = raw_semantic_projection._fsync_directory

        def fail_directory_sync(_path: Path) -> None:
            raise OSError("injected projection directory fsync failure")

        monkeypatch.setattr(
            raw_semantic_projection,
            "_fsync_directory",
            fail_directory_sync,
        )
        monkeypatch.setattr(orchestrator, "is_available", lambda: False)

        failed = orchestrator.run_pending_ingest(force=True)

        assert failed["files_processed"] == []
        assert failed["per_raw"][0]["succeeded"] is False
        assert failed["per_raw"][0]["supervision"]["failure_class"] == (
            "ingest.runtime_semantic_projection_interrupted"
        )
        assert parent.name not in orchestrator._load_state()["processed_raw_files"]
        assert parent in orchestrator.get_pending_raw_files()

        monkeypatch.setattr(
            raw_semantic_projection,
            "_fsync_directory",
            original_fsync_directory,
        )
        resumed = orchestrator.run_pending_ingest(force=True)

        assert resumed["files_processed"] == [parent.name]
        assert parent.name in orchestrator._load_state()["processed_raw_files"]
        assert raw_semantic_projection.projection_bundle_state_for_parent(parent) == (
            "completed"
        )

    def test_projection_internal_failure_without_manifest_stays_deferred(
        self, isolated_wiki: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from chronovisor.core import background_jobs
        from chronovisor.ingest import failure_supervisor

        parent = isolated_wiki / "raw" / "no-manifest-parent.md"
        parent.write_text("valid source with no projection intent", encoding="utf-8")
        monkeypatch.setattr(background_jobs, "start_self_heal_background", lambda _path: None)
        failure_supervisor.record_raw_failure(
            raw_path=parent,
            error=(
                "raw semantic projection failed [internal_error]: "
                "RuntimeError: invariant broken"
            ),
            raw_text=parent.read_text(encoding="utf-8"),
        )

        assert failure_supervisor.operational_deferred_raw_files([parent]) == {
            parent.name: "pending_local_repair"
        }

    def test_deferred_snapshot_cannot_overwrite_concurrent_failure_writer(
        self, isolated_wiki: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import threading

        from chronovisor.ingest import failure_supervisor

        deferred_raw = isolated_wiki / "raw" / "deferred.md"
        writer_raw = isolated_wiki / "raw" / "writer.md"
        deferred_raw.write_text("deferred", encoding="utf-8")
        writer_raw.write_text("writer", encoding="utf-8")
        state_path = isolated_wiki / "runtime" / "failures" / "state.json"
        state_path.parent.mkdir(parents=True, exist_ok=True)
        state_path.write_text(
            json.dumps(
                {
                    "failures": {
                        deferred_raw.name: {
                            "fingerprint": "ingest.runtime_schema_invalid",
                            "failure_class": "ingest.runtime_schema_invalid",
                            "self_heal_queued": True,
                            "packet_path": None,
                        }
                    }
                }
            ),
            encoding="utf-8",
        )
        original_load = failure_supervisor._load_state
        snapshot_loaded = threading.Event()
        release_reader = threading.Event()

        def slow_load():
            snapshot = original_load()
            if threading.current_thread().name == "deferred-reader":
                snapshot_loaded.set()
                assert release_reader.wait(timeout=5)
            return snapshot

        monkeypatch.setattr(failure_supervisor, "_load_state", slow_load)
        reader = threading.Thread(
            target=failure_supervisor.operational_deferred_raw_files,
            args=([deferred_raw],),
            name="deferred-reader",
        )
        writer = threading.Thread(
            target=failure_supervisor.record_raw_failure,
            kwargs={
                "raw_path": writer_raw,
                "error": "independent deterministic failure",
                "raw_text": "writer",
            },
            name="failure-writer",
        )

        reader.start()
        assert snapshot_loaded.wait(timeout=5)
        writer.start()
        assert writer.is_alive()
        release_reader.set()
        reader.join(timeout=5)
        writer.join(timeout=5)

        assert not reader.is_alive()
        assert not writer.is_alive()
        state = original_load()
        assert deferred_raw.name in state["failures"]
        assert writer_raw.name in state["failures"]

    @pytest.mark.parametrize(
        "packet_state",
        [
            "missing",
            "invalid",
            "pending_local_repair",
            "local_quarantined",
            "frontier_quarantined",
            "human_required",
        ],
    )
    def test_operational_packet_state_is_fail_closed_deferred(
        self,
        isolated_wiki: Path,
        monkeypatch: pytest.MonkeyPatch,
        packet_state: str,
    ) -> None:
        from chronovisor.core import background_jobs
        from chronovisor.ingest import failure_supervisor

        raw_path = isolated_wiki / "raw" / "deferred.md"
        raw_path.write_text("valid source", encoding="utf-8")
        monkeypatch.setattr(background_jobs, "start_self_heal_background", lambda _path: None)
        result = failure_supervisor.record_raw_failure(
            raw_path=raw_path,
            error="triage structured failure [schema_invalid]: malformed schema",
            raw_text="valid source",
        )
        assert result.packet_path is not None
        packet_path = Path(result.packet_path)
        if packet_state == "missing":
            packet_path.unlink()
            expected = "packet_missing"
        elif packet_state == "invalid":
            packet_path.write_text("not-json", encoding="utf-8")
            expected = "packet_invalid"
        else:
            packet = json.loads(packet_path.read_text(encoding="utf-8"))
            packet["status"] = packet_state
            packet_path.write_text(json.dumps(packet), encoding="utf-8")
            expected = packet_state

        assert failure_supervisor.operational_deferred_raw_files([raw_path]) == {
            raw_path.name: expected
        }
        assert raw_path.exists()

    def test_serial_execution_no_concurrent_threads(
        self, isolated_wiki: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The Phase 2 design replaces ``start_ingest`` (which spawned
        a worker thread) with a synchronous ``run_ingest`` call. We
        verify by counting the active thread delta around the batch:
        no per-raw worker thread should be spawned.
        """
        import threading

        from chronovisor.ingest import ingest as ingest_mod
        from chronovisor.ingest import orchestrator

        for i in range(3):
            (isolated_wiki / "raw" / f"r{i}.md").write_text("body")

        thread_counts: list[int] = []

        def fake_run_ingest(
            content, job_id, on_complete=None, on_finally=None, *, metadata=None
        ):
            # Sample active thread count INSIDE each "ingest" invocation.
            thread_counts.append(threading.active_count())
            if on_complete:
                on_complete()
            if on_finally:
                on_finally(failed=False, triage_failed=False)

        monkeypatch.setattr(ingest_mod, "run_ingest", fake_run_ingest)
        monkeypatch.setattr(orchestrator, "is_available", lambda: True)

        baseline = threading.active_count()
        orchestrator.run_pending_ingest(force=True)

        # Every per-raw call ran on the same thread as the batch driver
        # (no Thread() per raw). Allow +/- 1 for unrelated background
        # threads, but assert no growth across calls.
        assert all(c <= baseline + 1 for c in thread_counts), thread_counts


# ---------------------------------------------------------------------------
# Phase 6: signature backward compatibility + field-naming independence
# ---------------------------------------------------------------------------


class TestPhase6Compatibility:
    def test_run_ingest_positional_signature_still_works(
        self, isolated_wiki: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The Phase 3 keyword-only ``metadata`` parameter must not
        break callers that still pass the original 4 positional args."""
        from chronovisor.core import jobs
        from chronovisor.ingest import ingest

        plan = [{"type": "create", "filename": "misc/p.md", "title": "P"}]
        monkeypatch.setattr(ingest, "_triage", lambda _content: plan)
        monkeypatch.setattr(
            ingest,
            "_generate_one",
            lambda _op, _raw, **_kw: {
                "type": "create",
                "filename": "misc/p.md",
                "content": "---\ntitle: P\nupdated: 2026-04-28\n---\nbody",
            },
        )
        monkeypatch.setattr(ingest, "is_available", lambda: True)

        on_complete_called = []
        on_finally_called = []
        job = jobs.job_store.create(processor="ollama")

        # Original 4-positional call shape — no metadata kwarg.
        ingest.run_ingest(
            "raw",
            job.job_id,
            lambda: on_complete_called.append(True),
            lambda failed, triage_failed: on_finally_called.append(
                (failed, triage_failed)
            ),
            frontier_reviewer=ingest._run_ingest_frontier_review,
        )

        finished = jobs.job_store.get(job.job_id)
        assert finished.status == jobs.JobStatus.COMPLETED
        assert finished.pages_created == ["p"]
        assert on_complete_called == [True]
        assert on_finally_called == [(False, False)]

    def test_triage_keywords_field_independent_of_raw_keywords(
        self, isolated_wiki: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The triage plan dict can carry its own ``keywords`` field for
        related-page search (``ingest._build_focused_context``) without
        being conflated with the raw frontmatter's ``raw_keywords``.
        Verify by feeding a triage plan with ``keywords`` AND a metadata
        ``raw_keywords`` — they must reach the operation as two distinct
        signals."""
        from chronovisor.core import jobs
        from chronovisor.ingest import ingest

        plan = [
            {
                "type": "create",
                "filename": "misc/p.md",
                "title": "P",
                "keywords": ["search-only-1", "search-only-2"],
            }
        ]
        monkeypatch.setattr(ingest, "_triage", lambda _content: plan)

        captured_op: dict = {}
        captured_kw: list[list[str] | None] = []

        def stub_generate(op, _raw, *, raw_keywords=None):
            # The op coming into generate still has its triage-side
            # ``keywords`` field — that's the search-related signal.
            captured_op.update(op)
            captured_kw.append(raw_keywords)
            return {
                "type": "create",
                "filename": op["filename"],
                "content": "---\ntitle: P\nupdated: 2026-04-28\n---\nbody",
            }

        monkeypatch.setattr(ingest, "_generate_one", stub_generate)
        monkeypatch.setattr(ingest, "is_available", lambda: True)

        job = jobs.job_store.create(processor="ollama")
        ingest.run_ingest(
            "raw",
            job.job_id,
            metadata={"raw_keywords": ["fm-only"]},
            frontier_reviewer=ingest._run_ingest_frontier_review,
        )

        # Triage's ``keywords`` survived on the op (used downstream by
        # _build_focused_context), distinct from raw_keywords.
        assert captured_op.get("keywords") == ["search-only-1", "search-only-2"]
        # raw_keywords reached generate as the dedicated metadata channel,
        # NOT mixed into the triage keywords list.
        assert captured_kw == [["fm-only"]]


class TestSearchBeforeCreate:
    def test_create_with_existing_title_is_rewritten_to_update(
        self, isolated_wiki: Path
    ) -> None:
        from chronovisor.ingest import ingest

        _seed_page(
            isolated_wiki,
            "work/workplace-stress-and-mentoring.md",
            "---\n"
            "title: Workplace Stress and Mentoring\n"
            "updated: 2026-07-01\n"
            "tags: [d/career, t/analysis, s/2026]\n"
            "---\n"
            "Existing body.\n",
        )
        plan = [
            {
                "type": "create",
                "filename": "work/workplace-stress-mentoring.md",
                "title": "Workplace Stress and Mentoring",
                "summary": "Mentoring and workplace stress notes.",
            }
        ]

        rewritten = ingest._dedupe_create_ops_with_existing(plan, "raw")

        assert rewritten[0]["type"] == "update"
        assert rewritten[0]["filename"] == "work/workplace-stress-and-mentoring.md"
        assert rewritten[0]["existing_page_id"] == "workplace-stress-and-mentoring"


class TestReadBackVerification:
    def test_changed_page_passes_when_search_returns_page(
        self, isolated_wiki: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from chronovisor.core import index_store, search
        from chronovisor.core.search_types import ScoredPage
        from chronovisor.ingest import ingest_readback

        class Store:
            def refresh(self) -> None:
                pass

            def meta(self, page_id: str) -> dict:
                return {
                    "page_id": page_id,
                    "title": "P",
                    "updated": "2026-07-06",
                    "path": str(isolated_wiki / "pages" / "p.md"),
                    "summary": "Durable test summary",
                    "recall_questions": ["How do we recall P?"],
                }

        monkeypatch.setattr(index_store, "get_store", lambda: Store())
        monkeypatch.setattr(
            search,
            "search",
            lambda _query, top_n=10, semantic=True: (
                [ScoredPage("p", "P", "", "2026-07-06", 1.0)],
                "hybrid",
            ),
        )

        result = ingest_readback.verify_changed_pages_read_back(["p"])

        assert result == {"checked": 1, "passed": 1, "failed": []}

    def test_verify_resolves_query_and_log_paths_through_ingest_facade(
        self, isolated_wiki: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from chronovisor.core import index_store, search
        from chronovisor.ingest import ingest, ingest_readback

        queries: list[tuple[dict, str]] = []
        run_log = isolated_wiki / "runtime" / "patched-runs.jsonl"
        failure_log = isolated_wiki / "runtime" / "patched-failures.jsonl"

        class Store:
            def refresh(self) -> None:
                pass

            def meta(self, page_id: str) -> dict:
                return {"page_id": page_id, "title": "ignored"}

        def query(meta: dict, page_id: str) -> str:
            queries.append((meta, page_id))
            return "patched query"

        monkeypatch.setattr(index_store, "get_store", lambda: Store())
        monkeypatch.setattr(search, "search", lambda *_a, **_k: ([], "hybrid"))
        monkeypatch.setattr(ingest_readback, "_read_back_query", query)
        monkeypatch.setattr(ingest_readback, "_read_back_run_log", lambda: run_log)
        monkeypatch.setattr(
            ingest_readback, "_read_back_failure_log", lambda: failure_log
        )
        monkeypatch.setattr(ingest, "_safe_log", lambda *_a, **_k: None)

        result = ingest_readback.verify_changed_pages_read_back(["p"])

        assert queries == [({"page_id": "p", "title": "ignored"}, "p")]
        assert result["failed"][0]["query"] == "patched query"
        assert run_log.exists()
        assert failure_log.exists()

    def test_refresh_waits_for_semantic_delta_before_read_back(
        self, isolated_wiki: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from chronovisor.core import claims, index_store, search
        from chronovisor.ingest import ingest, ingest_readback, state_register

        events: list[object] = []

        class Store:
            def refresh(self) -> None:
                events.append("store-refresh")

        def update_embeddings(page_ids=None, *, strict=False):
            events.append(("semantic-index", list(page_ids or ()), strict))
            return len(page_ids or ())

        def read_back(page_ids, *, top_n=10):
            events.append(("read-back", list(page_ids), top_n))
            return {"checked": 1, "passed": 1, "failed": []}

        monkeypatch.setattr(ingest, "_rebuild_index", lambda: events.append("index"))
        monkeypatch.setattr(index_store, "get_store", lambda: Store())
        monkeypatch.setattr(search, "update_embeddings", update_embeddings)
        monkeypatch.setattr(
            claims,
            "append_page_claims",
            lambda page_ids, **kwargs: events.append(
                ("claims", list(page_ids), kwargs["source_raw"], kwargs["op"])
            ),
        )
        monkeypatch.setattr(
            state_register,
            "refresh_state_register",
            lambda page_ids, **kwargs: events.append(
                ("state", list(page_ids), kwargs["source_raw"])
            ),
        )
        monkeypatch.setattr(
            ingest_readback, "verify_changed_pages_read_back", read_back
        )

        result = ingest_readback._refresh_ingest_derived_artifacts(
            ["p"],
            source_raw="raw.md",
        )

        assert result == {"checked": 1, "passed": 1, "failed": []}
        assert events == [
            "index",
            "store-refresh",
            ("semantic-index", ["p"], True),
            ("claims", ["p"], "raw.md", "ingest"),
            ("state", ["p"], "raw.md"),
            ("read-back", ["p"], 10),
        ]

    def test_refresh_failures_are_logged_and_nonfatal(
        self, isolated_wiki: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from chronovisor.core import claims, index_store, search
        from chronovisor.ingest import ingest, ingest_readback, state_register

        events: list[str] = []
        logs: list[str] = []

        def fail(stage: str) -> None:
            events.append(stage)
            raise RuntimeError(stage)

        class Store:
            def refresh(self) -> None:
                fail("store")

        def read_back(page_ids, *, top_n=10):
            events.append("read-back")
            return {"checked": len(page_ids), "passed": len(page_ids), "failed": []}

        monkeypatch.setattr(ingest, "_rebuild_index", lambda: fail("index"))
        monkeypatch.setattr(index_store, "get_store", lambda: Store())
        monkeypatch.setattr(search, "update_embeddings", lambda **_kwargs: fail("semantic"))
        monkeypatch.setattr(claims, "append_page_claims", lambda *_a, **_k: fail("claims"))
        monkeypatch.setattr(
            state_register,
            "refresh_state_register",
            lambda *_a, **_k: fail("state"),
        )
        monkeypatch.setattr(ingest, "_safe_log", logs.append)
        monkeypatch.setattr(
            ingest_readback, "verify_changed_pages_read_back", read_back
        )

        result = ingest_readback._refresh_ingest_derived_artifacts(
            ["p"], source_raw="raw.md"
        )

        assert result == {"checked": 1, "passed": 1, "failed": []}
        assert events == ["index", "store", "semantic", "claims", "state", "read-back"]
        assert logs == [
            "ingest | index.md rebuild failed (non-fatal): index",
            "ingest | index_store refresh failed: store",
            "ingest | semantic index enqueue failed: semantic",
            "ingest | claim ledger failed (non-fatal): claims",
            "ingest | state register refresh failed (non-fatal): state",
        ]


# ---------------------------------------------------------------------------
# Triage plan schema validation (R3-High)
# ---------------------------------------------------------------------------


class _QueueStructuredTransport:
    def __init__(self, *responses: str | Exception) -> None:
        self.responses = list(responses)
        self.requests: list[object] = []

    def __call__(self, request):
        self.requests.append(request)
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response


class TestTriagePlanSchema:
    def test_ollama_grammar_schema_uses_host_side_numeric_bounds(self) -> None:
        from chronovisor.ingest.ingest import (
            _TRIAGE_PLAN_VALIDATION_SCHEMA,
            TRIAGE_PLAN_SCHEMA,
        )

        encoded = json.dumps(TRIAGE_PLAN_SCHEMA, sort_keys=True)
        assert TRIAGE_PLAN_SCHEMA["items"]["additionalProperties"] is False
        assert TRIAGE_PLAN_SCHEMA["items"]["required"] == [
            "type",
            "filename",
            "title",
            "keywords",
            "summary",
        ]
        for repetition_bound in ("minItems", "maxItems", "minLength", "maxLength"):
            assert repetition_bound not in encoded
        assert "at most 8 operations" in ollama.TRIAGE_SYSTEM_PROMPT
        assert "filename to 200 characters" in ollama.TRIAGE_SYSTEM_PROMPT
        assert "title to 300" in ollama.TRIAGE_SYSTEM_PROMPT
        assert "summary to 2000" in ollama.TRIAGE_SYSTEM_PROMPT
        assert "1 to 32 keywords" in ollama.TRIAGE_SYSTEM_PROMPT
        assert "each at most 200 characters" in ollama.TRIAGE_SYSTEM_PROMPT
        assert _TRIAGE_PLAN_VALIDATION_SCHEMA["maxItems"] == 8
        properties = _TRIAGE_PLAN_VALIDATION_SCHEMA["items"]["properties"]
        assert properties["filename"]["maxLength"] == 200
        assert properties["title"]["maxLength"] == 300
        assert properties["keywords"]["maxItems"] == 32
        assert properties["keywords"]["items"]["maxLength"] == 200
        assert properties["summary"]["maxLength"] == 2_000

    def test_valid_plan_passes_through(self) -> None:
        from chronovisor.ingest.ingest import _validate_triage_plan

        plan = [
            {
                "type": "create",
                "filename": "ai/foo.md",
                "title": "Foo",
                "keywords": ["foo"],
                "summary": "Foo knowledge",
            },
            {"type": "update", "filename": "bar.md"},
        ]
        assert _validate_triage_plan(plan) == plan

    def test_bare_create_is_rejected_with_folder_repair_contract(self) -> None:
        from chronovisor.ingest.ingest import _triage_plan_validation_issues

        operation = {
            "type": "create",
            "filename": "root-page.md",
            "title": "Root Page",
            "keywords": ["root", "page"],
            "summary": "A durable page that needs classification.",
        }

        issues = _triage_plan_validation_issues([operation])

        assert len(issues) == 1
        assert issues[0].pointer == "/0/filename"
        assert issues[0].keyword == "createPathDepth"
        assert issues[0].expected["format"] == "folder/page-id.md"

    def test_live_triage_repairs_bare_create_into_existing_folder(
        self, isolated_wiki: Path
    ) -> None:
        from chronovisor.ingest import ingest

        _seed_page(
            isolated_wiki,
            "ai/existing.md",
            "---\ntitle: Existing\nupdated: 2026-07-18\n---\nbody\n",
        )
        invalid = {
            "type": "create",
            "filename": "new-topic.md",
            "title": "New Topic",
            "keywords": ["new", "topic"],
            "summary": "Durable AI knowledge.",
        }
        repaired = {**invalid, "filename": "ai/new-topic.md"}
        transport = _QueueStructuredTransport(
            json.dumps([invalid]), json.dumps([repaired])
        )

        assert ingest._triage("AI topic", transport=transport) == [repaired]
        assert len(transport.requests) == 2
        initial_prompt = transport.requests[0].messages[-1]["content"]
        assert "Existing top-level folders" in initial_prompt
        assert "ai/" in initial_prompt
        assert "Every create filename must be folder/page.md" in initial_prompt
        feedback = transport.requests[1].messages[-1]["content"]
        assert '"keyword":"createPathDepth"' in feedback

    @pytest.mark.parametrize(
        ("feedback_bytes", "expect_repair"),
        [(4_025, True), (4_097, False)],
    )
    def test_live_triage_uses_exact_4kib_feedback_byte_cap(
        self,
        isolated_wiki: Path,
        monkeypatch: pytest.MonkeyPatch,
        feedback_bytes: int,
        expect_repair: bool,
    ) -> None:
        from chronovisor.decision import local_structured
        from chronovisor.ingest import ingest

        invalid = [
            {
                "type": "update",
                "filename": "missing.md",
                "title": "Missing page",
                "keywords": ["missing", "repair"],
                "summary": "This schema-valid update needs a foldered create.",
            }
        ]
        repaired = [
            {
                "type": "create",
                "filename": "memory/repaired-feedback.md",
                "title": "Repaired feedback",
                "keywords": ["repair", "feedback"],
                "summary": "The exact validator feedback reached repair.",
            }
        ]

        def issue_with_message(message: str) -> object:
            return ingest.ValidationIssue(
                pointer="/0/filename",
                keyword="exactFeedback",
                expected="a grounded foldered create",
                received={"type": "string", "value": "missing.md"},
                message=message,
            )

        def render_feedback(issue: object) -> str:
            errors = json.dumps(
                [issue.to_dict()],
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            )
            return local_structured._REPAIR_TEMPLATE.format(errors=errors)

        base_feedback = render_feedback(issue_with_message(""))
        filler = "F" * (feedback_bytes - len(base_feedback.encode("utf-8")))
        issue = issue_with_message(filler)
        expected_feedback = render_feedback(issue)
        assert len(expected_feedback.encode("utf-8")) == feedback_bytes

        def validation_issues(value, **_kwargs):
            return [issue] if value == invalid else []

        monkeypatch.setattr(
            ingest,
            "_triage_plan_validation_issues",
            validation_issues,
        )
        responses = [json.dumps(invalid)]
        if expect_repair:
            responses.append(json.dumps(repaired))
        transport = _QueueStructuredTransport(*responses)

        if expect_repair:
            assert ingest._triage("raw content", transport=transport) == repaired
            assert len(transport.requests) == 2
            repair_feedback = transport.requests[1].messages[-1]["content"]
            assert repair_feedback == expected_feedback
            assert filler in repair_feedback
        else:
            with pytest.raises(ingest.IngestTriageFailure) as raised:
                ingest._triage(
                    "raw content",
                    transport=transport,
                    raise_on_failure=True,
                )
            assert raised.value.failure_class == "feedback_too_large"
            assert len(transport.requests) == 1

    def test_live_triage_repairs_missing_bare_update_into_foldered_create(
        self, isolated_wiki: Path
    ) -> None:
        from chronovisor.ingest import ingest

        invalid = {
            "type": "update",
            "filename": "new-topic.md",
            "title": "New Topic",
            "keywords": ["new", "topic"],
            "summary": "Durable knowledge without an existing page.",
        }
        repaired = {
            **invalid,
            "type": "create",
            "filename": "new-domain/new-topic.md",
        }
        transport = _QueueStructuredTransport(
            json.dumps([invalid]), json.dumps([repaired])
        )

        assert ingest._triage("new domain knowledge", transport=transport) == [repaired]
        feedback = transport.requests[1].messages[-1]["content"]
        assert '"keyword":"missingUpdateNeedsFolderedCreate"' in feedback

    def test_live_triage_repairs_missing_legacy_style_update_before_post_validation(
        self, isolated_wiki: Path
    ) -> None:
        from chronovisor.ingest import ingest

        invalid = {
            "type": "update",
            "filename": "Legacy Folder/New Topic.md",
            "title": "New Topic",
            "keywords": ["new", "topic"],
            "summary": "Durable knowledge without an existing legacy target.",
        }
        repaired = {
            **invalid,
            "type": "create",
            "filename": "new-domain/new-topic.md",
        }
        transport = _QueueStructuredTransport(
            json.dumps([invalid]),
            json.dumps([repaired]),
        )

        assert ingest._triage("new domain knowledge", transport=transport) == [repaired]
        assert len(transport.requests) == 2
        feedback = transport.requests[1].messages[-1]["content"]
        assert '"keyword":"missingUpdateNeedsFolderedCreate"' in feedback
        assert "valid ASCII kebab-case filename" in feedback

    def test_more_than_8_operations_is_rejected_by_host_validator(self) -> None:
        from chronovisor.ingest.ingest import _triage_plan_validation_issues

        plan = [
            {"type": "update", "filename": f"page-{index}.md"} for index in range(9)
        ]

        issues = _triage_plan_validation_issues(plan)

        assert [(issue.pointer, issue.keyword, issue.expected) for issue in issues] == [
            ("", "maxItems", 8)
        ]

    def test_host_validator_enforces_the_session_output_budget(self) -> None:
        from chronovisor.ingest.ingest import _triage_plan_validation_issues

        plan = [
            {
                "type": "update",
                "filename": f"page-{index}.md",
                "title": f"Page {index}",
                "keywords": ["page"],
                "summary": "x" * 1_000,
            }
            for index in range(8)
        ]

        issues = _triage_plan_validation_issues(plan)

        assert [(issue.pointer, issue.keyword, issue.expected) for issue in issues] == [
            ("", "maxUtf8Bytes", 8_000)
        ]

    @pytest.mark.parametrize(
        ("field_patch", "pointer", "keyword", "expected"),
        [
            ({"filename": ""}, "/0/filename", "minLength", 1),
            ({"filename": "x" * 201}, "/0/filename", "maxLength", 200),
            ({"title": ""}, "/0/title", "minLength", 1),
            ({"title": "x" * 301}, "/0/title", "maxLength", 300),
            ({"keywords": []}, "/0/keywords", "minItems", 1),
            ({"keywords": ["x"] * 33}, "/0/keywords", "maxItems", 32),
            ({"keywords": [""]}, "/0/keywords/0", "minLength", 1),
            ({"keywords": ["x" * 201]}, "/0/keywords/0", "maxLength", 200),
            ({"summary": ""}, "/0/summary", "minLength", 1),
            ({"summary": "x" * 2_001}, "/0/summary", "maxLength", 2_000),
        ],
    )
    def test_host_validator_preserves_ollama_omitted_numeric_bound(
        self,
        field_patch: dict,
        pointer: str,
        keyword: str,
        expected: int,
    ) -> None:
        from chronovisor.ingest.ingest import _triage_plan_validation_issues

        operation = {"type": "update", "filename": "bounded.md", **field_patch}
        issues = _triage_plan_validation_issues([operation])

        assert any(
            issue.pointer == pointer
            and issue.keyword == keyword
            and issue.expected == expected
            for issue in issues
        )

    @pytest.mark.parametrize(
        ("field_patch", "pointer"),
        [
            ({"filename": "   "}, "/0/filename"),
            ({"title": "   "}, "/0/title"),
            ({"summary": "   "}, "/0/summary"),
            ({"keywords": ["   "]}, "/0/keywords/0"),
        ],
    )
    def test_host_validator_rejects_whitespace_only_fields(
        self, field_patch: dict, pointer: str
    ) -> None:
        from chronovisor.ingest.ingest import _triage_plan_validation_issues

        operation = {
            "type": "update",
            "filename": "bounded.md",
            "title": "Bounded",
            "keywords": ["bounded"],
            "summary": "Update bounded knowledge.",
            **field_patch,
        }

        issues = _triage_plan_validation_issues([operation])

        assert any(issue.pointer == pointer for issue in issues)

    @pytest.mark.parametrize(
        "filename",
        [
            "lessons-learned.md",
            "memory/LESSONS-LEARNED.md",
            "current-state",
            "system/user-profile.md",
        ],
    )
    def test_host_validator_rejects_reserved_system_target(
        self,
        isolated_wiki: Path,
        filename: str,
    ) -> None:
        from chronovisor.ingest import ingest

        issues = ingest._triage_plan_validation_issues(
            [
                {
                    "type": "update",
                    "filename": filename,
                    "title": "Reserved",
                    "keywords": ["reserved"],
                    "summary": "This must be retargeted from normal ingest.",
                }
            ]
        )

        reserved = [
            issue for issue in issues if issue.keyword == "reservedSystemTarget"
        ]
        assert len(reserved) == 1
        assert reserved[0].pointer == "/0/filename"
        assert "preserve every unrelated valid operation" in str(reserved[0].expected)

    def test_host_validator_reserves_every_installed_system_page(
        self,
        isolated_wiki: Path,
    ) -> None:
        from chronovisor.ingest import ingest

        (isolated_wiki / "system" / "claude-code.md").write_text(
            "---\ntitle: Claude Code\n---\noperational system memory\n",
            encoding="utf-8",
        )

        issues = ingest._triage_plan_validation_issues(
            [
                {
                    "type": "update",
                    "filename": "memory/CLAUDE-CODE.md",
                    "title": "Claude Code",
                    "keywords": ["claude-code"],
                    "summary": "An installed operational system page stays reserved.",
                }
            ]
        )

        assert [issue.keyword for issue in issues] == ["reservedSystemTarget"]

    def test_live_triage_repairs_reserved_only_plan_to_noop_in_same_session(
        self,
        isolated_wiki: Path,
    ) -> None:
        from chronovisor.ingest import ingest

        invalid = [
            {
                "type": "update",
                "filename": "lessons-learned.md",
                "title": "Lessons learned",
                "keywords": ["lessons-learned"],
                "summary": "Reserved system memory is not a normal ingest target.",
            }
        ]
        transport = _QueueStructuredTransport(json.dumps(invalid), "[]")

        assert ingest._triage("raw content", transport=transport) == []
        assert len(transport.requests) == 2
        feedback = transport.requests[1].messages[-1]["content"]
        assert '"keyword":"reservedSystemTarget"' in feedback
        assert '"pointer":"/0/filename"' in feedback
        assert "normal knowledge page" in feedback

    def test_live_triage_regenerates_complete_mixed_reserved_plan(
        self,
        isolated_wiki: Path,
    ) -> None:
        from chronovisor.ingest import ingest

        valid_unrelated = {
            "type": "create",
            "filename": "memory/valid-unrelated.md",
            "title": "Valid unrelated",
            "keywords": ["valid", "unrelated"],
            "summary": "A grounded normal knowledge-page operation.",
        }
        invalid_reserved = {
            "type": "update",
            "filename": "lessons-learned.md",
            "title": "Lessons learned",
            "keywords": ["lessons-learned"],
            "summary": "A grounded fact that needs a normal page target.",
        }
        repaired_reserved = {
            **invalid_reserved,
            "type": "create",
            "filename": "memory/grounded-lesson.md",
            "title": "Grounded lesson",
        }
        repaired = [valid_unrelated, repaired_reserved]
        transport = _QueueStructuredTransport(
            json.dumps([valid_unrelated, invalid_reserved]),
            json.dumps(repaired),
        )

        assert ingest._triage("raw content", transport=transport) == repaired
        assert len(transport.requests) == 2
        repair_request = transport.requests[1]
        assert repair_request.messages[-2] == {
            "role": "assistant",
            "content": json.dumps([valid_unrelated, invalid_reserved]),
        }
        feedback = repair_request.messages[-1]["content"]
        assert '"keyword":"reservedSystemTarget"' in feedback
        assert "do not drop or alter unrelated valid operations" in feedback

    def test_live_triage_repairs_host_bounded_oversized_plan(
        self, isolated_wiki: Path
    ) -> None:
        from chronovisor.ingest import ingest

        invalid = [
            {
                "type": "update",
                "filename": f"page-{index}.md",
                "title": f"Page {index}",
                "keywords": ["page"],
                "summary": "Update the page.",
            }
            for index in range(9)
        ]
        transport = _QueueStructuredTransport(json.dumps(invalid), "[]")

        assert ingest._triage("raw content", transport=transport) == []
        assert len(transport.requests) == 2
        feedback = transport.requests[1].messages[-1]["content"]
        assert '"keyword":"maxItems"' in feedback
        assert '"pointer":""' in feedback

    def test_live_triage_repairs_whitespace_only_field(
        self, isolated_wiki: Path
    ) -> None:
        from chronovisor.ingest import ingest

        _seed_page(
            isolated_wiki,
            "bounded.md",
            "---\ntitle: Bounded\nupdated: 2026-01-01\n---\nExisting body.",
        )

        invalid = [
            {
                "type": "update",
                "filename": "bounded.md",
                "title": "   ",
                "keywords": ["bounded"],
                "summary": "Update bounded knowledge.",
            }
        ]
        valid = [{**invalid[0], "title": "Bounded"}]
        transport = _QueueStructuredTransport(json.dumps(invalid), json.dumps(valid))

        assert ingest._triage("raw content", transport=transport) == valid
        feedback = transport.requests[1].messages[-1]["content"]
        assert '"pointer":"/0/title"' in feedback

    def test_live_triage_missing_update_is_retyped_to_create(
        self, isolated_wiki: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from chronovisor.ingest import ingest

        plan = [
            {
                "type": "update",
                "filename": "career/career-transition-strategy-2026.md",
                "title": "Career Transition Strategy 2026",
                "keywords": ["career", "transition", "strategy", "2026"],
                "summary": "Career transition strategy memory",
            }
        ]
        monkeypatch.setattr(
            ingest,
            "_generate_with_progress",
            lambda *_args, **_kwargs: json.dumps(plan),
        )

        out = ingest._triage("raw content")

        assert out == [
            {
                "type": "create",
                "filename": "career/career-transition-strategy-2026.md",
                "summary": "Career transition strategy memory",
                "title": "Career Transition Strategy 2026",
                "keywords": ["career", "transition", "strategy", "2026"],
            }
        ]

    def test_live_triage_existing_update_stays_update(
        self, isolated_wiki: Path
    ) -> None:
        from chronovisor.ingest.ingest import _validate_triage_plan

        _seed_page(
            isolated_wiki,
            "career-transition-strategy-2026.md",
            "---\ntitle: Career\nupdated: 2026-01-01\n---\nold",
        )
        plan = [
            {
                "type": "update",
                "filename": "career-transition-strategy-2026.md",
                "summary": "Append new memory",
            }
        ]

        assert _validate_triage_plan(plan, coerce_missing_updates=True) == plan

    def test_missing_update_without_summary_gets_neutral_create_topic(
        self, isolated_wiki: Path
    ) -> None:
        from chronovisor.ingest.ingest import _validate_triage_plan

        out = _validate_triage_plan(
            [{"type": "update", "filename": "memory/missing-topic.md"}],
            coerce_missing_updates=True,
        )

        assert out == [
            {
                "type": "create",
                "filename": "memory/missing-topic.md",
                "title": "Missing Topic",
                "summary": "Missing Topic",
                "keywords": ["missing", "topic"],
            }
        ]

    def test_live_triage_repairs_malformed_json_in_same_session(
        self, isolated_wiki: Path
    ) -> None:
        from chronovisor.ingest import ingest

        transport = _QueueStructuredTransport(
            '[{"type":"create","filename":',
            json.dumps(
                [
                    {
                        "type": "create",
                        "filename": "memory/repaired.md",
                        "title": "Repaired",
                        "keywords": ["repaired"],
                        "summary": "Durable repaired plan",
                    }
                ]
            ),
        )

        out = ingest._triage("raw content", transport=transport)

        assert out == [
            {
                "type": "create",
                "filename": "memory/repaired.md",
                "title": "Repaired",
                "keywords": ["repaired"],
                "summary": "Durable repaired plan",
            }
        ]
        assert len(transport.requests) == 2
        second = transport.requests[1]
        assert [message["role"] for message in second.messages] == [
            "system",
            "user",
            "assistant",
            "user",
        ]
        assert '"keyword":"parse"' in second.messages[-1]["content"]

    def test_live_triage_repairs_unknown_keys_and_missing_create_fields(
        self, isolated_wiki: Path
    ) -> None:
        from chronovisor.ingest import ingest

        invalid = [
            {
                "type": "create",
                "filename": "memory/repaired-fields.md",
                "title": "Repaired fields",
                "keywords: [": ["bad-key"],
                'summary": ': {"summary": "nested under a malformed key"},
            }
        ]
        valid = [
            {
                "type": "create",
                "filename": "memory/repaired-fields.md",
                "title": "Repaired fields",
                "keywords": ["repaired", "fields"],
                "summary": "Known fields restored after exact validator feedback.",
            }
        ]
        transport = _QueueStructuredTransport(
            json.dumps(invalid),
            json.dumps(valid),
        )

        out = ingest._triage("raw content", transport=transport)

        assert out == valid
        assert len(transport.requests) == 2
        repair_request = transport.requests[1]
        assert repair_request.messages[-2] == {
            "role": "assistant",
            "content": json.dumps(invalid),
        }
        assert "Validator errors" in repair_request.messages[-1]["content"]
        assert "additionalProperties" in repair_request.messages[-1]["content"]
        assert "keywords: [" in repair_request.messages[-1]["content"]

    def test_live_triage_repairs_schema_valid_missing_create_fields(
        self, isolated_wiki: Path
    ) -> None:
        from chronovisor.ingest import ingest

        invalid = [
            {
                "type": "create",
                "filename": "memory/semantic-repair.md",
            }
        ]
        valid = [
            {
                "type": "create",
                "filename": "memory/semantic-repair.md",
                "title": "Semantic repair",
                "keywords": ["semantic", "repair"],
                "summary": "Host validation repaired schema-valid omissions.",
            }
        ]
        transport = _QueueStructuredTransport(
            json.dumps(invalid),
            json.dumps(valid),
        )

        out = ingest._triage("raw content", transport=transport)

        assert out == valid
        assert len(transport.requests) == 2
        feedback = transport.requests[1].messages[-1]["content"]
        assert "Validator errors" in feedback
        assert '"keyword":"required"' in feedback
        assert '"pointer":"/0/title"' in feedback
        assert '"pointer":"/0/summary"' in feedback
        assert '"pointer":"/0/keywords"' in feedback

    def test_create_requires_known_non_empty_semantic_fields(self) -> None:
        from chronovisor.ingest.ingest import _validate_triage_plan

        base = {"type": "create", "filename": "memory/strict.md"}
        assert _validate_triage_plan([base]) is None
        assert _validate_triage_plan([{**base, "title": "T"}]) is None
        assert (
            _validate_triage_plan(
                [
                    {
                        **base,
                        "title": "T",
                        "summary": "S",
                        "keywords": [],
                    }
                ]
            )
            is None
        )
        assert (
            _validate_triage_plan(
                [
                    {
                        **base,
                        "title": "T",
                        "summary": "S",
                        "keywords": ["strict"],
                        "diagnostic": "not allowed",
                    }
                ]
            )
            is None
        )

    def test_live_triage_three_invalid_responses_fail_closed(
        self, isolated_wiki: Path
    ) -> None:
        from chronovisor.ingest import ingest

        transport = _QueueStructuredTransport(
            '[{"type":"create","filename":',
            '{"type":"create","filename":"memory/no-array.md"}',
            '[{"type":"create"}]',
        )

        assert ingest._triage("raw content", transport=transport) is None
        assert len(transport.requests) == 3

    def test_live_triage_transport_exception_fails_closed(
        self, isolated_wiki: Path
    ) -> None:
        from chronovisor.ingest import ingest

        transport = _QueueStructuredTransport(RuntimeError("ollama offline"))

        assert ingest._triage("raw content", transport=transport) is None
        assert len(transport.requests) == 1

    def test_live_triage_grows_context_for_large_input(
        self, isolated_wiki: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from chronovisor.core.runtime_config import IngestConfig
        from chronovisor.ingest import ingest

        transport = _QueueStructuredTransport("[]")
        monkeypatch.setattr(
            ingest,
            "load_ingest_config",
            lambda: IngestConfig(num_ctx=32768, max_num_ctx=262144),
        )

        assert ingest._triage("x" * 20_000, transport=transport) == []
        assert len(transport.requests) == 1
        assert transport.requests[0].num_ctx == 65536

    def test_live_triage_over_configured_cap_clears_progress(
        self, isolated_wiki: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from chronovisor.core.runtime_config import IngestConfig
        from chronovisor.ingest import ingest

        transport = _QueueStructuredTransport("[]")
        progress: list[dict] = []
        monkeypatch.setattr(
            ingest,
            "load_ingest_config",
            lambda: IngestConfig(num_ctx=32768, max_num_ctx=32768),
        )

        assert (
            ingest._triage(
                "x" * 20_000,
                transport=transport,
                progress_callback=progress.append,
            )
            is None
        )
        assert transport.requests == []
        assert progress[-1]["active"] is False
        assert progress[-1]["failure_class"] == "context_window_exceeded"

    @pytest.mark.parametrize(
        ("required", "expected"),
        [
            (1, 32768),
            (32768, 32768),
            (32769, 65536),
            (65537, 131072),
            (131073, 262144),
        ],
    )
    def test_ingest_context_bucket_boundaries(
        self, required: int, expected: int
    ) -> None:
        from chronovisor.ingest import ingest

        assert (
            ingest._select_ingest_context(
                required,
                num_ctx=32768,
                max_num_ctx=262144,
            )
            == expected
        )

    def test_ingest_context_over_max_fails_closed(self) -> None:
        from chronovisor.ingest import ingest

        with pytest.raises(ingest.IngestContextCapacityError):
            ingest._select_ingest_context(
                262145,
                num_ctx=32768,
                max_num_ctx=262144,
            )

    def test_structured_transport_forwards_complete_request(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from chronovisor.decision.local_structured import ChatRequest
        from chronovisor.ingest import ingest

        captured: dict = {}

        def fake_generate(prompt: str, **kwargs) -> str:
            captured["prompt"] = prompt
            captured.update(kwargs)
            return "[]"

        monkeypatch.setattr(ingest, "_generate_with_progress", fake_generate)

        def callback(_event: dict) -> None:
            return None

        request = ChatRequest(
            model="ornith:test",
            messages=(
                {"role": "system", "content": "system"},
                {"role": "user", "content": "user"},
            ),
            schema={"type": "array"},
            num_ctx=131072,
            num_predict=2048,
            keep_alive="90s",
            read_timeout_ms=180000,
            max_output_chars=4000,
            temperature=0,
            seed=0,
        )

        assert ingest._structured_generate_transport(callback)(request) == "[]"
        assert captured == {
            "prompt": "<USER>\nuser",
            "system": "system",
            "progress_callback": callback,
            "model": "ornith:test",
            "num_ctx": 131072,
            "num_predict": 2048,
            "keep_alive": "90s",
            "read_timeout_ms": 180000,
            "temperature": 0,
            "seed": 0,
            "return_metadata": True,
            "format": {"type": "array"},
        }

    def test_structured_transport_preserves_completion_metadata(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from chronovisor.core import ollama
        from chronovisor.decision.local_structured import ChatRequest
        from chronovisor.ingest import ingest

        response = ollama.GenerateResponse(
            content="[]",
            done=False,
            done_reason=None,
        )

        def fake_generate(_prompt: str, **kwargs):
            assert kwargs["return_metadata"] is True
            return response

        monkeypatch.setattr(ingest, "_generate_with_progress", fake_generate)
        request = ChatRequest(
            model="ornith:test",
            messages=(
                {"role": "system", "content": "system"},
                {"role": "user", "content": "user"},
            ),
            schema={"type": "array"},
            num_ctx=131072,
            num_predict=2048,
            keep_alive="90s",
            read_timeout_ms=180000,
            max_output_chars=4000,
        )

        assert ingest._structured_generate_transport()(request) is response

    def test_native_structured_transport_preserves_chat_roles(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from chronovisor.decision.local_structured import ChatRequest
        from chronovisor.ingest import ingest

        captured: dict = {}

        def fake_chat(messages, **kwargs):
            captured["messages"] = messages
            captured.update(kwargs)
            return "[]"

        monkeypatch.setattr(ingest.ollama_runtime, "chat", fake_chat)
        request = ChatRequest(
            model="ornith:test",
            messages=(
                {"role": "system", "content": "system"},
                {"role": "user", "content": "original"},
                {"role": "assistant", "content": '{"bad":true}'},
                {"role": "user", "content": "validator feedback"},
            ),
            schema={"type": "array"},
            num_ctx=65536,
            num_predict=2048,
            keep_alive="90s",
            read_timeout_ms=180000,
            max_output_chars=4000,
            temperature=0,
            seed=0,
        )

        assert ingest._structured_chat_transport()(request) == "[]"
        assert captured["messages"] == [dict(message) for message in request.messages]
        assert captured["return_metadata"] is True
        assert captured["format"] == request.schema

    def test_production_triage_selects_native_chat_transport(
        self, isolated_wiki: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from contextlib import nullcontext

        from chronovisor.ingest import ingest

        native = _QueueStructuredTransport("[]")
        monkeypatch.setattr(ingest, "_structured_chat_transport", lambda: native)
        monkeypatch.setattr(
            ingest.ollama_runtime,
            "model_resource_lease",
            lambda **_kwargs: nullcontext(),
        )
        monkeypatch.setattr(
            ingest, "_admit_ingest_context", lambda _config, selected: selected
        )

        assert ingest._triage("ephemeral raw", raise_on_failure=True) == []
        assert len(native.requests) == 1
        assert [message["role"] for message in native.requests[0].messages] == [
            "system",
            "user",
        ]

    def test_successful_triage_clears_live_progress(self, isolated_wiki: Path) -> None:
        from chronovisor.ingest import ingest

        events: list[dict] = []
        transport = _QueueStructuredTransport("[]")

        assert (
            ingest._triage(
                "ephemeral raw", transport=transport, progress_callback=events.append
            )
            == []
        )
        assert events[-1]["event"] == "done"
        assert events[-1]["active"] is False

    def test_failed_triage_clears_live_progress(self, isolated_wiki: Path) -> None:
        from chronovisor.ingest import ingest

        events: list[dict] = []
        transport = _QueueStructuredTransport(RuntimeError("connection reset"))

        assert (
            ingest._triage(
                "ephemeral raw", transport=transport, progress_callback=events.append
            )
            is None
        )
        assert events[-1]["event"] == "error"
        assert events[-1]["active"] is False

    def test_structured_transport_supports_narrow_generate_fixture(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from chronovisor.decision.local_structured import ChatRequest
        from chronovisor.ingest import ingest

        captured: dict[str, str] = {}

        def narrow_generate(prompt: str, *, system: str | None) -> str:
            captured["prompt"] = prompt
            captured["system"] = system or ""
            return "[]"

        monkeypatch.setattr(ingest, "_generate_with_progress", narrow_generate)
        request = ChatRequest(
            model="ornith:test",
            messages=(
                {"role": "system", "content": "system"},
                {"role": "user", "content": "user"},
            ),
            schema={"type": "array"},
            num_ctx=131072,
            num_predict=2048,
            keep_alive="90s",
            read_timeout_ms=180000,
            max_output_chars=4000,
            temperature=0,
            seed=0,
        )

        assert (
            ingest._structured_generate_transport(lambda _event: None)(request) == "[]"
        )
        assert captured == {"prompt": "<USER>\nuser", "system": "system"}

    def test_empty_plan_passes(self) -> None:
        from chronovisor.ingest.ingest import _validate_triage_plan

        assert _validate_triage_plan([]) == []

    def test_string_entry_rejected(self) -> None:
        from chronovisor.ingest.ingest import _validate_triage_plan

        assert _validate_triage_plan(["not a dict"]) is None

    def test_unknown_type_rejected(self) -> None:
        from chronovisor.ingest.ingest import _validate_triage_plan

        assert _validate_triage_plan([{"type": "delete", "filename": "x.md"}]) is None

    def test_missing_filename_rejected(self) -> None:
        from chronovisor.ingest.ingest import _validate_triage_plan

        assert _validate_triage_plan([{"type": "create"}]) is None
        assert _validate_triage_plan([{"type": "create", "filename": ""}]) is None
        assert _validate_triage_plan([{"type": "create", "filename": "   "}]) is None

    def test_non_string_filename_rejected(self) -> None:
        from chronovisor.ingest.ingest import _validate_triage_plan

        assert _validate_triage_plan([{"type": "create", "filename": 123}]) is None


class TestPageGenerationBudget:
    def test_exact_live_boundary_reduces_only_output_reservation(self) -> None:
        from chronovisor.ingest import ingest

        # Reproduce the live preflight value exactly: the immutable
        # prompt/history envelope plus three 8192-token output reservations
        # required 268078 tokens against Ornith's hard 262144 ceiling.
        immutable_input_bytes = 238_318
        prompt = "x" * immutable_input_bytes
        assert (
            ingest._required_generate_context_tokens(
                prompt,
                None,
                num_predict=8_192,
            )
            == 268_078
        )

        budget = ingest._select_page_generation_budget(
            prompt,
            None,
            configured_num_predict=8_192,
            num_ctx=32_768,
            max_num_ctx=262_144,
        )

        assert budget.num_predict == 6_214
        assert budget.required_num_ctx == 262_144
        assert budget.num_ctx == 262_144
        assert prompt == "x" * immutable_input_bytes

    def test_input_that_cannot_fit_fixed_output_floor_defers(self) -> None:
        from chronovisor.ingest import ingest

        # At the fixed 2048-token floor this immutable input requires 262145.
        prompt = "x" * 250_817
        assert (
            ingest._required_generate_context_tokens(
                prompt,
                None,
                num_predict=ingest._MIN_ADAPTIVE_PAGE_NUM_PREDICT,
            )
            == 262_145
        )

        with pytest.raises(
            ingest.IngestContextCapacityError,
            match=r"minimum num_predict 2048.*max_num_ctx 262144",
        ):
            ingest._select_page_generation_budget(
                prompt,
                None,
                configured_num_predict=8_192,
                num_ctx=32_768,
                max_num_ctx=262_144,
            )

    def test_normal_request_preserves_configured_output_budget(self) -> None:
        from chronovisor.ingest import ingest

        budget = ingest._select_page_generation_budget(
            "x" * 1_000,
            None,
            configured_num_predict=8_192,
            num_ctx=32_768,
            max_num_ctx=262_144,
        )

        assert budget.num_predict == 8_192
        assert budget.required_num_ctx == 30_760
        assert budget.num_ctx == 32_768

    def test_generate_calls_reuse_adapted_budget_for_every_repair(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        from chronovisor.core.runtime_config import IngestConfig
        from chronovisor.ingest import ingest

        invalid_one = "=== NEW PAGE: memory/adaptive.md ===\nmissing frontmatter"
        invalid_two = "=== NEW PAGE: memory/adaptive.md ===\nstill invalid"
        valid = (
            "=== NEW PAGE: memory/adaptive.md ===\n"
            "---\ntitle: Adaptive\nupdated: 2026-07-14\n---\nbody\n"
            "=== END PAGE ==="
        )
        responses = iter([invalid_one, invalid_two, valid])
        calls: list[tuple[int, int, str]] = []

        monkeypatch.setattr(
            ingest,
            "load_ingest_config",
            lambda: IngestConfig(
                num_ctx=32_768,
                max_num_ctx=262_144,
                num_predict=8_192,
            ),
        )
        monkeypatch.setattr(
            ingest,
            "_build_focused_context",
            lambda *_args, **_kwargs: "",
        )

        def fake_generate(prompt: str, **kwargs) -> str:
            calls.append((kwargs["num_ctx"], kwargs["num_predict"], prompt))
            return next(responses)

        monkeypatch.setattr(ingest, "_generate_with_progress", fake_generate)
        diagnostics: dict = {}
        raw = "r" * 230_000

        result = ingest._generate_one(
            {
                "type": "create",
                "filename": "memory/adaptive.md",
                "title": "Adaptive",
                "summary": "Preserve all source bytes",
            },
            raw,
            diagnostics=diagnostics,
        )

        assert result is not None
        assert len(calls) == 3
        assert len({(num_ctx, num_predict) for num_ctx, num_predict, _ in calls}) == 1
        selected_num_ctx, selected_num_predict, _ = calls[0]
        assert selected_num_ctx == 262_144
        assert 2_048 <= selected_num_predict < 8_192
        assert diagnostics["num_predict"] == selected_num_predict
        assert diagnostics["required_num_ctx"] <= selected_num_ctx
        assert all(raw in transcript for _, _, transcript in calls)
        assert invalid_one in calls[1][2]
        assert invalid_two in calls[2][2]


class TestOversizedAppendOnlyUpdateContext:
    def test_sections_are_lossless_and_ignore_headings_inside_fences(self) -> None:
        from chronovisor.ingest import ingest

        page = (
            "---\ntitle: Existing\nupdated: 2026-01-01\n---\n"
            "# Alpha\nalpha start\n"
            "```python\n## not-a-heading\nprint('still alpha')\n"
            "``` trailing text\n## still-inside-fence\n```\n"
            "alpha end\n"
            "## Relevant section\nunique product value evidence\n"
            "### Nested evidence\nstays in the complete H2 section\n"
        )

        sections = ingest._markdown_sections(page)

        assert "".join(section.content for section in sections) == page
        assert [section.heading for section in sections] == [
            None,
            "# Alpha",
            "## Relevant section",
        ]
        assert "## not-a-heading" in sections[1].content
        assert "## still-inside-fence" in sections[1].content
        assert "### Nested evidence" in sections[2].content
        assert all(
            section.sha256
            == hashlib.sha256(section.content.encode("utf-8")).hexdigest()
            for section in sections
        )

    def test_compact_context_is_deterministic_whole_section_and_read_only(
        self,
        isolated_wiki: Path,
    ) -> None:
        from chronovisor.ingest import ingest

        page_text = (
            "---\ntitle: Existing\nupdated: 2026-01-01\n---\n"
            "# Unrelated\nother material\n"
            "```text\n## fenced heading\n```\n"
            "## Product value\nproduct value structure evidence\ncomplete tail\n"
        )
        page_path = _seed_page(
            isolated_wiki,
            "career/oversized-existing.md",
            page_text,
        )
        relevant = next(
            section
            for section in ingest._markdown_sections(page_text)
            if section.heading == "## Product value"
        )
        op = {
            "type": "update",
            "filename": "career/oversized-existing.md",
            "title": "Product value",
            "keywords": ["product", "value"],
            "summary": "Add grounded structure evidence",
        }

        first = ingest._build_compact_update_context(
            op,
            "product value structure evidence",
            max_selected_bytes=len(relevant.content.encode("utf-8")),
        )
        second = ingest._build_compact_update_context(
            op,
            "product value structure evidence",
            max_selected_bytes=len(relevant.content.encode("utf-8")),
        )

        assert first is not None
        assert second == first
        assert first.selected_sections == (relevant,)
        assert relevant.content in first.text
        assert "other material" not in first.text
        assert '"## fenced heading"' not in first.text
        assert f"page_bytes: {len(page_text.encode('utf-8'))}" in first.text
        assert hashlib.sha256(page_text.encode("utf-8")).hexdigest() in first.text
        for section in ingest._markdown_sections(page_text):
            digest_b64url = base64.urlsafe_b64encode(
                bytes.fromhex(section.sha256)
            ).rstrip(b"=")
            row = (
                f"{section.start_line}-{section.end_line}\t"
                f"{len(section.content.encode('utf-8'))}\t"
                f"{digest_b64url.decode('ascii')}\t"
                f"{json.dumps(section.heading or '[preamble]', ensure_ascii=False)}"
            )
            assert row in first.text
        assert page_path.read_text(encoding="utf-8") == page_text

    def test_compact_defers_when_strongest_relevant_section_does_not_fit(
        self,
        isolated_wiki: Path,
    ) -> None:
        from chronovisor.ingest import ingest

        _seed_page(
            isolated_wiki,
            "memory/oversized-relevant.md",
            (
                "---\ntitle: Existing\nupdated: 2026-01-01\n---\n"
                "# Unrelated\nsmall unrelated filler\n"
                "## Needle evidence\nneedle evidence " + "x" * 256 + "\n"
            ),
        )
        op = {
            "type": "update",
            "filename": "memory/oversized-relevant.md",
            "title": "Needle evidence",
            "keywords": ["needle"],
            "summary": "Add needle evidence",
        }

        compact = ingest._build_compact_update_context(
            op,
            "needle evidence",
            max_selected_bytes=64,
        )

        assert compact is None

    def test_compact_defers_when_no_section_has_positive_relevance(
        self,
        isolated_wiki: Path,
    ) -> None:
        from chronovisor.ingest import ingest

        _seed_page(
            isolated_wiki,
            "memory/no-relevance.md",
            (
                "---\ntitle: Existing\nupdated: 2026-01-01\n---\n"
                "# Completely unrelated\nsmall filler\n"
            ),
        )
        op = {
            "type": "update",
            "filename": "memory/no-relevance.md",
            "title": "Needle evidence",
            "keywords": ["needle"],
            "summary": "Add needle evidence",
        }

        compact = ingest._build_compact_update_context(
            op,
            "needle evidence",
            max_selected_bytes=8_192,
        )

        assert compact is None

    def test_h1_h2_manifest_keeps_nested_headings_and_full_digests_under_bound(
        self,
    ) -> None:
        from chronovisor.ingest import ingest

        page_text = (
            "---\ntitle: Large outline\nupdated: 2026-01-01\n---\n"
            "# Complete title section\n"
            + "".join(
                f"## {index}. 完全な日本語見出し product value evidence {index}\n"
                f"top-level body {index}\n"
                f"### Nested heading {index}\nnested body {index}\n"
                for index in range(1, 95)
            )
        )

        sections = ingest._markdown_sections(page_text)
        outline = ingest._render_compact_update_context(
            page_id="large-outline",
            page_text=page_text,
            sections=sections,
            selected=(),
        )

        assert len(sections) == 96  # frontmatter + H1 + all 94 H2 sections
        assert "### Nested heading 94" in sections[-1].content
        assert len(outline.encode("utf-8")) <= ingest._MAX_COMPACT_UPDATE_OUTLINE_BYTES
        for section in sections:
            digest_b64url = base64.urlsafe_b64encode(
                bytes.fromhex(section.sha256)
            ).rstrip(b"=")
            assert digest_b64url.decode("ascii") in outline
            assert (
                json.dumps(
                    section.heading or "[preamble]",
                    ensure_ascii=False,
                )
                in outline
            )

    def test_compact_binding_uses_exact_crlf_disk_bytes(
        self,
        isolated_wiki: Path,
    ) -> None:
        from chronovisor.ingest import ingest

        page_text = (
            "---\r\ntitle: Existing\r\nupdated: 2026-01-01\r\n"
            "tags: [d/tools-config, t/reference, s/2026]\r\n---\r\n"
            "# Existing\r\nbody\r\n## Product value\r\nexact evidence\r\n"
        )
        page_path = _seed_page(
            isolated_wiki,
            "memory/exact-crlf.md",
            "placeholder",
        )
        page_path.write_bytes(page_text.encode("utf-8"))
        op = {
            "type": "update",
            "filename": "memory/exact-crlf.md",
            "title": "Product value",
            "keywords": ["product", "value"],
            "summary": "Add exact evidence",
        }

        compact = ingest._build_compact_update_context(
            op,
            "product value exact evidence",
            max_selected_bytes=8_192,
        )

        assert compact is not None
        assert compact.page_bytes == len(page_text.encode("utf-8"))
        assert (
            compact.page_sha256 == hashlib.sha256(page_text.encode("utf-8")).hexdigest()
        )
        assert "\r\n" in compact.selected_sections[-1].content
        planned, _totals = ingest._prepare_operations(
            [
                {
                    "type": "update",
                    "filename": "memory/exact-crlf.md",
                    "content": "## Grounded append\nnew body",
                    "_compact_update_preimage_sha256": compact.page_sha256,
                }
            ],
            read_only=True,
        )
        assert len(planned) == 1
        assert planned[0].previous_text == page_text
        assert page_path.read_bytes() == page_text.encode("utf-8")

    def test_exact_live_291077_boundary_compacts_without_dropping_raw(
        self,
        isolated_wiki: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        from chronovisor.core.runtime_config import IngestConfig
        from chronovisor.ingest import ingest

        section_prefix = (
            "# Product value\nproduct value and performance evidence\n"
        )
        strongest_section = section_prefix + "x" * (
            12 * 1_024 - len(section_prefix.encode("utf-8"))
        )
        assert len(strongest_section.encode("utf-8")) == 12 * 1_024
        page_text = (
            "---\ntitle: Interview Narrative\nupdated: 2026-01-01\n---\n"
            "# Historical background\nOMITTED-FULL-PAGE-BODY\n"
            + strongest_section
        )
        page_path = _seed_page(
            isolated_wiki,
            "career/interview-narrative-product-value-vs-politics.md",
            page_text,
        )
        original_page_bytes = page_path.read_bytes()
        op = {
            "type": "update",
            "filename": "career/interview-narrative-product-value-vs-politics.md",
            "title": "Interview Narrative",
            "keywords": ["product", "value", "performance"],
            "summary": "Add product-value interview evidence",
        }
        raw = "RAW-291077 product value and performance evidence"
        original_raw_bytes = raw.encode("utf-8")
        empty_prompt = ingest._build_page_generation_prompt(
            context="",
            raw_content=raw,
            op_type="update",
            filename=op["filename"],
            title=op["title"],
            summary=op["summary"],
            feedback_block="",
            current_date=date.today().isoformat(),
        )
        fixed_without_context = (
            ingest._required_generate_context_tokens(
                empty_prompt,
                ingest.UPDATE_SYSTEM_PROMPT,
                num_predict=8_192,
            )
            - 3 * 8_192
        )
        full_context = "x" * (266_501 - fixed_without_context)
        full_prompt = ingest._build_page_generation_prompt(
            context=full_context,
            raw_content=raw,
            op_type="update",
            filename=op["filename"],
            title=op["title"],
            summary=op["summary"],
            feedback_block="",
            current_date=date.today().isoformat(),
        )
        assert (
            ingest._required_generate_context_tokens(
                full_prompt,
                ingest.UPDATE_SYSTEM_PROMPT,
                num_predict=8_192,
            )
            == 291_077
        )

        monkeypatch.setattr(
            ingest,
            "load_ingest_config",
            lambda: IngestConfig(
                num_ctx=32_768,
                max_num_ctx=262_144,
                num_predict=8_192,
                max_related_context_bytes=8_192,
            ),
        )
        monkeypatch.setattr(
            ingest,
            "_build_focused_context",
            lambda *_args, **_kwargs: full_context,
        )
        calls: list[tuple[str, dict]] = []

        def fake_generate(prompt: str, **kwargs) -> str:
            calls.append((prompt, kwargs))
            return (
                "=== UPDATE PAGE: "
                "career/interview-narrative-product-value-vs-politics.md ===\n"
                "## New evidence\ngrounded append\n"
                "=== END PAGE ==="
            )

        monkeypatch.setattr(ingest, "_generate_with_progress", fake_generate)
        diagnostics: dict = {}

        result = ingest._generate_one(op, raw, diagnostics=diagnostics)

        assert result is not None
        assert len(calls) == 1
        compact_prompt, kwargs = calls[0]
        assert raw in compact_prompt
        assert full_context not in compact_prompt
        assert page_text not in compact_prompt
        assert "OMITTED-FULL-PAGE-BODY" not in compact_prompt
        assert "Complete deterministic H1/H2 section manifest (TSV):" in compact_prompt
        assert strongest_section in compact_prompt
        assert "Omitted section bytes are still present on disk" in kwargs["system"]
        assert kwargs["num_predict"] == 8_192
        assert diagnostics["original_required_num_ctx"] == 291_077
        assert diagnostics["required_num_ctx"] <= 262_144
        assert diagnostics["context_strategy"] == "append_only_outline_sections"
        assert (
            diagnostics["context_page_sha256"]
            == hashlib.sha256(page_text.encode("utf-8")).hexdigest()
        )
        assert (
            result["_compact_update_preimage_sha256"]
            == diagnostics["context_page_sha256"]
        )
        assert page_path.read_bytes() == original_page_bytes
        assert raw.encode("utf-8") == original_raw_bytes

    def test_compaction_unavailable_preserves_adaptive_full_context_fallback(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        from chronovisor.core.runtime_config import IngestConfig
        from chronovisor.ingest import ingest

        op = {
            "type": "update",
            "filename": "memory/adaptive-fallback.md",
            "title": "Adaptive fallback",
            "summary": "Preserve the full prompt",
        }
        raw = "raw-full-fallback"
        empty_prompt = ingest._build_page_generation_prompt(
            context="",
            raw_content=raw,
            op_type="update",
            filename=op["filename"],
            title=op["title"],
            summary=op["summary"],
            feedback_block="",
            current_date=date.today().isoformat(),
        )
        fixed_without_context = (
            ingest._required_generate_context_tokens(
                empty_prompt,
                ingest.UPDATE_SYSTEM_PROMPT,
                num_predict=8_192,
            )
            - 3 * 8_192
        )
        full_context = "f" * ((268_078 - 3 * 8_192) - fixed_without_context)
        monkeypatch.setattr(
            ingest,
            "load_ingest_config",
            lambda: IngestConfig(
                num_ctx=32_768,
                max_num_ctx=262_144,
                num_predict=8_192,
            ),
        )
        monkeypatch.setattr(
            ingest,
            "_build_focused_context",
            lambda *_args, **_kwargs: full_context,
        )
        monkeypatch.setattr(
            ingest,
            "_build_compact_update_context",
            lambda *_args, **_kwargs: None,
        )
        calls: list[tuple[str, dict]] = []

        def fake_generate(prompt: str, **kwargs) -> str:
            calls.append((prompt, kwargs))
            return (
                "=== UPDATE PAGE: memory/adaptive-fallback.md ===\n"
                "## Update\nbody\n=== END PAGE ==="
            )

        monkeypatch.setattr(ingest, "_generate_with_progress", fake_generate)

        result = ingest._generate_one(op, raw)

        assert result is not None
        assert len(calls) == 1
        assert full_context in calls[0][0]
        assert calls[0][1]["num_predict"] == 6_214
        assert "_compact_update_preimage_sha256" not in result

    def test_prepare_rejects_stale_compact_preimage(
        self,
        isolated_wiki: Path,
    ) -> None:
        from chronovisor.ingest import ingest

        original = (
            "---\r\ntitle: Existing\r\nupdated: 2026-01-01\r\n"
            "tags: [d/tools-config, t/reference, s/2026]\r\n---\r\n"
            "# Existing\r\nbody\r\n"
        )
        page_path = _seed_page(
            isolated_wiki,
            "memory/stale-compact.md",
            "placeholder",
        )
        page_path.write_bytes(original.encode("utf-8"))
        operation = {
            "type": "update",
            "filename": "memory/stale-compact.md",
            "content": "## Grounded append\nnew body",
            "_compact_update_preimage_sha256": hashlib.sha256(
                original.encode("utf-8")
            ).hexdigest(),
        }
        changed = original + "concurrent change\r\n"
        page_path.write_bytes(changed.encode("utf-8"))

        with pytest.raises(
            IngestApplyError,
            match="compact update preimage changed before prepare",
        ):
            ingest._prepare_operations([operation], read_only=True)
        assert page_path.read_bytes() == changed.encode("utf-8")

    def test_prepare_accepts_matching_compact_preimage_without_writing(
        self,
        isolated_wiki: Path,
    ) -> None:
        from chronovisor.ingest import ingest

        original = (
            "---\ntitle: Existing\nupdated: 2026-01-01\n"
            "tags: [d/tools-config, t/reference, s/2026]\n---\n# Existing\nbody\n"
        )
        page_path = _seed_page(
            isolated_wiki,
            "memory/matching-compact.md",
            original,
        )
        operation = {
            "type": "update",
            "filename": "memory/matching-compact.md",
            "content": "## Grounded append\nnew body",
            "_compact_update_preimage_sha256": hashlib.sha256(
                original.encode("utf-8")
            ).hexdigest(),
        }

        planned, _totals = ingest._prepare_operations([operation], read_only=True)

        assert len(planned) == 1
        assert planned[0].previous_text == original
        assert "## Grounded append\nnew body" in planned[0].new_body
        assert page_path.read_text(encoding="utf-8") == original

    def test_compact_context_that_still_cannot_fit_defers(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        from chronovisor.core.runtime_config import IngestConfig
        from chronovisor.ingest import ingest

        huge = "x" * 300_000
        compact = ingest._CompactUpdateContext(
            text=huge,
            page_id="too-large",
            page_sha256="a" * 64,
            page_bytes=len(huge),
            section_count=1,
            selected_sections=(
                ingest._MarkdownSection(
                    start_line=1,
                    end_line=1,
                    heading="# Too large",
                    content=huge,
                    sha256=hashlib.sha256(huge.encode("utf-8")).hexdigest(),
                ),
            ),
        )
        monkeypatch.setattr(
            ingest,
            "load_ingest_config",
            lambda: IngestConfig(
                num_ctx=32_768,
                max_num_ctx=262_144,
                num_predict=8_192,
            ),
        )
        monkeypatch.setattr(
            ingest,
            "_build_focused_context",
            lambda *_args, **_kwargs: huge,
        )
        monkeypatch.setattr(
            ingest,
            "_build_compact_update_context",
            lambda *_args, **_kwargs: compact,
        )
        monkeypatch.setattr(
            ingest,
            "_generate_with_progress",
            lambda *_args, **_kwargs: pytest.fail("transport must not run"),
        )

        with pytest.raises(RuntimeError, match="context_window_exceeded"):
            ingest._generate_one(
                {
                    "type": "update",
                    "filename": "memory/too-large.md",
                    "title": "Too large",
                    "summary": "Still too large",
                },
                "raw remains complete",
            )


class TestIngestContextAdmission:
    def test_high_context_evicts_every_other_resident_model(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from chronovisor.core import ollama
        from chronovisor.core.runtime_config import IngestConfig
        from chronovisor.ingest import ingest

        config = IngestConfig(
            model="ornith:ingest",
            num_ctx=32768,
            max_num_ctx=262144,
            memory_reserve_gib=16,
        )
        unloaded: list[str] = []
        planned: dict = {}
        monkeypatch.setattr(
            ollama,
            "resident_model_rows",
            lambda: {
                "ornith:ingest": (30 * ollama.GIB, 32768),
                "decision:35b": (30 * ollama.GIB, 32768),
                "recall:9b": (9 * ollama.GIB, 8192),
            },
        )
        monkeypatch.setattr(
            ollama,
            "unload_named_model",
            lambda model: unloaded.append(model) or True,
        )

        def fake_plan(models, **kwargs):
            planned["models"] = models
            planned.update(kwargs)
            return ollama.ModelResidencyPlan(
                num_ctx=262144,
                max_resident_models=1,
                capacity_bytes=80 * ollama.GIB,
                reserve_bytes=16 * ollama.GIB,
                available_bytes=80 * ollama.GIB,
                total_bytes=128 * ollama.GIB,
                estimated_model_bytes=(("ornith:ingest", 60 * ollama.GIB),),
                role_contexts=(("ornith:ingest", 262144),),
                resident_models=("ornith:ingest",),
                calibrated_models=("ornith:ingest",),
                source="test",
                reuse_larger_context=True,
            )

        monkeypatch.setattr(ollama, "plan_model_residency", fake_plan)

        assert ingest._admit_ingest_context(config, 262144) == 262144
        assert unloaded == ["decision:35b", "recall:9b"]
        assert planned["models"] == ["ornith:ingest"]
        assert planned["num_ctx"] == 262144
        assert planned["max_num_ctx"] == 262144
        assert planned["configured_max_resident"] == 1
        assert planned["reuse_larger_context"] is True

    def test_lower_context_reclaims_unrelated_models_only_after_initial_stall(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from chronovisor.core import ollama
        from chronovisor.core.runtime_config import IngestConfig
        from chronovisor.ingest import ingest

        config = IngestConfig(
            model="ornith:ingest",
            num_ctx=32768,
            max_num_ctx=262144,
            memory_reserve_gib=16,
        )
        unloaded: list[str] = []
        plan_calls: list[dict] = []
        monkeypatch.setattr(
            ollama,
            "resident_model_rows",
            lambda: {
                "decision:35b": (30 * ollama.GIB, 65536),
                "recall:9b": (9 * ollama.GIB, 8192),
            },
        )
        monkeypatch.setattr(
            ollama,
            "unload_named_model",
            lambda model: unloaded.append(model) or True,
        )

        def fake_plan(models, **kwargs):
            plan_calls.append({"models": models, **kwargs})
            admitted = int(len(plan_calls) > 1)
            return ollama.ModelResidencyPlan(
                num_ctx=65536,
                max_resident_models=admitted,
                capacity_bytes=60 * ollama.GIB,
                reserve_bytes=16 * ollama.GIB,
                available_bytes=60 * ollama.GIB,
                total_bytes=128 * ollama.GIB,
                estimated_model_bytes=(("ornith:ingest", 30 * ollama.GIB),),
                role_contexts=(("ornith:ingest", 65536),),
                resident_models=(),
                calibrated_models=(("ornith:ingest",) if admitted else ()),
                source="test",
                reuse_larger_context=True,
            )

        monkeypatch.setattr(ollama, "plan_model_residency", fake_plan)

        assert ingest._admit_ingest_context(config, 65536) == 65536
        assert len(plan_calls) == 2
        assert unloaded == ["decision:35b", "recall:9b"]

    def test_stage_two_uses_related_budget_and_dynamic_bucket(
        self,
        isolated_wiki: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        from chronovisor.core.runtime_config import IngestConfig
        from chronovisor.ingest import ingest

        related = _seed_page(
            isolated_wiki,
            "memory/large-related.md",
            "---\ntitle: Related\nupdated: 2026-01-01\n---\n" + "z" * 20000,
        )
        captured: dict = {}
        monkeypatch.setattr(
            ingest,
            "load_ingest_config",
            lambda: IngestConfig(
                num_ctx=32768,
                max_num_ctx=262144,
                num_predict=8192,
                max_related_context_bytes=512,
            ),
        )
        monkeypatch.setattr(
            ingest,
            "_search_related_pages",
            lambda *_args, **_kwargs: [related],
        )

        def fake_generate(prompt: str, **kwargs) -> str:
            captured["prompt"] = prompt
            captured.update(kwargs)
            return (
                "=== NEW PAGE: memory/new.md ===\n"
                "---\ntitle: New\nupdated: 2026-01-01\n---\nbody\n"
                "=== END PAGE ==="
            )

        monkeypatch.setattr(ingest, "_generate_with_progress", fake_generate)

        result = ingest._generate_one(
            {
                "type": "create",
                "filename": "memory/new.md",
                "title": "New",
                "keywords": ["related"],
            },
            "r" * 40000,
        )

        assert result is not None
        # The admitted bucket includes the complete initial + two-repair
        # history, not only the first page-generation turn.
        assert captured["num_ctx"] == 131072
        assert "z" * 1000 not in captured["prompt"]
        assert "r" * 40000 in captured["prompt"]

    @pytest.mark.parametrize(
        ("response", "failure_class"),
        [
            (
                ollama.GenerateResponse(
                    content=(
                        "=== NEW PAGE: memory/new.md ===\n"
                        "---\ntitle: New\nupdated: 2026-01-01\n---\nbody\n"
                        "=== END PAGE ==="
                    ),
                    done=False,
                ),
                "completion_incomplete",
            ),
            (
                ollama.GenerateResponse(
                    content=(
                        "=== NEW PAGE: memory/new.md ===\n"
                        "---\ntitle: New\nupdated: 2026-01-01\n---\nbody\n"
                        "=== END PAGE ==="
                    ),
                    done=False,
                    streamed=True,
                ),
                "stream_incomplete",
            ),
        ],
    )
    def test_stage_two_rejects_incomplete_completion_before_page_parse(
        self,
        isolated_wiki: Path,
        monkeypatch: pytest.MonkeyPatch,
        response: ollama.GenerateResponse,
        failure_class: str,
    ) -> None:
        from chronovisor.ingest import ingest

        calls = 0

        def fake_generate(_prompt: str, **kwargs):
            nonlocal calls
            calls += 1
            assert kwargs["return_metadata"] is True
            return response

        monkeypatch.setattr(ingest, "_generate_with_progress", fake_generate)

        with pytest.raises(
            RuntimeError,
            match=rf"ingest generation {failure_class}:",
        ):
            ingest._generate_one(
                {
                    "type": "create",
                    "filename": "memory/new.md",
                    "title": "New",
                },
                "raw",
            )

        assert calls == 1

    def test_stage_two_replaces_output_truncation_from_original_evidence(
        self,
        isolated_wiki: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        from chronovisor.ingest import ingest

        partial = "PARTIAL_COMPLETION_MUST_NOT_BECOME_HISTORY"
        target = _seed_page(
            isolated_wiki,
            "memory/new.md",
            "---\ntitle: Existing\nupdated: 2026-01-01\n---\nexisting\n",
        )
        preimage = target.read_bytes()
        valid = "=== UPDATE PAGE: memory/new.md ===\n## Added\n\nbody\n=== END PAGE ==="
        responses = iter(
            [
                ingest.ollama_runtime.GenerateResponse(
                    content=partial,
                    done=True,
                    done_reason="length",
                    prompt_eval_count=100,
                    eval_count=8192,
                ),
                ingest.ollama_runtime.GenerateResponse(
                    content=valid,
                    done=True,
                    done_reason="stop",
                    prompt_eval_count=200,
                    eval_count=100,
                ),
            ]
        )
        calls: list[tuple[str, dict]] = []
        progress: list[dict] = []

        def fake_generate(prompt: str, **kwargs):
            calls.append((prompt, kwargs))
            return next(responses)

        monkeypatch.setattr(ingest, "_generate_with_progress", fake_generate)
        diagnostics: dict = {}

        result = ingest._generate_one(
            {
                "type": "update",
                "filename": "memory/new.md",
                "title": "New",
            },
            "ORIGINAL_GROUNDED_EVIDENCE",
            diagnostics=diagnostics,
            progress_callback=progress.append,
        )

        assert result is not None
        assert result["content"].endswith("body")
        assert len(calls) == 2
        assert "ORIGINAL_GROUNDED_EVIDENCE" in calls[1][0]
        assert "reached the transport output limit and was discarded" in calls[1][0]
        assert "return only the new append body inside the wrapper" in calls[1][0]
        assert "do not repeat, summarize, or rewrite" in calls[1][0]
        assert partial not in calls[1][0]
        assert f"at most {calls[0][1]['num_predict'] // 2} output tokens" in calls[1][0]
        assert calls[1][1]["num_predict"] == calls[0][1]["num_predict"]
        assert "within 4096 output tokens" in calls[1][1]["system"]
        assert "`=== UPDATE PAGE: memory/new.md ===`" in calls[1][1]["system"]
        assert "exact final line `=== END PAGE ===`" in calls[1][1]["system"]
        assert diagnostics["attempts"] == 2
        assert diagnostics["repair_turns"] == 1
        assert diagnostics["output_truncation_retries"] == 1
        assert [event["event"] for event in progress] == ["repair"]
        assert target.read_bytes() == preimage

    def test_stage_two_resets_validator_history_when_repair_truncates(
        self,
        isolated_wiki: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        from chronovisor.ingest import ingest

        target = _seed_page(
            isolated_wiki,
            "memory/new.md",
            "---\ntitle: Existing\nupdated: 2026-01-01\n---\nexisting\n",
        )
        preimage = target.read_bytes()
        invalid = (
            "=== UPDATE PAGE: memory/new.md ===\n"
            "## Added\n\ninvalid first response\n"
            "=== END PAGE ===\nUNEXPECTED_TRAILING_TEXT"
        )
        partial = "TRUNCATED_VALIDATOR_REPAIR_MUST_BE_DISCARDED"
        valid = "=== UPDATE PAGE: memory/new.md ===\n## Added\n\nbody\n=== END PAGE ==="
        responses = iter(
            [
                ingest.ollama_runtime.GenerateResponse(
                    content=invalid,
                    done=True,
                    done_reason="stop",
                    prompt_eval_count=200,
                    eval_count=100,
                ),
                ingest.ollama_runtime.GenerateResponse(
                    content=partial,
                    done=True,
                    done_reason="length",
                    prompt_eval_count=200,
                    eval_count=8192,
                ),
                ingest.ollama_runtime.GenerateResponse(
                    content=valid,
                    done=True,
                    done_reason="stop",
                    prompt_eval_count=200,
                    eval_count=100,
                ),
            ]
        )
        prompts: list[str] = []
        progress: list[dict] = []

        def fake_generate(prompt: str, **_kwargs):
            prompts.append(prompt)
            return next(responses)

        monkeypatch.setattr(ingest, "_generate_with_progress", fake_generate)
        diagnostics: dict = {}

        result = ingest._generate_one(
            {
                "type": "update",
                "filename": "memory/new.md",
                "title": "New",
            },
            "ORIGINAL_GROUNDED_EVIDENCE",
            diagnostics=diagnostics,
            progress_callback=progress.append,
        )

        assert result is not None
        assert len(prompts) == 3
        assert invalid in prompts[1]
        assert "code: missing_end_marker" in prompts[1]
        assert "ORIGINAL_GROUNDED_EVIDENCE" in prompts[2]
        assert "reached the transport output limit and was discarded" in prompts[2]
        assert invalid not in prompts[2]
        assert partial not in prompts[2]
        assert "Validator errors:" not in prompts[2]
        assert diagnostics["attempts"] == 3
        assert diagnostics["repair_turns"] == 2
        assert diagnostics["output_truncation_retries"] == 1
        assert [event["event"] for event in progress] == ["repair", "repair"]
        planned, _totals = ingest._prepare_operations([result], read_only=True)
        [prepared] = planned
        assert prepared.previous_text == preimage.decode("utf-8")
        from chronovisor.core.frontmatter import parse as parse_frontmatter

        _meta, prepared_body = parse_frontmatter(prepared.new_body)
        assert prepared_body == "existing\n\n## Added\n\nbody\n"
        assert target.read_bytes() == preimage

    def test_stage_two_bounds_repeated_output_truncation(
        self,
        isolated_wiki: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        from chronovisor.ingest import ingest

        responses = iter(
            [
                ingest.ollama_runtime.GenerateResponse(
                    content=f"PARTIAL_{index}",
                    done=True,
                    done_reason="length",
                    prompt_eval_count=100,
                    eval_count=8192,
                )
                for index in range(3)
            ]
        )
        calls: list[tuple[str, dict]] = []
        progress: list[dict] = []

        def fake_generate(prompt: str, **kwargs):
            calls.append((prompt, kwargs))
            return next(responses)

        monkeypatch.setattr(ingest, "_generate_with_progress", fake_generate)
        diagnostics: dict = {}

        with pytest.raises(
            RuntimeError,
            match="ingest generation output_truncated:",
        ):
            ingest._generate_one(
                {
                    "type": "create",
                    "filename": "memory/new.md",
                    "title": "New",
                },
                "ORIGINAL_GROUNDED_EVIDENCE",
                diagnostics=diagnostics,
                progress_callback=progress.append,
            )

        assert len(calls) == 3
        assert "PARTIAL_0" not in calls[1][0]
        assert "PARTIAL_1" not in calls[2][0]
        assert all("ORIGINAL_GROUNDED_EVIDENCE" in prompt for prompt, _ in calls)
        assert len({kwargs["num_predict"] for _, kwargs in calls}) == 1
        assert "within 4096 output tokens" in calls[1][1]["system"]
        assert "within 2048 output tokens" in calls[2][1]["system"]
        assert "within 4096 output tokens" not in calls[2][1]["system"]
        for prompt, kwargs in calls[1:]:
            assert "`=== NEW PAGE: memory/new.md ===`" in kwargs["system"]
            assert "exact final line `=== END PAGE ===`" in kwargs["system"]
            assert "PARTIAL_" not in prompt
        assert diagnostics["attempts"] == 3
        assert diagnostics["failure_class"] == "output_truncated"
        assert diagnostics["output_truncation_retries"] == 2
        assert [event["event"] for event in progress] == [
            "repair",
            "repair",
            "error",
        ]

    def test_related_budget_never_truncates_current_update_target(
        self,
        isolated_wiki: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        from chronovisor.ingest import ingest

        target_body = "target-byte-exact-" + "T" * 4_000
        target = _seed_page(
            isolated_wiki,
            "memory/target.md",
            "---\ntitle: Target\nupdated: 2026-01-01\n---\n" + target_body,
        )
        related = _seed_page(
            isolated_wiki,
            "memory/related.md",
            "---\ntitle: Related\nupdated: 2026-01-01\n---\n" + "R" * 2_000,
        )
        monkeypatch.setattr(
            ingest,
            "_search_related_pages",
            lambda *_args, **_kwargs: [target, related],
        )

        context = ingest._build_focused_context(
            {
                "type": "update",
                "filename": "memory/target.md",
                "keywords": ["target"],
            },
            "raw",
            max_bytes=64,
        )

        assert target_body in context
        assert "R" * 100 not in context
        assert context.count("Current content of [[target]]") == 1

    def test_related_budget_selects_only_complete_page_blocks(
        self,
        isolated_wiki: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        from chronovisor.ingest import ingest

        too_large = _seed_page(
            isolated_wiki,
            "memory/too-large.md",
            "---\ntitle: Too large\nupdated: 2026-01-01\n---\n" + "L" * 1_000,
        )
        complete = _seed_page(
            isolated_wiki,
            "memory/complete.md",
            "---\ntitle: Complete\nupdated: 2026-01-01\n---\nsmall-body",
        )
        monkeypatch.setattr(
            ingest,
            "_search_related_pages",
            lambda *_args, **_kwargs: [too_large, complete],
        )

        context = ingest._build_focused_context(
            {
                "type": "create",
                "filename": "memory/new.md",
                "keywords": ["memory"],
            },
            "raw",
            max_bytes=256,
        )

        assert "[[complete]]" in context
        assert "small-body" in context
        assert "[[too-large]]" not in context
        assert "L" * 20 not in context

    def test_generation_capacity_failure_is_not_retried(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from chronovisor.core.jobs import job_store
        from chronovisor.ingest import ingest

        calls = 0

        def fail_once(*_args, **_kwargs):
            nonlocal calls
            calls += 1
            raise RuntimeError(
                "ingest generation capacity_unavailable: measured memory admission failed"
            )

        monkeypatch.setattr(ingest, "_generate_one_with_progress", fail_once)
        job_id = job_store.create(processor="test").job_id

        with pytest.raises(RuntimeError, match="capacity_unavailable"):
            ingest._generate_local_operations(
                [{"type": "create", "filename": "memory/new.md"}],
                content="raw",
                raw_keywords=None,
                source_raw="raw.md",
                job_id=job_id,
                frontier_feedback=None,
            )
        assert calls == 1


class TestRecallMetadataStructuredSession:
    def test_live_metadata_reuses_larger_resident_context(
        self, isolated_wiki: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from contextlib import contextmanager

        from chronovisor.core import ollama
        from chronovisor.core.runtime_config import IngestConfig
        from chronovisor.decision.local_structured import LocalStructuredResult
        from chronovisor.ingest import ingest

        planned: list[dict] = []
        sessions: list[dict] = []
        leases: list[bool] = []

        @contextmanager
        def fake_lease(*, exclusive: bool = False):
            leases.append(exclusive)
            yield

        class FakeSession:
            def __init__(self, **kwargs):
                sessions.append(kwargs)

            def run(self, _prompt, _schema):
                return LocalStructuredResult(
                    ok=True,
                    model="ornith:test",
                    value={
                        "summary": "summary",
                        "recall_questions": ["question"],
                    },
                )

        monkeypatch.setattr(ingest, "is_available", lambda: True)
        monkeypatch.setattr(
            ingest,
            "load_ingest_config",
            lambda: IngestConfig(
                model="ornith:test",
                num_ctx=32768,
                max_num_ctx=262144,
            ),
        )
        monkeypatch.setattr(ingest.ollama_runtime, "model_resource_lease", fake_lease)
        monkeypatch.setattr(
            ollama,
            "unload_named_model",
            lambda model: pytest.fail(f"metadata shrank/reloaded resident {model}"),
        )

        def plan(models, **kwargs):
            planned.append({"models": models, **kwargs})
            return ollama.ModelResidencyPlan(
                num_ctx=32768,
                max_resident_models=1,
                capacity_bytes=80 * ollama.GIB,
                reserve_bytes=16 * ollama.GIB,
                available_bytes=80 * ollama.GIB,
                total_bytes=128 * ollama.GIB,
                estimated_model_bytes=(("ornith:test", 30 * ollama.GIB),),
                role_contexts=(("ornith:test", 65536),),
                resident_models=("ornith:test",),
                calibrated_models=("ornith:test",),
                source="test",
                reuse_larger_context=True,
            )

        monkeypatch.setattr(ollama, "plan_model_residency", plan)
        monkeypatch.setattr(ingest, "LocalStructuredSession", FakeSession)

        result = ingest._generate_recall_metadata("Title", "Body", "page-id")

        assert result == {
            "summary": "summary",
            "recall_questions": ["question"],
        }
        assert planned[0]["num_ctx"] == 32768
        assert planned[0]["reuse_larger_context"] is True
        assert leases == [True]
        assert sessions[0]["num_ctx"] == 65536

    def test_malformed_json_is_repaired_in_same_session(
        self, isolated_wiki: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from chronovisor.ingest import ingest

        monkeypatch.setattr(ingest, "is_available", lambda: True)
        transport = _QueueStructuredTransport(
            '{"summary":',
            json.dumps(
                {
                    "summary": "短い要約",
                    "recall_questions": ["何を決めた?", "次に何をする?"],
                },
                ensure_ascii=False,
            ),
        )

        result = ingest._generate_recall_metadata(
            "Title", "Body", "page-id", transport=transport
        )

        assert result == {
            "summary": "短い要約",
            "recall_questions": ["何を決めた?", "次に何をする?"],
        }
        assert len(transport.requests) == 2
        assert '"keyword":"parse"' in transport.requests[1].messages[-1]["content"]

    def test_three_invalid_responses_use_deterministic_fallback(
        self, isolated_wiki: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from chronovisor.ingest import ingest

        monkeypatch.setattr(ingest, "is_available", lambda: True)
        transport = _QueueStructuredTransport(
            '{"summary":',
            "[]",
            '{"summary":"x","recall_questions":[]}',
        )
        expected = ingest._fallback_recall_metadata("Title", "Body", "page-id")

        result = ingest._generate_recall_metadata(
            "Title", "Body", "page-id", transport=transport
        )

        assert result == expected
        assert len(transport.requests) == 3

    def test_transport_exception_uses_deterministic_fallback(
        self, isolated_wiki: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from chronovisor.ingest import ingest

        monkeypatch.setattr(ingest, "is_available", lambda: True)
        transport = _QueueStructuredTransport(RuntimeError("ollama offline"))
        expected = ingest._fallback_recall_metadata("Title", "Body", "page-id")

        result = ingest._generate_recall_metadata(
            "Title", "Body", "page-id", transport=transport
        )

        assert result == expected
        assert len(transport.requests) == 1


# ---------------------------------------------------------------------------
# Apply prepare phase: all-or-nothing collision (R3-Critical)
# ---------------------------------------------------------------------------


class TestApplyPreparePhase:
    def test_collision_converts_without_blocking_batch(
        self, isolated_wiki: Path
    ) -> None:
        """A create for an existing page_id is treated as an update, so it
        does not strand unrelated successful operations in the same batch."""
        existing = isolated_wiki / "pages" / "a" / "blocking.md"
        existing.parent.mkdir(parents=True, exist_ok=True)
        existing.write_text(
            "---\ntitle: existing\nupdated: 2026-01-01\n---\noriginal\n"
        )

        ops = [
            {
                "type": "create",
                "filename": "fresh/safe-page.md",
                "content": "---\ntitle: Fresh\nupdated: 2026-04-28\n---\nbody",
            },
            {
                "type": "create",
                "filename": "other/blocking.md",  # different folder, same stem
                "content": "---\ntitle: Dup\nupdated: 2026-04-28\n---\nx",
            },
        ]
        created, updated = _apply_operations(ops)

        assert created == ["safe-page"]
        assert updated == ["blocking"]
        assert (isolated_wiki / "pages" / "fresh" / "safe-page.md").exists()
        text = existing.read_text()
        assert "original" in text
        assert "\nx\n" in text
        assert "title: Dup" not in text

    def test_duplicate_page_id_within_batch_rejected(self, isolated_wiki: Path) -> None:
        ops = [
            {
                "type": "create",
                "filename": "a/dup.md",
                "content": "---\ntitle: A\nupdated: 2026-04-28\n---\nbody1",
            },
            {
                "type": "create",
                "filename": "b/dup.md",
                "content": "---\ntitle: B\nupdated: 2026-04-28\n---\nbody2",
            },
        ]
        with pytest.raises(IngestApplyError, match="duplicate page_id"):
            _apply_operations(ops)
        # Neither file written.
        assert not (isolated_wiki / "pages" / "a" / "dup.md").exists()
        assert not (isolated_wiki / "pages" / "b" / "dup.md").exists()

    @pytest.mark.parametrize("op_type", ["create", "update"])
    def test_system_page_id_is_reserved_from_ingest(
        self, isolated_wiki: Path, op_type: str
    ) -> None:
        system_page = isolated_wiki / "system" / "lessons-learned.md"
        system_page.write_text(
            "---\ntitle: System Lessons\nupdated: 2026-07-11\n---\ncanonical\n"
        )
        op = {
            "type": op_type,
            "filename": "generated/lessons-learned.md",
            "content": "---\ntitle: Generated\nupdated: 2026-07-11\n---\nbody",
        }

        with pytest.raises(IngestApplyError, match="reserved system page_id"):
            _apply_operations([op])

        assert system_page.read_text().endswith("canonical\n")
        assert not (
            isolated_wiki / "pages" / "generated" / "lessons-learned.md"
        ).exists()

    def test_legacy_reserved_proposal_is_neither_reused_nor_applied(
        self,
        isolated_wiki: Path,
    ) -> None:
        from chronovisor.decision.decision_lane_prompts import (
            validate_ingest_proposal_envelope,
        )
        from chronovisor.ingest import ingest

        system_page = isolated_wiki / "system" / "lessons-learned.md"
        canonical = "---\ntitle: System Lessons\nupdated: 2026-07-11\n---\ncanonical\n"
        system_page.write_text(canonical, encoding="utf-8")
        target = (
            isolated_wiki / "pages" / "generated" / "lessons-learned.md"
        ).resolve()
        planned = [
            ingest.PreparedIngestOperation(
                op_type="create",
                path=target,
                page_id="lessons-learned",
                new_body=("---\ntitle: Generated\nupdated: 2026-07-11\n---\nunsafe\n"),
                previous_text=None,
                source_operation_index=0,
                source_operation_type="create",
                source_filename="generated/lessons-learned.md",
            )
        ]
        operations = [
            {
                "type": "create",
                "filename": "generated/lessons-learned.md",
                "content": planned[0].new_body,
            }
        ]
        raw_content = "legacy artifact with a reserved target"
        source_key = ingest._ingest_source_key(raw_content, None)
        proposal = ingest._build_ingest_frontier_proposal(
            raw_content=raw_content,
            raw_keywords=None,
            source_raw="semantic-legacy-reserved.md",
            operations=operations,
            planned=planned,
            link_totals={"resolved": 0, "rewritten": 0, "unwrapped": 0},
        )
        proposal_path, _review_path = ingest._ingest_artifact_paths(source_key)
        proposal_sha256 = ingest._canonical_json_sha256(proposal)
        assert validate_ingest_proposal_envelope(proposal)
        ingest._write_ingest_artifact(
            proposal_path,
            {
                "schema_version": ingest.INGEST_FRONTIER_ARTIFACT_SCHEMA_VERSION,
                "kind": "ingest_frontier_proposal_artifact",
                "source_key": source_key,
                "proposal_sha256": proposal_sha256,
                "proposal": proposal,
            },
        )

        assert (
            ingest._load_ingest_proposal(
                proposal_path,
                source_key=source_key,
                raw_content=raw_content,
            )
            is None
        )
        with pytest.raises(IngestApplyError, match="reserved system page_id"):
            ingest._apply_prepared_operations(planned)

        assert system_page.read_text(encoding="utf-8") == canonical
        assert not target.exists()

    def test_rollback_on_write_failure_restores_previous_state(
        self,
        isolated_wiki: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Force atomic_write to fail on the second op and verify the first
        op's effect is rolled back."""
        # Seed a page so op[0] is a real update we can roll back.
        target = isolated_wiki / "pages" / "x" / "page.md"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("---\ntitle: X\nupdated: 2026-01-01\n---\noriginal body\n")
        original = target.read_text()

        # First call: real write (the update succeeds). Second call: explode
        # (the create's write fails). Subsequent calls succeed so the
        # rollback path can actually run.
        from chronovisor.core import link_fix

        real_write = link_fix.atomic_write
        call_count = {"n": 0}

        def flaky_write(path, content):
            call_count["n"] += 1
            if call_count["n"] == 2:
                raise OSError("simulated disk full")
            real_write(path, content)

        monkeypatch.setattr(link_fix, "atomic_write", flaky_write)

        ops = [
            {
                "type": "update",
                "filename": "page.md",
                "content": "addendum block 1",
            },
            {
                "type": "create",
                "filename": "y/new.md",
                "content": "---\ntitle: Y\nupdated: 2026-04-28\n---\nbody",
            },
        ]
        with pytest.raises(IngestApplyError, match="apply write failed"):
            _apply_operations(ops)

        # Update was rolled back to the original text.
        assert target.read_text() == original
        # Create never wrote (write failed before append).
        assert not (isolated_wiki / "pages" / "y" / "new.md").exists()

    def test_casefold_collision_converts_to_update(self, isolated_wiki: Path) -> None:
        """``Foo.md`` and ``foo.md`` resolve to the same inode on
        case-insensitive macOS filesystems. We route the create into the
        existing page rather than writing a second logical duplicate."""
        existing = isolated_wiki / "pages" / "a" / "Foo.md"
        existing.parent.mkdir(parents=True, exist_ok=True)
        existing.write_text("---\ntitle: cased\nupdated: 2026-01-01\n---\nbody\n")
        ops = [
            {
                "type": "create",
                "filename": "b/foo.md",  # lowercase variant
                "content": "---\ntitle: dup\nupdated: 2026-04-28\n---\nx",
            }
        ]
        created, updated = _apply_operations(ops)

        assert created == []
        assert updated == ["Foo"]
        text = existing.read_text()
        assert "body" in text
        assert "\nx\n" in text
        assert not (isolated_wiki / "pages" / "b" / "foo.md").exists()


# ---------------------------------------------------------------------------
# chronovisor_ingest now persists raw + uses orchestrator (R3-Medium)
# ---------------------------------------------------------------------------


class TestWikiIngestRouting:
    def test_chronovisor_ingest_writes_raw_and_consults_orchestrator(
        self, isolated_wiki: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from chronovisor.hosts import server
        from chronovisor.ingest import orchestrator

        # The MCP tool object wraps the function; call .fn.
        tool_fn = (
            server.chronovisor_ingest.fn
            if hasattr(server.chronovisor_ingest, "fn")
            else server.chronovisor_ingest
        )

        # Patch RAW_DIR in server (it grabbed the path at import time).
        monkeypatch.setattr(raw_record, "RAW_DIR", isolated_wiki / "raw")

        # Stub the orchestrator path so we don't actually start ingest.
        captured = {"called": False, "force": None}

        def fake_run(force: bool = False) -> dict:
            captured["called"] = True
            captured["force"] = force
            return {"triggered": False, "reason": "test stub"}

        monkeypatch.setattr(orchestrator, "run_pending_ingest", fake_run)

        result = tool_fn("hello world content")
        # Must have written exactly one new raw file with the supplied content.
        raws = list((isolated_wiki / "raw").glob("*.md"))
        assert len(raws) == 1
        assert raws[0].read_text() == "hello world content"
        # Must have consulted the orchestrator with force=True (default).
        assert captured["called"] is True
        assert captured["force"] is True
        assert "test stub" in result


class TestWikiSaveRawRouting:
    def test_idempotency_key_reuses_first_complete_raw(
        self, isolated_wiki: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from chronovisor.hosts import server
        from chronovisor.ingest import orchestrator

        tool_fn = (
            server.chronovisor_record.fn
            if hasattr(server.chronovisor_record, "fn")
            else server.chronovisor_record
        )
        monkeypatch.setattr(raw_record, "RAW_DIR", isolated_wiki / "raw")
        monkeypatch.setattr(
            orchestrator, "should_ingest", lambda: (False, "below threshold")
        )
        key = "codex-0123456789abcdef01234567-from0-to5"

        first = json.loads(
            tool_fn("first complete payload", trigger_ingest=False, idempotency_key=key)
        )
        second = json.loads(
            tool_fn("first complete payload", trigger_ingest=False, idempotency_key=key)
        )

        assert first["saved"] == f"save-{key}.md"
        assert second["saved"] == first["saved"]
        assert first["deduplicated"] is False
        assert second["deduplicated"] is True
        assert (
            isolated_wiki / "raw" / first["saved"]
        ).read_text() == "first complete payload"
        assert len(list((isolated_wiki / "raw").glob("*.md"))) == 1

    def test_idempotency_key_rejects_different_unverified_payload(
        self, isolated_wiki: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from chronovisor.hosts import server
        from chronovisor.ingest import orchestrator

        tool_fn = (
            server.chronovisor_record.fn
            if hasattr(server.chronovisor_record, "fn")
            else server.chronovisor_record
        )
        monkeypatch.setattr(raw_record, "RAW_DIR", isolated_wiki / "raw")
        monkeypatch.setattr(
            orchestrator, "should_ingest", lambda: (False, "below threshold")
        )
        key = "codex-0123456789abcdef01234567-from0-to5"

        tool_fn("first complete payload", trigger_ingest=False, idempotency_key=key)
        with pytest.raises(RuntimeError, match="idempotency key collision"):
            tool_fn("corrupt retry payload", trigger_ingest=False, idempotency_key=key)

        target = isolated_wiki / "raw" / f"save-{key}.md"
        assert target.read_text() == "first complete payload"

    def test_idempotency_key_accepts_different_self_verified_retry_receipt(
        self, isolated_wiki: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from chronovisor.core.save_transaction import (
            attach_save_transaction_marker,
            make_save_transaction,
        )
        from chronovisor.hosts import server
        from chronovisor.ingest import orchestrator

        tool_fn = (
            server.chronovisor_record.fn
            if hasattr(server.chronovisor_record, "fn")
            else server.chronovisor_record
        )
        monkeypatch.setattr(raw_record, "RAW_DIR", isolated_wiki / "raw")
        monkeypatch.setattr(
            orchestrator, "should_ingest", lambda: (False, "below threshold")
        )
        transaction = make_save_transaction(
            host="codex",
            session_file=isolated_wiki / "session.jsonl",
            session_id="session-1",
            after_line=0,
            until_line=5,
        )
        first_content = attach_save_transaction_marker(
            transaction, "first writer output"
        )
        retry_content = attach_save_transaction_marker(
            transaction, "different retry output"
        )

        first = json.loads(
            tool_fn(
                first_content,
                trigger_ingest=False,
                idempotency_key=transaction.idempotency_key,
            )
        )
        retry = json.loads(
            tool_fn(
                retry_content,
                trigger_ingest=False,
                idempotency_key=transaction.idempotency_key,
            )
        )

        assert retry["saved"] == first["saved"]
        assert retry["deduplicated"] is True
        assert (isolated_wiki / "raw" / first["saved"]).read_text() == first_content

    def test_idempotency_key_rejects_corrupt_existing_receipt(
        self, isolated_wiki: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from chronovisor.core.save_transaction import (
            attach_save_transaction_marker,
            make_save_transaction,
        )
        from chronovisor.hosts import server
        from chronovisor.ingest import orchestrator

        tool_fn = (
            server.chronovisor_record.fn
            if hasattr(server.chronovisor_record, "fn")
            else server.chronovisor_record
        )
        monkeypatch.setattr(raw_record, "RAW_DIR", isolated_wiki / "raw")
        monkeypatch.setattr(
            orchestrator, "should_ingest", lambda: (False, "below threshold")
        )
        transaction = make_save_transaction(
            host="codex",
            session_file=isolated_wiki / "session.jsonl",
            session_id="session-1",
            after_line=0,
            until_line=5,
        )
        content = attach_save_transaction_marker(transaction, "complete payload")
        result = json.loads(
            tool_fn(
                content,
                trigger_ingest=False,
                idempotency_key=transaction.idempotency_key,
            )
        )
        target = isolated_wiki / "raw" / result["saved"]
        target.write_text(content.replace("complete payload", "corrupt payload"))

        with pytest.raises(RuntimeError, match="different or corrupt"):
            tool_fn(
                content,
                trigger_ingest=False,
                idempotency_key=transaction.idempotency_key,
            )

    def test_trigger_ingest_false_defers_threshold_ingest(
        self, isolated_wiki: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from chronovisor.hosts import server
        from chronovisor.ingest import orchestrator

        tool_fn = (
            server.chronovisor_record.fn
            if hasattr(server.chronovisor_record, "fn")
            else server.chronovisor_record
        )
        monkeypatch.setattr(raw_record, "RAW_DIR", isolated_wiki / "raw")
        monkeypatch.setattr(
            orchestrator, "should_ingest", lambda: (True, "threshold met")
        )

        captured = {"called": False}

        def fake_run_pending_ingest() -> dict:
            captured["called"] = True
            return {"triggered": True}

        monkeypatch.setattr(orchestrator, "run_pending_ingest", fake_run_pending_ingest)

        result = json.loads(tool_fn("hello world content", trigger_ingest=False))

        assert result["ingest_pending"] is True
        assert result["ingest_deferred"] is True
        assert "ingest_triggered" not in result
        assert captured["called"] is False

    def test_save_raw_filename_includes_readable_keyword_slug(
        self, isolated_wiki: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from chronovisor.hosts import server
        from chronovisor.ingest import orchestrator

        tool_fn = (
            server.chronovisor_record.fn
            if hasattr(server.chronovisor_record, "fn")
            else server.chronovisor_record
        )
        monkeypatch.setattr(raw_record, "RAW_DIR", isolated_wiki / "raw")
        monkeypatch.setattr(
            orchestrator, "should_ingest", lambda: (False, "below threshold")
        )

        result = json.loads(
            tool_fn(
                "body",
                session_id="codex-019e80cb-8df9-7aa1-ba75-cc58c6f759ae",
                keywords=["chronovisor", "dashboard", "self-heal"],
                trigger_ingest=False,
            )
        )

        assert result["saved"].startswith("20")
        assert "-codex-chronovisor-dashboard-self-heal-" in result["saved"]
        assert result["saved"].endswith(".md")
        assert result["raw_slug"] == "chronovisor-dashboard-self-heal"

    def test_save_raw_filename_falls_back_to_content_slug(
        self, isolated_wiki: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from chronovisor.hosts import server
        from chronovisor.ingest import orchestrator

        tool_fn = (
            server.chronovisor_record.fn
            if hasattr(server.chronovisor_record, "fn")
            else server.chronovisor_record
        )
        monkeypatch.setattr(raw_record, "RAW_DIR", isolated_wiki / "raw")
        monkeypatch.setattr(
            orchestrator, "should_ingest", lambda: (False, "below threshold")
        )

        result = json.loads(
            tool_fn(
                "# Codex Session Memory Save\n\n## Memory\n\nDashboard filename cleanup",
                session_id="codex-019e80cb-8df9-7aa1-ba75-cc58c6f759ae",
                keywords=[],
                trigger_ingest=False,
            )
        )

        assert "-codex-dashboard-filename-cleanup-" in result["saved"]


# ---------------------------------------------------------------------------
# R4-Critical: log failures must not break rollback inclusion
# ---------------------------------------------------------------------------


class TestLogFailuresDontBreakRollback:
    def test_shared_lock_entry_rechecks_update_preimage(
        self, isolated_wiki: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A correction committed after ingest prepare must not be overwritten."""
        from contextlib import contextmanager

        from chronovisor.core import page_mutation

        target = isolated_wiki / "pages" / "x" / "page.md"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("---\ntitle: X\nupdated: 2026-01-01\n---\noriginal\n")
        correction = "---\ntitle: X\nupdated: 2026-07-11\n---\nfrontier correction\n"

        @contextmanager
        def correction_wins_before_ingest_commit():
            target.write_text(correction)
            yield

        monkeypatch.setattr(
            page_mutation,
            "chronovisor_mutation_lock",
            correction_wins_before_ingest_commit,
        )

        with pytest.raises(IngestApplyError, match="page changed before ingest apply"):
            _apply_operations(
                [{"type": "update", "filename": "page.md", "content": "stale addendum"}]
            )

        assert target.read_text() == correction

    def test_log_failure_does_not_drop_entry_from_rollback_set(
        self, isolated_wiki: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """If _append_log raised AFTER atomic_write succeeded but BEFORE the
        previous code did `written.append(entry)`, the page would be
        modified on disk yet absent from the rollback list — silently
        partial state. Reordering plus _safe_log fixes this."""
        from chronovisor.ingest import ingest as ingest_mod

        # Seed an existing page so op[0] becomes a real update we can roll back.
        target = isolated_wiki / "pages" / "x" / "page.md"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("---\ntitle: X\nupdated: 2026-01-01\n---\noriginal\n")
        original = target.read_text()

        # Make _append_log raise on every call: that used to mask the
        # rollback set; now _safe_log swallows it.
        def boom(*_a, **_kw):
            raise RuntimeError("simulated log disk failure")

        monkeypatch.setattr(ingest_mod, "_append_log", boom)

        # Make the second write fail to trigger rollback.
        from chronovisor.core import link_fix

        real_write = link_fix.atomic_write
        n = {"calls": 0}

        def flaky(path, content):
            n["calls"] += 1
            if n["calls"] == 2:
                raise OSError("disk full")
            real_write(path, content)

        monkeypatch.setattr(link_fix, "atomic_write", flaky)

        ops = [
            {"type": "update", "filename": "page.md", "content": "addendum"},
            {
                "type": "create",
                "filename": "y/new.md",
                "content": "---\ntitle: Y\nupdated: 2026-04-28\n---\nbody",
            },
        ]
        with pytest.raises(IngestApplyError, match="apply write failed"):
            _apply_operations(ops)

        # Update was rolled back even though _append_log raised on every call.
        assert target.read_text() == original

    def test_rollback_skips_when_other_writer_modified_file(
        self, isolated_wiki: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """CAS check: if another process modified the file between our
        write and our rollback (e.g. chronovisor_apply ran), we must NOT clobber
        their change with our pre-batch snapshot. Skip and log."""
        from chronovisor.core import link_fix

        target = isolated_wiki / "pages" / "x" / "page.md"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("---\ntitle: X\nupdated: 2026-01-01\n---\noriginal\n")

        real_write = link_fix.atomic_write
        n = {"calls": 0}

        # After the first successful write, a "rogue" writer overwrites the
        # file with their own content. Then op 2 fails and rollback runs.
        def rogue_then_fail(path, content):
            n["calls"] += 1
            if n["calls"] == 1:
                real_write(path, content)
                # Simulate a concurrent writer changing the file.
                path.write_text(
                    "---\ntitle: rogue\nupdated: 2026-04-28\n---\nrogue body\n"
                )
                return
            if n["calls"] == 2:
                raise OSError("disk full")
            real_write(path, content)

        monkeypatch.setattr(link_fix, "atomic_write", rogue_then_fail)

        ops = [
            {"type": "update", "filename": "page.md", "content": "addendum"},
            {
                "type": "create",
                "filename": "y/new.md",
                "content": "---\ntitle: Y\nupdated: 2026-04-28\n---\nbody",
            },
        ]
        with pytest.raises(IngestApplyError, match="apply write failed"):
            _apply_operations(ops)

        # Rogue writer's content should remain (CAS check skipped rollback).
        assert "rogue body" in target.read_text()


# ---------------------------------------------------------------------------
# R4-High: raw filename collision avoidance
# ---------------------------------------------------------------------------


class TestRawFilenameCollision:
    def test_allocate_raw_path_returns_unique_paths_under_contention(
        self, isolated_wiki: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:

        # Patch RAW_DIR in server so allocate writes into the isolated wiki.
        monkeypatch.setattr(raw_record, "RAW_DIR", isolated_wiki / "raw")

        paths = {raw_record.allocate_raw_path() for _ in range(50)}
        assert len(paths) == 50  # all unique
        for p in paths:
            assert p.exists()
            assert p.parent == isolated_wiki / "raw"

    def test_allocate_raw_path_creates_raw_dir_if_missing(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:

        raw_dir = tmp_path / "wiki" / "raw"
        monkeypatch.setattr(raw_record, "RAW_DIR", raw_dir)

        path = raw_record.allocate_raw_path(prefix="codex", topic_slug="readable-name")

        assert path.exists()
        assert path.parent == raw_dir
        assert "-codex-readable-name-" in path.name


# ---------------------------------------------------------------------------
# R4-High: chronovisor_ingest force=True bypasses threshold
# ---------------------------------------------------------------------------


class TestWikiIngestForce:
    def test_force_triggers_below_threshold(
        self, isolated_wiki: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from chronovisor.ingest import ingest as ingest_mod
        from chronovisor.ingest import orchestrator

        # 1 pending raw — far below INGEST_THRESHOLD (5). Without force, the
        # orchestrator should refuse; with force=True, it should trigger.
        (isolated_wiki / "raw" / "single.md").write_text("body")

        monkeypatch.setattr(orchestrator, "is_available", lambda: True)

        def _noop_run_ingest(
            content, job_id, on_complete=None, on_finally=None, *, metadata=None
        ):
            if on_complete:
                on_complete()
            if on_finally:
                on_finally(failed=False, triage_failed=False)

        monkeypatch.setattr(ingest_mod, "run_ingest", _noop_run_ingest)

        deferred = orchestrator.run_pending_ingest(force=False)
        assert deferred["triggered"] is False

        forced = orchestrator.run_pending_ingest(force=True)
        assert forced["triggered"] is True
        assert "force=True" in forced["reason"]


# ---------------------------------------------------------------------------
# R4-Medium: filename schema hardening
# ---------------------------------------------------------------------------


class TestFilenameSchemaStrict:
    @staticmethod
    def _create(filename: str) -> dict:
        return {
            "type": "create",
            "filename": filename,
            "title": "Filename test",
            "keywords": ["filename"],
            "summary": "Validate the create filename contract.",
        }

    def test_control_char_rejected(self) -> None:
        from chronovisor.ingest.ingest import _validate_triage_plan

        for c in ("\x00", "\n", "\t", "\x07", "\x7f"):
            assert _validate_triage_plan([self._create(f"foo{c}bar.md")]) is None, c

    def test_long_filename_rejected(self) -> None:
        from chronovisor.ingest.ingest import _validate_triage_plan

        long_name = "a" * 250 + ".md"
        assert _validate_triage_plan([self._create(long_name)]) is None

    def test_non_kebab_case_rejected(self) -> None:
        from chronovisor.ingest.ingest import _validate_triage_plan

        for bad in (
            "Foo.md",  # uppercase
            "snake_case.md",  # underscore
            "a/b/c.md",  # nested folders
            "a..md",  # consecutive dots-ish
            "-leading.md",  # leading dash
            "trailing-.md",  # trailing dash before suffix
        ):
            assert _validate_triage_plan([self._create(bad)]) is None, bad

    def test_kebab_with_required_folder_accepted(self) -> None:
        from chronovisor.ingest.ingest import _validate_triage_plan

        for good in ("ai/foo.md", "ai/career-note"):
            out = _validate_triage_plan([self._create(good)])
            assert out is not None, good


# ---------------------------------------------------------------------------
# R4-Medium: Unicode NFC/NFD collision detection
# ---------------------------------------------------------------------------


class TestUnicodeCollision:
    def test_nfc_vs_nfd_treated_as_same_page(self, isolated_wiki: Path) -> None:
        """café (NFC, 4 chars) vs café (NFD, 5 chars: e + combining acute)
        live as the same logical page on macOS APFS. Both should map to one
        normalized key for collision detection.

        Note: validation in _validate_triage_plan rejects non-ASCII so this
        is exercised at the apply layer. We bypass triage and call apply
        directly with a non-ASCII filename to confirm the collision logic
        itself catches it. (A real plan would never reach this case because
        _validate_triage_plan filters non-ASCII first — that's defense in
        depth.)
        """
        import unicodedata

        from chronovisor.ingest.ingest import _normalize_for_collision

        nfc = unicodedata.normalize("NFC", "café")
        nfd = unicodedata.normalize("NFD", "café")
        assert nfc != nfd  # bytes differ
        assert _normalize_for_collision(nfc) == _normalize_for_collision(nfd)
        # Casefold also handled.
        assert _normalize_for_collision("CAFÉ") == _normalize_for_collision("café")


# ---------------------------------------------------------------------------
# R4-Medium: rebuild_index failure is non-fatal
# ---------------------------------------------------------------------------


class TestRebuildIndexNonFatal:
    def test_rebuild_index_error_does_not_block_completion(
        self, isolated_wiki: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """index.md is a derived artifact. If rebuild fails after pages
        have been written, we must still report COMPLETED and call
        on_complete — otherwise raws stay pending and retry will collide
        on every page we already created."""
        from chronovisor.core import jobs
        from chronovisor.ingest import ingest as ingest_mod

        monkeypatch.setattr(ingest_mod, "is_available", lambda: True)
        monkeypatch.setattr(
            ingest_mod,
            "_triage",
            lambda _content: [
                {"type": "create", "filename": "ai/foo.md", "title": "Foo"}
            ],
        )
        monkeypatch.setattr(
            ingest_mod,
            "_generate_one",
            lambda _op, _raw, **_kw: {
                "type": "create",
                "filename": "ai/foo.md",
                "content": ("---\ntitle: Foo\nupdated: 2026-04-28\n---\nbody"),
            },
        )

        def boom() -> None:
            raise RuntimeError("simulated rebuild failure")

        monkeypatch.setattr(ingest_mod, "_rebuild_index", boom)

        on_complete_calls = []
        job = jobs.job_store.create(processor="ollama")
        ingest_mod.run_ingest(
            "raw",
            job.job_id,
            on_complete=lambda: on_complete_calls.append(True),
            frontier_reviewer=ingest_mod._run_ingest_frontier_review,
        )

        finished = jobs.job_store.get(job.job_id)
        assert finished.status == jobs.JobStatus.COMPLETED, finished.error
        assert finished.pages_created == ["foo"]
        assert on_complete_calls == [True]
        # Page is on disk despite rebuild failure.
        assert (isolated_wiki / "pages" / "ai" / "foo.md").exists()

    def test_real_rebuild_index_io_failure_still_completes(
        self, isolated_wiki: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Stronger version: don't mock _rebuild_index — let it actually run,
        but make INDEX_FILE be a directory so the real write_text raises
        IsADirectoryError. We still expect COMPLETED + on_complete."""
        from chronovisor.core import jobs
        from chronovisor.ingest import ingest as ingest_mod

        # Replace INDEX_FILE with a directory (cannot write_text).
        idx_path = isolated_wiki / "index.md"
        idx_path.mkdir(parents=True, exist_ok=False)
        monkeypatch.setattr(ingest_mod, "INDEX_FILE", idx_path)

        monkeypatch.setattr(ingest_mod, "is_available", lambda: True)
        monkeypatch.setattr(
            ingest_mod,
            "_triage",
            lambda _content: [
                {"type": "create", "filename": "ai/bar.md", "title": "Bar"}
            ],
        )
        monkeypatch.setattr(
            ingest_mod,
            "_generate_one",
            lambda _op, _raw, **_kw: {
                "type": "create",
                "filename": "ai/bar.md",
                "content": ("---\ntitle: Bar\nupdated: 2026-04-28\n---\nbody"),
            },
        )

        on_complete_calls = []
        job = jobs.job_store.create(processor="ollama")
        ingest_mod.run_ingest(
            "raw",
            job.job_id,
            on_complete=lambda: on_complete_calls.append(True),
            frontier_reviewer=ingest_mod._run_ingest_frontier_review,
        )

        finished = jobs.job_store.get(job.job_id)
        assert finished.status == jobs.JobStatus.COMPLETED, finished.error
        assert on_complete_calls == [True]


# ---------------------------------------------------------------------------
# R5: parallel raw allocation
# ---------------------------------------------------------------------------


class TestRawAllocationParallel:
    def test_save_publish_is_invisible_to_ingest_until_content_is_complete(
        self, isolated_wiki: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Ingest must never observe the zero-byte reservation window."""
        import threading
        from concurrent.futures import ThreadPoolExecutor

        from chronovisor.hosts import server
        from chronovisor.ingest import orchestrator

        monkeypatch.setattr(raw_record, "RAW_DIR", isolated_wiki / "raw")
        tool_fn = (
            server.chronovisor_record.fn
            if hasattr(server.chronovisor_record, "fn")
            else server.chronovisor_record
        )
        real_link = raw_record.link_raw_no_replace
        ready_to_publish = threading.Event()
        allow_publish = threading.Event()

        def paused_link(staging: Path, target: Path) -> None:
            # _publish_raw reaches this point only after write + fsync.
            ready_to_publish.set()
            assert allow_publish.wait(5)
            real_link(staging, target)

        monkeypatch.setattr(raw_record, "link_raw_no_replace", paused_link)
        content = "complete raw payload\n" * 100

        with ThreadPoolExecutor(max_workers=1) as pool:
            future = pool.submit(
                tool_fn,
                content,
                "codex-test-session",
                ["atomic-publish"],
                False,
            )
            assert ready_to_publish.wait(5)
            try:
                assert orchestrator.get_pending_raw_files() == []
                assert list((isolated_wiki / "raw").glob("*.md")) == []
                staging = list((isolated_wiki / "raw").glob(".*.tmp"))
                assert len(staging) == 1
                staged_content = staging[0].read_text()
                assert staged_content.endswith(content)
                assert "raw_keywords: [atomic-publish]" in staged_content
            finally:
                allow_publish.set()
            result = json.loads(future.result(timeout=5))

        published = isolated_wiki / "raw" / result["saved"]
        assert published.read_text() == staged_content
        assert orchestrator.get_pending_raw_files() == [published]
        assert list((isolated_wiki / "raw").glob(".*.tmp")) == []

    def test_concurrent_threads_get_unique_paths(
        self, isolated_wiki: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Real concurrency test: 50 threads racing into _allocate_raw_path
        must each receive a distinct path. The sequential test in R4 only
        proved the single-thread case."""
        import threading
        from concurrent.futures import ThreadPoolExecutor


        monkeypatch.setattr(raw_record, "RAW_DIR", isolated_wiki / "raw")

        N = 50
        barrier = threading.Barrier(N)

        def worker() -> Path:
            barrier.wait()  # all threads release at once → maximize collision
            return raw_record.allocate_raw_path()

        with ThreadPoolExecutor(max_workers=N) as ex:
            paths = list(ex.map(lambda _i: worker(), range(N)))

        assert len(set(paths)) == N
        for p in paths:
            assert p.exists()

    def test_session_id_with_traversal_chars_sanitized(
        self, isolated_wiki: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A malicious or malformed session_id ('../escape') must NOT let
        the raw file land outside RAW_DIR."""

        monkeypatch.setattr(raw_record, "RAW_DIR", isolated_wiki / "raw")

        path = raw_record.allocate_raw_path(prefix="../../etc/passwd")
        assert path.parent == (isolated_wiki / "raw")
        # Sanitizer collapses path-traversal chars to dashes.
        assert "../" not in path.name
        assert "/" not in path.name


# ---------------------------------------------------------------------------
# R5-Critical: post-apply _append_log failures don't override COMPLETED
# ---------------------------------------------------------------------------


class TestPostApplyLogSafety:
    def test_safe_log_actually_calls_append_log(
        self, isolated_wiki: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Regression test for the recursion bug Codex caught: a stray sed
        pass once rewrote the body of ``_safe_log`` to call itself, making
        every "safe" log silently no-op (and burn a recursion limit). All
        the atomicity tests passed only because nothing was ever logged.
        Verify that calling _safe_log with a working _append_log does in
        fact write through."""
        from chronovisor.ingest import ingest as ingest_mod

        captured: list[str] = []
        monkeypatch.setattr(ingest_mod, "_append_log", lambda msg: captured.append(msg))

        ingest_mod._safe_log("hello via safe_log")
        assert captured == ["hello via safe_log"]

    def test_log_failure_after_apply_still_completes_and_calls_on_complete(
        self, isolated_wiki: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The R5-Critical regression: after _apply_operations + COMPLETED
        was set, a raising _append_log used to fall through to the outer
        except, override status with FAILED, and skip on_complete. Pages
        persist but raws stay pending → next tick collides on every page.

        With _safe_log wrapping every post-apply log call, this can't
        happen. Verify by patching _append_log to raise on every call."""
        from chronovisor.core import jobs
        from chronovisor.ingest import ingest as ingest_mod

        monkeypatch.setattr(ingest_mod, "is_available", lambda: True)
        monkeypatch.setattr(
            ingest_mod,
            "_triage",
            lambda _content: [
                {"type": "create", "filename": "ai/baz.md", "title": "Baz"}
            ],
        )
        monkeypatch.setattr(
            ingest_mod,
            "_generate_one",
            lambda _op, _raw, **_kw: {
                "type": "create",
                "filename": "ai/baz.md",
                "content": ("---\ntitle: Baz\nupdated: 2026-04-28\n---\nbody"),
            },
        )

        def boom(*_a, **_kw):
            raise RuntimeError("simulated log disk failure")

        monkeypatch.setattr(ingest_mod, "_append_log", boom)

        on_complete_calls = []
        job = jobs.job_store.create(processor="ollama")
        ingest_mod.run_ingest(
            "raw",
            job.job_id,
            on_complete=lambda: on_complete_calls.append(True),
            frontier_reviewer=ingest_mod._run_ingest_frontier_review,
        )

        finished = jobs.job_store.get(job.job_id)
        assert finished.status == jobs.JobStatus.COMPLETED, finished.error
        assert finished.pages_created == ["baz"]
        assert on_complete_calls == [True]


# ---------------------------------------------------------------------------
# R5-Medium: filename schema split — legacy update should still work
# ---------------------------------------------------------------------------


class TestFilenameSchemaUpdateLeniency:
    def test_legacy_filename_accepted_for_update(self) -> None:
        from chronovisor.ingest.ingest import _validate_triage_plan

        # These would never pass the create regex, but for update we let
        # them through so legacy corpus pages remain updatable. The actual
        # existence check happens in _apply_operations.
        for legacy in (
            "Foo.md",
            "snake_case_page.md",
            "MixedCase/foo.md",
            "ai/UPPERCASE.md",
        ):
            out = _validate_triage_plan([{"type": "update", "filename": legacy}])
            assert out is not None, legacy

    def test_create_still_strict(self) -> None:
        from chronovisor.ingest.ingest import _validate_triage_plan

        # Same names rejected for create.
        for bad in ("Foo.md", "snake_case.md", "MixedCase/foo.md"):
            out = _validate_triage_plan([{"type": "create", "filename": bad}])
            assert out is None, bad

    def test_control_char_still_rejected_for_update(self) -> None:
        from chronovisor.ingest.ingest import _validate_triage_plan

        assert (
            _validate_triage_plan([{"type": "update", "filename": "foo\x00.md"}])
            is None
        )

    def test_traversal_rejected_for_both_op_types(self) -> None:
        from chronovisor.ingest.ingest import _validate_triage_plan

        for op_type in ("create", "update"):
            assert (
                _validate_triage_plan(
                    [{"type": op_type, "filename": "../../etc/passwd.md"}]
                )
                is None
            ), op_type


# ---------------------------------------------------------------------------
# R6: session bootstrap contract
# ---------------------------------------------------------------------------


class TestWikiInit:
    def test_returns_system_pages_with_status(
        self, isolated_wiki: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from chronovisor.core import ollama
        from chronovisor.hosts import server

        for page_id in ("user-profile", "current-state", "lessons-learned"):
            (isolated_wiki / "system" / f"{page_id}.md").write_text(
                f"---\ntitle: {page_id}\nupdated: 2026-04-28\n---\n"
                f"body for [[{page_id}]]\n"
            )
        (isolated_wiki / "raw" / "pending.md").write_text("raw")

        monkeypatch.setattr(server, "CHRONOVISOR_ROOT", isolated_wiki)
        monkeypatch.setattr(server, "RAW_DIR", isolated_wiki / "raw")
        monkeypatch.setattr(server, "SYSTEM_DIR", isolated_wiki / "system")
        monkeypatch.setattr(ollama, "is_available", lambda: False)

        payload = json.loads(server.chronovisor_init())

        assert payload["status"]["page_count"] == 0
        assert payload["status"]["raw_total"] == 1
        assert payload["status"]["raw_pending"] == 1
        assert payload["status"]["ollama_status"] == "stopped"
        assert set(payload["system_pages"]) == {
            "user-profile",
            "current-state",
            "lessons-learned",
        }
        assert (
            "body for [[user-profile]]"
            in payload["system_pages"]["user-profile"]["content"]
        )


# ---------------------------------------------------------------------------
# R6: protected regions in link extraction and auto-fix
# ---------------------------------------------------------------------------


class TestLinkFixProtectedRegions:
    def test_unclosed_fence_links_are_ignored_by_extractor(self) -> None:
        from chronovisor.core.link_fix import extract_targets

        text = "before [[real]]\n```python\nx = data[[1]]\ny = [[not-a-link]]\n"
        assert extract_targets(text, strip=True) == ["real"]

    def test_lint_replace_leaves_code_frontmatter_and_inline_code(self) -> None:
        from chronovisor.ingest.lint import _replace_link_in_content

        content = (
            "---\ntitle: [[ghost]]\n---\n"
            "body [[ghost#sec|Ghost Page]] and [[other]]\n"
            "`[[ghost]]`\n"
            "```python\nx = data[[ghost]]\n```\n"
        )
        new_content, count = _replace_link_in_content(content, "ghost", "real-page")

        assert count == 1
        assert "title: [[ghost]]" in new_content
        assert "`[[ghost]]`" in new_content
        assert "x = data[[ghost]]" in new_content
        assert "[[real-page#sec|Ghost Page]]" in new_content
        assert "[[other]]" in new_content

    def test_lint_plaintext_fallback_uses_alias(self) -> None:
        from chronovisor.ingest.lint import _replace_link_in_content

        new_content, count = _replace_link_in_content(
            "See [[ghost|visible name]] and [[ghost#old]].",
            "ghost",
            None,
        )

        assert count == 2
        assert new_content == "See visible name and ghost."
