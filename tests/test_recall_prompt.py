from __future__ import annotations

import json

import pytest

from chronovisor.recall.recall_prompt import normalize_recall_prompt


def test_known_question_reply_envelope_keeps_human_meaning_without_transport_id() -> None:
    prompt = (
        '<send_user_message_question_reply>\n'
        '[{"questionItemId":"[\\"request_user_input_async\\",\\"call_123\\",0]",'
        '"question":"改善計画の渡し方はどちらがよいですか？",'
        '"answer":"既存と同じ _handoff にHTMLでも保存する"}]\n'
        '</send_user_message_question_reply>'
    )

    cleaned, reasons = normalize_recall_prompt(prompt)

    assert cleaned == (
        "改善計画の渡し方はどちらがよいですか？\n"
        "既存と同じ _handoff にHTMLでも保存する"
    )
    assert "extracted question reply envelope" in reasons
    assert "request_user_input_async" not in cleaned
    assert "call_123" not in cleaned


@pytest.mark.parametrize(
    "prompt",
    [
        '<send_user_message_question_reply>{"question":"q","answer":"a"}</send_user_message_question_reply>',
        'prefix <send_user_message_question_reply>[]</send_user_message_question_reply>',
        '<send_user_message_question_reply>[]</send_user_message_question_reply> suffix',
        '<send_user_message_question_reply data="host">[]</send_user_message_question_reply>',
        '<send_user_message_question_reply>[]</send_user_message_question_reply extra>',
        '<send_user_message_question_reply>[{"questionItemId":"id","question":"q","answer":"a"}]',
        '<send_user_message_question_reply>[{"questionItemId":"id","question":"q","answer":"a"}]</send_user_message_question_reply><send_user_message_question_reply>',
    ],
)
def test_question_reply_only_unwraps_exact_outer_envelope(prompt: str) -> None:
    cleaned, reasons = normalize_recall_prompt(prompt)

    assert cleaned == prompt
    assert reasons == []


@pytest.mark.parametrize(
    "items",
    [
        '{"questionItemId":"id","question":"q","answer":"a"}',
        '[{"questionItemId":"id","question":"q","answer":"a"',
        '[{"questionItemId":"id","question":{"nested":"q"},"answer":"a"}]',
        '[{"questionItemId":"id","question":"q","answer":["a"]}]',
        '[{"questionItemId":["id"],"question":"q","answer":"a"}]',
        '[{"questionItemId":"id","question":"q","answer":"a","extra":"x"}]',
        '[{"questionItemId":"id","question":"q"}]',
        '[{"questionItemId":"id","question":" ","answer":"a"}]',
        '[{"questionItemId":"","question":"q","answer":"a"}]',
    ],
)
def test_invalid_question_reply_schema_is_kept_verbatim(items: str) -> None:
    prompt = (
        "<send_user_message_question_reply>"
        f"{items}"
        "</send_user_message_question_reply>"
    )

    cleaned, reasons = normalize_recall_prompt(prompt)

    assert cleaned == prompt
    assert reasons == []


def test_question_reply_duplicate_keys_are_not_normalized() -> None:
    prompt = (
        "<send_user_message_question_reply>"
        '[{"questionItemId":"id","question":"q","question":"q2","answer":"a"}]'
        "</send_user_message_question_reply>"
    )

    cleaned, reasons = normalize_recall_prompt(prompt)

    assert cleaned == prompt
    assert reasons == []


def test_question_reply_preserves_user_authored_structured_text() -> None:
    prompt = (
        "<user-note>{\"question\": \"keep this JSON\"}</user-note>\n"
        "```json\n{\"answer\": \"keep this code block\"}\n```"
    )

    cleaned, reasons = normalize_recall_prompt(prompt)

    assert cleaned == prompt
    assert reasons == []


def test_question_reply_rejects_long_input_without_truncation() -> None:
    item = {
        "questionItemId": "id",
        "question": "q",
        "answer": "a" * 8_193,
    }
    prompt = (
        "<send_user_message_question_reply>"
        f"{json.dumps([item], ensure_ascii=False)}"
        "</send_user_message_question_reply>"
    )

    cleaned, reasons = normalize_recall_prompt(prompt)

    assert cleaned == prompt
    assert reasons == []


def test_question_reply_keeps_multiple_items_in_order() -> None:
    items = [
        {"questionItemId": "id-1", "question": "first question", "answer": "first answer"},
        {"questionItemId": "id-2", "question": "second question", "answer": "second answer"},
    ]
    prompt = (
        "<send_user_message_question_reply>"
        f"{json.dumps(items)}"
        "</send_user_message_question_reply>"
    )

    cleaned, reasons = normalize_recall_prompt(prompt)

    assert cleaned == (
        "first question\nfirst answer\nsecond question\nsecond answer"
    )
    assert reasons == ["extracted question reply envelope"]
