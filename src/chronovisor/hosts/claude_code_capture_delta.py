"""Claude Code transcript delta selection."""

from __future__ import annotations

from dataclasses import replace

from chronovisor.raw.claude_code_transcript import (
    TranscriptRecord,
    TranscriptSlice,
    serialize_transcript_records,
)


class ClaudeCodeSaveError(RuntimeError):
    pass


def bounded_transcript_slice(
    transcript_slice: TranscriptSlice,
    *,
    max_chars: int,
) -> TranscriptSlice:
    """Return a byte-bounded ordered prefix; never admit an oversized first row."""
    if max_chars < 1:
        raise ClaudeCodeSaveError("max_chars must be a positive byte limit")
    if len(_serialized_records_bytes(transcript_slice.records)) <= max_chars:
        return transcript_slice
    selected: list[TranscriptRecord] = []
    for record in transcript_slice.records:
        candidate = [*selected, record]
        if len(_serialized_records_bytes(candidate)) > max_chars:
            break
        selected.append(record)
    return replace(
        transcript_slice,
        records=selected,
        scanned_until_line=selected[-1].line if selected else transcript_slice.after_line,
        user_turn_count=sum(record.role == "user" for record in selected),
    )


def bounded_transcript_slice_for_layout(
    transcript_slice: TranscriptSlice,
    *,
    max_chars: int,
    layout: str,
) -> TranscriptSlice:
    """Allow one oversized native JSONL row only in the lossless v2 layout."""

    bounded = bounded_transcript_slice(transcript_slice, max_chars=max_chars)
    if layout != "v2" or bounded.records or not transcript_slice.records:
        return bounded
    first = transcript_slice.records[0]
    return replace(
        transcript_slice,
        records=[first],
        scanned_until_line=first.line,
        user_turn_count=1 if first.role == "user" else 0,
    )


def _serialized_records_bytes(records: list[TranscriptRecord]) -> bytes:
    return serialize_transcript_records(records).encode("utf-8")
