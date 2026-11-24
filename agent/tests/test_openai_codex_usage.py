from __future__ import annotations

from src.providers.openai_codex import CodexAIMessage, _message_chunks_from_events


def test_completed_event_exposes_codex_token_usage() -> None:
    [chunk] = list(
        _message_chunks_from_events(
            [
                {
                    "type": "response.completed",
                    "response": {
                        "status": "completed",
                        "usage": {
                            "input_tokens": 12,
                            "output_tokens": 7,
                            "total_tokens": 19,
                        },
                    },
                }
            ]
        )
    )

    assert chunk.usage_metadata == {
        "input_tokens": 12,
        "output_tokens": 7,
        "total_tokens": 19,
    }


def test_completed_event_passes_cached_and_reasoning_tokens() -> None:
    [chunk] = list(
        _message_chunks_from_events(
            [
                {
                    "type": "response.completed",
                    "response": {
                        "status": "completed",
                        "usage": {
                            "input_tokens": 12,
                            "output_tokens": 7,
                            "total_tokens": 19,
                            "input_tokens_details": {"cached_tokens": 5},
                            "output_tokens_details": {"reasoning_tokens": 3},
                        },
                    },
                }
            ]
        )
    )

    assert chunk.usage_metadata == {
        "input_tokens": 12,
        "output_tokens": 7,
        "total_tokens": 19,
        "input_token_details": {"cache_read": 5},
        "output_token_details": {"reasoning": 3},
    }


def test_completed_event_exposes_codex_response_model() -> None:
    [chunk] = list(
        _message_chunks_from_events(
            [
                {
                    "type": "response.completed",
                    "response": {
                        "status": "completed",
                        "model": "gpt-5.4",
                    },
                }
            ]
        )
    )

    assert chunk.response_metadata["model_name"] == "gpt-5.4"


def test_codex_message_accumulation_preserves_final_usage() -> None:
    message = CodexAIMessage(content="hello") + CodexAIMessage(
        usage_metadata={
            "input_tokens": 12,
            "output_tokens": 7,
            "total_tokens": 19,
        }
    )

    assert message.content == "hello"
    assert message.usage_metadata == {
        "input_tokens": 12,
        "output_tokens": 7,
        "total_tokens": 19,
    }


def test_codex_message_accumulation_preserves_response_model() -> None:
    message = CodexAIMessage(content="hello") + CodexAIMessage(
        response_metadata={
            "finish_reason": "stop",
            "model_name": "gpt-5.4",
        }
    )

    assert message.content == "hello"
    assert message.response_metadata["model_name"] == "gpt-5.4"


def test_completed_event_without_usage_stays_unreported() -> None:
    [chunk] = list(
        _message_chunks_from_events(
            [{"type": "response.completed", "response": {"status": "completed"}}]
        )
    )

    assert chunk.usage_metadata is None


def _incomplete(reason: str) -> dict:
    return {
        "type": "response.incomplete",
        "response": {
            "status": "incomplete",
            "incomplete_details": {"reason": reason},
            "model": "gpt-5.5",
            "usage": {"input_tokens": 90, "output_tokens": 40, "total_tokens": 130},
        },
    }


def test_incomplete_event_reports_truncation_and_usage() -> None:
    [text, final] = list(
        _message_chunks_from_events(
            [
                {"type": "response.output_text.delta", "delta": "Revenue grew"},
                _incomplete("max_output_tokens"),
            ]
        )
    )
    message = text + final

    assert message.content == "Revenue grew"
    assert message.response_metadata["finish_reason"] == "length"
    assert message.response_metadata["model_name"] == "gpt-5.5"
    assert message.usage_metadata == {
        "input_tokens": 90,
        "output_tokens": 40,
        "total_tokens": 130,
    }


def test_incomplete_event_cut_by_content_filter_is_reported_as_such() -> None:
    [chunk] = list(_message_chunks_from_events([_incomplete("content_filter")]))

    assert chunk.response_metadata["finish_reason"] == "content_filter"
