"""Matrix should acknowledge an inbound message with a reaction, then clear
it once the reply is sent, matching the pattern already used by Telegram,
Discord, Feishu, and Slack.
"""

from __future__ import annotations

import asyncio

import pytest

pytest.importorskip("nio")

from nio import RoomSendResponse

from src.channels.bus.events import OutboundMessage
from src.channels.matrix import MatrixChannel, _StreamBuf


def _make_channel() -> MatrixChannel:
    return MatrixChannel({}, bus=None)


class _FakeClient:
    def __init__(self) -> None:
        self.sent: list[tuple[str, str, dict]] = []
        self.redacted: list[tuple[str, str]] = []
        self._reaction_counter = 0

    async def room_send(self, room_id: str, message_type: str, content: dict, **kwargs):
        self.sent.append((room_id, message_type, content))
        if message_type == "m.reaction":
            self._reaction_counter += 1
            return RoomSendResponse(event_id=f"reaction-evt-{self._reaction_counter}", room_id=room_id)
        return RoomSendResponse(event_id="reply-evt", room_id=room_id)

    async def room_redact(self, room_id: str, event_id: str, **kwargs):
        self.redacted.append((room_id, event_id))


class TestReceiptAcknowledgment:
    def test_sending_a_reaction_targets_the_inbound_event_with_the_configured_emoji(self) -> None:
        channel = _make_channel()
        channel.client = _FakeClient()
        room_id = "!room:example.org"

        asyncio.run(channel._send_reaction(room_id, "inbound-evt", channel.config.react_emoji))

        assert len(channel.client.sent) == 1
        sent_room_id, message_type, content = channel.client.sent[0]
        assert sent_room_id == room_id
        assert message_type == "m.reaction"
        assert content["m.relates_to"] == {
            "rel_type": "m.annotation",
            "event_id": "inbound-evt",
            "key": channel.config.react_emoji,
        }
        assert channel._pending_reactions[room_id] == "reaction-evt-1"

    def test_missing_client_does_not_raise(self) -> None:
        channel = _make_channel()
        channel.client = None

        asyncio.run(channel._send_reaction("!room:example.org", "inbound-evt", "👀"))

        assert channel._pending_reactions == {}


class TestAcknowledgmentClearsOnCompletion:
    def test_plain_send_redacts_the_pending_reaction(self) -> None:
        channel = _make_channel()
        client = _FakeClient()
        channel.client = client
        room_id = "!room:example.org"

        asyncio.run(channel._send_reaction(room_id, "inbound-evt", channel.config.react_emoji))
        asyncio.run(
            channel.send(OutboundMessage(channel="matrix", chat_id=room_id, content="Final answer", metadata={}))
        )

        assert client.redacted == [(room_id, "reaction-evt-1")]
        assert room_id not in channel._pending_reactions

    def test_progress_send_does_not_redact(self) -> None:
        channel = _make_channel()
        client = _FakeClient()
        channel.client = client
        room_id = "!room:example.org"

        asyncio.run(channel._send_reaction(room_id, "inbound-evt", channel.config.react_emoji))
        asyncio.run(
            channel.send(
                OutboundMessage(
                    channel="matrix", chat_id=room_id, content="Still working...",
                    metadata={"_progress": True},
                )
            )
        )

        assert client.redacted == []
        assert channel._pending_reactions[room_id] == "reaction-evt-1"

    def test_streaming_reply_end_redacts_the_pending_reaction(self) -> None:
        channel = _make_channel()
        client = _FakeClient()
        channel.client = client
        room_id = "!room:example.org"

        asyncio.run(channel._send_reaction(room_id, "inbound-evt", channel.config.react_emoji))
        channel._stream_bufs[room_id] = _StreamBuf(text="Final answer", event_id="preview-evt")

        asyncio.run(channel.send_delta(room_id, "", {"_stream_end": True}))

        assert client.redacted == [(room_id, "reaction-evt-1")]
        assert room_id not in channel._pending_reactions
