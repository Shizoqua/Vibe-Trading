"""Regression: two distinct inbound Feishu messages whose attachments share
a server-reported filename must not overwrite each other. ``filename`` came
straight from the server with no per-message uniqueness and no collision
check, so ``file_path = media_dir / filename`` collided across messages.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

import src.channels.feishu as feishu_mod
from src.channels.bus.queue import MessageBus
from src.channels.feishu import FeishuChannel


def _make_channel() -> FeishuChannel:
    return FeishuChannel({"app_id": "cli_a", "app_secret": "sec"}, MessageBus())


def test_same_named_attachments_from_different_messages_get_distinct_paths(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(feishu_mod, "get_media_dir", lambda name: tmp_path)
    channel = _make_channel()

    async def scenario() -> tuple[str, str]:
        channel._download_file_sync = lambda message_id, file_key, resource_type: (
            b"FIRST-USERS-FILE",
            "report.pdf",
        )
        path1, _ = await channel._download_and_save_media(
            "file", {"file_key": "key_AAAA111111"}, message_id="msg_from_alice"
        )

        channel._download_file_sync = lambda message_id, file_key, resource_type: (
            b"SECOND-USERS-FILE",
            "report.pdf",
        )
        path2, _ = await channel._download_and_save_media(
            "file", {"file_key": "key_BBBB222222"}, message_id="msg_from_bob"
        )
        return path1, path2

    path1, path2 = asyncio.run(scenario())

    assert path1 != path2
    assert Path(path1).read_bytes() == b"FIRST-USERS-FILE"
    assert Path(path2).read_bytes() == b"SECOND-USERS-FILE"


def test_single_attachment_still_downloads_normally(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(feishu_mod, "get_media_dir", lambda name: tmp_path)
    channel = _make_channel()

    async def scenario() -> str:
        channel._download_file_sync = lambda message_id, file_key, resource_type: (
            b"ONLY-CONTENT",
            "report.pdf",
        )
        path, _ = await channel._download_and_save_media(
            "file", {"file_key": "key_AAAA111111"}, message_id="msg_from_alice"
        )
        return path

    path = asyncio.run(scenario())

    assert Path(path).name == "report.pdf"
    assert Path(path).read_bytes() == b"ONLY-CONTENT"
