"""Regression: two attachments in the same email sharing a filename (a
forwarded thread, or the same file sent twice) must not overwrite each other.
dest was built from uid + filename alone, with no collision handling.
"""

from __future__ import annotations

from email.message import EmailMessage
from pathlib import Path

import pytest

import src.channels.email as email_mod
from src.channels.email import EmailChannel


def _message_with_two_same_named_attachments() -> EmailMessage:
    msg = EmailMessage()
    msg["From"] = "sender@example.com"
    msg["To"] = "bot@example.com"
    msg["Subject"] = "two attachments same name"
    msg.set_content("see attached")
    msg.add_attachment(
        b"CONTENTS-OF-FIRST-FILE-IMPORTANT", maintype="application",
        subtype="octet-stream", filename="report.pdf",
    )
    msg.add_attachment(
        b"CONTENTS-OF-SECOND-FILE-DIFFERENT", maintype="application",
        subtype="octet-stream", filename="report.pdf",
    )
    return msg


def test_same_named_attachments_get_distinct_paths(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(email_mod, "get_media_dir", lambda name: tmp_path)
    msg = _message_with_two_same_named_attachments()

    saved = EmailChannel._extract_attachments(
        msg, uid="42", allowed_types=["*"], max_size=10_000_000, max_count=5,
    )

    assert len(saved) == 2
    assert len(set(saved)) == 2
    assert saved[0].read_bytes() == b"CONTENTS-OF-FIRST-FILE-IMPORTANT"
    assert saved[1].read_bytes() == b"CONTENTS-OF-SECOND-FILE-DIFFERENT"


def test_single_attachment_still_downloads_normally(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(email_mod, "get_media_dir", lambda name: tmp_path)
    msg = EmailMessage()
    msg["From"] = "sender@example.com"
    msg["To"] = "bot@example.com"
    msg["Subject"] = "one attachment"
    msg.set_content("see attached")
    msg.add_attachment(
        b"ONLY-CONTENT", maintype="application", subtype="octet-stream", filename="report.pdf",
    )

    saved = EmailChannel._extract_attachments(
        msg, uid="42", allowed_types=["*"], max_size=10_000_000, max_count=5,
    )

    assert len(saved) == 1
    assert saved[0].name == "42_report.pdf"
    assert saved[0].read_bytes() == b"ONLY-CONTENT"
