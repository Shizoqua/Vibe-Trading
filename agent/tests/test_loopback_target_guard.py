"""Regression tests for ``_is_allowed_loopback_target``, the narrow opt-in
used by ``validate_url_target(allow_loopback=True)``.

This helper must allow loopback *only* when every resolved address for the
host is actually loopback. A naive implementation that checks whether *any*
resolved address is loopback (e.g. via ``any()`` instead of ``all()``) would
let a DNS-rebinding style attacker pass a hostname that resolves to a mix of
a loopback address and a public address, bypassing the SSRF guard entirely
for internal-only callers that opt into ``allow_loopback=True``.
"""

from __future__ import annotations

import ipaddress

from src.channels.utils import _is_allowed_loopback_target

_LOOPBACK_V4 = ipaddress.ip_address("127.0.0.1")
_LOOPBACK_V6 = ipaddress.ip_address("::1")
_PUBLIC_V4 = ipaddress.ip_address("8.8.8.8")


def test_localhost_hostname_with_loopback_addrs_allowed() -> None:
    assert _is_allowed_loopback_target("localhost", [_LOOPBACK_V4]) is True


def test_localhost_hostname_is_case_and_trailing_dot_insensitive() -> None:
    assert _is_allowed_loopback_target("LOCALHOST", [_LOOPBACK_V4]) is True
    assert _is_allowed_loopback_target("localhost.", [_LOOPBACK_V6]) is True


def test_localhost_hostname_with_no_resolved_addrs_is_blocked() -> None:
    # Empty address list must not vacuously pass.
    assert _is_allowed_loopback_target("localhost", []) is False


def test_localhost_hostname_with_mixed_addrs_is_blocked() -> None:
    # DNS rebinding: "localhost" resolving to a loopback *and* a public
    # address must still be rejected -- every address must be loopback.
    assert _is_allowed_loopback_target("localhost", [_LOOPBACK_V4, _PUBLIC_V4]) is False


def test_literal_loopback_ip_hostname_is_allowed() -> None:
    # hostname is the IP literal itself (not the string "localhost")
    assert _is_allowed_loopback_target("127.0.0.1", [_LOOPBACK_V4]) is True


def test_non_localhost_hostname_with_mixed_addrs_is_blocked() -> None:
    # The critical SSRF-bypass case: an attacker-controlled hostname that
    # resolves to both a loopback address and a public address must be
    # rejected outright, not allowed because *some* address is loopback.
    assert (
        _is_allowed_loopback_target("evil.example.com", [_LOOPBACK_V4, _PUBLIC_V4])
        is False
    )


def test_non_localhost_hostname_with_no_resolved_addrs_is_blocked() -> None:
    # Empty address list for a non-"localhost" hostname must not vacuously
    # pass either.
    assert _is_allowed_loopback_target("evil.example.com", []) is False


def test_non_localhost_hostname_all_public_is_blocked() -> None:
    assert _is_allowed_loopback_target("evil.example.com", [_PUBLIC_V4]) is False
