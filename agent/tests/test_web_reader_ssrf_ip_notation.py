"""Regression: _url_allowed must not be bypassable by respelling a blocked
IP address in an alternate IPv4 notation.

ipaddress.ip_address() only parses the canonical dotted-quad/IPv6 form. The
same loopback or cloud-metadata address can also be written as a plain
decimal integer, in octal, with hex octets, or in shorthand dotted form --
forms socket.inet_aton() (and many HTTP clients) happily normalize to the
real address. Before this, any of those forms fell into the "not an IP, so
it's a normal hostname" branch and were waved through unchecked.
"""

from __future__ import annotations

import pytest

from src.tools.web_reader_tool import _url_allowed


@pytest.mark.parametrize(
    "url",
    [
        "http://2130706433/",  # decimal for 127.0.0.1
        "http://017700000001/",  # octal for 127.0.0.1
        "http://127.1/",  # shorthand for 127.0.0.1
        "http://0x7f.0.0.1/",  # hex octet for 127.0.0.1
        "http://2852039166/",  # decimal for 169.254.169.254 (cloud metadata)
    ],
)
def test_alternate_ipv4_notations_for_blocked_addresses_are_rejected(url: str) -> None:
    allowed, _ = _url_allowed(url)
    assert allowed is False


def test_canonical_loopback_is_still_rejected() -> None:
    allowed, _ = _url_allowed("http://127.0.0.1/")
    assert allowed is False


@pytest.mark.parametrize(
    "url",
    [
        "http://example.com/",
        "http://8.8.8.8/",
    ],
)
def test_genuine_hostnames_and_public_ips_are_still_allowed(url: str) -> None:
    allowed, _ = _url_allowed(url)
    assert allowed is True
