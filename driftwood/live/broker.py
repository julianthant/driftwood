"""Alpaca paper client. The host is a constant, not a caller-supplied string."""

from __future__ import annotations

PAPER_URL = "https://paper-api.alpaca.markets"


def assert_paper_url(url: str | None) -> str:
    """Accept only the Alpaca paper host.

    ``None`` or an empty string means the caller did not choose a host, so
    the paper host is used. Any other string is refused before a client
    exists, including the live host, a lookalike domain, and plain HTTP.
    """

    if url is None or url == "":
        return PAPER_URL
    if url != PAPER_URL:
        raise ValueError(
            f"Refusing to connect. Driftwood only talks to {PAPER_URL}. Got {url!r}."
        )
    return PAPER_URL


def make_paper_client(api_key: str, secret_key: str, url: str | None = None):
    """Build a trading client pointed at the paper host.

    The URL check runs before the SDK is imported. The client is always
    constructed with ``paper=True`` and ``PAPER_URL``. The string the caller
    passed is never forwarded.
    """

    assert_paper_url(url)
    if not api_key or not secret_key:
        raise ValueError("ALPACA_API_KEY and ALPACA_SECRET_KEY are required")
    from alpaca.trading.client import TradingClient

    return TradingClient(
        api_key=api_key,
        secret_key=secret_key,
        paper=True,
        url_override=PAPER_URL,
    )
