"""Shared bar fixtures. Nothing here touches the network."""

from __future__ import annotations

from datetime import date

import pandas as pd


def bar(
    symbol: str = "AAPL",
    session: str | date = "2024-01-02",
    open_: float = 100.0,
    high: float = 102.0,
    low: float = 99.0,
    close: float = 101.0,
    adj_close: float | None = None,
    volume: int = 1_000_000,
) -> dict[str, object]:
    return {
        "symbol": symbol,
        "date": session,
        "open": open_,
        "high": high,
        "low": low,
        "close": close,
        "adj_close": close if adj_close is None else adj_close,
        "volume": volume,
    }


def bars_frame(rows: list[dict[str, object]]) -> pd.DataFrame:
    return pd.DataFrame(rows)
