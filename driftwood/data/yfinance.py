"""Turn a yfinance download into the bar columns Driftwood stores.

The network call lives in :func:`fetch_yfinance_bars`. Everything else is a
pure reshape, which is what the tests exercise. yfinance has changed its
column layout more than once, so the normalizer accepts the layouts this
project has actually seen rather than one hard-coded ``MultiIndex`` order.
"""

from __future__ import annotations

import warnings
from collections.abc import Sequence

import pandas as pd

_FIELD_NAMES = {
    "open": "open",
    "high": "high",
    "low": "low",
    "close": "close",
    "adj close": "adj_close",
    "adjclose": "adj_close",
    "volume": "volume",
}


class DataUnavailable(RuntimeError):
    """The vendor returned no rows for the requested window."""


def fetch_yfinance_bars(
    symbols: Sequence[str],
    start: str,
    end: str,
) -> pd.DataFrame:
    """Download daily bars. ``end`` is exclusive, matching yfinance.

    ``auto_adjust`` is forced off so the raw close and the adjusted close are
    both available. Adjustment, when we want it, is an explicit later choice.
    """

    import yfinance as yf

    requested = [symbol.strip().upper() for symbol in symbols]
    raw = yf.download(
        tickers=requested,
        start=start,
        end=end,
        auto_adjust=False,
        actions=False,
        group_by="column",
        threads=False,
        progress=False,
        interval="1d",
    )
    if raw is None or raw.empty:
        raise DataUnavailable(f"yfinance returned no rows for {start} to {end}")
    return normalize_download(raw, requested)


def normalize_download(raw: pd.DataFrame, symbols: Sequence[str]) -> pd.DataFrame:
    """Normalize a yfinance frame to symbol, date, OHLCV, and adjusted close."""

    if raw is None or raw.empty:
        return _empty_market_frame()

    if isinstance(raw.columns, pd.MultiIndex):
        level0 = raw.columns.get_level_values(0)
        fields_first = _recognized_fields(level0) >= 3
        symbol_level = 1 if fields_first else 0
        frames = []
        already_warned = False
        for symbol in pd.unique(raw.columns.get_level_values(symbol_level)):
            piece, did_warn = _frame_from_ohlcv(
                raw.xs(symbol, axis=1, level=symbol_level),
                str(symbol),
                warn_missing_adj=not already_warned,
            )
            already_warned = already_warned or did_warn
            frames.append(piece)
        if not frames:
            return _empty_market_frame()
        return pd.concat(frames, ignore_index=True)

    if len(list(symbols)) != 1:
        raise ValueError("A flat yfinance frame only makes sense for one symbol")
    piece, _warned = _frame_from_ohlcv(raw, str(list(symbols)[0]), warn_missing_adj=True)
    return piece


def _frame_from_ohlcv(
    frame: pd.DataFrame,
    symbol: str,
    *,
    warn_missing_adj: bool,
) -> tuple[pd.DataFrame, bool]:
    work = frame.copy()
    if isinstance(work.columns, pd.MultiIndex):
        work.columns = work.columns.get_level_values(-1)
    if isinstance(work.index, pd.DatetimeIndex):
        work = work.reset_index()

    renamed: dict[object, str] = {}
    for column in work.columns:
        field = _canonical_field(column)
        if field is not None:
            renamed[column] = field
        elif str(column).strip().lower() in {"date", "datetime", "index"}:
            renamed[column] = "date"
    work = work.rename(columns=renamed)
    if "date" not in work.columns:
        raise ValueError(f"Price frame for {symbol} has no date column")
    for column in ("open", "high", "low", "close", "volume"):
        if column not in work.columns:
            raise ValueError(f"Price frame for {symbol} is missing {column}")

    warned = False
    if "adj_close" not in work.columns:
        if warn_missing_adj:
            warnings.warn(
                "Adjusted close was missing; storing the raw close in adj_close. "
                "Returns will not be split- or dividend-adjusted.",
                stacklevel=2,
            )
            warned = True
        work["adj_close"] = work["close"]

    keep = work.loc[:, ["date", "open", "high", "low", "close", "adj_close", "volume"]].copy()
    keep.insert(0, "symbol", symbol.upper())
    # Aligned multi-ticker downloads pad missing sessions with all-NaN prices.
    # Those are not failed prints; they are empty cells in a rectangle.
    price_cols = ["open", "high", "low", "close"]
    keep = keep.dropna(subset=price_cols, how="all").reset_index(drop=True)
    return keep, warned


def _canonical_field(value: object) -> str | None:
    text = " ".join(str(value).strip().lower().replace("_", " ").split())
    return _FIELD_NAMES.get(text)


def _recognized_fields(values: pd.Index) -> int:
    return sum(1 for value in pd.unique(values) if _canonical_field(value))


def _empty_market_frame() -> pd.DataFrame:
    return pd.DataFrame(
        columns=["symbol", "date", "open", "high", "low", "close", "adj_close", "volume"]
    )
