"""Deterministic daily bars for tests, the notebook, and offline demos.

The paths are a geometric random walk with a small intraday range. They are
not a market simulator and they are not used as a backtest of real edge.
They exist so every later stage can run with no network and no API key.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd


def synthetic_bars(
    symbols: Sequence[str],
    *,
    start: str = "2018-01-02",
    periods: int = 252,
    seed: int = 7,
    split_symbol: str | None = None,
) -> pd.DataFrame:
    """Build business-day OHLCV bars.

    ``split_symbol``, when set and present, gets a 2-for-1 split halfway
    through the window. Raw prices halve and volume doubles. ``adj_close``
    stays on the pre-split scale so adjusted returns do not show a fake crash.
    """

    if periods < 2:
        raise ValueError("periods must be at least 2")
    if not symbols:
        raise ValueError("symbols must not be empty")

    dates = pd.bdate_range(start, periods=periods)
    frames: list[pd.DataFrame] = []
    split_at = periods // 2
    for index, symbol in enumerate(symbols):
        frames.append(
            _one_symbol(
                symbol=symbol.upper(),
                dates=dates,
                seed=seed + index * 1009,
                level=40.0 + 15.0 * (index % 8),
                vol=0.008 + 0.002 * (index % 5),
                apply_split=split_symbol is not None and symbol.upper() == split_symbol.upper(),
                split_at=split_at,
            )
        )
    return pd.concat(frames, ignore_index=True)


def _one_symbol(
    *,
    symbol: str,
    dates: pd.DatetimeIndex,
    seed: int,
    level: float,
    vol: float,
    apply_split: bool,
    split_at: int,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    shocks = rng.normal(0.0003, vol, size=len(dates))
    close = level * np.exp(np.cumsum(shocks))
    open_ = np.empty_like(close)
    open_[0] = close[0]
    open_[1:] = close[:-1]
    high = np.maximum(open_, close) * (1.0 + vol * 0.25)
    low = np.minimum(open_, close) * (1.0 - vol * 0.25)
    volume = rng.integers(1_000_000, 8_000_000, size=len(dates))
    adj_close = close.copy()

    if apply_split:
        close = close.copy()
        open_ = open_.copy()
        high = high.copy()
        low = low.copy()
        volume = volume.copy()
        close[split_at:] /= 2.0
        open_[split_at:] /= 2.0
        high[split_at:] /= 2.0
        low[split_at:] /= 2.0
        volume[split_at:] *= 2

    return pd.DataFrame(
        {
            "symbol": symbol,
            "date": dates,
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "adj_close": adj_close,
            "volume": volume.astype("int64"),
        }
    )
