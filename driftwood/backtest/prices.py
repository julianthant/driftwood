"""Adjusted bars for a research backtest.

The simulator needs an open, high, low, and close. The economic series is
the adjusted close. Scaling the raw bar by ``adj_close / close`` puts the
whole session on that series. It is the usual approximation when the vendor
did not supply an adjusted high and low. A split that halved the raw close
and left ``adj_close`` alone is undone by the factor.

Assigning ``close`` to ``adj_close`` exactly can leave it one rounding error
outside the scaled high or low. High and low are then expanded so the bar
still contains the open and the close. The fill rejects a bar that does not.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def adjusted_ohlc(frame: pd.DataFrame) -> pd.DataFrame:
    """Return a copy with open, high, low, and close on the adjusted scale.

    ``volume`` is left as stored. ``adj_close`` is copied into ``close``.
    High and low are expanded if that copy sits a rounding error outside them.
    """

    required = ["open", "high", "low", "close", "adj_close"]
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise ValueError(f"bars are missing columns: {missing}")
    out = frame.copy()
    close = out["close"].astype(float)
    adj = out["adj_close"].astype(float)
    if (close <= 0).any() or (adj <= 0).any():
        raise ValueError("adjusted bars need positive close and adj_close")
    if not np.isfinite(close.to_numpy()).all() or not np.isfinite(adj.to_numpy()).all():
        raise ValueError("adjusted bars need finite close and adj_close")
    factor = adj / close
    for column in ("open", "high", "low"):
        out[column] = out[column].astype(float) * factor
    out["close"] = adj
    # close * (adj / close) is not always the same float as adj. When the
    # raw close sat on the high or the low, the adjusted close can fall
    # just outside the scaled bar. The simulator treats that as a bad print.
    high = out["high"].to_numpy(dtype=float)
    low = out["low"].to_numpy(dtype=float)
    open_ = out["open"].to_numpy(dtype=float)
    close_ = out["close"].to_numpy(dtype=float)
    out["high"] = np.maximum(high, np.maximum(open_, close_))
    out["low"] = np.minimum(low, np.minimum(open_, close_))
    return out
