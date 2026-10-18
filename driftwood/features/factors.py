"""Trailing features known at the close.

``SIGNAL_TIME`` is the close of the session. The open, high, low, close,
adjusted close, and volume on that row are treated as already published.
Nothing in this module reads a later row. Forward returns live in
``labels.py`` on purpose, so a reviewer can see the only file allowed to
look ahead.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from driftwood.features.panel import prepare_panel
from driftwood.features.transforms import cross_sectional_zscore

# A decision stamped at the close may use this session's full daily bar.
SIGNAL_TIME = "close"

RAW_FEATURES: tuple[str, ...] = (
    "ret_1d",
    "rev_5",
    "mom_21",
    "mom_63",
    "mom_12_1",
    "vol_21",
    "volume_shock",
    "dist_sma_21",
    "hl_range",
    "log_volume",
)
Z_FEATURES: tuple[str, ...] = tuple(f"z_{name}" for name in RAW_FEATURES)
FEATURE_COLUMNS: tuple[str, ...] = RAW_FEATURES + Z_FEATURES
# A linear model should see the z-scores. Raw columns mix units (a return
# next to a log volume) and would let the largest unit dominate the fit.
MODEL_FEATURES: tuple[str, ...] = Z_FEATURES
LABEL_COLUMNS: tuple[str, ...] = ("fwd_ret_1d", "fwd_ret_5d")

_PRICE_COLUMNS = ["symbol", "date", "high", "low", "close", "adj_close", "volume"]
_VOL_WINDOW = 21
_VOL_MIN_PERIODS = 15


def build_features(bars: pd.DataFrame) -> pd.DataFrame:
    """Causal feature panel. Rows with a short history stay, with missing values."""

    work = prepare_panel(bars, _PRICE_COLUMNS)
    if work.empty:
        return _empty_feature_frame()

    grouped_price = work.groupby("symbol", sort=False)["adj_close"]
    work["ret_1d"] = _finite(grouped_price.pct_change())
    work["rev_5"] = _finite(grouped_price.pct_change(5))
    work["mom_21"] = _finite(grouped_price.pct_change(21))
    work["mom_63"] = _finite(grouped_price.pct_change(63))
    # 12-1 momentum: the year ending one month ago, not the year ending today.
    # The skipped month is where short-term reversal usually lives.
    month_ago = grouped_price.shift(21)
    year_ago = grouped_price.shift(252)
    work["mom_12_1"] = _finite(month_ago / year_ago - 1.0)

    work["vol_21"] = (
        work.groupby("symbol", sort=False)["ret_1d"]
        .rolling(_VOL_WINDOW, min_periods=_VOL_MIN_PERIODS)
        .std()
        .reset_index(level=0, drop=True)
    )
    trailing_volume = (
        work.groupby("symbol", sort=False)["volume"]
        .rolling(_VOL_WINDOW, min_periods=_VOL_MIN_PERIODS)
        .mean()
        .reset_index(level=0, drop=True)
    )
    work["volume_shock"] = _finite(work["volume"] / trailing_volume)
    trailing_price = (
        work.groupby("symbol", sort=False)["adj_close"]
        .rolling(_VOL_WINDOW, min_periods=_VOL_MIN_PERIODS)
        .mean()
        .reset_index(level=0, drop=True)
    )
    work["dist_sma_21"] = _finite(work["adj_close"] / trailing_price - 1.0)
    # High, low, and close share the raw scale, including on a split day.
    # Mixing them with adj_close would turn the split into a fake range.
    work["hl_range"] = _finite((work["high"] - work["low"]) / work["close"])
    positive_volume = work["volume"].where(work["volume"] > 0)
    work["log_volume"] = np.log(positive_volume.astype(float))

    scored = cross_sectional_zscore(work, list(RAW_FEATURES))
    columns = ["symbol", "date", *FEATURE_COLUMNS]
    return scored.loc[:, columns].reset_index(drop=True)


def _finite(series: pd.Series) -> pd.Series:
    return series.replace([np.inf, -np.inf], np.nan)


def _empty_feature_frame() -> pd.DataFrame:
    columns: dict[str, pd.Series] = {
        "symbol": pd.Series(dtype="object"),
        "date": pd.Series(dtype="datetime64[ns]"),
    }
    for name in FEATURE_COLUMNS:
        columns[name] = pd.Series(dtype="float64")
    return pd.DataFrame(columns)
