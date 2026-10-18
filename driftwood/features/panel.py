"""Shared preparation for feature and label frames.

Both builders have to see the same sorted panel. If one sorted and the other
did not, a later merge could line a label up with the wrong day.
"""

from __future__ import annotations

import pandas as pd


def prepare_panel(bars: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Return ``columns`` sorted by symbol and date, one row per key."""

    missing = [column for column in columns if column not in bars.columns]
    if missing:
        raise ValueError(f"Bars are missing columns: {missing}")
    work = bars.loc[:, columns].copy()
    work["symbol"] = work["symbol"].astype("string").str.strip().str.upper()
    work["date"] = _normalize_dates(work["date"])
    if work["symbol"].isna().any() or work["symbol"].eq("").any():
        raise ValueError("Every bar needs a symbol")
    if work["date"].isna().any():
        raise ValueError("Every bar needs a date")
    if work.duplicated(["symbol", "date"]).any():
        raise ValueError("Features require unique (symbol, date) rows")
    return work.sort_values(["symbol", "date"]).reset_index(drop=True)


def _normalize_dates(values: pd.Series) -> pd.Series:
    dates = pd.to_datetime(values, errors="coerce")
    if isinstance(dates.dtype, pd.DatetimeTZDtype):
        dates = dates.dt.tz_convert("UTC").dt.tz_localize(None)
    return dates.dt.normalize()
