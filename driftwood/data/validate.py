"""Reject bars that would corrupt later research.

Validation is deliberately strict and boring. A bad high/low relationship or a
negative price should fail here, not show up as a fake return in a model.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

REQUIRED_COLUMNS: tuple[str, ...] = (
    "symbol",
    "date",
    "open",
    "high",
    "low",
    "close",
    "adj_close",
    "volume",
)

_PRICE_COLUMNS: tuple[str, ...] = ("open", "high", "low", "close", "adj_close")


def empty_bars() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "symbol": pd.Series(dtype="object"),
            "date": pd.Series(dtype="object"),
            "open": pd.Series(dtype="float64"),
            "high": pd.Series(dtype="float64"),
            "low": pd.Series(dtype="float64"),
            "close": pd.Series(dtype="float64"),
            "adj_close": pd.Series(dtype="float64"),
            "volume": pd.Series(dtype="int64"),
        }
    )


def validate_bars(frame: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, int]]:
    """Return clean bars and a count of rows dropped for each reason.

    Duplicate ``(symbol, date)`` keys keep the last row in the batch. The
    earlier copies are counted as ``duplicate_in_batch``.
    """

    if frame.columns.duplicated().any():
        raise ValueError("Bars frame has duplicate column names")
    missing = [column for column in REQUIRED_COLUMNS if column not in frame.columns]
    if missing:
        raise ValueError(f"Bars are missing columns: {missing}")
    if frame.empty:
        return empty_bars(), {}

    # Row checks are positional. A caller-supplied index may contain duplicates.
    frame = frame.reset_index(drop=True)
    work = pd.DataFrame(index=frame.index)
    work["symbol"] = frame["symbol"].astype("string").str.strip().str.upper()
    dates = pd.to_datetime(frame["date"], errors="coerce")
    if isinstance(dates.dtype, pd.DatetimeTZDtype):
        dates = dates.dt.tz_convert("UTC").dt.tz_localize(None)
    work["date"] = dates
    for column in _PRICE_COLUMNS:
        work[column] = pd.to_numeric(frame[column], errors="coerce")
    work["volume"] = pd.to_numeric(frame["volume"], errors="coerce")

    reasons = pd.Series("", index=work.index, dtype="object")

    def tag(mask: pd.Series, reason: str) -> None:
        choose = mask.fillna(False).astype(bool) & reasons.eq("")
        reasons.loc[choose] = reason

    symbol_bad = work["symbol"].isna() | work["symbol"].fillna("").eq("")
    tag(symbol_bad | work["date"].isna(), "missing_symbol_or_date")

    price = work.loc[:, list(_PRICE_COLUMNS)]
    finite = np.isfinite(price.to_numpy(dtype=float)).all(axis=1)
    tag(pd.Series(~finite, index=work.index), "non_finite_price")
    tag((price <= 0).any(axis=1), "non_positive_price")
    inconsistent = (
        (work["high"] < work["low"])
        | (work["high"] < work["open"])
        | (work["high"] < work["close"])
        | (work["low"] > work["open"])
        | (work["low"] > work["close"])
    )
    tag(inconsistent, "ohlc_inconsistent")

    volume = work["volume"].to_numpy(dtype=float)
    fractional = pd.Series(np.abs(volume - np.floor(volume)) > 1e-6, index=work.index)
    # NaN comparisons are false; missing volume is caught by isna below.
    fractional = fractional.fillna(False)
    bad_volume = work["volume"].isna() | (work["volume"] < 0) | fractional
    tag(bad_volume, "bad_volume")

    counts = {str(key): int(value) for key, value in reasons[reasons.ne("")].value_counts().items()}
    good = work.loc[reasons.eq("")].copy()
    if good.empty:
        return empty_bars(), counts

    before = len(good)
    good = good.drop_duplicates(["symbol", "date"], keep="last")
    duplicates = before - len(good)
    if duplicates:
        counts["duplicate_in_batch"] = int(duplicates)

    good["volume"] = good["volume"].round().astype("int64")
    good["date"] = pd.to_datetime(good["date"]).dt.date
    good = good.reset_index(drop=True)
    return good.loc[:, list(REQUIRED_COLUMNS)], counts
