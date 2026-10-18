"""Cross-sectional transforms that only see the same day's names.

A scaler fit on the whole sample is a lookahead leak: the z-score of a name
in 2018 would depend on how wild 2024 was. Standardizing inside the date
does not. The work is done on a date-by-symbol grid so it stays a vectorized
NumPy operation rather than a Python call per day.
"""

from __future__ import annotations

import pandas as pd

WINSOR_LOWER = 0.05
WINSOR_UPPER = 0.95


def cross_sectional_zscore(
    frame: pd.DataFrame,
    columns: list[str],
    *,
    prefix: str = "z_",
) -> pd.DataFrame:
    """Winsorize each column inside the date, then z-score it.

    The raw column is left untouched. A day with no cross-sectional variance
    becomes 0 for names that had a value, and stays missing where the input
    was missing. Standard deviation uses the population divisor (ddof=0):
    this is a rescaling, not an estimate of a volatility.
    """

    if frame.empty:
        out = frame.copy()
        for column in columns:
            out[f"{prefix}{column}"] = pd.Series(dtype="float64")
        return out

    out = frame.copy()
    keys = out.loc[:, ["date", "symbol"]]
    for column in columns:
        wide = out.pivot(index="date", columns="symbol", values=column)
        scored = _zscore_wide(wide)
        lookup = scored.stack().rename("value").reset_index()
        merged = keys.merge(lookup, on=["date", "symbol"], how="left")
        out[f"{prefix}{column}"] = merged["value"].to_numpy()
    return out


def _zscore_wide(wide: pd.DataFrame) -> pd.DataFrame:
    lower = wide.quantile(WINSOR_LOWER, axis=1)
    upper = wide.quantile(WINSOR_UPPER, axis=1)
    clipped = wide.clip(lower=lower, upper=upper, axis=0)
    center = clipped.mean(axis=1)
    scale = clipped.std(axis=1, ddof=0)
    scored = clipped.sub(center, axis=0).div(scale.mask(scale <= 0), axis=0)
    zero_scale = ~(scale > 0)
    scored = scored.mask(zero_scale, 0.0)
    return scored.where(wide.notna())
