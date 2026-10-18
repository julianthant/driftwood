"""Executable check that a feature function does not read the future.

Truncation is the test. Build the features twice: once on the whole panel,
once on the rows dated on or before a cutoff. Every feature value on or
before that cutoff has to match. A centered window, a negative shift, or a
scaler fit on the full sample will move a past value and fail here.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

import numpy as np
import pandas as pd

FeatureFn = Callable[[pd.DataFrame], pd.DataFrame]


def assert_causal_features(
    bars: pd.DataFrame,
    feature_fn: FeatureFn,
    columns: Sequence[str],
    cutoffs: Sequence[pd.Timestamp | str],
) -> None:
    """Raise ``AssertionError`` if ``columns`` change when the future is removed."""

    if not cutoffs:
        raise ValueError("Provide at least one cutoff")
    panel = _dated(bars)
    full = _dated(feature_fn(panel))
    missing = [column for column in columns if column not in full.columns]
    if missing:
        raise AssertionError(f"{feature_fn.__name__} did not return {missing}")

    problems: list[str] = []
    for cutoff in cutoffs:
        cutoff_ts = pd.Timestamp(cutoff).normalize()
        past_bars = panel.loc[panel["date"] <= cutoff_ts].copy()
        rebuilt = _dated(feature_fn(past_bars))
        problems.extend(_compare(full, rebuilt, columns, cutoff_ts))
        if len(problems) >= 5:
            break
    if problems:
        detail = "\n".join(problems[:5])
        raise AssertionError(f"Feature values changed when history was truncated:\n{detail}")


def _compare(
    full: pd.DataFrame,
    rebuilt: pd.DataFrame,
    columns: Sequence[str],
    cutoff: pd.Timestamp,
) -> list[str]:
    left = full.loc[full["date"] <= cutoff, ["symbol", "date", *columns]]
    right = rebuilt.loc[:, ["symbol", "date", *columns]]
    left = left.sort_values(["symbol", "date"]).reset_index(drop=True)
    right = right.sort_values(["symbol", "date"]).reset_index(drop=True)
    problems: list[str] = []
    if len(left) != len(right) or not left[["symbol", "date"]].equals(right[["symbol", "date"]]):
        problems.append(
            f"cutoff {cutoff.date()}: row keys differ "
            f"(full {len(left)} rows, truncated {len(right)} rows)"
        )
        return problems
    for column in columns:
        match = _equal(left[column].to_numpy(dtype=float), right[column].to_numpy(dtype=float))
        if bool(match.all()):
            continue
        position = int(np.flatnonzero(~match)[0])
        problems.append(
            f"cutoff {cutoff.date()} {left.loc[position, 'symbol']} {column}: "
            f"full={left.loc[position, column]!r} truncated={right.loc[position, column]!r}"
        )
        if len(problems) >= 5:
            return problems
    return problems


def _equal(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    return np.isclose(left, right, rtol=1e-9, atol=1e-12, equal_nan=True)


def _dated(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    dates = pd.to_datetime(out["date"], errors="coerce")
    if isinstance(dates.dtype, pd.DatetimeTZDtype):
        dates = dates.dt.tz_convert("UTC").dt.tz_localize(None)
    out["date"] = dates.dt.normalize()
    return out
