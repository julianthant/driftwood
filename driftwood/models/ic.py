"""Information coefficient: did the forecast rank the names correctly that day?

The headline number is the average, across days, of the cross-sectional
correlation between the prediction and the realized forward return. Spearman
(rank) correlation is the default. A single pooled regression of returns on
predictions is reported next to it and is not the same statistic.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.api as sm


def information_coefficient(
    frame: pd.DataFrame,
    prediction: str = "prediction",
    label: str = "fwd_ret_1d",
    method: str = "spearman",
) -> pd.DataFrame:
    """One row per date. Days with fewer than two usable names are missing."""

    rows: list[dict[str, object]] = []
    for date, group in frame.groupby("date", sort=True):
        pair = group[[prediction, label]].dropna()
        usable = len(pair) >= 2 and pair[prediction].nunique() >= 2 and pair[label].nunique() >= 2
        ic = float(pair[prediction].corr(pair[label], method=method)) if usable else np.nan
        rows.append({"date": pd.Timestamp(date), "ic": ic, "n": int(len(pair))})
    return pd.DataFrame(rows, columns=["date", "ic", "n"])


def summarize_ic(daily: pd.DataFrame, hac_lags: int = 1) -> dict[str, float | int]:
    """Mean IC, the dispersion of daily IC, and two t-statistics.

    The plain t-stat treats daily ICs as independent. The HAC t-stat allows
    ``hac_lags`` of autocorrelation. For a one-day label, lag 1 is a small
    allowance. For a five-day label, overlapping return windows make the
    plain t-stat too flattering, and the HAC number is the one to quote.
    """

    series = daily["ic"].dropna()
    n = int(len(series))
    if n == 0:
        return {
            "n_dates": 0,
            "mean_ic": np.nan,
            "std_ic": np.nan,
            "ic_ir": np.nan,
            "t_stat": np.nan,
            "hac_t_stat": np.nan,
            "hit_rate": np.nan,
        }
    mean = float(series.mean())
    std = float(series.std(ddof=1)) if n > 1 else float("nan")
    ir = mean / std if np.isfinite(std) and std > 0 else float("nan")
    t_stat = mean / (std / np.sqrt(n)) if np.isfinite(std) and std > 0 else float("nan")
    hac = _hac_mean_t(series.to_numpy(dtype=float), hac_lags) if n > hac_lags + 2 else float("nan")
    return {
        "n_dates": n,
        "mean_ic": mean,
        "std_ic": std,
        "ic_ir": ir,
        "t_stat": float(t_stat),
        "hac_t_stat": hac,
        "hit_rate": float((series > 0).mean()),
    }


def ic_by_year(daily: pd.DataFrame, hac_lags: int = 1) -> pd.DataFrame:
    frame = daily.dropna(subset=["ic"]).copy()
    if frame.empty:
        return pd.DataFrame(columns=["year", "n_dates", "mean_ic", "std_ic", "ic_ir", "hit_rate"])
    frame["year"] = pd.to_datetime(frame["date"]).dt.year
    rows: list[dict[str, float | int]] = []
    for year, group in frame.groupby("year", sort=True):
        summary = summarize_ic(group, hac_lags=hac_lags)
        summary["year"] = int(year)
        rows.append(summary)
    return pd.DataFrame(rows)


def prediction_regression(
    frame: pd.DataFrame,
    prediction: str = "prediction",
    label: str = "fwd_ret_1d",
) -> dict[str, float | int]:
    """Pooled OLS of the realized return on the prediction, across names and days.

    This is not the information coefficient. A positive slope says the
    forecast has the right sign in a pooled sense. The IC says whether the
    ranking inside each day was right.
    """

    pair = frame[[prediction, label]].dropna()
    if len(pair) < 3 or pair[prediction].nunique() < 2:
        return {
            "n": int(len(pair)),
            "intercept": np.nan,
            "slope": np.nan,
            "slope_t": np.nan,
            "r2": np.nan,
        }
    design = sm.add_constant(pair[prediction].to_numpy(dtype=float))
    fit = sm.OLS(pair[label].to_numpy(dtype=float), design).fit()
    return {
        "n": int(len(pair)),
        "intercept": float(fit.params[0]),
        "slope": float(fit.params[1]),
        "slope_t": float(fit.tvalues[1]),
        "r2": float(fit.rsquared),
    }


def _hac_mean_t(values: np.ndarray, lags: int) -> float:
    if lags < 0:
        raise ValueError("hac_lags cannot be negative")
    design = np.ones((len(values), 1))
    fit = sm.OLS(values, design).fit(cov_type="HAC", cov_kwds={"maxlags": int(lags)})
    return float(fit.tvalues[0])
