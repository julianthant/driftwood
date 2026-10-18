"""Information coefficient versus a pooled regression of the same rows."""

import numpy as np
import pandas as pd
import pytest

from driftwood.models.ic import information_coefficient, prediction_regression, summarize_ic


def test_spearman_ic_is_one_for_a_perfect_ranking_and_missing_when_tied() -> None:
    frame = pd.DataFrame(
        {
            "date": pd.to_datetime(["2020-01-02"] * 4),
            "symbol": list("ABCD"),
            "prediction": [1.0, 2.0, 3.0, 4.0],
            "fwd_ret_1d": [10.0, 20.0, 30.0, 40.0],
        }
    )
    perfect = information_coefficient(frame)
    assert perfect.loc[0, "ic"] == 1.0
    reversed_labels = frame.copy()
    reversed_labels["fwd_ret_1d"] = [40.0, 30.0, 20.0, 10.0]
    assert information_coefficient(reversed_labels).loc[0, "ic"] == -1.0
    tied = frame.copy()
    tied["prediction"] = 1.0
    assert np.isnan(information_coefficient(tied).loc[0, "ic"])


def test_summarize_reports_mean_hit_rate_and_a_finite_hac_t() -> None:
    daily = pd.DataFrame(
        {
            "date": pd.bdate_range("2020-01-01", periods=8),
            "ic": [0.2, 0.1, -0.05, 0.3, 0.0, 0.15, 0.05, 0.25],
            "n": [10] * 8,
        }
    )
    summary = summarize_ic(daily, hac_lags=1)
    assert summary["n_dates"] == 8
    assert summary["mean_ic"] == np.mean(daily["ic"])
    assert summary["hit_rate"] == 6 / 8
    assert np.isfinite(summary["t_stat"])
    assert np.isfinite(summary["hac_t_stat"])
    empty = summarize_ic(daily.iloc[0:0], hac_lags=1)
    assert empty["n_dates"] == 0
    assert np.isnan(empty["mean_ic"])


def test_pooled_regression_slope_is_not_the_daily_ic() -> None:
    prediction = np.linspace(-1.0, 1.0, 20)
    frame = pd.DataFrame({"prediction": prediction, "fwd_ret_1d": 0.5 * prediction + 0.01})
    fit = prediction_regression(frame)
    assert fit["n"] == 20
    assert fit["slope"] == pytest.approx(0.5)
    assert fit["r2"] > 0.99
    # A single pooled slope can be large on a day when the cross-sectional rank is flat.
    flat_day = pd.DataFrame(
        {
            "date": pd.to_datetime(["2020-01-02"] * 4),
            "symbol": list("ABCD"),
            "prediction": [1.0, 1.0, 1.0, 1.0],
            "fwd_ret_1d": [1.0, 2.0, 3.0, 4.0],
        }
    )
    assert np.isnan(information_coefficient(flat_day).loc[0, "ic"])
