"""Hand checks for feature definitions. No vendor data."""

import numpy as np
import pandas as pd
import pytest

from driftwood.features.factors import (
    FEATURE_COLUMNS,
    LABEL_COLUMNS,
    MODEL_FEATURES,
    RAW_FEATURES,
    Z_FEATURES,
    build_features,
)
from driftwood.features.labels import build_labels
from driftwood.features.pipeline import build_research_frame
from driftwood.features.transforms import cross_sectional_zscore


def _path(symbol: str, close: np.ndarray, dates: pd.DatetimeIndex) -> pd.DataFrame:
    close = np.asarray(close, dtype=float)
    return pd.DataFrame(
        {
            "symbol": symbol,
            "date": dates,
            "open": close,
            "high": close + 1.0,
            "low": close - 0.5,
            "close": close,
            "adj_close": close,
            "volume": np.full(len(close), 1_000),
        }
    )


def test_trailing_definitions_match_a_hand_calculation() -> None:
    dates = pd.bdate_range("2016-01-04", periods=400)
    close = np.arange(100, 500, dtype=float)
    features = build_features(_path("AAPL", close, dates))
    returns = pd.Series(close).pct_change()
    t = 300
    assert features["ret_1d"].iloc[t] == pytest.approx(close[t] / close[t - 1] - 1.0)
    assert features["rev_5"].iloc[t] == pytest.approx(close[t] / close[t - 5] - 1.0)
    assert features["mom_21"].iloc[t] == pytest.approx(close[t] / close[t - 21] - 1.0)
    assert features["mom_63"].iloc[t] == pytest.approx(close[t] / close[t - 63] - 1.0)
    assert features["mom_12_1"].iloc[t] == pytest.approx(close[t - 21] / close[t - 252] - 1.0)
    window = returns.iloc[t - 20 : t + 1]
    assert len(window) == 21
    assert window.notna().all()
    assert features["vol_21"].iloc[t] == pytest.approx(window.std(ddof=1))
    assert features["volume_shock"].iloc[t] == pytest.approx(1.0)
    sma = close[t - 20 : t + 1].mean()
    assert features["dist_sma_21"].iloc[t] == pytest.approx(close[t] / sma - 1.0)
    assert features["hl_range"].iloc[t] == pytest.approx(1.5 / close[t])
    assert features["log_volume"].iloc[t] == pytest.approx(np.log(1000.0))
    assert np.isnan(features["mom_12_1"].iloc[200])


def test_labels_are_the_forward_simple_return() -> None:
    dates = pd.bdate_range("2024-01-02", periods=8)
    close = np.array([100, 110, 121, 100, 100, 100, 100, 150], dtype=float)
    labels = build_labels(_path("MSFT", close, dates))
    assert labels["fwd_ret_1d"].iloc[0] == pytest.approx(0.10)
    assert labels["fwd_ret_1d"].iloc[1] == pytest.approx(0.10)
    assert np.isnan(labels["fwd_ret_1d"].iloc[-1])
    assert labels["fwd_ret_5d"].iloc[0] == pytest.approx(close[5] / close[0] - 1.0)
    assert np.isnan(labels["fwd_ret_5d"].iloc[-5])


def test_zscore_ignores_other_dates_and_flattens_a_constant_day() -> None:
    frame = pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-01-02"] * 3 + ["2024-01-03"] * 3),
            "symbol": ["A", "B", "C", "A", "B", "C"],
            "ret_1d": [1.0, 2.0, 3.0, 100.0, 100.0, 100.0],
        }
    )
    scored = cross_sectional_zscore(frame, ["ret_1d"])
    day_one = scored.loc[scored["date"] == "2024-01-02", "z_ret_1d"]
    day_two = scored.loc[scored["date"] == "2024-01-03", "z_ret_1d"]
    assert day_one.mean() == pytest.approx(0.0)
    assert day_one.std(ddof=0) == pytest.approx(1.0)
    assert day_two.tolist() == pytest.approx([0.0, 0.0, 0.0])

    clipped = frame.loc[frame["date"] == "2024-01-02", "ret_1d"]
    alone = cross_sectional_zscore(
        pd.DataFrame(
            {
                "date": frame["date"].iloc[:3].to_numpy(),
                "symbol": frame["symbol"].iloc[:3].to_numpy(),
                "ret_1d": clipped.to_numpy(),
            }
        ),
        ["ret_1d"],
    )
    assert alone["z_ret_1d"].tolist() == pytest.approx(day_one.tolist())


def test_research_frame_keeps_features_and_labels_on_the_same_key() -> None:
    dates = pd.bdate_range("2024-01-02", periods=30)
    bars = pd.concat(
        [
            _path("AAPL", np.linspace(10, 20, 30), dates),
            _path("MSFT", np.linspace(30, 40, 30), dates),
        ],
        ignore_index=True,
    )
    bars = bars.sample(frac=1.0, random_state=1)
    frame = build_research_frame(bars)
    features = build_features(bars)
    labels = build_labels(bars)
    merged = features.merge(labels, on=["symbol", "date"])
    assert len(frame) == len(merged)
    assert list(frame.columns) == ["symbol", "date", *FEATURE_COLUMNS, *LABEL_COLUMNS]
    assert set(MODEL_FEATURES).isdisjoint(LABEL_COLUMNS)
    assert set(Z_FEATURES).isdisjoint(RAW_FEATURES)
    pd.testing.assert_series_equal(
        frame.sort_values(["symbol", "date"])["fwd_ret_1d"].reset_index(drop=True),
        labels.sort_values(["symbol", "date"])["fwd_ret_1d"].reset_index(drop=True),
        check_names=False,
    )


def test_duplicate_keys_and_missing_columns_are_rejected() -> None:
    dates = pd.bdate_range("2024-01-02", periods=2)
    bars = _path("AAPL", np.array([10.0, 11.0]), dates)
    doubled = pd.concat([bars, bars], ignore_index=True)
    with pytest.raises(ValueError, match="unique"):
        build_features(doubled)
    with pytest.raises(ValueError, match="missing columns"):
        build_features(bars.drop(columns=["volume"]))
