"""Walk-forward splits, a planted signal, and the rule that test labels cannot move the forecast."""

import numpy as np
import pandas as pd
import pytest

from driftwood.models.estimators import ColumnBaseline, LightGBMModel, RidgeModel
from driftwood.models.ic import information_coefficient
from driftwood.models.walkforward import (
    InsufficientHistory,
    WalkForwardConfig,
    iter_folds,
    walk_forward,
)


def test_expanding_folds_leave_a_purge_and_do_not_overlap() -> None:
    dates = pd.bdate_range("2020-01-01", periods=80)
    config = WalkForwardConfig(min_train_days=30, test_days=15, embargo_days=5, label_horizon=1)
    folds = list(iter_folds(dates, config))
    assert len(folds) == 3
    seen: list[pd.Timestamp] = []
    for fold in folds:
        assert set(fold.train_dates).isdisjoint(fold.test_dates)
        assert set(fold.embargo_dates).isdisjoint(fold.train_dates)
        assert set(fold.embargo_dates).isdisjoint(fold.test_dates)
        assert fold.train_dates[-1] < fold.embargo_dates[0] < fold.test_dates[0]
        assert len(fold.embargo_dates) == config.purge
        assert len(fold.test_dates) == 15
        seen.extend(fold.test_dates)
    assert len(seen) == len(set(seen))
    assert folds[0].train_dates[0] == dates[0]
    assert len(folds[-1].train_dates) > len(folds[0].train_dates)


def test_purge_covers_the_label_horizon_even_when_the_embargo_is_zero() -> None:
    dates = pd.bdate_range("2020-01-01", periods=80)
    config = WalkForwardConfig(min_train_days=30, test_days=15, embargo_days=0, label_horizon=5)
    assert config.purge == 5
    folds = list(iter_folds(dates, config))
    assert len(folds[0].embargo_dates) == 5
    assert folds[0].test_dates[0] == dates[35]


def test_rolling_window_stops_growing() -> None:
    dates = pd.bdate_range("2020-01-01", periods=80)
    config = WalkForwardConfig(
        min_train_days=30,
        test_days=15,
        embargo_days=5,
        label_horizon=1,
        expanding=False,
        train_days=40,
    )
    folds = list(iter_folds(dates, config))
    assert len(folds[0].train_dates) == 30
    assert len(folds[1].train_dates) == 40
    assert len(folds[-1].train_dates) == 40


def test_short_history_has_no_folds() -> None:
    dates = pd.bdate_range("2020-01-01", periods=10)
    config = WalkForwardConfig(min_train_days=30, test_days=5, embargo_days=1)
    assert list(iter_folds(dates, config)) == []
    frame = _panel(n_names=4, n_dates=10, noise=0.0)
    with pytest.raises(InsufficientHistory):
        walk_forward(frame, ["z_mom_21"], "fwd_ret_1d", RidgeModel, config)


def test_config_rejects_a_negative_embargo() -> None:
    with pytest.raises(ValueError, match="embargo"):
        WalkForwardConfig(embargo_days=-1)


def test_ridge_recovers_a_planted_cross_sectional_signal() -> None:
    frame = _panel(n_names=8, n_dates=80, noise=0.0)
    config = _short_config()
    result = walk_forward(
        frame,
        ["z_mom_21"],
        "fwd_ret_1d",
        lambda: RidgeModel(alpha=1.0),
        config,
    )
    assert len(result.folds) == 3
    first_test = pd.Timestamp(frame["date"].drop_duplicates().iloc[35])
    assert result.predictions["date"].min() == first_test
    daily = information_coefficient(result.predictions)
    assert daily["ic"].mean() > 0.9
    weights = result.weights.loc[result.weights["feature"] == "z_mom_21", "weight"]
    assert (weights > 0).all()
    # In-sample rows are a summary number, not a forecast the backtest could trade.
    assert result.predictions["fold"].nunique() == 3
    train_dates = set(iter_folds(frame["date"].unique(), config).__next__().train_dates)
    assert train_dates.isdisjoint(set(result.predictions["date"]))


def test_a_fold_ignores_labels_from_its_embargo_and_its_test_window() -> None:
    """Later folds may learn from returns that have since realized. This fold may not.

    Scrambling every out-of-sample date at once would change fold 2, because
    fold 2's training window includes fold 0's test dates. That is the point
    of walk-forward. The invariant is narrower: labels after this fold's last
    training date do not move this fold's forecast.
    """

    frame = _panel(n_names=8, n_dates=80, noise=0.0, seed=3)
    config = _short_config()
    original = walk_forward(frame, ["z_mom_21"], "fwd_ret_1d", lambda: RidgeModel(), config)
    folds = list(iter_folds(frame["date"].unique(), config))
    rng = np.random.default_rng(0)
    for fold_id, fold in enumerate(folds):
        scrambled = frame.copy()
        mask = scrambled["date"] > fold.train_dates[-1]
        scrambled.loc[mask, "fwd_ret_1d"] = rng.permutation(
            scrambled.loc[mask, "fwd_ret_1d"].to_numpy()
        )
        rerun = walk_forward(
            scrambled,
            ["z_mom_21"],
            "fwd_ret_1d",
            lambda: RidgeModel(),
            config,
        )
        _assert_same_forecast(original, rerun, fold_id)
        if fold_id == len(folds) - 1:
            _assert_same_forecast(original, rerun)


def test_momentum_baseline_copies_the_feature_and_ignores_the_target() -> None:
    frame = _panel(n_names=6, n_dates=50, noise=0.0, seed=4)
    garbage = frame.copy()
    garbage["fwd_ret_1d"] = np.random.default_rng(1).normal(size=len(garbage))
    config = WalkForwardConfig(min_train_days=20, test_days=10, embargo_days=2, label_horizon=1)
    result = walk_forward(
        garbage,
        ["z_mom_21"],
        "fwd_ret_1d",
        lambda: ColumnBaseline("z_mom_21"),
        config,
    )
    merged = result.predictions.merge(frame, on=["symbol", "date"], how="left")
    np.testing.assert_allclose(merged["prediction"], merged["z_mom_21"])
    assert result.model_name == "momentum"


def test_lightgbm_fit_is_repeatable() -> None:
    frame = _panel(n_names=6, n_dates=40, noise=0.01, seed=5)
    config = WalkForwardConfig(
        min_train_days=15,
        test_days=10,
        embargo_days=2,
        label_horizon=1,
        min_train_rows=10,
    )

    def factory() -> LightGBMModel:
        return LightGBMModel(n_estimators=20)

    first = walk_forward(frame, ["z_mom_21"], "fwd_ret_1d", factory, config)
    second = walk_forward(frame, ["z_mom_21"], "fwd_ret_1d", factory, config)
    pd.testing.assert_series_equal(
        first.predictions["prediction"].reset_index(drop=True),
        second.predictions["prediction"].reset_index(drop=True),
    )
    assert not first.predictions.empty


def _assert_same_forecast(original, rerun, fold_id: int | None = None) -> None:
    left = original.predictions
    right = rerun.predictions
    weights_left = original.weights
    weights_right = rerun.weights
    if fold_id is not None:
        left = left.loc[left["fold"] == fold_id]
        right = right.loc[right["fold"] == fold_id]
        weights_left = weights_left.loc[weights_left["fold"] == fold_id]
        weights_right = weights_right.loc[weights_right["fold"] == fold_id]
    columns = ["symbol", "date", "prediction"]
    left = left.sort_values(["symbol", "date"]).reset_index(drop=True)
    right = right.sort_values(["symbol", "date"]).reset_index(drop=True)
    pd.testing.assert_frame_equal(left[columns], right[columns])
    pd.testing.assert_frame_equal(
        weights_left.reset_index(drop=True),
        weights_right.reset_index(drop=True),
    )


def _short_config() -> WalkForwardConfig:
    return WalkForwardConfig(min_train_days=30, test_days=15, embargo_days=5, label_horizon=1)


def _panel(n_names: int, n_dates: int, noise: float, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2018-01-02", periods=n_dates)
    symbols = [f"S{i:02d}" for i in range(n_names)]
    rows: list[dict[str, object]] = []
    for date in dates:
        signal = rng.normal(size=n_names)
        signal = (signal - signal.mean()) / signal.std(ddof=0)
        label = 0.02 * signal + noise * rng.normal(size=n_names)
        for symbol, value, target in zip(symbols, signal, label, strict=True):
            rows.append(
                {
                    "symbol": symbol,
                    "date": date,
                    "z_mom_21": float(value),
                    "fwd_ret_1d": float(target),
                }
            )
    return pd.DataFrame(rows)
