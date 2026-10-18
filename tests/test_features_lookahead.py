"""The truncation test is the product. A leak has to fail it."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from driftwood.data.synthetic import synthetic_bars
from driftwood.features.factors import FEATURE_COLUMNS, build_features
from driftwood.features.labels import build_labels
from driftwood.features.pit import assert_causal_features


def _panel() -> pd.DataFrame:
    return synthetic_bars(["AAPL", "MSFT", "XOM", "JPM"], periods=280, seed=11)


def test_built_features_do_not_change_when_the_future_is_cut_off() -> None:
    bars = _panel()
    dates = pd.to_datetime(bars["date"]).sort_values().unique()
    cutoffs = [dates[180], dates[220], dates[260]]
    assert_causal_features(bars, build_features, FEATURE_COLUMNS, cutoffs)


def test_editing_the_future_does_not_move_past_features_but_does_move_labels() -> None:
    bars = _panel()
    dates = pd.to_datetime(bars["date"]).sort_values().unique()
    cutoff = pd.Timestamp(dates[200])
    edited = bars.copy()
    edited["date"] = pd.to_datetime(edited["date"])
    future = edited["date"] > cutoff
    for column in ("open", "high", "low", "close", "adj_close", "volume"):
        edited.loc[future, column] = edited.loc[future, column] * 10.0

    original = build_features(bars)
    rebuilt = build_features(edited)
    past = original["date"] <= cutoff
    pd.testing.assert_frame_equal(
        original.loc[past].reset_index(drop=True),
        rebuilt.loc[past].reset_index(drop=True),
    )

    original_labels = build_labels(bars)
    edited_labels = build_labels(edited)
    on_cutoff = (original_labels["date"] == cutoff) & (original_labels["symbol"] == "AAPL")
    before = original_labels.loc[on_cutoff, "fwd_ret_1d"].item()
    after = edited_labels.loc[on_cutoff, "fwd_ret_1d"].item()
    assert before != pytest.approx(after)


def test_shuffled_rows_produce_the_same_features() -> None:
    bars = _panel()
    shuffled = bars.sample(frac=1.0, random_state=3)
    left = build_features(bars).sort_values(["symbol", "date"]).reset_index(drop=True)
    right = build_features(shuffled).sort_values(["symbol", "date"]).reset_index(drop=True)
    pd.testing.assert_frame_equal(left, right)


def test_feature_modules_do_not_shift_backward_or_center_a_window() -> None:
    root = Path("driftwood/features")
    causal = (root / "factors.py").read_text(encoding="utf-8")
    causal += (root / "transforms.py").read_text(encoding="utf-8")
    causal += (root / "panel.py").read_text(encoding="utf-8")
    assert ".shift(-" not in causal
    assert "center=True" not in causal
    labels = (root / "labels.py").read_text(encoding="utf-8")
    assert ".shift(-1)" in labels
    assert ".shift(-5)" in labels


def test_truncation_check_catches_a_negative_shift_and_a_full_sample_scaler() -> None:
    bars = _panel().sort_values(["symbol", "date"])
    dates = pd.to_datetime(bars["date"]).sort_values().unique()
    cutoffs = [dates[150]]

    def negative_shift(frame: pd.DataFrame) -> pd.DataFrame:
        work = frame.sort_values(["symbol", "date"]).copy()
        work["leak"] = work.groupby("symbol")["adj_close"].shift(-1)
        return work.loc[:, ["symbol", "date", "leak"]]

    def centered(frame: pd.DataFrame) -> pd.DataFrame:
        work = frame.sort_values(["symbol", "date"]).copy()
        work["leak"] = work.groupby("symbol")["adj_close"].transform(
            lambda series: series.rolling(7, center=True).mean()
        )
        return work.loc[:, ["symbol", "date", "leak"]]

    def full_sample_z(frame: pd.DataFrame) -> pd.DataFrame:
        work = frame.sort_values(["symbol", "date"]).copy()
        returns = work.groupby("symbol")["adj_close"].pct_change()
        scale = returns.std(ddof=0)
        work["leak"] = (returns - returns.mean()) / scale
        return work.loc[:, ["symbol", "date", "leak"]]

    with pytest.raises(AssertionError, match="leak"):
        assert_causal_features(bars, negative_shift, ["leak"], cutoffs)
    with pytest.raises(AssertionError, match="leak"):
        assert_causal_features(bars, centered, ["leak"], cutoffs)
    with pytest.raises(AssertionError, match="leak"):
        assert_causal_features(bars, full_sample_z, ["leak"], cutoffs)


def test_same_day_high_is_allowed_and_tomorrows_high_is_not() -> None:
    bars = _panel().sort_values(["symbol", "date"]).reset_index(drop=True)
    bars["date"] = pd.to_datetime(bars["date"])
    target = bars.index[(bars["symbol"] == "AAPL")][40]
    today = bars.loc[target, "date"]
    changed_today = bars.copy()
    changed_today.loc[target, "high"] = changed_today.loc[target, "high"] + 5.0
    original = build_features(bars)
    moved = build_features(changed_today)
    key = (original["symbol"] == "AAPL") & (original["date"] == today)
    assert original.loc[key, "hl_range"].item() != pytest.approx(moved.loc[key, "hl_range"].item())

    tomorrow = bars.index[(bars["symbol"] == "AAPL") & (bars["date"] > today)][0]
    changed_tomorrow = bars.copy()
    changed_tomorrow.loc[tomorrow, "high"] = changed_tomorrow.loc[tomorrow, "high"] + 5.0
    later = build_features(changed_tomorrow)
    assert original.loc[key, "hl_range"].item() == pytest.approx(later.loc[key, "hl_range"].item())
    assert np.isfinite(original.loc[key, "hl_range"].item())
