"""Dollar-neutral weights, one clip, and one refill of the room under the cap."""

import pandas as pd
import pytest

from driftwood.portfolio import construct_weights


def test_a_symmetric_book_is_dollar_neutral_and_grosses_to_one() -> None:
    scores = pd.Series({"A": -2.0, "B": -1.0, "C": 1.0, "D": 2.0})
    weights = construct_weights(scores, max_weight=0.5)
    assert weights.sum() == pytest.approx(0.0)
    assert weights.abs().sum() == pytest.approx(1.0)
    assert weights["D"] == pytest.approx(1 / 3)
    assert weights["A"] == pytest.approx(-1 / 3)


def test_an_all_positive_score_is_long_the_winners_and_short_the_laggards() -> None:
    weights = construct_weights(pd.Series({"A": 1.0, "B": 2.0, "C": 3.0}), max_weight=0.6)
    assert weights["A"] == pytest.approx(-0.5)
    assert weights["B"] == pytest.approx(0.0)
    assert weights["C"] == pytest.approx(0.5)
    assert weights.sum() == pytest.approx(0.0)


def test_the_cap_is_not_broken_to_rebuild_the_gross() -> None:
    weights = construct_weights(
        pd.Series({"A": -2.0, "B": -1.0, "C": 1.0, "D": 2.0}),
        gross=1.0,
        max_weight=0.2,
    )
    assert weights.abs().max() <= 0.2 + 1e-12
    assert weights.abs().sum() == pytest.approx(0.8)
    assert weights.sum() == pytest.approx(0.0)


def test_an_asymmetric_cap_can_leave_a_net() -> None:
    scores = pd.Series({"A": 3.0, "B": 0.0, "C": 0.0, "D": -1.0})
    weights = construct_weights(scores, max_weight=0.2)
    assert weights.abs().max() <= 0.2 + 1e-12
    assert weights.sum() != pytest.approx(0.0)
    assert weights.abs().sum() < 1.0


def test_missing_scores_are_dropped_and_a_flat_cross_section_is_zero() -> None:
    weights = construct_weights(pd.Series({"A": 1.0, "B": float("nan"), "C": 1.0}))
    assert list(weights.index) == ["A", "C"]
    assert weights.sum() == pytest.approx(0.0)
    with pytest.raises(ValueError, match="max_weight"):
        construct_weights(pd.Series({"A": 1.0, "B": -1.0}), max_weight=0)
