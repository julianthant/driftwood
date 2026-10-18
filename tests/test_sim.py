"""Property tests for the C++ fill. Python does not have a second implementation."""

from pathlib import Path

import driftwood._sim as sim
import pytest

from driftwood.execution import execute_order

OPEN, HIGH, LOW, CLOSE = 100.0, 110.0, 90.0, 105.0
DOWN = (100.0, 110.0, 90.0, 95.0)


def test_extension_is_the_native_module() -> None:
    assert sim.__file__.endswith((".so", ".pyd", ".dylib"))


def test_the_python_wrapper_does_not_price_a_fill() -> None:
    source = Path("driftwood/execution.py").read_text(encoding="utf-8")
    assert "sqrt" not in source
    assert "path_price" not in source
    native = sim.simulate(
        OPEN, HIGH, LOW, CLOSE, 1_000_000, 40, "buy", "twap",
        half_spread_bps=1.0, participation_limit=1.0, n_bins=4,
    )
    wrapped = execute_order(
        open=OPEN, high=HIGH, low=LOW, close=CLOSE, volume=1_000_000,
        quantity=40, side="buy", algo="twap", half_spread_bps=1.0,
        participation_limit=1.0, n_bins=4,
    )
    assert wrapped.filled == native.filled
    assert wrapped.cash == native.cash
    assert [bin.fill_price for bin in wrapped.bins] == [bin.fill_price for bin in native.bins]


def test_path_stays_inside_the_bar_and_the_anchors_match_the_day() -> None:
    assert sim.session_anchors(OPEN, HIGH, LOW, CLOSE) == [OPEN, LOW, HIGH, CLOSE]
    assert sim.session_anchors(*DOWN) == [DOWN[0], DOWN[1], DOWN[2], DOWN[3]]
    for bar in ((OPEN, HIGH, LOW, CLOSE), DOWN, (50.0, 50.0, 50.0, 50.0)):
        for n_bins in (1, 3, 8, 13):
            prices = sim.bin_prices(*bar, n_bins)
            assert len(prices) == n_bins
            assert min(prices) >= bar[2] - 1e-12
            assert max(prices) <= bar[1] + 1e-12
    # Bin 0 of the path is not the open. Immediate is the schedule that uses the open.
    assert sim.bin_prices(OPEN, HIGH, LOW, CLOSE, 8)[0] != pytest.approx(OPEN)


def test_volume_profile_is_a_symmetric_u() -> None:
    profile = sim.volume_profile(8)
    assert sum(profile) == pytest.approx(1.0)
    assert profile[0] == pytest.approx(profile[-1])
    assert profile[0] > profile[len(profile) // 2]
    assert all(weight > 0 for weight in profile)


def test_immediate_small_order_prints_at_the_open() -> None:
    fill = sim.simulate(
        OPEN, HIGH, LOW, CLOSE, 1_000_000, 25, "buy", "immediate",
        half_spread_bps=0.0, participation_limit=1.0, n_bins=8,
    )
    assert fill.filled == pytest.approx(25)
    assert fill.unfilled == pytest.approx(0)
    assert len(fill.bins) == 1
    assert fill.bins[0].bin == 0
    assert fill.bins[0].raw_price == pytest.approx(OPEN)
    assert fill.raw_vwap == pytest.approx(OPEN)


def test_twap_splits_equally_and_vwap_follows_the_profile() -> None:
    twap = sim.simulate(
        OPEN, HIGH, LOW, CLOSE, 1e9, 80, "buy", "twap",
        half_spread_bps=0.0, participation_limit=1.0, n_bins=8,
    )
    assert [bin.quantity for bin in twap.bins] == pytest.approx([10.0] * 8)
    profile = sim.volume_profile(8)
    vwap = sim.simulate(
        OPEN, HIGH, LOW, CLOSE, 1e9, 80, "sell", "vwap",
        half_spread_bps=0.0, participation_limit=1.0, n_bins=8,
    )
    expected = [80.0 * weight for weight in profile]
    assert [bin.quantity for bin in vwap.bins] == pytest.approx(expected)
    assert vwap.filled == pytest.approx(80)


def test_participation_cap_carries_the_leftover_forward() -> None:
    volume = 1000.0
    limit = 0.1
    profile = sim.volume_profile(4)
    fill = sim.simulate(
        OPEN, HIGH, LOW, CLOSE, volume, 500, "buy", "twap",
        half_spread_bps=0.0, participation_limit=limit, n_bins=4,
    )
    caps = [limit * volume * weight for weight in profile]
    assert [bin.quantity for bin in fill.bins] == pytest.approx(caps)
    assert fill.filled == pytest.approx(sum(caps))
    assert fill.unfilled == pytest.approx(500 - sum(caps))
    assert fill.bins[0].raw_price != pytest.approx(fill.bins[-1].raw_price)


def test_immediate_walks_forward_only_after_the_open_is_full() -> None:
    fill = sim.simulate(
        OPEN, HIGH, LOW, CLOSE, 1000, 100, "buy", "immediate",
        half_spread_bps=0.0, participation_limit=0.1, n_bins=4,
    )
    assert fill.bins[0].raw_price == pytest.approx(OPEN)
    assert len(fill.bins) > 1
    assert any(bin.raw_price != pytest.approx(OPEN) for bin in fill.bins[1:])


def test_costs_make_buys_pay_more_and_a_round_trip_lose_money() -> None:
    buy = sim.simulate(
        OPEN, HIGH, LOW, CLOSE, 1e9, 50, "buy", "immediate",
        half_spread_bps=1.0, participation_limit=1.0, n_bins=4,
    )
    sell = sim.simulate(
        OPEN, HIGH, LOW, CLOSE, 1e9, 50, "sell", "immediate",
        half_spread_bps=1.0, participation_limit=1.0, n_bins=4,
    )
    assert buy.bins[0].fill_price > buy.bins[0].raw_price
    assert sell.bins[0].fill_price < sell.bins[0].raw_price
    assert buy.cash < 0
    assert sell.cash > 0
    assert buy.cash + sell.cash < 0

    no_cost_buy = sim.simulate(
        OPEN, HIGH, LOW, CLOSE, 1e9, 50, "buy", "twap",
        half_spread_bps=0.0, participation_limit=1.0, n_bins=4,
    )
    no_cost_sell = sim.simulate(
        OPEN, HIGH, LOW, CLOSE, 1e9, 50, "sell", "twap",
        half_spread_bps=0.0, participation_limit=1.0, n_bins=4,
    )
    assert no_cost_buy.cash + no_cost_sell.cash == pytest.approx(0.0)
    for bin in no_cost_buy.bins:
        assert bin.fill_price == pytest.approx(bin.raw_price)


def test_commission_is_in_the_cash_and_not_in_the_fill_price() -> None:
    fill = sim.simulate(
        100, 100, 100, 100, 1e6, 10, "buy", "immediate",
        commission_per_share=0.01, half_spread_bps=0.0, participation_limit=1.0, n_bins=1,
    )
    assert fill.bins[0].fill_price == pytest.approx(100)
    assert fill.cash == pytest.approx(-(100 * 10 + 0.01 * 10))


def test_impact_scales_with_the_square_root_of_participation() -> None:
    small = sim.simulate(
        50, 50, 50, 50, 1000, 40, "buy", "immediate",
        half_spread_bps=0.0, impact_coef=1.0, sigma=0.02,
        participation_limit=1.0, n_bins=1,
    )
    large = sim.simulate(
        50, 50, 50, 50, 1000, 160, "buy", "immediate",
        half_spread_bps=0.0, impact_coef=1.0, sigma=0.02,
        participation_limit=1.0, n_bins=1,
    )
    assert small.bins[0].participation == pytest.approx(0.04)
    assert small.bins[0].fill_price == pytest.approx(50 * 1.004)
    assert large.bins[0].fill_price == pytest.approx(50 * 1.008)
    sell = sim.simulate(
        50, 50, 50, 50, 1000, 40, "sell", "immediate",
        half_spread_bps=0.0, impact_coef=1.0, sigma=0.02,
        participation_limit=1.0, n_bins=1,
    )
    assert sell.bins[0].fill_price == pytest.approx(50 * 0.996)
    assert sell.bins[0].fill_price < sell.bins[0].raw_price


def test_zero_quantity_is_empty_and_zero_volume_is_unfilled() -> None:
    empty = sim.simulate(OPEN, HIGH, LOW, CLOSE, 1000, 0, "buy", "vwap")
    assert empty.filled == 0
    assert empty.unfilled == 0
    assert empty.cash == 0
    assert list(empty.bins) == []
    missed = sim.simulate(OPEN, HIGH, LOW, CLOSE, 0, 10, "sell", "twap", n_bins=4)
    assert missed.filled == 0
    assert missed.unfilled == pytest.approx(10)
    assert missed.cash == 0
    assert list(missed.bins) == []


def test_bad_orders_are_rejected() -> None:
    with pytest.raises(ValueError, match="negative"):
        sim.simulate(OPEN, HIGH, LOW, CLOSE, 1000, -1, "buy", "twap")
    with pytest.raises(ValueError, match="OHLC"):
        sim.simulate(100, 90, 110, 100, 1000, 1, "buy", "twap")
    with pytest.raises(ValueError, match="algo"):
        sim.simulate(OPEN, HIGH, LOW, CLOSE, 1000, 1, "buy", "market")
    with pytest.raises(ValueError, match="participation"):
        sim.simulate(OPEN, HIGH, LOW, CLOSE, 1000, 1, "buy", "twap", participation_limit=0)
    with pytest.raises(ValueError, match="sigma"):
        sim.simulate(OPEN, HIGH, LOW, CLOSE, 1000, 1, "buy", "twap", sigma=-0.1)


def test_a_sell_price_does_not_go_negative() -> None:
    fill = sim.simulate(
        100, 100, 100, 100, 1e6, 5, "sell", "immediate",
        half_spread_bps=20_000, participation_limit=1.0, n_bins=1,
    )
    assert fill.bins[0].fill_price == 0.0
    assert fill.cash == pytest.approx(0.0)
