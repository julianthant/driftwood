"""The backtest trades the next session, and it trades the score rather than the label."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from driftwood.backtest import BacktestConfig, run_backtest, write_backtest_report
from driftwood.backtest.engine import summarize
from driftwood.backtest.prices import adjusted_ohlc
from driftwood.cli import main
from driftwood.data.synthetic import synthetic_bars
from driftwood.execution import execute_order


def test_the_engine_does_not_mention_the_label() -> None:
    source = Path("driftwood/backtest/engine.py").read_text(encoding="utf-8")
    assert "fwd_ret" not in source
    assert "label" not in source


def test_adjusted_prices_undo_a_split_in_the_raw_close() -> None:
    bars = synthetic_bars(["AAA"], periods=10, seed=1, split_symbol="AAA")
    adjusted = adjusted_ohlc(bars)
    split = bars.iloc[5]
    scaled = adjusted.iloc[5]
    factor = split["adj_close"] / split["close"]
    assert factor == pytest.approx(2.0)
    assert scaled["close"] == pytest.approx(split["adj_close"])
    assert scaled["open"] == pytest.approx(split["open"] * factor)
    assert scaled["low"] <= scaled["open"] <= scaled["high"]


def test_scaling_onto_adj_close_keeps_the_bar_inside_the_fill() -> None:
    """A close that sits on the raw high can round just outside after scaling.

    These are vendor prints from the research universe. Before the envelope
    repair, the C++ fill rejected them with "OHLC is inconsistent".
    """

    bars = pd.DataFrame(
        {
            "open": [89.260002, 134.809998],
            "high": [89.989998, 136.800003],
            "low": [87.970001, 134.740005],
            "close": [87.970001, 136.800003],
            "adj_close": [63.036067962646484, 106.09721374511719],
        }
    )
    adjusted = adjusted_ohlc(bars)
    for row in adjusted.itertuples(index=False):
        assert row.low <= row.open <= row.high
        assert row.low <= row.close <= row.high
        execute_order(
            open=row.open,
            high=row.high,
            low=row.low,
            close=row.close,
            volume=1_000_000,
            quantity=1,
            side="buy",
            algo="immediate",
        )


def test_same_day_jump_is_not_a_profit_and_the_fill_is_the_next_session() -> None:
    dates = pd.bdate_range("2018-01-02", periods=4)
    bars = _panel(
        dates,
        {
            "AAA": [100, 100, (100, 200, 100, 200), 200],
            "BBB": [100, 100, 100, 100],
        },
    )
    scores = pd.DataFrame(
        {"symbol": ["AAA", "BBB"], "date": [dates[2], dates[2]], "score": [1.0, -1.0]}
    )
    result = run_backtest(bars, scores, algos=("immediate",), config=_flat_config())
    equity = result.equity.set_index("date")["equity"]
    assert equity.loc[dates[2]] == pytest.approx(1000.0)
    assert result.trades["fill_date"].nunique() == 1
    assert result.trades["fill_date"].iloc[0] == dates[3]
    assert set(result.trades["signal_date"]) == {dates[2]}
    bought = result.trades.loc[result.trades["symbol"] == "AAA"].iloc[0]
    assert bought.raw_vwap == pytest.approx(200.0)
    assert equity.loc[dates[3]] == pytest.approx(1000.0)


def test_changing_todays_score_does_not_change_todays_fill() -> None:
    dates = pd.bdate_range("2018-01-02", periods=4)
    bars = _panel(dates, {"AAA": [100, 100, 100, 100], "BBB": [100, 100, 100, 100]})
    base = _scores(dates[0], 1.0)
    later = pd.concat([base, _scores(dates[2], 1.0)], ignore_index=True)
    flipped = pd.concat([base, _scores(dates[2], -1.0)], ignore_index=True)
    config = _flat_config()
    first = run_backtest(bars, later, algos=("immediate",), config=config)
    second = run_backtest(bars, flipped, algos=("immediate",), config=config)
    early = first.trades["fill_date"] < dates[2]
    pd.testing.assert_frame_equal(
        first.trades.loc[early].reset_index(drop=True),
        second.trades.loc[second.trades["fill_date"] < dates[2]].reset_index(drop=True),
    )
    assert not first.trades.loc[first.trades["fill_date"] == dates[3], "side"].equals(
        second.trades.loc[second.trades["fill_date"] == dates[3], "side"]
    )


def test_sigma_from_the_fill_date_is_ignored() -> None:
    dates = pd.bdate_range("2018-01-02", periods=3)
    bars = _panel(dates, {"AAA": [100, 100, 100], "BBB": [100, 100, 100]}, volume=100)
    scores = _scores(dates[0], 1.0)
    volatility = pd.DataFrame(
        {
            "symbol": ["AAA", "BBB", "AAA", "BBB"],
            "date": [dates[0], dates[0], dates[1], dates[1]],
            "sigma": [0.0, 0.0, 1.0, 1.0],
        }
    )
    config = BacktestConfig(
        initial_capital=1000.0,
        gross=1.0,
        max_weight=0.5,
        half_spread_bps=0.0,
        impact_coef=1.0,
        participation_limit=1.0,
        n_bins=1,
    )
    result = run_backtest(
        bars, scores, algos=("immediate",), config=config, volatility=volatility
    )
    assert result.trades["cost"].sum() == pytest.approx(0.0)
    assert not result.trades.empty


def test_zero_cost_flat_schedules_match_and_a_spread_round_trip_loses_money() -> None:
    dates = pd.bdate_range("2018-01-02", periods=4)
    bars = _panel(dates, {"AAA": [100, 100, 100, 100], "BBB": [100, 100, 100, 100]})
    scores = pd.concat([_scores(dates[0], 1.0), _scores(dates[1], 0.0)], ignore_index=True)
    flat = run_backtest(bars, scores, config=_flat_config())
    finals = flat.equity.groupby("algo")["equity"].last()
    assert finals.nunique() == 1
    assert finals.iloc[0] == pytest.approx(1000.0)
    costly = run_backtest(
        bars,
        scores,
        config=BacktestConfig(
            initial_capital=1000.0,
            gross=1.0,
            max_weight=0.5,
            half_spread_bps=10.0,
            participation_limit=1.0,
            n_bins=1,
        ),
    )
    assert (costly.summary["total_cost"] > 0).all()
    assert (costly.summary["total_return"] < 0).all()


def test_a_wide_bar_makes_immediate_and_twap_pay_different_prices() -> None:
    dates = pd.bdate_range("2018-01-02", periods=3)
    bars = _panel(
        dates,
            {"AAA": [100, (100, 140, 80, 130), 130], "BBB": [100, (100, 120, 60, 70), 70]},
    )
    result = run_backtest(bars, _scores(dates[0], 1.0), config=_flat_config())
    prices = result.trades.loc[result.trades["symbol"] == "AAA"].set_index("algo")["raw_vwap"]
    assert prices["immediate"] == pytest.approx(100.0)
    assert prices["twap"] != pytest.approx(prices["immediate"])
    ends = result.equity.groupby("algo")["equity"].last()
    assert ends["immediate"] != pytest.approx(ends["twap"])


def test_the_deadband_skips_a_small_book() -> None:
    dates = pd.bdate_range("2018-01-02", periods=3)
    bars = _panel(dates, {"AAA": [100, 100, 100], "BBB": [100, 100, 100]})
    config = BacktestConfig(
        initial_capital=1000.0,
        gross=1.0,
        max_weight=0.5,
        min_trade_dollars=1_000_000.0,
        half_spread_bps=0.0,
        participation_limit=1.0,
    )
    result = run_backtest(bars, _scores(dates[0], 1.0), algos=("twap",), config=config)
    assert result.trades.empty
    assert result.equity["equity"].iloc[-1] == pytest.approx(1000.0)


def test_summary_drawdown_and_a_flat_sharpe() -> None:
    equity = pd.DataFrame(
        {
            "algo": ["immediate", "immediate", "immediate"],
            "equity": [100.0, 110.0, 90.0],
            "ret": [np.nan, 0.1, 90 / 110 - 1],
            "turnover": [0.0, 0.0, 0.0],
        }
    )
    summary = summarize(equity, pd.DataFrame(columns=["algo", "cost"]))
    assert summary.loc[0, "max_drawdown"] == pytest.approx(90 / 110 - 1)
    assert summary.loc[0, "total_return"] == pytest.approx(-0.1)
    flat = equity.copy()
    flat["ret"] = [np.nan, 0.01, 0.01]
    flat["equity"] = [100.0, 101.0, 102.01]
    assert np.isnan(summarize(flat, pd.DataFrame(columns=["algo", "cost"])).loc[0, "sharpe"])


def test_backtest_command_writes_the_comparison(tmp_path) -> None:
    db = tmp_path / "bt.duckdb"
    report = tmp_path / "report"
    symbols = ["AAPL", "MSFT", "NVDA", "JPM"]
    assert (
        main(
            ["ingest", "--synthetic", "--periods", "70", "--symbols", *symbols, "--db", str(db)]
        )
        == 0
    )
    assert main(["features", "--db", str(db)]) == 0
    assert (
        main(
            [
                "train",
                "--db",
                str(db),
                "--min-train-days",
                "30",
                "--test-days",
                "15",
                "--embargo",
                "2",
            ]
        )
        == 0
    )
    code = main(
        [
            "backtest",
            "--db",
            str(db),
            "--max-weight",
            "0.5",
            "--half-spread-bps",
            "1",
            "--report",
            str(report),
        ]
    )
    assert code == 0
    text = (report / "backtest.md").read_text(encoding="utf-8")
    assert "t+1" in text
    assert "immediate" in text and "twap" in text and "vwap" in text
    assert main(["backtest", "--db", str(tmp_path / "empty.duckdb")]) == 1


def _flat_config() -> BacktestConfig:
    return BacktestConfig(
        initial_capital=1000.0,
        gross=1.0,
        max_weight=0.5,
        half_spread_bps=0.0,
        participation_limit=1.0,
        n_bins=1,
    )


def _scores(date: pd.Timestamp, direction: float) -> pd.DataFrame:
    return pd.DataFrame(
        {"symbol": ["AAA", "BBB"], "date": [date, date], "score": [direction, -direction]}
    )


def _panel(dates, paths: dict[str, list[object]], volume: float = 1e9) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for symbol, closes in paths.items():
        for date, spec in zip(dates, closes, strict=True):
            if isinstance(spec, tuple):
                open_, high, low, close = spec
            else:
                open_ = high = low = close = float(spec)
            rows.append(
                {
                    "symbol": symbol,
                    "date": date,
                    "open": open_,
                    "high": high,
                    "low": low,
                    "close": close,
                    "adj_close": close,
                    "volume": volume,
                }
            )
    return pd.DataFrame(rows)


def test_report_mentions_the_next_session(tmp_path) -> None:
    dates = pd.bdate_range("2018-01-02", periods=3)
    bars = _panel(dates, {"AAA": [100, 100, 100], "BBB": [100, 100, 100]})
    result = run_backtest(bars, _scores(dates[0], 1.0), config=_flat_config())
    path = write_backtest_report(result, tmp_path)
    assert "session t+1" in path.read_text(encoding="utf-8")
