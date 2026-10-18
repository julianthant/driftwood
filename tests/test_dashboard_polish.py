"""Regression checks for the real-study dashboard presentation."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from dashboard import charts
from dashboard.formatting import compact_money
from driftwood.data.store import MarketStore


def test_model_picker_starts_on_a_saved_backtest(tmp_path, monkeypatch):
    from streamlit.testing.v1 import AppTest

    from driftwood.cli import main

    path = tmp_path / "models.duckdb"
    assert (
        main(
            [
                "ingest",
                "--synthetic",
                "--periods",
                "60",
                "--symbols",
                "AAA",
                "BBB",
                "CCC",
                "--db",
                str(path),
            ]
        )
        == 0
    )
    assert main(["features", "--db", str(path)]) == 0
    assert (
        main(
            [
                "train",
                "--db",
                str(path),
                "--min-train-days",
                "20",
                "--test-days",
                "10",
                "--embargo",
                "2",
            ]
        )
        == 0
    )
    with MarketStore(path) as store:
        store.execute(
            "INSERT INTO predictions SELECT symbol, date, prediction, label, fold, "
            "'momentum' FROM predictions WHERE model = 'ridge'"
        )
    assert main(["backtest", "--db", str(path)]) == 0
    monkeypatch.setenv("DRIFTWOOD_DB", str(path))
    at = AppTest.from_file(str(Path("dashboard/app.py").resolve()), default_timeout=30).run()
    assert not at.exception
    picker = next(item for item in at.sidebar.selectbox if item.label == "Model")
    assert picker.value == "ridge"
    assert len(at.get("plotly_chart")) == 1
    at.sidebar.radio[0].set_value("Model / Signals").run()
    assert not any("Legacy predictions" in item.value for item in at.caption)
    at.sidebar.radio[0].set_value("Overview").run()
    picker = next(item for item in at.sidebar.selectbox if item.label == "Model")
    picker.set_value("momentum").run()
    assert not at.exception
    assert any("no saved backtest" in item.value for item in at.info)
    at.sidebar.radio[0].set_value("Backtest").run()
    assert next(item for item in at.sidebar.selectbox if item.label == "Model").value == "momentum"
    at.sidebar.radio[0].set_value("Model / Signals").run()
    assert any("Legacy predictions" in item.value for item in at.caption)


def test_coefficient_axis_is_explicit_instead_of_micro_units():
    frame = pd.DataFrame({"feature": ["z_ret_1d", "z_vol_21"], "weight": [-0.0002, 0.0003]})
    chart = charts.contribution_chart(frame, coefficient=True)
    assert list(chart.data[0].y) == pytest.approx([-2, 3])
    assert "10⁻⁴" in chart.layout.yaxis.title.text
    assert chart.layout.yaxis.tickformat == ".1f"


def test_relative_cost_omits_flat_immediate_trace_and_keeps_differences():
    dates = pd.date_range("2020-01-01", periods=3)
    daily = pd.DataFrame(
        {
            "algo": np.repeat(["immediate", "twap", "vwap"], 3),
            "date": list(dates) * 3,
            "cumulative_cost": [0, 10, 30, 0, 12, 35, 0, 8, 29],
        }
    )
    chart = charts.cost_chart(daily)
    assert [trace.name for trace in chart.data] == ["TWAP", "VWAP"]
    assert list(chart.data[0].y) == [0, 2, 5]
    assert list(chart.data[1].y) == [0, -2, -1]
    assert chart.layout.yaxis.zeroline is False


def test_comparison_charts_keep_schedule_names_colors_and_units():
    summary = pd.DataFrame(
        {
            "algo": ["vwap", "immediate", "twap"],
            "total_cost": [300, 200, 400],
            "cost_bps": [1.5, 2, 1],
            "total_return": [0.1, -0.1, 0],
            "sharpe": [1, -1, 0],
            "max_drawdown": [-0.1, -0.3, -0.2],
        }
    )
    costs = charts.comparison_chart(summary, costs=True)
    assert list(costs.data[0].x) == ["Immediate", "TWAP", "VWAP"]
    assert list(costs.data[0].y) == [200, 400, 300]
    assert list(costs.data[1].y) == [2, 1, 1.5]
    assert list(costs.data[0].marker.color) == list(charts.COLORS.values())
    assert len(charts.comparison_chart(summary).data) == 3


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (165535.87, "$165.5k"),
        (-165535.87, "-$165.5k"),
        (1_600_000, "$1.6m"),
        (1_600_000_000, "$1.6b"),
        (0, "$0"),
        (np.nan, "—"),
    ],
)
def test_compact_money_keeps_sign_and_scale(value, expected):
    assert compact_money(value) == expected


def test_overlaid_schedule_lines_have_distinct_styles():
    daily = pd.DataFrame(
        {
            "algo": ["immediate", "twap", "vwap"],
            "date": pd.to_datetime(["2020-01-01"] * 3),
            "growth": [1, 1, 1],
        }
    )
    figure = charts.equity_chart(daily)
    assert [trace.name for trace in figure.data] == ["Immediate", "TWAP", "VWAP"]
    assert len({trace.line.dash for trace in figure.data}) == 3


def test_coverage_chart_sorts_shorter_histories_first():
    frame = pd.DataFrame({"symbol": ["CCC", "AAA", "BBB"], "sessions": [15, 5, 15]})
    chart = charts.coverage_chart(frame)
    assert list(chart.data[0].x) == ["AAA", "BBB", "CCC"]
    assert chart.layout.height <= 260
