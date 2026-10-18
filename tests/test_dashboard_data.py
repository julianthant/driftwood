"""Offline dashboard contracts, including persistence and enforced read-only access."""

import json
import subprocess
import sys
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from dashboard import charts, data, views
from driftwood.backtest import BacktestConfig, run_backtest
from driftwood.backtest.persist import input_signature, write_backtest
from driftwood.data.store import MarketStore
from driftwood.data.synthetic import synthetic_bars
from driftwood.features.persist import write_research_tables
from driftwood.features.pipeline import build_research_frame
from driftwood.models.persist import write_predictions


@pytest.fixture
def study_db(tmp_path):
    path = tmp_path / "study.duckdb"
    bars = synthetic_bars(["AAA", "BBB", "CCC"], periods=90, seed=7)
    bars["source"] = "synthetic"
    bars["ingested_at"] = pd.Timestamp("2020-01-01")
    research = build_research_frame(bars)
    predictions = research.loc[:, ["symbol", "date", "fwd_ret_1d"]].copy()
    predictions["prediction"] = predictions["symbol"].map({"AAA": -0.1, "BBB": 0.0, "CCC": 0.1})
    predictions["fold"] = 0
    with MarketStore(path) as store:
        store.upsert_bars(bars)
        store.replace_universe(
            pd.DataFrame(
                {
                    "symbol": ["AAA", "BBB", "CCC", "MISSING"],
                    "name": ["a", "b", "c", "missing"],
                    "sector": ["test"] * 4,
                }
            )
        )
        write_research_tables(store, research)
        write_predictions(store, predictions, model_name="ridge", label_column="fwd_ret_1d")
        scores = predictions.rename(columns={"prediction": "score"})
        result = run_backtest(bars, scores, config=BacktestConfig(max_weight=0.5))
        write_backtest(store, result, model="ridge", signature=input_signature(store, "ridge"))
    return path, result


def test_helpers_import_without_ui_or_broker():
    process = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; import dashboard.data, dashboard.views, "
            "dashboard.charts; assert 'streamlit' not in sys.modules; "
            "assert not any(m.startswith('driftwood.live') or m.startswith('alpaca') "
            "for m in sys.modules)",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert process.returncode == 0, process.stderr


def test_read_only_connection_cannot_write_or_create(tmp_path, study_db):
    path, _ = study_db
    with MarketStore(path, read_only=True) as store:
        with pytest.raises(duckdb.Error):
            store.execute("DELETE FROM bars")
    missing = tmp_path / "absent" / "study.duckdb"
    with pytest.raises(duckdb.Error):
        MarketStore(missing, read_only=True)
    assert not missing.parent.exists()


def test_dashboard_queries_leave_database_unchanged(study_db):
    path, result = study_db
    before = path.read_bytes()
    revision = data.file_revision(str(path))
    study = data.load_study(str(path))
    assert study.sessions == 90
    absent = study.coverage.set_index("symbol").loc["MISSING"]
    assert absent["sessions"] == 0 and absent["missing"] == 90
    assert study.missing.loc["MISSING"].eq(1).all()
    assert study.missing.loc["AAA"].eq(0).all()
    assert study.models == ("ridge",) and study.sources == ("synthetic",)
    quality = data.load_quality(str(path), "ridge")
    assert quality.label == "fwd_ret_1d" and quality.hac_lags == 1
    assert len(quality.daily) == 90
    assert quality.daily["rolling_ic"].iloc[:62].isna().all()
    attribution = data.load_attribution(str(path), "ridge")
    assert attribution.by_signal["contribution"].sum() == pytest.approx(
        attribution.portfolio_return, abs=1e-10
    )
    saved = data.load_backtest(str(path), "ridge")
    assert_frame_equal(
        saved.summary.loc[:, result.summary.columns].reset_index(drop=True),
        result.summary.sort_values("algo").reset_index(drop=True),
    )
    for algo, frame in saved.daily.groupby("algo"):
        summary = saved.summary.set_index("algo").loc[algo]
        assert frame["growth"].iloc[0] == 1
        assert frame["growth"].iloc[-1] - 1 == pytest.approx(summary["total_return"])
        assert frame["drawdown"].min() == pytest.approx(summary["max_drawdown"])
        assert frame["cumulative_cost"].iloc[-1] == pytest.approx(summary["total_cost"])
    assert saved.config["max_weight"] == 0.5
    assert before == path.read_bytes()
    assert revision == data.file_revision(str(path))


def test_snapshot_replacement_keeps_other_models_and_invalidates_corrected_inputs(study_db):
    path, result = study_db
    with MarketStore(path) as store:
        write_backtest(store, result, model="other", signature="other-inputs")
        write_backtest(store, result, model="ridge", signature=input_signature(store, "ridge"))
        counts = store.execute(
            "SELECT model, COUNT(*) AS n FROM backtest_daily GROUP BY model ORDER BY model"
        ).df()
        assert list(counts["n"]) == [270, 270]
        assert list(counts["model"]) == ["other", "ridge"]
        store.execute("UPDATE predictions SET prediction = prediction + 1 WHERE model = 'ridge'")
    with pytest.raises(RuntimeError, match="changed"):
        data.load_backtest(str(path), "ridge")


@pytest.mark.parametrize(
    "update",
    [
        "UPDATE bars SET adj_close = adj_close * 1.001",
        "UPDATE features SET vol_21 = COALESCE(vol_21, 0) + 0.001",
    ],
)
def test_snapshot_detects_price_and_volatility_corrections(study_db, update):
    path, _ = study_db
    with MarketStore(path) as store:
        store.execute(update)
    with pytest.raises(RuntimeError, match="changed"):
        data.load_backtest(str(path), "ridge")


def test_missing_results_and_legacy_horizon(study_db):
    path, _ = study_db
    with pytest.raises(RuntimeError, match="no saved backtest"):
        data.load_backtest(str(path), "untrained")
    with MarketStore(path) as store:
        store.execute("DROP TABLE model_metadata")
        store.execute("DROP TABLE backtest_runs")
    quality = data.load_quality(str(path), "ridge")
    assert quality.hac_lags == 5 and "legacy" in quality.label
    with pytest.raises(RuntimeError, match="No saved backtest"):
        data.load_backtest(str(path), "ridge")


def test_time_series_transformations_preserve_gaps_and_schedule_isolation():
    daily = pd.DataFrame(
        {
            "algo": ["a", "b", "a", "b"],
            "date": pd.to_datetime(["2020-01-01", "2020-01-01", "2020-01-02", "2020-01-02"]),
            "equity": [100, 200, 90, 220],
            "cost": [1, 2, 3, 4],
        }
    )
    transformed = views.backtest_series(daily)
    assert list(transformed["growth"]) == pytest.approx([1, 0.9, 1, 1.1])
    assert list(transformed["drawdown"]) == pytest.approx([0, -0.1, 0, 0])
    assert list(transformed["cumulative_cost"]) == [1, 4, 2, 6]
    daily_ic = pd.DataFrame(
        {"date": pd.date_range("2020-01-01", periods=5), "ic": [0.5, -0.2, np.nan, 0.3, 0.1]}
    )
    series = views.ic_series(daily_ic, window=2)
    assert series["rolling_ic"].iloc[1] == pytest.approx(0.15)
    assert series["rolling_ic"].iloc[2:4].isna().all()
    assert series["cumulative_ic"].iloc[-1] == pytest.approx(0.7)
    assert np.isnan(series["cumulative_ic"].iloc[2])


def test_missing_view_uses_observed_sessions_not_weekdays():
    dates = pd.DataFrame(
        {
            "symbol": ["AAA", "AAA", "BBB"],
            "date": pd.to_datetime(["2020-01-02", "2020-01-06", "2020-01-02"]),
        }
    )
    missing = views.missing_by_month(dates, ["AAA", "BBB", "CCC"])
    assert list(missing.iloc[:, 0]) == [0, 0.5, 1]


def test_chart_series_have_consistent_units(study_db):
    path, _ = study_db
    result = data.load_backtest(str(path), "ridge")
    equity = charts.equity_chart(result.daily)
    assert len(equity.data) == 3
    assert {trace.line.color for trace in equity.data} == set(charts.COLORS.values())
    bars = charts.contribution_chart(pd.DataFrame({"signal": ["a"], "contribution": [0.01]}))
    assert list(bars.data[0].y) == [100]
    assert json.loads(equity.to_json())["layout"]["yaxis"]["title"]["text"] == "Growth of $1"


def test_file_revision_tracks_wal(tmp_path):
    path = tmp_path / "revision.duckdb"
    path.write_bytes(b"database")
    revision = data.file_revision(str(path))
    Path(str(path) + ".wal").write_bytes(b"committed update")
    assert data.file_revision(str(path)) != revision


def test_all_dashboard_pages_render_and_schedule_switches(study_db, monkeypatch):
    from streamlit.testing.v1 import AppTest

    path, result = study_db
    monkeypatch.setenv("DRIFTWOOD_DB", str(path))
    at = AppTest.from_file(str(Path("dashboard/app.py").resolve()), default_timeout=30).run()
    expected = {
        "Overview": 1,
        "Backtest": 2,
        "Execution": 2,
        "Model / Signals": 3,
        "Attribution": 1,
        "Data": 2,
    }
    for page, charts_count in expected.items():
        at.sidebar.radio[0].set_value(page).run()
        assert not at.exception, [e.message for e in at.exception]
        assert len(at.get("plotly_chart")) == charts_count, page
        assert not at.error
    at.sidebar.radio[0].set_value("Backtest").run()
    schedule = next(item for item in at.sidebar.selectbox if item.label == "Schedule")
    schedule.set_value("twap").run()
    assert not at.exception
    expected_return = result.summary.set_index("algo").loc["twap", "total_return"]
    assert at.metric[0].value == f"{expected_return:.2%}"


@pytest.mark.parametrize("state", ["missing", "empty", "corrupt", "bars_only"])
def test_friendly_database_empty_states(tmp_path, monkeypatch, state):
    from streamlit.testing.v1 import AppTest

    path = tmp_path / f"{state}.duckdb"
    if state == "corrupt":
        path.write_text("not a database", encoding="utf-8")
    elif state in {"empty", "bars_only"}:
        with MarketStore(path) as store:
            if state == "bars_only":
                bars = synthetic_bars(["AAA", "BBB"], periods=10)
                bars["source"] = "synthetic"
                bars["ingested_at"] = pd.Timestamp("2020-01-01")
                store.upsert_bars(bars)
    monkeypatch.setenv("DRIFTWOOD_DB", str(path))
    at = AppTest.from_file(str(Path("dashboard/app.py").resolve()), default_timeout=30).run()
    assert not at.exception
    assert at.info
    assert any("driftwood" in item.value and str(path) in item.value for item in at.code)
    if state == "missing":
        assert not path.exists()
    if state == "bars_only":
        at.sidebar.radio[0].set_value("Data").run()
        assert len(at.get("plotly_chart")) == 2
        assert not at.exception


@pytest.mark.parametrize("state", ["backtests", "attribution", "stale"])
def test_partial_studies_keep_navigation_and_give_recovery_commands(study_db, monkeypatch, state):
    from streamlit.testing.v1 import AppTest

    path, _ = study_db
    with MarketStore(path) as store:
        if state == "backtests":
            store.execute("DROP TABLE backtest_runs")
        elif state == "attribution":
            store.execute("DROP TABLE features")
        else:
            store.execute("UPDATE predictions SET prediction = prediction + 1")
    monkeypatch.setenv("DRIFTWOOD_DB", str(path))
    at = AppTest.from_file(str(Path("dashboard/app.py").resolve()), default_timeout=30).run()
    page = "Attribution" if state == "attribution" else "Backtest"
    at.sidebar.radio[0].set_value(page).run()
    assert not at.exception
    assert at.info
    expected_command = "attribute" if state == "attribution" else "backtest"
    assert any(f"driftwood {expected_command}" in item.value for item in at.code)
    at.sidebar.radio[0].set_value("Data").run()
    assert not at.exception
    assert len(at.get("plotly_chart")) == 2
