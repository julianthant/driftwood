import numpy as np
import pandas as pd
import pytest

from driftwood.cli import main
from driftwood.data.store import MarketStore
from driftwood.data.synthetic import synthetic_bars
from driftwood.features.factors import FEATURE_COLUMNS, LABEL_COLUMNS
from driftwood.features.persist import read_features, read_labels, write_research_tables
from driftwood.features.pipeline import build_research_frame


def test_tables_round_trip_and_features_have_no_label_column(tmp_path) -> None:
    bars = synthetic_bars(["AAPL", "MSFT"], periods=40, seed=5)
    frame = build_research_frame(bars)
    with MarketStore(tmp_path / "feat.duckdb") as store:
        feature_rows, label_rows = write_research_tables(store, frame)
        assert feature_rows == len(frame)
        assert label_rows == len(frame)
        stored_features = read_features(store)
        stored_labels = read_labels(store)
        assert "fwd_ret_1d" not in stored_features.columns
        assert list(stored_labels.columns) == ["symbol", "date", *LABEL_COLUMNS]
        assert set(FEATURE_COLUMNS).issubset(stored_features.columns)

    expected = frame.sort_values(["symbol", "date"]).reset_index(drop=True)
    got = stored_features.sort_values(["symbol", "date"]).reset_index(drop=True)
    pd.testing.assert_series_equal(
        got["mom_21"].reset_index(drop=True),
        expected["mom_21"].reset_index(drop=True),
        check_names=False,
    )
    labels = stored_labels.sort_values(["symbol", "date"]).reset_index(drop=True)
    pd.testing.assert_series_equal(
        labels["fwd_ret_1d"],
        expected["fwd_ret_1d"],
        check_names=False,
    )


def test_missing_feature_table_is_a_clear_error(tmp_path) -> None:
    with MarketStore(tmp_path / "empty.duckdb") as store:
        with pytest.raises(RuntimeError, match="driftwood features"):
            read_features(store)


def test_features_command_builds_tables_from_synthetic_bars(tmp_path) -> None:
    db = tmp_path / "cli.duckdb"
    assert (
        main(
            [
                "ingest",
                "--synthetic",
                "--periods",
                "30",
                "--symbols",
                "AAPL",
                "MSFT",
                "--db",
                str(db),
            ]
        )
        == 0
    )
    assert main(["features", "--db", str(db)]) == 0
    with MarketStore(db) as store:
        features = read_features(store)
        assert len(features) == 60
        assert features["z_ret_1d"].notna().sum() > 0
        assert np.isnan(features["z_mom_12_1"]).all()


def test_features_command_refuses_an_empty_database(tmp_path) -> None:
    db = tmp_path / "no-bars.duckdb"
    MarketStore(db).close()
    assert main(["features", "--db", str(db)]) == 1
