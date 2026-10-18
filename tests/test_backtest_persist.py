"""Backtest snapshots retain exact accounting and replace a model atomically."""

from dataclasses import replace

import duckdb
import pandas as pd
import pytest

from driftwood.backtest import run_backtest
from driftwood.backtest.persist import input_signature, write_backtest
from driftwood.data.store import MarketStore
from driftwood.data.synthetic import synthetic_bars
from driftwood.models.persist import write_predictions


def test_snapshot_roundtrip_replacement_and_rollback(tmp_path):
    bars = synthetic_bars(["AAA", "BBB"], periods=5)
    scores = bars.loc[:, ["symbol", "date"]].copy()
    scores["score"] = [1.0] * 5 + [-1.0] * 5
    result = run_backtest(bars, scores)
    with MarketStore(tmp_path / "backtest.duckdb") as store:
        write_backtest(store, result, model="ridge", signature="first")
        write_backtest(store, result, model="momentum", signature="second")
        write_backtest(store, result, model="ridge", signature="replacement")
        daily = store.execute("SELECT * FROM backtest_daily WHERE model = 'ridge'").df()
        assert len(daily) == len(result.equity)
        assert daily["cost"].sum() == pytest.approx(result.trades["cost"].sum())
        saved = store.execute("SELECT * FROM backtest_summary ORDER BY model, algo").df()
        assert len(saved) == 6
        broken = replace(result, summary=result.summary.drop(columns="total_return"))
        with pytest.raises(duckdb.Error):
            write_backtest(store, broken, model="ridge", signature="must-rollback")
        pd.testing.assert_frame_equal(
            saved, store.execute("SELECT * FROM backtest_summary ORDER BY model, algo").df()
        )
        assert (
            store.execute("SELECT signature FROM backtest_runs WHERE model = 'ridge'").fetchone()[0]
            == "replacement"
        )
        assert store.execute(
            "SELECT COUNT(*) FROM backtest_daily WHERE model = 'ridge'"
        ).fetchone()[0] == len(result.equity)


def test_signature_tracks_inputs_and_prediction_metadata_tracks_label(tmp_path):
    bars = synthetic_bars(["AAA", "BBB"], periods=5)
    bars["source"] = "synthetic"
    bars["ingested_at"] = pd.Timestamp("2020-01-01")
    predictions = bars.loc[:, ["symbol", "date"]].copy()
    predictions["prediction"] = [0.1] * 5 + [-0.1] * 5
    predictions["fwd_ret_5d"] = 0.0
    predictions["fold"] = 0
    with MarketStore(tmp_path / "signatures.duckdb") as store:
        store.upsert_bars(bars)
        write_predictions(store, predictions, model_name="ridge", label_column="fwd_ret_5d")
        original = input_signature(store, "ridge")
        assert (
            store.execute(
                "SELECT label_column FROM model_metadata WHERE model = 'ridge'"
            ).fetchone()[0]
            == "fwd_ret_5d"
        )
        store.execute("UPDATE predictions SET prediction = prediction + 1")
        assert input_signature(store, "ridge") != original
        previous = input_signature(store, "ridge")
        store.execute("UPDATE bars SET close = close * 1.001")
        assert input_signature(store, "ridge") != previous
