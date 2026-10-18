"""The train command writes out-of-sample predictions and an IC report, offline."""

import pandas as pd

from driftwood.cli import main
from driftwood.data.store import MarketStore
from driftwood.models.persist import read_predictions, write_predictions


def test_train_command_writes_predictions_and_an_out_of_sample_report(tmp_path) -> None:
    db = tmp_path / "train.duckdb"
    report = tmp_path / "report"
    symbols = ["AAPL", "MSFT", "NVDA", "JPM"]
    assert (
        main(
            [
                "ingest",
                "--synthetic",
                "--periods",
                "60",
                "--symbols",
                *symbols,
                "--db",
                str(db),
            ]
        )
        == 0
    )
    assert main(["features", "--db", str(db)]) == 0
    code = main(
        [
            "train",
            "--db",
            str(db),
            "--model",
            "ridge",
            "--min-train-days",
            "20",
            "--test-days",
            "10",
            "--embargo",
            "2",
            "--report",
            str(report),
        ]
    )
    assert code == 0
    text = (report / "ic.md").read_text(encoding="utf-8")
    assert "out of sample" in text
    assert "not the IC" in text
    assert (report / "ic_daily.csv").exists()
    assert (report / "folds.csv").exists()
    with MarketStore(db) as store:
        predictions = read_predictions(store)
    assert not predictions.empty
    assert set(predictions["model"]) == {"ridge"}
    assert {"prediction", "label", "fold"}.issubset(predictions.columns)


def test_a_second_model_does_not_erase_the_first(tmp_path) -> None:
    frame = pd.DataFrame(
        {
            "symbol": ["AAPL", "MSFT"],
            "date": pd.to_datetime(["2020-01-02", "2020-01-02"]),
            "prediction": [0.1, -0.1],
            "fwd_ret_1d": [0.01, -0.02],
            "fold": [0, 0],
        }
    )
    other = frame.copy()
    other["prediction"] = [0.2, -0.2]
    with MarketStore(tmp_path / "preds.duckdb") as store:
        write_predictions(store, frame, model_name="ridge", label_column="fwd_ret_1d")
        write_predictions(store, other, model_name="momentum", label_column="fwd_ret_1d")
        stored = read_predictions(store)
    assert set(stored["model"]) == {"ridge", "momentum"}
    assert len(stored) == 4


def test_train_before_features_exits_nonzero(tmp_path) -> None:
    db = tmp_path / "empty.duckdb"
    with MarketStore(db):
        pass
    code = main(
        [
            "train",
            "--db",
            str(db),
            "--min-train-days",
            "5",
            "--test-days",
            "5",
            "--embargo",
            "0",
        ]
    )
    assert code == 1


def test_momentum_model_needs_a_year_and_says_so_on_a_short_book(tmp_path) -> None:
    db = tmp_path / "short.duckdb"
    assert (
        main(
            [
                "ingest",
                "--synthetic",
                "--periods",
                "40",
                "--symbols",
                "AAPL",
                "MSFT",
                "NVDA",
                "JPM",
                "--db",
                str(db),
            ]
        )
        == 0
    )
    assert main(["features", "--db", str(db)]) == 0
    code = main(
        [
            "train",
            "--db",
            str(db),
            "--model",
            "momentum",
            "--min-train-days",
            "10",
            "--test-days",
            "5",
            "--embargo",
            "1",
        ]
    )
    assert code == 1
