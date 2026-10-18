"""Attribution is an identity: the pieces add up to weight times return."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from driftwood.attribution import attribute_return, attribute_store, write_attribution_report
from driftwood.cli import main
from driftwood.data.store import MarketStore
from driftwood.models.persist import read_coefficients, write_coefficients, write_predictions


def test_contributions_sum_to_the_book_including_a_zero_exposure_name() -> None:
    rng = np.random.default_rng(0)
    symbols = [f"S{i}" for i in range(20)]
    weights = pd.Series(rng.normal(size=20), index=symbols)
    returns = pd.Series(rng.normal(scale=0.02, size=20), index=symbols)
    exposures = pd.DataFrame(
        rng.normal(size=(20, 5)),
        index=symbols,
        columns=[f"z_{i}" for i in range(5)],
    )
    exposures.iloc[0] = 0.0
    result = attribute_return(weights, returns, exposures)
    book = float((weights * returns).sum())
    assert abs(float(result["contribution"].sum()) - book) < 1e-10
    unattributed = result.loc[result["signal"] == "unattributed", "contribution"].iloc[0]
    assert unattributed == pytest.approx(float(weights.iloc[0] * returns.iloc[0]))


def test_one_signal_takes_the_whole_contribution() -> None:
    weights = pd.Series({"AAA": 0.4})
    returns = pd.Series({"AAA": 0.05})
    exposures = pd.DataFrame({"mom": [1.0], "rev": [0.0]}, index=["AAA"])
    result = attribute_return(weights, returns, exposures).set_index("signal")["contribution"]
    assert result["mom"] == pytest.approx(0.02)
    assert result["rev"] == pytest.approx(0.0)
    assert result["unattributed"] == pytest.approx(0.0)


def test_an_empty_book_attributes_nothing() -> None:
    result = attribute_return(
        pd.Series(dtype=float),
        pd.Series(dtype=float),
        pd.DataFrame(columns=["z_ret_1d", "z_mom_12_1"]),
    )
    assert float(result["contribution"].sum()) == 0.0
    assert "unattributed" in set(result["signal"])


def test_coefficients_are_replaced_per_model(tmp_path) -> None:
    ridge = pd.DataFrame(
        {
            "fold": [0, 0],
            "feature": ["z_ret_1d", "z_mom_12_1"],
            "weight": [0.2, -0.1],
        }
    )
    other = pd.DataFrame({"fold": [1], "feature": ["z_mom_12_1"], "weight": [1.0]})
    with MarketStore(tmp_path / "coef.duckdb") as store:
        assert write_coefficients(store, ridge, model_name="ridge") == 2
        assert write_coefficients(store, other, model_name="momentum") == 1
        again = ridge.copy()
        again["weight"] = [0.3, -0.2]
        write_coefficients(store, again, model_name="ridge")
        stored = read_coefficients(store)
    ridge_rows = stored.loc[stored["model"] == "ridge"]
    assert set(ridge_rows["weight"]) == {0.3, -0.2}
    assert stored.loc[stored["model"] == "momentum", "feature"].tolist() == ["z_mom_12_1"]


def test_attribute_command_uses_the_next_bar_and_states_the_identity(tmp_path) -> None:
    db = tmp_path / "attr.duckdb"
    report = tmp_path / "report"
    symbols = ["AAPL", "MSFT", "NVDA", "JPM"]
    assert (
        main(["ingest", "--synthetic", "--periods", "70", "--symbols", *symbols, "--db", str(db)])
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
        ["attribute", "--db", str(db), "--max-weight", "0.5", "--report", str(report)]
    )
    assert code == 0
    text = (report / "attribution.md").read_text(encoding="utf-8")
    assert "close-to-close" in text
    assert "equity-curve" in text
    frame = pd.read_csv(report / "attribution.csv")
    with MarketStore(db) as store:
        stored = read_coefficients(store, "ridge")
        result = attribute_store(store, model="ridge", max_weight=0.5)
    assert not stored.empty
    assert abs(float(frame["contribution"].sum()) - result.portfolio_return) < 1e-10
    assert "fold" in result.note
    assert main(["attribute", "--db", str(tmp_path / "empty.duckdb")]) == 1


def test_missing_coefficients_fall_back_to_unit_betas(tmp_path) -> None:
    db = tmp_path / "fallback.duckdb"
    symbols = ["AAPL", "MSFT", "NVDA", "JPM"]
    assert (
        main(["ingest", "--synthetic", "--periods", "40", "--symbols", *symbols, "--db", str(db)])
        == 0
    )
    assert main(["features", "--db", str(db)]) == 0
    with MarketStore(db) as store:
        bars = store.read_bars()
        dates = sorted(pd.to_datetime(bars["date"]).unique())
        signal = dates[10]
        rows = []
        for offset, symbol in enumerate(symbols):
            rows.append(
                {
                    "symbol": symbol,
                    "date": pd.Timestamp(signal).date(),
                    "prediction": float(offset - 1.5),
                    "label": 0.0,
                    "fold": 0,
                }
            )
        write_predictions(store, pd.DataFrame(rows), model_name="ridge", label_column="label")
        result = attribute_store(store, model="ridge", max_weight=0.5)
    assert "beta of 1" in result.note
    assert abs(float(result.by_signal["contribution"].sum()) - result.portfolio_return) < 1e-10
    path = write_attribution_report(result, Path(tmp_path) / "note")
    assert "equity-curve" in path.read_text(encoding="utf-8")
