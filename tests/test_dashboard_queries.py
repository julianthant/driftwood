"""Query and transformation regressions for real dashboard studies."""

import numpy as np
import pandas as pd
import pytest

from dashboard import data, views
from driftwood.data.store import MarketStore


def test_attribution_rows_follow_chart_order_and_include_total():
    frame = pd.DataFrame(
        {"signal": ["positive", "negative", "zero"], "contribution": [0.012, -0.005, 0]}
    )
    table = views.attribution_table(frame)
    assert table["signal"].tolist() == ["negative", "zero", "positive", "total"]
    assert table["contribution"].iloc[-1] == pytest.approx(0.007)


def test_model_order_prefers_saved_runs_then_ridge():
    models = ("momentum", "ridge", "tree")
    assert views.model_order(models, ("ridge",)) == ("ridge", "momentum", "tree")
    assert views.model_order(models, ("tree",)) == ("tree", "ridge", "momentum")
    assert views.model_order(models, ("momentum", "ridge"))[0] == "ridge"
    assert views.model_order(models, ())[0] == "ridge"
    assert views.model_order((), ()) == ()


def test_cost_bps_uses_actual_traded_notional_and_handles_flat_books():
    daily = pd.DataFrame(
        {"algo": ["immediate", "twap", "immediate", "vwap"], "traded_notional": [100, 1000, 300, 0]}
    )
    summary = pd.DataFrame({"algo": ["immediate", "twap", "vwap"], "total_cost": [2, 3, 0]})
    result = views.execution_summary(summary, daily).set_index("algo")
    assert result.loc["immediate", "total_traded_notional"] == 400
    assert result.loc["immediate", "cost_bps"] == 50
    assert result.loc["twap", "cost_bps"] == 30
    assert np.isnan(result.loc["vwap", "cost_bps"])


@pytest.mark.parametrize(
    ("rates", "expected"),
    [
        ([1.0, 1.0 + 1e-10, 1.0 - 1e-10], 1.0),
        ([0.0, 0.0, 0.0], 0.0),
        ([1.0, 1.001, 1.0], None),
        ([1.0, np.nan, 1.0], None),
        ([1.0, np.inf, 1.0], None),
        ([], None),
    ],
)
def test_shared_cost_rate_preserves_real_differences_and_undefined_rates(rates, expected):
    assert views.shared_cost_bps(pd.DataFrame({"cost_bps": rates})) == expected


def test_label_metadata_distinguishes_legacy_predictions(tmp_path):
    # The missing-model case exercises the schema without requiring training.
    path = tmp_path / "metadata.duckdb"
    with MarketStore(path) as store:
        store.execute(
            "CREATE TABLE predictions (symbol VARCHAR, date DATE, prediction DOUBLE, "
            "label DOUBLE, model VARCHAR)"
        )
        store.execute("CREATE TABLE model_metadata (model VARCHAR, label_column VARCHAR)")
        store.execute("INSERT INTO model_metadata VALUES ('ridge', 'fwd_ret_1d')")
    quality = data.load_quality(str(path), "ridge")
    assert quality.label_known
    legacy = data.load_quality(str(path), "legacy")
    assert not legacy.label_known
