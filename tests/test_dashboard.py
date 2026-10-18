"""The study page is a view. Importing it does not load a broker or a UI runtime."""

import sys
from pathlib import Path

import pandas as pd

from driftwood.data.synthetic import synthetic_bars


def test_views_import_without_the_dashboard_runtime() -> None:
    sys.modules.pop("streamlit", None)
    import dashboard.views as views

    assert "streamlit" not in sys.modules
    bars = synthetic_bars(["AAA", "BBB"], periods=5, seed=1)
    coverage = views.coverage_table(bars)
    assert list(coverage["symbol"]) == ["AAA", "BBB"]
    assert int(coverage["sessions"].sum()) == 10
    contributions = pd.DataFrame(
        {"signal": ["mom", "unattributed"], "contribution": [0.01, 0.0]}
    )
    table = views.attribution_table(contributions)
    assert table.iloc[-1]["signal"] == "total"
    assert table.iloc[-1]["contribution"] == 0.01
    orders = views.orders_table(
        pd.DataFrame({"symbol": ["AAA"], "side": ["buy"], "quantity": [1.0], "extra": [7]})
    )
    assert list(orders.columns) == ["symbol", "side", "quantity"]


def test_the_app_source_does_not_send_orders() -> None:
    source = Path("dashboard/app.py").read_text(encoding="utf-8")
    assert "submit_order" not in source
    assert "alpaca" not in source
    assert "def main" in source
