from datetime import date

import pandas as pd

from driftwood.data.store import MarketStore
from driftwood.data.universe import describe_symbols
from tests.conftest import bar, bars_frame


def _ready(rows: list[dict[str, object]]) -> pd.DataFrame:
    frame = bars_frame(rows)
    frame["source"] = "test"
    frame["ingested_at"] = pd.Timestamp("2024-01-03 15:00:00")
    return frame


def test_upsert_replaces_the_same_session_and_filters_reads(tmp_path) -> None:
    path = tmp_path / "market.duckdb"
    with MarketStore(path) as store:
        store.upsert_bars(_ready([bar(close=101.0), bar(symbol="MSFT", close=50, high=51, low=49)]))
        store.upsert_bars(_ready([bar(close=111.0, high=112.0)]))
        assert store.latest_date() == date(2024, 1, 2)

        again = MarketStore(path)
        try:
            bars = again.read_bars()
            assert len(bars) == 2
            apple = bars.loc[bars["symbol"] == "AAPL", "close"].item()
            assert apple == 111.0

            window = again.read_bars(symbols=["MSFT"], start="2024-01-02", end="2024-01-02")
            assert list(window["symbol"]) == ["MSFT"]

            # The symbol is a bound parameter, including one that looks like SQL.
            weird = again.read_bars(symbols=["AAPL' OR 1=1 --"])
            assert weird.empty
            assert again.coverage().set_index("symbol").loc["AAPL", "rows"] == 1
        finally:
            again.close()


def test_empty_store_has_schema_and_universe_round_trip(tmp_path) -> None:
    with MarketStore(tmp_path / "empty.duckdb") as store:
        bars = store.read_bars()
        assert list(bars.columns)[:4] == ["symbol", "date", "open", "high"]
        assert bars.empty
        assert store.latest_date() is None
        store.replace_universe(describe_symbols(["AAPL", "ZZZZ"]))
        stored = store.read_universe()
        assert set(stored["symbol"]) == {"AAPL", "ZZZZ"}
        store.replace_universe(describe_symbols(["MSFT"]))
        assert list(store.read_universe()["symbol"]) == ["MSFT"]


def test_memory_database_does_not_touch_the_workdir() -> None:
    with MarketStore(":memory:") as store:
        assert store.upsert_bars(_ready([bar()])) == 1
        assert len(store.read_bars()) == 1
