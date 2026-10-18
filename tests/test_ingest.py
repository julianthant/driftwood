"""Ingestion tests build frames locally or replace the yfinance entry point."""

from driftwood.data.ingest import ingest_frame, ingest_synthetic, ingest_yfinance
from driftwood.data.store import MarketStore
from driftwood.data.synthetic import synthetic_bars
from tests.conftest import bar, bars_frame


def test_synthetic_split_does_not_crash_the_adjusted_close() -> None:
    bars = synthetic_bars(["AAPL", "MSFT"], periods=40, seed=1, split_symbol="AAPL")
    apple = bars.loc[bars["symbol"] == "AAPL"].sort_values("date").reset_index(drop=True)
    microsoft = bars.loc[bars["symbol"] == "MSFT"].reset_index(drop=True)
    mid = len(apple) // 2
    assert (apple.loc[: mid - 1, "adj_close"] == apple.loc[: mid - 1, "close"]).all()
    assert (apple.loc[mid:, "adj_close"] == apple.loc[mid:, "close"] * 2).all()
    raw_return = apple["close"].pct_change().iloc[mid]
    adj_return = apple["adj_close"].pct_change().iloc[mid]
    assert raw_return < -0.4
    assert abs(adj_return) < 0.2
    assert (microsoft["close"] == microsoft["adj_close"]).all()
    assert apple["high"].ge(apple["close"]).all()
    assert apple["low"].le(apple["open"]).all()


def test_synthetic_is_deterministic_and_ingests_twice(tmp_path) -> None:
    first = synthetic_bars(["AAPL"], periods=8, seed=4)
    second = synthetic_bars(["AAPL"], periods=8, seed=4)
    assert first.equals(second)

    with MarketStore(tmp_path / "synth.duckdb") as store:
        result = ingest_synthetic(store, ["AAPL", "MSFT"], periods=12, seed=4, split_symbol="AAPL")
        assert result.ok
        assert result.rows_written == 24
        assert result.symbols_missing == ()
        again = ingest_synthetic(store, ["AAPL", "MSFT"], periods=12, seed=4)
        assert again.rows_written == 24
        assert len(store.read_bars()) == 24
        assert len(store.read_runs()) == 2
        assert set(store.read_universe()["symbol"]) == {"AAPL", "MSFT"}


def test_bad_rows_are_counted_and_missing_symbols_are_reported(tmp_path) -> None:
    frame = bars_frame(
        [
            bar(symbol="AAPL"),
            bar(symbol="MSFT", high=1, low=2, open_=1.5, close=1.4),
        ]
    )
    with MarketStore(tmp_path / "bad.duckdb") as store:
        result = ingest_frame(store, frame, source="fixture", requested=["AAPL", "MSFT", "NVDA"])
        assert result.rows_written == 1
        assert result.rows_rejected == 1
        assert result.symbols_missing == ("MSFT", "NVDA")
        assert list(store.read_bars()["symbol"]) == ["AAPL"]
        assert store.read_runs().iloc[0]["status"] == "ok"


def test_all_rejected_rows_leave_an_empty_run(tmp_path) -> None:
    frame = bars_frame([bar(volume=-1)])
    with MarketStore(tmp_path / "empty-run.duckdb") as store:
        result = ingest_frame(store, frame, source="fixture", requested=["AAPL"])
        assert not result.ok
        assert result.rows_written == 0
        assert store.read_runs().iloc[0]["status"] == "empty"
        assert store.read_bars().empty


def test_yfinance_ingest_uses_the_injected_downloader(tmp_path, monkeypatch) -> None:
    def fake(symbols, start, end):
        assert start == "2020-01-01"
        assert end == "2021-01-01"
        return synthetic_bars(symbols, periods=5, seed=2)

    monkeypatch.setattr("driftwood.data.ingest.fetch_yfinance_bars", fake)
    with MarketStore(tmp_path / "yf.duckdb") as store:
        result = ingest_yfinance(store, ["msft"], "2020-01-01", "2021-01-01")
        assert result.source == "yfinance"
        stored = store.read_bars()
        assert list(stored["symbol"].unique()) == ["MSFT"]
        assert len(stored) == 5
        assert (stored["source"] == "yfinance").all()


def test_failed_upsert_records_an_error_run(tmp_path, monkeypatch) -> None:
    with MarketStore(tmp_path / "err.duckdb") as store:

        def boom(_frame):
            raise RuntimeError("disk")

        monkeypatch.setattr(store, "upsert_bars", boom)
        try:
            ingest_frame(store, bars_frame([bar()]), source="fixture", requested=["AAPL"])
        except RuntimeError:
            pass
        else:
            raise AssertionError("expected RuntimeError")
        assert store.read_runs().iloc[0]["status"] == "error"
