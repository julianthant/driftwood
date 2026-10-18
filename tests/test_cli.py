from driftwood.cli import main
from driftwood.data.store import MarketStore
from driftwood.data.yfinance import DataUnavailable


def test_synthetic_cli_writes_the_default_universe(tmp_path) -> None:
    db = tmp_path / "cli.duckdb"
    code = main(
        [
            "ingest",
            "--synthetic",
            "--periods",
            "5",
            "--seed",
            "9",
            "--db",
            str(db),
        ]
    )
    assert code == 0
    with MarketStore(db) as store:
        assert store.read_bars()["symbol"].nunique() == 50
        assert len(store.read_bars()) == 250


def test_database_path_comes_from_the_environment(tmp_path, monkeypatch) -> None:
    db = tmp_path / "from-env.duckdb"
    monkeypatch.setenv("DRIFTWOOD_DB", str(db))
    code = main(["ingest", "--synthetic", "--periods", "4", "--symbols", "AAPL", "MSFT"])
    assert code == 0
    assert db.exists()


def test_yfinance_command_is_mocked(tmp_path, monkeypatch) -> None:
    db = tmp_path / "cli-yf.duckdb"

    def fake(symbols, start, end):
        from driftwood.data.synthetic import synthetic_bars

        assert symbols == ["NVDA"]
        return synthetic_bars(symbols, periods=6, seed=1)

    monkeypatch.setattr("driftwood.data.ingest.fetch_yfinance_bars", fake)
    code = main(
        [
            "ingest",
            "--start",
            "2016-01-01",
            "--end",
            "2026-01-01",
            "--symbols",
            "NVDA",
            "--db",
            str(db),
        ]
    )
    assert code == 0
    with MarketStore(db) as store:
        assert len(store.read_bars()) == 6


def test_vendor_failure_is_a_nonzero_exit(tmp_path, monkeypatch) -> None:
    def fake(_symbols, _start, _end):
        raise DataUnavailable("no rows")

    monkeypatch.setattr("driftwood.data.ingest.fetch_yfinance_bars", fake)
    code = main(
        [
            "ingest",
            "--start",
            "2020-01-01",
            "--end",
            "2020-02-01",
            "--symbols",
            "AAPL",
            "--db",
            str(tmp_path / "none.duckdb"),
        ]
    )
    assert code == 1


def test_real_download_requires_dates() -> None:
    try:
        main(["ingest", "--symbols", "AAPL"])
    except SystemExit as exc:
        assert exc.code == 2
    else:
        raise AssertionError("expected argparse to reject the command")
