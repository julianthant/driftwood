"""yfinance is never called. These tests only reshape frames that look like it."""

import warnings

import pandas as pd
import pytest

from driftwood.data.yfinance import DataUnavailable, fetch_yfinance_bars, normalize_download


def _values(rows: int, scale: float) -> dict[str, list[float]]:
    return {
        "Open": [10.0 * scale + i for i in range(rows)],
        "High": [12.0 * scale + i for i in range(rows)],
        "Low": [9.0 * scale + i for i in range(rows)],
        "Close": [11.0 * scale + i for i in range(rows)],
        "Adj Close": [10.5 * scale + i for i in range(rows)],
        "Volume": [1_000 + i for i in range(rows)],
    }


def test_fields_then_tickers_layout() -> None:
    index = pd.bdate_range("2024-01-02", periods=2)
    fields = ["Open", "High", "Low", "Close", "Adj Close", "Volume"]
    columns = pd.MultiIndex.from_product([fields, ["AAPL", "MSFT"]])
    frame_data = {}
    for ticker, scale in (("AAPL", 1.0), ("MSFT", 2.0)):
        values = _values(2, scale)
        for field in fields:
            frame_data[(field, ticker)] = values[field]
    raw = pd.DataFrame(frame_data, index=index, columns=columns)
    out = normalize_download(raw, ["AAPL", "MSFT"])
    assert set(out["symbol"]) == {"AAPL", "MSFT"}
    assert len(out) == 4
    apple = out.loc[out["symbol"] == "AAPL"].sort_values("date")
    assert apple["adj_close"].iloc[0] != apple["close"].iloc[0]


def test_tickers_then_fields_layout_and_flat_single_name() -> None:
    index = pd.to_datetime(["2024-01-02"])
    columns = pd.MultiIndex.from_product(
        [["IBM"], ["Open", "High", "Low", "Close", "Adj Close", "Volume"]]
    )
    raw = pd.DataFrame([[10, 11, 9, 10.5, 10.4, 500]], index=index, columns=columns)
    out = normalize_download(raw, ["IBM"])
    assert out.loc[0, "symbol"] == "IBM"
    assert out.loc[0, "volume"] == 500

    flat = pd.DataFrame(
        {"Open": [10.0], "High": [11.0], "Low": [9.0], "Close": [10.5], "Volume": [5]},
        index=index,
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        missing_adj = normalize_download(flat, ["XOM"])
    assert missing_adj.loc[0, "adj_close"] == missing_adj.loc[0, "close"]
    assert any("Adjusted close was missing" in str(item.message) for item in caught)


def test_flat_frame_with_several_symbols_is_rejected() -> None:
    raw = pd.DataFrame({"Open": [1.0], "High": [1.0], "Low": [1.0], "Close": [1.0], "Volume": [1]})
    with pytest.raises(ValueError, match="one symbol"):
        normalize_download(raw, ["AAPL", "MSFT"])


def test_fetch_passes_adjustment_off_and_never_uses_the_network(monkeypatch) -> None:
    captured: dict[str, object] = {}

    def fake_download(**kwargs: object) -> pd.DataFrame:
        captured.update(kwargs)
        index = pd.bdate_range("2024-01-02", periods=1)
        columns = pd.MultiIndex.from_product(
            [["Open", "High", "Low", "Close", "Adj Close", "Volume"], ["AAPL"]]
        )
        return pd.DataFrame(
            [[10, 11, 9, 10.5, 10.4, 1000]],
            index=index,
            columns=columns,
        )

    monkeypatch.setattr("yfinance.download", fake_download)
    frame = fetch_yfinance_bars(["aapl"], "2024-01-01", "2024-02-01")
    assert captured["auto_adjust"] is False
    assert captured["threads"] is False
    assert captured["start"] == "2024-01-01"
    assert frame.loc[0, "symbol"] == "AAPL"


def test_empty_download_raises(monkeypatch) -> None:
    monkeypatch.setattr("yfinance.download", lambda **_kwargs: pd.DataFrame())
    with pytest.raises(DataUnavailable):
        fetch_yfinance_bars(["AAPL"], "2024-01-01", "2024-02-01")
