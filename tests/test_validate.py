import pandas as pd

from driftwood.data.validate import validate_bars
from tests.conftest import bar, bars_frame


def test_keeps_a_valid_bar_and_normalizes_the_symbol() -> None:
    clean, reasons = validate_bars(bars_frame([bar(symbol=" aapl ")]))
    assert reasons == {}
    assert clean.loc[0, "symbol"] == "AAPL"
    assert clean.loc[0, "volume"] == 1_000_000
    assert str(clean.loc[0, "date"]) == "2024-01-02"


def test_rejects_bad_prices_volume_and_keeps_the_last_duplicate() -> None:
    frame = bars_frame(
        [
            bar(close=101.0, volume=10),
            bar(close=105.0, high=106.0, volume=11),
            bar(symbol="MSFT", high=90.0, low=95.0, open_=94.0, close=93.0),
            bar(symbol="NVDA", close=float("nan")),
            bar(symbol="XOM", volume=-5),
            bar(symbol="CVX", volume=1.5),
            bar(symbol="", close=10, high=11, low=9),
        ]
    )
    clean, reasons = validate_bars(frame)
    assert list(clean["symbol"]) == ["AAPL"]
    assert clean.loc[0, "close"] == 105.0
    assert reasons["duplicate_in_batch"] == 1
    assert reasons["ohlc_inconsistent"] == 1
    assert reasons["non_finite_price"] == 1
    assert reasons["bad_volume"] == 2
    assert reasons["missing_symbol_or_date"] == 1
    assert sum(reasons.values()) + len(clean) == len(frame)


def test_rejects_non_positive_prices() -> None:
    clean, reasons = validate_bars(bars_frame([bar(low=0.0, open_=1, high=1, close=1)]))
    assert clean.empty
    assert reasons == {"non_positive_price": 1}


def test_duplicate_index_labels_do_not_collapse_rows() -> None:
    frame = bars_frame(
        [
            bar(symbol="AAPL"),
            bar(symbol="MSFT", open_=50, high=51, low=49, close=50),
        ]
    )
    frame.index = [0, 0]
    clean, reasons = validate_bars(frame)
    assert reasons == {}
    assert set(clean["symbol"]) == {"AAPL", "MSFT"}


def test_missing_column_is_an_error() -> None:
    frame = pd.DataFrame([{"symbol": "AAPL", "date": "2024-01-02"}])
    try:
        validate_bars(frame)
    except ValueError as exc:
        assert "missing columns" in str(exc)
    else:
        raise AssertionError("expected ValueError")
