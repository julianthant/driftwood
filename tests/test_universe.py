from driftwood.data.universe import UNIVERSE, describe_symbols, symbols


def test_universe_has_fifty_unique_liquid_names() -> None:
    tickers = symbols()
    assert len(UNIVERSE) == 50
    assert len(set(tickers)) == 50
    assert "AAPL" in tickers
    assert "BRK-B" not in tickers
    assert all(item.sector and item.name for item in UNIVERSE)


def test_unknown_symbol_is_kept_and_labeled() -> None:
    frame = describe_symbols(["aapl", "ZZZZ"])
    assert frame.loc[frame["symbol"] == "AAPL", "sector"].item() == "Information Technology"
    assert frame.loc[frame["symbol"] == "ZZZZ", "name"].item() == "Unknown"
