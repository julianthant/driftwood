"""The research universe: about 50 liquid US large caps.

The list is a current snapshot of names that are easy to trade and easy to
explain. It is not a point-in-time S&P 500 membership list. See the design
notes for the survivorship-bias consequence.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class Listing:
    symbol: str
    name: str
    sector: str


# GICS-style sectors. Names are the common English names, not legal entities.
UNIVERSE: tuple[Listing, ...] = (
    Listing("AAPL", "Apple", "Information Technology"),
    Listing("MSFT", "Microsoft", "Information Technology"),
    Listing("NVDA", "NVIDIA", "Information Technology"),
    Listing("AVGO", "Broadcom", "Information Technology"),
    Listing("ORCL", "Oracle", "Information Technology"),
    Listing("CRM", "Salesforce", "Information Technology"),
    Listing("AMD", "Advanced Micro Devices", "Information Technology"),
    Listing("CSCO", "Cisco", "Information Technology"),
    Listing("ADBE", "Adobe", "Information Technology"),
    Listing("QCOM", "Qualcomm", "Information Technology"),
    Listing("IBM", "IBM", "Information Technology"),
    Listing("INTU", "Intuit", "Information Technology"),
    Listing("TXN", "Texas Instruments", "Information Technology"),
    Listing("AMAT", "Applied Materials", "Information Technology"),
    Listing("GOOGL", "Alphabet Class A", "Communication Services"),
    Listing("META", "Meta Platforms", "Communication Services"),
    Listing("DIS", "Walt Disney", "Communication Services"),
    Listing("CMCSA", "Comcast", "Communication Services"),
    Listing("NFLX", "Netflix", "Communication Services"),
    Listing("UNH", "UnitedHealth Group", "Health Care"),
    Listing("LLY", "Eli Lilly", "Health Care"),
    Listing("JNJ", "Johnson & Johnson", "Health Care"),
    Listing("ABBV", "AbbVie", "Health Care"),
    Listing("MRK", "Merck", "Health Care"),
    Listing("TMO", "Thermo Fisher Scientific", "Health Care"),
    Listing("ABT", "Abbott Laboratories", "Health Care"),
    Listing("AMGN", "Amgen", "Health Care"),
    Listing("JPM", "JPMorgan Chase", "Financials"),
    Listing("V", "Visa", "Financials"),
    Listing("MA", "Mastercard", "Financials"),
    Listing("BAC", "Bank of America", "Financials"),
    Listing("WFC", "Wells Fargo", "Financials"),
    Listing("GS", "Goldman Sachs", "Financials"),
    Listing("BLK", "BlackRock", "Financials"),
    Listing("AXP", "American Express", "Financials"),
    Listing("AMZN", "Amazon", "Consumer Discretionary"),
    Listing("TSLA", "Tesla", "Consumer Discretionary"),
    Listing("HD", "Home Depot", "Consumer Discretionary"),
    Listing("MCD", "McDonald's", "Consumer Discretionary"),
    Listing("NKE", "Nike", "Consumer Discretionary"),
    Listing("HON", "Honeywell", "Industrials"),
    Listing("CAT", "Caterpillar", "Industrials"),
    Listing("WMT", "Walmart", "Consumer Staples"),
    Listing("COST", "Costco", "Consumer Staples"),
    Listing("PG", "Procter & Gamble", "Consumer Staples"),
    Listing("KO", "Coca-Cola", "Consumer Staples"),
    Listing("PEP", "PepsiCo", "Consumer Staples"),
    Listing("XOM", "Exxon Mobil", "Energy"),
    Listing("CVX", "Chevron", "Energy"),
    Listing("NEE", "NextEra Energy", "Utilities"),
)


def symbols() -> list[str]:
    """Ticker symbols in stable universe order."""

    return [item.symbol for item in UNIVERSE]


def listing_map() -> dict[str, Listing]:
    return {item.symbol: item for item in UNIVERSE}


def describe_symbols(requested: list[str]) -> pd.DataFrame:
    """Attach name and sector to the symbols a run actually requested.

    Symbols outside the static universe are stored as ``Unknown`` so an ad-hoc
    download is still recorded. They are not silently dropped.
    """

    known = listing_map()
    rows: list[dict[str, str]] = []
    for symbol in requested:
        item = known.get(symbol.upper())
        if item is None:
            rows.append({"symbol": symbol.upper(), "name": "Unknown", "sector": "Unknown"})
        else:
            rows.append({"symbol": item.symbol, "name": item.name, "sector": item.sector})
    return pd.DataFrame(rows, columns=["symbol", "name", "sector"])


def universe_frame() -> pd.DataFrame:
    return describe_symbols(symbols())
