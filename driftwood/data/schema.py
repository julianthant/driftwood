"""DuckDB table definitions for daily bars.

One row is one symbol on one session. ``close`` is the vendor close stored
beside ``adj_close``. Current yfinance, even with auto-adjust off, returns
split-adjusted OHLC, and ``adj_close`` is that series after dividends as
well. The synthetic book is different on purpose: its raw close jumps on a
split and ``adj_close`` does not. Both columns are kept so that difference
stays visible.
"""

from __future__ import annotations

BAR_COLUMNS: tuple[str, ...] = (
    "symbol",
    "date",
    "open",
    "high",
    "low",
    "close",
    "adj_close",
    "volume",
    "source",
    "ingested_at",
)

PRICE_COLUMNS: tuple[str, ...] = ("open", "high", "low", "close", "adj_close", "volume")

DDL: str = """
CREATE TABLE IF NOT EXISTS bars (
    symbol VARCHAR NOT NULL,
    date DATE NOT NULL,
    open DOUBLE NOT NULL,
    high DOUBLE NOT NULL,
    low DOUBLE NOT NULL,
    close DOUBLE NOT NULL,
    adj_close DOUBLE NOT NULL,
    volume BIGINT NOT NULL,
    source VARCHAR NOT NULL,
    ingested_at TIMESTAMP NOT NULL,
    PRIMARY KEY (symbol, date)
);

CREATE TABLE IF NOT EXISTS universe_members (
    symbol VARCHAR PRIMARY KEY,
    name VARCHAR NOT NULL,
    sector VARCHAR NOT NULL
);

CREATE TABLE IF NOT EXISTS ingest_runs (
    run_id VARCHAR PRIMARY KEY,
    started_at TIMESTAMP NOT NULL,
    finished_at TIMESTAMP,
    source VARCHAR NOT NULL,
    symbols_requested INTEGER NOT NULL,
    rows_received INTEGER NOT NULL,
    rows_written INTEGER NOT NULL,
    rows_rejected INTEGER NOT NULL,
    status VARCHAR NOT NULL,
    notes VARCHAR
);
"""
