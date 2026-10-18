# Changelog

All notable changes to Driftwood are recorded here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and versions follow
the milestone that introduced them.

## [Unreleased]

## [0.2.0] - 2026-10-18

### Added

- Close-of-day features (trailing returns, 12-minus-1 momentum, volatility,
  volume shock, distance from a moving average, range, log volume) and
  same-day cross-sectional z-scores.
- Forward one-day and five-day labels, stored in a separate table from the
  features.
- A truncation test that fails if a feature changes when the future is
  removed, including negative cases for a centered window and a full-sample
  scaler.
- `driftwood features`, which rebuilds both tables from the bars already stored.

## [0.1.0] - 2026-10-18

### Added

- Daily bar ingestion from yfinance into a local DuckDB file, with validation and upserts.
- A fixed research universe of 50 liquid US stocks, plus a deterministic synthetic book for offline use.
- An exploratory notebook, a `driftwood ingest` command, and design notes for the data stage.
- Continuous integration that lints and runs the test suite on Linux, Windows, and macOS.

### Changed

- The data notes now match the current yfinance feed. With auto-adjust off,
  OHLC is already split-adjusted and `adj_close` also reflects dividends.
  The synthetic book is still the example where a raw split looks like a crash.
