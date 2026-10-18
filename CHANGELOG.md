# Changelog

All notable changes to Driftwood are recorded here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and versions follow
the milestone that introduced them.

## [Unreleased]

## [0.3.0] - 2026-10-18

### Added

- A walk-forward ridge model on the cross-sectional z-scores, a 12-minus-1
  momentum baseline that does not fit a coefficient, and a small
  deterministic LightGBM check.
- A purge between train and test of at least the label horizon, so a
  forward return cannot reach into the window being scored.
- An information-coefficient report: daily Spearman IC as the headline,
  Pearson beside it, a pooled regression labeled as a different statistic,
  and a Newey-West t-stat for overlapping labels.
- `driftwood train`, which writes out-of-sample predictions to the study
  database. Retraining one model leaves the other models' rows in place.

### Fixed

- macOS CI installs Homebrew `libomp` so the LightGBM wheel can load.

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
