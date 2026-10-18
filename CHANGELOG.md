# Changelog

All notable changes to Driftwood are recorded here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and versions follow
the milestone that introduced them.

## [Unreleased]

### Added

- Six read-only dashboard pages: Overview, Backtest, Execution,
  Model / Signals, Attribution, and Data, with native Streamlit controls,
  interactive Plotly charts, a light navy and teal theme, and CLI recovery
  commands for empty or stale studies.
- Per-model DuckDB backtest snapshots with daily equity, returns, costs,
  turnover, summary, configuration, timestamp, and an input signature.
  CLI backtests save these even without `--report`; reruns replace that
  model's schedule set and preserve other models.
- Stored prediction label metadata for horizon-aware IC diagnostics.

### Changed

- Dashboard model selection starts with saved backtests and prefers ridge
  within each availability group. Schedule comparisons use consistent labels,
  colors, and line styles, with separate performance and cost panels.
- Execution costs show USD, basis points of actual traded notional, and
  cumulative differences from Immediate. Zero traded notional leaves cost
  bps missing; large currency KPIs use compact labels.
- Signal labels, coefficient units, and attribution totals are clearer.
  Coverage tables use dates without times; complete coverage replaces the
  heatmap with "No missing symbol-days", and partial heatmaps show affected
  symbols only. The viewer toolbar hides Deploy.
- Dashboard reads use bounded memory caches with a five-minute expiry,
  database and WAL revision keys, and a manual Refresh data control.
- Corrected bars, model scores, or signal-date volatility require a fresh
  backtest before the dashboard displays its saved results.
- Runtime dependencies include Plotly 5.18 or newer and Streamlit 1.51 or
  newer.

## [0.7.0] - 2026-10-18

### Added

- A research agent that proposes `write_report`, `record_note`,
  `retrain`, or `submit_orders`, then stops. `--approve` is the only way
  an action runs. The model cannot set that list.
- A proposal that carries a broker URL or an open-market flag is refused.
  An approved submit uses the Alpaca paper host only.
- A daily note rendered by Driftwood, with fixed sections. The model does
  not write the page. `docs/examples/daily_report.md` is one render of it.
- `driftwood agent`. The chat model is ChatXAI, default `grok-4`, via
  `XAI_API_KEY` and `XAI_MODEL`.

## [0.6.0] - 2026-10-18

### Added

- A paper-only order plan. `driftwood paper` prints next-session share
  deltas and does not construct a broker client. `--submit` sends DAY
  market orders to `https://paper-api.alpaca.markets` and refuses any
  other host before a client exists.
- Submit re-reads paper equity and positions, and it refuses to send
  while that market is open unless `--allow-while-open` is set.
- Fold coefficients stored beside predictions, replaced per model.
- An attribution that splits the target book's close-to-close return
  across signals and sums back to that return. `driftwood attribute`
  writes the note.
- A read-only Streamlit page in `dashboard/`. It does not send orders.

## [0.5.0] - 2026-10-18

### Added

- Dollar-neutral target weights from one day's scores, with a gross target
  and a per-name cap applied once.
- A backtest that sizes on the signal close and fills on the next session
  only, using adjusted prices and the C++ simulator.
- A comparison of immediate, TWAP, and VWAP on the same targets: total
  return, Sharpe with a zero risk-free rate, max drawdown, turnover, and
  cost. `driftwood backtest` writes the note.

### Fixed

- Scaling a bar onto `adj_close` no longer leaves the close a rounding
  error outside the high or the low. The C++ fill was rejecting those
  sessions, which stopped a real-data backtest.

## [0.4.0] - 2026-10-18

### Added

- A C++17 execution simulator, built into `driftwood._sim` with pybind11
  and scikit-build-core. Python calls it and does not reimplement the fill.
- Immediate (market-on-open), TWAP, and VWAP schedules on a deterministic
  path that stays inside the daily bar. Unfilled size carries forward until
  a participation cap, then is left unfilled at the close.
- Half-spread, per-share commission, and a square-root impact. Buys pay
  more than the raw price and sells receive less. A round trip loses the
  spread.

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
- A training row is dropped when that name's forward return lands in the
  test window because the name skipped a session inside the purge. The
  day's label mean is taken after the drop.
- `--label-horizon` cannot be shorter than the label being scored. The IC
  note says the pooled slope t-stat is not a test that the book made money.

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
