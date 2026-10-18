# driftwood

A quantitative research and paper-trading pipeline for US stocks, built as a
portfolio project for quant and algorithm developer internships.

The research loop is: collect market data, build features, train return
prediction models, validate them on later periods, simulate execution costs,
paper trade, and explain each day's result by signal.

## Status

| Milestone | What it adds | State |
| --- | --- | --- |
| 1 | Daily bars from yfinance, DuckDB storage, exploratory notebook | Done |
| 2 | Features, with tests that future data cannot leak in | Done |
| 3 | Baseline model, walk-forward validation, information coefficient | Done |
| 4 | C++17 execution simulator, costs, and fill tests | Not started |
| 5 | Backtest and immediate vs TWAP vs VWAP | Not started |
| 6 | Alpaca paper trading, signal attribution, Streamlit dashboard | Not started |
| 7 | Research agent with a human approval gate and a daily report | Not started |

Initial scope: 50 liquid US stocks, daily bars, about 10 years of history.
Trading, when it arrives, uses Alpaca's paper endpoint only. Credentials come
from environment variables.

## Concepts used so far

**OHLCV.** Each row is one stock on one session: the first trade (open), the
highest and lowest prints (high, low), the last trade (close), and the number
of shares traded (volume). Driftwood stores one row per symbol per day.

**Adjusted close.** A split or a dividend changes the raw close without
changing the economic value of a holding. `adj_close` is the vendor's close
after those adjustments. Use it for returns. Keep the other close so you can
see what the adjustment did. Current yfinance, with auto-adjust off, already
returns split-adjusted open, high, low, and close; `adj_close` then differs
by dividends, not by the split. The synthetic book includes a 2-for-1 split
in one name so the raw crash is still visible: the raw return on the split
day is about -50%, and the adjusted return is an ordinary day.

**Survivorship bias.** The universe is today's liquid large caps, not the
list of stocks that were in the S&P 500 on each historical date. Names that
shrank, failed, or left the index never appear. Backtests on this list will
look kinder than a real trading book. That limitation is accepted for the
first version and called out in `docs/DESIGN_DECISIONS.md`.

**Lookahead bias.** A feature dated Monday that uses Tuesday's close, in a
backtest that pretends the Monday trade was possible. Driftwood stamps
features at the close. The row for day `t` may use that day's bar and earlier
bars. It may not use day `t+1`. The same day's high and low are allowed,
because the close has already printed them. They would not be allowed for an
order placed at the open. The forward return is a label, computed in a
separate function and stored in a separate table. The test rebuilds the
features on history cut off at a date and requires the past values to match.
A centered moving average fails that test. So does a z-score fit on the
whole sample.

**Walk-forward validation.** A random split of days would let a model train
on Tuesday and "predict" Monday. Walk-forward does not. The calendar is cut
into successive test windows (63 sessions by default, about one quarter).
Before each window, a purge is dropped: the embargo you asked for, and at
least as many sessions as the label looks ahead. A one-day label on the
session before the test uses the first test session's close, so that row
cannot be in the fit. The default training window expands. Once a test
window has happened, the next fold may learn from it, because those returns
are no longer in the future. A rolling window is available when older
regimes should fall out of the fit. In-sample fit quality is one number in
the report. Those rows are not stored as forecasts.

**Information coefficient.** On one day, the IC is the correlation across
names between the forecast and the realized forward return. Spearman (rank)
correlation is the headline: did we order the book correctly? The mean IC
averages that daily number. Positive means the ranking was right more often
than not. It is not a return, and it is not a Sharpe ratio. A pooled
regression of every return on every forecast is a different number (a hedge
ratio) and the report says so. The plain t-stat treats days as independent.
When the label overlaps itself, as a five-day return does, that t-stat is
too flattering. Quote the Newey-West t-stat in the report instead.

TWAP, VWAP, and performance attribution are later milestones. Their
explanations will land here with the code that uses them.

## Local setup

Python 3.11 or newer. The project is meant to run on Windows, Linux, and macOS.

From the repository root in Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e ".[dev]"
```

On Linux or macOS:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev]"
```

On macOS, the LightGBM wheel links OpenMP and does not ship it. Install
that library before the package, or `import lightgbm` fails:

```bash
brew install libomp
```

An editable install imports the source tree, so edits apply without
reinstalling. Copy `.env.example` to `.env` if you want a local database path
or, later, API keys. Tests do not read that file and do not need keys.

The notebook's charts need matplotlib, which is optional:

```bash
python -m pip install -e ".[notebook]"
```

## Run a study

Synthetic data, no network, useful for a first look:

```bash
driftwood ingest --synthetic --periods 504 --split-symbol AAPL --db data/driftwood.duckdb
```

Real daily history. yfinance treats `--end` as exclusive, so the end date you
pass is the first day you do not want:

```bash
driftwood ingest --start 2016-01-01 --end 2026-01-01 --db data/driftwood.duckdb
```

Omit `--symbols` to use the built-in 50 names. Running the command again
replaces bars for the same symbol and date and appends a row to the ingest
log. The database file lives under `data/` and is gitignored.

Build features and labels from the bars already in the file. This does not
download anything. Features for day `t` use information known at that close.
Labels are the next-day and five-day adjusted returns, written to their own
table:

```bash
driftwood features --db data/driftwood.duckdb
```

`mom_12_1` (a year of momentum, skipping the most recent month) stays empty
until a name has about a year of bars. A short synthetic book will say so in
the non-null fraction the command prints.

Fit the walk-forward model on the stored features. The default is ridge on
the z-scores, an expanding window of at least 252 sessions, a 63-session
test, and a 5-session embargo. Predictions are written to the `predictions`
table. `--report` writes the IC write-up next to the CSV tables. `reports/`
is gitignored:

```bash
driftwood train --db data/driftwood.duckdb --report reports/ic
```

`--model momentum` predicts `z_mom_12_1` and does not fit a coefficient. It
needs about a year of bars or it refuses, rather than training on an empty
column. `--model lgbm` is a small deterministic gradient-boosted tree, a
check on the linear fit, not the model to quote. `--label fwd_ret_5d` sets
the purge to at least five sessions. `--rolling` keeps a fixed training
length (`--train-days`, default 756) instead of expanding.

A short synthetic file does not have 252 sessions. Shrink the windows:

```bash
driftwood train --db data/driftwood.duckdb \
  --min-train-days 60 --test-days 20 --embargo 5 --report reports/ic
```

Training again with the same `--model` replaces that model's rows. A
different model is kept beside it, so ridge and momentum can be compared.
The label is stored next to the forecast so the report can be rebuilt. A
later backtest has to trade the `prediction` column.

Then open the notebook from the repository root:

```bash
jupyter lab notebooks/01_explore_market_data.ipynb
```

If `data/driftwood.duckdb` is missing, the notebook builds a two-year
synthetic book so it can still be read top to bottom.

## Tests

```bash
python -m ruff check .
python -m pytest -q
```

The suite does not call yfinance, Alpaca, or a language model. Downloads are
replaced with a fake, and the fake is checked to make sure the real client
would have been asked for unadjusted prices.

## Layout

| Path | Role |
| --- | --- |
| `driftwood/data/` | Universe, validation, DuckDB store, yfinance and synthetic ingestion |
| `driftwood/features/` | Close-of-day features, forward labels, and the truncation check |
| `driftwood/models/` | Walk-forward models, the information coefficient, and the IC report |
| `driftwood/cli.py` | `driftwood ingest`, `driftwood features`, `driftwood train` |
| `notebooks/` | Exploratory notebook |
| `tests/` | Automated checks |
| `docs/DESIGN_DECISIONS.md` | Interview study guide and the decision log |
| `data/`, `artifacts/`, `reports/`, `logs/` | Local outputs, not committed |

Planned packages that are not created yet: `portfolio`, `backtest`, `live`,
`attribution`, and `agent`. The C++ simulator will live in `cpp/sim/`, and
the Streamlit app in `dashboard/`.

## Credentials

Nothing in this milestone needs a key. `.env.example` lists the variables
later stages will read (`ALPACA_API_KEY`, `ALPACA_SECRET_KEY`, `XAI_API_KEY`,
`XAI_MODEL`). Real values stay in `.env` or the shell. The paper-trading
stage will refuse any host other than `paper-api.alpaca.markets`.
