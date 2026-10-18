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
| 4 | C++17 execution simulator, costs, and fill tests | Done |
| 5 | Backtest and immediate vs TWAP vs VWAP | Done |
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
after those adjustments. Use it for returns. Keep the raw close so you can
see what the adjustment did. The synthetic book includes a 2-for-1 split in
one name for exactly this comparison: the raw return on the split day is
about -50%, and the adjusted return is an ordinary day.

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

**TWAP and VWAP.** TWAP (time-weighted average price) splits an order into
equal share counts across the session. VWAP (volume-weighted average price)
splits it in proportion to volume. A daily bar does not contain the
intraday tape, so both schedules run on a model. The session is eight bins
by default. Volume is a U-shape, heavier at the open and the close. An
immediate order is a market-on-open: the first bin is priced at the open,
and any shares that do not fit the participation cap walk forward into
later bins.

**Fill and cost.** The raw price is that path, and it stays inside the
day's high and low. A buy pays more than the raw price, and a sell receives
less. The gap is half the spread (one basis point by default) plus a
square-root impact: a coefficient, times a volatility the caller already
knew, times the square root of the fraction of the bin the order took.
Commission is per share and shows up in cash, not in the price. A round
trip of the same schedule loses the spread. The impact coefficient is not
a fitted Almgren-Chriss model. Nothing here is a portfolio backtest yet.

**Next-session backtest.** The score on the close of day `t` is turned into
target weights with that close. The order is filled on day `t+1`. The
return that already happened, from the previous close to today's close, is
marked before the new order exists. It is not a profit of today's score.
Prices in the backtest are adjusted: the raw open, high, and low are scaled
by `adj_close / close`, so a split is not a crash. The volatility handed to
the fill is the one stored on the signal date, not the fill date.

**Dollar-neutral weights.** Each name's score is compared with the average
score that day. Names above the average are long, names below it are short.
The weights aim at a gross exposure (the sum of absolute weights) and no
single name can exceed a cap. The cap is applied once. If it binds, the
book can finish below the gross target, and the dollars long and short can
stop matching. That is preferred to a loop that quietly breaks the cap.

**Sharpe, drawdown, turnover.** The Sharpe ratio here is the mean daily
return divided by the standard deviation of daily returns, times the square
root of 252, with a risk-free rate of zero. It is not a promise about the
future. Max drawdown is the worst peak-to-trough loss on the equity curve,
stored as a negative number. Turnover is the dollars traded divided by
equity. The cost column is the spread, the impact, and the commission. It
is not the market's move. Immediate, TWAP, and VWAP are run on the same
targets so the difference is the schedule, not a different portfolio.

Performance attribution is a later milestone. Its explanation will land
here with the code that uses it.

## Local setup

Python 3.11 or newer, a C++17 compiler, and CMake. The project is meant to
run on Windows, Linux, and macOS. `pip install` builds the execution
simulator (`driftwood._sim`). No separate compile step.

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

Price one order against one bar. The numbers come from the C++ extension.
`sigma` is trailing volatility the caller already knew. Pass yesterday's
number when a backtest exists. Passing today's close-to-close volatility
would include today's return:

```python
from driftwood.execution import execute_order

fill = execute_order(
    open=100, high=110, low=90, close=105, volume=1_000_000,
    quantity=200, side="buy", algo="twap", sigma=0.02,
)
print(fill.filled, fill.fill_vwap, fill.cash)
```

`algo` is `immediate`, `twap`, or `vwap`. Cash is from the trader's side, so
a buy is negative. `fill.unfilled` is what the participation cap could not
take by the close.

Compare the three schedules on the stored predictions. The command trades
the `prediction` column. It does not trade the stored label. Sigma is
`vol_21` from the feature table, from the signal date:

```bash
driftwood backtest --db data/driftwood.duckdb --report reports/backtest
```

The note in `reports/backtest.md` has total return, Sharpe, max drawdown,
average turnover, and cost for each schedule. A short synthetic file needs
a trained model first, with the walk-forward windows shrunk, and
`--max-weight` raised if only a few names are in the file. The default cap
of 0.1 cannot build a gross of 1 out of two names.

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
| `driftwood/execution.py` | Thin call into the C++ fill. No second implementation of the price |
| `driftwood/portfolio/` | Dollar-neutral weights from one day's scores |
| `driftwood/backtest/` | Next-session backtest and the schedule comparison |
| `cpp/sim/` | C++17 path, TWAP/VWAP schedule, participation cap, and costs |
| `driftwood/cli.py` | `ingest`, `features`, `train`, `backtest` |
| `notebooks/` | Exploratory notebook |
| `tests/` | Automated checks |
| `docs/DESIGN_DECISIONS.md` | Interview study guide and the decision log |
| `data/`, `artifacts/`, `reports/`, `logs/` | Local outputs, not committed |

Planned packages that are not created yet: `live`, `attribution`, and
`agent`. The Streamlit app will live in `dashboard/`.

## Credentials

Nothing in this milestone needs a key. `.env.example` lists the variables
later stages will read (`ALPACA_API_KEY`, `ALPACA_SECRET_KEY`, `XAI_API_KEY`,
`XAI_MODEL`). Real values stay in `.env` or the shell. The paper-trading
stage will refuse any host other than `paper-api.alpaca.markets`.
