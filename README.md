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
| 2 | Features, with tests that future data cannot leak in | Not started |
| 3 | Baseline model, walk-forward validation, information coefficient | Not started |
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
after those adjustments. Use it for returns. Keep the raw close so you can
see what the adjustment did. The synthetic book includes a 2-for-1 split in
one name for exactly this comparison: the raw return on the split day is
about -50%, and the adjusted return is an ordinary day.

**Survivorship bias.** The universe is today's liquid large caps, not the
list of stocks that were in the S&P 500 on each historical date. Names that
shrank, failed, or left the index never appear. Backtests on this list will
look kinder than a real trading book. That limitation is accepted for the
first version and called out in `docs/DESIGN_DECISIONS.md`.

Later milestones add lookahead bias, the information coefficient,
walk-forward validation, TWAP, VWAP, and performance attribution. Those
explanations will land in this section with the code that uses them.

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

An editable install imports the source tree, so edits apply without
reinstalling. Copy `.env.example` to `.env` if you want a local database path
or, later, API keys. Tests do not read that file and do not need keys.

The notebook's charts need matplotlib, which is optional:

```bash
python -m pip install -e ".[notebook]"
```

## Run the data stage

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
| `driftwood/cli.py` | `driftwood ingest` |
| `notebooks/` | Exploratory notebook |
| `tests/` | Automated checks |
| `docs/DESIGN_DECISIONS.md` | Interview study guide and the decision log |
| `data/`, `artifacts/`, `reports/`, `logs/` | Local outputs, not committed |

Planned packages that are not created yet: `features`, `models`, `portfolio`,
`backtest`, `live`, `attribution`, and `agent`. The C++ simulator will live
in `cpp/sim/`, and the Streamlit app in `dashboard/`.

## Credentials

Nothing in this milestone needs a key. `.env.example` lists the variables
later stages will read (`ALPACA_API_KEY`, `ALPACA_SECRET_KEY`, `XAI_API_KEY`,
`XAI_MODEL`). Real values stay in `.env` or the shell. The paper-trading
stage will refuse any host other than `paper-api.alpaca.markets`.
