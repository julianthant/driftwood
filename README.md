# driftwood

A quantitative research and paper-trading pipeline for US stocks, built as a
portfolio project for quant and algorithm developer internships.

The intended research loop is: collect market data, build features, train return
prediction models, validate them on future periods, simulate execution costs,
paper trade, and explain daily performance by signal.

**Current status:** minimal project scaffold. No market-data ingestion, models,
simulator, or trading functionality is implemented yet.

## Build and learn, one milestone at a time

We will discuss the design before implementation, explain finance concepts as
they appear, and stop after each milestone for review.

1. Daily market-data ingestion, DuckDB storage, and an exploratory notebook.
2. Features and tests that prevent using future information.
3. A baseline model, walk-forward validation, and information coefficient (IC)
   reports measuring the correlation between predictions and realized returns.
4. A C++17 execution simulator, Python bindings, costs, and fill tests.
5. Backtesting and immediate, time-weighted (TWAP), and volume-weighted (VWAP)
   execution comparisons.
6. Daily paper trading, per-signal performance attribution, and a dashboard.
7. A research agent with human approval, daily reports, and documented results.

Initial research scope: roughly 50 liquid S&P 500 stocks, daily bars, and about
10 years of history. Trading integrations will use Alpaca's paper endpoint only;
credentials will come from environment variables.

## Local setup

Python 3.11 is the initial development version. Target platforms are Windows,
Linux, and macOS.

From the repository root in Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e .
```

On Linux or macOS:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -e .
```

If `.venv` already exists, skip its creation. An editable install makes imports
use the source files in this folder, so edits are available without reinstalling.
There are no runtime dependencies yet; we will add them as each milestone needs
them. CMake, pybind11, and Windows MSVC setup will be introduced with milestone 4.

## Planned components

Python modules will live under `driftwood/`: `data`, `features`, `models`,
`portfolio`, `backtest`, `live`, `attribution`, and `agent`. The execution simulator
will live in `cpp/sim/`, with a Streamlit app in `dashboard/`, exploratory work in
`notebooks/`, and automated checks in `tests/`. These components will be created
as we reach their milestones.

The planned stack is pandas, NumPy, DuckDB, yfinance, statsmodels, scikit-learn,
LightGBM, C++17/CMake/pybind11, Alpaca paper trading, Streamlit, LangGraph,
LangChain, langchain-xai (ChatXAI with a configurable model), and pytest.

Local datasets belong in the root `data/` folder; generated outputs belong in
`artifacts/`, `reports/`, or `logs/`. Git ignores those folders, local databases,
virtual environments, credentials, and build products. The `driftwood/data/`
source module will remain tracked.
