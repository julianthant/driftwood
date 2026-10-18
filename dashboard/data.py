"""Read-only dashboard queries. No UI runtime, credentials or execution path."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from dashboard.views import backtest_series, coverage_table, ic_series, missing_by_month
from driftwood.attribution.study import Attribution, attribute_store
from driftwood.backtest.persist import input_signature
from driftwood.data.store import MarketStore
from driftwood.models.ic import information_coefficient, summarize_ic
from driftwood.models.persist import read_coefficients


@dataclass
class Study:
    coverage: pd.DataFrame
    missing: pd.DataFrame
    sessions: int
    first: pd.Timestamp
    last: pd.Timestamp
    sources: tuple[str, ...]
    models: tuple[str, ...]


@dataclass
class ModelQuality:
    daily: pd.DataFrame
    summary: dict[str, float | int]
    coefficients: pd.DataFrame
    label: str
    hac_lags: int


@dataclass
class Backtest:
    daily: pd.DataFrame
    summary: pd.DataFrame
    config: dict[str, object]
    created_at: pd.Timestamp


def file_revision(path: str) -> tuple[tuple[int, int], ...]:
    """Include the WAL so a committed CLI write invalidates the UI cache."""
    revisions = []
    for candidate in (Path(path), Path(path + ".wal")):
        try:
            stat = candidate.stat()
            revisions.append((stat.st_mtime_ns, stat.st_size))
        except FileNotFoundError:
            revisions.append((0, 0))
    return tuple(revisions)


def _has_table(store: MarketStore, name: str) -> bool:
    return bool(
        store.execute(
            "SELECT COUNT(*) FROM information_schema.tables WHERE table_name = ?", [name]
        ).fetchone()[0]
    )


def load_study(path: str) -> Study:
    with MarketStore(path, read_only=True) as store:
        dates = store.execute("SELECT symbol, date FROM bars ORDER BY date, symbol").df()
        if dates.empty:
            raise RuntimeError("The database has no market bars yet.")
        universe = store.read_universe()["symbol"].astype(str).tolist()
        symbols = sorted(set(universe) | set(dates["symbol"].astype(str)))
        coverage = coverage_table(dates).set_index("symbol").reindex(symbols).reset_index()
        coverage["sessions"] = coverage["sessions"].fillna(0).astype(int)
        coverage["missing"] = dates["date"].nunique() - coverage["sessions"]
        models = ()
        if _has_table(store, "predictions"):
            models = tuple(
                store.execute("SELECT DISTINCT model FROM predictions ORDER BY model")
                .df()["model"]
                .astype(str)
            )
        sources = tuple(
            store.execute("SELECT DISTINCT source FROM bars ORDER BY source")
            .df()["source"]
            .astype(str)
        )
    return Study(
        coverage=coverage,
        missing=missing_by_month(dates, symbols),
        sessions=int(dates["date"].nunique()),
        first=pd.Timestamp(dates["date"].min()),
        last=pd.Timestamp(dates["date"].max()),
        sources=sources,
        models=models,
    )


def load_quality(path: str, model: str) -> ModelQuality:
    with MarketStore(path, read_only=True) as store:
        predictions = store.execute(
            "SELECT symbol, date, prediction, label FROM predictions "
            "WHERE model = ? ORDER BY date, symbol",
            [model],
        ).df()
        label = "unknown (legacy predictions)"
        if _has_table(store, "model_metadata"):
            row = store.execute(
                "SELECT label_column FROM model_metadata WHERE model = ?", [model]
            ).fetchone()
            if row:
                label = str(row[0])
        coefficients = (
            read_coefficients(store, model)
            if _has_table(store, "coefficients")
            else pd.DataFrame(columns=["model", "fold", "feature", "weight"])
        )
    lags = 1 if label == "fwd_ret_1d" else 5
    daily = information_coefficient(predictions, label="label")
    return ModelQuality(
        ic_series(daily), summarize_ic(daily, hac_lags=lags), coefficients, label, lags
    )


def load_backtest(path: str, model: str) -> Backtest:
    with MarketStore(path, read_only=True) as store:
        if not _has_table(store, "backtest_runs"):
            raise RuntimeError("No saved backtest yet. Older CSV reports need a fresh CLI run.")
        run = store.execute(
            "SELECT signature, config, created_at FROM backtest_runs WHERE model = ?", [model]
        ).fetchone()
        if run is None:
            raise RuntimeError("This model has no saved backtest yet.")
        if run[0] != input_signature(store, model):
            raise RuntimeError(
                "Prices, scores or volatility changed after this backtest. Rebuild it."
            )
        daily = store.execute(
            "SELECT * FROM backtest_daily WHERE model = ? ORDER BY algo, date", [model]
        ).df()
        summary = store.execute(
            "SELECT * FROM backtest_summary WHERE model = ? ORDER BY algo", [model]
        ).df()
    return Backtest(backtest_series(daily), summary, json.loads(run[1]), pd.Timestamp(run[2]))


def load_attribution(path: str, model: str) -> Attribution:
    with MarketStore(path, read_only=True) as store:
        if not _has_table(store, "features"):
            raise RuntimeError(
                "Features are missing. Rebuild features before attributing the book."
            )
        return attribute_store(store, model=model)
