"""Compact, per-model backtest snapshots for read-only research views."""

from __future__ import annotations

import json
from dataclasses import asdict

from driftwood.backtest.engine import BacktestResult
from driftwood.data.store import MarketStore


def input_signature(store: MarketStore, model: str) -> str:
    """Fingerprint the actual price, score and volatility inputs, including corrections."""
    bars = store.execute(
        "SELECT COUNT(*), "
        "SUM(HASH(symbol, date, open, high, low, close, adj_close, volume)::HUGEINT) "
        "FROM bars"
    ).fetchone()
    scores = store.execute(
        "SELECT COUNT(*), SUM(HASH(symbol, date, prediction)::HUGEINT) "
        "FROM predictions WHERE model = ?",
        [model],
    ).fetchone()
    has_features = store.execute(
        "SELECT COUNT(*) FROM information_schema.columns "
        "WHERE table_name = 'features' AND column_name = 'vol_21'"
    ).fetchone()[0]
    sigma = None
    if has_features:
        sigma = store.execute(
            "SELECT COUNT(*), SUM(HASH(symbol, date, vol_21)::HUGEINT) FROM features"
        ).fetchone()
    return json.dumps([bars, scores, sigma])


def write_backtest(
    store: MarketStore, result: BacktestResult, *, model: str, signature: str
) -> None:
    """Atomically replace one model's schedules; preserve every other model."""
    daily = result.equity.copy()
    daily["date"] = daily["date"].dt.date
    if result.trades.empty:
        daily["cost"] = 0.0
    else:
        costs = result.trades.groupby(["algo", "fill_date"])["cost"].sum().rename("cost")
        costs = costs.reset_index().rename(columns={"fill_date": "date"})
        costs["date"] = costs["date"].dt.date
        daily = daily.merge(costs, on=["algo", "date"], how="left", validate="one_to_one")
        daily["cost"] = daily["cost"].fillna(0.0)
    summary = result.summary.copy()
    daily["model"] = model
    summary["model"] = model
    store.register_frame("_backtest_daily", daily)
    store.register_frame("_backtest_summary", summary)
    store.execute("BEGIN")
    try:
        store.execute(
            "CREATE TABLE IF NOT EXISTS backtest_runs "
            "(model VARCHAR PRIMARY KEY, signature VARCHAR, config VARCHAR, created_at TIMESTAMP)"
        )
        for table in ("backtest_daily", "backtest_summary"):
            store.execute(
                f"CREATE TABLE IF NOT EXISTS {table} AS SELECT * FROM _{table} WHERE FALSE"
            )
            store.execute(f"DELETE FROM {table} WHERE model = ?", [model])
            store.execute(f"INSERT INTO {table} SELECT * FROM _{table}")
        store.execute("DELETE FROM backtest_runs WHERE model = ?", [model])
        store.execute(
            "INSERT INTO backtest_runs VALUES (?, ?, ?, CURRENT_TIMESTAMP)",
            [model, signature, json.dumps(asdict(result.config), sort_keys=True)],
        )
        store.execute("COMMIT")
    except Exception:
        store.execute("ROLLBACK")
        raise
    finally:
        store.unregister_frame("_backtest_daily")
        store.unregister_frame("_backtest_summary")
