"""Dataframe helpers for the study view.

Importing this module does not import the dashboard runtime. The page that
draws these frames is not allowed to send an order.
"""

from __future__ import annotations

import pandas as pd


def coverage_table(bars: pd.DataFrame) -> pd.DataFrame:
    """One row per symbol: how many sessions, and the first and last date."""

    columns = ["symbol", "sessions", "first", "last"]
    if bars.empty:
        return pd.DataFrame(columns=columns)
    frame = bars.loc[:, ["symbol", "date"]].copy()
    frame["date"] = pd.to_datetime(frame["date"])
    grouped = frame.groupby("symbol", sort=True)["date"]
    return pd.DataFrame(
        {
            "symbol": grouped.count().index.astype(str),
            "sessions": grouped.count().to_numpy(),
            "first": grouped.min().to_numpy(),
            "last": grouped.max().to_numpy(),
        }
    )


def attribution_table(contributions: pd.DataFrame) -> pd.DataFrame:
    """Signal contributions plus a total row. The total is the book's return."""

    frame = contributions.loc[:, ["signal", "contribution"]].copy()
    total = float(frame["contribution"].sum()) if not frame.empty else 0.0
    extra = pd.DataFrame([{"signal": "total", "contribution": total}])
    return pd.concat([frame, extra], ignore_index=True)


def orders_table(orders: pd.DataFrame) -> pd.DataFrame:
    """Symbol, side, and quantity for a preview. Empty input stays empty."""

    columns = ["symbol", "side", "quantity"]
    if orders.empty:
        return pd.DataFrame(columns=columns)
    return orders.loc[:, columns].reset_index(drop=True)


def backtest_series(daily: pd.DataFrame) -> pd.DataFrame:
    """Add normalized equity, underwater drawdown and accumulated fill costs."""
    frame = daily.sort_values(["algo", "date"]).copy()
    grouped = frame.groupby("algo", sort=False)
    frame["growth"] = frame["equity"] / grouped["equity"].transform("first")
    frame["drawdown"] = frame["equity"] / grouped["equity"].cummax() - 1.0
    frame["cumulative_cost"] = grouped["cost"].cumsum()
    return frame


def ic_series(daily: pd.DataFrame, window: int = 63) -> pd.DataFrame:
    """A full-window rolling mean; undefined ICs stay missing, including in the sum."""
    frame = daily.sort_values("date").copy()
    frame["rolling_ic"] = frame["ic"].rolling(window, min_periods=window).mean()
    frame["cumulative_ic"] = frame["ic"].cumsum()
    return frame


def missing_by_month(dates: pd.DataFrame, symbols: list[str]) -> pd.DataFrame:
    """Missing fraction relative to sessions observed anywhere in this database."""
    frame = dates.loc[:, ["symbol", "date"]].drop_duplicates().copy()
    frame["month"] = pd.to_datetime(frame["date"]).dt.to_period("M").dt.to_timestamp()
    sessions = frame.drop_duplicates("date").groupby("month").size()
    counts = frame.groupby(["symbol", "month"]).size().unstack("month")
    counts = counts.reindex(index=symbols, columns=sessions.index).fillna(0)
    return 1.0 - counts.div(sessions, axis="columns")
