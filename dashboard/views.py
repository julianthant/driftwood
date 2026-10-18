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
