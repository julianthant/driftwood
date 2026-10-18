"""Turn one day's scores into share deltas for the next session.

Sizing matches the backtest: weights from the signal cross-section, shares
from the signal close, and an order only for the difference from the shares
already held. The order is not filled here.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np
import pandas as pd

from driftwood.portfolio.construct import construct_weights


@dataclass(frozen=True)
class OrderIntent:
    symbol: str
    side: str
    quantity: float
    signal_date: pd.Timestamp
    signal_price: float
    client_order_id: str


def client_order_id(signal_date: object, symbol: str, side: str) -> str:
    """Broker client id, at most 48 characters, shaped ``dw-YYYYMMDD-symbol-side``."""

    stamp = pd.Timestamp(signal_date).strftime("%Y%m%d")
    return f"dw-{stamp}-{symbol}-{side}"[:48]


def plan_orders(
    scores: pd.Series,
    prices: pd.Series,
    *,
    equity: float,
    signal_date: object,
    positions: Mapping[str, float] | None = None,
    gross: float = 1.0,
    max_weight: float = 0.1,
    min_names: int = 2,
    min_trade_dollars: float = 0.0,
) -> list[OrderIntent]:
    """Share deltas from a flat or existing book.

    ``prices`` are the adjusted closes on the signal date. A name without a
    positive price is skipped. Fewer than ``min_names`` scores produces no
    orders, which leaves the current book in place. Equity at or below zero
    targets a flat book.
    """

    held = {str(symbol): float(qty) for symbol, qty in (positions or {}).items()}
    clean = _as_series(scores).dropna()
    if int(clean.shape[0]) < min_names:
        return []
    if equity <= 0:
        weights = pd.Series(0.0, index=clean.index, dtype=float)
    else:
        weights = construct_weights(clean, gross=gross, max_weight=max_weight)
    known_prices = _as_series(prices)
    stamp = pd.Timestamp(signal_date).normalize()
    symbols = set(weights.index) | set(held)
    orders: list[OrderIntent] = []
    for symbol in sorted(symbols):
        price = known_prices.get(symbol, np.nan)
        if not np.isfinite(price) or float(price) <= 0:
            continue
        price = float(price)
        target = float(weights.get(symbol, 0.0)) * float(equity) / price
        delta = target - held.get(symbol, 0.0)
        if abs(delta) * price < min_trade_dollars or delta == 0.0:
            continue
        side = "buy" if delta > 0 else "sell"
        orders.append(
            OrderIntent(
                symbol=symbol,
                side=side,
                quantity=abs(float(delta)),
                signal_date=stamp,
                signal_price=price,
                client_order_id=client_order_id(stamp, symbol, side),
            )
        )
    return orders


def _as_series(values: pd.Series) -> pd.Series:
    return pd.Series(values.to_numpy(dtype=float), index=values.index.astype(str))
