"""Plan a paper session from the study file, and submit it only when asked.

The default path reads the local database and prints orders. It does not
construct a broker client. A submit re-reads equity and positions from the
paper account and plans again, so a flat preview is not what gets sent.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import pandas as pd

from driftwood.data.store import MarketStore
from driftwood.live.orders import OrderIntent, plan_orders
from driftwood.models.persist import read_predictions


class MarketOpenError(RuntimeError):
    """A DAY market order would fill the session that is already open."""


@dataclass(frozen=True)
class PaperPlan:
    signal_date: pd.Timestamp
    orders: tuple[OrderIntent, ...]
    equity: float


def plan_from_store(
    store: MarketStore,
    *,
    model: str,
    equity: float,
    positions: Mapping[str, float] | None = None,
    gross: float = 1.0,
    max_weight: float = 0.1,
    min_names: int = 2,
    min_trade_dollars: float = 0.0,
) -> PaperPlan:
    """Latest shared close of bars and predictions, sized into share deltas."""

    bars = store.read_bars()
    if bars.empty:
        raise RuntimeError("no bars in the database; run driftwood ingest first")
    try:
        predictions = read_predictions(store)
    except RuntimeError as exc:
        raise RuntimeError("no predictions in the database; run driftwood train first") from exc
    signal_date, scores, prices = signal_snapshot(bars, predictions, model)
    orders = plan_orders(
        scores,
        prices,
        equity=equity,
        signal_date=signal_date,
        positions=positions,
        gross=gross,
        max_weight=max_weight,
        min_names=min_names,
        min_trade_dollars=min_trade_dollars,
    )
    return PaperPlan(signal_date=signal_date, orders=tuple(orders), equity=float(equity))


def signal_snapshot(
    bars: pd.DataFrame,
    predictions: pd.DataFrame,
    model: str,
) -> tuple[pd.Timestamp, pd.Series, pd.Series]:
    """Scores and adjusted closes on the last date present in both tables."""

    chosen = predictions.loc[predictions["model"] == model].copy()
    if chosen.empty:
        raise RuntimeError(f"no predictions for model {model}")
    chosen["date"] = _normalize(chosen["date"])
    frame = bars.copy()
    frame["date"] = _normalize(frame["date"])
    shared = sorted(set(chosen["date"]).intersection(set(frame["date"])))
    if not shared:
        raise RuntimeError("predictions and bars do not share a date")
    signal_date = pd.Timestamp(shared[-1])
    day_scores = chosen.loc[chosen["date"] == signal_date]
    if day_scores["symbol"].duplicated().any():
        raise ValueError(f"duplicate predictions on {signal_date.date()}")
    scores = pd.Series(
        day_scores["prediction"].to_numpy(dtype=float),
        index=day_scores["symbol"].astype(str),
    )
    day_bars = frame.loc[frame["date"] == signal_date].drop_duplicates("symbol")
    prices = pd.Series(
        day_bars["adj_close"].to_numpy(dtype=float),
        index=day_bars["symbol"].astype(str),
    )
    return signal_date, scores, prices


def broker_equity_and_positions(client: object) -> tuple[float, dict[str, float]]:
    """Signed share counts and account equity from a paper client."""

    account = client.get_account()
    equity = float(account.equity)
    positions: dict[str, float] = {}
    for position in client.get_all_positions():
        quantity = float(position.qty)
        side = str(getattr(position, "side", "")).lower()
        if "short" in side:
            quantity = -abs(quantity)
        positions[str(position.symbol)] = quantity
    return equity, positions


def submit_plan(
    client: object,
    orders: tuple[OrderIntent, ...] | list[OrderIntent],
    *,
    allow_while_open: bool = False,
) -> list[OrderIntent]:
    """Send DAY market orders. Refuse while the paper market is open.

    A signal is known at the close. A DAY order sent while the market is
    open would fill that same session. ``allow_while_open`` is the explicit
    override. Zero quantities are not sent.
    """

    clock = client.get_clock()
    if bool(getattr(clock, "is_open", False)) and not allow_while_open:
        raise MarketOpenError(
            "The paper market is open. A DAY order would fill this session, "
            "and the signal was the prior close. Pass --allow-while-open to send anyway."
        )
    from alpaca.trading.enums import OrderSide, OrderType, TimeInForce
    from alpaca.trading.requests import MarketOrderRequest

    sent: list[OrderIntent] = []
    for order in orders:
        if order.quantity <= 0:
            continue
        side = OrderSide.BUY if order.side == "buy" else OrderSide.SELL
        request = MarketOrderRequest(
            symbol=order.symbol,
            qty=float(order.quantity),
            side=side,
            type=OrderType.MARKET,
            time_in_force=TimeInForce.DAY,
            client_order_id=order.client_order_id,
        )
        client.submit_order(request)
        sent.append(order)
    return sent


def _normalize(values: pd.Series) -> pd.Series:
    dates = pd.to_datetime(values)
    if isinstance(dates.dtype, pd.DatetimeTZDtype):
        dates = dates.dt.tz_convert("UTC").dt.tz_localize(None)
    return dates.dt.normalize()
