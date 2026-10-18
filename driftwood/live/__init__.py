"""Paper-trading intents. Nothing in this package talks to a live broker."""

from driftwood.live.broker import PAPER_URL, assert_paper_url, make_paper_client
from driftwood.live.orders import OrderIntent, client_order_id, plan_orders
from driftwood.live.run import (
    MarketOpenError,
    PaperPlan,
    broker_equity_and_positions,
    plan_from_store,
    submit_plan,
)

__all__ = [
    "PAPER_URL",
    "MarketOpenError",
    "OrderIntent",
    "PaperPlan",
    "assert_paper_url",
    "broker_equity_and_positions",
    "client_order_id",
    "make_paper_client",
    "plan_from_store",
    "plan_orders",
    "submit_plan",
]
