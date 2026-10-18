"""Run an approved whitelist action. Anything else is refused.

``submit_orders`` ignores a host and an "allow while open" flag inside the
model's arguments. The host check uses the paper constant. The open-market
flag comes from the person who started the command.
"""

from __future__ import annotations

import os
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from driftwood.agent.proposals import WHITELIST
from driftwood.agent.report import render_daily_report
from driftwood.data.store import MarketStore
from driftwood.live.broker import make_paper_client
from driftwood.live.run import (
    MarketOpenError,
    broker_equity_and_positions,
    plan_from_store,
    submit_plan,
)
from driftwood.models.train import train_store
from driftwood.models.walkforward import InsufficientHistory, WalkForwardConfig

Submit = Callable[[dict[str, object]], str]
Retrain = Callable[[str], str]


@dataclass
class Toolset:
    db_path: str
    model_name: str
    report_dir: str | None
    allow_while_open: bool
    gross: float = 1.0
    max_weight: float = 0.1
    submit: Submit | None = None
    retrain: Retrain | None = None


def default_submit(tools: Toolset, args: dict[str, object]) -> str:
    """Send the replanned paper book. Model-supplied flags are not read."""

    del args
    key = os.environ.get("ALPACA_API_KEY", "")
    secret = os.environ.get("ALPACA_SECRET_KEY", "")
    client = make_paper_client(key, secret, url=None)
    equity, positions = broker_equity_and_positions(client)
    with MarketStore(tools.db_path) as store:
        plan = plan_from_store(
            store,
            model=tools.model_name,
            equity=equity,
            positions=positions,
            gross=tools.gross,
            max_weight=tools.max_weight,
        )
    sent = submit_plan(client, plan.orders, allow_while_open=tools.allow_while_open)
    symbols = ", ".join(f"{order.symbol} {order.side} {order.quantity:.4f}" for order in sent)
    return f"submitted {len(sent)} paper orders" + (f": {symbols}" if symbols else "")


def default_retrain(tools: Toolset, model_name: str) -> str:
    with MarketStore(tools.db_path) as store:
        trained = train_store(
            store,
            model=model_name,
            label="fwd_ret_1d",
            config=WalkForwardConfig(),
        )
    return f"retrained {trained.result.model_name}, {trained.rows_written} predictions"


def run_actions(
    *,
    as_of: str,
    context: str,
    proposals: list[dict[str, object]],
    approved_ids: list[str],
    tools: Toolset,
) -> dict[str, object]:
    approved = set(approved_ids)
    notes: list[str] = []
    results: list[dict[str, object]] = []
    by_action = {str(item["action"]): item for item in proposals}

    for action in WHITELIST:
        if action == "write_report" or action not in by_action:
            continue
        if action not in approved:
            results.append({"id": action, "status": "skipped", "detail": "not approved"})
            continue
        results.append(_run_one(action, by_action[action], tools, notes))

    report_path = ""
    if "write_report" in by_action and "write_report" not in approved:
        results.append({"id": "write_report", "status": "skipped", "detail": "not approved"})
    elif "write_report" in approved and "write_report" in by_action:
        detail = "rendered the daily note"
        if tools.report_dir:
            folder = Path(tools.report_dir)
            folder.mkdir(parents=True, exist_ok=True)
            report_path = str(folder / "daily.md")
            detail = f"wrote {report_path}"
        results.append({"id": "write_report", "status": "ran", "detail": detail})
    elif "write_report" in approved:
        results.append(
            {"id": "write_report", "status": "skipped", "detail": "not proposed"}
        )

    for proposal in proposals:
        action = str(proposal["action"])
        if action in WHITELIST:
            continue
        status = "refused" if action in approved else "skipped"
        detail = "not on the whitelist" if action in approved else "not approved"
        results.append({"id": action, "status": status, "detail": detail})

    report = render_daily_report(
        as_of=as_of,
        context=context,
        proposals=proposals,
        approved_ids=list(approved_ids),
        results=results,
        notes=notes,
    )
    if report_path:
        Path(report_path).write_text(report, encoding="utf-8")
    return {
        "results": results,
        "notes": notes,
        "report": report,
        "report_path": report_path,
    }


def _run_one(
    action: str,
    proposal: dict[str, object],
    tools: Toolset,
    notes: list[str],
) -> dict[str, object]:
    args = proposal.get("args")
    payload = dict(args) if isinstance(args, dict) else {}
    try:
        if action == "record_note":
            text = " ".join(str(payload.get("text", proposal.get("reason", ""))).split())
            if not text:
                return {"id": action, "status": "refused", "detail": "empty note"}
            notes.append(text)
            return {"id": action, "status": "ran", "detail": text}
        if action == "retrain":
            requested = str(payload.get("model", tools.model_name))
            if requested not in {"ridge", "momentum", "lgbm"}:
                return {"id": action, "status": "refused", "detail": f"unknown model {requested}"}
            runner = tools.retrain or (lambda name: default_retrain(tools, name))
            return {"id": action, "status": "ran", "detail": runner(requested)}
        if action == "submit_orders":
            _refuse_model_host(payload)
            sender = tools.submit or (lambda given: default_submit(tools, given))
            return {"id": action, "status": "ran", "detail": sender({})}
    except (MarketOpenError, InsufficientHistory, RuntimeError, ValueError) as exc:
        return {"id": action, "status": "refused", "detail": str(exc)}
    return {"id": action, "status": "refused", "detail": "not on the whitelist"}


def _refuse_model_host(args: dict[str, object]) -> None:
    """A URL in the proposal is never a reason to connect somewhere else."""

    if "url" in args or "base_url" in args or "allow_while_open" in args:
        raise ValueError(
            "Refusing submit_orders. The proposal tried to set a host or the "
            "open-market flag. The paper host and that flag are not model inputs."
        )
