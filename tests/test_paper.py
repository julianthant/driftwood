"""Paper orders stay off the live host, and a dry run does not build a client."""

import sys
from types import SimpleNamespace

import pandas as pd
import pytest

from driftwood.backtest import BacktestConfig, run_backtest
from driftwood.cli import main
from driftwood.live.broker import PAPER_URL, make_paper_client
from driftwood.live.orders import OrderIntent, client_order_id, plan_orders
from driftwood.live.run import MarketOpenError, broker_equity_and_positions, submit_plan


def test_only_the_paper_host_builds_a_client(monkeypatch: pytest.MonkeyPatch) -> None:
    recorded: dict[str, object] = {}

    class Fake:
        def __init__(self, **kwargs: object) -> None:
            recorded.update(kwargs)

    monkeypatch.setitem(
        sys.modules,
        "alpaca.trading.client",
        SimpleNamespace(TradingClient=Fake),
    )
    make_paper_client("key", "secret", url=None)
    assert recorded["paper"] is True
    assert recorded["url_override"] == PAPER_URL
    assert recorded["secret_key"] == "secret"
    recorded.clear()
    make_paper_client("key", "secret", url=PAPER_URL)
    assert recorded["url_override"] == PAPER_URL
    assert recorded["paper"] is True


@pytest.mark.parametrize(
    "url",
    [
        "https://api.alpaca.markets",
        "https://paper-api.alpaca.markets.evil.com",
        "http://paper-api.alpaca.markets",
    ],
)
def test_a_live_or_lookalike_host_is_refused_before_the_client_exists(
    monkeypatch: pytest.MonkeyPatch,
    url: str,
) -> None:
    class Fake:
        def __init__(self, **kwargs: object) -> None:
            raise AssertionError("client constructed")

    monkeypatch.setitem(
        sys.modules,
        "alpaca.trading.client",
        SimpleNamespace(TradingClient=Fake),
    )
    with pytest.raises(ValueError, match="paper-api.alpaca.markets"):
        make_paper_client("key", "secret", url=url)


def test_client_order_id_names_the_signal_and_stays_short() -> None:
    identifier = client_order_id("2026-01-02", "AAPL", "buy")
    assert identifier == "dw-20260102-AAPL-buy"
    assert "AAPL" in identifier
    assert identifier.endswith("buy")
    assert len(identifier) <= 48
    assert len(client_order_id("2026-01-02", "A" * 80, "sell")) <= 48


def test_flat_plan_matches_the_next_session_backtest() -> None:
    dates = pd.bdate_range("2018-01-02", periods=2)
    bars = _panel(dates, {"AAA": [100, 100], "BBB": [100, 100]})
    scores = pd.Series({"AAA": 1.0, "BBB": -1.0})
    orders = plan_orders(
        scores,
        pd.Series({"AAA": 100.0, "BBB": 100.0}),
        equity=1000.0,
        signal_date=dates[0],
        gross=1.0,
        max_weight=0.5,
    )
    result = run_backtest(
        bars,
        pd.DataFrame(
            {
                "symbol": ["AAA", "BBB"],
                "date": [dates[0], dates[0]],
                "score": [1.0, -1.0],
            }
        ),
        algos=("immediate",),
        config=BacktestConfig(
            initial_capital=1000.0,
            gross=1.0,
            max_weight=0.5,
            half_spread_bps=0.0,
            participation_limit=1.0,
            n_bins=1,
        ),
    )
    planned = {order.symbol: order for order in orders}
    assert set(planned) == {"AAA", "BBB"}
    for trade in result.trades.itertuples(index=False):
        order = planned[trade.symbol]
        assert trade.side == order.side
        assert trade.filled == pytest.approx(order.quantity)


def test_an_existing_position_and_a_different_equity_change_the_delta() -> None:
    scores = pd.Series({"AAA": 1.0, "BBB": -1.0})
    prices = pd.Series({"AAA": 100.0, "BBB": 100.0})
    flat = {
        order.symbol: order
        for order in plan_orders(
            scores, prices, equity=1000, signal_date="2018-01-02", max_weight=0.5
        )
    }
    held = plan_orders(
        scores,
        prices,
        equity=1000,
        signal_date="2018-01-02",
        positions={"AAA": flat["AAA"].quantity, "BBB": 0.0},
        max_weight=0.5,
    )
    assert "AAA" not in {order.symbol for order in held}
    sold = next(order for order in held if order.symbol == "BBB")
    assert sold.side == "sell"
    assert sold.quantity == pytest.approx(flat["BBB"].quantity)
    doubled = {
        order.symbol: order.quantity
        for order in plan_orders(
            scores, prices, equity=2000, signal_date="2018-01-02", max_weight=0.5
        )
    }
    assert doubled["AAA"] == pytest.approx(flat["AAA"].quantity * 2)


def test_a_small_dollar_delta_is_skipped() -> None:
    scores = pd.Series({"AAA": 1.0, "BBB": -1.0})
    prices = pd.Series({"AAA": 100.0, "BBB": 100.0})
    assert (
        plan_orders(
            scores,
            prices,
            equity=1000,
            signal_date="2018-01-02",
            max_weight=0.5,
            min_trade_dollars=1_000_000,
        )
        == []
    )


def test_submit_refuses_an_open_market_and_sends_a_day_order_when_it_is_closed() -> None:
    order = OrderIntent(
        symbol="AAPL",
        side="buy",
        quantity=1.5,
        signal_date=pd.Timestamp("2026-01-02"),
        signal_price=100.0,
        client_order_id="dw-20260102-AAPL-buy",
    )
    closed = _FakeClient(is_open=False)
    sent = submit_plan(closed, [order])
    assert sent == [order]
    assert len(closed.submitted) == 1
    request = closed.submitted[0]
    assert request.symbol == "AAPL"
    assert request.qty == pytest.approx(1.5)
    assert request.side.value == "buy"
    assert request.time_in_force.value == "day"
    assert request.type.value == "market"
    assert request.client_order_id == order.client_order_id

    opened = _FakeClient(is_open=True)
    with pytest.raises(MarketOpenError):
        submit_plan(opened, [order])
    assert opened.submitted == []
    submit_plan(opened, [order], allow_while_open=True)
    assert len(opened.submitted) == 1

    zero = OrderIntent(
        symbol="MSFT",
        side="sell",
        quantity=0.0,
        signal_date=pd.Timestamp("2026-01-02"),
        signal_price=100.0,
        client_order_id="dw-20260102-MSFT-sell",
    )
    quiet = _FakeClient(is_open=False)
    assert submit_plan(quiet, [zero]) == []
    assert quiet.submitted == []


def test_broker_positions_are_signed() -> None:
    client = _FakeClient(is_open=False)
    client.account = SimpleNamespace(equity="250000")
    client.positions = [
        SimpleNamespace(symbol="AAPL", qty="3", side="long"),
        SimpleNamespace(symbol="MSFT", qty="2", side="short"),
    ]
    equity, positions = broker_equity_and_positions(client)
    assert equity == pytest.approx(250000)
    assert positions == {"AAPL": 3.0, "MSFT": -2.0}


def test_dry_run_does_not_build_a_client(tmp_path, monkeypatch, capsys) -> None:
    def boom(*args: object, **kwargs: object) -> None:
        raise AssertionError("client constructed")

    monkeypatch.setattr("driftwood.cli.make_paper_client", boom)
    db = _trained(tmp_path)
    code = main(["paper", "--db", str(db), "--max-weight", "0.5"])
    captured = capsys.readouterr()
    assert code == 0
    assert "dry-run" in captured.out
    assert "ALPACA" not in captured.out
    assert main(["paper", "--db", str(tmp_path / "empty.duckdb")]) == 1
    bare = tmp_path / "bars.duckdb"
    assert (
        main(
            [
                "ingest",
                "--synthetic",
                "--periods",
                "5",
                "--symbols",
                "AAPL",
                "MSFT",
                "--db",
                str(bare),
            ]
        )
        == 0
    )
    assert main(["paper", "--db", str(bare)]) == 1


def test_submit_replans_from_the_paper_account(tmp_path, monkeypatch, capsys) -> None:
    client = _FakeClient(is_open=False)
    client.account = SimpleNamespace(equity="2000")
    seen: dict[str, object] = {}

    def factory(*args: object, **kwargs: object) -> _FakeClient:
        seen["args"] = args
        seen["kwargs"] = kwargs
        return client

    monkeypatch.setattr("driftwood.cli.make_paper_client", factory)
    monkeypatch.setenv("ALPACA_API_KEY", "key")
    monkeypatch.setenv("ALPACA_SECRET_KEY", "secret")
    db = _trained(tmp_path)
    assert main(["paper", "--db", str(db), "--max-weight", "0.5", "--capital", "1000"]) == 0
    preview = _printed_quantities(capsys.readouterr().out)
    code = main(
        ["paper", "--db", str(db), "--max-weight", "0.5", "--capital", "1000", "--submit"]
    )
    submitted_text = capsys.readouterr().out
    assert code == 0
    assert "submitted" in submitted_text
    assert seen["args"] == ("key", "secret")
    sent = _printed_quantities(submitted_text)
    assert sent
    assert set(sent) == set(preview)
    for symbol, quantity in sent.items():
        assert quantity == pytest.approx(preview[symbol] * 2)
    assert client.submitted
    assert all(request.qty > 0 for request in client.submitted)

    client.clock = SimpleNamespace(is_open=True)
    client.submitted.clear()
    refused = main(["paper", "--db", str(db), "--max-weight", "0.5", "--submit"])
    assert refused == 1
    assert client.submitted == []


class _FakeClient:
    def __init__(self, *, is_open: bool) -> None:
        self.clock = SimpleNamespace(is_open=is_open)
        self.account = SimpleNamespace(equity="1000000")
        self.positions: list[object] = []
        self.submitted: list[object] = []

    def get_clock(self) -> SimpleNamespace:
        return self.clock

    def get_account(self) -> SimpleNamespace:
        return self.account

    def get_all_positions(self) -> list[object]:
        return self.positions

    def submit_order(self, request: object) -> None:
        self.submitted.append(request)


def _printed_quantities(text: str) -> dict[str, float]:
    quantities: dict[str, float] = {}
    for line in text.splitlines():
        parts = line.split()
        if len(parts) == 3 and parts[1] in {"buy", "sell"}:
            quantities[parts[0]] = float(parts[2])
    return quantities


def _trained(tmp_path) -> object:
    db = tmp_path / "paper.duckdb"
    symbols = ["AAPL", "MSFT", "NVDA", "JPM"]
    assert (
        main(
            ["ingest", "--synthetic", "--periods", "70", "--symbols", *symbols, "--db", str(db)]
        )
        == 0
    )
    assert main(["features", "--db", str(db)]) == 0
    assert (
        main(
            [
                "train",
                "--db",
                str(db),
                "--min-train-days",
                "30",
                "--test-days",
                "15",
                "--embargo",
                "2",
            ]
        )
        == 0
    )
    return db


def _panel(dates, paths: dict[str, list[float]]) -> pd.DataFrame:
    rows = []
    for symbol, closes in paths.items():
        for date, close in zip(dates, closes, strict=True):
            rows.append(
                {
                    "symbol": symbol,
                    "date": date,
                    "open": close,
                    "high": close,
                    "low": close,
                    "close": close,
                    "adj_close": close,
                    "volume": 1e9,
                }
            )
    return pd.DataFrame(rows)
