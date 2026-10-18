"""Backtest that can trade only the next session.

A score dated on the close of session ``t`` is sized with that close. The
order is sent to the C++ fill against session ``t+1``. The return from
``t-1`` to ``t`` is already in the mark before that order exists, so the
same-day move is not a profit. Volatility passed into the fill is the
sigma stored on session ``t``, never the sigma of the fill date.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from driftwood.backtest.prices import adjusted_ohlc
from driftwood.execution import execute_order
from driftwood.portfolio.construct import construct_weights

_BAR_COLUMNS = ["symbol", "date", "open", "high", "low", "close", "adj_close", "volume"]


@dataclass(frozen=True)
class BacktestConfig:
    initial_capital: float = 1_000_000.0
    gross: float = 1.0
    max_weight: float = 0.1
    min_names: int = 2
    min_trade_dollars: float = 0.0
    half_spread_bps: float = 1.0
    commission_per_share: float = 0.0
    impact_coef: float = 0.0
    participation_limit: float = 0.1
    n_bins: int = 8
    periods_per_year: int = 252

    def __post_init__(self) -> None:
        if self.initial_capital <= 0:
            raise ValueError("initial_capital must be positive")
        if self.min_names < 1:
            raise ValueError("min_names must be positive")
        if self.min_trade_dollars < 0:
            raise ValueError("min_trade_dollars cannot be negative")
        if self.half_spread_bps < 0 or self.commission_per_share < 0 or self.impact_coef < 0:
            raise ValueError("costs cannot be negative")
        if self.periods_per_year < 1:
            raise ValueError("periods_per_year must be positive")


@dataclass
class BacktestResult:
    equity: pd.DataFrame
    trades: pd.DataFrame
    summary: pd.DataFrame
    config: BacktestConfig


def run_backtest(
    bars: pd.DataFrame,
    scores: pd.DataFrame,
    *,
    algos: tuple[str, ...] = ("immediate", "twap", "vwap"),
    config: BacktestConfig | None = None,
    volatility: pd.DataFrame | None = None,
) -> BacktestResult:
    """Run each algorithm on the same targets. Books do not share a fill.

    ``scores`` has ``symbol``, ``date``, and ``score``. ``volatility``, when
    given, has ``symbol``, ``date``, and ``sigma`` known at that close.
    """

    config = config or BacktestConfig()
    if not algos:
        raise ValueError("algos must name at least one schedule")
    panel = _prepare_bars(bars)
    score_map = _score_map(scores)
    sigma_map = _sigma_map(volatility)
    by_date: dict[pd.Timestamp, pd.DataFrame] = {}
    for date, frame in panel.groupby("date", sort=True):
        by_date[pd.Timestamp(date)] = frame.set_index("symbol")
    dates = sorted(by_date)

    equities: list[pd.DataFrame] = []
    trades: list[pd.DataFrame] = []
    for algo in algos:
        equity, trade = _run_one(dates, by_date, score_map, sigma_map, algo, config)
        equities.append(equity)
        trades.append(trade)
    equity_frame = pd.concat(equities, ignore_index=True)
    trade_frame = pd.concat(trades, ignore_index=True) if trades else _empty_trades()
    return BacktestResult(
        equity=equity_frame,
        trades=trade_frame,
        summary=summarize(equity_frame, trade_frame, config.periods_per_year),
        config=config,
    )


def summarize(
    equity: pd.DataFrame,
    trades: pd.DataFrame,
    periods_per_year: int = 252,
) -> pd.DataFrame:
    """One row per algorithm. Sharpe uses a risk-free rate of zero."""

    rows: list[dict[str, object]] = []
    for algo, frame in equity.groupby("algo", sort=False):
        rets = frame["ret"].dropna()
        curve = frame["equity"].astype(float)
        peak = curve.cummax()
        drawdown = curve / peak - 1.0
        mean = float(rets.mean()) if len(rets) else float("nan")
        std = float(rets.std(ddof=1)) if len(rets) > 1 else float("nan")
        if np.isfinite(std) and std > 0:
            sharpe = mean / std * np.sqrt(periods_per_year)
        else:
            sharpe = float("nan")
        algo_trades = trades.loc[trades["algo"] == algo] if not trades.empty else trades
        total_cost = float(algo_trades["cost"].sum()) if not algo_trades.empty else 0.0
        rows.append(
            {
                "algo": algo,
                "total_return": float(curve.iloc[-1] / curve.iloc[0] - 1.0),
                "sharpe": float(sharpe),
                "max_drawdown": float(drawdown.min()),
                "avg_turnover": float(frame["turnover"].mean()),
                "total_cost": total_cost,
            }
        )
    return pd.DataFrame(rows)


def _run_one(
    dates: list[pd.Timestamp],
    by_date: dict[pd.Timestamp, pd.DataFrame],
    score_map: dict[pd.Timestamp, pd.Series],
    sigma_map: dict[tuple[str, pd.Timestamp], float],
    algo: str,
    config: BacktestConfig,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    cash = float(config.initial_capital)
    shares: dict[str, float] = {}
    last_price: dict[str, float] = {}
    pending: list[dict[str, object]] = []
    equity_rows: list[dict[str, object]] = []
    trade_rows: list[dict[str, object]] = []

    for date in dates:
        day = by_date[date]
        traded_notional = 0.0
        for order in pending:
            symbol = str(order["symbol"])
            if symbol not in day.index:
                continue
            bar = day.loc[symbol]
            quantity = float(order["quantity"])
            if quantity <= 0:
                continue
            fill = execute_order(
                open=float(bar["open"]),
                high=float(bar["high"]),
                low=float(bar["low"]),
                close=float(bar["close"]),
                volume=float(bar["volume"]),
                quantity=quantity,
                side=str(order["side"]),
                algo=algo,
                sigma=float(order["sigma"]),
                n_bins=config.n_bins,
                commission_per_share=config.commission_per_share,
                half_spread_bps=config.half_spread_bps,
                impact_coef=config.impact_coef,
                participation_limit=config.participation_limit,
            )
            filled = float(fill.filled)
            if filled <= 0:
                continue
            signed = filled if order["side"] == "buy" else -filled
            shares[symbol] = shares.get(symbol, 0.0) + signed
            cash += float(fill.cash)
            raw_notional = float(fill.raw_vwap) * filled
            cost = _fill_cost(fill.cash, raw_notional, str(order["side"]))
            traded_notional += raw_notional
            trade_rows.append(
                {
                    "algo": algo,
                    "signal_date": order["signal_date"],
                    "fill_date": date,
                    "symbol": symbol,
                    "side": order["side"],
                    "filled": filled,
                    "unfilled": float(fill.unfilled),
                    "raw_vwap": float(fill.raw_vwap),
                    "fill_vwap": float(fill.fill_vwap),
                    "cash": float(fill.cash),
                    "cost": cost,
                }
            )
        pending = []

        for symbol in day.index:
            last_price[symbol] = float(day.loc[symbol, "close"])
        equity = cash + sum(qty * last_price.get(symbol, 0.0) for symbol, qty in shares.items())
        previous = equity_rows[-1]["equity"] if equity_rows else None
        base = float(previous) if previous is not None else equity
        equity_rows.append(
            {
                "algo": algo,
                "date": date,
                "equity": equity,
                "ret": np.nan if previous is None else equity / float(previous) - 1.0,
                "cash": cash,
                "turnover": 0.0 if base == 0 else traded_notional / base,
                "traded_notional": traded_notional,
            }
        )
        pending = _orders_for_next_session(date, day, score_map, sigma_map, shares, equity, config)

    equity_frame = pd.DataFrame(equity_rows)
    trade_frame = pd.DataFrame(trade_rows) if trade_rows else _empty_trades()
    return equity_frame, trade_frame


def _orders_for_next_session(
    date: pd.Timestamp,
    day: pd.DataFrame,
    score_map: dict[pd.Timestamp, pd.Series],
    sigma_map: dict[tuple[str, pd.Timestamp], float],
    shares: dict[str, float],
    equity: float,
    config: BacktestConfig,
) -> list[dict[str, object]]:
    scores = score_map.get(date)
    if scores is None or int(scores.notna().sum()) < config.min_names:
        return []
    if equity <= 0:
        weights = pd.Series(0.0, index=scores.dropna().index, dtype=float)
    else:
        weights = construct_weights(scores, gross=config.gross, max_weight=config.max_weight)
    symbols = set(weights.index) | set(shares)
    orders: list[dict[str, object]] = []
    for symbol in sorted(symbols):
        if symbol not in day.index:
            continue
        price = float(day.loc[symbol, "close"])
        if price <= 0:
            continue
        target = float(weights.get(symbol, 0.0)) * equity / price
        delta = target - shares.get(symbol, 0.0)
        if abs(delta) * price < config.min_trade_dollars:
            continue
        if delta == 0:
            continue
        orders.append(
            {
                "symbol": symbol,
                "signal_date": date,
                "side": "buy" if delta > 0 else "sell",
                "quantity": abs(delta),
                "sigma": float(sigma_map.get((symbol, date), 0.0)),
            }
        )
    return orders


def _fill_cost(cash: float, raw_notional: float, side: str) -> float:
    if side == "buy":
        return -cash - raw_notional
    return raw_notional - cash


def _prepare_bars(bars: pd.DataFrame) -> pd.DataFrame:
    missing = [column for column in _BAR_COLUMNS if column not in bars.columns]
    if missing:
        raise ValueError(f"bars are missing columns: {missing}")
    frame = bars.loc[:, _BAR_COLUMNS].copy()
    frame["date"] = _normalize_dates(frame["date"])
    frame["symbol"] = frame["symbol"].astype(str)
    if frame.duplicated(["symbol", "date"]).any():
        raise ValueError("bars have duplicate symbol and date rows")
    if frame.empty:
        raise ValueError("bars are empty")
    return adjusted_ohlc(frame)


def _score_map(scores: pd.DataFrame) -> dict[pd.Timestamp, pd.Series]:
    required = ["symbol", "date", "score"]
    missing = [column for column in required if column not in scores.columns]
    if missing:
        raise ValueError(f"scores are missing columns: {missing}")
    frame = scores.loc[:, required].copy()
    frame["date"] = _normalize_dates(frame["date"])
    frame["symbol"] = frame["symbol"].astype(str)
    if frame.duplicated(["symbol", "date"]).any():
        raise ValueError("scores have duplicate symbol and date rows")
    mapped: dict[pd.Timestamp, pd.Series] = {}
    for date, group in frame.groupby("date", sort=True):
        mapped[pd.Timestamp(date)] = group.set_index("symbol")["score"].astype(float)
    return mapped


def _sigma_map(volatility: pd.DataFrame | None) -> dict[tuple[str, pd.Timestamp], float]:
    if volatility is None or volatility.empty:
        return {}
    required = ["symbol", "date", "sigma"]
    missing = [column for column in required if column not in volatility.columns]
    if missing:
        raise ValueError(f"volatility is missing columns: {missing}")
    frame = volatility.loc[:, required].dropna().copy()
    frame["date"] = _normalize_dates(frame["date"])
    frame["symbol"] = frame["symbol"].astype(str)
    return {
        (str(row.symbol), pd.Timestamp(row.date)): float(row.sigma)
        for row in frame.itertuples(index=False)
    }


def _normalize_dates(values: pd.Series) -> pd.Series:
    dates = pd.to_datetime(values)
    if isinstance(dates.dtype, pd.DatetimeTZDtype):
        dates = dates.dt.tz_convert("UTC").dt.tz_localize(None)
    return dates.dt.normalize()


def _empty_trades() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "algo",
            "signal_date",
            "fill_date",
            "symbol",
            "side",
            "filled",
            "unfilled",
            "raw_vwap",
            "fill_vwap",
            "cash",
            "cost",
        ]
    )
