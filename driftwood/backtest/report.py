"""Write the algorithm comparison as markdown plus the equity and summary tables."""

from __future__ import annotations

from pathlib import Path

from driftwood.backtest.engine import BacktestResult


def write_backtest_report(result: BacktestResult, directory: Path) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    result.equity.to_csv(directory / "equity.csv", index=False)
    result.summary.to_csv(directory / "summary.csv", index=False)
    result.trades.to_csv(directory / "trades.csv", index=False)
    path = directory / "backtest.md"
    path.write_text(_markdown(result), encoding="utf-8")
    return path


def _markdown(result: BacktestResult) -> str:
    config = result.config
    lines = [
        "# Backtest",
        "",
        "A score on the close of session t is sized with that close. The order",
        "is filled by the C++ simulator on session t+1. The move from the",
        "previous close to this close is marked before the new order exists,",
        "so it is not a profit of this signal.",
        "",
        "Prices are the adjusted series. The raw open, high, and low are scaled",
        "by adj_close / close. Volume is the stored volume. Sigma passed into",
        "the fill is the volatility known on the signal date.",
        "",
        "Sharpe uses a risk-free rate of zero and annualizes with the square",
        "root of the periods-per-year setting. Max drawdown is the worst",
        "equity-over-peak minus one, so a loss is negative. Turnover is traded",
        "notional divided by equity. Cost is spread, impact, and commission.",
        "It is not the market move.",
        "",
        f"- Initial capital: {config.initial_capital:.0f}",
        f"- Gross target: {config.gross}",
        f"- Max absolute weight: {config.max_weight}",
        f"- Half spread (bps): {config.half_spread_bps}",
        f"- Impact coefficient: {config.impact_coef}",
        f"- Participation limit: {config.participation_limit}",
        "",
        "| Algo | Total return | Sharpe | Max drawdown | Avg turnover | Cost |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for row in result.summary.itertuples(index=False):
        lines.append(
            f"| {row.algo} | {_num(row.total_return)} | {_num(row.sharpe)} | "
            f"{_num(row.max_drawdown)} | {_num(row.avg_turnover)} | {_num(row.total_cost)} |"
        )
    lines.append("")
    lines.append(
        "A higher total return is not, by itself, a better schedule. The three "
        "rows differ in when they cross the spread and in the path price they "
        "pay. None of them is a live trading result."
    )
    lines.append("")
    return "\n".join(lines)


def _num(value: object) -> str:
    try:
        number = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return "n/a"
    if number != number:
        return "n/a"
    return f"{number:.4f}"
