"""Write the out-of-sample IC report as markdown plus CSV tables."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from driftwood.models.ic import (
    ic_by_year,
    information_coefficient,
    prediction_regression,
    summarize_ic,
)
from driftwood.models.walkforward import WalkForwardResult


def write_ic_report(
    result: WalkForwardResult,
    directory: Path,
    *,
    label_column: str = "fwd_ret_1d",
) -> Path:
    """Write ``ic.md`` and the tables next to it. Returns the markdown path."""

    directory.mkdir(parents=True, exist_ok=True)
    daily = information_coefficient(
        result.predictions, "prediction", label_column, method="spearman"
    )
    pearson = information_coefficient(
        result.predictions, "prediction", label_column, method="pearson"
    )
    summary = summarize_ic(daily, hac_lags=result.config.label_horizon)
    pearson_summary = summarize_ic(pearson, hac_lags=result.config.label_horizon)
    by_year = ic_by_year(daily, hac_lags=result.config.label_horizon)
    regression = prediction_regression(result.predictions, "prediction", label_column)

    daily.to_csv(directory / "ic_daily.csv", index=False)
    by_year.to_csv(directory / "ic_by_year.csv", index=False)
    result.folds.to_csv(directory / "folds.csv", index=False)
    result.weights.to_csv(directory / "weights.csv", index=False)

    path = directory / "ic.md"
    path.write_text(
        _markdown(result, summary, pearson_summary, by_year, regression, label_column),
        encoding="utf-8",
    )
    return path


def _markdown(
    result: WalkForwardResult,
    summary: dict[str, float | int],
    pearson: dict[str, float | int],
    by_year: pd.DataFrame,
    regression: dict[str, float | int],
    label_column: str,
) -> str:
    config = result.config
    window = "expanding" if config.expanding else f"rolling {config.train_days} sessions"
    lines = [
        "# Information coefficient",
        "",
        "These forecasts are out of sample. Each test window was predicted by a model",
        "fit only on earlier sessions, with a purge in between so the training labels",
        "do not reach into the test window.",
        "",
        "The information coefficient on a day is the correlation, across names, between",
        "the forecast and the realized forward return. The mean IC averages that number",
        "across days. A positive mean means the ranking was right more often than not.",
        "It is not a return and it is not a Sharpe ratio.",
        "",
        f"- Model: `{result.model_name}`",
        f"- Label: `{label_column}`",
        f"- Window: {window}",
        f"- Minimum training sessions: {config.min_train_days}",
        f"- Test sessions: {config.test_days}",
        f"- Purge sessions: {config.purge}",
        f"- Embargo requested: {config.embargo_days}",
        f"- Label horizon: {config.label_horizon}",
        f"- Folds: {len(result.folds)}",
        "",
        "## Spearman IC (headline)",
        "",
        f"- Days: {summary['n_dates']}",
        f"- Mean IC: {_num(summary['mean_ic'])}",
        f"- Std of daily IC: {_num(summary['std_ic'])}",
        f"- IC IR (mean / std): {_num(summary['ic_ir'])}",
        f"- Hit rate (share of days with IC > 0): {_num(summary['hit_rate'])}",
        f"- t-stat, treating days as independent: {_num(summary['t_stat'])}",
        f"- t-stat, Newey-West with {config.label_horizon} lag(s): {_num(summary['hac_t_stat'])}",
        "",
        "Quote the Newey-West t-stat when the label horizon is longer than one day.",
        "Overlapping return windows make the independent-days t-stat too flattering.",
        "",
        "## Pearson IC",
        "",
        f"- Mean IC: {_num(pearson['mean_ic'])}",
        f"- IC IR: {_num(pearson['ic_ir'])}",
        "",
        "## Pooled regression",
        "",
        "Ordinary least squares of the realized return on the forecast, pooling every",
        "name and day. This is not the IC. The slope is the hedge ratio of the forecast.",
        "",
        f"- N: {regression['n']}",
        f"- Slope: {_num(regression['slope'])}",
        f"- Slope t-stat: {_num(regression['slope_t'])}",
        f"- R-squared: {_num(regression['r2'])}",
        "",
        "## In-sample, for the overfit comparison",
        "",
        "This is the average of each fold's in-sample mean Spearman IC. It is not a",
        "number to trade. If it is far above the out-of-sample mean, the fit is",
        "memorizing the training window.",
        "",
        f"- In-sample mean IC: {_num(result.in_sample_mean_ic)}",
        "",
        "## By year",
        "",
    ]
    if by_year.empty:
        lines.append("No dated IC observations.")
    else:
        lines.append("| Year | Days | Mean IC | IC IR | Hit rate |")
        lines.append("| --- | --- | --- | --- | --- |")
        for row in by_year.itertuples(index=False):
            lines.append(
                f"| {int(row.year)} | {int(row.n_dates)} | {_num(row.mean_ic)} | "
                f"{_num(row.ic_ir)} | {_num(row.hit_rate)} |"
            )
    lines.append("")
    return "\n".join(lines)


def _num(value: object) -> str:
    try:
        number = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return "n/a"
    if number != number:  # NaN
        return "n/a"
    return f"{number:.4f}"
