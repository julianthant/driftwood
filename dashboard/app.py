"""Read-only research workbench. Run: streamlit run dashboard/app.py."""

from __future__ import annotations

import math
import sys
from pathlib import Path

import duckdb
import streamlit as st

root = str(Path(__file__).resolve().parents[1])
if root not in sys.path:
    sys.path.insert(0, root)

from dashboard import charts, data  # noqa: E402
from dashboard.formatting import compact_money, schedule_label, signal_label  # noqa: E402
from dashboard.views import attribution_table, shared_cost_bps  # noqa: E402
from driftwood.config import database_path, load_env  # noqa: E402

PAGES = ("Overview", "Backtest", "Execution", "Model / Signals", "Attribution", "Data")
READ_ERRORS = (OSError, duckdb.Error, RuntimeError, ValueError, KeyError)


@st.cache_data(ttl=300, max_entries=8, show_spinner=False)
def study_data(path: str, revision: tuple) -> data.Study:
    return data.load_study(path)


@st.cache_data(ttl=300, max_entries=16, show_spinner=False)
def quality_data(path: str, model: str, revision: tuple) -> data.ModelQuality:
    return data.load_quality(path, model)


@st.cache_data(ttl=300, max_entries=16, show_spinner=False)
def backtest_data(path: str, model: str, revision: tuple) -> data.Backtest:
    return data.load_backtest(path, model)


@st.cache_data(ttl=300, max_entries=16, show_spinner=False)
def attribution_data(path: str, model: str, revision: tuple):
    return data.load_attribution(path, model)


def command(action: str, path: str, model: str = "ridge") -> str:
    suffix = f'--db "{path}"'
    if action in {"train", "backtest", "attribute"}:
        suffix += f" --model {model}"
    return f"driftwood {action} {suffix}"


def empty_state(message: str, commands: list[str]) -> None:
    st.info(message)
    st.caption("Run from the repository root, then refresh in Settings.")
    st.code("\n".join(commands), language="bash")


def plot(chart) -> None:
    st.plotly_chart(
        chart,
        width="stretch",
        theme=None,
        config={"displaylogo": False, "scrollZoom": False, "displayModeBar": False},
    )


def number(value: float, spec: str = ".2f") -> str:
    return format(value, spec) if math.isfinite(value) else "—"


def metrics(items: list[tuple[str, str]]) -> None:
    for column, (label, value) in zip(st.columns(len(items)), items, strict=True):
        column.metric(label, value)


def get_backtest(path: str, model: str, revision: tuple) -> data.Backtest | None:
    try:
        return backtest_data(path, model, revision)
    except READ_ERRORS as exc:
        message = (
            str(exc)
            if isinstance(exc, RuntimeError)
            else "The saved backtest could not be read. Close any CLI writer and rebuild it."
        )
        empty_state(message, [command("backtest", path, model)])
        return None


def overview(study: data.Study, path: str, model: str | None, revision: tuple) -> None:
    st.caption(f"{study.first:%d %b %Y} - {study.last:%d %b %Y}")
    headline = [("Universe", f"{len(study.coverage):,}"), ("Sessions", f"{study.sessions:,}")]
    if model:
        quality = quality_data(path, model, revision)
        headline.extend(
            [
                ("Mean IC", number(quality.summary["mean_ic"], ".3f")),
                ("IC t-stat", number(quality.summary["hac_t_stat"])),
            ]
        )
    metrics(headline)
    if model:
        result = get_backtest(path, model, revision)
        if result is not None:
            best = result.summary.sort_values("total_return", ascending=False).iloc[0]
            st.subheader(f"Best sample return · {schedule_label(str(best['algo']))}")
            metrics(
                [
                    ("Total return", number(best["total_return"], ".1%")),
                    ("Sharpe", number(best["sharpe"])),
                    ("Max drawdown", number(best["max_drawdown"], ".1%")),
                    ("Fill cost", compact_money(best["total_cost"])),
                ]
            )
            st.caption(
                f"Fill cost: {number(best['cost_bps'], '.1f')} bps of traded notional."
            )
            plot(charts.equity_chart(result.daily))
            st.caption("Growth of $1 after simulated fill costs.")
    else:
        empty_state(
            "Market data is ready. Train a model to see signal and execution evidence.",
            [
                command("features", path),
                command("train", path),
            ],
        )


def backtest(path: str, model: str, revision: tuple, controls) -> None:
    result = get_backtest(path, model, revision)
    if result is None:
        return
    schedules = result.summary["algo"].astype(str).tolist()
    selected = controls.selectbox("Schedule", schedules, format_func=schedule_label)
    summary = result.summary.set_index("algo").loc[selected]
    daily = result.daily.loc[result.daily["algo"] == selected]
    metrics(
        [
            ("Total return", number(summary["total_return"], ".1%")),
            ("Sharpe", number(summary["sharpe"])),
            ("Max drawdown", number(summary["max_drawdown"], ".1%")),
            ("Fill cost", compact_money(summary["total_cost"])),
        ]
    )
    st.caption(
        f"Fill cost: {number(summary['cost_bps'], '.1f')} bps of "
        f"{compact_money(summary['total_traded_notional'])} traded notional."
    )
    st.subheader("Equity curve")
    plot(charts.equity_chart(daily))
    st.caption("Growth of $1 after simulated fill costs.")
    st.subheader("Drawdown")
    plot(charts.drawdown_chart(daily))
    st.caption("Loss from the running equity peak.")
    with st.expander("Run assumptions and daily ledger"):
        st.caption(f"Saved {result.created_at:%d %b %Y}")
        st.json(result.config)
        ledger = daily[
            ["date", "algo", "equity", "drawdown", "turnover", "cost", "traded_notional"]
        ].copy()
        ledger["algo"] = ledger["algo"].map(schedule_label)
        st.dataframe(
            ledger.style.format(
                {
                    "equity": compact_money,
                    "drawdown": "{:.1%}",
                    "turnover": "{:.1%}",
                    "cost": compact_money,
                    "traded_notional": compact_money,
                }
            ),
            hide_index=True,
            width="stretch",
            column_config={
                "algo": "Schedule",
                "date": st.column_config.DateColumn("Session", format="DD MMM YYYY"),
                "equity": "Equity",
                "drawdown": "Drawdown",
                "turnover": "Turnover",
                "cost": "Fill cost",
                "traded_notional": "Traded notional",
            },
        )


def execution(path: str, model: str, revision: tuple) -> None:
    result = get_backtest(path, model, revision)
    if result is None:
        return
    absent = {"immediate", "twap", "vwap"} - set(result.summary["algo"])
    if absent:
        empty_state(
            "Only part of the schedule comparison is saved. Run all three schedules:",
            [
                command("backtest", path, model) + " --algos immediate twap vwap",
            ],
        )
    table = result.summary.drop(columns=["model", "total_traded_notional"]).rename(
        columns={
            "algo": "Schedule",
            "total_return": "Total return",
            "sharpe": "Sharpe",
            "max_drawdown": "Max drawdown",
            "avg_turnover": "Turnover",
            "total_cost": "Fill cost",
            "cost_bps": "Cost (bps)",
        }
    )
    table["Schedule"] = table["Schedule"].map(schedule_label)
    st.dataframe(
        table.style.format(
            {
                "Total return": "{:.1%}",
                "Sharpe": "{:.2f}",
                "Max drawdown": "{:.1%}",
                "Turnover": "{:.1%}",
                "Fill cost": compact_money,
                "Cost (bps)": "{:.1f}",
            },
            na_rep="—",
        ),
        hide_index=True,
        width="stretch",
    )
    st.caption("Turnover is the average daily traded notional / prior equity.")
    plot(charts.comparison_chart(result.summary))
    st.subheader("Equity comparison")
    plot(charts.equity_chart(result.daily))
    st.subheader("Fill costs")
    rate = shared_cost_bps(result.summary)
    if rate is not None:
        spread_only = (
            result.config.get("impact_coef") == 0
            and result.config.get("commission_per_share") == 0
        )
        st.caption(
            f"{number(rate, '.1f')} bps of traded notional across schedules"
            + ("; spread only, no impact or commission." if spread_only else ".")
        )
    plot(charts.comparison_chart(result.summary, costs=True))
    if "immediate" in set(result.summary["algo"]) and len(result.summary) > 1:
        st.subheader("Cost relative to Immediate")
        plot(charts.cost_chart(result.daily))
        st.caption("Cumulative fill cost minus Immediate; positive means higher cost.")


def model_quality(path: str, model: str, revision: tuple) -> None:
    quality = quality_data(path, model, revision)
    metrics(
        [
            ("Mean IC", number(quality.summary["mean_ic"], ".3f")),
            ("IC t-stat", number(quality.summary["hac_t_stat"])),
            ("Positive IC", number(quality.summary["hit_rate"], ".1%")),
        ]
    )
    st.caption(f"Label: {quality.label} · Newey-West lags: {quality.hac_lags}")
    if not quality.label_known:
        st.caption("Legacy predictions: label unknown; five-lag allowance.")
    if not quality.summary["n_dates"]:
        empty_state(
            "No session has enough distinct forecasts and realized labels to compute IC.",
            [command("train", path, model)],
        )
        return
    st.subheader("Daily IC")
    plot(charts.ic_chart(quality.daily))
    st.caption("Spearman rank correlation with a 63-session rolling mean.")
    distribution, cumulative, coefficients = st.tabs(
        ["Distribution", "Cumulative IC", "Coefficients"]
    )
    with distribution:
        plot(charts.ic_histogram(quality.daily))
        st.caption("Positive IC means forecasts ranked realized returns correctly.")
    with cumulative:
        plot(charts.ic_chart(quality.daily, cumulative=True))
        st.caption("Sum of daily IC, not a compounded return.")
    with coefficients:
        if quality.coefficients.empty:
            st.info("No linear coefficients are available for this model.")
        else:
            latest = quality.coefficients["fold"].max()
            plot(
                charts.contribution_chart(
                    quality.coefficients.loc[quality.coefficients["fold"] == latest],
                    coefficient=True,
                )
            )
            st.caption(f"Fold {latest}: weights on standardized features.")
            with st.expander("All fold coefficients"):
                st.dataframe(
                    quality.coefficients.style.format({"weight": "{:.2e}"}),
                    hide_index=True,
                    width="stretch",
                )


def attribution(path: str, model: str, revision: tuple) -> None:
    try:
        result = attribution_data(path, model, revision)
    except READ_ERRORS:
        empty_state(
            "Attribution needs features, predictions and a later bar with matching symbols.",
            [
                command("features", path),
                command("train", path, model),
                command("attribute", path, model),
            ],
        )
        return
    metrics(
        [
            ("Signal close", result.signal_date.strftime("%d %b %Y")),
            ("Next close", result.next_date.strftime("%d %b %Y")),
            ("Book return", number(result.portfolio_return, ".1%")),
        ]
    )
    st.subheader("Contributions by signal")
    plot(charts.contribution_chart(result.by_signal))
    st.caption("Next close-to-close target-book return, before execution costs.")
    st.caption(
        "No stored coefficients; attribution uses unit feature weights."
        if result.note.startswith("No coefficients")
        else result.note.split(". ")[0].rstrip(".") + "."
    )
    if "Feature rows are missing" in result.note or "No feature row" in result.note:
        st.caption("Signal-day features are missing; the return is unattributed.")
    elif "None of the coefficient columns" in result.note:
        st.caption("No matching feature columns; the return is unattributed.")
    table = attribution_table(result.by_signal)
    table["signal"] = table["signal"].map(signal_label)
    table["Contribution (bps)"] = table.pop("contribution") * 10_000
    st.table(
        table.rename(columns={"signal": "Signal"})
        .style.format({"Contribution (bps)": "{:.1f}"})
        .hide(axis="index"),
        border="horizontal",
    )


def market_data(study: data.Study) -> None:
    observed = int(study.coverage["sessions"].sum())
    possible = len(study.coverage) * study.sessions
    metrics(
        [
            ("Universe", f"{len(study.coverage):,}"),
            ("Sessions", f"{study.sessions:,}"),
            ("Observed symbol-days", f"{observed:,}"),
            ("Missing symbol-days", f"{possible - observed:,}"),
        ]
    )
    st.caption(f"{study.first:%d %b %Y} - {study.last:%d %b %Y}")
    st.subheader("Sessions per symbol")
    plot(charts.coverage_chart(study.coverage))
    st.dataframe(
        study.coverage.sort_values(["sessions", "symbol"]).style.format(
            {"sessions": "{:,.0f}", "missing": "{:,.0f}"}
        ),
        hide_index=True,
        width="stretch",
        column_config={
            "symbol": "Symbol",
            "sessions": "Sessions",
            "missing": "Missing sessions",
            "first": st.column_config.DateColumn("First session", format="DD MMM YYYY"),
            "last": st.column_config.DateColumn("Last session", format="DD MMM YYYY"),
        },
    )
    st.subheader("Missing data")
    if possible == observed:
        st.success("No missing symbol-days")
    else:
        affected = study.missing.loc[study.missing.gt(0).any(axis=1)]
        plot(charts.missing_chart(affected))
        st.caption("Monthly gaps relative to sessions observed for any symbol.")


def main() -> None:
    load_env()
    st.set_page_config(
        page_title="Driftwood · Research", layout="wide", initial_sidebar_state="expanded"
    )
    st.markdown(
        f"<style>{Path(__file__).with_name('style.css').read_text()}</style>",
        unsafe_allow_html=True,
    )
    st.sidebar.title("driftwood")
    st.sidebar.caption("Research · read only")
    page = st.sidebar.radio("Research", PAGES)
    controls = st.sidebar.container()
    st.sidebar.divider()
    with st.sidebar.expander("Settings"):
        path = str(Path(st.text_input("Database path", value=str(database_path()))).expanduser())
        path = str(Path(path).resolve())
        if st.button("Refresh data", width="stretch"):
            for loader in (study_data, quality_data, backtest_data, attribution_data):
                loader.clear()
    source_slot = st.sidebar.empty()
    source_slot.caption("Simulated fills only")
    st.title("Model / signals" if page == "Model / Signals" else page)
    if not Path(path).is_file():
        empty_state(
            "No database at this path. Create an offline study to get started:",
            [
                command("ingest --synthetic --periods 504", path),
                command("features", path),
                command("train", path) + " --min-train-days 60 --test-days 20 --embargo 5",
                command("backtest", path),
                command("attribute", path),
            ],
        )
        return
    revision = data.file_revision(path)
    try:
        with st.spinner("Reading the study…"):
            study = study_data(path, revision)
        model = controls.selectbox("Model", study.models) if study.models else None
        source = ", ".join(study.sources)
        source_slot.caption(f"{source} · simulated fills")
        if page == "Data":
            market_data(study)
        elif page == "Overview":
            overview(study, path, model, revision)
        elif model is None:
            empty_state(
                "No predictions yet. Build features, then fit a walk-forward model:",
                [
                    command("features", path),
                    command("train", path),
                ],
            )
        elif page == "Backtest":
            backtest(path, model, revision, controls)
        else:
            {
                "Execution": execution,
                "Model / Signals": model_quality,
                "Attribution": attribution,
            }[page](path, model, revision)
    except READ_ERRORS:
        empty_state(
            "This study could not be read. Check the database path, close any running "
            "CLI writer, and refresh. For an empty study, start with ingestion:",
            [
                command("ingest --synthetic --periods 504", path),
                command("features", path),
                command("train", path),
            ],
        )


if __name__ == "__main__":
    main()
