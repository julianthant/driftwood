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
from dashboard.views import attribution_table  # noqa: E402
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
    st.caption("Run from the repository root, then choose Refresh data in the sidebar.")
    st.code("\n".join(commands), language="bash")


def plot(chart) -> None:
    st.plotly_chart(
        chart,
        width="stretch",
        theme=None,
        config={"displaylogo": False, "scrollZoom": False},
    )


def number(value: float, spec: str = ".3f") -> str:
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
    metrics(
        [
            ("Universe · symbols", f"{len(study.coverage):,}"),
            ("Observed sessions", f"{study.sessions:,}"),
            ("First session", study.first.strftime("%d %b %Y")),
            ("Last session", study.last.strftime("%d %b %Y")),
        ]
    )
    if model:
        quality = quality_data(path, model, revision)
        st.subheader("Signal evidence")
        metrics(
            [
                ("Mean Spearman IC", number(quality.summary["mean_ic"])),
                ("IC t-stat · Newey–West", number(quality.summary["hac_t_stat"], ".2f")),
                ("Scored sessions", f"{quality.summary['n_dates']:,}"),
            ]
        )
        st.caption(
            f"Walk-forward predictions · {model} · label: {quality.label} · "
            f"HAC allowance: {quality.hac_lags} sessions. IC measures ranking, before costs."
        )
        result = get_backtest(path, model, revision)
        if result is not None:
            best = result.summary.sort_values("total_return", ascending=False).iloc[0]
            st.subheader(f"Execution snapshot · {str(best['algo']).upper()}")
            metrics(
                [
                    ("Total return", number(best["total_return"], ".2%")),
                    ("Annualized Sharpe", number(best["sharpe"], ".2f")),
                    ("Max drawdown", number(best["max_drawdown"], ".2%")),
                    ("Fill cost · USD", "$" + number(best["total_cost"], ",.2f")),
                ]
            )
            st.caption(
                "Highest return in this sample, shown for orientation. "
                "This is a retrospective comparison, not a schedule recommendation."
            )
            plot(charts.equity_chart(result.daily))
            st.caption(
                "Growth of $1 after simulated fill costs. All schedules use the same scores."
            )
    else:
        empty_state(
            "Market data is ready. Train a model to see signal and execution evidence.",
            [
                command("features", path),
                command("train", path),
            ],
        )
    st.subheader("From market data to a measured result")
    st.markdown(
        "**Daily bars → close-of-day features → purged walk-forward forecasts → "
        "next-session C++ fills → signal attribution.**\n\n"
        "Each forecast uses past training data. The backtest sizes at the signal close and fills "
        "on the next session. Immediate, TWAP and VWAP share targets but pay different path prices "
        "and costs. Attribution explains the target book’s next close-to-close return."
    )
    st.caption(
        "Research limits: a fixed present-day universe has survivorship bias; "
        "intraday paths are modeled from daily bars and impact is uncalibrated. "
        "These are research results, not live performance."
    )


def backtest(path: str, model: str, revision: tuple) -> None:
    result = get_backtest(path, model, revision)
    if result is None:
        return
    schedules = result.summary["algo"].astype(str).tolist()
    selected = st.sidebar.selectbox("Schedule", schedules, format_func=str.upper)
    summary = result.summary.set_index("algo").loc[selected]
    daily = result.daily.loc[result.daily["algo"] == selected]
    metrics(
        [
            ("Total return", number(summary["total_return"], ".2%")),
            ("Annualized Sharpe", number(summary["sharpe"], ".2f")),
            ("Max drawdown", number(summary["max_drawdown"], ".2%")),
            ("Fill cost · USD", "$" + number(summary["total_cost"], ",.2f")),
        ]
    )
    st.subheader("Equity curve")
    plot(charts.equity_chart(daily))
    st.caption(
        "Growth of $1, marked to adjusted closing prices after spread, impact and commission."
    )
    st.subheader("Below the high-water mark")
    plot(charts.drawdown_chart(daily))
    st.caption(
        "Equity divided by its running peak minus one. Zero means a new high; "
        "negative values show the loss from that high."
    )
    with st.expander("Run assumptions and daily ledger"):
        st.caption(
            f"Saved {result.created_at} · Model {model}. "
            "The ledger includes the initial flat period before predictions start."
        )
        st.json(result.config)
        st.dataframe(daily, hide_index=True, width="stretch")


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
    st.subheader("Same targets, three execution schedules")
    table = result.summary.drop(columns="model").rename(
        columns={
            "algo": "Schedule",
            "total_return": "Total return",
            "sharpe": "Sharpe",
            "max_drawdown": "Max drawdown",
            "avg_turnover": "Avg daily turnover",
            "total_cost": "Cost (USD)",
        }
    )
    st.dataframe(
        table.style.format(
            {
                "Total return": "{:.2%}",
                "Sharpe": "{:.2f}",
                "Max drawdown": "{:.2%}",
                "Avg daily turnover": "{:.2%}",
                "Cost (USD)": "${:,.2f}",
            },
            na_rep="—",
        ),
        hide_index=True,
        width="stretch",
    )
    st.caption(
        "Sharpe uses zero risk-free rate and 252 sessions/year. Turnover is traded raw "
        "notional divided by prior equity, averaged over the full stored window."
    )
    st.subheader("Equity comparison")
    plot(charts.equity_chart(result.daily))
    st.caption(
        "Independent books with identical signals and starting capital. "
        "The path inside each daily bar is simulated, not observed intraday trading."
    )
    st.subheader("The cost of crossing the spread")
    plot(charts.cost_chart(result.daily))
    st.caption(
        "Accumulated spread, impact and commission in USD, measured from fill cash "
        "versus raw notional. Market price moves are excluded."
    )


def model_quality(path: str, model: str, revision: tuple) -> None:
    quality = quality_data(path, model, revision)
    metrics(
        [
            ("Mean Spearman IC", number(quality.summary["mean_ic"])),
            ("IC t-stat · Newey–West", number(quality.summary["hac_t_stat"], ".2f")),
            ("Positive IC sessions", number(quality.summary["hit_rate"], ".1%")),
        ]
    )
    st.caption(
        f"Model: {model} · label: {quality.label} · HAC lags: {quality.hac_lags}. "
        "Legacy predictions without label metadata use a five-lag allowance."
    )
    if not quality.summary["n_dates"]:
        empty_state(
            "No session has enough distinct forecasts and realized labels to compute IC.",
            [command("train", path, model)],
        )
        return
    st.subheader("Did the forecast rank the cross-section correctly?")
    plot(charts.ic_chart(quality.daily))
    st.caption(
        "Daily Spearman correlation across symbols, before costs. The darker line is a "
        "63-session mean requiring a full window of valid ICs; undefined days remain gaps."
    )
    left, right = st.columns(2)
    with left:
        st.subheader("Distribution of daily IC")
        plot(charts.ic_histogram(quality.daily))
        st.caption(
            "A positive IC means higher forecasts ranked above lower forecasts "
            "in realized returns. Missing ICs are excluded."
        )
    with right:
        st.subheader("Cumulative IC")
        plot(charts.ic_chart(quality.daily, cumulative=True))
        st.caption(
            "A running sum of valid daily ICs. This shows consistency of ranking; "
            "it is not a compounded return or an equity curve."
        )
    st.subheader("What the model learned")
    if quality.coefficients.empty:
        st.info(
            "No coefficients are available for this model. "
            "Tree models do not provide linear coefficients; ridge stores them by fold."
        )
    else:
        latest = quality.coefficients["fold"].max()
        plot(
            charts.contribution_chart(
                quality.coefficients.loc[quality.coefficients["fold"] == latest], coefficient=True
            )
        )
        st.caption(
            f"Latest fold ({latest}). Linear weights multiply the day's feature z-scores; "
            "their size is not a causal effect."
        )
        with st.expander("All fold coefficients"):
            st.dataframe(quality.coefficients, hide_index=True, width="stretch")


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
            ("Target-book return", number(result.portfolio_return, ".2%")),
        ]
    )
    st.subheader("Contributions by signal")
    plot(charts.contribution_chart(result.by_signal))
    st.caption(
        "Contributions add to the target book's next close-to-close return, in basis points. "
        "This accounting excludes execution costs and differs from filled backtest P&L."
    )
    st.write(result.note)
    st.caption(
        "Attribution uses the library defaults: gross target 1.0 and max weight 10%. "
        "It is computed from the latest eligible scored session; no report file is required."
    )
    table = attribution_table(result.by_signal)
    table["Contribution (bps)"] = table.pop("contribution") * 10_000
    st.dataframe(
        table.style.format({"Contribution (bps)": "{:.3f}"}),
        hide_index=True,
        width="stretch",
    )


def market_data(study: data.Study) -> None:
    observed = int(study.coverage["sessions"].sum())
    possible = len(study.coverage) * study.sessions
    metrics(
        [
            ("Universe · symbols", f"{len(study.coverage):,}"),
            ("Observed sessions", f"{study.sessions:,}"),
            ("Observed symbol-days", f"{observed:,}"),
            ("Missing symbol-days", f"{possible - observed:,}"),
        ]
    )
    st.caption(
        f"{study.first:%d %b %Y} – {study.last:%d %b %Y} · "
        "Universe includes requested symbols with no bars."
    )
    st.subheader("Sessions per symbol")
    plot(charts.coverage_chart(study.coverage))
    st.caption(
        "The number of stored daily bars for each symbol. Shorter histories deserve a check."
    )
    st.dataframe(study.coverage, hide_index=True, width="stretch")
    st.subheader("Where observations are missing")
    plot(charts.missing_chart(study.missing))
    st.caption(
        "Monthly fraction of missing observations relative to dates seen for any symbol "
        "in this database. Includes gaps before listing and after the last observation. "
        "This is not an exchange-calendar audit: a session missing for every symbol "
        "cannot be detected here."
    )


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
    st.sidebar.caption("Quantitative research workbench")
    page = st.sidebar.radio("Research", PAGES)
    path = str(
        Path(st.sidebar.text_input("Database path", value=str(database_path())))
        .expanduser()
        .resolve()
    )
    if st.sidebar.button("Refresh data", width="stretch"):
        for loader in (study_data, quality_data, backtest_data, attribution_data):
            loader.clear()
    st.sidebar.divider()
    st.sidebar.caption(
        "Read only · Local study\n\nNo credentials required. All fills are simulated."
    )
    st.title(page)
    st.caption(
        {
            "Overview": "The research pipeline, from universe coverage to execution outcomes.",
            "Backtest": "Follow one schedule through returns, costs and drawdowns.",
            "Execution": "Compare how execution timing changes the same target portfolio.",
            "Model / Signals": "Inspect out-of-sample ranking quality across walk-forward folds.",
            "Attribution": "Understand which signals contributed to the latest eligible session.",
            "Data": "Inspect the history underneath every forecast and fill.",
        }[page]
    )
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
        model = st.sidebar.selectbox("Model", study.models) if study.models else None
        source = ", ".join(study.sources)
        st.caption(
            f"Source: {source} · "
            + (
                "Synthetic data is an offline demonstration, not evidence of trading edge."
                if "synthetic" in study.sources
                else "Historical study; simulated execution."
            )
        )
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
        else:
            {
                "Backtest": backtest,
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
