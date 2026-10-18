"""Plotly figures with shared research colors and explicit units."""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from dashboard.formatting import compact_money, schedule_label, signal_label
from dashboard.views import relative_costs, shared_cost_bps

COLORS = {"immediate": "#087f8c", "twap": "#c16319", "vwap": "#334155"}
DASHES = {"immediate": "solid", "twap": "dash", "vwap": "dot"}
TEAL = "#087f8c"
RED = "#ba4653"
CHART_HEIGHT = 300


def figure(y_title: str) -> go.Figure:
    chart = go.Figure()
    chart.update_layout(
        template="plotly_white",
        height=CHART_HEIGHT,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="#ffffff",
        font={"family": "Arial, sans-serif", "size": 13, "color": "#172b4d"},
        margin={"l": 18, "r": 18, "t": 40, "b": 25},
        legend={"orientation": "h", "y": 1.06, "x": 0, "yanchor": "bottom"},
        hovermode="x unified",
        yaxis_title=y_title,
        colorway=list(COLORS.values()),
    )
    chart.update_xaxes(showgrid=False, zeroline=False, hoverformat="%d %b %Y")
    chart.update_yaxes(gridcolor="#e8edf4", zeroline=False)
    return chart


def equity_chart(daily: pd.DataFrame) -> go.Figure:
    chart = figure("Growth of $1")
    for algo, frame in daily.groupby("algo", sort=True):
        chart.add_trace(
            go.Scatter(
                x=frame["date"],
                y=frame["growth"],
                name=schedule_label(str(algo)),
                mode="lines",
                line={
                    "color": COLORS.get(str(algo), TEAL),
                    "width": 2,
                    "dash": DASHES.get(str(algo), "solid"),
                },
                hovertemplate="%{y:.2f}×<extra>%{fullData.name}</extra>",
            )
        )
    chart.add_hline(y=1, line_dash="dot", line_color="#8b9ab1")
    return chart


def drawdown_chart(daily: pd.DataFrame) -> go.Figure:
    chart = figure("Drawdown")
    algo = str(daily["algo"].iloc[0])
    color = COLORS.get(algo, TEAL)
    rgb = ",".join(str(int(color[i : i + 2], 16)) for i in (1, 3, 5))
    chart.add_trace(
        go.Scatter(
            x=daily["date"],
            y=daily["drawdown"],
            name=schedule_label(algo),
            mode="lines",
            fill="tozeroy",
            line={"color": color, "width": 1.5, "dash": DASHES.get(algo, "solid")},
            fillcolor=f"rgba({rgb},0.12)",
            hovertemplate="%{y:.1%}<extra></extra>",
        )
    )
    chart.update_yaxes(tickformat=".1%")
    return chart


def cost_chart(daily: pd.DataFrame) -> go.Figure:
    """Show incremental dollars versus Immediate; the reference is an annotation."""
    chart = figure("Extra cost ($)")
    for algo, frame in relative_costs(daily).groupby("algo", sort=True):
        chart.add_trace(
            go.Scatter(
                x=frame["date"],
                y=frame["cost_difference"],
                name=schedule_label(str(algo)),
                mode="lines",
                line={
                    "color": COLORS.get(str(algo), TEAL),
                    "width": 2,
                    "dash": DASHES.get(str(algo), "solid"),
                },
                hovertemplate="$%{y:.3~s}<extra>%{fullData.name}</extra>",
            )
        )
    chart.update_yaxes(tickprefix="$", tickformat=".2~s")
    chart.add_hline(
        y=0,
        line_dash="dot",
        line_color="#8b9ab1",
        annotation_text="Immediate",
        annotation_position="bottom right",
    )
    return chart


def comparison_chart(summary: pd.DataFrame, *, costs: bool = False) -> go.Figure:
    """Separate panels keep dollars, basis points, returns and Sharpe in their own units."""
    metrics = (
        [
            ("total_cost", "Fill cost ($)", "$.3~s"),
            ("cost_bps", "Cost rate (bps)", ".1f"),
        ]
        if costs
        else [
            ("total_return", "Total return", ".1%"),
            ("sharpe", "Sharpe", ".2f"),
            ("max_drawdown", "Max drawdown", ".1%"),
        ]
    )
    if costs and shared_cost_bps(summary) is not None:
        metrics = metrics[:1]
    chart = make_subplots(rows=1, cols=len(metrics), subplot_titles=[m[1] for m in metrics])
    chart.update_layout(figure("").layout)
    ordered = summary.set_index("algo").reindex([a for a in COLORS if a in set(summary["algo"])])
    labels = [schedule_label(str(algo)) for algo in ordered.index]
    colors = [COLORS[str(algo)] for algo in ordered.index]
    for column, (metric, _, spec) in enumerate(metrics, start=1):
        values = ordered[metric]
        text = (
            [compact_money(value) for value in values]
            if metric == "total_cost"
            else [format(value, spec) if pd.notna(value) else "n/a" for value in values]
        )
        chart.add_trace(
            go.Bar(
                x=labels,
                y=values,
                marker_color=colors,
                text=text,
                textposition="outside",
                cliponaxis=False,
                showlegend=False,
                hovertemplate=(
                    "%{x}<br>%{y:$.3~s}<extra></extra>"
                    if metric == "total_cost"
                    else f"%{{x}}<br>%{{y:{spec}}}<extra></extra>"
                ),
            ),
            row=1,
            col=column,
        )
        chart.update_xaxes(showgrid=False, tickangle=0, row=1, col=column)
        chart.update_yaxes(gridcolor="#e8edf4", zeroline=False, tickformat=spec, row=1, col=column)
    chart.update_layout(hovermode="closest", margin={"t": 55, "b": 30}, bargap=0.55)
    chart.update_annotations(font_size=13)
    return chart


def ic_chart(daily: pd.DataFrame, *, cumulative: bool = False) -> go.Figure:
    chart = figure("IC sum" if cumulative else "Spearman IC")
    traces = (
        [("cumulative_ic", "Cumulative IC", TEAL)]
        if cumulative
        else [("ic", "Daily IC", "#a6b4c8"), ("rolling_ic", "63-session mean", TEAL)]
    )
    for column, label, color in traces:
        chart.add_trace(
            go.Scatter(
                x=daily["date"],
                y=daily[column],
                name=label,
                mode="lines",
                line={"color": color, "width": 2.3 if column != "ic" else 0.7},
                connectgaps=False,
                hovertemplate="%{y:.2f}<extra>%{fullData.name}</extra>",
            )
        )
    chart.add_hline(y=0, line_dash="dot", line_color="#8b9ab1")
    return chart


def ic_histogram(daily: pd.DataFrame) -> go.Figure:
    chart = figure("Sessions")
    chart.add_trace(
        go.Histogram(
            x=daily["ic"].dropna(),
            nbinsx=30,
            marker_color=TEAL,
            name="Daily IC",
            hovertemplate="IC: %{x:.2f}<br>Sessions: %{y}<extra></extra>",
        )
    )
    chart.update_layout(xaxis_title="Daily IC", hovermode="closest", bargap=0.08)
    return chart


def contribution_chart(frame: pd.DataFrame, *, coefficient: bool = False) -> go.Figure:
    label, value = ("feature", "weight") if coefficient else ("signal", "contribution")
    ordered = frame.sort_values(value, kind="stable")
    chart = figure("Coefficient (×10⁻⁴)" if coefficient else "Contribution (bps)")
    chart.add_trace(
        go.Bar(
            x=ordered[label].map(signal_label),
            y=ordered[value] * 10_000,
            customdata=ordered[value],
            marker_color=[TEAL if item >= 0 else RED for item in ordered[value]],
            hovertemplate=(
                "%{x}<br>Coefficient: %{customdata:.2e}<extra></extra>"
                if coefficient
                else "%{x}<br>%{y:.1f} bps<extra></extra>"
            ),
        )
    )
    chart.update_layout(hovermode="closest")
    chart.update_xaxes(tickangle=-25, tickfont_size=11)
    if coefficient:
        chart.update_yaxes(tickformat=".1f")
    return chart


def coverage_chart(coverage: pd.DataFrame) -> go.Figure:
    chart = figure("Sessions")
    coverage = coverage.sort_values(["sessions", "symbol"], kind="stable")
    chart.add_trace(
        go.Bar(
            x=coverage["symbol"],
            y=coverage["sessions"],
            marker_color=TEAL,
            hovertemplate="%{x}: %{y:,} sessions<extra></extra>",
        )
    )
    chart.update_layout(hovermode="closest")
    chart.update_yaxes(tickformat=",d")
    return chart


def missing_chart(missing: pd.DataFrame) -> go.Figure:
    chart = figure("Symbol")
    chart.add_trace(
        go.Heatmap(
            z=missing.to_numpy(),
            x=missing.columns,
            y=missing.index,
            zmin=0,
            zmax=1,
            colorscale=[[0, "#e8f5f3"], [0.01, "#f4dfa8"], [1, RED]],
            colorbar={"title": "Missing", "tickformat": ".0%"},
            hovertemplate="%{y}<br>%{x|%b %Y}<br>Missing: %{z:.1%}<extra></extra>",
        )
    )
    chart.update_layout(hovermode="closest")
    chart.update_xaxes(type="date")
    return chart
