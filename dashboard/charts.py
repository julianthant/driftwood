"""Plotly figures with shared research colors and explicit units."""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go

COLORS = {"immediate": "#087f8c", "twap": "#3f63b5", "vwap": "#996214"}
TEAL = "#087f8c"
RED = "#ba4653"


def figure(y_title: str, *, height: int = 330) -> go.Figure:
    chart = go.Figure()
    chart.update_layout(
        template="plotly_white",
        height=height,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="#ffffff",
        font={"family": "Arial, sans-serif", "size": 13, "color": "#172b4d"},
        margin={"l": 18, "r": 18, "t": 35, "b": 20},
        legend={"orientation": "h", "y": 1.12, "x": 0},
        hovermode="x unified",
        yaxis_title=y_title,
        colorway=list(COLORS.values()),
    )
    chart.update_xaxes(showgrid=False, zeroline=False)
    chart.update_yaxes(gridcolor="#e8edf4", zerolinecolor="#c4cedd")
    return chart


def equity_chart(daily: pd.DataFrame) -> go.Figure:
    chart = figure("Growth of $1", height=380)
    for algo, frame in daily.groupby("algo", sort=True):
        chart.add_trace(
            go.Scatter(
                x=frame["date"],
                y=frame["growth"],
                name=str(algo).upper(),
                mode="lines",
                line={"color": COLORS.get(str(algo), TEAL), "width": 2},
                hovertemplate="%{y:.4f}×<extra>%{fullData.name}</extra>",
            )
        )
    chart.add_hline(y=1, line_dash="dot", line_color="#8b9ab1")
    return chart


def drawdown_chart(daily: pd.DataFrame) -> go.Figure:
    chart = figure("Drawdown")
    chart.add_trace(
        go.Scatter(
            x=daily["date"],
            y=daily["drawdown"],
            name="Drawdown",
            mode="lines",
            fill="tozeroy",
            line={"color": RED, "width": 1.5},
            fillcolor="rgba(186,70,83,0.14)",
            hovertemplate="%{y:.2%}<extra></extra>",
        )
    )
    chart.update_yaxes(tickformat=".0%")
    return chart


def cost_chart(daily: pd.DataFrame) -> go.Figure:
    chart = figure("Cumulative cost (USD)")
    for algo, frame in daily.groupby("algo", sort=True):
        chart.add_trace(
            go.Scatter(
                x=frame["date"],
                y=frame["cumulative_cost"],
                name=str(algo).upper(),
                mode="lines",
                line={"color": COLORS.get(str(algo), TEAL), "width": 2},
                hovertemplate="$%{y:,.2f}<extra>%{fullData.name}</extra>",
            )
        )
    chart.update_yaxes(tickprefix="$", tickformat=",")
    return chart


def ic_chart(daily: pd.DataFrame, *, cumulative: bool = False) -> go.Figure:
    chart = figure("Cumulative IC (sum)" if cumulative else "Spearman IC")
    traces = (
        [("cumulative_ic", "Cumulative IC", TEAL)]
        if cumulative
        else [("ic", "Daily IC", "#7185a3"), ("rolling_ic", "63-session mean", TEAL)]
    )
    for column, label, color in traces:
        chart.add_trace(
            go.Scatter(
                x=daily["date"],
                y=daily[column],
                name=label,
                mode="lines",
                line={"color": color, "width": 2 if column != "ic" else 1},
                connectgaps=False,
                hovertemplate="%{y:.3f}<extra>%{fullData.name}</extra>",
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
    chart.update_layout(xaxis_title="Daily Spearman IC", hovermode="closest", bargap=0.08)
    return chart


def contribution_chart(frame: pd.DataFrame, *, coefficient: bool = False) -> go.Figure:
    label, value = ("feature", "weight") if coefficient else ("signal", "contribution")
    ordered = frame.sort_values(value)
    chart = figure("Coefficient" if coefficient else "Contribution (bps)")
    chart.add_trace(
        go.Bar(
            x=ordered[label],
            y=ordered[value] * (1 if coefficient else 10_000),
            marker_color=[TEAL if item >= 0 else RED for item in ordered[value]],
            hovertemplate="%{x}<br>%{y:.3f}<extra></extra>",
        )
    )
    chart.update_layout(hovermode="closest")
    return chart


def coverage_chart(coverage: pd.DataFrame) -> go.Figure:
    chart = figure("Observed sessions")
    chart.add_trace(
        go.Bar(
            x=coverage["symbol"],
            y=coverage["sessions"],
            marker_color=TEAL,
            hovertemplate="%{x}: %{y:,} sessions<extra></extra>",
        )
    )
    chart.update_layout(hovermode="closest")
    return chart


def missing_chart(missing: pd.DataFrame) -> go.Figure:
    chart = figure("Symbol", height=max(300, min(850, 150 + len(missing) * 17)))
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
