# Dashboard design

A research workbench for reviewing signal quality and the costs of execution.
Use Streamlit's native sidebar, controls, metrics and tables. Navigation separates
Overview, Backtest, Execution, Model / Signals, Attribution and Data, so only the
selected analysis loads.

Use a light canvas (`#f5f7fb`), white panels, navy text (`#172b4d`) and teal
accent (`#087f8c`). Schedule labels are Immediate, TWAP and VWAP. Use Immediate
teal (`#087f8c`, solid), TWAP blue (`#3564b8`, dashed) and VWAP purple
(`#9b3f8b`, dotted), including drawdown. Retain trace labels. Negative signal contributions use brick red
(`#ba4653`). Use system sans-serif typography and tabular numerals for metrics.
Keep chart grids subtle, legends horizontal, titles descriptive and units explicit.
No external font, image or animation is required. Use viewer toolbar mode.

Default to models with saved backtests, preferring ridge within each availability
group. Preserve the user's selection. Compare return, Sharpe and max drawdown in
separate bar panels. Show fill costs in USD and basis points of actual traded
notional. Cumulative cost differences subtract Immediate on the same date, with
Immediate labeled as the zero reference. Zero traded notional leaves cost bps
missing. Keep dollar precision in data and hover; compact large currency KPIs.

Use readable signal names. Match the attribution table's order to its chart,
show all rows plus Total, and format contributions to two decimal places in bps.
Scale coefficient bars by 10,000 with a ×10⁻⁴ axis label and raw scientific hover.
Show the legacy-label allowance only when metadata is absent. Sort coverage by
sessions then symbol, use a 250px bar chart and date-only table columns. Complete
coverage shows "No missing symbol-days"; partial heatmaps show affected symbols.

The first viewport pairs study coverage with signal and execution metrics and a
large normalized equity comparison. Captions explain the accounting and limits.
Charts stay interactive, and detailed tables remain available. On narrow screens,
Streamlit stacks columns and collapses the sidebar. Empty states give the next CLI
command; results are never fabricated. Synthetic sources are labeled.

The app only reads existing files. Heavy transforms use bounded memory caches,
keyed by the database and WAL revision. Simulation runs in the CLI and uses the
existing engine. No trade controls or credential inputs belong here.
