# Dashboard design

A research workbench for reviewing signal quality and the costs of execution.
Use Streamlit's native sidebar, controls, metrics and tables. Navigation separates
Overview, Backtest, Execution, Model / Signals, Attribution and Data, so only the
selected analysis loads.

Use a light canvas (`#f5f7fb`), white panels, navy text (`#172b4d`) and teal
accent (`#087f8c`). Chart schedules are immediate teal, TWAP blue (`#3f63b5`),
and VWAP amber (`#996214`); always retain trace labels. Losses use brick red
(`#ba4653`). Use system sans-serif typography and tabular numerals for metrics.
Keep chart grids subtle, legends horizontal, titles descriptive and units explicit.
No external font, image or animation is required.

The first viewport pairs study coverage with signal and execution metrics and a
large normalized equity comparison. Captions explain the accounting and limits.
Charts stay interactive, and detailed tables remain available. On narrow screens,
Streamlit stacks columns and collapses the sidebar. Empty states give the next CLI
command; results are never fabricated. Synthetic sources are labeled.

The app only reads existing files. Heavy transforms use bounded memory caches,
keyed by the database and WAL revision. Simulation runs in the CLI and uses the
existing engine. No trade controls or credential inputs belong here.
