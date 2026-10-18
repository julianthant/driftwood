# Dashboard design

A read-only research workbench with six native Streamlit pages: Overview,
Backtest, Execution, Model / Signals, Attribution and Data. Keep labels and
captions short. Put research caveats and accounting explanations in the README.
Source and simulation status belong only in the sidebar.

Use a light canvas (`#f5f7fb`), white panels, navy text (`#172b4d`) and teal
accent (`#087f8c`). Schedule labels are Immediate, TWAP and VWAP. Use Immediate
teal (`#087f8c`, solid), TWAP orange (`#c16319`, dashed) and VWAP slate
(`#334155`, dotted) across charts, including drawdown. Negative signal
contributions use brick red (`#ba4653`). Retain trace labels, subtle grids,
horizontal legends and explicit units. Use system sans-serif typography.

Keep KPI cards equal in height and charts full-width at 300px. Model / Signals
shows daily IC followed by full-width Distribution, Cumulative IC and Coefficients
tabs. Use readable signal names. Show every attribution row plus Total in chart
order. Scale coefficient bars by 10,000 with a ×10⁻⁴ axis and raw scientific hover.
Use compact currency, one decimal for percentages and bps, two for Sharpe and
t-statistics, three for mean IC, grouped counts and dates such as 04 Jan 2016.
Formatting and compact hovers preserve raw data. Show the legacy-label note only
when metadata is absent.

Default to saved backtests, preferring ridge within each availability group;
preserve user selection. Put Database path and Refresh data inside collapsed
Settings below Model and Schedule. Use minimal client toolbar mode and hide
`stToolbar` with CSS while retaining sidebar collapse.

Compare return, Sharpe and max drawdown separately. Retain USD cost bars. Show
equal finite realized cost rates once, using an absolute tolerance of 1e-6 bps
for floating-point summation noise only. Undefined or nonfinite rates do not
collapse; differing rates retain separate bars. Zero traded notional leaves
cost bps missing. Identify zero impact and commission as spread only. Relative
cumulative costs subtract Immediate on the same date, with a labeled zero
reference and no Immediate trace.

Sort coverage by sessions then symbol and show dates without times. Complete
coverage shows "No missing symbol-days"; partial heatmaps show affected symbols.
Streamlit owns responsive stacking. Empty states give CLI recovery commands.

The app reads existing files with bounded memory caches keyed by database and
WAL revision. Simulation stays in the CLI and uses the existing engine.
No trade controls or credential inputs belong here.
