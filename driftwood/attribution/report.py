"""Write the attribution note. The identity is the point of the page."""

from __future__ import annotations

from pathlib import Path

from driftwood.attribution.study import Attribution


def write_attribution_report(result: Attribution, directory: str | Path) -> Path:
    folder = Path(directory)
    folder.mkdir(parents=True, exist_ok=True)
    frame = result.by_signal.copy()
    total = float(frame["contribution"].sum())
    frame.to_csv(folder / "attribution.csv", index=False)
    lines = [
        "# Attribution",
        "",
        f"Model: {result.model}",
        f"Signal date: {result.signal_date.date()}",
        f"Next session: {result.next_date.date()}",
        "",
        "Contributions sum to the target book's close-to-close return, "
        f"{total:.6f} (the summed weight times next-session adjusted return is "
        f"{result.portfolio_return:.6f}).",
        "They are not the filled equity-curve return. Costs and the intraday "
        "fill stay in the backtest report.",
        "",
        result.note,
        "",
        "| signal | contribution |",
        "| --- | --- |",
    ]
    for row in frame.itertuples(index=False):
        lines.append(f"| {row.signal} | {row.contribution:.6f} |")
    lines.append(f"| total | {total:.6f} |")
    lines.append("")
    path = folder / "attribution.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path
