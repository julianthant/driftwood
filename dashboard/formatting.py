"""Shared display labels and units for the research workbench."""

from __future__ import annotations

import math

SCHEDULE_LABELS = {"immediate": "Immediate", "twap": "TWAP", "vwap": "VWAP"}
SIGNAL_LABELS = {
    "z_ret_1d": "1-day return",
    "z_mom_21": "1-month momentum",
    "z_mom_63": "3-month momentum",
    "z_mom_12_1": "12–1 momentum",
    "z_rev_5": "5-day reversal",
    "z_vol_21": "21-day volatility",
    "z_volume_shock": "Volume shock",
    "z_dist_sma_21": "Distance to 21-day SMA",
    "z_hl_range": "High–low range",
    "z_log_volume": "Log volume",
    "unattributed": "Unattributed",
    "total": "Total",
}


def schedule_label(algo: str) -> str:
    return SCHEDULE_LABELS.get(algo, algo)


def signal_label(signal: str) -> str:
    return SIGNAL_LABELS.get(signal, signal)


def compact_money(value: float) -> str:
    """USD with one decimal for thousands, millions and billions; missing stays missing."""
    if not math.isfinite(value):
        return "—"
    sign = "-" if value < 0 else ""
    amount = abs(value)
    for scale, suffix in ((1e9, "b"), (1e6, "m"), (1e3, "k")):
        if amount >= scale:
            return f"{sign}${amount / scale:,.1f}{suffix}"
    return f"{sign}${amount:,.0f}"
