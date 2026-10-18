"""Python-facing entry to the C++ execution simulator.

Nothing in this module decides a fill price. The schedule, the path, the
participation cap, and the cost all live in ``driftwood._sim``. Callers pass
``sigma`` from information they already had. This function does not look at
tomorrow, and it does not open the database.
"""

from __future__ import annotations

from typing import Any

DEFAULT_BINS = 8
DEFAULT_HALF_SPREAD_BPS = 1.0
DEFAULT_PARTICIPATION = 0.1


def execute_order(
    *,
    open: float,
    high: float,
    low: float,
    close: float,
    volume: float,
    quantity: float,
    side: str,
    algo: str,
    sigma: float = 0.0,
    n_bins: int = DEFAULT_BINS,
    commission_per_share: float = 0.0,
    half_spread_bps: float = DEFAULT_HALF_SPREAD_BPS,
    impact_coef: float = 0.0,
    participation_limit: float = DEFAULT_PARTICIPATION,
) -> Any:
    """Fill one order against one daily bar. Returns the C++ ``Fill`` object."""

    simulator = _simulator()
    return simulator.simulate(
        open,
        high,
        low,
        close,
        volume,
        quantity,
        side,
        algo,
        commission_per_share=commission_per_share,
        half_spread_bps=half_spread_bps,
        impact_coef=impact_coef,
        participation_limit=participation_limit,
        sigma=sigma,
        n_bins=n_bins,
    )


def _simulator():
    try:
        import driftwood._sim as simulator
    except ImportError as exc:
        raise ImportError(
            "driftwood._sim is not built. Reinstall with `pip install -e .` "
            "on a machine with a C++17 compiler and CMake."
        ) from exc
    return simulator
