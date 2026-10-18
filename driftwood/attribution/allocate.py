"""Exact split of weight times return across signal exposures.

Each name's contribution ``w_i * r_i`` is handed to its signals in proportion
to ``u_{i,k} / sum_k u_{i,k}``. A name with weight and no exposure total goes
to ``unattributed``. The pieces sum back to ``sum(w * r)``. This is an
accounting identity, not a regression.
"""

from __future__ import annotations

import pandas as pd

_DENOM_FLOOR = 1e-12


def attribute_return(
    weights: pd.Series,
    returns: pd.Series,
    exposures: pd.DataFrame,
) -> pd.DataFrame:
    """Return one row per signal plus ``unattributed``.

    ``weights`` and ``returns`` share a symbol index. ``exposures`` has a
    symbol index and one column per signal. Missing exposures are treated as
    zero. A weight without a return is an error, because the identity would
    otherwise drop part of the book without saying so.
    """

    book = pd.Series(weights.to_numpy(dtype=float), index=weights.index.astype(str))
    realized = pd.Series(returns.to_numpy(dtype=float), index=returns.index.astype(str))
    realized = realized.reindex(book.index)
    if realized.isna().any():
        missing = realized.index[realized.isna()].tolist()
        raise ValueError(f"returns are missing for {missing}")

    signals = list(exposures.columns) if exposures is not None else []
    aligned = exposures.reindex(book.index) if signals else pd.DataFrame(index=book.index)
    if signals:
        aligned = aligned.astype(float).fillna(0.0)

    contribution = book.to_numpy(dtype=float) * realized.to_numpy(dtype=float)
    totals = {str(name): 0.0 for name in signals}
    unattributed = 0.0
    matrix = aligned.to_numpy(dtype=float) if signals else None
    for index, name_contribution in enumerate(contribution):
        weight = float(book.iloc[index])
        if matrix is None:
            if weight != 0.0:
                unattributed += float(name_contribution)
            continue
        denominator = float(matrix[index].sum())
        if abs(denominator) <= _DENOM_FLOOR:
            if weight != 0.0:
                unattributed += float(name_contribution)
            continue
        for column, name in enumerate(signals):
            share = float(matrix[index, column]) / denominator
            totals[str(name)] += float(name_contribution) * share

    rows = [{"signal": name, "contribution": value} for name, value in totals.items()]
    rows.append({"signal": "unattributed", "contribution": unattributed})
    return pd.DataFrame(rows)
