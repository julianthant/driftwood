"""Dollar-neutral weights from one day's scores.

The book is long the names above the day's average score and short the names
below it. Weights are scaled to a gross target, clipped once, and then the
unused room under the cap is filled once. There is no second pass. If the cap
binds, the gross can finish below the target and the net can finish away from
zero. That is the tradeoff of refusing to iterate the cap away.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def construct_weights(
    scores: pd.Series,
    *,
    gross: float = 1.0,
    max_weight: float = 0.1,
) -> pd.Series:
    """Map one cross-section to weights. Missing scores are dropped.

    The result sums to about zero before the cap. After the cap it may not.
    ``gross`` is the sum of absolute weights the book is aiming for. A single
    name can be at most ``max_weight`` in absolute value.
    """

    if gross < 0:
        raise ValueError("gross cannot be negative")
    if max_weight <= 0:
        raise ValueError("max_weight must be positive")
    clean = scores.dropna().astype(float)
    if clean.empty or gross == 0:
        return pd.Series(0.0, index=clean.index, dtype=float)

    centered = clean - float(clean.mean())
    denominator = float(centered.abs().sum())
    if denominator == 0.0:
        return pd.Series(0.0, index=clean.index, dtype=float)

    raw = centered / denominator * gross
    capped = raw.clip(lower=-max_weight, upper=max_weight)
    shortfall = gross - float(capped.abs().sum())
    if shortfall <= 1e-12:
        return capped

    room = (max_weight - capped.abs()).clip(lower=0.0)
    room = room.where(capped != 0.0, 0.0)
    room_sum = float(room.sum())
    if room_sum <= 0.0:
        return capped
    extra = room / room_sum * min(shortfall, room_sum)
    return capped + np.sign(capped) * extra
