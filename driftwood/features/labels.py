"""Forward returns. This is the only feature-side module allowed to look ahead.

The label on day ``t`` is the return a holder earns after the close of day
``t``. It is a target, not an input. Keeping it out of ``factors.py`` makes
the boundary something a test can check by reading the source.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from driftwood.features.factors import LABEL_COLUMNS
from driftwood.features.panel import prepare_panel


def build_labels(bars: pd.DataFrame) -> pd.DataFrame:
    """Next-session and five-session simple returns of adjusted close.

    ``fwd_ret_1d`` on date ``t`` uses the adjusted close on the next stored
    session for that symbol, which is ``t+1`` when the panel has no gaps.
    The last rows of each name are missing, because the future is not known
    there. That missingness is the point.
    """

    work = prepare_panel(bars, ["symbol", "date", "adj_close"])
    out = work.loc[:, ["symbol", "date"]].copy()
    if work.empty:
        for name in LABEL_COLUMNS:
            out[name] = pd.Series(dtype="float64")
        return out

    grouped = work.groupby("symbol", sort=False)["adj_close"]
    out["fwd_ret_1d"] = _finite(grouped.shift(-1) / work["adj_close"] - 1.0)
    out["fwd_ret_5d"] = _finite(grouped.shift(-5) / work["adj_close"] - 1.0)
    return out.loc[:, ["symbol", "date", *LABEL_COLUMNS]].reset_index(drop=True)


def _finite(series: pd.Series) -> pd.Series:
    return series.replace([np.inf, -np.inf], np.nan)
