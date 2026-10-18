"""Join features and labels without letting the label leak back into a feature."""

from __future__ import annotations

import pandas as pd

from driftwood.features.factors import FEATURE_COLUMNS, LABEL_COLUMNS, build_features
from driftwood.features.labels import build_labels


def build_research_frame(bars: pd.DataFrame) -> pd.DataFrame:
    """One row per symbol and date, with inputs and targets in named columns.

    The merge is one-to-one on the key. Training code is expected to pass
    ``MODEL_FEATURES`` to a model and ``LABEL_COLUMNS`` to the target, rather
    than every numeric column in this frame.
    """

    features = build_features(bars)
    labels = build_labels(bars)
    frame = features.merge(labels, on=["symbol", "date"], how="left", validate="one_to_one")
    ordered = ["symbol", "date", *FEATURE_COLUMNS, *LABEL_COLUMNS]
    return frame.loc[:, ordered]
