"""Walk-forward return models and information-coefficient reports."""

from driftwood.models.estimators import ColumnBaseline, LightGBMModel, RidgeModel
from driftwood.models.ic import information_coefficient, prediction_regression, summarize_ic
from driftwood.models.walkforward import (
    InsufficientHistory,
    WalkForwardConfig,
    iter_folds,
    walk_forward,
)

__all__ = [
    "ColumnBaseline",
    "InsufficientHistory",
    "LightGBMModel",
    "RidgeModel",
    "WalkForwardConfig",
    "information_coefficient",
    "iter_folds",
    "prediction_regression",
    "summarize_ic",
    "walk_forward",
]
