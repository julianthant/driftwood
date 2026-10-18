"""Walk-forward return models and information-coefficient reports."""

from driftwood.models.estimators import ColumnBaseline, LightGBMModel, RidgeModel
from driftwood.models.ic import information_coefficient, prediction_regression, summarize_ic
from driftwood.models.persist import read_predictions, write_predictions
from driftwood.models.report import write_ic_report
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
    "read_predictions",
    "summarize_ic",
    "walk_forward",
    "write_ic_report",
    "write_predictions",
]
