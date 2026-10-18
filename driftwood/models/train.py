"""Fit a walk-forward model on the feature and label tables already stored."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from driftwood.data.store import MarketStore
from driftwood.features.factors import MODEL_FEATURES
from driftwood.features.persist import read_features, read_labels
from driftwood.models.estimators import ColumnBaseline, LightGBMModel
from driftwood.models.persist import write_predictions
from driftwood.models.report import write_ic_report
from driftwood.models.walkforward import (
    EstimatorFactory,
    WalkForwardConfig,
    WalkForwardResult,
    default_ridge_factory,
    walk_forward,
)

_LABEL_HORIZON = {"fwd_ret_1d": 1, "fwd_ret_5d": 5}


@dataclass(frozen=True)
class TrainResult:
    result: WalkForwardResult
    rows_written: int
    report_path: Path | None


def horizon_for_label(label: str) -> int:
    try:
        return _LABEL_HORIZON[label]
    except KeyError as exc:
        known = ", ".join(sorted(_LABEL_HORIZON))
        raise ValueError(f"Unknown label {label!r}. Expected one of: {known}.") from exc


def feature_columns_for(model: str) -> tuple[str, ...]:
    if model == "momentum":
        return ("z_mom_12_1",)
    if model in {"ridge", "lgbm"}:
        return MODEL_FEATURES
    raise ValueError(f"Unknown model {model!r}. Expected ridge, momentum, or lgbm.")


def estimator_factory(model: str, alpha: float = 1.0) -> EstimatorFactory:
    if model == "ridge":
        return default_ridge_factory(alpha)
    if model == "momentum":

        def momentum() -> ColumnBaseline:
            return ColumnBaseline("z_mom_12_1")

        return momentum
    if model == "lgbm":

        def boosted() -> LightGBMModel:
            return LightGBMModel()

        return boosted
    raise ValueError(f"Unknown model {model!r}. Expected ridge, momentum, or lgbm.")


def load_training_frame(store: MarketStore) -> pd.DataFrame:
    """Inner-join features to labels. The label columns are the only forward returns."""

    features = read_features(store)
    labels = read_labels(store)
    if features.empty or labels.empty:
        return features.iloc[0:0]
    return features.merge(labels, on=["symbol", "date"], how="inner", validate="one_to_one")


def train_store(
    store: MarketStore,
    *,
    model: str,
    label: str,
    config: WalkForwardConfig,
    alpha: float = 1.0,
    report_dir: Path | None = None,
    feature_columns: Sequence[str] | None = None,
) -> TrainResult:
    frame = load_training_frame(store)
    if frame.empty:
        raise RuntimeError("Feature or label table is empty. Run `driftwood features` first.")
    columns = tuple(feature_columns) if feature_columns is not None else feature_columns_for(model)
    result = walk_forward_model(frame, columns, label, model, config, alpha)
    rows = write_predictions(
        store,
        result.predictions,
        model_name=result.model_name,
        label_column=label,
    )
    report_path = None
    if report_dir is not None:
        report_path = write_ic_report(result, report_dir, label_column=label)
    return TrainResult(result=result, rows_written=rows, report_path=report_path)


def walk_forward_model(
    frame: pd.DataFrame,
    feature_columns: Sequence[str],
    label: str,
    model: str,
    config: WalkForwardConfig,
    alpha: float = 1.0,
) -> WalkForwardResult:
    return walk_forward(
        frame,
        feature_columns,
        label,
        estimator_factory(model, alpha),
        config,
    )
