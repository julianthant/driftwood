"""Expanding or rolling walk-forward with a purge before each test window.

The purge is the embargo the caller asked for, and at least the label
horizon. A one-day forward return on the session before the test uses the
first test session's close. That row has to stay out of the fit. Five purged
sessions is the default so a five-day label is covered too.

The purge is counted on the shared calendar. A name that skips a session can
still step over it, because its label is the next stored print. Each fit
drops a training row when that name's own label ends on or after the first
test session, and only then demeans the label inside the day.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd

from driftwood.models.estimators import RidgeModel
from driftwood.models.ic import information_coefficient

EstimatorFactory = Callable[[], object]


class InsufficientHistory(ValueError):
    """The panel is shorter than the training window plus the purge."""


@dataclass(frozen=True)
class WalkForwardConfig:
    min_train_days: int = 252
    test_days: int = 63
    embargo_days: int = 5
    label_horizon: int = 1
    expanding: bool = True
    train_days: int = 756
    demean_label: bool = True
    min_train_rows: int = 20

    def __post_init__(self) -> None:
        if self.min_train_days < 1:
            raise ValueError("min_train_days must be positive")
        if self.test_days < 1:
            raise ValueError("test_days must be positive")
        if self.embargo_days < 0:
            raise ValueError("embargo_days cannot be negative")
        if self.label_horizon < 1:
            raise ValueError("label_horizon must be at least 1")
        if not self.expanding and self.train_days < self.min_train_days:
            raise ValueError("train_days must cover min_train_days")

    @property
    def purge(self) -> int:
        """Sessions removed between the last training date and the first test date."""

        return max(self.embargo_days, self.label_horizon)


@dataclass(frozen=True)
class Fold:
    train_dates: tuple[pd.Timestamp, ...]
    test_dates: tuple[pd.Timestamp, ...]
    embargo_dates: tuple[pd.Timestamp, ...]


@dataclass
class WalkForwardResult:
    predictions: pd.DataFrame
    folds: pd.DataFrame
    weights: pd.DataFrame
    model_name: str
    in_sample_mean_ic: float
    config: WalkForwardConfig


def iter_folds(dates: Sequence[object], config: WalkForwardConfig) -> Iterator[Fold]:
    """Yield train, embargo, and test dates. Dates are unique sessions, sorted."""

    calendar = _calendar(dates)
    purge = config.purge
    cursor = config.min_train_days + purge
    while cursor < len(calendar):
        test_stop = min(cursor + config.test_days, len(calendar))
        train_stop = cursor - purge
        if config.expanding:
            train_start = 0
        else:
            train_start = max(0, train_stop - config.train_days)
        train = calendar[train_start:train_stop]
        if len(train) < config.min_train_days:
            break
        yield Fold(
            train_dates=tuple(train),
            test_dates=tuple(calendar[cursor:test_stop]),
            embargo_dates=tuple(calendar[train_stop:cursor]),
        )
        if test_stop == len(calendar):
            break
        cursor = test_stop


def walk_forward(
    frame: pd.DataFrame,
    feature_columns: Sequence[str],
    label_column: str,
    estimator_factory: EstimatorFactory,
    config: WalkForwardConfig | None = None,
) -> WalkForwardResult:
    """Fit a fresh model on each training window and predict the following test window.

    Predictions are out of sample. In-sample fit quality is reduced to one
    number, ``in_sample_mean_ic``, so those rows cannot be mistaken for the
    forecast the backtest should trade.
    """

    config = config or WalkForwardConfig()
    panel = _prepare(frame, feature_columns, label_column, config)
    folds = list(iter_folds(panel["date"].unique(), config))
    if not folds:
        raise InsufficientHistory(
            "No walk-forward folds. Lengthen the history or shorten min_train_days."
        )

    predictions: list[pd.DataFrame] = []
    fold_rows: list[dict[str, object]] = []
    weight_rows: list[pd.DataFrame] = []
    insample_means: list[float] = []
    model_name = "model"

    for fold_id, fold in enumerate(folds):
        train = _training_rows(panel, fold, label_column, config)
        usable = [column for column in feature_columns if train[column].notna().any()]
        fit_rows = train.dropna(subset=[*usable, "_target"])
        if not usable or len(fit_rows) < config.min_train_rows:
            continue
        model = estimator_factory()
        model_name = str(getattr(model, "name", model_name))
        model.fit(fit_rows[usable], fit_rows["_target"])
        insample = fit_rows.loc[:, ["symbol", "date", label_column]].copy()
        insample["prediction"] = model.predict(fit_rows[usable])
        insample_ic = information_coefficient(insample, "prediction", label_column)
        observed = insample_ic["ic"].dropna()
        if len(observed):
            insample_means.append(float(observed.mean()))

        test = panel.loc[panel["date"].isin(fold.test_dates)].dropna(subset=usable)
        if test.empty:
            continue
        out = test.loc[:, ["symbol", "date", label_column]].copy()
        out["prediction"] = model.predict(test[usable])
        out["fold"] = fold_id
        predictions.append(out)
        fold_rows.append(
            {
                "fold": fold_id,
                "train_start": fold.train_dates[0],
                "train_end": fold.train_dates[-1],
                "test_start": fold.test_dates[0],
                "test_end": fold.test_dates[-1],
                "embargo": len(fold.embargo_dates),
                "n_train": int(len(fit_rows)),
                "n_test": int(len(test)),
                "n_features": len(usable),
            }
        )
        weights = model.weights()
        weight_rows.append(
            pd.DataFrame(
                {
                    "fold": fold_id,
                    "feature": weights.index.astype(str),
                    "weight": weights.to_numpy(dtype=float),
                }
            )
        )

    if not predictions:
        raise InsufficientHistory("Every fold was empty after dropping missing features or labels.")

    return WalkForwardResult(
        predictions=pd.concat(predictions, ignore_index=True),
        folds=pd.DataFrame(fold_rows),
        weights=pd.concat(weight_rows, ignore_index=True),
        model_name=model_name,
        in_sample_mean_ic=float(np.mean(insample_means)) if insample_means else float("nan"),
        config=config,
    )


def default_ridge_factory(alpha: float = 1.0) -> EstimatorFactory:
    def factory() -> RidgeModel:
        return RidgeModel(alpha=alpha)

    return factory


def _prepare(
    frame: pd.DataFrame,
    feature_columns: Sequence[str],
    label_column: str,
    config: WalkForwardConfig,
) -> pd.DataFrame:
    required = ["symbol", "date", label_column, *feature_columns]
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise ValueError(f"Research frame is missing columns: {missing}")
    panel = frame.loc[:, required].copy()
    panel["date"] = _normalize_dates(panel["date"])
    return _attach_label_end(panel, config.label_horizon)


def _attach_label_end(panel: pd.DataFrame, horizon: int) -> pd.DataFrame:
    """Date of the print ``horizon`` sessions later in that name's own history.

    This matches ``build_labels`` when the frame has one row per stored
    session. A missing session makes the end date later than the shared
    calendar would suggest.
    """

    out = panel.reset_index(drop=True)
    ordered = out.sort_values(["symbol", "date"], kind="mergesort")
    ends = ordered.groupby("symbol", sort=False)["date"].shift(-horizon)
    out["_label_end"] = ends.sort_index()
    return out


def _training_rows(
    panel: pd.DataFrame,
    fold: Fold,
    label_column: str,
    config: WalkForwardConfig,
) -> pd.DataFrame:
    """Training rows whose labels finish before the test window.

    The cross-sectional mean is taken after the drop. A name whose label
    reaches into the test must not move the target of the names that stay.
    """

    train = panel.loc[panel["date"].isin(fold.train_dates)].copy()
    test_start = fold.test_dates[0]
    finished_before_test = train["_label_end"].isna() | (train["_label_end"] < test_start)
    train = train.loc[finished_before_test]
    if config.demean_label:
        daily_mean = train.groupby("date")[label_column].transform("mean")
        train["_target"] = train[label_column] - daily_mean
    else:
        train["_target"] = train[label_column]
    return train


def _calendar(dates: Sequence[object]) -> list[pd.Timestamp]:
    index = pd.DatetimeIndex(pd.to_datetime(list(dates)))
    if index.tz is not None:
        index = index.tz_convert("UTC").tz_localize(None)
    index = index.normalize().unique().sort_values()
    return list(index)


def _normalize_dates(values: pd.Series) -> pd.Series:
    dates = pd.to_datetime(values)
    if isinstance(dates.dtype, pd.DatetimeTZDtype):
        dates = dates.dt.tz_convert("UTC").dt.tz_localize(None)
    return dates.dt.normalize()
