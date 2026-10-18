"""Small return models.

Ridge is the default because a coefficient can be read out loud. The momentum
baseline predicts the 12-minus-1 z-score and does not fit anything. LightGBM
is the nonlinear check, with deterministic settings so a test can rerun it.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.linear_model import Ridge


class RidgeModel:
    """L2-regularized linear model on columns that are already cross-sectionally scaled."""

    name = "ridge"

    def __init__(self, alpha: float = 1.0) -> None:
        if alpha < 0:
            raise ValueError("Ridge alpha must be non-negative")
        self.alpha = alpha
        self._model = Ridge(alpha=alpha)
        self.columns_: list[str] = []

    def fit(self, frame: pd.DataFrame, target: pd.Series) -> RidgeModel:
        self.columns_ = list(frame.columns)
        self._model.fit(_matrix(frame), np.asarray(target, dtype=float))
        return self

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        return np.asarray(self._model.predict(_matrix(frame[self.columns_])), dtype=float)

    def weights(self) -> pd.Series:
        return pd.Series(self._model.coef_, index=self.columns_, dtype=float)


class ColumnBaseline:
    """Use one feature as the forecast. Fitting ignores the target on purpose."""

    name = "momentum"

    def __init__(self, column: str = "z_mom_12_1") -> None:
        self.column = column
        self.columns_: list[str] = [column]

    def fit(self, frame: pd.DataFrame, target: pd.Series) -> ColumnBaseline:
        del target
        if self.column not in frame.columns:
            raise ValueError(f"Momentum baseline needs a {self.column} column")
        self.columns_ = [self.column]
        return self

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        return frame[self.column].to_numpy(dtype=float)

    def weights(self) -> pd.Series:
        return pd.Series({self.column: 1.0})


class LightGBMModel:
    """A small gradient-boosted tree with no subsampling, so the fit is repeatable."""

    name = "lgbm"

    def __init__(self, n_estimators: int = 40, learning_rate: float = 0.05) -> None:
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.columns_: list[str] = []
        self._model: LGBMRegressor | None = None

    def fit(self, frame: pd.DataFrame, target: pd.Series) -> LightGBMModel:
        self.columns_ = list(frame.columns)
        self._model = LGBMRegressor(
            n_estimators=self.n_estimators,
            learning_rate=self.learning_rate,
            num_leaves=8,
            subsample=1.0,
            colsample_bytree=1.0,
            random_state=0,
            verbosity=-1,
            n_jobs=1,
        )
        self._model.fit(_matrix(frame), np.asarray(target, dtype=float))
        return self

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        if self._model is None:
            raise RuntimeError("LightGBM model has not been fit")
        return np.asarray(self._model.predict(_matrix(frame[self.columns_])), dtype=float)

    def weights(self) -> pd.Series:
        if self._model is None:
            raise RuntimeError("LightGBM model has not been fit")
        return pd.Series(self._model.feature_importances_, index=self.columns_, dtype=float)


def _matrix(frame: pd.DataFrame) -> np.ndarray:
    return frame.to_numpy(dtype=float)
