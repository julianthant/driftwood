"""Attribute the last scored session that already has a next bar.

The return is the next adjusted close over the signal adjusted close, minus
one. Weights are the same dollar-neutral book the backtest would have
targeted. Exposures are the latest fold's coefficient times that day's
z-score. When no coefficients were stored, each z-score is used with a beta
of one, and the note says so.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from driftwood.attribution.allocate import attribute_return
from driftwood.data.store import MarketStore
from driftwood.features.factors import MODEL_FEATURES
from driftwood.features.persist import read_features
from driftwood.models.persist import read_coefficients, read_predictions
from driftwood.portfolio.construct import construct_weights


@dataclass(frozen=True)
class Attribution:
    signal_date: pd.Timestamp
    next_date: pd.Timestamp
    by_signal: pd.DataFrame
    portfolio_return: float
    note: str
    model: str


def attribute_store(
    store: MarketStore,
    *,
    model: str,
    gross: float = 1.0,
    max_weight: float = 0.1,
    min_names: int = 2,
) -> Attribution:
    bars = store.read_bars()
    if bars.empty:
        raise RuntimeError("no bars in the database; run driftwood ingest first")
    try:
        predictions = read_predictions(store)
    except RuntimeError as exc:
        raise RuntimeError("no predictions in the database; run driftwood train first") from exc
    signal_date, next_date = _window(predictions, bars, model)
    scores, signal_prices = _cross_section(predictions, bars, model, signal_date)
    next_prices = _prices_on(bars, next_date)
    shared = scores.dropna().index.intersection(signal_prices.index).intersection(next_prices.index)
    usable = []
    for symbol in shared:
        start = float(signal_prices[symbol])
        end = float(next_prices[symbol])
        if start > 0 and end > 0:
            usable.append(symbol)
    scores = scores.loc[usable].dropna()
    if int(scores.shape[0]) < min_names:
        raise RuntimeError("not enough names with a score and a next bar to attribute")
    weights = construct_weights(scores, gross=gross, max_weight=max_weight)
    returns = next_prices.loc[weights.index] / signal_prices.loc[weights.index] - 1.0
    exposures, note = _exposures(store, model, signal_date, weights.index)
    by_signal = attribute_return(weights, returns, exposures)
    portfolio_return = float((weights * returns).sum())
    return Attribution(
        signal_date=signal_date,
        next_date=next_date,
        by_signal=by_signal,
        portfolio_return=portfolio_return,
        note=note,
        model=model,
    )


def _window(
    predictions: pd.DataFrame,
    bars: pd.DataFrame,
    model: str,
) -> tuple[pd.Timestamp, pd.Timestamp]:
    chosen = predictions.loc[predictions["model"] == model]
    if chosen.empty:
        raise RuntimeError(f"no predictions for model {model}")
    pred_dates = sorted(_normalize(chosen["date"]).unique())
    bar_dates = sorted(_normalize(bars["date"]).unique())
    signal = None
    for stamp in pred_dates:
        if any(bar > stamp for bar in bar_dates):
            signal = pd.Timestamp(stamp)
    if signal is None:
        raise RuntimeError("no prediction date has a later bar")
    next_date = pd.Timestamp(next(bar for bar in bar_dates if bar > signal))
    return signal, next_date


def _cross_section(
    predictions: pd.DataFrame,
    bars: pd.DataFrame,
    model: str,
    signal_date: pd.Timestamp,
) -> tuple[pd.Series, pd.Series]:
    chosen = predictions.loc[predictions["model"] == model].copy()
    chosen["date"] = _normalize(chosen["date"])
    day = chosen.loc[chosen["date"] == signal_date]
    scores = pd.Series(day["prediction"].to_numpy(dtype=float), index=day["symbol"].astype(str))
    return scores, _prices_on(bars, signal_date)


def _prices_on(bars: pd.DataFrame, day: pd.Timestamp) -> pd.Series:
    frame = bars.copy()
    frame["date"] = _normalize(frame["date"])
    rows = frame.loc[frame["date"] == day].drop_duplicates("symbol")
    return pd.Series(rows["adj_close"].to_numpy(dtype=float), index=rows["symbol"].astype(str))


def _exposures(
    store: MarketStore,
    model: str,
    signal_date: pd.Timestamp,
    symbols: pd.Index,
) -> tuple[pd.DataFrame, str]:
    betas, note = _betas(store, model)
    try:
        features = read_features(store)
    except RuntimeError:
        features = pd.DataFrame()
    if features.empty:
        blank = pd.DataFrame(0.0, index=symbols, columns=list(betas.index))
        return blank, note + " Feature rows are missing, so the return is unattributed."
    features = features.copy()
    features["date"] = _normalize(features["date"])
    day = features.loc[features["date"] == signal_date]
    if day.empty:
        blank = pd.DataFrame(0.0, index=symbols, columns=list(betas.index))
        return blank, note + " No feature row on the signal date, so the return is unattributed."
    present = [column for column in betas.index if column in day.columns]
    if not present:
        blank = pd.DataFrame(0.0, index=symbols, columns=list(betas.index))
        return blank, note + " None of the coefficient columns are in the feature table."
    panel = day.drop_duplicates("symbol").copy()
    panel.index = panel["symbol"].astype(str)
    panel = panel.reindex(symbols.astype(str))
    exposures = panel.loc[:, present].astype(float).fillna(0.0)
    exposures = exposures.mul(betas.loc[present].astype(float), axis=1)
    return exposures, note


def _betas(store: MarketStore, model: str) -> tuple[pd.Series, str]:
    try:
        stored = read_coefficients(store, model)
    except RuntimeError:
        stored = pd.DataFrame()
    if stored.empty:
        betas = pd.Series(1.0, index=list(MODEL_FEATURES), dtype=float)
        note = (
            f"No coefficients for {model}. Using a beta of 1 on each z-score. "
            "Names whose exposures sum to zero are unattributed."
        )
        return betas, note
    latest = int(stored["fold"].max())
    chosen = stored.loc[stored["fold"] == latest]
    betas = pd.Series(
        chosen["weight"].to_numpy(dtype=float),
        index=chosen["feature"].astype(str),
        dtype=float,
    )
    return betas, f"Exposures use fold {latest} coefficients times the signal-date z-score."


def _normalize(values: pd.Series) -> pd.Series:
    dates = pd.to_datetime(values)
    if isinstance(dates.dtype, pd.DatetimeTZDtype):
        dates = dates.dt.tz_convert("UTC").dt.tz_localize(None)
    return dates.dt.normalize()
