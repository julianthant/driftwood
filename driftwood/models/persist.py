"""Store out-of-sample predictions in the study database.

The realized label is stored beside the forecast so a report can be rebuilt.
A backtest has to trade the prediction column. The label is the answer, and
it is named that way so it does not look like another feature.
"""

from __future__ import annotations

import pandas as pd

from driftwood.data.store import MarketStore


def write_predictions(
    store: MarketStore,
    predictions: pd.DataFrame,
    *,
    model_name: str,
    label_column: str,
) -> int:
    payload = predictions.loc[:, ["symbol", "date", "prediction", label_column, "fold"]].copy()
    payload = payload.rename(columns={label_column: "label"})
    payload["model"] = model_name
    payload["date"] = pd.to_datetime(payload["date"]).dt.date
    store.execute("BEGIN")
    try:
        store.execute(
            """
            CREATE TABLE IF NOT EXISTS predictions (
                symbol VARCHAR NOT NULL,
                date DATE NOT NULL,
                prediction DOUBLE,
                label DOUBLE,
                fold INTEGER NOT NULL,
                model VARCHAR NOT NULL,
                PRIMARY KEY (symbol, date, model)
            )
            """
        )
        # A second model is a comparison, not a replacement of the first one.
        store.execute("DELETE FROM predictions WHERE model = ?", [model_name])
        if not payload.empty:
            store.register_frame("_incoming_predictions", payload)
            store.execute(
                """
                INSERT INTO predictions (symbol, date, prediction, label, fold, model)
                SELECT symbol, date, prediction, label, fold, model
                FROM _incoming_predictions
                """
            )
        store.execute("COMMIT")
    except Exception:
        store.execute("ROLLBACK")
        raise
    finally:
        store.unregister_frame("_incoming_predictions")
    return int(len(payload))


def write_coefficients(
    store: MarketStore,
    weights: pd.DataFrame,
    *,
    model_name: str,
) -> int:
    """Replace one model's fold coefficients. Other models stay in the table."""

    payload = weights.loc[:, ["fold", "feature", "weight"]].copy()
    payload["model"] = model_name
    payload["fold"] = payload["fold"].astype(int)
    payload["feature"] = payload["feature"].astype(str)
    payload["weight"] = payload["weight"].astype(float)
    store.execute("BEGIN")
    try:
        store.execute(
            """
            CREATE TABLE IF NOT EXISTS coefficients (
                model VARCHAR NOT NULL,
                fold INTEGER NOT NULL,
                feature VARCHAR NOT NULL,
                weight DOUBLE,
                PRIMARY KEY (model, fold, feature)
            )
            """
        )
        store.execute("DELETE FROM coefficients WHERE model = ?", [model_name])
        if not payload.empty:
            store.register_frame("_incoming_coefficients", payload)
            store.execute(
                """
                INSERT INTO coefficients (model, fold, feature, weight)
                SELECT model, fold, feature, weight
                FROM _incoming_coefficients
                """
            )
        store.execute("COMMIT")
    except Exception:
        store.execute("ROLLBACK")
        raise
    finally:
        store.unregister_frame("_incoming_coefficients")
    return int(len(payload))


def read_coefficients(store: MarketStore, model: str | None = None) -> pd.DataFrame:
    try:
        frame = store.execute(
            """
            SELECT model, fold, feature, weight
            FROM coefficients
            ORDER BY model, fold, feature
            """
        ).df()
    except Exception as exc:
        message = str(exc).lower()
        if "does not exist" in message or "not found" in message:
            raise RuntimeError("No coefficients table. Run `driftwood train` first.") from exc
        raise
    if model is not None and not frame.empty:
        frame = frame.loc[frame["model"] == model].reset_index(drop=True)
    return frame


def read_predictions(store: MarketStore) -> pd.DataFrame:
    try:
        frame = store.execute(
            """
            SELECT symbol, date, prediction, label, fold, model
            FROM predictions
            ORDER BY date, symbol
            """
        ).df()
    except Exception as exc:
        message = str(exc).lower()
        if "does not exist" in message or "not found" in message:
            raise RuntimeError("No predictions table. Run `driftwood train` first.") from exc
        raise
    if not frame.empty:
        frame["date"] = pd.to_datetime(frame["date"])
    return frame
