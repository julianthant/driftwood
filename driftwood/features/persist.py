"""Store features and labels in the study database, in separate tables.

The feature table has no forward-return column. A ``SELECT *`` from it cannot
become a training set that includes the answer.
"""

from __future__ import annotations

import pandas as pd

from driftwood.data.store import MarketStore
from driftwood.features.factors import FEATURE_COLUMNS, LABEL_COLUMNS

_FEATURE_TABLE = "features"
_LABEL_TABLE = "labels"


def write_research_tables(store: MarketStore, frame: pd.DataFrame) -> tuple[int, int]:
    """Rebuild both tables from a research frame. Returns row counts."""

    features = frame.loc[:, ["symbol", "date", *FEATURE_COLUMNS]].copy()
    labels = frame.loc[:, ["symbol", "date", *LABEL_COLUMNS]].copy()
    _replace_table(store, _FEATURE_TABLE, features, FEATURE_COLUMNS)
    _replace_table(store, _LABEL_TABLE, labels, LABEL_COLUMNS)
    return len(features), len(labels)


def read_features(store: MarketStore) -> pd.DataFrame:
    return _read(store, _FEATURE_TABLE)


def read_labels(store: MarketStore) -> pd.DataFrame:
    return _read(store, _LABEL_TABLE)


def _replace_table(
    store: MarketStore,
    table: str,
    frame: pd.DataFrame,
    value_columns: tuple[str, ...],
) -> None:
    payload = frame.loc[:, ["symbol", "date", *value_columns]].copy()
    payload["date"] = pd.to_datetime(payload["date"]).dt.date
    column_sql = ",\n".join(f"    {name} DOUBLE" for name in value_columns)
    ddl = f"""
        CREATE TABLE {table} (
            symbol VARCHAR NOT NULL,
            date DATE NOT NULL,
            {column_sql},
            PRIMARY KEY (symbol, date)
        )
    """
    incoming = f"_incoming_{table}"
    store.execute("BEGIN")
    try:
        store.execute(f"DROP TABLE IF EXISTS {table}")
        store.execute(ddl)
        if not payload.empty:
            store.register_frame(incoming, payload)
            names = ", ".join(["symbol", "date", *value_columns])
            store.execute(f"INSERT INTO {table} ({names}) SELECT {names} FROM {incoming}")
        store.execute("COMMIT")
    except Exception:
        store.execute("ROLLBACK")
        raise
    finally:
        store.unregister_frame(incoming)


def _read(store: MarketStore, table: str) -> pd.DataFrame:
    try:
        frame = store.execute(f"SELECT * FROM {table} ORDER BY symbol, date").df()
    except Exception as exc:
        message = str(exc).lower()
        if "does not exist" in message or "not found" in message:
            raise RuntimeError(
                f"No {table} table in this database. Run `driftwood features` first."
            ) from exc
        raise
    if not frame.empty:
        frame["date"] = pd.to_datetime(frame["date"])
    return frame
