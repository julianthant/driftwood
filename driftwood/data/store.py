"""DuckDB persistence for daily bars.

The connection is local and single-process. Research runs open the file, read
a frame, and close it. There is no database server to keep running.
"""

from __future__ import annotations

import json
from datetime import date, datetime
from pathlib import Path

import duckdb
import pandas as pd

from driftwood.data.schema import BAR_COLUMNS, DDL


class MarketStore:
    """A DuckDB file holding bars, the universe used for a study, and ingest runs."""

    def __init__(self, path: str | Path) -> None:
        self.path: str | Path
        if str(path) == ":memory:":
            self.path = ":memory:"
        else:
            self.path = Path(path)
            self.path.parent.mkdir(parents=True, exist_ok=True)
        self._con = duckdb.connect(str(self.path))
        self._con.execute(DDL)

    def close(self) -> None:
        if self._con is not None:
            self._con.close()
            self._con = None

    def __enter__(self) -> MarketStore:
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> bool:
        self.close()
        return False

    def upsert_bars(self, frame: pd.DataFrame) -> int:
        """Insert bars, replacing any existing row with the same symbol and date."""

        if frame.empty:
            return 0
        missing = [column for column in BAR_COLUMNS if column not in frame.columns]
        if missing:
            raise ValueError(f"Cannot store bars, missing columns: {missing}")

        payload = frame.loc[:, list(BAR_COLUMNS)].copy()
        payload["date"] = pd.to_datetime(payload["date"]).dt.date
        payload["ingested_at"] = pd.to_datetime(payload["ingested_at"])
        payload["volume"] = payload["volume"].astype("int64")

        self._con.register("_incoming_bars", payload)
        try:
            self._con.execute("BEGIN")
            self._con.execute(
                """
                INSERT INTO bars (
                    symbol, date, open, high, low, close, adj_close,
                    volume, source, ingested_at
                )
                SELECT
                    symbol, date, open, high, low, close, adj_close,
                    volume, source, ingested_at
                FROM _incoming_bars
                ON CONFLICT (symbol, date) DO UPDATE SET
                    open = EXCLUDED.open,
                    high = EXCLUDED.high,
                    low = EXCLUDED.low,
                    close = EXCLUDED.close,
                    adj_close = EXCLUDED.adj_close,
                    volume = EXCLUDED.volume,
                    source = EXCLUDED.source,
                    ingested_at = EXCLUDED.ingested_at
                """
            )
            self._con.execute("COMMIT")
        except Exception:
            self._con.execute("ROLLBACK")
            raise
        finally:
            self._con.unregister("_incoming_bars")
        return int(len(payload))

    def replace_universe(self, frame: pd.DataFrame) -> None:
        """Replace the stored universe with the symbols this study requested."""

        expected = ["symbol", "name", "sector"]
        missing = [column for column in expected if column not in frame.columns]
        if missing:
            raise ValueError(f"Universe frame is missing columns: {missing}")
        payload = frame.loc[:, expected].copy()
        self._con.execute("BEGIN")
        try:
            self._con.execute("DELETE FROM universe_members")
            if not payload.empty:
                self._con.register("_incoming_universe", payload)
                self._con.execute(
                    """
                    INSERT INTO universe_members (symbol, name, sector)
                    SELECT symbol, name, sector FROM _incoming_universe
                    """
                )
            self._con.execute("COMMIT")
        except Exception:
            self._con.execute("ROLLBACK")
            raise
        finally:
            self._con.unregister("_incoming_universe")

    def record_run(
        self,
        *,
        run_id: str,
        started_at: datetime,
        finished_at: datetime,
        source: str,
        symbols_requested: int,
        rows_received: int,
        rows_written: int,
        rows_rejected: int,
        status: str,
        notes: dict[str, object] | None = None,
    ) -> None:
        self._con.execute(
            """
            INSERT INTO ingest_runs (
                run_id, started_at, finished_at, source, symbols_requested,
                rows_received, rows_written, rows_rejected, status, notes
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                run_id,
                started_at,
                finished_at,
                source,
                symbols_requested,
                rows_received,
                rows_written,
                rows_rejected,
                status,
                json.dumps(notes or {}, sort_keys=True),
            ],
        )

    def read_bars(
        self,
        symbols: list[str] | None = None,
        start: str | date | datetime | None = None,
        end: str | date | datetime | None = None,
    ) -> pd.DataFrame:
        """Read bars sorted by symbol and date.

        ``start`` and ``end`` are inclusive calendar bounds on the session date.
        Symbols are bound as SQL parameters, not interpolated into the query.
        """

        query = (
            "SELECT symbol, date, open, high, low, close, adj_close, "
            "volume, source, ingested_at FROM bars WHERE 1 = 1"
        )
        params: list[object] = []
        if symbols:
            placeholders = ", ".join(["?"] * len(symbols))
            query += f" AND symbol IN ({placeholders})"
            params.extend(symbols)
        if start is not None:
            query += " AND date >= ?"
            params.append(_as_date(start))
        if end is not None:
            query += " AND date <= ?"
            params.append(_as_date(end))
        query += " ORDER BY symbol, date"
        frame = self._con.execute(query, params).df()
        if not frame.empty:
            frame["date"] = pd.to_datetime(frame["date"])
        else:
            frame["date"] = pd.to_datetime(frame["date"])
        return frame

    def coverage(self) -> pd.DataFrame:
        return self._con.execute(
            """
            SELECT symbol,
                   COUNT(*) AS rows,
                   MIN(date) AS start_date,
                   MAX(date) AS end_date
            FROM bars
            GROUP BY symbol
            ORDER BY symbol
            """
        ).df()

    def read_universe(self) -> pd.DataFrame:
        return self._con.execute(
            "SELECT symbol, name, sector FROM universe_members ORDER BY symbol"
        ).df()

    def read_runs(self) -> pd.DataFrame:
        return self._con.execute(
            """
            SELECT run_id, started_at, finished_at, source, symbols_requested,
                   rows_received, rows_written, rows_rejected, status, notes
            FROM ingest_runs
            ORDER BY started_at
            """
        ).df()

    def latest_date(self) -> date | None:
        row = self._con.execute("SELECT MAX(date) FROM bars").fetchone()
        if row is None or row[0] is None:
            return None
        value = row[0]
        if isinstance(value, datetime):
            return value.date()
        if isinstance(value, date):
            return value
        return pd.Timestamp(value).date()


def _as_date(value: str | date | datetime) -> date:
    stamp = pd.Timestamp(value)
    if stamp.tzinfo is not None:
        stamp = stamp.tz_convert("UTC").tz_localize(None)
    return stamp.date()
