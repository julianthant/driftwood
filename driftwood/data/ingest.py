"""Write a bar frame into DuckDB and record what happened.

Callers that talk to yfinance use :func:`ingest_yfinance`. Tests and the
notebook use :func:`ingest_synthetic` or pass a frame to :func:`ingest_frame`.
"""

from __future__ import annotations

import logging
import uuid
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime

import pandas as pd

from driftwood.data.store import MarketStore
from driftwood.data.synthetic import synthetic_bars
from driftwood.data.universe import describe_symbols
from driftwood.data.universe import symbols as universe_symbols
from driftwood.data.validate import validate_bars
from driftwood.data.yfinance import fetch_yfinance_bars

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class IngestResult:
    run_id: str
    source: str
    rows_received: int
    rows_written: int
    rows_rejected: int
    symbols_written: int
    symbols_missing: tuple[str, ...]
    reject_reasons: dict[str, int]

    @property
    def ok(self) -> bool:
        return self.rows_written > 0

    def summary(self) -> str:
        missing = ", ".join(self.symbols_missing) if self.symbols_missing else "none"
        reasons = self.reject_reasons or {}
        return (
            f"ingest {self.source}: wrote {self.rows_written} rows "
            f"for {self.symbols_written} symbols, rejected {self.rows_rejected} "
            f"({reasons}), missing symbols: {missing}"
        )


def ingest_yfinance(
    store: MarketStore,
    symbols: Sequence[str] | None,
    start: str,
    end: str,
) -> IngestResult:
    requested = _requested(symbols)
    frame = fetch_yfinance_bars(requested, start, end)
    return ingest_frame(store, frame, source="yfinance", requested=requested)


def ingest_synthetic(
    store: MarketStore,
    symbols: Sequence[str] | None,
    *,
    start: str = "2018-01-02",
    periods: int = 252,
    seed: int = 7,
    split_symbol: str | None = None,
) -> IngestResult:
    requested = _requested(symbols)
    frame = synthetic_bars(
        requested,
        start=start,
        periods=periods,
        seed=seed,
        split_symbol=split_symbol,
    )
    return ingest_frame(store, frame, source="synthetic", requested=requested)


def ingest_frame(
    store: MarketStore,
    frame: pd.DataFrame,
    *,
    source: str,
    requested: Sequence[str] | None = None,
) -> IngestResult:
    """Validate ``frame`` and upsert the surviving rows.

    Re-running with the same symbol and date replaces the stored bar. The
    ingest log appends a new row every time, so a correction is visible.
    """

    started = _utcnow()
    run_id = str(uuid.uuid4())
    requested_list = [symbol.upper() for symbol in requested] if requested is not None else []
    try:
        clean, reasons = validate_bars(frame)
        rejected = int(sum(reasons.values()))
        stamped = clean.copy()
        if not stamped.empty:
            stamped["source"] = source
            stamped["ingested_at"] = started
        written = store.upsert_bars(stamped) if not stamped.empty else 0
        written_symbols = set(stamped["symbol"].unique()) if written else set()
        if requested_list:
            missing = tuple(symbol for symbol in requested_list if symbol not in written_symbols)
            store.replace_universe(describe_symbols(requested_list))
        else:
            missing = ()
        status = "ok" if written else "empty"
        result = IngestResult(
            run_id=run_id,
            source=source,
            rows_received=int(len(frame)),
            rows_written=int(written),
            rows_rejected=rejected,
            symbols_written=len(written_symbols),
            symbols_missing=missing,
            reject_reasons=reasons,
        )
        store.record_run(
            run_id=run_id,
            started_at=started,
            finished_at=_utcnow(),
            source=source,
            symbols_requested=len(requested_list),
            rows_received=result.rows_received,
            rows_written=result.rows_written,
            rows_rejected=result.rows_rejected,
            status=status,
            notes={"reject_reasons": reasons, "symbols_missing": list(missing)},
        )
        logger.info(result.summary())
        return result
    except Exception as exc:
        store.record_run(
            run_id=run_id,
            started_at=started,
            finished_at=_utcnow(),
            source=source,
            symbols_requested=len(requested_list),
            rows_received=int(len(frame)),
            rows_written=0,
            rows_rejected=0,
            status="error",
            notes={"error": type(exc).__name__},
        )
        raise


def _requested(symbols: Sequence[str] | None) -> list[str]:
    if symbols:
        return [symbol.strip().upper() for symbol in symbols if symbol.strip()]
    return universe_symbols()


def _utcnow() -> datetime:
    return datetime.now(UTC).replace(tzinfo=None)
