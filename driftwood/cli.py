"""Command line entry points."""

from __future__ import annotations

import argparse
import logging
import sys

from driftwood.config import database_path, load_env
from driftwood.data.ingest import ingest_synthetic, ingest_yfinance
from driftwood.data.store import MarketStore
from driftwood.data.yfinance import DataUnavailable
from driftwood.features.factors import MODEL_FEATURES
from driftwood.features.persist import write_research_tables
from driftwood.features.pipeline import build_research_frame


def main(argv: list[str] | None = None) -> int:
    load_env()
    parser = argparse.ArgumentParser(
        prog="driftwood",
        description="Research and paper-trading pipeline for US stocks.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    ingest = sub.add_parser("ingest", help="Download or synthesize daily bars into DuckDB")
    ingest.add_argument(
        "--start",
        help="First session date, YYYY-MM-DD (yfinance start, inclusive)",
    )
    ingest.add_argument(
        "--end",
        help="End date, YYYY-MM-DD. Passed through to yfinance, which treats it as exclusive.",
    )
    ingest.add_argument(
        "--synthetic",
        action="store_true",
        help="Write a deterministic random-walk book instead of calling yfinance",
    )
    ingest.add_argument("--periods", type=int, default=252, help="Business days when --synthetic")
    ingest.add_argument("--seed", type=int, default=7)
    ingest.add_argument(
        "--split-symbol",
        default=None,
        help="With --synthetic, apply a 2-for-1 split to this symbol",
    )
    ingest.add_argument(
        "--symbols",
        nargs="*",
        help="Tickers to load. Defaults to the built-in liquid universe.",
    )
    ingest.add_argument("--db", default=None, help="DuckDB file. Defaults to DRIFTWOOD_DB or data/")

    features = sub.add_parser(
        "features",
        help="Build close-of-day features and forward labels from stored bars",
    )
    features.add_argument(
        "--db",
        default=None,
        help="DuckDB file. Defaults to DRIFTWOOD_DB or data/",
    )

    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    if args.command == "ingest":
        return _ingest(parser, args)
    if args.command == "features":
        return _features(args)
    parser.error(f"Unknown command {args.command}")
    return 2


def cli() -> None:
    """Setuptools console-script target. Propagates the integer exit code."""

    raise SystemExit(main())


def _ingest(parser: argparse.ArgumentParser, args: argparse.Namespace) -> int:
    if args.periods < 2:
        parser.error("--periods must be at least 2")
    path = database_path(args.db)
    try:
        with MarketStore(path) as store:
            if args.synthetic:
                result = ingest_synthetic(
                    store,
                    args.symbols,
                    start=args.start or "2018-01-02",
                    periods=args.periods,
                    seed=args.seed,
                    split_symbol=args.split_symbol,
                )
            else:
                if not args.start or not args.end:
                    parser.error("ingest requires --start and --end, or pass --synthetic")
                result = ingest_yfinance(store, args.symbols, args.start, args.end)
    except DataUnavailable as exc:
        print(str(exc), file=sys.stderr)
        return 1
    print(result.summary())
    print(f"database: {path}")
    return 0 if result.ok else 1


def _features(args: argparse.Namespace) -> int:
    path = database_path(args.db)
    with MarketStore(path) as store:
        bars = store.read_bars()
        if bars.empty:
            print("no bars in the database; run driftwood ingest first", file=sys.stderr)
            return 1
        frame = build_research_frame(bars)
        feature_rows, label_rows = write_research_tables(store, frame)
    print(f"features: {feature_rows} rows, labels: {label_rows} rows")
    print(f"database: {path}")
    for column in ("z_ret_1d", "z_mom_12_1"):
        if column not in MODEL_FEATURES:
            continue
        present = frame[column].notna().mean()
        print(f"{column} non-null: {present:.1%}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
