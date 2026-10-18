"""Command line entry points."""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

from driftwood.attribution import attribute_store, write_attribution_report
from driftwood.backtest import BacktestConfig, run_backtest, write_backtest_report
from driftwood.config import database_path, load_env
from driftwood.data.ingest import ingest_synthetic, ingest_yfinance
from driftwood.data.store import MarketStore
from driftwood.data.yfinance import DataUnavailable
from driftwood.features.factors import MODEL_FEATURES
from driftwood.features.persist import read_features, write_research_tables
from driftwood.features.pipeline import build_research_frame
from driftwood.live.broker import make_paper_client
from driftwood.live.run import (
    MarketOpenError,
    PaperPlan,
    broker_equity_and_positions,
    plan_from_store,
    submit_plan,
)
from driftwood.models.ic import information_coefficient, summarize_ic
from driftwood.models.persist import read_predictions
from driftwood.models.train import horizon_for_label, train_store
from driftwood.models.walkforward import InsufficientHistory, WalkForwardConfig


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

    train = sub.add_parser(
        "train",
        help="Walk-forward fit on stored features and write out-of-sample predictions",
    )
    train.add_argument("--db", default=None, help="DuckDB file. Defaults to DRIFTWOOD_DB or data/")
    train.add_argument(
        "--model",
        choices=("ridge", "momentum", "lgbm"),
        default="ridge",
        help="ridge fits the z-scores, momentum copies z_mom_12_1, lgbm is the tree check",
    )
    train.add_argument(
        "--label",
        choices=("fwd_ret_1d", "fwd_ret_5d"),
        default="fwd_ret_1d",
    )
    train.add_argument("--min-train-days", type=int, default=252)
    train.add_argument("--test-days", type=int, default=63)
    train.add_argument(
        "--embargo",
        type=int,
        default=5,
        help="Sessions dropped between train and test. The purge is at least the label horizon.",
    )
    train.add_argument(
        "--label-horizon",
        type=int,
        default=None,
        help=(
            "Extend the horizon inferred from the label "
            "(1 for fwd_ret_1d, 5 for fwd_ret_5d). Cannot be shorter."
        ),
    )
    train.add_argument(
        "--train-days",
        type=int,
        default=756,
        help="Training length when --rolling is set. Ignored for the default expanding window.",
    )
    train.add_argument(
        "--rolling",
        action="store_true",
        help="Use a fixed training length instead of an expanding window",
    )
    train.add_argument(
        "--alpha",
        type=float,
        default=1.0,
        help="Ridge penalty. Ignored by the momentum and tree models.",
    )
    train.add_argument(
        "--report",
        default=None,
        help="Directory for the IC report. Omit to store predictions without writing files.",
    )

    backtest = sub.add_parser(
        "backtest",
        help="Fill next-session orders for immediate, TWAP, and VWAP",
    )
    backtest.add_argument(
        "--db",
        default=None,
        help="DuckDB file. Defaults to DRIFTWOOD_DB or data/",
    )
    backtest.add_argument("--model", default="ridge", help="Prediction model name to trade")
    backtest.add_argument(
        "--algos",
        nargs="+",
        choices=("immediate", "twap", "vwap"),
        default=["immediate", "twap", "vwap"],
    )
    backtest.add_argument("--capital", type=float, default=1_000_000.0)
    backtest.add_argument("--gross", type=float, default=1.0)
    backtest.add_argument("--max-weight", type=float, default=0.1)
    backtest.add_argument("--half-spread-bps", type=float, default=1.0)
    backtest.add_argument("--impact", type=float, default=0.0)
    backtest.add_argument("--participation", type=float, default=0.1)
    backtest.add_argument("--min-trade-dollars", type=float, default=0.0)
    backtest.add_argument(
        "--report",
        default=None,
        help="Directory for the comparison. Omit to print the summary only.",
    )

    paper = sub.add_parser(
        "paper",
        help="Preview next-session paper orders. Nothing is sent unless --submit is set",
    )
    paper.add_argument("--db", default=None, help="DuckDB file. Defaults to DRIFTWOOD_DB or data/")
    paper.add_argument("--model", default="ridge", help="Prediction model name to trade")
    paper.add_argument("--capital", type=float, default=1_000_000.0)
    paper.add_argument("--gross", type=float, default=1.0)
    paper.add_argument("--max-weight", type=float, default=0.1)
    paper.add_argument("--min-trade-dollars", type=float, default=0.0)
    paper.add_argument(
        "--submit",
        action="store_true",
        help="Send DAY market orders to the Alpaca paper host. Off by default.",
    )
    paper.add_argument(
        "--allow-while-open",
        action="store_true",
        help="Send even if the paper market is open. A DAY order would then fill today.",
    )

    attribute = sub.add_parser(
        "attribute",
        help="Split the last scored close-to-close return across signals",
    )
    attribute.add_argument(
        "--db",
        default=None,
        help="DuckDB file. Defaults to DRIFTWOOD_DB or data/",
    )
    attribute.add_argument("--model", default="ridge")
    attribute.add_argument("--gross", type=float, default=1.0)
    attribute.add_argument("--max-weight", type=float, default=0.1)
    attribute.add_argument(
        "--report",
        default=None,
        help="Directory for the attribution note. Omit to print the table only.",
    )

    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    if args.command == "ingest":
        return _ingest(parser, args)
    if args.command == "features":
        return _features(args)
    if args.command == "train":
        return _train(parser, args)
    if args.command == "backtest":
        return _backtest(args)
    if args.command == "paper":
        return _paper(args)
    if args.command == "attribute":
        return _attribute(args)
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


def _train(parser: argparse.ArgumentParser, args: argparse.Namespace) -> int:
    inferred = horizon_for_label(args.label)
    if args.label_horizon is None:
        horizon = inferred
    elif args.label_horizon < inferred:
        parser.error(
            f"--label-horizon must be at least {inferred} for {args.label}, "
            f"got {args.label_horizon}"
        )
    else:
        horizon = args.label_horizon
    try:
        config = WalkForwardConfig(
            min_train_days=args.min_train_days,
            test_days=args.test_days,
            embargo_days=args.embargo,
            label_horizon=horizon,
            expanding=not args.rolling,
            train_days=args.train_days,
        )
    except ValueError as exc:
        parser.error(str(exc))
    path = database_path(args.db)
    report_dir = Path(args.report) if args.report else None
    try:
        with MarketStore(path) as store:
            trained = train_store(
                store,
                model=args.model,
                label=args.label,
                config=config,
                alpha=args.alpha,
                report_dir=report_dir,
            )
    except (RuntimeError, InsufficientHistory, ValueError) as exc:
        print(str(exc), file=sys.stderr)
        return 1
    daily = information_coefficient(trained.result.predictions, "prediction", args.label)
    summary = summarize_ic(daily, hac_lags=config.label_horizon)
    print(
        f"model: {trained.result.model_name}, "
        f"predictions: {trained.rows_written} rows, "
        f"folds: {len(trained.result.folds)}"
    )
    print(f"mean spearman IC: {_format_ic(summary['mean_ic'])}")
    if report_dir is not None:
        print(f"report: {trained.report_path}")
    print(f"database: {path}")
    return 0


def _backtest(args: argparse.Namespace) -> int:
    path = database_path(args.db)
    try:
        config = BacktestConfig(
            initial_capital=args.capital,
            gross=args.gross,
            max_weight=args.max_weight,
            min_trade_dollars=args.min_trade_dollars,
            half_spread_bps=args.half_spread_bps,
            impact_coef=args.impact,
            participation_limit=args.participation,
        )
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    with MarketStore(path) as store:
        bars = store.read_bars()
        if bars.empty:
            print("no bars in the database; run driftwood ingest first", file=sys.stderr)
            return 1
        try:
            predictions = read_predictions(store)
        except RuntimeError as exc:
            print(str(exc), file=sys.stderr)
            return 1
        try:
            features = read_features(store)
        except RuntimeError:
            features = None
    chosen = predictions.loc[predictions["model"] == args.model]
    if chosen.empty:
        print(f"no predictions for model {args.model}", file=sys.stderr)
        return 1
    scores = chosen.loc[:, ["symbol", "date", "prediction"]].rename(columns={"prediction": "score"})
    volatility = None
    if features is not None and "vol_21" in features.columns:
        volatility = features.loc[:, ["symbol", "date", "vol_21"]].rename(
            columns={"vol_21": "sigma"}
        )
    try:
        result = run_backtest(
            bars,
            scores,
            algos=tuple(args.algos),
            config=config,
            volatility=volatility,
        )
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    print(result.summary.to_string(index=False))
    if args.report:
        report = write_backtest_report(result, Path(args.report))
        print(f"report: {report}")
    print(f"database: {path}")
    return 0


def _paper(args: argparse.Namespace) -> int:
    path = database_path(args.db)
    try:
        with MarketStore(path) as store:
            plan = plan_from_store(
                store,
                model=args.model,
                equity=args.capital,
                gross=args.gross,
                max_weight=args.max_weight,
                min_trade_dollars=args.min_trade_dollars,
            )
    except (RuntimeError, ValueError) as exc:
        print(str(exc), file=sys.stderr)
        return 1
    if not args.submit:
        _print_plan("dry-run", plan)
        print(f"database: {path}")
        return 0
    api_key = os.environ.get("ALPACA_API_KEY", "")
    secret_key = os.environ.get("ALPACA_SECRET_KEY", "")
    if not api_key or not secret_key:
        print(
            "ALPACA_API_KEY and ALPACA_SECRET_KEY are required to submit",
            file=sys.stderr,
        )
        return 1
    try:
        client = make_paper_client(api_key, secret_key, url=os.environ.get("ALPACA_BASE_URL"))
        _refuse_if_open(client, allow_while_open=args.allow_while_open)
        equity, positions = broker_equity_and_positions(client)
        with MarketStore(path) as store:
            plan = plan_from_store(
                store,
                model=args.model,
                equity=equity,
                positions=positions,
                gross=args.gross,
                max_weight=args.max_weight,
                min_trade_dollars=args.min_trade_dollars,
            )
        submit_plan(client, plan.orders, allow_while_open=args.allow_while_open)
    except (MarketOpenError, ValueError) as exc:
        print(str(exc), file=sys.stderr)
        return 1
    _print_plan("submitted", plan)
    print(f"database: {path}")
    return 0


def _attribute(args: argparse.Namespace) -> int:
    path = database_path(args.db)
    try:
        with MarketStore(path) as store:
            result = attribute_store(
                store,
                model=args.model,
                gross=args.gross,
                max_weight=args.max_weight,
            )
    except (RuntimeError, ValueError) as exc:
        print(str(exc), file=sys.stderr)
        return 1
    print(
        f"signal {result.signal_date.date()} -> {result.next_date.date()}, "
        f"close-to-close {result.portfolio_return:.6f}"
    )
    print(result.note)
    print(result.by_signal.to_string(index=False))
    print("Contributions sum to the target book's close-to-close return.")
    print("They are not the filled equity-curve return.")
    if args.report:
        report = write_attribution_report(result, Path(args.report))
        print(f"report: {report}")
    print(f"database: {path}")
    return 0


def _print_plan(kind: str, plan: PaperPlan) -> None:
    print(f"{kind}: {len(plan.orders)} orders for signal {plan.signal_date.date()}")
    for order in plan.orders:
        print(f"{order.symbol} {order.side} {order.quantity:.6f}")


def _refuse_if_open(client: object, *, allow_while_open: bool) -> None:
    clock = client.get_clock()
    if bool(getattr(clock, "is_open", False)) and not allow_while_open:
        raise MarketOpenError(
            "The paper market is open. A DAY order would fill this session, "
            "and the signal was the prior close. Pass --allow-while-open to send anyway."
        )


def _format_ic(value: object) -> str:
    try:
        number = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return "n/a"
    if number != number:
        return "n/a"
    return f"{number:.4f}"


if __name__ == "__main__":
    raise SystemExit(main())
