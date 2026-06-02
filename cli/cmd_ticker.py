"""
cli/cmd_ticker.py
─────────────────
CLI subcommand: equitydesk ticker
"""

import argparse

from bll.ticker_service import TickerService
from utils.logger import get_logger

log = get_logger(__name__)


def register(subparsers) -> None:
    p = subparsers.add_parser("ticker", help="Download / manage OHLCV price data")
    sub = p.add_subparsers(dest="ticker_cmd", metavar="ACTION")

    # --- historical ---
    hist = sub.add_parser("historical", help="Run full / incremental historical loader")
    hist.add_argument("--config",      default="config/tickers.csv")
    hist.add_argument("--data-root",   default="data/ticker")
    hist.add_argument("--check-rebase", action="store_true")

    # --- live ---
    live = sub.add_parser("live", help="Download today's candle only")
    live.add_argument("--config",    default="config/tickers.csv")
    live.add_argument("--data-root", default="data/ticker")

    # --- stats ---
    sub.add_parser("stats", help="Show data coverage stats")


def run(args: argparse.Namespace) -> None:
    svc = TickerService(data_root=getattr(args, "data_root", "data/ticker"))

    if args.ticker_cmd == "historical":
        print("▶ Running historical ticker loader…")
        for line in svc.run_historical(
            config_file=args.config,
            check_rebase=args.check_rebase,
        ):
            print(line)

    elif args.ticker_cmd == "live":
        print("▶ Running live ticker loader…")
        for line in svc.run_live(config_file=args.config):
            print(line)

    elif args.ticker_cmd == "stats":
        stats = svc.get_stats()
        print(f"Tickers : {stats['n_tickers']}")
        print(f"Points  : {stats['n_points']}")
        print(f"Last    : {stats['last_date']}")

    else:
        print("Usage: equitydesk ticker {historical|live|stats} [options]")
