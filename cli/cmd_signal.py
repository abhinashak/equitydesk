"""
cli/cmd_signal.py
─────────────────
CLI subcommand: equitydesk signal
"""

import argparse

from bll.signal_service import SignalService
from utils.logger import get_logger

log = get_logger(__name__)


def register(subparsers) -> None:
    p = subparsers.add_parser("signal", help="Generate and query momentum signals")
    sub = p.add_subparsers(dest="signal_cmd", metavar="ACTION")

    gen = sub.add_parser("generate", help="Run signal_generator.py")
    gen.add_argument("--config",     default="config/tickers.csv")
    gen.add_argument("--signal-dir", default="data/signal_momentum")

    show = sub.add_parser("show", help="Show latest signal snapshot")
    show.add_argument("--signal-dir", default="data/signal_momentum")
    show.add_argument("--ticker",     default=None, help="Filter to one ticker")
    show.add_argument("--top",        type=int, default=20)

    sql = sub.add_parser("screen", help="Run a SQL screen against the signal table")
    sql.add_argument("query", help="DuckDB SQL query (use table name 'signals')")
    sql.add_argument("--signal-dir", default="data/signal_momentum")


def run(args: argparse.Namespace) -> None:
    svc = SignalService(signal_dir=getattr(args, "signal_dir", "data/signal_momentum"))

    if args.signal_cmd == "generate":
        print("▶ Generating signals…")
        for line in svc.run_signal_generation(config_file=args.config):
            print(line)

    elif args.signal_cmd == "show":
        if args.ticker:
            df = svc.get_signals_for_ticker(args.ticker)
        else:
            df = svc.get_latest_signals()
        if df is None or df.empty:
            print("No signal data found.")
        else:
            print(df.head(args.top).to_string(index=False))

    elif args.signal_cmd == "screen":
        df = svc.run_sql_screen(args.query)
        print(df.to_string(index=False))

    else:
        print("Usage: equitydesk signal {generate|show|screen} [options]")
