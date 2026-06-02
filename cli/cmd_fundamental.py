"""
cli/cmd_fundamental.py
──────────────────────
CLI subcommand: equitydesk fundamental
"""

import argparse

from bll.fundamental_service import FundamentalService
from dal.ticker_config import TickerConfigDAL
from utils.logger import get_logger

log = get_logger(__name__)


def register(subparsers) -> None:
    p = subparsers.add_parser("fundamental", help="Download and merge fundamental data")
    sub = p.add_subparsers(dest="fund_cmd", metavar="ACTION")

    # --- run ---
    run_p = sub.add_parser("run", help="Run incremental fundamental loader")
    run_p.add_argument("--tickers",    nargs="*", help="Specific tickers (default: all from config)")
    run_p.add_argument("--out-dir",    default="data/fundamental")
    run_p.add_argument("--parser",     default="screener_parser.py")
    run_p.add_argument("--sleep",      type=int, default=10)

    # --- coverage ---
    cov_p = sub.add_parser("coverage", help="Show table coverage summary")
    cov_p.add_argument("--out-dir", default="data/fundamental")


def run(args: argparse.Namespace) -> None:
    if args.fund_cmd == "run":
        out_dir = getattr(args, "out_dir", "data/fundamental")
        svc     = FundamentalService(
            out_dir=out_dir,
            parser_script=getattr(args, "parser", "screener_parser.py"),
            sleep_secs=getattr(args, "sleep", 10),
        )
        tickers = getattr(args, "tickers", None)
        if not tickers:
            cfg     = TickerConfigDAL()
            tickers = cfg.load()["Name"].dropna().tolist()

        print(f"▶ Running fundamental loader for {len(tickers)} tickers…")
        for line in svc.run(tickers):
            print(line)

    elif args.fund_cmd == "coverage":
        out_dir = getattr(args, "out_dir", "data/fundamental")
        svc     = FundamentalService(out_dir=out_dir)
        df      = svc.get_coverage()
        print(df.to_string(index=False))

    else:
        print("Usage: equitydesk fundamental {run|coverage} [options]")
