"""
cli/cmd_portfolio.py
─────────────────────
CLI subcommand: equitydesk portfolio
"""

import argparse

from bll.portfolio_service import PortfolioService
from utils.logger import get_logger

log = get_logger(__name__)


def register(subparsers) -> None:
    p = subparsers.add_parser("portfolio", help="Portfolio evaluation and live snapshot")
    sub = p.add_subparsers(dest="port_cmd", metavar="ACTION")

    sub.add_parser("weights",    help="Show target weights")
    sub.add_parser("exclusions", help="Show excluded symbols")

    snap = sub.add_parser("snapshot", help="Live portfolio snapshot (mock positions)")


def run(args: argparse.Namespace) -> None:
    svc = PortfolioService()

    if args.port_cmd == "weights":
        weights = svc.load_weights()
        if not weights:
            print("No weights file found.")
        else:
            for ticker, w in sorted(weights.items(), key=lambda x: -x[1]):
                print(f"{ticker:<20}  {w:.4f}  ({w*100:.2f}%)")

    elif args.port_cmd == "exclusions":
        excl = svc.load_exclusions()
        if not excl:
            print("No exclusions.")
        else:
            for sym in excl:
                print(f"  {sym}")

    elif args.port_cmd == "snapshot":
        # Mock positions for demo
        mock_positions = {"BEL": 100, "BDL": 50, "ASTRAMICRO": 200}
        df = svc.get_live_snapshot(mock_positions)
        print(df.to_string(index=False))

    else:
        print("Usage: equitydesk portfolio {weights|exclusions|snapshot}")
