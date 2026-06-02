#!/usr/bin/env python3
"""
cli/main.py
───────────
EquityDesk – Console / standalone mode.

Usage:
    python -m cli.main <command> [options]

Commands:
    ticker      Manage / download ticker OHLCV data
    fundamental Download and merge fundamental data
    signal      Generate momentum signals
    config      View / edit configuration
    portfolio   Portfolio utilities
    trade       Trade execution planning

Run `python -m cli.main <command> --help` for command-specific options.
"""

import argparse
import sys

from cli import (
    cmd_config,
    cmd_fundamental,
    cmd_portfolio,
    cmd_signal,
    cmd_ticker,
    cmd_trade,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="equitydesk",
        description="EquityDesk — NSE equity research console",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest="command", metavar="COMMAND")

    cmd_ticker.register(sub)
    cmd_fundamental.register(sub)
    cmd_signal.register(sub)
    cmd_config.register(sub)
    cmd_portfolio.register(sub)
    cmd_trade.register(sub)

    return parser


def main(argv=None):
    parser = build_parser()
    args   = parser.parse_args(argv)

    if args.command is None:
        parser.print_help()
        sys.exit(0)

    # Each cmd_* module exposes a run(args) function
    dispatch = {
        "ticker":      cmd_ticker.run,
        "fundamental": cmd_fundamental.run,
        "signal":      cmd_signal.run,
        "config":      cmd_config.run,
        "portfolio":   cmd_portfolio.run,
        "trade":       cmd_trade.run,
    }
    dispatch[args.command](args)


if __name__ == "__main__":
    main()
