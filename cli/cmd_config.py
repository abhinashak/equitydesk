"""
cli/cmd_config.py
─────────────────
CLI subcommand: equitydesk config
"""

import argparse

from bll.config_service import ConfigService
from utils.logger import get_logger

log = get_logger(__name__)


def register(subparsers) -> None:
    p = subparsers.add_parser("config", help="View and edit configuration")
    sub = p.add_subparsers(dest="cfg_cmd", metavar="ACTION")

    sub.add_parser("show",    help="Print all app config values")
    sub.add_parser("tickers", help="Print ticker list")
    sub.add_parser("periods", help="Print period definitions")

    set_p = sub.add_parser("set", help="Set a config key")
    set_p.add_argument("key")
    set_p.add_argument("value")

    add_t = sub.add_parser("add-ticker", help="Add a ticker")
    add_t.add_argument("name")
    add_t.add_argument("--yahoo",   default=None)
    add_t.add_argument("--sector",  default="")
    add_t.add_argument("--cap",     default="Mid-cap")

    del_t = sub.add_parser("del-ticker", help="Delete a ticker")
    del_t.add_argument("name")


def run(args: argparse.Namespace) -> None:
    svc = ConfigService()

    if args.cfg_cmd == "show":
        cfg = svc.get_app_config()
        width = max(len(k) for k in cfg) + 2
        for k, v in sorted(cfg.items()):
            print(f"{k:<{width}} = {v}")

    elif args.cfg_cmd == "tickers":
        df = svc.get_tickers()
        print(df.to_string(index=False))

    elif args.cfg_cmd == "periods":
        import json
        print(json.dumps(svc.get_periods(), indent=2))

    elif args.cfg_cmd == "set":
        svc.set_app_value(args.key, args.value)
        print(f"✅ {args.key} = {args.value}")

    elif args.cfg_cmd == "add-ticker":
        yahoo = args.yahoo or f"{args.name}.NS"
        svc.add_ticker({
            "Name": args.name,
            "Yahoo Symbol": yahoo,
            "Sector": args.sector,
            "market_cap": args.cap,
            "domestic_market_pct": 100.0,
            "num_clients": 1,
            "num_sectors_served": 1,
        })
        print(f"✅ Added {args.name}")

    elif args.cfg_cmd == "del-ticker":
        svc.delete_ticker(args.name)
        print(f"✅ Deleted {args.name}")

    else:
        print("Usage: equitydesk config {show|tickers|periods|set|add-ticker|del-ticker}")
