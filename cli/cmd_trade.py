"""
cli/cmd_trade.py
────────────────
CLI subcommand: equitydesk trade
"""

import argparse

from bll.trade_service import TradeService
from bll.config_service import ConfigService
from utils.logger import get_logger

log = get_logger(__name__)


def register(subparsers) -> None:
    p = subparsers.add_parser("trade", help="Trade execution planning and order management")
    sub = p.add_subparsers(dest="trade_cmd", metavar="ACTION")

    sub.add_parser("account",   help="Show account / margin summary")
    sub.add_parser("positions", help="Show current open positions")

    plan = sub.add_parser("plan", help="Build phased execution plan")
    plan.add_argument("--capital", type=float, default=1_000_000.0,
                      help="Total capital in ₹")
    plan.add_argument("--phases",  type=int,   default=3)


def run(args: argparse.Namespace) -> None:
    cfg = ConfigService().get_app_config()
    svc = TradeService(
        kite_base_url=cfg.get("KITE_BASE_URL", "http://localhost:8080"),
        live_orders_file=cfg.get("LIVE_ORDERS_FILE", "data/live_orders.json"),
        mock_mode=cfg.get("MOCK_MODE", "true").lower() == "true",
    )

    if args.trade_cmd == "account":
        summary = svc.get_account_summary()
        for k, v in summary.items():
            print(f"{k:<25} {v}")

    elif args.trade_cmd == "positions":
        df = svc.get_current_positions()
        if df.empty:
            print("No open positions.")
        else:
            print(df.to_string(index=False))

    elif args.trade_cmd == "plan":
        from bll.portfolio_service import PortfolioService
        port_svc = PortfolioService()
        weights  = port_svc.load_weights()
        if not weights:
            print("No weights found in data/target_weights.txt")
            return

        # Mock prices for demo
        mock_prices = {t: 500.0 for t in weights}
        target_df   = svc.compute_target_positions(
            weights=weights,
            total_capital=args.capital,
            prices=mock_prices,
        )
        target_qty = dict(zip(target_df["ticker"], target_df["target_qty"]))
        plan_df    = svc.build_execution_plan(
            current={},
            target=target_qty,
            prices=mock_prices,
            phases=args.phases,
        )
        if plan_df.empty:
            print("No trades needed.")
        else:
            print(plan_df.to_string(index=False))

    else:
        print("Usage: equitydesk trade {account|positions|plan}")
