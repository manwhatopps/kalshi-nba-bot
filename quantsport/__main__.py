"""python -m quantsport <command>

  research      re-fit the rating models, test them against Kalshi's prices, save what the fund may use
  daily         one full investment analysis (scan -> estimate -> chain -> audit -> tiers); --mode research|paper|live_review
  settle        grade open paper positions and past decisions
  performance   the performance audit: P&L by tier, calibration by sport and market
  positions     where each open card stands now, and the hedge when one leg is left (read-only)
  backtest      walk-forward tests, the negative control, Monte Carlo of the card programme
  dashboard     write data/quantsport/dashboard.html from the latest run
  tickets       list order tickets from the latest run (LIVE REVIEW)
  approve N     approve ticket N at a terminal (one use, time-limited); --init creates the approval key
  config        show the risk configuration; --set key=value changes data/quantsport.json
"""
from __future__ import annotations

import argparse
import json


def main() -> None:
    ap = argparse.ArgumentParser(prog="quantsport", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("research")
    d = sub.add_parser("daily")
    d.add_argument("--mode", choices=["research", "paper", "live_review"], default=None)
    d.add_argument("--hours", type=float, default=None)
    sub.add_parser("settle")
    sub.add_parser("performance")
    sub.add_parser("positions")
    sub.add_parser("backtest")
    sub.add_parser("dashboard")
    sub.add_parser("tickets")
    a = sub.add_parser("approve")
    a.add_argument("n", nargs="?", type=int)
    a.add_argument("--init", action="store_true")
    c = sub.add_parser("config")
    c.add_argument("--set", nargs="*")
    args = ap.parse_args()
    from . import config, store
    if args.cmd == "research":
        from . import gamedata
        from .models import stack
        gamedata.build_pre(refresh_ncaaf=True)                       # pick up any new college football price history
        fits = stack.fit_ratings()
        cur = json.loads(stack.MODELS.read_text()) if stack.MODELS.exists() else {}
        cur["ratings"] = {s: par for s, (_, _, par) in fits.items()}
        stack.MODELS.write_text(json.dumps(cur, indent=1))
        stack.evaluate(stack.market_table(fits, rebuild=True))
    elif args.cmd == "daily":
        from . import dashboard, pipeline
        o = pipeline.run(mode=args.mode, hours=args.hours)
        if "error" not in o:
            print(f"\nDashboard: {dashboard.write()}")
    elif args.cmd == "settle":
        from bot.kalshi import data_client
        from . import audit
        print(audit.settle(data_client(), store.connect()))
    elif args.cmd == "performance":
        from bot.kalshi import data_client
        from . import audit
        audit.report(data_client(), store.connect())
    elif args.cmd == "positions":
        from bot.kalshi import data_client
        from . import monitor
        print(monitor.report(monitor.status(data_client(), store.connect())))
    elif args.cmd == "backtest":
        from . import backtest
        backtest.run()
    elif args.cmd == "dashboard":
        from . import dashboard
        print(dashboard.write())
    elif args.cmd in ("tickets", "approve"):
        from . import execution, pipeline
        f = pipeline.OUT / "latest.json"
        o = json.loads(f.read_text()) if f.exists() else {"decisions": []}
        tk = execution.tickets(o["decisions"])
        if args.cmd == "approve" and args.init:
            execution._key(create=True)
            print(f"approval key ready at {execution.key_path()} (outside the repository)")
        elif not tk:
            print("No order tickets: the latest run accepted no single position.")
        elif args.cmd == "tickets" or args.n is None:
            for i, t in enumerate(tk):
                print(f"  [{i}] BUY {t['side'].upper()} {t['quantity']} x {t['title']} limit {t['limit_price'] * 100:.0f}c, max ${t['max_cost']:.2f}  ({t['reason'][:70]})")
        else:
            token = execution.approve(store.connect(), tk[args.n])
            print(f"approved once: token {token} (valid 10 minutes). Nothing has been sent; sending needs live_review mode and a key.")
    elif args.cmd == "config":
        if args.set:
            ch = {}
            for kv in args.set:
                k, v = kv.split("=", 1)
                ch[k] = json.loads(v)
            config.save(ch)
        print(json.dumps(config.load(), indent=1))


if __name__ == "__main__":
    main()
