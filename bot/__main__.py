"""CLI.  python -m bot scan | place | settle | report | backtest   (add -h for options)"""
from __future__ import annotations

import argparse
import uuid
from collections import defaultdict

from . import config, ledger
from .kalshi import data_client, trade_client


def build(args):
    from .combos import GameSims, candidate_legs, search
    from .data import load_logs
    from .markets import fetch_legs
    from .model import Model, market_scores
    from . import odds

    k = data_client()
    print("Fetching Kalshi NBA markets ...")
    legs = fetch_legs(k, include_live=args.include_live, hours_ahead=args.hours)
    games = defaultdict(list)
    for l in legs:
        games[l.game].append(l)
    print(f"  {len(legs)} open legs across {len(games)} upcoming games")
    if not legs:
        return None
    print("Loading game logs + simulating games ...")
    model = Model(load_logs(topup=not args.no_topup))
    gsims = {}
    for g, ls in games.items():
        need = defaultdict(set)
        for l in ls:
            if l.kind == "prop":
                need[(l.player, l.team)].update(l.stats)
        sim = model.simulate_game(ls[0].away, ls[0].home, sorted(need), market=market_scores(ls), need=need)
        gsims[g] = GameSims(ls, sim)
    if config.ODDS_API_KEY:
        book = odds.fetch_book_probs({(ls[0].away, ls[0].home) for ls in games.values()})
        print(f"  sportsbook: matched {odds.attach(legs, book)} legs")

    print("Finding the Kalshi combo collection ...")
    best, limit = None, {}
    for c in k.collections():
        evs = c.get("associated_events") or [{"ticker": t, "size_max": 1} for t in c.get("associated_event_tickers", [])]
        lim = {e["ticker"]: (e.get("size_max") or 1) for e in evs if e["ticker"].startswith("KXNBA")}
        if len(lim) > len(limit):
            best, limit = c["collection_ticker"], lim
    print(f"  using {best} ({len(limit)} NBA events comboable)")
    pool = candidate_legs(legs, max_pool=args.pool, max_disagree=args.max_disagree,
                          min_ask=args.min_leg_price, max_ask=args.max_leg_price)
    run_id = uuid.uuid4().hex[:8]
    ledger.log_legs(run_id, [l for l in legs if l.p_model is not None])
    print(f"  {len(pool)} legs where fair value > Kalshi mid")
    combos = search(pool, gsims, limit, min_legs=args.min_legs, max_legs=args.max_legs,
                    min_mult=args.min_mult, top=args.top, max_mult=args.max_mult, same_game=args.same_game,
                    flat_stake=args.flat_stake)
    return dict(k=k, legs=legs, pool=pool, combos=combos, collection=best, run_id=run_id, gsims=gsims)


def show(res, args):
    pool, combos = res["pool"], res["combos"]
    print("\nTOP VALUE LEGS (fair = blend of model / book / Kalshi mid)")
    print(f"  {'leg':48s} {'ask':>5s} {'mid':>5s} {'model':>6s} {'book':>5s} {'fair':>5s}")
    for l in pool[:args.show_legs]:
        b = f"{l.p_book:.2f}" if l.p_book is not None else "  -  "
        print(f"  {l.title[:48]:48s} {l.yes_ask:5.2f} {l.mid:5.2f} {l.p_model:6.2f} {b:>5s} {l.p_fair:5.2f}")
    print(f"\nTOP COMBOS (>= {args.min_mult}x, {args.min_legs}-{args.max_legs} legs)  "
          f"est = estimated maker quote; real price comes from `place`")
    if not combos:
        print("  none with positive estimated EV. That's a normal, healthy answer — no bet is a bet.")
    for i, c in enumerate(combos, 1):
        print(f"\n#{i}  {c.mult:6.1f}x  est {c.est_price:.4f}  fair {c.p_fair:.4f}  EV {c.ev:+.1%}  "
              f"corr-lift {c.corr_lift:.2f}  stake ${c.stake()}")
        for l in c.legs:
            print(f"     - {l.title}  [{l.ticker}]")
        ledger.log_combo(res["run_id"], c, "suggested")


def main():
    ap = argparse.ArgumentParser(prog="python -m bot")
    sub = ap.add_subparsers(dest="cmd")
    for name in ("scan", "place"):
        p = sub.add_parser(name)
        p.add_argument("--min-mult", type=float, default=10, help="minimum payout multiplier (default 10x)")
        p.add_argument("--max-mult", type=float, default=1000, help="ignore combos paying more than this")
        p.add_argument("--same-game", action="store_true", help="only same-game parlays (correlation matters most)")
        p.add_argument("--flat-stake", type=float, default=0.0,
                       help="bet a flat $ per combo instead of Kelly (still capped by MAX_STAKE_PER_COMBO)")
        p.add_argument("--min-legs", type=int, default=2)
        p.add_argument("--max-legs", type=int, default=6)
        p.add_argument("--top", type=int, default=10)
        p.add_argument("--pool", type=int, default=40, help="how many value legs feed the combo search")
        p.add_argument("--max-disagree", type=float, default=0.20,
                       help="skip legs where model and Kalshi differ by more than this (stale news)")
        p.add_argument("--min-leg-price", type=float, default=0.15,
                       help="ignore legs priced below this (thin long shots are where models are worst)")
        p.add_argument("--max-leg-price", type=float, default=0.92)
        p.add_argument("--hours", type=float, default=36, help="only games starting within N hours")
        p.add_argument("--include-live", action="store_true", help="include in-progress games (model is pre-game!)")
        p.add_argument("--no-topup", action="store_true", help="skip nba_api refresh")
        p.add_argument("--show-legs", type=int, default=15)
        if name == "place":
            p.add_argument("--live", action="store_true", help="actually accept quotes (default: quotes only)")
            p.add_argument("--min-ev", type=float, default=0.05, help="min EV vs the REAL quote to accept")
            p.add_argument("--max-rfqs", type=int, default=3)
    h = sub.add_parser("history", help="download Kalshi settled markets + pre-game prices")
    h.add_argument("--days", type=int, default=35)
    h.add_argument("--league", default="NBA,NFL")
    f = sub.add_parser("favorites", help="backtest the many-legs-at-90c+ strategy on Kalshi history")
    f.add_argument("--league", default="NBA")
    f.add_argument("--when", default="pre", choices=["pre", "live1", "live2"],
                   help="price snapshot: pre-game, ~1h into the game, ~2h into the game")
    f.add_argument("--samples", type=int, default=200)
    sub.add_parser("models", help="score all model types vs Kalshi prices on settled props")
    c = sub.add_parser("card", help="build tonight's favorites card (paper)")
    c.add_argument("--league", default="NBA")
    c.add_argument("--legs", type=int, default=50)
    c.add_argument("--min-price", type=float, default=0.90)
    c.add_argument("--max-price", type=float, default=0.995)
    c.add_argument("--per-game", type=int, default=8)
    c.add_argument("--include-live", action="store_true")
    c.add_argument("--naive", action="store_true", help="just take the highest-priced legs")
    mx = sub.add_parser("mix", help="tonight's efficient mixes: few mid-priced legs to a target payout (paper)")
    mx.add_argument("--league", default="NBA")
    mx.add_argument("--targets", type=float, nargs="+", default=[5, 10, 25])
    mx.add_argument("--lo", type=float, default=0.50)
    mx.add_argument("--hi", type=float, default=0.72)
    mx.add_argument("--per-game", type=int, default=2)
    mx.add_argument("--max-spread", type=float, default=0.04)
    mx.add_argument("--include-live", action="store_true")
    eff = sub.add_parser("mixes", help="backtest which mix of leg prices is most efficient at each payout")
    eff.add_argument("--league", default="NBA")
    eff.add_argument("--samples", type=int, default=150)
    so = sub.add_parser("social", help="read-only: Kalshi leaderboard + named trades in tonight's markets")
    so.add_argument("--period", default="monthly")
    so.add_argument("--top", type=int, default=100)
    so.add_argument("--series", default="KXNBAGAME,KXNBAPTS,KXNBAREB,KXNBAAST,KXNBA3PT")
    so.add_argument("--show", type=int, default=25)
    si = sub.add_parser("sims", help="20,000-simulation hit rates for tonight's player props next to Kalshi's price")
    si.add_argument("--min-price", type=float, default=0.50)
    si.add_argument("--max-price", type=float, default=0.80)
    si.add_argument("--show", type=int, default=30)
    bk = sub.add_parser("bankroll", help="what to expect: streaks, chance of profit, Kelly stake, bets to prove a strategy")
    bk.add_argument("--bankroll", type=float, default=config.BANKROLL)
    bk.add_argument("--stake", type=float, default=config.MAX_STAKE_PER_COMBO)
    bk.add_argument("--bets", type=int, default=200)
    bk.add_argument("--maker-cut", type=float, default=0.10)
    bk.add_argument("--league", default="NBA")
    ed = sub.add_parser("edges", help="scan every leg type for a real hit-rate advantage (split-sample + FDR control)")
    ed.add_argument("--league", default="NBA")
    tr = sub.add_parser("trades", help="trade-level study: how takers vs resting orders really did (all market types)")
    tr.add_argument("--download", action="store_true", help="fetch missing days + results first")
    tr.add_argument("--since", default="2026-10-20")
    tr.add_argument("--league", default="NBA")
    wt = sub.add_parser("watch", help="live paper-trade: NBA moneyline first touch of 95c during games")
    wt.add_argument("--hours", type=float, default=8)
    wt.add_argument("--thresholds", type=float, nargs="+", default=[0.95])
    be = sub.add_parser("best", help="statistically most probable combos tonight, by payout target (paper)")
    be.add_argument("--league", default="NBA")
    be.add_argument("--targets", type=float, nargs="+", default=[3, 5, 10, 25])
    be.add_argument("--min-price", type=float, default=0.35)
    be.add_argument("--max-price", type=float, default=0.95)
    be.add_argument("--per-game", type=int, default=1)
    be.add_argument("--max-spread", type=float, default=0.03)
    be.add_argument("--include-live", action="store_true")
    be.add_argument("--hours", type=float, default=30, help="only games starting within this many hours")
    be.add_argument("--game", default="", help="only this game, e.g. DALUTA (same-game card: raise --per-game)")
    be.add_argument("--groups", nargs="+", default=["1 moneyline", "4 player props", "2 spreads", "3 totals"])
    sub.add_parser("settle")
    sub.add_parser("report")
    b = sub.add_parser("backtest")
    b.add_argument("--days", type=int, default=21)
    b.add_argument("--rebuild-priors", action="store_true")
    args = ap.parse_args()
    cmd = args.cmd or "scan"
    if cmd == "scan" and not hasattr(args, "min_mult"):
        args = ap.parse_args(["scan"])
    if cmd in ("scan", "place"):
        res = build(args)
        if not res:
            return
        show(res, args)
        if cmd == "place":
            from .execute import place
            place(trade_client(), res["combos"], res["collection"], res["run_id"], live=args.live,
                  min_ev=args.min_ev, max_rfqs=args.max_rfqs)
    elif cmd == "history":
        from .history import download
        download(days=args.days, leagues=tuple(args.league.split(",")))
    elif cmd == "favorites":
        from .report_fav import run
        run(args.league, args.when, args.samples)
    elif cmd == "models":
        from . import ensemble, history
        from .data import load_logs
        df = ensemble.score_history(history.load("NBA"), load_logs(topup=False))
        ensemble.report(ensemble.stack_walk_forward(df))
        w = ensemble.fit_live_weights(df)
        print(f"\n  live-scan weights saved (fitted on {w['n']:,} legs): market {w['w_mid']:.2f}, model {w['w_model']:.2f}")
    elif cmd == "trades":
        from .report_trades import run
        run(args)
    elif cmd == "best":
        from .report_fav import best
        best(args)
    elif cmd == "watch":
        from .watch import run
        run(thresholds=tuple(args.thresholds), hours=args.hours)
    elif cmd == "edges":
        from . import edges, history
        edges.run(history.load(args.league))
    elif cmd == "bankroll":
        from . import bankroll, history
        try:
            hist = history.load(args.league)
        except SystemExit:
            hist = None
        bankroll.run(hist, args.bankroll, args.stake, args.bets, args.maker_cut)
    elif cmd == "mix":
        from .report_fav import mix
        mix(args)
    elif cmd == "mixes":
        from . import history, mixes
        mixes.run(history.load(args.league), "taker", samples=args.samples)
    elif cmd == "social":
        from .report_fav import social
        social(args)
    elif cmd == "sims":
        a = ap.parse_args(["scan", "--no-topup", "--top", "0"])
        res = build(a)
        if res:
            props = [l for l in res["legs"] if l.kind == "prop" and l.p_model is not None
                     and args.min_price <= l.yes_ask <= args.max_price and l.spread <= 0.05 and l.yes_bid > 0]
            from .combos import blend
            props.sort(key=lambda l: l.p_model, reverse=True)
            print(f"\n{len(props)} props priced {args.min_price:.2f}-{args.max_price:.2f}. 'sims' = share of 20,000 simulated "
                  f"games where it hit; 'best est.' weights sims and price the way history says to.")
            print(f"  {'prop':46s} {'sims':>5s} {'Kalshi':>7s} {'best est.':>9s}")
            for l in props[:args.show]:
                print(f"  {l.title[:46]:46s} {l.p_model:5.0%} {l.mid:7.0%} {blend(l):9.0%}")
    elif cmd == "card":
        from .report_fav import card
        card(args)
    elif cmd == "settle":
        print(f"settled {ledger.settle(data_client())} rows")
        ledger.report()
        from .watch import settle as settle_live
        settle_live(data_client())
    elif cmd == "report":
        ledger.report()
    elif cmd == "backtest":
        from .backtest import run
        from .data import load_logs
        from .model import build_priors
        logs = load_logs(topup=False)
        if args.rebuild_priors:
            build_priors(logs)
        run(logs, days=args.days)


if __name__ == "__main__":
    main()
