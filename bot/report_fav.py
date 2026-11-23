"""Printed reports for the favorites strategy: `python -m bot favorites` and `python -m bot card`."""
from __future__ import annotations

import uuid

import numpy as np
import pandas as pd

from . import history, ledger
from .favorites import (backtest_cards, build_card, build_mix, calibration, legs_table, live_legs, sweep)
from .kalshi import data_client, fee

pd.set_option("display.width", 220)
pd.set_option("display.max_rows", 200)


def _fmt(df, pct=(), f3=()):
    df = df.copy()
    for c in pct:
        df[c] = df[c].map(lambda v: f"{v:+.1%}" if pd.notna(v) else "")
    for c in f3:
        df[c] = df[c].map(lambda v: f"{v:.3f}" if pd.notna(v) else "")
    return df.to_string(index=False)


def run(league="NBA", when="pre", samples=200):
    hist = history.load(league)
    legs = legs_table(hist, when)
    print(f"\n{league} | snapshot={when} | {hist.series.nunique()} market types | {len(legs):,} priced legs "
          f"(YES at ask, NO at 1-bid) | {legs.game.nunique()} games | {legs.date.nunique()} nights "
          f"({legs.date.min():%b %d} - {legs.date.max():%b %d})")

    print("\n1) DO EXPENSIVE LEGS WIN AS OFTEN AS THEY COST?  (edge = win rate - price; net = after Kalshi fee)")
    cal = calibration(legs)
    print(_fmt(cal[["bucket", "n", "games", "price", "win", "edge", "net", "roi"]],
               pct=("edge", "net", "roi"), f3=("price", "win")))
    print("\n   by side (price >= 0.90):")
    hi = legs[legs.cost >= 0.90]
    cs = calibration(hi, by=["side"])
    print(_fmt(cs[["side", "bucket", "n", "price", "win", "net", "roi"]], pct=("net", "roi"), f3=("price", "win")))

    print("\n   by market type and side, all legs priced 0.90-0.995 (sorted by ROI; +/- is a 95% interval):")
    rows = []
    for (s, side), g in hi[hi.cost <= 0.995].groupby(["series", "side"]):
        if len(g) < 150:
            continue
        gs = g.net.groupby(g.game).sum()
        rows.append(dict(series=s, side=side, n=len(g), price=g.cost.mean(), win=g.win.mean(),
                         roi=g.net.sum() / g.cost.sum(), ci=1.96 * np.sqrt((gs ** 2).sum()) / g.cost.sum()))
    seg = pd.DataFrame(rows).sort_values("roi", ascending=False)
    print(_fmt(seg, pct=("roi", "ci"), f3=("price", "win")))

    print(f"\n2) WHOLE CARDS, night by night ({samples} random legal cards per night; one leg per event).")
    print("   hit_rate = share of cards where EVERY leg won; needed = hit rate required to break even;")
    print("   payout_x assumes the combo costs the product of leg prices (best case, no extra maker margin).")
    sw = sweep(legs, "random", samples=samples)
    print(_fmt(sw, pct=("roi", "roi_se"), f3=("hit_rate", "needed", "avg_legs_hit", "payout_x")))

    print("\n3) SAME, but picking legs by walk-forward edge (learned only from earlier nights):")
    sw2 = sweep(legs, "smart", pmins=(0.80, 0.90, 0.95))
    if len(sw2):
        print(_fmt(sw2, pct=("roi", "roi_se"), f3=("hit_rate", "needed", "avg_legs_hit", "payout_x")))
    else:
        print("   no night had enough legs with positive learned edge.")
    return legs, cal, seg, sw, sw2


def card(args):
    k = data_client()
    now = live_legs(k, args.league, include_live=args.include_live)
    if now.empty:
        print("no open legs")
        return
    try:
        hist_legs = legs_table(history.load(args.league))
    except SystemExit:
        hist_legs = pd.DataFrame()
    c = build_card(now, hist_legs, args.legs, args.min_price, args.max_price, args.per_game, smart=not args.naive)
    if c is None or c.empty:
        print("no legs in that price range")
        return
    price = float(np.prod(c.cost.values))
    p_ind = float(np.prod(c.p_hat.values))
    print(f"\n{args.league} FAVORITES CARD: {len(c)} legs priced {args.min_price:.2f}+ across "
          f"{c.game.nunique()} games ({len(now):,} legs scanned)")
    for r in c.sort_values(["game", "cost"], ascending=[True, False]).itertuples():
        print(f"  {r.side.upper():3s} {r.cost:.2f}  est {r.p_hat:.3f}  {r.title[:70]}  [{r.ticker}]")
    print(f"\n  product of leg prices : {price:.4f}  -> pays about {1 / (price + fee(min(price, .999))):.1f}x")
    print(f"  est. chance all hit   : {p_ind:.4f}  (legs treated as independent, learned win rates)")
    print(f"  est. EV per $1        : {p_ind / (price + fee(min(price, .999))) - 1:+.1%}  before any combo-maker margin")
    ledger.log_card(uuid.uuid4().hex[:8], c, f"card{len(c)}", p_ind, price,
                    note=f"min_price={args.min_price}")


def mix(args):
    """Tonight's most efficient mixes: few mid-priced legs to a target payout."""
    k = data_client()
    now = live_legs(k, args.league, include_live=args.include_live, max_spread=0.06)
    if now.empty:
        print("no open legs")
        return
    try:
        hist_legs = legs_table(history.load(args.league))
    except SystemExit:
        hist_legs = pd.DataFrame()
    run_id = uuid.uuid4().hex[:8]
    print(f"\n{args.league}: {len(now):,} legs scanned, {now.game.nunique()} games. "
          f"Legs priced {args.lo:.2f}-{args.hi:.2f}, books no wider than {args.max_spread * 100:.0f}c.")
    for target in args.targets:
        c = build_mix(now, hist_legs, target, args.lo, args.hi, args.per_game, args.max_spread)
        if c is None:
            print(f"\n~{target:g}x: not enough tight-book legs in that price range tonight")
            continue
        ask, mid, p = float(np.prod(c.cost)), float(np.prod(c.mid)), float(np.prod(c.p_hat))
        print(f"\n~{target:g}x CARD: {len(c)} legs")
        for r in c.itertuples():
            print(f"   {r.side.upper():3s} {r.cost:.2f}  hits about {r.p_hat:.0%}  {r.title[:64]}  [{r.ticker}]")
        print(f"   chance every leg hits : about {p:.1%}  (1 in {1 / p:.0f})")
        print(f"   legs are worth        : {mid:.4f} at mid prices = {1 / mid:.1f}x")
        print(f"   a fair-ish quote      : up to {mid * 1.10:.4f} ({1 / (mid * 1.10):.1f}x). Above that the maker's cut is over 10%.")
        ledger.log_card(run_id, c, f"mix{target:g}x", p, mid * 1.10, note=f"{args.lo}-{args.hi}")


def social(args):
    """Read-only look at Kalshi's public leaderboard and the named trades in tonight's markets."""
    import time
    import requests
    S = "https://api.elections.kalshi.com/v1/social"
    s = requests.Session()
    s.headers["User-Agent"] = "Mozilla/5.0"
    lb = s.get(S + "/leaderboard", params=dict(metric_name="projected_pnl", time_period=args.period,
                                               limit=args.top, category="Sports"), timeout=30).json()
    rank = {x["nickname"]: x["rank"] for x in lb.get("rank_list", []) if x.get("nickname")}
    print(f"Top {len(rank)} Sports traders by {args.period} profit loaded.")
    rows = []
    for series in args.series.split(","):
        time.sleep(1.0)
        t = s.get(S + "/trades", params=dict(series_ticker=series, page_size=100), timeout=30).json()
        for x in t.get("trades", []):
            n = x.get("taker_nickname")
            if n:
                rows.append(dict(trader=n, rank=rank.get(n), series=series, ticker=x["ticker"],
                                 side=x.get("taker_side"), action=x.get("taker_action"),
                                 price=x.get("price_dollars"), contracts=x.get("count"), time=x.get("create_date", "")[:16]))
    df = pd.DataFrame(rows)
    if df.empty:
        print("No named trades in the latest pages (most traders keep their activity private).")
        return
    top = df[df["rank"].notna()]
    print(f"{len(df)} named trades found; {len(top)} by leaderboard traders.")
    print((top if len(top) else df).sort_values("time", ascending=False).head(args.show).to_string(index=False))


def best(args):
    """`python -m bot best`: the statistically most probable combos for tonight, by payout target."""
    import json
    from . import trades as T
    from .favorites import build_best
    k = data_client()
    parts = []
    for lg in args.league.split(","):          # "NBA,NFL" builds cross-sport cards
        x = live_legs(k, lg, include_live=args.include_live, max_spread=0.06, hours_ahead=args.hours)
        if len(x):
            parts.append(x.assign(league=lg))
    if not parts:
        print("no open legs")
        return
    now = pd.concat(parts, ignore_index=True)
    if args.game:
        now = now[now.game.str.contains(args.game.upper())]
    calib = json.loads(T.CALIB.read_text()) if T.CALIB.exists() else T.leg_calibration()
    run_id = uuid.uuid4().hex[:8]
    print(f"\n{args.league}: {len(now):,} legs scanned across {now.game.nunique()} games. Legs {args.min_price:.2f}-{args.max_price:.2f}, "
          f"books within {args.max_spread * 100:.0f}c, at most {args.per_game} leg(s) per game.")
    print("'hits about' = price adjusted by how that kind of leg actually settled this season.")
    for target in args.targets:
        c = build_best(now, calib, args.league.split(",")[0], target, args.min_price, args.max_price, args.per_game,
                       args.max_spread, groups=tuple(args.groups))
        if c is None or len(c) < 2:
            print(f"\n~{target:g}x: not enough qualifying legs tonight")
            continue
        ask, mid, p = float(np.prod(c.cost)), float(np.prod(c.mid)), float(np.prod(c.p_hat))
        reached = ask <= 1 / target
        print(f"\n~{target:g}x: {len(c)} legs" + ("" if reached else f"  (only reaches {1 / ask:.1f}x with these limits tonight)"))
        for r in c.itertuples():
            print(f"   {r.side.upper():3s} {r.cost:.2f}  hits about {r.p_hat:.0%}  {r.title[:60]}  [{r.ticker}]")
        print(f"   chance every leg hits : about {p:.1%} (1 in {1 / p:.1f})")
        print(f"   legs' value           : {mid:.4f} at mid prices = {1 / mid:.1f}x fair payout")
        print(f"   do not pay more than  : {mid * 1.10:.4f} ({1 / (mid * 1.10):.1f}x)   est. value kept at that price: {p / (mid * 1.10):.0%}")
        ledger.log_card(run_id, c, f"best{target:g}x", p, mid * 1.10, note=f"best {args.min_price}-{args.max_price}")
