"""Printed reports for the favorites strategy: `python -m bot favorites` and `python -m bot card`."""
from __future__ import annotations

import uuid

import numpy as np
import pandas as pd

from . import history, ledger
from .favorites import (backtest_cards, build_card, calibration, legs_table, live_legs, sweep)
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
