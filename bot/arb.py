"""Risk-free and near-risk-free structures inside Kalshi's own markets (no sportsbook needed).

  two-sided   the two teams of a game are separate YES markets. If YES-A + YES-B (with fees) costs under $1,
              one of them must pay $1: an arbitrage.
  ladder      a player's thresholds are separate markets (15+, 20+, 25+ points...). Buying YES on the lower
              line and NO on the higher line always pays at least $1, and $2 if he lands in between: a "middle".
              If the pair costs under $1 it is an arbitrage; otherwise its expected value is
              1 + P(in between) - cost, with P taken from the mid prices.

Kalshi's market makers keep these tight, so the usual finding is "none"; the scan is a check, and it shows what
a middle really costs after two spreads and two fees.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .book import TAKER


def _fee(p):
    return TAKER * p * (1 - p)


def scan(k, leagues=("NBA", "NFL"), hours: float = 30, include_live: bool = False) -> dict:
    from .favorites import live_legs
    parts = [live_legs(k, lg, hours_ahead=hours, include_live=include_live, max_spread=0.30).assign(league=lg) for lg in leagues]
    x = pd.concat([p for p in parts if len(p)], ignore_index=True) if any(len(p) for p in parts) else pd.DataFrame()
    out = dict(two=pd.DataFrame(), ladders=pd.DataFrame(), n_games=0, n_ladders=0, n_pairs=0)
    if x.empty:
        return out
    yes, no = x[x.side == "yes"].set_index("ticker"), x[x.side == "no"].set_index("ticker")
    # two-sided game markets
    rows = []
    g = yes[yes.series.isin(["KXNBAGAME", "KXNFLGAME"])]
    for ev, e in g.groupby("event_ticker"):
        if len(e) != 2:
            continue
        a, b = e.iloc[0], e.iloc[1]
        cost = a.cost + b.cost + _fee(a.cost) + _fee(b.cost)
        rows.append(dict(event=ev, legs=f"YES {a.title[:28]} {a.cost:.2f} + YES {b.title[:28]} {b.cost:.2f}", cost=cost, edge=1 - cost))
    out["two"], out["n_games"] = pd.DataFrame(rows), len(rows)
    # ladders: same event, same player, numeric thresholds
    t = yes.reset_index()
    parts_ = t.ticker.str.split("-")
    t["thr"] = pd.to_numeric(parts_.str[-1], errors="coerce")
    t["player"] = parts_.str[2]
    t = t[t.thr.notna() & (parts_.str.len() == 4)]
    rows = []
    for (ev, pl), e in t.groupby(["event_ticker", "player"]):
        e = e.sort_values("thr")
        if len(e) < 2:
            continue
        out["n_ladders"] += 1
        v = e[["ticker", "thr", "cost", "mid", "title"]].values
        for i in range(len(v)):
            for j in range(i + 1, len(v)):
                lo, hi = v[i], v[j]
                if hi[0] not in no.index:
                    continue
                n_cost = float(no.loc[hi[0], "cost"])
                cost = lo[2] + n_cost + _fee(lo[2]) + _fee(n_cost)
                p_mid = max(lo[3] - hi[3], 0.0)
                out["n_pairs"] += 1
                rows.append(dict(event=ev, player=lo[4].split(":")[0], low=f"YES {lo[4].split(': ')[-1]} at {lo[2]:.2f}",
                                 high=f"NO {hi[4].split(': ')[-1]} at {n_cost:.2f}", cost=cost, p_between=p_mid, ev=1 + p_mid - cost, arb=cost < 1))
    out["ladders"] = pd.DataFrame(rows)
    return out


def run(args) -> None:
    from .kalshi import data_client
    pd.set_option("display.width", 230)
    r = scan(data_client(), hours=args.hours, include_live=args.live)
    print(f"\nINSIDE-KALSHI ARBITRAGE AND MIDDLES   {r['n_games']} games, {r['n_ladders']} player ladders, {r['n_pairs']:,} low/high pairs checked")
    two = r["two"]
    if len(two):
        a = two[two.edge > 0]
        print(f"  Two-sided games: {len(a)} arbitrage(s). Cheapest pair costs {two.cost.min():.3f} for a certain $1 "
              f"(the average pair costs {two.cost.mean():.3f}: that {two.cost.mean() - 1:.1%} over $1 is the spread plus two fees).")
        for x in a.itertuples():
            print(f"     ARB  {x.legs}  cost {x.cost:.3f}")
    L = r["ladders"]
    if L.empty:
        print("  No player ladders with two-sided quotes right now (props usually post the morning of the game).")
        return
    arbs = L[L.arb]
    print(f"  Ladder pairs that cost under $1 (arbitrage): {len(arbs)}")
    for x in arbs.sort_values("cost").head(10).itertuples():
        print(f"     ARB  {x.player}: {x.low} + {x.high} = {x.cost:.3f} for at least $1")
    print(f"  Middles: average pair costs {L.cost.mean():.3f} and is worth {1 + L.p_between.mean():.3f}; expected value {L.ev.mean():+.3f} per pair. "
          f"{int((L.ev > 0).sum())} of {len(L):,} have a positive expected value at these quotes.")
    best = L.sort_values("ev", ascending=False).head(8)
    print("  Best middles right now (expected value per pair, using mid prices for the chance he lands in between):")
    for x in best.itertuples():
        print(f"     {x.ev:+.3f}  {x.player[:22]:22s} {x.low:32s} {x.high:32s} cost {x.cost:.3f}, lands between {x.p_between:.0%}")
    print("  A positive number here leans on the mid price being right on thin quotes; treat it as a lead to check, not a sure thing.")
