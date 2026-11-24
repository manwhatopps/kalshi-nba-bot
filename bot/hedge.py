"""Hedging: what it costs, when it is worth it, and the exact order to place.

On an exchange a hedge is simply buying the other side. For a card (combo) that is still alive, the other
side of "the next leg hits" is that leg's opposite contract in its own market, bought before that game starts,
so hedging fits a pre-game-only routine: the early games finish, the card is alive, the late game has not begun.

Three sizes are printed, for a ticket that pays W if the next leg hits and whose opposite contract costs c
(fee included) and pays $1:

  lock          buy W contracts: the same amount is banked whichever way the leg goes, W x (1 - c) less the stake
  break-even    buy stake / (1 - c) contracts: the stake comes back if the leg misses, the rest rides
  Kelly         the size that maximises the expected growth of the WHOLE bankroll B:
                h = (1 - p)(B + W)/c - p B/(1 - c), never below 0 or above W
                where p is the leg's fair chance. At a fair price (c = 1 - p) this is W: hedge fully.
                Every cent of spread and fee pulls it down, and for a ticket that is small next to the
                bankroll it is 0: do not hedge.

A hedge never adds expected value. It pays the spread and a second fee to make the result less swingy, which is
worth something only when the ticket has become a large share of the bankroll. The Kelly line is that trade-off
done properly. With several legs still to play at different times, W is the ticket's value if the next leg hits
(the payout times the fair chance of the legs after it), and the same sum is done again before each later game.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd

from . import book as B
from . import fund

FEE = B.TAKER


def cost_with_fee(price: float) -> float:
    return price + FEE * price * (1 - price)


def sizes(W: float, stake: float, c: float, p: float, bankroll: float) -> dict:
    """Contracts of the opposite side for each hedge, and what each leaves. W = value if the leg hits, c = opposite's
    cost per contract with fee, p = fair chance the leg hits, bankroll = money outside this ticket."""
    out = {}
    kelly = float(np.clip((1 - p) * (bankroll + W) / c - p * bankroll / (1 - c), 0, W)) if 0 < c < 1 else 0.0
    for name, h in (("no hedge", 0.0), ("break-even", min(stake / (1 - c), W) if c < 1 else 0.0), ("Kelly", kelly), ("lock", W)):
        hit, miss = W - h * c - stake, h * (1 - c) - stake
        out[name] = dict(contracts=h, cost=h * c, if_hits=hit, if_misses=miss, expected=p * hit + (1 - p) * miss,
                         growth=p * math.log(max(bankroll + stake + hit, 1e-9)) + (1 - p) * math.log(max(bankroll + stake + miss, 1e-9)))
    base = out["no hedge"]["expected"]
    for v in out.values():
        v["ev_given_up"] = base - v["expected"]
    return out


def plan(k, link: str, stake: float, pays: float, bankroll: float | None = None) -> dict:
    """Read a combo's legs live and work out the hedge before its next game."""
    from .combo_check import check, parse
    r = check(k, parse(link))
    d = r["legs"]
    bankroll = fund.load_cfg()["bankroll"] if bankroll is None else bankroll
    m = k.request("GET", f"/markets/{r['ticker']}")["market"]
    legs = m.get("mve_selected_legs") or []
    d = d.assign(ticker=[l["market_ticker"] for l in legs], side=[l["side"] for l in legs])
    out = dict(ticker=r["ticker"], legs=d, stake=stake, pays=pays, bankroll=bankroll, dead=bool((d.state == "LOST").any()),
               won=int((d.state == "WON").sum()), sellback=float(m.get("yes_bid_dollars") or 0))
    pend = d[~d.state.isin(["WON", "LOST"])]
    out["pending"] = pend
    if out["dead"] or pend.empty:
        return out
    live = pend[pend.state == "playing now"]
    nxt = pend[pend.state == "not started"]
    out["live_legs"] = len(live)
    if nxt.empty:
        return out
    # the next decision: the earliest game that has not started. Legs in the same game are hedged together only roughly.
    first = nxt.iloc[0]
    same = nxt[nxt.code == first.code]
    later = pend[~pend.index.isin(same.index)]
    out["next"], out["same_game"] = first, len(same)
    p_next = float(np.prod(same.price))
    W = pays * float(np.prod(later.price)) if len(later) else pays         # what the ticket is worth if this game's legs hit
    ob = B.orderbooks(k, [first.ticker]).get(first.ticker) or {"yes": [], "no": []}
    opp = "no" if first.side == "yes" else "yes"
    offers = B.offers(ob, opp)
    ask = offers[0][0] if offers else float("nan")
    out.update(W=W, p_next=p_next, opp=opp, opp_ask=ask, opp_size=sum(q for _, q in offers[:3]), value_now=W * p_next,
               sizes=sizes(W, stake, cost_with_fee(ask), p_next, bankroll) if ask == ask and len(same) == 1 else None)
    return out


def run(args) -> None:
    from .kalshi import data_client
    money = lambda v: f"${v:,.2f}"
    if not args.link:                      # the sums alone, no market lookup
        c, p = cost_with_fee(args.price), 1 - args.price if args.chance is None else args.chance
        s = sizes(args.pays, args.stake, c, p, args.bankroll if args.bankroll is not None else fund.load_cfg()["bankroll"])
        print(f"\nHEDGE SUMS  ticket pays {money(args.pays)} if the leg hits (stake {money(args.stake)}); the other side costs {args.price * 100:.0f}c "
              f"({c * 100:.1f}c with the fee); leg's chance {p:.0%}")
        _table(s, money)
        return
    r = plan(data_client(), args.link, args.stake, args.pays, args.bankroll)
    d = r["legs"]
    print(f"\nHEDGE  {r['ticker']}   stake {money(r['stake'])}, pays {money(r['pays'])}   {r['won']} of {len(d)} legs already won")
    if r["dead"]:
        print("   A leg has lost. The ticket is dead: there is nothing to hedge.")
        return
    if r["pending"].empty:
        print("   Every leg has won. Nothing left to hedge.")
        return
    for x in r["pending"].itertuples():
        print(f"   still to come: {x.pick[:34]:34s} {x.game[:30]:30s} {x.price * 100:5.1f}c  {x.state}")
    if "next" not in r:
        print("   Every remaining leg is in a game that has already started. Pre-game only: no hedge to place now.")
        return
    n = r["next"]
    print(f"\n   Ticket is worth about {money(r['value_now'])} now (payout x the fair chance of what is left).")
    print(f"   Next game not yet started: {n.game}. If its leg hits the ticket is worth about {money(r['W'])}; chance {r['p_next']:.0%}.")
    if r["sellback"] > 0:
        print(f"   Selling the combo itself: best bid {r['sellback'] * 100:.3g}c a contract.")
    if r["same_game"] > 1:
        print(f"   {r['same_game']} legs are in that one game, so no single contract is the exact opposite. Hedge the leg most likely to miss, or leave it.")
        return
    if r["sizes"] is None:
        print(f"   Nobody is offering the other side ({r['opp'].upper()} on {n.pick}) right now: no hedge available.")
        return
    print(f"   The hedge: BUY {r['opp'].upper()} on '{n.pick}' at up to {r['opp_ask'] * 100:.0f}c ({r['opp_size']:.0f} contracts offered near that price), before that game starts.")
    _table(r["sizes"], money)


def _table(s: dict, money) -> None:
    print(f"\n   {'':12s} {'contracts':>9s} {'hedge cost':>11s} {'if leg hits':>12s} {'if it misses':>13s} {'expected':>10s} {'given up':>9s}")
    for name, v in s.items():
        print(f"   {name:12s} {v['contracts']:9.0f} {money(v['cost']):>11s} {money(v['if_hits']):>12s} {money(v['if_misses']):>13s} "
              f"{money(v['expected']):>10s} {money(v['ev_given_up']):>9s}")
    best = max(s, key=lambda n_: s[n_]["growth"])
    k_ = s["Kelly"]
    print(f"   'given up' is the expected value the hedge costs (spread and a second fee). For your bankroll the growth-best line is '{best}'"
          + (f": buy {k_['contracts']:.0f} contracts." if best == "Kelly" and k_["contracts"] >= 1 else "."))
    if k_["contracts"] < 1:
        print("   The ticket is small next to the bankroll, so hedging only costs money here. Let it ride.")
