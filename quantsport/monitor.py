"""Open-position monitor: where every open card stands right now, and the hedge when one leg is left.

For each open position: which legs have won, which are still to play and at what price, the chance the card is still
alive (the product of the remaining legs' current prices, one game per leg so they are close to independent) and what
it is worth now. A card cannot be sold back (Kalshi combos have no exit), so the only way to take money off the table
is to buy the opposite side of a remaining leg. When exactly one leg is left the monitor prints that hedge three ways
(break-even, Kelly, full lock) with what each costs in expected value: `bot.hedge.sizes`.

`python -m quantsport positions`. Read-only: it never places anything.
"""
from __future__ import annotations

import json

from bot import book as B
from bot import hedge as H

from . import audit, store


def leg_state(leg: str, res: dict, books: dict) -> dict:
    """One leg: settled (won / lost) or still to play with the current price of our side."""
    t, side = leg.rsplit(":", 1)
    r = res.get(t)
    if r is not None:
        return dict(ticker=t, side=side, state="won" if r == side else "lost")
    bk = books.get(t)
    if not bk or not (bk.get("yes") or bk.get("no")):
        return dict(ticker=t, side=side, state="pending", mid=None, opp_ask=None)
    tp = B.top(bk)
    return dict(ticker=t, side=side, state="pending", mid=tp[f"{side}_mid"], bid=tp[f"{side}_bid"],
                opp_ask=tp["no_ask" if side == "yes" else "yes_ask"], spread=tp["spread"])


def position_status(pos: dict, legs: list[dict], bankroll: float) -> dict:
    """Roll the legs up into the card's standing. `pos` needs stake and pays."""
    won = sum(l["state"] == "won" for l in legs)
    lost = any(l["state"] == "lost" for l in legs)
    pending = [l for l in legs if l["state"] == "pending"]
    out = dict(pos, legs=legs, legs_won=won, legs_left=len(pending), dead=lost, alive=None, value=None, hedge=None, note="")
    if lost:
        out["note"] = "a leg has lost: the card is dead and will settle at zero"
        out["alive"], out["value"] = 0.0, 0.0
        return out
    if not pending:
        out["alive"], out["value"], out["note"] = 1.0, pos["pays"], "every leg has won: waiting for settlement"
        return out
    if any(l.get("mid") is None for l in pending):
        out["note"] = "no current price for a remaining leg"
        return out
    alive = 1.0
    for l in pending:
        alive *= l["mid"]
    out["alive"], out["value"] = alive, alive * pos["pays"]
    if len(pending) == 1:
        l = pending[0]
        c = H.cost_with_fee(l["opp_ask"]) if l["opp_ask"] and 0 < l["opp_ask"] < 1 else None
        if c and c < 1:
            s = H.sizes(pos["pays"], pos["stake"], c, l["mid"], max(bankroll, 0.01))
            out["hedge"] = dict(leg=l["ticker"], opposite="no" if l["side"] == "yes" else "yes", opposite_cost=c,
                                plans={k_: dict(contracts=v["contracts"], cost=v["cost"], if_hits=v["if_hits"], if_misses=v["if_misses"], ev_given_up=v["ev_given_up"])
                                       for k_, v in s.items() if k_ != "no hedge"})
            out["note"] = (f"one leg left at {l['mid']:.0%}. The opposite side costs {c * 100:.0f}c with the fee, so every hedged dollar gives up "
                           f"about {max(c - (1 - l['mid']), 0) * 100:.1f}c of expected value: hedge only to cut a swing that is too big for the bankroll")
        else:
            out["note"] = f"one leg left at {l['mid']:.0%}; no opposite side on offer to hedge with"
    else:
        out["note"] = f"{len(pending)} legs left: {alive:.1%} to cash. Nothing to hedge until one leg remains"
    return out


def status(k, con) -> list[dict]:
    pos = store.frame(con, "SELECT id, ref, label, tier, stake, pays, tickers, expires FROM positions WHERE status='open' ORDER BY expires")
    if pos.empty:
        return []
    legs = {r.id: json.loads(r.tickers) for r in pos.itertuples()}
    titles = {}                                                     # plain names for the legs, from the cards they came from
    for c in store.frame(con, "SELECT legs FROM cards WHERE card_id IN (%s)" % ",".join("?" * len(pos)), tuple(pos.ref)).legs:
        for l in json.loads(c):
            titles[(l["ticker"], l["side"])] = l.get("title", "")
    tick = sorted({l.rsplit(":", 1)[0] for v in legs.values() for l in v})
    res = audit.results_for(k, tick)
    live = [t for t in tick if t not in res]
    try:
        books = B.orderbooks(k, live) if live else {}
    except Exception as e:
        print(f"  ! order books unavailable: {str(e)[:80]}")
        books = {}
    from . import config, risk
    bankroll = risk.state(con, config.load())["cash"]
    out = []
    for r in pos.itertuples():
        ls = [leg_state(l, res, books) for l in legs[r.id]]
        for l in ls:
            l["title"] = titles.get((l["ticker"], l["side"])) or l["ticker"]
        out.append(position_status(dict(id=int(r.id), label=r.label, tier=r.tier, stake=float(r.stake), pays=float(r.pays), expires=r.expires), ls, bankroll))
    return out


def report(rows: list[dict]) -> str:
    if not rows:
        return "\nOPEN POSITIONS: none."
    L = [f"\nOPEN POSITIONS: {len(rows)} (paper). A combo cannot be sold back; the only exit is hedging the last leg."]
    for r in rows:
        worth = "" if r["value"] is None else f", {r['alive']:.1%} to cash, worth about ${r['value']:.2f} now"
        L.append(f"  {r['label']:30s} stake ${r['stake']:.2f} pays ${r['pays']:.2f}: {r['legs_won']} won, {r['legs_left']} to play{worth}")
        L.append(f"      {r['note']}")
        for l in r["legs"]:
            name = l.get("title") or l["ticker"]
            if l["state"] == "pending" and l.get("mid") is not None and not r["dead"]:
                L.append(f"        to play  {l['side'].upper():3s} {name[:60]}  now {l['mid']:.0%}")
            elif l["state"] == "lost":
                L.append(f"        LOST     {l['side'].upper():3s} {name[:60]}")
        if r["hedge"]:
            h = r["hedge"]
            L.append(f"      hedge with {h['opposite'].upper()} on {h['leg']} at {h['opposite_cost'] * 100:.0f}c a contract (fee included):")
            for name, v in h["plans"].items():
                L.append(f"        {name:10s} {v['contracts']:6.2f} contracts for ${v['cost']:.2f}: leg hits {v['if_hits']:+.2f}, misses {v['if_misses']:+.2f}, "
                         f"expected value given up ${v['ev_given_up']:.2f}")
    return "\n".join(L)
