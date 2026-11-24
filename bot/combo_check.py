"""Check a Kalshi combo before buying it: paste its link or ticker.

Looks up the combo's legs (public), prices each leg from the live order book, multiplies them into the
chance the whole card hits, and compares that with what the combo itself is selling for.

  keep = (chance it hits) / (price with fee).  1.00 = a fair bet.  0.10 = each $1 buys 10c of value.

Legs from different games are close to independent (measured: real combos with one leg per game hit 1.01x
the product of their leg prices), so the product is the right yardstick. When several legs come from the
same game the true chance can be higher (up to about 1.7x at five or more per game), and the report says so.
"""
from __future__ import annotations

import re
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from . import book as B
from .kalshi import data_client


def parse(link: str) -> str:
    """Market ticker from a kalshi.com combo link (…?marketTicker=…) or a bare ticker."""
    m = re.search(r"marketTicker=([A-Za-z0-9._-]+)", link)
    if m:
        return m.group(1).upper()
    m = re.search(r"(KXMVE[A-Z0-9]+-[A-Z0-9]+-[A-Z0-9]+)", link.upper())
    if m:
        return m.group(1)
    raise ValueError("no combo market ticker found; paste the link from the combo's page (it contains marketTicker=...)")


def check(k, ticker: str) -> dict:
    m = k.request("GET", f"/markets/{ticker}")["market"]
    legs = m.get("mve_selected_legs") or []
    if not legs:
        raise ValueError(f"{ticker} is not a combo market")
    ticks = [l["market_ticker"] for l in legs]
    ms = {}
    for i in range(0, len(ticks), 100):
        for x in k.request("GET", "/markets", params={"tickers": ",".join(ticks[i:i + 100]), "limit": 100}).get("markets") or []:
            ms[x["ticker"]] = x
    ob = B.orderbooks(k, ticks)
    now, rows, titles = datetime.now(timezone.utc), [], {}
    for l in legs:
        t, x = l["market_ticker"], ms.get(l["market_ticker"], {})
        ev = l["event_ticker"]
        if ev not in titles:
            try:
                titles[ev] = (k.request("GET", f"/events/{ev}")["event"].get("title") or ev).split(":")[0]
            except Exception:
                titles[ev] = ev
        tp = B.top(ob.get(t) or {"yes": [], "no": []})
        side = l["side"]
        bid, ask = (tp["yes_bid"], tp["yes_ask"]) if side == "yes" else (tp["no_bid"], tp["no_ask"])
        res = x.get("result") or ""
        p = (1.0 if res == side else 0.0) if res in ("yes", "no") else (bid + ask) / 2
        occ = x.get("occurrence_datetime") or x.get("expected_expiration_time")
        end = datetime.fromisoformat(occ.replace("Z", "+00:00")) if occ else None
        state = ("WON" if res == side else "LOST") if res in ("yes", "no") else (
            "playing now" if end is not None and (end - now).total_seconds() < 3 * 3600 + 900 else "not started")
        name = x.get("yes_sub_title") or x.get("title") or t
        rows.append(dict(game=titles[ev], pick=f"{'NO ' if side == 'no' else ''}{name}", price=p, bid=bid, ask=ask, state=state,
                         code=t.split("-")[1] if "-" in t else t))
    d = pd.DataFrame(rows)
    chance = float(np.prod(d.price))
    last = float(m.get("last_price_dollars") or 0)
    tr = k.request("GET", "/markets/trades", params={"ticker": ticker, "limit": 1000}).get("trades") or []
    px = pd.Series([float(t["yes_price_dollars"]) for t in tr if t.get("taker_side") == "yes"], dtype=float)
    n = pd.Series([float(t["count_fp"]) for t in tr if t.get("taker_side") == "yes"], dtype=float)
    cost = last + B.TAKER * last * (1 - last) if last > 0 else float("nan")
    return dict(ticker=ticker, legs=d, chance=chance, last=last, cost=cost, keep=chance / cost if cost == cost and cost > 0 else float("nan"),
                per_game=float(len(d) / d.code.nunique()), trades=len(px), contracts=float(n.sum()), paid=float((px * n).sum()),
                price_lo=float(px.min()) if len(px) else float("nan"), price_hi=float(px.max()) if len(px) else float("nan"),
                status=m.get("status"), result=m.get("result") or "")


def run(args) -> None:
    r = check(data_client(), parse(args.link))
    d = r["legs"].sort_values("price")
    print(f"\nCOMBO CHECK  {r['ticker']}   {len(d)} legs across {d.code.nunique()} games")
    for x in d.itertuples():
        print(f"   {x.price * 100:5.1f}c  {x.pick[:30]:30s} {x.game[:36]:36s} {x.state}")
    if (d.state == "LOST").any():
        print("\n   A leg has already lost: this combo is dead.")
        return
    print(f"\n   Chance every leg hits (leg prices multiplied): {r['chance']:.4%} = about 1 in {1 / r['chance']:,.0f}.  A fair payout would be {1 / r['chance']:,.0f}x.")
    if r["per_game"] > 1.2:
        print(f"   Legs share games ({r['per_game']:.1f} per game): same-game legs move together, so the true chance may be up to about 1.7x that.")
    if not r["last"] > 0:
        print("   The combo has not traded, so there is no price to compare. Ask for a quote and compare the payout shown with the fair payout above.")
        return
    print(f"   It sells for {r['last'] * 100:.3g}c a contract ({r['cost'] * 100:.3g}c with the fee): $1 pays ${1 / r['cost']:,.0f} if it hits.")
    print(f"   KEEP: {r['keep']:.2f}. Each $1 buys about {min(r['keep'], 9.99) * 100:.0f}c of value; the expected loss is {max(1 - r['keep'], 0):.0%} of the stake.")
    if r["trades"]:
        print(f"   So far: {r['trades']} purchases, {r['contracts']:,.0f} contracts, ${r['paid']:,.2f} paid in total, at {r['price_lo'] * 100:.3g}c to {r['price_hi'] * 100:.3g}c.")
    v = ("fair or better: rare, re-check the leg prices" if r["keep"] >= 1 else "close to fair for a combo" if r["keep"] >= 0.85 else
         "the usual cost of a combo" if r["keep"] >= 0.70 else "expensive" if r["keep"] >= 0.40 else "very expensive: most of the stake is given away at the price")
    print(f"   VERDICT: {v}.")
    weak = d[d.price < 0.5]
    if len(weak):
        print(f"   The legs that decide it: {', '.join(f'{x.pick} ({x.price * 100:.0f}c)' for x in weak.itertuples())}. Together they are a {float(np.prod(weak.price)):.1%} shot.")
