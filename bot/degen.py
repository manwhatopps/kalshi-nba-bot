"""Degen mode: the biggest-payout cards the data can defend, with honest odds.

What two seasons of settled props say about lottery cards (FINDINGS 14 and 16):
  * build from UNDERS, never overs: pre-game overs hit 1-5 points less often than they cost, unders
    about what they cost; ~50x cards of unders kept 92-99% of fair value, cards of overs 82-86%;
  * legs priced 50-75c on tight books; nothing at 95c+; no long-shot legs;
  * one leg per event (Kalshi's rule), a few per game at most, spread across games;
  * only buy if the quote pays at least the target - the card is built with a 25% cushion over it.
No card here has a positive expected value. This command makes the tickets as good as they can be
and says how likely each is to hit.
"""
from __future__ import annotations

import json
import uuid

import numpy as np
import pandas as pd

from . import config, fund, ledger
from .kalshi import data_client

CUSHION = 0.8        # build the card so the target payout is 80% of its fair payout


def under_form() -> dict:
    """Per prop type: how pre-game unders priced 50-75c actually did over both seasons (hit - paid),
    shrunk toward zero when there are few legs. Used only to order otherwise-equal legs."""
    try:
        from .props_study import load_pre
        d = load_pre()
        d = d[(d.pick == "UNDER") & (d.cost >= 0.50) & (d.cost < 0.75)]
        g = d.groupby("series").agg(n=("win", "size"), gap=("win", "mean"), paid=("cost", "mean"))
        return ((g.gap - g.paid) * g.n / (g.n + 2000)).to_dict()
    except Exception:
        return {}


def candidates(k, leagues=("NBA", "NFL"), hours: float = 14, lo: float = 0.50, hi: float = 0.75,
               max_spread: float = 0.03) -> pd.DataFrame:
    from .favorites import live_legs
    from .props_study import STAT
    parts = []
    for lg in leagues:
        x = live_legs(k, lg, hours_ahead=hours, max_spread=0.06)
        if len(x):
            parts.append(x.assign(league=lg))
    if not parts:
        return pd.DataFrame()
    c = pd.concat(parts, ignore_index=True)
    c = c[(c.side == "no") & c.series.isin(list(STAT)) & (c.cost >= lo) & (c.cost < hi) & (c.spread <= max_spread)].copy()
    form = under_form()
    c["form"] = c.series.map(form).fillna(0.0)
    c["p_hat"] = (c.mid + c.form).clip(0.02, 0.98)             # mid price nudged by that prop type's record
    c["stat"] = c.series.map(STAT)
    # the logical chain for every NBA leg: recent form vs the line, minutes, defense-vs-role, pace, rest
    try:
        from . import matchup
        ctx = matchup.context()
        ex = [matchup.explain(ctx, r.title, r.series, r.ticker, r.game, r.side, r.cost) if r.league == "NBA"
              else dict(lines=["NFL leg: no style model yet"], pro=0, con=0) for r in c.itertuples()]
        c["pro"], c["con"] = [e["pro"] for e in ex], [e["con"] for e in ex]
        c["why"] = [" | ".join(e["lines"][:3]) for e in ex]
    except Exception as e:
        print(f"  (chain unavailable: {e})")
        c["pro"], c["con"], c["why"] = 0, 0, ""
    c["chain"] = c.pro - c.con
    return c[~((c.con >= 2) & (c.con > c.pro))]            # drop legs where the chain clearly says OVER


def build(c: pd.DataFrame, need: float, skip: set | None = None, max_per_game: int = 3, max_per_stat: int = 4,
          seed: int = 0) -> pd.DataFrame | None:
    """Fewest legs whose fair payout is at least need / CUSHION. Best prop-type record first, then tightest
    book; `skip` holds tickers already used so several different cards can be dealt in one night."""
    rng = np.random.default_rng(seed)
    c = c[~c.ticker.isin(skip or set())].copy()
    c["_r"] = rng.random(len(c))
    c = c.sort_values(["chain", "form", "spread", "_r"], ascending=[False, False, True, True])
    ev, gm, st, keep, price, thr = set(), {}, {}, [], 1.0, CUSHION / need
    rows = list(c.itertuples())

    def ok(r):
        return not (r.event_ticker in ev or gm.get(r.game, 0) >= max_per_game or st.get(r.series, 0) >= max_per_stat)

    def take(i, r):
        nonlocal price
        ev.add(r.event_ticker); gm[r.game] = gm.get(r.game, 0) + 1; st[r.series] = st.get(r.series, 0) + 1
        keep.append(i); price *= r.mid

    for i, r in enumerate(rows):
        if not ok(r):
            continue
        if price * 0.5 <= thr:        # one more leg can finish the card: take the dearest leg that still gets there,
            last = [(j, x) for j, x in enumerate(rows) if j not in keep and ok(x) and price * x.mid <= thr]
            if last:                  # so the card does not overshoot the payout (and give away hit chance)
                j, x = max(last, key=lambda t: t[1].mid)
                take(j, x)
                return c.iloc[keep]
        take(i, r)
        if price <= thr:
            return c.iloc[keep]
    return None


def run(args):
    cfg = fund.load_cfg()
    tickets = [tuple(t) for t in (json.loads(args.tickets) if args.tickets else cfg.get("degen_mode", [[10, 1000], [5, 1000], [5, 5000]]))]
    k = data_client()
    c = candidates(k, hours=args.hours)
    print(f"\nDEGEN MODE  {len(c)} qualifying UNDER legs tonight (50-75c, books within 3c) across "
          f"{c.game.nunique() if len(c) else 0} games, after dropping legs where the chain clearly points to the over.")
    print("  'chain' = how many of the matchup points (recent form vs the line, minutes, what this defense gives this kind of player,")
    print("  pace, rest) support the under. Tested on 68,262 settled props, the chain did NOT beat Kalshi's price; it is context, not an edge.")
    if c.empty:
        print("  nothing qualifies right now (props usually post the morning of the game)")
        return
    run_id, used, p_miss, spend = uuid.uuid4().hex[:8], set(), 1.0, 0.0
    for n, (stake, goal) in enumerate(tickets, 1):
        need = goal / stake
        card = build(c, need, skip=used, seed=n)
        if card is None:
            print(f"\n  TICKET {n}: ${stake:g} -> ${goal:,.0f} ({need:,.0f}x): not enough different legs left tonight")
            continue
        used |= set(card.ticker)
        mid, p = float(np.prod(card.mid)), float(np.prod(card.p_hat))
        p_miss *= 1 - p
        spend += stake
        print(f"\n  TICKET {n}: ${stake:g} to win ${goal:,.0f}   {len(card)} legs across {card.game.nunique()} games   "
              f"(all UNDERS = buy NO)")
        for r in card.sort_values("game").itertuples():
            print(f"      NO {r.cost:.2f}  {r.title[:60]}   [chain: {r.pro} for, {r.con} against]")
            if args.why and r.why:
                for line in r.why.split(" | "):
                    print(f"            - {line}")
        print(f"      fair payout {1 / mid:,.0f}x.  Chance it hits: about 1 in {1 / p:,.0f}.")
        print(f"      BUY ONLY IF the quote pays {need:,.0f}x or more (${stake:g} returns ${goal:,.0f}). "
              f"Expected loss at that quote: ${stake * (1 - need * p):.2f}.")
        ledger.log_card(run_id, card, f"degen_{int(goal)}", p, 1 / need, stake=stake, note=f"degen mode {stake}->{goal}")
    p_night = 1 - p_miss
    print(f"\n  TONIGHT: ${spend:g} of tickets, about a {p_night:.1%} chance that at least one hits (1 night in {1 / max(p_night, 1e-9):,.0f}).")
    print(f"  A WEEK of nights like this: ${spend * 7:g} spent, about a {1 - (1 - p_night) ** 7:.0%} chance of at least one hit, "
          f"about a {(1 - p_night) ** 7:.0%} chance of none.")
    print("  Those odds assume the quotes meet the buy-only-if payout. If a quote is lower, skip the ticket: do not chase it.")
