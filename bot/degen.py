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
               max_spread: float = 0.03, rules: dict | None = None) -> pd.DataFrame:
    """`rules` = Kalshi's combo rules from slips.collection_rules: {event: (yes_only, max legs from that event)}.
    With them, legs Kalshi would refuse are dropped and several players from one prop group may be used
    (Kalshi allows that for player props; 60% of real combos do it). Without them: one leg per event."""
    from .favorites import live_legs
    from .props_study import STAT
    parts, win_prob = [], {}
    for lg in leagues:
        x = live_legs(k, lg, hours_ahead=hours, max_spread=0.06)
        if len(x):
            parts.append(x.assign(league=lg))
            if lg == "NFL":      # Kalshi's own price that each team wins: who is the underdog tonight
                ml = x[(x.series == "KXNFLGAME") & (x.side == "yes")]
                win_prob = {t.split("-")[-1]: float(m_) for t, m_ in zip(ml.ticker, ml.mid)}
    if not parts:
        return pd.DataFrame()
    c = pd.concat(parts, ignore_index=True)
    c = c[(c.side == "no") & c.series.isin(list(STAT)) & (c.cost >= lo) & (c.cost < hi) & (c.spread <= max_spread)].copy()
    form = under_form()
    c["form"] = c.series.map(form).fillna(0.0)
    c["p_hat"] = (c.mid + c.form).clip(0.02, 0.98)             # mid price nudged by that prop type's record
    c["stat"] = c.series.map(STAT)
    c["player"] = c.ticker.str.split("-").str[2]
    if rules:
        c = c[[e in rules and not (rules[e][0] and s != "yes") for e, s in zip(c.event_ticker, c.side)]].copy()
        c["ev_max"] = [rules[e][1] or 99 for e in c.event_ticker]
    else:
        c["ev_max"] = 1
    if c.empty:
        return c
    c = c.reset_index(drop=True)
    ctx = None
    try:
        from . import matchup
        ctx = matchup.context()
    except Exception as e:
        print(f"  (NBA context unavailable: {e})")
    # what is knowable before the game about each leg, and how that kind of leg has paid (bot/legs.py)
    c["pref"], c["cat_why"], c["nfl_why"] = 0.0, [[] for _ in range(len(c))], [[] for _ in range(len(c))]
    c["cats"] = [[] for _ in range(len(c))]
    try:
        from . import legs
        for lg in ("NBA", "NFL"):
            sub = c[c.league == lg]
            if sub.empty or (lg == "NBA" and ctx is None):
                continue
            f = legs.classify(legs.nba_features(sub, ctx) if lg == "NBA" else legs.nfl_features(sub, win_prob), lg)
            c.loc[f.index, "pref"] = f.pref
            for i, w, names in zip(f.index, f.cat_why, f.cats_all):
                c.at[i, "cat_why"] = w
                c.at[i, "cats"] = names
            if lg == "NFL":
                for i, r in zip(f.index, f.itertuples()):
                    c.at[i, "nfl_why"] = legs.nfl_why(r)
    except Exception as e:
        print(f"  (leg categories unavailable: {e})")
    # the logical chain for every leg: NBA from form, minutes, matchup scenario, pace, rest; NFL from his game log and the game line
    pro, con, why = [], [], []
    for r in c.itertuples():
        if r.league == "NBA" and ctx is not None:
            try:
                e = matchup.explain(ctx, r.title, r.series, r.ticker, r.game, r.side, r.cost)
            except Exception as ex:
                e = dict(lines=[f"(chain failed: {ex})"], pro=0, con=0)
            pro.append(e["pro"]); con.append(e["con"]); why.append(" | ".join(list(e["lines"][:7]) + list(r.cat_why)))
        else:
            pro.append(0); con.append(0); why.append(" | ".join(r.nfl_why) if r.nfl_why else "")
    c["pro"], c["con"], c["why"] = pro, con, why
    c["chain"] = c.pro - c.con
    c = c.drop(columns=["cat_why", "nfl_why"])
    c = c[c.pref >= 0]                                      # drop legs in a category that has trailed the baseline in both seasons
    return c[~((c.con >= 2) & (c.con > c.pro))]            # drop legs where the chain clearly says OVER


def build(c: pd.DataFrame, need: float, skip: set | None = None, max_per_game: int = 3, max_per_stat: int = 4,
          seed: int = 0) -> pd.DataFrame | None:
    """Fewest legs whose fair payout is at least need / CUSHION. Best prop-type record first, then tightest
    book; `skip` holds tickers already used so several different cards can be dealt in one night."""
    rng = np.random.default_rng(seed)
    c = c[~c.ticker.isin(skip or set())].copy()
    c["_r"] = rng.random(len(c))
    if "pref" not in c:
        c["pref"] = 0.0
    c = c.sort_values(["pref", "chain", "form", "spread", "_r"], ascending=[False, False, False, True, True])   # preferred categories first
    ev, who, gm, st, keep, price, thr = {}, set(), {}, {}, [], 1.0, CUSHION / need
    rows = list(c.itertuples())

    def ok(r):      # Kalshi's per-event limit, one leg per player (so legs stay close to independent), spread over games
        return not (ev.get(r.event_ticker, 0) >= getattr(r, "ev_max", 1) or (r.game, r.player) in who
                    or gm.get(r.game, 0) >= max_per_game or st.get(r.series, 0) >= max_per_stat)

    def take(i, r):
        nonlocal price
        ev[r.event_ticker] = ev.get(r.event_ticker, 0) + 1; who.add((r.game, r.player))
        gm[r.game] = gm.get(r.game, 0) + 1; st[r.series] = st.get(r.series, 0) + 1
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
