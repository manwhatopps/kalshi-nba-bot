"""Turn legs + simulations into ranked combos.

Pricing logic (the honest version):
  * p_fair   = blend of stats model, de-vigged sportsbook (optional) and Kalshi mid.
  * joint    = same-game legs: Monte-Carlo with the model's correlation structure;
               different games: independent (multiplied).
  * est_price= what a Kalshi combo market maker would likely quote: the joint probability using
               Kalshi MID prices as marginals (same correlation) plus COMBO_VIG. Makers price
               correlation too, so the only real edge is where our marginals beat Kalshi's.
  * The real price comes from an RFQ quote (`place` command) — always trust that over est_price.
"""
from __future__ import annotations

import math
import os
from collections import defaultdict
from dataclasses import dataclass

import numpy as np

from . import config
from .kalshi import fee
from .model import leg_values, leg_line

COMBO_VIG = float(os.getenv("COMBO_VIG", "0.10"))


@dataclass
class Combo:
    legs: list
    p_fair: float
    p_mid: float
    est_price: float
    corr_lift: float

    @property
    def mult(self) -> float:
        return 1.0 / self.est_price

    @property
    def ev(self) -> float:
        cost = self.est_price + fee(self.est_price)
        return self.p_fair / cost - 1

    flat_stake: float = 0.0  # if set, overrides Kelly ("lottery ticket" mode), still capped

    def stake(self) -> float:
        if self.flat_stake > 0:
            return round(min(self.flat_stake, config.MAX_STAKE_PER_COMBO), 2) if self.ev > 0 else 0.0
        b = self.mult - 1
        f = (self.p_fair * (b + 1) - 1) / b if b > 0 else 0
        return round(max(0.0, min(config.KELLY_FRACTION * f * config.BANKROLL, config.MAX_STAKE_PER_COMBO)), 2)


_STACK = None


def _stack_weights():
    global _STACK
    if _STACK is None:
        import json
        f = config.DATA_DIR / "stack_weights.json"
        _STACK = json.loads(f.read_text()) if f.exists() else {}
    return _STACK


def blend(leg) -> float:
    """Fair probability. For player props, if `python -m bot models` has been run, use the weights it
    learned from Kalshi's settled markets (which put ~all the weight on the market price) instead of
    the hand-set W_MODEL."""
    w = _stack_weights()
    # Applied to game lines too: the team-score model has not yet beaten the market in a backtest,
    # so it gets no more trust than the prop models earned.
    if w and leg.p_model is not None and leg.p_book is None:
        lg = lambda p: math.log(min(max(p, 0.005), 0.995) / (1 - min(max(p, 0.005), 0.995)))
        z = w["b0"] + w["w_mid"] * lg(leg.mid) + w["w_model"] * lg(leg.p_model)
        return float(min(max(1 / (1 + math.exp(-z)), 0.003), 0.997))
    w_m = config.W_MODEL if leg.p_model is not None else 0
    w_b = config.W_BOOK if leg.p_book is not None else 0
    w_k = max(0.0, 1 - w_m - w_b)
    tot = w_m + w_b + w_k
    p = (w_m * (leg.p_model or 0) + w_b * (leg.p_book or 0) + w_k * leg.mid) / tot
    return float(min(max(p, 0.003), 0.997))


class GameSims:
    """Per-game rank matrices so any subset of same-game legs can be evaluated fast."""

    def __init__(self, legs, sim, seed=11):
        rng = np.random.default_rng(seed)
        self.idx = {}
        ranks = []
        for leg in legs:
            v = leg_values(leg, sim)
            if v is None:
                continue
            line = leg_line(leg)
            leg.p_model = float(np.clip((v > line).mean(), 0.003, 0.997))
            jitter = rng.uniform(-0.49, 0.49, len(v)) if leg.kind == "prop" else 0
            r = np.argsort(np.argsort(v + jitter, kind="stable"), kind="stable")
            self.idx[leg.ticker] = len(ranks)
            ranks.append(r)
        self.R = np.array(ranks) if ranks else np.zeros((0, 0))
        self.n = self.R.shape[1] if ranks else 0

    def hits(self, ticker, p):
        r = self.R[self.idx[ticker]]
        return r >= self.n * (1 - p)

    def joint(self, tickers, ps):
        m = np.ones(self.n, bool)
        for t, p in zip(tickers, ps):
            m &= self.hits(t, p)
        return float(m.mean())


def evaluate(legs, gsims: dict) -> tuple[float, float]:
    by_game = defaultdict(list)
    for l in legs:
        by_game[l.game].append(l)
    pf = pm = 1.0
    for g, ls in by_game.items():
        gs = gsims[g]
        t = [l.ticker for l in ls]
        if len(ls) == 1:
            pf *= ls[0].p_fair
            pm *= ls[0].mid
        else:
            pf *= max(gs.joint(t, [l.p_fair for l in ls]), 1.0 / gs.n)
            pm *= max(gs.joint(t, [l.mid for l in ls]), 1.0 / gs.n)
    return pf, pm


def make_combo(legs, gsims) -> Combo:
    pf, pm = evaluate(legs, gsims)
    indep = math.prod(l.p_fair for l in legs)
    price = min(0.99, pm * (1 + COMBO_VIG))
    return Combo(list(legs), pf, pm, price, pf / indep if indep else 1)


def candidate_legs(legs, max_pool=40, max_disagree=0.20, max_spread=0.08, min_ask=0.15, max_ask=0.92,
                   min_edge=0.02):
    pool = []
    for l in legs:
        if l.p_model is None:
            continue
        if not (min_ask <= l.yes_ask <= max_ask) or l.spread > max_spread or l.yes_bid <= 0:
            continue
        if abs(l.p_model - l.mid) > max_disagree:
            l.notes.append("model and market disagree a lot (injury/news?) - skipped")
            continue
        l.p_fair = blend(l)
        if l.p_fair - l.mid >= min_edge:
            pool.append(l)
    # rank by edge in standard-deviation units so long shots don't dominate on ratio alone
    pool.sort(key=lambda l: (l.p_fair - l.mid) / math.sqrt(l.mid * (1 - l.mid)), reverse=True)
    return pool[:max_pool]


def search(pool, gsims, event_limit: dict, min_legs=2, max_legs=6, min_mult=5.0, beam=250, top=15,
           max_mult=1000.0, same_game=False, flat_stake=0.0):
    """Beam search over leg sets. event_limit: event_ticker -> max legs allowed from that event."""
    def ok(legs):
        if same_game and len({l.game for l in legs}) > 1:
            return False
        cnt = defaultdict(int)
        players = set()
        for l in legs:
            cnt[l.event_ticker] += 1
            if cnt[l.event_ticker] > event_limit.get(l.event_ticker, 0):
                return False
            if l.kind == "prop":
                k = (l.player, l.stats)
                if k in players:
                    return False
                players.add(k)
        return True

    level = [[l] for l in pool if event_limit.get(l.event_ticker, 0) >= 1]
    results, seen = [], set()
    for size in range(2, max_legs + 1):
        nxt = []
        for base in level:
            last = pool.index(base[-1])
            for l in pool[last + 1:]:
                cand = base + [l]
                key = frozenset(x.ticker for x in cand)
                if key in seen or not ok(cand):
                    continue
                seen.add(key)
                c = make_combo(cand, gsims)
                nxt.append(c)
        nxt.sort(key=lambda c: c.ev, reverse=True)
        level = [c.legs for c in nxt[:beam]]
        if size >= min_legs:
            results += [c for c in nxt[:beam] if min_mult <= c.mult <= max_mult and c.ev > 0]
    results.sort(key=lambda c: c.ev, reverse=True)
    out, used = [], defaultdict(int)
    for c in results:  # diversify: don't show 15 near-copies of the same combo
        if any(used[l.ticker] >= 3 for l in c.legs):
            continue
        for l in c.legs:
            used[l.ticker] += 1
        c.flat_stake = flat_stake
        out.append(c)
        if len(out) >= top:
            break
    return out
