"""Bankroll math for long-shot cards: what to expect, how much to stake, when to stop.

Inputs are measured, not assumed: for each leg-price band the settled history gives the hit rate
vs price, which fixes how often a card built from that band really hits at a given payout
("keep" = real hit rate / break-even hit rate). From that:

  kelly_fraction   growth-optimal share of bankroll (0 when the bet is not +EV)
  simulate         Monte-Carlo bankroll paths for flat stakes: chance of being ahead, drawdowns,
                   losing streaks, chance of going broke
  bets_to_decide   Wald's sequential test: how many bets before results can tell a winning
                   strategy (keep 1.10) from a losing one (keep 0.90)
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd


def kelly_fraction(p: float, price: float) -> float:
    """Binary contract bought at `price` paying 1. Growth-optimal fraction of bankroll to stake."""
    return max(0.0, (p - price) / (1 - price)) if 0 < price < 1 else 0.0


def kelly_simultaneous(ps, prices, grid: int = 200) -> float:
    """Common fraction f staked on EACH of several independent bets placed at once (numeric max of
    expected log growth via simulation). Shows why stakes must shrink when many cards run together."""
    ps, prices = np.asarray(ps, float), np.asarray(prices, float)
    rng = np.random.default_rng(0)
    wins = rng.random((20000, len(ps))) < ps
    payoff = np.where(wins, 1 / prices - 1, -1.0)
    best, best_g = 0.0, 0.0
    for f in np.linspace(0, 1 / len(ps), grid)[1:]:
        w = 1 + f * payoff.sum(1)
        if (w <= 0).any():
            break
        g = np.log(w).mean()
        if g > best_g:
            best, best_g = f, g
    return float(best)


def simulate(p: float, payout: float, stake: float, bankroll: float, n_bets: int, paths: int = 20000,
             seed: int = 1) -> dict:
    """Flat `stake` per bet; a win returns stake*payout. Stops betting when the bankroll can't cover a stake."""
    rng = np.random.default_rng(seed)
    wins = rng.random((paths, n_bets)) < p
    pnl = np.where(wins, stake * (payout - 1), -stake)
    bal = bankroll + np.cumsum(pnl, axis=1)
    broke = (bal < stake)
    first_broke = np.where(broke.any(1), broke.argmax(1), n_bets)
    idx = np.arange(n_bets)[None, :]
    bal = np.where(idx > first_broke[:, None], np.take_along_axis(bal, np.minimum(first_broke, n_bets - 1)[:, None], 1), bal)
    final = bal[:, -1]
    peak = np.maximum.accumulate(np.concatenate([np.full((paths, 1), bankroll), bal], 1), 1)[:, 1:]
    streak = np.zeros(paths, int); cur = np.zeros(paths, int)
    for j in range(n_bets):
        cur = np.where(wins[:, j], 0, cur + 1)
        streak = np.maximum(streak, cur)
    return dict(ahead=float((final > bankroll).mean()), median_final=float(np.median(final)),
                p10_final=float(np.percentile(final, 10)), p90_final=float(np.percentile(final, 90)),
                broke=float(broke.any(1).mean()), no_hits=float((~wins.any(1)).mean()),
                worst_drawdown_med=float(np.median((peak - bal).max(1))),
                losing_streak_med=int(np.median(streak)), losing_streak_p95=int(np.percentile(streak, 95)),
                expected=float(bankroll + n_bets * stake * (p * payout - 1)))


def bets_to_decide(payout: float, keep_good: float = 1.10, keep_bad: float = 0.90, alpha: float = 0.05) -> float:
    """Expected number of bets for a sequential test (5% error rates) to tell a card type that keeps
    110% of fair value from one that keeps 90%, if the truth is the better one."""
    p1, p0 = keep_good / payout, keep_bad / payout
    a = math.log((1 - alpha) / alpha)
    step = p1 * math.log(p1 / p0) + (1 - p1) * math.log((1 - p1) / (1 - p0))
    return a * (1 - 2 * alpha) / step


def run(hist: pd.DataFrame | None, bankroll: float, stake: float, n_bets: int, maker_cut: float = 0.10):
    from .favorites import legs_table
    from .mixes import leg_efficiency
    pd.set_option("display.width", 220)
    rows = []
    if hist is not None:
        le = leg_efficiency(legs_table(hist)).set_index("band")
        bands = [("35-50c", "35-50c legs"), ("60-70c", "60-70c legs"), ("90-95c", "90-95c legs"), ("95-99c", "95-99c legs")]
        for band, label in bands:
            if band not in le.index:
                continue
            k10, per_leg = le.loc[band, "value_kept_at_10x"], -math.log(le.loc[band, "price"])
            for payout in (5, 10, 25, 50, 100):
                keep = k10 ** (math.log(payout) / math.log(10)) / (1 + maker_cut)
                rows.append((f"{payout}x from {label}", payout, keep, math.log(payout) / per_leg))
    else:
        rows = [(f"{m}x at fair odds less {maker_cut:.0%}", m, 1 / (1 + maker_cut), float("nan")) for m in (5, 10, 25, 50, 100)]
    out = []
    for name, payout, keep, legs in rows:
        p = min(keep / payout, 0.999)
        s = simulate(p, payout, stake, bankroll, n_bets)
        out.append(dict(card=name, legs=legs, hits=f"1 in {1 / p:.0f}", keep=keep,
                        kelly_stake=round(kelly_fraction(p, 1 / payout) * bankroll, 2),
                        ahead=s["ahead"], broke=s["broke"], no_hits=s["no_hits"], median_end=s["median_final"],
                        bad_end=s["p10_final"], good_end=s["p90_final"],
                        typical_streak=s["losing_streak_med"], bad_streak=s["losing_streak_p95"],
                        bets_to_prove=bets_to_decide(payout)))
    df = pd.DataFrame(out)
    print(f"\nBANKROLL ${bankroll:g}, flat ${stake:g} per card, {n_bets} cards. keep = share of fair value after the "
          f"measured per-leg cost and a {maker_cut:.0%} maker cut.")
    print("ahead/broke/no_hits = chance of ending above the start / unable to cover a stake / never hitting once.")
    print("median_end, bad_end (1 in 10), good_end (1 in 10) = bankroll after the run. streaks = losing cards in a row.")
    fmt = df.copy()
    for c in ("ahead", "broke", "no_hits"):
        fmt[c] = fmt[c].map("{:.0%}".format)
    for c in ("median_end", "bad_end", "good_end"):
        fmt[c] = fmt[c].map("${:,.0f}".format)
    fmt["keep"] = fmt.keep.map("{:.0%}".format)
    fmt["legs"] = fmt.legs.map(lambda v: f"{v:.0f}" if v == v else "")
    fmt["bets_to_prove"] = fmt.bets_to_prove.map("{:,.0f}".format)
    print(fmt.to_string(index=False))
    print("\nkelly_stake is $0 whenever keep is below 100%: the growth-optimal bet on a negative-expectation card is nothing.")
    print("bets_to_prove = cards needed before results can separate a winning strategy (keeps 110%) from a losing one (90%).")
    return df
