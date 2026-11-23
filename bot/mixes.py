"""Which MIX of legs is the most efficient way to reach a target payout?

Recipes are compared at the same payout (e.g. ~10x): a few legs at 50-70c, many legs at 90c+,
or a blend. For each night in Kalshi's settled history we deal many random legal cards per
recipe (one leg per event, capped per game) and record how often EVERY leg hit.

  hit_rate   how often the whole card hit (the "will it hit" number)
  fair_rate  how often it would need to hit for the payout to be fair (1 / payout)
  keep       hit_rate / fair_rate: the share of fair value you keep. 1.00 = fair bet, 0.80 = you
             are paying 20% over the odds. Higher is better; this is the efficiency of the mix.
Also `leg_efficiency`: per price band, how much value one leg loses per unit of payout it adds.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .favorites import legs_table

BANDS = {"35-50c": (0.35, 0.50), "50-60c": (0.50, 0.60), "60-70c": (0.60, 0.70), "70-80c": (0.70, 0.80),
         "80-90c": (0.80, 0.90), "90-95c": (0.90, 0.95), "95-99c": (0.95, 0.995)}
RECIPES = {  # name -> list of (band, share of the card's legs); legs are added until the target payout
    "all 35-50c": [((0.35, 0.50), 1)],
    "all 50-70c": [((0.50, 0.70), 1)],
    "all 70-85c": [((0.70, 0.85), 1)],
    "all 85-93c": [((0.85, 0.93), 1)],
    "all 90c+": [((0.90, 0.995), 1)],
    "all 95c+": [((0.95, 0.995), 1)],
    "half 50-70c, half 90c+": [((0.50, 0.70), 1), ((0.90, 0.995), 1)],
    "one 50-70c anchor + 90c+": [((0.50, 0.70), 0), ((0.90, 0.995), 1)],  # share 0 = exactly one leg
    "2 anchors 60-80c + 90c+": [((0.60, 0.80), -2), ((0.90, 0.995), 1)],  # negative = exactly that many
}


def leg_efficiency(legs: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for name, (lo, hi) in BANDS.items():
        g = legs[(legs.cost >= lo) & (legs.cost < hi)]
        if len(g) < 100:
            continue
        price, win = g.cost.mean(), g.win.mean()
        e = (g.win - g.cost)
        se = np.sqrt((e.groupby(g.game).sum() ** 2).sum()) / len(g)
        payout_bits = -np.log(price)                      # how much one leg multiplies the payout (log)
        rows.append(dict(band=name, legs=len(g), price=price, hit_rate=win, gap=win - price, gap_se=se,
                         legs_for_10x=np.log(10) / payout_bits,
                         value_kept_at_10x=(win / price) ** (np.log(10) / payout_bits)))
    return pd.DataFrame(rows)


def _deal(day_bands: list, plan: list, target: float, per_game: int, rng, max_legs: int = 80):
    """Add legs round-robin by recipe shares until product of costs <= 1/target."""
    ev, gm, cost, wins, n = set(), {}, 1.0, [], 0
    order = [rng.permutation(len(b)) for b in day_bands]
    pos = [0] * len(day_bands)

    def take(bi):
        nonlocal cost, n
        b = day_bands[bi]
        while pos[bi] < len(order[bi]):
            i = order[bi][pos[bi]]
            pos[bi] += 1
            e, g = b.event_ticker.values[i], b.game.values[i]
            if e in ev or gm.get(g, 0) >= per_game:
                continue
            ev.add(e)
            gm[g] = gm.get(g, 0) + 1
            cost *= b.cost.values[i]
            wins.append(b.win.values[i])
            n += 1
            return True
        return False

    for bi, (_, share) in enumerate(plan):               # fixed-count anchors first
        for _ in range(1 if share == 0 else (-share if share < 0 else 0)):
            if not take(bi):
                return None
    fill = [bi for bi, (_, share) in enumerate(plan) if share > 0]
    while cost > 1 / target and n < max_legs:
        progressed = False
        for bi in fill:
            if cost <= 1 / target:
                break
            progressed |= take(bi)
        if not progressed:
            return None
    if cost > 1 / target * 1.15:
        return None
    return cost, float(all(w > 0 for w in wins)), n


def compare(legs: pd.DataFrame, targets=(5, 10, 25, 50, 100), samples: int = 300, per_game: int = 6,
            seed: int = 3) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    out = []
    by_date = {d: g for d, g in legs.groupby("date")}
    for name, plan in RECIPES.items():
        for T in targets:
            nightly = []
            for d, day in by_date.items():
                bands = [day[(day.cost >= lo) & (day.cost < hi)] for (lo, hi), _ in plan]
                if any(len(b) == 0 for b in bands):
                    continue
                res = [r for r in (_deal(bands, plan, T, per_game, rng) for _ in range(samples)) if r]
                if len(res) < samples * 0.5:
                    continue
                c = np.array([r[0] for r in res]); h = np.array([r[1] for r in res])
                nightly.append((h.mean(), c.mean(), (h / c).mean(), np.mean([r[2] for r in res])))
            if len(nightly) < 8:
                continue
            a = np.array(nightly)
            keep = a[:, 2]
            out.append(dict(recipe=name, target=f"{T}x", nights=len(a), legs=a[:, 3].mean(), payout=1 / a[:, 1].mean(),
                            hit_rate=a[:, 0].mean(), fair_rate=a[:, 1].mean(), keep=keep.mean(),
                            keep_se=keep.std() / np.sqrt(len(a))))
    return pd.DataFrame(out)


def run(hist: pd.DataFrame, basis: str = "taker", samples: int = 300):
    legs = legs_table(hist, "pre", max_spread=0.15 if basis == "taker" else 0.06, basis=basis)
    pd.set_option("display.width", 220)
    print(f"\n{len(legs):,} legs | {legs.series.nunique()} market types | {legs.date.nunique()} nights | prices = {basis}")
    le = leg_efficiency(legs)
    print("\nPER LEG, by price band (value_kept_at_10x = share of fair value left after stacking legs to ~10x):")
    print(le.round(3).to_string(index=False))
    cmp_ = compare(legs, samples=samples)
    for T, g in cmp_.groupby("target", sort=False):
        print(f"\nTARGET ~{T}: recipes ranked by share of fair value kept")
        print(g.sort_values("keep", ascending=False).drop(columns="target").round(3).to_string(index=False))
    return le, cmp_
