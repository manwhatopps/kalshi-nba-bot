"""Historical testing. Everything here uses only what was known before each game, and real Kalshi prices.

  1. the rating model against Kalshi's pre-game price, walk-forward (models/stack.py)
  2. a negative control: what would have happened had the fund trusted the model and bet its disagreements
  3. the fund's actual rule on singles (price unless a model has earned its place): how many bets it would have made
  4. pre-game prop rules and leg categories through the 3-standard-error walk-forward test (bot/walkforward.py)
  5. Monte Carlo of the card programme at different quote qualities: what the tier budgets do to the bankroll

Limits, stated: historical COMBO quotes do not exist in any public data, so card results are simulated from leg prices
and a range of assumed quote qualities, never from invented execution prices. Game-market prices are the last trade in
the three hours before the start, which a taker could not always have had in size.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config
from . import mathx as M
from .models import stack


def trust_the_model(d: pd.DataFrame, threshold: float = 0.05) -> pd.DataFrame:
    """Bet every side where the rating model's chance beats the price plus fee by `threshold`. Uses the YES price for
    YES bets and 1 - price for NO bets (no spread added, so this flatters the strategy)."""
    rows = []
    for (sport, kind), g in d.groupby(["sport", "kind"]):
        fee_y, fee_n = M.TAKER * g.p_mkt * (1 - g.p_mkt), M.TAKER * g.p_mkt * (1 - g.p_mkt)
        yes = g[(g.p_model - g.p_mkt - fee_y) > threshold]
        no = g[((1 - g.p_model) - (1 - g.p_mkt) - fee_n) > threshold]
        cost = np.concatenate([(yes.p_mkt + M.TAKER * yes.p_mkt * (1 - yes.p_mkt)).values, ((1 - no.p_mkt) + M.TAKER * no.p_mkt * (1 - no.p_mkt)).values])
        win = np.concatenate([yes.y.values, 1 - no.y.values])
        game = np.concatenate([yes.game.values, no.game.values])
        if len(cost) < 30:
            continue
        net = pd.Series(win - cost).groupby(game).sum()
        G = len(net)
        se = float(np.sqrt(((net - net.mean()) ** 2).sum() * G / max(G - 1, 1)) / cost.sum())
        rows.append(dict(sport=sport, market=kind, bets=len(cost), won=float(win.mean()), needed=float(cost.mean()), ret=float((win - cost).sum() / cost.sum()),
                         z=float((win - cost).sum() / cost.sum() / se) if se > 0 else float("nan")))
    return pd.DataFrame(rows)


def card_programme(cfg: dict, weeks: int = 26, paths: int = 4000, seed: int = 9) -> pd.DataFrame:
    """The card tiers for `weeks`, each spending its weekly allowance out of its own pot (losses shrink the pot, wins refill
    it, an empty pot stops the tier), for several quote qualities (keep = payout offered / fair payout)."""
    rng = np.random.default_rng(seed)
    B0 = float(cfg["bankroll"])
    lo, hi = cfg["small_range"]
    plan = {}
    for tier, mults in (("small", list(np.linspace(lo + 0.2, hi - 0.2, 5))), ("diversified", list(cfg["diversified_targets"])), ("degen", list(cfg["degen_targets"]))):
        n = int(cfg["weekly_spend"][tier] * B0 // cfg["card_stake"][tier])
        plan[tier] = (float(cfg["card_stake"][tier]), [float(mults[i % len(mults)]) for i in range(n)], cfg["budget"][tier] * B0)
    rows = []
    for keep in (0.72, 0.86, 0.90, 1.00):
        pots = {t: np.full(paths, v[2]) for t, v in plan.items()}
        spent = np.zeros(paths)
        for w in range(weeks):
            for t, (stake, mults, _) in plan.items():
                for j, mult in enumerate(mults):
                    mult = mults[(j + w) % len(mults)] if t != "small" else mult
                    can = pots[t] >= stake
                    hit = rng.random(paths) < 1 / mult
                    pots[t] = np.where(can, pots[t] + np.where(hit, stake * mult * keep - stake, -stake), pots[t])
                    spent += np.where(can, stake, 0.0)
        total = sum(pots.values())
        start = sum(v[2] for v in plan.values())
        bal = B0 - start + total
        rows.append(dict(quote_keeps=f"{keep:.0%} of fair", pots=start, median_end=float(np.median(bal)), bad_1in10=float(np.quantile(bal, 0.1)),
                         good_1in10=float(np.quantile(bal, 0.9)), worst_possible=B0 - start, ahead=float((bal > B0).mean()),
                         degen_pot_empty=float((pots["degen"] < plan["degen"][0]).mean()), small_pot_empty=float((pots["small"] < plan["small"][0]).mean()),
                         avg_spent=float(spent.mean()), expected_cost=float(spent.mean()) * (1 - keep)))
    return pd.DataFrame(rows)


def run(verbose: bool = True) -> dict:
    pd.set_option("display.width", 230)
    cfg = config.load()
    d = stack.market_table()
    d = d[(d.p_mkt > 0.05) & (d.p_mkt < 0.95) & (d.n_games >= 5)]
    ev = stack.evaluate(d, save=True, verbose=verbose)
    ctl = trust_the_model(d)
    helps = [k for k, v in ev["stack"].items() if v["helps"]]
    if verbose:
        print("\nNEGATIVE CONTROL: had the fund trusted the rating model and bet every disagreement of 5+ points (after the fee, no spread charged):")
        print(ctl.round(3).to_string(index=False))
        tot = ctl.assign(c=lambda x: x.bets * x.needed)
        print(f"  all together: {int(ctl.bets.sum()):,} bets, return {float((ctl.ret * tot.c).sum() / tot.c.sum()):+.1%} per $1.")
        print(f"\nTHE FUND'S RULE ON SINGLES: our probability is the price unless a blend beat the price on later data with a model weight over 2 standard errors.")
        print(f"  Market types where that happened: {', '.join(helps) if helps else 'none'}.  Singles the chain would have accepted in the whole history from game models: "
              f"{'see blends above' if helps else '0'}.")
    walk = None
    try:
        from bot import walkforward as W
        b = W.load_pre_bets()
        rows = []
        for z in (2.0, 3.0):
            x = W.run(z=z, bets=b, max_stake=5.0)
            rows.append(dict(bar=f"{z:g} standard errors", rules_used=len(x["first_active"]), bets=x["bets"], staked=float(x["staked"]), profit=float(x["pnl"]),
                             end=float(x["final"]), max_drawdown=float(x["max_drawdown"])))
        walk = pd.DataFrame(rows)
        if verbose:
            print(f"\nPRE-GAME PROP RULES AND LEG CATEGORIES, walk-forward on {len(b):,} candidate bets ({b.rule.nunique()} rules): each night sees only earlier nights.")
            print(walk.round(2).to_string(index=False))
    except Exception as e:
        if verbose:
            print(f"\n(prop walk-forward unavailable: {e})")
    mc = card_programme(cfg)
    if verbose:
        print(f"\nMONTE CARLO: the card tiers for 26 weeks from ${cfg['bankroll']:.0f} (4,000 paths). Each tier spends its weekly allowance out of its own pot;")
        print("an empty pot stops that tier, so the cards can never lose more than the pots. Quote quality is the unknown.")
        print(mc.round(2).to_string(index=False))
        print("  Reading: the floor is the bankroll minus the pots. At the 72-86% of fair real buyers have been getting, the degen pot is usually gone within half a year.")
    return dict(evaluate=ev["table"], control=ctl, walk=walk, cards=mc)
