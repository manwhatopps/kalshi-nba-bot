"""The fund: one bankroll, run like a small hedge fund.

  CORE sleeve    single bets where the data shows a measured edge, sized by Kelly. The stake is set
                 from the edge's LOWER confidence bound while an edge is unconfirmed, and spread
                 across the night's independent triggers instead of riding on one game.
  DEGEN sleeve   a small, capped budget for lottery combos ($5-10 aiming at $1,000-$10,000). These
                 are negative-expectation by every measurement here, so the sleeve is a fixed cost
                 you choose, never sized up, and built to lose as little as possible per ticket
                 (fewest legs, fairly priced legs, a walk-away payout).
  RESERVE        everything else. Never deployed. It is what keeps a bad night from ending the fund.

Edges come from data only (data/first_touch.parquet), are graded, and are re-graded every day as new
games arrive. Nothing here places orders; `watch` and `fund` log paper positions with dollar stakes.
"""
from __future__ import annotations

import json
import math

import numpy as np
import pandas as pd

from . import config
from .trades import TAKER_FEE

CFG = config.DATA_DIR / "fund.json"
DEFAULT = dict(bankroll=100.0, core_pct=0.50, degen_tickets=[[5, 1000], [5, 10000]], max_concurrent_pct=0.50,
               stop_loss_pct=0.30, z=1.64)
DISCOVERED_THROUGH = "2026-11-21"


def load_cfg() -> dict:
    c = dict(DEFAULT)
    if CFG.exists():
        c.update(json.loads(CFG.read_text()))
    return c


def save_cfg(c: dict) -> None:
    CFG.write_text(json.dumps(c, indent=1))


def cost_with_fee(price: float) -> float:
    return price + TAKER_FEE * price * (1 - price)


def kelly(p: float, price: float) -> float:
    c = cost_with_fee(price)
    return max(0.0, (p - c) / (1 - c))


# ---------------------------------------------------------------- edge registry
def _per_game(ft: pd.DataFrame, league: str, series: str, th: float) -> pd.DataFrame:
    d = ft[(ft.league == league) & (ft.series == series) & (ft.threshold == th)]
    return d.sort_values("created_time").groupby("game").first()   # the two tickers of a game are one bet


def edge_registry(z: float = 1.64) -> pd.DataFrame:
    from .favorites import slate_date
    fp = config.DATA_DIR / "first_touch.parquet"
    if not fp.exists():
        return pd.DataFrame()
    ft = pd.read_parquet(fp)
    games_all = ft[ft.league == "NBA"].game.nunique()
    rows = []
    for league, series, th, label in (("NBA", "KXNBAGAME", 0.95, "NBA in-game moneyline, first touch of 95c"),
                                      ("NBA", "KXNBAGAME", 0.97, "NBA in-game moneyline, first touch of 97c"),
                                      ("NBA", "KXNBAGAME", 0.90, "NBA in-game moneyline, first touch of 90c"),
                                      ("NFL", "KXNFLGAME", 0.95, "NFL in-game moneyline, first touch of 95c")):
        g = _per_game(ft, league, series, th)
        if len(g) < 20:
            continue
        g = g.assign(date=[slate_date(x) for x in g.index])
        ins, oos = g[g.date <= pd.Timestamp(DISCOVERED_THROUGH)], g[g.date > pd.Timestamp(DISCOVERED_THROUGH)]
        n, p, price = len(ins), ins.win.mean(), ins.cost.mean()
        se = math.sqrt(max(p * (1 - p), 1e-9) / n)
        c = cost_with_fee(price)
        lb = p - z * se
        on, op = len(oos), (oos.win.mean() if len(oos) else float("nan"))
        if on >= 100 and op - z * math.sqrt(max(op * (1 - op), 1e-9) / on) > cost_with_fee(oos.cost.mean()):
            grade = "A confirmed"
        elif lb > c:
            grade = "B candidate"
        else:
            grade = "C unproven"
        rows.append(dict(edge=label, league=league, threshold=th, bets=n, losses=int((1 - ins.win).sum()), won=p,
                         price=price, cost=c, edge_pct=p / c - 1, se=se, lower_bound=lb, kelly_full=kelly(p, price),
                         kelly_safe=kelly(lb, price), grade=grade, new_bets=on, new_won=op,
                         per_night=n / max(ins.date.nunique(), 1),
                         game_share=n / games_all if league == "NBA" else float("nan")))
    return pd.DataFrame(rows)


def stake_fraction(e: pd.Series) -> float:
    """Share of bankroll per bet: half Kelly once confirmed; the lower-bound Kelly while a candidate; 0 if unproven."""
    return {"A": 0.5 * e.kelly_full, "B": e.kelly_safe, "C": 0.0}[e.grade[0]]


# ---------------------------------------------------------------- simulation
def simulate(cfg: dict, p_win: float, price: float, per_night: float, style: str, nights: int = 60,
             degen: str = "nightly", degen_keep: float = 0.5, paths: int = 4000, seed: int = 2,
             stake_frac: float | None = None) -> dict:
    """style 'one'    = the whole core sleeve on the first trigger of the night;
       style 'spread' = stake_frac of bankroll on every trigger, total at risk capped at the sleeve.
       degen 'nightly' | 'weekly' | 'profits' (only out of money won above the starting bankroll) | 'none'.
       degen_keep = share of fair value a lottery ticket really returns (1.0 fair; history says far less)."""
    rng = np.random.default_rng(seed)
    c = cost_with_fee(price)
    win_ret = 1 / c - 1
    start = cfg["bankroll"]
    bal = np.full(paths, start)
    core_pnl, degen_pnl = np.zeros(paths), np.zeros(paths)
    peak, maxdd = bal.copy(), np.zeros(paths)
    for night in range(nights):
        alive = bal > 1.0
        k = rng.poisson(per_night, paths)
        sleeve = cfg["core_pct"] * bal
        if style == "one":
            n_bets = np.minimum(k, 1)
            stake = sleeve
        else:
            n_bets = k
            stake = np.minimum((stake_frac or 0) * bal, sleeve / np.maximum(k, 1))
        wins = rng.binomial(n_bets, p_win)
        pnl = np.where(alive, stake * (wins * win_ret - (n_bets - wins)), 0.0)
        core_pnl += pnl
        bal = bal + pnl
        buy = degen == "nightly" or (degen == "weekly" and night % 7 == 0)
        for st, goal in cfg["degen_tickets"]:
            if degen == "none":
                break
            ok = (bal > start + st) if degen == "profits" else np.full(paths, buy)
            ok &= bal > st
            mult = goal / st
            hit = rng.random(paths) < degen_keep / mult
            d = np.where(ok, np.where(hit, goal - st, -st), 0.0)
            degen_pnl += d
            bal = bal + d
        peak = np.maximum(peak, bal)
        maxdd = np.maximum(maxdd, peak - bal)
    return dict(median=float(np.median(bal)), p10=float(np.percentile(bal, 10)), p90=float(np.percentile(bal, 90)),
                ahead=float((bal > start).mean()), halved=float((bal < 0.5 * start).mean()),
                core=float(core_pnl.mean()), degen=float(degen_pnl.mean()), big_hit=float((degen_pnl > 500).mean()),
                drawdown=float(np.median(maxdd)))
