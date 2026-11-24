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
               stop_loss_pct=0.30, z=1.64, max_stake_per_bet=5.0,
               pre_game_only=True)   # owner's decision 2026-11-23: every bet is placed before the game; no live betting   # $5: about what these thin markets fill
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


# The walk-forward logic (bot/walkforward.py), seeing only earlier nights, first activated the
# double-double under on this date. Bets after it are out-of-sample for that mechanical selection.
DD_ACTIVATED = "2026-04-01"
DD_BANDS = ["0.35-0.50", "0.50-0.65", "0.65-0.80"]


def _stats(g: pd.DataFrame, z: float) -> dict:
    """Win rate, break-even and return for a set of bets; errors clustered by game."""
    n = len(g)
    if n == 0:
        return dict(n=0, won=float("nan"), price=float("nan"), cost=float("nan"), edge=float("nan"),
                    se=float("nan"), lb=float("nan"), lost=0, games=0)
    c = g.cost + TAKER_FEE * g.cost * (1 - g.cost)
    net = g.win - c
    cl = net.groupby(g.game).sum() if "game" in g.columns else net
    G = len(cl)
    se_roi = math.sqrt(max((cl ** 2).sum() - cl.sum() ** 2 / G, 0) * G / max(G - 1, 1)) / c.sum() if G > 1 else 1.0
    p, price, cost = g.win.mean(), g.cost.mean(), c.mean()
    se = se_roi * cost                                   # the same uncertainty expressed as a win rate
    return dict(n=n, won=p, price=price, cost=cost, edge=net.sum() / c.sum(), se=se, lb=p - z * se,
                lost=int((1 - g.win).sum()), games=G)


def dd_under_bets() -> pd.DataFrame:
    """NBA double-double UNDER: one bet per market, the first in-game fill where the under (NO) could be
    bought at 35-80c (the bands the walk-forward logic has active). Both seasons."""
    from .favorites import slate_date
    parts = []
    for season, f in (("last season", "prop_touch_prior.parquet"), ("this season", "prop_touch.parquet")):
        fp = config.DATA_DIR / f
        if not fp.exists():
            continue
        t = pd.read_parquet(fp)
        t = t[(t.series == "KXNBA2D") & (t.taker_side == "no") & t.band.isin(DD_BANDS)]
        t = t.loc[t.groupby("ticker").created_time.idxmin()].assign(season=season)
        parts.append(t)
    if not parts:
        return pd.DataFrame()
    d = pd.concat(parts, ignore_index=True)
    d["win"] = (d.result == "no").astype(float)
    d["date"] = d.game.map(slate_date)
    d["resolve"] = d.created_time + pd.to_timedelta((150 - d.mins).clip(lower=5), unit="m")
    return d


def edge_registry(z: float = 1.64) -> pd.DataFrame:
    """Every candidate rule, measured on separate sets of games:
         found on     the games where the rule was noticed (it will always look good there)
         independent  games it was NOT found on: a prior season it was never fitted to and/or every
                      game played since discovery
       Grades: A = holds on 100+ independent bets; B = positive at the low end of its error range, too
       few independent bets to judge; C = fails or is indistinguishable from zero."""
    from .favorites import slate_date
    from .trades import PRIOR_TOUCH
    fp = config.DATA_DIR / "first_touch.parquet"
    cands = []
    if fp.exists():
        ft = pd.read_parquet(fp)
        prior = pd.read_parquet(PRIOR_TOUCH) if PRIOR_TOUCH.exists() else ft.iloc[0:0]
        for league, series, th, label in (("NBA", "KXNBAGAME", 0.95, "NBA in-game moneyline, first touch of 95c"),
                                          ("NBA", "KXNBAGAME", 0.97, "NBA in-game moneyline, first touch of 97c"),
                                          ("NBA", "KXNBAGAME", 0.90, "NBA in-game moneyline, first touch of 90c"),
                                          ("NFL", "KXNFLGAME", 0.95, "NFL in-game moneyline, first touch of 95c")):
            g = _per_game(ft, league, series, th).reset_index()
            if len(g) < 20:
                continue
            g["date"] = g.game.map(slate_date)
            arch = _per_game(prior, league, series, th).reset_index()
            cut = pd.Timestamp(DISCOVERED_THROUGH)
            cands.append(dict(id=f"ml{int(th * 100)}_{league.lower()}", label=label, league=league, threshold=th,
                              found=g[g.date <= cut], ind=pd.concat([arch, g[g.date > cut]]), arch=len(arch),
                              fwd=g[g.date > cut], note="found on this season only"))
    dd = dd_under_bets()
    if len(dd):
        cut = pd.Timestamp(DD_ACTIVATED)
        cands.append(dict(id="dd_under", label="NBA in-game double-double UNDER, first fill at 35-80c", league="NBA",
                          threshold=0.0, found=dd[dd.date < cut], ind=dd[dd.date >= cut], arch=0, fwd=dd[dd.date >= cut],
                          note=f"the walk-forward logic activated it on {DD_ACTIVATED} from earlier nights only"))
    rows = []
    for c in cands:
        d, i = _stats(c["found"], z), _stats(c["ind"], z)
        alls = _stats(pd.concat([c["found"], c["ind"]]), z)
        if i["n"] >= 100:
            grade = "A confirmed" if i["lb"] > i["cost"] else "C failed on independent games"
        else:
            grade = "B candidate" if d["lb"] > d["cost"] else "C unproven"
        use = i if grade[0] == "A" else d      # once confirmed, size from the independent games only
        nights = c["found"].date.nunique() if "date" in c["found"] else 1
        seas = ""
        if "season" in c["found"].columns:
            seas = "; ".join(f"{s}: {len(x)} bets, won {x.win.mean():.1%}, {_stats(x, z)['edge']:+.1%}"
                             for s, x in c["found"].groupby("season"))
        rows.append(dict(id=c["id"], edge=c["label"], league=c["league"], threshold=c["threshold"], note=c["note"],
                         bets=d["n"], games=d["games"], losses=d["lost"], won=d["won"], price=d["price"], cost=d["cost"],
                         edge_pct=d["edge"], se=d["se"], lower_bound=d["lb"], by_season=seas,
                         ind_bets=i["n"], ind_losses=i["lost"], ind_won=i["won"], ind_cost=i["cost"], ind_edge=i["edge"],
                         arch_bets=c["arch"], new_bets=len(c["fwd"]),
                         new_won=c["fwd"].win.mean() if len(c["fwd"]) else float("nan"),
                         kelly_full=max(0.0, (use["won"] - use["cost"]) / (1 - use["cost"])),
                         kelly_safe=max(0.0, (use["lb"] - use["cost"]) / (1 - use["cost"])),
                         grade=grade, per_night=d["n"] / max(nights, 1),
                         game_share=d["n"] / max(d["games"], 1)))
    return pd.DataFrame(rows)


def stake_fraction(e: pd.Series) -> float:
    """Share of bankroll per bet: half Kelly once confirmed; a quarter of the lower-bound Kelly while a
    candidate (bets on the same night are not independent, and the edge is not yet confirmed); 0 if unproven."""
    return {"A": 0.5 * e.kelly_full, "B": 0.25 * e.kelly_safe, "C": 0.0}[e.grade[0]]


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
