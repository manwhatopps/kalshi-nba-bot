"""Walk-forward calibration of the player-prop model on past games (no lookahead).

For each day in the window, the model only sees games before that day, projects every rotation
player, and prices "over" lines near the projection. Kalshi has no free price history, so this
measures CALIBRATION (are 30% calls right 30% of the time?), not edge vs the market — the
`report` command measures edge on live paper trades.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats as st

from .data import LOG_TO_KALSHI_TEAM
from .model import Model, load_priors


def p_over(mu, var, line):
    k = np.floor(line)
    if var <= mu * 1.02:
        return 1 - st.poisson.cdf(k, mu)
    r = mu * mu / (var - mu)
    return 1 - st.nbinom.cdf(k, r, r / (r + mu))


def run(logs: pd.DataFrame, days: int = 21, stats=("pts", "reb", "ast", "fg3m")):
    pri = load_priors(logs)
    end = logs.game_date.max()
    dates = sorted(d for d in logs.game_date.unique() if d > end - pd.Timedelta(days=days))
    rows = []
    for d in dates:
        m = Model(logs, as_of=d, priors=pri)
        today = logs[(logs.game_date == d) & (logs.minutes >= 1)]
        for r in today.itertuples():
            team = LOG_TO_KALSHI_TEAM.get(r.team, r.team)
            opp = LOG_TO_KALSHI_TEAM.get(r.opponent, r.opponent)
            p = m.project(r.player_name, team, opp)
            if p is None or p.minutes < 15:
                continue
            for s in stats:
                mu = p.mean[s]
                for off in (-0.5, 0.5) if s != "pts" else (-3.5, 0.5, 4.5):
                    line = max(0.5, np.floor(mu) + off)
                    rows.append((s, line, p_over(mu, p.var[s], line), float(getattr(r, s) > line)))
    df = pd.DataFrame(rows, columns=["stat", "line", "p", "hit"])
    print(f"\nWalk-forward over {len(dates)} game days, {len(df)} player-line predictions")
    print(f"  Brier {((df.p - df.hit) ** 2).mean():.4f}  vs coin-flip-at-base-rate "
          f"{((df.hit.mean() - df.hit) ** 2).mean():.4f}")
    df["bin"] = pd.cut(df.p, [0, .1, .2, .3, .4, .5, .6, .7, .8, .9, 1])
    cal = df.groupby("bin", observed=True).agg(n=("hit", "size"), predicted=("p", "mean"), actual=("hit", "mean"))
    print("\nCalibration (predicted vs actual hit rate):")
    print(cal.round(3).to_string())
    print("\nBy stat:")
    print(df.groupby("stat").apply(lambda g: pd.Series({
        "n": len(g), "brier": ((g.p - g.hit) ** 2).mean(), "pred": g.p.mean(), "actual": g.hit.mean()}),
        include_groups=False).round(3).to_string())
    return df
