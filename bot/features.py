"""Leak-free features + three independent probability models for "player stat > line".

  p_nb   negative-binomial stats model (per-minute rates x projected minutes x opponent factor)
  p_hit  recency-weighted hit rate over the player's last 20 games, shrunk toward p_nb
  p_gbm  gradient-boosted classifier (sklearn HistGradientBoosting) trained on prior seasons
Everything is computed "as of" the morning of the game: only earlier games are visible.
"""
from __future__ import annotations

import pickle

import numpy as np
import pandas as pd
from scipy import stats as st

from . import config
from .data import STATS, KALSHI_TO_LOG_TEAM, LOG_TO_KALSHI_TEAM
from .model import Model, load_priors

STAT_SETS = {"pts": ("pts",), "reb": ("reb",), "ast": ("ast",), "fg3m": ("fg3m",), "stl": ("stl",),
             "blk": ("blk",), "pra": ("pts", "reb", "ast"), "stocks": ("stl", "blk")}
STAT_ID = {v: i for i, v in enumerate(STAT_SETS.values())}
LADDERS = {("pts",): range(5, 46, 5), ("reb",): range(2, 17, 2), ("ast",): range(2, 15, 2),
           ("fg3m",): range(1, 8), ("stl",): range(1, 5), ("blk",): range(1, 5),
           ("pts", "reb", "ast"): range(10, 61, 5), ("stl", "blk"): range(1, 6)}
FEATS = ["p_nb", "hr5", "hr10", "hr20", "mu", "sd", "line", "z", "minutes", "min_sd", "days_since",
         "games", "home", "opp_f", "stat_id"]
GBM_PATH = config.DATA_DIR / "gbm_props.pkl"


def moment(proj, stats: tuple, self_corr) -> tuple[float, float]:
    mu = sum(proj.mean[s] for s in stats)
    var = sum(proj.var[s] for s in stats)
    for i, a in enumerate(stats):
        for b in stats[i + 1:]:
            var += 2 * self_corr[STATS.index(a)][STATS.index(b)] * np.sqrt(proj.var[a] * proj.var[b])
    return mu, max(var, mu * 1.02)


def p_over(mu: float, var: float, line: float) -> float:
    k = np.floor(line)
    if var <= mu * 1.03:
        return float(1 - st.poisson.cdf(k, mu))
    r = mu * mu / (var - mu)
    return float(1 - st.nbinom.cdf(k, r, r / (r + mu)))


def leg_features(model: Model, name: str, team: str, opp: str, home: bool, stats: tuple, line: float):
    """team/opp are Kalshi codes. Returns dict of features or None if the player can't be projected."""
    key = (name, team, opp)
    cache = model.__dict__.setdefault("_fcache", {})
    if key not in cache:
        proj = model.project(name, team, opp)
        g = model.find_player(name, team) if proj else None
        cache[key] = (proj, g.tail(20) if g is not None else None)
    proj, g = cache[key]
    if proj is None:
        return None
    mu, var = moment(proj, stats, model.pri["self"])
    sd = float(np.sqrt(var))
    x = g[list(stats)].sum(axis=1).values.astype(float)
    over = (x > line).astype(float)
    w = 0.5 ** (np.arange(len(x))[::-1] / 8.0)
    pnb = p_over(mu, var, line)
    n_eff = w.sum() ** 2 / (w ** 2).sum()
    return dict(p_nb=pnb, hr5=over[-5:].mean(), hr10=over[-10:].mean(), hr20=over.mean(),
                p_hit=float(((over * w).sum() / w.sum() * n_eff + 6 * pnb) / (n_eff + 6)),
                mu=mu, sd=sd, line=line, z=(line + 0.5 - mu) / sd, minutes=proj.minutes,
                min_sd=float(g.minutes.tail(10).std() if len(g) > 1 else 5.0), days_since=proj.days_since,
                games=proj.games, home=float(home), opp_f=model.opp_factor(KALSHI_TO_LOG_TEAM.get(opp, opp), stats[0]),
                stat_id=STAT_ID.get(stats, -1))


def training_frame(logs: pd.DataFrame, start, end, min_minutes: float = 10) -> pd.DataFrame:
    """Synthetic (player, game, line) rows on Kalshi-style ladders with the real outcome."""
    pri = load_priors(logs)
    dates = sorted(d for d in logs.game_date.unique() if pd.Timestamp(start) <= d <= pd.Timestamp(end))
    rows = []
    for d in dates:
        m = Model(logs, as_of=d, priors=pri)
        for r in logs[logs.game_date == d].itertuples():
            team, opp = (LOG_TO_KALSHI_TEAM.get(t, t) for t in (r.team, r.opponent))
            for stats, ladder in LADDERS.items():
                actual = sum(getattr(r, s) for s in stats)
                for k in ladder:
                    f = leg_features(m, r.player_name, team, opp, bool(r.home), stats, k - 0.5)
                    if f is None or f["minutes"] < min_minutes:
                        break
                    if 0.02 < f["p_nb"] < 0.98:
                        f["y"] = float(actual > k - 0.5)
                        f["date"] = d
                        rows.append(f)
    return pd.DataFrame(rows)


def train_gbm(logs: pd.DataFrame, start="2025-11-10", end="2026-04-12"):
    from sklearn.ensemble import HistGradientBoostingClassifier
    df = training_frame(logs, start, end)
    clf = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, max_leaf_nodes=31,
                                         l2_regularization=1.0, categorical_features=[FEATS.index("stat_id")])
    clf.fit(df[FEATS].values, df.y.values)
    GBM_PATH.write_bytes(pickle.dumps({"clf": clf, "trained_through": str(end), "rows": len(df)}))
    return clf, df


def load_gbm():
    return pickle.loads(GBM_PATH.read_bytes())["clf"] if GBM_PATH.exists() else None
