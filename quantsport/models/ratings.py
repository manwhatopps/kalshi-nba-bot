"""Team ratings in points, updated game by game, for NBA, NFL and college football.

A team's rating is how many points better than average it is. Before each game the model predicts
    margin = rating(home) - rating(away) + home advantage          total = league base + pace(home) + pace(away)
and afterwards moves both ratings a fraction K of the miss (an Elo update in points; blowouts are capped).
Ratings carry over between seasons shrunk toward zero. College teams outside the top division start well below zero.

Every prediction is made before the game is seen, so the whole history is a walk-forward test by construction.
The spread of the misses gives sigma, and with a normal curve that turns a margin or total into the chance of
any Kalshi game market: winner, "wins by over X", "over X points", team totals.

This is deliberately the simplest respectable model: the question the fund asks is not "is it clever" but
"does it add anything to Kalshi's price", and `quantsport/models/stack.py` measures exactly that.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.stats import norm

PARAMS = {  # k: share of the miss moved into the ratings; hfa: home points; carry: share kept between seasons; cap: blowout cap
    "NBA": dict(k=0.045, hfa=2.4, carry=0.70, cap=25, kt=0.05, start=0.0),
    "NFL": dict(k=0.060, hfa=1.6, carry=0.65, cap=24, kt=0.05, start=0.0),
    "NCAAF": dict(k=0.110, hfa=2.6, carry=0.60, cap=35, kt=0.08, start=-14.0),
}
SIGMA0 = {"NBA": (12.5, 18.0), "NFL": (13.0, 13.5), "NCAAF": (16.0, 17.0)}     # starting margin / total spread, replaced by measured misses


@dataclass
class Pred:
    mu: float
    sigma: float
    total: float
    sigma_t: float
    games_home: int
    games_away: int

    def p_win(self, home: bool = True) -> float:
        p = 1 - norm.cdf(0, self.mu, self.sigma)
        return float(p if home else 1 - p)

    def p_margin_over(self, line: float, home: bool = True) -> float:
        """Chance this team wins by MORE than `line` points."""
        return float(1 - norm.cdf(line, self.mu if home else -self.mu, self.sigma))

    def p_total_over(self, line: float) -> float:
        return float(1 - norm.cdf(line, self.total, self.sigma_t))

    def p_team_over(self, line: float, home: bool = True) -> float:
        mean = (self.total + (self.mu if home else -self.mu)) / 2
        return float(1 - norm.cdf(line, mean, np.sqrt(self.sigma ** 2 + self.sigma_t ** 2) / 2))


class Ratings:
    def __init__(self, sport: str, **override):
        self.sport, self.p = sport, {**PARAMS[sport], **override}
        self.r, self.t, self.n = {}, {}, {}
        self.base, self.season = None, None
        self.sig, self.sig_t = SIGMA0[sport]
        self._e2, self._t2, self._cnt = 0.0, 0.0, 0

    def _get(self, team: str, top: bool = True) -> float:
        if team not in self.r:
            self.r[team], self.t[team], self.n[team] = (self.p["start"] if not top else 0.0), 0.0, 0
        return self.r[team]

    def predict(self, home: str, away: str, neutral: bool = False, home_top: bool = True, away_top: bool = True) -> Pred:
        rh, ra = self._get(home, home_top), self._get(away, away_top)
        base = self.base if self.base is not None else {"NBA": 228.0, "NFL": 45.0, "NCAAF": 53.0}[self.sport]
        return Pred(mu=rh - ra + (0.0 if neutral else self.p["hfa"]), sigma=self.sig, total=base + self.t[home] + self.t[away],
                    sigma_t=self.sig_t, games_home=self.n[home], games_away=self.n[away])

    def update(self, home: str, away: str, home_pts: float, away_pts: float, neutral: bool, season: str, pred: Pred) -> None:
        if self.season is not None and season != self.season:              # new season: shrink everything toward average
            for d in (self.r, self.t):
                for k_ in d:
                    d[k_] *= self.p["carry"]
        self.season = season
        margin, total = home_pts - away_pts, home_pts + away_pts
        err = float(np.clip(margin - pred.mu, -self.p["cap"], self.p["cap"]))
        self.r[home] += self.p["k"] * err
        self.r[away] -= self.p["k"] * err
        et = total - pred.total
        self.t[home] += self.p["kt"] * et / 2
        self.t[away] += self.p["kt"] * et / 2
        self.base = total if self.base is None else self.base + 0.01 * (total - self.base)
        self.n[home] += 1
        self.n[away] += 1
        self._e2 += (margin - pred.mu) ** 2
        self._t2 += et ** 2
        self._cnt += 1
        if self._cnt >= 200:                                                # sigma from the model's own past misses
            self.sig, self.sig_t = float(np.sqrt(self._e2 / self._cnt)), float(np.sqrt(self._t2 / self._cnt))


def walk(sport: str, games: pd.DataFrame, **override) -> tuple[pd.DataFrame, Ratings]:
    """Run through finished games in date order. Returns one prediction row per game (made before it) and the final ratings."""
    R = Ratings(sport, **override)
    rows = []
    top_h = games.home_fbs.values if "home_fbs" in games else np.ones(len(games), bool)
    top_a = games.away_fbs.values if "away_fbs" in games else np.ones(len(games), bool)
    for i, g in enumerate(games.itertuples()):
        p = R.predict(g.home, g.away, bool(g.neutral), bool(top_h[i]), bool(top_a[i]))
        rows.append((g.date, g.season, g.home, g.away, g.home_pts, g.away_pts, p.mu, p.sigma, p.total, p.sigma_t, p.games_home, p.games_away))
        R.update(g.home, g.away, g.home_pts, g.away_pts, bool(g.neutral), g.season, p)
    d = pd.DataFrame(rows, columns=["date", "season", "home", "away", "home_pts", "away_pts", "mu", "sigma", "total", "sigma_t", "n_home", "n_away"])
    d["p_home"] = 1 - norm.cdf(0, d.mu, d.sigma)
    return d, R


def tune(sport: str, games: pd.DataFrame, train_until: str) -> dict:
    """Pick K, home advantage and carry-over on the TRAINING seasons only (lowest log-loss on who won)."""
    best, best_ll = {}, 9e9
    tr = games[games.date < pd.Timestamp(train_until)]
    for k in (0.03, 0.045, 0.06, 0.08, 0.11, 0.14):
        for hfa in (1.5, 2.5, 3.5):
            for carry in (0.5, 0.7, 0.85):
                d, _ = walk(sport, tr, k=k, hfa=hfa, carry=carry)
                d = d.iloc[len(d) // 3:]                              # skip the warm-up third
                y = (d.home_pts > d.away_pts).astype(float)
                p = d.p_home.clip(0.02, 0.98)
                ll = float(-(y * np.log(p) + (1 - y) * np.log(1 - p)).mean())
                if ll < best_ll:
                    best, best_ll = dict(k=k, hfa=hfa, carry=carry), ll
    return best | dict(train_logloss=best_ll)
