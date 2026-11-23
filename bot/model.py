"""Stats model: projects every player's box score and both team scores, then simulates each game
jointly with a Gaussian copula so same-game legs (e.g. star 30+ pts AND game over 230) carry
realistic correlation instead of being multiplied as if independent.

Correlation priors (same-player stat pairs, teammates, opponents, player-vs-team-score) are
estimated from historical game logs once and cached in data/corr_priors.json.
"""
from __future__ import annotations

import json
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats as st

from . import config
from .data import STATS, KALSHI_TO_LOG_TEAM, norm_name

N_SIMS = 20000
HOME_ADV = 1.5
# Game-score noise calibrated to well-known NBA residuals vs. closing lines: margin SD ~12.5,
# total SD ~18  =>  per-team SD 11.0 and team-score correlation 0.35.
TEAM_SD, TEAM_RHO = 11.0, 0.35
MARGIN_SD, TOTAL_SD = 12.5, 18.0
W_MARKET_SCORE = 0.75  # how much the team-score means trust Kalshi's own spread/total ladders
PRIOR_PATH = config.DATA_DIR / "corr_priors.json"


# ---------------------------------------------------------------- correlation priors
def _zscores(logs: pd.DataFrame) -> pd.DataFrame:
    g = logs.groupby(["season", "player_id"])
    ok = (g.minutes.transform("count") >= 15) & (g.minutes.transform("mean") >= 18)
    d = logs[ok].copy()
    g = d.groupby(["season", "player_id"])
    for s in STATS:
        sd = g[s].transform("std").replace(0, np.nan)
        d["z_" + s] = (d[s] - g[s].transform("mean")) / sd
    tp = logs.groupby(["game_id", "team", "season"], as_index=False).pts.sum().rename(columns={"pts": "tpts"})
    tg = tp.groupby(["season", "team"]).tpts
    tp["z_tpts"] = (tp.tpts - tg.transform("mean")) / tg.transform("std")
    return d, tp


def build_priors(logs: pd.DataFrame) -> dict:
    hist = logs[logs.season != config.CURRENT_SEASON]
    if hist.empty:
        hist = logs
    d, tp = _zscores(hist)
    zc = ["z_" + s for s in STATS]
    d = d.dropna(subset=zc)
    self_c = d[zc].corr().values

    def cross(a, b):
        x = a[[c + "_a" for c in zc]].values
        y = b[[c + "_b" for c in zc]].values
        x = (x - x.mean(0)) / x.std(0)
        y = (y - y.mean(0)) / y.std(0)
        m = x.T @ y / len(x)
        return ((m + m.T) / 2)

    small = d[["game_id", "team", "opponent", "player_id", *zc]]
    mates = small.merge(small, on=["game_id", "team"], suffixes=("_a", "_b"))
    mates = mates[mates.player_id_a < mates.player_id_b]
    opps = small.merge(small.rename(columns={"team": "opponent", "opponent": "team"}),
                       on=["game_id", "team"], suffixes=("_a", "_b"))
    opps = opps[opps.player_id_a < opps.player_id_b]
    mate_c = cross(mates, mates)
    opp_c = cross(opps, opps)

    j = d.merge(tp[["game_id", "team", "z_tpts"]], on=["game_id", "team"])
    j = j.merge(tp[["game_id", "team", "z_tpts"]].rename(columns={"team": "opponent", "z_tpts": "z_opp"}),
                on=["game_id", "opponent"])
    own_team = [float(j[c].corr(j.z_tpts)) for c in zc]
    opp_team = [float(j[c].corr(j.z_opp)) for c in zc]
    tt = tp.merge(tp, on="game_id", suffixes=("_a", "_b"))
    tt = tt[tt.team_a < tt.team_b]
    team_team = float(tt.z_tpts_a.corr(tt.z_tpts_b))
    team_sd = float((tp.tpts - tp.groupby(["season", "team"]).tpts.transform("mean")).std())

    g = d.groupby(["season", "player_id"])
    phi = {s: float(np.nanmedian((g[s].var() / g[s].mean().clip(lower=0.3)).values)) for s in STATS}
    pri = dict(self=self_c.tolist(), mate=mate_c.tolist(), opp=opp_c.tolist(), own_team=own_team,
               opp_team=opp_team, team_team=team_team, team_sd=team_sd, phi=phi, n_games=int(d.game_id.nunique()))
    PRIOR_PATH.write_text(json.dumps(pri, indent=1))
    return pri


def load_priors(logs: pd.DataFrame, rebuild: bool = False) -> dict:
    if PRIOR_PATH.exists() and not rebuild:
        return json.loads(PRIOR_PATH.read_text())
    return build_priors(logs)


# ---------------------------------------------------------------- projections
@dataclass
class Proj:
    name: str
    team: str
    minutes: float
    mean: dict
    var: dict
    games: int
    days_since: int


def _ewm_weights(n: int, half_life: float) -> np.ndarray:
    w = 0.5 ** (np.arange(n)[::-1] / half_life)
    return w / w.sum()


class Model:
    def __init__(self, logs: pd.DataFrame, as_of: pd.Timestamp | None = None, priors: dict | None = None):
        self.as_of = pd.Timestamp(as_of) if as_of is not None else logs.game_date.max() + pd.Timedelta(days=1)
        self.logs = logs[logs.game_date < self.as_of]
        self.pri = priors or load_priors(logs)
        self._by_name = {k: g for k, g in self.logs.groupby("name_key")}
        self._team_ratings()

    # team offense/defense over last 15 games, per stat
    def _team_ratings(self):
        tg = self.logs.groupby(["game_id", "game_date", "team", "opponent"], as_index=False)[STATS].sum()
        tg = tg.sort_values("game_date")
        recent_for = tg.groupby("team").tail(15).groupby("team")[STATS].mean()
        recent_against = tg.groupby("opponent").tail(15).groupby("opponent")[STATS].mean()
        self.league = tg[tg.game_date >= self.as_of - pd.Timedelta(days=120)][STATS].mean()
        self.t_for, self.t_against = recent_for, recent_against

    def opp_factor(self, opp: str, s: str) -> float:
        if opp not in self.t_against.index or self.league[s] <= 0:
            return 1.0
        return float((self.t_against.loc[opp, s] / self.league[s]) ** 0.5)

    def team_mean(self, team: str, opp: str, home: bool) -> float:
        L = self.league["pts"]
        off = self.t_for.loc[team, "pts"] if team in self.t_for.index else L
        dfn = self.t_against.loc[opp, "pts"] if opp in self.t_against.index else L
        return float(off * dfn / L + (HOME_ADV if home else -HOME_ADV) / 2)

    def find_player(self, name: str, kalshi_team: str) -> pd.DataFrame | None:
        team = KALSHI_TO_LOG_TEAM.get(kalshi_team, kalshi_team)
        key = norm_name(name)
        g = self._by_name.get(key)
        if g is None:  # fallback: same last name on same team
            last = key.split(" ")[-1]
            cands = [k for k in self._by_name if k.split(" ")[-1] == last and k[:1] == key[:1]]
            cands = [k for k in cands if (self._by_name[k].team.iloc[-1] == team)]
            if len(cands) != 1:
                return None
            g = self._by_name[cands[0]]
        return g

    def project(self, name: str, kalshi_team: str, opp_kalshi: str) -> Proj | None:
        g = self.find_player(name, kalshi_team)
        if g is None or len(g) < 5:
            return None
        g = g.tail(30)
        days_since = int((self.as_of - g.game_date.iloc[-1]).days)
        if days_since > 21:
            return None  # long absence: minutes unknowable from logs
        opp = KALSHI_TO_LOG_TEAM.get(opp_kalshi, opp_kalshi)
        mins = g.minutes.values
        m_w = _ewm_weights(min(len(g), 10), 4.0)
        proj_min = float((mins[-len(m_w):] * m_w).sum())
        w = _ewm_weights(len(g), 10.0)
        mean, var = {}, {}
        tot_min = (mins * w).sum()
        for s in STATS:
            x = g[s].values.astype(float)
            rate = (x * w).sum() / max(tot_min, 1e-6)
            mu = max(rate * proj_min * self.opp_factor(opp, s), 0.02)
            emp_mu = max((x * w).sum(), 0.02)
            emp_var = ((x - emp_mu) ** 2 * w).sum() * len(x) / max(len(x) - 1, 1)
            v = 0.5 * emp_var * (mu / emp_mu) + 0.5 * mu * self.pri["phi"][s]
            mean[s], var[s] = mu, max(v, mu * 1.05)
        return Proj(name, kalshi_team, proj_min, mean, var, len(g), days_since)

    # ---------------------------------------------------------------- joint simulation
    def simulate_game(self, away: str, home: str, players: list[tuple[str, str]], n: int = N_SIMS,
                      seed: int = 7, market: dict | None = None, need: dict | None = None) -> dict:
        """players: [(name, kalshi_team)]. Returns {'home_pts','away_pts', (name,stat): array, 'proj': {...}}."""
        A, H = (KALSHI_TO_LOG_TEAM.get(t, t) for t in (away, home))
        projs = {}
        for name, team in players:
            if (name, team) in projs:
                continue
            p = self.project(name, team, home if team == away else away)
            if p:
                projs[(name, team)] = p
        keys = list(projs)
        k, ns = len(keys), len(STATS)
        dim = 2 + k * ns
        C = np.eye(dim)
        C[0, 1] = C[1, 0] = TEAM_RHO
        S, M, O = (np.array(self.pri[x]) for x in ("self", "mate", "opp"))
        own_t, opp_t = np.array(self.pri["own_team"]), np.array(self.pri["opp_team"])
        for i, ki in enumerate(keys):
            bi = 2 + i * ns
            C[bi:bi + ns, bi:bi + ns] = S
            is_home = ki[1] == home
            C[bi:bi + ns, 0] = C[0, bi:bi + ns] = own_t if is_home else opp_t
            C[bi:bi + ns, 1] = C[1, bi:bi + ns] = opp_t if is_home else own_t
            for j in range(i + 1, k):
                bj = 2 + j * ns
                blk = M if keys[j][1] == ki[1] else O
                C[bi:bi + ns, bj:bj + ns] = blk
                C[bj:bj + ns, bi:bi + ns] = blk.T
        np.fill_diagonal(C, 1.0)
        w, V = np.linalg.eigh(C)  # project to nearest PSD correlation matrix
        C = V @ np.diag(np.clip(w, 1e-4, None)) @ V.T
        dd = np.sqrt(np.diag(C))
        C = C / np.outer(dd, dd)
        rng = np.random.default_rng(seed)
        Z = rng.standard_normal((n, dim)) @ np.linalg.cholesky(C).T
        mh, ma = self.team_mean(H, A, True), self.team_mean(A, H, False)
        market = market or {}
        if market.get("margin") is not None or market.get("total") is not None:
            mm = W_MARKET_SCORE * market["margin"] + (1 - W_MARKET_SCORE) * (mh - ma) \
                if market.get("margin") is not None else mh - ma
            tt = W_MARKET_SCORE * market["total"] + (1 - W_MARKET_SCORE) * (mh + ma) \
                if market.get("total") is not None else mh + ma
            mh, ma = (tt + mm) / 2, (tt - mm) / 2
        out = {"home_pts": mh + TEAM_SD * Z[:, 0], "away_pts": ma + TEAM_SD * Z[:, 1], "proj": projs,
               "mean_margin": mh - ma, "mean_total": mh + ma}
        U = st.norm.cdf(Z[:, 2:])
        for i, key in enumerate(keys):
            p = projs[key]
            for si, s in enumerate(STATS):
                if need is not None and s not in need.get(key, ()):
                    continue
                out[(key, s)] = _discrete(U[:, i * ns + si], p.mean[s], p.var[s])
        return out


def _discrete(u: np.ndarray, mu: float, v: float) -> np.ndarray:
    """Inverse-CDF of Poisson / negative binomial via a lookup table (much faster than .ppf)."""
    top = int(mu + 12 * np.sqrt(v) + 10)
    k = np.arange(top + 1)
    if v <= mu * 1.02:
        cdf = st.poisson.cdf(k, mu)
    else:
        r = mu * mu / (v - mu)
        cdf = st.nbinom.cdf(k, r, r / (r + mu))
    return np.searchsorted(cdf, u).astype(float)


def market_scores(legs) -> dict:
    """Kalshi-implied mean home margin and mean total from the game's own ladders."""
    mpts, tpts = [], []
    for l in legs:
        p = l.mid
        if not 0.12 <= p <= 0.88:
            continue
        if l.kind in ("win", "spread"):
            x = 0.0 if l.kind == "win" else l.line
            if l.team == l.home:      # P(home margin > x) = p
                mpts.append(x + MARGIN_SD * st.norm.ppf(p))
            else:                      # P(home margin < -x) = p
                mpts.append(-x - MARGIN_SD * st.norm.ppf(p))
        elif l.kind == "total":
            tpts.append(l.line + TOTAL_SD * st.norm.ppf(p))
    return {"margin": float(np.median(mpts)) if mpts else None,
            "total": float(np.median(tpts)) if tpts else None}


def leg_values(leg, sim: dict) -> np.ndarray | None:
    """Continuous outcome per simulation; leg hits when value > leg.line."""
    if leg.kind == "prop":
        key = (leg.player, leg.team)
        if key not in sim["proj"]:
            return None
        return sum(sim[(key, s)] for s in leg.stats)
    h, a = sim["home_pts"], sim["away_pts"]
    if leg.kind == "total":
        return h + a
    margin = (h - a) if leg.team == leg.home else (a - h)
    if leg.kind == "win":
        return margin
    if leg.kind == "spread":
        return margin
    return None


def leg_line(leg) -> float:
    return 0.0 if leg.kind == "win" else leg.line
