"""Team play styles, player roles and "defense vs role" factors - all walk-forward (as of each game,
only earlier games are used), built from player box scores.

  team_styles    per team and game: pace, offensive/defensive rating, three-point rate, free-throw
                 rate, assist rate, offensive-rebound rate, turnover rate - for the offense and for
                 what the defense ALLOWS - over the team's previous 15 games
  roles          six player role types from k-means on per-36 production (a stand-in for positions)
  dvr            defense-vs-role: for each defense, role and stat, how much opponents of that role
                 produced relative to their own norm (per minute) over the defense's previous 20 games
  player_form    each player's rolling production and minutes before each game

Box scores cannot see schemes directly (drop, switch, zone). What they can see is what a scheme
gives up: threes allowed, free throws allowed, assists allowed, offensive rebounds allowed, turnovers
forced, pace. Those are the proxies used here.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config

BASE = ["pts", "reb", "ast", "fg3m", "stl", "blk"]
ROLE_NAMES = {}
CACHE = config.DATA_DIR / "styles_cache"


def team_games(logs: pd.DataFrame) -> pd.DataFrame:
    cols = ["pts", "reb", "ast", "fgm", "fga", "fg3m", "fg3a", "ftm", "fta", "oreb", "dreb", "stl", "blk", "tov"]
    t = logs.groupby(["game_id", "game_date", "season", "team", "opponent", "home"], as_index=False)[cols].sum()
    o = t[["game_id", "team"] + cols].rename(columns={c: "opp_" + c for c in cols}).rename(columns={"team": "opponent"})
    t = t.merge(o, on=["game_id", "opponent"])
    t["poss"] = t.fga + 0.44 * t.fta - t.oreb + t.tov
    t["opp_poss"] = t.opp_fga + 0.44 * t.opp_fta - t.opp_oreb + t.opp_tov
    return t.sort_values(["team", "game_date"]).reset_index(drop=True)


def team_styles(logs: pd.DataFrame, window: int = 15) -> pd.DataFrame:
    """One row per team-game with style metrics from the PREVIOUS `window` games (shifted: no lookahead)."""
    t = team_games(logs)
    g = t.groupby("team")
    def roll(col):
        return g[col].transform(lambda x: x.shift(1).rolling(window, min_periods=5).sum())
    R = {c: roll(c) for c in ["pts", "opp_pts", "poss", "opp_poss", "fga", "fg3a", "fta", "ast", "fgm", "oreb", "dreb", "tov",
                              "stl", "blk", "opp_fga", "opp_fg3a", "opp_fta", "opp_ast", "opp_fgm", "opp_oreb", "opp_dreb",
                              "opp_tov", "opp_reb", "reb", "opp_fg3m", "fg3m", "opp_stl", "opp_blk"]}
    n = g.pts.transform(lambda x: x.shift(1).rolling(window, min_periods=5).count())
    s = t[["game_id", "game_date", "season", "team", "opponent", "home"]].copy()
    s["pace"] = (R["poss"] + R["opp_poss"]) / (2 * n)
    s["off_rtg"] = 100 * R["pts"] / R["poss"]
    s["def_rtg"] = 100 * R["opp_pts"] / R["opp_poss"]
    s["net_rtg"] = s.off_rtg - s.def_rtg
    s["three_rate"] = R["fg3a"] / R["fga"]
    s["ft_rate"] = R["fta"] / R["fga"]
    s["ast_rate"] = R["ast"] / R["fgm"]
    s["oreb_rate"] = R["oreb"] / (R["oreb"] + R["opp_dreb"])
    s["tov_rate"] = R["tov"] / R["poss"]
    s["d_three_rate"] = R["opp_fg3a"] / R["opp_fga"]          # share of opponents' shots that are threes
    s["d_three_made"] = R["opp_fg3m"] / n
    s["d_ft_rate"] = R["opp_fta"] / R["opp_fga"]
    s["d_ast_rate"] = R["opp_ast"] / R["opp_fgm"]
    s["d_oreb_rate"] = R["opp_oreb"] / (R["opp_oreb"] + R["dreb"])
    s["d_tov_rate"] = R["opp_tov"] / R["opp_poss"]
    s["d_blk"] = 100 * R["blk"] / R["opp_poss"]
    s["d_stl"] = 100 * R["stl"] / R["opp_poss"]
    s["rest_days"] = g.game_date.diff().dt.days.fillna(5).clip(upper=5)
    return s


STYLE_COLS = ["pace", "off_rtg", "def_rtg", "three_rate", "ft_rate", "ast_rate", "oreb_rate", "tov_rate",
              "d_three_rate", "d_ft_rate", "d_ast_rate", "d_oreb_rate", "d_tov_rate", "d_blk", "d_stl"]


def roles(logs: pd.DataFrame, k: int = 6, seed: int = 3) -> pd.DataFrame:
    """Role per player-season from per-36 production. Fit on full seasons; a player keeps last season's role
    until he has 10 games in the new one."""
    from sklearn.cluster import KMeans
    g = logs.groupby(["season", "player_id", "name_key"], as_index=False).agg(
        gp=("minutes", "size"), minutes=("minutes", "sum"), pts=("pts", "sum"), reb=("reb", "sum"), ast=("ast", "sum"),
        fg3a=("fg3a", "sum"), fga=("fga", "sum"), fta=("fta", "sum"), stl=("stl", "sum"), blk=("blk", "sum"), oreb=("oreb", "sum"))
    g["mpg"] = g.minutes / g.gp
    feats = ["pts36", "reb36", "ast36", "three_share", "ft_rate", "blk36", "oreb36"]
    g["pts36"], g["reb36"], g["ast36"] = (36 * g[c] / g.minutes for c in ("pts", "reb", "ast"))
    g["blk36"], g["oreb36"] = 36 * g.blk / g.minutes, 36 * g.oreb / g.minutes
    g["three_share"] = g.fg3a / g.fga.clip(lower=1)
    g["ft_rate"] = g.fta / g.fga.clip(lower=1)
    fit = g[(g.gp >= 20) & (g.mpg >= 12)]
    mu, sd = fit[feats].mean(), fit[feats].std()
    km = KMeans(k, n_init=10, random_state=seed).fit(((fit[feats] - mu) / sd).values)
    ok = (g.gp >= 5) & (g.minutes > 60)
    g.loc[ok, "role_id"] = km.predict(((g.loc[ok, feats] - mu) / sd).fillna(0).values)
    cen = pd.DataFrame(km.cluster_centers_, columns=feats)
    names, left = {}, set(cen.index)
    for label, key in (("primary scorer", lambda c: c.pts36), ("rim big", lambda c: c.reb36 + c.blk36 - c.three_share),
                       ("versatile big / forward", lambda c: c.reb36 + 0.5 * c.blk36), ("playmaking guard", lambda c: c.ast36),
                       ("spot-up shooter", lambda c: c.three_share)):
        i = max(left, key=lambda j: key(cen.loc[j]))
        names[i] = label
        left.discard(i)
    for i in left:
        names[i] = "role player"
    g["role"] = g.role_id.map(names)
    ROLE_NAMES.update(names)
    return g[["season", "player_id", "name_key", "gp", "mpg", "role", "pts36", "reb36", "ast36", "three_share", "blk36"]]


def player_form(logs: pd.DataFrame) -> pd.DataFrame:
    """Per player-game: production and minutes over his previous games (shifted), plus per-minute norms."""
    d = logs.sort_values(["player_id", "game_date"]).copy()
    g = d.groupby("player_id")
    d["min_l10"] = g.minutes.transform(lambda x: x.shift(1).rolling(10, min_periods=3).mean())
    d["min_l3"] = g.minutes.transform(lambda x: x.shift(1).rolling(3, min_periods=2).mean())
    d["games_prior"] = g.cumcount()
    d["dd"] = ((d[["pts", "reb", "ast", "stl", "blk"]] >= 10).sum(axis=1) >= 2).astype(float)
    d["td"] = ((d[["pts", "reb", "ast", "stl", "blk"]] >= 10).sum(axis=1) >= 3).astype(float)
    g = d.groupby("player_id")
    d["dd_l20"] = g.dd.transform(lambda x: x.shift(1).rolling(20, min_periods=5).mean())
    d["td_l20"] = g.td.transform(lambda x: x.shift(1).rolling(20, min_periods=5).mean())
    d["days_rest"] = g.game_date.diff().dt.days
    msum = g.minutes.transform(lambda x: x.shift(1).rolling(30, min_periods=5).sum())
    for s in BASE:
        d[s + "_l10"] = g[s].transform(lambda x: x.shift(1).rolling(10, min_periods=3).mean())
        d[s + "_sd"] = g[s].transform(lambda x: x.shift(1).rolling(20, min_periods=5).std())
        d[s + "_rate"] = g[s].transform(lambda x: x.shift(1).rolling(30, min_periods=5).sum()) / msum   # per minute
        d[s + "_exp"] = d[s + "_rate"] * d.minutes                                                      # norm for the minutes played
    return d


def dvr(form: pd.DataFrame, role_tab: pd.DataFrame, window: int = 20, prior: float = 400.0) -> pd.DataFrame:
    """Defense-vs-role. Per defense, role, game: opponents of that role produced `ratio` x their own
    per-minute norm over the defense's previous `window` games (shrunk toward 1.0)."""
    seasons = sorted(form.season.unique())
    prev = {s: seasons[i - 1] if i else None for i, s in enumerate(seasons)}
    r = role_tab.set_index(["season", "player_id"]).role
    cur = pd.Series(list(zip(form.season, form.player_id)), index=form.index).map(r)
    last = pd.Series(list(zip(form.season.map(prev), form.player_id)), index=form.index).map(r)
    f = form.assign(role=np.where(form.games_prior % 1000 >= 0, cur.fillna(last), last)).dropna(subset=["role"])
    f = f[f.pts_exp.notna()]
    agg = {s: "sum" for s in BASE} | {s + "_exp": "sum" for s in BASE} | {"minutes": "sum"}
    t = f.groupby(["opponent", "role", "game_id", "game_date"], as_index=False).agg(agg).rename(columns={"opponent": "defense"})
    t = t.sort_values(["defense", "role", "game_date"])
    g = t.groupby(["defense", "role"])
    out = t[["defense", "role", "game_id", "game_date"]].copy()
    for s in BASE:
        a = g[s].transform(lambda x: x.shift(1).rolling(window, min_periods=4).sum())
        e = g[s + "_exp"].transform(lambda x: x.shift(1).rolling(window, min_periods=4).sum())
        scale = {"pts": 1.0, "reb": 0.4, "ast": 0.25, "fg3m": 0.1, "stl": 0.05, "blk": 0.04}[s]
        out["dvr_" + s] = (a + prior * scale) / (e + prior * scale)
    return out, f[["game_id", "player_id", "role"]]


def build(logs: pd.DataFrame) -> dict:
    form = player_form(logs)
    role_tab = roles(logs)
    d, pr = dvr(form, role_tab)
    return dict(styles=team_styles(logs), roles=role_tab, form=form, dvr=d, player_role=pr)


TAGS = [("pace", 1, "plays fast", "plays slow"), ("three_rate", 1, "three-point heavy offense", "attacks inside the arc"),
        ("ft_rate", 1, "gets to the line", None), ("ast_rate", 1, "ball-movement offense", "isolation-heavy offense"),
        ("oreb_rate", 1, "crashes the offensive glass", None),
        ("d_three_rate", 1, "defense gives up threes", "defense runs shooters off the line"),
        ("d_ft_rate", 1, "defense fouls a lot", "defense avoids fouls"),
        ("d_ast_rate", 1, "defense allows assisted looks", "switch-style defense (few assists allowed)"),
        ("d_oreb_rate", 1, "weak on the defensive glass", "cleans the defensive glass"),
        ("d_tov_rate", 1, "pressure defense (forces turnovers)", None), ("d_blk", 1, "rim protection (blocks)", None),
        ("def_rtg", -1, "top defense", "weak defense"), ("off_rtg", 1, "top offense", "weak offense")]


def latest_profiles(styles: pd.DataFrame, season: str) -> pd.DataFrame:
    s = styles[styles.season == season].dropna(subset=["pace"]).sort_values("game_date").groupby("team").tail(1).set_index("team")
    z = (s[STYLE_COLS] - s[STYLE_COLS].mean()) / s[STYLE_COLS].std()
    tags = []
    for team in s.index:
        tg = []
        for col, sign, hi, lo in TAGS:
            v = z.at[team, col] * sign
            if v >= 1.0 and hi:
                tg.append(hi)
            elif v <= -1.0 and lo:
                tg.append(lo)
        tags.append("; ".join(tg))
    s = s.assign(tags=tags)
    for c in STYLE_COLS:
        s[c + "_rank"] = s[c].rank(ascending=False).astype(int)
    return s
