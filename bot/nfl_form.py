"""NFL legs: does anything knowable before kickoff separate the unders that hit from the ones that miss?

Written after the 2026-11-23 Monday-night test cards (5 misses in 14 legs). Each miss was looked up: three of
the five were unders taken AGAINST the player's recent games (Ekeler had 2+ catches in 3 of his last 6 and 4 the
week before; Gesicki 3+ catches in half his games; Croskey-Merritt had averaged 48 rushing yards before missing
five weeks). The card builder never looked at a single NFL game log. This module does, and tests on two seasons
of settled Kalshi markets whether it would have helped.

Data: nflverse weekly player stats and schedule lines (CC-BY 4.0), downloaded at run time into data/ext/nfl.
Walk-forward: a market only sees the player's earlier games.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config
from .data import norm_name
from .nfl_scenarios import _get

SERIES = {"KXNFLREC": ("receptions",), "KXNFLRECYDS": ("receiving_yards",), "KXNFLRSHYDS": ("rushing_yards",),
          "KXNFLPASSYDS": ("passing_yards",), "KXNFLPASSTDS": ("passing_tds",),
          "KXNFLTD": ("rushing_tds", "receiving_tds"), "KXNFLANYTD": ("rushing_tds", "receiving_tds")}
WORD = {"KXNFLREC": "receptions", "KXNFLRECYDS": "receiving yards", "KXNFLRSHYDS": "rushing yards", "KXNFLPASSYDS": "passing yards",
        "KXNFLPASSTDS": "passing touchdowns", "KXNFLTD": "touchdowns", "KXNFLANYTD": "touchdowns"}
LAST, SMOOTH = 6, 2.0
CUT = pd.Timestamp("2026-07-01")


def logs(seasons=(2025, 2026), fresh: bool = False) -> pd.DataFrame:
    """One row per player and game with the stats the prop markets are written on."""
    parts = []
    for y in seasons:
        f = config.DATA_DIR / "ext" / "nfl" / f"stats_player_week_{y}.parquet"
        if fresh and y == max(seasons) and f.exists():
            f.unlink()
        w = _get("stats_player", f"stats_player_week_{y}")
        parts.append(w[w.season_type == "REG"])
    w = pd.concat(parts, ignore_index=True)
    g = _get("schedules", "games")[["game_id", "gameday", "away_team", "home_team", "spread_line", "total_line"]]
    w = w.merge(g, on="game_id", how="left")
    w["game_date"] = pd.to_datetime(w.gameday)
    w["name_key"] = w.player_display_name.fillna("").map(norm_name)
    home = w.team == w.home_team
    # nflverse spread_line is the home team's margin the market expects (positive = home favoured)
    w["team_spread"] = np.where(home, w.spread_line, -w.spread_line)
    w["team_total"] = w.total_line / 2 + w.team_spread / 2
    cols = ["name_key", "player_display_name", "position", "team", "opponent_team", "game_id", "game_date", "week", "season", "team_spread", "team_total",
            "targets", "carries", "attempts"] + sorted({c for v in SERIES.values() for c in v})
    return w[cols].fillna({c: 0 for c in cols if c not in ("team_spread", "team_total")}).sort_values(["name_key", "game_date"]).reset_index(drop=True)


def markets() -> pd.DataFrame:
    from .favorites import slate_date
    from .trades import PRIOR_RESULTS, RESULTS
    parts = []
    for pre, res in (("prop_pre_prior.parquet", PRIOR_RESULTS), ("prop_pre.parquet", RESULTS)):
        p = pd.read_parquet(config.DATA_DIR / pre)
        p = p[p.series.isin(list(SERIES))]
        w = p.pivot_table(index=["ticker", "series", "game", "result"], columns="taker_side", values="cost").reset_index()
        for c in ("yes", "no"):
            if c not in w:
                w[c] = np.nan
        w["p_mkt"] = np.where(w.yes.notna() & w.no.notna(), (w.yes + 1 - w.no) / 2, w.yes.fillna(1 - w.no))
        r = pd.read_parquet(res)[["ticker", "title", "floor_strike"]]
        parts.append(w.merge(r, on="ticker"))
    m = pd.concat(parts, ignore_index=True)
    m["name_key"] = m.title.str.replace(r"(:| records ).*$", "", regex=True).map(norm_name)
    m["game_date"] = pd.to_datetime(m.game.map(slate_date))
    m["y"] = (m.result == "yes").astype(float)
    m["line"] = m.floor_strike.fillna(0.5) + 0.5            # "N+" needs N
    m["season"] = np.where(m.game_date < CUT, "last season", "this season")
    return m.dropna(subset=["game_date", "p_mkt"])


def form_for(L: pd.DataFrame, m: pd.DataFrame) -> pd.DataFrame:
    """Adds, per market, what the player's previous LAST games say: n games, how many reached the line, average,
    weeks since he last played, and the smoothed form probability."""
    out = []
    for series, cols in SERIES.items():
        mm = m[m.series == series]
        if mm.empty:
            continue
        v = L[["name_key", "game_date", "position", "team"]].copy()
        v["val"] = L[list(cols)].sum(axis=1)
        v["usage"] = L.targets if "rec" in cols[0] else (L.carries if "rush" in cols[0] else L.attempts)
        j = mm[["ticker", "name_key", "game_date", "line"]].merge(v, on="name_key", suffixes=("", "_g"))
        j = j[j.game_date_g < j.game_date]
        j["rk"] = j.groupby("ticker").game_date_g.rank(ascending=False, method="first")
        j = j[j.rk <= LAST]
        a = j.groupby("ticker").agg(n=("val", "size"), k=("val", lambda s: 0), avg=("val", "mean"), last_played=("game_date_g", "max"),
                                    position=("position", "last"), team=("team", "last"), usage=("usage", "mean"))
        a["k"] = (j.val >= j.line).groupby(j.ticker).sum()
        a["last_val"] = j[j.rk == 1].set_index("ticker").val
        out.append(a)
    f = pd.concat(out)
    d = m.merge(f, left_on="ticker", right_index=True, how="inner")
    d["p_form"] = (d.k + SMOOTH * d.p_mkt) / (d.n + SMOOTH)
    lg = lambda p: np.log(np.clip(p, 0.02, 0.98) / (1 - np.clip(p, 0.02, 0.98)))
    d["form_gap"] = lg(d.p_form) - lg(d.p_mkt)                 # + = his recent games say the OVER is likelier than the price does
    d["weeks_out"] = ((d.game_date - d.last_played).dt.days / 7).round()
    return d


def dataset() -> pd.DataFrame:
    L = logs()
    d = form_for(L, markets())
    d = d[(d.n >= 3) & (d.p_mkt > 0.08) & (d.p_mkt < 0.92)].copy()
    sp = L.groupby(["team", "game_date"])[["team_spread", "team_total"]].first().reset_index()
    return d.merge(sp, on=["team", "game_date"], how="left")


def _gap(g: pd.DataFrame) -> pd.Series:
    """How the UNDER did against its price in this group (errors clustered by game)."""
    x = g.p_mkt - g.y                                          # under hit - under price
    cl = x.groupby(g.game).sum()
    G = len(cl)
    se = np.sqrt(((cl - cl.mean()) ** 2).sum() * G / max(G - 1, 1)) / len(x) if G > 1 else np.nan
    return pd.Series(dict(markets=len(g), under_hit=1 - g.y.mean(), under_price=1 - g.p_mkt.mean(), gap=x.mean(), z=x.mean() / se if se and se > 0 else np.nan))


def study(verbose: bool = True) -> dict:
    import statsmodels.api as sm
    pd.set_option("display.width", 230)
    d = dataset()
    lg = lambda p: np.log(p / (1 - p))
    d["form3"] = pd.cut(d.form_gap, [-9, -0.4, 0.4, 9], labels=["recent games say UNDER", "neutral", "recent games say OVER"])
    d["back"] = np.where(d.weeks_out >= 3, "first game back after 3+ weeks out", "played recently")
    d["fav"] = pd.cut(d.team_spread, [-99, -3, 3, 99], labels=["underdog by 3+", "close game", "favourite by 3+"])
    out = dict(data=d)
    fits = {}
    for season, g in d.groupby("season"):
        X = pd.DataFrame({"price": lg(g.p_mkt), "form": (g.form_gap - g.form_gap.mean()) / g.form_gap.std()})
        fits[season] = sm.Logit(g.y.values, sm.add_constant(X)).fit(disp=0, cov_type="cluster", cov_kwds={"groups": pd.factorize(g.game)[0]})
    out["fits"] = {s: (float(f.params["price"]), float(f.params["form"]), float(f.tvalues["form"])) for s, f in fits.items()}
    out["form"] = d.groupby(["form3", "season"], observed=True).apply(_gap)
    out["back_tab"] = d.groupby(["back", "season"], observed=True).apply(_gap)
    out["pos"] = d[d.position.isin(["RB", "WR", "TE", "QB"])].assign(what=lambda x: x.series.map(WORD)).groupby(["what", "position", "season"], observed=True).apply(_gap)
    out["script"] = d.assign(what=lambda x: x.series.map(WORD)).groupby(["what", "fav", "season"], observed=True).apply(_gap)
    if verbose:
        print(f"\nNFL LEGS: what was knowable before kickoff. {len(d):,} settled Kalshi NFL prop markets with a pre-game price and 3+ earlier games "
              f"for the player ({(d.season == 'last season').sum():,} last season, {(d.season == 'this season').sum():,} this season).")
        print("gap = how often the UNDER hit minus what it cost (before the fee, which is about 1.6 points at these prices). z over 2 in BOTH seasons = real.\n")
        for s, (pr, fo, z) in out["fits"].items():
            print(f"  {s}: after Kalshi's price (weight {pr:.2f}), one standard deviation of recent form moves the odds of the over by {fo:+.3f} (z {z:+.1f})")
        for name, key in (("RECENT FORM against the line (share of his last 6 games that reached it, vs the price)", "form"),
                          ("COMING BACK from missed games", "back_tab")):
            print(f"\n  {name}")
            print(out[key].round(3).to_string())
        t = out["pos"]
        t = t[t.markets >= 250]
        print("\n  BY PROP AND POSITION (250+ markets)")
        print(t.round(3).to_string())
        t = out["script"]
        t = t[t.markets >= 250]
        print("\n  BY EXPECTED GAME SCRIPT (the point spread for the player's team)")
        print(t.round(3).to_string())
    return out
