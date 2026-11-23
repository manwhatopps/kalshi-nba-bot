"""Does the matchup "logical chain" add anything to Kalshi's price?  And the chain itself, per leg.

For every settled NBA prop market with a pre-game price (both seasons) we attach what was knowable
that morning: the player's recent form against the line, minutes trend, rest, his role, what the
opposing defense had been giving up to that role in that stat (defense-vs-role), the opponent's
defensive rating and style, and the game's expected pace. Then:

  * `test()`   regress the result on the market price plus those factors, with errors clustered by
               game, separately in each season. A factor that matters shows the same sign and a
               meaningful z-score in BOTH seasons. Also scored out-of-sample: fit on last season,
               predict this season, compare with the price alone.
  * `explain()` the chain for one leg in words, used to annotate cards.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats as st

from . import config
from .data import KALSHI_TO_LOG_TEAM, norm_name
from .favorites import slate_date
from . import styles as S

STATS = {"KXNBAPTS": ("pts",), "KXNBAREB": ("reb",), "KXNBAAST": ("ast",), "KXNBA3PT": ("fg3m",), "KXNBASTL": ("stl",),
         "KXNBABLK": ("blk",), "KXNBAPRA": ("pts", "reb", "ast"), "KXNBASTOCK": ("stl", "blk")}
FEATS = ["form_gap", "dvr", "opp_def", "pace", "min_trend", "b2b", "opp_b2b", "home", "blowout"]
DATA = config.DATA_DIR / "matchup_dataset.parquet"


def _logit(p):
    p = np.clip(p, 0.01, 0.99)
    return np.log(p / (1 - p))


def markets() -> pd.DataFrame:
    """One row per settled NBA prop market: implied OVER probability before the game, the line, the result."""
    from .trades import PRIOR_RESULTS, RESULTS
    parts = []
    for season, pre, res in (("last season", "prop_pre_prior.parquet", PRIOR_RESULTS), ("this season", "prop_pre.parquet", RESULTS)):
        p = pd.read_parquet(config.DATA_DIR / pre)
        p = p[p.series.isin(list(STATS) + ["KXNBA2D", "KXNBA3D"])]
        w = p.pivot_table(index=["ticker", "series", "game", "result"], columns="taker_side", values="cost").reset_index()
        for c in ("yes", "no"):
            if c not in w:
                w[c] = np.nan
        w["p_mkt"] = np.where(w.yes.notna() & w.no.notna(), (w.yes + 1 - w.no) / 2, w.yes.fillna(1 - w.no))
        r = pd.read_parquet(res)[["ticker", "title", "floor_strike"]]
        parts.append(w.merge(r, on="ticker").assign(season_label=season))
    m = pd.concat(parts, ignore_index=True)
    m["name_key"] = m.title.str.split(":").str[0].map(norm_name)
    seg = m.ticker.str.split("-").str[-2].where(~m.series.isin(["KXNBA2D", "KXNBA3D"]), m.ticker.str.split("-").str[-1])
    m["kteam"] = seg.str[:3]
    m["away"], m["home_team"] = m.game.str[-6:-3], m.game.str[-3:]
    m = m[(m.kteam == m.away) | (m.kteam == m.home_team)]
    m["team"] = m.kteam.replace(KALSHI_TO_LOG_TEAM)
    m["opp"] = np.where(m.kteam == m.away, m.home_team, m.away)
    m["opp"] = m.opp.replace(KALSHI_TO_LOG_TEAM)
    m["game_date"] = m.game.map(slate_date)
    m["y"] = (m.result == "yes").astype(float)
    return m.dropna(subset=["game_date", "p_mkt"])


def attach(m: pd.DataFrame, B: dict) -> pd.DataFrame:
    """Join each market to what was knowable that morning (all tables are shifted: earlier games only)."""
    form, sty = B["form"], B["styles"]
    keep = ["name_key", "team", "game_date", "game_id", "player_id", "home", "min_l10", "min_l3", "days_rest", "games_prior",
            "dd_l20", "td_l20"] + [s + sfx for s in S.BASE for sfx in ("_l10", "_sd")]
    d = m.merge(form[keep], on=["name_key", "team", "game_date"], how="inner")
    d = d.merge(B["player_role"], on=["game_id", "player_id"], how="left")
    d = d.merge(B["dvr"].rename(columns={"defense": "opp"}).drop(columns="game_date"), on=["opp", "role", "game_id"], how="left")
    own = sty[["team", "game_id", "pace", "net_rtg", "rest_days"]].rename(columns={"pace": "own_pace", "net_rtg": "own_net", "rest_days": "own_rest"})
    opp = sty[["team", "game_id", "pace", "def_rtg", "net_rtg", "rest_days"] + [c for c in S.STYLE_COLS if c.startswith("d_")]].rename(
        columns={"team": "opp", "pace": "opp_pace", "def_rtg": "opp_def_rtg", "net_rtg": "opp_net", "rest_days": "opp_rest"})
    d = d.merge(own, on=["team", "game_id"], how="left").merge(opp, on=["opp", "game_id"], how="left")
    lg = sty.groupby("game_date")[["pace", "def_rtg"]].mean().rolling(15, min_periods=1).mean().rename(columns={"pace": "lg_pace", "def_rtg": "lg_def"})
    d = d.merge(lg, left_on="game_date", right_index=True, how="left")
    # form vs the line
    mu, var, dv = np.zeros(len(d)), np.zeros(len(d)), np.zeros(len(d))
    wsum = np.zeros(len(d))
    for series, stats in STATS.items():
        mask = (d.series == series).values
        for s_ in stats:
            mu[mask] += d.loc[mask, s_ + "_l10"].values
            var[mask] += (d.loc[mask, s_ + "_sd"].values ** 2)
            w = d.loc[mask, s_ + "_l10"].clip(lower=0.1).values
            dv[mask] += d.loc[mask, "dvr_" + s_].fillna(1.0).values * w
            wsum[mask] += w
    d["form_mean"], d["form_sd"] = mu, np.sqrt(var * 1.15).clip(min=0.5)
    d["p_form"] = 1 - st.norm.cdf((d.floor_strike + 0.5 - d.form_mean) / d.form_sd)
    for series, col in (("KXNBA2D", "dd_l20"), ("KXNBA3D", "td_l20")):
        mask = d.series == series
        d.loc[mask, "p_form"] = (d.loc[mask, col] * 20 + d.loc[mask, "p_mkt"] * 5) / 25       # shrunk rate
        dv[mask.values] = (d.loc[mask, ["dvr_pts", "dvr_reb", "dvr_ast"]].fillna(1.0).mean(axis=1)).values
        wsum[mask.values] = 1.0
    d["dvr"] = np.where(wsum > 0, dv / np.where(wsum > 0, wsum, 1), 1.0)
    d["form_gap"] = _logit(d.p_form) - _logit(d.p_mkt)            # form says more (+) or less (-) likely than the price
    d["opp_def"] = d.opp_def_rtg - d.lg_def                       # + = opponent defends worse than average
    d["pace"] = (d.own_pace + d.opp_pace) / 2 - d.lg_pace
    d["min_trend"] = (d.min_l3 / d.min_l10 - 1).clip(-0.5, 0.5)
    d["b2b"] = (d.own_rest <= 1).astype(float)
    d["opp_b2b"] = (d.opp_rest <= 1).astype(float)
    d["home"] = d.home.astype(float)
    d["blowout"] = (d.own_net - d.opp_net).abs()
    return d.dropna(subset=FEATS + ["p_mkt", "y"])


def dataset(rebuild: bool = False) -> pd.DataFrame:
    if DATA.exists() and not rebuild:
        return pd.read_parquet(DATA)
    from .data import load_logs
    B = S.build(load_logs(topup=False))
    d = attach(markets(), B)
    d.to_parquet(DATA, index=False)
    return d


def _fit(d: pd.DataFrame, cols: list[str]):
    import statsmodels.api as sm
    X = d[cols].copy()
    X = (X - X.mean()) / X.std().replace(0, 1)
    X.insert(0, "price", _logit(d.p_mkt.values))
    X = sm.add_constant(X)
    return sm.Logit(d.y.values, X).fit(disp=0, cov_type="cluster", cov_kwds={"groups": pd.factorize(d.game)[0]})


def test(rebuild: bool = False):
    pd.set_option("display.width", 230)
    d = dataset(rebuild)
    d = d[(d.p_mkt > 0.08) & (d.p_mkt < 0.92)]
    print(f"\nMATCHUP FACTORS vs KALSHI'S PRICE: {len(d):,} settled NBA prop markets with a pre-game price "
          f"({(d.season_label == 'last season').sum():,} last season, {(d.season_label == 'this season').sum():,} this season).")
    print("Each number is how much one standard deviation of the factor moves the odds that the OVER hits, AFTER the price is "
          "accounted for,\nwith its z-score (|z| > 2 is distinguishable from zero). A real factor has the same sign and |z| > 2 in both seasons.\n")
    rows, fits = [], {}
    for season, g in d.groupby("season_label"):
        fits[season] = _fit(g, FEATS)
    names = {"form_gap": "recent form vs the line (last 10 games)", "dvr": "defense-vs-role: what this defense gives this kind of player",
             "opp_def": "opponent's overall defensive rating", "pace": "expected pace of the game", "min_trend": "minutes trending up (last 3 vs last 10)",
             "b2b": "player's team on a back-to-back", "opp_b2b": "opponent on a back-to-back", "home": "home game",
             "blowout": "mismatch size (blowout risk)", "price": "Kalshi's price itself (1.00 = fully trusted)"}
    for f in ["price"] + FEATS:
        row = {"factor": names[f]}
        for season in ("last season", "this season"):
            r = fits[season]
            row[season] = f"{r.params[f]:+.3f} (z {r.tvalues[f]:+.1f})"
        a, b = fits["last season"], fits["this season"]
        real = f != "price" and np.sign(a.params[f]) == np.sign(b.params[f]) and abs(a.tvalues[f]) > 2 and abs(b.tvalues[f]) > 2
        row["verdict"] = "REAL in both seasons" if real else ("" if f == "price" else "not established")
        rows.append(row)
    print(pd.DataFrame(rows).to_string(index=False))

    # out of sample: fit on last season, score this season
    import statsmodels.api as sm
    a, b = d[d.season_label == "last season"], d[d.season_label == "this season"]

    def prep(x, ref, cols):
        X = (x[cols] - ref[cols].mean()) / ref[cols].std().replace(0, 1)
        X.insert(0, "price", _logit(x.p_mkt.values))
        return sm.add_constant(X, has_constant="add")

    def ll(p, y):
        p = np.clip(p, 1e-4, 1 - 1e-4)
        return float(-(y * np.log(p) + (1 - y) * np.log(1 - p)).mean())
    full = sm.Logit(a.y.values, prep(a, a, FEATS)).fit(disp=0)
    only = sm.Logit(a.y.values, prep(a, a, [])).fit(disp=0)
    print(f"\nOUT OF SAMPLE (fit on last season, scored on this season's {len(b):,} markets; lower log-loss = better):")
    print(f"   Kalshi price as is            : {ll(b.p_mkt.values, b.y.values):.5f}")
    print(f"   price, recalibrated           : {ll(only.predict(prep(b, a, [])), b.y.values):.5f}")
    print(f"   price + all matchup factors   : {ll(full.predict(prep(b, a, FEATS)), b.y.values):.5f}")
    return d, fits


# ---------------------------------------------------------------- the chain for tonight's legs
STAT_WORD = {"pts": "points", "reb": "rebounds", "ast": "assists", "fg3m": "threes", "stl": "steals", "blk": "blocks"}


def context(logs: pd.DataFrame | None = None) -> dict:
    """Everything needed to explain tonight's legs, as of the latest game on file."""
    from .data import load_logs
    logs = logs if logs is not None else load_logs(topup=False)
    B = S.build(logs)
    cur = config.CURRENT_SEASON
    recent = logs[logs.game_date >= logs.game_date.max() - pd.Timedelta(days=400)].sort_values("game_date")
    last = {k: g.tail(20) for k, g in recent.groupby(["name_key", "team"])}
    roles = B["roles"].sort_values(["season", "gp"])
    r_now = roles[(roles.season == cur) & (roles.gp >= 5)].set_index("name_key").role.to_dict()
    r_old = roles[roles.season != cur].groupby("name_key").role.last().to_dict()
    prof = S.latest_profiles(B["styles"], cur)
    dv = B["dvr"].sort_values("game_date").groupby(["defense", "role"]).tail(1).set_index(["defense", "role"])
    last_game = logs.groupby("team").game_date.max().to_dict()
    return dict(last=last, role=lambda k: r_now.get(k) or r_old.get(k), prof=prof, dvr=dv, last_game=last_game)


def explain(ctx: dict, title: str, series: str, ticker: str, game: str, side: str, price: float) -> dict:
    """The logical chain for one NBA leg. Returns {'lines': [...], 'for': n, 'against': n}; a point is 'for'
    when it supports the side being bought."""
    name = norm_name(title.split(":")[0])
    seg = ticker.split("-")[-1 if series in ("KXNBA2D", "KXNBA3D") else -2]
    kteam = seg[:3]
    away, home = game[-6:-3], game[-3:]
    if kteam not in (away, home):
        return dict(lines=["no context (team not matched)"], pro=0, con=0)
    team, opp = (KALSHI_TO_LOG_TEAM.get(t, t) for t in (kteam, home if kteam == away else away))
    g = ctx["last"].get((name, team))
    if g is None or len(g) < 5:
        return dict(lines=["fewer than 5 recent games on file: no form to judge"], pro=0, con=0)
    under = side == "no"
    pro = con = 0
    lines = []

    def vote(supports_under: bool | None):
        nonlocal pro, con
        if supports_under is None:
            return
        if supports_under == under:
            pro += 1
        else:
            con += 1

    l10 = g.tail(10)
    if series in ("KXNBA2D", "KXNBA3D"):
        need = 2 if series == "KXNBA2D" else 3
        flags = ((g[["pts", "reb", "ast", "stl", "blk"]] >= 10).sum(axis=1) >= need)
        rate = flags.mean()
        word = "double-double" if need == 2 else "triple-double"
        lines.append(f"had a {word} in {int(flags.sum())} of his last {len(g)} games ({rate:.0%}); the price implies {1 - price if under else price:.0%}")
        vote(rate < (1 - price if under else price) - 0.05 if abs(rate - (1 - price if under else price)) > 0.05 else None)
        stats = ("reb", "ast")
    else:
        stats = STATS.get(series)
        if not stats:
            return dict(lines=["no context for this market type"], pro=0, con=0)
        try:
            k = float(title.split(":")[1].strip().split("+")[0])
        except (IndexError, ValueError):
            return dict(lines=["could not read the line"], pro=0, con=0)
        vals = l10[list(stats)].sum(axis=1)
        cleared = int((vals >= k).sum())
        what = " + ".join(STAT_WORD[s] for s in stats)
        lines.append(f"averaging {vals.mean():.1f} {what} over his last {len(l10)}; reached {k:g}+ in {cleared} of them "
                     f"({cleared / len(l10):.0%}); the price implies {1 - price if under else price:.0%}")
        implied_over = 1 - price if under else price
        diff = cleared / len(l10) - implied_over
        vote(diff < 0 if abs(diff) > 0.10 else None)
    m3, m10 = g.minutes.tail(3).mean(), l10.minutes.mean()
    if abs(m3 / m10 - 1) > 0.12:
        lines.append(f"minutes {'down' if m3 < m10 else 'up'}: {m3:.0f} a game over the last 3 vs {m10:.0f} over the last 10")
        vote(m3 < m10)
    role = ctx["role"](name)
    if role and (opp, role) in ctx["dvr"].index:
        row = ctx["dvr"].loc[(opp, role)]
        ratio = float(np.mean([row["dvr_" + s] for s in stats]))
        lab = "stingy" if ratio < 0.97 else ("generous" if ratio > 1.03 else "average")
        lines.append(f"role: {role}. {opp} has allowed that role {ratio:.0%} of its usual {' / '.join(STAT_WORD[s] for s in stats)} ({lab})")
        vote(None if lab == "average" else lab == "stingy")
    if opp in ctx["prof"].index and team in ctx["prof"].index:
        p, o = ctx["prof"].loc[opp], ctx["prof"].loc[team]
        pace = (p.pace + o.pace) / 2 - ctx["prof"].pace.mean()
        bits = [f"{opp} defense ranks {int(31 - p.def_rtg_rank)} of 30"]
        if abs(pace) >= 1.0:
            bits.append(f"game pace {'fast' if pace > 0 else 'slow'} ({pace:+.1f} possessions vs average)")
            vote(pace < 0)
        if p.tags:
            bits.append(f"{opp}: {p.tags}")
        lines.append("; ".join(bits))
        vote(True if p.def_rtg_rank >= 24 else (False if p.def_rtg_rank <= 7 else None))   # rank 30 = best defense
    lg = ctx["last_game"].get(team)
    if lg is not None and (slate_date(game) - lg).days == 1:
        lines.append("second night of a back-to-back")
        vote(True)
    return dict(lines=lines, pro=pro, con=con)
